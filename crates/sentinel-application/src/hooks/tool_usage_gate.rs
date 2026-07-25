//! Tool Usage Gate
//!
//! `PreToolUse` hook that blocks mutating tools if required preconditions
//! aren't met:
//! 1. Sequential thinking must be present in the live session transcript.
//! 2. The active Claude native TaskList for this session must contain at
//!    least one task.
//! 3. The live transcript must show an approved plan (`ExitPlanMode`).
//! 4. The active session TaskList must contain an `in_progress` task.
//!
//! Production authority comes from the live transcript and the active
//! session's native TaskList under `~/.claude/tasks/{session_id}/`.
//! No alternate authority source is accepted for this gate.

use sentinel_domain::events::{HookInput, HookOutput};
use sentinel_domain::ports::ReversibilityClassifierPort;
use sentinel_domain::ReversibilityClass;
use std::path::Path;

use super::{EnvPort, FileSystemPort};

// ── Config (shipped defaults + operator override) ────────────────────────
//
// Mirrors the `memory_provision` pattern: shipped TOML compiled in via
// `include_str!`, wholesale-replaced by the operator override at
// `~/.claude/sentinel/config/tool-usage-gate.toml` when present.
const SHIPPED_DEFAULTS: &str = include_str!("../../../../config/tool-usage-gate-defaults.toml");

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolUsageGateConfig {
    /// Check 1: require `mcp__sequential-thinking__sequentialthinking` in the
    /// live transcript before mutating tools. `false` skips ONLY this check;
    /// the task-list / plan-mode / in_progress-task checks are unaffected.
    #[serde(default = "default_true")]
    sequential_thinking_check: bool,
    /// Master switch. `false` disables the whole gate: every tool call is
    /// allowed and no graph authority is consulted. For controlled
    /// environments (benchmark arms); missing/corrupt config falls back to
    /// `true` — fail closed to enforcement.
    #[serde(default = "default_true")]
    enabled: bool,
}

const fn default_true() -> bool {
    true
}

impl Default for ToolUsageGateConfig {
    fn default() -> Self {
        Self {
            sequential_thinking_check: true,
            enabled: true,
        }
    }
}

impl ToolUsageGateConfig {
    /// Parse failures fall back to the enforced defaults (fail closed to
    /// enforcement — a corrupt override must not silently relax the gate).
    fn from_toml_or_default(s: &str) -> Self {
        toml::from_str(s).unwrap_or_else(|e| {
            eprintln!("[sentinel] tool_usage_gate: config TOML parse failed ({e}); using enforced defaults");
            Self::default()
        })
    }
}

/// Load shipped defaults, then (if present) replace wholesale with the
/// operator override file — same semantics as `memory_provision::load_config`.
///
/// The override path is rooted at `FileSystemPort::claude_dir()` (not
/// `home_dir().join(".claude")`) so it honors adapters that override Claude's
/// state directory, notably `SENTINEL_CLAUDE_DIR` for isolated sandbox profiles.
fn load_config(fs: &dyn FileSystemPort) -> ToolUsageGateConfig {
    let mut cfg = ToolUsageGateConfig::from_toml_or_default(SHIPPED_DEFAULTS);
    let path = fs
        .claude_dir()
        .join("sentinel")
        .join("config")
        .join("tool-usage-gate.toml");
    if let Ok(content) = fs.read_to_string(&path) {
        cfg = ToolUsageGateConfig::from_toml_or_default(&content);
    }
    cfg
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlanState {
    Missing,
    InPlanMode,
    Approved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptSignals {
    sequential_thinking_used: bool,
    plan_state: PlanState,
}

/// Parse the live transcript as the authority for session preconditions.
///
/// Malformed non-empty lines are treated as authority failure. The gate must
/// not accept an untrusted transcript by skipping over corrupted records.
fn read_transcript_signals(
    fs: &dyn FileSystemPort,
    transcript_path: &Path,
) -> Result<TranscriptSignals, String> {
    let content = fs.read_to_string(transcript_path).map_err(|err| {
        format!(
            "failed to read transcript {}: {err}",
            transcript_path.display()
        )
    })?;
    let mut signals = TranscriptSignals {
        sequential_thinking_used: false,
        plan_state: PlanState::Missing,
    };

    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: serde_json::Value = serde_json::from_str(line).map_err(|err| {
            format!(
                "malformed transcript JSON at {} line {}: {err}",
                transcript_path.display(),
                line_idx + 1
            )
        })?;
        if entry.get("type").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        let Some(blocks) = entry
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
        else {
            continue;
        };

        for block in blocks {
            if block.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
                continue;
            }
            match block.get("name").and_then(|v| v.as_str()) {
                Some(name) if name.contains("sequentialthinking") => {
                    signals.sequential_thinking_used = true;
                }
                Some("EnterPlanMode") => {
                    signals.plan_state = PlanState::InPlanMode;
                }
                Some("ExitPlanMode") => {
                    signals.plan_state = PlanState::Approved;
                }
                _ => {}
            }
        }
    }

    Ok(signals)
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Task {
    id: String,
    #[serde(default)]
    subject: String,
    #[serde(default)]
    status: String,
}

fn read_active_session_tasks(
    fs: &dyn FileSystemPort,
    session_id: &str,
) -> Result<Vec<Task>, String> {
    let home = fs
        .home_dir()
        .ok_or_else(|| "cannot determine home directory for session task lookup".to_string())?;
    let session_dir = super::session_task_dir(fs, &home, session_id);
    if !fs.is_dir(&session_dir) {
        return Ok(Vec::new());
    }
    let entries = fs
        .read_dir(&session_dir)
        .map_err(|err| format!("failed to list task dir {}: {err}", session_dir.display()))?;

    let mut tasks = Vec::new();
    for path in entries {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if !Path::new(&name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
            || name.starts_with('.')
        {
            continue;
        }
        let content = fs
            .read_to_string(&path)
            .map_err(|err| format!("failed to read task file {}: {err}", path.display()))?;
        let task = serde_json::from_str::<Task>(&content)
            .map_err(|err| format!("failed to parse task file {}: {err}", path.display()))?;
        tasks.push(task);
    }
    tasks.sort_by(|a, b| {
        let a_num: u32 = a.id.parse().unwrap_or(u32::MAX);
        let b_num: u32 = b.id.parse().unwrap_or(u32::MAX);
        a_num.cmp(&b_num).then(a.id.cmp(&b.id))
    });
    Ok(tasks)
}

fn pending_task_hint(tasks: &[Task]) -> Option<String> {
    tasks
        .iter()
        .find(|task| task.status == "pending")
        .map(|task| {
            let subject = if task.subject.trim().is_empty() {
                "(no subject)"
            } else {
                task.subject.as_str()
            };
            format!("Task #{} is pending: \"{}\".", task.id, subject)
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolUsageDecision {
    AllowNoTool,
    AllowTriviallyReversible,
    AllowA3Handoff,
    Allow,
    DenyMissingSessionId,
    DenyMissingTranscriptPath,
    DenyTranscriptAuthority,
    DenyTaskListAuthority,
    DenyMissingSequentialThinking,
    DenyMissingTaskList,
    DenyPlanInProgress,
    DenyMissingApprovedPlan,
    DenyMissingInProgressTask,
}

#[derive(Debug, Clone)]
pub struct ToolUsageEvaluation {
    pub tool: Option<String>,
    pub tool_present: bool,
    pub reversibility_class: Option<ReversibilityClass>,
    pub a3_enabled: bool,
    pub a3_handoff: bool,
    pub gate_required: bool,
    pub session_id: Option<String>,
    pub session_id_present: bool,
    pub transcript_path: Option<String>,
    pub transcript_path_present: bool,
    pub transcript_authority_read: bool,
    pub transcript_authority_error: Option<String>,
    pub sequential_thinking_used: bool,
    /// Whether the `sequential-thinking` MCP server is registered this session
    /// (`~/.claude.json`), i.e. whether the seq-thinking demand is enforceable.
    /// When false the gate fails open on a missing seq-thinking signal (the
    /// deadlock guard below) and the LangGraph replica must model the same.
    pub sequential_thinking_requirable: bool,
    pub plan_state: PlanState,
    pub task_authority_read: bool,
    pub task_authority_error: Option<String>,
    pub task_count: usize,
    pub in_progress_task_present: bool,
    pub pending_task_hint: Option<String>,
    pub should_deny: bool,
    pub decision: ToolUsageDecision,
}

impl ToolUsageEvaluation {
    #[must_use]
    pub const fn graph_authority_required(&self) -> bool {
        self.gate_required
    }
}

/// Process a `PreToolUse` event. Routes by reversibility class (A6, per
/// `docs/a6-reversibility-graded-tripwires.md`):
///
/// - `TriviallyReversible` -> allow without extra gate context (memory writes,
///   plan files, read-only ops, list/get MCP tools per the shipped TOML).
/// - `ReversibleWithEffort` -> run the four-check stack (transcript-confirmed
///   sequential thinking + active session task list + approved plan +
///   `in_progress` task).
/// - `Irreversible` / `Catastrophic` -> when `a3_enabled` is `true`,
///   short-circuit to `allow()` so the A3 `dry_run_then_commit` hook
///   handles the gating via its separate-model-family auditor. When
///   `a3_enabled` is `false` (tests or non-hook callers only), continue
///   to the four-check stack as the strongest available gate.
pub fn process(
    input: &HookInput,
    fs: &dyn FileSystemPort,
    _env: &dyn EnvPort,
    classifier: &dyn ReversibilityClassifierPort,
    a3_enabled: bool,
) -> HookOutput {
    let evaluation = evaluate(input, fs, classifier, a3_enabled);
    output_from_evaluation(&evaluation)
}

/// True when a `sequential-thinking` MCP server is registered in
/// `~/.claude.json` — checked in BOTH the top-level `mcpServers` map AND every
/// `projects.*.mcpServers` map (Claude Code stores user-scope servers at the
/// top level and project-scope servers under the project path). Matching is
/// case-insensitive on the server key containing `"sequential"`, so both
/// `sequential-thinking` and `sequentialthinking` register.
///
/// When this returns `false` the `mcp__sequential-thinking__sequentialthinking`
/// tool cannot be called this session, so the gate must NOT hard-block on a
/// signal that can never be produced. Any IO/parse failure returns `false`
/// (treat an unreadable config as "not registered" -> fail open, don't block).
fn sequential_thinking_mcp_registered(fs: &dyn FileSystemPort) -> bool {
    let Some(home) = fs.home_dir() else {
        return false;
    };
    let Ok(content) = fs.read_to_string(&home.join(".claude.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    let has_seq_key = |servers: &serde_json::Value| -> bool {
        servers.as_object().is_some_and(|m| {
            m.keys()
                .any(|k| k.to_ascii_lowercase().contains("sequential"))
        })
    };
    if json.get("mcpServers").is_some_and(has_seq_key) {
        return true;
    }
    json.get("projects")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|projects| {
            projects
                .values()
                .any(|proj| proj.get("mcpServers").is_some_and(has_seq_key))
        })
}

pub fn evaluate(
    input: &HookInput,
    fs: &dyn FileSystemPort,
    classifier: &dyn ReversibilityClassifierPort,
    a3_enabled: bool,
) -> ToolUsageEvaluation {
    let mut evaluation = ToolUsageEvaluation {
        tool: input.tool_name.clone(),
        tool_present: input
            .tool_name
            .as_deref()
            .is_some_and(|tool| !tool.is_empty()),
        reversibility_class: None,
        a3_enabled,
        a3_handoff: false,
        gate_required: false,
        session_id: input.session_id.clone().filter(|id| !id.is_empty()),
        session_id_present: input.session_id.as_deref().is_some_and(|id| !id.is_empty()),
        transcript_path: input
            .transcript_path
            .clone()
            .filter(|path| !path.is_empty()),
        transcript_path_present: input
            .transcript_path
            .as_deref()
            .is_some_and(|path| !path.is_empty()),
        transcript_authority_read: false,
        transcript_authority_error: None,
        sequential_thinking_used: false,
        sequential_thinking_requirable: false,
        plan_state: PlanState::Missing,
        task_authority_read: false,
        task_authority_error: None,
        task_count: 0,
        in_progress_task_present: false,
        pending_task_hint: None,
        should_deny: false,
        decision: ToolUsageDecision::AllowNoTool,
    };

    let tool = match &input.tool_name {
        Some(t) => t.as_str(),
        None => return evaluation,
    };

    // Master switch (operator config `enabled = false`): the whole gate stands
    // down — allow without classifying, and leave `gate_required` false so no
    // graph authority runs (nothing to keep in parity when nothing decides).
    if !load_config(fs).enabled {
        eprintln!(
            "[sentinel] tool_usage_gate: disabled by operator config \
             (tool-usage-gate.toml enabled = false) — allowing without checks."
        );
        return evaluation;
    }

    // A6 Phase 4b + A3 Phase 4: class-based dispatch with A3 hand-off.
    let null_input = serde_json::Value::Null;
    let tool_input_ref = input.tool_input.as_ref().unwrap_or(&null_input);
    let class = classifier.classify(tool, tool_input_ref);
    evaluation.reversibility_class = Some(class);
    match class {
        ReversibilityClass::TriviallyReversible => {
            evaluation.decision = ToolUsageDecision::AllowTriviallyReversible;
            return evaluation;
        }
        ReversibilityClass::ReversibleWithEffort => {}
        ReversibilityClass::Irreversible | ReversibilityClass::Catastrophic => {
            if a3_enabled {
                evaluation.a3_handoff = true;
                evaluation.decision = ToolUsageDecision::AllowA3Handoff;
                return evaluation;
            }
        }
    }

    evaluation.gate_required = true;

    let session_id = match &evaluation.session_id {
        Some(id) if !id.is_empty() => id.as_str(),
        _ => {
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyMissingSessionId;
            return evaluation;
        }
    };

    let transcript_path = match evaluation
        .transcript_path
        .as_deref()
        .filter(|p| !p.is_empty())
    {
        Some(path) => Path::new(path),
        None => {
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyMissingTranscriptPath;
            return evaluation;
        }
    };
    let transcript = match read_transcript_signals(fs, transcript_path) {
        Ok(signals) => {
            evaluation.transcript_authority_read = true;
            signals
        }
        Err(err) => {
            evaluation.transcript_authority_error = Some(err);
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyTranscriptAuthority;
            return evaluation;
        }
    };
    evaluation.sequential_thinking_used = transcript.sequential_thinking_used;
    evaluation.plan_state = transcript.plan_state;

    let tasks = match read_active_session_tasks(fs, session_id) {
        Ok(tasks) => {
            evaluation.task_authority_read = true;
            tasks
        }
        Err(err) => {
            evaluation.task_authority_error = Some(err);
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyTaskListAuthority;
            return evaluation;
        }
    };
    evaluation.task_count = tasks.len();
    evaluation.in_progress_task_present = tasks.iter().any(|task| task.status == "in_progress");
    evaluation.pending_task_hint = pending_task_hint(&tasks);

    // DEADLOCK GUARD: only demand sequential thinking when the check is
    // enabled by operator config AND the `sequential-thinking` MCP server is
    // actually registered in `~/.claude.json` (so
    // `mcp__sequential-thinking__sequentialthinking` can be called and the
    // transcript signal can be earned). When it is NOT registered, the tool
    // does not exist this session — denying on its absence bricks every code
    // edit with no fix path (the exact self-bricking class the
    // task_decomposition_gate had). Both fail-open legs live in
    // `sequential_thinking_requirable` so the LangGraph replica can model the
    // same policy instead of mismatching on either leg.
    evaluation.sequential_thinking_requirable =
        load_config(fs).sequential_thinking_check && sequential_thinking_mcp_registered(fs);
    if !transcript.sequential_thinking_used {
        if evaluation.sequential_thinking_requirable {
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyMissingSequentialThinking;
            return evaluation;
        }
        eprintln!(
            "[sentinel] tool_usage_gate: seq-thinking requirement skipped — disabled by \
             operator config or MCP server not registered in ~/.claude.json (cannot \
             demand a signal that cannot be produced this session)."
        );
    }

    if tasks.is_empty() {
        evaluation.should_deny = true;
        evaluation.decision = ToolUsageDecision::DenyMissingTaskList;
        return evaluation;
    }

    match transcript.plan_state {
        PlanState::Approved => {}
        PlanState::InPlanMode => {
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyPlanInProgress;
            return evaluation;
        }
        PlanState::Missing => {
            evaluation.should_deny = true;
            evaluation.decision = ToolUsageDecision::DenyMissingApprovedPlan;
            return evaluation;
        }
    }

    if !evaluation.in_progress_task_present {
        evaluation.should_deny = true;
        evaluation.decision = ToolUsageDecision::DenyMissingInProgressTask;
        return evaluation;
    }

    evaluation.decision = ToolUsageDecision::Allow;
    evaluation
}

pub fn output_from_evaluation(evaluation: &ToolUsageEvaluation) -> HookOutput {
    match evaluation.decision {
        ToolUsageDecision::AllowNoTool
        | ToolUsageDecision::AllowTriviallyReversible
        | ToolUsageDecision::AllowA3Handoff
        | ToolUsageDecision::Allow => HookOutput::allow(),
        ToolUsageDecision::DenyMissingSessionId => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Hook input did not include a session_id, \
             so Sentinel cannot verify transcript and TaskList authority.",
        ),
        ToolUsageDecision::DenyMissingTranscriptPath => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Hook input did not include transcript_path, \
             so Sentinel cannot verify sequential thinking or plan approval authority.",
        ),
        ToolUsageDecision::DenyTranscriptAuthority => {
            let err = evaluation
                .transcript_authority_error
                .as_deref()
                .unwrap_or("unknown transcript authority error");
            HookOutput::deny(format!(
                "🔴 [Tool Usage Gate] BLOCKED: Transcript authority unavailable: {err}"
            ))
        }
        ToolUsageDecision::DenyTaskListAuthority => {
            let err = evaluation
                .task_authority_error
                .as_deref()
                .unwrap_or("unknown TaskList authority error");
            HookOutput::deny(format!(
                "🔴 [Tool Usage Gate] BLOCKED: TaskList authority unavailable: {err}"
            ))
        }
        ToolUsageDecision::DenyMissingSequentialThinking => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Use `mcp__sequential-thinking__sequentialthinking` \
             to think through your approach before making code changes.",
        ),
        ToolUsageDecision::DenyMissingTaskList => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Create a task with `TaskCreate` before \
             making code changes. All work must be tracked as a task. \
             Note: `TodoWrite` is NOT accepted — the operator's CLAUDE.md mandates the \
             agent-harness `TaskCreate`/`TaskUpdate` (TaskList) tool.",
        ),
        ToolUsageDecision::DenyPlanInProgress => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Plan mode is active but no approved plan \
             is recorded yet. Call `ExitPlanMode` with the plan content before \
             making code changes.",
        ),
        ToolUsageDecision::DenyMissingApprovedPlan => HookOutput::deny(
            "🔴 [Tool Usage Gate] BLOCKED: Approved plan not found in the live \
             transcript. Enter plan mode with `EnterPlanMode`, then call \
             `ExitPlanMode` with the plan content before making code changes.",
        ),
        ToolUsageDecision::DenyMissingInProgressTask => {
            let hint = evaluation.pending_task_hint.clone().unwrap_or_default();
            let msg = if hint.is_empty() {
                "🔴 [Tool Usage Gate] BLOCKED: Create a task with `TaskCreate` and have \
                 one in `in_progress` before making code changes. All work must be \
                 tracked as an active task. Note: `TodoWrite` is NOT accepted — \
                 the operator's CLAUDE.md mandates the agent-harness `TaskCreate`/`TaskUpdate` \
                 (TaskList) tool."
                    .to_string()
            } else {
                format!(
                    "🔴 [Tool Usage Gate] BLOCKED: Mark a task as `in_progress` before making \
                     code changes. {hint} Use `TaskUpdate(taskId: \"<id>\", \
                     status: \"in_progress\")`. Note: `TodoWrite` is NOT accepted — \
                     the operator's CLAUDE.md mandates the agent-harness TaskList tool."
                )
            };
            HookOutput::deny(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_domain::port_errors::FileSystemError;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::{NamedTempFile, TempDir};

    struct RealFs {
        home: PathBuf,
    }

    impl RealFs {
        fn new(home: PathBuf) -> Self {
            Self { home }
        }
    }

    impl FileSystemPort for RealFs {
        fn home_dir(&self) -> Option<PathBuf> {
            Some(self.home.clone())
        }

        fn read_to_string(&self, p: &Path) -> Result<String, FileSystemError> {
            Ok(fs::read_to_string(p)?)
        }

        fn write(&self, p: &Path, c: &[u8]) -> Result<(), FileSystemError> {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)?;
            }
            Ok(fs::write(p, c)?)
        }

        fn create_dir_all(&self, p: &Path) -> Result<(), FileSystemError> {
            Ok(fs::create_dir_all(p)?)
        }

        fn read_dir(&self, p: &Path) -> Result<Vec<PathBuf>, FileSystemError> {
            Ok(fs::read_dir(p)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .collect())
        }

        fn exists(&self, p: &Path) -> bool {
            p.exists()
        }

        fn is_dir(&self, p: &Path) -> bool {
            p.is_dir()
        }

        fn metadata(&self, p: &Path) -> Result<fs::Metadata, FileSystemError> {
            Ok(fs::metadata(p)?)
        }

        fn append(&self, p: &Path, c: &[u8]) -> Result<(), FileSystemError> {
            use std::io::Write;
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::OpenOptions::new().create(true).append(true).open(p)?;
            file.write_all(c)?;
            Ok(())
        }
    }

    fn permissive_classifier() -> crate::reversibility_classifier::StaticReversibilityClassifier {
        crate::reversibility_classifier::StaticReversibilityClassifier::empty()
            .with_default(ReversibilityClass::ReversibleWithEffort)
    }

    fn classifier_for(
        class: ReversibilityClass,
    ) -> crate::reversibility_classifier::StaticReversibilityClassifier {
        crate::reversibility_classifier::StaticReversibilityClassifier::empty().with("Edit", class)
    }

    fn assistant_tool_use(name: &str) -> serde_json::Value {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": name, "input": {}}]
            }
        })
    }

    fn write_transcript(entries: &[serde_json::Value]) -> NamedTempFile {
        use std::io::Write;
        let mut file = NamedTempFile::new().unwrap();
        for entry in entries {
            writeln!(file, "{}", serde_json::to_string(entry).unwrap()).unwrap();
        }
        file
    }

    fn write_broken_transcript() -> NamedTempFile {
        use std::io::Write;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "{}",
            serde_json::to_string(&assistant_tool_use(
                "mcp__sequential-thinking__sequentialthinking"
            ))
            .unwrap()
        )
        .unwrap();
        writeln!(file, "not-json").unwrap();
        file
    }

    fn seed_task(home: &Path, session_id: &str, id: &str, subject: &str, status: &str) -> PathBuf {
        let dir = home.join(".claude").join("tasks").join(session_id);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{id}.json"));
        fs::write(
            &path,
            serde_json::json!({
                "id": id,
                "subject": subject,
                "status": status
            })
            .to_string(),
        )
        .unwrap();
        path
    }

    /// Write a `~/.claude.json` into the test home that registers a
    /// `sequential-thinking` MCP server, so `sequential_thinking_mcp_registered`
    /// sees it as available and Check 1 hard-blocks (rather than failing open).
    fn seed_claude_json_with_sequential(home: &Path) {
        fs::write(
            home.join(".claude.json"),
            serde_json::json!({
                "mcpServers": {
                    "sequential-thinking": { "command": "mcp-router --single sequential-thinking-mcp" }
                }
            })
            .to_string(),
        )
        .unwrap();
    }

    fn edit_input(session_id: &str, transcript: &NamedTempFile) -> HookInput {
        HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some(session_id.to_string()),
            transcript_path: Some(transcript.path().to_string_lossy().into_owned()),
            ..Default::default()
        }
    }

    fn deny_reason(output: &HookOutput) -> String {
        output
            .hook_specific_output
            .as_ref()
            .and_then(|hook| hook.permission_decision_reason.as_deref())
            .unwrap_or("")
            .to_string()
    }

    #[test]
    fn unreadable_transcript_denies_transcript_authority() {
        // Fail CLOSED: if the transcript path is present but the file cannot be
        // read, the gate cannot verify sequential-thinking / plan authority, so
        // it must DENY — not silently allow. Pins the fail-closed behavior so a
        // refactor can't flip this read error into an allow.
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let fs = RealFs::new(tmp.path().to_path_buf());
        // A transcript_path that points at a file which does not exist →
        // RealFs::read_to_string returns Err → DenyTranscriptAuthority.
        let missing = tmp.path().join("does-not-exist.jsonl");
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some("sess".to_string()),
            transcript_path: Some(missing.to_string_lossy().into_owned()),
            ..Default::default()
        };

        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(
            output.blocked,
            Some(true),
            "an unreadable transcript must DENY (fail closed), not allow"
        );
        assert!(
            deny_reason(&output).contains("Transcript authority"),
            "deny reason should name the transcript-authority failure, got: {}",
            deny_reason(&output)
        );
    }

    #[test]
    fn unreadable_task_dir_denies_tasklist_authority() {
        // Fail CLOSED: once past the transcript check, if the active-session
        // task directory cannot be read/parsed, the gate cannot confirm an
        // in_progress task exists, so it must DENY rather than allow the edit.
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        // A well-formed transcript that satisfies the sequential-thinking check
        // so evaluation proceeds to the task-authority read.
        let transcript = write_transcript(&[
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("EnterPlanMode"),
            assistant_tool_use("ExitPlanMode"),
        ]);
        // Seed a valid task then poison the task dir with an unparseable file so
        // read_active_session_tasks returns Err → DenyTaskListAuthority.
        seed_task(home, "sess", "1", "First", "in_progress");
        fs::write(home.join(".claude/tasks/sess/bad.json"), "not-json").unwrap();

        let input = edit_input("sess", &transcript);
        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(
            output.blocked,
            Some(true),
            "an unreadable/garbage task dir must DENY (fail closed), not allow"
        );
        assert!(
            deny_reason(&output).contains("TaskList authority"),
            "deny reason should name the TaskList-authority failure, got: {}",
            deny_reason(&output)
        );
    }

    #[test]
    fn transcript_signals_track_sequential_and_latest_plan_state() {
        let tmp = TempDir::new().unwrap();
        let fs = RealFs::new(tmp.path().to_path_buf());
        let transcript = write_transcript(&[
            assistant_tool_use("ExitPlanMode"),
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("EnterPlanMode"),
        ]);

        let signals = read_transcript_signals(&fs, transcript.path()).unwrap();
        assert!(signals.sequential_thinking_used);
        assert_eq!(signals.plan_state, PlanState::InPlanMode);
    }

    #[test]
    fn malformed_transcript_is_authority_error() {
        let tmp = TempDir::new().unwrap();
        let fs = RealFs::new(tmp.path().to_path_buf());
        let transcript = write_broken_transcript();

        let err = read_transcript_signals(&fs, transcript.path()).unwrap_err();
        assert!(err.contains("malformed transcript JSON"));
    }

    #[test]
    fn reads_only_active_session_task_dir_strictly() {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_task(home, "session-a", "2", "Second", "pending");
        seed_task(home, "session-a", "1", "First", "in_progress");
        seed_task(home, "session-b", "1", "Other session", "in_progress");

        let tasks = read_active_session_tasks(&fs, "session-a").unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].id, "1");
        assert_eq!(tasks[1].id, "2");

        fs::write(home.join(".claude/tasks/session-a/bad.json"), "not-json").unwrap();
        let err = read_active_session_tasks(&fs, "session-a").unwrap_err();
        assert!(err.contains("failed to parse task file"));
    }

    #[test]
    fn prefixed_session_task_dir_is_accepted_without_cross_session_scan() {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        let session_id = "e2ea5630-3c79-409c-9ca4-423975a5a5fb";
        seed_task(
            home,
            "session-e2ea5630",
            "1",
            "Prefixed session",
            "in_progress",
        );

        let tasks = read_active_session_tasks(&fs, session_id).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].subject, "Prefixed session");
    }

    #[test]
    fn missing_session_id_denies() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let fs = RealFs::new(tmp.path().to_path_buf());
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            ..Default::default()
        };

        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("session_id"));
    }

    #[test]
    fn missing_transcript_path_denies() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let fs = RealFs::new(tmp.path().to_path_buf());
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some("sess".to_string()),
            ..Default::default()
        };

        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("transcript_path"));
    }

    #[test]
    fn blocks_without_sequential_thinking_even_with_plan_and_task() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("sequential-thinking"));
    }

    /// Seed the operator override config into the test home:
    /// `~/.claude/sentinel/config/tool-usage-gate.toml`.
    fn seed_gate_config(home: &Path, contents: &str) {
        let dir = home.join(".claude").join("sentinel").join("config");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("tool-usage-gate.toml"), contents).unwrap();
    }

    #[test]
    fn operator_config_disables_check1_only() {
        // Same setup as the block-template test above (seq-thinking NOT used,
        // MCP registered so Check 1 would normally hard-block) — but with the
        // operator override setting sequential_thinking_check = false. Check 1
        // is skipped; the plan is approved and an in_progress task exists, so
        // the edit is ALLOWED.
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        seed_gate_config(home, "sequential_thinking_check = false\n");
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert!(
            output.blocked.is_none(),
            "operator config must skip Check 1 when other checks pass; got: {output:?}"
        );
    }

    #[test]
    fn operator_config_enabled_false_disables_whole_gate() {
        // enabled = false: NO task seeded, seq-thinking not used, no plan —
        // every check would block, but the master switch stands the gate down
        // entirely and no graph authority is required.
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        seed_gate_config(home, "enabled = false\n");
        let transcript = write_transcript(&[]);

        let evaluation = evaluate(
            &edit_input("sess", &transcript),
            &fs,
            &permissive_classifier(),
            false,
        );
        assert!(!evaluation.should_deny, "disabled gate must not deny");
        assert!(
            !evaluation.graph_authority_required(),
            "disabled gate must not consult the graph authority"
        );
        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert!(
            output.blocked.is_none(),
            "disabled gate must allow: {output:?}"
        );
    }

    #[test]
    fn operator_config_does_not_bypass_other_checks() {
        // Check 1 disabled but NO task seeded → the task-list check must still
        // block. Proves the config knob is Check-1-only, not a gate bypass.
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        seed_gate_config(home, "sequential_thinking_check = false\n");
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(
            output.blocked,
            Some(true),
            "config knob must not bypass the task-list/active-task checks"
        );
        assert!(
            !deny_reason(&output).contains("sequential-thinking"),
            "block must come from a later check, not Check 1"
        );
    }

    #[test]
    fn corrupt_operator_config_fails_closed_to_enforcement() {
        // An unparseable override must NOT relax the gate: Check 1 still blocks.
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        seed_gate_config(home, "sequential_thinking_check = maybe???\n");
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(
            output.blocked,
            Some(true),
            "corrupt config must fail closed (Check 1 enforced)"
        );
        assert!(deny_reason(&output).contains("sequential-thinking"));
    }

    #[test]
    fn operator_config_with_unknown_key_fails_closed_to_enforcement() {
        // deny_unknown_fields: a typo'd key (e.g. `sequential_thinking_chekc`)
        // must NOT be silently accepted. If it were, serde would ignore the
        // unknown key, the intended `false` override would never apply, and the
        // gate would stay enforced with no warning — confusing for operators.
        // The override parse must fail, and `from_toml_or_default` must fall
        // back to the enforced default (Check 1 on).
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        seed_claude_json_with_sequential(home);
        // Typo'd key + a *valid* `false` that should be ignored because the
        // whole document is rejected on the unknown field.
        seed_gate_config(
            home,
            "sequential_thinking_chekc = false\nsequential_thinking_check = false\n",
        );
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        // Unknown key => override rejected => enforced default (Check 1 on) =>
        // seq-thinking absent from this transcript => blocked by Check 1.
        assert_eq!(
            output.blocked,
            Some(true),
            "unknown config key must fail closed (override rejected, Check 1 enforced)"
        );
        assert!(deny_reason(&output).contains("sequential-thinking"));
    }

    #[test]
    fn allows_when_sequential_thinking_mcp_not_registered() {
        // DEADLOCK GUARD: seq-thinking NOT used and the MCP NOT registered in
        // ~/.claude.json (no such file in this temp home). The gate must NOT
        // demand a tool that cannot be called — Check 1 fails open and the
        // request proceeds (here it reaches the plan/task checks). Specifically
        // it must NOT deny with the sequential-thinking message.
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        // No .claude.json seeded → MCP unregistered.
        let transcript = write_transcript(&[assistant_tool_use("ExitPlanMode")]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        // The seq-thinking requirement is skipped; with a plan + active task the
        // request is allowed. Crucially, it is NOT blocked for sequential-thinking.
        let denied_for_seq =
            output.blocked.unwrap_or(false) && deny_reason(&output).contains("sequential-thinking");
        assert!(
            !denied_for_seq,
            "must NOT block on sequential-thinking when its MCP is unregistered"
        );
    }

    #[test]
    fn blocks_without_session_task() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let fs = RealFs::new(tmp.path().to_path_buf());
        let transcript = write_transcript(&[
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("ExitPlanMode"),
        ]);

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("TaskCreate"));
        assert!(deny_reason(&output).contains("TodoWrite"));
    }

    #[test]
    fn blocks_when_plan_is_not_approved() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        let transcript = write_transcript(&[
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("EnterPlanMode"),
        ]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("ExitPlanMode"));
        assert!(deny_reason(&output).contains("no approved plan"));
    }

    #[test]
    fn blocks_without_in_progress_task_and_uses_pending_hint() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        let transcript = write_transcript(&[
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("ExitPlanMode"),
        ]);
        seed_task(home, "sess", "1", "Pending work", "pending");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        let reason = deny_reason(&output);
        assert!(reason.contains("Task #1 is pending"));
        assert!(reason.contains("in_progress"));
    }

    #[test]
    fn allows_with_live_transcript_and_active_task() {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        let transcript = write_transcript(&[
            assistant_tool_use("mcp__sequential-thinking__sequentialthinking"),
            assistant_tool_use("EnterPlanMode"),
            assistant_tool_use("ExitPlanMode"),
        ]);
        seed_task(home, "sess", "1", "Active", "in_progress");

        let output = process(
            &edit_input("sess", &transcript),
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert!(output.blocked.is_none(), "got: {output:?}");
    }

    #[test]
    fn old_temp_marker_files_do_not_authorize_gate() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let home = tmp.path();
        let fs = RealFs::new(home.to_path_buf());
        let session = "sess-marker";
        for name in [
            "claude-sequential-used-sess-marker",
            "claude-task-created-sess-marker",
            "claude-plan-approved-sess-marker",
            "claude-task-active-sess-marker",
        ] {
            fs::write(std::env::temp_dir().join(name), b"1").unwrap();
        }
        seed_task(home, session, "1", "Active", "in_progress");

        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some(session.to_string()),
            ..Default::default()
        };
        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &permissive_classifier(),
            false,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("transcript_path"));
    }

    #[test]
    fn trivially_reversible_short_circuits_before_authority_checks() {
        let tmp = TempDir::new().unwrap();
        let fs = RealFs::new(tmp.path().to_path_buf());
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some("sess".to_string()),
            ..Default::default()
        };

        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &classifier_for(ReversibilityClass::TriviallyReversible),
            false,
        );
        assert!(output.blocked.is_none());
    }

    #[test]
    fn a3_enabled_defers_irreversible_to_a3() {
        let tmp = TempDir::new().unwrap();
        let fs = RealFs::new(tmp.path().to_path_buf());
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some("sess".to_string()),
            ..Default::default()
        };

        for class in [
            ReversibilityClass::Irreversible,
            ReversibilityClass::Catastrophic,
        ] {
            let output = process(
                &input,
                &fs,
                &crate::hooks::test_support::StubEnv::new(),
                &classifier_for(class),
                true,
            );
            assert!(output.blocked.is_none(), "class {class:?} must defer");
        }
    }

    #[test]
    fn a3_enabled_still_gates_reversible_with_effort() {
        let tmp = TempDir::new().unwrap();
        seed_gate_config(tmp.path(), "enabled = true\n");
        let fs = RealFs::new(tmp.path().to_path_buf());
        let input = HookInput {
            tool_name: Some("Edit".to_string()),
            session_id: Some("sess".to_string()),
            ..Default::default()
        };

        let output = process(
            &input,
            &fs,
            &crate::hooks::test_support::StubEnv::new(),
            &classifier_for(ReversibilityClass::ReversibleWithEffort),
            true,
        );
        assert_eq!(output.blocked, Some(true));
        assert!(deny_reason(&output).contains("transcript_path"));
    }
}
