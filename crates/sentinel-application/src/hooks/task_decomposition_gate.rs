//! Task Decomposition Gate — require a live, decomposed task list before
//! mutating work.
//!
//! The user's standing complaint is that "tasks are not always enforced":
//! the agent starts editing files / running state-changing commands without
//! ever decomposing the work into tracked tasks. This `PreToolUse` gate closes
//! that gap. Before a *mutating* tool is allowed, there must be evidence that
//! a decomposed task list exists for this session.
//!
//! Design mirrors `bug_task_gate` and `skill_invocation_gate`:
//!   - READ tools and Task* tools are NEVER gated. Gating them would deadlock
//!     the session — the very tools the agent uses to satisfy the gate
//!     (`TaskCreate`, `TaskList`, `Read`, sequential-thinking) must always
//!     pass, otherwise there is no fix path.
//!   - MUTATING tools (`Edit`, `Write`, `NotebookEdit`, and obvious
//!     state-changing `Bash` commands) are blocked ONLY when no live task
//!     list can be observed for the current session.
//!   - The "is there a live task list" check reads the
//!     session's on-disk task files (`~/.claude/tasks/{session_id}/*.json`,
//!     the same files `task_persist` snapshots from). If that state cannot be
//!     read (no session id, no home dir, unreadable dir), the gate fails closed:
//!     mutating tools block because compliance has not been proven.
//!   - "Decomposed earlier" survives task completion. The task store prunes a
//!     task's `*.json` file when it completes, so a session that correctly
//!     finished all its tasks would otherwise present an empty dir and get
//!     re-blocked on its next mutating tool — punishing the agent for closing
//!     out its work. To avoid that false positive the gate also queries the
//!     store's monotonic `.highwatermark` counter (the highest task id ever
//!     issued this session, which persists across pruning): if it is `>= 1`
//!     the session has decomposed at least once and the gate allows even when
//!     no open task files remain. Genuinely fresh sessions (no highwatermark,
//!     or `0`) are still gated on their first mutating tool.
//!
//! The block message is `[Sentinel-Authority]`-prefixed (added automatically
//! at the output boundary by `HookOutput::into_pretool_output`) and instructs
//! the agent to create a properly decomposed task list using the exact
//! conventions sentinel expects (numbered tasks, blocking dependency graph,
//! priority tags + emoji, real Na/Nb child subtasks, status CRUD).

use sentinel_domain::events::{HookEnvelope, HookInput, HookOutput};

// Shipped defaults compiled in; wholesale-replaced by the operator override at
// `~/.claude/sentinel/config/task-decomposition-gate.toml` when present —
// same pattern as tool_usage_gate.
const SHIPPED_DEFAULTS: &str =
    include_str!("../../../../config/task-decomposition-gate-defaults.toml");

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskDecompositionGateConfig {
    /// Master switch. `false` disables the whole gate: mutating tools are
    /// never blocked for a missing task list and no graph authority is
    /// consulted. For controlled environments (benchmark arms);
    /// missing/corrupt config falls back to `true` — fail closed to
    /// enforcement.
    #[serde(default = "default_true")]
    enabled: bool,
}

const fn default_true() -> bool {
    true
}

impl Default for TaskDecompositionGateConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl TaskDecompositionGateConfig {
    fn from_toml_or_default(s: &str) -> Self {
        toml::from_str(s).unwrap_or_else(|e| {
            eprintln!(
                "[sentinel] task_decomposition_gate: config TOML parse failed ({e}); \
                 using enforced defaults"
            );
            Self::default()
        })
    }
}

fn load_config(fs: &dyn super::FileSystemPort) -> TaskDecompositionGateConfig {
    let mut cfg = TaskDecompositionGateConfig::from_toml_or_default(SHIPPED_DEFAULTS);
    let path = fs
        .claude_dir()
        .join("sentinel")
        .join("config")
        .join("task-decomposition-gate.toml");
    if let Ok(content) = fs.read_to_string(&path) {
        cfg = TaskDecompositionGateConfig::from_toml_or_default(&content);
    }
    cfg
}
use std::path::PathBuf;

use super::{FileSystemPort, HookContext};

/// Read-only / progress-toward-decomposition tools that must NEVER be gated.
/// This is the critical safety list: the tools the agent uses to *satisfy* the
/// gate (Task*, Read, Skill, sequential-thinking) and all pure-read tools.
/// Gating any of these deadlocks the session and the fix path.
const ALLOWED_TOOLS: &[&str] = &[
    "Read",
    "Glob",
    "Grep",
    "LS",
    "LSP",
    "NotebookRead",
    "WebSearch",
    "WebFetch",
    "ToolSearch",
    "Skill",
    "TaskList",
    "TaskGet",
    "TaskCreate",
    "TaskUpdate",
    "TaskOutput",
];

/// Tools that are unconditionally treated as mutating.
const MUTATING_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit", "MultiEdit"];

/// Obvious state-changing substrings in a Bash command. Conservative on
/// purpose — read-only bash (ls, cat, grep, git status/log/diff, find) must
/// pass. We only flag commands that clearly produce lasting change.
const MUTATING_BASH_MARKERS: &[&str] = &[
    "git commit",
    "git push",
    "git merge",
    "git reset",
    "git rebase",
    "git cherry-pick",
    "git stash",
    "git rm",
    "git mv",
    "git tag",
    "rm ",
    "rm -",
    "mv ",
    "cp ",
    "mkdir ",
    "touch ",
    "cargo build",
    "cargo install",
    "cargo publish",
    "npm install",
    "npm i ",
    "pnpm install",
    "yarn add",
    "yarn install",
    "pip install",
    "sed -i",
    "tee ",
    // NOTE: stdout/file redirection (`>`, `>>`) is intentionally NOT listed
    // here. It is detected by the dedicated `>`-after-stderr-strip check in
    // `is_mutating_bash`, which correctly exempts stderr redirections
    // (`2>&1`, `2>>err.log`). Listing `>>`/`>` here would re-match the raw
    // command and wrongly flag those stderr forms as writes.
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskDecompositionDecision {
    Allow,
    Block,
}

#[derive(Debug, Clone)]
pub struct TaskDecompositionEvaluation {
    pub tool: Option<String>,
    pub session_id: Option<String>,
    pub bash_command: Option<String>,
    pub allowed_tool: bool,
    pub bash_tool: bool,
    pub bash_command_present: bool,
    pub mutating_tool: bool,
    pub task_state_readable: bool,
    pub task_list_confirmed: bool,
    pub unreadable_task_state: bool,
    pub should_block: bool,
    pub decision: TaskDecompositionDecision,
}

impl TaskDecompositionEvaluation {
    #[must_use]
    pub const fn graph_authority_required(&self) -> bool {
        self.mutating_tool
    }
}

/// Whether the tool is on the always-allow list.
fn is_allowed_tool(tool_name: &str) -> bool {
    if ALLOWED_TOOLS.contains(&tool_name) {
        return true;
    }
    // Treat the entire sequential-thinking MCP namespace as a read tool.
    tool_name.starts_with("mcp__sequential-thinking__")
}

/// Extract the Bash command string from a `HookInput`'s `tool_input`.
fn bash_command(input: &HookInput) -> Option<String> {
    input
        .tool_input
        .as_ref()?
        .get("command")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Decide whether a Bash command is state-changing. Conservative: returns
/// `false` for read-only commands and `true` only when an obvious mutating
/// marker is present.
fn is_mutating_bash(command: &str) -> bool {
    // A stdout/file redirection (`>`/`>>`) is a write. But STDERR redirections
    // (`2>&1`, `2>/dev/null`, `N>&M`, `2>file`) are NOT writes to tracked state
    // — they are ubiquitous on read-only commands (`ls 2>&1`, `find … 2>/dev/null`).
    // Strip those first so they don't trip the `>` check and block reads.
    let without_stderr = strip_stderr_redirections(command);
    if without_stderr.contains('>') {
        return true;
    }
    MUTATING_BASH_MARKERS
        .iter()
        .any(|marker| without_stderr.contains(marker))
}

/// Remove stderr-redirection tokens from a command so they don't read as
/// stdout/file writes. Handles the common forms:
///   - `2>&1`, `2>&-`        (fd dup / close)
///   - `2>/dev/null`, `2>file`, `2>>file`  (stderr to a sink — not tracked state)
///   - any `N>&M` fd-dup where the source fd is a digit
///
/// Deliberately conservative: only the recognized `\d>` / `\d>>` / `\d>&` forms
/// are stripped; a bare `>` or `>>` (stdout/file write) is left intact.
fn strip_stderr_redirections(command: &str) -> String {
    let bytes = command.as_bytes();
    let mut out = String::with_capacity(command.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // A redirection introduced by a leading fd digit, e.g. `2>`, `2>>`, `2>&1`.
        // Only treat it as stderr (strippable) when the fd is NOT 1 (stdout).
        if c.is_ascii_digit() && i + 1 < bytes.len() && bytes[i + 1] == b'>' && c != '1' {
            // Consume the digit and `>`.
            i += 2;
            // Optional second `>` (append form `2>>`).
            if i < bytes.len() && bytes[i] == b'>' {
                i += 1;
            }
            // Optional `&` + target fd (`2>&1`, `2>&-`).
            if i < bytes.len() && bytes[i] == b'&' {
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'-') {
                    i += 1;
                }
            } else {
                // `2>/dev/null`, `2>file` — skip the (optional) whitespace and
                // the redirection target path token.
                while i < bytes.len() && bytes[i] == b' ' {
                    i += 1;
                }
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' {
                    i += 1;
                }
            }
            out.push(' ');
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Classify whether the tool about to run is a mutating operation that the
/// gate cares about. `Bash` is inspected; everything else is decided by name.
fn is_mutating_tool(input: &HookInput, tool_name: &str) -> bool {
    if MUTATING_TOOLS.contains(&tool_name) {
        return true;
    }
    if tool_name == "Bash" {
        return bash_command(input).is_some_and(|c| is_mutating_bash(&c));
    }
    false
}

/// Best-effort: has this session decomposed its work?
///
/// Resolves the session's task dir via [`super::session_task_dir`] (which
/// handles both the literal-`{session_id}` and `session-{first8}` naming the
/// harness emits — see that fn for why both exist). Returns:
///   - `Some(true)`  — at least one open task file exists (a decomposed list is
///                     live), OR the session decomposed earlier this session
///                     (the `.highwatermark` counter is `>= 1`) even if every
///                     task has since completed and its file was pruned.
///   - `Some(false)` — the session dir EXISTS and is readable but shows no
///                     evidence the session ever decomposed (present, but no
///                     task files and no highwatermark). Definitive
///                     non-compliance -> block.
///   - `None`        — state could not be confirmed: no session id, no home
///                     dir, the resolved session dir is MISSING, or the dir
///                     read errored. Mutating callers FAIL OPEN on `None`.
///
/// Why a MISSING dir is `None`, not `Some(false)`: `super::session_task_dir`
/// already resolves BOTH harness naming forms, so a still-missing dir means the
/// native store wrote nowhere we can see this run — NOT that the agent skipped
/// decomposition. Treating that as definitive `Some(false)` BRICKS the session
/// (every mutating tool blocked, the dir-creating native `TaskCreate` invisible,
/// no fix path). So we report it honestly as "unconfirmable" and fail open,
/// while still hard-blocking the genuine "dir present, empty, no tasks" case.
fn has_live_task_list(fs: &dyn FileSystemPort, session_id: Option<&str>) -> Option<bool> {
    let session_id = session_id.filter(|s| !s.is_empty())?;
    let home = fs.home_dir()?;
    let session_dir = super::session_task_dir(fs, &home, session_id);
    if !fs.is_dir(&session_dir) {
        // Dir resolved by `session_task_dir` (both naming forms) still doesn't
        // exist -> the native store has not written here this run. This is
        // UNCONFIRMABLE, not definitive non-decomposition, so return `None`
        // (fail open) rather than `Some(false)`: a missing store we don't own
        // must never brick the session. The genuine "never decomposed" case is
        // caught below as a *present-but-empty* dir -> `Some(false)`.
        return None;
    }
    let entries = fs.read_dir(&session_dir).ok()?;
    if entries.iter().any(is_task_json) {
        return Some(true);
    }
    // No open task files — but completed tasks are pruned from disk, so an
    // empty dir does NOT mean the session never decomposed. Query the
    // monotonic highwatermark: if the session ever issued a task id, the
    // decomposition discipline is established and we must not re-block.
    Some(decomposed_earlier(fs, &session_dir))
}

/// Has this session ever issued a task id? Reads the store's `.highwatermark`
/// (the highest task id allocated this session, which persists across task
/// completion/pruning) and returns `true` iff it parses to a value `>= 1`.
/// Any read/parse failure is treated as "no evidence" (`false`) — the caller
/// only reaches here after confirming the session dir is readable, so a
/// missing/unparseable highwatermark legitimately means "never decomposed".
fn decomposed_earlier(fs: &dyn FileSystemPort, session_dir: &std::path::Path) -> bool {
    let hw_path = session_dir.join(".highwatermark");
    fs.read_to_string(&hw_path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .is_some_and(|n| n >= 1)
}

/// True for a non-dotfile `*.json` task file (skips `.lock`, `.highwatermark`).
fn is_task_json(path: &PathBuf) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if name.starts_with('.') {
        return false;
    }
    std::path::Path::new(&name)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
}

/// The decomposition instructions emitted in the block message. Kept as a
/// single const so the message stays in lockstep with the documented
/// convention and tests can assert on its key markers.
const DECOMPOSITION_GUIDANCE: &str = "Before mutating work, create a decomposed task list with `TaskCreate`:\n\
  • DECOMPOSITION: break the work into numbered tasks (#1, #2, …) wired as BLOCKING tasks via addBlockedBy/addBlocks (a real dependency graph — these ARE native Claude Code Task fields).\n\
  • PRIORITY on every task (Claude Code has no native priority field, so use the convention): a [P0]/[P1]/[P2]/[P3] tag in the subject AND metadata.priority, list ordered by priority.\n\
  • TWO-EMOJI PREFIX on every subject: a STATUS emoji FIRST, then the PRIORITY color emoji, then the number. Status leads, priority follows — both always visible. e.g. `⬜ 🔴 1a [P0] — …`.\n\
  • PRIORITY color emoji (2nd position): 🔴 P0 critical, 🟠 P1 high, 🟡 P2 medium, 🟢 P3 low. NEVER dropped — it stays through every status.\n\
  • SUBTASKS as REAL child tasks (no native nesting, so use the convention): each subtask of #N is its OWN task, subject-prefixed `Na`/`Nb`/`Nc` (e.g. #1 → `1a — …`, `1b — …`, `1c — …`), with its own [P]/emoji/status, wired to the parent via addBlocks (subtask blocks parent) and chained to each other via addBlockedBy where ordered. Every subtask is independently tracked — own in_progress/completed — not a checklist line in prose.\n\
  • PARENT MIRROR CHECKLIST: each parent task's description holds a ☐/☑ checklist mirroring its child subtasks (`☑ Na — …` done, `☐ Nb — …` pending) so the parent shows subtask progress at a glance. The child tasks are the source of truth; the checklist tracks them.\n\
  • A descriptive explanation + a topical emoji per task.\n\
  • STATUS EMOJI (1st position): ⬜ pending, 🔄 in_progress, ✅ completed, ❌ failed. Update it on every transition (⬜→🔄 on start, 🔄→✅ on finish, 🔄→❌ on error) — the priority color in 2nd position never changes. On completion tick ☑ in the parent checklist; on failure leave ☐ with a note.\n\
  • Native status enum is pending|in_progress|completed|failed (failed IS valid — use ❌ and keep the task, don't delete it).\n\
  • Keep updated ALWAYS — proper CRUD the entire time: in_progress on start, completed + ✅ on finish (or failed if it errors), and re-decompose/add tasks as new work is discovered.";

/// `PreToolUse` handler — block mutating tools when no live decomposed task
/// list can be observed for the session. Read tools / Task* tools always pass.
/// Fails closed when task state can't be read.
pub fn process(input: &HookInput, ctx: &HookContext<'_>) -> HookOutput {
    let evaluation = evaluate(input, ctx);
    output_from_evaluation(&evaluation)
}

#[must_use]
pub fn evaluate(input: &HookInput, ctx: &HookContext<'_>) -> TaskDecompositionEvaluation {
    let tool_name = input.tool_name.as_deref().unwrap_or("");
    let tool = input.tool_name.clone();
    let session_id = input.session_id.clone();
    let bash_tool = tool_name == "Bash";
    let bash_command = bash_tool.then(|| bash_command(input)).flatten();
    let bash_command_present = bash_command.is_some();

    // Master switch (operator config `enabled = false`): the whole gate stands
    // down. Report the tool as allowed so `graph_authority_required()` stays
    // false — no graph runs, so there is no replica to keep in parity.
    if !load_config(ctx.fs).enabled {
        eprintln!(
            "[sentinel] task_decomposition_gate: disabled by operator config \
             (task-decomposition-gate.toml enabled = false) — allowing without checks."
        );
        return TaskDecompositionEvaluation {
            tool,
            session_id,
            bash_command,
            allowed_tool: true,
            bash_tool,
            bash_command_present,
            mutating_tool: false,
            task_state_readable: false,
            task_list_confirmed: false,
            unreadable_task_state: false,
            should_block: false,
            decision: TaskDecompositionDecision::Allow,
        };
    }

    // Never gate read / task tools — they are the fix path.
    if is_allowed_tool(tool_name) {
        return TaskDecompositionEvaluation {
            tool,
            session_id,
            bash_command,
            allowed_tool: true,
            bash_tool,
            bash_command_present,
            mutating_tool: false,
            task_state_readable: false,
            task_list_confirmed: false,
            unreadable_task_state: false,
            should_block: false,
            decision: TaskDecompositionDecision::Allow,
        };
    }

    // Only gate mutating operations.
    let mutating_tool = is_mutating_tool(input, tool_name);
    if !mutating_tool {
        return TaskDecompositionEvaluation {
            tool,
            session_id,
            bash_command,
            allowed_tool: false,
            bash_tool,
            bash_command_present,
            mutating_tool,
            task_state_readable: false,
            task_list_confirmed: false,
            unreadable_task_state: false,
            should_block: false,
            decision: TaskDecompositionDecision::Allow,
        };
    }

    // Task-state check. Block ONLY on `Some(false)` — a session dir that EXISTS,
    // is readable, and positively shows no decomposition (no task files, no
    // highwatermark). `Some(true)` (live list / decomposed earlier) allows;
    // `None` (UNCONFIRMABLE — no session id / no home / MISSING resolved dir /
    // read error) FAILS OPEN. The resolver (`session_task_dir`) already tries
    // both harness naming forms, so a still-missing dir is a store/gate desync
    // we don't own, NOT skipped decomposition — failing closed on it bricks the
    // session with no fix path. The teeth that matter (an agent skipping
    // decomposition while the store is healthy) are preserved by the
    // present-but-empty `Some(false)` block.
    let state = has_live_task_list(ctx.fs, input.session_id.as_deref());
    let task_state_readable = state.is_some();
    let task_list_confirmed = state == Some(true);
    let unreadable_task_state = state.is_none();
    let should_block = state == Some(false);
    TaskDecompositionEvaluation {
        tool,
        session_id,
        bash_command,
        allowed_tool: false,
        bash_tool,
        bash_command_present,
        mutating_tool,
        task_state_readable,
        task_list_confirmed,
        unreadable_task_state,
        should_block,
        decision: if should_block {
            TaskDecompositionDecision::Block
        } else {
            TaskDecompositionDecision::Allow
        },
    }
}

#[must_use]
pub fn output_from_evaluation(evaluation: &TaskDecompositionEvaluation) -> HookOutput {
    if !matches!(evaluation.decision, TaskDecompositionDecision::Block) {
        return HookOutput::allow();
    }
    let tool_name = evaluation.tool.as_deref().unwrap_or("");
    // Both Some(false) and None block. Distinguish the message so an unreadable
    // store is diagnosable.
    let prefix = if evaluation.unreadable_task_state {
        "Task state could not be read (no session id, home dir, or readable \
         task store), so a decomposed task list cannot be confirmed. "
    } else {
        ""
    };
    let envelope = HookEnvelope::block(
        "Task Decomposition Gate",
        format!(
            "{prefix}No live decomposed task list found for this session, but you are \
             about to use a mutating tool (`{tool_name}`). {DECOMPOSITION_GUIDANCE}"
        ),
    );
    HookOutput::block(envelope.render())
}

#[cfg(test)]
mod tests {

    fn seed_gate_enabled(home: &std::path::Path) {
        let dir = home.join(".claude").join("sentinel").join("config");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("task-decomposition-gate.toml"),
            "enabled = true
",
        )
        .unwrap();
    }
    use super::*;
    use crate::hooks::test_support::{stub_ctx, StubFs};
    use std::path::{Path, PathBuf};

    /// FS that reports a caller-supplied home dir so tests can isolate
    /// `~/.claude/tasks/`.
    struct ScopedHomeFs {
        home: PathBuf,
    }
    impl FileSystemPort for ScopedHomeFs {
        fn home_dir(&self) -> Option<PathBuf> {
            Some(self.home.clone())
        }
        fn read_to_string(
            &self,
            p: &Path,
        ) -> Result<String, sentinel_domain::port_errors::FileSystemError> {
            Ok(std::fs::read_to_string(p)?)
        }
        fn write(
            &self,
            p: &Path,
            c: &[u8],
        ) -> Result<(), sentinel_domain::port_errors::FileSystemError> {
            if let Some(par) = p.parent() {
                std::fs::create_dir_all(par)?;
            }
            Ok(std::fs::write(p, c)?)
        }
        fn create_dir_all(
            &self,
            p: &Path,
        ) -> Result<(), sentinel_domain::port_errors::FileSystemError> {
            Ok(std::fs::create_dir_all(p)?)
        }
        fn read_dir(
            &self,
            p: &Path,
        ) -> Result<Vec<PathBuf>, sentinel_domain::port_errors::FileSystemError> {
            Ok(std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .collect())
        }
        fn exists(&self, p: &Path) -> bool {
            p.exists()
        }
        fn is_dir(&self, p: &Path) -> bool {
            p.is_dir()
        }
        fn metadata(
            &self,
            p: &Path,
        ) -> Result<std::fs::Metadata, sentinel_domain::port_errors::FileSystemError> {
            Ok(std::fs::metadata(p)?)
        }
        fn append(
            &self,
            _: &Path,
            _: &[u8],
        ) -> Result<(), sentinel_domain::port_errors::FileSystemError> {
            Ok(())
        }
    }

    /// Build a `HookContext` whose FS is the supplied scoped-home adapter.
    fn ctx_with_fs(fs: &'static ScopedHomeFs) -> HookContext<'static> {
        let base = stub_ctx();
        HookContext {
            git: base.git,
            vector_store: None,
            fs,
            process: base.process,
            llm: None,
            memory_mcp: base.memory_mcp,
            env: base.env,
            linear_lookup: None,
        }
    }

    /// Seed `~/.claude/tasks/{session}/1.json` with a task of the given status.
    fn seed_task(home: &Path, session: &str, status: &str) {
        let dir = home.join(".claude").join("tasks").join(session);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("1.json"),
            format!(
                r#"{{"id":"1","subject":"do thing","status":"{status}","blocks":[],"blockedBy":[]}}"#
            ),
        )
        .unwrap();
    }

    /// Seed an empty session dir holding only a `.highwatermark` of `value`,
    /// emulating a session that decomposed tasks which have since completed
    /// (their `*.json` files pruned, the monotonic counter left behind).
    fn seed_highwatermark(home: &Path, session: &str, value: &str) {
        let dir = home.join(".claude").join("tasks").join(session);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".highwatermark"), value).unwrap();
    }

    /// Create a PRESENT but EMPTY session task dir (no task files, no
    /// highwatermark) — the genuine "session never decomposed" case that must
    /// still BLOCK, as opposed to a MISSING dir (which now fails open).
    fn seed_empty_dir(home: &Path, session: &str) {
        let dir = home.join(".claude").join("tasks").join(session);
        std::fs::create_dir_all(&dir).unwrap();
    }

    fn input(tool: &str, session: Option<&str>) -> HookInput {
        HookInput {
            tool_name: Some(tool.to_string()),
            session_id: session.map(str::to_string),
            ..Default::default()
        }
    }

    fn bash_input(command: &str, session: Option<&str>) -> HookInput {
        HookInput {
            tool_name: Some("Bash".to_string()),
            session_id: session.map(str::to_string),
            tool_input: Some(serde_json::json!({ "command": command })),
            ..Default::default()
        }
    }

    // --- classification unit tests --------------------------------------

    #[test]
    fn read_tools_are_allowed() {
        for t in [
            "Read",
            "Glob",
            "Grep",
            "LS",
            "TaskList",
            "TaskGet",
            "NotebookRead",
            "WebFetch",
            "WebSearch",
        ] {
            assert!(is_allowed_tool(t), "{t} must be allowed");
        }
    }

    #[test]
    fn task_create_and_update_are_allowed() {
        assert!(is_allowed_tool("TaskCreate"));
        assert!(is_allowed_tool("TaskUpdate"));
    }

    #[test]
    fn sequential_thinking_namespace_is_allowed() {
        assert!(is_allowed_tool(
            "mcp__sequential-thinking__sequentialthinking"
        ));
    }

    #[test]
    fn edit_write_are_mutating() {
        assert!(is_mutating_tool(&input("Edit", None), "Edit"));
        assert!(is_mutating_tool(&input("Write", None), "Write"));
        assert!(is_mutating_tool(
            &input("NotebookEdit", None),
            "NotebookEdit"
        ));
    }

    #[test]
    fn read_only_bash_is_not_mutating() {
        for c in [
            "ls -la",
            "cat README.md",
            "grep foo bar.rs",
            "git status",
            "git log --oneline",
            "git diff HEAD",
            "find . -name '*.rs'",
        ] {
            assert!(
                !is_mutating_bash(c),
                "read-only bash should not be mutating: {c}"
            );
        }
    }

    #[test]
    fn stderr_redirections_are_not_mutating() {
        // The common stderr forms on read-only commands must NOT be flagged as
        // writes — this was the bug that blocked `ls 2>&1`, `find … 2>/dev/null`.
        for c in [
            "ls -la 2>&1",
            "find . -name '*.rs' 2>/dev/null",
            "grep foo bar.rs 2>&1 | head",
            "cat f 2>>errs.log",    // stderr append to a sink — not tracked state
            "command -v tool 2>&-", // close stderr
            "node -e 'x' 2>&1",
        ] {
            assert!(
                !is_mutating_bash(c),
                "stderr redirection must not be mutating: {c}"
            );
        }
    }

    #[test]
    fn stdout_redirections_are_still_mutating() {
        // A real stdout/file write must still be caught even when a stderr
        // redirection is also present.
        for c in [
            "echo hi > out.txt",
            "cat a >> b",
            "make 2>&1 > build.log", // stdout file write alongside stderr dup
            "tool > result 2>/dev/null",
        ] {
            assert!(is_mutating_bash(c), "stdout write must be mutating: {c}");
        }
    }

    #[test]
    fn mutating_bash_is_detected() {
        for c in [
            "git commit -m x",
            "git push origin main",
            "git merge feat --no-edit",
            "git reset --hard",
            "rm -rf build",
            "mv a b",
            "cp src dst",
            "cargo build --release",
            "npm install",
            "sed -i 's/a/b/' f",
            "echo hi > out.txt",
            "cat a >> b",
        ] {
            assert!(is_mutating_bash(c), "should be mutating: {c}");
        }
    }

    // --- process() integration tests ------------------------------------

    #[test]
    fn read_tool_allows_even_without_tasks() {
        let ctx = stub_ctx();
        let out = process(&input("Read", Some("sess-1")), &ctx);
        assert!(out.blocked.is_none(), "Read must never be blocked");
    }

    #[test]
    fn task_create_allows_even_without_tasks() {
        let ctx = stub_ctx();
        let out = process(&input("TaskCreate", Some("sess-1")), &ctx);
        assert!(out.blocked.is_none(), "TaskCreate must never be blocked");
    }

    #[test]
    fn edit_with_no_task_list_blocks() {
        let tmp = tempfile::tempdir().unwrap();
        seed_gate_enabled(tmp.path());
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        seed_empty_dir(tmp.path(), "sess-empty");
        let ctx = ctx_with_fs(fs);
        // Dir PRESENT but EMPTY → Some(false) → block (genuine non-decomposition).
        let out = process(&input("Edit", Some("sess-empty")), &ctx);
        assert_eq!(
            out.blocked,
            Some(true),
            "Edit with a present-but-empty task dir must block"
        );
        assert!(out
            .reason
            .as_ref()
            .is_some_and(|r| r.contains("decomposed task list")));
    }

    #[test]
    fn operator_config_enabled_false_disables_whole_gate() {
        // Same present-but-empty task dir that blocks above — but the operator
        // override disables the gate, so the edit is allowed and no graph
        // authority is required.
        let tmp = tempfile::tempdir().unwrap();
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        seed_empty_dir(tmp.path(), "sess-empty");
        let cfg_dir = tmp.path().join(".claude").join("sentinel").join("config");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(
            cfg_dir.join("task-decomposition-gate.toml"),
            "enabled = false\n",
        )
        .unwrap();
        let ctx = ctx_with_fs(fs);
        let evaluation = evaluate(&input("Edit", Some("sess-empty")), &ctx);
        assert!(!evaluation.should_block, "disabled gate must not block");
        assert!(
            !evaluation.graph_authority_required(),
            "disabled gate must not consult the graph authority"
        );
        let out = process(&input("Edit", Some("sess-empty")), &ctx);
        assert!(out.blocked.is_none(), "disabled gate must allow: {out:?}");
    }

    #[test]
    fn edit_with_live_task_list_allows() {
        let tmp = tempfile::tempdir().unwrap();
        seed_task(tmp.path(), "sess-live", "in_progress");
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        let out = process(&input("Edit", Some("sess-live")), &ctx);
        assert!(
            out.blocked.is_none(),
            "Edit with a live task list must be allowed"
        );
    }

    #[test]
    fn read_only_bash_allows_without_tasks() {
        let tmp = tempfile::tempdir().unwrap();
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        let out = process(&bash_input("git status", Some("sess-empty")), &ctx);
        assert!(
            out.blocked.is_none(),
            "read-only bash must be allowed even with no tasks"
        );
    }

    #[test]
    fn mutating_bash_with_no_tasks_blocks() {
        let tmp = tempfile::tempdir().unwrap();
        seed_gate_enabled(tmp.path());
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        seed_empty_dir(tmp.path(), "sess-empty");
        let ctx = ctx_with_fs(fs);
        let out = process(&bash_input("git commit -m wip", Some("sess-empty")), &ctx);
        assert_eq!(
            out.blocked,
            Some(true),
            "mutating bash with a present-but-empty task dir must block"
        );
    }

    #[test]
    fn mutating_bash_with_live_tasks_allows() {
        let tmp = tempfile::tempdir().unwrap();
        seed_task(tmp.path(), "sess-live", "pending");
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        let out = process(&bash_input("cargo build", Some("sess-live")), &ctx);
        assert!(
            out.blocked.is_none(),
            "mutating bash with a live task list must be allowed"
        );
    }

    #[test]
    fn fails_open_when_no_session_id() {
        // No session id → has_live_task_list returns None (UNCONFIRMABLE) → the
        // gate must FAIL OPEN (allow). The gate must never brick a session over
        // state it cannot read; only a present-but-empty dir (`Some(false)`)
        // blocks. An unreadable/absent task store is not evidence of skipped
        // decomposition.
        let ctx = stub_ctx();
        let out = process(&input("Edit", None), &ctx);
        assert!(
            out.blocked.is_none(),
            "missing session id must FAIL OPEN (allow) — never brick on unconfirmable state"
        );
    }

    #[test]
    fn edit_with_missing_task_dir_allows() {
        // THE deadlock case: session id present, but the resolved task dir does
        // NOT exist (native store wrote nowhere we can see). has_live_task_list
        // → None → fail OPEN. Must NOT block, or the session is bricked with no
        // fix path (TaskCreate can't create a dir the gate would see).
        let tmp = tempfile::tempdir().unwrap();
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        // No dir seeded for "sess-missing" → missing → None → allow.
        let out = process(&input("Edit", Some("sess-missing")), &ctx);
        assert!(
            out.blocked.is_none(),
            "Edit with a MISSING task dir must fail OPEN (allow) — never deadlock"
        );
    }

    #[test]
    fn has_live_task_list_none_for_missing_dir() {
        // Unit-level: a session whose dir was never created resolves to None
        // (unconfirmable), not Some(false) (definitive non-decomposition).
        let tmp = tempfile::tempdir().unwrap();
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert!(has_live_task_list(&fs, Some("never-created")).is_none());
    }

    #[test]
    fn has_live_task_list_none_without_session() {
        let fs = StubFs;
        assert!(has_live_task_list(&fs, None).is_none());
        assert!(has_live_task_list(&fs, Some("")).is_none());
    }

    #[test]
    fn has_live_task_list_false_for_empty_session_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join(".claude").join("tasks").join("sess-x");
        std::fs::create_dir_all(&session_dir).unwrap();
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert_eq!(has_live_task_list(&fs, Some("sess-x")), Some(false));
    }

    #[test]
    fn has_live_task_list_true_with_task_file() {
        let tmp = tempfile::tempdir().unwrap();
        seed_task(tmp.path(), "sess-y", "pending");
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert_eq!(has_live_task_list(&fs, Some("sess-y")), Some(true));
    }

    #[test]
    fn has_live_task_list_true_when_highwatermark_set_but_no_task_files() {
        // Session decomposed tasks earlier; they completed and their files were
        // pruned, leaving only the monotonic `.highwatermark`. Must NOT re-block.
        let tmp = tempfile::tempdir().unwrap();
        seed_highwatermark(tmp.path(), "sess-done", "4");
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert_eq!(has_live_task_list(&fs, Some("sess-done")), Some(true));
    }

    #[test]
    fn has_live_task_list_true_for_session_prefixed_dir() {
        // The harness wrote tasks under `session-{first8}/` but the hook receives
        // the full UUID. The resolver must find the prefixed dir. This is the
        // exact bug that made the gate fail closed and block every mutating tool.
        let tmp = tempfile::tempdir().unwrap();
        let full_uuid = "e2ea5630-3c79-409c-9ca4-423975a5a5fb";
        seed_task(tmp.path(), "session-e2ea5630", "in_progress");
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert_eq!(
            has_live_task_list(&fs, Some(full_uuid)),
            Some(true),
            "must resolve the session-{{first8}} dir from the full-UUID session id"
        );
    }

    #[test]
    fn edit_allowed_when_tasks_under_session_prefixed_dir() {
        // End-to-end: mutating tool must be ALLOWED when the live task list lives
        // under the `session-{first8}` naming the harness chose for this session.
        let tmp = tempfile::tempdir().unwrap();
        seed_task(tmp.path(), "session-dac1d29b", "pending");
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        let out = process(
            &input("Edit", Some("dac1d29b-1111-2222-3333-444455556666")),
            &ctx,
        );
        assert!(
            out.blocked.is_none(),
            "Edit must be allowed when tasks live under session-{{first8}}"
        );
    }

    #[test]
    fn has_live_task_list_false_when_highwatermark_zero() {
        // A `0` highwatermark means no task id was ever issued → still gated.
        let tmp = tempfile::tempdir().unwrap();
        seed_highwatermark(tmp.path(), "sess-zero", "0");
        let fs = ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        };
        assert_eq!(has_live_task_list(&fs, Some("sess-zero")), Some(false));
    }

    #[test]
    fn edit_after_task_completion_allows() {
        // Integration: only a `.highwatermark` remains (all tasks completed).
        // A follow-up mutating tool must be allowed — no re-block-after-completion.
        let tmp = tempfile::tempdir().unwrap();
        seed_highwatermark(tmp.path(), "sess-done", "7");
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        let ctx = ctx_with_fs(fs);
        let out = process(&input("Edit", Some("sess-done")), &ctx);
        assert!(
            out.blocked.is_none(),
            "Edit after completing all tasks must be allowed (highwatermark proves prior decomposition)"
        );
    }

    #[test]
    fn block_message_contains_convention_markers() {
        let tmp = tempfile::tempdir().unwrap();
        seed_gate_enabled(tmp.path());
        let fs: &'static ScopedHomeFs = Box::leak(Box::new(ScopedHomeFs {
            home: tmp.path().to_path_buf(),
        }));
        seed_empty_dir(tmp.path(), "sess-empty");
        let ctx = ctx_with_fs(fs);
        let out = process(&input("Write", Some("sess-empty")), &ctx);
        let reason = out.reason.expect("block reason present");
        assert!(reason.contains("DECOMPOSITION"));
        assert!(reason.contains("PRIORITY"));
        assert!(reason.contains("SUBTASKS"));
        assert!(reason.contains("Na"));
        assert!(reason.contains("addBlockedBy"));
        assert!(reason.contains("MIRROR CHECKLIST"));
        assert!(reason.contains("STATUS EMOJI"));
        assert!(reason.contains("TWO-EMOJI"));
        assert!(reason.contains('⬜'));
        assert!(reason.contains('🔄'));
        assert!(reason.contains('✅'));
        assert!(reason.contains('❌'));
        assert!(reason.contains("pending|in_progress|completed|failed"));
    }
}
