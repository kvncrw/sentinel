//! Task Enforcement Gate — `PreToolUse` hard gate on code mutations.
//!
//! Tasks are the operator's lifeblood: work should be tracked. This gate
//! **blocks** the code-mutating tools (`Edit` / `Write` / `NotebookEdit`) when
//! the session has no `in_progress` task, so edits can't happen off-book. But
//! it is a *helpful* block, not a dead end — the deny message tells the agent
//! exactly how to proceed: create + start a task for this work (and offer the
//! operator the choice of "create it for you" vs "you create it"). Enforce
//! hygiene, stay supportive.
//!
//! ## When it blocks
//! Only when ALL of these hold, so it never surprise-bricks a session:
//! - the tool is `Edit`, `Write`, or `NotebookEdit` (other tools always pass —
//!   crucially `TaskCreate`/`TaskUpdate` are never gated, so the unblock path
//!   create→start→retry always works);
//! - the cwd is inside a git repo (edits outside any repo — `~/.claude`
//!   config, the scratchpad, memory files — are meta-work and always allowed);
//! - the session's task dir is resolvable and readable;
//! - there is **no** `in_progress` task.
//!
//! ## Fail-open, always
//! Any uncertainty resolves to ALLOW: no concrete session id, unresolved /
//! missing / unreadable task dir, or not in a repo → allow. A gate that fails
//! closed on a read glitch would brick the session; this one never does.
//!
//! ## Master switch
//! `task-enforcement-gate.toml` `enabled` toggles the gate (operator override at
//! `~/.claude/sentinel/config/`). Shipped default is `false` — off by default,
//! same posture as the process gates (PR #32), since this is a hard mutation-
//! block that can deadlock headless agents. Same config-file lever as the other
//! gates — no hidden env setting. A corrupt/unparseable config falls back to
//! enabled (fail closed to enforcement).

use std::path::Path;

use sentinel_domain::events::{HookInput, HookOutput};

use super::{concrete_input_session_id, session_task_dir, FileSystemPort, HookContext};

/// Tools this gate blocks when no task is in progress — the code mutators.
const GATED_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit"];

/// Shipped defaults, compiled in. Operator override lives at
/// `~/.claude/sentinel/config/task-enforcement-gate.toml` (same schema).
const SHIPPED_DEFAULTS: &str =
    include_str!("../../../../config/task-enforcement-gate-defaults.toml");

/// Gate config — the master switch, mirroring `task_decomposition_gate` and
/// `tool_usage_gate`. Replaces the former `SENTINEL_TASK_GATE` env kill-switch
/// so this gate is toggled through the same config-file lever as every other
/// gate, with no hidden env setting.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskEnforcementGateConfig {
    /// Master switch. `false` disables the whole gate: mutating tools are never
    /// blocked for a missing in_progress task. Missing/corrupt config falls
    /// back to `true` — fail closed to enforcement, same contract as the other
    /// gates.
    #[serde(default = "default_true")]
    enabled: bool,
}

const fn default_true() -> bool {
    true
}

impl Default for TaskEnforcementGateConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl TaskEnforcementGateConfig {
    fn from_toml_or_default(s: &str) -> Self {
        toml::from_str(s).unwrap_or_else(|e| {
            eprintln!(
                "[sentinel] task_enforcement_gate: config TOML parse failed ({e}); \
                 using enforced defaults"
            );
            Self::default()
        })
    }
}

/// Load config: shipped defaults, then the operator override file if present.
fn load_config(fs: &dyn FileSystemPort) -> TaskEnforcementGateConfig {
    let mut cfg = TaskEnforcementGateConfig::from_toml_or_default(SHIPPED_DEFAULTS);
    let path = fs
        .claude_dir()
        .join("sentinel")
        .join("config")
        .join("task-enforcement-gate.toml");
    if let Ok(content) = fs.read_to_string(&path) {
        cfg = TaskEnforcementGateConfig::from_toml_or_default(&content);
    }
    cfg
}

/// A native task row — only the status matters here.
#[derive(Debug, serde::Deserialize)]
struct TaskStatusRow {
    #[serde(default)]
    status: String,
}

/// The gate's decision, with enough context for tests and structured output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Allowed — not a gated tool, not in a repo, gate off, dir unreadable, or
    /// an in_progress task exists.
    Allow,
    /// Blocked — a gated code mutation with no in_progress task.
    Block { tool: String },
}

/// Count the in_progress / total task files in `dir`. Returns `(in_progress,
/// total_task_files)`. Unreadable dir or files are skipped, not fatal.
fn scan_tasks(fs: &dyn FileSystemPort, dir: &Path) -> (usize, usize) {
    let Ok(entries) = fs.read_dir(dir) else {
        return (0, 0);
    };
    let mut in_progress = 0;
    let mut total = 0;
    for path in entries {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name.starts_with('.')
            || !Path::new(&name)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
            continue;
        }
        let Ok(content) = fs.read_to_string(&path) else {
            continue;
        };
        let Ok(row) = serde_json::from_str::<TaskStatusRow>(&content) else {
            continue;
        };
        total += 1;
        if row.status == "in_progress" {
            in_progress += 1;
        }
    }
    (in_progress, total)
}

/// Decide whether to gate this tool call. Pure over the injected `fs`/`git`.
#[must_use]
pub fn evaluate(input: &HookInput, ctx: &HookContext<'_>) -> GateDecision {
    // Master switch (operator config `enabled = false`): gate stands down.
    if !load_config(ctx.fs).enabled {
        return GateDecision::Allow;
    }

    // Only gate the code-mutating tools.
    let tool_name = input.tool_name.as_deref().unwrap_or("");
    if !GATED_TOOLS.contains(&tool_name) {
        return GateDecision::Allow;
    }

    // Only gate edits inside a git repo — meta-work (config, scratchpad,
    // memory) outside any repo is never gated.
    let cwd = input.cwd.as_deref().unwrap_or(".");
    if ctx.git.repo_root(cwd).is_none() {
        return GateDecision::Allow;
    }

    // Resolve this session's task dir. Any uncertainty → allow (fail-open).
    let Some(session_id) = concrete_input_session_id(input) else {
        return GateDecision::Allow;
    };
    let Some(home) = ctx.fs.home_dir() else {
        return GateDecision::Allow;
    };
    let dir = session_task_dir(ctx.fs, &home, session_id);
    if !ctx.fs.is_dir(&dir) {
        return GateDecision::Allow;
    }

    let (in_progress, _total) = scan_tasks(ctx.fs, &dir);
    if in_progress > 0 {
        GateDecision::Allow
    } else {
        GateDecision::Block {
            tool: tool_name.to_string(),
        }
    }
}

/// The helpful deny message — enforces the gate but hands the agent a clear,
/// supportive path forward.
fn block_message(tool: &str) -> String {
    format!(
        "No task is in progress, so `{tool}` is blocked — work must be tracked. \
         Before editing: create a task for this work (`TaskCreate`) and mark it \
         `in_progress` (`TaskUpdate`), then retry. If it isn't obvious what the \
         task should be, ASK the operator whether they'd like you to create and \
         start one for them, or prefer to do it themselves — enforce the hygiene, \
         but be helpful about it. (`TaskCreate`/`TaskUpdate` are never gated. To \
         disable this gate set `enabled = false` in task-enforcement-gate.toml.)"
    )
}

/// `PreToolUse` handler. Uses `HookOutput::deny` so the block is a real
/// platform-enforced `permissionDecision: Deny` AND carries the
/// `[Sentinel-Authority]` provenance prefix (the agent's only signal that the
/// directive is a trusted sentinel deny, not arbitrary tool-result text).
#[must_use]
pub fn process(input: &HookInput, ctx: &HookContext<'_>) -> HookOutput {
    match evaluate(input, ctx) {
        GateDecision::Allow => HookOutput::allow(),
        GateDecision::Block { tool } => HookOutput::deny(block_message(&tool)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::test_support::{stub_ctx_with_fs, TestHomeFs};
    use sentinel_domain::port_errors::GitError;

    /// A git port that reports a specific repo root — the default `StubGit`
    /// returns `None`, which would make the gate always fail-open. Only
    /// `repo_root` is exercised; every other method is a benign stub.
    struct RepoGit {
        root: String,
    }
    impl super::super::GitStatusPort for RepoGit {
        fn has_uncommitted_changes(&self, _: &str) -> Result<bool, GitError> {
            Ok(false)
        }
        fn changed_files(&self, _: &str) -> Result<Vec<String>, GitError> {
            Ok(vec![])
        }
        fn current_branch(&self, _: &str) -> Result<String, GitError> {
            Ok("main".into())
        }
        fn is_worktree(&self, _: &str) -> bool {
            false
        }
        fn has_unpushed_commits(&self, _: &str) -> Result<bool, GitError> {
            Ok(false)
        }
        fn repo_root(&self, _: &str) -> Option<String> {
            Some(self.root.clone())
        }
        fn list_worktree_names(&self, _: &str) -> Vec<String> {
            Vec::new()
        }
        fn merge_base(&self, _: &str, _: &str) -> Option<String> {
            None
        }
        fn rev_list_count(&self, _: &str, _: &str) -> Option<u32> {
            None
        }
        fn rev_list_count_range(&self, _: &str, _: &str) -> Option<u32> {
            None
        }
        fn diff_names(&self, _: &str, _: &str) -> Option<Vec<String>> {
            None
        }
        fn merged_local_branches(&self, _: &str, _: &str) -> Vec<String> {
            Vec::new()
        }
        fn merged_remote_branches(&self, _: &str, _: &str) -> Vec<String> {
            Vec::new()
        }
        fn head_sha(&self, _: &str) -> Option<String> {
            None
        }
    }

    fn write_task(dir: &Path, file: &str, status: &str) {
        std::fs::write(
            dir.join(file),
            format!(r#"{{"id":"1","subject":"x","status":"{status}"}}"#),
        )
        .unwrap();
    }

    /// Write an operator override enabling the gate. The shipped default is OFF
    /// (off-by-default, PR #32 posture), so tests that exercise the ACTIVE gate
    /// must opt it in — mirrors `task_decomposition_gate::seed_gate_enabled`.
    fn seed_gate_enabled(home: &Path) {
        let dir = home.join(".claude").join("sentinel").join("config");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("task-enforcement-gate.toml"), "enabled = true\n").unwrap();
    }

    const SID: &str = "gate-sess-1";

    /// Create a tempdir home, seed a `session-<first8>` task dir with the given
    /// status files, and return the tempdir.
    fn setup(status_files: &[(&str, &str)]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp
            .path()
            .join(".claude")
            .join("tasks")
            .join(format!("session-{}", &SID[..8.min(SID.len())]));
        std::fs::create_dir_all(&dir).unwrap();
        for (f, s) in status_files {
            write_task(&dir, f, s);
        }
        // Gate ships OFF by default; the active-gate tests want it enabled.
        // (operator_config_enabled_false_disables_gate overwrites this after.)
        seed_gate_enabled(tmp.path());
        tmp
    }

    fn input_for(tool: &str, cwd: &Path, sid: Option<&str>) -> HookInput {
        HookInput {
            tool_name: Some(tool.to_string()),
            cwd: Some(cwd.to_string_lossy().to_string()),
            session_id: sid.map(str::to_string),
            ..Default::default()
        }
    }

    /// Build a HookContext with a real-disk fs rooted at `home` and a git port
    /// reporting `repo` as the repo root (or the default no-repo git when
    /// `repo` is None).
    fn ctx_with<'a>(
        fs: &'a dyn FileSystemPort,
        git: &'a dyn super::super::GitStatusPort,
    ) -> HookContext<'a> {
        let mut c = stub_ctx_with_fs(fs);
        c.git = git;
        c
    }

    #[test]
    fn blocks_edit_when_no_in_progress_task() {
        let tmp = setup(&[("1.json", "pending"), ("2.json", "completed")]);
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        let input = input_for("Edit", tmp.path(), Some(SID));

        assert_eq!(
            evaluate(&input, &ctx),
            GateDecision::Block {
                tool: "Edit".into()
            },
            "tasks exist but none in_progress → block"
        );
        let out = process(&input, &ctx);
        // deny() puts the tagged reason in hookSpecificOutput.permission_decision_reason.
        let reason = out
            .hook_specific_output
            .and_then(|h| h.permission_decision_reason)
            .expect("deny reason present");
        assert!(
            reason.starts_with("[Sentinel-Authority] "),
            "authority prefix: {reason}"
        );
        assert!(reason.contains("TaskCreate"), "actionable: {reason}");
        assert!(reason.contains("ASK the operator"), "supportive: {reason}");
    }

    #[test]
    fn allows_edit_when_a_task_is_in_progress() {
        let tmp = setup(&[("1.json", "in_progress")]);
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        let input = input_for("Write", tmp.path(), Some(SID));
        assert_eq!(evaluate(&input, &ctx), GateDecision::Allow);
    }

    #[test]
    fn operator_config_enabled_false_disables_gate() {
        // Same present-but-no-in_progress task dir that blocks above — but the
        // operator override disables the gate via the config file, so the edit
        // is allowed. Replaces the former SENTINEL_TASK_GATE env kill-switch.
        let tmp = setup(&[("1.json", "pending")]);
        let cfg_dir = tmp.path().join(".claude").join("sentinel").join("config");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(cfg_dir.join("task-enforcement-gate.toml"), "enabled = false\n").unwrap();
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        let input = input_for("Edit", tmp.path(), Some(SID));
        assert_eq!(
            evaluate(&input, &ctx),
            GateDecision::Allow,
            "enabled = false → allow"
        );
    }

    #[test]
    fn non_mutating_tools_always_allowed() {
        let tmp = setup(&[("1.json", "pending")]);
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        for tool in ["Read", "TaskCreate", "TaskUpdate", "Bash", "Grep"] {
            let input = input_for(tool, tmp.path(), Some(SID));
            assert_eq!(evaluate(&input, &ctx), GateDecision::Allow, "{tool}");
        }
    }

    #[test]
    fn edits_outside_a_repo_are_allowed() {
        let tmp = setup(&[("1.json", "pending")]);
        let fs = TestHomeFs::new(tmp.path());
        // Default StubGit → repo_root None → not in a repo → allow.
        let ctx = stub_ctx_with_fs(&fs);
        let input = input_for("Edit", tmp.path(), Some(SID));
        assert_eq!(
            evaluate(&input, &ctx),
            GateDecision::Allow,
            "non-repo → allow"
        );
    }

    #[test]
    fn missing_task_dir_fails_open() {
        // Home tmp but NO task dir created → is_dir false → allow.
        let tmp = tempfile::tempdir().unwrap();
        seed_gate_enabled(tmp.path()); // active gate, so the allow is fail-open not off
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        let input = input_for("Edit", tmp.path(), Some("no-such-sess"));
        assert_eq!(
            evaluate(&input, &ctx),
            GateDecision::Allow,
            "no dir → allow"
        );
    }

    #[test]
    fn no_session_id_fails_open() {
        let tmp = setup(&[("1.json", "pending")]);
        let fs = TestHomeFs::new(tmp.path());
        let git = RepoGit {
            root: tmp.path().to_string_lossy().to_string(),
        };
        let ctx = ctx_with(&fs, &git);
        let input = input_for("Edit", tmp.path(), None);
        assert_eq!(
            evaluate(&input, &ctx),
            GateDecision::Allow,
            "no sid → allow"
        );
    }

    #[test]
    fn shipped_default_disables_the_gate() {
        // The compiled-in shipped default keeps the gate OFF — off-by-default,
        // same posture as the process gates (PR #32). An operator opts in with
        // enabled = true.
        assert!(
            !TaskEnforcementGateConfig::from_toml_or_default(SHIPPED_DEFAULTS).enabled,
            "shipped default must be disabled (off by default)"
        );
    }

    #[test]
    fn corrupt_config_fails_closed_to_enabled() {
        // A garbage override must fall back to enforcement, not silently disable.
        assert!(
            TaskEnforcementGateConfig::from_toml_or_default("not =[ valid toml").enabled,
            "corrupt config must fail closed to enabled"
        );
        // An explicit disable still parses and disables.
        assert!(
            !TaskEnforcementGateConfig::from_toml_or_default("enabled = false").enabled,
            "explicit enabled=false must disable"
        );
    }
}
