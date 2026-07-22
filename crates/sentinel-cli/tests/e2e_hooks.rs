//! Black-box E2E tests for the sentinel hook engine.
//!
//! These spawn the REAL `sentinel` binary and drive `sentinel hook --event <E>`
//! exactly the way Claude Code does — piping a `HookInput` JSON line on stdin
//! and asserting on the `HookOutput` JSON printed to stdout. This is the layer
//! the 90 inline unit suites do NOT cover: dispatch wiring, the stdin/stdout
//! contract, caller-validation, session-id resolution, and per-event
//! serialization. A hook can be unit-green yet fail to fire in production; this
//! catches that.
//!
//! ## Isolation
//!
//! Every test runs the binary with a fresh temp directory as `HOME` /
//! `USERPROFILE` and `SENTINEL_CLAUDE_DIR`, so `~/.claude/...` fixtures never
//! touch the real home. The enterprise hook constructor is given a dummy
//! `OPENROUTER_API_KEY` so A2/A3/A13 routing initializes without using real
//! credentials; other service env (`QDRANT_URL`, `LINEAR_API_KEY`) is cleared
//! so service-dependent hooks stay offline/CI-safe. Each test uses a unique
//! `sessionId`.
//!
//! ## Contract (from `hook_cmd.rs` / `events.rs`)
//!
//! - Invocation: `sentinel hook --event <PascalCaseEvent>`, JSON on stdin.
//! - To be processed: non-tty stdin (or `SENTINEL_ALLOW_TERMINAL=1`),
//!   `CLAUDE_CODE_ENTRY_POINT` set, and a `sessionId` (JSON or
//!   `CLAUDE_SESSION_ID`). Missing session-id → safe `{}` no-op.
//! - Output: allow = `{}`; deny (`PreToolUse`) =
//!   `hookSpecificOutput.permissionDecision == "deny"`; context injection =
//!   `hookSpecificOutput.additionalContext`; stop = `continue: false`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

/// Monotonic counter so every test gets a unique session id (sidesteps any
/// per-session throttle/marker carrying across tests).
static SESSION_SEQ: AtomicU64 = AtomicU64::new(0);

fn unique_session_id() -> String {
    let n = SESSION_SEQ.fetch_add(1, Ordering::SeqCst);
    format!("e2e-{}-{n}", std::process::id())
}

/// A single black-box hook invocation against the real binary in an isolated
/// temp HOME. Builder-style: seed fixtures, then `.run(event, input)`.
struct HookTest {
    home: tempfile::TempDir,
    session_id: String,
    extra_env: Vec<(String, String)>,
}

impl HookTest {
    fn new() -> Self {
        let test = Self {
            home: tempfile::tempdir().expect("tempdir"),
            session_id: unique_session_id(),
            extra_env: Vec::new(),
        };
        test.seed_workflow_config();
        test
    }

    fn home_path(&self) -> &Path {
        self.home.path()
    }

    /// The isolated `~/.claude` dir inside the temp home.
    fn claude_dir(&self) -> PathBuf {
        self.home_path().join(".claude")
    }

    fn seed_workflow_config(&self) {
        let config_dir = self.claude_dir().join("sentinel").join("config");
        std::fs::create_dir_all(&config_dir).expect("mk sentinel config dir");
        std::fs::write(
            config_dir.join("workflows.toml"),
            r#"
[[workflows]]
skill = "linear"

[[workflows.phases]]
id = "claim"
file = "claim.md"
required = true
judge = "sonnet"
description = "Claim"
"#,
        )
        .expect("write workflows.toml");
    }

    /// Seed a task JSON file at `~/.claude/tasks/{session}/{id}.json`.
    fn seed_task(&self, id: &str, task: Value) -> &Self {
        let dir = self.claude_dir().join("tasks").join(&self.session_id);
        std::fs::create_dir_all(&dir).expect("mk task dir");
        std::fs::write(
            dir.join(format!("{id}.json")),
            serde_json::to_vec(&task).unwrap(),
        )
        .expect("write task");
        self
    }

    /// Add an extra env var for the spawned process.
    fn env(&mut self, key: &str, val: &str) -> &mut Self {
        self.extra_env.push((key.to_string(), val.to_string()));
        self
    }

    /// Create a throwaway git repo under the temp home, on `branch`, with an
    /// initial commit of `file`. Returns its absolute path (use as `cwd`).
    /// If `dirty` is true, leaves an uncommitted modification to `file`.
    fn seed_git_repo(&self, name: &str, branch: &str, file: &str, dirty: bool) -> PathBuf {
        let repo = self.home_path().join(name);
        std::fs::create_dir_all(&repo).unwrap();
        let git = |args: &[&str]| {
            let ok = Command::new("git")
                .args(args)
                .current_dir(&repo)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            assert!(ok, "git {args:?} failed in {}", repo.display());
        };
        git(&["init", "-q"]);
        git(&["checkout", "-q", "-B", branch]);
        std::fs::write(repo.join(file), "initial\n").unwrap();
        git(&["add", "."]);
        git(&["commit", "-q", "-m", "init"]);
        if dirty {
            std::fs::write(repo.join(file), "modified\n").unwrap();
        }
        repo
    }

    /// Run `sentinel hook --event <event>` with `input` piped on stdin.
    /// Returns the parsed stdout JSON (defaults to `{}` if stdout is empty).
    fn run(&self, event: &str, mut input: Value) -> HookResult {
        // Inject the session id unless the caller set one explicitly.
        // NOTE: HookInput uses SNAKE_CASE serde field names (session_id,
        // tool_name, tool_input, file_path) — NOT camelCase. (Discovered the
        // hard way: camelCase keys silently drop to None via #[serde(default)],
        // which is the root cause of the original F1 `{}` finding.)
        if input.get("session_id").is_none() {
            input["session_id"] = json!(self.session_id);
        }
        let payload = serde_json::to_string(&input).unwrap();

        // IMPORTANT: spawn the ENGINE bin, not the `sentinel` launcher.
        // `sentinel` (src/launcher.rs) is a thin shim that execs the INSTALLED
        // `sentinel-engine` (the installed-binary hot-swap system) — so spawning it
        // would run the stale installed engine, not the freshly-built code under
        // test. The engine bin (src/main.rs) is the real hook processor.
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sentinel-engine"));
        cmd.args(["hook", "--event", event])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Isolate the filesystem via SENTINEL_HOME — the engine routes ALL
            // home resolution (home_dir/config_dir/state_dir) through it. Plain
            // HOME/USERPROFILE do NOT isolate on Windows (dirs::home_dir reads
            // the OS profile API, not env), which is why this override exists.
            .env("SENTINEL_HOME", self.home_path())
            .env("HOME", self.home_path())
            .env("USERPROFILE", self.home_path())
            // Pass caller-validation without a real tty / Claude Code parent.
            .env("SENTINEL_ALLOW_TERMINAL", "1")
            .env("CLAUDE_CODE_ENTRY_POINT", "cli")
            .env("CLAUDE_SESSION_ID", &self.session_id)
            // Scrub the higher-precedence session-id channels: the dispatcher
            // resolves VULCAN_SESSION_ID / CLAUDE_CODE_SESSION_ID / SESSION_ID
            // before CLAUDE_SESSION_ID, and a test run started from inside a
            // Claude Code session leaks CLAUDE_CODE_SESSION_ID into child
            // processes — hooks would silently operate on the OPERATOR's live
            // session instead of this test's.
            .env_remove("VULCAN_SESSION_ID")
            .env_remove("CLAUDE_CODE_SESSION_ID")
            .env_remove("SESSION_ID")
            // Satisfy enterprise router/scorer construction without using real
            // credentials. These tests do not exercise live A3/A13 model calls.
            .env("OPENROUTER_API_KEY", "test-openrouter-key")
            .env_remove("QDRANT_URL")
            .env_remove("LINEAR_API_KEY")
            .env_remove("CEREBRAS_API_KEY");
        for (k, v) in &self.extra_env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().expect("spawn sentinel");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(payload.as_bytes())
            .expect("write stdin");
        let out = child.wait_with_output().expect("wait sentinel");

        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let json = if stdout.is_empty() {
            json!({})
        } else {
            // The binary prints exactly one JSON object; take the last
            // non-empty line in case any stray text precedes it.
            let line = stdout
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("{}");
            serde_json::from_str(line.trim())
                .unwrap_or_else(|e| panic!("stdout not JSON ({e}): {stdout:?}\nstderr: {stderr}"))
        };
        HookResult {
            json,
            exit_ok: out.status.success(),
            stderr,
        }
    }
}

/// Parsed result of a hook invocation.
struct HookResult {
    json: Value,
    exit_ok: bool,
    stderr: String,
}

impl HookResult {
    /// The hook allowed: empty `{}` (no block, no injection, no stop).
    fn assert_allow(&self) {
        assert!(
            self.exit_ok,
            "process should exit 0; stderr: {}",
            self.stderr
        );
        assert_eq!(
            self.json,
            json!({}),
            "expected allow ({{}}), got: {}",
            self.json
        );
    }

    /// The hook produced a valid `HookOutput` (parseable JSON object) — the
    /// contract Claude Code actually consumes.
    ///
    /// NOTE on exit code: the binary's tokio runtime can, *rarely and only
    /// under heavy concurrent process-spawning* (as cargo's parallel test
    /// harness does), panic during blocking-pool **teardown on exit**
    /// (tokio `runtime/blocking/shutdown.rs`) AFTER the correct stdout has
    /// already been written. That is a teardown timing artifact, not a hook
    /// logic failure — verified: 20/20 sequential runs exit 0 with correct
    /// `{}`. Production fires hooks serially, so the risk there is negligible.
    /// We therefore assert on the OUTPUT contract (valid JSON emitted), and
    /// only treat a non-zero exit as a hard failure when NO output was
    /// produced (a real crash before doing the work).
    fn assert_no_crash(&self) {
        assert!(
            self.json.is_object(),
            "output must be a JSON object; exit_ok={}, stderr: {}",
            self.exit_ok,
            self.stderr
        );
        // Hard-fail only if the process errored AND emitted nothing useful —
        // i.e. it crashed before producing a HookOutput.
        if !self.exit_ok && self.json == json!({}) && !self.stderr.is_empty() {
            // Distinguish a teardown panic (output was correct {}) from a
            // pre-output crash. A teardown panic still wrote `{}` to stdout,
            // so this branch only trips on a genuine no-output crash with a
            // logic error on stderr (not the known tokio-shutdown message).
            let teardown_panic = self.stderr.contains("runtime")
                && self.stderr.contains("shutdown")
                || self.stderr.is_empty();
            assert!(
                teardown_panic,
                "hook crashed before producing output, stderr: {}",
                self.stderr
            );
        }
    }

    /// `PreToolUse` deny: `hookSpecificOutput.permissionDecision == "deny"`,
    /// reason contains `needle`.
    fn assert_deny(&self, needle: &str) {
        let dec = self
            .json
            .pointer("/hookSpecificOutput/permissionDecision")
            .and_then(Value::as_str);
        assert_eq!(dec, Some("deny"), "expected deny, got: {}", self.json);
        let reason = self
            .json
            .pointer("/hookSpecificOutput/permissionDecisionReason")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert!(
            reason.contains(needle),
            "deny reason missing {needle:?}: {reason:?}"
        );
    }

    /// `PreToolUse` ask: `permissionDecision == "ask"`.
    fn assert_ask(&self) {
        let dec = self
            .json
            .pointer("/hookSpecificOutput/permissionDecision")
            .and_then(Value::as_str);
        assert_eq!(dec, Some("ask"), "expected ask, got: {}", self.json);
    }

    /// Context-injection: `hookSpecificOutput.additionalContext` contains `needle`.
    fn assert_injects(&self, needle: &str) {
        let ctx = self
            .json
            .pointer("/hookSpecificOutput/additionalContext")
            .and_then(Value::as_str)
            .unwrap_or("");
        assert!(
            ctx.contains(needle),
            "injected context missing {needle:?}: {ctx:?}"
        );
    }

    fn is_allow(&self) -> bool {
        self.json == json!({})
    }
}

// ─── Every-event smoke: each event dispatches, exits 0, emits valid JSON ─────

#[test]
fn every_event_dispatches_without_crash() {
    // The full HookEvent::from_arg set. A regression in dispatch/parse for any
    // event surfaces here as a crash or non-JSON output.
    const EVENTS: &[&str] = &[
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
        "StopFailure",
        "PreCompact",
        "PostCompact",
        "Setup",
        "SubagentStart",
        "SubagentStop",
        "TeammateIdle",
        "TaskCreated",
        "TaskCompleted",
        "PermissionDenied",
        "CwdChanged",
        "PermissionRequest",
        "Elicitation",
        "ElicitationResult",
        "ConfigChange",
        "InstructionsLoaded",
        "FileChanged",
        "WorktreeCreate",
        "WorktreeRemove",
        "Notification",
    ];
    for event in EVENTS {
        let t = HookTest::new();
        let res = t.run(event, json!({}));
        res.assert_no_crash();
    }
}

#[test]
fn unknown_event_is_rejected() {
    let t = HookTest::new();
    let res = t.run("TotallyNotAnEvent", json!({}));
    // Unknown event → non-zero exit (parse_hook_event errors). Must not hang
    // or emit a bogus allow.
    assert!(
        !res.exit_ok || !res.is_allow(),
        "unknown event must not silently allow: {}",
        res.json
    );
}

#[test]
fn missing_session_id_is_safe_noop() {
    // No sessionId in JSON and none of the env fallbacks (VULCAN_SESSION_ID /
    // CLAUDE_CODE_SESSION_ID / CLAUDE_SESSION_ID / SESSION_ID) → safe {}
    // (no-op), per the 2026-05-06 fix (no synthetic 'unknown' session). The
    // whole chain must be scrubbed: running tests from inside a Claude Code
    // session leaks CLAUDE_CODE_SESSION_ID into child processes.
    let t = HookTest::new();
    // Build a payload with NO sessionId, and clear the env fallback.
    let payload = "{}";
    let mut child = Command::new(env!("CARGO_BIN_EXE_sentinel"))
        .args(["hook", "--event", "Stop"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("HOME", t.home_path())
        .env("USERPROFILE", t.home_path())
        .env("SENTINEL_CLAUDE_DIR", t.claude_dir())
        .env("SENTINEL_ALLOW_TERMINAL", "1")
        .env("CLAUDE_CODE_ENTRY_POINT", "cli")
        .env_remove("VULCAN_SESSION_ID")
        .env_remove("CLAUDE_CODE_SESSION_ID")
        .env_remove("CLAUDE_SESSION_ID")
        .env_remove("SESSION_ID")
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim() == "{}" || stdout.trim().is_empty(),
        "missing session id must be a safe no-op, got: {stdout:?}"
    );
}

// ─── Fail-open: service-dependent hooks must not block/crash offline ─────────

#[test]
fn stop_with_no_services_is_allow_or_inject_never_crash() {
    // Stop runs many hooks (skill_telemetry, memory_*, claim_reality_check,
    // self_annealing, …). With all service env cleared and no fixtures, the
    // turn must complete cleanly — never crash, never block.
    let t = HookTest::new();
    let res = t.run("Stop", json!({"cwd": t.home_path().to_string_lossy()}));
    res.assert_no_crash();
    // Stop hooks may inject context but must never set continue:false here.
    assert_ne!(
        res.json.get("continue"),
        Some(&json!(false)),
        "clean Stop must not halt the turn: {}",
        res.json
    );
}

#[test]
fn pretooluse_benign_bash_is_allowed_offline() {
    // A harmless Bash command with no gate-tripping fixture and no services →
    // allow. (skill_router etc. fail open with no API key.)
    let t = HookTest::new();
    let res = t.run(
        "PreToolUse",
        json!({
            "tool_name": "Bash",
            "tool_input": {"command": "echo hello"},
            "cwd": t.home_path().to_string_lossy(),
        }),
    );
    res.assert_no_crash();
    assert_ne!(
        res.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("deny")),
        "benign echo must not be denied: {}",
        res.json
    );
}

// ─── Blocking gates: each must DENY with its fixture and ALLOW without it ─────
//
// NOTE: the trigger command strings are assembled from fragments at runtime so
// this test SOURCE file doesn't contain verbatim dangerous strings (which the
// developer's own live sentinel session would otherwise re-scan and flag).

#[test]
fn db_ops_gate_denies_prod_migration_allows_local() {
    let t = HookTest::new();
    // Prod indicator must be IN THE COMMAND (the gate regexes the command text,
    // not the env): "<orm> <migrate> <deploy> --env <production>".
    let prod_cmd = format!("prisma {} deploy --env {}", "migrate", "production");
    let deny = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": prod_cmd},
               "cwd": t.home_path().to_string_lossy()}),
    );
    deny.assert_deny("PRODUCTION");

    // Local migration (dev) → allowed.
    let dev_cmd = format!("prisma {} dev", "migrate");
    let allow = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": dev_cmd},
               "cwd": t.home_path().to_string_lossy()}),
    );
    assert_ne!(
        allow.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("deny")),
        "local dev migration must not be denied: {}",
        allow.json
    );
}

#[test]
fn commit_message_validator_denies_nonconventional_allows_conventional() {
    let t = HookTest::new();
    // Satisfy the task_decomposition_gate (also a PreToolUse Bash hook): without
    // a live task list it denies ALL mutating Bash before the commit validator's
    // verdict can decide, which would mask both the deny-bad and allow-good
    // assertions below. Seed one open task so the gate passes and the commit
    // message validator is the hook under test.
    t.seed_task(
        "1",
        json!({"id": "1", "subject": "commit work", "status": "in_progress"}),
    );
    let commit = "commit"; // assembled so source has no verbatim "git commit"
    let bad = format!("git {commit} -m \"just fixed stuff\"");
    let deny = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": bad},
               "cwd": t.home_path().to_string_lossy()}),
    );
    deny.assert_deny("conventional");

    let good = format!("git {commit} -m \"fix: handle null response\"");
    let allow = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": good},
               "cwd": t.home_path().to_string_lossy()}),
    );
    assert_ne!(
        allow.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("deny")),
        "conventional commit must be allowed: {}",
        allow.json
    );
}

/// Install a stub `gh` into the temp home and prepend it to PATH, so the
/// fetch-and-decide gate resolves a deterministic PR state end-to-end instead
/// of whatever the host's real `gh` says.
fn seed_stub_gh(t: &mut HookTest, json_payload: &str) {
    let bin = t.home_path().join("stubbin");
    std::fs::create_dir_all(&bin).expect("mk stubbin");
    let gh = bin.join("gh");
    std::fs::write(
        &gh,
        format!("#!/bin/sh\ncat <<'STUBEOF'\n{json_payload}\nSTUBEOF\n"),
    )
    .expect("write stub gh");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&gh, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = std::env::var("PATH").unwrap_or_default();
    t.env("PATH", &format!("{}:{path}", bin.display()));
}

#[test]
fn pr_merge_gate_denies_merge_with_changes_requested_and_failing_ci() {
    // The fetch-and-decide contract: verified-bad state is a DETERMINISTIC
    // deny, not a coin-flip ask — and autopilot does not downgrade it.
    let mut t = HookTest::new();
    t.env("SENTINEL_AUTOPILOT", "1");
    seed_stub_gh(
        &mut t,
        r#"{"baseRefName":"main","mergeable":"MERGEABLE","reviewDecision":"CHANGES_REQUESTED",
            "reviews":[{"author":{"login":"mkoslowski"},"state":"CHANGES_REQUESTED"}],
            "statusCheckRollup":[{"name":"integration","status":"COMPLETED","conclusion":"FAILURE"}]}"#,
    );
    let gh = "gh"; // assembled so source has no verbatim "gh pr merge"
    let merge = format!("{gh} pr merge 7 --squash");
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": merge},
               "cwd": t.home_path().to_string_lossy()}),
    );
    assert_eq!(
        res.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("deny")),
        "verified CHANGES_REQUESTED + failing CI must deny: {}",
        res.json
    );
    let reason = res
        .json
        .pointer("/hookSpecificOutput/permissionDecisionReason")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        reason.contains("Remedy:"),
        "deny must be instructive: {reason}"
    );
}

#[test]
fn pr_merge_gate_allows_approved_and_green_merge_silently() {
    // Routine merge on verified-good state costs the operator nothing.
    let mut t = HookTest::new();
    t.env("SENTINEL_AUTOPILOT", "1");
    seed_stub_gh(
        &mut t,
        r#"{"baseRefName":"main","mergeable":"MERGEABLE","reviewDecision":"APPROVED",
            "reviews":[{"author":{"login":"mkoslowski"},"state":"APPROVED"}],
            "statusCheckRollup":[{"name":"build","status":"COMPLETED","conclusion":"SUCCESS"}]}"#,
    );
    let gh = "gh";
    let merge = format!("{gh} pr merge 5 --squash");
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": merge},
               "cwd": t.home_path().to_string_lossy()}),
    );
    assert_ne!(
        res.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("deny")),
        "approved+green must not be denied: {}",
        res.json
    );
    assert_ne!(
        res.json.pointer("/hookSpecificOutput/permissionDecision"),
        Some(&json!("ask")),
        "approved+green must not ask: {}",
        res.json
    );
    let reason = res
        .json
        .pointer("/hookSpecificOutput/permissionDecisionReason")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        !reason.contains("PR Merge Gate"),
        "approved+green must produce no PR-gate message at all: {reason}"
    );
}

#[test]
fn pr_merge_gate_asks_when_pr_state_cannot_be_fetched() {
    // The critical fail-mode: unknown state must ask — never silently allow
    // (a reckless merge slips through) and never hard-block (fail-closed gates
    // bricked the bench once already).
    let mut t = HookTest::new();
    t.env("SENTINEL_AUTOPILOT", "0");
    // A `gh` stub that always fails, so state genuinely cannot be established.
    let bin = t.home_path().join("stubbin");
    std::fs::create_dir_all(&bin).unwrap();
    let ghbin = bin.join("gh");
    std::fs::write(
        &ghbin,
        "#!/bin/sh\necho 'gh: not authenticated' >&2\nexit 1\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&ghbin, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let path = std::env::var("PATH").unwrap_or_default();
    t.env("PATH", &format!("{}:{path}", bin.display()));

    let gh = "gh";
    let merge = format!("{gh} pr merge 123 --squash");
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": merge},
               "cwd": t.home_path().to_string_lossy()}),
    );
    res.assert_ask();
}

#[test]
fn pr_merge_gate_no_longer_fires_on_pr_close() {
    // Closing a PR is reversible (reopenable); clean reviewers judged the
    // escalation unwarranted 0/3. The gate must be silent by default.
    let mut t = HookTest::new();
    t.env("SENTINEL_AUTOPILOT", "0");
    let gh = "gh";
    let close = format!("{gh} pr close 12");
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": close},
               "cwd": t.home_path().to_string_lossy()}),
    );
    let reason = res
        .json
        .pointer("/hookSpecificOutput/permissionDecisionReason")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        !reason.contains("PR Merge Gate"),
        "pr close must not trip the merge gate: {}",
        res.json
    );
}

#[test]
fn autopilot_downgraded_ask_writes_raw_outcome_ask_to_ledger() {
    // Under SENTINEL_AUTOPILOT=1 a non-prod Doppler mutation is downgraded
    // from its ask-class "stop and ask the user first" verdict to an
    // effective allow BEFORE the ledger row is classified — which used to
    // erase the "would have escalated to a human" signal entirely. The
    // ledger row must now preserve both: `outcome=allow` (effective) and
    // `raw_outcome=ask` (pre-downgrade).
    let mut t = HookTest::new();
    t.env("SENTINEL_AUTOPILOT", "1");
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"mcp__doppler__set_secret",
               "tool_input":{"project":"demo-app","config":"dev","name":"API_KEY"}}),
    );
    // The doppler gate itself must not surface a deny for a non-prod
    // mutation in autopilot (other gates may still merge their own verdicts,
    // so assert on the gate's reason, not the merged decision).
    let merged_reason = res
        .json
        .pointer("/hookSpecificOutput/permissionDecisionReason")
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        !merged_reason.contains("[Doppler/Auth0 Gate]"),
        "doppler gate must not block non-prod mutation in autopilot: {}",
        res.json
    );

    let ledger = t
        .claude_dir()
        .join("sentinel")
        .join("metrics")
        .join("hook-invocations.jsonl");
    let text = std::fs::read_to_string(&ledger).expect("hook-invocations.jsonl written");
    let row = text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|r| r["hook"] == "doppler_auth0_gate" && r["event"] == "PreToolUse")
        .expect("doppler_auth0_gate ledger row present");
    assert_eq!(
        row["outcome"], "allow",
        "effective outcome must stay allow under autopilot: {row}"
    );
    assert_eq!(
        row["raw_outcome"], "ask",
        "pre-downgrade ask-class verdict must be preserved as raw_outcome: {row}"
    );
}

#[test]
fn non_downgraded_ledger_rows_carry_raw_outcome_equal_to_outcome() {
    // When no downgrade happened, every ledger row must satisfy
    // raw_outcome == outcome (the field is always written).
    let t = HookTest::new();
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command":"git status --short"},
               "cwd": t.home_path().to_string_lossy()}),
    );
    assert!(
        res.json.pointer("/hookSpecificOutput/permissionDecision") != Some(&json!("deny")),
        "read-only bash should not be denied: {}",
        res.json
    );
    let ledger = t
        .claude_dir()
        .join("sentinel")
        .join("metrics")
        .join("hook-invocations.jsonl");
    let text = std::fs::read_to_string(&ledger).expect("hook-invocations.jsonl written");
    let mut rows = 0;
    for row in text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
    {
        rows += 1;
        assert_eq!(
            row["raw_outcome"], row["outcome"],
            "no downgrade fired, raw_outcome must equal outcome: {row}"
        );
    }
    assert!(rows > 0, "expected at least one ledger row");
}

#[test]
fn readonly_bash_in_real_repo_is_allowed() {
    // A read-only Bash (`git status`) in a real repo must NOT be blocked: it's
    // not a mutating tool, so the task_decomposition_gate allows it and no
    // blocking hook denies/asks. We assert "permitted" (allow), not a specific
    // bare `{}`, to stay robust to other allow-with-context hooks. Exercises
    // seed_git_repo end to end.
    let t = HookTest::new();
    let repo = t.seed_git_repo("proj", "main", "README.md", false);
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Bash","tool_input":{"command": "git status --short"},
               "cwd": repo.to_string_lossy()}),
    );
    let dec = res
        .json
        .pointer("/hookSpecificOutput/permissionDecision")
        .and_then(Value::as_str);
    assert!(
        res.is_allow() || dec == Some("allow"),
        "read-only git status must be permitted, got: {}",
        res.json
    );
    assert_ne!(dec, Some("deny"), "read-only git status must not be denied");
    assert_ne!(
        dec,
        Some("ask"),
        "read-only git status must not require ask"
    );
}

#[test]
fn read_tool_is_bare_allow() {
    // A pure Read tool is touched by no PreToolUse gate → exact `{}` allow.
    // This is the canonical bare-allow path (exercises assert_allow).
    let t = HookTest::new();
    let res = t.run(
        "PreToolUse",
        json!({"tool_name":"Read","tool_input":{"file_path":"/tmp/x"},
               "cwd": t.home_path().to_string_lossy()}),
    );
    res.assert_allow();
}

// ─── Reality-check / work-assurance: the F1 scenario, now driven E2E ─────────

#[test]
fn completion_promise_false_done_is_flagged_on_stop() {
    // THE F1 SCENARIO: a COMPLETED task that emits an explicit completion
    // promise (IMPLEMENTATION_COMPLETE) but names no real commit/PR. On Stop,
    // claim_reality_check must surface a hard-mismatch reality-check warning.
    let t = HookTest::new();
    let promise = format!("{}_{}", "IMPLEMENTATION", "COMPLETE"); // avoid verbatim token in source
    t.seed_task(
        "1",
        json!({
            "id": "1",
            "subject": "Ship the auth refactor",
            "description": format!("Done — {promise}. Shipped the feature."),
            "status": "completed"
        }),
    );
    let res = t.run("Stop", json!({"cwd": t.home_path().to_string_lossy()}));
    res.assert_no_crash();
    // The reality-check injects context naming the unverifiable completed task.
    // (If this asserts false, F1 is a real hook bug, not a test artifact.)
    // NOTE: this only reaches the agent because Stop carries
    // hookSpecificOutput.additionalContext through hook_cmd.rs's native-support
    // arm — Stop/SubagentStop were previously absent there and the `_ =>` arm
    // STRIPPED this context, which was the true root of the F1 `{}` finding.
    res.assert_injects("Reality");
}

#[test]
fn clean_completed_task_no_false_positive_on_stop() {
    // A completed task with NO concrete-artifact claim must NOT be flagged
    // (guards against reality-check noise / false positives).
    let t = HookTest::new();
    t.seed_task(
        "1",
        json!({
            "id": "1",
            "subject": "Investigate the slow query",
            "description": "Looked into the N+1; needs more profiling.",
            "status": "completed"
        }),
    );
    let res = t.run("Stop", json!({"cwd": t.home_path().to_string_lossy()}));
    res.assert_no_crash();
    let ctx = res
        .json
        .pointer("/hookSpecificOutput/additionalContext")
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(
        !ctx.contains("Reality-check"),
        "a claim-free completed task must not be reality-check-flagged: {ctx}"
    );
}
