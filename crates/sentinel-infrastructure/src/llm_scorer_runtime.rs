//! Shared LLM-bridge plumbing for the three scorer adapters:
//! [`crate::dry_run_auditor`], [`crate::eval_scorer`], and
//! [`crate::spec_challenge_scorer`].
//!
//! ## What lives here
//!
//! All three adapters need identical infrastructure:
//! - A type-erased `PromptFn` seam for rig-core async calls.
//! - `real_env` — the production `std::env::var` resolver.
//! - `read_timeout` — parse `<NAMESPACE>_TIMEOUT_SECS` with an explicit default.
//! - `sidecar` — the per-scorer lazily-built tokio sidecar runtime
//!   (each scorer passes its own `thread_name` so logs are distinct).
//! - `run_blocking` — the `std::thread::scope` + `Handle::block_on`
//!   bridge that drives the async `PromptFn` from the sync `score()`
//!   method without tripping "Cannot start a runtime from within a
//!   runtime" when the caller is already inside a `#[tokio::main]`
//!   multi-thread runtime. Each caller maps the timeout / error result
//!   into its own typed error via the `map_timeout` closure.
//! - `strip_code_fence` — strip ```` ```json ```` / ```` ``` ```` wrappers
//!   that models sometimes emit despite instructions not to.
//! - `preview` — truncate a string for error messages.
//! - `build_openrouter_prompt_fn` / `build_ollama_prompt_fn` — construct
//!   the openrouter / ollama-local / ollama-cloud `PromptFn` from
//!   environment, returning both the function and the `provider_prefix`
//!   string.
//!
//! ## Preserved invariant: nested-runtime safety (fix #18)
//!
//! `run_blocking` MUST drive `handle.block_on(...)` on a **dedicated
//! `std::thread::scope` thread**, never on the calling thread.  The
//! calling thread may be a tokio worker (the `PreToolUse` hook
//! dispatch); calling `block_on` there would panic.  The spawned
//! thread is outside every runtime's worker pool, so the
//! runtime-entry guard is never tripped.  This is the exact pattern
//! that fixed the browserbase hook crash (SEN-18) — do not "simplify"
//! it by calling `block_on` directly.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use crate::llm_http::ChatClient;
use anyhow::Result;
use futures::future::BoxFuture;
use tracing::warn;

/// Output-token bound for OpenRouter scorer/auditor calls. A reasoning model
/// (e.g. `gpt-5.5-pro`) runs UNBOUNDED — and the call stalls — if no
/// `max_tokens` is sent, so every scorer prompt must cap it. Sized for a JSON
/// verdict (scores + a paragraph of reasoning) PLUS the model's hidden
/// reasoning tokens, which are billed against this same budget. Must stay
/// ≥ 16: OpenAI rejects `max_output_tokens < 16` with a 400, which would make
/// the cross-vendor dual auditor's OpenAI leg error on every call and trip the
/// `block_for_inconclusive` path (a silent permanent-block, the same failure
/// mode the Fable-5-suspension fix removed).
const OPENROUTER_SCORER_MAX_TOKENS: u32 = 2048;

// ---------------------------------------------------------------------------
// Public type alias
// ---------------------------------------------------------------------------

/// Type-erased prompt function: `(model_id, system, user_msg) -> response_text`.
///
/// Every scorer adapter stores one of these behind its struct and calls
/// it from the sync `score()` method via [`run_blocking`].
pub type PromptFn =
    Arc<dyn Fn(String, String, String) -> BoxFuture<'static, anyhow::Result<String>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

/// Production env resolver — wraps `std::env::var`.
///
/// Tests inject HashMap-backed closures via the private `*_from_env_with`
/// variants instead, avoiding the process-wide env mutation that Rust 2024
/// marks as `unsafe`.
pub fn real_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Read `key` from the resolver, treating **empty or whitespace-only values
/// as unset** (returns `None`). The returned value is trimmed.
///
/// Rationale: `VAR=""` in a wrapper script or CI matrix is an operator saying
/// "not configured", not "configured to the empty string". Without this,
/// `OLLAMA_BASE_URL=""` suppressed a valid `OLLAMA_HOST`, and
/// `OLLAMA_API_KEY=""` selected cloud mode with an empty bearer token.
///
/// NOTE: `read_timeout` deliberately does NOT use this — a set-but-empty
/// `*_TIMEOUT_SECS` stays a hard config error (documented, tested contract)
/// rather than silently falling back to the default.
pub fn env_non_empty<F>(env: &F, key: &str) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    env(key)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Read a `<namespace>_TIMEOUT_SECS` variable from the supplied env resolver.
///
/// Absence uses `default`. Set-but-empty, zero, or malformed values are config
/// errors so scorer constructors cannot silently retime production judging.
///
/// Each scorer passes its own env-var name:
///
/// | Scorer                  | var_name                                  |
/// |-------------------------|-------------------------------------------|
/// | `dry_run_auditor`       | `"SENTINEL_AUDITOR_TIMEOUT_SECS"`         |
/// | `eval_scorer`           | `"SENTINEL_EVAL_SCORER_TIMEOUT_SECS"`     |
/// | `spec_challenge_scorer` | `"SENTINEL_SPEC_CHALLENGE_SCORER_TIMEOUT_SECS"` |
pub fn read_timeout<F>(env: &F, var_name: &str, default: Duration) -> Result<Duration>
where
    F: Fn(&str) -> Option<String>,
{
    let Some(raw) = env(var_name) else {
        return Ok(default);
    };
    if raw.trim().is_empty() {
        return Err(anyhow::anyhow!("{var_name} is set but empty"));
    }
    let secs = raw.trim().parse::<u64>().map_err(|err| {
        anyhow::anyhow!("{var_name} must be a positive integer, got {raw:?}: {err}")
    })?;
    if secs == 0 {
        return Err(anyhow::anyhow!("{var_name} must be greater than zero"));
    }
    Ok(Duration::from_secs(secs))
}

// ---------------------------------------------------------------------------
// Sidecar runtime
// ---------------------------------------------------------------------------

/// Lazily build a named sidecar tokio runtime for a scorer.
///
/// Each call site passes a unique `thread_name` so runtime worker
/// threads are identifiable in traces / thread dumps:
///
/// | Scorer                  | `thread_name`                              |
/// |-------------------------|--------------------------------------------|
/// | `dry_run_auditor`       | `"sentinel-auditor-sidecar"`               |
/// | `eval_scorer`           | `"sentinel-eval-scorer-sidecar"`           |
/// | `spec_challenge_scorer` | `"sentinel-spec-challenge-scorer-sidecar"` |
///
/// The runtime is stored in a `OnceLock<Option<Runtime>>` supplied by
/// the caller so each scorer gets its own independent runtime slot.
/// Returns `None` when the runtime could not be built (logs a warning).
pub fn sidecar(
    once: &'static OnceLock<Option<tokio::runtime::Runtime>>,
    thread_name: &'static str,
) -> Option<&'static tokio::runtime::Runtime> {
    once.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name(thread_name)
            .build()
            .map_err(|e| warn!(?e, thread_name, "failed to build scorer sidecar runtime"))
            .ok()
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// Nested-runtime-safe blocking bridge  (#18 fix)
// ---------------------------------------------------------------------------

/// Drive an async future from a sync context via a sidecar runtime,
/// without panicking even when the **calling thread is already a tokio
/// worker** (the `PreToolUse` hook dispatch situation that caused the
/// browserbase crash).
///
/// ## Why this works
///
/// `tokio::runtime::Handle::block_on` blocks the *current* thread.
/// If the current thread is a tokio worker it panics.  The fix: spawn
/// a fresh `std::thread::scope` thread — that thread is outside every
/// runtime's worker pool — and call `block_on` there.
///
/// ## Parameters
///
/// - `handle` — a clone of the sidecar runtime's `Handle`.
/// - `timeout` — applied around the future with
///   `tokio::time::timeout`.
/// - `fut` — the `PromptFn` output, already constructed by the
///   caller.
/// - `map_timeout` — maps an `Elapsed` event into the caller's typed
///   error (avoids any dependency on specific error types here).
///
/// ## Return
///
/// `Ok(String)` on success, `Err(E)` on network error (propagated
/// from the future) or timeout (mapped by `map_timeout`), or
/// `Err(E)` when the worker thread panics (via the
/// `unwrap_or_else` branch).
pub fn run_blocking<E>(
    handle: tokio::runtime::Handle,
    timeout: Duration,
    fut: BoxFuture<'static, anyhow::Result<String>>,
    map_network_err: impl FnOnce(String) -> E + Send + 'static,
    map_timeout: impl FnOnce(Duration) -> E + Send + 'static,
    map_panic: impl FnOnce() -> E + Send + 'static,
) -> Result<String, E>
where
    E: Send + 'static,
{
    std::thread::scope(|s| {
        s.spawn(move || {
            handle.block_on(async move {
                match tokio::time::timeout(timeout, fut).await {
                    Ok(Ok(text)) => Ok(text),
                    Ok(Err(err)) => Err(map_network_err(format!("{err:#}"))),
                    Err(_elapsed) => Err(map_timeout(timeout)),
                }
            })
        })
        .join()
        .unwrap_or_else(|_| Err(map_panic()))
    })
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

/// Strip a markdown code fence (```` ```json ```` or ```` ``` ````) that
/// models sometimes wrap their JSON in despite instructions.
///
/// Returns an owned `String` — callers pass it to `serde_json::from_str`.
pub fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        return rest.trim_end_matches("```").trim().to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        return rest.trim_end_matches("```").trim().to_string();
    }
    trimmed.to_string()
}

/// Truncate `text` to at most `max_chars` Unicode scalar values for use
/// in error messages.  Appends `"..."` when truncation occurs.
pub fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}

// ---------------------------------------------------------------------------
// Provider dispatch
// ---------------------------------------------------------------------------

/// Dummy bearer token sent to local Ollama (which ignores the value on
/// its OpenAI-compat endpoint).
const OLLAMA_LOCAL_DUMMY_KEY: &str = "ollama-local";

/// Default base URL for local Ollama's OpenAI-compatible endpoint.
pub const DEFAULT_OLLAMA_LOCAL_BASE_URL: &str = "http://localhost:11434/v1";

/// Default base URL for Ollama Cloud's OpenAI-compatible endpoint.
pub const DEFAULT_OLLAMA_CLOUD_BASE_URL: &str = "https://ollama.com/v1";

/// Build a `PromptFn` that routes to the `OpenRouter` provider.
///
/// Returns `(PromptFn, "openrouter")`.
///
/// ## Parameters
///
/// - `key` — the `OPENROUTER_API_KEY` value.
/// - `scorer_label` — a short human-readable label used in error
///   messages, e.g. `"auditor"`, `"scorer"`,
///   `"spec-challenge scorer"`.
pub fn build_openrouter_prompt_fn(
    key: &str,
    scorer_label: &'static str,
) -> Result<(PromptFn, String)> {
    let client = ChatClient::openrouter(key)
        .map_err(|e| anyhow::anyhow!("failed to build OpenRouter client: {e}"))?;
    let prompt_fn: PromptFn = Arc::new(move |model_id, system, user_msg| {
        let client = client.clone();
        Box::pin(async move {
            // Bound output tokens — a reasoning model (gpt-5.5-pro) otherwise
            // runs unbounded and stalls; OpenAI also 400s `max_output_tokens`
            // below 16. See OPENROUTER_SCORER_MAX_TOKENS.
            client
                .complete(
                    &model_id,
                    Some(&system),
                    &user_msg,
                    Some(OPENROUTER_SCORER_MAX_TOKENS),
                    None,
                )
                .await
                .map_err(|e| anyhow::anyhow!("openrouter {scorer_label} ({model_id}): {e}"))
        })
    });
    Ok((prompt_fn, "openrouter".to_string()))
}

/// Resolve the Ollama-style endpoint from the env resolver.
///
/// Returns `(base_url, api_key, provider_prefix)`. Extracted from
/// [`build_ollama_prompt_fn`] so the resolution order is directly unit-
/// testable:
///
/// 1. `OLLAMA_API_KEY` set → cloud mode: `OLLAMA_BASE_URL` (default
///    [`DEFAULT_OLLAMA_CLOUD_BASE_URL`]), bearer auth, `"ollama-cloud"`.
/// 2. Keyless + `OLLAMA_BASE_URL` set → that URL **verbatim** (point it at
///    any OpenAI-compatible `/v1` root: vLLM, litellm, ollama),
///    `"ollama-local"`.
/// 3. Keyless, no base URL → `OLLAMA_HOST` (default
///    `http://localhost:11434`) with `/v1` appended, `"ollama-local"`.
///
/// Empty / whitespace-only values count as unset for all three vars
/// ([`env_non_empty`]). When both `OLLAMA_BASE_URL` and `OLLAMA_HOST` are
/// set keyless, `OLLAMA_BASE_URL` wins and a one-line warning makes the
/// precedence visible.
fn resolve_ollama_endpoint<F>(env: &F) -> (String, String, String)
where
    F: Fn(&str) -> Option<String>,
{
    // Empty/whitespace-only values are "unset" for every var here (see
    // `env_non_empty`): an empty OLLAMA_API_KEY must not select cloud mode
    // with an empty bearer, and an empty OLLAMA_BASE_URL must not suppress a
    // valid OLLAMA_HOST.
    env_non_empty(env, "OLLAMA_API_KEY").map_or_else(
        || {
            // Local mode: honor an explicit OLLAMA_BASE_URL first — it is
            // used verbatim, so it can point at ANY OpenAI-compatible /v1
            // endpoint (vLLM, litellm, ollama). Only fall back to the
            // host-plus-appended-/v1 form when no base URL is given.
            let base_url = env_non_empty(env, "OLLAMA_BASE_URL");
            let host = env_non_empty(env, "OLLAMA_HOST");
            if let (Some(base), Some(host)) = (&base_url, &host) {
                // Cross-component precedence notice: BASE_URL beats HOST for
                // every consumer of this plumbing (A3 auditor router path,
                // A13 scorer). Make the reroute visible, never silent.
                warn!(
                    base_url = %crate::llm_http::redact_url_userinfo(base),
                    ignored_host = %crate::llm_http::redact_url_userinfo(host),
                    "OLLAMA_BASE_URL and OLLAMA_HOST are both set; OLLAMA_BASE_URL wins and is used verbatim"
                );
            }
            let base = base_url.unwrap_or_else(|| {
                let host = host.unwrap_or_else(|| "http://localhost:11434".to_string());
                format!("{}/v1", host.trim_end_matches('/'))
            });
            (
                base,
                OLLAMA_LOCAL_DUMMY_KEY.to_string(),
                "ollama-local".to_string(),
            )
        },
        |key| {
            let base = env_non_empty(env, "OLLAMA_BASE_URL")
                .unwrap_or_else(|| DEFAULT_OLLAMA_CLOUD_BASE_URL.to_string());
            (base, key, "ollama-cloud".to_string())
        },
    )
}

/// Build a `PromptFn` that routes to an Ollama-style **OpenAI-compatible
/// chat-completions endpoint** (local daemon, vLLM, litellm, or Ollama
/// Cloud), auto-detecting which mode to use from the env resolver.
///
/// The wire protocol is always OpenAI-compatible `/chat/completions`
/// (via [`ChatClient::openai_compat`]) — never Ollama's native
/// `/api/generate` — so any `/v1` server works (vLLM and litellm are
/// first-class targets, not just the ollama daemon).
///
/// - If `OLLAMA_API_KEY` is set → **cloud / authenticated mode**. Uses
///   `OLLAMA_BASE_URL` (default [`DEFAULT_OLLAMA_CLOUD_BASE_URL`]) with
///   bearer auth.
/// - Otherwise → **local mode**. Uses `OLLAMA_BASE_URL` verbatim when
///   set (point it at any OpenAI-compatible `/v1` root, e.g. a vLLM
///   serve); else `OLLAMA_HOST` (default `http://localhost:11434`) with
///   `/v1` appended. A dummy bearer token is sent because local
///   OpenAI-compat endpoints ignore it.
///
/// Returns `(PromptFn, provider_prefix)` where `provider_prefix` is
/// `"ollama-cloud"` or `"ollama-local"`.
///
/// ## Parameters
///
/// - `env` — the env resolver (same seam used everywhere in the scorers).
/// - `scorer_label` — short label for error messages.
pub fn build_ollama_prompt_fn<F>(env: &F, scorer_label: &'static str) -> Result<(PromptFn, String)>
where
    F: Fn(&str) -> Option<String>,
{
    let (base_url, api_key, provider_prefix) = resolve_ollama_endpoint(env);

    let client = ChatClient::openai_compat(&base_url, &api_key)
        .map_err(|e| anyhow::anyhow!("failed to build ollama client (base_url={base_url}): {e}"))?;
    let provider_for_closure = provider_prefix.clone();
    let prompt_fn: PromptFn = Arc::new(move |model_id, system, user_msg| {
        let client = client.clone();
        let provider = provider_for_closure.clone();
        Box::pin(async move {
            client
                .complete(&model_id, Some(&system), &user_msg, None, None)
                .await
                .map_err(|e| anyhow::anyhow!("{provider} {scorer_label} ({model_id}): {e}"))
        })
    });
    Ok((prompt_fn, provider_prefix))
}

// ---------------------------------------------------------------------------
// Unit tests for the shared utilities
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // strip_code_fence — existing coverage
    // -----------------------------------------------------------------------

    #[test]
    fn strip_code_fence_json_variant() {
        let input = "```json\n{\"k\":1}\n```";
        assert_eq!(strip_code_fence(input), r#"{"k":1}"#);
    }

    #[test]
    fn strip_code_fence_bare_variant() {
        let input = "```\n{\"k\":1}\n```";
        assert_eq!(strip_code_fence(input), r#"{"k":1}"#);
    }

    #[test]
    fn strip_code_fence_passthrough_plain_json() {
        let input = r#"{"k":1}"#;
        assert_eq!(strip_code_fence(input), r#"{"k":1}"#);
    }

    // -----------------------------------------------------------------------
    // strip_code_fence — additional edge cases
    // -----------------------------------------------------------------------

    /// Leading/trailing whitespace around the fenced block is stripped.
    #[test]
    fn strip_code_fence_trims_outer_whitespace() {
        let input = "  \n```json\n{\"k\":1}\n```\n  ";
        assert_eq!(strip_code_fence(input), r#"{"k":1}"#);
    }

    /// A fence with no closing ``` still returns the interior content (the
    /// implementation uses `trim_end_matches` which silently handles missing
    /// trailing fence).
    #[test]
    fn strip_code_fence_missing_closing_fence() {
        let input = "```json\n{\"k\":1}";
        // trim_end_matches("```") on "{\"k\":1}" finds no match → returns content as-is
        assert_eq!(strip_code_fence(input), r#"{"k":1}"#);
    }

    /// An empty string produces an empty string (no panic).
    #[test]
    fn strip_code_fence_empty_input() {
        assert_eq!(strip_code_fence(""), "");
    }

    /// Only whitespace produces an empty string.
    #[test]
    fn strip_code_fence_only_whitespace() {
        assert_eq!(strip_code_fence("   \n\t  "), "");
    }

    /// A fence whose interior is itself a JSON object spread across multiple
    /// lines is returned intact (newlines preserved inside the content).
    #[test]
    fn strip_code_fence_multiline_json_content() {
        let input = "```json\n{\n  \"a\": 1,\n  \"b\": 2\n}\n```";
        assert_eq!(strip_code_fence(input), "{\n  \"a\": 1,\n  \"b\": 2\n}");
    }

    // -----------------------------------------------------------------------
    // preview — existing coverage
    // -----------------------------------------------------------------------

    #[test]
    fn preview_truncates() {
        let s = "x".repeat(300);
        let p = preview(&s, 100);
        assert_eq!(p.chars().count(), 103);
        assert!(p.ends_with("..."));
    }

    #[test]
    fn preview_passthrough_short() {
        assert_eq!(preview("hi", 100), "hi");
    }

    // -----------------------------------------------------------------------
    // preview — additional edge cases
    // -----------------------------------------------------------------------

    /// Exactly at the limit is returned unchanged (boundary value).
    #[test]
    fn preview_exact_limit_not_truncated() {
        let s = "a".repeat(10);
        assert_eq!(preview(&s, 10), s);
    }

    /// One character over the limit triggers truncation with ellipsis.
    #[test]
    fn preview_one_over_limit_truncated() {
        let s = "a".repeat(11);
        let p = preview(&s, 10);
        assert!(p.ends_with("..."));
        assert_eq!(p.chars().count(), 13); // 10 content + 3 ellipsis
    }

    /// Unicode multi-byte characters are counted by scalar value, not byte.
    #[test]
    fn preview_unicode_counted_by_scalar() {
        // Each '©' is 2 bytes but 1 scalar value.
        let s: String = "©".repeat(5);
        // Limit of 5 → no truncation (exactly at boundary).
        assert_eq!(preview(&s, 5), s);
        // Limit of 4 → truncation.
        let p = preview(&s, 4);
        assert!(p.ends_with("..."));
    }

    /// max_chars = 0 always truncates any non-empty string.
    #[test]
    fn preview_zero_limit_always_truncates() {
        let p = preview("hello", 0);
        assert_eq!(p, "...");
    }

    /// An empty string at any limit returns empty (nothing to truncate).
    #[test]
    fn preview_empty_string_any_limit() {
        assert_eq!(preview("", 0), "");
        assert_eq!(preview("", 100), "");
    }

    // -----------------------------------------------------------------------
    // read_timeout — existing coverage
    // -----------------------------------------------------------------------

    #[test]
    fn read_timeout_returns_default_when_absent() {
        let t = read_timeout(&|_: &str| None, "NO_SUCH_VAR", Duration::from_secs(30)).unwrap();
        assert_eq!(t, Duration::from_secs(30));
    }

    #[test]
    fn read_timeout_parses_override() {
        let t = read_timeout(
            &|k: &str| {
                if k == "MY_TIMEOUT" {
                    Some("42".to_string())
                } else {
                    None
                }
            },
            "MY_TIMEOUT",
            Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(t, Duration::from_secs(42));
    }

    #[test]
    fn read_timeout_rejects_non_numeric() {
        let err = read_timeout(
            &|_| Some("not-a-number".to_string()),
            "VAR",
            Duration::from_secs(30),
        )
        .unwrap_err();
        assert!(err.to_string().contains("VAR"), "{err:#}");
    }

    // -----------------------------------------------------------------------
    // read_timeout — additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn read_timeout_zero_value_is_invalid() {
        let err =
            read_timeout(&|_| Some("0".to_string()), "VAR", Duration::from_secs(30)).unwrap_err();
        assert!(err.to_string().contains("greater than zero"), "{err:#}");
    }

    #[test]
    fn read_timeout_float_string_is_invalid() {
        let err =
            read_timeout(&|_| Some("3.5".to_string()), "VAR", Duration::from_secs(10)).unwrap_err();
        assert!(err.to_string().contains("VAR"), "{err:#}");
    }

    #[test]
    fn read_timeout_negative_string_is_invalid() {
        let err =
            read_timeout(&|_| Some("-5".to_string()), "VAR", Duration::from_secs(10)).unwrap_err();
        assert!(err.to_string().contains("VAR"), "{err:#}");
    }

    #[test]
    fn read_timeout_empty_string_is_invalid() {
        let err =
            read_timeout(&|_| Some(String::new()), "VAR", Duration::from_secs(10)).unwrap_err();
        assert!(err.to_string().contains("empty"), "{err:#}");
    }

    /// Only the named variable is read; other keys do not influence the result.
    #[test]
    fn read_timeout_only_reads_named_var() {
        let t = read_timeout(
            &|k: &str| match k {
                "RIGHT_VAR" => Some("99".to_string()),
                "WRONG_VAR" => Some("1".to_string()),
                _ => None,
            },
            "RIGHT_VAR",
            Duration::from_secs(30),
        )
        .unwrap();
        assert_eq!(t, Duration::from_secs(99));
    }

    // -----------------------------------------------------------------------
    // run_blocking — nested-runtime safety (SEN-18 regression)
    // -----------------------------------------------------------------------

    /// Helper: build a tiny dedicated sidecar runtime for the tests below.
    /// Each test builds its own so they stay independent.
    fn test_sidecar() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("sentinel-test-sidecar")
            .build()
            .expect("test sidecar runtime must build")
    }

    /// Happy path: a trivially-ready future resolves and `run_blocking`
    /// returns its value without panicking.
    #[test]
    fn run_blocking_returns_value_from_ready_future() {
        let rt = test_sidecar();
        let handle = rt.handle().clone();
        let fut: BoxFuture<'static, anyhow::Result<String>> =
            Box::pin(async { Ok("hello".to_string()) });

        let result = run_blocking(
            handle,
            Duration::from_secs(5),
            fut,
            |msg| format!("network: {msg}"),
            |dur| format!("timeout after {dur:?}"),
            || "panic".to_string(),
        );
        assert_eq!(result.unwrap(), "hello");
    }

    /// A future that returns `Err` is mapped through `map_network_err`.
    #[test]
    fn run_blocking_maps_network_error() {
        let rt = test_sidecar();
        let handle = rt.handle().clone();
        let fut: BoxFuture<'static, anyhow::Result<String>> =
            Box::pin(async { Err(anyhow::anyhow!("connection refused")) });

        let result = run_blocking(
            handle,
            Duration::from_secs(5),
            fut,
            |msg| format!("network: {msg}"),
            |dur| format!("timeout after {dur:?}"),
            || "panic".to_string(),
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("network:"),
            "expected network prefix, got: {err}"
        );
        assert!(
            err.contains("connection refused"),
            "expected cause, got: {err}"
        );
    }

    /// A future that never resolves hits the timeout and `map_timeout` is
    /// called with the configured duration.
    #[test]
    fn run_blocking_maps_timeout() {
        let rt = test_sidecar();
        let handle = rt.handle().clone();
        // A future that waits longer than our timeout.
        let fut: BoxFuture<'static, anyhow::Result<String>> = Box::pin(async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok("should not reach here".to_string())
        });
        let timeout = Duration::from_millis(50);

        let result = run_blocking(
            handle,
            timeout,
            fut,
            |msg| format!("network: {msg}"),
            |dur| format!("timed out after {dur:?}"),
            || "panic".to_string(),
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("timed out after"),
            "expected timeout message, got: {err}"
        );
    }

    /// Regression (SEN-18): `run_blocking` MUST NOT panic when called from
    /// within a tokio multi-thread runtime worker thread.  The old pattern
    /// of calling `sidecar.block_on()` directly on the caller's thread
    /// panicked because the caller was already a tokio worker.  The scoped-
    /// thread bridge fixes this.  This test would panic (not merely fail)
    /// against the pre-fix code.
    #[tokio::test]
    async fn run_blocking_does_not_panic_when_called_from_tokio_worker() {
        // Spawn `run_blocking` on a blocking thread of the CURRENT multi-thread
        // runtime — exactly the call path the PreToolUse hook dispatcher uses.
        let result = tokio::task::spawn_blocking(move || {
            let rt = test_sidecar();
            let handle = rt.handle().clone();
            let fut: BoxFuture<'static, anyhow::Result<String>> =
                Box::pin(async { Ok("nested-safe".to_string()) });
            run_blocking(
                handle,
                Duration::from_secs(5),
                fut,
                |msg| format!("network: {msg}"),
                |dur| format!("timeout after {dur:?}"),
                || "panic".to_string(),
            )
        })
        .await
        .expect("spawn_blocking task must not panic");

        assert_eq!(result.unwrap(), "nested-safe");
    }

    // -----------------------------------------------------------------------
    // build_ollama_prompt_fn — provider dispatch (pure construction, no network)
    // -----------------------------------------------------------------------

    /// Without `OLLAMA_API_KEY` the function constructs a local-mode client
    /// and returns the `"ollama-local"` provider prefix.
    #[test]
    fn build_ollama_prompt_fn_local_mode_when_no_api_key() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => None,
                "OLLAMA_HOST" => None, // use default
                _ => None,
            }
        };
        let (_, prefix) = build_ollama_prompt_fn(&env, "scorer").unwrap();
        assert_eq!(prefix, "ollama-local");
    }

    /// With `OLLAMA_API_KEY` set the function selects cloud mode and returns
    /// the `"ollama-cloud"` provider prefix.
    #[test]
    fn build_ollama_prompt_fn_cloud_mode_when_api_key_set() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => Some("fake-cloud-key".to_string()),
                "OLLAMA_BASE_URL" => None, // use default
                _ => None,
            }
        };
        let (_, prefix) = build_ollama_prompt_fn(&env, "scorer").unwrap();
        assert_eq!(prefix, "ollama-cloud");
    }

    /// `OLLAMA_HOST` is respected in local mode (custom host is accepted).
    #[test]
    fn build_ollama_prompt_fn_local_mode_custom_host_accepted() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => None,
                "OLLAMA_HOST" => Some("http://10.0.0.5:11434".to_string()),
                _ => None,
            }
        };
        // Construction must succeed (the client builder accepts any base URL).
        let (_, prefix) = build_ollama_prompt_fn(&env, "scorer")
            .unwrap_or_else(|e| panic!("custom OLLAMA_HOST should be accepted: {e}"));
        assert_eq!(prefix, "ollama-local");
    }

    /// `OLLAMA_BASE_URL` is respected in cloud mode (custom base URL accepted).
    #[test]
    fn build_ollama_prompt_fn_cloud_mode_custom_base_url_accepted() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => Some("key".to_string()),
                "OLLAMA_BASE_URL" => Some("https://my-ollama.example.com/v1".to_string()),
                _ => None,
            }
        };
        let (_, prefix) = build_ollama_prompt_fn(&env, "scorer")
            .unwrap_or_else(|e| panic!("custom OLLAMA_BASE_URL should be accepted: {e}"));
        assert_eq!(prefix, "ollama-cloud");
    }

    // -----------------------------------------------------------------------
    // resolve_ollama_endpoint — base URL resolution order
    // -----------------------------------------------------------------------

    /// Keyless + `OLLAMA_BASE_URL` → the URL is used VERBATIM (no `/v1`
    /// appended, no localhost fallback). This is the vLLM/litellm seat:
    /// the consumer points straight at an OpenAI-compatible `/v1` root.
    #[test]
    fn resolve_ollama_endpoint_keyless_base_url_used_verbatim() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_BASE_URL" => Some("http://172.16.100.195:8000/v1".to_string()),
                _ => None,
            }
        };
        let (base, key, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(base, "http://172.16.100.195:8000/v1");
        assert_eq!(key, OLLAMA_LOCAL_DUMMY_KEY);
        assert_eq!(prefix, "ollama-local");
    }

    /// Keyless, no base URL → `OLLAMA_HOST` with `/v1` appended.
    #[test]
    fn resolve_ollama_endpoint_keyless_host_gets_v1_appended() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_HOST" => Some("http://10.0.0.5:11434/".to_string()),
                _ => None,
            }
        };
        let (base, _, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(base, "http://10.0.0.5:11434/v1");
        assert_eq!(prefix, "ollama-local");
    }

    /// Keyless, nothing set → localhost daemon default.
    #[test]
    fn resolve_ollama_endpoint_keyless_defaults_to_localhost() {
        let env = |_: &str| -> Option<String> { None };
        let (base, _, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(base, "http://localhost:11434/v1");
        assert_eq!(prefix, "ollama-local");
    }

    /// `OLLAMA_BASE_URL` wins over `OLLAMA_HOST` when both are set keyless.
    #[test]
    fn resolve_ollama_endpoint_base_url_wins_over_host() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_BASE_URL" => Some("http://vllm.example:8000/v1".to_string()),
                "OLLAMA_HOST" => Some("http://ignored:11434".to_string()),
                _ => None,
            }
        };
        let (base, _, _) = resolve_ollama_endpoint(&env);
        assert_eq!(base, "http://vllm.example:8000/v1");
    }

    // -----------------------------------------------------------------------
    // env_non_empty — empty/whitespace values are unset
    // -----------------------------------------------------------------------

    #[test]
    fn env_non_empty_returns_trimmed_value() {
        let env = |_: &str| Some("  qwen3:8b  ".to_string());
        assert_eq!(env_non_empty(&env, "VAR"), Some("qwen3:8b".to_string()));
    }

    #[test]
    fn env_non_empty_treats_empty_and_whitespace_as_unset() {
        let empty = |_: &str| Some(String::new());
        assert_eq!(env_non_empty(&empty, "VAR"), None);
        let ws = |_: &str| Some("   \t ".to_string());
        assert_eq!(env_non_empty(&ws, "VAR"), None);
        let absent = |_: &str| None;
        assert_eq!(env_non_empty(&absent, "VAR"), None);
    }

    /// An EMPTY `OLLAMA_BASE_URL` must not suppress a valid `OLLAMA_HOST` —
    /// it counts as unset and resolution falls through to the host.
    #[test]
    fn resolve_ollama_endpoint_empty_base_url_falls_through_to_host() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_BASE_URL" => Some(String::new()),
                "OLLAMA_HOST" => Some("http://10.0.0.5:11434".to_string()),
                _ => None,
            }
        };
        let (base, _, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(base, "http://10.0.0.5:11434/v1");
        assert_eq!(prefix, "ollama-local");
    }

    /// An EMPTY `OLLAMA_API_KEY` must stay keyless (local mode), never cloud
    /// mode with an empty bearer token.
    #[test]
    fn resolve_ollama_endpoint_empty_api_key_stays_keyless() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => Some(String::new()),
                _ => None,
            }
        };
        let (base, key, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(prefix, "ollama-local");
        assert_eq!(key, OLLAMA_LOCAL_DUMMY_KEY);
        assert_eq!(base, "http://localhost:11434/v1");
    }

    /// Whitespace-only values behave identically to empty ones.
    #[test]
    fn resolve_ollama_endpoint_whitespace_values_are_unset() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => Some("  ".to_string()),
                "OLLAMA_BASE_URL" => Some("\t".to_string()),
                "OLLAMA_HOST" => None,
                _ => None,
            }
        };
        let (base, _, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(prefix, "ollama-local");
        assert_eq!(base, "http://localhost:11434/v1");
    }

    /// With `OLLAMA_API_KEY` → cloud mode, real key, cloud default base URL.
    #[test]
    fn resolve_ollama_endpoint_cloud_mode_with_key() {
        let env = |k: &str| -> Option<String> {
            match k {
                "OLLAMA_API_KEY" => Some("real-key".to_string()),
                _ => None,
            }
        };
        let (base, key, prefix) = resolve_ollama_endpoint(&env);
        assert_eq!(base, DEFAULT_OLLAMA_CLOUD_BASE_URL);
        assert_eq!(key, "real-key");
        assert_eq!(prefix, "ollama-cloud");
    }

    /// `build_openrouter_prompt_fn` returns the `"openrouter"` provider prefix.
    #[test]
    fn build_openrouter_prompt_fn_returns_openrouter_prefix() {
        // Any non-empty string is accepted by the client builder without a
        // live network call — the key is only validated on the first request.
        let (_, prefix) = build_openrouter_prompt_fn("fake-key", "scorer").unwrap();
        assert_eq!(prefix, "openrouter");
    }
}
