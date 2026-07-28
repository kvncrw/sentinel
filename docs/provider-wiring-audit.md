# Provider Wiring Audit — how every LLM-consuming component resolves its config

Date: 2026-07-28. Scope: every component in `sentinel-infrastructure` /
`sentinel-application` / `sentinel-cli` that constructs an LLM client, surveyed
after the A13 spec-challenge scorer gained a first-class `ollama` provider.

Motivation: the 2026-07-27 benchmark incident — SessionStart hooks failed
closed because the A13 scorer hard-required `OPENROUTER_API_KEY` even when the
spec gate was ObserveOnly and the seat was a local vLLM serve. `defd9de` fixed
the ordering; this audit maps the remaining provider-resolution sprawl so the
next component doesn't reproduce the same class of failure.

## Component table

| Component | File | Provider selection | Env vars read | Default provider / model | Behavior on missing config |
|---|---|---|---|---|---|
| **A3 dry-run auditor** (`RigAuditor`) | `crates/sentinel-infrastructure/src/dry_run_auditor.rs` | `SENTINEL_AUDITOR_PROVIDER` for `from_env()` (**`ollama` value rejected** — "route through A2"); at runtime the hook dispatcher only uses `via_router`/`for_profile`, where the A2 profile's `VendorClass` picks the transport | `OPENROUTER_API_KEY` (all non-Ollama vendor classes), `OLLAMA_API_KEY`/`OLLAMA_BASE_URL`/`OLLAMA_HOST` (Ollama profiles), `SENTINEL_AUDITOR_MODEL`, `SENTINEL_AUDITOR_TIMEOUT_SECS` | openrouter / `anthropic/claude-opus-4.8` (env path); profile's `model_id` (router path) | **Fail-closed** at hook dispatch: router-config load failure, zero profiles, or client-construction failure all `write_fail_closed_response` |
| **A12 eval scorer** (`LlmEvalScorer`) | `crates/sentinel-infrastructure/src/eval_scorer.rs` | `SENTINEL_EVAL_SCORER_PROVIDER` — **`ollama` value rejected at runtime**; the full ollama impl exists but is `#[cfg(test)]` | `OPENROUTER_API_KEY`, `SENTINEL_EVAL_SCORER_MODEL`, `SENTINEL_EVAL_SCORER_TIMEOUT_SECS` | openrouter / `anthropic/claude-opus-4.8` | **Fail-closed per command**: `sentinel eval` CLI and MCP `eval_run` error out |
| **A13 spec-challenge scorer** (`LlmSpecChallengeScorer`) | `crates/sentinel-infrastructure/src/spec_challenge_scorer.rs` | `SENTINEL_SPEC_CHALLENGE_SCORER_PROVIDER` — `openrouter` (default) or, **since this PR, first-class `ollama`** (any OpenAI-compatible `/v1`: vLLM, litellm, ollama daemon, Ollama Cloud) | `OPENROUTER_API_KEY` (openrouter), `OLLAMA_BASE_URL`/`OLLAMA_HOST`/`OLLAMA_API_KEY` (ollama), `SENTINEL_SPEC_CHALLENGE_SCORER_MODEL` (required for ollama), `SENTINEL_SPEC_CHALLENGE_SCORER_TIMEOUT_SECS` | openrouter / `anthropic/claude-opus-4.7` | **Mode-dependent** (since `defd9de`): fail-closed when spec-challenge mode allows blocking; degrade-to-`None` + warn in ObserveOnly |
| **Adversarial step/phase judge** (`MultiModelJudge` / `JudgeProvider`) | `crates/sentinel-infrastructure/src/rig_judge.rs` | none — OpenRouter only, model set hardcoded in `sentinel-domain/src/judge.rs` (`JudgeModel`: Codex/Sonnet/Opus/Kimi) | `OPENROUTER_API_KEY` only. No model, timeout, or provider override envs | openrouter / hardcoded per-tier model IDs | **Split**: `sentinel mcp` startup errors (fail-closed); hook-side step judge falls back to `NoJudge`, which returns *insufficient* verdicts ("Set OPENROUTER_API_KEY") — fail-closed at proof-seal time, not at dispatch |
| **Skill-router AI classifier** (`RigClassifier`) | `crates/sentinel-infrastructure/src/rig_classifier.rs` | none — OpenRouter only, model hardcoded (`anthropic/claude-opus-4-7`) | `OPENROUTER_API_KEY` only | openrouter / hardcoded | **Fail-open**: `from_env() -> Option`, logs a warning, router degrades to non-AI routing; also raced against an 8 s timeout |
| **Worker delegation + BA/severity orchestrators** (`OpenRouterLlm`) | `crates/sentinel-infrastructure/src/openrouter_llm.rs` (used by MCP `delegate_codex`, `delegate_kimi_context_scan`, `ba_draft`, `severity_scan`) | none — OpenRouter only; worker models hardcoded (`openai/gpt-5.5-pro`, `moonshotai/kimi-k2.6`) | `OPENROUTER_API_KEY` only | openrouter / hardcoded per worker | **Fail-closed per tool call** with a clear "set OPENROUTER_API_KEY" error in the MCP result |
| **Transport layer** (`ChatClient`) | `crates/sentinel-infrastructure/src/llm_http.rs` | n/a — takes `base_url` + `key` as arguments; `openrouter()` is `openai_compat(OPENROUTER_BASE_URL, key)` | none directly | n/a | n/a — this is the one layer that is already uniform |
| **Shared scorer plumbing** | `crates/sentinel-infrastructure/src/llm_scorer_runtime.rs` | `build_openrouter_prompt_fn` / `build_ollama_prompt_fn` (+ `resolve_ollama_endpoint`) | `OLLAMA_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_HOST` | n/a | Construction `Err` propagated to caller |

### Behavior changes shipped with this audit (upgrading operators, read this)

Two semantics changes landed in the shared plumbing alongside the A13
promotion — both apply to **every** consumer of `build_ollama_prompt_fn`
(A3 auditor router path, A13 scorer), not just A13:

1. **`OLLAMA_BASE_URL` now beats `OLLAMA_HOST` in keyless mode.** Previously
   keyless mode ignored `OLLAMA_BASE_URL` entirely and used `OLLAMA_HOST`+`/v1`.
   Now: `OLLAMA_BASE_URL` verbatim → `OLLAMA_HOST`+`/v1` → localhost default.
   A deployment setting BOTH vars keyless reroutes from the host to the base
   URL on upgrade; a one-line `warn!` (userinfo-redacted URLs) fires whenever
   both are set so the precedence is visible.
2. **Empty / whitespace-only env values count as unset** for the provider-
   resolution vars (`env_non_empty`): `OLLAMA_BASE_URL=""` falls through to
   `OLLAMA_HOST`; `OLLAMA_API_KEY=""` stays keyless (no more cloud mode with
   an empty bearer); empty A13 provider → default provider; empty A13
   key/model → same error as missing. Exception: set-but-empty
   `*_TIMEOUT_SECS` remains a hard config error (pre-existing contract).

Related hardening: `ChatClient` error strings redact URL userinfo
(`redact_url_userinfo`) so config-embedded credentials cannot leak into
ObserveOnly logs or fail-closed hook responses.

Non-LLM env consumers checked and excluded: `linear_lookup` / `linear_enforcer`
(`LINEAR_API_KEY`), `memory_provision` (`QDRANT_API_KEY`), `mcp_guardian`
(secret-reference healing), `evidence_browserbase`.

## Findings

1. **Three parallel provider-dispatch implementations.** A3, A12, and A13 each
   carry a near-identical `from_env_with` → `{openrouter,ollama}_from_env_with`
   block with copy-pasted error strings. They have already drifted: A13 now
   accepts `provider=ollama`, A12 rejects it with its ollama impl dead behind
   `#[cfg(test)]`, and A3 rejects it with a "route through A2" message. Same
   env-var *shape*, three different behaviors.
2. **Two config authorities for the same question.** A3 resolves provider via
   the A2 capability-router TOML profiles; A12/A13 resolve via env vars. An
   operator pointing the estate at a local seat has to know which mechanism
   each gate uses.
3. **Key-presence changes routing semantics.** `OLLAMA_API_KEY` presence flips
   local↔cloud mode *and* changes which base-URL default applies. Before this
   PR, keyless mode silently ignored `OLLAMA_BASE_URL` — exactly the "dumb
   config shit" class: the var looked honored but wasn't. Fixed here
   (`resolve_ollama_endpoint`), but the flip-on-key-presence design remains a
   footgun.
4. **Fail behavior on missing config is per-component folklore.** Fail-closed
   at dispatch (A3), fail-closed per command (A12), mode-dependent (A13),
   fail-open with warning (classifier), fail-closed-at-seal via `NoJudge`
   (judge). None of this is discoverable except by reading each call site;
   the benchmark incident happened precisely because A13's behavior was the
   strictest one in the table at the time.
5. **Hardcoded models with no override** in the judge, classifier, and worker
   paths. Fine for the adversarial-tier design intent, but it means "run
   sentinel fully local" is impossible for those components today, and any
   OpenRouter outage or account issue takes them down with no escape hatch.

## Recommendation

Introduce one shared **`ProviderConfig` resolution module** (natural home:
`sentinel-infrastructure/src/llm_scorer_runtime.rs` growing into
`llm_provider.rs`) and migrate call sites to it incrementally:

1. **One resolution function, namespaced env vars.**
   `ProviderConfig::resolve(namespace, env) -> Result<ProviderConfig>` reading
   `SENTINEL_<NS>_PROVIDER` / `SENTINEL_<NS>_MODEL` / `SENTINEL_<NS>_TIMEOUT_SECS`
   plus the shared credential vars (`OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`,
   `OLLAMA_API_KEY`, `OLLAMA_HOST`). A3/A12/A13 become one-liners over it, and
   a new gate gets both providers for free instead of forking the block a
   fourth time. Error strings come from one place and always name the exact
   env var + provider + namespace.
2. **Uniform provider enum.** `openrouter | ollama` (ollama = any
   OpenAI-compatible `/v1`; vLLM and litellm are first-class targets). The
   judge/classifier/worker paths keep their model pinning but should accept
   the same enum so a local seat is at least *possible* behind an explicit
   opt-in.
3. **Declared, not implied, fail behavior.** Give the shared module an
   explicit `FailPolicy { Closed, Open, ModeDependent }` chosen by the caller
   and logged at construction, so "what happens if the key is missing" is
   grep-able in one place instead of five. The A13 ObserveOnly rule
   (`defd9de`) generalizes: *components that cannot act on their output in the
   current mode must not fail closed on missing provider config.*
4. **Kill the key-presence mode flip.** Make local vs cloud explicit
   (`SENTINEL_<NS>_PROVIDER=ollama` + `OLLAMA_BASE_URL` always wins when set;
   `OLLAMA_API_KEY` only adds auth) rather than letting a credential's
   existence silently re-route traffic.
5. **Sequencing.** (a) extract the shared resolver + move A12's ollama path
   out of `#[cfg(test)]` using it; (b) port A13 and A3's env path; (c) let A2
   profiles *reference* a named ProviderConfig instead of embedding their own
   vendor→env mapping; (d) judge/classifier/worker opt-in last.

This is deliberately a report, not a refactor — no call sites were changed
beyond the A13 scorer promotion and the `OLLAMA_BASE_URL` keyless fix that
motivated it.
