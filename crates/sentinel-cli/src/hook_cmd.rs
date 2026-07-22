//! `sentinel hook` — Process hook events through the LangGraph authority path.

use std::collections::HashMap;
use std::io::Write as _;

use anyhow::{anyhow, Context, Result};
use tracing::debug;

use sentinel_application::hook_metrics::{time_and_record, InvocationContext};
use sentinel_application::hooks;
use sentinel_domain::events::{HookEvent, HookOutput, HookSpecificOutput};
use sentinel_domain::state::SessionState;
use sentinel_domain::workflow::{SkillSteps, SkillWorkflow};

use sentinel_domain::capability::VendorClass;
use sentinel_domain::ports::{AuditorPort, LlmPort, ReversibilityClassifierPort, VectorStorePort};
use sentinel_infrastructure::ba_config::BaEnforcementConfig;
use sentinel_infrastructure::capability_router::TomlCapabilityRouter;
use sentinel_infrastructure::dry_run_auditor::RigAuditor;
use sentinel_infrastructure::git::RealGit;
use sentinel_infrastructure::memory_mcp_client::MemoryMcpClient;
use sentinel_infrastructure::provenance_store::JsonlProvenanceStore;
use sentinel_infrastructure::requirement_matrix::FilesystemRequirementMatrix;
use sentinel_infrastructure::reversibility::LayeredReversibilityClassifier;
use sentinel_infrastructure::spec_challenge_config::SpecChallengeConfig;
use sentinel_infrastructure::spec_challenge_scorer::LlmSpecChallengeScorer;
use sentinel_infrastructure::spec_challenge_store::FilesystemSpecChallengeStore;
use std::sync::Arc;

use crate::phase_graph_projection::{
    activate_phase_graph_workflow, load_workflow_configs, project_phase_graph_workflows,
};

fn required_graph_tool<'a>(
    input: &'a sentinel_domain::events::HookInput,
    evaluation_tool: Option<&'a str>,
) -> Result<&'a str> {
    evaluation_tool
        .or(input.tool_name.as_deref())
        .map(str::trim)
        .filter(|tool| !tool.is_empty())
        .ok_or_else(|| anyhow!("LangGraph authority requires concrete tool identity"))
}

fn required_graph_session<'a>(
    input: &'a sentinel_domain::events::HookInput,
    evaluation_session: Option<&'a str>,
) -> Result<&'a str> {
    evaluation_session
        .or(input.session_id.as_deref())
        .map(str::trim)
        .filter(|session| !session.is_empty())
        .ok_or_else(|| anyhow!("LangGraph authority requires concrete session identity"))
}

macro_rules! require_hook_graph_authorization {
    ($authorization:expr, $label:literal) => {
        match $authorization {
            Ok(Some(authorization)) => authorization,
            Ok(None) => {
                return HookOutput::deny(format!(
                    "{} LangGraph authority did not produce a terminal checkpoint",
                    $label
                ));
            }
            Err(err) => {
                return HookOutput::deny(format!(
                    "{} LangGraph authority checkpoint validation failed: {err}",
                    $label
                ));
            }
        }
    };
}

pub async fn run(event: &str, matcher: Option<&str>) -> Result<()> {
    run_internal(event, matcher).await
}

fn load_configured_skill_steps(
    config_dir: &std::path::Path,
    workflows: &HashMap<String, SkillWorkflow>,
    skill: &str,
) -> Result<Option<SkillSteps>> {
    if !workflows.contains_key(skill) {
        return Ok(None);
    }

    match sentinel_infrastructure::config::load_skill_steps(config_dir, skill)
        .with_context(|| format!("failed to load step config for configured workflow '{skill}'"))?
    {
        Some(steps) => Ok(Some(steps)),
        None => Err(anyhow!(
            "configured LangGraph workflow '{skill}' is missing required step config '{}'",
            config_dir
                .join("steps")
                .join(format!("{skill}.toml"))
                .display()
        )),
    }
}

async fn run_internal(event: &str, matcher: Option<&str>) -> Result<()> {
    let start_time = std::time::Instant::now();
    debug!(event, ?matcher, "Processing hook event");

    // Parse event type
    // **Attack #120 fix**: Fail closed on unknown event types instead of defaulting
    // to Stop. Silently treating an unknown event as Stop would run Stop-phase hooks
    // (execution_log, skill_telemetry, context_monitor, etc.) which could confuse
    // state tracking. Better to error out so the issue is immediately visible.
    let hook_event = parse_hook_event(event)?;

    // **Stdin authentication**: Validate that we're being invoked by a plausible
    // caller (Claude Code / node process). This is defense-in-depth — not a hard
    // guarantee, but raises the bar from "any process on the system" to "a process
    // that can convincingly mimic Claude Code's invocation pattern".
    if let Err(e) = validate_caller() {
        // Caller validation failed — return safe empty JSON and exit early.
        debug!(error = %e, "Caller validation failed — returning safe response");
        return write_safe_allow_response();
    }

    // Read input from stdin
    let input = sentinel_infrastructure::stdin::read_hook_input()?;

    // Load the authoritative workflow catalog. Missing or unreadable
    // workflows.toml is a hard error; treating it as an empty catalog would
    // silently remove LangGraph workflow enforcement.
    let config_dir = sentinel_infrastructure::config::config_dir();
    let workflows: HashMap<String, SkillWorkflow> = load_workflow_configs()?;

    // **Attack #67 fix**: Acquire session lock BEFORE loading state.
    // Hold through processing + save to prevent concurrent hook invocations
    // from overwriting each other's state changes (lost updates).
    //
    // **Attack #128 note**: Lock safety on panic — `_session_lock` is a
    // `std::fs::File` handle with fs2 advisory lock. Rust's Drop trait
    // guarantees the file handle is closed on unwind (panic), which releases
    // the advisory lock. No manual cleanup needed.
    // **Bug fix (2026-05-06)**: Previously fell back to "unknown" silently when
    // session_id was missing — which meant CLI invocations from bash (no stdin
    // HookInput) would do work against a synthetic "unknown" session, never
    // matching the active Claude Code session. State writes/reads landed in the
    // wrong place (ghost session_id), and tasks.md regeneration appeared to
    // succeed but operated on the wrong session's task store.
    //
    // Now: prefer input.session_id, then the env-var chain (Claude Code exports
    // CLAUDE_CODE_SESSION_ID into MCP/hook children — verified in the
    // deobfuscated 2.1.201 bundle; the wrapper/cron contexts may set
    // VULCAN_SESSION_ID / CLAUDE_SESSION_ID). Must match the same precedence
    // channel_events uses, or the dispatcher resolves a different id than the
    // event producer. Then fail loudly instead of corrupting state under
    // "unknown".
    let session_id_owned: String = match input.session_id.as_deref() {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => match [
            "VULCAN_SESSION_ID",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_SESSION_ID",
            "SESSION_ID",
        ]
        .into_iter()
        .find_map(|k| std::env::var(k).ok().filter(|s| !s.is_empty()))
        {
            Some(s) => s,
            None => {
                eprintln!(
                    "[sentinel] hook: no session_id available — pass via stdin HookInput.session_id or set CLAUDE_CODE_SESSION_ID. Refusing to operate against synthetic 'unknown' session."
                );
                // Return safe empty JSON so callers don't crash, but do not
                // proceed with state mutations.
                println!("{{}}");
                return Ok(());
            }
        },
    };
    let session_id: &str = &session_id_owned;

    // **Rate limiting**: Check per-session invocation rate BEFORE acquiring session lock.
    // This prevents flood attacks from even contending for the lock, reducing DoS impact.
    sentinel_infrastructure::rate_limit::check_rate_limit(session_id)?;

    let _session_lock = sentinel_infrastructure::state_store::acquire_session_lock(session_id)?;
    let mut state = match sentinel_infrastructure::state_store::load(session_id) {
        Ok(Some(state)) => state,
        Ok(None) => SessionState::new(session_id),
        Err(err) => {
            return write_fail_closed_response(
                hook_event,
                format!("session state load failed: {err:#}"),
            );
        }
    };

    if matches!(
        hook_event,
        HookEvent::UserPromptSubmit | HookEvent::PreToolUse
    ) {
        if let Err(e) = project_phase_graph_workflows(&mut state, &workflows).await {
            return write_fail_closed_response(
                hook_event,
                format!("phase graph checkpoint projection failed: {e}"),
            );
        }
    }

    // Load step configs lazily, but fail closed for configured workflows. A
    // missing or malformed step plan must not make step_gate/phase_validator run
    // as if no step authority exists.
    let mut step_configs: HashMap<String, SkillSteps> = HashMap::new();
    if let Some(skill) = state.active_skill.as_deref() {
        match load_configured_skill_steps(&config_dir, &workflows, skill) {
            Ok(Some(steps)) => {
                step_configs.insert(skill.to_string(), steps);
            }
            Ok(None) => {}
            Err(e) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("step config load failed: {e:#}"),
                );
            }
        }
    }

    let git = RealGit;

    // Construct vector store adapter (None if Qdrant not configured)
    let vector_store: Option<Arc<dyn VectorStorePort>> =
        sentinel_infrastructure::qdrant::QdrantAdapter::from_config()
            .map(|a| Arc::new(a) as Arc<dyn VectorStorePort>);

    // Construct filesystem, process, and env adapters
    let real_fs = sentinel_infrastructure::filesystem::RealFileSystem;
    let real_process = sentinel_infrastructure::process::RealProcess;
    let real_env = sentinel_infrastructure::env::RealEnv;

    // Construct LLM adapter. STANDARDIZED on Rig + OpenRouter
    // (`OPENROUTER_API_KEY`) — the same gateway the adversarial judge uses.
    // No direct-vendor SDK path. `None` if the key is unset.
    let llm: Option<Arc<dyn LlmPort>> =
        sentinel_infrastructure::openrouter_llm::OpenRouterLlm::from_env()
            .ok()
            .map(|c| Arc::new(c) as Arc<dyn LlmPort>);

    // Construct memory-mcp client. Unset env vars use shipped defaults; invalid
    // env config fails closed instead of silently disabling or retiming memory.
    let memory_mcp = match MemoryMcpClient::from_env() {
        Ok(client) => client,
        Err(err) => {
            return write_fail_closed_response(
                hook_event,
                format!("failed to configure memory-mcp client: {err:#}"),
            );
        }
    };

    // A6 Phase 4a: construct the layered reversibility classifier from the
    // shipped `config/reversibility-defaults.toml`. This is an enterprise gate:
    // config load failures must block instead of making the classifier inert.
    let reversibility_classifier = match LayeredReversibilityClassifier::with_shipped_defaults() {
        Ok(classifier) => classifier,
        Err(err) => {
            return write_fail_closed_response(
                hook_event,
                format!("failed to load shipped reversibility defaults: {err}"),
            );
        }
    };

    // A2 Phase 4: construct the capability router from shipped
    // defaults + optional operator overrides at
    // `~/.claude/sentinel/config/agents.toml`. Load failures block because
    // A2 routing is the authority for enterprise auditor selection.
    let agents_overrides_path = config_dir.join("agents.toml");
    let capability_router =
        match TomlCapabilityRouter::with_shipped_and_overrides(Some(&agents_overrides_path)) {
            Ok(r) => r,
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("failed to load Sentinel capability router config: {err}"),
                );
            }
        };

    // A3 Phase 4 + A2 Phase 4: construct the dry-run auditor through the
    // capability router. No env-only substitute: A2 is the enterprise routing
    // authority and A3 must not silently downgrade to historical selection.
    if capability_router.profiles().is_empty() {
        return write_fail_closed_response(
            hook_event,
            "Sentinel capability router loaded zero agent profiles; refusing to run without A2 routing",
        );
    }
    let auditor: Arc<dyn AuditorPort> = match RigAuditor::via_router(
        &capability_router,
        capability_router.profiles(),
        VendorClass::Anthropic,
    ) {
        Ok(a) => {
            tracing::info!(
                "A3 auditor selected via A2 capability router (acting vendor: Anthropic)"
            );
            Arc::new(a) as Arc<dyn AuditorPort>
        }
        Err(err) => {
            return write_fail_closed_response(
                hook_event,
                format!("failed to select A3 auditor via capability router: {err}"),
            );
        }
    };

    // BA1+3 Phase 4c: construct provenance store + requirement matrix
    // adapters at session start. Construction failures block: these are the
    // enterprise BA proof substrates, not optional side layers.
    let provenance_store: Arc<JsonlProvenanceStore> =
        match JsonlProvenanceStore::with_default_path() {
            Ok(s) => Arc::new(s),
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("failed to initialize BA provenance store: {err}"),
                );
            }
        };
    let requirement_matrix: Arc<FilesystemRequirementMatrix> =
        match FilesystemRequirementMatrix::with_default_path() {
            Ok(m) => Arc::new(m),
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("failed to initialize BA requirement matrix: {err}"),
                );
            }
        };
    // A13 spec-challenge store — persists the agent's SpecChallenge for
    // proof-chain re-verification. Persistence is required for the gate to
    // be auditable, so initialization failures block.
    let spec_challenge_store: Arc<FilesystemSpecChallengeStore> =
        match FilesystemSpecChallengeStore::with_default_path() {
            Ok(s) => Arc::new(s),
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("failed to initialize A13 spec-challenge store: {err}"),
                );
            }
        };
    // A13 semantic scorer — required for Catastrophic-class work and
    // StrictBlocking. Do not let missing scorer env silently make A13 inert.
    let spec_challenge_scorer: Arc<LlmSpecChallengeScorer> =
        match LlmSpecChallengeScorer::from_env() {
            Ok(s) => Arc::new(s),
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("failed to initialize A13 semantic scorer: {err}"),
                );
            }
        };
    // A13 enforcement config — shipped default is DefaultBlocking,
    // overridable via
    // ~/.claude/sentinel/config/spec-challenge.toml. Mirrors the
    // BA1+3 ba-enforcement config below.
    let spec_challenge_overrides_path = config_dir.join("spec-challenge.toml");
    let spec_challenge_config =
        match SpecChallengeConfig::with_shipped_and_overrides(Some(&spec_challenge_overrides_path))
        {
            Ok(c) => c,
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("spec-challenge.toml load failed: {err}"),
                );
            }
        };
    // Load BA1+3 enforcement config: shipped DefaultBlocking
    // defaults overridden by operator-supplied
    // ~/.claude/sentinel/config/ba-enforcement.toml.
    let ba_enforcement_overrides_path = config_dir.join("ba-enforcement.toml");
    let ba_enforcement =
        match BaEnforcementConfig::with_shipped_and_overrides(Some(&ba_enforcement_overrides_path))
        {
            Ok(c) => c,
            Err(err) => {
                return write_fail_closed_response(
                    hook_event,
                    format!("ba-enforcement.toml load failed: {err}"),
                );
            }
        };

    // Real-time single-issue Linear lookup for the PM gate. Missing token is
    // represented as `None`; targeted Linear start attempts fail closed inside
    // the gate instead of consulting stale local assignment data.
    let linear_lookup = match sentinel_infrastructure::linear_lookup::LinearLookup::from_env() {
        Ok(lookup) => lookup,
        Err(err) => {
            return write_fail_closed_response(
                hook_event,
                format!("Linear lookup initialization failed: {err}"),
            );
        }
    };

    // Bundle all ports into HookContext
    let ctx = hooks::HookContext {
        git: &git,
        vector_store: vector_store.as_deref(),
        fs: &real_fs,
        process: &real_process,
        llm: llm.as_deref(),
        memory_mcp: &memory_mcp,
        env: &real_env,
        linear_lookup: linear_lookup
            .as_ref()
            .map(|l| l as &dyn hooks::LinearLookupPort),
    };

    // Process through matching hooks based on event type
    let mut output = HookOutput::allow();

    // Resolved once per hook event so the per-call `time_and_record`
    // wrapper can stamp the JSONL row without each hook re-running git.
    let cwd_for_metrics = input.cwd.as_deref().unwrap_or(".");
    let repo_root_for_metrics = ctx.git.repo_root(cwd_for_metrics);

    match hook_event {
        HookEvent::UserPromptSubmit => {
            let arm_output = handle_user_prompt_submit(
                &input,
                &mut state,
                &ctx,
                &workflows,
                &step_configs,
                repo_root_for_metrics.as_deref(),
            )
            .await;
            output.merge(&arm_output);
        }

        HookEvent::PreToolUse => {
            let arm_output = handle_pre_tool_use(
                &input,
                &mut state,
                &ctx,
                &git,
                &reversibility_classifier,
                auditor.as_ref(),
                true,
                Some(provenance_store.as_ref()),
                Some(requirement_matrix.as_ref()),
                Some(spec_challenge_store.as_ref()),
                Some(spec_challenge_scorer.as_ref()),
                spec_challenge_config,
                &ba_enforcement,
                repo_root_for_metrics.as_deref(),
                &workflows,
                &step_configs,
            );
            output.merge(&arm_output);
        }

        HookEvent::PostToolUse => {
            let arm_output = handle_post_tool_use(
                &input,
                &mut state,
                &ctx,
                &step_configs,
                Some(provenance_store.as_ref()),
            )
            .await;
            output.merge(&arm_output);
        }

        HookEvent::Stop => {
            let arm_output = handle_stop(&input, &ctx, &state);
            output.merge(&arm_output);
        }

        HookEvent::SessionStart => {
            // **Attack #57 fix**: Only create fresh state if no existing state was loaded.
            // Unconditional reset lets an attacker trigger SessionStart mid-session
            // (via crafted event) to wipe all workflow progress and phase gates.
            // If state already exists (loaded from disk at line 64), preserve it.
            if state.tool_calls == 0
                && !state.has_any_graph_workflow()
                && state.phases_read.is_empty()
            {
                // Genuinely new session — use fresh state (already created above)
                state = SessionState::new(session_id);
            } else {
                debug!(
                    session_id,
                    tool_calls = state.tool_calls,
                    workflows = state.graph_workflow_count(),
                    "SessionStart received for active session — preserving existing state"
                );
            }

            let mk_ctx = |hook: &'static str| InvocationContext {
                event: "SessionStart",
                hook,
                tool: None,
                session_id: input.session_id.as_deref(),
                repo_root: repo_root_for_metrics.as_deref(),
            };

            // Session init — log session, sync marketplace repo, inject startup context
            let init_output = time_and_record(ctx.fs, &mk_ctx("session_init"), || {
                hooks::session_init::process(&input, &ctx)
            });
            output.merge(&init_output);

            // Memory provision — verify the memory engine binaries, provision the
            // Qdrant credential from Doppler (fetch when missing/invalid/expired),
            // and idempotently register the memory MCP server in ~/.claude.json.
            // Depends on session-init. Never denies; never installs.
            let provision_output = time_and_record(ctx.fs, &mk_ctx("memory_provision"), || {
                hooks::memory_provision::process(&input, &ctx)
            });
            output.merge(&provision_output);

            // Task rehydrate — inject persistent tasks from previous sessions
            let rehydrate_output = time_and_record(ctx.fs, &mk_ctx("task_rehydrate"), || {
                hooks::task_rehydrate::process(&input, &ctx)
            });
            output.merge(&rehydrate_output);

            // Memory verify removed from SessionStart — Qdrant HTTP calls blocked
            // startup for 5-20s. Verification now runs on PreCompact (background,
            // non-critical path) where latency doesn't affect user experience.

            // Dependency freshness check — detect outdated deps (any language)
            let dep_output = time_and_record(ctx.fs, &mk_ctx("dep_check"), || {
                hooks::dep_check::process(&input, &ctx)
            });
            output.merge(&dep_output);

            // Upstream freshness — notify (never mutate) if local default
            // branch is behind origin. Zero-network: compares against the
            // already-fetched ref, so no latency/auth on the critical path.
            let freshness_output = time_and_record(ctx.fs, &mk_ctx("upstream_freshness"), || {
                hooks::upstream_freshness::process(&input, &ctx, HookEvent::SessionStart)
            });
            output.merge(&freshness_output);
        }

        HookEvent::PreCompact => {
            let mk_ctx = |hook: &'static str| InvocationContext {
                event: "PreCompact",
                hook,
                tool: None,
                session_id: input.session_id.as_deref(),
                repo_root: repo_root_for_metrics.as_deref(),
            };

            // Pre-compact snapshot — save session state before context compaction
            let compact_output = time_and_record(ctx.fs, &mk_ctx("pre_compact"), || {
                hooks::pre_compact::process(&input, &ctx)
            });
            output.merge(&compact_output);

            // Session index — upsert transcript exchanges to Qdrant for search
            let index_output = time_and_record(ctx.fs, &mk_ctx("session_index"), || {
                hooks::session_index::process(&input, &ctx)
            });
            output.merge(&index_output);

            // Memory verify — re-check stored memories against ground truth
            // (24h cooldown, 3s wall-clock timeout, skips without Qdrant +
            // LLM). Moved here from SessionStart on 2026-04 because
            // verification blocked startup 5-20s; PreCompact is the right
            // home — background, non-critical, runs once per long session.
            let verify_output = time_and_record(ctx.fs, &mk_ctx("memory_verify"), || {
                hooks::memory_verify::process(&input, &ctx)
            });
            output.merge(&verify_output);
        }

        HookEvent::TeammateIdle => {
            // Team quality gate — remind teammate to check for remaining work
            let idle_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "TeammateIdle",
                    hook: "teammate_idle",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::teammate_idle::process(&input, &ctx),
            );
            output.merge(&idle_output);
        }

        HookEvent::TaskCompleted => {
            let mk_ctx = |hook: &'static str| InvocationContext {
                event: "TaskCompleted",
                hook,
                tool: None,
                session_id: input.session_id.as_deref(),
                repo_root: repo_root_for_metrics.as_deref(),
            };

            // Task verification gate — verify work before marking complete
            let completed_output = time_and_record(ctx.fs, &mk_ctx("task_completed"), || {
                hooks::task_completed::process(&input, &ctx)
            });
            output.merge(&completed_output);

            // Task persist — snapshot task list to persistent markdown + JSON
            let persist_output = time_and_record(ctx.fs, &mk_ctx("task_persist"), || {
                hooks::task_persist::process(&input, &ctx)
            });
            output.merge(&persist_output);
        }

        // ── New events added from Claude Code v2.1.88 source analysis ──
        HookEvent::SessionEnd => {
            let mk_ctx = |hook: &'static str| InvocationContext {
                event: "SessionEnd",
                hook,
                tool: None,
                session_id: input.session_id.as_deref(),
                repo_root: repo_root_for_metrics.as_deref(),
            };

            // Session cleanup — flush state, log session end (1.5s timeout!)
            let end_output = time_and_record(ctx.fs, &mk_ctx("session_end"), || {
                hooks::session_end::process(&input, &ctx)
            });
            output.merge(&end_output);

            // Session summary — write a prose "where we left off" note beside the
            // persisted task graph so the next SessionStart resumes with context
            // (read back by task_rehydrate). Fail-open; one git log + one write.
            let summary_output = time_and_record(ctx.fs, &mk_ctx("session_summary"), || {
                hooks::session_summary::process(&input, &ctx)
            });
            output.merge(&summary_output);
        }

        HookEvent::PostCompact => {
            // Restore critical state after context compaction
            let compact_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "PostCompact",
                    hook: "post_compact",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::post_compact::process(&input, &ctx),
            );
            output.merge(&compact_output);
        }

        HookEvent::SubagentStart => {
            // Inject skill context into spawned agents
            let subagent_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "SubagentStart",
                    hook: "subagent_start",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::subagent_start::process(&input, &ctx),
            );
            output.merge(&subagent_output);
        }

        HookEvent::SubagentStop => {
            // Log agent completion for telemetry
            let subagent_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "SubagentStop",
                    hook: "subagent_stop",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::subagent_stop::process(&input, &ctx),
            );
            output.merge(&subagent_output);
        }

        HookEvent::TaskCreated => {
            let mk_ctx = |hook: &'static str| InvocationContext {
                event: "TaskCreated",
                hook,
                tool: None,
                session_id: input.session_id.as_deref(),
                repo_root: repo_root_for_metrics.as_deref(),
            };

            // Log task creation for telemetry
            let task_output = time_and_record(ctx.fs, &mk_ctx("task_created"), || {
                hooks::task_created::process(&input, &ctx)
            });
            output.merge(&task_output);

            // Task persist — snapshot task list to persistent markdown + JSON
            let persist_output = time_and_record(ctx.fs, &mk_ctx("task_persist"), || {
                hooks::task_persist::process(&input, &ctx)
            });
            output.merge(&persist_output);
        }

        HookEvent::Setup => {
            // Repo init/maintenance
            let setup_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "Setup",
                    hook: "setup",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::setup::process(&input, &ctx),
            );
            output.merge(&setup_output);
        }

        HookEvent::CwdChanged => {
            // Working directory changed — re-detect project context
            let cwd_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "CwdChanged",
                    hook: "cwd_changed",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::cwd_changed::process(&input, &ctx),
            );
            output.merge(&cwd_output);

            // Upstream freshness — notify if the repo we just switched into has
            // a local default branch behind origin. Same zero-network check.
            let freshness_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "CwdChanged",
                    hook: "upstream_freshness",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::upstream_freshness::process(&input, &ctx, HookEvent::CwdChanged),
            );
            output.merge(&freshness_output);
        }

        HookEvent::StopFailure => {
            // API error at end of turn — log for diagnostics
            let failure_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "StopFailure",
                    hook: "stop_failure",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::stop_failure::process(&input, &ctx),
            );
            output.merge(&failure_output);
        }

        HookEvent::PermissionDenied => {
            // Auto-mode denied a tool call — log for diagnostics
            let denied_output = time_and_record(
                ctx.fs,
                &InvocationContext {
                    event: "PermissionDenied",
                    hook: "permission_denied",
                    tool: None,
                    session_id: input.session_id.as_deref(),
                    repo_root: repo_root_for_metrics.as_deref(),
                },
                || hooks::permission_denied::process(&input, &ctx),
            );
            output.merge(&denied_output);
        }

        HookEvent::PostToolUseFailure => {
            // Tool execution failed — log for diagnostics
            let tool_name = input.tool_name.as_deref().unwrap_or("unknown");
            let is_timeout = input
                .extra
                .get("is_timeout")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let error = input
                .extra
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tracing::debug!(tool_name, is_timeout, error, "Tool execution failed");

            let metrics_dir = sentinel_infrastructure::paths::home_root_or_fatal()
                .join(".claude")
                .join("sentinel")
                .join("metrics");
            let entry = serde_json::json!({
                "event": "tool_failure",
                "tool_name": tool_name,
                "is_timeout": is_timeout,
                "error": error,
                "session_id": input.session_id,
                "ts": chrono::Utc::now().to_rfc3339(),
            });
            if std::fs::create_dir_all(&metrics_dir).is_ok() {
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(metrics_dir.join("errors.jsonl"))
                {
                    use std::io::Write;
                    let _ = writeln!(file, "{entry}");
                }
            }
        }

        HookEvent::PermissionRequest => {
            // Log permission request details for future auto-approve rules.
            // Currently pass-through — no auto-decisions. The value of this hook
            // will come when we add specific auto-approve rules for trusted tools
            // in certain contexts (e.g., auto-allow Edit in a known project dir).
            let tool = input.tool_name.as_deref().unwrap_or("unknown");
            let has_suggestions = input
                .permission_suggestions
                .as_ref()
                .map_or(0, std::vec::Vec::len);
            tracing::debug!(
                tool,
                has_suggestions,
                "PermissionRequest — pass through (no auto-decisions yet)"
            );
        }

        HookEvent::Elicitation => {
            // MCP server requesting user input — log details, pass through.
            // Auto-responding to elicitation without understanding the context is risky.
            // Future: auto-accept known servers (e.g., sentinel, codex) for trusted prompts.
            let server = input
                .extra
                .get("mcp_server_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let message = input
                .extra
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tracing::debug!(
                server,
                message,
                "Elicitation request from MCP server — pass through"
            );
        }

        HookEvent::ElicitationResult => {
            // Post-elicitation response — pass through for now
            tracing::debug!("ElicitationResult received — pass through");
        }

        HookEvent::ConfigChange => {
            // Settings or skill file changed — validate and warn on dangerous changes.
            let source = input
                .extra
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let file_path = input
                .extra
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tracing::debug!(source, file_path, "ConfigChange detected");

            // Warn if disableAllHooks is set (kill-switch that disables all enforcement)
            if (source == "user_settings"
                || source == "project_settings"
                || source == "local_settings")
                && !file_path.is_empty()
            {
                if let Ok(settings_content) = std::fs::read_to_string(file_path) {
                    if settings_content.contains("\"disableAllHooks\"")
                        && settings_content.contains("true")
                    {
                        output.system_message = Some(
                                "[sentinel] WARNING: disableAllHooks detected in settings — all hook enforcement will be disabled!".to_string()
                            );
                    }
                }
            }

            // Log skill file changes for telemetry
            if source == "skills" {
                tracing::info!(file_path, "Skill file changed");
            }
        }

        HookEvent::InstructionsLoaded => {
            // CLAUDE.md or other instruction file loaded — log details.
            let file_path = input
                .extra
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let memory_type = input
                .extra
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let load_reason = input
                .extra
                .get("load_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tracing::debug!(file_path, memory_type, load_reason, "Instructions loaded");

            // Log managed/enterprise overrides — these can silently change behavior
            if memory_type == "Managed" {
                tracing::info!(
                    file_path,
                    "Managed (enterprise) instructions loaded — may override user settings"
                );
            }
        }

        HookEvent::FileChanged => {
            // Watched file changed — log and inject context for important files.
            let file_path = input
                .file_path
                .as_deref()
                .or_else(|| input.extra.get("file_path").and_then(|v| v.as_str()))
                .unwrap_or("");
            let event_type = input
                .extra
                .get("event")
                .and_then(|v| v.as_str())
                .unwrap_or("change");
            tracing::info!(file_path, event_type, "Watched file changed");

            if file_path.ends_with("CLAUDE.md") {
                output.system_message =
                    Some("[sentinel] CLAUDE.md changed — context may need refresh".to_string());
            } else if file_path.ends_with("settings.json") {
                output.system_message = Some(
                    "[sentinel] settings.json changed — hook configuration may have been updated"
                        .to_string(),
                );
            }
        }

        HookEvent::WorktreeCreate => {
            // Git worktree created — pass through for now
            tracing::debug!("WorktreeCreate received — pass through");
        }

        HookEvent::WorktreeRemove => {
            // Git worktree removed — pass through for now
            tracing::debug!("WorktreeRemove received — pass through");
        }

        HookEvent::Notification => {
            // Internal notification — pass through for now
            tracing::debug!("Notification received — pass through");
        }
    }

    // Record hook invocation with actual elapsed time
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    state.record_hook_invocation(event, elapsed_ms);

    // Save state AFTER all processing so phase reads and tool calls are durable.
    // Continuing after a failed write would let the next hook evaluate stale
    // gate state, so this is a hard failure.
    if let Err(e) = sentinel_infrastructure::state_store::save(&mut state) {
        tracing::error!(error = %e, "failed to persist hook state");
        return write_fail_closed_response(
            hook_event,
            format!("failed to persist hook state: {e}"),
        );
    }

    // Transform output for Claude Code's JSON schema
    match hook_event {
        HookEvent::PreToolUse => {
            // Transform blocked/reason into hookSpecificOutput with permissionDecision.
            output = output.into_pretool_output();
        }
        HookEvent::UserPromptSubmit
        | HookEvent::PostToolUse
        | HookEvent::PostToolUseFailure
        | HookEvent::SessionStart
        | HookEvent::Setup
        | HookEvent::SubagentStart
        | HookEvent::Notification
        | HookEvent::PermissionRequest
        | HookEvent::PermissionDenied
        | HookEvent::Elicitation
        | HookEvent::ElicitationResult
        | HookEvent::CwdChanged
        | HookEvent::FileChanged
        | HookEvent::WorktreeCreate
        // Stop / SubagentStop carry context-injection from the reality-check,
        // self-annealing, good-citizen and hygiene-reminder hooks via
        // `hookSpecificOutput.additionalContext` (Claude Code surfaces it on the
        // next turn). They were previously absent here, so the `_ =>` arm below
        // STRIPPED that context — silently dropping every Stop-time reminder.
        // (Found by the E2E hook harness; this was the real reason a false-done
        // reality-check never reached the agent.)
        | HookEvent::Stop
        | HookEvent::SubagentStop => {
            // These events support hookSpecificOutput natively
        }
        _ => {
            // Strip hookSpecificOutput for events Claude Code doesn't process
            output.hook_specific_output = None;
        }
    }

    // Inject project context only for prompt-scoped events. Adding it to every
    // tool hook bloats the transcript during tool-heavy sessions and drives
    // premature compaction.
    if should_attach_project_context(hook_event) {
        // Claude Code exports CLAUDE_PROJECT_DIR (an absolute path), not
        // CLAUDE_PROJECT (a name) — the latter is never set, so the old branch
        // was dead (verified in the deobfuscated 2.1.201 bundle). Derive the
        // project name from the dir's basename.
        if let Some(project) = std::env::var("CLAUDE_PROJECT_DIR").ok().and_then(|dir| {
            std::path::Path::new(dir.trim())
                .file_name()
                .and_then(|n| n.to_str())
                .map(str::to_string)
        }) {
            if !project.is_empty() {
                let project_header = format!("[Project Context] Active project: {project}");
                if let Some(ref mut hso) = output.hook_specific_output {
                    match &hso.additional_context {
                        Some(existing) => {
                            hso.additional_context =
                                Some(format!("{project_header}\n\n{existing}"));
                        }
                        None => {
                            hso.additional_context = Some(project_header);
                        }
                    }
                } else {
                    output.hook_specific_output = Some(HookSpecificOutput {
                        hook_event_name: hook_event.to_string(),
                        additional_context: Some(project_header),
                        ..HookSpecificOutput::default()
                    });
                }
            }
        }
    }

    // Write output to stdout
    sentinel_infrastructure::stdout::write_hook_output(&output)?;

    // Force-exit immediately after writing output. This hook process is
    // short-lived; the tokio multi-thread runtime holds background threads
    // (reqwest connection pool, etc.) that delay normal OS exit by several
    // seconds. Claude Code observes the process as "still running" until
    // those threads drain, which trips the 3s test timeout and wedges the
    // REPL in production.
    std::process::exit(0);
}

// ── Event arm handlers ────────────────────────────────────────────────────────
//
// Each function below corresponds to one `HookEvent` arm extracted from
// `run_internal`'s match. The extraction is purely mechanical — no behaviour
// changes. `run_internal` stays responsible for adapter construction, session
// state load/save, output transformation, and stdout emission.

/// Handle `UserPromptSubmit`: skill router, phase validator, two-phase prompt
/// hooks (doc drift, commit hygiene, context monitor, …).
async fn handle_user_prompt_submit(
    input: &sentinel_domain::events::HookInput,
    state: &mut sentinel_domain::state::SessionState,
    ctx: &hooks::HookContext<'_>,
    workflows: &HashMap<String, SkillWorkflow>,
    step_configs: &HashMap<String, sentinel_domain::workflow::SkillSteps>,
    repo_root: Option<&str>,
) -> HookOutput {
    let mut output = HookOutput::allow();

    // Skill router — Opus AI classification on EVERY message.
    // No regex substitute. If AI fails or times out, return no-match.
    //
    // Only initialize the AI classifier when there is a non-empty
    // prompt. openrouter::Client::new() blocks ~1-4s on network I/O
    // during init (rig-core v0.35). Skipping on no-prompt inputs
    // keeps hooks fast and tests under the 3s timeout.
    let has_prompt = input
        .prompt
        .as_deref()
        .is_some_and(|p| !p.trim().is_empty());
    // IMPORTANT: `RigClassifier::from_env()` constructs a reqwest/rig client
    // which can block for several seconds on Windows (TLS root-cert loading via
    // schannel). It MUST run inside the timeout, not before it, so that the 8 s
    // budget covers both the sync client init *and* the async classify call.
    //
    // We offload the blocking init to a spawn_blocking task so it doesn't starve
    // the async executor; the surrounding timeout cancels the future if the whole
    // operation (init + classify) exceeds 8 s.
    let router_output = if let Ok(out) =
        tokio::time::timeout(std::time::Duration::from_secs(8), async {
            let classifier = if has_prompt {
                tokio::task::spawn_blocking(
                    sentinel_infrastructure::rig_classifier::RigClassifier::from_env,
                )
                .await
                .ok()
                .flatten()
            } else {
                None
            };
            hooks::skill_router::process(
                input,
                classifier
                    .as_ref()
                    .map(|c| c as &dyn sentinel_application::classifier::AiClassifier),
                ctx.fs,
            )
            .await
        })
        .await
    {
        out
    } else {
        tracing::warn!("Skill router timed out (8s) — no routing for this message");
        hooks::skill_router::build_no_match_output(ctx.fs)
    };
    output.merge(&router_output);

    // Extract detected skill from router output and update state.
    // When no skill matches, clear active_skill so the phase gate
    // doesn't keep blocking on a stale skill from earlier in the session.
    if let Some(ref hso) = router_output.hook_specific_output {
        if let Some(ref ac) = hso.additional_context {
            if let Some(skill) = extract_skill_name(ac) {
                // Set active_skill for context injection (SKILL.md loading),
                // but ONLY register workflow-bearing skills when explicitly
                // invoked via slash command. The skill router's regex matching
                // is too aggressive — it matches casual phrases like "do it"
                // to "execute", "deploy this" to "deploy", etc. This registers
                // a workflow with mandatory phases that blocks ALL tool calls
                // until the user reads phase files they never intended to use.
                //
                // Slash commands set the prompt to exactly "/<skill>" which the
                // router detects. Casual matches come from normal conversation.
                let is_slash_command = input
                    .prompt
                    .as_deref()
                    .is_some_and(|p: &str| p.trim().starts_with('/'));

                if workflows.contains_key(&skill) && !is_slash_command {
                    // Skill has a workflow but was NOT explicitly invoked.
                    // Set active_skill for context injection (so SKILL.md loads)
                    // but do NOT call set_active_skill() which would register
                    // the workflow and trigger the phase gate.
                    state.active_skill = Some(skill.clone());
                    debug!(
                        skill = %skill,
                        "Skill detected via regex without slash command — setting context only"
                    );
                } else if let Some(workflow) = workflows.get(&skill) {
                    // Explicit slash command — activate the durable LangGraph
                    // phase graph and project the checkpoint it writes. The
                    // active marker is set only after graph activation succeeds.
                    match activate_phase_graph_workflow(state, &skill, workflow).await {
                        Ok(activation) => {
                            debug!(
                                skill = %skill,
                                activation = ?activation,
                                "LangGraph workflow activated from slash command"
                            );
                        }
                        Err(err) => {
                            return fail_closed_output_for_event(
                                sentinel_domain::events::HookEvent::UserPromptSubmit,
                                format!(
                                    "LangGraph phase activation failed for workflow '{skill}': {err}"
                                ),
                            );
                        }
                    }
                } else {
                    // No workflow definition — just set for context
                    state.active_skill = Some(skill);
                }
            } else if ac.contains("No skill matched") {
                state.active_skill = None;
            }
        }
    }

    let mut prompt_step_configs = step_configs.clone();
    if let Some(skill) = state.active_skill.as_deref() {
        if !prompt_step_configs.contains_key(skill) {
            let config_dir = sentinel_infrastructure::config::config_dir();
            match load_configured_skill_steps(&config_dir, workflows, skill) {
                Ok(Some(steps)) => {
                    prompt_step_configs.insert(skill.to_string(), steps);
                }
                Ok(None) => {}
                Err(e) => {
                    return HookOutput::deny(format!(
                        "[Sentinel-Authority] phase_validator: step config load failed for \
                         active skill '{skill}': {e:#}"
                    ));
                }
            }
        }
    }

    // Build the metrics envelope for this branch.
    let mk_ctx = |hook: &'static str| InvocationContext {
        event: "UserPromptSubmit",
        hook,
        tool: None,
        session_id: input.session_id.as_deref(),
        repo_root,
    };

    // Production override — arm/lock the session-wide prod-action grant on the
    // operator's "production override" / "production lock" phrases, and emit
    // the dual-display (systemMessage + additionalContext) transition notice.
    // Runs first so the arm/lock state is set before any other gate reacts.
    let prod_override_output = time_and_record(ctx.fs, &mk_ctx("production_override"), || {
        authorize_production_override_with_graph(input, state)
    });
    output.merge(&prod_override_output);

    // Phase validator — inject phase + step progress context
    let validator_output = time_and_record(ctx.fs, &mk_ctx("phase_validator"), || {
        hooks::phase_validator::process(input, state, workflows, &prompt_step_configs, ctx.fs)
    });
    output.merge(&validator_output);

    // Error reporter — inject Linear filing instructions for unresolved errors
    let error_output = time_and_record(ctx.fs, &mk_ctx("error_reporter"), || {
        hooks::error_reporter::process(input, ctx)
    });
    output.merge(&error_output);

    // Hygiene override — detect override commands in prompt
    let override_output = time_and_record(ctx.fs, &mk_ctx("hygiene_override"), || {
        hooks::hygiene_override::process(input, ctx)
    });
    output.merge(&override_output);

    // Worktree reminder — remind to use EnterWorktree in git repos
    let worktree_output = time_and_record(ctx.fs, &mk_ctx("worktree_reminder"), || {
        hooks::worktree_reminder::process(input, ctx)
    });
    output.merge(&worktree_output);

    // Orchestration nudge — suggest agent teams / Explore subagents /
    // skill invocation based on prompt heuristics.
    let orchestration_output = time_and_record(ctx.fs, &mk_ctx("orchestration_nudge"), || {
        hooks::orchestration_nudge::process(input, ctx)
    });
    output.merge(&orchestration_output);

    // Todo loader — inject active todos into context
    let todo_output = time_and_record(ctx.fs, &mk_ctx("todo_loader"), || {
        hooks::todo_loader::process(input, ctx)
    });
    output.merge(&todo_output);

    // Task status line — tasks as first-class citizens: inject the live native
    // TaskList (this session's tasks, with emoji + id + subject) on EVERY turn.
    let task_line_output = time_and_record(ctx.fs, &mk_ctx("task_status_line"), || {
        hooks::task_status_line::process(input, ctx)
    });
    output.merge(&task_line_output);

    // Linear inbound sync — poll the Hookdeck Events API for already-captured
    // Linear webhook deliveries and reconcile in-progress @linear-tagged tasks
    // whose issue moved to a terminal state, injecting TaskUpdate instructions.
    // Best-effort, fail-open, throttled to one Hookdeck poll per session window.
    let linear_inbound_output = time_and_record(ctx.fs, &mk_ctx("linear_inbound_sync"), || {
        hooks::linear_inbound_sync::process(input, ctx)
    });
    output.merge(&linear_inbound_output);

    // --- Two-phase hooks (read state written by Stop, inject instructions) ---

    // Doc drift — inject update instructions for stale docs
    let drift_output = time_and_record(ctx.fs, &mk_ctx("doc_drift"), || {
        hooks::doc_drift::process_prompt(input, ctx)
    });
    output.merge(&drift_output);

    // Doc cleanup — inject cleanup instructions for junk docs
    let cleanup_output = time_and_record(ctx.fs, &mk_ctx("doc_cleanup"), || {
        hooks::doc_cleanup::process_prompt(input, ctx)
    });
    output.merge(&cleanup_output);

    // Commit hygiene — remind about uncommitted changes
    let commit_output = time_and_record(ctx.fs, &mk_ctx("commit_hygiene"), || {
        hooks::commit_hygiene::process_prompt(input, ctx)
    });
    output.merge(&commit_output);

    // Context monitor — inject zone-specific strategy guidance
    let ctx_prompt_output = time_and_record(ctx.fs, &mk_ctx("context_monitor"), || {
        hooks::context_monitor::process_prompt(input, ctx)
    });
    output.merge(&ctx_prompt_output);

    // Verification gate — remind to verify before claiming completion
    let verify_prompt_output = time_and_record(ctx.fs, &mk_ctx("verification_gate"), || {
        hooks::verification_gate::process_prompt(input, ctx, state)
    });
    output.merge(&verify_prompt_output);

    // Activity tracker — inject session activity summary when context is elevated
    let activity_prompt_output = time_and_record(ctx.fs, &mk_ctx("activity_tracker"), || {
        hooks::activity_tracker::process_prompt(input, ctx)
    });
    output.merge(&activity_prompt_output);

    // Hygiene reminders — inject push/worktree/changelog reminders
    let reminders_prompt_output = time_and_record(ctx.fs, &mk_ctx("hygiene_reminders"), || {
        hooks::hygiene_reminders::process_prompt(input, ctx)
    });
    output.merge(&reminders_prompt_output);

    // Memory inject — search Qdrant for semantically relevant memories
    let memory_output = time_and_record(ctx.fs, &mk_ctx("memory_inject"), || {
        hooks::memory_inject::process(input, ctx)
    });
    output.merge(&memory_output);

    output
}

/// Handle `PreToolUse`: all blocking gates (phase, hygiene, dry-run, BA,
/// commit, …).
#[allow(clippy::too_many_arguments)]
fn handle_pre_tool_use(
    input: &sentinel_domain::events::HookInput,
    state: &mut sentinel_domain::state::SessionState,
    ctx: &hooks::HookContext<'_>,
    git: &RealGit,
    reversibility_classifier: &LayeredReversibilityClassifier,
    auditor: &dyn AuditorPort,
    a3_enabled: bool,
    provenance_store: Option<&JsonlProvenanceStore>,
    requirement_matrix: Option<&FilesystemRequirementMatrix>,
    spec_challenge_store: Option<&FilesystemSpecChallengeStore>,
    spec_challenge_scorer: Option<&LlmSpecChallengeScorer>,
    spec_challenge_config: SpecChallengeConfig,
    ba_enforcement: &BaEnforcementConfig,
    repo_root: Option<&str>,
    workflows: &HashMap<String, SkillWorkflow>,
    step_configs: &HashMap<String, SkillSteps>,
) -> HookOutput {
    let mut output = HookOutput::allow();

    // Build the fixed metrics envelope once — every wrapped hook
    // call stamps a JSONL row through `time_and_record` with this
    // context. Hooks themselves are unchanged; the wrapper just
    // measures wall-clock duration and records the outcome.
    let metrics_ctx = InvocationContext {
        event: "PreToolUse",
        hook: "", // overwritten per-call below via .with_hook(...)
        tool: input.tool_name.as_deref(),
        session_id: input.session_id.as_deref(),
        repo_root,
    };
    let mk_ctx = |hook: &'static str| InvocationContext {
        event: metrics_ctx.event,
        hook,
        tool: metrics_ctx.tool,
        session_id: metrics_ctx.session_id,
        repo_root: metrics_ctx.repo_root,
    };

    // Bug task gate — block mutating tools when a bug signal was
    // observed in tool output but no TaskCreate has filed it yet.
    let bug_gate_output = time_and_record(ctx.fs, &mk_ctx("bug_task_gate"), || {
        authorize_bug_task_with_graph(input, ctx)
    });
    output.merge(&bug_gate_output);

    // Linear PM-enforcement gate — hard-block invalid Linear starts using live
    // Linear authority for the ticket being moved.
    let linear_pm_output = time_and_record(ctx.fs, &mk_ctx("linear_pm_gate"), || {
        authorize_linear_pm_with_graph(input, ctx)
    });
    output.merge(&linear_pm_output);

    // Task decomposition gate — block mutating tools (Edit/Write/NotebookEdit,
    // state-changing Bash) when no live decomposed task list exists for the
    // session. Allowlists Read/Glob/Grep/Task*/Skill/sequential-thinking so the
    // gate never blocks the fix path; fails closed when task state is unreadable.
    let task_decomp_output = time_and_record(ctx.fs, &mk_ctx("task_decomposition_gate"), || {
        authorize_task_decomposition_with_graph(input, ctx)
    });
    output.merge(&task_decomp_output);

    // Task enforcement gate — helpful-block Edit/Write/NotebookEdit when the
    // session has no in_progress task. Fails open on any read error / missing
    // task dir / non-repo; kill switch SENTINEL_TASK_GATE=off. The deny message
    // guides the agent to create+start a task (or ask the operator).
    let task_enforcement_output = time_and_record(ctx.fs, &mk_ctx("task_enforcement_gate"), || {
        hooks::task_enforcement_gate::process(input, ctx)
    });
    output.merge(&task_enforcement_output);

    // Skill invocation gate — block tools when a skill was detected
    // by skill_router but not yet invoked. Allowlists Read/Glob/Grep/
    // Skill/Task* so the gate doesn't refuse to let Claude clear it.
    let skill_gate_output = time_and_record(ctx.fs, &mk_ctx("skill_invocation_gate"), || {
        authorize_skill_invocation_with_graph(input, ctx)
    });
    output.merge(&skill_gate_output);

    // Phase gate — check workflow state + track Read() calls on phase files
    let gate_output = time_and_record(ctx.fs, &mk_ctx("phase_gate"), || {
        authorize_phase_gate_with_graph(input, state, workflows, ctx.fs)
    });
    output.merge(&gate_output);

    if gate_output.blocked == Some(true) {
        state.record_blocked();
    }

    // Ticket quality gate — enforce Linear Definition-of-Ready on create/update.
    // Scoped to exactly those two MCP tools so discovery/read tools and
    // non-Linear work pass through untouched.
    let ticket_quality_output = time_and_record(ctx.fs, &mk_ctx("ticket_quality_gate"), || {
        authorize_ticket_quality_with_graph(input)
    });
    output.merge(&ticket_quality_output);

    // BA1 provenance_validate — structural enforcement for citations.
    // Self-gates on input.extra.artifacts presence; non-BA tools
    // pass through silently. Mode-configurable via
    // ba-enforcement.toml; shipped default is DefaultBlocking.
    let prov_output = time_and_record(ctx.fs, &mk_ctx("provenance_validate"), || {
        authorize_provenance_validate_with_graph(
            input,
            provenance_store,
            ba_enforcement.provenance_validate_mode,
        )
    });
    output.merge(&prov_output);

    // BA3 requirements_traceability_gate — structural enforcement for
    // recommendation→requirement traceability. Self-gates on
    // input.extra.requirement_refs / is_recommendation; non-BA tools
    // pass through silently. Mode-configurable via
    // ba-enforcement.toml; shipped default is DefaultBlocking.
    let trace_output = time_and_record(ctx.fs, &mk_ctx("requirements_traceability_gate"), || {
        authorize_requirements_traceability_with_graph(
            input,
            requirement_matrix,
            ba_enforcement.requirements_traceability_mode,
        )
    });
    output.merge(&trace_output);

    // A13 spec_challenge_gate — reversibility-class-graded completeness
    // check on the agent's SpecChallenge (input.extra.spec_challenge).
    // Self-gates: TriviallyReversible work skips, ReversibleWithEffort is
    // optional, non-A13 tools without a challenge pass through. The
    // reversibility class is computed from the same classifier the other
    // gates use. Mode + axis threshold come from spec_challenge_config,
    // which ships DefaultBlocking. Activating this gate hard-blocks missing
    // or incomplete Irreversible+ challenges unless an operator explicitly
    // chooses ObserveOnly for diagnostics.
    // Dispatch only when a tool name is present so the classifier has
    // something to key on.
    if let Some(tool) = input.tool_name.as_deref() {
        let null_input = serde_json::Value::Null;
        let tool_input_ref = input.tool_input.as_ref().unwrap_or(&null_input);
        let class = reversibility_classifier.classify(tool, tool_input_ref);
        let spec_output = time_and_record(ctx.fs, &mk_ctx("spec_challenge_gate"), || {
            authorize_spec_challenge_with_graph(
                input,
                class,
                spec_challenge_store,
                spec_challenge_scorer,
                spec_challenge_config.mode,
                spec_challenge_config.catastrophic_axis_threshold,
            )
        });
        output.merge(&spec_output);
    }

    // A3 dry-run-then-commit gate — fires for ALL tools (the hook itself
    // short-circuits to allow() for class < Irreversible). Auditor construction
    // is mandatory during hook startup, so this path has no inert mode.
    // Skip it entirely when an earlier gate already blocked this call:
    // `auditor.score()` is an LLM round-trip (up to a 30s timeout) and
    // there's no point paying that latency to audit a tool that's
    // already denied. Cheap string-matching gates above stay
    // unconditional; only this network-bound one is guarded.
    if output.blocked != Some(true) {
        let dry_run_output = time_and_record(ctx.fs, &mk_ctx("dry_run_then_commit"), || {
            authorize_dry_run_then_commit_with_graph(
                input,
                ctx.fs,
                reversibility_classifier,
                auditor,
            )
        });
        output.merge(&dry_run_output);
    }

    // Git hygiene — block on protected branch without worktree + uncommitted file limit
    if matches!(input.tool_name.as_deref(), Some("Edit" | "Write")) {
        let hygiene_output = time_and_record(ctx.fs, &mk_ctx("git_hygiene"), || {
            authorize_git_hygiene_with_graph(input, git, ctx.fs, state)
        });
        output.merge(&hygiene_output);

        // tasks.md auto-block guard — block edits/writes that would
        // mutate the SENTINEL:TASKS auto block (owned by task_persist).
        let tasks_guard_output = time_and_record(ctx.fs, &mk_ctx("tasks_md_guard"), || {
            authorize_tasks_md_guard_with_graph(input, ctx)
        });
        output.merge(&tasks_guard_output);

        // Tool usage gate — require sequential thinking + task creation.
        // When `a3_enabled`, Irreversible/Catastrophic short-circuit to
        // allow() inside the gate so A3's dry_run_then_commit hook (run
        // above) owns those classes via its separate-model-family auditor.
        let usage_output = time_and_record(ctx.fs, &mk_ctx("tool_usage_gate"), || {
            authorize_tool_usage_with_graph(input, ctx.fs, reversibility_classifier, a3_enabled)
        });
        output.merge(&usage_output);
    }

    // Doppler/Auth0 gate — block mutation tools (any tool type)
    let doppler_output = time_and_record(ctx.fs, &mk_ctx("doppler_auth0_gate"), || {
        authorize_doppler_auth0_with_graph(input, ctx)
    });
    output.merge(&doppler_output);

    // Production-action notice — when the session-wide production override
    // is armed (see production_override), surface a non-blocking dual-display
    // notice on any prod-touching mutating tool call so the operator sees each
    // prod action as it happens. No-op when not armed or on reads/non-prod.
    let prod_action_output = time_and_record(ctx.fs, &mk_ctx("production_action_notice"), || {
        authorize_production_action_notice_with_graph(input, state)
    });
    output.merge(&prod_action_output);

    // Agent revocation kill switch — deny tool calls carrying
    // a revoked agent_id. No-op for the main session (no
    // agent_id on input).
    let revoke_output = time_and_record(ctx.fs, &mk_ctx("agent_revocation"), || {
        authorize_agent_revocation_with_graph(input, state)
    });
    output.merge(&revoke_output);

    // Step gate — for step tools, require a loaded step config and the prereq
    // StepProof in state. Falls through only for non-step tools.
    let step_output = time_and_record(ctx.fs, &mk_ctx("step_gate"), || {
        authorize_step_gate_with_graph(input, state, step_configs)
    });
    output.merge(&step_output);

    // Pre-commit verification — block git commit/push without test evidence (Bash only)
    if matches!(input.tool_name.as_deref(), Some("Bash")) {
        let commit_output = time_and_record(ctx.fs, &mk_ctx("pre_commit_verification"), || {
            authorize_pre_commit_verification_with_graph(input, ctx, state)
        });
        output.merge(&commit_output);

        // Commit message validator — enforce conventional commits (Bash only)
        let msg_output = time_and_record(ctx.fs, &mk_ctx("commit_message_validator"), || {
            authorize_commit_message_with_graph(input, ctx)
        });
        output.merge(&msg_output);

        // Pre-push browser test — block git push without a browser test (Bash only)
        let browser_test_output = time_and_record(ctx.fs, &mk_ctx("pre_push_browser_test"), || {
            authorize_pre_push_browser_test_with_graph(input, ctx)
        });
        output.merge(&browser_test_output);

        // PR merge gate — fetch the PR's real review/CI state and decide
        // (allow silently / deny with the blocking facts / ask when unknown).
        let pr_output = time_and_record(ctx.fs, &mk_ctx("pr_merge_gate"), || {
            authorize_pr_merge_with_graph(input, ctx.env, ctx.process)
        });
        output.merge(&pr_output);

        // DB ops gate — block production database operations (Bash only)
        let db_output = time_and_record(ctx.fs, &mk_ctx("db_ops_gate"), || {
            authorize_db_ops_with_graph(input)
        });
        output.merge(&db_output);

        // Plan title gate — graph-authorized block for ExitPlanMode when the
        // plan has no derivable title, so plan_organizer can always file it
        // under a descriptive name. Ignores non-ExitPlanMode tools.
        let plan_title_output = time_and_record(ctx.fs, &mk_ctx("plan_title_gate"), || {
            authorize_plan_title_with_graph(input)
        });
        output.merge(&plan_title_output);

        // NOTE: the output_compressor ("RTK") hook was REMOVED — it rewrote
        // every noisy Bash command through `sentinel compress`, which echoed a
        // `[sentinel][compress] …` annotation on every call for ~0–2% savings
        // while eating captured output. Net negative; the dispatch is gone.
    }

    output
}

fn authorize_linear_pm_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::linear_pm_gate::evaluate_pretool(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::linear_pm_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_linear_pm_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Linear PM LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    let bool_checks = [
        (
            "target_tool",
            graph_run.state.target_tool,
            evaluation.target_tool,
        ),
        (
            "tool_input_present",
            graph_run.state.tool_input_present,
            evaluation.tool_input_present,
        ),
        (
            "start_transition",
            graph_run.state.start_transition,
            evaluation.start_transition,
        ),
        (
            "issue_key_present",
            graph_run.state.issue_key_present,
            evaluation.issue_key_present,
        ),
        (
            "issue_fetched",
            graph_run.state.issue_fetched,
            evaluation.issue_fetched,
        ),
        (
            "blocked_ticket",
            graph_run.state.blocked_ticket,
            evaluation.blocked_ticket,
        ),
        (
            "estimate_present",
            graph_run.state.estimate_present,
            evaluation.estimate_present,
        ),
        (
            "oversized_ticket",
            graph_run.state.oversized_ticket,
            evaluation.oversized_ticket,
        ),
        (
            "project_has_milestones",
            graph_run.state.project_has_milestones,
            evaluation.project_has_milestones,
        ),
        (
            "milestone_present",
            graph_run.state.milestone_present,
            evaluation.milestone_present,
        ),
        (
            "missing_milestone",
            graph_run.state.missing_milestone,
            evaluation.missing_milestone,
        ),
        (
            "target_priority_present",
            graph_run.state.target_priority_present,
            evaluation.target_priority_present,
        ),
        (
            "target_assignee_present",
            graph_run.state.target_assignee_present,
            evaluation.target_assignee_present,
        ),
        (
            "higher_priority_available",
            graph_run.state.higher_priority_available,
            evaluation.higher_priority_available,
        ),
        (
            "should_block",
            graph_run.state.should_block,
            evaluation.should_block,
        ),
    ];
    for (name, graph_value, evaluation_value) in bool_checks {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Linear PM LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    if (graph_run.state.estimate_points - evaluation.estimate_points).abs() > f64::EPSILON {
        return HookOutput::deny(format!(
            "Linear PM LangGraph authority mismatch: graph estimate_points={} but hook \
             evaluation estimate_points={}",
            graph_run.state.estimate_points, evaluation.estimate_points
        ));
    }

    if graph_run.state.target_priority != evaluation.target_priority {
        return HookOutput::deny(format!(
            "Linear PM LangGraph authority mismatch: graph target_priority={} but hook \
             evaluation target_priority={}",
            graph_run.state.target_priority, evaluation.target_priority
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.linear_pm_authorization(), "Linear PM");

    let expected_decision = linear_pm_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Linear PM LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::linear_pm_graph::linear_pm_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::linear_pm_graph::linear_pm_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_linear_pm_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Linear PM LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::linear_pm_gate::output_from_evaluation(&evaluation)
}

fn linear_pm_expected_decision(
    evaluation: &hooks::linear_pm_gate::LinearPmEvaluation,
) -> sentinel_infrastructure::linear_pm_graph::LinearPmDecision {
    match evaluation.decision {
        hooks::linear_pm_gate::LinearPmDecision::Allow => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::Allow
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockMissingIssueIdentifier => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::BlockMissingIssueIdentifier
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockLiveAuthorityUnavailable => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::
                BlockLiveAuthorityUnavailable
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockBlockedTicket => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::BlockBlockedTicket
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockOversizedTicket => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::BlockOversizedTicket
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockMissingMilestone => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::BlockMissingMilestone
        }
        hooks::linear_pm_gate::LinearPmDecision::BlockHigherPriorityAvailable => {
            sentinel_infrastructure::linear_pm_graph::LinearPmDecision::BlockHigherPriorityAvailable
        }
    }
}

fn run_linear_pm_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::linear_pm_gate::LinearPmEvaluation,
) -> Result<sentinel_infrastructure::linear_pm_graph::LinearPmGraphRun> {
    let identifier = linear_pm_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::linear_pm_graph::LinearPmState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::linear_pm_graph::build_linear_pm_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build linear PM graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::linear_pm_graph::run_linear_pm_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run linear PM graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("linear PM graph timed out"))?
}

fn linear_pm_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::linear_pm_gate::LinearPmEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:issue-key-present-{}:block-{}:decision-{}",
        evaluation.issue_key_present,
        evaluation.should_block,
        sentinel_infrastructure::linear_pm_graph::linear_pm_decision_label(
            linear_pm_expected_decision(evaluation)
        )
    );
    if evaluation.issue_key_present {
        let issue_key = evaluation
            .issue_key
            .as_deref()
            .map(str::trim)
            .filter(|issue_key| !issue_key.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Linear PM LangGraph authority requires concrete issue key evidence when \
                     issue_key_present=true"
                )
            })?;
        identifier.push_str(":issue-key-sha256:");
        identifier.push_str(&sentinel_infrastructure::linear_pm_graph::sha256(issue_key));
    }
    Ok(identifier)
}

fn write_linear_pm_graph_audit(
    run: &sentinel_infrastructure::linear_pm_graph::LinearPmGraphRun,
    authorization: &sentinel_infrastructure::linear_pm_graph::LinearPmAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("linear-pm.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create linear PM graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open linear PM graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "linear_pm",
        "decision": sentinel_infrastructure::linear_pm_graph::
            linear_pm_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write linear PM graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate linear PM graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_bug_task_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::bug_task_gate::evaluate_pretool(input, ctx);
    hooks::bug_task_gate::apply_pretool_side_effects(&evaluation, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::bug_task_gate::output_from_pretool_evaluation(&evaluation);
    }

    let graph_run = match run_bug_task_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Bug task LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph should_block={} but hook evaluation \
             should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.pending_bug_present != evaluation.pending_bug_present {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph pending_bug_present={} but hook \
             evaluation pending_bug_present={}",
            graph_run.state.pending_bug_present, evaluation.pending_bug_present
        ));
    }

    if graph_run.state.pending_bug_stale != evaluation.pending_bug_stale {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph pending_bug_stale={} but hook \
             evaluation pending_bug_stale={}",
            graph_run.state.pending_bug_stale, evaluation.pending_bug_stale
        ));
    }

    if graph_run.state.allowed_tool != evaluation.allowed_tool {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph allowed_tool={} but hook evaluation \
             allowed_tool={}",
            graph_run.state.allowed_tool, evaluation.allowed_tool
        ));
    }

    if graph_run.state.evidence_present != evaluation.evidence_present {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph evidence_present={} but hook evaluation \
             evidence_present={}",
            graph_run.state.evidence_present, evaluation.evidence_present
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.bug_task_authorization(), "Bug task");

    let expected_decision = bug_task_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Bug task LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::bug_task_graph::bug_task_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::bug_task_graph::bug_task_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_bug_task_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Bug task LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::bug_task_gate::output_from_pretool_evaluation(&evaluation)
}

fn bug_task_expected_decision(
    evaluation: &hooks::bug_task_gate::BugTaskEvaluation,
) -> sentinel_infrastructure::bug_task_graph::BugTaskDecision {
    match evaluation.decision {
        hooks::bug_task_gate::BugTaskDecision::Allow => {
            sentinel_infrastructure::bug_task_graph::BugTaskDecision::Allow
        }
        hooks::bug_task_gate::BugTaskDecision::Block => {
            sentinel_infrastructure::bug_task_graph::BugTaskDecision::Block
        }
    }
}

fn run_bug_task_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::bug_task_gate::BugTaskEvaluation,
) -> Result<sentinel_infrastructure::bug_task_graph::BugTaskGraphRun> {
    let identifier = bug_task_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::bug_task_graph::BugTaskState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::bug_task_graph::build_bug_task_graph().await
            {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build bug task graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::bug_task_graph::run_bug_task_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run bug task graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("bug task graph timed out"))?
}

fn bug_task_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::bug_task_gate::BugTaskEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:repo-root-present-{}:evidence-present-{}:block-{}:allowed-{}",
        evaluation.repo_root_present,
        evaluation.evidence_present,
        evaluation.should_block,
        evaluation.allowed_tool
    );
    if evaluation.repo_root_present {
        let repo_root = evaluation
            .repo_root
            .as_deref()
            .map(str::trim)
            .filter(|repo_root| !repo_root.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Bug task LangGraph authority requires concrete repo root evidence when \
                     repo_root_present=true"
                )
            })?;
        identifier.push_str(":repo-root-sha256:");
        identifier.push_str(&sentinel_infrastructure::bug_task_graph::sha256(repo_root));
    }
    if evaluation.evidence_present {
        let evidence = evaluation
            .evidence
            .as_deref()
            .map(str::trim)
            .filter(|evidence| !evidence.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Bug task LangGraph authority requires concrete bug evidence when \
                     evidence_present=true"
                )
            })?;
        identifier.push_str(":evidence-sha256:");
        identifier.push_str(&sentinel_infrastructure::bug_task_graph::sha256(evidence));
    }
    Ok(identifier)
}

fn write_bug_task_graph_audit(
    run: &sentinel_infrastructure::bug_task_graph::BugTaskGraphRun,
    authorization: &sentinel_infrastructure::bug_task_graph::BugTaskAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("bug-task.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create bug task graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open bug task graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "bug_task",
        "decision": sentinel_infrastructure::bug_task_graph::
            bug_task_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write bug task graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate bug task graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_git_hygiene_with_graph(
    input: &sentinel_domain::events::HookInput,
    git: &dyn sentinel_domain::ports::GitStatusPort,
    fs: &dyn sentinel_domain::ports::FileSystemPort,
    _state: &SessionState,
) -> HookOutput {
    let evaluation = hooks::git_hygiene::evaluate(input, git, fs);
    if !evaluation.graph_authority_required() {
        return hooks::git_hygiene::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_git_hygiene_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Git hygiene LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    let bool_checks = [
        (
            "hook_applies",
            graph_run.state.hook_applies,
            evaluation.hook_applies,
        ),
        (
            "protected_branch_block",
            graph_run.state.protected_branch_block,
            evaluation.protected_branch_block,
        ),
        (
            "uncommitted_file_limit_exceeded",
            graph_run.state.uncommitted_file_limit_exceeded,
            evaluation.uncommitted_file_limit_exceeded,
        ),
        (
            "should_deny",
            graph_run.state.should_deny,
            evaluation.should_deny,
        ),
        (
            "branch_known",
            graph_run.state.branch_known,
            evaluation.branch_known,
        ),
        (
            "protected_branch",
            graph_run.state.protected_branch,
            evaluation.protected_branch,
        ),
        ("worktree", graph_run.state.worktree, evaluation.worktree),
        (
            "merge_in_progress",
            graph_run.state.merge_in_progress,
            evaluation.merge_in_progress,
        ),
        (
            "has_uncommitted_changes_known",
            graph_run.state.has_uncommitted_changes_known,
            evaluation.has_uncommitted_changes_known,
        ),
        (
            "has_uncommitted_changes",
            graph_run.state.has_uncommitted_changes,
            evaluation.has_uncommitted_changes,
        ),
        (
            "changed_files_known",
            graph_run.state.changed_files_known,
            evaluation.changed_files_known,
        ),
    ];
    for (name, graph_value, evaluation_value) in bool_checks {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Git hygiene LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    if graph_run.state.changed_file_count != evaluation.changed_file_count as u64 {
        return HookOutput::deny(format!(
            "Git hygiene LangGraph authority mismatch: graph changed_file_count={} but hook \
             evaluation changed_file_count={}",
            graph_run.state.changed_file_count, evaluation.changed_file_count
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.git_hygiene_authorization(), "Git hygiene");

    let expected_decision = git_hygiene_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Git hygiene LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::git_hygiene_graph::git_hygiene_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::git_hygiene_graph::git_hygiene_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_git_hygiene_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Git hygiene LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::git_hygiene::output_from_evaluation(&evaluation)
}

fn git_hygiene_expected_decision(
    evaluation: &hooks::git_hygiene::GitHygieneEvaluation,
) -> sentinel_infrastructure::git_hygiene_graph::GitHygieneDecision {
    match evaluation.decision {
        hooks::git_hygiene::GitHygieneDecision::Allow => {
            sentinel_infrastructure::git_hygiene_graph::GitHygieneDecision::Allow
        }
        hooks::git_hygiene::GitHygieneDecision::DenyProtectedBranch => {
            sentinel_infrastructure::git_hygiene_graph::GitHygieneDecision::DenyProtectedBranch
        }
        hooks::git_hygiene::GitHygieneDecision::DenyUncommittedFileLimit => {
            sentinel_infrastructure::git_hygiene_graph::GitHygieneDecision::DenyUncommittedFileLimit
        }
    }
}

fn run_git_hygiene_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::git_hygiene::GitHygieneEvaluation,
) -> Result<sentinel_infrastructure::git_hygiene_graph::GitHygieneGraphRun> {
    let identifier = git_hygiene_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::git_hygiene_graph::GitHygieneState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::git_hygiene_graph::build_git_hygiene_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build git hygiene graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::git_hygiene_graph::run_git_hygiene_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run git hygiene graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("git hygiene graph timed out"))?
}

fn git_hygiene_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::git_hygiene::GitHygieneEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let cwd_hash = sentinel_infrastructure::git_hygiene_graph::sha256(&evaluation.cwd);
    let mut identifier = format!(
        "{session_id}:{tool}:{cwd_hash}:file-path-present-{}:deny-{}:protected-{}:dirty-limit-{}",
        evaluation.file_path_present,
        evaluation.should_deny,
        evaluation.protected_branch_block,
        evaluation.uncommitted_file_limit_exceeded
    );
    if evaluation.file_path_present {
        let file_path = evaluation
            .file_path
            .as_deref()
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Git hygiene LangGraph authority requires concrete file path evidence when \
                     file_path_present=true"
                )
            })?;
        identifier.push_str(":file-path-sha256:");
        identifier.push_str(&sentinel_infrastructure::git_hygiene_graph::sha256(
            file_path,
        ));
    }
    Ok(identifier)
}

fn write_git_hygiene_graph_audit(
    run: &sentinel_infrastructure::git_hygiene_graph::GitHygieneGraphRun,
    authorization: &sentinel_infrastructure::git_hygiene_graph::GitHygieneAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("git-hygiene.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create git hygiene graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open git hygiene graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "git_hygiene",
        "decision": sentinel_infrastructure::git_hygiene_graph::
            git_hygiene_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write git hygiene graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate git hygiene graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_phase_gate_with_graph(
    input: &sentinel_domain::events::HookInput,
    state: &mut SessionState,
    workflows: &HashMap<String, SkillWorkflow>,
    fs: &dyn hooks::FileSystemPort,
) -> HookOutput {
    let evaluation = hooks::phase_gate::evaluate(input, state, workflows, fs);
    if !evaluation.graph_authority_required() {
        return hooks::phase_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_phase_gate_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Phase gate LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    for (name, graph_value, evaluation_value) in [
        (
            "tool_present",
            graph_run.state.tool_present,
            evaluation.tool_present,
        ),
        (
            "dangerous_mcp_tool",
            graph_run.state.dangerous_mcp_tool,
            evaluation.dangerous_mcp_tool,
        ),
        (
            "safe_mcp_tool",
            graph_run.state.safe_mcp_tool,
            evaluation.safe_mcp_tool,
        ),
        (
            "tool_call_recorded",
            graph_run.state.tool_call_recorded,
            evaluation.tool_call_recorded,
        ),
        (
            "phase_read_recorded",
            graph_run.state.phase_read_recorded,
            evaluation.phase_read_recorded,
        ),
        (
            "phase_hash_recorded",
            graph_run.state.phase_hash_recorded,
            evaluation.phase_hash_recorded,
        ),
        ("blocked", graph_run.state.blocked, evaluation.blocked),
        (
            "reason_present",
            graph_run.state.reason_present,
            evaluation.reason_present,
        ),
    ] {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Phase gate LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    for (name, graph_value, evaluation_value) in [
        (
            "tool_calls_before",
            graph_run.state.tool_calls_before,
            evaluation.tool_calls_before,
        ),
        (
            "tool_calls_after",
            graph_run.state.tool_calls_after,
            evaluation.tool_calls_after,
        ),
        (
            "phases_read_before",
            graph_run.state.phases_read_before,
            evaluation.phases_read_before as u64,
        ),
        (
            "phases_read_after",
            graph_run.state.phases_read_after,
            evaluation.phases_read_after as u64,
        ),
        (
            "phase_hashes_before",
            graph_run.state.phase_hashes_before,
            evaluation.phase_hashes_before as u64,
        ),
        (
            "phase_hashes_after",
            graph_run.state.phase_hashes_after,
            evaluation.phase_hashes_after as u64,
        ),
        (
            "reason_len",
            graph_run.state.reason_len,
            evaluation.reason_len as u64,
        ),
    ] {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Phase gate LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    let evaluation_reason_sha256 = evaluation
        .reason_present
        .then(|| evaluation.reason_sha256.clone());
    if graph_run.state.reason_sha256 != evaluation_reason_sha256 {
        return HookOutput::deny(
            "Phase gate LangGraph authority mismatch: graph reason digest did not match hook \
             evaluation",
        );
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.phase_gate_authorization(), "Phase gate");

    let expected_decision =
        sentinel_infrastructure::phase_gate_graph::expected_decision_from_app(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Phase gate LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::phase_gate_graph::phase_gate_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::phase_gate_graph::phase_gate_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_phase_gate_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Phase gate LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::phase_gate::output_from_evaluation(&evaluation)
}

fn run_phase_gate_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::phase_gate::PhaseGateEvaluation,
) -> Result<sentinel_infrastructure::phase_gate_graph::PhaseGateGraphRun> {
    let identifier = phase_gate_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::phase_gate_graph::PhaseGateState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::phase_gate_graph::build_phase_gate_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build phase gate graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::phase_gate_graph::run_phase_gate_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run phase gate graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("phase gate graph timed out"))?
}

fn phase_gate_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::phase_gate::PhaseGateEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let decision =
        sentinel_infrastructure::phase_gate_graph::expected_decision_from_app(evaluation);
    let mut identifier = format!(
        "{session_id}:{tool}:calls-{}-{}:phases-{}-{}:hashes-{}-{}:blocked-{}:reason-present-{}:{}",
        evaluation.tool_calls_before,
        evaluation.tool_calls_after,
        evaluation.phases_read_before,
        evaluation.phases_read_after,
        evaluation.phase_hashes_before,
        evaluation.phase_hashes_after,
        evaluation.blocked,
        evaluation.reason_present,
        sentinel_infrastructure::phase_gate_graph::phase_gate_decision_label(decision)
    );
    if evaluation.reason_present {
        identifier.push_str(":reason-sha256:");
        identifier.push_str(&evaluation.reason_sha256);
    }
    Ok(identifier)
}

fn write_phase_gate_graph_audit(
    run: &sentinel_infrastructure::phase_gate_graph::PhaseGateGraphRun,
    authorization: &sentinel_infrastructure::phase_gate_graph::PhaseGateAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("phase-gate.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create phase gate graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open phase gate graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "phase_gate",
        "decision": sentinel_infrastructure::phase_gate_graph::
            phase_gate_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write phase gate graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate phase gate graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_tool_usage_with_graph(
    input: &sentinel_domain::events::HookInput,
    fs: &dyn hooks::FileSystemPort,
    classifier: &dyn ReversibilityClassifierPort,
    a3_enabled: bool,
) -> HookOutput {
    let evaluation = hooks::tool_usage_gate::evaluate(input, fs, classifier, a3_enabled);
    if !evaluation.graph_authority_required() {
        return hooks::tool_usage_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_tool_usage_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Tool usage LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    for (name, graph_value, evaluation_value) in [
        (
            "tool_present",
            graph_run.state.tool_present,
            evaluation.tool_present,
        ),
        (
            "reversibility_class_present",
            graph_run.state.reversibility_class_present,
            evaluation.reversibility_class.is_some(),
        ),
        (
            "a3_enabled",
            graph_run.state.a3_enabled,
            evaluation.a3_enabled,
        ),
        (
            "a3_handoff",
            graph_run.state.a3_handoff,
            evaluation.a3_handoff,
        ),
        (
            "gate_required",
            graph_run.state.gate_required,
            evaluation.gate_required,
        ),
        (
            "session_id_present",
            graph_run.state.session_id_present,
            evaluation.session_id_present,
        ),
        (
            "transcript_path_present",
            graph_run.state.transcript_path_present,
            evaluation.transcript_path_present,
        ),
        (
            "transcript_authority_read",
            graph_run.state.transcript_authority_read,
            evaluation.transcript_authority_read,
        ),
        (
            "transcript_authority_error_present",
            graph_run.state.transcript_authority_error_present,
            evaluation.transcript_authority_error.is_some(),
        ),
        (
            "sequential_thinking_used",
            graph_run.state.sequential_thinking_used,
            evaluation.sequential_thinking_used,
        ),
        (
            "sequential_thinking_requirable",
            graph_run.state.sequential_thinking_requirable,
            evaluation.sequential_thinking_requirable,
        ),
        (
            "task_authority_read",
            graph_run.state.task_authority_read,
            evaluation.task_authority_read,
        ),
        (
            "task_authority_error_present",
            graph_run.state.task_authority_error_present,
            evaluation.task_authority_error.is_some(),
        ),
        (
            "in_progress_task_present",
            graph_run.state.in_progress_task_present,
            evaluation.in_progress_task_present,
        ),
        (
            "pending_task_hint_present",
            graph_run.state.pending_task_hint_present,
            evaluation.pending_task_hint.is_some(),
        ),
        (
            "should_deny",
            graph_run.state.should_deny,
            evaluation.should_deny,
        ),
    ] {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Tool usage LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    if graph_run.state.reversibility_class != evaluation.reversibility_class {
        return HookOutput::deny(format!(
            "Tool usage LangGraph authority mismatch: graph reversibility_class={:?} but hook \
             evaluation reversibility_class={:?}",
            graph_run.state.reversibility_class, evaluation.reversibility_class
        ));
    }

    if graph_run.state.plan_state != evaluation.plan_state {
        return HookOutput::deny(format!(
            "Tool usage LangGraph authority mismatch: graph plan_state={:?} but hook evaluation \
             plan_state={:?}",
            graph_run.state.plan_state, evaluation.plan_state
        ));
    }

    if graph_run.state.task_count != evaluation.task_count as u64 {
        return HookOutput::deny(format!(
            "Tool usage LangGraph authority mismatch: graph task_count={} but hook evaluation \
             task_count={}",
            graph_run.state.task_count, evaluation.task_count
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.tool_usage_authorization(), "Tool usage");

    let expected_decision =
        sentinel_infrastructure::tool_usage_graph::expected_decision_from_app(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Tool usage LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::tool_usage_graph::tool_usage_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::tool_usage_graph::tool_usage_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_tool_usage_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Tool usage LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::tool_usage_gate::output_from_evaluation(&evaluation)
}

fn run_tool_usage_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::tool_usage_gate::ToolUsageEvaluation,
) -> Result<sentinel_infrastructure::tool_usage_graph::ToolUsageGraphRun> {
    let identifier = tool_usage_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::tool_usage_graph::ToolUsageState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::tool_usage_graph::build_tool_usage_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build tool usage graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::tool_usage_graph::run_tool_usage_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run tool usage graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("tool usage graph timed out"))?
}

fn tool_usage_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::tool_usage_gate::ToolUsageEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let decision =
        sentinel_infrastructure::tool_usage_graph::expected_decision_from_app(evaluation);
    let mut identifier = format!(
        "{session_id}:{tool}:class-present-{}:transcript-path-present-{}:tasks-{}:deny-{}:{}",
        evaluation.reversibility_class.is_some(),
        evaluation.transcript_path_present,
        evaluation.task_count,
        evaluation.should_deny,
        sentinel_infrastructure::tool_usage_graph::tool_usage_decision_label(decision)
    );
    if let Some(class) = evaluation.reversibility_class {
        identifier.push_str(":class:");
        identifier.push_str(&class.to_string());
    }
    if evaluation.transcript_path_present {
        let transcript_path = evaluation
            .transcript_path
            .as_deref()
            .map(str::trim)
            .filter(|transcript_path| !transcript_path.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Tool usage LangGraph authority requires concrete transcript path evidence \
                     when transcript_path_present=true"
                )
            })?;
        identifier.push_str(":transcript-path-sha256:");
        identifier.push_str(&sentinel_infrastructure::tool_usage_graph::sha256(
            transcript_path,
        ));
    }
    Ok(identifier)
}

fn write_tool_usage_graph_audit(
    run: &sentinel_infrastructure::tool_usage_graph::ToolUsageGraphRun,
    authorization: &sentinel_infrastructure::tool_usage_graph::ToolUsageAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("tool-usage.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create tool usage graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open tool usage graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "tool_usage",
        "decision": sentinel_infrastructure::tool_usage_graph::
            tool_usage_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write tool usage graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate tool usage graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_skill_invocation_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::skill_invocation_gate::evaluate_pretool(input, ctx);
    hooks::skill_invocation_gate::apply_pretool_side_effects(&evaluation, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::skill_invocation_gate::output_from_pretool_evaluation(&evaluation);
    }

    let graph_run = match run_skill_invocation_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Skill invocation LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.pending_skill_present != evaluation.pending_skill_present {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph pending_skill_present={} but \
             hook evaluation pending_skill_present={}",
            graph_run.state.pending_skill_present, evaluation.pending_skill_present
        ));
    }

    if graph_run.state.pending_skill_stale != evaluation.pending_skill_stale {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph pending_skill_stale={} but hook \
             evaluation pending_skill_stale={}",
            graph_run.state.pending_skill_stale, evaluation.pending_skill_stale
        ));
    }

    if graph_run.state.pending_state_session_matches != evaluation.pending_state_session_matches {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph pending_state_session_matches={} \
             but hook evaluation pending_state_session_matches={}",
            graph_run.state.pending_state_session_matches, evaluation.pending_state_session_matches
        ));
    }

    if graph_run.state.allowed_tool != evaluation.allowed_tool {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph allowed_tool={} but hook \
             evaluation allowed_tool={}",
            graph_run.state.allowed_tool, evaluation.allowed_tool
        ));
    }

    if graph_run.state.skill_present != evaluation.skill_present {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph skill_present={} but hook \
             evaluation skill_present={}",
            graph_run.state.skill_present, evaluation.skill_present
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.skill_invocation_authorization(),
        "Skill invocation"
    );

    let expected_decision = skill_invocation_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::skill_invocation_graph::skill_invocation_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::skill_invocation_graph::skill_invocation_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_skill_invocation_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Skill invocation LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::skill_invocation_gate::output_from_pretool_evaluation(&evaluation)
}

fn skill_invocation_expected_decision(
    evaluation: &hooks::skill_invocation_gate::SkillInvocationEvaluation,
) -> sentinel_infrastructure::skill_invocation_graph::SkillInvocationDecision {
    match evaluation.decision {
        hooks::skill_invocation_gate::SkillInvocationDecision::Allow => {
            sentinel_infrastructure::skill_invocation_graph::SkillInvocationDecision::Allow
        }
        hooks::skill_invocation_gate::SkillInvocationDecision::Block => {
            sentinel_infrastructure::skill_invocation_graph::SkillInvocationDecision::Block
        }
    }
}

fn run_skill_invocation_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::skill_invocation_gate::SkillInvocationEvaluation,
) -> Result<sentinel_infrastructure::skill_invocation_graph::SkillInvocationGraphRun> {
    let identifier = skill_invocation_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::skill_invocation_graph::SkillInvocationState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::skill_invocation_graph::build_skill_invocation_graph(
                )
                .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build skill invocation graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::skill_invocation_graph::
                    run_skill_invocation_decision_report(&graph, state)
                        .await
                        .map_err(|e| anyhow!("run skill invocation graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("skill invocation graph timed out"))?
}

fn skill_invocation_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::skill_invocation_gate::SkillInvocationEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, evaluation.session_id.as_deref())?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:skill-present-{}:block-{}:allowed-{}",
        evaluation.skill.is_some(),
        evaluation.should_block,
        evaluation.allowed_tool
    );
    if let Some(skill) = evaluation
        .skill
        .as_deref()
        .map(str::trim)
        .filter(|skill| !skill.is_empty())
    {
        identifier.push_str(":skill-sha256:");
        identifier.push_str(&sentinel_infrastructure::skill_invocation_graph::sha256(
            skill,
        ));
    }
    Ok(identifier)
}

fn write_skill_invocation_graph_audit(
    run: &sentinel_infrastructure::skill_invocation_graph::SkillInvocationGraphRun,
    authorization: &sentinel_infrastructure::skill_invocation_graph::SkillInvocationAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("skill-invocation.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create skill invocation graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open skill invocation graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "skill_invocation",
        "decision": sentinel_infrastructure::skill_invocation_graph::
            skill_invocation_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write skill invocation graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate skill invocation graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_task_decomposition_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::task_decomposition_gate::evaluate(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::task_decomposition_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_task_decomposition_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Task decomposition LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.mutating_tool != evaluation.mutating_tool {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph mutating_tool={} but hook \
             evaluation mutating_tool={}",
            graph_run.state.mutating_tool, evaluation.mutating_tool
        ));
    }

    if graph_run.state.task_state_readable != evaluation.task_state_readable {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph task_state_readable={} but \
             hook evaluation task_state_readable={}",
            graph_run.state.task_state_readable, evaluation.task_state_readable
        ));
    }

    if graph_run.state.task_list_confirmed != evaluation.task_list_confirmed {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph task_list_confirmed={} but \
             hook evaluation task_list_confirmed={}",
            graph_run.state.task_list_confirmed, evaluation.task_list_confirmed
        ));
    }

    if graph_run.state.unreadable_task_state != evaluation.unreadable_task_state {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph unreadable_task_state={} but \
             hook evaluation unreadable_task_state={}",
            graph_run.state.unreadable_task_state, evaluation.unreadable_task_state
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.task_decomposition_authorization(),
        "Task decomposition"
    );

    let expected_decision = task_decomposition_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::task_decomposition_graph::task_decomposition_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::task_decomposition_graph::task_decomposition_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_task_decomposition_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Task decomposition LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::task_decomposition_gate::output_from_evaluation(&evaluation)
}

fn task_decomposition_expected_decision(
    evaluation: &hooks::task_decomposition_gate::TaskDecompositionEvaluation,
) -> sentinel_infrastructure::task_decomposition_graph::TaskDecompositionDecision {
    match evaluation.decision {
        hooks::task_decomposition_gate::TaskDecompositionDecision::Allow => {
            sentinel_infrastructure::task_decomposition_graph::TaskDecompositionDecision::Allow
        }
        hooks::task_decomposition_gate::TaskDecompositionDecision::Block => {
            sentinel_infrastructure::task_decomposition_graph::TaskDecompositionDecision::Block
        }
    }
}

fn run_task_decomposition_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::task_decomposition_gate::TaskDecompositionEvaluation,
) -> Result<sentinel_infrastructure::task_decomposition_graph::TaskDecompositionGraphRun> {
    let identifier = task_decomposition_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::task_decomposition_graph::TaskDecompositionState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::task_decomposition_graph::build_task_decomposition_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build task decomposition graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::task_decomposition_graph::
                    run_task_decomposition_decision_report(&graph, state)
                        .await
                        .map_err(|e| anyhow!("run task decomposition graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("task decomposition graph timed out"))?
}

fn task_decomposition_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::task_decomposition_gate::TaskDecompositionEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, evaluation.session_id.as_deref())?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:bash-command-present-{}:block-{}:readable-{}:confirmed-{}",
        evaluation.bash_command_present,
        evaluation.should_block,
        evaluation.task_state_readable,
        evaluation.task_list_confirmed
    );
    if let Some(command) = evaluation.bash_command.as_deref() {
        identifier.push_str(":bash-command-sha256:");
        identifier.push_str(&sentinel_infrastructure::task_decomposition_graph::sha256(
            command,
        ));
    }
    Ok(identifier)
}

fn write_task_decomposition_graph_audit(
    run: &sentinel_infrastructure::task_decomposition_graph::TaskDecompositionGraphRun,
    authorization: &sentinel_infrastructure::task_decomposition_graph::TaskDecompositionAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("task-decomposition.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create task decomposition graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open task decomposition graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "task_decomposition",
        "decision": sentinel_infrastructure::task_decomposition_graph::
            task_decomposition_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write task decomposition graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate task decomposition graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_plan_title_with_graph(input: &sentinel_domain::events::HookInput) -> HookOutput {
    let evaluation = hooks::plan_title_gate::evaluate(input);
    if !evaluation.graph_authority_required() {
        return hooks::plan_title_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_plan_title_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Plan title LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Plan title LangGraph authority mismatch: graph should_block={} but hook evaluation \
             should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.plan_text_present != evaluation.plan_text_present {
        return HookOutput::deny(format!(
            "Plan title LangGraph authority mismatch: graph plan_text_present={} but hook \
             evaluation plan_text_present={}",
            graph_run.state.plan_text_present, evaluation.plan_text_present
        ));
    }

    if graph_run.state.derivable_title != evaluation.derivable_title {
        return HookOutput::deny(format!(
            "Plan title LangGraph authority mismatch: graph derivable_title={} but hook \
             evaluation derivable_title={}",
            graph_run.state.derivable_title, evaluation.derivable_title
        ));
    }

    if graph_run.state.title_line_present != evaluation.title_line_present {
        return HookOutput::deny(format!(
            "Plan title LangGraph authority mismatch: graph title_line_present={} but hook \
             evaluation title_line_present={}",
            graph_run.state.title_line_present, evaluation.title_line_present
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.plan_title_authorization(), "Plan title");

    let expected_decision = plan_title_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Plan title LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::plan_title_graph::plan_title_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::plan_title_graph::plan_title_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_plan_title_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Plan title LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::plan_title_gate::output_from_evaluation(&evaluation)
}

fn plan_title_expected_decision(
    evaluation: &hooks::plan_title_gate::PlanTitleEvaluation,
) -> sentinel_infrastructure::plan_title_graph::PlanTitleDecision {
    match evaluation.decision {
        hooks::plan_title_gate::PlanTitleDecision::Allow => {
            sentinel_infrastructure::plan_title_graph::PlanTitleDecision::Allow
        }
        hooks::plan_title_gate::PlanTitleDecision::BlockMissingPlan => {
            sentinel_infrastructure::plan_title_graph::PlanTitleDecision::BlockMissingPlan
        }
        hooks::plan_title_gate::PlanTitleDecision::BlockTitleless => {
            sentinel_infrastructure::plan_title_graph::PlanTitleDecision::BlockTitleless
        }
    }
}

fn run_plan_title_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::plan_title_gate::PlanTitleEvaluation,
) -> Result<sentinel_infrastructure::plan_title_graph::PlanTitleGraphRun> {
    let identifier = plan_title_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::plan_title_graph::PlanTitleState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::plan_title_graph::build_plan_title_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build plan title graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::plan_title_graph::run_plan_title_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run plan title graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("plan title graph timed out"))?
}

fn plan_title_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::plan_title_gate::PlanTitleEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:plan-present-{}:block-{}:title-{}",
        evaluation.plan_text_present, evaluation.should_block, evaluation.derivable_title
    );
    if let Some(plan_text) = evaluation.plan_text.as_deref() {
        identifier.push_str(":plan-sha256:");
        identifier.push_str(&sentinel_infrastructure::plan_title_graph::sha256(
            plan_text,
        ));
    }
    Ok(identifier)
}

fn write_plan_title_graph_audit(
    run: &sentinel_infrastructure::plan_title_graph::PlanTitleGraphRun,
    authorization: &sentinel_infrastructure::plan_title_graph::PlanTitleAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("plan-title.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create plan title graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open plan title graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "plan_title",
        "decision": sentinel_infrastructure::plan_title_graph::
            plan_title_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write plan title graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate plan title graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_commit_message_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::commit_message_validator::evaluate(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::commit_message_validator::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_commit_message_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Commit message LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Commit message LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.conventional != evaluation.conventional {
        return HookOutput::deny(format!(
            "Commit message LangGraph authority mismatch: graph conventional={} but hook \
             evaluation conventional={}",
            graph_run.state.conventional, evaluation.conventional
        ));
    }

    if graph_run.state.linear_ref_required != evaluation.linear_ref_required {
        return HookOutput::deny(format!(
            "Commit message LangGraph authority mismatch: graph linear_ref_required={} but hook \
             evaluation linear_ref_required={}",
            graph_run.state.linear_ref_required, evaluation.linear_ref_required
        ));
    }

    if graph_run.state.linear_ref_present != evaluation.linear_ref_present {
        return HookOutput::deny(format!(
            "Commit message LangGraph authority mismatch: graph linear_ref_present={} but hook \
             evaluation linear_ref_present={}",
            graph_run.state.linear_ref_present, evaluation.linear_ref_present
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.commit_message_authorization(),
        "Commit message"
    );

    let expected_decision = commit_message_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Commit message LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::commit_message_graph::commit_message_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::commit_message_graph::commit_message_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_commit_message_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Commit message LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::commit_message_validator::output_from_evaluation(&evaluation)
}

fn commit_message_expected_decision(
    evaluation: &hooks::commit_message_validator::CommitMessageEvaluation,
) -> sentinel_infrastructure::commit_message_graph::CommitMessageDecision {
    match evaluation.decision {
        hooks::commit_message_validator::CommitMessageDecision::Allow => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::Allow
        }
        hooks::commit_message_validator::CommitMessageDecision::AllowAmend => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::AllowAmend
        }
        hooks::commit_message_validator::CommitMessageDecision::AllowNoMessage => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::AllowNoMessage
        }
        hooks::commit_message_validator::CommitMessageDecision::AllowConventional => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::AllowConventional
        }
        hooks::commit_message_validator::CommitMessageDecision::BlockMalformed => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::BlockMalformed
        }
        hooks::commit_message_validator::CommitMessageDecision::BlockMissingLinearRef => {
            sentinel_infrastructure::commit_message_graph::CommitMessageDecision::BlockMissingLinearRef
        }
    }
}

fn run_commit_message_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::commit_message_validator::CommitMessageEvaluation,
) -> Result<sentinel_infrastructure::commit_message_graph::CommitMessageGraphRun> {
    let identifier = commit_message_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::commit_message_graph::CommitMessageState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::commit_message_graph::build_commit_message_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build commit message graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::commit_message_graph::run_commit_message_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run commit message graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("commit message graph timed out"))?
}

fn commit_message_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::commit_message_validator::CommitMessageEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:command-present-{}:message-present-{}:block-{}:linear-required-{}",
        evaluation.command_present,
        evaluation.message.is_some(),
        evaluation.should_block,
        evaluation.linear_ref_required
    );
    if evaluation.command_present {
        let command = evaluation
            .command
            .as_deref()
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Commit message LangGraph authority requires concrete command evidence when \
                     command_present=true"
                )
            })?;
        identifier.push_str(":command-sha256:");
        identifier.push_str(&sentinel_infrastructure::commit_message_graph::sha256(
            command,
        ));
    }
    if let Some(message) = evaluation
        .message
        .as_deref()
        .map(str::trim)
        .filter(|message| !message.is_empty())
    {
        identifier.push_str(":message-sha256:");
        identifier.push_str(&sentinel_infrastructure::commit_message_graph::sha256(
            message,
        ));
    }
    Ok(identifier)
}

fn write_commit_message_graph_audit(
    run: &sentinel_infrastructure::commit_message_graph::CommitMessageGraphRun,
    authorization: &sentinel_infrastructure::commit_message_graph::CommitMessageAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("commit-message.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create commit message graph audit dir {}", parent.display())
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open commit message graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "commit_message",
        "decision": sentinel_infrastructure::commit_message_graph::commit_message_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write commit message graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate commit message graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_pre_commit_verification_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
    state: &SessionState,
) -> HookOutput {
    let evaluation = hooks::pre_commit_verification::evaluate(input, ctx, state);
    if !evaluation.graph_authority_required() {
        return hooks::pre_commit_verification::output_from_evaluation(input, &evaluation);
    }

    let graph_run = match run_pre_commit_verification_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Pre-commit verification LangGraph authority failed; refusing unaudited \
                 decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Pre-commit verification LangGraph authority mismatch: graph should_block={} but \
             hook evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.recorded_evidence_present != evaluation.recorded_evidence_present {
        return HookOutput::deny(format!(
            "Pre-commit verification LangGraph authority mismatch: graph \
             recorded_evidence_present={} but hook evaluation recorded_evidence_present={}",
            graph_run.state.recorded_evidence_present, evaluation.recorded_evidence_present
        ));
    }

    if graph_run.state.signed_override_active != evaluation.signed_override_active {
        return HookOutput::deny(format!(
            "Pre-commit verification LangGraph authority mismatch: graph \
             signed_override_active={} but hook evaluation signed_override_active={}",
            graph_run.state.signed_override_active, evaluation.signed_override_active
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.pre_commit_verification_authorization(),
        "Pre-commit verification"
    );

    let expected_decision = pre_commit_verification_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Pre-commit verification LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::pre_commit_verification_graph::
                pre_commit_verification_decision_label(authorization.decision()),
            sentinel_infrastructure::pre_commit_verification_graph::
                pre_commit_verification_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_pre_commit_verification_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Pre-commit verification LangGraph audit write failed; refusing unaudited decision: \
             {e:#}"
        ));
    }

    hooks::pre_commit_verification::output_from_evaluation(input, &evaluation)
}

fn pre_commit_verification_expected_decision(
    evaluation: &hooks::pre_commit_verification::PreCommitVerificationEvaluation,
) -> sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision {
    match evaluation.decision {
        hooks::pre_commit_verification::PreCommitDecision::Allow => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::Allow
        }
        hooks::pre_commit_verification::PreCommitDecision::AllowContentOnlyRepo => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::AllowContentOnlyRepo
        }
        hooks::pre_commit_verification::PreCommitDecision::AllowDocsOnly => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::AllowDocsOnly
        }
        hooks::pre_commit_verification::PreCommitDecision::AllowSignedOverride => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::AllowSignedOverride
        }
        hooks::pre_commit_verification::PreCommitDecision::AllowRecordedEvidence => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::AllowRecordedEvidence
        }
        hooks::pre_commit_verification::PreCommitDecision::Block => {
            sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationDecision::Block
        }
    }
}

fn run_pre_commit_verification_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pre_commit_verification::PreCommitVerificationEvaluation,
) -> Result<sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationGraphRun> {
    let identifier = pre_commit_verification_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::pre_commit_verification_graph::
        PreCommitVerificationState::from_evaluation(identifier, evaluation);
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::pre_commit_verification_graph::
                build_pre_commit_verification_graph()
                .await
            {
                Ok(graph) => graph,
                Err(e) => {
                    return Some(Err(anyhow!(
                        "build pre-commit verification graph: {e}"
                    )));
                }
            };
            Some(
                sentinel_infrastructure::pre_commit_verification_graph::
                    run_pre_commit_verification_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run pre-commit verification graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("pre-commit verification graph timed out"))?
}

fn pre_commit_verification_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pre_commit_verification::PreCommitVerificationEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let action =
        sentinel_infrastructure::pre_commit_verification_graph::action_label(evaluation.action);
    let mut identifier = format!(
        "{session_id}:{tool}:{action}:command-present-{}:block-{}:evidence-{}:override-{}",
        evaluation.command_present,
        evaluation.should_block,
        evaluation.recorded_evidence_present,
        evaluation.signed_override_active
    );
    if evaluation.command_present {
        let command = evaluation
            .command
            .as_deref()
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Pre-commit verification LangGraph authority requires concrete command \
                     evidence when command_present=true"
                )
            })?;
        identifier.push_str(":command-sha256:");
        identifier.push_str(
            &sentinel_infrastructure::pre_commit_verification_graph::command_sha256(command),
        );
    }
    Ok(identifier)
}

fn write_pre_commit_verification_graph_audit(
    run: &sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationGraphRun,
    authorization: &sentinel_infrastructure::pre_commit_verification_graph::PreCommitVerificationAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("pre-commit-verification.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create pre-commit verification graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open pre-commit verification graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "pre_commit_verification",
        "decision": sentinel_infrastructure::pre_commit_verification_graph::
            pre_commit_verification_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write pre-commit verification graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate pre-commit verification graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_pre_push_browser_test_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::pre_push_browser_test::evaluate(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::pre_push_browser_test::output_from_evaluation(input, &evaluation);
    }

    let graph_run = match run_pre_push_browser_test_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Pre-push browser-test LangGraph authority failed; refusing unaudited decision: \
                 {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.repo_browser_test_configured != evaluation.repo_browser_test_configured {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph authority mismatch: graph \
             repo_browser_test_configured={} but hook evaluation repo_browser_test_configured={}",
            graph_run.state.repo_browser_test_configured, evaluation.repo_browser_test_configured
        ));
    }

    if graph_run.state.frontend_changes != evaluation.frontend_changes {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph authority mismatch: graph frontend_changes={} but \
             hook evaluation frontend_changes={}",
            graph_run.state.frontend_changes, evaluation.frontend_changes
        ));
    }

    if graph_run.state.recent_browser_test != evaluation.recent_browser_test {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph authority mismatch: graph recent_browser_test={} \
             but hook evaluation recent_browser_test={}",
            graph_run.state.recent_browser_test, evaluation.recent_browser_test
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.pre_push_browser_authorization(),
        "Pre-push browser-test"
    );

    let expected_decision = pre_push_browser_test_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::pre_push_browser_test_graph::pre_push_browser_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::pre_push_browser_test_graph::pre_push_browser_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_pre_push_browser_test_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Pre-push browser-test LangGraph audit write failed; refusing unaudited decision: \
             {e:#}"
        ));
    }

    hooks::pre_push_browser_test::output_from_evaluation(input, &evaluation)
}

fn pre_push_browser_test_expected_decision(
    evaluation: &hooks::pre_push_browser_test::PrePushBrowserEvaluation,
) -> sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision {
    match evaluation.decision {
        hooks::pre_push_browser_test::PrePushBrowserDecision::Allow => {
            sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision::Allow
        }
        hooks::pre_push_browser_test::PrePushBrowserDecision::AllowNoBrowserConfig => {
            sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision::AllowNoBrowserConfig
        }
        hooks::pre_push_browser_test::PrePushBrowserDecision::AllowNoFrontendChanges => {
            sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision::AllowNoFrontendChanges
        }
        hooks::pre_push_browser_test::PrePushBrowserDecision::AllowRecentBrowserTest => {
            sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision::AllowRecentBrowserTest
        }
        hooks::pre_push_browser_test::PrePushBrowserDecision::Block => {
            sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserDecision::Block
        }
    }
}

fn run_pre_push_browser_test_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pre_push_browser_test::PrePushBrowserEvaluation,
) -> Result<sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserGraphRun> {
    let identifier = pre_push_browser_test_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::pre_push_browser_test_graph::
                build_pre_push_browser_graph()
                .await
            {
                Ok(graph) => graph,
                Err(e) => {
                    return Some(Err(anyhow!(
                        "build pre-push browser-test graph: {e}"
                    )));
                }
            };
            Some(
                sentinel_infrastructure::pre_push_browser_test_graph::
                    run_pre_push_browser_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run pre-push browser-test graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("pre-push browser-test graph timed out"))?
}

fn pre_push_browser_test_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pre_push_browser_test::PrePushBrowserEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let command = evaluation
        .command
        .as_deref()
        .filter(|command| !command.trim().is_empty())
        .ok_or_else(|| {
            anyhow!("pre-push browser-test LangGraph authority requires concrete command evidence")
        })?;
    let command_hash =
        sentinel_infrastructure::pre_push_browser_test_graph::command_sha256(command);
    Ok(format!(
        "{session_id}:{tool}:{command_hash}:block-{}:repo-browser-{}:frontend-{}:recent-{}",
        evaluation.should_block,
        evaluation.repo_browser_test_configured,
        evaluation.frontend_changes,
        evaluation.recent_browser_test
    ))
}

fn write_pre_push_browser_test_graph_audit(
    run: &sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserGraphRun,
    authorization: &sentinel_infrastructure::pre_push_browser_test_graph::PrePushBrowserAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("pre-push-browser-test.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create pre-push browser-test graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open pre-push browser-test graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "pre_push_browser_test",
        "decision": sentinel_infrastructure::pre_push_browser_test_graph::
            pre_push_browser_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write pre-push browser-test graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate pre-push browser-test graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_agent_revocation_with_graph(
    input: &sentinel_domain::events::HookInput,
    state: &SessionState,
) -> HookOutput {
    let evaluation = hooks::agent_revocation::evaluate(input, state);
    if !evaluation.graph_authority_required() {
        return hooks::agent_revocation::output_from_evaluation(input, &evaluation);
    }

    let graph_run = match run_agent_revocation_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Agent revocation LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.revoked != evaluation.revoked {
        return HookOutput::deny(format!(
            "Agent revocation LangGraph authority mismatch: graph revoked={} but hook \
             evaluation revoked={}",
            graph_run.state.revoked, evaluation.revoked
        ));
    }

    if graph_run.state.should_deny != evaluation.should_deny {
        return HookOutput::deny(format!(
            "Agent revocation LangGraph authority mismatch: graph should_deny={} but hook \
             evaluation should_deny={}",
            graph_run.state.should_deny, evaluation.should_deny
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.agent_revocation_authorization(),
        "Agent revocation"
    );

    let expected_decision = agent_revocation_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Agent revocation LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::agent_revocation_graph::agent_revocation_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::agent_revocation_graph::agent_revocation_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_agent_revocation_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Agent revocation LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::agent_revocation::output_from_evaluation(input, &evaluation)
}

fn agent_revocation_expected_decision(
    evaluation: &hooks::agent_revocation::AgentRevocationEvaluation,
) -> sentinel_infrastructure::agent_revocation_graph::AgentRevocationDecision {
    match evaluation.decision {
        hooks::agent_revocation::AgentRevocationDecision::Allow => {
            sentinel_infrastructure::agent_revocation_graph::AgentRevocationDecision::Allow
        }
        hooks::agent_revocation::AgentRevocationDecision::Deny => {
            sentinel_infrastructure::agent_revocation_graph::AgentRevocationDecision::Deny
        }
    }
}

fn run_agent_revocation_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::agent_revocation::AgentRevocationEvaluation,
) -> Result<sentinel_infrastructure::agent_revocation_graph::AgentRevocationGraphRun> {
    let identifier = agent_revocation_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::agent_revocation_graph::AgentRevocationState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::agent_revocation_graph::build_agent_revocation_graph(
                )
                .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build agent revocation graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::agent_revocation_graph::
                    run_agent_revocation_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run agent revocation graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("agent revocation graph timed out"))?
}

fn agent_revocation_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::agent_revocation::AgentRevocationEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:agent-present-{}:revoked-{}:deny-{}",
        evaluation.agent_id.is_some(),
        evaluation.revoked,
        evaluation.should_deny
    );
    if let Some(agent_id) = evaluation
        .agent_id
        .as_deref()
        .map(str::trim)
        .filter(|agent_id| !agent_id.is_empty())
    {
        identifier.push_str(":agent-sha256:");
        identifier.push_str(&sentinel_infrastructure::agent_revocation_graph::sha256(
            agent_id,
        ));
    }
    Ok(identifier)
}

fn write_agent_revocation_graph_audit(
    run: &sentinel_infrastructure::agent_revocation_graph::AgentRevocationGraphRun,
    authorization: &sentinel_infrastructure::agent_revocation_graph::AgentRevocationAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("agent-revocation.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create agent revocation graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open agent revocation graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "agent_revocation",
        "decision": sentinel_infrastructure::agent_revocation_graph::
            agent_revocation_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write agent revocation graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate agent revocation graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_step_gate_with_graph(
    input: &sentinel_domain::events::HookInput,
    state: &SessionState,
    step_configs: &HashMap<String, SkillSteps>,
) -> HookOutput {
    let evaluation = hooks::step_gate::evaluate(input, state, step_configs);
    if !evaluation.graph_authority_required() {
        return hooks::step_gate::output_from_evaluation(input, &evaluation);
    }

    let graph_run = match run_step_gate_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Step gate LangGraph authority failed; refusing unaudited step decision: {e:#}"
            ));
        }
    };

    for (name, graph_value, evaluation_value) in [
        ("step_tool", graph_run.state.step_tool, evaluation.step_tool),
        (
            "step_config_loaded",
            graph_run.state.step_config_loaded,
            evaluation.step_config_loaded,
        ),
        (
            "step_declared",
            graph_run.state.step_declared,
            evaluation.step_declared,
        ),
        (
            "prerequisite_present",
            graph_run.state.prerequisite_present,
            evaluation.prerequisite_present,
        ),
        (
            "graph_workflow_present",
            graph_run.state.graph_workflow_present,
            evaluation.graph_workflow_present,
        ),
        (
            "prerequisite_graph_completed",
            graph_run.state.prerequisite_graph_completed,
            evaluation.prerequisite_graph_completed,
        ),
        (
            "proof_chain_present",
            graph_run.state.proof_chain_present,
            evaluation.proof_chain_present,
        ),
        (
            "step_proof_present",
            graph_run.state.step_proof_present,
            evaluation.step_proof_present,
        ),
        (
            "should_deny",
            graph_run.state.should_deny,
            evaluation.should_deny,
        ),
    ] {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Step gate LangGraph authority mismatch: graph {name}={graph_value} but hook \
                 evaluation {name}={evaluation_value}"
            ));
        }
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.step_gate_authorization(), "Step gate");

    let expected_decision = step_gate_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Step gate LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::step_gate_graph::step_gate_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::step_gate_graph::step_gate_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_step_gate_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Step gate LangGraph audit write failed; refusing unaudited step decision: {e:#}"
        ));
    }

    hooks::step_gate::output_from_evaluation(input, &evaluation)
}

fn step_gate_expected_decision(
    evaluation: &hooks::step_gate::StepGateEvaluation,
) -> sentinel_infrastructure::step_gate_graph::StepGateDecision {
    match evaluation.decision {
        hooks::step_gate::StepGateDecision::Allow => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::Allow
        }
        hooks::step_gate::StepGateDecision::AllowFirstStep => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::AllowFirstStep
        }
        hooks::step_gate::StepGateDecision::AllowPrerequisiteProof => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::AllowPrerequisiteProof
        }
        hooks::step_gate::StepGateDecision::DenyMissingStepConfig => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyMissingStepConfig
        }
        hooks::step_gate::StepGateDecision::DenyStepNotDeclared => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyStepNotDeclared
        }
        hooks::step_gate::StepGateDecision::DenyMissingGraphWorkflow => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyMissingGraphWorkflow
        }
        hooks::step_gate::StepGateDecision::DenyPrerequisiteNotCompleted => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyPrerequisiteNotCompleted
        }
        hooks::step_gate::StepGateDecision::DenyMissingProofChain => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyMissingProofChain
        }
        hooks::step_gate::StepGateDecision::DenyMissingStepProof => {
            sentinel_infrastructure::step_gate_graph::StepGateDecision::DenyMissingStepProof
        }
    }
}

fn run_step_gate_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::step_gate::StepGateEvaluation,
) -> Result<sentinel_infrastructure::step_gate_graph::StepGateGraphRun> {
    let identifier = step_gate_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::step_gate_graph::StepGateState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::step_gate_graph::build_step_gate_graph().await {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build step gate graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::step_gate_graph::run_step_gate_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run step gate graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("step gate graph timed out"))?
}

fn step_gate_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::step_gate::StepGateEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:skill-present-{}:step-present-{}:deny-{}:decision-{}",
        evaluation.skill.is_some(),
        evaluation.step_id.is_some(),
        evaluation.should_deny,
        sentinel_infrastructure::step_gate_graph::step_gate_decision_label(
            step_gate_expected_decision(evaluation)
        )
    );
    if let Some(skill) = evaluation
        .skill
        .as_deref()
        .map(str::trim)
        .filter(|skill| !skill.is_empty())
    {
        identifier.push_str(":skill-sha256:");
        identifier.push_str(&sentinel_infrastructure::step_gate_graph::sha256(skill));
    }
    if let Some(step_id) = evaluation
        .step_id
        .as_deref()
        .map(str::trim)
        .filter(|step_id| !step_id.is_empty())
    {
        identifier.push_str(":step-sha256:");
        identifier.push_str(&sentinel_infrastructure::step_gate_graph::sha256(step_id));
    }
    Ok(identifier)
}

fn write_step_gate_graph_audit(
    run: &sentinel_infrastructure::step_gate_graph::StepGateGraphRun,
    authorization: &sentinel_infrastructure::step_gate_graph::StepGateAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("step-gate.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create step gate graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open step gate graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "step_gate",
        "decision": sentinel_infrastructure::step_gate_graph::
            step_gate_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write step gate graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate step gate graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_ticket_quality_with_graph(input: &sentinel_domain::events::HookInput) -> HookOutput {
    let evaluation = hooks::ticket_quality_gate::evaluate(input);
    if !evaluation.graph_authority_required() {
        return hooks::ticket_quality_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_ticket_quality_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Ticket quality LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_deny != evaluation.should_deny {
        return HookOutput::deny(format!(
            "Ticket quality LangGraph authority mismatch: graph should_deny={} but hook \
             evaluation should_deny={}",
            graph_run.state.should_deny, evaluation.should_deny
        ));
    }

    if graph_run.state.missing_field_count != evaluation.missing.len() as u64 {
        return HookOutput::deny(format!(
            "Ticket quality LangGraph authority mismatch: graph missing_field_count={} but hook \
             evaluation missing_field_count={}",
            graph_run.state.missing_field_count,
            evaluation.missing.len()
        ));
    }

    if graph_run.state.missing_estimate != evaluation.missing_estimate
        || graph_run.state.missing_priority != evaluation.missing_priority
        || graph_run.state.missing_label_ids != evaluation.missing_label_ids
        || graph_run.state.missing_description != evaluation.missing_description
    {
        return HookOutput::deny(
            "Ticket quality LangGraph authority mismatch: graph missing-field flags differ from \
             hook evaluation",
        );
    }

    let expected_malformed = matches!(
        evaluation.decision,
        hooks::ticket_quality_gate::TicketQualityDecision::DenyMalformedInput
    );
    if graph_run.state.malformed_input != expected_malformed {
        return HookOutput::deny(format!(
            "Ticket quality LangGraph authority mismatch: graph malformed_input={} but hook \
             evaluation malformed_input={expected_malformed}",
            graph_run.state.malformed_input
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.ticket_quality_authorization(),
        "Ticket quality"
    );

    let expected_decision = ticket_quality_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Ticket quality LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::ticket_quality_graph::ticket_quality_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::ticket_quality_graph::ticket_quality_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_ticket_quality_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Ticket quality LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::ticket_quality_gate::output_from_evaluation(&evaluation)
}

fn ticket_quality_expected_decision(
    evaluation: &hooks::ticket_quality_gate::TicketQualityEvaluation,
) -> sentinel_infrastructure::ticket_quality_graph::TicketQualityDecision {
    match evaluation.decision {
        hooks::ticket_quality_gate::TicketQualityDecision::Allow => {
            sentinel_infrastructure::ticket_quality_graph::TicketQualityDecision::Allow
        }
        hooks::ticket_quality_gate::TicketQualityDecision::DenyMalformedInput => {
            sentinel_infrastructure::ticket_quality_graph::TicketQualityDecision::DenyMalformedInput
        }
        hooks::ticket_quality_gate::TicketQualityDecision::DenyMissingFields => {
            sentinel_infrastructure::ticket_quality_graph::TicketQualityDecision::DenyMissingFields
        }
    }
}

fn run_ticket_quality_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::ticket_quality_gate::TicketQualityEvaluation,
) -> Result<sentinel_infrastructure::ticket_quality_graph::TicketQualityGraphRun> {
    let identifier = ticket_quality_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::ticket_quality_graph::TicketQualityState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::ticket_quality_graph::build_ticket_quality_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build ticket quality graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::ticket_quality_graph::run_ticket_quality_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run ticket quality graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("ticket quality graph timed out"))?
}

fn ticket_quality_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::ticket_quality_gate::TicketQualityEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:tool-input-present-{}:deny-{}:missing-{}:malformed-{}",
        evaluation.tool_input_present,
        evaluation.should_deny,
        evaluation.missing.len(),
        matches!(
            evaluation.decision,
            hooks::ticket_quality_gate::TicketQualityDecision::DenyMalformedInput
        )
    );
    if evaluation.tool_input_present {
        let input_hash = evaluation
            .tool_input_sha256
            .as_deref()
            .map(str::trim)
            .filter(|input_hash| !input_hash.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Ticket quality LangGraph authority requires concrete tool input digest when \
                     tool_input_present=true"
                )
            })?;
        identifier.push_str(":tool-input-sha256:");
        identifier.push_str(input_hash);
    }
    Ok(identifier)
}

fn write_ticket_quality_graph_audit(
    run: &sentinel_infrastructure::ticket_quality_graph::TicketQualityGraphRun,
    authorization: &sentinel_infrastructure::ticket_quality_graph::TicketQualityAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("ticket-quality.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create ticket quality graph audit dir {}", parent.display())
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open ticket quality graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "ticket_quality",
        "decision": sentinel_infrastructure::ticket_quality_graph::
            ticket_quality_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write ticket quality graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate ticket quality graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_tasks_md_guard_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::tasks_md_guard::evaluate(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::tasks_md_guard::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_tasks_md_guard_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Tasks.md guard LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Tasks.md guard LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.edit_overlaps_auto_block != evaluation.edit_overlaps_auto_block {
        return HookOutput::deny(format!(
            "Tasks.md guard LangGraph authority mismatch: graph edit_overlaps_auto_block={} but \
             hook evaluation edit_overlaps_auto_block={}",
            graph_run.state.edit_overlaps_auto_block, evaluation.edit_overlaps_auto_block
        ));
    }

    if graph_run.state.write_changes_auto_block != evaluation.write_changes_auto_block {
        return HookOutput::deny(format!(
            "Tasks.md guard LangGraph authority mismatch: graph write_changes_auto_block={} but \
             hook evaluation write_changes_auto_block={}",
            graph_run.state.write_changes_auto_block, evaluation.write_changes_auto_block
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.tasks_md_guard_authorization(),
        "Tasks.md guard"
    );

    let expected_decision = tasks_md_guard_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Tasks.md guard LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::tasks_md_guard_graph::tasks_md_guard_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::tasks_md_guard_graph::tasks_md_guard_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_tasks_md_guard_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Tasks.md guard LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::tasks_md_guard::output_from_evaluation(&evaluation)
}

fn tasks_md_guard_expected_decision(
    evaluation: &hooks::tasks_md_guard::TasksMdGuardEvaluation,
) -> sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardDecision {
    match evaluation.decision {
        hooks::tasks_md_guard::TasksMdGuardDecision::Allow => {
            sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardDecision::Allow
        }
        hooks::tasks_md_guard::TasksMdGuardDecision::Block => {
            sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardDecision::Block
        }
    }
}

fn run_tasks_md_guard_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::tasks_md_guard::TasksMdGuardEvaluation,
) -> Result<sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardGraphRun> {
    let identifier = tasks_md_guard_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::tasks_md_guard_graph::build_tasks_md_guard_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build tasks.md guard graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::tasks_md_guard_graph::run_tasks_md_guard_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run tasks.md guard graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("tasks.md guard graph timed out"))?
}

fn tasks_md_guard_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::tasks_md_guard::TasksMdGuardEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:file-path-present-{}:block-{}:edit-{}:write-{}",
        evaluation.file_path_present,
        evaluation.should_block,
        evaluation.edit_overlaps_auto_block,
        evaluation.write_changes_auto_block
    );
    if evaluation.file_path_present {
        let file_path = evaluation
            .file_path
            .as_deref()
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Tasks.md guard LangGraph authority requires concrete file path evidence when \
                     file_path_present=true"
                )
            })?;
        identifier.push_str(":file-path-sha256:");
        identifier.push_str(&sentinel_infrastructure::tasks_md_guard_graph::sha256(
            file_path,
        ));
    }
    Ok(identifier)
}

fn write_tasks_md_guard_graph_audit(
    run: &sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardGraphRun,
    authorization: &sentinel_infrastructure::tasks_md_guard_graph::TasksMdGuardAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("tasks-md-guard.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create tasks.md guard graph audit dir {}", parent.display())
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open tasks.md guard graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "tasks_md_guard",
        "decision": sentinel_infrastructure::tasks_md_guard_graph::
            tasks_md_guard_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write tasks.md guard graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate tasks.md guard graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_production_override_with_graph(
    input: &sentinel_domain::events::HookInput,
    state: &mut SessionState,
) -> HookOutput {
    let evaluation = hooks::production_override::evaluate(input, state.production_override_armed());

    let graph_run = match run_production_override_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Production override LangGraph authority failed; refusing unaudited transition: \
                 {e:#}"
            ));
        }
    };

    let expected_transition =
        sentinel_infrastructure::production_override_graph::transition_label(evaluation.transition);
    if graph_run.state.transition != expected_transition {
        return HookOutput::deny(format!(
            "Production override LangGraph authority mismatch: graph transition={} but hook \
             evaluation transition={expected_transition}",
            graph_run.state.transition
        ));
    }

    if graph_run.state.target_armed != evaluation.target_armed {
        return HookOutput::deny(format!(
            "Production override LangGraph authority mismatch: graph target_armed={} but hook \
             evaluation target_armed={}",
            graph_run.state.target_armed, evaluation.target_armed
        ));
    }

    if graph_run.state.notice_required != evaluation.notice_required {
        return HookOutput::deny(format!(
            "Production override LangGraph authority mismatch: graph notice_required={} but hook \
             evaluation notice_required={}",
            graph_run.state.notice_required, evaluation.notice_required
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.production_override_authorization(),
        "Production override"
    );

    let expected_decision = production_override_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Production override LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::production_override_graph::production_override_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::production_override_graph::production_override_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_production_override_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Production override LangGraph audit write failed; refusing unaudited transition: \
             {e:#}"
        ));
    }

    hooks::production_override::apply_authorized_evaluation(state, &evaluation)
}

fn production_override_expected_decision(
    evaluation: &hooks::production_override::ProductionOverrideEvaluation,
) -> sentinel_infrastructure::production_override_graph::ProductionOverrideDecision {
    match evaluation.decision {
        hooks::production_override::ProductionOverrideDecision::AllowNoop => {
            sentinel_infrastructure::production_override_graph::ProductionOverrideDecision::AllowNoop
        }
        hooks::production_override::ProductionOverrideDecision::Arm => {
            sentinel_infrastructure::production_override_graph::ProductionOverrideDecision::Arm
        }
        hooks::production_override::ProductionOverrideDecision::Lock => {
            sentinel_infrastructure::production_override_graph::ProductionOverrideDecision::Lock
        }
    }
}

fn run_production_override_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::production_override::ProductionOverrideEvaluation,
) -> Result<sentinel_infrastructure::production_override_graph::ProductionOverrideGraphRun> {
    let identifier = production_override_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::production_override_graph::ProductionOverrideState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::production_override_graph::
                build_production_override_graph()
                .await
            {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build production override graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::production_override_graph::
                    run_production_override_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run production override graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("production override graph timed out"))?
}

fn production_override_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::production_override::ProductionOverrideEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, evaluation.session_id.as_deref())?;
    let transition =
        sentinel_infrastructure::production_override_graph::transition_label(evaluation.transition);
    let mut identifier = format!(
        "{session_id}:prompt-present-{}:{transition}:prior-{}:target-{}:notice-{}",
        evaluation.prompt_present,
        evaluation.prior_armed,
        evaluation.target_armed,
        evaluation.notice_required
    );
    if let Some(prompt_hash) = evaluation.prompt_sha256.as_deref() {
        identifier.push_str(":prompt-sha256:");
        identifier.push_str(prompt_hash);
    }
    Ok(identifier)
}

fn write_production_override_graph_audit(
    run: &sentinel_infrastructure::production_override_graph::ProductionOverrideGraphRun,
    authorization: &sentinel_infrastructure::production_override_graph::ProductionOverrideAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("production-override.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create production override graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open production override graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "production_override",
        "decision": sentinel_infrastructure::production_override_graph::production_override_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write production override graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate production override graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_production_action_notice_with_graph(
    input: &sentinel_domain::events::HookInput,
    state: &SessionState,
) -> HookOutput {
    let evaluation =
        hooks::production_action_notice::evaluate(input, state.production_override_armed());
    if !evaluation.graph_authority_required() {
        return hooks::production_action_notice::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_production_action_notice_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Production action notice LangGraph authority failed; refusing unaudited \
                 production action: {e:#}"
            ));
        }
    };

    for (name, graph_value, evaluation_value) in [
        (
            "production_override_armed",
            graph_run.state.production_override_armed,
            evaluation.production_override_armed,
        ),
        (
            "tool_present",
            graph_run.state.tool_present,
            evaluation.tool_present,
        ),
        ("pure_read", graph_run.state.pure_read, evaluation.pure_read),
        (
            "mutating_tool",
            graph_run.state.mutating_tool,
            evaluation.mutating_tool,
        ),
        (
            "mentions_prod",
            graph_run.state.mentions_prod,
            evaluation.mentions_prod,
        ),
        (
            "should_notice",
            graph_run.state.should_notice,
            evaluation.should_notice,
        ),
    ] {
        if graph_value != evaluation_value {
            return HookOutput::deny(format!(
                "Production action notice LangGraph authority mismatch: graph {name}={graph_value} \
                 but hook evaluation {name}={evaluation_value}"
            ));
        }
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.production_action_notice_authorization(),
        "Production action notice"
    );

    let expected_decision = production_action_notice_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Production action notice LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::production_action_notice_graph::
                production_action_notice_decision_label(authorization.decision()),
            sentinel_infrastructure::production_action_notice_graph::
                production_action_notice_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_production_action_notice_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Production action notice LangGraph audit write failed; refusing unaudited production \
             action: {e:#}"
        ));
    }

    hooks::production_action_notice::output_from_evaluation(&evaluation)
}

fn production_action_notice_expected_decision(
    evaluation: &hooks::production_action_notice::ProductionActionNoticeEvaluation,
) -> sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeDecision {
    match evaluation.decision {
        hooks::production_action_notice::ProductionActionNoticeDecision::AllowSilent => {
            sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeDecision::
                AllowSilent
        }
        hooks::production_action_notice::ProductionActionNoticeDecision::Notice => {
            sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeDecision::
                Notice
        }
    }
}

fn run_production_action_notice_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::production_action_notice::ProductionActionNoticeEvaluation,
) -> Result<sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeGraphRun>
{
    let identifier = production_action_notice_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeState::from_evaluation(
            identifier, evaluation,
        );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::production_action_notice_graph::
                build_production_action_notice_graph()
                .await
            {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build production action notice graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::production_action_notice_graph::
                    run_production_action_notice_decision_report(&graph, state)
                        .await
                        .map_err(|e| anyhow!("run production action notice graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("production action notice graph timed out"))?
}

fn production_action_notice_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::production_action_notice::ProductionActionNoticeEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:haystack-present-{}:notice-{}:prod-{}",
        evaluation.haystack_present, evaluation.should_notice, evaluation.mentions_prod
    );
    if evaluation.haystack_present {
        identifier.push_str(":haystack-sha256:");
        identifier.push_str(
            &sentinel_infrastructure::production_action_notice_graph::sha256(&evaluation.haystack),
        );
    }
    Ok(identifier)
}

fn write_production_action_notice_graph_audit(
    run: &sentinel_infrastructure::production_action_notice_graph::ProductionActionNoticeGraphRun,
    authorization: &sentinel_infrastructure::production_action_notice_graph::
        ProductionActionNoticeAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("production-action-notice.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create production action notice graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open production action notice graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "production_action_notice",
        "decision": sentinel_infrastructure::production_action_notice_graph::
            production_action_notice_decision_label(authorization.decision()),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write production action notice graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate production action notice graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_pr_merge_with_graph(
    input: &sentinel_domain::events::HookInput,
    env: &dyn hooks::EnvPort,
    process: &dyn hooks::ProcessPort,
) -> HookOutput {
    let evaluation = hooks::pr_merge_gate::evaluate(input, env, process);
    if !evaluation.graph_authority_required() {
        return hooks::pr_merge_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_pr_merge_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "PR merge LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.permission_prompt_required != evaluation.permission_prompt_required {
        return HookOutput::deny(format!(
            "PR merge LangGraph authority mismatch: graph permission_prompt_required={} but \
             hook evaluation permission_prompt_required={}",
            graph_run.state.permission_prompt_required, evaluation.permission_prompt_required
        ));
    }

    if graph_run.state.context_reminder_required != evaluation.context_reminder_required {
        return HookOutput::deny(format!(
            "PR merge LangGraph authority mismatch: graph context_reminder_required={} but hook \
             evaluation context_reminder_required={}",
            graph_run.state.context_reminder_required, evaluation.context_reminder_required
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.pr_merge_authorization(), "PR merge");

    let expected_decision = pr_merge_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "PR merge LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::pr_merge_graph::pr_merge_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::pr_merge_graph::pr_merge_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_pr_merge_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "PR merge LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::pr_merge_gate::output_from_evaluation(&evaluation)
}

fn pr_merge_expected_decision(
    evaluation: &hooks::pr_merge_gate::PrMergeEvaluation,
) -> sentinel_infrastructure::pr_merge_graph::PrMergeDecision {
    match evaluation.decision {
        hooks::pr_merge_gate::PrMergeDecision::Allow => {
            sentinel_infrastructure::pr_merge_graph::PrMergeDecision::Allow
        }
        hooks::pr_merge_gate::PrMergeDecision::Ask => {
            sentinel_infrastructure::pr_merge_graph::PrMergeDecision::Ask
        }
        hooks::pr_merge_gate::PrMergeDecision::AllowAutopilotReminder => {
            sentinel_infrastructure::pr_merge_graph::PrMergeDecision::AllowAutopilotReminder
        }
        hooks::pr_merge_gate::PrMergeDecision::Deny => {
            sentinel_infrastructure::pr_merge_graph::PrMergeDecision::Deny
        }
    }
}

fn run_pr_merge_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pr_merge_gate::PrMergeEvaluation,
) -> Result<sentinel_infrastructure::pr_merge_graph::PrMergeGraphRun> {
    let identifier = pr_merge_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::pr_merge_graph::PrMergeState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::pr_merge_graph::build_pr_merge_graph().await
            {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build PR merge graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::pr_merge_graph::run_pr_merge_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run PR merge graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("PR merge graph timed out"))?
}

fn pr_merge_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::pr_merge_gate::PrMergeEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let operation = sentinel_infrastructure::pr_merge_graph::operation_label(evaluation.operation);
    let command = evaluation
        .command
        .as_deref()
        .filter(|command| !command.trim().is_empty())
        .ok_or_else(|| {
            anyhow!("PR merge LangGraph authority requires concrete command evidence")
        })?;
    let command_hash = sentinel_infrastructure::pr_merge_graph::command_sha256(command);
    // The fetched verdict is part of the thread identity: the same command can
    // legitimately resolve to allow/deny/ask on different fetches (a PR gets
    // approved, CI goes green), and each must be a distinct audited decision.
    Ok(format!(
        "{session_id}:{tool}:{operation}:{command_hash}:verdict-{}:ask-{}:context-{}",
        evaluation.verdict.label(),
        evaluation.permission_prompt_required,
        evaluation.context_reminder_required
    ))
}

fn write_pr_merge_graph_audit(
    run: &sentinel_infrastructure::pr_merge_graph::PrMergeGraphRun,
    authorization: &sentinel_infrastructure::pr_merge_graph::PrMergeAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("pr-merge.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create PR merge graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open PR merge graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "pr_merge",
        "decision": sentinel_infrastructure::pr_merge_graph::pr_merge_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write PR merge graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate PR merge graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_db_ops_with_graph(input: &sentinel_domain::events::HookInput) -> HookOutput {
    let evaluation = hooks::db_ops_gate::evaluate(input);
    if !evaluation.graph_authority_required() {
        return hooks::db_ops_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_db_ops_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Database operations LangGraph authority failed; refusing unaudited decision: \
                 {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Database operations LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.database_operation != evaluation.database_operation {
        return HookOutput::deny(format!(
            "Database operations LangGraph authority mismatch: graph database_operation={} but \
             hook evaluation database_operation={}",
            graph_run.state.database_operation, evaluation.database_operation
        ));
    }

    if graph_run.state.production_target != evaluation.production_target {
        return HookOutput::deny(format!(
            "Database operations LangGraph authority mismatch: graph production_target={} but \
             hook evaluation production_target={}",
            graph_run.state.production_target, evaluation.production_target
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.db_ops_authorization(), "Database operations");

    let expected_decision = db_ops_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Database operations LangGraph authority mismatch: graph decision={} but hook \
             evaluation decision={}",
            sentinel_infrastructure::db_ops_graph::db_ops_decision_label(authorization.decision()),
            sentinel_infrastructure::db_ops_graph::db_ops_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_db_ops_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Database operations LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::db_ops_gate::output_from_evaluation(&evaluation)
}

fn db_ops_expected_decision(
    evaluation: &hooks::db_ops_gate::DbOpsEvaluation,
) -> sentinel_infrastructure::db_ops_graph::DbOpsDecision {
    match evaluation.decision {
        hooks::db_ops_gate::DbOpsDecision::Allow => {
            sentinel_infrastructure::db_ops_graph::DbOpsDecision::Allow
        }
        hooks::db_ops_gate::DbOpsDecision::Block => {
            sentinel_infrastructure::db_ops_graph::DbOpsDecision::Block
        }
    }
}

fn run_db_ops_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::db_ops_gate::DbOpsEvaluation,
) -> Result<sentinel_infrastructure::db_ops_graph::DbOpsGraphRun> {
    let identifier = db_ops_graph_identifier(input, evaluation)?;
    let state =
        sentinel_infrastructure::db_ops_graph::DbOpsState::from_evaluation(identifier, evaluation);
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::db_ops_graph::build_db_ops_graph().await {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build database operations graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::db_ops_graph::run_db_ops_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run database operations graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("database operations graph timed out"))?
}

fn db_ops_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::db_ops_gate::DbOpsEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:command-present-{}:block-{}:prod-{}",
        evaluation.command_present, evaluation.should_block, evaluation.production_target
    );
    if evaluation.command_present {
        let command = evaluation
            .command
            .as_deref()
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "Database operations LangGraph authority requires concrete command evidence \
                     when command_present=true"
                )
            })?;
        identifier.push_str(":command-sha256:");
        identifier.push_str(&sentinel_infrastructure::db_ops_graph::command_sha256(
            command,
        ));
    }
    Ok(identifier)
}

fn write_db_ops_graph_audit(
    run: &sentinel_infrastructure::db_ops_graph::DbOpsGraphRun,
    authorization: &sentinel_infrastructure::db_ops_graph::DbOpsAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("db-ops.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create database operations graph audit dir {}",
                parent.display()
            )
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| {
            format!(
                "open database operations graph audit {}",
                graph_runs.display()
            )
        })?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "db_ops",
        "decision": sentinel_infrastructure::db_ops_graph::db_ops_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row).with_context(|| {
        format!(
            "write database operations graph audit {}",
            graph_runs.display()
        )
    })?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate database operations graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_doppler_auth0_with_graph(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
) -> HookOutput {
    let evaluation = hooks::doppler_auth0_gate::evaluate(input, ctx);
    if !evaluation.graph_authority_required() {
        return hooks::doppler_auth0_gate::output_from_evaluation(&evaluation);
    }

    let graph_run = match run_doppler_auth0_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "Doppler/Auth0 LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "Doppler/Auth0 LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.signed_override_active != evaluation.signed_override_active {
        return HookOutput::deny(format!(
            "Doppler/Auth0 LangGraph authority mismatch: graph signed_override_active={} but \
             hook evaluation signed_override_active={}",
            graph_run.state.signed_override_active, evaluation.signed_override_active
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.doppler_auth0_authorization(), "Doppler/Auth0");

    let expected_decision = doppler_auth0_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "Doppler/Auth0 LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::doppler_auth0_graph::doppler_auth0_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::doppler_auth0_graph::doppler_auth0_decision_label(
                expected_decision
            )
        ));
    }

    if let Err(e) = write_doppler_auth0_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "Doppler/Auth0 LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::doppler_auth0_gate::output_from_evaluation(&evaluation)
}

fn doppler_auth0_expected_decision(
    evaluation: &hooks::doppler_auth0_gate::DopplerAuth0Evaluation,
) -> sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision {
    match evaluation.decision {
        hooks::doppler_auth0_gate::DopplerAuth0Decision::Allow => {
            sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision::Allow
        }
        hooks::doppler_auth0_gate::DopplerAuth0Decision::AllowReadOnly => {
            sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision::AllowReadOnly
        }
        hooks::doppler_auth0_gate::DopplerAuth0Decision::AllowAutopilotNonProd => {
            sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision::AllowAutopilotNonProd
        }
        hooks::doppler_auth0_gate::DopplerAuth0Decision::AllowSignedOverride => {
            sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision::AllowSignedOverride
        }
        hooks::doppler_auth0_gate::DopplerAuth0Decision::Block => {
            sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Decision::Block
        }
    }
}

fn run_doppler_auth0_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::doppler_auth0_gate::DopplerAuth0Evaluation,
) -> Result<sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0GraphRun> {
    let identifier = doppler_auth0_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0State::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::doppler_auth0_graph::build_doppler_auth0_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build Doppler/Auth0 graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::doppler_auth0_graph::run_doppler_auth0_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run Doppler/Auth0 graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("Doppler/Auth0 graph timed out"))?
}

fn doppler_auth0_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::doppler_auth0_gate::DopplerAuth0Evaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let operation = evaluation
        .operation
        .as_deref()
        .map(str::trim)
        .filter(|operation| !operation.is_empty())
        .ok_or_else(|| {
            anyhow!("Doppler/Auth0 LangGraph authority requires concrete operation evidence")
        })?;
    let provider =
        sentinel_infrastructure::doppler_auth0_graph::provider_label(evaluation.provider);
    Ok(format!(
        "{session_id}:{tool}:provider-{provider}:operation-{operation}:tool-input-present-{}:\
         prod-{}:autopilot-{}:read-only-{}:mutation-{}:block-{}:override-{}",
        evaluation.tool_input_present,
        evaluation.production_target,
        evaluation.autopilot,
        evaluation.read_only,
        evaluation.mutation,
        evaluation.should_block,
        evaluation.signed_override_active
    ))
}

fn write_doppler_auth0_graph_audit(
    run: &sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0GraphRun,
    authorization: &sentinel_infrastructure::doppler_auth0_graph::DopplerAuth0Authorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("doppler-auth0.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create Doppler/Auth0 graph audit dir {}", parent.display())
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open Doppler/Auth0 graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "doppler_auth0",
        "decision": sentinel_infrastructure::doppler_auth0_graph::doppler_auth0_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write Doppler/Auth0 graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n").with_context(|| {
        format!(
            "terminate Doppler/Auth0 graph audit {}",
            graph_runs.display()
        )
    })?;
    Ok(())
}

fn authorize_dry_run_then_commit_with_graph(
    input: &sentinel_domain::events::HookInput,
    fs: &dyn sentinel_application::hooks::FileSystemPort,
    classifier: &dyn ReversibilityClassifierPort,
    auditor: &dyn AuditorPort,
) -> HookOutput {
    let evaluation = hooks::dry_run_then_commit::evaluate(input, fs, classifier, auditor);
    if !evaluation.graph_authority_required() {
        return hooks::dry_run_then_commit::output_from_evaluation(fs, &evaluation);
    }

    let graph_run = match run_a3_dry_run_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "A3 dry-run LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "A3 dry-run LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    if graph_run.state.approval_marker_should_be_recorded
        != evaluation.approval_marker_should_be_recorded
    {
        return HookOutput::deny(format!(
            "A3 dry-run LangGraph authority mismatch: graph approval_marker_should_be_recorded={} \
             but hook evaluation approval_marker_should_be_recorded={}",
            graph_run.state.approval_marker_should_be_recorded,
            evaluation.approval_marker_should_be_recorded
        ));
    }

    let authorization =
        require_hook_graph_authorization!(graph_run.dry_run_authorization(), "A3 dry-run");

    let expected_decision = a3_dry_run_expected_decision(&evaluation);
    if authorization.decision() != expected_decision {
        return HookOutput::deny(format!(
            "A3 dry-run LangGraph authority mismatch: graph decision={} but hook evaluation \
             decision={}",
            sentinel_infrastructure::dry_run_graph::dry_run_decision_label(
                authorization.decision()
            ),
            sentinel_infrastructure::dry_run_graph::dry_run_decision_label(expected_decision)
        ));
    }

    if let Err(e) = write_a3_dry_run_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "A3 dry-run LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::dry_run_then_commit::output_from_evaluation(fs, &evaluation)
}

fn a3_dry_run_expected_decision(
    evaluation: &hooks::dry_run_then_commit::DryRunGateEvaluation,
) -> sentinel_infrastructure::dry_run_graph::DryRunDecision {
    match evaluation.decision {
        hooks::dry_run_then_commit::DryRunGateDecision::Allow => {
            sentinel_infrastructure::dry_run_graph::DryRunDecision::Allow
        }
        hooks::dry_run_then_commit::DryRunGateDecision::AllowAndRecordApproval => {
            sentinel_infrastructure::dry_run_graph::DryRunDecision::AllowAndRecordApproval
        }
        hooks::dry_run_then_commit::DryRunGateDecision::Block => {
            sentinel_infrastructure::dry_run_graph::DryRunDecision::Block
        }
    }
}

fn run_a3_dry_run_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::dry_run_then_commit::DryRunGateEvaluation,
) -> Result<sentinel_infrastructure::dry_run_graph::DryRunGraphRun> {
    let identifier = a3_dry_run_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::dry_run_graph::DryRunState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph = match sentinel_infrastructure::dry_run_graph::build_dry_run_graph().await {
                Ok(graph) => graph,
                Err(e) => return Some(Err(anyhow!("build A3 dry-run graph: {e}"))),
            };
            Some(
                sentinel_infrastructure::dry_run_graph::run_dry_run_decision_report(&graph, state)
                    .await
                    .map_err(|e| anyhow!("run A3 dry-run graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("A3 dry-run graph timed out"))?
}

fn a3_dry_run_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::dry_run_then_commit::DryRunGateEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, evaluation.session_id.as_deref())?;
    let tool = required_graph_tool(input, evaluation.tool.as_deref())?;
    let mut identifier = format!(
        "{session_id}:{tool}:action-hash-present-{}:block-{}:record-{}",
        evaluation.action_hash.is_some(),
        evaluation.should_block,
        evaluation.approval_marker_should_be_recorded
    );
    if let Some(action_hash) = evaluation
        .action_hash
        .as_deref()
        .map(str::trim)
        .filter(|action_hash| !action_hash.is_empty())
    {
        identifier.push_str(":action-hash:");
        identifier.push_str(action_hash);
    }
    Ok(identifier)
}

fn write_a3_dry_run_graph_audit(
    run: &sentinel_infrastructure::dry_run_graph::DryRunGraphRun,
    authorization: &sentinel_infrastructure::dry_run_graph::DryRunAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("a3-dry-run.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create A3 dry-run graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open A3 dry-run graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "dry_run",
        "decision": sentinel_infrastructure::dry_run_graph::dry_run_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write A3 dry-run graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate A3 dry-run graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_spec_challenge_with_graph(
    input: &sentinel_domain::events::HookInput,
    class: sentinel_domain::reversibility::ReversibilityClass,
    spec_challenge_store: Option<&FilesystemSpecChallengeStore>,
    spec_challenge_scorer: Option<&LlmSpecChallengeScorer>,
    mode: hooks::spec_challenge_gate::A13EnforcementMode,
    catastrophic_axis_threshold: f32,
) -> HookOutput {
    if !hooks::spec_challenge_gate::is_a13_signal(input, class) {
        return HookOutput::allow();
    }

    let Some(store) = spec_challenge_store else {
        return HookOutput::deny(
            "A13 spec-challenge validation needs a SpecChallenge store adapter; refusing to \
             downgrade to an unaudited allow",
        );
    };

    let Some(evaluation) = hooks::spec_challenge_gate::evaluate_with_threshold(
        input,
        class,
        Some(store as &dyn sentinel_domain::ports::SpecChallengeStorePort),
        spec_challenge_scorer.map(|s| s as &dyn sentinel_domain::ports::SpecChallengeScorerPort),
        mode,
        catastrophic_axis_threshold,
    ) else {
        return HookOutput::allow();
    };

    let graph_run = match run_a13_spec_challenge_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "A13 spec-challenge LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "A13 spec-challenge LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.spec_challenge_authorization(),
        "A13 spec-challenge"
    );

    if let Err(e) = write_a13_spec_challenge_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "A13 spec-challenge LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::spec_challenge_gate::output_from_evaluation(&evaluation)
}

fn run_a13_spec_challenge_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::spec_challenge_gate::SpecChallengeEvaluation,
) -> Result<sentinel_infrastructure::spec_challenge_graph::SpecChallengeRun> {
    let identifier = a13_spec_challenge_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::spec_challenge_graph::SpecChallengeState::from_evaluation(
        identifier, evaluation,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::spec_challenge_graph::build_spec_challenge_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build A13 spec-challenge graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::spec_challenge_graph::run_spec_challenge_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run A13 spec-challenge graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("A13 spec-challenge graph timed out"))?
}

fn a13_spec_challenge_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::spec_challenge_gate::SpecChallengeEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, None)?;
    let mut identifier = format!(
        "{session_id}:{tool}:challenge-present-{}:findings-{}",
        evaluation.challenge.is_some(),
        sentinel_infrastructure::spec_challenge_graph::SpecChallengeState::from_evaluation(
            "identifier-preview",
            evaluation,
        )
        .blocking_finding_count
    );
    if let Some(work_id) = evaluation
        .challenge
        .as_ref()
        .map(|challenge| challenge.work_id.as_str())
        .map(str::trim)
        .filter(|work_id| !work_id.is_empty())
    {
        identifier.push_str(":work-id:");
        identifier.push_str(work_id);
    }
    Ok(identifier)
}

fn write_a13_spec_challenge_graph_audit(
    run: &sentinel_infrastructure::spec_challenge_graph::SpecChallengeRun,
    authorization: &sentinel_infrastructure::spec_challenge_graph::SpecChallengeAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("a13-spec-challenge.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create A13 graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open A13 graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "spec_challenge",
        "decision": sentinel_infrastructure::spec_challenge_graph::spec_challenge_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write A13 graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate A13 graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_provenance_validate_with_graph(
    input: &sentinel_domain::events::HookInput,
    provenance_store: Option<&JsonlProvenanceStore>,
    mode: hooks::provenance_validate::ValidationMode,
) -> HookOutput {
    if !hooks::provenance_validate::is_ba1_signal(input) {
        return HookOutput::allow();
    }

    let Some(provenance) = provenance_store else {
        return HookOutput::deny(
            "BA1 provenance validation needs a provenance store adapter; refusing to downgrade \
             to an unaudited allow",
        );
    };

    let Some(evaluation) = hooks::provenance_validate::evaluate(input, provenance, mode) else {
        return HookOutput::allow();
    };

    let graph_run = match run_ba1_provenance_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "BA1 provenance LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "BA1 provenance LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.ba_provenance_authorization(),
        "BA1 provenance"
    );

    if let Err(e) = write_ba1_provenance_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "BA1 provenance LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::provenance_validate::output_from_evaluation(&evaluation)
}

fn run_ba1_provenance_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::provenance_validate::ProvenanceValidationEvaluation,
) -> Result<sentinel_infrastructure::ba_provenance_graph::BaProvenanceRun> {
    let identifier = ba1_provenance_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::ba_provenance_graph::BaProvenanceState::from_checks(
        identifier,
        &evaluation.checks,
        evaluation.mode,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::ba_provenance_graph::build_ba_provenance_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build BA1 provenance graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::ba_provenance_graph::run_ba_provenance_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run BA1 provenance graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("BA1 provenance graph timed out"))?
}

fn ba1_provenance_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::provenance_validate::ProvenanceValidationEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, None)?;
    Ok(format!(
        "{session_id}:{tool}:citations-{}:findings-{}",
        evaluation.checks.len(),
        evaluation
            .checks
            .iter()
            .map(|check| check.findings.len())
            .sum::<usize>()
    ))
}

fn write_ba1_provenance_graph_audit(
    run: &sentinel_infrastructure::ba_provenance_graph::BaProvenanceRun,
    authorization: &sentinel_infrastructure::ba_provenance_graph::BaProvenanceAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("ba-provenance.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create BA1 graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open BA1 graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "ba_provenance",
        "decision": sentinel_infrastructure::ba_provenance_graph::ba_provenance_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write BA1 graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate BA1 graph audit {}", graph_runs.display()))?;
    Ok(())
}

fn authorize_requirements_traceability_with_graph(
    input: &sentinel_domain::events::HookInput,
    requirement_matrix: Option<&FilesystemRequirementMatrix>,
    mode: hooks::requirements_traceability_gate::ValidationMode,
) -> HookOutput {
    if !hooks::requirements_traceability_gate::is_ba3_signal(input) {
        return HookOutput::allow();
    }

    let Some(matrix) = requirement_matrix else {
        return HookOutput::deny(
            "BA3 requirements traceability needs a requirement-matrix adapter; refusing to \
             downgrade to an unaudited allow",
        );
    };

    let Some(evaluation) = hooks::requirements_traceability_gate::evaluate(input, matrix, mode)
    else {
        return HookOutput::allow();
    };

    let graph_run = match run_ba3_requirements_graph(input, &evaluation) {
        Ok(run) => run,
        Err(e) => {
            return HookOutput::deny(format!(
                "BA3 requirements LangGraph authority failed; refusing unaudited decision: {e:#}"
            ));
        }
    };

    if graph_run.state.should_block != evaluation.should_block {
        return HookOutput::deny(format!(
            "BA3 requirements LangGraph authority mismatch: graph should_block={} but hook \
             evaluation should_block={}",
            graph_run.state.should_block, evaluation.should_block
        ));
    }

    let authorization = require_hook_graph_authorization!(
        graph_run.ba_requirements_authorization(),
        "BA3 requirements"
    );

    if let Err(e) = write_ba3_requirements_graph_audit(&graph_run, &authorization) {
        return HookOutput::deny(format!(
            "BA3 requirements LangGraph audit write failed; refusing unaudited decision: {e:#}"
        ));
    }

    hooks::requirements_traceability_gate::output_from_evaluation(&evaluation)
}

fn run_ba3_requirements_graph(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::requirements_traceability_gate::RequirementsTraceabilityEvaluation,
) -> Result<sentinel_infrastructure::ba_requirements_graph::BaRequirementsRun> {
    let identifier = ba3_requirements_graph_identifier(input, evaluation)?;
    let state = sentinel_infrastructure::ba_requirements_graph::BaRequirementsState::from_check(
        identifier,
        &evaluation.check,
        evaluation.mode,
    );
    let result = hooks::run_async_timeout(
        async move {
            let graph =
                match sentinel_infrastructure::ba_requirements_graph::build_ba_requirements_graph()
                    .await
                {
                    Ok(graph) => graph,
                    Err(e) => return Some(Err(anyhow!("build BA3 requirements graph: {e}"))),
                };
            Some(
                sentinel_infrastructure::ba_requirements_graph::run_ba_requirements_decision_report(
                    &graph, state,
                )
                .await
                .map_err(|e| anyhow!("run BA3 requirements graph: {e}")),
            )
        },
        std::time::Duration::from_secs(2),
    );
    result.ok_or_else(|| anyhow!("BA3 requirements graph timed out"))?
}

fn ba3_requirements_graph_identifier(
    input: &sentinel_domain::events::HookInput,
    evaluation: &hooks::requirements_traceability_gate::RequirementsTraceabilityEvaluation,
) -> Result<String> {
    let session_id = required_graph_session(input, None)?;
    let tool = required_graph_tool(input, None)?;
    Ok(format!(
        "{session_id}:{tool}:refs-{}:findings-{}",
        evaluation.check.references.len(),
        evaluation.check.findings.len()
    ))
}

fn write_ba3_requirements_graph_audit(
    run: &sentinel_infrastructure::ba_requirements_graph::BaRequirementsRun,
    authorization: &sentinel_infrastructure::ba_requirements_graph::BaRequirementsAuthorization,
) -> Result<()> {
    let graph_runs = sentinel_infrastructure::paths::sentinel_root()
        .join("metrics")
        .join("ba-requirements.graph-runs.jsonl");
    if let Some(parent) = graph_runs.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create BA3 graph audit dir {}", parent.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&graph_runs)
        .with_context(|| format!("open BA3 graph audit {}", graph_runs.display()))?;
    let row = serde_json::json!({
        "workflow_authority": "langgraph",
        "graph": "ba_requirements",
        "decision": sentinel_infrastructure::ba_requirements_graph::ba_requirements_decision_label(
            authorization.decision()
        ),
        "authorization_checkpoint": authorization.checkpoint_ref(),
        "thread_id": run.thread_id.clone(),
        "run": run,
    });
    serde_json::to_writer(&mut file, &row)
        .with_context(|| format!("write BA3 graph audit {}", graph_runs.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate BA3 graph audit {}", graph_runs.display()))?;
    Ok(())
}

/// Handle `PostToolUse`: audit extract, bug/skill gate post-processing,
/// MCP health, todo interceptor, step judge, …
async fn handle_post_tool_use(
    input: &sentinel_domain::events::HookInput,
    state: &mut sentinel_domain::state::SessionState,
    ctx: &hooks::HookContext<'_>,
    step_configs: &HashMap<String, sentinel_domain::workflow::SkillSteps>,
    provenance_store: Option<&JsonlProvenanceStore>,
) -> HookOutput {
    let mut output = HookOutput::allow();

    let cwd_for_metrics = input.cwd.as_deref().unwrap_or(".");
    let repo_root_for_metrics = ctx.git.repo_root(cwd_for_metrics);
    let mk_ctx = |hook: &'static str| InvocationContext {
        event: "PostToolUse",
        hook,
        tool: input.tool_name.as_deref(),
        session_id: input.session_id.as_deref(),
        repo_root: repo_root_for_metrics.as_deref(),
    };

    // BA1 audit-extract — lift documented-connector retrievals into
    // sentinel's provenance audit chain. Fires only for mcp__* tools
    // that emit a structured `provenance_audit` field; silently
    // skips otherwise. Observational (always allows).
    if let Some(prov) = provenance_store {
        let audit_output = hooks::audit_extract::process(input, prov);
        output.merge(&audit_output);
    }

    // Bug task gate — scan tool output for bug signals (cargo test
    // FAILED, error[Exxxx], panicked at) and record pending-bug
    // state. Also clears state when a TaskCreate references the bug.
    let bug_gate_post = hooks::bug_task_gate::process_posttool(input, ctx);
    output.merge(&bug_gate_post);

    // Skill invocation gate — clear pending-skill state when the
    // detected skill is finally invoked (Skill tool with matching
    // name) or its SKILL.md is read.
    let skill_gate_post = hooks::skill_invocation_gate::process_posttool(input, ctx);
    output.merge(&skill_gate_post);

    // MCP health — detect MCP server failures and log to errors.jsonl
    let mcp_output = time_and_record(ctx.fs, &mk_ctx("mcp_health"), || {
        hooks::mcp_health::process(input, ctx)
    });
    output.merge(&mcp_output);

    // Todo interceptor — persist rich todos from TodoWrite calls
    let todo_output = time_and_record(ctx.fs, &mk_ctx("todo_interceptor"), || {
        hooks::todo_interceptor::process(input, ctx)
    });
    output.merge(&todo_output);

    // Activity tracker — log every tool call to activity-log.jsonl
    let activity_output = hooks::activity_tracker::process_post_tool(input, ctx);
    output.merge(&activity_output);

    // Project-scope tracker — publish the session's current working repo to
    // ~/.vulcan/hookdeck/session-<key>.project so the hookdeck channel router
    // scopes incoming webhooks to the session actually working in that repo
    // (fixes home-launch sessions receiving no events + cross-session bleed).
    let project_scope_output = time_and_record(ctx.fs, &mk_ctx("project_scope_tracker"), || {
        hooks::project_scope_tracker::process(input, ctx)
    });
    output.merge(&project_scope_output);

    // Browser test recorder — write state file on successful session release
    // (mcp__browserbase__release_session or mcp__cdp__close_instance)
    let browser_test_post_output = hooks::pre_push_browser_test::process_post_tool(input, ctx);
    output.merge(&browser_test_post_output);

    // Prompt-injection nudge — scan tool result for injection
    // shapes and inject an "untrusted output, ignore embedded
    // directives" warning when matched. Always allows; the
    // signal is via additionalContext.
    let nudge_output = hooks::prompt_injection_nudge::process(input, ctx);
    output.merge(&nudge_output);

    // AskUserQuestion task-resync nudge — when a decision made via
    // AskUserQuestion looks direction-changing, inject an advisory
    // reminder to re-sync the affected task subtree (child subjects AND
    // descriptions, plus deleting superseded chains) before the next
    // unit of work. Soft nudge only — always allows; the signal is via
    // additionalContext. Gated on the tool name, so all other tools are ignored.
    if matches!(input.tool_name.as_deref(), Some("AskUserQuestion")) {
        let resync_output = time_and_record(ctx.fs, &mk_ctx("ask_question_resync_nudge"), || {
            hooks::ask_question_resync_nudge::process(input, ctx)
        });
        output.merge(&resync_output);
    }

    // Plan organizer — inject plan file organization instructions (ExitPlanMode only)
    if matches!(input.tool_name.as_deref(), Some("ExitPlanMode")) {
        let plan_output = time_and_record(ctx.fs, &mk_ctx("plan_organizer"), || {
            hooks::plan_organizer::process(input, ctx)
        });
        output.merge(&plan_output);
    }

    // Account cascade — auto-switch all MCP servers after account change
    let cascade_output = hooks::account_cascade::process(input, ctx);
    output.merge(&cascade_output);

    // Build/deploy notify — push channel events for cargo build, test, git push
    let build_output = hooks::build_notify::process(input, ctx);
    output.merge(&build_output);

    // PR auto-monitor — inject CronCreate for PR monitoring (Bash only)
    if matches!(input.tool_name.as_deref(), Some("Bash")) {
        let pr_monitor_output = hooks::pr_auto_monitor::process(input);
        output.merge(&pr_monitor_output);

        // Build auto-monitor — suggest monitoring for background builds (Bash only)
        let build_monitor_output = hooks::build_auto_monitor::process(input);
        output.merge(&build_monitor_output);

        // Test evidence recorder — append a JSONL entry for any
        // Bash command matching a test/build pattern. Read by
        // `pre_commit_verification`; replaces transcript-parsing.
        let evidence_output = hooks::test_evidence_recorder::process(input, ctx);
        output.merge(&evidence_output);

        // Good citizen observer — scan Bash output for warnings,
        // dead-code, test failures, and open-task markers. Records
        // observations for the Stop reminder.
        let citizen_output = hooks::good_citizen_observer::process_post_tool(input, ctx);
        output.merge(&citizen_output);
    }

    // Linear lifecycle — inject CronCreate for issue lifecycle monitoring
    let linear_output = hooks::linear_lifecycle::process(input);
    output.merge(&linear_output);

    // Declarative auto-cron — reads config/autocron-defaults.toml (+ the operator
    // overlay), matches the current tool call against its rules, and injects a
    // CronCreate/loop suggestion. Replaces pr_auto_monitor's gh-pr-create cron
    // branch and linear_lifecycle's state-change cron branch (those literals were
    // migrated to rows). Not inside the Bash guard: each rule declares its own
    // `tool`, so this covers Bash + MCP (e.g. mcp__linear__update_issue) + TaskUpdate.
    let autocron_output = hooks::autocron::process(input);
    output.merge(&autocron_output);

    // Step judge (M1.4): run the adversarial AI judge against completed
    // step-tool evidence and produce the verdict consumed by
    // `submit_step_complete`.
    //
    // Non-sufficient verdicts are always surfaced here, and
    // `submit_step_complete` refuses to seal them through the proof engine.
    // The hook itself never blocks — PostToolUse is the wrong layer; the
    // proof chain is the enforcement substrate.
    {
        match sentinel_infrastructure::rig_judge::MultiModelJudge::from_env() {
            Ok(judge) => {
                let (sj_output, outcome) =
                    hooks::step_judge::process(input, state, step_configs, &judge).await;
                output.merge(&sj_output);

                use sentinel_application::hooks::step_judge::StepJudgeOutcome;
                if let StepJudgeOutcome::Judged {
                    skill,
                    phase_id,
                    step_id,
                    evidence,
                    verdict,
                    judge_model,
                    ..
                } = &outcome
                {
                    tracing::info!(
                        skill = %skill,
                        phase = %phase_id,
                        step = %step_id,
                        judge = %judge_model,
                        sufficient = verdict.sufficient,
                        confidence = verdict.confidence,
                        "step_judge verdict produced"
                    );

                    // step_anomaly (M1.9) — layered on step_judge: where the judge
                    // asks "did this step succeed?", step_anomaly asks "does this
                    // run *look like* a normal run of this step?" against the
                    // session's prior runs of the same skill. OBSERVATIONAL ONLY —
                    // PostToolUse never blocks (the proof chain is the enforcement
                    // substrate); anomalies surface as context for the model.
                    //
                    // History = this session's live chain for the skill PLUS every
                    // archived cross-session chain for the same skill (the full
                    // historical distribution). Merged into one ProofChain so the
                    // detectors see the widest baseline; empty on a cold machine
                    // (no archive, no live chain) => detectors stay quiet. Duration
                    // is still 0 here (see #58 — the step isn't sealed at this
                    // PostToolUse site, so no duration_ms exists yet).
                    let mut history =
                        sentinel_domain::proof::ProofChain::new(skill.clone(), String::new());
                    if let Some(live) = state.proof_chain(skill) {
                        history.entries.extend(live.entries.iter().cloned());
                    }
                    let home = sentinel_infrastructure::paths::home_root_or_fatal();
                    for archived in sentinel_application::proof_archive::read_chains_for_skill(
                        ctx.fs, &home, skill,
                    ) {
                        history.entries.extend(archived.entries);
                    }
                    let detectors = hooks::step_anomaly::default_detectors();
                    let anomaly_report = hooks::step_anomaly::run_detectors(
                        &detectors, skill, phase_id, step_id, evidence, 0, &history,
                    );
                    if !anomaly_report.is_clean() {
                        let lines: Vec<String> = anomaly_report
                            .anomalies
                            .iter()
                            .map(|a| {
                                format!(
                                    "  • [{:?}, σ={:.1}] {}",
                                    a.dimension, a.severity, a.reasoning
                                )
                            })
                            .collect();
                        tracing::info!(
                            skill = %skill, step = %step_id,
                            count = anomaly_report.anomalies.len(),
                            "step_anomaly detected behavioral anomalies"
                        );
                        output.merge(&sentinel_domain::events::HookOutput::inject_context(
                            sentinel_domain::events::HookEvent::PostToolUse,
                            format!(
                                "🔬 [Anomaly] Step '{step_id}' of '{skill}/{phase_id}' \
                             looks unusual vs prior runs this session:\n{}",
                                lines.join("\n")
                            ),
                        ));
                    }
                    // Surface a non-sufficient verdict to the model so it
                    // knows the step did not pass.
                    if !verdict.sufficient {
                        let warn_ctx = format!(
                            "🟠 [Judge:enforce] Step '{step_id}' of '{skill}/{phase_id}' \
                         judged INSUFFICIENT (confidence {:.2}): {} — submit_step_complete \
                         will refuse to seal this step.",
                            verdict.confidence, verdict.reasoning,
                        );
                        output.merge(&sentinel_domain::events::HookOutput::inject_context(
                            sentinel_domain::events::HookEvent::PostToolUse,
                            warn_ctx,
                        ));
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "step_judge unavailable: no AI judge provider configured; set OPENROUTER_API_KEY"
                );
                if input.tool_name.as_deref().is_some_and(|tool| {
                    tool.starts_with("mcp__skills__") && tool.contains("__step_")
                }) {
                    output.merge(&sentinel_domain::events::HookOutput::inject_context(
                        sentinel_domain::events::HookEvent::PostToolUse,
                        format!(
                            "[Sentinel-Authority] step_judge: no judge provider is configured; \
                             submit_step_complete will not seal this step. Set OPENROUTER_API_KEY. \
                             Error: {err:#}"
                        ),
                    ));
                }
            }
        }
    }

    output
}

/// Handle `Stop`: two-phase state-detection hooks, memory pipeline,
/// proof chain archive.
fn handle_stop(
    input: &sentinel_domain::events::HookInput,
    ctx: &hooks::HookContext<'_>,
    state: &sentinel_domain::state::SessionState,
) -> HookOutput {
    let mut output = HookOutput::allow();

    let cwd_for_metrics = input.cwd.as_deref().unwrap_or(".");
    let repo_root_for_metrics = ctx.git.repo_root(cwd_for_metrics);
    let mk_ctx = |hook: &'static str| InvocationContext {
        event: "Stop",
        hook,
        tool: None,
        session_id: input.session_id.as_deref(),
        repo_root: repo_root_for_metrics.as_deref(),
    };

    // Execution log — capture [RUN]/[STEP]/[PHASE] markers from transcript
    let exec_output = time_and_record(ctx.fs, &mk_ctx("execution_log"), || {
        hooks::execution_log::process(input, ctx)
    });
    output.merge(&exec_output);

    // Skill telemetry — aggregate skill usage metrics
    let telemetry_output = time_and_record(ctx.fs, &mk_ctx("skill_telemetry"), || {
        hooks::skill_telemetry::process(input, ctx)
    });
    output.merge(&telemetry_output);

    // --- Two-phase hooks (detect state, write for UserPromptSubmit to read) ---

    // Context monitor — capture context window usage zone
    let ctx_output = time_and_record(ctx.fs, &mk_ctx("context_monitor"), || {
        hooks::context_monitor::process_stop(input, ctx)
    });
    output.merge(&ctx_output);

    // Commit hygiene — detect uncommitted changes
    let hygiene_output = time_and_record(ctx.fs, &mk_ctx("commit_hygiene"), || {
        hooks::commit_hygiene::process_stop(input, ctx)
    });
    output.merge(&hygiene_output);

    // Doc cleanup — scan for junk docs
    let doc_output = time_and_record(ctx.fs, &mk_ctx("doc_cleanup"), || {
        hooks::doc_cleanup::process_stop(input, ctx)
    });
    output.merge(&doc_output);

    // Doc drift — detect stale README/CLAUDE.md/CHANGELOG
    let drift_output = time_and_record(ctx.fs, &mk_ctx("doc_drift"), || {
        hooks::doc_drift::process_stop(input, ctx)
    });
    output.merge(&drift_output);

    // Hygiene reminders — detect unpushed commits, stale worktrees, changelog gaps
    let reminders_output = time_and_record(ctx.fs, &mk_ctx("hygiene_reminders"), || {
        hooks::hygiene_reminders::process_stop(input, ctx)
    });
    output.merge(&reminders_output);

    // Verification gate — detect unverified completion claims
    let verify_output = time_and_record(ctx.fs, &mk_ctx("verification_gate"), || {
        hooks::verification_gate::process_stop(input, ctx)
    });
    output.merge(&verify_output);

    // Task coverage check — warn if uncommitted changes but no active task
    let coverage_output = time_and_record(ctx.fs, &mk_ctx("task_coverage_check"), || {
        hooks::task_coverage_check::process(input, ctx)
    });
    output.merge(&coverage_output);

    // Claim reality check — sweep newly-completed (✅) tasks and flag any whose
    // commit/PR/merge claim doesn't hold against git/gh (false-done detection).
    let reality_output = time_and_record(ctx.fs, &mk_ctx("claim_reality_check"), || {
        hooks::claim_reality_check::process(input, ctx)
    });
    output.merge(&reality_output);

    // Self-annealing — when a phase gate fails repeatedly (the forensic
    // failed_submissions record), surface a [Self-Annealing] remediation; when
    // operator-armed (SENTINEL_ALLOW_SELF_ANNEAL=1), open a PR hardening the
    // skill. Reads SessionState for the failure counts.
    let anneal_output = time_and_record(ctx.fs, &mk_ctx("self_annealing"), || {
        hooks::self_annealing::process(input, ctx, state)
    });
    output.merge(&anneal_output);

    // Good citizen observer — surface unaddressed warnings/findings
    // observed during the turn, prompt agent to file TaskCreate.
    let citizen_output = time_and_record(ctx.fs, &mk_ctx("good_citizen_observer"), || {
        hooks::good_citizen_observer::process_stop(input, ctx)
    });
    output.merge(&citizen_output);

    // Activity tracker — build session summary from activity log
    let activity_stop_output = time_and_record(ctx.fs, &mk_ctx("activity_tracker"), || {
        hooks::activity_tracker::process_stop(input, ctx)
    });
    output.merge(&activity_stop_output);

    // Task persist — final snapshot catches any TaskUpdate calls mid-turn
    let task_persist_output = time_and_record(ctx.fs, &mk_ctx("task_persist"), || {
        hooks::task_persist::process(input, ctx)
    });
    output.merge(&task_persist_output);

    // Memory extract — periodic session transcript re-indexing.
    // (Flat-.md capture path is removed; turn-capture below replaces it.)
    let memory_extract_output = time_and_record(ctx.fs, &mk_ctx("memory_extract"), || {
        hooks::memory_extract::process(input, ctx)
    });
    output.merge(&memory_extract_output);

    // Memory turn-capture — LLM extracts atoms from this turn and
    // routes them through the dual-judge memory_capture gate.
    let memory_turn_output = time_and_record(ctx.fs, &mk_ctx("memory_turn_capture"), || {
        hooks::memory_turn_capture::process(input, ctx)
    });
    output.merge(&memory_turn_output);

    // Memory feedback — boost used memories, flag corrections
    let memory_feedback_output = time_and_record(ctx.fs, &mk_ctx("memory_feedback"), || {
        hooks::memory_feedback::process(input, ctx)
    });
    output.merge(&memory_feedback_output);

    // Cross-session proof chain archive (#39). Required so query_proof_corpus
    // can answer across sessions, not just live state.
    let home = sentinel_infrastructure::paths::home_root_or_fatal();
    if let Err(error) = sentinel_application::proof_archive::archive_chains(state, ctx.fs, &home) {
        output.merge(&fail_closed_output_for_event(
            HookEvent::Stop,
            format!("proof chain archive failed during Stop: {error}"),
        ));
    }

    output
}

fn parse_hook_event(event: &str) -> Result<HookEvent> {
    if let Some(e) = HookEvent::from_arg(event) {
        Ok(e)
    } else {
        eprintln!(
            "[sentinel] ERROR: Unknown hook event type '{event}'. \
             Valid events: SessionStart, SessionEnd, UserPromptSubmit, PreToolUse, \
             PostToolUse, PostToolUseFailure, Stop, StopFailure, PreCompact, PostCompact, \
             Setup, SubagentStart, SubagentStop, TeammateIdle, TaskCreated, TaskCompleted, \
             PermissionDenied, CwdChanged"
        );
        anyhow::bail!("Unknown hook event type: '{event}'");
    }
}

const fn should_attach_project_context(hook_event: HookEvent) -> bool {
    matches!(
        hook_event,
        HookEvent::SessionStart
            | HookEvent::UserPromptSubmit
            | HookEvent::SubagentStart
            | HookEvent::Setup
    )
}

fn fail_closed_output_for_event(hook_event: HookEvent, message: impl Into<String>) -> HookOutput {
    let message = message.into();
    if hook_event == HookEvent::PreToolUse {
        return HookOutput::deny(message).into_pretool_output();
    }

    let message = format!("[Sentinel-Authority] {message}");
    let mut output = HookOutput {
        system_message: Some(message.clone()),
        ..HookOutput::allow()
    };
    if should_attach_project_context(hook_event) {
        output.hook_specific_output = Some(HookSpecificOutput {
            hook_event_name: hook_event.to_string(),
            additional_context: Some(message),
            ..HookSpecificOutput::default()
        });
    }
    output
}

fn write_fail_closed_response(hook_event: HookEvent, message: impl Into<String>) -> Result<()> {
    sentinel_infrastructure::stdout::write_hook_output(&fail_closed_output_for_event(
        hook_event, message,
    ))
}

fn write_safe_allow_response() -> Result<()> {
    sentinel_infrastructure::stdout::write_hook_output(&HookOutput::allow())
}

/// Validate that the caller is plausibly Claude Code.
///
/// Two checks:
/// 1. **Stdin is piped** (not a terminal) — hooks are always invoked via pipe
///    from Claude Code's hook runner, never interactively. Hard fail unless
///    `SENTINEL_ALLOW_TERMINAL=1` is set.
/// 2. **`CLAUDE_CODE` env marker** — Claude Code sets `CLAUDE_CODE_ENTRYPOINT`
///    when spawning hook subprocesses. Its absence is suspicious.
///
/// If validation fails, a `caller_rejected` event is logged to the security
/// audit log and the function returns `Err`, causing `run()` to output `{}`
/// (safe allow) and exit early.
///
/// This is defense-in-depth, not a security boundary. A determined attacker can
/// still set the env var and pipe crafted JSON, but they must:
///   1. Know the sentinel CLI exists and its arguments
///   2. Set `CLAUDE_CODE_ENTRYPOINT` in their environment
///   3. Construct valid `HookInput` JSON with a real `session_id`
///   4. Pipe it correctly (not type interactively)
fn validate_caller() -> Result<()> {
    // Escape hatch for debugging / manual testing
    if std::env::var("SENTINEL_ALLOW_TERMINAL").as_deref() == Ok("1") {
        return Ok(());
    }

    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        eprintln!(
            "[sentinel] BLOCKED: Hook invoked from interactive terminal. \
             Hooks must be called by Claude Code via pipe. \
             Set SENTINEL_ALLOW_TERMINAL=1 to override for debugging."
        );
        let _ = sentinel_infrastructure::security_log::log_security_event(
            "caller_rejected",
            "unknown",
            "Hook invoked from interactive terminal (stdin is TTY)",
        );
        anyhow::bail!("Caller validation failed: stdin is a terminal");
    }

    // Check for Claude Code environment marker.
    // Claude Code sets CLAUDE_CODE_ENTRYPOINT (e.g. "cli", "sdk").
    // Absence is not definitive proof of abuse (could be an older version),
    // so keep it as a debug-only signal instead of stderr noise. Claude treats
    // stderr from hooks as a non-blocking hook error banner.
    if std::env::var("CLAUDE_CODE_ENTRYPOINT").is_err() {
        debug!("CLAUDE_CODE_ENTRYPOINT not set for hook invocation");
    }

    // Optional parent-process attestation.
    //
    // This was introduced as defense-in-depth, but the Windows `wmic`/`tasklist`
    // probe can outlive the hook's JSON response and keep the process alive long
    // enough for Claude Code to appear wedged after pressing Enter. Keep the
    // simpler stdin/env checks on by default and make the heavier attestation
    // opt-in for debugging or forensics.
    #[cfg(windows)]
    if std::env::var("SENTINEL_ENABLE_PARENT_ATTESTATION").as_deref() == Ok("1") {
        if let Some(parent) = get_parent_process_name() {
            #[cfg(windows)]
            let valid_parents = [
                "node.exe",
                "bun.exe",
                "claude.exe",
                "cmd.exe",
                "powershell.exe",
                "pwsh.exe",
                "bash.exe",
                "sentinel-engine.exe",
                "sentinel.exe",
            ];
            #[cfg(not(windows))]
            let valid_parents = [
                "node",
                "bun",
                "claude",
                "bash",
                "zsh",
                "sh",
                "sentinel-engine",
                "sentinel",
            ];
            if !valid_parents.iter().any(|v| parent.contains(v)) {
                eprintln!(
                    "[sentinel] WARNING: Parent process '{parent}' is not a known Claude Code runtime."
                );
                let _ = sentinel_infrastructure::security_log::log_security_event(
                    "caller_rejected",
                    "unknown",
                    &format!("Parent process '{parent}' is not a known Claude Code runtime"),
                );
            }
        } else {
            tracing::warn!(
                "step_judge unavailable: no AI judge provider configured; set OPENROUTER_API_KEY"
            );
        }
    }

    Ok(())
}

/// Get the parent process name on Windows using tasklist + wmic.
///
/// Strategy: Use `wmic process where ProcessId=<PID> get ParentProcessId` to find
/// the parent PID, then `tasklist /FI "PID eq <PPID>"` to get its name.
/// Both are native Windows tools that start in ~30ms (vs `PowerShell` ~2s).
///
/// Returns None if the check fails for any reason (fail-open for resilience).
#[cfg(windows)]
fn get_parent_process_name() -> Option<String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;

    let pid = std::process::id();

    // Step 1: Get parent PID via wmic (~30ms)
    let output = std::process::Command::new("wmic")
        .args([
            "process",
            "where",
            &format!("ProcessId={pid}"),
            "get",
            "ParentProcessId",
            "/VALUE",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parent_pid: u32 = stdout
        .lines()
        .find(|l| l.starts_with("ParentProcessId="))
        .and_then(|l| l.strip_prefix("ParentProcessId="))
        .and_then(|s| s.trim().parse().ok())?;

    // Step 2: Get parent process name via tasklist (~30ms)
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {parent_pid}"), "/FO", "CSV", "/NH"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    // CSV format: "Image Name","PID","Session Name","Session#","Mem Usage"
    let name = stdout
        .lines()
        .next()?
        .split(',')
        .next()?
        .trim_matches('"')
        .to_lowercase();

    if name.is_empty() || name.contains("no tasks") {
        None
    } else {
        Some(name)
    }
}

// check_version_drift removed — Claude Code uses bun (not npm).
// The synchronous `npm view` call added 4-20s to every cold SessionStart.

/// Extract skill name from router context like "[Skill Router] Detected skill: linear. MANDATORY..."
fn extract_skill_name(context: &str) -> Option<String> {
    let prefix = "Detected skill: ";
    let start = context.find(prefix)? + prefix.len();
    let rest = &context[start..];
    let end = rest.find('.')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_domain::judge::JudgeModel;
    use sentinel_domain::workflow::{WorkflowPhase, WorkflowState};
    use std::process::Stdio;
    use std::sync::MutexGuard;

    use crate::phase_graph_projection::phase_graph_db_path;

    fn env_lock() -> MutexGuard<'static, ()> {
        crate::test_env::lock()
    }

    fn sentinel_engine_bin() -> Option<std::path::PathBuf> {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?;
        let exe = if cfg!(windows) {
            "sentinel-engine.exe"
        } else {
            "sentinel-engine"
        };
        let debug = repo_root.join("target").join("debug").join(exe);
        if debug.exists() {
            return Some(debug);
        }
        let release = repo_root.join("target").join("release").join(exe);
        release.exists().then_some(release)
    }

    fn write_authoritative_workflow_config(home: &std::path::Path) {
        let config_dir = home.join(".claude").join("sentinel").join("config");
        std::fs::create_dir_all(&config_dir).expect("create sentinel config dir");
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

    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let original = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    struct NoopEnv;

    /// `gh` that always fails, so pr_merge_gate's fetch lands in the
    /// unknown-state branch deterministically (no host `gh`/network involved).
    struct UnfetchableGh;

    impl sentinel_domain::ports::ProcessPort for UnfetchableGh {
        fn run(
            &self,
            _: &str,
            _: &[&str],
            _: Option<&str>,
        ) -> Result<sentinel_domain::ports::ProcessOutput, sentinel_domain::port_errors::ProcessError>
        {
            Err(sentinel_domain::port_errors::ProcessError::Io(
                "gh not available".to_string(),
            ))
        }
        fn spawn_detached(
            &self,
            _: &str,
            _: &[&str],
        ) -> Result<(), sentinel_domain::port_errors::ProcessError> {
            Ok(())
        }
    }

    impl sentinel_domain::ports::EnvPort for NoopEnv {
        fn var(&self, _key: &str) -> Option<String> {
            None
        }

        fn var_os(&self, _key: &str) -> Option<std::ffi::OsString> {
            None
        }
    }

    struct NoopMemoryMcp;

    #[async_trait::async_trait]
    impl sentinel_domain::ports::MemoryMcpPort for NoopMemoryMcp {
        async fn call_tool(
            &self,
            _name: &str,
            _arguments: serde_json::Map<String, serde_json::Value>,
        ) -> std::result::Result<serde_json::Value, sentinel_domain::port_errors::MemoryMcpError>
        {
            Ok(serde_json::json!({}))
        }
    }

    #[derive(Clone)]
    struct StaticLinearLookup {
        result: std::result::Result<serde_json::Value, sentinel_domain::ports::LinearLookupError>,
    }

    impl sentinel_domain::ports::LinearLookupPort for StaticLinearLookup {
        fn fetch_issue(
            &self,
            _identifier_or_id: &str,
        ) -> std::result::Result<serde_json::Value, sentinel_domain::ports::LinearLookupError>
        {
            self.result.clone()
        }
    }

    struct FrontendDiffGit;

    impl sentinel_domain::ports::GitStatusPort for FrontendDiffGit {
        fn has_uncommitted_changes(
            &self,
            _: &str,
        ) -> std::result::Result<bool, sentinel_domain::port_errors::GitError> {
            Ok(false)
        }

        fn changed_files(
            &self,
            _: &str,
        ) -> std::result::Result<Vec<String>, sentinel_domain::port_errors::GitError> {
            Ok(Vec::new())
        }

        fn current_branch(
            &self,
            _: &str,
        ) -> std::result::Result<String, sentinel_domain::port_errors::GitError> {
            Ok("main".to_string())
        }

        fn is_worktree(&self, _: &str) -> bool {
            false
        }

        fn has_unpushed_commits(
            &self,
            _: &str,
        ) -> std::result::Result<bool, sentinel_domain::port_errors::GitError> {
            Ok(true)
        }

        fn repo_root(&self, _: &str) -> Option<String> {
            None
        }

        fn list_worktree_names(&self, _: &str) -> Vec<String> {
            Vec::new()
        }

        fn merge_base(&self, _: &str, _: &str) -> Option<String> {
            Some("base".to_string())
        }

        fn rev_list_count(&self, _: &str, _: &str) -> Option<u32> {
            Some(1)
        }

        fn rev_list_count_range(&self, _: &str, _: &str) -> Option<u32> {
            Some(1)
        }

        fn diff_names(&self, _: &str, _: &str) -> Option<Vec<String>> {
            Some(vec!["src/App.tsx".to_string()])
        }

        fn merged_local_branches(&self, _: &str, _: &str) -> Vec<String> {
            Vec::new()
        }

        fn merged_remote_branches(&self, _: &str, _: &str) -> Vec<String> {
            Vec::new()
        }

        fn head_sha(&self, _: &str) -> Option<String> {
            Some("head".to_string())
        }
    }

    struct TasksRepoGit {
        root: std::path::PathBuf,
    }

    impl sentinel_domain::ports::GitStatusPort for TasksRepoGit {
        fn has_uncommitted_changes(
            &self,
            _: &str,
        ) -> std::result::Result<bool, sentinel_domain::port_errors::GitError> {
            Ok(false)
        }

        fn changed_files(
            &self,
            _: &str,
        ) -> std::result::Result<Vec<String>, sentinel_domain::port_errors::GitError> {
            Ok(Vec::new())
        }

        fn current_branch(
            &self,
            _: &str,
        ) -> std::result::Result<String, sentinel_domain::port_errors::GitError> {
            Ok("main".to_string())
        }

        fn is_worktree(&self, _: &str) -> bool {
            false
        }

        fn has_unpushed_commits(
            &self,
            _: &str,
        ) -> std::result::Result<bool, sentinel_domain::port_errors::GitError> {
            Ok(false)
        }

        fn repo_root(&self, _: &str) -> Option<String> {
            Some(self.root.display().to_string())
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

    fn workflow(skill: &str) -> SkillWorkflow {
        SkillWorkflow {
            skill: skill.to_string(),
            phases: vec![
                WorkflowPhase {
                    id: "claim".to_string(),
                    file: "claim.md".to_string(),
                    required: true,
                    judge: JudgeModel::Sonnet,
                    description: "claim".to_string(),
                    required_dyad: None,
                },
                WorkflowPhase {
                    id: "fetch".to_string(),
                    file: "fetch.md".to_string(),
                    required: true,
                    judge: JudgeModel::Sonnet,
                    description: "fetch".to_string(),
                    required_dyad: None,
                },
            ],
            blocked_tool_prefixes: Vec::new(),
            blocked_bash_patterns: Vec::new(),
            bash_allowlist: Vec::new(),
        }
    }

    #[test]
    fn configured_workflow_missing_step_config_is_hard_error() {
        let workflows = HashMap::from([("linear".to_string(), workflow("linear"))]);
        let tmp = tempfile::tempdir().expect("tmpdir");

        let err = load_configured_skill_steps(tmp.path(), &workflows, "linear")
            .expect_err("configured workflow must require step config");
        let message = err.to_string();

        assert!(message.contains("configured LangGraph workflow 'linear'"));
        assert!(message.contains("missing required step config"));
        // Platform-agnostic: Path::display() uses `\` on Windows, `/` on Unix.
        assert!(message.contains("steps"));
        assert!(message.contains("linear.toml"));
    }

    #[test]
    fn unconfigured_skill_does_not_require_step_config() {
        let workflows = HashMap::from([("linear".to_string(), workflow("linear"))]);
        let tmp = tempfile::tempdir().expect("tmpdir");

        let steps =
            load_configured_skill_steps(tmp.path(), &workflows, "unconfigured").expect("load");

        assert!(steps.is_none());
    }

    fn inject_old_workflows_field(
        state: &mut SessionState,
        skill: &str,
        workflow_state: WorkflowState,
    ) {
        let mut value = serde_json::to_value(&*state).expect("session state serializes");
        let old_workflows = serde_json::json!({
            skill: serde_json::to_value(workflow_state).expect("workflow state serializes")
        });
        value
            .as_object_mut()
            .expect("session state is an object")
            .insert("workflows".to_string(), old_workflows);
        *state = serde_json::from_value(value).expect("session state deserializes");
    }

    #[test]
    fn test_extract_skill_name() {
        let ctx =
            "[Skill Router] Detected skill: linear. MANDATORY: You MUST call Skill(skill: \"linear\")...";
        assert_eq!(extract_skill_name(ctx), Some("linear".to_string()));
    }

    #[test]
    fn test_extract_skill_name_none() {
        assert_eq!(extract_skill_name("no skill here"), None);
    }

    #[test]
    fn test_no_skill_matched_clears_active_skill() {
        // Simulates the scenario: skill router detected "linear" on message 1,
        // then "No skill matched" on message 2. active_skill should be cleared.
        let no_match_context = "[Skill Router] No skill matched — general conversation mode.";
        assert_eq!(extract_skill_name(no_match_context), None);
        // The clearing happens in the caller (run()) when extract_skill_name returns
        // None AND the context contains "No skill matched"
        assert!(no_match_context.contains("No skill matched"));
    }

    #[test]
    fn test_project_context_only_attaches_to_prompt_scoped_events() {
        assert!(should_attach_project_context(HookEvent::SessionStart));
        assert!(should_attach_project_context(HookEvent::UserPromptSubmit));
        assert!(should_attach_project_context(HookEvent::SubagentStart));
        assert!(should_attach_project_context(HookEvent::Setup));
        assert!(!should_attach_project_context(HookEvent::PreToolUse));
        assert!(!should_attach_project_context(HookEvent::PostToolUse));
        assert!(!should_attach_project_context(
            HookEvent::PostToolUseFailure
        ));
        assert!(!should_attach_project_context(HookEvent::PostCompact));
    }

    #[test]
    fn commit_message_graph_authority_writes_langgraph_audit_for_malformed_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("commit-message-graph-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "git commit -m 'updated the thing'"
            })),
            ..Default::default()
        };

        let output = authorize_commit_message_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("commit-message.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"commit_message\""));
        assert!(graph_rows.contains("\"decision\":\"block-malformed\""));
    }

    #[test]
    fn pre_commit_verification_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let repo = tempfile::tempdir().expect("temp repo");
        std::fs::write(repo.path().join("Cargo.toml"), "[package]\nname = \"x\"\n")
            .expect("build marker");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let state = SessionState::new("pre-commit-graph-test");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("pre-commit-graph-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "git commit -m 'untested change'"
            })),
            cwd: Some(repo.path().display().to_string()),
            ..Default::default()
        };

        let output = authorize_pre_commit_verification_with_graph(&input, &ctx, &state);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("pre-commit-verification.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"pre_commit_verification\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn pre_push_browser_test_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let work = tempfile::tempdir().expect("temp work");
        let repo = work.path().join("firefly-pro-crm");
        std::fs::create_dir_all(&repo).expect("repo dir");
        let projects = home
            .path()
            .join(".claude")
            .join("skills")
            .join("linear")
            .join("projects");
        std::fs::create_dir_all(&projects).expect("project config dir");
        std::fs::write(
            projects.join("firefly.md"),
            "name: firefly-pro\naliases: [\"firefly-pro-crm\"]\nbrowser_test_email: test@example.com\n",
        )
        .expect("browser project config");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = FrontendDiffGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("pre-push-browser-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "git push origin feature/ui"
            })),
            cwd: Some(repo.display().to_string()),
            ..Default::default()
        };

        let output = authorize_pre_push_browser_test_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("pre-push-browser-test.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"pre_push_browser_test\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn pre_push_browser_test_graph_identifier_requires_command_evidence() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("pre-push-browser-no-command-id".into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({})),
            ..Default::default()
        };
        let evaluation = hooks::pre_push_browser_test::PrePushBrowserEvaluation {
            tool: Some("Bash".into()),
            command: None,
            bash_tool: true,
            command_present: false,
            git_push: true,
            repo_browser_test_configured: true,
            frontend_changes: true,
            session_id_present: true,
            recent_browser_test: false,
            should_block: true,
            decision: hooks::pre_push_browser_test::PrePushBrowserDecision::Block,
        };

        let err = pre_push_browser_test_graph_identifier(&input, &evaluation)
            .expect_err("identifier must require command evidence");

        assert!(format!("{err:#}").contains("command evidence"));
    }

    #[test]
    fn agent_revocation_graph_authority_writes_langgraph_audit_for_deny() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let mut state = SessionState::new("agent-revocation-graph-test");
        state.revoke_agent("agent-x");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("agent-revocation-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "echo should-not-run"
            })),
            agent_id: Some("agent-x".into()),
            ..Default::default()
        };

        let output = authorize_agent_revocation_with_graph(&input, &state);

        assert_eq!(
            output
                .hook_specific_output
                .as_ref()
                .and_then(|h| h.permission_decision),
            Some(sentinel_domain::events::PermissionDecision::Deny)
        );
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("agent-revocation.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"agent_revocation\""));
        assert!(graph_rows.contains("\"decision\":\"deny\""));
    }

    #[test]
    fn step_gate_graph_authority_writes_langgraph_audit_for_missing_graph_workflow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let state = SessionState::new("step-gate-graph-test");
        let mut step_configs = HashMap::new();
        step_configs.insert(
            "linear".to_string(),
            SkillSteps {
                skill: "linear".to_string(),
                federation_version: "1".to_string(),
                phases: vec![sentinel_domain::workflow::PhaseSteps {
                    phase_id: "claim".to_string(),
                    steps: vec![sentinel_domain::workflow::WorkflowStep {
                        id: "1".to_string(),
                        description: "fetch ticket".to_string(),
                        blocker: false,
                        baseline_threshold: 0,
                        judge: None,
                        timeout_ms: None,
                        retry_policy: Default::default(),
                        circuit_breaker: Default::default(),
                        provides: Vec::new(),
                        requires: Vec::new(),
                        external: Vec::new(),
                        inaccessible: false,
                        deprecated: None,
                        r#override: None,
                        extra: serde_json::Value::Null,
                    }],
                }],
            },
        );
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("step-gate-{}", uuid::Uuid::new_v4())),
            tool_name: Some("mcp__skills__linear__step_1".into()),
            ..Default::default()
        };

        let output = authorize_step_gate_with_graph(&input, &state, &step_configs);

        assert_eq!(
            output
                .hook_specific_output
                .as_ref()
                .and_then(|h| h.permission_decision),
            Some(sentinel_domain::events::PermissionDecision::Deny)
        );
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("step-gate.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"step_gate\""));
        assert!(graph_rows.contains("\"decision\":\"deny-missing-graph-workflow\""));
    }

    #[test]
    fn ticket_quality_graph_authority_writes_langgraph_audit_for_deny() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("ticket-quality-{}", uuid::Uuid::new_v4())),
            tool_name: Some("mcp__linear__create_issue".into()),
            tool_input: Some(serde_json::json!({
                "title": "Thin ticket",
                "priority": 0,
                "description": "fix it"
            })),
            ..Default::default()
        };

        let output = authorize_ticket_quality_with_graph(&input);

        assert_eq!(
            output
                .hook_specific_output
                .as_ref()
                .and_then(|h| h.permission_decision),
            Some(sentinel_domain::events::PermissionDecision::Deny)
        );
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("ticket-quality.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"ticket_quality\""));
        assert!(graph_rows.contains("\"decision\":\"deny-missing-fields\""));
    }

    #[test]
    fn tasks_md_guard_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let repo = tempfile::tempdir().expect("temp repo");
        let tasks_path = repo.path().join("tasks.md");
        std::fs::write(
            &tasks_path,
            format!(
                "# Tasks\n\n{}\nauto item\n{}\n",
                hooks::task_persist::MARKER_START,
                hooks::task_persist::MARKER_END
            ),
        )
        .expect("tasks.md");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = TasksRepoGit {
            root: repo.path().to_path_buf(),
        };
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("tasks-md-guard-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Edit".into()),
            file_path: Some(tasks_path.display().to_string()),
            tool_input: Some(serde_json::json!({
                "file_path": tasks_path.display().to_string(),
                "old_string": "auto item",
                "new_string": "manual edit"
            })),
            ..Default::default()
        };

        let output = authorize_tasks_md_guard_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("tasks-md-guard.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"tasks_md_guard\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn tasks_md_guard_graph_authority_audits_guarded_tool_without_file_path() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let repo = tempfile::tempdir().expect("temp repo");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = TasksRepoGit {
            root: repo.path().to_path_buf(),
        };
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!(
                "tasks-md-guard-missing-path-{}",
                uuid::Uuid::new_v4()
            )),
            tool_name: Some("Edit".into()),
            tool_input: Some(serde_json::json!({
                "old_string": "auto item",
                "new_string": "manual edit"
            })),
            ..Default::default()
        };

        let output = authorize_tasks_md_guard_with_graph(&input, &ctx);

        assert_ne!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("tasks-md-guard.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"tasks_md_guard\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
        assert!(graph_rows.contains("\"file_path_present\":false"));
        assert!(!graph_rows.contains("file-path-sha256"));
    }

    #[test]
    fn task_decomposition_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let session_id = format!("task-decomposition-{}", uuid::Uuid::new_v4());
        // Seed a PRESENT-BUT-EMPTY session task dir: post-fix the gate fails OPEN
        // on a MISSING dir (unconfirmable) and only blocks on a present-but-empty
        // one (genuinely never decomposed). This test asserts a BLOCK + block
        // audit, so it must present the empty-dir case.
        std::fs::create_dir_all(home.path().join(".claude").join("tasks").join(&session_id))
            .expect("seed empty session task dir");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(session_id),
            tool_name: Some("Edit".into()),
            ..Default::default()
        };

        let output = authorize_task_decomposition_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("task-decomposition.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"task_decomposition\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn task_decomposition_graph_identifier_records_absent_bash_command_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("task-decomposition-absent-command-id".into()),
            tool_name: Some("Edit".into()),
            ..Default::default()
        };
        let evaluation = hooks::task_decomposition_gate::TaskDecompositionEvaluation {
            tool: Some("Edit".into()),
            session_id: Some("task-decomposition-absent-command-id".into()),
            bash_command: None,
            allowed_tool: false,
            bash_tool: false,
            bash_command_present: false,
            mutating_tool: true,
            task_state_readable: true,
            task_list_confirmed: false,
            unreadable_task_state: false,
            should_block: true,
            decision: hooks::task_decomposition_gate::TaskDecompositionDecision::Block,
        };

        let identifier =
            task_decomposition_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("bash-command-present-false"));
        assert!(!identifier.contains("no-bash-command"));
        assert!(!identifier.contains("bash-command-sha256"));
    }

    #[test]
    fn graph_identifiers_do_not_use_synthetic_missing_evidence_tokens() {
        let source = include_str!("hook_cmd.rs");
        let fallback_patterns = [
            ["unwrap_or_else(|| ", "\"missing"].concat(),
            ["unwrap_or(", "\"missing"].concat(),
            ["unwrap_or_else(|| ", "\"no-"].concat(),
            ["unwrap_or(", "\"no-"].concat(),
            ["unknown", "-operation"].concat(),
        ];

        for pattern in fallback_patterns {
            assert!(
                !source.contains(&pattern),
                "graph identifiers must model absent evidence with presence flags, not {pattern}"
            );
        }
    }

    #[test]
    fn linear_pm_graph_authority_writes_langgraph_audit_for_oversized_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let lookup = StaticLinearLookup {
            result: Ok(serde_json::json!({
                "identifier": "FPCRM-606",
                "estimate": 8,
                "state": { "name": "Todo", "type": "backlog" },
                "projectHasMilestones": false
            })),
        };
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: Some(&lookup),
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("linear-pm-{}", uuid::Uuid::new_v4())),
            tool_name: Some("mcp__linear__update_issue".into()),
            tool_input: Some(serde_json::json!({
                "identifier": "FPCRM-606",
                "started": true
            })),
            ..Default::default()
        };

        let output = authorize_linear_pm_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("linear-pm.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"linear_pm\""));
        assert!(graph_rows.contains("\"decision\":\"block-oversized-ticket\""));
    }

    #[test]
    fn production_action_notice_graph_authority_writes_langgraph_audit_for_notice() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let mut state = SessionState::new("production-action-notice-test");
        state.arm_production_override(None);
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("production-action-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "wrangler deploy --env production"
            })),
            ..Default::default()
        };

        let output = authorize_production_action_notice_with_graph(&input, &state);

        assert!(output.system_message.is_some());
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("production-action-notice.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"production_action_notice\""));
        assert!(graph_rows.contains("\"decision\":\"notice\""));
    }

    #[test]
    fn production_action_notice_graph_identifier_records_absent_haystack_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("production-action-absent-haystack-id".into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({})),
            ..Default::default()
        };
        let mut evaluation = hooks::production_action_notice::evaluate(&input, true);
        evaluation.haystack_present = false;
        evaluation.haystack.clear();
        evaluation.mentions_prod = false;
        evaluation.should_notice = false;
        evaluation.decision =
            hooks::production_action_notice::ProductionActionNoticeDecision::AllowSilent;

        let identifier =
            production_action_notice_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("haystack-present-false"));
        assert!(!identifier.contains("missing-haystack"));
        assert!(!identifier.contains("haystack-sha256"));
    }

    #[test]
    fn tasks_md_guard_graph_identifier_records_absent_file_path_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("tasks-md-absent-file-id".into()),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({})),
            ..Default::default()
        };
        let evaluation = hooks::tasks_md_guard::TasksMdGuardEvaluation {
            tool: Some("Bash".into()),
            file_path: None,
            guarded_tool: false,
            edit_tool: false,
            write_tool: false,
            file_path_present: false,
            project_tasks_md: false,
            existing_file_present: false,
            old_string_present: false,
            content_present: false,
            edit_overlaps_auto_block: false,
            write_changes_auto_block: false,
            should_block: false,
            decision: hooks::tasks_md_guard::TasksMdGuardDecision::Allow,
        };

        let identifier = tasks_md_guard_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("file-path-present-false"));
        assert!(!identifier.contains("missing-file"));
        assert!(!identifier.contains("file-path-sha256"));
    }

    #[test]
    fn tasks_md_guard_graph_identifier_rejects_present_file_path_without_evidence() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("tasks-md-missing-file-evidence-id".into()),
            tool_name: Some("Edit".into()),
            tool_input: Some(serde_json::json!({})),
            ..Default::default()
        };
        let evaluation = hooks::tasks_md_guard::TasksMdGuardEvaluation {
            tool: Some("Edit".into()),
            file_path: None,
            guarded_tool: true,
            edit_tool: true,
            write_tool: false,
            file_path_present: true,
            project_tasks_md: true,
            existing_file_present: true,
            old_string_present: true,
            content_present: false,
            edit_overlaps_auto_block: false,
            write_changes_auto_block: false,
            should_block: false,
            decision: hooks::tasks_md_guard::TasksMdGuardDecision::Allow,
        };

        let err = tasks_md_guard_graph_identifier(&input, &evaluation).unwrap_err();

        assert!(err.to_string().contains("concrete file path evidence"));
    }

    #[test]
    fn git_hygiene_graph_authority_writes_langgraph_audit_for_protected_branch_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let repo = tempfile::tempdir().expect("temp repo");
        let src_dir = repo.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("src dir");
        let target = src_dir.join("lib.rs");
        std::fs::write(&target, "pub fn existing() {}\n").expect("target source");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = TasksRepoGit {
            root: repo.path().to_path_buf(),
        };
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let state = SessionState::new("git-hygiene-graph-test");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("git-hygiene-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Edit".into()),
            cwd: Some(repo.path().display().to_string()),
            file_path: Some(target.display().to_string()),
            tool_input: Some(serde_json::json!({
                "file_path": target.display().to_string(),
                "old_string": "pub fn existing() {}",
                "new_string": "pub fn existing() { }"
            })),
            ..Default::default()
        };

        let output = authorize_git_hygiene_with_graph(&input, &git, &fs, &state);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("git-hygiene.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"git_hygiene\""));
        assert!(graph_rows.contains("\"decision\":\"deny-protected-branch\""));
    }

    #[test]
    fn git_hygiene_graph_identifier_records_absent_file_path_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("git-hygiene-absent-file-id".into()),
            tool_name: Some("Edit".into()),
            cwd: Some("/repo".into()),
            ..Default::default()
        };
        let evaluation = hooks::git_hygiene::GitHygieneEvaluation {
            tool: Some("Edit".into()),
            cwd: "/repo".into(),
            file_path: None,
            edit_write_tool: true,
            file_path_present: false,
            path_inside_repo: true,
            session_env_path: false,
            hook_applies: true,
            effective_repo: Some("/repo".into()),
            branch_known: true,
            branch: Some("feature/git-hygiene".into()),
            protected_branch: false,
            worktree: false,
            merge_in_progress: false,
            protected_branch_block: false,
            has_uncommitted_changes_known: true,
            has_uncommitted_changes: false,
            changed_files_known: false,
            changed_files: Vec::new(),
            changed_file_count: 0,
            uncommitted_file_limit_exceeded: false,
            should_deny: false,
            decision: hooks::git_hygiene::GitHygieneDecision::Allow,
        };

        let identifier = git_hygiene_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("file-path-present-false"));
        assert!(!identifier.contains("missing-file"));
        assert!(!identifier.contains("file-path-sha256"));
    }

    #[test]
    fn git_hygiene_graph_identifier_rejects_present_file_path_without_evidence() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("git-hygiene-missing-file-evidence-id".into()),
            tool_name: Some("Edit".into()),
            cwd: Some("/repo".into()),
            ..Default::default()
        };
        let evaluation = hooks::git_hygiene::GitHygieneEvaluation {
            tool: Some("Edit".into()),
            cwd: "/repo".into(),
            file_path: None,
            edit_write_tool: true,
            file_path_present: true,
            path_inside_repo: true,
            session_env_path: false,
            hook_applies: true,
            effective_repo: Some("/repo".into()),
            branch_known: true,
            branch: Some("feature/git-hygiene".into()),
            protected_branch: false,
            worktree: false,
            merge_in_progress: false,
            protected_branch_block: false,
            has_uncommitted_changes_known: true,
            has_uncommitted_changes: false,
            changed_files_known: false,
            changed_files: Vec::new(),
            changed_file_count: 0,
            uncommitted_file_limit_exceeded: false,
            should_deny: false,
            decision: hooks::git_hygiene::GitHygieneDecision::Allow,
        };

        let err = git_hygiene_graph_identifier(&input, &evaluation).unwrap_err();

        assert!(err.to_string().contains("concrete file path evidence"));
    }

    #[test]
    fn tool_usage_graph_authority_writes_langgraph_audit_for_allow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let classifier =
            sentinel_application::reversibility_classifier::StaticReversibilityClassifier::empty()
                .with(
                    "Edit",
                    sentinel_domain::reversibility::ReversibilityClass::ReversibleWithEffort,
                );
        let session_id = format!("tool-usage-{}", uuid::Uuid::new_v4());
        let transcript_path = home.path().join("transcript.jsonl");
        let transcript_rows = [
            serde_json::json!({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": "mcp__sequential-thinking__sequentialthinking",
                        "input": {}
                    }]
                }
            }),
            serde_json::json!({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "name": "ExitPlanMode",
                        "input": {}
                    }]
                }
            }),
        ];
        let transcript = transcript_rows
            .iter()
            .map(|row| serde_json::to_string(row).expect("transcript row"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&transcript_path, format!("{transcript}\n")).expect("write transcript");
        let task_dir = home.path().join(".claude").join("tasks").join(&session_id);
        std::fs::create_dir_all(&task_dir).expect("task dir");
        std::fs::write(
            task_dir.join("1.json"),
            serde_json::json!({
                "id": "1",
                "subject": "Graph-authorized edit",
                "status": "in_progress"
            })
            .to_string(),
        )
        .expect("task file");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(session_id),
            tool_name: Some("Edit".into()),
            transcript_path: Some(transcript_path.display().to_string()),
            tool_input: Some(serde_json::json!({
                "file_path": "/tmp/sentinel-tool-usage.txt",
                "old_string": "old",
                "new_string": "new"
            })),
            ..Default::default()
        };

        let output = authorize_tool_usage_with_graph(&input, &fs, &classifier, true);

        assert_eq!(output.blocked, None);
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("tool-usage.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"tool_usage\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
        assert!(graph_rows.contains("\"authorization_checkpoint\""));
    }

    #[test]
    fn phase_gate_graph_authority_writes_langgraph_audit_for_allow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let mut state = SessionState::new("phase-gate-graph-test");
        let workflows = HashMap::new();
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("phase-gate-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Edit".into()),
            tool_input: Some(serde_json::json!({
                "file_path": "/tmp/sentinel-phase-gate.txt",
                "old_string": "old",
                "new_string": "new"
            })),
            ..Default::default()
        };

        let output = authorize_phase_gate_with_graph(&input, &mut state, &workflows, &fs);

        assert_eq!(output.blocked, None);
        assert_eq!(state.tool_calls, 1);
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("phase-gate.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"phase_gate\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
        assert!(graph_rows.contains("\"authorization_checkpoint\""));
    }

    #[test]
    fn phase_gate_graph_identifier_records_absent_reason_without_fake_hash() {
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let mut state = SessionState::new("phase-gate-absent-id");
        let workflows = HashMap::new();
        let input = sentinel_domain::events::HookInput {
            session_id: Some("phase-gate-absent-id".into()),
            tool_name: Some("Edit".into()),
            tool_input: Some(serde_json::json!({
                "file_path": "/tmp/sentinel-phase-gate.txt",
                "old_string": "old",
                "new_string": "new"
            })),
            ..Default::default()
        };
        let evaluation = hooks::phase_gate::evaluate(&input, &mut state, &workflows, &fs);

        let identifier = phase_gate_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("reason-present-false"));
        assert!(!identifier.contains("no-reason"));
        assert!(!identifier.contains("reason-sha256"));
    }

    #[test]
    fn bug_task_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let repo = tempfile::tempdir().expect("temp repo");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = TasksRepoGit {
            root: repo.path().to_path_buf(),
        };
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let session_id = format!("bug-task-{}", uuid::Uuid::new_v4());
        let post_input = sentinel_domain::events::HookInput {
            session_id: Some(session_id.clone()),
            tool_name: Some("Bash".into()),
            cwd: Some(repo.path().display().to_string()),
            tool_result: Some(serde_json::Value::String(
                "test result: FAILED. 0 passed; 1 failed".to_string(),
            )),
            ..Default::default()
        };
        let post_output = hooks::bug_task_gate::process_posttool(&post_input, &ctx);
        assert_eq!(post_output.blocked, None);
        let pre_input = sentinel_domain::events::HookInput {
            session_id: Some(session_id),
            tool_name: Some("Bash".into()),
            cwd: Some(repo.path().display().to_string()),
            tool_input: Some(serde_json::json!({
                "command": "cargo build"
            })),
            ..Default::default()
        };

        let output = authorize_bug_task_with_graph(&pre_input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("bug-task.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"bug_task\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn skill_invocation_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let session_id = format!("skill-invocation-{}", uuid::Uuid::new_v4());
        let state_dir = home.path().join(".claude").join("sentinel").join("state");
        std::fs::create_dir_all(&state_dir).expect("state dir");
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(session_id.as_bytes());
        let state_path = state_dir.join(format!(
            "skill-pending-{}.json",
            hex::encode(&hasher.finalize()[..6])
        ));
        std::fs::write(
            state_path,
            serde_json::json!({
                "skill": "linear",
                "skill_path": "~/.claude/skills/linear/SKILL.md",
                "detected_at": chrono::Utc::now().to_rfc3339(),
                "session_id": session_id,
            })
            .to_string(),
        )
        .expect("pending skill state");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(session_id),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "cargo build"
            })),
            ..Default::default()
        };

        let output = authorize_skill_invocation_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("skill-invocation.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"skill_invocation\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn plan_title_graph_authority_writes_langgraph_audit_for_titleless_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("plan-title-{}", uuid::Uuid::new_v4())),
            tool_name: Some("ExitPlanMode".into()),
            tool_input: Some(serde_json::json!({
                "plan": "   \n\n"
            })),
            ..Default::default()
        };

        let output = authorize_plan_title_with_graph(&input);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("plan-title.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"plan_title\""));
        assert!(graph_rows.contains("\"decision\":\"block-titleless\""));
    }

    #[test]
    fn plan_title_graph_identifier_records_absent_plan_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("plan-title-absent-id".into()),
            tool_name: Some("ExitPlanMode".into()),
            tool_input: Some(serde_json::json!({})),
            ..Default::default()
        };
        let evaluation = hooks::plan_title_gate::evaluate(&input);

        let identifier = plan_title_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("plan-present-false"));
        assert!(!identifier.contains("missing-plan"));
        assert!(!identifier.contains("plan-sha256"));
    }

    #[test]
    fn production_override_graph_authority_writes_langgraph_audit_for_arm() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let mut state = SessionState::new("production-override-graph-test");
        let input = sentinel_domain::events::HookInput {
            session_id: Some("production-override-graph-test".into()),
            prompt: Some("production override - hotfix deploy".into()),
            ..Default::default()
        };

        let output = authorize_production_override_with_graph(&input, &mut state);

        assert!(state.production_override_armed());
        assert!(output.system_message.is_some());
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("production-override.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"production_override\""));
        assert!(graph_rows.contains("\"decision\":\"arm\""));
    }

    #[test]
    fn production_override_graph_authority_writes_langgraph_audit_for_noop() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let mut state = SessionState::new("production-override-noop-graph-test");
        let input = sentinel_domain::events::HookInput {
            session_id: Some("production-override-noop-graph-test".into()),
            prompt: Some("please summarize current status".into()),
            ..Default::default()
        };

        let output = authorize_production_override_with_graph(&input, &mut state);

        assert!(!state.production_override_armed());
        assert!(output.system_message.is_none());
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("production-override.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"production_override\""));
        assert!(graph_rows.contains("\"decision\":\"allow-noop\""));
        assert!(graph_rows.contains("\"authorization_checkpoint\""));
    }

    #[test]
    fn production_override_graph_identifier_records_absent_prompt_without_fake_hash() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("production-override-absent-id".into()),
            prompt: None,
            ..Default::default()
        };
        let evaluation = hooks::production_override::evaluate(&input, false);

        let identifier =
            production_override_graph_identifier(&input, &evaluation).expect("identifier");

        assert!(identifier.contains("prompt-present-false"));
        assert!(!identifier.contains("missing-prompt"));
        assert!(!identifier.contains("prompt-sha256"));
    }

    #[test]
    fn pr_merge_graph_authority_writes_langgraph_audit_for_ask() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let env = NoopEnv;
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("pr-merge-graph-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "gh pr merge 123 --squash"
            })),
            ..Default::default()
        };

        let output = authorize_pr_merge_with_graph(&input, &env, &UnfetchableGh);

        assert!(output.blocked.is_none());
        let hso = output.hook_specific_output.expect("hook-specific output");
        assert_eq!(
            hso.permission_decision,
            Some(sentinel_domain::events::PermissionDecision::Ask)
        );
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("pr-merge.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"pr_merge\""));
        assert!(graph_rows.contains("\"decision\":\"ask\""));
    }

    #[test]
    fn db_ops_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("db-ops-graph-{}", uuid::Uuid::new_v4())),
            tool_name: Some("Bash".into()),
            tool_input: Some(serde_json::json!({
                "command": "psql production -c 'DROP TABLE users;'"
            })),
            ..Default::default()
        };

        let output = authorize_db_ops_with_graph(&input);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("db-ops.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"db_ops\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn doppler_auth0_graph_authority_writes_langgraph_audit_for_block() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("doppler-auth0-graph-{}", uuid::Uuid::new_v4())),
            tool_name: Some("mcp__doppler__set_secret".into()),
            tool_input: Some(serde_json::json!({
                "project": "sentinel",
                "config": "prod",
                "name": "OPENROUTER_API_KEY",
                "value": "redacted"
            })),
            ..Default::default()
        };

        let output = authorize_doppler_auth0_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("doppler-auth0.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"doppler_auth0\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
    }

    #[test]
    fn doppler_auth0_graph_authority_records_absent_tool_input_as_prod_evidence() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let git = sentinel_infrastructure::git::RealGit;
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let process = sentinel_infrastructure::process::RealProcess;
        let env = NoopEnv;
        let memory_mcp = NoopMemoryMcp;
        let ctx = hooks::HookContext {
            git: &git,
            vector_store: None,
            fs: &fs,
            process: &process,
            llm: None,
            memory_mcp: &memory_mcp,
            env: &env,
            linear_lookup: None,
        };
        let input = sentinel_domain::events::HookInput {
            session_id: Some(format!("doppler-auth0-no-input-{}", uuid::Uuid::new_v4())),
            tool_name: Some("mcp__doppler__set_secret".into()),
            ..Default::default()
        };

        let output = authorize_doppler_auth0_with_graph(&input, &ctx);

        assert_eq!(output.blocked, Some(true));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("doppler-auth0.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"doppler_auth0\""));
        assert!(graph_rows.contains("\"decision\":\"block\""));
        assert!(graph_rows.contains("\"tool_input_present\":false"));
        assert!(graph_rows.contains("\"production_target\":true"));
        let synthetic_operation = ["unknown", "-operation"].concat();
        assert!(!graph_rows.contains(&synthetic_operation));
    }

    #[test]
    fn doppler_auth0_graph_identifier_rejects_missing_operation_evidence() {
        let input = sentinel_domain::events::HookInput {
            session_id: Some("doppler-auth0-missing-operation-id".into()),
            tool_name: Some("mcp__doppler__".into()),
            ..Default::default()
        };
        let evaluation = hooks::doppler_auth0_gate::DopplerAuth0Evaluation {
            tool: Some("mcp__doppler__".into()),
            operation: None,
            provider: hooks::doppler_auth0_gate::DopplerAuth0Provider::Doppler,
            router_management: false,
            read_only: false,
            mutation: true,
            autopilot: false,
            tool_input_present: true,
            production_target: true,
            session_id_present: true,
            signed_override_active: false,
            auth0_override_supported: false,
            should_block: true,
            decision: hooks::doppler_auth0_gate::DopplerAuth0Decision::Block,
            block_reason: Some("blocked".into()),
        };

        let err = doppler_auth0_graph_identifier(&input, &evaluation).unwrap_err();

        assert!(err.to_string().contains("concrete operation evidence"));
    }

    #[test]
    fn a3_dry_run_graph_authority_writes_langgraph_audit_for_allow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");
        let fs = sentinel_infrastructure::filesystem::RealFileSystem;
        let classifier =
            sentinel_application::reversibility_classifier::StaticReversibilityClassifier::empty()
                .with(
                    "Edit",
                    sentinel_domain::reversibility::ReversibilityClass::Irreversible,
                );
        let auditor = sentinel_application::auditor::StaticAuditor::pass(0.95);
        let session_id = format!("a3-graph-hook-test-{}", uuid::Uuid::new_v4());
        let fixture_path = format!("/tmp/{session_id}.txt");
        let input = sentinel_domain::events::HookInput {
            session_id: Some(session_id.clone()),
            tool_name: Some("Edit".into()),
            tool_input: Some(serde_json::json!({
                "file_path": fixture_path,
                "old_string": "old",
                "new_string": "new",
                "_intent": "apply the approved test edit",
                "_reasoning": "the edit is scoped to the test fixture",
                "_expected_effect": "the fixture content changes from old to new"
            })),
            ..Default::default()
        };

        let output = authorize_dry_run_then_commit_with_graph(&input, &fs, &classifier, &auditor);

        assert_eq!(output.blocked, None);
        let tool_input = input.tool_input.as_ref().unwrap();
        let action_hash = hooks::dry_run_then_commit::action_hash_for("Edit", tool_input);
        assert!(hooks::dry_run_then_commit::has_dry_run_approval(
            &fs,
            &session_id,
            &action_hash
        ));
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("a3-dry-run.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"dry_run\""));
        assert!(graph_rows.contains("\"decision\":\"allow-and-record-approval\""));
    }

    #[test]
    fn ba3_requirements_graph_authority_fails_closed_without_matrix_adapter() {
        let input = sentinel_domain::events::HookInput {
            extra: serde_json::Map::from_iter([(
                "is_recommendation".to_string(),
                serde_json::Value::Bool(true),
            )]),
            ..Default::default()
        };

        let output = authorize_requirements_traceability_with_graph(
            &input,
            None,
            hooks::requirements_traceability_gate::ValidationMode::DefaultBlocking,
        );

        assert_eq!(output.blocked, Some(true));
        let reason = output
            .hook_specific_output
            .as_ref()
            .and_then(|h| h.permission_decision_reason.as_deref())
            .unwrap_or_default();
        assert!(reason.contains("requirement-matrix adapter"));
    }

    #[test]
    fn ba1_provenance_graph_authority_fails_closed_without_store_adapter() {
        let input = sentinel_domain::events::HookInput {
            extra: serde_json::Map::from_iter([(
                "artifacts".to_string(),
                serde_json::json!([{
                    "artifact_id": "artifact-1",
                    "content_hash": "hash-v1",
                    "provenance_class": "SystemOfRecord",
                    "retrieved_at": chrono::Utc::now(),
                }]),
            )]),
            ..Default::default()
        };

        let output = authorize_provenance_validate_with_graph(
            &input,
            None,
            hooks::provenance_validate::ValidationMode::DefaultBlocking,
        );

        assert_eq!(output.blocked, Some(true));
        let reason = output
            .hook_specific_output
            .as_ref()
            .and_then(|h| h.permission_decision_reason.as_deref())
            .unwrap_or_default();
        assert!(reason.contains("provenance store adapter"));
    }

    #[test]
    fn ba1_provenance_graph_authority_writes_langgraph_audit_for_allow() {
        use sentinel_domain::ports::ProvenanceWritePort;

        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");

        let retrieved_at = chrono::Utc::now();
        let store = JsonlProvenanceStore::at_path(home.path().join("records.jsonl"));
        store
            .record(sentinel_domain::ba::RetrievalRecord {
                artifact_id: "artifact-1".into(),
                connector_name: "mcp__linear__get_issue".into(),
                content_hash: "hash-v1".into(),
                provenance_class: sentinel_domain::ba::ProvenanceClass::SystemOfRecord,
                session_id: "ba1-graph-hook-test".into(),
                retrieved_at,
            })
            .unwrap();

        let input = sentinel_domain::events::HookInput {
            session_id: Some("ba1-graph-hook-test".into()),
            tool_name: Some("mcp__ba_orchestrator__publish_internal_brief".into()),
            extra: serde_json::Map::from_iter([(
                "artifacts".to_string(),
                serde_json::json!([{
                    "artifact_id": "artifact-1",
                    "content_hash": "hash-v1",
                    "provenance_class": "SystemOfRecord",
                    "retrieved_at": retrieved_at,
                }]),
            )]),
            ..Default::default()
        };

        let output = authorize_provenance_validate_with_graph(
            &input,
            Some(&store),
            hooks::provenance_validate::ValidationMode::DefaultBlocking,
        );

        assert_eq!(output.blocked, None);
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("ba-provenance.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"ba_provenance\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
    }

    fn complete_spec_challenge(work_id: &str) -> sentinel_domain::spec_challenge::SpecChallenge {
        use chrono::{TimeZone, Utc};
        use sentinel_domain::reversibility::ReversibilityClass;
        use sentinel_domain::spec_challenge::{
            Alternative, Ambiguity, Assumption, AssumptionConfidence, ChallengeCategory,
            GapResolution, SpecChallenge, SpecGap, SpecReference, WorkId,
        };

        SpecChallenge {
            work_id: WorkId::new(work_id).unwrap(),
            agent_id: "agent-x".to_string(),
            challenged_spec: SpecReference {
                hash: "abc".to_string(),
                source: "issue X".to_string(),
            },
            reversibility_class: ReversibilityClass::Irreversible,
            assumptions: ChallengeCategory::new(vec![Assumption {
                statement: "postgres".to_string(),
                confidence: AssumptionConfidence::High,
                blast_if_wrong: ReversibilityClass::Irreversible,
            }]),
            gaps: ChallengeCategory::new(vec![SpecGap {
                topic: "auth".to_string(),
                how_resolved: GapResolution::OperatorClarified,
                inference_source: None,
            }]),
            ambiguities: ChallengeCategory::new(vec![Ambiguity {
                spec_excerpt: "ship fast".to_string(),
                interpretations: vec!["p99".to_string(), "throughput".to_string()],
                chosen: "p99".to_string(),
                rationale: "user-visible".to_string(),
            }]),
            alternatives_considered: ChallengeCategory::new(vec![Alternative {
                description: "redis".to_string(),
                why_rejected: "durability".to_string(),
            }]),
            constraints_not_satisfied: ChallengeCategory::none("all met"),
            created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        }
    }

    #[test]
    fn a13_spec_challenge_graph_authority_fails_closed_without_store_adapter() {
        let challenge = complete_spec_challenge("a13-missing-store");
        let input = sentinel_domain::events::HookInput {
            extra: serde_json::Map::from_iter([(
                "spec_challenge".to_string(),
                serde_json::to_value(challenge).unwrap(),
            )]),
            ..Default::default()
        };

        let output = authorize_spec_challenge_with_graph(
            &input,
            sentinel_domain::reversibility::ReversibilityClass::Irreversible,
            None,
            None,
            hooks::spec_challenge_gate::A13EnforcementMode::DefaultBlocking,
            hooks::spec_challenge_gate::DEFAULT_CATASTROPHIC_AXIS_THRESHOLD,
        );

        assert_eq!(output.blocked, Some(true));
        let reason = output
            .hook_specific_output
            .as_ref()
            .and_then(|h| h.permission_decision_reason.as_deref())
            .unwrap_or_default();
        assert!(reason.contains("SpecChallenge store adapter"));
    }

    #[test]
    fn a13_spec_challenge_graph_authority_writes_langgraph_audit_for_allow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let store_dir = tempfile::tempdir().expect("store dir");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");

        let challenge = complete_spec_challenge("a13-graph-hook-test");
        let store = FilesystemSpecChallengeStore::at_dir(store_dir.path().to_path_buf());
        let input = sentinel_domain::events::HookInput {
            session_id: Some("a13-graph-hook-test".into()),
            tool_name: Some("mcp__ba_orchestrator__publish_internal_brief".into()),
            extra: serde_json::Map::from_iter([(
                "spec_challenge".to_string(),
                serde_json::to_value(challenge).unwrap(),
            )]),
            ..Default::default()
        };

        let output = authorize_spec_challenge_with_graph(
            &input,
            sentinel_domain::reversibility::ReversibilityClass::Irreversible,
            Some(&store),
            None,
            hooks::spec_challenge_gate::A13EnforcementMode::DefaultBlocking,
            hooks::spec_challenge_gate::DEFAULT_CATASTROPHIC_AXIS_THRESHOLD,
        );

        assert_eq!(output.blocked, None);
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("a13-spec-challenge.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"spec_challenge\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
    }

    #[test]
    fn ba3_requirements_graph_authority_writes_langgraph_audit_for_allow() {
        let _lock = env_lock();
        let home = tempfile::tempdir().expect("temp home");
        let matrix_dir = tempfile::tempdir().expect("matrix dir");
        let _home = EnvGuard::set("SENTINEL_HOME", home.path());
        let _backend = EnvGuard::set("SENTINEL_DECISION_GRAPH_CHECKPOINTER", "sqlite");

        let requirement = sentinel_domain::ba::RequirementRef {
            orchestration_id: "case-1".into(),
            matrix_row_id: "R-001".into(),
            content_hash: "hash-v1".into(),
            statement: "Stakeholder need".into(),
        };
        std::fs::write(
            matrix_dir.path().join("case-1.json"),
            serde_json::to_string(&serde_json::json!({ "rows": [requirement.clone()] })).unwrap(),
        )
        .unwrap();
        let matrix = FilesystemRequirementMatrix::at_dir(matrix_dir.path().to_path_buf());
        let input = sentinel_domain::events::HookInput {
            session_id: Some("ba3-graph-hook-test".into()),
            tool_name: Some("mcp__ba_orchestrator__publish_internal_brief".into()),
            extra: serde_json::Map::from_iter([
                (
                    "is_recommendation".to_string(),
                    serde_json::Value::Bool(true),
                ),
                (
                    "requirement_refs".to_string(),
                    serde_json::json!([requirement]),
                ),
            ]),
            ..Default::default()
        };

        let output = authorize_requirements_traceability_with_graph(
            &input,
            Some(&matrix),
            hooks::requirements_traceability_gate::ValidationMode::DefaultBlocking,
        );

        assert_eq!(output.blocked, None);
        let graph_rows = std::fs::read_to_string(
            home.path()
                .join(".claude")
                .join("sentinel")
                .join("metrics")
                .join("ba-requirements.graph-runs.jsonl"),
        )
        .expect("graph audit rows");
        assert!(graph_rows.contains("\"workflow_authority\":\"langgraph\""));
        assert!(graph_rows.contains("\"graph\":\"ba_requirements\""));
        assert!(graph_rows.contains("\"decision\":\"allow\""));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn project_phase_graph_workflows_prefers_checkpoint_over_old_workflows_field() {
        let _lock = env_lock();
        let tmp = tempfile::tempdir().expect("tmpdir");
        let _home = EnvGuard::set("SENTINEL_HOME", tmp.path());
        let session = "hook-graph-authority";
        let wf = workflow("linear");

        let db_path = phase_graph_db_path(session).expect("db path");
        let saver = sentinel_graph::phase_checkpointer_from_env(&db_path)
            .await
            .expect("saver");
        let graph =
            sentinel_graph::compile_skill_graph_with_checkpointer(&wf, saver).expect("compile");
        graph
            .run_until_gate("linear", session)
            .await
            .expect("initial gate checkpoint");
        graph
            .apply_verdict("linear", session, "claim", true)
            .await
            .expect("graph verdict");

        let mut state = SessionState::new(session);
        state.set_active_skill("linear");
        let mut stale = WorkflowState::new("linear", session);
        stale.completed_phases = vec!["stale".to_string()];
        inject_old_workflows_field(&mut state, "linear", stale);
        let workflows = HashMap::from([("linear".to_string(), wf)]);

        project_phase_graph_workflows(&mut state, &workflows)
            .await
            .expect("projection");

        let projected = state.graph_workflow("linear").expect("projected workflow");
        assert_eq!(projected.completed_phases, vec!["claim".to_string()]);
        assert_eq!(projected.current_phase, Some(1));
        assert!(state.has_graph_workflow("linear"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn project_phase_graph_workflows_ignores_old_workflows_field_without_checkpoint() {
        let _lock = env_lock();
        let tmp = tempfile::tempdir().expect("tmpdir");
        let _home = EnvGuard::set("SENTINEL_HOME", tmp.path());
        let session = "hook-graph-empty-authority";
        let wf = workflow("linear");
        let workflows = HashMap::from([("linear".to_string(), wf)]);

        let mut state = SessionState::new(session);
        state.set_active_skill("linear");
        let mut stale = WorkflowState::new("linear", session);
        stale.completed_phases = vec!["stale".to_string()];
        inject_old_workflows_field(&mut state, "linear", stale);

        project_phase_graph_workflows(&mut state, &workflows)
            .await
            .expect("projection");

        assert!(
            state.graph_workflow("linear").is_none(),
            "without a graph checkpoint, old workflow state must be ignored"
        );
        assert!(!state.has_graph_workflow("linear"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn project_phase_graph_workflows_does_not_create_unrelated_workflows() {
        let _lock = env_lock();
        let tmp = tempfile::tempdir().expect("tmpdir");
        let _home = EnvGuard::set("SENTINEL_HOME", tmp.path());
        let workflows = HashMap::from([("linear".to_string(), workflow("linear"))]);
        let mut state = SessionState::new("hook-graph-unrelated");

        project_phase_graph_workflows(&mut state, &workflows)
            .await
            .expect("projection");

        assert!(!state.has_any_graph_workflow());
    }

    /// Regression: hook process must exit within a bounded window (no lingering
    /// threads) while still allowing the enterprise startup path to initialize
    /// A2 routing, A3 auditor wiring, scorer clients, and graph projection under
    /// full-suite load.
    #[tokio::test]
    async fn test_hook_exits_within_timeout() {
        let _lock = env_lock();
        let Some(engine) = sentinel_engine_bin() else {
            eprintln!("Skipping: sentinel-engine not found in target/debug or target/release");
            return;
        };

        let home = tempfile::tempdir().expect("temp home");
        write_authoritative_workflow_config(home.path());
        let mut child = tokio::process::Command::new(&engine)
            .args(["hook", "--event", "UserPromptSubmit"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("SENTINEL_HOME", home.path())
            .env("HOME", home.path())
            .env("USERPROFILE", home.path())
            .env("OPENROUTER_API_KEY", "test-openrouter-key")
            .env("SENTINEL_SPEC_CHALLENGE_SCORER_PROVIDER", "openrouter")
            .env(
                "SENTINEL_SPEC_CHALLENGE_SCORER_MODEL",
                "anthropic/claude-opus-4.7",
            )
            .env("SENTINEL_AUDITOR_PROVIDER", "openrouter")
            .env_remove("QDRANT_URL")
            .env_remove("LINEAR_API_KEY")
            .env_remove("CEREBRAS_API_KEY")
            .env_remove("OLLAMA_API_KEY")
            .env("SENTINEL_ALLOW_TERMINAL", "1")
            .kill_on_drop(true)
            .spawn()
            .expect("failed to spawn sentinel-engine");

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(b"{\"session_id\":\"regression-exit-test\"}\n")
                .await
                .unwrap();
            stdin.shutdown().await.unwrap();
        }

        // Windows git subprocesses are slower, and full-suite cargo test load
        // can briefly starve child processes. Keep this far below "hung" while
        // avoiding scheduler false positives.
        let timeout_secs = if cfg!(windows) { 25 } else { 20 };
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await;

        assert!(
            result.is_ok(),
            "sentinel-engine did not exit within {timeout_secs}s — possible hang"
        );
    }

    /// Regression: stdout must be valid JSON (no tracing leaks).
    #[tokio::test]
    async fn test_hook_stdout_is_valid_json() {
        let _lock = env_lock();
        let Some(engine) = sentinel_engine_bin() else {
            eprintln!("Skipping: sentinel-engine not found in target/debug or target/release");
            return;
        };

        let home = tempfile::tempdir().expect("temp home");
        write_authoritative_workflow_config(home.path());
        let mut child = tokio::process::Command::new(&engine)
            .args(["hook", "--event", "UserPromptSubmit"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("SENTINEL_HOME", home.path())
            .env("HOME", home.path())
            .env("USERPROFILE", home.path())
            .env("OPENROUTER_API_KEY", "test-openrouter-key")
            .env("SENTINEL_SPEC_CHALLENGE_SCORER_PROVIDER", "openrouter")
            .env(
                "SENTINEL_SPEC_CHALLENGE_SCORER_MODEL",
                "anthropic/claude-opus-4.7",
            )
            .env("SENTINEL_AUDITOR_PROVIDER", "openrouter")
            .env_remove("QDRANT_URL")
            .env_remove("LINEAR_API_KEY")
            .env_remove("CEREBRAS_API_KEY")
            .env_remove("OLLAMA_API_KEY")
            .env("SENTINEL_ALLOW_TERMINAL", "1")
            .kill_on_drop(true)
            .spawn()
            .expect("failed to spawn");

        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(b"{\"session_id\":\"regression-json-test\"}\n")
                .await
                .unwrap();
            stdin.shutdown().await.unwrap();
        }

        // Hook startup can contend with other subprocess tests in the full
        // package suite; keep enough headroom to test JSON stdout rather than
        // scheduler timing.
        let timeout_secs = if cfg!(windows) { 25 } else { 20 };
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            child.wait_with_output(),
        )
        .await
        .expect("timed out")
        .expect("wait failed");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.is_empty(), "stdout should not be empty");

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(stdout.trim());
        assert!(parsed.is_ok(), "stdout is not valid JSON: {stdout}");

        assert!(
            !stdout.contains("[2m"),
            "stdout contains ANSI escape (tracing leak)"
        );
        assert!(
            !stdout.contains("WARN"),
            "stdout contains WARN (tracing leak)"
        );
    }

    /// Regression: `from_env()` (and any other blocking init) must run INSIDE the
    /// 8-second timeout, not before it.  Previously, `RigClassifier::from_env()` was
    /// called synchronously outside the `tokio::time::timeout` block, so a stall in
    /// TLS root-cert loading (Windows schannel) would hang the hook indefinitely.
    ///
    /// This test validates the structural fix: a `spawn_blocking` task that sleeps 2 s
    /// (simulating a slow Windows TLS init) inside the timeout block should be
    /// *abandoned* after the timeout fires, not awaited to completion.
    #[tokio::test]
    async fn test_classifier_init_timeout_fires_when_blocked() {
        let short_timeout = std::time::Duration::from_millis(100);
        let start = std::time::Instant::now();

        // Simulate the fixed code path: spawn_blocking wrapping a slow from_env,
        // both running inside a tokio::time::timeout.
        let result: Result<Option<()>, tokio::time::error::Elapsed> =
            tokio::time::timeout(short_timeout, async {
                // Mimic RigClassifier::from_env taking seconds (e.g. Windows TLS cert load).
                // Keep this short enough that the detached blocking task does
                // not poison neighboring subprocess timeout tests.
                let classifier: Option<()> = tokio::task::spawn_blocking(|| {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    None::<()>
                })
                .await
                .ok()
                .flatten();
                classifier
            })
            .await;

        // The timeout must fire — if from_env had been outside the timeout (the old
        // bug) this whole call would have blocked for the full simulated stall.
        assert!(result.is_err(), "timeout should have fired");

        // And it should fire promptly (well under 1 s).
        assert!(
            start.elapsed() < std::time::Duration::from_secs(1),
            "timeout took too long: {:?}",
            start.elapsed()
        );
    }
}
