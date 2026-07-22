//! Graph-backed PR merge/close authorization.
//!
//! The application hook computes deterministic facts for `gh pr merge` and
//! `gh pr close` Bash commands. This graph authorizes the resulting allow,
//! ask, or autopilot-reminder decision through durable LangGraph checkpoints so
//! PR lifecycle actions are audited as first-class decisions.

use langgraph_core::application::services::{CompilationResult, GraphCompiler};
use langgraph_core::domain::value_objects::{
    NodeConfig, NodeError, StateError, StateSchema, END, START,
};
use langgraph_core::StateGraphBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use sentinel_application::hooks::pr_merge_gate::{PrMergeEvaluation, PrMergeOperation};

use crate::decision_graph_introspection::{
    checkpoint_history, emit_decision_node_event, stream_decision_run,
    terminal_decision_checkpoint_result, topology, validate_decision_graph_run, write_history,
    DecisionGraphCheckpointSnapshot, DecisionGraphStreamPart, DecisionGraphTopology,
    DecisionGraphWriteHistoryEntry,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrMergeDecision {
    #[default]
    Unclassified,
    Allow,
    Ask,
    AllowAutopilotReminder,
    /// Verified PR state contradicts the merge — hard block.
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrMergeState {
    pub identifier: String,
    pub tool: Option<String>,
    pub bash_tool: bool,
    pub command_present: bool,
    pub command_sha256: Option<String>,
    pub operation: String,
    pub pr_lifecycle_operation: bool,
    /// The fetched-state verdict the hook derived: `not_applicable`, `clear`,
    /// `blocked`, or `unknown`. This is the fact the decision hangs on now —
    /// the graph replica must model it or it will deny every real decision on
    /// an authority mismatch.
    pub merge_verdict: String,
    pub autopilot: bool,
    pub permission_prompt_required: bool,
    pub context_reminder_required: bool,
    pub decision: PrMergeDecision,
}

impl PrMergeState {
    #[must_use]
    pub fn from_evaluation(identifier: impl Into<String>, evaluation: &PrMergeEvaluation) -> Self {
        let command_sha256 = evaluation
            .command
            .as_deref()
            .filter(|_| evaluation.command_present)
            .map(command_sha256);
        Self {
            identifier: identifier.into(),
            tool: evaluation.tool.clone(),
            bash_tool: evaluation.bash_tool,
            command_present: evaluation.command_present,
            command_sha256,
            operation: operation_label(evaluation.operation).to_string(),
            pr_lifecycle_operation: evaluation.gated,
            merge_verdict: evaluation.verdict.label().to_string(),
            autopilot: evaluation.autopilot,
            permission_prompt_required: evaluation.permission_prompt_required,
            context_reminder_required: evaluation.context_reminder_required,
            decision: PrMergeDecision::Unclassified,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PrMergeGraphRun {
    pub state: PrMergeState,
    pub thread_id: String,
    pub checkpoints: Vec<DecisionGraphCheckpointSnapshot<PrMergeState>>,
    pub write_history: Vec<DecisionGraphWriteHistoryEntry>,
    pub stream: Vec<DecisionGraphStreamPart>,
    pub topology: DecisionGraphTopology,
}

#[derive(Debug, Clone)]
pub struct PrMergeAuthorization {
    decision: PrMergeDecision,
    thread_id: String,
    checkpoint_id: String,
}

impl PrMergeAuthorization {
    #[must_use]
    pub fn decision(&self) -> PrMergeDecision {
        self.decision
    }

    #[must_use]
    pub fn checkpoint_ref(&self) -> String {
        format!("{}#{}", self.thread_id, self.checkpoint_id)
    }
}

impl PrMergeGraphRun {
    #[must_use]
    pub fn pr_merge_authorization(&self) -> Result<Option<PrMergeAuthorization>, String> {
        if self.state.decision == PrMergeDecision::Unclassified {
            return Ok(None);
        }
        let checkpoint_id = terminal_decision_checkpoint_result(
            "pr_merge",
            &self.thread_id,
            &self.state,
            &self.checkpoints,
            &self.write_history,
        )?
        .checkpoint_id
        .clone();
        Ok(Some(PrMergeAuthorization {
            decision: self.state.decision,
            thread_id: self.thread_id.clone(),
            checkpoint_id,
        }))
    }
}

const CLASSIFY: &str = "classify";
const ALLOW: &str = "allow";
const ASK: &str = "ask";
const ALLOW_AUTOPILOT_REMINDER: &str = "allow_autopilot_reminder";
const DENY: &str = "deny";

pub type PrMergeGraph = CompilationResult<PrMergeState>;

#[must_use]
pub const fn operation_label(operation: PrMergeOperation) -> &'static str {
    match operation {
        PrMergeOperation::None => "none",
        PrMergeOperation::Merge => "merge",
        PrMergeOperation::Close => "close",
    }
}

#[must_use]
pub fn pr_merge_decision_label(decision: PrMergeDecision) -> &'static str {
    match decision {
        PrMergeDecision::Unclassified => "unclassified",
        PrMergeDecision::Allow => "allow",
        PrMergeDecision::Ask => "ask",
        PrMergeDecision::AllowAutopilotReminder => "allow-autopilot-reminder",
        PrMergeDecision::Deny => "deny",
    }
}

#[must_use]
pub fn command_sha256(command: &str) -> String {
    hex::encode(Sha256::digest(command.as_bytes()))
}

fn hex_digest_present(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn optional_hex_digest_present(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(hex_digest_present)
}

fn expected_decision(state: &PrMergeState) -> PrMergeDecision {
    if state.merge_verdict == "blocked" {
        PrMergeDecision::Deny
    } else if state.permission_prompt_required {
        PrMergeDecision::Ask
    } else if state.context_reminder_required {
        PrMergeDecision::AllowAutopilotReminder
    } else {
        PrMergeDecision::Allow
    }
}

fn node_config(
    node: &str,
    checkpointer_backend: &str,
    checkpointer_scope: &str,
    checkpointer_tenant_scope: &str,
) -> NodeConfig {
    NodeConfig::new()
        .with_metadata("sentinel.graph", "pr_merge")
        .with_metadata("sentinel.node", node)
        .with_metadata("sentinel.checkpointer_backend", checkpointer_backend)
        .with_metadata("sentinel.checkpointer_scope", checkpointer_scope)
        .with_metadata(
            "sentinel.checkpointer_tenant_scope",
            checkpointer_tenant_scope,
        )
}

fn pr_merge_state_schema() -> StateSchema<PrMergeState> {
    StateSchema::new()
        .with_serializable_validation()
        .with_json_schema(serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": [
                "identifier",
                "tool",
                "bash_tool",
                "command_present",
                "command_sha256",
                "operation",
                "pr_lifecycle_operation",
                "merge_verdict",
                "autopilot",
                "permission_prompt_required",
                "context_reminder_required",
                "decision"
            ],
            "properties": {
                "identifier": { "type": "string", "minLength": 1 },
                "tool": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "bash_tool": { "type": "boolean" },
                "command_present": { "type": "boolean" },
                "command_sha256": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "operation": { "type": "string", "enum": ["none", "merge", "close"] },
                "pr_lifecycle_operation": { "type": "boolean" },
                "merge_verdict": {
                    "type": "string",
                    "enum": ["not_applicable", "clear", "blocked", "unknown"]
                },
                "autopilot": { "type": "boolean" },
                "permission_prompt_required": { "type": "boolean" },
                "context_reminder_required": { "type": "boolean" },
                "decision": {
                    "type": "string",
                    "enum": [
                        "Unclassified",
                        "Allow",
                        "Ask",
                        "AllowAutopilotReminder",
                        "Deny"
                    ]
                }
            },
            "x-sentinel": {
                "graph": "pr_merge",
                "authority": "langgraph"
            }
        }))
        .with_validator(|state: &PrMergeState| {
            let Some(tool) = state
                .tool
                .as_deref()
                .map(str::trim)
                .filter(|tool| !tool.is_empty())
            else {
                return Err(StateError::ValidationFailed(
                    "LangGraph tool-authority state requires concrete tool identity".to_string(),
                ));
            };
            if state.bash_tool && tool != "Bash" {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge bash_tool requires Bash, got {tool}"
                )));
            }
            if !state.command_present {
                if state.command_sha256.is_some()
                    || state.operation != "none"
                    || state.pr_lifecycle_operation
                    || state.merge_verdict != "not_applicable"
                    || state.permission_prompt_required
                    || state.context_reminder_required
                {
                    return Err(StateError::ValidationFailed(
                        "pr_merge missing-command state cannot carry PR lifecycle facts"
                            .to_string(),
                    ));
                }
            } else if !optional_hex_digest_present(&state.command_sha256) {
                return Err(StateError::ValidationFailed(
                    "pr_merge command_sha256 must be a 64-character hex digest".to_string(),
                ));
            }
            // `gh pr merge` always gates. `gh pr close` only gates when the
            // default-off SENTINEL_PR_CLOSE_GATE opt-in is set, so a close can
            // legitimately arrive un-gated; what can never happen is a gated
            // decision on a command that is neither.
            if state.pr_lifecycle_operation
                && !matches!(state.operation.as_str(), "merge" | "close")
            {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge pr_lifecycle_operation requires a merge/close operation, got {}",
                    state.operation
                )));
            }
            if state.operation == "merge" && !state.pr_lifecycle_operation {
                return Err(StateError::ValidationFailed(
                    "pr_merge merge operations are always gated".to_string(),
                ));
            }
            // A verdict only exists where the gate actually fetched state.
            let verdict_expected_applicable = state.pr_lifecycle_operation;
            let verdict_applicable = state.merge_verdict != "not_applicable";
            if verdict_applicable != verdict_expected_applicable {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge merge_verdict must be not_applicable exactly when the gate does \
                     not apply: gated={}, merge_verdict={}",
                    state.pr_lifecycle_operation, state.merge_verdict
                )));
            }
            if !state.bash_tool
                && (state.pr_lifecycle_operation
                    || state.permission_prompt_required
                    || state.context_reminder_required)
            {
                return Err(StateError::ValidationFailed(
                    "pr_merge non-Bash state cannot authorize PR lifecycle operations".to_string(),
                ));
            }
            if state.permission_prompt_required && state.context_reminder_required {
                return Err(StateError::ValidationFailed(
                    "pr_merge cannot require both permission prompt and context reminder"
                        .to_string(),
                ));
            }
            let unknown_state = state.merge_verdict == "unknown";
            let expected_prompt = state.bash_tool
                && state.pr_lifecycle_operation
                && unknown_state
                && !state.autopilot;
            if state.permission_prompt_required != expected_prompt {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge permission_prompt_required must match Bash/operation/autopilot \
                     verdict policy: expected {expected_prompt}, got {}",
                    state.permission_prompt_required
                )));
            }
            let expected_context =
                state.bash_tool && state.pr_lifecycle_operation && unknown_state && state.autopilot;
            if state.context_reminder_required != expected_context {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge context_reminder_required must match Bash/operation/autopilot \
                     verdict policy: expected {expected_context}, got {}",
                    state.context_reminder_required
                )));
            }
            if state.decision != PrMergeDecision::Unclassified
                && state.decision != expected_decision(state)
            {
                return Err(StateError::ValidationFailed(format!(
                    "pr_merge terminal decision must match derived authorization: terminal \
                     decision {:?} does not match expected {:?}",
                    state.decision,
                    expected_decision(state)
                )));
            }
            Ok(())
        })
}

async fn classify_node(state: PrMergeState) -> Result<PrMergeState, NodeError> {
    let mut next = state;
    next.decision = expected_decision(&next);
    Ok(next)
}

pub async fn build_pr_merge_graph() -> Result<PrMergeGraph, String> {
    let checkpointer = crate::decision_graph_store::checkpointer_for_graph("pr_merge").await?;
    build_pr_merge_graph_with_checkpointer(checkpointer).await
}

#[cfg(test)]
async fn build_pr_merge_graph_with_ephemeral_sqlite() -> Result<PrMergeGraph, String> {
    build_pr_merge_graph_with_database_path(":memory:").await
}

#[cfg(test)]
async fn build_pr_merge_graph_with_database_path(db_path: &str) -> Result<PrMergeGraph, String> {
    let checkpointer = crate::decision_graph_store::checkpointer_for_config(
        crate::decision_graph_store::DecisionGraphCheckpointerConfig::Sqlite {
            database_path: db_path.to_string(),
        },
    )
    .await?;
    build_pr_merge_graph_with_checkpointer(checkpointer).await
}

async fn build_pr_merge_graph_with_checkpointer(
    checkpointer: crate::decision_graph_store::DecisionGraphCheckpointer,
) -> Result<PrMergeGraph, String> {
    let checkpointer_backend = checkpointer.backend();
    let checkpointer_scope = checkpointer.scope();
    let checkpointer_tenant_scope = checkpointer.tenant_scope_metadata_value();
    let schema = pr_merge_state_schema();
    let builder = StateGraphBuilder::<PrMergeState>::with_schema(schema.clone())
        .with_input_schema(schema.clone())
        .with_output_schema(schema.clone())
        .with_context_schema(schema)
        .set_node_defaults(crate::decision_graph_introspection::decision_node_defaults())
        .add_async_node_with_config_and_error_handler(
            CLASSIFY,
            |s: PrMergeState| async move {
                emit_decision_node_event("pr_merge", CLASSIFY, &s.identifier)?;
                classify_node(s).await
            },
            node_config(
                CLASSIFY,
                checkpointer_backend,
                checkpointer_scope,
                checkpointer_tenant_scope,
            ),
            crate::decision_graph_introspection::decision_node_error_handler,
        )
        .add_async_node_with_config_and_error_handler(
            ALLOW,
            |s: PrMergeState| async move {
                emit_decision_node_event("pr_merge", ALLOW, &s.identifier)?;
                let mut next = s;
                next.decision = PrMergeDecision::Allow;
                Ok::<_, NodeError>(next)
            },
            node_config(
                ALLOW,
                checkpointer_backend,
                checkpointer_scope,
                checkpointer_tenant_scope,
            ),
            crate::decision_graph_introspection::decision_node_error_handler,
        )
        .add_async_node_with_config_and_error_handler(
            ASK,
            |s: PrMergeState| async move {
                emit_decision_node_event("pr_merge", ASK, &s.identifier)?;
                let mut next = s;
                next.decision = PrMergeDecision::Ask;
                Ok::<_, NodeError>(next)
            },
            node_config(
                ASK,
                checkpointer_backend,
                checkpointer_scope,
                checkpointer_tenant_scope,
            ),
            crate::decision_graph_introspection::decision_node_error_handler,
        )
        .add_async_node_with_config_and_error_handler(
            ALLOW_AUTOPILOT_REMINDER,
            |s: PrMergeState| async move {
                emit_decision_node_event("pr_merge", ALLOW_AUTOPILOT_REMINDER, &s.identifier)?;
                let mut next = s;
                next.decision = PrMergeDecision::AllowAutopilotReminder;
                Ok::<_, NodeError>(next)
            },
            node_config(
                ALLOW_AUTOPILOT_REMINDER,
                checkpointer_backend,
                checkpointer_scope,
                checkpointer_tenant_scope,
            ),
            crate::decision_graph_introspection::decision_node_error_handler,
        )
        .add_async_node_with_config_and_error_handler(
            DENY,
            |s: PrMergeState| async move {
                emit_decision_node_event("pr_merge", DENY, &s.identifier)?;
                let mut next = s;
                next.decision = PrMergeDecision::Deny;
                Ok::<_, NodeError>(next)
            },
            node_config(
                DENY,
                checkpointer_backend,
                checkpointer_scope,
                checkpointer_tenant_scope,
            ),
            crate::decision_graph_introspection::decision_node_error_handler,
        )
        .add_edge(START, CLASSIFY)
        .add_conditional_edge(CLASSIFY, |s: &PrMergeState| match expected_decision(s) {
            PrMergeDecision::Allow => ALLOW.into(),
            PrMergeDecision::Ask => ASK.into(),
            PrMergeDecision::AllowAutopilotReminder => ALLOW_AUTOPILOT_REMINDER.into(),
            PrMergeDecision::Deny => DENY.into(),
            PrMergeDecision::Unclassified => ALLOW.into(),
        })
        .add_edge(ALLOW, END)
        .add_edge(ASK, END)
        .add_edge(ALLOW_AUTOPILOT_REMINDER, END)
        .add_edge(DENY, END);
    let graph = builder.build().map_err(|e| e.to_string())?;
    GraphCompiler::new()
        .with_checkpointer(checkpointer.into_saver())
        .compile_with_config(graph)
        .map_err(|e| e.to_string())
}

pub async fn run_pr_merge_decision_report(
    compiled: &PrMergeGraph,
    state: PrMergeState,
) -> Result<PrMergeGraphRun, String> {
    let thread_id = crate::decision_graph_store::run_thread_id_for_compiled(
        compiled,
        "pr_merge",
        &state.identifier,
        &state,
    )?;
    let identifier = state.identifier.clone();
    let streamed =
        stream_decision_run(compiled, &thread_id, "pr_merge", &identifier, state).await?;
    let checkpoints = checkpoint_history(compiled, &thread_id).await?;
    let write_history = write_history(compiled, &thread_id, None).await?;
    validate_decision_graph_run(
        "pr_merge",
        &thread_id,
        &streamed.state,
        &streamed.stream,
        &checkpoints,
        &write_history,
    )?;
    Ok(PrMergeGraphRun {
        state: streamed.state,
        thread_id,
        checkpoints,
        write_history,
        stream: streamed.stream,
        topology: pr_merge_graph_topology(compiled)?,
    })
}

pub fn pr_merge_graph_topology(compiled: &PrMergeGraph) -> Result<DecisionGraphTopology, String> {
    topology("pr_merge", compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_application::hooks::pr_merge_gate::{
        MergeVerdict, PrMergeDecision as AppDecision, PrMergeEvaluation,
    };

    /// Build an evaluation exactly the way the hook does, so the replica is
    /// tested against the real decision policy rather than a restatement of it.
    fn evaluation(
        operation: PrMergeOperation,
        autopilot: bool,
        verdict: MergeVerdict,
    ) -> PrMergeEvaluation {
        let gated = !matches!(operation, PrMergeOperation::None);
        let unknown = gated && matches!(verdict, MergeVerdict::Unknown);
        let permission_prompt_required = unknown && !autopilot;
        let context_reminder_required = unknown && autopilot;
        let decision = if matches!(verdict, MergeVerdict::Blocked) {
            AppDecision::Deny
        } else if permission_prompt_required {
            AppDecision::Ask
        } else if context_reminder_required {
            AppDecision::AllowAutopilotReminder
        } else {
            AppDecision::Allow
        };
        PrMergeEvaluation {
            tool: Some("Bash".to_string()),
            command: Some(match operation {
                PrMergeOperation::None => "gh pr view 1".to_string(),
                PrMergeOperation::Merge => "gh pr merge 1 --squash".to_string(),
                PrMergeOperation::Close => "gh pr close 1".to_string(),
            }),
            bash_tool: true,
            command_present: true,
            operation,
            gated,
            pr_number: Some("1".to_string()),
            verdict: if gated {
                verdict
            } else {
                MergeVerdict::NotApplicable
            },
            state_summary: None,
            blocking_reasons: Vec::new(),
            autopilot,
            permission_prompt_required,
            context_reminder_required,
            decision,
        }
    }

    fn missing_command_evaluation() -> PrMergeEvaluation {
        PrMergeEvaluation {
            tool: Some("Bash".to_string()),
            command: None,
            bash_tool: true,
            command_present: false,
            operation: PrMergeOperation::None,
            gated: false,
            pr_number: None,
            verdict: MergeVerdict::NotApplicable,
            state_summary: None,
            blocking_reasons: Vec::new(),
            autopilot: false,
            permission_prompt_required: false,
            context_reminder_required: false,
            decision: AppDecision::Allow,
        }
    }

    #[tokio::test]
    async fn graph_authorizes_merge_ask() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let state = PrMergeState::from_evaluation(
            "pr-merge-ask",
            &evaluation(PrMergeOperation::Merge, false, MergeVerdict::Unknown),
        );
        assert_eq!(state.tool.as_deref(), Some("Bash"));
        assert!(optional_hex_digest_present(&state.command_sha256));
        let run = run_pr_merge_decision_report(&graph, state).await.unwrap();
        assert_eq!(run.state.decision, PrMergeDecision::Ask);
        assert!(run
            .pr_merge_authorization()
            .unwrap()
            .unwrap()
            .checkpoint_ref()
            .contains('#'));
    }

    #[tokio::test]
    async fn graph_authorizes_missing_command_with_absent_command_hash() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let state =
            PrMergeState::from_evaluation("pr-missing-command", &missing_command_evaluation());
        assert_eq!(state.command_sha256, None);
        let run = run_pr_merge_decision_report(&graph, state).await.unwrap();
        assert_eq!(run.state.decision, PrMergeDecision::Allow);
    }

    #[tokio::test]
    async fn graph_authorizes_autopilot_context_reminder() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let state = PrMergeState::from_evaluation(
            "pr-merge-autopilot",
            &evaluation(PrMergeOperation::Merge, true, MergeVerdict::Unknown),
        );
        let run = run_pr_merge_decision_report(&graph, state).await.unwrap();
        assert_eq!(run.state.decision, PrMergeDecision::AllowAutopilotReminder);
    }

    #[tokio::test]
    async fn graph_schema_rejects_forged_prompt_skip() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let mut state = PrMergeState::from_evaluation(
            "pr-merge-forged",
            &evaluation(PrMergeOperation::Merge, false, MergeVerdict::Unknown),
        );
        state.permission_prompt_required = false;
        let err = run_pr_merge_decision_report(&graph, state)
            .await
            .unwrap_err();
        assert!(err.contains("permission_prompt_required"), "{err}");
    }

    #[tokio::test]
    async fn graph_schema_rejects_missing_tool_identity() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let mut evaluation = evaluation(PrMergeOperation::Merge, false, MergeVerdict::Unknown);
        evaluation.tool = None;
        let state = PrMergeState::from_evaluation("pr-merge-missing-tool", &evaluation);
        let err = run_pr_merge_decision_report(&graph, state)
            .await
            .unwrap_err();
        assert!(err.contains("tool identity"), "{err}");
    }

    #[tokio::test]
    async fn graph_schema_rejects_present_command_without_command_digest() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let mut state = PrMergeState::from_evaluation(
            "pr-merge-missing-digest",
            &evaluation(PrMergeOperation::Merge, false, MergeVerdict::Unknown),
        );
        state.command_sha256 = None;
        let err = run_pr_merge_decision_report(&graph, state)
            .await
            .unwrap_err();
        assert!(err.contains("command_sha256"), "{err}");
    }

    #[tokio::test]
    async fn graph_schema_rejects_missing_command_with_extra_command_digest() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let mut state = PrMergeState::from_evaluation(
            "pr-missing-command-extra",
            &missing_command_evaluation(),
        );
        state.command_sha256 = Some(command_sha256("gh pr merge 1 --squash"));
        let err = run_pr_merge_decision_report(&graph, state)
            .await
            .unwrap_err();
        assert!(err.contains("missing-command"), "{err}");
    }

    #[tokio::test]
    async fn graph_authorizes_clear_merge_as_silent_allow() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let state = PrMergeState::from_evaluation(
            "pr-merge-clear",
            &evaluation(PrMergeOperation::Merge, true, MergeVerdict::Clear),
        );
        assert_eq!(state.merge_verdict, "clear");
        let run = run_pr_merge_decision_report(&graph, state).await.unwrap();
        assert_eq!(run.state.decision, PrMergeDecision::Allow);
    }

    #[tokio::test]
    async fn graph_authorizes_blocked_merge_as_deny_even_in_autopilot() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let state = PrMergeState::from_evaluation(
            "pr-merge-blocked",
            &evaluation(PrMergeOperation::Merge, true, MergeVerdict::Blocked),
        );
        let run = run_pr_merge_decision_report(&graph, state).await.unwrap();
        assert_eq!(run.state.decision, PrMergeDecision::Deny);
        assert_eq!(pr_merge_decision_label(run.state.decision), "deny");
    }

    #[tokio::test]
    async fn graph_schema_rejects_verdict_on_an_ungated_command() {
        let graph = build_pr_merge_graph_with_ephemeral_sqlite().await.unwrap();
        let mut state = PrMergeState::from_evaluation(
            "pr-merge-ungated-verdict",
            &evaluation(PrMergeOperation::None, false, MergeVerdict::NotApplicable),
        );
        state.merge_verdict = "blocked".to_string();
        let err = run_pr_merge_decision_report(&graph, state)
            .await
            .unwrap_err();
        assert!(err.contains("merge_verdict"), "{err}");
    }
}
