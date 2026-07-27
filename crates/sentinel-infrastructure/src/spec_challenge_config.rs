//! A13 Phase 5 — Enforcement-mode config loader.
//!
//! Loads operator [`A13EnforcementMode`] choice + Catastrophic
//! axis threshold from a shipped TOML baseline merged with an
//! optional operator override at
//! `~/.claude/sentinel/config/spec-challenge.toml`.
//!
//! Mirrors [`crate::ba_config`]: hand-rolled mode-string parsing
//! (so the application-layer `A13EnforcementMode` enum stays
//! decoupled from infrastructure config concerns), explicit
//! error on unknown mode strings (operator sees the typo at
//! startup, not at production runtime).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use sentinel_application::hooks::spec_challenge_gate;
use serde::Deserialize;

/// Shipped baseline. The intended enterprise baseline remains
/// `DefaultBlocking`, but the embedded shipped mode is temporarily
/// `ObserveOnly`: no tool can emit `extra.spec_challenge` yet
/// (`mcp__sentinel__emit_challenge` does not exist), so a blocking
/// default would deny unsatisfiably. Restore `DefaultBlocking` in the
/// shipped TOML once the emit path lands.
/// `catastrophic_axis_threshold` defaults to `0.7`.
pub const SHIPPED_SPEC_CHALLENGE_DEFAULTS: &str =
    include_str!("../../../config/spec-challenge-defaults.toml");

/// Operator-facing config for the A13 spec-challenge gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecChallengeConfig {
    pub mode: spec_challenge_gate::A13EnforcementMode,
    pub catastrophic_axis_threshold: f32,
}

impl SpecChallengeConfig {
    /// Production default: class-graded blocking, threshold `0.7`.
    #[must_use]
    pub const fn default_blocking() -> Self {
        Self {
            mode: spec_challenge_gate::A13EnforcementMode::DefaultBlocking,
            catastrophic_axis_threshold: spec_challenge_gate::DEFAULT_CATASTROPHIC_AXIS_THRESHOLD,
        }
    }

    /// Explicit diagnostics-only mode. Not used as a production fallback.
    #[must_use]
    pub const fn observe_only() -> Self {
        Self {
            mode: spec_challenge_gate::A13EnforcementMode::ObserveOnly,
            catastrophic_axis_threshold: spec_challenge_gate::DEFAULT_CATASTROPHIC_AXIS_THRESHOLD,
        }
    }

    /// Parse from a TOML string. Missing keys fall back to
    /// [`Self::default_blocking`] defaults; unknown mode strings
    /// surface as `Err`.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let toml_doc: SpecChallengeToml =
            toml::from_str(s).context("failed to parse spec-challenge TOML")?;
        let mode = toml_doc.spec_challenge_gate.as_ref().map_or(
            Ok(spec_challenge_gate::A13EnforcementMode::DefaultBlocking),
            |s| parse_mode(&s.mode),
        )?;
        let threshold = toml_doc
            .spec_challenge_gate
            .as_ref()
            .and_then(|s| s.catastrophic_axis_threshold)
            .unwrap_or(spec_challenge_gate::DEFAULT_CATASTROPHIC_AXIS_THRESHOLD);
        if !(0.0..=1.0).contains(&threshold) {
            return Err(anyhow!(
                "catastrophic_axis_threshold must be in [0.0, 1.0]; got {threshold}"
            ));
        }
        Ok(Self {
            mode,
            catastrophic_axis_threshold: threshold,
        })
    }

    /// Load the shipped defaults (compile-time embedded TOML).
    pub fn shipped() -> Result<Self> {
        Self::from_toml_str(SHIPPED_SPEC_CHALLENGE_DEFAULTS)
            .context("failed to parse shipped spec-challenge-defaults.toml")
    }

    /// Load shipped defaults + apply operator overrides from
    /// `path` if the file exists. Missing path → shipped only.
    pub fn with_shipped_and_overrides(overrides_path: Option<&Path>) -> Result<Self> {
        let mut config = Self::shipped()?;
        if let Some(path) = overrides_path {
            if !path.exists() {
                tracing::debug!(
                    path = %path.display(),
                    "spec-challenge override not present; using shipped defaults"
                );
                return Ok(config);
            }
            let bytes = std::fs::read_to_string(path).with_context(|| {
                format!(
                    "failed to read spec-challenge override at {}",
                    path.display()
                )
            })?;
            config = Self::from_toml_str(&bytes).with_context(|| {
                format!(
                    "failed to parse spec-challenge override at {}",
                    path.display()
                )
            })?;
            tracing::info!(
                path = %path.display(),
                mode = ?config.mode,
                threshold = config.catastrophic_axis_threshold,
                "loaded spec-challenge operator override"
            );
        }
        Ok(config)
    }
}

fn parse_mode(s: &str) -> Result<spec_challenge_gate::A13EnforcementMode> {
    match s {
        "ObserveOnly" => Ok(spec_challenge_gate::A13EnforcementMode::ObserveOnly),
        "DefaultBlocking" => Ok(spec_challenge_gate::A13EnforcementMode::DefaultBlocking),
        "StrictBlocking" => Ok(spec_challenge_gate::A13EnforcementMode::StrictBlocking),
        other => Err(anyhow!(
            "unknown spec_challenge_gate mode {other:?}; \
             expected one of: ObserveOnly, DefaultBlocking, StrictBlocking"
        )),
    }
}

#[derive(Debug, Deserialize)]
struct SpecChallengeToml {
    spec_challenge_gate: Option<SpecChallengeSection>,
}

#[derive(Debug, Deserialize)]
struct SpecChallengeSection {
    mode: String,
    catastrophic_axis_threshold: Option<f32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sentinel_application::hooks::spec_challenge_gate::A13EnforcementMode;
    use tempfile::TempDir;

    #[test]
    fn shipped_defaults_parse_cleanly() {
        let config = SpecChallengeConfig::shipped().unwrap();
        // Shipped default is temporarily ObserveOnly: DefaultBlocking is
        // unsatisfiable until `mcp__sentinel__emit_challenge` exists, so the
        // gate could never be cleared by any agent. Flip back with the config.
        assert_eq!(config.mode, A13EnforcementMode::ObserveOnly);
        assert!((config.catastrophic_axis_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn shipped_matches_default_blocking_constant_except_mode() {
        // The compiled-in `default_blocking()` constant still describes the
        // intended enterprise baseline; only the shipped MODE is temporarily
        // relaxed (see config/spec-challenge-defaults.toml). Everything else —
        // thresholds, axis config — must stay in lockstep, so this still
        // catches drift between the constant and the shipped file.
        let shipped = SpecChallengeConfig::shipped().unwrap();
        let const_default = SpecChallengeConfig::default_blocking();
        assert_eq!(shipped.mode, A13EnforcementMode::ObserveOnly);
        assert_eq!(const_default.mode, A13EnforcementMode::DefaultBlocking);
        assert_eq!(
            SpecChallengeConfig {
                mode: const_default.mode,
                ..shipped
            },
            const_default,
            "shipped config drifted from default_blocking() beyond the mode"
        );
    }

    #[test]
    fn explicit_observe_only_parses() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "ObserveOnly"
        "#;
        let config = SpecChallengeConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::ObserveOnly);
    }

    #[test]
    fn explicit_default_blocking_parses() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "DefaultBlocking"
        "#;
        let config = SpecChallengeConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::DefaultBlocking);
    }

    #[test]
    fn explicit_strict_blocking_parses() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "StrictBlocking"
        "#;
        let config = SpecChallengeConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::StrictBlocking);
    }

    #[test]
    fn unknown_mode_string_errors() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "TotallyMadeUp"
        "#;
        let err = SpecChallengeConfig::from_toml_str(toml).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("TotallyMadeUp"));
        assert!(msg.contains("ObserveOnly"));
    }

    #[test]
    fn missing_section_falls_back_to_default_blocking() {
        let toml = "";
        let config = SpecChallengeConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::DefaultBlocking);
        assert!((config.catastrophic_axis_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn threshold_override_parses() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "DefaultBlocking"
            catastrophic_axis_threshold = 0.85
        "#;
        let config = SpecChallengeConfig::from_toml_str(toml).unwrap();
        assert!((config.catastrophic_axis_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn threshold_out_of_range_errors() {
        let toml = r#"
            [spec_challenge_gate]
            mode = "DefaultBlocking"
            catastrophic_axis_threshold = 1.5
        "#;
        let err = SpecChallengeConfig::from_toml_str(toml).unwrap_err();
        assert!(format!("{err:#}").contains("must be in"));

        let toml = r#"
            [spec_challenge_gate]
            mode = "DefaultBlocking"
            catastrophic_axis_threshold = -0.1
        "#;
        let err = SpecChallengeConfig::from_toml_str(toml).unwrap_err();
        assert!(format!("{err:#}").contains("must be in"));
    }

    #[test]
    fn malformed_toml_errors() {
        let err = SpecChallengeConfig::from_toml_str("this is not toml [[[").unwrap_err();
        assert!(format!("{err:#}").contains("parse"));
    }

    #[test]
    fn with_shipped_and_overrides_none_uses_shipped() {
        let config = SpecChallengeConfig::with_shipped_and_overrides(None).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::ObserveOnly);
    }

    #[test]
    fn with_shipped_and_overrides_missing_file_uses_shipped() {
        let path = std::path::Path::new("/nonexistent/path/spec-challenge.toml");
        let config = SpecChallengeConfig::with_shipped_and_overrides(Some(path)).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::ObserveOnly);
    }

    #[test]
    fn with_shipped_and_overrides_applies_operator_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("spec-challenge.toml");
        std::fs::write(
            &path,
            r#"
                [spec_challenge_gate]
                mode = "StrictBlocking"
                catastrophic_axis_threshold = 0.9
            "#,
        )
        .unwrap();
        let config = SpecChallengeConfig::with_shipped_and_overrides(Some(&path)).unwrap();
        assert_eq!(config.mode, A13EnforcementMode::StrictBlocking);
        assert!((config.catastrophic_axis_threshold - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn config_is_send_sync_copy() {
        fn assert_send_sync_copy<T: Send + Sync + Copy>() {}
        assert_send_sync_copy::<SpecChallengeConfig>();
    }
}
