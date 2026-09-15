//! Smart routing configuration (extracted 1:1 from Ai00-X
//! `client/src/crates/core/src/service/config/types.rs`; field names and
//! serde semantics are identical so existing config JSON keeps parsing).

use crate::tier::RouteClass;
use serde::{Deserialize, Serialize};

/// Smart routing configuration for the model-selection router.
///
/// The router classifies each user request (auto model mode) into a
/// complexity tier (R0-R3) and dispatches it to the configured model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    /// Master switch. Only takes effect when the session model is "auto".
    pub enabled: bool,
    /// Model reference per route tier (model id, "primary", "fast" or "rwkv-local").
    pub tier_models: RouterTierModels,
    /// Fallback model reference when classification fails or the engine is unavailable.
    pub fallback: String,
    /// Under-routing safety threshold: if argmax is R0/R1 and P(R2)+P(R3)
    /// exceeds this value, upgrade to R2 (prefer over-routing to under-routing).
    pub safety_threshold: f32,
    /// Sticky tier: short messages never route below the previous turn's tier.
    pub sticky_enabled: bool,
    /// Classification inference timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tier_models: RouterTierModels::default(),
            fallback: "primary".to_string(),
            safety_threshold: 0.45,
            sticky_enabled: true,
            timeout_ms: 3000,
        }
    }
}

/// Model references for each route tier of the smart router.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterTierModels {
    /// R0 trivial chat — local RWKV model by default.
    pub r0: String,
    /// R1 simple task — local RWKV model by default.
    pub r1: String,
    /// R2 complex task — mid-tier ("fast") model by default.
    pub r2: String,
    /// R3 high-stakes task — flagship ("primary") model by default.
    pub r3: String,
}

impl Default for RouterTierModels {
    fn default() -> Self {
        Self {
            r0: "rwkv-local".to_string(),
            r1: "rwkv-local".to_string(),
            r2: "fast".to_string(),
            r3: "primary".to_string(),
        }
    }
}

impl RouterTierModels {
    /// Returns the model reference for the given route tier.
    pub fn model_for(&self, tier: &RouteClass) -> &str {
        match tier {
            RouteClass::R0 => &self.r0,
            RouteClass::R1 => &self.r1,
            RouteClass::R2 => &self.r2,
            RouteClass::R3 => &self.r3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrip_json() {
        let cfg = RouterConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.safety_threshold, 0.45);
        assert!(parsed.sticky_enabled);
        assert_eq!(parsed.timeout_ms, 3000);
        assert_eq!(parsed.tier_models.model_for(&RouteClass::R0), "rwkv-local");
        assert_eq!(parsed.tier_models.model_for(&RouteClass::R3), "primary");
    }

    #[test]
    fn partial_json_fills_defaults() {
        let parsed: RouterConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.fallback, "primary");
        assert_eq!(parsed.timeout_ms, 3000);
    }
}
