//! Sidecar config (`config.json`, sidecar working directory or `--config`).
//!
//! Schema (plan §3.3) — `tiers.R0-R3` map a classification tier to an
//! upstream, `fallback` catches engine-failure/unconfigured-tier paths,
//! `router` configures the 0.1B classifier and the generation pool:
//!
//! ```json
//! {
//!   "tiers": {
//!     "R0": {"upstream": "builtin-rwkv", "model": "rwkv-0.4b.st", "tokenizer": "vocab.json"},
//!     "R3": {"upstream": "openai", "base_url": "https://api.deepseek.com/v1",
//!            "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"}
//!   },
//!   "fallback": {"upstream": "openai", "base_url": "...", "model": "...", "api_key_env": "..."},
//!   "router": {"classifier": {"model": "rwkv-0.1b.st", "vocab": "vocab.json",
//!                             "head": "router_head_0b.json", "timeout_ms": 500},
//!              "generation": {"max_loaded_models": 2}},
//!   "server": {"host": "127.0.0.1", "port": 21750},
//!   "evolution": {"data_dir": "evolution-data", "head_path": "models/rwkv/router_head_0b.json"}
//! }
//! ```
//!
//! API keys are read from the environment named by `api_key_env` at request
//! time — never stored on disk. Server listens on 127.0.0.1 only (local
//! gateway; do not expose without adding auth).

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// An upstream that serves generation for a tier.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "upstream", rename_all = "kebab-case")]
pub enum UpstreamConfig {
    /// Embedded rwkv-rsv generation (any scale 0.1B–13B, LRU pooled).
    #[serde(rename = "builtin-rwkv")]
    BuiltinRwkv {
        /// RWKV model file (`.st`).
        model: String,
        /// World tokenizer vocab JSON.
        tokenizer: String,
    },
    /// OpenAI-compatible HTTP endpoint (base_url includes `/v1`).
    Openai {
        base_url: String,
        model: String,
        /// Env var holding the API key.
        api_key_env: String,
    },
    /// Anthropic-compatible HTTP endpoint (base_url without `/v1`).
    Anthropic {
        base_url: String,
        model: String,
        api_key_env: String,
    },
}

impl UpstreamConfig {
    pub fn kind(&self) -> &'static str {
        match self {
            UpstreamConfig::BuiltinRwkv { .. } => "builtin-rwkv",
            UpstreamConfig::Openai { .. } => "openai",
            UpstreamConfig::Anthropic { .. } => "anthropic",
        }
    }
}

/// Per-tier upstream mapping. Missing tier = not served (falls back).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct TiersConfig {
    pub r0: Option<UpstreamConfig>,
    pub r1: Option<UpstreamConfig>,
    pub r2: Option<UpstreamConfig>,
    pub r3: Option<UpstreamConfig>,
}

impl TiersConfig {
    pub fn entry(&self, tier: rwkv_router::RouteClass) -> Option<&UpstreamConfig> {
        use rwkv_router::RouteClass::*;
        match tier {
            R0 => self.r0.as_ref(),
            R1 => self.r1.as_ref(),
            R2 => self.r2.as_ref(),
            R3 => self.r3.as_ref(),
        }
    }
}

/// 0.1B resident classifier paths.
#[derive(Debug, Clone, Deserialize)]
pub struct ClassifierConfig {
    pub model: String,
    pub vocab: String,
    pub head: String,
    /// Classification call timeout (ms). Default 500 (gateway budget is
    /// tighter than the library default 5000: routing must stay invisible).
    #[serde(default = "default_classify_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_classify_timeout_ms() -> u64 {
    500
}

/// Generation pool sizing.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    /// Max concurrently loaded tier models (LRU eviction).
    pub max_loaded_models: usize,
    /// Accepted for config compatibility; idle release is future work (the
    /// LRU cap already bounds VRAM).
    pub idle_release_secs: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_loaded_models: 1,
            idle_release_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct RouterSection {
    pub classifier: Option<ClassifierConfig>,
    pub generation: GenerationConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 21750,
        }
    }
}

/// Evolution loop persistence (optional; without it capture/evolve APIs are
/// disabled and routing runs route-only).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EvolutionSection {
    pub data_dir: PathBuf,
    /// Eval/replay packs directory (`eval_pack.json` required to evolve).
    pub packs_dir: Option<PathBuf>,
    pub head_path: Option<String>,
    pub capture_limit: Option<usize>,
    pub min_labeled_for_evolve: Option<usize>,
    pub auto_evolve_step: Option<usize>,
}

impl Default for EvolutionSection {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("evolution-data"),
            packs_dir: None,
            head_path: None,
            capture_limit: None,
            min_labeled_for_evolve: None,
            auto_evolve_step: None,
        }
    }
}

/// Full sidecar configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SidecarConfig {
    pub tiers: TiersConfig,
    pub fallback: Option<UpstreamConfig>,
    pub router: RouterSection,
    pub server: ServerSection,
    pub evolution: Option<EvolutionSection>,
}

impl SidecarConfig {
    /// Loads and validates `config.json` from `path`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read config '{}': {e}", path.display()))?;
        let config: SidecarConfig = serde_json::from_str(&text)
            .map_err(|e| format!("invalid config '{}': {e}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Structural validation with actionable error messages.
    pub fn validate(&self) -> Result<(), String> {
        for (label, upstream) in [
            ("fallback", self.fallback.as_ref()),
            ("R0", self.tiers.r0.as_ref()),
            ("R1", self.tiers.r1.as_ref()),
            ("R2", self.tiers.r2.as_ref()),
            ("R3", self.tiers.r3.as_ref()),
        ] {
            let Some(u) = upstream else { continue };
            match u {
                UpstreamConfig::BuiltinRwkv { model, tokenizer } => {
                    if model.is_empty() || tokenizer.is_empty() {
                        return Err(format!(
                            "{label}: builtin-rwkv requires model and tokenizer"
                        ));
                    }
                }
                UpstreamConfig::Openai {
                    base_url,
                    model,
                    api_key_env,
                }
                | UpstreamConfig::Anthropic {
                    base_url,
                    model,
                    api_key_env,
                } => {
                    if base_url.is_empty() || model.is_empty() || api_key_env.is_empty() {
                        return Err(format!(
                            "{label}: {} upstream requires base_url, model and api_key_env",
                            u.kind()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plan_example_config() {
        let json = r#"{
            "tiers": {
                "R0": {"upstream": "builtin-rwkv", "model": "rwkv-0.4b.st", "tokenizer": "vocab.json"},
                "R1": {"upstream": "builtin-rwkv", "model": "rwkv-0.4b.st", "tokenizer": "vocab.json"},
                "R3": {"upstream": "openai", "base_url": "https://api.deepseek.com/v1",
                        "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"}
            },
            "fallback": {"upstream": "anthropic", "base_url": "https://api.anthropic.com",
                          "model": "claude-sonnet-4-5", "api_key_env": "ANTHROPIC_API_KEY"},
            "router": {
                "classifier": {"model": "rwkv-0.1b.st", "vocab": "vocab.json", "head": "head.json"},
                "generation": {"max_loaded_models": 2, "idle_release_secs": 300}
            },
            "server": {"port": 21750}
        }"#;
        let config: SidecarConfig = serde_json::from_str(json).unwrap();
        config.validate().unwrap();
        assert_eq!(config.server.port, 21750);
        assert_eq!(config.tiers.r0.as_ref().unwrap().kind(), "builtin-rwkv");
        assert!(config.tiers.r2.is_none());
        assert_eq!(config.router.generation.max_loaded_models, 2);
        assert_eq!(config.router.classifier.as_ref().unwrap().timeout_ms, 500);
    }

    #[test]
    fn defaults_fill_missing_sections() {
        let config: SidecarConfig = serde_json::from_str("{}").unwrap();
        config.validate().unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 21750);
        assert_eq!(config.router.generation.max_loaded_models, 1);
        assert!(config.evolution.is_none());
    }

    #[test]
    fn rejects_builtin_entry_without_tokenizer() {
        // Internally-tagged enums fail deserialization on missing fields,
        // so malformed entries are caught at parse time (validate() covers
        // explicit empty strings).
        let json = r#"{"tiers": {"R1": {"upstream": "builtin-rwkv", "model": "m.st"}}}"#;
        assert!(serde_json::from_str::<SidecarConfig>(json).is_err());
        let json =
            r#"{"tiers": {"R1": {"upstream": "builtin-rwkv", "model": "m.st", "tokenizer": ""}}}"#;
        let config: SidecarConfig = serde_json::from_str(json).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.contains("tokenizer"), "unexpected: {err}");
    }

    #[test]
    fn rejects_openai_entry_without_api_key_env() {
        let json = r#"{"tiers": {"R3": {"upstream": "openai", "base_url": "https://x/v1", "model": "m"}}}"#;
        assert!(serde_json::from_str::<SidecarConfig>(json).is_err());
        let json = r#"{"tiers": {"R3": {"upstream": "openai", "base_url": "https://x/v1", "model": "m", "api_key_env": ""}}}"#;
        let config: SidecarConfig = serde_json::from_str(json).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.contains("api_key_env"), "unexpected: {err}");
    }
}
