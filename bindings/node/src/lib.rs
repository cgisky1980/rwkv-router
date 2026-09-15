//! Node.js bindings for RWKV-Router (napi-rs; npm package `rwkv-router`).
//!
//! Thin surface over [`rwkv_router::RouterSession`]: every structured result
//! (decision / stats / evolve report / generation output) is serialized via
//! serde and returned as a plain JS object — one conversion path, zero
//! per-type shims, automatically in sync with the Rust API.
//!
//! Quickstart:
//!
//! ```js
//! import { RouterSession } from 'rwkv-router'
//!
//! const s = new RouterSession()
//! const d = s.route('帮我写一首诗')      // -> { route: 'R2', confidence: .., ... }
//! s.loadClassifier('rwkv-0.1b.st', 'vocab.json', 'router_head_0b.json')
//! s.attachGeneration('R0', 'vocab.json', 'rwkv-0.1b.st')
//! const out = s.generate('R0', '你好')   // -> { text: .., output_tokens: .., ... }
//! ```
//!
//! All methods are synchronous (v0.1): they block the calling JS thread for
//! the duration of the call — fine for routing (sub-ms) and one-off CLI
//! scripts; wrap long `evolve()` calls in a worker if needed.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use rwkv_router::{EvolutionConfig, GenParams, RouteClass, RouterSession as CoreSession};

fn parse_tier(tier: &str) -> Result<RouteClass> {
    RouteClass::parse_from_str(tier).ok_or_else(|| {
        Error::from_reason(format!("invalid tier '{tier}' (expected \"R0\"..\"R3\")"))
    })
}

fn backend_err(e: String) -> Error {
    Error::from_reason(e)
}

impl Default for RouterSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-evolving smart router session (R0-R3 tiering + optional RWKV
/// classifier/generation pool + optional evolution loop).
///
/// Engines load from local files; build failures throw. `route()` never
/// throws for routing itself — without a classifier it falls back to rules.
#[napi]
pub struct RouterSession {
    inner: CoreSession,
}

#[napi]
impl RouterSession {
    /// Creates a session with default routing parameters.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: CoreSession::new(rwkv_router::RouterConfig::default()),
        }
    }

    /// One routing decision. Sticky-tier context is keyed by `sessionId`
    /// (`turnIndex` counts turns inside that session). Feeds the evolution
    /// capture store when evolution is configured.
    #[napi]
    pub fn route(
        &self,
        input: String,
        summary: Option<String>,
        session_id: Option<String>,
        turn_index: Option<u32>,
    ) -> Result<serde_json::Value> {
        let decision = self.inner.route(
            session_id.as_deref().unwrap_or("node"),
            &input,
            summary.as_deref(),
            turn_index.unwrap_or(0) as usize,
        );
        serde_json::to_value(&decision).map_err(|e| backend_err(e.to_string()))
    }

    /// Stateless decision (no sticky table, no capture) — testing entry.
    #[napi]
    pub fn route_preview(
        &self,
        input: String,
        summary: Option<String>,
    ) -> Result<serde_json::Value> {
        let decision = self.inner.route_preview(&input, summary.as_deref());
        serde_json::to_value(&decision).map_err(|e| backend_err(e.to_string()))
    }

    /// Attaches the built-in RWKV classifier (resident 0.1B + trained MLP
    /// head). `timeoutMs` bounds each classify call.
    #[napi]
    pub fn load_classifier(
        &mut self,
        model: String,
        vocab: String,
        head: String,
        timeout_ms: Option<u32>,
    ) -> Result<()> {
        self.inner
            .load_classifier(
                &model,
                &vocab,
                &head,
                timeout_ms.map(|v| v as u64).unwrap_or(5000),
            )
            .map_err(backend_err)
    }

    /// Maps a tier ("R0".."R3") to a local RWKV model for embedded
    /// generation. Models load lazily on first `generate()`; `maxLoaded`
    /// bounds the LRU pool of concurrently resident models.
    #[napi]
    pub fn attach_generation(
        &mut self,
        tier: String,
        vocab: String,
        model: String,
        max_loaded: Option<u32>,
    ) -> Result<()> {
        let tier = parse_tier(&tier)?;
        self.inner
            .attach_generation(tier, &vocab, &model, max_loaded.unwrap_or(1) as usize)
            .map_err(backend_err)
    }

    /// Generates with the tier's attached model. Unspecified sampling
    /// parameters use engine defaults (512 tokens, temp 1.0, top_p 0.9).
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        tier: String,
        prompt: String,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<u32>,
        presence_penalty: Option<f64>,
        frequency_penalty: Option<f64>,
        stop: Option<Vec<String>>,
    ) -> Result<serde_json::Value> {
        let tier = parse_tier(&tier)?;
        let mut params = GenParams::default();
        if let Some(v) = max_tokens {
            params.max_tokens = v as usize;
        }
        if let Some(v) = temperature {
            params.temperature = v as f32;
        }
        if let Some(v) = top_p {
            params.top_p = v as f32;
        }
        if let Some(v) = top_k {
            params.top_k = v as usize;
        }
        if let Some(v) = presence_penalty {
            params.presence_penalty = v as f32;
        }
        if let Some(v) = frequency_penalty {
            params.frequency_penalty = v as f32;
        }
        if let Some(v) = stop {
            params.stop = v;
        }
        let output = self
            .inner
            .generate(tier, &prompt, &params)
            .map_err(backend_err)?;
        serde_json::to_value(&output).map_err(|e| backend_err(e.to_string()))
    }

    /// Enables the self-evolution loop: capture -> label -> AdamW fine-tune
    /// -> eval-pack gate -> head hot-reload. `packsDir` (holding
    /// `eval_pack.json`) is required to actually run `evolve()`.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn configure_evolution(
        &mut self,
        data_dir: String,
        head_path: Option<String>,
        packs_dir: Option<String>,
        capture_limit: Option<u32>,
        min_labeled_for_evolve: Option<u32>,
        auto_evolve_step: Option<u32>,
    ) -> Result<()> {
        let mut config = EvolutionConfig {
            data_dir: data_dir.into(),
            head_path: head_path.map(Into::into).unwrap_or_default(),
            packs_dir: packs_dir.map(Into::into),
            ..Default::default()
        };
        if let Some(v) = capture_limit {
            config.capture_limit = v as usize;
        }
        if let Some(v) = min_labeled_for_evolve {
            config.min_labeled_for_evolve = v as usize;
        }
        if let Some(v) = auto_evolve_step {
            config.auto_evolve_step = v as usize;
        }
        self.inner.configure_evolution(config).map_err(backend_err)
    }

    /// Capture-store statistics (total / labelled / per-tier counts).
    #[napi]
    pub fn capture_stats(&self) -> Result<serde_json::Value> {
        let stats = self.inner.capture_stats().map_err(backend_err)?;
        serde_json::to_value(&stats).map_err(|e| backend_err(e.to_string()))
    }

    /// Captured samples (text + current label + probs) for labeling flows.
    #[napi]
    pub fn capture_list(
        &self,
        offset: Option<u32>,
        limit: Option<u32>,
    ) -> Result<serde_json::Value> {
        let items = self
            .inner
            .capture_list(offset.unwrap_or(0) as usize, limit.unwrap_or(50) as usize)
            .map_err(backend_err)?;
        serde_json::to_value(&items).map_err(|e| backend_err(e.to_string()))
    }

    /// Labels the sample at `idx` with `tier` ("R0".."R3"); `tier=null`
    /// clears the label.
    #[napi]
    pub fn capture_label(&self, idx: u32, tier: Option<String>) -> Result<()> {
        let label = match tier.as_deref() {
            Some(t) => Some(parse_tier(t)?.index() as u8),
            None => None,
        };
        self.inner
            .capture_label(idx as usize, label)
            .map_err(backend_err)
    }

    /// Runs one evolution cycle (blocking): fine-tune on labelled samples,
    /// gate against the eval pack (deploys only if accuracy does not
    /// regress), backup + hot-reload the head.
    #[napi]
    pub fn evolve(&self) -> Result<serde_json::Value> {
        let result = self.inner.evolve().map_err(backend_err)?;
        serde_json::to_value(&result).map_err(|e| backend_err(e.to_string()))
    }
}
