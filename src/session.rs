//! RouterSession: one assembled routing stack (router + optional builtin
//! engines + optional evolution loop) behind a single object.
//!
//! This is the unit the FFI handle ([`crate::ffi`]) and the language
//! bindings wrap. Pure Rust — the pyo3/napi layers expose it directly
//! instead of going through the C ABI.
//!
//! Capture wiring: [`RouterSession::route`] uses
//! [`SmartRouter::route_with_capture`] and feeds the bundle to the
//! configured [`Evolution`] — the standalone equivalent of the client's
//! desktop-side capture wiring.

use std::sync::Arc;

use crate::config::RouterConfig;
#[cfg(feature = "rwkv")]
use crate::engine::ClassifyEngine as _;
use crate::evolution::{CaptureItem, CaptureStats, Evolution, EvolutionConfig, EvolveResult};
use crate::router::{RouteCapture, SmartRouter};
#[cfg(feature = "rwkv")]
use crate::tier::RouteClass;

#[cfg(feature = "rwkv")]
use crate::builtin::{
    ClassifyEngineConfig, GenOutput, GenParams, GenerateEngineConfig, RwkvClassifyEngine,
    RwkvGenerateEngine, TierModelPaths,
};

/// Assembled routing stack. Construct with [`RouterSession::new`], then
/// optionally attach a classifier ([`Self::load_classifier`], feature
/// `rwkv`), the evolution loop ([`Self::configure_evolution`]) and tier
/// generation ([`Self::attach_generation`], feature `rwkv`).
pub struct RouterSession {
    /// Routing parameters (tier models / thresholds / fallback reference).
    pub config: RouterConfig,
    router: Arc<SmartRouter>,
    evolution: Option<Arc<Evolution>>,
    /// Generation pool config assembled by [`Self::attach_generation`]
    /// (feature `rwkv`); the engine is (re)built eagerly — models load
    /// lazily on first generate, so attach-before-generate is cheap.
    #[cfg(feature = "rwkv")]
    gen_paths: TierModelPaths,
    #[cfg(feature = "rwkv")]
    gen_vocab: Option<String>,
    #[cfg(feature = "rwkv")]
    generate: Option<Arc<RwkvGenerateEngine>>,
    /// Keep the classifier handle alive: the router holds an `Arc<dyn
    /// ClassifyEngine>` trait object, but the concrete engine also carries
    /// the head used by [`Self::classify_head_path`] (evolution deploy
    /// target) and enables reload-on-deploy.
    #[cfg(feature = "rwkv")]
    classify: Option<Arc<RwkvClassifyEngine>>,
    #[cfg(feature = "rwkv")]
    classify_head_path: Option<String>,
}

impl RouterSession {
    /// Creates a session with an empty router (rules + fallback only until
    /// a classifier is attached).
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            router: Arc::new(SmartRouter::new()),
            evolution: None,
            #[cfg(feature = "rwkv")]
            gen_paths: TierModelPaths::default(),
            #[cfg(feature = "rwkv")]
            gen_vocab: None,
            #[cfg(feature = "rwkv")]
            generate: None,
            #[cfg(feature = "rwkv")]
            classify: None,
            #[cfg(feature = "rwkv")]
            classify_head_path: None,
        }
    }

    /// Shared router handle (hot-swap engines / custom wiring).
    pub fn router(&self) -> &Arc<SmartRouter> {
        &self.router
    }

    // ---------------------------------------------------------------------
    // Classification (feature rwkv)
    // ---------------------------------------------------------------------

    /// Loads the built-in 0.1B classifier (model + vocab + head) and
    /// registers it on the router (and the evolution loop, if configured).
    /// `timeout_ms` bounds each classify call (0 → library default 5000).
    #[cfg(feature = "rwkv")]
    pub fn load_classifier(
        &mut self,
        model_path: &str,
        vocab_path: &str,
        head_path: &str,
        timeout_ms: u64,
    ) -> Result<(), String> {
        let engine = Arc::new(RwkvClassifyEngine::new(ClassifyEngineConfig {
            model_path: model_path.to_string(),
            vocab_path: vocab_path.to_string(),
            head_path: head_path.to_string(),
            timeout_ms: if timeout_ms == 0 { 5000 } else { timeout_ms },
            ..Default::default()
        })?);
        self.router.set_engine(engine.clone());
        if let Some(ev) = &self.evolution {
            ev.set_engine(engine.clone());
        }
        self.classify = Some(engine);
        self.classify_head_path = Some(head_path.to_string());
        Ok(())
    }

    /// Re-reads the classifier head file (evolution hot-deploy).
    #[cfg(feature = "rwkv")]
    pub fn reload_classifier_head(&self) -> Result<(), String> {
        match &self.classify {
            Some(engine) => engine.reload(),
            None => Err("no classifier attached".to_string()),
        }
    }

    /// Classifier head JSON path (evolution deploy target).
    #[cfg(feature = "rwkv")]
    pub fn classify_head_path(&self) -> Option<&str> {
        self.classify_head_path.as_deref()
    }

    // ---------------------------------------------------------------------
    // Evolution
    // ---------------------------------------------------------------------

    /// Configures the evolution loop (sample store + eval gate + head
    /// deploy). Safe to call before or after [`Self::load_classifier`].
    /// `head_path` is only a bookkeeping default here; deploy goes to the
    /// classifier's actual head path when attached.
    pub fn configure_evolution(&mut self, config: EvolutionConfig) -> Result<(), String> {
        if self.evolution.is_some() {
            return Err("evolution already configured".to_string());
        }
        let evolution = Arc::new(Evolution::new(config));
        #[cfg(feature = "rwkv")]
        if let Some(engine) = &self.classify {
            evolution.set_engine(engine.clone());
        }
        self.evolution = Some(evolution);
        Ok(())
    }

    /// The evolution loop, if configured.
    pub fn evolution(&self) -> Option<&Arc<Evolution>> {
        self.evolution.as_ref()
    }

    // ---------------------------------------------------------------------
    // Routing (+ capture wiring)
    // ---------------------------------------------------------------------

    /// Routes one request; feeds the capture bundle to the evolution loop
    /// when configured. Mirrors the client's route+capture wiring.
    pub fn route(
        &self,
        session_id: &str,
        user_input: &str,
        summary: Option<&str>,
        turn_index: usize,
    ) -> crate::postprocess::RoutingDecision {
        let (decision, capture) = self.router.route_with_capture(
            session_id,
            user_input,
            summary,
            turn_index,
            &self.config,
        );
        if let (Some(evolution), Some(capture)) = (&self.evolution, capture) {
            let RouteCapture {
                input,
                probs,
                hidden,
                prev_tier,
                num_embd,
            } = capture;
            evolution.capture_route_sample(&input, &hidden, prev_tier, &probs, num_embd);
        }
        decision
    }

    /// Stateless preview (settings-page test entry semantics).
    pub fn route_preview(
        &self,
        user_input: &str,
        summary: Option<&str>,
    ) -> crate::postprocess::RoutingDecision {
        self.router.route_preview(user_input, summary, &self.config)
    }

    // ---------------------------------------------------------------------
    // Capture/evolve passthrough (FFI/binding surface)
    // ---------------------------------------------------------------------

    pub fn capture_stats(&self) -> Result<CaptureStats, String> {
        Ok(self
            .evolution
            .as_ref()
            .ok_or("evolution not configured")?
            .capture_stats())
    }

    pub fn capture_list(&self, offset: usize, limit: usize) -> Result<Vec<CaptureItem>, String> {
        Ok(self
            .evolution
            .as_ref()
            .ok_or("evolution not configured")?
            .capture_list(offset, limit))
    }

    /// Labels (or clears with `None`) the sample at `idx`.
    pub fn capture_label(&self, idx: usize, label: Option<u8>) -> Result<(), String> {
        self.evolution
            .as_ref()
            .ok_or("evolution not configured")?
            .capture_label(idx, label)
    }

    /// Runs one evolution cycle (blocking: fine-tune + eval gate + deploy).
    pub fn evolve(&self) -> Result<EvolveResult, String> {
        Ok(self
            .evolution
            .as_ref()
            .ok_or("evolution not configured")?
            .evolve())
    }

    // ---------------------------------------------------------------------
    // Generation (feature rwkv)
    // ---------------------------------------------------------------------

    /// Attaches (or rebuilds) the tier generation pool with one tier's model
    /// path. Call before the first [`Self::generate`] — later rebuilds drop
    /// any loaded models (they reload lazily). `max_loaded` bounds the LRU
    /// pool (concurrently resident tier models).
    #[cfg(feature = "rwkv")]
    pub fn attach_generation(
        &mut self,
        tier: RouteClass,
        vocab_path: &str,
        model_path: &str,
        max_loaded: usize,
    ) -> Result<(), String> {
        if vocab_path.is_empty() || model_path.is_empty() {
            return Err("attach_generation: vocab_path and model_path are required".to_string());
        }
        let mut paths = self.gen_paths.clone();
        match tier {
            RouteClass::R0 => paths.r0 = Some(model_path.to_string()),
            RouteClass::R1 => paths.r1 = Some(model_path.to_string()),
            RouteClass::R2 => paths.r2 = Some(model_path.to_string()),
            RouteClass::R3 => paths.r3 = Some(model_path.to_string()),
        }
        let engine = Arc::new(RwkvGenerateEngine::new(GenerateEngineConfig {
            vocab_path: vocab_path.to_string(),
            tier_models: paths.clone(),
            max_loaded,
            ..Default::default()
        })?);
        self.gen_paths = paths;
        self.gen_vocab = Some(vocab_path.to_string());
        self.generate = Some(engine);
        Ok(())
    }

    /// Generates from the tier's model (loads it on first use; LRU-pooled).
    #[cfg(feature = "rwkv")]
    pub fn generate(
        &self,
        tier: RouteClass,
        prompt: &str,
        params: &GenParams,
    ) -> Result<GenOutput, String> {
        self.generate
            .as_ref()
            .ok_or_else(|| {
                "generation not attached (call attach_generation for this tier first)".to_string()
            })?
            .generate(tier, prompt, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_without_engine_falls_back() {
        let session = RouterSession::new(RouterConfig::default());
        let d = session.route("s1", "help me debug this traceback", None, 0);
        assert_eq!(d.source, crate::postprocess::DecisionSource::Fallback);
    }

    #[test]
    fn trivial_ack_routes_r0_without_evolution() {
        let session = RouterSession::new(RouterConfig::default());
        let d = session.route("s1", "ok", None, 0);
        assert_eq!(d.route, RouteClass::R0);
        assert!(session.evolution().is_none());
    }

    #[test]
    fn capture_and_evolution_surfaces_error_when_unconfigured() {
        let session = RouterSession::new(RouterConfig::default());
        assert!(session.capture_stats().is_err());
        assert!(session.evolve().is_err());
        assert!(session.capture_label(0, Some(1)).is_err());
    }

    #[cfg(feature = "rwkv")]
    #[test]
    fn generation_requires_attach() {
        let session = RouterSession::new(RouterConfig::default());
        let err = session
            .generate(RouteClass::R1, "hello", &GenParams::default())
            .unwrap_err();
        assert!(err.contains("attach_generation"), "unexpected: {err}");
    }

    #[cfg(feature = "rwkv")]
    #[test]
    fn attach_generation_validates_paths() {
        let mut session = RouterSession::new(RouterConfig::default());
        let err = session
            .attach_generation(RouteClass::R1, "", "model.st", 1)
            .unwrap_err();
        assert!(err.contains("required"), "unexpected: {err}");
    }
}
