//! SmartRouter orchestration: rules + classification + post-processing.
//!
//! Pipeline per request:
//! 1. trivial-ack short circuit (pure rules, zero cost) → R0;
//! 2. engine classification (state embedding + trained MLP head), with the
//!    session's rolling summary prepended for context awareness;
//! 3. post-processing rule stack (safety upgrade / sticky tier).
//!
//! Per-session sticky state lives here (LRU-bounded). Classification
//! failures degrade gracefully to a fallback decision without touching the
//! sticky table, so a transient engine outage cannot drag a session's tier
//! floor down.
//!
//! This is the standalone (self-contained) SmartRouter: the engine is
//! injected at construction (or hot-swapped later). The Ai00-X client runs
//! its own embedded variant with the same semantics — the orchestration
//! behaviors are pinned by the tests in this file.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::config::RouterConfig;
use crate::engine::ClassifyEngine;
use crate::postprocess::{
    fallback_decision, postprocess, trivial_ack_decision, DecisionSource, RoutingDecision,
};
use crate::rules::{is_short_message, is_trivial_ack};
use crate::tier::RouteClass;

/// Upper bound on tracked sticky-tier sessions (oldest evicted when exceeded).
const STICKY_TABLE_LIMIT: usize = 1024;

/// Raw classification details from one route call — the capture bundle hosts
/// feed to [`crate::Evolution::capture_route_sample`] to build the training
/// sample store. `None` for trivial-ack / fallback paths (no model output).
#[derive(Debug, Clone)]
pub struct RouteCapture {
    /// Built classify input (summary + request) — the text the head scored.
    pub input: String,
    /// Raw model probabilities (pre-post-processing), R0-R3.
    pub probs: [f32; 4],
    /// Mean-pooled hidden state the head consumed.
    pub hidden: Vec<f32>,
    /// Previous turn's sticky tier fed to the head (v4 one-hot feature).
    pub prev_tier: Option<u8>,
    /// Engine embedding dimension.
    pub num_embd: usize,
}

/// Internal classification result: the built input text, raw pre-postprocess
/// decision and the hidden state the head consumed (capture payload).
struct RawClassification {
    input: String,
    hidden: Vec<f32>,
    decision: RoutingDecision,
}

/// Smart router: per-session sticky tier state + classification pipeline.
pub struct SmartRouter {
    /// Classification engine (settable after construction for hot swap).
    engine: RwLock<Option<Arc<dyn ClassifyEngine>>>,
    /// session_id -> tier of the previous routed turn.
    sticky_tiers: Mutex<HashMap<String, RouteClass>>,
    /// Insertion order for LRU eviction (oldest first).
    sticky_order: Mutex<VecDeque<String>>,
}

impl Default for SmartRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl SmartRouter {
    /// Creates a router without an engine; classification degrades to the
    /// fallback decision until [`set_engine`](Self::set_engine) is called.
    pub fn new() -> Self {
        Self {
            engine: RwLock::new(None),
            sticky_tiers: Mutex::new(HashMap::new()),
            sticky_order: Mutex::new(VecDeque::new()),
        }
    }

    /// Creates a router with the given classification engine.
    pub fn with_engine(engine: Arc<dyn ClassifyEngine>) -> Self {
        Self {
            engine: RwLock::new(Some(engine)),
            sticky_tiers: Mutex::new(HashMap::new()),
            sticky_order: Mutex::new(VecDeque::new()),
        }
    }

    /// Hot-swaps the classification engine (e.g. after evolution re-deploys
    /// a fine-tuned head, or the host finishes lazy-loading the model).
    pub fn set_engine(&self, engine: Arc<dyn ClassifyEngine>) {
        *self.engine.write().unwrap_or_else(|e| e.into_inner()) = Some(engine);
    }

    /// Classifies the request and produces a routing decision.
    ///
    /// 1. Trivial-ack short circuit. Does NOT touch the sticky table:
    ///    an acknowledgment mid-task must not reset the session's tier
    ///    floor (e.g. "ok" inside an R3 session, followed by "继续",
    ///    should stay sticky-lifted to R3).
    /// 2. Model classification, with the session's rolling summary prepended
    ///    for context awareness. `capture=true` flows to the engine so real
    ///    routing decisions can feed the evolution sample store.
    /// 3. Post-processing rule stack (safety upgrade / sticky tier).
    pub fn route(
        &self,
        session_id: &str,
        user_input: &str,
        summary: Option<&str>,
        turn_index: usize,
        config: &RouterConfig,
    ) -> RoutingDecision {
        self.route_with_capture(session_id, user_input, summary, turn_index, config)
            .0
    }

    /// [`route`] plus the raw [`RouteCapture`] bundle: hosts wiring the
    /// evolution sample store use this to capture (text, hidden, probs)
    /// without re-running inference. Trivial-ack / fallback paths yield
    /// `None` (no model output worth capturing).
    pub fn route_with_capture(
        &self,
        session_id: &str,
        user_input: &str,
        summary: Option<&str>,
        turn_index: usize,
        config: &RouterConfig,
    ) -> (RoutingDecision, Option<RouteCapture>) {
        if is_trivial_ack(user_input) {
            let decision = trivial_ack_decision();
            log::info!(
                "[SmartRouter] trivial-ack short circuit: session={}, tier=R0",
                session_id
            );
            return (decision, None);
        }

        // prev_tier（sticky 表）作为 v4 head 的 one-hot 特征与后处理共用。
        let prev_tier = self.previous_tier(session_id);
        let raw = match self.classify(
            user_input,
            summary,
            prev_tier.map(|t| t.index() as u8),
            config,
            true,
        ) {
            Ok(raw) => raw,
            Err(e) => {
                log::warn!(
                    "[SmartRouter] classification unavailable, falling back: session={}, error={}",
                    session_id,
                    e
                );
                // Fallback has no tier semantics — leave the sticky table
                // untouched so a transient failure cannot drag the floor down.
                return (fallback_decision(), None);
            }
        };

        let is_short = is_short_message(user_input);
        let decision = postprocess(&raw.decision.probabilities, prev_tier, is_short, config);
        if decision.source == DecisionSource::Model {
            log::debug!(
                "[SmartRouter] decision: session={}, turn={}, route={}, confidence={:.3}, \
                 safety_applied={}, sticky_applied={}",
                session_id,
                turn_index,
                decision.route,
                decision.confidence,
                decision.safety_applied,
                decision.sticky_applied
            );
        }

        self.remember_tier(session_id, decision.route);
        let num_embd = self
            .engine
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|e| e.num_embd())
            .unwrap_or_else(|| raw.hidden.len());
        (
            decision,
            Some(RouteCapture {
                input: raw.input,
                probs: raw.decision.probabilities,
                hidden: raw.hidden,
                prev_tier: prev_tier.map(|t| t.index() as u8),
                num_embd,
            }),
        )
    }

    /// Resolves the model reference for a routing decision.
    /// Returns the fallback reference when the decision is not model-backed.
    pub fn resolve_model_ref(decision: &RoutingDecision, config: &RouterConfig) -> String {
        match decision.source {
            DecisionSource::Model | DecisionSource::TrivialAck => {
                config.tier_models.model_for(&decision.route).to_string()
            }
            DecisionSource::Fallback => config.fallback.clone(),
        }
    }

    /// Stateless preview of the routing pipeline for a single request:
    /// same rules + classification + post-processing as [`route`], but the
    /// sticky table is neither read nor written. Used by test/UI entries so
    /// manual tests cannot pollute live session state (and the engine gets
    /// `capture=false` so previews don't pollute the evolution samples).
    pub fn route_preview(
        &self,
        user_input: &str,
        summary: Option<&str>,
        config: &RouterConfig,
    ) -> RoutingDecision {
        if is_trivial_ack(user_input) {
            return trivial_ack_decision();
        }
        let raw = match self.classify(user_input, summary, None, config, false) {
            Ok(raw) => raw,
            Err(e) => {
                log::warn!("[SmartRouter] preview classification failed: {}", e);
                return fallback_decision();
            }
        };
        postprocess(
            &raw.decision.probabilities,
            None,
            is_short_message(user_input),
            config,
        )
    }

    /// Builds the classifier input: summary + request when a session summary
    /// exists, bare request otherwise (both are trained distributions).
    pub fn build_classify_input(user_input: &str, summary: Option<&str>) -> String {
        match summary {
            Some(s) if !s.trim().is_empty() => {
                format!("Summary: {}\nRequest: {}", s.trim(), user_input)
            }
            _ => user_input.to_string(),
        }
    }

    /// Runs classification on the request. Returns the built input text, the
    /// raw pre-postprocess decision and the hidden state (for capture).
    /// `capture` marks real routing calls.
    fn classify(
        &self,
        user_input: &str,
        summary: Option<&str>,
        prev_tier: Option<u8>,
        config: &RouterConfig,
        capture: bool,
    ) -> Result<RawClassification, String> {
        let engine = self
            .engine
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or("classify engine not registered")?;
        if !engine.is_initialized() {
            return Err("classify engine not initialized".to_string());
        }
        let _ = config; // timeout policy is the host's/engine's responsibility

        let input = Self::build_classify_input(user_input, summary);
        let (probs, hidden) = engine.classify(&input, prev_tier, capture)?;

        if probs.len() != 4 {
            return Err(format!(
                "classify returned {} probabilities, expected 4",
                probs.len()
            ));
        }

        Ok(RawClassification {
            input,
            hidden,
            decision: RoutingDecision {
                route: RouteClass::R1, // Placeholder; postprocess computes the final route.
                confidence: 0.0,
                probabilities: [probs[0], probs[1], probs[2], probs[3]],
                margin: 0.0,
                sticky_applied: false,
                safety_applied: false,
                source: DecisionSource::Model,
            },
        })
    }

    fn previous_tier(&self, session_id: &str) -> Option<RouteClass> {
        let table = self.sticky_tiers.lock().unwrap_or_else(|e| e.into_inner());
        table.get(session_id).copied()
    }

    fn remember_tier(&self, session_id: &str, tier: RouteClass) {
        let mut table = self.sticky_tiers.lock().unwrap_or_else(|e| e.into_inner());
        let mut order = self.sticky_order.lock().unwrap_or_else(|e| e.into_inner());
        if table.contains_key(session_id) {
            // Refresh recency: move the session to the back (most recent).
            order.retain(|k| k != session_id);
        } else {
            // Evict the oldest sessions (LRU) instead of clearing everything.
            while table.len() >= STICKY_TABLE_LIMIT {
                match order.pop_front() {
                    Some(key) => {
                        table.remove(&key);
                    }
                    None => break,
                }
            }
        }
        order.push_back(session_id.to_string());
        table.insert(session_id.to_string(), tier);
    }
}

static GLOBAL_SMART_ROUTER: OnceLock<SmartRouter> = OnceLock::new();

/// Returns the process-wide router singleton (convenience for FFI/binding
/// layers; prefer owning a `SmartRouter` when the host can manage lifetime).
pub fn global_router() -> &'static SmartRouter {
    GLOBAL_SMART_ROUTER.get_or_init(SmartRouter::new)
}

/// Registers (or hot-swaps) the classification engine on the global router.
pub fn set_global_engine(engine: Arc<dyn ClassifyEngine>) {
    global_router().set_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mock::MockEngine;

    fn config() -> RouterConfig {
        RouterConfig::default()
    }

    #[test]
    fn trivial_ack_routes_to_r0_without_engine() {
        let router = SmartRouter::new();
        let decision = router.route("s1", "thanks", None, 0, &config());
        assert_eq!(decision.route, RouteClass::R0);
        assert_eq!(decision.source, DecisionSource::TrivialAck);
    }

    #[test]
    fn non_trivial_falls_back_when_engine_unavailable() {
        // No engine registered -> classification must degrade to the fallback
        // decision instead of panicking.
        let router = SmartRouter::new();
        let decision = router.route(
            "s1",
            "help me debug this traceback: ...",
            None,
            0,
            &config(),
        );
        assert_eq!(decision.source, DecisionSource::Fallback);
    }

    #[test]
    fn unready_engine_falls_back() {
        // Engine registered but not initialized -> same graceful degradation.
        struct NotReady;
        impl ClassifyEngine for NotReady {
            fn classify(
                &self,
                _: &str,
                _: Option<u8>,
                _: bool,
            ) -> Result<(Vec<f32>, Vec<f32>), String> {
                Err("not ready".to_string())
            }
            fn num_embd(&self) -> usize {
                0
            }
            fn is_initialized(&self) -> bool {
                false
            }
        }
        let router = SmartRouter::with_engine(Arc::new(NotReady));
        let decision = router.route("s1", "help me debug this", None, 0, &config());
        assert_eq!(decision.source, DecisionSource::Fallback);
    }

    #[test]
    fn model_decision_uses_mock_engine_and_remembers_tier() {
        // argmax R3 with high confidence -> route R3, sticky floor recorded.
        let router = SmartRouter::with_engine(Arc::new(MockEngine::new([0.02, 0.03, 0.05, 0.9])));
        let decision = router.route("s1", "help me debug this traceback", None, 0, &config());
        assert_eq!(decision.route, RouteClass::R3);
        assert_eq!(decision.source, DecisionSource::Model);
        assert_eq!(router.previous_tier("s1"), Some(RouteClass::R3));
    }

    #[test]
    fn sticky_lifts_short_followup() {
        // R3 session, then a short follow-up ("继续") the model classifies
        // low -> sticky floor lifts it back to R3.
        let router = SmartRouter::with_engine(Arc::new(MockEngine::new([0.02, 0.03, 0.05, 0.9])));
        router.route("s1", "help me debug this traceback", None, 0, &config());
        // Second turn: the engine now classifies low (argmax R0); short
        // message + sticky floor R3 must lift the decision to R3.
        router.set_engine(Arc::new(MockEngine::new([0.7, 0.2, 0.05, 0.05])));
        let decision = router.route("s1", "继续", None, 1, &config());
        assert_eq!(decision.route, RouteClass::R3);
        assert!(decision.sticky_applied);
    }

    #[test]
    fn trivial_ack_does_not_pollute_sticky_table() {
        let router = SmartRouter::new();
        // Trivial ack routes to R0 but must NOT be remembered — otherwise an
        // "ok" mid-task resets the sticky floor and later short messages
        // ("继续") would no longer be lifted back to the task's tier.
        router.route("s1", "ok", None, 0, &config());
        assert_eq!(router.previous_tier("s1"), None);
    }

    #[test]
    fn fallback_does_not_pollute_sticky_table() {
        let router = SmartRouter::new();
        // Engine unavailable -> fallback decision; sticky table stays empty.
        let d1 = router.route("s1", "help me debug this", None, 0, &config());
        assert_eq!(d1.source, DecisionSource::Fallback);
        let d2 = router.route("s1", "another request", None, 1, &config());
        assert_eq!(d2.source, DecisionSource::Fallback);
        assert_eq!(router.previous_tier("s1"), None);
    }

    #[test]
    fn trivial_ack_keeps_existing_sticky_tier() {
        let router = SmartRouter::with_engine(Arc::new(MockEngine::new([0.02, 0.03, 0.05, 0.9])));
        router.route("s3", "help me debug this traceback", None, 0, &config());
        // Trivial ack mid-task: routes R0 but the sticky floor must survive.
        let decision = router.route("s3", "ok", None, 1, &config());
        assert_eq!(decision.route, RouteClass::R0);
        assert_eq!(decision.source, DecisionSource::TrivialAck);
        assert_eq!(router.previous_tier("s3"), Some(RouteClass::R3));
    }

    #[test]
    fn sticky_tables_are_session_isolated() {
        let router = SmartRouter::new();
        router.route("s2", "ok", None, 0, &config());
        assert_eq!(router.previous_tier("s2"), None);
        assert_eq!(router.previous_tier("other"), None);
    }

    #[test]
    fn preview_neither_reads_nor_writes_sticky() {
        // Turn 1 of session s1: argmax R3 -> sticky floor R3 recorded.
        let router = SmartRouter::with_engine(Arc::new(MockEngine::new([0.02, 0.03, 0.05, 0.9])));
        router.route("s1", "help me debug this traceback", None, 0, &config());
        // Preview "继续" with the engine classifying R1: stateless — no
        // sticky context, so the decision must stay R1 (not lifted to R3),
        // and the preview must not corrupt s1's recorded floor either.
        router.set_engine(Arc::new(MockEngine::new([0.1, 0.6, 0.2, 0.1])));
        let preview = router.route_preview("继续", None, &config());
        assert_eq!(preview.route, RouteClass::R1);
        assert!(!preview.sticky_applied);
        assert_eq!(router.previous_tier("s1"), Some(RouteClass::R3));
    }

    #[test]
    fn set_engine_hot_swap_takes_effect() {
        let router = SmartRouter::new();
        assert_eq!(
            router
                .route("s1", "help me debug this", None, 0, &config())
                .source,
            DecisionSource::Fallback
        );
        router.set_engine(Arc::new(MockEngine::new([0.02, 0.03, 0.05, 0.9])));
        let decision = router.route("s1", "help me debug this", None, 1, &config());
        assert_eq!(decision.source, DecisionSource::Model);
        assert_eq!(decision.route, RouteClass::R3);
    }

    #[test]
    fn classify_input_formats() {
        // Bare request (no summary) matches the legacy distribution.
        assert_eq!(
            SmartRouter::build_classify_input("fix this bug", None),
            "fix this bug"
        );
        assert_eq!(
            SmartRouter::build_classify_input("fix this bug", Some("   ")),
            "fix this bug"
        );
        // Summary + request uses the trained two-segment format.
        assert_eq!(
            SmartRouter::build_classify_input("改成中文", Some("Translating docs to English")),
            "Summary: Translating docs to English\nRequest: 改成中文"
        );
    }

    #[test]
    fn sticky_table_evicts_oldest_not_all() {
        let router = SmartRouter::new();
        // Fill to the limit: s0..s1023 (s0 is the oldest).
        for i in 0..STICKY_TABLE_LIMIT {
            router.remember_tier(&format!("s{i}"), RouteClass::R1);
        }
        // One more session evicts only the oldest (s0), not everything.
        router.remember_tier("s_new", RouteClass::R2);
        assert_eq!(router.previous_tier("s0"), None);
        assert_eq!(router.previous_tier("s1"), Some(RouteClass::R1));
        assert_eq!(router.previous_tier("s1023"), Some(RouteClass::R1));
        assert_eq!(router.previous_tier("s_new"), Some(RouteClass::R2));
    }

    #[test]
    fn sticky_table_refreshes_recency_on_update() {
        let router = SmartRouter::new();
        // Fill exactly to the limit: s0..s1023 (no eviction yet).
        for i in 0..STICKY_TABLE_LIMIT {
            router.remember_tier(&format!("s{i}"), RouteClass::R1);
        }
        // Touch s0 again — it becomes most recent and must survive eviction.
        router.remember_tier("s0", RouteClass::R3);
        router.remember_tier("s_extra", RouteClass::R2);
        assert_eq!(router.previous_tier("s0"), Some(RouteClass::R3));
        assert_eq!(router.previous_tier("s1"), None); // oldest evicted instead
        assert_eq!(router.previous_tier("s_extra"), Some(RouteClass::R2));
    }

    #[test]
    fn resolve_model_ref_uses_tier_mapping() {
        let decision = RoutingDecision {
            route: RouteClass::R3,
            confidence: 0.9,
            probabilities: [0.0, 0.05, 0.05, 0.9],
            margin: 0.85,
            sticky_applied: false,
            safety_applied: false,
            source: DecisionSource::Model,
        };
        let cfg = config();
        assert_eq!(SmartRouter::resolve_model_ref(&decision, &cfg), "primary");

        let fb = fallback_decision();
        assert_eq!(SmartRouter::resolve_model_ref(&fb, &cfg), "primary");
    }

    #[test]
    fn global_router_accepts_engine() {
        // The global singleton must stay usable across calls (shared state);
        // only exercise the registration path without asserting tier values
        // (other tests may have re-registered the engine).
        crate::set_global_engine(Arc::new(MockEngine::new([0.7, 0.1, 0.1, 0.1])));
        let router = crate::global_router();
        let d = router.route("global-test", "ok", None, 0, &config());
        assert_eq!(d.source, DecisionSource::TrivialAck);
    }
}
