//! # RWKV-Router
//!
//! Self-evolving smart router extracted from Ai00-X (standalone component).
//!
//! Pipeline: rule short-circuit (trivial ack) -> RWKV local classification
//! (mean-hidden state embedding + trained MLP head) -> post-processing rule
//! stack (safety upgrade / sticky tier) -> R0-R3 tier decision.
//!
//! Two usage modes:
//! - **route-only**: consume [`RoutingDecision`] and let the host pick models;
//! - **local-stack** (feature `rwkv`): map each tier to RWKV 0.1B-13B models
//!   with built-in inference (sidecar).
//!
//! The self-evolution loop (capture -> label -> pure-Rust AdamW fine-tune ->
//! eval gate -> hot reload) ships in [`evolution`]; it never degrades routing
//! because a frozen eval gate rejects any regression before deployment.
//!
//! Quick start (route-only, no engine yet — rules + fallback only):
//!
//! ```
//! use rwkv_router::{global_router, RouterConfig, SmartRouter};
//!
//! let router = SmartRouter::new();
//! let decision = router.route("s1", "ok", None, 0, &RouterConfig::default());
//! assert_eq!(decision.route.to_string(), "R0");
//!
//! // Process-wide singleton with hot-swappable engine:
//! let _ = global_router();
//! ```

pub mod config;
pub mod engine;
pub mod evolution;
pub mod head;
pub mod postprocess;
pub mod router;
pub mod rules;
pub mod session;
pub mod tier;
pub mod training;

#[cfg(feature = "rwkv")]
pub mod builtin;
#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "rwkv")]
pub use builtin::{
    ClassifyEngineConfig, GenOutput, GenParams, GenerateEngineConfig, RwkvClassifyEngine,
    RwkvGenerateEngine, TierModelPaths,
};
pub use config::{RouterConfig, RouterTierModels};
pub use engine::ClassifyEngine;
pub use evolution::{
    CaptureItem, CaptureRecord, CaptureStats, Evolution, EvolutionConfig, EvolutionEvent,
    EvolutionEventSink, EvolveResult,
};
pub use head::RouterHead;
pub use postprocess::{
    fallback_decision, postprocess, softmax, trivial_ack_decision, DecisionSource, RoutingDecision,
};
pub use router::{global_router, set_global_engine, RouteCapture, SmartRouter};
pub use rules::{is_short_message, is_trivial_ack};
pub use session::RouterSession;
pub use tier::RouteClass;
pub use training::{finetune_head, FinetuneOptions, FinetuneReport, TrainingSample};
