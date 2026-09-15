//! Evolution progress events (replaces the client's Tauri `app.emit`).
//!
//! Hosts (sidecar HTTP SSE, Python/Node callbacks, desktop `app.emit` shim)
//! implement [`EvolutionEventSink`] to receive progress; the evolution loop
//! runs on its own std::thread and must never block on the sink.

use serde::Serialize;

use super::EvolveResult;

/// One progress event from the evolution loop.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvolutionEvent {
    /// Evolution run started (auto or manual).
    Started,
    /// Stage update: `loading` / `training` / `gating` / `done`.
    Phase { stage: String, detail: String },
    /// Final outcome (ok / rejected / skipped + metrics).
    Finished(EvolveResult),
}

/// Receives evolution progress events. Called from the evolution thread —
/// implementations must be fast and non-blocking.
pub trait EvolutionEventSink: Send + Sync {
    fn on_event(&self, event: EvolutionEvent);
}
