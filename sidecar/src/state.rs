//! Shared gateway state: assembled [`RouterSession`] + config + evolve status.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rwkv_router::{EvolveResult, RouterSession};

use crate::config::SidecarConfig;

/// POST /v1/evolve runs the (blocking) evolution cycle on a worker thread;
/// this tracks progress and the last report for GET /v1/evolve/status.
#[derive(Default)]
pub struct EvolveStatus {
    running: AtomicBool,
    last: Mutex<Option<EvolveResult>>,
}

impl EvolveStatus {
    /// Marks the loop entered; returns false if already running (409).
    pub fn begin(&self) -> bool {
        !self.running.swap(true, Ordering::SeqCst)
    }

    pub fn finish(&self, result: EvolveResult) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn last_report(&self) -> Option<EvolveResult> {
        self.last.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// axum shared state (immutably shared: the session's mutating calls —
/// load_classifier / attach_generation / configure_evolution — happen once
/// at startup before the listener binds).
pub struct AppState {
    pub session: Arc<RouterSession>,
    pub config: Arc<SidecarConfig>,
    pub http: reqwest::Client,
    pub evolve: Arc<EvolveStatus>,
}

pub type SharedState = Arc<AppState>;
