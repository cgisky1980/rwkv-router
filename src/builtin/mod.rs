//! Built-in RWKV engines (feature `rwkv`), backed by [rwkv-rsv] (Vulkan).
//!
//! Two independent engines:
//!
//! - [`RwkvClassifyEngine`] — resident 0.1B model for tier classification:
//!   tokenize → mean-pooled last-layer hidden state (`forward_seq_mean_hidden`)
//!   → trained MLP head ([`RouterHead`]). Implements [`ClassifyEngine`], so it
//!   plugs straight into the [`SmartRouter`](crate::SmartRouter).
//! - [`RwkvGenerateEngine`] — LRU pool of tier generation models (0.1B–13B):
//!   loads a tier's model on demand, evicts the least-recently-used tier when
//!   the cap is exceeded, and runs autoregressive sampling on a worker thread.
//!
//! ## Threading model
//!
//! `rwkv-rsv` is a synchronous API and `GpuModel` is **not `Send`**. Each
//! engine therefore spawns a dedicated OS thread that owns the model and all
//! inference state; the public handle is a channel-based wrapper (`Send +
//! Sync`) that forwards requests and blocks on a response channel. Timeouts
//! are enforced on the handle side so a wedged GPU call can never deadlock
//! the caller (the worker thread is left to finish or die on its own).
//!
//! The Ai00-X client does NOT use these engines — it injects its own
//! `ClassifyEngine` (async pool in `rwkv_llm.rs`). These engines serve the
//! standalone product: sidecar gateway, FFI/binding hosts, demos.

mod generate;

pub use generate::{
    GenOutput, GenParams, GenerateEngineConfig, RwkvGenerateEngine, TierModelPaths,
};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

use rwkv_rsv::gpu_model::{Bundle, GpuModel, ModelBuilder, State};
use rwkv_rsv::tokenizer::Tokenizer;

use crate::engine::ClassifyEngine;
use crate::head::RouterHead;

/// Classification input truncation bound (tokens): two-segment budget =
/// session summary (~128) + current request (~128); aligns with the 256-token
/// GEMM padding tier, so latency matches the client's trained distribution.
const CLASSIFY_MAX_TOKENS: usize = 256;

/// Configuration for [`RwkvClassifyEngine`].
#[derive(Debug, Clone)]
pub struct ClassifyEngineConfig {
    /// RWKV model file (`.st`, int8/fp16) for the resident classifier.
    /// Typically a 0.1B model: ~0.2 GB VRAM, sub-second prefill.
    pub model_path: String,
    /// World tokenizer vocab JSON (shared across RWKV World models).
    pub vocab_path: String,
    /// Trained MLP head JSON (`router_head.json`). Required: the engine is
    /// useless without it, so a missing/mismatched head fails construction.
    pub head_path: String,
    /// Classification call timeout (handle side). Default 5000 ms.
    pub timeout_ms: u64,
    /// Model/head load timeout at construction (worker side). Default 120 s.
    pub load_timeout_secs: u64,
}

impl Default for ClassifyEngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            vocab_path: String::new(),
            head_path: String::new(),
            timeout_ms: 5000,
            load_timeout_secs: 120,
        }
    }
}

/// Classify 响应：(probs, hidden) 或错误。
type ClassifyResponse = Result<(Vec<f32>, Vec<f32>), String>;

enum ClassifyRequest {
    /// Classify `input` (already built by the router: summary + request).
    Classify {
        input: String,
        prev_tier: Option<u8>,
        respond: mpsc::Sender<ClassifyResponse>,
    },
    /// Re-read the head file (evolution hot-deploy) and re-check its dim.
    ReloadHead {
        respond: mpsc::Sender<Result<(), String>>,
    },
}

/// Worker state (lives on the dedicated inference thread; never crosses it).
struct ClassifyWorker {
    model: GpuModel,
    state: State,
    initial_state: Vec<f32>,
    tokenizer: Tokenizer,
    head: RouterHead,
    head_path: String,
}

impl ClassifyWorker {
    /// Builds the worker state (model + tokenizer + head). Runs **on the
    /// worker thread** — `GpuModel` is not `Send`, so the model is loaded
    /// where it lives and never crosses threads.
    fn build(config: &ClassifyEngineConfig) -> Result<Self, String> {
        log::info!("[builtin] loading classify model: {}", config.model_path);
        let bundle: Bundle = ModelBuilder::new(&config.model_path)
            .build()
            .map_err(|e| format!("failed to load classify model '{}': {e}", config.model_path))?;
        let Bundle { mut model, state } = bundle;

        let initial_state = model
            .state_back(&state)
            .map_err(|e| format!("failed to snapshot classify initial state: {e}"))?;
        let state = model
            .create_state()
            .map_err(|e| format!("failed to create classify state: {e}"))?;

        let vocab = std::fs::read_to_string(&config.vocab_path)
            .map_err(|e| format!("failed to read vocab '{}': {e}", config.vocab_path))?;
        let tokenizer =
            Tokenizer::new(&vocab).map_err(|e| format!("failed to parse vocab: {e}"))?;

        let num_embd = model.info().num_emb;
        let head = RouterHead::from_json_file(&config.head_path)
            .map_err(|e| format!("failed to load router head '{}': {e}", config.head_path))?;
        if head.expected_hidden_dim() != num_embd {
            return Err(format!(
                "router head expects hidden {} != model n_embd {num_embd} (head/model mismatch)",
                head.expected_hidden_dim()
            ));
        }

        Ok(Self {
            model,
            state,
            initial_state,
            tokenizer,
            head,
            head_path: config.head_path.clone(),
        })
    }

    /// Extracts the mean-pooled last-layer hidden state (state embedding):
    /// reset to the zero state → prefill → mean hidden.
    fn extract_hidden(&mut self, request: &str) -> Result<Vec<f32>, String> {
        let mut tokens = self
            .tokenizer
            .encode(request.as_bytes())
            .map_err(|e| format!("failed to encode classify request: {e}"))?;
        if tokens.is_empty() {
            // Empty text cannot be prefilled; trivial inputs should have been
            // intercepted by the router's trivial-ack rule before reaching us.
            return Err("classify request encoded to zero tokens".to_string());
        }
        tokens.truncate(CLASSIFY_MAX_TOKENS);
        self.model
            .state_load(&self.state, &self.initial_state)
            .map_err(|e| format!("failed to reset classify state: {e}"))?;
        self.model
            .forward_seq_mean_hidden(&mut self.state, &tokens)
            .map_err(|e| format!("classify prefill failed: {e}"))
    }
}

/// Resident 0.1B classification engine (channel handle to a worker thread).
pub struct RwkvClassifyEngine {
    tx: Mutex<mpsc::Sender<ClassifyRequest>>,
    num_embd: usize,
    ready: AtomicBool,
    timeout: Duration,
}

impl std::fmt::Debug for RwkvClassifyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwkvClassifyEngine")
            .field("num_embd", &self.num_embd)
            .field("ready", &self.ready.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl RwkvClassifyEngine {
    /// Spawns the inference worker and waits for it to load model + head.
    ///
    /// Loading happens **inside the worker thread** (`GpuModel` is not
    /// `Send`); construction blocks until the load result arrives so errors
    /// surface directly. A 0.1B model loads in seconds; the load timeout is
    /// generous to also cover big-model setups and first GPU init.
    pub fn new(config: ClassifyEngineConfig) -> Result<Self, String> {
        if config.model_path.is_empty() || config.vocab_path.is_empty() {
            return Err("classify engine: model_path and vocab_path are required".to_string());
        }
        let timeout = Duration::from_millis(config.timeout_ms.max(100));
        let load_timeout = Duration::from_secs(config.load_timeout_secs.max(10));

        let (tx, rx) = mpsc::channel::<ClassifyRequest>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<usize, String>>();
        std::thread::Builder::new()
            .name("rwkv-router-classify".to_string())
            .spawn(move || {
                let worker = match ClassifyWorker::build(&config) {
                    Ok(w) => w,
                    Err(e) => {
                        log::error!("[builtin] classify worker init failed: {e}");
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(worker.model.info().num_emb));
                classify_worker_loop(worker, rx);
            })
            .map_err(|e| format!("failed to spawn classify worker: {e}"))?;

        let num_embd = match ready_rx.recv_timeout(load_timeout) {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("classify worker init timed out".to_string()),
        };

        Ok(Self {
            tx: Mutex::new(tx),
            num_embd,
            ready: AtomicBool::new(true),
            timeout,
        })
    }

    fn send(&self, req: ClassifyRequest) -> Result<(), String> {
        self.tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(req)
            .map_err(|_| "classify worker terminated".to_string())
    }
}

impl ClassifyEngine for RwkvClassifyEngine {
    /// Classifies `input` into `(probs, hidden)`. `capture` is accepted for
    /// the [`ClassifyEngine`] contract; the engine always returns the hidden
    /// vector and lets the caller (router/evolution coordinator) decide what
    /// to store.
    fn classify(
        &self,
        input: &str,
        prev_tier: Option<u8>,
        _capture: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (rtx, rrx) = mpsc::channel();
        self.send(ClassifyRequest::Classify {
            input: input.to_string(),
            prev_tier,
            respond: rtx,
        })?;
        match rrx.recv_timeout(self.timeout) {
            Ok(res) => res,
            Err(mpsc::RecvTimeoutError::Timeout) => Err("classify timed out".to_string()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("classify worker terminated".to_string())
            }
        }
    }

    fn num_embd(&self) -> usize {
        self.num_embd
    }

    fn is_initialized(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Hot-reloads the head file after the evolution loop deploys a new one.
    /// The new head must match the model's hidden dim (checked in the worker).
    fn reload(&self) -> Result<(), String> {
        let (rtx, rrx) = mpsc::channel();
        self.send(ClassifyRequest::ReloadHead { respond: rtx })?;
        match rrx.recv_timeout(self.timeout) {
            Ok(res) => res,
            Err(mpsc::RecvTimeoutError::Timeout) => Err("head reload timed out".to_string()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("classify worker terminated".to_string())
            }
        }
    }
}

fn classify_worker_loop(mut worker: ClassifyWorker, rx: mpsc::Receiver<ClassifyRequest>) {
    while let Ok(req) = rx.recv() {
        match req {
            ClassifyRequest::Classify {
                input,
                prev_tier,
                respond,
            } => {
                let result = worker
                    .extract_hidden(&input)
                    .and_then(|hidden| {
                        let probs = worker.head.forward(&hidden, prev_tier)?;
                        Ok((probs.to_vec(), hidden))
                    })
                    .map_err(|e| {
                        log::warn!("[builtin] classify failed: {e}");
                        e
                    });
                let _ = respond.send(result);
            }
            ClassifyRequest::ReloadHead { respond } => {
                let result = RouterHead::from_json_file(&worker.head_path)
                    .map_err(|e| format!("head reload failed: {e}"))
                    .and_then(|head| {
                        if head.expected_hidden_dim() != worker.model.info().num_emb {
                            return Err(format!(
                                "reloaded head expects hidden {} != model n_embd {}",
                                head.expected_hidden_dim(),
                                worker.model.info().num_emb
                            ));
                        }
                        worker.head = head;
                        log::info!("[builtin] router head reloaded: {}", worker.head_path);
                        Ok(())
                    });
                let _ = respond.send(result);
            }
        }
    }
    log::info!("[builtin] classify worker exiting (channel closed)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_config_defaults() {
        let cfg = ClassifyEngineConfig::default();
        assert_eq!(cfg.timeout_ms, 5000);
        assert!(cfg.model_path.is_empty());
    }

    #[test]
    fn constructor_validates_required_paths() {
        // Missing model/vocab paths fail before any GPU work.
        let err = RwkvClassifyEngine::new(ClassifyEngineConfig::default()).unwrap_err();
        assert!(err.contains("required"), "unexpected error: {err}");

        let cfg = ClassifyEngineConfig {
            model_path: "missing.st".to_string(),
            vocab_path: "missing_vocab.json".to_string(),
            ..Default::default()
        };
        let err = RwkvClassifyEngine::new(cfg).unwrap_err();
        // Model load comes first (before vocab read and head parse).
        assert!(
            err.contains("failed to load classify model"),
            "unexpected error: {err}"
        );
    }
}
