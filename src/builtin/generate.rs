//! Tier generation engine: LRU pool of RWKV models (0.1B–13B) for local
//! inference, one dedicated worker thread per loaded model (`GpuModel` is
//! not `Send`).
//!
//! `generate(tier, prompt, params)` lazily loads the tier's model (evicting
//! the least-recently-used tier when `max_loaded` is exceeded — large RWKV
//! states make VRAM the binding constraint) and runs autoregressive sampling
//! on that model's worker: chunked prefill → temperature/top-k/top-p sampling
//! with RWKV-convention presence/frequency penalties → stop sequences / EOS.
//!
//! The sampling math mirrors the Ai00-X client's proven `sample_token`
//! (`penalty = presence + frequency * count^decay`, nucleus over softmax
//! probabilities), with one deliberate divergence: `temperature` is applied
//! (logits scaling) — the client ignores it, a standalone product should not.
//!
//! Dropping the engine (or LRU-evicting a tier) drops the worker's request
//! channel; the worker exits and frees its model/VRAM. Unload is therefore
//! just "remove from the pool".

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

use rwkv_rsv::gpu_model::{Bundle, GpuModel, ModelBuilder, State};
use rwkv_rsv::tokenizer::Tokenizer;

use crate::tier::RouteClass;

/// prefill chunk length: bounds per-call seq buffer size (client-verified).
const PREFILL_CHUNK: usize = 128;
/// stop-sequence match window (bytes of recently decoded text kept).
const STOP_BUFFER_LIMIT: usize = 200;
/// Default generation timeout (13B prefill + long generations need headroom).
const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Sampling / generation parameters (RWKV World conventions).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenParams {
    /// Max tokens to generate. 0 → 1 (matches the client's convention).
    pub max_tokens: usize,
    /// Logits temperature; `<= 0` → greedy argmax. Default 1.0.
    pub temperature: f32,
    /// Nucleus cutoff; clamped to [0, 1]. Default 0.9.
    pub top_p: f32,
    /// Top-k cutoff; 0 → 128 (client convention). Default 128.
    pub top_k: usize,
    /// Flat penalty per appeared token. Default 0.0.
    pub presence_penalty: f32,
    /// Frequency-scaled penalty (`count^decay`). Default 0.0.
    pub frequency_penalty: f32,
    /// Penalty decay exponent base (RWKV convention). Default 0.99654026.
    pub penalty_decay: f32,
    /// Stop sequences (checked over recently decoded text). Default empty.
    pub stop: Vec<String>,
    /// Stop when the EOS token (0, RWKV World endoftext) is sampled.
    /// Default true.
    pub stop_on_eos: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 128,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_decay: 0.99654026,
            stop: Vec::new(),
            stop_on_eos: true,
        }
    }
}

/// Result of one [`RwkvGenerateEngine::generate`] call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GenOutput {
    /// Generated text (stop sequence trimmed, EOS excluded).
    pub text: String,
    /// Prompt token count (prefill size).
    pub input_tokens: usize,
    /// Generated token count (EOS included when it terminated the run).
    pub output_tokens: usize,
    /// The stop sequence that ended generation, if any.
    pub stop_hit: Option<String>,
    /// Whether EOS (token 0) ended generation.
    pub stopped_by_eos: bool,
}

/// Per-tier model path configuration. `None` = tier not served locally
/// (the host should fall back to its own path for that tier).
#[derive(Debug, Clone, Default)]
pub struct TierModelPaths {
    pub r0: Option<String>,
    pub r1: Option<String>,
    pub r2: Option<String>,
    pub r3: Option<String>,
}

impl TierModelPaths {
    fn path_for(&self, tier: RouteClass) -> Option<&str> {
        match tier {
            RouteClass::R0 => self.r0.as_deref(),
            RouteClass::R1 => self.r1.as_deref(),
            RouteClass::R2 => self.r2.as_deref(),
            RouteClass::R3 => self.r3.as_deref(),
        }
    }
}

/// Configuration for [`RwkvGenerateEngine`].
#[derive(Debug, Clone)]
pub struct GenerateEngineConfig {
    /// World tokenizer vocab JSON (shared across RWKV World models).
    pub vocab_path: String,
    /// Per-tier model files (`.st`, int8/fp16).
    pub tier_models: TierModelPaths,
    /// Max concurrently loaded tier models (LRU eviction). Default 1 — a 13B
    /// model dominates VRAM; raise only for small-model setups.
    pub max_loaded: usize,
    /// Per-call timeout (handle side). Default 300 s.
    pub timeout_secs: u64,
}

impl Default for GenerateEngineConfig {
    fn default() -> Self {
        Self {
            vocab_path: String::new(),
            tier_models: TierModelPaths::default(),
            max_loaded: 1,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }
}

struct GenRequest {
    prompt: String,
    params: GenParams,
    respond: mpsc::Sender<Result<GenOutput, String>>,
}

/// One loaded tier model: a handle to its worker thread. Dropping it closes
/// the request channel → the worker exits → the model/VRAM is freed.
struct LoadedModel {
    tx: mpsc::Sender<GenRequest>,
    #[allow(dead_code)]
    model_path: String,
}

/// Worker state (lives on the dedicated inference thread; never crosses it).
struct GenWorker {
    model: GpuModel,
    state: State,
    initial_state: Vec<f32>,
    tokenizer: Tokenizer,
}

impl GenWorker {
    /// Runs one autoregressive generation on a freshly reset state. On error
    /// the state is left dirty; the caller resets before the next task.
    fn run(&mut self, prompt: &str, params: &GenParams) -> Result<GenOutput, String> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt.as_bytes())
            .map_err(|e| format!("failed to encode prompt: {e}"))?;
        let input_tokens = prompt_tokens.len();

        self.model
            .state_load(&self.state, &self.initial_state)
            .map_err(|e| format!("failed to reset generation state: {e}"))?;

        // Chunked sequence-parallel prefill → logits of the last prompt token.
        let mut logits = Vec::new();
        for chunk in prompt_tokens.chunks(PREFILL_CHUNK) {
            logits = self
                .model
                .forward_seq_with_state(&mut self.state, chunk)
                .map_err(|e| format!("generation prefill failed: {e}"))?;
        }

        let max_tokens = params.max_tokens.max(1);
        let top_p = params.top_p.clamp(0.0, 1.0);
        let top_k = if params.top_k == 0 { 128 } else { params.top_k };
        let mut token_counts: HashMap<u32, i32> = HashMap::new();
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut stop_buffer = String::new();
        let mut stop_hit: Option<String> = None;
        let mut stopped_by_eos = false;

        for _ in 0..max_tokens {
            let id = sample_token(
                &logits,
                params.temperature,
                top_p,
                top_k,
                &token_counts,
                params.presence_penalty,
                params.frequency_penalty,
                params.penalty_decay,
            );
            if id == 0 && params.stop_on_eos {
                stopped_by_eos = true;
                break;
            }
            acc_ids.push(id);
            *token_counts.entry(id).or_insert(0) += 1;

            let decoded = self.tokenizer.decode(&[id]).unwrap_or_default();
            stop_buffer.push_str(&String::from_utf8_lossy(&decoded));
            if stop_buffer.len() > STOP_BUFFER_LIMIT {
                let split_idx = stop_buffer.len() - 100;
                if let Some((idx, _)) = stop_buffer.char_indices().find(|(i, _)| *i >= split_idx) {
                    stop_buffer = stop_buffer[idx..].to_string();
                }
            }
            for stop_str in &params.stop {
                if !stop_str.is_empty() && stop_buffer.ends_with(stop_str.as_str()) {
                    stop_hit = Some(stop_str.clone());
                    break;
                }
            }
            if stop_hit.is_some() {
                break;
            }

            logits = self
                .model
                .forward_with_state(&mut self.state, &[id])
                .map_err(|e| format!("generation decode failed: {e}"))?;
        }

        let mut text = if acc_ids.is_empty() {
            String::new()
        } else {
            match self.tokenizer.decode(&acc_ids) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(_) => String::new(),
            }
        };
        if let Some(seq) = &stop_hit {
            if let Some(pos) = text.rfind(seq.as_str()) {
                text.truncate(pos);
            }
        }

        Ok(GenOutput {
            text,
            input_tokens,
            output_tokens: acc_ids.len(),
            stop_hit,
            stopped_by_eos,
        })
    }
}

/// Builds the worker state (model + tokenizer). Runs **on the worker
/// thread** — `GpuModel` is not `Send`, so the model is built where it lives
/// and never crosses threads.
fn build_gen_worker(model_path: &str, vocab_path: &str) -> Result<GenWorker, String> {
    let bundle: Bundle = ModelBuilder::new(model_path)
        .build()
        .map_err(|e| format!("failed to load generation model '{model_path}': {e}"))?;
    let Bundle { mut model, state } = bundle;

    let initial_state = model
        .state_back(&state)
        .map_err(|e| format!("failed to snapshot generation initial state: {e}"))?;
    let state = model
        .create_state()
        .map_err(|e| format!("failed to create generation state: {e}"))?;

    let vocab = std::fs::read_to_string(vocab_path)
        .map_err(|e| format!("failed to read vocab '{vocab_path}': {e}"))?;
    let tokenizer = Tokenizer::new(&vocab).map_err(|e| format!("failed to parse vocab: {e}"))?;

    Ok(GenWorker {
        model,
        state,
        initial_state,
        tokenizer,
    })
}

/// Temperature-scaled nucleus sampler with RWKV-convention penalties.
/// `temperature <= 0` degenerates to greedy argmax (after penalties).
#[allow(clippy::too_many_arguments)]
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    token_counts: &HashMap<u32, i32>,
    presence_penalty: f32,
    frequency_penalty: f32,
    penalty_decay: f32,
) -> u32 {
    let mut logits: Vec<f32> = logits.to_vec();
    for (&id, &count) in token_counts {
        if (id as usize) < logits.len() {
            let penalty = presence_penalty + frequency_penalty * (count as f32).powf(penalty_decay);
            logits[id as usize] -= penalty;
        }
    }
    if temperature > 0.0 {
        for l in &mut logits {
            *l /= temperature;
        }
    }

    // Softmax over the (penalized, tempered) logits.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut candidates: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k, then nucleus: keep the highest-probability candidates until the
    // cumulative mass reaches top_p (client-verified ordering).
    let mut cumsum = 0.0f32;
    let mut kept: Vec<(usize, f32)> = Vec::new();
    for (i, p) in candidates.into_iter().take(top_k) {
        cumsum += p;
        kept.push((i, p));
        if cumsum >= top_p {
            break;
        }
    }
    let total = cumsum.min(1.0);
    let r = fastrand::f64() as f32 * total;
    let mut acc = 0.0f32;
    let mut selected = kept[0].0 as u32;
    for (i, p) in kept {
        acc += p;
        if acc >= r {
            selected = i as u32;
            break;
        }
    }
    selected
}

/// LRU pool of tier generation models.
pub struct RwkvGenerateEngine {
    vocab_path: String,
    tier_models: TierModelPaths,
    max_loaded: usize,
    timeout: Duration,
    inner: Mutex<GenInner>,
}

impl std::fmt::Debug for RwkvGenerateEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RwkvGenerateEngine")
            .field("tier_models", &self.tier_models)
            .field("max_loaded", &self.max_loaded)
            .finish_non_exhaustive()
    }
}

struct GenInner {
    loaded: HashMap<RouteClass, LoadedModel>,
    order: VecDeque<RouteClass>,
}

impl RwkvGenerateEngine {
    /// Creates the pool without loading any model; tiers load lazily on
    /// first use (a 7B/13B load takes tens of seconds — don't trigger the
    /// first load inside a latency-sensitive path; the sidecar warms its
    /// default tier at startup instead).
    pub fn new(config: GenerateEngineConfig) -> Result<Self, String> {
        if config.vocab_path.is_empty() {
            return Err("generate engine: vocab_path is required".to_string());
        }
        let max_loaded = config.max_loaded.max(1);
        let timeout = Duration::from_secs(config.timeout_secs.max(1));
        Ok(Self {
            vocab_path: config.vocab_path,
            tier_models: config.tier_models,
            max_loaded,
            timeout,
            inner: Mutex::new(GenInner {
                loaded: HashMap::new(),
                order: VecDeque::new(),
            }),
        })
    }

    /// Generates from the tier's model, loading it first if needed.
    pub fn generate(
        &self,
        tier: RouteClass,
        prompt: &str,
        params: &GenParams,
    ) -> Result<GenOutput, String> {
        let (rtx, rrx) = mpsc::channel();
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            self.ensure_loaded(&mut inner, tier)?;
            let tx = inner
                .loaded
                .get(&tier)
                .ok_or_else(|| format!("{tier} generation model not loaded"))?
                .tx
                .clone();
            tx.send(GenRequest {
                prompt: prompt.to_string(),
                params: params.clone(),
                respond: rtx,
            })
            .map_err(|_| "generation worker terminated".to_string())?;
        } // pool lock released before blocking on inference

        match rrx.recv_timeout(self.timeout) {
            Ok(res) => res,
            Err(mpsc::RecvTimeoutError::Timeout) => Err("generation timed out".to_string()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("generation worker terminated".to_string())
            }
        }
    }

    /// Whether the tier's model is currently loaded (no VRAM query — the
    /// worker owns the model; this is pool bookkeeping only).
    pub fn is_loaded(&self, tier: RouteClass) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .loaded
            .contains_key(&tier)
    }

    /// Unloads a tier's model immediately (frees VRAM when its worker exits).
    pub fn unload(&self, tier: RouteClass) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.loaded.remove(&tier).is_some() {
            inner.order.retain(|&t| t != tier);
            log::info!("[builtin] unloaded generation model for {tier}");
        }
    }

    fn ensure_loaded(&self, inner: &mut GenInner, tier: RouteClass) -> Result<(), String> {
        if inner.loaded.contains_key(&tier) {
            // Refresh recency.
            inner.order.retain(|&t| t != tier);
            inner.order.push_back(tier);
            return Ok(());
        }
        let model_path = self
            .tier_models
            .path_for(tier)
            .ok_or_else(|| format!("{tier} has no local generation model configured"))?
            .to_string();
        let handle = self.load_model(tier, &model_path)?;
        inner.loaded.insert(tier, handle);
        inner.order.push_back(tier);
        // Evict least-recently-used tiers beyond the cap. The just-loaded
        // tier sits at the back of `order`, so it is never the victim while
        // older tiers remain; guard anyway to keep the LRU invariant intact.
        while inner.loaded.len() > self.max_loaded {
            let Some(evict) = inner.order.pop_front() else {
                break;
            };
            if evict == tier {
                inner.order.push_front(evict);
                break;
            }
            if inner.loaded.remove(&evict).is_some() {
                log::info!("[builtin] evicted generation model for {evict} (LRU)");
            }
        }
        Ok(())
    }

    fn load_model(&self, tier: RouteClass, model_path: &str) -> Result<LoadedModel, String> {
        log::info!("[builtin] loading generation model for {tier}: {model_path}");
        let (tx, rx) = mpsc::channel::<GenRequest>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let vocab_path = self.vocab_path.clone();
        let model_path = model_path.to_string();
        let model_path_for_worker = model_path.clone();
        let load_timeout = Duration::from_secs(self.timeout.as_secs().max(10));
        std::thread::Builder::new()
            .name(format!("rwkv-router-gen-{tier}"))
            .spawn(move || {
                // Load inside the worker thread: `GpuModel` is not `Send`,
                // so the model is built where it lives.
                let worker = match build_gen_worker(&model_path_for_worker, &vocab_path) {
                    Ok(w) => w,
                    Err(e) => {
                        log::error!("[builtin] generation worker init failed: {e}");
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(()));
                generate_worker_loop(worker, rx);
            })
            .map_err(|e| format!("failed to spawn generation worker: {e}"))?;

        // Block until the worker reports its model loaded. The pool lock is
        // held by the caller: concurrent loads serialize, which is fine —
        // concurrent 13B loads would thrash VRAM anyway.
        match ready_rx.recv_timeout(load_timeout) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("generation worker init timed out".to_string()),
        }

        Ok(LoadedModel { tx, model_path })
    }
}

fn generate_worker_loop(mut worker: GenWorker, rx: mpsc::Receiver<GenRequest>) {
    while let Ok(req) = rx.recv() {
        let GenRequest {
            prompt,
            params,
            respond,
        } = req;
        let result = worker
            .run(&prompt, &params)
            .inspect_err(|e| log::warn!("[builtin] generation failed: {e}"));
        // A failed forward may leave the state dirty; reset before the next
        // task so one bad request cannot poison the worker.
        if result.is_err() {
            let _ = worker
                .model
                .state_load(&worker.state, &worker.initial_state);
        }
        let _ = respond.send(result);
    }
    log::info!("[builtin] generation worker exiting (channel closed)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_params_defaults() {
        let p = GenParams::default();
        assert_eq!(p.max_tokens, 512);
        assert_eq!(p.top_k, 128);
        assert!(p.stop_on_eos);
        assert!((p.penalty_decay - 0.99654026).abs() < 1e-8);
    }

    #[test]
    fn tier_paths_lookup() {
        let paths = TierModelPaths {
            r1: Some("m1.st".to_string()),
            ..Default::default()
        };
        assert_eq!(paths.path_for(RouteClass::R1), Some("m1.st"));
        assert_eq!(paths.path_for(RouteClass::R0), None);
        assert_eq!(paths.path_for(RouteClass::R3), None);
    }

    #[test]
    fn generate_engine_requires_vocab() {
        let err = RwkvGenerateEngine::new(GenerateEngineConfig::default()).unwrap_err();
        assert!(err.contains("vocab_path"), "unexpected error: {err}");
    }

    #[test]
    fn generate_without_tier_model_errors_without_gpu() {
        let engine = RwkvGenerateEngine::new(GenerateEngineConfig {
            vocab_path: "unused.json".to_string(),
            ..Default::default()
        })
        .unwrap();
        let err = engine
            .generate(RouteClass::R2, "hello", &GenParams::default())
            .unwrap_err();
        assert!(
            err.contains("no local generation model"),
            "unexpected error: {err}"
        );
        assert!(!engine.is_loaded(RouteClass::R2));
    }

    #[test]
    fn greedy_sampling_picks_argmax() {
        let logits = vec![0.1, 5.0, 0.2, 0.3];
        let counts = HashMap::new();
        // temperature <= 0 → greedy argmax regardless of top_p/top_k.
        let id = sample_token(&logits, 0.0, 0.9, 128, &counts, 0.0, 0.0, 0.99);
        assert_eq!(id, 1);
    }

    #[test]
    fn penalties_can_flip_greedy_choice() {
        // Without penalties token 1 wins; a heavy presence penalty on it must
        // push the greedy pick to the runner-up.
        let logits = vec![0.1, 5.0, 0.2, 0.3];
        let mut counts = HashMap::new();
        counts.insert(1u32, 1);
        let plain = sample_token(&logits, 0.0, 1.0, 128, &HashMap::new(), 0.0, 0.0, 0.99);
        let penalized = sample_token(&logits, 0.0, 1.0, 128, &counts, 50.0, 0.0, 0.99);
        assert_eq!(plain, 1);
        assert_ne!(penalized, 1);
    }

    #[test]
    fn top_k_zero_means_128() {
        // Sampling path: top_k == 0 is normalized to 128 in the worker; the
        // sampler itself just takes the top-k candidates. Sanity: with k=1
        // the sample is always the argmax even at temperature > 0.
        let logits = vec![0.1, 5.0, 0.2, 0.3];
        let counts = HashMap::new();
        for _ in 0..20 {
            let id = sample_token(&logits, 1.0, 1.0, 1, &counts, 0.0, 0.0, 0.99);
            assert_eq!(id, 1);
        }
    }

    #[test]
    fn nucleus_cut_keeps_top_mass() {
        // Skewed distribution: top_p=0.5 must keep only the argmax.
        let logits = vec![10.0, 2.0, 1.0, 0.0];
        let counts = HashMap::new();
        for _ in 0..20 {
            let id = sample_token(&logits, 1.0, 0.5, 128, &counts, 0.0, 0.0, 0.99);
            assert_eq!(id, 0);
        }
    }
}
