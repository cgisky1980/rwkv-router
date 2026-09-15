//! Classification engine abstraction.
//!
//! The `SmartRouter` orchestrates rules + post-processing around a tier
//! classifier, but never touches a concrete inference runtime. Hosts inject
//! any [`ClassifyEngine`] implementation:
//!
//! - the built-in RWKV engine (`builtin` module, feature `rwkv`) — 0.1B
//!   resident model, mean-pooled hidden state → trained MLP head;
//! - a host-supplied engine bridging any embedding backbone (llama.cpp,
//!   remote embedding API, ...);
//! - test mocks.
//!
//! The trait is **synchronous** and the core crate has zero tokio/platform
//! dependencies: classification runs on the caller's thread (inference is
//! delegated to the engine, which may use its own worker threads internally).
//! Timeouts are the host's responsibility (wrap the call, or configure the
//! built-in engine).

/// Tier classification engine: turns a request into R0-R3 probabilities.
///
/// `classify` returns `(probs, hidden)`:
/// - `probs`: 4 softmax probabilities for R0/R1/R2/R3 (after the MLP head);
/// - `hidden`: the raw embedding vector (num_embd dims) the head consumed —
///   carried through so the evolution loop can capture training samples
///   without re-running inference.
///
/// `capture` marks the call as a *real* routing decision (vs a stateless
/// preview): engines/coordinators may use it to feed the evolution sample
/// store. Preview calls must not pollute the sample library.
pub trait ClassifyEngine: Send + Sync {
    /// Classify `input` into tier probabilities.
    ///
    /// `prev_tier` (0-3) is the previous turn's routed tier; v4+ heads
    /// consume it as a one-hot numeric feature (v1 heads ignore it).
    fn classify(
        &self,
        input: &str,
        prev_tier: Option<u8>,
        capture: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String>;

    /// Embedding dimension of the classification backbone (head input size).
    fn num_embd(&self) -> usize;

    /// Whether the engine is loaded and ready to classify.
    fn is_initialized(&self) -> bool;

    /// Hot-reloads the classification head after the evolution loop re-deploys
    /// it. Default no-op for engines without a hot-reload path (the host then
    /// reloads/restarts on its own); the built-in RWKV engine overrides this.
    fn reload(&self) -> Result<(), String> {
        let _ = self;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use super::ClassifyEngine;

    /// Deterministic engine returning a fixed probability vector — drives the
    /// orchestration tests without any inference runtime.
    pub(crate) struct MockEngine {
        probs: [f32; 4],
        num_embd: usize,
    }

    impl MockEngine {
        pub(crate) fn new(probs: [f32; 4]) -> Self {
            Self { probs, num_embd: 8 }
        }
    }

    impl ClassifyEngine for MockEngine {
        fn classify(
            &self,
            _input: &str,
            _prev_tier: Option<u8>,
            _capture: bool,
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            Ok((self.probs.to_vec(), vec![0.0; self.num_embd]))
        }

        fn num_embd(&self) -> usize {
            self.num_embd
        }

        fn is_initialized(&self) -> bool {
            true
        }
    }
}
