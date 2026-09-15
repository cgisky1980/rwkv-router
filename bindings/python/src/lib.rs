//! Python bindings for RWKV-Router (pyo3; PyPI package `rwkv-router`).
//!
//! Thin surface over [`rwkv_router::RouterSession`]: every structured result
//! (decision / stats / evolve report / generation output) is serialized via
//! serde and returned as a plain Python `dict` — one conversion path, zero
//! per-type shims, automatically in sync with the Rust API.
//!
//! Quickstart:
//!
//! ```python
//! from rwkv_router import RouterSession
//!
//! s = RouterSession()
//! d = s.route("帮我写一首诗")          # -> {"route": "R2", "confidence": .., ...}
//! s.load_classifier("rwkv-0.1b.st", "vocab.json", "router_head_0b.json")
//! s.attach_generation("R0", "vocab.json", "rwkv-0.1b.st")
//! out = s.generate("R0", "你好")       # -> {"text": .., "output_tokens": .., ...}
//! ```

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;

use rwkv_router::{EvolutionConfig, GenParams, RouteClass, RouterSession as CoreSession};

// ---------------------------------------------------------------------------
// serde → Python conversion (single bridge for all structured results)
// ---------------------------------------------------------------------------

fn json_to_py(py: Python<'_>, value: serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py_any(py)?,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)?
            } else if let Some(u) = n.as_u64() {
                u.into_py_any(py)?
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_py_any(py)?
            }
        }
        Value::String(s) => s.into_py_any(py)?,
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.unbind().into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.unbind().into_any()
        }
    })
}

fn to_py<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let json = serde_json::to_value(value)
        .map_err(|e| PyValueError::new_err(format!("serialize failed: {e}")))?;
    json_to_py(py, json)
}

fn parse_tier(tier: &str) -> PyResult<RouteClass> {
    RouteClass::parse_from_str(tier).ok_or_else(|| {
        PyValueError::new_err(format!("invalid tier '{tier}' (expected \"R0\"..\"R3\")"))
    })
}

fn backend_err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

// ---------------------------------------------------------------------------
// RouterSession
// ---------------------------------------------------------------------------

/// Self-evolving smart router session (R0-R3 tiering + optional RWKV
/// classifier/generation pool + optional evolution loop).
///
/// Engines load from local files; build failures raise `RuntimeError`.
/// `route()` never raises for routing itself — without a classifier it
/// falls back to rules (R0 for trivial acknowledgements, R1 otherwise).
#[pyclass]
pub struct RouterSession {
    inner: CoreSession,
}

#[pymethods]
impl RouterSession {
    /// Creates a session with default routing parameters.
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreSession::new(rwkv_router::RouterConfig::default()),
        }
    }

    /// route(input, summary=None, session_id="py", turn_index=None) -> dict
    ///
    /// One routing decision. Sticky-tier context is keyed by `session_id`
    /// (`turn_index` counts turns inside that session). Feeds the evolution
    /// capture store when evolution is configured.
    #[pyo3(signature = (input, summary=None, session_id="py", turn_index=None))]
    fn route(
        &self,
        py: Python<'_>,
        input: &str,
        summary: Option<&str>,
        session_id: &str,
        turn_index: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let decision = py.detach(|| {
            self.inner
                .route(session_id, input, summary, turn_index.unwrap_or(0))
        });
        to_py(py, &decision)
    }

    /// route_preview(input, summary=None) -> dict
    ///
    /// Stateless decision (no sticky table, no capture) — testing entry.
    #[pyo3(signature = (input, summary=None))]
    fn route_preview(
        &self,
        py: Python<'_>,
        input: &str,
        summary: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let decision = py.detach(|| self.inner.route_preview(input, summary));
        to_py(py, &decision)
    }

    /// load_classifier(model, vocab, head, timeout_ms=5000)
    ///
    /// Attaches the built-in RWKV classifier (resident 0.1B + trained MLP
    /// head). `timeout_ms` bounds each classify call.
    #[pyo3(signature = (model, vocab, head, timeout_ms=5000))]
    fn load_classifier(
        &mut self,
        model: &str,
        vocab: &str,
        head: &str,
        timeout_ms: u64,
    ) -> PyResult<()> {
        self.inner
            .load_classifier(model, vocab, head, timeout_ms)
            .map_err(backend_err)
    }

    /// attach_generation(tier, vocab, model, max_loaded=1)
    ///
    /// Maps a tier ("R0".."R3") to a local RWKV model for embedded
    /// generation. Models load lazily on first `generate()`; `max_loaded`
    /// bounds the LRU pool of concurrently resident models.
    #[pyo3(signature = (tier, vocab, model, max_loaded=1))]
    fn attach_generation(
        &mut self,
        tier: &str,
        vocab: &str,
        model: &str,
        max_loaded: usize,
    ) -> PyResult<()> {
        let tier = parse_tier(tier)?;
        self.inner
            .attach_generation(tier, vocab, model, max_loaded)
            .map_err(backend_err)
    }

    /// generate(tier, prompt, *, max_tokens=None, temperature=None,
    ///          top_p=None, top_k=None, presence_penalty=None,
    ///          frequency_penalty=None, stop=None) -> dict
    ///
    /// Generates with the tier's attached model. Unspecified sampling
    /// parameters use engine defaults (512 tokens, temp 1.0, top_p 0.9).
    #[pyo3(signature = (tier, prompt, *, max_tokens=None, temperature=None,
                        top_p=None, top_k=None, presence_penalty=None,
                        frequency_penalty=None, stop=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        tier: &str,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        stop: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let tier = parse_tier(tier)?;
        let mut params = GenParams::default();
        if let Some(v) = max_tokens {
            params.max_tokens = v;
        }
        if let Some(v) = temperature {
            params.temperature = v;
        }
        if let Some(v) = top_p {
            params.top_p = v;
        }
        if let Some(v) = top_k {
            params.top_k = v;
        }
        if let Some(v) = presence_penalty {
            params.presence_penalty = v;
        }
        if let Some(v) = frequency_penalty {
            params.frequency_penalty = v;
        }
        if let Some(v) = stop {
            params.stop = v;
        }
        let output = py
            .detach(move || self.inner.generate(tier, prompt, &params))
            .map_err(backend_err)?;
        to_py(py, &output)
    }

    /// configure_evolution(data_dir, *, head_path=None, packs_dir=None,
    ///                     capture_limit=None, min_labeled_for_evolve=None,
    ///                     auto_evolve_step=None)
    ///
    /// Enables the self-evolution loop: capture -> label -> AdamW fine-tune
    /// -> eval-pack gate -> head hot-reload. `packs_dir` (holding
    /// `eval_pack.json`) is required to actually run `evolve()`.
    #[pyo3(signature = (data_dir, *, head_path=None, packs_dir=None,
                        capture_limit=None, min_labeled_for_evolve=None,
                        auto_evolve_step=None))]
    #[allow(clippy::too_many_arguments)]
    fn configure_evolution(
        &mut self,
        data_dir: String,
        head_path: Option<String>,
        packs_dir: Option<String>,
        capture_limit: Option<usize>,
        min_labeled_for_evolve: Option<usize>,
        auto_evolve_step: Option<usize>,
    ) -> PyResult<()> {
        let mut config = EvolutionConfig {
            data_dir: PathBuf::from(data_dir),
            head_path: head_path.map(PathBuf::from).unwrap_or_default(),
            packs_dir: packs_dir.map(PathBuf::from),
            ..Default::default()
        };
        if let Some(v) = capture_limit {
            config.capture_limit = v;
        }
        if let Some(v) = min_labeled_for_evolve {
            config.min_labeled_for_evolve = v;
        }
        if let Some(v) = auto_evolve_step {
            config.auto_evolve_step = v;
        }
        self.inner.configure_evolution(config).map_err(backend_err)
    }

    /// capture_stats() -> dict
    ///
    /// Capture-store statistics (total / labelled / per-tier counts).
    fn capture_stats<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let stats = self.inner.capture_stats().map_err(backend_err)?;
        to_py(py, &stats)
    }

    /// capture_list(offset=0, limit=50) -> list[dict]
    ///
    /// Captured samples (text + current label + probs) for labeling flows.
    #[pyo3(signature = (offset=0, limit=50))]
    fn capture_list(&self, py: Python<'_>, offset: usize, limit: usize) -> PyResult<Py<PyAny>> {
        let items = self
            .inner
            .capture_list(offset, limit)
            .map_err(backend_err)?;
        to_py(py, &items)
    }

    /// capture_label(idx, tier=None)
    ///
    /// Labels the sample at `idx` with `tier` ("R0".."R3"); `tier=None`
    /// clears the label.
    #[pyo3(signature = (idx, tier=None))]
    fn capture_label(&self, idx: usize, tier: Option<&str>) -> PyResult<()> {
        let label = match tier {
            Some(t) => Some(parse_tier(t)?.index() as u8),
            None => None,
        };
        self.inner.capture_label(idx, label).map_err(backend_err)
    }

    /// evolve() -> dict
    ///
    /// Runs one evolution cycle (blocking): fine-tune on labelled samples,
    /// gate against the eval pack (deploys only if accuracy does not
    /// regress), backup + hot-reload the head.
    fn evolve(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = py.detach(|| self.inner.evolve()).map_err(backend_err)?;
        to_py(py, &result)
    }
}

// Explicit module name: the fn name must differ from the extern crate
// `rwkv_router` to avoid a name-resolution ambiguity inside this crate.
#[pymodule(name = "rwkv_router")]
fn py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RouterSession>()?;
    Ok(())
}
