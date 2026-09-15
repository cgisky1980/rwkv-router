//! C ABI (feature `ffi`): JSON-in / JSON-out exports over an opaque
//! [`RouterSession`] handle, for FFI languages beyond Python/Node (which use
//! pyo3/napi directly, see plan §3.2/D5).
//!
//! ## Conventions
//!
//! - **Handles** (`RouterHandle`): created by [`rwkv_router_new`], freed by
//!   [`rwkv_router_free`]. Handle-taking functions return `NULL`/`-1` on
//!   failure and set the thread-local last error, readable via
//!   [`rwkv_router_last_error`].
//! - **Strings**: returned via `*mut c_char` (Rust allocator), ALWAYS a JSON
//!   envelope — `{"ok":true,"data":...}` or `{"ok":false,"error":"..."}` —
//!   so callers never parse NULL. Free with [`rwkv_router_string_free`].
//! - **Panics** never cross the boundary (all exports are wrapped in
//!   `catch_unwind` and surface as error envelopes).
//! - Evolution calls (`rwkv_router_evolve`) block: hosts run them on their
//!   own thread.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use serde_json::{json, Value};

use crate::config::RouterConfig;
use crate::evolution::EvolutionConfig;
use crate::session::RouterSession;
use crate::tier::RouteClass;

/// Opaque session handle.
pub type RouterHandle = *mut RouterSession;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn take_last_error() -> Option<CString> {
    LAST_ERROR.with(|e| e.borrow_mut().take())
}

/// Null-terminated UTF-8 slice or error.
unsafe fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("required string argument is NULL".to_string());
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|e| format!("invalid UTF-8 string argument: {e}"))
}

/// Serializes into an ok/error envelope and hands ownership to the caller.
fn envelope_ptr(res: Result<Value, String>) -> *mut c_char {
    let envelope = match res {
        Ok(data) => json!({ "ok": true, "data": data }),
        Err(error) => {
            log::warn!("[ffi] error envelope: {error}");
            json!({ "ok": false, "error": error })
        }
    };
    match CString::new(envelope.to_string()) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(), // interior NUL is impossible in to_string()
    }
}

/// Runs `f`, converting panics into errors (no unwinding across the ABI).
fn guard(f: impl FnOnce() -> Result<Value, String>) -> *mut c_char {
    envelope_ptr(match catch_unwind(AssertUnwindSafe(f)) {
        Ok(res) => res,
        Err(p) => Err(match p.downcast_ref::<&str>() {
            Some(s) => format!("panic in rwkv-router call: {s}"),
            None => "panic in rwkv-router call".to_string(),
        }),
    })
}

/// Wraps a handle-taking body: null-handle + panic guards, int return.
fn guard_handle_int(
    handle: RouterHandle,
    f: impl FnOnce(&mut RouterSession) -> Result<(), String>,
) -> i32 {
    let body = move || -> Result<(), String> {
        let session = unsafe { handle_as_mut(handle) }?;
        f(session)
    };
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(&e);
            -1
        }
        Err(_) => {
            set_last_error("panic in rwkv-router call");
            -1
        }
    }
}

unsafe fn handle_as_mut<'a>(handle: RouterHandle) -> Result<&'a mut RouterSession, String> {
    if handle.is_null() {
        return Err("null router handle".to_string());
    }
    Ok(&mut *handle)
}

unsafe fn handle_as_ref<'a>(handle: RouterHandle) -> Result<&'a RouterSession, String> {
    if handle.is_null() {
        return Err("null router handle".to_string());
    }
    Ok(&*handle)
}

fn tier_from_str(s: &str) -> Result<RouteClass, String> {
    RouteClass::parse_from_str(s).ok_or_else(|| format!("invalid tier '{s}' (expected R0-R3)"))
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Creates a session from a [`RouterConfig`] JSON. Returns NULL on failure
/// (see `rwkv_router_last_error`).
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_new(config_json: *const c_char) -> RouterHandle {
    let result = catch_unwind(|| -> Result<Box<RouterSession>, String> {
        let s = unsafe { cstr(config_json) }?;
        let config: RouterConfig =
            serde_json::from_str(s).map_err(|e| format!("invalid RouterConfig JSON: {e}"))?;
        Ok(Box::new(RouterSession::new(config)))
    });
    match result {
        Ok(Ok(session)) => Box::into_raw(session),
        Ok(Err(e)) => {
            set_last_error(&e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic in rwkv_router_new");
            ptr::null_mut()
        }
    }
}

/// Frees a session (NULL is a no-op).
///
/// # Safety
/// `handle` must be a handle from [`rwkv_router_new`] not yet freed, and
/// must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_free(handle: RouterHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Crate version (static string, do not free).
#[no_mangle]
pub extern "C" fn rwkv_router_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Reads and clears the last error on this thread (may return NULL).
#[no_mangle]
pub extern "C" fn rwkv_router_last_error() -> *const c_char {
    match take_last_error() {
        Some(c) => c.into_raw() as *const c_char,
        None => ptr::null(),
    }
}

/// Frees a string returned by this library (NULL is a no-op).
///
/// # Safety
/// `ptr` must be a pointer returned by this library's string-returning
/// exports and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

// ---------------------------------------------------------------------------
// Engines (feature rwkv)
// ---------------------------------------------------------------------------

/// Loads the built-in 0.1B classifier (model + vocab + head) into the
/// session. 0 on success, -1 on failure (see `rwkv_router_last_error`).
#[cfg(feature = "rwkv")]
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_load_classifier(
    handle: RouterHandle,
    model_path: *const c_char,
    vocab_path: *const c_char,
    head_path: *const c_char,
) -> i32 {
    guard_handle_int(handle, |session| {
        let model = unsafe { cstr(model_path)? };
        let vocab = unsafe { cstr(vocab_path)? };
        let head = unsafe { cstr(head_path)? };
        session.load_classifier(model, vocab, head, 0)
    })
}

/// Attaches the tier generation pool entry (`tier` = "R0".."R3").
/// 0 on success, -1 on failure (see `rwkv_router_last_error`).
#[cfg(feature = "rwkv")]
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_attach_generation(
    handle: RouterHandle,
    tier: *const c_char,
    vocab_path: *const c_char,
    model_path: *const c_char,
) -> i32 {
    guard_handle_int(handle, |session| {
        let tier = tier_from_str(unsafe { cstr(tier)? })?;
        let vocab = unsafe { cstr(vocab_path)? };
        let model = unsafe { cstr(model_path)? };
        session.attach_generation(tier, vocab, model, 1)
    })
}

/// Generates from the tier's model. Params JSON (all optional, defaults
/// apply): `{"max_tokens":512,"temperature":1.0,"top_p":0.9,"top_k":128,
/// "presence_penalty":0,"frequency_penalty":0,"stop":["..."],"stop_on_eos":true}`.
/// Blocking (model loads lazily on first use).
#[cfg(feature = "rwkv")]
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_generate(
    handle: RouterHandle,
    tier: *const c_char,
    prompt: *const c_char,
    params_json: *const c_char,
) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let tier = tier_from_str(unsafe { cstr(tier)? })?;
        let prompt = unsafe { cstr(prompt)? };
        let params: crate::builtin::GenParams = if params_json.is_null() {
            Default::default()
        } else {
            serde_json::from_str(unsafe { cstr(params_json)? })
                .map_err(|e| format!("invalid GenParams JSON: {e}"))?
        };
        let out = session.generate(tier, prompt, &params)?;
        Ok(serde_json::to_value(&out).map_err(|e| e.to_string())?)
    })
}

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

/// Configures the evolution loop from an [`EvolutionConfig`] JSON (all
/// fields optional). 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_configure_evolution(
    handle: RouterHandle,
    config_json: *const c_char,
) -> i32 {
    guard_handle_int(handle, |session| {
        let raw: Value = if config_json.is_null() {
            json!({})
        } else {
            serde_json::from_str(unsafe { cstr(config_json)? })
                .map_err(|e| format!("invalid EvolutionConfig JSON: {e}"))?
        };
        let base = EvolutionConfig::default();
        let data_dir = raw
            .get("data_dir")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| base.data_dir.display().to_string());
        let head_path = raw
            .get("head_path")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| base.head_path.display().to_string());
        let packs_dir = raw.get("packs_dir").and_then(Value::as_str);
        let config = EvolutionConfig {
            data_dir: data_dir.into(),
            head_path: head_path.into(),
            packs_dir: packs_dir.map(std::path::PathBuf::from),
            capture_limit: raw
                .get("capture_limit")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(base.capture_limit),
            min_labeled_for_evolve: raw
                .get("min_labeled_for_evolve")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(base.min_labeled_for_evolve),
            auto_evolve_step: raw
                .get("auto_evolve_step")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(base.auto_evolve_step),
        };
        session.configure_evolution(config)
    })
}

/// Captured-sample stats.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_capture_stats(handle: RouterHandle) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let stats = session.capture_stats()?;
        Ok(serde_json::to_value(&stats).map_err(|e| e.to_string())?)
    })
}

/// Lists captured samples `[{idx,ts,text,probs,prev_tier,label,source}, ...]`.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_capture_list(
    handle: RouterHandle,
    offset: usize,
    limit: usize,
) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let items = session.capture_list(offset, limit)?;
        Ok(serde_json::to_value(&items).map_err(|e| e.to_string())?)
    })
}

/// Labels (or clears with `label < 0`) the sample at `idx`.
/// 0 on success, -1 on failure.
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_capture_label(
    handle: RouterHandle,
    idx: usize,
    label: i32,
) -> i32 {
    guard_handle_int(handle, |session| {
        let label = if label < 0 { None } else { Some(label as u8) };
        session.capture_label(idx, label)
    })
}

/// Runs one evolution cycle (blocking: fine-tune + eval gate + deploy).
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_evolve(handle: RouterHandle) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let result = session.evolve()?;
        Ok(serde_json::to_value(&result).map_err(|e| e.to_string())?)
    })
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/// Routes one request. `summary` may be NULL. Returns a Decision envelope
/// (`{"ok":true,"data":{...RoutingDecision...}}`).
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_route(
    handle: RouterHandle,
    session_id: *const c_char,
    user_input: *const c_char,
    summary: *const c_char,
    turn_index: usize,
) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let session_id = unsafe { cstr(session_id)? };
        let user_input = unsafe { cstr(user_input)? };
        let summary = if summary.is_null() {
            None
        } else {
            Some(unsafe { cstr(summary)? })
        };
        let decision = session.route(session_id, user_input, summary, turn_index);
        Ok(serde_json::to_value(&decision).map_err(|e| e.to_string())?)
    })
}

/// Stateless preview (no sticky-table / capture side effects).
#[no_mangle]
pub unsafe extern "C" fn rwkv_router_route_preview(
    handle: RouterHandle,
    user_input: *const c_char,
    summary: *const c_char,
) -> *mut c_char {
    guard(move || {
        let session = unsafe { handle_as_ref(handle)? };
        let user_input = unsafe { cstr(user_input)? };
        let summary = if summary.is_null() {
            None
        } else {
            Some(unsafe { cstr(summary)? })
        };
        let decision = session.route_preview(user_input, summary);
        Ok(serde_json::to_value(&decision).map_err(|e| e.to_string())?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_shapes() {
        let ok = envelope_ptr(Ok(json!({ "route": "R0" })));
        let ok_s = unsafe { CStr::from_ptr(ok) }.to_str().unwrap();
        assert!(ok_s.contains("\"ok\":true") && ok_s.contains("R0"));
        unsafe { rwkv_router_string_free(ok) };

        let err = envelope_ptr(Err("boom".to_string()));
        let err_s = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
        assert!(err_s.contains("\"ok\":false") && err_s.contains("boom"));
        unsafe { rwkv_router_string_free(err) };
    }

    #[test]
    fn session_lifecycle_without_engines() {
        let cfg = serde_json::to_string(&RouterConfig::default()).unwrap();
        let cfg_c = CString::new(cfg).unwrap();
        let handle = unsafe { rwkv_router_new(cfg_c.as_ptr()) };
        assert!(!handle.is_null());

        // Route envelope works without any engine (fallback decision).
        let sid = CString::new("s1").unwrap();
        let input = CString::new("help me debug this traceback").unwrap();
        let out =
            unsafe { rwkv_router_route(handle, sid.as_ptr(), input.as_ptr(), ptr::null(), 0) };
        let out_s = unsafe { CStr::from_ptr(out) }.to_str().unwrap();
        assert!(out_s.contains("\"ok\":true"), "unexpected: {out_s}");
        assert!(out_s.contains("Fallback"));
        unsafe { rwkv_router_string_free(out) };

        // Evolution unconfigured -> error envelope.
        let out = unsafe { rwkv_router_evolve(handle) };
        let out_s = unsafe { CStr::from_ptr(out) }.to_str().unwrap();
        assert!(out_s.contains("\"ok\":false"), "unexpected: {out_s}");
        unsafe { rwkv_router_string_free(out) };

        // Evolution configure -> label/stats work (empty store).
        let evo = CString::new("{\"data_dir\":\"ffi-test-evolution\"}").unwrap();
        assert_eq!(
            unsafe { rwkv_router_configure_evolution(handle, evo.as_ptr()) },
            0
        );
        let out = unsafe { rwkv_router_capture_stats(handle) };
        let out_s = unsafe { CStr::from_ptr(out) }.to_str().unwrap();
        assert!(out_s.contains("\"total\":0"), "unexpected: {out_s}");
        unsafe { rwkv_router_string_free(out) };
        assert_eq!(unsafe { rwkv_router_capture_label(handle, 0, -1) }, -1); // empty store

        unsafe { rwkv_router_free(handle) };

        // Null handle errors land in last_error.
        assert_eq!(
            unsafe { rwkv_router_capture_label(ptr::null_mut(), 0, 1) },
            -1
        );
        let e = rwkv_router_last_error();
        assert!(!e.is_null());
        unsafe { rwkv_router_string_free(e as *mut c_char) };
    }

    #[test]
    fn invalid_config_json_returns_null() {
        let bad = CString::new("not json").unwrap();
        assert!(unsafe { rwkv_router_new(bad.as_ptr()) }.is_null());
        let e = rwkv_router_last_error();
        assert!(!e.is_null());
        unsafe { rwkv_router_string_free(e as *mut c_char) };
    }
}
