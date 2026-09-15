//! Face 2 — decision & evolution API (programmatic access, plan §3.3).

use std::collections::HashMap;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::proxy::AUTO_MODEL;
use crate::state::SharedState;

/// GET /v1/health.
pub async fn health(State(state): State<SharedState>) -> Response {
    let classifier_loaded = state.config.router.classifier.is_some();
    let mut tiers = HashMap::new();
    for tier in [
        rwkv_router::RouteClass::R0,
        rwkv_router::RouteClass::R1,
        rwkv_router::RouteClass::R2,
        rwkv_router::RouteClass::R3,
    ] {
        if let Some(u) = state.config.tiers.entry(tier) {
            tiers.insert(tier.to_string(), u.kind());
        }
    }
    let body = json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "classifier_configured": classifier_loaded,
        "evolution_configured": state.session.evolution().is_some(),
        "tiers": tiers,
        "fallback": state.config.fallback.as_ref().map(|u| u.kind()),
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// GET /v1/models — logical model list (OpenAI SDK format).
pub async fn models(State(_state): State<SharedState>) -> Response {
    let data: Vec<Value> = [AUTO_MODEL, "R0", "R1", "R2", "R3"]
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "rwkv-router"
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(json!({"object": "list", "data": data})),
    )
        .into_response()
}

/// POST /v1/route {session_id?, input, summary?, turn_index?}.
pub async fn route(State(state): State<SharedState>, Json(body): Json<Value>) -> Response {
    let Ok(input) = serde_json::from_value::<RouteRequest>(body) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok": false, "error": "expected {input: string, session_id?: string, summary?: string, turn_index?: number}"})),
        )
            .into_response();
    };
    let session = state.session.clone();
    let decision = tokio::task::spawn_blocking(move || {
        session.route(
            &input.session_id,
            &input.input,
            input.summary.as_deref(),
            input.turn_index.unwrap_or(0),
        )
    })
    .await
    .unwrap_or_else(|e| {
        log::error!("[api] route task panicked: {e}");
        rwkv_router::fallback_decision()
    });
    match serde_json::to_value(&decision) {
        Ok(v) => (StatusCode::OK, Json(json!({"ok": true, "data": v}))).into_response(),
        Err(e) => internal_error(e.to_string()),
    }
}

#[derive(Deserialize)]
struct RouteRequest {
    #[serde(default = "default_session_id")]
    session_id: String,
    input: String,
    summary: Option<String>,
    turn_index: Option<usize>,
}

fn default_session_id() -> String {
    "api".to_string()
}

/// GET /v1/capture/stats.
pub async fn capture_stats(State(state): State<SharedState>) -> Response {
    match state.session.capture_stats() {
        Ok(stats) => match serde_json::to_value(&stats) {
            Ok(v) => (StatusCode::OK, Json(json!({"ok": true, "data": v}))).into_response(),
            Err(e) => internal_error(e.to_string()),
        },
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": e})),
        )
            .into_response(),
    }
}

/// GET /v1/capture/list?offset=0&limit=50.
pub async fn capture_list(
    State(state): State<SharedState>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let offset = q.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50);
    match state.session.capture_list(offset, limit) {
        Ok(items) => match serde_json::to_value(&items) {
            Ok(v) => (StatusCode::OK, Json(json!({"ok": true, "data": v}))).into_response(),
            Err(e) => internal_error(e.to_string()),
        },
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": e})),
        )
            .into_response(),
    }
}

/// POST /v1/capture/label {idx, label} — label < 0 clears.
pub async fn capture_label(State(state): State<SharedState>, Json(body): Json<Value>) -> Response {
    let parsed: Result<(usize, i64), String> = (|| {
        let idx = body
            .get("idx")
            .and_then(Value::as_u64)
            .ok_or("missing idx")? as usize;
        let label = body
            .get("label")
            .and_then(Value::as_i64)
            .ok_or("missing label")?;
        Ok((idx, label))
    })();
    let (idx, label) = match parsed {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"ok": false, "error": e})),
            )
                .into_response()
        }
    };
    let label = if (0..=3).contains(&label) {
        Some(label as u8)
    } else {
        None // negative or out-of-range clears the label
    };
    match state.session.capture_label(idx, label) {
        Ok(()) => (StatusCode::OK, Json(json!({"ok": true}))).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": e})),
        )
            .into_response(),
    }
}

/// POST /v1/evolve — starts the (blocking) cycle on a worker thread, 202.
pub async fn evolve(State(state): State<SharedState>) -> Response {
    if !state.evolve.begin() {
        return (
            StatusCode::CONFLICT,
            Json(json!({"ok": false, "error": "evolution already running"})),
        )
            .into_response();
    }
    let session = state.session.clone();
    let evolve_status = state.evolve.clone();
    let spawned = std::thread::Builder::new()
        .name("rwkv-router-evolve".to_string())
        .spawn(move || {
            let result = session.evolve().inspect_err(|e| {
                log::warn!("[api] evolve failed: {e}");
            });
            if let Ok(result) = &result {
                log::info!("[api] evolve finished: status={}", result.status);
            }
            // A failed call still records an outcome (status field explains).
            evolve_status.finish(match result {
                Ok(r) => r,
                Err(e) => rwkv_router::EvolveResult {
                    status: "error".to_string(),
                    message: e,
                    ..Default::default()
                },
            });
        });
    match spawned {
        Ok(_) => (
            StatusCode::ACCEPTED,
            Json(json!({"ok": true, "data": {"started": true}})),
        )
            .into_response(),
        Err(e) => {
            state.evolve.finish(rwkv_router::EvolveResult {
                status: "error".to_string(),
                message: format!("failed to spawn evolve thread: {e}"),
                ..Default::default()
            });
            internal_error(format!("failed to spawn evolve thread: {e}"))
        }
    }
}

/// GET /v1/evolve/status.
pub async fn evolve_status(State(state): State<SharedState>) -> Response {
    let running = state.evolve.is_running();
    let last = state
        .evolve
        .last_report()
        .map(|r| serde_json::to_value(&r).ok())
        .unwrap_or(None);
    (
        StatusCode::OK,
        Json(json!({"ok": true, "data": {"running": running, "last": last}})),
    )
        .into_response()
}

fn internal_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"ok": false, "error": message})),
    )
        .into_response()
}
