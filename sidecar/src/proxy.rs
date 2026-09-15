//! Face 1 — OpenAI/Anthropic-compatible proxy with tier routing (plan §3.3).
//!
//! Semantics (D8/D9): same-protocol passthrough, model rewrite on the
//! upstream request only; responses relay untouched. `model = "ai00-auto"`
//! (default) classifies the last user message into R0–R3 and picks the
//! tier's upstream; explicit `R0`–`R3` model aliases force a tier; any other
//! model name passes through to the `fallback` upstream unchanged. Request/
//! response bodies are never persisted — only routing decision metadata
//! (tier/source/latency) reaches the capture store.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use serde_json::{json, Value};

use rwkv_router::{DecisionSource, RouteClass};

use crate::config::UpstreamConfig;
use crate::state::SharedState;

/// Default model alias that triggers classification.
pub const AUTO_MODEL: &str = "ai00-auto";

/// Wire protocol of the incoming request (and thus of its upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Openai,
    Anthropic,
}

impl Protocol {
    fn error_body(&self, message: &str) -> Value {
        match self {
            Protocol::Openai => json!({
                "error": {"message": message, "type": "gateway_error", "code": "rwkv_router_gateway"}
            }),
            Protocol::Anthropic => json!({
                "type": "error",
                "error": {"type": "gateway_error", "message": message}
            }),
        }
    }
}

/// Entry: POST /v1/chat/completions.
pub async fn openai_chat(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(state, headers, body, Protocol::Openai).await
}

/// Entry: POST /v1/messages.
pub async fn anthropic_messages(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(state, headers, body, Protocol::Anthropic).await
}

async fn handle(
    state: SharedState,
    headers: HeaderMap,
    body: Bytes,
    protocol: Protocol,
) -> Response {
    let started = Instant::now();
    let request: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                &state,
                protocol,
                StatusCode::BAD_REQUEST,
                None,
                None,
                &format!("invalid request body: {e}"),
                started,
            );
        }
    };

    let requested_model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(AUTO_MODEL)
        .to_string();
    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ---- Routing decision -------------------------------------------------
    let forced_tier = parse_tier_alias(&requested_model);
    let (decision, tier) = if let Some(tier) = forced_tier {
        (None, Some(tier))
    } else if requested_model == AUTO_MODEL {
        let Some(input) = last_user_text(protocol, &request) else {
            return error_response(
                &state,
                protocol,
                StatusCode::BAD_REQUEST,
                None,
                None,
                "request has no user message to classify",
                started,
            );
        };
        let session_id = headers
            .get("x-session-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("gateway")
            .to_string();
        let turn_index = message_count(protocol, &request);
        let summary = None; // v1: rolling summaries are a client-side feature
        let session = state.session.clone();
        let input_cloned = input.clone();
        let decision = tokio::task::spawn_blocking(move || {
            session.route(&session_id, &input_cloned, summary, turn_index)
        })
        .await
        .unwrap_or_else(|e| {
            log::error!("[proxy] route task panicked: {e}");
            rwkv_router::fallback_decision()
        });
        (Some(decision.clone()), Some(decision.route))
    } else {
        // Unknown model name → passthrough to fallback upstream, unchanged.
        (None, None)
    };

    // ---- Upstream resolution ---------------------------------------------
    let upstream = match tier {
        Some(_t)
            if decision
                .as_ref()
                .is_some_and(|d| d.source == DecisionSource::Fallback) =>
        {
            state.config.fallback.as_ref().ok_or_else(|| {
                "classifier fell back and no fallback upstream is configured".to_string()
            })
        }
        Some(t) => state
            .config
            .tiers
            .entry(t)
            .or(state.config.fallback.as_ref())
            .ok_or_else(|| format!("{t} has no upstream configured and no fallback exists")),
        None => state.config.fallback.as_ref().ok_or_else(|| {
            format!(
                "model '{requested_model}' requires the fallback upstream, which is not configured"
            )
        }),
    };
    let upstream = match upstream {
        Ok(u) => u,
        Err(message) => {
            return error_response(
                &state,
                protocol,
                StatusCode::SERVICE_UNAVAILABLE,
                tier_label(decision.as_ref(), tier),
                None,
                &message,
                started,
            );
        }
    };

    let source_label =
        tier_label(decision.as_ref(), tier).unwrap_or_else(|| "passthrough".to_string());
    log::info!(
        "[proxy] {} model={requested_model} tier={source_label} upstream={} stream={stream}",
        protocol_label(protocol),
        upstream.kind()
    );

    // ---- Dispatch ---------------------------------------------------------
    let mut response = match upstream {
        UpstreamConfig::BuiltinRwkv { model, tokenizer } => {
            handle_builtin(
                &state,
                protocol,
                &request,
                stream,
                tier.unwrap_or(RouteClass::R1),
                model,
                tokenizer,
                &requested_model,
                started,
            )
            .await
        }
        UpstreamConfig::Openai { .. } | UpstreamConfig::Anthropic { .. } => {
            handle_http_upstream(
                &state,
                protocol,
                &headers,
                &request,
                stream,
                upstream,
                &requested_model,
                started,
            )
            .await
        }
    };

    // Metadata headers (never disturb the payload).
    let latency_ms = started.elapsed().as_millis() as u64;
    let h = response.headers_mut();
    if let Ok(v) = header::HeaderValue::from_str(&source_label) {
        h.insert("x-ai00-tier", v);
    }
    if let Ok(v) = header::HeaderValue::from_str(upstream.kind()) {
        h.insert("x-ai00-upstream", v);
    }
    if let Ok(v) = header::HeaderValue::from_str(&latency_ms.to_string()) {
        h.insert("x-ai00-latency-ms", v);
    }
    response
}

/// builtin-rwkv generation: format the conversation into a RWKV chat prompt
/// and generate locally (blocking pool; models load lazily via the LRU pool).
/// When `stream` is requested the finished text is emitted as an emulated
/// SSE sequence (client-compatible deltas; real token streaming is v2).
#[allow(clippy::too_many_arguments)]
async fn handle_builtin(
    state: &SharedState,
    protocol: Protocol,
    request: &Value,
    stream: bool,
    tier: RouteClass,
    model_path: &str,
    tokenizer_path: &str,
    requested_model: &str,
    started: Instant,
) -> Response {
    let messages = extract_messages(protocol, request);
    let mut params = gen_params_from_request(protocol, request);
    params.stop.push("\nUser:".to_string());
    params.stop.push("\nSystem:".to_string());

    let session = state.session.clone();
    let tier_label = tier.to_string();
    let prompt = build_rwkv_prompt(&messages);
    let result = tokio::task::spawn_blocking(move || {
        // Attach is idempotent at startup; if the gateway started without
        // this tier's model the error surfaces here with a clear message.
        session
            .generate(tier, &prompt, &params)
            .map_err(|e| format!("builtin generation failed ({tier_label}): {e}"))
    })
    .await
    .unwrap_or_else(|e| Err(format!("generation task panicked: {e}")));

    let text = match result {
        Ok(out) => out.text,
        Err(message) => {
            return error_response(
                state,
                protocol,
                StatusCode::BAD_GATEWAY,
                Some(tier.to_string()),
                Some("builtin-rwkv"),
                &message,
                started,
            );
        }
    };
    log::info!("[proxy] builtin generated {} chars", text.len());

    let _ = (model_path, tokenizer_path); // paths are startup-only config
    let latency_ms = started.elapsed().as_millis() as u64;
    if stream {
        emulated_sse(protocol, requested_model, &text, latency_ms)
    } else {
        let body = match protocol {
            Protocol::Openai => openai_completion_json(requested_model, &text),
            Protocol::Anthropic => anthropic_message_json(requested_model, &text),
        };
        json_response(StatusCode::OK, &body)
    }
}

/// HTTP upstream forwarding: rewrite the request's `model` to the upstream
/// model, forward, and relay status/body — byte-for-byte for SSE.
#[allow(clippy::too_many_arguments)]
async fn handle_http_upstream(
    state: &SharedState,
    protocol: Protocol,
    inbound_headers: &HeaderMap,
    request: &Value,
    stream: bool,
    upstream: &UpstreamConfig,
    requested_model: &str,
    started: Instant,
) -> Response {
    let _ = requested_model; // response passthrough keeps the upstream model
    let (url, upstream_model, api_key_env) = match upstream {
        UpstreamConfig::Openai {
            base_url,
            model,
            api_key_env,
        } => (
            format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model.clone(),
            api_key_env.clone(),
        ),
        UpstreamConfig::Anthropic {
            base_url,
            model,
            api_key_env,
        } => {
            let base = base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{base}/messages")
            } else {
                format!("{base}/v1/messages")
            };
            (url, model.clone(), api_key_env.clone())
        }
        UpstreamConfig::BuiltinRwkv { .. } => unreachable!("dispatch guarantees HTTP upstream"),
    };

    let Some(api_key) = std::env::var(&api_key_env).ok().filter(|k| !k.is_empty()) else {
        return error_response(
            state,
            protocol,
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
            Some(upstream.kind()),
            &format!(
                "environment variable {api_key_env} is not set (required for the upstream API key)"
            ),
            started,
        );
    };

    // Rewrite only the model field; everything else passes through as-is.
    let mut upstream_body = request.clone();
    upstream_body["model"] = Value::String(upstream_model);

    let mut req = state
        .http
        .post(&url)
        .header(header::CONTENT_TYPE, "application/json");
    match upstream {
        UpstreamConfig::Openai { .. } => {
            req = req.bearer_auth(&api_key);
        }
        UpstreamConfig::Anthropic { .. } => {
            req = req.header("x-api-key", &api_key).header(
                "anthropic-version",
                inbound_headers
                    .get("anthropic-version")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("2023-06-01"),
            );
        }
        UpstreamConfig::BuiltinRwkv { .. } => unreachable!(),
    }

    let upstream_response = match req.json(&upstream_body).send().await {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                state,
                protocol,
                StatusCode::BAD_GATEWAY,
                None,
                Some(upstream.kind()),
                &format!("upstream request failed: {e}"),
                started,
            );
        }
    };

    let status = upstream_response.status();
    let axum_status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream_response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if stream && content_type.contains("text/event-stream") {
        // Byte-for-byte SSE passthrough.
        let body = Body::from_stream(upstream_response.bytes_stream());
        Response::builder()
            .status(axum_status)
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .unwrap_or_else(|e| plain_error(protocol, StatusCode::BAD_GATEWAY, &e.to_string()))
    } else {
        // Non-streaming (or upstream ignored stream): relay the body.
        match upstream_response.bytes().await {
            Ok(bytes) => Response::builder()
                .status(axum_status)
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(bytes))
                .unwrap_or_else(|e| plain_error(protocol, StatusCode::BAD_GATEWAY, &e.to_string())),
            Err(e) => error_response(
                state,
                protocol,
                StatusCode::BAD_GATEWAY,
                None,
                Some(upstream.kind()),
                &format!("upstream body read failed: {e}"),
                started,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Request parsing helpers
// ---------------------------------------------------------------------------

/// "R0".."R3" model aliases force a tier; anything else is not a tier.
fn parse_tier_alias(model: &str) -> Option<RouteClass> {
    match model.to_ascii_uppercase().as_str() {
        "R0" => Some(RouteClass::R0),
        "R1" => Some(RouteClass::R1),
        "R2" => Some(RouteClass::R2),
        "R3" => Some(RouteClass::R3),
        _ => None,
    }
}

/// OpenAI `content` is a string or `[{type:"text", text}]` parts.
fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let is_text = p.get("type").and_then(Value::as_str) == Some("text");
                is_text.then(|| p.get("text").and_then(Value::as_str).unwrap_or_default())
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn last_user_text(protocol: Protocol, request: &Value) -> Option<String> {
    let messages = extract_messages(protocol, request);
    messages
        .iter()
        .rev()
        .find(|(role, _)| role == "user")
        .map(|(_, text)| text.clone())
        .filter(|t| !t.trim().is_empty())
}

fn message_count(protocol: Protocol, request: &Value) -> usize {
    extract_messages(protocol, request).len()
}

/// Normalized (role, text) message list across both protocols; the Anthropic
/// top-level `system` string becomes a leading system message.
fn extract_messages(protocol: Protocol, request: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match protocol {
        Protocol::Anthropic => {
            if let Some(system) = request.get("system") {
                let text = content_to_text(system);
                if !text.is_empty() {
                    out.push(("system".to_string(), text));
                }
            }
        }
        Protocol::Openai => {}
    }
    if let Some(list) = request.get("messages").and_then(Value::as_array) {
        for m in list {
            let role = m
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let text = m.get("content").map(content_to_text).unwrap_or_default();
            if !text.is_empty() {
                out.push((role, text));
            }
        }
    }
    out
}

/// GenParams from the protocol's standard sampling fields (v1 surface).
fn gen_params_from_request(protocol: Protocol, request: &Value) -> rwkv_router::GenParams {
    let mut p = rwkv_router::GenParams::default();
    if let Some(v) = request.get("max_tokens").and_then(Value::as_u64) {
        p.max_tokens = (v as usize).clamp(1, 32 * 1024);
    } else if matches!(protocol, Protocol::Openai) {
        p.max_tokens = 512;
    } else {
        p.max_tokens = 1024; // Anthropic requires max_tokens; be generous if absent
    }
    if let Some(v) = request.get("temperature").and_then(Value::as_f64) {
        p.temperature = v as f32;
    }
    if let Some(v) = request.get("top_p").and_then(Value::as_f64) {
        p.top_p = (v as f32).clamp(0.0, 1.0);
    }
    if let Some(v) = request.get("presence_penalty").and_then(Value::as_f64) {
        p.presence_penalty = v as f32;
    }
    if let Some(v) = request.get("frequency_penalty").and_then(Value::as_f64) {
        p.frequency_penalty = v as f32;
    }
    match protocol {
        Protocol::Openai => {
            if let Some(stops) = request.get("stop").and_then(Value::as_array) {
                p.stop = stops
                    .iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect();
            }
        }
        Protocol::Anthropic => {
            if let Some(stops) = request.get("stop_sequences").and_then(Value::as_array) {
                p.stop = stops
                    .iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect();
            }
        }
    }
    p
}

/// Minimal RWKV chat template (World models): `Role: text` blocks; the
/// generation stops on the next role marker (also enforced via stop strings).
fn build_rwkv_prompt(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, text) in messages {
        let _ = writeln_like(&mut prompt, role, text);
    }
    prompt.push_str("Assistant:");
    prompt
}

fn writeln_like(prompt: &mut String, role: &str, text: &str) -> std::fmt::Result {
    use std::fmt::Write;
    writeln!(prompt, "{role}: {text}\n")
}

// ---------------------------------------------------------------------------
// Response shaping (builtin path)
// ---------------------------------------------------------------------------

fn approx_tokens(text: &str) -> u64 {
    (text.chars().count() as u64 / 4).max(1)
}

fn new_id(prefix: &str) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    format!("{prefix}-{ts}")
}

fn openai_completion_json(model: &str, text: &str) -> Value {
    json!({
        "id": new_id("chatcmpl-rvr"),
        "object": "chat.completion",
        "created": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": approx_tokens(text),
            "total_tokens": approx_tokens(text)
        }
    })
}

fn anthropic_message_json(model: &str, text: &str) -> Value {
    json!({
        "id": new_id("msg_rvr"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 0, "output_tokens": approx_tokens(text)}
    })
}

/// Emits the finished builtin text as an SSE sequence shaped for the
/// protocol, so stream-requesting agents work against local generation.
fn emulated_sse(protocol: Protocol, model: &str, text: &str, latency_ms: u64) -> Response {
    let mut sse = String::new();
    let mut push = |event: &str, data: &Value| {
        if protocol == Protocol::Anthropic && !event.is_empty() {
            sse.push_str(&format!("event: {event}\n"));
        }
        sse.push_str(&format!("data: {data}\n\n"));
    };

    match protocol {
        Protocol::Openai => {
            let id = new_id("chatcmpl-rvr");
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default();
            let base = json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model
            });
            let mut first = base.clone();
            first["choices"] = json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]);
            push("", &first);
            let mut chunk = base.clone();
            chunk["choices"] =
                json!([{"index": 0, "delta": {"content": text}, "finish_reason": null}]);
            push("", &chunk);
            let mut last = base;
            last["choices"] = json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]);
            push("", &last);
            sse.push_str("data: [DONE]\n\n");
        }
        Protocol::Anthropic => {
            let id = new_id("msg_rvr");
            push(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {"id": id, "type": "message", "role": "assistant", "model": model,
                                "content": [], "stop_reason": null, "stop_sequence": null,
                                "usage": {"input_tokens": 1, "output_tokens": 0}}
                }),
            );
            push(
                "content_block_start",
                &json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
            );
            push(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                }),
            );
            push(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            );
            push(
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": approx_tokens(text)}
                }),
            );
            push("message_stop", &json!({"type": "message_stop"}));
        }
    }

    let _ = latency_ms; // surfaced via x-ai00-latency-ms header
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(sse))
        .unwrap_or_else(|e| plain_error(protocol, StatusCode::BAD_GATEWAY, &e.to_string()))
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn tier_label(
    decision: Option<&rwkv_router::RoutingDecision>,
    tier: Option<RouteClass>,
) -> Option<String> {
    if let Some(d) = decision {
        return Some(match d.source {
            DecisionSource::Fallback => format!("{}(fallback)", d.route),
            _ => d.route.to_string(),
        });
    }
    tier.map(|t| format!("{t}(forced)"))
}

fn protocol_label(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Openai => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|e| plain_error(Protocol::Openai, StatusCode::BAD_GATEWAY, &e.to_string()))
}

fn plain_error(protocol: Protocol, status: StatusCode, message: &str) -> Response {
    json_response(status, &protocol.error_body(message))
}

#[allow(clippy::too_many_arguments)]
fn error_response(
    _state: &SharedState,
    protocol: Protocol,
    status: StatusCode,
    tier: Option<String>,
    upstream: Option<&str>,
    message: &str,
    started: Instant,
) -> Response {
    log::warn!(
        "[proxy] {} error: status={status} tier={} upstream={} latency={}ms message={message}",
        protocol_label(protocol),
        tier.as_deref().unwrap_or("-"),
        upstream.unwrap_or("-"),
        started.elapsed().as_millis()
    );
    plain_error(protocol, status, message)
}
