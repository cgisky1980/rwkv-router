//! Face 3 — MCP server (`rwkv-router mcp`): agent tools over stdio.
//!
//! Hand-rolled minimal MCP implementation (newline-delimited JSON-RPC 2.0 on
//! stdin/stdout — the MCP stdio transport). Zero extra dependencies; the four
//! routing tools close the evolution loop from inside an agent conversation:
//! the agent can query decisions, label corrections, and trigger evolution.
//!
//! Tools (plan §3.3 面 3):
//! - `route`         — one routing decision (`input`, optional `summary`,
//!   `session_id`, `turn_index`)
//! - `router_stats`  — capture-store statistics
//! - `router_label`  — label (`idx`, `tier` = "R0".."R3" | "clear")
//! - `router_evolve` — run one evolution cycle (blocking: fine-tune + eval
//!   gate + deploy)
//!
//! Logging goes to stderr only; stdout carries protocol frames exclusively.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::build_session;
use crate::config::SidecarConfig;
use rwkv_router::{RouteClass, RouterSession};

/// Protocol versions we are compatible with; the newest is returned when the
/// client requests something unknown. Known versions are echoed so the
/// negotiation always lands on a version both sides understand.
const SUPPORTED_VERSIONS: [&str; 3] = ["2024-11-05", "2025-03-26", "2025-06-18"];
const SERVER_NAME: &str = "rwkv-router";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn run(config: &SidecarConfig) -> Result<(), String> {
    let session = build_session(config)?;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    log::info!("[mcp] {SERVER_NAME} v{SERVER_VERSION} listening on stdio");

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("read stdin: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                // Malformed frame: reply with a parse error (id null) and keep
                // serving — a broken line must not kill the session.
                let resp = json!({
                    "jsonrpc": "2.0", "id": null,
                    "error": {"code": -32700, "message": format!("parse error: {e}")}
                });
                write_frame(&mut out, &resp)?;
                continue;
            }
        };
        if let Some(resp) = handle_request(&session, req) {
            write_frame(&mut out, &resp)?;
        }
    }
    Ok(())
}

fn write_frame(out: &mut impl Write, resp: &Value) -> Result<(), String> {
    let text = serde_json::to_string(resp).map_err(|e| format!("serialize response: {e}"))?;
    writeln!(out, "{text}").map_err(|e| format!("write stdout: {e}"))?;
    out.flush().map_err(|e| format!("flush stdout: {e}"))
}

/// Pure dispatcher (unit-testable): one JSON-RPC request → optional response.
/// Notifications (no id) produce no response.
fn handle_request(session: &RouterSession, req: Value) -> Option<Value> {
    let method = req.get("method")?.as_str()?.to_string();
    let id = match req.get("id") {
        Some(v) if !v.is_null() => v.clone(),
        _ => return None, // notification: no reply
    };
    let params = req.get("params");
    let result = match method.as_str() {
        "initialize" => Ok(initialize(params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(tools_list()),
        "tools/call" => tools_call(session, params),
        _ => Err(json!({
            "code": -32601, "message": format!("method not found: {method}")
        })),
    };
    Some(match result {
        Ok(v) => json!({"jsonrpc": "2.0", "id": id, "result": v}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

fn initialize(params: Option<&Value>) -> Value {
    let requested = params
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_str());
    let version = match requested {
        Some(v) if SUPPORTED_VERSIONS.contains(&v) => v.to_string(),
        _ => SUPPORTED_VERSIONS[SUPPORTED_VERSIONS.len() - 1].to_string(),
    };
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
    })
}

fn tool_def(name: &str, description: &str, schema: Value) -> Value {
    json!({"name": name, "description": description, "inputSchema": schema})
}

fn tools_list() -> Value {
    let tools = [
        tool_def(
            "route",
            "Route one user input through the self-evolving smart router. \
             Returns the tier decision (R0-R3), chosen backend model and \
             source. Feeds the evolution loop's capture store when evolution \
             is configured.",
            json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The user input text to classify"},
                    "summary": {"type": "string", "description": "Optional conversation summary prefix"},
                    "session_id": {"type": "string", "description": "Session id for sticky-tier context (default 'mcp')"},
                    "turn_index": {"type": "integer", "description": "Zero-based conversation turn (default 0)"}
                },
                "required": ["input"]
            }),
        ),
        tool_def(
            "router_stats",
            "Capture-store statistics: total/labelled sample counts and \
             per-tier distribution. Requires evolution to be configured.",
            json!({"type": "object", "properties": {}}),
        ),
        tool_def(
            "router_label",
            "Label (or clear) a captured sample so the next evolution can \
             learn from it. `tier` is one of \"R0\", \"R1\", \"R2\", \"R3\" \
             or \"clear\".",
            json!({
                "type": "object",
                "properties": {
                    "idx": {"type": "integer", "description": "Sample index (see router_stats / capture list)"},
                    "tier": {"type": "string", "enum": ["R0", "R1", "R2", "R3", "clear"]}
                },
                "required": ["idx", "tier"]
            }),
        ),
        tool_def(
            "router_evolve",
            "Run one self-evolution cycle on the captured samples: AdamW \
             fine-tune of the routing head, eval-pack gate (never deploys a \
             worse head), backup + hot-reload. Blocking; may take seconds to \
             minutes depending on sample count. Requires evolution and an \
             eval pack.",
            json!({"type": "object", "properties": {}}),
        ),
    ];
    json!({"tools": tools})
}

fn tools_call(session: &RouterSession, params: Option<&Value>) -> Result<Value, Value> {
    let params = params.ok_or_else(|| rpc_err(-32602, "missing params"))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| rpc_err(-32602, "params.name must be a string"))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let outcome: Result<String, String> = match name {
        "route" => tool_route(session, &args),
        "router_stats" => session
            .capture_stats()
            .and_then(|s| serde_json::to_string_pretty(&s).map_err(|e| e.to_string())),
        "router_label" => tool_label(session, &args),
        "router_evolve" => session
            .evolve()
            .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string())),
        _ => Err(format!("unknown tool: {name}")),
    };
    match outcome {
        Ok(text) => Ok(json!({"content": [{"type": "text", "text": text}]})),
        Err(e) => Ok(json!({
            "content": [{"type": "text", "text": e}],
            "isError": true
        })),
    }
}

fn tool_route(session: &RouterSession, args: &Value) -> Result<String, String> {
    let input = args
        .get("input")
        .and_then(|v| v.as_str())
        .ok_or("route: missing required argument 'input'")?;
    let summary = args.get("summary").and_then(|v| v.as_str());
    let session_id = args
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("mcp");
    let turn_index = args.get("turn_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let decision = session.route(session_id, input, summary, turn_index);
    serde_json::to_string_pretty(&decision).map_err(|e| format!("serialize decision: {e}"))
}

fn tool_label(session: &RouterSession, args: &Value) -> Result<String, String> {
    let idx = args
        .get("idx")
        .and_then(|v| v.as_u64())
        .ok_or("router_label: missing required argument 'idx'")? as usize;
    let tier = args
        .get("tier")
        .and_then(|v| v.as_str())
        .ok_or("router_label: missing required argument 'tier'")?;
    let label = if tier.eq_ignore_ascii_case("clear") {
        None
    } else {
        let tier = RouteClass::parse_from_str(tier).ok_or_else(|| {
            format!("router_label: invalid tier '{tier}' (expected R0-R3 or clear)")
        })?;
        Some(tier.index() as u8)
    };
    session.capture_label(idx, label)?;
    Ok(format!(
        "ok: sample {idx} label = {}",
        label
            .map(|v| v.to_string())
            .unwrap_or_else(|| "clear".into())
    ))
}

fn rpc_err(code: i64, message: &str) -> Value {
    json!({"code": code, "message": message})
}

#[cfg(test)]
mod tests {
    use super::*;
    use rwkv_router::RouterConfig;

    fn session() -> RouterSession {
        RouterSession::new(RouterConfig::default())
    }

    fn call(method: &str, params: Value, id: i64) -> Value {
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        handle_request(&session(), req).expect("request should produce a response")
    }

    #[test]
    fn initialize_negotiates_known_version_and_advertises_tools() {
        let resp = call("initialize", json!({"protocolVersion": "2025-06-18"}), 1);
        let result = &resp["result"];
        assert_eq!(result["protocolVersion"], "2025-06-18");
        assert_eq!(result["capabilities"]["tools"], json!({}));
        assert_eq!(result["serverInfo"]["name"], "rwkv-router");
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn initialize_unknown_version_falls_back_to_latest() {
        let resp = call("initialize", json!({"protocolVersion": "1999-01-01"}), 2);
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
    }

    #[test]
    fn ping_returns_empty_result() {
        let resp = call("ping", json!({}), 3);
        assert_eq!(resp["result"], json!({}));
    }

    #[test]
    fn tools_list_exposes_four_routing_tools_with_schemas() {
        let resp = call("tools/list", json!({}), 4);
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 4);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            ["route", "router_stats", "router_label", "router_evolve"]
        );
        for t in tools {
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn route_tool_returns_decision_json() {
        let resp = call(
            "tools/call",
            json!({"name": "route", "arguments": {"input": "ok"}}),
            5,
        );
        assert!(resp["result"]["isError"].is_null());
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let decision: Value = serde_json::from_str(text).unwrap();
        assert_eq!(decision["route"], "R0"); // trivial-ack short-circuit
    }

    #[test]
    fn route_tool_missing_input_is_error_content() {
        let resp = call("tools/call", json!({"name": "route", "arguments": {}}), 6);
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("input"));
    }

    #[test]
    fn evolution_tools_error_when_unconfigured() {
        for name in ["router_stats", "router_evolve", "router_label"] {
            let args = if name == "router_label" {
                json!({"idx": 0, "tier": "R1"})
            } else {
                json!({})
            };
            let resp = call("tools/call", json!({"name": name, "arguments": args}), 7);
            assert_eq!(resp["result"]["isError"], true, "tool {name}");
            assert!(resp["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("evolution"));
        }
    }

    #[test]
    fn unknown_tool_reports_is_error_not_protocol_error() {
        let resp = call("tools/call", json!({"name": "nope", "arguments": {}}), 8);
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown tool"));
    }

    #[test]
    fn unknown_method_is_jsonrpc_error() {
        let resp = call("resources/list", json!({}), 9);
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn notification_produces_no_response() {
        let req = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        assert!(handle_request(&session(), req).is_none());
    }
}
