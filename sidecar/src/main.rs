//! rwkv-router sidecar binary: local intelligent routing gateway.
//!
//! Subcommands (plan §3.3):
//! - `serve` (default) — daemon: OpenAI/Anthropic proxy + decision API
//! - `mcp` — MCP server (agent tools)
//! - `route <text>` — one-shot decision, prints JSON
//! - `stats` / `label <idx> <tier>` / `evolve` — capture store operations

mod api;
mod config;
mod download;
mod mcp;
mod proxy;
mod state;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use rwkv_router::{EvolutionConfig, RouteClass, RouterSession};
use state::{AppState, EvolveStatus, SharedState};

use crate::config::SidecarConfig;

/// Self-evolving smart router gateway (R0–R3 → local RWKV / cloud upstream).
#[derive(Parser)]
#[command(name = "rwkv-router", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the gateway daemon (default).
    Serve {
        /// Path to config.json (default: ./config.json).
        #[arg(long)]
        config: Option<String>,
        /// Override server.port from config.
        #[arg(long)]
        port: Option<u16>,
    },
    /// Run the MCP server (agent tools over stdio).
    Mcp {
        #[arg(long)]
        config: Option<String>,
    },
    /// One-shot routing decision for <text>, prints the decision JSON.
    Route {
        text: String,
        /// Optional session id (sticky-tier context).
        #[arg(long, default_value = "cli")]
        session: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Print capture-store statistics.
    Stats {
        #[arg(long)]
        config: Option<String>,
    },
    /// Label (or clear with `clear`) the sample at <idx>.
    Label {
        idx: usize,
        /// "R0".."R3" or "clear".
        tier: String,
        #[arg(long)]
        config: Option<String>,
    },
    /// Run one evolution cycle (blocking).
    Evolve {
        #[arg(long)]
        config: Option<String>,
    },
    /// Download the model bundle (0.1B classifier + vocab + pretrained head)
    /// from ModelScope / HF (hf-mirror fallback) into --dir.
    Fetch {
        /// Target directory (default: ./models).
        #[arg(long, default_value = "models")]
        dir: String,
        /// Also download a tier generation model: "3b" or "7b".
        #[arg(long)]
        tier_models: Option<String>,
    },
}

fn load_config(path: Option<&String>) -> Result<SidecarConfig, String> {
    let path = path
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("config.json"));
    SidecarConfig::load(&path)
        .map_err(|e| format!("{e}\nhint: copy sidecar/config.example.json to config.json and edit"))
}

/// Assembles the RouterSession from config: classifier (optional, resilient),
/// generation pool (builtin tiers), evolution (optional).
pub(crate) fn build_session(config: &SidecarConfig) -> Result<Arc<RouterSession>, String> {
    let mut session = RouterSession::new(rwkv_router::RouterConfig::default());

    // 0.1B classifier — load failure degrades to fallback routing, never
    // aborts startup (the gateway stays useful without the head).
    if let Some(cls) = &config.router.classifier {
        match session.load_classifier(&cls.model, &cls.vocab, &cls.head, cls.timeout_ms) {
            Ok(()) => log::info!(
                "[serve] classifier loaded: {} (head {})",
                cls.model,
                cls.head
            ),
            Err(e) => log::error!("[serve] classifier load failed, requests will fall back: {e}"),
        }
    } else {
        log::info!("[serve] no classifier configured — route-only fallback mode");
    }

    // Generation pool: one entry per builtin tier (models load lazily).
    let mut builtin: Vec<(RouteClass, &str, &str)> = Vec::new();
    for tier in [
        RouteClass::R0,
        RouteClass::R1,
        RouteClass::R2,
        RouteClass::R3,
    ] {
        if let Some(config::UpstreamConfig::BuiltinRwkv { model, tokenizer }) =
            config.tiers.entry(tier)
        {
            builtin.push((tier, tokenizer.as_str(), model.as_str()));
        }
    }
    if let Some(&(first_tier, vocab, _)) = builtin.first() {
        let max_loaded = config.router.generation.max_loaded_models;
        for &(tier, vocab, model) in &builtin {
            session
                .attach_generation(tier, vocab, model, max_loaded)
                .map_err(|e| format!("failed to attach generation for {tier}: {e}"))?;
        }
        log::info!(
            "[serve] generation pool attached: {} builtin tier(s), max_loaded={max_loaded} (vocab {vocab})",
            builtin.len()
        );
        let _ = first_tier;
    }

    // Evolution loop.
    if let Some(evo) = &config.evolution {
        let head_path = evo
            .head_path
            .clone()
            .or_else(|| config.router.classifier.as_ref().map(|c| c.head.clone()))
            .unwrap_or_default();
        let mut evo_config = EvolutionConfig {
            data_dir: evo.data_dir.clone(),
            head_path: head_path.into(),
            packs_dir: evo.packs_dir.clone(),
            ..Default::default()
        };
        if let Some(v) = evo.capture_limit {
            evo_config.capture_limit = v;
        }
        if let Some(v) = evo.min_labeled_for_evolve {
            evo_config.min_labeled_for_evolve = v;
        }
        if let Some(v) = evo.auto_evolve_step {
            evo_config.auto_evolve_step = v;
        }
        session
            .configure_evolution(evo_config)
            .map_err(|e| format!("failed to configure evolution: {e}"))?;
        log::info!(
            "[serve] evolution configured: data_dir={}",
            evo.data_dir.display()
        );
    }

    Ok(Arc::new(session))
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    let result = match cli.command.unwrap_or(Command::Serve {
        config: None,
        port: None,
    }) {
        Command::Serve { config, port } => run_serve(config, port),
        Command::Mcp { config } => run_mcp(config),
        Command::Route {
            text,
            session,
            config,
        } => run_route(&text, session, config),
        Command::Stats { config } => run_stats(config),
        Command::Label { idx, tier, config } => run_label(idx, tier, config),
        Command::Evolve { config } => run_evolve(config),
        Command::Fetch { dir, tier_models } => {
            download::run(std::path::Path::new(&dir), tier_models.as_deref())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_serve(config_path: Option<String>, port_override: Option<u16>) -> Result<(), String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| format!("failed to create tokio runtime: {e}"))?;
    runtime.block_on(async_serve(config_path, port_override))
}

async fn async_serve(
    config_path: Option<String>,
    port_override: Option<u16>,
) -> Result<(), String> {
    let config = Arc::new(load_config(config_path.as_ref())?);
    let session = build_session(&config)?;

    let port = port_override.unwrap_or(config.server.port);
    let host = config.server.host.clone();
    let state: SharedState = Arc::new(AppState {
        session,
        config,
        http: reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?,
        evolve: Arc::new(EvolveStatus::default()),
    });

    let app = axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(proxy::openai_chat),
        )
        .route(
            "/v1/messages",
            axum::routing::post(proxy::anthropic_messages),
        )
        .route("/v1/models", axum::routing::get(api::models))
        .route("/v1/health", axum::routing::get(api::health))
        .route("/v1/route", axum::routing::post(api::route))
        .route("/v1/capture/stats", axum::routing::get(api::capture_stats))
        .route("/v1/capture/list", axum::routing::get(api::capture_list))
        .route("/v1/capture/label", axum::routing::post(api::capture_label))
        .route("/v1/evolve", axum::routing::post(api::evolve))
        .route("/v1/evolve/status", axum::routing::get(api::evolve_status))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|e| format!("failed to bind {host}:{port}: {e}"))?;
    log::info!("[serve] rwkv-router gateway listening on http://{host}:{port}");
    axum::serve(listener, app)
        .await
        .map_err(|e| format!("server error: {e}"))
}

fn run_mcp(config_path: Option<String>) -> Result<(), String> {
    let config = load_config(config_path.as_ref())?;
    mcp::run(&config)
}

fn run_route(text: &str, session_id: String, config_path: Option<String>) -> Result<(), String> {
    let config = load_config(config_path.as_ref())?;
    let session = build_session(&config)?;
    let decision = session.route(&session_id, text, None, 0);
    let json =
        serde_json::to_string_pretty(&decision).map_err(|e| format!("serialize decision: {e}"))?;
    println!("{json}");
    Ok(())
}

fn run_stats(config_path: Option<String>) -> Result<(), String> {
    let config = load_config(config_path.as_ref())?;
    let session = build_session(&config)?;
    let stats = session
        .capture_stats()
        .map_err(|e| format!("capture stats unavailable: {e}"))?;
    let json = serde_json::to_string_pretty(&stats).map_err(|e| format!("serialize stats: {e}"))?;
    println!("{json}");
    Ok(())
}

fn run_label(idx: usize, tier: String, config_path: Option<String>) -> Result<(), String> {
    let config = load_config(config_path.as_ref())?;
    let session = build_session(&config)?;
    let label = if tier.eq_ignore_ascii_case("clear") {
        None
    } else {
        let tier = RouteClass::parse_from_str(&tier)
            .ok_or_else(|| format!("invalid tier '{tier}' (expected R0-R3 or clear)"))?;
        Some(tier.index() as u8)
    };
    session
        .capture_label(idx, label)
        .map_err(|e| format!("label failed: {e}"))?;
    println!(
        "ok: sample {idx} label = {}",
        label
            .map(|v| v.to_string())
            .unwrap_or_else(|| "clear".into())
    );
    Ok(())
}

fn run_evolve(config_path: Option<String>) -> Result<(), String> {
    let config = load_config(config_path.as_ref())?;
    let session = build_session(&config)?;
    let result = session
        .evolve()
        .map_err(|e| format!("evolve failed: {e}"))?;
    let json =
        serde_json::to_string_pretty(&result).map_err(|e| format!("serialize result: {e}"))?;
    println!("{json}");
    Ok(())
}
