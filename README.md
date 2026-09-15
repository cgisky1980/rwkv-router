# RWKV-Router

**Self-evolving local-first smart router for LLM requests — R0–R3 tiering, pure-Rust self-evolution loop, and an OpenAI/Anthropic-compatible local gateway.**

[English](README.md) | [中文文档](README.zh-CN.md)

## Research backing

RWKV-Router is a production-grade open-source implementation of the routing paradigm validated by **Stanford University & Together AI** in *Intelligence per Watt*: **88.7% of real AI queries can be served by local models**, with cloud APIs reserved only for requests a small router assigns high confidence to.

- Paper: <https://arxiv.org/abs/2511.07885>
- Official code: <https://github.com/HazyResearch/intelligence-per-watt>

RWKV-Router goes one step further than the paper: it adds a **third piece** the paper's static router lacks — a fully local, pure-Rust **self-evolution loop** that keeps improving the router from real traffic.

## Related projects & ecosystem

- **[rwkv7-state-embedding](https://github.com/cgisky1980/rwkv7-state-embedding)** — the research foundation of our classifier. That project (with its own paper) systematically studies extracting semantic embeddings from RWKV-7 internal states (hidden state & WKV state) and shows that a supervised projection on the hidden state reaches **0.93 task-classification accuracy**. The 0.1B classifier head in RWKV-Router — mean-hidden state embedding + trainable MLP head — is the engineering productization of exactly that method.
- **RWKV official**:
  - Website: <https://www.rwkv.com>
  - GitHub org: <https://github.com/RWKV>
  - Model training repo: <https://github.com/BlinkDL/RWKV-LM>
- **rwkv-rsv** — the pure-Rust + Vulkan inference runtime embedded in this repo as `vendor/rwkv-rsv/` (resident 0.1B classifier + tier generation pool).

## Accuracy

| Stage | Standard eval set | Notes |
|---|---|---|
| Pretrained head (out of the box) | ~77–80% | Usable immediately after `fetch` |
| **After self-evolution (real traffic → label → fine-tune → gate)** | **~100%** | Measured on our post-evolution test run |

The headline claim of this project is the second row: the shipped pretrained head is a solid starting point, and a few automatic evolution cycles on real traffic push routing accuracy to **near-perfect** — with a built-in eval-pack gate guaranteeing each new head never regresses (evolution cannot make the router worse).

## Core features

- **R0–R3 four-tier routing**: every request passes a rule stack first (trivial-ack short-circuit / safety escalation / sticky tiers), then a resident RWKV 0.1B classifier (mean-hidden state embedding + trainable MLP head) emits a tier + confidence decision.
- **Two usage modes** (same component, config decides):
  - **Route-only**: return the R0–R3 decision; generation stays with the host / external endpoints;
  - **Local-stack**: R0/R1 → small RWKV, R2 → 7B-class, R3 → 13B-class, embedded rwkv-rsv (Vulkan) inference — fully offline.
- **Self-evolution loop (automatic, and provably non-regressing)**: real routing captures `(text, hidden, probs)` at zero cost → labeling → **pure-Rust AdamW in-place fine-tuning** (no PyTorch) → eval-pack gate (a new head ships only if accuracy does not regress) → backup + hot-reload. Forgetting is mitigated by frozen mean/std, low learning rate, and class-balanced replay.
- **Five distribution forms**: Rust crate / C ABI dynamic library / Python package (pyo3) / Node package (napi-rs) / **local smart-routing gateway single binary** (CLI + HTTP + MCP).

## Why a sidecar gateway instead of an SDK

Agents like Codex CLI, Claude Code, Cursor and openclaw don't install SDKs — their integration surface is exactly three things: an OpenAI-compatible API, an Anthropic-compatible API, and MCP. `rwkv-router serve` is a local gateway: point the agent's `base_url` at `127.0.0.1:21750`; every request is classified by the 0.1B router first, then forwarded per tier to the real backend (local RWKV / cloud API) with the response passed through (SSE included). **The agent never notices** — it thinks it is talking to one model, while R0/R1 never leave the machine and only R2/R3 hit the cloud.

```
Codex CLI / Claude Code / Cursor / any OpenAI SDK
        │  base_url = http://127.0.0.1:21750/v1
        ▼
┌─────────────────────────────────────────────────┐
│  rwkv-router serve (local gateway, one binary)  │
│                                                 │
│  rules ──► 0.1B classify (MLP head) ──► R0│R1│R2│R3 │
│                    ▲                            │
│                    └── self-evolution loop      │
│                        (capture/label/finetune/gate) │
└─────────┬──────────────┬──────────────┬──────────┘
          ▼              ▼              ▼
   local RWKV 0.1B   local RWKV 7B    cloud API (DeepSeek/Claude/...)
   (R0/R1 embedded)  (R2 embedded)    (R2/R3 passthrough)
```

## Quick start

### 1. Sidecar gateway

```bash
# Build (Rust; --release required)
cargo build --release -p rwkv-router-sidecar

# One command to download the model bundle: 0.1B classifier backbone (int8)
# + tokenizer + pretrained self-evolving head
# (ModelScope first, automatic hf-mirror / HF fallback; lands in ./models/)
./target/release/rwkv-router fetch
# Optional: also download tier generation models (local R2/R3 generation, 3GB/7.5GB)
./target/release/rwkv-router fetch --tier-models 3b

# Config (config.example.json paths already match the fetch output)
cp sidecar/config.example.json config.json
# Edit config.json: fill in cloud upstreams (used for R2/R3 passthrough)

# Start the gateway (listens on 127.0.0.1:21750 by default)
./target/release/rwkv-router serve
```

It also runs with no model files at all: without a classifier the gateway falls back to rules + fallback routing (trivial-ack → R0, everything else → fallback upstream / R1).

Cloud-only routing with a single config stanza (no local models needed):

```json
{
  "fallback": { "upstream": "openai", "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY" }
}
```

### 2. Zero-code integration for CLI/IDE agents

| Agent | Configuration |
|---|---|
| Codex CLI | `~/.codex/config.toml`: `model_provider = { base_url = "http://127.0.0.1:21750/v1", api_key = "dummy" }` |
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:21750 ANTHROPIC_AUTH_TOKEN=dummy claude` |
| Cursor / Continue / Cline | Settings → Model → OpenAI Base URL = `http://127.0.0.1:21750/v1` |
| openclaw and other OSS agents | same as above (OpenAI / Anthropic base URL) |
| Any OpenAI SDK | `OPENAI_BASE_URL=http://127.0.0.1:21750/v1 OPENAI_API_KEY=dummy` |
| MCP ecosystem | register `rwkv-router mcp` under `mcpServers` (see MCP below) |

### 3. CLI subcommands

```bash
rwkv-router serve                 # gateway daemon (default subcommand)
rwkv-router fetch                 # download model bundle (MS/HF, optional --tier-models 3b|7b)
rwkv-router route "write me a poem"  # one-shot decision, prints JSON (--session / --config)
rwkv-router stats                 # capture store statistics
rwkv-router label 12 R2           # label sample #12 with the correct tier ("clear" to undo)
rwkv-router evolve                # run one evolution cycle manually (fine-tune + gate + hot deploy)
rwkv-router mcp                   # MCP server (stdio)
```

## MCP integration

Routing capability (decide / label / evolve) exposed as agent tools — labeling itself can be done by the agent mid-conversation, closing the loop:

```json
// claude_desktop_config.json / Claude Code / Codex CLI MCP config
{ "mcpServers": { "rwkv-router": { "command": "rwkv-router", "args": ["mcp"] } } }
```

| Tool | Args | Description |
|---|---|---|
| `route` | `input`, `summary?`, `session_id?`, `turn_index?` | One routing decision (feeds the evolution capture store) |
| `router_stats` | — | Capture store statistics (total/labeled/per-tier) |
| `router_label` | `idx`, `tier` (`"R0".."R3"` or `"clear"`) | Label / unlabel a sample |
| `router_evolve` | — | Run one evolution cycle (blocking; fine-tune + gate + hot deploy) |

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy (SSE passthrough); model `ai00-auto` routes automatically, or pin `R0`..`R3` |
| `/v1/messages` | POST | Anthropic-compatible proxy (same-protocol passthrough + model rewriting) |
| `/v1/models` | GET | Model list (`ai00-auto` + four tier names) |
| `/v1/health` | GET | Health check (classifier/evolution/fallback status) |
| `/v1/route` | POST | Decision API: `{input, session_id?, summary?, turn_index?}` |
| `/v1/capture/stats` | GET | Capture store statistics |
| `/v1/capture/list` | GET | Capture sample list (paged) |
| `/v1/capture/label` | POST | Label: `{idx, tier: "R0".."R3" \| null}` |
| `/v1/evolve` | POST | Trigger one evolution cycle |
| `/v1/evolve/status` | GET | Evolution run status |

The server binds `127.0.0.1` only (local gateway; add auth yourself if exposing it). API keys are read from the env var named by `api_key_env` and never persisted.

## Self-evolution loop

```
real routing ──capture──► samples.jsonl (text + f16 hidden + probs, ~3KB/row)
                            │
                            ▼ label (HTTP /v1/capture/label, CLI label, MCP router_label)
                         labeled samples
                            │
                            ▼ evolve() (AdamW fine-tune of the current MLP head, frozen mean/std)
                         new head ──eval-pack gate──► deploy only if accuracy holds (backup + hot reload)
```

The gate requires `eval_pack.json` under `packs_dir` (an independent reference set, disjoint from captured samples):

```json
{
  "version": 1,
  "base_dim": 768,
  "samples": [
    { "h": "<base64 f16 little-endian hidden>", "t": 2, "p": null }
  ]
}
```

An optional `replay_pool.json` (same format) is mixed in class-balanced during fine-tuning to prevent forgetting.

## Library usage (embedded, without the gateway)

### Rust

```rust
use rwkv_router::{RouterSession, RouterConfig};

let mut session = RouterSession::new(RouterConfig::default());
// Optional: attach the 0.1B classifier (feature "rwkv")
// session.load_classifier("rwkv-0.1b.st", "vocab.json", "router_head_0b.json", 500)?;
let decision = session.route("s1", "summarize this document", None, 0);
println!("tier = {}", decision.route);   // R0..R3
```

The default feature set is pure logic (zero tokio/GPU deps); the `rwkv` feature embeds RWKV inference (classifier + LRU tiered generation pool), `ffi` enables C ABI exports.

> The embedded inference stack `rwkv-rsv` (pure Rust + Vulkan) is **vendored** into this repo (`vendor/rwkv-rsv/`, ~2.5MB) — clone and build with no external dependencies; shaders fall back to committed `.spv` binaries so no Vulkan SDK is needed to compile (editing shader sources requires `glslangValidator`).

### Python (pyo3, PyPI name `rwkv-router`)

```bash
# Build the wheel from source (uv required; maturin is pulled by uv)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe maturin
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
.venv/Scripts/maturin.exe develop --release -m bindings/python/Cargo.toml
```

```python
from rwkv_router import RouterSession

s = RouterSession()
d = s.route("write me a poem")         # -> {"route": "R2", "confidence": .., ...}
s.load_classifier("rwkv-0.1b.st", "vocab.json", "router_head_0b.json")
s.attach_generation("R0", "vocab.json", "rwkv-0.1b.st")
out = s.generate("R0", "hello")        # -> {"text": .., "output_tokens": ..}
s.configure_evolution("evolution-data", packs_dir="packs")
print(s.capture_stats())
```

All structured results (decisions / stats / evolution reports / generation output) are returned as `dict` — one serde bridge, automatically in sync with the Rust API.

### Node.js (napi-rs, npm name `rwkv-router`)

```bash
# Build from source (the .node artifact goes under bindings/node/, see index.js loader)
cargo build --release -p rwkv-router-node
# Windows: copy target/release/rwkv_router_node.dll bindings/node/rwkv-router.win32-x64-msvc.node
```

```js
import { RouterSession } from 'rwkv-router'

const s = new RouterSession()
const d = s.route('write me a poem')   // -> { route: 'R2', confidence: .., ... }
s.loadClassifier('rwkv-0.1b.st', 'vocab.json', 'router_head_0b.json')
const out = s.generate('R0', 'hello')
```

### C ABI (any FFI language)

`cargo build --release --features ffi` produces `rwkv_router.dll/.so/.dylib`; the full handle-based API — `rwkv_router_new / route / load_classifier / attach_generation / generate / configure_evolution / capture_stats / capture_label / capture_list / evolve / free / string_free` — with JSON-string structured in/out params. See `src/ffi.rs`.

## Training workbench

The [`training/`](training/) directory ships the training text sources (~60MB) plus the full script chain: dataset slicing (`build_dataset_v5.py`), hidden extraction (gateway capture → `convert_capture.py` bridge, or offline bulk), head training (`train_mlp.py`, torch), eval-pack/replay-pool building (`build_eval_pack.py`) and evaluation (`eval_head.py`). Intermediate artifacts (feature files, eval packs, heads) are intentionally **not** distributed — run the pipeline yourself. Contracts (feature rows / head JSON / eval pack) are documented in [training/README.md](training/README.md).

## Configuration reference (config.json)

| Field | Type | Default | Description |
|---|---|---|---|
| `tiers.R0..R3` | upstream? | — | tier → backend mapping; missing tiers fall back |
| `fallback` | upstream? | — | fallback for engine failure / unconfigured tiers |
| `router.classifier` | object? | — | `{model, vocab, head, timeout_ms=500}`, resident 0.1B classifier |
| `router.generation.max_loaded_models` | usize | 1 | concurrently resident tier models (LRU eviction, controls VRAM) |
| `server.host` / `server.port` | string/u16 | `127.0.0.1` / `21750` | listen address |
| `evolution.data_dir` | path | `evolution-data` | samples.jsonl / auto_state.json directory |
| `evolution.packs_dir` | path? | — | directory holding eval_pack.json (enables evolution) |
| `evolution.head_path` | path? | classifier.head | evolution deployment target |
| `evolution.capture_limit` | usize | 2000 | capture cap (FIFO) |
| `evolution.min_labeled_for_evolve` | usize | 20 | min labeled samples before fine-tuning |
| `evolution.auto_evolve_step` | usize | 50 | auto-evolve every N new labels |

Upstream is one of three: `builtin-rwkv` (`model` + `tokenizer`), `openai` (`base_url` including `/v1` + `model` + `api_key_env`), `anthropic` (`base_url` without `/v1`).

## Relation to the Ai00-X client

The pure-logic layer (head / training / postprocess / rules / tier / config) is **single-sourced** with the client: the client `pub use` re-exports this crate, no forking. The orchestration layer (SmartRouter engine bridge / evolution orchestration) is built independently inside this crate; the client keeps its own implementation, unchanged. This repo builds, tests and releases standalone.

## Build / test

```bash
cargo test                 # core lib + sidecar (81 tests)
cargo clippy --workspace --all-targets   # zero warnings
cargo fmt --all
cargo build --release -p rwkv-router-sidecar   # gateway single binary
```

Feature matrix: default (pure logic) / `rwkv` (embedded RWKV inference) / `ffi` (C ABI). Binding crates (pyo3/napi) are built via the maturin/napi toolchains; verification per section above.

## License

MIT
