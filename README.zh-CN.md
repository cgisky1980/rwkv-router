# RWKV-Router

**自进化的本地优先 LLM 智能路由组件 —— R0–R3 分级、纯 Rust 自进化闭环、OpenAI/Anthropic 兼容本地网关。**

[English](README.md) | [中文文档](README.zh-CN.md)

> **主打卖点——省 token：** 斯坦福 & Together AI 的 *Intelligence per Watt*（百万真实查询）显示，端云混合路由可节省 **60–80%** 的能耗/算力/成本——且**准确率 80% 的路由器就能吃下约 80% 的理论最优收益**。RWKV-Router 预训练头开箱即达这条线；自进化后达 **~100%**（标准评估集），接近理论最优：**最多 ~89% 的云端 token 根本不用出机**。

## 研究背书

RWKV-Router 是 **斯坦福大学（Stanford University）& Together AI** 在 *Intelligence per Watt* 中验证过的路由范式的完整开源实现：**88.7% 的真实 AI 查询可由本地模型处理**，云端 API 只需承接小路由器判定为高置信的那部分请求。

- 论文：<https://arxiv.org/abs/2511.07885>
- 官方代码：<https://github.com/HazyResearch/intelligence-per-watt>

RWKV-Router 比论文更进一步：补上了论文静态路由器没有的**第三块**——完全本地、纯 Rust 的**自进化闭环**，让路由器从真实流量中持续变强。

### 节省预估 —— 论文基线 → RWKV-Router

论文 Q3 结论：与纯云端基线相比，端云混合路由可降低 **60–80% 的能耗、算力与美元成本**；**准确率 80% 的路由器（论文所称"现实可达目标"）已能捕获约 80% 的理论最优收益**。

| 路由准确率 | 论文推算收益 | RWKV-Router 现状 |
|---|---|---|
| 80% —— 论文的现实可达目标 | **~80% 理论最优节省** | 预训练头开箱即达（~77–80%） |
| **~100% —— 自进化后** | **60–80% 区间上限 ≈ 理论最优** | 进化闭环跑完后标准评估集实测 |

换算成 token：论文证明 **88.7%** 的真实查询可本地处理——接近理论最优的路由器之下，云端 token 支出降为基线的约 **11% → 最多省 ~89% 的云端 token**。

> 实际节省取决于流量构成——agent 流量（确认、格式化、分类、短摘要）天然偏向可本地服务的层级，通常落在区间有利端。

## 相关项目与生态

- **[rwkv7-state-embedding](https://github.com/cgisky1980/rwkv7-state-embedding)** —— 本项目分类器的学术基础。该项目（附论文）系统研究了如何从 RWKV-7 内部状态（hidden state 与 WKV state）提取语义嵌入，证明在 hidden state 上做监督投影可达 **0.93 的任务分类准确率**。RWKV-Router 的 0.1B 分类头——mean-hidden 状态嵌入 + 可训练 MLP 头——正是这一方法的工程化落地。
- **RWKV 官方**：
  - 官网：<https://www.rwkv.com>
  - GitHub 组织：<https://github.com/RWKV>
  - 模型训练仓库：<https://github.com/BlinkDL/RWKV-LM>
- **rwkv-rsv** —— 纯 Rust + Vulkan 推理运行时，以 `vendor/rwkv-rsv/` 内嵌于本仓库（常驻 0.1B 分类器 + 层级生成池）。

## 准确率

| 阶段 | 标准评估集 | 说明 |
|---|---|---|
| 预训练头（开箱即用） | ~77–80% | `fetch` 下载后立即可用 |
| **自动进化后（真实流量 → 标注 → 微调 → 闸门）** | **~100%** | 实际测试结果 |

本项目的核心卖点就是第二行：随仓库分发的预训练头是可靠的起点，几轮自动进化后路由准确率达到**接近 100%**（实际测试）——并且内置 eval_pack 闸门保证新头上线的条件是准确率不回退（**进化不可致劣**）。

## 核心特性

- **R0–R3 四级路由**：每条请求先过规则栈（trivial-ack 短路 / 安全升级 / sticky 层级），再用常驻 RWKV 0.1B 做分类（mean-hidden 状态嵌入 + 可训练 MLP 头），输出层级 + 置信度决策。
- **两种使用模式**（同一组件，配置决定）：
  - **仅路由 route-only**：输出 R0–R3 决策，生成交给宿主/外部端点；
  - **全本地 local-stack**：R0/R1 → RWKV 小模型、R2 → 7B 级、R3 → 13B 级，内嵌 rwkv-rsv（Vulkan）推理，全离线。
- **自进化闭环（可自动进化，且进化不可致劣）**：真实路由时零成本捕获 `(文本, hidden, probs)` → 标注 → **纯 Rust AdamW 原位微调**（无 PyTorch 依赖）→ eval_pack 闸门（新头准确率不回退才上线）→ 备份 + 热部署。防遗忘设计：mean/std 冻结、低学习率、回放池按类均衡。
- **五种发行形态**：Rust crate / C ABI 动态库 / Python 包（pyo3）/ Node 包（napi-rs）/ **本地智能路由网关单二进制**（CLI + HTTP + MCP）。

## 为什么是代理网关（Sidecar）而不是一个 SDK

Codex CLI、Claude Code、Cursor、openclaw 这类 agent 不装 SDK，接入面只有三个——OpenAI 兼容 API、Anthropic 兼容 API、MCP。`rwkv-router serve` 是一个本地网关：agent 把 base_url 指到 `127.0.0.1:21750`，每个请求先被 0.1B 分类，再按层级转发到实际后端（本地 RWKV / 云端 API）并透传响应（含 SSE）。**agent 无感知**——它以为在和一个模型说话，实际 R0/R1 从不出本机，R2/R3 才上云。

```
Codex CLI / Claude Code / Cursor / 任何 OpenAI SDK
        │  base_url = http://127.0.0.1:21750/v1
        ▼
┌─────────────────────────────────────────────────┐
│  rwkv-router serve（本地网关，单二进制）             │
│                                                 │
│  规则栈 ──► 0.1B 分类（MLP 头）──► R0 │ R1 │ R2 │ R3 │
│                    ▲                            │
│                    └── 自进化闭环（捕获/标注/微调/闸门）│
└─────────┬──────────────┬──────────────┬──────────┘
          ▼              ▼              ▼
   本地 RWKV 0.1B   本地 RWKV 7B    云端 API（DeepSeek/Claude/...）
   （R0/R1 内嵌生成）  （内嵌生成）     （R2/R3 透传）
```

## 快速开始

### 1. Sidecar 网关

```bash
# 构建（Rust，--release 必须）
cargo build --release -p rwkv-router-sidecar

# 一条命令下载模型包：0.1B 分类骨干(int8) + tokenizer + 预训练自进化头
# （ModelScope 优先，hf-mirror / HF 自动回退；默认落 ./models/）
./target/release/rwkv-router fetch
# 可选：同时下载层级生成模型（R2/R3 本地生成用，3GB/7.5GB）
./target/release/rwkv-router fetch --tier-models 3b

# 准备配置（config.example.json 的路径与 fetch 产物对齐，开箱即用）
cp sidecar/config.example.json config.json
# 编辑 config.json：填入云端 upstream（R2/R3 透传用）

# 启动网关（默认监听 127.0.0.1:21750）
./target/release/rwkv-router serve
```

无模型文件也能跑：没有 classifier 时网关进入 rules+fallback 模式（trivial-ack → R0，其余 → fallback upstream / R1）。

一行配置跑通云端分流（不需要任何本地模型）：

```json
{
  "fallback": { "upstream": "openai", "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY" }
}
```

### 2. CLI/IDE Agent 零代码接入

| Agent | 接入配置 |
|---|---|
| Codex CLI | `~/.codex/config.toml`: `model_provider = { base_url = "http://127.0.0.1:21750/v1", api_key = "dummy" }` |
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:21750 ANTHROPIC_AUTH_TOKEN=dummy claude` |
| Cursor / Continue / Cline | Settings → Model → OpenAI Base URL = `http://127.0.0.1:21750/v1` |
| openclaw 等开源 agent | 同上（OpenAI / Anthropic base URL 二选一） |
| 通用 OpenAI SDK | `OPENAI_BASE_URL=http://127.0.0.1:21750/v1 OPENAI_API_KEY=dummy` |
| MCP 生态 | `mcpServers` 注册 `rwkv-router mcp`（见下文 MCP） |

### 3. CLI 子命令

```bash
rwkv-router serve                 # 网关守护进程（默认子命令）
rwkv-router fetch                 # 下载模型包（MS/HF，含可选 --tier-models 3b|7b）
rwkv-router route "帮我写首诗"     # 单次决策，打印 JSON（可 --session / --config）
rwkv-router stats                 # 捕获样本库统计
rwkv-router label 12 R2           # 给第 12 条样本标注正确层级（clear 可撤销）
rwkv-router evolve                # 手动跑一轮进化（微调 + 闸门 + 热部署）
rwkv-router mcp                   # MCP server（stdio）
```

## MCP 接入

把路由能力（决策 / 标注 / 进化）暴露为 agent 工具——标注动作本身也可以由 agent 在对话中完成，闭环的最后一块：

```json
// claude_desktop_config.json / Claude Code / Codex CLI MCP 配置
{ "mcpServers": { "rwkv-router": { "command": "rwkv-router", "args": ["mcp"] } } }
```

| 工具 | 参数 | 说明 |
|---|---|---|
| `route` | `input`, `summary?`, `session_id?`, `turn_index?` | 一次路由决策（喂进化捕获库） |
| `router_stats` | — | 捕获样本统计（总数/已标注/分层分布） |
| `router_label` | `idx`, `tier`（`"R0".."R3"` 或 `"clear"`） | 标注/撤销标注 |
| `router_evolve` | — | 跑一轮进化（阻塞；微调+闸门+热部署） |

## HTTP API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI 兼容代理（含 SSE 流式透传），model 可用 `ai00-auto` 自动分流或直接指定 `R0`..`R3` |
| `/v1/messages` | POST | Anthropic 兼容代理（同协议透传 + model 重写） |
| `/v1/models` | GET | 模型列表（`ai00-auto` + 四个层级名） |
| `/v1/health` | GET | 健康检查（classifier/evolution/fallback 状态） |
| `/v1/route` | POST | 决策 API：`{input, session_id?, summary?, turn_index?}` |
| `/v1/capture/stats` | GET | 捕获库统计 |
| `/v1/capture/list` | GET | 捕获样本列表（分页） |
| `/v1/capture/label` | POST | 标注：`{idx, tier: "R0".."R3" \| null}` |
| `/v1/evolve` | POST | 触发一轮进化 |
| `/v1/evolve/status` | GET | 进化运行状态 |

服务只监听 `127.0.0.1`（本地网关；对外暴露需自行加鉴权）。API key 一律从 `api_key_env` 指定的环境变量读取，永不落盘。

## 自进化闭环

```
真实路由 ──捕获──► samples.jsonl（文本 + f16 hidden + probs，~3KB/条）
                     │
                     ▼ 标注（HTTP /v1/capture/label、CLI label、MCP router_label）
                  已标注样本
                     │
                     ▼ evolve()（AdamW 微调当前 MLP 头，mean/std 冻结防遗忘）
                  新头 ──eval_pack 闸门──► 准确率不回退才部署（备份旧头 + 引擎热重载）
```

闸门需要 `packs_dir` 下有 `eval_pack.json`（独立参照，不含捕获样本）：

```json
{
  "version": 1,
  "base_dim": 768,
  "samples": [
    { "h": "<base64 f16 little-endian hidden>", "t": 2, "p": null }
  ]
}
```

可选 `replay_pool.json`（同格式）为回放池，按类均衡混入训练防遗忘。

## 库使用（嵌入式，不经网关）

### Rust

```rust
use rwkv_router::{RouterSession, RouterConfig};

let mut session = RouterSession::new(RouterConfig::default());
// 可选：挂 0.1B 分类器（feature "rwkv"）
// session.load_classifier("rwkv-0.1b.st", "vocab.json", "router_head_0b.json", 500)?;
let decision = session.route("s1", "帮我总结这份文档", None, 0);
println!("tier = {}", decision.route);   // R0..R3
```

默认 feature 是纯逻辑（零 tokio/GPU 依赖）；`rwkv` feature 启用内嵌 RWKV 推理（分类器 + LRU 分层生成池），`ffi` feature 启用 C ABI 导出。

> 内嵌推理栈 `rwkv-rsv`（纯 Rust + Vulkan）以 **vendor** 方式随仓库分发（`vendor/rwkv-rsv/`，约 2.5MB）——克隆即构建，无需外部依赖；shader 以 committed `.spv` 回退，无 Vulkan SDK 也能编译（编辑 shader 源码则需要 `glslangValidator`）。

### Python（pyo3，PyPI 包名 `rwkv-router`）

```bash
# 从源码构建 wheel（需 uv；maturin 由 uv 拉取）
uv venv .venv
uv pip install --python .venv/Scripts/python.exe maturin
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
.venv/Scripts/maturin.exe develop --release -m bindings/python/Cargo.toml
```

```python
from rwkv_router import RouterSession

s = RouterSession()
d = s.route("帮我写一首诗")            # -> {"route": "R2", "confidence": .., ...}
s.load_classifier("rwkv-0.1b.st", "vocab.json", "router_head_0b.json")
s.attach_generation("R0", "vocab.json", "rwkv-0.1b.st")
out = s.generate("R0", "你好")         # -> {"text": .., "output_tokens": ..}
s.configure_evolution("evolution-data", packs_dir="packs")
print(s.capture_stats())
```

所有结构化结果（决策/统计/进化报告/生成输出）统一以 `dict` 返回——serde 一条桥接路径，与 Rust API 自动同步。

### Node.js（napi-rs，npm 包名 `rwkv-router`）

```bash
# 从源码构建（产物 .node 放 bindings/node/ 下，见 index.js loader 说明）
cargo build --release -p rwkv-router-node
# Windows: copy target/release/rwkv_router_node.dll bindings/node/rwkv-router.win32-x64-msvc.node
```

```js
import { RouterSession } from 'rwkv-router'

const s = new RouterSession()
const d = s.route('帮我写一首诗')       // -> { route: 'R2', confidence: .., ... }
s.loadClassifier('rwkv-0.1b.st', 'vocab.json', 'router_head_0b.json')
const out = s.generate('R0', '你好')
```

### C ABI（任意 FFI 语言）

`cargo build --release --features ffi` 产出 `rwkv_router.dll/.so/.dylib`；`rwkv_router_new / route / load_classifier / attach_generation / generate / configure_evolution / capture_stats / capture_label / capture_list / evolve / free / string_free` 全套句柄式 API，结构化出入参均为 JSON 字符串。见 `src/ffi.rs`。

## 训练工作台

[`training/`](training/) 目录提供训练文本源（~60MB）与完整脚本链：数据切片（`build_dataset_v5.py`）→ hidden 提取（网关采集 → `convert_capture.py` 桥接，或离线批量）→ 头训练（`train_mlp.py`，torch）→ 闸门包/回放池构建（`build_eval_pack.py`）→ 评估（`eval_head.py`）。中间产物（特征文件/闸门包/头）**有意不随仓库分发**——请自行跑流水线。契约文档（特征行/head JSON/eval pack）见 [training/README.md](training/README.md)。

## 配置参考（config.json）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `tiers.R0..R3` | upstream? | — | 层级 → 后端映射；缺省层级走 fallback |
| `fallback` | upstream? | — | 引擎失败/未配置层级的兜底 |
| `router.classifier` | object? | — | `{model, vocab, head, timeout_ms=500}`，0.1B 常驻分类 |
| `router.generation.max_loaded_models` | usize | 1 | 并发驻留层模型数（LRU 淘汰，控制显存） |
| `server.host` / `server.port` | string/u16 | `127.0.0.1` / `21750` | 监听地址 |
| `evolution.data_dir` | path | `evolution-data` | samples.jsonl / auto_state.json 目录 |
| `evolution.packs_dir` | path? | — | eval_pack.json 所在目录（配置后开启进化） |
| `evolution.head_path` | path? | classifier.head | 进化部署目标 |
| `evolution.capture_limit` | usize | 2000 | 捕获上限（FIFO） |
| `evolution.min_labeled_for_evolve` | usize | 20 | 触发微调的最少已标注数 |
| `evolution.auto_evolve_step` | usize | 50 | 每新增 N 条已标注自动进化 |

upstream 三选一：`builtin-rwkv`（`model` + `tokenizer`）、`openai`（`base_url` 含 `/v1` + `model` + `api_key_env`）、`anthropic`（`base_url` 不含 `/v1`）。

## 与客户端（Ai00-X）的关系

纯逻辑层（head / training / postprocess / rules / tier / config）与客户端 **单一数据源**：客户端 `pub use` 重导出本 crate，不分叉。编排层（SmartRouter 引擎桥 / 进化编排）在本 crate 内独立自建；客户端保留自身实现，零改动。本仓库可独立构建、测试、发布。

## 构建 / 测试

```bash
cargo test                 # 核心库 + sidecar（81 tests）
cargo clippy --workspace --all-targets   # 零警告
cargo fmt --all
cargo build --release -p rwkv-router-sidecar   # 网关单二进制
```

feature 矩阵：默认（纯逻辑）/ `rwkv`（内嵌 RWKV 推理）/ `ffi`（C ABI）。绑定 crate（pyo3/napi）仅经 maturin/napi 工具链构建，验证方式见上文各节。

## License

MIT
