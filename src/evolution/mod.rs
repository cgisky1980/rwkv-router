//! 智能路由「自我进化」数据与训练编排（投影器原位微调闭环）。
//!
//! 数据流：真实路由分类时（`capture=true`）零成本捕获 `(文本, hidden,
//! prev_tier, probs)` → 宿主标注正确层级 → `evolve()` 在 Rust 端用 AdamW
//! 微调当前头（`crate::training`）→ eval_pack 闸门（新头准确率不回退才
//! 上线）→ 备份旧头 + 写入 + 引擎热重载。
//!
//! 线程约束：捕获热路径只做内存 push + 文件追加（µs 级），训练/闸门跑在
//! 独立 std::thread（自动触发）或调用方线程（`evolve()` 阻塞语义，sidecar
//! /绑定层自己开线程）。
//!
//! 防遗忘约束（与 training.rs 呼应）：mean/std 冻结、低学习率微调、
//! eval_pack 独立参照（不含捕获样本）、回放池按类均衡混入。

mod events;

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use half::f16;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub use events::{EvolutionEvent, EvolutionEventSink};

use crate::head::RouterHead;
use crate::training::{finetune_head, FinetuneOptions, TrainingSample};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRecord {
    /// Unix 毫秒时间戳。
    pub ts: u64,
    /// 分类输入原文（含 Summary 前缀，与 classify 输入一致）。
    pub text: String,
    /// mean-pooled hidden，f16 little-endian hex（768 维 ≈ 3KB）。
    pub hidden_hex: String,
    /// 上一轮 sticky 层级（0-3），None = 首轮/未知。
    pub prev_tier: Option<u8>,
    /// 分类头输出的 4 类概率（R0-R3，后处理前）。
    pub probs: [f32; 4],
    /// 采集时的骨干维度（换路由模型后旧样本自动失效过滤）。
    pub num_embd: usize,
    /// 正确层级标签（0-3）；None = 未标注。
    #[serde(default)]
    pub label: Option<u8>,
    /// 采集来源：route = 真实路由；preview = 测试入口（默认不采集）。
    #[serde(default)]
    pub source: String,
}

impl CaptureRecord {
    fn hidden_f32(&self) -> Vec<f32> {
        (0..self.hidden_hex.len() / 4)
            .map(|i| {
                let bytes = hex::decode(&self.hidden_hex[i * 4..i * 4 + 4]).unwrap_or_default();
                if bytes.len() == 2 {
                    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// hex 编解码（无 hex crate 依赖时的本地实现）。
mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        if !s.len().is_multiple_of(2) {
            return Err("odd hex length".to_string());
        }
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string()))
            .collect()
    }

    pub fn encode(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
}

/// mean-pooled hidden → f16 LE hex（768 维 ≈ 3KB/样本）。
pub fn hidden_to_hex(hidden: &[f32]) -> String {
    let mut buf = Vec::with_capacity(hidden.len() * 2);
    for &v in hidden {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes()[..2]);
    }
    hex::encode(&buf)
}

/// Evolution paths + thresholds (all injectable; defaults mirror the client).
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// 数据目录：`samples.jsonl`（捕获样本）与 `auto_state.json`（自动进化标记）。
    pub data_dir: PathBuf,
    /// 模型包目录：`eval_pack.json`（闸门评估包，进化必需）与
    /// `replay_pool.json`（回放池，可选防遗忘）。None = 进化不可用。
    pub packs_dir: Option<PathBuf>,
    /// 分类头 JSON 路径（进化部署目标）。
    pub head_path: PathBuf,
    /// 捕获样本上限（FIFO 淘汰）。f16+hex ≈ 3.1KB/条 → 默认满载 ~6MB。
    pub capture_limit: usize,
    /// 触发一次微调所需的最少已标注样本数。
    pub min_labeled_for_evolve: usize,
    /// 自动进化步长：每新增 N 条已标注样本自动触发一次微调。
    pub auto_evolve_step: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("router_evolution"),
            packs_dir: None,
            head_path: PathBuf::from("router_head.json"),
            capture_limit: 2000,
            min_labeled_for_evolve: 20,
            auto_evolve_step: 200,
        }
    }
}

struct Store {
    records: Vec<CaptureRecord>,
    path: PathBuf,
}

/// 进化编排器：样本存储 + 标注 + 微调闸门 + 热部署。
///
/// 实例自包含（无全局静态）：sidecar / 绑定层各持有一份；与 [`crate::SmartRouter`]
/// 共享同一个引擎 Arc（分类 `capture=true` 时由引擎侧回调捕获，部署成功后
/// 调引擎 `reload` 热生效）。
pub struct Evolution {
    config: EvolutionConfig,
    sink: RwLock<Option<Arc<dyn EvolutionEventSink>>>,
    engine: RwLock<Option<Arc<dyn crate::engine::ClassifyEngine>>>,
    store: Mutex<Store>,
    /// 进化进行中互斥（防止并发触发多个训练线程）。
    running: AtomicBool,
    /// 上次自动进化触发时的已标注样本数（持久化，重启不重复触发）。
    last_auto_mark: Mutex<usize>,
}

impl Evolution {
    /// Creates an evolution coordinator, loading existing samples from
    /// `<data_dir>/samples.jsonl` (missing file = empty store).
    pub fn new(config: EvolutionConfig) -> Self {
        if let Some(dir) = config.packs_dir.as_ref() {
            let _ = std::fs::create_dir_all(dir);
        }
        let path = config.data_dir.join("samples.jsonl");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let records = load_records(&path);
        log::info!(
            "[router-evolution] store loaded: {} records from {}",
            records.len(),
            path.display()
        );
        let last_mark = load_auto_mark(&config.data_dir.join("auto_state.json"));
        Self {
            store: Mutex::new(Store { records, path }),
            engine: RwLock::new(None),
            sink: RwLock::new(None),
            running: AtomicBool::new(false),
            last_auto_mark: Mutex::new(last_mark),
            config,
        }
    }

    /// Attaches the classification engine (for `num_embd` checks and head
    /// hot-reload after a successful deploy).
    pub fn set_engine(&self, engine: Arc<dyn crate::engine::ClassifyEngine>) {
        *self.engine.write().unwrap_or_else(|e| e.into_inner()) = Some(engine);
    }

    /// Attaches a progress sink (sidecar SSE / binding callbacks / UI emit).
    pub fn set_sink(&self, sink: Arc<dyn EvolutionEventSink>) {
        *self.sink.write().unwrap_or_else(|e| e.into_inner()) = Some(sink);
    }

    fn emit(&self, stage: &str, detail: &str) {
        if let Some(sink) = self.sink.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            sink.on_event(EvolutionEvent::Phase {
                stage: stage.to_string(),
                detail: detail.to_string(),
            });
        }
    }

    // -----------------------------------------------------------------------
    // 捕获（热路径）
    // -----------------------------------------------------------------------

    /// 捕获一次真实路由分类（引擎 `capture=true` 时调用）。任何失败静默降级
    /// （不影响路由）。超限 FIFO：一次性裁剪到上限的一半，避免频繁重写。
    pub fn capture_route_sample(
        &self,
        text: &str,
        hidden: &[f32],
        prev_tier: Option<u8>,
        probs: &[f32],
        num_embd: usize,
    ) {
        if text.trim().is_empty() || hidden.is_empty() || probs.len() != 4 {
            return;
        }
        let rec = CaptureRecord {
            ts: now_ms(),
            text: text.chars().take(2000).collect(),
            hidden_hex: hidden_to_hex(hidden),
            prev_tier,
            probs: [probs[0], probs[1], probs[2], probs[3]],
            num_embd,
            label: None,
            source: "route".to_string(),
        };
        let mut need_rewrite = false;
        let result = self.store.lock().map(|mut s| {
            s.records.push(rec);
            if s.records.len() > self.config.capture_limit {
                need_rewrite = true;
            } else if let Some(last) = s.records.last().cloned() {
                if let Err(e) = append_record(&mut s, &last) {
                    log::warn!("[router-evolution] capture append failed: {e}");
                }
            }
        });
        if let Err(e) = result {
            log::warn!("[router-evolution] capture lock poisoned: {e}");
            return;
        }
        if need_rewrite {
            let limit = self.config.capture_limit;
            let _ = self.store.lock().map(|mut s| {
                let drop_n = s.records.len() - limit / 2;
                s.records.drain(..drop_n);
                if let Err(e) = persist_all(&s) {
                    log::warn!("[router-evolution] capture rewrite failed: {e}");
                }
            });
        }
    }

    // -----------------------------------------------------------------------
    // 标注面板数据面（stats / list / label / delete / clear）
    // -----------------------------------------------------------------------

    pub fn capture_stats(&self) -> CaptureStats {
        let s = self.store.lock().unwrap_or_else(|e| e.into_inner());
        let labeled = s.records.iter().filter(|r| r.label.is_some()).count();
        CaptureStats {
            total: s.records.len(),
            labeled,
            unlabeled: s.records.len() - labeled,
            limit: self.config.capture_limit,
            min_labeled_for_evolve: self.config.min_labeled_for_evolve,
        }
    }

    /// 样本列表（最新优先）。
    pub fn capture_list(&self, offset: usize, limit: usize) -> Vec<CaptureItem> {
        let s = self.store.lock().unwrap_or_else(|e| e.into_inner());
        s.records
            .iter()
            .rev()
            .skip(offset)
            .take(limit.clamp(1, 200))
            .enumerate()
            .map(|(i, r)| CaptureItem {
                idx: s.records.len() - 1 - offset - i,
                ts: r.ts,
                text: r.text.chars().take(300).collect(),
                probs: r.probs,
                prev_tier: r.prev_tier,
                label: r.label,
                source: r.source.clone(),
            })
            .collect()
    }

    /// 标注正确层级（0-3）；`None` 取消标注。idx 为当前存储数组下标。
    /// 设值（非清除）后检查自动进化阈值。
    pub fn capture_label(self: &Arc<Self>, idx: usize, label: Option<u8>) -> Result<(), String> {
        if let Some(l) = label {
            if l > 3 {
                return Err(format!("invalid tier label {l} (0-3)"));
            }
        }
        {
            let mut s = self.store.lock().unwrap_or_else(|e| e.into_inner());
            let rec = s
                .records
                .get_mut(idx)
                .ok_or_else(|| format!("index {idx} out of range"))?;
            rec.label = label;
            persist_all(&s)?;
        }
        if label.is_some() {
            self.maybe_auto_evolve();
        }
        Ok(())
    }

    pub fn capture_delete(&self, idx: usize) -> Result<(), String> {
        let mut s = self.store.lock().unwrap_or_else(|e| e.into_inner());
        if idx >= s.records.len() {
            return Err(format!("index {idx} out of range"));
        }
        s.records.remove(idx);
        persist_all(&s)
    }

    pub fn capture_clear(&self) -> Result<(), String> {
        let mut s = self.store.lock().unwrap_or_else(|e| e.into_inner());
        s.records.clear();
        persist_all(&s)
    }

    // -----------------------------------------------------------------------
    // 自动进化：每新增 auto_evolve_step 条已标注样本自动触发一次微调
    // -----------------------------------------------------------------------

    /// 下一次自动进化阈值（stats 展示用）。
    pub fn next_auto_evolve_at(&self) -> usize {
        self.last_auto_mark() + self.config.auto_evolve_step
    }

    fn last_auto_mark(&self) -> usize {
        *self
            .last_auto_mark
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// 标注落库后调用：满足「新增标注 ≥ step」且无进化进行中时，后台触发
    /// 自动进化。
    ///
    /// 触发即推进标记并持久化（无论 ok/rejected/skipped）——闸门拒绝说明这
    /// 批数据无收益，凑满下一批再试；否则每次标注都会反复白跑训练。
    pub fn maybe_auto_evolve(self: &Arc<Self>) {
        let labeled = {
            let s = self.store.lock().unwrap_or_else(|e| e.into_inner());
            s.records.iter().filter(|r| r.label.is_some()).count()
        };
        let mark = self.last_auto_mark();
        if labeled < mark.saturating_add(self.config.auto_evolve_step) {
            return;
        }
        // 占坑失败 = 手动进化进行中：不推进标记，下次标注再查。
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        self.save_auto_mark(labeled);
        let this = Arc::clone(self);
        std::thread::spawn(move || {
            log::info!("[router-evolution] auto-evolve triggered ({labeled} labeled)");
            this.emit("loading", "auto-evolve started");
            let result = this.run_evolution_inner();
            this.running.store(false, Ordering::SeqCst);
            if let Some(sink) = this.sink.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
                sink.on_event(EvolutionEvent::Finished(result.clone()));
            }
            log::info!(
                "[router-evolution] auto-evolve finished: {} ({})",
                result.status,
                result.message
            );
        });
    }

    fn save_auto_mark(&self, n: usize) {
        *self
            .last_auto_mark
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = n;
        let path = self.config.data_dir.join("auto_state.json");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let body = serde_json::json!({ "last_mark": n }).to_string();
        let _ = std::fs::write(&path, body);
    }

    // -----------------------------------------------------------------------
    // 进化主流程（阻塞；独立线程由调用方/auto 触发负责）
    // -----------------------------------------------------------------------

    /// 手动触发一次进化（阻塞直到训练+闸门+部署完成；并发调用返回 skipped）。
    pub fn evolve(&self) -> EvolveResult {
        if self.running.swap(true, Ordering::SeqCst) {
            return EvolveResult {
                status: "skipped".to_string(),
                message: "evolution already running".to_string(),
                train_samples: 0,
                eval_samples: 0,
                epochs_run: 0,
                baseline_acc: 0.0,
                new_acc: 0.0,
            };
        }
        if let Some(sink) = self.sink.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            sink.on_event(EvolutionEvent::Started);
        }
        let result = self.run_evolution_inner();
        self.running.store(false, Ordering::SeqCst);
        if let Some(sink) = self.sink.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            sink.on_event(EvolutionEvent::Finished(result.clone()));
        }
        result
    }

    fn run_evolution_inner(&self) -> EvolveResult {
        let fail = |status: &str, message: String| EvolveResult {
            status: status.to_string(),
            message,
            train_samples: 0,
            eval_samples: 0,
            epochs_run: 0,
            baseline_acc: 0.0,
            new_acc: 0.0,
        };

        // 0. 加载当前头（其 base_dim 即期望骨干维度——头与骨干不匹配时分类
        //    侧本就会报错，进化以头文件为准）。
        self.emit("loading", "collecting labeled samples");
        let base = match RouterHead::from_json_file(&self.config.head_path) {
            Ok(h) => h,
            Err(e) => return fail("skipped", format!("load head failed: {e}")),
        };
        let expected_dim = base.expected_hidden_dim();

        // 1. 收集已标注样本（按头期望维度过滤）。
        let labeled: Vec<TrainingSample> = {
            let s = self.store.lock().unwrap_or_else(|e| e.into_inner());
            s.records
                .iter()
                .filter(|r| r.num_embd == expected_dim)
                .filter_map(|r| {
                    r.label.map(|label| TrainingSample {
                        hidden: r.hidden_f32(),
                        prev_tier: r.prev_tier,
                        label,
                    })
                })
                .collect()
        };
        if labeled.len() < self.config.min_labeled_for_evolve {
            return fail(
                "skipped",
                format!(
                    "need >= {} labeled samples, have {}",
                    self.config.min_labeled_for_evolve,
                    labeled.len()
                ),
            );
        }

        // 2. 加载闸门评估包。
        self.emit("loading", "loading eval pack");
        let packs_dir = match self.config.packs_dir.as_ref() {
            Some(d) => d,
            None => return fail("skipped", "eval pack dir not configured".to_string()),
        };
        let (pack_dim, eval_samples) = match load_pack_file(&packs_dir.join("eval_pack.json")) {
            Ok(v) => v,
            Err(e) => return fail("skipped", e),
        };
        if pack_dim != expected_dim {
            return fail(
                "skipped",
                format!("eval_pack base_dim {pack_dim} != head {expected_dim}"),
            );
        }

        // 2.5 加载回放池（与闸门集互斥；缺失则退化为纯捕获微调）。
        let replay = match load_replay_pool(&packs_dir.join("replay_pool.json")) {
            Ok((replay_dim, pool)) => {
                if replay_dim != 0 && replay_dim != expected_dim {
                    log::warn!(
                        "[router_evolution] replay_pool base_dim {replay_dim} != head {expected_dim}, replay disabled"
                    );
                    Vec::new()
                } else {
                    pool
                }
            }
            Err(e) => {
                log::warn!("[router_evolution] replay_pool load failed, replay disabled: {e}");
                Vec::new()
            }
        };
        let mut train_samples = labeled;
        if replay.is_empty() {
            log::info!(
                "[router_evolution] replay pool not deployed, training on captured samples only"
            );
        } else {
            // 1:1 回放（按类均衡），池子不足时全部混入。
            let mixed = replay_mix(&replay, train_samples.len());
            log::info!(
                "[router_evolution] replay mixing: {} replay + {} captured",
                mixed.len(),
                train_samples.len()
            );
            train_samples.extend(mixed);
        }

        // 3. 原位微调（冻结 mean/std，低学习率）。
        self.emit(
            "training",
            &format!("{} train samples", train_samples.len()),
        );
        let opts = FinetuneOptions::default();
        // 早停集不传 eval_pack（避免 selection bias 虚高闸门），训练器内部自切。
        let (new_head, report) = match finetune_head(&base, &train_samples, None, &opts) {
            Ok(v) => v,
            Err(e) => return fail("skipped", format!("finetune failed: {e}")),
        };

        // 4. 闸门：新头在 eval_pack 上不回退（容差 0.5%）才上线。
        self.emit("gating", "comparing heads on eval pack");
        let new_acc = eval_head_acc(&new_head, &eval_samples);
        // baseline（report.baseline_eval_acc 是训练切分上的，闸门用全量 pack 重算）。
        let baseline_acc = eval_head_acc(&base, &eval_samples);

        if new_acc < baseline_acc - 0.005 {
            let msg = format!(
                "gated: new head acc {new_acc:.4} < baseline {baseline_acc:.4} - 0.005; old head kept"
            );
            log::warn!("[router-evolution] {msg}");
            self.emit("done", &msg);
            return EvolveResult {
                status: "rejected".to_string(),
                message: msg,
                train_samples: report.train_samples,
                eval_samples: eval_samples.len(),
                epochs_run: report.epochs_run,
                baseline_acc,
                new_acc,
            };
        }

        // 5. 备份旧头 + 写入新头 + 热重载。
        let head_path = &self.config.head_path;
        let backup = head_path.with_extension("json.bak");
        if let Err(e) = std::fs::copy(head_path, &backup) {
            log::warn!("[router-evolution] head backup failed: {e}");
        }
        let weights = new_head.trainable();
        let new_json = serde_json::json!({
            "version": 1,
            "input_dim": new_head.input_dim(),
            "hidden_dim": new_head.hidden_dim(),
            "base_dim": new_head.expected_hidden_dim(),
            "mean": new_head.mean_iter().collect::<Vec<f32>>(),
            "std": new_head.std_iter().collect::<Vec<f32>>(),
            "w1": weights.w1, "b1": weights.b1,
            "ln_g": weights.ln_g, "ln_b": weights.ln_b,
            "w2": weights.w2, "b2": weights.b2,
        });
        if let Err(e) = std::fs::write(head_path, new_json.to_string()) {
            return fail("skipped", format!("write head failed: {e}"));
        }
        log::info!(
            "[router-evolution] evolved: acc {:.4} -> {:.4} ({} samples, {} epochs); reloading head",
            baseline_acc,
            new_acc,
            report.train_samples,
            report.epochs_run
        );
        self.emit("done", "head updated, reloading");
        // 引擎热重载（内置引擎覆盖 reload；宿主自实现引擎默认 no-op）。
        if let Some(engine) = self
            .engine
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            if let Err(e) = engine.reload() {
                log::warn!(
                    "[router-evolution] hot reload failed (restart or manual reload needed): {e}"
                );
            }
        }
        EvolveResult {
            status: "ok".to_string(),
            message: format!(
                "evolved: acc {:.4} -> {:.4} ({} labeled, {} epochs)",
                baseline_acc, new_acc, report.train_samples, report.epochs_run
            ),
            train_samples: report.train_samples,
            eval_samples: eval_samples.len(),
            epochs_run: report.epochs_run,
            baseline_acc,
            new_acc,
        }
    }
}

// ---------------------------------------------------------------------------
// 数据面类型（绑定/HTTP 层直接序列化）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CaptureStats {
    pub total: usize,
    pub labeled: usize,
    pub unlabeled: usize,
    pub limit: usize,
    pub min_labeled_for_evolve: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaptureItem {
    pub idx: usize,
    pub ts: u64,
    pub text: String,
    pub probs: [f32; 4],
    pub prev_tier: Option<u8>,
    pub label: Option<u8>,
    pub source: String,
}

/// 进化结果（status = "ok" | "rejected" | "skipped"）。
#[derive(Debug, Clone, Default, Serialize)]
pub struct EvolveResult {
    pub status: String,
    pub message: String,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub epochs_run: usize,
    pub baseline_acc: f32,
    pub new_acc: f32,
}

// ---------------------------------------------------------------------------
// 存储原语
// ---------------------------------------------------------------------------

fn load_records(path: &PathBuf) -> Vec<CaptureRecord> {
    let mut records = Vec::new();
    if let Ok(f) = std::fs::File::open(path) {
        let reader = std::io::BufReader::new(f);
        for line in reader.lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<CaptureRecord>(&line) {
                Ok(r) => records.push(r),
                Err(e) => log::warn!("[router-evolution] skip bad record: {e}"),
            }
        }
    }
    records
}

/// 全量重写存储文件（标注/删除/淘汰时；tmp + rename 原子替换）。
fn persist_all(store: &Store) -> Result<(), String> {
    let tmp = store.path.with_extension("jsonl.tmp");
    let content: String = store
        .records
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .map(|mut s| {
            s.push('\n');
            s
        })
        .collect();
    std::fs::write(&tmp, content).map_err(|e| format!("persist failed: {e}"))?;
    std::fs::rename(&tmp, &store.path).map_err(|e| format!("persist rename failed: {e}"))?;
    Ok(())
}

/// 单条追加（捕获热路径：打开-append-关闭，µs~ms 级）。
fn append_record(store: &mut Store, rec: &CaptureRecord) -> Result<(), String> {
    let line = serde_json::to_string(rec).map_err(|e| e.to_string())?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&store.path)
        .map_err(|e| format!("append open failed: {e}"))?;
    writeln!(f, "{line}").map_err(|e| format!("append write failed: {e}"))?;
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 上次自动进化标记（持久化；文件缺失/损坏 = 0）。
fn load_auto_mark(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("last_mark").and_then(|x| x.as_u64()))
        .unwrap_or(0) as usize
}

// ---------------------------------------------------------------------------
// 模型包（eval / replay）
// ---------------------------------------------------------------------------

fn load_pack_file(path: &Path) -> Result<(usize, Vec<TrainingSample>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    #[derive(Deserialize)]
    struct Pack {
        version: u32,
        base_dim: usize,
        samples: Vec<PackSample>,
    }
    #[derive(Deserialize)]
    struct PackSample {
        h: String,
        t: u8,
        p: Option<u8>,
    }
    let pack: Pack = serde_json::from_str(&text).map_err(|e| format!("pack parse failed: {e}"))?;
    if pack.version != 1 {
        return Err(format!("unsupported pack version {}", pack.version));
    }
    let mut samples = Vec::with_capacity(pack.samples.len());
    for s in &pack.samples {
        let raw = B64
            .decode(&s.h)
            .map_err(|e| format!("pack hidden decode failed: {e}"))?;
        let hidden: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        samples.push(TrainingSample {
            hidden,
            prev_tier: s.p,
            label: s.t,
        });
    }
    Ok((pack.base_dim, samples))
}

/// 回放池（与闸门集互斥切分，防灾难性遗忘）；未部署 = Ok(空)，退化为纯捕获微调。
fn load_replay_pool(path: &Path) -> Result<(usize, Vec<TrainingSample>), String> {
    if !path.exists() {
        return Ok((0, Vec::new()));
    }
    load_pack_file(path)
}

/// 按类均衡取 target 条回放样本（round-robin 逐类轮取，池序固定 = 结果确定）。
fn replay_mix(pool: &[TrainingSample], target: usize) -> Vec<TrainingSample> {
    if pool.is_empty() || target == 0 {
        return Vec::new();
    }
    let mut by_class: [Vec<&TrainingSample>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for s in pool {
        by_class[(s.label as usize).min(3)].push(s);
    }
    let mut out = Vec::with_capacity(target.min(pool.len()));
    let mut taken = [0usize; 4];
    while out.len() < target {
        let mut progressed = false;
        for (cls, rows) in by_class.iter().enumerate() {
            if out.len() >= target {
                break;
            }
            if taken[cls] < rows.len() {
                out.push(rows[taken[cls]].clone());
                taken[cls] += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    out
}

/// 头在样本集上的 top-1 准确率。
fn eval_head_acc(head: &RouterHead, samples: &[TrainingSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut hit = 0usize;
    for s in samples {
        if let Ok(probs) = head.forward(&s.hidden, s.prev_tier) {
            let argmax = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if argmax == s.label as usize {
                hit += 1;
            }
        }
    }
    hit as f32 / samples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mock::MockEngine;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "rwkv-router-test-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn toy_head_json() -> String {
        // v4 形状（base_dim=2，input_dim=7 = 2 + 5 one-hot），mean=0/std=1。
        // w1 前两列恒等投影（z0=x0, z1=x1），one-hot 维权重 0（忽略 prev_tier）。
        // ln_g=1/ln_b=0；w2（行主 [4][2]）：logits = [y0, y1, y0+y1, 0]。
        r#"{
            "version": 1, "input_dim": 7, "hidden_dim": 2, "base_dim": 2,
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "w1": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "b1": [0.0, 0.0],
            "ln_g": [1.0, 1.0], "ln_b": [0.0, 0.0],
            "w2": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#
        .to_string()
    }

    fn capture(evo: &Evolution, i: usize, num_embd: usize) {
        evo.capture_route_sample(
            &format!("request number {i} with enough text to be meaningful"),
            &[0.5, -0.25],
            None,
            &[0.1, 0.2, 0.3, 0.4],
            num_embd,
        );
    }

    #[test]
    fn capture_roundtrip_and_fifo_eviction() {
        let dir = temp_dir("fifo");
        let cfg = EvolutionConfig {
            data_dir: dir.clone(),
            capture_limit: 6,
            ..EvolutionConfig::default()
        };
        let evo = Evolution::new(cfg);
        for i in 0..8 {
            capture(&evo, i, 2);
        }
        // 7 条时触发裁剪到 3，第 8 条后 = 4。
        assert_eq!(evo.capture_stats().total, 4);
        // 重启等价：新实例从磁盘加载。
        drop(evo);
        let evo2 = Evolution::new(EvolutionConfig {
            data_dir: dir.clone(),
            capture_limit: 6,
            ..EvolutionConfig::default()
        });
        assert_eq!(evo2.capture_stats().total, 4);
        let items = evo2.capture_list(0, 10);
        assert_eq!(items.len(), 4);
        assert_eq!(
            items[0].text,
            "request number 7 with enough text to be meaningful"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn label_stats_and_validation() {
        let dir = temp_dir("label");
        let evo = Evolution::new(EvolutionConfig {
            data_dir: dir.clone(),
            ..EvolutionConfig::default()
        });
        for i in 0..3 {
            capture(&evo, i, 2);
        }
        let evo = Arc::new(evo);
        assert!(evo.capture_label(0, Some(2)).is_ok());
        assert!(evo.capture_label(0, Some(4)).is_err()); // 0-3 only
        assert!(evo.capture_label(99, Some(1)).is_err()); // out of range
        let stats = evo.capture_stats();
        assert_eq!(stats.labeled, 1);
        assert_eq!(stats.unlabeled, 2);
        // 取消标注。
        assert!(evo.capture_label(0, None).is_ok());
        assert_eq!(evo.capture_stats().labeled, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replay_mix_balances_classes() {
        let pool: Vec<TrainingSample> = (0..8)
            .map(|i| TrainingSample {
                hidden: vec![i as f32; 2],
                prev_tier: None,
                label: (i % 4) as u8, // 每类 2 条
            })
            .collect();
        let mixed = replay_mix(&pool, 6);
        assert_eq!(mixed.len(), 6);
        // round-robin：前 4 条各类一条，后 2 条轮到类 0/1。
        let counts = [0u32; 4];
        let mut counts = counts.map(|_| 0u32);
        for s in &mixed {
            counts[s.label as usize] += 1;
        }
        assert_eq!(counts, [2, 2, 1, 1]);
        // 池子不足：全部混入且顺序确定。
        let mixed2 = replay_mix(&pool, 100);
        assert_eq!(mixed2.len(), 8);
        assert_eq!(mixed2[0].label, 0);
        assert_eq!(mixed2[1].label, 1);
        // 空池。
        assert!(replay_mix(&[], 5).is_empty());
    }

    #[test]
    fn evolve_skips_without_pack_dir() {
        let dir = temp_dir("nopack");
        std::fs::write(dir.join("head.json"), toy_head_json()).unwrap();
        let evo = Evolution::new(EvolutionConfig {
            data_dir: dir.clone(),
            head_path: dir.join("head.json"),
            packs_dir: None,
            ..EvolutionConfig::default()
        });
        for i in 0..30 {
            capture(&evo, i, 2);
        }
        let r = evo.evolve();
        assert_eq!(r.status, "skipped");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn evolve_skips_with_insufficient_labels() {
        let dir = temp_dir("fewlabels");
        std::fs::write(dir.join("head.json"), toy_head_json()).unwrap();
        let packs = dir.join("packs");
        std::fs::create_dir_all(&packs).unwrap();
        let evo = Evolution::new(EvolutionConfig {
            data_dir: dir.clone(),
            head_path: dir.join("head.json"),
            packs_dir: Some(packs),
            ..EvolutionConfig::default()
        });
        for i in 0..5 {
            capture(&evo, i, 2);
        }
        let r = evo.evolve();
        assert_eq!(r.status, "skipped");
        assert!(r.message.contains("labeled samples"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn evolve_ok_path_deploys_head_and_fires_sink() {
        let dir = temp_dir("evolve-ok");
        let head_path = dir.join("head.json");
        std::fs::write(&head_path, toy_head_json()).unwrap();
        let packs = dir.join("packs");
        std::fs::create_dir_all(&packs).unwrap();

        // 参照头（算 eval pack 标签 = 基线 100%）。
        let base = RouterHead::from_json_str(&toy_head_json()).unwrap();

        // eval pack：24 条，标签 = 参照头 argmax（基线 acc = 1.0）。
        let mut pack_samples = Vec::new();
        for i in 0..24 {
            let hidden = vec![((i % 7) as f32 - 3.0) * 0.5, ((i % 5) as f32 - 2.0) * 0.5];
            let probs = base.forward(&hidden, None).unwrap();
            let label = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u8;
            pack_samples.push(serde_json::json!({
                "h": b64_f16_hexless(&hidden),
                "t": label,
                "p": null,
            }));
        }
        std::fs::write(
            packs.join("eval_pack.json"),
            serde_json::json!({"version": 1, "base_dim": 2, "samples": pack_samples}).to_string(),
        )
        .unwrap();

        let evo = Arc::new(Evolution::new(EvolutionConfig {
            data_dir: dir.clone(),
            head_path: head_path.clone(),
            packs_dir: Some(packs),
            ..EvolutionConfig::default()
        }));
        evo.set_engine(Arc::new(MockEngine::new([0.25; 4])));

        // 事件收集 sink。
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen2 = Arc::clone(&seen);
        struct Sink(Arc<Mutex<Vec<String>>>);
        impl EvolutionEventSink for Sink {
            fn on_event(&self, event: EvolutionEvent) {
                let s = match &event {
                    EvolutionEvent::Started => "started".to_string(),
                    EvolutionEvent::Phase { stage, .. } => stage.clone(),
                    EvolutionEvent::Finished(r) => format!("finished:{}", r.status),
                };
                self.0.lock().unwrap().push(s);
            }
        }
        evo.set_sink(Arc::new(Sink(seen2)));

        // 24 条已标注样本（标签 = 参照头 argmax → 训练不破坏基线）。
        for i in 0..24 {
            evo.capture_route_sample(
                &format!("sample {i} text for evolution training"),
                &[((i % 7) as f32 - 3.0) * 0.5, ((i % 5) as f32 - 2.0) * 0.5],
                None,
                &[0.25; 4],
                2,
            );
        }
        for idx in 0..24 {
            let hidden = vec![
                ((idx % 7) as f32 - 3.0) * 0.5,
                ((idx % 5) as f32 - 2.0) * 0.5,
            ];
            let probs = base.forward(&hidden, None).unwrap();
            let label = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u8;
            assert!(evo.capture_label(idx, Some(label)).is_ok());
        }

        let r = evo.evolve();
        assert_eq!(r.status, "ok", "message: {}", r.message);
        assert_eq!(r.eval_samples, 24);
        assert!(r.baseline_acc >= 0.999);

        // 新头落盘 + 备份存在 + 可解析。
        let reloaded = RouterHead::from_json_file(&head_path).unwrap();
        assert_eq!(reloaded.expected_hidden_dim(), 2);
        assert!(head_path.with_extension("json.bak").exists());

        // sink 事件齐全。
        let events = seen.lock().unwrap().clone();
        assert!(events.contains(&"started".to_string()));
        assert!(events.contains(&"training".to_string()));
        assert!(events.contains(&"gating".to_string()));
        assert!(events.contains(&"finished:ok".to_string()));

        // 并发触发互斥：evolve 进行中不可能（此处已结束），但 repeated ok 合法。
        std::fs::remove_dir_all(&dir).ok();
    }

    /// eval pack 的 hidden 用 base64(f16 LE)（与 load_pack_file 对齐）。
    fn b64_f16_hexless(hidden: &[f32]) -> String {
        let mut buf = Vec::with_capacity(hidden.len() * 2);
        for &v in hidden {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes()[..2]);
        }
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(&buf)
    }
}
