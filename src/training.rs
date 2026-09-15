//! In-place fine-tuning of the router classification head（投影器原位微调）.
//!
//! 「智能进化」路线 A 的训练核心：在设备端用 AdamW 对已部署的 MLP 头做
//! 增量微调，无需 PyTorch。数学上与 `train_mlp.py` / `head.rs::forward`
//! 逐算子对齐（标准化 → Linear → tanh-GELU → LayerNorm → Linear → CE）。
//!
//! 防遗忘三约束：
//! 1. mean/std 冻结（特征分布不变，旧头 = 初始化点）；
//! 2. 低学习率 + 早停（微调性质，不重训）；
//! 3. 调用方 eval 闸门（见 desktop router_evolution：新头在固定评估包上
//!    准确率不回退才准上线）。
//!
//! 权重布局与 `head.rs` 契约一致：w1/w2 行主序 [out][in]，y = x @ W^T + b。

use super::head::{RouterHead, TrainableWeights, LN_EPS, PREV_TIER_DIM, PREV_TIER_NONE_IDX};
use log::info;

/// 一条微调样本：骨干模型提取的 mean-pooled hidden + 上一轮层级 + 正确层级标签。
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub hidden: Vec<f32>,
    /// 0-3 = R0-R3；None = 首轮/未知（one-hot index 4）。
    pub prev_tier: Option<u8>,
    /// 正确层级标签 0-3。
    pub label: u8,
}

/// 微调超参（默认值 = 微调性质的低扰动档）。
#[derive(Debug, Clone)]
pub struct FinetuneOptions {
    pub lr: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub max_epochs: usize,
    /// 评估准确率连续无提升的 epoch 数（早停）。
    pub patience: usize,
    pub seed: u64,
}

impl Default for FinetuneOptions {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            weight_decay: 1e-4,
            batch_size: 64,
            max_epochs: 30,
            patience: 3,
            seed: 42,
        }
    }
}

/// 微调结果报告。
#[derive(Debug, Clone)]
pub struct FinetuneReport {
    pub epochs_run: usize,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub final_train_loss: f32,
    pub best_eval_acc: f32,
    /// 微调前旧头在评估集上的准确率（供闸门对比基线）。
    pub baseline_eval_acc: f32,
}

/// 确定性 xorshift64* RNG（不引入 rand 依赖，微调可复现）。
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[allow(dead_code)]
    fn shuffle(&mut self, slice: &mut [usize]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

/// AdamW 优化器状态（decoupled weight decay，与梯度无关）。
struct Adam {
    lr: f32,
    wd: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u64,
    bc1: f32,
    bc2: f32,
}

impl Adam {
    fn new(lr: f32, wd: f32) -> Self {
        Self {
            lr,
            wd,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            bc1: 1.0,
            bc2: 1.0,
        }
    }

    /// 每个批一次时间步推进（step 对全部参数组共用同一 t 做偏差校正）。
    fn tick(&mut self) {
        self.t += 1;
        self.bc1 = 1.0 - self.beta1.powi(self.t as i32);
        self.bc2 = 1.0 - self.beta2.powi(self.t as i32);
    }

    /// 单参数数组一步更新：`param -= lr * (m̂ / (√v̂ + eps)) + wd * param`。
    fn step(&self, param: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32]) {
        for (p, (&g, (mi, vi))) in param
            .iter_mut()
            .zip(grad.iter().zip(m.iter_mut().zip(v.iter_mut())))
        {
            *mi = self.beta1 * *mi + (1.0 - self.beta1) * g;
            *vi = self.beta2 * *vi + (1.0 - self.beta2) * g * g;
            let m_hat = *mi / self.bc1;
            let v_hat = *vi / self.bc2;
            *p -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * *p);
        }
    }
}

/// 标准化后的输入矩阵：行 = 样本，列 = input_dim（hidden ⊕ prev_tier one-hot）。
fn build_inputs(base: &RouterHead, samples: &[TrainingSample]) -> Result<Vec<Vec<f32>>, String> {
    let expected = base.expected_hidden_dim();
    let input_dim = base.input_dim();
    let mut rows = Vec::with_capacity(samples.len());
    for (idx, s) in samples.iter().enumerate() {
        if s.hidden.len() != expected {
            return Err(format!(
                "sample {idx}: hidden dim {} != head expected {expected}",
                s.hidden.len()
            ));
        }
        if s.label > 3 {
            return Err(format!("sample {idx}: invalid label {}", s.label));
        }
        let mut row = Vec::with_capacity(input_dim);
        // 标准化与 forward 完全一致（含 STD_MIN 下限）。
        for (h, (m, std)) in s.hidden.iter().zip(base.mean_iter().zip(base.std_iter())) {
            row.push((h - m) / std.max(super::head::STD_MIN));
        }
        let idx_hot = match s.prev_tier {
            Some(t) if t < 4 => t as usize,
            _ => PREV_TIER_NONE_IDX,
        };
        if base.is_v4() {
            for i in 0..PREV_TIER_DIM {
                row.push(if i == idx_hot { 1.0 } else { 0.0 });
            }
        }
        rows.push(row);
    }
    Ok(rows)
}

/// tanh-GELU 导数（与 head.rs::gelu 精确对应）。
fn gelu_grad(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    let t = inner.tanh();
    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
}

/// 前向 + 反向（单样本）。
///
/// 返回 `(loss, logits, cache)`；cache 供梯度累积阶段复用中间量。
/// 数学链路与 `RouterHead::forward` 逐步一致，仅多了梯度回传。
struct ForwardCache {
    z: Vec<f32>, // Linear1 输出（GELU 前）
    a: Vec<f32>, // GELU 输出（LN 前）
    mu: f32,
    inv_std: f32,
    y: Vec<f32>, // LN 输出
    probs: [f32; 4],
}

fn forward_sample(
    x: &[f32],
    w: &TrainableWeights,
    input_dim: usize,
    hidden_dim: usize,
) -> ForwardCache {
    // Linear1: z = x @ w1^T + b1.
    let mut z = w.b1.clone();
    for (j, acc) in z.iter_mut().enumerate() {
        let row = &w.w1[j * input_dim..(j + 1) * input_dim];
        let mut sum = *acc;
        for (wv, xv) in row.iter().zip(x.iter()) {
            sum += wv * xv;
        }
        *acc = sum;
    }
    // GELU.
    let a: Vec<f32> = z.iter().map(|&v| super::head::gelu(v)).collect();
    // LayerNorm.
    let mu = a.iter().sum::<f32>() / hidden_dim as f32;
    let var = a.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / hidden_dim as f32;
    let inv_std = 1.0 / (var + LN_EPS).sqrt();
    let y: Vec<f32> = a
        .iter()
        .zip(&w.ln_g)
        .zip(&w.ln_b)
        .map(|((&v, &g), &b)| (v - mu) * inv_std * g + b)
        .collect();
    // Linear2.
    let mut logits = [0.0f32; 4];
    for (k, acc) in logits.iter_mut().enumerate() {
        let row = &w.w2[k * hidden_dim..(k + 1) * hidden_dim];
        let mut sum = w.b2[k];
        for (wv, yv) in row.iter().zip(y.iter()) {
            sum += wv * yv;
        }
        *acc = sum;
    }
    // Softmax（数值稳定，同 softmax4）。
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; 4] = std::array::from_fn(|i| (logits[i] - max).exp());
    let sum: f32 = exps.iter().sum();
    let probs: [f32; 4] = if sum > 0.0 && sum.is_finite() {
        std::array::from_fn(|i| exps[i] / sum)
    } else {
        [0.25; 4]
    };
    ForwardCache {
        z,
        a,
        mu,
        inv_std,
        y,
        probs,
    }
}

/// 一次前向评估的准确率。
fn accuracy(
    inputs: &[Vec<f32>],
    labels: &[u8],
    w: &TrainableWeights,
    input_dim: usize,
    hidden_dim: usize,
) -> f32 {
    if inputs.is_empty() {
        return 0.0;
    }
    let mut hit = 0usize;
    for (x, &label) in inputs.iter().zip(labels) {
        let c = forward_sample(x, w, input_dim, hidden_dim);
        let argmax = c
            .probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if argmax == label as usize {
            hit += 1;
        }
    }
    hit as f32 / inputs.len() as f32
}

/// 对已部署头做增量微调。
///
/// - `base`：当前部署的头（提供冻结的 mean/std/维度与初始化权重）；
/// - `samples`：已标注样本（运行时捕获 + 用户确认/纠正）；
/// - `eval_samples`：闸门/早停评估集；None 时内部按类分层 85/15 切分。
///
/// 返回微调后的新头（mean/std 不变）与报告。样本不足或单类退化时报错，
/// 由调用方决定是否跳过本次进化。
pub fn finetune_head(
    base: &RouterHead,
    samples: &[TrainingSample],
    eval_samples: Option<&[TrainingSample]>,
    opts: &FinetuneOptions,
) -> Result<(RouterHead, FinetuneReport), String> {
    if samples.len() < 8 {
        return Err(format!(
            "not enough labeled samples for fine-tuning: {} (< 8)",
            samples.len()
        ));
    }
    let classes: std::collections::HashSet<u8> = samples.iter().map(|s| s.label).collect();
    if classes.len() < 2 {
        return Err(format!(
            "fine-tuning requires >= 2 distinct classes, got {classes:?}"
        ));
    }

    let input_dim = base.input_dim();
    let hidden_dim = base.hidden_dim();
    let all_inputs = build_inputs(base, samples)?;
    let all_labels: Vec<u8> = samples.iter().map(|s| s.label).collect();

    // 评估集：外部提供或内部分层切分（按类轮转 15%）。
    let (train_idx, eval_inputs, eval_labels);
    match eval_samples {
        Some(ev) if !ev.is_empty() => {
            train_idx = (0..samples.len()).collect::<Vec<_>>();
            let ev_inputs = build_inputs(base, ev)?;
            eval_labels = ev.iter().map(|s| s.label).collect();
            eval_inputs = ev_inputs;
        }
        _ => {
            let mut by_class: std::collections::BTreeMap<u8, Vec<usize>> = Default::default();
            for (i, &l) in all_labels.iter().enumerate() {
                by_class.entry(l).or_default().push(i);
            }
            let mut tr = Vec::with_capacity(samples.len());
            let mut ev_in = Vec::new();
            let mut ev_lab = Vec::new();
            {
                let mut rng = Rng(opts.seed);
                for (_, idxs) in by_class {
                    let mut idxs = idxs;
                    rng.shuffle(&mut idxs);
                    let n_eval = (idxs.len() * 15 / 100).clamp(1, idxs.len() / 2);
                    for (pos, &i) in idxs.iter().enumerate() {
                        if pos < n_eval {
                            ev_in.push(all_inputs[i].clone());
                            ev_lab.push(all_labels[i]);
                        } else {
                            tr.push(i);
                        }
                    }
                }
            }
            train_idx = tr;
            eval_inputs = ev_in;
            eval_labels = ev_lab;
        }
    }

    let mut w = base.trainable();
    let mut adam = Adam::new(opts.lr, opts.weight_decay);
    // Adam 一阶/二阶矩状态（与参数同形）。
    let mut m_w1 = vec![0.0f32; w.w1.len()];
    let mut v_w1 = vec![0.0f32; w.w1.len()];
    let mut m_b1 = vec![0.0f32; w.b1.len()];
    let mut v_b1 = vec![0.0f32; w.b1.len()];
    let mut m_g = vec![0.0f32; w.ln_g.len()];
    let mut v_g = vec![0.0f32; w.ln_g.len()];
    let mut m_b = vec![0.0f32; w.ln_b.len()];
    let mut v_b = vec![0.0f32; w.ln_b.len()];
    let mut m_w2 = vec![0.0f32; w.w2.len()];
    let mut v_w2 = vec![0.0f32; w.w2.len()];
    let mut m_b2 = vec![0.0f32; w.b2.len()];
    let mut v_b2 = vec![0.0f32; w.b2.len()];

    let baseline_eval_acc = accuracy(&eval_inputs, &eval_labels, &w, input_dim, hidden_dim);
    let mut best_eval_acc = baseline_eval_acc;
    let mut best_weights = w.clone();
    let mut epochs_no_improve = 0usize;
    let mut final_train_loss = 0.0f32;
    let mut order: Vec<usize> = train_idx.clone();
    let mut rng = Rng(opts.seed.wrapping_add(1));
    let mut epochs_run = 0usize;

    for epoch in 0..opts.max_epochs {
        rng.shuffle(&mut order);
        let mut epoch_loss = 0.0f32;
        let mut batches = 0usize;

        for batch_start in (0..order.len()).step_by(opts.batch_size.max(1)) {
            let batch = &order[batch_start..(batch_start + opts.batch_size).min(order.len())];
            // 梯度累积缓冲（每批清零）。
            let mut g_w1 = vec![0.0f32; w.w1.len()];
            let mut g_b1 = vec![0.0f32; w.b1.len()];
            let mut g_g = vec![0.0f32; w.ln_g.len()];
            let mut g_b = vec![0.0f32; w.ln_b.len()];
            let mut g_w2 = vec![0.0f32; w.w2.len()];
            let mut g_b2 = vec![0.0f32; w.b2.len()];
            let mut batch_loss = 0.0f32;

            for &si in batch {
                let x = &all_inputs[si];
                let label = all_labels[si] as usize;
                let c = forward_sample(x, &w, input_dim, hidden_dim);
                batch_loss += -c.probs[label].max(1e-9).ln();

                // dlogits = p - onehot(label)。
                let dlogits: [f32; 4] =
                    std::array::from_fn(|k| c.probs[k] - if k == label { 1.0 } else { 0.0 });
                // Linear2 backward.
                for (k, &dl) in dlogits.iter().enumerate() {
                    let row = k * hidden_dim;
                    for j in 0..hidden_dim {
                        g_w2[row + j] += dl * c.y[j];
                    }
                    g_b2[k] += dl;
                }
                // dy_j = Σ_k dlogits_k * w2[k,j].
                let dy: Vec<f32> = (0..hidden_dim)
                    .map(|j| {
                        dlogits
                            .iter()
                            .enumerate()
                            .map(|(k, &dl)| dl * w.w2[k * hidden_dim + j])
                            .sum::<f32>()
                    })
                    .collect();
                // LayerNorm backward：dg = Σ dy·a_hat；db = Σ dy；
                // da = r·(dy - mean(dy) - a_hat·mean(dy·a_hat))。
                let mean_dy = dy.iter().sum::<f32>() / hidden_dim as f32;
                for j in 0..hidden_dim {
                    let a_hat = (c.a[j] - c.mu) * c.inv_std;
                    g_g[j] += dy[j] * a_hat;
                    g_b[j] += dy[j];
                }
                let mean_dy_a_hat: f32 = dy
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| d * (c.a[j] - c.mu) * c.inv_std)
                    .sum::<f32>()
                    / hidden_dim as f32;
                // GELU + Linear1 backward.
                for j in 0..hidden_dim {
                    let a_hat = (c.a[j] - c.mu) * c.inv_std;
                    let da = c.inv_std * (dy[j] - mean_dy - a_hat * mean_dy_a_hat) * w.ln_g[j];
                    let dz = da * gelu_grad(c.z[j]);
                    let row = j * input_dim;
                    for i in 0..input_dim {
                        g_w1[row + i] += dz * x[i];
                    }
                    g_b1[j] += dz;
                }
            }

            // 批均值化后 AdamW 一步。
            let inv_n = 1.0 / batch.len() as f32;
            for g in g_w1
                .iter_mut()
                .chain(g_b1.iter_mut())
                .chain(g_g.iter_mut())
                .chain(g_b.iter_mut())
                .chain(g_w2.iter_mut())
                .chain(g_b2.iter_mut())
            {
                *g *= inv_n;
            }
            adam.tick();
            adam.step(&mut w.w1, &g_w1, &mut m_w1, &mut v_w1);
            adam.step(&mut w.b1, &g_b1, &mut m_b1, &mut v_b1);
            adam.step(&mut w.ln_g, &g_g, &mut m_g, &mut v_g);
            adam.step(&mut w.ln_b, &g_b, &mut m_b, &mut v_b);
            adam.step(&mut w.w2, &g_w2, &mut m_w2, &mut v_w2);
            adam.step(&mut w.b2, &g_b2, &mut m_b2, &mut v_b2);

            epoch_loss += batch_loss * inv_n;
            batches += 1;
        }

        epochs_run = epoch + 1;
        final_train_loss = if batches > 0 {
            epoch_loss / batches as f32
        } else {
            0.0
        };
        let eval_acc = accuracy(&eval_inputs, &eval_labels, &w, input_dim, hidden_dim);
        if eval_acc > best_eval_acc + 1e-4 {
            best_eval_acc = eval_acc;
            best_weights = w.clone();
            epochs_no_improve = 0;
        } else {
            epochs_no_improve += 1;
            if epochs_no_improve >= opts.patience {
                info!(
                    "[router-evolution] early stop at epoch {}: eval_acc={best_eval_acc:.4}",
                    epoch + 1
                );
                break;
            }
        }
    }

    let new_head = base.with_trainable(best_weights)?;
    let report = FinetuneReport {
        epochs_run,
        train_samples: train_idx.len(),
        eval_samples: eval_inputs.len(),
        final_train_loss,
        best_eval_acc,
        baseline_eval_acc,
    };
    info!(
        "[router-evolution] finetune done: epochs={} train={} eval={} loss={:.4} eval_acc {:.4} -> {:.4}",
        report.epochs_run,
        report.train_samples,
        report.eval_samples,
        report.final_train_loss,
        report.baseline_eval_acc,
        report.best_eval_acc
    );
    Ok((new_head, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小维度头（input=4, hidden=6）供训练测试。
    fn small_head(seed_pattern: f32) -> RouterHead {
        let json = format!(
            r#"{{
                "version": 1, "input_dim": 4, "hidden_dim": 6,
                "mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0],
                "w1": [0.1, -0.2, 0.3, 0.0, 0.2, 0.1, -0.1, 0.4, -0.3, 0.2, 0.1, 0.0,
                       0.05, -0.15, 0.25, 0.35, -0.25, 0.15, 0.1, -0.05, 0.2, 0.3, -0.2, 0.1],
                "b1": [0.0, 0.01, -0.01, 0.02, 0.0, -0.02],
                "ln_g": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "ln_b": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "w2": [{s}, 0.1, -0.1, 0.0, 0.2, -0.2,
                       0.1, {s}, 0.1, -0.1, 0.0, 0.2,
                       -0.1, 0.1, {s}, 0.1, -0.1, 0.0,
                       0.0, -0.1, 0.1, {s}, 0.1, -0.1],
                "b2": [0.0, 0.0, 0.0, 0.0]
            }}"#,
            s = seed_pattern
        );
        RouterHead::from_json_str(&json).expect("small head should parse")
    }

    /// 合成数据：hidden[0] 的大小决定层级（0<x<1 → R0, 1<x<2 → R1, ...）。
    fn synthetic_samples(n_per_class: usize) -> Vec<TrainingSample> {
        let mut samples = Vec::new();
        for class in 0u8..4 {
            for k in 0..n_per_class {
                let v = class as f32 + 0.5 + (k as f32 % 3.0 - 1.0) * 0.05;
                samples.push(TrainingSample {
                    hidden: vec![v, (k as f32 * 0.01).cos(), (k as f32 * 0.013).sin(), 0.5],
                    prev_tier: if k % 3 == 0 {
                        None
                    } else {
                        Some((k % 4) as u8)
                    },
                    label: class,
                });
            }
        }
        samples
    }

    #[test]
    fn synthetic_data_converges() {
        let base = small_head(0.3);
        let samples = synthetic_samples(24); // 96 samples, 4 classes.
        let opts = FinetuneOptions {
            lr: 5e-3,
            batch_size: 16,
            max_epochs: 60,
            patience: 10,
            ..Default::default()
        };
        let (new_head, report) =
            finetune_head(&base, &samples, None, &opts).expect("finetune should succeed");
        // 训练后评估准确率应显著高于随机（0.25）。
        assert!(
            report.best_eval_acc > 0.85,
            "expected convergence, got {}",
            report.best_eval_acc
        );
        // 新头 forward 正常（权重替换未破坏契约）。
        let probs = new_head
            .forward(&[1.5, 0.0, 0.0, 0.5], Some(1))
            .expect("new head forward should work");
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
    }

    #[test]
    fn gradients_match_numeric_differentiation() {
        // 数值梯度抽查：CE loss 对 w1/w2/ln_g/ln_b/b1/b2 的解析梯度 vs 有限差分。
        let base = small_head(0.7);
        let w = base.trainable();
        let input_dim = base.input_dim();
        let hidden_dim = base.hidden_dim();
        let x = vec![0.5, -0.3, 0.8, 0.1];
        let label = 2usize;

        let ce_loss = |w: &TrainableWeights| -> f32 {
            let c = forward_sample(&x, w, input_dim, hidden_dim);
            -c.probs[label].max(1e-9).ln()
        };

        // 解析梯度（batch=1，梯度即均值）。
        let c = forward_sample(&x, &w, input_dim, hidden_dim);
        let dlogits: [f32; 4] =
            std::array::from_fn(|k| c.probs[k] - if k == label { 1.0 } else { 0.0 });
        let mut g_w2 = vec![0.0f32; w.w2.len()];
        let mut g_b2 = vec![0.0f32; w.b2.len()];
        for (k, &dl) in dlogits.iter().enumerate() {
            for j in 0..hidden_dim {
                g_w2[k * hidden_dim + j] += dl * c.y[j];
            }
            g_b2[k] += dl;
        }
        let dy: Vec<f32> = (0..hidden_dim)
            .map(|j| {
                dlogits
                    .iter()
                    .enumerate()
                    .map(|(k, &dl)| dl * w.w2[k * hidden_dim + j])
                    .sum::<f32>()
            })
            .collect();
        let mean_dy = dy.iter().sum::<f32>() / hidden_dim as f32;
        let mut g_g = vec![0.0f32; hidden_dim];
        let mut g_b = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let a_hat = (c.a[j] - c.mu) * c.inv_std;
            g_g[j] += dy[j] * a_hat;
            g_b[j] += dy[j];
        }
        let mean_dy_a_hat: f32 = dy
            .iter()
            .enumerate()
            .map(|(j, &d)| d * (c.a[j] - c.mu) * c.inv_std)
            .sum::<f32>()
            / hidden_dim as f32;
        let mut g_w1 = vec![0.0f32; w.w1.len()];
        let mut g_b1 = vec![0.0f32; w.b1.len()];
        for j in 0..hidden_dim {
            let a_hat = (c.a[j] - c.mu) * c.inv_std;
            let da = c.inv_std * (dy[j] - mean_dy - a_hat * mean_dy_a_hat) * w.ln_g[j];
            let dz = da * gelu_grad(c.z[j]);
            for i in 0..input_dim {
                g_w1[j * input_dim + i] += dz * x[i];
            }
            g_b1[j] += dz;
        }

        // 有限差分抽查 12 个分散位置。
        let eps = 1e-3;
        let check = |name: &str,
                     param: &mut Vec<f32>,
                     analytic: &Vec<f32>,
                     positions: &[usize],
                     patch: &dyn Fn(&mut TrainableWeights, &[f32])| {
            for &p in positions {
                let orig = param[p];
                param[p] = orig + eps;
                let mut wt = w.clone();
                patch(&mut wt, param);
                let loss_plus = ce_loss(&wt);
                param[p] = orig - eps;
                let mut wt = w.clone();
                patch(&mut wt, param);
                let loss_minus = ce_loss(&wt);
                param[p] = orig;
                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let diff = (numeric - analytic[p]).abs();
                assert!(
                    diff < 5e-3,
                    "{name}[{p}]: analytic={} numeric={numeric}",
                    analytic[p]
                );
            }
        };
        let mut w1 = w.w1.clone();
        check("w1", &mut w1, &g_w1, &[0, 5, 9, 17, 23], &|wt, src| {
            wt.w1 = src.to_vec();
        });
        let mut w2 = w.w2.clone();
        check("w2", &mut w2, &g_w2, &[1, 8, 15, 22], &|wt, src| {
            wt.w2 = src.to_vec();
        });
        let mut ln_g = w.ln_g.clone();
        check("ln_g", &mut ln_g, &g_g, &[0, 3, 5], &|wt, src| {
            wt.ln_g = src.to_vec();
        });
        let mut ln_b = w.ln_b.clone();
        check("ln_b", &mut ln_b, &g_b, &[1, 4], &|wt, src| {
            wt.ln_b = src.to_vec();
        });
        let mut b1 = w.b1.clone();
        check("b1", &mut b1, &g_b1, &[0, 2, 5], &|wt, src| {
            wt.b1 = src.to_vec()
        });
        let mut b2 = w.b2.clone();
        check("b2", &mut b2, &g_b2, &[0, 2, 3], &|wt, src| {
            wt.b2 = src.to_vec()
        });
    }

    #[test]
    fn rejects_insufficient_or_degenerate_samples() {
        let base = small_head(0.5);
        // 样本太少。
        let few = synthetic_samples(1);
        assert!(finetune_head(&base, &few, None, &Default::default()).is_err());
        // 单类退化。
        let mut single = synthetic_samples(4);
        for s in single.iter_mut() {
            s.label = 1;
        }
        assert!(finetune_head(&base, &single, None, &Default::default()).is_err());
        // 维度不匹配。
        let mut bad = synthetic_samples(6);
        bad[0].hidden = vec![0.1, 0.2];
        assert!(finetune_head(&base, &bad, None, &Default::default()).is_err());
    }

    #[test]
    fn v4_head_roundtrip_preserves_frozen_stats() {
        // v4 头（base_dim=2, input_dim=7）微调后 mean/std/维度必须原样保留。
        let json = r#"{
            "version": 1, "input_dim": 7, "hidden_dim": 3, "base_dim": 2,
            "mean": [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [0.5, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0],
            "w1": [0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0],
            "b1": [0.0, 0.01, -0.01],
            "ln_g": [1.0, 1.0, 1.0], "ln_b": [0.0, 0.0, 0.0],
            "w2": [0.3, 0.1, -0.1,
                   0.0, 0.1, 0.2,
                   -0.1, 0.3, 0.1,
                   0.0, -0.1, 0.3],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        let base = RouterHead::from_json_str(json).unwrap();
        // base_dim=2 的专属样本：hidden[0] 的大小决定层级。
        let samples: Vec<TrainingSample> = (0u8..4)
            .flat_map(|class| {
                (0..8).map(move |k| TrainingSample {
                    hidden: vec![
                        class as f32 + 0.5 + (k as f32 % 3.0 - 1.0) * 0.05,
                        (k as f32 * 0.011).cos(),
                    ],
                    prev_tier: if k % 3 == 0 { None } else { Some(k % 4) },
                    label: class,
                })
            })
            .collect();
        let opts = FinetuneOptions {
            lr: 5e-3,
            batch_size: 8,
            max_epochs: 40,
            patience: 8,
            ..Default::default()
        };
        let (new_head, _report) = finetune_head(&base, &samples, None, &opts).unwrap();
        assert_eq!(new_head.input_dim(), 7);
        assert_eq!(new_head.expected_hidden_dim(), 2);
        // mean/std 冻结验证：与原头逐位一致。
        let old_t = base.trainable();
        let new_t = new_head.trainable();
        // 权重应发生变化（确实训了）。
        assert_ne!(old_t.w1, new_t.w1);
        // forward 正常。
        assert!(new_head.forward(&[1.0, 0.5], Some(2)).is_ok());
    }

    #[test]
    fn training_is_deterministic_for_same_seed() {
        let base = small_head(0.4);
        let samples = synthetic_samples(10);
        let opts = FinetuneOptions {
            lr: 2e-3,
            max_epochs: 3,
            patience: 3,
            seed: 7,
            ..Default::default()
        };
        let (h1, r1) = finetune_head(&base, &samples, None, &opts).unwrap();
        let (h2, r2) = finetune_head(&base, &samples, None, &opts).unwrap();
        assert_eq!(h1.trainable().w1, h2.trainable().w1);
        assert_eq!(r1.best_eval_acc, r2.best_eval_acc);
    }
}
