//! Trained MLP classification head for the smart router.
//!
//! Ports the state-embedding classifier from the rwkv-router paper path
//! (`paper/scripts/03_classification.py`, test_acc=0.9325): a two-layer MLP
//! over the mean-pooled last-layer hidden state of the RWKV backbone.
//!
//! Pipeline (matching the PyTorch training exactly):
//!   standardize (x-mean)/std -> Linear -> GELU -> LayerNorm -> Linear -> softmax
//!
//! Weight JSON contract (exported by `client/scripts/router_head/train_mlp.py`):
//! ```json
//! { "version": 1, "input_dim": 2560, "hidden_dim": 256,
//!   "mean": [input_dim], "std": [input_dim],
//!   "w1": [hidden_dim * input_dim], "b1": [hidden_dim],   // w1 row-major [out][in]
//!   "ln_g": [hidden_dim], "ln_b": [hidden_dim],
//!   "w2": [4 * hidden_dim], "b2": [4] }                    // w2 row-major [out][in]
//! ```
//! Linear weights use the PyTorch `nn.Linear.weight` layout ([out_features,
//! in_features] row-major), so `y = x @ W^T + b` — the exporter can flatten
//! the tensor directly without transposing.
//!
//! v4 optional field `base_dim`: when present, `input_dim == base_dim + 5` and
//! the head expects `hidden[base_dim] ‖ prev_tier_onehot[5]` (one-hot R0-R3,
//! index 4 = None/first-turn). `forward(hidden, prev_tier)` builds the one-hot
//! from the runtime sticky-tier value. Heads WITHOUT `base_dim` keep the v1
//! format (prev_tier ignored) — old head files run unchanged on new code.

use serde::Deserialize;
use std::path::Path;

pub(crate) const LN_EPS: f32 = 1e-5;
pub(crate) const STD_MIN: f32 = 1e-6;
const MAX_HEAD_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// v4 prev_tier one-hot 维度：R0-R3 + None（首轮/未知）。
pub const PREV_TIER_DIM: usize = 5;
/// one-hot 中 None（首轮/未知）的下标。
pub const PREV_TIER_NONE_IDX: usize = 4;

/// Trained router classification head (pure logic, no GPU dependency).
#[derive(Debug, Clone)]
pub struct RouterHead {
    input_dim: usize,
    hidden_dim: usize,
    /// v4: hidden 维度（input_dim = base_dim + 5 prev_tier one-hot）。
    /// None = v1 老 head（纯 hidden 输入，prev_tier 忽略）。
    base_dim: Option<usize>,
    mean: Vec<f32>,
    std: Vec<f32>,
    /// Linear1 weight, row-major [hidden_dim][input_dim].
    w1: Vec<f32>,
    b1: Vec<f32>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    /// Linear2 weight, row-major [4][hidden_dim].
    w2: Vec<f32>,
    b2: Vec<f32>,
}

/// 可训练参数快照（微调器读写用）。
///
/// 刻意不含 mean/std：特征标准化统计冻结不变，是增量微调防遗忘的
/// 核心约束（旧头 = 初始化点，新样本沿用同一特征分布）。
#[derive(Debug, Clone)]
pub struct TrainableWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub ln_g: Vec<f32>,
    pub ln_b: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

#[derive(Deserialize)]
struct HeadJson {
    version: u32,
    input_dim: usize,
    hidden_dim: usize,
    #[serde(default)]
    base_dim: Option<usize>,
    mean: Vec<f32>,
    std: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl RouterHead {
    /// Loads a head from a JSON weight file.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let meta = std::fs::metadata(path).map_err(|e| format!("router head stat failed: {e}"))?;
        if meta.len() > MAX_HEAD_FILE_BYTES {
            return Err(format!("router head file too large: {} bytes", meta.len()));
        }
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("router head read failed: {e}"))?;
        Self::from_json_str(&text)
    }

    /// Parses a head from a JSON string.
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let raw: HeadJson =
            serde_json::from_str(text).map_err(|e| format!("router head parse failed: {e}"))?;
        if raw.version != 1 {
            return Err(format!("unsupported router head version: {}", raw.version));
        }
        if let Some(b) = raw.base_dim {
            if raw.input_dim != b + PREV_TIER_DIM {
                return Err(format!(
                    "router head input_dim {} != base_dim {b} + {} (prev-tier one-hot)",
                    raw.input_dim, PREV_TIER_DIM
                ));
            }
        }
        let head = RouterHead {
            input_dim: raw.input_dim,
            hidden_dim: raw.hidden_dim,
            base_dim: raw.base_dim,
            mean: raw.mean,
            std: raw.std,
            w1: raw.w1,
            b1: raw.b1,
            ln_g: raw.ln_g,
            ln_b: raw.ln_b,
            w2: raw.w2,
            b2: raw.b2,
        };
        head.validate()?;
        Ok(head)
    }

    fn validate(&self) -> Result<(), String> {
        let (i, h) = (self.input_dim, self.hidden_dim);
        let checks = [
            (self.mean.len(), i, "mean"),
            (self.std.len(), i, "std"),
            (self.w1.len(), h * i, "w1"),
            (self.b1.len(), h, "b1"),
            (self.ln_g.len(), h, "ln_g"),
            (self.ln_b.len(), h, "ln_b"),
            (self.w2.len(), 4 * h, "w2"),
            (self.b2.len(), 4, "b2"),
        ];
        for (actual, expected, name) in checks {
            if actual != expected {
                return Err(format!(
                    "router head field '{name}' has {actual} elements, expected {expected}"
                ));
            }
        }
        if i == 0 || h == 0 {
            return Err("router head dims must be positive".to_string());
        }
        Ok(())
    }

    /// Expected hidden-state dimension (must equal the backbone n_embd).
    /// v4 head（有 base_dim）期望 hidden 长度 = base_dim；老 head = input_dim。
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// 期望的 hidden 维度：v4 = base_dim，v1 = input_dim。
    /// 引擎侧据此校验 head/模型匹配（v4 head 的 input_dim 含 5 维 one-hot）。
    pub fn expected_hidden_dim(&self) -> usize {
        self.base_dim.unwrap_or(self.input_dim)
    }

    /// MLP hidden layer width.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Classifies a mean-pooled hidden state into R0-R3 probabilities.
    /// `prev_tier`：上一轮路由等级（sticky 表，0-3）；None = 首轮/未知。
    /// v4 head（有 base_dim）拼接 one-hot[5]（R0-R3 + None 位）；老 head 忽略。
    pub fn forward(&self, hidden: &[f32], prev_tier: Option<u8>) -> Result<[f32; 4], String> {
        let expected = self.expected_hidden_dim();
        if hidden.len() != expected {
            return Err(format!(
                "hidden dim {} != head expected {expected} (model/head mismatch)",
                hidden.len()
            ));
        }

        // v4：拼接 prev_tier one-hot（index 0-3 = R0-R3，4 = None/首轮）。
        let x: Vec<f32> = match self.base_dim {
            Some(_) => {
                let mut v = Vec::with_capacity(self.input_dim);
                v.extend_from_slice(hidden);
                let idx = match prev_tier {
                    Some(t) if t < 4 => t as usize,
                    _ => PREV_TIER_NONE_IDX,
                };
                for i in 0..PREV_TIER_DIM {
                    v.push(if i == idx { 1.0 } else { 0.0 });
                }
                v
            }
            None => hidden.to_vec(),
        };

        // 1. Standardize with train-set statistics.
        let x: Vec<f32> = x
            .iter()
            .zip(&self.mean)
            .zip(&self.std)
            .map(|((&h, &m), &s)| (h - m) / s.max(STD_MIN))
            .collect();

        // 2. Linear1: z = x @ w1^T + b1.
        let mut z = self.b1.clone();
        for (j, acc) in z.iter_mut().enumerate() {
            let row = &self.w1[j * self.input_dim..(j + 1) * self.input_dim];
            let mut sum = *acc;
            for (wv, xv) in row.iter().zip(&x) {
                sum += wv * xv;
            }
            *acc = sum;
        }

        // 3. GELU (tanh approximation; max error vs exact ~3e-4).
        for v in z.iter_mut() {
            *v = gelu(*v);
        }

        // 4. LayerNorm (eps=1e-5, matching PyTorch default).
        let mu: f32 = z.iter().sum::<f32>() / z.len() as f32;
        let var: f32 = z.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / z.len() as f32;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        let y: Vec<f32> = z
            .iter()
            .zip(&self.ln_g)
            .zip(&self.ln_b)
            .map(|((&v, &g), &b)| (v - mu) * inv_std * g + b)
            .collect();

        // 5. Linear2: logits = y @ w2^T + b2, then softmax.
        let mut logits = [0.0f32; 4];
        for (j, acc) in logits.iter_mut().enumerate() {
            let row = &self.w2[j * self.hidden_dim..(j + 1) * self.hidden_dim];
            let mut sum = self.b2[j];
            for (wv, yv) in row.iter().zip(&y) {
                sum += wv * yv;
            }
            *acc = sum;
        }
        Ok(softmax4(&logits))
    }

    /// Exports the trainable parameter snapshot (mean/std excluded, frozen).
    pub fn trainable(&self) -> TrainableWeights {
        TrainableWeights {
            w1: self.w1.clone(),
            b1: self.b1.clone(),
            ln_g: self.ln_g.clone(),
            ln_b: self.ln_b.clone(),
            w2: self.w2.clone(),
            b2: self.b2.clone(),
        }
    }

    /// Rebuilds the head with replaced trainable weights (mean/std/dims kept).
    pub fn with_trainable(&self, w: TrainableWeights) -> Result<RouterHead, String> {
        let candidate = RouterHead {
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            base_dim: self.base_dim,
            mean: self.mean.clone(),
            std: self.std.clone(),
            w1: w.w1,
            b1: w.b1,
            ln_g: w.ln_g,
            ln_b: w.ln_b,
            w2: w.w2,
            b2: w.b2,
        };
        candidate.validate()?;
        Ok(candidate)
    }

    /// Whether this head takes the v4 form (prev_tier one-hot input).
    pub fn is_v4(&self) -> bool {
        self.base_dim.is_some()
    }

    /// Iterators over the frozen standardization statistics (微调器读取用).
    pub fn mean_iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.mean.iter().copied()
    }

    pub fn std_iter(&self) -> impl Iterator<Item = f32> + '_ {
        self.std.iter().copied()
    }
}

pub(crate) fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn softmax4(logits: &[f32; 4]) -> [f32; 4] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: [f32; 4] = std::array::from_fn(|i| (logits[i] - max).exp());
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        std::array::from_fn(|i| exps[i] / sum)
    } else {
        [0.25; 4]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tiny head (input=2, hidden=2) with known weights so the
    /// forward pass can be verified by hand.
    fn toy_head() -> RouterHead {
        // mean=0, std=1 -> standardization is identity.
        // w1 (row-major [2][2]): z0 = x0, z1 = x1 (identity Linear).
        // ln_g=1, ln_b=0 -> LN is pure normalization.
        // w2 (row-major [4][2]): logits = [y0, y1, y0+y1, 0].
        let json = r#"{
            "version": 1, "input_dim": 2, "hidden_dim": 2,
            "mean": [0.0, 0.0], "std": [1.0, 1.0],
            "w1": [1.0, 0.0, 0.0, 1.0], "b1": [0.0, 0.0],
            "ln_g": [1.0, 1.0], "ln_b": [0.0, 0.0],
            "w2": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        RouterHead::from_json_str(json).expect("toy head should parse")
    }

    #[test]
    fn forward_matches_hand_computation() {
        let head = toy_head();
        // x = [1, 1] -> z = [gelu(1), gelu(1)] (equal) -> LN -> y = [0, 0]
        // logits = [0, 0, 0, 0] -> uniform softmax.
        let probs = head.forward(&[1.0, 1.0], None).unwrap();
        for p in probs {
            assert!((p - 0.25).abs() < 1e-6, "{probs:?}");
        }
    }

    #[test]
    fn forward_distinguishes_directions() {
        let head = toy_head();
        // x = [3, 0]: z = [gelu(3), 0] -> LN -> y = [+1, -1]
        //   logits = [1, -1, 0, 0] -> argmax 0.
        let probs0 = head.forward(&[3.0, 0.0], None).unwrap();
        // x = [0, 3]: mirrored -> logits = [-1, 1, 0, 0] -> argmax 1.
        let probs1 = head.forward(&[0.0, 3.0], None).unwrap();
        let argmax = |p: &[f32; 4]| {
            p.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(argmax(&probs0), 0, "{probs0:?}");
        assert_eq!(argmax(&probs1), 1, "{probs1:?}");
        assert!(probs0[0] > 0.5, "{probs0:?}");
        assert!(probs1[1] > 0.5, "{probs1:?}");
    }

    #[test]
    fn standardization_is_applied() {
        let json = r#"{
            "version": 1, "input_dim": 1, "hidden_dim": 1,
            "mean": [2.0], "std": [0.5],
            "w1": [1.0], "b1": [0.0],
            "ln_g": [1.0], "ln_b": [0.0],
            "w2": [1.0, 0.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        let head = RouterHead::from_json_str(json).unwrap();
        // x = (3-2)/0.5 = 2 -> z = gelu(2) -> LN(单元素) = 0 -> logits 全 0 -> uniform。
        let probs = head.forward(&[3.0], None).unwrap();
        assert!((probs[0] - 0.25).abs() < 1e-6, "{probs:?}");
    }

    #[test]
    fn rejects_dim_mismatch_and_bad_fields() {
        let head = toy_head();
        assert!(head.forward(&[1.0], None).is_err());
        assert!(head.forward(&[1.0, 2.0, 3.0], None).is_err());

        let bad = r#"{
            "version": 1, "input_dim": 2, "hidden_dim": 2,
            "mean": [0.0, 0.0], "std": [1.0, 1.0],
            "w1": [1.0, 0.0], "b1": [0.0, 0.0],
            "ln_g": [1.0, 1.0], "ln_b": [0.0, 0.0],
            "w2": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 1.0]
        }"#;
        assert!(RouterHead::from_json_str(bad).is_err());
        assert!(RouterHead::from_json_str("{}").is_err());
    }

    #[test]
    fn input_dim_reported() {
        assert_eq!(toy_head().input_dim(), 2);
        assert_eq!(toy_head().expected_hidden_dim(), 2);
    }

    /// v4 契约：base_dim 存在 → input_dim = base_dim + 5；hidden 长度 = base_dim；
    /// prev_tier one-hot 参与 forward（不同 prev_tier → 不同概率）。
    fn toy_head_v4() -> RouterHead {
        // input_dim = 1 + 5 = 6：hidden[0] + onehot[5]。
        // w1 行 [0]（hidden）权重 1，onehot 维度 0-4 权重 [2,2,2,2,2] —— z = x0 + 2*onehot 激活位。
        let json = r#"{
            "version": 1, "input_dim": 6, "hidden_dim": 1, "base_dim": 1,
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "w1": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0], "b1": [0.0],
            "ln_g": [1.0], "ln_b": [0.0],
            "w2": [1.0, 0.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        RouterHead::from_json_str(json).expect("v4 toy head should parse")
    }

    #[test]
    fn v4_contract_parses_and_validates_dims() {
        let head = toy_head_v4();
        assert_eq!(head.input_dim(), 6);
        assert_eq!(head.expected_hidden_dim(), 1);
        // hidden 长度须为 base_dim（1），不是 input_dim（6）。
        assert!(head.forward(&[1.0], None).is_ok());
        assert!(head.forward(&[1.0, 2.0], None).is_err());
        assert!(head.forward(&[1.0; 6], None).is_err());

        // base_dim 与 input_dim 不一致 → 拒绝。
        let bad = r#"{
            "version": 1, "input_dim": 5, "hidden_dim": 1, "base_dim": 1,
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0, 1.0],
            "w1": [1.0, 2.0, 2.0, 2.0, 2.0], "b1": [0.0],
            "ln_g": [1.0], "ln_b": [0.0],
            "w2": [1.0, 0.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        assert!(RouterHead::from_json_str(bad).is_err());
    }

    #[test]
    fn v4_prev_tier_changes_output() {
        // hidden[2] + onehot[5]：w1 第 2 行接 onehot 维度 0（R0 位，权重 3）——
        // prev_tier=R0 时 z1 非零，LN 归一化后 logits 改变 → 概率改变。
        let json = r#"{
            "version": 1, "input_dim": 7, "hidden_dim": 2, "base_dim": 2,
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "w1": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], "b1": [0.0, 0.0],
            "ln_g": [1.0, 1.0], "ln_b": [0.0, 0.0],
            "w2": [1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0, 0.0]
        }"#;
        let head = RouterHead::from_json_str(json).unwrap();
        let a = head.forward(&[1.0, 0.0], Some(0)).unwrap(); // z=[gelu(1), gelu(3)]
        let b = head.forward(&[1.0, 0.0], Some(1)).unwrap(); // z=[gelu(1), 0]
        assert_ne!(a[0], b[0], "prev_tier must change probabilities");
        assert_ne!(a[1], b[1], "{a:?} vs {b:?}");
        // 非法 prev_tier（>3）按 None 处理（index 4）。
        let c = head.forward(&[1.0, 0.0], Some(9)).unwrap();
        let d = head.forward(&[1.0, 0.0], None).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn v1_head_ignores_prev_tier() {
        let head = toy_head();
        // 老 head（无 base_dim）：prev_tier 参数完全不影响输出。
        let a = head.forward(&[3.0, 0.0], None).unwrap();
        let b = head.forward(&[3.0, 0.0], Some(3)).unwrap();
        assert_eq!(a, b);
    }
}
