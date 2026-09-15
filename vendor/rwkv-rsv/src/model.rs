//! RWKV-7 模型定义：权重结构、状态、前向推理
//! 严格对齐 numpy baseline.py 实现

use half::f16;

pub const LN_EPS: f32 = 1.0e-5;
pub const GN_EPS: f32 = 64.0e-5;
pub const L2_EPS: f32 = 1.0e-12;

/// 校准激活采集器（Phase D/Phase C：校准加权 k-means 与 nnq 输出域 LUT 优化的来源）。
///
/// 在 CPU fp32 前向中，逐 token 采集 6 大类量化矩阵的**输入激活样本**（每矩阵存至多
/// `cap` 个 [lens[j]] 向量），最后导出 safetensors 供 `tools/quantize_any4.py --calib`
/// 读取（shape [S, lens[j]]，S=实际采集样本数）。
/// 6 类矩阵的输入激活（收缩维）：
///   att.receptance ← xr[C]，att.key ← xk[C]，att.value ← xv[C]，
///   att.output     ← y_g = y·g [C]，ffn.key ← xb[C]，ffn.value ← r2=relu²(xb@fk) [fh]
#[derive(Debug)]
pub struct CalibCollector {
    /// 每类矩阵的收缩维长度 [c,c,c,c,c,fh]
    lens: [usize; 6],
    /// samples[layer][j] = 至多 cap 个长度 lens[j] 的激活向量
    samples: Vec<[Vec<Vec<f32>>; 6]>,
    cap: usize,
    count: usize,
}

static CALIB_NAMES: [&str; 6] = [
    "att.receptance.weight",
    "att.key.weight",
    "att.value.weight",
    "att.output.weight",
    "ffn.key.weight",
    "ffn.value.weight",
];

impl CalibCollector {
    pub fn new(n_layer: usize, c: usize, fh: usize, cap: usize) -> Self {
        let lens = [c, c, c, c, c, fh];
        let samples = (0..n_layer)
            .map(|_| std::array::from_fn(|_| Vec::with_capacity(cap)))
            .collect();
        Self {
            lens,
            samples,
            cap,
            count: 0,
        }
    }

    /// 是否已采集满 cap 个样本（满后不再采集，避免长 prompt 无界占用）。
    pub fn full(&self) -> bool {
        self.count >= self.cap
    }

    /// 采集一个 token 的 6 类输入激活样本（由调用方保证未满时调用）。
    #[allow(clippy::too_many_arguments)]
    fn accum(
        &mut self,
        layer: usize,
        xr: &[f32],
        xk: &[f32],
        xv: &[f32],
        y_g: &[f32],
        xb: &[f32],
        r2: &[f32],
    ) {
        let s = &mut self.samples[layer];
        s[0].push(xr.to_vec());
        s[1].push(xk.to_vec());
        s[2].push(xv.to_vec());
        s[3].push(y_g.to_vec());
        s[4].push(xb.to_vec());
        s[5].push(r2.to_vec());
    }

    /// 每 token 结束时调用一次，推进 token 计数（full() 据此判断）。
    fn count_token(&mut self) {
        self.count += 1;
    }

    /// 序列化为 safetensors：键=blocks.{li}.{name}，形状 [count, lens[j]]。
    pub fn serialize(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use safetensors::tensor::{Dtype, TensorView};
        let mut owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (li, s) in self.samples.iter().enumerate() {
            for (j, name) in CALIB_NAMES.iter().enumerate() {
                let n = s[j].len();
                let len = self.lens[j];
                let mut flat = Vec::with_capacity(n * len);
                for v in &s[j] {
                    debug_assert_eq!(v.len(), len);
                    flat.extend_from_slice(v);
                }
                owned.push((
                    format!("blocks.{li}.{name}"),
                    bytemuck::cast_slice(&flat).to_vec(),
                    vec![n, len],
                ));
            }
        }
        let mut views: Vec<(String, TensorView)> = Vec::with_capacity(owned.len());
        for (key, bytes, shape) in &owned {
            views.push((
                key.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ));
        }
        Ok(safetensors::serialize(views, None)?)
    }
}

/// 模型超参数
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: usize,
    pub n_embd: usize, // C
    pub vocab: usize,
    pub head_size: usize, // N = 64
    pub n_head: usize,    // H = C / N
    // 低秩中间维度（随模型变化，从 safetensors 形状推导）
    pub ffn_hidden: usize, // FFN 隐藏维度
    pub w_mid: usize,      // w 低秩中维度
    pub a_mid: usize,      // a 低秩中维度
    pub v_mid: usize,      // v 低秩中维度
    pub g_mid: usize,      // g 低秩中维度
}

/// RWKV-7 权重（从 safetensors 加载，全部转 f32）
#[derive(Debug)]
pub struct Model {
    pub config: ModelConfig,
    pub emb_weight: Vec<f32>, // [vocab, C]
    pub ln0_w: Vec<f32>,
    pub ln0_b: Vec<f32>,
    pub ln_out_w: Vec<f32>,
    pub ln_out_b: Vec<f32>,
    pub head_weight: Vec<f32>, // [C, vocab] (预转置)
    pub layers: Vec<Layer>,
    /// 可选校准激活采集器（`enable_calib` 开启；`&self` 下用 RefCell 内部可变）。
    pub calib: std::cell::RefCell<Option<CalibCollector>>,
}

#[derive(Debug)]
pub struct Layer {
    // ln1
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    // att — token shift coefficients
    pub x_r: Vec<f32>,
    pub x_w: Vec<f32>,
    pub x_k: Vec<f32>,
    pub x_v: Vec<f32>,
    pub x_a: Vec<f32>,
    pub x_g: Vec<f32>,
    // att — biases
    pub w0: Vec<f32>,
    pub a0: Vec<f32>,
    pub v0: Vec<f32>,
    // att — low-rank weights [in, out] (no transpose needed)
    pub w1: Vec<f32>,
    pub w2: Vec<f32>, // w1: [C, mid_w], w2: [mid_w, C]
    pub a1: Vec<f32>,
    pub a2: Vec<f32>,
    pub v1: Vec<f32>,
    pub v2: Vec<f32>,
    pub g1: Vec<f32>,
    pub g2: Vec<f32>,
    // att — element-wise
    pub r_k: Vec<f32>, // [H, N]
    pub k_k: Vec<f32>,
    pub k_a: Vec<f32>, // [C]
    // att — linear weights (预转置为 [in, out]，forward 时直接用 matvec)
    pub receptance_w: Vec<f32>, // [C, C] (原 PyTorch [out, in] 已转置)
    pub key_w: Vec<f32>,        // [C, C]
    pub value_w: Vec<f32>,      // [C, C]
    pub output_w: Vec<f32>,     // [C, C]
    pub ln_x_w: Vec<f32>,
    pub ln_x_b: Vec<f32>,
    // ln2
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    // ffn (预转置为 [in, out])
    pub ffn_x_k: Vec<f32>,
    pub ffn_key_w: Vec<f32>,   // [C, 3072] (原 [3072, C] 已转置)
    pub ffn_value_w: Vec<f32>, // [3072, C] (原 [C, 3072] 已转置)
}

/// 每层 RNN 状态
#[derive(Debug, Clone)]
pub struct LayerState {
    pub tmix_x: Vec<f32>,   // [C] token shift
    pub tmix_rnn: Vec<f32>, // [H, N, N] DPLR state
    pub cmix_x: Vec<f32>,   // [C] token shift
}

impl LayerState {
    pub fn new(n_embd: usize, n_head: usize, head_size: usize) -> Self {
        Self {
            tmix_x: vec![0.0; n_embd],
            tmix_rnn: vec![0.0; n_head * head_size * head_size],
            cmix_x: vec![0.0; n_embd],
        }
    }
}

/// safetensors view → Vec<f32>
fn tensor_to_f32(data: &safetensors::tensor::TensorView) -> Vec<f32> {
    match data.dtype() {
        safetensors::tensor::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data.data()).to_vec(),
        safetensors::tensor::Dtype::F16 => {
            let f16s: &[f16] = bytemuck::cast_slice(data.data());
            f16s.iter().map(|x| x.to_f32()).collect()
        }
        d => panic!("unsupported dtype: {d:?}"),
    }
}

/// 矩阵转置: [rows, cols] → [cols, rows] (行主序)
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// int8 CPU 反量化（非对称 per-group=128，与 tools/quantize_any4.py --bits 8 及 GPU 侧一致）：
/// `w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]`。
/// - idx: [M*K/4] uint32，每 uint32 打包 4 个 uint8 权重（低位字节在前：b0=byte0 … b3=byte3）
/// - sz:  [M*K/128] uint32（scale fp16 低 16 位 | zero fp16 高 16 位）
fn dequant_int8(idx: &[u32], sz: &[u32], m: usize, k: usize) -> Vec<f32> {
    let kg = k / 128;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        let row = &mut w[r * k..(r + 1) * k];
        for (g, chunk) in row.chunks_mut(128).enumerate() {
            let szv = sz[r * kg + g];
            let scale = f16::from_bits((szv & 0xFFFF) as u16).to_f32();
            let zero = f16::from_bits((szv >> 16) as u16).to_f32();
            for (j, wv) in chunk.iter_mut().enumerate() {
                let ki = g * 128 + j;
                let pack = idx[r * (k / 4) + ki / 4];
                let q = ((pack >> ((ki % 4) * 8)) & 0xFF) as f32;
                *wv = scale * q + zero;
            }
        }
    }
    w
}

/// 读线性权重 [out, in]（行主序 f32）。
/// 路由：原 fp16/fp32 键 > int8 反量化（与 gpu_model.rs 一致性）。
fn linear_to_f32(
    st: &safetensors::SafeTensors,
    key: &str,
    m: usize,
    k: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if let Ok(t) = st.tensor(key) {
        return Ok(tensor_to_f32(&t));
    }
    let idx_t = st.tensor(&format!("{key}.int8_idx"))?;
    let sz_t = st.tensor(&format!("{key}.int8_sz"))?;
    let idx: &[u32] = bytemuck::cast_slice(idx_t.data());
    let sz: &[u32] = bytemuck::cast_slice(sz_t.data());
    assert_eq!(idx.len(), m * k / 4, "{key}.int8_idx 形状不符");
    assert_eq!(sz.len(), m * k / 128, "{key}.int8_sz 形状不符");
    log::info!("{key}: 原键缺失，使用 int8 反量化（{m}x{k}）");
    Ok(dequant_int8(idx, sz, m, k))
}

impl Model {
    pub fn from_safetensors(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;

        let n_layer = 1 + max_layer(&st);
        let emb = st.tensor("emb.weight")?;
        let (vocab, n_embd) = (emb.shape()[0], emb.shape()[1]);
        // head_size / n_head 从 r_k 形状 [H, N] 推导（跨模型通用）
        let rk = st.tensor("blocks.0.att.r_k")?;
        let n_head = rk.shape()[0];
        let head_size = rk.shape()[1];

        // 低秩中间维度从各 low-rank 权重形状推导（一维不等于 C 的那个）
        let mid_of = |t: &safetensors::tensor::TensorView| -> usize {
            let s = t.shape();
            if s[0] == n_embd { s[1] } else { s[0] }
        };
        let w_mid = mid_of(&st.tensor("blocks.0.att.w1")?);
        let a_mid = mid_of(&st.tensor("blocks.0.att.a1")?);
        let v_mid = mid_of(&st.tensor("blocks.0.att.v1")?);
        let g_mid = mid_of(&st.tensor("blocks.0.att.g1")?);
        // 原 fp16 键可能被 int8 量化键替换，此时从 int8_idx [M, K] 推导 M
        let ffn_hidden = match st.tensor("blocks.0.ffn.key.weight") {
            Ok(t) => t.shape()[0],
            Err(_) => st.tensor("blocks.0.ffn.key.weight.int8_idx")?.shape()[0],
        };

        let config = ModelConfig {
            n_layer,
            n_embd,
            vocab,
            head_size,
            n_head,
            ffn_hidden,
            w_mid,
            a_mid,
            v_mid,
            g_mid,
        };
        log::info!(
            "model: n_layer={n_layer} n_embd={n_embd} vocab={vocab} n_head={n_head} head_size={head_size} \
             ffn_hidden={ffn_hidden} w_mid={w_mid} a_mid={a_mid} v_mid={v_mid} g_mid={g_mid}"
        );

        let emb_weight = tensor_to_f32(&emb);
        let ln0_w = tensor_to_f32(&st.tensor("blocks.0.ln0.weight")?);
        let ln0_b = tensor_to_f32(&st.tensor("blocks.0.ln0.bias")?);
        let ln_out_w = tensor_to_f32(&st.tensor("ln_out.weight")?);
        let ln_out_b = tensor_to_f32(&st.tensor("ln_out.bias")?);
        // head.weight 原始 [vocab, C] → 预转置为 [C, vocab]；原键缺失时回退 int8 反量化
        let head_raw = linear_to_f32(&st, "head.weight", vocab, n_embd)?;
        let head_weight = transpose(&head_raw, vocab, n_embd);

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(Layer::load(&st, i, n_embd, &config)?);
        }

        Ok(Self {
            config,
            emb_weight,
            ln0_w,
            ln0_b,
            ln_out_w,
            ln_out_b,
            head_weight,
            layers,
            calib: std::cell::RefCell::new(None),
        })
    }

    /// 开启校准激活采集（cap：每矩阵最多采集的样本数）。
    pub fn enable_calib(&self, cap: usize) {
        *self.calib.borrow_mut() = Some(CalibCollector::new(
            self.config.n_layer,
            self.config.n_embd,
            self.config.ffn_hidden,
            cap,
        ));
    }

    /// 导出已采集的校准激活样本为 safetensors 字节。
    pub fn dump_calib(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let calib = self.calib.borrow();
        let c = calib.as_ref().ok_or("calib not enabled")?;
        c.serialize()
    }

    pub fn init_state(&self) -> Vec<LayerState> {
        let c = self.config.n_embd;
        let (h, n) = (self.config.n_head, self.config.head_size);
        (0..self.config.n_layer)
            .map(|_| LayerState::new(c, h, n))
            .collect()
    }

    /// 前向推理，返回最后一个 token 的 logits
    pub fn forward(&self, tokens: &[u32], state: &mut [LayerState]) -> Vec<f32> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        let (wm, am, vm, gm, fh) = (
            self.config.w_mid,
            self.config.a_mid,
            self.config.v_mid,
            self.config.g_mid,
            self.config.ffn_hidden,
        );
        let emb_ln = layer_norm_rows(
            &self.emb_weight,
            &self.ln0_w,
            &self.ln0_b,
            c,
            self.config.vocab,
            LN_EPS,
        );

        let mut logits = vec![0.0f32; self.config.vocab];

        for &token in tokens {
            let mut x: Vec<f32> = emb_ln[token as usize * c..(token as usize + 1) * c].to_vec();
            let mut v_first: Option<Vec<f32>> = None;

            for (i, layer) in self.layers.iter().enumerate() {
                // ===== Time Mixing =====
                let ln1 = layer_norm(&x, &layer.ln1_w, &layer.ln1_b, LN_EPS);
                let prev = state[i].tmix_x.clone();
                state[i].tmix_x = ln1.clone();

                let xr = lerp(&ln1, &prev, &layer.x_r);
                let xw = lerp(&ln1, &prev, &layer.x_w);
                let xk = lerp(&ln1, &prev, &layer.x_k);
                let xv = lerp(&ln1, &prev, &layer.x_v);
                let xa = lerp(&ln1, &prev, &layer.x_a);
                let xg = lerp(&ln1, &prev, &layer.x_g);

                // r/k/v: 权重已预转置为 [in, out]，直接 matvec
                let r = matvec(&xr, &layer.receptance_w, c, c); // [C]
                let k = matvec(&xk, &layer.key_w, c, c);
                let v = matvec(&xv, &layer.value_w, c, c);

                // v_first 逻辑
                let v = match &v_first {
                    None => {
                        v_first = Some(v.clone());
                        v
                    }
                    Some(vf) => {
                        // v = LERP(v, v_first, sigmoid(v0 + xv @ v1 @ v2))
                        let mid = matvec(&xv, &layer.v1, c, vm); // [vm]
                        let full = matvec(&mid, &layer.v2, vm, c); // [C]
                        let gate = sigmoid(&add_bias(&full, &layer.v0));
                        lerp(&v, vf, &gate)
                    }
                };

                // w = exp(-sigmoid(w0 + tanh(xw @ w1) @ w2) / sqrt(e))
                let w_mid = matvec(&xw, &layer.w1, c, wm); // [wm]
                let w_tanh = tanh_vec(&w_mid);
                let w_full = matvec(&w_tanh, &layer.w2, wm, c); // [C]
                let w_sig = sigmoid(&add_bias(&w_full, &layer.w0));
                let w = exp_vec(&scale_vec(&w_sig, -1.0f32 / std::f32::consts::E.sqrt()));

                // a = sigmoid(a0 + xa @ a1 @ a2)
                let a_mid = matvec(&xa, &layer.a1, c, am); // [am]
                let a_full = matvec(&a_mid, &layer.a2, am, c); // [C]
                let a = sigmoid(&add_bias(&a_full, &layer.a0));

                // kk = k * k_k, then L2 normalize
                let kk = mul_vec(&k, &layer.k_k);
                // k = LERP(k, k * a, k_a)
                let k_a_vec = mul_vec(&k, &a);
                let k_mod = lerp(&k, &k_a_vec, &layer.k_a);

                // reshape to [H, N]
                // w, k, v, kk, a are [C] = [H*N]
                // r_k is already [H, N]

                // kk L2 normalization (per group)
                let kk_l2 = l2_norm_groups(&kk, h, n);

                // DPLR: S = S * w + (S @ A) * B + V * K^T
                // A = kk_l2, B = -kk_l2 * a
                let b_vec = mul_vec(&kk_l2, &neg_vec(&a)); // -kk_l2 * a

                let rnn = &mut state[i].tmix_rnn;
                dplr_update(rnn, &k_mod, &v, &w, &kk_l2, &b_vec, h, n);

                // y = S @ r
                let y = dplr_matvec(rnn, &r, h, n);

                // Group norm
                let y = group_norm(&y, &layer.ln_x_w, &layer.ln_x_b, h, n, GN_EPS);

                // y += sum(r * k * r_k, axis=1, keepdims=True) * v
                let extra = sum_rk_rk(&r, &k_mod, &layer.r_k, &v, h, n, c);
                let y = add_vec(&y, &extra);

                // g = sigmoid(xg @ g1) @ g2
                let g_mid = matvec(&xg, &layer.g1, c, gm); // [gm]
                let g_sig = sigmoid(&g_mid);
                let g = matvec(&g_sig, &layer.g2, gm, c); // [C]

                let y_g = mul_vec(&y, &g);

                // output: x0 + (y * g) @ oW (权重已预转置)
                let y_out = matvec(&y_g, &layer.output_w, c, c);
                x = add_vec(&x, &y_out);

                // ===== Channel Mixing =====
                let ln2 = layer_norm(&x, &layer.ln2_w, &layer.ln2_b, LN_EPS);
                let prev_c = state[i].cmix_x.clone();
                state[i].cmix_x = ln2.clone();
                let xb = lerp(&ln2, &prev_c, &layer.ffn_x_k);

                // FFN: 权重已预转置，直接 matvec
                let r2 = relu_sq(&matvec(&xb, &layer.ffn_key_w, c, fh)); // [fh]
                let v2 = matvec(&r2, &layer.ffn_value_w, fh, c); // [C]
                x = add_vec(&x, &v2);

                // 可选校准采样：记录 6 类量化矩阵的输入激活样本
                if self.calib.borrow().as_ref().is_some_and(|cb| !cb.full()) {
                    let mut cb = self.calib.borrow_mut();
                    if let Some(c) = cb.as_mut() {
                        c.accum(i, &xr, &xk, &xv, &y_g, &xb, &r2);
                    }
                }
            }

            // ln_out + head (head 已预转置为 [C, vocab])
            let x_norm = layer_norm(&x, &self.ln_out_w, &self.ln_out_b, LN_EPS);
            logits = matvec(&x_norm, &self.head_weight, c, self.config.vocab);

            // 本 token 的全部层采样完成，推进 token 计数（full() 据此判断采集是否满）
            if self.calib.borrow().as_ref().is_some() {
                self.calib.borrow_mut().as_mut().unwrap().count_token();
            }
        }

        logits
    }
}

fn max_layer(st: &safetensors::SafeTensors) -> usize {
    st.names()
        .iter()
        .filter_map(|k| {
            if k.starts_with("blocks.") {
                k.split('.').nth(1).and_then(|s| s.parse::<usize>().ok())
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
}

impl Layer {
    fn load(
        st: &safetensors::SafeTensors,
        idx: usize,
        c: usize,
        cfg: &ModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let p = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let key = format!("blocks.{idx}.{name}");
            Ok(tensor_to_f32(&st.tensor(&key)?))
        };
        // 预转置 PyTorch Linear 权重 [out, in] → [in, out]；原键缺失时回退 int8 反量化
        let p2t = |name: &str,
                   out_dim: usize,
                   in_dim: usize|
         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let key = format!("blocks.{idx}.att.{name}.weight");
            let raw = linear_to_f32(st, &key, out_dim, in_dim)?;
            Ok(transpose(&raw, out_dim, in_dim))
        };
        let pft = |name: &str,
                   out_dim: usize,
                   in_dim: usize|
         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let key = format!("blocks.{idx}.ffn.{name}.weight");
            let raw = linear_to_f32(st, &key, out_dim, in_dim)?;
            Ok(transpose(&raw, out_dim, in_dim))
        };
        // 低秩权重：CPU matvec 需要 [in, out] 行主序。
        // 各模型原始布局不同（g1d=[in,out]、g1h=[out,in]），按实际形状自适应转置。
        let p_lr = |name: &str,
                    in_dim: usize,
                    out_dim: usize|
         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let key = format!("blocks.{idx}.att.{name}");
            let t = st.tensor(&key)?;
            let shape = t.shape();
            let data = tensor_to_f32(&t);
            Ok(if shape[0] == in_dim && shape[1] == out_dim {
                data // 已是 [in, out]
            } else if shape[0] == out_dim && shape[1] == in_dim {
                transpose(&data, out_dim, in_dim) // [out, in] → [in, out]
            } else {
                panic!(
                    "{key}: unexpected shape {shape:?}, want [{in_dim},{out_dim}] or [{out_dim},{in_dim}]"
                )
            })
        };

        let (wm, am, vm, gm, fh) = (cfg.w_mid, cfg.a_mid, cfg.v_mid, cfg.g_mid, cfg.ffn_hidden);

        Ok(Self {
            ln1_w: p("ln1.weight")?,
            ln1_b: p("ln1.bias")?,
            x_r: p("att.x_r")?,
            x_w: p("att.x_w")?,
            x_k: p("att.x_k")?,
            x_v: p("att.x_v")?,
            x_a: p("att.x_a")?,
            x_g: p("att.x_g")?,
            w0: p("att.w0")?,
            a0: p("att.a0")?,
            v0: p("att.v0")?,
            w1: p_lr("w1", c, wm)?, // [in=C, out=mid_w]
            w2: p_lr("w2", wm, c)?,
            a1: p_lr("a1", c, am)?,
            a2: p_lr("a2", am, c)?,
            v1: p_lr("v1", c, vm)?,
            v2: p_lr("v2", vm, c)?,
            g1: p_lr("g1", c, gm)?,
            g2: p_lr("g2", gm, c)?,
            r_k: p("att.r_k")?,
            k_k: p("att.k_k")?,
            k_a: p("att.k_a")?,
            receptance_w: p2t("receptance", c, c)?, // [C, C] 转置
            key_w: p2t("key", c, c)?,
            value_w: p2t("value", c, c)?,
            output_w: p2t("output", c, c)?,
            ln_x_w: p("att.ln_x.weight")?,
            ln_x_b: p("att.ln_x.bias")?,
            ln2_w: p("ln2.weight")?,
            ln2_b: p("ln2.bias")?,
            ffn_x_k: p("ffn.x_k")?,
            ffn_key_w: pft("key", fh, c)?,     // [C, ffn_hidden] 转置
            ffn_value_w: pft("value", c, fh)?, // [ffn_hidden, C] 转置
        })
    }
}

// ===== 数学工具函数 =====

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n + eps;
    let inv_std = var.sqrt().recip();
    x.iter()
        .zip(w)
        .zip(b)
        .map(|((&xi, &wi), &bi)| (xi - mean) * inv_std * wi + bi)
        .collect()
}

fn layer_norm_rows(
    data: &[f32],
    w: &[f32],
    b: &[f32],
    c: usize,
    rows: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        let row = &data[r * c..(r + 1) * c];
        let normed = layer_norm(row, w, b, eps);
        out[r * c..(r + 1) * c].copy_from_slice(&normed);
    }
    out
}

fn lerp(x: &[f32], y: &[f32], w: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(y)
        .zip(w)
        .map(|((&xi, &yi), &wi)| xi + wi * (yi - xi))
        .collect()
}

fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

fn tanh_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.tanh()).collect()
}

fn exp_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.exp()).collect()
}

fn scale_vec(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|v| v * s).collect()
}

fn neg_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| -v).collect()
}

fn add_bias(x: &[f32], b: &[f32]) -> Vec<f32> {
    x.iter().zip(b).map(|(&xi, &bi)| xi + bi).collect()
}

fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&ai, &bi)| ai + bi).collect()
}

fn mul_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&ai, &bi)| ai * bi).collect()
}

fn relu_sq(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0).powi(2)).collect()
}

/// 矩阵向量乘（直接）: y = x @ W, W shape [in, out] (行主序), x shape [in]
/// y[j] = sum_i x[i] * W[i*out + j]
fn matvec(x: &[f32], w: &[f32], inp: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0; out];
    for i in 0..inp {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * out..(i + 1) * out];
        for j in 0..out {
            y[j] += xi * row[j];
        }
    }
    y
}

/// L2 归一化（分组）: x * max(||x||, eps)^-1, 按组归一化
fn l2_norm_groups(x: &[f32], h: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; h * n];
    for hi in 0..h {
        let group = &x[hi * n..(hi + 1) * n];
        let norm = group.iter().map(|v| v * v).sum::<f32>().sqrt().max(L2_EPS);
        let inv = norm.recip();
        for j in 0..n {
            out[hi * n + j] = group[j] * inv;
        }
    }
    out
}

/// DPLR 状态更新: S = S * w + (S @ a) * b^T + v * k^T
/// S: [H, N, N], w/k/v/a/b: [H, N]
#[allow(clippy::too_many_arguments)]
fn dplr_update(
    s: &mut [f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    a: &[f32],
    b: &[f32],
    h: usize,
    n: usize,
) {
    for hi in 0..h {
        let s_slice = &mut s[hi * n * n..(hi + 1) * n * n];
        let w_h = &w[hi * n..(hi + 1) * n];
        let k_h = &k[hi * n..(hi + 1) * n];
        let v_h = &v[hi * n..(hi + 1) * n];
        let a_h = &a[hi * n..(hi + 1) * n];
        let b_h = &b[hi * n..(hi + 1) * n];

        // Sa = S @ a → [N]
        let mut sa = vec![0.0; n];
        for i in 0..n {
            let row = &s_slice[i * n..(i + 1) * n];
            for j in 0..n {
                sa[i] += row[j] * a_h[j];
            }
        }

        // S = S * w + sa * b^T + v * k^T
        for i in 0..n {
            for j in 0..n {
                s_slice[i * n + j] = s_slice[i * n + j] * w_h[j] + sa[i] * b_h[j] + v_h[i] * k_h[j];
            }
        }
    }
}

/// y = S @ r, S: [H, N, N], r: [H*N] → y: [H*N]
fn dplr_matvec(s: &[f32], r: &[f32], h: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; h * n];
    for hi in 0..h {
        let s_slice = &s[hi * n * n..(hi + 1) * n * n];
        let r_h = &r[hi * n..(hi + 1) * n];
        for i in 0..n {
            let row = &s_slice[i * n..(i + 1) * n];
            for j in 0..n {
                y[hi * n + i] += row[j] * r_h[j];
            }
        }
    }
    y
}

/// Group norm: 每组独立归一化
fn group_norm(x: &[f32], w: &[f32], b: &[f32], h: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; h * n];
    for hi in 0..h {
        let group = &x[hi * n..(hi + 1) * n];
        let mean = group.iter().sum::<f32>() / n as f32;
        let var = group.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32 + eps;
        let inv_std = var.sqrt().recip();
        for j in 0..n {
            y[hi * n + j] = (group[j] - mean) * inv_std * w[hi * n + j] + b[hi * n + j];
        }
    }
    y
}

/// sum(r * k * r_k, axis=1, keepdims=True) * v
/// r, k, v: [H*N], r_k: [H, N]
fn sum_rk_rk(
    r: &[f32],
    k: &[f32],
    r_k: &[f32],
    v: &[f32],
    h: usize,
    n: usize,
    c: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; c];
    for hi in 0..h {
        let r_h = &r[hi * n..(hi + 1) * n];
        let k_h = &k[hi * n..(hi + 1) * n];
        let rk_h = &r_k[hi * n..(hi + 1) * n];
        let v_h = &v[hi * n..(hi + 1) * n];

        let mut s = 0.0;
        for j in 0..n {
            s += r_h[j] * k_h[j] * rk_h[j];
        }
        for j in 0..n {
            out[hi * n + j] = s * v_h[j];
        }
    }
    out
}

/// int8 量化反量化的单元测试（`cargo test` 运行，自包含，不依赖 GPU / 真实模型）。
///
/// 覆盖：
/// - **格式契约**：Python 量化器（`tools/quantize_any4.py --bits 8`）打包的 idx/sz 与
///   本文件 `dequant_int8` 解包一致（每 u32 打包 4 个 uint8、scale/zero 位序）。
/// - **量化精度**：合成高斯权重 → 量化 → 反量化，验证 cos 近无损、rel ≤ 2%。
#[cfg(test)]
mod quant_tests {
    use super::dequant_int8;
    use half::f16;

    #[test]
    fn int8_byte_unpack_and_scale_zero() {
        // int8：每 uint32 打包 4 个 uint8（低位字节 b0=byte0 … b3=byte3）。
        // idx[0] = 0x0000_5123 → b0=0x23=35, b1=0x51=81, b2=0x00, b3=0x00
        let m = 1usize;
        let k = 128usize;
        let mut idx = vec![0u32; m * k / 4];
        idx[0] = 0x0000_5123;
        let scale = 0.5f32;
        let zero = 1.0f32;
        let s = f16::from_f32(scale).to_bits() as u32;
        let z = f16::from_f32(zero).to_bits() as u32;
        let sz = vec![(z << 16) | (s & 0xFFFF); m * k / 128];
        let w = dequant_int8(&idx, &sz, m, k);
        // w = scale*q + zero
        assert_eq!(w[0], 0.5 * 35.0 + 1.0, "b0=35");
        assert_eq!(w[1], 0.5 * 81.0 + 1.0, "b1=81");
        assert_eq!(w[2], 0.5 * 0.0 + 1.0, "b2=0");
        assert_eq!(w[3], 0.5 * 0.0 + 1.0, "b3=0");
        for (i, v) in w.iter().enumerate().skip(4) {
            assert_eq!(*v, 1.0, "k{i} 应解出 zero（q=0）");
        }
    }

    #[test]
    fn int8_constant_group_exact_rebuild() {
        // 常数组（scale=0）：w = zero 精确重建
        let m = 1usize;
        let k = 128usize;
        let idx = vec![0u32; m * k / 4];
        let zero = -2.5f32;
        let z = f16::from_f32(zero).to_bits() as u32;
        let sz = vec![z << 16; m * k / 128]; // scale=0, zero=-2.5
        let w = dequant_int8(&idx, &sz, m, k);
        for v in w {
            assert_eq!(v, -2.5, "常数组应精确重建为 zero");
        }
    }

    #[test]
    fn int8_precision_meets_target() {
        // int8 非对称 per-group=128，相对误差应显著低于 4-bit（近无损）。
        // 只要求 cos 极高、rel 很低，验证解包/反量化正确。
        let m = 32usize;
        let k = 2560usize;
        let group = 128usize;
        let w = gauss(m, k);
        // 合成 int8 量化（与 Python quantize_int8_matrix 等价的 min-max 量化）
        let kg = k / group;
        let mut idxp = vec![0u32; m * k / 4];
        let mut sz = vec![0u32; m * kg];
        for r in 0..m {
            for g in 0..kg {
                let sl = r * k + g * group;
                let mut mn = f32::MAX;
                let mut mx = f32::MIN;
                for j in 0..group {
                    let v = w[sl + j];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let sc = (mx - mn) / 255.0;
                let zero = mn;
                let inv = if sc <= 0.0 { 1.0 } else { 1.0 / sc };
                let s16 = f16::from_f32(sc).to_bits() as u32;
                let z16 = f16::from_f32(zero).to_bits() as u32;
                sz[r * kg + g] = (z16 << 16) | (s16 & 0xFFFF);
                for j in 0..group {
                    let q = ((w[sl + j] - zero) * inv).round().clamp(0.0, 255.0) as u32;
                    let pack = r * (k / 4) + (g * group + j) / 4;
                    idxp[pack] |= q << (((g * group + j) % 4) * 8);
                }
            }
        }
        let w_hat = dequant_int8(&idxp, &sz, m, k);
        let cos = cos_sim(&w_hat, &w);
        let rel = rel_err(&w_hat, &w);
        assert!(cos >= 0.999, "int8 cos={cos:.6} 应近无损");
        assert!(rel <= 0.02, "int8 rel={rel:.4} 应 ≤2%");
    }

    // ---------- 量化精度闭环：合成高斯权重 → 量化 → 反量化 ----------

    fn gauss(m: usize, k: usize) -> Vec<f32> {
        // 12 个均匀求和近似高斯（中心极限定理）
        let mut r = fastrand::Rng::with_seed(42);
        (0..m * k)
            .map(|_| {
                let s: f32 = (0..12).map(|_| r.f32() - 0.5).sum();
                s
            })
            .collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0f64;
        let mut n = 0f64;
        for (x, y) in a.iter().zip(b) {
            let d = (*x as f64) - (*y as f64);
            s += d * d;
            n += (*y as f64) * (*y as f64);
        }
        (s / n.max(1e-12)).sqrt() as f32
    }

    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na * nb).sqrt().max(1e-12)) as f32
    }
}
