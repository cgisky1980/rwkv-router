use std::error::Error;

use rwkv_rsv::{backend, gpu_model, model, runtime};

fn topk_indices(logits: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().map(|&(i, _)| i).collect()
}

/// 读取 prepare_calib_prompts.py 生成的多语言 prompt 文件。
/// 格式：u32 magic=C411B0C0, u32 n_prompts, n_prompts × { u32 len; len × u32 token }。
fn load_prompts(path: &str) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err("calib prompts 文件过短".into());
    }
    let mut pos = 0usize;
    let rd = |bytes: &[u8], pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };
    let magic = rd(&bytes, &mut pos);
    assert_eq!(magic, 0xC411B0C0, "calib prompts magic 不符");
    let n = rd(&bytes, &mut pos) as usize;
    let mut prompts = Vec::with_capacity(n);
    for _ in 0..n {
        let len = rd(&bytes, &mut pos) as usize;
        let mut toks = Vec::with_capacity(len);
        for _ in 0..len {
            toks.push(rd(&bytes, &mut pos));
        }
        prompts.push(toks);
    }
    Ok(prompts)
}

/// 查询 GPU 专用(显存)与共享(系统内存)占用，返回 MB。
/// 通过 Windows 性能计数器（nvidia-smi 不可用），每次调用约 1s。
fn query_gpu_mem_mb() -> (f64, f64) {
    let ps = r#"
$d = 0.0; $s = 0.0
try {
    $d = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples |
         Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
} catch {}
try {
    $s = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples |
         Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum
} catch {}
Write-Output ("{0} {1}" -f $d, $s)
"#;
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", ps])
        .output();
    let (ded, shared) = match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            let parts: Vec<f64> = s
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            if parts.len() >= 2 {
                (parts[0], parts[1])
            } else {
                (0.0, 0.0)
            }
        }
        Err(_) => (0.0, 0.0),
    };
    (ded / (1024.0 * 1024.0), shared / (1024.0 * 1024.0))
}

/// 连续生成显存测试：`cargo run --release -- memtest`
/// 每步用上一 token 的 logits argmax 采样下一个 token（模拟真实自回归生成），
/// 周期性采样显存，检查是否随推理轮次累积上升；同时统计速度。
fn run_memtest(gpu_model: &mut gpu_model::GpuModel) -> Result<(), Box<dyn Error>> {
    use std::time::Instant;

    let n = std::env::var("GEN_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);
    let report_every = std::env::var("REPORT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200usize);

    // 初始 prompt（" Eiffel"），预热后从最后一个 token 开始自回归
    let prompt: Vec<u32> = vec![304, 25740, 109];
    gpu_model.reset_state()?;
    let _ = gpu_model.forward(&prompt)?;
    let mut last = *prompt.last().unwrap();

    // 预热一轮（kernel 缓存、状态初始化后）
    let _ = gpu_model.forward(&[last])?;

    let t_start = Instant::now();
    log::info!("== memtest: 连续生成 {n} tokens, 每 {report_every} token 采样一次显存 ==");
    let (d0, s0) = query_gpu_mem_mb();
    log::info!(
        "token 0/{}  基线: dedicated={d0:.0} MB shared={s0:.0} MB",
        n
    );

    // ===== SELFLOOP_ONLY=1：跳过逐 token 段，只测 self-loop（隔离 GPU 加热干扰）=====
    if std::env::var("SELFLOOP_ONLY").is_ok() {
        let sl_n = std::env::var("SELFLOOP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(n);
        let seed = last;
        // 预热 self-loop（kernel 缓存 build）
        gpu_model.reset_state()?;
        let _ = gpu_model.forward_argmax_selfloop(seed, 4)?;
        let t_sl = Instant::now();
        gpu_model.reset_state()?;
        let _ = gpu_model.forward_argmax_selfloop(seed, sl_n)?;
        let total_sl = t_sl.elapsed().as_secs_f64();
        log::info!(
            "== SELFLOOP_ONLY: self-loop {sl_n} tokens, 总耗时 {total_sl:.1}s, 平均 {:.1} tok/s ==",
            sl_n as f64 / total_sl
        );
        return Ok(());
    }

    // ===== AB_BENCH=1：受控 A/B 实测 =====
    // 先于下方长 memtest 循环执行并直接返回，避免长循环把 GPU 加热后再测 self-loop。
    // 交替测 逐token 与 self-loop，多轮取中位数，消除热降频对测量顺序的偏向。
    if std::env::var("AB_BENCH").is_ok() {
        return bench_ab(gpu_model);
    }

    for i in 0..n {
        // GPU 采样：forward_argmax 在 GPU 端 argmax，只回传 4 字节 token 索引（不下载 logits）
        last = gpu_model.forward_argmax(&[last])?;

        if (i + 1) % report_every == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let avg_per_tok = elapsed * 1000.0 / (i + 1) as f64;
            let (d, s) = query_gpu_mem_mb();
            log::info!(
                "token {}/{}  累计 {:.1}s  平均 {:.2} ms/token  dedicated={d:.0} MB  shared={s:.0} MB",
                i + 1,
                n,
                elapsed,
                avg_per_tok,
            );
        }
    }

    let total = t_start.elapsed().as_secs_f64();
    let (df, sf) = query_gpu_mem_mb();
    log::info!(
        "== memtest done: {n} tokens, 总耗时 {total:.1}s, 平均 {:.1} tok/s ==",
        n as f64 / total
    );
    log::info!(
        "== 最终显存: dedicated={df:.0} MB (基线 {d0:.0})  shared={sf:.0} MB (基线 {s0:.0}) =="
    );

    // ===== GPU self-loop 批量生成基准对比（SELFLOOP_BENCH=1 时）=====
    // 与上面的逐 token (forward_argmax, 每 token 一次 submit+wait) 对比 tok/s。
    if std::env::var("SELFLOOP_BENCH").is_ok() {
        let sl_n = std::env::var("SELFLOOP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(n);
        let seed = last; // 用上面逐 token 生成的最后一个 token 作为 self-loop 的 seed

        // 预热 self-loop（kernel 缓存 build + 状态已在上面建立）
        gpu_model.reset_state()?;
        let _ = gpu_model.forward_argmax_selfloop(seed, 4)?;

        let t_sl = Instant::now();
        gpu_model.reset_state()?;
        let _ = gpu_model.forward_argmax_selfloop(seed, sl_n)?;
        let total_sl = t_sl.elapsed().as_secs_f64();
        log::info!(
            "== SELFLOOP_BENCH: self-loop {sl_n} tokens, 总耗时 {total_sl:.1}s, 平均 {:.1} tok/s ==",
            sl_n as f64 / total_sl
        );
        log::info!(
            "== SPEEDUP: 逐token {:.1} tok/s  vs  self-loop {:.1} tok/s  =>  {:.2}x ==",
            n as f64 / total,
            sl_n as f64 / total_sl,
            (sl_n as f64 / total_sl) / (n as f64 / total),
        );
    }
    Ok(())
}

/// 受控 A/B 实测：同热态下交替测 逐token 与 GPU self-loop，多轮取中位数。
/// 前提：GPU 已降温。设计要点：
///   - AB_ROUNDS 轮交替，每轮两条路径各测 AB_N tokens；
///   - 每轮先给两条路径各跑 4 token 预热，使两者处于同一热态/缓存态；
///   - 轮内测量顺序按轮号奇偶交替，抵消单调升温对先后者的偏向；
///   - 取中位数而非均值，剔除热降频极端抖动。
///
/// 逐token 用 forward_argmax（每 token 一次 submit+wait+下载 argmax）；
/// self-loop 用 forward_argmax_selfloop（单次 submit 连跑，只在末尾一次性下载）。
fn bench_ab(gpu_model: &mut gpu_model::GpuModel) -> Result<(), Box<dyn Error>> {
    use std::time::Instant;

    let n = std::env::var("AB_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);
    let rounds = std::env::var("AB_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5usize);
    // 固定 seed，仅作计时用；不要求两条路径输出一致（计时只看 forward 轮数）
    let seed = 304u32;

    log::info!(
        "== AB_BENCH: 每轮各测 {n} tokens, {rounds} 轮交替 (须 GPU 已降温; AB_N/AB_ROUNDS 可调) =="
    );

    let mut tok_times: Vec<f64> = Vec::new(); // 逐token 每轮耗时
    let mut sl_times: Vec<f64> = Vec::new(); // self-loop 每轮耗时

    for r in 0..rounds {
        // 预热：两条路径各跑 4 token，使两者处于同一热态/缓存态
        gpu_model.reset_state()?;
        let _ = gpu_model.forward_argmax_selfloop(seed, 4)?;
        gpu_model.reset_state()?;
        let mut s = seed;
        for _ in 0..4 {
            s = gpu_model.forward_argmax(&[s])?;
        }

        // 轮内测量顺序按奇偶交替，抵消单调升温
        if r % 2 == 0 {
            // 先 self-loop，后 逐token
            gpu_model.reset_state()?;
            let t0 = Instant::now();
            let _ = gpu_model.forward_argmax_selfloop(seed, n)?;
            sl_times.push(t0.elapsed().as_secs_f64());

            gpu_model.reset_state()?;
            let t0 = Instant::now();
            let mut s = seed;
            for _ in 0..n {
                s = gpu_model.forward_argmax(&[s])?;
            }
            tok_times.push(t0.elapsed().as_secs_f64());
        } else {
            // 先 逐token，后 self-loop
            gpu_model.reset_state()?;
            let t0 = Instant::now();
            let mut s = seed;
            for _ in 0..n {
                s = gpu_model.forward_argmax(&[s])?;
            }
            tok_times.push(t0.elapsed().as_secs_f64());

            gpu_model.reset_state()?;
            let t0 = Instant::now();
            let _ = gpu_model.forward_argmax_selfloop(seed, n)?;
            sl_times.push(t0.elapsed().as_secs_f64());
        }

        log::info!(
            "AB round {r}: 逐token {:.1} tok/s | self-loop {:.1} tok/s",
            n as f64 / tok_times[r],
            n as f64 / sl_times[r]
        );
    }

    // 取中位数，剔除热降频极端抖动
    tok_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tok_med = tok_times[tok_times.len() / 2];
    let sl_med = sl_times[sl_times.len() / 2];
    log::info!(
        "== AB 中位数: 逐token {:.1} tok/s ({:.3} ms/tok)  vs  self-loop {:.1} tok/s ({:.3} ms/tok)  =>  self-loop {:.2}x ==",
        n as f64 / tok_med,
        tok_med * 1000.0 / n as f64,
        n as f64 / sl_med,
        sl_med * 1000.0 / n as f64,
        (n as f64 / sl_med) / (n as f64 / tok_med)
    );
    Ok(())
}

fn init() -> Result<(), Box<dyn Error>> {
    use simplelog::{ColorChoice, CombinedLogger, LevelFilter, TermLogger, WriteLogger};

    std::fs::create_dir_all("logs")?;
    std::fs::create_dir_all("outputs")?;

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/rwkv-rsv_{}.log", timestamp);
    let file = std::fs::File::create(&filename)?;

    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Info,
            Default::default(),
            Default::default(),
            ColorChoice::Auto,
        ),
        WriteLogger::new(LevelFilter::Info, Default::default(), file),
    ])?;

    fastrand::seed(514);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    init()?;
    log::info!("rwkv-rsv: RWKV-7 inference with Vulkan");

    // 加载模型（默认 3B，可用环境变量 MODEL_PATH 切换小模型）
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    log::info!("loading model: {model_path}");
    let model = model::Model::from_safetensors(&model_path)?;
    log::info!("model loaded");

    // probe: " Eiffel" → tokens [304, 25740, 109]
    let tokens: Vec<u32> = vec![304, 25740, 109];
    log::info!("probe tokens: {tokens:?}");

    let mut state = model.init_state();
    let logits = model.forward(&tokens, &mut state);

    // top-10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    log::info!("== top-10 logits ==");
    for (rank, (token, logit)) in indexed.iter().take(10).enumerate() {
        let rank = rank + 1;
        let prob = logit.exp() / logits.iter().map(|v| v.exp()).sum::<f32>();
        log::info!("{rank}: token={token} logit={logit:.6} prob={prob:.8}");
    }

    // ===== 跨模型 logits 对比：SAVE_LOGITS 存 f32 bin，COMPARE_LOGITS 读入对比 =====
    // 用途：any4 量化模型 vs 原 fp16 模型的端到端量化误差（CPU fp32 logits 为共同基准）
    if let Ok(p) = std::env::var("SAVE_LOGITS") {
        std::fs::write(&p, bytemuck::cast_slice(&logits))?;
        log::info!("logits saved → {p} ({} f32)", logits.len());
    }
    if let Ok(p) = std::env::var("COMPARE_LOGITS") {
        let bytes = std::fs::read(&p)?;
        let ref_logits: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(ref_logits.len(), logits.len(), "logits 长度不符");
        let mut max_d = 0.0f32;
        let mut sum_sq = 0.0f32;
        for (a, b) in logits.iter().zip(ref_logits) {
            let d = (a - b).abs();
            if d > max_d {
                max_d = d;
            }
            sum_sq += d * d;
        }
        let rmse = (sum_sq / logits.len() as f32).sqrt();
        let ref_top10 = topk_indices(ref_logits)[..10].to_vec();
        let cur_top10 = topk_indices(&logits)[..10].to_vec();
        let agree = ref_top10.iter().filter(|t| cur_top10.contains(t)).count();
        log::info!("== COMPARE_LOGITS vs {p} ==");
        log::info!("max_abs_diff: {max_d:.6}  rmse: {rmse:.6}");
        log::info!("ref top10: {ref_top10:?}");
        log::info!(
            "cur top10: {cur_top10:?}  (集合重合 {agree}/10, top1 一致: {})",
            ref_top10[0] == cur_top10[0]
        );
    }

    // ===== teacher-forced Top-1 一致率（AUXStar 风格精度指标）=====
    // 用法（两次运行，均在 GPU 加载前返回）：
    //   1) fp16 模型：TOP1_REF_SAVE=outputs\top1_ref.bin  → CPU fp32 greedy 生成参考序列（u32 bin）
    //   2) any4 模型：TOP1_REF_COMPARE=outputs\top1_ref.bin → teacher-forcing 逐位置比 argmax
    // 两阶段均走 CPU fp32（any4 模型自动反量化），隔离纯量化误差；
    // GPU 内核与 CPU 的一致性由 ARGMAX_VERIFY / SELFLOOP_VERIFY 单独保证。
    if let Ok(p) = std::env::var("TOP1_REF_SAVE") {
        let n = std::env::var("TOP1_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize);
        let mut st = model.init_state();
        let mut lg = model.forward(&tokens, &mut st);
        let mut seq = tokens.clone();
        for i in 0..n {
            let cur = topk_indices(&lg)[0] as u32;
            seq.push(cur);
            if i + 1 < n {
                lg = model.forward(&[cur], &mut st);
            }
        }
        let bytes: Vec<u8> = seq.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&p, bytes)?;
        log::info!(
            "== TOP1_REF_SAVE: prompt {} + gen {} tokens → {p} ==",
            tokens.len(),
            n
        );
        log::info!(
            "ref seq[gen 0..16]: {:?}",
            &seq[tokens.len()..(tokens.len() + 16).min(seq.len())]
        );
        return Ok(());
    }
    if let Ok(p) = std::env::var("TOP1_REF_COMPARE") {
        let bytes = std::fs::read(&p)?;
        let ref_seq: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let prompt_len = tokens.len();
        assert!(ref_seq.len() > prompt_len, "参考序列过短");
        assert!(
            ref_seq[..prompt_len] == tokens[..],
            "参考序列 prompt 前缀不一致"
        );
        // teacher forcing：逐个喂参考 token，每个位置比较 argmax 与下一个参考 token
        let mut st = model.init_state();
        let mut lg = model.forward(&tokens, &mut st);
        let mut agree = 0usize;
        let mut total = 0usize;
        let mut first_div: Option<usize> = None;
        for i in prompt_len..ref_seq.len() {
            let pred = topk_indices(&lg)[0] as u32;
            let want = ref_seq[i];
            if pred == want {
                agree += 1;
            } else if first_div.is_none() {
                first_div = Some(i - prompt_len);
                log::info!(
                    "首次分叉@生成第 {} token: pred={pred} want={want}",
                    i - prompt_len
                );
            }
            total += 1;
            if i + 1 < ref_seq.len() {
                lg = model.forward(&[want], &mut st);
            }
        }
        log::info!(
            "== TOP1_REF_COMPARE vs {p}: teacher-forced top1 一致率 {agree}/{total} = {:.1}% ; 首次分叉位置 {first_div:?} ==",
            agree as f64 / total.max(1) as f64 * 100.0
        );
        return Ok(());
    }

    // ===== 多样多语言 prompt 的 teacher-forced Top-1 一致率（真实场景精度）=====
    // 用法（两次运行，均在 GPU 加载前返回）：
    //   1) fp16：TOP1_MULTI_SAVE=outputs\top1_multi.bin → 读 CALIB_PROMPTS，每条 prompt greedy 生成 TOP1_N token
    //   2) any4：TOP1_MULTI_COMPARE=outputs\top1_multi.bin → 逐条 teacher-forcing 比 argmax，聚合一致率
    // 与单 prompt 版不同：在真实多语言多样 prompt 上聚合，模拟实际部署的量化准确度。
    if std::env::var("TOP1_MULTI_SAVE").is_ok() || std::env::var("TOP1_MULTI_COMPARE").is_ok() {
        let prompts_path = std::env::var("CALIB_PROMPTS")
            .unwrap_or_else(|_| r"c:\work\niceui\rwkv-rsv\outputs\calib_prompts.bin".to_string());
        let prompts = load_prompts(&prompts_path)?;
        let gen_len = std::env::var("TOP1_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize);
        if let Ok(p) = std::env::var("TOP1_MULTI_SAVE") {
            let plimit = std::env::var("TOP1_PROMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(prompts.len());
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(&0x31_50_4F_54u32.to_le_bytes()); // 'TOP1'
            buf.extend_from_slice(&(plimit as u32).to_le_bytes());
            for prompt in prompts.iter().take(plimit) {
                let mut st = model.init_state();
                let mut lg = model.forward(prompt, &mut st);
                let mut gen_toks: Vec<u32> = Vec::with_capacity(gen_len);
                for _ in 0..gen_len {
                    let cur = topk_indices(&lg)[0] as u32;
                    gen_toks.push(cur);
                    lg = model.forward(&[cur], &mut st);
                }
                buf.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
                buf.extend_from_slice(&(gen_toks.len() as u32).to_le_bytes());
                for t in prompt {
                    buf.extend_from_slice(&t.to_le_bytes());
                }
                for t in &gen_toks {
                    buf.extend_from_slice(&t.to_le_bytes());
                }
            }
            std::fs::write(&p, buf)?;
            log::info!(
                "== TOP1_MULTI_SAVE: {} 条 prompt × gen {gen_len} → {p} ==",
                plimit
            );
            return Ok(());
        }
        if let Ok(p) = std::env::var("TOP1_MULTI_COMPARE") {
            let bytes = std::fs::read(&p)?;
            let mut pos = 0usize;
            let rd = |bytes: &[u8], pos: &mut usize| -> u32 {
                let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
                *pos += 4;
                v
            };
            let magic = rd(&bytes, &mut pos);
            assert_eq!(magic, 0x31_50_4F_54, "multi-ref magic 不符");
            let n = rd(&bytes, &mut pos) as usize;
            let mut total = 0usize;
            let mut agree = 0usize;
            let mut first_div: Option<(usize, usize)> = None;
            for pi in 0..n {
                let plen = rd(&bytes, &mut pos) as usize;
                let glen = rd(&bytes, &mut pos) as usize;
                let mut seq = Vec::with_capacity(plen + glen);
                for _ in 0..plen + glen {
                    seq.push(rd(&bytes, &mut pos));
                }
                let prompt = &seq[..plen];
                let mut st = model.init_state();
                let mut lg = model.forward(prompt, &mut st);
                for gi in 0..glen {
                    let pred = topk_indices(&lg)[0] as u32;
                    let want = seq[plen + gi];
                    total += 1;
                    if pred == want {
                        agree += 1;
                    } else if first_div.is_none() {
                        first_div = Some((pi, gi));
                        log::info!("首次分叉 prompt#{pi} gen#{gi}: pred={pred} want={want}");
                    }
                    if gi + 1 < glen {
                        lg = model.forward(&[want], &mut st);
                    }
                }
            }
            log::info!(
                "== TOP1_MULTI_COMPARE vs {p}: teacher-forced top1 一致率 {agree}/{total} = {:.1}% ; 首次分叉 {first_div:?} ==",
                agree as f64 / total.max(1) as f64 * 100.0
            );
            return Ok(());
        }
    }

    // ===== 校准激活采集：CALIB_SAMPLES=path → 采集 S 个 token 的 6 类矩阵输入激活样本 =====
    // 用法：CALIB_SAMPLES=outputs\calib_samples.st CALIB_N=512 cargo run --release
    // 采集源：CALIB_PROMPTS（calib_prompts.bin，prepare_calib_prompts.py 产出）里的真实多语言多样 prompt，
    // 每条 prompt 独立 init_state，截断前缀 + greedy(top-1) 续写到 per-prompt 预算，再换下一条，
    // 直到 CALIB_N。用 greedy 而非采样：与验证（TOP1_MULTI_COMPARE）解码一致，保证校准/验证激活分布匹配。
    // 产出键=blocks.{li}.{name}，形状 [S, len]，供 quantize_any4.py --nnq-calib 用。
    if let Ok(p) = std::env::var("CALIB_SAMPLES") {
        let cap = std::env::var("CALIB_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512usize);
        // 每条 prompt 最多贡献的 token 数（prompt 前缀 + greedy 续写），
        // 避免单条长 prompt 占满 cap，保证跨多条多语言 prompt 的多样性。
        let per_prompt = std::env::var("CALIB_PER_PROMPT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize);
        let prompts_path = std::env::var("CALIB_PROMPTS")
            .unwrap_or_else(|_| r"c:\work\niceui\rwkv-rsv\outputs\calib_prompts.bin".to_string());
        let prompts = load_prompts(&prompts_path)?;
        model.enable_calib(cap);
        let mut collected = 0usize;
        let mut used_prompts = 0usize;
        for prompt in &prompts {
            if collected >= cap {
                break;
            }
            let mut budget = per_prompt.min(cap - collected);
            // 截断 prompt 前缀到剩余预算，避免长 prompt 占满 cap
            let prefix = &prompt[..budget.min(prompt.len())];
            let mut st = model.init_state();
            let mut lg = model.forward(prefix, &mut st);
            collected += prefix.len();
            budget -= prefix.len();
            used_prompts += 1;
            // greedy 续写，把本 prompt 的预算用完（与验证解码一致，保证校准/验证分布匹配）
            while budget > 0 && collected < cap {
                let cur = topk_indices(&lg)[0] as u32;
                lg = model.forward(&[cur], &mut st);
                collected += 1;
                budget -= 1;
            }
        }
        std::fs::write(&p, model.dump_calib()?)?;
        log::info!(
            "== CALIB_SAMPLES: 采集 {collected} token/层（greedy，每条 ≤{per_prompt}，{used_prompts} 条多语言 prompt）→ {p} =="
        );
        return Ok(());
    }

    // ===== mock 推理：自由自回归生成对比（GEN_SIM）=====
    // 验证 any4 量化是否"接近无损"：fp16 与 any4 各自从同一真实多语言 prompt **自由自回归**生成
    // （非 teacher-forced），对比生成轨迹一致性。用未参与 nnq512 校准的 calib_prompts.bin
    // 多语言集，避免校准泄漏。详细记录见 参考/any4论文要点.md §mock。
    //   1) fp16：GEN_SIM_SAVE=outputs\gen_sim_ref.bin → 每条 prompt 自由自回归生成 GEN_N token，存参考序列
    //   2) any4：GEN_SIM_COMPARE=outputs\gen_sim_ref.bin → 自由自回归生成，对比：
    //        - teacher-forced 单步一致率（喂参考 token，测单步条件概率 argmax 质量）
    //        - 自由自回归平均首次分叉位置（量化误差在自回归轨迹中放大到改变 argmax 的用时）
    //        - 自由自回归全程逐位重合率（含分叉后重收敛）
    //        - 完整序列一致率（严格体验）
    // 规模控制：GEN_PROMPTS（用前 N 条，默认 64）、GEN_N（生成 token 数，默认 16）。
    if std::env::var("GEN_SIM_SAVE").is_ok() || std::env::var("GEN_SIM_COMPARE").is_ok() {
        let prompts_path = std::env::var("CALIB_PROMPTS")
            .unwrap_or_else(|_| r"c:\work\niceui\rwkv-rsv\outputs\calib_prompts.bin".to_string());
        let prompts = load_prompts(&prompts_path)?;
        let gen_n = std::env::var("GEN_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16usize);
        let plimit = std::env::var("GEN_PROMPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64usize)
            .min(prompts.len());
        if let Ok(p) = std::env::var("GEN_SIM_SAVE") {
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(&0x4D49_5347u32.to_le_bytes()); // 'GSIM'
            buf.extend_from_slice(&(plimit as u32).to_le_bytes());
            for (pi, prompt) in prompts.iter().take(plimit).enumerate() {
                log::info!(
                    "GEN_SIM_SAVE prompt {}/{} len {}",
                    pi + 1,
                    plimit,
                    prompt.len()
                );
                let mut st = model.init_state();
                let mut lg = model.forward(prompt, &mut st);
                let mut gen_toks: Vec<u32> = Vec::with_capacity(gen_n);
                for _ in 0..gen_n {
                    let cur = topk_indices(&lg)[0] as u32;
                    gen_toks.push(cur);
                    lg = model.forward(&[cur], &mut st);
                }
                buf.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
                buf.extend_from_slice(&(gen_toks.len() as u32).to_le_bytes());
                for t in prompt {
                    buf.extend_from_slice(&t.to_le_bytes());
                }
                for t in &gen_toks {
                    buf.extend_from_slice(&t.to_le_bytes());
                }
            }
            std::fs::write(&p, buf)?;
            log::info!("== GEN_SIM_SAVE: {plimit} 条 prompt × gen {gen_n} → {p} ==");
            return Ok(());
        }
        if let Ok(p) = std::env::var("GEN_SIM_COMPARE") {
            let bytes = std::fs::read(&p)?;
            let mut pos = 0usize;
            let rd = |bytes: &[u8], pos: &mut usize| -> u32 {
                let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
                *pos += 4;
                v
            };
            let magic = rd(&bytes, &mut pos);
            assert_eq!(magic, 0x4D49_5347, "gen_sim magic 不符");
            let n = rd(&bytes, &mut pos) as usize;
            // teacher-forced 单步一致率
            let mut tf_total = 0usize;
            let mut tf_agree = 0usize;
            // 自由自回归
            let mut ar_total = 0usize;
            let mut ar_match = 0usize; // 分叉前重合数（一致长度）
            let mut div_positions: Vec<usize> = Vec::new(); // 实际分叉位置
            let mut seq_match = 0usize; // 完整序列一致数
            let mut sum_glen = 0usize; // 所有条目生成长度之和（用于平均一致长度分母/贡献）
            let mut sum_first_pos = 0usize; // 每条的"首分叉前一致长度"：分叉条=gi，完整条=glen
            for _ in 0..n {
                let plen = rd(&bytes, &mut pos) as usize;
                let glen = rd(&bytes, &mut pos) as usize;
                sum_glen += glen;
                let mut seq = Vec::with_capacity(plen + glen);
                for _ in 0..plen + glen {
                    seq.push(rd(&bytes, &mut pos));
                }
                let prompt = &seq[..plen];
                // teacher-forced：喂参考 token，测单步条件概率 argmax
                let mut st = model.init_state();
                let mut lg = model.forward(prompt, &mut st);
                for gi in 0..glen {
                    let pred = topk_indices(&lg)[0] as u32;
                    let want = seq[plen + gi];
                    tf_total += 1;
                    if pred == want {
                        tf_agree += 1;
                    }
                    lg = model.forward(&[want], &mut st);
                }
                // 自由自回归：喂自己生成的 token，测轨迹稳定性
                let mut st = model.init_state();
                let mut lg = model.forward(prompt, &mut st);
                let mut diverged = false;
                for gi in 0..glen {
                    let pred = topk_indices(&lg)[0] as u32;
                    let want = seq[plen + gi];
                    ar_total += 1;
                    if pred == want {
                        if !diverged {
                            ar_match += 1;
                        }
                    } else if !diverged {
                        diverged = true;
                        div_positions.push(gi);
                        sum_first_pos += gi;
                        log::info!("自回归首分叉 prompt  gi#{gi}: pred={pred} want={want}");
                    }
                    lg = model.forward(&[pred], &mut st);
                }
                if !diverged {
                    seq_match += 1;
                    sum_first_pos += glen;
                }
            }
            let nf = n.max(1) as f64;
            let avg_first = sum_first_pos as f64 / nf;
            let (min_fd, max_fd) = if div_positions.is_empty() {
                (sum_glen, sum_glen)
            } else {
                (
                    *div_positions.iter().min().unwrap(),
                    *div_positions.iter().max().unwrap(),
                )
            };
            log::info!("== GEN_SIM mock 推理（{n} 条未校准多语言 prompt × gen {gen_n}）==");
            log::info!(
                "  teacher-forced 单步一致率: {tf_agree}/{tf_total} = {:.1}%",
                tf_agree as f64 / tf_total.max(1) as f64 * 100.0
            );
            let avg_glen = sum_glen as f64 / nf;
            log::info!(
                "  自由自回归 平均一致长度(首分叉前): {avg_first:.2} / {avg_glen:.0} token  首分叉区间 [{min_fd}, {max_fd}]",
            );
            log::info!(
                "  自由自回归 完整序列一致: {seq_match}/{n} = {:.1}%",
                seq_match as f64 / nf * 100.0
            );
            log::info!(
                "  自由自回归 全程逐位重合(含分叉后): {ar_match}/{ar_total} = {:.1}%",
                ar_match as f64 / ar_total.max(1) as f64 * 100.0
            );
            return Ok(());
        }
    }

    // ===== GPU 对比验证 =====
    let backend_kind = backend::detect_backend();
    log::info!(
        "backend auto-selected: {backend_kind:?} (cuda hw available: {})",
        backend::cuda_available()
    );
    let backend = backend::create_backend(backend_kind)?;
    let mut gpu_model = gpu_model::GpuModel::from_safetensors(backend, &model_path)?;
    log::info!("gpu model loaded");

    // ===== state_tune 演示：cargo run --release -- statetune =====
    // 验证 web-rwkv 风格 State 的「前进 → back 取态 → load 回灌」闭环：
    //   1) 用外部 State 前向推理，得到训练目标 token 的 logits；
    //   2) state_back 整态下载到 CPU（布局与 state_load 一致）；
    //   3) 模拟 state tuning：对取出的态做轻量扰动（tuning 方向性调整）；
    //   4) state_load 回灌 GPU，再次前向，验证 logits 随 tuned 态变化（target token 概率应有差异）。
    // 该闭环即 ai00-server 会话持久化 / state tuning 文件存取所依赖的序列化原语。
    if std::env::args().any(|a| a == "statetune") {
        let tunes = std::env::var("TUNE_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3usize);
        log::info!("== statetune: 前进→back→load 闭环演示 (tuning {tunes} 次) ==");

        // 一个真实 prompt 作为训练目标序列
        let train_tokens: Vec<u32> = vec![304, 25740, 109];
        let mut state = gpu_model.create_state()?;
        let logits_base = gpu_model.forward_with_state(&mut state, &train_tokens)?;
        let base_top = topk_indices(&logits_base)[..5].to_vec();
        log::info!("base (未 tuning) top5: {base_top:?}");

        for i in 1..=tunes {
            // 1) 取态
            let st = gpu_model.state_back(&state)?;
            log::info!("  round {i}: state_back {} f32", st.len());

            // 2) tuning：对态做确定性扰动（放大一个小量的方向，模拟 state tuning 调整）
            let mut tuned = st.clone();
            let amp = 0.01f32 * i as f32;
            for v in tuned.iter_mut().step_by(37) {
                *v *= 1.0 + amp;
            }

            // 3) 回灌
            gpu_model.state_load(&state, &tuned)?;

            // 4) 用 tuned 态继续前向（在 train_tokens 后追加同 token，观察状态变化）
            let logits_t = gpu_model.forward_with_state(&mut state, &[train_tokens[0]])?;
            let t_top = topk_indices(&logits_t)[..5].to_vec();
            log::info!(
                "  round {i}: tuned 态前向 top5: {t_top:?}  (与 base 之差 {:.6})",
                logits_t
                    .iter()
                    .zip(&logits_base)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v))
            );
        }
        log::info!("== statetune 演示完成 ==");
        return Ok(());
    }

    // ===== 显存累积测试：cargo run --release -- memtest =====
    // 连续自回归生成 GEN_TOKENS（默认 1000）个 token，周期性采样显存，检查是否累积上升并统计速度。
    if std::env::args().any(|a| a == "memtest") {
        return run_memtest(&mut gpu_model);
    }

    // ===== 诊断：隔离单 token 的 seq vs tok 对比（DIAG=1 时早早返回）=====
    if std::env::var("DIAG").is_ok() {
        // 确定性测试：连跑两次 forward_seq（不打断批处理），确认 seq 路径本身确定
        let single = vec![304u32];
        gpu_model.reset_state()?;
        let s1 = gpu_model.forward_seq(&single)?;
        gpu_model.reset_state()?;
        let s2 = gpu_model.forward_seq(&single)?;
        let d0 = s1
            .iter()
            .zip(&s2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== DIAG seq run1 vs run2 max_abs_diff: {d0:.6} ==");
        gpu_model.reset_state()?;
        let t1 = gpu_model.forward(&single)?;
        let d1 = s1
            .iter()
            .zip(&t1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== DIAG single-token seq vs tok max_abs_diff: {d1:.6} ==");
        log::info!("DIAG seq top5: {:?}", topk_indices(&s1)[..5].to_vec());
        log::info!("DIAG tok top5: {:?}", topk_indices(&t1)[..5].to_vec());

        // 正确性 ground truth：seq 单 token vs CPU fp32 单 token
        let mut cpu_state = model.init_state();
        let cpu1 = model.forward(&single, &mut cpu_state);
        let d_cpu = s1
            .iter()
            .zip(&cpu1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== DIAG single-token seq vs CPU max_abs_diff: {d_cpu:.6} ==");
        log::info!("DIAG seq top5: {:?}", topk_indices(&s1)[..5].to_vec());
        log::info!("DIAG cpu top5: {:?}", topk_indices(&cpu1)[..5].to_vec());

        // tok 路径单 token vs CPU（确认 tok 是正确基准）
        let mut cpu_state2 = model.init_state();
        let cpu2 = model.forward(&single, &mut cpu_state2);
        let d_tok_cpu = t1
            .iter()
            .zip(&cpu2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== DIAG single-token tok vs CPU max_abs_diff: {d_tok_cpu:.6} ==");
        log::info!("DIAG tok top5: {:?}", topk_indices(&t1)[..5].to_vec());
        log::info!("DIAG cpu2 top5: {:?}", topk_indices(&cpu2)[..5].to_vec());

        // 逐层定位非确定性：两次运行 forward_seq 逐层快照 x，找首发发散层
        let d = 8;
        gpu_model.reset_state()?;
        let a1 = gpu_model.snapshot_seq_layers(&single, d)?;
        gpu_model.reset_state()?;
        let a2 = gpu_model.snapshot_seq_layers(&single, d)?;
        for li in 0..a1.len() {
            let md = a1[li]
                .iter()
                .zip(&a2[li])
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, |m, v| m.max(v));
            log::info!("== DIAG layer {li} x run1 vs run2 max_abs_diff: {md:.6} ==");
            if md > 1e-4 {
                log::info!("   run1 x[0..{d}] = {:?}", &a1[li]);
                log::info!("   run2 x[0..{d}] = {:?}", &a2[li]);
            }
        }
        // 逐层定位：seq 快照 vs tok 快照，找首发发散层
        gpu_model.reset_state()?;
        let snaps_seq = gpu_model.snapshot_seq_layers(&single, d)?;
        let snaps_tok = gpu_model.snapshot_layers(&single, d)?;
        for li in 0..snaps_seq.len() {
            let md = snaps_seq[li]
                .iter()
                .zip(&snaps_tok[li])
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, |m, v| m.max(v));
            log::info!("== DIAG layer {li} seq vs tok max_abs_diff: {md:.6} ==");
            if md > 1e-3 {
                log::info!("   seq x[0..{d}] = {:?}", &snaps_seq[li]);
                log::info!("   tok x[0..{d}] = {:?}", &snaps_tok[li]);
            }
        }
        // 诊断：对比 seq 与 tok 路径的各层 tmix_rnn state 前 N 元素（隔离 dplr 状态差异）
        {
            let n = 16usize;
            let nlayer = gpu_model.layers_len();
            gpu_model.reset_state()?;
            gpu_model.forward_seq(&single)?;
            let mut seq_states = Vec::new();
            for i in 0..nlayer {
                seq_states.push(gpu_model.download_state_rnn(i, n)?);
            }
            gpu_model.reset_state()?;
            gpu_model.forward(&single)?;
            let mut tok_states = Vec::new();
            for i in 0..nlayer {
                tok_states.push(gpu_model.download_state_rnn(i, n)?);
            }
            for i in 0..nlayer {
                let md = seq_states[i]
                    .iter()
                    .zip(&tok_states[i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                if md > 1e-7 {
                    log::info!("== DIAG state[{i}].tmix_rnn seq vs tok max_abs_diff: {md:.6} ==");
                }
            }
            // 打印最后几层
            let last4 = [28usize, 29, 30, 31];
            for &i in last4.iter() {
                if i < nlayer {
                    let md = seq_states[i]
                        .iter()
                        .zip(&tok_states[i])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, |m, v| m.max(v));
                    log::info!("== DIAG state[{i}].tmix_rnn seq vs tok max_abs_diff: {md:.9} ==");
                    log::info!("   seq [0..8] = {:?}", &seq_states[i][..8]);
                    log::info!("   tok [0..8] = {:?}", &tok_states[i][..8]);
                }
            }
        }
        // 诊断：双 token 跨 token 状态分量对比（tmix_x / cmix_x / tmix_rnn），定位跨 token 分歧源
        {
            let two: Vec<u32> = vec![304, 25740];
            let n = 8usize;
            let nlayer = gpu_model.layers_len();
            gpu_model.reset_state()?;
            gpu_model.forward_seq(&two)?;
            let mut seq2 = Vec::new();
            for i in 0..nlayer {
                seq2.push((
                    gpu_model.download_state_tmix_x(i, n)?,
                    gpu_model.download_state_cmix_x(i, n)?,
                    gpu_model.download_state_rnn(i, n)?,
                ));
            }
            gpu_model.reset_state()?;
            gpu_model.forward(&two)?;
            let mut tok2 = Vec::new();
            for i in 0..nlayer {
                tok2.push((
                    gpu_model.download_state_tmix_x(i, n)?,
                    gpu_model.download_state_cmix_x(i, n)?,
                    gpu_model.download_state_rnn(i, n)?,
                ));
            }
            for i in 0..nlayer {
                let d_tx = seq2[i]
                    .0
                    .iter()
                    .zip(&tok2[i].0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                let d_cx = seq2[i]
                    .1
                    .iter()
                    .zip(&tok2[i].1)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                let d_rnn = seq2[i]
                    .2
                    .iter()
                    .zip(&tok2[i].2)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                if d_tx > 1e-4 || d_cx > 1e-4 || d_rnn > 1e-4 {
                    log::info!(
                        "== DIAG2 tok-vs-seq after 2tok layer[{i}] tmix_x={d_tx:.6} cmix_x={d_cx:.6} tmix_rnn={d_rnn:.6} =="
                    );
                    if d_tx > 1e-4 {
                        log::info!("   seq tmix_x[0..{n}] = {:?}", &seq2[i].0);
                        log::info!("   tok tmix_x[0..{n}] = {:?}", &tok2[i].0);
                    }
                }
                if i >= 8 && (d_tx > 1e-4 || d_cx > 1e-4 || d_rnn > 1e-4) {
                    break;
                }
            }
        }
        // 诊断：对比 seq 与 tok 路径进入第 i 层前的 x_norm 和 x 输入，定位首发差异来源
        {
            let n = 8usize;
            let nlayer = gpu_model.layers_len();
            for i in 0..nlayer {
                let (x_seq, xn_seq) = gpu_model.diag_seq_x_and_xnorm_before_layer(&single, i, n)?;
                let (x_tok, xn_tok) = gpu_model.diag_tok_x_and_xnorm_before_layer(&single, i, n)?;
                let md_x = x_seq
                    .iter()
                    .zip(&x_tok)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                let md_xn = xn_seq
                    .iter()
                    .zip(&xn_tok)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |m, v| m.max(v));
                if md_x > 1e-7 || md_xn > 1e-7 {
                    log::info!(
                        "== DIAG before_layer[{i}] x_diff={md_x:.9}, x_norm_diff={md_xn:.9} =="
                    );
                    if md_x > 1e-7 {
                        log::info!("   x_seq[0..{n}] = {:?}", &x_seq);
                        log::info!("   x_tok[0..{n}] = {:?}", &x_tok);
                    }
                    if md_xn > 1e-7 {
                        log::info!("   xn_seq[0..{n}] = {:?}", &xn_seq);
                        log::info!("   xn_tok[0..{n}] = {:?}", &xn_tok);
                    }
                    // 前 5 层就停（首发层最重要）
                    if i >= 5 {
                        break;
                    }
                }
            }
        }
        return Ok(());
    }

    // 单 token 二分定位：先比较第一个 token 的输出
    {
        let single = vec![tokens[0]];
        let mut cpu_state1 = model.init_state();
        let cpu1 = model.forward(&single, &mut cpu_state1);
        let gpu1 = gpu_model.forward(&single)?;
        let d1 = cpu1
            .iter()
            .zip(&gpu1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== single-token (first) diff: {d1:.6} ==");
        log::info!("cpu1 top5: {:?}", topk_indices(&cpu1)[..5].to_vec());
        log::info!("gpu1 top5: {:?}", topk_indices(&gpu1)[..5].to_vec());
    }

    // 重置 gpu_model 状态，跑完整 tokens
    gpu_model.reset_state()?;
    let gpu_logits = gpu_model.forward(&tokens)?;
    log::info!("gpu forward done");

    // ===== GPU argmax 采样正确性验证（ARGMAX_VERIFY=1 时）：forward_argmax vs forward+CPU argmax =====
    if std::env::var("ARGMAX_VERIFY").is_ok() {
        gpu_model.reset_state()?;
        let gpu_tok = gpu_model.forward_argmax(&tokens)?;
        let cpu_tok = topk_indices(&gpu_logits)[0] as u32;
        log::info!(
            "== ARGMAX_VERIFY: forward_argmax token={gpu_tok} vs forward+CPU argmax={cpu_tok} match={} ==",
            gpu_tok == cpu_tok
        );

        // GPU self-loop 批量生成验证：self-loop 与逐 token forward_argmax 应产生相同序列
        gpu_model.reset_state()?;
        let seed = tokens[0];
        let sl_n = std::env::var("SELFLOOP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8usize);
        let sl = gpu_model.forward_argmax_selfloop(seed, sl_n)?;
        // 逐 token 参考序列：先 reset 再逐 token argmax（self-loop 内部已保持状态）
        gpu_model.reset_state()?;
        let mut ref_seq = Vec::with_capacity(sl_n);
        let mut t0 = seed;
        for _ in 0..sl_n {
            t0 = gpu_model.forward_argmax(&[t0])?;
            ref_seq.push(t0);
        }
        let match_all = sl == ref_seq;
        log::info!(
            "== SELFLOOP_VERIFY: N={sl_n} selfloop={sl:?}\n                      reference={ref_seq:?}\n                      match={match_all} =="
        );
    }

    // GPU vs CPU top-k 对比
    let topk_cpu: Vec<usize> = {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed[..10].iter().map(|&(i, _)| i).collect()
    };
    let topk_gpu: Vec<usize> = {
        let mut indexed: Vec<(usize, f32)> = gpu_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed[..10].iter().map(|&(i, _)| i).collect()
    };

    // 统计 logits 差异
    let mut max_diff = 0.0f32;
    let mut sum_sq = 0.0f32;
    let n = logits.len();
    for i in 0..n {
        let d = (gpu_logits[i] - logits[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_sq += d * d;
    }
    let rmse = (sum_sq / n as f32).sqrt();

    log::info!("== GPU vs CPU ==");
    log::info!("max_abs_diff: {max_diff:.6}");
    log::info!("rmse: {rmse:.6}");
    log::info!("cpu top10: {topk_cpu:?}");
    log::info!("gpu top10: {topk_gpu:?}");

    // ===== forward_seq vs CPU fp32 参考（正确性 ground truth）=====
    {
        let mut cpu_state = model.init_state();
        let cpu_ref = model.forward(&tokens, &mut cpu_state);
        gpu_model.reset_state()?;
        let seq_logits = gpu_model.forward_seq(&tokens)?;
        let max_diff = seq_logits
            .iter()
            .zip(&cpu_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== forward_seq vs CPU fp32 ==");
        log::info!("seq vs cpu max_abs_diff: {max_diff:.6}");
        log::info!("seq top5: {:?}", topk_indices(&seq_logits)[..5].to_vec());
        log::info!("cpu top5: {:?}", topk_indices(&cpu_ref)[..5].to_vec());
    }

    // ===== 基准性能测量 =====
    {
        use std::time::Instant;

        // 构造 N 个 token 的固定序列（伪随机，测纯推理吞吐，不重新生成 logits 采样）
        let n_tokens = std::env::var("N_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64usize);
        let mut rng = fastrand::Rng::with_seed(7);
        let bench_tokens: Vec<u32> = (0..n_tokens).map(|_| rng.u32(0..10_000u32)).collect();
        log::info!("== benchmark: {n_tokens} tokens, pure inference (no sampling) ==");

        // CPU 吞吐（设置 SKIP_CPU=1 可跳过慢速 CPU 基准，加快迭代验证）
        if std::env::var("SKIP_CPU").is_err() {
            let mut st = model.init_state();
            // 预热（加载 GPU 模型后冷缓存）
            let _ = model.forward(&bench_tokens, &mut st);
            let mut st = model.init_state();
            let t0 = Instant::now();
            let _ = model.forward(&bench_tokens, &mut st);
            let dt = t0.elapsed().as_secs_f64();
            log::info!(
                "CPU : {n_tokens} tokens in {:.3} s  →  {:.1} tokens/s  ({:.3} ms/token)",
                dt,
                n_tokens as f64 / dt,
                dt * 1000.0 / n_tokens as f64
            );
        }

        // GPU 逐 token 吞吐（真实推理）
        // 多次运行取平均，减少 GPU 降频/缓存波动（单次测量误差可达 ±50%）
        {
            let iters = std::env::var("BENCH_ITERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5usize);
            // 预热
            gpu_model.reset_state()?;
            let _ = gpu_model.forward(&bench_tokens)?;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                gpu_model.reset_state()?;
                let t0 = Instant::now();
                let _ = gpu_model.forward(&bench_tokens)?;
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // 去掉最长一次（可能受降频/干扰拖累），取剩余平均
            let kept = &times[..times.len().saturating_sub(1)];
            let dt = kept.iter().sum::<f64>() / kept.len() as f64;
            log::info!(
                "GPU(逐token): {n_tokens} tokens in {:.3} s  →  {:.1} tokens/s  ({:.3} ms/token)  [iters={times:?}]",
                dt,
                n_tokens as f64 / dt,
                dt * 1000.0 / n_tokens as f64
            );
        }

        // GPU sequence-parallel 吞吐（对标 albatross forward_seq）
        // 多次运行取平均，减少 GPU 降频/缓存波动（单次测量误差可达 ±50%）
        {
            let iters = std::env::var("BENCH_ITERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5usize);
            // 预热
            gpu_model.reset_state()?;
            let _ = gpu_model.forward_seq(&bench_tokens)?;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                gpu_model.reset_state()?;
                let t0 = Instant::now();
                let _ = gpu_model.forward_seq(&bench_tokens)?;
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // 去掉最长一次（可能受降频/干扰拖累），取剩余平均
            let kept = &times[..times.len().saturating_sub(1)];
            let dt = kept.iter().sum::<f64>() / kept.len() as f64;
            log::info!(
                "GPU(seq):   {n_tokens} tokens in {:.3} s  →  {:.1} tokens/s  ({:.3} ms/token)  [iters={times:?}, avg={:.3}s]",
                dt,
                n_tokens as f64 / dt,
                dt * 1000.0 / n_tokens as f64,
                times
                    .iter()
                    .map(|x| x * 1000.0)
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|x| format!("{x:.1}"))
                    .collect::<Vec<_>>()
                    .join("/")
            );
        }

        // GPU 逐 token + argmax 采样吞吐（对标 albatross torch.argmax，只回传 token 索引）
        // 与上面的 GPU(逐token) 同进程同热态，直接对比 GPU 采样是否引入额外开销
        {
            let iters = std::env::var("BENCH_ITERS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5usize);
            gpu_model.reset_state()?;
            let _ = gpu_model.forward_argmax(&bench_tokens)?;
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                gpu_model.reset_state()?;
                let t0 = Instant::now();
                let _ = gpu_model.forward_argmax(&bench_tokens)?;
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let kept = &times[..times.len().saturating_sub(1)];
            let dt = kept.iter().sum::<f64>() / kept.len() as f64;
            log::info!(
                "GPU(argmax): {n_tokens} tokens in {:.3} s  →  {:.1} tokens/s  ({:.3} ms/token)  [iters={times:?}]",
                dt,
                n_tokens as f64 / dt,
                dt * 1000.0 / n_tokens as f64
            );
        }
    }

    // ===== forward_seq 正确性验证（与逐 token 输出对比）=====
    {
        let probe_tokens: Vec<u32> = vec![304, 25740, 109];
        // 单 token 定位：隔离 GEMM 与跨 token state 传递
        let single = vec![probe_tokens[0]];
        gpu_model.reset_state()?;
        let s1 = gpu_model.forward_seq(&single)?;
        gpu_model.reset_state()?;
        let t1 = gpu_model.forward(&single)?;
        let d1 = s1
            .iter()
            .zip(&t1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== forward_seq vs forward (single token) ==");
        log::info!("seq1 top5: {:?}", topk_indices(&s1)[..5].to_vec());
        log::info!("tok1 top5: {:?}", topk_indices(&t1)[..5].to_vec());
        log::info!("single-token max_abs_diff: {d1:.6}");

        gpu_model.reset_state()?;
        let seq_logits = gpu_model.forward_seq(&probe_tokens)?;
        gpu_model.reset_state()?;
        let tok_logits = gpu_model.forward(&probe_tokens)?;
        let max_diff = seq_logits
            .iter()
            .zip(&tok_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |m, v| m.max(v));
        log::info!("== forward_seq vs forward(逐token) ==");
        log::info!("seq top5: {:?}", topk_indices(&seq_logits)[..5].to_vec());
        log::info!("tok top5: {:?}", topk_indices(&tok_logits)[..5].to_vec());
        log::info!("max_abs_diff: {max_diff:.6}");
    }

    // ===== 诊断：tensor-core GEMM 用真实各类权重 + 随机激活（K=C=2560）=====
    {
        use std::fs::File;
        let file = File::open(&model_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;

        let m = 256usize;
        // 随机激活 [-0.5, 0.5]
        let mut a = vec![0.0f32; m * 2560];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        let a16 = a
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect::<Vec<_>>();

        let mut rt2 = runtime::Runtime::new()?;
        let a16t = rt2.create_tensor_f16(m * 2560)?;
        rt2.upload_f16(&a16t, &a)?;

        for (wname, wcpu_t) in [
            (
                "blocks.0.att.receptance.weight",
                &model.layers[0].receptance_w,
            ),
            ("blocks.0.att.key.weight", &model.layers[0].key_w),
            ("blocks.0.att.value.weight", &model.layers[0].value_w),
            ("blocks.0.att.output.weight", &model.layers[0].output_w),
        ] {
            // int8 模型原 fp16 键已删除，回退用 CPU 端反量化权重（[in,out] 转置回 [out,in]）
            let (n, k, wf): (usize, usize, Vec<f32>) = match st.tensor(wname) {
                Ok(w) => {
                    let shape = w.shape();
                    let wf: Vec<f32> = match w.dtype() {
                        safetensors::tensor::Dtype::F32 => {
                            bytemuck::cast_slice::<u8, f32>(w.data()).to_vec()
                        }
                        safetensors::tensor::Dtype::F16 => {
                            bytemuck::cast_slice::<u8, u16>(w.data())
                                .iter()
                                .map(|&b| half::f16::from_bits(b).to_f32())
                                .collect()
                        }
                        d => panic!("unsupported diag dtype {d:?}"),
                    };
                    (shape[0], shape[1], wf)
                }
                Err(_) => {
                    let c = model.config.n_embd;
                    let mut wf = vec![0.0f32; c * c];
                    for j in 0..c {
                        for z in 0..c {
                            wf[j * c + z] = wcpu_t[z * c + j];
                        }
                    }
                    (c, c, wf)
                }
            };
            let wmax = wf.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            log::info!("diag {wname} shape [{n}, {k}] max|w|={wmax:.3}");

            let b16t = rt2.create_tensor_f16(n * k)?;
            rt2.upload_f16(&b16t, &wf)?;
            let mut c = rt2.create_tensor(m * n)?;
            rt2.begin_batch()?;
            rt2.gemm(&a16t, &b16t, &mut c, m, n, k)?;
            rt2.end_batch()?;
            let got = rt2.download(&c)?;

            // 参考1：fp16 输入+fp16 权重（隔离 GEMM 内核正确性）
            // 参考2：fp32 输入+fp32 权重（基准，量化误差上限）
            let b16ref: Vec<half::f16> = wf.iter().map(|&x| half::f16::from_f32(x)).collect();
            let mut max_gemm = 0.0f32;
            let mut max_quant = 0.0f32;
            for i in 0..4 {
                for j in 0..n {
                    let mut s16 = 0.0f32;
                    let mut s32 = 0.0f32;
                    for z in 0..k {
                        s16 += a16[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                        s32 += a[i * k + z] * wf[j * k + z];
                    }
                    let g = (got[i * n + j] - s16).abs();
                    let q = (got[i * n + j] - s32).abs();
                    if g > max_gemm {
                        max_gemm = g;
                    }
                    if q > max_quant {
                        max_quant = q;
                    }
                }
            }
            log::info!(
                "diag gemm {wname} (m={m} n={n} k={k}) 内核误差={max_gemm:.6} 量化误差(fp16-vs-fp32)={max_quant:.6}"
            );
        }
    }

    Ok(())
}
