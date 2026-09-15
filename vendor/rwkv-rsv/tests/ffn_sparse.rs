//! Vulkan 稀疏 FFN value 投影回归测试：x += r2 @ ffn_value（r2=relu²，约 96% 稀疏）。
//! 直接调用 Runtime::ffn_value_sparse_add，与 CPU 参考对比。对标 CUDA 的
//! `ffn_value_sparse_matches_cpu` 单测。仅在 Vulkan 后端下有效（通用 Runtime 层）。
use rwkv_rsv::runtime::Runtime;

#[test]
fn ffn_value_sparse_matches_cpu() {
    let c = 256usize;
    let fh = 512usize;
    const TILE: usize = 128;
    const C_TILE: usize = 256;

    // 构造 A[c][fh]（行主序：[c, fh]）
    let mut a = vec![0.0f32; c * fh];
    for cc in 0..c {
        for f in 0..fh {
            a[cc * fh + f] = ((cc as f32) + 1.0) * 0.001 + (f as f32) * 0.0001;
        }
    }
    // r2[fh]：96% 稀疏
    let mut r2 = vec![0.0f32; fh];
    for i in 0..fh / 16 {
        r2[i * 16] = (i as f32) * 0.5 + 1.0;
    }

    // CPU 参考 x = A @ r2（[c]）
    let mut x_cpu = vec![0.0f32; c];
    for cc in 0..c {
        let mut s = 0.0f32;
        for f in 0..fh {
            s += r2[f] * a[cc * fh + f];
        }
        x_cpu[cc] = s;
    }

    // 构造 value_tiled（与 gpu_model::load_ffn_value_tiled 一致）
    let c_blocks = c / C_TILE;
    let mut tiled = vec![0.0f32; fh * c];
    for f in 0..fh {
        let f_block = f / TILE;
        let f_local = f % TILE;
        for cc in 0..c {
            let c_block = cc / C_TILE;
            let c_local = cc % C_TILE;
            tiled[((f_block * c_blocks + c_block) * TILE) * C_TILE + f_local * C_TILE + c_local] =
                a[cc * fh + f];
        }
    }

    let mut rt = Runtime::new().expect("runtime");
    let vt = rt.create_tensor_f16(tiled.len()).expect("vt");
    rt.upload_f16(&vt, &tiled).expect("upload vt");
    let rg = rt.create_tensor(r2.len()).expect("rg");
    rt.upload(&rg, &r2).expect("upload r2");
    let mut xg = rt.create_tensor(c).expect("xg");
    let zero = vec![0.0f32; c];
    rt.upload(&xg, &zero).expect("upload x");

    rt.begin_batch().expect("begin");
    rt.ffn_value_sparse_add(&vt, &rg, &mut xg, c, fh)
        .expect("sparse");
    rt.end_batch().expect("end");

    let got = rt.download(&xg).expect("download");
    let max_diff = got
        .iter()
        .zip(x_cpu.iter())
        .map(|(g, cc)| (g - cc).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-2, "mismatch max_diff={max_diff:e}");
}
