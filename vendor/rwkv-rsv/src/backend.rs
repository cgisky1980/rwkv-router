//! 后端统一抽象层
//!
//! 定义平台无关的张量句柄 `TensorId` 与算子级抽象 `ComputeBackend` trait。
//! 现有 Vulkan runtime（`runtime::Runtime`）被封装为 `VulkanBackend`，未来可新增
//! `CudaBackend` / `MetalBackend` 等实现，通过 `detect_backend()` 在启动时自动选择。
//!
//! 设计约束：
//! - 算子签名一律使用 `TensorId`，不泄漏任何 Vulkan/CUDA 专属类型。
//! - 批处理只暴露 `begin_batch`/`end_batch`，后端内部实现各自语义（Vulkan command
//!   buffer vs CUDA stream）。
//! - 不破坏现有 CPU fp32 参考与逐层精度验证。

use std::collections::HashMap;

use crate::runtime::{GpuTensor, GpuTensor16, GpuTensorInt8, GpuTensorU32, R, Runtime};

/// 平台无关张量句柄（由后端分配，内部映射到设备内存）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

/// 张量数据类型（平台无关）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    U32,
}

/// 后端类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Vulkan,
    Cuda,
}

/// int8 量化权重（平台无关组合句柄）：`w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]`。
/// 一组 `idx`/`sz` 二路设备张量 + 形状元数据。
#[derive(Debug, Clone, Copy)]
pub struct Int8Handle {
    pub idx: TensorId,
    pub sz: TensorId,
    pub m: usize,
    pub k: usize,
}

/// 算子级后端抽象。首期抽象张量管理与 decode（单 token）路径全部算子。
/// 所有张量以平台无关 `TensorId` 传递；量化权重以 `Int8Handle` 组合句柄传递。
pub trait ComputeBackend {
    // —— 张量管理 ——
    fn create_tensor(&mut self, len: usize, dtype: TensorDtype) -> R<TensorId>;
    fn upload(&self, t: TensorId, data: &[f32]) -> R<()>;
    /// 部分上传：把 data 写入张量 [offset, offset+len) 段（f32 元素偏移），
    /// 其余部分不动（batch State 的 slot 回灌用）。
    fn upload_part(&self, t: TensorId, offset: usize, data: &[f32]) -> R<()> {
        let _ = (t, offset, data);
        Err("upload_part not supported by this backend".into())
    }
    fn upload_u32(&self, t: TensorId, data: &[u32]) -> R<()>;
    fn download(&self, t: TensorId) -> R<Vec<f32>>;
    fn download_u32(&self, t: TensorId) -> R<Vec<u32>>;

    // —— 批处理（一次提交多条 kernel）——
    fn begin_batch(&mut self) -> R<()>;
    fn end_batch(&mut self) -> R<()>;

    /// 清空累计的 per-kernel profiling 时间（仅诊断；CUDA 覆盖，其余为 no-op）。
    fn clear_kernel_prof(&mut self) {}

    /// 打印累计的 per-kernel profiling 时间（仅诊断；CUDA 覆盖，其余为 no-op）。
    fn dump_kernel_prof(&mut self) {}

    // —— CUDA graph 捕获/重放（decode 每 token launch 开销优化）——
    /// 开始捕获后续 kernel 启动到 CUDA graph（重放时不再逐次 cuLaunchKernel）。
    /// 默认 no-op；仅支持的后端（CUDA）覆盖。
    fn begin_graph_capture(&mut self) -> R<()> {
        Ok(())
    }
    /// 结束捕获并实例化可执行 graph。
    fn end_graph_capture(&mut self) -> R<()> {
        Ok(())
    }
    /// 重放已捕获的可执行 graph（固定的一组 kernel 序列）。
    fn graph_replay(&mut self) -> R<()> {
        Ok(())
    }
    /// 是否支持 CUDA graph 捕获/重放（self-loop 用）。默认 false；仅 CUDA 覆盖为 true。
    /// 不支持的后端（Vulkan）在 self-loop 时改为把完整前向逐 token 记录进同一批次。
    fn supports_graph_capture(&self) -> bool {
        false
    }

    // —— CUDA prefill graph（整段 prefill 一次抓到图，消除跨层 launch 开销）——
    /// 当前 T 是否已有捕获好的 prefill graph（无 → 需重新捕获）。
    /// 默认 no-op；仅支持的后端（CUDA）覆盖。
    fn prefill_graph_valid(&mut self, _t: usize) -> R<bool> {
        Ok(false)
    }
    /// 开始捕获整段 prefill（须在 x 上传之后调用），`t` 为本次 token 数。
    fn begin_prefill_capture(&mut self, _t: usize) -> R<()> {
        Ok(())
    }
    /// 结束捕获并按 `begin_prefill_capture` 的 t 绑定保存可执行 graph。
    fn end_prefill_capture(&mut self) -> R<()> {
        Ok(())
    }
    /// 重放当前 T 的 prefill graph（已上传新 x）。
    fn prefill_graph_replay(&mut self) -> R<()> {
        Ok(())
    }

    // —— host 前后端（embedding gather / 采样）——
    /// 把 token 索引（u32 位模式）写入 host-visible 缓冲（无 kernel、无 spec）。
    fn store_token_host(&self, tok: TensorId, token: u32) -> R<()>;
    /// 异步写入 sampler 参数行（selfloop 逐轮 seed/hist_len 更新）：
    /// 参数写入后端 pinned 暂存区第 `row` 行，`cuMemcpyHtoDAsync` 流序执行
    /// （此前排队的 kernel 之后），零 host 同步。仅支持的后端（CUDA）覆盖；
    /// 不支持时调用方回退同步 `store_sampler_host`。
    #[allow(clippy::too_many_arguments)]
    fn store_sampler_async(
        &self,
        _sampler: TensorId,
        _row: usize,
        _temperature: f32,
        _top_k: u32,
        _top_p: f32,
        _seed: u32,
        _repetition_penalty: f32,
        _frequency_penalty: f32,
        _presence_penalty: f32,
        _hist_len: u32,
    ) -> R<()> {
        Err("store_sampler_async not supported by this backend".into())
    }
    /// pinned 暂存区行数上限（async sampler 路径的 selfloop n 上限）。
    fn sampler_async_rows(&self) -> usize {
        0
    }
    /// 从同进程另一后端导入全部张量（权重共享多流并发）：同一 CUDA primary
    /// ctx 下设备指针跨实例有效；导入张量保持原 TensorId，新实例不拥有
    ///（Drop 不释放）。仅支持的后端（CUDA）覆盖。
    fn import_tensors_from(&mut self, _src: &dyn ComputeBackend) -> R<()> {
        Err("import_tensors_from not supported by this backend".into())
    }
    /// trait 对象向下转型（`import_tensors_from` 需读取源后端内部张量表）。
    fn as_any(&self) -> &dyn std::any::Any;
    /// 从 fp16 表按 token 索引取一行 → dst（f32）；tok 为 host 缓冲（存 token 位模式）。
    fn gather_row_device_f16(
        &mut self,
        src: TensorId,
        dst: TensorId,
        tok: TensorId,
        c: usize,
    ) -> R<()>;
    /// fp16 缓冲间的设备侧拷贝（v_first 快照用）。
    fn copy_device_f16(&mut self, src: TensorId, dst: TensorId) -> R<()>;

    // —— 核心算子（RWKV-7 decode 路径）——
    /// y = x @ A（fp16 权重，f32 输入/输出）；w=m*k, x=k*n, y=m*n
    #[allow(clippy::too_many_arguments)] // 算子签名沿用 Runtime 的扁平参数约定
    fn gemv_f16(
        &mut self,
        w: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        n: usize,
    ) -> R<()>;
    /// y = layer_norm(x) * gamma + beta；c=通道数, h=每行元素数, rows=批行数
    #[allow(clippy::too_many_arguments)] // 算子签名沿用 Runtime 的扁平参数约定
    fn norm(
        &mut self,
        x: TensorId,
        gamma: TensorId,
        beta: TensorId,
        y: TensorId,
        c: usize,
        h: usize,
        eps: f32,
        rows: usize,
    ) -> R<()>;

    /// 深度融合 time-mix：ln1 = layer_norm(x) + 6 次 lerp(xr/xw/xk/xv/xa/xg) + state 写回。
    #[allow(clippy::too_many_arguments)]
    fn norm_lerp6(
        &mut self,
        x: TensorId,
        state: TensorId,
        gamma: TensorId,
        beta: TensorId,
        x_r: TensorId,
        x_w: TensorId,
        x_k: TensorId,
        x_v: TensorId,
        x_a: TensorId,
        x_g: TensorId,
        o_r: TensorId,
        o_w: TensorId,
        o_k: TensorId,
        o_v: TensorId,
        o_a: TensorId,
        o_g: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()>;

    /// 深度融合 channel-mix：xb = ln2(ln2_w/ln2_b) + ffn_x_k*(prev_c - ln2) + state 写回。
    #[allow(clippy::too_many_arguments)]
    fn cmix_norm_lerp(
        &mut self,
        x: TensorId,
        state: TensorId,
        gamma: TensorId,
        beta: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()>;

    /// 融合 kernel：fuse_ka + dplr(S 更新) + group_norm + sum_rk_rk，一次 launch。
    #[allow(clippy::too_many_arguments)]
    fn fuse_ka_dplr_norm(
        &mut self,
        s: TensorId,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        r: TensorId,
        v: TensorId,
        w: TensorId,
        gamma: TensorId,
        beta: TensorId,
        r_k: TensorId,
        k_mod: TensorId,
        y: TensorId,
        y_norm: TensorId,
        h: usize,
        n: usize,
        eps: f32,
        gn_eps: f32,
    ) -> R<()>;

    /// 深度融合 gemv（fp16 版）：r/k/v 三个 C×C 投影 + v1/w1/a1/g1 四个 mid 投影，一次 dispatch。
    #[allow(clippy::too_many_arguments)]
    fn gemv_rkv_stage1(
        &mut self,
        r: TensorId,
        k: TensorId,
        v: TensorId,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        out_r: TensorId,
        out_k: TensorId,
        out_v: TensorId,
        out_vm: TensorId,
        out_wm: TensorId,
        out_am: TensorId,
        out_gm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()>;

    /// 深度融合 gemv（int8 量化版，r/k/v 三路同量化格式）。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_rkv_stage1(
        &mut self,
        r: &Int8Handle,
        k: &Int8Handle,
        v: &Int8Handle,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        out_r: TensorId,
        out_k: TensorId,
        out_v: TensorId,
        out_vm: TensorId,
        out_wm: TensorId,
        out_am: TensorId,
        out_gm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()>;

    /// 低秩链第二级融合（w/a/g/v 二级投影 + 激活，1 次 dispatch）。
    #[allow(clippy::too_many_arguments)]
    fn gemv_lowrank_chain4(
        &mut self,
        w2: TensorId,
        a2: TensorId,
        v2: TensorId,
        g2: TensorId,
        w_mid: TensorId,
        a_mid: TensorId,
        v_mid: TensorId,
        g_mid: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        v_first: TensorId,
        out_w: TensorId,
        out_a: TensorId,
        out_v: TensorId,
        out_g: TensorId,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
    ) -> R<()>;

    /// y = relu²(x @ A)（fp16 权重）——ffn.key 用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_f16_relu2(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;
    /// y = relu²(x @ A)（int8 权重）——ffn.key 用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_relu2(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;

    /// y += (x .* g) @ A（fp16 权重，f32 累加）——att.output 用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_f16_mul_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;
    /// y += (x .* g) @ A（int8 权重）——att.output 用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_mul_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;

    /// y += x @ A（fp16 权重，f32 累加）——ffn.value 残差用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_f16_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;
    /// y += x @ A（int8 权重）——ffn.value 残差用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;
    /// y = x @ A（int8 权重，覆盖写）——head 用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_plain(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()>;

    /// 稀疏 FFN value 投影：x += r2 @ ffn_value（r2=relu² 约 96% 稀疏）。
    /// `value_tiled` 为平铺布局 [fh, C]（对齐 Albatross cmix_sparse_down），CUDA 稀疏内核用；
    /// 默认实现退化为稠密 `gemv_f16_add`（Vulkan 等不支持稀疏内核的 backend 用 `value_w16`）。
    /// `value_w16` 为 Some 时（fp16 模型）才可退化为稠密；None（int8 模型）时调用方应避免进入本路径。
    #[allow(clippy::too_many_arguments)]
    fn ffn_value_sparse_add(
        &mut self,
        value_w16: Option<TensorId>,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
    ) -> R<()> {
        let _ = value_tiled;
        let w16 = value_w16.ok_or_else(|| {
            "sparse FFN 需要稠密 fp16 权重（int8 模型请走稠密量化路径）".to_string()
        })?;
        self.gemv_f16_add(w16, r2, x, c, fh, 1)
    }

    /// 是否支持稀疏 FFN 内核（CUDA 原生支持；Vulkan 等返回 false，走稠密量化路径）。
    fn supports_sparse_ffn(&self) -> bool {
        false
    }

    // —— batch 并发算子（单实例多序列，权重共享读一次算 B 份；仅 CUDA 支持）——
    // 布局约定：所有 per-slot 张量为 [batch, ...]（slot 主序）；权重张量跨 slot 共享。
    // 默认实现返回 Err（Vulkan 等后端暂不支持 batch 并发路径）。

    /// batch 版 embedding gather：按 tok[b] 各取一行 → dst[b*C + i]（f32）。
    #[allow(clippy::too_many_arguments)]
    fn gather_rows_device_f16(
        &mut self,
        _src: TensorId,
        _dst: TensorId,
        _tok: TensorId,
        _c: usize,
        _batch: usize,
    ) -> R<()> {
        Err("gather_rows_device_f16 not supported by this backend".into())
    }
    /// batch 版 norm_lerp6：x/state/xr..xg/or_..og 均为 [batch, C]；gamma/beta 共享。
    #[allow(clippy::too_many_arguments)]
    fn norm_lerp6_batch(
        &mut self,
        _x: TensorId,
        _state: TensorId,
        _gamma: TensorId,
        _beta: TensorId,
        _xr: TensorId,
        _xw: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _or: TensorId,
        _ow: TensorId,
        _ok: TensorId,
        _ov: TensorId,
        _oa: TensorId,
        _og: TensorId,
        _c: usize,
        _eps: f32,
        _batch: usize,
    ) -> R<()> {
        Err("norm_lerp6_batch not supported by this backend".into())
    }
    /// batch 版 cmix_norm_lerp：x/state/out_xb 为 [batch, C]；gamma/beta/coeff 共享。
    #[allow(clippy::too_many_arguments)]
    fn cmix_norm_lerp_batch(
        &mut self,
        _x: TensorId,
        _state: TensorId,
        _gamma: TensorId,
        _beta: TensorId,
        _coeff: TensorId,
        _out_xb: TensorId,
        _c: usize,
        _eps: f32,
        _batch: usize,
    ) -> R<()> {
        Err("cmix_norm_lerp_batch not supported by this backend".into())
    }
    /// batch 版 gemv_int8_rkv_stage1：x 输入与输出为 [batch, ...]；权重共享。
    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_rkv_stage1_batch(
        &mut self,
        _r: &Int8Handle,
        _k: &Int8Handle,
        _v: &Int8Handle,
        _v1: TensorId,
        _w1: TensorId,
        _a1: TensorId,
        _g1: TensorId,
        _xr: TensorId,
        _xk: TensorId,
        _xv: TensorId,
        _xw: TensorId,
        _xa: TensorId,
        _xg: TensorId,
        _out_r: TensorId,
        _out_k: TensorId,
        _out_v: TensorId,
        _out_vm: TensorId,
        _out_wm: TensorId,
        _out_am: TensorId,
        _out_gm: TensorId,
        _c: usize,
        _vm: usize,
        _wm: usize,
        _am: usize,
        _gm: usize,
        _batch: usize,
    ) -> R<()> {
        Err("gemv_int8_rkv_stage1_batch not supported by this backend".into())
    }
    /// batch 版 gemv_lowrank_chain4：mid 输入与 v_first/输出为 [batch, ...]；权重共享。
    #[allow(clippy::too_many_arguments)]
    fn gemv_lowrank_chain4_batch(
        &mut self,
        _w2: TensorId,
        _a2: TensorId,
        _v2: TensorId,
        _g2: TensorId,
        _w_mid: TensorId,
        _a_mid: TensorId,
        _v_mid: TensorId,
        _g_mid: TensorId,
        _w0: TensorId,
        _a0: TensorId,
        _v0: TensorId,
        _scale: TensorId,
        _v_first: TensorId,
        _out_w: TensorId,
        _out_a: TensorId,
        _out_v: TensorId,
        _out_g: TensorId,
        _m: usize,
        _kw: usize,
        _ka: usize,
        _kv: usize,
        _kg: usize,
        _batch: usize,
    ) -> R<()> {
        Err("gemv_lowrank_chain4_batch not supported by this backend".into())
    }
    /// batch 版 fuse_ka_dplr_norm：kernel 本身支持 batch 维（grid.y）。
    #[allow(clippy::too_many_arguments)]
    fn fuse_ka_dplr_norm_batch(
        &mut self,
        _s: TensorId,
        _k: TensorId,
        _k_k: TensorId,
        _a: TensorId,
        _k_a: TensorId,
        _r: TensorId,
        _v: TensorId,
        _w: TensorId,
        _gamma: TensorId,
        _beta: TensorId,
        _r_k: TensorId,
        _k_mod: TensorId,
        _y: TensorId,
        _y_norm: TensorId,
        _h: usize,
        _n: usize,
        _eps: f32,
        _gn_eps: f32,
        _batch: usize,
    ) -> R<()> {
        Err("fuse_ka_dplr_norm_batch not supported by this backend".into())
    }
    /// batch 版 ffn_value_sparse_add：r2 为 [batch, fh]，x 为 [batch, C]；权重共享。
    #[allow(clippy::too_many_arguments)]
    fn ffn_value_sparse_add_batch(
        &mut self,
        _value_tiled: TensorId,
        _r2: TensorId,
        _x: TensorId,
        _c: usize,
        _fh: usize,
        _batch: usize,
    ) -> R<()> {
        Err("ffn_value_sparse_add_batch not supported by this backend".into())
    }
    /// batch 版采样：logits/temp/mask/counter 为 [batch, n]，token 为 [batch]，
    /// sampler 为 [batch, 8]（每 slot 独立参数），hist 为 [batch, hist_len]。
    #[allow(clippy::too_many_arguments)]
    fn sample_into_host_seeded_batch(
        &mut self,
        _logits: TensorId,
        _token: TensorId,
        _n: usize,
        _temp: TensorId,
        _mask: TensorId,
        _counter: TensorId,
        _sampler: TensorId,
        _hist: TensorId,
        _batch: usize,
    ) -> R<()> {
        Err("sample_into_host_seeded_batch not supported by this backend".into())
    }
    /// batch 版 record_token：把 in_tok[b] 追加到 out_seq[b*stride + cnt[b]++]。
    fn record_tokens(
        &mut self,
        _in_tok: TensorId,
        _out_seq: TensorId,
        _cnt: TensorId,
        _stride: usize,
        _batch: usize,
    ) -> R<()> {
        Err("record_tokens not supported by this backend".into())
    }
    /// batch 版异步 sampler 上传：每轮一行宽行（batch*8 f32），pinned 流序零同步。
    /// `row` 为轮次（0..n）；seeds 为每 slot 的 seed（长度 = batch）。
    #[allow(clippy::too_many_arguments)]
    fn store_sampler_async_batch(
        &self,
        _sampler: TensorId,
        _row: usize,
        _temperature: f32,
        _top_k: u32,
        _top_p: f32,
        _seeds: &[u32],
        _repetition_penalty: f32,
        _frequency_penalty: f32,
        _presence_penalty: f32,
        _hist_len: u32,
    ) -> R<()> {
        Err("store_sampler_async_batch not supported by this backend".into())
    }

    /// GPU argmax：把 logits 的 argmax 索引写入 token 缓冲（f32 位模式存 uint）。
    fn argmax(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()>;
    /// GPU 采样（temperature/top-k/top-p + OpenAI 兼容惩罚），结果写 token 缓冲。
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        history: &[u32],
    ) -> R<()>;

    // —— 内存/缓存管理 ——
    /// 清空后端 kernel 缓存（seq 路径在 T 变化重建缓冲后调用，避免 descriptor pool 耗尽）。
    fn clear_cache(&mut self);
    /// 释放 f32/f16/u32 张量的 host（系统内存）缓冲（权重上传完成后调用）。
    fn drop_host(&mut self, t: TensorId);
    /// 释放张量的设备内存并从注册表移除（seq 缓冲按 T 重建时调用，防设备内存泄漏）。
    fn free_tensor(&mut self, t: TensorId);

    // —— 设备侧拷贝 ——
    /// device→device 拷贝（f32）：v_first 快照 / 状态缓冲用。
    fn copy_device(&mut self, src: TensorId, dst: TensorId) -> R<()>;
    /// 拷贝 tensor x 的第 token 行（stride 行宽）到 [C] 缓冲 y（状态更新用）。
    fn copy_token(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        stride: usize,
        token: usize,
    ) -> R<()>;

    // —— seq/prefill 路径算子（tensor-core GEMM / 融合 kernel）——
    /// C[M,N] = A[M,K] @ B[N,K]^T（fp16 输入，f32 累加/输出）。M/N 需为 TILE 倍数，K 为 32 倍数。
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()>;
    /// C[M,N] = (A[M,K] @ B[N,K]^T) + bias[n]（fp16 输入，f32 输出）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_bias(
        &mut self,
        a: TensorId,
        b: TensorId,
        bias: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()>;
    /// C[M,N] = (A[M,K] @ B[N,K]^T) + x[M,N]（原地累加残差，fp16 输入，f32 输出）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_add(
        &mut self,
        a: TensorId,
        b: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()>;
    /// f32 → f16 转换（token 并行，支持跨步布局）。
    #[allow(clippy::too_many_arguments)]
    fn to_f16(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()>;
    /// f32 → f16 三输入融合转换（一次把 xr/xk/xv 转成 fp16）。
    #[allow(clippy::too_many_arguments)]
    fn to_f16_triple(
        &mut self,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        yr: TensorId,
        yk: TensorId,
        yv: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()>;
    /// int8 → fp16 反量化（prefill：解到 fp16 scratch 供 GEMM 消费）。
    #[allow(clippy::too_many_arguments)]
    fn dequant_int8_to_f16(&mut self, a: &Int8Handle, out: TensorId, m: usize, k: usize) -> R<()>;
    /// y = sigmoid(a)（f32，token 并行）。
    #[allow(clippy::too_many_arguments)]
    fn elementwise_sigmoid(&mut self, a: TensorId, y: TensorId, c: usize, batch: usize) -> R<()>;
    /// y = sigmoid(y)（原地，f32）。
    #[allow(clippy::too_many_arguments)]
    fn elementwise_sigmoid_inplace(&mut self, y: TensorId, c: usize, batch: usize) -> R<()>;
    /// 融合 fuse_ka：k_mod/kk_l2/b（f32，token 并行）。
    #[allow(clippy::too_many_arguments)]
    fn fuse_ka(
        &mut self,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        k_mod: TensorId,
        kk_l2: TensorId,
        b: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()>;
    /// y += sum(r * k_mod * r_k, 按 head 归约) * v（f32）。
    #[allow(clippy::too_many_arguments)]
    fn sum_rk_rk(
        &mut self,
        r: TensorId,
        k_mod: TensorId,
        r_k: TensorId,
        v: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()>;
    /// token shift + time-mix 插值（sequence-parallel，f32）。
    #[allow(clippy::too_many_arguments)]
    fn seq_shift(
        &mut self,
        x: TensorId,
        state: TensorId,
        tm: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> R<()>;
    /// DPLR 状态更新（sequence-parallel，内部循环 T）。
    #[allow(clippy::too_many_arguments)]
    fn dplr_seq(
        &mut self,
        s: TensorId,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
    ) -> R<()>;
    // —— batch prefill 算子（B 序列 × T_pad token，[batch, T, C] 布局）——
    /// batch 版 token shift：t=0 读该 slot 的 state 段。仅 CUDA。
    #[allow(clippy::too_many_arguments)]
    fn seq_shift_batch(
        &mut self,
        _x: TensorId,
        _state: TensorId,
        _tm: TensorId,
        _y: TensorId,
        _c: usize,
        _t: usize,
        _stride_x: usize,
        _stride_y: usize,
        _batch: usize,
    ) -> R<()> {
        Err("seq_shift_batch not supported by this backend".into())
    }
    /// batch 版 copy_token：每 slot 把 lens[b]-1 行拷到 state[b]。仅 CUDA。
    fn copy_token_batch(
        &mut self,
        _x: TensorId,
        _state: TensorId,
        _lens: TensorId,
        _c: usize,
        _t: usize,
        _batch: usize,
    ) -> R<()> {
        Err("copy_token_batch not supported by this backend".into())
    }
    /// batch 版 DPLR 状态更新：s 为 [batch, H, N*N]，lens 截断实际长度。仅 CUDA。
    #[allow(clippy::too_many_arguments)]
    fn dplr_seq_batch(
        &mut self,
        _s: TensorId,
        _r: TensorId,
        _w: TensorId,
        _k: TensorId,
        _v: TensorId,
        _a: TensorId,
        _b: TensorId,
        _y: TensorId,
        _lens: TensorId,
        _h: usize,
        _n: usize,
        _t: usize,
        _c: usize,
        _batch: usize,
    ) -> R<()> {
        Err("dplr_seq_batch not supported by this backend".into())
    }
    /// 分段均值：x [batch, t_pad, c] → out [batch, c]（每 slot 前 lens[b] 行均值，
    /// pad 行不计入）。批量 mean-hidden 特征提取用。仅 CUDA。
    fn segmean(
        &mut self,
        _x: TensorId,
        _out: TensorId,
        _lens: TensorId,
        _c: usize,
        _t_pad: usize,
        _batch: usize,
    ) -> R<()> {
        Err("segmean not supported by this backend".into())
    }
    /// C[M,N] = relu²(A[M,K] @ B[N,K]^T)（fp16 输入，f32 输出）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_relu2(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()>;
    /// C[M,N] = tanh(A[M,K] @ B[N,K]^T)（fp16 输入，f32 输出）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_tanh(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()>;
    /// y = exp(a * b[0])（b 为每 batch 一个标量；f32）。
    #[allow(clippy::too_many_arguments)]
    fn elementwise_scale_exp(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()>;
    /// y = a * b（f32，token 并行）。
    #[allow(clippy::too_many_arguments)]
    fn elementwise_mul(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()>;
    /// v = v + gate * (v_first - v)（f32，token 并行，stride 为行宽）。
    #[allow(clippy::too_many_arguments)]
    fn v_first_lerp(
        &mut self,
        v: TensorId,
        gate: TensorId,
        v_first: TensorId,
        c: usize,
        t: usize,
        stride: usize,
    ) -> R<()>;
    /// y[a,b] = x[a] @ A（f32 权重，f32 输入/输出，支持跨步批量）——GEMM_DIAG 参考用。
    #[allow(clippy::too_many_arguments)]
    fn gemv_seq(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        x_stride: usize,
        y_stride: usize,
        batch: usize,
    ) -> R<()>;

    // —— self-loop 采样（写 host-visible 缓冲）——
    /// 把采样参数写入 host-visible 缓冲（sampler，F32 len8）。
    #[allow(clippy::too_many_arguments)]
    fn store_sampler_host(
        &self,
        sampler: TensorId,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()>;
    /// GPU 采样（self-loop 批量版）：结果写回 token 的 host-visible 缓冲。
    /// temp/mask/sampler/hist 为 F32，counter 为 U32，均由调用方预建。
    #[allow(clippy::too_many_arguments)]
    fn sample_into_host_seeded(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        hist: TensorId,
    ) -> R<()>;
    /// 把 host-visible 缓冲 in_tok[0] 追加到 out_seq[cnt]，cnt 自增（self-loop 记录用）。
    fn record_token(&mut self, in_tok: TensorId, out_seq: TensorId, cnt: TensorId) -> R<()>;
    /// GPU argmax 直接把索引写入 token 的 host-visible 缓冲（self-loop 用）。
    #[allow(clippy::too_many_arguments)]
    fn argmax_into_host(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()>;
}

/// CUDA 算子实现是否就绪。
/// 当前 `CudaBackend` 为骨架（张量管理可用、算子未实现），故暂为 `false`；
/// 待按 Albatross 补齐 decode 核心算子后翻转为 `true`，`detect_backend()` 即优先选择 CUDA。
pub const CUDA_READY: bool = true; // TEMP baseline check

/// 探测可用后端：CUDA 可用**且算子已就绪**优先，否则 Vulkan。
/// 返回的是「能真正跑通模型」的后端，避免骨架阶段选中 CUDA 后运行报错。
pub fn detect_backend() -> BackendKind {
    // BACKEND=cuda / BACKEND=vulkan 显式覆盖（便于后端对比/调试）。
    if let Ok(v) = std::env::var("BACKEND") {
        return match v.to_ascii_lowercase().as_str() {
            "cuda" => BackendKind::Cuda,
            "vulkan" => BackendKind::Vulkan,
            _ => {
                log::warn!("unknown BACKEND={v}, falling back to auto-detect");
                BackendKind::Vulkan
            }
        };
    }
    if CUDA_READY && cuda_available() {
        BackendKind::Cuda
    } else {
        BackendKind::Vulkan
    }
}

/// CUDA 硬件可用性（真实探测，不依赖算子实现）：驱动可加载 + `cuInit` 成功 + ≥1 设备。
pub fn cuda_available() -> bool {
    crate::backend_cuda::cuda_available()
}

/// 按 `BackendKind` 构造对应后端实例。
pub fn create_backend(kind: BackendKind) -> R<Box<dyn ComputeBackend>> {
    match kind {
        BackendKind::Vulkan => Ok(Box::new(VulkanBackend::new()?)),
        BackendKind::Cuda => Ok(Box::new(crate::backend_cuda::CudaBackend::new()?)),
    }
}

/// Vulkan 后端：封装现有 `Runtime`。
///
/// 内部维护 `TensorId → 设备张量` 映射表；算子通过 `TensorId` 查表并转调 `Runtime`。
/// 张量类型在创建时由 `TensorDtype` 决定，算子期望的类型不符时返回错误。
#[derive(Debug)]
pub struct VulkanBackend {
    rt: Runtime,
    tensors: HashMap<TensorId, VulkanTensor>,
    lens: HashMap<TensorId, usize>, // 张量长度（download_u32 需要 len 参数）
    next_id: u32,
}

/// 后端内部张量包装（统一映射表元素类型）。
/// `U32` 变体为 int8 量化算子（idx/sz）承载设备数据。
#[derive(Debug)]
enum VulkanTensor {
    F32(GpuTensor),
    F16(GpuTensor16),
    U32(GpuTensorU32),
}

impl VulkanBackend {
    /// 创建 Vulkan 后端（初始化 Vulkan 实例/设备）。
    pub fn new() -> R<Self> {
        Ok(Self {
            rt: Runtime::new()?,
            tensors: HashMap::new(),
            lens: HashMap::new(),
            next_id: 0,
        })
    }

    // —— 张量访问辅助 ——
    // 算子实现统一用它按 dtype 取张量（返回 owned 克隆，Tensor<T> 为 Arc 包装，克隆仅计数递增），
    // 规避 `get`/`get_mut` 及 `self` 整体借用与 `self.rt` 可变借用的冲突。
    fn get_f32(&self, t: TensorId, op: &str) -> R<GpuTensor> {
        match self
            .tensors
            .get(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}"))?
        {
            VulkanTensor::F32(g) => Ok(g.clone()),
            _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
        }
    }
    fn get_f16(&self, t: TensorId, op: &str) -> R<GpuTensor16> {
        match self
            .tensors
            .get(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}"))?
        {
            VulkanTensor::F16(g) => Ok(g.clone()),
            _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
        }
    }
    fn get_u32(&self, t: TensorId, op: &str) -> R<GpuTensorU32> {
        match self
            .tensors
            .get(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}"))?
        {
            VulkanTensor::U32(g) => Ok(g.clone()),
            _ => Err(format!("{op}: tensor {t:?} must be u32").into()),
        }
    }
    fn take_f32(&mut self, t: TensorId, op: &str) -> R<GpuTensor> {
        match self
            .tensors
            .remove(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}"))?
        {
            VulkanTensor::F32(g) => Ok(g),
            other => {
                self.tensors.insert(t, other);
                Err(format!("{op}: tensor {t:?} must be f32").into())
            }
        }
    }
    fn take_f16(&mut self, t: TensorId, op: &str) -> R<GpuTensor16> {
        match self
            .tensors
            .remove(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}"))?
        {
            VulkanTensor::F16(g) => Ok(g),
            other => {
                self.tensors.insert(t, other);
                Err(format!("{op}: tensor {t:?} must be f16").into())
            }
        }
    }
    fn put_f32(&mut self, t: TensorId, g: GpuTensor) {
        self.tensors.insert(t, VulkanTensor::F32(g));
    }
    fn put_f16(&mut self, t: TensorId, g: GpuTensor16) {
        self.tensors.insert(t, VulkanTensor::F16(g));
    }

    // —— 量化句柄解析 ——
    /// 从平台无关 `Int8Handle` 组装出 Runtime 所需的 `GpuTensorInt8`。
    fn int8_ref(&self, a: &Int8Handle, op: &str) -> R<GpuTensorInt8> {
        let idx = self.get_u32(a.idx, op)?;
        let sz = self.get_u32(a.sz, op)?;
        Ok(GpuTensorInt8 {
            idx,
            sz,
            m: a.m,
            k: a.k,
        })
    }
}

impl ComputeBackend for VulkanBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn create_tensor(&mut self, len: usize, dtype: TensorDtype) -> R<TensorId> {
        let id = TensorId(self.next_id);
        self.next_id += 1;
        let t = match dtype {
            TensorDtype::F32 => VulkanTensor::F32(self.rt.create_tensor(len)?),
            TensorDtype::F16 => VulkanTensor::F16(self.rt.create_tensor_f16(len)?),
            TensorDtype::U32 => VulkanTensor::U32(self.rt.create_tensor_u32(len)?),
        };
        self.tensors.insert(id, t);
        self.lens.insert(id, len);
        Ok(id)
    }

    fn upload(&self, t: TensorId, data: &[f32]) -> R<()> {
        match self.tensors.get(&t).ok_or("upload: unknown tensor")? {
            VulkanTensor::F32(g) => self.rt.upload(g, data),
            VulkanTensor::F16(g) => self.rt.upload_f16(g, data),
            VulkanTensor::U32(_) => Err("upload: u32 tensor requires u32 data".into()),
        }
    }

    fn upload_u32(&self, t: TensorId, data: &[u32]) -> R<()> {
        match self.tensors.get(&t).ok_or("upload_u32: unknown tensor")? {
            VulkanTensor::U32(g) => self.rt.upload_u32(g, data),
            _ => Err("upload_u32: t must be u32".into()),
        }
    }

    fn download(&self, t: TensorId) -> R<Vec<f32>> {
        match self.tensors.get(&t).ok_or("download: unknown tensor")? {
            VulkanTensor::F32(g) => self.rt.download(g),
            VulkanTensor::F16(g) => self.rt.download_f16(g),
            VulkanTensor::U32(_) => Err("download: u32 tensor unsupported here".into()),
        }
    }

    fn download_u32(&self, t: TensorId) -> R<Vec<u32>> {
        let len = *self
            .lens
            .get(&t)
            .ok_or("download_u32: unknown tensor len")?;
        match self.tensors.get(&t).ok_or("download_u32: unknown tensor")? {
            VulkanTensor::U32(g) => self.rt.download_u32(g, len),
            _ => Err("download_u32: t must be u32".into()),
        }
    }

    fn begin_batch(&mut self) -> R<()> {
        self.rt.begin_batch()
    }

    fn end_batch(&mut self) -> R<()> {
        self.rt.end_batch()
    }

    fn gemv_f16(
        &mut self,
        w: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        n: usize,
    ) -> R<()> {
        // 输出张量 y 需要可变引用：从 map 取出为 owned，执行后放回。
        let mut y_owned = match self.tensors.remove(&y).ok_or("gemv_f16: unknown y")? {
            VulkanTensor::F32(g) => g,
            other => {
                self.tensors.insert(y, other);
                return Err("gemv_f16: y must be f32".into());
            }
        };
        let w_g = self.get_f16(w, "gemv_f16")?;
        let x_g = self.get_f32(x, "gemv_f16")?;
        let res = self.rt.gemv_f16(&w_g, &x_g, &mut y_owned, m, k, n);
        self.tensors.insert(y, VulkanTensor::F32(y_owned));
        res
    }

    fn norm(
        &mut self,
        x: TensorId,
        gamma: TensorId,
        beta: TensorId,
        y: TensorId,
        c: usize,
        h: usize,
        eps: f32,
        rows: usize,
    ) -> R<()> {
        // 输出张量 y 需要可变引用：从 map 取出为 owned，执行后放回。
        let mut y_owned = match self.tensors.remove(&y).ok_or("norm: unknown y")? {
            VulkanTensor::F32(g) => g,
            other => {
                self.tensors.insert(y, other);
                return Err("norm: y must be f32".into());
            }
        };
        let x_g = self.get_f32(x, "norm")?;
        let gamma_g = self.get_f32(gamma, "norm")?;
        let beta_g = self.get_f32(beta, "norm")?;
        let res = self
            .rt
            .norm(&x_g, &gamma_g, &beta_g, &mut y_owned, c, h, eps, rows);
        self.tensors.insert(y, VulkanTensor::F32(y_owned));
        res
    }

    fn store_token_host(&self, tok: TensorId, token: u32) -> R<()> {
        let tok = self.get_f32(tok, "store_token_host")?;
        self.rt.store_token_host(&tok, token)
    }

    fn gather_row_device_f16(
        &mut self,
        src: TensorId,
        dst: TensorId,
        tok: TensorId,
        c: usize,
    ) -> R<()> {
        let mut dst_o = self.take_f32(dst, "gather_row_device_f16")?;
        let src_g = self.get_f16(src, "gather_row_device_f16")?;
        let tok_g = self.get_f32(tok, "gather_row_device_f16")?;
        let res = self.rt.gather_row_device_f16(&src_g, &mut dst_o, &tok_g, c);
        self.put_f32(dst, dst_o);
        res
    }

    fn copy_device_f16(&mut self, src: TensorId, dst: TensorId) -> R<()> {
        let mut dst_o = self.take_f16(dst, "copy_device_f16")?;
        let src_g = self.get_f16(src, "copy_device_f16")?;
        let res = self.rt.copy_device_f16(&src_g, &mut dst_o);
        self.put_f16(dst, dst_o);
        res
    }

    fn norm_lerp6(
        &mut self,
        x: TensorId,
        state: TensorId,
        gamma: TensorId,
        beta: TensorId,
        x_r: TensorId,
        x_w: TensorId,
        x_k: TensorId,
        x_v: TensorId,
        x_a: TensorId,
        x_g: TensorId,
        o_r: TensorId,
        o_w: TensorId,
        o_k: TensorId,
        o_v: TensorId,
        o_a: TensorId,
        o_g: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let mut s_o = self.take_f32(state, "norm_lerp6")?;
        let mut or_o = self.take_f32(o_r, "norm_lerp6")?;
        let mut ow_o = self.take_f32(o_w, "norm_lerp6")?;
        let mut ok_o = self.take_f32(o_k, "norm_lerp6")?;
        let mut ov_o = self.take_f32(o_v, "norm_lerp6")?;
        let mut oa_o = self.take_f32(o_a, "norm_lerp6")?;
        let mut og_o = self.take_f32(o_g, "norm_lerp6")?;
        let res = {
            let xin_g = self.get_f32(x, "norm_lerp6")?;
            let gamma_g = self.get_f32(gamma, "norm_lerp6")?;
            let beta_g = self.get_f32(beta, "norm_lerp6")?;
            let xr_g = self.get_f32(x_r, "norm_lerp6")?;
            let xw_g = self.get_f32(x_w, "norm_lerp6")?;
            let xk_g = self.get_f32(x_k, "norm_lerp6")?;
            let xv_g = self.get_f32(x_v, "norm_lerp6")?;
            let xa_g = self.get_f32(x_a, "norm_lerp6")?;
            let xg_g = self.get_f32(x_g, "norm_lerp6")?;
            self.rt.norm_lerp6(
                &xin_g, &mut s_o, &gamma_g, &beta_g, &xr_g, &xw_g, &xk_g, &xv_g, &xa_g, &xg_g,
                &mut or_o, &mut ow_o, &mut ok_o, &mut ov_o, &mut oa_o, &mut og_o, c, eps,
            )
        };
        self.put_f32(state, s_o);
        self.put_f32(o_r, or_o);
        self.put_f32(o_w, ow_o);
        self.put_f32(o_k, ok_o);
        self.put_f32(o_v, ov_o);
        self.put_f32(o_a, oa_o);
        self.put_f32(o_g, og_o);
        res
    }

    fn cmix_norm_lerp(
        &mut self,
        x: TensorId,
        state: TensorId,
        gamma: TensorId,
        beta: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let mut s_o = self.take_f32(state, "cmix_norm_lerp")?;
        let mut xb_o = self.take_f32(out_xb, "cmix_norm_lerp")?;
        let res = {
            let x_g = self.get_f32(x, "cmix_norm_lerp")?;
            let gamma_g = self.get_f32(gamma, "cmix_norm_lerp")?;
            let beta_g = self.get_f32(beta, "cmix_norm_lerp")?;
            let coeff_g = self.get_f32(coeff, "cmix_norm_lerp")?;
            self.rt.cmix_norm_lerp(
                &x_g, &mut s_o, &gamma_g, &beta_g, &coeff_g, &mut xb_o, c, eps,
            )
        };
        self.put_f32(state, s_o);
        self.put_f32(out_xb, xb_o);
        res
    }

    fn fuse_ka_dplr_norm(
        &mut self,
        s: TensorId,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        r: TensorId,
        v: TensorId,
        w: TensorId,
        gamma: TensorId,
        beta: TensorId,
        r_k: TensorId,
        k_mod: TensorId,
        y: TensorId,
        y_norm: TensorId,
        h: usize,
        n: usize,
        eps: f32,
        gn_eps: f32,
    ) -> R<()> {
        let mut s_o = self.take_f32(s, "fuse_ka_dplr_norm")?;
        let mut km_o = self.take_f32(k_mod, "fuse_ka_dplr_norm")?;
        let mut y_o = self.take_f32(y, "fuse_ka_dplr_norm")?;
        let mut yn_o = self.take_f32(y_norm, "fuse_ka_dplr_norm")?;
        let res = {
            let k_g = self.get_f32(k, "fuse_ka_dplr_norm")?;
            let k_k_g = self.get_f32(k_k, "fuse_ka_dplr_norm")?;
            let a_g = self.get_f16(a, "fuse_ka_dplr_norm")?;
            let k_a_g = self.get_f32(k_a, "fuse_ka_dplr_norm")?;
            let r_g = self.get_f32(r, "fuse_ka_dplr_norm")?;
            let v_g = self.get_f16(v, "fuse_ka_dplr_norm")?;
            let w_g = self.get_f16(w, "fuse_ka_dplr_norm")?;
            let gamma_g = self.get_f32(gamma, "fuse_ka_dplr_norm")?;
            let beta_g = self.get_f32(beta, "fuse_ka_dplr_norm")?;
            let r_k_g = self.get_f32(r_k, "fuse_ka_dplr_norm")?;
            self.rt.fuse_ka_dplr_norm(
                &mut s_o, &k_g, &k_k_g, &a_g, &k_a_g, &r_g, &v_g, &w_g, &gamma_g, &beta_g, &r_k_g,
                &mut km_o, &mut y_o, &mut yn_o, h, n, eps, gn_eps,
            )
        };
        self.put_f32(s, s_o);
        self.put_f32(k_mod, km_o);
        self.put_f32(y, y_o);
        self.put_f32(y_norm, yn_o);
        res
    }

    fn gemv_rkv_stage1(
        &mut self,
        r: TensorId,
        k: TensorId,
        v: TensorId,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        out_r: TensorId,
        out_k: TensorId,
        out_v: TensorId,
        out_vm: TensorId,
        out_wm: TensorId,
        out_am: TensorId,
        out_gm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        let mut or_o = self.take_f32(out_r, "gemv_rkv_stage1")?;
        let mut ok_o = self.take_f32(out_k, "gemv_rkv_stage1")?;
        let mut ov_o = self.take_f16(out_v, "gemv_rkv_stage1")?;
        let mut ovm_o = self.take_f32(out_vm, "gemv_rkv_stage1")?;
        let mut owm_o = self.take_f32(out_wm, "gemv_rkv_stage1")?;
        let mut oam_o = self.take_f32(out_am, "gemv_rkv_stage1")?;
        let mut ogm_o = self.take_f32(out_gm, "gemv_rkv_stage1")?;
        let res = {
            let r_g = self.get_f16(r, "gemv_rkv_stage1")?;
            let k_g = self.get_f16(k, "gemv_rkv_stage1")?;
            let v_g = self.get_f16(v, "gemv_rkv_stage1")?;
            let v1_g = self.get_f32(v1, "gemv_rkv_stage1")?;
            let w1_g = self.get_f32(w1, "gemv_rkv_stage1")?;
            let a1_g = self.get_f32(a1, "gemv_rkv_stage1")?;
            let g1_g = self.get_f32(g1, "gemv_rkv_stage1")?;
            let xr_g = self.get_f32(xr, "gemv_rkv_stage1")?;
            let xk_g = self.get_f32(xk, "gemv_rkv_stage1")?;
            let xv_g = self.get_f32(xv, "gemv_rkv_stage1")?;
            let xw_g = self.get_f32(xw, "gemv_rkv_stage1")?;
            let xa_g = self.get_f32(xa, "gemv_rkv_stage1")?;
            let xg_g = self.get_f32(xg, "gemv_rkv_stage1")?;
            self.rt.gemv_rkv_stage1(
                &r_g, &k_g, &v_g, &v1_g, &w1_g, &a1_g, &g1_g, &xr_g, &xk_g, &xv_g, &xw_g, &xa_g,
                &xg_g, &mut or_o, &mut ok_o, &mut ov_o, &mut ovm_o, &mut owm_o, &mut oam_o,
                &mut ogm_o, c, vm, wm, am, gm,
            )
        };
        self.put_f32(out_r, or_o);
        self.put_f32(out_k, ok_o);
        self.put_f16(out_v, ov_o);
        self.put_f32(out_vm, ovm_o);
        self.put_f32(out_wm, owm_o);
        self.put_f32(out_am, oam_o);
        self.put_f32(out_gm, ogm_o);
        res
    }

    fn gemv_int8_rkv_stage1(
        &mut self,
        r: &Int8Handle,
        k: &Int8Handle,
        v: &Int8Handle,
        v1: TensorId,
        w1: TensorId,
        a1: TensorId,
        g1: TensorId,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        xw: TensorId,
        xa: TensorId,
        xg: TensorId,
        out_r: TensorId,
        out_k: TensorId,
        out_v: TensorId,
        out_vm: TensorId,
        out_wm: TensorId,
        out_am: TensorId,
        out_gm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        let mut or_o = self.take_f32(out_r, "gemv_int8_rkv_stage1")?;
        let mut ok_o = self.take_f32(out_k, "gemv_int8_rkv_stage1")?;
        let mut ov_o = self.take_f16(out_v, "gemv_int8_rkv_stage1")?;
        let mut ovm_o = self.take_f32(out_vm, "gemv_int8_rkv_stage1")?;
        let mut owm_o = self.take_f32(out_wm, "gemv_int8_rkv_stage1")?;
        let mut oam_o = self.take_f32(out_am, "gemv_int8_rkv_stage1")?;
        let mut ogm_o = self.take_f32(out_gm, "gemv_int8_rkv_stage1")?;
        let res = {
            let ra8 = self.int8_ref(r, "gemv_int8_rkv_stage1")?;
            let ka8 = self.int8_ref(k, "gemv_int8_rkv_stage1")?;
            let va8 = self.int8_ref(v, "gemv_int8_rkv_stage1")?;
            let v1_g = self.get_f32(v1, "gemv_int8_rkv_stage1")?;
            let w1_g = self.get_f32(w1, "gemv_int8_rkv_stage1")?;
            let a1_g = self.get_f32(a1, "gemv_int8_rkv_stage1")?;
            let g1_g = self.get_f32(g1, "gemv_int8_rkv_stage1")?;
            let xr_g = self.get_f32(xr, "gemv_int8_rkv_stage1")?;
            let xk_g = self.get_f32(xk, "gemv_int8_rkv_stage1")?;
            let xv_g = self.get_f32(xv, "gemv_int8_rkv_stage1")?;
            let xw_g = self.get_f32(xw, "gemv_int8_rkv_stage1")?;
            let xa_g = self.get_f32(xa, "gemv_int8_rkv_stage1")?;
            let xg_g = self.get_f32(xg, "gemv_int8_rkv_stage1")?;
            self.rt.gemv_int8_rkv_stage1(
                &ra8, &ka8, &va8, &v1_g, &w1_g, &a1_g, &g1_g, &xr_g, &xk_g, &xv_g, &xw_g, &xa_g,
                &xg_g, &mut or_o, &mut ok_o, &mut ov_o, &mut ovm_o, &mut owm_o, &mut oam_o,
                &mut ogm_o, c, vm, wm, am, gm,
            )
        };
        self.put_f32(out_r, or_o);
        self.put_f32(out_k, ok_o);
        self.put_f16(out_v, ov_o);
        self.put_f32(out_vm, ovm_o);
        self.put_f32(out_wm, owm_o);
        self.put_f32(out_am, oam_o);
        self.put_f32(out_gm, ogm_o);
        res
    }

    fn gemv_lowrank_chain4(
        &mut self,
        w2: TensorId,
        a2: TensorId,
        v2: TensorId,
        g2: TensorId,
        w_mid: TensorId,
        a_mid: TensorId,
        v_mid: TensorId,
        g_mid: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        v_first: TensorId,
        out_w: TensorId,
        out_a: TensorId,
        out_v: TensorId,
        out_g: TensorId,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
    ) -> R<()> {
        let mut ow_o = self.take_f16(out_w, "gemv_lowrank_chain4")?;
        let mut oa_o = self.take_f16(out_a, "gemv_lowrank_chain4")?;
        let mut ov_o = self.take_f16(out_v, "gemv_lowrank_chain4")?;
        let mut og_o = self.take_f16(out_g, "gemv_lowrank_chain4")?;
        let res = {
            let w2_g = self.get_f32(w2, "gemv_lowrank_chain4")?;
            let a2_g = self.get_f32(a2, "gemv_lowrank_chain4")?;
            let v2_g = self.get_f32(v2, "gemv_lowrank_chain4")?;
            let g2_g = self.get_f32(g2, "gemv_lowrank_chain4")?;
            let wm_g = self.get_f32(w_mid, "gemv_lowrank_chain4")?;
            let am_g = self.get_f32(a_mid, "gemv_lowrank_chain4")?;
            let vm_g = self.get_f32(v_mid, "gemv_lowrank_chain4")?;
            let gm_g = self.get_f32(g_mid, "gemv_lowrank_chain4")?;
            let w0_g = self.get_f32(w0, "gemv_lowrank_chain4")?;
            let a0_g = self.get_f32(a0, "gemv_lowrank_chain4")?;
            let v0_g = self.get_f32(v0, "gemv_lowrank_chain4")?;
            let scale_g = self.get_f32(scale, "gemv_lowrank_chain4")?;
            let vf_g = self.get_f16(v_first, "gemv_lowrank_chain4")?;
            self.rt.gemv_lowrank_chain4(
                &w2_g, &a2_g, &v2_g, &g2_g, &wm_g, &am_g, &vm_g, &gm_g, &w0_g, &a0_g, &v0_g,
                &scale_g, &vf_g, &mut ow_o, &mut oa_o, &mut ov_o, &mut og_o, m, kw, ka, kv, kg,
            )
        };
        self.put_f16(out_w, ow_o);
        self.put_f16(out_a, oa_o);
        self.put_f16(out_v, ov_o);
        self.put_f16(out_g, og_o);
        res
    }

    fn gemv_f16_relu2(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_f16_relu2")?;
        let res = {
            let a_g = self.get_f16(a, "gemv_f16_relu2")?;
            let x_g = self.get_f32(x, "gemv_f16_relu2")?;
            self.rt.gemv_f16_relu2(&a_g, &x_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn gemv_int8_relu2(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_int8_relu2")?;
        let res = {
            let a_g = self.int8_ref(a, "gemv_int8_relu2")?;
            let x_g = self.get_f32(x, "gemv_int8_relu2")?;
            self.rt.gemv_int8_relu2(&a_g, &x_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn gemv_f16_mul_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_f16_mul_add")?;
        let res = {
            let a_g = self.get_f16(a, "gemv_f16_mul_add")?;
            let x_g = self.get_f32(x, "gemv_f16_mul_add")?;
            let g_g = self.get_f16(g, "gemv_f16_mul_add")?;
            self.rt
                .gemv_f16_mul_add(&a_g, &x_g, &g_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn gemv_int8_mul_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_int8_mul_add")?;
        let res = {
            let a_g = self.int8_ref(a, "gemv_int8_mul_add")?;
            let x_g = self.get_f32(x, "gemv_int8_mul_add")?;
            let g_g = self.get_f16(g, "gemv_int8_mul_add")?;
            self.rt
                .gemv_int8_mul_add(&a_g, &x_g, &g_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn gemv_f16_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_f16_add")?;
        let res = {
            let a_g = self.get_f16(a, "gemv_f16_add")?;
            let x_g = self.get_f32(x, "gemv_f16_add")?;
            self.rt.gemv_f16_add(&a_g, &x_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn ffn_value_sparse_add(
        &mut self,
        _value_w16: Option<TensorId>,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
    ) -> R<()> {
        let mut x_o = self.take_f32(x, "ffn_value_sparse_add")?;
        let res = {
            let vt_g = self.get_f16(value_tiled, "ffn_value_sparse_add")?;
            let r2_g = self.get_f32(r2, "ffn_value_sparse_add")?;
            self.rt.ffn_value_sparse_add(&vt_g, &r2_g, &mut x_o, c, fh)
        };
        self.put_f32(x, x_o);
        res
    }

    fn supports_sparse_ffn(&self) -> bool {
        self.rt.supports_sparse_ffn()
    }

    fn gemv_int8_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_int8_add")?;
        let res = {
            let a_g = self.int8_ref(a, "gemv_int8_add")?;
            let x_g = self.get_f32(x, "gemv_int8_add")?;
            self.rt.gemv_int8_add(&a_g, &x_g, &mut y_o, m, k, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn gemv_int8_plain(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        // 直接 int8 gemv 覆盖写（gemv_int8.comp，grid.x = m/4）。
        // 不再走「整行反量化到 fp16 scratch」路径：head(vocab=65536) 反量化的 dispatch
        // grid.x=163840 超过 Vulkan maxComputeWorkGroupCount(65535) 会 ERROR_DEVICE_LOST。
        let mut y_owned = match self
            .tensors
            .remove(&y)
            .ok_or("gemv_int8_plain: unknown y")?
        {
            VulkanTensor::F32(g) => g,
            other => {
                self.tensors.insert(y, other);
                return Err("gemv_int8_plain: y must be f32".into());
            }
        };
        let a_g = self.int8_ref(a, "gemv_int8_plain")?;
        let x_g = self.get_f32(x, "gemv_int8_plain")?;
        let res = self
            .rt
            .gemv_int8_plain(&a_g, &x_g, &mut y_owned, m, k, batch);
        self.tensors.insert(y, VulkanTensor::F32(y_owned));
        res
    }

    fn argmax(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        let mut tok_o = self.take_f32(token, "argmax")?;
        let logits_g = self.get_f32(logits, "argmax")?;
        let res = self.rt.argmax(&logits_g, &mut tok_o, n);
        self.put_f32(token, tok_o);
        res
    }

    fn sample(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        history: &[u32],
    ) -> R<()> {
        let mut tok_o = self.take_f32(token, "sample")?;
        let logits_g = self.get_f32(logits, "sample")?;
        let res = self.rt.sample(
            &logits_g,
            &mut tok_o,
            n,
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            history,
        );
        self.put_f32(token, tok_o);
        res
    }

    fn clear_cache(&mut self) {
        self.rt.clear_cache();
    }

    fn drop_host(&mut self, t: TensorId) {
        match self.tensors.get_mut(&t) {
            Some(VulkanTensor::F32(g)) => self.rt.drop_host(g),
            Some(VulkanTensor::F16(g)) => self.rt.drop_host_f16(g),
            Some(VulkanTensor::U32(g)) => self.rt.drop_host_u32(g),
            None => {}
        }
    }

    fn free_tensor(&mut self, t: TensorId) {
        // 移除注册表条目；GpuTensor 为 Arc 包装，末引用 Drop 时释放设备缓冲。
        self.tensors.remove(&t);
        self.lens.remove(&t);
    }

    fn copy_device(&mut self, src: TensorId, dst: TensorId) -> R<()> {
        let mut dst_o = self.take_f32(dst, "copy_device")?;
        let src_g = self.get_f32(src, "copy_device")?;
        let res = self.rt.copy_device(&src_g, &mut dst_o);
        self.put_f32(dst, dst_o);
        res
    }

    fn copy_token(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        stride: usize,
        token: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "copy_token")?;
        let x_g = self.get_f32(x, "copy_token")?;
        let res = self.rt.copy_token(&x_g, &mut y_o, c, stride, token);
        self.put_f32(y, y_o);
        res
    }

    fn gemm(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let mut c_o = self.take_f32(c, "gemm")?;
        let a_g = self.get_f16(a, "gemm")?;
        let b_g = self.get_f16(b, "gemm")?;
        let res = self.rt.gemm(&a_g, &b_g, &mut c_o, m, n, k);
        self.put_f32(c, c_o);
        res
    }

    fn gemm_bias(
        &mut self,
        a: TensorId,
        b: TensorId,
        bias: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let mut c_o = self.take_f32(c, "gemm_bias")?;
        let a_g = self.get_f16(a, "gemm_bias")?;
        let b_g = self.get_f16(b, "gemm_bias")?;
        let bias_g = self.get_f32(bias, "gemm_bias")?;
        let res = self.rt.gemm_bias(&a_g, &b_g, &bias_g, &mut c_o, m, n, k);
        self.put_f32(c, c_o);
        res
    }

    fn gemm_add(
        &mut self,
        a: TensorId,
        b: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemm_add")?;
        let a_g = self.get_f16(a, "gemm_add")?;
        let b_g = self.get_f16(b, "gemm_add")?;
        let x_g = self.get_f32(x, "gemm_add")?;
        let res = self.rt.gemm_add(&a_g, &b_g, &x_g, &mut y_o, m, n, k);
        self.put_f32(y, y_o);
        res
    }

    fn to_f16(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let mut y_o = self.take_f16(y, "to_f16")?;
        let x_g = self.get_f32(x, "to_f16")?;
        let res = self
            .rt
            .to_f16(&x_g, &mut y_o, c, t, m_pad, x_stride, y_stride);
        self.put_f16(y, y_o);
        res
    }

    fn to_f16_triple(
        &mut self,
        xr: TensorId,
        xk: TensorId,
        xv: TensorId,
        yr: TensorId,
        yk: TensorId,
        yv: TensorId,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let mut yr_o = self.take_f16(yr, "to_f16_triple")?;
        let mut yk_o = self.take_f16(yk, "to_f16_triple")?;
        let mut yv_o = self.take_f16(yv, "to_f16_triple")?;
        let res = {
            let xr_g = self.get_f32(xr, "to_f16_triple")?;
            let xk_g = self.get_f32(xk, "to_f16_triple")?;
            let xv_g = self.get_f32(xv, "to_f16_triple")?;
            self.rt.to_f16_triple(
                &xr_g, &xk_g, &xv_g, &mut yr_o, &mut yk_o, &mut yv_o, c, t, m_pad, x_stride,
                y_stride,
            )
        };
        self.put_f16(yr, yr_o);
        self.put_f16(yk, yk_o);
        self.put_f16(yv, yv_o);
        res
    }

    fn dequant_int8_to_f16(&mut self, a: &Int8Handle, out: TensorId, m: usize, k: usize) -> R<()> {
        let out_o = self.take_f16(out, "dequant_int8_to_f16")?;
        let a8 = self.int8_ref(a, "dequant_int8_to_f16")?;
        let res = self.rt.dequant_int8_to_f16(&a8, &out_o, m, k);
        self.put_f16(out, out_o);
        res
    }

    fn elementwise_sigmoid(&mut self, a: TensorId, y: TensorId, c: usize, batch: usize) -> R<()> {
        let mut y_o = self.take_f32(y, "elementwise_sigmoid")?;
        let a_g = self.get_f32(a, "elementwise_sigmoid")?;
        let res = self.rt.elementwise_sigmoid(&a_g, &mut y_o, c, batch);
        self.put_f32(y, y_o);
        res
    }

    fn elementwise_sigmoid_inplace(&mut self, y: TensorId, c: usize, batch: usize) -> R<()> {
        let mut y_o = self.take_f32(y, "elementwise_sigmoid_inplace")?;
        let res = self.rt.elementwise_sigmoid_inplace(&mut y_o, c, batch);
        self.put_f32(y, y_o);
        res
    }

    fn fuse_ka(
        &mut self,
        k: TensorId,
        k_k: TensorId,
        a: TensorId,
        k_a: TensorId,
        k_mod: TensorId,
        kk_l2: TensorId,
        b: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let mut km_o = self.take_f32(k_mod, "fuse_ka")?;
        let mut kk_o = self.take_f32(kk_l2, "fuse_ka")?;
        let mut b_o = self.take_f32(b, "fuse_ka")?;
        let res = {
            let k_g = self.get_f32(k, "fuse_ka")?;
            let k_k_g = self.get_f32(k_k, "fuse_ka")?;
            let a_g = self.get_f32(a, "fuse_ka")?;
            let k_a_g = self.get_f32(k_a, "fuse_ka")?;
            self.rt.fuse_ka(
                &k_g, &k_k_g, &a_g, &k_a_g, &mut km_o, &mut kk_o, &mut b_o, h, n, batch,
            )
        };
        self.put_f32(k_mod, km_o);
        self.put_f32(kk_l2, kk_o);
        self.put_f32(b, b_o);
        res
    }

    fn sum_rk_rk(
        &mut self,
        r: TensorId,
        k_mod: TensorId,
        r_k: TensorId,
        v: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "sum_rk_rk")?;
        let res = {
            let r_g = self.get_f32(r, "sum_rk_rk")?;
            let km_g = self.get_f32(k_mod, "sum_rk_rk")?;
            let rk_g = self.get_f32(r_k, "sum_rk_rk")?;
            let v_g = self.get_f32(v, "sum_rk_rk")?;
            self.rt
                .sum_rk_rk(&r_g, &km_g, &rk_g, &v_g, &mut y_o, h, n, batch)
        };
        self.put_f32(y, y_o);
        res
    }

    fn seq_shift(
        &mut self,
        x: TensorId,
        state: TensorId,
        tm: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "seq_shift")?;
        let res = {
            let x_g = self.get_f32(x, "seq_shift")?;
            let state_g = self.get_f32(state, "seq_shift")?;
            let tm_g = self.get_f32(tm, "seq_shift")?;
            self.rt
                .seq_shift(&x_g, &state_g, &tm_g, &mut y_o, c, t, stride_x, stride_y)
        };
        self.put_f32(y, y_o);
        res
    }

    fn dplr_seq(
        &mut self,
        s: TensorId,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
    ) -> R<()> {
        let mut s_o = self.take_f32(s, "dplr_seq")?;
        let mut y_o = self.take_f32(y, "dplr_seq")?;
        let res = {
            let r_g = self.get_f32(r, "dplr_seq")?;
            let w_g = self.get_f32(w, "dplr_seq")?;
            let k_g = self.get_f32(k, "dplr_seq")?;
            let v_g = self.get_f32(v, "dplr_seq")?;
            let a_g = self.get_f32(a, "dplr_seq")?;
            let b_g = self.get_f32(b, "dplr_seq")?;
            self.rt.dplr_seq(
                &mut s_o, &r_g, &w_g, &k_g, &v_g, &a_g, &b_g, &mut y_o, h, n, t, c,
            )
        };
        self.put_f32(s, s_o);
        self.put_f32(y, y_o);
        res
    }

    fn gemm_relu2(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let mut c_o = self.take_f32(c, "gemm_relu2")?;
        let a_g = self.get_f16(a, "gemm_relu2")?;
        let b_g = self.get_f16(b, "gemm_relu2")?;
        let res = self.rt.gemm_relu2(&a_g, &b_g, &mut c_o, m, n, k);
        self.put_f32(c, c_o);
        res
    }

    fn gemm_tanh(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let mut c_o = self.take_f32(c, "gemm_tanh")?;
        let a_g = self.get_f16(a, "gemm_tanh")?;
        let b_g = self.get_f16(b, "gemm_tanh")?;
        let res = self.rt.gemm_tanh(&a_g, &b_g, &mut c_o, m, n, k);
        self.put_f32(c, c_o);
        res
    }

    fn elementwise_scale_exp(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "elementwise_scale_exp")?;
        let a_g = self.get_f32(a, "elementwise_scale_exp")?;
        let b_g = self.get_f32(b, "elementwise_scale_exp")?;
        let res = self
            .rt
            .elementwise_scale_exp(&a_g, &b_g, &mut y_o, c, batch);
        self.put_f32(y, y_o);
        res
    }

    fn elementwise_mul(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "elementwise_mul")?;
        let a_g = self.get_f32(a, "elementwise_mul")?;
        let b_g = self.get_f32(b, "elementwise_mul")?;
        let res = self.rt.elementwise_mul(&a_g, &b_g, &mut y_o, c, batch);
        self.put_f32(y, y_o);
        res
    }

    fn v_first_lerp(
        &mut self,
        v: TensorId,
        gate: TensorId,
        v_first: TensorId,
        c: usize,
        t: usize,
        stride: usize,
    ) -> R<()> {
        let v_o = self.take_f32(v, "v_first_lerp")?;
        let gate_g = self.get_f32(gate, "v_first_lerp")?;
        let vf_g = self.get_f32(v_first, "v_first_lerp")?;
        let res = self.rt.v_first_lerp(&v_o, &gate_g, &vf_g, c, t, stride);
        self.put_f32(v, v_o);
        res
    }

    fn gemv_seq(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        x_stride: usize,
        y_stride: usize,
        batch: usize,
    ) -> R<()> {
        let mut y_o = self.take_f32(y, "gemv_seq")?;
        let a_g = self.get_f32(a, "gemv_seq")?;
        let x_g = self.get_f32(x, "gemv_seq")?;
        let res = self
            .rt
            .gemv_seq(&a_g, &x_g, &mut y_o, m, k, x_stride, y_stride, batch);
        self.put_f32(y, y_o);
        res
    }

    fn store_sampler_host(
        &self,
        sampler: TensorId,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        let sampler_g = self.get_f32(sampler, "store_sampler_host")?;
        self.rt.store_sampler_host(
            &sampler_g,
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            hist_len,
        )
    }

    fn sample_into_host_seeded(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        hist: TensorId,
    ) -> R<()> {
        let logits_g = self.get_f32(logits, "sample_into_host_seeded")?;
        let tok_g = self.get_f32(token, "sample_into_host_seeded")?;
        let tok_host = tok_g
            .host
            .as_ref()
            .ok_or("sample_into_host_seeded: token host dropped")?;
        let temp_g = self.get_f32(temp, "sample_into_host_seeded")?;
        let mask_g = self.get_f32(mask, "sample_into_host_seeded")?;
        let counter_g = self.get_u32(counter, "sample_into_host_seeded")?;
        let sampler_g = self.get_f32(sampler, "sample_into_host_seeded")?;
        let hist_g = self.get_f32(hist, "sample_into_host_seeded")?;
        self.rt.sample_into_host_seeded(
            &logits_g, tok_host, n, &temp_g, &mask_g, &counter_g, &sampler_g, &hist_g,
        )
    }

    fn record_token(&mut self, in_tok: TensorId, out_seq: TensorId, cnt: TensorId) -> R<()> {
        let in_tok_g = self.get_f32(in_tok, "record_token")?;
        let in_tok_host = in_tok_g
            .host
            .as_ref()
            .ok_or("record_token: in_tok host dropped")?;
        let out_g = self.get_f32(out_seq, "record_token")?;
        let mut cnt_o = self.take_f32(cnt, "record_token")?;
        let res = self.rt.record_token(in_tok_host, &out_g, &mut cnt_o);
        self.put_f32(cnt, cnt_o);
        res
    }

    fn argmax_into_host(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        let logits_g = self.get_f32(logits, "argmax_into_host")?;
        let tok_g = self.get_f32(token, "argmax_into_host")?;
        let tok_host = tok_g
            .host
            .as_ref()
            .ok_or("argmax_into_host: token host dropped")?;
        self.rt.argmax_into_host(&logits_g, tok_host, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ComputeBackend;

    /// 验证抽象层 gemv_f16 算子正确性：y = x @ A，A 全 1 → y[m] = K。
    /// 走 `ComputeBackend` trait（VulkanBackend），确认抽象层可用。
    #[test]
    fn backend_gemv_f16_ones() {
        let mut b = VulkanBackend::new().expect("create vulkan backend");
        let (m, n, k) = (1024usize, 1usize, 256usize);

        let w = b.create_tensor(m * k, TensorDtype::F16).expect("create w");
        let x = b.create_tensor(k * n, TensorDtype::F32).expect("create x");
        let y = b.create_tensor(m * n, TensorDtype::F32).expect("create y");

        b.upload(w, &vec![1.0f32; m * k]).unwrap();
        b.upload(x, &vec![1.0f32; k * n]).unwrap();

        b.begin_batch().unwrap();
        b.gemv_f16(w, x, y, m, k, n).unwrap();
        b.end_batch().unwrap();

        let got = b.download(y).unwrap();
        let mut max_diff = 0.0f32;
        for &g in got.iter() {
            max_diff = max_diff.max((g - k as f32).abs());
        }
        log::info!("backend gemv_f16 ones K=32 max_abs_diff: {max_diff:.6}");
        assert!(max_diff < 1e-2, "ones mismatch, max_abs_diff={max_diff}");
    }

    /// 验证抽象层 norm 算子正确性：y = (x-mean)/std * gamma + beta。
    /// 走 `ComputeBackend` trait，确认抽象层可用。
    #[test]
    fn backend_norm_matches_reference() {
        let mut b = VulkanBackend::new().expect("create vulkan backend");
        let (c, h, rows) = (256usize, 1usize, 4usize);
        let n = c * h * rows;
        let eps = 1e-5f32;

        let x = b.create_tensor(n, TensorDtype::F32).expect("create x");
        let gamma = b.create_tensor(c, TensorDtype::F32).expect("gamma");
        let beta = b.create_tensor(c, TensorDtype::F32).expect("beta");
        let y = b.create_tensor(n, TensorDtype::F32).expect("y");

        let xd: Vec<f32> = (0..n).map(|i| (i as f32) / 7.0 - 3.0).collect();
        let gd: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32) * 0.1).collect();
        let bd: Vec<f32> = (0..c).map(|i| -0.5 + (i as f32) * 0.05).collect();

        b.upload(x, &xd).unwrap();
        b.upload(gamma, &gd).unwrap();
        b.upload(beta, &bd).unwrap();

        b.begin_batch().unwrap();
        b.norm(x, gamma, beta, y, c, h, eps, rows).unwrap();
        b.end_batch().unwrap();

        let got = b.download(y).unwrap();

        // CPU 参考：逐 batch 行归一化
        for r in 0..rows {
            let mean: f32 = xd[r * c * h..(r + 1) * c * h].iter().sum::<f32>() / (c * h) as f32;
            let var: f32 = xd[r * c * h..(r + 1) * c * h]
                .iter()
                .map(|v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                / (c * h) as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..c * h {
                let idx = r * c * h + i;
                let expected = (xd[idx] - mean) * inv * gd[i % c] + bd[i % c];
                let diff = (got[idx] - expected).abs();
                assert!(
                    diff < 5e-2,
                    "norm mismatch row={r} i={i} got={} exp={expected}",
                    got[idx]
                );
            }
        }
        log::info!("backend norm matches reference");
    }
}
