//! Vulkan 推理 runtime 模块
//! 提供 GPU 张量管理和 RWKV-7 单 token 推理所需的算子。
//!
//! 所有算子均假设 batch=1（单 token 推理），dispatch 的 batch 维度恒为 1。

use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;

use half::f16;
use vulkanalia::prelude::v1_4::*;

use crate::vulkan::app::{App, Kernel, QUERY_POOL_SIZE, Tensor, Uniform};
use crate::vulkan::asset;
use crate::vulkan::layout::Layout;

/// 统一错误类型
pub type R<T> = Result<T, Box<dyn Error>>;

/// GPU 端张量：包含 host 可见缓冲（用于上传/下载）和 device local 缓冲（用于计算）。
/// `host` 为 Option：权重上传完成后可调用 `drop_host` 释放 host 缓冲，节省系统内存。
#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub host: Option<Tensor<f32>>,
    pub device: Tensor<f32>,
    pub len: usize,
}

/// GPU 端 fp16 张量（tensor-core GEMM 用，fp32io16 模式）。
/// 权重以 fp16 存储，激活由 fp32 经 `to_f16` 转换而来。
/// `host` 为 Option：权重上传完成后可调用 `drop_host` 释放 host 缓冲。
#[derive(Debug, Clone)]
pub struct GpuTensor16 {
    pub host: Option<Tensor<f16>>,
    pub device: Tensor<f16>,
    pub len: usize,
}

/// GPU 端 u32 张量（int8 打包索引 / scale-zero 对用）。
#[derive(Debug, Clone)]
pub struct GpuTensorU32 {
    pub host: Option<Tensor<u32>>,
    pub device: Tensor<u32>,
}

/// int8 非对称 per-group 量化权重（无 LUT，256 级均匀，近无损）：
/// `w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]`
/// - idx: [M, K/4] uint32（每 uint32 打包 4 个 uint8 权重，低位在前）
/// - sz:  [M, K/128] uint32（scale fp16 低 16 位 | zero fp16 高 16 位）
#[derive(Debug)]
pub struct GpuTensorInt8 {
    pub idx: GpuTensorU32,
    pub sz: GpuTensorU32,
    pub m: usize,
    pub k: usize,
}

/// Vulkan 推理 runtime
#[derive(Debug)]
pub struct Runtime {
    pub app: App,
    // 多 command buffer 流水线：
    //   持久 compute command buffer，整段 forward 记录所有 dispatch + device→device 拷贝，
    //   一次性 submit + wait，替代每算子一次的 submit+wait_idle 串行同步。
    cmd: vk::CommandBuffer,
    /// 是否正在批处理记录中（begin_batch opened，未 end_batch）。
    /// 只有 recording 时才允许记录 dispatch / 拷贝。
    recording: bool,
    /// 保留 kernel（其持有 descriptor set 与 uniform 资源）直到 submit 完成，
    /// 避免延迟 submit 期间资源被提前释放。
    pending: Vec<Kernel>,
    /// kernel 缓存：按 (shader, spec, n_params) 复用已创建的 pipeline + descriptor set。
    /// **不含 buffer 地址**（与旧版的关键差异）：地址只写入 uniform 池的 slot 内容，
    /// 由 DYNAMIC offset 在 cmd_bind 时指向——同形状不同层的 kernel 共享一个 pipeline，
    /// 首次 prefill 的 ~1700 次 pipeline/descriptor/uniform 创建降到 ~20 次。
    cache: HashMap<KernelKey, Kernel>,
    /// uniform 池：单个 host-visible mapped 大 buffer，所有 dispatch 的 Params 都
    /// bump 分配在其中（按 min_uniform_buffer_offset_alignment 对齐）。
    /// 池配合 UNIFORM_BUFFER_DYNAMIC descriptor：kernel 的 descriptor 绑定整个池 +
    /// 定长 range，每次 dispatch 只写 slot 内容 + cmd_bind 传 offset。
    /// 声明在 cache 之后：drop 顺序保证 cache 中的 Kernel（引用池）先释放。
    uniform_pool: Uniform,
    /// uniform 池 bump 游标（字节）。begin/end_batch 重置为 0（end_batch 已
    /// queue_wait_idle，GPU 空闲，slot 内容可安全覆盖）。
    pool_cursor: usize,
    /// uniform 池 slot 表（批内 dedup）：params → 池内偏移。
    /// decode/selfloop 每 token 的 kernel params 完全一致（buffer 地址不变），
    /// dedup 后批内 slot 数 = 不同 params 组合数（~几百），与 token 数无关——
    /// 对齐旧版按 (shader, spec, params) 缓存 uniform 的去重语义。
    /// 未 dedup 时 1000-token selfloop（~26 万 dispatch × 对齐 slot）会耗尽池。
    pool_slots: HashMap<Vec<u64>, usize>,
    /// dynamic offset 对齐（min_uniform_buffer_offset_alignment，池为 0 时取 1）。
    pool_align: usize,
    /// 本批内已写入、尚未被读取同步的缓冲（address → buffer handle）。
    /// 用于 buffer 级内存 barrier：仅同步当前 kernel 真正读取的缓冲，让无依赖的
    /// kernel 并发执行（替代每 dispatch 的全局全量 barrier，消除序列化）。
    written: HashMap<u64, vk::Buffer>,
    /// 本批内已读取过的缓冲（address → buffer handle）。
    /// 用于检测 WAR（写后读）竞争：后续 kernel 写入该缓冲前需先插入执行依赖，
    /// 确保前一个 kernel 的读取已完成。
    read: HashMap<u64, vk::Buffer>,
    /// 本批内写入过的所有缓冲（address → buffer handle），end_batch 才清空。
    /// 仅用于诊断：检测本批写入过但读取时未含 barrier 的缓冲（定位缺失同步）。
    written_batch: HashMap<u64, vk::Buffer>,
    /// 诊断：累计 record_kernel 纯 host 记录耗时（不含 GPU 执行），PROF_HOST=1 时打印。
    prof_host: std::time::Duration,
    prof_kernels: usize,
    prof_key: std::time::Duration,
    prof_cache: std::time::Duration,
    prof_bar: std::time::Duration,
    prof_disp: std::time::Duration,
    /// GPU 时间戳剖析（PROF_GPU=1）：记录每个 kernel 的 (label, begin_query, end_query, est_bytes)。
    /// est_bytes 为估算的读写字节数（用于带宽利用率 = est_bytes / 耗时 / 峰值带宽）。
    prof_gpu: bool,
    prof_gpu_entries: Vec<(String, u32, u32, u64)>,
    prof_gpu_count: u32,
}

/// 唯一标识一个已创建的 kernel（pipeline）。
/// `n_params` 为 Params 的 u64 字段数（决定 DYNAMIC descriptor 的 range），
/// 同一 shader 的 Params 布局恒定，故该维度仅作防御性区分。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KernelKey {
    shader: String,
    spec: Vec<u32>,
    n_params: usize,
}

impl Runtime {
    /// 创建新的 Vulkan runtime，并预分配一个持久 compute command buffer
    pub fn new() -> R<Self> {
        let app = App::new()?;
        let cmd_buf = app.allocate_compute_command_buffers(1)?;
        let cmd = cmd_buf[0];
        // uniform 池：默认 8MB（≈ 3 万次 dispatch 的 Params），UNIFORM_POOL_MB 可覆盖。
        // host-visible + host-cached：写入即 mapped memcpy，无 staging。
        let pool_mb: usize = std::env::var("UNIFORM_POOL_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let uniform_pool = app.create_uniform(pool_mb * 1024 * 1024)?;
        let pool_align = app.properties.limits.min_uniform_buffer_offset_alignment as usize;
        let pool_align = pool_align.max(1);
        Ok(Self {
            app,
            cmd,
            recording: false,
            pending: Vec::new(),
            cache: HashMap::new(),
            uniform_pool,
            pool_cursor: 0,
            pool_slots: HashMap::new(),
            pool_align,
            written: HashMap::new(),
            read: HashMap::new(),
            written_batch: HashMap::new(),
            prof_host: std::time::Duration::ZERO,
            prof_kernels: 0,
            prof_key: std::time::Duration::ZERO,
            prof_cache: std::time::Duration::ZERO,
            prof_bar: std::time::Duration::ZERO,
            prof_disp: std::time::Duration::ZERO,
            prof_gpu: false,
            prof_gpu_entries: Vec::new(),
            prof_gpu_count: 0,
        })
    }

    /// 创建一维 f32 GPU 张量
    pub fn create_tensor(&self, len: usize) -> R<GpuTensor> {
        let layout = Layout::from_shape([len]);
        let host = self.app.create_tensor::<f32>(&layout, None, true)?;
        let device = self.app.create_tensor::<f32>(&layout, None, false)?;
        Ok(GpuTensor {
            host: Some(host),
            device,
            len,
        })
    }

    /// 创建一维 fp16 GPU 张量（tensor-core GEMM 用）
    pub fn create_tensor_f16(&self, len: usize) -> R<GpuTensor16> {
        let layout = Layout::from_shape([len]);
        let host = self.app.create_tensor::<f16>(&layout, None, true)?;
        let device = self.app.create_tensor::<f16>(&layout, None, false)?;
        Ok(GpuTensor16 {
            host: Some(host),
            device,
            len,
        })
    }

    /// 创建一维 u32 GPU 张量（int8 idx/sz 用）
    pub fn create_tensor_u32(&self, len: usize) -> R<GpuTensorU32> {
        let layout = Layout::from_shape([len]);
        let host = self.app.create_tensor::<u32>(&layout, None, true)?;
        let device = self.app.create_tensor::<u32>(&layout, None, false)?;
        Ok(GpuTensorU32 {
            host: Some(host),
            device,
        })
    }

    /// 上传 u32 数据到 GPU（host → device）
    pub fn upload_u32(&self, tensor: &GpuTensorU32, data: &[u32]) -> R<()> {
        let host = tensor.host.as_ref().ok_or("upload_u32: host dropped")?;
        host.copy_from(data, 0)?;
        self.copy_buffer(host, &tensor.device)?;
        Ok(())
    }

    /// 释放 u32 张量的 host（系统内存）缓冲（见 `drop_host`）。
    pub fn drop_host_u32(&self, t: &mut GpuTensorU32) {
        t.host = None;
    }

    /// 释放 host（系统内存）缓冲。权重上传完成后调用，host 缓冲不再需要时可省内存。
    /// 注意：释放后不可再对该张量调用 `upload`/`download`。
    pub fn drop_host(&self, t: &mut GpuTensor) {
        t.host = None;
    }

    /// 释放 fp16 张量的 host（系统内存）缓冲（见 `drop_host`）。
    pub fn drop_host_f16(&self, t: &mut GpuTensor16) {
        t.host = None;
    }

    /// 上传 f32 数据到 fp16 张量（host 端 f32→f16 转换）
    pub fn upload_f16(&self, tensor: &GpuTensor16, data: &[f32]) -> R<()> {
        let f16s: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        let host = tensor.host.as_ref().ok_or("upload_f16: host dropped")?;
        host.copy_from(&f16s, 0)?;
        self.copy_buffer(host, &tensor.device)?;
        Ok(())
    }

    /// 从 fp16 张量下载数据（转为 f32 返回）。host 已释放时用临时 staging 缓冲（诊断用）。
    pub fn download_f16(&self, tensor: &GpuTensor16) -> R<Vec<f32>> {
        let staging;
        let host = match &tensor.host {
            Some(h) => h,
            None => {
                staging =
                    self.app
                        .create_tensor::<f16>(&Layout::from_shape([tensor.len]), None, true)?;
                &staging
            }
        };
        self.copy_buffer(&tensor.device, host)?;
        let mut f16s = vec![f16::from_f32(0.0); tensor.len];
        host.copy_to(&mut f16s)?;
        Ok(f16s.iter().map(|x| x.to_f32()).collect())
    }

    /// 从 u32 张量下载数据（device → host）。host 已释放时用临时 staging 缓冲（诊断用）。
    pub fn download_u32(&self, tensor: &GpuTensorU32, len: usize) -> R<Vec<u32>> {
        let staging;
        let host = match &tensor.host {
            Some(h) => h,
            None => {
                staging = self
                    .app
                    .create_tensor::<u32>(&Layout::from_shape([len]), None, true)?;
                &staging
            }
        };
        self.copy_buffer(&tensor.device, host)?;
        let mut data = vec![0u32; len];
        host.copy_to(&mut data)?;
        Ok(data)
    }

    /// 上传数据到 GPU（host → device）
    pub fn upload(&self, tensor: &GpuTensor, data: &[f32]) -> R<()> {
        let host = tensor.host.as_ref().ok_or("upload: host dropped")?;
        host.copy_from(data, 0)?;
        self.copy_buffer(host, &tensor.device)?;
        Ok(())
    }

    /// 从 GPU 下载数据（device → host）
    pub fn download(&self, tensor: &GpuTensor) -> R<Vec<f32>> {
        let host = tensor.host.as_ref().ok_or("download: host dropped")?;
        self.copy_buffer(&tensor.device, host)?;
        let mut data = vec![0.0f32; tensor.len];
        host.copy_to(&mut data)?;
        Ok(data)
    }

    /// 记录并执行 buffer 拷贝命令（src → dst），使用 compute 队列
    fn copy_buffer<T: crate::vulkan::num::Scalar>(
        &self,
        src: &Tensor<T>,
        dst: &Tensor<T>,
    ) -> R<()> {
        let cmd_buf = self.app.allocate_compute_command_buffers(1)?;
        let cmd = cmd_buf[0];
        unsafe {
            self.app
                .device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.app
                .device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;
            dst.cmd_copy_from(cmd, src);
            self.app.device.end_command_buffer(cmd)?;
            let buffers = [cmd];
            let submit = vk::SubmitInfo::builder().command_buffers(&buffers);
            let submits = [submit.build()];
            self.app
                .device
                .queue_submit(self.app.compute.queue, &submits, vk::Fence::null())?;
            self.app.device.queue_wait_idle(self.app.compute.queue)?;
        }
        Ok(())
    }

    /// 开启一次批处理记录：reset 并 begin 持久 command buffer，清空待提交资源
    pub fn begin_batch(&mut self) -> R<()> {
        unsafe {
            self.app
                .device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            self.app
                .device
                .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::builder())?;
        }
        self.recording = true;
        self.pending.clear();
        // uniform 池游标/slot 表复位（上一批已 wait_idle，slot 内容可安全覆盖）。
        self.pool_cursor = 0;
        self.pool_slots.clear();
        // 上一批 end_batch 已 queue_wait_idle 全量同步，所有缓冲写入对当前批可见；
        // 清空 written，避免把上一批的写入误判为本批需要同步。
        self.written.clear();
        self.read.clear();
        // GPU 时间戳剖析：PROF_GPU=1 时清空并复位查询池，记录本批每个 kernel 的执行时间。
        self.prof_gpu = std::env::var("PROF_GPU").is_ok();
        if self.prof_gpu {
            self.prof_gpu_entries.clear();
            self.prof_gpu_count = 0;
            unsafe {
                self.app
                    .cmd_reset_query_pool(self.cmd, 0, QUERY_POOL_SIZE as u32);
            }
        }
        Ok(())
    }

    /// 结束批处理：end command buffer，一次性 submit + wait，随后释放 kernel 资源（归还 descriptor set）
    pub fn end_batch(&mut self) -> R<()> {
        if !self.recording {
            return Ok(());
        }
        unsafe {
            self.app.device.end_command_buffer(self.cmd)?;
            let buffers = [self.cmd];
            let submit = vk::SubmitInfo::builder().command_buffers(&buffers);
            let submits = [submit.build()];
            self.app
                .device
                .queue_submit(self.app.compute.queue, &submits, vk::Fence::null())?;
            self.app.device.queue_wait_idle(self.app.compute.queue)?;
        }
        self.recording = false;
        self.pending.clear();
        self.written.clear();
        self.read.clear();
        self.written_batch.clear();
        // uniform 池游标/slot 表复位：GPU 已 wait_idle，下批 dispatch 从池头复用 slot。
        self.pool_cursor = 0;
        self.pool_slots.clear();
        // GPU 时间戳剖析：读取并按 kernel label 聚合，输出每个 kernel 的执行时间与带宽利用率。
        if self.prof_gpu && !self.prof_gpu_entries.is_empty() {
            let data = self.app.query_timestamps_n(self.prof_gpu_count)?;
            let period = self.app.properties.limits.timestamp_period as f64; // ns/count
            // 峰值带宽（GB/s）：RTX 2080 Ti 默认 616，可用 PEAK_GBS 覆盖为实测值。
            let peak_gbs: f64 = std::env::var("PEAK_GBS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(616.0);
            let mut acc: HashMap<String, (f64, u32, u64)> = HashMap::new();
            let mut total = 0.0;
            for (label, b, e, est_bytes) in &self.prof_gpu_entries {
                let dur_ns =
                    ((data[*e as usize]).saturating_sub(data[*b as usize]) as f64) * period;
                let ent = acc.entry(label.clone()).or_insert((0.0, 0, 0));
                ent.0 += dur_ns;
                ent.1 += 1;
                ent.2 += *est_bytes;
                total += dur_ns;
            }
            // 按总耗时降序打印，便于定位瓶颈 kernel
            let mut rows: Vec<_> = acc.into_iter().collect();
            rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            log::info!("[PROF_GPU] peak bandwidth = {peak_gbs:.0} GB/s (PEAK_GBS 可覆盖)");
            for (label, (sum_ns, count, sum_bytes)) in rows {
                let avg_ns = sum_ns / count as f64;
                let avg_bytes = sum_bytes as f64 / count as f64;
                let bw_gbs = if avg_ns > 0.0 {
                    avg_bytes / (avg_ns / 1e9) / 1e9
                } else {
                    0.0
                };
                let util_pct = if peak_gbs > 0.0 {
                    bw_gbs / peak_gbs * 100.0
                } else {
                    0.0
                };
                log::info!(
                    "[PROF_GPU] {label:>30}  count={count:>3}  total={:>8.3}ms  avg={:>8.4}ms  est={:>9.1}MiB  bw={:>6.1}GB/s ({util_pct:>5.1}%)",
                    sum_ns / 1e6,
                    avg_ns / 1e6,
                    avg_bytes / ((1 << 20) as f64),
                    bw_gbs,
                );
            }
            log::info!(
                "[PROF_GPU] SUM {:.3}ms over {} kernels",
                total / 1e6,
                self.prof_gpu_entries.len()
            );
            self.prof_gpu_entries.clear();
            self.prof_gpu_count = 0;
        }
        if std::env::var("PROF_HOST").is_ok() {
            log::info!(
                "[PROF_HOST] batch: {} kernels, host={:.3}ms [key={:.2}ms cache={:.2}ms barrier={:.2}ms dispatch={:.2}ms]",
                self.prof_kernels,
                self.prof_host.as_secs_f64() * 1e3,
                self.prof_key.as_secs_f64() * 1e3,
                self.prof_cache.as_secs_f64() * 1e3,
                self.prof_bar.as_secs_f64() * 1e3,
                self.prof_disp.as_secs_f64() * 1e3,
            );
            self.prof_host = std::time::Duration::ZERO;
            self.prof_kernels = 0;
            self.prof_key = std::time::Duration::ZERO;
            self.prof_cache = std::time::Duration::ZERO;
            self.prof_bar = std::time::Duration::ZERO;
            self.prof_disp = std::time::Duration::ZERO;
        }
        Ok(())
    }

    /// device→device 拷贝：record 到批处理 command buffer（需在 begin_batch 内调用）
    /// 替代原 download+upload 的 host 往返，消除中间同步点。
    pub fn copy_device(&mut self, src: &GpuTensor, dst: &mut GpuTensor) -> R<()> {
        self.record_barriers(&[src.device.buffer], &[dst.device.buffer])?;
        unsafe {
            dst.device.cmd_copy_from(self.cmd, &src.device);
        }
        self.mark_written(&[dst.device.buffer]);
        Ok(())
    }

    /// device→device 拷贝（fp16 张量）：record 到批处理 command buffer。
    /// 用于 v_first 快照（fp16 v 缓冲 → fp16 v_first 缓冲）。
    pub fn copy_device_f16(&mut self, src: &GpuTensor16, dst: &mut GpuTensor16) -> R<()> {
        self.record_barriers(&[src.device.buffer], &[dst.device.buffer])?;
        unsafe {
            dst.device.cmd_copy_from(self.cmd, &src.device);
        }
        self.mark_written(&[dst.device.buffer]);
        Ok(())
    }

    /// 清空 kernel 缓存（释放 descriptor set / pipeline）。
    /// sequence-parallel 在序列长度 T 变化时会重建工作缓冲区，导致新的 device address
    /// 与新的 spec（含 T），旧缓存不再命中。若不清空，缓存会随 T 变化无限累积，
    /// 最终耗尽 descriptor pool（ERROR_OUT_OF_POOL_MEMORY）。
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// 在批处理 command buffer 中，为 kernel 的读/写缓冲插入内存 barrier。
    /// 默认用 buffer 级屏障：按逐缓冲依赖插入 barrier，允许无依赖的 kernel 并发执行，
    /// 正确处理 RAW/WAR/WAW 三类竞争（sole-token 与多 token 均验证确定、对齐 CPU）。
    /// 仅当显式设置 FULL_BARRIER=1 时走全量内存屏障（诊断回退，保证正确但串行化 kernel）。
    fn record_barriers(&mut self, reads: &[vk::Buffer], writes: &[vk::Buffer]) -> R<()> {
        // FULL_BARRIER=1：全量内存屏障（诊断回退，覆盖全部竞争但串行化）。
        if std::env::var("FULL_BARRIER").is_ok() {
            unsafe {
                let mb = vk::MemoryBarrier::builder()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ,
                    )
                    .build();
                let mbs = [mb];
                self.app.device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &mbs,
                    &[] as &[vk::BufferMemoryBarrier],
                    &[] as &[vk::ImageMemoryBarrier],
                );
            }
            self.written.clear();
            return Ok(());
        }

        let mut barriers: Vec<vk::BufferMemoryBarrier> = Vec::new();
        // RAW（读后写）+ WAW（写后写）：读/写缓冲若先前已被本批写入，需先让该写入可见/有序。
        // 同一 barrier（src=WRITE, dst=READ|WRITE）同时覆盖 RAW 与 WAW。
        for buf in reads.iter().chain(writes.iter()) {
            if let Some(&b) = self.written.get(&buf.as_raw()) {
                barriers.push(
                    vk::BufferMemoryBarrier::builder()
                        .buffer(b)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .src_access_mask(
                            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::TRANSFER_READ,
                        )
                        .build(),
                );
            }
        }
        // WAR（写后读）：写缓冲若先前被本批读过，需插入执行依赖（src_access 为空），
        // 确保前一个 kernel 的读取完成后才允许本 kernel 写入，防止覆盖未读数据。
        for buf in writes {
            if let Some(&b) = self.read.get(&buf.as_raw()) {
                barriers.push(
                    vk::BufferMemoryBarrier::builder()
                        .buffer(b)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .build(),
                );
            }
        }
        if std::env::var("LOG_BARRIERS").is_ok() {
            let mut msgs: Vec<String> = Vec::new();
            for buf in reads {
                let in_w = self.written.contains_key(&buf.as_raw());
                msgs.push(format!(
                    "{}{:#x}",
                    if in_w { "B" } else { "-" },
                    buf.as_raw()
                ));
            }
            log::info!("[BAR] {msgs:?}");
            // 诊断：本批写入过但当前读取未含 barrier 的缓冲（缺失同步）
            let missing: Vec<String> = reads
                .iter()
                .filter(|b| {
                    self.written_batch.contains_key(&b.as_raw())
                        && !self.written.contains_key(&b.as_raw())
                })
                .map(|b| format!("{:#x}", b.as_raw()))
                .collect();
            if !missing.is_empty() {
                log::warn!(
                    "[MISSING-BARRIER] read of batch-written buffer w/o barrier: {missing:?}"
                );
            }
        }
        if !barriers.is_empty() {
            unsafe {
                self.app.device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[] as &[vk::MemoryBarrier],
                    &barriers,
                    &[] as &[vk::ImageMemoryBarrier],
                );
            }
        }
        // 更新依赖集合：
        //   - 读缓冲：RAW 已同步，从 written 移除；并记录到 read（供后续 WAR 检测）。
        //   - 写缓冲：加入 written（供后续 RAW/WAW 检测）；并从 read 移除（已写完，本次读取作废）。
        for buf in reads {
            self.written.remove(&buf.as_raw());
            self.read.insert(buf.as_raw(), *buf);
        }
        for buf in writes {
            self.written.insert(buf.as_raw(), *buf);
            self.read.remove(&buf.as_raw());
        }
        Ok(())
    }

    /// 记录本次 dispatch 写入的缓冲，供后续 kernel 读取时检测同步需求。
    fn mark_written(&mut self, writes: &[vk::Buffer]) {
        for w in writes {
            self.written.insert(w.as_raw(), *w);
            self.written_batch.insert(w.as_raw(), *w);
        }
    }

    /// 创建 uniform 绑定（1 个 DYNAMIC UNIFORM_BUFFER at binding 0）。
    /// DYNAMIC + uniform 池：descriptor 绑定整个池 buffer + 定长 range，
    /// 实际偏移由 cmd_bind 的 dynamic offsets 提供（每次 dispatch 独立 slot）。
    fn uniform_binding() -> [vk::DescriptorSetLayoutBindingBuilder<'static>; 1] {
        [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)]
    }

    /// 创建 kernel、打包 uniform、绑定，并 record 到批处理 command buffer
    /// params 为 device address 列表
    ///
    /// kernel（pipeline + descriptor set）按 (shader, spec, n_params) 缓存——
    /// **key 不含 buffer 地址**（旧版含地址导致同形状不同层的 kernel 各建一套
    /// pipeline，首次 prefill ~1700 次创建 ≈ 240ms）。地址改为写入 uniform 池的
    /// slot 内容，由 DYNAMIC offset 在 cmd_bind 时指向：
    ///   1. 池 slot 按 params 内容批内 dedup（decode/selfloop 每 token params 一致，
    ///      池占用与 token 数无关，对齐旧版缓存 uniform 的去重语义）；
    ///   2. cache 命中 → 复用 pipeline，仅 memcpy params 到 slot（~几十 ns）；
    ///      未命中 → 创建 pipeline + descriptor（绑定池 + range），此后永久复用。
    fn record_kernel(
        &mut self,
        shader_name: &str,
        specialization: &[u32],
        params: &[u64],
        dispatch: (u32, u32, u32),
        reads: &[vk::Buffer],
        writes: &[vk::Buffer],
    ) -> R<()> {
        let prof_host = std::env::var("PROF_HOST").is_ok();
        let t0 = std::time::Instant::now();
        let key = KernelKey {
            shader: shader_name.to_string(),
            spec: specialization.to_vec(),
            n_params: params.len(),
        };
        let t_key = if prof_host {
            t0.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        // uniform 池分配本 dispatch 的 slot（批内 dedup：相同 params 复用偏移）。
        // decode/selfloop 每 token 的 params 逐位一致，dedup 使池占用与 token 数
        // 无关；两个 kernel 即便 shader 不同，只要 params 逐位相同，各自的 shader
        // 按自身布局解读相同字节，结果与各自调用方传入的值一致，共享安全。
        let slot_bytes = std::mem::size_of_val(params);
        let offset = if let Some(&off) = self.pool_slots.get(params) {
            off
        } else {
            let off = (self.pool_cursor + self.pool_align - 1) & !(self.pool_align - 1);
            if off + slot_bytes > self.uniform_pool.size {
                return Err(format!(
                    "uniform pool exhausted: need offset {off} + {slot_bytes} bytes, \
                     capacity {} bytes（批内不同 params 组合过多；可用 UNIFORM_POOL_MB 调大）",
                    self.uniform_pool.size
                )
                .into());
            }
            self.uniform_pool.copy_from_at(params, off)?;
            self.pool_cursor = off + slot_bytes;
            self.pool_slots.insert(params.to_vec(), off);
            off
        };
        // 缓存命中：复用 pipeline（descriptor 已绑池 + range），uniform 内容已写 slot。
        // NO_CACHE=1 时绕过缓存（诊断用，验证缓存是否引入竞态；仍走池）。
        let no_cache = std::env::var("NO_CACHE").is_ok();
        let t_cache0 = std::time::Instant::now();
        let kernel = if !no_cache {
            if let Some(k) = self.cache.get(&key) {
                k.clone()
            } else {
                let code = Self::load_shader(shader_name)?;
                let bindings = Self::uniform_binding();
                let kernel = self
                    .app
                    .create_kernel(code.as_ref(), specialization, &bindings)?;
                kernel
                    .binder()
                    .bind_uniform_with_range(&self.uniform_pool, slot_bytes, 0, 0)
                    .build();
                self.cache.insert(key.clone(), kernel.clone());
                kernel
            }
        } else {
            let code = Self::load_shader(shader_name)?;
            let bindings = Self::uniform_binding();
            let kernel = self
                .app
                .create_kernel(code.as_ref(), specialization, &bindings)?;
            kernel
                .binder()
                .bind_uniform_with_range(&self.uniform_pool, slot_bytes, 0, 0)
                .build();
            kernel
        };
        let t_cache = if prof_host {
            t_cache0.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        if std::env::var("LOG_BARRIERS").is_ok() {
            let fmt = |bs: &[vk::Buffer]| -> String {
                bs.iter()
                    .map(|b| format!("{:#x}", b.as_raw()))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            log::info!("[K] {} R[{}] W[{}]", shader_name, fmt(reads), fmt(writes));
        }
        // buffer 级/全量 barrier：按读/写缓冲处理 RAW/WAR/WAW 依赖
        let t_bar0 = std::time::Instant::now();
        self.record_barriers(reads, writes)?;
        let t_bar = if prof_host {
            t_bar0.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        // GPU 时间戳：dispatch 前写 begin，dispatch 后写 end（仅 PROF_GPU=1 时开启）
        let gpu_q = if self.prof_gpu && self.prof_gpu_count + 2 <= QUERY_POOL_SIZE as u32 {
            let b = self.prof_gpu_count;
            let e = self.prof_gpu_count + 1;
            self.prof_gpu_count += 2;
            unsafe {
                self.app
                    .cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, b);
            }
            Some((b, e))
        } else {
            None
        };
        unsafe {
            // DYNAMIC offset：descriptor 指向 uniform 池的本 dispatch slot。
            kernel.cmd_bind(self.cmd, &[offset as u32]);
            self.app
                .device
                .cmd_dispatch(self.cmd, dispatch.0, dispatch.1, dispatch.2);
        }
        if let Some((b, e)) = gpu_q {
            unsafe {
                self.app
                    .cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, e);
            }
            let est_bytes = est_kernel_bytes(shader_name, specialization, dispatch);
            self.prof_gpu_entries
                .push((shader_name.to_string(), b, e, est_bytes));
        }
        let t_disp = if prof_host {
            t_bar0.elapsed() - t_bar
        } else {
            std::time::Duration::ZERO
        };
        self.mark_written(writes);
        self.pending.push(kernel);
        if prof_host {
            self.prof_host += t0.elapsed();
            self.prof_kernels += 1;
            self.prof_key += t_key;
            self.prof_cache += t_cache;
            self.prof_bar += t_bar;
            self.prof_disp += t_disp;
        }
        Ok(())
    }

    /// 加载嵌入的 SPIR-V 着色器
    fn load_shader(name: &str) -> R<Cow<'static, [u8]>> {
        let file = asset::Asset::get(name).ok_or_else(|| format!("shader not found: {name}"))?;
        Ok(file.data)
    }

    // ===== GEMV 算子 =====
    // A 存储为 [M, K] 行主序（PyTorch nn.Linear 权重 [out, in] 原始布局）
    // shader 逻辑 A[k, m] = k + STRIDE_A_Y * m，对应 A_stored[m, k] = m*K + k

    // ===== fp32io16 GEMV（fp16 权重、f32 输入/输出、f32 累加）=====
    // 单 token 路径用，复用已加载的 fp16 权重（_w16），减半权重内存带宽。

    /// y = x @ A（fp16 权重，f32 输入/输出）
    pub fn gemv_f16(
        &mut self,
        a16: &GpuTensor16,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [a16.device.address, 0, x.device.address, y.device.address];
        self.record_kernel(
            "shaders/spv/gemv_f32io.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[a16.device.buffer, x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = relu²(x @ A)（fp16 权重，f32 输入/输出）
    pub fn gemv_f16_relu2(
        &mut self,
        a16: &GpuTensor16,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [a16.device.address, 0, x.device.address, y.device.address];
        self.record_kernel(
            "shaders/spv/gemv_f32io_relu2.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[a16.device.buffer, x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// int8 → fp16 反量化（prefill 方案A）：把 int8 量化权重 [M, K] 解到 fp16 scratch，
    /// 供 tensor-core GEMM 消费。输出布局与 load_linear_f16 上传的 [M, K] 行主序一致。
    /// dispatch: (ceil(M*K/4 / 256), 1, 1)，每线程处理 1 个 uint32（4 权重）。
    pub fn dequant_int8_to_f16(
        &mut self,
        a: &GpuTensorInt8,
        out: &GpuTensor16,
        m: usize,
        k: usize,
    ) -> R<()> {
        debug_assert_eq!(a.m, m);
        debug_assert_eq!(a.k, k);
        debug_assert!(out.len >= m * k);
        let spec = [m as u32, k as u32];
        let params = [
            a.idx.device.address,
            a.sz.device.address,
            out.device.address,
        ];
        let threads = (m * (k / 4)) as u32;
        self.record_kernel(
            "shaders/spv/dequant_int8_f16.spv",
            &spec,
            &params,
            (threads.div_ceil(256), 1, 1),
            &[a.idx.device.buffer, a.sz.device.buffer],
            &[out.device.buffer],
        )?;
        Ok(())
    }

    /// y = relu²(x @ A)（int8 量化权重，f32 输入/输出）
    /// 与 gemv_f16_relu2 同构，权重带宽降到 fp16 的一半（1 byte/权重）。
    pub fn gemv_int8_relu2(
        &mut self,
        a: &GpuTensorInt8,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        debug_assert_eq!(a.m, m);
        debug_assert_eq!(a.k, k);
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a.idx.device.address,
            a.sz.device.address,
            0,
            x.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_int8_relu2.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[a.idx.device.buffer, a.sz.device.buffer, x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = x @ A（int8 量化权重，f32 输入/输出，覆盖写）。
    /// head 投影用：直接 int8 gemv 覆盖写 logits，避免走「整行反量化到 fp16 scratch」路径——
    /// 后者对 vocab=65536 的 head 会发出 grid.x=163840 的 dispatch，超过 Vulkan 的
    /// maxComputeWorkGroupCount(65535) 导致 ERROR_DEVICE_LOST。
    pub fn gemv_int8_plain(
        &mut self,
        a: &GpuTensorInt8,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        debug_assert_eq!(a.m, m);
        debug_assert_eq!(a.k, k);
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a.idx.device.address,
            a.sz.device.address,
            0,
            x.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_int8.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[a.idx.device.buffer, a.sz.device.buffer, x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = y + (x .* g) @ A（int8 量化权重，f32 输入/输出/累加，g 为 fp16 门控）。
    /// 与 gemv_f16_mul_add 同构（att.output 用），残差累加目标 y 即输入 acc。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_int8_mul_add(
        &mut self,
        a: &GpuTensorInt8,
        x: &GpuTensor,
        g: &GpuTensor16,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        debug_assert_eq!(a.m, m);
        debug_assert_eq!(a.k, k);
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a.idx.device.address,
            a.sz.device.address,
            x.device.address,
            g.device.address,
            y.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_int8_add_mul.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[
                a.idx.device.buffer,
                a.sz.device.buffer,
                x.device.buffer,
                g.device.buffer,
                y.device.buffer,
            ],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = y + x @ A（int8 量化权重，f32 输入/输出/累加）。
    /// 与 gemv_f16_add 同构（ffn.value 用），残差累加目标 y 即输入 acc。
    pub fn gemv_int8_add(
        &mut self,
        a: &GpuTensorInt8,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        debug_assert_eq!(a.m, m);
        debug_assert_eq!(a.k, k);
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a.idx.device.address,
            a.sz.device.address,
            x.device.address,
            0,
            y.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_int8_add.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[
                a.idx.device.buffer,
                a.sz.device.buffer,
                x.device.buffer,
                y.device.buffer,
            ],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = y + (x .* g) @ A（fp16 权重，f32 输入/输出/累加）。
    /// 把 y_g = y_norm * g 的 mul 与 x += y_out 的残差累加都折叠进 output gemv，省 2 次独立 dispatch。
    /// 残差累加目标 y 即输入 acc（y 读旧值、写新值）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f16_mul_add(
        &mut self,
        a16: &GpuTensor16,
        x: &GpuTensor,
        g: &GpuTensor16,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a16.device.address,
            x.device.address,
            g.device.address,
            y.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_f32io_add_mul.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[
                a16.device.buffer,
                x.device.buffer,
                g.device.buffer,
                y.device.buffer,
            ],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = y + x @ A（fp16 权重，f32 输入/输出/累加）。
    /// 把 x += v2 的残差累加折叠进 ffn_value gemv，省 1 次独立 dispatch。
    /// 残差累加目标 y 即输入 acc（y 读旧值、写新值）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f16_add(
        &mut self,
        a16: &GpuTensor16,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> R<()> {
        let rows = gemv_rows_for(m);
        let spec = gemv_spec(&self.app, m, k, rows);
        let params = [
            a16.device.address,
            x.device.address,
            0u64,
            y.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/gemv_f32io_add.spv",
            &spec,
            &params,
            ((m / rows) as u32, batch as u32, 1),
            &[a16.device.buffer, x.device.buffer, y.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// 稀疏 FFN value 投影：x += r2 @ ffn_value（fp16 平铺权重，r2 约 96% 稀疏）。
    /// 只读 r2 非零列的权重并 buffer 级 fp32 原子累加（VK_EXT_shader_atomic_float）。
    /// ffn_value_sparse_add.comp，dispatch (fh/TILE, c/C_TILE, 1)。
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_value_sparse_add(
        &mut self,
        value_tiled: &GpuTensor16,
        r2: &GpuTensor,
        x: &mut GpuTensor,
        c: usize,
        fh: usize,
    ) -> R<()> {
        const TILE: usize = 128;
        const C_TILE: usize = 256;
        debug_assert_eq!(fh % TILE, 0, "fh 需为 TILE 整数倍");
        debug_assert_eq!(c % C_TILE, 0, "c 需为 C_TILE 整数倍");
        let spec = [c as u32, fh as u32, TILE as u32, C_TILE as u32];
        // 参数顺序需与 ffn_value_sparse_add.comp 的 Params 成员顺序一致：
        //   BufR2 buf_r2; BufW buf_w; BufX buf_x;  →  [r2, value_tiled, x]
        let params = [
            r2.device.address,
            value_tiled.device.address,
            x.device.address,
        ];
        self.record_kernel(
            "shaders/spv/ffn_value_sparse_add.spv",
            &spec,
            &params,
            ((fh / TILE) as u32, (c / C_TILE) as u32, 1),
            &[value_tiled.device.buffer, r2.device.buffer, x.device.buffer],
            &[x.device.buffer],
        )?;
        Ok(())
    }

    /// 是否支持稀疏 FFN 内核（需设备支持 VK_EXT_shader_atomic_float 的 buffer fp32 原子）。
    pub fn supports_sparse_ffn(&self) -> bool {
        self.app.properties.atomic_float
    }

    /// GPU argmax：从 logits [N] 找最大值索引，写入 token（len 1 的 f32 缓冲，字节存 uint）。
    /// 相比把 65536 个 logits 下载到 CPU 再遍历，只回传 4 字节索引，省去每 token 的
    /// 大块 device→host 传输与 CPU 遍历开销（对标 albatross 的 torch.argmax 全 GPU 采样）。
    pub fn argmax(&mut self, logits: &GpuTensor, token: &mut GpuTensor, n: usize) -> R<()> {
        let spec = [n as u32];
        let params = [logits.device.address, token.device.address];
        self.record_kernel(
            "shaders/spv/argmax.spv",
            &spec,
            &params,
            (1, 1, 1),
            &[logits.device.buffer],
            &[token.device.buffer],
        )?;
        Ok(())
    }

    /// GPU argmax 直接写回 host-visible 缓冲（token_host，字节存 uint）。
    /// 供 GPU self-loop 使用：argmax 结果写入 gather 读取的同一 host 缓冲，
    /// 下一轮 gather 自动跟随，CPU 无需回读/回传 token，单 batch 内多 token 自循环。
    pub fn argmax_into_host(
        &mut self,
        logits: &GpuTensor,
        token_host: &Tensor<f32>,
        n: usize,
    ) -> R<()> {
        let spec = [n as u32];
        let params = [logits.device.address, token_host.address];
        self.record_kernel(
            "shaders/spv/argmax.spv",
            &spec,
            &params,
            (1, 1, 1),
            &[logits.device.buffer],
            &[token_host.buffer],
        )?;
        Ok(())
    }

    /// GPU 采样：penalty(repetition/frequency/presence) + temperature + top-k + top-p 过滤后按概率采样，
    /// 写入 token（len 1 f32 位模式存 uint）。
    /// 相比下载 65536 个 logits 到 CPU 采样，只在 GPU 完成并回传 4 字节索引。
    /// `history` 为已生成的 token（用于惩罚计数），空则跳过惩罚阶段。
    /// 自建采样临时缓冲（temp/mask/counter/sampler/hist），单次调用（非 self-loop）使用；
    /// 自回归每 token 一次独立 begin/end batch，临时缓冲可安全随调用释放。
    #[allow(clippy::too_many_arguments)] // sampler 标量参数直接透传 shader，保持扁平
    pub fn sample(
        &mut self,
        logits: &GpuTensor,
        token: &mut GpuTensor,
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
        let temp = self.create_tensor(n)?;
        let mask = self.create_tensor(n)?;
        let counter = self.create_tensor_u32(n)?;
        let sampler = self.create_tensor(8)?;
        self.store_sampler_host(
            &sampler,
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            history.len() as u32,
        )?;
        // 历史 token 缓冲（hist_len=0 时 shader 跳过直方图，缓冲仅需合法地址）
        let hist = self.create_tensor_u32(history.len().max(1))?;
        if !history.is_empty() {
            self.upload_u32(&hist, history)?;
        }
        self.record_sample(
            logits,
            token.device.address,
            &token.device.buffer,
            &temp,
            &mask,
            &counter,
            &sampler,
            hist.device.address,
            &hist.device.buffer,
            n,
        )
    }

    /// GPU 采样（self-loop 批量版）：采样结果直接写回 host-visible 缓冲（token_host）。
    /// 复用调用方预建的 temp/mask/counter/sampler 缓冲（存活到 batch 提交后），供单 batch 内
    /// 连续多轮采样使用；sampler 的 seed 与 hist_len 由调用方每轮用 `store_sampler_host` 更新。
    /// `hist` 为历史 token 缓冲（self-loop 中即累积的 token_seq）。
    #[allow(clippy::too_many_arguments)] // 各缓冲引用扁平透传 record_sample，保持结构清晰
    pub fn sample_into_host_seeded(
        &mut self,
        logits: &GpuTensor,
        token_host: &Tensor<f32>,
        n: usize,
        temp: &GpuTensor,
        mask: &GpuTensor,
        counter: &GpuTensorU32,
        sampler: &GpuTensor,
        hist: &GpuTensor,
    ) -> R<()> {
        self.record_sample(
            logits,
            token_host.address,
            &token_host.buffer,
            temp,
            mask,
            counter,
            sampler,
            hist.device.address,
            &hist.device.buffer,
            n,
        )
    }

    /// 记录一次采样 kernel（共享实现）。
    #[allow(clippy::too_many_arguments)] // 各缓冲地址 + 长度，扁平透传 shader
    fn record_sample(
        &mut self,
        logits: &GpuTensor,
        token_addr: u64,
        token_buf: &vk::Buffer,
        temp: &GpuTensor,
        mask: &GpuTensor,
        counter: &GpuTensorU32,
        sampler: &GpuTensor,
        hist_addr: u64,
        hist_buf: &vk::Buffer,
        n: usize,
    ) -> R<()> {
        // sampler 由 CPU 直写 host-visible 缓冲，需 HOST_WRITE→SHADER_READ 屏障
        self.host_write_to_shader_barrier(&sampler.device.buffer)?;
        let spec = [n as u32];
        let params = [
            logits.device.address,
            token_addr,
            temp.device.address,
            mask.device.address,
            counter.device.address,
            hist_addr,
            sampler.device.address,
        ];
        self.record_kernel(
            "shaders/spv/sample.spv",
            &spec,
            &params,
            (1, 1, 1),
            &[
                logits.device.buffer,
                sampler.device.buffer,
                *hist_buf,
                temp.device.buffer,
                mask.device.buffer,
            ],
            &[
                *token_buf,
                temp.device.buffer,
                mask.device.buffer,
                counter.device.buffer,
            ],
        )?;
        Ok(())
    }

    /// 把 host-visible 缓冲 in_tok[0]（token 索引，字节存 uint）追加到序列缓冲 out_seq[cnt]，
    /// 并将 cnt 自增。供 GPU self-loop 记录每轮生成的 token，便于一次性下载验证。
    /// spec 恒空（不重建 pipeline）。dispatch (1,1,1)。
    pub fn record_token(
        &mut self,
        in_tok: &Tensor<f32>,
        out_seq: &GpuTensor,
        cnt: &mut GpuTensor,
    ) -> R<()> {
        let params = [in_tok.address, out_seq.device.address, cnt.device.address];
        self.record_kernel(
            "shaders/spv/record_token.spv",
            &[],
            &params,
            (1, 1, 1),
            &[in_tok.buffer, cnt.device.buffer],
            &[out_seq.device.buffer, cnt.device.buffer],
        )?;
        Ok(())
    }

    /// 把 token 索引直接写入 host-visible 缓冲（tok.host，f32 位模式存 uint）。
    /// CPU 侧 memcpy，无 kernel、无 specialization constant——每 token 值不同不再重建
    /// pipeline（消除 store_u32 spec constant 导致的回退）。gather_row_device 直接从
    /// 该 host 缓冲的 device address 读取，仍为全 GPU 循环体，为 GPU self-loop 铺路。
    pub fn store_token_host(&self, tok: &GpuTensor, token: u32) -> R<()> {
        let host = tok.host.as_ref().ok_or("store_token_host: host dropped")?;
        host.copy_from(&[f32::from_bits(token)], 0)?;
        Ok(())
    }

    /// 把采样参数写入 host-visible 缓冲（sampler）。CPU 侧 memcpy，无 kernel、无 spec constant；
    /// 采样 shader 直接从 device address 读取。8 元素：
    ///   [0]=temperature 位模式, [1]=top_k, [2]=top_p 位模式, [3]=seed,
    ///   [4]=repetition_penalty 位模式, [5]=frequency_penalty 位模式, [6]=presence_penalty 位模式,
    ///   [7]=hist_len。
    /// 打包约定：shader 对 [0,2,4,5,6] 用 `uintBitsToFloat` 读回 f32，故直接写入 f32 值；
    /// 对 [1,3,7] 用原生 uint 读，故用 `f32::from_bits` 保持原始 u32 位模式。
    #[allow(clippy::too_many_arguments)] // sampler 标量参数直接透传 shader，保持扁平
    pub fn store_sampler_host(
        &self,
        sampler: &GpuTensor,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        let host = sampler
            .host
            .as_ref()
            .ok_or("store_sampler_host: host dropped")?;
        host.copy_from(
            &[
                temperature,
                f32::from_bits(top_k),
                top_p,
                f32::from_bits(seed),
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                f32::from_bits(hist_len),
            ],
            0,
        )?;
        Ok(())
    }

    /// 记录 HOST_WRITE → COMPUTE(SHADER_READ) 内存屏障。
    /// CPU 直接写入 host-visible 缓冲后、shader 读取前调用，确保写入对 GPU 可见。
    /// （HOST_COHERENT 保证写后无需 flush，但执行顺序仍需显式 barrier。）
    fn host_write_to_shader_barrier(&mut self, buf: &vk::Buffer) -> R<()> {
        unsafe {
            let bmb = vk::BufferMemoryBarrier::builder()
                .buffer(*buf)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build();
            let bmbs = [bmb];
            self.app.device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &bmbs,
                &[] as &[vk::ImageMemoryBarrier],
            );
        }
        Ok(())
    }

    /// 参数化 embedding gather：从 src 表 [VOCAB, C]，按 host-visible token 索引缓冲
    /// （tok.host[0]，f32 位模式存 uint）gather 一行到 dst [C]。索引由 CPU 直接写入
    /// host 缓冲（无 kernel、无 spec constant），gather 前插入 HOST_WRITE→SHADER_READ
    /// 屏障保证可见性。循环体不依赖具体 token 值，为 GPU self-loop 铺路。
    /// （emb_ln 已 fp16 化，当前仅余诊断/回退用途）
    #[allow(dead_code)]
    pub fn gather_row_device(
        &mut self,
        src: &GpuTensor,
        dst: &mut GpuTensor,
        tok: &GpuTensor,
        c: usize,
    ) -> R<()> {
        let tok_host = tok
            .host
            .as_ref()
            .ok_or("gather_row_device: tok host dropped")?;
        let spec = [c as u32];
        let params = [tok_host.address, src.device.address, dst.device.address];
        self.host_write_to_shader_barrier(&tok_host.buffer)?;
        self.record_kernel(
            "shaders/spv/gather_row.spv",
            &spec,
            &params,
            (c.div_ceil(256) as u32, 1, 1),
            &[tok_host.buffer, src.device.buffer],
            &[dst.device.buffer],
        )?;
        Ok(())
    }

    /// fp16 源的 embedding gather：从 fp16 表 [VOCAB, C] 按 token 索引 gather 一行，
    /// 转 fp32 写入 dst [C]。用于 emb_ln fp16 化（省一半显存）后单 token decode 路径。
    pub fn gather_row_device_f16(
        &mut self,
        src: &GpuTensor16,
        dst: &mut GpuTensor,
        tok: &GpuTensor,
        c: usize,
    ) -> R<()> {
        let tok_host = tok
            .host
            .as_ref()
            .ok_or("gather_row_device_f16: tok host dropped")?;
        let spec = [c as u32];
        let params = [tok_host.address, src.device.address, dst.device.address];
        self.host_write_to_shader_barrier(&tok_host.buffer)?;
        self.record_kernel(
            "shaders/spv/gather_row_f16.spv",
            &spec,
            &params,
            (c.div_ceil(256) as u32, 1, 1),
            &[tok_host.buffer, src.device.buffer],
            &[dst.device.buffer],
        )?;
        Ok(())
    }

    /// 深度融合：gemv_rkv_f16（r/k/v 三个 C×C 投影）+ lowrank_stage1（v1/w1/a1/g1 四个 mid 投影）
    /// → 1 次 dispatch。dispatch (C + VM+WM+AM+GM, 1, 1)，前 C 个 workgroup 算 r/k/v（fp16 权重），
    /// 后 mid 个 workgroup 算 mid 投影（fp32 权重）。省 1 次 dispatch/层。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_rkv_stage1(
        &mut self,
        r16: &GpuTensor16,
        k16: &GpuTensor16,
        v16: &GpuTensor16,
        v1: &GpuTensor,
        w1: &GpuTensor,
        a1: &GpuTensor,
        g1: &GpuTensor,
        xr: &GpuTensor,
        xk: &GpuTensor,
        xv: &GpuTensor,
        xw: &GpuTensor,
        xa: &GpuTensor,
        xg: &GpuTensor,
        out_r: &mut GpuTensor,
        out_k: &mut GpuTensor,
        out_v: &mut GpuTensor16,
        out_vm: &mut GpuTensor,
        out_wm: &mut GpuTensor,
        out_am: &mut GpuTensor,
        out_gm: &mut GpuTensor,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        let spec = [
            c as u32,
            vm as u32,
            wm as u32,
            am as u32,
            gm as u32,
            self.app.properties.subgroup_size, // 5: SUBGROUP_SIZE（跨硬件自适应）
        ];
        let params = [
            r16.device.address,
            k16.device.address,
            v16.device.address,
            v1.device.address,
            w1.device.address,
            a1.device.address,
            g1.device.address,
            xr.device.address,
            xk.device.address,
            xv.device.address,
            xw.device.address,
            xa.device.address,
            xg.device.address,
            out_r.device.address,
            out_k.device.address,
            out_v.device.address,
            out_vm.device.address,
            out_wm.device.address,
            out_am.device.address,
            out_gm.device.address,
        ];
        let reads = vec![
            r16.device.buffer,
            k16.device.buffer,
            v16.device.buffer,
            v1.device.buffer,
            w1.device.buffer,
            a1.device.buffer,
            g1.device.buffer,
            xr.device.buffer,
            xk.device.buffer,
            xv.device.buffer,
            xw.device.buffer,
            xa.device.buffer,
            xg.device.buffer,
        ];
        let writes = vec![
            out_r.device.buffer,
            out_k.device.buffer,
            out_v.device.buffer,
            out_vm.device.buffer,
            out_wm.device.buffer,
            out_am.device.buffer,
            out_gm.device.buffer,
        ];
        // r/k/v 每 workgroup 处理 GEMV_ROWS 行（与 gemv_rkv_stage1.comp 的 ROWS 严格一致），
        // 后 VM+WM+AM+GM 个 workgroup 各算一个 mid 输出。
        debug_assert!(
            c.is_multiple_of(GEMV_ROWS),
            "C must be divisible by GEMV_ROWS"
        );
        self.record_kernel(
            "shaders/spv/gemv_rkv_stage1.spv",
            &spec,
            &params,
            ((c / GEMV_ROWS + vm + wm + am + gm) as u32, 1, 1),
            &reads,
            &writes,
        )?;
        Ok(())
    }

    /// 深度融合（int8 版）：gemv_int8 r/k/v（三个 C×C 投影，int8 量化权重）
    /// + lowrank_stage1（v1/w1/a1/g1 四个 mid 投影，fp32 权重）→ 1 次 dispatch。
    ///
    /// 与 gemv_rkv_stage1 同构，差异：r/k/v 用 int8 权重（无 LUT），带宽 fp16 的 ~50%。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_int8_rkv_stage1(
        &mut self,
        r_a8: &GpuTensorInt8,
        k_a8: &GpuTensorInt8,
        v_a8: &GpuTensorInt8,
        v1: &GpuTensor,
        w1: &GpuTensor,
        a1: &GpuTensor,
        g1: &GpuTensor,
        xr: &GpuTensor,
        xk: &GpuTensor,
        xv: &GpuTensor,
        xw: &GpuTensor,
        xa: &GpuTensor,
        xg: &GpuTensor,
        out_r: &mut GpuTensor,
        out_k: &mut GpuTensor,
        out_v: &mut GpuTensor16,
        out_vm: &mut GpuTensor,
        out_wm: &mut GpuTensor,
        out_am: &mut GpuTensor,
        out_gm: &mut GpuTensor,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        debug_assert_eq!((r_a8.m, r_a8.k), (c, c));
        debug_assert_eq!((k_a8.m, k_a8.k), (c, c));
        debug_assert_eq!((v_a8.m, v_a8.k), (c, c));
        let spec = [
            c as u32,
            vm as u32,
            wm as u32,
            am as u32,
            gm as u32,
            self.app.properties.subgroup_size, // 5: SUBGROUP_SIZE（跨硬件自适应）
        ];
        let params = [
            r_a8.idx.device.address,
            r_a8.sz.device.address,
            k_a8.idx.device.address,
            k_a8.sz.device.address,
            v_a8.idx.device.address,
            v_a8.sz.device.address,
            v1.device.address,
            w1.device.address,
            a1.device.address,
            g1.device.address,
            xr.device.address,
            xk.device.address,
            xv.device.address,
            xw.device.address,
            xa.device.address,
            xg.device.address,
            out_r.device.address,
            out_k.device.address,
            out_v.device.address,
            out_vm.device.address,
            out_wm.device.address,
            out_am.device.address,
            out_gm.device.address,
        ];
        let reads = vec![
            r_a8.idx.device.buffer,
            r_a8.sz.device.buffer,
            k_a8.idx.device.buffer,
            k_a8.sz.device.buffer,
            v_a8.idx.device.buffer,
            v_a8.sz.device.buffer,
            v1.device.buffer,
            w1.device.buffer,
            a1.device.buffer,
            g1.device.buffer,
            xr.device.buffer,
            xk.device.buffer,
            xv.device.buffer,
            xw.device.buffer,
            xa.device.buffer,
            xg.device.buffer,
        ];
        let writes = vec![
            out_r.device.buffer,
            out_k.device.buffer,
            out_v.device.buffer,
            out_vm.device.buffer,
            out_wm.device.buffer,
            out_am.device.buffer,
            out_gm.device.buffer,
        ];
        debug_assert!(
            c.is_multiple_of(GEMV_ROWS),
            "C must be divisible by GEMV_ROWS"
        );
        self.record_kernel(
            "shaders/spv/gemv_int8_rkv_stage1.spv",
            &spec,
            &params,
            ((c / GEMV_ROWS + vm + wm + am + gm) as u32, 1, 1),
            &reads,
            &writes,
        )?;
        Ok(())
    }

    /// 融合 4 条低秩链第二级（w2/a2/g2/v2）→ 1 次 dispatch。
    /// 同时计算 w/a/g/v 的二级投影 + 激活 + v 链原地 lerp（v 原地写）。
    /// uniform 字段顺序必须与 gemv_lowrank_chain4.comp 的 Params 结构一致。
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_lowrank_chain4(
        &mut self,
        w2: &GpuTensor,
        a2: &GpuTensor,
        v2: &GpuTensor,
        g2: &GpuTensor,
        w_mid: &GpuTensor,
        a_mid: &GpuTensor,
        v_mid: &GpuTensor,
        g_mid: &GpuTensor,
        w0: &GpuTensor,
        a0: &GpuTensor,
        v0: &GpuTensor,
        scale: &GpuTensor,
        v_first: &GpuTensor16,
        out_w: &mut GpuTensor16,
        out_a: &mut GpuTensor16,
        out_v: &mut GpuTensor16,
        out_g: &mut GpuTensor16,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
    ) -> R<()> {
        let spec = [
            m as u32,
            kw as u32,
            ka as u32,
            kv as u32,
            kg as u32,
            self.app.properties.subgroup_size, // 5: SUBGROUP_SIZE（跨硬件自适应）
        ];
        let params = [
            w2.device.address,
            a2.device.address,
            v2.device.address,
            g2.device.address,
            w_mid.device.address,
            a_mid.device.address,
            v_mid.device.address,
            g_mid.device.address,
            w0.device.address,
            a0.device.address,
            v0.device.address,
            scale.device.address,
            v_first.device.address,
            out_w.device.address,
            out_a.device.address,
            out_v.device.address,
            out_g.device.address,
        ];
        let reads = vec![
            w2.device.buffer,
            a2.device.buffer,
            v2.device.buffer,
            g2.device.buffer,
            w_mid.device.buffer,
            a_mid.device.buffer,
            v_mid.device.buffer,
            g_mid.device.buffer,
            w0.device.buffer,
            a0.device.buffer,
            v0.device.buffer,
            scale.device.buffer,
            v_first.device.buffer,
            out_v.device.buffer, // v 链原地写：既读又写
        ];
        let writes = vec![
            out_w.device.buffer,
            out_a.device.buffer,
            out_v.device.buffer,
            out_g.device.buffer,
        ];
        // subgroup-per-row：每 workgroup 处理 BLOCK_SIZE/subgroup_size 个输出行
        // （shader: row = gl_WorkGroupID.x * NUM_SUBGROUPS + gl_SubgroupID，BLOCK_SIZE=256）。
        // ⚠ 双锚点：dispatch 网格必须与 shader 行映射同源（2026-08-09 两次静默错误教训），
        // 此处 rows_per_wg 与 shader 的 NUM_SUBGROUPS 均 = 256/subgroup_size。
        let rows_per_wg = 256 / self.app.properties.subgroup_size as usize;
        assert!(
            m.is_multiple_of(rows_per_wg),
            "gemv_lowrank_chain4: M={m} 不整除 rows_per_wg={rows_per_wg}"
        );
        self.record_kernel(
            "shaders/spv/gemv_lowrank_chain4.spv",
            &spec,
            &params,
            ((m / rows_per_wg) as u32, 1, 1),
            &reads,
            &writes,
        )?;
        Ok(())
    }

    // ===== Sequence-parallel GEMV 算子（权重/偏置跨 token 共享） =====

    /// 通用 sequence-parallel gemv：dispatch (M, T, 1)。
    /// b 为 None 时传 0 地址（非 affine 变体），否则当作偏置（affine 变体）。
    /// x_stride / y_stride 为输入/输出 [T, *] token 步长。
    #[allow(clippy::too_many_arguments)]
    fn gemv_seq_impl(
        &mut self,
        shader: &str,
        a: &GpuTensor,
        b: Option<&GpuTensor>,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        x_stride: usize,
        y_stride: usize,
        batch: usize,
    ) -> R<()> {
        let spec = gemv_seq_spec(&self.app, m, k, x_stride, y_stride);
        let b_addr = b.map_or(0, |t| t.device.address);
        let params = [a.device.address, b_addr, x.device.address, y.device.address];
        let mut reads = vec![a.device.buffer, x.device.buffer];
        if let Some(bt) = b {
            reads.push(bt.device.buffer);
        }
        self.record_kernel(
            shader,
            &spec,
            &params,
            (m as u32, batch as u32, 1),
            &reads,
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = x @ A（sequence-parallel，权重共享）
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_seq(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        k: usize,
        x_stride: usize,
        y_stride: usize,
        batch: usize,
    ) -> R<()> {
        self.gemv_seq_impl(
            "shaders/spv/gemv_f32_f32.spv",
            a,
            None,
            x,
            y,
            m,
            k,
            x_stride,
            y_stride,
            batch,
        )
    }

    // ===== fp16 转换 + tensor-core GEMM（fp32io16 模式） =====

    /// f32 → f16 转换（sequence-parallel，token 并行）。
    /// 把 fp32 激活 [T, C] 转成 fp16 [M_PAD, C]，供 tensor-core GEMM 使用。
    /// M_PAD = ceil(T / TILE_M) * TILE_M；非对齐 token（token >= T）写 0，保证 GEMM 填充行输出为 0。
    /// dispatch: (M_PAD, 1, 1)。
    #[allow(clippy::too_many_arguments, clippy::wrong_self_convention)]
    pub fn to_f16(
        &mut self,
        x: &GpuTensor,
        y: &mut GpuTensor16,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let spec = [c as u32, t as u32, x_stride as u32, y_stride as u32];
        let params = [x.device.address, y.device.address];
        self.record_kernel(
            "shaders/spv/to_f16.spv",
            &spec,
            &params,
            (m_pad as u32, 1, 1),
            &[x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// f32 → f16 三输入融合转换（sequence-parallel，token 并行）。
    /// 一次 launch 同时把 xr/xk/xv 三个 [T, C] fp32 转成 fp16 [M_PAD, C]，
    /// 减少三次 to_f16 kernel launch 开销。spec 与单个 to_f16 相同：
    /// [C, T, STRIDE_X, STRIDE_Y]，三个输入/输出共享相同的 C 与步长。
    #[allow(clippy::too_many_arguments, clippy::wrong_self_convention)]
    pub fn to_f16_triple(
        &mut self,
        xr: &GpuTensor,
        xk: &GpuTensor,
        xv: &GpuTensor,
        yr: &mut GpuTensor16,
        yk: &mut GpuTensor16,
        yv: &mut GpuTensor16,
        c: usize,
        t: usize,
        m_pad: usize,
        x_stride: usize,
        y_stride: usize,
    ) -> R<()> {
        let spec = [c as u32, t as u32, x_stride as u32, y_stride as u32];
        let params = [
            xr.device.address,
            xk.device.address,
            xv.device.address,
            yr.device.address,
            yk.device.address,
            yv.device.address,
        ];
        self.record_kernel(
            "shaders/spv/to_f16_triple.spv",
            &spec,
            &params,
            (m_pad as u32, 1, 1),
            &[xr.device.buffer, xk.device.buffer, xv.device.buffer],
            &[yr.device.buffer, yk.device.buffer, yv.device.buffer],
        )?;
        Ok(())
    }

    /// tensor-core GEMM：C[M,N] = A[M,K] @ B[N,K]^T（fp32io16）。
    ///
    /// A/B 为 fp16，累加与输出为 fp32。A 为激活 [M, K]（行主序，M=token 数），
    /// B 为权重 [N, K]（行主序，即 PyTorch nn.Linear 的 [out, in] 布局），
    /// 输出 C = A @ B^T 为 [M, N]（行主序）。
    ///
    /// 要求 M/N 为 TILE_M=256 / TILE_N=256 的整数倍，K 为 TILE_K=32 的整数倍。
    /// dispatch: (M/256, N/256, 1)。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &mut self,
        a: &GpuTensor16,
        b: &GpuTensor16,
        c: &mut GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let spec = gemm_spec(&self.app, m, n, k);
        let params = [a.device.address, b.device.address, 0, c.device.address];
        let (gx, gy) = gemm_grid(&self.app, m, n);
        self.record_kernel(
            "shaders/spv/gemm_f16_f32.spv",
            &spec,
            &params,
            (gx, gy, 1),
            &[a.device.buffer, b.device.buffer],
            &[c.device.buffer],
        )?;
        Ok(())
    }

    /// tensor-core GEMM with residual addition：C[M,N] = (A[M,K] @ B[N,K]^T) + x[M,N]（fp32io16）。
    /// 融合残差相加（输出投影：y_out = output_w @ y_g + x），省去单独 elementwise_add kernel launch。
    /// x 既是输入也是输出，直接原地累加。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_add(
        &mut self,
        a: &GpuTensor16,
        b: &GpuTensor16,
        x: &GpuTensor,
        y: &mut GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let spec = gemm_spec(&self.app, m, n, k);
        let params = [
            a.device.address,
            b.device.address,
            x.device.address,
            y.device.address,
        ];
        let (gx, gy) = gemm_grid(&self.app, m, n);
        self.record_kernel(
            "shaders/spv/gemm_f16_f32_affine.spv",
            &spec,
            &params,
            (gx, gy, 1),
            &[a.device.buffer, b.device.buffer, x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// tensor-core GEMM + relu² 激活：C[M,N] = relu²(A[M,K] @ B[N,K]^T)（fp32io16）。
    /// 等价于 gemv_relu2_seq，用于 ffn_key 投影。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_relu2(
        &mut self,
        a: &GpuTensor16,
        b: &GpuTensor16,
        c: &mut GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let spec = gemm_spec(&self.app, m, n, k);
        let params = [a.device.address, b.device.address, 0, c.device.address];
        let (gx, gy) = gemm_grid(&self.app, m, n);
        self.record_kernel(
            "shaders/spv/gemm_f16_f32_relu2.spv",
            &spec,
            &params,
            (gx, gy, 1),
            &[a.device.buffer, b.device.buffer],
            &[c.device.buffer],
        )?;
        Ok(())
    }

    /// tensor-core GEMM + bias 向量：C[M,N] = A[M,K] @ B[N,K]^T + bias[N]（fp32io16）。
    /// bias 按输出列索引（行广播），用于低秩投影的仿射（w1/w2/a1/a2/v1/v2/g1/g2）。
    /// 注意：bias 不能在 gemm.comp 内用 coopmat 下标 result[i] 施加——coopmat 的 [i] 访问
    /// 按硬件 tiled 布局（非行主序），无法用 i%N 推断列。故用独立 elementwise kernel
    /// （gemm_bias.comp）在行主序 [M,N] 输出上直接定位列加 bias。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bias(
        &mut self,
        a: &GpuTensor16,
        b: &GpuTensor16,
        bias: &GpuTensor,
        c: &mut GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        // 第一步：纯 tensor-core GEMM
        self.gemm(a, b, c, m, n, k)?;
        // 第二步：C[m,n] += bias[n]（独立 kernel，行主序定位列）
        let spec = [m as u32, n as u32, n as u32];
        let params = [c.device.address, bias.device.address];
        self.record_kernel(
            "shaders/spv/gemm_bias.spv",
            &spec,
            &params,
            ((m * n / 256).max(1) as u32, 1, 1),
            &[c.device.buffer, bias.device.buffer],
            &[c.device.buffer],
        )?;
        Ok(())
    }

    /// tensor-core GEMM + tanh 激活：C[M,N] = tanh(A[M,K] @ B[N,K]^T)（fp32io16）。
    /// 用于低秩第一级投影（w = tanh(xw @ w1)）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_tanh(
        &mut self,
        a: &GpuTensor16,
        b: &GpuTensor16,
        c: &mut GpuTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> R<()> {
        let spec = gemm_spec(&self.app, m, n, k);
        let params = [a.device.address, b.device.address, 0, c.device.address];
        let (gx, gy) = gemm_grid(&self.app, m, n);
        self.record_kernel(
            "shaders/spv/gemm_f16_f32_tanh.spv",
            &spec,
            &params,
            (gx, gy, 1),
            &[a.device.buffer, b.device.buffer],
            &[c.device.buffer],
        )?;
        Ok(())
    }

    // ===== Norm 算子 =====

    /// Layer/Group Norm with affine（gamma、beta 跨 batch 共享）
    /// dispatch: (H, batch, 1)
    #[allow(clippy::too_many_arguments)]
    pub fn norm(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        h: usize,
        eps: f32,
        batch: usize,
    ) -> R<()> {
        let spec = norm_spec(c, h, eps, self.app.properties.subgroup_size);
        let params = [
            x.device.address,
            gamma.device.address,
            beta.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/norm_f32_f32_affine.spv",
            &spec,
            &params,
            (h as u32, batch as u32, 1),
            &[x.device.buffer, gamma.device.buffer, beta.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// 深度融合：ln1 = layer_norm(x) + 6 次 lerp(xr/xw/xk/xv/xa/xg) + state.tmix_x 写回。
    /// 单 token（batch=1, H=1）路径，一次 dispatch 完成。state 既是 prev_x 读取又是 tmix_x 写回。
    #[allow(clippy::too_many_arguments)]
    pub fn norm_lerp6(
        &mut self,
        x: &GpuTensor,
        state: &mut GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        x_r: &GpuTensor,
        x_w: &GpuTensor,
        x_k: &GpuTensor,
        x_v: &GpuTensor,
        x_a: &GpuTensor,
        x_g: &GpuTensor,
        o_r: &mut GpuTensor,
        o_w: &mut GpuTensor,
        o_k: &mut GpuTensor,
        o_v: &mut GpuTensor,
        o_a: &mut GpuTensor,
        o_g: &mut GpuTensor,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let spec = [c as u32, eps.to_bits(), self.app.properties.subgroup_size];
        let params = [
            x.device.address,
            state.device.address,
            gamma.device.address,
            beta.device.address,
            x_r.device.address,
            x_w.device.address,
            x_k.device.address,
            x_v.device.address,
            x_a.device.address,
            x_g.device.address,
            o_r.device.address,
            o_w.device.address,
            o_k.device.address,
            o_v.device.address,
            o_a.device.address,
            o_g.device.address,
        ];
        self.record_kernel(
            "shaders/spv/norm_lerp6.spv",
            &spec,
            &params,
            (c.div_ceil(NORM_LERP6_BLOCK) as u32, 1, 1),
            &[
                x.device.buffer,
                state.device.buffer,
                gamma.device.buffer,
                beta.device.buffer,
                x_r.device.buffer,
                x_w.device.buffer,
                x_k.device.buffer,
                x_v.device.buffer,
                x_a.device.buffer,
                x_g.device.buffer,
            ],
            &[
                state.device.buffer,
                o_r.device.buffer,
                o_w.device.buffer,
                o_k.device.buffer,
                o_v.device.buffer,
                o_a.device.buffer,
                o_g.device.buffer,
            ],
        )?;
        Ok(())
    }

    /// 深度融合：cmix 的 ln2 = layer_norm(x) + prev_c 读入 + state.cmix_x 写回 + lerp(xb)。
    /// 单 token（batch=1, H=1）路径，一次 dispatch 完成。state 既是 prev_c 读取又是 cmix_x 写回。
    #[allow(clippy::too_many_arguments)]
    pub fn cmix_norm_lerp(
        &mut self,
        x: &GpuTensor,
        state: &mut GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        coeff: &GpuTensor,
        out_xb: &mut GpuTensor,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let spec = [c as u32, eps.to_bits(), self.app.properties.subgroup_size];
        let params = [
            x.device.address,
            state.device.address,
            gamma.device.address,
            beta.device.address,
            coeff.device.address,
            out_xb.device.address,
        ];
        self.record_kernel(
            "shaders/spv/cmix_norm_lerp.spv",
            &spec,
            &params,
            (1, 1, 1),
            &[
                x.device.buffer,
                state.device.buffer,
                gamma.device.buffer,
                beta.device.buffer,
                coeff.device.buffer,
            ],
            &[state.device.buffer, out_xb.device.buffer],
        )?;
        Ok(())
    }

    // ===== Elementwise 算子 =====
    // dispatch: (batch, 1, 1) — 每个工作组（token）处理其 C 个元素

    /// 单输入 elementwise 操作（input_b、input_c 指向 a 自身）
    fn elementwise_unary(
        &mut self,
        shader: &str,
        a: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let spec = elementwise_spec(c);
        let addr = a.device.address;
        let params = [addr, addr, addr, y.device.address];
        self.record_kernel(
            shader,
            &spec,
            &params,
            (batch as u32, 1, 1),
            &[a.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// 双输入 elementwise 操作（input_c 指向 a 自身）
    fn elementwise_binary(
        &mut self,
        shader: &str,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let spec = elementwise_spec(c);
        let params = [
            a.device.address,
            b.device.address,
            a.device.address,
            y.device.address,
        ];
        self.record_kernel(
            shader,
            &spec,
            &params,
            (batch as u32, 1, 1),
            &[a.device.buffer, b.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    pub fn elementwise_sigmoid(
        &mut self,
        a: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_unary(
            "shaders/spv/elementwise_f32_f32_sigmoid.spv",
            a,
            y,
            c,
            batch,
        )
    }

    /// 原地 sigmoid（a 与 y 为同一张量，避免 &/&mut 双借用）
    /// 单输入 OP=1 各线程仅读写自己 i 位置，原地安全。
    pub fn elementwise_sigmoid_inplace(
        &mut self,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let spec = elementwise_spec(c);
        let addr = y.device.address;
        let params = [addr, addr, addr, addr];
        self.record_kernel(
            "shaders/spv/elementwise_f32_f32_sigmoid.spv",
            &spec,
            &params,
            (batch as u32, 1, 1),
            &[y.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn elementwise_exp(
        &mut self,
        a: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_unary("shaders/spv/elementwise_f32_f32_exp.spv", a, y, c, batch)
    }

    #[allow(dead_code)]
    pub fn elementwise_tanh(
        &mut self,
        a: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_unary("shaders/spv/elementwise_f32_f32_tanh.spv", a, y, c, batch)
    }

    #[allow(dead_code)]
    pub fn elementwise_neg(
        &mut self,
        a: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_unary("shaders/spv/elementwise_f32_f32_neg.spv", a, y, c, batch)
    }

    pub fn elementwise_mul(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_binary("shaders/spv/elementwise_f32_f32_mul.spv", a, b, y, c, batch)
    }

    #[allow(dead_code)]
    pub fn elementwise_mul_neg(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        self.elementwise_binary(
            "shaders/spv/elementwise_f32_f32_mul_neg.spv",
            a,
            b,
            y,
            c,
            batch,
        )
    }

    /// y = a * b[0]（b 为每 batch 一个标量；input_c 指向 a 自身）
    #[allow(dead_code)]
    pub fn elementwise_scale(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let spec = elementwise_spec(c);
        let addr = a.device.address;
        let params = [a.device.address, b.device.address, addr, y.device.address];
        self.record_kernel(
            "shaders/spv/elementwise_f32_f32_scale.spv",
            &spec,
            &params,
            (batch as u32, 1, 1),
            &[a.device.buffer, b.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// y = exp(a * b[0])（input_c 指向 a 自身）
    pub fn elementwise_scale_exp(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let spec = elementwise_spec(c);
        let addr = a.device.address;
        let params = [a.device.address, b.device.address, addr, y.device.address];
        self.record_kernel(
            "shaders/spv/elementwise_f32_f32_scale_exp.spv",
            &spec,
            &params,
            (batch as u32, 1, 1),
            &[a.device.buffer, b.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    // ===== DPLR 算子 =====

    // ==== k/k_a 融合算子 ====

    /// 融合 kk=l2_norm(k*k_k)、k_mod=lerp、b=-kk_l2*a 为一个 kernel
    /// 一次性输出 k_mod / kk_l2 / b，替代原 4-5 次 launch 与多轮 read/write。
    /// dispatch: (H, batch, 1) — 每个工作组处理一个 (batch, head)
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_ka(
        &mut self,
        k: &GpuTensor,
        k_k: &GpuTensor,
        a: &GpuTensor,
        k_a: &GpuTensor,
        k_mod: &mut GpuTensor,
        kk_l2: &mut GpuTensor,
        b: &mut GpuTensor,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let spec = fuse_ka_spec(h, n, self.app.properties.subgroup_size);
        let params = [
            k.device.address,
            k_k.device.address,
            a.device.address,
            k_a.device.address,
            k_mod.device.address,
            kk_l2.device.address,
            b.device.address,
        ];
        self.record_kernel(
            "shaders/spv/fuse_ka.spv",
            &spec,
            &params,
            (h as u32, batch as u32, 1),
            &[
                k.device.buffer,
                k_k.device.buffer,
                a.device.buffer,
                k_a.device.buffer,
            ],
            &[k_mod.device.buffer, kk_l2.device.buffer, b.device.buffer],
        )?;
        Ok(())
    }

    /// 融合 fuse_ka + dplr + group_norm + sum_rk_rk（单 token 路径）：
    /// 一次 dispatch 完成 k_mod/kk_l2/b、S 更新、y=S@r、y_norm=group_norm(y)+sum(r*k_mod*r_k)*v。
    /// 替代 fuse_ka_dplr + norm_sum_rk_rk 两次 dispatch，省 1 次/层。
    /// uniform 字段顺序必须与 fuse_ka_dplr_norm.comp 的 Params 结构一致。
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_ka_dplr_norm(
        &mut self,
        s: &mut GpuTensor,
        k: &GpuTensor,
        k_k: &GpuTensor,
        a: &GpuTensor16,
        k_a: &GpuTensor,
        r: &GpuTensor,
        v: &GpuTensor16,
        w: &GpuTensor16,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        r_k: &GpuTensor,
        k_mod: &mut GpuTensor,
        y: &mut GpuTensor,
        y_norm: &mut GpuTensor,
        h: usize,
        n: usize,
        eps: f32,
        gn_eps: f32,
    ) -> R<()> {
        // ⚠ 双锚点：spec 第 5 项（constant_id=4）= SUBGROUP_SIZE，与 shader 的
        // subgroup-per-row 行/列分派同源（NUM_SUBGROUPS = 128/SUBGROUP_SIZE）。
        let spec = [
            h as u32,
            n as u32,
            eps.to_bits(),
            gn_eps.to_bits(),
            self.app.properties.subgroup_size,
        ];
        let params = [
            s.device.address,
            k.device.address,
            k_k.device.address,
            a.device.address,
            k_a.device.address,
            r.device.address,
            v.device.address,
            w.device.address,
            k_mod.device.address,
            y.device.address,
            gamma.device.address,
            beta.device.address,
            r_k.device.address,
            y_norm.device.address,
        ];
        self.record_kernel(
            "shaders/spv/fuse_ka_dplr_norm.spv",
            &spec,
            &params,
            (h as u32, 1, 1),
            &[
                s.device.buffer,
                k.device.buffer,
                k_k.device.buffer,
                a.device.buffer,
                k_a.device.buffer,
                r.device.buffer,
                v.device.buffer,
                w.device.buffer,
                gamma.device.buffer,
                beta.device.buffer,
                r_k.device.buffer,
            ],
            &[
                s.device.buffer,
                k_mod.device.buffer,
                y.device.buffer,
                y_norm.device.buffer,
            ],
        )?;
        Ok(())
    }

    // ===== sum_rk_rk 归约算子 =====
    /// y += sum(r * k_mod * r_k, 按 head 归约) * v
    /// 替代原 download → CPU 循环 → upload 的 PCIe 往返。
    /// dispatch: (H, batch, 1) — 每个工作组处理一个 (batch, head)
    #[allow(clippy::too_many_arguments)]
    pub fn sum_rk_rk(
        &mut self,
        r: &GpuTensor,
        k_mod: &GpuTensor,
        r_k: &GpuTensor,
        v: &GpuTensor,
        y: &mut GpuTensor,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let spec = sum_rk_rk_spec(h, n, self.app.properties.subgroup_size);
        let params = [
            r.device.address,
            k_mod.device.address,
            r_k.device.address,
            v.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/sum_rk_rk.spv",
            &spec,
            &params,
            (h as u32, batch as u32, 1),
            &[
                r.device.buffer,
                k_mod.device.buffer,
                r_k.device.buffer,
                v.device.buffer,
                y.device.buffer,
            ],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    // ===== Sequence-parallel 算子 =====
    /// token shift + time-mix 插值（sequence-parallel）：
    #[allow(clippy::too_many_arguments)]
    pub fn seq_shift(
        &mut self,
        x: &GpuTensor,
        state: &GpuTensor,
        tm: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> R<()> {
        let spec = seq_shift_spec(c, t, stride_x, stride_y);
        let params = [
            x.device.address,
            state.device.address,
            tm.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/seq_shift_f32_f32.spv",
            &spec,
            &params,
            (t as u32, 1, 1),
            &[x.device.buffer, state.device.buffer, tm.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// v_first 混合（sequence-parallel）：v[t] = v[t] + gate[t]*(v_first - v[t])，t>=1
    /// dispatch: (T, 1, 1)
    pub fn v_first_lerp(
        &mut self,
        v: &GpuTensor,
        gate: &GpuTensor,
        v_first: &GpuTensor,
        c: usize,
        t: usize,
        stride: usize,
    ) -> R<()> {
        let spec = [
            c as u32,      // 0: C
            t as u32,      // 1: T
            stride as u32, // 2: STRIDE_V
        ];
        let params = [
            v.device.address,
            gate.device.address,
            v_first.device.address,
        ];
        self.record_kernel(
            "shaders/spv/v_first_lerp_f32_f32.spv",
            &spec,
            &params,
            (t as u32, 1, 1),
            &[v.device.buffer, gate.device.buffer, v_first.device.buffer],
            &[v.device.buffer],
        )?;
        Ok(())
    }

    /// 拷贝一个 token 行到 [C] 状态缓冲（状态更新用）
    /// dispatch: (1, 1, 1)
    pub fn copy_token(
        &mut self,
        x: &GpuTensor,
        y: &mut GpuTensor,
        c: usize,
        stride: usize,
        token: usize,
    ) -> R<()> {
        let spec = [
            c as u32,      // 0: C
            stride as u32, // 1: STRIDE
            token as u32,  // 2: TOKEN
        ];
        let params = [x.device.address, y.device.address];
        self.record_kernel(
            "shaders/spv/copy_token_f32_f32.spv",
            &spec,
            &params,
            (1, 1, 1),
            &[x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }

    /// DPLR 状态更新（sequence-parallel，内部循环 T）：
    ///   S 在寄存器中跨 token 传递，单次 launch 处理整个序列。
    /// dispatch: (H, 1, 1)
    #[allow(clippy::too_many_arguments)]
    pub fn dplr_seq(
        &mut self,
        s: &mut GpuTensor,
        r: &GpuTensor,
        w: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &mut GpuTensor,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
    ) -> R<()> {
        let spec = dplr_seq_spec(h, n, t, c);
        let params = [
            s.device.address,
            r.device.address,
            w.device.address,
            k.device.address,
            v.device.address,
            a.device.address,
            b.device.address,
            y.device.address,
        ];
        self.record_kernel(
            "shaders/spv/dplr_seq_f32_f32.spv",
            &spec,
            &params,
            (h as u32, 1, 1),
            &[
                s.device.buffer,
                r.device.buffer,
                w.device.buffer,
                k.device.buffer,
                v.device.buffer,
                a.device.buffer,
                b.device.buffer,
            ],
            &[s.device.buffer, y.device.buffer],
        )?;
        Ok(())
    }

    // ===== L2 Norm 算子 =====

    /// 按 head 做 L2 normalize
    /// dispatch: (H, batch=1, 1)
    #[allow(dead_code)]
    pub fn l2_norm(&mut self, x: &GpuTensor, y: &mut GpuTensor, h: usize, n: usize) -> R<()> {
        let spec = l2_norm_spec(h, n, self.app.properties.subgroup_size);
        let params = [x.device.address, y.device.address];
        self.record_kernel(
            "shaders/spv/l2_norm_f32_f32.spv",
            &spec,
            &params,
            (h as u32, 1, 1),
            &[x.device.buffer],
            &[y.device.buffer],
        )?;
        Ok(())
    }
}

// ===== Specialization constant 构造器 =====

/// gemv specialization (constant_id 0-10)
/// 每个 workgroup 处理的行数（与 gemv_f32io.comp 的 ROWS 一致）。要求 M % GEMV_ROWS == 0。
/// 注意：ROWS 是 gemv_f32io.comp 的编译期常量（=4），无法从 runtime 传入，故 GEMV_ROWS 必须与
/// shader 的 ROWS 严格一致。若两者不一致（如 ROWS=4 而 GEMV_ROWS=2），dispatch 的 workgroup 数
/// 会与 shader 实际处理的行数错位，导致越界访问并触发 DEVICE_LOST。
const GEMV_ROWS: usize = 4;

/// gemv 每 workgroup 处理行数：实测结论（2026-08-09, RTX 2080 Ti）固定 ROWS=4 最优。
/// 曾尝试按 M 自适应（大 M 如 ffn_key M=10240 用 ROWS=16 减 X 重读），实测放弃：
///   - X 重读理论不成立：ROWS=4 时 X 重读总量 = 2560 wg × 10KB = 25MB，摊到内核时长
///     （ffn_key fp16 ~0.126ms）仅占 L2 带宽零头；X（10KB）常驻 L2，无 DRAM 回取问题。
///   - ROWS=16 在 int8 kernel 引发寄存器压力（acc[16] + 解包数学），decode 吞吐反降 ~12%。
///   - 热降频噪声（同内核跨 token 波动可达 2×）曾导致 "18% 带宽利用率" 的误判。
///
/// 保留 constant_id=12 管线仅为实验便利；rows 必须 == shader ROWS，否则 grid 错位算错结果
/// （如 head gemv M=65536 若 rows=16 而 shader ROWS=4 → 仅 1/4 行被计算，logits 错误）。
fn gemv_rows_for(m: usize) -> usize {
    let _ = m;
    GEMV_ROWS
}

/// norm_lerp6 的 workgroup 大小（须与 norm_lerp6.comp 的 BLOCK_SIZE 一致）。
/// dispatch 网格 = ceil(C / BLOCK_SIZE)，把 apply 并行铺到多个 workgroup / SM。
const NORM_LERP6_BLOCK: usize = 256;

/// A 存储为 [M, K] 行主序：A_stored[m, k] = m*K + k
/// shader 逻辑 A[k, m] = k*STRIDE_A_X + m*STRIDE_A_Y = k + m*K ✓
/// gemv specialization (constant_id 0-10, 11: SUBGROUP_SIZE)
/// subgroup_size 为设备真实 subgroup（NVIDIA=32，AMD=64，Intel 可变），传入 gemv_f32io 家族
/// shader 的 constant_id=11，保证 NUM_SUBGROUPS = BLOCK_SIZE/SUBGROUP_SIZE 跨硬件正确。
fn gemv_spec(app: &App, m: usize, k: usize, rows: usize) -> [u32; 13] {
    [
        m as u32,                     // 0: M (output dim)
        k as u32,                     // 1: K (input dim)
        1,                            // 2: STRIDE_A_X
        k as u32,                     // 3: STRIDE_A_Y
        (m * k) as u32,               // 4: STRIDE_A_Z
        1,                            // 5: STRIDE_B_X (bias)
        m as u32,                     // 6: STRIDE_B_Y
        1,                            // 7: STRIDE_X_X
        k as u32,                     // 8: STRIDE_X_Y
        1,                            // 9: STRIDE_Y_X
        m as u32,                     // 10: STRIDE_Y_Y
        app.properties.subgroup_size, // 11: SUBGROUP_SIZE（跨硬件自适应）
        rows as u32,                  // 12: ROWS（每 workgroup 行数，大 M 用大 ROWS）
    ]
}

/// sequence-parallel gemv specialization（constant_id 0-10, 11: SUBGROUP_SIZE）
/// 与 gemv_spec 相同，但 A（权重）与偏置 B 跨 token 共享（STRIDE_A_Z=0, STRIDE_B_Y=0），
/// 输入 X 与输出 Y 为 [T, *]（token 主序），分别以 x_stride / y_stride 为 token 步长。
fn gemv_seq_spec(app: &App, m: usize, k: usize, x_stride: usize, y_stride: usize) -> [u32; 12] {
    [
        m as u32,                     // 0: M (output dim)
        k as u32,                     // 1: K (input dim)
        1,                            // 2: STRIDE_A_X
        k as u32,                     // 3: STRIDE_A_Y
        0,                            // 4: STRIDE_A_Z (权重跨 token 共享)
        1,                            // 5: STRIDE_B_X (bias)
        0,                            // 6: STRIDE_B_Y (偏置跨 token 共享)
        1,                            // 7: STRIDE_X_X
        x_stride as u32,              // 8: STRIDE_X_Y (token 步长)
        1,                            // 9: STRIDE_Y_X
        y_stride as u32,              // 10: STRIDE_Y_Y (token 步长)
        app.properties.subgroup_size, // 11: SUBGROUP_SIZE（跨硬件自适应）
    ]
}

/// norm specialization (constant_id 0-14, 15: SUBGROUP_SIZE)
/// gamma/beta 跨 batch 共享：STRIDE_G_Z = STRIDE_B_Z = 0
fn norm_spec(c: usize, h: usize, eps: f32, subgroup_size: u32) -> [u32; 16] {
    let eps_bits = eps.to_bits();
    [
        c as u32,       // 0: C (channel)
        h as u32,       // 1: H (heads)
        1,              // 2: STRIDE_X_X
        c as u32,       // 3: STRIDE_X_Y
        (c * h) as u32, // 4: STRIDE_X_Z
        1,              // 5: STRIDE_Y_X
        c as u32,       // 6: STRIDE_Y_Y
        (c * h) as u32, // 7: STRIDE_Y_Z
        1,              // 8: STRIDE_G_X
        c as u32,       // 9: STRIDE_G_Y
        0,              // 10: STRIDE_G_Z (gamma 跨 batch 共享)
        1,              // 11: STRIDE_B_X
        c as u32,       // 12: STRIDE_B_Y
        0,              // 13: STRIDE_B_Z (beta 跨 batch 共享)
        eps_bits,       // 14: EPSILON (f32::to_bits)
        subgroup_size,  // 15: SUBGROUP_SIZE（跨硬件自适应）
    ]
}

/// elementwise specialization (constant_id 0-8)
fn elementwise_spec(c: usize) -> [u32; 9] {
    [
        c as u32, // 0: C
        1,        // 1: STRIDE_A_X
        c as u32, // 2: STRIDE_A_Y
        1,        // 3: STRIDE_B_X
        c as u32, // 4: STRIDE_B_Y
        1,        // 5: STRIDE_C_X
        c as u32, // 6: STRIDE_C_Y
        1,        // 7: STRIDE_Y_X
        c as u32, // 8: STRIDE_Y_Y
    ]
}

/// l2_norm specialization (constant_id 0-7, 8: SUBGROUP_SIZE)
/// X/Y shape: (N, H, batch)，stride: (1, N, H*N)
#[allow(dead_code)]
fn l2_norm_spec(h: usize, n: usize, subgroup_size: u32) -> [u32; 9] {
    [
        h as u32,       // 0: H (heads)
        n as u32,       // 1: N (head_size)
        1,              // 2: STRIDE_X_X
        n as u32,       // 3: STRIDE_X_Y
        (h * n) as u32, // 4: STRIDE_X_Z
        1,              // 5: STRIDE_Y_X
        n as u32,       // 6: STRIDE_Y_Y
        (h * n) as u32, // 7: STRIDE_Y_Z
        subgroup_size,  // 8: SUBGROUP_SIZE（跨硬件自适应）
    ]
}

/// 估算单个 kernel 的读写字节数（用于带宽利用率 = est_bytes / 耗时 / 峰值带宽）。
/// 依据 shader 名 + 特化常量 + dispatch 网格分类估算，仅供 PROF_GPU 性能排查定位
/// 内存受限 vs 计算受限 vs 启动开销，非精确值。
fn est_kernel_bytes(shader: &str, spec: &[u32], dispatch: (u32, u32, u32)) -> u64 {
    let name = shader.rsplit('/').next().unwrap_or(shader);
    let s = |i: usize| spec.get(i).copied().unwrap_or(0) as usize;
    let (gx, gy, _gz) = dispatch;

    // GEMM：A f16[M,K] + B f16[N,K] + C f32[M,N]；_add/_affine 额外读旧 C
    if name.starts_with("gemm_") {
        let (m, n, k) = (s(6), s(7), s(8));
        let mut bytes = 2 * m * k + 2 * n * k + 4 * m * n;
        if name.contains("_add") || name.contains("_affine") {
            bytes += 4 * m * n;
        }
        return bytes as u64;
    }
    // fp16 r/k/v 深度融合：spec 同上。r/k/v fp16 权重 3×C·C·2B，其余同上。
    if name.starts_with("gemv_rkv_stage1") {
        let (c, vm, wm, am, gm) = (s(0), s(1), s(2), s(3), s(4));
        let mid = vm + wm + am + gm;
        let bytes = 3 * (2 * c * c) + mid * c * 4 + 6 * c * 4 + 2 * c * 4 + c * 2 + mid * 4;
        return bytes as u64;
    }
    // GEMV：A f16[M,K] + X f32[K] + Y f32[M]；_add/_add_mul 额外读旧 Y
    if name.starts_with("gemv_") {
        let (m, k) = (s(0), s(1));
        let mut bytes = 2 * m * k + 4 * k + 4 * m;
        if (name.contains("_add") || name.contains("_add_mul")) && !name.contains("_relu2") {
            bytes += 4 * m;
        }
        return bytes as u64;
    }
    // elementwise：输入(1~2 个 f32 张量) + 输出 f32 存取，dispatch.x = token 数
    if name.starts_with("elementwise_") {
        let tot = s(0) * (gx as usize);
        let inputs = if name.contains("_mul") || name.contains("_add") {
            2
        } else {
            1
        };
        return (4 * tot * (inputs + 1)) as u64;
    }
    // 深度融合 norm（norm_lerp6 / cmix_norm_lerp）：spec=[c, eps]，多张量 f32 读写
    if name.starts_with("norm_lerp") || name.starts_with("cmix_norm") {
        return (4 * s(0) * 16) as u64;
    }
    // 普通 norm：X + gamma + beta 读 + Y 写，dispatch.x = H
    if name.starts_with("norm_") {
        let tot = s(0) * s(1) * (gx as usize);
        return (4 * tot * 4) as u64;
    }
    // fuse_ka / fuse_ka_dplr_norm：spec=[H, N]，读写 k/v/r/w/state 等约十多个张量
    if name.starts_with("fuse_ka") {
        return (4 * s(0) * s(1) * 12) as u64;
    }
    // to_f16 / to_f16_triple：读 f32 写 f16
    if name.starts_with("to_f16") {
        return (4 * s(0) + 2 * s(0)) as u64;
    }
    // argmax：读 N 个 logits，只写 4B
    if name.starts_with("argmax") {
        return (4 * s(0)) as u64;
    }
    // 兜底：dispatch 网格 × 每工作组 256 线程 × 4B 粗略估算
    (gx as u64) * (gy as u64).max(1) * 1024
}

/// fuse_ka specialization (constant_id 0: H, 1: N, 2: EPSILON, 3: SUBGROUP_SIZE)
/// 每个工作组处理一个 head（N 个元素）
fn fuse_ka_spec(h: usize, n: usize, subgroup_size: u32) -> [u32; 4] {
    let eps = 1.0e-12f32;
    [h as u32, n as u32, eps.to_bits(), subgroup_size]
}

/// sum_rk_rk specialization (constant_id 0: H, 1: N, 2: SUBGROUP_SIZE)
/// 每个工作组处理一个 head（N 个元素）
fn sum_rk_rk_spec(h: usize, n: usize, subgroup_size: u32) -> [u32; 3] {
    [h as u32, n as u32, subgroup_size]
}

/// seq_shift specialization (constant_id 0-3)
/// C: 通道数；T: 序列长度；STRIDE_X_Y / STRIDE_Y_Y: 输入/输出 token 步长
fn seq_shift_spec(c: usize, t: usize, stride_x: usize, stride_y: usize) -> [u32; 4] {
    [c as u32, t as u32, stride_x as u32, stride_y as u32]
}

/// tensor-core GEMM specialization（constant_id 0-41）。
/// 固定 tiling：TILE_M=64, TILE_N=64, TILE_K=32, MAT_M=MAT_N=MAT_K=16。
/// TILE_M/TILE_N 取 64 以最大化 workgroup 数量（M=256,N=2560 时 10→160 个），
/// 提升 GPU 并行度（此前 256 分块仅 10 个 workgroup，GPU 大量闲置）。
/// A/B 为 fp16，fp16 行按 40 个元素 Padding（TILE_K + 8），共享内存使用该 Padding 布局。
///
/// 全局内存布局（行主序）：
///   A: [M, K]，A[m,k] = m*K + k
///   B: [N, K]，B[n,k] = n*K + k
///   C: [M, N]，C[m,n] = m*N + n
///
/// 共享内存常量（SH_*）固定为 fp16 + 默认 tiling 下的 Padding 值。
/// GEMM 瓦片配置（B1 调优用，可通过环境变量覆盖）。
/// 用于 spec 构建与 dispatch 网格计算，保证两处一致。
/// 根据硬件能力自适应选择 GEMM 瓦片尺寸
/// 考虑厂商特性（NVIDIA/AMD/Intel）和共享内存限制
fn gemm_tile(app: &App) -> (u32, u32, u32) {
    let env = |name: &str, dflt: u32| -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(dflt)
    };

    // 获取硬件信息
    let vendor_id = app.properties.vendor_id;
    let max_shared_memory = app.properties.limits.max_compute_shared_memory_size;
    let subgroup_size = app.properties.subgroup_size;

    // 根据厂商选择基础瓦片尺寸
    let (base_tile_m, base_tile_n, base_tile_k) = match vendor_id {
        // NVIDIA (0x10DE): 通常有更大的共享内存和更好的张量核心支持
        0x10DE => {
            // 检查共享内存大小，高端卡支持更大瓦片
            if max_shared_memory >= 163840 {
                // 高端 NVIDIA 卡（如 RTX 3080+）
                (128, 128, 32)
            } else if max_shared_memory >= 98304 {
                // 中端 NVIDIA 卡（如 RTX 2080 Ti）
                (64, 64, 64)
            } else {
                // 低端 NVIDIA 卡
                (64, 64, 32)
            }
        }
        // AMD (0x1002): 通常有较大的共享内存但不同的缓存层次结构
        0x1002 => {
            // AMD 偏好较小的 K 维度瓦片以适应其缓存结构
            if max_shared_memory >= 65536 {
                (64, 64, 32)
            } else {
                (32, 32, 32)
            }
        }
        // Intel (0x8086): 集成显卡通常共享内存较小
        0x8086 => {
            // Intel 集成显卡使用更小的瓦片
            (32, 32, 32)
        }
        // 其他厂商使用保守的默认值
        _ => (64, 64, 32),
    };

    // 确保瓦片尺寸与 subgroup 大小兼容
    let tile_m = base_tile_m.max(subgroup_size);
    let tile_n = base_tile_n.max(subgroup_size);
    let tile_k = base_tile_k;

    // 应用环境变量覆盖
    let (tile_m, tile_n, tile_k) = (
        env("GEMM_TILE_M", tile_m),
        env("GEMM_TILE_N", tile_n),
        env("GEMM_TILE_K", tile_k),
    );

    // 共享内存上限校验：计算当前瓦片配置的共享内存使用量
    // gemm.comp 中共享内存计算：
    //   SH_A_BUF = TILE_M * (TILE_K + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4
    //   SH_B_BUF = TILE_N * (TILE_K + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4
    //   总共享内存 = (SH_A_BUF + SH_B_BUF) * 16 字节 (uvec4)
    const NUM_ELEMENT_VEC4: u32 = 4; // 每个 uvec4 包含 4 个元素
    let sh_a_buf = tile_m * (tile_k + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4;
    let sh_b_buf = tile_n * (tile_k + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4;
    let total_shared_memory = (sh_a_buf + sh_b_buf) * 16; // uvec4 = 16 字节

    // 如果超过硬件限制，逐步减小瓦片尺寸直到满足要求
    if total_shared_memory > max_shared_memory {
        log::warn!(
            "GEMM 瓦片配置 ({}, {}, {}) 需要 {} 字节共享内存，超过硬件限制 {} 字节，自动调整瓦片尺寸",
            tile_m,
            tile_n,
            tile_k,
            total_shared_memory,
            max_shared_memory
        );

        // 按优先级减小瓦片尺寸：先减小 K，再减小 M/N
        let mut adjusted_tile_k = tile_k;
        let mut adjusted_tile_m = tile_m;
        let mut adjusted_tile_n = tile_n;

        // 最多尝试 10 次调整
        for _ in 0..10 {
            // 重新计算共享内存使用量
            let sh_a_buf =
                adjusted_tile_m * (adjusted_tile_k + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4;
            let sh_b_buf =
                adjusted_tile_n * (adjusted_tile_k + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4;
            let total = (sh_a_buf + sh_b_buf) * 16;

            if total <= max_shared_memory {
                log::info!(
                    "调整后的 GEMM 瓦片配置: ({}, {}, {})，共享内存使用量: {} 字节",
                    adjusted_tile_m,
                    adjusted_tile_n,
                    adjusted_tile_k,
                    total
                );
                return (adjusted_tile_m, adjusted_tile_n, adjusted_tile_k);
            }

            // 按优先级减小瓦片尺寸
            if adjusted_tile_k > 16 {
                adjusted_tile_k /= 2;
            } else if adjusted_tile_m > subgroup_size {
                adjusted_tile_m = (adjusted_tile_m / 2).max(subgroup_size);
            } else if adjusted_tile_n > subgroup_size {
                adjusted_tile_n = (adjusted_tile_n / 2).max(subgroup_size);
            } else {
                // 无法进一步减小，使用最小配置
                log::warn!("无法进一步减小瓦片尺寸，使用最小配置");
                return (subgroup_size, subgroup_size, 16);
            }
        }

        // 如果多次调整后仍无法满足要求，使用保守的最小配置
        log::warn!("多次调整后仍无法满足共享内存限制，使用保守的最小配置");
        (subgroup_size, subgroup_size, 16)
    } else {
        (tile_m, tile_n, tile_k)
    }
}

/// 由瓦片尺寸计算 dispatch 网格（M/TILE_M × N/TILE_N）。
fn gemm_grid(app: &App, m: usize, n: usize) -> (u32, u32) {
    let (tm, tn, _) = gemm_tile(app);
    (
        m.div_ceil(tm as usize).max(1) as u32,
        n.div_ceil(tn as usize).max(1) as u32,
    )
}

#[allow(non_snake_case)]
fn gemm_spec(app: &App, m: usize, n: usize, k: usize) -> [u32; 42] {
    let (TILE_M, TILE_N, TILE_K) = gemm_tile(app);
    const MAT_M: u32 = 16;
    const MAT_N: u32 = 16;
    const MAT_K: u32 = 16;
    let ROW: u32 = TILE_K + 8; // fp16 行 Padding（40）
    let SUBTILE_M: u32 = TILE_M / 4; // 16
    let SUBTILE_N: u32 = TILE_N / 2; // 32
    let m = m as u32;
    let n = n as u32;
    let k = k as u32;
    [
        MAT_M,           // 0: MAT_M
        MAT_N,           // 1: MAT_N
        MAT_K,           // 2: MAT_K
        TILE_M,          // 3: TILE_M
        TILE_N,          // 4: TILE_N
        TILE_K,          // 5: TILE_K
        m,               // 6: M
        n,               // 7: N
        k,               // 8: K
        1,               // 9: STRIDE_A_TILE_X
        k,               // 10: STRIDE_A_TILE_Y
        TILE_K,          // 11: STRIDE_A_TILE_Z（K 块内偏移，须为 TILE_K）
        TILE_M * k,      // 12: STRIDE_A_TILE_W
        1,               // 13: STRIDE_B_TILE_X
        k,               // 14: STRIDE_B_TILE_Y
        TILE_K,          // 15: STRIDE_B_TILE_Z（K 块内偏移，须为 TILE_K）
        TILE_N * k,      // 16: STRIDE_B_TILE_W
        1,               // 17: STRIDE_SH_A_TILE_X
        ROW,             // 18: STRIDE_SH_A_TILE_Y
        1,               // 19: STRIDE_SH_B_TILE_X
        ROW,             // 20: STRIDE_SH_B_TILE_Y
        1,               // 21: STRIDE_SH_A_MAT_X
        ROW,             // 22: STRIDE_SH_A_MAT_Y
        MAT_K,           // 23: STRIDE_SH_A_SUBTILE_X
        ROW * MAT_M,     // 24: STRIDE_SH_A_SUBTILE_Y
        ROW * SUBTILE_M, // 25: STRIDE_SH_A_SUBTILE_Z
        1,               // 26: STRIDE_SH_B_MAT_X
        ROW,             // 27: STRIDE_SH_B_MAT_Y
        MAT_K,           // 28: STRIDE_SH_B_SUBTILE_X
        ROW * MAT_N,     // 29: STRIDE_SH_B_SUBTILE_Y
        ROW * SUBTILE_N, // 30: STRIDE_SH_B_SUBTILE_Z
        1,               // 31: STRIDE_C_SUBTILE_X
        n,               // 32: STRIDE_C_SUBTILE_Y
        MAT_M * n,       // 33: STRIDE_C_SUBTILE_Z
        MAT_N,           // 34: STRIDE_C_SUBTILE_W
        SUBTILE_M * n,   // 35: STRIDE_C_TILE_X
        SUBTILE_N,       // 36: STRIDE_C_TILE_Y
        TILE_M * n,      // 37: STRIDE_C_TILE_Z
        TILE_N,          // 38: STRIDE_C_TILE_W
        0,               // 39: STRIDE_A_BATCH
        0,               // 40: STRIDE_B_BATCH
        0,               // 41: STRIDE_C_BATCH
    ]
}

/// dplr_seq specialization (constant_id 0-8)
/// H: heads；N: head_size；T: 序列长度；C: 嵌入维度（=H*N）
/// STRIDE_S_Y= N, STRIDE_S_Z= N*N, STRIDE_V_X=1, STRIDE_V_Y=N, STRIDE_V_Z=C
fn dplr_seq_spec(h: usize, n: usize, t: usize, c: usize) -> [u32; 9] {
    [
        h as u32,       // 0: H
        n as u32,       // 1: N
        t as u32,       // 2: T
        1,              // 3: STRIDE_S_X (j axis)
        n as u32,       // 4: STRIDE_S_Y (i axis)
        (n * n) as u32, // 5: STRIDE_S_Z (head axis)
        1,              // 6: STRIDE_V_X (i axis within head)
        n as u32,       // 7: STRIDE_V_Y (head axis)
        c as u32,       // 8: STRIDE_V_Z (token stride)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 受控输入 GEMM：A、B 全 1，验证 C[m,n] = K（隔离核心计算）。
    #[test]
    fn gemm_ones() {
        let mut rt = Runtime::new().expect("create runtime");
        for p in &rt.app.properties.cooperative_matrix {
            eprintln!(
                "coopmat: {}x{}x{} a={} b={} c={} o={}",
                p.m, p.n, p.k, p.a, p.b, p.c, p.o
            );
        }
        // 需满足 M/N 为 256 倍数、K 为 32 倍数
        let (m, n, k) = (256usize, 256usize, 32usize);

        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();

        // 受控输入：A、B 全 1，期望 C[m,n] = K。
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];

        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();

        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();

        let got = rt.download(&c).unwrap();
        let mut max_diff = 0.0f32;
        for &g in got.iter() {
            max_diff = max_diff.max((g - k as f32).abs());
        }
        log::info!("gemm ones K=32 max_abs_diff: {max_diff:.6}");
        for r in 0..4 {
            eprintln!("row{r}: {}", got[r * n]);
        }
        assert!(max_diff < 1e-2, "ones mismatch, max_abs_diff={max_diff}");
    }

    /// 验证 tensor-core GEMM（fp32io16）正确性：C[m,n] = sum_k fp16(A[m,k])*fp16(B[n,k])。
    #[test]
    fn gemm_fp32io16_matches_reference() {
        let mut rt = Runtime::new().expect("create runtime");
        for p in &rt.app.properties.cooperative_matrix {
            eprintln!(
                "coopmat: {}x{}x{} a={} b={} c={} o={}",
                p.m, p.n, p.k, p.a, p.b, p.c, p.o
            );
        }
        // 需满足 M/N 为 256 倍数、K 为 32 倍数
        let (m, n, k) = (256usize, 256usize, 256usize);

        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();

        // 随机小值（[-0.5, 0.5]），避免 fp16 溢出
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }

        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();

        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();

        let got = rt.download(&c).unwrap();

        // CPU 参考：fp16 输入、f32 累加
        let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
        let mut exp = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for z in 0..k {
                    s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                }
                exp[i * n + j] = s;
            }
        }

        let mut max_diff = 0.0f32;
        for idx in 0..m * n {
            max_diff = max_diff.max((got[idx] - exp[idx]).abs());
        }
        log::info!("gemm max_abs_diff: {max_diff:.6}");
        assert!(max_diff < 1e-2, "gemm mismatch, max_abs_diff={max_diff}");

        // 额外用一个不同的 (M, N) 验证（N=2560 形状，模拟模型维度）
        let (m, n, k) = (256usize, 2560usize, 256usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
        let mut exp = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for z in 0..k {
                    s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                }
                exp[i * n + j] = s;
            }
        }
        let mut max_diff = 0.0f32;
        for idx in 0..m * n {
            max_diff = max_diff.max((got[idx] - exp[idx]).abs());
        }
        log::info!("gemm (256x2560) max_abs_diff: {max_diff:.6}");
        assert!(max_diff < 1e-2, "gemm mismatch, max_abs_diff={max_diff}");
    }

    /// 诊断性：同一 GEMM 跑两次，比较输出是否确定（隔离内核竞态）。
    /// 模型主投影维度 (M=256, N=2560, K=2560)，随机输入。
    #[test]
    fn gemm_determinism() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (256usize, 2560usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c1 = rt.create_tensor(m * n).unwrap();
        let mut c2 = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();

        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c1, m, n, k).unwrap();
        rt.end_batch().unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c2, m, n, k).unwrap();
        rt.end_batch().unwrap();

        let g1 = rt.download(&c1).unwrap();
        let g2 = rt.download(&c2).unwrap();
        let mut max_diff = 0.0f32;
        for idx in 0..m * n {
            max_diff = max_diff.max((g1[idx] - g2[idx]).abs());
        }
        log::info!("gemm determinism run1 vs run2 max_abs_diff: {max_diff:.6}");
        assert!(
            max_diff < 1e-4,
            "gemm non-deterministic, max_abs_diff={max_diff}"
        );

        // 覆盖模型全部 GEMM 形状/变体：rkv(M=256,N=2560,K=2560)、ffn_key(M=256,N=10240,K=2560)、
        // ffn_value(M=256,N=2560,K=10240)、output gemm_add、ffn_key gemm_relu2。
        let cases: Vec<(usize, usize, usize, &str)> = vec![
            (64, 2560, 2560, "rkv_m64"),
            (256, 10240, 2560, "ffn_key"),
            (256, 2560, 10240, "ffn_value"),
        ];
        for (m, n, k, name) in cases {
            let a16 = rt.create_tensor_f16(m * k).unwrap();
            let b16 = rt.create_tensor_f16(n * k).unwrap();
            let mut o1 = rt.create_tensor(m * n).unwrap();
            let mut o2 = rt.create_tensor(m * n).unwrap();
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; n * k];
            for v in a.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in b.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            rt.upload_f16(&a16, &a).unwrap();
            rt.upload_f16(&b16, &b).unwrap();
            rt.begin_batch().unwrap();
            rt.gemm(&a16, &b16, &mut o1, m, n, k).unwrap();
            rt.end_batch().unwrap();
            rt.begin_batch().unwrap();
            rt.gemm(&a16, &b16, &mut o2, m, n, k).unwrap();
            rt.end_batch().unwrap();
            let g1 = rt.download(&o1).unwrap();
            let g2 = rt.download(&o2).unwrap();
            let mut md = 0.0f32;
            for idx in 0..m * n {
                md = md.max((g1[idx] - g2[idx]).abs());
            }
            log::info!("gemm determinism {name} ({m}x{n}x{k}) max_abs_diff: {md:.6}");
            assert!(
                md < 1e-4,
                "gemm {name} non-deterministic, max_abs_diff={md}"
            );
        }

        // gemm_add 确定性（残差融合）
        {
            let (m, n, k) = (256usize, 2560usize, 2560usize);
            let a16 = rt.create_tensor_f16(m * k).unwrap();
            let b16 = rt.create_tensor_f16(n * k).unwrap();
            let x = rt.create_tensor(m * n).unwrap();
            let mut o1 = rt.create_tensor(m * n).unwrap();
            let mut o2 = rt.create_tensor(m * n).unwrap();
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; n * k];
            let mut xv = vec![0.0f32; m * n];
            for v in a.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in b.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in xv.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            rt.upload_f16(&a16, &a).unwrap();
            rt.upload_f16(&b16, &b).unwrap();
            rt.upload(&x, &xv).unwrap();
            rt.begin_batch().unwrap();
            rt.gemm_add(&a16, &b16, &x, &mut o1, m, n, k).unwrap();
            rt.end_batch().unwrap();
            rt.begin_batch().unwrap();
            rt.gemm_add(&a16, &b16, &x, &mut o2, m, n, k).unwrap();
            rt.end_batch().unwrap();
            let g1 = rt.download(&o1).unwrap();
            let g2 = rt.download(&o2).unwrap();
            let mut md = 0.0f32;
            for idx in 0..m * n {
                md = md.max((g1[idx] - g2[idx]).abs());
            }
            log::info!("gemm_add determinism max_abs_diff: {md:.6}");
            assert!(md < 1e-4, "gemm_add non-deterministic, max_abs_diff={md}");
        }
    }

    /// 诊断性：模型主投影维度 (M=256, N=2560, K=2560)，验证 K 大时无 DEVICE_LOST。
    #[test]
    fn gemm_model_k2560() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (256usize, 2560usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        // 用全 1 输入，避免 fp16 溢出；期望 C[m,n] = K。
        for v in a.iter_mut() {
            *v = 1.0;
        }
        for v in b.iter_mut() {
            *v = 1.0;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let mut max_diff = 0.0f32;
        for &g in got.iter() {
            max_diff = max_diff.max((g - k as f32).abs());
        }
        log::info!("gemm (256x2560x2560) ones max_abs_diff: {max_diff:.6}");
        assert!(max_diff < 1e-2, "gemm mismatch, max_abs_diff={max_diff}");
    }

    /// 诊断性：ffn_value 维度 (M=256, N=2560, K=10240)。
    #[test]
    fn gemm_ffn_k10240() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (256usize, 2560usize, 10240usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let mut max_diff = 0.0f32;
        for &g in got.iter() {
            max_diff = max_diff.max((g - k as f32).abs());
        }
        log::info!("gemm (256x2560x10240) ones max_abs_diff: {max_diff:.6}");
        assert!(max_diff < 1e-2, "gemm mismatch, max_abs_diff={max_diff}");
    }

    /// 诊断性：gemm_relu2（ffn_key 维度 N=10240）。
    #[test]
    fn gemm_relu2_ffn() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (256usize, 10240usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k];
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm_relu2(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        // relu²(K) = K²
        let expect = (k as f32) * (k as f32);
        let mut max_diff = 0.0f32;
        for &g in got.iter() {
            max_diff = max_diff.max((g - expect).abs());
        }
        log::info!("gemm_relu2 (256x10240x2560) ones max_abs_diff: {max_diff:.6}");
        assert!(
            max_diff < 1e-2,
            "gemm_relu2 mismatch, max_abs_diff={max_diff}"
        );
    }

    /// 模型规模随机值 GEMM：M=256, N=2560, K=2560，验证大 K 下 fp32io16 数值正确。
    #[test]
    fn gemm_model_scale_random() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (256usize, 2560usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
        let mut max_diff = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for z in 0..k {
                    s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                }
                max_diff = max_diff.max((got[i * n + j] - s).abs());
            }
        }
        log::info!("gemm (256x2560x2560) random max_abs_diff: {max_diff:.6}");
        assert!(
            max_diff < 1e-2,
            "gemm model-scale mismatch, max_abs_diff={max_diff}"
        );
    }

    /// gemm_bias 与 gemm_tanh 数值验证（低秩投影形状，fp32io16）。
    /// gemm_bias: C[m,n] = tanh-free sum_k fp16(A[m,k])*fp16(B[n,k]) + bias[n]
    /// gemm_tanh: C[m,n] = tanh(sum_k fp16(A[m,k])*fp16(B[n,k]))
    #[test]
    fn gemm_bias_tanh_lowrank() {
        let mut rt = Runtime::new().expect("create runtime");
        // 先验证纯 gemm（无 bias）对低秩第一级形状 M=64, N=128, K=2560 是否正确，
        // 隔离是 GEMM 本身问题还是 bias 变体问题。
        {
            let (m, n, k) = (64usize, 128usize, 2560usize);
            let a16 = rt.create_tensor_f16(m * k).unwrap();
            let b16 = rt.create_tensor_f16(n * k).unwrap();
            let mut c = rt.create_tensor(m * n).unwrap();
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; n * k];
            for v in a.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in b.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            rt.upload_f16(&a16, &a).unwrap();
            rt.upload_f16(&b16, &b).unwrap();
            rt.begin_batch().unwrap();
            rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
            rt.end_batch().unwrap();
            let got = rt.download(&c).unwrap();
            let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
            let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            let mut max_diff = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for z in 0..k {
                        s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                    }
                    max_diff = max_diff.max((got[i * n + j] - s).abs());
                }
            }
            log::info!("gemm plain (64x128x2560) max_abs_diff: {max_diff:.6}");
            assert!(
                max_diff < 1e-2,
                "gemm plain (64x128x2560) mismatch, max_abs_diff={max_diff}"
            );
        }
        // 低秩两级投影形状：第一级 M=64, N=mid_pad, K=C；第二级 M=64, N=C, K=mid_pad。
        // 用 3B 的 mid_pad=128（w/a 96→128）、K=C=2560。
        let cases: Vec<(usize, usize, usize, &str)> = vec![
            (64, 128, 2560, "w1_a1"),
            (64, 2560, 128, "w2_a2"),
            (64, 64, 2560, "v1"),
            (64, 2560, 64, "v2"),
        ];
        for (m, n, k, name) in cases {
            let a16 = rt.create_tensor_f16(m * k).unwrap();
            let b16 = rt.create_tensor_f16(n * k).unwrap();
            let bias = rt.create_tensor(n).unwrap();
            let mut c = rt.create_tensor(m * n).unwrap();
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; n * k];
            let mut bv = vec![0.0f32; n];
            for v in a.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in b.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            for v in bv.iter_mut() {
                *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
            }
            rt.upload_f16(&a16, &a).unwrap();
            rt.upload_f16(&b16, &b).unwrap();
            rt.upload(&bias, &bv).unwrap();
            rt.begin_batch().unwrap();
            rt.gemm_bias(&a16, &b16, &bias, &mut c, m, n, k).unwrap();
            rt.end_batch().unwrap();
            let got = rt.download(&c).unwrap();
            let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
            let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            let mut max_diff = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for z in 0..k {
                        s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                    }
                    s += bv[j];
                    max_diff = max_diff.max((got[i * n + j] - s).abs());
                }
            }
            log::info!("gemm_bias {name} ({m}x{n}x{k}) max_abs_diff: {max_diff:.6}");
            assert!(
                max_diff < 1e-2,
                "gemm_bias {name} ({m}x{n}x{k}) mismatch, max_abs_diff={max_diff}"
            );
        }

        // gemm_tanh：M=64, N=128, K=2560（w1 第一级）
        let (m, n, k) = (64usize, 128usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
        let mut max_diff = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for z in 0..k {
                    s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                }
                max_diff = max_diff.max((got[i * n + j] - s).abs());
            }
        }
        log::info!("gemm plain (64x128x2560) max_abs_diff: {max_diff:.6}");
        assert!(
            max_diff < 1e-2,
            "gemm plain (64x128x2560) mismatch, max_abs_diff={max_diff}"
        );

        // gemm_tanh：M=64, N=128, K=2560（w1 第一级）
        let (m, n, k) = (64usize, 128usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.begin_batch().unwrap();
        rt.gemm_tanh(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let got = rt.download(&c).unwrap();
        let a16ref: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
        let mut max_diff = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for z in 0..k {
                    s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
                }
                max_diff = max_diff.max((got[i * n + j] - s.tanh()).abs());
            }
        }
        log::info!("gemm_tanh (64x128x2560) max_abs_diff: {max_diff:.6}");
        assert!(
            max_diff < 1e-2,
            "gemm_tanh mismatch, max_abs_diff={max_diff}"
        );
    }

    /// 诊断性：GEMM 吞吐基准（IGNORE 保留，用 -- --ignored 运行）。
    /// 测量每种模型投影维度的 FP16 tensor-core 吞吐（TFLOPS）与耗时，定位 GEMM 是否瓶颈。
    #[test]
    #[ignore = "manual throughput benchmark"]
    fn gemm_throughput() {
        use std::time::Instant;
        let _ = simplelog::CombinedLogger::init(vec![simplelog::TermLogger::new(
            simplelog::LevelFilter::Info,
            Default::default(),
            Default::default(),
            simplelog::ColorChoice::Auto,
        )]);
        let mut rt = Runtime::new().expect("create runtime");
        // (名称, M, N, K, 迭代次数)
        let cases: [(String, usize, usize, usize, usize); 4] = [
            ("rkv(out) 256x2560x2560".into(), 256, 2560, 2560, 200),
            ("ffn_key  256x10240x2560".into(), 256, 10240, 2560, 200),
            ("ffn_value 256x2560x10240".into(), 256, 2560, 10240, 200),
            ("output   256x2560x2560".into(), 256, 2560, 2560, 200),
        ];
        for (name, m, n, k, iters) in cases {
            let a16 = rt.create_tensor_f16(m * k).unwrap();
            let b16 = rt.create_tensor_f16(n * k).unwrap();
            let mut c = rt.create_tensor(m * n).unwrap();
            let a = vec![1.0f32; m * k];
            let b = vec![1.0f32; n * k];
            rt.upload_f16(&a16, &a).unwrap();
            rt.upload_f16(&b16, &b).unwrap();
            // 预热
            rt.begin_batch().unwrap();
            rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
            rt.end_batch().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                rt.begin_batch().unwrap();
                rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
                rt.end_batch().unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
            let tflops = flops / dt / 1e12;
            let ms_per = dt * 1000.0 / iters as f64;
            log::info!(
                "[gemm_thr] {name}: {iters} iters in {dt:.4}s -> {tflops:.2} TFLOPS ({ms_per:.3} ms/gemm)"
            );
        }
    }

    /// 临时诊断：M=64（single-token 场景）的 gemm 与 gemm_add 正确性。
    #[test]
    fn gemm_m64_add_tmp() {
        let mut rt = Runtime::new().expect("create runtime");
        let (m, n, k) = (64usize, 2560usize, 2560usize);
        let a16 = rt.create_tensor_f16(m * k).unwrap();
        let b16 = rt.create_tensor_f16(n * k).unwrap();
        let mut c = rt.create_tensor(m * n).unwrap();
        let x = rt.create_tensor(m * n).unwrap();
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for v in a.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        for v in b.iter_mut() {
            *v = (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5;
        }
        let xv: Vec<f32> = (0..m * n)
            .map(|_| (fastrand::u32(..) as f32 / u32::MAX as f32) - 0.5)
            .collect();
        rt.upload_f16(&a16, &a).unwrap();
        rt.upload_f16(&b16, &b).unwrap();
        rt.upload(&x, &xv).unwrap();
        let a16ref: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
        let b16ref: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
        let ref_val = |i: usize, j: usize| -> f32 {
            let mut s = 0.0f32;
            for z in 0..k {
                s += a16ref[i * k + z].to_f32() * b16ref[j * k + z].to_f32();
            }
            s
        };
        // 纯 gemm M=64
        rt.begin_batch().unwrap();
        rt.gemm(&a16, &b16, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let g1 = rt.download(&c).unwrap();
        let mut md0 = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                md0 = md0.max((g1[i * n + j] - ref_val(i, j)).abs());
            }
        }
        log::info!("gemm M=64 (N2560 K2560) max_abs_diff: {md0:.6}");
        // gemm_add M=64
        rt.begin_batch().unwrap();
        rt.gemm_add(&a16, &b16, &x, &mut c, m, n, k).unwrap();
        rt.end_batch().unwrap();
        let g2 = rt.download(&c).unwrap();
        let mut md1 = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                md1 = md1.max((g2[i * n + j] - (ref_val(i, j) + xv[i * n + j])).abs());
            }
        }
        log::info!("gemm_add M=64 (N2560 K2560) max_abs_diff: {md1:.6}");
        assert!(md0 < 1e-2 && md1 < 1e-2, "M=64 mismatch g={md0} add={md1}");
    }
}
