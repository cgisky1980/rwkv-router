//! CUDA 后端（骨架）。
//!
//! 通过 `libloading` 动态加载 CUDA 驱动（Windows `nvcuda.dll` / Linux `libcuda.so`），
//! 用 CUDA Driver API 实现平台无关 `TensorId` 的设备内存管理与上传/下载。
//! 设计对齐 [Albatross（信天翁）](https://github.com/BlinkDL/Albatross) 的 CUDA 后端：
//! 设备侧缓冲以裸 device pointer 存放，融合算子（norm_lerp6 / fuse_ka_dplr_norm /
//! gemv_rkv_stage1 等）后续逐一实现为 CUDA kernel。
//!
//! 当前为**骨架**：张量管理（分配/上传/下载/批处理边界）可用并通过单测，
//! 全部算子返回“未实现”错误，待后续按 Albatross 的 kernel 逐一补齐后，
//! `detect_backend()` 才会优先选择 CUDA。

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

use half::f16;

use crate::backend::{ComputeBackend, Int8Handle, TensorDtype, TensorId};
use crate::runtime::R;

/// 检查 CUDA 调用成功，否则返回错误（`op` 为当前操作名）。
macro_rules! cu_check {
    ($e:expr, $op:literal) => {
        let _r = unsafe { $e };
        if _r != CUDA_SUCCESS {
            return Err(format!("CudaBackend: {} failed with CUresult {_r}", $op).into());
        }
    };
}

/// CUDA 驱动函数返回码 `CUresult`。
type CuResult = c_int;
const CUDA_SUCCESS: CuResult = 0;

/// NVRTC 返回码 `nvrtcResult`。
type NvrtcResult = c_int;
const NVRTC_SUCCESS: NvrtcResult = 0;

/// NVRTC 程序句柄（`nvrtcProgram`）。
type NvrtcProgram = *mut c_void;
/// CUDA 模块句柄（`CUmodule`）。
type CuModule = *mut c_void;
/// CUDA 函数句柄（`CUfunction`）。
type CuFunction = *mut c_void;

/// CUDA 驱动 API 函数指针（`cudaDriver.h` 中 `CUresult` 返回的符号）。
type FnCuInit = unsafe extern "C" fn(u32) -> CuResult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> CuResult;
type FnCuPrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult;
type FnCuPrimaryCtxRelease = unsafe extern "C" fn(c_int) -> CuResult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuDeviceComputeCapability = unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> CuResult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> CuResult;
type FnCuMemFree = unsafe extern "C" fn(u64) -> CuResult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult;
type FnCuMemcpyHtoDAsync = unsafe extern "C" fn(u64, *const c_void, usize, CuStream) -> CuResult;
type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CuResult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CuResult;
type FnCuModuleLoadDataEx = unsafe extern "C" fn(
    *mut CuModule,
    *const c_void,
    u32,
    *const c_int,
    *const *mut c_void,
) -> CuResult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    usize,
    *mut c_void,
    *const *mut c_void,
    *const *mut c_void,
) -> CuResult;
/// CUDA stream（`CUstream`）。
type CuStream = *mut c_void;
/// CUDA event（`CUevent`）。
type CuEvent = *mut c_void;
type FnCuStreamCreate = unsafe extern "C" fn(*mut CuStream, u32) -> CuResult;
type FnCuStreamDestroy = unsafe extern "C" fn(CuStream) -> CuResult;
type FnCuStreamSynchronize = unsafe extern "C" fn(CuStream) -> CuResult;
type FnCuEventCreate = unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult;
type FnCuEventRecord = unsafe extern "C" fn(CuEvent, CuStream) -> CuResult;
type FnCuEventSynchronize = unsafe extern "C" fn(CuEvent) -> CuResult;
type FnCuEventElapsedTime = unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult;
type FnCuEventDestroy = unsafe extern "C" fn(CuEvent) -> CuResult;
/// CUDA graph（`CUgraph`）与可执行 graph（`CUgraphExec`）。
type CuGraph = *mut c_void;
type CuGraphExec = *mut c_void;
type FnCuGraphBeginCapture = unsafe extern "C" fn(CuStream, u32) -> CuResult;
type FnCuGraphEndCapture = unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult;
type FnCuGraphInstantiate = unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult;
type FnCuGraphLaunch = unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult;
type FnCuGraphDestroy = unsafe extern "C" fn(CuGraph) -> CuResult;
type FnCuGraphExecDestroy = unsafe extern "C" fn(CuGraphExec) -> CuResult;
/// `CU_STREAM_CAPTURE_MODE_GLOBAL = 1`：捕获整条 stream 的图。
const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 1;

/// NVRTC API 函数指针（`nvrtc.h` 中 `nvrtcResult` 返回的符号）。
type FnNvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type FnNvrtcCompileProgram =
    unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
type FnNvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type FnNvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type FnNvrtcGetPTXSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type FnNvrtcGetPTX = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type FnNvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;
type FnNvrtcGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;
type FnCuGetErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;

/// 加载并持有 CUDA 驱动 + NVRTC 函数指针（一次加载，全进程共享）。
struct CudaDriver {
    _lib: libloading::Library,
    _nvrtc_lib: libloading::Library,
    cu_init: FnCuInit,
    cu_device_get_count: FnCuDeviceGetCount,
    cu_device_get: FnCuDeviceGet,
    cu_primary_ctx_retain: FnCuPrimaryCtxRetain,
    cu_primary_ctx_release: FnCuPrimaryCtxRelease,
    cu_ctx_set_current: FnCuCtxSetCurrent,
    cu_device_compute_capability: FnCuDeviceComputeCapability,
    cu_mem_alloc_v2: FnCuMemAlloc,
    cu_mem_free_v2: FnCuMemFree,
    cu_memcpy_dtoh_v2: FnCuMemcpyDtoH,
    cu_memcpy_htod_async: FnCuMemcpyHtoDAsync,
    cu_mem_host_alloc: FnCuMemHostAlloc,
    cu_mem_free_host: FnCuMemFreeHost,
    cu_module_load_data_ex: FnCuModuleLoadDataEx,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
    cu_stream_create: FnCuStreamCreate,
    cu_stream_destroy: FnCuStreamDestroy,
    cu_stream_synchronize: FnCuStreamSynchronize,
    cu_event_create: FnCuEventCreate,
    cu_event_record: FnCuEventRecord,
    cu_event_synchronize: FnCuEventSynchronize,
    cu_event_elapsed_time: FnCuEventElapsedTime,
    cu_event_destroy: FnCuEventDestroy,
    cu_graph_begin_capture: FnCuGraphBeginCapture,
    cu_graph_end_capture: FnCuGraphEndCapture,
    cu_graph_instantiate: FnCuGraphInstantiate,
    cu_graph_launch: FnCuGraphLaunch,
    cu_graph_destroy: FnCuGraphDestroy,
    cu_graph_exec_destroy: FnCuGraphExecDestroy,
    cu_get_error_string: FnCuGetErrorString,
    nvrtc_create_program: FnNvrtcCreateProgram,
    nvrtc_compile_program: FnNvrtcCompileProgram,
    nvrtc_get_program_log_size: FnNvrtcGetProgramLogSize,
    nvrtc_get_program_log: FnNvrtcGetProgramLog,
    nvrtc_get_ptx_size: FnNvrtcGetPTXSize,
    nvrtc_get_ptx: FnNvrtcGetPTX,
    nvrtc_destroy_program: FnNvrtcDestroyProgram,
    nvrtc_get_error_string: FnNvrtcGetErrorString,
    // per-kernel profiling（PROF_CUDA_KERNEL=1）：launch 内用 cuEvent 测每个 kernel 的
    // GPU 执行时间，按 func→name 累计。仅用于诊断，不影响正常路径。
    // 用 Mutex 包裹以保持 CudaDriver: Send+Sync（事件指针为裸指针，仅诊断、单线程访问）。
    prof: Mutex<KernelProfiler>,
}

/// per-kernel profiling 状态（仅在诊断模式下由主线程访问，故 unsafe Send+Sync）。
struct KernelProfiler {
    enabled: bool,
    evs: Option<(CuEvent, CuEvent)>,
    times: HashMap<String, (f64, usize)>,
    names: HashMap<usize, String>,
}
unsafe impl Send for KernelProfiler {}
unsafe impl Sync for KernelProfiler {}

/// 取符号；`names` 依次尝试（优先 `_v2` 版本化符号，回退到旧名）。
unsafe fn sym<F: Copy>(lib: &libloading::Library, name: &str, names: &[&[u8]]) -> R<F> {
    for n in names {
        if let Ok(s) = unsafe { lib.get::<F>(n) } {
            return Ok(*s);
        }
    }
    Err(format!("CudaDriver: symbol {name} not found").into())
}

impl CudaDriver {
    /// 打开 CUDA 驱动并加载所需函数指针。
    fn open() -> R<Self> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("nvcuda.dll")?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libcuda.so.1")
                    .or_else(|_| libloading::Library::new("libcuda.so"))?
            }
        };
        // NVRTC 库（随 CUDA toolkit 分发）：Windows `nvrtc64_120_0.dll`（12.x 固定名），
        // Linux `libnvrtc.so.12`。
        let nvrtc_lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("nvrtc64_120_0.dll")
                    .or_else(|_| libloading::Library::new("nvrtc64_121_0.dll"))?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libnvrtc.so.12")
                    .or_else(|_| libloading::Library::new("libnvrtc.so"))?
            }
        };
        let cu_init = unsafe { sym(&lib, "cuInit", &[b"cuInit\0"]) }?;
        let cu_device_get_count =
            unsafe { sym(&lib, "cuDeviceGetCount", &[b"cuDeviceGetCount\0"]) }?;
        let cu_device_get = unsafe { sym(&lib, "cuDeviceGet", &[b"cuDeviceGet\0"]) }?;
        let cu_primary_ctx_retain = unsafe {
            sym(
                &lib,
                "cuPrimaryCtxRetain",
                &[b"cuPrimaryCtxRetain\0", b"cuDevicePrimaryCtxRetain\0"],
            )?
        };
        let cu_primary_ctx_release = unsafe {
            sym(
                &lib,
                "cuPrimaryCtxRelease",
                &[
                    b"cuPrimaryCtxRelease\0",
                    b"cuDevicePrimaryCtxRelease_v2\0",
                    b"cuDevicePrimaryCtxRelease\0",
                ],
            )?
        };
        let cu_ctx_set_current = unsafe {
            sym(
                &lib,
                "cuCtxSetCurrent",
                &[b"cuCtxSetCurrent\0", b"cuCtxSetCurrent\0"],
            )?
        };
        let cu_device_compute_capability = unsafe {
            sym(
                &lib,
                "cuDeviceComputeCapability",
                &[b"cuDeviceComputeCapability\0"],
            )?
        };
        let cu_mem_alloc_v2 =
            unsafe { sym(&lib, "cuMemAlloc", &[b"cuMemAlloc_v2\0", b"cuMemAlloc\0"])? };
        let cu_mem_free_v2 =
            unsafe { sym(&lib, "cuMemFree", &[b"cuMemFree_v2\0", b"cuMemFree\0"])? };
        let cu_memcpy_dtoh_v2 = unsafe {
            sym(
                &lib,
                "cuMemcpyDtoH",
                &[b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"],
            )?
        };
        let cu_memcpy_htod_async = unsafe {
            sym(
                &lib,
                "cuMemcpyHtoDAsync",
                &[b"cuMemcpyHtoDAsync_v2\0", b"cuMemcpyHtoDAsync\0"],
            )?
        };
        let cu_mem_host_alloc = unsafe {
            sym(
                &lib,
                "cuMemHostAlloc",
                &[b"cuMemHostAlloc\0", b"cuMemHostAlloc_v2\0"],
            )?
        };
        let cu_mem_free_host = unsafe {
            sym(
                &lib,
                "cuMemFreeHost",
                &[b"cuMemFreeHost\0", b"cuMemFreeHost_v2\0"],
            )?
        };
        let cu_module_load_data_ex =
            unsafe { sym(&lib, "cuModuleLoadDataEx", &[b"cuModuleLoadDataEx\0"]) }?;
        let cu_module_get_function =
            unsafe { sym(&lib, "cuModuleGetFunction", &[b"cuModuleGetFunction\0"]) }?;
        let cu_launch_kernel = unsafe { sym(&lib, "cuLaunchKernel", &[b"cuLaunchKernel\0"]) }?;
        let cu_stream_create = unsafe { sym(&lib, "cuStreamCreate", &[b"cuStreamCreate\0"]) }?;
        let cu_stream_destroy = unsafe { sym(&lib, "cuStreamDestroy", &[b"cuStreamDestroy\0"]) }?;
        let cu_stream_synchronize =
            unsafe { sym(&lib, "cuStreamSynchronize", &[b"cuStreamSynchronize\0"]) }?;
        let cu_event_create = unsafe { sym(&lib, "cuEventCreate", &[b"cuEventCreate\0"]) }?;
        let cu_event_record = unsafe { sym(&lib, "cuEventRecord", &[b"cuEventRecord\0"]) }?;
        let cu_event_synchronize =
            unsafe { sym(&lib, "cuEventSynchronize", &[b"cuEventSynchronize\0"]) }?;
        let cu_event_elapsed_time =
            unsafe { sym(&lib, "cuEventElapsedTime", &[b"cuEventElapsedTime\0"]) }?;
        let cu_event_destroy = unsafe { sym(&lib, "cuEventDestroy", &[b"cuEventDestroy\0"]) }?;
        let cu_graph_begin_capture =
            unsafe { sym(&lib, "cuStreamBeginCapture", &[b"cuStreamBeginCapture\0"]) }?;
        let cu_graph_end_capture =
            unsafe { sym(&lib, "cuStreamEndCapture", &[b"cuStreamEndCapture\0"]) }?;
        let cu_graph_instantiate = unsafe {
            sym(
                &lib,
                "cuGraphInstantiate",
                &[b"cuGraphInstantiate\0", b"cuGraphInstantiate_v2\0"],
            )
        }?;
        let cu_graph_launch = unsafe { sym(&lib, "cuGraphLaunch", &[b"cuGraphLaunch\0"]) }?;
        let cu_graph_destroy = unsafe { sym(&lib, "cuGraphDestroy", &[b"cuGraphDestroy\0"]) }?;
        let cu_graph_exec_destroy =
            unsafe { sym(&lib, "cuGraphExecDestroy", &[b"cuGraphExecDestroy\0"]) }?;
        let cu_get_error_string =
            unsafe { sym(&lib, "cuGetErrorString", &[b"cuGetErrorString\0"]) }?;

        let nvrtc_create_program =
            unsafe { sym(&nvrtc_lib, "nvrtcCreateProgram", &[b"nvrtcCreateProgram\0"]) }?;
        let nvrtc_compile_program = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcCompileProgram",
                &[b"nvrtcCompileProgram\0"],
            )
        }?;
        let nvrtc_get_program_log_size = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcGetProgramLogSize",
                &[b"nvrtcGetProgramLogSize\0"],
            )?
        };
        let nvrtc_get_program_log =
            unsafe { sym(&nvrtc_lib, "nvrtcGetProgramLog", &[b"nvrtcGetProgramLog\0"]) }?;
        let nvrtc_get_ptx_size =
            unsafe { sym(&nvrtc_lib, "nvrtcGetPTXSize", &[b"nvrtcGetPTXSize\0"]) }?;
        let nvrtc_get_ptx = unsafe { sym(&nvrtc_lib, "nvrtcGetPTX", &[b"nvrtcGetPTX\0"]) }?;
        let nvrtc_destroy_program = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcDestroyProgram",
                &[b"nvrtcDestroyProgram\0"],
            )
        }?;
        let nvrtc_get_error_string = unsafe {
            sym(
                &nvrtc_lib,
                "nvrtcGetErrorString",
                &[b"nvrtcGetErrorString\0"],
            )
        }?;
        Ok(Self {
            _lib: lib,
            _nvrtc_lib: nvrtc_lib,
            cu_init,
            cu_device_get_count,
            cu_device_get,
            cu_primary_ctx_retain,
            cu_primary_ctx_release,
            cu_ctx_set_current,
            cu_device_compute_capability,
            cu_mem_alloc_v2,
            cu_mem_free_v2,
            cu_memcpy_dtoh_v2,
            cu_memcpy_htod_async,
            cu_mem_host_alloc,
            cu_mem_free_host,
            cu_module_load_data_ex,
            cu_module_get_function,
            cu_launch_kernel,
            cu_stream_create,
            cu_stream_destroy,
            cu_stream_synchronize,
            cu_event_create,
            cu_event_record,
            cu_event_synchronize,
            cu_event_elapsed_time,
            cu_event_destroy,
            cu_graph_begin_capture,
            cu_graph_end_capture,
            cu_graph_instantiate,
            cu_graph_launch,
            cu_graph_destroy,
            cu_graph_exec_destroy,
            cu_get_error_string,
            nvrtc_create_program,
            nvrtc_compile_program,
            nvrtc_get_program_log_size,
            nvrtc_get_program_log,
            nvrtc_get_ptx_size,
            nvrtc_get_ptx,
            nvrtc_destroy_program,
            nvrtc_get_error_string,
            prof: Mutex::new(KernelProfiler {
                enabled: false,
                evs: None,
                times: HashMap::new(),
                names: HashMap::new(),
            }),
        })
    }

    /// 启用 per-kernel profiling（惰性创建事件）。
    fn enable_kernel_profiling(&self) {
        let mut p = self.prof.lock().unwrap();
        if p.enabled {
            return;
        }
        p.times.clear();
        p.names.clear();
        p.enabled = true;
    }

    /// 注册 func→kernel 名字映射（由 CudaBackend::kernel 在首次编译后调用）。
    fn register_kernel_name(&self, func: CuFunction, name: &str) {
        if func.is_null() {
            return;
        }
        let mut p = self.prof.lock().unwrap();
        if p.enabled {
            p.names.insert(func as usize, name.to_string());
        }
    }

    /// 清空累计的 per-kernel 时间（保留 enabled/names），用于隔离单段剖析。
    fn clear_prof(&self) {
        let mut p = self.prof.lock().unwrap();
        p.times.clear();
    }

    /// 打印累计的 per-kernel 时间并清空。
    fn dump_prof(&self) {
        let p = self.prof.lock().unwrap();
        if !p.enabled {
            return;
        }
        let mut rows: Vec<_> = p.times.iter().collect();
        rows.sort_by(|a, b| {
            b.1.0
                .partial_cmp(&a.1.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let total: f64 = p.times.values().map(|(ms, _)| *ms).sum();
        for (name, (ms, cnt)) in rows {
            let avg = *ms / *cnt as f64;
            let pct = *ms / total.max(1e-9) * 100.0;
            log::info!(
                "[PROF_KERNEL] {name:>28} cnt={cnt:>3} total={ms:>9.3}ms avg={avg:>8.4}ms ({pct:>5.1}%)"
            );
        }
        log::info!("[PROF_KERNEL] SUM {total:.3}ms");
    }

    /// NVRTC 将 CUDA C 源码编译为 PTX 字节。
    fn nvrtc_to_ptx(&self, src: &str, name: &str, device: i32) -> R<Vec<u8>> {
        // 统一前置：include CUDA 官方 fp16 头，提供原生 __half/half2/__hfma2 等硬件指令。
        // （NVRTC 编译已加 --include-path 到 CUDA PATH，见 nvrtc_to_ptx。）
        let pre = r#"
#include "cuda_fp16.h"
// 8 字节对齐的 4×fp16 向量加载（对标 Albatross row1_linear 的 half4 加载），
// 提升 gemv 权重读取的内存带宽利用率。
__device__ __forceinline__ void load_half4_f4(const __half* p, float& a, float& b, float& c, float& d) {
    unsigned long long v = *reinterpret_cast<const unsigned long long*>(p);
    a = __half2float(__ushort_as_half((unsigned short)(v & 0xffffu)));
    b = __half2float(__ushort_as_half((unsigned short)((v >> 16) & 0xffffu)));
    c = __half2float(__ushort_as_half((unsigned short)((v >> 32) & 0xffffu)));
    d = __half2float(__ushort_as_half((unsigned short)((v >> 48) & 0xffffu)));
}
"#;
        let full = format!("{pre}{src}");
        let src_c =
            CString::new(full).map_err(|_| format!("CudaBackend: kernel {name} has NUL byte"))?;
        let name_c =
            CString::new(name).map_err(|_| format!("CudaBackend: kernel name {name} has NUL"))?;
        let mut prog: NvrtcProgram = std::ptr::null_mut();
        let r = unsafe {
            (self.nvrtc_create_program)(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != NVRTC_SUCCESS {
            return Err(format!(
                "CudaBackend: nvrtcCreateProgram({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        // 按设备 compute capability 选 PTX 架构（旧卡如 Turing sm_75 编译 compute_80 会报
        // NO_BINARY_FOR_GPU）。多数 kernel 用 sm_75 也可被 JIT 到更高架构，故取实际能力。
        let mut maj: c_int = 0;
        let mut min: c_int = 0;
        cu_check!(
            (self.cu_device_compute_capability)(&mut maj, &mut min, device),
            "cuDeviceComputeCapability"
        );
        let arch = if maj >= 8 {
            "compute_80"
        } else if maj == 7 {
            "compute_70"
        } else {
            return Err(
                format!("CudaBackend: unsupported GPU compute capability {maj}.{min}").into(),
            );
        };
        let opt = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        // 让 NVRTC 使用 CUDA 官方头文件（cuda_fp16.h 等），从而能生成 __hfma2 等硬件半精度指令。
        // 优先取 CUDA_PATH 环境变量，回退到常见安装路径。
        let cuda_inc = std::env::var("CUDA_PATH")
            .map(|p| format!("{p}\\include"))
            .unwrap_or_else(|_| {
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include".to_string()
            });
        let inc_opt = CString::new(format!("--include-path={cuda_inc}")).unwrap();
        let opts = [opt.as_ptr(), inc_opt.as_ptr()];
        let r = unsafe { (self.nvrtc_compile_program)(prog, 2, opts.as_ptr()) };
        if r != NVRTC_SUCCESS {
            let log = self.nvrtc_program_log(prog);
            unsafe { (self.nvrtc_destroy_program)(&mut prog) };
            return Err(format!(
                "CudaBackend: nvrtcCompileProgram({name}) failed ({}): {log}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        let mut size: usize = 0;
        let r = unsafe { (self.nvrtc_get_ptx_size)(prog, &mut size) };
        if r != NVRTC_SUCCESS {
            unsafe { (self.nvrtc_destroy_program)(&mut prog) };
            return Err(format!(
                "CudaBackend: nvrtcGetPTXSize({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        let mut ptx = vec![0u8; size];
        let r = unsafe { (self.nvrtc_get_ptx)(prog, ptx.as_mut_ptr() as *mut c_char) };
        unsafe { (self.nvrtc_destroy_program)(&mut prog) };
        if r != NVRTC_SUCCESS {
            return Err(format!(
                "CudaBackend: nvrtcGetPTX({name}) failed: {}",
                self.nvrtc_error_str(r)
            )
            .into());
        }
        Ok(ptx)
    }

    /// 读取 NVRTC 编译日志（失败时诊断用）。
    fn nvrtc_program_log(&self, prog: NvrtcProgram) -> String {
        let mut size: usize = 0;
        if unsafe { (self.nvrtc_get_program_log_size)(prog, &mut size) } != NVRTC_SUCCESS {
            return String::new();
        }
        if size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        unsafe { (self.nvrtc_get_program_log)(prog, buf.as_mut_ptr() as *mut c_char) };
        String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string()
    }

    /// 取 NVRTC 错误码对应的可读字符串。
    fn nvrtc_error_str(&self, code: NvrtcResult) -> String {
        let p = unsafe { (self.nvrtc_get_error_string)(code) };
        if p.is_null() {
            format!("nvrtc error {code}")
        } else {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// 把 PTX 字节加载为 CU 模块。
    fn load_module(&self, ptx: &[u8]) -> R<CuModule> {
        let mut module: CuModule = std::ptr::null_mut();
        let r = unsafe {
            (self.cu_module_load_data_ex)(
                &mut module,
                ptx.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != CUDA_SUCCESS {
            let msg = self.cuda_error_str(r);
            return Err(format!("CudaBackend: cuModuleLoadDataEx failed: {r} {msg}").into());
        }
        Ok(module)
    }

    /// 取 CUDA 驱动错误码对应的可读字符串。
    fn cuda_error_str(&self, code: c_int) -> String {
        let mut p: *const c_char = std::ptr::null();
        if unsafe { (self.cu_get_error_string)(code, &mut p) } == CUDA_SUCCESS && !p.is_null() {
            unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("cuda error {code}")
        }
    }

    /// 从模块取指定 kernel 函数句柄。
    fn get_function(&self, module: CuModule, func_name: &str) -> R<CuFunction> {
        let ptx_name = CString::new(func_name)
            .map_err(|_| format!("CudaBackend: func name {func_name} has NUL"))?;
        let mut func: CuFunction = std::ptr::null_mut();
        let r = unsafe { (self.cu_module_get_function)(&mut func, module, ptx_name.as_ptr()) };
        if r != CUDA_SUCCESS {
            return Err(
                format!("CudaBackend: cuModuleGetFunction({func_name}) failed: {r}").into(),
            );
        }
        Ok(func)
    }

    /// 启动 kernel 到 `stream`。`params` 为参数值指针数组（每个元素指向一个参数值）。
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: CuStream,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
    ) -> R<()> {
        self.launch_smem(stream, func, grid, block, params, 0)
    }

    /// 启动 kernel，支持动态共享内存大小 `smem_bytes`（字节）。其余同 `launch`。
    #[allow(clippy::too_many_arguments)]
    fn launch_smem(
        &self,
        stream: CuStream,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
        smem_bytes: usize,
    ) -> R<()> {
        // per-kernel profiling：launch 前 record begin，launch 后 record end，同步后累计。
        // 仅 PROF_CUDA_KERNEL=1 时启用；同步会阻塞，仅供诊断。
        {
            let prof_enabled = self.prof.lock().unwrap().enabled;
            if prof_enabled {
                let (begin, end) = {
                    let mut p = self.prof.lock().unwrap();
                    if p.evs.is_none() {
                        let mut a: CuEvent = std::ptr::null_mut();
                        let mut b: CuEvent = std::ptr::null_mut();
                        unsafe {
                            (self.cu_event_create)(&mut a, 0);
                            (self.cu_event_create)(&mut b, 0);
                        }
                        p.evs = Some((a, b));
                    }
                    p.evs.unwrap()
                };
                unsafe { (self.cu_event_record)(begin, stream) };
                let r = unsafe {
                    (self.cu_launch_kernel)(
                        func,
                        grid.0,
                        grid.1,
                        grid.2,
                        block.0,
                        block.1,
                        block.2,
                        smem_bytes,
                        stream,
                        params.as_ptr(),
                        std::ptr::null(),
                    )
                };
                if r != CUDA_SUCCESS {
                    return Err(format!("CudaBackend: cuLaunchKernel failed: {r}").into());
                }
                unsafe { (self.cu_event_record)(end, stream) };
                unsafe { (self.cu_event_synchronize)(end) };
                let mut ms: f32 = 0.0;
                unsafe { (self.cu_event_elapsed_time)(&mut ms, begin, end) };
                let name = self
                    .prof
                    .lock()
                    .unwrap()
                    .names
                    .get(&(func as usize))
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let mut t = self.prof.lock().unwrap();
                let e = t.times.entry(name).or_insert((0.0, 0));
                e.0 += ms as f64;
                e.1 += 1;
                return Ok(());
            }
        }
        let r = unsafe {
            (self.cu_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                smem_bytes,
                stream,
                params.as_ptr(),
                std::ptr::null(),
            )
        };
        if r != CUDA_SUCCESS {
            return Err(format!("CudaBackend: cuLaunchKernel failed: {r}").into());
        }
        Ok(())
    }
}

/// cuBLAS 句柄（`cublasHandle_t`）。
type CublasHandle = *mut c_void;
type CublasStatus = c_int;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
/// `cublasOperation_t`：CUBLAS_OP_N=0, CUBLAS_OP_T=1。
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
/// `cudaDataType_t`：CUDA_R_32F=0, CUDA_R_16F=2。
const CUDA_R_32F: c_int = 0;
const CUDA_R_16F: c_int = 2;
/// `cublasComputeType_t`：CUBLAS_COMPUTE_32F=68（fp16 输入晋升 fp32 累加）。
const CUBLAS_COMPUTE_32F: c_int = 68;
/// `cublasGemmAlgo_t`：CUBLAS_GEMM_DEFAULT=-1。
const CUBLAS_GEMM_DEFAULT: c_int = -1;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasDestroy = unsafe extern "C" fn(CublasHandle) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(CublasHandle, CuStream) -> CublasStatus;
type FnCublasGemmEx = unsafe extern "C" fn(
    CublasHandle,
    c_int,         // transa
    c_int,         // transb
    c_int,         // m
    c_int,         // n
    c_int,         // k
    *const c_void, // alpha
    *const c_void, // A
    c_int,         // Atype
    c_int,         // lda
    *const c_void, // B
    c_int,         // Btype
    c_int,         // ldb
    *const c_void, // beta
    *mut c_void,   // C
    c_int,         // Ctype
    c_int,         // ldc
    c_int,         // computeType
    c_int,         // algo
) -> CublasStatus;

/// 加载并持有 cuBLAS 函数指针（一次加载，全进程共享）。
struct CublasDriver {
    _lib: libloading::Library,
    cublas_create: FnCublasCreate,
    cublas_destroy_v2: FnCublasDestroy,
    cublas_set_stream: FnCublasSetStream,
    cublas_gemm_ex: FnCublasGemmEx,
}

impl CublasDriver {
    fn open() -> R<Self> {
        let lib = unsafe {
            #[cfg(target_os = "windows")]
            {
                libloading::Library::new("cublas64_12.dll")
                    .or_else(|_| libloading::Library::new("cublas64_11.dll"))?
            }
            #[cfg(not(target_os = "windows"))]
            {
                libloading::Library::new("libcublas.so.12")
                    .or_else(|_| libloading::Library::new("libcublas.so.11"))?
            }
        };
        let cublas_create = unsafe {
            sym(
                &lib,
                "cublasCreate",
                &[b"cublasCreate_v2\0", b"cublasCreate\0"],
            )
        }?;
        let cublas_destroy_v2 = unsafe {
            sym(
                &lib,
                "cublasDestroy",
                &[b"cublasDestroy_v2\0", b"cublasDestroy\0"],
            )?
        };
        let cublas_set_stream = unsafe {
            sym(
                &lib,
                "cublasSetStream",
                &[b"cublasSetStream_v2\0", b"cublasSetStream\0"],
            )?
        };
        let cublas_gemm_ex = unsafe { sym(&lib, "cublasGemmEx", &[b"cublasGemmEx\0"]) }?;
        Ok(Self {
            _lib: lib,
            cublas_create,
            cublas_destroy_v2,
            cublas_set_stream,
            cublas_gemm_ex,
        })
    }
}

/// 全局 cuBLAS 驱动（惰性加载一次；库不存在时返回 None，gemm 回退自定义 kernel）。
fn cublas_driver() -> Option<&'static CublasDriver> {
    static D: OnceLock<Option<CublasDriver>> = OnceLock::new();
    D.get_or_init(|| match CublasDriver::open() {
        Ok(d) => Some(d),
        Err(e) => {
            log::warn!("CublasDriver::open failed (fallback to custom gemm): {e}");
            None
        }
    })
    .as_ref()
}

/// 全局 CUDA 驱动（惰性加载一次）。
fn driver() -> Option<&'static CudaDriver> {
    static D: OnceLock<Option<CudaDriver>> = OnceLock::new();
    D.get_or_init(|| match CudaDriver::open() {
        Ok(d) => Some(d),
        Err(e) => {
            log::warn!("CudaDriver::open failed: {e}");
            None
        }
    })
    .as_ref()
}

/// 探测 CUDA 是否可用：驱动可加载、`cuInit` 成功且存在 ≥1 个设备。
pub fn cuda_available() -> bool {
    let Some(d) = driver() else {
        log::warn!("cuda_available: CudaDriver::open failed");
        return false;
    };
    let mut count: i32 = 0;
    if unsafe { (d.cu_init)(0) } != CUDA_SUCCESS {
        log::warn!("cuda_available: cuInit failed");
        return false;
    }
    if unsafe { (d.cu_device_get_count)(&mut count) } != CUDA_SUCCESS {
        log::warn!("cuda_available: cuDeviceGetCount failed");
        return false;
    }
    count > 0
}

/// 设备侧张量（骨架阶段仅需 device pointer + len）。
#[derive(Debug, Clone)]
enum CudaTensor {
    F32 { dptr: u64, len: usize },
    F16 { dptr: u64, len: usize },
    U32 { dptr: u64, len: usize },
}

/// CUDA 后端骨架：持有驱动 + 主上下文 + 张量映射 + kernel 缓存。
pub struct CudaBackend {
    drv: &'static CudaDriver,
    /// 主上下文句柄：`cuPrimaryCtxRetain` 保留其存活，进程退出前由 `Drop` 释放。
    #[allow(dead_code)]
    ctx: *mut c_void,
    device: i32,
    tensors: HashMap<TensorId, CudaTensor>,
    lens: HashMap<TensorId, usize>,
    next_id: u32,
    /// kernel 缓存：kernel 名 → (已加载模块, 函数句柄)。模块需保持存活，故随函数一起存。
    kernels: HashMap<String, (CuModule, CuFunction)>,
    /// 计算 stream：所有 kernel 与异步拷贝都在其上排队，`download` 前 `cuStreamSynchronize`。
    stream: CuStream,
    /// GPU 批剖析（PROF_CUDA_GPU=1）：begin_batch 记 start，end_batch 记 end 并同步打印耗时。
    prof_ev_start: CuEvent,
    prof_ev_end: CuEvent,
    prof_gpu: bool,
    /// per-kernel 剖析（PROF_CUDA_KERNEL=1）：launch 内逐 kernel 计时，end_batch 打印。
    prof_kernel: bool,
    /// 是否正在 CUDA stream 捕获（begin_graph_capture..end_graph_capture 之间）。
    graph_capturing: bool,
    /// 已实例化的可执行 graph（decode self-loop 每 token 重放用）。
    exec_graph: Option<CuGraphExec>,
    /// prefill graph：按 token 数 T 缓存，整段 prefill 一次捕获、同 T 重放。
    prefill_graphs: HashMap<usize, CuGraphExec>,
    /// 正在捕获的 prefill 对应 T（end_prefill_capture 用）。
    prefill_t: usize,
    /// cuBLAS 句柄（prefill GEMM 用；cuBLAS 不可用时为 None → 回退自定义 kernel）。
    cublas: Option<CublasHandle>,
    /// cuBLAS GEMM 剖析（PROF_GEMM=1）：逐 (m,n,k,op) 累计次数与耗时，end_batch 打印。
    gemm_prof_ev_start: CuEvent,
    gemm_prof_ev_end: CuEvent,
    gemm_prof: bool,
    gemm_times: HashMap<(usize, usize, usize, i32), (u64, f64)>,
    /// pinned host 暂存区（sampler 异步行 + 小上传 scratch）。
    pinned: *mut c_void,
    /// pinned 暂存区总行数（固定 PINNED_ROWS；batch 宽行时按 batch 分组）。
    pinned_rows: usize,
    /// 从其它后端导入的张量（`import_tensors_from` 权重共享）：Drop 时不释放。
    foreign: std::collections::HashSet<TensorId>,
}

/// 全局 CUDA 上下文锁：串行化 `CudaBackend::new()` 的创建期（cuPrimaryCtxRetain/
/// cuCtxSetCurrent 并发竞争防护）。仅覆盖初始化阶段——创建完成后即释放，
/// 同进程多个后端实例可以共存（同线程顺序使用；见客户端常驻路由小模型场景）。
static CUDA_CTX_LOCK: Mutex<()> = Mutex::new(());

/// pinned 暂存区单行字节数（sampler 参数行 = 8 个 f32）。
const PINNED_ROW_BYTES: usize = 32;
/// pinned 暂存区行数（async sampler 路径的上限：selfloop n ≤ 行数）。
const PINNED_ROWS: usize = 8192;
/// pinned 上传 scratch 大小（≤ 此大小的同步上传走常驻 pinned scratch，避免
/// pageable 源在多线程并发 `cuMemcpyHtoDAsync` 时踩踏驱动内部共享 staging）。
const PINNED_UPLOAD_SCRATCH: usize = 64 * 1024 * 1024;

impl CudaBackend {
    /// 创建 CUDA 后端：初始化驱动、取首个设备、保留主上下文。
    pub fn new() -> R<Self> {
        // 加锁：串行化创建期（多线程并发 new 的 cuPrimaryCtxRetain/cuCtxSetCurrent 竞争防护）。
        // 守卫在函数返回时释放——后端存活期间不持有锁，多实例可共存。
        let _ctx_lock = CUDA_CTX_LOCK
            .lock()
            .map_err(|_| "CUDA context lock poisoned")?;
        let drv = driver().ok_or("CudaBackend: CUDA driver unavailable")?;
        let mut count: i32 = 0;
        cu_check!((drv.cu_init)(0), "cuInit");
        cu_check!((drv.cu_device_get_count)(&mut count), "cuDeviceGetCount");
        if count <= 0 {
            return Err("CudaBackend: no CUDA device".into());
        }
        let mut device: i32 = 0;
        cu_check!((drv.cu_device_get)(&mut device, 0), "cuDeviceGet");
        let mut ctx: *mut c_void = std::ptr::null_mut();
        cu_check!(
            (drv.cu_primary_ctx_retain)(&mut ctx, device),
            "cuPrimaryCtxRetain"
        );
        // 主上下文 retain 后须绑定到当前线程，否则 cuMemAlloc/cuLaunchKernel 报 INVALID_CONTEXT(201)。
        cu_check!((drv.cu_ctx_set_current)(ctx), "cuCtxSetCurrent");
        // 创建计算 stream（默认标志 0）。
        let mut stream: CuStream = std::ptr::null_mut();
        cu_check!((drv.cu_stream_create)(&mut stream, 0), "cuStreamCreate");
        // 创建剖析事件（PROF_CUDA_GPU=1 时用）。
        let mut ev_start: CuEvent = std::ptr::null_mut();
        let mut ev_end: CuEvent = std::ptr::null_mut();
        cu_check!((drv.cu_event_create)(&mut ev_start, 0), "cuEventCreate");
        cu_check!((drv.cu_event_create)(&mut ev_end, 0), "cuEventCreate");
        let mut gemm_ev_start: CuEvent = std::ptr::null_mut();
        let mut gemm_ev_end: CuEvent = std::ptr::null_mut();
        cu_check!(
            (drv.cu_event_create)(&mut gemm_ev_start, 0),
            "cuEventCreate"
        );
        cu_check!((drv.cu_event_create)(&mut gemm_ev_end, 0), "cuEventCreate");
        // pinned host 暂存区：前 PINNED_ROWS 行作 sampler 异步参数行，尾部作上传 scratch。
        let pinned_bytes = PINNED_ROWS * PINNED_ROW_BYTES + PINNED_UPLOAD_SCRATCH;
        let mut pinned: *mut c_void = std::ptr::null_mut();
        cu_check!(
            (drv.cu_mem_host_alloc)(&mut pinned, pinned_bytes, 0),
            "cuMemHostAlloc(pinned)"
        );
        let prof_gpu = std::env::var("PROF_CUDA_GPU").is_ok();
        let gemm_prof = std::env::var("PROF_GEMM").is_ok();
        // per-kernel profiling（诊断用）：需要可变的 drv 引用来启用。
        let prof_kernel = std::env::var("PROF_CUDA_KERNEL").is_ok();
        if prof_kernel {
            drv.enable_kernel_profiling();
        }
        // 创建 cuBLAS 句柄（当前上下文已绑定）。不可用时回退自定义 gemm kernel。
        // CUBLAS=0 可临时禁用 cuBLAS（诊断/对拍用）。
        let cublas = if std::env::var("CUBLAS").as_deref() == Ok("0") {
            None
        } else {
            cublas_driver().and_then(|cd| {
                let mut h: CublasHandle = std::ptr::null_mut();
                if unsafe { (cd.cublas_create)(&mut h) } != CUBLAS_STATUS_SUCCESS || h.is_null() {
                    log::warn!("cublasCreate failed; falling back to custom gemm");
                    return None;
                }
                // 把 cuBLAS 绑定到计算 stream，保证与其它 kernel 的序关系。
                if unsafe { (cd.cublas_set_stream)(h, stream) } != CUBLAS_STATUS_SUCCESS {
                    log::warn!("cublasSetStream failed; falling back to custom gemm");
                    unsafe { (cd.cublas_destroy_v2)(h) };
                    return None;
                }
                Some(h)
            })
        };
        Ok(Self {
            drv,
            ctx,
            device,
            tensors: HashMap::new(),
            lens: HashMap::new(),
            next_id: 0,
            kernels: HashMap::new(),
            stream,
            prof_ev_start: ev_start,
            prof_ev_end: ev_end,
            prof_gpu,
            prof_kernel,
            gemm_prof_ev_start: gemm_ev_start,
            gemm_prof_ev_end: gemm_ev_end,
            gemm_prof,
            gemm_times: HashMap::new(),
            graph_capturing: false,
            exec_graph: None,
            prefill_graphs: HashMap::new(),
            prefill_t: 0,
            cublas,
            pinned,
            pinned_rows: PINNED_ROWS,
            foreign: std::collections::HashSet::new(),
        })
    }

    /// 编译并缓存 kernel（同名复用已编译模块）。`src` 为 CUDA C 源码，`entry` 为 __global__ 函数名。
    fn kernel(&mut self, key: &str, src: &str, entry: &str) -> R<CuFunction> {
        if let Some((_, f)) = self.kernels.get(key) {
            return Ok(*f);
        }
        log::info!("compiling kernel {key} ({entry})");
        let ptx = self.drv.nvrtc_to_ptx(src, key, self.device)?;
        let module = self.drv.load_module(&ptx)?;
        let func = self.drv.get_function(module, entry)?;
        self.kernels.insert(key.to_string(), (module, func));
        self.drv.register_kernel_name(func, key);
        Ok(func)
    }

    fn alloc(&self, bytes: usize) -> R<u64> {
        let mut dptr: u64 = 0;
        cu_check!((self.drv.cu_mem_alloc_v2)(&mut dptr, bytes), "cuMemAlloc");
        Ok(dptr)
    }

    /// 部分上传：只拷贝前置 `n` 个元素（每元素 4 字节），device 其余部分不动。
    /// 对齐 Vulkan 的 `host.copy_from(data, 0)` 语义（允许 data.len() <= 张量 len）。
    ///
    /// 上传源必须为 pinned：pageable 源的 cuMemcpyHtoDAsync 走驱动内部**共享
    /// staging 缓冲**，多线程并发上传互相践踏；同步 cuMemcpyHtoD 则与非阻塞
    /// stream 无顺序关系（kernel 可能先于 DMA 启动）。≤ scratch 的上传用常驻
    /// pinned scratch；更大的上传临时分配 pinned（一次性权重加载）。拷贝以流序
    /// 异步挂到本 stream，前后同步保证：先序于后续 kernel、完成后才返回
    ///（pageable 源可释放、scratch 可复用）。
    fn memcpy_htod_n(&self, dptr: u64, data: &[u8], n: usize) -> R<()> {
        let bytes = n * 4;
        assert!(bytes <= data.len(), "memcpy_htod_n: n 超出 data");
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod)"
        );
        self.htod_pinned(dptr, data, bytes)
    }

    /// 部分上传（fp16）：只拷贝前置 `n` 个元素（每元素 2 字节），device 其余部分不动。
    /// 语义同 memcpy_htod_n（pinned 源 + 流序异步 + 前后同步）。
    fn memcpy_htod_n2(&self, dptr: u64, data: &[u8], n: usize) -> R<()> {
        let bytes = n * 2;
        assert!(bytes <= data.len(), "memcpy_htod_n2: n 超出 data");
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod2)"
        );
        self.htod_pinned(dptr, data, bytes)
    }

    /// pinned 源流序上传 + 完成同步（调用前须已排空本 stream）。
    fn htod_pinned(&self, dptr: u64, data: &[u8], bytes: usize) -> R<()> {
        let scratch_off = PINNED_ROWS * PINNED_ROW_BYTES;
        if bytes <= PINNED_UPLOAD_SCRATCH {
            // 常驻 scratch：host 拷入 → 流序异步 DMA → 同步完成（scratch 可复用）。
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (self.pinned as *mut u8).add(scratch_off),
                    bytes,
                );
            }
            let src = unsafe { (self.pinned as *const u8).add(scratch_off) };
            cu_check!(
                (self.drv.cu_memcpy_htod_async)(dptr, src as *const c_void, bytes, self.stream),
                "cuMemcpyHtoDAsync(scratch)"
            );
        } else {
            // 大上传（权重加载）：临时 pinned 分配，完成后释放。
            let mut tmp: *mut c_void = std::ptr::null_mut();
            cu_check!(
                (self.drv.cu_mem_host_alloc)(&mut tmp, bytes, 0),
                "cuMemHostAlloc(htod tmp)"
            );
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), tmp as *mut u8, bytes);
            }
            cu_check!(
                (self.drv.cu_memcpy_htod_async)(dptr, tmp as *const c_void, bytes, self.stream),
                "cuMemcpyHtoDAsync(tmp pinned)"
            );
            cu_check!(
                (self.drv.cu_stream_synchronize)(self.stream),
                "cuStreamSynchronize(htod tmp-wait)"
            );
            unsafe {
                (self.drv.cu_mem_free_host)(tmp);
            }
            return Ok(());
        }
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(htod-wait)"
        );
        Ok(())
    }

    fn memcpy_dtoh(&self, dptr: u64, out: &mut [u8]) -> R<()> {
        // 同步 stream：确保 kernel 已把结果写入 device 内存，再同步拷回 host。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(dtoh)"
        );
        cu_check!(
            (self.drv.cu_memcpy_dtoh_v2)(out.as_mut_ptr() as *mut c_void, dptr, out.len()),
            "cuMemcpyDtoH"
        );
        Ok(())
    }

    fn get(&self, t: TensorId, op: &str) -> R<CudaTensor> {
        self.tensors
            .get(&t)
            .cloned()
            .ok_or(format!("{op}: unknown tensor {t:?}").into())
    }

    /// 取 f32 张量设备指针。
    fn f32_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::F32 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
        }
    }

    /// 取 f16 张量设备指针。
    fn f16_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::F16 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
        }
    }

    /// 取 u32 张量设备指针。
    fn u32_ptr(&self, t: TensorId, op: &str) -> R<u64> {
        match self.get(t, op)? {
            CudaTensor::U32 { dptr, .. } => Ok(dptr),
            _ => Err(format!("{op}: tensor {t:?} must be u32").into()),
        }
    }

    /// 统一调度 gemv_variant kernel（wtype/op 见 GEMV_VARIANT_SRC）。
    /// 传入已解析的设备指针；未用指针传 0。
    /// batch>1 时走 `gemv_variant_mb`（权重复用版：每 block 读一次权重算
    /// BGRP 个 slot，带宽 ≈ 1/ceil(B/BGRP)——信天翁 rows 模型）；batch==1 走原版。
    #[allow(clippy::too_many_arguments)]
    fn gemv_variant_dispatch(
        &mut self,
        af16: u64,
        aidx: u64,
        alut: u64,
        asz: u64,
        xd: u64,
        gd: u64,
        yd: u64,
        m: usize,
        k: usize,
        batch: usize,
        wtype: i32,
        op: i32,
    ) -> R<()> {
        let (func, grid) = if batch > 1 {
            const GEMV_MB_BGRP: usize = 4;
            let func = self.kernel("gemv_variant_mb", GEMV_VARIANT_MB_SRC, "gemv_variant_mb")?;
            (
                func,
                ((m / 4) as u32, batch.div_ceil(GEMV_MB_BGRP) as u32, 1u32),
            )
        } else {
            let func = self.kernel("gemv_variant", GEMV_VARIANT_SRC, "gemv_variant")?;
            (func, ((m / 4) as u32, 1u32, 1u32))
        };
        let block = (128u32, 1u32, 1u32);
        let m_i = m as i32;
        let k_i = k as i32;
        let b_i = batch as i32;
        let params = [
            &af16 as *const u64 as *mut c_void,
            &aidx as *const u64 as *mut c_void,
            &alut as *const u64 as *mut c_void,
            &asz as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
            &wtype as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
        ];
        // 当前各分支均不使用动态共享内存（int8 预解量化曾尝试但同步开销反超，已回退）。
        let smem = 0usize;
        self.drv
            .launch_smem(self.stream, func, grid, block, &params, smem)
    }

    #[allow(dead_code)] // 骨架阶段算子未实现，待补齐后由算子消费
    fn take(&mut self, t: TensorId, op: &str) -> R<CudaTensor> {
        self.tensors
            .remove(&t)
            .ok_or(format!("{op}: unknown tensor {t:?}").into())
    }

    #[allow(dead_code)] // 骨架阶段算子未实现，待补齐后由算子消费
    fn put(&mut self, t: TensorId, v: CudaTensor) {
        self.tensors.insert(t, v);
    }

    /// 统一调度 gemm kernel（op 见 GEMM_SRC：0=plain,1=bias,2=add,3=relu2,4=tanh）。
    /// `bias`/`x` 为可选残差指针；不需要时传空 map 的 None，kernel 内部以 0 表示 null。
    #[allow(clippy::too_many_arguments)]
    fn gemm_dispatch(
        &mut self,
        ad: u64,
        bd: u64,
        bias: Option<u64>,
        xd: Option<u64>,
        cd: u64,
        m: usize,
        n: usize,
        k: usize,
        op: i32,
    ) -> R<()> {
        // cuBLAS 路径：C = A @ B^T（A:[m,k] f16, B:[n,k] f16, C:[m,n] f32）。
        // op 语义：0=纯 GEMM, 1=+bias, 2=+x, 3=relu², 4=tanh。
        // CUBLAS_GEMM=0：保留 cuBLAS 句柄但 gemm 走自定义 kernel（隔离 cuBLAS gemm 与句柄污染）。
        let use_cublas_gemm =
            self.cublas.is_some() && std::env::var("CUBLAS_GEMM").as_deref() != Ok("0");
        if let Some(h) = self.cublas.filter(|_| use_cublas_gemm) {
            if std::env::var("CUBLAS_DIAG").is_ok() {
                log::info!("[CUBLAS_DIAG] gemm m={m} n={n} k={k} op={op}");
            }
            if self.gemm_prof {
                unsafe {
                    (self.drv.cu_event_record)(self.gemm_prof_ev_start, self.stream);
                }
            }
            // op==2 的残差 x 是独立缓冲，不能用 beta=1 累加到 C（C 初始非 x），故 beta=0 后 epilogue 加 x。
            let beta: f32 = 0.0;
            let driver = cublas_driver().ok_or("cublas driver gone")?;
            let alpha: f32 = 1.0;
            // 自定义 kernel 语义 C[m,n] = A[m,k] @ B^T[n,k]（A/B 均行主序 f16，C 行主序 f32）。
            // cuBLAS 输出列主序 C_cm = B @ A^T（形 [n,m]），ldc=n 即得到行主序 C[m,n]；
            // 故 transa=OP_T(权重 B)、transb=OP_N(输入 A)、m=n输出、n=m token、lda=ldb=k。
            let (m_i, n_i, k_i) = (n as c_int, m as c_int, k as c_int);
            let (lda, ldb, ldc) = (k as c_int, k as c_int, n as c_int);
            let r = unsafe {
                (driver.cublas_gemm_ex)(
                    h,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m_i,
                    n_i,
                    k_i,
                    &alpha as *const f32 as *const c_void,
                    bd as *const c_void,
                    CUDA_R_16F,
                    lda,
                    ad as *const c_void,
                    CUDA_R_16F,
                    ldb,
                    &beta as *const f32 as *const c_void,
                    cd as *mut c_void,
                    CUDA_R_32F,
                    ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                )
            };
            if r != CUBLAS_STATUS_SUCCESS {
                return Err(format!("cublasGemmEx failed: {r}").into());
            }
            if self.gemm_prof {
                let mut ms: f32 = 0.0;
                unsafe {
                    (self.drv.cu_event_record)(self.gemm_prof_ev_end, self.stream);
                    (self.drv.cu_event_synchronize)(self.gemm_prof_ev_end);
                    (self.drv.cu_event_elapsed_time)(
                        &mut ms,
                        self.gemm_prof_ev_start,
                        self.gemm_prof_ev_end,
                    );
                }
                let e = self.gemm_times.entry((m, n, k, op)).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += (ms as f64).max(0.0);
            }
            // epilogue：op1 bias / op2 add x / op3 relu² / op4 tanh（op0 无需）。
            if op != 0 {
                let func = self.kernel("gemm_epilogue", GEMM_EPILOGUE_SRC, "rwkv_gemm_epilogue")?;
                let total = (m * n) as u32;
                let grid = (total.div_ceil(256), 1u32, 1u32);
                let block = (256u32, 1u32, 1u32);
                let (m_i, n_i, op_i) = (m as i32, n as i32, op);
                let (bias_v, x_v) = (bias.unwrap_or(0), xd.unwrap_or(0));
                let params = [
                    &cd as *const u64 as *mut c_void,
                    &bias_v as *const u64 as *mut c_void,
                    &x_v as *const u64 as *mut c_void,
                    &m_i as *const i32 as *mut c_void,
                    &n_i as *const i32 as *mut c_void,
                    &op_i as *const i32 as *mut c_void,
                ];
                self.drv.launch(self.stream, func, grid, block, &params)?;
            }
            return Ok(());
        }
        // 回退：自定义 kernel（cuBLAS 不可用时）。
        let func = self.kernel("gemm", GEMM_SRC, "rwkv_gemm")?;
        let grid = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1u32);
        let block = (16u32, 16u32, 1u32);
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let (bias_v, x_v) = (bias.unwrap_or(0), xd.unwrap_or(0));
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &bias_v as *const u64 as *mut c_void,
            &x_v as *const u64 as *mut c_void,
            &cd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &op as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        for (id, v) in self.tensors.iter() {
            // 导入的共享权重张量（build_shared 权重共享）不释放：归源实例所有。
            if self.foreign.contains(id) {
                continue;
            }
            let dptr = match v {
                CudaTensor::F32 { dptr, .. }
                | CudaTensor::F16 { dptr, .. }
                | CudaTensor::U32 { dptr, .. } => *dptr,
            };
            unsafe {
                (self.drv.cu_mem_free_v2)(dptr);
            }
        }
        if let Some(exec) = self.exec_graph {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
        for (_, exec) in self.prefill_graphs.drain() {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
        if let (Some(h), Some(cd)) = (self.cublas, cublas_driver()) {
            unsafe {
                (cd.cublas_destroy_v2)(h);
            }
        }
        unsafe {
            (self.drv.cu_event_destroy)(self.prof_ev_start);
            (self.drv.cu_event_destroy)(self.prof_ev_end);
            (self.drv.cu_event_destroy)(self.gemm_prof_ev_start);
            (self.drv.cu_event_destroy)(self.gemm_prof_ev_end);
            (self.drv.cu_stream_destroy)(self.stream);
            if !self.pinned.is_null() {
                (self.drv.cu_mem_free_host)(self.pinned);
            }
        }
        unsafe {
            (self.drv.cu_primary_ctx_release)(self.device);
        }
    }
}

/// gemv_f16 CUDA kernel：y[m] = Σ_k x[k]·A[m·K + k]。
/// A 为 fp16 行主序 (M,K)，x 为 f32，y 为 f32；batch 走 gridDim.y。
/// 每个 block 处理 4 行输出（对齐 Vulkan `GEMV_ROWS`），block 内 128 线程跨 K 归约。
const GEMV_F16_SRC: &str = r#"
extern "C" __global__ void gemv_f16(
    const __half* __restrict__ A,   // (M, K) row-major fp16
    const float*  __restrict__ x,   // (K * batch)
    float* __restrict__ y,          // (M * batch)
    const int m,
    const int k,
    const int batch)
{
    const int tid   = threadIdx.x;
    const int b     = blockIdx.y;
    const int row0  = blockIdx.x * 4;
    const int k0    = b * k;
    const int m0    = b * m;
    // 半精度向量化累积（对标 Albatross row1_linear_exact4）：x 转 half2、权重按 half2 读，
    // __hfma2 每次迭代做 2×FP16 FMA（吞吐为 FP32 的 2 倍）。4 行分别持有 2 个 half2 累加器。
    half2 hacc[4][2];
    #pragma unroll
    for (int r = 0; r < 4; r++) { hacc[r][0] = __half2half2(0.f); hacc[r][1] = __half2half2(0.f); }

    // 向量化主循环：每线程每次迭代处理 4 个 k（x 按 float4 读、权重按 8B half4 读）。
    const int k4 = k & ~3;
    for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
        const float4 xv = *reinterpret_cast<const float4*>(x + k0 + kq);
        const half2 hx01 = __floats2half2_rn(xv.x, xv.y);
        const half2 hx23 = __floats2half2_rn(xv.z, xv.w);
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const __half* wj = A + (row0 + r) * k + kq;
            const half2 w01 = *reinterpret_cast<const half2*>(wj);
            const half2 w23 = *reinterpret_cast<const half2*>(wj + 2);
            hacc[r][0] = __hfma2(hx01, w01, hacc[r][0]);
            hacc[r][1] = __hfma2(hx23, w23, hacc[r][1]);
        }
    }
    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const float2 f0 = __half22float2(hacc[r][0]);
        const float2 f1 = __half22float2(hacc[r][1]);
        acc[r] = f0.x + f0.y + f1.x + f1.y;
    }
    // 尾部标量兜底（k 非 4 倍数时）。
    for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
        const float xv = x[k0 + kk];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            acc[r] += __half2float(A[(row0 + r) * k + kk]) * xv;
        }
    }

    // warp shuffle 归约（对齐 Albatross row1_linear_exact4_kernel<128,2>）：
    // 每个 warp（32 线程）先 shfl 归约本 warp 的 4 行，再 3 个 warp 结果由 tid0 汇总。
    // 相比共享内存全归约（7 步 __syncthreads），只同步 1 次，减少 block 内同步开销。
    __shared__ float partial[4 /*warp*/][4 /*row*/];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[warp][r] = v;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; w++) sum += partial[w][r];
            if (row0 + r < m) y[m0 + row0 + r] = sum;
        }
    }
}
"#;

// ==== batch 并发 kernel（单实例多序列：B slot 共享权重，一次读权重算 B 份）====

/// norm_lerp6 batch CUDA kernel：x/state/or_..og 为 [batch, C]（slot 主序）；
/// gamma/beta 与 **xr..xg（lerp 系数，共享权重）** 跨 slot 共享 [C]——无 slot 偏移。
/// dispatch (ceil(c/BLOCK), batch, 1)：grid.y = slot id。
const NORM_LERP6_BATCH_SRC: &str = r#"
extern "C" __global__ void norm_lerp6_batch(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ xr,
    const float* __restrict__ xw,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    const float* __restrict__ xa,
    const float* __restrict__ xg,
    float* __restrict__ or_,
    float* __restrict__ ow,
    float* __restrict__ ok,
    float* __restrict__ ov,
    float* __restrict__ oa,
    float* __restrict__ og,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;
    const int off  = blockIdx.y * c;   // slot 基址

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[off + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off_ = 16; off_ > 0; off_ >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off_);
        sq  += __shfl_down_sync(0xffffffffu, sq, off_);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[off + i];
        const float ln1  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[off + i];
        or_[off + i] = ln1 + xr[i] * (prev - ln1);
        ow[off + i]  = ln1 + xw[i] * (prev - ln1);
        ok[off + i]  = ln1 + xk[i] * (prev - ln1);
        ov[off + i]  = ln1 + xv[i] * (prev - ln1);
        oa[off + i]  = ln1 + xa[i] * (prev - ln1);
        og[off + i]  = ln1 + xg[i] * (prev - ln1);
        state[off + i] = ln1;
    }
}
"#;

/// cmix_norm_lerp batch CUDA kernel：x/state/out_xb 为 [batch, C]，gamma/beta/coeff 共享。
/// dispatch (ceil(c/BLOCK), batch, 1)。
const CMIX_NORM_LERP_BATCH_SRC: &str = r#"
extern "C" __global__ void cmix_norm_lerp_batch(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ coeff,
    float* __restrict__ out_xb,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;
    const int off  = blockIdx.y * c;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[off + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off_ = 16; off_ > 0; off_ >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off_);
        sq  += __shfl_down_sync(0xffffffffu, sq, off_);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[off + i];
        const float ln2  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[off + i];
        out_xb[off + i] = ln2 + coeff[i] * (prev - ln2);
        state[off + i] = ln2;
    }
}
"#;

/// gather_rows_f16 batch CUDA kernel：按 tok[b] 各取一行 → dst[b*C + i]（f32）。
/// dispatch (ceil(C/256), batch, 1)。
const GATHER_ROWS_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_gather_rows_f16(
    const unsigned int* __restrict__ in_tok,  // [batch] token 索引（f32 位模式）
    const __half*  __restrict__ in_src,       // [VOCAB, C] fp16
    float* __restrict__ out_dst,              // [batch, C] fp32
    const int c)
{
    const int b = blockIdx.y;
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idx = in_tok[b];
    if (index < c) {
        out_dst[b * c + index] = __half2float(in_src[(size_t)idx * (size_t)c + (size_t)index]);
    }
}
"#;

/// gemv_int8_rkv_stage1 batch CUDA kernel（权重复用版）：每 block 读一次 int8 权重
/// 与 scale/zero，在寄存器累加器中复用给 BGRP 个 slot——带宽 ≈ 1/ceil(B/BGRP)
///（信天翁 rows 模型；旧 grid.y=slot 版每 slot 各读全量权重，带宽 ×B 零增益）。
/// x 输入（xr..xg）与输出为 [batch, ...]（slot 主序）。
/// dispatch (C/ROWS + VM + WM + AM + GM, ceil(batch/BGRP), 1)。
const GEMV_INT8_RKV_STAGE1_BATCH_SRC: &str = r#"
__device__ __forceinline__ void unpack_int8_sz_batch(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}

extern "C" __global__ void __launch_bounds__(128, 4) gemv_int8_rkv_stage1_batch(
    const unsigned int* __restrict__ R_idx,
    const unsigned int* __restrict__ R_sz,
    const unsigned int* __restrict__ K_idx,
    const unsigned int* __restrict__ K_sz,
    const unsigned int* __restrict__ V_idx,
    const unsigned int* __restrict__ V_sz,
    const float*  __restrict__ V1,
    const float*  __restrict__ W1,
    const float*  __restrict__ A1,
    const float*  __restrict__ G1,
    const float*  __restrict__ xr,            // [batch, C]
    const float*  __restrict__ xk,
    const float*  __restrict__ xv,
    const float*  __restrict__ xw,
    const float*  __restrict__ xa,
    const float*  __restrict__ xg,
    float* __restrict__ out_r,                // [batch, C]
    float* __restrict__ out_k,
    __half* __restrict__ out_v,               // [batch, C] fp16
    float* __restrict__ out_vm,               // [batch, VM]
    float* __restrict__ out_wm,               // [batch, WM]
    float* __restrict__ out_am,               // [batch, AM]
    float* __restrict__ out_gm,               // [batch, GM]
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm,
    const int batch)
{
    constexpr int ROWS = 4;
    constexpr int KG_MAX = 32;
    // BGRP=2：累加器 3 矩阵×4 行×2 slot = 24 half2（48 寄存器）。BGRP=4 时 96 寄存器
    // 溢出到 local memory，实测反比单序列慢（profiling 22.8% 热点根因）。
    constexpr int BGRP = 2;
    const int tid  = threadIdx.x;
    const int flat = blockIdx.x;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);

    if (flat < c / ROWS) {
        const int row_base = flat * ROWS;
        const int KV = c / 4;
        const int KG = c / 128;

        __shared__ float s_scale[3][ROWS][KG_MAX];
        __shared__ float s_zero[3][ROWS][KG_MAX];

        for (int i = tid; i < 3 * ROWS * KG; i += blockDim.x) {
            const int mat = i / (ROWS * KG);
            const int rem = i % (ROWS * KG);
            const int r   = rem / KG;
            const int g   = rem % KG;
            const int row = row_base + r;
            const unsigned int* szp = (mat == 0) ? R_sz : ((mat == 1) ? K_sz : V_sz);
            float sc, zr;
            unpack_int8_sz_batch(szp[row * KG + g], sc, zr);
            s_scale[mat][r][g] = sc;
            s_zero[mat][r][g]  = zr;
        }
        __syncthreads();

        // 累加器：3 矩阵 × ROWS 行 × BGRP slot（half2，48 寄存器）。
        half2 acc_r[ROWS][BGRP], acc_k[ROWS][BGRP], acc_v[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                acc_r[r][b] = __half2half2(0.f);
                acc_k[r][b] = __half2half2(0.f);
                acc_v[r][b] = __half2half2(0.f);
            }
        // 主循环：int8 idx 读一次 + 反量化一次 → 逐 slot FMA（权重读 1 份算 bcnt 份）。
        for (int kk = tid; kk < KV; kk += blockDim.x) {
            const int g = kk >> 5;
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int irow = (row_base + r) * KV + kk;
                const unsigned int pr = R_idx[irow];
                const unsigned int pk = K_idx[irow];
                const unsigned int pv = V_idx[irow];
                const float scr = s_scale[0][r][g], zrr = s_zero[0][r][g];
                const float sck = s_scale[1][r][g], zrk = s_zero[1][r][g];
                const float scv = s_scale[2][r][g], zrv = s_zero[2][r][g];
                __align__(16) __half wr[4], wk[4], wv[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int nbr = (pr >> (8 * j)) & 0xFF;
                    const int nbk = (pk >> (8 * j)) & 0xFF;
                    const int nbv = (pv >> (8 * j)) & 0xFF;
                    wr[j] = __float2half(scr * (float)nbr + zrr);
                    wk[j] = __float2half(sck * (float)nbk + zrk);
                    wv[j] = __float2half(scv * (float)nbv + zrv);
                }
                const half2 wr0 = *reinterpret_cast<const half2*>(&wr[0]);
                const half2 wr1 = *reinterpret_cast<const half2*>(&wr[2]);
                const half2 wk0 = *reinterpret_cast<const half2*>(&wk[0]);
                const half2 wk1 = *reinterpret_cast<const half2*>(&wk[2]);
                const half2 wv0 = *reinterpret_cast<const half2*>(&wv[0]);
                const half2 wv1 = *reinterpret_cast<const half2*>(&wv[2]);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    const int off = (b0 + b) * c + 4 * kk;
                    const half2 hxr0 = __floats2half2_rn(xr[off],     xr[off + 1]);
                    const half2 hxr1 = __floats2half2_rn(xr[off + 2], xr[off + 3]);
                    const half2 hxk0 = __floats2half2_rn(xk[off],     xk[off + 1]);
                    const half2 hxk1 = __floats2half2_rn(xk[off + 2], xk[off + 3]);
                    const half2 hxv0 = __floats2half2_rn(xv[off],     xv[off + 1]);
                    const half2 hxv1 = __floats2half2_rn(xv[off + 2], xv[off + 3]);
                    acc_r[r][b] = __hfma2(hxr0, wr0, acc_r[r][b]);
                    acc_r[r][b] = __hfma2(hxr1, wr1, acc_r[r][b]);
                    acc_k[r][b] = __hfma2(hxk0, wk0, acc_k[r][b]);
                    acc_k[r][b] = __hfma2(hxk1, wk1, acc_k[r][b]);
                    acc_v[r][b] = __hfma2(hxv0, wv0, acc_v[r][b]);
                    acc_v[r][b] = __hfma2(hxv1, wv1, acc_v[r][b]);
                }
            }
        }
        float lr[ROWS][BGRP], lk[ROWS][BGRP], lv[ROWS][BGRP];
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                const float2 rrf = __half22float2(acc_r[r][b]);
                const float2 rkf = __half22float2(acc_k[r][b]);
                const float2 rvf = __half22float2(acc_v[r][b]);
                lr[r][b] = rrf.x + rrf.y;
                lk[r][b] = rkf.x + rkf.y;
                lv[r][b] = rvf.x + rvf.y;
            }

        __shared__ float partial_r[4 /*warp*/][ROWS][BGRP];
        __shared__ float partial_k[4 /*warp*/][ROWS][BGRP];
        __shared__ float partial_v[4 /*warp*/][ROWS][BGRP];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                float vr = lr[r][b];
                float vk = lk[r][b];
                float vv = lv[r][b];
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) {
                    vr += __shfl_down_sync(0xffffffffu, vr, off2);
                    vk += __shfl_down_sync(0xffffffffu, vk, off2);
                    vv += __shfl_down_sync(0xffffffffu, vv, off2);
                }
                if (lane == 0) {
                    partial_r[warp][r][b] = vr;
                    partial_k[warp][r][b] = vk;
                    partial_v[warp][r][b] = vv;
                }
            }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row_base + r;
                if (row < c) {
                    #pragma unroll
                    for (int b = 0; b < BGRP; b++) {
                        if (b >= bcnt) break;
                        float sr_ = 0.f, sk_ = 0.f, sv_ = 0.f;
                        #pragma unroll
                        for (int w = 0; w < 4; w++) {
                            sr_ += partial_r[w][r][b];
                            sk_ += partial_k[w][r][b];
                            sv_ += partial_v[w][r][b];
                        }
                        const int off = (b0 + b) * c + row;
                        out_r[off] = sr_;
                        out_k[off] = sk_;
                        out_v[off] = __float2half(sv_);
                    }
                }
            }
        }
        return;
    }

    // mid 投影分支：权重行读一次，逐 slot 累加（bcnt 份 dot 共享权重读取）。
    const int mid_idx = flat - c / ROWS;
    float local_dot[BGRP];
    #pragma unroll
    for (int b = 0; b < BGRP; b++) local_dot[b] = 0.f;
    int chain = 3;
    int row = 0;
    const float* wsrc = nullptr;
    const float* xsrc[BGRP];
    if (mid_idx < vm) {
        chain = 0; row = mid_idx;
        wsrc = V1 + (long long)row * c;
        for (int b = 0; b < bcnt; b++) xsrc[b] = xv + (b0 + b) * c;
    } else if (mid_idx < vm + wm) {
        chain = 1; row = mid_idx - vm;
        wsrc = W1 + (long long)row * c;
        for (int b = 0; b < bcnt; b++) xsrc[b] = xw + (b0 + b) * c;
    } else if (mid_idx < vm + wm + am) {
        chain = 2; row = mid_idx - vm - wm;
        wsrc = A1 + (long long)row * c;
        for (int b = 0; b < bcnt; b++) xsrc[b] = xa + (b0 + b) * c;
    } else {
        row = mid_idx - vm - wm - am;
        wsrc = G1 + (long long)row * c;
        for (int b = 0; b < bcnt; b++) xsrc[b] = xg + (b0 + b) * c;
    }
    for (int kk = tid; kk < c; kk += blockDim.x) {
        const float w = wsrc[kk];
        for (int b = 0; b < bcnt; b++) local_dot[b] += w * xsrc[b][kk];
    }
    __shared__ float sm[128][BGRP];
    #pragma unroll
    for (int b = 0; b < BGRP; b++) sm[tid][b] = local_dot[b];
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int b = 0; b < BGRP; b++) sm[tid][b] += sm[tid + stride][b];
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int b = 0; b < BGRP; b++) {
            if (b >= bcnt) break;
            const float result = sm[0][b];
            if (chain == 0) out_vm[(b0 + b) * vm + row] = result;
            else if (chain == 1) out_wm[(b0 + b) * wm + row] = tanhf(result);
            else if (chain == 2) out_am[(b0 + b) * am + row] = result;
            else out_gm[(b0 + b) * gm + row] = result;
        }
    }
}
"#;

/// gemv_lowrank_chain4 batch CUDA kernel（warp-per-row 版）：每 block 8 warp
/// 各算 1 个输出行 × BGRP slot（原版整 block 归约 1 行，M=2560 时 grid 太大
/// 占 14.9%——每 block 只做 512B 功重，syncthreads 6 次为主因）。
/// dispatch (M/8, ceil(batch/BGRP), 1)；block=256（8 warp × 32 thread）。
const GEMV_LOWRANK_CHAIN4_BATCH_SRC: &str = r#"
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void __launch_bounds__(256, 2) gemv_lowrank_chain4_batch(
    const float*  __restrict__ W2,   // [M, KW] fp32 行主序（共享权重）
    const float*  __restrict__ A2,   // [M, KA]
    const float*  __restrict__ V2,   // [M, KV]
    const float*  __restrict__ G2,   // [M, KG]
    const float*  __restrict__ xw,   // [batch, KW]
    const float*  __restrict__ xa,   // [batch, KA]
    const float*  __restrict__ xv,   // [batch, KV]
    const float*  __restrict__ xg,   // [batch, KG]
    const float*  __restrict__ w0,   // [M]（共享）
    const float*  __restrict__ a0,   // [M]
    const float*  __restrict__ v0,   // [M]
    const float*  __restrict__ scale,// [1]
    const __half* __restrict__ v_first, // [batch, M] fp16
    __half* __restrict__ out_w,      // [batch, M] fp16
    __half* __restrict__ out_a,      // [batch, M] fp16
    __half* __restrict__ out_v,      // [batch, M] fp16（读改写）
    __half* __restrict__ out_g,      // [batch, M] fp16
    const int m,
    const int kw,
    const int ka,
    const int kv,
    const int kg,
    const int batch)
{
    constexpr int BGRP = 4;
    constexpr int WARPS = 8;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row  = blockIdx.x * WARPS + warp;
    if (row >= m) return;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);

    const float sc = scale[0];
    #pragma unroll
    for (int bi = 0; bi < BGRP; bi++) {
        if (bi >= bcnt) break;
        const int b = b0 + bi;
        const int mo = b * m + row;
        const float* xwb = xw + b * kw;
        const float* xab = xa + b * ka;
        const float* xvb = xv + b * kv;
        const float* xgb = xg + b * kg;

        float lw = 0.f, la = 0.f, lv = 0.f, lg = 0.f;
        for (int k = lane; k < kw; k += 32) lw += xwb[k] * W2[row * kw + k];
        for (int k = lane; k < ka; k += 32) la += xab[k] * A2[row * ka + k];
        for (int k = lane; k < kv; k += 32) lv += xvb[k] * V2[row * kv + k];
        for (int k = lane; k < kg; k += 32) lg += sigmoidf_(xgb[k]) * G2[row * kg + k];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lw += __shfl_down_sync(0xffffffffu, lw, off);
            la += __shfl_down_sync(0xffffffffu, la, off);
            lv += __shfl_down_sync(0xffffffffu, lv, off);
            lg += __shfl_down_sync(0xffffffffu, lg, off);
        }
        if (lane == 0) {
            out_w[mo] = __float2half(expf(sc * sigmoidf_(lw + w0[row])));
            out_a[mo] = __float2half(sigmoidf_(la + a0[row]));
            const float vcur = __half2float(out_v[mo]);
            out_v[mo] = __float2half(vcur + sigmoidf_(lv + v0[row]) * (__half2float(v_first[mo]) - vcur));
            out_g[mo] = __float2half(lg);
        }
    }
}
"#;

/// ffn_value_sparse_add batch CUDA kernel：r2 为 [batch, fh]，x 为 [batch, C]。
/// dispatch (fh/TILE, c/C_TILE, batch)。
const FFN_VALUE_SPARSE_BATCH_SRC: &str = r#"
extern "C" __global__ void ffn_value_sparse_add_batch(
    const float*    __restrict__ r2,          // [batch, fh] relu² 输出
    const __half*   __restrict__ value_tiled, // [fh*C] 平铺布局（共享）
    float*          __restrict__ x,           // [batch, C] 就地原子累加
    const int c,
    const int fh)
{
    constexpr int TILE    = 128;
    constexpr int C_TILE  = 256;
    __shared__ float r2_slice[TILE];
    __shared__ int   nnz_ids[TILE];
    __shared__ int   nnz_count;
    __shared__ int   warp_counts[TILE / 32];
    __shared__ int   warp_prefix[TILE / 32];

    const int f_block = blockIdx.x;
    const int c_block = blockIdx.y;
    const int b       = blockIdx.z;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int start_f = f_block * TILE;
    const float* r2b  = r2 + b * fh;
    float* xb         = x + b * c;

    float r2v = 0.f;
    bool  nonzero = false;
    int   local_pos = 0;
    if (tid < TILE) {
        r2v = r2b[start_f + tid];
        r2_slice[tid] = r2v;
        nonzero = (r2v != 0.0f);
        unsigned mask = __ballot_sync(0xffffffffu, nonzero);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) warp_counts[warp] = __popc(mask);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
#pragma unroll
        for (int w = 0; w < TILE / 32; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();
    if (tid < TILE && nonzero) {
        nnz_ids[warp_prefix[warp] + local_pos] = tid;
    }
    __syncthreads();

    const int c_blocks = c / C_TILE;
    const int tile_base = ((f_block * c_blocks + c_block) * TILE) * C_TILE;
    const int c0 = c_block * C_TILE + tid * 2;
    float acc0 = 0.f, acc1 = 0.f;
    for (int i = 0; i < nnz_count; i += 2) {
        const int f0 = nnz_ids[i];
        const __half* w0 = value_tiled + (long long)tile_base + f0 * C_TILE + tid * 2;
        const float a0 = r2_slice[f0];
        acc0 += a0 * __half2float(w0[0]);
        acc1 += a0 * __half2float(w0[1]);
        if (i + 1 < nnz_count) {
            const int f1 = nnz_ids[i + 1];
            const __half* w1 = value_tiled + (long long)tile_base + f1 * C_TILE + tid * 2;
            const float a1 = r2_slice[f1];
            acc0 += a1 * __half2float(w1[0]);
            acc1 += a1 * __half2float(w1[1]);
        }
    }
    atomicAdd(xb + c0, acc0);
    atomicAdd(xb + c0 + 1, acc1);
}
"#;

/// rwkv_sample batch CUDA kernel：B slot 并行采样（每 slot 独立 logits/参数/seed）。
/// logits/temp/mask/counter 为 [batch, n]；token 为 [batch]；sampler 为 [batch, 8]；
/// hist 为 [batch, hist_len]。dispatch (1, batch, 1)，block=112。
const SAMPLE_BATCH_SRC: &str = r#"
__device__ __forceinline__ float u01_batch(unsigned int s) {
    s += 0x9E3779B9u;
    unsigned int z = s;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z ^= z >> 16;
    return (float)z / 4294967296.0f;
}

extern "C" __global__ void rwkv_sample_batch(
    const float*      __restrict__ logits,   // [batch, n]
    float*            __restrict__ token,    // [batch] 写入索引的 f32 位模式
    float*            __restrict__ temp,     // [batch, n] 工作区
    float*            __restrict__ mask,     // [batch, n] 工作区
    unsigned int*     __restrict__ counter,  // [batch, n] 直方图
    const float*      __restrict__ sampler,  // [batch, 8] 参数
    const unsigned int* __restrict__ hist,   // [batch, hist_len] 历史 token
    const int n)
{
    const int tid = threadIdx.x;
    const int b   = blockIdx.y;
    const int bo  = b * n;
    const float* sampler_b = sampler + b * 8;
    constexpr int BS = 112;
    constexpr int MAXK = 50;
    __shared__ float s_val[BS][MAXK];
    __shared__ int   s_idx[BS][MAXK];
    __shared__ float s_topval[MAXK];
    __shared__ int   s_topidx[MAXK];
    __shared__ float s_sorted[MAXK];
    __shared__ int   s_sortedidx[MAXK];
    __shared__ float s_fval[BS];
    __shared__ int   s_fidx[BS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_u;
    __shared__ float s_threshold;
    __shared__ float g_cutoff;

    const float temperature = sampler_b[0];
    const unsigned int top_k = __float_as_uint(sampler_b[1]);
    const float top_p = sampler_b[2];
    const unsigned int seed = __float_as_uint(sampler_b[3]);
    const float rep = sampler_b[4];
    const float freq = sampler_b[5];
    const float pres = sampler_b[6];
    const unsigned int hist_len = __float_as_uint(sampler_b[7]);
    const bool do_topk = (top_k > 0u && top_k < (unsigned int)n);
    const int K = do_topk ? (int)top_k : 0;

    float* temp_b = temp + bo;
    float* mask_b = mask + bo;
    unsigned int* counter_b = counter + bo;
    const float* logits_b = logits + bo;
    const unsigned int* hist_b = hist + b * hist_len;

    // 1. 载入 logits
    for (int i = tid; i < n; i += BS) temp_b[i] = logits_b[i];

    // 2. 惩罚
    if (hist_len > 0u && (rep != 1.0f || freq != 0.0f || pres != 0.0f)) {
        for (int i = tid; i < n; i += BS) counter_b[i] = 0u;
        __syncthreads();
        for (int h = tid; h < (int)hist_len; h += BS) {
            atomicAdd(&counter_b[hist_b[h]], 1u);
        }
        __syncthreads();
        for (int i = tid; i < n; i += BS) {
            const unsigned int cnt = counter_b[i];
            float l = temp_b[i];
            if (cnt > 0u) {
                if (rep != 1.0f) l = l > 0.0f ? l / rep : l * rep;
                if (pres != 0.0f) l -= pres;
            }
            if (freq != 0.0f) l -= freq * (float)cnt;
            temp_b[i] = l;
        }
        __syncthreads();
    }

    // 3. temperature
    float invT = 1.0f / temperature;
    if (!(temperature > 0.0f)) invT = 1.0f;
    for (int i = tid; i < n; i += BS) temp_b[i] *= invT;
    __syncthreads();

    if (K > 0 && K <= MAXK) {
        // ================= 快速路径：单遍 top-K =================
        for (int j = 0; j < MAXK; j++) { s_val[tid][j] = -1e30f; s_idx[tid][j] = -1; }
        for (int i = tid; i < n; i += BS) {
            const float v = temp_b[i];
            if (v > s_val[tid][K - 1]) {
                int pos = K - 1;
                while (pos > 0 && v > s_val[tid][pos - 1]) {
                    s_val[tid][pos] = s_val[tid][pos - 1];
                    s_idx[tid][pos] = s_idx[tid][pos - 1];
                    --pos;
                }
                s_val[tid][pos] = v;
                s_idx[tid][pos] = i;
            }
        }
        __syncthreads();

        if (tid == 0) {
            auto sift = [&](int i, int h) {
                while (true) {
                    int l = 2 * i + 1, r = 2 * i + 2, m = i;
                    if (l < h && s_topval[l] < s_topval[m]) m = l;
                    if (r < h && s_topval[r] < s_topval[m]) m = r;
                    if (m == i) break;
                    float tv = s_topval[i]; s_topval[i] = s_topval[m]; s_topval[m] = tv;
                    int ti = s_topidx[i]; s_topidx[i] = s_topidx[m]; s_topidx[m] = ti;
                    i = m;
                }
            };
            for (int j = 0; j < K; j++) { s_topval[j] = s_val[0][j]; s_topidx[j] = s_idx[0][j]; }
            for (int j = K / 2 - 1; j >= 0; j--) sift(j, K);
            for (int th = 1; th < BS; th++) {
                for (int j = 0; j < K; j++) {
                    const float v = s_val[th][j];
                    if (v <= -1e29f) break;
                    if (v > s_topval[0]) {
                        s_topval[0] = v; s_topidx[0] = s_idx[th][j];
                        sift(0, K);
                    }
                }
            }
            for (int r = K; r > 0; r--) {
                s_sorted[r - 1] = s_topval[0];
                s_sortedidx[r - 1] = s_topidx[0];
                s_topval[0] = s_topval[r - 1];
                s_topidx[0] = s_topidx[r - 1];
                sift(0, r - 1);
            }
            s_threshold = s_sorted[K - 1];  // 第 K 大（降序末位）= 保留边界
        }
        __syncthreads();

        for (int i = tid; i < n; i += BS) if (temp_b[i] < s_threshold) temp_b[i] = -1e30f;
        __syncthreads();
    } else {
        // ================= 兜底路径 =================
        if (do_topk) {
            for (int i = tid; i < n; i += BS) mask_b[i] = 0.0f;
            __syncthreads();
            if (tid == 0) s_threshold = -1e30f;
            __syncthreads();
            for (unsigned int round = 0u; round < top_k; round++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask_b[i] == 0.0f && temp_b[i] > lm) { lm = temp_b[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                for (int step = BS >> 1; step > 0; step >>= 1) {
                    if (tid < step) {
                        const float bv = s_fval[tid + step];
                        const int   bi = s_fidx[tid + step];
                        if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                            s_fval[tid] = bv; s_fidx[tid] = bi;
                        }
                    }
                    __syncthreads();
                }
                if (tid == 0) { s_threshold = s_fval[0]; mask_b[s_fidx[0]] = 1.0f; }
                __syncthreads();
            }
            for (int i = tid; i < n; i += BS) if (temp_b[i] < s_threshold) temp_b[i] = -1e30f;
            __syncthreads();
        }
    }

    // 7. softmax：max -> exp -> normalize
    // 归约为非幂 block 安全版：BS=112 非 2 的幂，纯树归约（step 减半）会让
    // 部分 warp 的结果成为"孤儿"（如 step=7 时 s_fval[5..6] 不再被合并），
    // max 漏读/sum 漏加 → m 偏小 → exp 上溢 inf（实测 temp≈0 时选错 token）。
    // 修法：先把尾部 [P2, BS) 并入 [0, BS-P2)（P2 = ≤BS 的最大 2 幂），再 2 幂树归约。
    {
        float lm = -1e30f;
        for (int i = tid; i < n; i += BS) lm = fmaxf(lm, temp_b[i]);
        s_fval[tid] = lm;
        __syncthreads();
        {
            constexpr int P2 = 64;  // BS=112 → 64 + 48
            if (tid >= P2 && tid < BS) s_fval[tid - P2] = fmaxf(s_fval[tid - P2], s_fval[tid]);
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] = fmaxf(s_fval[tid], s_fval[tid + step]);
                __syncthreads();
            }
        }
        const float m = s_fval[0];
        __syncthreads();
        float s = 0.0f;
        for (int i = tid; i < n; i += BS) {
            const float v = expf(temp_b[i] - m);
            temp_b[i] = v;
            s += v;
        }
        s_fval[tid] = s;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (total > 0.0f) {
            for (int i = tid; i < n; i += BS) temp_b[i] /= total;
        }
        __syncthreads();
    }

    // 8. top-p
    if (top_p > 0.0f && top_p < 1.0f) {
        if (K > 0 && K <= MAXK) {
            if (tid == 0) {
                float cum = 0.0f, cutoffv = -1e30f;
                for (int j = K - 1; j >= 0; j--) {
                    const int idx = s_sortedidx[j];
                    cum += temp_b[idx];
                    cutoffv = temp_b[idx];
                    if (cum >= top_p) break;
                }
                g_cutoff = cutoffv;
            }
        } else {
            for (int i = tid; i < n; i += BS) mask_b[i] = 0.0f;
            __syncthreads();
            if (tid == 0) g_cutoff = 0.0f;
            __syncthreads();
            float cum = 0.0f;
            for (int cnt = 0; cnt < 512 && cum < top_p; cnt++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask_b[i] == 0.0f && temp_b[i] > lm) { lm = temp_b[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                for (int step = BS >> 1; step > 0; step >>= 1) {
                    if (tid < step) {
                        const float bv = s_fval[tid + step];
                        const int   bi = s_fidx[tid + step];
                        if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                            s_fval[tid] = bv; s_fidx[tid] = bi;
                        }
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    mask_b[s_fidx[0]] = 1.0f;
                    cum += s_fval[0];
                    g_cutoff = s_fval[0];
                }
                __syncthreads();
            }
            __syncthreads();
        }
        for (int i = tid; i < n; i += BS) if (temp_b[i] < g_cutoff) temp_b[i] = 0.0f;
        __syncthreads();
    }

    // 9. 采样
    if (K > 0 && K <= MAXK) {
        if (tid == 0) {
            float total = 0.0f;
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp_b[idx] > 0.0f) total += temp_b[idx];
            }
            const float u = u01_batch(seed) * total;
            float acc = 0.0f;
            int chosen = s_sortedidx[K - 1];
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp_b[idx] > 0.0f) {
                    acc += temp_b[idx];
                    if (acc > u) { chosen = idx; break; }
                }
            }
            token[b] = __int_as_float(chosen);
        }
    } else {
        float ts = 0.0f;
        for (int i = tid; i < n; i += BS) ts += temp_b[i];
        s_fval[tid] = ts;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (tid == 0) s_u = u01_batch(seed) * total;
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                acc += temp_b[i];
                if (acc > s_u) { chosen = i; break; }
            }
            token[b] = __int_as_float(chosen);
        }
    }
}
"#;

/// record_tokens batch CUDA kernel：把 in_tok[b] 各自追加到
/// out_seq[b*stride + atomicAdd(&cnt[b])]（每 slot 独立段独立计数）。
/// dispatch (batch, 1, 1)，每 block 单线程。
const RECORD_TOKENS_SRC: &str = r#"
extern "C" __global__ void rwkv_record_tokens(
    const unsigned int* __restrict__ in_tok,  // [batch] token（f32 位模式）
    unsigned int* __restrict__ out_seq,       // [batch, stride] 序列缓冲
    unsigned int* __restrict__ cnt,           // [batch] 计数器（各自原子自增）
    const int stride)
{
    const int b = blockIdx.x;
    const unsigned int i = atomicAdd(&cnt[b], 1u);
    out_seq[b * stride + i] = in_tok[b];
}
"#;

/// norm_lerp6 CUDA kernel：单 token 深度融合。
/// 语义对齐 Vulkan `norm_lerp6.comp`（f32 张量）：
///   ln1 = (x - mean) * inv_std * gamma + beta；mean/inv_std 对全 C 归约。
///   o_*[i] = ln1 + x_*[i] * (prev[i] - ln1)；state[i] = ln1。
/// 多 block 并行：grid = ceil(c/BLOCK)。每个 block 先对全 C 做**冗余归约**
/// （x 仅 ~10KB，命中 L2，各 block 独立算出同一 mean/inv_std），再并行 apply
/// 本 block 负责的 C 片段。避免单 block 时 67/68 个 SM 空闲的延迟瓶颈。
const NORM_LERP6_SRC: &str = r#"
extern "C" __global__ void norm_lerp6(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ xr,
    const float* __restrict__ xw,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    const float* __restrict__ xa,
    const float* __restrict__ xg,
    float* __restrict__ or_,
    float* __restrict__ ow,
    float* __restrict__ ok,
    float* __restrict__ ov,
    float* __restrict__ oa,
    float* __restrict__ og,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;

    // Phase 1：每个 block 独立对全 C 做冗余归约（x 命中 L2，开销小）。
    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    // Phase 2：每个 block apply 自己负责的 C 片段（gx*BLOCK .. min+BLOCK）。
    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[i];
        const float ln1  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[i];
        or_[i] = ln1 + xr[i] * (prev - ln1);
        ow[i]  = ln1 + xw[i] * (prev - ln1);
        ok[i]  = ln1 + xk[i] * (prev - ln1);
        ov[i]  = ln1 + xv[i] * (prev - ln1);
        oa[i]  = ln1 + xa[i] * (prev - ln1);
        og[i]  = ln1 + xg[i] * (prev - ln1);
        state[i] = ln1;
    }
}
"#;

/// cmix_norm_lerp CUDA kernel：channel-mix 深度融合。
/// 语义对齐 Vulkan `cmix_norm_lerp.comp`（f32 张量）：
///   ln2 = (x - mean) * inv_std * gamma + beta；out_xb[i] = ln2 + coeff[i]*(prev-ln2)；state[i]=ln2。
/// 多 block 并行：grid = ceil(c/BLOCK)，每个 block 冗余归约全 C 后分段 apply。
const CMIX_NORM_LERP_SRC: &str = r#"
extern "C" __global__ void cmix_norm_lerp(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ coeff,
    float* __restrict__ out_xb,
    const int c,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;
    const int gx   = blockIdx.x;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    const int start = gx * blockDim.x;
    const int end   = min(start + blockDim.x, c);
    #pragma unroll 4
    for (int i = start + tid; i < end; i += blockDim.x) {
        const float val  = x[i];
        const float ln2  = (val - mean) * inv_std * gamma[i] + beta[i];
        const float prev = state[i];
        out_xb[i] = ln2 + coeff[i] * (prev - ln2);
        state[i] = ln2;
    }
}
"#;

/// norm CUDA kernel：per-row layer norm + affine。
/// 语义对齐 Vulkan `norm.comp`（f32 输入/输出，affine）：
///   layout x[b][head][c]，gamma/beta 跨 batch 共享（[head][c]）。
///   每个 block（256 线程）归一化一个 (head,batch) 行。
const NORM_SRC: &str = r#"
extern "C" __global__ void rwkv_norm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    const int c,
    const int h,
    const float eps)
{
    __shared__ float s_val[32];
    __shared__ float s_sq[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = (blockDim.x + 31) >> 5;

    const int row   = blockIdx.x;
    const int b     = row / h;
    const int hh    = row - b * h;
    const int x_base = b * c * h + hh * c;
    const int g_base = hh * c;

    float sum = 0.f;
    float sq  = 0.f;
    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[x_base + i];
        sum += v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        sq  += __shfl_down_sync(0xffffffffu, sq, off);
    }
    if (lane == 0) { s_val[warp] = sum; s_sq[warp] = sq; }
    __syncthreads();
    if (tid == 0) {
        float tsum = 0.f;
        float tsq  = 0.f;
        for (int w = 0; w < nw; ++w) { tsum += s_val[w]; tsq += s_sq[w]; }
        const float mean = tsum / (float)c;
        const float variance = tsq / (float)c - mean * mean;
        s_val[0] = mean;
        s_sq[0]  = rsqrtf(variance + eps);
    }
    __syncthreads();
    const float mean    = s_val[0];
    const float inv_std = s_sq[0];

    for (int i = tid; i < c; i += blockDim.x) {
        const float v = x[x_base + i];
        y[x_base + i] = (v - mean) * inv_std * gamma[g_base + i] + beta[g_base + i];
    }
}
"#;

/// fuse_ka_dplr_norm CUDA kernel：fuse_ka + dplr(S 更新) + group_norm + sum_rk_rk 一次 dispatch。
/// 语义对齐 Vulkan `fuse_ka_dplr_norm.comp`（单 token 路径）：
///   kk_l2_i = normalize(k_i * k_k_i)；b_i = -kk_l2_i * a_i
///   k_mod_i = k_i * (1 + k_a_i * (a_i - 1))
///   S 更新：S[row,j] = S[row,j]*w[j] + sa[row]*b[j] + v[row]*k_mod[j]；y[row] = S@r
///   y_norm[row] = group_norm(y) + sum(r*k_mod*r_k) * v[row]
/// 每个 block 处理一个 (head,batch)；128 线程 = 4 warp。
/// warp-per-row 映射：warp 串行处理 row ≡ warp (mod 4) 的状态行，lane ↔ 连续列
/// （j0=lane、j1=lane+32）——state 行访问每条 warp 指令 128B 连续（100% coalesced；
/// 旧映射 row=t/2 隔行分片时每 2 lane 才 8B 连续，仅 25% sector 利用率）。
/// sa/y 行内归约用 warp 蝶形 shuffle（替代 shared 树归约，行循环内零 barrier）。
/// a/v/w 为 fp16，其余 f32。
const FUSE_KA_DPLR_NORM_SRC: &str = r#"
extern "C" __global__ void fuse_ka_dplr_norm(
    float* __restrict__ s,          // [batch][head][n][n] 状态（in-place 更新）
    const float* __restrict__ k,    // [batch][head][n]
    const float* __restrict__ kk,   // k_k [head][n]
    const __half* __restrict__ a,   // [batch][head][n]
    const float* __restrict__ ka,   // k_a [head][n]
    const float* __restrict__ r,    // [batch][head][n]
    const __half* __restrict__ v,   // [batch][head][n]
    const __half* __restrict__ w,   // [batch][head][n]
    const float* __restrict__ gamma,// [head][n]
    const float* __restrict__ beta, // [head][n]
    const float* __restrict__ rk,   // r_k [head][n]
    float* __restrict__ km,         // k_mod [batch][head][n]
    float* __restrict__ /*y*/,      // [batch][head][n]（本融合 kernel 不再写 y）
    float* __restrict__ yn,         // y_norm [batch][head][n]
    const int h,
    const int n,
    const float eps,
    const float gn_eps)
{
    constexpr int N_MAX = 64;

    __shared__ float sh_a[N_MAX];
    __shared__ float sh_b[N_MAX];
    __shared__ float sh_k[N_MAX];
    __shared__ float sh_w[N_MAX];
    __shared__ float sh_r[N_MAX];
    __shared__ float yv[N_MAX];
    __shared__ float sq[128];
    __shared__ float sqY[N_MAX];
    __shared__ float sqY2[N_MAX];
    __shared__ float ssRed[N_MAX];
    __shared__ float mean;
    __shared__ float inv_std;
    __shared__ float s_acc;

    const int head  = blockIdx.x;
    const int batch = blockIdx.y;
    const int t     = threadIdx.x;
    const int warp  = t >> 5;    // 0..3
    const int lane  = t & 31;    // 0..31
    if (t >= 2 * n) return;      // 与旧映射 row = t/2 >= n 的截断一致（n=64 时恒 false）

    const int v_base = batch * (h * n) + head * n;
    const int w_base = head * n;
    const int s_base = batch * (h * n * n) + head * (n * n);

    // Phase 0：L2 范数（保持旧版冗余归约：每行 2 线程各算一次 → 2 倍和，与 CPU 参考一致）
    {
        const int r2 = t >> 1;
        const float k_i  = k[v_base + r2];
        const float kk_i = k_i * kk[w_base + r2];
        sq[t] = kk_i * kk_i;
    }
    __syncthreads();
    for (int step = 128 >> 1; step > 0; step >>= 1) {
        if (t < step) sq[t] += sq[t + step];
        __syncthreads();
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sq[0]), eps);

    // Phase 1：t < n 的线程一人一列填充按列 shared（lane↔列，coalesced）
    if (t < n) {
        const float kc  = k[v_base + t];
        const float kkc = kc * kk[w_base + t];
        const float ac  = __half2float(a[v_base + t]);
        const float kl2 = kkc * inv_norm;
        sh_a[t] = kl2;
        sh_b[t] = -kl2 * ac;
        sh_k[t] = kc * (1.0f + ka[w_base + t] * (ac - 1.0f));
        sh_w[t] = __half2float(w[v_base + t]);
        sh_r[t] = r[v_base + t];
    }
    __syncthreads();
    if (t < n) km[v_base + t] = sh_k[t];

    // Phase 2+3：warp-per-row —— 每 warp 串行处理 row ≡ warp (mod 4) 的状态行，
    // lane ↔ 连续列（j0=lane、j1=lane+32），S 行元素寄存器化（每 lane 2 个），
    // 一遍读一遍写；sa/y 行内归约用 warp 蝶形 shuffle（行循环内零 barrier）。
    const int j0 = lane;
    const int j1 = lane + 32;
    const bool has_j1 = j1 < n;
    for (int row = warp; row < n; row += 4) {
        const int s_row_base = s_base + row * n;
        const float v_i = __half2float(v[v_base + row]);
        float s0 = s[s_row_base + j0];
        float s1 = has_j1 ? s[s_row_base + j1] : 0.0f;
        // sa[row] = sum_j S[row,j] * kk_l2[j]：本 lane 2 列部分和 + 蝶形归约（全 lane 得全和）
        float sa_part = s0 * sh_a[j0];
        if (has_j1) sa_part = fmaf(s1, sh_a[j1], sa_part);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sa_part += __shfl_xor_sync(0xffffffffu, sa_part, mask);
        }
        // S[row,j] = S[row,j]*w[j] + sa[row]*b[j] + v[row]*k_mod[j]
        s0 = s0 * sh_w[j0] + sa_part * sh_b[j0] + v_i * sh_k[j0];
        if (has_j1) s1 = s1 * sh_w[j1] + sa_part * sh_b[j1] + v_i * sh_k[j1];
        // y[row] = sum_j S_new[row,j] * r[j]
        float y_part = s0 * sh_r[j0];
        if (has_j1) y_part = fmaf(s1, sh_r[j1], y_part);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            y_part += __shfl_xor_sync(0xffffffffu, y_part, mask);
        }
        if (lane == 0) yv[row] = y_part;
        s[s_row_base + j0] = s0;
        if (has_j1) s[s_row_base + j1] = s1;
    }
    __syncthreads();

    // Phase 4+5：group-norm(y) 与 s 归约（t < n 一人一行，树归约同旧版结构）
    if (t < n) {
        const float y_i = yv[t];
        sqY[t]   = y_i;
        sqY2[t]  = y_i * y_i;
        ssRed[t] = sh_r[t] * sh_k[t] * rk[w_base + t];
    }
    __syncthreads();
    for (int step = n >> 1; step > 0; step >>= 1) {
        if (t < step) {
            sqY[t]   += sqY[t + step];
            sqY2[t]  += sqY2[t + step];
            ssRed[t] += ssRed[t + step];
        }
        __syncthreads();
    }
    if (t == 0) {
        const float ssum = sqY[0];
        const float ssq  = sqY2[0];
        mean    = ssum / (float)n;
        const float variance = ssq / (float)n - mean * mean;
        inv_std = rsqrtf(variance + gn_eps);
        s_acc   = ssRed[0];
    }
    __syncthreads();

    // Phase 6：y_norm[row] = (y[row]-mean)*inv_std*gamma[row]+beta[row] + s*v[row]
    if (t < n) {
        const float v_i = __half2float(v[v_base + t]);
        const float normalized =
            (yv[t] - mean) * inv_std * gamma[w_base + t] + beta[w_base + t];
        yn[v_base + t] = normalized + s_acc * v_i;
    }
}
"#;

/// gemv_rkv_stage1 CUDA kernel：r/k/v 三个 C×C 投影 + v1/w1/a1/g1 四个 mid 投影，一次 dispatch。
/// 语义对齐 Vulkan `gemv_rkv_stage1.comp`：
///   r = xr @ R^T；k = xk @ K^T；v = xv @ V^T（fp16 权重，f32 输入/累加，v 输出 fp16）
///   v_mid = xv @ V1；w_mid = tanh(xw @ W1)；a_mid = xa @ A1；g_mid = xg @ G1（fp32 权重）
/// dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)：每 block 128 线程。
///   前 C/ROWS 个 block 各算 ROWS=4 行 r/k/v；后各算一个 mid 输出。
const GEMV_RKV_STAGE1_SRC: &str = r#"
extern "C" __global__ void gemv_rkv_stage1(
    const __half* __restrict__ R,   // [C,C] fp16
    const __half* __restrict__ K,   // [C,C] fp16
    const __half* __restrict__ V,   // [C,C] fp16
    const float*  __restrict__ V1,  // [VM,C] fp32
    const float*  __restrict__ W1,  // [WM,C] fp32
    const float*  __restrict__ A1,  // [AM,C] fp32
    const float*  __restrict__ G1,  // [GM,C] fp32
    const float*  __restrict__ xr,  // [C]
    const float*  __restrict__ xk,  // [C]
    const float*  __restrict__ xv,  // [C]
    const float*  __restrict__ xw,  // [C]
    const float*  __restrict__ xa,  // [C]
    const float*  __restrict__ xg,  // [C]
    float* __restrict__ out_r,      // [C]
    float* __restrict__ out_k,      // [C]
    __half* __restrict__ out_v,     // [C] fp16
    float* __restrict__ out_vm,     // [VM]
    float* __restrict__ out_wm,     // [WM]
    float* __restrict__ out_am,     // [AM]
    float* __restrict__ out_gm,     // [GM]
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm)
{
    constexpr int ROWS = 4;
    const int tid = threadIdx.x;
    const int flat = blockIdx.x;

    if (flat < c / ROWS) {
        const int row_base = flat * ROWS;
        // 半精度累加器（half2），对齐 Albatross h2stage_hfma2 版本：__hfma2 输出 half2，
        // 累加保持在 half2，最终一次转 float 归约。
        half2 lr2[ROWS];
        half2 lk2[ROWS];
        half2 lv2[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            lr2[r] = __half2half2(0.f);
            lk2[r] = __half2half2(0.f);
            lv2[r] = __half2half2(0.f);
        }
        // 向量化主循环：对齐 Albatross rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe。
        // 每线程每次迭代处理 2 个 k（x 转 half2、权重按 half2 读），用 __hfma2 半精度乘加
        // （FP16 FMA 吞吐为 FP32 的 2 倍），权重与 x 均只读一次即可贡献给 ROWS 行。
        // kq 步进 2 且 blockDim 为偶数，保证 half2 的 4 字节对齐。
        const int c2 = c & ~1;
        for (int kq = tid * 2; kq < c2; kq += blockDim.x * 2) {
            const half2 hxr = __floats2half2_rn(xr[kq], xr[kq + 1]);
            const half2 hxk = __floats2half2_rn(xk[kq], xk[kq + 1]);
            const half2 hxv = __floats2half2_rn(xv[kq], xv[kq + 1]);
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const half2 wr = *reinterpret_cast<const half2*>(R + (long long)(row_base + r) * c + kq);
                const half2 wk = *reinterpret_cast<const half2*>(K + (long long)(row_base + r) * c + kq);
                const half2 wv = *reinterpret_cast<const half2*>(V + (long long)(row_base + r) * c + kq);
                lr2[r] = __hfma2(hxr, wr, lr2[r]);
                lk2[r] = __hfma2(hxk, wk, lk2[r]);
                lv2[r] = __hfma2(hxv, wv, lv2[r]);
            }
        }
        // 累加器 half2 → float（每行 2 分量求和），供后续 warp 归约。
        float lr[ROWS];
        float lk[ROWS];
        float lv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float2 rf = __half22float2(lr2[r]);
            const float2 kf = __half22float2(lk2[r]);
            const float2 vf = __half22float2(lv2[r]);
            lr[r] = rf.x + rf.y;
            lk[r] = kf.x + kf.y;
            lv[r] = vf.x + vf.y;
        }
        // 尾部标量兜底（c 为奇数时）。
        if ((c & 1) && tid == 0) {
            const int kk = c - 1;
            const float xrv = xr[kk];
            const float xkv = xk[kk];
            const float xvv = xv[kk];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int a = (row_base + r) * c + kk;
                if (row_base + r < c) {
                    lr[r] += __half2float(R[a]) * xrv;
                    lk[r] += __half2float(K[a]) * xkv;
                    lv[r] += __half2float(V[a]) * xvv;
                }
            }
        }
        // warp shuffle 归约 sr/sk/sv（对齐 Albatross row1_linear_exact4_kernel）。
        __shared__ float partial_r[4 /*warp*/][ROWS];
        __shared__ float partial_k[4 /*warp*/][ROWS];
        __shared__ float partial_v[4 /*warp*/][ROWS];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float vr = lr[r];
            float vk = lk[r];
            float vv = lv[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vr += __shfl_down_sync(0xffffffffu, vr, off);
                vk += __shfl_down_sync(0xffffffffu, vk, off);
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) { partial_r[warp][r] = vr; partial_k[warp][r] = vk; partial_v[warp][r] = vv; }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float sr_ = 0.f, sk_ = 0.f, sv_ = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    sr_ += partial_r[w][r]; sk_ += partial_k[w][r]; sv_ += partial_v[w][r];
                }
                const int row = row_base + r;
                if (row < c) {
                    out_r[row] = sr_;
                    out_k[row] = sk_;
                    out_v[row] = __float2half(sv_);
                }
            }
        }
        return;
    }

    // mid 投影分支
    const int mid_idx = flat - c / ROWS;
    float local_dot = 0.f;
    int chain = 3;
    int row = 0;
    if (mid_idx < vm) {
        chain = 0; row = mid_idx;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += V1[row * c + kk] * xv[kk];
    } else if (mid_idx < vm + wm) {
        chain = 1; row = mid_idx - vm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += W1[row * c + kk] * xw[kk];
    } else if (mid_idx < vm + wm + am) {
        chain = 2; row = mid_idx - vm - wm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += A1[row * c + kk] * xa[kk];
    } else {
        row = mid_idx - vm - wm - am;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += G1[row * c + kk] * xg[kk];
    }
    __shared__ float sm[128];
    sm[tid] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float result = sm[0];
        if (chain == 0) out_vm[row] = result;
        else if (chain == 1) out_wm[row] = tanhf(result);
        else if (chain == 2) out_am[row] = result;
        else out_gm[row] = result;
    }
}
"#;

/// gemv_int8_rkv_stage1 CUDA kernel：r/k/v 三个 C×C int8 量化投影 + 四个 mid fp32 投影，一次 dispatch。
/// 语义对齐 Vulkan `gemv_int8_rkv_stage1.comp`：
///   r = xr @ R^T；k = xk @ K^T；v = xv @ V^T（int8 权重，f32 输入/累加，v 输出 fp16）
///   v_mid = xv @ V1；w_mid = tanh(xw @ W1)；a_mid = xa @ A1；g_mid = xg @ G1（fp32 权重）
/// int8 格式（每矩阵 W[C,C] 行主序，K=C 收缩，group=128）：
///   idx: uint32 [C, C/4]（每 uint32 打包 4 个 uint8 权重，字节序 = 权重索引低位优先）
///   sz:  uint32 [C, C/128]（每元素 = (scale: fp16 低16位 | zero: fp16 高16位)）
/// 反量化 w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]（无 LUT，直接字节提取）。
/// dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)：每 block 128 线程。
///   前 C/ROWS 个 block 各算 ROWS=4 行 r/k/v；后各算一个 mid 输出（与 fp16 版一致）。
const GEMV_INT8_RKV_STAGE1_SRC: &str = r#"
/// int8 量化权重解包辅助：`sz` 每元素 = (scale: fp16 低16位 | zero: fp16 高16位)。
__device__ __forceinline__ void unpack_int8_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}

extern "C" __global__ void gemv_int8_rkv_stage1(
    const unsigned int* __restrict__ R_idx,   // int8 idx [C, C/4]（4 字节/uint32）
    const unsigned int* __restrict__ R_sz,    // int8 sz  [C, C/128]
    const unsigned int* __restrict__ K_idx,
    const unsigned int* __restrict__ K_sz,
    const unsigned int* __restrict__ V_idx,
    const unsigned int* __restrict__ V_sz,
    const float*  __restrict__ V1,            // [VM,C] fp32
    const float*  __restrict__ W1,            // [WM,C]
    const float*  __restrict__ A1,            // [AM,C]
    const float*  __restrict__ G1,            // [GM,C]
    const float*  __restrict__ xr,            // [C]
    const float*  __restrict__ xk,            // [C]
    const float*  __restrict__ xv,            // [C]
    const float*  __restrict__ xw,            // [C]
    const float*  __restrict__ xa,            // [C]
    const float*  __restrict__ xg,            // [C]
    float* __restrict__ out_r,                // [C]
    float* __restrict__ out_k,                // [C]
    __half* __restrict__ out_v,               // [C] fp16
    float* __restrict__ out_vm,               // [VM]
    float* __restrict__ out_wm,               // [WM]
    float* __restrict__ out_am,               // [AM]
    float* __restrict__ out_gm,               // [GM]
    const int c,
    const int vm,
    const int wm,
    const int am,
    const int gm)
{
    constexpr int ROWS = 4;
    constexpr int KG_MAX = 32;   // C/128 上限（C ≤ 4096）。C=2560 → KG=20。
    const int tid  = threadIdx.x;
    const int flat = blockIdx.x;

    // r/k/v 分支：每个 block 处理 ROWS=4 行，int8 反量化 + 归约。
    if (flat < c / ROWS) {
        const int row_base = flat * ROWS;
        const int KV = c / 4;    // 每行 uint32 idx 数（4 字节/uint32）
        const int KG = c / 128;  // 每行 group 数

        __shared__ float s_scale[3][ROWS][KG_MAX];
        __shared__ float s_zero[3][ROWS][KG_MAX];

        // Phase 0：协作加载 3 矩阵 × ROWS 行的 scale/zero。无 LUT。
        for (int i = tid; i < 3 * ROWS * KG; i += blockDim.x) {
            const int mat = i / (ROWS * KG);
            const int rem = i % (ROWS * KG);
            const int r   = rem / KG;
            const int g   = rem % KG;
            const int row = row_base + r;
            const unsigned int* szp = (mat == 0) ? R_sz : ((mat == 1) ? K_sz : V_sz);
            float sc, zr;
            unpack_int8_sz(szp[row * KG + g], sc, zr);
            s_scale[mat][r][g] = sc;
            s_zero[mat][r][g]  = zr;
        }
        __syncthreads();

        // Phase 1：主循环，每 iter 反量化 4 权重/矩阵/行（1 uint32），半精度 __hfma2 累加。
        // 对齐 fp16 版：反量化结果转 __half 拼 half2，与 x 的 half2 用 __hfma2 乘加
        // （FP16 FMA 吞吐为 FP32 的 2 倍），累加保持在 half2，最终一次转 float 归约。
        half2 acc_r[ROWS], acc_k[ROWS], acc_v[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            acc_r[r] = __half2half2(0.f);
            acc_k[r] = __half2half2(0.f);
            acc_v[r] = __half2half2(0.f);
        }
        for (int kk = tid; kk < KV; kk += blockDim.x) {
            const half2 hxr0 = __floats2half2_rn(xr[4 * kk],     xr[4 * kk + 1]);
            const half2 hxr1 = __floats2half2_rn(xr[4 * kk + 2], xr[4 * kk + 3]);
            const half2 hxk0 = __floats2half2_rn(xk[4 * kk],     xk[4 * kk + 1]);
            const half2 hxk1 = __floats2half2_rn(xk[4 * kk + 2], xk[4 * kk + 3]);
            const half2 hxv0 = __floats2half2_rn(xv[4 * kk],     xv[4 * kk + 1]);
            const half2 hxv1 = __floats2half2_rn(xv[4 * kk + 2], xv[4 * kk + 3]);
            const int g = kk >> 5;   // 32 个 uint32/组
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int irow = (row_base + r) * KV + kk;
                const unsigned int pr = R_idx[irow];
                const unsigned int pk = K_idx[irow];
                const unsigned int pv = V_idx[irow];
                const float scr = s_scale[0][r][g], zrr = s_zero[0][r][g];
                const float sck = s_scale[1][r][g], zrk = s_zero[1][r][g];
                const float scv = s_scale[2][r][g], zrv = s_zero[2][r][g];
                __align__(16) __half wr[4], wk[4], wv[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int nbr = (pr >> (8 * j)) & 0xFF;
                    const int nbk = (pk >> (8 * j)) & 0xFF;
                    const int nbv = (pv >> (8 * j)) & 0xFF;
                    wr[j] = __float2half(scr * (float)nbr + zrr);
                    wk[j] = __float2half(sck * (float)nbk + zrk);
                    wv[j] = __float2half(scv * (float)nbv + zrv);
                }
                const half2 wr0 = *reinterpret_cast<const half2*>(&wr[0]);
                const half2 wr1 = *reinterpret_cast<const half2*>(&wr[2]);
                const half2 wk0 = *reinterpret_cast<const half2*>(&wk[0]);
                const half2 wk1 = *reinterpret_cast<const half2*>(&wk[2]);
                const half2 wv0 = *reinterpret_cast<const half2*>(&wv[0]);
                const half2 wv1 = *reinterpret_cast<const half2*>(&wv[2]);
                acc_r[r] = __hfma2(hxr0, wr0, acc_r[r]);
                acc_r[r] = __hfma2(hxr1, wr1, acc_r[r]);
                acc_k[r] = __hfma2(hxk0, wk0, acc_k[r]);
                acc_k[r] = __hfma2(hxk1, wk1, acc_k[r]);
                acc_v[r] = __hfma2(hxv0, wv0, acc_v[r]);
                acc_v[r] = __hfma2(hxv1, wv1, acc_v[r]);
            }
        }
        // 累加器 half2 → float（每行 2 分量求和），供后续 warp 归约。
        float lr[ROWS], lk[ROWS], lv[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float2 rrf = __half22float2(acc_r[r]);
            const float2 rkf = __half22float2(acc_k[r]);
            const float2 rvf = __half22float2(acc_v[r]);
            lr[r] = rrf.x + rrf.y;
            lk[r] = rkf.x + rkf.y;
            lv[r] = rvf.x + rvf.y;
        }

        // warp shuffle 归约（3 矩阵 × ROWS 行）。
        __shared__ float partial_r[4 /*warp*/][ROWS];
        __shared__ float partial_k[4 /*warp*/][ROWS];
        __shared__ float partial_v[4 /*warp*/][ROWS];
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float vr = lr[r];
            float vk = lk[r];
            float vv = lv[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                vr += __shfl_down_sync(0xffffffffu, vr, off);
                vk += __shfl_down_sync(0xffffffffu, vk, off);
                vv += __shfl_down_sync(0xffffffffu, vv, off);
            }
            if (lane == 0) { partial_r[warp][r] = vr; partial_k[warp][r] = vk; partial_v[warp][r] = vv; }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                float sr_ = 0.f, sk_ = 0.f, sv_ = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    sr_ += partial_r[w][r]; sk_ += partial_k[w][r]; sv_ += partial_v[w][r];
                }
                const int row = row_base + r;
                if (row < c) {
                    out_r[row] = sr_;
                    out_k[row] = sk_;
                    out_v[row] = __float2half(sv_);
                }
            }
        }
        return;
    }

    // mid 投影分支（与 fp16 版一致）。
    const int mid_idx = flat - c / ROWS;
    float local_dot = 0.f;
    int chain = 3;
    int row = 0;
    if (mid_idx < vm) {
        chain = 0; row = mid_idx;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += V1[row * c + kk] * xv[kk];
    } else if (mid_idx < vm + wm) {
        chain = 1; row = mid_idx - vm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += W1[row * c + kk] * xw[kk];
    } else if (mid_idx < vm + wm + am) {
        chain = 2; row = mid_idx - vm - wm;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += A1[row * c + kk] * xa[kk];
    } else {
        row = mid_idx - vm - wm - am;
        for (int kk = tid; kk < c; kk += blockDim.x) local_dot += G1[row * c + kk] * xg[kk];
    }
    __shared__ float sm[128];
    sm[tid] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] += sm[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float result = sm[0];
        if (chain == 0) out_vm[row] = result;
        else if (chain == 1) out_wm[row] = tanhf(result);
        else if (chain == 2) out_am[row] = result;
        else out_gm[row] = result;
    }
}
"#;

/// gemv_lowrank_chain4 CUDA kernel：融合 4 条低秩链第二级（w/a/g/v 的 w2/a2/g2/v2），
/// 一次 dispatch。语义对齐 Vulkan `gemv_lowrank_chain4.comp`，每 block 处理一个输出行（M=C）：
///   w：out = exp(scale[0] * sigmoid(xw @ W2[row] + w0[row]))          [K_W, scale, w0]
///   a：out = sigmoid(xa @ A2[row] + a0[row])                          [K_A, a0]
///   v：out_v[row] += sigmoid(xv @ V2[row] + v0[row]) * (v_first[row] - out_v[row])（原地）
///   g：out = sum_k sigmoid(xg[k]) * G2[row,k]（g 链 sigmoid 作用于 mid，无 bias）
/// 权重矩阵行主序 [M, K] fp32；x 为 mid 向量 [K] fp32；v_first 与四个输出为 fp16。
/// dispatch (M, 1, 1)：每 block 256 线程跨线程归约后由 tid==0 写输出。
const GEMV_LOWRANK_CHAIN4_SRC: &str = r#"
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void gemv_lowrank_chain4(
    const float*  __restrict__ W2,   // [M, KW] fp32 行主序
    const float*  __restrict__ A2,   // [M, KA]
    const float*  __restrict__ V2,   // [M, KV]
    const float*  __restrict__ G2,   // [M, KG]
    const float*  __restrict__ xw,   // [KW]
    const float*  __restrict__ xa,   // [KA]
    const float*  __restrict__ xv,   // [KV]
    const float*  __restrict__ xg,   // [KG]
    const float*  __restrict__ w0,   // [M]
    const float*  __restrict__ a0,   // [M]
    const float*  __restrict__ v0,   // [M]
    const float*  __restrict__ scale,// [1]
    const __half* __restrict__ v_first, // [M] fp16
    __half* __restrict__ out_w,      // [M] fp16
    __half* __restrict__ out_a,      // [M] fp16
    __half* __restrict__ out_v,      // [M] fp16（读改写）
    __half* __restrict__ out_g,      // [M] fp16
    const int m,
    const int kw,
    const int ka,
    const int kv,
    const int kg)
{
    constexpr int BLOCK = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float lw = 0.f, la = 0.f, lv = 0.f, lg = 0.f;
    for (int k = tid; k < kw; k += BLOCK) lw += xw[k] * W2[row * kw + k];
    for (int k = tid; k < ka; k += BLOCK) la += xa[k] * A2[row * ka + k];
    for (int k = tid; k < kv; k += BLOCK) lv += xv[k] * V2[row * kv + k];
    for (int k = tid; k < kg; k += BLOCK) lg += sigmoidf(xg[k]) * G2[row * kg + k];

    __shared__ float sw[BLOCK], sa[BLOCK], svr[BLOCK], sg[BLOCK];
    sw[tid] = lw; sa[tid] = la; svr[tid] = lv; sg[tid] = lg;
    __syncthreads();
    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sw[tid] += sw[tid + stride];
            sa[tid] += sa[tid + stride];
            svr[tid] += svr[tid + stride];
            sg[tid] += sg[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_w[row] = __float2half(expf(scale[0] * sigmoidf(sw[0] + w0[row])));
        out_a[row] = __float2half(sigmoidf(sa[0] + a0[row]));
        const float vcur = __half2float(out_v[row]);
        out_v[row] = __float2half(vcur + sigmoidf(svr[0] + v0[row]) * (__half2float(v_first[row]) - vcur));
        out_g[row] = __float2half(sg[0]);
    }
}
"#;

/// gemv_variant CUDA kernel：统一处理 9 个 gemv 变体（权重类型 × 输出变换）。
/// 语义对齐 Vulkan `gemv_f32io_relu2` / `gemv_f32io_add_mul` / `gemv_f32io_add`：
///   relu2   ：y = relu²(x @ A)  = max(0, dot)²
///   mul_add ：y += (x .* g) @ A（g 为 fp16 门控，逐元素作用于 x）
///   add     ：y += x @ A
/// 权重反量化由 `wtype` 选择（0=f16、2=int8），`op` 选择输出变换
/// （0=relu2、1=mul_add、2=add）。dispatch (M/4, batch, 1)，每 block 128 线程处理 4 行。
/// 与 gemv_f16 同构：每 block 处理 ROWS=4 行，跨线程归约后 tid==0 写输出。
const GEMV_VARIANT_SRC: &str = r#"
__device__ __forceinline__ void unpack_variant_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float relu2f(float x) { return x > 0.f ? x * x : 0.f; }

extern "C" __global__ void gemv_variant(
    const __half*         __restrict__ Af16,  // fp16 [M*K]（wtype==0 用）
    const unsigned int*   __restrict__ aidx,  // int8 idx [M,K/4]
    const __half*         __restrict__ alut,  // 保留（当前未使用）
    const unsigned int*   __restrict__ asz,   // int8 sz [M,K/128]
    const float*          __restrict__ x,     // [K*batch]
    const __half*         __restrict__ g,     // [K*batch] fp16 门控（op==1 用）
    float*                __restrict__ y,     // [M*batch]（累加式读改写）
    const int m,
    const int k,
    const int batch,
    const int wtype,   // 0=f16, 2=int8
    const int op)      // 0=relu2, 1=mul_add, 2=add
{
    const int tid  = threadIdx.x;
    const int b    = blockIdx.y;
    const int row0 = blockIdx.x * 4;
    const int k0   = b * k;
    const int m0   = b * m;
    const int kvi  = k / 4;   // int8 每行 uint32 数
    const int kg   = k / 128; // int8 每行 group 数
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    if (wtype == 0) {
        // fp16 向量化主循环：每线程每次迭代处理 4 个 k（x 按 float4、权重按 8B half4 读）。
        // 半精度累积（__hfma2，吞吐为 FP32 2 倍），4 行各持 2 个 half2 累加器。
        half2 hacc[4][2];
        #pragma unroll
        for (int r = 0; r < 4; r++) { hacc[r][0] = __half2half2(0.f); hacc[r][1] = __half2half2(0.f); }
        const int k4 = k & ~3;
        for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
            const float4 xv = *reinterpret_cast<const float4*>(x + k0 + kq);
            float gx = 1.f, gy = 1.f, gz = 1.f, gw = 1.f;
            if (op == 1) {
                // g 为 fp16，按 8B half4 加载（勿用 float4，避免 16B 对齐越界）。
                load_half4_f4(g + k0 + kq, gx, gy, gz, gw);
            }
            const half2 hx01 = __floats2half2_rn(xv.x * gx, xv.y * gy);
            const half2 hx23 = __floats2half2_rn(xv.z * gz, xv.w * gw);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const __half* wj = Af16 + (row0 + r) * k + kq;
                hacc[r][0] = __hfma2(hx01, *reinterpret_cast<const half2*>(wj), hacc[r][0]);
                hacc[r][1] = __hfma2(hx23, *reinterpret_cast<const half2*>(wj + 2), hacc[r][1]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float2 f0 = __half22float2(hacc[r][0]);
            const float2 f1 = __half22float2(hacc[r][1]);
            acc[r] = f0.x + f0.y + f1.x + f1.y;
        }
        // 尾部标量兜底（k 非 4 倍数时）。
        for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
            const float xv = x[k0 + kk];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                acc[r] += __half2float(Af16[row * k + kk]) * xv;
            }
        }
    } else {
        // int8 向量化路径：每线程每次迭代处理 4 行 × 4 个 k（1 uint32 = 4 字节权重），
        // 反量化后拼 half2，与 x 的 half2 用 __hfma2 累加（FP16 FMA 吞吐为 FP32 的 2 倍）。
        // scale/zero 按 group 内循环解一次，字节提取兼内联。
        half2 hacc[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) hacc[r] = __half2half2(0.f);
        const int kvi4 = k / 4;   // int8 每行 uint32 数（4 字节/uint32）
        for (int kq = tid; kq < kvi4; kq += blockDim.x) {
            const int kbase = kq * 4;
            const int gr = kg > 0 ? (kbase / 128) : 0;
            float gv0 = 1.f, gv1 = 1.f, gv2 = 1.f, gv3 = 1.f;
            if (op == 1) {
                const __half* gq = g + k0 + kbase;
                gv0 = __half2float(gq[0]); gv1 = __half2float(gq[1]);
                gv2 = __half2float(gq[2]); gv3 = __half2float(gq[3]);
            }
            const half2 hx01 = __floats2half2_rn(x[k0 + kbase] * gv0, x[k0 + kbase + 1] * gv1);
            const half2 hx23 = __floats2half2_rn(x[k0 + kbase + 2] * gv2, x[k0 + kbase + 3] * gv3);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const unsigned int p = aidx[row * kvi4 + kq];
                float sc, zr;
                unpack_variant_sz(asz[row * kg + gr], sc, zr);
                __align__(16) __half w[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int byte = (int)((p >> (j * 8)) & 0xFFu);
                    w[j] = __float2half(sc * (float)byte + zr);
                }
                const half2 w01 = *reinterpret_cast<const half2*>(&w[0]);
                const half2 w23 = *reinterpret_cast<const half2*>(&w[2]);
                hacc[r] = __hfma2(hx01, w01, hacc[r]);
                hacc[r] = __hfma2(hx23, w23, hacc[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float2 f = __half22float2(hacc[r]);
            acc[r] = f.x + f.y;
        }
    }

    // warp shuffle 归约（对齐 Albatross row1_linear_exact4_kernel<128,2>）：只有 1 次 __syncthreads。
    __shared__ float partial[4 /*warp*/][4 /*row*/];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[warp][r] = v;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < 4; w++) sum += partial[w][r];
            const int row = row0 + r;
            if (row < m) {
                if (op == 0) y[m0 + row] = relu2f(sum);
                else if (op == 3) y[m0 + row] = sum;
                else y[m0 + row] += sum;
            }
        }
    }
}
"#;

/// gemv_variant_mb CUDA kernel：batch 并发的**权重复用**版（信天翁 rows 模型）。
/// 与 gemv_variant 的区别：不是 grid.y=slot 各读全量权重（带宽 ×B），而是
/// 每 block 一次读权重（4 行 × K），在寄存器累加器中复用给 BGRP 个 slot——
/// weight-bound 场景带宽 ≈ 1/ceil(B/BGRP)。
/// dispatch (M/4, ceil(batch/BGRP), 1)；x/y 为 [batch, ...]（slot 主序）。
/// op 语义与 gemv_variant 一致（0=relu2、1=mul_add、2=add、3=plain）。
const GEMV_VARIANT_MB_SRC: &str = r#"
__device__ __forceinline__ void unpack_mb_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
__device__ __forceinline__ float relu2_mb(float x) { return x > 0.f ? x * x : 0.f; }
__device__ __forceinline__ void load_half4_f4_mb(
    const __half* p, float& x0, float& x1, float& x2, float& x3)
{
    const half2 h01 = *reinterpret_cast<const half2*>(p);
    const half2 h23 = *reinterpret_cast<const half2*>(p + 2);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    x0 = f01.x; x1 = f01.y; x2 = f23.x; x3 = f23.y;
}

extern "C" __global__ void __launch_bounds__(128, 4) gemv_variant_mb(
    const __half*         __restrict__ Af16,  // fp16 [M*K]（wtype==0 用）
    const unsigned int*   __restrict__ aidx,  // int8 idx [M,K/4]
    const __half*         __restrict__ alut,  // 保留（当前未使用）
    const unsigned int*   __restrict__ asz,   // int8 sz [M,K/128]
    const float*          __restrict__ x,     // [batch, K]
    const __half*         __restrict__ g,     // [batch, K] fp16 门控（op==1 用）
    float*                __restrict__ y,     // [batch, M]（op==2 累加式读改写）
    const int m,
    const int k,
    const int batch,
    const int wtype,   // 0=f16, 2=int8
    const int op)      // 0=relu2, 1=mul_add, 2=add, 3=plain
{
    constexpr int BGRP = 4;   // 每 block 复用一份权重的 slot 数（8 会寄存器溢出，
                              // 实测 spill 后反比单序列慢——profiling 43.5% 热点根因）
    const int tid  = threadIdx.x;
    const int b0   = blockIdx.y * BGRP;
    const int bcnt = min(BGRP, batch - b0);
    const int row0 = blockIdx.x * 4;
    const int kvi4 = k / 4;
    const int kg   = k / 128;

    // half2 累加器：4 行 × BGRP slot（fp16 FMA 吞吐路径，与单序列版一致）。
    half2 hacc[4][BGRP];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) hacc[r][b] = __half2half2(0.f);

    if (wtype == 0) {
        // ===== fp16 路径：权重 half4 读一次，逐 slot FMA =====
        const int k4 = k & ~3;
        for (int kq = tid * 4; kq < k4; kq += blockDim.x * 4) {
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b >= bcnt) break;
                const float4 xv = *reinterpret_cast<const float4*>(x + (b0 + b) * k + kq);
                float gx = 1.f, gy = 1.f, gz = 1.f, gw = 1.f;
                if (op == 1) {
                    load_half4_f4_mb(g + (b0 + b) * k + kq, gx, gy, gz, gw);
                }
                const half2 hx01 = __floats2half2_rn(xv.x * gx, xv.y * gy);
                const half2 hx23 = __floats2half2_rn(xv.z * gz, xv.w * gw);
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    const __half* wj = Af16 + (row0 + r) * k + kq;
                    hacc[r][b] = __hfma2(hx01, *reinterpret_cast<const half2*>(wj), hacc[r][b]);
                    hacc[r][b] = __hfma2(hx23, *reinterpret_cast<const half2*>(wj + 2), hacc[r][b]);
                }
            }
        }
    } else {
        // ===== int8 路径：反量化一次，逐 slot FMA =====
        for (int kq = tid; kq < kvi4; kq += blockDim.x) {
            const int kbase = kq * 4;
            const int gr = kg > 0 ? (kbase / 128) : 0;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const unsigned int p = aidx[row * kvi4 + kq];
                float sc, zr;
                unpack_mb_sz(asz[row * kg + gr], sc, zr);
                __align__(16) __half w[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int byte = (int)((p >> (8 * j)) & 0xFFu);
                    w[j] = __float2half(sc * (float)byte + zr);
                }
                const half2 w01 = *reinterpret_cast<const half2*>(&w[0]);
                const half2 w23 = *reinterpret_cast<const half2*>(&w[2]);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    const int xb = (b0 + b) * k + kbase;
                    float gv0 = 1.f, gv1 = 1.f, gv2 = 1.f, gv3 = 1.f;
                    if (op == 1) {
                        const __half* gq = g + xb;
                        gv0 = __half2float(gq[0]); gv1 = __half2float(gq[1]);
                        gv2 = __half2float(gq[2]); gv3 = __half2float(gq[3]);
                    }
                    const half2 hx01 = __floats2half2_rn(x[xb] * gv0, x[xb + 1] * gv1);
                    const half2 hx23 = __floats2half2_rn(x[xb + 2] * gv2, x[xb + 3] * gv3);
                    hacc[r][b] = __hfma2(hx01, w01, hacc[r][b]);
                    hacc[r][b] = __hfma2(hx23, w23, hacc[r][b]);
                }
            }
        }
    }

    // half2 → float（尾部标量并入用 float 累加）。
    float acc[4][BGRP];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int b = 0; b < BGRP; b++) {
            const float2 f = __half22float2(hacc[r][b]);
            acc[r][b] = f.x + f.y;
        }

    // 尾部标量兜底（fp16 路径 k 非 4 倍数时；int8 路径 k 恒为 4 倍数）。
    if (wtype == 0) {
        const int k4 = k & ~3;
        for (int kk = k4 + tid; kk < k; kk += blockDim.x) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = row0 + r;
                if (row >= m) continue;
                const float wv = __half2float(Af16[row * k + kk]);
                #pragma unroll
                for (int b = 0; b < BGRP; b++) {
                    if (b >= bcnt) break;
                    acc[r][b] += wv * x[(b0 + b) * k + kk];
                }
            }
        }
    }

    // warp shuffle 归约（4 行 × BGRP slot；只有 1 次 __syncthreads）。
    __shared__ float partial[4 /*warp*/][4 /*row*/][BGRP];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int b = 0; b < BGRP; b++) {
            float v = acc[r][b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) partial[warp][r][b] = v;
        }
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const int row = row0 + r;
            if (row >= m) continue;
            #pragma unroll
            for (int b = 0; b < BGRP; b++) {
                if (b >= bcnt) break;
                float sum = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; w++) sum += partial[w][r][b];
                const int yb = (b0 + b) * m + row;
                if (op == 0) y[yb] = relu2_mb(sum);
                else if (op == 3) y[yb] = sum;
                else y[yb] += sum;
            }
        }
    }
}
"#;

/// 稀疏 FFN value 投影内核：x += r2 @ ffn_value，r2 已 relu²（约 96% 稀疏）。
/// 对齐 Albatross `cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel` 的平铺布局与稀疏遍历：
/// 每 block 处理一个 f 片（TILE=128）和一个 c 片（C_TILE=256），只读取 r2 非零 f 对应的权重列，
/// 把 52MB 权重读取降到 ~2MB（带宽 ~17× 削减）。
/// value_tiled 布局：元素 (f,c) → [f_block][c_block][f_local][c_local]，
///   f_block=f/128, c_block=c/256, tile_base=((f_block*c_blocks+c_block)*128)*256。
/// dispatch (fh/128, c/256, 1)，每 block 128 线程。x 已含残差，跨 f_block 用原子累加。
const FFN_VALUE_SPARSE_SRC: &str = r#"
extern "C" __global__ void ffn_value_sparse_add(
    const float*    __restrict__ r2,          // [fh] relu² 输出
    const __half*   __restrict__ value_tiled, // [fh*C] 平铺布局
    float*          __restrict__ x,           // [C] 就地原子累加（已含残差）
    const int c,
    const int fh)
{
    constexpr int TILE    = 128;
    constexpr int C_TILE  = 256; // 2 * TILE
    __shared__ float r2_slice[TILE];
    __shared__ int   nnz_ids[TILE];
    __shared__ int   nnz_count;
    __shared__ int   warp_counts[TILE / 32];
    __shared__ int   warp_prefix[TILE / 32];

    const int f_block = blockIdx.x;
    const int c_block = blockIdx.y;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const int start_f = f_block * TILE;

    // 读 r2 片并统计非零（r2 已 relu²，非零即 r2v != 0）。
    float r2v = 0.f;
    bool  nonzero = false;
    int   local_pos = 0;
    if (tid < TILE) {
        r2v = r2[start_f + tid];
        r2_slice[tid] = r2v;
        nonzero = (r2v != 0.0f);
        unsigned mask = __ballot_sync(0xffffffffu, nonzero);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) warp_counts[warp] = __popc(mask);
    }
    __syncthreads();
    if (tid == 0) {
        int s = 0;
#pragma unroll
        for (int w = 0; w < TILE / 32; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();
    if (tid < TILE && nonzero) {
        nnz_ids[warp_prefix[warp] + local_pos] = tid;
    }
    __syncthreads();

    const int c_blocks = c / C_TILE;
    const int tile_base = ((f_block * c_blocks + c_block) * TILE) * C_TILE;
    const int c0 = c_block * C_TILE + tid * 2;
    float acc0 = 0.f, acc1 = 0.f;
    for (int i = 0; i < nnz_count; i += 2) {
        const int f0 = nnz_ids[i];
        const __half* w0 = value_tiled + (long long)tile_base + f0 * C_TILE + tid * 2;
        const float a0 = r2_slice[f0];
        acc0 += a0 * __half2float(w0[0]);
        acc1 += a0 * __half2float(w0[1]);
        if (i + 1 < nnz_count) {
            const int f1 = nnz_ids[i + 1];
            const __half* w1 = value_tiled + (long long)tile_base + f1 * C_TILE + tid * 2;
            const float a1 = r2_slice[f1];
            acc0 += a1 * __half2float(w1[0]);
            acc1 += a1 * __half2float(w1[1]);
        }
    }
    atomicAdd(x + c0, acc0);
    atomicAdd(x + c0 + 1, acc1);
}
"#;

/// argmax CUDA kernel：在 logits [N] 中找最大值索引，写入 token[0]（f32 位模式存 uint）。
/// 语义对齐 Vulkan `argmax.comp`：单 block（256 线程）协作扫描，shared 树归约取全局 argmax，
/// 平局取更小索引（与 CPU 严格大于 argmax 一致）。dispatch (1,1,1)。
const ARGMAX_SRC: &str = r#"
extern "C" __global__ void rwkv_argmax(
    const float* __restrict__ logits,   // [N]
    float* __restrict__ token,          // [1] 写入 argmax 索引的 f32 位模式
    const int n)
{
    const int tid = threadIdx.x;
    constexpr int BS = 256;
    __shared__ float s_max[BS];
    __shared__ int   s_idx[BS];

    // 每线程沿 stride 扫描，取局部最大（严格大于，平局取小索引）。
    float lm = -1e30f;
    int   li = 0;
    for (int i = tid; i < n; i += BS) {
        const float v = logits[i];
        if (v > lm) { lm = v; li = i; }
    }
    s_max[tid] = lm;
    s_idx[tid] = li;
    __syncthreads();

    // shared 树归约，平局取更小索引。
    for (int step = BS >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            const float av = s_max[tid];
            const int   ai = s_idx[tid];
            const float bv = s_max[tid + step];
            const int   bi = s_idx[tid + step];
            if (bv > av || (bv == av && bi < ai)) {
                s_max[tid] = bv;
                s_idx[tid] = bi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        token[0] = __int_as_float(s_idx[0]);
    }
}
"#;

/// 统一 sample CUDA kernel：penalty(repetition/frequency/presence) + temperature + top-k + top-p
/// 过滤后按概率采样，写入 token[0]（f32 位模式存 uint）。语义对齐 Vulkan `sample.comp`。
/// 单 block（256 线程）协作，dispatch (1,1,1)。采样参数从 sampler 缓冲读取（f32 位模式存 uint）：
///   sampler[0]=temperature  sampler[1]=top_k(uint)  sampler[2]=top_p  sampler[3]=seed(uint)
///   sampler[4]=repetition_penalty  sampler[5]=frequency_penalty  sampler[6]=presence_penalty
///   sampler[7]=hist_len(uint)
/// 流程：载入 logits → 惩罚（counter 直方图统计历史 token）→ temperature → top-k（迭代求第 k 大
/// 阈值）→ softmax（max 归约→exp→归一化）→ top-p（累积截断）→ splitmix 采样定位 token。
const SAMPLE_SRC: &str = r#"
__device__ __forceinline__ float u01(unsigned int s) {
    s += 0x9E3779B9u;
    unsigned int z = s;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    z ^= z >> 16;
    return (float)z / 4294967296.0f;
}

extern "C" __global__ void rwkv_sample(
    const float*      __restrict__ logits,   // [n]
    float*            __restrict__ token,    // [1] 写入索引的 f32 位模式
    float*            __restrict__ temp,     // [n] 工作区
    float*            __restrict__ mask,     // [n] 工作区
    unsigned int*     __restrict__ counter,  // [n] 直方图
    const float*      __restrict__ sampler,  // [8] 参数
    const unsigned int* __restrict__ hist,   // [hist_len] 历史 token
    const int n)
{
    const int tid = threadIdx.x;
    constexpr int BS = 112;
    constexpr int MAXK = 50;   // 快速路径支持的最大 top_k（覆盖常见 50）
    // 单遍 top-K 快速路径共享缓冲：每线程一个局部有序 top-K。
    __shared__ float s_val[BS][MAXK];
    __shared__ int   s_idx[BS][MAXK];
    // 全局 top-K 结果（s_topval/s_topidx 为最小堆，s_sorted/s_sortedidx 为降序结果）。
    __shared__ float s_topval[MAXK];
    __shared__ int   s_topidx[MAXK];
    __shared__ float s_sorted[MAXK];
    __shared__ int   s_sortedidx[MAXK];
    // 兜底路径（top_k 未设或 > MAXK）用的小块归约缓冲。
    __shared__ float s_fval[BS];
    __shared__ int   s_fidx[BS];
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_u;
    __shared__ float s_threshold;
    __shared__ float g_cutoff;

    const float temperature = sampler[0];
    const unsigned int top_k = __float_as_uint(sampler[1]);
    const float top_p = sampler[2];
    const unsigned int seed = __float_as_uint(sampler[3]);
    const float rep = sampler[4];
    const float freq = sampler[5];
    const float pres = sampler[6];
    const unsigned int hist_len = __float_as_uint(sampler[7]);
    const bool do_topk = (top_k > 0u && top_k < (unsigned int)n);
    const int K = do_topk ? (int)top_k : 0;

    // 1. 载入 logits
    for (int i = tid; i < n; i += BS) temp[i] = logits[i];

    // 2. 惩罚
    if (hist_len > 0u && (rep != 1.0f || freq != 0.0f || pres != 0.0f)) {
        for (int i = tid; i < n; i += BS) counter[i] = 0u;
        __syncthreads();
        for (int h = tid; h < (int)hist_len; h += BS) {
            atomicAdd(&counter[hist[h]], 1u);
        }
        __syncthreads();
        for (int i = tid; i < n; i += BS) {
            const unsigned int cnt = counter[i];
            float l = temp[i];
            if (cnt > 0u) {
                if (rep != 1.0f) l = l > 0.0f ? l / rep : l * rep;
                if (pres != 0.0f) l -= pres;
            }
            if (freq != 0.0f) l -= freq * (float)cnt;
            temp[i] = l;
        }
        __syncthreads();
    }

    // 3. temperature
    float invT = 1.0f / temperature;
    if (!(temperature > 0.0f)) invT = 1.0f;
    for (int i = tid; i < n; i += BS) temp[i] *= invT;
    __syncthreads();

    if (K > 0 && K <= MAXK) {
        // ================= 快速路径：单遍 top-K =================
        // 4. 每线程维护局部有序 top-K（降序），一次扫描完成。
        for (int j = 0; j < MAXK; j++) { s_val[tid][j] = -1e30f; s_idx[tid][j] = -1; }
        for (int i = tid; i < n; i += BS) {
            const float v = temp[i];
            if (v > s_val[tid][K - 1]) {
                int pos = K - 1;
                while (pos > 0 && v > s_val[tid][pos - 1]) {
                    s_val[tid][pos] = s_val[tid][pos - 1];
                    s_idx[tid][pos] = s_idx[tid][pos - 1];
                    --pos;
                }
                s_val[tid][pos] = v;
                s_idx[tid][pos] = i;
            }
        }
        __syncthreads();

        // 5. tid==0 用最小堆合并 BS*K 个候选 → 全局 top-K，再降序提取。
        if (tid == 0) {
            auto sift = [&](int i, int h) {
                while (true) {
                    int l = 2 * i + 1, r = 2 * i + 2, m = i;
                    if (l < h && s_topval[l] < s_topval[m]) m = l;
                    if (r < h && s_topval[r] < s_topval[m]) m = r;
                    if (m == i) break;
                    float tv = s_topval[i]; s_topval[i] = s_topval[m]; s_topval[m] = tv;
                    int ti = s_topidx[i]; s_topidx[i] = s_topidx[m]; s_topidx[m] = ti;
                    i = m;
                }
            };
            // 用第 0 行初始化最小堆
            for (int j = 0; j < K; j++) { s_topval[j] = s_val[0][j]; s_topidx[j] = s_idx[0][j]; }
            for (int j = K / 2 - 1; j >= 0; j--) sift(j, K);
            // 插入其余行候选
            for (int th = 1; th < BS; th++) {
                for (int j = 0; j < K; j++) {
                    const float v = s_val[th][j];
                    if (v <= -1e29f) break; // 空槽
                    if (v > s_topval[0]) {
                        s_topval[0] = v; s_topidx[0] = s_idx[th][j];
                        sift(0, K);
                    }
                }
            }
            // 降序提取到 s_sorted
            for (int r = K; r > 0; r--) {
                s_sorted[r - 1] = s_topval[0];
                s_sortedidx[r - 1] = s_topidx[0];
                s_topval[0] = s_topval[r - 1];
                s_topidx[0] = s_topidx[r - 1];
                sift(0, r - 1);
            }
            s_threshold = s_sorted[K - 1]; // 第 K 大（降序末位）= 保留边界
            // 历史注：曾误取 s_sorted[0]（最大值），top-K 名义保留 50 实际只留
            // top-1 + 并列——温度较高时采样分布被错误坍缩到单点。
        }
        __syncthreads();

        // 6. 低于阈值置 -inf（保留 top-K 及阈值并列项）
        for (int i = tid; i < n; i += BS) if (temp[i] < s_threshold) temp[i] = -1e30f;
        __syncthreads();
    } else {
        // ================= 兜底路径：top_k 未设或 > MAXK，保持原逻辑 =================
        if (do_topk) {
            for (int i = tid; i < n; i += BS) mask[i] = 0.0f;
            __syncthreads();
            if (tid == 0) s_threshold = -1e30f;
            __syncthreads();
            for (unsigned int round = 0u; round < top_k; round++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask[i] == 0.0f && temp[i] > lm) { lm = temp[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                for (int step = BS >> 1; step > 0; step >>= 1) {
                    if (tid < step) {
                        const float bv = s_fval[tid + step];
                        const int   bi = s_fidx[tid + step];
                        if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                            s_fval[tid] = bv; s_fidx[tid] = bi;
                        }
                    }
                    __syncthreads();
                }
                if (tid == 0) { s_threshold = s_fval[0]; mask[s_fidx[0]] = 1.0f; }
                __syncthreads();
            }
            for (int i = tid; i < n; i += BS) if (temp[i] < s_threshold) temp[i] = -1e30f;
            __syncthreads();
        }
    }

    // 7. softmax：max -> exp -> normalize
    // 归约为非幂 block 安全版（BS=112 非 2 的幂，纯树归约会孤儿化部分 warp 的
    // 结果——同 batch 版注释，max 漏读/sum 漏加 → softmax 全错）。
    {
        float lm = -1e30f;
        for (int i = tid; i < n; i += BS) lm = fmaxf(lm, temp[i]);
        s_fval[tid] = lm;
        __syncthreads();
        {
            constexpr int P2 = 64;  // BS=112 → 64 + 48
            if (tid >= P2 && tid < BS) s_fval[tid - P2] = fmaxf(s_fval[tid - P2], s_fval[tid]);
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] = fmaxf(s_fval[tid], s_fval[tid + step]);
                __syncthreads();
            }
        }
        const float m = s_fval[0];
        __syncthreads();
        float s = 0.0f;
        for (int i = tid; i < n; i += BS) {
            const float v = expf(temp[i] - m);
            temp[i] = v;
            s += v;
        }
        s_fval[tid] = s;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (total > 0.0f) {
            for (int i = tid; i < n; i += BS) temp[i] /= total;
        }
        __syncthreads();
    }

    // 8. top-p：从最大概率起累积达 top_p 后截断（在全局 top-K 降序列表上单遍完成）
    if (top_p > 0.0f && top_p < 1.0f) {
        if (K > 0 && K <= MAXK) {
            if (tid == 0) {
                float cum = 0.0f, cutoffv = -1e30f;
                for (int j = K - 1; j >= 0; j--) {
                    const int idx = s_sortedidx[j];
                    cum += temp[idx];
                    cutoffv = temp[idx];
                    if (cum >= top_p) break;
                }
                g_cutoff = cutoffv;
            }
        } else {
            // 兜底：迭代 reduce_max（仅在无 top-k 时用到，罕见）
            for (int i = tid; i < n; i += BS) mask[i] = 0.0f;
            __syncthreads();
            if (tid == 0) g_cutoff = 0.0f;
            __syncthreads();
            float cum = 0.0f;
            for (int cnt = 0; cnt < 512 && cum < top_p; cnt++) {
                float lm = -1e30f; int li = 0;
                for (int i = tid; i < n; i += BS) {
                    if (mask[i] == 0.0f && temp[i] > lm) { lm = temp[i]; li = i; }
                }
                s_fval[tid] = lm; s_fidx[tid] = li;
                __syncthreads();
                for (int step = BS >> 1; step > 0; step >>= 1) {
                    if (tid < step) {
                        const float bv = s_fval[tid + step];
                        const int   bi = s_fidx[tid + step];
                        if (bv > s_fval[tid] || (bv == s_fval[tid] && bi < s_fidx[tid])) {
                            s_fval[tid] = bv; s_fidx[tid] = bi;
                        }
                    }
                    __syncthreads();
                }
                if (tid == 0) {
                    mask[s_fidx[0]] = 1.0f;
                    cum += s_fval[0];
                    g_cutoff = s_fval[0];
                }
                __syncthreads();
            }
            __syncthreads();
        }
        for (int i = tid; i < n; i += BS) if (temp[i] < g_cutoff) temp[i] = 0.0f;
        __syncthreads();
    }

    // 9. 采样：在 top-K（降序）上做前缀和定位（兜底路径用全量扫描）。
    if (K > 0 && K <= MAXK) {
        if (tid == 0) {
            float total = 0.0f;
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp[idx] > 0.0f) total += temp[idx];
            }
            const float u = u01(seed) * total;
            float acc = 0.0f;
            int chosen = s_sortedidx[K - 1];
            for (int j = K - 1; j >= 0; j--) {
                const int idx = s_sortedidx[j];
                if (temp[idx] > 0.0f) {
                    acc += temp[idx];
                    if (acc > u) { chosen = idx; break; }
                }
            }
            token[0] = __int_as_float(chosen);
        }
    } else {
        float ts = 0.0f;
        for (int i = tid; i < n; i += BS) ts += temp[i];
        s_fval[tid] = ts;
        __syncthreads();
        {
            constexpr int P2 = 64;
            if (tid >= P2 && tid < BS) s_fval[tid - P2] += s_fval[tid];
            __syncthreads();
            for (int step = P2 >> 1; step > 0; step >>= 1) {
                if (tid < step) s_fval[tid] += s_fval[tid + step];
                __syncthreads();
            }
        }
        const float total = s_fval[0];
        __syncthreads();
        if (tid == 0) s_u = u01(seed) * total;
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                acc += temp[i];
                if (acc > s_u) { chosen = i; break; }
            }
            token[0] = __int_as_float(chosen);
        }
    }
}
"#;

/// record_token CUDA kernel：把 in_tok[0]（f32 位模式存 uint 的 token 索引）追加到
/// out_seq[atomicAdd(cnt)]，随后 cnt 自增。语义对齐 Vulkan `record_token.comp`。
/// 单线程（dispatch (1,1,1)），供 GPU self-loop 记录每轮生成的 token。
const RECORD_TOKEN_SRC: &str = r#"
extern "C" __global__ void rwkv_record_token(
    const unsigned int* __restrict__ in_tok,  // [1] token（f32 位模式）
    unsigned int* __restrict__ out_seq,       // [n] 序列缓冲
    unsigned int* __restrict__ cnt)           // [1] 计数器（原子自增）
{
    const unsigned int i = atomicAdd(&cnt[0], 1u);
    out_seq[i] = in_tok[0];
}
"#;

/// gather_row_device_f16 CUDA kernel：从 fp16 表 src[VOCAB, C] 按 token 索引读一行，
/// 转 fp32 写入 dst[C]。索引来自 tok[0]（f32 位模式存 uint）。
/// 语义对齐 Vulkan `gather_row_f16.comp`：dispatch (ceil(C/256), 1, 1)。
const GATHER_ROW_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_gather_row_f16(
    const unsigned int* __restrict__ in_tok,  // [1] token 索引（f32 位模式）
    const __half*  __restrict__ in_src,       // [VOCAB, C] fp16
    float* __restrict__ out_dst,              // [C] fp32
    const int c)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idx = in_tok[0];
    if (index < c) {
        out_dst[index] = __half2float(in_src[(size_t)idx * (size_t)c + (size_t)index]);
    }
}
"#;

/// copy_device_f16 CUDA kernel：f16 设备到设备全量拷贝（v_first 快照用）。
/// 语义对齐 Vulkan `copy_token.comp` 的设备侧拷贝分支；len 为元素数，一维拷贝。
const COPY_DEVICE_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_device_f16(
    const __half* __restrict__ src,  // [len]
    __half* __restrict__ dst,        // [len]
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[i] = src[i];
}
"#;

/// copy_device（f32）CUDA kernel：f32 设备到设备全量拷贝（v_first 快照 / 状态缓冲用）。
const COPY_DEVICE_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_device(
    const float* __restrict__ src,  // [len]
    float* __restrict__ dst,        // [len]
    const int len)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) dst[i] = src[i];
}
"#;

/// copy_token CUDA kernel：y[i] = x[token*stride + i]（sequence-parallel 状态更新用）。
/// 语义对齐 Vulkan `copy_token.comp`。
const COPY_TOKEN_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_token(
    const float* __restrict__ x,   // [T, C]
    float* __restrict__ y,         // [C]
    const int c,
    const int stride,              // token 行步长
    const int token)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < c) y[i] = x[(size_t)token * (size_t)stride + (size_t)i];
}
"#;

/// elementwise_sigmoid CUDA kernel：y = sigmoid(a) = 1/(1+exp(-a))。
/// 语义对齐 Vulkan `elementwise_f32_f32_sigmoid`（OP=1）；grid.y = batch。
const ELEMENTWISE_SIGMOID_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_sigmoid(
    const float* __restrict__ a,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const int b = blockIdx.y;
    const int base = b * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = 1.0f / (1.0f + __expf(-a[base + i]));
    }
}
"#;

/// elementwise_scale_exp CUDA kernel：y = exp(a * b[0])（b 为共享 f32 标量）。
/// 语义对齐 Vulkan `elementwise_f32_f32_scale_exp`（OP=9）；grid.y = batch。
const ELEMENTWISE_SCALE_EXP_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_scale_exp(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const float sc = b[0]; // 全局共享标量（与 Vulkan elementwise.comp OP9 一致）
    const int b0 = blockIdx.y;
    const int base = b0 * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = __expf(a[base + i] * sc);
    }
}
"#;

/// elementwise_mul CUDA kernel：y = a * b（逐元素，grid.y = batch）。
/// 语义对齐 Vulkan `elementwise_f32_f32_mul`（OP=5）。
const ELEMENTWISE_MUL_SRC: &str = r#"
extern "C" __global__ void rwkv_elementwise_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ y,
    const int c,
    const int batch)
{
    const int b0 = blockIdx.y;
    const int base = b0 * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        y[base + i] = a[base + i] * b[base + i];
    }
}
"#;

/// to_f16 CUDA kernel：f32 → f16（token 并行，sequence-parallel）。
/// 语义对齐 Vulkan `to_f16.comp`：非对齐 token（token>=T）写 0，供 GEMM 填充行输出为 0。
const TO_F16_SRC: &str = r#"
extern "C" __global__ void rwkv_to_f16(
    const float* __restrict__ x,   // [T, C] f32
    __half* __restrict__ y,        // [M_PAD, C] f16
    const int c,
    const int t,
    const int x_stride,
    const int y_stride)
{
    const int token = blockIdx.x;
    const size_t yb = (size_t)token * (size_t)y_stride;
    if (token >= t) {
        for (int i = threadIdx.x; i < c; i += blockDim.x) y[yb + i] = __float2half(0.0f);
        return;
    }
    const size_t xb = (size_t)token * (size_t)x_stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) y[yb + i] = __float2half(x[xb + i]);
}
"#;

/// to_f16_triple CUDA kernel：一次把 xr/xk/xv 三个 [T,C] f32 转成 [M_PAD,C] f16。
/// 语义对齐 Vulkan `to_f16_triple.comp`。
const TO_F16_TRIPLE_SRC: &str = r#"
extern "C" __global__ void rwkv_to_f16_triple(
    const float* __restrict__ xr,
    const float* __restrict__ xk,
    const float* __restrict__ xv,
    __half* __restrict__ yr,
    __half* __restrict__ yk,
    __half* __restrict__ yv,
    const int c,
    const int t,
    const int x_stride,
    const int y_stride)
{
    const int token = blockIdx.x;
    const size_t yb = (size_t)token * (size_t)y_stride;
    if (token >= t) {
        for (int i = threadIdx.x; i < c; i += blockDim.x) {
            yr[yb + i] = __float2half(0.0f);
            yk[yb + i] = __float2half(0.0f);
            yv[yb + i] = __float2half(0.0f);
        }
        return;
    }
    const size_t xb = (size_t)token * (size_t)x_stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        yr[yb + i] = __float2half(xr[xb + i]);
        yk[yb + i] = __float2half(xk[xb + i]);
        yv[yb + i] = __float2half(xv[xb + i]);
    }
}
"#;

/// dequant_int8_to_f16 CUDA kernel：int8 [M,K] 反量化为 fp16 [M,K]。
/// 语义对齐 Vulkan `dequant_int8_f16.comp`：idx uint32[M,K/4]（4 uint8/uint32）、sz uint32[M,K/128]。
const DEQUANT_INT8_SRC: &str = r#"
__device__ __forceinline__ void unpack_quant_sz(
    unsigned int sz, float& scale, float& zero)
{
    scale = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    zero  = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
}
extern "C" __global__ void rwkv_dequant_int8(
    const unsigned int* __restrict__ idx,  // [M, K/4]
    const unsigned int* __restrict__ sz,   // [M, K/128]
    __half* __restrict__ w,                // [M, K] f16 输出
    const int m,
    const int k)
{
    const int kv = k / 4;
    const int kg = k / 128;
    const long total = (long)m * kv;
    const long linear = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) return;
    const int mm = (int)(linear / kv);
    const int kk = (int)(linear % kv);
    const int g = kk / 32;
    float sc, zr;
    unpack_quant_sz(sz[(size_t)mm * kg + g], sc, zr);
    const unsigned int ipack = idx[linear];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const int byte = (int)((ipack >> (8 * j)) & 0xFFu);
        const float wv = sc * (float)byte + zr;
        w[(size_t)mm * k + kk * 4 + j] = __float2half(wv);
    }
}
"#;

/// fuse_ka CUDA kernel：k_mod = k*(1+k_a*(a-1))、kk_l2 = normalize(k*k_k)、b = -kk_l2*a。
/// 语义对齐 Vulkan `fuse_ka.comp`（EPSILON=1e-12）；grid = (H, batch)、block 256。
const FUSE_KA_SRC: &str = r#"
extern "C" __global__ void rwkv_fuse_ka(
    const float* __restrict__ k,    // [batch, H*N]
    const float* __restrict__ kk_w, // [H*N] 共享
    const float* __restrict__ a,    // [batch, H*N]
    const float* __restrict__ ka_w, // [H*N] 共享
    float* __restrict__ km,         // [batch, H*N]
    float* __restrict__ kl,         // [batch, H*N]
    float* __restrict__ b,          // [batch, H*N]
    const int h,
    const int n,
    const int batch)
{
    const int head = blockIdx.x;
    const int bidx = blockIdx.y;
    const int tid  = threadIdx.x;
    const int base   = bidx * (h * n) + head * n;
    const int wbase  = head * n;
    float local_sq = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        const float kkv = k[base + j] * kk_w[wbase + j];
        local_sq += kkv * kkv;
    }
    __shared__ float sdata[256];
    sdata[tid] = local_sq;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float inv_norm = 1.0f / fmaxf(sqrtf(sdata[0]), 1e-12f);
    for (int j = tid; j < n; j += blockDim.x) {
        const int addr  = base + j;
        const int waddr = wbase + j;
        const float kv_  = k[addr];
        const float kkv  = kv_ * kk_w[waddr];
        const float k_l2 = kkv * inv_norm;
        const float av   = a[addr];
        km[addr] = kv_ * (1.0f + ka_w[waddr] * (av - 1.0f));
        kl[addr] = k_l2;
        b[addr]  = -k_l2 * av;
    }
}
"#;

/// sum_rk_rk CUDA kernel：s = Σ_j r[j]*k_mod[j]*r_k[j]（head 归约），y[j] += s*v[j]。
/// 语义对齐 Vulkan `sum_rk_rk.comp`；grid = (H, batch)、block 256。
const SUM_RK_RK_SRC: &str = r#"
extern "C" __global__ void rwkv_sum_rk_rk(
    const float* __restrict__ r,    // [batch, H*N]
    const float* __restrict__ km,   // k_mod [batch, H*N]
    const float* __restrict__ rk,   // [H*N] 共享
    const float* __restrict__ v,    // [batch, H*N]
    float* __restrict__ y,          // [batch, H*N] 累加
    const int h,
    const int n,
    const int batch)
{
    const int head = blockIdx.x;
    const int bidx = blockIdx.y;
    const int tid  = threadIdx.x;
    const int base = bidx * (h * n) + head * n;
    float local = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        local += r[base + j] * km[base + j] * rk[head * n + j];
    }
    __shared__ float sdata[256];
    sdata[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float s = sdata[0];
    for (int j = tid; j < n; j += blockDim.x) {
        y[base + j] += s * v[base + j];
    }
}
"#;

/// seq_shift CUDA kernel：result[t] = x[t] + tm*(prev - x[t])，prev 为 x[t-1] 或 token-shift state（t=0）。
/// 语义对齐 Vulkan `seq_shift.comp`；grid = (T, 1)、block 256。
const SEQ_SHIFT_SRC: &str = r#"
extern "C" __global__ void rwkv_seq_shift(
    const float* __restrict__ x,   // [T, C]
    const float* __restrict__ s,   // token-shift state [C]
    const float* __restrict__ tm,  // [C]
    float* __restrict__ y,         // [T, C]
    const int c,
    const int t,
    const int stride_x,
    const int stride_y)
{
    const int token = blockIdx.x;
    if (token >= t) return;
    const size_t xbase = (size_t)token * stride_x;
    const size_t ybase = (size_t)token * stride_y;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float cur  = x[xbase + i];
        const float prev = (token == 0) ? s[i] : x[xbase - stride_x + i];
        const float tmv  = tm[i];
        y[ybase + i] = cur + tmv * (prev - cur);
    }
}
"#;

/// v_first_lerp CUDA kernel：v[t] = v[t] + gate[t]*(v_first[t] - v[t])。
/// 语义对齐 Vulkan `v_first_lerp.comp`；grid = (T, 1)、block 256。
const V_FIRST_LERP_SRC: &str = r#"
extern "C" __global__ void rwkv_v_first_lerp(
    float* __restrict__ v,          // [T, C] in/out
    const float* __restrict__ g,    // [T, C] gate
    const float* __restrict__ vf,   // [T, C] v_first
    const int c,
    const int t,
    const int stride)
{
    const int token = blockIdx.x;
    if (token >= t) return;
    const size_t base = (size_t)token * stride;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float vv = v[base + i];
        const float gv = g[base + i];
        const float fv = vf[base + i];
        v[base + i] = vv + gv * (fv - vv);
    }
}
"#;

/// seq_shift_batch CUDA kernel：batch prefill 的 token shift（slot 边界 t=0 读该 slot 的 state）。
/// x/y 为 [batch, T, C]（slot 主序），s 为 [batch, C]。dispatch (T, batch, 1)。
const SEQ_SHIFT_BATCH_SRC: &str = r#"
extern "C" __global__ void rwkv_seq_shift_batch(
    const float* __restrict__ x,   // [batch, T, C]
    const float* __restrict__ s,   // token-shift state [batch, C]
    const float* __restrict__ tm,  // [C]（共享）
    float* __restrict__ y,         // [batch, T, C]
    const int c,
    const int t,
    const int stride_x,
    const int stride_y)
{
    const int token = blockIdx.x;
    const int b      = blockIdx.y;
    if (token >= t) return;
    const size_t base = ((size_t)b * t + token) * stride_x;
    const size_t ybase = ((size_t)b * t + token) * stride_y;
    const float* sprev = (token == 0) ? (s + (size_t)b * c) : (x + base - stride_x);
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float cur = x[base + i];
        const float prev = sprev[i];
        const float tmv = tm[i];
        y[ybase + i] = cur + tmv * (prev - cur);
    }
}
"#;

/// copy_token_batch CUDA kernel：每 slot 把 x 的第 lens[b]-1 行拷到 state[b]。
/// x 为 [batch, T, C]，state 为 [batch, C]。dispatch (batch, 1, 1)，block 256。
const COPY_TOKEN_BATCH_SRC: &str = r#"
extern "C" __global__ void rwkv_copy_token_batch(
    const float* __restrict__ x,      // [batch, T, C]
    float* __restrict__ state,        // [batch, C]
    const int* __restrict__ lens,     // [batch]（实际 prompt 长度，>=1）
    const int c,
    const int t)
{
    const int b = blockIdx.x;
    const int last = lens[b] - 1;
    const float* src = x + ((size_t)b * t + last) * c;
    float* dst = state + (size_t)b * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        dst[i] = src[i];
    }
}
"#;

/// segmean CUDA kernel：分段均值——x [batch, T_pad, C] 每 slot 对前 lens[b] 行求均值
/// → out [batch, C]（pad 行不计入）。批量 mean-hidden 特征提取用（语义对齐单序列
/// forward_seq_mean_hidden 的 T 维均值，仅累加顺序不同）。
/// dispatch (batch, 1, 1)，block 256（跨 C stride，循环 token 累加）。
const SEGMEAN_SRC: &str = r#"
extern "C" __global__ void rwkv_segmean(
    const float* __restrict__ x,     // [batch, t_pad, c]
    float* __restrict__ out,         // [batch, c]
    const int* __restrict__ lens,    // [batch]（实际 token 数，>=1）
    const int c,
    const int t_pad)
{
    const int b = blockIdx.x;
    const int len = lens[b];
    const float* xb = x + (size_t)b * t_pad * c;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < len; t++) {
            acc += xb[(size_t)t * c + i];
        }
        out[(size_t)b * c + i] = acc / (float)len;
    }
}
"#;

/// dplr_seq_batch CUDA kernel：batch prefill 的 DPLR 状态更新。
/// s 为 [batch, H, N*N]（batch State 布局），r/w/k/v/a/b/y 为 [batch, T, C]，
/// lens[b] 截断实际长度（padding 段不进 state）。dispatch (ceil(H*N/8), batch, 1)。
const DPLR_SEQ_BATCH_SRC: &str = r#"
__device__ __forceinline__ float dplr_b_halfwarp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, 16);
    }
    return v;
}
extern "C" __global__ void rwkv_dplr_seq_batch(
    float* __restrict__ s,          // [batch, H, N*N] 状态（in/out）
    const float* __restrict__ r,    // [batch, T, C]
    const float* __restrict__ w,    // [batch, T, C]
    const float* __restrict__ k,    // [batch, T, C]
    const float* __restrict__ v,    // [batch, T, C]
    const float* __restrict__ a,    // [batch, T, C]
    const float* __restrict__ b,    // [batch, T, C]
    float* __restrict__ y,          // [batch, T, C] 输出
    const int* __restrict__ lens,   // [batch]
    const int h,
    const int n,
    const int t,
    const int c)
{
    const int bslot = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int half = lane >> 4;
    const int subl = lane & 15;
    const int row  = (int)(blockIdx.x * 8 + warp * 2 + half);
    if (row >= h * n) return;
    const int head = row / n;
    const int i    = row % n;
    const int j0 = subl, j1 = subl + 16, j2 = subl + 32, j3 = subl + 48;
    // slot 内偏移：state 段 + token 基址。
    const size_t s_base = (size_t)bslot * h * n * n + (size_t)head * n * n + (size_t)i * n;
    const size_t slot_tok = (size_t)bslot * t;   // 该 slot 在 [batch,T,C] 中的 token 基址
    const size_t tok_off = (size_t)head * n;      // 该 head 在 [C] 内的列偏移
    float s0 = s[s_base + j0];
    float s1 = s[s_base + j1];
    float s2 = s[s_base + j2];
    float s3 = s[s_base + j3];
    const int len = lens[bslot];
    for (int tt = 0; tt < len; tt++) {
        const size_t e = (slot_tok + tt) * c + tok_off;
        const float vv = v[e + i];
        const float a0 = a[e + j0], a1 = a[e + j1], a2 = a[e + j2], a3 = a[e + j3];
        float sa = s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        sa = dplr_b_halfwarp_sum_all_xor(sa);
        const float w0 = w[e + j0], w1 = w[e + j1], w2 = w[e + j2], w3 = w[e + j3];
        const float k0 = k[e + j0], k1 = k[e + j1], k2 = k[e + j2], k3 = k[e + j3];
        const float b0 = b[e + j0], b1 = b[e + j1], b2 = b[e + j2], b3 = b[e + j3];
        const float r0 = r[e + j0], r1 = r[e + j1], r2 = r[e + j2], r3 = r[e + j3];
        s0 = s0 * w0 + k0 * vv + sa * b0;
        s1 = s1 * w1 + k1 * vv + sa * b1;
        s2 = s2 * w2 + k2 * vv + sa * b2;
        s3 = s3 * w3 + k3 * vv + sa * b3;
        float yv = s0 * r0 + s1 * r1 + s2 * r2 + s3 * r3;
        yv = dplr_b_halfwarp_sum_all_xor(yv);
        if (subl == 0) y[e + i] = yv;
    }
    s[s_base + j0] = s0;
    s[s_base + j1] = s1;
    s[s_base + j2] = s2;
    s[s_base + j3] = s3;
}
"#;

/// dplr_seq CUDA kernel：sequence-parallel DPLR 状态更新（内部循环 T）。
/// 语义对齐 Vulkan `dplr_seq.comp`：每个 block 一个 head，64 线程（==N），S 行存寄存器跨 token 传递。
/// 要求 n <= 64（RWKV-7 恒 N=64）。
const DPLR_SEQ_SRC: &str = r#"
// 每个状态行（head, i）用一个 half-warp（16 线程，每线程管 4 个 j 列）并行处理，
// 整块 GPU 并行处理全部 h*n 个状态行（旧版仅 h*n 线程、仅 h 个 block，占用率极低）。
// 行间独立：sa[i] = sum_j a[j]*s[i][j] 由 half-warp 归约，随后逐列更新状态。
__device__ __forceinline__ float dplr_halfwarp_sum_all_xor(float v) {
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, 16);
    }
    return v;
}
extern "C" __global__ void rwkv_dplr_seq(
    float* __restrict__ s,          // [H, N*N] 状态（in/out）
    const float* __restrict__ r,    // [T, C]
    const float* __restrict__ w,    // [T, C]
    const float* __restrict__ k,    // [T, C]
    const float* __restrict__ v,    // [T, C]
    const float* __restrict__ a,    // [T, C]
    const float* __restrict__ b,    // [T, C]
    float* __restrict__ y,          // [T, C] 输出
    const int h,
    const int n,
    const int t,
    const int c)
{
    // 128 线程/block = 4 warp = 8 half-warp，每 half-warp 管一个状态行。
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;      // 0..3
    const int lane = tid & 31;
    const int half = lane >> 4;     // 0/1
    const int subl = lane & 15;     // 0..15
    const int row  = (int)(blockIdx.x * 8 + warp * 2 + half); // 全局状态行 index
    if (row >= h * n) return;
    const int head = row / n;
    const int i    = row % n;
    // 每线程管 4 列：j0, j0+16, j0+32, j0+48（n=64）。
    const int j0 = subl, j1 = subl + 16, j2 = subl + 32, j3 = subl + 48;
    const size_t s_base = (size_t)head * n * n + (size_t)i * n; // 该行状态首地址
    const size_t tok_off = (size_t)head * n;                    // 该 head 在 [C] 内的列偏移
    float s0 = s[s_base + j0];
    float s1 = s[s_base + j1];
    float s2 = s[s_base + j2];
    float s3 = s[s_base + j3];
    for (int tt = 0; tt < t; tt++) {
        const size_t e = tok_off + (size_t)tt * c;
        const float vv = v[e + i];
        // 本线程持有的 4 列 a/w/k/b/r
        const float a0 = a[e + j0], a1 = a[e + j1], a2 = a[e + j2], a3 = a[e + j3];
        // sa = sum_j a[j]*s[j]，先算本线程 4 列 partial 再 half-warp 归约
        float sa = s0 * a0 + s1 * a1 + s2 * a2 + s3 * a3;
        sa = dplr_halfwarp_sum_all_xor(sa);
        const float w0 = w[e + j0], w1 = w[e + j1], w2 = w[e + j2], w3 = w[e + j3];
        const float k0 = k[e + j0], k1 = k[e + j1], k2 = k[e + j2], k3 = k[e + j3];
        const float b0 = b[e + j0], b1 = b[e + j1], b2 = b[e + j2], b3 = b[e + j3];
        const float r0 = r[e + j0], r1 = r[e + j1], r2 = r[e + j2], r3 = r[e + j3];
        s0 = s0 * w0 + k0 * vv + sa * b0;
        s1 = s1 * w1 + k1 * vv + sa * b1;
        s2 = s2 * w2 + k2 * vv + sa * b2;
        s3 = s3 * w3 + k3 * vv + sa * b3;
        float yv = s0 * r0 + s1 * r1 + s2 * r2 + s3 * r3;
        yv = dplr_halfwarp_sum_all_xor(yv);
        if (subl == 0) y[e + i] = yv;
    }
    s[s_base + j0] = s0;
    s[s_base + j1] = s1;
    s[s_base + j2] = s2;
    s[s_base + j3] = s3;
}
"#;

/// gemm 统一 CUDA kernel：C[M,N] = A[M,K] @ B[N,K]^T（A/B fp16，C fp32）。
/// op: 0=plain, 1=+bias[n], 2=+x[M,N], 3=relu2, 4=tanh。每线程计算一个 C 元素。
/// 语义对齐 Vulkan `gemm*.comp`（tensor-core 版的结果）。
const GEMM_SRC: &str = r#"
__device__ __forceinline__ float gemm_dplr_relu2f(float x) { return x > 0.f ? x * x : 0.f; }
extern "C" __global__ void rwkv_gemm(
    const __half* __restrict__ a,    // [M, K] f16
    const __half* __restrict__ b,    // [N, K] f16
    const float* __restrict__ bias,  // [N]（op==1 用）
    const float* __restrict__ x,     // [M, N]（op==2 用）
    float* __restrict__ c,           // [M, N] f32
    const int m,
    const int n,
    const int k,
    const int op)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    const __half* arow = a + (size_t)row * k;
    const __half* bcol = b + (size_t)col * k;
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) {
        acc += __half2float(arow[kk]) * __half2float(bcol[kk]);
    }
    float v;
    if (op == 1) v = acc + bias[col];
    else if (op == 2) v = acc + x[(size_t)row * n + col];
    else if (op == 3) v = gemm_dplr_relu2f(acc);
    else if (op == 4) v = tanhf(acc);
    else v = acc;
    c[(size_t)row * n + col] = v;
}
"#;

/// cuBLAS GEMM 的 epilogue：对 C[m,n]（f32）就地补充 op==1 bias / op==2 加 x / op==3 relu2 / op==4 tanh。
const GEMM_EPILOGUE_SRC: &str = r#"
extern "C" __global__ void rwkv_gemm_epilogue(
    float* __restrict__ c,          // [m, n]
    const float* __restrict__ bias, // [n]（op==1 用）
    const float* __restrict__ x,    // [m, n]（op==2 用）
    const int m,
    const int n,
    const int op)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (linear >= total) return;
    const int col = linear % n;
    float v = c[linear];
    if (op == 1) v += bias[col];
    else if (op == 2) v += x[linear];
    else if (op == 3) v = v > 0.f ? v * v : 0.f;
    else if (op == 4) v = tanhf(v);
    c[linear] = v;
}
"#;

/// gemv_seq CUDA kernel：y[b, m] = Σ_k x[b*x_stride + k] * A[m*k + k]（A f32 权重，跨步批量）。
/// 语义对齐 Vulkan `gemv_f32_f32`（gemv_seq_impl）；grid = (m, batch)、每 block 计算一行。
const GEMV_SEQ_SRC: &str = r#"
extern "C" __global__ void rwkv_gemv_seq(
    const float* __restrict__ a,      // [m, k] f32
    const float* __restrict__ x,      // [batch, x_stride]
    float* __restrict__ y,            // [batch, y_stride]
    const int m,
    const int k,
    const int x_stride,
    const int y_stride,
    const int batch)
{
    const int row = blockIdx.x;
    const int bidx = blockIdx.y;
    if (row >= m || bidx >= batch) return;
    const size_t xb = (size_t)bidx * x_stride;
    const float* arow = a + (size_t)row * k;
    float acc = 0.0f;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        acc += arow[kk] * x[xb + kk];
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)bidx * y_stride + row] = sdata[0];
}
"#;

impl ComputeBackend for CudaBackend {
    fn create_tensor(&mut self, len: usize, dtype: TensorDtype) -> R<TensorId> {
        self.next_id += 1;
        let id = TensorId(self.next_id);
        let bytes = match dtype {
            TensorDtype::F32 | TensorDtype::U32 => len * 4,
            TensorDtype::F16 => len * 2,
        };
        let dptr = self.alloc(bytes)?;
        let t = match dtype {
            TensorDtype::F32 => CudaTensor::F32 { dptr, len },
            TensorDtype::F16 => CudaTensor::F16 { dptr, len },
            TensorDtype::U32 => CudaTensor::U32 { dptr, len },
        };
        self.tensors.insert(id, t);
        self.lens.insert(id, len);
        Ok(id)
    }

    fn upload(&self, t: TensorId, data: &[f32]) -> R<()> {
        match self.get(t, "upload")? {
            CudaTensor::F32 { dptr, len } => {
                // 对齐 Vulkan：允许部分上传（data.len() <= len），只写前置元素，余下不动。
                // 模型会用 padded 缓冲（如 seq x 为 m_pad*C）只上传实际 t*C，其余不读。
                if data.len() > len {
                    return Err(format!("upload: len mismatch ({} > {len})", data.len()).into());
                }
                let bytes = bytemuck::cast_slice::<f32, u8>(data);
                self.memcpy_htod_n(dptr, bytes, data.len())?;
                Ok(())
            }
            CudaTensor::F16 { dptr, len } => {
                if data.len() > len {
                    return Err(
                        format!("upload(f16): len mismatch ({} > {len})", data.len()).into(),
                    );
                }
                let f16s: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                let bytes = bytemuck::cast_slice::<f16, u8>(&f16s);
                self.memcpy_htod_n2(dptr, bytes, f16s.len())?;
                Ok(())
            }
            CudaTensor::U32 { .. } => Err("upload: u32 tensor requires upload_u32".into()),
        }
    }

    fn upload_part(&self, t: TensorId, offset: usize, data: &[f32]) -> R<()> {
        // 部分上传：只写 [offset, offset+len) 段（元素偏移），其余不动。
        // F32 直拷；F16 先转半精度再按 2 字节元素偏移上传（v_first 用）。
        let (dptr, bytes, off_bytes, src_buf): (u64, usize, usize, Vec<u8>) =
            match self.get(t, "upload_part")? {
                CudaTensor::F32 { dptr, len } => {
                    if offset + data.len() > len {
                        return Err(format!(
                            "upload_part: range {}..{} exceeds len {len}",
                            offset,
                            offset + data.len()
                        )
                        .into());
                    }
                    (
                        dptr,
                        data.len() * 4,
                        offset * 4,
                        bytemuck::cast_slice::<f32, u8>(data).to_vec(),
                    )
                }
                CudaTensor::F16 { dptr, len } => {
                    if offset + data.len() > len {
                        return Err(format!(
                            "upload_part(f16): range {}..{} exceeds len {len}",
                            offset,
                            offset + data.len()
                        )
                        .into());
                    }
                    let f16s: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                    (
                        dptr,
                        f16s.len() * 2,
                        offset * 2,
                        bytemuck::cast_slice::<f16, u8>(&f16s).to_vec(),
                    )
                }
                _ => return Err("upload_part: tensor must be f32 or f16".into()),
            };
        // 源数据走 pinned scratch（同 htod_pinned 语义），拷贝到张量偏移段。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_part)"
        );
        let scratch_off = PINNED_ROWS * PINNED_ROW_BYTES;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.as_ptr(),
                (self.pinned as *mut u8).add(scratch_off),
                bytes,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(scratch_off) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(
                dptr + off_bytes as u64,
                src as *const c_void,
                bytes,
                self.stream
            ),
            "cuMemcpyHtoDAsync(upload_part)"
        );
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(upload_part-wait)"
        );
        Ok(())
    }

    fn upload_u32(&self, t: TensorId, data: &[u32]) -> R<()> {
        match self.get(t, "upload_u32")? {
            CudaTensor::U32 { dptr, len } => {
                // 对齐 Vulkan：允许部分上传（data.len() <= len）。
                if data.len() > len {
                    return Err(format!("upload_u32: len mismatch ({} > {len})", data.len()).into());
                }
                let bytes = bytemuck::cast_slice::<u32, u8>(data);
                self.memcpy_htod_n(dptr, bytes, data.len())?;
                Ok(())
            }
            _ => Err("upload_u32: t must be u32".into()),
        }
    }

    fn download(&self, t: TensorId) -> R<Vec<f32>> {
        match self.get(t, "download")? {
            CudaTensor::F32 { dptr, len } => {
                let mut out = vec![0u8; len * 4];
                self.memcpy_dtoh(dptr, &mut out)?;
                Ok(bytemuck::cast_slice::<u8, f32>(&out).to_vec())
            }
            CudaTensor::F16 { dptr, len } => {
                let mut bytes = vec![0u8; len * 2];
                self.memcpy_dtoh(dptr, &mut bytes)?;
                let f16s: &[f16] = bytemuck::cast_slice(&bytes);
                Ok(f16s.iter().map(|&v| v.to_f32()).collect())
            }
            CudaTensor::U32 { .. } => Err("download: u32 tensor unsupported here".into()),
        }
    }

    fn download_u32(&self, t: TensorId) -> R<Vec<u32>> {
        let len = *self
            .lens
            .get(&t)
            .ok_or("download_u32: unknown tensor len")?;
        match self.get(t, "download_u32")? {
            CudaTensor::U32 { dptr, .. } => {
                let mut bytes = vec![0u8; len * 4];
                self.memcpy_dtoh(dptr, &mut bytes)?;
                Ok(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec())
            }
            _ => Err("download_u32: t must be u32".into()),
        }
    }

    fn begin_batch(&mut self) -> R<()> {
        // 剖析：record start 事件到 stream（测本批纯 GPU 执行时间）。
        // 捕获期间不允许 cuEventRecord（stream 处于 capture mode），跳过。
        if self.prof_gpu && !self.graph_capturing {
            cu_check!(
                (self.drv.cu_event_record)(self.prof_ev_start, self.stream),
                "cuEventRecord(start)"
            );
        }
        Ok(())
    }

    fn clear_kernel_prof(&mut self) {
        self.drv.clear_prof();
    }

    fn dump_kernel_prof(&mut self) {
        self.drv.dump_prof();
    }

    fn end_batch(&mut self) -> R<()> {
        if self.prof_kernel && !self.graph_capturing {
            self.drv.dump_prof();
        }
        if self.gemm_prof && !self.graph_capturing && !self.gemm_times.is_empty() {
            let mut total = 0.0f64;
            let mut rows: Vec<_> = self.gemm_times.iter().collect();
            rows.sort_by(|a, b| {
                b.1.1
                    .partial_cmp(&a.1.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for ((m, n, k, op), (cnt, ms)) in &rows {
                total += ms;
                log::info!(
                    "[PROF_GEMM] m={m:>5} n={n:>5} k={k:>5} op={op} cnt={cnt:>3} total={ms:>8.2}ms avg={:>7.3}ms",
                    ms / *cnt as f64
                );
            }
            log::info!("[PROF_GEMM] SUM {total:.2}ms");
            self.gemm_times.clear();
        }
        if self.prof_gpu && !self.graph_capturing {
            cu_check!(
                (self.drv.cu_event_record)(self.prof_ev_end, self.stream),
                "cuEventRecord(end)"
            );
            unsafe {
                (self.drv.cu_event_synchronize)(self.prof_ev_end);
            }
            let mut ms: f32 = 0.0;
            unsafe {
                (self.drv.cu_event_elapsed_time)(&mut ms, self.prof_ev_start, self.prof_ev_end);
            }
            log::info!("[CUDA_GPU] batch: {ms:.3} ms");
        }
        Ok(())
    }

    fn begin_graph_capture(&mut self) -> R<()> {
        // 先确保 stream 空闲（无排队 kernel），再开始捕获。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(before capture)"
        );
        cu_check!(
            (self.drv.cu_graph_begin_capture)(self.stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
            "cuStreamBeginCapture"
        );
        self.graph_capturing = true;
        Ok(())
    }

    fn end_graph_capture(&mut self) -> R<()> {
        let mut graph: CuGraph = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_end_capture)(self.stream, &mut graph),
            "cuStreamEndCapture"
        );
        self.graph_capturing = false;
        // 实例化可执行 graph（flags=0），随后销毁源 graph。
        let mut exec: CuGraphExec = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_instantiate)(&mut exec, graph, 0),
            "cuGraphInstantiate"
        );
        cu_check!((self.drv.cu_graph_destroy)(graph), "cuGraphDestroy");
        // 替换旧的 exec graph（若存在）。
        if let Some(old) = self.exec_graph.replace(exec) {
            cu_check!((self.drv.cu_graph_exec_destroy)(old), "cuGraphExecDestroy");
        }
        Ok(())
    }

    fn graph_replay(&mut self) -> R<()> {
        let exec = self.exec_graph.ok_or("graph_replay: no captured graph")?;
        cu_check!(
            (self.drv.cu_graph_launch)(exec, self.stream),
            "cuGraphLaunch"
        );
        Ok(())
    }

    fn supports_graph_capture(&self) -> bool {
        true
    }

    fn prefill_graph_valid(&mut self, t: usize) -> R<bool> {
        Ok(self.prefill_graphs.contains_key(&t))
    }

    fn begin_prefill_capture(&mut self, t: usize) -> R<()> {
        // 确保 stream 空闲（无排队 kernel），再开始捕获。
        cu_check!(
            (self.drv.cu_stream_synchronize)(self.stream),
            "cuStreamSynchronize(before prefill capture)"
        );
        cu_check!(
            (self.drv.cu_graph_begin_capture)(self.stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
            "cuGraphBeginCapture(prefill)"
        );
        self.prefill_t = t;
        self.graph_capturing = true;
        Ok(())
    }

    fn end_prefill_capture(&mut self) -> R<()> {
        let mut graph: CuGraph = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_end_capture)(self.stream, &mut graph),
            "cuGraphEndCapture(prefill)"
        );
        self.graph_capturing = false;
        let mut exec: CuGraphExec = std::ptr::null_mut();
        cu_check!(
            (self.drv.cu_graph_instantiate)(&mut exec, graph, 0),
            "cuGraphInstantiate(prefill)"
        );
        cu_check!(
            (self.drv.cu_graph_destroy)(graph),
            "cuGraphDestroy(prefill)"
        );
        // 绑定到当前 T 的 prefill graph；若同 T 已存在则销毁旧的。
        if let Some(old) = self.prefill_graphs.insert(self.prefill_t, exec) {
            cu_check!(
                (self.drv.cu_graph_exec_destroy)(old),
                "cuGraphExecDestroy(old prefill)"
            );
        }
        Ok(())
    }

    fn prefill_graph_replay(&mut self) -> R<()> {
        let exec = self
            .prefill_graphs
            .get(&self.prefill_t)
            .copied()
            .ok_or("prefill_graph_replay: no captured graph for this T")?;
        cu_check!(
            (self.drv.cu_graph_launch)(exec, self.stream),
            "cuGraphLaunch(prefill)"
        );
        Ok(())
    }

    fn store_token_host(&self, tok: TensorId, token: u32) -> R<()> {
        // token 张量为 F32（f32 位模式存 uint 索引），直接上传单个元素。
        self.upload(tok, &[f32::from_bits(token)])
    }

    fn store_sampler_async(
        &self,
        sampler: TensorId,
        row: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        if row >= self.pinned_rows {
            return Err(format!(
                "store_sampler_async: row {row} >= pinned rows {}",
                self.pinned_rows
            )
            .into());
        }
        let dptr = match self.get(sampler, "store_sampler_async")? {
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("store_sampler_async: sampler must be f32".into()),
        };
        let data = [
            temperature,
            f32::from_bits(top_k),
            top_p,
            f32::from_bits(seed),
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            f32::from_bits(hist_len),
        ];
        // 写 pinned 行 + 流序异步拷贝（零 host 同步；拷贝在 stream 中排在
        // 此前已提交的 kernel 之后，供下一轮 graph replay 读取）。
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                (self.pinned as *mut u8).add(row * PINNED_ROW_BYTES),
                PINNED_ROW_BYTES,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(row * PINNED_ROW_BYTES) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(
                dptr,
                src as *const c_void,
                PINNED_ROW_BYTES,
                self.stream
            ),
            "cuMemcpyHtoDAsync(sampler)"
        );
        Ok(())
    }

    fn sampler_async_rows(&self) -> usize {
        self.pinned_rows
    }

    fn import_tensors_from(&mut self, src: &dyn ComputeBackend) -> R<()> {
        let src = src
            .as_any()
            .downcast_ref::<CudaBackend>()
            .ok_or("import_tensors_from: src is not a CudaBackend")?;
        if src.device != self.device {
            return Err("import_tensors_from: device mismatch".into());
        }
        // 全量复制张量表（同 TensorId → 同设备指针；同一 primary ctx 下有效）。
        for (id, ct) in &src.tensors {
            if self.tensors.insert(*id, ct.clone()).is_some() {
                return Err(format!("import_tensors_from: tensor id {id:?} collision").into());
            }
            let len = *src
                .lens
                .get(id)
                .ok_or("import_tensors_from: src lens missing")?;
            self.lens.insert(*id, len);
            self.foreign.insert(*id);
        }
        // 后续新张量 id 接续源后端计数（新实例自建工作缓冲/状态不与共享权重冲突）。
        self.next_id = self.next_id.max(src.next_id);
        // kernel 缓存同样共享：CuModule/CuFunction 句柄在同一 primary ctx 下跨
        // 实例有效（省每实例 ~30s 的 nvrtc 编译）。
        for (k, v) in &src.kernels {
            self.kernels.insert(k.clone(), *v);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // ==== batch 并发算子（单实例多序列：B slot 共享权重，一次读权重算 B 份）====

    fn gather_rows_device_f16(
        &mut self,
        s: TensorId,
        d: TensorId,
        t: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        // in_src 为 fp16 表 [VOCAB, C]；out_dst 为 f32 [batch, C]；in_tok 为 [batch]（f32 位模式存 uint）。
        let src_d = self.f16_ptr(s, "gather_rows_device_f16")?;
        let dst_d = self.f32_ptr(d, "gather_rows_device_f16")?;
        let tok_d = match self.get(t, "gather_rows_device_f16")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("gather_rows_device_f16: t must be u32 or f32".into()),
        };
        let func = self.kernel(
            "gather_rows_f16",
            GATHER_ROWS_F16_SRC,
            "rwkv_gather_rows_f16",
        )?;
        let grid = ((c as u32).div_ceil(256), batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &tok_d as *const u64 as *mut c_void,
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_lerp6_batch(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        xr: TensorId,
        xw: TensorId,
        xk: TensorId,
        xv: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ow: TensorId,
        ok: TensorId,
        ov: TensorId,
        oa: TensorId,
        og: TensorId,
        c: usize,
        eps: f32,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, xr) = (
            f32(x, "norm_lerp6_batch")?,
            f32(s, "norm_lerp6_batch")?,
            f32(g, "norm_lerp6_batch")?,
            f32(b, "norm_lerp6_batch")?,
            f32(xr, "norm_lerp6_batch")?,
        );
        let (xw, xk, xv, xa, xg) = (
            f32(xw, "norm_lerp6_batch")?,
            f32(xk, "norm_lerp6_batch")?,
            f32(xv, "norm_lerp6_batch")?,
            f32(xa, "norm_lerp6_batch")?,
            f32(xg, "norm_lerp6_batch")?,
        );
        let (or_, ow, ok) = (
            f32(or, "norm_lerp6_batch")?,
            f32(ow, "norm_lerp6_batch")?,
            f32(ok, "norm_lerp6_batch")?,
        );
        let (ov, oa, og) = (
            f32(ov, "norm_lerp6_batch")?,
            f32(oa, "norm_lerp6_batch")?,
            f32(og, "norm_lerp6_batch")?,
        );
        let func = self.kernel("norm_lerp6_batch", NORM_LERP6_BATCH_SRC, "norm_lerp6_batch")?;
        let grid = (c.div_ceil(256).max(1) as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &xr as *const u64 as *mut c_void,
            &xw as *const u64 as *mut c_void,
            &xk as *const u64 as *mut c_void,
            &xv as *const u64 as *mut c_void,
            &xa as *const u64 as *mut c_void,
            &xg as *const u64 as *mut c_void,
            &or_ as *const u64 as *mut c_void,
            &ow as *const u64 as *mut c_void,
            &ok as *const u64 as *mut c_void,
            &ov as *const u64 as *mut c_void,
            &oa as *const u64 as *mut c_void,
            &og as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn cmix_norm_lerp_batch(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, coeff, out_xb) = (
            f32(x, "cmix_norm_lerp_batch")?,
            f32(s, "cmix_norm_lerp_batch")?,
            f32(g, "cmix_norm_lerp_batch")?,
            f32(b, "cmix_norm_lerp_batch")?,
            f32(coeff, "cmix_norm_lerp_batch")?,
            f32(out_xb, "cmix_norm_lerp_batch")?,
        );
        let func = self.kernel(
            "cmix_norm_lerp_batch",
            CMIX_NORM_LERP_BATCH_SRC,
            "cmix_norm_lerp_batch",
        )?;
        let grid = (c.div_ceil(256).max(1) as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &coeff as *const u64 as *mut c_void,
            &out_xb as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    /// batch 版：kernel 与单序列版同一份（内部已带 batch=blockIdx.y 维），仅 grid.y=batch。
    #[allow(clippy::too_many_arguments)]
    fn fuse_ka_dplr_norm_batch(
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
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let sd = f32(s, "fuse_ka_dplr_norm_batch")?;
        let kd = f32(k, "fuse_ka_dplr_norm_batch")?;
        let kkd = f32(k_k, "fuse_ka_dplr_norm_batch")?;
        let ad = f16(a, "fuse_ka_dplr_norm_batch")?;
        let kad = f32(k_a, "fuse_ka_dplr_norm_batch")?;
        let rd = f32(r, "fuse_ka_dplr_norm_batch")?;
        let vd = f16(v, "fuse_ka_dplr_norm_batch")?;
        let wd = f16(w, "fuse_ka_dplr_norm_batch")?;
        let gd = f32(gamma, "fuse_ka_dplr_norm_batch")?;
        let bd = f32(beta, "fuse_ka_dplr_norm_batch")?;
        let rkd = f32(r_k, "fuse_ka_dplr_norm_batch")?;
        let kmd = f32(k_mod, "fuse_ka_dplr_norm_batch")?;
        let yd = f32(y, "fuse_ka_dplr_norm_batch")?;
        let ynd = f32(y_norm, "fuse_ka_dplr_norm_batch")?;

        let func = self.kernel(
            "fuse_ka_dplr_norm",
            FUSE_KA_DPLR_NORM_SRC,
            "fuse_ka_dplr_norm",
        )?;
        // 每个 block 处理一个 (head, slot)；kernel 内 batch=blockIdx.y。
        let grid = (h as u32, batch as u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let h_i = h as i32;
        let n_i = n as i32;
        let params = [
            &sd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &kkd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &kad as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &rkd as *const u64 as *mut c_void,
            &kmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ynd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &gn_eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_int8_rkv_stage1_batch(
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
        batch: usize,
    ) -> R<()> {
        let ridx = self.u32_ptr(r.idx, "gemv_int8_rkv_stage1_batch")?;
        let rsz = self.u32_ptr(r.sz, "gemv_int8_rkv_stage1_batch")?;
        let kidx = self.u32_ptr(k.idx, "gemv_int8_rkv_stage1_batch")?;
        let ksz = self.u32_ptr(k.sz, "gemv_int8_rkv_stage1_batch")?;
        let vidx = self.u32_ptr(v.idx, "gemv_int8_rkv_stage1_batch")?;
        let vsz = self.u32_ptr(v.sz, "gemv_int8_rkv_stage1_batch")?;
        let v1d = self.f32_ptr(v1, "gemv_int8_rkv_stage1_batch")?;
        let w1d = self.f32_ptr(w1, "gemv_int8_rkv_stage1_batch")?;
        let a1d = self.f32_ptr(a1, "gemv_int8_rkv_stage1_batch")?;
        let g1d = self.f32_ptr(g1, "gemv_int8_rkv_stage1_batch")?;
        let xrd = self.f32_ptr(xr, "gemv_int8_rkv_stage1_batch")?;
        let xkd = self.f32_ptr(xk, "gemv_int8_rkv_stage1_batch")?;
        let xvd = self.f32_ptr(xv, "gemv_int8_rkv_stage1_batch")?;
        let xwd = self.f32_ptr(xw, "gemv_int8_rkv_stage1_batch")?;
        let xad = self.f32_ptr(xa, "gemv_int8_rkv_stage1_batch")?;
        let xgd = self.f32_ptr(xg, "gemv_int8_rkv_stage1_batch")?;
        let ord = self.f32_ptr(out_r, "gemv_int8_rkv_stage1_batch")?;
        let okd = self.f32_ptr(out_k, "gemv_int8_rkv_stage1_batch")?;
        let ovd = self.f16_ptr(out_v, "gemv_int8_rkv_stage1_batch")?;
        let ovmd = self.f32_ptr(out_vm, "gemv_int8_rkv_stage1_batch")?;
        let owmd = self.f32_ptr(out_wm, "gemv_int8_rkv_stage1_batch")?;
        let oamd = self.f32_ptr(out_am, "gemv_int8_rkv_stage1_batch")?;
        let ogmd = self.f32_ptr(out_gm, "gemv_int8_rkv_stage1_batch")?;

        let func = self.kernel(
            "gemv_int8_rkv_stage1_batch",
            GEMV_INT8_RKV_STAGE1_BATCH_SRC,
            "gemv_int8_rkv_stage1_batch",
        )?;
        // grid.y = slot 分组数（kernel 内复用权重给 BGRP=2 个 slot，寄存器约束）。
        const RKV_MB_BGRP: usize = 2;
        let grid = (
            (c / 4 + vm + wm + am + gm) as u32,
            batch.div_ceil(RKV_MB_BGRP) as u32,
            1u32,
        );
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let batch_i = batch as i32;
        let params = [
            &ridx as *const u64 as *mut c_void,
            &rsz as *const u64 as *mut c_void,
            &kidx as *const u64 as *mut c_void,
            &ksz as *const u64 as *mut c_void,
            &vidx as *const u64 as *mut c_void,
            &vsz as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_lowrank_chain4_batch(
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
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let w2d = f32(w2, "gemv_lowrank_chain4_batch")?;
        let a2d = f32(a2, "gemv_lowrank_chain4_batch")?;
        let v2d = f32(v2, "gemv_lowrank_chain4_batch")?;
        let g2d = f32(g2, "gemv_lowrank_chain4_batch")?;
        let wmd = f32(w_mid, "gemv_lowrank_chain4_batch")?;
        let amd = f32(a_mid, "gemv_lowrank_chain4_batch")?;
        let vmd = f32(v_mid, "gemv_lowrank_chain4_batch")?;
        let gmd = f32(g_mid, "gemv_lowrank_chain4_batch")?;
        let w0d = f32(w0, "gemv_lowrank_chain4_batch")?;
        let a0d = f32(a0, "gemv_lowrank_chain4_batch")?;
        let v0d = f32(v0, "gemv_lowrank_chain4_batch")?;
        let scaled = f32(scale, "gemv_lowrank_chain4_batch")?;
        let vfd = f16(v_first, "gemv_lowrank_chain4_batch")?;
        let owd = f16(out_w, "gemv_lowrank_chain4_batch")?;
        let oad = f16(out_a, "gemv_lowrank_chain4_batch")?;
        let ovd = f16(out_v, "gemv_lowrank_chain4_batch")?;
        let ogd = f16(out_g, "gemv_lowrank_chain4_batch")?;

        let func = self.kernel(
            "gemv_lowrank_chain4_batch",
            GEMV_LOWRANK_CHAIN4_BATCH_SRC,
            "gemv_lowrank_chain4_batch",
        )?;
        // warp-per-row：grid.x = ceil(M/8)（每 block 8 warp 各 1 行），grid.y = slot 分组。
        const CHAIN4_WARPS: usize = 8;
        const CHAIN4_BGRP: usize = 4;
        let grid = (
            m.div_ceil(CHAIN4_WARPS) as u32,
            batch.div_ceil(CHAIN4_BGRP) as u32,
            1u32,
        );
        let block = (256u32, 1u32, 1u32);
        let m_i = m as i32;
        let kw_i = kw as i32;
        let ka_i = ka as i32;
        let kv_i = kv as i32;
        let kg_i = kg as i32;
        let batch_i = batch as i32;
        let params = [
            &w2d as *const u64 as *mut c_void,
            &a2d as *const u64 as *mut c_void,
            &v2d as *const u64 as *mut c_void,
            &g2d as *const u64 as *mut c_void,
            &wmd as *const u64 as *mut c_void,
            &amd as *const u64 as *mut c_void,
            &vmd as *const u64 as *mut c_void,
            &gmd as *const u64 as *mut c_void,
            &w0d as *const u64 as *mut c_void,
            &a0d as *const u64 as *mut c_void,
            &v0d as *const u64 as *mut c_void,
            &scaled as *const u64 as *mut c_void,
            &vfd as *const u64 as *mut c_void,
            &owd as *const u64 as *mut c_void,
            &oad as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ogd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &kw_i as *const i32 as *mut c_void,
            &ka_i as *const i32 as *mut c_void,
            &kv_i as *const i32 as *mut c_void,
            &kg_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn ffn_value_sparse_add_batch(
        &mut self,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
        batch: usize,
    ) -> R<()> {
        let vt = self.f16_ptr(value_tiled, "ffn_value_sparse_add_batch")?;
        let rd = self.f32_ptr(r2, "ffn_value_sparse_add_batch")?;
        let xd = self.f32_ptr(x, "ffn_value_sparse_add_batch")?;
        let func = self.kernel(
            "ffn_value_sparse_add_batch",
            FFN_VALUE_SPARSE_BATCH_SRC,
            "ffn_value_sparse_add_batch",
        )?;
        let grid = ((fh / 128) as u32, (c / 256) as u32, batch as u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let fh_i = fh as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &vt as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &fh_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_into_host_seeded_batch(
        &mut self,
        logits: TensorId,
        token: TensorId,
        n: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        hist: TensorId,
        batch: usize,
    ) -> R<()> {
        let logits_d = self.f32_ptr(logits, "sample_into_host_seeded_batch")?;
        let token_d = self.f32_ptr(token, "sample_into_host_seeded_batch")?;
        let temp_d = self.f32_ptr(temp, "sample_into_host_seeded_batch")?;
        let mask_d = self.f32_ptr(mask, "sample_into_host_seeded_batch")?;
        let counter_d = self.u32_ptr(counter, "sample_into_host_seeded_batch")?;
        let sampler_d = self.f32_ptr(sampler, "sample_into_host_seeded_batch")?;
        let hist_d = match self.get(hist, "sample_into_host_seeded_batch")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("sample_into_host_seeded_batch: hist must be u32 or f32".into()),
        };
        let func = self.kernel("rwkv_sample_batch", SAMPLE_BATCH_SRC, "rwkv_sample_batch")?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (112u32, 1u32, 1u32);
        let n_i = n as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &temp_d as *const u64 as *mut c_void,
            &mask_d as *const u64 as *mut c_void,
            &counter_d as *const u64 as *mut c_void,
            &sampler_d as *const u64 as *mut c_void,
            &hist_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    fn record_tokens(
        &mut self,
        in_tok: TensorId,
        out_seq: TensorId,
        cnt: TensorId,
        stride: usize,
        batch: usize,
    ) -> R<()> {
        let ptr = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32 or f32").into()),
            }
        };
        let in_tok_d = ptr(in_tok, "record_tokens")?;
        let out_d = ptr(out_seq, "record_tokens")?;
        let cnt_d = ptr(cnt, "record_tokens")?;
        let func = self.kernel(
            "rwkv_record_tokens",
            RECORD_TOKENS_SRC,
            "rwkv_record_tokens",
        )?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (1u32, 1u32, 1u32);
        let stride_i = stride as i32;
        let params = [
            &in_tok_d as *const u64 as *mut c_void,
            &out_d as *const u64 as *mut c_void,
            &cnt_d as *const u64 as *mut c_void,
            &stride_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    /// batch 版异步 sampler 上传：宽行（batch*8 f32 = batch*32 字节）。
    /// pinned 区按宽行切分：可用轮数 = PINNED_ROWS / batch（行宽随 batch 增大）。
    #[allow(clippy::too_many_arguments)]
    fn store_sampler_async_batch(
        &self,
        sampler: TensorId,
        row: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seeds: &[u32],
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        hist_len: u32,
    ) -> R<()> {
        let batch = seeds.len();
        let row_bytes = batch * PINNED_ROW_BYTES;
        let max_rows = self.pinned_rows / batch.max(1);
        if row >= max_rows {
            return Err(format!(
                "store_sampler_async_batch: row {row} >= max rows {max_rows} (batch {batch})"
            )
            .into());
        }
        let dptr = match self.get(sampler, "store_sampler_async_batch")? {
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("store_sampler_async_batch: sampler must be f32".into()),
        };
        // 每 slot 8 个 f32：temperature/top_k/top_p/seed/rep/freq/pres/hist_len。
        let mut data = Vec::with_capacity(batch * 8);
        for &seed in seeds {
            data.extend_from_slice(&[
                temperature,
                f32::from_bits(top_k),
                top_p,
                f32::from_bits(seed),
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                f32::from_bits(hist_len),
            ]);
        }
        // 写 pinned 宽行 + 流序异步拷贝（零 host 同步）。
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                (self.pinned as *mut u8).add(row * row_bytes),
                row_bytes,
            );
        }
        let src = unsafe { (self.pinned as *const u8).add(row * row_bytes) };
        cu_check!(
            (self.drv.cu_memcpy_htod_async)(dptr, src as *const c_void, row_bytes, self.stream),
            "cuMemcpyHtoDAsync(sampler batch)"
        );
        Ok(())
    }

    fn gather_row_device_f16(&mut self, s: TensorId, d: TensorId, t: TensorId, c: usize) -> R<()> {
        // in_src 为 fp16 表 [VOCAB, C]；out_dst 为 f32 [C]；in_tok 为 f32 位模式存 uint 索引。
        let src_d = self.f16_ptr(s, "gather_row_device_f16")?;
        let dst_d = self.f32_ptr(d, "gather_row_device_f16")?;
        // tok 为 F32（current_token 用 f32 位模式存 uint）或 U32，位模式相同，取 dptr。
        let tok_d = match self.get(t, "gather_row_device_f16")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("gather_row_device_f16: t must be u32 or f32".into()),
        };
        let func = self.kernel("gather_row_f16", GATHER_ROW_F16_SRC, "rwkv_gather_row_f16")?;
        let grid = ((c as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &tok_d as *const u64 as *mut c_void,
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_device_f16(&mut self, s: TensorId, d: TensorId) -> R<()> {
        let src_d = self.f16_ptr(s, "copy_device_f16")?;
        let dst_d = self.f16_ptr(d, "copy_device_f16")?;
        let len = *self
            .lens
            .get(&s)
            .ok_or("copy_device_f16: unknown src len")?;
        let func = self.kernel(
            "copy_device_f16",
            COPY_DEVICE_F16_SRC,
            "rwkv_copy_device_f16",
        )?;
        let grid = ((len as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let len_i = len as i32;
        let params = [
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &len_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        // 与 kernel 每 block 4 行对齐：M 需为 4 的倍数（GEMV_ROWS）。
        if !m.is_multiple_of(4) {
            return Err(format!("gemv_f16: M={m} must be divisible by 4 (GEMV_ROWS)").into());
        }
        // w 为 fp16 权重 (M,K)；x 为 f32 (K·batch)；y 为 f32 (M·batch)。
        let (a, x, y) = match self.get(w, "gemv_f16")? {
            CudaTensor::F16 { dptr, .. } => {
                let x = match self.get(x, "gemv_f16")? {
                    CudaTensor::F32 { dptr, .. } => dptr,
                    _ => return Err("gemv_f16: x must be f32 tensor".into()),
                };
                let y = match self.get(y, "gemv_f16")? {
                    CudaTensor::F32 { dptr, .. } => dptr,
                    _ => return Err("gemv_f16: y must be f32 tensor".into()),
                };
                (dptr, x, y)
            }
            _ => return Err("gemv_f16: w must be f16 tensor".into()),
        };
        // 编译并缓存 kernel。
        let func = self.kernel("gemv_f16", GEMV_F16_SRC, "gemv_f16")?;
        // grid.x = M/4，grid.y = batch；block = 128 线程。
        let grid = ((m / 4) as u32, n as u32, 1);
        let block = (128u32, 1u32, 1u32);
        // 按值传参（需取地址，指针参数为 u64 设备地址）。
        let m_i = m as i32;
        let k_i = k as i32;
        let batch_i = n as i32;
        let params = [
            &a as *const u64 as *mut c_void,
            &x as *const u64 as *mut c_void,
            &y as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &batch_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn norm(
        &mut self,
        x: TensorId,
        g: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        h: usize,
        eps: f32,
        rows: usize,
    ) -> R<()> {
        // 全部为 f32 张量，取 device 指针。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (xd, gd, bd, yd) = (
            f32(x, "norm")?,
            f32(g, "norm")?,
            f32(b, "norm")?,
            f32(y, "norm")?,
        );
        let func = self.kernel("norm", NORM_SRC, "rwkv_norm")?;
        // 每个 block 归一化一个 (head,batch) 行：grid.x = rows*h（对齐 Vulkan 的 (h, batch) 网格）。
        // h=1 时退化为 rows（layer norm）；h>1 时覆盖全部 head（group norm），
        // 否则非首 head 输出不被写入，残留脏数据导致跨 run 非确定。
        let grid = ((rows * h) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let h_i = h as i32;
        let params = [
            &xd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn norm_lerp6(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        xr: TensorId,
        xw: TensorId,
        xk: TensorId,
        xv: TensorId,
        xa: TensorId,
        xg: TensorId,
        or: TensorId,
        ow: TensorId,
        ok: TensorId,
        ov: TensorId,
        oa: TensorId,
        og: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        // 全部为 f32 张量，取 device 指针。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, xr) = (
            f32(x, "norm_lerp6")?,
            f32(s, "norm_lerp6")?,
            f32(g, "norm_lerp6")?,
            f32(b, "norm_lerp6")?,
            f32(xr, "norm_lerp6")?,
        );
        let (xw, xk, xv, xa, xg) = (
            f32(xw, "norm_lerp6")?,
            f32(xk, "norm_lerp6")?,
            f32(xv, "norm_lerp6")?,
            f32(xa, "norm_lerp6")?,
            f32(xg, "norm_lerp6")?,
        );
        let (or, ow, ok) = (
            f32(or, "norm_lerp6")?,
            f32(ow, "norm_lerp6")?,
            f32(ok, "norm_lerp6")?,
        );
        let (ov, oa, og) = (
            f32(ov, "norm_lerp6")?,
            f32(oa, "norm_lerp6")?,
            f32(og, "norm_lerp6")?,
        );
        let func = self.kernel("norm_lerp6", NORM_LERP6_SRC, "norm_lerp6")?;
        // 多 block 并行：每个 block 负责 256 个 C 片段（冗余归约 + 分段 apply）。
        let grid = (c.div_ceil(256).max(1) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &xr as *const u64 as *mut c_void,
            &xw as *const u64 as *mut c_void,
            &xk as *const u64 as *mut c_void,
            &xv as *const u64 as *mut c_void,
            &xa as *const u64 as *mut c_void,
            &xg as *const u64 as *mut c_void,
            &or as *const u64 as *mut c_void,
            &ow as *const u64 as *mut c_void,
            &ok as *const u64 as *mut c_void,
            &ov as *const u64 as *mut c_void,
            &oa as *const u64 as *mut c_void,
            &og as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn cmix_norm_lerp(
        &mut self,
        x: TensorId,
        s: TensorId,
        g: TensorId,
        b: TensorId,
        coeff: TensorId,
        out_xb: TensorId,
        c: usize,
        eps: f32,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (x, s, g, b, coeff, out_xb) = (
            f32(x, "cmix_norm_lerp")?,
            f32(s, "cmix_norm_lerp")?,
            f32(g, "cmix_norm_lerp")?,
            f32(b, "cmix_norm_lerp")?,
            f32(coeff, "cmix_norm_lerp")?,
            f32(out_xb, "cmix_norm_lerp")?,
        );
        let func = self.kernel("cmix_norm_lerp", CMIX_NORM_LERP_SRC, "cmix_norm_lerp")?;
        // 多 block 并行：每个 block 负责 256 个 C 片段（冗余归约 + 分段 apply）。
        let grid = (c.div_ceil(256).max(1) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let c_i = c as i32;
        let params = [
            &x as *const u64 as *mut c_void,
            &s as *const u64 as *mut c_void,
            &g as *const u64 as *mut c_void,
            &b as *const u64 as *mut c_void,
            &coeff as *const u64 as *mut c_void,
            &out_xb as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        // s 为 f32（in-place 更新，写 device 指针）；a/v/w 为 fp16；其余 f32。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let sd = f32(s, "fuse_ka_dplr_norm")?;
        let kd = f32(k, "fuse_ka_dplr_norm")?;
        let kkd = f32(k_k, "fuse_ka_dplr_norm")?;
        let ad = f16(a, "fuse_ka_dplr_norm")?;
        let kad = f32(k_a, "fuse_ka_dplr_norm")?;
        let rd = f32(r, "fuse_ka_dplr_norm")?;
        let vd = f16(v, "fuse_ka_dplr_norm")?;
        let wd = f16(w, "fuse_ka_dplr_norm")?;
        let gd = f32(gamma, "fuse_ka_dplr_norm")?;
        let bd = f32(beta, "fuse_ka_dplr_norm")?;
        let rkd = f32(r_k, "fuse_ka_dplr_norm")?;
        let kmd = f32(k_mod, "fuse_ka_dplr_norm")?;
        let yd = f32(y, "fuse_ka_dplr_norm")?;
        let ynd = f32(y_norm, "fuse_ka_dplr_norm")?;

        let func = self.kernel(
            "fuse_ka_dplr_norm",
            FUSE_KA_DPLR_NORM_SRC,
            "fuse_ka_dplr_norm",
        )?;
        // 每个 block 处理一个 (head,batch)，128 线程 = N*SPLIT。
        let grid = (h as u32, 1u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let h_i = h as i32;
        let n_i = n as i32;
        let params = [
            &sd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &kkd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &kad as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &rkd as *const u64 as *mut c_void,
            &kmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ynd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &eps as *const f32 as *mut c_void,
            &gn_eps as *const f32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        or: TensorId,
        ok: TensorId,
        ov: TensorId,
        ovm: TensorId,
        owm: TensorId,
        oam: TensorId,
        ogm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        // r/k/v 与 out_v 为 fp16；x 与其余 out 为 f32；v1/w1/a1/g1 权重为 f32。
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let rd = f16(r, "gemv_rkv_stage1")?;
        let kd = f16(k, "gemv_rkv_stage1")?;
        let vd = f16(v, "gemv_rkv_stage1")?;
        let v1d = f32(v1, "gemv_rkv_stage1")?;
        let w1d = f32(w1, "gemv_rkv_stage1")?;
        let a1d = f32(a1, "gemv_rkv_stage1")?;
        let g1d = f32(g1, "gemv_rkv_stage1")?;
        let xrd = f32(xr, "gemv_rkv_stage1")?;
        let xkd = f32(xk, "gemv_rkv_stage1")?;
        let xvd = f32(xv, "gemv_rkv_stage1")?;
        let xwd = f32(xw, "gemv_rkv_stage1")?;
        let xad = f32(xa, "gemv_rkv_stage1")?;
        let xgd = f32(xg, "gemv_rkv_stage1")?;
        let ord = f32(or, "gemv_rkv_stage1")?;
        let okd = f32(ok, "gemv_rkv_stage1")?;
        let ovd = f16(ov, "gemv_rkv_stage1")?;
        let ovmd = f32(ovm, "gemv_rkv_stage1")?;
        let owmd = f32(owm, "gemv_rkv_stage1")?;
        let oamd = f32(oam, "gemv_rkv_stage1")?;
        let ogmd = f32(ogm, "gemv_rkv_stage1")?;

        let func = self.kernel("gemv_rkv_stage1", GEMV_RKV_STAGE1_SRC, "gemv_rkv_stage1")?;
        // dispatch (C/ROWS + VM + WM + AM + GM, 1, 1)，block=128。
        let grid = ((c / 4 + vm + wm + am + gm) as u32, 1u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        or: TensorId,
        ok: TensorId,
        ov: TensorId,
        ovm: TensorId,
        owm: TensorId,
        oam: TensorId,
        ogm: TensorId,
        c: usize,
        vm: usize,
        wm: usize,
        am: usize,
        gm: usize,
    ) -> R<()> {
        // int8 句柄：idx/sz 均为 u32；mid 权重与 x 为 f32；out_v 为 f16。
        let u32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let r_idx = u32(r.idx, "gemv_int8_rkv_stage1")?;
        let r_sz = u32(r.sz, "gemv_int8_rkv_stage1")?;
        let k_idx = u32(k.idx, "gemv_int8_rkv_stage1")?;
        let k_sz = u32(k.sz, "gemv_int8_rkv_stage1")?;
        let v_idx = u32(v.idx, "gemv_int8_rkv_stage1")?;
        let v_sz = u32(v.sz, "gemv_int8_rkv_stage1")?;
        let v1d = f32(v1, "gemv_int8_rkv_stage1")?;
        let w1d = f32(w1, "gemv_int8_rkv_stage1")?;
        let a1d = f32(a1, "gemv_int8_rkv_stage1")?;
        let g1d = f32(g1, "gemv_int8_rkv_stage1")?;
        let xrd = f32(xr, "gemv_int8_rkv_stage1")?;
        let xkd = f32(xk, "gemv_int8_rkv_stage1")?;
        let xvd = f32(xv, "gemv_int8_rkv_stage1")?;
        let xwd = f32(xw, "gemv_int8_rkv_stage1")?;
        let xad = f32(xa, "gemv_int8_rkv_stage1")?;
        let xgd = f32(xg, "gemv_int8_rkv_stage1")?;
        let ord = f32(or, "gemv_int8_rkv_stage1")?;
        let okd = f32(ok, "gemv_int8_rkv_stage1")?;
        let ovd = f16(ov, "gemv_int8_rkv_stage1")?;
        let ovmd = f32(ovm, "gemv_int8_rkv_stage1")?;
        let owmd = f32(owm, "gemv_int8_rkv_stage1")?;
        let oamd = f32(oam, "gemv_int8_rkv_stage1")?;
        let ogmd = f32(ogm, "gemv_int8_rkv_stage1")?;

        let func = self.kernel(
            "gemv_int8_rkv_stage1",
            GEMV_INT8_RKV_STAGE1_SRC,
            "gemv_int8_rkv_stage1",
        )?;
        let grid = ((c / 4 + vm + wm + am + gm) as u32, 1u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let vm_i = vm as i32;
        let wm_i = wm as i32;
        let am_i = am as i32;
        let gm_i = gm as i32;
        let params = [
            &r_idx as *const u64 as *mut c_void,
            &r_sz as *const u64 as *mut c_void,
            &k_idx as *const u64 as *mut c_void,
            &k_sz as *const u64 as *mut c_void,
            &v_idx as *const u64 as *mut c_void,
            &v_sz as *const u64 as *mut c_void,
            &v1d as *const u64 as *mut c_void,
            &w1d as *const u64 as *mut c_void,
            &a1d as *const u64 as *mut c_void,
            &g1d as *const u64 as *mut c_void,
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &xwd as *const u64 as *mut c_void,
            &xad as *const u64 as *mut c_void,
            &xgd as *const u64 as *mut c_void,
            &ord as *const u64 as *mut c_void,
            &okd as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ovmd as *const u64 as *mut c_void,
            &owmd as *const u64 as *mut c_void,
            &oamd as *const u64 as *mut c_void,
            &ogmd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &vm_i as *const i32 as *mut c_void,
            &wm_i as *const i32 as *mut c_void,
            &am_i as *const i32 as *mut c_void,
            &gm_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_lowrank_chain4(
        &mut self,
        w2: TensorId,
        a2: TensorId,
        v2: TensorId,
        g2: TensorId,
        wm: TensorId,
        am: TensorId,
        vm: TensorId,
        gm: TensorId,
        w0: TensorId,
        a0: TensorId,
        v0: TensorId,
        scale: TensorId,
        vf: TensorId,
        ow: TensorId,
        oa: TensorId,
        ov: TensorId,
        og: TensorId,
        m: usize,
        kw: usize,
        ka: usize,
        kv: usize,
        kg: usize,
    ) -> R<()> {
        // 权重、mid、bias、scale 为 f32；v_first 与 4 个输出为 f16。
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let f16 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F16 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f16").into()),
            }
        };
        let w2d = f32(w2, "gemv_lowrank_chain4")?;
        let a2d = f32(a2, "gemv_lowrank_chain4")?;
        let v2d = f32(v2, "gemv_lowrank_chain4")?;
        let g2d = f32(g2, "gemv_lowrank_chain4")?;
        let wmd = f32(wm, "gemv_lowrank_chain4")?;
        let amd = f32(am, "gemv_lowrank_chain4")?;
        let vmd = f32(vm, "gemv_lowrank_chain4")?;
        let gmd = f32(gm, "gemv_lowrank_chain4")?;
        let w0d = f32(w0, "gemv_lowrank_chain4")?;
        let a0d = f32(a0, "gemv_lowrank_chain4")?;
        let v0d = f32(v0, "gemv_lowrank_chain4")?;
        let scaled = f32(scale, "gemv_lowrank_chain4")?;
        let vfd = f16(vf, "gemv_lowrank_chain4")?;
        let owd = f16(ow, "gemv_lowrank_chain4")?;
        let oad = f16(oa, "gemv_lowrank_chain4")?;
        let ovd = f16(ov, "gemv_lowrank_chain4")?;
        let ogd = f16(og, "gemv_lowrank_chain4")?;

        let func = self.kernel(
            "gemv_lowrank_chain4",
            GEMV_LOWRANK_CHAIN4_SRC,
            "gemv_lowrank_chain4",
        )?;
        let grid = (m as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let m_i = m as i32;
        let kw_i = kw as i32;
        let ka_i = ka as i32;
        let kv_i = kv as i32;
        let kg_i = kg as i32;
        let params = [
            &w2d as *const u64 as *mut c_void,
            &a2d as *const u64 as *mut c_void,
            &v2d as *const u64 as *mut c_void,
            &g2d as *const u64 as *mut c_void,
            &wmd as *const u64 as *mut c_void,
            &amd as *const u64 as *mut c_void,
            &vmd as *const u64 as *mut c_void,
            &gmd as *const u64 as *mut c_void,
            &w0d as *const u64 as *mut c_void,
            &a0d as *const u64 as *mut c_void,
            &v0d as *const u64 as *mut c_void,
            &scaled as *const u64 as *mut c_void,
            &vfd as *const u64 as *mut c_void,
            &owd as *const u64 as *mut c_void,
            &oad as *const u64 as *mut c_void,
            &ovd as *const u64 as *mut c_void,
            &ogd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &kw_i as *const i32 as *mut c_void,
            &ka_i as *const i32 as *mut c_void,
            &kv_i as *const i32 as *mut c_void,
            &kg_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn gemv_f16_relu2(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_relu2")?;
        let xd = self.f32_ptr(x, "gemv_f16_relu2")?;
        let yd = self.f32_ptr(y, "gemv_f16_relu2")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, 0, yd, m, k, b, 0, 0)
    }
    fn gemv_int8_relu2(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_relu2")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_relu2")?;
        let xd = self.f32_ptr(x, "gemv_int8_relu2")?;
        let yd = self.f32_ptr(y, "gemv_int8_relu2")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 0)
    }
    fn gemv_f16_mul_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_mul_add")?;
        let xd = self.f32_ptr(x, "gemv_f16_mul_add")?;
        let gd = self.f16_ptr(g, "gemv_f16_mul_add")?;
        let yd = self.f32_ptr(y, "gemv_f16_mul_add")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, gd, yd, m, k, b, 0, 1)
    }
    fn gemv_int8_mul_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        g: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_mul_add")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_mul_add")?;
        let xd = self.f32_ptr(x, "gemv_int8_mul_add")?;
        let gd = self.f16_ptr(g, "gemv_int8_mul_add")?;
        let yd = self.f32_ptr(y, "gemv_int8_mul_add")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, gd, yd, m, k, b, 2, 1)
    }
    fn gemv_f16_add(
        &mut self,
        a: TensorId,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let af16 = self.f16_ptr(a, "gemv_f16_add")?;
        let xd = self.f32_ptr(x, "gemv_f16_add")?;
        let yd = self.f32_ptr(y, "gemv_f16_add")?;
        self.gemv_variant_dispatch(af16, 0, 0, 0, xd, 0, yd, m, k, b, 0, 2)
    }
    fn gemv_int8_add(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_add")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_add")?;
        let xd = self.f32_ptr(x, "gemv_int8_add")?;
        let yd = self.f32_ptr(y, "gemv_int8_add")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 2)
    }
    /// y = x @ A（int8 量化权重，f32 输出，覆盖写）——head 用。
    fn gemv_int8_plain(
        &mut self,
        a: &Int8Handle,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
        b: usize,
    ) -> R<()> {
        let aidx = self.u32_ptr(a.idx, "gemv_int8_plain")?;
        let asz = self.u32_ptr(a.sz, "gemv_int8_plain")?;
        let xd = self.f32_ptr(x, "gemv_int8_plain")?;
        let yd = self.f32_ptr(y, "gemv_int8_plain")?;
        self.gemv_variant_dispatch(0, aidx, 0, asz, xd, 0, yd, m, k, b, 2, 3)
    }

    fn ffn_value_sparse_add(
        &mut self,
        value_w16: Option<TensorId>,
        value_tiled: TensorId,
        r2: TensorId,
        x: TensorId,
        c: usize,
        fh: usize,
    ) -> R<()> {
        let _ = value_w16; // CudaBackend 走稀疏内核，稠密权重仅作回退占位
        let vt = self.f16_ptr(value_tiled, "ffn_value_sparse_add")?;
        let rd = self.f32_ptr(r2, "ffn_value_sparse_add")?;
        let xd = self.f32_ptr(x, "ffn_value_sparse_add")?;
        let func = self.kernel(
            "ffn_value_sparse_add",
            FFN_VALUE_SPARSE_SRC,
            "ffn_value_sparse_add",
        )?;
        let grid = ((fh / 128) as u32, (c / 256) as u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let c_i = c as i32;
        let fh_i = fh as i32;
        let params = [
            &rd as *const u64 as *mut c_void,
            &vt as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &fh_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }

    fn supports_sparse_ffn(&self) -> bool {
        true
    }

    fn argmax(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        let logits_d = self.f32_ptr(logits, "argmax")?;
        let token_d = self.f32_ptr(token, "argmax")?;
        let func = self.kernel("rwkv_argmax", ARGMAX_SRC, "rwkv_argmax")?;
        let grid = (1u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let n_i = n as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        // 自建采样临时缓冲（对齐 Vulkan/CUDA 统一 kernel 语义）。
        let temp = self.create_tensor(n, TensorDtype::F32)?;
        let mask = self.create_tensor(n, TensorDtype::F32)?;
        let counter = self.create_tensor(n, TensorDtype::U32)?;
        let sampler = self.create_tensor(8, TensorDtype::F32)?;
        self.store_sampler_host(
            sampler,
            temperature,
            top_k,
            top_p,
            seed,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            history.len() as u32,
        )?;
        // 历史 token 缓冲（hist_len=0 时 kernel 跳过直方图，缓冲仅需合法地址）
        let hist = self.create_tensor(history.len().max(1), TensorDtype::U32)?;
        if !history.is_empty() {
            self.upload_u32(hist, history)?;
        }
        self.sample_into_host_seeded(logits, token, n, temp, mask, counter, sampler, hist)
    }
    fn clear_cache(&mut self) {
        // CUDA kernel 与缓冲地址/形状无关（地址全部为启动参数，PTX 由静态源码
        // 编译），跨 T 变化可安全复用，无需清空 kernels 重编译。
        // 仅销毁 prefill graph：graph 内 bake 了缓冲指针，T 变化重建 seq 缓冲后失效。
        for (_, exec) in self.prefill_graphs.drain() {
            unsafe {
                (self.drv.cu_graph_exec_destroy)(exec);
            }
        }
    }
    fn drop_host(&mut self, _t: TensorId) {
        // CUDA 后端张量仅持有设备指针，无 host 镜像缓冲，无需释放。
    }

    fn free_tensor(&mut self, t: TensorId) {
        // 移除注册表条目并释放设备内存（防 seq 缓冲按 T 重建时泄漏）。
        if let Some(tensor) = self.tensors.remove(&t) {
            self.lens.remove(&t);
            let dptr = match tensor {
                CudaTensor::F32 { dptr, .. }
                | CudaTensor::F16 { dptr, .. }
                | CudaTensor::U32 { dptr, .. } => dptr,
            };
            unsafe {
                (self.drv.cu_mem_free_v2)(dptr);
            }
        }
    }
    fn copy_device(&mut self, src: TensorId, dst: TensorId) -> R<()> {
        let src_d = self.f32_ptr(src, "copy_device")?;
        let dst_d = self.f32_ptr(dst, "copy_device")?;
        let len = *self.lens.get(&src).ok_or("copy_device: unknown src len")?;
        let func = self.kernel("copy_device", COPY_DEVICE_SRC, "rwkv_copy_device")?;
        let grid = ((len as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let len_i = len as i32;
        let params = [
            &src_d as *const u64 as *mut c_void,
            &dst_d as *const u64 as *mut c_void,
            &len_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_token(
        &mut self,
        x: TensorId,
        y: TensorId,
        c: usize,
        stride: usize,
        token: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "copy_token")?;
        let yd = self.f32_ptr(y, "copy_token")?;
        let func = self.kernel("copy_token", COPY_TOKEN_SRC, "rwkv_copy_token")?;
        let grid = ((c as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, stride_i, token_i) = (c as i32, stride as i32, token as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &stride_i as *const i32 as *mut c_void,
            &token_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm")?,
            self.f16_ptr(b, "gemm")?,
            self.f32_ptr(c, "gemm")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 0)
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
        let (ad, bd, biasd, cd) = (
            self.f16_ptr(a, "gemm_bias")?,
            self.f16_ptr(b, "gemm_bias")?,
            self.f32_ptr(bias, "gemm_bias")?,
            self.f32_ptr(c, "gemm_bias")?,
        );
        self.gemm_dispatch(ad, bd, Some(biasd), None, cd, m, n, k, 1)
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
        let (ad, bd, xd, yd) = (
            self.f16_ptr(a, "gemm_add")?,
            self.f16_ptr(b, "gemm_add")?,
            self.f32_ptr(x, "gemm_add")?,
            self.f32_ptr(y, "gemm_add")?,
        );
        self.gemm_dispatch(ad, bd, None, Some(xd), yd, m, n, k, 2)
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
        let xd = self.f32_ptr(x, "to_f16")?;
        let yd = self.f16_ptr(y, "to_f16")?;
        let func = self.kernel("to_f16", TO_F16_SRC, "rwkv_to_f16")?;
        let grid = (m_pad as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, xs, ys) = (c as i32, t as i32, x_stride as i32, y_stride as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (xrd, xkd, xvd) = (
            self.f32_ptr(xr, "to_f16_triple")?,
            self.f32_ptr(xk, "to_f16_triple")?,
            self.f32_ptr(xv, "to_f16_triple")?,
        );
        let (yrd, ykd, yvd) = (
            self.f16_ptr(yr, "to_f16_triple")?,
            self.f16_ptr(yk, "to_f16_triple")?,
            self.f16_ptr(yv, "to_f16_triple")?,
        );
        let func = self.kernel("to_f16_triple", TO_F16_TRIPLE_SRC, "rwkv_to_f16_triple")?;
        let grid = (m_pad as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, xs, ys) = (c as i32, t as i32, x_stride as i32, y_stride as i32);
        let params = [
            &xrd as *const u64 as *mut c_void,
            &xkd as *const u64 as *mut c_void,
            &xvd as *const u64 as *mut c_void,
            &yrd as *const u64 as *mut c_void,
            &ykd as *const u64 as *mut c_void,
            &yvd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn dequant_int8_to_f16(&mut self, a: &Int8Handle, out: TensorId, m: usize, k: usize) -> R<()> {
        let (idx, sz) = (
            self.u32_ptr(a.idx, "dequant_int8_to_f16")?,
            self.u32_ptr(a.sz, "dequant_int8_to_f16")?,
        );
        let w = self.f16_ptr(out, "dequant_int8_to_f16")?;
        let func = self.kernel("dequant_int8", DEQUANT_INT8_SRC, "rwkv_dequant_int8")?;
        let total = m * (k / 4);
        let grid = ((total as u32).div_ceil(256), 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (m_i, k_i) = (m as i32, k as i32);
        let params = [
            &idx as *const u64 as *mut c_void,
            &sz as *const u64 as *mut c_void,
            &w as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_sigmoid(&mut self, a: TensorId, y: TensorId, c: usize, batch: usize) -> R<()> {
        let ad = self.f32_ptr(a, "elementwise_sigmoid")?;
        let yd = self.f32_ptr(y, "elementwise_sigmoid")?;
        let func = self.kernel(
            "elementwise_sigmoid",
            ELEMENTWISE_SIGMOID_SRC,
            "rwkv_elementwise_sigmoid",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_sigmoid_inplace(&mut self, y: TensorId, c: usize, batch: usize) -> R<()> {
        // 原地 sigmoid：a 与 y 指向同一张量。
        self.elementwise_sigmoid(y, y, c, batch)
    }
    fn fuse_ka(
        &mut self,
        k: TensorId,
        kk_w: TensorId,
        a: TensorId,
        ka_w: TensorId,
        k_mod: TensorId,
        kk_l2: TensorId,
        b: TensorId,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (kd, kk_d, ad, ka_d, km_d, kl_d, bd) = (
            f32(k, "fuse_ka")?,
            f32(kk_w, "fuse_ka")?,
            f32(a, "fuse_ka")?,
            f32(ka_w, "fuse_ka")?,
            f32(k_mod, "fuse_ka")?,
            f32(kk_l2, "fuse_ka")?,
            f32(b, "fuse_ka")?,
        );
        let func = self.kernel("fuse_ka", FUSE_KA_SRC, "rwkv_fuse_ka")?;
        let grid = (h as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (h_i, n_i, b_i) = (h as i32, n as i32, batch as i32);
        let params = [
            &kd as *const u64 as *mut c_void,
            &kk_d as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &ka_d as *const u64 as *mut c_void,
            &km_d as *const u64 as *mut c_void,
            &kl_d as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (rd, km_d, rk_d, vd, yd) = (
            f32(r, "sum_rk_rk")?,
            f32(k_mod, "sum_rk_rk")?,
            f32(r_k, "sum_rk_rk")?,
            f32(v, "sum_rk_rk")?,
            f32(y, "sum_rk_rk")?,
        );
        let func = self.kernel("sum_rk_rk", SUM_RK_RK_SRC, "rwkv_sum_rk_rk")?;
        let grid = (h as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (h_i, n_i, b_i) = (h as i32, n as i32, batch as i32);
        let params = [
            &rd as *const u64 as *mut c_void,
            &km_d as *const u64 as *mut c_void,
            &rk_d as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (xd, sd, tmd, yd) = (
            self.f32_ptr(x, "seq_shift")?,
            self.f32_ptr(state, "seq_shift")?,
            self.f32_ptr(tm, "seq_shift")?,
            self.f32_ptr(y, "seq_shift")?,
        );
        let func = self.kernel("seq_shift", SEQ_SHIFT_SRC, "rwkv_seq_shift")?;
        let grid = (t as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, sx, sy) = (c as i32, t as i32, stride_x as i32, stride_y as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &tmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &sx as *const i32 as *mut c_void,
            &sy as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    #[allow(clippy::too_many_arguments)]
    fn seq_shift_batch(
        &mut self,
        x: TensorId,
        state: TensorId,
        tm: TensorId,
        y: TensorId,
        c: usize,
        t: usize,
        stride_x: usize,
        stride_y: usize,
        batch: usize,
    ) -> R<()> {
        let (xd, sd, tmd, yd) = (
            self.f32_ptr(x, "seq_shift_batch")?,
            self.f32_ptr(state, "seq_shift_batch")?,
            self.f32_ptr(tm, "seq_shift_batch")?,
            self.f32_ptr(y, "seq_shift_batch")?,
        );
        let func = self.kernel(
            "seq_shift_batch",
            SEQ_SHIFT_BATCH_SRC,
            "rwkv_seq_shift_batch",
        )?;
        let grid = (t as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, sx, sy) = (c as i32, t as i32, stride_x as i32, stride_y as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &tmd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &sx as *const i32 as *mut c_void,
            &sy as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn copy_token_batch(
        &mut self,
        x: TensorId,
        state: TensorId,
        lens: TensorId,
        c: usize,
        t: usize,
        batch: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "copy_token_batch")?;
        let sd = self.f32_ptr(state, "copy_token_batch")?;
        let ld = self.u32_ptr(lens, "copy_token_batch")?;
        let func = self.kernel(
            "copy_token_batch",
            COPY_TOKEN_BATCH_SRC,
            "rwkv_copy_token_batch",
        )?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i) = (c as i32, t as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &sd as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    #[allow(clippy::too_many_arguments)]
    fn dplr_seq_batch(
        &mut self,
        s: TensorId,
        r: TensorId,
        w: TensorId,
        k: TensorId,
        v: TensorId,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        lens: TensorId,
        h: usize,
        n: usize,
        t: usize,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (sd, rd, wd, kd, vd, ad, bd, yd) = (
            f32(s, "dplr_seq_batch")?,
            f32(r, "dplr_seq_batch")?,
            f32(w, "dplr_seq_batch")?,
            f32(k, "dplr_seq_batch")?,
            f32(v, "dplr_seq_batch")?,
            f32(a, "dplr_seq_batch")?,
            f32(b, "dplr_seq_batch")?,
            f32(y, "dplr_seq_batch")?,
        );
        let ld = self.u32_ptr(lens, "dplr_seq_batch")?;
        let func = self.kernel("dplr_seq_batch", DPLR_SEQ_BATCH_SRC, "rwkv_dplr_seq_batch")?;
        let blocks = h * n;
        let grid = ((blocks as u32).div_ceil(8), batch as u32, 1u32);
        let block = (128u32, 1u32, 1u32);
        let (h_i, n_i, t_i, c_i) = (h as i32, n as i32, t as i32, c as i32);
        let params = [
            &sd as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn segmean(
        &mut self,
        x: TensorId,
        out: TensorId,
        lens: TensorId,
        c: usize,
        t_pad: usize,
        batch: usize,
    ) -> R<()> {
        let xd = self.f32_ptr(x, "segmean")?;
        let od = self.f32_ptr(out, "segmean")?;
        let ld = self.u32_ptr(lens, "segmean")?;
        let func = self.kernel("segmean", SEGMEAN_SRC, "rwkv_segmean")?;
        let grid = (batch as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i) = (c as i32, t_pad as i32);
        let params = [
            &xd as *const u64 as *mut c_void,
            &od as *const u64 as *mut c_void,
            &ld as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let f32 = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be f32").into()),
            }
        };
        let (sd, rd, wd, kd, vd, ad, bd, yd) = (
            f32(s, "dplr_seq")?,
            f32(r, "dplr_seq")?,
            f32(w, "dplr_seq")?,
            f32(k, "dplr_seq")?,
            f32(v, "dplr_seq")?,
            f32(a, "dplr_seq")?,
            f32(b, "dplr_seq")?,
            f32(y, "dplr_seq")?,
        );
        let func = self.kernel("dplr_seq", DPLR_SEQ_SRC, "rwkv_dplr_seq")?;
        // 128 线程/block 处理 8 个状态行（4 warp × 2 half-warp），grid.x = ceil(h*n/8)。
        let blocks = h * n;
        let grid = ((blocks as u32).div_ceil(8), 1u32, 1u32);
        let block = (128u32, 1u32, 1u32); // 要求 n<=64
        let (h_i, n_i, t_i, c_i) = (h as i32, n as i32, t as i32, c as i32);
        let params = [
            &sd as *const u64 as *mut c_void,
            &rd as *const u64 as *mut c_void,
            &wd as *const u64 as *mut c_void,
            &kd as *const u64 as *mut c_void,
            &vd as *const u64 as *mut c_void,
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &h_i as *const i32 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm_relu2")?,
            self.f16_ptr(b, "gemm_relu2")?,
            self.f32_ptr(c, "gemm_relu2")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 3)
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
        let (ad, bd, cd) = (
            self.f16_ptr(a, "gemm_tanh")?,
            self.f16_ptr(b, "gemm_tanh")?,
            self.f32_ptr(c, "gemm_tanh")?,
        );
        self.gemm_dispatch(ad, bd, None, None, cd, m, n, k, 4)
    }
    fn elementwise_scale_exp(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let (ad, bd, yd) = (
            self.f32_ptr(a, "elementwise_scale_exp")?,
            self.f32_ptr(b, "elementwise_scale_exp")?,
            self.f32_ptr(y, "elementwise_scale_exp")?,
        );
        let func = self.kernel(
            "elementwise_scale_exp",
            ELEMENTWISE_SCALE_EXP_SRC,
            "rwkv_elementwise_scale_exp",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn elementwise_mul(
        &mut self,
        a: TensorId,
        b: TensorId,
        y: TensorId,
        c: usize,
        batch: usize,
    ) -> R<()> {
        let (ad, bd, yd) = (
            self.f32_ptr(a, "elementwise_mul")?,
            self.f32_ptr(b, "elementwise_mul")?,
            self.f32_ptr(y, "elementwise_mul")?,
        );
        let func = self.kernel(
            "elementwise_mul",
            ELEMENTWISE_MUL_SRC,
            "rwkv_elementwise_mul",
        )?;
        let grid = (1u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, b_i) = (c as i32, batch as i32);
        let params = [
            &ad as *const u64 as *mut c_void,
            &bd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (vd, gd, fvd) = (
            self.f32_ptr(v, "v_first_lerp")?,
            self.f32_ptr(gate, "v_first_lerp")?,
            self.f32_ptr(v_first, "v_first_lerp")?,
        );
        let func = self.kernel("v_first_lerp", V_FIRST_LERP_SRC, "rwkv_v_first_lerp")?;
        let grid = (t as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (c_i, t_i, s_i) = (c as i32, t as i32, stride as i32);
        let params = [
            &vd as *const u64 as *mut c_void,
            &gd as *const u64 as *mut c_void,
            &fvd as *const u64 as *mut c_void,
            &c_i as *const i32 as *mut c_void,
            &t_i as *const i32 as *mut c_void,
            &s_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let (ad, xd, yd) = (
            self.f32_ptr(a, "gemv_seq")?,
            self.f32_ptr(x, "gemv_seq")?,
            self.f32_ptr(y, "gemv_seq")?,
        );
        let func = self.kernel("gemv_seq", GEMV_SEQ_SRC, "rwkv_gemv_seq")?;
        let grid = (m as u32, batch as u32, 1u32);
        let block = (256u32, 1u32, 1u32);
        let (m_i, k_i, xs, ys, b_i) = (
            m as i32,
            k as i32,
            x_stride as i32,
            y_stride as i32,
            batch as i32,
        );
        let params = [
            &ad as *const u64 as *mut c_void,
            &xd as *const u64 as *mut c_void,
            &yd as *const u64 as *mut c_void,
            &m_i as *const i32 as *mut c_void,
            &k_i as *const i32 as *mut c_void,
            &xs as *const i32 as *mut c_void,
            &ys as *const i32 as *mut c_void,
            &b_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
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
        let data = [
            temperature,
            f32::from_bits(top_k),
            top_p,
            f32::from_bits(seed),
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            f32::from_bits(hist_len),
        ];
        self.upload(sampler, &data)
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
        let logits_d = self.f32_ptr(logits, "sample_into_host_seeded")?;
        let token_d = self.f32_ptr(token, "sample_into_host_seeded")?;
        let temp_d = self.f32_ptr(temp, "sample_into_host_seeded")?;
        let mask_d = self.f32_ptr(mask, "sample_into_host_seeded")?;
        let counter_d = self.u32_ptr(counter, "sample_into_host_seeded")?;
        let sampler_d = self.f32_ptr(sampler, "sample_into_host_seeded")?;
        // hist 可能是 U32（sample 自建）或 F32（self-loop 的 token_seq，位模式存索引）。
        let hist_d = match self.get(hist, "sample_into_host_seeded")? {
            CudaTensor::U32 { dptr, .. } => dptr,
            CudaTensor::F32 { dptr, .. } => dptr,
            _ => return Err("sample_into_host_seeded: hist must be u32 or f32".into()),
        };
        let func = self.kernel("rwkv_sample", SAMPLE_SRC, "rwkv_sample")?;
        let grid = (1u32, 1u32, 1u32);
        // 注意：kernel 内共享内存数组按 BS=112 定义（s_val/s_idx），block 必须与 BS 一致，
        // 否则 tid>=112 的线程访问 s_val[tid] 越界 → 非法内存访问（sticky error 700）。
        let block = (112u32, 1u32, 1u32);
        let n_i = n as i32;
        let params = [
            &logits_d as *const u64 as *mut c_void,
            &token_d as *const u64 as *mut c_void,
            &temp_d as *const u64 as *mut c_void,
            &mask_d as *const u64 as *mut c_void,
            &counter_d as *const u64 as *mut c_void,
            &sampler_d as *const u64 as *mut c_void,
            &hist_d as *const u64 as *mut c_void,
            &n_i as *const i32 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn record_token(&mut self, in_tok: TensorId, out_seq: TensorId, cnt: TensorId) -> R<()> {
        // in_tok/out_seq/cnt 存 token 位模式，以 u32 读取（F32/U32 均可，位模式相同）。
        let ptr = |t: TensorId, op: &str| -> R<u64> {
            match self.get(t, op)? {
                CudaTensor::U32 { dptr, .. } => Ok(dptr),
                CudaTensor::F32 { dptr, .. } => Ok(dptr),
                _ => Err(format!("{op}: tensor {t:?} must be u32 or f32").into()),
            }
        };
        let in_tok_d = ptr(in_tok, "record_token")?;
        let out_d = ptr(out_seq, "record_token")?;
        let cnt_d = ptr(cnt, "record_token")?;
        let func = self.kernel("rwkv_record_token", RECORD_TOKEN_SRC, "rwkv_record_token")?;
        let grid = (1u32, 1u32, 1u32);
        let block = (1u32, 1u32, 1u32);
        let params = [
            &in_tok_d as *const u64 as *mut c_void,
            &out_d as *const u64 as *mut c_void,
            &cnt_d as *const u64 as *mut c_void,
        ];
        self.drv.launch(self.stream, func, grid, block, &params)
    }
    fn argmax_into_host(&mut self, logits: TensorId, token: TensorId, n: usize) -> R<()> {
        // CUDA 后端 token 即为设备 F32 缓冲，与 argmax 一致（语义相同，写位模式）。
        self.argmax(logits, token, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试辅助：创建张量（避免闭包捕获 `&mut b` 造成借用冲突）。
    fn mk_tensor(b: &mut CudaBackend, len: usize, dtype: TensorDtype) -> TensorId {
        b.create_tensor(len, dtype).expect("create tensor")
    }

    /// 探测 CUDA（驱动可加载 + ≥1 设备）。无 CUDA 环境下跳过。
    #[test]
    fn detect_cuda_available() {
        // 仅验证探测函数可调用且返回 bool；具体真值取决于本机硬件。
        let _ = cuda_available();
    }

    /// 骨架张量管理：create/upload/download 往返（f32 与 f16）。
    /// 无 CUDA 设备时跳过（cuda_available() 为 false）。
    #[test]
    fn tensor_upload_download_roundtrip() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping roundtrip test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // f32 往返
        let n = 256usize;
        let t = b.create_tensor(n, TensorDtype::F32).expect("create f32");
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        b.upload(t, &data).unwrap();
        let got = b.download(t).unwrap();
        let mut max_diff = 0.0f32;
        for (a, g) in data.iter().zip(got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert!(
            max_diff == 0.0,
            "f32 roundtrip mismatch, max_diff={max_diff}"
        );

        // f16 往返（经 half 转换，允许舍入误差）
        let t16 = b.create_tensor(n, TensorDtype::F16).expect("create f16");
        b.upload(t16, &data).unwrap();
        let got16 = b.download(t16).unwrap();
        let mut max_diff16 = 0.0f32;
        for (a, g) in data.iter().zip(got16.iter()) {
            max_diff16 = max_diff16.max((a - g).abs());
        }
        assert!(
            max_diff16 < 1e-2,
            "f16 roundtrip mismatch, max_diff={max_diff16}"
        );
        log::info!(
            "CUDA tensor upload/download roundtrip OK (f32 max_diff={max_diff}, f16 max_diff={max_diff16})"
        );
    }

    /// gemv_f16 与 CPU 参考对比：y[m] = Σ_k x[k]·A[m·K+k]（fp16 权重，f32 累加）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 8usize;
        let k = 256usize;
        let batch = 2usize;

        // 随机权重（NaiveXorshift，避免依赖外部 rand 种子）。
        let mut seed = 0x9E3779B9u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let a: Vec<f32> = (0..m * k).map(|_| rng()).collect();
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（fp16 权重量化后计算，与 GPU 一致）。
        let mut expect = vec![0.0f32; m * batch];
        for bb in 0..batch {
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let w = f16::from_f32(a[mm * k + kk]).to_f32();
                    acc += w * x[bb * k + kk];
                }
                expect[bb * m + mm] = acc;
            }
        }

        let w = b.create_tensor(m * k, TensorDtype::F16).expect("create w");
        let xt = b
            .create_tensor(k * batch, TensorDtype::F32)
            .expect("create x");
        let yt = b
            .create_tensor(m * batch, TensorDtype::F32)
            .expect("create y");
        b.upload(w, &a).unwrap();
        b.upload(xt, &x).unwrap();
        b.gemv_f16(w, xt, yt, m, k, batch).expect("gemv_f16");
        // 同步：cuMemcpyDtoH 隐式同步，kernel 已完成。
        let got = b.download(yt).unwrap();

        let mut max_diff = 0.0f32;
        for (e, g) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - g).abs());
        }
        assert!(
            max_diff < 1e-2,
            "gemv_f16 mismatch, max_diff={max_diff}\nexpect={expect:?}\ngot={got:?}"
        );
        log::info!("gemv_f16 vs CPU reference OK (max_diff={max_diff})");
    }

    /// norm_lerp6 与 CPU 参考对比：ln1 = LN(x) + 6 次 lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_lerp6_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm_lerp6 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let eps = 1e-5f32;

        let mut seed = 0x9E3779B9u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let xd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let coeffs: Vec<Vec<f32>> = (0..6).map(|_| (0..c).map(|_| rng()).collect()).collect();

        // CPU 参考
        let mean: f32 = xd.iter().sum::<f32>() / c as f32;
        let var: f32 = xd.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let ln1: Vec<f32> = (0..c)
            .map(|i| (xd[i] - mean) * inv_std * gd[i] + bd[i])
            .collect();
        let mut expect_o = vec![Vec::new(); 6];
        let mut expect_s = vec![0.0f32; c];
        for i in 0..c {
            for j in 0..6 {
                expect_o[j].push(ln1[i] + coeffs[j][i] * (sd[i] - ln1[i]));
            }
            expect_s[i] = ln1[i];
        }

        let x = mk_tensor(&mut b, c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, c, TensorDtype::F32);
        let ct: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        let outs: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        for j in 0..6 {
            b.upload(ct[j], &coeffs[j]).unwrap();
        }
        b.norm_lerp6(
            x, s, g, bb, ct[0], ct[1], ct[2], ct[3], ct[4], ct[5], outs[0], outs[1], outs[2],
            outs[3], outs[4], outs[5], c, eps,
        )
        .expect("norm_lerp6");
        let got_s = b.download(s).unwrap();
        for j in 0..6 {
            let got = b.download(outs[j]).unwrap();
            let mut diff = 0.0f32;
            for (e, gr) in expect_o[j].iter().zip(got.iter()) {
                diff = diff.max((e - gr).abs());
            }
            assert!(diff < 1e-3, "norm_lerp6 out[{j}] mismatch, max_diff={diff}");
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "norm_lerp6 state mismatch, max_diff={s_diff}"
        );
        log::info!("norm_lerp6 vs CPU reference OK (out/state max_diff<1e-3)");
    }

    /// cmix_norm_lerp 与 CPU 参考对比：ln2 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn cmix_norm_lerp_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cmix_norm_lerp test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let eps = 1e-5f32;

        let mut seed = 0x12345678u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let xd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let cd: Vec<f32> = (0..c).map(|_| rng()).collect();

        let mean: f32 = xd.iter().sum::<f32>() / c as f32;
        let var: f32 = xd.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut expect_xb = vec![0.0f32; c];
        let mut expect_s = vec![0.0f32; c];
        for i in 0..c {
            let ln2 = (xd[i] - mean) * inv_std * gd[i] + bd[i];
            expect_xb[i] = ln2 + cd[i] * (sd[i] - ln2);
            expect_s[i] = ln2;
        }

        let x = mk_tensor(&mut b, c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, c, TensorDtype::F32);
        let co = mk_tensor(&mut b, c, TensorDtype::F32);
        let o = mk_tensor(&mut b, c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.upload(co, &cd).unwrap();
        b.cmix_norm_lerp(x, s, g, bb, co, o, c, eps)
            .expect("cmix_norm_lerp");
        let got_xb = b.download(o).unwrap();
        let got_s = b.download(s).unwrap();
        let mut xb_diff = 0.0f32;
        for (e, gr) in expect_xb.iter().zip(got_xb.iter()) {
            xb_diff = xb_diff.max((e - gr).abs());
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(xb_diff < 1e-3, "cmix xb mismatch, max_diff={xb_diff}");
        assert!(s_diff < 1e-3, "cmix state mismatch, max_diff={s_diff}");
        log::info!("cmix_norm_lerp vs CPU reference OK (xb/state max_diff<1e-3)");
    }

    /// norm 与 CPU 参考对比：y = LN(x) * gamma + beta，逐 (head,batch) 行，跨 batch 共享 gamma/beta。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let h = 4usize;
        let batch = 3usize;
        let rows = batch * h;
        let eps = 1e-5f32;

        let mut seed = 0x0F0F0F0Fu32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // x 布局 [batch][head][c]；gamma/beta 布局 [head][c]（跨 batch 共享）。
        let xd: Vec<f32> = (0..batch * h * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..h * c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..h * c).map(|_| rng()).collect();

        // CPU 参考
        let mut expect = vec![0.0f32; batch * h * c];
        for bb in 0..batch {
            for hh in 0..h {
                let x_base = bb * h * c + hh * c;
                let g_base = hh * c;
                let mean: f32 = (0..c).map(|i| xd[x_base + i]).sum::<f32>() / c as f32;
                let var: f32 = (0..c)
                    .map(|i| {
                        let v = xd[x_base + i] - mean;
                        v * v
                    })
                    .sum::<f32>()
                    / c as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for i in 0..c {
                    expect[x_base + i] =
                        (xd[x_base + i] - mean) * inv_std * gd[g_base + i] + bd[g_base + i];
                }
            }
        }

        let x = mk_tensor(&mut b, batch * h * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, h * c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, h * c, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * h * c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.norm(x, g, bb, y, c, h, eps, rows).expect("norm");
        let got = b.download(y).unwrap();
        let mut max_diff = 0.0f32;
        for (e, gr) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - gr).abs());
        }
        assert!(max_diff < 1e-3, "norm mismatch, max_diff={max_diff}");
        log::info!("norm vs CPU reference OK (max_diff<1e-3)");
    }

    /// xorshift rng（batch 测试共享）。
    fn test_rng(seed: u32) -> impl FnMut() -> f32 {
        let mut seed = seed;
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// norm_lerp6_batch 与 CPU 参考对比：B slot 各自独立归一化 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn norm_lerp6_batch_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping norm_lerp6_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let batch = 4usize;
        let eps = 1e-5f32;
        let mut rng = test_rng(0xABCDEF01);

        let xd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        // lerp 系数为共享权重 [C]（跨 slot 共享，与线上 x_r..x_g 一致）。
        let coeffs: Vec<Vec<f32>> = (0..6).map(|_| (0..c).map(|_| rng()).collect()).collect();

        // CPU 参考（每 slot 独立归一化，系数共享）
        let mut expect_o = vec![vec![0.0f32; batch * c]; 6];
        let mut expect_s = vec![0.0f32; batch * c];
        for bi in 0..batch {
            let base = bi * c;
            let mean: f32 = (0..c).map(|i| xd[base + i]).sum::<f32>() / c as f32;
            let var: f32 = (0..c)
                .map(|i| {
                    let v = xd[base + i] - mean;
                    v * v
                })
                .sum::<f32>()
                / c as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..c {
                let ln1 = (xd[base + i] - mean) * inv_std * gd[i] + bd[i];
                for j in 0..6 {
                    expect_o[j][base + i] = ln1 + coeffs[j][i] * (sd[base + i] - ln1);
                }
                expect_s[base + i] = ln1;
            }
        }

        let x = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let beta = mk_tensor(&mut b, c, TensorDtype::F32);
        let ct: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, c, TensorDtype::F32))
            .collect();
        let outs: Vec<_> = (0..6)
            .map(|_| mk_tensor(&mut b, batch * c, TensorDtype::F32))
            .collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(beta, &bd).unwrap();
        for j in 0..6 {
            b.upload(ct[j], &coeffs[j]).unwrap();
        }
        b.norm_lerp6_batch(
            x, s, g, beta, ct[0], ct[1], ct[2], ct[3], ct[4], ct[5], outs[0], outs[1], outs[2],
            outs[3], outs[4], outs[5], c, eps, batch,
        )
        .expect("norm_lerp6_batch");
        let got_s = b.download(s).unwrap();
        for j in 0..6 {
            let got = b.download(outs[j]).unwrap();
            let mut diff = 0.0f32;
            for (e, gr) in expect_o[j].iter().zip(got.iter()) {
                diff = diff.max((e - gr).abs());
            }
            assert!(
                diff < 1e-3,
                "norm_lerp6_batch out[{j}] mismatch, max_diff={diff}"
            );
        }
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "norm_lerp6_batch state mismatch, max_diff={s_diff}"
        );
        log::info!("norm_lerp6_batch vs CPU reference OK (batch={batch})");
    }

    /// cmix_norm_lerp_batch 与 CPU 参考对比：B slot 各自独立 ln2 + lerp + state 写回。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn cmix_norm_lerp_batch_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cmix_norm_lerp_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let batch = 4usize;
        let eps = 1e-5f32;
        let mut rng = test_rng(0x12349999);

        let xd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let sd: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let gd: Vec<f32> = (0..c).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..c).map(|_| rng()).collect();
        let cd: Vec<f32> = (0..c).map(|_| rng()).collect();

        let mut expect_xb = vec![0.0f32; batch * c];
        let mut expect_s = vec![0.0f32; batch * c];
        for bi in 0..batch {
            let base = bi * c;
            let mean: f32 = (0..c).map(|i| xd[base + i]).sum::<f32>() / c as f32;
            let var: f32 = (0..c)
                .map(|i| {
                    let v = xd[base + i] - mean;
                    v * v
                })
                .sum::<f32>()
                / c as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..c {
                let ln2 = (xd[base + i] - mean) * inv_std * gd[i] + bd[i];
                expect_xb[base + i] = ln2 + cd[i] * (sd[base + i] - ln2);
                expect_s[base + i] = ln2;
            }
        }

        let x = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let g = mk_tensor(&mut b, c, TensorDtype::F32);
        let beta = mk_tensor(&mut b, c, TensorDtype::F32);
        let co = mk_tensor(&mut b, c, TensorDtype::F32);
        let o = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(beta, &bd).unwrap();
        b.upload(co, &cd).unwrap();
        b.cmix_norm_lerp_batch(x, s, g, beta, co, o, c, eps, batch)
            .expect("cmix_norm_lerp_batch");
        let got_xb = b.download(o).unwrap();
        let got_s = b.download(s).unwrap();
        let mut diff = 0.0f32;
        for (e, gr) in expect_xb.iter().zip(got_xb.iter()) {
            diff = diff.max((e - gr).abs());
        }
        assert!(
            diff < 1e-3,
            "cmix_norm_lerp_batch xb mismatch, max_diff={diff}"
        );
        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        assert!(
            s_diff < 1e-3,
            "cmix_norm_lerp_batch state mismatch, max_diff={s_diff}"
        );
        log::info!("cmix_norm_lerp_batch vs CPU reference OK (batch={batch})");
    }

    /// gather_rows_device_f16 与 CPU 参考对比：B slot 各自按 tok[b] 取行。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gather_rows_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gather_rows test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize;
        let vocab = 1024usize;
        let batch = 4usize;
        let mut rng = test_rng(0x5A5A5A5A);

        let src: Vec<f32> = (0..vocab * c).map(|_| rng()).collect();
        let toks: Vec<u32> = vec![0, 17, 512, vocab as u32 - 1];
        // 期望值做 f32→f16→f32 round-trip（upload 时后端转 f16 存储）。
        let mut expect = Vec::with_capacity(batch * c);
        for &t in &toks {
            let t = t as usize;
            for i in 0..c {
                expect.push(src[t * c + i]);
            }
        }

        let src_t = mk_tensor(&mut b, vocab * c, TensorDtype::F16);
        let dst = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let tok = mk_tensor(&mut b, batch, TensorDtype::F32);
        b.upload(src_t, &src).unwrap();
        b.upload(
            tok,
            &toks.iter().map(|t| f32::from_bits(*t)).collect::<Vec<_>>(),
        )
        .unwrap();
        b.gather_rows_device_f16(src_t, dst, tok, c, batch)
            .expect("gather_rows");
        let got = b.download(dst).unwrap();
        let mut max_diff = 0.0f32;
        for (e, gr) in expect.iter().zip(got.iter()) {
            max_diff = max_diff.max((e - gr).abs());
        }
        assert!(max_diff < 1e-2, "gather_rows mismatch, max_diff={max_diff}");
        log::info!("gather_rows vs CPU reference OK (batch={batch})");
    }

    /// sample_into_host_seeded_batch 与 CPU argmax 一致性：temperature≈0 时退化为
    /// argmax（确定性），验证每 slot 的 top-1 选择互不干扰。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn sample_batch_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping sample_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let n = 1024usize;
        let batch = 4usize;
        let mut rng = test_rng(0xFEED1234);

        let logits: Vec<f32> = (0..batch * n).map(|_| rng() * 10.0).collect();
        // 每 slot 的期望 argmax（CPU 参考）
        let expect: Vec<usize> = logits
            .chunks(n)
            .map(|chunk| {
                let mut best = 0usize;
                for i in 1..n {
                    if chunk[i] > chunk[best] {
                        best = i;
                    }
                }
                best
            })
            .collect();

        let logits_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let token_t = mk_tensor(&mut b, batch, TensorDtype::F32);
        let temp_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let mask_t = mk_tensor(&mut b, batch * n, TensorDtype::F32);
        let counter_t = mk_tensor(&mut b, batch * n, TensorDtype::U32);
        let sampler_t = mk_tensor(&mut b, batch * 8, TensorDtype::F32);
        let hist_t = mk_tensor(&mut b, batch, TensorDtype::U32);
        b.upload(logits_t, &logits).unwrap();

        // 先跑单序列版 4 次（同一数据/参数），对照 batch 版——隔离"原版 bug vs batch 偏移 bug"。
        let single_tokens: Vec<u32> = (0..batch)
            .map(|bi| {
                let single_logits = mk_tensor(&mut b, n, TensorDtype::F32);
                let single_tok = mk_tensor(&mut b, 1, TensorDtype::F32);
                b.upload(single_logits, &logits[bi * n..(bi + 1) * n])
                    .unwrap();
                b.sample(
                    single_logits,
                    single_tok,
                    n,
                    0.0001,
                    if std::env::var("K1").is_ok() { 1 } else { 50 },
                    1.0,
                    42 + bi as u32,
                    1.0,
                    0.0,
                    0.0,
                    &[],
                )
                .expect("single sample");
                b.download(single_tok).unwrap()[0].to_bits()
            })
            .collect();
        eprintln!("single-seq tokens: {single_tokens:?}");

        // sampler 参数：temperature=0.0001（≈argmax）、top_k=50、top_p=1.0、seed 逐 slot。
        let mut sampler_data = Vec::with_capacity(batch * 8);
        for bi in 0..batch {
            sampler_data.extend_from_slice(&[
                0.0001,
                f32::from_bits(50),
                1.0,
                f32::from_bits(42 + bi as u32),
                1.0,
                0.0,
                0.0,
                f32::from_bits(0u32),
            ]);
        }
        b.upload(sampler_t, &sampler_data).unwrap();

        b.sample_into_host_seeded_batch(
            logits_t, token_t, n, temp_t, mask_t, counter_t, sampler_t, hist_t, batch,
        )
        .expect("sample_batch");
        let got = b.download(token_t).unwrap();
        for bi in 0..batch {
            let tok = got[bi].to_bits();
            assert_eq!(
                tok, expect[bi] as u32,
                "sample_batch slot {bi}: got {tok} expect {}",
                expect[bi]
            );
        }
        log::info!("sample_batch vs CPU argmax OK (batch={batch}, temp≈0)");
    }

    /// record_tokens 与 CPU 参考对比：B slot 各自独立计数追加。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn record_tokens_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping record_tokens test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let batch = 4usize;
        let stride = 8usize;
        let rounds = 3usize;

        // 每 slot 预置计数（验证原子追加起点）。
        let init_cnt: Vec<u32> = vec![1, 0, 5, 2];
        let mut expect_seq = vec![0u32; batch * stride];
        let mut expect_cnt = init_cnt.clone();

        let tok = mk_tensor(&mut b, batch, TensorDtype::F32);
        let seq = mk_tensor(&mut b, batch * stride, TensorDtype::F32);
        let cnt = mk_tensor(&mut b, batch, TensorDtype::F32);
        // cnt 存 u32 位模式（f32 缓冲），初值必须按位写入而非数值转换。
        b.upload(
            cnt,
            &init_cnt
                .iter()
                .map(|c| f32::from_bits(*c))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // 逐轮：round r 每 slot 追加 token = slot*100 + r。
        for r in 0..rounds {
            let toks: Vec<u32> = (0..batch).map(|bi| bi as u32 * 100 + r as u32).collect();
            b.upload(
                tok,
                &toks.iter().map(|t| f32::from_bits(*t)).collect::<Vec<_>>(),
            )
            .unwrap();
            b.record_tokens(tok, seq, cnt, stride, batch)
                .expect("record_tokens");
            for bi in 0..batch {
                let pos = expect_cnt[bi];
                expect_seq[bi * stride + pos as usize] = toks[bi];
                expect_cnt[bi] += 1;
            }
        }

        let got_seq = b.download(seq).unwrap();
        for (e, gr) in expect_seq.iter().zip(got_seq.iter()) {
            assert_eq!(e, &gr.to_bits(), "record_tokens seq mismatch");
        }
        let got_cnt = b.download(cnt).unwrap();
        for (e, gr) in expect_cnt.iter().zip(got_cnt.iter()) {
            assert_eq!(e, &gr.to_bits(), "record_tokens cnt mismatch");
        }
        log::info!("record_tokens vs CPU reference OK (batch={batch})");
    }

    /// gemv_variant_mb（batch 权重复用版）vs 逐 slot 单序列版数值一致性：
    /// int8 relu2 / mul_add / plain 三种 op，batch=6（含 BGRP 分组边界 4 的跨界）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_variant_mb_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_variant_mb test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 64usize; // M 为 4 的倍数（GEMV_ROWS）
        let k = 256usize; // K 为 128 的倍数（int8 group）
        let batch = 6usize; // 跨 BGRP=4 分组边界
        let mut rng = test_rng(0x77AA33CC);

        // int8 量化权重（scale/zero 打包进 sz；idx 打包 4×uint8）。
        let mut w_host = vec![0.0f32; m * k];
        for v in w_host.iter_mut() {
            *v = rng() * 2.0;
        }
        let (idx, sz) = {
            let mut idx = vec![0u32; m * (k / 4)];
            let mut sz = vec![0u32; m * (k / 128)];
            for row in 0..m {
                for g in 0..k / 128 {
                    let scale = 0.01f32;
                    let zero = 0.5f32;
                    let s16 = half::f16::from_f32(scale).to_bits() as u32;
                    let z16 = half::f16::from_f32(zero).to_bits() as u32;
                    sz[row * (k / 128) + g] = (s16 & 0xFFFF) | (z16 << 16);
                    for j in 0..k / 4 {
                        let mut packed = 0u32;
                        for q in 0..4 {
                            let wv = w_host[row * k + j * 4 + q];
                            let qv = ((wv - zero) / scale).round().clamp(-128.0, 127.0) as i32;
                            let qv = (qv as u32) & 0xFF;
                            packed |= qv << (8 * q);
                        }
                        idx[row * (k / 4) + j] = packed;
                    }
                }
            }
            (idx, sz)
        };

        // 激活 [batch, K] + 门控 fp16 + 残差初值。
        let x: Vec<f32> = (0..batch * k).map(|_| rng()).collect();
        let g: Vec<f32> = (0..batch * k).map(|_| 0.5 + 0.5 * rng()).collect();
        let y_init: Vec<f32> = (0..batch * m).map(|_| rng()).collect();

        for op in [0i32, 1i32, 3i32] {
            // 逐 slot 单序列基准（batch=1 原版 kernel，分 slot 调）。
            let mut expect = vec![0.0f32; batch * m];
            for bi in 0..batch {
                let a8 = crate::backend::Int8Handle {
                    idx: mk_tensor(&mut b, m * (k / 4), TensorDtype::U32),
                    sz: mk_tensor(&mut b, m * (k / 128), TensorDtype::U32),
                    m,
                    k,
                };
                let xt = mk_tensor(&mut b, k, TensorDtype::F32);
                let gt = mk_tensor(&mut b, k, TensorDtype::F16);
                let yt = mk_tensor(&mut b, m, TensorDtype::F32);
                b.upload_u32(a8.idx, &idx).unwrap();
                b.upload_u32(a8.sz, &sz).unwrap();
                b.upload(xt, &x[bi * k..(bi + 1) * k]).unwrap();
                b.upload(gt, &g[bi * k..(bi + 1) * k]).unwrap();
                // op==2(add) 用残差初值；其余覆盖写（mb 版同语义，覆盖即可对比）。
                b.upload(yt, &y_init[bi * m..(bi + 1) * m]).unwrap();
                match op {
                    0 => b.gemv_int8_relu2(&a8, xt, yt, m, k, 1).unwrap(),
                    1 => b.gemv_int8_mul_add(&a8, xt, gt, yt, m, k, 1).unwrap(),
                    _ => b.gemv_int8_plain(&a8, xt, yt, m, k, 1).unwrap(),
                }
                let got = b.download(yt).unwrap();
                expect[bi * m..(bi + 1) * m].copy_from_slice(&got);
                // 释放临时张量（避免注册表膨胀）。
                for t in [a8.idx, a8.sz, xt, gt, yt] {
                    b.free_tensor(t);
                }
            }

            // batch mb 版（一次算 6 slot，跨 BGRP=4 分组）。
            let a8 = crate::backend::Int8Handle {
                idx: mk_tensor(&mut b, m * (k / 4), TensorDtype::U32),
                sz: mk_tensor(&mut b, m * (k / 128), TensorDtype::U32),
                m,
                k,
            };
            let xt = mk_tensor(&mut b, batch * k, TensorDtype::F32);
            let gt = mk_tensor(&mut b, batch * k, TensorDtype::F16);
            let yt = mk_tensor(&mut b, batch * m, TensorDtype::F32);
            b.upload_u32(a8.idx, &idx).unwrap();
            b.upload_u32(a8.sz, &sz).unwrap();
            b.upload(xt, &x).unwrap();
            b.upload(gt, &g).unwrap();
            b.upload(yt, &y_init).unwrap();
            match op {
                0 => b.gemv_int8_relu2(&a8, xt, yt, m, k, batch).unwrap(),
                1 => b.gemv_int8_mul_add(&a8, xt, gt, yt, m, k, batch).unwrap(),
                _ => b.gemv_int8_plain(&a8, xt, yt, m, k, batch).unwrap(),
            }
            let got = b.download(yt).unwrap();

            // half2 累加顺序与单序列版一致（per slot 独立），容差收紧到 fp16 级。
            let mut diff = 0.0f32;
            for (e, gv) in expect.iter().zip(got.iter()) {
                diff = diff.max((e - gv).abs());
            }
            let tol = if op == 0 { 5e-2 } else { 2e-1 };
            assert!(
                diff < tol,
                "gemv_variant_mb op={op} mismatch, max_diff={diff} (batch={batch})"
            );
            log::info!("gemv_variant_mb op={op} vs single OK (batch={batch}, max_diff={diff:.5})");
            for t in [a8.idx, a8.sz, xt, gt, yt] {
                b.free_tensor(t);
            }
        }
    }

    /// gemv_int8_rkv_stage1_batch（权重复用版）vs 逐 slot 单序列版数值一致性：
    /// r/k/v + 4 mid 投影，batch=6（跨 BGRP=4 分组边界）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_rkv_stage1_batch_matches_single() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping rkv_stage1_batch test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize;
        let vm = 32usize;
        let wm = 32usize;
        let am = 32usize;
        let gm = 32usize;
        let batch = 6usize;
        let mut rng = test_rng(0x1357BEEF);

        // int8 量化权重制造（与 mb 测试同一打包格式）。
        fn mk_a8(m: usize, k: usize, rng: &mut dyn FnMut() -> f32) -> (Vec<u32>, Vec<u32>) {
            let mut idx = vec![0u32; m * (k / 4)];
            let mut sz = vec![0u32; m * (k / 128)];
            for row in 0..m {
                for g in 0..k / 128 {
                    let scale = 0.01f32;
                    let zero = 0.5f32;
                    let s16 = half::f16::from_f32(scale).to_bits() as u32;
                    let z16 = half::f16::from_f32(zero).to_bits() as u32;
                    sz[row * (k / 128) + g] = (s16 & 0xFFFF) | (z16 << 16);
                    for j in 0..k / 4 {
                        let mut packed = 0u32;
                        for q in 0..4 {
                            let wv = rng() * 2.0;
                            let qv = ((wv - zero) / scale).round().clamp(-128.0, 127.0) as i32;
                            packed |= ((qv as u32) & 0xFF) << (8 * q);
                        }
                        idx[row * (k / 4) + j] = packed;
                    }
                }
            }
            (idx, sz)
        }
        let (r_idx, r_sz) = mk_a8(c, c, &mut rng);
        let (k_idx, k_sz) = mk_a8(c, c, &mut rng);
        let (v_idx, v_sz) = mk_a8(c, c, &mut rng);
        // mid 权重（fp32 [mid, C]）。
        let v1: Vec<f32> = (0..vm * c).map(|_| rng()).collect();
        let w1: Vec<f32> = (0..wm * c).map(|_| rng()).collect();
        let a1: Vec<f32> = (0..am * c).map(|_| rng()).collect();
        let g1: Vec<f32> = (0..gm * c).map(|_| rng()).collect();
        // 激活 [batch, C]。
        let xr: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xk: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xv: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xw: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xa: Vec<f32> = (0..batch * c).map(|_| rng()).collect();
        let xg: Vec<f32> = (0..batch * c).map(|_| rng()).collect();

        // 逐 slot 单序列基准。
        let mut expect_r = vec![0.0f32; batch * c];
        let mut expect_k = vec![0.0f32; batch * c];
        let mut expect_v = vec![0.0f32; batch * c];
        let mut expect_vm = vec![0.0f32; batch * vm];
        let mut expect_wm = vec![0.0f32; batch * wm];
        let mut expect_am = vec![0.0f32; batch * am];
        let mut expect_gm = vec![0.0f32; batch * gm];
        {
            let (r_i, r_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let (k_i, k_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let (v_i, v_s) = (
                mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
                mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
            );
            let rh = crate::backend::Int8Handle {
                idx: r_i,
                sz: r_s,
                m: c,
                k: c,
            };
            let kh = crate::backend::Int8Handle {
                idx: k_i,
                sz: k_s,
                m: c,
                k: c,
            };
            let vh = crate::backend::Int8Handle {
                idx: v_i,
                sz: v_s,
                m: c,
                k: c,
            };
            let (v1t, w1t, a1t, g1t) = (
                mk_tensor(&mut b, vm * c, TensorDtype::F32),
                mk_tensor(&mut b, wm * c, TensorDtype::F32),
                mk_tensor(&mut b, am * c, TensorDtype::F32),
                mk_tensor(&mut b, gm * c, TensorDtype::F32),
            );
            b.upload_u32(rh.idx, &r_idx).unwrap();
            b.upload_u32(rh.sz, &r_sz).unwrap();
            b.upload_u32(kh.idx, &k_idx).unwrap();
            b.upload_u32(kh.sz, &k_sz).unwrap();
            b.upload_u32(vh.idx, &v_idx).unwrap();
            b.upload_u32(vh.sz, &v_sz).unwrap();
            b.upload(v1t, &v1).unwrap();
            b.upload(w1t, &w1).unwrap();
            b.upload(a1t, &a1).unwrap();
            b.upload(g1t, &g1).unwrap();
            for bi in 0..batch {
                let (xrt, xkt, xvt, xwt, xat, xgt) = (
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                );
                let (ort, okt, ovt, ovmt, owmt, oamt, ogmt) = (
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F32),
                    mk_tensor(&mut b, c, TensorDtype::F16),
                    mk_tensor(&mut b, vm, TensorDtype::F32),
                    mk_tensor(&mut b, wm, TensorDtype::F32),
                    mk_tensor(&mut b, am, TensorDtype::F32),
                    mk_tensor(&mut b, gm, TensorDtype::F32),
                );
                b.upload(xrt, &xr[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xkt, &xk[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xvt, &xv[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xwt, &xw[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xat, &xa[bi * c..(bi + 1) * c]).unwrap();
                b.upload(xgt, &xg[bi * c..(bi + 1) * c]).unwrap();
                b.gemv_int8_rkv_stage1(
                    &rh, &kh, &vh, v1t, w1t, a1t, g1t, xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt,
                    ovmt, owmt, oamt, ogmt, c, vm, wm, am, gm,
                )
                .unwrap();
                expect_r[bi * c..(bi + 1) * c].copy_from_slice(&b.download(ort).unwrap());
                expect_k[bi * c..(bi + 1) * c].copy_from_slice(&b.download(okt).unwrap());
                expect_v[bi * c..(bi + 1) * c].copy_from_slice(&b.download(ovt).unwrap());
                expect_vm[bi * vm..(bi + 1) * vm].copy_from_slice(&b.download(ovmt).unwrap());
                expect_wm[bi * wm..(bi + 1) * wm].copy_from_slice(&b.download(owmt).unwrap());
                expect_am[bi * am..(bi + 1) * am].copy_from_slice(&b.download(oamt).unwrap());
                expect_gm[bi * gm..(bi + 1) * gm].copy_from_slice(&b.download(ogmt).unwrap());
                for t in [
                    xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt, ovmt, owmt, oamt, ogmt,
                ] {
                    b.free_tensor(t);
                }
            }
            for t in [
                rh.idx, rh.sz, kh.idx, kh.sz, vh.idx, vh.sz, v1t, w1t, a1t, g1t,
            ] {
                b.free_tensor(t);
            }
        }

        // batch mb 版。
        let (r_i, r_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let (k_i, k_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let (v_i, v_s) = (
            mk_tensor(&mut b, c * (c / 4), TensorDtype::U32),
            mk_tensor(&mut b, c * (c / 128), TensorDtype::U32),
        );
        let rh = crate::backend::Int8Handle {
            idx: r_i,
            sz: r_s,
            m: c,
            k: c,
        };
        let kh = crate::backend::Int8Handle {
            idx: k_i,
            sz: k_s,
            m: c,
            k: c,
        };
        let vh = crate::backend::Int8Handle {
            idx: v_i,
            sz: v_s,
            m: c,
            k: c,
        };
        let (v1t, w1t, a1t, g1t) = (
            mk_tensor(&mut b, vm * c, TensorDtype::F32),
            mk_tensor(&mut b, wm * c, TensorDtype::F32),
            mk_tensor(&mut b, am * c, TensorDtype::F32),
            mk_tensor(&mut b, gm * c, TensorDtype::F32),
        );
        let (xrt, xkt, xvt, xwt, xat, xgt) = (
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
        );
        let (ort, okt, ovt, ovmt, owmt, oamt, ogmt) = (
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F32),
            mk_tensor(&mut b, batch * c, TensorDtype::F16),
            mk_tensor(&mut b, batch * vm, TensorDtype::F32),
            mk_tensor(&mut b, batch * wm, TensorDtype::F32),
            mk_tensor(&mut b, batch * am, TensorDtype::F32),
            mk_tensor(&mut b, batch * gm, TensorDtype::F32),
        );
        b.upload_u32(rh.idx, &r_idx).unwrap();
        b.upload_u32(rh.sz, &r_sz).unwrap();
        b.upload_u32(kh.idx, &k_idx).unwrap();
        b.upload_u32(kh.sz, &k_sz).unwrap();
        b.upload_u32(vh.idx, &v_idx).unwrap();
        b.upload_u32(vh.sz, &v_sz).unwrap();
        b.upload(v1t, &v1).unwrap();
        b.upload(w1t, &w1).unwrap();
        b.upload(a1t, &a1).unwrap();
        b.upload(g1t, &g1).unwrap();
        b.upload(xrt, &xr).unwrap();
        b.upload(xkt, &xk).unwrap();
        b.upload(xvt, &xv).unwrap();
        b.upload(xwt, &xw).unwrap();
        b.upload(xat, &xa).unwrap();
        b.upload(xgt, &xg).unwrap();
        b.gemv_int8_rkv_stage1_batch(
            &rh, &kh, &vh, v1t, w1t, a1t, g1t, xrt, xkt, xvt, xwt, xat, xgt, ort, okt, ovt, ovmt,
            owmt, oamt, ogmt, c, vm, wm, am, gm, batch,
        )
        .unwrap();

        let got_r = b.download(ort).unwrap();
        let got_k = b.download(okt).unwrap();
        let got_v = b.download(ovt).unwrap();
        let got_vm = b.download(ovmt).unwrap();
        let got_wm = b.download(owmt).unwrap();
        let got_am = b.download(oamt).unwrap();
        let got_gm = b.download(ogmt).unwrap();

        for (name, e, g, tol) in [
            ("r", &expect_r, &got_r, 2e-1f32),
            ("k", &expect_k, &got_k, 2e-1),
            ("v", &expect_v, &got_v, 2e-1),
            ("vm", &expect_vm, &got_vm, 1e-2),
            ("wm", &expect_wm, &got_wm, 1e-2),
            ("am", &expect_am, &got_am, 1e-2),
            ("gm", &expect_gm, &got_gm, 1e-2),
        ] {
            let mut diff = 0.0f32;
            for (ev, gv) in e.iter().zip(g.iter()) {
                diff = diff.max((ev - gv).abs());
            }
            assert!(
                diff < tol,
                "rkv_stage1_batch {name} mismatch, max_diff={diff} (batch={batch})"
            );
            log::info!("rkv_stage1_batch {name} vs single OK (max_diff={diff:.5})");
        }
    }

    /// segmean 与 CPU 参考对比：x [batch, t_pad, c] 每 slot 前 lens[b] 行均值。
    /// 含 lens < t_pad（padding 段必须不计入）与 batch=1 边界。无 CUDA 设备时跳过。
    #[test]
    fn segmean_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping segmean test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 512usize;
        let t_pad = 8usize;
        let batch = 5usize;
        let lens: Vec<u32> = vec![8, 1, 5, 3, 8]; // 覆盖全 pad / 全长 / 中间档
        let mut rng = test_rng(0x5EED5EED);
        let x: Vec<f32> = (0..batch * t_pad * c).map(|_| rng()).collect();

        // CPU 参考（只对前 lens[b] 行求均值）。
        let mut expect = vec![vec![0.0f32; c]; batch];
        for bi in 0..batch {
            let l = lens[bi] as usize;
            for t in 0..l {
                for i in 0..c {
                    expect[bi][i] += x[(bi * t_pad + t) * c + i];
                }
            }
            for v in expect[bi].iter_mut() {
                *v /= l as f32;
            }
        }

        let xt = mk_tensor(&mut b, batch * t_pad * c, TensorDtype::F32);
        let ot = mk_tensor(&mut b, batch * c, TensorDtype::F32);
        let lt = mk_tensor(&mut b, batch, TensorDtype::U32);
        b.upload(xt, &x).unwrap();
        b.upload_u32(lt, &lens).unwrap();
        b.segmean(xt, ot, lt, c, t_pad, batch).unwrap();
        let got = b.download(ot).unwrap();
        let mut max_diff = 0.0f32;
        for bi in 0..batch {
            for i in 0..c {
                max_diff = max_diff.max((expect[bi][i] - got[bi * c + i]).abs());
            }
        }
        assert!(max_diff < 1e-4, "segmean mismatch, max_diff={max_diff}");
        log::info!("segmean vs CPU reference OK (batch={batch}, max_diff={max_diff:.7})");
    }

    /// fuse_ka_dplr_norm 与 CPU 参考对比：fuse_ka + dplr(S 更新) + group_norm + sum_rk_rk。
    /// 无 CUDA 设备时跳过。batch=1（单 token 路径）。
    #[test]
    fn fuse_ka_dplr_norm_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping fuse_ka_dplr_norm test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let h = 3usize;
        let n = 64usize;
        let eps = 1e-12f32;
        let gn_eps = 1e-6f32;

        let mut seed = 0xABCDEF01u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let s_size = h * n * n;
        let k_size = h * n;
        let sd: Vec<f32> = (0..s_size).map(|_| rng() * 0.1).collect();
        let kd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let kkd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let ad: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let kad: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let rd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let vd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let wd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let gd: Vec<f32> = (0..k_size).map(|_| 0.5 + rng()).collect();
        let bd: Vec<f32> = (0..k_size).map(|_| rng()).collect();
        let rkd: Vec<f32> = (0..k_size).map(|_| rng()).collect();

        // CPU 参考（batch=1，逐 head）
        let mut expect_s = sd.clone();
        let mut expect_kmod = vec![0.0f32; k_size];
        let mut expect_yn = vec![0.0f32; k_size];
        for head in 0..h {
            let v_b = head * n;
            let w_b = head * n;
            let s_b = head * n * n;
            // L2 范数（与 shader/CUDA 相同的全块 2 倍归约）
            let mut sq_sum = 0.0f32;
            for row in 0..n {
                let kk = kd[v_b + row] * kkd[w_b + row];
                sq_sum += kk * kk;
            }
            sq_sum *= 2.0; // 128 线程冗余归约（每行 ct=0/1 各算一次）
            let inv_norm = 1.0 / sq_sum.sqrt().max(eps);

            // Phase 1: 按列 kk_l2 / b / k_mod / w / r
            let mut sh_a = vec![0.0f32; n];
            let mut sh_b = vec![0.0f32; n];
            let mut sh_k = vec![0.0f32; n];
            let mut sh_w = vec![0.0f32; n];
            let mut sh_r = vec![0.0f32; n];
            for j in 0..n {
                let kc = kd[v_b + j];
                let kkc = kc * kkd[w_b + j];
                let ac = f16::from_f32(ad[v_b + j]).to_f32();
                let kl2 = kkc * inv_norm;
                sh_a[j] = kl2;
                sh_b[j] = -kl2 * ac;
                sh_k[j] = kc * (1.0 + kad[w_b + j] * (ac - 1.0));
                sh_w[j] = f16::from_f32(wd[v_b + j]).to_f32();
                sh_r[j] = rd[v_b + j];
            }
            expect_kmod[v_b..v_b + n].copy_from_slice(&sh_k[..n]);

            // Phase 2: sa[row] = sum_j S[row,j] * sh_a[j]
            let mut sa_val = vec![0.0f32; n];
            for row in 0..n {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += expect_s[s_b + row * n + j] * sh_a[j];
                }
                sa_val[row] = acc;
            }

            // Phase 3: S 更新 + y[row] = S@r
            let mut yv = vec![0.0f32; n];
            for row in 0..n {
                let vi = f16::from_f32(vd[v_b + row]).to_f32();
                let sv = sa_val[row];
                let mut yp = 0.0f32;
                for j in 0..n {
                    let s_ij = expect_s[s_b + row * n + j];
                    let new_s = s_ij * sh_w[j] + sv * sh_b[j] + vi * sh_k[j];
                    expect_s[s_b + row * n + j] = new_s;
                    yp += new_s * sh_r[j];
                }
                yv[row] = yp;
            }

            // Phase 4+5: group-norm(y) + s = sum(r*k_mod*r_k)
            let mean: f32 = yv.iter().sum::<f32>() / n as f32;
            let var: f32 = yv.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv_std = 1.0 / (var + gn_eps).sqrt();
            let mut s_acc = 0.0f32;
            for row in 0..n {
                s_acc += sh_r[row] * sh_k[row] * rkd[w_b + row];
            }

            // Phase 6: y_norm
            for row in 0..n {
                let vi = f16::from_f32(vd[v_b + row]).to_f32();
                let normalized = (yv[row] - mean) * inv_std * gd[w_b + row] + bd[w_b + row];
                expect_yn[v_b + row] = normalized + s_acc * vi;
            }
        }

        let s = mk_tensor(&mut b, s_size, TensorDtype::F32);
        let k = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let kk = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let a = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let ka = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let r = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let v = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let w = mk_tensor(&mut b, k_size, TensorDtype::F16);
        let g = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let bb = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let rk = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let km = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let y = mk_tensor(&mut b, k_size, TensorDtype::F32);
        let yn = mk_tensor(&mut b, k_size, TensorDtype::F32);
        b.upload(s, &sd).unwrap();
        b.upload(k, &kd).unwrap();
        b.upload(kk, &kkd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(ka, &kad).unwrap();
        b.upload(r, &rd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(w, &wd).unwrap();
        b.upload(g, &gd).unwrap();
        b.upload(bb, &bd).unwrap();
        b.upload(rk, &rkd).unwrap();
        b.fuse_ka_dplr_norm(
            s, k, kk, a, ka, r, v, w, g, bb, rk, km, y, yn, h, n, eps, gn_eps,
        )
        .expect("fuse_ka_dplr_norm");
        let got_s = b.download(s).unwrap();
        let got_km = b.download(km).unwrap();
        let got_yn = b.download(yn).unwrap();

        let mut s_diff = 0.0f32;
        for (e, gr) in expect_s.iter().zip(got_s.iter()) {
            s_diff = s_diff.max((e - gr).abs());
        }
        let mut km_diff = 0.0f32;
        for (e, gr) in expect_kmod.iter().zip(got_km.iter()) {
            km_diff = km_diff.max((e - gr).abs());
        }
        let mut yn_diff = 0.0f32;
        for (e, gr) in expect_yn.iter().zip(got_yn.iter()) {
            yn_diff = yn_diff.max((e - gr).abs());
        }
        assert!(s_diff < 1e-2, "fuse_ka s mismatch, max_diff={s_diff}");
        assert!(km_diff < 1e-3, "fuse_ka k_mod mismatch, max_diff={km_diff}");
        assert!(
            yn_diff < 1e-2,
            "fuse_ka y_norm mismatch, max_diff={yn_diff}"
        );
        log::info!("fuse_ka_dplr_norm vs CPU reference OK (s/km/yn max_diff<1e-2)");
    }

    /// 生成 int8 量化权重：`w[m,k] = scale[m,k/128] * idx_byte + zero[m,k/128]`。
    /// 返回 `(idx 打包 uint32 [m,k/4], sz [m,k/128], 参考 w [m*k])`。
    /// 每行 scale/zero 随机；CPU 参考按 fp16 舍入后的 scale/zero 反量化以对齐 GPU。
    fn make_int8_weights(
        m: usize,
        k: usize,
        rng: &mut impl FnMut() -> f32,
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let kv = k / 4;
        let kg = k / 128;
        let mut idx = vec![0u32; m * kv];
        let mut w = vec![0.0f32; m * k];
        // 每行一份 fp16 舍入的 scale/zero。
        // 真实 int8 量化：scale = range/128（byte∈[0,255]），反量化权重与原始 fp16 同值域。
        // 测试用 rng()~[-1,1] 直接作 scale 会得到 ±128 的权重，超出 fp16 累加精度，故缩小 128 倍。
        let mut scale = vec![0.0f32; m];
        let mut zero = vec![0.0f32; m];
        for s in scale.iter_mut() {
            *s = f16::from_f32(rng() / 128.0).to_f32();
        }
        for z in zero.iter_mut() {
            *z = f16::from_f32(rng() / 128.0).to_f32();
        }
        for mm in 0..m {
            for kk in 0..k {
                let byte = (rng() * 256.0) as u32 % 256;
                idx[mm * kv + kk / 4] |= byte << ((kk % 4) * 8);
                w[mm * k + kk] = scale[mm] * (byte as f32) + zero[mm];
            }
        }
        let mut sz = vec![0u32; m * kg];
        for mm in 0..m {
            let sc = f16::from_f32(scale[mm]);
            let zr = f16::from_f32(zero[mm]);
            let pack = (sc.to_bits() as u32) | ((zr.to_bits() as u32) << 16);
            for g in 0..kg {
                sz[mm * kg + g] = pack;
            }
        }
        (idx, sz, w)
    }

    /// gemv_int8_rkv_stage1 与 CPU 参考对比：int8 量化 r/k/v 投影 + mid fp32 投影。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_rkv_stage1_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_int8_rkv_stage1 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let c = 256usize; // 整除 128(group)/4(打包)/4(ROWS)
        let vm = 2usize;
        let wm = 3usize;
        let am = 4usize;
        let gm = 5usize;

        let mut seed = 0x55667788u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // r/k/v int8 权重
        let (r_idx, r_sz, r_w) = make_int8_weights(c, c, &mut rng);
        let (k_idx, k_sz, k_w) = make_int8_weights(c, c, &mut rng);
        let (v_idx, v_sz, v_w) = make_int8_weights(c, c, &mut rng);
        // mid 权重（fp32）
        let v1d: Vec<f32> = (0..vm * c).map(|_| rng()).collect();
        let w1d: Vec<f32> = (0..wm * c).map(|_| rng()).collect();
        let a1d: Vec<f32> = (0..am * c).map(|_| rng()).collect();
        let g1d: Vec<f32> = (0..gm * c).map(|_| rng()).collect();
        // 输入
        let xr: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xk: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xv: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xw: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xa: Vec<f32> = (0..c).map(|_| rng()).collect();
        let xg: Vec<f32> = (0..c).map(|_| rng()).collect();

        // CPU 参考
        let dot = |w: &[f32], x: &[f32], m: usize, k: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; m];
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += w[mm * k + kk] * x[kk];
                }
                y[mm] = acc;
            }
            y
        };
        let expect_r = dot(&r_w, &xr, c, c);
        let expect_k = dot(&k_w, &xk, c, c);
        let expect_v = dot(&v_w, &xv, c, c);
        // out_v 为 fp16 张量：GPU 输出 = f16 舍入后的 CPU 参考，比较前对齐精度。
        let expect_v: Vec<f32> = expect_v
            .iter()
            .map(|&x| f16::from_f32(x).to_f32())
            .collect();
        let mut expect_vm = vec![0.0f32; vm];
        let mut expect_wm = vec![0.0f32; wm];
        let mut expect_am = vec![0.0f32; am];
        let mut expect_gm = vec![0.0f32; gm];
        for i in 0..vm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += v1d[i * c + kk] * xv[kk];
            }
            expect_vm[i] = acc;
        }
        for i in 0..wm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += w1d[i * c + kk] * xw[kk];
            }
            expect_wm[i] = acc.tanh();
        }
        for i in 0..am {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += a1d[i * c + kk] * xa[kk];
            }
            expect_am[i] = acc;
        }
        for i in 0..gm {
            let mut acc = 0.0f32;
            for kk in 0..c {
                acc += g1d[i * c + kk] * xg[kk];
            }
            expect_gm[i] = acc;
        }

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32], m: usize, k: usize| {
            let it = b
                .create_tensor(idx.len(), TensorDtype::U32)
                .expect("create");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("create");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let rh = make_handle(&mut b, &r_idx, &r_sz, c, c);
        let kh = make_handle(&mut b, &k_idx, &k_sz, c, c);
        let vh = make_handle(&mut b, &v_idx, &v_sz, c, c);

        let v1 = mk_tensor(&mut b, vm * c, TensorDtype::F32);
        let w1 = mk_tensor(&mut b, wm * c, TensorDtype::F32);
        let a1 = mk_tensor(&mut b, am * c, TensorDtype::F32);
        let g1 = mk_tensor(&mut b, gm * c, TensorDtype::F32);
        let xr_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xk_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xv_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xw_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xa_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let xg_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let or_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let ok_t = mk_tensor(&mut b, c, TensorDtype::F32);
        let ov_t = mk_tensor(&mut b, c, TensorDtype::F16);
        let ovm_t = mk_tensor(&mut b, vm, TensorDtype::F32);
        let owm_t = mk_tensor(&mut b, wm, TensorDtype::F32);
        let oam_t = mk_tensor(&mut b, am, TensorDtype::F32);
        let ogm_t = mk_tensor(&mut b, gm, TensorDtype::F32);
        b.upload(v1, &v1d).unwrap();
        b.upload(w1, &w1d).unwrap();
        b.upload(a1, &a1d).unwrap();
        b.upload(g1, &g1d).unwrap();
        b.upload(xr_t, &xr).unwrap();
        b.upload(xk_t, &xk).unwrap();
        b.upload(xv_t, &xv).unwrap();
        b.upload(xw_t, &xw).unwrap();
        b.upload(xa_t, &xa).unwrap();
        b.upload(xg_t, &xg).unwrap();

        b.gemv_int8_rkv_stage1(
            &rh, &kh, &vh, v1, w1, a1, g1, xr_t, xk_t, xv_t, xw_t, xa_t, xg_t, or_t, ok_t, ov_t,
            ovm_t, owm_t, oam_t, ogm_t, c, vm, wm, am, gm,
        )
        .expect("gemv_int8_rkv_stage1");

        let got_r = b.download(or_t).unwrap();
        let got_k = b.download(ok_t).unwrap();
        let got_v = b.download(ov_t).unwrap();
        let got_vm = b.download(ovm_t).unwrap();
        let got_wm = b.download(owm_t).unwrap();
        let got_am = b.download(oam_t).unwrap();
        let got_gm = b.download(ogm_t).unwrap();

        let maxd = |a: &[f32], bv: &[f32]| -> f32 {
            a.iter()
                .zip(bv.iter())
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        let dr = maxd(&expect_r, &got_r);
        let dk = maxd(&expect_k, &got_k);
        let dv = maxd(&expect_v, &got_v);
        let dvm = maxd(&expect_vm, &got_vm);
        let dwm = maxd(&expect_wm, &got_wm);
        let dam = maxd(&expect_am, &got_am);
        let dgm = maxd(&expect_gm, &got_gm);
        assert!(dr < 1e-2, "int8 r mismatch, max_diff={dr}");
        assert!(dk < 1e-2, "int8 k mismatch, max_diff={dk}");
        assert!(dv < 1e-2, "int8 v mismatch, max_diff={dv}");
        assert!(dvm < 1e-3, "int8 vm mismatch, max_diff={dvm}");
        assert!(dwm < 1e-3, "int8 wm mismatch, max_diff={dwm}");
        assert!(dam < 1e-3, "int8 am mismatch, max_diff={dam}");
        assert!(dgm < 1e-3, "int8 gm mismatch, max_diff={dgm}");
        log::info!("gemv_int8_rkv_stage1 vs CPU OK (r/k/v<1e-2, vm/wm/am/gm<1e-3)");
    }

    /// gemv_variant int8（wtype=2）路径与 CPU 参考对比：覆盖 relu2/mul_add/add 三个 op。
    /// int8 idx [M,K/4]（每 uint32 4 字节）+ sz [M,K/128]（scale/zero 各 half）。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_variant_int8_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_variant int8 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k, batch) = (8usize, 256usize, 2usize);

        let mut seed = 0xA11CE8u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let (idx, sz, _w) = make_int8_weights(m, k, &mut rng);
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();
        let g: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（int8 反量化权重）：w[m,k] = scale[m,k/128]*byte + zero[m,..]
        let kv = k / 4;
        let kg = k / 128;
        let dequant = |mm: usize, kk: usize| -> f32 {
            let byte = (idx[mm * kv + kk / 4] >> ((kk % 4) * 8)) & 0xFF;
            let sc = half::f16::from_bits((sz[mm * kg + kk / 128] & 0xFFFF) as u16).to_f32();
            let zr = half::f16::from_bits((sz[mm * kg + kk / 128] >> 16) as u16).to_f32();
            sc * (byte as f32) + zr
        };
        let relu2 = |v: f32| if v > 0.0 { v * v } else { 0.0 };
        let gemv = |op_sel: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; m * batch];
            for bb in 0..batch {
                for mm in 0..m {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        let wv = dequant(mm, kk);
                        let gv = if op_sel == 1 {
                            half::f16::from_f32(g[bb * k + kk]).to_f32()
                        } else {
                            1.0
                        };
                        acc += wv * x[bb * k + kk] * gv;
                    }
                    y[bb * m + mm] = if op_sel == 0 { relu2(acc) } else { acc };
                }
            }
            y
        };
        let expect_relu2 = gemv(0);
        let expect_mul = gemv(1);
        let expect_add = gemv(2);

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("c");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("c");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let h = make_handle(&mut b, &idx, &sz);
        let xt = mk_tensor(&mut b, k * batch, TensorDtype::F32);
        let gt = mk_tensor(&mut b, k * batch, TensorDtype::F16);
        let yt = mk_tensor(&mut b, m * batch, TensorDtype::F32);
        b.upload(xt, &x).unwrap();
        let g16: Vec<f32> = g.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        b.upload(gt, &g16).unwrap();

        // relu2 (op=0)
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_relu2(&h, xt, yt, m, k, batch).unwrap();
        let got_r2 = b.download(yt).unwrap();
        let md_r2 = got_r2
            .iter()
            .zip(&expect_relu2)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_r2 < 2e-2, "int8 relu2 mismatch max_diff={md_r2}");

        // mul_add (op=1)：y 累加式
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_mul_add(&h, xt, gt, yt, m, k, batch).unwrap();
        let got_mul = b.download(yt).unwrap();
        let md_mul = got_mul
            .iter()
            .zip(&expect_mul)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_mul < 2e-2, "int8 mul_add mismatch max_diff={md_mul}");

        // add (op=2)：y 累加式
        b.upload(yt, &vec![0.0f32; m * batch]).unwrap();
        b.gemv_int8_add(&h, xt, yt, m, k, batch).unwrap();
        let got_add = b.download(yt).unwrap();
        let md_add = got_add
            .iter()
            .zip(&expect_add)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md_add < 2e-2, "int8 add mismatch max_diff={md_add}");

        log::info!(
            "gemv_variant int8 vs CPU OK (relu2={md_r2:.2e}, mul_add={md_mul:.2e}, add={md_add:.2e})"
        );
    }

    /// gemv_int8_plain（op=3，覆盖写）与 CPU 参考对比：y[m] = Σ_k x[k]·w[m,k]。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_int8_plain_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_int8_plain test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k, batch) = (8usize, 256usize, 2usize);

        let mut seed = 0x219F3DA7u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let (idx, sz, _w) = make_int8_weights(m, k, &mut rng);
        let x: Vec<f32> = (0..k * batch).map(|_| rng()).collect();

        // CPU 参考（int8 反量化权重）：w[m,k] = scale[m,k/128]*byte + zero[m,..]
        let kv = k / 4;
        let kg = k / 128;
        let dequant = |mm: usize, kk: usize| -> f32 {
            let byte = (idx[mm * kv + kk / 4] >> ((kk % 4) * 8)) & 0xFF;
            let sc = half::f16::from_bits((sz[mm * kg + kk / 128] & 0xFFFF) as u16).to_f32();
            let zr = half::f16::from_bits((sz[mm * kg + kk / 128] >> 16) as u16).to_f32();
            sc * (byte as f32) + zr
        };
        let mut expect = vec![0.0f32; m * batch];
        for bb in 0..batch {
            for mm in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += dequant(mm, kk) * x[bb * k + kk];
                }
                expect[bb * m + mm] = acc;
            }
        }

        let make_handle = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("c");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("c");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let h = make_handle(&mut b, &idx, &sz);
        let xt = mk_tensor(&mut b, k * batch, TensorDtype::F32);
        let yt = mk_tensor(&mut b, m * batch, TensorDtype::F32);
        b.upload(xt, &x).unwrap();
        // 覆盖写：y 预填脏数据，验证 op=3 确实覆盖而非累加。
        b.upload(yt, &vec![123.0f32; m * batch]).unwrap();
        b.gemv_int8_plain(&h, xt, yt, m, k, batch).unwrap();
        let got = b.download(yt).unwrap();
        let md = got
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md < 2e-2, "int8 plain mismatch max_diff={md}");
        log::info!("gemv_int8_plain vs CPU OK (max_diff={md:.2e})");
    }

    /// ffn_value_sparse_add 与 CPU 参考对比：x += r2 @ value，r2（relu²）约 96% 稀疏。
    /// 反量化 int8 后的 fp16 平铺权重走同一内核，故此测试覆盖 int8 稀疏 FFN 路径。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn ffn_value_sparse_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping ffn_value_sparse test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, fh) = (512usize, 256usize);
        const TILE: usize = 128;
        const C_TILE: usize = 256;

        let mut seed = 0x51A2C4E6u32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // value: [c, fh] 行主序（解码 gemv 按 [C, fh]）。
        let value: Vec<f32> = (0..c * fh).map(|_| rng()).collect();
        // r2: [fh]，只保留 ~6% 非零（模拟 relu² 稀疏；内核按非零列只读）。
        let mut r2 = vec![0.0f32; fh];
        for v in r2.iter_mut() {
            if rng() > 0.94 {
                *v = rng().abs();
            }
        }
        // 初始 x（含残差）与 CPU 参考。
        let mut x = vec![0.0f32; c];
        let mut expect = vec![0.0f32; c];
        for cc in 0..c {
            let v = rng();
            x[cc] = v;
            expect[cc] = v;
        }
        for f in 0..fh {
            if r2[f] != 0.0 {
                for cc in 0..c {
                    expect[cc] += r2[f] * value[cc * fh + f];
                }
            }
        }
        // 构建平铺布局（与 gpu_model::load_ffn_value_tiled 一致）。
        let c_blocks = c / C_TILE;
        let mut tiled = vec![0.0f32; fh * c];
        for f in 0..fh {
            let f_block = f / TILE;
            let f_local = f % TILE;
            for cc in 0..c {
                let c_block = cc / C_TILE;
                let c_local = cc % C_TILE;
                tiled[((f_block * c_blocks + c_block) * TILE) * C_TILE
                    + f_local * C_TILE
                    + c_local] = value[cc * fh + f];
            }
        }
        let vt = b.create_tensor(tiled.len(), TensorDtype::F16).unwrap();
        b.upload(vt, &tiled).unwrap();
        let rt = b.create_tensor(r2.len(), TensorDtype::F32).unwrap();
        let xt = b.create_tensor(x.len(), TensorDtype::F32).unwrap();
        b.upload(rt, &r2).unwrap();
        b.upload(xt, &x).unwrap();
        // int8 无稠密 fp16，value_w16 传 None；CUDA 内核忽略该参数。
        b.ffn_value_sparse_add(None, vt, rt, xt, c, fh).unwrap();
        let got = b.download(xt).unwrap();
        let md = got
            .iter()
            .zip(&expect)
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(md < 2e-2, "ffn_value_sparse mismatch max_diff={md}");
        log::info!("ffn_value_sparse vs CPU OK (max_diff={md:.2e})");
    }

    /// gemv_lowrank_chain4 与 CPU 参考对比：融合 4 条低秩链第二级。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gemv_lowrank_chain4_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_lowrank_chain4 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let m = 64usize;
        let kw = 96usize;
        let ka = 96usize;
        let kv = 64usize;
        let kg = 128usize;

        let mut seed = 0x90abcdefu32;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let w2: Vec<f32> = (0..m * kw).map(|_| rng()).collect();
        let a2: Vec<f32> = (0..m * ka).map(|_| rng()).collect();
        let v2: Vec<f32> = (0..m * kv).map(|_| rng()).collect();
        let g2: Vec<f32> = (0..m * kg).map(|_| rng()).collect();
        let wm: Vec<f32> = (0..kw).map(|_| rng()).collect();
        let am: Vec<f32> = (0..ka).map(|_| rng()).collect();
        let vm: Vec<f32> = (0..kv).map(|_| rng()).collect();
        let gm: Vec<f32> = (0..kg).map(|_| rng()).collect();
        let w0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let a0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let v0: Vec<f32> = (0..m).map(|_| rng()).collect();
        let scale = vec![rng()];
        let vf: Vec<f32> = (0..m).map(|_| rng()).collect();
        let ov_init: Vec<f32> = (0..m).map(|_| rng()).collect();

        // CPU 参考（sigmoid = 1/(1+exp(-x))）
        let sig = |x: f32| 1.0f32 / (1.0 + (-x).exp());
        let mut ew = vec![0.0f32; m];
        let mut ea = vec![0.0f32; m];
        let mut ev = vec![0.0f32; m];
        let mut eg = vec![0.0f32; m];
        for r in 0..m {
            let mut lw = 0.0f32;
            let mut la = 0.0f32;
            let mut lv = 0.0f32;
            let mut lg = 0.0f32;
            for k in 0..kw {
                lw += wm[k] * w2[r * kw + k];
            }
            for k in 0..ka {
                la += am[k] * a2[r * ka + k];
            }
            for k in 0..kv {
                lv += vm[k] * v2[r * kv + k];
            }
            for k in 0..kg {
                lg += sig(gm[k]) * g2[r * kg + k];
            }
            ew[r] = (scale[0] * sig(lw + w0[r])).exp();
            ea[r] = sig(la + a0[r]);
            ev[r] = ov_init[r] + sig(lv + v0[r]) * (vf[r] - ov_init[r]);
            eg[r] = lg;
        }

        let w2t = mk_tensor(&mut b, m * kw, TensorDtype::F32);
        let a2t = mk_tensor(&mut b, m * ka, TensorDtype::F32);
        let v2t = mk_tensor(&mut b, m * kv, TensorDtype::F32);
        let g2t = mk_tensor(&mut b, m * kg, TensorDtype::F32);
        let wmt = mk_tensor(&mut b, kw, TensorDtype::F32);
        let amt = mk_tensor(&mut b, ka, TensorDtype::F32);
        let vmt = mk_tensor(&mut b, kv, TensorDtype::F32);
        let gmt = mk_tensor(&mut b, kg, TensorDtype::F32);
        let w0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let a0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let v0t = mk_tensor(&mut b, m, TensorDtype::F32);
        let st = mk_tensor(&mut b, 1, TensorDtype::F32);
        let vft = mk_tensor(&mut b, m, TensorDtype::F16);
        let ow = mk_tensor(&mut b, m, TensorDtype::F16);
        let oa = mk_tensor(&mut b, m, TensorDtype::F16);
        let ov = mk_tensor(&mut b, m, TensorDtype::F16);
        let og = mk_tensor(&mut b, m, TensorDtype::F16);
        b.upload(w2t, &w2).unwrap();
        b.upload(a2t, &a2).unwrap();
        b.upload(v2t, &v2).unwrap();
        b.upload(g2t, &g2).unwrap();
        b.upload(wmt, &wm).unwrap();
        b.upload(amt, &am).unwrap();
        b.upload(vmt, &vm).unwrap();
        b.upload(gmt, &gm).unwrap();
        b.upload(w0t, &w0).unwrap();
        b.upload(a0t, &a0).unwrap();
        b.upload(v0t, &v0).unwrap();
        b.upload(st, &scale).unwrap();
        b.upload(vft, &vf).unwrap();
        b.upload(ow, &ov_init).unwrap();
        b.upload(oa, &ev).unwrap();
        b.upload(ov, &ov_init).unwrap();
        b.upload(og, &ev).unwrap();

        b.gemv_lowrank_chain4(
            w2t, a2t, v2t, g2t, wmt, amt, vmt, gmt, w0t, a0t, v0t, st, vft, ow, oa, ov, og, m, kw,
            ka, kv, kg,
        )
        .expect("gemv_lowrank_chain4");

        let gw = b.download(ow).unwrap();
        let ga = b.download(oa).unwrap();
        let gv = b.download(ov).unwrap();
        let gg = b.download(og).unwrap();
        let maxd = |a: &[f32], bv: &[f32]| -> f32 {
            a.iter()
                .zip(bv.iter())
                .fold(0.0f32, |mm, (x, y)| mm.max((x - y).abs()))
        };
        let dw = maxd(&ew, &gw);
        let da = maxd(&ea, &ga);
        let dv = maxd(&ev, &gv);
        let dg = maxd(&eg, &gg);
        assert!(dw < 1e-2, "lowrank w mismatch, max_diff={dw}");
        assert!(da < 1e-2, "lowrank a mismatch, max_diff={da}");
        assert!(dv < 1e-2, "lowrank v mismatch, max_diff={dv}");
        assert!(dg < 1e-2, "lowrank g mismatch, max_diff={dg}");
        log::info!("gemv_lowrank_chain4 vs CPU OK (<1e-2)");
    }

    /// argmax 与 CPU 参考对比：logits [N] 的最大值索引（f32 位模式存 token[0]）。
    /// 覆盖平局取小索引的语义。无 CUDA 设备时跳过。
    #[test]
    fn argmax_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping argmax test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // 用例 1：唯一最大值。
        let n = 65536usize;
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, c)| a.partial_cmp(c).unwrap().then(ib.cmp(ia)))
            .map(|(i, _)| i)
            .unwrap();

        let lt = b.create_tensor(n, TensorDtype::F32).expect("create logits");
        let tok_t = b.create_tensor(1, TensorDtype::F32).expect("create token");
        b.upload(lt, &logits).unwrap();
        b.argmax(lt, tok_t, n).expect("argmax");
        let got = b.download(tok_t).unwrap();
        let got_idx = f32::to_bits(got[0]) as usize;
        assert_eq!(
            got_idx, max_idx,
            "argmax unique max mismatch: got {got_idx}, expect {max_idx}"
        );

        // 用例 2：平局取更小索引。构造两个相同最大值。
        let n2 = 1024usize;
        let mut logits2 = vec![0.0f32; n2];
        logits2[0] = 5.0;
        logits2[999] = 5.0;
        let lt2 = b
            .create_tensor(n2, TensorDtype::F32)
            .expect("create logits2");
        let tok2 = b.create_tensor(1, TensorDtype::F32).expect("create token2");
        b.upload(lt2, &logits2).unwrap();
        b.argmax(lt2, tok2, n2).expect("argmax tie");
        let got2 = b.download(tok2).unwrap();
        let got_idx2 = f32::to_bits(got2[0]) as usize;
        assert_eq!(
            got_idx2, 0,
            "argmax tie mismatch: got {got_idx2}, expect 0 (smaller index)"
        );
        log::info!("argmax vs CPU OK (unique={max_idx}, tie=0)");
    }

    /// sample 确定性验证：构造一个 logits 使某 token 概率≈1，采样必返回该 token。
    /// 覆盖空 history（无惩罚）、temperature/top-k/top-p/seed 路径。无 CUDA 设备时跳过。
    #[test]
    fn sample_deterministic() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping sample_deterministic test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let sample_tok = |b: &mut CudaBackend,
                          logits: &[f32],
                          temp: f32,
                          top_k: u32,
                          top_p: f32,
                          seed: u32,
                          freq: f32,
                          history: &[u32]|
         -> usize {
            let lt = b
                .create_tensor(logits.len(), TensorDtype::F32)
                .expect("create logits");
            let tok = b.create_tensor(1, TensorDtype::F32).expect("create token");
            b.upload(lt, logits).unwrap();
            b.sample(
                lt,
                tok,
                logits.len(),
                temp,
                top_k,
                top_p,
                seed,
                1.0,
                freq,
                0.0,
                history,
            )
            .expect("sample");
            let got = b.download(tok).unwrap();
            f32::to_bits(got[0]) as usize
        };

        // 用例 1：logits[5] 独占极大值 → 任何 seed/参数下必返回 5。
        let n = 4096usize;
        let mut logits = vec![0.0f32; n];
        logits[5] = 100.0;
        for seed in [1u32, 42u32, 0xdeadbeefu32] {
            assert_eq!(
                sample_tok(&mut b, &logits, 1.0, 0, 1.0, seed, 0.0, &[]),
                5,
                "sample dominant-token mismatch (seed={seed})"
            );
        }

        // 用例 2：top_k=1 强制取全局最大（即使有多个相近值）。
        let mut logits2 = vec![1.0f32; n];
        logits2[7] = 2.0;
        assert_eq!(sample_tok(&mut b, &logits2, 0.5, 1, 1.0, 7, 0.0, &[]), 7);

        // 用例 3：frequency 惩罚路径。token0/token1 等大，history 含 token0 多次 →
        // freq=1.0 使 token0 logit -= 出现次数，必选 token1。其余 token 设极低 logit，
        // 避免零 logit 的大量 token 在 softmax 中淹没 token1。
        let mut logits3 = vec![-100.0f32; n];
        logits3[0] = 5.0;
        logits3[1] = 5.0;
        let history = vec![0u32, 0, 0, 0, 0, 0]; // token0 出现 6 次
        let got = sample_tok(&mut b, &logits3, 1.0, 0, 1.0, 99, 1.0, &history);
        assert_eq!(
            got, 1,
            "sample frequency-penalty mismatch: got {got}, expect 1"
        );
        log::info!("sample_deterministic OK (dominant=5, topk=7, freq-penalty=1)");
    }

    /// record_token 与 store_token_host 验证：连续写入 token 索引，序列缓冲按序累积。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn record_token_accumulates() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping record_token test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let n = 8usize;
        let in_tok = b.create_tensor(1, TensorDtype::F32).expect("create in_tok");
        let out_seq = b
            .create_tensor(n, TensorDtype::F32)
            .expect("create out_seq");
        let cnt = b.create_tensor(1, TensorDtype::F32).expect("create cnt");
        b.upload(out_seq, &vec![0.0; n]).unwrap();
        b.upload(cnt, &[0.0; 1]).unwrap();

        let tokens = [7u32, 3u32, 5u32, 1u32];
        for tk in tokens {
            b.store_token_host(in_tok, tk).expect("store_token_host");
            b.record_token(in_tok, out_seq, cnt).expect("record_token");
        }

        // 按位解释序列缓冲为 u32，验证顺序累积。
        let seq = b.download(out_seq).unwrap();
        let got: Vec<u32> = seq.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            &got[..tokens.len()],
            &tokens,
            "record_token sequence mismatch"
        );
        let cnt_got = b.download(cnt).unwrap();
        assert_eq!(cnt_got[0].to_bits(), tokens.len() as u32, "cnt mismatch");
        log::info!("record_token vs store_token_host OK (seq={tokens:?})");
    }

    /// gather_row_device_f16 与 CPU 参考对比：从 fp16 表 src[VOCAB,C] 按 token 索引取一行转 f32。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn gather_row_device_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gather_row_device_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let vocab = 64usize;
        let c = 512usize;
        // 构造 fp16 表（f32 值，上传时经 f16 舍入）。
        let table_f32: Vec<f32> = (0..vocab * c).map(|i| ((i as f32) * 0.13).sin()).collect();
        let src = b
            .create_tensor(vocab * c, TensorDtype::F16)
            .expect("create src");
        b.upload(src, &table_f32).unwrap();

        // 取 token 行：idx=42。
        let token = 42u32;
        let tok = b.create_tensor(1, TensorDtype::F32).expect("create tok");
        b.store_token_host(tok, token).unwrap();
        let dst = b.create_tensor(c, TensorDtype::F32).expect("create dst");
        b.gather_row_device_f16(src, dst, tok, c)
            .expect("gather_row");

        let got = b.download(dst).unwrap();
        // 参考：f16 舍入后的该行。
        let ref_row: Vec<f32> = table_f32[token as usize * c..(token as usize + 1) * c].to_vec();
        let mut max_diff = 0.0f32;
        for (a, g) in ref_row.iter().zip(got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert!(
            max_diff < 1e-3,
            "gather_row_device_f16 mismatch: max_diff={max_diff}"
        );
        log::info!("gather_row_device_f16 vs CPU OK (token={token}, max_diff={max_diff})");
    }

    /// copy_device_f16 与 CPU 参考对比：f16 设备到设备全量拷贝。
    /// 无 CUDA 设备时跳过。
    #[test]
    fn copy_device_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping copy_device_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        let n = 1024usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).cos() * 3.0).collect();
        let src = b.create_tensor(n, TensorDtype::F16).expect("create src");
        b.upload(src, &data).unwrap();
        let dst = b.create_tensor(n, TensorDtype::F16).expect("create dst");
        b.copy_device_f16(src, dst).expect("copy_device_f16");

        let src_got = b.download(src).unwrap();
        let dst_got = b.download(dst).unwrap();
        let mut max_diff = 0.0f32;
        for (a, g) in src_got.iter().zip(dst_got.iter()) {
            max_diff = max_diff.max((a - g).abs());
        }
        assert_eq!(
            max_diff, 0.0,
            "copy_device_f16 mismatch: max_diff={max_diff}"
        );
        log::info!("copy_device_f16 OK (max_diff=0.0, n={n})");
    }

    /// 统一的 gemm 参考实现：C[i*n+j] = sum_kk A[i*k+kk]*B[j*k+kk] + op。
    fn ref_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, op: i32) -> Vec<f32> {
        // 输入先经 f16 舍入（kernel 以 fp16 读取）。
        let f16r = |v: f32| {
            let h = f16::from_f32(v);
            h.to_f32()
        };
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += f16r(a[i * k + kk]) * f16r(b[j * k + kk]);
                }
                c[i * n + j] = match op {
                    1 => acc + 1.0, // bias 常量（测试上传 bias 全 1）
                    2 => acc + 1.0, // x 常量
                    3 => {
                        if acc > 0.0 {
                            acc * acc
                        } else {
                            0.0
                        }
                    }
                    4 => acc.tanh(),
                    _ => acc,
                };
            }
        }
        c
    }

    /// 诊断：直接用 cuBLAS 驱动做一次最小 fp16 gemm（C = A@B^T），
    /// 隔离 cuBLAS 库/驱动/参数问题（700 异步非法地址排查）。
    #[test]
    fn cublas_direct_gemm_probe() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping cublas probe");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let drv = match cublas_driver() {
            Some(d) => d,
            None => {
                log::info!("cublas driver unavailable");
                return;
            }
        };
        let m = 8usize;
        let n = 6usize;
        let k = 16usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let wt: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.07).cos()).collect();
        let ad = b.create_tensor(m * k, TensorDtype::F16).unwrap();
        let bd = b.create_tensor(n * k, TensorDtype::F16).unwrap();
        let cd = b.create_tensor(m * n, TensorDtype::F32).unwrap();
        b.upload(ad, &a).unwrap();
        b.upload(bd, &wt).unwrap();
        let (ad_, bd_, cd_) = (
            b.f16_ptr(ad, "probe").unwrap(),
            b.f16_ptr(bd, "probe").unwrap(),
            b.f32_ptr(cd, "probe").unwrap(),
        );
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (m_i, n_i, k_i) = (n as c_int, m as c_int, k as c_int);
        let (lda, ldb, ldc) = (k as c_int, k as c_int, n as c_int);
        let r = unsafe {
            (drv.cublas_gemm_ex)(
                b.cublas.unwrap(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m_i,
                n_i,
                k_i,
                &alpha as *const f32 as *const c_void,
                bd_ as *const c_void,
                CUDA_R_16F,
                lda,
                ad_ as *const c_void,
                CUDA_R_16F,
                ldb,
                &beta as *const f32 as *const c_void,
                cd_ as *mut c_void,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        };
        log::info!("[CUBLAS_PROBE] gemm_ex status={r}");
        let got = b.download(cd).unwrap();
        let exp = ref_gemm(&a, &wt, m, n, k, 0);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        log::info!("[CUBLAS_PROBE] max_diff={md}");
    }

    /// gemm 系列（plain/bias/add/relu2/tanh）与 CPU 参考对比。
    #[test]
    fn gemm_variants_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemm_variants test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, n, k) = (8usize, 6usize, 16usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.13).sin()).collect();
        let bb: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.07).cos()).collect();
        let at = b.create_tensor(m * k, TensorDtype::F16).expect("a");
        let bt = b.create_tensor(n * k, TensorDtype::F16).expect("b");
        b.upload(at, &a).unwrap();
        b.upload(bt, &bb).unwrap();

        // plain
        let ct = b.create_tensor(m * n, TensorDtype::F32).expect("c");
        b.gemm(at, bt, ct, m, n, k).expect("gemm");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 0);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm mismatch max_diff={md}");

        // bias
        let bias_t = b.create_tensor(n, TensorDtype::F32).expect("bias");
        b.upload(bias_t, &vec![1.0f32; n]).unwrap();
        b.gemm_bias(at, bt, bias_t, ct, m, n, k).expect("gemm_bias");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 1);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_bias mismatch max_diff={md}");

        // add（x 全 1）
        let xt = b.create_tensor(m * n, TensorDtype::F32).expect("x");
        b.upload(xt, &vec![1.0f32; m * n]).unwrap();
        b.gemm_add(at, bt, xt, ct, m, n, k).expect("gemm_add");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 2);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_add mismatch max_diff={md}");

        // relu2
        b.gemm_relu2(at, bt, ct, m, n, k).expect("gemm_relu2");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 3);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_relu2 mismatch max_diff={md}");

        // tanh
        b.gemm_tanh(at, bt, ct, m, n, k).expect("gemm_tanh");
        let got = b.download(ct).unwrap();
        let exp = ref_gemm(&a, &bb, m, n, k, 4);
        let md = got
            .iter()
            .zip(&exp)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-3, "gemm_tanh mismatch max_diff={md}");
        log::info!("gemm variants (plain/bias/add/relu2/tanh) vs CPU OK");
    }

    /// copy_device（f32）与 copy_token 与 CPU 参考对比。
    #[test]
    fn copy_device_token_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping copy_device_token test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");

        // copy_device：全量 f32 拷贝。
        let n = 512usize;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).cos()).collect();
        let src = b.create_tensor(n, TensorDtype::F32).expect("src");
        b.upload(src, &data).unwrap();
        let dst = b.create_tensor(n, TensorDtype::F32).expect("dst");
        b.copy_device(src, dst).expect("copy_device");
        let got = b.download(dst).unwrap();
        let md = got
            .iter()
            .zip(&data)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert_eq!(md, 0.0, "copy_device mismatch max_diff={md}");

        // copy_token：取 x 的第 token 行。
        let (c, t) = (16usize, 8usize);
        let x: Vec<f32> = (0..t * c).map(|i| (i as f32) * 0.5).collect();
        let xt = b.create_tensor(t * c, TensorDtype::F32).expect("x");
        b.upload(xt, &x).unwrap();
        let token = 3usize;
        let yt = b.create_tensor(c, TensorDtype::F32).expect("y");
        b.copy_token(xt, yt, c, c, token).expect("copy_token");
        let got = b.download(yt).unwrap();
        for i in 0..c {
            assert!(
                (got[i] - x[token * c + i]).abs() < 1e-6,
                "copy_token mismatch at {i}"
            );
        }
        log::info!("copy_device & copy_token vs CPU OK");
    }

    /// to_f16 / to_f16_triple 与 CPU 参考对比。
    #[test]
    fn to_f16_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping to_f16 test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, t, m_pad) = (8usize, 4usize, 6usize); // m_pad > t，验证填充行写 0
        let x: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.9).sin()).collect();
        let xt = b.create_tensor(t * c, TensorDtype::F32).expect("x");
        b.upload(xt, &x).unwrap();
        let yt = b.create_tensor(m_pad * c, TensorDtype::F16).expect("y");
        b.to_f16(xt, yt, c, t, m_pad, c, c).expect("to_f16");
        let got = b.download(yt).unwrap();
        for tok in 0..m_pad {
            for i in 0..c {
                let expect = if tok < t {
                    f16::from_f32(x[tok * c + i]).to_f32()
                } else {
                    0.0
                };
                assert!(
                    (got[tok * c + i] - expect).abs() < 1e-6,
                    "to_f16 mismatch at tok={tok} i={i}"
                );
            }
        }

        // to_f16_triple：三输入一次转换。
        let (xr_t, xk_t, xv_t) = (
            b.create_tensor(t * c, TensorDtype::F32).expect("xr"),
            b.create_tensor(t * c, TensorDtype::F32).expect("xk"),
            b.create_tensor(t * c, TensorDtype::F32).expect("xv"),
        );
        let xk: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        let xv: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).tan()).collect();
        b.upload(xr_t, &x).unwrap();
        b.upload(xk_t, &xk).unwrap();
        b.upload(xv_t, &xv).unwrap();
        let (yr_t, yk_t, yv_t) = (
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yr"),
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yk"),
            b.create_tensor(m_pad * c, TensorDtype::F16).expect("yv"),
        );
        b.to_f16_triple(xr_t, xk_t, xv_t, yr_t, yk_t, yv_t, c, t, m_pad, c, c)
            .expect("to_f16_triple");
        let gr = b.download(yr_t).unwrap();
        let gk = b.download(yk_t).unwrap();
        let gv = b.download(yv_t).unwrap();
        for tok in 0..m_pad {
            for i in 0..c {
                let (er, ek, ev) = if tok < t {
                    (
                        f16::from_f32(x[tok * c + i]).to_f32(),
                        f16::from_f32(xk[tok * c + i]).to_f32(),
                        f16::from_f32(xv[tok * c + i]).to_f32(),
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };
                assert!((gr[tok * c + i] - er).abs() < 1e-6, "triple r mismatch");
                assert!((gk[tok * c + i] - ek).abs() < 1e-6, "triple k mismatch");
                assert!((gv[tok * c + i] - ev).abs() < 1e-6, "triple v mismatch");
            }
        }
        log::info!("to_f16 & to_f16_triple vs CPU OK");
    }

    /// elementwise 系列（sigmoid/inplace/scale_exp/mul）与 CPU 参考对比。
    #[test]
    fn elementwise_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping elementwise test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, batch) = (64usize, 3usize);
        let a: Vec<f32> = (0..c * batch).map(|i| ((i as f32) * 0.11).sin()).collect();
        let bb: Vec<f32> = (0..c * batch).map(|i| ((i as f32) * 0.05) + 0.5).collect();
        let at = b.create_tensor(c * batch, TensorDtype::F32).expect("a");
        let bt = b.create_tensor(c * batch, TensorDtype::F32).expect("b");
        let yt = b.create_tensor(c * batch, TensorDtype::F32).expect("y");
        b.upload(at, &a).unwrap();
        b.upload(bt, &bb).unwrap();

        b.elementwise_sigmoid(at, yt, c, batch).expect("sigmoid");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = 1.0 / (1.0 + (-a[i]).exp());
            assert!((got[i] - e).abs() < 1e-5, "sigmoid mismatch at {i}");
        }

        b.elementwise_sigmoid_inplace(yt, c, batch)
            .expect("sigmoid_inplace");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = 1.0 / (1.0 + (-(1.0 / (1.0 + (-a[i]).exp()))).exp());
            assert!((got[i] - e).abs() < 1e-5, "sigmoid_inplace mismatch at {i}");
        }

        b.elementwise_scale_exp(at, bt, yt, c, batch)
            .expect("scale_exp");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            // kernel 语义：y = exp(a * b[0])，b 为全局共享标量（与 Vulkan elementwise.comp OP9 一致）。
            let e = (a[i] * bb[0]).exp();
            assert!((got[i] - e).abs() < 1e-4, "scale_exp mismatch at {i}");
        }

        b.elementwise_mul(at, bt, yt, c, batch).expect("mul");
        let got = b.download(yt).unwrap();
        for i in 0..c * batch {
            let e = a[i] * bb[i];
            assert!((got[i] - e).abs() < 1e-6, "mul mismatch at {i}");
        }
        log::info!("elementwise (sigmoid/inplace/scale_exp/mul) vs CPU OK");
    }

    /// fuse_ka 与 sum_rk_rk 与 CPU 参考对比。
    #[test]
    fn fuse_ka_sum_rk_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping fuse_ka_sum_rk test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (h, n, batch) = (2usize, 8usize, 3usize);
        let hn = h * n;

        let k = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kk = mk_tensor(&mut b, hn, TensorDtype::F32);
        let a = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let ka = mk_tensor(&mut b, hn, TensorDtype::F32);
        let k_mod = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kk_l2 = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let bb = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let kd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.3).cos()).collect();
        let kkd: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.1) + 0.5).collect();
        let ad: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.2).sin()).collect();
        let kad: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.4) + 1.0).collect();
        b.upload(k, &kd).unwrap();
        b.upload(kk, &kkd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(ka, &kad).unwrap();
        b.fuse_ka(k, kk, a, ka, k_mod, kk_l2, bb, h, n, batch)
            .expect("fuse_ka");

        // 参考：对每个 (bidx, head) 计算。
        let mut exp_km = vec![0.0f32; batch * hn];
        let mut exp_kl = vec![0.0f32; batch * hn];
        let mut exp_b = vec![0.0f32; batch * hn];
        for bidx in 0..batch {
            for head in 0..h {
                let base = bidx * hn + head * n;
                let wbase = head * n;
                let mut sq = 0.0f32;
                for j in 0..n {
                    let kkv = kd[base + j] * kkd[wbase + j];
                    sq += kkv * kkv;
                }
                let inv = 1.0 / sq.sqrt().max(1e-12);
                for j in 0..n {
                    let kv_ = kd[base + j];
                    let kkv = kv_ * kkd[wbase + j];
                    let k_l2 = kkv * inv;
                    let av = ad[base + j];
                    exp_km[base + j] = kv_ * (1.0 + kad[wbase + j] * (av - 1.0));
                    exp_kl[base + j] = k_l2;
                    exp_b[base + j] = -k_l2 * av;
                }
            }
        }
        let got_km = b.download(k_mod).unwrap();
        let got_kl = b.download(kk_l2).unwrap();
        let got_b = b.download(bb).unwrap();
        let md = |g: &[f32], e: &[f32]| {
            g.iter()
                .zip(e)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert!(md(&got_km, &exp_km) < 1e-5, "fuse_ka km mismatch");
        assert!(md(&got_kl, &exp_kl) < 1e-5, "fuse_ka kl mismatch");
        assert!(md(&got_b, &exp_b) < 1e-5, "fuse_ka b mismatch");

        // sum_rk_rk：y[b,h*n+j] += sum_j r*km*rk * v。
        let r = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let rk = mk_tensor(&mut b, hn, TensorDtype::F32);
        let v = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * hn, TensorDtype::F32);
        let rd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.6).cos()).collect();
        let rkd: Vec<f32> = (0..hn).map(|i| ((i as f32) * 0.7) + 0.1).collect();
        let vd: Vec<f32> = (0..batch * hn).map(|i| ((i as f32) * 0.8).sin()).collect();
        let yd: Vec<f32> = (0..batch * hn).map(|i| (i as f32) * 0.01).collect();
        b.upload(r, &rd).unwrap();
        b.upload(rk, &rkd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(y, &yd).unwrap();
        b.sum_rk_rk(r, k_mod, rk, v, y, h, n, batch)
            .expect("sum_rk_rk");
        let got = b.download(y).unwrap();
        let mut exp_y = yd.clone();
        for bidx in 0..batch {
            for head in 0..h {
                let base = bidx * hn + head * n;
                let mut s = 0.0f32;
                for j in 0..n {
                    s += rd[base + j] * exp_km[base + j] * rkd[head * n + j];
                }
                for j in 0..n {
                    exp_y[base + j] += s * vd[base + j];
                }
            }
        }
        let md = got
            .iter()
            .zip(&exp_y)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-5, "sum_rk_rk mismatch max_diff={md}");
        log::info!("fuse_ka & sum_rk_rk vs CPU OK");
    }

    /// seq_shift 与 v_first_lerp 与 CPU 参考对比。
    #[test]
    fn seq_shift_v_first_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping seq_shift_v_first test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (c, t) = (16usize, 8usize);

        let x = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let s = mk_tensor(&mut b, c, TensorDtype::F32);
        let tm = mk_tensor(&mut b, c, TensorDtype::F32);
        let y = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let xd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).cos()).collect();
        let sd: Vec<f32> = (0..c).map(|i| ((i as f32) * 0.5) + 1.0).collect();
        let tmd: Vec<f32> = (0..c).map(|i| ((i as f32) * 0.1) + 0.3).collect();
        b.upload(x, &xd).unwrap();
        b.upload(s, &sd).unwrap();
        b.upload(tm, &tmd).unwrap();
        b.seq_shift(x, s, tm, y, c, t, c, c).expect("seq_shift");
        let got = b.download(y).unwrap();
        for tok in 0..t {
            for i in 0..c {
                let cur = xd[tok * c + i];
                let prev = if tok == 0 {
                    sd[i]
                } else {
                    xd[(tok - 1) * c + i]
                };
                let e = cur + tmd[i] * (prev - cur);
                assert!((got[tok * c + i] - e).abs() < 1e-5, "seq_shift mismatch");
            }
        }

        // v_first_lerp：v = v + gate*(v_first - v)（原地 v）。
        let g = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let vf = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let gd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.3).sin()).collect();
        let vfd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        b.upload(g, &gd).unwrap();
        b.upload(vf, &vfd).unwrap();
        b.v_first_lerp(x, g, vf, c, t, c).expect("v_first_lerp");
        let got = b.download(x).unwrap();
        for i in 0..t * c {
            let e = xd[i] + gd[i] * (vfd[i] - xd[i]);
            assert!((got[i] - e).abs() < 1e-5, "v_first_lerp mismatch at {i}");
        }
        log::info!("seq_shift & v_first_lerp vs CPU OK");
    }

    /// dplr_seq 与 CPU 参考对比（n<=64，逐线程独立状态）。
    /// 注：kernel 按 RWKV-7 的 N=64 设计（half-warp 16 线程 × 4 列），故用 n=64 测试。
    #[test]
    fn dplr_seq_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping dplr_seq test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (h, n, t) = (2usize, 64usize, 6usize);
        let c = h * n;

        let s = mk_tensor(&mut b, h * n * n, TensorDtype::F32);
        let r = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let w = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let k = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let v = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let a = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let bb = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let y = mk_tensor(&mut b, t * c, TensorDtype::F32);
        let sd: Vec<f32> = (0..h * n * n).map(|i| ((i as f32) * 0.1).cos()).collect();
        let rd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.2).sin()).collect();
        let wd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.3) + 0.5).collect();
        let kd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.4).cos()).collect();
        let vd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.5).sin()).collect();
        let ad: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.6) + 1.0).collect();
        let bd: Vec<f32> = (0..t * c).map(|i| ((i as f32) * 0.7).cos()).collect();
        b.upload(s, &sd).unwrap();
        b.upload(r, &rd).unwrap();
        b.upload(w, &wd).unwrap();
        b.upload(k, &kd).unwrap();
        b.upload(v, &vd).unwrap();
        b.upload(a, &ad).unwrap();
        b.upload(bb, &bd).unwrap();
        b.dplr_seq(s, r, w, k, v, a, bb, y, h, n, t, c)
            .expect("dplr_seq");
        let got_y = b.download(y).unwrap();
        let got_s = b.download(s).unwrap();

        let mut exp_y = vec![0.0f32; t * c];
        let mut exp_s = sd.clone();
        for head in 0..h {
            // 每线程 i 独立 sreg。
            let mut sreg = vec![vec![0.0f32; n]; n];
            for i in 0..n {
                for j in 0..n {
                    sreg[i][j] = sd[head * n * n + i * n + j];
                }
            }
            for tt in 0..t {
                for i in 0..n {
                    let vv = vd[head * n + i + tt * c];
                    let mut sa = 0.0f32;
                    for j in 0..n {
                        sa += ad[head * n + j + tt * c] * sreg[i][j];
                    }
                    let mut yv = 0.0f32;
                    for j in 0..n {
                        sreg[i][j] = sreg[i][j] * wd[head * n + j + tt * c]
                            + kd[head * n + j + tt * c] * vv
                            + sa * bd[head * n + j + tt * c];
                        yv += sreg[i][j] * rd[head * n + j + tt * c];
                    }
                    exp_y[head * n + i + tt * c] = yv;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    exp_s[head * n * n + i * n + j] = sreg[i][j];
                }
            }
        }
        // 测试数据（wd/ad 递增）导致递归状态数值指数爆炸（可达 1e8），
        // 绝对容差不适用，改为相对容差（kernel 与参考逐位一致）。
        let rel = |got: &[f32], exp: &[f32]| -> f32 {
            let md = got
                .iter()
                .zip(exp)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            let scale = exp.iter().fold(0.0f32, |m, y| m.max(y.abs())).max(1.0);
            md / scale
        };
        let rel_y = rel(&got_y, &exp_y);
        let rel_s = rel(&got_s, &exp_s);
        assert!(rel_y < 1e-4, "dplr_seq y relative mismatch={rel_y}");
        assert!(rel_s < 1e-4, "dplr_seq s relative mismatch={rel_s}");
        log::info!("dplr_seq vs CPU OK (rel_y={rel_y} rel_s={rel_s})");
    }

    /// gemv_seq 与 CPU 参考对比。
    #[test]
    fn gemv_seq_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping gemv_seq test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k, batch) = (6usize, 16usize, 3usize);
        let x_stride = k;
        let y_stride = m;

        let a = mk_tensor(&mut b, m * k, TensorDtype::F32);
        let x = mk_tensor(&mut b, batch * x_stride, TensorDtype::F32);
        let y = mk_tensor(&mut b, batch * y_stride, TensorDtype::F32);
        let ad_: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.3).sin()).collect();
        let xd: Vec<f32> = (0..batch * k).map(|i| ((i as f32) * 0.4).cos()).collect();
        b.upload(a, &ad_).unwrap();
        b.upload(x, &xd).unwrap();
        b.gemv_seq(a, x, y, m, k, x_stride, y_stride, batch)
            .expect("gemv_seq");
        let got = b.download(y).unwrap();
        for bidx in 0..batch {
            for row in 0..m {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += ad_[row * k + kk] * xd[bidx * k + kk];
                }
                assert!(
                    (got[bidx * y_stride + row] - acc).abs() < 1e-5,
                    "gemv_seq mismatch bidx={bidx} row={row}"
                );
            }
        }
        log::info!("gemv_seq vs CPU OK");
    }

    /// dequant_int8_to_f16 与 CPU 参考对比。
    #[test]
    fn dequant_matches_cpu() {
        if !cuda_available() {
            log::info!("CUDA unavailable, skipping dequant test");
            return;
        }
        let mut b = CudaBackend::new().expect("create cuda backend");
        let (m, k) = (4usize, 128usize); // k 为 128 倍数，满足 int8 分组
        let make_int8 = |b: &mut CudaBackend, idx: &[u32], sz: &[u32]| {
            let it = b.create_tensor(idx.len(), TensorDtype::U32).expect("idx");
            let st = b.create_tensor(sz.len(), TensorDtype::U32).expect("sz");
            b.upload_u32(it, idx).unwrap();
            b.upload_u32(st, sz).unwrap();
            Int8Handle {
                idx: it,
                sz: st,
                m,
                k,
            }
        };
        let kg = k / 128;
        let sz: Vec<u32> = (0..m * kg)
            .map(|i| {
                let scale = ((i as f32) * 0.05) + 1.0;
                let zero = ((i as f32) * 0.01) + 0.1;
                scale.to_bits().wrapping_add(zero.to_bits())
            })
            .collect();
        // 参考反量化。
        let sz_scale = |i: usize| -> (f32, f32) {
            let s = sz[i];
            let scale = half::f16::from_bits((s & 0xFFFF) as u16).to_f32();
            let zero = half::f16::from_bits((s >> 16) as u16).to_f32();
            (scale, zero)
        };

        // int8：每个 uint32 装 4 个字节。
        let kv8 = k / 4;
        let mut i_idx = vec![0u32; m * kv8];
        for (i, v) in i_idx.iter_mut().enumerate() {
            *v = ((i as u32) & 0xFF)
                | (((i as u32) & 0xFF) << 8)
                | (((i as u32) & 0xFF) << 16)
                | (((i as u32) & 0xFF) << 24);
        }
        let hi = make_int8(&mut b, &i_idx, &sz);
        let out8 = b.create_tensor(m * k, TensorDtype::F16).expect("out8");
        b.dequant_int8_to_f16(&hi, out8, m, k)
            .expect("dequant_int8");
        let got = b.download(out8).unwrap();
        let mut exp8 = vec![0.0f32; m * k];
        for mm in 0..m {
            for kk in 0..kv8 {
                let ipack = i_idx[mm * kv8 + kk];
                let g = kk / 32;
                let (sc, zr) = sz_scale(mm * kg + g);
                for j in 0..4 {
                    let byte = (ipack >> (8 * j)) & 0xFF;
                    let wv = sc * (byte as f32) + zr;
                    exp8[mm * k + kk * 4 + j] = half::f16::from_f32(wv).to_f32();
                }
            }
        }
        let md = got
            .iter()
            .zip(&exp8)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(md < 1e-5, "dequant_int8 mismatch max_diff={md}");
        log::info!("dequant_int8_to_f16 vs CPU OK");
    }
}
