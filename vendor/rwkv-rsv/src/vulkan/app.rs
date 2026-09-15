#![allow(unsafe_op_in_unsafe_fn)]

use std::{
    borrow::Cow,
    collections::HashSet,
    error::Error,
    marker::PhantomData,
    pin::Pin,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use derive_more::{Deref, Display};
use itertools::Itertools;
use thiserror::Error;

use super::{layout, num};
use vulkanalia::{
    loader::{LIBRARY, LibloadingLoader},
    prelude::v1_4::*,
    vk::{
        KhrCooperativeMatrixExtensionInstanceCommands,
        KhrExternalMemoryWin32ExtensionDeviceCommands,
    },
};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const INSTANCE_EXTENSIONS: [vk::Extension; 2] = [
    vk::KHR_PORTABILITY_ENUMERATION_EXTENSION,
    vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION,
];
const DEVICE_EXTENSIONS: [vk::Extension; 4] = [
    vk::KHR_COOPERATIVE_MATRIX_EXTENSION,
    vk::KHR_EXTERNAL_MEMORY_EXTENSION,
    vk::KHR_EXTERNAL_MEMORY_WIN32_EXTENSION,
    vk::KHR_DEDICATED_ALLOCATION_EXTENSION,
];

const DESCRIPTOR_COUNT_UNIFORM_BUFFER: usize = 2048;
const DESCRIPTOR_COUNT_SAMPLER: usize = 4;
const DESCRIPTOR_COUNT_STORAGE_IMAGE: usize = 1;
const DESCRIPTOR_POOL_MAX_SETS: usize = 2048;
pub const QUERY_POOL_SIZE: usize = 4096;

#[allow(clippy::get_first)]
fn parse_version(version: &str) -> u32 {
    let parts: Vec<&str> = version.split('.').collect();
    let major = parts.get(0).and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    vk::make_version(major, minor, patch)
}

/// Format property: bytes per block.
pub fn vk_format_bpb(format: vk::Format) -> usize {
    match format {
        vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_USCALED
        | vk::Format::R8_SSCALED
        | vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8_SRGB => 1,
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_USCALED
        | vk::Format::R8G8_SSCALED
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_SRGB => 2,
        vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_USCALED
        | vk::Format::R8G8B8_SSCALED
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8_SRGB => 3,
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_USCALED
        | vk::Format::R8G8B8A8_SSCALED
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_USCALED
        | vk::Format::B8G8R8A8_SSCALED
        | vk::Format::B8G8R8A8_UINT
        | vk::Format::B8G8R8A8_SINT
        | vk::Format::B8G8R8A8_SRGB => 4,
        vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_USCALED
        | vk::Format::R16_SSCALED
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT => 2,
        vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_USCALED
        | vk::Format::R16G16_SSCALED
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT => 4,
        vk::Format::R16G16B16_UNORM
        | vk::Format::R16G16B16_SNORM
        | vk::Format::R16G16B16_USCALED
        | vk::Format::R16G16B16_SSCALED
        | vk::Format::R16G16B16_UINT
        | vk::Format::R16G16B16_SINT
        | vk::Format::R16G16B16_SFLOAT => 6,
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_USCALED
        | vk::Format::R16G16B16A16_SSCALED
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK => 8,
        vk::Format::BC2_UNORM_BLOCK | vk::Format::BC2_SRGB_BLOCK => 16,
        vk::Format::BC3_UNORM_BLOCK | vk::Format::BC3_SRGB_BLOCK => 16,
        vk::Format::BC4_UNORM_BLOCK | vk::Format::BC4_SNORM_BLOCK => 8,
        vk::Format::BC5_UNORM_BLOCK | vk::Format::BC5_SNORM_BLOCK => 16,
        vk::Format::BC6H_UFLOAT_BLOCK | vk::Format::BC6H_SFLOAT_BLOCK => 16,
        vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => 16,
        _ => unimplemented!("unsupported format"),
    }
}

/// Format property: pixels per block.
pub fn vk_format_ppb(format: vk::Format) -> [usize; 3] {
    match format {
        // Uncompressed formats have a block size of 1 (1x1 pixel block)
        vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_USCALED
        | vk::Format::R8_SSCALED
        | vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8_SRGB
        | vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_USCALED
        | vk::Format::R8G8_SSCALED
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_SRGB
        | vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_USCALED
        | vk::Format::R8G8B8_SSCALED
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8_SRGB
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_USCALED
        | vk::Format::R8G8B8A8_SSCALED
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_USCALED
        | vk::Format::R16_SSCALED
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_USCALED
        | vk::Format::R16G16_SSCALED
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16B16_UNORM
        | vk::Format::R16G16B16_SNORM
        | vk::Format::R16G16B16_USCALED
        | vk::Format::R16G16B16_SSCALED
        | vk::Format::R16G16B16_UINT
        | vk::Format::R16G16B16_SINT
        | vk::Format::R16G16B16_SFLOAT
        | vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_USCALED
        | vk::Format::R16G16B16A16_SSCALED
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_USCALED
        | vk::Format::B8G8R8A8_SSCALED
        | vk::Format::B8G8R8A8_UINT
        | vk::Format::B8G8R8A8_SINT
        | vk::Format::B8G8R8A8_SRGB => [1, 1, 1],
        vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK
        | vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK => [4, 4, 1],
        _ => unimplemented!("unsupported format"),
    }
}

#[derive(Debug, Clone)]
pub struct Submit {
    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub family: u32,
}

impl Submit {
    /// Creates a new command pool for the specified queue family.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It calls Vulkan API functions which may have undefined behavior if used incorrectly
    /// - The device must be valid and properly initialized
    /// - The queue family index must be valid for the device
    ///
    /// Callers must ensure:
    /// - The device is a valid Vulkan device
    /// - The queue family index exists on the device
    /// - The queue index is valid for the queue family
    ///
    /// # Arguments
    ///
    /// - `device`: The Vulkan device to create the command pool for
    /// - `family`: The queue family index for the command pool
    /// - `queue`: The queue index within the queue family
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the command pool is successfully created, or an error if:
    /// - The device is invalid
    /// - The queue family index is invalid
    /// - Command pool creation fails
    pub unsafe fn new(device: &Device, family: u32, queue: u32) -> Result<Self, vk::ErrorCode> {
        let queue = device.get_device_queue(family, queue);
        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family);
        let pool = device.create_command_pool(&info, None)?;
        Ok(Self {
            queue,
            pool,
            family,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Properties {
    pub memory: properties::Memory,
    pub cooperative_matrix: Vec<properties::CooperativeMatrix>,
    pub vk13: vk::PhysicalDeviceVulkan13Properties,
    pub limits: vk::PhysicalDeviceLimits,
    /// 设备实际 subgroup 大小（NVIDIA=32，AMD=64，Intel 可变），来自 PhysicalDeviceSubgroupProperties。
    pub subgroup_size: u32,
    /// 设备厂商 ID（NVIDIA=0x10DE，AMD=0x1002，Intel=0x8086），用于硬件特定优化。
    pub vendor_id: u32,
    /// 设备是否支持 `VK_EXT_shader_atomic_float` 的 buffer 级 fp32 原子累加。
    /// 决定稀疏 FFN 稀疏内核是否可用（不支持则回退稠密 gemv）。
    pub atomic_float: bool,
}

impl std::fmt::Display for Properties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "properties:")?;
        writeln!(f, "cooperative matrix:")?;
        for p in &self.cooperative_matrix {
            writeln!(f, "\t{p}")?;
        }
        Ok(())
    }
}

pub mod properties {
    use super::{Display, vk};

    #[derive(Debug, Clone)]
    pub struct Memory {
        pub types: Vec<vk::MemoryType>,
        pub heaps: Vec<vk::MemoryHeap>,
    }

    impl Memory {
        pub fn find(&self, bits: u32, properties: vk::MemoryPropertyFlags) -> Option<usize> {
            self.types
                .iter()
                .enumerate()
                .filter(|&(index, _)| (1 << index as u32) & bits != 0)
                .find(|(_, r#type)| r#type.property_flags.contains(properties))
                .map(|(index, _)| index)
        }
    }

    impl From<vk::PhysicalDeviceMemoryProperties> for Memory {
        fn from(value: vk::PhysicalDeviceMemoryProperties) -> Self {
            let types = value
                .memory_types
                .iter()
                .take(value.memory_type_count as usize)
                .copied()
                .collect();
            let heaps = value
                .memory_heaps
                .iter()
                .take(value.memory_heap_count as usize)
                .copied()
                .collect();
            Self { types, heaps }
        }
    }

    #[derive(Debug, Clone, Copy, Display)]
    #[display("{m:>2} x {n:>2} x {k:>2}\t{a} x {b} + {c} → {o}")]
    pub struct CooperativeMatrix {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub a: super::num::DataType,
        pub b: super::num::DataType,
        pub c: super::num::DataType,
        pub o: super::num::DataType,
    }

    impl TryFrom<vk::CooperativeMatrixPropertiesKHR> for CooperativeMatrix {
        type Error = super::num::DataTypeError;

        fn try_from(value: vk::CooperativeMatrixPropertiesKHR) -> Result<Self, Self::Error> {
            Ok(Self {
                m: value.m_size as usize,
                n: value.n_size as usize,
                k: value.k_size as usize,
                a: value.a_type.try_into()?,
                b: value.b_type.try_into()?,
                c: value.c_type.try_into()?,
                o: value.result_type.try_into()?,
            })
        }
    }
}

mod inner {
    use std::{ptr::NonNull, sync::Mutex};

    use derive_more::{Deref, DerefMut};
    use vulkanalia::prelude::v1_4::*;

    #[derive(Debug)]
    pub struct App {
        pub instance: Instance,
        pub device: Device,
        pub properties: super::Properties,
        pub compute: super::Submit,
        pub transfer: super::Submit,
        pub pipeline_cache: vk::PipelineCache,
        pub descriptor_pool: vk::DescriptorPool,
        pub query_pool: vk::QueryPool,
    }

    #[derive(Debug)]
    pub struct Memory {
        pub app: super::App,
        pub handle: vk::DeviceMemory,
        pub external: vk::ExternalMemoryHandleTypeFlags,
    }

    #[derive(Debug, Default, Clone, Copy, Deref, DerefMut)]
    pub struct Ptr(Option<NonNull<u8>>);

    impl Ptr {
        #[inline]
        pub fn new<T>(ptr: *mut T) -> Self {
            Self(NonNull::new(ptr as *mut u8))
        }
    }

    unsafe impl Send for Ptr {}

    unsafe impl Sync for Ptr {}

    #[derive(Debug)]
    pub struct Tensor {
        pub app: super::App,
        pub memory: super::Memory,
        pub buffer: vk::Buffer,
        pub layout: super::layout::Layout,
        pub r#type: super::num::DataType,
        pub address: vk::DeviceAddress,
        pub size: usize,
        pub ptr: Mutex<Ptr>,
    }

    #[derive(Debug)]
    pub struct Uniform {
        pub app: super::App,
        pub memory: super::Memory,
        pub buffer: vk::Buffer,
        pub size: usize,
        pub ptr: Mutex<Ptr>,
    }

    #[derive(Debug)]
    pub struct Image {
        pub app: super::App,
        pub memory: super::Memory,
        pub r#type: (vk::ImageType, vk::ImageViewType),
        pub image: vk::Image,
        pub view: vk::ImageView,
        pub format: vk::Format,
        pub extent: [usize; 3],
        pub layers: usize,
        pub size: usize,
    }

    #[derive(Debug)]
    pub struct Sampler {
        pub app: super::App,
        pub image: super::Image,
        pub sampler: vk::Sampler,
    }

    #[derive(Debug)]
    pub struct CommandBuffer {
        pub app: super::App,
        pub pool: vk::CommandPool,
        pub handle: Vec<vk::CommandBuffer>,
    }

    #[derive(Debug)]
    pub struct Kernel {
        pub app: super::App,
        pub module: vk::ShaderModule,
        pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
        pub descriptor_sets: Vec<vk::DescriptorSet>,
        pub pipeline_layout: vk::PipelineLayout,
        pub pipeline: vk::Pipeline,
        pub resources: Mutex<Vec<super::KernelResource>>,
    }

    #[derive(Debug)]
    pub struct Semaphore {
        pub app: super::App,
        pub handle: vk::Semaphore,
    }
}

impl Drop for inner::App {
    fn drop(&mut self) {
        let device = &self.device;
        let instance = &self.instance;
        unsafe {
            device.destroy_command_pool(self.compute.pool, None);
            device.destroy_command_pool(self.transfer.pool, None);
            device.destroy_pipeline_cache(self.pipeline_cache, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_query_pool(self.query_pool, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}

impl Drop for inner::Memory {
    fn drop(&mut self) {
        unsafe {
            self.app.device.free_memory(self.handle, None);
        }
    }
}

impl Drop for inner::Tensor {
    fn drop(&mut self) {
        unsafe {
            self.app.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl Drop for inner::Uniform {
    fn drop(&mut self) {
        unsafe {
            self.app.device.destroy_buffer(self.buffer, None);
        }
    }
}

impl Drop for inner::Image {
    fn drop(&mut self) {
        unsafe {
            self.app.device.destroy_image_view(self.view, None);
            self.app.device.destroy_image(self.image, None);
        }
    }
}

impl Drop for inner::Sampler {
    fn drop(&mut self) {
        unsafe {
            self.app.device.destroy_sampler(self.sampler, None);
        }
    }
}

impl Drop for inner::CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.app
                .device
                .free_command_buffers(self.pool, &self.handle);
        }
    }
}

impl Drop for inner::Kernel {
    fn drop(&mut self) {
        let device = &self.app.device;
        let descriptor_pool = self.app.descriptor_pool;
        let Self {
            module,
            descriptor_set_layouts,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            ..
        } = self;

        unsafe {
            device.destroy_shader_module(*module, None);
            descriptor_set_layouts
                .iter()
                .for_each(|&layout| device.destroy_descriptor_set_layout(layout, None));
            device
                .free_descriptor_sets(descriptor_pool, descriptor_sets)
                .expect("failed to free descriptor sets");
            device.destroy_pipeline_layout(*pipeline_layout, None);
            device.destroy_pipeline(*pipeline, None);
        }
    }
}

impl Drop for inner::Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.app.device.destroy_semaphore(self.handle, None);
        }
    }
}

#[derive(Debug, Clone, Deref)]
pub struct App(Arc<inner::App>);

impl App {
    /// Creates a new Vulkan application instance.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    /// - Calls Vulkan API functions which may have undefined behavior if used incorrectly
    /// - Vulkan validation layers are only enabled in debug builds
    ///
    /// Callers must ensure:
    /// - Vulkan drivers are properly installed on the system
    /// - The system has a compatible GPU with required Vulkan extensions
    unsafe fn new_unsafe() -> Result<Self, Box<dyn Error>> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|_| "failed to load vulkan")?;
        log::info!("{}", entry.version()?);

        let layers: HashSet<_> = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|properties| properties.layer_name)
            .collect();
        if VALIDATION_ENABLED && !layers.contains(&VALIDATION_LAYER) {
            return Err("validation layer not found".into());
        }

        let layers = match VALIDATION_ENABLED {
            true => vec![VALIDATION_LAYER],
            false => vec![],
        };
        let layers = layers.iter().map(|name| name.as_ptr()).collect_vec();

        let extensions = {
            // enumerate and validate instance extensions
            log::info!("enumerating available instance extensions...");
            let properties = entry.enumerate_instance_extension_properties(None)?;

            log::info!("found {} available instance extensions", properties.len());
            properties
                .iter()
                .for_each(|ext| log::info!("\t{}", ext.extension_name));

            let supported: HashSet<_> = properties
                .into_iter()
                .map(|property| property.extension_name)
                .collect();
            INSTANCE_EXTENSIONS
                .iter()
                .filter(|ext| !supported.contains(&ext.name))
                .for_each(|ext| log::warn!("instance does not support extension {}", ext.name));
            INSTANCE_EXTENSIONS
                .iter()
                .filter(|ext| supported.contains(&ext.name))
                .map(|ext| ext.name.as_ptr())
                .collect_vec()
        };

        let info = vk::ApplicationInfo::builder()
            .application_name(env!("CARGO_PKG_NAME").as_bytes())
            .application_version(parse_version(env!("CARGO_PKG_VERSION")))
            .api_version(vk::make_version(1, 3, 0));

        let info = vk::InstanceCreateInfo::builder()
            .application_info(&info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = entry.create_instance(&info, None)?;

        let (device, compute_family_index, transfer_family_index, atomic_float) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .filter_map(|device| {
                let properties = instance
                    .enumerate_device_extension_properties(device, None)
                    .ok()?;
                let supported: HashSet<_> = properties
                    .iter()
                    .map(|properties| properties.extension_name)
                    .collect();
                if let Some(ext) = DEVICE_EXTENSIONS
                    .iter()
                    .find(|ext| !supported.contains(&ext.name))
                {
                    let properties = instance.get_physical_device_properties(device);
                    log::warn!(
                        "physical device {} does not support extension {}",
                        properties.device_name,
                        ext.name
                    );
                    return None;
                }
                // 稀疏 FFN 稀疏内核需要 buffer 级 fp32 原子累加（VK_EXT_shader_atomic_float）。
                // 非必需：不支持时回退稠密 gemv（supports_sparse_ffn=false）。
                let atomic_float = supported.contains(&vk::EXT_SHADER_ATOMIC_FLOAT_EXTENSION.name);

                let queue_families = instance.get_physical_device_queue_family_properties(device);
                let compute_family_index = queue_families
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.timestamp_valid_bits != 0)
                    .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))?
                    .0 as u32;
                let transfer_family_index = queue_families
                    .iter()
                    .enumerate()
                    .filter(|&(index, _)| index as u32 != compute_family_index)
                    .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
                    .min_by_key(|(_, p)| p.queue_flags.bits().count_ones())?
                    .0 as u32;

                Some((
                    device,
                    compute_family_index,
                    transfer_family_index,
                    atomic_float,
                ))
            })
            .min_by_key(|&(device, _, _, _)| {
                let properties = instance.get_physical_device_properties(device);
                match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 2,
                }
            })
            .ok_or("cannot find physical device")?;

        let cooperative_matrix = instance
            .get_physical_device_cooperative_matrix_properties_khr(device)?
            .into_iter()
            .filter(|p| p.scope == vk::ScopeKHR::SUBGROUP)
            .filter_map(|p| p.try_into().ok())
            .collect_vec();

        let mut vk13 = vk::PhysicalDeviceVulkan13Properties::builder();
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::builder();
        let mut properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut vk13)
            .push_next(&mut subgroup_props);
        instance.get_physical_device_properties2(device, &mut properties2);

        // 先取走 properties2 的 Copy 字段，结束对 subgroup_props 的可变借用，再 build 读取 subgroup_size。
        let properties = properties2.properties;
        let vendor_id = properties.vendor_id; // 在变量被遮蔽前捕获 vendor_id
        let subgroup_props = subgroup_props.build();
        let device_name = properties.device_name.as_bytes();
        let device_name = std::ffi::CStr::from_bytes_until_nul(device_name).unwrap_or_default();
        let device_name = device_name.to_str().unwrap_or("invalid device name");
        log::info!("\tdevice name: {device_name}");
        log::info!("\tdevice type: {:?}", properties.device_type);
        log::info!(
            "\tdriver version: {}.{}.{}",
            vk::version_major(properties.driver_version),
            vk::version_minor(properties.driver_version),
            vk::version_patch(properties.driver_version)
        );
        log::info!(
            "\tAPI version: {}.{}.{}",
            vk::version_major(properties.api_version),
            vk::version_minor(properties.api_version),
            vk::version_patch(properties.api_version)
        );
        log::info!("\tvendor ID: 0x{:04X}", vendor_id);
        log::info!("\tdevice ID: 0x{:04X}", properties.device_id);

        let limits = properties.limits;
        let vk13 = vk13.build();

        let properties = instance.get_physical_device_memory_properties(device);

        log::info!("\tmemory heaps: {} heap(s)", properties.memory_heap_count);
        for i in 0..properties.memory_heap_count as usize {
            let heap = &properties.memory_heaps[i];
            let size = heap.size / 1024 / 1024;
            let flags = heap.flags;
            log::info!("\t\theap {i}: {size} MB, flags: {flags:?}");
        }
        log::info!("\tmemory types: {} type(s)", properties.memory_type_count);

        let memory = properties.into();

        let properties = Properties {
            memory,
            cooperative_matrix,
            vk13,
            limits,
            subgroup_size: subgroup_props.subgroup_size,
            vendor_id, // 使用之前捕获的 vendor_id
            atomic_float,
        };
        log::info!("{properties}");
        log::info!("\tsubgroup size: {}", subgroup_props.subgroup_size);

        let mut extensions: Vec<*const std::os::raw::c_char> = DEVICE_EXTENSIONS
            .iter()
            .map(|ext| ext.name.as_ptr())
            .collect();
        // 稀疏 FFN 稀疏内核需要 VK_EXT_shader_atomic_float（buffer 级 fp32 原子累加）；设备不支持则跳过。
        if atomic_float {
            extensions.push(vk::EXT_SHADER_ATOMIC_FLOAT_EXTENSION.name.as_ptr());
        }
        assert_ne!(compute_family_index, transfer_family_index);
        let infos = &[
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(compute_family_index)
                .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(transfer_family_index)
                .queue_priorities(&[0.0]),
        ];

        let mut features_11 =
            vk::PhysicalDeviceVulkan11Features::builder().storage_buffer_16bit_access(true);
        let mut features_12 = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .shader_float16(true)
            .shader_int8(true)
            .shader_subgroup_extended_types(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .timeline_semaphore(true);
        let mut features_13 = vk::PhysicalDeviceVulkan13Features::builder()
            .maintenance4(true)
            .synchronization2(true)
            .compute_full_subgroups(true)
            .subgroup_size_control(true);
        let mut features_cooperative_matrix =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::builder().cooperative_matrix(true);
        // 稀疏 FFN 稀疏内核：buffer 级 fp32 原子累加（仅当设备支持该扩展时启用）。
        let mut features_atomic_float: Option<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> =
            atomic_float.then(|| {
                vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::builder()
                    .shader_buffer_float32_atomic_add(true)
                    .build()
            });

        let mut info = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&extensions)
            .queue_create_infos(infos)
            .push_next(&mut features_11)
            .push_next(&mut features_12)
            .push_next(&mut features_13)
            .push_next(&mut features_cooperative_matrix);
        if let Some(f) = features_atomic_float.as_mut() {
            info = info.push_next(f);
        }
        let device = instance.create_device(device, &info, None)?;

        let compute = Submit::new(&device, compute_family_index, 0)?;
        let transfer = Submit::new(&device, transfer_family_index, 0)?;

        let info = vk::PipelineCacheCreateInfo::builder();
        let pipeline_cache = device.create_pipeline_cache(&info, None)?;

        let sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(DESCRIPTOR_COUNT_UNIFORM_BUFFER as u32),
            // uniform 池化路径（runtime record_kernel）：所有算子 kernel 的
            // uniform 绑定均为 DYNAMIC 类型（offset 随 dispatch 变化）。
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(DESCRIPTOR_COUNT_UNIFORM_BUFFER as u32),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(DESCRIPTOR_COUNT_SAMPLER as u32),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(DESCRIPTOR_COUNT_STORAGE_IMAGE as u32),
        ];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&sizes)
            .max_sets(DESCRIPTOR_POOL_MAX_SETS as u32);
        let descriptor_pool = device.create_descriptor_pool(&info, None)?;

        let info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERY_POOL_SIZE as u32);
        let query_pool = device.create_query_pool(&info, None)?;

        Ok(Self(Arc::new(inner::App {
            instance,
            device,
            properties,
            compute,
            transfer,
            pipeline_cache,
            descriptor_pool,
            query_pool,
        })))
    }

    unsafe fn create_memory_unsafe(
        &self,
        size: u64,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Memory, MemoryError> {
        let index = self
            .properties
            .memory
            .find(bits, properties)
            .ok_or(MemoryError::NotFound)? as u32;
        let mut info = vk::MemoryAllocateFlagsInfo::builder()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)
            .device_mask(0);
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .push_next(&mut info);
        let handle = self.device.allocate_memory(&info, None)?;

        let app = self.clone();
        let external = vk::ExternalMemoryHandleTypeFlags::empty();
        Ok(Memory(Arc::new(inner::Memory {
            app,
            handle,
            external,
        })))
    }

    unsafe fn create_external_memory_unsafe(
        &self,
        size: u64,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
        ext: Option<&mut impl vk::Cast<Target = impl vk::ExtendsMemoryAllocateInfo>>,
    ) -> Result<Memory, MemoryError> {
        let index = self
            .properties
            .memory
            .find(bits, properties)
            .ok_or(MemoryError::NotFound)? as u32;
        let mut export = vk::ExportMemoryAllocateInfo::builder().handle_types(external);
        let mut info = vk::MemoryAllocateFlagsInfo::builder()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS)
            .device_mask(0);
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .push_next(&mut export)
            .push_next(&mut info);
        let info = match ext {
            Some(ext) => info.push_next(ext),
            None => info,
        };
        let handle = self.device.allocate_memory(&info, None)?;

        let app = self.clone();
        Ok(Memory(Arc::new(inner::Memory {
            app,
            handle,
            external,
        })))
    }

    /// Imports external device memory from a Win32 handle created by another API (e.g., D3D12).
    ///
    /// This function imports Vulkan device memory from an external Win32 handle,
    /// allowing memory shared by other graphics APIs to be used within Vulkan.
    ///
    /// # Arguments
    ///
    /// - `size`: The size of memory to import in bytes
    /// - `bits`: Memory type bits from Vulkan memory requirements
    /// - `properties`: Memory property flags specifying the type of memory to import
    /// - `ext`: External memory handle type flags (e.g., `OPAQUE_WIN32`, `D3D12_RESOURCE`)
    /// - `handle`: The Win32 handle to import
    ///
    /// # Returns
    ///
    /// Returns `Ok(Memory)` containing the imported Vulkan device memory handle.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable memory type is found for the specified properties
    /// - Memory import fails
    ///
    /// # Safety
    ///
    /// The caller must ensure that the handle is valid.
    pub unsafe fn import_external_memory_unsafe(
        &self,
        size: u64,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
        handle: vk::HANDLE,
        ext: Option<&mut impl vk::Cast<Target = impl vk::ExtendsMemoryAllocateInfo>>,
    ) -> Result<Memory, MemoryError> {
        let index = self
            .properties
            .memory
            .find(bits, properties)
            .ok_or(MemoryError::NotFound)? as u32;
        let mut import = vk::ImportMemoryWin32HandleInfoKHR::builder()
            .handle_type(external)
            .handle(handle);
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(index)
            .push_next(&mut import);
        let info = match ext {
            Some(ext) => info.push_next(ext),
            None => info,
        };
        let handle = self.device.allocate_memory(&info, None)?;

        let app = self.clone();
        Ok(Memory(Arc::new(inner::Memory {
            app,
            handle,
            external,
        })))
    }

    unsafe fn create_tensor_unsafe<T: super::num::Scalar>(
        &self,
        layout: impl super::layout::IntoLayout,
        len: Option<usize>,
        mapped: bool,
    ) -> Result<Tensor<T>, TensorError> {
        let layout = layout.into_layout();
        let r#type = T::DATA_TYPE;
        let len = len.unwrap_or_else(|| layout.co_size());
        let size = len * size_of::<T>();

        if layout.co_size() > len {
            return Err(TensorError::Len(len, layout.co_size()));
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.device.create_buffer(&info, None)?;
        let req = self.device.get_buffer_memory_requirements(buffer);

        let properties = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let properties = match mapped {
            true => properties,
            false => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };

        let memory = self.create_memory(req.size, req.memory_type_bits, properties)?;
        self.device.bind_buffer_memory(buffer, *memory, 0)?;

        let info = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
        let address = self.device.get_buffer_device_address(&info);

        let ptr = match mapped {
            true => inner::Ptr::new(self.device.map_memory(
                *memory,
                0,
                req.size,
                vk::MemoryMapFlags::empty(),
            )?),
            false => inner::Ptr::default(),
        };
        let ptr = Mutex::new(ptr);

        let app = self.clone();
        let inner = Arc::new(inner::Tensor {
            app,
            buffer,
            memory,
            layout,
            r#type,
            address,
            size,
            ptr,
        });
        let phantom = PhantomData;
        Ok(Tensor { inner, phantom })
    }

    unsafe fn create_uniform_unsafe(&self, size: usize) -> Result<Uniform, UniformError> {
        let info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.device.create_buffer(&info, None)?;

        let req = self.device.get_buffer_memory_requirements(buffer);
        let properties = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let memory = self.create_memory(req.size, req.memory_type_bits, properties)?;

        self.device.bind_buffer_memory(buffer, *memory, 0)?;

        let ptr = self
            .device
            .map_memory(*memory, 0, req.size, Default::default())?;
        let ptr = Mutex::new(inner::Ptr::new(ptr));

        let app = self.clone();
        Ok(Uniform(Arc::new(inner::Uniform {
            app,
            memory,
            buffer,
            ptr,
            size,
        })))
    }

    unsafe fn create_image_unsafe(
        &self,
        r#type: (vk::ImageType, vk::ImageViewType),
        extent: [usize; 3],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Image, ImageError> {
        let (depth, layers) = match r#type.0 {
            vk::ImageType::_1D => (1, extent[2]),
            vk::ImageType::_2D => (1, extent[2]),
            vk::ImageType::_3D => (extent[2], 1),
            _ => return Err(ImageError::Type),
        };
        let extent3d = vk::Extent3D::builder()
            .width(extent[0] as u32)
            .height(extent[1] as u32)
            .depth(depth as u32);
        let info = vk::ImageCreateInfo::builder()
            .image_type(r#type.0)
            .extent(extent3d)
            .format(format)
            .mip_levels(1)
            .array_layers(layers as u32)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = self.device.create_image(&info, None)?;

        let req = self.device.get_image_memory_requirements(image);
        let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let memory = self.create_memory(req.size, req.memory_type_bits, properties)?;

        self.device.bind_image_memory(image, *memory, 0)?;

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(layers as u32);
        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(r#type.1)
            .format(format)
            .components(components)
            .subresource_range(subresource);
        let view = self
            .device
            .create_image_view(&info, None)
            .map_err(ImageError::Vulkan)?;

        let app = self.clone();
        let extent = [extent[0], extent[1], depth];
        let ppb = vk_format_ppb(format);
        let size = layers
            * extent[0].div_ceil(ppb[0])
            * extent[1].div_ceil(ppb[1])
            * extent[2].div_ceil(ppb[2])
            * vk_format_bpb(format);

        Ok(Image(Arc::new(inner::Image {
            app,
            memory,
            r#type,
            image,
            view,
            format,
            extent,
            layers,
            size,
        })))
    }

    unsafe fn create_external_image_unsafe(
        &self,
        r#type: (vk::ImageType, vk::ImageViewType),
        extent: [usize; 3],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Image, ImageError> {
        let (depth, layers) = match r#type.0 {
            vk::ImageType::_1D => (1, extent[2]),
            vk::ImageType::_2D => (1, extent[2]),
            vk::ImageType::_3D => (extent[2], 1),
            _ => return Err(ImageError::Type),
        };

        let extent3d = vk::Extent3D::builder()
            .width(extent[0] as u32)
            .height(extent[1] as u32)
            .depth(depth as u32);
        let mut info = vk::ExternalMemoryImageCreateInfo::builder().handle_types(external);
        let info = vk::ImageCreateInfo::builder()
            .push_next(&mut info)
            .image_type(r#type.0)
            .extent(extent3d)
            .format(format)
            .mip_levels(1)
            .array_layers(layers as u32)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = self.device.create_image(&info, None)?;
        let req = self.device.get_image_memory_requirements(image);
        let size = req.size;
        let bits = req.memory_type_bits;
        let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let mut info = vk::MemoryDedicatedAllocateInfo::builder().image(image);
        let ext = Some(&mut info);
        let memory = self.create_external_memory(size, bits, properties, external, ext)?;

        self.device.bind_image_memory(image, *memory, 0)?;

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(layers as u32);
        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(r#type.1)
            .format(format)
            .components(components)
            .subresource_range(subresource);
        let view = self
            .device
            .create_image_view(&info, None)
            .map_err(ImageError::Vulkan)?;

        let app = self.clone();
        let extent = [extent[0], extent[1], depth];
        let ppb = vk_format_ppb(format);
        let size = layers
            * extent[0].div_ceil(ppb[0])
            * extent[1].div_ceil(ppb[1])
            * extent[2].div_ceil(ppb[2])
            * vk_format_bpb(format);

        Ok(Image(Arc::new(inner::Image {
            app,
            memory,
            r#type,
            image,
            view,
            format,
            extent,
            layers,
            size,
        })))
    }

    /// Imports an image from an external Win32 handle.
    ///
    /// This function creates a Vulkan image and imports external memory from a Win32 handle
    /// (typically from Direct3D 12 or other interop scenarios). The image is created with
    /// optimal tiling and bound to the imported memory.
    ///
    /// # Arguments
    ///
    /// - `r#type`: A tuple of `(ImageType, ImageViewType)` specifying the image dimensionality
    ///   and how the image should be viewed
    /// - `extent`: The dimensions of the image as `[width, height, depth_or_layers]`.
    ///   For 2D images, the third element represents array layers.
    ///   For 3D images, the third element represents depth.
    /// - `format`: The Vulkan format of the image
    /// - `usage`: Image usage flags specifying how the image will be used
    /// - `ext`: External memory handle type flags (e.g., `D3D12_RESOURCE_BIT`)
    /// - `dx_handle`: The Win32 handle to the external resource to import
    ///
    /// # Returns
    ///
    /// Returns `Ok(Image)` containing the imported Vulkan image with bound memory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image creation fails
    /// - No suitable memory type is found
    /// - Memory import fails
    /// - Memory binding fails
    /// - Image view creation fails
    ///
    /// # Safety
    ///
    /// The caller must ensure that the handle is valid.
    pub unsafe fn import_external_image_unsafe(
        &self,
        r#type: (vk::ImageType, vk::ImageViewType),
        extent: [usize; 3],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
        handle: vk::HANDLE,
    ) -> Result<Image, ImageError> {
        let (depth, layers) = match r#type.0 {
            vk::ImageType::_1D => (1, extent[2]),
            vk::ImageType::_2D => (1, extent[2]),
            vk::ImageType::_3D => (extent[2], 1),
            _ => return Err(ImageError::Type),
        };

        let extent3d = vk::Extent3D::builder()
            .width(extent[0] as u32)
            .height(extent[1] as u32)
            .depth(depth as u32);
        let mut info = vk::ExternalMemoryImageCreateInfo::builder().handle_types(external);
        let info = vk::ImageCreateInfo::builder()
            .push_next(&mut info)
            .image_type(r#type.0)
            .extent(extent3d)
            .format(format)
            .mip_levels(1)
            .array_layers(layers as u32)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::_1);

        let image = self.device.create_image(&info, None)?;

        let req = self.device.get_image_memory_requirements(image);
        let mut info = vk::MemoryDedicatedAllocateInfo::builder().image(image);
        let ext = Some(&mut info);
        let memory = self.import_external_memory_unsafe(
            req.size,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            external,
            handle,
            ext,
        )?;

        self.device.bind_image_memory(image, *memory, 0)?;

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(layers as u32);

        let info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(r#type.1)
            .format(format)
            .subresource_range(*subresource);
        let view = self.device.create_image_view(&info, None)?;

        let app = self.clone();
        let extent = [extent[0], extent[1], depth];
        let ppb = vk_format_ppb(format);
        let size = layers
            * extent[0].div_ceil(ppb[0])
            * extent[1].div_ceil(ppb[1])
            * extent[2].div_ceil(ppb[2])
            * vk_format_bpb(format);

        Ok(Image(Arc::new(inner::Image {
            app,
            memory,
            r#type,
            image,
            view,
            format,
            extent,
            layers,
            size,
        })))
    }

    unsafe fn create_shader_unsafe(&self, code: &[u8]) -> Result<vk::ShaderModule, vk::ErrorCode> {
        use std::mem::{align_of, size_of};

        let ptr = code.as_ptr() as usize;
        let size = code.len();

        let pa = ptr.is_multiple_of(align_of::<u32>());
        let sa = size.is_multiple_of(size_of::<u32>());

        let code = if pa && sa {
            let code = bytemuck::cast_slice(code);
            Cow::Borrowed(code)
        } else {
            let mut buffer = vec![0u32; size.div_ceil(size_of::<u32>())];
            let bytes = bytemuck::cast_slice_mut(&mut buffer);
            bytes[0..size].copy_from_slice(code);
            Cow::Owned(buffer)
        };

        let info = vk::ShaderModuleCreateInfo::builder()
            .code(&code)
            .code_size(size);
        let module = self.device.create_shader_module(&info, None)?;
        Ok(module)
    }

    unsafe fn create_kernel_unsafe(
        &self,
        code: &[u8],
        specialization: &[u32],
        bindings: &[impl vk::Cast<Target = vk::DescriptorSetLayoutBinding>],
    ) -> Result<Kernel, vk::ErrorCode> {
        let module = self.create_shader_unsafe(code)?;

        let entries = specialization
            .iter()
            .enumerate()
            .map(|(id, _)| {
                let id = id as u32;
                let size = std::mem::size_of::<u32>();
                let offset = id * size as u32;
                vk::SpecializationMapEntry::builder()
                    .constant_id(id)
                    .size(size)
                    .offset(offset)
            })
            .collect_vec();

        let specialization_info = vk::SpecializationInfo::builder()
            .data(bytemuck::cast_slice(specialization))
            .map_entries(&entries);

        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        let descriptor_set_layout = self.device.create_descriptor_set_layout(&info, None)?;
        let descriptor_set_layouts = vec![descriptor_set_layout];

        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&descriptor_set_layouts);
        let descriptor_sets = self.device.allocate_descriptor_sets(&info)?;

        let info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);
        let pipeline_layout = self.device.create_pipeline_layout(&info, None)?;

        // 跨硬件自适应：required_subgroup_size 不能写死 32（NVIDIA=32，AMD=64，Intel 可变）。
        // 读设备原生 subgroup_size（PhysicalDeviceSubgroupProperties），为 0 时回退 32 保底。
        let subgroup_size = match self.properties.subgroup_size {
            0 => 32,
            s => s,
        };
        let mut info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::builder()
            .required_subgroup_size(subgroup_size);
        let info = vk::PipelineShaderStageCreateInfo::builder()
            .push_next(&mut info)
            .specialization_info(&specialization_info)
            .module(module)
            .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(c"main".to_bytes());

        let info = vk::ComputePipelineCreateInfo::builder()
            .stage(info)
            .layout(pipeline_layout);
        let pipelines = self
            .device
            .create_compute_pipelines(self.pipeline_cache, &[info], None)?;

        let pipeline = pipelines.0[0];

        let app = self.clone();
        let resources = Mutex::new(Vec::new());

        Ok(Kernel(Arc::new(inner::Kernel {
            app,
            module,
            descriptor_set_layouts,
            descriptor_sets,
            pipeline_layout,
            pipeline,
            resources,
        })))
    }

    unsafe fn create_semaphore_unsafe(
        &self,
        r#type: vk::SemaphoreType,
    ) -> Result<Semaphore, vk::ErrorCode> {
        let mut info = vk::SemaphoreTypeCreateInfo::builder().semaphore_type(r#type);
        let info = vk::SemaphoreCreateInfo::builder().push_next(&mut info);
        let handle = self.device.create_semaphore(&info, None)?;

        let app = self.clone();
        Ok(Semaphore(Arc::new(inner::Semaphore { app, handle })))
    }

    unsafe fn allocate_transfer_command_buffers_unsafe(
        &self,
        count: usize,
    ) -> Result<CommandBuffer, vk::ErrorCode> {
        let app = self.clone();
        let pool = self.transfer.pool;
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(count as u32)
            .command_pool(pool);
        let handle = self.device.allocate_command_buffers(&info)?;
        Ok(CommandBuffer(Arc::new(inner::CommandBuffer {
            app,
            pool,
            handle,
        })))
    }

    unsafe fn allocate_compute_command_buffers_unsafe(
        &self,
        count: usize,
    ) -> Result<CommandBuffer, vk::ErrorCode> {
        let app = self.clone();
        let pool = self.compute.pool;
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(count as u32)
            .command_pool(pool);
        let handle = self.device.allocate_command_buffers(&info)?;
        Ok(CommandBuffer(Arc::new(inner::CommandBuffer {
            app,
            pool,
            handle,
        })))
    }

    unsafe fn query_timestamps_unsafe(&self) -> Result<[u64; QUERY_POOL_SIZE], vk::ErrorCode> {
        let mut data = [0u64; QUERY_POOL_SIZE];
        self.device.get_query_pool_results(
            self.query_pool,
            0,
            QUERY_POOL_SIZE as u32,
            bytemuck::cast_slice_mut(&mut data),
            size_of::<u64>() as u64,
            vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
        )?;
        Ok(data)
    }

    /// 读取 query pool 中前 `count` 个已写入的 timestamp（避免等待未写入/未复位的查询）。
    pub fn query_timestamps_n(&self, count: u32) -> Result<Vec<u64>, vk::ErrorCode> {
        unsafe {
            let mut data = vec![0u64; count as usize];
            self.device.get_query_pool_results(
                self.query_pool,
                0,
                count,
                bytemuck::cast_slice_mut(&mut data),
                size_of::<u64>() as u64,
                vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
            )?;
            Ok(data)
        }
    }

    /// Creates a new Vulkan application instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Vulkan drivers are not properly installed
    /// - No compatible GPU with required Vulkan extensions is found
    /// - Memory allocation fails during initialization
    #[inline]
    pub fn new() -> Result<Self, Box<dyn Error>> {
        unsafe { Self::new_unsafe() }
    }

    /// Allocates device memory with the specified size and properties.
    ///
    /// # Arguments
    ///
    /// - `size`: The size of memory to allocate in bytes
    /// - `bits`: Memory type bits from Vulkan memory requirements
    /// - `properties`: Memory property flags specifying the type of memory to allocate
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable memory type is found for the specified properties
    /// - Memory allocation fails
    #[inline]
    pub fn create_memory(
        &self,
        size: u64,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Memory, MemoryError> {
        unsafe { self.create_memory_unsafe(size, bits, properties) }
    }

    /// Creates external device memory that can be shared with other APIs (e.g., D3D12).
    ///
    /// This function allocates Vulkan device memory with external memory support,
    /// allowing it to be exported and used by other graphics APIs through a Win32 handle.
    ///
    /// # Arguments
    ///
    /// - `size`: The size of memory to allocate in bytes
    /// - `bits`: Memory type bits from Vulkan memory requirements
    /// - `properties`: Memory property flags specifying the type of memory to allocate
    /// - `external`: External memory handle type flags (e.g., `OPAQUE_WIN32`, `D3D12_RESOURCE`)
    ///
    /// # Returns
    ///
    /// Returns `Ok(Memory)` containing:
    /// - The allocated Vulkan device memory handle
    /// - A Win32 handle that can be used to share the memory with other APIs
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable memory type is found for the specified properties
    /// - Memory allocation fails
    /// - Win32 handle retrieval fails
    pub fn create_external_memory(
        &self,
        size: u64,
        bits: u32,
        properties: vk::MemoryPropertyFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
        ext: Option<&mut impl vk::Cast<Target = impl vk::ExtendsMemoryAllocateInfo>>,
    ) -> Result<Memory, MemoryError> {
        unsafe { self.create_external_memory_unsafe(size, bits, properties, external, ext) }
    }

    /// Creates a tensor with the specified layout and optional length.
    ///
    /// # Arguments
    ///
    /// - `layout`: The layout specification for the tensor
    /// - `len`: Optional length of the tensor buffer. If `None`, the length will be set to `layout.co_size()`.
    ///   If `Some(value)`, the value must be greater than or equal to `layout.co_size()`.
    ///   Otherwise, an error will be returned indicating that the length is too small.
    /// - `mapped`: Whether the buffer should be mapped to host memory
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The provided `len` is smaller than `layout.co_size()`
    /// - Memory allocation fails
    /// - Buffer creation fails
    #[inline]
    pub fn create_tensor<T: super::num::Scalar>(
        &self,
        layout: impl super::layout::IntoLayout,
        len: Option<usize>,
        mapped: bool,
    ) -> Result<Tensor<T>, TensorError> {
        unsafe { self.create_tensor_unsafe::<T>(layout, len, mapped) }
    }

    /// Creates a uniform buffer with the specified size.
    ///
    /// # Arguments
    ///
    /// - `size`: The size of the buffer in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer creation fails
    /// - Memory allocation fails
    /// - Memory mapping fails
    #[inline]
    pub fn create_uniform(&self, size: usize) -> Result<Uniform, UniformError> {
        unsafe { self.create_uniform_unsafe(size) }
    }

    /// Creates an image with the specified type, extent, format, and usage.
    ///
    /// # Arguments
    ///
    /// - `r#type`: A tuple of (ImageType, ImageViewType) specifying the image and view types
    /// - `extent`: The dimensions of the image as [width, height, depth]
    /// - `format`: The format of the image pixels
    /// - `usage`: Flags specifying how the image will be used
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image creation fails
    /// - Memory allocation fails
    /// - Image view creation fails
    #[inline]
    pub fn create_image(
        &self,
        r#type: (vk::ImageType, vk::ImageViewType),
        extent: [usize; 3],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Image, ImageError> {
        unsafe { self.create_image_unsafe(r#type, extent, format, usage) }
    }

    /// Creates an external image with the specified type, extent, format, usage, and external memory handle type.
    ///
    /// This function creates an image that can be shared with external APIs or processes
    /// through the specified external memory handle type.
    ///
    /// # Arguments
    ///
    /// - `r#type`: A tuple of (ImageType, ImageViewType) specifying the image and view types
    /// - `extent`: The dimensions of the image as [width, height, depth]
    /// - `format`: The format of the image pixels
    /// - `usage`: Flags specifying how the image will be used
    /// - `ext`: External memory handle type flags for sharing the image
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image creation fails
    /// - Memory allocation fails
    /// - Image view creation fails
    /// - External memory handle creation fails
    #[inline]
    pub fn create_external_image(
        &self,
        r#type: (vk::ImageType, vk::ImageViewType),
        extent: [usize; 3],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        ext: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Image, ImageError> {
        unsafe { self.create_external_image_unsafe(r#type, extent, format, usage, ext) }
    }

    /// Creates a compute kernel (shader) with the specified code, specialization constants, and descriptor bindings.
    ///
    /// # Arguments
    ///
    /// - `code`: The SPIR-V shader bytecode
    /// - `specialization`: Array of specialization constant values
    /// - `bindings`: Descriptor set layout bindings that define the kernel's resource interface
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Shader module creation fails
    /// - Descriptor set layout creation fails
    /// - Pipeline layout creation fails
    /// - Compute pipeline creation fails
    #[inline]
    pub fn create_kernel(
        &self,
        code: &[u8],
        specialization: &[u32],
        bindings: &[impl vk::Cast<Target = vk::DescriptorSetLayoutBinding>],
    ) -> Result<Kernel, vk::ErrorCode> {
        unsafe { self.create_kernel_unsafe(code, specialization, bindings) }
    }

    /// Creates a semaphore with the specified type.
    ///
    /// # Arguments
    ///
    /// - `r#type`: The type of semaphore to create (binary or timeline)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Semaphore creation fails
    #[inline]
    pub fn create_semaphore(&self, r#type: vk::SemaphoreType) -> Result<Semaphore, vk::ErrorCode> {
        unsafe { self.create_semaphore_unsafe(r#type) }
    }

    #[inline]
    pub fn allocate_compute_command_buffers(
        &self,
        count: usize,
    ) -> Result<CommandBuffer, vk::ErrorCode> {
        unsafe { self.allocate_compute_command_buffers_unsafe(count) }
    }

    #[inline]
    pub fn allocate_transfer_command_buffers(
        &self,
        count: usize,
    ) -> Result<CommandBuffer, vk::ErrorCode> {
        unsafe { self.allocate_transfer_command_buffers_unsafe(count) }
    }

    /// Resets a range of query pool entries.
    ///
    /// This function records a command to reset a range of query pool entries to an
    /// inactive state. After reset, the queries can be reused for new measurements.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It calls Vulkan API functions which may have undefined behavior if used incorrectly
    /// - The command buffer must be in the recording state
    /// - The query pool must be valid and properly initialized
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The query pool is valid and not destroyed during command buffer execution
    /// - The range [first, first + count - 1] is within the query pool's bounds
    /// - Proper Vulkan synchronization is maintained
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the reset command
    /// - `first`: The first query index to reset
    /// - `count`: The number of queries to reset
    #[inline]
    pub unsafe fn cmd_reset_query_pool(&self, cmd: vk::CommandBuffer, first: u32, count: u32) {
        self.device
            .cmd_reset_query_pool(cmd, self.query_pool, first, count);
    }

    /// Writes a timestamp to a query pool entry.
    ///
    /// This function records a command to write a timestamp value to a query pool entry
    /// when the specified pipeline stage is reached. The timestamp can be used for
    /// performance measurements and profiling.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It calls Vulkan API functions which may have undefined behavior if used incorrectly
    /// - The command buffer must be in the recording state
    /// - The query pool must be valid and properly initialized
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The query pool is valid and not destroyed during command buffer execution
    /// - The query index is within the query pool's bounds
    /// - The pipeline stage flag is valid for the device
    /// - Proper Vulkan synchronization is maintained
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the timestamp command
    /// - `stage`: The pipeline stage at which to write the timestamp
    /// - `query`: The query index to write the timestamp to
    #[inline]
    pub unsafe fn cmd_write_timestamp(
        &self,
        cmd: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        query: u32,
    ) {
        self.device
            .cmd_write_timestamp(cmd, stage, self.query_pool, query);
    }

    pub fn query_timestamps(&self) -> Result<[u64; QUERY_POOL_SIZE], vk::ErrorCode> {
        unsafe { self.query_timestamps_unsafe() }
    }
}

#[derive(Debug, Clone)]
pub struct Memory(Arc<inner::Memory>);

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("failed to find proper memory")]
    NotFound,
    #[error(transparent)]
    Vulkan(#[from] vk::ErrorCode),
}

impl std::ops::Deref for Memory {
    type Target = vk::DeviceMemory;

    fn deref(&self) -> &Self::Target {
        &self.0.handle
    }
}

impl Memory {
    /// Get the external memory handle.
    #[inline]
    pub fn handle_win32(&self) -> Result<vk::HANDLE, vk::ErrorCode> {
        let info = vk::MemoryGetWin32HandleInfoKHR::builder()
            .memory(self.0.handle)
            .handle_type(self.0.external);
        unsafe { self.0.app.device.get_memory_win32_handle_khr(&info) }
    }
}

#[derive(Debug, Clone, Deref)]
#[deref(forward)]
pub struct Tensor<T> {
    #[deref]
    inner: Arc<inner::Tensor>,
    #[deref(ignore)]
    phantom: PhantomData<T>,
}

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("tensor len too small: {0} < {1}")]
    Len(usize, usize),
    #[error("tensor size too small: {0} < {1} + {2}")]
    Size(usize, usize, usize),
    #[error("failed to lock tensor")]
    Lock,
    #[error("tensor is not mapped")]
    Unmapped,
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Vulkan(#[from] vk::ErrorCode),
}

impl<T: super::num::Scalar> Tensor<T> {
    unsafe fn copy_from_unsafe(&self, data: &[T], offset: usize) -> Result<Self, TensorError> {
        let size = size_of_val(data);
        let offset = offset * size_of::<T>();
        if self.size < size + offset {
            return Err(TensorError::Size(self.size, size, offset));
        }
        let ptr = self
            .ptr
            .lock()
            .map_err(|_| TensorError::Lock)?
            .ok_or(TensorError::Unmapped)?
            .add(offset);
        let src = NonNull::from_ref(data).cast();
        ptr.copy_from_nonoverlapping(src, size);
        Ok(self.clone())
    }

    unsafe fn copy_to_unsafe(&self, data: &mut [T]) -> Result<Self, TensorError> {
        if self.size != size_of_val(data) {
            log::warn!("size mismatch: {} != {}", self.size, size_of_val(data));
        }
        let size = self.size.min(size_of_val(data));
        let ptr = self
            .ptr
            .lock()
            .map_err(|_| TensorError::Lock)?
            .ok_or(TensorError::Unmapped)?;
        let dst = NonNull::from_ref(data).cast();
        ptr.copy_to_nonoverlapping(dst, size);
        Ok(self.clone())
    }

    unsafe fn clear_unsafe(&self) -> Result<Self, Box<dyn Error>> {
        let ptr = self
            .ptr
            .lock()
            .map_err(|_| TensorError::Lock)?
            .ok_or(TensorError::Unmapped)?;
        ptr.write_bytes(0, self.size);
        Ok(self.clone())
    }

    /// Copies data to the tensor's memory.
    ///
    /// # Arguments
    ///
    /// - `data`: A slice of data to upload. The data type must implement `bytemuck::Pod` for safe transmutation.
    /// - `offset`: The offset in the tensor's memory to start uploading in element count.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the upload succeeds, or an error if:
    ///
    /// - Tensor size in bytes is less then `size_of_data(data) + offset * size_of::<T>()`, or
    /// - The tensor is not mapped.
    #[inline]
    pub fn copy_from(&self, data: &[T], offset: usize) -> Result<Self, TensorError> {
        unsafe { self.copy_from_unsafe(data, offset) }
    }

    /// Copies data from the tensor's memory.
    ///
    /// # Arguments
    ///
    /// - `data`: A mutable slice of data to download. The data type must implement `bytemuck::Pod` for safe transmutation.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the download succeeds, or an error if the tensor is not mapped.
    #[inline]
    pub fn copy_to(&self, data: &mut [T]) -> Result<Self, TensorError> {
        unsafe { self.copy_to_unsafe(data) }
    }

    /// Clears the tensor's memory by setting all bytes to zero.
    ///
    /// This function writes zeros to the entire memory region of the tensor,
    /// effectively resetting all tensor elements to their zero value.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the clear operation succeeds, or an error if:
    /// - The tensor is not mapped to host memory
    /// - The memory lock cannot be acquired
    #[inline]
    pub fn clear(&self) -> Result<Self, Box<dyn Error>> {
        unsafe { self.clear_unsafe() }
    }

    /// Copies data from another tensor using a Vulkan command buffer.
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the copy operation
    /// - `src`: The source tensor to copy data from
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the copy operation is successfully recorded, or an error if:
    /// - The source and destination tensors have different sizes
    /// - The Vulkan operation fails
    ///
    /// # Safety
    ///
    /// This is an unsafe method because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The source and destination tensors must be valid and not destroyed during the operation
    /// - The caller must ensure proper Vulkan synchronization and command buffer lifecycle management
    pub unsafe fn cmd_copy_from(&self, cmd: vk::CommandBuffer, src: &Tensor<T>) -> Self {
        if self.size != src.size {
            log::warn!("size mismatch: {} != {}", self.size, src.size);
        }
        let size = self.size.min(src.size) as u64;
        let copy = vk::BufferCopy::builder().size(size);
        self.app
            .device
            .cmd_copy_buffer(cmd, src.buffer, self.buffer, &[copy]);
        self.clone()
    }

    /// Copies data from another tensor to a range of this tensor using a Vulkan command buffer.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The source and destination tensors must be valid and properly initialized
    /// - `dst_offset + size` must not exceed the destination tensor's buffer size
    /// - `src_offset + size` must not exceed the source tensor's buffer size
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the copy operation
    /// - `src`: The source tensor to copy data from
    /// - `dst_offset`: The offset in the destination tensor (in bytes)
    /// - `src_offset`: The offset in the source tensor (in bytes)
    /// - `size`: The size of data to copy (in bytes)
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    pub unsafe fn cmd_copy_from_range(
        &self,
        cmd: vk::CommandBuffer,
        src: &Tensor<T>,
        dst_offset: u64,
        src_offset: u64,
        size: u64,
    ) -> Self {
        let copy = vk::BufferCopy::builder()
            .dst_offset(dst_offset)
            .src_offset(src_offset)
            .size(size);
        self.app
            .device
            .cmd_copy_buffer(cmd, src.buffer, self.buffer, &[copy]);
        self.clone()
    }
}

impl<T, Idx> std::ops::Index<Idx> for Tensor<T>
where
    T: super::num::Scalar,
    super::layout::Layout: super::layout::IndexFn<Idx, Output = usize>,
{
    type Output = T;

    fn index(&self, index: Idx) -> &Self::Output {
        use super::layout::IndexFn;
        let index = self.layout.value(index);

        let ptr = self
            .ptr
            .lock()
            .expect("failed to lock")
            .expect("tensor not mapped")
            .cast::<T>();

        let len = self.layout.co_size();
        if index >= len {
            panic!("index {index} out of bounds for tensor of size {len}");
        }

        // SAFETY: We know that the pointer is valid and the index is within bounds.
        unsafe { &*ptr.add(index).as_ptr() }
    }
}

pub trait TensorBarrierCommand<T> {
    /// Inserts a pipeline barrier for the tensor buffer.
    ///
    /// This function records a buffer memory barrier command that synchronizes access
    /// to the tensor's buffer memory between different queue families and pipeline stages.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The barrier parameters must be correctly specified to avoid data races
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The barrier parameters correctly reflect the actual memory access patterns
    /// - Proper Vulkan synchronization is maintained throughout the command buffer
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the barrier command
    /// - `src_queue_family_index`: Source queue family index for ownership transfer
    /// - `dst_queue_family_index`: Destination queue family index for ownership transfer
    /// - `src_access_mask`: Source access mask specifying previous access types
    /// - `dst_access_mask`: Destination access mask specifying subsequent access types
    /// - `src_stage_mask`: Source pipeline stage mask
    /// - `dst_stage_mask`: Destination pipeline stage mask
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self;
}

impl<T: super::num::Scalar> TensorBarrierCommand<T> for Tensor<T> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        let barrier = vk::BufferMemoryBarrier::builder()
            .buffer(self.buffer)
            .offset(0)
            .size(self.size as u64)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);
        self.app.device.cmd_pipeline_barrier(
            cmd,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[barrier],
            &[] as &[vk::ImageMemoryBarrier],
        );
        self.clone()
    }
}

impl<S, T, const N: usize> TensorBarrierCommand<S> for [T; N]
where
    S: super::num::Scalar,
    T: std::borrow::Borrow<Tensor<S>> + Clone,
{
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        let app = match self.first().cloned() {
            Some(tensor) => tensor.borrow().app.clone(),
            None => return self.clone(),
        };
        let barriers = self
            .iter()
            .map(|tensor| {
                let tensor = tensor.borrow();
                vk::BufferMemoryBarrier::builder()
                    .buffer(tensor.buffer)
                    .offset(0)
                    .size(tensor.size as u64)
                    .src_queue_family_index(src_queue_family_index)
                    .dst_queue_family_index(dst_queue_family_index)
                    .src_access_mask(src_access_mask)
                    .dst_access_mask(dst_access_mask)
            })
            .collect_vec();
        app.device.cmd_pipeline_barrier(
            cmd,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &barriers,
            &[] as &[vk::ImageMemoryBarrier],
        );
        self.clone()
    }
}

#[derive(Debug, Clone, Deref)]
#[deref(forward)]
pub struct Uniform(Arc<inner::Uniform>);

#[derive(Debug, Error)]
pub enum UniformError {
    #[error("failed to lock uniform")]
    Lock,
    #[error("uniform is not mapped")]
    Unmapped,
    #[error("uniform range out of bounds: offset {0} + size {1} > capacity {2}")]
    Range(usize, usize, usize),
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Vulkan(#[from] vk::ErrorCode),
}

impl Uniform {
    unsafe fn copy_from_unsafe(&self, data: &[impl bytemuck::Pod]) -> Result<Self, UniformError> {
        if self.size != size_of_val(data) {
            log::warn!("size mismatch: {} != {}", self.size, size_of_val(data));
        }
        let size = self.size.min(size_of_val(data));
        let ptr = self
            .ptr
            .lock()
            .map_err(|_| UniformError::Lock)?
            .ok_or(UniformError::Unmapped)?;
        let src = NonNull::from_ref(data).cast::<u8>();
        ptr.copy_from_nonoverlapping(src, size);
        Ok(self.clone())
    }

    /// Copies data to the uniform buffer.
    ///
    /// # Arguments
    ///
    /// - `data`: A slice of data to upload
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the upload succeeds, or an error if the operation fails.
    #[inline]
    pub fn copy_from(&self, data: &[impl bytemuck::Pod]) -> Result<Self, UniformError> {
        unsafe { self.copy_from_unsafe(data) }
    }

    /// Copies data into the uniform buffer at a byte offset（uniform 池 bump 分配用）。
    ///
    /// 与 `copy_from` 不同：本方法写入 mapped 区域的 `[offset, offset+size)` 段，
    /// 越界（超出 buffer 容量）返回 `UniformError::Range`。
    ///
    /// # Safety 说明
    ///
    /// 调用方须保证 offset 满足 `min_uniform_buffer_offset_alignment` 对齐
    /// （作为 DYNAMIC offset descriptor 使用时）。
    pub fn copy_from_at(
        &self,
        data: &[impl bytemuck::Pod],
        offset: usize,
    ) -> Result<Self, UniformError> {
        unsafe { self.copy_from_at_unsafe(data, offset) }
    }

    unsafe fn copy_from_at_unsafe(
        &self,
        data: &[impl bytemuck::Pod],
        offset: usize,
    ) -> Result<Self, UniformError> {
        let size = size_of_val(data);
        if offset.checked_add(size).is_none_or(|end| end > self.size) {
            return Err(UniformError::Range(offset, size, self.size));
        }
        let ptr = self
            .ptr
            .lock()
            .map_err(|_| UniformError::Lock)?
            .ok_or(UniformError::Unmapped)?;
        let src = NonNull::from_ref(data).cast::<u8>();
        // SAFETY: [offset, offset+size) 已校验落在 mapped 区域内
        unsafe { ptr.add(offset).copy_from_nonoverlapping(src, size) };
        Ok(self.clone())
    }
}

#[derive(Debug, Error)]
pub enum ImageError {
    #[error("invalid type")]
    Type,
    #[error("size mismatch: {0} != {1}")]
    Size(usize, usize),
    #[error("layer out of range: total {0}, given {1}")]
    Range(usize, usize),
    #[error(transparent)]
    Memory(#[from] MemoryError),
    #[error(transparent)]
    Vulkan(#[from] vk::ErrorCode),
}

#[derive(Debug, Clone, Deref)]
#[deref(forward)]
pub struct Image(Arc<inner::Image>);

impl Image {
    /// Copies data from a tensor to this image using a Vulkan command buffer.
    ///
    /// This function records a buffer-to-image copy command that transfers data
    /// from a tensor's buffer memory to this image's memory.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The source tensor and destination image must be valid and properly initialized
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The source tensor contains valid data and is properly synchronized
    /// - The image is in the correct layout for transfer operations
    /// - Proper Vulkan synchronization is maintained throughout the command buffer
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the copy operation
    /// - `src`: The source tensor containing the data to copy
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the copy operation is successfully recorded, or an error if:
    /// - The source and destination have different sizes
    /// - The Vulkan operation fails
    pub unsafe fn cmd_copy_from<T: super::num::Scalar>(
        &self,
        cmd: vk::CommandBuffer,
        src: &Tensor<T>,
    ) -> Result<Self, ImageError> {
        if self.size != src.size {
            return Err(ImageError::Size(self.size, src.size));
        }
        let subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(self.layers as u32)
            .mip_level(0);
        let extent3d = vk::Extent3D {
            width: self.extent[0] as u32,
            height: self.extent[1] as u32,
            depth: self.extent[2] as u32,
        };
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource)
            .image_offset(vk::Offset3D::default())
            .image_extent(extent3d);
        self.app.device.cmd_copy_buffer_to_image(
            cmd,
            src.buffer,
            self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        Ok(self.clone())
    }

    /// Copies data from a tensor to a range of this image using a Vulkan command buffer.
    ///
    /// This function records a buffer-to-image copy command that transfers data
    /// from a tensor's buffer memory to a specified range of this image's array layers.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The source tensor and destination image must be valid and properly initialized
    /// - The range parameters must be within bounds
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the copy operation
    /// - `src`: The source tensor containing the data to copy
    /// - `base`: The first array layer to copy to
    /// - `count`: The number of array layers to copy
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the copy operation is successfully recorded, or an error if:
    /// - The range is out of bounds
    /// - The Vulkan operation fails
    pub unsafe fn cmd_copy_from_range<T: super::num::Scalar>(
        &self,
        cmd: vk::CommandBuffer,
        src: &Tensor<T>,
        base: usize,
        count: usize,
    ) -> Result<Self, ImageError> {
        let end = base + count;
        let layers = match self.r#type.1 {
            vk::ImageViewType::_3D => self.extent[2],
            vk::ImageViewType::_2D_ARRAY => self.layers,
            _ => 0,
        };
        if end > layers {
            return Err(ImageError::Range(layers, end));
        }

        let subresource = match self.r#type.1 {
            vk::ImageViewType::_2D_ARRAY => vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(base as u32)
                .layer_count(count as u32)
                .mip_level(0),
            _ => vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0),
        };
        let offset = match self.r#type.1 {
            vk::ImageViewType::_3D => vk::Offset3D {
                x: 0,
                y: 0,
                z: base as i32,
            },
            _ => vk::Offset3D::default(),
        };
        let extent3d = match self.r#type.1 {
            vk::ImageViewType::_3D => vk::Extent3D {
                width: self.extent[0] as u32,
                height: self.extent[1] as u32,
                depth: count as u32,
            },
            _ => vk::Extent3D {
                width: self.extent[0] as u32,
                height: self.extent[1] as u32,
                depth: 1,
            },
        };
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource)
            .image_offset(offset)
            .image_extent(extent3d);
        self.app.device.cmd_copy_buffer_to_image(
            cmd,
            src.buffer,
            self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        Ok(self.clone())
    }

    unsafe fn create_sampler_unsafe(&self) -> Result<Sampler, ImageError> {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
        let sampler = self
            .app
            .device
            .create_sampler(&info, None)
            .map_err(ImageError::Vulkan)?;

        let app = self.app.clone();
        let image = self.clone();

        Ok(Sampler(Arc::new(inner::Sampler {
            app,
            image,
            sampler,
        })))
    }

    #[inline]
    pub fn create_sampler(&self) -> Result<Sampler, ImageError> {
        unsafe { self.create_sampler_unsafe() }
    }
}

pub trait ImageBarrierCommand {
    /// Inserts an image memory barrier for the image.
    ///
    /// This function records an image memory barrier command that synchronizes access
    /// to the image memory between different queue families, pipeline stages, and image layouts.
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It records Vulkan commands that must be properly synchronized
    /// - The command buffer must be in the recording state
    /// - The barrier parameters must be correctly specified to avoid data races
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The barrier parameters correctly reflect the actual image access patterns
    /// - Proper Vulkan synchronization is maintained throughout the command buffer
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to record the barrier command
    /// - `old_layout`: The current image layout before the barrier
    /// - `new_layout`: The desired image layout after the barrier
    /// - `src_queue_family_index`: Source queue family index for ownership transfer
    /// - `dst_queue_family_index`: Destination queue family index for ownership transfer
    /// - `src_access_mask`: Source access mask specifying previous access types
    /// - `dst_access_mask`: Destination access mask specifying subsequent access types
    /// - `src_stage_mask`: Source pipeline stage mask
    /// - `dst_stage_mask`: Destination pipeline stage mask
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    #[allow(clippy::too_many_arguments)]
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self;
}

impl ImageBarrierCommand for Image {
    #[allow(clippy::too_many_arguments)]
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(self.layers as u32)
            .base_mip_level(0)
            .level_count(1);
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(self.image)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .subresource_range(subresource);
        self.app.device.cmd_pipeline_barrier(
            cmd,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );
        self.clone()
    }
}

impl<T, const N: usize> ImageBarrierCommand for [T; N]
where
    T: std::borrow::Borrow<Image> + Clone,
{
    unsafe fn cmd_barrier(
        &self,
        cmd: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_queue_family_index: u32,
        dst_queue_family_index: u32,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        let app = match self.first() {
            Some(image) => &image.borrow().app,
            None => return self.clone(),
        };
        let barriers = self
            .iter()
            .map(|image| {
                let image = image.borrow();
                let subresource = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(image.layers as u32)
                    .base_mip_level(0)
                    .level_count(1);
                vk::ImageMemoryBarrier::builder()
                    .image(image.image)
                    .old_layout(old_layout)
                    .new_layout(new_layout)
                    .src_queue_family_index(src_queue_family_index)
                    .dst_queue_family_index(dst_queue_family_index)
                    .src_access_mask(src_access_mask)
                    .dst_access_mask(dst_access_mask)
                    .subresource_range(subresource)
            })
            .collect_vec();
        app.device.cmd_pipeline_barrier(
            cmd,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &barriers,
        );
        self.clone()
    }
}

#[derive(Debug, Clone, Deref)]
#[deref(forward)]
pub struct Sampler(Arc<inner::Sampler>);

#[derive(Debug, Clone)]
pub struct CommandBuffer(Arc<inner::CommandBuffer>);

impl std::ops::Deref for CommandBuffer {
    type Target = [vk::CommandBuffer];

    fn deref(&self) -> &Self::Target {
        &self.0.handle
    }
}

#[derive(Debug, Clone, Deref)]
#[deref(forward)]
pub struct Kernel(Arc<inner::Kernel>);

impl Kernel {
    /// Binds descriptor sets and compute pipeline to a command buffer.
    ///
    /// This function performs two Vulkan operations:
    /// 1. Binds the kernel's descriptor sets to the command buffer
    /// 2. Binds the kernel's compute pipeline to the command buffer
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It calls Vulkan API functions which may have undefined behavior if used incorrectly
    /// - The command buffer must be in the recording state
    /// - The offsets array must have the correct length for the descriptor sets being bound
    /// - Proper Vulkan synchronization must be maintained
    ///
    /// Callers must ensure:
    /// - The command buffer is valid and in the recording state
    /// - The offsets array length matches the number of descriptor sets being bound
    /// - The kernel and its resources are valid and properly initialized
    /// - Proper Vulkan synchronization is maintained throughout the command buffer
    ///
    /// # Arguments
    ///
    /// - `cmd`: The Vulkan command buffer to bind to
    /// - `offsets`: Dynamic offsets for descriptor set bindings
    ///
    /// # Note
    ///
    /// This function should be called after recording any necessary memory barriers
    /// and before recording dispatch commands for the compute shader.
    pub unsafe fn cmd_bind(&self, cmd: vk::CommandBuffer, offsets: &[u32]) {
        self.app.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &self.descriptor_sets,
            offsets,
        );
        self.app
            .device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
    }

    /// Creates a new binder for configuring descriptor set bindings.
    ///
    /// This function returns a `Binder` instance that allows you to configure
    /// descriptor set bindings for this kernel. The binder uses a fluent API
    /// design that supports method chaining.
    ///
    /// # Usage
    ///
    /// After calling `binder()`, you must chain calls to `bind_*` methods to
    /// configure the bindings, and finally call `.build()` to execute the
    /// descriptor set updates:
    ///
    /// ```ignore
    /// kernel.binder()
    ///     .bind_uniform(&uniform, 0, 0)
    ///     .bind_sampler(&sampler, 0, 1)
    ///     .build();
    /// ```
    ///
    /// The `.build()` method **must** be called at the end of the binding chain
    /// to actually perform the Vulkan descriptor set updates. Without calling
    /// `.build()`, no bindings will take effect.
    ///
    /// # Returns
    ///
    /// Returns a new `Binder` instance configured for this kernel.
    #[inline]
    pub fn binder(&self) -> Binder {
        Binder {
            kernel: self.clone(),
            writes: Vec::new(),
            copies: Vec::new(),
            infos: Vec::new(),
            resources: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum KernelResource {
    Uniform(Arc<inner::Uniform>),
    Sampler(Arc<inner::Sampler>),
    Image(Arc<inner::Image>),
}

impl KernelResource {
    #[inline]
    pub fn from_uniform(uniform: &Uniform) -> Self {
        Self::Uniform(uniform.0.clone())
    }

    #[inline]
    pub fn from_sampler(sampler: &Sampler) -> Self {
        Self::Sampler(sampler.0.clone())
    }

    #[inline]
    pub fn from_image(image: &Image) -> Self {
        Self::Image(image.0.clone())
    }
}

#[derive(Debug, Clone)]
pub enum DescriptorInfo {
    Buffer(Pin<Box<[vk::DescriptorBufferInfo]>>),
    Image(Pin<Box<[vk::DescriptorImageInfo]>>),
}

#[derive(Debug, Clone)]
pub struct Binder {
    kernel: Kernel,
    writes: Vec<vk::WriteDescriptorSet>,
    copies: Vec<vk::CopyDescriptorSet>,
    infos: Vec<DescriptorInfo>,
    resources: Vec<KernelResource>,
}

impl Binder {
    unsafe fn bind_uniform_unsafe(mut self, uniform: &Uniform, set: usize, binding: usize) -> Self {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform.buffer)
            .range(uniform.size as u64)
            .offset(0)
            .build();
        let info: Pin<Box<[_]>> = Box::pin([info]);
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.kernel.descriptor_sets[set])
            .dst_binding(binding as u32)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info)
            .build();
        self.infos.push(DescriptorInfo::Buffer(info));
        self.writes.push(write);
        self.resources.push(KernelResource::from_uniform(uniform));
        self
    }

    /// 绑定 uniform 池的一个定长窗口（DYNAMIC offset descriptor）。
    ///
    /// 与 `bind_uniform` 的区别：descriptor 类型为 `UNIFORM_BUFFER_DYNAMIC`，
    /// `range` 指定单次 dispatch 读取的窗口大小（而非整个 buffer）；
    /// 实际偏移在每次 `cmd_bind` 时通过 dynamic offsets 提供。
    /// 用于 uniform 池化：一个池 buffer 服务任意多次 dispatch，
    /// 消除 per-dispatch 的 buffer/memory/descriptor 创建。
    unsafe fn bind_uniform_with_range_unsafe(
        mut self,
        uniform: &Uniform,
        range: usize,
        set: usize,
        binding: usize,
    ) -> Self {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform.buffer)
            .range(range as u64)
            .offset(0)
            .build();
        let info: Pin<Box<[_]>> = Box::pin([info]);
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.kernel.descriptor_sets[set])
            .dst_binding(binding as u32)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&info)
            .build();
        self.infos.push(DescriptorInfo::Buffer(info));
        self.writes.push(write);
        self.resources.push(KernelResource::from_uniform(uniform));
        self
    }

    unsafe fn bind_sampler_unsafe(mut self, sampler: &Sampler, set: usize, binding: usize) -> Self {
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(sampler.image.view)
            .sampler(sampler.sampler)
            .build();
        let info: Pin<Box<[_]>> = Box::pin([info]);
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.kernel.descriptor_sets[set])
            .dst_binding(binding as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info)
            .build();
        self.infos.push(DescriptorInfo::Image(info));
        self.writes.push(write);
        self.resources.push(KernelResource::from_sampler(sampler));
        self
    }

    unsafe fn bind_image_unsafe(mut self, image: &Image, set: usize, binding: usize) -> Self {
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(image.view)
            .build();
        let info: Pin<Box<[_]>> = Box::pin([info]);
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.kernel.descriptor_sets[set])
            .dst_binding(binding as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&info)
            .build();
        self.infos.push(DescriptorInfo::Image(info));
        self.writes.push(write);
        self.resources.push(KernelResource::from_image(image));
        self
    }

    unsafe fn build_unsafe(mut self) {
        self.kernel
            .app
            .device
            .update_descriptor_sets(&self.writes, &self.copies);
        self.kernel
            .resources
            .lock()
            .expect("failed to lock resources")
            .append(&mut self.resources);
    }

    /// Binds a uniform buffer to the specified descriptor set and binding.
    ///
    /// This function adds a uniform buffer binding to the descriptor set update operations.
    /// The actual update will be performed when `build()` is called.
    ///
    /// # Arguments
    ///
    /// - `uniform`: The uniform buffer to bind
    /// - `set`: The descriptor set index to bind to
    /// - `binding`: The binding index within the descriptor set
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    pub fn bind_uniform(self, uniform: &Uniform, set: usize, binding: usize) -> Self {
        unsafe { self.bind_uniform_unsafe(uniform, set, binding) }
    }

    /// 绑定 uniform 池窗口（DYNAMIC offset，详见 `bind_uniform_with_range_unsafe`）。
    ///
    /// # Arguments
    ///
    /// - `uniform`: 池 uniform buffer（整个池）
    /// - `range`: 单次 dispatch 的 Params 窗口字节数
    /// - `set` / `binding`: descriptor set 与 binding 索引
    pub fn bind_uniform_with_range(
        self,
        uniform: &Uniform,
        range: usize,
        set: usize,
        binding: usize,
    ) -> Self {
        unsafe { self.bind_uniform_with_range_unsafe(uniform, range, set, binding) }
    }

    /// Binds a sampler to the specified descriptor set and binding.
    ///
    /// This function adds a sampler binding to the descriptor set update operations.
    /// The actual update will be performed when `build()` is called.
    ///
    /// # Arguments
    ///
    /// - `sampler`: The sampler to bind
    /// - `set`: The descriptor set index to bind to
    /// - `binding`: The binding index within the descriptor set
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    pub fn bind_sampler(self, sampler: &Sampler, set: usize, binding: usize) -> Self {
        unsafe { self.bind_sampler_unsafe(sampler, set, binding) }
    }

    /// Binds an image to the specified descriptor set and binding.
    ///
    /// This function adds an image binding to the descriptor set update operations.
    /// The actual update will be performed when `build()` is called.
    ///
    /// # Arguments
    ///
    /// - `image`: The image to bind
    /// - `set`: The descriptor set index to bind to
    /// - `binding`: The binding index within the descriptor set
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow method chaining.
    pub fn bind_image(self, image: &Image, set: usize, binding: usize) -> Self {
        unsafe { self.bind_image_unsafe(image, set, binding) }
    }

    /// Executes all pending descriptor set update operations.
    ///
    /// This function must be called at the end of the binding chain to actually
    /// perform all the descriptor set updates that were queued by previous `bind_*` calls.
    ///
    /// # Note
    ///
    /// This function consumes the `Binder` and performs the actual Vulkan descriptor
    /// set update operations. Without calling this function, no bindings will take effect.
    pub fn build(self) {
        unsafe { self.build_unsafe() }
    }
}

#[derive(Debug, Clone)]
pub struct Semaphore(Arc<inner::Semaphore>);

impl std::ops::Deref for Semaphore {
    type Target = vk::Semaphore;

    fn deref(&self) -> &Self::Target {
        &self.0.handle
    }
}
