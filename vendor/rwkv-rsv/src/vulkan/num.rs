use bytemuck::{Pod, Zeroable};
use derive_more::Display;
use half::f16;
use thiserror::Error;
use vulkanalia::prelude::v1_4::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Error)]
pub enum DataTypeError {
    #[error("invalid data type: {0:?}")]
    Invalid(vk::ComponentTypeKHR),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataType {
    F16,
    F32,
    I8,
    I32,
    U8,
    U32,
}

impl DataType {
    /// Element Size in bytes.
    pub const fn size(self) -> usize {
        match self {
            DataType::F16 => 2,
            DataType::F32 => 4,
            DataType::I8 => 1,
            DataType::I32 => 4,
            DataType::U8 => 1,
            DataType::U32 => 4,
        }
    }
}

impl TryFrom<vk::ComponentTypeKHR> for DataType {
    type Error = DataTypeError;

    fn try_from(value: vk::ComponentTypeKHR) -> Result<Self, Self::Error> {
        match value {
            vk::ComponentTypeKHR::FLOAT16 => Ok(Self::F16),
            vk::ComponentTypeKHR::FLOAT32 => Ok(Self::F32),
            vk::ComponentTypeKHR::SINT8 => Ok(Self::I8),
            vk::ComponentTypeKHR::SINT32 => Ok(Self::I32),
            vk::ComponentTypeKHR::UINT8 => Ok(Self::U8),
            vk::ComponentTypeKHR::UINT32 => Ok(Self::U32),
            _ => Err(DataTypeError::Invalid(value)),
        }
    }
}

pub trait FromI8 {
    fn from_i8(value: i8) -> Self;
}

impl FromI8 for i8 {
    fn from_i8(value: i8) -> Self {
        value
    }
}

impl FromI8 for u8 {
    fn from_i8(value: i8) -> Self {
        value as u8
    }
}

impl FromI8 for i32 {
    fn from_i8(value: i8) -> Self {
        value as i32
    }
}

impl FromI8 for u32 {
    fn from_i8(value: i8) -> Self {
        value as u32
    }
}

impl FromI8 for f16 {
    fn from_i8(value: i8) -> Self {
        f16::from_f32(value as f32)
    }
}

impl FromI8 for f32 {
    fn from_i8(value: i8) -> Self {
        value as f32
    }
}

pub trait Scalar: Sized + Clone + Copy + PartialEq + Zeroable + Pod + Send + Sync {
    const DATA_TYPE: DataType;
}

impl Scalar for f16 {
    const DATA_TYPE: DataType = DataType::F16;
}

impl Scalar for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

impl Scalar for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}

impl Scalar for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}

impl Scalar for i32 {
    const DATA_TYPE: DataType = DataType::I32;
}
