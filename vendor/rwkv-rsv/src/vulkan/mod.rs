// vendored Vulkan 封装库：保留未被本项目使用的 API 面（dead_code），
// 且 serde/rayon 为上游可选 feature 未在本项目声明（unexpected_cfgs），统一在此抑制。
#![allow(dead_code, unexpected_cfgs)]

pub mod app;
pub mod asset;
pub mod layout;
pub mod num;
