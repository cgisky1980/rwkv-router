//! rwkv-rsv：基于 Vulkan 的 RWKV-7 推理库。
//!
//! 作为 library 对标 [web-rwkv](https://github.com/cryscan/web-rwkv)：提供
//! 分词器、模型加载、状态创建/更新与一次前向推理（prompt tokens → logits）。
//! OpenAI 兼容 API 等服务层由上层（如 `ai00-cli`）构建。
//!
//! 推理路径（按模型文件自动路由）：fp16 / int8。

pub mod backend;
pub mod backend_cuda;
pub mod gpu_model;
pub mod model;
pub mod runtime;
pub mod tokenizer;
pub mod vulkan;
