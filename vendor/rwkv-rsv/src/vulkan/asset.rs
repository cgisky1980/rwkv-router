use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "assets"]
#[include = "shaders/spv/*.spv"]
pub struct Asset;
