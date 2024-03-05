use llama_cpp_2::{llama_backend::LlamaBackend, model::LlamaModel, token::LlamaToken};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};


pub struct LlamaResult {
    pub n_tokens: i32,
    pub n_decode: i32,
    pub duration: Duration,
    pub generated_tokens: Vec<LlamaToken>,
    pub generated_tokens_data: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelDefinition {
    pub path: String,
    pub name: String,
    pub config: ModelConfig,
    pub chat_template: ChatTemplate,
}

#[derive(Serialize, Deserialize, Clone)]
/// Represents the configuration options for a model.
pub struct ModelConfig {
    pub mirostat: Option<i32>,       // default: 0
    pub mirostat_eta: Option<f32>,   // default: 0.1
    pub mirostat_tau: Option<f32>,   // default: 5.0
    pub use_gpu: Option<bool>,       // default: true
    pub num_ctx: Option<i32>,        // default: 2048
    pub num_gqa: Option<i32>,        // no default specified
    pub num_gpu: Option<i32>,        // no default specified
    pub num_thread: Option<i32>,     // no default specified
    pub repeat_last_n: Option<i32>,  // default: 64
    pub repeat_penalty: Option<f32>, // default: 1.1
    pub temperature: Option<f32>,    // default: 0.8
    pub seed: Option<i32>,           // default: 0
    pub stop: Option<String>,        // no default specified
    pub tfs_z: Option<f32>,          // default: 1
    pub num_predict: Option<i32>,    // default: 128
    pub top_k: Option<i32>,          // default: 40
    pub top_p: Option<f32>,          // default: 0.9
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatTemplate {
    pub user_template: String,
    pub system_template: String,
    pub assistant_template: String,
    pub assistant_prompt_template: String,
    pub stops: Vec<String>,
}
impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            mirostat: Some(0),
            mirostat_eta: Some(0.1),
            mirostat_tau: Some(5.0),
            use_gpu: Some(true),
            num_ctx: Some(2048),
            num_gqa: None,
            num_gpu: None,
            num_thread: None,
            repeat_last_n: Some(64),
            repeat_penalty: Some(1.1),
            temperature: Some(0.8),
            seed: Some(0),
            stop: None,
            tfs_z: Some(1.0),
            num_predict: Some(128),
            top_k: Some(40),
            top_p: Some(0.9),
        }
    }
}

pub struct ModelState {
    pub model: LlamaModel,
    pub config: ModelConfig,
    pub chat_template: ChatTemplate,
}


pub struct ModelManager {
    pub backend: LlamaBackend,
    pub models: HashMap<String, ModelState>,
}