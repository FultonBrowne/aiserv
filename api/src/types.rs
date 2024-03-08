use serde::{Deserialize, Serialize};
use shurbai::types::ModelDefinition;

#[derive(Serialize, Deserialize)]
pub struct StatusMessage {
    pub message: String,
}

/// Config structs:
#[derive(Serialize, Deserialize)]
pub struct Config {
    //   pub host: String, // for when I make this deployable
    pub models: Vec<ModelDefinition>,
}
/// API structs:

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerMetadata {
    pub name: String,
    pub version: String,
    pub timestamp: i64,
}

impl ServerMetadata {
    pub fn new() -> Self {
        Self {
            name: "Friday by Shurburt".to_string(),
            version: "0.1.0".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LlmParams {
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateCall {
    pub model: String,
    pub prompt: String,
    pub stream: Option<bool>,
    pub generate_params: Option<LlmParams>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolArguments {
    name: String,
    description: String,
    data_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub arguments: Option<Vec<ToolArguments>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolParameters {
    name: String,
    argument: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Option<Vec<ToolParameters>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateCall {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub generate_params: Option<LlmParams>,
    pub tools: Option<Vec<ToolDefinition>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    pub meta: ServerMetadata,
    pub model: String,
    pub response: String,
    pub took: u128,
    pub halt_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateResponse {
    pub meta: ServerMetadata,
    pub model: String,
    pub response: String,
    pub took: u128,
    pub halt_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GeneratreResponseChuck {
    pub meta: ServerMetadata,
    pub token_str: String,
    pub model: String,
    pub halt_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateResponseChuck {
    pub meta: ServerMetadata,
    pub token_str: String,
    pub role: String,
    pub model: String,
    pub halt_reason: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelListObject {
    pub name: String,
    pub type_str: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ListModelsResponse {
    pub meta: ServerMetadata,
    pub models: Vec<ModelListObject>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ErrorResponse {
    pub meta: ServerMetadata,
    pub message: String,
}

impl ErrorResponse {
    pub fn new(message: &str) -> Self {
        Self {
            meta: ServerMetadata::new(),
            message: message.to_string(),
        }
    }
}
