use serde::{Deserialize, Serialize};
use serde_json::Value;
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
            name: "Shurbai".to_string(),
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
pub struct ChatGenerateCall {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenerateResponse {
    pub meta: ServerMetadata,
    pub model: String,
    pub response: String,
    pub took: u128,
    pub halt_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateResponse {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GeneratreResponseChuck {
    pub meta: ServerMetadata,
    pub token_str: String,
    pub model: String,
    pub halt_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateResponseChuck {}

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

/// OpenAI structs
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefContainer {
    pub function: FunctionDefinition,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChoiceObject {
    pub index: i32,
    pub message: Message,
    pub logprobs: serde_json::Value,
    pub finish_reason: String,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub type_str: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionDefContainer>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<FunctionDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionsResponse {
    pub choices: Vec<ChoiceObject>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelsResponseOutput {
    pub data: Vec<ModelResponse>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelResponse {
    pub id: String,
    pub object: String,
    pub max_tokens: Option<i32>,
    pub name: String,
    pub owned_by: String,
    // Include other relevant fields as per the API documentation.
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamResponseChunk {
    pub id: String,
    pub object: String,
    pub system_fingerprint: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamChoice {
    pub index: usize,
    pub logprobs: Option<Value>, // or a more specific type if you know the structure
    pub finish_reason: Option<String>,
    pub delta: ChoiceDelta,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChoiceDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    // Add other fields as necessary based on API documentation
}
