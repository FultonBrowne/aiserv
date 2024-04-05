use serde::{Deserialize, Serialize};
use serde_json::Value;
use shurbai::types::ModelDefinition;

/// XML proccessing structs:
#[derive(Debug)]
pub struct XmlState {
    pub in_xml_tag: bool,
    pub in_xml_body: bool,
    pub xml_tag_name: String,
    pub xml_body_content: String,
    pub current_accumulated_tag: String,
    pub halt_output: bool,
}

impl XmlState {
    pub fn new() -> Self {
        XmlState {
            in_xml_tag: false,
            in_xml_body: false,
            xml_tag_name: String::new(),
            xml_body_content: String::new(),
            current_accumulated_tag: String::new(),
            halt_output: false,
        }
    }

    // Define other methods here as needed
}

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
    pub arguments: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatGenerateCall {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub generate_params: Option<LlmParams>,
    pub tool_call_only: Option<bool>, // set this to only call tools and nothing else
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

impl ChatGenerateResponseChuck {
    pub fn new_token(model: &str, token_str: &str) -> ChatGenerateResponseChuck {
        ChatGenerateResponseChuck {
            meta: ServerMetadata::new(), // Assuming this constructs a new ServerMetadata
            token_str: token_str.to_string(),
            role: "assistant".to_string(),
            model: model.to_string(),
            halt_reason: None,
            tool_calls: None,
        }
    }

    pub fn new_tool_call(model: &str, tool_calls: Vec<ToolCall>) -> ChatGenerateResponseChuck {
        ChatGenerateResponseChuck {
            meta: ServerMetadata::new(), // Assuming this constructs a new ServerMetadata
            token_str: "".to_string(),
            role: "assistant".to_string(),
            model: model.to_string(),
            halt_reason: None,
            tool_calls: Some(tool_calls),
        }
    }

    pub fn new_halt(full_text: &str, halt_reason: &str) -> ChatGenerateResponseChuck {
        ChatGenerateResponseChuck {
            meta: ServerMetadata::new(), // Assuming this constructs a new ServerMetadata
            token_str: full_text.to_string(),
            role: "assistant".to_string(),
            model: "".to_string(),
            halt_reason: Some(halt_reason.to_string()),
            tool_calls: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingsResponse {
    pub meta: ServerMetadata,
    pub model: String,
    pub embeddings: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub prompt: String,
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
