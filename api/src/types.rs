use serde::{Deserialize, Serialize};
use shurbai::types::ModelDefinition;

#[derive(Serialize, Deserialize)]
pub struct StatusMessage{
    pub message: String
}

/// Config structs:
#[derive(Serialize, Deserialize)]
pub struct Config {
 //   pub host: String, // for when I make this deployable
    pub models: Vec<ModelDefinition>,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
// OpenAI structs:
#[derive(Serialize, Deserialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<FunctionDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatCompletionsResponse {
    pub choices: Vec<Message>,
}