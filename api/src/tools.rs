use std::sync::Arc;

use shurbai::types::ModelManager;

use crate::types::{ChatCompletionsRequest, FunctionCall};

const TOOL_PROMPT : &str = "Based on the above conversation, consider running the following tools, if none of them are relevant output \"{}\""; //If we fine tune a model we may not need this

pub fn predict_tool_calls(
    model_name: String,
    model_manager: &Arc<ModelManager>,
    prompt: String,
    request: &ChatCompletionsRequest,
) -> Vec<FunctionCall> {
    let mut functions = Vec::new();
    let model = model_manager
        .models
        .get(&model_name)
        .expect("Model not found");

    functions.push(FunctionCall {
        name: "tool".to_string(),
        arguments: serde_json::Value::Null,
    });
    functions
}
