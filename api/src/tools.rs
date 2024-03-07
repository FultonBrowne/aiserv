use crate::types::FunctionCall;


pub fn predict_tool_calls() -> Vec<FunctionCall>{
    let mut functions = Vec::new();
    functions.push(FunctionCall {
        name: "tool".to_string(),
        arguments: serde_json::Value::Null,
    });
    functions
}