use crate::types::{ToolCall, ToolDefinition};

pub fn build_tool_defs_text(tool_calls: Vec<ToolDefinition>) -> String {
    serde_json::to_string(&tool_calls).unwrap() // TODO: make it plane text
}

pub fn parse_tool_call(tool_call: &str) -> Option<ToolCall> {
    serde_json::from_str(tool_call).unwrap_or(None)
}
