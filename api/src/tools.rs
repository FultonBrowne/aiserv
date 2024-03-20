use serde_json::Error;
use shurbai::{
    pretty_generate,
    types::{ModelManager, ModelState},
};

use crate::{
    prompt,
    types::{Message, ToolCall, ToolDefinition},
};

pub fn predict_tool_calls(
    model: &ModelState,
    model_manager: &ModelManager,
    messages: &mut Vec<Message>,
    tool_calls: &mut Vec<ToolCall>,
    tool_defs: &Vec<ToolDefinition>,
) -> String {
    let prompt_json = serde_json::to_string_pretty(tool_defs).expect("Failed to serialize prompt");
    let tools_prompt_obj = Message {
        role: "tool".to_string(),
        content: prompt_json,
    };
    messages.insert(0, tools_prompt_obj);
    let prompt = prompt::generate_chat_prompt(&messages, &model.chat_template, true)
        .expect("Failed to generate prompt");
    let blank_strings: Vec<String> = Vec::new(); // Blank list stop for list
    let result = pretty_generate(
        model,
        &model_manager.backend,
        &prompt,
        512,
        &blank_strings,
        None,
        Some(true),
    )
    .expect("Failed to generate tool completions"); // we want to hard code most params here and disable stops
    let r_str = result.generated_tokens_data.concat();
    let r_json: Result<Vec<ToolCall>, Error> = serde_json::from_str(&r_str);

    let r_json: Vec<ToolCall> = match r_json {
        Ok(vec) => vec, // If r_str was a list, this will succeed
        Err(e) => {
            println!("Failed to parse as list {}", e);
            // If it was not a list, try as a single object
            serde_json::from_str(&r_str)
                .map(|single: ToolCall| vec![single]) // Wrap the single object in a Vec
                .unwrap_or_else(|_| Vec::new()) // In case of error, default to empty Vec
        }
    };

    // Lets try a little something
    tool_calls.extend(r_json.iter().cloned());
    // ChatGPT ass code
    tool_calls.retain(|tool_call| {
        tool_defs
            .iter()
            .any(|tool_def| tool_def.name == tool_call.name)
    });
    messages.push(Message {
        role: "tool_calls".to_string(),
        content: r_str,
    });
    if tool_calls.is_empty() {
        prompt
    } else {
        prompt + result.generated_tokens_data.concat().as_str()
    }
}
