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
    println!("{}", prompt_json);
    let tools_prompt_obj = Message {
        role: "tool".to_string(),
        content: prompt_json,
    };
    messages.insert(0, tools_prompt_obj);
    let prompt = prompt::generate_chat_prompt(&messages, &model.chat_template)
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
        Err(_) => {
            // If it was not a list, try as a single object
            serde_json::from_str(&r_str)
                .map(|single: ToolCall| vec![single]) // Wrap the single object in a Vec
                .unwrap_or_else(|_| Vec::new()) // In case of error, default to empty Vec
        }
    };

    tool_calls.extend(r_json.iter().cloned());
    if tool_calls.is_empty() {
        prompt
    } else {
        prompt + result.generated_tokens_data.concat().as_str()
    }
}
