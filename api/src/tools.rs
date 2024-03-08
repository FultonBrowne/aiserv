use shurbai::{
    pretty_generate,
    types::{ModelManager, ModelState},
};

use crate::{
    prompt,
    types::{Message, ToolCall, ToolDefinition},
};

const TOOL_PROMPT : &str = "Based on the above conversation, consider running the following tools, if none of them are relevant output \"{}\"\n"; //If we fine tune a model we may not need this

pub fn predict_tool_calls(
    model: &ModelState,
    model_manager: &ModelManager,
    messages: &mut Vec<Message>,
    tool_calls: &mut Vec<ToolCall>,
    tool_defs: &Vec<ToolDefinition>,
) -> String {
    let prompt_json = serde_json::to_string(tool_defs).expect("Failed to serialize prompt");
    let system_prompt = format!("{}\n{}", TOOL_PROMPT, prompt_json);
    let tools_prompt_obj = Message {
        role: "system".to_string(),
        content: system_prompt,
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
    println!("{:?}", result.generated_tokens_data);

    tool_calls.push(ToolCall {
        name: "tool".to_string(),
        arguments: None,
    });
    prompt + result.generated_tokens_data.concat().as_str()
}
