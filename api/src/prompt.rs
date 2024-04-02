use serde::Serialize;
use shurbai::types::ChatTemplate;
use std::io::{Error, Result};
use tinytemplate::TinyTemplate;

use crate::types::Message;

#[derive(Serialize)]
struct TemplateContext {
    content: String,
}

#[derive(Serialize)]
struct ToolsTemplateContext {
    tools: String,
    template: String,
}

const TOOL_TEMPLATE_JSON: &str = r#"{
  "name": "name_of_tool",
  "arguments": {
      "name_of_argument_one": "content_of_argument_one",
      "name_of_argument_two": "content_of_argument_two"
    }
}"#;

pub fn generate_chat_prompt(messages: &Vec<Message>, template: &ChatTemplate) -> Result<String> {
    let assistant_template = template.assistant_template.clone();
    let mut tt = TinyTemplate::new();
    tt.add_template("system", template.system_template.as_str())
        .expect("Could not add system template");
    tt.add_template("user", template.user_template.as_str())
        .expect("Could not add user template");
    tt.add_template("assistant", assistant_template.as_str())
        .expect("Could not add assistant template");
    tt.add_template("tool", template.tool_template.as_str())
        .expect("Could not add tool template");
    tt.add_template(
        "tool_calls",
        template
            .tool_response_template
            .as_ref()
            .unwrap_or(&assistant_template),
    )
    .expect("Could not add tool template");
    tt.set_default_formatter(&tinytemplate::format_unescaped);
    let mut prompt = String::new();
    for message in messages {
        if message.role == "tool" {
            let ctx = &ToolsTemplateContext {
                tools: message.content.clone(),
                template: TOOL_TEMPLATE_JSON.to_string(),
            };
            prompt.push_str(
                &tt.render("tool", &ctx)
                    .expect("Could not render tool template"),
            );
            continue;
        }
        let ctx = &TemplateContext {
            content: message.content.clone(),
        };
        match message.role.as_str() {
            "user" => prompt.push_str(
                &tt.render("user", &ctx)
                    .expect("Could not render user template"),
            ),
            "assistant" => prompt.push_str(
                &tt.render("assistant", &ctx)
                    .expect("Could not render assistant template"),
            ),
            "system" => prompt.push_str(
                &tt.render("system", &ctx)
                    .expect("Could not render system template"),
            ),
            "tool_response" => prompt.push_str(
                &tt.render("tool_calls", &ctx)
                    .expect("Could not render tool_calls template"),
            ),
            _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid role")),
        }
    }
    prompt.push_str(&template.assistant_prompt_template);
    Ok(prompt)
}
