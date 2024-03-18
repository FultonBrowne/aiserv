use serde::Serialize;
use shurbai::types::ChatTemplate;
use std::io::{Error, Result};
use tinytemplate::TinyTemplate;

use crate::types::Message;

#[derive(Serialize)]
struct TemplateContext {
    content: String,
}

pub fn generate_chat_prompt(messages: &Vec<Message>, template: &ChatTemplate) -> Result<String> {
    let mut tt = TinyTemplate::new();
    tt.add_template("system", template.system_template.as_str())
        .expect("Could not add system template");
    tt.add_template("user", template.user_template.as_str())
        .expect("Could not add user template");
    tt.add_template("assistant", template.assistant_template.as_str())
        .expect("Could not add assistant template");
    tt.add_template("tool", template.tool_template.as_str())
        .expect("Could not add tool template");
    tt.set_default_formatter(&tinytemplate::format_unescaped);
    let mut prompt = String::new();
    for message in messages {
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
            "tool" => prompt.push_str(
                &tt.render("tool", &ctx)
                    .expect("Could not render system template"),
            ),
            _ => return Err(Error::new(std::io::ErrorKind::InvalidInput, "Invalid role")),
        }
    }
    prompt.push_str(&template.assistant_prompt_template);
    Ok(prompt)
}
