use std::sync::Arc;

use serde::Serialize;
use shurbai::types::ModelManager;
use tokio::sync::{mpsc::Sender, Mutex};

use crate::types::XmlState;

pub fn has_model(model_manager: &ModelManager, model_name: &String) -> bool {
    model_manager.models.contains_key(model_name)
}

#[macro_export]
macro_rules! get_model {
    ($model_manager:expr, $model_name:expr) => {
        $model_manager
            .models
            .get($model_name)
            .expect("Model not found")
    };
}

pub fn send_to_stream<T>(tx_arc: Arc<Mutex<Sender<T>>>, body: &T)
where
    T: Serialize + Clone + Send + 'static,
{
    let tx_clone = tx_arc.clone();
    let body_clone = body.clone();
    tokio::spawn(async move {
        tx_clone
            .clone()
            .lock()
            .await
            .send(body_clone)
            .await
            .expect("Failed to stream");
    });
}

pub fn process_xml_token(xml_state: &mut XmlState, token: &str) -> Option<String> {
    if token.contains("<") && !xml_state.in_xml_body {
        xml_state.in_xml_tag = true;
        xml_state.current_accumulated_tag.clear();
        xml_state.halt_output = true;
    }
    if xml_state.in_xml_tag {
        xml_state.current_accumulated_tag.push_str(token);
        if xml_state.current_accumulated_tag.ends_with(">") {
            xml_state.in_xml_tag = false;
            if xml_state.current_accumulated_tag.starts_with("</") {
                // Closing tag detected
                xml_state.in_xml_body = false;
                let content = xml_state.xml_body_content.to_string();
                xml_state.halt_output = false;
                xml_state.xml_body_content.clear(); // Clear the content for the next tag
                return Some(content);
            } else {
                // Opening tag detected
                xml_state.in_xml_body = true;
                xml_state.xml_tag_name = xml_state.current_accumulated_tag.to_string();
            }
            xml_state.current_accumulated_tag.clear();
        }
    } else if xml_state.in_xml_body {
        // Check if the current token is part of a closing tag
        if token.contains("</") {
            xml_state.in_xml_tag = true; // Start accumulating tokens for the closing tag
            xml_state.current_accumulated_tag.push_str(token);
        } else {
            xml_state.xml_body_content.push_str(token);
        }
    }

    return None;
}
