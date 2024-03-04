use serde::{Deserialize, Serialize};
use shurbai::types::ModelDefinition;

#[derive(Serialize, Deserialize)]
pub struct Message{
    pub message: String
}


/// Config structs:
#[derive(Serialize, Deserialize)]
pub struct Config {
 //   pub host: String, // for when I make this deployable
    pub models: Vec<ModelDefinition>,
}