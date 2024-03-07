use std::sync::Arc;

use serde::Serialize;
use shurbai::types::ModelManager;
use tokio::sync::{mpsc::Sender, Mutex};

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
