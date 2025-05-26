use std::collections::HashMap;
use log::info;
use serde::{Deserialize, Serialize};
use crate::{Model, ImageNetConfig};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: String,
}

pub enum RegisteredModel {
    ImageNet(Model<ImageNetConfig>),
    // Future: Text(Model<TextModelConfig>),
}

impl RegisteredModel {
    pub fn infer(&self, data: &[u8]) -> Result<serde_json::Value, String> {
        match self {
            RegisteredModel::ImageNet(model) => {
                // For now, still expect preprocessed tensor
                // TODO: Move preprocessing here
                let tensor = model.tensor_from_raw_data(data)
                    .map_err(|e| format!("Invalid tensor data: {:?}", e))?;
                
                let result = model.run_inference(tensor)?;
                
                match result {
                    Some(res) => Ok(serde_json::json!({
                        "class_index": res.0,
                        "confidence": res.1
                    })),
                    None => Ok(serde_json::json!({
                        "class_index": 0,
                        "confidence": 0.0
                    }))
                }
            }
        }
    }
}

pub struct ModelRegistry {
    models: HashMap<u32, (RegisteredModel, ModelMetadata)>,
    next_id: u32,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn register_model(&mut self, model: RegisteredModel, metadata: ModelMetadata) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        
        info!("Registering model '{}' v{} with ID {}", metadata.name, metadata.version, id);
        self.models.insert(id, (model, metadata));
        id
    }
    
    pub fn get_model(&self, id: u32) -> Option<&(RegisteredModel, ModelMetadata)> {
        self.models.get(&id)
    }
    
    pub fn list_models(&self) -> Vec<(u32, &ModelMetadata)> {
        self.models.iter()
            .map(|(id, (_, metadata))| (*id, metadata))
            .collect()
    }
}