use std::collections::HashMap;
use log::info;
use serde::{Deserialize, Serialize};
use crate::{Model, ImageNetConfig, imagenet_labels};

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
    pub fn infer(&self, data: &[u8], metadata: &ModelMetadata) -> Result<serde_json::Value, String> {
        match self {
            RegisteredModel::ImageNet(model) => {
                // For now, still expect preprocessed tensor
                // TODO: Move preprocessing here
                let tensor = model.tensor_from_raw_data(data)
                    .map_err(|e| format!("Invalid tensor data: {:?}", e))?;
                
                let result = model.run_inference(tensor)?;
                
                let (class_idx, confidence) = match result {
                    Some(res) => (res.0, res.1),
                    None => (0, 0.0)
                };
                
                // Use ImageNet labels for this model type
                let label = imagenet_labels::get_imagenet_label(class_idx);
                
                Ok(serde_json::json!({
                    "output": label,
                    "metadata": {
                        "probability": confidence,
                        "class_index": class_idx
                    },
                    "model_info": {
                        "name": metadata.name,
                        "version": metadata.version,
                        "model_type": metadata.model_type
                    }
                }))
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