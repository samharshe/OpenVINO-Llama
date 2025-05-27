use std::collections::HashMap;
use log::info;
use serde::{Deserialize, Serialize};
use crate::{Model, ImageNetConfig, TextModelConfig, imagenet_labels, text_preprocessing};

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: String,
}

pub enum RegisteredModel {
    ImageNet(Model<ImageNetConfig>),
    Text {
        model: Model<TextModelConfig>,
        tokenizer: tokenizers::tokenizer::Tokenizer,
    },
}

impl RegisteredModel {
    pub fn infer(&self, data: &[u8], metadata: &ModelMetadata) -> Result<serde_json::Value, String> {
        match self {
            RegisteredModel::ImageNet(model) => {
                // Detect input type and handle accordingly
                let result = if crate::preprocessing::is_jpeg(data) {
                    // Handle raw JPEG input
                    info!("Detected JPEG input, preprocessing image");
                    model.infer_from_jpeg(data)?
                } else if crate::preprocessing::is_tensor(data) {
                    // Handle preprocessed tensor (backward compatibility)
                    info!("Detected tensor input, using preprocessed data");
                    let tensor = model.tensor_from_raw_data(data)
                        .map_err(|e| format!("Invalid tensor data: {:?}", e))?;
                    model.run_inference(tensor)?
                } else {
                    return Err("Invalid input format: expected JPEG or preprocessed tensor".to_string());
                };
                
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
            RegisteredModel::Text { model, tokenizer } => {
                // Check if input is valid UTF-8 text
                if !text_preprocessing::is_text(data) {
                    return Err("Invalid input: expected UTF-8 text".to_string());
                }
                
                info!("Processing text input");
                let output = model.infer_from_text(data, tokenizer)?;
                
                // Count tokens for metadata
                let text = std::str::from_utf8(data).unwrap(); // Safe because we validated above
                let encoding = tokenizer.encode(text, false)
                    .map_err(|e| format!("Tokenization failed: {}", e))?;
                let token_count = encoding.get_ids().len();
                
                Ok(serde_json::json!({
                    "output": output,
                    "metadata": {
                        "token_count": token_count,
                        "inference_time_ms": 0, // Placeholder - would need timing
                        "temperature": 0.7 // Placeholder - would come from model config
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