use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wasmtime::{Engine, Module};
use tokio::sync::broadcast;

use crate::runtime::WasmInstance;

#[derive(Debug)]
#[allow(dead_code)]
pub enum ValidationError {
    InvalidFormat,
    InvalidSize,
    InvalidDimensions,
    MalformedData,
}

#[derive(Debug)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
pub enum InferenceError {
    PreprocessingFailed(String),
    ModelLoadFailed(String),
    InferenceFailed(String),
    PostprocessingFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Image,
    Text,
    Multimodal,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
}

#[allow(dead_code)]
pub trait ModelConfig: Send + Sync {
    /// Validates input data format and size
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    
    /// Performs complete inference pipeline: preprocess -> infer -> postprocess
    fn infer(&self, data: &[u8]) -> Result<serde_json::Value, InferenceError>;
    
    /// Returns model metadata for responses
    fn model_info(&self) -> ModelInfo;
}

pub struct ImageModelConfig {
    engine: Arc<Engine>,
    module: Arc<Module>,
    _log_sender: broadcast::Sender<String>,
    name: String,
    version: String,
}

impl ImageModelConfig {
    pub fn new(
        engine: Arc<Engine>, 
        module: Arc<Module>,
        log_sender: broadcast::Sender<String>,
        name: String,
        version: String,
    ) -> Self {
        Self {
            engine,
            module,
            _log_sender: log_sender,
            name,
            version,
        }
    }
}

impl ModelConfig for ImageModelConfig {
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        // Check for JPEG magic bytes: FF D8 FF
        if data.len() < 3 {
            return Err(ValidationError::InvalidSize);
        }
        
        if data[0] != 0xFF || data[1] != 0xD8 || data[2] != 0xFF {
            return Err(ValidationError::InvalidFormat);
        }
        
        Ok(())
    }
    
    fn infer(&self, data: &[u8]) -> Result<serde_json::Value, InferenceError> {
        // For now, data is already tensor bytes (preprocessing done by server)
        // TODO: Move preprocessing here when we refactor preprocessing pipeline
        
        // Create WASM instance and run inference
        let mut wasm_instance = WasmInstance::new(self.engine.clone(), self.module.clone())
            .map_err(|e| InferenceError::ModelLoadFailed(e.to_string()))?;
        
        let result = wasm_instance.infer(data.to_vec())
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;
        
        Ok(result)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            model_type: ModelType::Image,
        }
    }
}

pub struct TextModelConfig {
    name: String,
    version: String,
}

impl TextModelConfig {
    #[allow(dead_code)]
    pub fn new(name: String, version: String) -> Self {
        Self { name, version }
    }
}

impl ModelConfig for TextModelConfig {
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        // Validate UTF-8 text
        if std::str::from_utf8(data).is_err() {
            return Err(ValidationError::InvalidFormat);
        }
        
        // Check reasonable size limits (max 10KB for demo)
        if data.len() > 10240 {
            return Err(ValidationError::InvalidSize);
        }
        
        Ok(())
    }
    
    fn infer(&self, data: &[u8]) -> Result<serde_json::Value, InferenceError> {
        let input_text = std::str::from_utf8(data)
            .map_err(|e| InferenceError::PreprocessingFailed(e.to_string()))?;
        
        // Mock response following the required JSON structure
        let response = serde_json::json!({
            "output": format!("Mock response to: {}", input_text),
            "metadata": {
                "token_count": input_text.split_whitespace().count(),
                "inference_time_ms": 42,
                "temperature": 0.7
            },
            "model_info": {
                "name": self.name,
                "version": self.version,
                "model_type": "Text"
            }
        });
        
        Ok(response)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            model_type: ModelType::Text,
        }
    }
}