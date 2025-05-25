use serde::{Deserialize, Serialize};
use std::sync::Arc;
use wasmtime::{Engine, Module};
use tokio::sync::broadcast;

use crate::runtime::WasmInstance;
use crate::tensor;

#[derive(Debug)]
pub enum ValidationError {
    InvalidFormat,
    InvalidSize,
    InvalidDimensions,
    MalformedData,
}

#[derive(Debug)]
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
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
}

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
    log_sender: broadcast::Sender<String>,
}

impl ImageModelConfig {
    pub fn new(
        engine: Arc<Engine>, 
        module: Arc<Module>,
        log_sender: broadcast::Sender<String>
    ) -> Self {
        Self {
            engine,
            module,
            log_sender,
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
        // Preprocess: JPEG to tensor
        let tensor_bytes = tensor::jpeg_to_raw_bgr(data.to_vec(), &self.log_sender)
            .map_err(|e| InferenceError::PreprocessingFailed(e.to_string()))?;
        
        // Create WASM instance and run inference
        let mut wasm_instance = WasmInstance::new(self.engine.clone(), self.module.clone())
            .map_err(|e| InferenceError::ModelLoadFailed(e.to_string()))?;
        
        let result = wasm_instance.infer(tensor_bytes)
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;
        
        Ok(result)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "mobilenet_v3_large".to_string(),
            version: "1.0".to_string(),
            model_type: ModelType::Image,
        }
    }
}