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

impl From<ValidationError> for InferenceError {
    fn from(err: ValidationError) -> Self {
        InferenceError::PreprocessingFailed(format!("Validation failed: {:?}", err))
    }
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
    log_sender: broadcast::Sender<String>,
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
            log_sender,
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
        // Validate input first
        self.validate_input(data)?;
        
        // Pass raw JPEG directly to WASM (WASM handles preprocessing now)
        self.log_sender.send("[ImageModelConfig] Passing raw JPEG to WASM for inference.".to_string()).ok();
        
        // Create WASM instance and run inference
        self.log_sender.send("[ImageModelConfig] Creating WASM instance for inference.".to_string()).ok();
        let mut wasm_instance = WasmInstance::new(self.engine.clone(), self.module.clone())
            .map_err(|e| InferenceError::ModelLoadFailed(e.to_string()))?;
        
        // Run inference on raw JPEG data
        self.log_sender.send("[ImageModelConfig] Running inference on raw JPEG data.".to_string()).ok();
        let result = wasm_instance.infer(data.to_vec())
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;
        
        self.log_sender.send("[ImageModelConfig] Inference complete, returning result.".to_string()).ok();
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
    engine: Arc<Engine>,
    module: Arc<Module>,
    log_sender: broadcast::Sender<String>,
    name: String,
    version: String,
}

impl TextModelConfig {
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
            log_sender,
            name,
            version,
        }
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
        // Validate input first
        self.validate_input(data)?;
        
        // Pass raw UTF-8 text directly to WASM (WASM handles tokenization now)
        self.log_sender.send("[TextModelConfig] Passing raw text to WASM for inference.".to_string()).ok();
        
        // Create WASM instance and run inference
        self.log_sender.send("[TextModelConfig] Creating WASM instance for text inference.".to_string()).ok();
        let mut wasm_instance = WasmInstance::new(self.engine.clone(), self.module.clone())
            .map_err(|e| InferenceError::ModelLoadFailed(e.to_string()))?;
        
        // Run inference on raw text data
        self.log_sender.send("[TextModelConfig] Running inference on text data.".to_string()).ok();
        let result = wasm_instance.infer(data.to_vec())
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;
        
        self.log_sender.send("[TextModelConfig] Text inference complete, returning result.".to_string()).ok();
        Ok(result)
    }
    
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            version: self.version.clone(),
            model_type: ModelType::Text,
        }
    }
}