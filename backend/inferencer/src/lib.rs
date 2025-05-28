use log::{error, warn, info};

pub mod registry;
pub mod imagenet_labels;
pub mod preprocessing;
pub mod text_preprocessing;

#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);

#[derive(Debug)]
pub enum ValidationError {
    InvalidDimensions,
    InvalidDataSize,
    InvalidFormat,
}

pub trait ModelConfig {
    fn output_size(&self) -> usize;
    fn input_dims(&self) -> &[u32];
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    fn tensor_type(&self) -> wasi_nn::TensorType;
}

pub struct ImageNetConfig;

impl ModelConfig for ImageNetConfig {
    fn output_size(&self) -> usize {
        1001 // ImageNet 1000 classes + background
    }
    
    fn input_dims(&self) -> &[u32] {
        &[1, 3, 224, 224] // NCHW format
    }
    
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        let expected_size = self.input_dims().iter().map(|&x| x as usize).product::<usize>() * std::mem::size_of::<f32>();
        if data.len() != expected_size {
            return Err(ValidationError::InvalidDataSize);
        }
        Ok(())
    }
    
    fn tensor_type(&self) -> wasi_nn::TensorType {
        wasi_nn::TENSOR_TYPE_F32
    }
}

pub struct TextModelConfig {
    pub vocab_size: usize,
    pub sequence_length: usize,
    input_dims: Vec<u32>,
}

impl TextModelConfig {
    pub fn new(vocab_size: usize, sequence_length: usize) -> Self {
        Self {
            vocab_size,
            sequence_length,
            input_dims: vec![1, sequence_length as u32],
        }
    }
}

impl ModelConfig for TextModelConfig {
    fn output_size(&self) -> usize {
        self.vocab_size
    }
    
    fn input_dims(&self) -> &[u32] {
        &self.input_dims
    }
    
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        // For text input, we validate UTF-8 strings, not token arrays
        if std::str::from_utf8(data).is_err() {
            return Err(ValidationError::InvalidFormat);
        }
        Ok(())
    }
    
    fn tensor_type(&self) -> wasi_nn::TensorType {
        wasi_nn::TENSOR_TYPE_I32 // Limited by wasi-nn 0.1.0, but data is i64
    }
}

pub struct Model<C: ModelConfig> {
    context_ptr: u32,
    _graph_ptr: u32,
    config: C,
}

// Backward compatibility alias
pub type MobilnetModel = Model<ImageNetConfig>;

impl<C: ModelConfig> Model<C> {
    pub fn from_buffer(xml: Vec<u8>, weights: Vec<u8>, config: C) -> Result<Self, String> {
        info!("Model::from_buffer called with XML size: {}, weights size: {}", 
             xml.len(), weights.len());
        
        info!("Calling wasi_nn::load...");
        let start = std::time::Instant::now();
        
        let _graph_ptr = unsafe {
            wasi_nn::load(
                &[&xml, &weights],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                wasi_nn::EXECUTION_TARGET_CPU,
            )
            .map_err(|e| {
                let err_msg = format!("Failed to load graph: {:?}", e);
                error!("{}", err_msg);
                error!("This happened after {} ms", start.elapsed().as_millis());
                err_msg
            })?
        };
        
        info!("wasi_nn::load completed successfully after {} ms, graph_ptr={}", 
             start.elapsed().as_millis(), _graph_ptr);
        
        info!("Initializing execution context...");
        let context_ptr = unsafe {
            wasi_nn::init_execution_context(_graph_ptr)
                .map_err(|e| {
                    let err_msg = format!("Failed to init execution context: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
        };
        
        info!("Execution context initialized, context_ptr={}", context_ptr);
        info!("Model loaded successfully with output size: {}", config.output_size());
        
        Ok(Self {
            context_ptr,
            _graph_ptr,
            config,
        })
    }

    pub fn run_inference(&self, tensor: wasi_nn::Tensor) -> Result<Option<InferenceResult>, String> {
        unsafe {
            wasi_nn::set_input(self.context_ptr, 0, tensor)
                .map_err(|e| {
                    let err_msg = format!("Failed to set input tensor: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
            ;

            wasi_nn::compute(self.context_ptr)
                .map_err(|e| {
                    let err_msg = format!("Failed to compute inference: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
            ;
        }

        let mut output_buffer = vec![0f32; self.config.output_size()];
        unsafe {
            wasi_nn::get_output(
                self.context_ptr,
                0,
                output_buffer.as_mut_ptr() as *mut u8,
                (output_buffer.len() * std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap(),
            )
            .map_err(|e| {
                let err_msg = format!("Failed to get output: {:?}", e);
                error!("{}", err_msg);
                err_msg
            })?
            ;
        }

        let mut results: Vec<InferenceResult> = output_buffer
            .iter()
            .skip(1)
            .enumerate()
            .map(|(class, prob)| InferenceResult(class, *prob))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(results.first().cloned())
    }

    pub fn tensor_from_raw_data<'a>(&'a self, tensor_data: &'a [u8]) -> Result<wasi_nn::Tensor<'a>, ValidationError> {
        self.config.validate_input(tensor_data)?;
        Ok(wasi_nn::Tensor {
            dimensions: self.config.input_dims(),
            r#type: self.config.tensor_type(),
            data: tensor_data,
        })
    }
}

impl<C: ModelConfig> Drop for Model<C> {
    fn drop(&mut self) {
        // Note: wasi-nn doesn't provide cleanup functions in 0.1.0
        // This is a placeholder for future resource cleanup
        warn!("Model dropped - resources may leak due to wasi-nn limitations");
    }
}

// Backward compatibility methods for MobilnetModel
impl MobilnetModel {
    pub fn from_buffer_compat(xml: Vec<u8>, weights: Vec<u8>) -> Self {
        match Self::from_buffer_result(xml, weights) {
            Ok(model) => model,
            Err(e) => panic!("{}", e), // Maintain original panic behavior for compatibility
        }
    }
    
    pub fn from_buffer_result(xml: Vec<u8>, weights: Vec<u8>) -> Result<Self, String> {
        <Model<ImageNetConfig>>::from_buffer(xml, weights, ImageNetConfig)
    }
    
    pub fn tensor_from_raw_bgr<'a>(&'a self, tensor_data: &'a [u8]) -> wasi_nn::Tensor<'a> {
        match self.tensor_from_raw_data(tensor_data) {
            Ok(tensor) => tensor,
            Err(e) => panic!("Invalid tensor data: {:?}", e), // Maintain original panic behavior
        }
    }
    
    pub fn run_inference_compat(&self, tensor: wasi_nn::Tensor) -> Result<Option<InferenceResult>, String> {
        <Model<ImageNetConfig>>::run_inference(self, tensor)
    }
    
    /// New method to handle raw JPEG input
    pub fn infer_from_jpeg(&self, jpeg_bytes: &[u8]) -> Result<Option<InferenceResult>, String> {
        // Preprocess JPEG to tensor
        let tensor_bytes = crate::preprocessing::jpeg_to_raw_bgr(jpeg_bytes)?;
        
        // Create tensor from preprocessed data
        let tensor = self.tensor_from_raw_data(&tensor_bytes)
            .map_err(|e| format!("Failed to create tensor: {:?}", e))?;
        
        // Run inference
        self.run_inference(tensor)
    }
}

// Text Model Implementation
pub type TextModel = Model<TextModelConfig>;

impl TextModel {
    /// Create a new text model from buffers with tokenizer
    pub fn from_buffer_with_tokenizer(xml: Vec<u8>, weights: Vec<u8>, tokenizer_json: Vec<u8>) -> Result<(Self, tokenizers::tokenizer::Tokenizer), String> {
        info!("Starting text model loading...");
        info!("XML size: {} bytes", xml.len());
        info!("Weights size: {} bytes", weights.len());
        info!("Tokenizer JSON size: {} bytes", tokenizer_json.len());
        
        // Create the tokenizer from JSON bytes
        info!("Creating tokenizer from JSON...");
        let tokenizer = tokenizers::tokenizer::Tokenizer::from_bytes(&tokenizer_json)
            .map_err(|e| {
                error!("Failed to create tokenizer: {}", e);
                format!("Failed to create tokenizer: {}", e)
            })?;
        info!("Tokenizer created successfully");
        
        // Create the model config
        let config = TextModelConfig {
            vocab_size: 32000, // TinyLlama vocab size
            sequence_length: 2048, // TinyLlama supports up to 2048
            input_dims: vec![1, 2048], // [batch_size, sequence_length]
        };
        info!("Created TextModelConfig with vocab_size={}, sequence_length={}", 
             config.vocab_size, config.sequence_length);
        
        info!("Loading model with wasi_nn::load...");
        let model = Model::<TextModelConfig>::from_buffer(xml, weights, config)?;
        info!("Model loaded successfully!");
        
        Ok((model, tokenizer))
    }
    
    /// Infer from raw UTF-8 text input
    pub fn infer_from_text(&self, text_bytes: &[u8], tokenizer: &tokenizers::tokenizer::Tokenizer) -> Result<String, String> {
        // Validate text input
        let text = std::str::from_utf8(text_bytes)
            .map_err(|e| format!("Invalid UTF-8 text: {}", e))?;
        
        info!("Processing text input: {} characters", text.len());
        
        // Tokenize the text
        let token_ids = crate::text_preprocessing::tokenize_text(text, tokenizer)?;
        info!("Tokenized to {} tokens", token_ids.len());
        
        // Create attention mask
        let attention_mask = crate::text_preprocessing::create_attention_mask(&token_ids);
        
        // Create position_ids (0, 1, 2, ..., sequence_length-1)
        let position_ids: Vec<u32> = (0..token_ids.len() as u32).collect();
        
        // Create beam_idx (single beam = [0])
        let beam_idx: Vec<i32> = vec![0];
        
        // Convert to i32 for tensor (casting down from expected i64 due to wasi-nn limitations)
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let attention_mask_i32: Vec<i32> = attention_mask.iter().map(|&mask| mask as i32).collect();
        let position_ids_i32: Vec<i32> = position_ids.iter().map(|&id| id as i32).collect();
        
        info!("Token IDs (first 10): {:?}", &token_ids_i32[..token_ids_i32.len().min(10)]);
        info!("Attention mask (first 10): {:?}", &attention_mask_i32[..attention_mask_i32.len().min(10)]);
        
        // Create input tensors with i32 data
        let input_ids_tensor = wasi_nn::Tensor {
            dimensions: &[1, token_ids_i32.len() as u32],
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: unsafe {
                std::slice::from_raw_parts(
                    token_ids_i32.as_ptr() as *const u8,
                    token_ids_i32.len() * std::mem::size_of::<i32>()
                )
            },
        };
        
        let attention_mask_tensor = wasi_nn::Tensor {
            dimensions: &[1, attention_mask_i32.len() as u32],
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: unsafe {
                std::slice::from_raw_parts(
                    attention_mask_i32.as_ptr() as *const u8,
                    attention_mask_i32.len() * std::mem::size_of::<i32>()
                )
            },
        };
        
        let position_ids_tensor = wasi_nn::Tensor {
            dimensions: &[1, position_ids_i32.len() as u32],
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: unsafe {
                std::slice::from_raw_parts(
                    position_ids_i32.as_ptr() as *const u8,
                    position_ids_i32.len() * std::mem::size_of::<i32>()
                )
            },
        };
        
        let beam_idx_tensor = wasi_nn::Tensor {
            dimensions: &[], // Scalar tensor (no dimensions)
            r#type: wasi_nn::TENSOR_TYPE_I32,
            data: unsafe {
                std::slice::from_raw_parts(
                    beam_idx.as_ptr() as *const u8,
                    beam_idx.len() * std::mem::size_of::<i32>()
                )
            },
        };
        
        // Set inputs (order based on XML: beam_idx=0, position_ids=1, attention_mask=2, input_ids=3)
        unsafe {
            // Set beam_idx (index 0)
            wasi_nn::set_input(self.context_ptr, 0, beam_idx_tensor)
                .map_err(|e| format!("Failed to set beam_idx: {:?}", e))?;
            
            // Set position_ids (index 1)  
            wasi_nn::set_input(self.context_ptr, 1, position_ids_tensor)
                .map_err(|e| format!("Failed to set position_ids: {:?}", e))?;
            
            // Set attention_mask (index 2)
            wasi_nn::set_input(self.context_ptr, 2, attention_mask_tensor)
                .map_err(|e| format!("Failed to set attention_mask: {:?}", e))?;
            
            // Set input_ids (index 3)
            wasi_nn::set_input(self.context_ptr, 3, input_ids_tensor)
                .map_err(|e| format!("Failed to set input_ids: {:?}", e))?;
            
            // Run inference
            wasi_nn::compute(self.context_ptr)
                .map_err(|e| format!("Failed to compute: {:?}", e))?;
        }
        
        // Get output (assuming output is logits)
        let output_size = self.config.vocab_size * token_ids.len();
        let mut output_buffer = vec![0f32; output_size];
        
        unsafe {
            wasi_nn::get_output(
                self.context_ptr,
                0,
                output_buffer.as_mut_ptr() as *mut u8,
                (output_buffer.len() * std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap(),
            )
            .map_err(|e| format!("Failed to get output: {:?}", e))?;
        }
        
        // For now, return a placeholder response
        // In a real implementation, we'd decode the logits back to text
        Ok(format!("Processed {} tokens from input text", token_ids.len()))
    }
}
