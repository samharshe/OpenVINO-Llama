use wasi_nn;
use std::fs;

// Simple println for debugging in WASM
macro_rules! log {
    ($($arg:tt)*) => {
        println!("[WASM] {}", format!($($arg)*));
    };
}

// Test loading the working image model first
#[export_name = "test_wasi_nn_load_image"]
pub extern "C" fn test_wasi_nn_load_image() -> i32 {
    // Try to load the image model via wasi-nn (use fixture paths)
    // Match backend pattern: read XML as string first
    log!("Reading image model XML...");
    let xml_string = match fs::read_to_string("fixture/model.xml") {
        Ok(s) => {
            log!("XML read successfully, length: {} chars", s.len());
            s
        },
        Err(e) => {
            log!("Failed to read XML: {:?}", e);
            return -1;
        }
    };
    
    log!("Reading image model BIN...");
    let bin_bytes = match fs::read("fixture/model.bin") {
        Ok(bytes) => {
            log!("BIN read successfully, size: {} bytes", bytes.len());
            bytes
        },
        Err(e) => {
            log!("Failed to read BIN: {:?}", e);
            return -2;
        }
    };
    
    // Convert string to bytes like backend does
    let xml_bytes = xml_string.into_bytes();
    log!("XML converted to {} bytes", xml_bytes.len());
    
    log!("Calling wasi_nn::load for image model...");
    let graph_ptr = unsafe {
        match wasi_nn::load(
            &[&xml_bytes, &bin_bytes],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        ) {
            Ok(ptr) => {
                log!("SUCCESS! Image model loaded, graph_ptr: {}", ptr);
                ptr
            },
            Err(e) => {
                log!("FAILED to load image model: {:?}", e);
                return -3;
            }
        }
    };
    
    graph_ptr as i32
}

// Export for WASM runtime
#[export_name = "test_wasi_nn_load"]
pub extern "C" fn test_wasi_nn_load() -> i32 {
    // Try to load the text model via wasi-nn (use fixture paths)
    // Match backend pattern: read XML as string first
    log!("Reading text model XML...");
    let xml_string = match fs::read_to_string("fixture/text_model/openvino_model.xml") {
        Ok(s) => {
            log!("Text XML read successfully, length: {} chars", s.len());
            s
        },
        Err(e) => {
            log!("Failed to read text XML: {:?}", e);
            return -1;
        }
    };
    
    log!("Reading text model BIN...");
    let bin_bytes = match fs::read("fixture/text_model/openvino_model.bin") {
        Ok(bytes) => {
            log!("Text BIN read successfully, size: {} bytes", bytes.len());
            bytes
        },
        Err(e) => {
            log!("Failed to read text BIN: {:?}", e);
            return -2;
        }
    };
    
    // Convert string to bytes like backend does
    let xml_bytes = xml_string.into_bytes();
    log!("Text XML converted to {} bytes", xml_bytes.len());
    
    log!("Calling wasi_nn::load for text model...");
    let graph_ptr = unsafe {
        match wasi_nn::load(
            &[&xml_bytes, &bin_bytes],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        ) {
            Ok(ptr) => {
                log!("SUCCESS! Text model loaded, graph_ptr: {}", ptr);
                ptr
            },
            Err(e) => {
                log!("FAILED to load text model: {:?}", e);
                return -3;
            }
        }
    };
    
    graph_ptr as i32
}

// Test inference on IMAGE model first to debug the tensor issue
#[export_name = "test_image_inference"]
pub extern "C" fn test_image_inference(graph_ptr: u32) -> i32 {
    log!("Testing inference on IMAGE model (graph_ptr: {})", graph_ptr);
    
    // Test basic inference with image model
    let context_ptr = unsafe {
        match wasi_nn::init_execution_context(graph_ptr) {
            Ok(ptr) => {
                log!("Execution context created: {}", ptr);
                ptr
            },
            Err(e) => {
                log!("Failed to create context: {:?}", e);
                return -1;
            }
        }
    };
    
    // Image model expects [1, 3, 224, 224] f32 tensor
    let image_size = 1 * 3 * 224 * 224;
    let mut image_data = vec![0.5f32; image_size]; // Dummy gray image
    
    let input_bytes: Vec<u8> = image_data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    
    log!("Image tensor: {} f32 values = {} bytes", image_size, input_bytes.len());
    
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &input_bytes,
    };
    
    // Set input
    unsafe {
        match wasi_nn::set_input(context_ptr, 0, tensor) {
            Ok(_) => log!("Image input tensor set successfully"),
            Err(e) => {
                log!("Failed to set image input: {:?}", e);
                return -2;
            }
        }
    }
    
    // Compute
    unsafe {
        match wasi_nn::compute(context_ptr) {
            Ok(_) => log!("Image inference completed successfully"),
            Err(e) => {
                log!("Failed to compute image inference: {:?}", e);
                return -3;
            }
        }
    }
    
    log!("SUCCESS! Image model inference works");
    0
}

#[export_name = "test_wasi_nn_inference"]
pub extern "C" fn test_wasi_nn_inference(graph_ptr: u32) -> i32 {
    log!("Testing inference on model with graph_ptr: {}", graph_ptr);
    
    // Test basic inference with dummy tokens
    let context_ptr = unsafe {
        match wasi_nn::init_execution_context(graph_ptr) {
            Ok(ptr) => {
                log!("Execution context created: {}", ptr);
                ptr
            },
            Err(e) => {
                log!("Failed to create context: {:?}", e);
                return -1;
            }
        }
    };
    
    // Create dummy input tensor: [1, 64] i32 tokens
    let dummy_tokens: Vec<i32> = vec![1, 2, 3]; // Just a few tokens
    let mut padded_tokens = dummy_tokens.clone();
    padded_tokens.resize(64, 0); // Pad to 64
    
    let input_data: Vec<u8> = padded_tokens
        .iter()
        .flat_map(|&i| i.to_le_bytes())
        .collect();
    
    log!("Input data size: {} bytes", input_data.len());
    log!("Padded tokens: {} tokens", padded_tokens.len());
    log!("Expected: 64 tokens * 4 bytes = 256 bytes");
    
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 64],  // Back to correct [1, 64] shape
        r#type: wasi_nn::TENSOR_TYPE_I32,
        data: &input_data,
    };
    
    log!("Setting input tensor with dimensions {:?}, type I32", tensor.dimensions);
    
    // Set input
    unsafe {
        match wasi_nn::set_input(context_ptr, 0, tensor) {
            Ok(_) => log!("Input tensor set successfully"),
            Err(e) => {
                log!("Failed to set input: {:?}", e);
                return -2;
            }
        }
    }
    
    // Compute
    unsafe {
        match wasi_nn::compute(context_ptr) {
            Ok(_) => log!("Inference completed successfully"),
            Err(e) => {
                log!("Failed to compute: {:?}", e);
                return -3;
            }
        }
    }
    
    // Try to get output
    let output_size = 1 * 64 * 50257; // Expected output size
    let mut output_buffer = vec![0f32; output_size];
    
    unsafe {
        match wasi_nn::get_output(
            context_ptr,
            0,
            output_buffer.as_mut_ptr() as *mut u8,
            (output_buffer.len() * std::mem::size_of::<f32>())
                .try_into()
                .unwrap(),
        ) {
            Ok(_) => log!("Output retrieved successfully"),
            Err(e) => {
                log!("Failed to get output: {:?}", e);
                return -4;
            }
        }
    }
    
    // Success - return output buffer size as verification
    output_buffer.len() as i32
}