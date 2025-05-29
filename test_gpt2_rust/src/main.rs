use anyhow::Result;
use tokenizers::Tokenizer;
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
use std::fs;

fn load_tokenizer() -> Result<Tokenizer> {
    println!("Loading tokenizer...");
    let tokenizer_bytes = fs::read("../ov_ir/tokenizer.json")?;
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("✓ Tokenizer loaded successfully");
    Ok(tokenizer)
}

fn tokenize_input(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
    let tokens = encoding.get_ids();
    
    // Convert to i32 and pad to 64 tokens
    let mut padded_tokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    padded_tokens.resize(64, 0);
    
    println!("Input: '{}'", text);
    println!("Tokens: {:?}", &tokens);
    println!("Padded to 64 tokens");
    
    Ok(padded_tokens)
}

fn setup_model() -> Result<(Core, openvino::CompiledModel)> {
    println!("\nSetting up OpenVINO model...");
    
    // Create OpenVINO Core
    let mut core = Core::new()?;
    println!("✓ OpenVINO Core created");
    
    // Read model
    let model_path = "../ov_ir/openvino_model.xml";
    let model = core.read_model_from_file(model_path, "../ov_ir/openvino_model.bin")?;
    println!("✓ Model loaded from {}", model_path);
    
    // Compile model for CPU
    let compiled = core.compile_model(&model, DeviceType::CPU)?;
    println!("✓ Model compiled for CPU");
    
    Ok((core, compiled))
}

fn run_inference(compiled_model: &mut openvino::CompiledModel, input_tokens: Vec<i32>) -> Result<Vec<f32>> {
    // Create input tensor
    let shape = Shape::new(&[1, 64])?;
    let mut input_tensor = Tensor::new(ElementType::I32, &shape)?;
    
    // Debug tensor allocation
    let raw_data = input_tensor.get_raw_data_mut()?;
    println!("Tensor allocated size: {} bytes, expected: {} bytes", 
             raw_data.len(), 64 * 4);
    
    if raw_data.len() != 64 * 4 {
        return Err(anyhow::anyhow!("Tensor size mismatch: got {} bytes, expected {} bytes", 
                                   raw_data.len(), 64 * 4));
    }
    
    // Set tensor data
    unsafe {
        let tensor_ptr = raw_data.as_mut_ptr() as *mut i32;
        let tensor_slice = std::slice::from_raw_parts_mut(tensor_ptr, 64);
        tensor_slice.copy_from_slice(&input_tokens);
    }
    println!("✓ Tensor data copied successfully");
    
    // Create inference request and run
    println!("Creating inference request...");
    let mut infer_request = compiled_model.create_infer_request()?;
    println!("✓ Inference request created");
    
    println!("Setting input tensor...");
    infer_request.set_input_tensor_by_index(0, &input_tensor)?;
    println!("✓ Input tensor set");
    
    println!("Running inference...");
    infer_request.infer()?;
    println!("✓ Inference completed");
    
    // Get output
    println!("Getting output tensor by index...");
    let output_tensor = infer_request.get_output_tensor_by_index(0)?;
    println!("✓ Output tensor retrieved");
    
    println!("Converting output data...");
    let output_data: Vec<f32> = output_tensor.get_data()?.to_vec();
    println!("✓ Output data converted, size: {}", output_data.len());
    
    Ok(output_data)
}

fn get_next_token(logits: &[f32]) -> usize {
    let vocab_size = 50257;
    let seq_len = 64;
    
    // Get logits for the last token position (position 63)
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_logits = &logits[last_pos_start..last_pos_start + vocab_size];
    
    // Find argmax (greedy decoding)
    let mut max_idx = 0;
    let mut max_val = last_logits[0];
    
    for (idx, &val) in last_logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    
    // Show top 2 tokens to see variation
    let second_idx = last_logits.iter().enumerate()
        .filter(|(i, _)| *i != max_idx)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(1);
    
    println!("Token {}: {:.6}, Token {}: {:.6}", 
             max_idx, max_val, second_idx, last_logits[second_idx]);
    
    max_idx
}

fn generate_text(
    tokenizer: &Tokenizer,
    compiled_model: &mut openvino::CompiledModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    println!("\n=== Generating text ===");
    println!("Prompt: '{}'", prompt);
    
    // Tokenize prompt
    let mut tokens = tokenize_input(tokenizer, prompt)?;
    let mut generated_tokens = Vec::new();
    
    // Generate tokens one by one
    for i in 0..max_tokens {
        println!("\nStep {}:", i + 1);
        
        // Run inference
        let logits = run_inference(compiled_model, tokens.clone())?;
        
        // Get next token
        let next_token = get_next_token(&logits);
        generated_tokens.push(next_token as u32);
        
        // Decode and show progress
        let decoded = tokenizer.decode(&generated_tokens, false)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
        println!("Generated token: {} -> Current text: '{}'", next_token, decoded);
        
        // Update input for next iteration (sliding window)
        for j in 0..63 {
            tokens[j] = tokens[j + 1];
        }
        tokens[63] = next_token as i32;
        
        // Stop if we hit end token
        if next_token == 50256 {
            println!("Hit end token, stopping generation");
            break;
        }
    }
    
    // Decode final result
    let prompt_tokens: Vec<u32> = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode for final decode: {}", e))?
        .get_ids().to_vec();
    let full_tokens: Vec<u32> = prompt_tokens.into_iter()
        .chain(generated_tokens.into_iter()).collect();
    let final_text = tokenizer.decode(&full_tokens, false)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    
    Ok(final_text)
}

fn main() -> Result<()> {
    println!("=== GPT-2 Rust Inference Demo ===\n");
    
    // Load tokenizer
    let tokenizer = load_tokenizer()?;
    
    // Setup model
    let (_core, mut compiled_model) = setup_model()?;
    
    // Test multiple prompts to see logit variation
    let prompts = [
        "Hello world",
        "The quick brown fox", 
        "In 1776",
        "Python is a",
        "Machine learning"
    ];
    
    println!("=== Testing Logit Variation ===");
    for prompt in &prompts {
        println!("\nPrompt: '{}'", prompt);
        let tokens = tokenize_input(&tokenizer, prompt)?;
        let logits = run_inference(&mut compiled_model, tokens)?;
        let _token = get_next_token(&logits);
    }
    
    println!("\n=== Full Generation Test ===");
    let prompt = "The quick brown fox";
    let generated = generate_text(&tokenizer, &mut compiled_model, prompt, 3)?;
    
    println!("\n=== Final Result ===");
    println!("Input: '{}'", prompt);
    println!("Output: '{}'", generated);
    
    Ok(())
}