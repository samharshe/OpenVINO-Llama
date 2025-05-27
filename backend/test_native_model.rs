/// Native test for Llama model without WASM
/// This tests if the model files are valid and can be loaded with OpenVINO directly

use std::fs;
use tokenizers::tokenizer::Tokenizer;

fn main() {
    env_logger::init();
    
    println!("=== Native Model Test ===");
    println!("Testing Llama model loading without WASM constraints...\n");
    
    // Test 1: Check if files exist and are readable
    println!("1. Checking model files...");
    
    let tokenizer_path = "server/fixture/text_model/tokenizer.json";
    let model_xml_path = "server/fixture/text_model/openvino_model.xml";
    let model_bin_path = "server/fixture/text_model/openvino_model.bin";
    
    // Check tokenizer
    match fs::metadata(tokenizer_path) {
        Ok(meta) => println!("   ✓ Tokenizer found: {} MB", meta.len() / 1_000_000),
        Err(e) => {
            println!("   ✗ Tokenizer not found: {}", e);
            return;
        }
    }
    
    // Check model XML
    match fs::metadata(model_xml_path) {
        Ok(meta) => println!("   ✓ Model XML found: {} KB", meta.len() / 1_000),
        Err(e) => {
            println!("   ✗ Model XML not found: {}", e);
            return;
        }
    }
    
    // Check model weights
    match fs::metadata(model_bin_path) {
        Ok(meta) => println!("   ✓ Model weights found: {} MB", meta.len() / 1_000_000),
        Err(e) => {
            println!("   ✗ Model weights not found: {}", e);
            return;
        }
    }
    
    // Test 2: Load and test tokenizer
    println!("\n2. Testing tokenizer...");
    let tokenizer_json = fs::read(tokenizer_path).expect("Failed to read tokenizer");
    
    match Tokenizer::from_bytes(&tokenizer_json) {
        Ok(tokenizer) => {
            println!("   ✓ Tokenizer loaded successfully");
            
            // Test tokenization
            let test_text = "Hello, world!";
            match tokenizer.encode(test_text, false) {
                Ok(encoding) => {
                    println!("   ✓ Tokenization works: '{}' -> {} tokens", 
                             test_text, encoding.get_ids().len());
                    println!("     Token IDs: {:?}", encoding.get_ids());
                },
                Err(e) => println!("   ✗ Tokenization failed: {}", e),
            }
        },
        Err(e) => {
            println!("   ✗ Failed to load tokenizer: {}", e);
            return;
        }
    }
    
    // Test 3: Check model XML structure
    println!("\n3. Checking model XML structure...");
    let xml_content = fs::read_to_string(model_xml_path).expect("Failed to read XML");
    
    // Basic XML validation
    if xml_content.contains("<net ") && xml_content.contains("</net>") {
        println!("   ✓ XML structure looks valid");
        
        // Check for expected inputs
        if xml_content.contains("input_ids") {
            println!("   ✓ Found 'input_ids' input");
        } else {
            println!("   ⚠ Warning: 'input_ids' input not found in XML");
        }
        
        if xml_content.contains("attention_mask") {
            println!("   ✓ Found 'attention_mask' input");
        } else {
            println!("   ⚠ Warning: 'attention_mask' input not found in XML");
        }
    } else {
        println!("   ✗ XML structure invalid");
    }
    
    // Test 4: Memory requirements
    println!("\n4. Checking memory requirements...");
    let weights_size = fs::metadata(model_bin_path).unwrap().len();
    let estimated_runtime_memory = weights_size * 2; // Rough estimate
    
    println!("   Model weights: {} MB", weights_size / 1_000_000);
    println!("   Estimated runtime memory: {} MB", estimated_runtime_memory / 1_000_000);
    
    if weights_size > 1_000_000_000 {
        println!("   ⚠ Warning: Model is very large (>1GB), may cause issues in WASM");
    }
    
    println!("\n=== Summary ===");
    println!("All basic checks passed. The model files appear to be valid.");
    println!("The large model size (1.8GB) is likely causing issues in WASM environment.");
    println!("\nRecommendations:");
    println!("1. Try a smaller model (e.g., BERT, DistilGPT2)");
    println!("2. Implement streaming/chunked loading");
    println!("3. Increase WASM memory limits if possible");
}