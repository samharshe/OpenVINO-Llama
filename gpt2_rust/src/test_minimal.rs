use tokenizers::tokenizer::Tokenizer;

fn main() {
    // Try creating a tokenizer programmatically first
    println!("Testing tokenizer creation...");
    
    // Create a minimal tokenizer in memory
    let json_str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "hello": 0,
                "world": 1
            },
            "unk_token": "[UNK]"
        }
    }"#;
    
    match Tokenizer::from_str(json_str) {
        Ok(_) => println!("✓ Minimal WordLevel tokenizer loaded!"),
        Err(e) => println!("✗ Failed to load minimal tokenizer: {}", e),
    }
    
    // Now try a minimal BPE tokenizer
    let bpe_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
            "merges": ["h e", "l l"],
            "unk_token": null
        }
    }"#;
    
    match Tokenizer::from_str(bpe_json) {
        Ok(_) => println!("✓ Minimal BPE tokenizer loaded!"),
        Err(e) => println!("✗ Failed to load BPE tokenizer: {}", e),
    }
}