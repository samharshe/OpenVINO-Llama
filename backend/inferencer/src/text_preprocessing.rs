use tokenizers::tokenizer::Tokenizer;

/// Check if data looks like UTF-8 text
pub fn is_text(data: &[u8]) -> bool {
    std::str::from_utf8(data).is_ok()
}

/// Tokenize text input using the provided tokenizer
pub fn tokenize_text(text: &str, tokenizer: &Tokenizer) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    
    Ok(encoding.get_ids().to_vec())
}

/// Create attention mask for the given token IDs (all 1s for real tokens)
pub fn create_attention_mask(token_ids: &[u32]) -> Vec<i32> {
    vec![1; token_ids.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_text() {
        assert!(is_text(b"Hello, world!"));
        assert!(is_text("こんにちは".as_bytes()));
        assert!(!is_text(&[0xFF, 0xD8, 0xFF])); // JPEG magic bytes
    }

    #[test]
    fn test_create_attention_mask() {
        let token_ids = vec![101, 2023, 2003, 1037, 3231, 102];
        let mask = create_attention_mask(&token_ids);
        assert_eq!(mask.len(), token_ids.len());
        assert!(mask.iter().all(|&x| x == 1));
    }
}