#!/usr/bin/env python3

import openvino as ov
import numpy as np
from transformers import GPT2TokenizerFast

def test_gpt2_openvino():
    """Test the GPT-2 OpenVINO model with basic inference."""
    
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Load the OpenVINO model
    model_path = "ov_ir/openvino_model.xml"
    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model, "CPU")
    
    # Get input and output details
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    print("=== Model Details ===")
    print(f"Model path: {model_path}")
    print(f"Input shape: {input_layer.shape}")
    print(f"Input type: {input_layer.element_type}")
    print(f"Output shape: {output_layer.shape}")
    print(f"Output type: {output_layer.element_type}")
    
    # Create sample input using tokenizer
    input_shape = input_layer.shape
    print(f"\n=== Test Inference ===")
    print(f"Input shape: {input_shape}")
    
    # Use a meaningful prompt
    prompt = "The quick brown fox"
    print(f"Input prompt: '{prompt}'")
    
    # Tokenize the input
    tokens = tokenizer.encode(prompt, return_tensors="np")
    print(f"Tokenized: {tokens[0].tolist()}")
    print(f"Tokens as text: {[tokenizer.decode([t]) for t in tokens[0]]}")
    
    # Pad or truncate to fit model input shape
    batch_size, seq_len = input_shape[0], input_shape[1]
    if tokens.shape[1] > seq_len:
        tokens = tokens[:, :seq_len]
    elif tokens.shape[1] < seq_len:
        padding = np.zeros((batch_size, seq_len - tokens.shape[1]), dtype=np.int32)
        tokens = np.concatenate([tokens, padding], axis=1)
    
    sample_input = tokens.astype(np.int32)
    print(f"Padded input shape: {sample_input.shape}")
    print(f"Padded input: {sample_input[0][:10]}... (showing first 10 tokens)")
    
    # Run inference
    print("\nRunning inference...")
    result = compiled_model([sample_input])
    output_data = result[output_layer]
    
    print(f"Output shape: {output_data.shape}")
    print(f"Output dtype: {output_data.dtype}")
    print(f"Output min/max: {output_data.min():.4f} / {output_data.max():.4f}")
    
    # Decode the output to see generated text
    if len(output_data.shape) == 3:  # [batch, seq, vocab]
        # Get logits for the last non-padded position
        input_length = len(tokens[0])
        last_logits = output_data[0, input_length - 1, :]  # Last input position
        
        # Get top 5 predicted tokens
        top_k = 5
        top_indices = np.argsort(last_logits)[-top_k:][::-1]
        
        # Apply temperature scaling and softmax more safely
        scaled_logits = last_logits[top_indices] - np.max(last_logits[top_indices])
        exp_logits = np.exp(scaled_logits)
        top_probs = exp_logits / np.sum(exp_logits)
        
        print(f"\n=== Next Token Predictions ===")
        print(f"Input text: '{prompt}'")
        print(f"Top {top_k} next token predictions:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            token_text = tokenizer.decode([idx])
            print(f"  {i+1}. Token {idx}: '{token_text}' (prob: {prob:.3f})")
        
        # Generate next token (greedy)
        next_token = top_indices[0]
        generated_text = prompt + tokenizer.decode([next_token])
        print(f"Generated text (1 token): '{generated_text}'")
        
    elif len(output_data.shape) == 2:  # [batch, vocab]
        top_k = 5
        top_indices = np.argsort(output_data[0])[-top_k:][::-1]
        print(f"Top {top_k} predicted tokens: {[tokenizer.decode([idx]) for idx in top_indices]}")
    
    print("\n=== Memory Usage ===")
    # Estimate memory usage
    input_size = np.prod(input_shape) * 4  # int32 = 4 bytes
    output_size = np.prod(output_data.shape) * 4  # float32 = 4 bytes
    print(f"Input tensor size: {input_size} bytes ({input_size/1024/1024:.2f} MB)")
    print(f"Output tensor size: {output_size} bytes ({output_size/1024/1024:.2f} MB)")
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    try:
        test_gpt2_openvino()
        print("SUCCESS: GPT-2 OpenVINO model test completed!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()