#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from transformers import GPT2TokenizerFast

def test_gpt2_tflite():
    """Test the GPT-2 TFLite model with basic inference."""
    
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Load the TFLite model
    model_path = "ov_ir/gpt2-8bit.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("=== Model Details ===")
    print(f"Model path: {model_path}")
    print(f"Number of inputs: {len(input_details)}")
    print(f"Number of outputs: {len(output_details)}")
    
    print("\n=== Input Details ===")
    for i, detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Index: {detail['index']}")
    
    print("\n=== Output Details ===")
    for i, detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Index: {detail['index']}")
    
    # Create sample input using tokenizer
    input_shape = input_details[0]['shape']
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
    batch_size, seq_len = input_shape
    if tokens.shape[1] > seq_len:
        tokens = tokens[:, :seq_len]
    elif tokens.shape[1] < seq_len:
        padding = np.zeros((batch_size, seq_len - tokens.shape[1]), dtype=input_details[0]['dtype'])
        tokens = np.concatenate([tokens, padding], axis=1)
    
    sample_input = tokens.astype(input_details[0]['dtype'])
    print(f"Padded input shape: {sample_input.shape}")
    print(f"Padded input: {sample_input[0][:10]}... (showing first 10 tokens)")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample_input)
    
    # Run inference
    print("\nRunning inference...")
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
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
    input_size = np.prod(input_shape) * np.dtype(input_details[0]['dtype']).itemsize
    output_size = np.prod(output_data.shape) * np.dtype(output_data.dtype).itemsize
    print(f"Input tensor size: {input_size} bytes ({input_size/1024/1024:.2f} MB)")
    print(f"Output tensor size: {output_size} bytes ({output_size/1024/1024:.2f} MB)")
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    try:
        test_gpt2_tflite()
        print("SUCCESS: GPT-2 TFLite model test completed!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()