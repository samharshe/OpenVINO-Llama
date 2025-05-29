#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def test_gpt2_tflite():
    """Test the GPT-2 TFLite model with basic inference."""
    
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
    
    # Create sample input (assuming token IDs)
    # For GPT-2, typical input is token IDs with shape [batch_size, sequence_length]
    input_shape = input_details[0]['shape']
    print(f"\n=== Test Inference ===")
    print(f"Input shape: {input_shape}")
    
    # Create dummy input - simple sequence of token IDs
    if len(input_shape) == 2:
        batch_size, seq_len = input_shape
        # Use some common token IDs (e.g., for "Hello world")
        sample_input = np.array([[15496, 995]], dtype=input_details[0]['dtype'])
        if sample_input.shape[1] < seq_len:
            # Pad with zeros if needed
            padding = np.zeros((batch_size, seq_len - sample_input.shape[1]), dtype=input_details[0]['dtype'])
            sample_input = np.concatenate([sample_input, padding], axis=1)
    else:
        # Fallback: create zeros with the expected shape
        sample_input = np.zeros(input_shape, dtype=input_details[0]['dtype'])
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample input dtype: {sample_input.dtype}")
    print(f"Sample input: {sample_input}")
    
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
    
    # Show first few logits if it's a typical language model output
    if len(output_data.shape) == 3:  # [batch, seq, vocab]
        print(f"First 10 logits for last position: {output_data[0, -1, :10]}")
    elif len(output_data.shape) == 2:  # [batch, vocab]
        print(f"First 10 logits: {output_data[0, :10]}")
    
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