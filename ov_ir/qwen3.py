import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from openvino import Type, PartialShape, Core
import os
from pathlib import Path

# Model configuration
model_id = "Qwen/Qwen3-0.6B"
output_dir = Path("../backend/server/fixture/text_model")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
pt_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,  # Use FP32 for conversion
    device_map="cpu"  # Keep on CPU for conversionno
)
pt_model.eval()

# Create example inputs with INT32 dtype
seq_length = 8
example_input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length), dtype=torch.int32)
example_attention_mask = torch.ones((1, seq_length), dtype=torch.int32)

# Check if model expects position_ids
dummy_input = {"input_ids": example_input_ids.long(), "attention_mask": example_attention_mask.long()}
with torch.no_grad():
    try:
        # Try with position_ids
        example_position_ids = torch.arange(seq_length, dtype=torch.int32).unsqueeze(0)
        dummy_input["position_ids"] = example_position_ids.long()
        _ = pt_model(**dummy_input)
        needs_position_ids = True
        print("Model accepts position_ids")
    except:
        # Model doesn't need position_ids
        needs_position_ids = False
        print("Model doesn't use position_ids")

# Prepare inputs for conversion
inputs = [
    ("input_ids", PartialShape([1, -1]), Type.i32),
    ("attention_mask", PartialShape([1, -1]), Type.i32),
]
example_input = {
    "input_ids": example_input_ids,
    "attention_mask": example_attention_mask
}

if needs_position_ids:
    inputs.append(("position_ids", PartialShape([1, -1]), Type.i32))
    example_input["position_ids"] = torch.arange(seq_length, dtype=torch.int32).unsqueeze(0)

print("Converting model to OpenVINO format with INT32 inputs...")

# Disable cache for conversion
pt_model.config.use_cache = False

ov_model = ov.convert_model(
    pt_model,
    input=inputs,
    example_input=example_input
)

# Skip quantization for now - just use the converted model
print("Skipping quantization for minimal model...")
ov_model_compressed = ov_model

# Save the model
output_xml = output_dir / "openvino_model.xml"
output_bin = output_dir / "openvino_model.bin"
print(f"Saving model to {output_xml}")
ov.save_model(ov_model_compressed, str(output_xml))

# Test inference with the converted model
print("\nTesting inference with INT32 inputs...")
compiled_model = core.compile_model(ov_model_compressed, "CPU")
infer_request = compiled_model.create_infer_request()

# Create test inputs
test_text = "Hello, world!"
test_tokens = tokenizer(test_text, return_tensors="pt")
test_input_ids = test_tokens["input_ids"].to(torch.int32)
test_attention_mask = test_tokens["attention_mask"].to(torch.int32)

# Set inputs
infer_request.set_tensor("input_ids", ov.Tensor(test_input_ids.numpy()))
infer_request.set_tensor("attention_mask", ov.Tensor(test_attention_mask.numpy()))
if needs_position_ids:
    test_position_ids = torch.arange(test_input_ids.shape[1], dtype=torch.int32).unsqueeze(0)
    infer_request.set_tensor("position_ids", ov.Tensor(test_position_ids.numpy()))

# Run inference
infer_request.infer()
output_tensor = infer_request.get_output_tensor(0)
logits = output_tensor.data

print(f"Model output shape: {logits.shape}")
print(f"Model size: {os.path.getsize(output_bin) / 1024 / 1024:.2f} MB")
print(f"Vocab size: {tokenizer.vocab_size}")
print("\nConversion complete!")