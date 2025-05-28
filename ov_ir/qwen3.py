import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino.runtime as ov
from openvino.runtime import Type, PartialShape, Core

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
pt_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
pt_model.eval()

# prepare a small int32 example for tracing
example_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 8), dtype=torch.int32)
example_attention_mask = torch.ones((1, 8), dtype=torch.int32)

# convert, forcing input_ids and attention_mask to i32
ov_model = ov.convert_model(
    pt_model,
    input=[
        ("input_ids",      PartialShape([-1, -1]), Type.i32),
        ("attention_mask", PartialShape([-1, -1]), Type.i32),
    ],
    example_input={"input_ids": example_input_ids, "attention_mask": example_attention_mask}
)

# save your IR
ov.save_model(ov_model, "qwen3_int32_inputs.xml", "qwen3_int32_inputs.bin")

# (Optional) compile & test
core = Core()
compiled = core.compile_model("qwen3_int32_inputs.xml", "CPU")
out = compiled({"input_ids": example_input_ids.numpy(),
               "attention_mask": example_attention_mask.numpy()})
