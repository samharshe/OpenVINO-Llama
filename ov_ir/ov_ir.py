from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "model_path"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)