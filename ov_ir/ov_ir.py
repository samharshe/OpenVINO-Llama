from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
import torch

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

original_tensor = torch.tensor
original_arange = torch.arange
original_zeros = torch.zeros
original_ones = torch.ones

def tensor_int32(*args, **kwargs):
    if 'dtype' not in kwargs and len(args) > 0:
        try:
            first_elem = args[0][0] if hasattr(args[0], '__getitem__') else args[0]
            if isinstance(first_elem, int):
                kwargs['dtype'] = torch.int32
        except:
            pass
    return original_tensor(*args, **kwargs)

def arange_int32(*args, **kwargs):
    if 'dtype' not in kwargs:
        kwargs['dtype'] = torch.int32
    return original_arange(*args, **kwargs)

def zeros_float32(*args, **kwargs):
    if 'dtype' not in kwargs:
        kwargs['dtype'] = torch.float32
    return original_zeros(*args, **kwargs)

def ones_float32(*args, **kwargs):
    if 'dtype' not in kwargs:
        kwargs['dtype'] = torch.float32
    return original_ones(*args, **kwargs)

torch.tensor = tensor_int32
torch.arange = arange_int32
torch.zeros = zeros_float32
torch.ones = ones_float32

class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        input_ids = input_ids.to(torch.int32)
        outputs = self.model(input_ids, use_cache=False)
        return outputs.logits

try:
    model = AutoModelForCausalLM.from_pretrained(
        "model",
        torch_dtype=torch.float32,
        device_map=None
    )

    wrapped_model = LogitsOnlyWrapper(model).to("cpu").eval()
    
    tokenizer = AutoTokenizer.from_pretrained("model")
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(torch.int32)

    with torch.no_grad():
        ov_model = ov.convert_model(wrapped_model, example_input=input_ids)

    ov.save_model(ov_model, "ov.xml")
    print("✅ Model converted with 32-bit defaults")

finally:
    torch.tensor = original_tensor
    torch.arange = original_arange
    torch.zeros = original_zeros
    torch.ones = original_ones