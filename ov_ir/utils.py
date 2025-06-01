import torch

def debug_tensor_types(model, input_tensor):
    """Hook to monitor tensor types throughout forward pass"""
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if output.dtype != torch.float32:
                    print(f"⚠️  {name}: {output.dtype}")
            elif isinstance(output, (tuple, list)):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor) and out.dtype != torch.float32:
                        print(f"⚠️  {name}[{i}]: {out.dtype}")
        return hook
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    for hook in hooks:
        hook.remove()
        
def check_model_dtypes(model):
    """Check dtypes of all parameters and buffers"""
    print("=== PARAMETERS ===")
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"❌ {name}: {param.dtype}")
    
    print("\n=== BUFFERS ===")
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"❌ {name}: {buffer.dtype}")
    
    print("\n=== ALL TENSORS ===")
    for name, tensor in model.state_dict().items():
        if tensor.dtype != torch.float32:
            print(f"❌ {name}: {tensor.dtype}")
