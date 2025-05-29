# PLAN.md

## Goal
Implement text generation (GPT-2) in the OpenVINO-Llama demo, working within Wasm/wasi-nn constraints.

## Constraints
- Memory: <4GB (Wasm limitation)
- Tensors: ≤32-bit (wasi-nn limitation)
- Runtime: OpenVINO via wasi-nn in WebAssembly
- Model: GPT-2 8-bit quantized (from TFLite)

## Approach: Incremental Testing with Verification

### Phase 1: Remove Existing Text Code
1. Identify all text-specific code in the codebase
2. Document what was removed and why
3. Verify image inference still works
4. Commit clean baseline

### Phase 2: Test TFLite Model (Python)
Working directory: `OpenVINO-GPT-2/`

1. Create simple Python script to test TFLite model
   - Load `ov_ir/gpt2-8bit.tflite`
   - Run basic inference with sample prompt
   - Verify output is sensible
   - Document tensor shapes, dtypes, memory usage

2. If model doesn't work:
   - Debug/fix the TFLite model
   - Find alternative GPT-2 TFLite model
   - Ensure 8-bit quantization is correct

### Phase 3: Convert to OpenVINO IR
1. Review existing conversion script (`ov_ir/tflite.ipynb`)
2. Create standalone conversion script with:
   - Explicit error checking
   - Tensor shape/dtype validation
   - Memory usage reporting
   
3. Convert TFLite → OpenVINO IR
   - Save as `.xml` and `.bin` files
   - Document exact OpenVINO version used
   - Verify all ops are supported

4. Test OpenVINO model (Python/C++)
   - Load and run inference
   - Compare outputs with TFLite version
   - Ensure deterministic results

### Phase 4: Native Rust Testing
1. Create minimal Rust program (not Wasm)
   - Load OpenVINO model
   - Run inference
   - No preprocessing complexity initially

2. Add preprocessing incrementally:
   - Tokenization (keep simple)
   - Input tensor preparation
   - Output decoding

### Phase 5: Wasm Integration
1. Test in isolated Wasm environment first
   - Just model loading
   - Then single inference
   - Monitor memory usage

2. Integrate into inferencer
   - Add to model registry
   - Implement minimal text preprocessing
   - Handle tensor constraints explicitly

3. Connect to server
   - Add text endpoint
   - Simple request/response
   - No streaming initially

### Phase 6: Frontend Integration
1. Add basic text input UI
2. Wire to backend endpoint
3. Display results

## Key Technical Decisions

### Model Choice
- GPT-2 small (124M params)
- 8-bit quantized for memory efficiency
- TFLite source ensures 32-bit tensor compatibility

### Preprocessing Strategy
- Minimal tokenizer (consider fixed vocabulary)
- Batch size = 1 (memory constraint)
- Short sequence lengths initially (e.g., 128 tokens)

### Error Handling
- Explicit checks at every boundary
- Fast failure with clear messages
- No silent truncation/conversion

### Testing Strategy
Each phase must pass before proceeding:
- Phase output = working code + test results
- Document failures and solutions
- Keep test cases for regression

## Success Criteria
- Basic text generation works in Wasm
- Memory usage <4GB
- No tensor size violations
- Clean, maintainable code
- Clear documentation of constraints/limitations