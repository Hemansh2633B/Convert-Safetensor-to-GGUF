# Python Safetensor to GGUF Converter

This directory contains a Python implementation for converting AI models from Safetensors format to GGUF format.

## Explanation of Key Parts

### Tensor Metadata Collection
- If `model.safetensors.index.json` exists, it reads the weight map to locate all sharded files.
- For each safetensors file, it uses `safe_open` with `framework="np"` to iterate over keys and extract metadata (shape, dtype) without loading the actual tensor data.
- Each tensor’s name, shape, original dtype, and file path are stored in a `TensorInfo` object.
- Tensors are sorted by name for deterministic ordering in the GGUF file.

### Offset Precomputation
- Based on the target dtype (original or quantized F16), the size of each tensor is calculated.
- Cumulative offsets (relative to the start of the data section) are computed. This allows writing tensor infos before the data in a single pass.

### GGUF Writing
- **Header:** Magic number, version, tensor count, metadata count.
- **Metadata:** Writes `general.architecture` and `<arch>.context_length` from `config.json`. Optionally adds `vocab_size` if present.
- **Tensor Infos:** For each tensor, writes its name, shape, GGML type, and precomputed offset.
- **Alignment:** Pads the file to a 32‑byte boundary before writing tensor data.
- **Tensor Data:** Loads each tensor from the safetensors file, converts if necessary (float32 → float16 when `--f16` is used and source is F32), and writes the raw bytes.

### Quantization
- When `--f16` is specified, any tensor with original dtype F32 is converted to float16 using NumPy’s `.astype(np.float16)`. Other dtypes are passed through unchanged.
- The GGML type in the tensor info is set accordingly (`GGML_TYPE_F16` for converted tensors).

### Error Handling
- The script checks for missing files, unsupported dtypes, and other I/O errors, printing informative messages and returning a non‑zero exit code.

## Limitations and Future Improvements
- **Additional Quantization:** Only F32 → F16 is implemented. To support other quantized types (e.g., Q4_0, Q8_0), you would need to integrate quantization functions (e.g., from llama.cpp).
- **Metadata:** Only basic metadata is included. For full compatibility, you may want to add tokenizer information (`tokenizer.json`), model hyperparameters, and a list of tensor names.
- **Performance:** For very large models, loading each tensor individually is acceptable, but memory usage is limited to one tensor at a time. The script uses safetensors lazy loading, so it’s efficient.
