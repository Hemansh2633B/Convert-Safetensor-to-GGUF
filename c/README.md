# C Safetensor to GGUF Converter

This directory contains a C implementation for converting AI models from Safetensors format to GGUF format.

## Explanation of Key Parts

- **JSON Parsing:** Uses cJSON to parse `config.json`, the safetensors index (if present), and the header of each `.safetensors` file.
- **Safetensors Header:** Each file starts with an 8‑byte little‑endian length, followed by a JSON object mapping tensor names to metadata (dtype, shape, data offsets). The absolute file offset for tensor data is computed as `8 + header_len + data_offset`.
- **Two‑Pass GGUF Writing:**
  1. Write the file header and metadata.
  2. Write placeholder bytes for all tensor infos (the exact number of bytes each will occupy). This reserves space.
  3. Write all tensor data (converting to float16 if requested) and record their offsets relative to the start of the data section.
  4. Seek back to the beginning of the tensor info area and overwrite the placeholders with real tensor info, including the correct offsets.
- **Endianness:** All multi‑byte values are written in little‑endian order using `write_le32` and `write_le64`, ensuring compatibility regardless of host endianness.
- **Float16 Conversion:** A simple `float_to_half` function performs round‑to‑nearest‑even conversion. It handles special cases (Inf, NaN, underflow, overflow).

## Limitations and Future Improvements

- **Quantization:** Only F32 → F16 is implemented. For other quantized types (Q4_0, Q8_0, etc.), you would need to integrate quantization routines (e.g., from llama.cpp).
- **Metadata:** Only `general.architecture` and `<arch>.context_length` are written. Extend `write_kv_*` calls to include additional parameters like `vocab_size`, `hidden_size`, and tokenizer data.
- **Error Handling:** The code performs basic error checking but may leak memory on some error paths (e.g., after partial allocation). A production version should carefully clean up.
- **Performance:** For large models, reading entire tensor data into memory may be inefficient. Consider memory‑mapped I/O or streaming in chunks.

Despite these limitations, this C program provides a solid foundation for converting SafeTensor models to GGUF and can be extended as needed.
