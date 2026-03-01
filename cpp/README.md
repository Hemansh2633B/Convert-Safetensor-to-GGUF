# C++ Safetensor to GGUF Converter

This directory contains a C++ implementation for converting AI models from Safetensors format to GGUF format.

## Explanation of Key Parts

### Safetensors Parsing
- Each file is opened, the 8‑byte header length is read, then the JSON header is parsed.
- For every tensor, we store its name, shape, dtype, absolute file offset, and size.

### Two‑Pass GGUF Writing
- **Pass 1:** Write the file header and metadata. Then write placeholder bytes for all tensor infos (exactly the size each will occupy). This reserves space so we can later fill them with real data without moving the file pointer.
- **Pass 2:** After writing all tensor data (and recording their offsets relative to the start of the data section), we seek back to the beginning of the tensor info area and write the actual tensor infos with the correct offsets.
- This avoids complex seeking after data is written and ensures the file is valid from the start.

### Float16 Conversion
- A simple `float_to_half` function is provided, using round‑to‑nearest‑even. For production, you may want a more optimized routine or use a library.
- If `--f16` is specified and the source tensor is F32, it is converted; otherwise, the original data is written as‑is (the GGUF tensor type is set accordingly).

### Metadata
- Only `general.architecture` and `<arch>.context_length` are written by default. You can easily extend this by adding more key‑value pairs from `config.json` (e.g., `vocab_size`, `hidden_size`).
- The metadata count in the header is fixed at 2 for this demo; if you add more keys, you must adjust the count accordingly (either pre‑count before writing the header or patch the header later).

### Error Handling
- Exceptions are thrown for any file or parsing error. The main function catches them and prints an error message.

## Limitations and Future Improvements
- **Quantization:** Only F32 → F16 is implemented. For other types (e.g., Q4_0, Q8_0), you would need to integrate quantization kernels (e.g., from llama.cpp).
- **Metadata:** Only basic metadata is included. A production converter would also add tokenizer data (from `tokenizer.json`), model hyperparameters, and possibly a list of tensor names.
- **Endianness:** The code assumes little‑endian host. For big‑endian systems, you would need to byte‑swap multi‑byte values.
- **Performance:** For very large models, reading all tensor data into memory at once may be prohibitive. A more advanced implementation would use memory‑mapped files or stream data in chunks.
- **Validation:** The code does not verify that the data written matches the expected tensor shapes/dtypes. Additional checks could prevent silent corruption.

Despite these limitations, the code provides a solid foundation for converting SafeTensor models to GGUF and can be extended as needed.
