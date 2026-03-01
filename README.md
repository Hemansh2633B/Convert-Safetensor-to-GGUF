
# Convert-Safetensor-to-GGUF

Convert AI models from Safetensors format to GGUF format using Python, C, or C++ implementations.

## Directory Structure

- `python/`   : Python implementation of Safetensor to GGUF converter
- `c/`        : C implementation of Safetensor to GGUF converter
- `cpp/`      : C++ implementation of Safetensor to GGUF converter
- `scripts/`  : Utility scripts and helpers
- `models/`   : Place your model files here (input/output)
- `docs/`     : Documentation and guides
- `convert_safetensor_to_gguf.py` : Legacy/standalone Python script

## Requirements
- Python 3.8+
- [safetensors](https://pypi.org/project/safetensors/)
- [llama-cpp-python](https://pypi.org/project/llama-cpp-python/) (for GGUF export)

## Usage

### Python
```bash
python python/safetensor_to_gguf.py <input.safetensors> <output.gguf>
```

### C
```bash
cd c && gcc safetensor_to_gguf.c -o safetensor_to_gguf
./safetensor_to_gguf <input.safetensors> <output.gguf>
```

### C++
```bash
cd cpp && g++ safetensor_to_gguf.cpp -o safetensor_to_gguf
./safetensor_to_gguf <input.safetensors> <output.gguf>
```

Place your model files in the `models/` directory for organization.

## Documentation
See the `docs/` folder for more details on structure and usage.

**Note:** Actual conversion logic is model-specific and may require additional implementation for your model type.