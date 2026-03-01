# Usage Guide

## Python
```bash
python python/safetensor_to_gguf.py <input.safetensors> <output.gguf>
```

## C
```bash
cd c && gcc safetensor_to_gguf.c -o safetensor_to_gguf
./safetensor_to_gguf <input.safetensors> <output.gguf>
```

## C++
```bash
cd cpp && g++ safetensor_to_gguf.cpp -o safetensor_to_gguf
./safetensor_to_gguf <input.safetensors> <output.gguf>
```

Place your model files in the `models/` directory for organization.
