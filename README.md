
# Convert-Safetensor-to-GGUF

This repository provides three implementations for converting Hugging Face models
from **SafeTensor** format to **GGUF**: C, C++, and Python. The examples below
walk through converting the `Qwen_0.6B-BF16_Base_Model` repository as a proof of
concept.

## Directory Structure

- `python/`   – Python implementation of the converter
- `c/`        – C implementation of the converter
- `cpp/`      – C++ implementation of the converter
- `scripts/`  – Utility scripts and helpers (empty by default)
- `models/`   – Suggested location for model directories (input/output)
- `docs/`     – Documentation and guides
- `convert_safetensor_to_gguf.py` – legacy standalone Python script

## Requirements

### Common
- A Unix-like shell (Linux/macOS) with `gcc`, `g++`, and `curl` available.
- `git` to clone model repositories.

### Python tool
- Python 3.8+. The converter will attempt to install missing packages
  (`numpy`, `safetensors`, `tqdm`) automatically when run, but you can also
  install them up front:
  ```bash
  pip install numpy safetensors tqdm
  ```

### C tool
The C converter depends on [cJSON](https://github.com/DaveGamble/cJSON).
Fetch the sources before building.

A helper script automates this and compiles the binary:
```bash
./c/build.sh
```

Alternatively, you can run the commands manually:
```bash
cd c
curl -L https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h -o cJSON.h
curl -L https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c -o cJSON.c
gcc -O3 safetensor_to_gguf.c cJSON.c -o safetensor_to_gguf -lm
```
### C++ tool
The C++ converter uses [nlohmann/json](https://github.com/nlohmann/json) as a
header‑only library. A build helper script will fetch the header and compile
for you:

```bash
./cpp/build.sh
```

Manual steps are still possible:

```bash
cd cpp
mkdir -p nlohmann
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o nlohmann/json.hpp
g++ -O3 safetensor_to_gguf.cpp -o safetensor_to_gguf -I. -lm
```
## Example Workflow (Qwen Model)

Two convenient helpers are provided:

* `Makefile` to build the converters and run the default example.
* `scripts/convert_all.sh` – a standalone shell script that clones a model,
  builds the C/C++ binaries, installs Python deps, and invokes all three
  converters.

Use either approach:

```bash
# build only
make build-c build-cpp

# run the full pipeline (clones Qwen model by default)
make run

# or run the script directly, optionally specifying repository or output prefix
./scripts/convert_all.sh https://github.com/Hemansh2633B/Qwen_0.6B-BF16_Base_Model.git qwen_model qwen_model
```

The default script call reproduces the step-by-step commands shown further
down in this document.


```bash
# 1. clone the model repository
git clone https://github.com/Hemansh2633B/Qwen_0.6B-BF16_Base_Model.git qwen_model

# 2. convert with C tool
./c/safetensor_to_gguf qwen_model qwen_model_c.gguf
# emits warnings about __metadata__ but completes successfully

# 3. convert with C++ tool
./cpp/safetensor_to_gguf qwen_model qwen_model_cpp.gguf
# explicit check skips __metadata__ entry

# 4. convert with Python tool
python python/safetensor_to_gguf.py qwen_model qwen_model_py.gguf
# handles BF16 tensors by reading raw bytes and converting to float32
```

All three converters produced GGUF files roughly 943 MiB in size; the Python
version is ~1.9 GiB because it expands BF16 → float32.

Pass `--f16` to any tool to quantize float32 tensors to float16 (C/C++ accept
the flag after the output filename; Python uses `--f16` argument).

## Notes & Troubleshooting

- The C/C++ implementations automatically skip non‑tensor entries such as
  `__metadata__` that may appear in the safetensors header.
- The Python converter now supports BF16 data by manually parsing the safetensors
  header to obtain byte offsets and converting the raw 16‑bit values into
  float32. It also ensures the output file handle is never shadowed when
  reading.
- Adjustments may be needed for model-specific dtypes or metadata fields.

### Common command mistakes

Here are some errors you might encounter while setting up or running the
converters, along with how they were resolved during development:

1. **Missing headers/libraries** – compiling C without downloading `cJSON.h`
or C++ without the `nlohmann/json.hpp` header results in “No such file”
errors. Use the provided build scripts (`c/build.sh`, `cpp/build.sh`) to
fetch dependencies automatically.

2. **Implicit declaration warnings in C** – functions like `write_kv_string`
were used before they were declared. Forward-declare helper routines above
`main()` or compile with stricter flags to spot these early.

3. **Ignored return values** – several `fread()` calls produced `-Wunused-result`
warnings; in production you should check their return value to avoid corrupt
reads.

4. **Python modules not found** – running the script without installing
`numpy`, `safetensors`, or `tqdm` raised `ModuleNotFoundError`. The Python
converter now auto-installs missing packages, but you can also preinstall them
via pip.

5. **Syntax/typo errors when editing code** – leftover diff markers (`-` or
`+`) or duplicate method definitions cause `SyntaxError`. Always run
`python -m py_compile` after editing.

6. **Data type support** – many safetensors models use BF16; attempting to
load them with vanilla numpy gives `TypeError: data type 'bfloat16' not
understood`. The Python tool now parses raw bytes to handle BF16.

Including these lessons in your workflow helps avoid the most common stumbling
blocks.

Feel free to adapt, extend, or integrate these converters into your own tools.