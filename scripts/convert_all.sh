#!/usr/bin/env bash
# clone a safetensors model repository and run all three converters
# Usage: ./scripts/convert_all.sh [model_url] [model_dir] [output_prefix]
set -euo pipefail

MODEL_URL=${1:-https://github.com/Hemansh2633B/Qwen_0.6B-BF16_Base_Model.git}
MODEL_DIR=${2:-qwen_model}
OUTPUT_PREFIX=${3:-qwen_model}

echo "[1/6] cloning $MODEL_URL -> $MODEL_DIR"
if [ -d "$MODEL_DIR" ]; then
    echo "directory $MODEL_DIR already exists, skipping clone"
else
    git clone "$MODEL_URL" "$MODEL_DIR"
fi

echo "[2/6] building C converter"
(cd c && ./build.sh)

echo "[3/6] building C++ converter"
(cd cpp && ./build.sh)

echo "[4/6] installing Python dependencies"
python3 -m pip install --user numpy safetensors tqdm

echo "[5/6] running C conversion"
./c/safetensor_to_gguf "$MODEL_DIR" "${OUTPUT_PREFIX}_c.gguf"

echo "[6/6] running C++ conversion"
./cpp/safetensor_to_gguf "$MODEL_DIR" "${OUTPUT_PREFIX}_cpp.gguf"

echo "running Python conversion"
python3 python/safetensor_to_gguf.py "$MODEL_DIR" "${OUTPUT_PREFIX}_py.gguf"

echo "all conversions finished"
