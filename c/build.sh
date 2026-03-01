#!/usr/bin/env bash
# build the C safetensor->GGUF converter, pulling in cJSON if necessary
set -euo pipefail

# fetch cJSON files if missing or outdated
if [ ! -f cJSON.h ] || [ ! -f cJSON.c ]; then
    echo "downloading cJSON..."
    curl -sfL https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h -o cJSON.h
    curl -sfL https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c -o cJSON.c
fi

echo "compiling C converter..."
gcc -O3 safetensor_to_gguf.c cJSON.c -o safetensor_to_gguf -lm

echo "built c/safetensor_to_gguf"
