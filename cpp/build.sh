#!/usr/bin/env bash
# build the C++ safetensor->GGUF converter, pulling in nlohmann/json header
set -euo pipefail

# ensure header exists
mkdir -p nlohmann
if [ ! -f nlohmann/json.hpp ]; then
    echo "downloading nlohmann/json.hpp..."
    curl -sfL https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp -o nlohmann/json.hpp
fi

echo "compiling C++ converter..."
g++ -O3 safetensor_to_gguf.cpp -o safetensor_to_gguf -I. -lm

echo "built cpp/safetensor_to_gguf"
