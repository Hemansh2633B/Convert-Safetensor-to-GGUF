#!/usr/bin/env python3
"""
Convert a Hugging Face model from SafeTensor format to GGUF.
Supports single and sharded safetensors, with optional FP16 quantization.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, BinaryIO

import numpy as np
from safetensors import safe_open

# -----------------------------------------------------------------------------
# GGUF constants
# -----------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF" little endian
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGML type enumerations (only those used here)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# Mapping from safetensors dtype string to GGML type
DTYPE_TO_GGML = {
    "F32": GGML_TYPE_F32,
    "F16": GGML_TYPE_F16,
    # Add more if needed (e.g., BF16, I8, etc.)
}

# Element size in bytes for each dtype (used for offset calculation)
ELEMENT_SIZE = {
    "F32": 4,
    "F16": 2,
}

# -----------------------------------------------------------------------------
# GGUF writing helpers (all values are written little‑endian)
# -----------------------------------------------------------------------------
def write_le32(f: BinaryIO, x: int) -> None:
    f.write(struct.pack('<I', x))

def write_le64(f: BinaryIO, x: int) -> None:
    f.write(struct.pack('<Q', x))

def write_string(f: BinaryIO, s: str) -> None:
    """Write a string with length prefix (uint32)."""
    encoded = s.encode('utf-8')
    write_le32(f, len(encoded))
    f.write(encoded)

def write_kv_string(f: BinaryIO, key: str, value: str) -> None:
    """Write a metadata key-value pair of type string."""
    write_string(f, key)
    write_le32(f, 8)          # GGUF_TYPE_STRING
    write_string(f, value)

def write_kv_uint32(f: BinaryIO, key: str, value: int) -> None:
    """Write a metadata key-value pair of type uint32."""
    write_string(f, key)
    write_le32(f, 4)          # GGUF_TYPE_UINT32
    write_le32(f, value)

def write_kv_bool(f: BinaryIO, key: str, value: bool) -> None:
    """Write a metadata key-value pair of type bool."""
    write_string(f, key)
    write_le32(f, 7)          # GGUF_TYPE_BOOL
    f.write(struct.pack('<?', value))

def write_tensor_info(f: BinaryIO, name: str, shape: List[int],
                      ggml_type: int, offset: int) -> None:
    """Write a tensor info block."""
    write_string(f, name)                     # name length + string
    write_le32(f, len(shape))                  # n_dims
    for dim in shape:
        write_le64(f, dim)                      # each dimension
    write_le32(f, ggml_type)                    # ggml type
    write_le64(f, offset)                        # offset in data section

def align(f: BinaryIO, alignment: int = GGUF_ALIGNMENT) -> None:
    """Write padding bytes to achieve alignment."""
    pos = f.tell()
    pad = (alignment - (pos % alignment)) % alignment
    if pad:
        f.write(b'\x00' * pad)

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
class TensorInfo:
    """Metadata about a tensor to be written."""
    def __init__(self, name: str, shape: List[int], dtype: str, file_path: Path):
        self.name = name
        self.shape = shape
        self.dtype = dtype          # original safetensors dtype string
        self.file_path = file_path
        self.data_size = self._compute_size()

    def _compute_size(self) -> int:
        """Compute size in bytes of the tensor (original)."""
        numel = 1
        for d in self.shape:
            numel *= d
        return numel * ELEMENT_SIZE.get(self.dtype, 4)  # fallback to 4 bytes

    def __repr__(self):
        return f"TensorInfo(name={self.name}, shape={self.shape}, dtype={self.dtype})"

# -----------------------------------------------------------------------------
# Main conversion logic
# -----------------------------------------------------------------------------
def collect_tensors(model_dir: Path) -> List[TensorInfo]:
    """
    Find all safetensors files and extract tensor metadata.
    Handles both single `model.safetensors` and sharded with index file.
    """
    index_path = model_dir / "model.safetensors.index.json"
    safetensors_files = []

    if index_path.exists():
        # Sharded model: read index to get file list
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        weight_map = index.get('weight_map', {})
        # Collect unique filenames
        filenames = set(weight_map.values())
        for fname in filenames:
            safetensors_files.append(model_dir / fname)
    else:
        # Single file
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No model.safetensors or index file found in {model_dir}")
        safetensors_files.append(single)

    # Now gather tensor info from each file
    tensors = []
    for sf_path in safetensors_files:
        with safe_open(sf_path, framework="np") as sf:
            for key in sf.keys():
                # Get metadata without loading data
                shape = sf.get_slice(key).get_shape()
                dtype = sf.get_slice(key).get_dtype()
                # Convert numpy dtype to safetensors string (e.g., float32 -> "F32")
                # safetensors uses strings like "F32", "F16", etc.
                # The dtype from get_dtype() is a numpy dtype, we map back.
                dtype_str = {
                    np.float32: "F32",
                    np.float16: "F16",
                    # add other types if needed
                }.get(dtype.type)
                if dtype_str is None:
                    raise ValueError(f"Unsupported dtype {dtype} for tensor {key}")

                tensors.append(TensorInfo(key, shape, dtype_str, sf_path))

    # Sort by name for deterministic order (optional but recommended)
    tensors.sort(key=lambda t: t.name)
    return tensors

def convert_and_write(tensors: List[TensorInfo], output_path: Path,
                      quantize_f16: bool, config: Dict[str, Any]) -> None:
    """
    Perform the actual GGUF conversion and write to output_path.
    """
    # Determine target dtype and compute offsets
    target_dtype = []
    data_sizes = []
    for t in tensors:
        if quantize_f16 and t.dtype == "F32":
            target_dtype.append("F16")
            data_sizes.append(t.data_size // 2)   # F32 -> F16 halves the size
        else:
            target_dtype.append(t.dtype)
            data_sizes.append(t.data_size)

    # Compute cumulative offsets (relative to start of data section)
    offsets = []
    cum = 0
    for sz in data_sizes:
        offsets.append(cum)
        cum += sz

    # Prepare metadata from config
    arch = config.get("model_type", "unknown")
    ctx_len = config.get("max_position_embeddings", 2048)

    # Count metadata entries: we add architecture and context_length
    metadata_count = 2
    # Optionally add more metadata (e.g., vocab_size) if present
    if "vocab_size" in config:
        metadata_count += 1

    with open(output_path, "wb") as f:
        # ---- Header ----
        write_le32(f, GGUF_MAGIC)
        write_le32(f, GGUF_VERSION)
        write_le64(f, len(tensors))
        write_le64(f, metadata_count)

        # ---- Metadata ----
        write_kv_string(f, "general.architecture", arch)
        write_kv_uint32(f, f"{arch}.context_length", ctx_len)
        if "vocab_size" in config:
            write_kv_uint32(f, f"{arch}.vocab_size", config["vocab_size"])

        # ---- Tensor infos (with placeholder offsets for now) ----
        # We'll write them now because we have precomputed offsets.
        # No need to seek back later.
        for i, t in enumerate(tensors):
            ggml_type = DTYPE_TO_GGML.get(target_dtype[i], GGML_TYPE_F32)
            write_tensor_info(f, t.name, t.shape, ggml_type, offsets[i])

        # ---- Align data section ----
        align(f)

        # ---- Write tensor data ----
        # For progress, we can use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(zip(tensors, target_dtype), desc="Writing tensors", unit="tensor")
        except ImportError:
            iterator = zip(tensors, target_dtype)
            print("Writing tensors... (install tqdm for progress bar)")

        for t, target in iterator:
            # Load the tensor from safetensors
            with safe_open(t.file_path, framework="np") as sf:
                data = sf.get_tensor(t.name)  # numpy array

            # Convert if needed
            if target == "F16" and t.dtype == "F32":
                data = data.astype(np.float16)
            # For other dtypes, we assume they are already correct.

            # Write raw bytes
            f.write(data.tobytes())

    print(f"Successfully converted {len(tensors)} tensors to {output_path}")
    if quantize_f16:
        print("Quantized float32 tensors to float16.")

def main():
    parser = argparse.ArgumentParser(description="Convert SafeTensor model to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to model directory containing config.json and safetensors files")
    parser.add_argument("output", type=Path, help="Output GGUF file path")
    parser.add_argument("--f16", action="store_true", help="Quantize float32 tensors to float16")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    quantize_f16 = args.f16

    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory", file=sys.stderr)
        return 1

    # 1. Load config.json
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. Collect tensor metadata from safetensors files
    try:
        tensors = collect_tensors(model_dir)
    except Exception as e:
        print(f"Error collecting tensors: {e}", file=sys.stderr)
        return 1

    if not tensors:
        print("No tensors found.", file=sys.stderr)
        return 1

    # 3. Convert and write GGUF
    try:
        convert_and_write(tensors, output_path, quantize_f16, config)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())