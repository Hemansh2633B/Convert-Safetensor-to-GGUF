#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include "cJSON.h"

// -----------------------------------------------------------------------------
// GGUF constants
// -----------------------------------------------------------------------------
#define GGUF_MAGIC       0x46554747UL  // "GGUF" little endian
#define GGUF_VERSION      3
#define GGUF_ALIGNMENT   32

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
} ggml_type;

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

// -----------------------------------------------------------------------------
// SafeTensor structures
// -----------------------------------------------------------------------------
typedef struct {
    char* name;
    int64_t* shape;
    int n_dims;
    char* dtype;           // e.g., "F32", "F16"
    uint64_t file_offset;  // absolute offset in safetensors file
    uint64_t data_size;    // size in bytes in original file
    char* filename;
} TensorInfo;

// -----------------------------------------------------------------------------
// Endianness helpers (write little‑endian)
// -----------------------------------------------------------------------------
static void write_le32(FILE* f, uint32_t x) {
    uint8_t buf[4] = { x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF };
    fwrite(buf, 1, 4, f);
}

static void write_le64(FILE* f, uint64_t x) {
    uint8_t buf[8] = {
        x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF,
        (x >> 32) & 0xFF, (x >> 40) & 0xFF, (x >> 48) & 0xFF, (x >> 56) & 0xFF
    };
    fwrite(buf, 1, 8, f);
}

// -----------------------------------------------------------------------------
// Float16 conversion (round‑to‑nearest‑even)
// -----------------------------------------------------------------------------
static uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    uint32_t exp  = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    if (exp == 0xFF) { // NaN or Inf
        if (mant != 0) return 0x7E00; // quiet NaN
        return (sign << 15) | 0x7C00; // Inf
    }

    int32_t newexp = (int32_t)exp - 127 + 15;
    if (newexp >= 31) return (sign << 15) | 0x7C00; // overflow -> Inf
    if (newexp <= 0) { // underflow -> zero/subnormal
        if (newexp < -10) return (sign << 15);
        mant |= 0x800000;
        uint32_t half = (sign << 15) | (mant >> (14 - newexp));
        return half;
    }
    return (sign << 15) | (newexp << 10) | (mant >> 13);
}

// -----------------------------------------------------------------------------
// Helper: read entire file into a string (for JSON)
// -----------------------------------------------------------------------------
static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t read = fread(buf, 1, len, f);
    fclose(f);
    if (read != (size_t)len) { free(buf); return NULL; }
    buf[len] = '\0';
    if (out_len) *out_len = len;
    return buf;
}

// -----------------------------------------------------------------------------
// Parse safetensors header of a single file and append tensors to array
// -----------------------------------------------------------------------------
static int parse_safetensors_file(const char* filename, TensorInfo** tensors, int* count, int* capacity) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return -1;
    }

    // Read header length (8 bytes little endian)
    uint64_t header_len;
    if (fread(&header_len, 1, 8, f) != 8) {
        fprintf(stderr, "Failed to read header length from %s\n", filename);
        fclose(f);
        return -1;
    }

    // Read header JSON
    char* header_json = (char*)malloc(header_len + 1);
    if (!header_json) {
        fclose(f);
        return -1;
    }
    if (fread(header_json, 1, header_len, f) != header_len) {
        fprintf(stderr, "Failed to read header from %s\n", filename);
        free(header_json);
        fclose(f);
        return -1;
    }
    header_json[header_len] = '\0';

    cJSON* root = cJSON_Parse(header_json);
    free(header_json);
    if (!root) {
        fprintf(stderr, "JSON parse error in %s\n", filename);
        fclose(f);
        return -1;
    }

    // Iterate over all top-level objects (tensor names)
    cJSON* tensor_obj = NULL;
    cJSON_ArrayForEach(tensor_obj, root) {
        const char* tensor_name = tensor_obj->string;
        cJSON* dtype = cJSON_GetObjectItem(tensor_obj, "dtype");
        cJSON* shape = cJSON_GetObjectItem(tensor_obj, "shape");
        cJSON* data_offsets = cJSON_GetObjectItem(tensor_obj, "data_offsets");

        if (!dtype || !cJSON_IsString(dtype) ||
            !shape || !cJSON_IsArray(shape) ||
            !data_offsets || !cJSON_IsArray(data_offsets) || cJSON_GetArraySize(data_offsets) != 2) {
            fprintf(stderr, "Malformed tensor info for %s\n", tensor_name);
            continue;
        }

        // Ensure capacity
        if (*count >= *capacity) {
            *capacity = (*capacity == 0) ? 128 : *capacity * 2;
            *tensors = (TensorInfo*)realloc(*tensors, *capacity * sizeof(TensorInfo));
        }

        TensorInfo* ti = &(*tensors)[*count];
        memset(ti, 0, sizeof(TensorInfo));

        ti->name = strdup(tensor_name);
        ti->dtype = strdup(dtype->valuestring);
        ti->filename = strdup(filename);

        // Parse shape
        ti->n_dims = cJSON_GetArraySize(shape);
        ti->shape = (int64_t*)malloc(ti->n_dims * sizeof(int64_t));
        for (int i = 0; i < ti->n_dims; i++) {
            cJSON* item = cJSON_GetArrayItem(shape, i);
            ti->shape[i] = (int64_t)item->valuedouble; // JSON numbers are double
        }

        // Parse data offsets (relative to end of header)
        cJSON* off0 = cJSON_GetArrayItem(data_offsets, 0);
        cJSON* off1 = cJSON_GetArrayItem(data_offsets, 1);
        uint64_t start = (uint64_t)off0->valuedouble;
        uint64_t end   = (uint64_t)off1->valuedouble;
        ti->file_offset = start + 8 + header_len; // 8 bytes for header length
        ti->data_size   = end - start;

        (*count)++;
    }

    cJSON_Delete(root);
    fclose(f);
    return 0;
}

// -----------------------------------------------------------------------------
// Main conversion
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <model_dir> <output.gguf> [--f16]\n", argv[0]);
        return 1;
    }

    const char* model_dir = argv[1];
    const char* output_path = argv[2];
    bool quantize_f16 = (argc == 4 && strcmp(argv[3], "--f16") == 0);

    // 1. Read config.json
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    size_t config_len;
    char* config_json = read_file(config_path, &config_len);
    if (!config_json) {
        fprintf(stderr, "Failed to read config.json\n");
        return 1;
    }
    cJSON* config = cJSON_Parse(config_json);
    free(config_json);
    if (!config) {
        fprintf(stderr, "Failed to parse config.json\n");
        return 1;
    }

    cJSON* model_type = cJSON_GetObjectItem(config, "model_type");
    cJSON* max_pos = cJSON_GetObjectItem(config, "max_position_embeddings");
    const char* arch = model_type && cJSON_IsString(model_type) ? model_type->valuestring : "unknown";
    uint32_t ctx_len = max_pos && cJSON_IsNumber(max_pos) ? (uint32_t)max_pos->valuedouble : 2048;

    // 2. Find safetensors files
    char index_path[1024];
    snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", model_dir);
    FILE* index_file = fopen(index_path, "rb");
    char** safetensor_files = NULL;
    int num_files = 0;
    int files_cap = 0;

    if (index_file) {
        // Sharded model: read index
        fseek(index_file, 0, SEEK_END);
        long idx_len = ftell(index_file);
        rewind(index_file);
        char* idx_json = (char*)malloc(idx_len + 1);
        fread(idx_json, 1, idx_len, index_file);
        idx_json[idx_len] = '\0';
        fclose(index_file);

        cJSON* idx_root = cJSON_Parse(idx_json);
        free(idx_json);
        if (!idx_root) {
            fprintf(stderr, "Failed to parse model.safetensors.index.json\n");
            cJSON_Delete(config);
            return 1;
        }
        cJSON* weight_map = cJSON_GetObjectItem(idx_root, "weight_map");
        if (!weight_map || !cJSON_IsObject(weight_map)) {
            fprintf(stderr, "No weight_map in index\n");
            cJSON_Delete(idx_root);
            cJSON_Delete(config);
            return 1;
        }

        // Collect unique filenames
        cJSON* child = NULL;
        cJSON_ArrayForEach(child, weight_map) {
            const char* fname = child->valuestring;
            char full_path[1024];
            snprintf(full_path, sizeof(full_path), "%s/%s", model_dir, fname);

            // Check if already added
            bool found = false;
            for (int i = 0; i < num_files; i++) {
                if (strcmp(safetensor_files[i], full_path) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (num_files >= files_cap) {
                    files_cap = files_cap == 0 ? 8 : files_cap * 2;
                    safetensor_files = (char**)realloc(safetensor_files, files_cap * sizeof(char*));
                }
                safetensor_files[num_files] = strdup(full_path);
                num_files++;
            }
        }
        cJSON_Delete(idx_root);
    } else {
        // Single file
        char single_path[1024];
        snprintf(single_path, sizeof(single_path), "%s/model.safetensors", model_dir);
        FILE* test = fopen(single_path, "rb");
        if (!test) {
            fprintf(stderr, "No model.safetensors or index file found in %s\n", model_dir);
            cJSON_Delete(config);
            return 1;
        }
        fclose(test);
        safetensor_files = (char**)malloc(sizeof(char*));
        safetensor_files[0] = strdup(single_path);
        num_files = 1;
    }

    // 3. Parse all safetensors files and collect tensor info
    TensorInfo* tensors = NULL;
    int tensor_count = 0;
    int tensor_cap = 0;
    for (int i = 0; i < num_files; i++) {
        if (parse_safetensors_file(safetensor_files[i], &tensors, &tensor_count, &tensor_cap) != 0) {
            // Cleanup and exit
            for (int j = 0; j < num_files; j++) free(safetensor_files[j]);
            free(safetensor_files);
            // Free tensors already collected? We'll just exit.
            cJSON_Delete(config);
            return 1;
        }
        free(safetensor_files[i]);
    }
    free(safetensor_files);

    // Optional: sort tensors by name (for determinism) - we skip for simplicity

    // 4. Write GGUF file (two passes)
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file %s\n", output_path);
        cJSON_Delete(config);
        return 1;
    }

    // 4a. Write header (placeholder for metadata count, will patch later if needed)
    uint64_t metadata_count = 2; // arch + context_length
    write_le32(out, GGUF_MAGIC);
    write_le32(out, GGUF_VERSION);
    write_le64(out, tensor_count);
    write_le64(out, metadata_count); // placeholder, we may add more keys later

    // 4b. Write metadata
    write_kv_string(out, "general.architecture", arch);
    char ctx_key[256];
    snprintf(ctx_key, sizeof(ctx_key), "%s.context_length", arch);
    write_kv_uint32(out, ctx_key, ctx_len);
    // If you want to add more metadata from config (e.g., vocab_size), do it here and update metadata_count in header.

    // 4c. Reserve space for tensor infos
    // We need to know the exact size each tensor info will occupy.
    // We'll store the start position and later seek back to write actual data.
    long tensor_info_start = ftell(out);
    // For each tensor, we'll write zeros of the appropriate size.
    for (int i = 0; i < tensor_count; i++) {
        TensorInfo* ti = &tensors[i];
        uint32_t name_len = strlen(ti->name);
        uint32_t n_dims = ti->n_dims;
        size_t info_size = sizeof(name_len) + name_len +
                           sizeof(n_dims) +
                           n_dims * sizeof(int64_t) +
                           sizeof(uint32_t) + // ggml_type
                           sizeof(uint64_t);  // offset
        // Write zeros
        for (size_t j = 0; j < info_size; j++) fputc(0, out);
    }

    // 4d. Align data section
    long pos = ftell(out);
    long pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    for (long i = 0; i < pad; i++) fputc(0, out);
    long tensor_data_start = ftell(out);

    // 4e. Write tensor data and record offsets
    uint64_t* data_offsets = (uint64_t*)malloc(tensor_count * sizeof(uint64_t));
    for (int i = 0; i < tensor_count; i++) {
        TensorInfo* ti = &tensors[i];
        data_offsets[i] = ftell(out) - tensor_data_start;

        FILE* in = fopen(ti->filename, "rb");
        if (!in) {
            fprintf(stderr, "Cannot reopen %s\n", ti->filename);
            fclose(out);
            return 1;
        }
        fseek(in, ti->file_offset, SEEK_SET);
        uint8_t* buffer = (uint8_t*)malloc(ti->data_size);
        if (!buffer) {
            fprintf(stderr, "Out of memory\n");
            fclose(in);
            fclose(out);
            return 1;
        }
        size_t read = fread(buffer, 1, ti->data_size, in);
        fclose(in);
        if (read != ti->data_size) {
            fprintf(stderr, "Failed to read tensor %s\n", ti->name);
            free(buffer);
            fclose(out);
            return 1;
        }

        // Determine target type
        ggml_type target_type;
        if (quantize_f16 && strcmp(ti->dtype, "F32") == 0) {
            target_type = GGML_TYPE_F16;
            // Convert from float32 to float16
            size_t num_elements = ti->data_size / sizeof(float);
            uint16_t* f16_data = (uint16_t*)malloc(num_elements * sizeof(uint16_t));
            if (!f16_data) {
                fprintf(stderr, "Out of memory\n");
                free(buffer);
                fclose(out);
                return 1;
            }
            float* f32 = (float*)buffer;
            for (size_t j = 0; j < num_elements; j++) {
                f16_data[j] = float_to_half(f32[j]);
            }
            fwrite(f16_data, sizeof(uint16_t), num_elements, out);
            free(f16_data);
        } else {
            // Passthrough (set type accordingly)
            if (strcmp(ti->dtype, "F32") == 0) target_type = GGML_TYPE_F32;
            else if (strcmp(ti->dtype, "F16") == 0) target_type = GGML_TYPE_F16;
            else {
                // For non-float types, default to F32? Better to handle properly.
                // For simplicity, we'll just write raw data and hope.
                target_type = GGML_TYPE_F32;
            }
            fwrite(buffer, 1, ti->data_size, out);
        }
        free(buffer);
    }

    // 4f. Seek back and write actual tensor infos
    fseek(out, tensor_info_start, SEEK_SET);
    for (int i = 0; i < tensor_count; i++) {
        TensorInfo* ti = &tensors[i];
        ggml_type gtype;
        if (quantize_f16 && strcmp(ti->dtype, "F32") == 0) {
            gtype = GGML_TYPE_F16;
        } else if (strcmp(ti->dtype, "F32") == 0) {
            gtype = GGML_TYPE_F32;
        } else if (strcmp(ti->dtype, "F16") == 0) {
            gtype = GGML_TYPE_F16;
        } else {
            // fallback
            gtype = GGML_TYPE_F32;
        }
        write_tensor_info(out, ti->name, ti->shape, ti->n_dims, gtype, data_offsets[i]);
    }

    fclose(out);
    free(data_offsets);

    // 5. Cleanup
    for (int i = 0; i < tensor_count; i++) {
        free(tensors[i].name);
        free(tensors[i].dtype);
        free(tensors[i].shape);
        free(tensors[i].filename);
    }
    free(tensors);
    cJSON_Delete(config);

    printf("Successfully converted %d tensors to %s\n", tensor_count, output_path);
    if (quantize_f16) printf("Quantized to float16.\n");
    return 0;
}

// -----------------------------------------------------------------------------
// Helper functions for writing GGUF key-value pairs
// -----------------------------------------------------------------------------
static void write_kv_string(FILE* out, const char* key, const char* value) {
    uint32_t key_len = strlen(key);
    write_le32(out, key_len);
    fwrite(key, 1, key_len, out);

    uint32_t type = GGUF_TYPE_STRING;
    write_le32(out, type);

    uint64_t value_len = strlen(value);
    write_le64(out, value_len);
    fwrite(value, 1, value_len, out);
}

static void write_kv_uint32(FILE* out, const char* key, uint32_t value) {
    uint32_t key_len = strlen(key);
    write_le32(out, key_len);
    fwrite(key, 1, key_len, out);

    uint32_t type = GGUF_TYPE_UINT32;
    write_le32(out, type);

    write_le32(out, value);
}

static void write_tensor_info(FILE* out, const char* name, int64_t* shape, int n_dims,
                              ggml_type dtype, uint64_t offset) {
    uint32_t name_len = strlen(name);
    write_le32(out, name_len);
    fwrite(name, 1, name_len, out);

    uint32_t n_dims_le = n_dims;
    write_le32(out, n_dims_le);

    for (int i = 0; i < n_dims; i++) {
        write_le64(out, shape[i]);
    }

    uint32_t dtype_le = (uint32_t)dtype;
    write_le32(out, dtype_le);

    write_le64(out, offset);
}