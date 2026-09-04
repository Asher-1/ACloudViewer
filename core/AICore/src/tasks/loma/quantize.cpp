// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/loma/quantize.hpp"

#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace aicore::loma {
namespace {

void SetError(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

bool ParseType(std::string name, ggml_type* type) {
    for (char& c : name) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    if (name == "f16") {
        *type = GGML_TYPE_F16;
    } else if (name == "q8_0") {
        *type = GGML_TYPE_Q8_0;
    } else {
        return false;
    }
    return true;
}

bool ToFloat(const ggml_tensor* tensor, std::vector<float>* output) {
    const int64_t count = ggml_nelements(tensor);
    output->resize(static_cast<size_t>(count));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(output->data(), tensor->data,
                    output->size() * sizeof(float));
        return true;
    }
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(tensor->data),
                              output->data(), count);
        return true;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(tensor->type);
    if (traits == nullptr || traits->to_float == nullptr ||
        tensor->ne[0] <= 0) {
        return false;
    }
    const int64_t rows = count / tensor->ne[0];
    const size_t bytes_per_row = ggml_row_size(tensor->type, tensor->ne[0]);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(
                static_cast<const uint8_t*>(tensor->data) + row * bytes_per_row,
                output->data() + row * tensor->ne[0], tensor->ne[0]);
    }
    return true;
}

bool IsWeight(const char* name, const ggml_tensor* tensor) {
    const std::string tensor_name(name);
    if (ggml_n_dims(tensor) < 2) return false;
    if (tensor_name.size() > 7 &&
        tensor_name.compare(tensor_name.size() - 7, 7, ".weight") == 0) {
        return true;
    }
    // The ONNX converter gives unnamed DINO MatMul initializers stable
    // `loma.val_<node>` names. They are all rank-2 projection weights; their
    // original names are retained in loma.descriptor_g.*_weights metadata.
    return tensor_name.rfind("loma.val_", 0) == 0 && ggml_n_dims(tensor) == 2;
}

bool SupportsQ8(const ggml_tensor* tensor) {
    // LoMa's convolution paths consume dense F32/F16 kernels. Q8 is valid for
    // the ViT and matcher mul_mat weights only; silently quantizing
    // convolutions would create an artifact the graph cannot execute.
    return ggml_n_dims(tensor) == 2 && tensor->ne[0] % 32 == 0;
}

}  // namespace

bool QuantizeModel(const std::string& input_gguf,
                   const std::string& output_gguf,
                   const std::string& type_name,
                   std::string* error) {
    ggml_type requested = GGML_TYPE_F32;
    if (!ParseType(type_name, &requested)) {
        SetError(error, "unknown LoMa type '" + type_name +
                                "' (expected f16 or q8_0)");
        return false;
    }

    ggml_context* input_context = nullptr;
    gguf_init_params input_params{/*no_alloc=*/false, /*ctx=*/&input_context};
    gguf_context* input = gguf_init_from_file(input_gguf.c_str(), input_params);
    if (input == nullptr || input_context == nullptr) {
        SetError(error, "failed to open LoMa GGUF: " + input_gguf);
        if (input != nullptr) gguf_free(input);
        if (input_context != nullptr) ggml_free(input_context);
        return false;
    }
    const int64_t architecture = gguf_find_key(input, "general.architecture");
    if (architecture < 0 ||
        std::strcmp(gguf_get_val_str(input, architecture), "loma") != 0) {
        SetError(error, "input is not a LoMa GGUF");
        gguf_free(input);
        ggml_free(input_context);
        return false;
    }

    const int64_t tensor_count = gguf_get_n_tensors(input);
    const int64_t rope_frequency_key =
            gguf_find_key(input, "loma.matcher.rope_frequency_tensor");
    // Matchers name this initializer differently across B/R/L/G exports. The
    // runtime transposes it before mul_mat, an operation unsupported for Q8.
    // Read the graph-derived GGUF contract instead of relying on val_* names.
    const char* rope_frequency_tensor =
            rope_frequency_key < 0
                    ? nullptr
                    : gguf_get_val_str(input, rope_frequency_key);
    gguf_context* output = gguf_init_empty();
    gguf_set_kv(output, input);
    ggml_init_params output_params{
            ggml_tensor_overhead() * static_cast<size_t>(tensor_count + 8),
            nullptr, /*no_alloc=*/true};
    ggml_context* output_context = ggml_init(output_params);
    if (output_context == nullptr) {
        SetError(error, "failed to initialize LoMa quantization context");
        gguf_free(output);
        gguf_free(input);
        ggml_free(input_context);
        return false;
    }

    ggml_quantize_init(requested);
    std::vector<std::vector<uint8_t>> storage;
    storage.reserve(static_cast<size_t>(tensor_count));
    std::vector<float> floats;
    int converted = 0;
    bool success = true;
    for (int64_t index = 0; index < tensor_count && success; ++index) {
        const char* name = gguf_get_tensor_name(input, index);
        ggml_tensor* source = ggml_get_tensor(input_context, name);
        if (source == nullptr || source->data == nullptr) {
            SetError(error, std::string("LoMa tensor has no data: ") + name);
            success = false;
            break;
        }
        bool rewrite = IsWeight(name, source);
        if (requested == GGML_TYPE_Q8_0 && rewrite &&
            (!SupportsQ8(source) ||
             (rope_frequency_tensor != nullptr &&
              std::strcmp(name, rope_frequency_tensor) == 0))) {
            rewrite = false;
        }
        ggml_type destination_type = rewrite ? requested : source->type;
        const int64_t dimensions[GGML_MAX_DIMS] = {
                source->ne[0], source->ne[1], source->ne[2], source->ne[3]};
        std::vector<uint8_t> bytes;
        if (rewrite) {
            if (!ToFloat(source, &floats)) {
                SetError(error,
                         std::string("cannot read LoMa tensor: ") + name);
                success = false;
                break;
            }
            const int64_t row_size = source->ne[0];
            const int64_t rows = ggml_nelements(source) / row_size;
            const size_t expected = ggml_row_size(destination_type, row_size) *
                                    static_cast<size_t>(rows);
            bytes.resize(expected);
            size_t written = 0;
            if (destination_type == GGML_TYPE_F16) {
                for (int64_t row = 0; row < rows; ++row) {
                    ggml_fp32_to_fp16_row(
                            floats.data() + row * row_size,
                            reinterpret_cast<ggml_fp16_t*>(bytes.data()) +
                                    row * row_size,
                            row_size);
                }
                written = expected;
            } else {
                written = ggml_quantize_chunk(destination_type, floats.data(),
                                              bytes.data(), 0, rows, row_size,
                                              nullptr);
            }
            if (written != expected) {
                SetError(error,
                         std::string("LoMa quantized byte count mismatch: ") +
                                 name);
                success = false;
                break;
            }
            ++converted;
        } else {
            bytes.assign(static_cast<const uint8_t*>(source->data),
                         static_cast<const uint8_t*>(source->data) +
                                 ggml_nbytes(source));
        }
        ggml_tensor* destination =
                ggml_new_tensor(output_context, destination_type,
                                ggml_n_dims(source), dimensions);
        ggml_set_name(destination, name);
        storage.emplace_back(std::move(bytes));
        destination->data = storage.back().data();
        gguf_add_tensor(output, destination);
        // gguf_add_tensor copies the descriptor. Bind the GGUF-owned copy as
        // well, rather than relying on the temporary ggml descriptor pointer.
        gguf_set_tensor_data(output, name, storage.back().data());
    }
    if (success && converted == 0) {
        SetError(error,
                 "no LoMa weights support requested quantization; refusing a "
                 "no-op artifact");
        success = false;
    }
    if (success &&
        !gguf_write_to_file(output, output_gguf.c_str(), /*only_meta=*/true)) {
        SetError(error, "failed to write LoMa GGUF metadata: " + output_gguf);
        success = false;
    }
    if (success) {
        std::ofstream stream(output_gguf, std::ios::binary | std::ios::app);
        const size_t alignment = gguf_get_alignment(output);
        // New gguf_context instances in ggml 0.21 do not populate
        // gguf_get_data_offset() until a file is re-opened. The documented
        // two-phase writer's actual boundary is its padded metadata size.
        const size_t data_offset = gguf_get_meta_size(output);
        if (!stream || static_cast<size_t>(stream.tellp()) != data_offset) {
            SetError(error, "LoMa GGUF metadata has an invalid data offset");
            success = false;
        }
        std::vector<char> padding(alignment, 0);
        for (int64_t index = 0; success && index < tensor_count; ++index) {
            const std::vector<uint8_t>& bytes =
                    storage[static_cast<size_t>(index)];
            if (!bytes.empty()) {
                stream.write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
            }
            const size_t pad =
                    (alignment - bytes.size() % alignment) % alignment;
            if (pad != 0)
                stream.write(padding.data(), static_cast<std::streamsize>(pad));
            if (!stream) {
                SetError(error, "failed while writing LoMa GGUF tensor data");
                success = false;
            }
        }
        stream.close();
        if (success) {
            ggml_context* verify_context = nullptr;
            gguf_init_params verify_params{/*no_alloc=*/true, &verify_context};
            gguf_context* verify =
                    gguf_init_from_file(output_gguf.c_str(), verify_params);
            if (verify == nullptr || verify_context == nullptr ||
                gguf_get_n_tensors(verify) != tensor_count) {
                SetError(error, "LoMa GGUF post-write validation failed");
                success = false;
            }
            if (verify != nullptr) gguf_free(verify);
            if (verify_context != nullptr) ggml_free(verify_context);
        }
    }
    if (success && error != nullptr) error->clear();
    ggml_free(output_context);
    gguf_free(output);
    gguf_free(input);
    ggml_free(input_context);
    return success;
}

}  // namespace aicore::loma
