#include "types.hpp"

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace aicore {
namespace lightglue {
namespace {

bool ParseType(std::string name, ggml_type *type) {
  for (char &c : name) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
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

bool IsLinearWeight(const std::string &name, int last_layer) {
  static const std::regex expression(
      "^block\\.([0-9]+)\\.(self\\.(qkv|output|ffn\\.(input|output))|"
      "cross\\.(qk|value|output|ffn\\.(input|output)))\\.weight$");
  std::smatch match;
  if (!std::regex_match(name, match, expression)) {
    return false;
  }
  const int layer = std::stoi(match[1].str());
  return layer >= 0 && layer <= last_layer;
}

bool ToFloat(const ggml_tensor *tensor, std::vector<float> *output) {
  const int64_t count = ggml_nelements(tensor);
  output->resize(static_cast<size_t>(count));
  if (tensor->type == GGML_TYPE_F32) {
    std::memcpy(output->data(), tensor->data, output->size() * sizeof(float));
    return true;
  }
  if (tensor->type == GGML_TYPE_F16) {
    ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(tensor->data),
                          output->data(), count);
    return true;
  }
  const ggml_type_traits *traits = ggml_get_type_traits(tensor->type);
  if (traits == nullptr || traits->to_float == nullptr) {
    return false;
  }
  const int64_t row_size = tensor->ne[0];
  const int64_t rows = count / row_size;
  const size_t source_row_bytes = ggml_row_size(tensor->type, row_size);
  for (int64_t row = 0; row < rows; ++row) {
    traits->to_float(static_cast<const uint8_t *>(tensor->data) +
                         row * source_row_bytes,
                     output->data() + row * row_size, row_size);
  }
  return true;
}

void SetError(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
}

} // namespace

bool quantize_model(const std::string &input_gguf,
                   const std::string &output_gguf, const std::string &type_name,
                   std::string *error) {
  ggml_type requested;
  if (!ParseType(type_name, &requested)) {
    SetError(error, "unknown quantization type '" + type_name +
                        "' (expected f16 or q8_0)");
    return false;
  }

  ggml_context *input_context = nullptr;
  gguf_init_params input_params{/*no_alloc=*/false, /*ctx=*/&input_context};
  gguf_context *input = gguf_init_from_file(input_gguf.c_str(), input_params);
  if (input == nullptr || input_context == nullptr) {
    SetError(error, "failed to open input GGUF: " + input_gguf);
    if (input != nullptr)
      gguf_free(input);
    if (input_context != nullptr)
      ggml_free(input_context);
    return false;
  }

  gguf_context *output = gguf_init_empty();
  gguf_set_kv(output, input);
  const int64_t layer_key = gguf_find_key(input, "lightglue.block_count");
  if (layer_key < 0) {
    SetError(error, "missing GGUF key: lightglue.block_count");
    gguf_free(output);
    gguf_free(input);
    ggml_free(input_context);
    return false;
  }
  const int last_layer =
      static_cast<int>(gguf_get_val_u32(input, layer_key)) - 1;
  const int64_t tensor_count = gguf_get_n_tensors(input);
  ggml_init_params output_params{
      ggml_tensor_overhead() * static_cast<size_t>(tensor_count + 8), nullptr,
      /*no_alloc=*/true};
  ggml_context *output_context = ggml_init(output_params);
  if (output_context == nullptr) {
    SetError(error, "failed to initialize quantization context");
    gguf_free(output);
    gguf_free(input);
    ggml_free(input_context);
    return false;
  }

  ggml_quantize_init(requested);

  std::vector<std::vector<uint8_t>> storage;
  storage.reserve(static_cast<size_t>(tensor_count));
  std::vector<float> floats;
  bool success = true;
  int quantized = 0;
  int retained = 0;

  for (int64_t i = 0; i < tensor_count && success; ++i) {
    const char *name = gguf_get_tensor_name(input, i);
    ggml_tensor *source = ggml_get_tensor(input_context, name);
    if (source == nullptr || source->data == nullptr) {
      SetError(error, std::string("tensor has no data: ") + name);
      success = false;
      break;
    }

    ggml_type destination_type = source->type;
    bool rewrite =
        IsLinearWeight(name, last_layer) && ggml_n_dims(source) == 2;
    if (rewrite) {
      if (requested == GGML_TYPE_F16) {
        destination_type = requested;
      } else if (source->ne[0] % 32 == 0) {
        destination_type = requested;
      } else {
        rewrite = false;
      }
    }

    const int64_t dimensions[GGML_MAX_DIMS] = {source->ne[0], source->ne[1],
                                               source->ne[2], source->ne[3]};
    ggml_tensor *destination = nullptr;
    std::vector<uint8_t> bytes;
    if (rewrite) {
      if (!ToFloat(source, &floats)) {
        SetError(error, std::string("cannot dequantize tensor: ") + name);
        success = false;
        break;
      }
      const int64_t row_size = source->ne[0];
      const int64_t rows = ggml_nelements(source) / row_size;
      const size_t expected =
          ggml_row_size(destination_type, row_size) * static_cast<size_t>(rows);
      bytes.resize(expected);
      size_t written = 0;
      if (destination_type == GGML_TYPE_F16) {
        for (int64_t row = 0; row < rows; ++row) {
          ggml_fp32_to_fp16_row(floats.data() + row * row_size,
                                reinterpret_cast<ggml_fp16_t *>(bytes.data()) +
                                    row * row_size,
                                row_size);
        }
        written = expected;
      } else {
        written = ggml_quantize_chunk(destination_type, floats.data(),
                                      bytes.data(), 0, rows, row_size, nullptr);
      }
      if (written != expected) {
        SetError(error,
                 std::string("quantized byte count mismatch for: ") + name);
        success = false;
        break;
      }
      destination = ggml_new_tensor(output_context, destination_type,
                                    ggml_n_dims(source), dimensions);
      ++quantized;
    } else {
      bytes.assign(static_cast<const uint8_t *>(source->data),
                   static_cast<const uint8_t *>(source->data) +
                       ggml_nbytes(source));
      destination = ggml_new_tensor(output_context, source->type,
                                    ggml_n_dims(source), dimensions);
      ++retained;
    }
    ggml_set_name(destination, name);
    storage.emplace_back(std::move(bytes));
    destination->data = storage.back().data();
    gguf_add_tensor(output, destination);
  }

  if (success &&
      !gguf_write_to_file(output, output_gguf.c_str(), /*only_meta=*/false)) {
    SetError(error, "failed to write output GGUF: " + output_gguf);
    success = false;
  }
  if (success) {
    std::fprintf(stderr, "quantize: %d linear weights -> %s, %d retained",
                 quantized, ggml_type_name(requested), retained);
    std::fprintf(stderr, "\n");
    if (error != nullptr)
      error->clear();
  }

  ggml_free(output_context);
  gguf_free(output);
  gguf_free(input);
  ggml_free(input_context);
  return success;
}

}  // namespace lightglue
}  // namespace aicore
