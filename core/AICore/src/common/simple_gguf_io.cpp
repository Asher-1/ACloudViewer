// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/simple_gguf_io.hpp"

#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace aicore {
namespace common {
namespace {

constexpr uint32_t kMagic = 0x46554747;
constexpr uint32_t kDtypeF32 = 0;
constexpr uint32_t kDtypeF16 = 1;
constexpr uint32_t kDtypeQ8 = 2;

bool ReadString(std::ifstream& in, std::string* out) {
    uint64_t len = 0;
    if (!in.read(reinterpret_cast<char*>(&len), sizeof(len))) {
        return false;
    }
    if (len > 4096) {
        return false;
    }
    out->resize(static_cast<size_t>(len));
    if (len == 0) {
        return true;
    }
    return static_cast<bool>(
            in.read(out->data(), static_cast<std::streamsize>(len)));
}

void SetError(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool EndsWithWeight(const std::string& name) {
    static const char* suffix = "_weight";
    if (name.size() < 7) {
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        const char a = name[name.size() - 7 + i];
        const char b = suffix[i];
        if (std::tolower(static_cast<unsigned char>(a)) !=
            std::tolower(static_cast<unsigned char>(b))) {
            return false;
        }
    }
    return true;
}

bool DequantPayload(uint32_t dtype,
                    const std::vector<int64_t>& dims,
                    const std::vector<uint8_t>& payload,
                    std::vector<float>* out) {
    int64_t count = 1;
    for (int64_t d : dims) {
        count *= d;
    }
    out->resize(static_cast<size_t>(count));
    if (dtype == kDtypeF32) {
        if (payload.size() != static_cast<size_t>(count) * sizeof(float)) {
            return false;
        }
        std::memcpy(out->data(), payload.data(), payload.size());
        return true;
    }
    if (dtype == kDtypeF16) {
        if (payload.size() !=
            static_cast<size_t>(count) * sizeof(ggml_fp16_t)) {
            return false;
        }
        ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t*>(payload.data()),
                out->data(), count);
        return true;
    }
    if (dtype == kDtypeQ8) {
        const int64_t row = dims.empty() ? count : dims[0];
        const int64_t rows = count / row;
        const ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        if (traits == nullptr || traits->to_float == nullptr) {
            return false;
        }
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, row);
        if (payload.size() != row_bytes * static_cast<size_t>(rows)) {
            return false;
        }
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(payload.data() + r * row_bytes,
                             out->data() + r * row, row);
        }
        return true;
    }
    return false;
}

bool WriteString(std::ofstream& out, const std::string& s) {
    const uint64_t len = s.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len == 0) {
        return static_cast<bool>(out);
    }
    out.write(s.data(), static_cast<std::streamsize>(len));
    return static_cast<bool>(out);
}

bool WriteTensor(std::ofstream& out,
                 const std::string& name,
                 uint32_t dtype,
                 const std::vector<int64_t>& dims,
                 const std::vector<uint8_t>& payload) {
    if (!WriteString(out, name)) {
        return false;
    }
    const uint32_t ndim = static_cast<uint32_t>(dims.size());
    out.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int64_t dim : dims) {
        const uint64_t d = static_cast<uint64_t>(dim);
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(out);
}

bool ParseType(const std::string& type_name, uint32_t* dtype, ggml_type* ggml) {
    std::string name = type_name;
    for (char& c : name) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (name == "f16") {
        *dtype = kDtypeF16;
        *ggml = GGML_TYPE_F16;
        return true;
    }
    if (name == "q8_0") {
        *dtype = kDtypeQ8;
        *ggml = GGML_TYPE_Q8_0;
        return true;
    }
    return false;
}

}  // namespace

bool load_simple_gguf_f32(const std::string& path,
                          SimpleFloatMap* tensors,
                          std::string* error) {
    if (tensors == nullptr) {
        SetError(error, "null tensors");
        return false;
    }
    tensors->clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        SetError(error, "failed to open GGUF: " + path);
        return false;
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t count = 0;
    uint64_t meta = 0;
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        !in.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
        !in.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        !in.read(reinterpret_cast<char*>(&meta), sizeof(meta))) {
        SetError(error, "truncated GGUF header");
        return false;
    }
    if (magic != kMagic) {
        SetError(error, "invalid GGUF magic");
        return false;
    }
    (void)version;
    for (uint64_t m = 0; m < meta; ++m) {
        std::string key;
        if (!ReadString(in, &key)) {
            SetError(error, "truncated metadata");
            return false;
        }
        uint32_t vtype = 0;
        if (!in.read(reinterpret_cast<char*>(&vtype), sizeof(vtype))) {
            return false;
        }
        if (vtype == 8) {
            std::string skip;
            if (!ReadString(in, &skip)) {
                return false;
            }
        } else if (vtype == 4) {
            in.seekg(8, std::ios::cur);
        } else if (vtype == 5) {
            in.seekg(4, std::ios::cur);
        } else if (vtype == 6) {
            in.seekg(1, std::ios::cur);
        } else {
            in.seekg(8, std::ios::cur);
        }
    }
    for (uint64_t t = 0; t < count; ++t) {
        std::string name;
        if (!ReadString(in, &name)) {
            SetError(error, "truncated tensor name");
            return false;
        }
        uint32_t dtype = 0;
        uint32_t ndim = 0;
        if (!in.read(reinterpret_cast<char*>(&dtype), sizeof(dtype)) ||
            !in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim))) {
            return false;
        }
        std::vector<int64_t> dims(static_cast<size_t>(ndim));
        int64_t elems = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            uint64_t dim = 0;
            if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
                return false;
            }
            dims[d] = static_cast<int64_t>(dim);
            elems *= dims[d];
        }
        size_t payload_bytes = 0;
        if (dtype == kDtypeF32) {
            payload_bytes = static_cast<size_t>(elems) * sizeof(float);
        } else if (dtype == kDtypeF16) {
            payload_bytes = static_cast<size_t>(elems) * sizeof(ggml_fp16_t);
        } else if (dtype == kDtypeQ8) {
            const int64_t row = dims.empty() ? elems : dims[0];
            const int64_t rows = elems / row;
            payload_bytes = ggml_row_size(GGML_TYPE_Q8_0, row) *
                            static_cast<size_t>(rows);
        } else {
            SetError(error, "unsupported dtype in " + name);
            return false;
        }
        std::vector<uint8_t> payload(payload_bytes);
        if (!in.read(reinterpret_cast<char*>(payload.data()),
                     static_cast<std::streamsize>(payload_bytes))) {
            SetError(error, "truncated payload: " + name);
            return false;
        }
        std::vector<float> floats;
        if (!DequantPayload(dtype, dims, payload, &floats)) {
            SetError(error, "dequant failed: " + name);
            return false;
        }
        (*tensors)[name] = std::move(floats);
    }
    return true;
}

bool quantize_simple_gguf_weights(const std::string& input_gguf,
                                  const std::string& output_gguf,
                                  const std::string& type_name,
                                  std::string* error) {
    uint32_t out_dtype = 0;
    ggml_type ggml_target = GGML_TYPE_F32;
    if (!ParseType(type_name, &out_dtype, &ggml_target)) {
        SetError(error, "unknown type (expected f16 or q8_0)");
        return false;
    }

    std::ifstream in(input_gguf, std::ios::binary);
    if (!in) {
        SetError(error, "failed to open input");
        return false;
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t count = 0;
    uint64_t meta = 0;
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        !in.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
        !in.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        !in.read(reinterpret_cast<char*>(&meta), sizeof(meta))) {
        SetError(error, "truncated header");
        return false;
    }
    if (magic != kMagic) {
        SetError(error, "invalid GGUF magic");
        return false;
    }

    for (uint64_t m = 0; m < meta; ++m) {
        std::string key;
        if (!ReadString(in, &key)) {
            SetError(error, "truncated metadata");
            return false;
        }
        uint32_t vtype = 0;
        if (!in.read(reinterpret_cast<char*>(&vtype), sizeof(vtype))) {
            return false;
        }
        if (vtype == 8) {
            std::string skip;
            if (!ReadString(in, &skip)) {
                return false;
            }
        } else if (vtype == 4) {
            in.seekg(8, std::ios::cur);
        } else if (vtype == 5) {
            in.seekg(4, std::ios::cur);
        } else if (vtype == 6) {
            in.seekg(1, std::ios::cur);
        } else {
            in.seekg(8, std::ios::cur);
        }
    }

    struct StoredTensor {
        std::string name;
        uint32_t dtype = kDtypeF32;
        std::vector<int64_t> dims;
        std::vector<float> floats;
    };
    std::vector<StoredTensor> tensors;
    tensors.reserve(static_cast<size_t>(count));

    for (uint64_t t = 0; t < count; ++t) {
        StoredTensor st;
        if (!ReadString(in, &st.name)) {
            SetError(error, "truncated tensor name");
            return false;
        }
        if (!in.read(reinterpret_cast<char*>(&st.dtype), sizeof(st.dtype))) {
            return false;
        }
        uint32_t ndim = 0;
        if (!in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim))) {
            return false;
        }
        st.dims.resize(ndim);
        int64_t elems = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            uint64_t dim = 0;
            if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
                return false;
            }
            st.dims[d] = static_cast<int64_t>(dim);
            elems *= st.dims[d];
        }
        size_t payload_bytes = 0;
        if (st.dtype == kDtypeF32) {
            payload_bytes = static_cast<size_t>(elems) * sizeof(float);
        } else if (st.dtype == kDtypeF16) {
            payload_bytes = static_cast<size_t>(elems) * sizeof(ggml_fp16_t);
        } else if (st.dtype == kDtypeQ8) {
            const int64_t row = st.dims.empty() ? elems : st.dims[0];
            payload_bytes = ggml_row_size(GGML_TYPE_Q8_0, row) *
                            static_cast<size_t>(elems / row);
        } else {
            SetError(error, "unsupported dtype: " + st.name);
            return false;
        }
        std::vector<uint8_t> payload(payload_bytes);
        if (!in.read(reinterpret_cast<char*>(payload.data()),
                     static_cast<std::streamsize>(payload_bytes))) {
            SetError(error, "truncated payload: " + st.name);
            return false;
        }
        if (!DequantPayload(st.dtype, st.dims, payload, &st.floats)) {
            SetError(error, "dequant failed: " + st.name);
            return false;
        }
        tensors.push_back(std::move(st));
    }

    ggml_quantize_init(ggml_target);
    int quantized = 0;
    std::ofstream out(output_gguf, std::ios::binary);
    if (!out) {
        SetError(error, "failed to open output");
        return false;
    }
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&meta), sizeof(meta));

    for (const auto& st : tensors) {
        uint32_t dtype = kDtypeF32;
        std::vector<uint8_t> payload(
                reinterpret_cast<const uint8_t*>(st.floats.data()),
                reinterpret_cast<const uint8_t*>(st.floats.data()) +
                        st.floats.size() * sizeof(float));

        if (EndsWithWeight(st.name) && st.dims.size() >= 2) {
            const int64_t row = st.dims[0];
            const int64_t rows = static_cast<int64_t>(st.floats.size()) / row;
            if (out_dtype == kDtypeF16) {
                payload.resize(static_cast<size_t>(st.floats.size()) *
                               sizeof(ggml_fp16_t));
                for (int64_t r = 0; r < rows; ++r) {
                    ggml_fp32_to_fp16_row(
                            st.floats.data() + r * row,
                            reinterpret_cast<ggml_fp16_t*>(payload.data()) +
                                    r * row,
                            row);
                }
                dtype = kDtypeF16;
                ++quantized;
            } else if (out_dtype == kDtypeQ8 && row % 32 == 0) {
                const size_t expected = ggml_row_size(GGML_TYPE_Q8_0, row) *
                                        static_cast<size_t>(rows);
                payload.resize(expected);
                const size_t written = ggml_quantize_chunk(
                        GGML_TYPE_Q8_0, st.floats.data(), payload.data(), 0,
                        rows, row, nullptr);
                if (written != expected) {
                    SetError(error, "q8 size mismatch: " + st.name);
                    return false;
                }
                dtype = kDtypeQ8;
                ++quantized;
            }
        }
        if (!WriteTensor(out, st.name, dtype, st.dims, payload)) {
            SetError(error, "write failed: " + st.name);
            return false;
        }
    }

    std::fprintf(stderr, "simple_gguf quantize: %d weights -> %s\n", quantized,
                 type_name.c_str());
    return true;
}

}  // namespace common
}  // namespace aicore
