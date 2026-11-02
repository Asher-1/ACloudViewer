// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace vlx {

enum class GgufType : uint32_t {
    U8 = 0,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Arr,
    U64 = 10,
    I64 = 11,
    F64 = 12,  // gguf.h: UINT64=10, INT64=11, FLOAT64=12
};

enum class GgmlDType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    I8 = 16,
    I16 = 18,
    I32 = 20,
    I64 = 28,
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};  // ggml order: ne[0] fastest
    GgmlDType dtype = GgmlDType::F32;
    uint64_t offset = 0;             // absolute file offset (resolved on load)
    std::vector<uint8_t> bytes;      // raw payload
    mutable std::vector<float> f32;  // dequantized F32 (lazy, on demand)
};

class GgufFile {
public:
    bool load(const std::string& path, std::string* err = nullptr);

    // KV accessors.
    bool has(const std::string& key) const {
        return kv_str_.count(key) || kv_i_.count(key) || kv_f_.count(key);
    }
    std::string str(const std::string& key, const std::string& def = "") const;
    int64_t i32(const std::string& key, int64_t def = 0) const;
    double f32v(const std::string& key, double def = 0.0) const;

    // Tensor access.
    const TensorInfo* tensor(const std::string& name) const;
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    // Dequantized F32 view (F32 passthrough; F16/Q8_0/I32 converted on first
    // call; the lazy cache is mutable so this stays const-friendly).
    const float* tensor_f32(const std::string& name) const;

    static size_t type_size(GgmlDType t);

private:
    std::vector<std::pair<std::string, GgufType>> kv_order_;
    std::map<std::string, std::string> kv_str_;
    std::map<std::string, int64_t> kv_i_;
    std::map<std::string, double> kv_f_;
    std::map<std::string, size_t> by_name_;
    std::vector<TensorInfo> tensors_;
    std::string arch_;
};

}  // namespace vlx
