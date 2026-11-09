// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "tasks/vlx_seek/vlx_gguf_loader.h"

#include <FileSystem.h>

#include <algorithm>
#include <cstdio>
#include <fstream>

#include "tasks/vlx_seek/vlx_gguf_loader.h"

namespace vlx {

namespace {

uint16_t read_le_u16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
uint32_t read_le_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
}
uint64_t read_le_u64(const uint8_t* p) {
    return uint64_t(read_le_u32(p)) | (uint64_t(read_le_u32(p + 4)) << 32);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h & 0x7c00u) >> 10;
    const uint32_t man = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal
            uint32_t e = 0;
            uint32_t m = man;
            while (!(m & 0x0400u)) {
                m <<= 1;
                ++e;
            }
            m &= 0x03ffu;
            bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

size_t q8_0_block_size() { return 32; }
size_t q8_0_type_size() { return 32 + 2; }  // fp16 scale + 32 * int8

void dequant_q8_0(const uint8_t* src, size_t n_blocks, float* dst) {
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = src + b * q8_0_type_size();
        const float d = f16_to_f32(read_le_u16(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        float* out = dst + b * q8_0_block_size();
        for (size_t i = 0; i < 32; ++i) out[i] = d * float(qs[i]);
    }
}

}  // namespace

size_t GgufFile::type_size(GgmlDType t) {
    switch (t) {
        case GgmlDType::F32:
        case GgmlDType::I32:
            return 4;
        case GgmlDType::F16:
        case GgmlDType::I16:
            return 2;
        case GgmlDType::I8:
            return 1;
        case GgmlDType::I64:
            return 8;
        case GgmlDType::Q8_0:
            return q8_0_type_size();
        default:
            return 0;  // unsupported for the vision sub-chain
    }
}

bool GgufFile::load(const std::string& path, std::string* err) {
    std::ifstream in = cloudViewer::utility::filesystem::OpenInputStream(
            path, std::ios::binary);
    if (!in) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };

    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) return fail("bad GGUF magic");
    // Header layout: magic(4) | version(u32) | n_tensors(u64) | n_kv(u64).
    uint8_t v4[4];
    in.read(reinterpret_cast<char*>(v4), 4);
    const uint32_t version = read_le_u32(v4);
    if (version < 3) return fail("GGUF v2 and older not supported");
    uint8_t h8[8];
    in.read(reinterpret_cast<char*>(h8), 8);
    const uint64_t n_tensors = read_le_u64(h8);
    in.read(reinterpret_cast<char*>(h8), 8);
    const uint64_t n_kv = read_le_u64(h8);

    auto read_str = [&](std::string* s) {
        uint8_t lb[8];
        in.read(reinterpret_cast<char*>(lb), 8);
        const uint64_t len = read_le_u64(lb);
        s->resize(size_t(len));
        in.read(s->data(), std::streamsize(len));
    };

    auto read_value = [&](GgufType t, const std::string& key) {
        switch (t) {
            case GgufType::Str:
                read_str(&kv_str_[key]);
                break;
            case GgufType::U8:
            case GgufType::Bool: {
                uint8_t v;
                in.read(reinterpret_cast<char*>(&v), 1);
                kv_i_[key] = v;
                break;
            }
            case GgufType::I8: {
                int8_t v;
                in.read(reinterpret_cast<char*>(&v), 1);
                kv_i_[key] = v;
                break;
            }
            case GgufType::U16: {
                uint16_t v;
                in.read(reinterpret_cast<char*>(&v), 2);
                kv_i_[key] = v;
                break;
            }
            case GgufType::I16: {
                int16_t v;
                in.read(reinterpret_cast<char*>(&v), 2);
                kv_i_[key] = v;
                break;
            }
            case GgufType::U32: {
                uint32_t v;
                in.read(reinterpret_cast<char*>(&v), 4);
                kv_i_[key] = v;
                break;
            }
            case GgufType::I32: {
                int32_t v;
                in.read(reinterpret_cast<char*>(&v), 4);
                kv_i_[key] = v;
                break;
            }
            case GgufType::U64: {
                uint64_t v;
                in.read(reinterpret_cast<char*>(&v), 8);
                kv_i_[key] = int64_t(v);
                break;
            }
            case GgufType::I64: {
                int64_t v;
                in.read(reinterpret_cast<char*>(&v), 8);
                kv_i_[key] = v;
                break;
            }
            case GgufType::F32: {
                float v;
                in.read(reinterpret_cast<char*>(&v), 4);
                kv_f_[key] = v;
                break;
            }
            case GgufType::F64: {
                double v;
                in.read(reinterpret_cast<char*>(&v), 8);
                kv_f_[key] = v;
                break;
            }
            case GgufType::Arr: {
                // header = u32 element type + u64 count (12 bytes)
                uint8_t t8[12];
                in.read(reinterpret_cast<char*>(t8), 12);
                const GgufType et = GgufType(read_le_u32(t8));
                const uint64_t n = read_le_u64(t8 + 4);
                // Strings and typed arrays: only the keys we need are kept;
                // everything else is skipped by its byte size.
                if (et == GgufType::Str) {
                    std::vector<std::string> vals;
                    for (uint64_t i = 0; i < n; ++i) {
                        std::string v;
                        read_str(&v);
                        vals.push_back(v);
                    }
                    kv_str_[key] =
                            vals.empty() ? "" : vals[0];  // first element wins
                } else {
                    const size_t ts = [&] {
                        switch (et) {
                            case GgufType::U8:
                            case GgufType::Bool:
                            case GgufType::I8:
                                return 1;
                            case GgufType::U16:
                            case GgufType::I16:
                                return 2;
                            case GgufType::U32:
                            case GgufType::I32:
                            case GgufType::F32:
                                return 4;
                            default:
                                return 8;
                        }
                    }();
                    std::vector<uint8_t> buf(size_t(n) * ts);
                    in.read(reinterpret_cast<char*>(buf.data()),
                            std::streamsize(buf.size()));
                    if (et == GgufType::F32 && n >= 1)
                        kv_f_[key] = *reinterpret_cast<float*>(buf.data());
                    else if (n >= 1) {
                        const uint8_t* p = buf.data();
                        kv_i_[key] = et == GgufType::I64 || et == GgufType::U64
                                             ? int64_t(read_le_u64(p))
                                             : int64_t(read_le_u32(p));
                    }
                }
                break;
            }
        }
    };

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        read_str(&key);
        uint8_t t4[4];
        in.read(reinterpret_cast<char*>(t4), 4);
        const GgufType t = GgufType(read_le_u32(t4));
        kv_order_.emplace_back(key, t);
        read_value(t, key);
    }
    arch_ = str("general.architecture", "vlxvision");
    // Tensor entry: name(str) | n_dims(u32) | ne[u64 x n_dims] | type(u32) |
    // offset(u64) NOTE: dimensions are 8 bytes each in GGUF v3.
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        read_str(&ti.name);
        uint8_t d4[4];
        in.read(reinterpret_cast<char*>(d4), 4);
        ti.n_dims = read_le_u32(d4);
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            uint8_t d8[8];
            in.read(reinterpret_cast<char*>(d8), 8);
            ti.ne[d] = read_le_u64(d8);
        }
        in.read(reinterpret_cast<char*>(d4), 4);
        ti.dtype = GgmlDType(read_le_u32(d4));
        uint8_t ob[8];
        in.read(reinterpret_cast<char*>(ob), 8);
        ti.offset = read_le_u64(ob);
        by_name_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }

    // Align each tensor's absolute offset: the data section starts on the
    // GGUF alignment boundary (32 by default, overridable via KV).
    uint64_t pos = uint64_t(in.tellg());
    const uint64_t align = uint64_t(i32("general.alignment", 32));
    pos = (pos + align - 1) & ~(align - 1);
    for (auto& t : tensors_) {
        t.offset += pos;
        const size_t ts = type_size(t.dtype);
        const uint64_t n = [&] {
            uint64_t n = 1;
            for (uint32_t d = 0; d < t.n_dims; ++d) n *= t.ne[d];
            if (t.dtype == GgmlDType::Q8_0)
                n = (n / 32) * q8_0_type_size();
            else
                n *= ts;
            return n;
        }();
        t.bytes.resize(size_t(n));
        in.seekg(std::streamsize(t.offset));
        in.read(reinterpret_cast<char*>(t.bytes.data()), std::streamsize(n));
    }
    return true;
}

std::string GgufFile::str(const std::string& key,
                          const std::string& def) const {
    auto it = kv_str_.find(key);
    return it == kv_str_.end() ? def : it->second;
}

int64_t GgufFile::i32(const std::string& key, int64_t def) const {
    auto it = kv_i_.find(key);
    return it == kv_i_.end() ? def : it->second;
}

double GgufFile::f32v(const std::string& key, double def) const {
    auto it = kv_f_.find(key);
    return it == kv_f_.end() ? def : it->second;
}

const TensorInfo* GgufFile::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const float* GgufFile::tensor_f32(const std::string& name) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return nullptr;
    TensorInfo* t = const_cast<TensorInfo*>(&tensors_[it->second]);
    if (!t->f32.empty()) return t->f32.data();
    uint64_t n_elem = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) n_elem *= t->ne[d];
    t->f32.resize(size_t(n_elem));
    switch (t->dtype) {
        case GgmlDType::F32:
            std::memcpy(t->f32.data(), t->bytes.data(), size_t(n_elem) * 4);
            break;
        case GgmlDType::F16:
            for (uint64_t i = 0; i < n_elem; ++i)
                t->f32[size_t(i)] = f16_to_f32(
                        read_le_u16(t->bytes.data() + 2 * size_t(i)));
            break;
        case GgmlDType::Q8_0:
            dequant_q8_0(t->bytes.data(), size_t(n_elem) / 32, t->f32.data());
            break;
        default:
            t->f32.clear();
            return nullptr;
    }
    return t->f32.data();
}

}  // namespace vlx
