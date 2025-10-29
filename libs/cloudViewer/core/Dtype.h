// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Logging.h>

#include <cstring>
#include <string>

#include "cloudViewer/Macro.h"
#include "cloudViewer/core/Dispatch.h"

namespace cloudViewer {
namespace core {

class CLOUDVIEWER_API Dtype {
public:
    static const Dtype Undefined;
    static const Dtype Float32;
    static const Dtype Float64;
    static const Dtype Int8;
    static const Dtype Int16;
    static const Dtype Int32;
    static const Dtype Int64;
    static const Dtype UInt8;
    static const Dtype UInt16;
    static const Dtype UInt32;
    static const Dtype UInt64;
    static const Dtype Bool;

public:
    enum class DtypeCode {
        Undefined,
        Bool,  // Needed to distinguish bool from uint8_t.
        Int,
        UInt,
        Float,
        Object,
    };

    Dtype() : Dtype(DtypeCode::Undefined, 1, "Undefined") {}

    explicit Dtype(DtypeCode dtype_code,
                   int64_t byte_size,
                   const std::string &name);

    /// Convert from C++ types to Dtype. Known types are explicitly specialized,
    /// e.g. FromType<float>(). Unsupported type results in an exception.
    template <typename T>
    static inline const Dtype FromType() {
        utility::LogError("Unsupported data for Dtype::FromType.");
    }

    int64_t ByteSize() const { return byte_size_; }

    DtypeCode GetDtypeCode() const { return dtype_code_; }

    bool IsObject() const { return dtype_code_ == DtypeCode::Object; }

    std::string ToString() const { return name_; }

    bool operator==(const Dtype &other) const;

    bool operator!=(const Dtype &other) const;

private:
    static constexpr size_t max_name_len_ = 16;
    DtypeCode dtype_code_;
    int64_t byte_size_;
    char name_[max_name_len_];  // MSVC warns if std::string is exported to DLL.
};

CLOUDVIEWER_API extern const Dtype Undefined;
CLOUDVIEWER_API extern const Dtype Float32;
CLOUDVIEWER_API extern const Dtype Float64;
CLOUDVIEWER_API extern const Dtype Int8;
CLOUDVIEWER_API extern const Dtype Int16;
CLOUDVIEWER_API extern const Dtype Int32;
CLOUDVIEWER_API extern const Dtype Int64;
CLOUDVIEWER_API extern const Dtype UInt8;
CLOUDVIEWER_API extern const Dtype UInt16;
CLOUDVIEWER_API extern const Dtype UInt32;
CLOUDVIEWER_API extern const Dtype UInt64;
CLOUDVIEWER_API extern const Dtype Bool;

template <>
inline const Dtype Dtype::FromType<float>() {
    return Dtype::Float32;
}

template <>
inline const Dtype Dtype::FromType<double>() {
    return Dtype::Float64;
}

template <>
inline const Dtype Dtype::FromType<int8_t>() {
    return Dtype::Int8;
}

template <>
inline const Dtype Dtype::FromType<int16_t>() {
    return Dtype::Int16;
}

template <>
inline const Dtype Dtype::FromType<int32_t>() {
    return Dtype::Int32;
}

template <>
inline const Dtype Dtype::FromType<int64_t>() {
    return Dtype::Int64;
}

template <>
inline const Dtype Dtype::FromType<uint8_t>() {
    return Dtype::UInt8;
}

template <>
inline const Dtype Dtype::FromType<uint16_t>() {
    return Dtype::UInt16;
}

template <>
inline const Dtype Dtype::FromType<uint32_t>() {
    return Dtype::UInt32;
}

template <>
inline const Dtype Dtype::FromType<uint64_t>() {
    return Dtype::UInt64;
}

template <>
inline const Dtype Dtype::FromType<bool>() {
    return Dtype::Bool;
}

}  // namespace core
}  // namespace cloudViewer
