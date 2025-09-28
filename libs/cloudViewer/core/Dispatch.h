// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Logging.h>

#include "cloudViewer/core/Dtype.h"

/// Call a numerical templated function based on Dtype. Wrap the function to
/// a lambda function to use DISPATCH_DTYPE_TO_TEMPLATE.
///
/// Before:
///     if (dtype == core::Float32) {
///         func<float>(args);
///     } else if (dtype == core::Float64) {
///         func<double>(args);
///     } else ...
///
/// Now:
///     DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
///        func<scalar_t>(args);
///     });
///
/// Inspired by:
///     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
#define DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, ...)                        \
    [&] {                                                             \
        if (DTYPE == cloudViewer::core::Float32) {                    \
            using scalar_t = float;                                   \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Float64) {             \
            using scalar_t = double;                                  \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int8) {                \
            using scalar_t = int8_t;                                  \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int16) {               \
            using scalar_t = int16_t;                                 \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int32) {               \
            using scalar_t = int32_t;                                 \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int64) {               \
            using scalar_t = int64_t;                                 \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt8) {               \
            using scalar_t = uint8_t;                                 \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt16) {              \
            using scalar_t = uint16_t;                                \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt32) {              \
            using scalar_t = uint32_t;                                \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt64) {              \
            using scalar_t = uint64_t;                                \
            return __VA_ARGS__();                                     \
        } else {                                                      \
            cloudViewer::utility::LogError("Unsupported data type."); \
        }                                                             \
    }()

#define DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(DTYPE, ...)    \
    [&] {                                                   \
        if (DTYPE == cloudViewer::core::Bool) {             \
            using scalar_t = bool;                          \
            return __VA_ARGS__();                           \
        } else {                                            \
            DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, __VA_ARGS__); \
        }                                                   \
    }()

#define DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(DTYPE, ...)                  \
    [&] {                                                             \
        if (DTYPE == cloudViewer::core::Float32) {                    \
            using scalar_t = float;                                   \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Float64) {             \
            using scalar_t = double;                                  \
            return __VA_ARGS__();                                     \
        } else {                                                      \
            cloudViewer::utility::LogError("Unsupported data type."); \
        }                                                             \
    }()

#define DISPATCH_FLOAT_INT_DTYPE_TO_TEMPLATE(FDTYPE, IDTYPE, ...)     \
    [&] {                                                             \
        if (FDTYPE == cloudViewer::core::Float32 &&                   \
            IDTYPE == cloudViewer::core::Int32) {                     \
            using scalar_t = float;                                   \
            using int_t = int32_t;                                    \
            return __VA_ARGS__();                                     \
        } else if (FDTYPE == cloudViewer::core::Float32 &&            \
                   IDTYPE == cloudViewer::core::Int64) {              \
            using scalar_t = float;                                   \
            using int_t = int64_t;                                    \
            return __VA_ARGS__();                                     \
        } else if (FDTYPE == cloudViewer::core::Float64 &&            \
                   IDTYPE == cloudViewer::core::Int32) {              \
            using scalar_t = double;                                  \
            using int_t = int32_t;                                    \
            return __VA_ARGS__();                                     \
        } else if (FDTYPE == cloudViewer::core::Float64 &&            \
                   IDTYPE == cloudViewer::core::Int64) {              \
            using scalar_t = double;                                  \
            using int_t = int64_t;                                    \
            return __VA_ARGS__();                                     \
        } else {                                                      \
            cloudViewer::utility::LogError("Unsupported data type."); \
        }                                                             \
    }()

#define DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(DTYPE, PREFIX, ...)     \
    [&] {                                                             \
        if (DTYPE == cloudViewer::core::Int8) {                       \
            using scalar_##PREFIX##_t = int8_t;                       \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int16) {               \
            using scalar_##PREFIX##_t = int16_t;                      \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int32) {               \
            using scalar_##PREFIX##_t = int32_t;                      \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::Int64) {               \
            using scalar_##PREFIX##_t = int64_t;                      \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt8) {               \
            using scalar_##PREFIX##_t = uint8_t;                      \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt16) {              \
            using scalar_##PREFIX##_t = uint16_t;                     \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt32) {              \
            using scalar_##PREFIX##_t = uint32_t;                     \
            return __VA_ARGS__();                                     \
        } else if (DTYPE == cloudViewer::core::UInt64) {              \
            using scalar_##PREFIX##_t = uint64_t;                     \
            return __VA_ARGS__();                                     \
        } else {                                                      \
            cloudViewer::utility::LogError("Unsupported data type."); \
        }                                                             \
    }()
