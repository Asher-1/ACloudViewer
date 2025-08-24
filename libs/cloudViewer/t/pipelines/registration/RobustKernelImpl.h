// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cmath>

#include "cloudViewer/core/CUDAUtils.h"
#include "cloudViewer/t/geometry/kernel/GeometryMacros.h"
#include "cloudViewer/t/pipelines/registration/RobustKernel.h"

#ifndef __CUDACC__
using std::abs;
using std::exp;
using std::max;
using std::min;
using std::pow;
#endif

using cloudViewer::t::pipelines::registration::RobustKernelMethod;

/// To use `Robust Kernel` functions please refer the unit-tests for
/// `t::registration` or the implementation use cases at
/// `t::pipelines::kernel::ComputePosePointToPlaneCUDA` and
/// `t::pipelines::kernel::ComputePosePointToPlaneCPU`.
///
/// \param METHOD registration::RobustKernelMethod Loss type.
/// \param scalar_t type: float / double.
/// \param scaling_parameter Scaling parameter for loss fine-tuning.
/// \param shape_parameter Shape parameter for Generalized Loss method.
#define DISPATCH_ROBUST_KERNEL_FUNCTION(METHOD, scalar_t, scaling_parameter, \
                                        shape_parameter, ...)                \
    [&] {                                                                    \
        scalar_t scale = static_cast<scalar_t>(scaling_parameter);           \
        if (METHOD == RobustKernelMethod::L2Loss) {                          \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0;                                                  \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::L1Loss) {                   \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0 / abs(residual);                                  \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::HuberLoss) {                \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return scale / max(abs(residual), scale);                    \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::CauchyLoss) {               \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0 / (1.0 + Square(residual / scale));               \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::GMLoss) {                   \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return scale / Square(scale + Square(residual));             \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::TukeyLoss) {                \
            auto GetWeightFromRobustKernel =                                 \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return Square(1.0 - Square(min((scalar_t)1.0,                \
                                               abs(residual) / scale)));     \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::GeneralizedLoss) {          \
            if (cloudViewer::IsClose(shape_parameter, 2.0, 1e-3)) {               \
                auto const_val = 1.0 / Square(scale);                        \
                auto GetWeightFromRobustKernel =                             \
                        [=] CLOUDVIEWER_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return const_val;                                        \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else if (cloudViewer::IsClose(shape_parameter, 0.0, 1e-3)) {        \
                auto GetWeightFromRobustKernel =                             \
                        [=] CLOUDVIEWER_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return 2.0 / (Square(residual) + 2 * Square(scale));     \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else if (shape_parameter < -1e7) {                             \
                auto GetWeightFromRobustKernel =                             \
                        [=] CLOUDVIEWER_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return exp(Square(residual / scale) / (-2.0)) /          \
                           Square(scale);                                    \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else {                                                         \
                auto GetWeightFromRobustKernel =                             \
                        [=] CLOUDVIEWER_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return pow((Square(residual / scale) /                   \
                                        abs(shape_parameter - 2.0) +         \
                                1),                                          \
                               ((shape_parameter / 2.0) - 1.0)) /            \
                           Square(scale);                                    \
                };                                                           \
                return __VA_ARGS__();                                        \
            }                                                                \
        } else {                                                             \
            utility::LogError("Unsupported method.");                        \
        }                                                                    \
    }()

/// \param scalar_t type: float / double.
/// \param METHOD_1 registration::RobustKernelMethod Loss type.
/// \param scaling_parameter_1 Scaling parameter for loss fine-tuning.
/// \param METHOD_2 registration::RobustKernelMethod Loss type.
/// \param scaling_parameter_2 Scaling parameter for loss fine-tuning.
#define DISPATCH_DUAL_ROBUST_KERNEL_FUNCTION(scalar_t, METHOD_1,            \
                                             scaling_parameter_1, METHOD_2, \
                                             scaling_parameter_2, ...)      \
    [&] {                                                                   \
        scalar_t scale_1 = static_cast<scalar_t>(scaling_parameter_1);      \
        scalar_t scale_2 = static_cast<scalar_t>(scaling_parameter_2);      \
        if (METHOD_1 == RobustKernelMethod::L2Loss &&                       \
            METHOD_2 == RobustKernelMethod::L2Loss) {                       \
            auto GetWeightFromRobustKernelFirst =                           \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return 1.0;                                                 \
            };                                                              \
            auto GetWeightFromRobustKernelSecond =                          \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return 1.0;                                                 \
            };                                                              \
            return __VA_ARGS__();                                           \
        } else if (METHOD_1 == RobustKernelMethod::L2Loss &&                \
                   METHOD_2 == RobustKernelMethod::TukeyLoss) {             \
            auto GetWeightFromRobustKernelFirst =                           \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return 1.0;                                                 \
            };                                                              \
            auto GetWeightFromRobustKernelSecond =                          \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return Square(1.0 - Square(min((scalar_t)1.0,               \
                                               abs(residual) / scale_2)));  \
            };                                                              \
            return __VA_ARGS__();                                           \
        } else if (METHOD_1 == RobustKernelMethod::TukeyLoss &&             \
                   METHOD_2 == RobustKernelMethod::L2Loss) {                \
            auto GetWeightFromRobustKernelFirst =                           \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return Square(1.0 - Square(min((scalar_t)1.0,               \
                                               abs(residual) / scale_1)));  \
            };                                                              \
            auto GetWeightFromRobustKernelSecond =                          \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return 1.0;                                                 \
            };                                                              \
            return __VA_ARGS__();                                           \
        } else if (METHOD_1 == RobustKernelMethod::TukeyLoss &&             \
                   METHOD_2 == RobustKernelMethod::TukeyLoss) {             \
            auto GetWeightFromRobustKernelFirst =                           \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return Square(1.0 - Square(min((scalar_t)1.0,               \
                                               abs(residual) / scale_1)));  \
            };                                                              \
            auto GetWeightFromRobustKernelSecond =                          \
                    [=] CLOUDVIEWER_HOST_DEVICE(scalar_t residual) -> scalar_t { \
                return Square(1.0 - Square(min((scalar_t)1.0,               \
                                               abs(residual) / scale_2)));  \
            };                                                              \
            return __VA_ARGS__();                                           \
        } else {                                                            \
            utility::LogError("Unsupported method.");                       \
        }                                                                   \
    }()
