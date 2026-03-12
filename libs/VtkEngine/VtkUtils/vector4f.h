// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file vector4f.h
/// @brief 4D vector struct (x, y, z, v).

#include "qVTK.h"

namespace VtkUtils {

/// @struct Vector4F
/// @brief 4-component vector (x, y, z, v).
struct QVTK_ENGINE_LIB_API Vector4F {
    qreal x;
    qreal y;
    qreal z;
    qreal v;
};

}  // namespace VtkUtils
