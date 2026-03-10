// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file point3f.h
/// @brief 3D point struct (x, y, z).

#include "qVTK.h"

namespace VtkUtils {

/// @struct Point3F
/// @brief 3-component point (x, y, z).
struct QVTK_ENGINE_LIB_API Point3F {
    qreal x;
    qreal y;
    qreal z;
};

}  // namespace VtkUtils
