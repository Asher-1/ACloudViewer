// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef VECTOR4F_H
#define VECTOR4F_H

#include "../qPCL.h"

namespace VtkUtils {

struct QPCL_ENGINE_LIB_API Vector4F {
    qreal x;
    qreal y;
    qreal z;
    qreal v;
};

}  // namespace VtkUtils
#endif  // VECTOR4F_H
