// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AssimpBlender.h"

AssimpBlender::AssimpBlender()
    : IoAbstractLoader({"_Blender Filter",
                        FileIOFilter::DEFAULT_PRIORITY,  // priority
                        QStringList{"blend"}, "blend",
                        QStringList{"qMeshIO - Blend file (*.blend)"},
                        QStringList(), Import}) {}
