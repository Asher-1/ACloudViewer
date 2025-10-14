// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AssimpGltf.h"

AssimpGltf::AssimpGltf()
    : IoAbstractLoader({"_glTF Filter",
                        FileIOFilter::DEFAULT_PRIORITY,  // priority
                        QStringList{"gltf", "glb"}, "gltf",
                        QStringList{"qMeshIO - glTF file (*.gltf *.glb)"},
                        QStringList(), Import}) {}
