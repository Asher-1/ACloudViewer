// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AssimpGltf.h"

#include "assimp/postprocess.h"

AssimpGltf::AssimpGltf()
    : IoAbstractLoader({"_glTF Filter",
                        FileIOFilter::DEFAULT_PRIORITY,  // priority
                        QStringList{"gltf", "glb"}, "gltf",
                        QStringList{"qMeshIO - glTF file (*.gltf *.glb)"},
                        QStringList(), Import}) {}

unsigned int AssimpGltf::_assimpPostProcessFlags() const {
    // Skip aiProcess_FindInvalidData: many glTF exporters (e.g. neural mesh
    // pipelines) ship zero-length placeholder normals. IoUtils drops invalid
    // normals and recomputes them from geometry when needed.
    return aiProcess_JoinIdenticalVertices | aiProcess_RemoveComponent |
           aiProcess_Triangulate | aiProcess_ValidateDataStructure;
}
