// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// MeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <QMap>

#include "assimp/material.h"
#include "assimp/matrix4x4.h"
#include "assimp/mesh.h"
#include "assimp/metadata.h"
#include "assimp/scene.h"
#include "ecvHObjectCaster.h"
#include "ecvMaterialSet.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

namespace IoUtils {
ccMaterialSet *createMaterialSetForMesh(const aiMesh *inMesh,
                                        const QString &inPath,
                                        const aiScene *inScene,
                                        const QString &inSourceFileName);

ccMesh *newCCMeshFromAIMesh(const aiMesh *inMesh);

// Converts a glTF POINT primitive (assimp: aiMesh without triangle faces) to
// a plain point cloud. Routing such meshes through ccMesh creates one
// degenerate one-vertex face per point and later triple-expands them in the
// VTK converter; a point cloud keeps them on the cheap point path.
ccPointCloud *newCCPointCloudFromAIMesh(const aiMesh *inMesh);

ccGLMatrix convertMatrix(const aiMatrix4x4 &inAssimpMatrix);

QVariant convertMetaValueToVariant(aiMetadata *inData,
                                   unsigned int inValueIndex);
}  // namespace IoUtils
