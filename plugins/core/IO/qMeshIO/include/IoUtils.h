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
                                        const aiScene *inScene);

ccMesh *newCCMeshFromAIMesh(const aiMesh *inMesh);

ccGLMatrix convertMatrix(const aiMatrix4x4 &inAssimpMatrix);

QVariant convertMetaValueToVariant(aiMetadata *inData,
                                   unsigned int inValueIndex);
}  // namespace IoUtils
