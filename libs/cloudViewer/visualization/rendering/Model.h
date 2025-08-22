// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvMesh.h>
#include "visualization/rendering/MaterialRecord.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

struct TriangleMeshModel {
    struct MeshInfo {
        std::shared_ptr<ccMesh> mesh;
        std::string mesh_name;
        unsigned int material_idx;
    };

    std::vector<MeshInfo> meshes_;
    std::vector<visualization::rendering::MaterialRecord> materials_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
