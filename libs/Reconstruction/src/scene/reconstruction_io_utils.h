// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <vector>

#include "base/reconstruction.h"
#include "util/hash_containers.h"

namespace colmap {

// Helper method to extract sorted camera, image, point3D identifiers.
// We sort the identifiers before writing to the stream, such that we produce
// deterministic output independent of standard library dependent ordering of
// the unordered map container.
template <typename ID_TYPE, typename DATA_TYPE>
std::vector<ID_TYPE> ExtractSortedIds(
        const NodeHashMap<ID_TYPE, DATA_TYPE>& data,
        const std::function<bool(const DATA_TYPE&)>& filter = nullptr) {
    std::vector<ID_TYPE> ids;
    ids.reserve(data.size());
    for (const auto& [id, d] : data) {
        if (filter == nullptr || filter(d)) {
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

void CreateOneRigPerCamera(Reconstruction& reconstruction);

void CreateFrameForImage(const Image& image,
                         const Rigid3d& cam_from_world,
                         Reconstruction& reconstruction);

NodeHashMap<image_t, Frame*> ExtractImageToFramePtr(
        Reconstruction& reconstruction);

}  // namespace colmap
