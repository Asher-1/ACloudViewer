// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>

#include "visualization/rendering/RendererHandle.h"
#include "visualization/rendering/RotationInteractorLogic.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

class CloudViewerScene;

class ModelInteractorLogic : public RotationInteractorLogic {
    using Super = RotationInteractorLogic;

public:
    ModelInteractorLogic(CloudViewerScene* scene,
                         Camera* camera,
                         double min_far_plane);
    virtual ~ModelInteractorLogic();

    void SetBoundingBox(const ccBBox& bounds) override;

    void SetModel(GeometryHandle axes,
                  const std::vector<GeometryHandle>& objects);

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;
    void Dolly(float dy, DragType drag_type) override;
    void Pan(int dx, int dy) override;

    void StartMouseDrag() override;
    void UpdateMouseDragUI() override;
    void EndMouseDrag() override;

private:
    CloudViewerScene* scene_;
    bool is_axes_visible_;

    ccBBox bounds_at_mouse_down_;
    std::map<std::string, Camera::Transform> transforms_at_mouse_down_;

    void UpdateBoundingBox(const Camera::Transform& t);
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
