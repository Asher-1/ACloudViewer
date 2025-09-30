// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/rendering/RotationInteractorLogic.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

class CameraInteractorLogic : public RotationInteractorLogic {
    using Super = RotationInteractorLogic;

public:
    CameraInteractorLogic(Camera* c, double min_far_plane);

    void SetBoundingBox(const ccBBox& bounds) override;

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;
    void Dolly(float dy, DragType type) override;
    void Dolly(float z_dist, Camera::Transform matrix_in) override;

    void Pan(int dx, int dy) override;

    /// Sets camera field of view
    void Zoom(int dy, DragType drag_type);

    void RotateLocal(float angle_rad, const Eigen::Vector3f& axis);
    void MoveLocal(const Eigen::Vector3f& v);

    void RotateFly(int dx, int dy);

    void StartMouseDrag() override;
    void ResetMouseDrag();
    void UpdateMouseDragUI() override;
    void EndMouseDrag() override;

private:
    double fov_at_mouse_down_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
