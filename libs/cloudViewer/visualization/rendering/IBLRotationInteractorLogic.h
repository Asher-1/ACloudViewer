// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/rendering/MatrixInteractorLogic.h"
#include "visualization/rendering/RendererHandle.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

class Scene;

class IBLRotationInteractorLogic : public MatrixInteractorLogic {
    using Super = MatrixInteractorLogic;

public:
    IBLRotationInteractorLogic(Scene* scene, Camera* camera);

    void Rotate(int dx, int dy) override;
    void RotateZ(int dx, int dy) override;

    void StartMouseDrag();
    void UpdateMouseDragUI();
    void EndMouseDrag();

    Camera::Transform GetCurrentRotation() const;

private:
    Scene* scene_;
    Camera* camera_;
    bool skybox_currently_visible_ = false;
    Camera::Transform ibl_rotation_at_mouse_down_;

    void ClearUI();
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
