// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/rendering/CameraInteractorLogic.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

class CameraSphereInteractorLogic : public CameraInteractorLogic {
    using Super = CameraInteractorLogic;

public:
    CameraSphereInteractorLogic(Camera* c, double min_far_plane);

    void Rotate(int dx, int dy) override;

    void StartMouseDrag() override;

private:
    float r_at_mousedown_;
    float theta_at_mousedown_;
    float phi_at_mousedown_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
