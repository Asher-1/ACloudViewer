// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

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
