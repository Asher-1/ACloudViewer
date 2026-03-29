// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/graphics/GUI.hpp>
#include <projects/basic/renderer/PointBasedView.hpp>

sibr::PointBasedView::PointBasedView(const sibr::BasicIBRScene::Ptr& ibrScene,
                                     uint render_w,
                                     uint render_h)
    : _scene(ibrScene), sibr::ViewBase(render_w, render_h) {
    const uint w = render_w;
    const uint h = render_h;

    //  Renderers.
    _pointBasedRenderer.reset(new PointBasedRenderer());
}

void sibr::PointBasedView::setScene(const sibr::BasicIBRScene::Ptr& newScene) {
    _scene = newScene;
    const uint w = getResolution().x();
    const uint h = getResolution().y();

    _pointBasedRenderer.reset(new PointBasedRenderer());
}

void sibr::PointBasedView::onRenderIBR(sibr::IRenderTarget& dst,
                                       const sibr::Camera& eye) {
    // Perform ULR rendering, either directly to the destination RT, or to the
    // intermediate RT when poisson blending is enabled.
    glViewport(0, 0, dst.w(), dst.h());
    dst.clear();
    _pointBasedRenderer->process(_scene->proxies()->proxy(), eye, dst, false);
}

void sibr::PointBasedView::onUpdate(Input& input) {}

void sibr::PointBasedView::onGUI() {
    if (ImGui::Begin("Point Based Mesh Renderer Settings")) {
        // Poisson settings.
        // ImGui::Checkbox("Poisson fix", &_poissonRenderer->enableFix());
    }
    ImGui::End();
}
