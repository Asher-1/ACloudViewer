// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/renderer/DepthRenderer.hpp>

#include "core/graphics/RenderUtility.hpp"

namespace sibr {

DepthRenderer::~DepthRenderer() {};

DepthRenderer::DepthRenderer(int w, int h) {
    _depthShader.init(
            "DepthShader",
            sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName(
                    "depthRenderer.vp")),
            sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName(
                    "depthRenderer.fp")));

    _depthShader_MVP.init(_depthShader, "MVP");
    _depth_RT.reset(new sibr::RenderTargetLum32F(w, h));
}

void DepthRenderer::render(const sibr::InputCamera& cam,
                           const Mesh& mesh,
                           bool backFaceCulling,
                           bool frontFaceCulling) {
    // sibr::Vector1f cc(1.0);
    //_depth_RT->clear(cc);

    glViewport(0, 0, _depth_RT->w(), _depth_RT->h());
    _depth_RT->bind();
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _depthShader.begin();
    _depthShader_MVP.set(cam.viewproj());

    mesh.render(true, backFaceCulling, sibr::Mesh::FillRenderMode,
                frontFaceCulling);

    _depthShader.end();
}

}  // namespace sibr