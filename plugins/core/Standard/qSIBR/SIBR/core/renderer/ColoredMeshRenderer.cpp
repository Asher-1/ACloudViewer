// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <core/renderer/ColoredMeshRenderer.hpp>

#include "core/graphics/Texture.hpp"

namespace sibr {
ColoredMeshRenderer::ColoredMeshRenderer(void) {
    _shader.init("ColoredMesh",
                 sibr::loadFile(sibr::getShadersDirectory("core") +
                                "/colored_mesh.vert"),
                 sibr::loadFile(sibr::getShadersDirectory("core") +
                                "/colored_mesh.frag"));
    _paramMVP.init(_shader, "MVP");
}

void ColoredMeshRenderer::process(const Mesh& mesh,
                                  const Camera& eye,
                                  IRenderTarget& target,
                                  sibr::Mesh::RenderMode mode,
                                  bool backFaceCulling) {
    // glViewport(0.f, 0.f, target.w(), target.h());
    target.bind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    _shader.begin();
    _paramMVP.set(eye.viewproj());
    mesh.render(true, backFaceCulling);
    _shader.end();
    target.unbind();
}

} /*namespace sibr*/
