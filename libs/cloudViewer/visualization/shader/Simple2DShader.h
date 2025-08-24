// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <vector>

#include "visualization/shader/ShaderWrapper.h"

class ccHObject;
namespace cloudViewer {
namespace visualization {

namespace glsl {

class Simple2DShader : public ShaderWrapper {
public:
    ~Simple2DShader() override { Release(); }

protected:
    Simple2DShader(const std::string &name) : ShaderWrapper(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;
    bool BindGeometry(const ccHObject &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool RenderGeometry(const ccHObject &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
    void UnbindGeometry() final;

protected:
    virtual bool PrepareRendering(const ccHObject &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const ccHObject &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                std::vector<Eigen::Vector3f> &points,
                                std::vector<Eigen::Vector3f> &colors) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_color_;
    GLuint vertex_color_buffer_;
};

class Simple2DShaderForSelectionPolygon : public Simple2DShader {
public:
    Simple2DShaderForSelectionPolygon()
        : Simple2DShader("Simple2DShaderForSelectionPolygon") {}

protected:
    bool PrepareRendering(const ccHObject &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const ccHObject &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &colors) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
