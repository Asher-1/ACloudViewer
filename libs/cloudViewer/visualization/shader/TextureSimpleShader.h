// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <Eigen/Core>

#include "visualization/shader/ShaderWrapper.h"

class ccHObject;
namespace cloudViewer {
namespace visualization {

namespace glsl {

class TextureSimpleShader : public ShaderWrapper {
public:
    ~TextureSimpleShader() override { Release(); }

protected:
    TextureSimpleShader(const std::string &name) : ShaderWrapper(name) {
        Compile();
    }

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
                                std::vector<Eigen::Vector2f> &uvs) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_uv_;
    GLuint texture_;
    GLuint MVP_;

    int num_materials_;
    std::vector<int> array_offsets_;
    std::vector<GLsizei> draw_array_sizes_;

    std::vector<GLuint> vertex_position_buffers_;
    std::vector<GLuint> vertex_uv_buffers_;
    std::vector<GLuint> texture_buffers_;
};

class TextureSimpleShaderForTriangleMesh : public TextureSimpleShader {
public:
    TextureSimpleShaderForTriangleMesh()
        : TextureSimpleShader("TextureSimpleShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const ccHObject &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const ccHObject &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector2f> &uvs) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
