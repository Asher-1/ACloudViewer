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

class PhongShader : public ShaderWrapper {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    ~PhongShader() override { Release(); }

protected:
    PhongShader(const std::string &name) : ShaderWrapper(name) { Compile(); }

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
                                std::vector<Eigen::Vector3f> &normals,
                                std::vector<Eigen::Vector3f> &colors) = 0;

protected:
    void SetLighting(const ViewControl &view, const RenderOption &option);

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_color_;
    GLuint vertex_color_buffer_;
    GLuint vertex_normal_;
    GLuint vertex_normal_buffer_;
    GLuint MVP_;
    GLuint V_;
    GLuint M_;
    GLuint light_position_world_;
    GLuint light_color_;
    GLuint light_diffuse_power_;
    GLuint light_specular_power_;
    GLuint light_specular_shininess_;
    GLuint light_ambient_;

    // At most support 4 lights
    gl_util::GLMatrix4f light_position_world_data_;
    gl_util::GLMatrix4f light_color_data_;
    gl_util::GLVector4f light_diffuse_power_data_;
    gl_util::GLVector4f light_specular_power_data_;
    gl_util::GLVector4f light_specular_shininess_data_;
    gl_util::GLVector4f light_ambient_data_;
};

class PhongShaderForPointCloud : public PhongShader {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    PhongShaderForPointCloud() : PhongShader("PhongShaderForPointCloud") {}

protected:
    bool PrepareRendering(const ccHObject &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const ccHObject &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector3f> &colors) final;
};

class PhongShaderForTriangleMesh : public PhongShader {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    PhongShaderForTriangleMesh() : PhongShader("PhongShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const ccHObject &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const ccHObject &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector3f> &colors) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
