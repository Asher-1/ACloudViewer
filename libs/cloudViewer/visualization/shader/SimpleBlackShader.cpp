// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/shader/SimpleBlackShader.h"

#include <HalfEdgeTriangleMesh.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include "visualization/shader/Shader.h"
#include "visualization/utility/ColorMap.h"

namespace cloudViewer {
namespace visualization {

namespace glsl {

bool SimpleBlackShader::Compile() {
    if (!CompileShaders(SimpleBlackVertexShader, NULL,
                        SimpleBlackFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleBlackShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleBlackShader::BindGeometry(const ccHObject &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    std::vector<Eigen::Vector3f> points;
    if (!PrepareBinding(geometry, option, view, points)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 points.data(), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool SimpleBlackShader::RenderGeometry(const ccHObject &geometry,
                                       const RenderOption &option,
                                       const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    return true;
}

void SimpleBlackShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        bound_ = false;
    }
}

bool SimpleBlackShaderForPointCloudNormal::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::POINT_CLOUD)) {
        PrintShaderWarning("Rendering type is not ccPointCloud.");
        return false;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleBlackShaderForPointCloudNormal::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points) {
    if (!geometry.isKindOf(CV_TYPES::POINT_CLOUD)) {
        PrintShaderWarning("Rendering type is not ccPointCloud.");
        return false;
    }
    const ccPointCloud &pointcloud = (const ccPointCloud &)geometry;
    if (!pointcloud.hasPoints()) {
        PrintShaderWarning("Binding failed with empty pointcloud.");
        return false;
    }
    points.resize(pointcloud.size() * 2);
    double line_length =
            option.point_size_ * 0.01 * view.GetBoundingBox().GetMaxExtent();
    for (size_t i = 0; i < pointcloud.size(); i++) {
        const auto &point = pointcloud.getEigenPoint(i);
        const auto &normal = pointcloud.getEigenNormal(i);
        points[i * 2] = point.cast<float>();
        points[i * 2 + 1] = (point + normal * line_length).cast<float>();
    }
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareRendering(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (!geometry.isKindOf(CV_TYPES::MESH) &&
        !geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        PrintShaderWarning("Rendering type is not ccMesh.");
        return false;
    }
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_POLYGON_OFFSET_FILL);
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareBinding(
        const ccHObject &geometry,
        const RenderOption &option,
        const ViewControl &view,
        std::vector<Eigen::Vector3f> &points) {
    if (!geometry.isKindOf(CV_TYPES::MESH) &&
        !geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        PrintShaderWarning("Rendering type is not ccMesh.");
        return false;
    }

    if (geometry.isKindOf(CV_TYPES::MESH)) {
        const ccMesh &mesh = (const ccMesh &)geometry;
        if (!mesh.hasTriangles()) {
            PrintShaderWarning("Binding failed with empty ccMesh.");
            return false;
        }
        points.resize(static_cast<std::size_t>(mesh.size()) * 3);
        for (unsigned int i = 0; i < mesh.size(); i++) {
            const cloudViewer::VerticesIndexes *triangle =
                    mesh.getTriangleVertIndexes(i);
            for (unsigned int j = 0; j < 3; j++) {
                unsigned int idx = i * 3 + j;
                unsigned int vi = triangle->i[j];
                const auto &vertex = mesh.getVertice(vi);
                points[idx] = vertex.cast<float>();
            }
        }
        draw_arrays_mode_ = GL_TRIANGLES;
        draw_arrays_size_ = GLsizei(points.size());
    } else if (geometry.isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        const geometry::HalfEdgeTriangleMesh &mesh =
                (const geometry::HalfEdgeTriangleMesh &)geometry;
        if (!mesh.hasTriangles()) {
            PrintShaderWarning(
                    "Binding failed with empty geometry::TriangleMesh.");
            return false;
        }
        points.resize(mesh.triangles_.size() * 3);
        for (size_t i = 0; i < mesh.triangles_.size(); i++) {
            const auto &triangle = mesh.triangles_[i];
            for (size_t j = 0; j < 3; j++) {
                size_t idx = i * 3 + j;
                size_t vi = triangle(j);
                const auto &vertex = mesh.vertices_[vi];
                points[idx] = vertex.cast<float>();
            }
        }
        draw_arrays_mode_ = GL_TRIANGLES;
        draw_arrays_size_ = GLsizei(points.size());
    }

    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
