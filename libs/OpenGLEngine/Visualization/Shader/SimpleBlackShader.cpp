// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include "Visualization/Shader/SimpleBlackShader.h"

#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include "Visualization/Shader/Shader.h"
#include "Visualization/Utility/ColorMap.h"

namespace cloudViewer {
namespace visualization {

namespace glsl {

bool SimpleBlackShader::Compile() {
    if (CompileShaders(SimpleBlackVertexShader, NULL,
                       SimpleBlackFragmentShader) == false) {
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
    if (PrepareBinding(geometry, option, view, points) == false) {
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
    if (PrepareRendering(geometry, option, view) == false) {
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
            option.point_size_ * 0.01 * view.GetBoundingBox().getMaxExtent();
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
    if (!geometry.isKindOf(CV_TYPES::MESH)) {
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
    if (!geometry.isKindOf(CV_TYPES::MESH)) {
        PrintShaderWarning("Rendering type is not ccMesh.");
        return false;
    }
    const ccMesh &mesh =
            (const ccMesh &)geometry;
    if (mesh.hasTriangles() == false) {
        PrintShaderWarning("Binding failed with empty ccMesh.");
        return false;
    }
    points.resize(mesh.size() * 3);
    for (unsigned int i = 0; i < mesh.size(); i++) {
        const CVLib::VerticesIndexes* triangle = 
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
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
