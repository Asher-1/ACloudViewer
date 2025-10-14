// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Avoid warning caused by redefinition of APIENTRY macro
// defined also in glfw3.h
#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>  // Make sure glew.h is included before gl.h
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <string>
#include <unordered_map>

namespace cloudViewer {
namespace visualization {
namespace gl_util {

// static std::unordered_map<int, unsigned int> GetTextureFormatMap();
// static std::unordered_map<int, unsigned int> GetTextureTypeMap();

const std::unordered_map<int, unsigned int> &GetTextureFormatMap();
const std::unordered_map<int, unsigned int> &GetTextureTypeMap();

typedef Eigen::Matrix<float, 3, 1, Eigen::ColMajor> GLVector3f;
typedef Eigen::Matrix<float, 4, 1, Eigen::ColMajor> GLVector4f;
typedef Eigen::Matrix<float, 4, 4, Eigen::ColMajor> GLMatrix4f;

GLMatrix4f LookAt(const Eigen::Vector3d &eye,
                  const Eigen::Vector3d &lookat,
                  const Eigen::Vector3d &up);

GLMatrix4f Perspective(double field_of_view_,
                       double aspect,
                       double z_near,
                       double z_far);

GLMatrix4f Ortho(double left,
                 double right,
                 double bottom,
                 double top,
                 double z_near,
                 double z_far);

Eigen::Vector3d Project(const Eigen::Vector3d &point,
                        const GLMatrix4f &mvp_matrix,
                        const int width,
                        const int height);

Eigen::Vector3d Unproject(const Eigen::Vector3d &screen_point,
                          const GLMatrix4f &mvp_matrix,
                          const int width,
                          const int height);

int ColorCodeToPickIndex(const Eigen::Vector4i &color);

}  // namespace gl_util
}  // namespace visualization
}  // namespace cloudViewer
