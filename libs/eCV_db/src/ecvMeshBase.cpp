// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMeshBase.h"

// LOCAL
#include <Logging.h>

#include "ecvBBox.h"
#include "ecvMesh.h"
#include "ecvQhull.h"
#include "ecvPointCloud.h"
#include "ecvOrientedBBox.h"
#include "ecvHObjectCaster.h"

// SYSTEM
#include <Eigen/Dense>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>
#include <unordered_map>
#include <array>

namespace cloudViewer {
namespace geometry {

ccBBox ecvMeshBase::getOwnBB(bool withGLFeatures) {
    return getAxisAlignedBoundingBox();
}

void ecvMeshBase::getBoundingBox(CCVector3 &bbMin, CCVector3 &bbMax) {
    bbMin = getMinBound();
    bbMax = getMaxBound();
}

ecvMeshBase &ecvMeshBase::clear() {
    vertices_.clear();
    vertex_normals_.clear();
    vertex_colors_.clear();
    return *this;
}

bool ecvMeshBase::isEmpty() const { return !hasVertices(); }


Eigen::Vector3d ecvMeshBase::getMinBound() const {
    return ComputeMinBound(vertices_);
}

Eigen::Vector3d ecvMeshBase::getMaxBound() const {
    return ComputeMaxBound(vertices_);
}

Eigen::Vector3d ecvMeshBase::getGeometryCenter() const {
    return ComputeCenter(vertices_);
}

ccBBox ecvMeshBase::getAxisAlignedBoundingBox() const {
    return ccBBox::CreateFromPoints(vertices_);
}

ecvOrientedBBox ecvMeshBase::getOrientedBoundingBox() const {
    return ecvOrientedBBox::CreateFromPoints(vertices_);
}

ecvMeshBase &ecvMeshBase::transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, vertices_);
    TransformNormals(transformation, vertex_normals_);
    return *this;
}

ecvMeshBase &ecvMeshBase::translate(const Eigen::Vector3d &translation,
                                bool relative) {
    TranslatePoints(translation, vertices_, relative);
    return *this;
}

ecvMeshBase &ecvMeshBase::scale(const double s, const Eigen::Vector3d &center) {
    ScalePoints(s, vertices_, center);
    return *this;
}

ecvMeshBase &ecvMeshBase::rotate(const Eigen::Matrix3d &R,
                             const Eigen::Vector3d &center) {
    RotatePoints(R, vertices_, center);
    RotateNormals(R, vertex_normals_);
    return *this;
}

ecvMeshBase &ecvMeshBase::operator+=(const ecvMeshBase &mesh) {
    if (mesh.isEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    size_t add_vert_num = mesh.vertices_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!hasVertices() || hasVertexNormals()) && mesh.hasVertexNormals()) {
        vertex_normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            vertex_normals_[old_vert_num + i] = mesh.vertex_normals_[i];
    } else {
        vertex_normals_.clear();
    }
    if ((!hasVertices() || hasVertexColors()) && mesh.hasVertexColors()) {
        vertex_colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            vertex_colors_[old_vert_num + i] = mesh.vertex_colors_[i];
    } else {
        vertex_colors_.clear();
    }
    vertices_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        vertices_[old_vert_num + i] = mesh.vertices_[i];
    return (*this);
}

ecvMeshBase ecvMeshBase::operator+(const ecvMeshBase &mesh) const {
    return (ecvMeshBase(*this) += mesh);
}

std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
ecvMeshBase::computeConvexHull() const {
    return Qhull::ComputeConvexHull(vertices_);
}

}  // namespace geometry
}  // namespace cloudViewer
