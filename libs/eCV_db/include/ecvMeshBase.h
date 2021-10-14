// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CV_CORE_LIB
#include <Eigen.h>
#include <Helper.h>
#include <GenericMesh.h>

// LOCAL
#include "eCV_db.h"
#include "ecvHObject.h"

class ccMesh;
class ccBBox;
class ecvOrientedBBox;
class ccPointCloud;

namespace cloudViewer {
namespace geometry {

/// \class ecvMeshBase
///
/// \brief ecvMeshBash Class.
///
/// Triangle mesh contains vertices. Optionally, the mesh may also contain
/// vertex normals and vertex colors.
class ECV_DB_LIB_API ecvMeshBase : public cloudViewer::GenericMesh, public ccHObject {

public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Default Constructor.
    ecvMeshBase(const char *name = "ecvMeshBase") : ccHObject(name) {}
    ~ecvMeshBase() override {}

    // inherited methods (ccHObject)
    inline virtual bool isSerializable() const override { return true; }
    inline virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::MESH_BASE; }
    virtual ccBBox getOwnBB(bool withGLFeatures = false) override;
    virtual void getBoundingBox(CCVector3 &bbMin, CCVector3 &bbMax) override;

    // inherited methods (GenericMesh)
    inline virtual unsigned size() const override {
        return static_cast<unsigned>(vertices_.size()); }
    // inherited methods (GenericIndexedMesh)
    virtual void placeIteratorAtBeginning() override {}
    virtual void forEach(genericTriangleAction action) override {}
    virtual cloudViewer::GenericTriangle *_getNextTriangle() override {
        return nullptr;
    }

public:
    virtual ecvMeshBase &clear();
    virtual bool isEmpty() const override;
    virtual Eigen::Vector3d getMinBound() const override;
    virtual Eigen::Vector3d getMaxBound() const override;
    virtual Eigen::Vector3d getGeometryCenter() const override;
    virtual ccBBox getAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox getOrientedBoundingBox() const override;
    virtual ecvMeshBase &transform(const Eigen::Matrix4d &transformation) override;
    virtual ecvMeshBase &translate(const Eigen::Vector3d &translation,
                                bool relative = true) override;
    virtual ecvMeshBase &scale(const double s,
                            const Eigen::Vector3d &center) override;
    virtual ecvMeshBase &rotate(const Eigen::Matrix3d &R,
                             const Eigen::Vector3d &center) override;

    ecvMeshBase &operator+=(const ecvMeshBase &mesh);
    ecvMeshBase operator+(const ecvMeshBase &mesh) const;

    /// Returns `True` if the mesh contains vertices.
    bool hasVertices() const { return vertices_.size() > 0; }

    /// Returns `True` if the mesh contains vertex normals.
    bool hasVertexNormals() const {
        return vertices_.size() > 0 &&
               vertex_normals_.size() == vertices_.size();
    }

    /// Returns `True` if the mesh contains vertex colors.
    bool hasVertexColors() const {
        return vertices_.size() > 0 &&
               vertex_colors_.size() == vertices_.size();
    }

    /// Normalize vertex normals to length 1.
    ecvMeshBase &normalizeNormals() {
        for (size_t i = 0; i < vertex_normals_.size(); i++) {
            vertex_normals_[i].normalize();
            if (std::isnan(vertex_normals_[i](0))) {
                vertex_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
        return *this;
    }

    /// \brief Assigns each vertex in the TriangleMesh the same color
    ///
    /// \param color RGB colors of vertices.
    ecvMeshBase &paintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
        return *this;
    }

    /// Function that computes the convex hull of the triangle mesh using qhull
    std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>> computeConvexHull() const;

protected:
    // Forward child class type to avoid indirect nonvirtual base
    ecvMeshBase(const std::vector<Eigen::Vector3d> &vertices,
                const char *name = "ecvMeshBase")
        : ccHObject(name), vertices_(vertices) {}

public:
    /// Vertex coordinates.
    std::vector<Eigen::Vector3d> vertices_;
    /// Vertex normals.
    std::vector<Eigen::Vector3d> vertex_normals_;
    /// RGB colors of vertices.
    std::vector<Eigen::Vector3d> vertex_colors_;
};

}  // namespace geometry
}  // namespace cloudViewer
