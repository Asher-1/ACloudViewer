// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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
#include <GenericMesh.h>
#include <Helper.h>

// LOCAL
#include "CV_db.h"
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
class CV_DB_LIB_API ecvMeshBase : public cloudViewer::GenericMesh,
                                  public ccHObject {
public:
    /// \brief Default Constructor.
    ecvMeshBase(const char *name = "ecvMeshBase") : ccHObject(name) {}
    ~ecvMeshBase() override {}

    // inherited methods (ccHObject)
    inline virtual bool isSerializable() const override { return true; }
    inline virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::MESH_BASE;
    }
    virtual ccBBox getOwnBB(bool withGLFeatures = false) override;
    virtual void getBoundingBox(CCVector3 &bbMin, CCVector3 &bbMax) override;

    // inherited methods (GenericMesh)
    inline virtual unsigned size() const override {
        return static_cast<unsigned>(vertices_.size());
    }
    // inherited methods (GenericIndexedMesh)
    virtual void placeIteratorAtBeginning() override {}
    virtual void forEach(genericTriangleAction action) override {}
    virtual cloudViewer::GenericTriangle *_getNextTriangle() override {
        return nullptr;
    }

public:
    virtual ecvMeshBase &clear();
    virtual bool IsEmpty() const override;
    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual ccBBox GetAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox GetOrientedBoundingBox() const override;
    virtual ecvMeshBase &Transform(
            const Eigen::Matrix4d &transformation) override;
    virtual ecvMeshBase &Translate(const Eigen::Vector3d &translation,
                                   bool relative = true) override;
    virtual ecvMeshBase &Scale(const double s,
                               const Eigen::Vector3d &center) override;
    virtual ecvMeshBase &Rotate(const Eigen::Matrix3d &R,
                                const Eigen::Vector3d &center) override;

    ecvMeshBase &operator+=(const ecvMeshBase &mesh);
    ecvMeshBase operator+(const ecvMeshBase &mesh) const;

    /// Returns `True` if the mesh contains vertices.
    bool HasVertices() const { return vertices_.size() > 0; }

    /// Returns `True` if the mesh contains vertex normals.
    bool HasVertexNormals() const {
        return vertices_.size() > 0 &&
               vertex_normals_.size() == vertices_.size();
    }

    /// Returns `True` if the mesh contains vertex colors.
    bool HasVertexColors() const {
        return vertices_.size() > 0 &&
               vertex_colors_.size() == vertices_.size();
    }

    /// Normalize vertex normals to length 1.
    ecvMeshBase &NormalizeNormals() {
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
    ecvMeshBase &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
        return *this;
    }

    /// Function that computes the convex hull of the triangle mesh using qhull
    std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>> ComputeConvexHull()
            const;

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
