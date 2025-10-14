// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_TETRA_MESH_HEADER
#define ECV_TETRA_MESH_HEADER

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <memory>
#include <vector>

// LOCAL
#include "ecvMeshBase.h"

// CV_CORE_LIB
#include <Eigen.h>
#include <Helper.h>

namespace cloudViewer {
namespace geometry {

/// \class TetraMesh
///
/// \brief Tetra mesh contains vertices and tetrahedra represented by the
/// indices to the vertices.
class ECV_DB_LIB_API TetraMesh : public ecvMeshBase {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    //! Default ccMesh constructor
    /** \param vertices the vertices cloud
     **/
    TetraMesh(const char *name = "TetraMesh") : ecvMeshBase(name) {}

    /// \brief Parameterized Constructor.
    ///
    /// \param vertices Vertex coordinates.
    /// \param tetras List of tetras denoted by the index of points forming the
    /// tetra.
    TetraMesh(
            const std::vector<Eigen::Vector3d> &vertices,
            const std::vector<Eigen::Vector4i,
                              cloudViewer::utility::Vector4i_allocator> &tetras,
            const char *name = "TetraMesh")
        : ecvMeshBase(name), tetras_(tetras) {}

    ~TetraMesh() override {}

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::TETRA_MESH;
    }

public:
    virtual TetraMesh &clear() override;

public:
    inline std::size_t TetraSize() const { return tetras_.size(); }

    TetraMesh &operator+=(const TetraMesh &mesh);
    TetraMesh operator+(const TetraMesh &mesh) const;

    /// \brief Function that removes duplicated verties, i.e., vertices that
    /// have identical coordinates.
    TetraMesh &RemoveDuplicatedVertices();

    /// \brief Function that removes duplicated tetrahedra, i.e., removes
    /// tetrahedra that reference the same four vertices, independent of their
    /// order.
    TetraMesh &RemoveDuplicatedTetras();

    /// \brief This function removes vertices from the tetra mesh that are not
    /// referenced in any tetrahedron of the mesh.
    TetraMesh &RemoveUnreferencedVertices();

    /// \brief Function that removes degenerate tetrahedra, i.e., tetrahedra
    /// that reference a single vertex multiple times in a single tetrahedron.
    /// They are usually the product of removing duplicated vertices.
    TetraMesh &RemoveDegenerateTetras();

    /// Returns `true` if the mesh contains tetras.
    bool HasTetras() const {
        return vertices_.size() > 0 && tetras_.size() > 0;
    }

    /// \brief Function to extract a triangle mesh of the specified iso-surface
    /// at a level This method applies primal contouring and generates triangles
    /// for each tetrahedron.
    ///
    /// \param level specifies the level.
    /// \param values specifies values per-vertex.
    std::shared_ptr<ccMesh> ExtractTriangleMesh(
            const std::vector<double> &values, double level);

    /// \brief Function that creates a tetrahedral mesh (TetraMeshFactory.cpp).
    /// from a point cloud.
    ///
    /// The method creates the Delaunay triangulation
    /// using the implementation from Qhull.
    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    CreateFromPointCloud(const ccPointCloud &point_cloud);

public:
    /// List of tetras denoted by the index of points forming the tetra.
    std::vector<Eigen::Vector4i, cloudViewer::utility::Vector4i_allocator>
            tetras_;
};

}  // namespace geometry
}  // namespace cloudViewer

#endif  // ECV_TETRA_MESH_HEADER
