// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.cloudViewer.org
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

#ifndef ECV_TETRA_MESH_HEADER
#define ECV_TETRA_MESH_HEADER

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <memory>
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

/// \class TetraMesh
///
/// \brief Tetra mesh contains vertices and tetrahedra represented by the
/// indices to the vertices.
class ECV_DB_LIB_API TetraMesh : public CVLib::GenericMesh, public ccHObject
{
public:
	//! Default ccMesh constructor
	/** \param vertices the vertices cloud
	**/
	TetraMesh(const char* name = "TetraMesh") : ccHObject(name) {}

    /// \brief Parameterized Constructor.
    ///
    /// \param vertices Vertex coordinates.
    /// \param tetras List of tetras denoted by the index of points forming the
    /// tetra.
	TetraMesh(const std::vector<Eigen::Vector3d> &vertices,
		const std::vector<Eigen::Vector4i,
		CVLib::utility::Vector4i_allocator> &tetras, 
		const char* name = "TetraMesh");

	~TetraMesh() override {}

	//inherited methods (ccHObject)
	virtual bool isSerializable() const override { return true; }
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::TETRA_MESH; }
	virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

	//inherited methods (GenericMesh)
	inline virtual unsigned size() const override {
		return static_cast<unsigned>(tetras_.size());
	}
	virtual void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override;

	// inherited methods (GenericIndexedMesh)
	virtual void placeIteratorAtBeginning() override {}
	virtual void forEach(genericTriangleAction action) override {}
	virtual CVLib::GenericTriangle* _getNextTriangle() override { return nullptr; }


public:
	inline virtual bool isEmpty() const override { return !hasVertices(); }

	virtual Eigen::Vector3d getMinBound() const override;
	virtual Eigen::Vector3d getMaxBound() const override;
	virtual Eigen::Vector3d getGeometryCenter() const override;
	virtual ccBBox getAxisAlignedBoundingBox() const override;
	virtual ecvOrientedBBox getOrientedBoundingBox() const override;
	virtual TetraMesh& transform(const Eigen::Matrix4d &transformation) override;
	virtual TetraMesh& translate(const Eigen::Vector3d &translation,
		bool relative = true) override;
	virtual TetraMesh& scale(const double s, const Eigen::Vector3d& center) override;
	virtual TetraMesh& rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d& center) override;

    TetraMesh &clear();
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
	TetraMesh &normalizeNormals() {
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
	TetraMesh &paintUniformColor(const Eigen::Vector3d &color) {
		ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
		return *this;
	}

	/// Function that computes the convex hull of the triangle mesh using qhull
	std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>>
		computeConvexHull() const;
public:
    TetraMesh &operator+=(const TetraMesh &mesh);
    TetraMesh operator+(const TetraMesh &mesh) const;

    /// \brief Function that removes duplicated verties, i.e., vertices that
    /// have identical coordinates.
    TetraMesh &removeDuplicatedVertices();

    /// \brief Function that removes duplicated tetrahedra, i.e., removes
    /// tetrahedra that reference the same four vertices, independent of their
    /// order.
    TetraMesh &removeDuplicatedTetras();

    /// \brief This function removes vertices from the tetra mesh that are not
    /// referenced in any tetrahedron of the mesh.
    TetraMesh &removeUnreferencedVertices();

    /// \brief Function that removes degenerate tetrahedra, i.e., tetrahedra
    /// that reference a single vertex multiple times in a single tetrahedron.
    /// They are usually the product of removing duplicated vertices.
    TetraMesh &removeDegenerateTetras();

    /// Returns `true` if the mesh contains tetras.
	bool hasTetras() const
	{
		return vertices_.size() > 0 && tetras_.size() > 0;
	}

    /// \brief Function to extract a triangle mesh of the specified iso-surface
    /// at a level This method applies primal contouring and generates triangles
    /// for each tetrahedron.
    ///
    /// \param level specifies the level.
    /// \param values specifies values per-vertex.
    std::shared_ptr<ccMesh> extractTriangleMesh(
            const std::vector<double> &values, double level);

    /// \brief Function that creates a tetrahedral mesh (TetraMeshFactory.cpp).
    /// from a point cloud.
    ///
    /// The method creates the Delaunay triangulation
    /// using the implementation from Qhull.
    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    CreateFromPointCloud(const ccPointCloud &point_cloud);

public:
	/// Vertex coordinates.
	std::vector<Eigen::Vector3d> vertices_;
	/// Vertex normals.
	std::vector<Eigen::Vector3d> vertex_normals_;
	/// RGB colors of vertices.
	std::vector<Eigen::Vector3d> vertex_colors_;

	/// List of tetras denoted by the index of points forming the tetra.
	std::vector<Eigen::Vector4i, CVLib::utility::Vector4i_allocator> tetras_;
};

}  // namespace geometry
}  // namespace cloudViewer

#endif // ECV_TETRA_MESH_HEADER