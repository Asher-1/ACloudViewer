// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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
#include <vector>

#include "ecvHObject.h"

class ccMesh;
class ccBBox;
class ecvOrientedBBox;
class ccPointCloud;
namespace cloudViewer {
namespace geometry {

class TetraMesh;

/// \class LineSet
///
/// \brief LineSet define a sets of lines in 3D. A typical application is to
/// display the point cloud correspondence pairs.
class ECV_DB_LIB_API LineSet : public ccHObject {
public:

	/// \brief Default Constructor.
	LineSet(const char* name = "LineSet") : ccHObject(name) {}
	/// \brief Parameterized Constructor.
	///
	///  Create a LineSet from given points and line indices
	///
	/// \param points Point coordinates.
	/// \param lines Lines denoted by the index of points forming the line.
	LineSet(const std::vector<Eigen::Vector3d> &points,
		const std::vector<Eigen::Vector2i> &lines, 
		const char* name = "LineSet")
		: ccHObject(name),
		points_(points),
		lines_(lines) {}
	~LineSet() override {}


	//inherited methods (ccHObject)
	virtual bool isSerializable() const override { return true; }

	//! Returns unique class ID
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::LINESET; }

	virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

public:
    LineSet &clear();
	inline virtual bool isEmpty() const override { return !hasPoints(); }
    virtual Eigen::Vector3d getMinBound() const override;
    virtual Eigen::Vector3d getMaxBound() const override;
    virtual Eigen::Vector3d getGeometryCenter() const override;
	virtual ccBBox getAxisAlignedBoundingBox() const override;
	virtual ecvOrientedBBox getOrientedBoundingBox() const override;
    virtual LineSet& transform(const Eigen::Matrix4d &transformation) override;
    virtual LineSet& translate(const Eigen::Vector3d &translation,
		bool relative = true) override;
	virtual LineSet& scale(const double s, const Eigen::Vector3d &center) override;
	virtual LineSet& rotate(const Eigen::Matrix3d &R, const Eigen::Vector3d &center) override;

    LineSet &operator+=(const LineSet &lineset);
    LineSet operator+(const LineSet &lineset) const;

    /// Returns `true` if the object contains points.
    bool hasPoints() const { return points_.size() > 0; }

    /// Returns `true` if the object contains lines.
    bool hasLines() const { return hasPoints() && lines_.size() > 0; }

    /// Returns `true` if the objects lines contains colors.
    bool hasColors() const {
        return hasLines() && colors_.size() == lines_.size();
    }

    /// \brief Returns the coordinates of the line at the given index.
    ///
    /// \param line_index Index of the line.
    std::pair<Eigen::Vector3d, Eigen::Vector3d> getLineCoordinate(
            size_t line_index) const {
        return std::make_pair(points_[lines_[line_index][0]],
                              points_[lines_[line_index][1]]);
    }

    /// \brief Assigns each line in the LineSet the same color.
    ///
    /// \param color Specifies the color to be applied.
    LineSet &paintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(colors_, lines_.size(), color);
        return *this;
    }

    /// \brief Factory function to create a LineSet from two PointClouds
    /// (\p cloud0, \p cloud1) and a correspondence set.
    ///
    /// \param cloud0 First point cloud.
    /// \param cloud1 Second point cloud.
    /// \param correspondences Set of correspondences.
    static std::shared_ptr<LineSet> CreateFromPointCloudCorrespondences(
            const ccPointCloud &cloud0,
            const ccPointCloud &cloud1,
            const std::vector<std::pair<int, int>> &correspondences);

    /// \brief Factory function to create a LineSet from an OrientedBoundingBox.
    ///
    /// \param box The input bounding box.
    static std::shared_ptr<LineSet> CreateFromOrientedBoundingBox(
            const ecvOrientedBBox &box);

    /// \brief Factory function to create a LineSet from an
    /// ccBBox.
    ///
    /// \param box The input bounding box.
    static std::shared_ptr<LineSet> CreateFromAxisAlignedBoundingBox(
            const ccBBox &box);

    /// Factory function to create a LineSet from edges of a triangle mesh.
    ///
    /// \param mesh The input triangle mesh.
    static std::shared_ptr<LineSet> CreateFromTriangleMesh(
            const ccMesh &mesh);

    /// Factory function to create a LineSet from edges of a tetra mesh.
    ///
    /// \param mesh The input tetra mesh.
    static std::shared_ptr<LineSet> CreateFromTetraMesh(const TetraMesh &mesh);

public:
    /// Points coordinates.
    std::vector<Eigen::Vector3d> points_;
    /// Lines denoted by the index of points forming the line.
    std::vector<Eigen::Vector2i> lines_;
    /// RGB colors of lines.
    std::vector<Eigen::Vector3d> colors_;
};

}  // namespace geometry
}  // namespace cloudViewer
