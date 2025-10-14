// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_ORIENTED_BOUNDING_BOX_HEADER
#define CV_ORIENTED_BOUNDING_BOX_HEADER

// Local
#include "BoundingBox.h"
#include "CVGeom.h"
#include "Eigen.h"

namespace cloudViewer {
/// \class OrientedBoundingBox
///
/// \brief A bounding box oriented along an arbitrary frame of reference.
///
/// The oriented bounding box is defined by its center position, rotation
/// maxtrix and extent.
class CV_CORE_LIB_API OrientedBoundingBox {
public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    /// \brief Default constructor.
    ///
    /// Creates an empty Oriented Bounding Box.
    OrientedBoundingBox()
        : center_(0, 0, 0),
          R_(Eigen::Matrix3d::Identity()),
          extent_(0, 0, 0),
          color_(0, 0, 0) {}
    /// \brief Parameterized constructor.
    ///
    /// \param center Specifies the center position of the bounding box.
    /// \param R The rotation matrix specifying the orientation of the
    /// bounding box with the original frame of reference.
    /// \param extent The extent of the bounding box.
    OrientedBoundingBox(const Eigen::Vector3d& center,
                        const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& extent)
        : center_(center), R_(R), extent_(extent), color_(0, 0, 0) {}
    virtual ~OrientedBoundingBox() {}

public:
    OrientedBoundingBox& Clear();

    /// Get the extent/length of the bounding box
    /// in x, y, and z dimension in its frame of reference.
    inline const Eigen::Vector3d& GetExtent() const { return extent_; }

    //! Returns center
    inline CCVector3 getCenter() const { return center_; }

    /// Returns the half extent of the bounding box in its frame of reference.
    Eigen::Vector3d GetHalfExtent() const { return GetExtent() * 0.5; }

    /// Returns the max extent of the bounding box in its frame of reference.
    inline double GetMaxExtent() const { return extent_.maxCoeff(); }

    /// Sets the bounding box color.
    inline void SetColor(const Eigen::Vector3d& color) { color_ = color; }
    /// Gets the bounding box color.
    inline const Eigen::Vector3d& GetColor() const { return color_; }

    inline const Eigen::Matrix3d& GetRotation() const { return R_; }

    inline const Eigen::Vector3d& GetPosition() const { return center_; }

    /// Returns the volume of the bounding box.
    double volume() const;

    /// Returns the eight points that define the bounding box.
    /// \verbatim
    ///      ------- x
    ///     /|
    ///    / |
    ///   /  | z
    ///  y
    ///      0 ------------------- 1
    ///       /|                /|
    ///      / |               / |
    ///     /  |              /  |
    ///    /   |             /   |
    /// 2 ------------------- 7  |
    ///   |    |____________|____| 6
    ///   |   /3            |   /
    ///   |  /              |  /
    ///   | /               | /
    ///   |/                |/
    /// 5 ------------------- 4
    /// \endverbatim
    std::vector<Eigen::Vector3d> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<Eigen::Vector3d>& points) const;

    std::vector<size_t> GetPointIndicesWithinBoundingBox(
            const std::vector<CCVector3>& points) const;

    /// Returns an oriented bounding box from the AxisAlignedBoundingBox.
    ///
    /// \param aabox AxisAlignedBoundingBox object from which
    /// OrientedBoundingBox is created.
    static OrientedBoundingBox CreateFromAxisAlignedBoundingBox(
            const BoundingBox& aabox);

public:
    /// The center point of the bounding box.
    Eigen::Vector3d center_;
    /// The rotation matrix of the bounding box to transform the original frame
    /// of reference to the frame of this box.
    Eigen::Matrix3d R_;
    /// The extent of the bounding box in its frame of reference.
    Eigen::Vector3d extent_;
    /// The color of the bounding box in RGB.
    Eigen::Vector3d color_;
};

}  // namespace cloudViewer

#endif  // CV_ORIENTED_BOUNDING_BOX_HEADER
