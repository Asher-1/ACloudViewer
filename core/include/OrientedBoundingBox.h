//##########################################################################
//#                                                                        #
//#                               CVLIB                                    #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef CV_ORIENTED_BOUNDING_BOX_HEADER
#define CV_ORIENTED_BOUNDING_BOX_HEADER

//Local
#include "Eigen/core"
#include "CVGeom.h"

namespace CVLib
{
	class BoundingBox;

	/// \class OrientedBoundingBox
	///
	/// \brief A bounding box oriented along an arbitrary frame of reference.
	///
	/// The oriented bounding box is defined by its center position, rotation
	/// maxtrix and extent.
	class CV_CORE_LIB_API OrientedBoundingBox
	{
	public:
		/// \brief Default constructor.
		///
		/// Creates an empty Oriented Bounding Box.
		OrientedBoundingBox()
			:center_(0, 0, 0),
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
			: center_(center),
			R_(R),
			extent_(extent),
			color_(0, 0, 0) {}
		virtual ~OrientedBoundingBox() {}
	public:
		OrientedBoundingBox& Clear();

		/// Get the extent/length of the bounding box 
		/// in x, y, and z dimension in its frame of reference.
		inline const Eigen::Vector3d& getExtent() const { return extent_; }

		/// Returns the half extent of the bounding box in its frame of reference.
		Eigen::Vector3d getHalfExtent() const { return getExtent() * 0.5; }

		/// Returns the max extent of the bounding box in its frame of reference.
		inline double getMaxExtent() const { return extent_.maxCoeff(); }

		/// Sets the bounding box color.
		inline void setColor(const Eigen::Vector3d& color) { color_ = color; }
		/// Gets the bounding box color.
		inline const Eigen::Vector3d& getColor() const { return color_; }

		inline const Eigen::Matrix3d& getRotation() const { return R_; }
		inline void setRotation(const Eigen::Matrix3d& rotation) { R_ = rotation; }

		inline const Eigen::Vector3d& getPosition() const { return center_; }

		/// Returns the volume of the bounding box.
		double volume() const;

		/// Returns the eight points that define the bounding box.
		std::vector<Eigen::Vector3d> getBoxPoints() const;

		/// Return indices to points that are within the bounding box.
		std::vector<size_t> getPointIndicesWithinBoundingBox(
			const std::vector<Eigen::Vector3d>& points) const;	

		std::vector<size_t> getPointIndicesWithinBoundingBox(
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

} //namespace

#endif // CV_ORIENTED_BOUNDING_BOX_HEADER
