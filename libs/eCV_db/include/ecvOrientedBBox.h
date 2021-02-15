//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_ORIENTED_BBOX_HEADER
#define ECV_ORIENTED_BBOX_HEADER

// LOCAL
#include "eCV_db.h"
#include "ecvHObject.h"
#include "ecvColorTypes.h"
#include "ecvDrawableObject.h"

// CV_CORE_LIB
#include <OrientedBoundingBox.h>

#ifdef USE_EIGEN
#endif // USE_EIGEN
#include <Eigen/Core>

//! Bounding box structure
/** Supports several operators such as addition (to a matrix or a vector) and
	multiplication (by a matrix or a scalar).
**/
class ccGLMatrix;
class ccGLMatrixd;
class ECV_DB_LIB_API ecvOrientedBBox : public cloudViewer::OrientedBoundingBox, public ccHObject
{
public:

	//! Default constructor
    ecvOrientedBBox()
        : cloudViewer::OrientedBoundingBox()
        , ccHObject("ecvOrientedBBox")
	{}

		/// \brief Parameterized constructor.
	///
	/// \param center Specifies the center position of the bounding box.
	/// \param R The rotation matrix specifying the orientation of the
	/// bounding box with the original frame of reference.
	/// \param extent The extent of the bounding box.
	ecvOrientedBBox(const Eigen::Vector3d& center,
		const Eigen::Matrix3d& R,
		const Eigen::Vector3d& extent,
		const std::string& name = "ecvOrientedBBox")
        : cloudViewer::OrientedBoundingBox(center, R, extent)
        , ccHObject(name.c_str())
    {}

	~ecvOrientedBBox() override {}

	//inherited methods (ccHObject)
	virtual bool isSerializable() const override { return true; }
	//! Returns unique class ID
	virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::ORIENTED_BBOX; }
	// Returns the entity's own bounding-box
	virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

	//! Draws oriented bounding box (OpenGL)
	/** \param context OpenGL context
	 *  \param col (R,G,B) color
	**/
    void draw (CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col);

	inline virtual bool isEmpty() const override { return volume() <= 0; }
	virtual Eigen::Vector3d getMinBound() const override;
	virtual Eigen::Vector3d getMaxBound() const override;
	virtual Eigen::Vector3d getGeometryCenter() const override;

	virtual ccBBox getAxisAlignedBoundingBox() const override;
	virtual ecvOrientedBBox getOrientedBoundingBox() const override;

	virtual ecvOrientedBBox& transform(
		const Eigen::Matrix4d& transformation) override;
	virtual ecvOrientedBBox& translate(
		const Eigen::Vector3d& translation,
		bool relative = true) override;
	virtual ecvOrientedBBox& scale(
		const double s, const Eigen::Vector3d& center) override;
	virtual ecvOrientedBBox& rotate(
		const Eigen::Matrix3d& R,
		const Eigen::Vector3d& center) override;

	//! Applies transformation to the bounding box
	const ecvOrientedBBox operator * (const ccGLMatrix& mat);
	//! Applies transformation to the bounding box
	const ecvOrientedBBox operator * (const ccGLMatrixd& mat);

	/// Creates an oriented bounding box using a PCA.
	/// Note, that this is only an approximation to the minimum oriented
	/// bounding box that could be computed for example with O'Rourke's
	/// algorithm (cf. http://cs.smith.edu/~jorourke/Papers/MinVolBox.pdf,
	/// https://www.geometrictools.com/Documentation/MinimumVolumeBox.pdf)
	static ecvOrientedBBox CreateFromPoints(const std::vector<Eigen::Vector3d>& points);
	static ecvOrientedBBox CreateFromPoints(const std::vector<CCVector3>& points);

	static ecvOrientedBBox CreateFromAxisAlignedBoundingBox(const ccBBox& aabox);

};

#endif // ECV_ORIENTED_BBOX_HEADER
