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

#ifndef ECV_BBOX_HEADER
#define ECV_BBOX_HEADER

// LOCAL
#include "eCV_db.h"
#include "ecvColorTypes.h"
#include "ecvDrawableObject.h"
#include "ecvGLMatrix.h"
#include "ecvHObject.h"

// CV_CORE_LIB
#include <SquareMatrix.h>
#include <BoundingBox.h>

//! Bounding box structure
/** Supports several operators such as addition (to a matrix or a vector) and
        multiplication (by a matrix or a scalar).
**/
class ECV_DB_LIB_API ccBBox : public ccHObject,
                              public cloudViewer::BoundingBox {
public:
    //! Default constructor
    ccBBox() : ccHObject("ccBBox"), cloudViewer::BoundingBox() {}

    //! Constructor from two vectors (lower min. and upper max. corners)
    ccBBox(const CCVector3& bbMinCorner,
           const CCVector3& bbMaxCorner,
           const std::string& name = "ccBBox")
        : ccHObject(name.c_str()),
          cloudViewer::BoundingBox(bbMinCorner, bbMaxCorner) {}

    /// \brief Parameterized constructor.
    ///
    /// \param min_bound Lower bounds of the bounding box for all axes.
    /// \param max_bound Upper bounds of the bounding box for all axes.
    ccBBox(const Eigen::Vector3d& min_bound,
           const Eigen::Vector3d& max_bound,
           const std::string& name = "ccBBox")
        : ccHObject(name.c_str()),
          cloudViewer::BoundingBox(min_bound, max_bound) {}

    //! Constructor from two vectors (lower min. and upper max. corners)
    explicit ccBBox(const cloudViewer::BoundingBox& bbox,
                    const std::string& name = "ccBBox")
        : ccHObject(name.c_str()), cloudViewer::BoundingBox(bbox) {}

    //! Applies transformation to the bounding box
    const ccBBox operator*(const ccGLMatrix& mat);
    //! Applies transformation to the bounding box
    const ccBBox operator*(const ccGLMatrixd& mat);

    ~ccBBox() override = default;

    // inherited methods (ccHObject)
    bool isSerializable() const override { return true; }
    //! Returns unique class ID
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::BBOX; }
    // Returns the entity's own bounding-box
    virtual inline ccBBox getOwnBB(bool withGLFeatures = false) override {
        return *this;
    }

public:  // inherited methods (ccHObject)
    inline virtual bool isEmpty() const override { return volume() <= 0; }

    virtual inline Eigen::Vector3d getMinBound() const override {
        return CCVector3d::fromArray(m_bbMin);
    }
    virtual inline Eigen::Vector3d getMaxBound() const override {
        return CCVector3d::fromArray(m_bbMax);
    }
    virtual inline Eigen::Vector3d getGeometryCenter() const override {
        return CCVector3d::fromArray(getCenter());
    }

    virtual inline ccBBox getAxisAlignedBoundingBox() const override {
        return *this;
    }
    virtual ecvOrientedBBox getOrientedBoundingBox() const override;

    virtual ccBBox& transform(const Eigen::Matrix4d& transformation) override;
    virtual ccBBox& translate(const Eigen::Vector3d& translation,
                              bool relative = true) override;
    virtual ccBBox& scale(const double s,
                          const Eigen::Vector3d& center) override;
    virtual ccBBox& rotate(const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& center) override;

    const ccBBox& operator+=(const ccBBox& other);

    // CCVector3: must override to fix candidate template ignored:
    // could not match 'QStringBuilder' against 'Vector3Tpl
    const ccBBox& operator+=(const CCVector3 & V) override;
    const ccBBox& operator-=(const CCVector3& V) override;
    const ccBBox& operator*=(float scaleFactor) override;
    const ccBBox& operator*=(const cloudViewer::SquareMatrix& mat) override;

    // Eigen::Vector3d
    const ccBBox& operator+=(const Eigen::Vector3d& V);
    const ccBBox& operator-=(const Eigen::Vector3d& V);
    const ccBBox& operator*=(double scaleFactor);
    const ccBBox& operator*=(const Eigen::Matrix3d& mat);

public:
    //! Draws bounding box (OpenGL)
    /** \param context OpenGL context
     *  \param col (R,G,B) color
     **/
    void draw(CC_DRAW_CONTEXT& context, const ecvColor::Rgb& col) const;

    /// Returns the 3D dimensions of the bounding box in string format.
    std::string getPrintInfo() const;

    inline void setMinBounds(const Eigen::Vector3d& minBound) {
        m_bbMin = minBound;
    }
    inline void setMaxBounds(const Eigen::Vector3d& maxBound) {
        m_bbMax = maxBound;
    }

    /// Creates the bounding box that encloses the set of points.
    ///
    /// \param points A list of points.
    static ccBBox CreateFromPoints(const std::vector<CCVector3>& points);

    static ccBBox CreateFromPoints(const std::vector<Eigen::Vector3d>& points);

    /// Get the extent/length of the bounding box in x, y, and z dimension.
    inline Eigen::Vector3d getExtent() const {
        return CCVector3d::fromArray(getDiagVec());
    }

    /// Returns the half extent of the bounding box.
    Eigen::Vector3d getHalfExtent() const { return getExtent() * 0.5; }

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    inline PointCoordinateType getMaxExtent() const {
        return (m_bbMax - m_bbMin).maxCoeff();
    }

    /// Returns the eight points that define the bounding box.
    std::vector<Eigen::Vector3d> getBoxPoints() const;
};

#endif  // ECV_BBOX_HEADER
