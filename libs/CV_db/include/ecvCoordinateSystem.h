// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "LineSet.h"
#include "ecvGenericPrimitive.h"

class ccPlane;

//! Coordinate System (primitive)
/** Coordinate System primitive
 **/
class CV_DB_LIB_API ccCoordinateSystem : public ccGenericPrimitive {
public:
    //! Default constructor
    /** Coordinate System is essentially just a way to visualize a transform
    matrix. \param transMat optional 3D transformation (can be set afterwards
    with ccDrawableObject::setGLTransformation) \param name name
    **/
    ccCoordinateSystem(PointCoordinateType displayScale,
                       PointCoordinateType axisWidth,
                       const ccGLMatrix* transMat = nullptr,
                       QString name = QString("CoordinateSystem"));

    //! Default constructor
    /** Coordinate System is essentially just a way to visualize a transform
    matrix. \param transMat optional 3D transformation (can be set afterwards
    with ccDrawableObject::setGLTransformation) \param name name
    **/
    ccCoordinateSystem(const ccGLMatrix* transMat,
                       QString name = QString("CoordinateSystem"));

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccCoordinateSystem(QString name = QString("CoordinateSystem"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::COORDINATESYSTEM;
    }

    // inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "CoordinateSystem"; }
    virtual ccGenericPrimitive* clone() const override;

    // Returns whether axis planes are Shown
    inline bool axisPlanesAreShown() const { return m_showAxisPlanes; }
    // Sets whether axis planes are Shown
    void ShowAxisPlanes(bool show);
    // Returns whether axis lines are Shown
    inline bool axisLinesAreShown() const { return m_showAxisLines; }
    // Sets whether axis lines are Shown
    void ShowAxisLines(bool show);

    // Returns axis width
    inline PointCoordinateType getAxisWidth() const { return m_width; }
    // Sets axis width
    void setAxisWidth(PointCoordinateType width);

    // Returns display scale
    inline PointCoordinateType getDisplayScale() const {
        return m_DisplayScale;
    }
    // Sets display scale
    void setDisplayScale(PointCoordinateType scale);

    // ccPlane get2AxisPlane(int axisNum);
    inline CCVector3 getOrigin() const {
        return m_transformation.getTranslationAsVec3D();
    }

    // Returns xy plane
    std::shared_ptr<ccPlane> getXYplane() const;
    // Returns yz plane
    std::shared_ptr<ccPlane> getYZplane() const;
    // Returns zx plane
    std::shared_ptr<ccPlane> getZXplane() const;

    virtual void clearDrawings() override;
    virtual void hideShowDrawings(CC_DRAW_CONTEXT& context) override;

    //! Default Display scale
    static constexpr PointCoordinateType DEFAULT_DISPLAY_SCALE = 1.0;
    //! Minimum Display scale
    static constexpr float MIN_DISPLAY_SCALE_F = 0.001f;
    //! Default Axis line width
    static constexpr PointCoordinateType AXIS_DEFAULT_WIDTH = 4.0;
    //! Minimum Axis line width
    static constexpr float MIN_AXIS_WIDTH_F = 1.0f;
    //! Maximum Axis line width
    static constexpr float MAX_AXIS_WIDTH_F = 16.0f;

protected:
    // inherited from ccDrawable
    void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    // inherited from ccGenericPrimitive
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    virtual bool buildUp() override;

    ccPlane createXYplane(const ccGLMatrix* transMat = nullptr) const;
    ccPlane createYZplane(const ccGLMatrix* transMat = nullptr) const;
    ccPlane createZXplane(const ccGLMatrix* transMat = nullptr) const;

    //! CoordinateSystem options
    PointCoordinateType m_DisplayScale;
    PointCoordinateType m_width;
    bool m_showAxisPlanes;
    bool m_showAxisLines;
    cloudViewer::geometry::LineSet m_axis;
};
