// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGenericPrimitive.h"

//! Torus (primitive)
/** 3D torus primitive (with circular or rectangular section)
 **/
class CV_DB_LIB_API ccTorus : public ccGenericPrimitive {
public:
    //! Default drawing precision
    /** \warning Never pass a 'constant initializer' by reference
     **/
    static const unsigned DEFAULT_DRAWING_PRECISION = 24;

    //! Default constructor
    /** Torus is defined in the XY plane by default
            \param insideRadius inside radius
            \param outsideRadius outside radius
            \param angle_rad subtended angle (in radians)
            \param rectangularSection whether section is rectangular or round
            \param rectSectionHeight section height (if rectangular torus)
            \param transMat optional 3D transformation (can be set afterwards
    with ccDrawableObject::setGLTransformation) \param name name \param
    precision drawing precision (main loop angular step = 360/(4*precision),
    circular section angular step = 360/precision)
    **/
    ccTorus(PointCoordinateType insideRadius,
            PointCoordinateType outsideRadius,
            double angle_rad = 2.0 * M_PI,
            bool rectangularSection = false,
            PointCoordinateType rectSectionHeight = 0,
            const ccGLMatrix* transMat = 0,
            QString name = QString("Torus"),
            unsigned precision = DEFAULT_DRAWING_PRECISION);

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccTorus(QString name = QString("Torus"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::TORUS;
    }

    // inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "Torus"; }
    virtual bool hasDrawingPrecision() const override { return true; }
    virtual ccGenericPrimitive* clone() const override;

    //! Returns the torus inside radius
    inline PointCoordinateType getInsideRadius() const {
        return m_insideRadius;
    }
    //! Returns the torus outside radius
    inline PointCoordinateType getOutsideRadius() const {
        return m_outsideRadius;
    }
    //! Returns the torus rectangular section height (along Y-axis) if
    //! applicable
    inline PointCoordinateType getRectSectionHeight() const {
        return m_rectSectionHeight;
    }
    //! Returns whether torus has a rectangular (true) or circular (false)
    //! section
    inline bool getRectSection() const { return m_rectSection; }
    //! Returns the torus subtended angle (in radians)
    inline double getAngleRad() const { return m_angle_rad; }

protected:
    // inherited from ccGenericPrimitive
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    virtual bool buildUp() override;

    //! Inside radius
    PointCoordinateType m_insideRadius;

    //! Outside radius
    PointCoordinateType m_outsideRadius;

    //! Whether torus has a rectangular (true) or circular (false) section
    bool m_rectSection;

    //! Rectangular section height (along Y-axis) if applicable
    PointCoordinateType m_rectSectionHeight;

    //! Subtended angle (in radians)
    double m_angle_rad;
};
