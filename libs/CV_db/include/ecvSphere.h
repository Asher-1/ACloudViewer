// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGenericPrimitive.h"

//! Sphere (primitive)
/** 3D sphere primitive
 **/
class CV_DB_LIB_API ccSphere : public ccGenericPrimitive {
public:
    //! Default constructor
    /** \param radius sphere radius
            \param transMat optional 3D transformation (can be set afterwards
    with ccDrawableObject::setGLTransformation) \param name name \param
    precision drawing precision (angular step = 360/precision)
    **/
    ccSphere(PointCoordinateType radius,
             const ccGLMatrix* transMat = 0,
             QString name = QString("Sphere"),
             unsigned precision = 24);

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccSphere(QString name = QString("Sphere"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::SPHERE;
    }

    // inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "Sphere"; }
    virtual bool hasDrawingPrecision() const override { return true; }
    virtual ccGenericPrimitive* clone() const override;

    //! Returns radius
    inline PointCoordinateType getRadius() const { return m_radius; }
    //! Sets radius
    /** \warning changes primitive content (calls
     *ccGenericPrimitive::updateRepresentation)
     **/
    void setRadius(PointCoordinateType radius);

protected:
    // inherited from ccGenericPrimitive
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    virtual bool buildUp() override;

    // inherited from ccHObject
    virtual void drawNameIn3D() override;

    //! Radius
    PointCoordinateType m_radius;
};
