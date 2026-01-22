// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvCone.h"

//! Cylinder (primitive)
/** 3D cylinder primitive
 **/
class CV_DB_LIB_API ccCylinder : public ccCone {
public:
    //! Default drawing precision
    /** \warning Never pass a 'constant initializer' by reference
     **/
    static const unsigned DEFAULT_DRAWING_PRECISION = 24;

    //! Default constructor
    /** Cylinder axis corresponds to the 'Z' dimension.
            Internally represented by a cone with the same top and bottom
    radius. \param radius cylinder radius \param height cylinder height
    (transformation should point to the axis center) \param transMat optional 3D
    transformation (can be set afterwards with
    ccDrawableObject::setGLTransformation) \param name name \param precision
    drawing precision (angular step = 360/precision)
    **/
    ccCylinder(PointCoordinateType radius,
               PointCoordinateType height,
               const ccGLMatrix* transMat = nullptr,
               QString name = QString("Cylinder"),
               unsigned precision = DEFAULT_DRAWING_PRECISION);

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccCylinder(QString name = QString("Cylinder"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::CYLINDER;
    }

    // inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "Cylinder"; }
    virtual ccGenericPrimitive* clone() const override;

    // inherited from ccCone
    virtual void setBottomRadius(PointCoordinateType radius) override;
    inline virtual void setTopRadius(PointCoordinateType radius) override {
        return setBottomRadius(radius);
    }
};
