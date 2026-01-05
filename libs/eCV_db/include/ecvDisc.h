// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGenericPrimitive.h"

//! Disc (primitive)
/** 3D disc primitive
 **/
class ECV_DB_LIB_API ccDisc : public ccGenericPrimitive {
public:
    //! Default drawing precision
    /** \warning Never pass a 'constant initializer' by reference
     **/
    static const unsigned DEFAULT_DRAWING_PRECISION = 72;

    //! Default constructor
    /** Simple disc constructor
        \param radius disc radius
        \param transMat optional 3D transformation (can be set afterwards with
    ccDrawableObject::setGLTransformation) \param name name \param precision
    drawing precision (angular step = 360/precision)
    **/
    ccDisc(PointCoordinateType radius,
           const ccGLMatrix* transMat = nullptr,
           QString name = QString("Disc"),
           unsigned precision = DEFAULT_DRAWING_PRECISION);

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccDisc(QString name = QString("Disc"));

    //! Returns radius
    inline PointCoordinateType getRadius() const { return m_radius; }
    //! Sets radius
    /** \warning changes primitive content (calls
     *ccGenericPrimitive::updateRepresentation)
     **/
    void setRadius(PointCoordinateType radius);

    //! Returns class ID
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::DISC; }

    // inherited from ccGenericPrimitive
    QString getTypeName() const override { return "Disc"; }
    bool hasDrawingPrecision() const override { return true; }
    ccGenericPrimitive* clone() const override;

    // inherited from ccHObject
    ccBBox getOwnFitBB(ccGLMatrix& trans) override;

protected:
    // inherited from ccGenericPrimitive
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    bool buildUp() override;

    //! Radius
    PointCoordinateType m_radius;
};
