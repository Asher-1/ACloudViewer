// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_DISH_PRIMITIVE_HEADER
#define ECV_DISH_PRIMITIVE_HEADER

// Local
#include "ecvGenericPrimitive.h"

//! Dish
/** Either a section of a sphere, or half of an ellipsoid!
 **/
class ECV_DB_LIB_API ccDish : public ccGenericPrimitive {
public:
    //! Default drawing precision
    /** \warning Never pass a 'constant initializer' by reference
     **/
    static const unsigned DEFAULT_DRAWING_PRECISION = 24;

    //! Default constructor
    /** The origin of the Dish is at the centre of the base, and the Z-axis lies
    along the normal to the base. \param radius base radius \param height
    maximum height of dished surface above base \param radius2 If radius2 is
    zero, dish is drawn as a section of sphere. If radius2 is >0, dish is
    defined as half of an ellipsoid. \param transMat optional 3D transformation
    (can be set afterwards with ccDrawableObject::setGLTransformation) \param
    name name \param precision drawing precision (angular step = 360/precision)
    **/
    ccDish(PointCoordinateType radius,
           PointCoordinateType height,
           PointCoordinateType radius2 = 0,
           const ccGLMatrix* transMat = 0,
           QString name = QString("Dish"),
           unsigned precision = DEFAULT_DRAWING_PRECISION);

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccDish(QString name = QString("Dish"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::DISH; }

    //! inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "Dish"; }
    virtual bool hasDrawingPrecision() const override { return true; }
    virtual ccGenericPrimitive* clone() const override;

    //! Returns the dish base radius
    inline PointCoordinateType getBaseRadius() const { return m_baseRadius; }
    //! Returns the dish second radius
    inline PointCoordinateType getSecondRadius() const {
        return m_secondRadius;
    }
    //! Returns the dish height
    inline PointCoordinateType getHeight() const { return m_height; }

protected:
    // inherited from ccGenericPrimitive
    virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in,
                                 short dataVersion,
                                 int flags,
                                 LoadedIDMap& oldToNewIDMap) override;
    virtual bool buildUp() override;

    //! Base radius
    PointCoordinateType m_baseRadius;
    //! Second radius
    PointCoordinateType m_secondRadius;
    //! Height
    PointCoordinateType m_height;
};

#endif  // ECV_DISH_PRIMITIVE_HEADER
