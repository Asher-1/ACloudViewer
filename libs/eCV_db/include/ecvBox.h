// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGenericPrimitive.h"

//! Box (primitive)
/** 3D box primitive
 **/
class ECV_DB_LIB_API ccBox : public ccGenericPrimitive {
public:
    //! Default constructor
    /** Box dimensions axis along each dimension are defined in a single 3D
    vector. A box is in fact composed of 6 planes (ccPlane). \param dims box
    dimensions \param transMat optional 3D transformation (can be set afterwards
    with ccDrawableObject::setGLTransformation) \param name name
    **/
    ccBox(const CCVector3& dims,
          const ccGLMatrix* transMat = nullptr,
          QString name = QString("Box"));

    //! Simplified constructor
    /** For ccHObject factory only!
     **/
    ccBox(QString name = QString("Box"));

    //! Returns class ID
    virtual CV_CLASS_ENUM getClassID() const override { return CV_TYPES::BOX; }

    // inherited from ccGenericPrimitive
    virtual QString getTypeName() const override { return "Box"; }
    virtual ccGenericPrimitive* clone() const override;

    //! Sets box dimensions
    inline void setDimensions(CCVector3& dims) {
        m_dims = dims;
        updateRepresentation();
    }

    //! Returns box dimensions
    const CCVector3& getDimensions() const { return m_dims; }

protected:
    // inherited from ccGenericPrimitive
    virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in,
                                 short dataVersion,
                                 int flags,
                                 LoadedIDMap& oldToNewIDMap) override;
    virtual bool buildUp() override;

    //! Box dimensions
    CCVector3 m_dims;
};
