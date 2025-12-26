// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvPolyline.h"

//! Circle (as a polyline)
/** Extends the ccPolyline class
 **/
class ECV_DB_LIB_API ccCircle : public ccPolyline {
public:
    //! Default constructor
    /** \param radius circle radius
        \param resolution circle displayed resolution
        \param uniqueID unique ID (handle with care)
    **/
    explicit ccCircle(double radius = 0.0,
                      unsigned resolution = 48,
                      unsigned uniqueID = ccUniqueIDGenerator::InvalidUniqueID);

    //! Copy constructor
    /** \param circle circle to copy/clone
     **/
    ccCircle(const ccCircle& circle);

    //! Destructor
    ~ccCircle() override = default;

    //! Returns class ID
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::CIRCLE; }

    // inherited methods (ccHObject)
    void applyGLTransformation(const ccGLMatrix& trans) override;

    //! Clones this circle
    ccCircle* clone() const;

    //! Sets the radius of the circle
    /**  \param radius the desired radius
     **/
    void setRadius(double radius);

    //! Returns the radius of the circle
    inline double getRadius() const { return m_radius; }

    //! Sets the resolution of the displayed circle
    /**  \param resolution the displayed resolution (>= 4)
     **/
    void setResolution(unsigned resolution);

    //! Returns the resolution of the displayed circle
    inline unsigned getResolution() const { return m_resolution; }

protected:
    //! Updates the internal representation
    void updateInternalRepresentation();

    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out) const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    //! Radius of the circle
    double m_radius;

    //! Resolution of the displayed circle
    unsigned m_resolution;
};
