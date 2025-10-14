// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_CORE_LIB
#include <CVGeom.h>

// LOCAL
#include "ecvColorTypes.h"
#include "ecvDrawContext.h"

//! Interface for a planar entity
class ccPlanarEntityInterface {
public:
    //! Default constructor
    ccPlanarEntityInterface();
    ccPlanarEntityInterface(unsigned int id);

    //! Show normal vector
    inline void showNormalVector(bool state) { m_showNormalVector = state; }
    //! Whether normal vector is shown or not
    inline bool normalVectorIsShown() const { return m_showNormalVector; }

    //! Returns the entity normal
    virtual CCVector3 getNormal() const = 0;

    //! Draws a normal vector (OpenGL)
    void glDrawNormal(CC_DRAW_CONTEXT& context,
                      const CCVector3& pos,
                      float scale,
                      const ecvColor::Rgb* color = 0);

    void clearNormalVector(CC_DRAW_CONTEXT& context);

protected:  // members
    //! Whether the facet normal vector should be displayed or not
    bool m_showNormalVector;
    unsigned int m_uniqueId;

    QString m_bodyId;
    QString m_headId;
};
