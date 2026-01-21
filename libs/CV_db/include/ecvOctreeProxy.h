// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvHObject.h"
#include "ecvOctree.h"

//! Octree structure proxy
/** Implements ccHObject while holding a (shared) pointer on the octree instance
 *(--> safer)
 **/
class CV_DB_LIB_API ccOctreeProxy : public ccHObject {
public:
    //! Default constructor
    ccOctreeProxy(ccOctree::Shared octree = ccOctree::Shared(0),
                  QString name = "Octree");

    //! Destructor
    virtual ~ccOctreeProxy();

    //! Sets the associated octree
    inline void setOctree(ccOctree::Shared octree) { m_octree = octree; }

    //! Returns the associated octree
    inline ccOctree::Shared getOctree() const { return m_octree; }

    inline void setOctreeVisibale(bool state) { m_octree->setVisible(state); }

    // Inherited from ccHObject
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::POINT_OCTREE;
    }
    virtual ccBBox getOwnBB(bool withGLFeatures = false) override;

protected:
    // Inherited from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

protected:  // members
    //! Associated octree
    ccOctree::Shared m_octree;
};
