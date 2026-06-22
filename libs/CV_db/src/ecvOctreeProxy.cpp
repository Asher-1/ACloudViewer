// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOctreeProxy.h"

#include "ecvBBox.h"
#include "ecvGenericGLDisplay.h"

// Local
// #include "ccCameraSensor.h"
// #include "ccNormalVectors.h"
// #include "ccBox.h"

// cloudViewer
// #include <ScalarFieldTools.h>
// #include <RayAndBox.h>

ccOctreeProxy::ccOctreeProxy(ccOctree::Shared octree /*=ccOctree::Shared(0)*/,
                             QString name /*="Octree"*/)
    : ccHObject(name), m_octree(octree) {
    setVisible(false);
    lockVisibility(false);
}

ccOctreeProxy::~ccOctreeProxy() {}

ccBBox ccOctreeProxy::getOwnBB(bool withGLFeatures /*=false*/) {
    if (!m_octree) {
        assert(false);
        return ccBBox();
    }

    return withGLFeatures ? m_octree->getSquareBB() : m_octree->getPointsBB();
}

void ccOctreeProxy::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!m_octree) {
        assert(false);
        return;
    }

    if (!MACRO_Draw3D(context)) return;

    if (!context.display) return;

    bool entityPickingMode = MACRO_EntityPicking(context);

    if (entityPickingMode) {
        // not fast at all!
        if (MACRO_FastEntityPicking(context)) return;
    }

    // Sync octree visibility: both enabled AND visible must be true.
    // forceRedraw means "re-render already visible things", NOT
    // "make invisible things visible".
    bool shouldBeVisible = isEnabled() && isVisible();
    setOctreeVisibale(shouldBeVisible);
    // Always call draw() even when hiding — the octree draw code needs to run
    // its cleanup path (removing VTK actors for previously visible cells).
    m_octree->draw(context);
}
