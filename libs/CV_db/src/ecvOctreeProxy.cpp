// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOctreeProxy.h"

#include "ecvBBox.h"
#include "ecvDisplayTools.h"

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

    if (ecvDisplayTools::GetMainWindow() == nullptr) return;

    bool entityPickingMode = MACRO_EntityPicking(context);

    if (entityPickingMode) {
        // not fast at all!
        if (MACRO_FastEntityPicking(context)) return;
    }

    setOctreeVisibale(isEnabled());
    m_octree->draw(context);
}
