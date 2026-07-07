// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccLineationTool.h"

#include <ecvViewManager.h>

#include "ccCompass.h"

ccLineationTool::ccLineationTool() : ccTool() {}

ccLineationTool::~ccLineationTool() {}

// called when the tool is set to disactive (for cleanup)
void ccLineationTool::toolDisactivated() { cancel(); }

// called when a point in a point cloud gets picked while this tool is active
void ccLineationTool::pointPicked(ccHObject* insertPoint,
                                  unsigned itemIdx,
                                  ccPointCloud* cloud,
                                  const CCVector3& P) {
    // try retrieve active lineation (will fail if there isn't one)
    ccLineation* l = dynamic_cast<ccLineation*>(
            m_app->dbRootObject()->find(m_lineation_id));
    if (!l)  // make a new one
    {
        // no active trace -> make a new one
        l = new ccLineation(cloud);
        m_lineation_id = l->getUniqueID();

        if (ecvGenericGLDisplay* eff =
                    ecvViewManager::instance().getEffectiveView()) {
            l->setDisplay(eff);
        }
        l->setVisible(true);
        l->setName("Lineation");

        // add to DB Tree
        insertPoint->addChild(l);
        m_app->addToDB(l, false, false, false, false);
    }

    // add point
    int index = l->addPointIndex(itemIdx);

    // is this the end point?
    if (l->size() == 2) {
        l->updateMetadata();  // calculate orientation & store. Also changes the
                              // name.
        l->showNameIn3D(ccCompass::drawName);

        l->invalidateBoundingBox();
        l->notifyGeometryUpdate();

        // Propagate bbox invalidation up the hierarchy so the parent group's
        // cached bbox VTK actor is removed and redrawn to include this
        // lineation
        for (ccHObject* p = l->getParent(); p; p = p->getParent()) {
            p->notifyGeometryUpdate();
        }

        // report orientation to console for convenience
        m_app->dispToConsole(QString("[ccCompass] Lineation = " + l->getName()),
                             ecvMainAppInterface::STD_CONSOLE_MESSAGE);

        m_app->refreshAll(false);
        m_app->updateUI();

        // start new lineation
        m_lineation_id = -1;
    }
}

// called when "Return" or "Space" is pressed, or the "Accept Button" is clicked
void ccLineationTool::accept() {
    cancel();  // removes any incomplete lineations
}

// called when the "Escape" is pressed, or the "Cancel" button is clicked
void ccLineationTool::cancel() {
    if (m_lineation_id != -1)  // there is an active lineation
    {
        ccPointPair* l = dynamic_cast<ccPointPair*>(
                m_app->dbRootObject()->find(m_lineation_id));
        if (l && l->size() < 2) {
            m_app->removeFromDB(l);  // remove incomplete lineation
            m_lineation_id = -1;
        }
    }
}
