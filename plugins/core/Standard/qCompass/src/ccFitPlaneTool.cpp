// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccFitPlaneTool.h"

// CV_DB_LIB
// #include <ecvPointCloud.h>

ccFitPlaneTool::ccFitPlaneTool() : ccTool() {}

ccFitPlaneTool::~ccFitPlaneTool() {
    if (m_mouseCircle) {
        assert(false);  // we should never end up here...
        m_mouseCircle->ownerIsDead();
        delete m_mouseCircle;
        m_mouseCircle = nullptr;
    }
}

// called when the tool is set to active (for initialization)
void ccFitPlaneTool::toolActivated() {
    m_mouseCircle = new ccMouseCircle(ecvDisplayTools::GetCurrentScreen());
    m_mouseCircle->setVisible(true);

    // set orthographic view (as this tool doesn't work in perspective mode)
    ecvDisplayTools::SetPerspectiveState(false, true);
}

// called when the tool is set to disactive (for cleanup)
void ccFitPlaneTool::toolDisactivated() {
    if (m_mouseCircle) {
        m_mouseCircle->setVisible(false);
        delete m_mouseCircle;
        m_mouseCircle = nullptr;
    }
}

// called when a point in a point cloud gets picked while this tool is active
void ccFitPlaneTool::pointPicked(ccHObject* insertPoint,
                                 unsigned itemIdx,
                                 ccPointCloud* cloud,
                                 const CCVector3& P) {
    // get or generate octree
    ccOctree::Shared oct = cloud->getOctree();
    if (!oct) {
        oct = cloud->computeOctree();  // if the user clicked "no" when asked to
                                       // compute the octree then tough....
    }

    // nearest neighbour search
    float r = m_mouseCircle->getRadiusWorld();
    unsigned char level =
            oct->findBestLevelForAGivenNeighbourhoodSizeExtraction(r);
    cloudViewer::DgmOctree::NeighboursSet set;
    int n = oct->getPointsInSphericalNeighbourhood(P, PointCoordinateType(r),
                                                   set, level);
    // Put data in a point cloud class and encapsulate as a "neighbourhood"
    cloudViewer::DgmOctreeReferenceCloud nCloud(&set, n);
    cloudViewer::Neighbourhood Z(&nCloud);

    // Fit plane!
    double rms = 0.0;  // output for rms
    ccFitPlane* pPlane = ccFitPlane::Fit(&nCloud, &rms);

    if (pPlane)  // valid fit
    {
        pPlane->updateAttributes(rms, m_mouseCircle->getRadiusWorld());

        // make plane to add to display
        pPlane->setVisible(true);
        pPlane->setSelectionBehavior(ccHObject::SELECTION_IGNORED);

        // add plane to scene graph
        insertPoint->addChild(pPlane);
        // pPlane->setDisplay(m_app->getActiveWindow());
        // pPlane->prepareDisplayForRefresh_recursive(); //not sure what this
        // does, but it looks like fun

        // add plane to TOC
        m_app->addToDB(pPlane, false, false, false, false);

        // report orientation to console for convenience
        m_app->dispToConsole(
                QString("[ccCompass] Surface orientation estimate = " +
                        pPlane->getName()),
                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}
