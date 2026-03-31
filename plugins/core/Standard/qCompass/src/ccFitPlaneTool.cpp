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
    ccOctree::Shared oct = cloud->getOctree();
    if (!oct) {
        oct = cloud->computeOctree();
        if (!oct) {
            m_app->dispToConsole(
                    "[ccFitPlaneTool] Failed to compute the cloud octree",
                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
    }

    PointCoordinateType r =
            static_cast<PointCoordinateType>(m_mouseCircle->getRadiusWorld());
    if (r <= 0) {
        m_app->dispToConsole(
                "[ccFitPlaneTool] Invalid search radius (pixel size may not "
                "be initialized). Please try moving the mouse first.",
                ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
        return;
    }
    unsigned char level =
            oct->findBestLevelForAGivenNeighbourhoodSizeExtraction(r);
    cloudViewer::DgmOctree::NeighboursSet set;
    int n = oct->getPointsInSphericalNeighbourhood(P, r, set, level);
    cloudViewer::DgmOctreeReferenceCloud nCloud(&set, n);

    double rms = 0.0;
    ccFitPlane* pPlane = ccFitPlane::Fit(&nCloud, &rms);

    if (pPlane) {
        pPlane->copyGlobalShiftAndScale(*cloud);

        const ecvViewportParameters& viewportParams =
                ecvDisplayTools::GetViewportParameters();
        CCVector3d viewDir = viewportParams.getViewDir();
        if (pPlane->getNormal().toDouble().dot(viewDir) > 0) {
            pPlane->flip();
        }

        pPlane->updateAttributes(rms, r);

        pPlane->setVisible(true);
        pPlane->setSelectionBehavior(ccHObject::SELECTION_IGNORED);

        insertPoint->addChild(pPlane);

        m_app->addToDB(pPlane, false, false, false, false);

        m_app->dispToConsole(
                QString("[ccCompass] Surface orientation estimate = " +
                        pPlane->getName()),
                ecvMainAppInterface::STD_CONSOLE_MESSAGE);
    }
}
