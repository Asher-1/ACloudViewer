// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccPinchNodeTool.h"

#include <CVLog.h>

ccPinchNodeTool::ccPinchNodeTool() : ccTool() {}

ccPinchNodeTool::~ccPinchNodeTool() {}

// called when a point in a point cloud gets picked while this tool is active
void ccPinchNodeTool::pointPicked(ccHObject* insertPoint,
                                  unsigned itemIdx,
                                  ccPointCloud* cloud,
                                  const CCVector3& P) {
    // get insert-point if there is an active GeoObject
    ccGeoObject* geoObj = ccGeoObject::getGeoObjectParent(insertPoint);
    if (geoObj)  // there is an active GeoObject
    {
        ccHObject* region = geoObj->getRegion(ccGeoObject::INTERIOR);
        if (!region) {
            CVLog::Error("[Compass] Internal error: no interior region");
            return;
        }
        insertPoint = region;
    } else {
        CVLog::Error(
                "[Compass] PinchNodes can only be added to GeoObjects. "
                "Please select one!");
        return;
    }

    // create a 1-point lineation object (highlights node-location)
    ccPointPair* l = new ccPinchNode(cloud);
    l->setName("tip");
    l->showNameIn3D(false);
    l->addPointIndex(itemIdx);

    // add to scene graph
    insertPoint->addChild(l);
    m_app->addToDB(l);
}

// called when the tool is set to active (for initialization)
void ccPinchNodeTool::toolActivated() {
    // donothing
}

// called when the tool is set to disactive (for cleanup)
void ccPinchNodeTool::toolDisactivated() {
    // donothing
}