// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <GenericIndexedCloudPersist.h>
#include <ecvCone.h>
#include <ecvCylinder.h>
#include <ecvDrawContext.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvSphere.h>

#include "ccMeasurement.h"

/*
Template class, based around ccPolyline, that measurements comprising individual
or pairs of points can derive from.
*/
class ccPointPair : public ccPolyline, public ccMeasurement {
public:
    ccPointPair(ccPointCloud* associatedCloud);
    ccPointPair(ccPolyline* obj);  // used to construct from a polyline with the
                                   // correct data

    virtual ~ccPointPair();

    virtual void updateMetadata() {};

    // get the direction of this pair (not normalized)
    CCVector3 getDirection();

    void draw(CC_DRAW_CONTEXT& context) override;
    void getTypeID_recursive(std::vector<hideInfo>& hdInfos,
                             bool relative) override;

protected:
    float m_relMarkerScale = 5.0f;
    ecvGenericGLDisplay* m_lastDrawnView = nullptr;

    // overidden from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    void hideShowSubActors(bool visible);

    // static functions
public:
    // returns true if object is/was a ccPointPair (as defined by its MetaData)
    static bool isPointPair(ccHObject* object);
};
