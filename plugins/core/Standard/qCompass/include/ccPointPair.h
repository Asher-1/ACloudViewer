// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_POINTPAIR_HEADER
#define ECV_POINTPAIR_HEADER

#include <GenericIndexedCloudPersist.h>
#include <ecvCone.h>
#include <ecvCylinder.h>
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

    virtual ~ccPointPair() {}

    virtual void updateMetadata() {};

    // get the direction of this pair (not normalized)
    CCVector3 getDirection();

protected:
    // size that the point-markers are drawn
    float m_relMarkerScale = 5.0f;

    // overidden from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    // static functions
public:
    // returns true if object is/was a ccPointPair (as defined by its MetaData)
    static bool isPointPair(ccHObject* object);
};

#endif  // ECV_POINTPAIR_HEADER
