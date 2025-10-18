// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvPointCloud.h>

#include "ccPointPair.h"

/*
Simple class used to create/represent/draw pinch-nodes created using qCompass
*/
class ccPinchNode : public ccPointPair {
public:
    // ctors
    ccPinchNode(ccPointCloud* associatedCloud);
    ccPinchNode(ccPolyline* obj);

    // write metadata specific to this object
    void updateMetadata() override;

    // returns true if obj is/was a pinchNode (as recorded by its metadata)
    static bool isPinchNode(ccHObject* obj);
};
