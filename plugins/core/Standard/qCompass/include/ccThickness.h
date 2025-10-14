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
Objects representing thickness measurements
*/
class ccThickness : public ccPointPair {
public:
    // ctors
    ccThickness(ccPointCloud* associatedCloud);
    ccThickness(ccPolyline* obj);

    // write metadata specific to this object
    void updateMetadata() override;

    // returns true if obj was/is a thickness measurement (as defined by the
    // objects metadata)
    static bool isThickness(ccHObject* obj);
};
