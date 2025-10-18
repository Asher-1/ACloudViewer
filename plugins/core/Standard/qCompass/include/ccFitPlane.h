// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvNormalVectors.h>
#include <ecvPlane.h>

#include "ccMeasurement.h"

/*
ccFitPlane is a class that wraps around ccPlane and is used for storing the
planes-of-best-fit created using qCompass.
*/
class ccFitPlane : public ccPlane, public ccMeasurement {
public:
    ccFitPlane(ccPlane* p);
    ~ccFitPlane();

    // update the metadata attributes of this plane
    void updateAttributes(float rms, float search_r);

    // create a FitPlane object from a point cloud
    static ccFitPlane* Fit(cloudViewer::GenericIndexedCloudPersist* cloud,
                           double* rms);

    // returns true if object is a plane created by ccCompass (has the
    // associated data)
    static bool isFitPlane(ccHObject* object);
};
