// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_LINEATION_HEADER
#define ECV_LINEATION_HEADER

#include <ecvPointCloud.h>

#include "ccPointPair.h"

/*
Class for representing/drawing lineations measured with qCompass.
*/
class ccLineation : public ccPointPair {
public:
    // ctors
    ccLineation(ccPointCloud* associatedCloud);
    ccLineation(ccPolyline* obj);

    // write metadata specific to this object
    void updateMetadata() override;

    // returns true if the given ccHObject is/was a ccLineation (as defined by
    // the objects metadata)
    static bool isLineation(ccHObject* obj);
};
#endif  // ECV_LINEATION_HEADER
