// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_NOTE_HEADER
#define ECV_NOTE_HEADER

#include <ecvPointCloud.h>

#include "ccPointPair.h"

/*
Simple class used to represent notes created with qCompass
*/
class ccNote : public ccPointPair {
public:
    // ctors
    ccNote(ccPointCloud* associatedCloud);
    ccNote(ccPolyline* obj);

    // write metadata specific to this object
    void updateMetadata() override;

    static bool isNote(ccHObject* obj);
};

#endif  // ECV_NOTE_HEADER
