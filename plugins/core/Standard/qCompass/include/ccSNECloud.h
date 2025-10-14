// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SNE_HEADER
#define ECV_SNE_HEADER

#include <ccMeasurement.h>
#include <ecvPointCloud.h>

/*
Class for representing/drawing lineations measured with qCompass.
*/
class ccSNECloud : public ccPointCloud, public ccMeasurement {
public:
    // ctors
    ccSNECloud();
    ccSNECloud(ccPointCloud* obj);

    // write metadata specific to this object
    void updateMetadata();

    // returns true if the given ccHObject is/was a ccLineation (as defined by
    // the objects metadata)
    static bool isSNECloud(ccHObject* obj);

protected:
    // overidden from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;
};
#endif  // ECV_SNE_HEADER
