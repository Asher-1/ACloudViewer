// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <qinputdialog.h>
#include <qmainwindow.h>

#include "ccGeoObject.h"
#include "ccPinchNode.h"
#include "ccTool.h"

/*
Tool used to create PinchNodes.
*/
class ccPinchNodeTool : public ccTool {
public:
    ccPinchNodeTool();
    virtual ~ccPinchNodeTool();

    // called when the tool is set to active (for initialization)
    virtual void toolActivated() override;

    // called when the tool is set to disactive (for cleanup)
    virtual void toolDisactivated() override;

    // called when a point in a point cloud gets picked while this tool is
    // active
    void pointPicked(ccHObject* insertPoint,
                     unsigned itemIdx,
                     ccPointCloud* cloud,
                     const CCVector3& P) override;
};
