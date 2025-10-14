// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ccLineation.h"
#include "ccTool.h"

/*
Tool used to create/measure lineations
*/
class ccLineationTool : public ccTool {
public:
    ccLineationTool();
    virtual ~ccLineationTool();

    // called when the tool is set to disactive (for cleanup)
    void toolDisactivated() override;

    // called when a point in a point cloud gets picked while this tool is
    // active
    void pointPicked(ccHObject* insertPoint,
                     unsigned itemIdx,
                     ccPointCloud* cloud,
                     const CCVector3& P) override;

    // called when "Return" or "Space" is pressed, or the "Accept Button" is
    // clicked
    void accept() override;  // do nothing

    // called when the "Escape" is pressed, or the "Cancel" button is clicked
    void cancel() override;  // do nothing
protected:
    int m_lineation_id = -1;  // ID of the lineation object being written to
};
