// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <qinputdialog.h>
#include <qmainwindow.h>

#include "ccNote.h"
#include "ccTool.h"

/*
Tool used to create notes and associated them with points in a cloud.
*/
class ccNoteTool : public ccTool {
public:
    ccNoteTool();
    virtual ~ccNoteTool();

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
