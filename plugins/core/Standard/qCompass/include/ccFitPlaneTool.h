// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_FITPLANETOOL_HEADER
#define ECV_FITPLANETOOL_HEADER

#include <DgmOctreeReferenceCloud.h>

#include "ccFitPlane.h"
#include "ccMouseCircle.h"
#include "ccTool.h"

/*
Tool that is activated during "Plane Mode", generating fit planes from
point-picks
*/
class ccFitPlaneTool : public ccTool {
public:
    ccFitPlaneTool();
    virtual ~ccFitPlaneTool();

    // called when the tool is set to active (for initialization)
    void toolActivated() override;

    // called when the tool is set to disactive (for cleanup)
    void toolDisactivated() override;

    // called when a point in a point cloud gets picked while this tool is
    // active
    void pointPicked(ccHObject* insertPoint,
                     unsigned itemIdx,
                     ccPointCloud* cloud,
                     const CCVector3& P) override;

    // mouse circle element used for the selection
    ccMouseCircle* m_mouseCircle = nullptr;
};

#endif  // ECV_FITPLANETOOL_HEADER
