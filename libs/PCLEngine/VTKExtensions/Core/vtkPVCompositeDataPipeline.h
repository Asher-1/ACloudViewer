// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro
#include "vtkCompositeDataPipeline.h"

class QPCL_ENGINE_LIB_API vtkPVCompositeDataPipeline
    : public vtkCompositeDataPipeline {
public:
    static vtkPVCompositeDataPipeline* New();
    vtkTypeMacro(vtkPVCompositeDataPipeline, vtkCompositeDataPipeline);
    void PrintSelf(ostream& os, vtkIndent indent) override;

protected:
    vtkPVCompositeDataPipeline();
    ~vtkPVCompositeDataPipeline() override;

    // Copy information for the given request.
    void CopyDefaultInformation(vtkInformation* request,
                                int direction,
                                vtkInformationVector** inInfoVec,
                                vtkInformationVector* outInfoVec) override;

    // Remove update/whole extent when resetting pipeline information.
    void ResetPipelineInformation(int port, vtkInformation*) override;

private:
    vtkPVCompositeDataPipeline(const vtkPVCompositeDataPipeline&) = delete;
    void operator=(const vtkPVCompositeDataPipeline&) = delete;
};
