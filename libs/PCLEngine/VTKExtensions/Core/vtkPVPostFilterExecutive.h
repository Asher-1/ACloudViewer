// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
#include "vtkPVCompositeDataPipeline.h"

class vtkInformationInformationVectorKey;
class vtkInformationStringVectorKey;

class QPCL_ENGINE_LIB_API vtkPVPostFilterExecutive
    : public vtkPVCompositeDataPipeline {
public:
    static vtkPVPostFilterExecutive* New();
    vtkTypeMacro(vtkPVPostFilterExecutive, vtkPVCompositeDataPipeline);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    static vtkInformationInformationVectorKey* POST_ARRAYS_TO_PROCESS();
    static vtkInformationStringVectorKey* POST_ARRAY_COMPONENT_KEY();

    /**
     * Returns the data object stored with the DATA_OBJECT() in the
     * input port
     */
    vtkDataObject* GetCompositeInputData(int port,
                                         int index,
                                         vtkInformationVector** inInfoVec);

    vtkInformation* GetPostArrayToProcessInformation(int idx);
    void SetPostArrayToProcessInformation(int idx, vtkInformation* inInfo);

protected:
    vtkPVPostFilterExecutive();
    ~vtkPVPostFilterExecutive() override;

    // Overridden to always return true
    int NeedToExecuteData(int outputPort,
                          vtkInformationVector** inInfoVec,
                          vtkInformationVector* outInfoVec) override;

    bool MatchingPropertyInformation(vtkInformation* inputArrayInfo,
                                     vtkInformation* postArrayInfo);

private:
    vtkPVPostFilterExecutive(const vtkPVPostFilterExecutive&) = delete;
    void operator=(const vtkPVPostFilterExecutive&) = delete;
};
