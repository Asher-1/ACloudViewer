// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVPostFilterExecutive.h
 * @brief Executive for post-filter pipeline with array processing support
 */

#include "qVTK.h"
#include "vtkPVCompositeDataPipeline.h"

class vtkInformationInformationVectorKey;
class vtkInformationStringVectorKey;

/**
 * @class vtkPVPostFilterExecutive
 * @brief Pipeline executive for vtkPVPostFilter array conversion requests
 */
class QVTK_ENGINE_LIB_API vtkPVPostFilterExecutive
    : public vtkPVCompositeDataPipeline {
public:
    static vtkPVPostFilterExecutive* New();
    vtkTypeMacro(vtkPVPostFilterExecutive, vtkPVCompositeDataPipeline);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    static vtkInformationInformationVectorKey* POST_ARRAYS_TO_PROCESS();
    static vtkInformationStringVectorKey* POST_ARRAY_COMPONENT_KEY();

    /**
     * Returns the data object stored with the DATA_OBJECT() in the input port
     * @param port Input port index
     * @param index Block index for composite data
     * @param inInfoVec Input information vectors
     * @return Data object or nullptr
     */
    vtkDataObject* GetCompositeInputData(int port,
                                         int index,
                                         vtkInformationVector** inInfoVec);

    /**
     * @param idx Index of post array to process
     * @return Information object for the array
     */
    vtkInformation* GetPostArrayToProcessInformation(int idx);
    /**
     * @param idx Index of post array
     * @param inInfo Information to set
     */
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
