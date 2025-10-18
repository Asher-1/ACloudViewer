// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  // needed for export macro

class vtkInformationStringKey;
class vtkInformationDoubleVectorKey;

class QPCL_ENGINE_LIB_API vtkPVInformationKeys {
public:
    /**
     * Key to store the label that should be used for labeling the time in the
     * UI
     */
    static vtkInformationStringKey* TIME_LABEL_ANNOTATION();

    //@{
    /**
     * Key to store the bounding box of the entire data set in pipeline
     * information.
     */
    static vtkInformationDoubleVectorKey* WHOLE_BOUNDING_BOX();
};
//@}

// VTK-HeaderTest-Exclude: vtkPVInformationKeys.h
