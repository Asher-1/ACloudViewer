// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
/**
 * @brief The CustomVtkCaptionWidget class
 * CustomVtkCaptionWidget
 */

#include <vtkCaptionWidget.h>

class QPCL_ENGINE_LIB_API CustomVtkCaptionWidget : public vtkCaptionWidget {
public:
    static CustomVtkCaptionWidget *New();

    vtkTypeMacro(CustomVtkCaptionWidget, vtkCaptionWidget);

    void SetHandleEnabled(bool state);
};
