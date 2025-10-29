// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CustomVtkCaptionWidget.h"

#include <vtkDoubleArray.h>
#include <vtkHandleWidget.h>
#include <vtkRenderer.h>

vtkStandardNewMacro(CustomVtkCaptionWidget);

void CustomVtkCaptionWidget::SetHandleEnabled(bool state) {
    this->HandleWidget->SetEnabled(state);
    state ? this->HandleWidget->ProcessEventsOn()
          : this->HandleWidget->ProcessEventsOff();
}
