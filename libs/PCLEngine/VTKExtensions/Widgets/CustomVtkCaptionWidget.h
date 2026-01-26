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
 * CustomVtkCaptionWidget with picking support for DB tree selection
 */

#include <vtkCaptionWidget.h>
#include <vtkCommand.h>
#include <vtkObject.h>
#include <vtkCallbackCommand.h>
#include <vtkSmartPointer.h>

// Forward declaration
class cc2DLabel;

class QPCL_ENGINE_LIB_API CustomVtkCaptionWidget : public vtkCaptionWidget {
public:
    static CustomVtkCaptionWidget *New();

    vtkTypeMacro(CustomVtkCaptionWidget, vtkCaptionWidget);

    void SetHandleEnabled(bool state);

    //! Set the associated cc2DLabel for selection notification
    void SetAssociatedLabel(cc2DLabel* label);

    //! Get the associated cc2DLabel
    cc2DLabel* GetAssociatedLabel() const { return m_associatedLabel; }

protected:
    CustomVtkCaptionWidget();
    ~CustomVtkCaptionWidget() override;

    //! Callback for widget interaction events
    static void OnWidgetInteraction(vtkObject* caller, unsigned long eventId,
                                    void* clientData, void* callData);

    cc2DLabel* m_associatedLabel;
    vtkSmartPointer<vtkCallbackCommand> m_interactionCallback;

private:
    CustomVtkCaptionWidget(const CustomVtkCaptionWidget&) = delete;
    void operator=(const CustomVtkCaptionWidget&) = delete;
};
