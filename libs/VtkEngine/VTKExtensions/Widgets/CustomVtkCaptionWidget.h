// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file CustomVtkCaptionWidget.h
 * @brief Caption widget with picking support for DB tree selection.
 */

#include <vtkCallbackCommand.h>
#include <vtkCaptionWidget.h>
#include <vtkCommand.h>
#include <vtkObject.h>
#include <vtkSmartPointer.h>

#include "qVTK.h"

// Forward declaration
class cc2DLabel;

/**
 * @class CustomVtkCaptionWidget
 * @brief Caption widget with cc2DLabel association for DB tree selection.
 */
class QVTK_ENGINE_LIB_API CustomVtkCaptionWidget : public vtkCaptionWidget {
public:
    static CustomVtkCaptionWidget* New();

    vtkTypeMacro(CustomVtkCaptionWidget, vtkCaptionWidget);

    /// @param state Enable/disable the handle
    void SetHandleEnabled(bool state);

    /// @param label Associated cc2DLabel for selection notification
    void SetAssociatedLabel(cc2DLabel* label);

    /// @return Associated cc2DLabel
    cc2DLabel* GetAssociatedLabel() const { return m_associatedLabel; }

protected:
    CustomVtkCaptionWidget();
    ~CustomVtkCaptionWidget() override;

    //! Callback for widget interaction events
    static void OnWidgetInteraction(vtkObject* caller,
                                    unsigned long eventId,
                                    void* clientData,
                                    void* callData);

    cc2DLabel* m_associatedLabel;
    vtkSmartPointer<vtkCallbackCommand> m_interactionCallback;

private:
    CustomVtkCaptionWidget(const CustomVtkCaptionWidget&) = delete;
    void operator=(const CustomVtkCaptionWidget&) = delete;
};
