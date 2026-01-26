// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CustomVtkCaptionWidget.h"

#include <vtkCommand.h>
#include <vtkDoubleArray.h>
#include <vtkHandleWidget.h>
#include <vtkRenderer.h>

// Qt
#include <QApplication>
#include <QMetaObject>
#include <QTimer>

// Include headers for cc2DLabel and ecvDisplayTools
#include <CVLog.h>
#include <CV_db/include/ecv2DLabel.h>
#include <CV_db/include/ecvDisplayTools.h>
#include <CV_db/include/ecvHObjectCaster.h>

vtkStandardNewMacro(CustomVtkCaptionWidget);

CustomVtkCaptionWidget::CustomVtkCaptionWidget() : m_associatedLabel(nullptr) {
    // Create callback for widget interaction events
    m_interactionCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    m_interactionCallback->SetCallback(OnWidgetInteraction);
    m_interactionCallback->SetClientData(this);

    // Listen for multiple interaction events to catch clicks on the widget
    // StartInteractionEvent: when user starts interacting (mouse down)
    // InteractionEvent: during interaction (mouse move while pressed)
    // LeftButtonPressEvent: explicit left button press
    this->AddObserver(vtkCommand::StartInteractionEvent, m_interactionCallback);
    this->AddObserver(vtkCommand::InteractionEvent, m_interactionCallback);
    this->AddObserver(vtkCommand::LeftButtonPressEvent, m_interactionCallback);
}

CustomVtkCaptionWidget::~CustomVtkCaptionWidget() {
    // Remove observer
    if (m_interactionCallback) {
        this->RemoveObserver(m_interactionCallback);
    }
    m_associatedLabel = nullptr;
}

void CustomVtkCaptionWidget::SetHandleEnabled(bool state) {
    this->HandleWidget->SetEnabled(state);
    state ? this->HandleWidget->ProcessEventsOn()
          : this->HandleWidget->ProcessEventsOff();
}

void CustomVtkCaptionWidget::SetAssociatedLabel(cc2DLabel* label) {
    m_associatedLabel = label;
}

void CustomVtkCaptionWidget::OnWidgetInteraction(vtkObject* caller,
                                                 unsigned long eventId,
                                                 void* clientData,
                                                 void* /*callData*/) {
    // Handle multiple event types to catch widget clicks
    // StartInteractionEvent: when user starts interacting (mouse down on
    // widget) LeftButtonPressEvent: explicit left button press on widget
    // InteractionEvent: during interaction (but we only want the initial click)
    if (eventId != vtkCommand::StartInteractionEvent &&
        eventId != vtkCommand::LeftButtonPressEvent) {
        return;
    }

    CustomVtkCaptionWidget* self =
            static_cast<CustomVtkCaptionWidget*>(clientData);
    if (!self || !self->m_associatedLabel) {
        return;
    }

    // Notify DB tree that the label is selected
    cc2DLabel* label = self->m_associatedLabel;

    if (!ecvDisplayTools::TheInstance()) {
        return;
    }

    // CRITICAL: Mark that a widget was clicked and stop deferred picking timer
    // to prevent it from overriding our widget selection. The deferred picking
    // timer is started in QVTKWidgetCustom::mouseReleaseEvent when
    // ProcessClickableItems returns false, and it would trigger doPicking()
    // which might clear our selection.
    ecvDisplayTools* tools = ecvDisplayTools::TheInstance();
    if (tools) {
        // Mark that a widget was clicked - this will prevent doPicking() from
        // executing and overriding our selection
        tools->m_widgetClicked = true;

        // Stop deferred picking timer immediately if already active
        if (tools->m_deferredPickingTimer.isActive()) {
            tools->m_deferredPickingTimer.stop();
        }
    }

    // Use QTimer::singleShot to safely emit signal from VTK callback
    // This ensures the signal is emitted in the Qt event loop
    QTimer::singleShot(0, [label, tools]() {
        if (!ecvDisplayTools::TheInstance()) {
            return;
        }

        // Stop deferred picking timer again in case mouseReleaseEvent started
        // it after our OnWidgetInteraction callback
        if (tools && tools->m_deferredPickingTimer.isActive()) {
            tools->m_deferredPickingTimer.stop();
        }

        // Directly emit the signal - we're now in Qt event loop
        emit ecvDisplayTools::TheInstance() -> entitySelectionChanged(label);
    });
}
