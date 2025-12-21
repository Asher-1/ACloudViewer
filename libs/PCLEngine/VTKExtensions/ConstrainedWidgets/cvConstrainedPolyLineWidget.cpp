// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvConstrainedPolyLineWidget.h"

#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkObjectFactory.h>
#include <vtkPolyLineRepresentation.h>
#include <vtkRenderWindowInteractor.h>

#include <algorithm>
#include <cctype>

vtkStandardNewMacro(cvConstrainedPolyLineWidget);

cvConstrainedPolyLineWidget::cvConstrainedPolyLineWidget()
    : KeyEventCallbackCommand(nullptr) {
    // Don't initialize KeyEventCallbackCommand here - wait for SetEnabled()
    // This avoids accessing parent class members before they're fully
    // initialized
}

cvConstrainedPolyLineWidget::~cvConstrainedPolyLineWidget() {
    // CRITICAL: Remove observers BEFORE deleting callback command
    // This prevents callbacks from being triggered during/after destruction
    if (this->Interactor && this->KeyEventCallbackCommand) {
        this->Interactor->RemoveObserver(this->KeyEventCallbackCommand);
    }

    // Cleanup our callback command if we created one
    if (this->KeyEventCallbackCommand) {
        this->KeyEventCallbackCommand->Delete();
        this->KeyEventCallbackCommand = nullptr;
    }
}

// From ParaView vtkPolyLineWidget::SetEnabled pattern
void cvConstrainedPolyLineWidget::SetInteractor(
        vtkRenderWindowInteractor* iren) {
    // Call parent class first
    this->Superclass::SetInteractor(iren);
}

void cvConstrainedPolyLineWidget::SetEnabled(int enabling) {
    // CRITICAL: We need to register our custom keyboard handler BEFORE calling
    // parent SetEnabled(), otherwise the parent class (vtkPolyLineWidget) will
    // register its own keyboard handler first, which may conflict with ours.

    if (enabling && this->Interactor) {
        // Create our custom keyboard event handler if not already created
        if (!this->KeyEventCallbackCommand) {
            this->KeyEventCallbackCommand = vtkCallbackCommand::New();
            this->KeyEventCallbackCommand->SetClientData(this);
            this->KeyEventCallbackCommand->SetCallback(
                    cvConstrainedPolyLineWidget::ProcessKeyEvents);
        }

        // Register keyboard event observers with HIGH priority (10.0) BEFORE
        // parent SetEnabled This ensures our handler is called before the
        // parent class's handler, preventing conflicts and crashes
        this->Interactor->AddObserver(vtkCommand::KeyPressEvent,
                                      this->KeyEventCallbackCommand,
                                      10.0);  // High priority
        this->Interactor->AddObserver(vtkCommand::KeyReleaseEvent,
                                      this->KeyEventCallbackCommand,
                                      10.0);  // High priority
    }

    // Call parent SetEnabled (this does necessary initialization for widget
    // interaction)
    this->Superclass::SetEnabled(enabling);

    if (enabling && this->Interactor && this->KeyEventCallbackCommand) {
        // CRITICAL: After parent SetEnabled, parent class (vtkPolyLineWidget)
        // has registered its own keyboard observers. We need to remove ALL
        // keyboard observers and keep only ours.

        // Remove ALL keyboard observers (including parent's problematic ones)
        this->Interactor->RemoveObservers(vtkCommand::KeyPressEvent);
        this->Interactor->RemoveObservers(vtkCommand::KeyReleaseEvent);

        // Re-add ONLY our observers with high priority
        this->Interactor->AddObserver(vtkCommand::KeyPressEvent,
                                      this->KeyEventCallbackCommand, 10.0);
        this->Interactor->AddObserver(vtkCommand::KeyReleaseEvent,
                                      this->KeyEventCallbackCommand, 10.0);
    }

    if (!enabling && this->Interactor) {
        // Remove our custom keyboard event observers
        if (this->KeyEventCallbackCommand) {
            this->Interactor->RemoveObserver(this->KeyEventCallbackCommand);
        }
        // Note: Parent class observers are cleaned up by parent's SetEnabled(0)
        // Our ProcessKeyEvents has safety checks (GetEnabled, GetProcessEvents)
        // to prevent crashes even if called after disable
    }
}

// Keyboard event handling - directly copied from ParaView
// vtkPolyLineWidget::ProcessKeyEvents This is the core method of the control
// layer:
// 1. Receives keyboard events
// 2. Gets the representation
// 3. Sets the constraint status on the entire representation
void cvConstrainedPolyLineWidget::ProcessKeyEvents(vtkObject*,
                                                   unsigned long event,
                                                   void* clientdata,
                                                   void*) {
    cvConstrainedPolyLineWidget* self =
            static_cast<cvConstrainedPolyLineWidget*>(clientdata);

    // CRITICAL Safety checks: ensure widget is in valid state
    // This prevents crashes when callbacks are triggered during/after
    // destruction
    if (!self) {
        return;  // Widget destroyed
    }

    // Extra safety: check if widget is being destroyed
    try {
        // Try to access a member - if widget is deleted, this may throw or
        // crash
        if (!self->GetEnabled()) {
            return;  // Widget disabled - don't process any events
        }
    } catch (...) {
        return;  // Widget in invalid state
    }

    if (!self->Interactor || !self->WidgetRep) {
        return;  // Widget not fully initialized or already destroyed
    }

    if (!self->GetProcessEvents()) {
        return;  // Widget locked - don't process events
    }

    vtkPolyLineRepresentation* rep =
            vtkPolyLineRepresentation::SafeDownCast(self->WidgetRep);
    if (!rep) {
        return;
    }

    // CRITICAL: Safe access to Interactor - may be null during destruction
    char* cKeySym = nullptr;
    try {
        if (self->Interactor) {
            cKeySym = self->Interactor->GetKeySym();
        }
    } catch (...) {
        return;  // Interactor in invalid state
    }

    if (!cKeySym) {
        return;  // No key symbol available
    }

    std::string keySym = cKeySym;
    std::transform(keySym.begin(), keySym.end(), keySym.begin(), ::toupper);

    // Only handle X, Y, Z keys - ignore all other keys to prevent conflicts
    // with application shortcuts (like 'P' for picking)
    bool isConstraintKey = (keySym == "X" || keySym == "Y" || keySym == "Z");
    if (!isConstraintKey) {
        return;  // Let other keys be handled by application shortcuts
    }

    // ParaView vtkPolyLineWidget implementation (vtkPolyLineWidget.cxx:279-300)
    if (event == vtkCommand::KeyPressEvent) {
        if (keySym == "X") {
            // Constrain entire representation to X-axis
            rep->SetXTranslationAxisOn();
        } else if (keySym == "Y") {
            // Constrain entire representation to Y-axis
            rep->SetYTranslationAxisOn();
        } else if (keySym == "Z") {
            // Constrain entire representation to Z-axis
            rep->SetZTranslationAxisOn();
        }
    } else if (event == vtkCommand::KeyReleaseEvent) {
        if (keySym == "X" || keySym == "Y" || keySym == "Z") {
            // Release all constraints
            rep->SetTranslationAxisOff();
        }
    }
}
