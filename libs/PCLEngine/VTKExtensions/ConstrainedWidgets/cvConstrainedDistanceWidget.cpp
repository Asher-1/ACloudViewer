// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// 控制层实现 - 从 ParaView vtkLineWidget2.cxx 直接移植

#include "cvConstrainedDistanceWidget.h"

#include "cvCustomAxisHandleRepresentation.h"

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkHandleRepresentation.h>
#include <vtkLineRepresentation.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVersion.h>

// STL
#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

vtkStandardNewMacro(cvConstrainedDistanceWidget);

//------------------------------------------------------------------------------
cvConstrainedDistanceWidget::cvConstrainedDistanceWidget()
    : KeyPressObserverId(0),
      KeyReleaseObserverId(0),
      KeyboardCallbackCommand(nullptr) {}

//------------------------------------------------------------------------------
cvConstrainedDistanceWidget::~cvConstrainedDistanceWidget() {
    if (this->Interactor) {
        if (this->KeyPressObserverId) {
            this->Interactor->RemoveObserver(this->KeyPressObserverId);
        }
        if (this->KeyReleaseObserverId) {
            this->Interactor->RemoveObserver(this->KeyReleaseObserverId);
        }
    }
    if (this->KeyboardCallbackCommand) {
        this->KeyboardCallbackCommand->Delete();
        this->KeyboardCallbackCommand = nullptr;
    }
}

//------------------------------------------------------------------------------
// Following ParaView vtkLineWidget2::SetInteractor pattern
void cvConstrainedDistanceWidget::SetInteractor(
        vtkRenderWindowInteractor* iren) {
    // Call superclass first
    this->Superclass::SetInteractor(iren);
}

//------------------------------------------------------------------------------
void cvConstrainedDistanceWidget::SetEnabled(int enabling) {
    // CRITICAL: We need to prevent the parent class (vtkLineWidget2) from
    // registering its own keyboard event handler, which conflicts with ours.
    // The parent class registers its keyboard handler in SetEnabled(1).

    if (enabling && this->Interactor && this->KeyPressObserverId == 0) {
        // Register our custom keyboard event observers BEFORE calling parent
        // SetEnabled
        if (!this->KeyboardCallbackCommand) {
            this->KeyboardCallbackCommand = vtkCallbackCommand::New();
            this->KeyboardCallbackCommand->SetCallback(
                    cvConstrainedDistanceWidget::ProcessKeyEvents);
            this->KeyboardCallbackCommand->SetClientData(this);
        }

        // Use high priority (10.0) to ensure our handler is called first
        this->KeyPressObserverId = this->Interactor->AddObserver(
                vtkCommand::KeyPressEvent, this->KeyboardCallbackCommand, 10.0);
        this->KeyReleaseObserverId = this->Interactor->AddObserver(
                vtkCommand::KeyReleaseEvent, this->KeyboardCallbackCommand,
                10.0);
    }

    // Call parent SetEnabled (this does necessary initialization for widget
    // interaction)
    this->Superclass::SetEnabled(enabling);

    if (enabling && this->Interactor) {
        // CRITICAL: After parent SetEnabled, parent class (vtkLineWidget2) has
        // registered its own keyboard observers. We need to remove ALL keyboard
        // observers and keep only ours. Save our observer IDs first
        unsigned long savedKeyPress = this->KeyPressObserverId;
        unsigned long savedKeyRelease = this->KeyReleaseObserverId;
        vtkCallbackCommand* savedCallback = this->KeyboardCallbackCommand;

        // Remove ALL keyboard observers (including parent's problematic ones)
        this->Interactor->RemoveObservers(vtkCommand::KeyPressEvent);
        this->Interactor->RemoveObservers(vtkCommand::KeyReleaseEvent);

        // Re-add ONLY our observers with high priority
        if (savedCallback) {
            this->KeyPressObserverId = this->Interactor->AddObserver(
                    vtkCommand::KeyPressEvent, savedCallback, 10.0);
            this->KeyReleaseObserverId = this->Interactor->AddObserver(
                    vtkCommand::KeyReleaseEvent, savedCallback, 10.0);
        }
    }

    if (!enabling && this->Interactor) {
        // Remove our custom keyboard event observers
        if (this->KeyPressObserverId) {
            this->Interactor->RemoveObserver(this->KeyPressObserverId);
            this->KeyPressObserverId = 0;
        }
        if (this->KeyReleaseObserverId) {
            this->Interactor->RemoveObserver(this->KeyReleaseObserverId);
            this->KeyReleaseObserverId = 0;
        }
        // Parent class should clean up its observers in its SetEnabled(0)
        // Our ProcessKeyEvents has strong safety checks to prevent crashes
        // even if parent's observers are somehow still called
    }
}

//------------------------------------------------------------------------------
void cvConstrainedDistanceWidget::ProcessKeyEvents(vtkObject* caller,
                                                   unsigned long event,
                                                   void* clientdata,
                                                   void* calldata) {
    cvConstrainedDistanceWidget* self =
            static_cast<cvConstrainedDistanceWidget*>(clientdata);

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

    vtkLineRepresentation* rep =
            vtkLineRepresentation::SafeDownCast(self->WidgetRep);
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

    // Only handle X, Y, Z, L keys - ignore all other keys to prevent conflicts
    // with application shortcuts (like 'P' for picking)
    bool isConstraintKey =
            (keySym == "X" || keySym == "Y" || keySym == "Z" || keySym == "L");
    if (!isConstraintKey) {
        return;  // Let other keys be handled by application shortcuts (don't
                 // abort)
    }

    if (event == vtkCommand::KeyPressEvent) {
        if (keySym == "X") {
            rep->GetPoint1Representation()->SetXTranslationAxisOn();
            rep->GetPoint2Representation()->SetXTranslationAxisOn();
            rep->GetLineHandleRepresentation()->SetXTranslationAxisOn();
            rep->GetPoint1Representation()->SetConstrained(true);
            rep->GetPoint2Representation()->SetConstrained(true);
            rep->GetLineHandleRepresentation()->SetConstrained(true);
        } else if (keySym == "Y") {
            rep->GetPoint1Representation()->SetYTranslationAxisOn();
            rep->GetPoint2Representation()->SetYTranslationAxisOn();
            rep->GetLineHandleRepresentation()->SetYTranslationAxisOn();
            rep->GetPoint1Representation()->SetConstrained(true);
            rep->GetPoint2Representation()->SetConstrained(true);
            rep->GetLineHandleRepresentation()->SetConstrained(true);
        } else if (keySym == "Z") {
            rep->GetPoint1Representation()->SetZTranslationAxisOn();
            rep->GetPoint2Representation()->SetZTranslationAxisOn();
            rep->GetLineHandleRepresentation()->SetZTranslationAxisOn();
            rep->GetPoint1Representation()->SetConstrained(true);
            rep->GetPoint2Representation()->SetConstrained(true);
            rep->GetLineHandleRepresentation()->SetConstrained(true);
        } else if (keySym == "L") {
            // Custom translation axis along the line direction
            // This works with all VTK versions using
            // cvCustomAxisHandleRepresentation
            double p1[3], p2[3], v[3];
            rep->GetPoint1WorldPosition(p1);
            rep->GetPoint2WorldPosition(p2);
            vtkMath::Subtract(p2, p1, v);
            vtkMath::Normalize(v);

            // Try to cast to cvCustomAxisHandleRepresentation
            auto* h1 = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetPoint1Representation());
            auto* h2 = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetPoint2Representation());
            auto* hLine = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetLineHandleRepresentation());

            if (h1 && h2 && hLine) {
                // Use custom axis handle representation (works in all VTK
                // versions)
                h1->SetCustomTranslationAxisOn();
                h1->SetCustomTranslationAxis(v);
                h2->SetCustomTranslationAxisOn();
                h2->SetCustomTranslationAxis(v);
                hLine->SetCustomTranslationAxisOn();
                hLine->SetCustomTranslationAxis(v);
                h1->SetConstrained(true);
                h2->SetConstrained(true);
                hLine->SetConstrained(true);
            } else {
                // Fallback: use base class methods if available
#if (VTK_MAJOR_VERSION > 9) || \
        (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3)
                // VTK 9.3+ has native support
                rep->GetPoint1Representation()->SetCustomTranslationAxisOn();
                rep->GetPoint1Representation()->SetCustomTranslationAxis(v);
                rep->GetPoint2Representation()->SetCustomTranslationAxisOn();
                rep->GetPoint2Representation()->SetCustomTranslationAxis(v);
                rep->GetLineHandleRepresentation()
                        ->SetCustomTranslationAxisOn();
                rep->GetLineHandleRepresentation()->SetCustomTranslationAxis(v);
                rep->GetPoint1Representation()->SetConstrained(true);
                rep->GetPoint2Representation()->SetConstrained(true);
                rep->GetLineHandleRepresentation()->SetConstrained(true);
#else
                // For older VTK without custom handles, log a warning
                CVLog::Warning(
                        "L key: Custom axis handle representation not in use. "
                        "Please use cvCustomAxisHandleRepresentation for full "
                        "functionality.");
#endif
            }
        }
    } else if (event == vtkCommand::KeyReleaseEvent) {
        if (keySym == "L" || keySym == "X" || keySym == "Y" || keySym == "Z") {
            // Try to cast to cvCustomAxisHandleRepresentation first
            auto* h1 = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetPoint1Representation());
            auto* h2 = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetPoint2Representation());
            auto* hLine = dynamic_cast<cvCustomAxisHandleRepresentation*>(
                    rep->GetLineHandleRepresentation());

            if (h1 && h2 && hLine) {
                // Use custom axis handle representation
                if (keySym == "L") {
                    h1->SetCustomTranslationAxisOff();
                    h2->SetCustomTranslationAxisOff();
                    hLine->SetCustomTranslationAxisOff();
                }
                // For X/Y/Z, SetTranslationAxis(-1) is called by base class
                h1->SetTranslationAxis(-1);
                h2->SetTranslationAxis(-1);
                hLine->SetTranslationAxis(-1);
            } else {
                // Fallback to standard methods
#if (VTK_MAJOR_VERSION > 9) || \
        (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3)
                // VTK 9.3+ has SetTranslationAxisOff()
                rep->GetPoint1Representation()->SetTranslationAxisOff();
                rep->GetPoint2Representation()->SetTranslationAxisOff();
                rep->GetLineHandleRepresentation()->SetTranslationAxisOff();
#else
                // VTK 9.2: Use SetTranslationAxis(-1) to disable
                rep->GetPoint1Representation()->SetTranslationAxis(-1);
                rep->GetPoint2Representation()->SetTranslationAxis(-1);
                rep->GetLineHandleRepresentation()->SetTranslationAxis(-1);
#endif
            }

            rep->GetPoint1Representation()->SetConstrained(false);
            rep->GetPoint2Representation()->SetConstrained(false);
            rep->GetLineHandleRepresentation()->SetConstrained(false);
        }
    }
}

//------------------------------------------------------------------------------
void cvConstrainedDistanceWidget::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
}
