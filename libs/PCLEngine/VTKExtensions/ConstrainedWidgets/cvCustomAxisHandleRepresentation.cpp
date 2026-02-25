// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvCustomAxisHandleRepresentation.h"

#include <vtkMath.h>
#include <vtkObjectFactory.h>

// For logging
#include <CVLog.h>

vtkStandardNewMacro(cvCustomAxisHandleRepresentation);

#if !((VTK_MAJOR_VERSION > 9) || \
      (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3))

//------------------------------------------------------------------------------
cvCustomAxisHandleRepresentation::cvCustomAxisHandleRepresentation()
    : CustomAxisEnabled(false) {
    this->CustomTranslationAxis[0] = 1.0;
    this->CustomTranslationAxis[1] = 0.0;
    this->CustomTranslationAxis[2] = 0.0;
}

//------------------------------------------------------------------------------
cvCustomAxisHandleRepresentation::~cvCustomAxisHandleRepresentation() = default;

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxisOn() {
    this->CustomAxisEnabled = true;
    // Clear base class TranslationAxis to prevent conflict with standard axes
    // (In VTK 9.3+, this would be Axis::Custom, but VTK 9.2 doesn't have it)
    this->Superclass::SetTranslationAxis(-1);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxisOff() {
    this->CustomAxisEnabled = false;
    // Also clear base class TranslationAxis to fully align with ParaView
    // behavior
    this->Superclass::SetTranslationAxis(-1);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxis(
        double axis[3]) {
    for (int i = 0; i < 3; i++) {
        this->CustomTranslationAxis[i] = axis[i];
    }
    vtkMath::Normalize(this->CustomTranslationAxis);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxis(double x,
                                                                double y,
                                                                double z) {
    double axis[3] = {x, y, z};
    this->SetCustomTranslationAxis(axis);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetTranslationAxisOff() {
    this->CustomAxisEnabled = false;
    // Also clear base class TranslationAxis to fully align with ParaView
    // behavior
    this->Superclass::SetTranslationAxis(-1);
}

//------------------------------------------------------------------------------
int cvCustomAxisHandleRepresentation::DetermineConstraintAxis(
        int constraint, double* x, double* startPickPoint) {
    if (this->CustomAxisEnabled) {
        return -1;  // Don't auto-constrain to X/Y/Z when custom axis is active
    }
    return this->Superclass::DetermineConstraintAxis(constraint, x,
                                                     startPickPoint);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::GetTranslationVector(const double* p1,
                                                            const double* p2,
                                                            double* v) const {
    double p12[3];
    vtkMath::Subtract(p2, p1, p12);

    if (this->CustomAxisEnabled) {
        vtkMath::ProjectVector(p12, this->CustomTranslationAxis, v);
    } else {
        this->Superclass::GetTranslationVector(p1, p2, v);
    }
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::Translate(const double* p1,
                                                 const double* p2) {
    double v[3];
    this->GetTranslationVector(p1, p2, v);
    this->Translate(v);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::Translate(const double* v) {
    if (this->CustomAxisEnabled) {
        double dir[3];
        vtkMath::ProjectVector(v, this->CustomTranslationAxis, dir);
        double* worldPos = this->GetWorldPosition();
        double newPos[3];
        for (int i = 0; i < 3; ++i) {
            newPos[i] = worldPos[i] + dir[i];
        }
        this->SetWorldPosition(newPos);
    } else {
        this->Superclass::Translate(v);
    }
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::PrintSelf(ostream& os,
                                                 vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);

    os << indent
       << "Custom Axis Enabled: " << (this->CustomAxisEnabled ? "On" : "Off")
       << "\n";
    os << indent << "Custom Translation Axis: ("
       << this->CustomTranslationAxis[0] << ", "
       << this->CustomTranslationAxis[1] << ", "
       << this->CustomTranslationAxis[2] << ")\n";
}

#else

// VTK 9.3+ implementation
//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetTranslationAxisOff() {
    // In VTK 9.3+, base class SetTranslationAxisOff() handles all axes including custom
    // Just forward to base class implementation
    this->Superclass::SetTranslationAxisOff();
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::PrintSelf(ostream& os,
                                                 vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
}

#endif
