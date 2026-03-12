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
    // Don't clear base class TranslationAxis - let it coexist
    // Custom axis takes precedence in GetTranslationVector/Translate
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
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxisOff() {
    this->CustomAxisEnabled = false;
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetTranslationAxisOff() {
    // CRITICAL: Clear both custom axis and base class axis
    this->CustomAxisEnabled = false;
    // Call base class to clear standard axes (sets TranslationAxis = NONE)
    this->Superclass::SetTranslationAxisOff();
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
        // Custom axis (L key): project onto CustomTranslationAxis
        // projection = (v · axis) * axis (where axis is unit vector)
        double dotProduct = vtkMath::Dot(p12, this->CustomTranslationAxis);
        v[0] = dotProduct * this->CustomTranslationAxis[0];
        v[1] = dotProduct * this->CustomTranslationAxis[1];
        v[2] = dotProduct * this->CustomTranslationAxis[2];
    } else {
        // Standard axes (X/Y/Z) or NONE
        // Access protected member TranslationAxis directly
        // (NONE=-1, XAxis=0, YAxis=1, ZAxis=2)
        int axis = this->TranslationAxis;

        if (axis == -1) {
            // NONE: no constraint, use full vector
            v[0] = p12[0];
            v[1] = p12[1];
            v[2] = p12[2];
        } else if (axis >= 0 && axis <= 2) {
            // X/Y/Z axis: project onto standard axis
            // In VTK 9.2, base class implementation may not work correctly
            // so we implement it ourselves
            v[0] = (axis == 0) ? p12[0] : 0.0;
            v[1] = (axis == 1) ? p12[1] : 0.0;
            v[2] = (axis == 2) ? p12[2] : 0.0;
        } else {
            // Fallback to base class for unknown axis values
            this->Superclass::GetTranslationVector(p1, p2, v);
        }
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
    double* worldPos = this->GetWorldPosition();
    double newPos[3];

    if (this->CustomAxisEnabled) {
        // Custom axis (L key): project and translate along custom axis
        // projection = (v · axis) * axis (where axis is unit vector)
        double dotProduct = vtkMath::Dot(v, this->CustomTranslationAxis);
        double dir[3];
        dir[0] = dotProduct * this->CustomTranslationAxis[0];
        dir[1] = dotProduct * this->CustomTranslationAxis[1];
        dir[2] = dotProduct * this->CustomTranslationAxis[2];

        for (int i = 0; i < 3; ++i) {
            newPos[i] = worldPos[i] + dir[i];
        }
        this->SetWorldPosition(newPos);
    } else {
        // Standard axes (X/Y/Z) or NONE
        // Access protected member TranslationAxis directly
        int axis = this->TranslationAxis;

        if (axis == -1) {
            // NONE: translate freely in all directions
            for (int i = 0; i < 3; ++i) {
                newPos[i] = worldPos[i] + v[i];
            }
            this->SetWorldPosition(newPos);
        } else if (axis >= 0 && axis <= 2) {
            // X/Y/Z axis: translate only along specified axis
            // In VTK 9.2, base class may not handle this correctly
            for (int i = 0; i < 3; ++i) {
                newPos[i] = worldPos[i];
            }
            newPos[axis] += v[axis];
            this->SetWorldPosition(newPos);
        } else {
            // Fallback to base class for unknown values
            this->Superclass::Translate(v);
        }
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

// VTK 9.3+ implementation - NO implementation needed!
// Base class (vtkHandleRepresentation via vtkWidgetRepresentation) provides:
// - enum Axis { NONE = -1, XAxis = 0, YAxis = 1, ZAxis = 2, Custom = 3 }
// - SetCustomTranslationAxisOn() { this->TranslationAxis = Axis::Custom; }
// - vtkSetVector3Macro(CustomTranslationAxis, double) - generates setter
// - SetTranslationAxisOff() { this->TranslationAxis = Axis::NONE; }
// - GetTranslationVector() - handles Custom axis projection
// - Translate() - handles Custom axis translation
//
// Our unified interface methods (SetCustomTranslationAxisOn,
// SetCustomTranslationAxis) are inherited directly from base class.
// No need to redefine or forward them - just use base class as-is!

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::PrintSelf(ostream& os,
                                                 vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
}

#endif
