// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvCustomAxisHandleRepresentation.h"

#include <vtkMath.h>
#include <vtkObjectFactory.h>

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
    if (!this->CustomAxisEnabled) {
        this->CustomAxisEnabled = true;
        // Disable standard axis constraints
        this->SetTranslationAxis(-1);  // -1 = Axis::NONE
        this->Modified();
    }
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxisOff() {
    if (this->CustomAxisEnabled) {
        this->CustomAxisEnabled = false;
        this->Modified();
    }
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxis(
        double axis[3]) {
    bool changed = false;
    for (int i = 0; i < 3; i++) {
        if (this->CustomTranslationAxis[i] != axis[i]) {
            this->CustomTranslationAxis[i] = axis[i];
            changed = true;
        }
    }

    if (changed) {
        // Normalize the axis vector
        vtkMath::Normalize(this->CustomTranslationAxis);
        this->Modified();
    }
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::SetCustomTranslationAxis(double x,
                                                                double y,
                                                                double z) {
    double axis[3] = {x, y, z};
    this->SetCustomTranslationAxis(axis);
}

//------------------------------------------------------------------------------
void cvCustomAxisHandleRepresentation::GetTranslationVector(const double* p1,
                                                            const double* p2,
                                                            double* v) const {
    double p12[3];
    vtkMath::Subtract(p2, p1, p12);

    if (this->CustomAxisEnabled) {
        // Project the translation vector onto the custom axis
        // Following ParaView's implementation in vtkHandleRepresentation.cxx
        vtkMath::ProjectVector(p12, this->CustomTranslationAxis, v);
    } else {
        // Use standard translation (or axis-constrained if TranslationAxis is
        // set)
        int translationAxis =
                const_cast<cvCustomAxisHandleRepresentation*>(this)
                        ->GetTranslationAxis();
        if (translationAxis == -1) {  // Axis::NONE
            // Free translation
            for (int i = 0; i < 3; ++i) {
                v[i] = p12[i];
            }
        } else {
            // Standard axis constraint (X=0, Y=1, Z=2)
            for (int i = 0; i < 3; ++i) {
                if (translationAxis == i) {
                    v[i] = p12[i];
                } else {
                    v[i] = 0.0;
                }
            }
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
    if (this->CustomAxisEnabled) {
        // Project the translation vector onto the custom axis
        // Following ParaView's implementation in vtkHandleRepresentation.cxx
        double dir[3];
        vtkMath::ProjectVector(v, this->CustomTranslationAxis, dir);

        // Apply the translation
        double* worldPos = this->GetWorldPosition();
        double newPos[3];
        for (int i = 0; i < 3; ++i) {
            newPos[i] = worldPos[i] + dir[i];
        }
        this->SetWorldPosition(newPos);
    } else {
        // Use base class implementation for standard axes
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

// VTK 9.3+ implementation - just use base class
void cvCustomAxisHandleRepresentation::PrintSelf(ostream& os,
                                                 vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
}

#endif
