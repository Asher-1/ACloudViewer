// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvConstrainedContourRepresentation.h"

#include <vtkActor2D.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

vtkStandardNewMacro(cvConstrainedContourRepresentation);

//----------------------------------------------------------------------------
cvConstrainedContourRepresentation::cvConstrainedContourRepresentation()
    : LabelSuffix(nullptr), ShowLabel(1) {
    // Create the label actor
    this->LabelActor = vtkSmartPointer<vtkTextActor>::New();

    // Configure label text properties
    vtkTextProperty* textProp = this->LabelActor->GetTextProperty();
    textProp->SetFontSize(20);
    textProp->SetBold(1);
    textProp->SetColor(1.0, 1.0, 1.0);  // White text
    textProp->SetShadow(1);
    textProp->SetJustificationToCentered();
    textProp->SetVerticalJustificationToCentered();

    // Default label text
    this->LabelActor->SetInput("Contour");

    // Initially visible
    this->LabelActor->SetVisibility(1);
}

//----------------------------------------------------------------------------
cvConstrainedContourRepresentation::~cvConstrainedContourRepresentation() {
    // Remove label actor from renderer before destruction
    if (this->Renderer && this->LabelActor) {
        this->Renderer->RemoveActor2D(this->LabelActor);
    }

    if (this->LabelSuffix) {
        delete[] this->LabelSuffix;
        this->LabelSuffix = nullptr;
    }
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::SetLabelSuffix(const char* suffix) {
    if (this->LabelSuffix) {
        delete[] this->LabelSuffix;
        this->LabelSuffix = nullptr;
    }

    if (suffix) {
        size_t len = strlen(suffix);
        this->LabelSuffix = new char[len + 1];
        strcpy(this->LabelSuffix, suffix);
    }

    this->UpdateLabel();
    this->Modified();
}

//----------------------------------------------------------------------------
const char* cvConstrainedContourRepresentation::GetLabelSuffix() const {
    return this->LabelSuffix;
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::SetShowLabel(int show) {
    if (this->ShowLabel != show) {
        this->ShowLabel = show;
        if (this->LabelActor) {
            this->LabelActor->SetVisibility(show && this->GetVisibility());
        }
        this->Modified();
    }
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::SetRenderer(vtkRenderer* ren) {
    // Remove from old renderer
    if (this->Renderer && this->LabelActor) {
        this->Renderer->RemoveActor2D(this->LabelActor);
    }

    // Call parent
    this->Superclass::SetRenderer(ren);

    // Add to new renderer
    if (this->Renderer && this->LabelActor) {
        this->Renderer->AddActor2D(this->LabelActor);
    }
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::SetVisibility(vtkTypeBool visible) {
    // Call parent
    this->Superclass::SetVisibility(visible);

    // Control label visibility
    if (this->LabelActor) {
        this->LabelActor->SetVisibility(visible && this->ShowLabel);
    }
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::BuildRepresentation() {
    // Call parent to build contour representation
    this->Superclass::BuildRepresentation();

    // Update label position and text
    this->UpdateLabel();
}

//----------------------------------------------------------------------------
void cvConstrainedContourRepresentation::UpdateLabel() {
    if (!this->LabelActor) {
        return;
    }

    // Update label text
    std::string labelText = "Contour";
    if (this->LabelSuffix) {
        labelText += " ";
        labelText += this->LabelSuffix;
    }
    this->LabelActor->SetInput(labelText.c_str());

    // Calculate label position (centroid of contour nodes)
    int numNodes = this->GetNumberOfNodes();
    if (numNodes > 0 && this->Renderer) {
        double centroid[3] = {0.0, 0.0, 0.0};

        // Calculate centroid in world coordinates
        for (int i = 0; i < numNodes; ++i) {
            double pos[3];
            this->GetNthNodeWorldPosition(i, pos);
            centroid[0] += pos[0];
            centroid[1] += pos[1];
            centroid[2] += pos[2];
        }

        centroid[0] /= numNodes;
        centroid[1] /= numNodes;
        centroid[2] /= numNodes;

        // Convert world coordinates to display coordinates
        this->Renderer->SetWorldPoint(centroid[0], centroid[1], centroid[2],
                                      1.0);
        this->Renderer->WorldToDisplay();
        double* displayCoord = this->Renderer->GetDisplayPoint();

        // Position label slightly above the centroid
        this->LabelActor->SetPosition(displayCoord[0], displayCoord[1] + 20);
    } else {
        // No nodes yet - position at default location
        this->LabelActor->SetPosition(50, 50);
    }
}
