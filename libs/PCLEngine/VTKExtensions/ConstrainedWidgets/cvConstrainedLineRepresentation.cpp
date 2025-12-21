// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvConstrainedLineRepresentation.h"

// VTK
#include <vtkActor.h>
#include <vtkAxisActor2D.h>
#include <vtkCoordinate.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPointHandleRepresentation3D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkTextProperty.h>
#include <vtkWindow.h>

// STL
#include <cstdio>
#include <sstream>

vtkStandardNewMacro(cvConstrainedLineRepresentation);

//------------------------------------------------------------------------------
cvConstrainedLineRepresentation::cvConstrainedLineRepresentation()
    : ShowLabel(1),
      LabelFormat(nullptr),
      LabelSuffix(nullptr),
      RulerMode(0),
      RulerDistance(1.0),
      NumberOfRulerTicks(5),
      Scale(1.0),
      Distance(0.0) {
    // Initialize AxisProperty (following ParaView vtkDistanceRepresentation2D
    // exactly)
    this->AxisProperty = vtkProperty2D::New();
    this->AxisProperty->SetColor(1.0, 1.0, 1.0);  // White color

    // Initialize AxisActor (following ParaView vtkDistanceRepresentation2D)
    this->AxisActor = vtkAxisActor2D::New();

    // CRITICAL: Use World coordinate system (ParaView way), NOT Display
    this->AxisActor->GetPoint1Coordinate()->SetCoordinateSystemToWorld();
    this->AxisActor->GetPoint2Coordinate()->SetCoordinateSystemToWorld();

    this->AxisActor->SetNumberOfLabels(5);
    this->AxisActor->LabelVisibilityOff();  // ParaView default
    this->AxisActor->AdjustLabelsOff();
    this->AxisActor->SetProperty(this->AxisProperty);
    this->AxisActor->SetTitle(
            "Distance");  // ParaView uses Title for distance label

    // CRITICAL: Hide the axis line itself to avoid double-line rendering
    // We only want to show the title (distance label), not the axis line
    // The actual line is already rendered by vtkLineRepresentation
    this->AxisActor->AxisVisibilityOff();  // Hide axis line
    this->AxisActor->TickVisibilityOff();  // Hide tick marks

    // Configure title text property (compact appearance)
    this->AxisActor->GetTitleTextProperty()->SetFontSize(8);  // Default font size (will be overridden by user settings)
    this->AxisActor->GetTitleTextProperty()->SetBold(0);      // Not bold
    this->AxisActor->GetTitleTextProperty()->SetItalic(0);    // Not italic
    this->AxisActor->GetTitleTextProperty()->SetShadow(1);
    this->AxisActor->GetTitleTextProperty()->SetFontFamilyToArial();
    this->AxisActor->GetTitleTextProperty()->SetColor(1.0, 1.0, 1.0);

    // Use vtkAxisActor2D's font scaling mechanism (affects both title and labels)
    // Set font factors once during initialization - these should not be changed
    // in BuildRepresentation() as they would interfere with user font size settings
    this->AxisActor->SetFontFactor(1.0);   // Use 1.0 to respect user's font size settings
    this->AxisActor->SetLabelFactor(0.8);  // Slightly smaller for axis labels

    // Default label format (ParaView default)
    this->SetLabelFormat("%-#6.3g");
}

//------------------------------------------------------------------------------
cvConstrainedLineRepresentation::~cvConstrainedLineRepresentation() {
    // Following ParaView vtkDistanceRepresentation2D destructor
    if (this->AxisProperty) {
        this->AxisProperty->Delete();
    }
    if (this->AxisActor) {
        this->AxisActor->Delete();
    }
    if (this->LabelFormat) {
        delete[] this->LabelFormat;
    }
    if (this->LabelSuffix) {
        delete[] this->LabelSuffix;
    }
}

//------------------------------------------------------------------------------
double cvConstrainedLineRepresentation::GetDistance() {
    double p1[3], p2[3];
    this->GetPoint1WorldPosition(p1);
    this->GetPoint2WorldPosition(p2);
    return sqrt(vtkMath::Distance2BetweenPoints(p1, p2));
}

//------------------------------------------------------------------------------
void cvConstrainedLineRepresentation::SetScale(double scale) {
    if (this->Scale != scale && scale > 0.0) {
        this->Scale = scale;
        this->Modified();
    }
}

//------------------------------------------------------------------------------
void cvConstrainedLineRepresentation::BuildRepresentation() {
    // Call parent class BuildRepresentation
    this->Superclass::BuildRepresentation();

    // Following ParaView vtkDistanceRepresentation2D::BuildRepresentation()
    // exactly

    // Compute the distance and set the label
    double p1[3], p2[3];
    this->GetPoint1WorldPosition(p1);
    this->GetPoint2WorldPosition(p2);
    this->Distance = sqrt(vtkMath::Distance2BetweenPoints(p1, p2));

    // Set axis position in WORLD coordinates (ParaView way)
    this->AxisActor->GetPoint1Coordinate()->SetValue(p1);
    this->AxisActor->GetPoint2Coordinate()->SetValue(p2);

    // Configure ruler mode (ParaView way)
    this->AxisActor->SetRulerMode(this->RulerMode);
    if (this->Scale != 0.0) {
        this->AxisActor->SetRulerDistance(this->RulerDistance / this->Scale);
    }
    this->AxisActor->SetNumberOfLabels(this->NumberOfRulerTicks);

    // CRITICAL: Always hide the axis line to avoid double-line rendering
    // Only show ticks in RulerMode
    this->AxisActor->AxisVisibilityOff();
    if (this->RulerMode) {
        this->AxisActor->TickVisibilityOn();  // Show ticks only in ruler mode
    } else {
        this->AxisActor->TickVisibilityOff();  // Hide ticks in normal mode
    }

    // Format and set the distance label as AxisActor Title (ParaView way)
    if (this->LabelFormat) {
        char labelString[512];
        snprintf(labelString, sizeof(labelString), this->LabelFormat,
                 this->Distance * this->Scale);

        // Append instance label suffix if present (e.g., " #1", " #2")
        if (this->LabelSuffix) {
            strncat(labelString, this->LabelSuffix,
                    sizeof(labelString) - strlen(labelString) - 1);
        }

        this->AxisActor->SetTitle(labelString);
    }

    // NOTE: Font properties (size, bold, italic, shadow, color) and font factors 
    // should NOT be set here as they would override user customizations set via 
    // applyFontProperties(). Font properties are initialized in the constructor 
    // and can be customized by the user. BuildRepresentation() is called frequently 
    // (on every render), so setting font properties here would continuously reset 
    // user preferences.

    // Control visibility (ShowLabel controls the entire AxisActor visibility)
    this->AxisActor->SetVisibility(this->ShowLabel);
}

//------------------------------------------------------------------------------
int cvConstrainedLineRepresentation::RenderOverlay(vtkViewport* viewport) {
    // Following ParaView vtkDistanceRepresentation2D::RenderOverlay() exactly
    this->BuildRepresentation();

    int count = this->Superclass::RenderOverlay(viewport);

    if (this->AxisActor->GetVisibility()) {
        count += this->AxisActor->RenderOverlay(viewport);
    }

    return count;
}

//------------------------------------------------------------------------------
int cvConstrainedLineRepresentation::RenderOpaqueGeometry(
        vtkViewport* viewport) {
    // Following ParaView vtkDistanceRepresentation2D::RenderOpaqueGeometry()
    // exactly
    this->BuildRepresentation();

    int count = this->Superclass::RenderOpaqueGeometry(viewport);

    if (this->AxisActor->GetVisibility()) {
        count += this->AxisActor->RenderOpaqueGeometry(viewport);
    }

    return count;
}

//------------------------------------------------------------------------------
void cvConstrainedLineRepresentation::ReplaceHandleRepresentations(
        vtkPointHandleRepresentation3D* handle) {
    if (!handle) {
        return;
    }

    // Save current positions
    double p1[3] = {0, 0, 0};
    double p2[3] = {0, 0, 0};
    if (this->Point1Representation) {
        this->Point1Representation->GetWorldPosition(p1);
    }
    if (this->Point2Representation) {
        this->Point2Representation->GetWorldPosition(p2);
    }

    // Get the actual class name to check if it's our custom type
    const char* className = handle->GetClassName();
    bool isCustomType =
            (strcmp(className, "cvCustomAxisHandleRepresentation") == 0);

    // Replace Point1Representation
    if (this->Point1Representation) {
        this->Point1Representation->Delete();
    }
    if (isCustomType) {
        // Use NewInstance for our custom type - it should work correctly
        this->Point1Representation =
                static_cast<vtkPointHandleRepresentation3D*>(
                        handle->NewInstance());
    } else {
        // For standard types, create a copy
        this->Point1Representation = handle->NewInstance();
    }
    this->Point1Representation->ShallowCopy(handle);
    this->Point1Representation->SetWorldPosition(p1);

    // Replace Point2Representation
    if (this->Point2Representation) {
        this->Point2Representation->Delete();
    }
    if (isCustomType) {
        this->Point2Representation =
                static_cast<vtkPointHandleRepresentation3D*>(
                        handle->NewInstance());
    } else {
        this->Point2Representation = handle->NewInstance();
    }
    this->Point2Representation->ShallowCopy(handle);
    this->Point2Representation->SetWorldPosition(p2);

    // Replace LineHandleRepresentation
    if (this->LineHandleRepresentation) {
        this->LineHandleRepresentation->Delete();
    }
    if (isCustomType) {
        this->LineHandleRepresentation =
                static_cast<vtkPointHandleRepresentation3D*>(
                        handle->NewInstance());
    } else {
        this->LineHandleRepresentation = handle->NewInstance();
    }
    this->LineHandleRepresentation->ShallowCopy(handle);

    this->Modified();
}

//------------------------------------------------------------------------------
void cvConstrainedLineRepresentation::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);

    os << indent << "Show Label: " << this->ShowLabel << "\n";
    os << indent
       << "Label Format: " << (this->LabelFormat ? this->LabelFormat : "(none)")
       << "\n";
    os << indent << "Ruler Mode: " << this->RulerMode << "\n";
    os << indent << "Ruler Distance: " << this->RulerDistance << "\n";
    os << indent << "Number Of Ruler Ticks: " << this->NumberOfRulerTicks
       << "\n";
    os << indent << "Scale: " << this->Scale << "\n";
}
