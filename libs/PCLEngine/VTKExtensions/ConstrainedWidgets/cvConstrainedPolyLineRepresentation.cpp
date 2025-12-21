// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvConstrainedPolyLineRepresentation.h"

#include <vtkActor2D.h>
#include <vtkCellArray.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkPropCollection.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkWindow.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

vtkStandardNewMacro(cvConstrainedPolyLineRepresentation);

cvConstrainedPolyLineRepresentation::cvConstrainedPolyLineRepresentation()
    : ShowAngleLabel(1),
      ShowAngleArc(1),
      ArcRadius(1.0),
      Angle(0.0),
      LabelSuffix(nullptr) {
    // Initialize angle label actor with default properties
    // These will be overridden by cvProtractorTool::applyTextPropertiesToLabel()
    this->AngleLabelActor = vtkTextActor::New();
    this->AngleLabelActor->SetTextScaleModeToNone();
    this->AngleLabelActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);  // White
    this->AngleLabelActor->GetTextProperty()->SetFontSize(
            20);  // Default font size for angle display
    this->AngleLabelActor->GetTextProperty()->SetBold(0);  // Not bold by default
    this->AngleLabelActor->GetTextProperty()->SetItalic(0);  // Not italic
    this->AngleLabelActor->GetTextProperty()->SetShadow(1);  // Shadow for visibility
    this->AngleLabelActor->GetTextProperty()->SetFontFamilyToArial();
    this->AngleLabelActor->SetVisibility(1);

    // Initialize angle arc geometry
    this->AngleArcPolyData = vtkPolyData::New();
    this->AngleArcMapper = vtkPolyDataMapper2D::New();
    this->AngleArcMapper->SetInputData(this->AngleArcPolyData);

    this->AngleArcActor = vtkActor2D::New();
    this->AngleArcActor->SetMapper(this->AngleArcMapper);
    this->AngleArcActor->GetProperty()->SetColor(1.0, 1.0, 0.0);  // Yellow arc
    this->AngleArcActor->GetProperty()->SetLineWidth(2.0);
    this->AngleArcActor->SetVisibility(1);
}

cvConstrainedPolyLineRepresentation::~cvConstrainedPolyLineRepresentation() {
    // CRITICAL: Remove actors from renderer before deleting them
    if (this->Renderer) {
        if (this->AngleLabelActor) {
            this->Renderer->RemoveActor2D(this->AngleLabelActor);
        }
        if (this->AngleArcActor) {
            this->Renderer->RemoveActor2D(this->AngleArcActor);
        }
    }

    if (this->LabelSuffix) {
        delete[] this->LabelSuffix;
        this->LabelSuffix = nullptr;
    }
    if (this->AngleLabelActor) {
        this->AngleLabelActor->Delete();
        this->AngleLabelActor = nullptr;
    }
    if (this->AngleArcActor) {
        this->AngleArcActor->Delete();
        this->AngleArcActor = nullptr;
    }
    if (this->AngleArcMapper) {
        this->AngleArcMapper->Delete();
        this->AngleArcMapper = nullptr;
    }
    if (this->AngleArcPolyData) {
        this->AngleArcPolyData->Delete();
        this->AngleArcPolyData = nullptr;
    }
}

void cvConstrainedPolyLineRepresentation::SetRenderer(vtkRenderer* ren) {
    // Remove actors from old renderer if present
    if (this->Renderer && this->Renderer != ren) {
        if (this->AngleLabelActor) {
            this->Renderer->RemoveActor2D(this->AngleLabelActor);
        }
        if (this->AngleArcActor) {
            this->Renderer->RemoveActor2D(this->AngleArcActor);
        }
    }

    // Call parent to set the renderer
    this->Superclass::SetRenderer(ren);

    // Add our custom 2D actors to the new renderer
    if (ren) {
        if (this->AngleLabelActor) {
            ren->AddActor2D(this->AngleLabelActor);
        }
        if (this->AngleArcActor) {
            ren->AddActor2D(this->AngleArcActor);
        }
    }
}

void cvConstrainedPolyLineRepresentation::BuildRepresentation() {
    // Call parent class to build the poly line
    this->Superclass::BuildRepresentation();

    // Only proceed if we have exactly 3 handles (angle measurement
    // configuration)
    if (this->GetNumberOfHandles() != 3) {
        return;
    }

    // Calculate and update angle
    this->Angle = this->GetAngle();

    // Update angle label text (following ParaView format)
    std::ostringstream labelStream;
    labelStream.precision(2);
    labelStream << std::fixed << this->Angle << "°";

    // Append instance label suffix if present (e.g., " #1", " #2")
    if (this->LabelSuffix) {
        labelStream << this->LabelSuffix;
    }

    this->AngleLabelActor->SetInput(labelStream.str().c_str());

    // DO NOT set text properties here!
    // Text properties should be set by cvProtractorTool::applyTextPropertiesToLabel()
    // after BuildRepresentation() is called.
    // Setting properties here would override user-configured settings.

    // Position label at the arc's center (bisector of the angle)
    // ALWAYS place label on the acute angle side (< 180 degrees)
    if (this->Renderer) {
        double p1[3], center[3], p2[3];
        this->GetHandlePosition(0, p1);
        this->GetHandlePosition(1, center);
        this->GetHandlePosition(2, p2);

        // Calculate vectors from center to points
        double vec1[3], vec2[3];
        vtkMath::Subtract(p1, center, vec1);
        vtkMath::Subtract(p2, center, vec2);
        double len1 = vtkMath::Norm(vec1);
        double len2 = vtkMath::Norm(vec2);
        vtkMath::Normalize(vec1);
        vtkMath::Normalize(vec2);

        // Calculate the bisector direction (average of normalized vectors)
        double bisector[3];
        bisector[0] = vec1[0] + vec2[0];
        bisector[1] = vec1[1] + vec2[1];
        bisector[2] = vec1[2] + vec2[2];
        vtkMath::Normalize(bisector);

        // Check if the angle is > 180 degrees
        // If so, flip the bisector to point to the acute angle side
        double dotProduct = vtkMath::Dot(vec1, vec2);
        double angleRadians =
                std::acos(std::max(-1.0, std::min(1.0, dotProduct)));
        double angleDegrees = vtkMath::DegreesFromRadians(angleRadians);

        // If angle > 180 degrees, the bisector points to the obtuse side
        // We need to flip it to point to the acute side
        if (angleDegrees > 180.0) {
            bisector[0] = -bisector[0];
            bisector[1] = -bisector[1];
            bisector[2] = -bisector[2];
        }

        // Calculate adaptive arc radius (MUST match BuildAngleArc logic)
        double minRayLength = std::min(len1, len2);
        double radiusPercentage;

        if (angleDegrees < 30.0) {
            // Small angle: larger radius for better visibility
            radiusPercentage = 0.30 - (angleDegrees / 30.0) * 0.05;
        } else if (angleDegrees < 90.0) {
            // Medium angle
            radiusPercentage = 0.25 - ((angleDegrees - 30.0) / 60.0) * 0.05;
        } else {
            // Large angle: smaller radius
            radiusPercentage = 0.20 - ((angleDegrees - 90.0) / 90.0) * 0.05;
        }

        double adaptiveRadius =
                std::min(this->ArcRadius, minRayLength * radiusPercentage);
        adaptiveRadius = std::max(adaptiveRadius, minRayLength * 0.01);
        adaptiveRadius = std::min(adaptiveRadius, minRayLength * 0.40);

        // Adaptive label distance: further away for small angles to avoid ray
        // occlusion For small angles (< 30°): use 2.5x arc radius to clear the
        // tight rays For medium angles (30° - 90°): use 2.0x arc radius For
        // large angles (> 90°): use 1.5x arc radius (rays are well separated)
        double labelDistanceMultiplier;
        if (angleDegrees < 30.0) {
            // Small angle: need more distance to avoid ray occlusion
            // Scale from 3.0x at 0° to 2.5x at 30°
            labelDistanceMultiplier = 3.0 - (angleDegrees / 30.0) * 0.5;
        } else if (angleDegrees < 90.0) {
            // Medium angle: moderate distance
            // Scale from 2.5x at 30° to 1.8x at 90°
            labelDistanceMultiplier =
                    2.5 - ((angleDegrees - 30.0) / 60.0) * 0.7;
        } else {
            // Large angle: rays are well separated, can be closer
            // Scale from 1.8x at 90° to 1.5x at 180°
            labelDistanceMultiplier =
                    1.8 - ((angleDegrees - 90.0) / 90.0) * 0.3;
        }

        // Position label along the bisector at adaptive distance
        // This ensures label is always clearly visible and not occluded by rays
        double labelDistance = adaptiveRadius * labelDistanceMultiplier;
        double labelPos[3];
        labelPos[0] = center[0] + bisector[0] * labelDistance;
        labelPos[1] = center[1] + bisector[1] * labelDistance;
        labelPos[2] = center[2] + bisector[2] * labelDistance;

        // Convert world coordinates to display coordinates
        this->Renderer->SetWorldPoint(labelPos[0], labelPos[1], labelPos[2],
                                      1.0);
        this->Renderer->WorldToDisplay();
        double* displayCoord = this->Renderer->GetDisplayPoint();

        // Calculate text width to center the label
        // Get text bounding box in display coordinates
        double bounds[4];  // [xmin, xmax, ymin, ymax]
        this->AngleLabelActor->GetBoundingBox(this->Renderer, bounds);

        // Calculate text width
        double textWidth = bounds[1] - bounds[0];

        // Center the text horizontally and add small vertical offset
        // Note: vtkTextActor's position is at the bottom-left of the text
        // Subtract half the text width to center it on the display coordinate
        double centeredX = displayCoord[0] - textWidth / 2.0;
        double centeredY =
                displayCoord[1] + 5;  // Small offset for visual separation

        this->AngleLabelActor->SetPosition(centeredX, centeredY);
    }

    // Build angle arc
    if (this->ShowAngleArc) {
        this->BuildAngleArc();
    }

    // Control visibility (respect both individual flags and representation
    // visibility)
    vtkTypeBool repVisible = this->GetVisibility();
    this->AngleLabelActor->SetVisibility(repVisible && this->ShowAngleLabel);
    this->AngleArcActor->SetVisibility(repVisible && this->ShowAngleArc);
}

double cvConstrainedPolyLineRepresentation::GetAngle() {
    // Following ParaView pqAnglePropertyWidget::updateLabels() (lines 155-161)
    if (this->GetNumberOfHandles() < 3) {
        return 0.0;
    }

    // Get the three points
    double p1[3], center[3], p2[3];
    this->GetHandlePosition(0, p1);      // Point 1
    this->GetHandlePosition(1, center);  // Center (vertex)
    this->GetHandlePosition(2, p2);      // Point 2

    // Calculate vectors from center
    double vec1[3], vec2[3];
    vtkMath::Subtract(p1, center, vec1);
    vtkMath::Subtract(p2, center, vec2);

    // Calculate angle using dot product
    // angle = acos(vec1·vec2 / (|vec1| * |vec2|))
    double norm1 = vtkMath::Norm(vec1);
    double norm2 = vtkMath::Norm(vec2);

    if (norm1 < 1e-10 || norm2 < 1e-10) {
        return 0.0;  // Avoid division by zero
    }

    double dotProduct = vtkMath::Dot(vec1, vec2);
    double cosAngle = dotProduct / (norm1 * norm2);

    // Clamp to avoid numerical errors with acos
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle));

    // Convert from radians to degrees (ParaView uses
    // vtkMath::DegreesFromRadians)
    double angleRadians = std::acos(cosAngle);
    return vtkMath::DegreesFromRadians(angleRadians);
}

void cvConstrainedPolyLineRepresentation::BuildAngleArc() {
    if (this->GetNumberOfHandles() < 3 || !this->Renderer) {
        return;
    }

    // Get the three points
    double p1[3], center[3], p2[3];
    this->GetHandlePosition(0, p1);
    this->GetHandlePosition(1, center);
    this->GetHandlePosition(2, p2);

    // Calculate vectors
    double vec1[3], vec2[3];
    vtkMath::Subtract(p1, center, vec1);
    vtkMath::Subtract(p2, center, vec2);
    double len1 = vtkMath::Norm(vec1);
    double len2 = vtkMath::Norm(vec2);
    vtkMath::Normalize(vec1);
    vtkMath::Normalize(vec2);

    // Calculate the angle between rays
    double dotProduct = vtkMath::Dot(vec1, vec2);
    double angleRadians = std::acos(std::max(-1.0, std::min(1.0, dotProduct)));
    double angleDegrees = vtkMath::DegreesFromRadians(angleRadians);

    // Adaptive arc radius based on angle size (VTK protractor style)
    // Small angles: larger radius for better visibility
    // Large angles: smaller radius to stay close to vertex
    double minRayLength = std::min(len1, len2);
    double radiusPercentage;

    if (angleDegrees < 30.0) {
        // Small angle (0° - 30°): use 25-30% of ray length
        // Larger radius makes small angles more visible
        radiusPercentage = 0.30 - (angleDegrees / 30.0) * 0.05;
    } else if (angleDegrees < 90.0) {
        // Medium angle (30° - 90°): use 20-25% of ray length
        radiusPercentage = 0.25 - ((angleDegrees - 30.0) / 60.0) * 0.05;
    } else {
        // Large angle (90° - 180°): use 15-20% of ray length
        // Smaller radius keeps arc near vertex when rays are far apart
        radiusPercentage = 0.20 - ((angleDegrees - 90.0) / 90.0) * 0.05;
    }

    double adaptiveRadius =
            std::min(this->ArcRadius, minRayLength * radiusPercentage);

    // Ensure radius is not too small (at least 1% of ray length)
    // and not too large (at most 40% for very small angles)
    adaptiveRadius = std::max(adaptiveRadius, minRayLength * 0.01);
    adaptiveRadius = std::min(adaptiveRadius, minRayLength * 0.40);

    // Create arc geometry in display coordinates
    vtkNew<vtkPoints> arcPoints;
    vtkNew<vtkCellArray> arcLines;

    const int numArcPoints = 30;  // More points for smoother arc

    for (int i = 0; i <= numArcPoints; ++i) {
        double t = static_cast<double>(i) / numArcPoints;

        // Interpolate between vec1 and vec2 using spherical interpolation
        double cosTheta = vtkMath::Dot(vec1, vec2);
        double theta = std::acos(std::max(-1.0, std::min(1.0, cosTheta)));

        double sinTheta = std::sin(theta);
        double a = (sinTheta > 1e-10) ? std::sin((1.0 - t) * theta) / sinTheta
                                      : 1.0 - t;
        double b = (sinTheta > 1e-10) ? std::sin(t * theta) / sinTheta : t;

        double interpVec[3];
        for (int j = 0; j < 3; ++j) {
            interpVec[j] = a * vec1[j] + b * vec2[j];
        }
        vtkMath::Normalize(interpVec);

        // Scale by adaptive arc radius and offset from center (world
        // coordinates)
        double arcPointWorld[3];
        for (int j = 0; j < 3; ++j) {
            arcPointWorld[j] = center[j] + adaptiveRadius * interpVec[j];
        }

        // Convert world coordinates to display coordinates
        this->Renderer->SetWorldPoint(arcPointWorld[0], arcPointWorld[1],
                                      arcPointWorld[2], 1.0);
        this->Renderer->WorldToDisplay();
        double* displayCoord = this->Renderer->GetDisplayPoint();

        // Insert as 2D point (Z=0 for 2D actor)
        arcPoints->InsertNextPoint(displayCoord[0], displayCoord[1], 0.0);
    }

    // Create polyline
    arcLines->InsertNextCell(numArcPoints + 1);
    for (int i = 0; i <= numArcPoints; ++i) {
        arcLines->InsertCellPoint(i);
    }

    this->AngleArcPolyData->SetPoints(arcPoints);
    this->AngleArcPolyData->SetLines(arcLines);
    this->AngleArcPolyData->Modified();
}

void cvConstrainedPolyLineRepresentation::SetVisibility(vtkTypeBool visible) {
    // Call parent to set visibility of line and handles
    this->Superclass::SetVisibility(visible);

    // CRITICAL: Also control visibility of our custom 2D actors
    // Force hide regardless of individual flags when representation is hidden
    if (this->AngleLabelActor) {
        if (visible) {
            this->AngleLabelActor->SetVisibility(this->ShowAngleLabel);
        } else {
            this->AngleLabelActor->SetVisibility(0);
        }
    }
    if (this->AngleArcActor) {
        if (visible) {
            this->AngleArcActor->SetVisibility(this->ShowAngleArc);
        } else {
            this->AngleArcActor->SetVisibility(0);
        }
    }
}

void cvConstrainedPolyLineRepresentation::GetActors2D(vtkPropCollection* pc) {
    // Call parent to get its 2D actors
    this->Superclass::GetActors2D(pc);

    // Add our custom 2D actors so VTK knows about them
    if (this->AngleLabelActor) {
        pc->AddItem(this->AngleLabelActor);
    }
    if (this->AngleArcActor) {
        pc->AddItem(this->AngleArcActor);
    }
}

void cvConstrainedPolyLineRepresentation::ReleaseGraphicsResources(
        vtkWindow* w) {
    this->Superclass::ReleaseGraphicsResources(w);
    if (this->AngleLabelActor) {
        this->AngleLabelActor->ReleaseGraphicsResources(w);
    }
    if (this->AngleArcActor) {
        this->AngleArcActor->ReleaseGraphicsResources(w);
    }
}

int cvConstrainedPolyLineRepresentation::RenderOverlay(vtkViewport* viewport) {
    int count = this->Superclass::RenderOverlay(viewport);

    if (this->ShowAngleLabel && this->AngleLabelActor) {
        count += this->AngleLabelActor->RenderOverlay(viewport);
    }
    if (this->ShowAngleArc && this->AngleArcActor) {
        count += this->AngleArcActor->RenderOverlay(viewport);
    }

    return count;
}

int cvConstrainedPolyLineRepresentation::RenderOpaqueGeometry(
        vtkViewport* viewport) {
    int count = this->Superclass::RenderOpaqueGeometry(viewport);

    if (this->ShowAngleArc && this->AngleArcActor) {
        count += this->AngleArcActor->RenderOpaqueGeometry(viewport);
    }

    return count;
}

int cvConstrainedPolyLineRepresentation::RenderTranslucentPolygonalGeometry(
        vtkViewport* viewport) {
    int count = this->Superclass::RenderTranslucentPolygonalGeometry(viewport);
    return count;
}

vtkTypeBool
cvConstrainedPolyLineRepresentation::HasTranslucentPolygonalGeometry() {
    return this->Superclass::HasTranslucentPolygonalGeometry();
}

// Compatibility API implementation for Display coordinates
void cvConstrainedPolyLineRepresentation::SetPoint1DisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 0) {
        double worldPos[3];
        this->Renderer->SetDisplayPoint(pos);
        this->Renderer->DisplayToWorld();
        this->Renderer->GetWorldPoint(worldPos);
        // Normalize homogeneous coordinate
        if (worldPos[3] != 0.0) {
            worldPos[0] /= worldPos[3];
            worldPos[1] /= worldPos[3];
            worldPos[2] /= worldPos[3];
        }
        this->SetHandlePosition(0, worldPos);
    }
}

void cvConstrainedPolyLineRepresentation::GetPoint1DisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 0) {
        double worldPos[3];
        this->GetHandlePosition(0, worldPos);
        this->Renderer->SetWorldPoint(worldPos[0], worldPos[1], worldPos[2],
                                      1.0);
        this->Renderer->WorldToDisplay();
        this->Renderer->GetDisplayPoint(pos);
    } else {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvConstrainedPolyLineRepresentation::SetCenterDisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 1) {
        double worldPos[3];
        this->Renderer->SetDisplayPoint(pos);
        this->Renderer->DisplayToWorld();
        this->Renderer->GetWorldPoint(worldPos);
        if (worldPos[3] != 0.0) {
            worldPos[0] /= worldPos[3];
            worldPos[1] /= worldPos[3];
            worldPos[2] /= worldPos[3];
        }
        this->SetHandlePosition(1, worldPos);
    }
}

void cvConstrainedPolyLineRepresentation::GetCenterDisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 1) {
        double worldPos[3];
        this->GetHandlePosition(1, worldPos);
        this->Renderer->SetWorldPoint(worldPos[0], worldPos[1], worldPos[2],
                                      1.0);
        this->Renderer->WorldToDisplay();
        this->Renderer->GetDisplayPoint(pos);
    } else {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void cvConstrainedPolyLineRepresentation::SetPoint2DisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 2) {
        double worldPos[3];
        this->Renderer->SetDisplayPoint(pos);
        this->Renderer->DisplayToWorld();
        this->Renderer->GetWorldPoint(worldPos);
        if (worldPos[3] != 0.0) {
            worldPos[0] /= worldPos[3];
            worldPos[1] /= worldPos[3];
            worldPos[2] /= worldPos[3];
        }
        this->SetHandlePosition(2, worldPos);
    }
}

void cvConstrainedPolyLineRepresentation::GetPoint2DisplayPosition(
        double pos[3]) {
    if (this->Renderer && this->GetNumberOfHandles() > 2) {
        double worldPos[3];
        this->GetHandlePosition(2, worldPos);
        this->Renderer->SetWorldPoint(worldPos[0], worldPos[1], worldPos[2],
                                      1.0);
        this->Renderer->WorldToDisplay();
        this->Renderer->GetDisplayPoint(pos);
    } else {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

// Compatibility: Get handle representations
vtkHandleRepresentation*
cvConstrainedPolyLineRepresentation::GetPoint1Representation() {
    // vtkCurveRepresentation doesn't expose GetHandleRepresentation, so we
    // return nullptr The handles are managed internally by
    // vtkPolyLineRepresentation
    return nullptr;
}

vtkHandleRepresentation*
cvConstrainedPolyLineRepresentation::GetCenterRepresentation() {
    return nullptr;
}

vtkHandleRepresentation*
cvConstrainedPolyLineRepresentation::GetPoint2Representation() {
    return nullptr;
}
