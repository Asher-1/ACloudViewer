// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvInteractorStyleDrawPolygon.h"

#include <vtkCommand.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <cmath>

// Manual implementation of New() with export macro
// vtkStandardNewMacro doesn't work well with QPCL_ENGINE_LIB_API
cvInteractorStyleDrawPolygon* cvInteractorStyleDrawPolygon::New() {
    VTK_STANDARD_NEW_BODY(cvInteractorStyleDrawPolygon);
}

//------------------------------------------------------------------------------
class cvInteractorStyleDrawPolygon::vtkInternal {
public:
    std::vector<vtkVector2i> points;

    void AddPoint(const vtkVector2i& point) { this->points.push_back(point); }

    void AddPoint(int x, int y) { this->AddPoint(vtkVector2i(x, y)); }

    vtkVector2i GetPoint(vtkIdType index) const { return this->points[index]; }

    vtkIdType GetNumberOfPoints() const {
        return static_cast<vtkIdType>(this->points.size());
    }

    void Clear() { this->points.clear(); }

    void DrawPixels(const vtkVector2i& StartPos,
                    const vtkVector2i& EndPos,
                    unsigned char* pixels,
                    const int* size) {
        int x1 = StartPos.GetX(), x2 = EndPos.GetX();
        int y1 = StartPos.GetY(), y2 = EndPos.GetY();

        double x = x2 - x1;
        double y = y2 - y1;
        double length = sqrt(x * x + y * y);
        if (length == 0) {
            return;
        }
        double addx = x / length;
        double addy = y / length;

        x = x1;
        y = y1;
        int row, col;
        for (double i = 0; i < length; i += 1) {
            col = static_cast<int>(x);
            row = static_cast<int>(y);

            // Bounds checking
            if (col >= 0 && col < size[0] && row >= 0 && row < size[1]) {
                int idx = 3 * (row * size[0] + col);
                pixels[idx] = 255 ^ pixels[idx];
                pixels[idx + 1] = 255 ^ pixels[idx + 1];
                pixels[idx + 2] = 255 ^ pixels[idx + 2];
            }
            x += addx;
            y += addy;
        }
    }
};

//------------------------------------------------------------------------------
cvInteractorStyleDrawPolygon::cvInteractorStyleDrawPolygon() {
    this->Internal = new vtkInternal();
    this->StartPosition[0] = this->StartPosition[1] = 0;
    this->EndPosition[0] = this->EndPosition[1] = 0;
    this->Moving = 0;
    this->DrawPolygonPixels = true;
    this->MinimumPointDistanceSquared = 100;  // 10 pixels minimum distance
    this->PixelArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
}

//------------------------------------------------------------------------------
cvInteractorStyleDrawPolygon::~cvInteractorStyleDrawPolygon() {
    delete this->Internal;
}

//------------------------------------------------------------------------------
std::vector<vtkVector2i> cvInteractorStyleDrawPolygon::GetPolygonPoints() {
    return this->Internal->points;
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::ClearPolygonPoints() {
    this->Internal->Clear();
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::OnMouseMove() {
    if (!this->Interactor || !this->Moving) {
        return;
    }

    // Get current position
    this->EndPosition[0] = this->Interactor->GetEventPosition()[0];
    this->EndPosition[1] = this->Interactor->GetEventPosition()[1];

    // Clamp to window bounds
    const int* size = this->Interactor->GetRenderWindow()->GetSize();
    this->EndPosition[0] = std::min(this->EndPosition[0], size[0] - 1);
    this->EndPosition[0] = std::max(this->EndPosition[0], 0);
    this->EndPosition[1] = std::min(this->EndPosition[1], size[1] - 1);
    this->EndPosition[1] = std::max(this->EndPosition[1], 0);

    vtkVector2i newPoint(this->EndPosition[0], this->EndPosition[1]);

    // Check if we have any points yet
    if (this->Internal->GetNumberOfPoints() > 0) {
        vtkVector2i lastPoint = this->Internal->GetPoint(
                this->Internal->GetNumberOfPoints() - 1);

        // Calculate squared distance manually (vtkVector2i doesn't have
        // operator-)
        int dx = newPoint.GetX() - lastPoint.GetX();
        int dy = newPoint.GetY() - lastPoint.GetY();
        int squaredDist = dx * dx + dy * dy;

        // Add new point if distance threshold is met (or threshold is 0)
        if (this->MinimumPointDistanceSquared <= 0 ||
            squaredDist > this->MinimumPointDistanceSquared) {
            this->Internal->AddPoint(newPoint);
        }
    }

    // FIX: Always redraw the polygon on every mouse move
    // This is the key fix for the flickering issue - VTK's implementation
    // only redraws when a new point is added, which causes the polygon to
    // disappear when moving slowly or when stopped.
    if (this->DrawPolygonPixels) {
        this->DrawPolygon();
    }
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::OnLeftButtonDown() {
    if (!this->Interactor) {
        return;
    }
    this->Moving = 1;

    vtkRenderWindow* renWin = this->Interactor->GetRenderWindow();
    if (!renWin) {
        return;
    }

    // Store starting position
    this->StartPosition[0] = this->Interactor->GetEventPosition()[0];
    this->StartPosition[1] = this->Interactor->GetEventPosition()[1];
    this->EndPosition[0] = this->StartPosition[0];
    this->EndPosition[1] = this->StartPosition[1];

    // Capture current frame buffer pixels
    // This is used as the base for XOR drawing
    this->PixelArray->Initialize();
    this->PixelArray->SetNumberOfComponents(3);
    const int* size = renWin->GetSize();
    this->PixelArray->SetNumberOfTuples(size[0] * size[1]);

    // Get current pixels from the front buffer
    renWin->GetPixelData(0, 0, size[0] - 1, size[1] - 1, 1, this->PixelArray);

    // Clear any previous polygon points and start fresh
    this->Internal->Clear();
    this->Internal->AddPoint(this->StartPosition[0], this->StartPosition[1]);

    this->InvokeEvent(vtkCommand::StartInteractionEvent);
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::OnLeftButtonUp() {
    if (!this->Interactor || !this->Moving) {
        return;
    }

    // Restore original pixels (remove polygon drawing)
    if (this->DrawPolygonPixels) {
        this->RestorePixels();
    }

    this->Moving = 0;

    // Fire the SelectionChangedEvent - this signals that polygon selection
    // is complete and the polygon points can be retrieved via
    // GetPolygonPoints()
    this->InvokeEvent(vtkCommand::SelectionChangedEvent);
    this->InvokeEvent(vtkCommand::EndInteractionEvent);
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::DrawPolygon() {
    if (!this->Interactor) {
        return;
    }

    vtkRenderWindow* renWin = this->Interactor->GetRenderWindow();
    if (!renWin) {
        return;
    }

    const int* size = renWin->GetSize();
    vtkIdType numPoints = this->Internal->GetNumberOfPoints();

    if (numPoints < 1) {
        return;
    }

    // Create a temporary copy of the original pixels
    vtkNew<vtkUnsignedCharArray> tmpPixelArray;
    tmpPixelArray->DeepCopy(this->PixelArray);
    unsigned char* pixels = tmpPixelArray->GetPointer(0);

    // Draw each line segment of the polygon
    for (vtkIdType i = 0; i < numPoints - 1; i++) {
        const vtkVector2i& a = this->Internal->GetPoint(i);
        const vtkVector2i& b = this->Internal->GetPoint(i + 1);
        this->Internal->DrawPixels(a, b, pixels, size);
    }

    // Draw a line from the last point to the current mouse position
    // This creates a "rubber band" effect showing where the polygon would close
    if (numPoints >= 1) {
        const vtkVector2i& lastPoint = this->Internal->GetPoint(numPoints - 1);
        vtkVector2i currentPos(this->EndPosition[0], this->EndPosition[1]);
        this->Internal->DrawPixels(lastPoint, currentPos, pixels, size);
    }

    // Draw a line from the current position back to the start to show
    // the closing edge of the polygon (if we have at least 2 points)
    if (numPoints >= 2) {
        const vtkVector2i& start = this->Internal->GetPoint(0);
        vtkVector2i currentPos(this->EndPosition[0], this->EndPosition[1]);
        this->Internal->DrawPixels(currentPos, start, pixels, size);
    }

    // Update the render window with the modified pixels
    renWin->SetPixelData(0, 0, size[0] - 1, size[1] - 1, pixels, 0);
    renWin->Frame();
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::RestorePixels() {
    if (!this->Interactor) {
        return;
    }

    vtkRenderWindow* renWin = this->Interactor->GetRenderWindow();
    if (!renWin) {
        return;
    }

    const int* size = renWin->GetSize();
    unsigned char* pixels = this->PixelArray->GetPointer(0);

    // Restore the original pixels
    renWin->SetPixelData(0, 0, size[0] - 1, size[1] - 1, pixels, 0);
    renWin->Frame();
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::DrawLine(const vtkVector2i& start,
                                            const vtkVector2i& end,
                                            unsigned char* pixels,
                                            const int* size) {
    this->Internal->DrawPixels(start, end, pixels, size);
}

//------------------------------------------------------------------------------
void cvInteractorStyleDrawPolygon::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
    os << indent << "Moving: " << this->Moving << endl;
    os << indent << "DrawPolygonPixels: " << this->DrawPolygonPixels << endl;
    os << indent
       << "MinimumPointDistanceSquared: " << this->MinimumPointDistanceSquared
       << endl;
    os << indent << "StartPosition: " << this->StartPosition[0] << ","
       << this->StartPosition[1] << endl;
    os << indent << "EndPosition: " << this->EndPosition[0] << ","
       << this->EndPosition[1] << endl;
    os << indent
       << "NumberOfPolygonPoints: " << this->Internal->GetNumberOfPoints()
       << endl;
}
