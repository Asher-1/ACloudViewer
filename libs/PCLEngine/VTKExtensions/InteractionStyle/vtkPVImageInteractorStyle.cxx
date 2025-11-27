// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkPVImageInteractorStyle.h"

#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkImageData.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkPropCollection.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTransform.h>

vtkStandardNewMacro(vtkPVImageInteractorStyle);

//-------------------------------------------------------------------------
vtkPVImageInteractorStyle::vtkPVImageInteractorStyle() {
    this->RotationFactor = 1.0;
}

//-------------------------------------------------------------------------
vtkPVImageInteractorStyle::~vtkPVImageInteractorStyle() = default;

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
    os << indent << "RotationFactor: " << this->RotationFactor << "\n";
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::OnLeftButtonDown() {
    // Left button: Pan (translate)
    // Don't call base class to avoid conflicts
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    this->FindPokedRenderer(x, y);
    if (!this->CurrentRenderer) {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartPan();
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::OnLeftButtonUp() {
    // End pan
    if (this->Interactor) {
        if (this->State == VTKIS_PAN) {
            this->EndPan();
        }
        this->ReleaseFocus();
    }
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::OnMiddleButtonDown() {
    // Middle button: Rotate around Z-axis (perpendicular to image plane)
    // Don't call base class to avoid conflicts
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    this->FindPokedRenderer(x, y);
    if (!this->CurrentRenderer) {
        return;
    }

    this->GrabFocus(this->EventCallbackCommand);
    this->StartRotate();
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::OnMiddleButtonUp() {
    // End rotate
    if (this->Interactor) {
        if (this->State == VTKIS_ROTATE) {
            this->EndRotate();
        }
        this->ReleaseFocus();
    }
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::OnMouseMove() {
    if (this->Interactor) {
        if (this->State == VTKIS_PAN) {
            // Pan: translate camera
            this->Pan();
        } else if (this->State == VTKIS_ROTATE) {
            // Rotate: rotate around Z-axis (perpendicular to image plane)
            this->Rotate();
        } else {
            vtkInteractorStyleImage::OnMouseMove();
        }
    }
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::Rotate() {
    if (this->CurrentRenderer == nullptr || this->Interactor == nullptr) {
        return;
    }

    vtkRenderWindowInteractor* rwi = this->Interactor;
    vtkCamera* camera = this->CurrentRenderer->GetActiveCamera();

    // Compute display center for rotation calculation
    int* size = this->CurrentRenderer->GetSize();
    double displayCenter[2];
    displayCenter[0] = size[0] / 2.0;
    displayCenter[1] = size[1] / 2.0;

    // Calculate vectors from center to mouse positions
    int x1 =
            rwi->GetLastEventPosition()[0] - static_cast<int>(displayCenter[0]);
    int x2 = rwi->GetEventPosition()[0] - static_cast<int>(displayCenter[0]);
    int y1 =
            rwi->GetLastEventPosition()[1] - static_cast<int>(displayCenter[1]);
    int y2 = rwi->GetEventPosition()[1] - static_cast<int>(displayCenter[1]);

    // Check for zero vectors to avoid division by zero
    if ((x2 == 0 && y2 == 0) || (x1 == 0 && y1 == 0)) {
        return;
    }

    // Calculate rotation angle using cross product (similar to
    // vtkPVTrackballRoll) This allows continuous rotation
    double len1 = sqrt(static_cast<double>(x1 * x1 + y1 * y1));
    double len2 = sqrt(static_cast<double>(x2 * x2 + y2 * y2));

    if (len1 == 0.0 || len2 == 0.0) {
        return;
    }

    // Calculate angle using cross product: sin(angle) = (x1*y2 - y1*x2) / (|v1|
    // * |v2|) Reduce rotation speed by multiplying with a factor (0.1 for
    // smoother rotation)
    double angle =
            vtkMath::DegreesFromRadians((x1 * y2 - y1 * x2) / (len1 * len2)) *
            0.1 * this->MotionFactor * this->RotationFactor;

    // Compute rotation axis (view direction for 2D images)
    double axis[3];
    double* pos = camera->GetPosition();
    double* fp = camera->GetFocalPoint();
    axis[0] = fp[0] - pos[0];
    axis[1] = fp[1] - pos[1];
    axis[2] = fp[2] - pos[2];
    vtkMath::Normalize(axis);

    // Calculate image center dynamically from image slice bounds
    // This ensures rotation center is always the image center, even after
    // panning
    double center[3] = {0.0, 0.0, 0.0};
    bool centerFound = false;

    // Find image slice and get its geometric center
    vtkPropCollection* props = this->CurrentRenderer->GetViewProps();
    props->InitTraversal();
    vtkProp* prop = nullptr;
    while ((prop = props->GetNextProp()) != nullptr) {
        vtkImageSlice* imageSlice = vtkImageSlice::SafeDownCast(prop);
        if (imageSlice && imageSlice->GetVisibility()) {
            // Get bounds of the image slice in world coordinates
            double bounds[6];
            imageSlice->GetBounds(bounds);

            // Check if bounds are valid and non-degenerate
            if (bounds[0] < bounds[1] && bounds[2] < bounds[3]) {
                // Calculate geometric center from bounds
                center[0] = (bounds[0] + bounds[1]) * 0.5;
                center[1] = (bounds[2] + bounds[3]) * 0.5;
                center[2] = (bounds[4] + bounds[5]) * 0.5;
                centerFound = true;
                break;  // Use first valid image slice
            }
        }
    }

    // If no image slice found, use camera focal point as fallback
    if (!centerFound) {
        camera->GetFocalPoint(center);
    }

    // Apply rotation using transform (allows continuous rotation)
    vtkTransform* transform = vtkTransform::New();
    transform->Identity();
    transform->Translate(center[0], center[1], center[2]);
    transform->RotateWXYZ(angle, axis[0], axis[1], axis[2]);
    transform->Translate(-center[0], -center[1], -center[2]);

    camera->ApplyTransform(transform);
    camera->OrthogonalizeViewUp();
    transform->Delete();

    if (this->AutoAdjustCameraClippingRange) {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    if (rwi->GetLightFollowCamera()) {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}

//-------------------------------------------------------------------------
void vtkPVImageInteractorStyle::Pan() {
    if (this->CurrentRenderer == nullptr || this->Interactor == nullptr) {
        return;
    }

    vtkRenderWindowInteractor* rwi = this->Interactor;
    vtkCamera* camera = this->CurrentRenderer->GetActiveCamera();

    // For parallel projection (2D images), use optimized panning
    if (camera->GetParallelProjection()) {
        camera->OrthogonalizeViewUp();
        double up[3], vpn[3], right[3];
        camera->GetViewUp(up);
        camera->GetViewPlaneNormal(vpn);
        vtkMath::Cross(vpn, up, right);

        int* size = this->CurrentRenderer->GetSize();
        int dx = rwi->GetEventPosition()[0] - rwi->GetLastEventPosition()[0];
        int dy = rwi->GetLastEventPosition()[1] -
                 rwi->GetEventPosition()[1];  // Y is flipped

        double scale = camera->GetParallelScale();
        double panX = (double)dx / (double)size[1] * scale * 2.0;
        double panY = (double)dy / (double)size[1] * scale * 2.0;

        double pos[3], fp[3];
        camera->GetPosition(pos);
        camera->GetFocalPoint(fp);

        double tmp;
        tmp = (right[0] * panX + up[0] * panY);
        pos[0] += tmp;
        fp[0] += tmp;
        tmp = (right[1] * panX + up[1] * panY);
        pos[1] += tmp;
        fp[1] += tmp;
        tmp = (right[2] * panX + up[2] * panY);
        pos[2] += tmp;
        fp[2] += tmp;

        camera->SetPosition(pos);
        camera->SetFocalPoint(fp);
    } else {
        // For perspective projection, use standard panning
        double focalPoint[4], pickPoint[4], prevPickPoint[4];
        double z;

        camera->GetFocalPoint(focalPoint);
        focalPoint[3] = 1.0;

        this->ComputeWorldToDisplay(this->CurrentRenderer, focalPoint[0],
                                    focalPoint[1], focalPoint[2], focalPoint);
        z = focalPoint[2];

        this->ComputeDisplayToWorld(
                this->CurrentRenderer, rwi->GetLastEventPosition()[0],
                rwi->GetLastEventPosition()[1], z, prevPickPoint);

        this->ComputeDisplayToWorld(this->CurrentRenderer,
                                    rwi->GetEventPosition()[0],
                                    rwi->GetEventPosition()[1], z, pickPoint);

        // Camera motion is reversed
        camera->SetFocalPoint(
                focalPoint[0] - (pickPoint[0] - prevPickPoint[0]),
                focalPoint[1] - (pickPoint[1] - prevPickPoint[1]),
                focalPoint[2] - (pickPoint[2] - prevPickPoint[2]));

        camera->SetPosition(
                camera->GetPosition()[0] - (pickPoint[0] - prevPickPoint[0]),
                camera->GetPosition()[1] - (pickPoint[1] - prevPickPoint[1]),
                camera->GetPosition()[2] - (pickPoint[2] - prevPickPoint[2]));
    }

    if (this->AutoAdjustCameraClippingRange) {
        this->CurrentRenderer->ResetCameraClippingRange();
    }

    if (rwi->GetLightFollowCamera()) {
        this->CurrentRenderer->UpdateLightsGeometryToFollowCamera();
    }

    rwi->Render();
}
