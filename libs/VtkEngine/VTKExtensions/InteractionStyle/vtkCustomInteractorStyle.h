// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkCustomInteractorStyle.h
 * @brief Custom VTK interactor style for CloudViewer with keyboard shortcuts.
 */

#include "vtkInteractorStyleBase.h"

class vtkRendererCollection;
class vtkPointPicker;

class vtkCameraManipulator;
class vtkCollection;

namespace VTKExtensions {

/**
 * @class vtkCustomInteractorStyle
 * @brief Custom VTK interactor style for CloudViewer applications.
 *
 * Defines rendering style and custom actions triggered by key presses:
 *
 * -        p, P   : switch to a point-based representation
 * -        w, W   : switch to a wireframe-based representation (where
 * available)
 * -        s, S   : switch to a surface-based representation (where available)
 * -        j, J   : take a .PNG snapshot of the current window view
 * -        c, C   : display current camera/window parameters
 * -        f, F   : fly to point mode
 * -        e, E   : exit the interactor
 * -        q, Q   : stop and call VTK's TerminateApp
 * -       + / -   : increment/decrement overall point size
 * -        g, G   : display scale grid (on/off)
 * -        u, U   : display lookup table (on/off)
 * -  r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y,
 * z}]
 * -  CTRL + s, S  : save camera parameters
 * -  CTRL + r, R  : restore camera parameters
 * -  ALT + s, S   : turn stereo mode on/off
 * -  ALT + f, F   : switch between maximized window mode and original size
 * -  SHIFT + left click   : select a point
 * -        x, X   : toggle rubber band selection mode for left mouse button
 */
class QVTK_ENGINE_LIB_API vtkCustomInteractorStyle
    : public vtkInteractorStyleBase {
public:
    static vtkCustomInteractorStyle* New();
    vtkTypeMacro(vtkCustomInteractorStyle, vtkInteractorStyleBase);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    vtkCustomInteractorStyle();
    ~vtkCustomInteractorStyle() override;

    void toggleAreaPicking();

    /// @param win Render window to set
    inline void setRenderWindow(const vtkSmartPointer<vtkRenderWindow>& win) {
        win_ = win;
    }

public:
    /// @param m Camera manipulator to add
    void AddManipulator(vtkCameraManipulator* m);
    void RemoveAllManipulators();

    vtkGetObjectMacro(CameraManipulators, vtkCollection);

    vtkSetVector3Macro(CenterOfRotation, double);
    vtkGetVector3Macro(CenterOfRotation, double);

    vtkSetMacro(RotationFactor, double);
    vtkGetMacro(RotationFactor, double);

    /// @param button Mouse button (1=left, 2=middle, 3=right)
    /// @param shift Shift modifier state
    /// @param control Control modifier state
    /// @return Matching camera manipulator or nullptr
    virtual vtkCameraManipulator* FindManipulator(int button,
                                                  int shift,
                                                  int control);

    /// @param fact Dolly factor
    /// @param position Mouse position [x, y]
    /// @param renderer Target renderer
    static void DollyToPosition(double fact,
                                int* position,
                                vtkRenderer* renderer);

    /// @param renderer Target renderer
    /// @param toX Target X coordinate
    /// @param toY Target Y coordinate
    /// @param fromX Source X coordinate
    /// @param fromY Source Y coordinate
    static void TranslateCamera(
            vtkRenderer* renderer, int toX, int toY, int fromX, int fromY);

    using vtkInteractorStyleTrackballCamera::Dolly;

protected:
    void zoomIn();
    void zoomOut();

    void OnKeyDown() override;
    void OnKeyUp() override;
    void OnChar() override;

    void OnMouseMove() override;
    void OnLeftButtonDown() override;
    void OnLeftButtonUp() override;
    void OnMiddleButtonDown() override;
    void OnMiddleButtonUp() override;
    void OnRightButtonDown() override;
    void OnRightButtonUp() override;
    void OnMouseWheelForward() override;
    void OnMouseWheelBackward() override;

    void Dolly(double factor) override;

    std::string lut_actor_id_;

    void updateLookUpTableDisplay(bool add_lut = false);

    vtkCameraManipulator* CurrentManipulator;
    double CenterOfRotation[3];
    double RotationFactor;

    vtkCollection* CameraManipulators;

    void OnButtonDown(int button, int shift, int control);
    void OnButtonUp(int button);
    void ResetLights();

    vtkCustomInteractorStyle(const vtkCustomInteractorStyle&) = delete;
    void operator=(const vtkCustomInteractorStyle&) = delete;
};

}  // namespace VTKExtensions
