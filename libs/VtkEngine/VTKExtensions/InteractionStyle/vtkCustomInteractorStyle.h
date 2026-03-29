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
 * Provides camera manipulation via mouse (left=rotate, middle=pan, right=zoom)
 * and the following keyboard shortcuts:
 *
 * View Controls:
 * -  r, R              : reset camera
 * -  r, R + ALT        : reset to viewpoint origin
 * -  f, F              : fly to picked point
 * -  Ctrl+Alt + O      : toggle perspective/parallel projection
 * -  Ctrl+Alt + F      : toggle maximize/restore window
 * -  Ctrl+Alt + +/-    : zoom in/out
 *
 * Display Controls:
 * -  Ctrl+Shift + P    : point representation
 * -  Ctrl+Shift + W    : wireframe representation
 * -  Ctrl+Shift + S    : surface representation
 * -  Ctrl+Shift + +/-  : increase/decrease point size
 * -  Ctrl+Alt + G      : toggle scale grid
 * -  Ctrl+Alt + K      : toggle lookup table
 *
 * Camera:
 * -  Ctrl + S          : save camera parameters
 * -  Ctrl + R          : restore camera parameters
 * -  Ctrl+Alt + C      : print camera parameters
 * -  Ctrl+Alt + J      : take screenshot (.PNG)
 *
 * Other:
 * -  Ctrl+Alt + S      : toggle stereo mode
 * -  a, A              : toggle rubber band selection
 * -  e, E              : exit the interactor
 * -  q, Q              : quit (TerminateApp)
 * -  h, H              : show help
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

    /// @brief Directly execute a keyboard shortcut from Qt key event data.
    /// Bypasses VTK's key event system for reliable cross-platform operation.
    /// @param key Lowercase letter ('j','c','g',...) or symbol ('+','-','=')
    /// @param ctrl Control modifier state
    /// @param alt Alt modifier state
    /// @param shift Shift modifier state
    /// @param iren Optional interactor override (used when this style is
    ///             detached from the active interactor, e.g. during selection)
    /// @return true if the shortcut was handled
    bool handleShortcut(char key,
                        bool ctrl,
                        bool alt,
                        bool shift,
                        vtkRenderWindowInteractor* iren = nullptr);

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
