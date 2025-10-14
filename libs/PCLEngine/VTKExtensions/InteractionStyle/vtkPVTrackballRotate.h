// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkPVTrackballRotate_h
#define vtkPVTrackballRotate_h

#include "qPCL.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class QPCL_ENGINE_LIB_API vtkPVTrackballRotate : public vtkCameraManipulator {
public:
    static vtkPVTrackballRotate* New();
    vtkTypeMacro(vtkPVTrackballRotate, vtkCameraManipulator);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    void OnMouseMove(int x,
                     int y,
                     vtkRenderer* ren,
                     vtkRenderWindowInteractor* rwi) override;
    void OnButtonDown(int x,
                      int y,
                      vtkRenderer* ren,
                      vtkRenderWindowInteractor* rwi) override;
    void OnButtonUp(int x,
                    int y,
                    vtkRenderer* ren,
                    vtkRenderWindowInteractor* rwi) override;
    //@}

    //@{
    /**
     * These methods are called on all registered manipulators, not just the
     * active one. Hence, these should just be used to record state and not
     * perform any interactions.
     * Overridden to capture if the x,y,z key is pressed.
     */
    void OnKeyUp(vtkRenderWindowInteractor* iren) override;
    void OnKeyDown(vtkRenderWindowInteractor* iren) override;
    //@}

    /**
     * Returns the currently pressed key code.
     */
    vtkGetMacro(KeyCode, char);

protected:
    vtkPVTrackballRotate();
    ~vtkPVTrackballRotate() override;

    char KeyCode;
    vtkPVTrackballRotate(const vtkPVTrackballRotate&) = delete;
    void operator=(const vtkPVTrackballRotate&) = delete;
};

#endif
