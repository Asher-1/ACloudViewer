// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkPVTrackballRoll_h
#define vtkPVTrackballRoll_h

#include "qPCL.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class QPCL_ENGINE_LIB_API vtkPVTrackballRoll : public vtkCameraManipulator {
public:
    static vtkPVTrackballRoll* New();
    vtkTypeMacro(vtkPVTrackballRoll, vtkCameraManipulator);
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

protected:
    vtkPVTrackballRoll();
    ~vtkPVTrackballRoll() override;

    vtkPVTrackballRoll(const vtkPVTrackballRoll&) = delete;
    void operator=(const vtkPVTrackballRoll&) = delete;
};

#endif
