// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkPVTrackballPan_h
#define vtkPVTrackballPan_h

#include "qPCL.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class QPCL_ENGINE_LIB_API vtkPVTrackballPan : public vtkCameraManipulator {
public:
    static vtkPVTrackballPan* New();
    vtkTypeMacro(vtkPVTrackballPan, vtkCameraManipulator);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    void OnMouseMove(int x,
                     int y,
                     vtkRenderer* ren,
                     vtkRenderWindowInteractor* iren) override;
    void OnButtonDown(int x,
                      int y,
                      vtkRenderer* ren,
                      vtkRenderWindowInteractor* iren) override;
    void OnButtonUp(int x,
                    int y,
                    vtkRenderer* ren,
                    vtkRenderWindowInteractor* iren) override;
    //@}

protected:
    vtkPVTrackballPan();
    ~vtkPVTrackballPan() override;

    vtkPVTrackballPan(const vtkPVTrackballPan&) = delete;
    void operator=(const vtkPVTrackballPan&) = delete;
};

#endif
