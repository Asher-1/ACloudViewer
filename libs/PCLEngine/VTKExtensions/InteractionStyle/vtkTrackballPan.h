// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkTrackballPan_h
#define vtkTrackballPan_h

#include "qPCL.h"  // needed for export macro
#include "vtkCameraManipulator.h"

class QPCL_ENGINE_LIB_API vtkTrackballPan : public vtkCameraManipulator {
public:
    static vtkTrackballPan* New();
    vtkTypeMacro(vtkTrackballPan, vtkCameraManipulator);
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
    vtkTrackballPan();
    ~vtkTrackballPan() override;

    vtkTrackballPan(const vtkTrackballPan&) = delete;
    void operator=(const vtkTrackballPan&) = delete;
};

#endif
