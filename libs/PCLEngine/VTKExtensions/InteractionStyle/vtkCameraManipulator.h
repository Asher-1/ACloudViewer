// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkCameraManipulator_h
#define vtkCameraManipulator_h

#include "qPCL.h"  // needed for export macro
#include "vtkObject.h"

class vtkCameraManipulatorGUIHelper;
class vtkRenderer;
class vtkRenderWindowInteractor;

class QPCL_ENGINE_LIB_API vtkCameraManipulator : public vtkObject {
public:
    static vtkCameraManipulator* New();
    vtkTypeMacro(vtkCameraManipulator, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Event bindings controlling the effects of pressing mouse buttons
     * or moving the mouse.
     */
    virtual void StartInteraction();
    virtual void EndInteraction();
    //@}

    virtual void OnMouseMove(int x,
                             int y,
                             vtkRenderer* ren,
                             vtkRenderWindowInteractor* iren);
    virtual void OnButtonDown(int x,
                              int y,
                              vtkRenderer* ren,
                              vtkRenderWindowInteractor* iren);
    virtual void OnButtonUp(int x,
                            int y,
                            vtkRenderer* ren,
                            vtkRenderWindowInteractor* iren);

    //@{
    /**
     * These methods are called on all registered manipulators, not just the
     * active one. Hence, these should just be used to record state and not
     * perform any interactions.
     */
    virtual void OnKeyUp(vtkRenderWindowInteractor* iren);
    virtual void OnKeyDown(vtkRenderWindowInteractor* iren);
    //@}

    //@{
    /**
     * These settings determine which button and modifiers the
     * manipulator responds to. Button can be either 1 (left), 2
     * (middle), and 3 right.
     */
    vtkSetMacro(Button, int);
    vtkGetMacro(Button, int);
    vtkSetMacro(Shift, int);
    vtkGetMacro(Shift, int);
    vtkBooleanMacro(Shift, int);
    vtkSetMacro(Control, int);
    vtkGetMacro(Control, int);
    vtkBooleanMacro(Control, int);
    //@}

    //@{
    /**
     * For setting the center of rotation.
     */
    vtkSetVector3Macro(Center, double);
    vtkGetVector3Macro(Center, double);
    //@}

    //@{
    /**
     * Set and get the rotation factor.
     */
    vtkSetMacro(RotationFactor, double);
    vtkGetMacro(RotationFactor, double);
    //@}

    //@{
    /**
     * Set and get the manipulator name.
     */
    vtkSetStringMacro(ManipulatorName);
    vtkGetStringMacro(ManipulatorName);
    //@}

    //@{
    /**
     * Get/Set the GUI helper.
     */
    void SetGUIHelper(vtkCameraManipulatorGUIHelper*);
    vtkGetObjectMacro(GUIHelper, vtkCameraManipulatorGUIHelper);

protected:
    vtkCameraManipulator();
    ~vtkCameraManipulator() override;
    //@}

    char* ManipulatorName;

    int Button;
    int Shift;
    int Control;

    double Center[3];
    double RotationFactor;
    double DisplayCenter[2];
    void ComputeDisplayCenter(vtkRenderer* ren);

    vtkCameraManipulatorGUIHelper* GUIHelper;

private:
    vtkCameraManipulator(const vtkCameraManipulator&) = delete;
    void operator=(const vtkCameraManipulator&) = delete;
};

#endif
