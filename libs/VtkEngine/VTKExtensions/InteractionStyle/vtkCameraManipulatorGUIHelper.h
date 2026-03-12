// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkCameraManipulatorGUIHelper.h
 * @brief Abstract interface for GUI updates from camera manipulators.
 */

#include "qVTK.h"  // needed for export macro
#include "vtkObject.h"

/**
 * @class vtkCameraManipulatorGUIHelper
 * @brief Abstract base for GUI helpers used by camera manipulators.
 */
class QVTK_ENGINE_LIB_API vtkCameraManipulatorGUIHelper : public vtkObject {
public:
    vtkTypeMacro(vtkCameraManipulatorGUIHelper, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Called by the manipulator to update the GUI.
     * This typically involves calling processing pending
     * events on the GUI.
     */
    virtual void UpdateGUI() = 0;

    /**
     * Some interactors use the bounds of the active source.
     * The method returns 0 if no active source is present or
     * not supported by GUI, otherwise returns 1 and the bounds
     * are filled into the passed argument array.
     * @param bounds Output array [xmin,xmax,ymin,ymax,zmin,zmax]
     * @return 1 on success, 0 if not applicable
     */
    virtual int GetActiveSourceBounds(double bounds[6]) = 0;

    //@{
    /**
     * Called to get/set the translation for the actor for the active
     * source in the active view. If applicable returns 1, otherwise
     * returns 0.
     * @param translate Output/input translation [x,y,z]
     * @return 1 on success, 0 if not applicable
     */
    virtual int GetActiveActorTranslate(double translate[3]) = 0;
    virtual int SetActiveActorTranslate(double translate[3]) = 0;
    //@}

    //@{
    /**
     * Get the center of rotation. Returns 0 if not applicable.
     * @param center Output array [x,y,z]
     * @return 1 on success, 0 if not applicable
     */
    virtual int GetCenterOfRotation(double center[3]) = 0;

protected:
    vtkCameraManipulatorGUIHelper();
    ~vtkCameraManipulatorGUIHelper() override;
    //@}

private:
    vtkCameraManipulatorGUIHelper(const vtkCameraManipulatorGUIHelper&) =
            delete;
    void operator=(const vtkCameraManipulatorGUIHelper&) = delete;
};
