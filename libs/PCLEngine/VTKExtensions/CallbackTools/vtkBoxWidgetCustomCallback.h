// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkBoxWidgetCallback_H
#define vtkBoxWidgetCallback_H

#include <vtkCommand.h>
#include <vtkSmartPointer.h>

#include "qPCL.h"

class vtkActor;
class vtkObject;

class QPCL_ENGINE_LIB_API vtkBoxWidgetCustomCallback : public vtkCommand {
public:
    static vtkBoxWidgetCustomCallback *New();
    virtual void Execute(vtkObject *caller, unsigned long, void *);

    /**
     * @brief SetActor set the current vtkActor in which the actor is picked
     * @param value
     */
    void SetActor(vtkSmartPointer<vtkActor> actor);

    inline void EnablePreview(bool enabled) { m_preview = enabled; }

private:
    bool m_preview = true;
    vtkBoxWidgetCustomCallback() {}
    vtkSmartPointer<vtkActor> m_actor;
};

#endif
