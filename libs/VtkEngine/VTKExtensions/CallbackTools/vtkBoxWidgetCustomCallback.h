// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkBoxWidgetCustomCallback.h
 * @brief VTK command callback for box widget interaction events
 */

#include <vtkCommand.h>
#include <vtkSmartPointer.h>

#include "qVTK.h"

class vtkActor;
class vtkObject;

/**
 * @class vtkBoxWidgetCustomCallback
 * @brief VTK command callback invoked when a box widget is manipulated
 *
 * Applies the box widget's transform to a target actor when the widget
 * is interacted with. Used for interactive box cutting/filtering.
 */
class QVTK_ENGINE_LIB_API vtkBoxWidgetCustomCallback : public vtkCommand {
public:
    static vtkBoxWidgetCustomCallback *New();
    /**
     * @brief Execute callback when box widget fires an event
     * @param caller The vtkBoxWidget that invoked this callback
     * @param eventId VTK event identifier (unused)
     * @param callData Event-specific data (unused)
     */
    virtual void Execute(vtkObject *caller,
                         unsigned long eventId,
                         void *callData);

    /**
     * @brief Set the actor to apply the box widget transform to
     * @param actor Target actor for the transform
     */
    void SetActor(vtkSmartPointer<vtkActor> actor);

    /**
     * @brief Enable or disable preview mode during interaction
     * @param enabled True to enable preview, false to disable
     */
    inline void EnablePreview(bool enabled) { m_preview = enabled; }

private:
    bool m_preview = true;
    vtkBoxWidgetCustomCallback() {}
    vtkSmartPointer<vtkActor> m_actor;
};
