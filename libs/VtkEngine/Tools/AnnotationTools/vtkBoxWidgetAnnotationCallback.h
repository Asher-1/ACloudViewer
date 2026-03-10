// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkBoxWidgetAnnotationCallback.h
 * @brief VTK command callbacks for annotation box widget interaction.
 */

#include <vtkCommand.h>

class Annotation;

/**
 * @class vtkBoxWidgetCallback0
 * @brief VTK command callback for annotation box widget (first event).
 */
class vtkBoxWidgetCallback0 : public vtkCommand {
public:
    static vtkBoxWidgetCallback0 *New();
    /// @param caller Caller object (vtkBoxWidget).
    virtual void Execute(vtkObject *caller, unsigned long, void *);

    /**
     * @brief setAnno set the current annotation in which the actor is picked
     * @param value Annotation to associate with this callback.
     */
    void setAnno(Annotation *value);

private:
    Annotation *anno;
};

/**
 * @class vtkBoxWidgetCallback1
 * @brief VTK command callback for annotation box widget (second event).
 */
class vtkBoxWidgetCallback1 : public vtkCommand {
public:
    static vtkBoxWidgetCallback1 *New();
    /// @param caller Caller object (vtkBoxWidget).
    virtual void Execute(vtkObject *caller, unsigned long, void *);

    /**
     * @brief setAnno set the current annotation in which the actor is picked
     * @param value Annotation to associate with this callback.
     */
    void setAnno(Annotation *value);

private:
    Annotation *anno;
};
