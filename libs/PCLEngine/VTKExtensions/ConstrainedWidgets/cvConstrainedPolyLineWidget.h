// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_CONSTRAINED_POLYLINE_WIDGET_H
#define CV_CONSTRAINED_POLYLINE_WIDGET_H

#include <vtkPolyLineWidget.h>

class vtkCallbackCommand;

/**
 * @brief PolyLine Widget with XYZ constraints (100% consistent with ParaView)
 *
 * Ported from ParaView vtkPolyLineWidget, supports keyboard constraints:
 * - X: Constrain to X-axis
 * - Y: Constrain to Y-axis
 * - Z: Constrain to Z-axis
 *
 * Note: This widget is used for angle measurement (3 handles: point1, center,
 * point2)
 *
 * Releasing the key automatically removes the constraint.
 *
 * @note 100% replication of ParaView vtkPolyLineWidget ProcessKeyEvents
 */
class cvConstrainedPolyLineWidget : public vtkPolyLineWidget {
public:
    static cvConstrainedPolyLineWidget* New();
    vtkTypeMacro(cvConstrainedPolyLineWidget, vtkPolyLineWidget);

    /**
     * @brief Override SetInteractor to register keyboard event observers
     *
     * Following ParaView's pattern from vtkPolyLineWidget::SetEnabled
     */
    void SetInteractor(vtkRenderWindowInteractor* iren) override;

    /**
     * @brief Override SetEnabled to register our custom keyboard handler
     *
     * This is where we replace the parent's KeyEventCallbackCommand with our
     * own
     */
    void SetEnabled(int enabling) override;

protected:
    cvConstrainedPolyLineWidget();
    ~cvConstrainedPolyLineWidget() override;

    /**
     * @brief Keyboard event handling - directly copied from ParaView
     * vtkPolyLineWidget::ProcessKeyEvents
     *
     * This is the core method of the control layer:
     * 1. Receives keyboard events
     * 2. Gets the representation
     * 3. Sets the constraint status on the entire representation
     *
     * ParaView's implementation (vtkPolyLineWidget.cxx:272-301):
     * - X key: Constrain all handles to X-axis
     * - Y key: Constrain all handles to Y-axis
     * - Z key: Constrain all handles to Z-axis
     * - Key release: Remove constraints
     */
    static void ProcessKeyEvents(vtkObject* object,
                                 unsigned long event,
                                 void* clientdata,
                                 void* calldata);

    vtkCallbackCommand* KeyEventCallbackCommand;

private:
    cvConstrainedPolyLineWidget(const cvConstrainedPolyLineWidget&) = delete;
    void operator=(const cvConstrainedPolyLineWidget&) = delete;
};

#endif  // CV_CONSTRAINED_POLYLINE_WIDGET_H
