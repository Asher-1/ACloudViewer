// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Control Layer: cvConstrainedDistanceWidget
// Based on ParaView: VTK/Interaction/Widgets/vtkLineWidget2.{h,cxx}
//
// Layer hierarchy:
// cvDistanceTool (Qt object) - Presentation Layer
//     ↓ owns
// cvConstrainedDistanceWidget - Control Layer (this class)
//     ↓ owns WidgetRep pointer
// vtkLineRepresentation - Representation Layer
//     ↓ owns (member variables)
// ├─ Point1Representation (vtkHandleRepresentation) - Data Layer
// ├─ Point2Representation (vtkHandleRepresentation) - Data Layer
// └─ LineHandleRepresentation (vtkHandleRepresentation) - Data Layer (middle
// handle)

#ifndef cvConstrainedDistanceWidget_h
#define cvConstrainedDistanceWidget_h

#include <vtkLineWidget2.h>

/**
 * @brief Line/Distance Widget with XYZL constraint support (100% consistent
 * with ParaView)
 *
 * Migrated from ParaView vtkLineWidget2, supports keyboard constraints:
 * - X: Constrain to X axis (VTK 9.2+)
 * - Y: Constrain to Y axis (VTK 9.2+)
 * - Z: Constrain to Z axis (VTK 9.2+)
 * - L: Constrain to line direction (VTK 9.3+ automatically enabled)
 *
 * Releasing the key automatically removes constraints
 *
 * @note Comparison with ParaView vtkLineWidget2:
 *       1. Fully inherits from vtkLineWidget2 (ParaView native widget)
 *       2. Includes LineHandle support (middle handle)
 *       3. 100% copy of ParaView ProcessKeyEvents code
 *
 * @note L key functionality automatically enabled/disabled based on VTK
 * version:
 *       - VTK >= 9.3: Full L key support
 *       - VTK <  9.3: L key has no effect
 */
class cvConstrainedDistanceWidget : public vtkLineWidget2 {
public:
    static cvConstrainedDistanceWidget* New();
    vtkTypeMacro(cvConstrainedDistanceWidget, vtkLineWidget2);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * @brief Set interactor - override to register keyboard events
     * Copied from ParaView vtkLineWidget2
     */
    void SetInteractor(vtkRenderWindowInteractor* iren) override;

    /**
     * @brief Override SetEnabled to manage keyboard event observers
     * We need to remove the parent class's keyboard handler to avoid conflicts
     */
    void SetEnabled(int enabling) override;

protected:
    cvConstrainedDistanceWidget();
    ~cvConstrainedDistanceWidget() override;

    /**
     * @brief Keyboard event handler - directly copied from ParaView
     * vtkLineWidget2::ProcessKeyEvents
     *
     * This is the core method of the control layer:
     * 1. Receive keyboard events
     * 2. Get handles (data layer) through representation (middle layer)
     * 3. Set constraint state of handles
     */
    static void ProcessKeyEvents(vtkObject* object,
                                 unsigned long event,
                                 void* clientdata,
                                 void* calldata);

private:
    cvConstrainedDistanceWidget(const cvConstrainedDistanceWidget&) = delete;
    void operator=(const cvConstrainedDistanceWidget&) = delete;

    // Keyboard observer IDs and callback command
    unsigned long KeyPressObserverId;
    unsigned long KeyReleaseObserverId;
    vtkCallbackCommand* KeyboardCallbackCommand;
};

#endif  // cvConstrainedDistanceWidget_h
