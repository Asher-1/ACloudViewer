// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef cvConstrainedLineRepresentation_h
#define cvConstrainedLineRepresentation_h

#include <vtkLineRepresentation.h>
#include <vtkPointHandleRepresentation3D.h>

class vtkAxisActor2D;
class vtkProperty2D;

/**
 * @brief Extended LineRepresentation with distance display and ruler features
 *
 * Combines the advantages of both:
 * - vtkLineRepresentation: Supports LineHandle, XYZL constraints
 * - vtkDistanceRepresentation: Distance display, ticks, Ruler mode
 *
 * This allows us to have simultaneously:
 * 1. XYZ/L key constraint functionality
 * 2. Distance value display
 * 3. Ruler tick display
 * 4. UI synchronization
 */
class cvConstrainedLineRepresentation : public vtkLineRepresentation {
public:
    static cvConstrainedLineRepresentation* New();
    vtkTypeMacro(cvConstrainedLineRepresentation, vtkLineRepresentation);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * @brief Get distance between the two points
     */
    virtual double GetDistance();

    /**
     * @brief Set/Get Ruler mode
     * Ruler mode displays tick marks
     */
    vtkSetMacro(RulerMode, vtkTypeBool);
    vtkGetMacro(RulerMode, vtkTypeBool);
    vtkBooleanMacro(RulerMode, vtkTypeBool);

    /**
     * @brief Set/Get Ruler distance (tick spacing)
     */
    vtkSetClampMacro(RulerDistance, double, 0, VTK_FLOAT_MAX);
    vtkGetMacro(RulerDistance, double);

    /**
     * @brief Set/Get number of ruler ticks
     */
    vtkSetClampMacro(NumberOfRulerTicks, int, 1, VTK_INT_MAX);
    vtkGetMacro(NumberOfRulerTicks, int);

    /**
     * @brief Set/Get scale factor
     */
    void SetScale(double scale);
    vtkGetMacro(Scale, double);

    /**
     * @brief Set/Get distance label format
     */
    vtkSetStringMacro(LabelFormat);
    vtkGetStringMacro(LabelFormat);

    /**
     * @brief Set/Get distance label suffix (e.g., " #1", " #2")
     */
    vtkSetStringMacro(LabelSuffix);
    vtkGetStringMacro(LabelSuffix);

    /**
     * @brief Show/Hide distance label
     */
    vtkSetMacro(ShowLabel, vtkTypeBool);
    vtkGetMacro(ShowLabel, vtkTypeBool);
    vtkBooleanMacro(ShowLabel, vtkTypeBool);

    /**
     * @brief Get Axis Actor (for tick display)
     */
    vtkGetObjectMacro(AxisActor, vtkAxisActor2D);

    /**
     * @brief Get axis property (ParaView way)
     */
    vtkGetObjectMacro(AxisProperty, vtkProperty2D);

    /**
     * @brief Override BuildRepresentation to update distance display
     */
    void BuildRepresentation() override;

    /**
     * @brief Override to render distance label and ticks
     */
    int RenderOverlay(vtkViewport* viewport) override;
    int RenderOpaqueGeometry(vtkViewport* viewport) override;

    /**
     * @brief Replace handle representations with custom types
     * This is needed because vtkLineRepresentation's constructor calls
     * InstantiateHandleRepresentation() before we can set custom handles
     *
     * Template version for type-safe creation
     */
    template <typename T>
    void ReplaceHandleRepresentationsTyped() {
        // Save current positions
        double p1[3] = {0, 0, 0};
        double p2[3] = {0, 0, 0};
        if (this->Point1Representation) {
            this->Point1Representation->GetWorldPosition(p1);
        }
        if (this->Point2Representation) {
            this->Point2Representation->GetWorldPosition(p2);
        }

        // Create a prototype for copying attributes
        T* prototype = T::New();

        // Replace Point1Representation
        if (this->Point1Representation) {
            this->Point1Representation->Delete();
        }
        this->Point1Representation = T::New();
        this->Point1Representation->ShallowCopy(prototype);
        this->Point1Representation->SetWorldPosition(p1);

        // Replace Point2Representation
        if (this->Point2Representation) {
            this->Point2Representation->Delete();
        }
        this->Point2Representation = T::New();
        this->Point2Representation->ShallowCopy(prototype);
        this->Point2Representation->SetWorldPosition(p2);

        // Replace LineHandleRepresentation
        if (this->LineHandleRepresentation) {
            this->LineHandleRepresentation->Delete();
        }
        this->LineHandleRepresentation = T::New();
        this->LineHandleRepresentation->ShallowCopy(prototype);

        prototype->Delete();
        this->Modified();
    }

    /**
     * @brief Replace handle representations with custom types (runtime version)
     */
    void ReplaceHandleRepresentations(vtkPointHandleRepresentation3D* handle);

protected:
    cvConstrainedLineRepresentation();
    ~cvConstrainedLineRepresentation() override;

    // Distance label related (following ParaView vtkDistanceRepresentation2D)
    vtkTypeBool ShowLabel;
    char* LabelFormat;
    char* LabelSuffix;  // Instance label suffix (e.g., " #1", " #2")

    // Ruler and axis related (following ParaView)
    vtkTypeBool RulerMode;
    double RulerDistance;
    int NumberOfRulerTicks;
    vtkAxisActor2D* AxisActor;
    vtkProperty2D* AxisProperty;
    double Scale;
    double Distance;

private:
    cvConstrainedLineRepresentation(const cvConstrainedLineRepresentation&) =
            delete;
    void operator=(const cvConstrainedLineRepresentation&) = delete;
};

#endif  // cvConstrainedLineRepresentation_h
