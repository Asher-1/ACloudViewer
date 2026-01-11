// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_CONSTRAINED_POLYLINE_REPRESENTATION_H
#define CV_CONSTRAINED_POLYLINE_REPRESENTATION_H

#include <vtkPolyLineRepresentation.h>

class vtkTextActor;
class vtkActor2D;
class vtkPolyData;
class vtkPolyDataMapper2D;
class vtkHandleRepresentation;

/**
 * @brief Extended PolyLineRepresentation adding angle display functionality.
 *
 * This extends vtkPolyLineRepresentation to add angle visualization for 3-point
 * angle measurement (Point1-Center-Point2), following ParaView's approach.
 *
 * Key features:
 * 1. XYZ constraint support (inherited from vtkPolyLineRepresentation)
 * 2. Angle calculation and display (custom functionality)
 * 3. Arc visualization between the two rays (custom functionality)
 *
 * @note This is used with 3 handles configured for angle measurement:
 *       Handle 0: Point1
 *       Handle 1: Center (vertex of the angle)
 *       Handle 2: Point2
 */
class cvConstrainedPolyLineRepresentation : public vtkPolyLineRepresentation {
public:
    static cvConstrainedPolyLineRepresentation* New();
    vtkTypeMacro(cvConstrainedPolyLineRepresentation,
                 vtkPolyLineRepresentation);

    /**
     * @brief Build the representation for the angle measurement
     *
     * Extends parent's BuildRepresentation to add:
     * - Angle calculation from 3 points
     * - Arc visualization
     * - Angle label display
     */
    void BuildRepresentation() override;

    /**
     * @brief Set the renderer for this representation
     *
     * Override to ensure our custom 2D actors (angle label and arc) are
     * properly added to the renderer.
     */
    void SetRenderer(vtkRenderer* ren) override;

    /**
     * @brief Calculate the angle formed by the three handles
     *
     * Following ParaView pqAnglePropertyWidget::updateLabels() implementation:
     * angle = acos(vec1.dot(vec2) / (|vec1| * |vec2|))
     *
     * @return Angle in degrees
     */
    double GetAngle();

    ///@{
    /**
     * @brief Compatibility API for vtkAngleRepresentation
     *
     * These methods provide the same interface as vtkAngleRepresentation
     * for easier integration with existing code.
     *
     * Mapping:
     * - Point1 → Handle 0
     * - Center → Handle 1 (vertex of the angle)
     * - Point2 → Handle 2
     */
    void SetPoint1WorldPosition(double pos[3]) {
        this->SetHandlePosition(0, pos);
    }
    void GetPoint1WorldPosition(double pos[3]) {
        this->GetHandlePosition(0, pos);
    }
    void SetPoint1DisplayPosition(double pos[3]);
    void GetPoint1DisplayPosition(double pos[3]);

    void SetCenterWorldPosition(double pos[3]) {
        this->SetHandlePosition(1, pos);
    }
    void GetCenterWorldPosition(double pos[3]) {
        this->GetHandlePosition(1, pos);
    }
    void SetCenterDisplayPosition(double pos[3]);
    void GetCenterDisplayPosition(double pos[3]);

    void SetPoint2WorldPosition(double pos[3]) {
        this->SetHandlePosition(2, pos);
    }
    void GetPoint2WorldPosition(double pos[3]) {
        this->GetHandlePosition(2, pos);
    }
    void SetPoint2DisplayPosition(double pos[3]);
    void GetPoint2DisplayPosition(double pos[3]);

    /**
     * @brief Compatibility: Get handle representations
     *
     * These provide access to individual handle representations
     * for direct manipulation if needed.
     */
    vtkHandleRepresentation* GetPoint1Representation();
    vtkHandleRepresentation* GetCenterRepresentation();
    vtkHandleRepresentation* GetPoint2Representation();

    /**
     * @brief Compatibility: Instantiate handle representations
     *
     * This is called automatically when SetNumberOfHandles is called,
     * but we provide it for API compatibility.
     */
    void InstantiateHandleRepresentation() {
        // vtkPolyLineRepresentation automatically creates handles
        // when SetNumberOfHandles is called
    }

    /**
     * @brief Override SetVisibility to also control arc and label actors
     */
    void SetVisibility(vtkTypeBool visible) override;
    ///@}

    ///@{
    /**
     * @brief Control visibility of the angle label
     */
    vtkSetMacro(ShowAngleLabel, vtkTypeBool);
    vtkGetMacro(ShowAngleLabel, vtkTypeBool);
    vtkBooleanMacro(ShowAngleLabel, vtkTypeBool);
    ///@}

    ///@{
    /**
     * @brief Set/Get angle label suffix (e.g., " #1", " #2")
     */
    vtkSetStringMacro(LabelSuffix);
    vtkGetStringMacro(LabelSuffix);
    ///@}

    ///@{
    /**
     * @brief Control visibility of the angle arc
     */
    vtkSetMacro(ShowAngleArc, vtkTypeBool);
    vtkGetMacro(ShowAngleArc, vtkTypeBool);
    vtkBooleanMacro(ShowAngleArc, vtkTypeBool);
    ///@}

    ///@{
    /**
     * @brief Set/Get the radius of the angle arc
     */
    vtkSetClampMacro(ArcRadius, double, 0.0, VTK_DOUBLE_MAX);
    vtkGetMacro(ArcRadius, double);
    ///@}

    /**
     * @brief Get the angle label actor for custom styling
     */
    vtkGetObjectMacro(AngleLabelActor, vtkTextActor);

    /**
     * @brief Get the angle arc actor for custom styling
     */
    vtkGetObjectMacro(AngleArcActor, vtkActor2D);

    /**
     * @brief Release graphics resources
     */
    void ReleaseGraphicsResources(vtkWindow* w) override;

    /**
     * @brief Render the angle representation
     */
    void GetActors2D(vtkPropCollection* pc) override;
    int RenderOverlay(vtkViewport* viewport) override;
    int RenderOpaqueGeometry(vtkViewport* viewport) override;
    int RenderTranslucentPolygonalGeometry(vtkViewport* viewport) override;
    vtkTypeBool HasTranslucentPolygonalGeometry() override;

protected:
    cvConstrainedPolyLineRepresentation();
    ~cvConstrainedPolyLineRepresentation() override;

    /**
     * @brief Build the angle arc geometry
     */
    void BuildAngleArc();

    // Display properties
    vtkTypeBool ShowAngleLabel;
    vtkTypeBool ShowAngleArc;
    double ArcRadius;
    double Angle;       // Stored angle in degrees
    char* LabelSuffix;  // Instance label suffix (e.g., " #1", " #2")

    // Angle label (text)
    vtkTextActor* AngleLabelActor;

    // Angle arc (visual indicator)
    vtkActor2D* AngleArcActor;
    vtkPolyDataMapper2D* AngleArcMapper;
    vtkPolyData* AngleArcPolyData;

private:
    cvConstrainedPolyLineRepresentation(
            const cvConstrainedPolyLineRepresentation&) = delete;
    void operator=(const cvConstrainedPolyLineRepresentation&) = delete;
};

#endif  // CV_CONSTRAINED_POLYLINE_REPRESENTATION_H
