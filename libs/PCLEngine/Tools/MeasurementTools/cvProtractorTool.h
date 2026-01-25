// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericMeasurementTool.h"
#include "ui_cvProtractorToolDlg.h"

// Forward declarations for VTK classes
class cvConstrainedPolyLineRepresentation;

// Include the custom constrained widgets and representations
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedPolyLineRepresentation.h"
#include "VTKExtensions/ConstrainedWidgets/cvConstrainedPolyLineWidget.h"

class cvProtractorTool : public cvGenericMeasurementTool {
    Q_OBJECT

public:
    explicit cvProtractorTool(QWidget* parent = nullptr);
    ~cvProtractorTool() override;

    virtual void start() override;
    virtual void reset() override;
    virtual void showWidget(bool state) override;
    virtual ccHObject* getOutput() override;

    virtual double getMeasurementValue() const override;
    virtual void getPoint1(double pos[3]) const override;
    virtual void getPoint2(double pos[3]) const override;
    virtual void getCenter(double pos[3]) const override;
    virtual void setPoint1(double pos[3]) override;
    virtual void setPoint2(double pos[3]) override;
    virtual void setCenter(double pos[3]) override;
    virtual void setColor(double r, double g, double b) override;
    virtual bool getColor(double& r, double& g, double& b) const override;
    virtual void lockInteraction() override;
    virtual void unlockInteraction() override;
    virtual void setInstanceLabel(const QString& label) override;
    // Font property setters are now implemented in base class
    // Only applyFontProperties() needs to be overridden

protected:
    virtual void initTool() override;
    virtual void createUi() override;
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) override;

private slots:
    //! Handle keyboard shortcut point picking
    void pickKeyboardPoint1(double x, double y, double z);
    void pickKeyboardCenter(double x, double y, double z);
    void pickKeyboardPoint2(double x, double y, double z);

    void on_point1XSpinBox_valueChanged(double arg1);
    void on_point1YSpinBox_valueChanged(double arg1);
    void on_point1ZSpinBox_valueChanged(double arg1);
    void on_centerXSpinBox_valueChanged(double arg1);
    void on_centerYSpinBox_valueChanged(double arg1);
    void on_centerZSpinBox_valueChanged(double arg1);
    void on_point2XSpinBox_valueChanged(double arg1);
    void on_point2YSpinBox_valueChanged(double arg1);
    void on_point2ZSpinBox_valueChanged(double arg1);
    void onAngleChanged(double angle);
    void onWorldPoint1Changed(double* pos);
    void onWorldPoint2Changed(double* pos);
    void onWorldCenterChanged(double* pos);
    void on_pickPoint1_toggled(bool checked);
    void on_pickCenter_toggled(bool checked);
    void on_pickPoint2_toggled(bool checked);
    void on_widgetVisibilityCheckBox_toggled(bool checked);
    void on_arcVisibilityCheckBox_toggled(bool checked);

private:
    void hookWidget(const vtkSmartPointer<cvConstrainedPolyLineWidget>& widget);
    void updateAngleDisplay();

    //! Helper to apply text properties to label actor (without rebuilding)
    void applyTextPropertiesToLabel();

    Ui::ProtractorToolDlg* m_configUi = nullptr;
    vtkSmartPointer<cvConstrainedPolyLineWidget>
            m_widget;  // Using PolyLineWidget (100% ParaView consistency)
    vtkSmartPointer<cvConstrainedPolyLineRepresentation>
            m_rep;  // PolyLine with 3 handles for angle measurement

    //! Current color (RGB, normalized to [0, 1])
    double m_currentColor[3] = {0.0, 1.0, 0.0};  // Default green

    //! Instance label suffix (e.g., " #1", " #2") for display in 3D view
    QString m_instanceLabel;

    //! Helper to apply font properties to VTK text property
    //! Uses font properties from base class (cvGenericMeasurementTool)
    void applyFontProperties() override;
};
