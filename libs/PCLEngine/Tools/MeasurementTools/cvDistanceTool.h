// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericMeasurementTool.h"
#include "ui_cvDistanceToolDlg.h"

// Forward declarations
class cvConstrainedDistanceWidget;
class cvConstrainedLineRepresentation;

class cvDistanceTool : public cvGenericMeasurementTool {
    Q_OBJECT

public:
    explicit cvDistanceTool(QWidget* parent = nullptr);
    ~cvDistanceTool() override;

    virtual void start() override;
    virtual void reset() override;
    virtual void showWidget(bool state) override;
    virtual ccHObject* getOutput() override;

    virtual double getMeasurementValue() const override;
    virtual void getPoint1(double pos[3]) const override;
    virtual void getPoint2(double pos[3]) const override;
    virtual void setPoint1(double pos[3]) override;
    virtual void setPoint2(double pos[3]) override;
    virtual void setColor(double r, double g, double b) override;
    virtual void lockInteraction() override;
    virtual void unlockInteraction() override;
    virtual void setInstanceLabel(const QString& label) override;

protected:
    virtual void initTool() override;
    virtual void createUi() override;
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) override;

private slots:
    //! Handle keyboard shortcut point picking
    void pickAlternatingPoint(double x, double y, double z);
    void pickKeyboardPoint1(double x, double y, double z);
    void pickKeyboardPoint2(double x, double y, double z);
    void pickNormalDirection(
            double px, double py, double pz, double nx, double ny, double nz);

    void on_point1XSpinBox_valueChanged(double arg1);
    void on_point1YSpinBox_valueChanged(double arg1);
    void on_point1ZSpinBox_valueChanged(double arg1);
    void on_point2XSpinBox_valueChanged(double arg1);
    void on_point2YSpinBox_valueChanged(double arg1);
    void on_point2ZSpinBox_valueChanged(double arg1);
    void onDistanceChanged(double dist);
    void onWorldPoint1Changed(double* pos);
    void onWorldPoint2Changed(double* pos);
    void on_pickPoint1_toggled(bool checked);
    void on_pickPoint2_toggled(bool checked);
    void on_rulerModeCheckBox_toggled(bool checked);
    void on_rulerDistanceSpinBox_valueChanged(double value);
    void on_numberOfTicksSpinBox_valueChanged(int value);
    void on_scaleSpinBox_valueChanged(double value);
    void on_labelFormatLineEdit_textChanged(const QString& text);
    void on_widgetVisibilityCheckBox_toggled(bool checked);
    void on_labelVisibilityCheckBox_toggled(bool checked);
    void on_lineWidthSpinBox_valueChanged(double value);

private:
    void hookWidget(const vtkSmartPointer<cvConstrainedDistanceWidget>& widget);
    void updateDistanceDisplay();

    Ui::DistanceToolDlg* m_configUi = nullptr;
    vtkSmartPointer<cvConstrainedDistanceWidget>
            m_widget;  // Constrained widget (supports XYZL constraints)
    vtkSmartPointer<cvConstrainedLineRepresentation>
            m_rep;  // Custom representation (supports distance display + ruler
                    // + XYZL constraints)

    //! Flag to track alternating point picking (true = pick point 1, false =
    //! pick point 2)
    bool m_pickPoint1Next = true;

    //! Current color (RGB, normalized to [0, 1])
    double m_currentColor[3] = {0.0, 1.0, 0.0};  // Default green

    //! Instance label suffix (e.g., " #1", " #2") for display in 3D view
    QString m_instanceLabel;

    //! Helper to apply font properties to VTK text property
    //! Uses font properties from base class (cvGenericMeasurementTool)
    void applyFontProperties() override;
};
