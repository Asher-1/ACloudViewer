// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericMeasurementTool.h"
#include "ui_cvDistanceToolDlg.h"

class vtkDistanceWidget;
class vtkDistanceRepresentation2D;
class vtkDistanceRepresentation3D;

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

protected:
    virtual void initTool() override;
    virtual void createUi() override;
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) override;

private slots:
    //! Handle keyboard shortcut point picking
    void pickAlternatingPoint(double x, double y, double z);
    void pickKeyboardPoint1(double x, double y, double z);
    void pickKeyboardPoint2(double x, double y, double z);
    void pickNormalDirection(double px, double py, double pz, double nx, double ny, double nz);
    
    void on_typeCombo_currentIndexChanged(int index);
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
    void on_widgetVisibilityCheckBox_toggled(bool checked);
    void on_labelVisibilityCheckBox_toggled(bool checked);
    void on_lineWidthSpinBox_valueChanged(double value);

private:
    void hookWidget(const vtkSmartPointer<vtkDistanceWidget>& widget);
    void updateDistanceDisplay();

    Ui::DistanceToolDlg* m_configUi = nullptr;
    vtkSmartPointer<vtkDistanceWidget> m_2dWidget;
    vtkSmartPointer<vtkDistanceWidget> m_3dWidget;
    vtkSmartPointer<vtkDistanceRepresentation2D> m_2dRep;
    vtkSmartPointer<vtkDistanceRepresentation3D> m_3dRep;
    
    //! Flag to track alternating point picking (true = pick point 1, false = pick point 2)
    bool m_pickPoint1Next = true;
};

