// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericMeasurementTool.h"
#include "ui_cvProtractorToolDlg.h"

class vtkAngleWidget;
class vtkAngleRepresentation2D;
class vtkAngleRepresentation3D;

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

protected:
    virtual void initTool() override;
    virtual void createUi() override;
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) override;

private slots:
    //! Handle keyboard shortcut point picking
    void pickKeyboardPoint1(double x, double y, double z);
    void pickKeyboardCenter(double x, double y, double z);
    void pickKeyboardPoint2(double x, double y, double z);
    
    void on_typeCombo_currentIndexChanged(int index);
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
    void on_ray1VisibilityCheckBox_toggled(bool checked);
    void on_ray2VisibilityCheckBox_toggled(bool checked);
    void on_arcVisibilityCheckBox_toggled(bool checked);

private:
    void hookWidget(const vtkSmartPointer<vtkAngleWidget>& widget);
    void updateAngleDisplay();

    Ui::ProtractorToolDlg* m_configUi = nullptr;
    vtkSmartPointer<vtkAngleWidget> m_2dWidget;
    vtkSmartPointer<vtkAngleWidget> m_3dWidget;
    // IMPORTANT: vtkAngleRepresentation2D::GetAngle() returns DEGREES
    //            vtkAngleRepresentation3D::GetAngle() returns RADIANS
    vtkSmartPointer<vtkAngleRepresentation2D> m_2dRep;
    vtkSmartPointer<vtkAngleRepresentation3D> m_3dRep;
};
