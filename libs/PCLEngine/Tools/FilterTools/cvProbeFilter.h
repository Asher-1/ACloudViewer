// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef TOOLS_PROBE_FILTER_H
#define TOOLS_PROBE_FILTER_H

#include "cvGenericFilter.h"

namespace Ui {
class cvProbeFilterDlg;
}

class vtkLineWidget;
class vtkBoxWidget;
class vtkSphereWidget;
class vtkPlaneWidget;
class vtkProbeFilter;
class vtkImplicitPlaneWidget;
class QCustomPlot;
class cvProbeFilter : public cvGenericFilter {
    Q_OBJECT
public:
    explicit cvProbeFilter(QWidget* parent = nullptr);

    virtual void apply() override;

    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

public:
    virtual void showInteractor(bool state) override;
    virtual void getInteractorBounds(ccBBox& bbox) override;
    virtual void getInteractorTransformation(ccGLMatrixd& trans) override;
    // virtual void getInteractorInfos(ccBBox& bbox, ccGLMatrixd& trans)
    // override;
    virtual void shift(const CCVector3d& v) override;

protected:
    virtual void modelReady() override;
    virtual void createUi() override;
    virtual void initFilter() override;
    virtual void dataChanged() override;

private slots:
    void on_sourceCombo_currentIndexChanged(int index);
    void onLineWidgetPointsChanged(double* point1, double* point2);
    void onSphereWidgetCenterChanged(double* center);
    void onImplicitPlaneWidgetOriginChanged(double* origin);
    void onImplicitPlaneWidgetNormalChanged(double* normal);
    void on_sphereRadius_valueChanged(double arg1);

private:
    void showProbeWidget();

protected:
    enum WidgetType { WT_Line, WT_Sphere, WT_Box, WT_ImplicitPlane };

    Ui::cvProbeFilterDlg* m_configUi = nullptr;

    WidgetType m_widgetType = WT_Line;
    vtkSmartPointer<vtkLineWidget> m_lineWidget;
    vtkSmartPointer<vtkSphereWidget> m_sphereWidget;
    vtkSmartPointer<vtkBoxWidget> m_boxWidget;
    vtkSmartPointer<vtkImplicitPlaneWidget> m_implicitPlaneWidget;

    vtkSmartPointer<vtkProbeFilter> m_probe;

    QCustomPlot* m_plotWidget = nullptr;

    double m_linePoint1[3];
    double m_linePoint2[3];

    double m_sphereCenter[3];
    double m_sphereRadius;

    double m_planeOrigin[3];
    double m_planeNormal[3];
};

#endif  // TOOLS_PROBE_FILTER_H
