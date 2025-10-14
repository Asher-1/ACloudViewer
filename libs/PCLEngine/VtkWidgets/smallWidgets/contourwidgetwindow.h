// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "basewidgetwindow.h"

namespace Ui {
class ContourWidgetConfig;
}

class vtkContourWidget;
class vtkContextActor;
class vtkPolyData;
class ContourWidgetWindow : public BaseWidgetWindow {
    Q_OBJECT
public:
    explicit ContourWidgetWindow(QWidget* parent = nullptr);
    ~ContourWidgetWindow();

protected:
    void createWidget();
    void createUi();

private slots:
    void on_clearButton_clicked();
    void on_showNodesCheckBox_toggled(bool checked);
    void onDataChanged(vtkPolyData* pd);

private:
    Ui::ContourWidgetConfig* m_configUi = nullptr;

    vtkSmartPointer<vtkContourWidget> m_contour;
};
