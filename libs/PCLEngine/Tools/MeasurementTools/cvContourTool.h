// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericMeasurementTool.h"
#include "ui_cvContourToolDlg.h"

class vtkContourWidget;
class vtkPolyData;

#include <map>

class cvContourTool : public cvGenericMeasurementTool {
    Q_OBJECT

public:
    explicit cvContourTool(QWidget* parent = nullptr);
    ~cvContourTool() override;

    virtual void start() override;
    virtual void reset() override;
    virtual void showWidget(bool state) override;
    virtual ccHObject* getOutput() override;
    virtual void setColor(double r, double g, double b) override;
    virtual bool getColor(double& r, double& g, double& b) const override;
    virtual void lockInteraction() override;
    virtual void unlockInteraction() override;
    virtual void setInstanceLabel(const QString& label) override;

protected:
    virtual void initTool() override;
    virtual void createUi() override;

private slots:
    void onDataChanged(vtkPolyData* pd);
    void on_widgetVisibilityCheckBox_toggled(bool checked);
    void on_showNodesCheckBox_toggled(bool checked);
    void on_closedLoopCheckBox_toggled(bool checked);
    void on_lineWidthSpinBox_valueChanged(double value);

private:
    void createNewContour();

private:
    Ui::ContourToolDlg* m_configUi = nullptr;
    // Map of ID to ContourWidget
    std::map<int, vtkSmartPointer<vtkContourWidget>> m_contours;
    int m_currentContourId = -1;

    // Static counter for contour IDs (global)
    static int s_contourIdCounter;

    // Instance ID for this tool
    int m_toolId;

    // Export counter
    int m_exportCounter;

    //! Current color (RGB, normalized to [0, 1])
    double m_currentColor[3] = {0.0, 1.0, 0.0};  // Default green

    //! Instance label (for multi-instance identification)
    QString m_instanceLabel;

    //! Helper to apply font properties to VTK text property
    //! Uses font properties from base class (cvGenericMeasurementTool)
    void applyFontProperties() override;
};
