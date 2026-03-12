// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvSmoothFilter.h
 * @brief Laplacian smoothing filter for VTK poly data surfaces.
 */

#include "cvGenericFilter.h"

namespace Ui {
class cvSmoothFilterDlg;
}

/**
 * @class cvSmoothFilter
 * @brief Applies vtkSmoothPolyDataFilter with configurable iterations and
 * convergence.
 */
class vtkSmoothPolyDataFilter;
class cvSmoothFilter : public cvGenericFilter {
    Q_OBJECT
public:
    /// @param parent Parent widget.
    explicit cvSmoothFilter(QWidget* parent = nullptr);

    virtual void apply() override;

    /// @return Smoothed mesh as ccHObject.
    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

private slots:
    void on_convergenceSpinBox_valueChanged(double arg1);
    void on_numOfIterationsSpinBox_valueChanged(int arg1);
    void on_featureEdgeSmoothingCheckBox_toggled(bool checked);
    void on_boundarySmoothingCheckBox_toggled(bool checked);
    void on_featureAngleSpinBox_valueChanged(double arg1);
    void on_edgeAngleSpinBox_valueChanged(double arg1);

protected:
    virtual void createUi() override;
    virtual void initFilter() override;
    virtual void dataChanged() override;

private:
    Ui::cvSmoothFilterDlg* m_configUi = nullptr;

    int m_numberOfIterations;

    bool m_featureEdgeSmoothing;
    bool m_boundarySmoothing;

    double m_convergence;
    double m_featureAngle;
    double m_edgeAngle;

    vtkSmartPointer<vtkSmoothPolyDataFilter> m_smoothFilter;
};
