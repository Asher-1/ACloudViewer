// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cvGenericFilter.h"

namespace Ui {
class cvDecimateFilterDlg;
}

class vtkDecimatePro;
class cvDecimateFilter : public cvGenericFilter {
    Q_OBJECT
public:
    explicit cvDecimateFilter(QWidget* parent = nullptr);

    virtual void apply() override;

    virtual ccHObject* getOutput() override;

    virtual void clearAllActor() override;

protected:
    virtual void initFilter() override;
    virtual void dataChanged() override;

private slots:
    void on_targetReductionSpinBox_valueChanged(double arg1);
    void on_preserveTopologyCheckBox_toggled(bool checked);
    void on_splittingCheckBox_toggled(bool checked);
    void on_presplitMeshCheckBox_toggled(bool checked);
    void on_accumulateErrorCheckBox_toggled(bool checked);
    void on_boundaryVertexDeletionCheckBox_toggled(bool checked);
    void on_featureAngleSpinBox_valueChanged(double arg1);
    void on_splitAngleSpinBox_valueChanged(double arg1);
    void on_outputPointsPrecisionSpinBox_valueChanged(int arg1);
    void on_inflectionPointRatioSpinBox_valueChanged(double arg1);
    void on_degreeSpinBox_valueChanged(int arg1);

protected:
    virtual void createUi() override;

private:
    Ui::cvDecimateFilterDlg* m_configUi = nullptr;

    double m_targetReduction;
    double m_featureAngle;
    double m_splitAngle;
    double m_inflectionPointRatio;

    bool m_preserveTopology;
    bool m_splitting;
    bool m_presplitMesh;
    bool m_accumulateError;
    bool m_boundaryVertexDeletion;

    int m_outputPointsPrecision;
    int m_degree;

    vtkSmartPointer<vtkDecimatePro> m_decimate;
};
