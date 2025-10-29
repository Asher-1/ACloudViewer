// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvDecimateFilter.h"

#include <CVLog.h>
#include <VtkUtils/vtkutils.h>
#include <vtkActor.h>
#include <vtkDecimatePro.h>
#include <vtkLODActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>

#include "ui_cvDecimateFilterDlg.h"
#include "ui_cvGenericFilterDlg.h"

cvDecimateFilter::cvDecimateFilter(QWidget* parent) : cvGenericFilter(parent) {
    m_keepMode = false;
    setWindowTitle(tr("Decimate"));
    createUi();
}

void cvDecimateFilter::apply() {
    if (m_keepMode) {
        return;
    }

    if (!m_dataObject) {
        CVLog::Error(QString("cvDecimateFilter::apply: null data object."));
        return;
    }
    if (!m_meshMode) {
        CVLog::Error(QString("cvDecimateFilter::apply: mesh supported only!"));
        return;
    }

    VtkUtils::vtkInitOnce(m_decimate);
    m_decimate->SetInputData(m_dataObject);
    m_decimate->SetTargetReduction(m_targetReduction);
    m_decimate->SetFeatureAngle(m_featureAngle);
    m_decimate->SetSplitAngle(m_splitAngle);
    m_decimate->SetInflectionPointRatio(m_inflectionPointRatio);
    m_decimate->SetPreserveTopology(m_preserveTopology);
    m_decimate->SetSplitting(m_splitting);
    m_decimate->SetPreSplitMesh(m_presplitMesh);
    m_decimate->SetAccumulateError(m_accumulateError);
    m_decimate->SetOutputPointsPrecision(m_outputPointsPrecision);
    m_decimate->SetDegree(m_degree);
    m_decimate->SetBoundaryVertexDeletion(m_boundaryVertexDeletion);

    m_decimate->Update();

    if (!m_filterActor) {
        VTK_CREATE(vtkPolyDataMapper, mapper);
        mapper->SetInputConnection(m_decimate->GetOutputPort());
        VtkUtils::vtkInitOnce(m_filterActor);
        m_filterActor->SetMapper(mapper);
        addActor(m_filterActor);
    }

    applyDisplayEffect();
}

ccHObject* cvDecimateFilter::getOutput() {
    if (!m_decimate) return nullptr;

    // set exported polydata
    m_decimate->Update();
    setResultData(m_decimate->GetOutput());
    return cvGenericFilter::getOutput();
}

void cvDecimateFilter::clearAllActor() { cvGenericFilter::clearAllActor(); }

void cvDecimateFilter::initFilter() {
    cvGenericFilter::initFilter();
    setDisplayEffect(DisplayEffect::Opaque);
}

void cvDecimateFilter::dataChanged() {
    m_keepMode = true;
    m_configUi->preserveTopologyCheckBox->setChecked(false);
    m_configUi->splittingCheckBox->setChecked(true);
    m_configUi->presplitMeshCheckBox->setChecked(false);
    m_configUi->accumulateErrorCheckBox->setChecked(false);
    m_configUi->boundaryVertexDeletionCheckBox->setChecked(true);

    m_configUi->targetReductionSpinBox->setValue(0.3);
    m_configUi->featureAngleSpinBox->setValue(10.0);
    m_configUi->splitAngleSpinBox->setValue(10.0);
    m_configUi->outputPointsPrecisionSpinBox->setValue(0);
    m_configUi->inflectionPointRatioSpinBox->setValue(1.0);
    m_configUi->degreeSpinBox->setValue(0);
    m_keepMode = false;
    apply();
}

void cvDecimateFilter::on_targetReductionSpinBox_valueChanged(double arg1) {
    m_targetReduction = arg1;
    apply();
}

void cvDecimateFilter::on_preserveTopologyCheckBox_toggled(bool checked) {
    m_preserveTopology = checked;
    apply();
}

void cvDecimateFilter::on_splittingCheckBox_toggled(bool checked) {
    m_splitting = checked;
    apply();
}

void cvDecimateFilter::on_presplitMeshCheckBox_toggled(bool checked) {
    m_presplitMesh = checked;
    apply();
}

void cvDecimateFilter::on_accumulateErrorCheckBox_toggled(bool checked) {
    m_accumulateError = checked;
    apply();
}

void cvDecimateFilter::on_boundaryVertexDeletionCheckBox_toggled(bool checked) {
    m_boundaryVertexDeletion = checked;
    apply();
}

void cvDecimateFilter::on_featureAngleSpinBox_valueChanged(double arg1) {
    m_featureAngle = arg1;
    apply();
}

void cvDecimateFilter::on_splitAngleSpinBox_valueChanged(double arg1) {
    m_splitAngle = arg1;
    apply();
}

void cvDecimateFilter::on_outputPointsPrecisionSpinBox_valueChanged(int arg1) {
    m_outputPointsPrecision = arg1;
    apply();
}

void cvDecimateFilter::on_inflectionPointRatioSpinBox_valueChanged(
        double arg1) {
    m_inflectionPointRatio = arg1;
    apply();
}

void cvDecimateFilter::on_degreeSpinBox_valueChanged(int arg1) {
    m_degree = arg1;
    apply();
}

void cvDecimateFilter::createUi() {
    cvGenericFilter::createUi();

    m_configUi = new Ui::cvDecimateFilterDlg;
    setupConfigWidget(m_configUi);
}
