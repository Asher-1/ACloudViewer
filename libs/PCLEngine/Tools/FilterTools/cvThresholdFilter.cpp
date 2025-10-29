// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvThresholdFilter.h"

#include <CVLog.h>
#include <VtkUtils/vtkutils.h>
#include <vtkActor.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkLODActor.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkThreshold.h>

#include "ui_cvIsoSurfaceFilterDlg.h"

cvThresholdFilter::cvThresholdFilter(QWidget* parent)
    : cvIsoSurfaceFilter(parent) {
    setWindowTitle(tr("Threshold"));
}

cvThresholdFilter::~cvThresholdFilter() {}

void cvThresholdFilter::apply() {
    if (!m_dataObject) {
        CVLog::Error(QString("Threshold::apply: null data object."));
        return;
    }

    VTK_CREATE(vtkThreshold, thresholdFilter);
    thresholdFilter->SetInputData(m_dataObject);
    thresholdFilter->SetAllScalars(false);
    // thresholdFilter->ThresholdBetween(m_minScalar, m_maxScalar);
    thresholdFilter->SetLowerThreshold(m_minScalar);
    thresholdFilter->SetUpperThreshold(m_maxScalar);
    thresholdFilter->Update();

    VtkUtils::vtkInitOnce(m_dssFilter);
    m_dssFilter->SetInputConnection(thresholdFilter->GetOutputPort());

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(m_minScalar, m_maxScalar);
    lut->SetNumberOfColors(m_numOfContours);
    lut->Build();

    if (!m_filterActor) {
        VtkUtils::vtkInitOnce(m_filterActor);
        VTK_CREATE(vtkPolyDataMapper, mapper);
        mapper->ScalarVisibilityOn();
        mapper->SetInputConnection(m_dssFilter->GetOutputPort());
        mapper->SetLookupTable(lut);
        m_filterActor->SetMapper(mapper);
        addActor(m_filterActor);
    }

    m_filterActor->GetMapper()->SetLookupTable(lut);
    if (m_scalarBar) {
        m_scalarBar->SetLookupTable(lut);
    }

    applyDisplayEffect();
}

ccHObject* cvThresholdFilter::getOutput() {
    if (!m_dssFilter) return nullptr;

    m_dssFilter->Update();
    setResultData(m_dssFilter->GetOutput());
    return cvGenericFilter::getOutput();
}

void cvThresholdFilter::clearAllActor() { cvIsoSurfaceFilter::clearAllActor(); }

void cvThresholdFilter::initFilter() {
    cvIsoSurfaceFilter::initFilter();

    if (m_configUi) {
        m_configUi->displayEffectCombo->setCurrentIndex(DisplayEffect::Opaque);
        m_configUi->polylinesRadioButton->setEnabled(false);
        m_configUi->cloudRadioButton->setEnabled(false);
    }
}
