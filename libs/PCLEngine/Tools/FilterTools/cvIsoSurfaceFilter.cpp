// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvIsoSurfaceFilter.h"

#include <Utils/vtk2cc.h>
#include <VtkUtils/gradientcombobox.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/vtkutils.h>
#include <vtkCellData.h>
#include <vtkContourFilter.h>
#include <vtkDataSet.h>
#include <vtkLODActor.h>
#include <vtkLookupTable.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkUnstructuredGrid.h>

#include <QWidget>

#include "ui_cvGenericFilterDlg.h"
#include "ui_cvIsoSurfaceFilterDlg.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

cvIsoSurfaceFilter::cvIsoSurfaceFilter(QWidget* parent)
    : cvGenericFilter(parent) {
    setWindowTitle(tr("Isosurface"));
    createUi();

    VtkUtils::SignalBlocker sb(m_configUi->numOfContoursSpinBox);
    m_configUi->numOfContoursSpinBox->setValue(m_numOfContours);
    m_configUi->gradientCombo->setCurrentIndex(0);

    connect(m_configUi->minScalarSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(onDoubleSpinBoxValueChanged(double)));
    connect(m_configUi->maxScalarSpinBox, SIGNAL(valueChanged(double)), this,
            SLOT(onDoubleSpinBoxValueChanged(double)));
    connect(m_configUi->numOfContoursSpinBox, SIGNAL(valueChanged(int)), this,
            SLOT(onSpinBoxValueChanged(int)));
    connect(m_configUi->displayEffectCombo, SIGNAL(currentIndexChanged(int)),
            this, SLOT(onComboBoxIndexChanged(int)));
}

cvIsoSurfaceFilter::~cvIsoSurfaceFilter() {}

void cvIsoSurfaceFilter::apply() {
    if (!m_dataObject) {
        CVLog::Error(QString("Isosurface::apply null data object."));
        return;
    }

    if (!m_meshMode) {
        CVLog::Error(QString("Isosurface::apply mesh supported only!"));
        return;
    }

    VtkUtils::vtkInitOnce(m_contourFilter);
    m_contourFilter->SetInputData(m_dataObject);
    m_contourFilter->ComputeScalarsOn();
    m_contourFilter->GenerateValues(m_numOfContours, m_minScalar, m_maxScalar);

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(m_minScalar, m_maxScalar);
    lut->SetNumberOfColors(m_numOfContours);
    lut->Build();
    // m_contourFilter->Update();

    if (!m_filterActor) {
        VtkUtils::vtkInitOnce(m_filterActor);
        VTK_CREATE(vtkPolyDataMapper, mapper);
        mapper->SetLookupTable(lut);
        mapper->SetInputConnection(m_contourFilter->GetOutputPort());
        mapper->ScalarVisibilityOn();
        m_filterActor->SetMapper(mapper);
        addActor(m_filterActor);
    }

    m_filterActor->GetMapper()->SetLookupTable(lut);
    if (m_scalarBar) {
        m_scalarBar->SetLookupTable(lut);
    }

    applyDisplayEffect();
}

ccHObject* cvIsoSurfaceFilter::getOutput() {
    if (!m_contourFilter) return nullptr;

    bool exportPolylines = m_configUi->polylinesRadioButton->isChecked();
    bool exportCloud = m_configUi->cloudRadioButton->isChecked();
    if (!exportPolylines && !exportCloud) {
        CVLog::Warning(QString("must check one export mode"));
        return nullptr;
    }

    // set exported polydata
    m_contourFilter->Update();

    vtkPolyData* polydata = nullptr;
    if (exportCloud) {
        setResultData(m_contourFilter->GetOutput());
        polydata = vtkPolyData::SafeDownCast(resultData());
    } else {
        polydata = m_contourFilter->GetOutput();
    }

    if (nullptr == polydata) return nullptr;

    ccHObject* result = new ccHObject();

    if (exportCloud) {
        m_meshMode = false;
        ccHObject* cloud = cvGenericFilter::getOutput();
        m_meshMode = true;
        if (cloud) {
            result->addChild(cloud);
        }
    }

    if (exportPolylines) {
        ccHObject::Container container = vtk2cc::ConvertToMultiPolylines(
                polydata, "Slice", ecvColor::green);
        if (!container.empty() && m_entity) {
            for (auto& obj : container) {
                if (!obj) {
                    continue;
                }
                ccPolyline* poly = ccHObjectCaster::ToPolyline(obj);
                if (!poly) continue;

                if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                    ccPointCloud* ccCloud =
                            ccHObjectCaster::ToPointCloud(m_entity);
                    poly->setGlobalScale(ccCloud->getGlobalScale());
                    poly->setGlobalShift(ccCloud->getGlobalShift());
                } else if (m_entity->isKindOf(CV_TYPES::MESH)) {
                    ccMesh* mesh = ccHObjectCaster::ToMesh(m_entity);
                    poly->setGlobalScale(
                            mesh->getAssociatedCloud()->getGlobalScale());
                    poly->setGlobalShift(
                            mesh->getAssociatedCloud()->getGlobalShift());
                }

                poly->setName(m_entity->getName() + "-" + poly->getName());
                result->addChild(poly);
            }
        }
    }

    if (result->getChildrenNumber() == 0) {
        delete result;
        result = nullptr;
    }

    return result;
}

void cvIsoSurfaceFilter::clearAllActor() {
    if (!m_viewer) return;

    if (m_modelActor) {
        vtkDataSet* data = m_modelActor->GetMapper()->GetInput();
        vtkSmartPointer<vtkDataArray> scalars =
                data->GetPointData()->GetScalars();
        if (scalars) {
            double minmax[2];
            scalars->GetRange(minmax);
            m_modelActor->GetMapper()->SetScalarRange(minmax);

            m_modelActor->GetMapper()->SetScalarModeToUsePointData();
            m_modelActor->GetMapper()->SetColorModeToDefault();
            m_modelActor->GetMapper()->SetInterpolateScalarsBeforeMapping(
                    getDefaultScalarInterpolationForDataSet(data));
            m_modelActor->GetMapper()->ScalarVisibilityOn();
        }
    }

    cvGenericFilter::clearAllActor();
}

void cvIsoSurfaceFilter::modelReady() {
    cvGenericFilter::modelReady();

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(m_dataObject);
    vtkDataSet* dataset = vtkDataSet::SafeDownCast(m_dataObject);
    QStringList scalarNames;
    double scalarRange[2];

    if (polydata) {
        // get scalar range
        polydata->GetScalarRange(scalarRange);
        // get scalar names from polydata
        vtkPointData* pointData = polydata->GetPointData();
        int numOfArray = pointData->GetNumberOfArrays();
        for (int i = 0; i < numOfArray; ++i)
            scalarNames.append(pointData->GetArrayName(i));
    } else if (dataset) {
        // get scalar range
        dataset->GetScalarRange(scalarRange);

        // get scalar names from dataset
        vtkCellData* cd = dataset->GetCellData();
        int arrayNum = cd->GetNumberOfArrays();
        if (arrayNum) {
            for (int i = 0; i < arrayNum; ++i)
                scalarNames.append(cd->GetArrayName(i));
        } else {
            vtkPointData* pd = dataset->GetPointData();
            arrayNum = pd->GetNumberOfArrays();
            for (int i = 0; i < arrayNum; ++i)
                scalarNames.append(pd->GetArrayName(i));
        }
    }

    setScalarRange(scalarRange[0], scalarRange[1]);

    m_configUi->scalarCombo->clear();
    m_configUi->scalarCombo->addItems(scalarNames);
    m_configUi->minScalarSpinBox->setRange(scalarRange[0], scalarRange[1]);
    m_configUi->minScalarSpinBox->setValue(scalarRange[0]);
    m_configUi->maxScalarSpinBox->setRange(scalarRange[0], scalarRange[1]);
    m_configUi->maxScalarSpinBox->setValue(scalarRange[1]);

    showScalarBar();

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(scalarRange[0], scalarRange[1]);
    lut->SetNumberOfColors(m_numOfContours);
    lut->Build();

    m_scalarBar->SetLookupTable(lut);
}

void cvIsoSurfaceFilter::createUi() {
    cvGenericFilter::createUi();

    m_configUi = new Ui::cvIsoSurfaceFilterDlg;
    setupConfigWidget(m_configUi);
}

void cvIsoSurfaceFilter::colorsChanged() {
    if (!m_filterActor || !m_scalarBar) return;

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(m_minScalar, m_maxScalar);
    lut->SetNumberOfColors(m_numOfContours);
    lut->Build();
    if (m_modelActor) m_modelActor->GetMapper()->SetLookupTable(lut);
    if (m_filterActor) m_filterActor->GetMapper()->SetLookupTable(lut);
    if (m_scalarBar) {
        m_scalarBar->SetNumberOfLabels(m_numOfContours);
        m_scalarBar->SetLookupTable(lut);
    }

    update();
}

void cvIsoSurfaceFilter::initFilter() {
    m_meshMode ? m_configUi->displayEffectCombo->setCurrentIndex(
                         DisplayEffect::Transparent)
               : m_configUi->displayEffectCombo->setCurrentIndex(
                         DisplayEffect::Opaque);

    if (m_modelActor) {
        // this will change origin data scalars!!!
        vtkSmartPointer<vtkDataArray> scalars = getActorScalars(m_modelActor);
        if (scalars) {
            m_modelActor->GetMapper()->ScalarVisibilityOn();
            m_modelActor->GetMapper()->SetScalarModeToUsePointData();
            m_modelActor->GetMapper()->SetColorModeToMapScalars();

            if (m_filterActor) {
                m_filterActor->GetMapper()->ScalarVisibilityOn();
                m_filterActor->GetMapper()->SetScalarModeToUsePointData();
                m_filterActor->GetMapper()->SetColorModeToMapScalars();

                if (m_configUi) {
                    m_configUi->gradientCombo->setCurrentIndex(3);
                }
            }
        }
    }
}

void cvIsoSurfaceFilter::dataChanged() { apply(); }

void cvIsoSurfaceFilter::onDoubleSpinBoxValueChanged(double value) {
    QDoubleSpinBox* dsb = qobject_cast<QDoubleSpinBox*>(sender());
    if (dsb == m_configUi->minScalarSpinBox)
        m_minScalar = value;
    else if (dsb == m_configUi->maxScalarSpinBox)
        m_maxScalar = value;

    apply();
}

void cvIsoSurfaceFilter::onSpinBoxValueChanged(int value) {
    m_numOfContours = value;
    apply();
}

void cvIsoSurfaceFilter::onComboBoxIndexChanged(int index) {
    QComboBox* cb = qobject_cast<QComboBox*>(sender());

    if (cb == m_configUi->displayEffectCombo)
        setDisplayEffect(static_cast<cvGenericFilter::DisplayEffect>(index));
    else if (cb == m_configUi->scalarCombo)
        int i = 0;  // todo
}

void cvIsoSurfaceFilter::on_gradientCombo_activated(int index) {
    Q_UNUSED(index)
    Widgets::GradientComboBox* gcb =
            qobject_cast<Widgets::GradientComboBox*>(sender());
    setScalarBarColors(gcb->currentColor1(), gcb->currentColor2());
}
