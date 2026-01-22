// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvStreamlineFilter.h"

#include <VtkUtils/linewidgetobserver.h>
#include <VtkUtils/pointwidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/utils.h>
#include <VtkUtils/vtkutils.h>
#include <vtkDataSet.h>
#include <vtkLODActor.h>
#include <vtkLineSource.h>
#include <vtkLineWidget.h>
#include <vtkPointSource.h>
#include <vtkPointWidget.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRuledSurfaceFilter.h>
#include <vtkRungeKutta2.h>
#include <vtkRungeKutta4.h>
#include <vtkRungeKutta45.h>
#include <vtkStreamTracer.h>
#include <vtkTubeFilter.h>

#include "ui_cvGenericFilterDlg.h"
#include "ui_cvStreamlineFilterDlg.h"

// CV_DB_LIB
#include <ecvBBox.h>

cvStreamlineFilter::cvStreamlineFilter(QWidget* parent)
    : cvGenericFilter(parent) {
    setWindowTitle(tr("Streamline"));
    createUi();

    m_configUi->configTubeGroupBox->hide();
    m_configUi->configRuledSurfaceGroupBox->hide();
    showConfigGroupBox(0);
}

cvStreamlineFilter::~cvStreamlineFilter() {}

void cvStreamlineFilter::createUi() {
    cvGenericFilter::createUi();

    m_configUi = new Ui::cvStreamlineFilterDlg;
    setupConfigWidget(m_configUi);
}

void cvStreamlineFilter::apply() {
    if (!m_dataObject) return;

    VtkUtils::vtkInitOnce(m_streamline);
    m_streamline->SetInputData(m_dataObject);
    m_streamline->SetIntegrationDirection(m_integrationDirection);

    if (m_source == SourceType::Point) {
        VTK_CREATE(vtkPointSource, ps);
        ps->SetCenter(m_sphereCenter);
        ps->SetNumberOfPoints(m_numberOfPoints);
        ps->SetRadius(m_radius);
        ps->Update();
        m_streamline->SetSourceConnection(ps->GetOutputPort());
    } else if (m_source == SourceType::Line) {
        VTK_CREATE(vtkLineSource, ls);
        ls->SetResolution(30);
        ls->SetPoint1(m_linePoint1);
        ls->SetPoint2(m_linePoint2);
        ls->Update();
        m_streamline->SetSourceConnection(ls->GetOutputPort());
    }

    m_streamline->SetMaximumPropagation(100);
    m_streamline->SetInitialIntegrationStep(0.1);

    switch (m_integratorType) {
        case 0: {
            VTK_CREATE(vtkRungeKutta2, integ);
            m_streamline->SetIntegrator(integ);
            break;
        }

        case 1: {
            VTK_CREATE(vtkRungeKutta4, integ);
            m_streamline->SetIntegrator(integ);
            break;
        }

        case 2: {
            VTK_CREATE(vtkRungeKutta45, integ);
            m_streamline->SetIntegrator(integ);
            break;
        }

    }  // switch

    m_streamline->Update();

    VTK_CREATE(vtkPolyDataMapper, streamLineMapper);

    switch (m_displayMode) {
        case None:
            streamLineMapper->SetInputConnection(m_streamline->GetOutputPort());
            break;

        case Surface: {
            VTK_CREATE(vtkRuledSurfaceFilter, surfaceFilter);
            surfaceFilter->SetOffset(0);
            surfaceFilter->SetOnRatio(.2);
            surfaceFilter->PassLinesOn();
            surfaceFilter->SetRuledModeToPointWalk();
            surfaceFilter->SetDistanceFactor(30);
#if 0
		surfaceFilter->SetOffset(m_ruledSurfaceParams.offset);
		surfaceFilter->SetOnRatio(m_ruledSurfaceParams.onRatio);
		surfaceFilter->SetPassLines(m_ruledSurfaceParams.passLines);
//        surfaceFilter->SetRuledMode(m_ruledSurfaceParams.ruledMode);
		surfaceFilter->SetDistanceFactor(m_ruledSurfaceParams.distanceFactor);
		surfaceFilter->SetResolution(m_ruledSurfaceParams.resolution);
		surfaceFilter->SetOrientLoops(m_ruledSurfaceParams.orientLoops);
		surfaceFilter->SetCloseSurface(m_ruledSurfaceParams.closeSurface);
		surfaceFilter->SetInputConnection(m_streamline->GetOutputPort());
#endif
            streamLineMapper->SetInputConnection(
                    surfaceFilter->GetOutputPort());
        } break;

        case Tube: {
            VTK_CREATE(vtkTubeFilter, tubeFilter);
            tubeFilter->SetNumberOfSides(m_tubeParams.numberOfSides);
            tubeFilter->SetRadiusFactor(m_tubeParams.radiusFactor);
            tubeFilter->SetCapping(m_tubeParams.capping);
            tubeFilter->SetInputConnection(m_streamline->GetOutputPort());
            streamLineMapper->SetInputConnection(tubeFilter->GetOutputPort());
        } break;
    }

    if (!m_filterActor) {
        VtkUtils::vtkInitOnce(m_filterActor);
        m_filterActor->SetMapper(streamLineMapper);
        addActor(m_filterActor);
    }

    m_filterActor->SetMapper(streamLineMapper);

    applyDisplayEffect();
}

ccHObject* cvStreamlineFilter::getOutput() {
    if (!m_filterActor) return nullptr;

    setResultData(m_streamline->GetOutput());
    return cvGenericFilter::getOutput();
}

void cvStreamlineFilter::initFilter() {
    cvGenericFilter::initFilter();
    if (m_configUi) {
        m_configUi->displayEffectCombo->setCurrentIndex(
                DisplayEffect::Transparent);
    }
}

void cvStreamlineFilter::dataChanged() {
    if (!m_dataObject) return;

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(m_dataObject);
    vtkDataSet* dataset = vtkDataSet::SafeDownCast(m_dataObject);
    double bounds[6];
    double center[3];

    if (polydata) {
        polydata->GetBounds(bounds);
        polydata->GetCenter(center);
    } else if (dataset) {
        dataset->GetBounds(bounds);
        dataset->GetCenter(center);
    }

    double point1[3];
    double point2[3];
    point1[0] = bounds[0];
    point1[1] = center[1];
    point1[2] = center[2];
    point2[0] = bounds[1];
    point2[1] = center[1];
    point2[2] = center[2];

    if (m_source == SourceType::Point)
        onPointPositionChanged(center);
    else if (m_source == SourceType::Line)
        onLinePointsChanged(point1, point2);

    apply();
}

void cvStreamlineFilter::showInteractor(bool state) {
    switch (m_source) {
        case SourceType::Point:
            if (m_pointWidget) {
                state ? m_pointWidget->On() : safeOff(m_pointWidget);
            }
            break;
        case SourceType::Line:
            if (m_lineWidget) {
                state ? m_lineWidget->On() : safeOff(m_lineWidget);
            }
            break;
        default:
            break;
    }
}

void cvStreamlineFilter::getInteractorBounds(ccBBox& bbox) {
    double bounds[6];

    bool valid = true;

    switch (m_source) {
        case SourceType::Point:
            if (m_pointWidget) {
                m_pointWidget->GetProp3D()->GetBounds(bounds);
            }
            break;
        case SourceType::Line:
            if (m_lineWidget) {
                m_lineWidget->GetProp3D()->GetBounds(bounds);
            }
            break;
        default:
            valid = false;
            break;
    }

    if (!valid) {
        bbox.setValidity(valid);
        return;
    }

    CCVector3 minCorner(bounds[0], bounds[2], bounds[4]);
    CCVector3 maxCorner(bounds[1], bounds[3], bounds[5]);
    bbox.minCorner() = minCorner;
    bbox.maxCorner() = maxCorner;
    bbox.setValidity(valid);
}

void cvStreamlineFilter::getInteractorTransformation(ccGLMatrixd& trans) {
    switch (m_source) {
        case SourceType::Point:
            if (m_pointWidget) {
                m_pointWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        case SourceType::Line:
            if (m_lineWidget) {
                m_lineWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        default:
            break;
    }
}

void cvStreamlineFilter::shift(const CCVector3d& v) {
    switch (m_source) {
        case SourceType::Point:
            if (m_pointWidget) {
                double newPos[3];
                CCVector3d::vadd(v.u, m_sphereCenter, newPos);
                onPointPositionChanged(newPos);
            }
            break;
        case SourceType::Line:
            if (m_lineWidget) {
                double newPos1[3];
                double newPos2[3];
                CCVector3d::vadd(v.u, m_linePoint1, newPos1);
                CCVector3d::vadd(v.u, m_linePoint2, newPos2);
                onLinePointsChanged(newPos1, newPos2);
            }
            break;
        default:
            break;
    }
}

void cvStreamlineFilter::clearAllActor() {
    if (m_lineWidget) {
        safeOff(m_lineWidget);
    }

    if (m_pointWidget) {
        safeOff(m_pointWidget);
    }

    cvGenericFilter::clearAllActor();
}

void cvStreamlineFilter::modelReady() {
    cvGenericFilter::modelReady();

    if (!m_dataObject) return;

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(m_dataObject);
    vtkDataSet* dataset = vtkDataSet::SafeDownCast(m_dataObject);
    double bounds[6];
    double center[3];

    if (polydata) {
        polydata->GetBounds(bounds);
        polydata->GetCenter(center);
    } else if (dataset) {
        dataset->GetBounds(bounds);
        dataset->GetCenter(center);
    }

    Utils::ArrayAssigner<double>()(m_sphereCenter, center);

    m_linePoint1[0] = bounds[0];
    m_linePoint1[1] = center[1];
    m_linePoint1[2] = center[2];
    m_linePoint2[0] = bounds[1];
    m_linePoint2[1] = center[1];
    m_linePoint2[2] = center[2];

    m_configUi->centerXSpinBox->setRange(bounds[0], bounds[1]);
    m_configUi->centerYSpinBox->setRange(bounds[2], bounds[3]);
    m_configUi->centerZSpinBox->setRange(bounds[4], bounds[5]);
    m_configUi->point1XSpinBox->setRange(bounds[0], bounds[1]);
    m_configUi->point1YSpinBox->setRange(bounds[2], bounds[3]);
    m_configUi->point1ZSpinBox->setRange(bounds[4], bounds[5]);
    m_configUi->point2XSpinBox->setRange(bounds[0], bounds[1]);
    m_configUi->point2YSpinBox->setRange(bounds[2], bounds[3]);
    m_configUi->point2ZSpinBox->setRange(bounds[4], bounds[5]);

    VtkUtils::SignalBlocker sb(m_configUi->centerXSpinBox);
    sb.addObject(m_configUi->centerYSpinBox);
    sb.addObject(m_configUi->centerZSpinBox);
    sb.addObject(m_configUi->point1XSpinBox);
    sb.addObject(m_configUi->point1YSpinBox);
    sb.addObject(m_configUi->point1ZSpinBox);
    sb.addObject(m_configUi->point2XSpinBox);
    sb.addObject(m_configUi->point2YSpinBox);
    sb.addObject(m_configUi->point2ZSpinBox);

    m_configUi->centerXSpinBox->setValue(m_sphereCenter[0]);
    m_configUi->centerYSpinBox->setValue(m_sphereCenter[1]);
    m_configUi->centerZSpinBox->setValue(m_sphereCenter[2]);
    m_configUi->point1XSpinBox->setValue(m_linePoint1[0]);
    m_configUi->point1YSpinBox->setValue(m_linePoint1[1]);
    m_configUi->point1ZSpinBox->setValue(m_linePoint1[2]);
    m_configUi->point2XSpinBox->setValue(m_linePoint2[0]);
    m_configUi->point2YSpinBox->setValue(m_linePoint2[1]);
    m_configUi->point2ZSpinBox->setValue(m_linePoint2[2]);

    showConfigGroupBox(m_configUi->sourceCombo->currentIndex());

    if (m_source == SourceType::Point)
        updatePointActor();
    else if (m_source == SourceType::Line)
        updateLineActor();
}

void cvStreamlineFilter::showConfigGroupBox(int index) {
    if (index == 0) {
        m_configUi->configPointGroupBox->show();
        m_configUi->configLineGroupBox->hide();

        if (m_lineWidget) m_lineWidget->Off();
    } else if (index == 1) {
        m_configUi->configLineGroupBox->show();
        m_configUi->configPointGroupBox->hide();

        if (m_pointWidget) m_pointWidget->Off();
    }

    update();
}

void cvStreamlineFilter::updatePointActor() {
    if (!m_pointWidget) {
        m_pointWidget = vtkPointWidget::New();
        VtkUtils::PointWidgetObserver* observer =
                new VtkUtils::PointWidgetObserver(this);
        connect(observer, SIGNAL(positionChanged(double*)), this,
                SLOT(onPointPositionChanged(double*)));
        observer->attach(m_pointWidget);
        m_pointWidget->SetProp3D(m_modelActor);
        m_pointWidget->SetInteractor(getInteractor());
        m_pointWidget->PlaceWidget();
        m_pointWidget->SetPlaceFactor(1.0);
        m_pointWidget->AllOff();
        m_pointWidget->GetProperty()->SetPointSize(4);
    }
    m_pointWidget->SetPosition(m_sphereCenter);
    m_pointWidget->On();
    update();
}

void cvStreamlineFilter::updateLineActor() {
    if (!m_lineWidget) {
        m_lineWidget = vtkLineWidget::New();
        VtkUtils::LineWidgetObserver* observer =
                new VtkUtils::LineWidgetObserver(this);
        connect(observer, SIGNAL(pointsChanged(double*, double*)), this,
                SLOT(onLinePointsChanged(double*, double*)));
        observer->attach(m_lineWidget);
        m_lineWidget->SetProp3D(m_modelActor);
        m_lineWidget->SetInteractor(getInteractor());
        m_lineWidget->SetPlaceFactor(1.0);
        m_lineWidget->SetClampToBounds(0);
        m_lineWidget->PlaceWidget();
    }
    m_lineWidget->SetPoint1(m_linePoint1);
    m_lineWidget->SetPoint2(m_linePoint2);
    m_lineWidget->On();
    update();
}

void cvStreamlineFilter::on_sourceCombo_currentIndexChanged(int index) {
    showConfigGroupBox(index);

    m_source = static_cast<SourceType>(index);
    if (m_source == Point)
        updatePointActor();
    else if (m_source)
        updateLineActor();

    apply();
}

void cvStreamlineFilter::on_numOfPointsSpinBox_valueChanged(int arg1) {
    m_numberOfPoints = arg1;
    apply();
}

void cvStreamlineFilter::on_radiusSpinBox_valueChanged(double arg1) {
    m_radius = arg1;
    apply();
}

void cvStreamlineFilter::on_centerXSpinBox_valueChanged(double arg1) {
    m_sphereCenter[0] = arg1;
    updatePointActor();
    apply();
}

void cvStreamlineFilter::on_centerYSpinBox_valueChanged(double arg1) {
    m_sphereCenter[1] = arg1;
    updatePointActor();
    apply();
}

void cvStreamlineFilter::on_centerZSpinBox_valueChanged(double arg1) {
    m_sphereCenter[2] = arg1;
    updatePointActor();
    apply();
}

void cvStreamlineFilter::on_integratorTypeCombo_currentIndexChanged(int index) {
    m_integratorType = index;
    apply();
}

void cvStreamlineFilter::on_integrationDirectionCombo_currentIndexChanged(
        int index) {
    m_integrationDirection = index;
    apply();
}

void cvStreamlineFilter::on_point1XSpinBox_valueChanged(double arg1) {
    m_linePoint1[0] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_point1YSpinBox_valueChanged(double arg1) {
    m_linePoint1[1] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_point1ZSpinBox_valueChanged(double arg1) {
    m_linePoint1[2] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_point2XSpinBox_valueChanged(double arg1) {
    m_linePoint2[0] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_point2YSpinBox_valueChanged(double arg1) {
    m_linePoint2[1] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_point2ZSpinBox_valueChanged(double arg1) {
    m_linePoint2[2] = arg1;
    updateLineActor();
    apply();
}

void cvStreamlineFilter::on_displayModeCombo_currentIndexChanged(int index) {
    m_displayMode = static_cast<DisplayMode>(index);

    switch (index) {
        case None:
            m_configUi->configTubeGroupBox->hide();
            m_configUi->configRuledSurfaceGroupBox->hide();
            break;

        case Tube:
            m_configUi->configTubeGroupBox->show();
            m_configUi->configRuledSurfaceGroupBox->hide();
            break;

        case Surface:
            m_configUi->configTubeGroupBox->hide();
            m_configUi->configRuledSurfaceGroupBox->show();
            break;
    }

    apply();
}

void cvStreamlineFilter::on_displayEffectCombo_currentIndexChanged(int index) {
    m_displayEffect = static_cast<DisplayEffect>(index);
    applyDisplayEffect();
}

void cvStreamlineFilter::on_numOfSidesSpinBox_valueChanged(int arg1) {
    m_tubeParams.numberOfSides = arg1;
    apply();
}

void cvStreamlineFilter::on_radiusFactorSpinBox_valueChanged(double arg1) {
    m_tubeParams.radiusFactor = arg1;
    apply();
}

void cvStreamlineFilter::on_cappingCheckBox_toggled(bool checked) {
    m_tubeParams.capping = checked;
    apply();
}

void cvStreamlineFilter::on_distanceFactorSpinBox_valueChanged(int arg1) {
    m_ruledSurfaceParams.distanceFactor = arg1;
    apply();
}

void cvStreamlineFilter::on_onRatioSpinBox_valueChanged(int arg1) {
    m_ruledSurfaceParams.onRatio = arg1;
    apply();
}

void cvStreamlineFilter::on_offsetSpinBox_valueChanged(int arg1) {
    m_ruledSurfaceParams.offset = arg1;
    apply();
}

void cvStreamlineFilter::on_passLinesCheckBox_toggled(bool checked) {
    m_ruledSurfaceParams.passLines = checked;
    apply();
}

void cvStreamlineFilter::on_closeSurfaceCheckBox_toggled(bool checked) {
    m_ruledSurfaceParams.closeSurface = checked;
    apply();
}

void cvStreamlineFilter::on_orientLoopsCheckBox_toggled(bool checked) {
    m_ruledSurfaceParams.orientLoops = checked;
    apply();
}

void cvStreamlineFilter::on_ruledModeCombo_currentIndexChanged(int index) {
    m_ruledSurfaceParams.ruledMode = index;
    apply();
}

void cvStreamlineFilter::on_resolutionSpinBox_valueChanged(int arg1) {
    //    m_ruledSurfaceParams.resolution = arg1;
    //    apply();
}

void cvStreamlineFilter::onLinePointsChanged(double* point1, double* point2) {
    if (Utils::ArrayComparator<double>()(m_linePoint1, point1) &&
        Utils::ArrayComparator<double>()(m_linePoint2, point2))
        return;

    Utils::ArrayAssigner<double>()(m_linePoint1, point1);
    Utils::ArrayAssigner<double>()(m_linePoint2, point2);
    updateLineActor();
    apply();
}

void cvStreamlineFilter::onPointPositionChanged(double* position) {
    if (Utils::ArrayComparator<double>()(m_sphereCenter, position)) return;

    Utils::ArrayAssigner<double>()(m_sphereCenter, position);
    updatePointActor();
    apply();
}
