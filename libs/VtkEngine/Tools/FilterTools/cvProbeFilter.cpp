// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvProbeFilter.h"

#include <VtkUtils/colorseries.h>
#include <VtkUtils/implicitplanewidgetobserver.h>
#include <VtkUtils/linewidgetobserver.h>
#include <VtkUtils/signalblocker.h>
#include <VtkUtils/spherewidgetobserver.h>
#include <VtkUtils/utils.h>
#include <VtkUtils/vtkutils.h>
#include <vtkActor.h>
#include <vtkBoxWidget.h>
#include <vtkImplicitPlaneWidget.h>
#include <vtkLODActor.h>
#include <vtkLineWidget.h>
#include <vtkPlane.h>
#include <vtkPlaneSource.h>
#include <vtkPlaneWidget.h>
#include <vtkProbeFilter.h>
#include <vtkProperty.h>
#include <vtkSphereWidget.h>
#include <vtkTransform.h>

#include "Tools/Common/qcustomplot.h"
#include "ui_cvGenericFilterDlg.h"
#include "ui_cvProbeFilterDlg.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvBBox.h>
#include <ecvDisplayTools.h>
#include <ecvFileUtils.h>

// QT
#include <QFileDialog>
#include <QFileInfo>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

cvProbeFilter::cvProbeFilter(QWidget* parent) : cvGenericFilter(parent) {
    setWindowTitle(tr("Probe"));
    createUi();
}

void cvProbeFilter::createUi() {
    cvGenericFilter::createUi();

    m_configUi = new Ui::cvProbeFilterDlg;
    setupConfigWidget(m_configUi);

    m_configUi->sphereGroupBox->hide();
    m_configUi->implicitPlaneGroupBox->hide();

    m_plotWidget = new QCustomPlot(this);
    m_plotWidget->xAxis2->setVisible(true);
    m_plotWidget->xAxis2->setTickLabels(false);
    m_plotWidget->yAxis2->setVisible(true);
    m_plotWidget->yAxis2->setTickLabels(false);
    m_plotWidget->legend->setVisible(true);
    m_plotWidget->legend->setFont(QFont("Helvetica", 9));
    m_plotWidget->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom |
                                  QCP::iSelectPlottables);
    connect(m_plotWidget->xAxis, SIGNAL(rangeChanged(QCPRange)),
            m_plotWidget->xAxis2, SLOT(setRange(QCPRange)));
    connect(m_plotWidget->yAxis, SIGNAL(rangeChanged(QCPRange)),
            m_plotWidget->yAxis2, SLOT(setRange(QCPRange)));
    m_plotWidget->clearGraphs();
    m_configUi->plotLayout->addWidget(m_plotWidget);
}

void cvProbeFilter::apply() {
    VtkUtils::vtkInitOnce(m_probe);
    m_probe->SetInputData(m_dataObject);

    switch (m_widgetType) {
        case WT_Line: {
            VTK_CREATE(vtkLineSource, ls);
            ls->SetPoint1(m_linePoint1);
            ls->SetPoint2(m_linePoint2);
            ls->Update();
            m_probe->SetSourceData(ls->GetOutput());
        } break;
        case WT_Sphere: {
            VTK_CREATE(vtkSphereSource, ss);
            ss->SetCenter(m_sphereCenter);
            ss->Update();
            m_probe->SetSourceData(ss->GetOutput());
        } break;
        case WT_Box:
            break;
        case WT_ImplicitPlane: {
            VTK_CREATE(vtkPlaneSource, ps);
            ps->SetOrigin(m_planeOrigin);
            ps->SetNormal(m_planeNormal);
            ps->Update();
            m_probe->SetSourceData(ps->GetOutput());

        } break;
    }

    m_probe->Update();

    vtkPointData* pointData = m_probe->GetOutput()->GetPointData();

    Utils::ColorSeries colorSeries;
    colorSeries.setScheme(Utils::ColorSeries::Cool);

    m_plotWidget->clearGraphs();
    int numOfArray = pointData->GetNumberOfArrays();
    for (int i = 0; i < numOfArray; ++i) {
        QString arrayName(pointData->GetArrayName(i));
        // skip this kind of data
        if (arrayName.compare("vtkValidPointMask", Qt::CaseInsensitive) == 0)
            continue;

        m_plotWidget->addGraph();
        m_plotWidget->graph(i)->setName(arrayName);
        m_plotWidget->graph(i)->setPen(colorSeries.nextColor());
        vtkDataArray* da = pointData->GetArray(i);
        int numOfTuples = da->GetNumberOfTuples();
        QVector<double> x(numOfTuples), y(numOfTuples);

#ifdef USE_TBB
        tbb::parallel_for(0, numOfTuples,
                          [&](int dataIndex)
#else

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int dataIndex = 0; dataIndex < numOfTuples; ++dataIndex)
#endif
                          {
                              x[dataIndex] = dataIndex;
                              double value = da->GetTuple1(dataIndex);
                              y[dataIndex] = value;
                          }
#ifdef USE_TBB
        );
#endif

        m_plotWidget->graph(i)->setData(x, y);
        m_plotWidget->graph(i)->rescaleAxes(true);
    }

    m_plotWidget->replot();
    update();
}

void cvProbeFilter::showInteractor(bool state) {
    switch (m_widgetType) {
        case WidgetType::WT_Line:
            if (m_lineWidget) {
                state ? m_lineWidget->On() : safeOff(m_lineWidget);
            }
            break;
        case WidgetType::WT_Box:
            if (m_boxWidget) {
                state ? m_boxWidget->On() : safeOff(m_boxWidget);
            }
            break;
        case WidgetType::WT_ImplicitPlane:
            if (m_implicitPlaneWidget) {
                state ? m_implicitPlaneWidget->On()
                      : safeOff(m_implicitPlaneWidget);
            }
            break;
        case WidgetType::WT_Sphere:
            if (m_sphereWidget) {
                state ? m_sphereWidget->On() : safeOff(m_sphereWidget);
            }
            break;
        default:
            break;
    }
}

void cvProbeFilter::getInteractorBounds(ccBBox& bbox) {
    double bounds[6];

    bool valid = true;
    switch (m_widgetType) {
        case WidgetType::WT_Line:
            if (m_lineWidget) {
                // xmin, xmax, ymin, ymax, zmin, zmax
                m_lineWidget->GetProp3D()->GetBounds(bounds);
            }
            break;
        case WidgetType::WT_Box:
            if (m_boxWidget) {
                m_boxWidget->GetProp3D()->GetBounds(bounds);
            }
            break;
        case WidgetType::WT_ImplicitPlane:
            if (m_implicitPlaneWidget) {
                m_implicitPlaneWidget->GetProp3D()->GetBounds(bounds);
            }
            break;
        case WidgetType::WT_Sphere:
            if (m_sphereWidget) {
                // xmin, xmax, ymin, ymax, zmin, zmax
                m_sphereWidget->GetProp3D()->GetBounds(bounds);
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

void cvProbeFilter::getInteractorTransformation(ccGLMatrixd& trans) {
    switch (m_widgetType) {
        case WidgetType::WT_Line:
            if (m_lineWidget) {
                m_lineWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        case WidgetType::WT_Box:
            if (m_boxWidget) {
                m_boxWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        case WidgetType::WT_ImplicitPlane:
            if (m_implicitPlaneWidget) {
                m_implicitPlaneWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        case WidgetType::WT_Sphere:
            if (m_sphereWidget) {
                m_sphereWidget->GetProp3D()->GetMatrix(trans.data());
            }
            break;
        default:
            break;
    }
}

void cvProbeFilter::shift(const CCVector3d& v) {
    switch (m_widgetType) {
        case WidgetType::WT_Line:
            if (m_lineWidget) {
                double newPos1[3];
                double newPos2[3];
                CCVector3d::vadd(v.u, m_linePoint1, newPos1);
                CCVector3d::vadd(v.u, m_linePoint2, newPos2);
                onLineWidgetPointsChanged(newPos1, newPos2);
            }

            break;
        case WidgetType::WT_Box:
            if (m_boxWidget) {
                VTK_CREATE(vtkTransform, trans);
                m_boxWidget->GetTransform(trans);
                trans->Translate(v.u);
                m_boxWidget->SetTransform(trans);
                apply();
            }
            break;
        case WidgetType::WT_ImplicitPlane:
            if (m_implicitPlaneWidget) {
                // Modify and update planeWidget
            }
            break;
        case WidgetType::WT_Sphere:
            if (m_sphereWidget) {
                double newPos[3];
                CCVector3d::vadd(v.u, m_sphereCenter, newPos);
                onSphereWidgetCenterChanged(newPos);
            }
            break;
        default:
            break;
    }

    showProbeWidget();
}

void cvProbeFilter::clearAllActor() {
    if (m_configUi) {
        delete m_configUi;
        m_configUi = nullptr;
    }

    if (m_plotWidget) {
        delete m_plotWidget;
        m_plotWidget = nullptr;
    }

    if (m_lineWidget) safeOff(m_lineWidget);
    if (m_sphereWidget) safeOff(m_sphereWidget);
    if (m_boxWidget) safeOff(m_boxWidget);
    if (m_implicitPlaneWidget) safeOff(m_implicitPlaneWidget);

    cvGenericFilter::clearAllActor();
}

void cvProbeFilter::initFilter() {
    cvGenericFilter::initFilter();
    setDisplayEffect(DisplayEffect::Transparent);
}

void cvProbeFilter::dataChanged() { apply(); }

ccHObject* cvProbeFilter::getOutput() {
    if (!m_probe) return nullptr;
    if (m_plotWidget) {
        // default output path (+ filename)
        QString filters = "*.png;;*.bmp;;*.pdf";
        QString selectedFilter = "*.png";
        QString selectedFilename = QFileDialog::getSaveFileName(
                ecvDisplayTools::GetCurrentScreen(),
                tr("export current plot figure"),
                ecvFileUtils::defaultDocPath(), filters, &selectedFilter);

        if (selectedFilename.isEmpty()) {
            // process cancelled by the user
            return nullptr;
        }

        QFileInfo fileInfo(selectedFilename);
        QString suffix = fileInfo.suffix();
        bool result = false;
        if (suffix.compare("png", Qt::CaseInsensitive) == 0) {
            result = m_plotWidget->savePng(selectedFilename, 1920, 1080);
        }
        /*else if (suffix.compare("jpg", Qt::CaseInsensitive) == 0)
        {
                result = m_plotWidget->saveJpg(selectedFilename, 1920, 1080);
        }*/
        else if (suffix.compare("bmp", Qt::CaseInsensitive) == 0) {
            result = m_plotWidget->saveBmp(selectedFilename, 1920, 1080);
        } else if (suffix.compare("pdf", Qt::CaseInsensitive) == 0) {
            result = m_plotWidget->savePdf(selectedFilename, true, 1920, 1080);
        }

        if (result) {
            CVLog::Print(tr("Probe figure has been saved in %1")
                                 .arg(selectedFilename));
        } else {
            CVLog::Warning(tr("failed to export probe figure!"));
        }
    }

    return nullptr;
}

void cvProbeFilter::on_sourceCombo_currentIndexChanged(int index) {
    switch (index) {
        case 0:
            m_configUi->lineGroupBox->show();
            m_configUi->sphereGroupBox->hide();
            m_configUi->implicitPlaneGroupBox->hide();
            break;

        case 1:
            m_configUi->lineGroupBox->hide();
            m_configUi->sphereGroupBox->show();
            m_configUi->implicitPlaneGroupBox->hide();
            break;

        case 3:
            m_configUi->implicitPlaneGroupBox->show();
            m_configUi->lineGroupBox->hide();
            m_configUi->sphereGroupBox->hide();
            break;
    }

    m_widgetType = static_cast<WidgetType>(index);
    showProbeWidget();

    apply();
}

void cvProbeFilter::onLineWidgetPointsChanged(double* point1, double* point2) {
    if (Utils::ArrayComparator<double>()(m_linePoint1, point1) &&
        Utils::ArrayComparator<double>()(m_linePoint2, point2)) {
#ifdef QT_DEBUG
        CVLog::Warning(QString("line point1 & ponit2 are not changed."));
#endif
        return;
    }

    VtkUtils::SignalBlocker sb(m_configUi->linePoint1XSpinBox);
    sb.addObject(m_configUi->linePoint1YSpinBox);
    sb.addObject(m_configUi->linePoint1ZSpinBox);
    sb.addObject(m_configUi->linePoint2XSpinBox);
    sb.addObject(m_configUi->linePoint2YSpinBox);
    sb.addObject(m_configUi->linePoint2ZSpinBox);

    m_configUi->linePoint1XSpinBox->setValue(point1[0]);
    m_configUi->linePoint1YSpinBox->setValue(point1[1]);
    m_configUi->linePoint1ZSpinBox->setValue(point1[2]);
    m_configUi->linePoint2ZSpinBox->setValue(point2[0]);
    m_configUi->linePoint2ZSpinBox->setValue(point2[1]);
    m_configUi->linePoint2ZSpinBox->setValue(point2[2]);

    Utils::ArrayAssigner<double>()(m_linePoint1, point1);
    Utils::ArrayAssigner<double>()(m_linePoint2, point2);
    apply();
}

void cvProbeFilter::onSphereWidgetCenterChanged(double* center) {
    if (Utils::ArrayComparator<double>()(m_sphereCenter, center)) {
#ifdef QT_DEBUG
        CVLog::Warning(QString("sphere center is not changed."));
#endif
        return;
    }

    VtkUtils::SignalBlocker sb(m_configUi->sphereCenterXSpinBox);
    sb.addObject(m_configUi->sphereCenterYSpinBox);
    sb.addObject(m_configUi->sphereCenterZSpinBox);
    m_configUi->sphereCenterXSpinBox->setValue(center[0]);
    m_configUi->sphereCenterYSpinBox->setValue(center[1]);
    m_configUi->sphereCenterZSpinBox->setValue(center[2]);

    Utils::ArrayAssigner<double>()(m_sphereCenter, center);
    apply();
}

void cvProbeFilter::onImplicitPlaneWidgetOriginChanged(double* origin) {
    if (Utils::ArrayComparator<double>()(m_planeOrigin, origin)) {
        CVLog::Warning(QString("plane origin is not changed."));
        return;
    }

    VtkUtils::SignalBlocker sb(m_configUi->originXSpinBox);
    sb.addObject(m_configUi->originYSpinBox);
    sb.addObject(m_configUi->originZSpinBox);

    m_configUi->originXSpinBox->setValue(origin[0]);
    m_configUi->originYSpinBox->setValue(origin[1]);
    m_configUi->originZSpinBox->setValue(origin[2]);

    Utils::ArrayAssigner<double>()(m_planeOrigin, origin);
    apply();
}

void cvProbeFilter::onImplicitPlaneWidgetNormalChanged(double* normal) {
    if (Utils::ArrayComparator<double>()(m_planeNormal, normal)) {
#ifdef QT_DEBUG
        CVLog::Warning(QString("plane normal is not changed."));
#endif
        return;
    }

    VtkUtils::SignalBlocker sb(m_configUi->normalXSpinBox);
    sb.addObject(m_configUi->normalYSpinBox);
    sb.addObject(m_configUi->normalZSpinBox);

    m_configUi->normalXSpinBox->setValue(normal[0]);
    m_configUi->normalYSpinBox->setValue(normal[1]);
    m_configUi->normalZSpinBox->setValue(normal[2]);

    Utils::ArrayAssigner<double>()(m_planeNormal, normal);
    apply();
}

void cvProbeFilter::modelReady() {
    cvGenericFilter::modelReady();
    showProbeWidget();
}

void cvProbeFilter::showProbeWidget() {
    switch (m_widgetType) {
        case WT_Line:
            if (!m_lineWidget) {
                VtkUtils::vtkInitOnce(m_lineWidget);
                m_lineWidget->SetInteractor(getInteractor());
                m_lineWidget->SetPlaceFactor(1);
                m_lineWidget->SetClampToBounds(1);
                m_lineWidget->SetProp3D(m_modelActor);
                m_lineWidget->PlaceWidget();
                //            m_lineWidget->AddObserver(vtkCommand::EndInteractionEvent,
                //            new LineCallback);
                VtkUtils::LineWidgetObserver* observer =
                        new VtkUtils::LineWidgetObserver(this);
                connect(observer, SIGNAL(pointsChanged(double*, double*)), this,
                        SLOT(onLineWidgetPointsChanged(double*, double*)));
                observer->attach(m_lineWidget);
            }

            m_lineWidget->SetPoint1(m_linePoint1);
            m_lineWidget->SetPoint2(m_linePoint2);
            m_lineWidget->On();

            // disable the other two
            if (m_sphereWidget) m_sphereWidget->Off();
            if (m_boxWidget) m_boxWidget->Off();
            if (m_implicitPlaneWidget) m_implicitPlaneWidget->Off();

            break;

        case WT_Sphere:
            if (!m_sphereWidget) {
                VtkUtils::vtkInitOnce(m_sphereWidget);
                m_sphereWidget->SetInteractor(getInteractor());
                m_sphereWidget->SetPlaceFactor(1);
                m_sphereWidget->SetProp3D(m_modelActor);
                m_sphereWidget->PlaceWidget();
                VtkUtils::SphereWidgetObserver* observer =
                        new VtkUtils::SphereWidgetObserver(this);
                connect(observer, SIGNAL(centerChanged(double*)), this,
                        SLOT(onSphereWidgetCenterChanged(double*)));
                observer->attach(m_sphereWidget);
                m_sphereRadius = m_sphereWidget->GetRadius();
            }
            m_sphereWidget->On();
            m_sphereWidget->SetCenter(m_sphereCenter);
            m_sphereWidget->SetRadius(m_sphereRadius);

            // disable the other two
            if (m_lineWidget) m_lineWidget->Off();
            if (m_boxWidget) m_boxWidget->Off();
            if (m_implicitPlaneWidget) m_implicitPlaneWidget->Off();

            break;

        case WT_Box:
            if (!m_boxWidget) {
                VtkUtils::vtkInitOnce(m_boxWidget);
                m_boxWidget->SetInteractor(getInteractor());
                m_boxWidget->SetPlaceFactor(1);
                m_boxWidget->SetProp3D(m_modelActor);
                m_boxWidget->PlaceWidget();
            }
            m_boxWidget->On();

            // disable the other two
            if (m_lineWidget) m_lineWidget->Off();
            if (m_sphereWidget) m_sphereWidget->Off();
            if (m_implicitPlaneWidget) m_implicitPlaneWidget->Off();

            break;

        case WT_ImplicitPlane:
            if (!m_implicitPlaneWidget) {
                VtkUtils::vtkInitOnce(m_implicitPlaneWidget);
                m_implicitPlaneWidget->SetInteractor(getInteractor());
                m_implicitPlaneWidget->SetPlaceFactor(1);
                m_implicitPlaneWidget->SetProp3D(m_modelActor);
                m_implicitPlaneWidget->SetOutlineTranslation(
                        0);  // make the outline non-movable
                m_implicitPlaneWidget->GetPlaneProperty()->SetOpacity(0.7);
                m_implicitPlaneWidget->GetPlaneProperty()->SetColor(.0, .0, .0);

                double bounds[6];
                if (isValidPolyData()) {
                    vtkPolyData* polyData =
                            vtkPolyData::SafeDownCast(m_dataObject);
                    polyData->GetBounds(bounds);
                    m_implicitPlaneWidget->SetOrigin(polyData->GetCenter());
                } else if (isValidDataSet()) {
                    vtkDataSet* dataSet =
                            vtkDataSet::SafeDownCast(m_dataObject);
                    dataSet->GetBounds(bounds);
                    m_implicitPlaneWidget->SetOrigin(dataSet->GetCenter());
                }
                //            m_implicitPlaneWidget->SetOutsideBounds(0);
                m_implicitPlaneWidget->PlaceWidget(bounds);

                VtkUtils::ImplicitPlaneWidgetObserver* observer =
                        new VtkUtils::ImplicitPlaneWidgetObserver(this);
                connect(observer, SIGNAL(originChanged(double*)), this,
                        SLOT(onImplicitPlaneWidgetOriginChanged(double*)));
                connect(observer, SIGNAL(normalChanged(double*)), this,
                        SLOT(onImplicitPlaneWidgetNormalChanged(double*)));
                observer->attach(m_implicitPlaneWidget);
            }
            m_implicitPlaneWidget->On();
            m_implicitPlaneWidget->UpdatePlacement();

            // disable the other two
            if (m_lineWidget) m_lineWidget->Off();
            if (m_sphereWidget) m_sphereWidget->Off();
            if (m_boxWidget) m_boxWidget->Off();

            break;
    }

    applyDisplayEffect();
}

void cvProbeFilter::on_sphereRadius_valueChanged(double arg1) {
    m_sphereRadius = arg1;
    showProbeWidget();
    apply();
}
