// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceViewWidget.h"

#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>

#include <QMouseEvent>
#include <QResizeEvent>
#include <QVBoxLayout>
#include <QVTKOpenGLNativeWidget.h>
#include <QWheelEvent>

#include <vtkCamera.h>
#include <vtkCubeAxesActor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkNew.h>
#include <vtkProp.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <cmath>
#include <cstdio>

static const char* kViewLabels[] = {"Top View", "Right Side View",
                                    "Front View", ""};

struct vtkOrthoSliceViewWidget::Impl {
    QVTKOpenGLNativeWidget* vtkWidget = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> renderers[4];
    vtkSmartPointer<vtkTextActor> annotations[4];
    vtkSmartPointer<vtkTextActor> subAnnotations[4];
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> sliceAxes2D[3];
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> sliceAxes3D;
    vtkSmartPointer<vtkCubeAxesActor> gridAxes[3];
    double slicePos[3] = {0.0, 0.0, 0.0};
    double geomBounds[6] = {-1, 1, -1, 1, -1, 1};
};

vtkOrthoSliceViewWidget::vtkOrthoSliceViewWidget(QWidget* parent)
    : QWidget(parent), d(new Impl) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    d->vtkWidget = new QVTKOpenGLNativeWidget(this);
    layout->addWidget(d->vtkWidget);

    d->renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    d->vtkWidget->setRenderWindow(d->renderWindow);

    double bgColor[3] = {0.15, 0.15, 0.15};
    for (int i = 0; i < 4; ++i) {
        d->renderers[i] = vtkSmartPointer<vtkRenderer>::New();
        d->renderers[i]->SetBackground(bgColor[0], bgColor[1], bgColor[2]);
        d->renderWindow->AddRenderer(d->renderers[i]);
    }

    // Top View (XZ plane) — camera looks down -Y
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetPosition(0, 1, 0);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetViewUp(0, 0, 1);

    // Right Side View (YZ plane) — camera looks along -X
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetPosition(-1, 0, 0);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetViewUp(0, 0, 1);

    // Front View (XY plane) — camera looks along -Z
    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetPosition(0, 0, -1);
    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetViewUp(0, 1, 0);

    // Perspective 3D view — default
    d->renderers[PERSPECTIVE_VIEW]->GetActiveCamera()->SetParallelProjection(0);

    for (int i = 0; i < 4; ++i) {
        d->annotations[i] = vtkSmartPointer<vtkTextActor>::New();
        d->annotations[i]->SetInput(kViewLabels[i]);
        d->annotations[i]->GetTextProperty()->SetFontSize(12);
        d->annotations[i]->GetTextProperty()->SetColor(0.8, 0.8, 0.8);
        d->annotations[i]->GetTextProperty()->SetBold(1);
        d->annotations[i]->GetTextProperty()->SetShadow(1);
        d->annotations[i]->GetPositionCoordinate()
                ->SetCoordinateSystemToNormalizedViewport();
        d->annotations[i]->GetPositionCoordinate()->SetValue(0.01, 0.02);
        d->renderers[i]->AddActor2D(d->annotations[i]);

        d->subAnnotations[i] = vtkSmartPointer<vtkTextActor>::New();
        d->subAnnotations[i]->SetInput("");
        d->subAnnotations[i]->GetTextProperty()->SetFontSize(10);
        d->subAnnotations[i]->GetTextProperty()->SetColor(0.6, 0.6, 0.6);
        d->subAnnotations[i]->GetTextProperty()->SetShadow(1);
        d->subAnnotations[i]->GetPositionCoordinate()
                ->SetCoordinateSystemToNormalizedViewport();
        d->subAnnotations[i]->GetPositionCoordinate()->SetValue(0.01, 0.08);
        d->renderers[i]->AddActor2D(d->subAnnotations[i]);
    }

    // Viewports (ParaView quad pattern)
    d->renderers[TOP_VIEW]->SetViewport(0.0, 0.5, 0.5, 1.0);
    d->renderers[SIDE_VIEW]->SetViewport(0.5, 0.5, 1.0, 1.0);
    d->renderers[FRONT_VIEW]->SetViewport(0.0, 0.0, 0.5, 0.5);
    d->renderers[PERSPECTIVE_VIEW]->SetViewport(0.5, 0.0, 1.0, 0.5);

    // Crosshair axes (ParaView SlicePositionAxes2D/3D pattern)
    // Colors: X=red(1,0,0), Y=yellow(1,1,0), Z=blue(0,0,1)
    for (int i = 0; i < 3; ++i) {
        d->sliceAxes2D[i] =
                vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor>::New();
        d->sliceAxes2D[i]->SetComputeNormals(0);
        d->sliceAxes2D[i]->SetPickable(1);
        d->sliceAxes2D[i]->SetUseBounds(true);
        d->sliceAxes2D[i]->SetScale(10, 10, 10);
        d->renderers[i]->AddActor(d->sliceAxes2D[i]);
    }
    d->sliceAxes3D =
            vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor>::New();
    d->sliceAxes3D->SetComputeNormals(0);
    d->sliceAxes3D->SetPickable(0);
    d->sliceAxes3D->SetUseBounds(true);
    d->sliceAxes3D->SetScale(10, 10, 10);
    d->renderers[PERSPECTIVE_VIEW]->AddActor(d->sliceAxes3D);

    const double axisColors[3][3] = {
            {1.0, 0.4, 0.4},
            {0.4, 1.0, 0.4},
            {0.4, 0.6, 1.0}};
    const char* axisLabels[3][3] = {
            {"X", "Z", ""},
            {"Y", "Z", ""},
            {"X", "Y", ""}};
    for (int i = 0; i < 3; ++i) {
        d->gridAxes[i] = vtkSmartPointer<vtkCubeAxesActor>::New();
        d->gridAxes[i]->SetBounds(d->geomBounds);
        d->gridAxes[i]->SetCamera(d->renderers[i]->GetActiveCamera());
        d->gridAxes[i]->SetFlyModeToStaticTriad();
        d->gridAxes[i]->SetGridLineLocation(
                vtkCubeAxesActor::VTK_GRID_LINES_ALL);
        d->gridAxes[i]->DrawXGridlinesOn();
        d->gridAxes[i]->DrawYGridlinesOn();
        d->gridAxes[i]->DrawZGridlinesOff();
        d->gridAxes[i]->SetXTitle(axisLabels[i][0]);
        d->gridAxes[i]->SetYTitle(axisLabels[i][1]);
        d->gridAxes[i]->SetZTitle(axisLabels[i][2]);
        d->gridAxes[i]->GetTitleTextProperty(0)->SetColor(
                axisColors[0][0], axisColors[0][1], axisColors[0][2]);
        d->gridAxes[i]->GetTitleTextProperty(1)->SetColor(
                axisColors[1][0], axisColors[1][1], axisColors[1][2]);
        d->gridAxes[i]->GetTitleTextProperty(2)->SetColor(
                axisColors[2][0], axisColors[2][1], axisColors[2][2]);
        d->gridAxes[i]->GetLabelTextProperty(0)->SetFontSize(9);
        d->gridAxes[i]->GetLabelTextProperty(1)->SetFontSize(9);
        d->gridAxes[i]->GetLabelTextProperty(2)->SetFontSize(9);
        d->gridAxes[i]->GetTitleTextProperty(0)->SetFontSize(10);
        d->gridAxes[i]->GetTitleTextProperty(1)->SetFontSize(10);
        d->gridAxes[i]->GetTitleTextProperty(2)->SetFontSize(10);
        d->gridAxes[i]->SetLabelOffset(5);
        d->gridAxes[i]->SetTitleOffset(10);
        d->renderers[i]->AddActor(d->gridAxes[i]);
    }

    d->vtkWidget->installEventFilter(this);
}

vtkOrthoSliceViewWidget::~vtkOrthoSliceViewWidget() {
    for (auto& r : d->renderers) {
        if (r && d->renderWindow) d->renderWindow->RemoveRenderer(r);
    }
    delete d;
}

vtkRenderer* vtkOrthoSliceViewWidget::getRenderer(int index) const {
    if (index >= 0 && index < 4) return d->renderers[index];
    return nullptr;
}

vtkRenderer* vtkOrthoSliceViewWidget::mainRenderer() const {
    return d->renderers[PERSPECTIVE_VIEW];
}

QVTKOpenGLNativeWidget* vtkOrthoSliceViewWidget::vtkWidget() const {
    return d->vtkWidget;
}

void vtkOrthoSliceViewWidget::setSlicePosition(double x, double y, double z) {
    d->slicePos[0] = x;
    d->slicePos[1] = y;
    d->slicePos[2] = z;

    for (int i = 0; i < 3; ++i) {
        auto* cam = d->renderers[i]->GetActiveCamera();
        double fp[3], pos[3];
        cam->GetFocalPoint(fp);
        cam->GetPosition(pos);
        double dir[3] = {pos[0] - fp[0], pos[1] - fp[1], pos[2] - fp[2]};
        double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                               dir[2] * dir[2]);
        if (len < 1e-10) len = 1.0;
        dir[0] /= len;
        dir[1] /= len;
        dir[2] /= len;

        cam->SetFocalPoint(x, y, z);
        cam->SetPosition(x + dir[0] * len, y + dir[1] * len,
                         z + dir[2] * len);
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "Top View (Y=%.5g)", d->slicePos[1]);
    d->annotations[TOP_VIEW]->SetInput(buf);
    snprintf(buf, sizeof(buf), "Z=%.5g, X=%.5g", d->slicePos[2],
             d->slicePos[0]);
    d->subAnnotations[TOP_VIEW]->SetInput(buf);

    snprintf(buf, sizeof(buf), "Right Side View (X=%.5g)", d->slicePos[0]);
    d->annotations[SIDE_VIEW]->SetInput(buf);
    snprintf(buf, sizeof(buf), "Y=%.5g, Z=%.5g", d->slicePos[1],
             d->slicePos[2]);
    d->subAnnotations[SIDE_VIEW]->SetInput(buf);

    snprintf(buf, sizeof(buf), "Front View (Z=%.5g)", d->slicePos[2]);
    d->annotations[FRONT_VIEW]->SetInput(buf);
    snprintf(buf, sizeof(buf), "X=%.5g, Y=%.5g", d->slicePos[0],
             d->slicePos[1]);
    d->subAnnotations[FRONT_VIEW]->SetInput(buf);

    // Move crosshair axes (ParaView SetSlicePosition pattern)
    // Small offset avoids Z-fighting with the slice plane
    d->sliceAxes2D[TOP_VIEW]->SetPosition(x, y + 0.01, z);
    d->sliceAxes2D[SIDE_VIEW]->SetPosition(x + 0.01, y, z);
    d->sliceAxes2D[FRONT_VIEW]->SetPosition(x, y, z + 0.01);
    d->sliceAxes3D->SetPosition(x, y, z);

    render();
}

void vtkOrthoSliceViewWidget::resetCameras() {
    for (int i = 0; i < 4; ++i) {
        d->renderers[i]->ResetCamera();
        d->renderers[i]->ResetCameraClippingRange();
    }
    render();
}

void vtkOrthoSliceViewWidget::render() {
    if (d->renderWindow) {
        d->renderWindow->Render();
    }
}

void vtkOrthoSliceViewWidget::setGeometryBounds(const double bounds[6]) {
    for (int i = 0; i < 6; ++i) d->geomBounds[i] = bounds[i];
    for (int i = 0; i < 3; ++i) {
        if (d->gridAxes[i]) d->gridAxes[i]->SetBounds(bounds);
    }
    double widths[3] = {bounds[1] - bounds[0], bounds[3] - bounds[2],
                        bounds[5] - bounds[4]};
    double maxW = std::max({widths[0], widths[1], widths[2]});
    if (maxW < 1e-10) maxW = 1.0;
    double scale = maxW * 2.0;
    for (int i = 0; i < 3; ++i) {
        if (d->sliceAxes2D[i]) d->sliceAxes2D[i]->SetScale(scale, scale, scale);
    }
    if (d->sliceAxes3D) d->sliceAxes3D->SetScale(scale, scale, scale);
}

void vtkOrthoSliceViewWidget::addActorToAll(vtkProp* actor) {
    if (!actor) return;
    for (int i = 0; i < 4; ++i) {
        d->renderers[i]->AddViewProp(actor);
    }
}

void vtkOrthoSliceViewWidget::getSlicePosition(double pos[3]) const {
    pos[0] = d->slicePos[0];
    pos[1] = d->slicePos[1];
    pos[2] = d->slicePos[2];
}

int vtkOrthoSliceViewWidget::hitTestViewIndex(const QPoint& pos) const {
    if (!d->vtkWidget) return -1;
    int w = d->vtkWidget->width();
    int h = d->vtkWidget->height();
    if (w <= 0 || h <= 0) return -1;

    double nx = static_cast<double>(pos.x()) / w;
    double ny = 1.0 - static_cast<double>(pos.y()) / h;

    if (nx < 0.5 && ny >= 0.5) return TOP_VIEW;
    if (nx >= 0.5 && ny >= 0.5) return SIDE_VIEW;
    if (nx < 0.5 && ny < 0.5) return FRONT_VIEW;
    return PERSPECTIVE_VIEW;
}

bool vtkOrthoSliceViewWidget::eventFilter(QObject* obj, QEvent* event) {
    if (obj == d->vtkWidget) {
        if (event->type() == QEvent::Wheel) {
            auto* we = static_cast<QWheelEvent*>(event);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
            QPoint pos = we->position().toPoint();
#else
            QPoint pos = we->pos();
#endif
            int viewIdx = hitTestViewIndex(pos);
            if (viewIdx >= 0 && viewIdx < 3) {
                int axisMap[3] = {1, 0, 2};
                double step = m_sliceIncrements[axisMap[viewIdx]];
                double delta = (we->angleDelta().y() > 0) ? step : -step;
                double x = d->slicePos[0];
                double y = d->slicePos[1];
                double z = d->slicePos[2];

                switch (viewIdx) {
                    case TOP_VIEW:
                        y += delta;
                        break;
                    case SIDE_VIEW:
                        x += delta;
                        break;
                    case FRONT_VIEW:
                        z += delta;
                        break;
                    default:
                        break;
                }

                setSlicePosition(x, y, z);
                emit slicePositionChanged(x, y, z);
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonDblClick) {
            auto* me = static_cast<QMouseEvent*>(event);
            int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx >= 0 && viewIdx < 3) {
                auto* ren = d->renderers[viewIdx].GetPointer();
                if (!ren) return false;

                int w = d->vtkWidget->width();
                int h = d->vtkWidget->height();

                double vp[4];
                ren->GetViewport(vp);

                double normX = static_cast<double>(me->pos().x()) / w;
                double normY =
                        1.0 - static_cast<double>(me->pos().y()) / h;

                double localX = (normX - vp[0]) / (vp[2] - vp[0]);
                double localY = (normY - vp[1]) / (vp[3] - vp[1]);

                if (localX < 0 || localX > 1 || localY < 0 || localY > 1)
                    return false;

                double worldPt[4];
                ren->SetDisplayPoint(
                        localX * (vp[2] - vp[0]) * w,
                        localY * (vp[3] - vp[1]) * h, 0.0);
                ren->DisplayToWorld();
                ren->GetWorldPoint(worldPt);

                if (std::abs(worldPt[3]) > 1e-10) {
                    worldPt[0] /= worldPt[3];
                    worldPt[1] /= worldPt[3];
                    worldPt[2] /= worldPt[3];
                }

                double x = d->slicePos[0];
                double y = d->slicePos[1];
                double z = d->slicePos[2];

                switch (viewIdx) {
                    case TOP_VIEW:
                        x = worldPt[0];
                        z = worldPt[2];
                        break;
                    case SIDE_VIEW:
                        y = worldPt[1];
                        z = worldPt[2];
                        break;
                    case FRONT_VIEW:
                        x = worldPt[0];
                        y = worldPt[1];
                        break;
                    default:
                        break;
                }

                setSlicePosition(x, y, z);
                emit slicePositionChanged(x, y, z);
                return true;
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void vtkOrthoSliceViewWidget::setSliceIncrement(int axis, double step) {
    if (axis >= 0 && axis < 3) m_sliceIncrements[axis] = step;
}

double vtkOrthoSliceViewWidget::sliceIncrement(int axis) const {
    return (axis >= 0 && axis < 3) ? m_sliceIncrements[axis] : m_sliceStep;
}

void vtkOrthoSliceViewWidget::setAnnotationsVisible(bool visible) {
    if (m_annotationsVisible == visible) return;
    m_annotationsVisible = visible;
    for (int i = 0; i < 3; ++i) {
        d->annotations[i]->SetVisibility(visible ? 1 : 0);
        d->subAnnotations[i]->SetVisibility(visible ? 1 : 0);
    }
    render();
}

void vtkOrthoSliceViewWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
}
