// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceViewWidget.h"

#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>

#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QVTKOpenGLNativeWidget.h>
#include <QWheelEvent>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkProperty.h>
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
    layout->setSpacing(0);

    auto* decoratorBar = new QWidget(this);
    decoratorBar->setObjectName("OrthoDecoratorBar");
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(0, 0, 0, 0);
    decLayout->setSpacing(1);

    auto* showingLabel = new QLabel(QStringLiteral("<b>Showing  </b>"), decoratorBar);
    decLayout->addWidget(showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_sourceCombo->addItem(tr("None"));
    decLayout->addWidget(m_sourceCombo, 1);
    decLayout->addStretch(1);
    layout->addWidget(decoratorBar);

    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &vtkOrthoSliceViewWidget::onSourceComboChanged);
    m_sourceCombo->installEventFilter(this);

    // === Row 2: Slice position / step / annotations ===
    auto* sliceBar = new QWidget(this);
    sliceBar->setObjectName("OrthoSliceBar");
    auto* sliceLayout = new QHBoxLayout(sliceBar);
    sliceLayout->setContentsMargins(2, 1, 2, 1);
    sliceLayout->setSpacing(4);

    const char* sliceLabels[] = {"X:", "Y:", "Z:"};
    for (int i = 0; i < 3; ++i) {
        sliceLayout->addWidget(new QLabel(tr(sliceLabels[i]), sliceBar));
        m_sliceSpin[i] = new QDoubleSpinBox(sliceBar);
        m_sliceSpin[i]->setRange(-1e6, 1e6);
        m_sliceSpin[i]->setDecimals(4);
        m_sliceSpin[i]->setValue(0.0);
        m_sliceSpin[i]->setFixedWidth(80);
        sliceLayout->addWidget(m_sliceSpin[i]);
        connect(m_sliceSpin[i],
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [this]() {
                    if (m_updatingSpinners) return;
                    setSlicePosition(m_sliceSpin[0]->value(),
                                     m_sliceSpin[1]->value(),
                                     m_sliceSpin[2]->value());
                    emit slicePositionChanged(m_sliceSpin[0]->value(),
                                              m_sliceSpin[1]->value(),
                                              m_sliceSpin[2]->value());
                });
    }

    sliceLayout->addSpacing(8);
    sliceLayout->addWidget(new QLabel(tr("Step:"), sliceBar));
    m_stepSpin = new QDoubleSpinBox(sliceBar);
    m_stepSpin->setRange(0.001, 1000.0);
    m_stepSpin->setDecimals(3);
    m_stepSpin->setValue(1.0);
    m_stepSpin->setFixedWidth(65);
    sliceLayout->addWidget(m_stepSpin);
    connect(m_stepSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double val) {
                for (int i = 0; i < 3; ++i) {
                    m_sliceIncrements[i] = val;
                    m_sliceSpin[i]->setSingleStep(val);
                }
            });

    sliceLayout->addSpacing(8);
    m_annotCheck = new QCheckBox(tr("Annotations"), sliceBar);
    m_annotCheck->setChecked(true);
    sliceLayout->addWidget(m_annotCheck);
    connect(m_annotCheck, &QCheckBox::toggled, this,
            &vtkOrthoSliceViewWidget::setAnnotationsVisible);

    sliceLayout->addStretch(1);
    m_statusLabel = new QLabel(sliceBar);
    m_statusLabel->setContentsMargins(4, 0, 4, 0);
    sliceLayout->addWidget(m_statusLabel);

    layout->addWidget(sliceBar);

    // === Row 3: Display properties (ParaView Representation / Styling) ===
    auto* dispBar = new QWidget(this);
    dispBar->setObjectName("OrthoDisplayBar");
    auto* dispLayout = new QHBoxLayout(dispBar);
    dispLayout->setContentsMargins(2, 1, 2, 1);
    dispLayout->setSpacing(4);

    dispLayout->addWidget(new QLabel(tr("Repr:"), dispBar));
    m_reprCombo = new QComboBox(dispBar);
    m_reprCombo->addItems({tr("Slices"), tr("Surface"), tr("Wireframe"),
                           tr("Points"), tr("Surface With Edges")});
    m_reprCombo->setCurrentIndex(0);
    dispLayout->addWidget(m_reprCombo);
    connect(m_reprCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { applyDisplayProperties(); });

    dispLayout->addSpacing(6);
    dispLayout->addWidget(new QLabel(tr("Opacity:"), dispBar));
    m_opacitySlider = new QSlider(Qt::Horizontal, dispBar);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(100);
    m_opacitySlider->setFixedWidth(80);
    dispLayout->addWidget(m_opacitySlider);
    m_opacityLabel = new QLabel(QStringLiteral("1.0"), dispBar);
    m_opacityLabel->setFixedWidth(28);
    dispLayout->addWidget(m_opacityLabel);
    connect(m_opacitySlider, &QSlider::valueChanged, this, [this](int val) {
        m_opacityLabel->setText(QString::number(val / 100.0, 'f', 2));
        applyDisplayProperties();
    });

    dispLayout->addSpacing(6);
    dispLayout->addWidget(new QLabel(tr("Pt Size:"), dispBar));
    m_pointSizeSpin = new QSpinBox(dispBar);
    m_pointSizeSpin->setRange(1, 50);
    m_pointSizeSpin->setValue(2);
    m_pointSizeSpin->setFixedWidth(50);
    dispLayout->addWidget(m_pointSizeSpin);
    connect(m_pointSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int) { applyDisplayProperties(); });

    dispLayout->addSpacing(6);
    dispLayout->addWidget(new QLabel(tr("Line W:"), dispBar));
    m_lineWidthSpin = new QDoubleSpinBox(dispBar);
    m_lineWidthSpin->setRange(0.1, 10.0);
    m_lineWidthSpin->setSingleStep(0.5);
    m_lineWidthSpin->setValue(1.0);
    m_lineWidthSpin->setFixedWidth(55);
    dispLayout->addWidget(m_lineWidthSpin);
    connect(m_lineWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double) { applyDisplayProperties(); });

    dispLayout->addStretch(1);
    layout->addWidget(dispBar);

    // === Row 4: Coloring & Scalar Coloring (ParaView Properties panel) ===
    auto* colorBar = new QWidget(this);
    colorBar->setObjectName("OrthoColorBar");
    auto* colorLayout = new QHBoxLayout(colorBar);
    colorLayout->setContentsMargins(2, 1, 2, 1);
    colorLayout->setSpacing(4);

    colorLayout->addWidget(new QLabel(tr("Coloring:"), colorBar));
    m_coloringCombo = new QComboBox(colorBar);
    m_coloringCombo->addItems({tr("Solid Color"), tr("Points"), tr("Normals"),
                               tr("TCoords")});
    colorLayout->addWidget(m_coloringCombo);

    colorLayout->addSpacing(8);
    m_mapScalarsCheck = new QCheckBox(tr("Map Scalars"), colorBar);
    m_mapScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_mapScalarsCheck);

    m_interpScalarsCheck = new QCheckBox(tr("Interp Scalars"), colorBar);
    m_interpScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_interpScalarsCheck);

    colorLayout->addStretch(1);
    layout->addWidget(colorBar);

    // === Row 5: Lighting (ParaView Properties panel) ===
    auto* lightBar = new QWidget(this);
    lightBar->setObjectName("OrthoLightBar");
    auto* lightLayout = new QHBoxLayout(lightBar);
    lightLayout->setContentsMargins(2, 1, 2, 1);
    lightLayout->setSpacing(4);

    m_disableLightingCheck = new QCheckBox(tr("No Light"), lightBar);
    lightLayout->addWidget(m_disableLightingCheck);
    connect(m_disableLightingCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Diffuse:"), lightBar));
    m_diffuseSlider = new QSlider(Qt::Horizontal, lightBar);
    m_diffuseSlider->setRange(0, 100);
    m_diffuseSlider->setValue(100);
    m_diffuseSlider->setFixedWidth(60);
    lightLayout->addWidget(m_diffuseSlider);
    m_diffuseLabel = new QLabel(QStringLiteral("1.0"), lightBar);
    m_diffuseLabel->setFixedWidth(24);
    lightLayout->addWidget(m_diffuseLabel);
    connect(m_diffuseSlider, &QSlider::valueChanged, this, [this](int val) {
        m_diffuseLabel->setText(QString::number(val / 100.0, 'f', 1));
        applyDisplayProperties();
    });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Interp:"), lightBar));
    m_interpCombo = new QComboBox(lightBar);
    m_interpCombo->addItems({tr("Flat"), tr("Gouraud"), tr("Phong")});
    m_interpCombo->setCurrentIndex(1);
    lightLayout->addWidget(m_interpCombo);
    connect(m_interpCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Spec:"), lightBar));
    m_specularSlider = new QSlider(Qt::Horizontal, lightBar);
    m_specularSlider->setRange(0, 100);
    m_specularSlider->setValue(0);
    m_specularSlider->setFixedWidth(50);
    lightLayout->addWidget(m_specularSlider);
    m_specularLabel = new QLabel(QStringLiteral("0"), lightBar);
    m_specularLabel->setFixedWidth(20);
    lightLayout->addWidget(m_specularLabel);
    connect(m_specularSlider, &QSlider::valueChanged, this, [this](int val) {
        m_specularLabel->setText(QString::number(val / 100.0, 'f', 1));
        applyDisplayProperties();
    });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Power:"), lightBar));
    m_specPowerSpin = new QSpinBox(lightBar);
    m_specPowerSpin->setRange(1, 200);
    m_specPowerSpin->setValue(100);
    m_specPowerSpin->setFixedWidth(50);
    lightLayout->addWidget(m_specPowerSpin);
    connect(m_specPowerSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int) { applyDisplayProperties(); });

    lightLayout->addStretch(1);
    layout->addWidget(lightBar);

    d->vtkWidget = new QVTKOpenGLNativeWidget(this);
    layout->addWidget(d->vtkWidget, 1);

    d->renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    d->vtkWidget->setRenderWindow(d->renderWindow);

    // ParaView: orthographic panes start at (0.5, 0.5, 0.5) gray
    for (int i = 0; i < 4; ++i) {
        d->renderers[i] = vtkSmartPointer<vtkRenderer>::New();
        d->renderers[i]->SetBackground(0.5, 0.5, 0.5);
        d->renderWindow->AddRenderer(d->renderers[i]);
    }
    // 3D perspective uses darker background
    d->renderers[PERSPECTIVE_VIEW]->SetBackground(0.15, 0.15, 0.15);

    // ParaView camera directions: Side +X, Top +Y, Front +Z
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetPosition(0, 1, 0);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    d->renderers[TOP_VIEW]->GetActiveCamera()->SetViewUp(1, 0, 0);

    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetPosition(1, 0, 0);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetFocalPoint(0, 0, 0);
    d->renderers[SIDE_VIEW]->GetActiveCamera()->SetViewUp(0, 1, 0);

    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetParallelProjection(1);
    d->renderers[FRONT_VIEW]->GetActiveCamera()->SetPosition(0, 0, 1);
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

    // ParaView vtkPVCenterAxesActor LUT: X=red, Y=yellow, Z=blue
    const double axisColors[3][3] = {
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}};
    const char* axisLabels[3][3] = {
            {"X", "Z", ""},
            {"Y", "Z", ""},
            {"X", "Y", ""}};
    // ParaView OrthoSlice does NOT show dense grid overlays — only crosshairs.
    // Keep vtkCubeAxesActor hidden by default for clean alignment.
    for (int i = 0; i < 3; ++i) {
        d->gridAxes[i] = vtkSmartPointer<vtkCubeAxesActor>::New();
        d->gridAxes[i]->SetBounds(d->geomBounds);
        d->gridAxes[i]->SetCamera(d->renderers[i]->GetActiveCamera());
        d->gridAxes[i]->SetFlyModeToStaticTriad();
        d->gridAxes[i]->DrawXGridlinesOff();
        d->gridAxes[i]->DrawYGridlinesOff();
        d->gridAxes[i]->DrawZGridlinesOff();
        d->gridAxes[i]->XAxisVisibilityOff();
        d->gridAxes[i]->YAxisVisibilityOff();
        d->gridAxes[i]->ZAxisVisibilityOff();
        d->gridAxes[i]->SetVisibility(0);
        d->renderers[i]->AddActor(d->gridAxes[i]);
    }

    setAnnotationsVisible(true);

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

void vtkOrthoSliceViewWidget::updateSliceSpinners() {
    m_updatingSpinners = true;
    for (int i = 0; i < 3; ++i) {
        if (m_sliceSpin[i]) m_sliceSpin[i]->setValue(d->slicePos[i]);
    }
    m_updatingSpinners = false;
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

    updateSliceSpinners();
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
    if (obj == m_sourceCombo && event->type() == QEvent::MouseButtonPress) {
        refreshSourceCombo();
    }
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
            if (viewIdx == PERSPECTIVE_VIEW) {
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

void vtkOrthoSliceViewWidget::populateFromRenderer(vtkRenderer* sourceRenderer) {
    if (!sourceRenderer) return;

    auto* actors = sourceRenderer->GetActors();
    if (!actors) return;

    actors->InitTraversal();
    vtkActor* actor = nullptr;
    while ((actor = actors->GetNextActor())) {
        addActorToAll(actor);
    }

    auto* props = sourceRenderer->GetViewProps();
    if (props) {
        props->InitTraversal();
        vtkProp* prop = nullptr;
        while ((prop = props->GetNextProp())) {
            if (!vtkActor::SafeDownCast(prop)) {
                d->renderers[PERSPECTIVE_VIEW]->AddViewProp(prop);
            }
        }
    }

    double bounds[6];
    sourceRenderer->ComputeVisiblePropBounds(bounds);
    bool hasBounds = (bounds[0] <= bounds[1] && bounds[2] <= bounds[3] &&
                      bounds[4] <= bounds[5]);
    if (hasBounds) {
        setGeometryBounds(bounds);
        double cx = (bounds[0] + bounds[1]) * 0.5;
        double cy = (bounds[2] + bounds[3]) * 0.5;
        double cz = (bounds[4] + bounds[5]) * 0.5;
        setSlicePosition(cx, cy, cz);
    }
    resetCameras();
}

void vtkOrthoSliceViewWidget::applyDisplayProperties() {
    double opacity = m_opacitySlider ? m_opacitySlider->value() / 100.0 : 1.0;
    int ptSize = m_pointSizeSpin ? m_pointSizeSpin->value() : 2;
    double lineW = m_lineWidthSpin ? m_lineWidthSpin->value() : 1.0;
    int reprIdx = m_reprCombo ? m_reprCombo->currentIndex() : 0;
    bool noLight = m_disableLightingCheck && m_disableLightingCheck->isChecked();
    double diffuse = m_diffuseSlider ? m_diffuseSlider->value() / 100.0 : 1.0;
    int interpIdx = m_interpCombo ? m_interpCombo->currentIndex() : 1;
    double specular = m_specularSlider ? m_specularSlider->value() / 100.0 : 0.0;
    int specPower = m_specPowerSpin ? m_specPowerSpin->value() : 100;

    for (int v = 0; v < 4; ++v) {
        auto* actors = d->renderers[v]->GetActors();
        if (!actors) continue;
        actors->InitTraversal();
        vtkActor* a = nullptr;
        while ((a = actors->GetNextActor())) {
            bool isBuiltIn = false;
            for (int k = 0; k < 3; ++k) {
                if (a == d->sliceAxes2D[k].GetPointer()) isBuiltIn = true;
            }
            if (a == d->sliceAxes3D.GetPointer()) isBuiltIn = true;
            if (isBuiltIn) continue;

            auto* prop = a->GetProperty();
            prop->SetOpacity(opacity);
            prop->SetPointSize(ptSize);
            prop->SetLineWidth(lineW);

            switch (reprIdx) {
                case 0: // Slices
                case 1: // Surface
                    prop->SetRepresentationToSurface();
                    prop->EdgeVisibilityOff();
                    break;
                case 2: // Wireframe
                    prop->SetRepresentationToWireframe();
                    break;
                case 3: // Points
                    prop->SetRepresentationToPoints();
                    break;
                case 4: // Surface With Edges
                    prop->SetRepresentationToSurface();
                    prop->EdgeVisibilityOn();
                    break;
            }

            prop->SetLighting(!noLight);
            prop->SetDiffuse(diffuse);
            prop->SetSpecular(specular);
            prop->SetSpecularPower(specPower);

            switch (interpIdx) {
                case 0:
                    prop->SetInterpolationToFlat();
                    break;
                case 1:
                    prop->SetInterpolationToGouraud();
                    break;
                case 2:
                    prop->SetInterpolationToPhong();
                    break;
            }
        }
    }
    render();
}

void vtkOrthoSliceViewWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
}

void vtkOrthoSliceViewWidget::setEntityListProvider(EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshSourceCombo();
}

void vtkOrthoSliceViewWidget::refreshSourceCombo() {
    if (!m_entityListProvider || !m_sourceCombo) return;

    int curIdx = m_sourceCombo->currentIndex();
    quintptr curPtr = (curIdx >= 0)
            ? m_sourceCombo->itemData(curIdx).value<quintptr>()
            : 0;

    m_sourceCombo->blockSignals(true);
    m_sourceCombo->clear();
    m_sourceCombo->addItem(tr("None"), QVariant::fromValue<quintptr>(0));

    auto entities = m_entityListProvider();
    int newIdx = 0;
    for (int i = 0; i < entities.size(); ++i) {
        ccHObject* e = entities[i];
        if (!e) continue;
        m_sourceCombo->addItem(
                e->getName(),
                QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(e)));
        if (reinterpret_cast<quintptr>(e) == curPtr) {
            newIdx = m_sourceCombo->count() - 1;
        }
    }
    m_sourceCombo->setCurrentIndex(newIdx);
    m_sourceCombo->blockSignals(false);
}

void vtkOrthoSliceViewWidget::onSourceComboChanged(int index) {
    if (index < 0) return;
    quintptr ptr = m_sourceCombo->itemData(index).value<quintptr>();
    auto* entity = reinterpret_cast<ccHObject*>(ptr);
    loadEntityIntoView(entity);
}

void vtkOrthoSliceViewWidget::loadEntityIntoView(ccHObject* entity) {
    for (int i = 0; i < 4; ++i) {
        auto* actors = d->renderers[i]->GetActors();
        if (!actors) continue;
        QList<vtkActor*> toRemove;
        actors->InitTraversal();
        vtkActor* a = nullptr;
        while ((a = actors->GetNextActor())) {
            bool isBuiltIn = false;
            for (int k = 0; k < 3; ++k) {
                if (a == d->sliceAxes2D[k].GetPointer()) isBuiltIn = true;
            }
            if (a == d->sliceAxes3D.GetPointer()) isBuiltIn = true;
            if (!isBuiltIn) toRemove.append(a);
        }
        for (auto* rem : toRemove) d->renderers[i]->RemoveActor(rem);
    }

    if (!entity) {
        if (m_statusLabel) m_statusLabel->setText(tr("No data"));
        render();
        return;
    }

    auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
    auto* mesh = ccHObjectCaster::ToMesh(entity);
    if (mesh && !cloud) {
        cloud = ccHObjectCaster::ToGenericPointCloud(mesh->getAssociatedCloud());
    }

    if (!cloud) {
        if (m_statusLabel)
            m_statusLabel->setText(
                    tr("%1 - no point data").arg(entity->getName()));
        render();
        return;
    }

    ccBBox box = entity->getOwnBB();
    if (box.isValid()) {
        double bounds[6] = {box.minCorner().x, box.maxCorner().x,
                            box.minCorner().y, box.maxCorner().y,
                            box.minCorner().z, box.maxCorner().z};
        setGeometryBounds(bounds);

        double cx = (bounds[0] + bounds[1]) * 0.5;
        double cy = (bounds[2] + bounds[3]) * 0.5;
        double cz = (bounds[4] + bounds[5]) * 0.5;
        setSlicePosition(cx, cy, cz);

        double sx = (bounds[1] - bounds[0]) * 0.1;
        double sy = (bounds[3] - bounds[2]) * 0.1;
        double sz = (bounds[5] - bounds[4]) * 0.1;
        double step = std::max({sx, sy, sz});
        if (step < 1e-6) step = 1.0;
        m_sliceIncrements[0] = sx > 1e-6 ? sx : step;
        m_sliceIncrements[1] = sy > 1e-6 ? sy : step;
        m_sliceIncrements[2] = sz > 1e-6 ? sz : step;
        if (m_stepSpin) m_stepSpin->setValue(step);
        for (int i = 0; i < 3; ++i) {
            if (m_sliceSpin[i]) m_sliceSpin[i]->setSingleStep(
                    m_sliceIncrements[i]);
        }
    }

    if (m_statusLabel)
        m_statusLabel->setText(
                tr("%1 (%2 pts)").arg(entity->getName()).arg(cloud->size()));

    resetCameras();
}

void vtkOrthoSliceViewWidget::onSourceComboAboutToShow() {
    refreshSourceCombo();
}
