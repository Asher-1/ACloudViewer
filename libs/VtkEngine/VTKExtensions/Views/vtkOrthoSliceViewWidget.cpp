// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceViewWidget.h"

#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>

#include <CVLog.h>
#include <Converters/Cc2Vtk.h>
#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <vtkCutter.h>
#include <vtkOutlineFilter.h>
#include <vtkPlane.h>
#include <vtkPlaneSource.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkLookupTable.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QTimer>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QResizeEvent>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
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
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <cmath>
#include <cstdio>

static const char* kViewLabels[] = {"Top View", "Right Side View",
                                    "Front View", "3D View"};

struct vtkOrthoSliceViewWidget::Impl {
    QVTKOpenGLNativeWidget* vtkWidget = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> renderers[4];
    vtkSmartPointer<vtkTextActor> annotations[4];
    vtkSmartPointer<vtkTextActor> subAnnotations[4];
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> sliceAxes2D[3];
    vtkSmartPointer<VTKExtensions::vtkPVCenterAxesActor> sliceAxes3D;
    vtkSmartPointer<vtkPolyData> entityPolyData;
    vtkSmartPointer<vtkPlane> slicePlanes[3];
    vtkSmartPointer<vtkCutter> sliceCutters[3];
    vtkSmartPointer<vtkActor> sliceActors[3];
    vtkSmartPointer<vtkCutter> sliceCutters3D[3];
    vtkSmartPointer<vtkActor> sliceActors3D[3];
    vtkSmartPointer<vtkActor> fullModelActor;
    vtkSmartPointer<vtkActor> outlineActor;
    vtkSmartPointer<vtkActor> planeIndicators[3];
    double planeInitCenter[3] = {0, 0, 0};
    vtkSmartPointer<vtkCubeAxesActor> gridAxes[3];
    double slicePos[3] = {0.0, 0.0, 0.0};
    double geomBounds[6] = {-1, 1, -1, 1, -1, 1};
    bool hasMeshCells = false;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientWidget;

    struct ExtraSliceData {
        vtkSmartPointer<vtkPlane> planes[3];
        vtkSmartPointer<vtkCutter> cutters2D[3];
        vtkSmartPointer<vtkCutter> cutters3D[3];
    };
    QList<ExtraSliceData> extraSlices;
};

vtkOrthoSliceViewWidget::vtkOrthoSliceViewWidget(QWidget* parent)
    : QWidget(parent), d(new Impl) {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto* selToolbar = new QWidget(this);
    selToolbar->setObjectName("OrthoSelectionBar");
    auto* selLayout = new QHBoxLayout(selToolbar);
    selLayout->setContentsMargins(2, 1, 2, 1);
    selLayout->setSpacing(2);
    selToolbar->setStyleSheet(
            "QWidget#OrthoSelectionBar { background: #2b2b2b; }"
            "QToolButton { border: 1px solid transparent; padding: 2px; }"
            "QToolButton:checked { background: #4a86c8; border: 1px solid "
            "#6aa6e8; }");

    static constexpr int kSelBtnSize = 20;
    auto makeSelBtn = [selToolbar](const QString& text, const QString& tip,
                                   bool checkable = true) {
        auto* btn = new QToolButton(selToolbar);
        btn->setText(text);
        btn->setToolTip(tip);
        btn->setCheckable(checkable);
        btn->setAutoRaise(true);
        btn->setFixedSize(kSelBtnSize + 16, kSelBtnSize);
        btn->setStyleSheet(
                "QToolButton { font-size: 10px; padding: 1px 3px; }");
        return btn;
    };

    auto* selLabel = new QLabel(QStringLiteral("<b>Selection:</b>"), selToolbar);
    selLabel->setStyleSheet("QLabel { color: #ccc; font-size: 10px; }");
    selLayout->addWidget(selLabel);

    auto* selPointsBtn = makeSelBtn(tr("Pts"), tr("Select Points"));
    auto* selCellsBtn = makeSelBtn(tr("Cells"), tr("Select Cells"));
    auto* selNoneBtn = makeSelBtn(tr("Off"), tr("Disable Selection"), true);
    selNoneBtn->setChecked(true);

    selLayout->addWidget(selPointsBtn);
    selLayout->addWidget(selCellsBtn);
    selLayout->addWidget(selNoneBtn);
    selLayout->addStretch(1);
    layout->addWidget(selToolbar);

    auto clearSelChecks = [selPointsBtn, selCellsBtn, selNoneBtn]() {
        selPointsBtn->setChecked(false);
        selCellsBtn->setChecked(false);
        selNoneBtn->setChecked(false);
    };
    connect(selPointsBtn, &QToolButton::clicked, this,
            [this, clearSelChecks, selPointsBtn]() {
                clearSelChecks();
                selPointsBtn->setChecked(true);
                m_selectionMode = SEL_POINTS;
            });
    connect(selCellsBtn, &QToolButton::clicked, this,
            [this, clearSelChecks, selCellsBtn]() {
                clearSelChecks();
                selCellsBtn->setChecked(true);
                m_selectionMode = SEL_CELLS;
            });
    connect(selNoneBtn, &QToolButton::clicked, this,
            [this, clearSelChecks, selNoneBtn]() {
                clearSelChecks();
                selNoneBtn->setChecked(true);
                m_selectionMode = SEL_NONE;
            });

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
                double maxInc = std::max({m_sliceIncrements[0],
                                          m_sliceIncrements[1],
                                          m_sliceIncrements[2]});
                if (maxInc < 1e-10) maxInc = 1.0;
                for (int i = 0; i < 3; ++i) {
                    double ratio = m_sliceIncrements[i] / maxInc;
                    if (ratio < 0.01) ratio = 1.0;
                    m_sliceIncrements[i] = val * ratio;
                    m_sliceSpin[i]->setSingleStep(m_sliceIncrements[i]);
                }
            });

    sliceLayout->addSpacing(8);
    m_annotCheck = new QCheckBox(tr("Annotations"), sliceBar);
    m_annotCheck->setChecked(true);
    sliceLayout->addWidget(m_annotCheck);
    connect(m_annotCheck, &QCheckBox::toggled, this,
            &vtkOrthoSliceViewWidget::setAnnotationsVisible);

    sliceLayout->addSpacing(4);
    auto* axesGridCheck = new QCheckBox(tr("Axes Grid"), sliceBar);
    axesGridCheck->setChecked(false);
    axesGridCheck->setToolTip(tr("Show/hide axes grid in slice views"));
    sliceLayout->addWidget(axesGridCheck);
    connect(axesGridCheck, &QCheckBox::toggled, this, [this](bool visible) {
        for (int i = 0; i < 3; ++i) {
            if (!d->gridAxes[i]) continue;
            d->gridAxes[i]->SetBounds(d->geomBounds);
            d->gridAxes[i]->SetCamera(d->renderers[i]->GetActiveCamera());
            d->gridAxes[i]->SetVisibility(visible ? 1 : 0);
            d->gridAxes[i]->XAxisVisibilityOff();
            d->gridAxes[i]->YAxisVisibilityOff();
            d->gridAxes[i]->ZAxisVisibilityOff();
            d->gridAxes[i]->DrawXGridlinesOff();
            d->gridAxes[i]->DrawYGridlinesOff();
            d->gridAxes[i]->DrawZGridlinesOff();
            if (visible) {
                if (i == TOP_VIEW) {
                    d->gridAxes[i]->XAxisVisibilityOn();
                    d->gridAxes[i]->ZAxisVisibilityOn();
                    d->gridAxes[i]->DrawXGridlinesOn();
                    d->gridAxes[i]->DrawZGridlinesOn();
                } else if (i == SIDE_VIEW) {
                    d->gridAxes[i]->YAxisVisibilityOn();
                    d->gridAxes[i]->ZAxisVisibilityOn();
                    d->gridAxes[i]->DrawYGridlinesOn();
                    d->gridAxes[i]->DrawZGridlinesOn();
                } else if (i == FRONT_VIEW) {
                    d->gridAxes[i]->XAxisVisibilityOn();
                    d->gridAxes[i]->YAxisVisibilityOn();
                    d->gridAxes[i]->DrawXGridlinesOn();
                    d->gridAxes[i]->DrawYGridlinesOn();
                }
            }
            d->renderers[i]->ResetCameraClippingRange();
        }
        render();
    });

    sliceLayout->addSpacing(8);
    auto* resetCamBtn = new QPushButton(tr("Reset"), sliceBar);
    resetCamBtn->setToolTip(tr("Reset all cameras to fit data"));
    resetCamBtn->setFixedWidth(48);
    sliceLayout->addWidget(resetCamBtn);
    connect(resetCamBtn, &QPushButton::clicked, this,
            &vtkOrthoSliceViewWidget::resetCameras);

    auto* centerBtn = new QPushButton(tr("Center"), sliceBar);
    centerBtn->setToolTip(tr("Center slices on geometry bounds"));
    centerBtn->setFixedWidth(48);
    sliceLayout->addWidget(centerBtn);
    connect(centerBtn, &QPushButton::clicked, this, [this]() {
        double cx = (d->geomBounds[0] + d->geomBounds[1]) * 0.5;
        double cy = (d->geomBounds[2] + d->geomBounds[3]) * 0.5;
        double cz = (d->geomBounds[4] + d->geomBounds[5]) * 0.5;
        setSlicePosition(cx, cy, cz);
        emit slicePositionChanged(cx, cy, cz);
        resetCameras();
    });

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
    connect(m_coloringCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { applyDisplayProperties(); });

    colorLayout->addSpacing(8);
    m_mapScalarsCheck = new QCheckBox(tr("Map Scalars"), colorBar);
    m_mapScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_mapScalarsCheck);
    connect(m_mapScalarsCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    m_interpScalarsCheck = new QCheckBox(tr("Interp Scalars"), colorBar);
    m_interpScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_interpScalarsCheck);
    connect(m_interpScalarsCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    m_useNanColorCheck = new QCheckBox(tr("Nan Color"), colorBar);
    m_useNanColorCheck->setToolTip(tr("Use Nan Color For Missing Arrays"));
    colorLayout->addWidget(m_useNanColorCheck);
    connect(m_useNanColorCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    colorLayout->addSpacing(8);
    m_renderTubesCheck = new QCheckBox(tr("Tubes"), colorBar);
    m_renderTubesCheck->setToolTip(tr("Render Lines As Tubes"));
    colorLayout->addWidget(m_renderTubesCheck);
    connect(m_renderTubesCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    m_renderSpheresCheck = new QCheckBox(tr("Spheres"), colorBar);
    m_renderSpheresCheck->setToolTip(tr("Render Points As Spheres"));
    colorLayout->addWidget(m_renderSpheresCheck);
    connect(m_renderSpheresCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    m_showOutlineCheck = new QCheckBox(tr("Outline"), colorBar);
    m_showOutlineCheck->setToolTip(tr("Show bounding outline"));
    m_showOutlineCheck->setChecked(false);
    colorLayout->addWidget(m_showOutlineCheck);
    connect(m_showOutlineCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

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

    lightLayout->addSpacing(4);
    m_specColorCheck = new QCheckBox(tr("Spec Color"), lightBar);
    m_specColorCheck->setToolTip(tr("Specular Color (ParaView)"));
    lightLayout->addWidget(m_specColorCheck);
    connect(m_specColorCheck, &QCheckBox::toggled,
            this, [this](bool) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Lum:"), lightBar));
    m_luminositySpin = new QDoubleSpinBox(lightBar);
    m_luminositySpin->setRange(0.0, 1.0);
    m_luminositySpin->setSingleStep(0.1);
    m_luminositySpin->setDecimals(1);
    m_luminositySpin->setValue(0.0);
    m_luminositySpin->setFixedWidth(50);
    m_luminositySpin->setToolTip(tr("Luminosity (ParaView Ambient)"));
    lightLayout->addWidget(m_luminositySpin);
    connect(m_luminositySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double) { applyDisplayProperties(); });

    lightLayout->addStretch(1);
    layout->addWidget(lightBar);
    dispBar->setVisible(false);
    dispBar->setMaximumHeight(0);
    colorBar->setVisible(false);
    colorBar->setMaximumHeight(0);
    lightBar->setVisible(false);
    lightBar->setMaximumHeight(0);

    d->vtkWidget = new QVTKOpenGLNativeWidget(this);
    layout->addWidget(d->vtkWidget, 1);

    d->renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    d->vtkWidget->setRenderWindow(d->renderWindow);

    for (int i = 0; i < 4; ++i) {
        d->renderers[i] = vtkSmartPointer<vtkRenderer>::New();
        d->renderers[i]->SetBackground(0.5, 0.5, 0.5);
        d->renderers[i]->SetUseFXAA(true);
        d->renderWindow->AddRenderer(d->renderers[i]);
    }
    d->renderers[PERSPECTIVE_VIEW]->SetBackground(0.32, 0.34, 0.43);
    d->renderers[PERSPECTIVE_VIEW]->SetBackground2(0.52, 0.54, 0.63);
    d->renderers[PERSPECTIVE_VIEW]->SetGradientBackground(true);

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
    for (int i = 0; i < 3; ++i) {
        d->gridAxes[i] = vtkSmartPointer<vtkCubeAxesActor>::New();
        d->gridAxes[i]->SetBounds(d->geomBounds);
        d->gridAxes[i]->SetCamera(d->renderers[i]->GetActiveCamera());
        d->gridAxes[i]->SetFlyModeToClosestTriad();
        d->gridAxes[i]->SetUse2DMode(true);
        d->gridAxes[i]->SetGridLineLocation(
                vtkCubeAxesActor::VTK_GRID_LINES_ALL);
        d->gridAxes[i]->GetTitleTextProperty(0)->SetFontSize(10);
        d->gridAxes[i]->GetTitleTextProperty(1)->SetFontSize(10);
        d->gridAxes[i]->GetTitleTextProperty(2)->SetFontSize(10);
        d->gridAxes[i]->GetLabelTextProperty(0)->SetFontSize(8);
        d->gridAxes[i]->GetLabelTextProperty(1)->SetFontSize(8);
        d->gridAxes[i]->GetLabelTextProperty(2)->SetFontSize(8);
        d->gridAxes[i]->GetXAxesGridlinesProperty()->SetColor(0.5, 0.5, 0.5);
        d->gridAxes[i]->GetYAxesGridlinesProperty()->SetColor(0.5, 0.5, 0.5);
        d->gridAxes[i]->GetZAxesGridlinesProperty()->SetColor(0.5, 0.5, 0.5);
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

    QTimer::singleShot(100, this, [this]() {
        if (!d->orientWidget && d->renderWindow) {
            auto* interactor = d->renderWindow->GetInteractor();
            if (interactor) {
                auto axes = vtkSmartPointer<vtkAxesActor>::New();
                axes->SetShaftTypeToCylinder();
                axes->SetTotalLength(1.0, 1.0, 1.0);
                axes->SetXAxisLabelText("X");
                axes->SetYAxisLabelText("Y");
                axes->SetZAxisLabelText("Z");
                d->orientWidget =
                        vtkSmartPointer<vtkOrientationMarkerWidget>::New();
                d->orientWidget->SetOutlineColor(0.2, 0.2, 0.2);
                d->orientWidget->SetOrientationMarker(axes);
                d->orientWidget->SetInteractor(interactor);
                d->orientWidget->SetCurrentRenderer(
                        d->renderers[PERSPECTIVE_VIEW]);
                d->orientWidget->SetViewport(0.85, 0.0, 1.0, 0.15);
                d->orientWidget->SetEnabled(true);
                d->orientWidget->InteractiveOff();
            }
        }
    });

    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &QWidget::customContextMenuRequested, this,
            [this, dispBar, colorBar, lightBar](const QPoint& pos) {
                QMenu menu(this);
                auto* resetAct = menu.addAction(tr("Reset Cameras"));
                connect(resetAct, &QAction::triggered, this,
                        &vtkOrthoSliceViewWidget::resetCameras);
                auto* centerAct = menu.addAction(tr("Center Slices"));
                connect(centerAct, &QAction::triggered, this, [this]() {
                    double cx = (d->geomBounds[0] + d->geomBounds[1]) * 0.5;
                    double cy = (d->geomBounds[2] + d->geomBounds[3]) * 0.5;
                    double cz = (d->geomBounds[4] + d->geomBounds[5]) * 0.5;
                    setSlicePosition(cx, cy, cz);
                    emit slicePositionChanged(cx, cy, cz);
                    resetCameras();
                });
                menu.addSeparator();
                auto toggleBar = [](QWidget* bar) {
                    bool show = !bar->isVisible();
                    bar->setVisible(show);
                    bar->setMaximumHeight(show ? QWIDGETSIZE_MAX : 0);
                };
                auto* dispAct = menu.addAction(
                        dispBar->isVisible()
                                ? tr("Hide Display Properties")
                                : tr("Show Display Properties"));
                connect(dispAct, &QAction::triggered, this, [dispBar, toggleBar]() {
                    toggleBar(dispBar);
                });
                auto* colorAct = menu.addAction(
                        colorBar->isVisible()
                                ? tr("Hide Coloring Bar")
                                : tr("Show Coloring Bar"));
                connect(colorAct, &QAction::triggered, this, [colorBar, toggleBar]() {
                    toggleBar(colorBar);
                });
                auto* lightAct = menu.addAction(
                        lightBar->isVisible()
                                ? tr("Hide Lighting Bar")
                                : tr("Show Lighting Bar"));
                connect(lightAct, &QAction::triggered, this, [lightBar, toggleBar]() {
                    toggleBar(lightBar);
                });
                menu.exec(mapToGlobal(pos));
            });

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

    if (d->slicePlanes[TOP_VIEW])
        d->slicePlanes[TOP_VIEW]->SetOrigin(x, y, z);
    if (d->slicePlanes[SIDE_VIEW])
        d->slicePlanes[SIDE_VIEW]->SetOrigin(x, y, z);
    if (d->slicePlanes[FRONT_VIEW])
        d->slicePlanes[FRONT_VIEW]->SetOrigin(x, y, z);

    for (int i = 0; i < 3; ++i) {
        if (d->sliceCutters[i]) {
            d->sliceCutters[i]->Modified();
            d->sliceCutters[i]->Update();
        }
        if (d->sliceCutters3D[i]) {
            d->sliceCutters3D[i]->Modified();
            d->sliceCutters3D[i]->Update();
        }
    }

    for (auto& extra : d->extraSlices) {
        for (int i = 0; i < 3; ++i) {
            if (extra.planes[i]) extra.planes[i]->SetOrigin(x, y, z);
            if (extra.cutters2D[i]) {
                extra.cutters2D[i]->Modified();
                extra.cutters2D[i]->Update();
            }
            if (extra.cutters3D[i]) {
                extra.cutters3D[i]->Modified();
                extra.cutters3D[i]->Update();
            }
        }
    }

    if (d->planeIndicators[TOP_VIEW])
        d->planeIndicators[TOP_VIEW]->SetPosition(
                0, y - d->planeInitCenter[1], 0);
    if (d->planeIndicators[SIDE_VIEW])
        d->planeIndicators[SIDE_VIEW]->SetPosition(
                x - d->planeInitCenter[0], 0, 0);
    if (d->planeIndicators[FRONT_VIEW])
        d->planeIndicators[FRONT_VIEW]->SetPosition(
                0, 0, z - d->planeInitCenter[2]);

    updateSliceSpinners();
    render();
}

void vtkOrthoSliceViewWidget::resetCameras() {
    for (int i = 0; i < 3; ++i) {
        if (d->sliceCutters[i]) d->sliceCutters[i]->Update();

        bool usedCutterBounds = false;
        if (d->sliceActors[i] && d->sliceCutters[i]) {
            auto* output = d->sliceCutters[i]->GetOutput();
            if (output && output->GetNumberOfPoints() > 0) {
                double cutBounds[6];
                output->GetBounds(cutBounds);
                bool validBounds = (cutBounds[1] > cutBounds[0]) ||
                                   (cutBounds[3] > cutBounds[2]) ||
                                   (cutBounds[5] > cutBounds[4]);
                if (validBounds) {
                    d->renderers[i]->ResetCamera(cutBounds);
                    usedCutterBounds = true;
                }
            }
        }
        if (!usedCutterBounds) {
            d->renderers[i]->ResetCamera(d->geomBounds);
        }
        d->renderers[i]->ResetCameraClippingRange();
        auto* cam = d->renderers[i]->GetActiveCamera();
        if (cam && cam->GetParallelProjection()) {
            cam->SetParallelScale(cam->GetParallelScale() * 1.05);
        }
    }

    if (d->fullModelActor) {
        d->renderers[PERSPECTIVE_VIEW]->ResetCamera();
    } else {
        d->renderers[PERSPECTIVE_VIEW]->ResetCamera(d->geomBounds);
    }
    d->renderers[PERSPECTIVE_VIEW]->ResetCameraClippingRange();
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
    if (widths[0] < 1e-10) widths[0] = 1.0;
    if (widths[1] < 1e-10) widths[1] = 1.0;
    if (widths[2] < 1e-10) widths[2] = 1.0;
    double sw[3] = {widths[0] * 2, widths[1] * 2, widths[2] * 2};
    for (int i = 0; i < 3; ++i) {
        if (d->sliceAxes2D[i]) d->sliceAxes2D[i]->SetScale(sw);
    }
    if (d->sliceAxes3D) d->sliceAxes3D->SetScale(sw);
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
                return false;
            }
        } else if (event->type() == QEvent::MouseButtonPress) {
            auto* me = static_cast<QMouseEvent*>(event);
            int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx >= 0 && viewIdx < 3)
                m_lastActiveQuadrant = viewIdx;
            if (me->button() == Qt::MiddleButton) {
                if (viewIdx >= 0 && viewIdx < 3) {
                    m_draggingSlice = true;
                    m_dragViewIdx = viewIdx;
                    m_dragLastPos = me->pos();
                    return true;
                }
            }
            if (me->button() == Qt::LeftButton && viewIdx >= 0 && viewIdx < 3) {
                m_panning2D = true;
                m_panLastPos = me->pos();
                return true;
            }
            if (me->button() == Qt::RightButton && viewIdx >= 0 && viewIdx < 3) {
                m_zooming2D = true;
                m_zoomViewIdx = viewIdx;
                m_zoomLastPos = me->pos();
                return true;
            }
        } else if (event->type() == QEvent::MouseMove && m_panning2D) {
            auto* me = static_cast<QMouseEvent*>(event);
            int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx < 0 || viewIdx > 3) viewIdx = m_lastActiveQuadrant;
            if (viewIdx >= 0 && viewIdx < 3) {
                auto* ren = d->renderers[viewIdx].GetPointer();
                auto* cam = ren ? ren->GetActiveCamera() : nullptr;
                if (cam) {
                    QPoint delta = me->pos() - m_panLastPos;
                    m_panLastPos = me->pos();
                    double scale = cam->GetParallelScale();
                    int h = d->vtkWidget->height();
                    double factor = 2.0 * scale / (h > 1 ? h : 1);
                    double fp[3], pos[3];
                    cam->GetFocalPoint(fp);
                    cam->GetPosition(pos);
                    double dx = -delta.x() * factor;
                    double dy = delta.y() * factor;
                    double vu[3];
                    cam->GetViewUp(vu);
                    double right[3];
                    double dir[3] = {fp[0] - pos[0], fp[1] - pos[1], fp[2] - pos[2]};
                    right[0] = dir[1] * vu[2] - dir[2] * vu[1];
                    right[1] = dir[2] * vu[0] - dir[0] * vu[2];
                    right[2] = dir[0] * vu[1] - dir[1] * vu[0];
                    double rlen = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
                    if (rlen > 1e-10) { right[0] /= rlen; right[1] /= rlen; right[2] /= rlen; }
                    for (int i = 0; i < 3; ++i) {
                        fp[i] += dx * right[i] + dy * vu[i];
                        pos[i] += dx * right[i] + dy * vu[i];
                    }
                    cam->SetFocalPoint(fp);
                    cam->SetPosition(pos);
                    ren->ResetCameraClippingRange();
                    d->renderWindow->Render();
                }
            }
            return true;
        } else if (event->type() == QEvent::MouseMove && m_zooming2D) {
            auto* me = static_cast<QMouseEvent*>(event);
            auto* ren = d->renderers[m_zoomViewIdx].GetPointer();
            auto* cam = ren ? ren->GetActiveCamera() : nullptr;
            if (cam && cam->GetParallelProjection()) {
                int dy = me->pos().y() - m_zoomLastPos.y();
                m_zoomLastPos = me->pos();
                double factor = 1.0 + dy * 0.005;
                cam->SetParallelScale(cam->GetParallelScale() * factor);
                ren->ResetCameraClippingRange();
                d->renderWindow->Render();
            }
            return true;
        } else if (event->type() == QEvent::MouseMove && m_draggingSlice) {
            auto* me = static_cast<QMouseEvent*>(event);
            QPoint delta = me->pos() - m_dragLastPos;
            m_dragLastPos = me->pos();
            int axisMap[3] = {1, 0, 2};
            double step = m_sliceIncrements[axisMap[m_dragViewIdx]];
            double scaledDelta = -delta.y() * step * 0.1;
            double x = d->slicePos[0];
            double y = d->slicePos[1];
            double z = d->slicePos[2];
            switch (m_dragViewIdx) {
                case TOP_VIEW: y += scaledDelta; break;
                case SIDE_VIEW: x += scaledDelta; break;
                case FRONT_VIEW: z += scaledDelta; break;
                default: break;
            }
            setSlicePosition(x, y, z);
            emit slicePositionChanged(x, y, z);
            return true;
        } else if (event->type() == QEvent::MouseButtonRelease) {
            if (m_panning2D) {
                m_panning2D = false;
                return true;
            }
            if (m_zooming2D) {
                m_zooming2D = false;
                m_zoomViewIdx = -1;
                return true;
            }
            if (m_draggingSlice) {
                m_draggingSlice = false;
                m_dragViewIdx = -1;
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
                if (w <= 0 || h <= 0) return false;

                double dpr = d->vtkWidget->devicePixelRatioF();
                double dispX = me->pos().x() * dpr;
                double dispY = (h - 1 - me->pos().y()) * dpr;

                ren->SetDisplayPoint(dispX, dispY, 0.0);
                ren->DisplayToWorld();

                double worldPt[4];
                ren->GetWorldPoint(worldPt);
                if (std::abs(worldPt[3]) > 1e-10) {
                    worldPt[0] /= worldPt[3];
                    worldPt[1] /= worldPt[3];
                    worldPt[2] /= worldPt[3];
                }

                double newPos[3] = {worldPt[0], worldPt[1], worldPt[2]};

                int axisMap[3] = {1, 0, 2};
                newPos[axisMap[viewIdx]] = d->slicePos[axisMap[viewIdx]];

                setSlicePosition(newPos[0], newPos[1], newPos[2]);
                emit slicePositionChanged(newPos[0], newPos[1], newPos[2]);
                return true;
            }
        } else if (event->type() == QEvent::KeyPress) {
            auto* ke = static_cast<QKeyEvent*>(event);
            double x = d->slicePos[0];
            double y = d->slicePos[1];
            double z = d->slicePos[2];
            bool handled = true;
            int q = m_lastActiveQuadrant;
            if (q >= 0 && q < 3) {
                int axisMap[3] = {1, 0, 2};
                double step = m_sliceIncrements[axisMap[q]];
                switch (ke->key()) {
                    case Qt::Key_Up:
                    case Qt::Key_Right:
                        switch (q) {
                            case TOP_VIEW: y += step; break;
                            case SIDE_VIEW: x += step; break;
                            case FRONT_VIEW: z += step; break;
                        }
                        break;
                    case Qt::Key_Down:
                    case Qt::Key_Left:
                        switch (q) {
                            case TOP_VIEW: y -= step; break;
                            case SIDE_VIEW: x -= step; break;
                            case FRONT_VIEW: z -= step; break;
                        }
                        break;
                    default: handled = false; break;
                }
            } else {
                double step = m_stepSpin ? m_stepSpin->value() : 1.0;
                switch (ke->key()) {
                    case Qt::Key_Up: y += step; break;
                    case Qt::Key_Down: y -= step; break;
                    case Qt::Key_Left: x -= step; break;
                    case Qt::Key_Right: x += step; break;
                    case Qt::Key_PageUp: z += step; break;
                    case Qt::Key_PageDown: z -= step; break;
                    default: handled = false; break;
                }
            }
            if (handled) {
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
    for (int i = 0; i < 4; ++i) {
        d->annotations[i]->SetVisibility(visible ? 1 : 0);
        d->subAnnotations[i]->SetVisibility(visible ? 1 : 0);
    }
    render();
}

void vtkOrthoSliceViewWidget::populateFromRenderer(vtkRenderer* sourceRenderer) {
    if (!sourceRenderer) return;

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

    int colorIdx = m_coloringCombo ? m_coloringCombo->currentIndex() : 0;
    bool mapScalars = m_mapScalarsCheck ? m_mapScalarsCheck->isChecked() : true;
    bool interpScalars = m_interpScalarsCheck ? m_interpScalarsCheck->isChecked() : true;
    bool useNanColor = m_useNanColorCheck && m_useNanColorCheck->isChecked();
    bool useSpecColor = m_specColorCheck && m_specColorCheck->isChecked();
    bool renderTubes = m_renderTubesCheck && m_renderTubesCheck->isChecked();
    bool renderSpheres = m_renderSpheresCheck && m_renderSpheresCheck->isChecked();
    bool showOutline = m_showOutlineCheck && m_showOutlineCheck->isChecked();

    if (d->outlineActor) d->outlineActor->SetVisibility(showOutline);

    auto applyColorToMapper = [&](vtkPolyDataMapper* mapper, vtkActor* a) {
        if (!mapper) return;
        if (colorIdx == 0) {
            mapper->ScalarVisibilityOff();
            a->GetProperty()->SetColor(0.85, 0.85, 0.85);
        } else {
            mapper->ScalarVisibilityOn();
            mapper->SetScalarModeToUsePointFieldData();
            if (mapScalars) mapper->SetColorModeToMapScalars();
            else mapper->SetColorModeToDirectScalars();
            mapper->SetInterpolateScalarsBeforeMapping(interpScalars);

            QString colorName = m_coloringCombo
                    ? m_coloringCombo->currentText() : QString();
            QByteArray nameBytes;
            const char* arrayName = nullptr;
            if (colorName == tr("Points")) {
                arrayName = nullptr;
                mapper->SetScalarModeToUsePointData();
            } else if (colorName == tr("Normals")) {
                arrayName = "Normals";
            } else if (colorName == tr("TCoords")) {
                arrayName = "TCoords";
            } else {
                nameBytes = colorName.toUtf8();
                arrayName = nameBytes.constData();
            }
            if (arrayName) {
                mapper->SelectColorArray(arrayName);
                auto* pd = vtkPolyData::SafeDownCast(mapper->GetInput());
                if (pd && pd->GetPointData()) {
                    auto* arr = pd->GetPointData()->GetArray(arrayName);
                    if (arr) {
                        double range[2];
                        arr->GetRange(range, -1);
                        mapper->SetScalarRange(range);
                    }
                }
            }
        }
    };

    auto applyLighting = [&](vtkProperty* prop) {
        double ambient = m_luminositySpin ? m_luminositySpin->value() : 0.0;
        prop->SetLighting(!noLight);
        prop->SetAmbient(ambient);
        prop->SetDiffuse(diffuse);
        prop->SetSpecular(specular);
        prop->SetSpecularPower(specPower);
        if (useSpecColor) {
            prop->SetSpecularColor(1.0, 1.0, 1.0);
        } else {
            double dc[3];
            prop->GetDiffuseColor(dc);
            prop->SetSpecularColor(dc);
        }
        switch (interpIdx) {
            case 0: prop->SetInterpolationToFlat(); break;
            case 1: prop->SetInterpolationToGouraud(); break;
            case 2: prop->SetInterpolationToPhong(); break;
        }
        prop->SetRenderLinesAsTubes(renderTubes);
        prop->SetRenderPointsAsSpheres(renderSpheres);
    };

    auto applyNanColor = [&](vtkPolyDataMapper* mapper) {
        if (!mapper || !mapper->GetLookupTable()) return;
        auto* lut = vtkLookupTable::SafeDownCast(mapper->GetLookupTable());
        if (!lut) return;
        if (useNanColor) {
            lut->SetNanColor(0.5, 0.0, 0.0, 1.0);
        } else {
            lut->SetNanColor(0.0, 0.0, 0.0, 0.0);
        }
    };

    for (int v = 0; v < 4; ++v) {
        bool is2DView = (v != PERSPECTIVE_VIEW);
        auto* actors = d->renderers[v]->GetActors();
        if (!actors) continue;
        actors->InitTraversal();
        vtkActor* a = nullptr;
        while ((a = actors->GetNextActor())) {
            bool isBuiltIn = false;
            for (int k = 0; k < 3; ++k) {
                if (a == d->sliceAxes2D[k].GetPointer()) isBuiltIn = true;
                if (a == d->sliceActors3D[k].GetPointer()) isBuiltIn = true;
                if (a == d->planeIndicators[k].GetPointer()) isBuiltIn = true;
            }
            if (a == d->sliceAxes3D.GetPointer()) isBuiltIn = true;
            if (a == d->outlineActor.GetPointer()) isBuiltIn = true;
            if (isBuiltIn) continue;

            auto* pdMapper = vtkPolyDataMapper::SafeDownCast(a->GetMapper());
            applyColorToMapper(pdMapper, a);
            applyNanColor(pdMapper);

            auto* prop = a->GetProperty();
            prop->SetOpacity(opacity);
            prop->SetPointSize(ptSize);
            prop->SetLineWidth(lineW);

            if (is2DView) {
                if (reprIdx == 0) {
                    prop->SetRepresentationToSurface();
                    prop->EdgeVisibilityOff();
                    prop->SetLighting(false);
                    prop->SetAmbient(1.0);
                    prop->SetDiffuse(0.0);
                } else {
                    switch (reprIdx) {
                        case 1: prop->SetRepresentationToSurface(); prop->EdgeVisibilityOff(); break;
                        case 2: prop->SetRepresentationToWireframe(); break;
                        case 3: prop->SetRepresentationToPoints(); break;
                        case 4: prop->SetRepresentationToSurface(); prop->EdgeVisibilityOn(); break;
                    }
                    applyLighting(prop);
                }
            } else {
                switch (reprIdx) {
                    case 0:
                    case 1: prop->SetRepresentationToSurface(); prop->EdgeVisibilityOff(); break;
                    case 2: prop->SetRepresentationToWireframe(); break;
                    case 3: prop->SetRepresentationToPoints(); break;
                    case 4: prop->SetRepresentationToSurface(); prop->EdgeVisibilityOn(); break;
                }
                applyLighting(prop);
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
    if (entities.size() > 1) {
        m_sourceCombo->addItem(tr("All"), QVariant::fromValue<quintptr>(1));
    }
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
    if (ptr == 1 && m_entityListProvider) {
        auto entities = m_entityListProvider();
        loadEntitiesIntoView(entities);
    } else {
        auto* entity = reinterpret_cast<ccHObject*>(ptr);
        loadEntityIntoView(entity);
    }
}

void vtkOrthoSliceViewWidget::loadEntityIntoView(ccHObject* entity) {
    d->extraSlices.clear();
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

    auto* mesh = ccHObjectCaster::ToMesh(entity);
    auto* genericMesh = ccHObjectCaster::ToGenericMesh(entity);
    auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
    if (!cloud && genericMesh) {
        cloud = ccHObjectCaster::ToGenericPointCloud(
                genericMesh->getAssociatedCloud());
    }
    for (unsigned ci = 0;
         (!genericMesh || !cloud) && ci < entity->getChildrenNumber(); ++ci) {
        auto* child = entity->getChild(ci);
        if (!genericMesh) {
            genericMesh = ccHObjectCaster::ToGenericMesh(child);
            if (!mesh) mesh = ccHObjectCaster::ToMesh(child);
        }
        if (!cloud) cloud = ccHObjectCaster::ToGenericPointCloud(child);
    }
    if (genericMesh && !cloud) {
        cloud = ccHObjectCaster::ToGenericPointCloud(
                genericMesh->getAssociatedCloud());
    }
    if (!mesh && genericMesh)
        mesh = ccHObjectCaster::ToMesh(genericMesh);

    if (!cloud) {
        if (m_statusLabel)
            m_statusLabel->setText(
                    tr("%1 - no point data").arg(entity->getName()));
        render();
        return;
    }

    auto* pcCloud = ccHObjectCaster::ToPointCloud(entity);
    if (!pcCloud && genericMesh) {
        pcCloud = ccHObjectCaster::ToPointCloud(
                genericMesh->getAssociatedCloud());
    }
    if (!pcCloud && mesh) {
        pcCloud = ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    }
    if (!pcCloud) pcCloud = ccHObjectCaster::ToPointCloud(cloud);
    for (unsigned ci = 0; !pcCloud && ci < entity->getChildrenNumber(); ++ci) {
        pcCloud = ccHObjectCaster::ToPointCloud(entity->getChild(ci));
    }

    d->entityPolyData = nullptr;
    d->hasMeshCells = false;
    if (d->fullModelActor) {
        d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->fullModelActor);
        d->fullModelActor = nullptr;
    }
    if (d->outlineActor) {
        d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->outlineActor);
        d->outlineActor = nullptr;
    }
    for (int i = 0; i < 3; ++i) {
        if (d->sliceActors[i]) {
            d->renderers[i]->RemoveActor(d->sliceActors[i]);
            d->sliceActors[i] = nullptr;
        }
        if (d->sliceActors3D[i]) {
            d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->sliceActors3D[i]);
            d->sliceActors3D[i] = nullptr;
        }
        d->sliceCutters[i] = nullptr;
        d->sliceCutters3D[i] = nullptr;
        d->slicePlanes[i] = nullptr;
    }

    CVLog::Print("[OrthoSlice] Loading entity: mesh=%p genericMesh=%p "
                 "pcCloud=%p cloud=%p",
                 mesh, genericMesh, pcCloud, cloud);

    ccGenericMesh* meshForConvert = genericMesh ? genericMesh : mesh;
    if (meshForConvert && pcCloud) {
        d->entityPolyData =
                Converters::Cc2Vtk::MeshToPolyData(pcCloud, meshForConvert);
        CVLog::Print("[OrthoSlice] MeshToPolyData: polydata=%p pts=%lld "
                     "cells=%lld polys=%lld",
                     d->entityPolyData.GetPointer(),
                     d->entityPolyData
                             ? d->entityPolyData->GetNumberOfPoints()
                             : 0,
                     d->entityPolyData
                             ? d->entityPolyData->GetNumberOfCells()
                             : 0,
                     d->entityPolyData
                             ? d->entityPolyData->GetNumberOfPolys()
                             : 0);
    }
    if (d->entityPolyData &&
        d->entityPolyData->GetNumberOfPoints() == 0 && pcCloud) {
        d->entityPolyData = nullptr;
    }
    if (!d->entityPolyData && pcCloud) {
        d->entityPolyData = Converters::Cc2Vtk::PointCloudToPolyData(pcCloud);
    }

    ccBBox box = entity->getOwnBB();
    double cx = 0, cy = 0, cz = 0;
    if (box.isValid()) {
        cx = (box.minCorner().x + box.maxCorner().x) * 0.5;
        cy = (box.minCorner().y + box.maxCorner().y) * 0.5;
        cz = (box.minCorner().z + box.maxCorner().z) * 0.5;
    }

    if (d->entityPolyData) {
        d->hasMeshCells =
                (d->entityPolyData->GetNumberOfCells() > 0 &&
                 d->entityPolyData->GetNumberOfPolys() > 0);

        {
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputData(d->entityPolyData);
            if (d->entityPolyData->GetPointData() &&
                d->entityPolyData->GetPointData()->GetScalars()) {
                mapper->ScalarVisibilityOn();
            } else {
                mapper->ScalarVisibilityOff();
            }
            mapper->Update();
            d->fullModelActor = vtkSmartPointer<vtkActor>::New();
            d->fullModelActor->SetMapper(mapper);
            bool hasScalars = d->entityPolyData->GetPointData() &&
                              d->entityPolyData->GetPointData()->GetScalars();
            if (d->hasMeshCells) {
                d->fullModelActor->GetProperty()->SetRepresentationToSurface();
                d->fullModelActor->GetProperty()->SetAmbient(0.2);
                d->fullModelActor->GetProperty()->SetDiffuse(0.8);
                d->fullModelActor->GetProperty()->LightingOn();
                d->fullModelActor->GetProperty()->SetInterpolationToPhong();
                if (!hasScalars) {
                    d->fullModelActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
                }
            } else {
                d->fullModelActor->GetProperty()->SetRepresentationToPoints();
                d->fullModelActor->GetProperty()->SetPointSize(2.0);
                d->fullModelActor->GetProperty()->SetAmbient(1.0);
                d->fullModelActor->GetProperty()->SetDiffuse(0.0);
                d->fullModelActor->GetProperty()->LightingOff();
                if (!hasScalars) {
                    d->fullModelActor->GetProperty()->SetColor(0.8, 0.8, 0.8);
                }
            }
            d->fullModelActor->SetVisibility(1);
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->fullModelActor);
            CVLog::Print("[OrthoSlice] 3D model actor added: %lld pts, "
                         "%lld polys, bounds=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]",
                         d->entityPolyData->GetNumberOfPoints(),
                         d->entityPolyData->GetNumberOfPolys(),
                         d->entityPolyData->GetBounds()[0],
                         d->entityPolyData->GetBounds()[1],
                         d->entityPolyData->GetBounds()[2],
                         d->entityPolyData->GetBounds()[3],
                         d->entityPolyData->GetBounds()[4],
                         d->entityPolyData->GetBounds()[5]);

            vtkNew<vtkOutlineFilter> outline;
            outline->SetInputData(d->entityPolyData);
            outline->Update();
            vtkNew<vtkPolyDataMapper> outMapper;
            outMapper->SetInputConnection(outline->GetOutputPort());
            d->outlineActor = vtkSmartPointer<vtkActor>::New();
            d->outlineActor->SetMapper(outMapper);
            d->outlineActor->GetProperty()->SetColor(0.3, 0.3, 0.3);
            d->outlineActor->GetProperty()->SetLineWidth(1.0);
            bool showOutline = m_showOutlineCheck &&
                               m_showOutlineCheck->isChecked();
            d->outlineActor->SetVisibility(showOutline);
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->outlineActor);
        }

        const double normals[3][3] = {{0,1,0}, {1,0,0}, {0,0,1}};
        const double origins[3][3] = {
                {cx, cy, cz}, {cx, cy, cz}, {cx, cy, cz}};
        for (int i = 0; i < 3; ++i) {
            d->slicePlanes[i] = vtkSmartPointer<vtkPlane>::New();
            d->slicePlanes[i]->SetNormal(normals[i]);
            d->slicePlanes[i]->SetOrigin(origins[i]);

            d->sliceCutters[i] = vtkSmartPointer<vtkCutter>::New();
            d->sliceCutters[i]->SetCutFunction(d->slicePlanes[i]);
            d->sliceCutters[i]->SetInputData(d->entityPolyData);
            d->sliceCutters[i]->Update();

            vtkNew<vtkPolyDataMapper> mapper;
            bool cutterHasOutput =
                    d->sliceCutters[i]->GetOutput() &&
                    d->sliceCutters[i]->GetOutput()->GetNumberOfPoints() > 0;

            if (d->hasMeshCells && cutterHasOutput) {
                mapper->SetInputConnection(
                        d->sliceCutters[i]->GetOutputPort());
            } else {
                mapper->SetInputData(d->entityPolyData);
            }
            mapper->ScalarVisibilityOn();

            d->sliceActors[i] = vtkSmartPointer<vtkActor>::New();
            d->sliceActors[i]->SetMapper(mapper);
            if (d->hasMeshCells && cutterHasOutput) {
                d->sliceActors[i]->GetProperty()->SetColor(0.0, 0.0, 0.0);
                d->sliceActors[i]->GetProperty()->SetLineWidth(3.0);
                d->sliceActors[i]->GetProperty()->SetAmbient(1.0);
                d->sliceActors[i]->GetProperty()->SetDiffuse(0.0);
                d->sliceActors[i]->GetProperty()->LightingOff();
            } else {
                d->sliceActors[i]->GetProperty()->SetColor(0.0, 0.0, 0.0);
                d->sliceActors[i]->GetProperty()->SetPointSize(3);
                d->sliceActors[i]->GetProperty()->
                        SetRepresentationToPoints();
                d->sliceActors[i]->GetProperty()->LightingOff();
            }
            d->renderers[i]->AddActor(d->sliceActors[i]);

            d->sliceCutters3D[i] = vtkSmartPointer<vtkCutter>::New();
            d->sliceCutters3D[i]->SetCutFunction(d->slicePlanes[i]);
            d->sliceCutters3D[i]->SetInputData(d->entityPolyData);
            d->sliceCutters3D[i]->Update();

            vtkNew<vtkPolyDataMapper> mapper3D;
            bool cut3dOk = d->sliceCutters3D[i]->GetOutput() &&
                           d->sliceCutters3D[i]->GetOutput()
                                           ->GetNumberOfPoints() > 0;
            if (d->hasMeshCells && cut3dOk) {
                mapper3D->SetInputConnection(
                        d->sliceCutters3D[i]->GetOutputPort());
            } else {
                mapper3D->SetInputData(d->entityPolyData);
            }
            mapper3D->ScalarVisibilityOn();

            double sliceContourColors[3][3] = {
                    {1.0, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};
            d->sliceActors3D[i] = vtkSmartPointer<vtkActor>::New();
            d->sliceActors3D[i]->SetMapper(mapper3D);
            d->sliceActors3D[i]->GetProperty()->SetColor(
                    sliceContourColors[i]);
            d->sliceActors3D[i]->GetProperty()->SetLineWidth(2.0);
            d->sliceActors3D[i]->GetProperty()->SetAmbient(1.0);
            d->sliceActors3D[i]->GetProperty()->SetDiffuse(0.0);
            d->sliceActors3D[i]->GetProperty()->LightingOff();
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->sliceActors3D[i]);
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (d->planeIndicators[i]) {
            d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->planeIndicators[i]);
            d->planeIndicators[i] = nullptr;
        }
    }
    if (box.isValid()) {
        double bounds[6] = {box.minCorner().x, box.maxCorner().x,
                            box.minCorner().y, box.maxCorner().y,
                            box.minCorner().z, box.maxCorner().z};
        setGeometryBounds(bounds);

        double sx = bounds[1] - bounds[0];
        double sy = bounds[3] - bounds[2];
        double sz = bounds[5] - bounds[4];
        double pad = 0.05;
        d->planeInitCenter[0] = cx;
        d->planeInitCenter[1] = cy;
        d->planeInitCenter[2] = cz;

        double planeColors[3][3] = {
                {1.0, 0.3, 0.3}, {0.3, 1.0, 0.3}, {0.3, 0.3, 1.0}};
        double edgeColors[3][3] = {
                {1.0, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};

        {
            auto ps = vtkSmartPointer<vtkPlaneSource>::New();
            ps->SetOrigin(bounds[0] - sx * pad, cy, bounds[4] - sz * pad);
            ps->SetPoint1(bounds[1] + sx * pad, cy, bounds[4] - sz * pad);
            ps->SetPoint2(bounds[0] - sx * pad, cy, bounds[5] + sz * pad);
            ps->Update();
            vtkNew<vtkPolyDataMapper> m;
            m->SetInputData(ps->GetOutput());
            d->planeIndicators[TOP_VIEW] = vtkSmartPointer<vtkActor>::New();
            d->planeIndicators[TOP_VIEW]->SetMapper(m);
            d->planeIndicators[TOP_VIEW]->GetProperty()->SetColor(planeColors[0]);
            d->planeIndicators[TOP_VIEW]->GetProperty()->SetOpacity(0.12);
            d->planeIndicators[TOP_VIEW]->GetProperty()->LightingOff();
            d->planeIndicators[TOP_VIEW]->GetProperty()->SetEdgeVisibility(true);
            d->planeIndicators[TOP_VIEW]->GetProperty()->SetEdgeColor(edgeColors[0]);
            d->planeIndicators[TOP_VIEW]->GetProperty()->SetLineWidth(1.5);
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->planeIndicators[TOP_VIEW]);
        }
        {
            auto ps = vtkSmartPointer<vtkPlaneSource>::New();
            ps->SetOrigin(cx, bounds[2] - sy * pad, bounds[4] - sz * pad);
            ps->SetPoint1(cx, bounds[3] + sy * pad, bounds[4] - sz * pad);
            ps->SetPoint2(cx, bounds[2] - sy * pad, bounds[5] + sz * pad);
            ps->Update();
            vtkNew<vtkPolyDataMapper> m;
            m->SetInputData(ps->GetOutput());
            d->planeIndicators[SIDE_VIEW] = vtkSmartPointer<vtkActor>::New();
            d->planeIndicators[SIDE_VIEW]->SetMapper(m);
            d->planeIndicators[SIDE_VIEW]->GetProperty()->SetColor(planeColors[1]);
            d->planeIndicators[SIDE_VIEW]->GetProperty()->SetOpacity(0.12);
            d->planeIndicators[SIDE_VIEW]->GetProperty()->LightingOff();
            d->planeIndicators[SIDE_VIEW]->GetProperty()->SetEdgeVisibility(true);
            d->planeIndicators[SIDE_VIEW]->GetProperty()->SetEdgeColor(edgeColors[1]);
            d->planeIndicators[SIDE_VIEW]->GetProperty()->SetLineWidth(1.5);
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->planeIndicators[SIDE_VIEW]);
        }
        {
            auto ps = vtkSmartPointer<vtkPlaneSource>::New();
            ps->SetOrigin(bounds[0] - sx * pad, bounds[2] - sy * pad, cz);
            ps->SetPoint1(bounds[1] + sx * pad, bounds[2] - sy * pad, cz);
            ps->SetPoint2(bounds[0] - sx * pad, bounds[3] + sy * pad, cz);
            ps->Update();
            vtkNew<vtkPolyDataMapper> m;
            m->SetInputData(ps->GetOutput());
            d->planeIndicators[FRONT_VIEW] = vtkSmartPointer<vtkActor>::New();
            d->planeIndicators[FRONT_VIEW]->SetMapper(m);
            d->planeIndicators[FRONT_VIEW]->GetProperty()->SetColor(planeColors[2]);
            d->planeIndicators[FRONT_VIEW]->GetProperty()->SetOpacity(0.12);
            d->planeIndicators[FRONT_VIEW]->GetProperty()->LightingOff();
            d->planeIndicators[FRONT_VIEW]->GetProperty()->SetEdgeVisibility(true);
            d->planeIndicators[FRONT_VIEW]->GetProperty()->SetEdgeColor(edgeColors[2]);
            d->planeIndicators[FRONT_VIEW]->GetProperty()->SetLineWidth(1.5);
            d->renderers[PERSPECTIVE_VIEW]->AddActor(d->planeIndicators[FRONT_VIEW]);
        }

        setSlicePosition(cx, cy, cz);

        double stepX = sx * 0.1;
        double stepY = sy * 0.1;
        double stepZ = sz * 0.1;
        double step = std::max({stepX, stepY, stepZ});
        if (step < 1e-6) step = 1.0;
        m_sliceIncrements[0] = stepX > 1e-6 ? stepX : step;
        m_sliceIncrements[1] = stepY > 1e-6 ? stepY : step;
        m_sliceIncrements[2] = stepZ > 1e-6 ? stepZ : step;
        if (m_stepSpin) m_stepSpin->setValue(step);
        for (int i = 0; i < 3; ++i) {
            if (m_sliceSpin[i]) m_sliceSpin[i]->setSingleStep(
                    m_sliceIncrements[i]);
        }
    }

    if (m_coloringCombo && d->entityPolyData) {
        m_coloringCombo->blockSignals(true);
        int oldIdx = m_coloringCombo->currentIndex();
        m_coloringCombo->clear();
        m_coloringCombo->addItem(tr("Solid Color"));
        auto* pd = d->entityPolyData->GetPointData();
        if (pd) {
            if (pd->GetScalars())
                m_coloringCombo->addItem(tr("Points"));
            if (pd->GetNormals())
                m_coloringCombo->addItem(tr("Normals"));
            if (pd->GetTCoords())
                m_coloringCombo->addItem(tr("TCoords"));
            for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
                QString name = pd->GetArrayName(i);
                if (name == "Normals" || name == "TCoords" || name.isEmpty())
                    continue;
                if (pd->GetScalars() &&
                    QString(pd->GetScalars()->GetName()) == name)
                    continue;
                m_coloringCombo->addItem(name);
            }
        }
        if (oldIdx >= 0 && oldIdx < m_coloringCombo->count())
            m_coloringCombo->setCurrentIndex(oldIdx);
        m_coloringCombo->blockSignals(false);
    }

    if (m_statusLabel)
        m_statusLabel->setText(
                tr("%1 (%2 pts)").arg(entity->getName()).arg(cloud->size()));

    applyDisplayProperties();
    resetCameras();

    QTimer::singleShot(100, this, [this]() {
        resetCameras();
        render();
    });
}

void vtkOrthoSliceViewWidget::loadEntitiesIntoView(
        const QList<ccHObject*>& entities) {
    if (entities.isEmpty()) {
        loadEntityIntoView(nullptr);
        return;
    }
    if (entities.size() == 1) {
        loadEntityIntoView(entities.first());
        return;
    }

    loadEntityIntoView(entities.first());
    d->extraSlices.clear();

    for (int ei = 1; ei < entities.size(); ++ei) {
        ccHObject* entity = entities[ei];
        if (!entity) continue;

        auto* mesh = ccHObjectCaster::ToMesh(entity);
        auto* genericMesh = ccHObjectCaster::ToGenericMesh(entity);
        auto* cloud = ccHObjectCaster::ToGenericPointCloud(entity);
        if (!cloud && genericMesh)
            cloud = ccHObjectCaster::ToGenericPointCloud(
                    genericMesh->getAssociatedCloud());
        for (unsigned ci = 0;
             (!genericMesh || !cloud) && ci < entity->getChildrenNumber();
             ++ci) {
            auto* child = entity->getChild(ci);
            if (!genericMesh) {
                genericMesh = ccHObjectCaster::ToGenericMesh(child);
                if (!mesh) mesh = ccHObjectCaster::ToMesh(child);
            }
            if (!cloud) cloud = ccHObjectCaster::ToGenericPointCloud(child);
        }
        if (genericMesh && !cloud)
            cloud = ccHObjectCaster::ToGenericPointCloud(
                    genericMesh->getAssociatedCloud());
        if (!mesh && genericMesh)
            mesh = ccHObjectCaster::ToMesh(genericMesh);
        if (!cloud) continue;

        auto* pcCloud = ccHObjectCaster::ToPointCloud(entity);
        if (!pcCloud && genericMesh)
            pcCloud = ccHObjectCaster::ToPointCloud(
                    genericMesh->getAssociatedCloud());
        if (!pcCloud && mesh)
            pcCloud = ccHObjectCaster::ToPointCloud(
                    mesh->getAssociatedCloud());
        if (!pcCloud) pcCloud = ccHObjectCaster::ToPointCloud(cloud);

        vtkSmartPointer<vtkPolyData> polyData;
        ccGenericMesh* meshForConvert = genericMesh ? genericMesh : mesh;
        if (meshForConvert && pcCloud)
            polyData = Converters::Cc2Vtk::MeshToPolyData(pcCloud,
                                                          meshForConvert);
        if (polyData && polyData->GetNumberOfPoints() == 0)
            polyData = nullptr;
        if (!polyData && pcCloud)
            polyData = Converters::Cc2Vtk::PointCloudToPolyData(pcCloud);
        if (!polyData) continue;

        bool hasMesh = (polyData->GetNumberOfCells() > 0 &&
                        polyData->GetNumberOfPolys() > 0);

        {
            vtkNew<vtkPolyDataMapper> modelMapper;
            modelMapper->SetInputData(polyData);
            modelMapper->ScalarVisibilityOff();
            modelMapper->Update();
            auto actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(modelMapper);
            actor->GetProperty()->SetRepresentationToWireframe();
            actor->GetProperty()->SetColor(0.3, 0.3, 0.3);
            actor->GetProperty()->SetAmbient(1.0);
            actor->GetProperty()->SetDiffuse(0.0);
            actor->GetProperty()->LightingOff();
            d->renderers[PERSPECTIVE_VIEW]->AddActor(actor);
        }

        const double normals[3][3] = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
        double origin[3] = {d->slicePos[0], d->slicePos[1], d->slicePos[2]};
        Impl::ExtraSliceData extraData;
        for (int i = 0; i < 3; ++i) {
            auto plane = vtkSmartPointer<vtkPlane>::New();
            plane->SetNormal(normals[i]);
            plane->SetOrigin(origin);
            extraData.planes[i] = plane;

            auto cutter2D = vtkSmartPointer<vtkCutter>::New();
            cutter2D->SetCutFunction(plane);
            cutter2D->SetInputData(polyData);
            cutter2D->Update();
            extraData.cutters2D[i] = cutter2D;

            vtkNew<vtkPolyDataMapper> mapper;
            bool cutOk = cutter2D->GetOutput() &&
                         cutter2D->GetOutput()->GetNumberOfPoints() > 0;
            if (hasMesh && cutOk)
                mapper->SetInputConnection(cutter2D->GetOutputPort());
            else
                mapper->SetInputData(polyData);
            mapper->ScalarVisibilityOn();

            auto actor = vtkSmartPointer<vtkActor>::New();
            actor->SetMapper(mapper);
            if (hasMesh && cutOk) {
                actor->GetProperty()->SetLineWidth(3.0);
                actor->GetProperty()->SetAmbient(1.0);
                actor->GetProperty()->SetDiffuse(0.0);
                actor->GetProperty()->LightingOff();
            } else {
                actor->GetProperty()->SetPointSize(3);
                actor->GetProperty()->SetRepresentationToPoints();
                actor->GetProperty()->LightingOff();
            }
            d->renderers[i]->AddActor(actor);

            auto cutter3D = vtkSmartPointer<vtkCutter>::New();
            cutter3D->SetCutFunction(plane);
            cutter3D->SetInputData(polyData);
            cutter3D->Update();
            extraData.cutters3D[i] = cutter3D;

            double contourColors[3][3] = {
                    {1.0, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};
            vtkNew<vtkPolyDataMapper> mapper3D;
            bool cut3dOk = cutter3D->GetOutput() &&
                           cutter3D->GetOutput()->GetNumberOfPoints() > 0;
            if (hasMesh && cut3dOk)
                mapper3D->SetInputConnection(cutter3D->GetOutputPort());
            else
                mapper3D->SetInputData(polyData);
            mapper3D->ScalarVisibilityOn();

            auto actor3D = vtkSmartPointer<vtkActor>::New();
            actor3D->SetMapper(mapper3D);
            actor3D->GetProperty()->SetColor(contourColors[i]);
            actor3D->GetProperty()->SetLineWidth(2.0);
            actor3D->GetProperty()->SetAmbient(1.0);
            actor3D->GetProperty()->SetDiffuse(0.0);
            actor3D->GetProperty()->LightingOff();
            d->renderers[PERSPECTIVE_VIEW]->AddActor(actor3D);
        }
        d->extraSlices.append(std::move(extraData));
    }

    if (m_statusLabel)
        m_statusLabel->setText(tr("All (%1 objects)").arg(entities.size()));
    render();
}

void vtkOrthoSliceViewWidget::onSourceComboAboutToShow() {
    refreshSourceCombo();
}
