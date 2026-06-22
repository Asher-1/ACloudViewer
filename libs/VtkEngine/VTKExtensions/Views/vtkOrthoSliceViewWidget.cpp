// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkOrthoSliceViewWidget.h"

#include <CVLog.h>
#include <Converters/Cc2Vtk.h>
#include <QVTKOpenGLNativeWidget.h>
#include <Shortcuts/ecvKeySequences.h>
#include <Tools/SelectionTools/cvSelectionHighlighter.h>
#include <VTKExtensions/Views/GridAxes/vtkGridAxesActor3D.h>
#include <VTKExtensions/Views/GridAxes/vtkGridAxesHelper.h>
#include <VTKExtensions/Views/vtkPVAxesWidget.h>
#include <VTKExtensions/Views/vtkPVCenterAxesActor.h>
#include <Visualization/VtkVis.h>
#include <ecvGenericMesh.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvViewTitleRegistry.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkAlgorithmOutput.h>
#include <vtkAreaPicker.h>
#include <vtkCamera.h>
#include <vtkCameraOrientationRepresentation.h>
#include <vtkCameraOrientationWidget.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCellPicker.h>
#include <vtkCommand.h>
#include <vtkCutter.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkExtractCells.h>
#include <vtkExtractSelectedFrustum.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGeometryFilter.h>
#include <vtkIdList.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkOutlineSource.h>
#include <vtkPlane.h>
#include <vtkPlaneSource.h>
#include <vtkPlanes.h>
#include <vtkPointData.h>
#include <vtkPointPicker.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataPlaneClipper.h>
#include <vtkProp.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTriangleFilter.h>
#include <vtkWeakPointer.h>

#include <QCheckBox>
#include <QComboBox>
#include <QContextMenuEvent>
#include <QCursor>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QResizeEvent>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

#include "vtkOrthoSliceRepresentation.h"

static const char* kViewLabels[] = {"Top View", "Right Side View", "Front View",
                                    "3D View"};

static ccBBox mergeEntitiesBounds(const QList<ccHObject*>& entities) {
    ccBBox merged;
    for (ccHObject* entity : entities) {
        if (!entity) continue;
        ccBBox b = entity->getBB_recursive(false, true);
        if (b.isValid()) merged += b;
    }
    return merged;
}

static vtkSmartPointer<vtkPolyDataPlaneClipper> createSliceClipper(
        vtkPolyData* input, vtkPlane* plane) {
    vtkNew<vtkTriangleFilter> tri;
    tri->SetInputData(input);
    tri->Update();
    auto clipper = vtkSmartPointer<vtkPolyDataPlaneClipper>::New();
    clipper->SetInputConnection(tri->GetOutputPort());
    clipper->SetPlane(plane);
    clipper->SetCapping(true);
    clipper->SetClippingLoops(true);
    clipper->Update();
    return clipper;
}

static bool hasValidSliceGeometry(vtkPolyData* pd) {
    return pd && pd->GetNumberOfPoints() > 0 &&
           (pd->GetNumberOfPolys() > 0 || pd->GetNumberOfLines() > 0 ||
            pd->GetNumberOfVerts() > 0);
}

static vtkPolyData* emptySlicePolyData() {
    static vtkNew<vtkPolyData> empty;
    return empty.GetPointer();
}

static void assignSliceMapperInput(vtkPolyDataMapper* mapper,
                                   vtkPolyData* sliceData) {
    if (!mapper) return;
    if (hasValidSliceGeometry(sliceData))
        mapper->SetInputData(sliceData);
    else
        mapper->SetInputData(emptySlicePolyData());
}

static vtkPolyData* getSliceCapSurface(vtkPolyDataPlaneClipper* clipper) {
    if (!clipper) return nullptr;
    vtkPolyData* cap = clipper->GetCap();
    if (cap && cap->GetNumberOfPoints() > 0) return cap;
    return nullptr;
}

static vtkPolyData* getSliceDisplayPolyData(vtkPolyDataPlaneClipper* clipper,
                                            vtkCutter* cutter) {
    // ParaView slice geometry is the plane cap or cutter contour only.
    // Never use clipper->GetOutput() — that is the clipped half-volume mesh.
    if (auto* cap = getSliceCapSurface(clipper)) return cap;
    if (cutter && cutter->GetOutput() &&
        cutter->GetOutput()->GetNumberOfPoints() > 0)
        return cutter->GetOutput();
    return nullptr;
}

// ParaView vtkPVInteractorStyle::TranslateCamera
static void translateCamera(
        vtkRenderer* renderer, int toX, int toY, int fromX, int fromY) {
    if (!renderer) return;
    vtkCamera* cam = renderer->GetActiveCamera();
    if (!cam) return;

    double viewFocus[4], focalDepth, viewPoint[3];
    double newPickPoint[4], oldPickPoint[4], motionVector[3];
    cam->GetFocalPoint(viewFocus);

    renderer->SetWorldPoint(viewFocus[0], viewFocus[1], viewFocus[2], 1.0);
    renderer->WorldToDisplay();
    renderer->GetDisplayPoint(viewFocus);
    focalDepth = viewFocus[2];

    renderer->SetDisplayPoint(toX, toY, focalDepth);
    renderer->DisplayToWorld();
    renderer->GetWorldPoint(newPickPoint);

    renderer->SetDisplayPoint(fromX, fromY, focalDepth);
    renderer->DisplayToWorld();
    renderer->GetWorldPoint(oldPickPoint);

    motionVector[0] = oldPickPoint[0] - newPickPoint[0];
    motionVector[1] = oldPickPoint[1] - newPickPoint[1];
    motionVector[2] = oldPickPoint[2] - newPickPoint[2];

    cam->GetFocalPoint(viewFocus);
    cam->GetPosition(viewPoint);
    cam->SetFocalPoint(motionVector[0] + viewFocus[0],
                       motionVector[1] + viewFocus[1],
                       motionVector[2] + viewFocus[2]);
    cam->SetPosition(motionVector[0] + viewPoint[0],
                     motionVector[1] + viewPoint[1],
                     motionVector[2] + viewPoint[2]);
}

// ParaView vtkPVInteractorStyle::DollyToPosition (parallel projection)
static void dollyParallelToPosition(vtkRenderer* ren,
                                    double fact,
                                    double anchorX,
                                    double anchorY) {
    if (!ren || !ren->GetRenderWindow() || std::abs(fact - 1.0) < 1e-12) return;
    auto* cam = ren->GetActiveCamera();
    if (!cam || !cam->GetParallelProjection()) return;

    int* winSize = ren->GetRenderWindow()->GetSize();
    if (!winSize || winSize[0] <= 0 || winSize[1] <= 0) return;

    const int cx = winSize[0] / 2;
    const int cy = winSize[1] / 2;
    const int ax = static_cast<int>(anchorX);
    const int ay = static_cast<int>(anchorY);

    translateCamera(ren, cx, cy, ax, ay);
    cam->SetParallelScale(cam->GetParallelScale() / fact);
    translateCamera(ren, ax, ay, cx, cy);
}

static void configureCameraOrientationRepresentation(
        vtkCameraOrientationRepresentation* rep) {
    if (!rep) return;
    rep->SetSize(100, 100);
    rep->AnchorToUpperRight();
    vtkTextProperty* labels[] = {
            rep->GetXPlusLabelProperty(),  rep->GetYPlusLabelProperty(),
            rep->GetZPlusLabelProperty(),  rep->GetXMinusLabelProperty(),
            rep->GetYMinusLabelProperty(), rep->GetZMinusLabelProperty()};
    for (auto* tp : labels) {
        if (!tp) continue;
        tp->SetFontSize(128);
        tp->SetFontFamilyToArial();
        tp->BoldOn();
        tp->SetJustificationToCentered();
        tp->SetVerticalJustificationToCentered();
    }
    rep->Modified();
}

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
    vtkSmartPointer<vtkPolyDataPlaneClipper> sliceClippers[3];
    vtkSmartPointer<vtkCutter> sliceCutters[3];
    vtkSmartPointer<vtkActor> sliceActors[3];
    vtkSmartPointer<vtkActor> sliceActors3D[3];
    vtkSmartPointer<vtkActor> fullModelActor;
    vtkSmartPointer<vtkActor> outlineActor;
    vtkSmartPointer<vtkOutlineSource> outlineSource;
    vtkSmartPointer<vtkActor> planeIndicators[3];
    vtkSmartPointer<vtkPlaneSource> planeSources[3];
    double planeInitCenter[3] = {0, 0, 0};
    vtkSmartPointer<vtkGridAxesActor3D> gridAxes[4];
    double slicePos[3] = {0.0, 0.0, 0.0};
    double geomBounds[6] = {-1, 1, -1, 1, -1, 1};
    bool hasMeshCells = false;
    vtkSmartPointer<vtkPVAxesWidget> orientWidget;
    vtkSmartPointer<vtkCameraOrientationWidget> cameraOrientWidget;

    struct ExtraSliceData {
        vtkSmartPointer<vtkPlane> planes[3];
        vtkSmartPointer<vtkPolyDataPlaneClipper> clippers[3];
        vtkSmartPointer<vtkCutter> cutters[3];
        vtkSmartPointer<vtkActor> actors2D[3];
        vtkSmartPointer<vtkActor> actors3D[3];
    };
    QList<ExtraSliceData> extraSlices;
    QList<std::shared_ptr<vtkOrthoSliceRepresentation>> sliceRepresentations;
    vtkSmartPointer<vtkActor> selectionHighlightActor;
};

vtkOrthoSliceViewWidget::vtkOrthoSliceViewWidget(QWidget* parent)
    : QWidget(parent), d(new Impl) {
    m_viewTypeKey = QStringLiteral("Orthographic Slice View");
    m_title = ecvViewTitleRegistry::instance().allocate(m_viewTypeKey);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->setSizeConstraint(QLayout::SetNoConstraint);
    setMinimumSize(200, 150);

    auto* decoratorBar = new QWidget(this);
    decoratorBar->setObjectName("OrthoDecoratorBar");
    m_decoratorBar = decoratorBar;
    auto* decLayout = new QHBoxLayout(decoratorBar);
    decLayout->setContentsMargins(0, 0, 0, 0);
    decLayout->setSpacing(1);

    auto* showingLabel =
            new QLabel(QStringLiteral("<b>Showing  </b>"), decoratorBar);
    decLayout->addWidget(showingLabel);

    m_sourceCombo = new QComboBox(decoratorBar);
    m_sourceCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_sourceCombo->addItem(tr("None"), QVariant::fromValue<quintptr>(0));
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
        m_sliceSpin[i]->setMaximumWidth(80);
        sliceLayout->addWidget(m_sliceSpin[i]);
        connect(m_sliceSpin[i],
                QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this]() {
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
    m_stepSpin->setMaximumWidth(65);
    sliceLayout->addWidget(m_stepSpin);
    connect(m_stepSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double val) {
                double maxInc =
                        std::max({m_sliceIncrements[0], m_sliceIncrements[1],
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
        m_axesGridVisible = visible;
        for (int i = 0; i < 4; ++i) {
            if (!d->gridAxes[i] || !d->renderers[i]) continue;
            d->gridAxes[i]->SetGridBounds(d->geomBounds);
            d->gridAxes[i]->SetGenerateGrid(false);
            d->gridAxes[i]->SetGenerateEdges(true);
            d->gridAxes[i]->SetGenerateTicks(true);
            d->gridAxes[i]->SetVisibility(visible ? 1 : 0);
            d->gridAxes[i]->Modified();
        }
        resetCameras();
    });

    sliceLayout->addSpacing(8);
    auto* resetCamBtn = new QPushButton(tr("Reset"), sliceBar);
    resetCamBtn->setToolTip(tr("Reset all cameras to fit data"));
    resetCamBtn->setMaximumWidth(48);
    sliceLayout->addWidget(resetCamBtn);
    connect(resetCamBtn, &QPushButton::clicked, this,
            &vtkOrthoSliceViewWidget::resetCameras);

    auto* centerBtn = new QPushButton(tr("Center"), sliceBar);
    centerBtn->setToolTip(tr("Center slices on geometry bounds"));
    centerBtn->setMaximumWidth(48);
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
    m_dispBar = dispBar;
    auto* dispLayout = new QHBoxLayout(dispBar);
    dispLayout->setContentsMargins(2, 1, 2, 1);
    dispLayout->setSpacing(4);

    dispLayout->addWidget(new QLabel(tr("Repr:"), dispBar));
    m_reprCombo = new QComboBox(dispBar);
    m_reprCombo->addItems({tr("Slices"), tr("Surface"), tr("Wireframe"),
                           tr("Points"), tr("Surface With Edges"),
                           tr("Feature Edges"), tr("Outline")});
    m_reprCombo->setCurrentIndex(0);
    dispLayout->addWidget(m_reprCombo);
    connect(m_reprCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { applyDisplayProperties(); });

    dispLayout->addSpacing(6);
    dispLayout->addWidget(new QLabel(tr("Opacity:"), dispBar));
    m_opacitySlider = new QSlider(Qt::Horizontal, dispBar);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(100);
    m_opacitySlider->setMaximumWidth(80);
    dispLayout->addWidget(m_opacitySlider);
    m_opacityLabel = new QLabel(QStringLiteral("1.0"), dispBar);
    m_opacityLabel->setMaximumWidth(28);
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
    m_pointSizeSpin->setMaximumWidth(50);
    dispLayout->addWidget(m_pointSizeSpin);
    connect(m_pointSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { applyDisplayProperties(); });

    dispLayout->addSpacing(6);
    dispLayout->addWidget(new QLabel(tr("Line W:"), dispBar));
    m_lineWidthSpin = new QDoubleSpinBox(dispBar);
    m_lineWidthSpin->setRange(0.1, 10.0);
    m_lineWidthSpin->setSingleStep(0.5);
    m_lineWidthSpin->setValue(1.0);
    m_lineWidthSpin->setMaximumWidth(55);
    dispLayout->addWidget(m_lineWidthSpin);
    connect(m_lineWidthSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { applyDisplayProperties(); });

    dispLayout->addStretch(1);
    layout->addWidget(dispBar);

    // === Row 4: Coloring & Scalar Coloring (ParaView Properties panel) ===
    auto* colorBar = new QWidget(this);
    colorBar->setObjectName("OrthoColorBar");
    m_colorBar = colorBar;
    auto* colorLayout = new QHBoxLayout(colorBar);
    colorLayout->setContentsMargins(2, 1, 2, 1);
    colorLayout->setSpacing(4);

    colorLayout->addWidget(new QLabel(tr("Coloring:"), colorBar));
    m_coloringCombo = new QComboBox(colorBar);
    m_coloringCombo->addItems(
            {tr("Solid Color"), tr("Points"), tr("Normals"), tr("TCoords")});
    colorLayout->addWidget(m_coloringCombo);
    connect(m_coloringCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { applyDisplayProperties(); });

    colorLayout->addSpacing(8);
    m_mapScalarsCheck = new QCheckBox(tr("Map Scalars"), colorBar);
    m_mapScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_mapScalarsCheck);
    connect(m_mapScalarsCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    m_interpScalarsCheck = new QCheckBox(tr("Interp Scalars"), colorBar);
    m_interpScalarsCheck->setChecked(true);
    colorLayout->addWidget(m_interpScalarsCheck);
    connect(m_interpScalarsCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    m_useNanColorCheck = new QCheckBox(tr("Nan Color"), colorBar);
    m_useNanColorCheck->setToolTip(tr("Use Nan Color For Missing Arrays"));
    colorLayout->addWidget(m_useNanColorCheck);
    connect(m_useNanColorCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    colorLayout->addSpacing(8);
    m_renderTubesCheck = new QCheckBox(tr("Tubes"), colorBar);
    m_renderTubesCheck->setToolTip(tr("Render Lines As Tubes"));
    colorLayout->addWidget(m_renderTubesCheck);
    connect(m_renderTubesCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    m_renderSpheresCheck = new QCheckBox(tr("Spheres"), colorBar);
    m_renderSpheresCheck->setToolTip(tr("Render Points As Spheres"));
    colorLayout->addWidget(m_renderSpheresCheck);
    connect(m_renderSpheresCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    m_showOutlineCheck = new QCheckBox(tr("Outline"), colorBar);
    m_showOutlineCheck->setToolTip(tr("Show bounding outline"));
    m_showOutlineCheck->setChecked(false);
    colorLayout->addWidget(m_showOutlineCheck);
    connect(m_showOutlineCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    colorLayout->addStretch(1);
    layout->addWidget(colorBar);

    // === Row 5: Lighting (ParaView Properties panel) ===
    auto* lightBar = new QWidget(this);
    lightBar->setObjectName("OrthoLightBar");
    m_lightBar = lightBar;
    auto* lightLayout = new QHBoxLayout(lightBar);
    lightLayout->setContentsMargins(2, 1, 2, 1);
    lightLayout->setSpacing(4);

    m_disableLightingCheck = new QCheckBox(tr("No Light"), lightBar);
    lightLayout->addWidget(m_disableLightingCheck);
    connect(m_disableLightingCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Diffuse:"), lightBar));
    m_diffuseSlider = new QSlider(Qt::Horizontal, lightBar);
    m_diffuseSlider->setRange(0, 100);
    m_diffuseSlider->setValue(100);
    m_diffuseSlider->setMaximumWidth(60);
    lightLayout->addWidget(m_diffuseSlider);
    m_diffuseLabel = new QLabel(QStringLiteral("1.0"), lightBar);
    m_diffuseLabel->setMaximumWidth(24);
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
    m_specularSlider->setMaximumWidth(50);
    lightLayout->addWidget(m_specularSlider);
    m_specularLabel = new QLabel(QStringLiteral("0"), lightBar);
    m_specularLabel->setMaximumWidth(20);
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
    m_specPowerSpin->setMaximumWidth(50);
    lightLayout->addWidget(m_specPowerSpin);
    connect(m_specPowerSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    m_specColorCheck = new QCheckBox(tr("Spec Color"), lightBar);
    m_specColorCheck->setToolTip(tr("Specular Color (ParaView)"));
    lightLayout->addWidget(m_specColorCheck);
    connect(m_specColorCheck, &QCheckBox::toggled, this,
            [this](bool) { applyDisplayProperties(); });

    lightLayout->addSpacing(4);
    lightLayout->addWidget(new QLabel(tr("Lum:"), lightBar));
    m_luminositySpin = new QDoubleSpinBox(lightBar);
    m_luminositySpin->setRange(0.0, 1.0);
    m_luminositySpin->setSingleStep(0.1);
    m_luminositySpin->setDecimals(1);
    m_luminositySpin->setValue(0.0);
    m_luminositySpin->setMaximumWidth(50);
    m_luminositySpin->setToolTip(tr("Luminosity (ParaView Ambient)"));
    lightLayout->addWidget(m_luminositySpin);
    connect(m_luminositySpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { applyDisplayProperties(); });

    lightLayout->addStretch(1);
    layout->addWidget(lightBar);
    dispBar->setVisible(true);
    colorBar->setVisible(false);
    colorBar->setMaximumHeight(0);
    lightBar->setVisible(false);
    lightBar->setMaximumHeight(0);

    d->vtkWidget = new QVTKOpenGLNativeWidget(this);
    d->vtkWidget->setMinimumSize(0, 0);
    d->vtkWidget->setContextMenuPolicy(Qt::NoContextMenu);
    layout->addWidget(d->vtkWidget, 1);

    d->renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    // ParaView vtkPVRenderView: overlay renderers (orientation widget) use
    // layer 1.
    d->renderWindow->SetNumberOfLayers(2);
    d->vtkWidget->setRenderWindow(d->renderWindow);

    for (int i = 0; i < 4; ++i) {
        d->renderers[i] = vtkSmartPointer<vtkRenderer>::New();
        // ParaView vtkPVOrthographicSliceView: 2D slice renderers use 0.5 grey.
        const double bg = (i < PERSPECTIVE_VIEW) ? 0.5 : 0.2;
        d->renderers[i]->SetBackground(bg, bg, bg);
        d->renderers[i]->SetUseFXAA(true);
        d->renderWindow->AddRenderer(d->renderers[i]);
    }
    d->renderers[PERSPECTIVE_VIEW]->SetBackground(0.22, 0.24, 0.33);
    d->renderers[PERSPECTIVE_VIEW]->SetBackground2(0.42, 0.44, 0.53);
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
        d->annotations[i]->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        d->annotations[i]->GetTextProperty()->SetBold(0);
        d->annotations[i]->GetTextProperty()->SetShadow(0);
        d->annotations[i]->GetTextProperty()->SetJustificationToLeft();
        d->annotations[i]->SetTextScaleModeToNone();
        d->annotations[i]
                ->GetPositionCoordinate()
                ->SetCoordinateSystemToNormalizedViewport();
        d->annotations[i]->GetPositionCoordinate()->SetValue(0.01, 0.02);
        d->renderers[i]->AddActor2D(d->annotations[i]);

        d->subAnnotations[i] = vtkSmartPointer<vtkTextActor>::New();
        d->subAnnotations[i]->SetInput("");
        d->subAnnotations[i]->GetTextProperty()->SetFontSize(12);
        d->subAnnotations[i]->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        d->subAnnotations[i]->GetTextProperty()->SetShadow(0);
        d->subAnnotations[i]
                ->GetPositionCoordinate()
                ->SetCoordinateSystemToNormalizedViewport();
        d->subAnnotations[i]->GetPositionCoordinate()->SetValue(0.01, 0.08);
        d->subAnnotations[i]->SetVisibility(0);
        d->renderers[i]->AddActor2D(d->subAnnotations[i]);
    }

    // Viewports (ParaView quad pattern)
    d->renderers[TOP_VIEW]->SetViewport(0.0, 0.5, 0.5, 1.0);
    d->renderers[SIDE_VIEW]->SetViewport(0.5, 0.5, 1.0, 1.0);
    d->renderers[FRONT_VIEW]->SetViewport(0.0, 0.0, 0.5, 0.5);
    d->renderers[PERSPECTIVE_VIEW]->SetViewport(0.5, 0.0, 1.0, 0.5);
    // ParaView vtkPVRenderView: scene on layer 0, orientation widgets on
    // layer 1.
    for (int i = 0; i < 4; ++i) {
        d->renderers[i]->SetLayer(0);
    }

    if (auto* iren = d->renderWindow->GetInteractor()) {
        auto style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
        style->SetDefaultRenderer(d->renderers[PERSPECTIVE_VIEW]);
        iren->SetInteractorStyle(style);
        iren->Enable();
    }

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
    d->sliceAxes3D->SetUseBounds(false);
    d->sliceAxes3D->SetScale(10, 10, 10);
    d->renderers[PERSPECTIVE_VIEW]->AddActor(d->sliceAxes3D);

    // ParaView-consistent axes grid: dual face masks + FrontfaceCulling(true).
    // FrontfaceCulling ensures only the back-facing face renders per view,
    // matching ParaView's default configuration (utilities_remotingviews.xml).
    const unsigned int faceMasks[4] = {
            vtkGridAxesHelper::MAX_ZX | vtkGridAxesHelper::MIN_ZX,
            vtkGridAxesHelper::MAX_YZ | vtkGridAxesHelper::MIN_YZ,
            vtkGridAxesHelper::MAX_XY | vtkGridAxesHelper::MIN_XY,
            vtkGridAxesHelper::MIN_XY | vtkGridAxesHelper::MIN_YZ |
                    vtkGridAxesHelper::MIN_ZX | vtkGridAxesHelper::MAX_XY |
                    vtkGridAxesHelper::MAX_YZ | vtkGridAxesHelper::MAX_ZX};
    for (int i = 0; i < 4; ++i) {
        d->gridAxes[i] = vtkSmartPointer<vtkGridAxesActor3D>::New();
        d->gridAxes[i]->SetGridBounds(d->geomBounds);
        d->gridAxes[i]->SetFaceMask(faceMasks[i]);
        d->gridAxes[i]->SetLabelMask(0xff);
        d->gridAxes[i]->SetGenerateGrid(false);
        d->gridAxes[i]->SetGenerateEdges(true);
        d->gridAxes[i]->SetGenerateTicks(true);
        d->gridAxes[i]->SetForceOpaque(true);
        if (i < 3) {
            // ParaView GridAxes3DActor default: LabelUniqueEdgesOnly=1
            d->gridAxes[i]->SetLabelUniqueEdgesOnly(true);
        }
        {
            auto prop = vtkSmartPointer<vtkProperty>::New();
            prop->SetColor(1.0, 1.0, 1.0);
            prop->SetLineWidth(1.0);
            prop->SetFrontfaceCulling(true);
            d->gridAxes[i]->SetProperty(prop);
        }
        for (int ax = 0; ax < 3; ++ax) {
            if (auto* tp = d->gridAxes[i]->GetTitleTextProperty(ax)) {
                tp->SetFontSize(12);
                tp->SetColor(1.0, 1.0, 1.0);
                tp->SetBold(0);
                tp->SetShadow(0);
            }
            if (auto* lp = d->gridAxes[i]->GetLabelTextProperty(ax)) {
                lp->SetFontSize(12);
                lp->SetColor(1.0, 1.0, 1.0);
                lp->SetShadow(0);
            }
        }
        d->gridAxes[i]->SetTitle(0, "X Axis");
        d->gridAxes[i]->SetTitle(1, "Y Axis");
        d->gridAxes[i]->SetTitle(2, "Z Axis");
        d->gridAxes[i]->SetVisibility(0);
        d->renderers[i]->AddViewProp(d->gridAxes[i]);
    }

    setAnnotationsVisible(true);

    QTimer::singleShot(100, this, [this]() {
        ensureOrientationWidgetsInitialized();
        QSettings settings;
        setOrientationMarkerVisible(
                settings.value("OrientationMarker/Visible", true).toBool());
        toggleCameraOrientationWidget(
                settings.value("CameraOrientationWidget/Visible", false)
                        .toBool());
    });

    setupDecoratorBarContextMenu();

    setContextMenuPolicy(Qt::NoContextMenu);

    d->vtkWidget->installEventFilter(this);

    m_rubberBandWidget = new QRubberBand(QRubberBand::Rectangle, d->vtkWidget);
}

vtkOrthoSliceViewWidget::~vtkOrthoSliceViewWidget() {
    disconnectExternalHighlighter();
    if (!m_viewTypeKey.isEmpty() && !m_title.isEmpty()) {
        ecvViewTitleRegistry::instance().release(m_viewTypeKey, m_title);
    }
    if (d->cameraOrientWidget) {
        d->cameraOrientWidget->SetEnabled(0);
        d->cameraOrientWidget->SetInteractor(nullptr);
        d->cameraOrientWidget->SetParentRenderer(nullptr);
    }
    if (d->orientWidget) {
        d->orientWidget->SetEnabled(0);
        d->orientWidget->SetInteractor(nullptr);
        d->orientWidget->SetParentRenderer(nullptr);
    }
    if (d->renderWindow && d->renderWindow->GetInteractor())
        d->renderWindow->GetInteractor()->Enable();
    if (d->vtkWidget)
        d->vtkWidget->setRenderWindow(
                static_cast<vtkGenericOpenGLRenderWindow*>(nullptr));
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

void vtkOrthoSliceViewWidget::updatePlaneIndicatorFromSlice(
        int viewIdx,
        vtkPlane* plane,
        vtkPlaneSource* ps,
        const double bounds[6]) {
    if (!plane || !ps) return;
    const double slicePos[3] = {d->slicePos[0], d->slicePos[1], d->slicePos[2]};
    const double sx = bounds[1] - bounds[0];
    const double sy = bounds[3] - bounds[2];
    const double sz = bounds[5] - bounds[4];
    const double pad = 0.05;
    switch (viewIdx) {
        case TOP_VIEW:
            ps->SetOrigin(bounds[0] - sx * pad, slicePos[1],
                          bounds[4] - sz * pad);
            ps->SetPoint1(bounds[1] + sx * pad, slicePos[1],
                          bounds[4] - sz * pad);
            ps->SetPoint2(bounds[0] - sx * pad, slicePos[1],
                          bounds[5] + sz * pad);
            break;
        case SIDE_VIEW:
            ps->SetOrigin(slicePos[0], bounds[2] - sy * pad,
                          bounds[4] - sz * pad);
            ps->SetPoint1(slicePos[0], bounds[3] + sy * pad,
                          bounds[4] - sz * pad);
            ps->SetPoint2(slicePos[0], bounds[2] - sy * pad,
                          bounds[5] + sz * pad);
            break;
        case FRONT_VIEW:
            ps->SetOrigin(bounds[0] - sx * pad, bounds[2] - sy * pad,
                          slicePos[2]);
            ps->SetPoint1(bounds[1] + sx * pad, bounds[2] - sy * pad,
                          slicePos[2]);
            ps->SetPoint2(bounds[0] - sx * pad, bounds[3] + sy * pad,
                          slicePos[2]);
            break;
        default:
            return;
    }
    ps->Update();
}

void vtkOrthoSliceViewWidget::createPlaneIndicators(const double bounds[6]) {
    auto* perspectiveRen = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!perspectiveRen) return;

    const double planeColors[3][3] = {
            {1.0, 0.3, 0.3}, {0.3, 1.0, 0.3}, {0.3, 0.3, 1.0}};
    const double edgeColors[3][3] = {
            {1.0, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};
    const double normals[3][3] = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

    for (int i = 0; i < 3; ++i) {
        if (d->planeIndicators[i]) {
            perspectiveRen->RemoveActor(d->planeIndicators[i]);
            d->planeIndicators[i] = nullptr;
        }
        d->planeSources[i] = nullptr;
        if (!d->slicePlanes[i]) {
            d->slicePlanes[i] = vtkSmartPointer<vtkPlane>::New();
            d->slicePlanes[i]->SetNormal(normals[i]);
            d->slicePlanes[i]->SetOrigin(d->slicePos[0], d->slicePos[1],
                                         d->slicePos[2]);
        }
        d->planeSources[i] = vtkSmartPointer<vtkPlaneSource>::New();
        updatePlaneIndicatorFromSlice(i, d->slicePlanes[i],
                                      d->planeSources[i].GetPointer(), bounds);
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(d->planeSources[i]->GetOutputPort());
        d->planeIndicators[i] = vtkSmartPointer<vtkActor>::New();
        d->planeIndicators[i]->SetMapper(mapper);
        d->planeIndicators[i]->GetProperty()->SetColor(
                planeColors[i][0], planeColors[i][1], planeColors[i][2]);
        d->planeIndicators[i]->GetProperty()->SetOpacity(0.12);
        d->planeIndicators[i]->GetProperty()->LightingOff();
        d->planeIndicators[i]->GetProperty()->SetEdgeVisibility(true);
        d->planeIndicators[i]->GetProperty()->SetEdgeColor(
                edgeColors[i][0], edgeColors[i][1], edgeColors[i][2]);
        d->planeIndicators[i]->GetProperty()->SetLineWidth(1.5);
        perspectiveRen->AddActor(d->planeIndicators[i]);
    }
}

void vtkOrthoSliceViewWidget::setSlicePosition(double x, double y, double z) {
    d->slicePos[0] = x;
    d->slicePos[1] = y;
    d->slicePos[2] = z;

    char buf[128];
    snprintf(buf, sizeof(buf), "Top View (Y=%g)\nZ=%g, X=%g", d->slicePos[1],
             d->slicePos[2], d->slicePos[0]);
    d->annotations[TOP_VIEW]->SetInput(buf);

    snprintf(buf, sizeof(buf), "Right Side View (X=%g)\nY=%g, Z=%g",
             d->slicePos[0], d->slicePos[1], d->slicePos[2]);
    d->annotations[SIDE_VIEW]->SetInput(buf);

    snprintf(buf, sizeof(buf), "Front View (Z=%g)\nX=%g, Y=%g", d->slicePos[2],
             d->slicePos[0], d->slicePos[1]);
    d->annotations[FRONT_VIEW]->SetInput(buf);

    // Move crosshair axes (ParaView SetSlicePosition pattern)
    // Small offset avoids Z-fighting with the slice plane
    d->sliceAxes2D[TOP_VIEW]->SetPosition(x, y + 0.01, z);
    d->sliceAxes2D[SIDE_VIEW]->SetPosition(x + 0.01, y, z);
    d->sliceAxes2D[FRONT_VIEW]->SetPosition(x, y, z + 0.01);
    d->sliceAxes3D->SetPosition(x, y, z);

    if (d->slicePlanes[TOP_VIEW]) d->slicePlanes[TOP_VIEW]->SetOrigin(x, y, z);
    if (d->slicePlanes[SIDE_VIEW])
        d->slicePlanes[SIDE_VIEW]->SetOrigin(x, y, z);
    if (d->slicePlanes[FRONT_VIEW])
        d->slicePlanes[FRONT_VIEW]->SetOrigin(x, y, z);

    for (int i = 0; i < 3; ++i) {
        if (d->sliceClippers[i]) {
            d->sliceClippers[i]->Modified();
            d->sliceClippers[i]->Update();
        }
        if (d->sliceCutters[i]) {
            d->sliceCutters[i]->Modified();
            d->sliceCutters[i]->Update();
        }
        if (d->sliceActors[i] && d->sliceActors[i]->GetMapper()) {
            if (auto* pdMapper = vtkPolyDataMapper::SafeDownCast(
                        d->sliceActors[i]->GetMapper())) {
                if (auto* pd = getSliceDisplayPolyData(
                            d->sliceClippers[i].GetPointer(),
                            d->sliceCutters[i].GetPointer())) {
                    assignSliceMapperInput(pdMapper, pd);
                    d->sliceActors[i]->SetVisibility(
                            hasValidSliceGeometry(pd) ? 1 : 0);
                }
            }
        }
        if (d->sliceActors3D[i] && d->sliceActors3D[i]->GetMapper()) {
            if (auto* pdMapper = vtkPolyDataMapper::SafeDownCast(
                        d->sliceActors3D[i]->GetMapper())) {
                if (auto* pd = getSliceDisplayPolyData(
                            d->sliceClippers[i].GetPointer(),
                            d->sliceCutters[i].GetPointer())) {
                    assignSliceMapperInput(pdMapper, pd);
                    d->sliceActors3D[i]->SetVisibility(
                            hasValidSliceGeometry(pd) ? 1 : 0);
                }
            }
        }
    }

    for (auto& extra : d->extraSlices) {
        for (int i = 0; i < 3; ++i) {
            if (extra.planes[i]) extra.planes[i]->SetOrigin(x, y, z);
            if (extra.clippers[i]) {
                extra.clippers[i]->Modified();
                extra.clippers[i]->Update();
            }
            if (extra.cutters[i]) {
                extra.cutters[i]->Modified();
                extra.cutters[i]->Update();
            }
            if (extra.actors2D[i] && extra.actors2D[i]->GetMapper()) {
                if (auto* pdMapper = vtkPolyDataMapper::SafeDownCast(
                            extra.actors2D[i]->GetMapper())) {
                    if (auto* pd = getSliceDisplayPolyData(
                                extra.clippers[i].GetPointer(),
                                extra.cutters[i].GetPointer())) {
                        assignSliceMapperInput(pdMapper, pd);
                        extra.actors2D[i]->SetVisibility(
                                hasValidSliceGeometry(pd) ? 1 : 0);
                    }
                }
            }
            if (extra.actors3D[i] && extra.actors3D[i]->GetMapper()) {
                if (auto* pdMapper = vtkPolyDataMapper::SafeDownCast(
                            extra.actors3D[i]->GetMapper())) {
                    if (auto* pd = getSliceDisplayPolyData(
                                extra.clippers[i].GetPointer(),
                                extra.cutters[i].GetPointer())) {
                        assignSliceMapperInput(pdMapper, pd);
                        extra.actors3D[i]->SetVisibility(
                                hasValidSliceGeometry(pd) ? 1 : 0);
                    }
                }
            }
        }
    }

    if (d->planeIndicators[TOP_VIEW] && d->planeSources[TOP_VIEW] &&
        d->slicePlanes[TOP_VIEW]) {
        updatePlaneIndicatorFromSlice(TOP_VIEW, d->slicePlanes[TOP_VIEW],
                                      d->planeSources[TOP_VIEW], d->geomBounds);
        d->planeIndicators[TOP_VIEW]->SetPosition(0, 0, 0);
    }
    if (d->planeIndicators[SIDE_VIEW] && d->planeSources[SIDE_VIEW] &&
        d->slicePlanes[SIDE_VIEW]) {
        updatePlaneIndicatorFromSlice(SIDE_VIEW, d->slicePlanes[SIDE_VIEW],
                                      d->planeSources[SIDE_VIEW],
                                      d->geomBounds);
        d->planeIndicators[SIDE_VIEW]->SetPosition(0, 0, 0);
    }
    if (d->planeIndicators[FRONT_VIEW] && d->planeSources[FRONT_VIEW] &&
        d->slicePlanes[FRONT_VIEW]) {
        updatePlaneIndicatorFromSlice(FRONT_VIEW, d->slicePlanes[FRONT_VIEW],
                                      d->planeSources[FRONT_VIEW],
                                      d->geomBounds);
        d->planeIndicators[FRONT_VIEW]->SetPosition(0, 0, 0);
    }

    if (!d->sliceRepresentations.isEmpty()) {
        for (const auto& rep : d->sliceRepresentations) {
            rep->setSliceOrigin(x, y, z);
            rep->update();
        }
    }

    updateSliceSpinners();
    render();
}

void vtkOrthoSliceViewWidget::activate3DInteractor() {
    if (!d->renderWindow) return;
    auto* iren = d->renderWindow->GetInteractor();
    if (!iren) return;
    iren->Enable();
    m_interactorDisabledFor2D = false;
    if (auto* style =
                vtkInteractorStyle::SafeDownCast(iren->GetInteractorStyle()))
        style->SetCurrentRenderer(d->renderers[PERSPECTIVE_VIEW]);
}

bool vtkOrthoSliceViewWidget::forwardMouseToInteractor(QMouseEvent* me,
                                                       QEvent::Type eventType) {
    if (!me || !d->renderWindow) return false;
    auto* iren = d->renderWindow->GetInteractor();
    auto* ren = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!iren || !ren) return false;

    activate3DInteractor();

    int* winSize = d->renderWindow->GetSize();
    if (!winSize || winSize[0] <= 0 || winSize[1] <= 0) return false;

    double dispXY[2];
    mapWidgetToRendererDisplay(me->pos(), ren, dispXY);
    const int evtX = static_cast<int>(dispXY[0]);
    const int evtY = static_cast<int>(dispXY[1]);

    // ParaView / VtkVis: disable interactor style so vtkCameraOrientationWidget
    // receives press/move/release instead of trackball rotation.
    vtkInteractorObserver* oldStyle = iren->GetInteractorStyle();
    if (oldStyle) {
        oldStyle->SetCurrentRenderer(ren);
    }
    iren->SetInteractorStyle(nullptr);
    iren->SetEventInformation(evtX, evtY, me->modifiers() & Qt::ControlModifier,
                              me->modifiers() & Qt::ShiftModifier);

    switch (eventType) {
        case QEvent::MouseButtonPress:
            // vtkCameraOrientationWidget requires a prior MouseMove to enter
            // the Hot state before it accepts Select on LeftButtonPress.
            iren->MouseMoveEvent();
            if (me->button() == Qt::LeftButton)
                iren->LeftButtonPressEvent();
            else if (me->button() == Qt::MiddleButton)
                iren->MiddleButtonPressEvent();
            else if (me->button() == Qt::RightButton)
                iren->RightButtonPressEvent();
            break;
        case QEvent::MouseButtonRelease:
            if (me->button() == Qt::LeftButton)
                iren->LeftButtonReleaseEvent();
            else if (me->button() == Qt::MiddleButton)
                iren->MiddleButtonReleaseEvent();
            else if (me->button() == Qt::RightButton)
                iren->RightButtonReleaseEvent();
            break;
        case QEvent::MouseMove:
            iren->MouseMoveEvent();
            break;
        default:
            iren->SetInteractorStyle(oldStyle);
            return false;
    }
    iren->SetInteractorStyle(oldStyle);

    if (d->cameraOrientWidget && d->cameraOrientWidget->GetEnabled()) {
        d->cameraOrientWidget->Render();
    }
    if (d->vtkWidget)
        d->vtkWidget->update();
    else
        render();
    return true;
}

bool vtkOrthoSliceViewWidget::isInCameraOrientationWidget(
        const QPoint& pos) const {
    if (!isCameraOrientationWidgetShown() || !d->cameraOrientWidget ||
        !d->renderWindow)
        return false;
    auto* rep = vtkCameraOrientationRepresentation::SafeDownCast(
            d->cameraOrientWidget->GetRepresentation());
    auto* ren = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!rep || !ren) return false;

    double dispXY[2];
    const_cast<vtkOrthoSliceViewWidget*>(this)->mapWidgetToRendererDisplay(
            pos, ren, dispXY);

    int size[2] = {100, 100};
    rep->GetSize(size);

    double vp[4];
    ren->GetViewport(vp);
    int* winSize = d->renderWindow->GetSize();
    if (!winSize || winSize[0] <= 0 || winSize[1] <= 0) return false;

    const double renX1 = vp[2] * winSize[0];
    const double renY1 = vp[3] * winSize[1];
    const double x0 = renX1 - size[0];
    const double y0 = renY1 - size[1];
    // Slightly inflate hit box (ParaView uses padding on the representation).
    const double pad = std::max(8.0, static_cast<double>(size[0]) * 0.15);
    return dispXY[0] >= x0 - pad && dispXY[0] <= renX1 + pad &&
           dispXY[1] >= y0 - pad && dispXY[1] <= renY1 + pad;
}

void vtkOrthoSliceViewWidget::ensureOrientationWidgetsInitialized() {
    if (!d->renderWindow) return;
    auto* interactor = d->renderWindow->GetInteractor();
    if (!interactor) return;

    if (!d->orientWidget) {
        d->orientWidget = vtkSmartPointer<vtkPVAxesWidget>::New();
        d->orientWidget->SetOutlineColor(0.2, 0.2, 0.2);
        d->orientWidget->SetParentRenderer(d->renderers[PERSPECTIVE_VIEW]);
        d->orientWidget->SetInteractor(interactor);
    }
    updateOrientationWidgetViewport();
}

void vtkOrthoSliceViewWidget::updateOrientationWidgetViewport() {
    auto* parentRen = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!parentRen) return;

    double vp[4];
    parentRen->GetViewport(vp);
    const double vx0 = vp[0];
    const double vy0 = vp[1];
    const double vx1 = vp[0] + 0.25 * (vp[2] - vp[0]);
    const double vy1 = vp[1] + 0.25 * (vp[3] - vp[1]);

    if (d->orientWidget) {
        d->orientWidget->SetParentRenderer(parentRen);
        d->orientWidget->SetViewport(vx0, vy0, vx1, vy1);
        d->orientWidget->SetEnabled(d->orientWidget->GetVisibility() ? 1 : 0);
    }
    if (d->cameraOrientWidget) {
        d->cameraOrientWidget->SetParentRenderer(parentRen);
        if (d->cameraOrientWidget->GetEnabled()) {
            if (auto* rep = vtkCameraOrientationRepresentation::SafeDownCast(
                        d->cameraOrientWidget->GetRepresentation())) {
                rep->AnchorToUpperRight();
                d->cameraOrientWidget->SquareResize();
            }
        }
    }
}

void vtkOrthoSliceViewWidget::resetCameras() {
    // Align with ParaView vtkPVOrthographicSliceView::ResetCamera():
    // always fit 2D orthographic views to geometry bounds via VTK's own
    // ResetCamera (bounding-sphere + viewport aspect). Do NOT override
    // ParallelScale manually — that caused slice/grid overflow and
    // inconsistent zoom between views.
    const double* bounds = d->geomBounds;
    for (int i = 0; i < 3; ++i) {
        d->renderers[i]->ResetCamera(bounds);
        d->renderers[i]->ResetCameraClippingRange();
    }

    d->renderers[PERSPECTIVE_VIEW]->ResetCamera(bounds);
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
    updateOutlineBounds();
    for (int i = 0; i < 4; ++i) {
        if (d->gridAxes[i]) {
            d->gridAxes[i]->SetGridBounds(bounds);
            d->gridAxes[i]->Modified();
        }
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

void vtkOrthoSliceViewWidget::updateOutlineBounds() {
    if (!d->outlineSource) return;
    d->outlineSource->SetBounds(d->geomBounds);
    d->outlineSource->Update();
}

void vtkOrthoSliceViewWidget::createOutlineActor() {
    auto* perspectiveRen = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!perspectiveRen) return;

    if (d->outlineActor) {
        perspectiveRen->RemoveActor(d->outlineActor);
        d->outlineActor = nullptr;
    }
    d->outlineSource = nullptr;

    d->outlineSource = vtkSmartPointer<vtkOutlineSource>::New();
    d->outlineSource->SetBounds(d->geomBounds);

    vtkNew<vtkPolyDataMapper> outMapper;
    outMapper->SetInputConnection(d->outlineSource->GetOutputPort());

    d->outlineActor = vtkSmartPointer<vtkActor>::New();
    d->outlineActor->SetMapper(outMapper);
    d->outlineActor->SetUseBounds(false);
    d->outlineActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    d->outlineActor->GetProperty()->SetLineWidth(1.0);
    d->outlineActor->GetProperty()->SetRepresentationToWireframe();
    d->outlineActor->GetProperty()->LightingOff();
    d->outlineActor->GetProperty()->SetAmbient(1.0);
    d->outlineActor->GetProperty()->SetDiffuse(0.0);
    const bool showOutline =
            m_showOutlineCheck && m_showOutlineCheck->isChecked();
    d->outlineActor->SetVisibility(showOutline ? 1 : 0);
    perspectiveRen->AddActor(d->outlineActor);
}

void vtkOrthoSliceViewWidget::addActorToAll(vtkProp* actor) {
    if (!actor) return;
    for (int i = 0; i < 4; ++i) {
        d->renderers[i]->AddViewProp(actor);
    }
}

void vtkOrthoSliceViewWidget::disconnectExternalHighlighter() {
    m_externalHighlighterLinked = false;
    QObject::disconnect(m_hlActorAddedConn);
    QObject::disconnect(m_hlActorRemovedConn);
    QObject::disconnect(m_hlClearedConn);
    QObject::disconnect(m_hlOverlayConn);
    m_hlActorAddedConn = {};
    m_hlActorRemovedConn = {};
    m_hlClearedConn = {};
    m_hlOverlayConn = {};

    auto* ren3D = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    for (auto it = m_externalHighlightClones.begin();
         it != m_externalHighlightClones.end(); ++it) {
        if (it.value() && ren3D) ren3D->RemoveActor(it.value());
    }
    m_externalHighlightClones.clear();
}

void vtkOrthoSliceViewWidget::connectExternalHighlighter(QObject* highlighter) {
    disconnectExternalHighlighter();

    auto* hl = qobject_cast<cvSelectionHighlighter*>(highlighter);
    if (!hl) return;
    m_externalHighlighterLinked = true;
    auto* ren3D = d->renderers[PERSPECTIVE_VIEW].GetPointer();

    auto cloneHighlightActor = [](vtkActor* src) -> vtkSmartPointer<vtkActor> {
        if (!src) return nullptr;
        auto clone = vtkSmartPointer<vtkActor>::New();
        clone->ShallowCopy(src);
        if (auto* srcMapper =
                    vtkDataSetMapper::SafeDownCast(src->GetMapper())) {
            auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
            mapper->ShallowCopy(srcMapper);
            clone->SetMapper(mapper);
        }
        clone->SetPickable(0);
        return clone;
    };

    auto safeUpdate = [this]() {
        if (d->vtkWidget && d->vtkWidget->isVisible()) d->vtkWidget->update();
    };

    auto removeClone = [this, ren3D](vtkActor* source) {
        if (!source || !ren3D) return;
        auto it = m_externalHighlightClones.find(source);
        if (it == m_externalHighlightClones.end()) return;
        if (it.value()) ren3D->RemoveActor(it.value());
        m_externalHighlightClones.erase(it);
    };

    // SELECTED: selectionOverlayUpdated drives d->selectionHighlightActor.
    m_hlActorAddedConn =
            connect(hl, &cvSelectionHighlighter::highlightActorAdded, this,
                    [this, ren3D, cloneHighlightActor, safeUpdate,
                     removeClone](vtkActor* actor) {
                        if (!actor || !ren3D) return;
                        removeClone(actor);
                        auto clone = cloneHighlightActor(actor);
                        if (!clone) return;
                        ren3D->AddActor(clone);
                        m_externalHighlightClones.insert(actor, clone);
                        applyDisplayProperties();
                        safeUpdate();
                    });
    m_hlActorRemovedConn =
            connect(hl, &cvSelectionHighlighter::highlightActorRemoved, this,
                    [this, removeClone, safeUpdate](vtkActor* actor) {
                        removeClone(actor);
                        applyDisplayProperties();
                        safeUpdate();
                    });
    m_hlClearedConn =
            connect(hl, &cvSelectionHighlighter::highlightsCleared, this,
                    [this, ren3D, safeUpdate]() {
                        for (auto it = m_externalHighlightClones.begin();
                             it != m_externalHighlightClones.end(); ++it) {
                            if (it.value() && ren3D)
                                ren3D->RemoveActor(it.value());
                        }
                        m_externalHighlightClones.clear();
                        if (d->selectionHighlightActor && ren3D) {
                            ren3D->RemoveActor(d->selectionHighlightActor);
                            d->selectionHighlightActor = nullptr;
                        }
                        safeUpdate();
                    });

    m_hlOverlayConn = connect(
            hl, &cvSelectionHighlighter::selectionOverlayUpdated, this,
            [this, ren3D, safeUpdate](vtkPolyData* poly, int kind) {
                m_selectionOverlayKind = kind;
                if (d->selectionHighlightActor && ren3D) {
                    ren3D->RemoveActor(d->selectionHighlightActor);
                    d->selectionHighlightActor = nullptr;
                }
                if (kind != cvSelectionHighlighter::SelectionOverlayNone &&
                    ren3D) {
                    d->selectionHighlightActor =
                            cvSelectionHighlighter::createSelectionOverlayActor(
                                    poly,
                                    static_cast<cvSelectionHighlighter::
                                                        SelectionOverlayKind>(
                                            kind));
                    if (d->selectionHighlightActor) {
                        ren3D->AddActor(d->selectionHighlightActor);
                    }
                }
                if (d->selectionHighlightActor) {
                    cvSelectionHighlighter::styleSelectionOverlayProperty(
                            d->selectionHighlightActor->GetProperty(),
                            static_cast<cvSelectionHighlighter::
                                                SelectionOverlayKind>(kind));
                }
                if (d->renderWindow) d->renderWindow->Render();
                safeUpdate();
            });
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
    if (obj == d->vtkWidget && event->type() == QEvent::MouseButtonPress) {
        emit clicked();
    }
    // Prevent the internal QVTKOpenGLNativeWidget from accepting all key
    // events. Only let VTK-specific interaction keys through; everything else
    // (modal shortcuts, menu shortcuts like Alt+C, etc.) goes to Qt.
    if (obj == d->vtkWidget && event->type() == QEvent::ShortcutOverride) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        int qkey = keyEvent->key();
        if (qkey != Qt::Key_unknown && qkey != Qt::Key_Control &&
            qkey != Qt::Key_Shift && qkey != Qt::Key_Alt &&
            qkey != Qt::Key_Meta) {
            event->ignore();
            return true;
        }
    }
    if (obj == d->vtkWidget) {
        const auto eventType = event->type();
        // Camera Orientation Widget: inject VTK events so the gizmo is not
        // overridden by trackball rotation in the 3D sub-viewport.
        if (isCameraOrientationWidgetShown() &&
            (eventType == QEvent::MouseButtonPress ||
             eventType == QEvent::MouseButtonRelease ||
             eventType == QEvent::MouseMove)) {
            auto* me = static_cast<QMouseEvent*>(event);
            if (hitTestViewIndex(me->pos()) == PERSPECTIVE_VIEW) {
                const bool overWidget = isInCameraOrientationWidget(me->pos());
                if (eventType == QEvent::MouseButtonPress && overWidget)
                    m_cameraWidgetCapturing = true;
                if (eventType == QEvent::MouseButtonRelease)
                    m_cameraWidgetCapturing = false;
                if (overWidget || m_cameraWidgetCapturing) {
                    return forwardMouseToInteractor(me, eventType);
                }
            }
        }
        if (eventType == QEvent::MouseButtonPress ||
            eventType == QEvent::MouseButtonRelease ||
            eventType == QEvent::MouseMove) {
            auto* me = static_cast<QMouseEvent*>(event);
            const int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx == PERSPECTIVE_VIEW && orientationMarkerVisible() &&
                !isCameraOrientationWidgetShown()) {
                activate3DInteractor();
                return false;
            }
        }
        if (eventType == QEvent::MouseMove ||
            eventType == QEvent::MouseButtonPress ||
            eventType == QEvent::MouseButtonRelease) {
            QPoint pos;
            if (eventType == QEvent::MouseMove) {
                pos = static_cast<QMouseEvent*>(event)->pos();
            } else {
                pos = static_cast<QMouseEvent*>(event)->pos();
            }
            if (hitTestViewIndex(pos) == PERSPECTIVE_VIEW)
                activate3DInteractor();
        }
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
                activate3DInteractor();
                return false;
            }
        } else if (event->type() == QEvent::ContextMenu) {
            return true;
        } else if (event->type() == QEvent::MouseButtonPress) {
            auto* me = static_cast<QMouseEvent*>(event);
            int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx == PERSPECTIVE_VIEW) activate3DInteractor();
            if (viewIdx >= 0 && viewIdx < 3) m_lastActiveQuadrant = viewIdx;
            m_interactionOriginView = viewIdx;
            m_rotating3D = false;
            if (me->button() == Qt::MiddleButton && viewIdx >= 0 &&
                viewIdx < 3) {
                m_rolling2D = true;
                m_rollViewIdx = viewIdx;
                m_rollLastPos = me->pos();
                return true;
            }
            if (me->button() == Qt::LeftButton && viewIdx >= 0 && viewIdx < 3) {
                // ParaView TwoDInteractorStyle: left button = pan.
                m_panPending = false;
                m_panning2D = true;
                m_panPressPos = me->pos();
                m_panLastPos = me->pos();
                return true;
            }
            if (me->button() == Qt::LeftButton && viewIdx == PERSPECTIVE_VIEW &&
                (m_selectionMode == SEL_RUBBER_POINTS ||
                 m_selectionMode == SEL_RUBBER_CELLS)) {
                m_rubberBandActive = true;
                m_rubberBandStart = me->pos();
                m_rubberBandEnd = me->pos();
                if (m_rubberBandWidget) {
                    m_rubberBandWidget->setGeometry(
                            QRect(m_rubberBandStart, QSize()));
                    m_rubberBandWidget->show();
                }
                return true;
            }
            if (me->button() == Qt::LeftButton && viewIdx == PERSPECTIVE_VIEW &&
                m_selectionMode != SEL_NONE) {
                double dpr = d->vtkWidget->devicePixelRatioF();
                int px = static_cast<int>(me->pos().x() * dpr);
                int py = static_cast<int>(
                        (d->vtkWidget->height() - 1 - me->pos().y()) * dpr);

                auto* ren = d->renderers[PERSPECTIVE_VIEW].GetPointer();
                if (ren && d->fullModelActor && !m_externalHighlighterLinked) {
                    if (d->selectionHighlightActor) {
                        d->renderers[PERSPECTIVE_VIEW]->RemoveActor(
                                d->selectionHighlightActor);
                        d->selectionHighlightActor = nullptr;
                    }

                    if (m_selectionMode == SEL_CELLS) {
                        vtkNew<vtkCellPicker> picker;
                        picker->SetTolerance(0.005);
                        picker->AddPickList(d->fullModelActor);
                        picker->PickFromListOn();
                        if (picker->Pick(px, py, 0, ren)) {
                            vtkIdType cellId = picker->GetCellId();
                            if (cellId >= 0 && d->entityPolyData) {
                                auto sel = vtkSmartPointer<vtkPolyData>::New();
                                auto cells =
                                        vtkSmartPointer<vtkCellArray>::New();
                                auto pts = vtkSmartPointer<vtkPoints>::New();
                                vtkIdType npts;
                                const vtkIdType* ptIds;
                                d->entityPolyData->GetCellPoints(cellId, npts,
                                                                 ptIds);
                                for (vtkIdType i = 0; i < npts; ++i) {
                                    double p[3];
                                    d->entityPolyData->GetPoint(ptIds[i], p);
                                    pts->InsertNextPoint(p);
                                }
                                vtkIdType newIds[16];
                                for (vtkIdType i = 0; i < npts && i < 16; ++i)
                                    newIds[i] = i;
                                cells->InsertNextCell(npts, newIds);
                                sel->SetPoints(pts);
                                sel->SetPolys(cells);

                                vtkNew<vtkPolyDataMapper> selMapper;
                                selMapper->SetInputData(sel);
                                d->selectionHighlightActor =
                                        vtkSmartPointer<vtkActor>::New();
                                d->selectionHighlightActor->SetMapper(
                                        selMapper);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetColor(1.0, 0.4, 0.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetLineWidth(3.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetPointSize(8.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetRepresentationToSurface();
                                d->selectionHighlightActor->GetProperty()
                                        ->SetAmbient(1.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->LightingOff();
                                d->renderers[PERSPECTIVE_VIEW]->AddActor(
                                        d->selectionHighlightActor);
                            }
                        }
                    } else {
                        vtkNew<vtkPointPicker> picker;
                        picker->SetTolerance(0.02);
                        picker->AddPickList(d->fullModelActor);
                        picker->PickFromListOn();
                        if (picker->Pick(px, py, 0, ren)) {
                            vtkIdType ptId = picker->GetPointId();
                            if (ptId >= 0 && d->entityPolyData) {
                                double p[3];
                                d->entityPolyData->GetPoint(ptId, p);

                                auto sphere =
                                        vtkSmartPointer<vtkPolyData>::New();
                                auto spts = vtkSmartPointer<vtkPoints>::New();
                                auto verts =
                                        vtkSmartPointer<vtkCellArray>::New();
                                vtkIdType id = spts->InsertNextPoint(p);
                                verts->InsertNextCell(1, &id);
                                sphere->SetPoints(spts);
                                sphere->SetVerts(verts);

                                vtkNew<vtkPolyDataMapper> selMapper;
                                selMapper->SetInputData(sphere);
                                d->selectionHighlightActor =
                                        vtkSmartPointer<vtkActor>::New();
                                d->selectionHighlightActor->SetMapper(
                                        selMapper);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetColor(1.0, 0.2, 0.2);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetPointSize(12.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->SetRepresentationToPoints();
                                d->selectionHighlightActor->GetProperty()
                                        ->SetAmbient(1.0);
                                d->selectionHighlightActor->GetProperty()
                                        ->LightingOff();
                                d->renderers[PERSPECTIVE_VIEW]->AddActor(
                                        d->selectionHighlightActor);
                            }
                        }
                    }
                    d->renderWindow->Render();
                }
                return true;
            }
            if (me->button() == Qt::LeftButton && viewIdx == PERSPECTIVE_VIEW &&
                m_selectionMode == SEL_NONE && !m_cameraWidgetCapturing &&
                !isCameraOrientationWidgetShown() &&
                !isInCameraOrientationWidget(me->pos())) {
                m_rotating3D = true;
            }
            if (me->button() == Qt::RightButton && viewIdx >= 0 &&
                viewIdx < 3) {
                // ParaView vtkPVTrackballZoom on TwoDInteractorStyle.
                m_zooming2D = true;
                m_zoomViewIdx = viewIdx;
                m_zoomLastPos = me->pos();
                auto* ren = d->renderers[viewIdx].GetPointer();
                if (ren) {
                    int* renSize = ren->GetSize();
                    m_zoomScale2D =
                            (renSize && renSize[1] > 0)
                                    ? 1.5 / static_cast<double>(renSize[1])
                                    : 1.5 / 512.0;
                }
                return true;
            }
        } else if (event->type() == QEvent::MouseMove && m_rotating3D) {
            activate3DInteractor();
            auto* me = static_cast<QMouseEvent*>(event);
            if (me->buttons() & Qt::LeftButton) {
                if (hitTestViewIndex(me->pos()) != PERSPECTIVE_VIEW)
                    return true;
                return false;
            }
        } else if (event->type() == QEvent::MouseMove && m_rubberBandActive) {
            auto* me = static_cast<QMouseEvent*>(event);
            m_rubberBandEnd = me->pos();
            if (m_rubberBandWidget) {
                m_rubberBandWidget->setGeometry(
                        QRect(m_rubberBandStart, m_rubberBandEnd).normalized());
            }
            return true;
        } else if (event->type() == QEvent::MouseMove && m_rolling2D) {
            auto* me = static_cast<QMouseEvent*>(event);
            auto* ren = d->renderers[m_rollViewIdx].GetPointer();
            auto* cam = ren ? ren->GetActiveCamera() : nullptr;
            if (cam) {
                int dx = me->pos().x() - m_rollLastPos.x();
                m_rollLastPos = me->pos();
                cam->Roll(-dx * 0.4);
                ren->ResetCameraClippingRange();
                d->renderWindow->Render();
            }
            return true;
        } else if (event->type() == QEvent::MouseMove && (m_panning2D)) {
            auto* me = static_cast<QMouseEvent*>(event);
            const int panViewIdx = (m_interactionOriginView >= 0 &&
                                    m_interactionOriginView < 3)
                                           ? m_interactionOriginView
                                           : m_lastActiveQuadrant;
            if (panViewIdx >= 0 && panViewIdx < 3) {
                auto* ren = d->renderers[panViewIdx].GetPointer();
                auto* cam = ren ? ren->GetActiveCamera() : nullptr;
                if (cam) {
                    QPoint delta = me->pos() - m_panLastPos;
                    m_panLastPos = me->pos();
                    double scale = cam->GetParallelScale();
                    double vp[4];
                    ren->GetViewport(vp);
                    int widgetH = d->vtkWidget->height();
                    double rendererH = (vp[3] - vp[1]) * widgetH;
                    double factor =
                            2.0 * scale / (rendererH > 1 ? rendererH : 1);
                    double fp[3], pos[3];
                    cam->GetFocalPoint(fp);
                    cam->GetPosition(pos);
                    double dx = -delta.x() * factor;
                    double dy = delta.y() * factor;
                    double vu[3];
                    cam->GetViewUp(vu);
                    double right[3];
                    double dir[3] = {fp[0] - pos[0], fp[1] - pos[1],
                                     fp[2] - pos[2]};
                    right[0] = dir[1] * vu[2] - dir[2] * vu[1];
                    right[1] = dir[2] * vu[0] - dir[0] * vu[2];
                    right[2] = dir[0] * vu[1] - dir[1] * vu[0];
                    double rlen = std::sqrt(right[0] * right[0] +
                                            right[1] * right[1] +
                                            right[2] * right[2]);
                    if (rlen > 1e-10) {
                        right[0] /= rlen;
                        right[1] /= rlen;
                        right[2] /= rlen;
                    }
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
                // ParaView vtkPVTrackballZoom::OnMouseMove (parallel
                // projection).
                const int dy = m_zoomLastPos.y() - me->pos().y();
                m_zoomLastPos = me->pos();
                const double k = dy * m_zoomScale2D;
                double newScale = (1.0 - k) * cam->GetParallelScale();
                if (newScale < 1e-9) newScale = 1e-9;
                cam->SetParallelScale(newScale);
                ren->ResetCameraClippingRange();
                d->renderWindow->Render();
            }
            return true;
        } else if (event->type() == QEvent::MouseButtonRelease) {
            m_rotating3D = false;
            m_interactionOriginView = -1;
            if (m_rubberBandActive) {
                m_rubberBandActive = false;
                auto* me = static_cast<QMouseEvent*>(event);
                m_rubberBandEnd = me->pos();
                if (m_rubberBandWidget) m_rubberBandWidget->hide();
                performRubberBandSelection();
                return true;
            }
            if (m_panning2D) {
                m_panning2D = false;
                return true;
            }
            if (m_zooming2D) {
                m_zooming2D = false;
                m_zoomViewIdx = -1;
                return true;
            }
            if (m_rolling2D) {
                m_rolling2D = false;
                m_rollViewIdx = -1;
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonDblClick) {
            m_panning2D = false;
            m_rolling2D = false;
            m_rotating3D = false;
            m_interactionOriginView = -1;
            auto* me = static_cast<QMouseEvent*>(event);
            int viewIdx = hitTestViewIndex(me->pos());
            if (viewIdx >= 0 && viewIdx < 3) {
                auto* ren = d->renderers[viewIdx].GetPointer();
                if (!ren) return false;

                double dpr = d->vtkWidget->devicePixelRatioF();
                int physH = static_cast<int>(d->vtkWidget->height() * dpr);
                if (physH <= 0) return false;

                double dispX = me->pos().x() * dpr;
                double dispY = physH - 1.0 - me->pos().y() * dpr;

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

                static constexpr int kPerpendicularAxis[3] = {1, 0, 2};
                int keepAxis = kPerpendicularAxis[viewIdx];
                newPos[keepAxis] = d->slicePos[keepAxis];

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
                            case TOP_VIEW:
                                y += step;
                                break;
                            case SIDE_VIEW:
                                x += step;
                                break;
                            case FRONT_VIEW:
                                z += step;
                                break;
                        }
                        break;
                    case Qt::Key_Down:
                    case Qt::Key_Left:
                        switch (q) {
                            case TOP_VIEW:
                                y -= step;
                                break;
                            case SIDE_VIEW:
                                x -= step;
                                break;
                            case FRONT_VIEW:
                                z -= step;
                                break;
                        }
                        break;
                    default:
                        handled = false;
                        break;
                }
            } else {
                double step = m_stepSpin ? m_stepSpin->value() : 1.0;
                switch (ke->key()) {
                    case Qt::Key_Up:
                        y += step;
                        break;
                    case Qt::Key_Down:
                        y -= step;
                        break;
                    case Qt::Key_Left:
                        x -= step;
                        break;
                    case Qt::Key_Right:
                        x += step;
                        break;
                    case Qt::Key_PageUp:
                        z += step;
                        break;
                    case Qt::Key_PageDown:
                        z -= step;
                        break;
                    default:
                        handled = false;
                        break;
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
    for (int i = 0; i < 3; ++i) {
        d->annotations[i]->SetVisibility(visible ? 1 : 0);
    }
    render();
}

void vtkOrthoSliceViewWidget::populateFromRenderer(
        vtkRenderer* sourceRenderer) {
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
    const bool reprChanged = (reprIdx != m_lastAppliedReprIdx);
    m_lastAppliedReprIdx = reprIdx;
    bool noLight =
            m_disableLightingCheck && m_disableLightingCheck->isChecked();
    double diffuse = m_diffuseSlider ? m_diffuseSlider->value() / 100.0 : 1.0;
    int interpIdx = m_interpCombo ? m_interpCombo->currentIndex() : 1;
    double specular =
            m_specularSlider ? m_specularSlider->value() / 100.0 : 0.0;
    int specPower = m_specPowerSpin ? m_specPowerSpin->value() : 100;

    int colorIdx = m_coloringCombo ? m_coloringCombo->currentIndex() : 0;
    bool mapScalars = m_mapScalarsCheck ? m_mapScalarsCheck->isChecked() : true;
    bool interpScalars =
            m_interpScalarsCheck ? m_interpScalarsCheck->isChecked() : true;
    bool useNanColor = m_useNanColorCheck && m_useNanColorCheck->isChecked();
    bool useSpecColor = m_specColorCheck && m_specColorCheck->isChecked();
    bool renderTubes = m_renderTubesCheck && m_renderTubesCheck->isChecked();
    bool renderSpheres =
            m_renderSpheresCheck && m_renderSpheresCheck->isChecked();
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
            if (mapScalars)
                mapper->SetColorModeToMapScalars();
            else
                mapper->SetColorModeToDirectScalars();
            mapper->SetInterpolateScalarsBeforeMapping(interpScalars);

            QString colorName = m_coloringCombo ? m_coloringCombo->currentText()
                                                : QString();
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

    auto isSliceActor2D = [&](vtkActor* a) -> bool {
        for (int k = 0; k < 3; ++k) {
            if (a == d->sliceActors[k].GetPointer()) return true;
        }
        for (const auto& extra : d->extraSlices) {
            for (int k = 0; k < 3; ++k) {
                if (a == extra.actors2D[k].GetPointer()) return true;
            }
        }
        for (const auto& rep : d->sliceRepresentations) {
            for (int k = 0; k < 3; ++k) {
                if (a ==
                    rep->sliceActor2D(static_cast<
                                      vtkOrthoSliceRepresentation::PlaneIndex>(
                            k)))
                    return true;
            }
        }
        return false;
    };

    auto isSliceActor3D = [&](vtkActor* a) -> bool {
        for (int k = 0; k < 3; ++k) {
            if (a == d->sliceActors3D[k].GetPointer()) return true;
        }
        for (const auto& extra : d->extraSlices) {
            for (int k = 0; k < 3; ++k) {
                if (a == extra.actors3D[k].GetPointer()) return true;
            }
        }
        for (const auto& rep : d->sliceRepresentations) {
            for (int k = 0; k < 3; ++k) {
                if (a ==
                    rep->sliceActor3D(static_cast<
                                      vtkOrthoSliceRepresentation::PlaneIndex>(
                            k)))
                    return true;
            }
        }
        return false;
    };

    auto isSurfaceModel3D = [&](vtkActor* a) -> bool {
        if (a == d->fullModelActor.GetPointer()) return true;
        for (const auto& rep : d->sliceRepresentations) {
            if (a == rep->surfaceActor3D()) return true;
        }
        return false;
    };

    auto applySlice2DStyle = [&](vtkActor* a, vtkPolyDataMapper* mapper) {
        if (!a) return;
        applyColorToMapper(mapper, a);
        applyNanColor(mapper);
        auto* prop = a->GetProperty();
        prop->SetOpacity(opacity);
        prop->SetLineWidth(lineW > 0.0 ? lineW : 3.0);
        prop->SetPointSize(ptSize > 0 ? ptSize : 3);
        bool showAsContour = false;
        if (mapper) {
            if (mapper->GetInputConnection(0, 0) != nullptr) {
                showAsContour = true;
            } else if (auto* pd =
                               vtkPolyData::SafeDownCast(mapper->GetInput())) {
                showAsContour = pd->GetNumberOfLines() > 0 ||
                                pd->GetNumberOfPolys() > 0;
            }
        }
        if (showAsContour ||
            (mapper && mapper->GetInput() &&
             vtkPolyData::SafeDownCast(mapper->GetInput()) &&
             vtkPolyData::SafeDownCast(mapper->GetInput())->GetNumberOfPolys() >
                     0)) {
            prop->SetRepresentationToSurface();
            prop->EdgeVisibilityOff();
        } else {
            prop->SetRepresentationToPoints();
        }
        if (noLight) {
            prop->LightingOff();
            prop->SetAmbient(1.0);
            prop->SetDiffuse(0.0);
            prop->SetSpecular(0.0);
        } else {
            applyLighting(prop);
        }
    };

    auto applySlice3DStyle = [&](vtkActor* a, vtkPolyDataMapper* mapper) {
        if (!a) return;
        applyColorToMapper(mapper, a);
        auto* prop = a->GetProperty();
        prop->SetOpacity(opacity);
        prop->SetLineWidth(lineW > 0.0 ? lineW : 1.0);
        bool hasPolys = mapper && mapper->GetInput() &&
                        vtkPolyData::SafeDownCast(mapper->GetInput()) &&
                        vtkPolyData::SafeDownCast(mapper->GetInput())
                                        ->GetNumberOfPolys() > 0;
        if (hasPolys) {
            prop->SetRepresentationToSurface();
            prop->EdgeVisibilityOff();
        } else {
            prop->SetRepresentationToWireframe();
            prop->SetLineWidth(lineW > 0.0 ? lineW : 2.0);
        }
        if (noLight) {
            prop->LightingOff();
            prop->SetAmbient(1.0);
            prop->SetDiffuse(0.0);
            prop->SetSpecular(0.0);
        } else {
            applyLighting(prop);
        }
    };

    auto applySurfaceRepr = [&](vtkProperty* prop) {
        switch (reprIdx) {
            case 1:
                prop->SetRepresentationToSurface();
                prop->EdgeVisibilityOff();
                break;
            case 2:
                prop->SetRepresentationToWireframe();
                break;
            case 3:
                prop->SetRepresentationToPoints();
                break;
            case 4:
                prop->SetRepresentationToSurface();
                prop->EdgeVisibilityOn();
                break;
            case 5:
                prop->SetRepresentationToWireframe();
                prop->EdgeVisibilityOn();
                break;
            case 6:
                prop->SetRepresentationToWireframe();
                prop->EdgeVisibilityOff();
                break;
            default:
                prop->SetRepresentationToSurface();
                prop->EdgeVisibilityOff();
                break;
        }
        applyLighting(prop);
    };

    auto isOutlineModel3D = [&](vtkActor* a) -> bool {
        if (a == d->outlineActor.GetPointer()) return true;
        for (const auto& rep : d->sliceRepresentations) {
            if (a == rep->outlineActor3D()) return true;
        }
        return false;
    };

    const bool showSlices = (reprIdx == 0);
    const bool showSurface = (reprIdx >= 1 && reprIdx <= 5);
    const bool showOutlineRepr = (reprIdx == 6);

    for (int v = 0; v < 4; ++v) {
        auto* actors = d->renderers[v]->GetActors();
        if (!actors) continue;
        actors->InitTraversal();
        vtkActor* a = nullptr;
        while ((a = actors->GetNextActor())) {
            bool isBuiltIn = false;
            for (int k = 0; k < 3; ++k) {
                if (a == d->sliceAxes2D[k].GetPointer()) isBuiltIn = true;
                if (a == d->planeIndicators[k].GetPointer()) isBuiltIn = true;
            }
            if (a == d->sliceAxes3D.GetPointer()) isBuiltIn = true;
            if (a == d->outlineActor.GetPointer()) isBuiltIn = true;
            for (const auto& rep : d->sliceRepresentations) {
                if (a == rep->outlineActor3D()) isBuiltIn = true;
            }
            for (int k = 0; k < 4; ++k) {
                if (d->gridAxes[k] &&
                    static_cast<vtkObjectBase*>(a) ==
                            static_cast<vtkObjectBase*>(
                                    d->gridAxes[k].GetPointer()))
                    isBuiltIn = true;
            }
            if (isBuiltIn) continue;
            if (d->selectionHighlightActor &&
                a == d->selectionHighlightActor.GetPointer()) {
                continue;
            }

            if (v != PERSPECTIVE_VIEW) {
                if (!isSliceActor2D(a)) continue;
                applySlice2DStyle(
                        a, vtkPolyDataMapper::SafeDownCast(a->GetMapper()));
                continue;
            }

            if (isSliceActor3D(a)) {
                a->SetVisibility(reprIdx == 0 ? 1 : 0);
                if (reprIdx == 0)
                    applySlice3DStyle(
                            a, vtkPolyDataMapper::SafeDownCast(a->GetMapper()));
                continue;
            }

            if (isSurfaceModel3D(a)) {
                a->SetVisibility(showSurface ? 1 : 0);
                if (!showSurface) continue;

                auto* pdMapper =
                        vtkPolyDataMapper::SafeDownCast(a->GetMapper());
                applyColorToMapper(pdMapper, a);
                applyNanColor(pdMapper);
                auto* prop = a->GetProperty();
                prop->SetOpacity(opacity);
                prop->SetPointSize(ptSize);
                prop->SetLineWidth(lineW);
                applySurfaceRepr(prop);
                continue;
            }

            if (isOutlineModel3D(a)) {
                a->SetVisibility(showOutlineRepr ? 1 : 0);
                if (!showOutlineRepr) continue;
                auto* prop = a->GetProperty();
                prop->SetOpacity(opacity);
                prop->SetLineWidth(lineW > 0.0 ? lineW : 1.0);
                prop->SetRepresentationToWireframe();
                if (noLight) {
                    prop->LightingOff();
                    prop->SetAmbient(1.0);
                    prop->SetDiffuse(0.0);
                } else {
                    applyLighting(prop);
                }
                continue;
            }

            if (reprIdx == 0) {
                a->SetVisibility(0);
            } else if (showOutlineRepr) {
                a->SetVisibility(0);
            } else {
                a->SetVisibility(1);
                auto* pdMapper =
                        vtkPolyDataMapper::SafeDownCast(a->GetMapper());
                applyColorToMapper(pdMapper, a);
                applyNanColor(pdMapper);
                auto* prop = a->GetProperty();
                prop->SetOpacity(opacity);
                prop->SetPointSize(ptSize);
                prop->SetLineWidth(lineW);
                applySurfaceRepr(prop);
            }
        }
    }

    auto styleOutlineActor = [&](vtkActor* outline) {
        if (!outline) return;
        auto* prop = outline->GetProperty();
        prop->SetOpacity(opacity);
        prop->SetLineWidth(lineW > 0.0 ? lineW : 1.0);
        prop->SetColor(1.0, 1.0, 1.0);
        prop->SetRepresentationToWireframe();
        prop->LightingOff();
        prop->SetAmbient(1.0);
        prop->SetDiffuse(0.0);
        outline->SetUseBounds(false);
    };
    if (showOutlineRepr) {
        styleOutlineActor(d->outlineActor);
        for (const auto& rep : d->sliceRepresentations)
            styleOutlineActor(rep->outlineActor3D());
    }

    if (d->outlineActor)
        d->outlineActor->SetVisibility(showOutlineRepr ? 1
                                                       : (showOutline ? 1 : 0));
    for (const auto& rep : d->sliceRepresentations) {
        if (auto* outline = rep->outlineActor3D())
            outline->SetVisibility(showOutlineRepr ? 1 : 0);
    }
    render();
}

void vtkOrthoSliceViewWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    updateOrientationWidgetViewport();
    resetCameras();
}

void vtkOrthoSliceViewWidget::setEntityListProvider(
        EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshSourceCombo();
}

void vtkOrthoSliceViewWidget::refreshSourceCombo() {
    if (!m_entityListProvider || !m_sourceCombo) return;

    int curIdx = m_sourceCombo->currentIndex();
    quintptr curPtr =
            (curIdx >= 0) ? m_sourceCombo->itemData(curIdx).value<quintptr>()
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

void vtkOrthoSliceViewWidget::performRubberBandSelection() {
    if (!d->fullModelActor || !d->renderWindow) return;
    auto* ren = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    if (!ren) return;

    double dpr = d->vtkWidget->devicePixelRatioF();
    int x0 = static_cast<int>(
            std::min(m_rubberBandStart.x(), m_rubberBandEnd.x()) * dpr);
    int y0 = static_cast<int>(
            (d->vtkWidget->height() - 1 -
             std::max(m_rubberBandStart.y(), m_rubberBandEnd.y())) *
            dpr);
    int x1 = static_cast<int>(
            std::max(m_rubberBandStart.x(), m_rubberBandEnd.x()) * dpr);
    int y1 = static_cast<int>(
            (d->vtkWidget->height() - 1 -
             std::min(m_rubberBandStart.y(), m_rubberBandEnd.y())) *
            dpr);

    if (std::abs(x1 - x0) < 4 && std::abs(y1 - y0) < 4) return;

    vtkNew<vtkAreaPicker> areaPicker;
    areaPicker->AreaPick(x0, y0, x1, y1, ren);

    auto* frustum = areaPicker->GetFrustum();
    if (!frustum || !d->entityPolyData) return;

    vtkNew<vtkExtractSelectedFrustum> extractor;
    extractor->SetInputData(d->entityPolyData);
    extractor->SetFrustum(frustum);
    extractor->Update();

    auto* rawOutput = extractor->GetOutput();
    auto* output = vtkDataSet::SafeDownCast(rawOutput);
    if (!output) return;

    auto* origIds =
            output->GetPointData()
                    ? output->GetPointData()->GetArray("vtkOriginalPointIds")
                    : nullptr;
    auto* origCellIds =
            output->GetCellData()
                    ? output->GetCellData()->GetArray("vtkOriginalCellIds")
                    : nullptr;

    if (d->selectionHighlightActor) {
        d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->selectionHighlightActor);
        d->selectionHighlightActor = nullptr;
    }

    m_selectedIndices.clear();

    bool pickPoints = (m_selectionMode == SEL_RUBBER_POINTS);
    vtkDataArray* ids = pickPoints ? origIds : origCellIds;
    if (!ids || ids->GetNumberOfTuples() == 0) {
        d->renderWindow->Render();
        return;
    }

    for (vtkIdType i = 0; i < ids->GetNumberOfTuples(); ++i) {
        m_selectedIndices.insert(static_cast<unsigned>(ids->GetTuple1(i)));
    }

    if (pickPoints && origIds) {
        vtkNew<vtkPoints> pts;
        vtkNew<vtkPolyData> hlPoly;
        for (vtkIdType i = 0; i < origIds->GetNumberOfTuples(); ++i) {
            vtkIdType ptId = static_cast<vtkIdType>(origIds->GetTuple1(i));
            if (ptId >= 0 && ptId < d->entityPolyData->GetNumberOfPoints()) {
                double p[3];
                d->entityPolyData->GetPoint(ptId, p);
                pts->InsertNextPoint(p);
            }
        }
        hlPoly->SetPoints(pts);
        vtkNew<vtkPolyDataMapper> hlMapper;
        hlMapper->SetInputData(hlPoly);
        d->selectionHighlightActor = vtkSmartPointer<vtkActor>::New();
        d->selectionHighlightActor->SetMapper(hlMapper);
        d->selectionHighlightActor->GetProperty()->SetColor(1.0, 0.2, 0.2);
        d->selectionHighlightActor->GetProperty()->SetPointSize(8.0);
        d->selectionHighlightActor->GetProperty()->SetRepresentationToPoints();
        d->selectionHighlightActor->GetProperty()->SetAmbient(1.0);
        d->selectionHighlightActor->GetProperty()->LightingOff();
        d->renderers[PERSPECTIVE_VIEW]->AddActor(d->selectionHighlightActor);
    } else if (!pickPoints && origCellIds) {
        vtkNew<vtkExtractCells> cellExtract;
        cellExtract->SetInputData(d->entityPolyData);
        vtkNew<vtkIdList> cellIds;
        for (vtkIdType i = 0; i < origCellIds->GetNumberOfTuples(); ++i) {
            cellIds->InsertNextId(
                    static_cast<vtkIdType>(origCellIds->GetTuple1(i)));
        }
        cellExtract->SetCellList(cellIds);
        cellExtract->Update();
        vtkNew<vtkPolyDataMapper> hlMapper;
        vtkNew<vtkGeometryFilter> geomFilter;
        geomFilter->SetInputConnection(cellExtract->GetOutputPort());
        geomFilter->Update();
        hlMapper->SetInputData(geomFilter->GetOutput());
        d->selectionHighlightActor = vtkSmartPointer<vtkActor>::New();
        d->selectionHighlightActor->SetMapper(hlMapper);
        d->selectionHighlightActor->GetProperty()->SetColor(0.2, 0.8, 0.2);
        d->selectionHighlightActor->GetProperty()->SetAmbient(1.0);
        d->selectionHighlightActor->GetProperty()->LightingOff();
        d->renderers[PERSPECTIVE_VIEW]->AddActor(d->selectionHighlightActor);
    }

    d->renderWindow->Render();

    if (!m_selectedIndices.isEmpty()) {
        if (m_statusLabel) {
            m_statusLabel->setText(
                    tr("Selected %1 %2")
                            .arg(m_selectedIndices.size())
                            .arg(pickPoints ? tr("points") : tr("cells")));
        }
    }
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

void vtkOrthoSliceViewWidget::clearEntityDisplay() {
    d->extraSlices.clear();
    d->sliceRepresentations.clear();

    if (d->selectionHighlightActor) {
        d->renderers[PERSPECTIVE_VIEW]->RemoveActor(d->selectionHighlightActor);
        d->selectionHighlightActor = nullptr;
    }

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
                if (a == d->planeIndicators[k].GetPointer()) isBuiltIn = true;
            }
            if (a == d->sliceAxes3D.GetPointer()) isBuiltIn = true;
            if (!isBuiltIn) toRemove.append(a);
        }
        for (auto* rem : toRemove) d->renderers[i]->RemoveActor(rem);
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
        d->outlineSource = nullptr;
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
        d->sliceClippers[i] = nullptr;
        d->sliceCutters[i] = nullptr;
        d->slicePlanes[i] = nullptr;
    }
}

void vtkOrthoSliceViewWidget::loadEntityIntoView(ccHObject* entity) {
    clearEntityDisplay();

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
    if (!mesh && genericMesh) mesh = ccHObjectCaster::ToMesh(genericMesh);

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

    ccGenericMesh* meshForConvert = genericMesh ? genericMesh : mesh;
    if (meshForConvert && pcCloud) {
        d->entityPolyData =
                Converters::Cc2Vtk::MeshToPolyData(pcCloud, meshForConvert);
    }
    if (d->entityPolyData && d->entityPolyData->GetNumberOfPoints() == 0 &&
        pcCloud) {
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
        d->hasMeshCells = (d->entityPolyData->GetNumberOfCells() > 0 &&
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

            createOutlineActor();
        }

        const double normals[3][3] = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};
        const double origins[3][3] = {{cx, cy, cz}, {cx, cy, cz}, {cx, cy, cz}};
        for (int i = 0; i < 3; ++i) {
            d->slicePlanes[i] = vtkSmartPointer<vtkPlane>::New();
            d->slicePlanes[i]->SetNormal(normals[i]);
            d->slicePlanes[i]->SetOrigin(origins[i]);

            d->sliceClippers[i] =
                    createSliceClipper(d->entityPolyData, d->slicePlanes[i]);
            d->sliceCutters[i] = vtkSmartPointer<vtkCutter>::New();
            d->sliceCutters[i]->SetCutFunction(d->slicePlanes[i]);
            d->sliceCutters[i]->SetInputData(d->entityPolyData);
            d->sliceCutters[i]->Update();

            vtkPolyData* slice2D =
                    getSliceDisplayPolyData(d->sliceClippers[i].GetPointer(),
                                            d->sliceCutters[i].GetPointer());
            const bool hasSliceSurface = hasValidSliceGeometry(slice2D);

            vtkNew<vtkPolyDataMapper> mapper;
            assignSliceMapperInput(mapper, slice2D);
            mapper->ScalarVisibilityOn();

            d->sliceActors[i] = vtkSmartPointer<vtkActor>::New();
            d->sliceActors[i]->SetMapper(mapper);
            d->sliceActors[i]->SetVisibility(hasSliceSurface ? 1 : 0);
            if (hasSliceSurface) {
                d->sliceActors[i]->GetProperty()->SetRepresentationToSurface();
                d->sliceActors[i]->GetProperty()->EdgeVisibilityOff();
            } else {
                d->sliceActors[i]->GetProperty()->SetColor(0.9, 0.9, 0.9);
                d->sliceActors[i]->GetProperty()->SetPointSize(3);
                d->sliceActors[i]->GetProperty()->SetRepresentationToPoints();
            }
            d->sliceActors[i]->GetProperty()->SetAmbient(1.0);
            d->sliceActors[i]->GetProperty()->SetDiffuse(0.0);
            d->sliceActors[i]->GetProperty()->LightingOff();
            d->renderers[i]->AddActor(d->sliceActors[i]);

            vtkNew<vtkPolyDataMapper> mapper3D;
            assignSliceMapperInput(mapper3D, slice2D);
            mapper3D->ScalarVisibilityOn();

            double sliceContourColors[3][3] = {
                    {1.0, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};
            d->sliceActors3D[i] = vtkSmartPointer<vtkActor>::New();
            d->sliceActors3D[i]->SetMapper(mapper3D);
            d->sliceActors3D[i]->SetVisibility(hasSliceSurface ? 1 : 0);
            if (hasSliceSurface) {
                d->sliceActors3D[i]
                        ->GetProperty()
                        ->SetRepresentationToSurface();
                d->sliceActors3D[i]->GetProperty()->EdgeVisibilityOff();
                d->sliceActors3D[i]->GetProperty()->SetOpacity(0.9);
            } else {
                d->sliceActors3D[i]->GetProperty()->SetColor(
                        sliceContourColors[i]);
                d->sliceActors3D[i]->GetProperty()->SetPointSize(2);
                d->sliceActors3D[i]->GetProperty()->SetRepresentationToPoints();
            }
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
            d->planeSources[i] = nullptr;
        }
    }
    if (box.isValid()) {
        double bounds[6] = {box.minCorner().x, box.maxCorner().x,
                            box.minCorner().y, box.maxCorner().y,
                            box.minCorner().z, box.maxCorner().z};
        setGeometryBounds(bounds);

        d->planeInitCenter[0] = cx;
        d->planeInitCenter[1] = cy;
        d->planeInitCenter[2] = cz;

        createPlaneIndicators(bounds);

        setSlicePosition(cx, cy, cz);

        double sx = bounds[1] - bounds[0];
        double sy = bounds[3] - bounds[2];
        double sz = bounds[5] - bounds[4];
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
            if (m_sliceSpin[i])
                m_sliceSpin[i]->setSingleStep(m_sliceIncrements[i]);
        }
    }

    if (m_coloringCombo && d->entityPolyData) {
        m_coloringCombo->blockSignals(true);
        int oldIdx = m_coloringCombo->currentIndex();
        m_coloringCombo->clear();
        m_coloringCombo->addItem(tr("Solid Color"));
        auto* pd = d->entityPolyData->GetPointData();
        if (pd) {
            if (pd->GetScalars()) m_coloringCombo->addItem(tr("Points"));
            if (pd->GetNormals()) m_coloringCombo->addItem(tr("Normals"));
            if (pd->GetTCoords()) m_coloringCombo->addItem(tr("TCoords"));
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

    clearEntityDisplay();

    auto entityToPolyData =
            [](ccHObject* entity) -> vtkSmartPointer<vtkPolyData> {
        if (!entity) return nullptr;
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
        if (!mesh && genericMesh) mesh = ccHObjectCaster::ToMesh(entity);
        if (!cloud) return nullptr;

        auto* pcCloud = ccHObjectCaster::ToPointCloud(entity);
        if (!pcCloud && genericMesh)
            pcCloud = ccHObjectCaster::ToPointCloud(
                    genericMesh->getAssociatedCloud());
        if (!pcCloud && mesh)
            pcCloud = ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
        if (!pcCloud) pcCloud = ccHObjectCaster::ToPointCloud(cloud);

        ccGenericMesh* meshForConvert = genericMesh ? genericMesh : mesh;
        vtkSmartPointer<vtkPolyData> polyData;
        if (meshForConvert && pcCloud)
            polyData =
                    Converters::Cc2Vtk::MeshToPolyData(pcCloud, meshForConvert);
        if (polyData && polyData->GetNumberOfPoints() == 0) polyData = nullptr;
        if (!polyData && pcCloud)
            polyData = Converters::Cc2Vtk::PointCloudToPolyData(pcCloud);
        return polyData;
    };

    vtkRenderer* orthoRenderers[3] = {d->renderers[TOP_VIEW].GetPointer(),
                                      d->renderers[SIDE_VIEW].GetPointer(),
                                      d->renderers[FRONT_VIEW].GetPointer()};
    auto* perspectiveRenderer = d->renderers[PERSPECTIVE_VIEW].GetPointer();

    d->entityPolyData = nullptr;
    for (ccHObject* entity : entities) {
        auto polyData = entityToPolyData(entity);
        if (!polyData) continue;

        auto rep = std::make_shared<vtkOrthoSliceRepresentation>();
        rep->setInputPolyData(polyData);
        rep->setSliceOrigin(d->slicePos[0], d->slicePos[1], d->slicePos[2]);
        rep->addToRenderers(orthoRenderers, perspectiveRenderer);
        d->sliceRepresentations.append(rep);

        if (!d->entityPolyData) d->entityPolyData = polyData;
    }

    if (m_statusLabel)
        m_statusLabel->setText(tr("All (%1 objects)").arg(entities.size()));

    ccBBox globalBox = mergeEntitiesBounds(entities);
    if (globalBox.isValid()) {
        double bounds[6] = {globalBox.minCorner().x, globalBox.maxCorner().x,
                            globalBox.minCorner().y, globalBox.maxCorner().y,
                            globalBox.minCorner().z, globalBox.maxCorner().z};
        setGeometryBounds(bounds);

        double cx = (bounds[0] + bounds[1]) * 0.5;
        double cy = (bounds[2] + bounds[3]) * 0.5;
        double cz = (bounds[4] + bounds[5]) * 0.5;
        double sx = bounds[1] - bounds[0];
        double sy = bounds[3] - bounds[2];
        double sz = bounds[5] - bounds[4];

        for (int i = 0; i < 3; ++i) {
            if (d->planeIndicators[i]) {
                d->renderers[PERSPECTIVE_VIEW]->RemoveActor(
                        d->planeIndicators[i]);
                d->planeIndicators[i] = nullptr;
                d->planeSources[i] = nullptr;
            }
        }

        createPlaneIndicators(bounds);

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
            if (m_sliceSpin[i])
                m_sliceSpin[i]->setSingleStep(m_sliceIncrements[i]);
        }

        setSlicePosition(cx, cy, cz);
    }

    if (m_coloringCombo && d->entityPolyData) {
        m_coloringCombo->blockSignals(true);
        int oldIdx = m_coloringCombo->currentIndex();
        m_coloringCombo->clear();
        m_coloringCombo->addItem(tr("Solid Color"));
        auto* pd = d->entityPolyData->GetPointData();
        if (pd) {
            if (pd->GetScalars()) m_coloringCombo->addItem(tr("Points"));
            if (pd->GetNormals()) m_coloringCombo->addItem(tr("Normals"));
            if (pd->GetTCoords()) m_coloringCombo->addItem(tr("TCoords"));
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

    applyDisplayProperties();
    resetCameras();

    QTimer::singleShot(100, this, [this]() {
        resetCameras();
        render();
    });
}

void vtkOrthoSliceViewWidget::onSourceComboAboutToShow() {
    refreshSourceCombo();
}

void vtkOrthoSliceViewWidget::set3DProjection(bool perspective) {
    vtkRenderer* ren = d->renderers[PERSPECTIVE_VIEW];
    if (!ren) return;
    ren->GetActiveCamera()->SetParallelProjection(!perspective);
    ren->ResetCamera();
    render();
}

void vtkOrthoSliceViewWidget::zoomToFit() {
    for (int i = 0; i < 4; ++i) {
        vtkRenderer* ren = d->renderers[i];
        if (ren) {
            ren->ResetCamera();
            ren->ResetCameraClippingRange();
        }
    }
    render();
}

void vtkOrthoSliceViewWidget::setViewPreset(int presetIndex) {
    static const double kPresets[][6] = {
            {1, 0, 0, 0, 0, 1},  {-1, 0, 0, 0, 0, 1}, {0, 1, 0, 0, 0, 1},
            {0, -1, 0, 0, 0, 1}, {0, 0, 1, 0, 1, 0},  {0, 0, -1, 0, 1, 0},
    };
    if (presetIndex < 0 || presetIndex > 5) return;
    vtkRenderer* ren = d->renderers[PERSPECTIVE_VIEW];
    if (!ren) return;
    auto* cam = ren->GetActiveCamera();
    double fp[3];
    cam->GetFocalPoint(fp);
    double dist = cam->GetDistance();
    const double* p = kPresets[presetIndex];
    cam->SetPosition(fp[0] + p[0] * dist, fp[1] + p[1] * dist,
                     fp[2] + p[2] * dist);
    cam->SetViewUp(p[3], p[4], p[5]);
    ren->ResetCameraClippingRange();
    render();
}

void vtkOrthoSliceViewWidget::setSelectionMode(int mode) {
    switch (mode) {
        case 0:
            m_selectionMode = SEL_CELLS;
            break;
        case 1:
            m_selectionMode = SEL_POINTS;
            break;
        case 2:
            m_selectionMode = SEL_RUBBER_CELLS;
            break;
        case 3:
            m_selectionMode = SEL_RUBBER_POINTS;
            break;
        default:
            m_selectionMode = SEL_NONE;
            break;
    }
}

void vtkOrthoSliceViewWidget::mapWidgetToRendererDisplay(
        const QPoint& pos, vtkRenderer* ren, double outXY[2]) const {
    outXY[0] = 0.0;
    outXY[1] = 0.0;
    if (!ren || !d->vtkWidget || !d->renderWindow) return;

    const int widgetW = d->vtkWidget->width();
    const int widgetH = d->vtkWidget->height();
    if (widgetW <= 0 || widgetH <= 0) return;

    int* winSize = d->renderWindow->GetSize();
    const double dpr = d->vtkWidget->devicePixelRatioF();
    const int winW = (winSize && winSize[0] > 0)
                             ? winSize[0]
                             : static_cast<int>(widgetW * dpr);
    const int winH = (winSize && winSize[1] > 0)
                             ? winSize[1]
                             : static_cast<int>(widgetH * dpr);

    double vp[4];
    ren->GetViewport(vp);
    const double nx = static_cast<double>(pos.x()) / widgetW;
    const double ny = 1.0 - static_cast<double>(pos.y()) / widgetH;
    outXY[0] = (vp[0] + nx * (vp[2] - vp[0])) * winW;
    outXY[1] = (vp[1] + ny * (vp[3] - vp[1])) * winH;
}

void vtkOrthoSliceViewWidget::setupDecoratorBarContextMenu() {
    if (!m_decoratorBar) return;

    m_decoratorBar->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_decoratorBar, &QWidget::customContextMenuRequested, this,
            [this](const QPoint& pos) {
                Q_UNUSED(pos);
                auto toggleBar = [](QWidget* bar) {
                    if (!bar) return;
                    const bool show = !bar->isVisible();
                    bar->setVisible(show);
                    bar->setMaximumHeight(show ? QWIDGETSIZE_MAX : 0);
                };

                QMenu menu(this);
                menu.addAction(tr("Reset Cameras"), this,
                               &vtkOrthoSliceViewWidget::resetCameras);
                menu.addAction(tr("Center Slices"), this, [this]() {
                    const double cx =
                            (d->geomBounds[0] + d->geomBounds[1]) * 0.5;
                    const double cy =
                            (d->geomBounds[2] + d->geomBounds[3]) * 0.5;
                    const double cz =
                            (d->geomBounds[4] + d->geomBounds[5]) * 0.5;
                    setSlicePosition(cx, cy, cz);
                    emit slicePositionChanged(cx, cy, cz);
                    resetCameras();
                });
                menu.addSeparator();
                menu.addAction(tr("Toggle Display Properties"), this,
                               [this, toggleBar]() { toggleBar(m_dispBar); });
                menu.addAction(tr("Toggle Coloring Bar"), this,
                               [this, toggleBar]() { toggleBar(m_colorBar); });
                menu.addAction(tr("Toggle Lighting Bar"), this,
                               [this, toggleBar]() { toggleBar(m_lightBar); });
                menu.exec(QCursor::pos());
            });
}

void vtkOrthoSliceViewWidget::setOrientationMarkerVisible(bool visible) {
    ensureOrientationWidgetsInitialized();
    if (!d->orientWidget) return;
    d->orientWidget->SetVisibility(visible ? 1 : 0);
    d->orientWidget->SetEnabled(visible ? 1 : 0);
    updateOrientationWidgetViewport();
    render();
}

bool vtkOrthoSliceViewWidget::orientationMarkerVisible() const {
    return d->orientWidget && d->orientWidget->GetVisibility() != 0;
}

void vtkOrthoSliceViewWidget::toggleCameraOrientationWidget(bool visible) {
    ensureOrientationWidgetsInitialized();
    auto* renderer = d->renderers[PERSPECTIVE_VIEW].GetPointer();
    auto* interactor =
            d->renderWindow ? d->renderWindow->GetInteractor() : nullptr;
    if (!renderer || !interactor) return;

    if (!d->cameraOrientWidget) {
        d->cameraOrientWidget =
                vtkSmartPointer<vtkCameraOrientationWidget>::New();
        renderer->SetLayer(0);
        d->cameraOrientWidget->SetParentRenderer(renderer);
        d->cameraOrientWidget->SetInteractor(interactor);
        d->cameraOrientWidget->SetAnimate(false);
        d->cameraOrientWidget->CreateDefaultRepresentation();
        if (auto* rep = vtkCameraOrientationRepresentation::SafeDownCast(
                    d->cameraOrientWidget->GetRepresentation())) {
            configureCameraOrientationRepresentation(rep);
            d->cameraOrientWidget->SquareResize();
        }
    }

    // ParaView vtkPVRenderView uses CameraOrientationWidget->On()/Off().
    if (visible) {
        d->cameraOrientWidget->On();
        interactor->Enable();
    } else {
        d->cameraOrientWidget->Off();
    }
    activate3DInteractor();
    updateOrientationWidgetViewport();
    render();
}

bool vtkOrthoSliceViewWidget::isCameraOrientationWidgetShown() const {
    if (!d->cameraOrientWidget) return false;
    if (auto* rep = d->cameraOrientWidget->GetRepresentation())
        return rep->GetVisibility() != 0;
    return false;
}
