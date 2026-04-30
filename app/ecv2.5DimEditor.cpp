// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecv2.5DimEditor.h"

// LOCAL
#include "MainWindow.h"
#include "ecvBoundingBoxEditorDlg.h"
#include "ecvPersistentSettings.h"

// CV_DB_LIB
#include <ecvGenericGLDisplay.h>
#include <ecvGenericPointCloud.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvRedrawScope.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>
#include <ecvViewportParameters.h>

// Qt
#include <QCoreApplication>
#include <QFrame>
#include <QSettings>

// System
#include <assert.h>

cc2Point5DimEditor::cc2Point5DimEditor() : m_bbEditorDlg(0), m_rasterCloud(0) {}

cc2Point5DimEditor::~cc2Point5DimEditor() {
    if (m_rasterCloud) {
        auto& vm = ecvViewManager::instance();
        if (vm.activeWidget()) {
            if (auto* v = vm.getEffectiveView()) {
                v->removeFromOwnDB(m_rasterCloud);
            }
        }
        delete m_rasterCloud;
        m_rasterCloud = 0;
    }
}

bool cc2Point5DimEditor::showGridBoxEditor() {
    if (m_bbEditorDlg) {
        unsigned char projDim = getProjectionDimension();
        assert(projDim < 3);
        m_bbEditorDlg->set2DMode(true, projDim);
        if (m_bbEditorDlg->exec()) {
            gridIsUpToDate(false);
            return true;
        }
    }

    return false;
}

void cc2Point5DimEditor::createBoundingBoxEditor(const ccBBox& gridBBox,
                                                 QWidget* parent) {
    if (!m_bbEditorDlg) {
        m_bbEditorDlg = new ccBoundingBoxEditorDlg(parent);
        m_bbEditorDlg->setBaseBBox(gridBBox, false);
    }
}

void cc2Point5DimEditor::create2DView(QFrame* parentFrame) {
    // NOTE: The embedded 2D preview is not yet implemented for the VTK
    // backend (the widget below is a placeholder). We no longer modify
    // global display state here to avoid polluting active 3D views in
    // multi-window setups.

    if (parentFrame) {
        auto layout = new QHBoxLayout;
        layout->setContentsMargins(0, 0, 0, 0);
        QWidget* widget = new QWidget();
        layout->addWidget(widget);
        parentFrame->setLayout(layout);
    }
}

bool cc2Point5DimEditor::getGridSize(unsigned& gridWidth,
                                     unsigned& gridHeight) const {
    // vertical dimension
    const unsigned char Z = getProjectionDimension();

    // cloud bounding-box --> grid size
    ccBBox box = getCustomBBox();

    // grid step
    double gridStep = getGridStep();

    return ccRasterGrid::ComputeGridSize(Z, box, gridStep, gridWidth,
                                         gridHeight);
}

QString cc2Point5DimEditor::getGridSizeAsString() const {
    unsigned gridWidth = 0, gridHeight = 0;
    if (!getGridSize(gridWidth, gridHeight)) {
        return QObject::tr("invalid grid box");
    }

    return QString("%1 x %2").arg(gridWidth).arg(gridHeight);
}

ccBBox cc2Point5DimEditor::getCustomBBox() const {
    return (m_bbEditorDlg ? m_bbEditorDlg->getBox() : ccBBox());
}

void cc2Point5DimEditor::update2DDisplayZoom(ccBBox& box) {
    auto& vm = ecvViewManager::instance();
    QWidget* aw = vm.activeWidget();
    if (!aw || !m_grid.isValid()) return;

    ecvGenericGLDisplay* effectiveView = vm.getEffectiveView();
    if (!effectiveView) return;

    ecvViewManager::ScopedRenderOverride renderGuard(effectiveView);

    ecvViewportParameters params = effectiveView->getViewportParameters();

    double realGridWidth = m_grid.width * m_grid.gridStep;
    double realGridHeight = m_grid.height * m_grid.gridStep;

    static const int screnMargin = 20;
    int screenWidth = std::max(1, aw->width() - 2 * screnMargin);
    int screenHeight = std::max(1, aw->height() - 2 * screnMargin);

    int pointSize = 1;
    if (static_cast<int>(m_grid.width) < screenWidth &&
        static_cast<int>(m_grid.height) < screenHeight) {
        int vPointSize = static_cast<int>(
                ceil(static_cast<float>(screenWidth) / m_grid.width));
        int hPointSize = static_cast<int>(
                ceil(static_cast<float>(screenHeight) / m_grid.height));
        pointSize = std::min(vPointSize, hPointSize);

        // if the grid is too small (i.e. necessary point size > 10)
        if (pointSize > 10) {
            pointSize = 10;
            screenWidth = m_grid.width * pointSize;
            screenHeight = m_grid.height * pointSize;
        }
    }

    params.pixelSize = static_cast<float>(std::max(
            realGridWidth / screenWidth, realGridHeight / screenHeight));
    params.zoom = 1.0f;
    params.defaultPointSize = static_cast<float>(pointSize);

    CCVector3 P = box.getCenter();
    CCVector3d Pd = CCVector3d::fromArray(P.u);
    params.setPivotPoint(Pd, true);
    params.setCameraCenter(Pd, true);

    effectiveView->setViewportParameters(params);

    if (auto* ctx = effectiveView->viewContext()) {
        ctx->validProjectionMatrix = false;
        ctx->validModelviewMatrix = false;
    }
    effectiveView->deprecate3DLayer();

    { ecvRedrawScope scope; }
}

ccPointCloud* cc2Point5DimEditor::convertGridToCloud(
        const std::vector<ccRasterGrid::ExportableFields>& exportedFields,
        bool interpolateSF,
        bool interpolateColors,
        bool resampleInputCloudXY,
        bool resampleInputCloudZ,
        ccGenericPointCloud* inputCloud,
        bool fillEmptyCells,
        double emptyCellsHeight,
        bool exportToOriginalCS) const {
    // projection dimension
    const unsigned char Z = getProjectionDimension();
    assert(Z <= 2);

    // cloud bounding-box
    ccBBox box = getCustomBBox();
    assert(box.isValid());

    return m_grid.convertToCloud(
            exportedFields, interpolateSF, interpolateColors,
            resampleInputCloudXY, resampleInputCloudZ, inputCloud, Z, box,
            fillEmptyCells, emptyCellsHeight, exportToOriginalCS);
}

ccRasterGrid::EmptyCellFillOption cc2Point5DimEditor::getFillEmptyCellsStrategy(
        QComboBox* comboBox) const {
    if (!comboBox) {
        assert(false);
        return ccRasterGrid::LEAVE_EMPTY;
    }

    switch (comboBox->currentIndex()) {
        case 0:
            return ccRasterGrid::LEAVE_EMPTY;
        case 1:
            return ccRasterGrid::FILL_MINIMUM_HEIGHT;
        case 2:
            return ccRasterGrid::FILL_AVERAGE_HEIGHT;
        case 3:
            return ccRasterGrid::FILL_MAXIMUM_HEIGHT;
        case 4:
            return ccRasterGrid::FILL_CUSTOM_HEIGHT;
        case 5:
            return ccRasterGrid::INTERPOLATE_DELAUNAY;
        default:
            // shouldn't be possible for this option!
            assert(false);
    }

    return ccRasterGrid::LEAVE_EMPTY;
}
