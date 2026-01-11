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

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvScalarField.h>

// Qt
#include <QCoreApplication>
#include <QFrame>
#include <QSettings>

// System
#include <assert.h>

cc2Point5DimEditor::cc2Point5DimEditor() : m_bbEditorDlg(0), m_rasterCloud(0) {}

cc2Point5DimEditor::~cc2Point5DimEditor() {
    if (m_rasterCloud) {
        if (ecvDisplayTools::GetMainWindow())
            ecvDisplayTools::RemoveFromOwnDB(m_rasterCloud);
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
    ecvGui::ParamStruct params = ecvDisplayTools::GetDisplayParameters();
    // black (text) & white (background) display by default
    params.backgroundCol = ecvColor::white;
    params.textDefaultCol = ecvColor::black;
    params.drawBackgroundGradient = false;
    params.decimateMeshOnMove = false;
    params.displayCross = false;
    params.colorScaleUseShader = false;
    ecvDisplayTools::SetDisplayParameters(params);
    ecvDisplayTools::SetPerspectiveState(false, true);
    ecvDisplayTools::SetInteractionMode(
            ecvDisplayTools::INTERACT_PAN |
            ecvDisplayTools::INTERACT_ZOOM_CAMERA |
            ecvDisplayTools::INTERACT_CLICKABLE_ITEMS);
    ecvDisplayTools::SetPickingMode(ecvDisplayTools::NO_PICKING);
    ecvDisplayTools::DisplayOverlayEntities(true);

    // add window to the input frame (if any)
    if (parentFrame) {
        auto layout = new QHBoxLayout;

        layout->setContentsMargins(0, 0, 0, 0);
        // TODO
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
    if (!ecvDisplayTools::GetMainWindow() || !m_grid.isValid()) return;

    // we compute the pixel size (in world coordinates)
    {
        ecvViewportParameters params = ecvDisplayTools::GetViewportParameters();

        double realGridWidth = m_grid.width * m_grid.gridStep;
        double realGridHeight = m_grid.height * m_grid.gridStep;

        static const int screnMargin = 20;
        int screenWidth =
                std::max(1, ecvDisplayTools::Width() - 2 * screnMargin);
        int screenHeight =
                std::max(1, ecvDisplayTools::Height() - 2 * screnMargin);

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

        ecvDisplayTools::SetViewportParameters(params);
        ecvDisplayTools::SetPointSize(pointSize);
    }

    // we set the pivot point on the box center
    CCVector3 P = box.getCenter();
    ecvDisplayTools::SetPivotPoint(CCVector3d::fromArray(P.u));
    ecvDisplayTools::SetCameraPos(CCVector3d::fromArray(P.u));

    ecvDisplayTools::InvalidateViewport();
    ecvDisplayTools::InvalidateVisualization();
    ecvDisplayTools::Deprecate3DLayer();
    ecvDisplayTools::RedrawDisplay();
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
