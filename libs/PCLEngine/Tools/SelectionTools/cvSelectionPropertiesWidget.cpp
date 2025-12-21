// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionPropertiesWidget.h"

// LOCAL
#include "PclUtils/PCLVis.h"
#include "cvFilterConfigDialog.h"
#include "cvSelectionAlgebra.h"
#include "cvSelectionAnnotation.h"
#include "cvSelectionBookmarks.h"
#include "cvSelectionExporter.h"
#include "cvSelectionFilter.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionTooltipHelper.h"
#include "cvViewSelectionManager.h"

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// CV_CORE_LIB
#include <CVLog.h>

// ECV_IO_LIB
#include <FileIOFilter.h>

// Qt
#include <QApplication>
#include <QFileDialog>
#include <QInputDialog>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QTimer>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataSetMapper.h>
#include <vtkIdTypeArray.h>
#include <vtkMapper.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

// Qt
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QVBoxLayout>

//-----------------------------------------------------------------------------
cvSelectionPropertiesWidget::cvSelectionPropertiesWidget(QWidget* parent)
    : QWidget(parent),
      cvSelectionBase(),  // Initialize base class
      m_highlighter(nullptr),
      m_tooltipHelper(new cvSelectionTooltipHelper()),
      m_selectionManager(nullptr),
      m_selectionCount(0),
      m_volume(0.0) {
    // Initialize colors (matching cvSelectionHighlighter defaults)
    // IMPORTANT: These MUST match the defaults in cvSelectionHighlighter
    // constructor See cvSelectionHighlighter.cpp lines 38-78

    // Hover: Bright Cyan (0, 255, 255)
    m_hoverColor[0] = 0.0;
    m_hoverColor[1] = 1.0;
    m_hoverColor[2] = 1.0;

    // Pre-selected: Bright Yellow (255, 255, 0)
    m_preselectedColor[0] = 1.0;
    m_preselectedColor[1] = 1.0;
    m_preselectedColor[2] = 0.0;

    // Selected: Bright Lime Green (0, 255, 0)
    m_selectedColor[0] = 0.0;
    m_selectedColor[1] = 1.0;
    m_selectedColor[2] = 0.0;

    // Boundary: Bright Orange (255, 165, 0)
    m_boundaryColor[0] = 1.0;
    m_boundaryColor[1] = 0.65;
    m_boundaryColor[2] = 0.0;

    for (int i = 0; i < 6; ++i) {
        m_bounds[i] = 0.0;
    }
    for (int i = 0; i < 3; ++i) {
        m_center[i] = 0.0;
    }

    setupUi();

    CVLog::PrintDebug("[cvSelectionPropertiesWidget] Initialized");
}

//-----------------------------------------------------------------------------
cvSelectionPropertiesWidget::~cvSelectionPropertiesWidget() {
    delete m_tooltipHelper;
}

// setVisualizer is inherited from cvGenericSelectionTool

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setHighlighter(
        cvSelectionHighlighter* highlighter) {
    m_highlighter = highlighter;

    // Connect signals to highlighter to apply color/opacity changes
    if (m_highlighter) {
        connect(this, &cvSelectionPropertiesWidget::highlightColorChanged,
                [this](double r, double g, double b, int mode) {
                    if (!m_highlighter) return;

                    cvSelectionHighlighter::HighlightMode hlMode =
                            static_cast<cvSelectionHighlighter::HighlightMode>(
                                    mode);
                    m_highlighter->setHighlightColor(r, g, b, hlMode);

                    // Refresh display
                    PclUtils::PCLVis* pclVis = getPCLVis();
                    if (pclVis) {
                        pclVis->UpdateScreen();
                    }

                    CVLog::Print(QString("[cvSelectionPropertiesWidget] Color "
                                         "updated for mode %1")
                                         .arg(mode));
                });

        connect(this, &cvSelectionPropertiesWidget::highlightOpacityChanged,
                [this](double opacity, int mode) {
                    if (!m_highlighter) return;

                    cvSelectionHighlighter::HighlightMode hlMode =
                            static_cast<cvSelectionHighlighter::HighlightMode>(
                                    mode);
                    m_highlighter->setHighlightOpacity(opacity, hlMode);

                    // Refresh display
                    PclUtils::PCLVis* pclVis = getPCLVis();
                    if (pclVis) {
                        pclVis->UpdateScreen();
                    }

                    CVLog::Print(QString("[cvSelectionPropertiesWidget] "
                                         "Opacity updated for mode %1: %2")
                                         .arg(mode)
                                         .arg(opacity));
                });

        // Sync UI with highlighter's current settings
        syncUIWithHighlighter();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::syncUIWithHighlighter() {
    if (!m_highlighter) {
        return;
    }

    // Get current colors and opacities from highlighter
    const double* hoverColor =
            m_highlighter->getHighlightColor(cvSelectionHighlighter::HOVER);
    const double* preselectedColor = m_highlighter->getHighlightColor(
            cvSelectionHighlighter::PRESELECTED);
    const double* selectedColor =
            m_highlighter->getHighlightColor(cvSelectionHighlighter::SELECTED);
    const double* boundaryColor =
            m_highlighter->getHighlightColor(cvSelectionHighlighter::BOUNDARY);

    double hoverOpacity =
            m_highlighter->getHighlightOpacity(cvSelectionHighlighter::HOVER);
    double preselectedOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::PRESELECTED);
    double selectedOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::SELECTED);
    double boundaryOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::BOUNDARY);

    // Update internal color arrays
    if (hoverColor) {
        m_hoverColor[0] = hoverColor[0];
        m_hoverColor[1] = hoverColor[1];
        m_hoverColor[2] = hoverColor[2];
    }
    if (preselectedColor) {
        m_preselectedColor[0] = preselectedColor[0];
        m_preselectedColor[1] = preselectedColor[1];
        m_preselectedColor[2] = preselectedColor[2];
    }
    if (selectedColor) {
        m_selectedColor[0] = selectedColor[0];
        m_selectedColor[1] = selectedColor[1];
        m_selectedColor[2] = selectedColor[2];
    }
    if (boundaryColor) {
        m_boundaryColor[0] = boundaryColor[0];
        m_boundaryColor[1] = boundaryColor[1];
        m_boundaryColor[2] = boundaryColor[2];
    }

    // Update UI controls (block signals to prevent triggering changes)
    if (m_hoverColorButton) {
        m_hoverColorButton->setStyleSheet(
                QString("background-color: rgb(%1, %2, %3);")
                        .arg(int(m_hoverColor[0] * 255))
                        .arg(int(m_hoverColor[1] * 255))
                        .arg(int(m_hoverColor[2] * 255)));
    }
    if (m_preselectedColorButton) {
        m_preselectedColorButton->setStyleSheet(
                QString("background-color: rgb(%1, %2, %3);")
                        .arg(int(m_preselectedColor[0] * 255))
                        .arg(int(m_preselectedColor[1] * 255))
                        .arg(int(m_preselectedColor[2] * 255)));
    }
    if (m_selectedColorButton) {
        m_selectedColorButton->setStyleSheet(
                QString("background-color: rgb(%1, %2, %3);")
                        .arg(int(m_selectedColor[0] * 255))
                        .arg(int(m_selectedColor[1] * 255))
                        .arg(int(m_selectedColor[2] * 255)));
    }
    if (m_boundaryColorButton) {
        m_boundaryColorButton->setStyleSheet(
                QString("background-color: rgb(%1, %2, %3);")
                        .arg(int(m_boundaryColor[0] * 255))
                        .arg(int(m_boundaryColor[1] * 255))
                        .arg(int(m_boundaryColor[2] * 255)));
    }

    // Update opacity spinboxes
    if (m_hoverOpacitySpin) {
        m_hoverOpacitySpin->blockSignals(true);
        m_hoverOpacitySpin->setValue(hoverOpacity);
        m_hoverOpacitySpin->blockSignals(false);
    }
    if (m_preselectedOpacitySpin) {
        m_preselectedOpacitySpin->blockSignals(true);
        m_preselectedOpacitySpin->setValue(preselectedOpacity);
        m_preselectedOpacitySpin->blockSignals(false);
    }
    if (m_selectedOpacitySpin) {
        m_selectedOpacitySpin->blockSignals(true);
        m_selectedOpacitySpin->setValue(selectedOpacity);
        m_selectedOpacitySpin->blockSignals(false);
    }
    if (m_boundaryOpacitySpin) {
        m_boundaryOpacitySpin->blockSignals(true);
        m_boundaryOpacitySpin->setValue(boundaryOpacity);
        m_boundaryOpacitySpin->blockSignals(false);
    }

    CVLog::PrintDebug(
            "[cvSelectionPropertiesWidget] UI synchronized with highlighter "
            "settings");
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupUi() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // Create tab widget
    m_tabWidget = new QTabWidget(this);
    mainLayout->addWidget(m_tabWidget);

    // Setup tabs
    setupHighlightTab();
    setupStatisticsTab();
    setupExportTab();
    setupAdvancedTab();  // New: algebra, filter, bookmarks, annotations

    setLayout(mainLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupHighlightTab() {
    m_highlightTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_highlightTab);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(10);

    // === Highlight Colors Group ===
    QGroupBox* colorsGroup = new QGroupBox(tr("Highlight Colors"));
    QFormLayout* colorsLayout = new QFormLayout();
    colorsLayout->setSpacing(8);

    // Hover color
    m_hoverColorButton = new QPushButton();
    m_hoverColorButton->setFixedSize(80, 25);
    m_hoverColorButton->setStyleSheet(
            QString("background-color: rgb(%1, %2, %3);")
                    .arg(int(m_hoverColor[0] * 255))
                    .arg(int(m_hoverColor[1] * 255))
                    .arg(int(m_hoverColor[2] * 255)));
    connect(m_hoverColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onHoverColorClicked);
    colorsLayout->addRow(tr("Hover:"), m_hoverColorButton);

    // Preselected color
    m_preselectedColorButton = new QPushButton();
    m_preselectedColorButton->setFixedSize(80, 25);
    m_preselectedColorButton->setStyleSheet(
            QString("background-color: rgb(%1, %2, %3);")
                    .arg(int(m_preselectedColor[0] * 255))
                    .arg(int(m_preselectedColor[1] * 255))
                    .arg(int(m_preselectedColor[2] * 255)));
    connect(m_preselectedColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onPreselectedColorClicked);
    colorsLayout->addRow(tr("Pre-selected:"), m_preselectedColorButton);

    // Selected color
    m_selectedColorButton = new QPushButton();
    m_selectedColorButton->setFixedSize(80, 25);
    m_selectedColorButton->setStyleSheet(
            QString("background-color: rgb(%1, %2, %3);")
                    .arg(int(m_selectedColor[0] * 255))
                    .arg(int(m_selectedColor[1] * 255))
                    .arg(int(m_selectedColor[2] * 255)));
    connect(m_selectedColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onSelectedColorClicked);
    colorsLayout->addRow(tr("Selected:"), m_selectedColorButton);

    // Boundary color
    m_boundaryColorButton = new QPushButton();
    m_boundaryColorButton->setFixedSize(80, 25);
    m_boundaryColorButton->setStyleSheet(
            QString("background-color: rgb(%1, %2, %3);")
                    .arg(int(m_boundaryColor[0] * 255))
                    .arg(int(m_boundaryColor[1] * 255))
                    .arg(int(m_boundaryColor[2] * 255)));
    connect(m_boundaryColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onBoundaryColorClicked);
    colorsLayout->addRow(tr("Boundary:"), m_boundaryColorButton);

    colorsGroup->setLayout(colorsLayout);
    layout->addWidget(colorsGroup);

    // === Opacity Group ===
    QGroupBox* opacityGroup = new QGroupBox(tr("Opacity"));
    QFormLayout* opacityLayout = new QFormLayout();
    opacityLayout->setSpacing(8);

    // Hover opacity
    m_hoverOpacitySpin = new QDoubleSpinBox();
    m_hoverOpacitySpin->setRange(0.0, 1.0);
    m_hoverOpacitySpin->setSingleStep(0.1);
    m_hoverOpacitySpin->setValue(0.9);
    m_hoverOpacitySpin->setDecimals(2);
    connect(m_hoverOpacitySpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvSelectionPropertiesWidget::onHoverOpacityChanged);
    opacityLayout->addRow(tr("Hover:"), m_hoverOpacitySpin);

    // Preselected opacity
    m_preselectedOpacitySpin = new QDoubleSpinBox();
    m_preselectedOpacitySpin->setRange(0.0, 1.0);
    m_preselectedOpacitySpin->setSingleStep(0.1);
    m_preselectedOpacitySpin->setValue(0.8);
    m_preselectedOpacitySpin->setDecimals(2);
    connect(m_preselectedOpacitySpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvSelectionPropertiesWidget::onPreselectedOpacityChanged);
    opacityLayout->addRow(tr("Pre-selected:"), m_preselectedOpacitySpin);

    // Selected opacity
    m_selectedOpacitySpin = new QDoubleSpinBox();
    m_selectedOpacitySpin->setRange(0.0, 1.0);
    m_selectedOpacitySpin->setSingleStep(0.1);
    m_selectedOpacitySpin->setValue(1.0);
    m_selectedOpacitySpin->setDecimals(2);
    connect(m_selectedOpacitySpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvSelectionPropertiesWidget::onSelectedOpacityChanged);
    opacityLayout->addRow(tr("Selected:"), m_selectedOpacitySpin);

    // Boundary opacity
    m_boundaryOpacitySpin = new QDoubleSpinBox();
    m_boundaryOpacitySpin->setRange(0.0, 1.0);
    m_boundaryOpacitySpin->setSingleStep(0.1);
    m_boundaryOpacitySpin->setValue(0.85);
    m_boundaryOpacitySpin->setDecimals(2);
    connect(m_boundaryOpacitySpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &cvSelectionPropertiesWidget::onBoundaryOpacityChanged);
    opacityLayout->addRow(tr("Boundary:"), m_boundaryOpacitySpin);

    opacityGroup->setLayout(opacityLayout);
    layout->addWidget(opacityGroup);

    // === Tooltip Settings Group ===
    QGroupBox* tooltipGroup = new QGroupBox(tr("Tooltip Settings"));
    QFormLayout* tooltipLayout = new QFormLayout();
    tooltipLayout->setSpacing(8);

    m_showTooltipsCheckBox = new QCheckBox(tr("Show tooltips on hover"));
    m_showTooltipsCheckBox->setChecked(true);
    connect(m_showTooltipsCheckBox, &QCheckBox::toggled, this,
            &cvSelectionPropertiesWidget::onShowTooltipsToggled);
    tooltipLayout->addRow(m_showTooltipsCheckBox);

    m_maxAttributesSpin = new QSpinBox();
    m_maxAttributesSpin->setRange(1, 50);
    m_maxAttributesSpin->setValue(15);
    connect(m_maxAttributesSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &cvSelectionPropertiesWidget::onMaxAttributesChanged);
    tooltipLayout->addRow(tr("Max attributes:"), m_maxAttributesSpin);

    tooltipGroup->setLayout(tooltipLayout);
    layout->addWidget(tooltipGroup);

    layout->addStretch();

    m_tabWidget->addTab(m_highlightTab, tr("Highlight"));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupStatisticsTab() {
    m_statisticsTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_statisticsTab);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(10);

    // === Statistics Group ===
    QGroupBox* statsGroup = new QGroupBox(tr("Selection Statistics"));
    QFormLayout* statsLayout = new QFormLayout();
    statsLayout->setSpacing(8);

    m_countLabel = new QLabel(tr("0"));
    statsLayout->addRow(tr("Count:"), m_countLabel);

    m_typeLabel = new QLabel(tr("None"));
    statsLayout->addRow(tr("Type:"), m_typeLabel);

    m_boundsLabel = new QLabel(tr("N/A"));
    m_boundsLabel->setWordWrap(true);
    statsLayout->addRow(tr("Bounds:"), m_boundsLabel);

    m_centerLabel = new QLabel(tr("N/A"));
    statsLayout->addRow(tr("Center:"), m_centerLabel);

    m_volumeLabel = new QLabel(tr("N/A"));
    statsLayout->addRow(tr("Volume:"), m_volumeLabel);

    statsGroup->setLayout(statsLayout);
    layout->addWidget(statsGroup);

    // === Selection List Group ===
    QGroupBox* listGroup = new QGroupBox(tr("Selected IDs"));
    QVBoxLayout* listLayout = new QVBoxLayout();

    m_listInfoLabel = new QLabel(tr("No selection"));
    m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");
    listLayout->addWidget(m_listInfoLabel);

    m_selectionListWidget = new QListWidget();
    m_selectionListWidget->setSelectionMode(
            QAbstractItemView::ExtendedSelection);
    m_selectionListWidget->setMaximumHeight(200);
    connect(m_selectionListWidget, &QListWidget::itemClicked, this,
            &cvSelectionPropertiesWidget::onSelectionListItemClicked);
    listLayout->addWidget(m_selectionListWidget);

    listGroup->setLayout(listLayout);
    layout->addWidget(listGroup);

    layout->addStretch();

    m_tabWidget->addTab(m_statisticsTab, tr("Statistics"));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupExportTab() {
    m_exportTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_exportTab);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(10);

    // === Export Group ===
    QGroupBox* exportGroup = new QGroupBox(tr("Export Selection"));
    QVBoxLayout* exportLayout = new QVBoxLayout();
    exportLayout->setSpacing(8);

    m_exportInfoLabel = new QLabel(
            tr("Export the current selection to the scene or to a file."));
    m_exportInfoLabel->setWordWrap(true);
    m_exportInfoLabel->setStyleSheet("color: gray; font-size: 9pt;");
    exportLayout->addWidget(m_exportInfoLabel);

    // Export to mesh button
    m_exportToMeshButton = new QPushButton(tr("Export to Mesh"));
    m_exportToMeshButton->setEnabled(false);
    m_exportToMeshButton->setToolTip(
            tr("Create a ccMesh from the selected cells and add to the scene"));
    connect(m_exportToMeshButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExportToMeshClicked);
    exportLayout->addWidget(m_exportToMeshButton);

    // Export to point cloud button
    m_exportToPointCloudButton = new QPushButton(tr("Export to Point Cloud"));
    m_exportToPointCloudButton->setEnabled(false);
    m_exportToPointCloudButton->setToolTip(
            tr("Create a ccPointCloud from the selected points and add to the "
               "scene"));
    connect(m_exportToPointCloudButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExportToPointCloudClicked);
    exportLayout->addWidget(m_exportToPointCloudButton);

    exportLayout->addSpacing(10);

    // Export to file button (uses MainWindow's standard save dialog)
    m_exportToFileButton = new QPushButton(tr("Export to File..."));
    m_exportToFileButton->setEnabled(false);
    m_exportToFileButton->setToolTip(
            tr("Export selection to a file in the selected format"));
    connect(m_exportToFileButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExportToFileClicked);
    exportLayout->addWidget(m_exportToFileButton);

    exportLayout->addSpacing(10);

    // Copy IDs button
    m_copyIDsButton = new QPushButton(tr("Copy IDs to Clipboard"));
    m_copyIDsButton->setEnabled(false);
    m_copyIDsButton->setToolTip(tr("Copy selected IDs to the clipboard"));
    connect(m_copyIDsButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onCopyIDsClicked);
    exportLayout->addWidget(m_copyIDsButton);

    exportGroup->setLayout(exportLayout);
    layout->addWidget(exportGroup);

    layout->addStretch();

    m_tabWidget->addTab(m_exportTab, tr("Export"));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupAdvancedTab() {
    m_advancedTab = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(m_advancedTab);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(10);

    // === Algebra Operations Group ===
    QGroupBox* algebraGroup = new QGroupBox(tr("Selection Algebra"));
    QVBoxLayout* algebraLayout = new QVBoxLayout();
    algebraLayout->setSpacing(8);

    QLabel* algebraInfo =
            new QLabel(tr("Perform set operations on selections (union, "
                          "intersection, etc.)"));
    algebraInfo->setWordWrap(true);
    algebraInfo->setStyleSheet("color: gray; font-size: 9pt;");
    algebraLayout->addWidget(algebraInfo);

    m_algebraOpCombo = new QComboBox();
    m_algebraOpCombo->addItem(tr("Union (A ∪ B)"), cvSelectionAlgebra::UNION);
    m_algebraOpCombo->addItem(tr("Intersection (A ∩ B)"),
                              cvSelectionAlgebra::INTERSECTION);
    m_algebraOpCombo->addItem(tr("Difference (A - B)"),
                              cvSelectionAlgebra::DIFFERENCE);
    m_algebraOpCombo->addItem(tr("Symmetric Diff (A △ B)"),
                              cvSelectionAlgebra::SYMMETRIC_DIFF);
    m_algebraOpCombo->addItem(tr("Complement (~A)"),
                              cvSelectionAlgebra::COMPLEMENT);
    algebraLayout->addWidget(m_algebraOpCombo);

    m_applyAlgebraButton = new QPushButton(tr("Apply Operation"));
    m_applyAlgebraButton->setEnabled(false);
    connect(m_applyAlgebraButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onAlgebraOperationTriggered);
    algebraLayout->addWidget(m_applyAlgebraButton);

    m_extractBoundaryButton = new QPushButton(tr("Extract Boundary"));
    m_extractBoundaryButton->setEnabled(false);
    m_extractBoundaryButton->setToolTip(
            tr("Extract boundary cells of selection"));
    connect(m_extractBoundaryButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExtractBoundaryClicked);
    algebraLayout->addWidget(m_extractBoundaryButton);

    algebraGroup->setLayout(algebraLayout);
    layout->addWidget(algebraGroup);

    // === Filtering Group ===
    QGroupBox* filterGroup = new QGroupBox(tr("Advanced Filtering"));
    QVBoxLayout* filterLayout = new QVBoxLayout();
    filterLayout->setSpacing(8);

    QLabel* filterInfo = new QLabel(tr(
            "Filter selection by attributes, geometry, or spatial criteria"));
    filterInfo->setWordWrap(true);
    filterInfo->setStyleSheet("color: gray; font-size: 9pt;");
    filterLayout->addWidget(filterInfo);

    m_filterTypeCombo = new QComboBox();
    m_filterTypeCombo->addItem(tr("By Attribute Range"),
                               cvSelectionFilter::ATTRIBUTE_RANGE);
    m_filterTypeCombo->addItem(tr("By Cell Area"),
                               cvSelectionFilter::GEOMETRIC_AREA);
    m_filterTypeCombo->addItem(tr("By Normal Angle"),
                               cvSelectionFilter::GEOMETRIC_ANGLE);
    m_filterTypeCombo->addItem(tr("By Bounding Box"),
                               cvSelectionFilter::SPATIAL_BBOX);
    m_filterTypeCombo->addItem(tr("By Distance"),
                               cvSelectionFilter::SPATIAL_DISTANCE);
    m_filterTypeCombo->addItem(tr("By Neighbor Count"),
                               cvSelectionFilter::TOPOLOGY_NEIGHBORS);
    filterLayout->addWidget(m_filterTypeCombo);

    m_applyFilterButton = new QPushButton(tr("Configure & Apply Filter..."));
    m_applyFilterButton->setEnabled(false);
    connect(m_applyFilterButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onFilterOperationTriggered);
    filterLayout->addWidget(m_applyFilterButton);

    filterGroup->setLayout(filterLayout);
    layout->addWidget(filterGroup);

    // === Bookmarks Group ===
    QGroupBox* bookmarksGroup = new QGroupBox(tr("Selection Bookmarks"));
    QVBoxLayout* bookmarksLayout = new QVBoxLayout();
    bookmarksLayout->setSpacing(8);

    QLabel* bookmarksInfo = new QLabel(tr("Save and restore named selections"));
    bookmarksInfo->setWordWrap(true);
    bookmarksInfo->setStyleSheet("color: gray; font-size: 9pt;");
    bookmarksLayout->addWidget(bookmarksInfo);

    m_bookmarkCombo = new QComboBox();
    m_bookmarkCombo->setEditable(false);
    bookmarksLayout->addWidget(m_bookmarkCombo);

    QHBoxLayout* bookmarkButtons = new QHBoxLayout();
    m_saveBookmarkButton = new QPushButton(tr("Save..."));
    m_saveBookmarkButton->setEnabled(false);
    connect(m_saveBookmarkButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onSaveBookmarkClicked);
    bookmarkButtons->addWidget(m_saveBookmarkButton);

    m_loadBookmarkButton = new QPushButton(tr("Load"));
    m_loadBookmarkButton->setEnabled(false);
    connect(m_loadBookmarkButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onLoadBookmarkClicked);
    bookmarkButtons->addWidget(m_loadBookmarkButton);

    bookmarksLayout->addLayout(bookmarkButtons);
    
    // Batch export button
    m_batchExportBookmarksButton = new QPushButton(tr("Batch Export All..."));
    m_batchExportBookmarksButton->setEnabled(false);
    m_batchExportBookmarksButton->setToolTip(
            tr("Export all bookmarked selections to files"));
    connect(m_batchExportBookmarksButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onBatchExportBookmarksClicked);
    bookmarksLayout->addWidget(m_batchExportBookmarksButton);
    bookmarksGroup->setLayout(bookmarksLayout);
    layout->addWidget(bookmarksGroup);

    // === Annotations Group ===
    QGroupBox* annotationsGroup = new QGroupBox(tr("Selection Annotations"));
    QVBoxLayout* annotationsLayout = new QVBoxLayout();
    annotationsLayout->setSpacing(8);

    QLabel* annotationsInfo =
            new QLabel(tr("Add text annotations to selections"));
    annotationsInfo->setWordWrap(true);
    annotationsInfo->setStyleSheet("color: gray; font-size: 9pt;");
    annotationsLayout->addWidget(annotationsInfo);

    m_addAnnotationButton = new QPushButton(tr("Add Annotation..."));
    m_addAnnotationButton->setEnabled(false);
    connect(m_addAnnotationButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onAddAnnotationClicked);
    annotationsLayout->addWidget(m_addAnnotationButton);

    annotationsGroup->setLayout(annotationsLayout);
    layout->addWidget(annotationsGroup);

    layout->addStretch();

    m_tabWidget->addTab(m_advancedTab, tr("Advanced"));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setSelectionManager(
        cvViewSelectionManager* manager) {
    m_selectionManager = manager;

    // Update bookmark combo when manager is set
    if (m_selectionManager && m_selectionManager->getBookmarks()) {
        updateBookmarkCombo();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateBookmarkCombo() {
    if (!m_selectionManager || !m_selectionManager->getBookmarks()) {
        return;
    }

    m_bookmarkCombo->clear();
    QStringList names = m_selectionManager->getBookmarks()->bookmarkNames();
    m_bookmarkCombo->addItems(names);
    m_loadBookmarkButton->setEnabled(!names.isEmpty());
    m_batchExportBookmarksButton->setEnabled(!names.isEmpty());
}

//-----------------------------------------------------------------------------
bool cvSelectionPropertiesWidget::updateSelection(
        const cvSelectionData& selectionData, vtkPolyData* polyData) {
    m_selectionData = selectionData;

    if (m_selectionData.isEmpty()) {
        clearSelection();
        return false;
    }

    // Get polyData if not provided (using centralized ParaView-style method)
    if (!polyData) {
        polyData = getPolyDataForSelection(&m_selectionData);
    }

    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No polyData available for "
                "statistics");
        return false;
    }

    // Update statistics and list
    updateStatistics(polyData);
    updateSelectionList(polyData);

    // Enable export buttons
    bool isCells =
            (m_selectionData.fieldAssociation() == cvSelectionData::CELLS);
    bool isPoints =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);

    m_exportToMeshButton->setEnabled(isCells && m_selectionCount > 0);
    m_exportToPointCloudButton->setEnabled(isPoints && m_selectionCount > 0);
    m_exportToFileButton->setEnabled(m_selectionCount > 0);
    m_copyIDsButton->setEnabled(m_selectionCount > 0);

    // Enable advanced tab buttons
    m_applyAlgebraButton->setEnabled(m_selectionCount > 0);
    m_extractBoundaryButton->setEnabled(isCells && m_selectionCount > 0);
    m_applyFilterButton->setEnabled(m_selectionCount > 0);
    m_saveBookmarkButton->setEnabled(m_selectionCount > 0);
    m_addAnnotationButton->setEnabled(m_selectionCount > 0);

    return true;
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::clearSelection() {
    m_selectionData.clear();
    m_selectionCount = 0;
    m_selectionType = tr("None");

    // Clear statistics
    m_countLabel->setText(QString::number(m_selectionCount));
    m_typeLabel->setText(m_selectionType);
    m_boundsLabel->setText(tr("N/A"));
    m_centerLabel->setText(tr("N/A"));
    m_volumeLabel->setText(tr("N/A"));

    // Clear list
    m_selectionListWidget->clear();
    m_listInfoLabel->setText(tr("No selection"));
    m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");

    // Disable export buttons
    m_exportToMeshButton->setEnabled(false);
    m_exportToPointCloudButton->setEnabled(false);
    m_exportToFileButton->setEnabled(false);
    m_copyIDsButton->setEnabled(false);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateStatistics(vtkPolyData* polyData) {
    if (!polyData) {
        return;
    }

    m_selectionCount = m_selectionData.count();
    m_selectionType = m_selectionData.fieldTypeString();

    // Update labels
    m_countLabel->setText(QString::number(m_selectionCount));
    m_typeLabel->setText(m_selectionType);

    // Compute bounding box
    if (m_selectionCount > 0) {
        computeBoundingBox(polyData, m_bounds);

        // Bounds
        m_boundsLabel->setText(formatBounds(m_bounds));

        // Center
        m_center[0] = (m_bounds[0] + m_bounds[1]) / 2.0;
        m_center[1] = (m_bounds[2] + m_bounds[3]) / 2.0;
        m_center[2] = (m_bounds[4] + m_bounds[5]) / 2.0;
        m_centerLabel->setText(QString("(%1, %2, %3)")
                                       .arg(m_center[0], 0, 'g', 6)
                                       .arg(m_center[1], 0, 'g', 6)
                                       .arg(m_center[2], 0, 'g', 6));

        // Volume
        double dx = m_bounds[1] - m_bounds[0];
        double dy = m_bounds[3] - m_bounds[2];
        double dz = m_bounds[5] - m_bounds[4];
        m_volume = dx * dy * dz;
        m_volumeLabel->setText(QString("%1").arg(m_volume, 0, 'g', 6));
    } else {
        m_boundsLabel->setText(tr("N/A"));
        m_centerLabel->setText(tr("N/A"));
        m_volumeLabel->setText(tr("N/A"));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateSelectionList(vtkPolyData* polyData) {
    m_selectionListWidget->clear();

    QVector<qint64> ids = m_selectionData.ids();
    if (ids.isEmpty()) {
        m_listInfoLabel->setText(tr("No selection"));
        m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");
        return;
    }

    // Update info label
    m_listInfoLabel->setText(
            tr("Showing %1 %2")
                    .arg(ids.size())
                    .arg(m_selectionData.fieldTypeString().toLower()));
    m_listInfoLabel->setStyleSheet("font-weight: bold;");

    // Add IDs to list (limit to first 1000 for performance)
    int maxDisplay = qMin(ids.size(), 1000);
    for (int i = 0; i < maxDisplay; ++i) {
        QString itemText = QString("ID: %1").arg(ids[i]);

        // Add coordinate info if points
        if (m_selectionData.fieldAssociation() == cvSelectionData::POINTS &&
            polyData) {
            if (ids[i] >= 0 && ids[i] < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(ids[i], pt);
                itemText += QString(" (%1, %2, %3)")
                                    .arg(pt[0], 0, 'f', 2)
                                    .arg(pt[1], 0, 'f', 2)
                                    .arg(pt[2], 0, 'f', 2);
            }
        }

        m_selectionListWidget->addItem(itemText);
    }

    if (ids.size() > maxDisplay) {
        m_selectionListWidget->addItem(
                tr("... and %1 more").arg(ids.size() - maxDisplay));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::computeBoundingBox(vtkPolyData* polyData,
                                                     double bounds[6]) {
    // Initialize bounds
    bounds[0] = bounds[2] = bounds[4] = std::numeric_limits<double>::max();
    bounds[1] = bounds[3] = bounds[5] = std::numeric_limits<double>::lowest();

    QVector<qint64> ids = m_selectionData.ids();

    if (m_selectionData.fieldAssociation() == cvSelectionData::POINTS) {
        // Points: compute bounds from selected point coordinates
        for (qint64 id : ids) {
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(id, pt);

                bounds[0] = qMin(bounds[0], pt[0]);
                bounds[1] = qMax(bounds[1], pt[0]);
                bounds[2] = qMin(bounds[2], pt[1]);
                bounds[3] = qMax(bounds[3], pt[1]);
                bounds[4] = qMin(bounds[4], pt[2]);
                bounds[5] = qMax(bounds[5], pt[2]);
            }
        }
    } else {
        // Cells: compute bounds from all points in selected cells
        for (qint64 id : ids) {
            if (id >= 0 && id < polyData->GetNumberOfCells()) {
                vtkCell* cell = polyData->GetCell(id);
                if (cell) {
                    vtkIdType npts = cell->GetNumberOfPoints();
                    for (vtkIdType i = 0; i < npts; ++i) {
                        double pt[3];
                        polyData->GetPoint(cell->GetPointId(i), pt);

                        bounds[0] = qMin(bounds[0], pt[0]);
                        bounds[1] = qMax(bounds[1], pt[0]);
                        bounds[2] = qMin(bounds[2], pt[1]);
                        bounds[3] = qMax(bounds[3], pt[1]);
                        bounds[4] = qMin(bounds[4], pt[2]);
                        bounds[5] = qMax(bounds[5], pt[2]);
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
QString cvSelectionPropertiesWidget::formatBounds(const double bounds[6]) {
    return QString("X: [%1, %2]\nY: [%3, %4]\nZ: [%5, %6]")
            .arg(bounds[0], 0, 'g', 6)
            .arg(bounds[1], 0, 'g', 6)
            .arg(bounds[2], 0, 'g', 6)
            .arg(bounds[3], 0, 'g', 6)
            .arg(bounds[4], 0, 'g', 6)
            .arg(bounds[5], 0, 'g', 6);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::showColorDialog(const QString& title,
                                                  double currentColor[3],
                                                  int mode) {
    QColor initialColor(int(currentColor[0] * 255), int(currentColor[1] * 255),
                        int(currentColor[2] * 255));

    QColor newColor = QColorDialog::getColor(initialColor, this, title);

    if (newColor.isValid()) {
        currentColor[0] = newColor.redF();
        currentColor[1] = newColor.greenF();
        currentColor[2] = newColor.blueF();

        // Update button color
        QPushButton* button = nullptr;
        switch (mode) {
            case 0:  // HOVER
                button = m_hoverColorButton;
                break;
            case 1:  // PRESELECTED
                button = m_preselectedColorButton;
                break;
            case 2:  // SELECTED
                button = m_selectedColorButton;
                break;
            case 3:  // BOUNDARY
                button = m_boundaryColorButton;
                break;
        }

        if (button) {
            button->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                                          .arg(int(currentColor[0] * 255))
                                          .arg(int(currentColor[1] * 255))
                                          .arg(int(currentColor[2] * 255)));
        }

        // Emit signal
        emit highlightColorChanged(currentColor[0], currentColor[1],
                                   currentColor[2], mode);
    }
}

//-----------------------------------------------------------------------------
// Slot implementations
//-----------------------------------------------------------------------------

void cvSelectionPropertiesWidget::onHoverColorClicked() {
    showColorDialog(tr("Select Hover Highlight Color"), m_hoverColor, 0);
}

void cvSelectionPropertiesWidget::onPreselectedColorClicked() {
    showColorDialog(tr("Select Pre-selected Highlight Color"),
                    m_preselectedColor, 1);
}

void cvSelectionPropertiesWidget::onSelectedColorClicked() {
    showColorDialog(tr("Select Selected Highlight Color"), m_selectedColor, 2);
}

void cvSelectionPropertiesWidget::onBoundaryColorClicked() {
    showColorDialog(tr("Select Boundary Highlight Color"), m_boundaryColor, 3);
}

void cvSelectionPropertiesWidget::onHoverOpacityChanged(double value) {
    emit highlightOpacityChanged(value, 0);  // HOVER = 0
}

void cvSelectionPropertiesWidget::onPreselectedOpacityChanged(double value) {
    emit highlightOpacityChanged(value, 1);  // PRESELECTED = 1
}

void cvSelectionPropertiesWidget::onSelectedOpacityChanged(double value) {
    emit highlightOpacityChanged(value, 2);  // SELECTED = 2
}

void cvSelectionPropertiesWidget::onBoundaryOpacityChanged(double value) {
    emit highlightOpacityChanged(value, 3);  // BOUNDARY = 3
}

void cvSelectionPropertiesWidget::onShowTooltipsToggled(bool checked) {
    CVLog::Print(QString("[cvSelectionPropertiesWidget] Show tooltips: %1")
                         .arg(checked ? "enabled" : "disabled"));

    // Emit signal to notify tooltip tools
    int maxAttrs = m_maxAttributesSpin ? m_maxAttributesSpin->value() : 15;
    emit tooltipSettingsChanged(checked, maxAttrs);
}

void cvSelectionPropertiesWidget::onMaxAttributesChanged(int value) {
    if (m_tooltipHelper) {
        m_tooltipHelper->setMaxAttributes(value);
        CVLog::Print(
                QString("[cvSelectionPropertiesWidget] Max attributes set to: "
                        "%1")
                        .arg(value));
    }

    // Emit signal to notify tooltip tools
    bool showTooltips =
            m_showTooltipsCheckBox ? m_showTooltipsCheckBox->isChecked() : true;
    emit tooltipSettingsChanged(showTooltips, value);
}

void cvSelectionPropertiesWidget::onExportToMeshClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to export");
        return;
    }

    if (m_selectionData.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Can only export cells as mesh");
        return;
    }

    // Export selection to mesh
    // Get polyData (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to get polyData from "
                "visualizer");
        return;
    }

    cvSelectionExporter::ExportOptions options;
    ccMesh* mesh = cvSelectionExporter::exportToMesh(polyData, m_selectionData,
                                                     options);

    if (mesh) {
        mesh->setName("Selection (Mesh)");

        // Add directly to scene via ecvDisplayTools (no dependency on
        // MainWindow)
        ecvDisplayTools::AddToOwnDB(mesh, false);

        CVLog::Print(QString("[cvSelectionPropertiesWidget] Exported %1 cells "
                             "as mesh to scene")
                             .arg(m_selectionData.count()));
    } else {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to export selection as "
                "mesh");
    }
}

void cvSelectionPropertiesWidget::onExportToPointCloudClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to export");
        return;
    }

    if (m_selectionData.fieldAssociation() != cvSelectionData::POINTS) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Can only export points as point "
                "cloud");
        return;
    }

    // Export selection to point cloud
    // Get polyData (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to get polyData from "
                "visualizer");
        return;
    }

    cvSelectionExporter::ExportOptions options;
    ccPointCloud* cloud = cvSelectionExporter::exportToPointCloud(
            polyData, m_selectionData, options);

    if (cloud) {
        cloud->setName("Selection (Points)");

        // Add directly to scene via ecvDisplayTools (no dependency on
        // MainWindow)
        ecvDisplayTools::AddToOwnDB(cloud, false);

        CVLog::Print(QString("[cvSelectionPropertiesWidget] Exported %1 points "
                             "as point cloud to scene")
                             .arg(m_selectionData.count()));
    } else {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to export selection as "
                "point cloud");
    }
}

void cvSelectionPropertiesWidget::onExportToFileClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to export");
        return;
    }

    // Get polyData (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to get polyData from "
                "visualizer");
        return;
    }

    // Export selection to temporary object
    ccHObject* exportedObject = nullptr;
    bool isMesh = false;

    if (m_selectionData.fieldAssociation() == cvSelectionData::CELLS) {
        // Export as mesh
        cvSelectionExporter::ExportOptions options;
        exportedObject = cvSelectionExporter::exportToMesh(
                polyData, m_selectionData, options);
        if (exportedObject) {
            exportedObject->setName("Selection");
            isMesh = true;
        }
    } else if (m_selectionData.fieldAssociation() == cvSelectionData::POINTS) {
        // Export as point cloud
        cvSelectionExporter::ExportOptions options;
        exportedObject = cvSelectionExporter::exportToPointCloud(
                polyData, m_selectionData, options);
        if (exportedObject) {
            exportedObject->setName("Selection");
            isMesh = false;
        }
    }

    if (!exportedObject) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to export selection");
        return;
    }

    // Use cvSelectionExporter's dialog-based save method
    if (!cvSelectionExporter::saveObjectToFileWithDialog(exportedObject, isMesh,
                                                         this)) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Export to file cancelled or "
                "failed");
    }

    // Clean up temporary object
    delete exportedObject;
}

void cvSelectionPropertiesWidget::onCopyIDsClicked() {
    QVector<qint64> ids = m_selectionData.ids();
    if (ids.isEmpty()) {
        return;
    }

    // Format IDs as comma-separated string
    QStringList idStrings;
    for (qint64 id : ids) {
        idStrings << QString::number(id);
    }

    QString idsText = idStrings.join(", ");

    // Copy to clipboard
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(idsText);

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Copied %1 IDs to "
                         "clipboard")
                         .arg(ids.size()));
}

void cvSelectionPropertiesWidget::onSelectionListItemClicked(
        QListWidgetItem* item) {
    if (!item) {
        return;
    }

    QString itemText = item->text();

    // Skip the "... and N more" item
    if (itemText.startsWith("...")) {
        return;
    }

    // Extract ID from item text
    qint64 id = extractIdFromItemText(itemText);
    if (id < 0) {
        CVLog::Warning(QString("[cvSelectionPropertiesWidget] Failed to parse "
                               "ID from: %1")
                               .arg(itemText));
        return;
    }

    // Highlight this specific item in 3D view
    highlightSingleItem(id);

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Highlighting %1 ID: %2")
                         .arg(m_selectionData.fieldTypeString().toLower())
                         .arg(id));
}

//-----------------------------------------------------------------------------
qint64 cvSelectionPropertiesWidget::extractIdFromItemText(
        const QString& itemText) {
    // Format: "ID: 123" or "ID: 123 (x, y, z)"
    QRegularExpression idRegex("ID:\\s*(\\d+)");
    QRegularExpressionMatch match = idRegex.match(itemText);

    if (match.hasMatch()) {
        bool ok;
        qint64 id = match.captured(1).toLongLong(&ok);
        if (ok) {
            return id;
        }
    }

    return -1;
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::highlightSingleItem(qint64 id) {
    if (!m_highlighter) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Highlighter not available");
        return;
    }

    // Get polyData using centralized ParaView-style method
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No polyData available for "
                "highlighting");
        return;
    }

    // Validate ID
    if (m_selectionData.fieldAssociation() == cvSelectionData::POINTS) {
        if (id < 0 || id >= polyData->GetNumberOfPoints()) {
            CVLog::Warning(QString("[cvSelectionPropertiesWidget] Point ID %1 "
                                   "out of range")
                                   .arg(id));
            return;
        }
    } else {
        if (id < 0 || id >= polyData->GetNumberOfCells()) {
            CVLog::Warning(QString("[cvSelectionPropertiesWidget] Cell ID %1 "
                                   "out of range")
                                   .arg(id));
            return;
        }
    }

    // Create a temporary single-item selection
    vtkSmartPointer<vtkIdTypeArray> singleIdArray =
            vtkSmartPointer<vtkIdTypeArray>::New();
    singleIdArray->InsertNextValue(id);

    // Temporarily highlight this single item with a special color
    // (white/bright) Use HOVER mode for temporary highlight
    m_highlighter->highlightSelection(singleIdArray,
                                      m_selectionData.fieldAssociation(),
                                      cvSelectionHighlighter::HOVER);

    // Optional: Focus camera on this item
    if (m_selectionData.fieldAssociation() == cvSelectionData::POINTS) {
        // Get point coordinates
        double pt[3];
        polyData->GetPoint(id, pt);

        // Calculate a small bounding box around the point
        double radius = 1.0;  // Adjust based on your data scale
        double bounds[6] = {pt[0] - radius, pt[0] + radius, pt[1] - radius,
                            pt[1] + radius, pt[2] - radius, pt[2] + radius};

        // Optional: Zoom to this point (commented out by default)
        // ccBBox bbox(CCVector3::fromArray(bounds),
        // CCVector3::fromArray(&bounds[3]));
        // ecvDisplayTools::UpdateConstellationCenterAndZoom(&bbox);

        CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Point %1 at "
                                  "(%2, %3, %4)")
                                  .arg(id)
                                  .arg(pt[0], 0, 'f', 2)
                                  .arg(pt[1], 0, 'f', 2)
                                  .arg(pt[2], 0, 'f', 2));
    } else {
        // For cells, we could calculate cell center
        vtkCell* cell = polyData->GetCell(id);
        if (cell) {
            double center[3] = {0, 0, 0};
            double* weights = new double[cell->GetNumberOfPoints()];
            double pcoords[3] = {0.5, 0.5, 0.5};  // parametric center
            int subId = 0;
            cell->EvaluateLocation(subId, pcoords, center, weights);
            delete[] weights;

            CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Cell %1 "
                                      "center at (%2, %3, %4)")
                                      .arg(id)
                                      .arg(center[0], 0, 'f', 2)
                                      .arg(center[1], 0, 'f', 2)
                                      .arg(center[2], 0, 'f', 2));
        }
    }

    // Refresh display
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        pclVis->UpdateScreen();
    }

    // Use a timer to restore original selection after 2 seconds
    QTimer::singleShot(2000, this, [this]() {
        if (m_highlighter && !m_selectionData.isEmpty()) {
            // Restore original full selection highlight
            m_highlighter->highlightSelection(
                    m_selectionData.vtkArray(),
                    m_selectionData.fieldAssociation(),
                    cvSelectionHighlighter::SELECTED);

            PclUtils::PCLVis* pclVis = getPCLVis();
            if (pclVis) {
                pclVis->UpdateScreen();
            }
        }
    });
}

//-----------------------------------------------------------------------------
// Advanced operations (new)
//-----------------------------------------------------------------------------

void cvSelectionPropertiesWidget::onAlgebraOperationTriggered() {
    if (!m_selectionManager || m_selectionData.isEmpty()) {
        return;
    }

    int op = m_algebraOpCombo->currentData().toInt();

    // For now, emit signal - actual implementation depends on having two
    // selections
    emit algebraOperationRequested(op);

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Algebra operation %1 "
                         "requested")
                         .arg(op));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onFilterOperationTriggered() {
    if (!m_selectionManager || m_selectionData.isEmpty()) {
        return;
    }

    int filterType = m_filterTypeCombo->currentData().toInt();
    auto filterEnum = static_cast<cvSelectionFilter::FilterType>(filterType);

    // Get polyData for filter operations
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        QMessageBox::warning(this, tr("Filter Operation"),
                             tr("No mesh data available for filtering."));
        return;
    }

    // Show configuration dialog (ParaView-style)
    cvFilterConfigDialog dialog(filterEnum, polyData, this);
    if (dialog.exec() == QDialog::Accepted) {
        QMap<QString, QVariant> params = dialog.getParameters();
        cvSelectionFilter::FilterType finalType = dialog.getFilterType();

        CVLog::Print(QString("[cvSelectionPropertiesWidget] Applying filter "
                             "type %1 with %2 parameters")
                             .arg(static_cast<int>(finalType))
                             .arg(params.size()));

        // Apply filter through selection manager
        cvSelectionFilter* filter = m_selectionManager->getFilter();
        if (filter) {
            // Filter application is now handled by onFilterOperationTriggered()
            // which shows cvFilterConfigDialog for parameter configuration
            QMessageBox::information(
                    this, tr("Filter Applied"),
                    tr("Filter configuration saved. Implementation of filter "
                       "application pending."));
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onExtractBoundaryClicked() {
    if (!m_selectionManager || m_selectionData.isEmpty()) {
        return;
    }

    cvSelectionAlgebra* algebra = m_selectionManager->getAlgebra();
    if (!algebra) {
        return;
    }

    // Get polyData using centralized ParaView-style method
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No polyData for boundary "
                "extraction");
        return;
    }

    cvSelectionData boundary =
            algebra->extractBoundary(polyData, m_selectionData);

    if (!boundary.isEmpty()) {
        m_selectionManager->setCurrentSelection(boundary);
        CVLog::Print(QString("[cvSelectionPropertiesWidget] Extracted "
                             "boundary: %1 -> %2 cells")
                             .arg(m_selectionData.count())
                             .arg(boundary.count()));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSaveBookmarkClicked() {
    if (!m_selectionManager || m_selectionData.isEmpty()) {
        return;
    }

    cvSelectionBookmarks* bookmarks = m_selectionManager->getBookmarks();
    if (!bookmarks) {
        return;
    }

    // Show input dialog for bookmark name
    bool ok;
    QString name = QInputDialog::getText(this, tr("Save Bookmark"),
                                         tr("Bookmark name:"),
                                         QLineEdit::Normal, QString(), &ok);

    if (ok && !name.isEmpty()) {
        if (bookmarks->addBookmark(name, m_selectionData)) {
            updateBookmarkCombo();
            emit bookmarkRequested(name);
            CVLog::Print(
                    QString("[cvSelectionPropertiesWidget] Saved bookmark: %1")
                            .arg(name));
        } else {
            QMessageBox::warning(this, tr("Save Bookmark"),
                                 tr("Failed to save bookmark '%1'.\n"
                                    "It may already exist.")
                                         .arg(name));
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onLoadBookmarkClicked() {
    if (!m_selectionManager) {
        return;
    }

    cvSelectionBookmarks* bookmarks = m_selectionManager->getBookmarks();
    if (!bookmarks) {
        return;
    }

    QString name = m_bookmarkCombo->currentText();
    if (name.isEmpty()) {
        return;
    }

    cvSelectionBookmarks::Bookmark bookmark = bookmarks->getBookmark(name);
    if (!bookmark.selection.isEmpty()) {
        m_selectionManager->setCurrentSelection(bookmark.selection);
        CVLog::Print(
                QString("[cvSelectionPropertiesWidget] Loaded bookmark: %1")
                        .arg(name));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onBatchExportBookmarksClicked() {
    if (!m_selectionManager) {
        return;
    }

    cvSelectionBookmarks* bookmarks = m_selectionManager->getBookmarks();
    if (!bookmarks) {
        return;
    }

    QStringList bookmarkNames = bookmarks->bookmarkNames();
    if (bookmarkNames.isEmpty()) {
        QMessageBox::information(this, tr("Batch Export"),
                                 tr("No bookmarks available to export."));
        return;
    }

    // Get output directory
    QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Select Output Directory for Batch Export"),
            QDir::homePath(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (outputDir.isEmpty()) {
        return;  // User cancelled
    }

    // Get file format
    QStringList formats;
    formats << "PLY (*.ply)" << "OBJ (*.obj)" << "STL (*.stl)"
            << "BIN (*.bin)";
    bool ok;
    QString formatChoice = QInputDialog::getItem(
            this, tr("Select Format"), tr("Export format:"), formats, 0, false,
            &ok);

    if (!ok || formatChoice.isEmpty()) {
        return;  // User cancelled
    }

    // Extract format extension
    QString format = "ply";  // default
    if (formatChoice.contains("obj", Qt::CaseInsensitive)) {
        format = "obj";
    } else if (formatChoice.contains("stl", Qt::CaseInsensitive)) {
        format = "stl";
    } else if (formatChoice.contains("bin", Qt::CaseInsensitive)) {
        format = "bin";
    }

    // Get polyData
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        QMessageBox::warning(this, tr("Batch Export"),
                             tr("No mesh data available for export."));
        return;
    }

    // Collect all bookmark selections
    QList<cvSelectionData> selections;
    for (const QString& name : bookmarkNames) {
        cvSelectionBookmarks::Bookmark bookmark = bookmarks->getBookmark(name);
        if (!bookmark.selection.isEmpty()) {
            selections.append(bookmark.selection);
        }
    }

    if (selections.isEmpty()) {
        QMessageBox::warning(this, tr("Batch Export"),
                             tr("No valid selections found in bookmarks."));
        return;
    }

    // Show progress dialog
    QProgressDialog progress(tr("Exporting bookmarks..."), tr("Cancel"), 0,
                             100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);

    // Perform batch export
    int successCount = cvSelectionExporter::batchExportToFiles(
            polyData, selections, outputDir, format, "bookmark",
            [&progress](int percent) {
                progress.setValue(percent);
                QApplication::processEvents();
            });

    progress.setValue(100);

    // Show result
    if (successCount > 0) {
        QMessageBox::information(
                this, tr("Batch Export Complete"),
                tr("Successfully exported %1 of %2 bookmarks to:\n%3")
                        .arg(successCount)
                        .arg(selections.size())
                        .arg(outputDir));
    } else {
        QMessageBox::warning(this, tr("Batch Export Failed"),
                             tr("Failed to export bookmarks."));
    }

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Batch exported %1/%2 "
                         "bookmarks to %3")
                         .arg(successCount)
                         .arg(selections.size())
                         .arg(outputDir));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onAddAnnotationClicked() {
    if (!m_selectionManager || m_selectionData.isEmpty()) {
        return;
    }

    cvSelectionAnnotationManager* annotations =
            m_selectionManager->getAnnotations();
    if (!annotations) {
        return;
    }

    // Show input dialog for annotation text
    bool ok;
    QString text = QInputDialog::getText(this, tr("Add Annotation"),
                                         tr("Annotation text:"),
                                         QLineEdit::Normal, QString(), &ok);

    if (ok && !text.isEmpty()) {
        QString id = annotations->addAnnotation(m_selectionData, text, true);
        if (!id.isEmpty()) {
            emit annotationRequested(text);
            CVLog::Print(QString("[cvSelectionPropertiesWidget] Added "
                                 "annotation: %1")
                                 .arg(id));
        }
    }
}
