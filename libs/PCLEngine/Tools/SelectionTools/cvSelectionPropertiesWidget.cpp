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
#include <QDateTime>
#include <QDialog>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QTabWidget>
#include <QTimer>

// STL
#include <cmath>

// QCustomPlot (PCLEngine uses its own copy)
#include <Tools/Common/qcustomplot.h>

// VTK
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
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
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QToolButton>
#include <QVBoxLayout>

// ParaView-style selection colors palette
const QColor cvSelectionPropertiesWidget::s_selectionColors[] = {
    QColor(255, 0, 0),      // Red
    QColor(0, 255, 0),      // Green
    QColor(0, 0, 255),      // Blue
    QColor(255, 255, 0),    // Yellow
    QColor(255, 0, 255),    // Magenta
    QColor(0, 255, 255),    // Cyan
    QColor(255, 128, 0),    // Orange
    QColor(128, 0, 255),    // Purple
    QColor(0, 255, 128),    // Spring Green
    QColor(255, 0, 128),    // Rose
};
const int cvSelectionPropertiesWidget::s_selectionColorsCount = 10;

//-----------------------------------------------------------------------------
cvSelectionPropertiesWidget::cvSelectionPropertiesWidget(QWidget* parent)
    : QWidget(parent),
      cvSelectionBase(),
      m_highlighter(nullptr),
      m_tooltipHelper(new cvSelectionTooltipHelper()),
      m_selectionManager(nullptr),
      m_selectionCount(0),
      m_volume(0.0),
      m_selectionNameCounter(0),
      m_lastHighlightedId(-1),
      m_exportToMeshButton(nullptr),
      m_exportToPointCloudButton(nullptr),
      m_exportToFileButton(nullptr),
      m_copyIDsButton(nullptr),
      m_freezeButton(nullptr),
      m_extractButton(nullptr),
      m_plotOverTimeButton(nullptr),
      m_hoverColorButton(nullptr),
      m_preselectedColorButton(nullptr),
      m_selectedColorButton(nullptr),
      m_boundaryColorButton(nullptr),
      m_hoverOpacitySpin(nullptr),
      m_preselectedOpacitySpin(nullptr),
      m_selectedOpacitySpin(nullptr),
      m_boundaryOpacitySpin(nullptr) {
    // Initialize saved preselected color (yellow by default)
    m_savedPreselectedColor[0] = 1.0;
    m_savedPreselectedColor[1] = 1.0;
    m_savedPreselectedColor[2] = 0.0;
    // Initialize colors (matching cvSelectionHighlighter defaults)
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

    // ParaView-style selection colors
    m_selectionColor = QColor(255, 0, 255);           // Magenta (like ParaView)
    m_interactiveSelectionColor = QColor(0, 255, 0);  // Green

    for (int i = 0; i < 6; ++i) {
        m_bounds[i] = 0.0;
    }
    for (int i = 0; i < 3; ++i) {
        m_center[i] = 0.0;
    }

    setupUi();
    
    // Set size policy to expand and fill available space
    // This is especially important when the widget is displayed alone (no DB object selected)
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    CVLog::PrintDebug("[cvSelectionPropertiesWidget] Initialized with ParaView-style UI");
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

    // Sync label properties from highlighter
    m_labelProperties.opacity = selectedOpacity;
    m_labelProperties.pointSize = m_highlighter->getPointSize(cvSelectionHighlighter::SELECTED);
    m_labelProperties.lineWidth = m_highlighter->getLineWidth(cvSelectionHighlighter::SELECTED);
    
    m_interactiveLabelProperties.opacity = hoverOpacity;
    m_interactiveLabelProperties.pointSize = m_highlighter->getPointSize(cvSelectionHighlighter::HOVER);
    m_interactiveLabelProperties.lineWidth = m_highlighter->getLineWidth(cvSelectionHighlighter::HOVER);

    // Sync ParaView-style selection colors
    if (selectedColor) {
        m_selectionColor = QColor::fromRgbF(selectedColor[0], selectedColor[1], selectedColor[2]);
        if (m_selectionColorButton) {
            m_selectionColorButton->setStyleSheet(
                QString("QPushButton { background-color: %1; color: white; }")
                    .arg(m_selectionColor.name()));
        }
    }
    if (hoverColor) {
        m_interactiveSelectionColor = QColor::fromRgbF(hoverColor[0], hoverColor[1], hoverColor[2]);
        if (m_interactiveSelectionColorButton) {
            m_interactiveSelectionColorButton->setStyleSheet(
                QString("QPushButton { background-color: %1; color: white; }")
                    .arg(m_interactiveSelectionColor.name()));
        }
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

    // Create scroll area for ParaView-style layout
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    m_scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(m_scrollContent);
    scrollLayout->setContentsMargins(5, 5, 5, 5);
    scrollLayout->setSpacing(10);

    // === ParaView-style sections ===
    setupSelectedDataHeader();
    scrollLayout->addWidget(m_selectedDataLabel);

    // Action buttons row (Freeze | Extract | Plot Over Time)
    QHBoxLayout* actionLayout = new QHBoxLayout();
    actionLayout->setSpacing(5);
    actionLayout->addWidget(m_freezeButton);
    actionLayout->addWidget(m_extractButton);
    actionLayout->addWidget(m_plotOverTimeButton);
    actionLayout->addStretch();
    scrollLayout->addLayout(actionLayout);

    setupSelectionDisplaySection();
    scrollLayout->addWidget(m_selectionDisplayGroup);

    setupSelectionEditorSection();
    scrollLayout->addWidget(m_selectionEditorGroup);

    setupSelectedDataSpreadsheet();
    scrollLayout->addWidget(m_selectedDataGroup);

    // === Tab Widget with all features (Statistics, Export, Advanced) ===
    m_tabWidget = new QTabWidget();
    m_tabWidget->setVisible(true);  // Always visible to provide access to all features
    setupStatisticsTab();
    setupExportTab();
    setupAdvancedTab();
    scrollLayout->addWidget(m_tabWidget);

    scrollLayout->addStretch();

    m_scrollContent->setLayout(scrollLayout);
    m_scrollArea->setWidget(m_scrollContent);
    mainLayout->addWidget(m_scrollArea);

    setLayout(mainLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectedDataHeader() {
    // Selected Data header label
    m_selectedDataLabel = new QLabel(tr("<b>Selected Data</b>"));
    m_selectedDataLabel->setStyleSheet(
        "QLabel { background-color: #e0e0e0; padding: 5px; border-radius: 3px; }");

    // Action buttons (ParaView-style with icons)
    
    // Freeze button (ParaView: converts selection to a frozen representation)
    m_freezeButton = new QPushButton(QIcon(":/Resources/images/svg/pqLock.png"), tr("Freeze"));
    m_freezeButton->setToolTip(tr("Freeze the current selection (convert to independent dataset)"));
    m_freezeButton->setFixedHeight(25);
    m_freezeButton->setEnabled(false);  // Enabled when selection exists
    connect(m_freezeButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onFreezeClicked);

    // Extract button (ParaView: pqExtractSelection.png/svg - creates new object from selection)
    QIcon extractIcon(":/Resources/images/svg/pqExtractSelection.png");
    if (extractIcon.isNull()) {
        extractIcon = QIcon(":/Resources/images/exportCloud.png");  // Fallback
    }
    m_extractButton = new QPushButton(extractIcon, tr("Extract"));
    m_extractButton->setToolTip(tr("Extract selected elements to a new dataset and add to scene"));
    m_extractButton->setFixedHeight(25);
    m_extractButton->setEnabled(false);  // Enabled when selection exists
    connect(m_extractButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExtractClicked);

    // Plot Over Time button (ParaView: pqPlotSelectionOverTime.png)
    // For CloudViewer: shows attribute distribution/histogram instead of time-series
    QIcon plotIcon(":/Resources/images/svg/pqPlotSelectionOverTime.png");
    m_plotOverTimeButton = new QPushButton(plotIcon, tr("Plot Distribution"));
    m_plotOverTimeButton->setToolTip(tr("Show distribution plots of selection attributes"));
    m_plotOverTimeButton->setFixedHeight(25);
    m_plotOverTimeButton->setEnabled(false);  // Enabled when selection exists
    connect(m_plotOverTimeButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onPlotOverTimeClicked);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectionDisplaySection() {
    m_selectionDisplayGroup = new QGroupBox(tr("Selection Display"));
    QVBoxLayout* displayLayout = new QVBoxLayout();
    displayLayout->setSpacing(0);  // ParaView-style: compact spacing

    // === Selection Labels === (ParaView-style header with separator line)
    QVBoxLayout* labelsHeaderLayout = new QVBoxLayout();
    labelsHeaderLayout->setSpacing(0);
    QLabel* labelsHeader = new QLabel(tr("<html><body><p><span style=\"font-weight:600;\">Selection Labels</span></p></body></html>"));
    labelsHeaderLayout->addWidget(labelsHeader);
    QFrame* labelsSeparator = new QFrame();
    labelsSeparator->setFrameShape(QFrame::HLine);
    labelsSeparator->setFrameShadow(QFrame::Sunken);
    labelsHeaderLayout->addWidget(labelsSeparator);
    displayLayout->addLayout(labelsHeaderLayout);

    // Cell Labels and Point Labels buttons (ParaView-style: horizontal, spacing=2)
    QHBoxLayout* labelsLayout = new QHBoxLayout();
    labelsLayout->setSpacing(2);

    // Cell Labels button with dropdown menu (ParaView: pqCellData.svg icon)
    m_cellLabelsButton = new QPushButton(tr("Cell Labels"));
    m_cellLabelsButton->setIcon(QIcon(":/Resources/images/svg/pqCellData.svg"));
    m_cellLabelsButton->setToolTip(tr("Set the array to label selected cells with"));
    m_cellLabelsMenu = new QMenu(this);
    m_cellLabelsMenu->addAction(tr("None"));
    m_cellLabelsMenu->addAction(tr("Cell ID"));
    m_cellLabelsMenu->addSeparator();
    m_cellLabelsButton->setMenu(m_cellLabelsMenu);
    connect(m_cellLabelsButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onCellLabelsClicked);
    labelsLayout->addWidget(m_cellLabelsButton);

    // Point Labels button with dropdown menu (ParaView: pqPointData.svg icon)
    m_pointLabelsButton = new QPushButton(tr("Point Labels"));
    m_pointLabelsButton->setIcon(QIcon(":/Resources/images/svg/pqPointData.svg"));
    m_pointLabelsButton->setToolTip(tr("Set the array to label selected points with"));
    m_pointLabelsMenu = new QMenu(this);
    m_pointLabelsMenu->addAction(tr("None"));
    m_pointLabelsMenu->addAction(tr("Point ID"));
    m_pointLabelsMenu->addSeparator();
    m_pointLabelsButton->setMenu(m_pointLabelsMenu);
    connect(m_pointLabelsButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onPointLabelsClicked);
    labelsLayout->addWidget(m_pointLabelsButton);

    displayLayout->addLayout(labelsLayout);

    // Edit Label Properties button (ParaView: pqAdvanced.svg icon)
    m_editLabelPropertiesButton = new QPushButton(tr("Edit Label Properties"));
    m_editLabelPropertiesButton->setIcon(QIcon(":/Resources/images/svg/pqAdvanced.png"));
    m_editLabelPropertiesButton->setToolTip(tr("Edit selection label properties"));
    connect(m_editLabelPropertiesButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onEditLabelPropertiesClicked);
    displayLayout->addWidget(m_editLabelPropertiesButton);

    // === Selection Appearance === (ParaView-style header with separator line)
    QVBoxLayout* appearanceHeaderLayout = new QVBoxLayout();
    appearanceHeaderLayout->setSpacing(0);
    QLabel* appearanceHeader = new QLabel(tr("<html><body><p><span style=\"font-weight:600;\">Selection Appearance</span></p></body></html>"));
    appearanceHeaderLayout->addWidget(appearanceHeader);
    QFrame* appearanceSeparator = new QFrame();
    appearanceSeparator->setFrameShape(QFrame::HLine);
    appearanceSeparator->setFrameShadow(QFrame::Sunken);
    appearanceHeaderLayout->addWidget(appearanceSeparator);
    displayLayout->addLayout(appearanceHeaderLayout);

    // Selection Color button (ParaView: pqColorChooserButton style)
    m_selectionColorButton = new QPushButton(tr("Selection Color"));
    m_selectionColorButton->setToolTip(tr("Set the color to use to show selected elements"));
    m_selectionColorButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    m_selectionColorButton->setStyleSheet(
        QString("QPushButton { background-color: %1; color: white; border: 1px solid gray; padding: 4px; }")
            .arg(m_selectionColor.name()));
    connect(m_selectionColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onSelectionColorClicked);
    displayLayout->addWidget(m_selectionColorButton);

    // === Interactive Selection === (ParaView-style header with separator line)
    QVBoxLayout* interactiveHeaderLayout = new QVBoxLayout();
    interactiveHeaderLayout->setSpacing(0);
    QLabel* interactiveHeader = new QLabel(tr("<html><body><p><span style=\"font-weight:600;\">Interactive Selection</span></p></body></html>"));
    interactiveHeaderLayout->addWidget(interactiveHeader);
    QFrame* interactiveSeparator = new QFrame();
    interactiveSeparator->setFrameShape(QFrame::HLine);
    interactiveSeparator->setFrameShadow(QFrame::Sunken);
    interactiveHeaderLayout->addWidget(interactiveSeparator);
    displayLayout->addLayout(interactiveHeaderLayout);

    // Interactive Selection Color button (ParaView: pqColorChooserButton style)
    m_interactiveSelectionColorButton = new QPushButton(tr("Interactive Selection Color"));
    m_interactiveSelectionColorButton->setToolTip(tr("Set the color to use to show selected elements during interaction"));
    m_interactiveSelectionColorButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    m_interactiveSelectionColorButton->setStyleSheet(
        QString("QPushButton { background-color: %1; color: white; border: 1px solid gray; padding: 4px; }")
            .arg(m_interactiveSelectionColor.name()));
    connect(m_interactiveSelectionColorButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onInteractiveSelectionColorClicked);
    displayLayout->addWidget(m_interactiveSelectionColorButton);

    // Edit Interactive Label Properties button (ParaView: pqAdvanced.svg icon)
    m_editInteractiveLabelPropertiesButton = new QPushButton(tr("Edit Interactive Label Properties"));
    m_editInteractiveLabelPropertiesButton->setIcon(QIcon(":/Resources/images/svg/pqAdvanced.png"));
    m_editInteractiveLabelPropertiesButton->setToolTip(tr("Edit interactive selection label properties"));
    connect(m_editInteractiveLabelPropertiesButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onEditInteractiveLabelPropertiesClicked);
    displayLayout->addWidget(m_editInteractiveLabelPropertiesButton);
    
    // Add vertical spacer at bottom (ParaView-style)
    displayLayout->addStretch();

    m_selectionDisplayGroup->setLayout(displayLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectionEditorSection() {
    m_selectionEditorGroup = new QGroupBox(tr("Selection Editor"));
    QVBoxLayout* editorLayout = new QVBoxLayout();
    editorLayout->setSpacing(5);  // ParaView-style spacing

    // Data Producer row (ParaView-style: spacing=5)
    QHBoxLayout* producerLayout = new QHBoxLayout();
    producerLayout->setSpacing(5);
    m_dataProducerLabel = new QLabel(tr("Data Producer"));
    m_dataProducerLabel->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    m_dataProducerLabel->setToolTip(tr("The dataset for which selections are saved"));
    m_dataProducerValue = new QLabel();
    m_dataProducerValue->setStyleSheet(
        "QLabel { background-color: snow; border: 1px inset grey; }");  // ParaView exact style
    m_dataProducerValue->setToolTip(tr("The dataset for which selections are saved"));
    m_dataProducerValue->setText(m_dataProducerName.isEmpty() ? QString() : m_dataProducerName);
    producerLayout->addWidget(m_dataProducerLabel);
    producerLayout->addWidget(m_dataProducerValue, 1);
    editorLayout->addLayout(producerLayout);

    // Element Type row (ParaView-style: spacing=9)
    QHBoxLayout* elementTypeLayout = new QHBoxLayout();
    elementTypeLayout->setSpacing(9);
    m_elementTypeLabel = new QLabel(tr("Element Type"));
    m_elementTypeLabel->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    m_elementTypeLabel->setToolTip(tr("The element type of the saved selections"));
    m_elementTypeValue = new QLabel();
    m_elementTypeValue->setStyleSheet(
        "QLabel { background-color: snow; border: 1px inset grey; }");  // ParaView exact style
    m_elementTypeValue->setToolTip(tr("The element type of the saved selections"));
    elementTypeLayout->addWidget(m_elementTypeLabel);
    elementTypeLayout->addWidget(m_elementTypeValue, 1);
    editorLayout->addLayout(elementTypeLayout);

    // Expression row (ParaView-style: spacing=29)
    QHBoxLayout* expressionLayout = new QHBoxLayout();
    expressionLayout->setSpacing(29);
    m_expressionLabel = new QLabel(tr("Expression"));
    m_expressionLabel->setToolTip(tr("Specify the expression which defines the relation between "
                                     "saved selections using boolean operators: !(NOT), &(AND), |(OR), ^(XOR) and ()."));
    m_expressionEdit = new QLineEdit();
    m_expressionEdit->setPlaceholderText(tr("e.g., (s0|s1)&s2|(s3&s4)|s5|s6|s7"));
    m_expressionEdit->setToolTip(tr("Specify the expression which defines the relation between "
                                    "saved selections using boolean operators: !(NOT), &(AND), |(OR), ^(XOR) and ()."));
    connect(m_expressionEdit, &QLineEdit::textChanged, this,
            &cvSelectionPropertiesWidget::onExpressionChanged);
    expressionLayout->addWidget(m_expressionLabel);
    expressionLayout->addWidget(m_expressionEdit, 1);
    editorLayout->addLayout(expressionLayout);

    // Selection table with toolbar (ParaView-style: ScrollArea with HBox)
    QHBoxLayout* tableLayout = new QHBoxLayout();
    tableLayout->setContentsMargins(0, 0, 0, 0);

    // Table: Name, Type, Color columns (ParaView: pqExpandableTableView)
    m_selectionEditorTable = new QTableWidget();
    m_selectionEditorTable->setColumnCount(3);
    m_selectionEditorTable->setHorizontalHeaderLabels({tr("Name"), tr("Type"), tr("Color")});
    m_selectionEditorTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_selectionEditorTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_selectionEditorTable->setAlternatingRowColors(true);
    m_selectionEditorTable->horizontalHeader()->setStretchLastSection(true);
    m_selectionEditorTable->verticalHeader()->setVisible(false);
    m_selectionEditorTable->setMinimumHeight(120);
    connect(m_selectionEditorTable, &QTableWidget::itemSelectionChanged, this,
            &cvSelectionPropertiesWidget::onSelectionEditorTableSelectionChanged);
    tableLayout->addWidget(m_selectionEditorTable);

    // Toolbar (vertical) - ParaView-style icons
    QVBoxLayout* toolbarLayout = new QVBoxLayout();
    toolbarLayout->setSpacing(0);

    // Add active selection button (ParaView: pqPlus.svg/png)
    m_addSelectionButton = new QToolButton();
    // Try SVG first, fallback to PNG if not available
    QIcon addIcon(":/Resources/images/svg/pqPlus.svg");
    if (addIcon.isNull()) {
        addIcon = QIcon(":/Resources/images/svg/pqPlus.png");
    }
    if (addIcon.isNull()) {
        addIcon = QIcon(":/Resources/images/ecvPlus.png"); // Final fallback
    }
    m_addSelectionButton->setIcon(addIcon);
    m_addSelectionButton->setToolTip(tr("Add active selection"));
    m_addSelectionButton->setIconSize(QSize(16, 16));  // Ensure visible size
    connect(m_addSelectionButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onAddActiveSelectionClicked);
    toolbarLayout->addWidget(m_addSelectionButton);

    // Remove selected selection button (ParaView: pqMinus.svg/png)
    m_removeSelectionButton = new QToolButton();
    QIcon removeIcon(":/Resources/images/svg/pqMinus.svg");
    if (removeIcon.isNull()) {
        removeIcon = QIcon(":/Resources/images/ecvMinus.png"); // Fallback
    }
    m_removeSelectionButton->setIcon(removeIcon);
    m_removeSelectionButton->setToolTip(tr("Remove selected selection from the saved selections. Remember to edit the Expression."));
    m_removeSelectionButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_removeSelectionButton->setEnabled(false);
    connect(m_removeSelectionButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onRemoveSelectedSelectionClicked);
    toolbarLayout->addWidget(m_removeSelectionButton);

    // Vertical spacer between buttons (ParaView-style)
    toolbarLayout->addStretch();

    // Remove all selections button (ParaView: pqDelete.svg - using smallTrash.png as alternative)
    m_removeAllSelectionsButton = new QToolButton();
    QIcon trashIcon(":/Resources/images/smallTrash.png");
    if (trashIcon.isNull()) {
        trashIcon = QIcon(":/Resources/images/ecvdelete.png"); // Fallback
    }
    m_removeAllSelectionsButton->setIcon(trashIcon);
    m_removeAllSelectionsButton->setToolTip(tr("Remove all saved selections"));
    m_removeAllSelectionsButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_removeAllSelectionsButton->setEnabled(false);
    connect(m_removeAllSelectionsButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onRemoveAllSelectionsClicked);
    toolbarLayout->addWidget(m_removeAllSelectionsButton);

    tableLayout->addLayout(toolbarLayout);
    editorLayout->addLayout(tableLayout);

    // Activate Combined Selections button (ParaView: pqApply.svg icon)
    QHBoxLayout* activateLayout = new QHBoxLayout();
    activateLayout->setSpacing(2);
    m_activateCombinedSelectionsButton = new QPushButton(tr("Activate Combined Selections"));
    m_activateCombinedSelectionsButton->setIcon(QIcon(":/Resources/images/smallValidate.png"));
    m_activateCombinedSelectionsButton->setToolTip(tr("Set the combined saved selections as the active selection"));
    m_activateCombinedSelectionsButton->setFocusPolicy(Qt::TabFocus);
    m_activateCombinedSelectionsButton->setDefault(true);
    m_activateCombinedSelectionsButton->setEnabled(false);
    connect(m_activateCombinedSelectionsButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onActivateCombinedSelectionsClicked);
    activateLayout->addWidget(m_activateCombinedSelectionsButton);
    editorLayout->addLayout(activateLayout);

    m_selectionEditorGroup->setLayout(editorLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectedDataSpreadsheet() {
    m_selectedDataGroup = new QGroupBox(tr("Selected Data"));
    QGridLayout* dataLayout = new QGridLayout();  // ParaView uses QGridLayout
    dataLayout->setSpacing(3);  // ParaView: spacing=3

    // Row 0: Attribute label, combo, buttons, checkbox (ParaView: columnstretch="0,1,0")
    // Column 0: Attribute label
    QLabel* attributeLabel = new QLabel(tr("<html><body><p><span style=\"font-weight:600;\">Attribute:</span></p></body></html>"));
    attributeLabel->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    dataLayout->addWidget(attributeLabel, 0, 0);

    // Column 1: Attribute combo box (ParaView-style with icons)
    m_attributeTypeCombo = new QComboBox();
    
    // Point Data with icon (ParaView: pqPointData.svg)
    QIcon pointDataIcon(":/Resources/images/svg/pqPointData.svg");
    if (pointDataIcon.isNull()) {
        pointDataIcon = QIcon(":/Resources/images/svg/pqPointData.png");
    }
    m_attributeTypeCombo->addItem(pointDataIcon, tr("Point Data"), 0);
    
    // Cell Data with icon (ParaView: pqCellData.svg)
    QIcon cellDataIcon(":/Resources/images/svg/pqCellData.svg");
    if (cellDataIcon.isNull()) {
        // Fallback: use a different icon if pqCellData is not available
        cellDataIcon = QIcon(":/Resources/images/svg/pqCellData.png");
    }
    m_attributeTypeCombo->addItem(cellDataIcon, tr("Cell Data"), 1);
    
    m_attributeTypeCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_attributeTypeCombo->setIconSize(QSize(16, 16));  // Ensure icons are visible
    connect(m_attributeTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &cvSelectionPropertiesWidget::onAttributeTypeChanged);
    dataLayout->addWidget(m_attributeTypeCombo, 0, 1);

    // Column 2: Toggle column visibility button (ParaView: pqRectilinearGrid16.png)
    m_toggleColumnVisibilityButton = new QToolButton();
    QIcon colVisIcon(":/Resources/images/interactors.png");
    if (colVisIcon.isNull()) {
        colVisIcon = QIcon(":/Resources/images/settings.png"); // Fallback
    }
    m_toggleColumnVisibilityButton->setIcon(colVisIcon);
    m_toggleColumnVisibilityButton->setToolTip(tr("Toggle column visibility"));
    m_toggleColumnVisibilityButton->setStatusTip(tr("Toggle column visibility"));
    m_toggleColumnVisibilityButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_toggleColumnVisibilityButton->setPopupMode(QToolButton::InstantPopup);
    connect(m_toggleColumnVisibilityButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onToggleColumnVisibility);
    dataLayout->addWidget(m_toggleColumnVisibilityButton, 0, 2);

    // Column 3: Toggle field data button (ParaView: pqGlobalData.svg)
    m_toggleFieldDataButton = new QToolButton();
    QIcon fieldDataIcon(":/Resources/images/svg/pqSolidColor.png");
    if (fieldDataIcon.isNull()) {
        fieldDataIcon = QIcon(":/Resources/images/color.png"); // Fallback
    }
    m_toggleFieldDataButton->setIcon(fieldDataIcon);
    m_toggleFieldDataButton->setToolTip(tr("Toggle field data visibility"));
    m_toggleFieldDataButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_toggleFieldDataButton->setCheckable(true);
    dataLayout->addWidget(m_toggleFieldDataButton, 0, 3);

    // Column 4: Invert selection checkbox
    m_invertSelectionCheck = new QCheckBox(tr("Invert selection"));
    m_invertSelectionCheck->setToolTip(tr("Invert the selection"));
    m_invertSelectionCheck->setEnabled(false);  // ParaView: enabled=false by default
    connect(m_invertSelectionCheck, &QCheckBox::toggled, this,
            &cvSelectionPropertiesWidget::onInvertSelectionToggled);
    dataLayout->addWidget(m_invertSelectionCheck, 0, 4);

    // Row 2: Spreadsheet table (ParaView: pqSpreadSheetViewWidget, spans all 5 columns)
    m_spreadsheetTable = new QTableWidget();
    m_spreadsheetTable->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
    m_spreadsheetTable->setMinimumHeight(120);  // ParaView: minimumHeight=120
    m_spreadsheetTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_spreadsheetTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_spreadsheetTable->setAlternatingRowColors(true);
    m_spreadsheetTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_spreadsheetTable->horizontalHeader()->setStretchLastSection(true);
    m_spreadsheetTable->verticalHeader()->setDefaultSectionSize(20);
    connect(m_spreadsheetTable, &QTableWidget::itemClicked, this,
            &cvSelectionPropertiesWidget::onSpreadsheetItemClicked);
    dataLayout->addWidget(m_spreadsheetTable, 2, 0, 1, 5);  // Row 2, spanning all 5 columns

    m_selectedDataGroup->setLayout(dataLayout);
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

    // === Selection Data Table (ParaView-style) ===
    QGroupBox* listGroup = new QGroupBox(tr("Selected Data"));
    QVBoxLayout* listLayout = new QVBoxLayout();

    m_listInfoLabel = new QLabel(tr("No selection"));
    m_listInfoLabel->setStyleSheet("font-weight: bold;");
    listLayout->addWidget(m_listInfoLabel);

    // ParaView-style table with columns for ID and coordinates/attributes
    m_selectionTableWidget = new QTableWidget();
    m_selectionTableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_selectionTableWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_selectionTableWidget->setAlternatingRowColors(true);
    m_selectionTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_selectionTableWidget->horizontalHeader()->setStretchLastSection(true);
    m_selectionTableWidget->verticalHeader()->setDefaultSectionSize(20);
    m_selectionTableWidget->setMinimumHeight(150);
    m_selectionTableWidget->setMaximumHeight(250);
    connect(m_selectionTableWidget, &QTableWidget::itemClicked, this,
            &cvSelectionPropertiesWidget::onSelectionTableItemClicked);
    listLayout->addWidget(m_selectionTableWidget);

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

    // Note: Export buttons are created in setupUi() and added to main interface
    // Only create them here if they don't exist (for backwards compatibility)
    if (!m_exportToMeshButton) {
        m_exportToMeshButton = new QPushButton(tr("Export to Mesh"));
        m_exportToMeshButton->setEnabled(false);
        m_exportToMeshButton->setToolTip(
                tr("Create a ccMesh from the selected cells and add to the scene"));
        connect(m_exportToMeshButton, &QPushButton::clicked, this,
                &cvSelectionPropertiesWidget::onExportToMeshClicked);
    }
    exportLayout->addWidget(m_exportToMeshButton);

    if (!m_exportToPointCloudButton) {
        m_exportToPointCloudButton = new QPushButton(tr("Export to Point Cloud"));
        m_exportToPointCloudButton->setEnabled(false);
        m_exportToPointCloudButton->setToolTip(
                tr("Create a ccPointCloud from the selected points and add to the "
                   "scene"));
        connect(m_exportToPointCloudButton, &QPushButton::clicked, this,
                &cvSelectionPropertiesWidget::onExportToPointCloudClicked);
    }
    exportLayout->addWidget(m_exportToPointCloudButton);

    exportLayout->addSpacing(10);

    if (!m_exportToFileButton) {
        m_exportToFileButton = new QPushButton(tr("Export to File..."));
        m_exportToFileButton->setEnabled(false);
        m_exportToFileButton->setToolTip(
                tr("Export selection to a file in the selected format"));
        connect(m_exportToFileButton, &QPushButton::clicked, this,
                &cvSelectionPropertiesWidget::onExportToFileClicked);
    }
    exportLayout->addWidget(m_exportToFileButton);

    exportLayout->addSpacing(10);

    if (!m_copyIDsButton) {
        m_copyIDsButton = new QPushButton(tr("Copy IDs to Clipboard"));
        m_copyIDsButton->setEnabled(false);
        m_copyIDsButton->setToolTip(tr("Copy selected IDs to the clipboard"));
        connect(m_copyIDsButton, &QPushButton::clicked, this,
                &cvSelectionPropertiesWidget::onCopyIDsClicked);
    }
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
    
    // Always use the manager's shared highlighter
    // This ensures all tools (including tooltip tools) share the same highlighter
    // so color settings are automatically synchronized
    if (m_selectionManager) {
        cvSelectionHighlighter* sharedHighlighter = m_selectionManager->getHighlighter();
        if (sharedHighlighter && sharedHighlighter != m_highlighter) {
            setHighlighter(sharedHighlighter);
        }
        
        // Initialize default label properties for annotations
        cvSelectionAnnotationManager* annotations = m_selectionManager->getAnnotations();
        if (annotations) {
            annotations->setDefaultLabelProperties(m_labelProperties, true);  // cell labels
            annotations->setDefaultLabelProperties(m_interactiveLabelProperties, false);  // point labels
        }
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
        // First try from selection manager (most reliable source)
        cvViewSelectionManager* manager = cvViewSelectionManager::instance();
        if (manager) {
            polyData = manager->getPolyData();
        }
        
        // Fallback to getPolyDataForSelection
        if (!polyData) {
            polyData = getPolyDataForSelection(&m_selectionData);
        }
    }

    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No polyData available for "
                "statistics");
        return false;
    }
    
    // Validate polyData before using
    try {
        vtkIdType numPoints = polyData->GetNumberOfPoints();
        vtkIdType numCells = polyData->GetNumberOfCells();
        if (numPoints < 0 || numCells < 0) {
            CVLog::Warning("[cvSelectionPropertiesWidget] Invalid polyData");
            return false;
        }
    } catch (...) {
        CVLog::Warning("[cvSelectionPropertiesWidget] polyData validation failed");
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
    
    // Enable header action buttons (ParaView-style)
    m_freezeButton->setEnabled(m_selectionCount > 0);
    m_extractButton->setEnabled(m_selectionCount > 0);
    m_plotOverTimeButton->setEnabled(m_selectionCount > 0);  // Show distribution plots

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

    // Clear table
    m_selectionTableWidget->clear();
    m_selectionTableWidget->setRowCount(0);
    m_selectionTableWidget->setColumnCount(0);
    m_listInfoLabel->setText(tr("No selection"));
    m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");

    // Disable export buttons
    m_exportToMeshButton->setEnabled(false);
    m_exportToPointCloudButton->setEnabled(false);
    m_exportToFileButton->setEnabled(false);
    m_copyIDsButton->setEnabled(false);
    
    // Disable header action buttons
    m_freezeButton->setEnabled(false);
    m_extractButton->setEnabled(false);
    m_plotOverTimeButton->setEnabled(false);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateStatistics(vtkPolyData* polyData) {
    if (!polyData) {
        CVLog::Warning("[cvSelectionPropertiesWidget::updateStatistics] polyData is nullptr");
        return;
    }
    
    // Validate polyData is still valid by checking basic properties
    // This helps catch use-after-free issues
    try {
        if (polyData->GetNumberOfPoints() < 0 || polyData->GetNumberOfCells() < 0) {
            CVLog::Warning("[cvSelectionPropertiesWidget::updateStatistics] Invalid polyData state");
            return;
        }
    } catch (...) {
        CVLog::Warning("[cvSelectionPropertiesWidget::updateStatistics] polyData access failed - possible dangling pointer");
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
    m_selectionTableWidget->clear();
    m_selectionTableWidget->setRowCount(0);

    QVector<qint64> ids = m_selectionData.ids();
    if (ids.isEmpty()) {
        m_selectionTableWidget->setColumnCount(0);
        m_listInfoLabel->setText(tr("No selection"));
        m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");
        return;
    }

    bool isPoints = (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);
    
    // Update info label (ParaView-style: "Selected Data (source.ply)")
    m_listInfoLabel->setText(
            tr("Showing %1 %2")
                    .arg(ids.size())
                    .arg(m_selectionData.fieldTypeString().toLower()));
    m_listInfoLabel->setStyleSheet("font-weight: bold;");

    // Setup columns based on selection type (ParaView-style)
    QStringList headers;
    if (isPoints) {
        headers << tr("Point ID") << tr("X") << tr("Y") << tr("Z");
        // Check for additional point attributes
        if (polyData && polyData->GetPointData()) {
            vtkPointData* pd = polyData->GetPointData();
            for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = pd->GetArray(i);
                if (arr && arr->GetName()) {
                    QString name = QString::fromUtf8(arr->GetName());
                    // Skip common coordinate arrays
                    if (name != "Points" && name != "Normals") {
                        headers << name;
                    }
                }
            }
        }
    } else {
        // Cells
        headers << tr("Cell ID") << tr("Type") << tr("Num Points");
        // Check for additional cell attributes
        if (polyData && polyData->GetCellData()) {
            vtkCellData* cd = polyData->GetCellData();
            for (int i = 0; i < cd->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = cd->GetArray(i);
                if (arr && arr->GetName()) {
                    headers << QString::fromUtf8(arr->GetName());
                }
            }
        }
    }
    
    m_selectionTableWidget->setColumnCount(headers.size());
    m_selectionTableWidget->setHorizontalHeaderLabels(headers);

    // Limit display for performance
    int maxDisplay = qMin(ids.size(), 500);
    m_selectionTableWidget->setRowCount(maxDisplay);

    for (int row = 0; row < maxDisplay; ++row) {
        qint64 id = ids[row];
        int col = 0;
        
        // ID column
        QTableWidgetItem* idItem = new QTableWidgetItem(QString::number(id));
        idItem->setData(Qt::UserRole, QVariant::fromValue(id));  // Store ID for click handling
        m_selectionTableWidget->setItem(row, col++, idItem);

        if (isPoints && polyData) {
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(id, pt);
                
                // X, Y, Z columns
                m_selectionTableWidget->setItem(row, col++, 
                    new QTableWidgetItem(QString::number(pt[0], 'g', 6)));
                m_selectionTableWidget->setItem(row, col++, 
                    new QTableWidgetItem(QString::number(pt[1], 'g', 6)));
                m_selectionTableWidget->setItem(row, col++, 
                    new QTableWidgetItem(QString::number(pt[2], 'g', 6)));
                
                // Additional attributes
                if (polyData->GetPointData()) {
                    vtkPointData* pd = polyData->GetPointData();
                    for (int i = 0; i < pd->GetNumberOfArrays(); ++i) {
                        vtkDataArray* arr = pd->GetArray(i);
                        if (arr && arr->GetName()) {
                            QString name = QString::fromUtf8(arr->GetName());
                            if (name != "Points" && name != "Normals") {
                                double val = arr->GetComponent(id, 0);
                                m_selectionTableWidget->setItem(row, col++,
                                    new QTableWidgetItem(QString::number(val, 'g', 6)));
                            }
                        }
                    }
                }
            }
        } else if (!isPoints && polyData) {
            // Cell data
            if (id >= 0 && id < polyData->GetNumberOfCells()) {
                vtkCell* cell = polyData->GetCell(id);
                if (cell) {
                    // Cell type
                    m_selectionTableWidget->setItem(row, col++,
                        new QTableWidgetItem(QString::number(cell->GetCellType())));
                    // Number of points
                    m_selectionTableWidget->setItem(row, col++,
                        new QTableWidgetItem(QString::number(cell->GetNumberOfPoints())));
                    
                    // Additional cell attributes
                    if (polyData->GetCellData()) {
                        vtkCellData* cd = polyData->GetCellData();
                        for (int i = 0; i < cd->GetNumberOfArrays(); ++i) {
                            vtkDataArray* arr = cd->GetArray(i);
                            if (arr && arr->GetName()) {
                                double val = arr->GetComponent(id, 0);
                                m_selectionTableWidget->setItem(row, col++,
                                    new QTableWidgetItem(QString::number(val, 'g', 6)));
                            }
                        }
                    }
                }
            }
        }
    }

    // Resize columns to contents
    m_selectionTableWidget->resizeColumnsToContents();
    
    // Update info if truncated
    if (ids.size() > maxDisplay) {
        m_listInfoLabel->setText(
            tr("Showing %1 of %2 %3")
                .arg(maxDisplay)
                .arg(ids.size())
                .arg(m_selectionData.fieldTypeString().toLower()));
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

void cvSelectionPropertiesWidget::onSelectionTableItemClicked(
        QTableWidgetItem* item) {
    if (!item) {
        return;
    }

    // Get the ID from the first column of the clicked row
    int row = item->row();
    QTableWidgetItem* idItem = m_selectionTableWidget->item(row, 0);
    if (!idItem) {
        return;
    }

    // Get stored ID from UserRole data
    QVariant idData = idItem->data(Qt::UserRole);
    if (!idData.isValid()) {
        return;
    }

    qint64 id = idData.toLongLong();

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

    // Determine the data type and validate ID
    bool isPointData = (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);
    QString dataType = isPointData ? tr("Point") : tr("Cell");
    
    if (isPointData) {
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

    // === ParaView-style: Use RED for interactive selection highlighting ===
    // Store original PRESELECTED color to restore later
    const double* originalColor = m_highlighter->getHighlightColor(
            cvSelectionHighlighter::PRESELECTED);
    double savedColor[3] = {originalColor[0], originalColor[1], originalColor[2]};
    
    // Set PRESELECTED mode to RED for emphasis (ParaView uses red for interactive selection)
    m_highlighter->setHighlightColor(1.0, 0.0, 0.0, cvSelectionHighlighter::PRESELECTED);
    
    // Highlight this single item with RED using PRESELECTED mode
    m_highlighter->highlightSelection(singleIdArray,
                                      m_selectionData.fieldAssociation(),
                                      cvSelectionHighlighter::PRESELECTED);
    
    // Store for later restoration
    m_savedPreselectedColor[0] = savedColor[0];
    m_savedPreselectedColor[1] = savedColor[1];
    m_savedPreselectedColor[2] = savedColor[2];
    m_lastHighlightedId = id;

    // Log the highlighted item with data type info
    if (isPointData) {
        double pt[3];
        polyData->GetPoint(id, pt);
        CVLog::Print(QString("[cvSelectionPropertiesWidget] RED highlight: %1 ID=%2 at (%3, %4, %5)")
                             .arg(dataType)
                             .arg(id)
                             .arg(pt[0], 0, 'f', 4)
                             .arg(pt[1], 0, 'f', 4)
                             .arg(pt[2], 0, 'f', 4));
    } else {
        vtkCell* cell = polyData->GetCell(id);
        if (cell) {
            double center[3] = {0, 0, 0};
            double* weights = new double[cell->GetNumberOfPoints()];
            double pcoords[3] = {0.5, 0.5, 0.5};
            int subId = 0;
            cell->EvaluateLocation(subId, pcoords, center, weights);
            delete[] weights;
            CVLog::Print(QString("[cvSelectionPropertiesWidget] RED highlight: %1 ID=%2 "
                                 "(Type:%3, Points:%4) center=(%5, %6, %7)")
                                 .arg(dataType)
                                 .arg(id)
                                 .arg(cell->GetCellType())
                                 .arg(cell->GetNumberOfPoints())
                                 .arg(center[0], 0, 'f', 4)
                                 .arg(center[1], 0, 'f', 4)
                                 .arg(center[2], 0, 'f', 4));
        }
    }

    // Refresh display immediately
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        pclVis->UpdateScreen();
    }

    // Use a timer to restore original selection after 3 seconds
    QTimer::singleShot(3000, this, [this]() {
        if (m_highlighter) {
            // Restore original PRESELECTED color
            m_highlighter->setHighlightColor(this->m_savedPreselectedColor[0], 
                                             this->m_savedPreselectedColor[1], 
                                             this->m_savedPreselectedColor[2],
                                             cvSelectionHighlighter::PRESELECTED);
            
            if (!m_selectionData.isEmpty()) {
                // Restore original full selection highlight with SELECTED mode (green)
                m_highlighter->highlightSelection(
                        m_selectionData.vtkArray(),
                        m_selectionData.fieldAssociation(),
                        cvSelectionHighlighter::SELECTED);
            }

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

// ============================================================================
// ParaView-style Selection Display slots
// ============================================================================

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onCellLabelsClicked() {
    // Dynamically populate the menu with available cell data arrays
    if (!m_cellLabelsMenu) return;
    
    m_cellLabelsMenu->clear();
    
    // Add default options
    QAction* noneAction = m_cellLabelsMenu->addAction(tr("None"));
    noneAction->setCheckable(true);
    noneAction->setChecked(m_currentCellLabelArray.isEmpty());
    connect(noneAction, &QAction::triggered, [this]() {
        m_currentCellLabelArray.clear();
        CVLog::PrintDebug("[cvSelectionPropertiesWidget] Cell labels disabled");
    });
    
    QAction* idAction = m_cellLabelsMenu->addAction(tr("Cell ID"));
    idAction->setCheckable(true);
    idAction->setChecked(m_currentCellLabelArray == "CellID");
    connect(idAction, &QAction::triggered, [this]() {
        m_currentCellLabelArray = "CellID";
        CVLog::PrintDebug("[cvSelectionPropertiesWidget] Cell labels set to Cell ID");
    });
    
    m_cellLabelsMenu->addSeparator();
    
    // Add available cell data arrays from current polyData
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (polyData && polyData->GetCellData()) {
        vtkCellData* cellData = polyData->GetCellData();
        for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
            vtkDataArray* arr = cellData->GetArray(i);
            if (arr && arr->GetName()) {
                QString name = QString::fromUtf8(arr->GetName());
                QAction* action = m_cellLabelsMenu->addAction(name);
                action->setCheckable(true);
                action->setChecked(m_currentCellLabelArray == name);
                connect(action, &QAction::triggered, [this, name]() {
                    m_currentCellLabelArray = name;
                    CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Cell labels set to %1").arg(name));
                });
            }
        }
    }
    
    m_cellLabelsMenu->exec(QCursor::pos());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onPointLabelsClicked() {
    // Dynamically populate the menu with available point data arrays
    if (!m_pointLabelsMenu) return;
    
    m_pointLabelsMenu->clear();
    
    // Add default options
    QAction* noneAction = m_pointLabelsMenu->addAction(tr("None"));
    noneAction->setCheckable(true);
    noneAction->setChecked(m_currentPointLabelArray.isEmpty());
    connect(noneAction, &QAction::triggered, [this]() {
        m_currentPointLabelArray.clear();
        CVLog::PrintDebug("[cvSelectionPropertiesWidget] Point labels disabled");
    });
    
    QAction* idAction = m_pointLabelsMenu->addAction(tr("Point ID"));
    idAction->setCheckable(true);
    idAction->setChecked(m_currentPointLabelArray == "PointID");
    connect(idAction, &QAction::triggered, [this]() {
        m_currentPointLabelArray = "PointID";
        CVLog::PrintDebug("[cvSelectionPropertiesWidget] Point labels set to Point ID");
    });
    
    m_pointLabelsMenu->addSeparator();
    
    // Add available point data arrays from current polyData
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (polyData && polyData->GetPointData()) {
        vtkPointData* pointData = polyData->GetPointData();
        for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
            vtkDataArray* arr = pointData->GetArray(i);
            if (arr && arr->GetName()) {
                QString name = QString::fromUtf8(arr->GetName());
                // Skip common coordinate arrays
                if (name != "Points" && name != "Normals") {
                    QAction* action = m_pointLabelsMenu->addAction(name);
                    action->setCheckable(true);
                    action->setChecked(m_currentPointLabelArray == name);
                    connect(action, &QAction::triggered, [this, name]() {
                        m_currentPointLabelArray = name;
                        CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Point labels set to %1").arg(name));
                    });
                }
            }
        }
    }
    
    m_pointLabelsMenu->exec(QCursor::pos());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onEditLabelPropertiesClicked() {
    cvSelectionLabelPropertiesDialog dialog(this, false);
    dialog.setProperties(m_labelProperties);
    connect(&dialog, &cvSelectionLabelPropertiesDialog::propertiesApplied,
            this, &cvSelectionPropertiesWidget::onLabelPropertiesApplied);
    dialog.exec();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionColorClicked() {
    QColor color = QColorDialog::getColor(m_selectionColor, this,
                                          tr("Select Selection Color"));
    if (color.isValid()) {
        m_selectionColor = color;
        m_selectionColorButton->setStyleSheet(
            QString("QPushButton { background-color: %1; color: white; }")
                .arg(color.name()));
        
        // Apply to highlighter (Selected mode)
        if (m_highlighter) {
            m_highlighter->setHighlightColor(color.redF(), color.greenF(), color.blueF(),
                                             cvSelectionHighlighter::SELECTED);
            PclUtils::PCLVis* pclVis = getPCLVis();
            if (pclVis) {
                pclVis->UpdateScreen();
            }
        }
        
        CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Selection color changed to %1")
                          .arg(color.name()));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onInteractiveSelectionColorClicked() {
    QColor color = QColorDialog::getColor(m_interactiveSelectionColor, this,
                                          tr("Select Interactive Selection Color"));
    if (color.isValid()) {
        m_interactiveSelectionColor = color;
        m_interactiveSelectionColorButton->setStyleSheet(
            QString("QPushButton { background-color: %1; color: white; }")
                .arg(color.name()));
        
        // Apply to highlighter (Hover mode for interactive)
        if (m_highlighter) {
            m_highlighter->setHighlightColor(color.redF(), color.greenF(), color.blueF(),
                                             cvSelectionHighlighter::HOVER);
            PclUtils::PCLVis* pclVis = getPCLVis();
            if (pclVis) {
                pclVis->UpdateScreen();
            }
        }
        
        CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Interactive selection color changed to %1")
                          .arg(color.name()));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onEditInteractiveLabelPropertiesClicked() {
    cvSelectionLabelPropertiesDialog dialog(this, true);
    dialog.setProperties(m_interactiveLabelProperties);
    connect(&dialog, &cvSelectionLabelPropertiesDialog::propertiesApplied,
            this, &cvSelectionPropertiesWidget::onInteractiveLabelPropertiesApplied);
    dialog.exec();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onLabelPropertiesApplied(
        const cvSelectionLabelPropertiesDialog::LabelProperties& props) {
    m_labelProperties = props;
    
    // Apply all properties to highlighter (SELECTED mode)
    if (m_highlighter) {
        // Apply opacity
        m_highlighter->setHighlightOpacity(props.opacity, cvSelectionHighlighter::SELECTED);
        
        // Apply point size and line width
        m_highlighter->setPointSize(props.pointSize, cvSelectionHighlighter::SELECTED);
        m_highlighter->setLineWidth(props.lineWidth, cvSelectionHighlighter::SELECTED);
        
        // Also apply to BOUNDARY mode (uses similar settings)
        m_highlighter->setHighlightOpacity(props.opacity, cvSelectionHighlighter::BOUNDARY);
        m_highlighter->setPointSize(props.pointSize, cvSelectionHighlighter::BOUNDARY);
        m_highlighter->setLineWidth(props.lineWidth, cvSelectionHighlighter::BOUNDARY);
        
        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }
    }
    
    // Apply font properties to annotations (cell labels)
    if (m_selectionManager) {
        cvSelectionAnnotationManager* annotations = m_selectionManager->getAnnotations();
        if (annotations) {
            // Set default properties for new annotations
            annotations->setDefaultLabelProperties(props, true);  // true = cell labels
            
            // Apply to all existing annotations
            annotations->applyLabelProperties(props, true);  // true = cell labels
        }
    }
    
    CVLog::Print(QString("[cvSelectionPropertiesWidget] Label properties applied: "
                         "opacity=%1, pointSize=%2, lineWidth=%3")
                 .arg(props.opacity)
                 .arg(props.pointSize)
                 .arg(props.lineWidth));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onInteractiveLabelPropertiesApplied(
        const cvSelectionLabelPropertiesDialog::LabelProperties& props) {
    m_interactiveLabelProperties = props;
    
    // Apply default properties to annotation manager for point labels (interactive)
    if (m_selectionManager) {
        cvSelectionAnnotationManager* annotations = m_selectionManager->getAnnotations();
        if (annotations) {
            // Set default properties for new annotations (point labels)
            annotations->setDefaultLabelProperties(props, false);  // false = point labels
            
            // Apply to all existing annotations (point labels)
            annotations->applyLabelProperties(props, false);  // false = point labels
        }
    }
    
    // Apply all properties to highlighter (HOVER and PRESELECTED modes)
    if (m_highlighter) {
        // Apply to HOVER mode
        m_highlighter->setHighlightOpacity(props.opacity, cvSelectionHighlighter::HOVER);
        m_highlighter->setPointSize(props.pointSize, cvSelectionHighlighter::HOVER);
        m_highlighter->setLineWidth(props.lineWidth, cvSelectionHighlighter::HOVER);
        
        // Also apply to PRESELECTED mode
        m_highlighter->setHighlightOpacity(props.opacity, cvSelectionHighlighter::PRESELECTED);
        m_highlighter->setPointSize(props.pointSize, cvSelectionHighlighter::PRESELECTED);
        m_highlighter->setLineWidth(props.lineWidth, cvSelectionHighlighter::PRESELECTED);
        
        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }
    }
    
    CVLog::Print(QString("[cvSelectionPropertiesWidget] Interactive label properties applied: "
                         "opacity=%1, pointSize=%2, lineWidth=%3")
                 .arg(props.opacity)
                 .arg(props.pointSize)
                 .arg(props.lineWidth));
}

// ============================================================================
// ParaView-style Selection Editor slots
// ============================================================================

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onExpressionChanged(const QString& text) {
    emit expressionChanged(text);
    
    // Update the activate button state
    m_activateCombinedSelectionsButton->setEnabled(
        !text.isEmpty() && !m_savedSelections.isEmpty());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onAddActiveSelectionClicked() {
    if (m_selectionData.isEmpty()) {
        QMessageBox::information(this, tr("Add Selection"),
                                 tr("No active selection to add."));
        return;
    }
    
    // Create new saved selection
    SavedSelection saved;
    saved.name = generateSelectionName();
    saved.type = tr("ID Selection");
    saved.color = generateSelectionColor();
    saved.data = m_selectionData;
    
    m_savedSelections.append(saved);
    updateSelectionEditorTable();
    
    // Update expression with new selection
    QString expr = m_expressionEdit->text();
    if (!expr.isEmpty()) {
        expr += "|";
    }
    expr += saved.name;
    m_expressionEdit->setText(expr);
    
    // Enable buttons
    m_removeAllSelectionsButton->setEnabled(true);
    m_activateCombinedSelectionsButton->setEnabled(!m_expressionEdit->text().isEmpty());
    
    emit selectionAdded(saved.data);
    
    CVLog::Print(QString("[cvSelectionPropertiesWidget] Added selection: %1")
                 .arg(saved.name));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onRemoveSelectedSelectionClicked() {
    QList<QTableWidgetItem*> selectedItems = m_selectionEditorTable->selectedItems();
    if (selectedItems.isEmpty()) {
        return;
    }
    
    // Get unique rows
    QSet<int> rows;
    for (QTableWidgetItem* item : selectedItems) {
        rows.insert(item->row());
    }
    
    // Remove in reverse order to maintain valid indices
    QList<int> sortedRows = rows.values();
    std::sort(sortedRows.begin(), sortedRows.end(), std::greater<int>());
    
    for (int row : sortedRows) {
        if (row >= 0 && row < m_savedSelections.size()) {
            QString name = m_savedSelections[row].name;
            m_savedSelections.removeAt(row);
            emit selectionRemoved(row);
            CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Removed selection: %1")
                              .arg(name));
        }
    }
    
    updateSelectionEditorTable();
    
    // Update button states
    m_removeAllSelectionsButton->setEnabled(!m_savedSelections.isEmpty());
    m_activateCombinedSelectionsButton->setEnabled(
        !m_expressionEdit->text().isEmpty() && !m_savedSelections.isEmpty());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onRemoveAllSelectionsClicked() {
    if (m_savedSelections.isEmpty()) {
        return;
    }
    
    int result = QMessageBox::question(this, tr("Remove All Selections"),
                                       tr("Remove all saved selections?"),
                                       QMessageBox::Yes | QMessageBox::No);
    if (result != QMessageBox::Yes) {
        return;
    }
    
    m_savedSelections.clear();
    m_selectionNameCounter = 0;
    m_expressionEdit->clear();
    updateSelectionEditorTable();
    
    // Update button states
    m_removeSelectionButton->setEnabled(false);
    m_removeAllSelectionsButton->setEnabled(false);
    m_activateCombinedSelectionsButton->setEnabled(false);
    
    emit allSelectionsRemoved();
    
    CVLog::Print("[cvSelectionPropertiesWidget] Removed all saved selections");
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onActivateCombinedSelectionsClicked() {
    // TODO: Implement expression evaluation and combine selections
    emit activateCombinedSelectionsRequested();
    
    CVLog::Print(QString("[cvSelectionPropertiesWidget] Activating combined selections with expression: %1")
                 .arg(m_expressionEdit->text()));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionEditorTableSelectionChanged() {
    bool hasSelection = !m_selectionEditorTable->selectedItems().isEmpty();
    m_removeSelectionButton->setEnabled(hasSelection);
}

// ============================================================================
// ParaView-style Find Data / Selected Data slots
// ============================================================================

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onAttributeTypeChanged(int index) {
    // Re-populate the spreadsheet based on selected attribute type
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (polyData) {
        updateSpreadsheetData(polyData);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onInvertSelectionToggled(bool checked) {
    emit invertSelectionRequested();
    
    CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Invert selection: %1")
                      .arg(checked ? "ON" : "OFF"));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onFreezeClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to freeze");
        return;
    }

    // Freeze selection: Create a static copy that won't change with new selections
    // In ParaView, this converts the selection to an "AppendSelection" filter
    // For CloudViewer, we save current selection to bookmarks with "Frozen_" prefix
    
    QString frozenName = QString("Frozen_%1").arg(
        QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    
    if (m_selectionManager && m_selectionManager->getBookmarks()) {
        m_selectionManager->getBookmarks()->addBookmark(frozenName, m_selectionData);
        updateBookmarkCombo();
        CVLog::Print(QString("[cvSelectionPropertiesWidget] Selection frozen as: %1").arg(frozenName));
        
        QMessageBox::information(this, tr("Freeze Selection"),
                               tr("Selection frozen as bookmark: %1\n"
                                  "Load it anytime from the Advanced tab.").arg(frozenName));
    } else {
        CVLog::Warning("[cvSelectionPropertiesWidget] Selection manager or bookmarks not available");
    }
    
    emit freezeSelectionRequested();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onExtractClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to extract");
        return;
    }

    // Extract selection: Create a new object from selected elements
    // This is equivalent to ParaView's "Extract Selection" filter
    // Creates a new ccPointCloud or ccMesh depending on selection type
    
    bool isCells = (m_selectionData.fieldAssociation() == cvSelectionData::CELLS);
    bool isPoints = (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);
    
    if (isCells) {
        // Extract as mesh
        onExportToMeshClicked();
        CVLog::Print("[cvSelectionPropertiesWidget] Extracted selection as Mesh");
    } else if (isPoints) {
        // Extract as point cloud
        onExportToPointCloudClicked();
        CVLog::Print("[cvSelectionPropertiesWidget] Extracted selection as Point Cloud");
    } else {
        CVLog::Warning("[cvSelectionPropertiesWidget] Unknown selection type for extraction");
    }
    
    emit extractSelectionRequested();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onPlotOverTimeClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to plot");
        return;
    }
    
    // Plot Over Time: For static data, show attribute distribution/histogram
    // In ParaView, this shows time-series plots; for CloudViewer, we show:
    // 1. Histogram of scalar field values in selection
    // 2. Distribution plots of various attributes
    
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No polydata available for plotting");
        return;
    }
    
    // Get current attribute type (Point Data or Cell Data)
    bool isPoints = (m_attributeTypeCombo->currentIndex() == 0);
    vtkDataSetAttributes* attributes = isPoints ? 
        static_cast<vtkDataSetAttributes*>(polyData->GetPointData()) :
        static_cast<vtkDataSetAttributes*>(polyData->GetCellData());
    
    if (!attributes) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No attributes available");
        return;
    }
    
    // Create a dialog to show distribution plots
    QDialog* plotDialog = new QDialog(this);
    plotDialog->setWindowTitle(tr("Selection Attribute Distribution"));
    plotDialog->setAttribute(Qt::WA_DeleteOnClose, true);
    plotDialog->resize(800, 600);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(plotDialog);
    
    // Create tab widget for different plots
    QTabWidget* plotTabs = new QTabWidget();
    mainLayout->addWidget(plotTabs);
    
    // Add histogram for each scalar array
    int numArrays = attributes->GetNumberOfArrays();
    for (int i = 0; i < numArrays; ++i) {
        vtkDataArray* array = attributes->GetArray(i);
        if (!array) continue;
        
        QString arrayName = QString::fromUtf8(array->GetName());
        int numComponents = array->GetNumberOfComponents();
        
        // Only plot scalar arrays (1 component)
        if (numComponents != 1) continue;
        
        // Create histogram widget
        QCustomPlot* histogram = new QCustomPlot();
        histogram->setMinimumHeight(400);
        
        // Compute histogram data
        vtkIdType numValues = array->GetNumberOfTuples();
        if (numValues > 0) {
            // Get data range
            double range[2];
            array->GetRange(range);
            
            // Compute histogram bins (use sqrt(N) as default)
            int numBins = std::max(10, std::min(100, static_cast<int>(std::sqrt(numValues))));
            std::vector<double> binCounts(numBins, 0.0);
            double binWidth = (range[1] - range[0]) / numBins;
            
            if (binWidth > 0) {
                for (vtkIdType j = 0; j < numValues; ++j) {
                    double value = array->GetComponent(j, 0);
                    int binIndex = static_cast<int>((value - range[0]) / binWidth);
                    if (binIndex >= numBins) binIndex = numBins - 1;
                    if (binIndex < 0) binIndex = 0;
                    binCounts[binIndex]++;
                }
                
                // Create bar chart
                QCPBars* bars = new QCPBars(histogram->xAxis, histogram->yAxis);
                QVector<double> xData, yData;
                
                for (int b = 0; b < numBins; ++b) {
                    double binCenter = range[0] + (b + 0.5) * binWidth;
                    xData.append(binCenter);
                    yData.append(binCounts[b]);
                }
                
                bars->setData(xData, yData);
                bars->setWidth(binWidth * 0.9);
                bars->setPen(QPen(QColor(100, 100, 100)));
                bars->setBrush(QColor(100, 150, 255, 150));
                
                // Configure axes
                histogram->xAxis->setLabel(arrayName);
                histogram->yAxis->setLabel(tr("Count"));
                histogram->xAxis->setRange(range[0], range[1]);
                histogram->rescaleAxes();
                
                // Add statistics text
                double sum = 0.0, sum2 = 0.0;
                for (vtkIdType j = 0; j < numValues; ++j) {
                    double value = array->GetComponent(j, 0);
                    sum += value;
                    sum2 += value * value;
                }
                double mean = sum / numValues;
                double variance = (sum2 / numValues) - (mean * mean);
                double stddev = std::sqrt(std::max(0.0, variance));
                
                // Add statistics as plot title
                QString statsText = QString("Count: %1  |  Min: %2  |  Max: %3  |  Mean: %4  |  Std Dev: %5")
                    .arg(numValues)
                    .arg(range[0], 0, 'f', 3)
                    .arg(range[1], 0, 'f', 3)
                    .arg(mean, 0, 'f', 3)
                    .arg(stddev, 0, 'f', 3);
                
                histogram->plotLayout()->insertRow(0);
                QCPPlotTitle* title = new QCPPlotTitle(histogram, statsText);
                title->setFont(QFont("sans", 9));
                histogram->plotLayout()->addElement(0, 0, title);
                
                histogram->replot();
            }
        }
        
        plotTabs->addTab(histogram, arrayName);
    }
    
    if (plotTabs->count() == 0) {
        QLabel* noDataLabel = new QLabel(tr("No scalar attributes available to plot"));
        noDataLabel->setAlignment(Qt::AlignCenter);
        plotTabs->addTab(noDataLabel, tr("No Data"));
    }
    
    // Add close button
    QPushButton* closeButton = new QPushButton(tr("Close"));
    connect(closeButton, &QPushButton::clicked, plotDialog, &QDialog::accept);
    mainLayout->addWidget(closeButton);
    
    plotDialog->show();
    
    emit plotOverTimeRequested();
    CVLog::Print("[cvSelectionPropertiesWidget] Showing attribute distribution plots");
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onToggleColumnVisibility() {
    // Create a menu to toggle column visibility
    QMenu menu(this);
    
    for (int col = 0; col < m_spreadsheetTable->columnCount(); ++col) {
        QString header = m_spreadsheetTable->horizontalHeaderItem(col)->text();
        QAction* action = menu.addAction(header);
        action->setCheckable(true);
        action->setChecked(!m_spreadsheetTable->isColumnHidden(col));
        action->setData(col);
        connect(action, &QAction::toggled, [this, col](bool visible) {
            m_spreadsheetTable->setColumnHidden(col, !visible);
        });
    }
    
    menu.exec(QCursor::pos());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSpreadsheetItemClicked(QTableWidgetItem* item) {
    if (!item) return;
    
    int row = item->row();
    
    // Get the ID from the first column
    QTableWidgetItem* idItem = m_spreadsheetTable->item(row, 0);
    if (idItem) {
        qint64 id = idItem->data(Qt::UserRole).toLongLong();
        highlightSingleItem(id);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateSelectionEditorTable() {
    m_selectionEditorTable->setRowCount(m_savedSelections.size());
    
    for (int i = 0; i < m_savedSelections.size(); ++i) {
        const SavedSelection& sel = m_savedSelections[i];
        
        // Name column
        QTableWidgetItem* nameItem = new QTableWidgetItem(sel.name);
        m_selectionEditorTable->setItem(i, 0, nameItem);
        
        // Type column
        QTableWidgetItem* typeItem = new QTableWidgetItem(sel.type);
        m_selectionEditorTable->setItem(i, 1, typeItem);
        
        // Color column (use background color)
        QTableWidgetItem* colorItem = new QTableWidgetItem(sel.color.name());
        colorItem->setBackground(sel.color);
        colorItem->setForeground(sel.color.lightness() > 128 ? Qt::black : Qt::white);
        m_selectionEditorTable->setItem(i, 2, colorItem);
    }
    
    m_selectionEditorTable->resizeColumnsToContents();
}

//-----------------------------------------------------------------------------
QString cvSelectionPropertiesWidget::generateSelectionName() {
    return QString("s%1").arg(m_selectionNameCounter++);
}

//-----------------------------------------------------------------------------
QColor cvSelectionPropertiesWidget::generateSelectionColor() const {
    int index = m_savedSelections.size() % s_selectionColorsCount;
    return s_selectionColors[index];
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setDataProducerName(const QString& name) {
    m_dataProducerName = name;
    if (m_dataProducerValue) {
        m_dataProducerValue->setText(name.isEmpty() ? tr("(none)") : name);
    }
    
    // Update the selected data header
    if (m_selectedDataLabel) {
        m_selectedDataLabel->setText(
            QString("<b>Selected Data (%1)</b>").arg(name.isEmpty() ? tr("none") : name));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateSpreadsheetData(vtkPolyData* polyData) {
    if (!polyData || !m_spreadsheetTable) {
        return;
    }
    
    bool isPointData = (m_attributeTypeCombo->currentIndex() == 0);
    
    // Clear existing data
    m_spreadsheetTable->clear();
    m_spreadsheetTable->setRowCount(0);
    
    // Get selection IDs
    if (m_selectionData.isEmpty()) {
        return;
    }
    
    const QVector<qint64>& ids = m_selectionData.ids();
    if (ids.isEmpty()) {
        return;
    }
    
    // Build column headers
    QStringList headers;
    headers << (isPointData ? tr("Point ID") : tr("Cell ID"));
    
    if (isPointData) {
        headers << tr("Points");  // Will show (x, y, z)
        
        // Add point data arrays
        vtkPointData* pointData = polyData->GetPointData();
        if (pointData) {
            for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = pointData->GetArray(i);
                if (arr && arr->GetName()) {
                    headers << QString::fromStdString(arr->GetName());
                }
            }
        }
    } else {
        headers << tr("Type") << tr("Num Points");
        
        // Add cell data arrays
        vtkCellData* cellData = polyData->GetCellData();
        if (cellData) {
            for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = cellData->GetArray(i);
                if (arr && arr->GetName()) {
                    headers << QString::fromStdString(arr->GetName());
                }
            }
        }
    }
    
    m_spreadsheetTable->setColumnCount(headers.size());
    m_spreadsheetTable->setHorizontalHeaderLabels(headers);
    
    // Populate rows
    int rowCount = std::min(1000, ids.size());  // Limit to 1000 rows
    m_spreadsheetTable->setRowCount(rowCount);
    
    for (int row = 0; row < rowCount; ++row) {
        qint64 id = ids[row];
        int col = 0;
        
        // ID column
        QTableWidgetItem* idItem = new QTableWidgetItem(QString::number(id));
        idItem->setData(Qt::UserRole, static_cast<qlonglong>(id));
        m_spreadsheetTable->setItem(row, col++, idItem);
        
        if (isPointData) {
            // Points column (x, y, z)
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(id, pt);
                QString coords = QString("%1, %2, %3")
                    .arg(pt[0], 0, 'f', 4)
                    .arg(pt[1], 0, 'f', 4)
                    .arg(pt[2], 0, 'f', 4);
                m_spreadsheetTable->setItem(row, col++, new QTableWidgetItem(coords));
                
                // Point data arrays
                vtkPointData* pointData = polyData->GetPointData();
                if (pointData) {
                    for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
                        vtkDataArray* arr = pointData->GetArray(i);
                        if (arr && arr->GetName()) {
                            double value = arr->GetTuple1(id);
                            m_spreadsheetTable->setItem(row, col++, 
                                new QTableWidgetItem(QString::number(value, 'g', 6)));
                        }
                    }
                }
            }
        } else {
            // Cell data
            if (id >= 0 && id < polyData->GetNumberOfCells()) {
                vtkCell* cell = polyData->GetCell(id);
                if (cell) {
                    // Type
                    QString typeName;
                    switch (cell->GetCellType()) {
                        case VTK_TRIANGLE: typeName = tr("Triangle"); break;
                        case VTK_QUAD: typeName = tr("Quad"); break;
                        case VTK_POLYGON: typeName = tr("Polygon"); break;
                        case VTK_LINE: typeName = tr("Line"); break;
                        case VTK_VERTEX: typeName = tr("Vertex"); break;
                        default: typeName = tr("Unknown"); break;
                    }
                    m_spreadsheetTable->setItem(row, col++, new QTableWidgetItem(typeName));
                    
                    // Num Points
                    m_spreadsheetTable->setItem(row, col++, 
                        new QTableWidgetItem(QString::number(cell->GetNumberOfPoints())));
                }
                
                // Cell data arrays
                vtkCellData* cellData = polyData->GetCellData();
                if (cellData) {
                    for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
                        vtkDataArray* arr = cellData->GetArray(i);
                        if (arr && arr->GetName()) {
                            double value = arr->GetTuple1(id);
                            m_spreadsheetTable->setItem(row, col++, 
                                new QTableWidgetItem(QString::number(value, 'g', 6)));
                        }
                    }
                }
            }
        }
    }
    
    m_spreadsheetTable->resizeColumnsToContents();
    
    CVLog::PrintDebug(QString("[cvSelectionPropertiesWidget] Updated spreadsheet with %1 rows")
                      .arg(rowCount));
}
