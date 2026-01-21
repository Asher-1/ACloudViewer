// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionPropertiesWidget.h"

// LOCAL
#include "PclUtils/PCLVis.h"
#include "cvExpanderButton.h"
#include "cvMultiColumnHeaderView.h"
#include "cvSelectionAlgebra.h"
#include "cvSelectionAnnotation.h"
#include "cvSelectionExporter.h"
#include "cvSelectionHighlighter.h"
#include "cvSelectionLabelPropertiesDialog.h"
#include "cvViewSelectionManager.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// CV_CORE_LIB
#include <CVLog.h>

// CV_IO_LIB
#include <FileIOFilter.h>

// Qt
#include <QApplication>
#include <QDateTime>
#include <QDialog>
#include <QEvent>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QTabWidget>
#include <QTimer>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// STL
#include <cmath>
#include <limits>

// QCustomPlot (PCLEngine uses its own copy)
#include <Tools/Common/qcustomplot.h>

// VTK
#include <vtkAbstractArray.h>
#include <vtkActor.h>
#include <vtkCell.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSetMapper.h>
#include <vtkFieldData.h>
#include <vtkIdTypeArray.h>
#include <vtkMapper.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkStringArray.h>
#include <vtkVariant.h>

// Qt
#include <QAbstractButton>
#include <QApplication>
#include <QBrush>
#include <QCheckBox>
#include <QClipboard>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QToolButton>
#include <QVBoxLayout>

// ParaView-style selection colors palette
const QColor cvSelectionPropertiesWidget::s_selectionColors[] = {
        QColor(255, 0, 0),    // Red
        QColor(0, 255, 0),    // Green
        QColor(0, 0, 255),    // Blue
        QColor(255, 255, 0),  // Yellow
        QColor(255, 0, 255),  // Magenta
        QColor(0, 255, 255),  // Cyan
        QColor(255, 128, 0),  // Orange
        QColor(128, 0, 255),  // Purple
        QColor(0, 255, 128),  // Spring Green
        QColor(255, 0, 128),  // Rose
};
const int cvSelectionPropertiesWidget::s_selectionColorsCount = 10;

//-----------------------------------------------------------------------------
cvSelectionPropertiesWidget::cvSelectionPropertiesWidget(QWidget* parent)
    : QWidget(parent),
      cvSelectionBase(),
      m_highlighter(nullptr),
      m_tooltipFormatter(new cvTooltipFormatter()),
      m_selectionManager(nullptr),
      m_selectionCount(0),
      m_volume(0.0),
      m_selectionNameCounter(0),
      m_extractCounter(0),
      m_lastHighlightedId(-1),
      // Expander buttons and containers (ParaView-style collapsible sections)
      m_selectedDataExpander(nullptr),
      m_selectedDataContainer(nullptr),
      m_selectionDisplayExpander(nullptr),
      m_selectionDisplayContainer(nullptr),
      m_selectionEditorExpander(nullptr),
      m_selectionEditorContainer(nullptr),
      m_createSelectionExpander(nullptr),
      m_createSelectionContainer(nullptr),
      m_selectedDataSpreadsheetExpander(nullptr),
      m_selectedDataSpreadsheetContainer(nullptr),
      m_compactStatsExpander(nullptr),
      m_compactStatsContainer(nullptr),
      // Selected Data header widgets
      m_freezeButton(nullptr),
      m_extractButton(nullptr),
      m_plotOverTimeButton(nullptr),
      // Create Selection section
      m_createSelectionGroup(nullptr),
      m_dataProducerCombo(nullptr),
      m_elementTypeCombo(nullptr),
      m_attributeCombo(nullptr),
      m_operatorCombo(nullptr),
      m_valueEdit(nullptr),
      m_processIdSpinBox(nullptr),
      m_findDataButton(nullptr),
      m_resetButton(nullptr),
      m_clearButton(nullptr),
      m_queriesLayout(nullptr),
      m_tabWidget(nullptr),
      // Legacy UI elements (kept for backward compatibility)
      m_selectionTableWidget(nullptr),
      m_listInfoLabel(nullptr),
      m_algebraOpCombo(nullptr),
      m_applyAlgebraButton(nullptr),
      m_extractBoundaryButton(nullptr),
      // Filter UI removed - not implemented
      m_addAnnotationButton(nullptr),
      // Legacy color/opacity controls
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

    // Colors are now stored in cvSelectionHighlighter (single source of truth)
    // UI buttons will be initialized from highlighter in
    // syncUIWithHighlighter()

    for (int i = 0; i < 6; ++i) {
        m_bounds[i] = 0.0;
    }
    for (int i = 0; i < 3; ++i) {
        m_center[i] = 0.0;
    }

    setupUi();

    // Set size policy to expand and fill available space
    // This is especially important when the widget is displayed alone (no DB
    // object selected)
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    CVLog::PrintVerbose(
            "[cvSelectionPropertiesWidget] Initialized with ParaView-style UI");
}

//-----------------------------------------------------------------------------
cvSelectionPropertiesWidget::~cvSelectionPropertiesWidget() {
    delete m_tooltipFormatter;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateScrollContentWidth() {
    // Helper function to update scroll content width
    // Called from multiple places to ensure consistency
    if (m_scrollContent && m_scrollArea && m_scrollArea->viewport()) {
        int viewportWidth = m_scrollArea->viewport()->width();
        if (viewportWidth > 0) {
            // CRITICAL: Set width BEFORE adjustSize to prevent it from being
            // reset Use setFixedWidth to ensure width constraint is maintained
            m_scrollContent->setFixedWidth(viewportWidth);
            // Force immediate layout update
            m_scrollContent->updateGeometry();
            // Update content size to recalculate layout (height only, width is
            // fixed)
            m_scrollContent->adjustSize();
            // Ensure width is still correct after adjustSize (defensive check)
            if (m_scrollContent->width() != viewportWidth) {
                m_scrollContent->setFixedWidth(viewportWidth);
            }
        }
    }
}

//-----------------------------------------------------------------------------
bool cvSelectionPropertiesWidget::eventFilter(QObject* obj, QEvent* event) {
    // Handle scroll area resize events to update content width
    if (obj == m_scrollArea && event->type() == QEvent::Resize) {
        updateScrollContentWidth();
    }
    // Handle viewport resize events (more reliable for drag resize)
    else if (obj == m_scrollArea->viewport() &&
             event->type() == QEvent::Resize) {
        updateScrollContentWidth();
    }
    // Handle resize events for color buttons (like ParaView's
    // pqColorChooserButton::resizeEvent)
    else if (event->type() == QEvent::Resize) {
        if (obj == m_selectionColorButton && m_highlighter) {
            QColor color = m_highlighter->getHighlightQColor(
                    cvSelectionHighlighter::SELECTED);
            updateColorButtonIcon(m_selectionColorButton, color);
        } else if (obj == m_interactiveSelectionColorButton && m_highlighter) {
            QColor color = m_highlighter->getHighlightQColor(
                    cvSelectionHighlighter::HOVER);
            updateColorButtonIcon(m_interactiveSelectionColorButton, color);
        }
    }
    // Call base class event filter
    return QWidget::eventFilter(obj, event);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    // Update scroll content width immediately when widget is resized (e.g., by
    // dragging) This ensures real-time responsiveness during drag resize
    // operations
    updateScrollContentWidth();
}

// setVisualizer is inherited from cvGenericSelectionTool

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setHighlighter(
        cvSelectionHighlighter* highlighter) {
    // Disconnect previous connections if highlighter changes
    if (m_highlighter && m_highlighter != highlighter) {
        disconnect(m_highlighter, nullptr, this, nullptr);
        disconnect(this, &cvSelectionPropertiesWidget::highlightColorChanged,
                   nullptr, nullptr);
        disconnect(this, &cvSelectionPropertiesWidget::highlightOpacityChanged,
                   nullptr, nullptr);
    }

    m_highlighter = highlighter;

    if (m_highlighter) {
        // Connect highlighter signals to UI update slots (bidirectional sync)
        // When highlighter properties change externally, update UI
        connect(m_highlighter, &cvSelectionHighlighter::colorChanged, this,
                &cvSelectionPropertiesWidget::onHighlighterColorChanged,
                Qt::UniqueConnection);
        connect(m_highlighter, &cvSelectionHighlighter::opacityChanged, this,
                &cvSelectionPropertiesWidget::onHighlighterOpacityChanged,
                Qt::UniqueConnection);
        connect(m_highlighter, &cvSelectionHighlighter::labelPropertiesChanged,
                this,
                &cvSelectionPropertiesWidget::
                        onHighlighterLabelPropertiesChanged,
                Qt::UniqueConnection);

        // Connect UI signals to highlighter (forward user changes)
        // Use Qt::UniqueConnection to prevent duplicate connections
        connect(
                this, &cvSelectionPropertiesWidget::highlightColorChanged, this,
                [this](double r, double g, double b, int mode) {
                    if (!m_highlighter) return;

                    cvSelectionHighlighter::HighlightMode hlMode =
                            static_cast<cvSelectionHighlighter::HighlightMode>(
                                    mode);

                    // Block signals to prevent feedback loop
                    m_highlighter->blockSignals(true);
                    m_highlighter->setHighlightColor(r, g, b, hlMode);
                    m_highlighter->blockSignals(false);

                    // Refresh display
                    PclUtils::PCLVis* pclVis = getPCLVis();
                    if (pclVis) {
                        pclVis->UpdateScreen();
                    }
                },
                Qt::UniqueConnection);

        connect(
                this, &cvSelectionPropertiesWidget::highlightOpacityChanged,
                this,
                [this](double opacity, int mode) {
                    if (!m_highlighter) return;

                    cvSelectionHighlighter::HighlightMode hlMode =
                            static_cast<cvSelectionHighlighter::HighlightMode>(
                                    mode);

                    // Block signals to prevent feedback loop
                    m_highlighter->blockSignals(true);
                    m_highlighter->setHighlightOpacity(opacity, hlMode);
                    m_highlighter->blockSignals(false);

                    // Refresh display
                    PclUtils::PCLVis* pclVis = getPCLVis();
                    if (pclVis) {
                        pclVis->UpdateScreen();
                    }
                },
                Qt::UniqueConnection);

        // Sync UI with highlighter's current settings
        syncUIWithHighlighter();
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::syncInternalColorArray(double r,
                                                         double g,
                                                         double b,
                                                         int mode) {
    // DEPRECATED: Colors are now stored in cvSelectionHighlighter
    // This method now only updates UI buttons to reflect color changes
    // Called when colors change via highlightColorChanged signal
    // Use ParaView-style icon-based color buttons
    QColor color = QColor::fromRgbF(r, g, b);

    switch (mode) {
        case cvSelectionHighlighter::HOVER:
            if (m_interactiveSelectionColorButton) {
                updateColorButtonIcon(m_interactiveSelectionColorButton, color);
            }
            if (m_hoverColorButton) {
                updateColorButtonIcon(m_hoverColorButton, color);
            }
            break;
        case cvSelectionHighlighter::PRESELECTED:
            if (m_preselectedColorButton) {
                updateColorButtonIcon(m_preselectedColorButton, color);
            }
            break;
        case cvSelectionHighlighter::SELECTED:
            if (m_selectionColorButton) {
                updateColorButtonIcon(m_selectionColorButton, color);
            }
            if (m_selectedColorButton) {
                updateColorButtonIcon(m_selectedColorButton, color);
            }
            break;
        case cvSelectionHighlighter::BOUNDARY:
            if (m_boundaryColorButton) {
                updateColorButtonIcon(m_boundaryColorButton, color);
            }
            break;
        default:
            break;
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateColorButtonIcon(QAbstractButton* button,
                                                        const QColor& color) {
    if (!button) return;

    // ParaView style: use button height * 0.75 for icon radius (same as
    // pqColorChooserButton) Reference: pqColorChooserButton::renderColorSwatch
    int buttonHeight = button->height();
    if (buttonHeight <= 0) {
        button->adjustSize();
        buttonHeight = button->height();
    }
    if (buttonHeight <= 0) {
        buttonHeight = button->sizeHint().height();
    }
    if (buttonHeight <= 0) {
        buttonHeight = 25;  // Fallback default
    }

    // Calculate radius based on height (ParaView style: IconRadiusHeightRatio =
    // 0.75)
    int radius = qRound(buttonHeight * 0.75);
    radius = std::max(radius, 10);  // Minimum 10px (ParaView default)

    // Create circular color swatch icon (ParaView-style)
    // Use exact same approach as pqColorChooserButton::renderColorSwatch
    QPixmap pix(radius, radius);
    pix.fill(QColor(0, 0, 0, 0));  // Transparent background

    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setBrush(QBrush(color));
    painter.drawEllipse(1, 1, radius - 2, radius - 2);
    painter.end();

    QIcon icon(pix);

    // Add high-dpi version for retina displays (ParaView exact style)
    QPixmap pix2x(radius * 2, radius * 2);
    pix2x.setDevicePixelRatio(2.0);
    pix2x.fill(QColor(0, 0, 0, 0));

    QPainter painter2x(&pix2x);
    painter2x.setRenderHint(QPainter::Antialiasing, true);
    painter2x.setBrush(QBrush(color));
    // ParaView uses: drawEllipse(2, 2, radius - 4, radius - 4) for 2x version
    painter2x.drawEllipse(2, 2, radius - 4, radius - 4);
    painter2x.end();

    icon.addPixmap(pix2x);

    button->setIcon(icon);

    // Set icon size (QToolButton will use this)
    if (QToolButton* toolButton = qobject_cast<QToolButton*>(button)) {
        toolButton->setIconSize(QSize(radius, radius));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onHighlighterColorChanged(int mode) {
    // Called when highlighter color changes externally
    // Update UI buttons to reflect the new color using ParaView-style icons
    if (!m_highlighter) return;

    cvSelectionHighlighter::HighlightMode hlMode =
            static_cast<cvSelectionHighlighter::HighlightMode>(mode);
    QColor color = m_highlighter->getHighlightQColor(hlMode);

    switch (mode) {
        case cvSelectionHighlighter::HOVER:
            if (m_hoverColorButton)
                updateColorButtonIcon(m_hoverColorButton, color);
            if (m_interactiveSelectionColorButton)
                updateColorButtonIcon(m_interactiveSelectionColorButton, color);
            break;
        case cvSelectionHighlighter::PRESELECTED:
            if (m_preselectedColorButton)
                updateColorButtonIcon(m_preselectedColorButton, color);
            break;
        case cvSelectionHighlighter::SELECTED:
            if (m_selectedColorButton)
                updateColorButtonIcon(m_selectedColorButton, color);
            if (m_selectionColorButton)
                updateColorButtonIcon(m_selectionColorButton, color);
            break;
        case cvSelectionHighlighter::BOUNDARY:
            if (m_boundaryColorButton)
                updateColorButtonIcon(m_boundaryColorButton, color);
            break;
    }

    CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] UI updated for "
                                "external color change (mode=%1)")
                                .arg(mode));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onHighlighterOpacityChanged(int mode) {
    // Called when highlighter opacity changes externally
    // Update UI spinboxes to reflect the new opacity
    if (!m_highlighter) return;

    cvSelectionHighlighter::HighlightMode hlMode =
            static_cast<cvSelectionHighlighter::HighlightMode>(mode);
    double opacity = m_highlighter->getHighlightOpacity(hlMode);

    QDoubleSpinBox* spinbox = nullptr;
    switch (mode) {
        case cvSelectionHighlighter::HOVER:
            spinbox = m_hoverOpacitySpin;
            break;
        case cvSelectionHighlighter::PRESELECTED:
            spinbox = m_preselectedOpacitySpin;
            break;
        case cvSelectionHighlighter::SELECTED:
            spinbox = m_selectedOpacitySpin;
            break;
        case cvSelectionHighlighter::BOUNDARY:
            spinbox = m_boundaryOpacitySpin;
            break;
    }

    if (spinbox) {
        spinbox->blockSignals(true);
        spinbox->setValue(opacity);
        spinbox->blockSignals(false);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onHighlighterLabelPropertiesChanged(
        bool interactive) {
    // Called when highlighter label properties change externally
    // This could trigger a full UI sync if needed
    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Label properties "
                    "changed externally (interactive=%1)")
                    .arg(interactive));
    // For now, just log - UI will be updated on next syncUIWithHighlighter()
    // call
}

//-----------------------------------------------------------------------------
QColor cvSelectionPropertiesWidget::getSelectionColor() const {
    if (m_highlighter) {
        return m_highlighter->getHighlightQColor(
                cvSelectionHighlighter::SELECTED);
    }
    return QColor(255, 0, 255);  // Default magenta
}

//-----------------------------------------------------------------------------
QColor cvSelectionPropertiesWidget::getInteractiveSelectionColor() const {
    if (m_highlighter) {
        return m_highlighter->getHighlightQColor(cvSelectionHighlighter::HOVER);
    }
    return QColor(0, 255, 0);  // Default green
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::syncUIWithHighlighter() {
    if (!m_highlighter) {
        return;
    }

    // Get current colors from highlighter (single source of truth)
    QColor hoverColor =
            m_highlighter->getHighlightQColor(cvSelectionHighlighter::HOVER);
    QColor preselectedColor = m_highlighter->getHighlightQColor(
            cvSelectionHighlighter::PRESELECTED);
    QColor selectedColor =
            m_highlighter->getHighlightQColor(cvSelectionHighlighter::SELECTED);
    QColor boundaryColor =
            m_highlighter->getHighlightQColor(cvSelectionHighlighter::BOUNDARY);

    double hoverOpacity =
            m_highlighter->getHighlightOpacity(cvSelectionHighlighter::HOVER);
    double preselectedOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::PRESELECTED);
    double selectedOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::SELECTED);
    double boundaryOpacity = m_highlighter->getHighlightOpacity(
            cvSelectionHighlighter::BOUNDARY);

    // Update UI controls - colors are read directly from highlighter
    // Use ParaView-style icon-based color buttons
    if (m_hoverColorButton) {
        updateColorButtonIcon(m_hoverColorButton, hoverColor);
    }
    if (m_preselectedColorButton) {
        updateColorButtonIcon(m_preselectedColorButton, preselectedColor);
    }
    if (m_selectedColorButton) {
        updateColorButtonIcon(m_selectedColorButton, selectedColor);
    }
    if (m_boundaryColorButton) {
        updateColorButtonIcon(m_boundaryColorButton, boundaryColor);
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

    // Label properties are now stored in highlighter (single source of truth)
    // No local copy needed - dialog will read directly from highlighter

    // Update ParaView-style selection color buttons using icons
    if (m_selectionColorButton) {
        updateColorButtonIcon(m_selectionColorButton, selectedColor);
    }
    if (m_interactiveSelectionColorButton) {
        updateColorButtonIcon(m_interactiveSelectionColorButton, hoverColor);
    }

    CVLog::PrintVerbose(
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
    // Set widgetResizable to false so content maintains its natural size
    // This allows scrollbars to appear when content exceeds available space
    m_scrollArea->setWidgetResizable(false);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    m_scrollContent = new QWidget();
    // Set size policy to allow content to expand naturally based on its
    // contents
    m_scrollContent->setSizePolicy(QSizePolicy::Preferred,
                                   QSizePolicy::Minimum);
    QVBoxLayout* scrollLayout = new QVBoxLayout(m_scrollContent);
    scrollLayout->setContentsMargins(5, 5, 5, 5);
    scrollLayout->setSpacing(
            0);  // ParaView-style: no spacing between expanders
    // Set size constraint to ensure layout calculates minimum size correctly
    scrollLayout->setSizeConstraint(QLayout::SetMinimumSize);

    // === ParaView-style sections with cvExpanderButton ===

    // 1. Create Selection section (ParaView's Find Data) - first section
    m_createSelectionExpander = new cvExpanderButton(m_scrollContent);
    m_createSelectionExpander->setText(tr("Create Selection"));
    m_createSelectionExpander->setChecked(true);  // Expanded by default
    scrollLayout->addWidget(m_createSelectionExpander);

    m_createSelectionContainer = new QWidget(m_scrollContent);
    m_createSelectionContainer->setMinimumHeight(50);
    setupCreateSelectionSection();
    scrollLayout->addWidget(m_createSelectionContainer);

    connect(m_createSelectionExpander, &cvExpanderButton::toggled,
            m_createSelectionContainer, &QWidget::setVisible);
    // Also connect to show/hide the action buttons below
    connect(m_createSelectionExpander, &cvExpanderButton::toggled,
            [this](bool checked) {
                if (m_findDataButton) m_findDataButton->setVisible(checked);
                if (m_resetButton) m_resetButton->setVisible(checked);
                if (m_clearButton) m_clearButton->setVisible(checked);
                // Update scroll content size when section is toggled
                // Ensure width is maintained and content size is updated
                QTimer::singleShot(0, this, [this]() {
                    // First ensure width is set, then adjust size
                    updateScrollContentWidth();
                });
            });

    // 2. Selected Data section (with Freeze/Extract/Plot Over Time buttons)
    m_selectedDataSpreadsheetExpander = new cvExpanderButton(m_scrollContent);
    m_selectedDataSpreadsheetExpander->setText(tr("Selected Data (none)"));
    m_selectedDataSpreadsheetExpander->setChecked(true);
    scrollLayout->addWidget(m_selectedDataSpreadsheetExpander);

    m_selectedDataSpreadsheetContainer = new QWidget(m_scrollContent);
    m_selectedDataSpreadsheetContainer->setMinimumHeight(200);
    setupSelectedDataSpreadsheet();
    scrollLayout->addWidget(m_selectedDataSpreadsheetContainer);

    connect(m_selectedDataSpreadsheetExpander, &cvExpanderButton::toggled,
            m_selectedDataSpreadsheetContainer, &QWidget::setVisible);
    // Update scroll content size when section is toggled
    connect(m_selectedDataSpreadsheetExpander, &cvExpanderButton::toggled,
            [this](bool) {
                QTimer::singleShot(0, this, [this]() {
                    // First ensure width is set, then adjust size
                    updateScrollContentWidth();
                });
            });

    // Action buttons row (Freeze, Extract, Plot Over Time)
    setupSelectedDataHeader();
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(2);
    buttonLayout->addWidget(m_freezeButton);
    buttonLayout->addWidget(m_extractButton);
    if (m_plotOverTimeButton) {
        buttonLayout->addWidget(m_plotOverTimeButton);
    }
    QWidget* buttonContainer = new QWidget(m_scrollContent);
    buttonContainer->setLayout(buttonLayout);
    scrollLayout->addWidget(buttonContainer);

    // 3. Selection Display section
    m_selectionDisplayExpander = new cvExpanderButton(m_scrollContent);
    m_selectionDisplayExpander->setText(tr("Selection Display"));
    m_selectionDisplayExpander->setChecked(true);
    scrollLayout->addWidget(m_selectionDisplayExpander);

    m_selectionDisplayContainer = new QWidget(m_scrollContent);
    m_selectionDisplayContainer->setMinimumHeight(50);
    setupSelectionDisplaySection();
    scrollLayout->addWidget(m_selectionDisplayContainer);

    connect(m_selectionDisplayExpander, &cvExpanderButton::toggled,
            m_selectionDisplayContainer, &QWidget::setVisible);
    // Update scroll content size when section is toggled
    connect(m_selectionDisplayExpander, &cvExpanderButton::toggled,
            [this](bool) {
                QTimer::singleShot(0, this, [this]() {
                    // First ensure width is set, then adjust size
                    updateScrollContentWidth();
                });
            });

    // 4. Selection Editor section (for combining and managing selections)
    m_selectionEditorExpander = new cvExpanderButton(m_scrollContent);
    m_selectionEditorExpander->setText(tr("Selection Editor"));
    m_selectionEditorExpander->setChecked(false);  // Collapsed by default
    scrollLayout->addWidget(m_selectionEditorExpander);

    m_selectionEditorContainer = new QWidget(m_scrollContent);
    m_selectionEditorContainer->setMinimumHeight(50);
    m_selectionEditorContainer->setVisible(false);  // Hidden by default
    setupSelectionEditorSection();
    scrollLayout->addWidget(m_selectionEditorContainer);

    connect(m_selectionEditorExpander, &cvExpanderButton::toggled,
            m_selectionEditorContainer, &QWidget::setVisible);
    // Update scroll content size when section is toggled
    connect(m_selectionEditorExpander, &cvExpanderButton::toggled,
            [this](bool) {
                QTimer::singleShot(0, this, [this]() {
                    // First ensure width is set, then adjust size
                    updateScrollContentWidth();
                });
            });

    // 5. Compact Statistics Section (ParaView-style: no tabs)
    m_compactStatsExpander = new cvExpanderButton(m_scrollContent);
    m_compactStatsExpander->setText(tr("Selection Statistics"));
    m_compactStatsExpander->setChecked(false);  // Collapsed by default
    scrollLayout->addWidget(m_compactStatsExpander);

    m_compactStatsContainer = new QWidget(m_scrollContent);
    m_compactStatsContainer->setMinimumHeight(50);
    m_compactStatsContainer->setVisible(false);  // Hidden by default
    setupCompactStatisticsSection();
    scrollLayout->addWidget(m_compactStatsContainer);

    connect(m_compactStatsExpander, &cvExpanderButton::toggled,
            m_compactStatsContainer, &QWidget::setVisible);
    // Update scroll content size when section is toggled
    connect(m_compactStatsExpander, &cvExpanderButton::toggled, [this](bool) {
        QTimer::singleShot(0, this, [this]() {
            // First ensure width is set, then adjust size
            updateScrollContentWidth();
        });
    });

    // Note: Tab widget removed to align with ParaView's simpler UI design
    // Export/Advanced features are now accessible via action buttons or menus
    m_tabWidget = nullptr;

    scrollLayout->addStretch();

    m_scrollContent->setLayout(scrollLayout);

    m_scrollArea->setWidget(m_scrollContent);
    // Set the scroll area to expand and fill available space
    m_scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    mainLayout->addWidget(m_scrollArea);

    // Install event filter on scroll area and viewport to detect resize events
    // This ensures content width matches scroll area width (prevents horizontal
    // scrolling) while allowing vertical scrolling when content height exceeds
    // available space
    m_scrollArea->installEventFilter(this);
    if (m_scrollArea->viewport()) {
        m_scrollArea->viewport()->installEventFilter(this);
    }

    // Update scroll content size after everything is set up
    // Use QTimer to ensure this happens after the widget is shown
    QTimer::singleShot(0, this, [this]() { updateScrollContentWidth(); });

    setLayout(mainLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectedDataHeader() {
    // Selected Data header label
    m_selectedDataLabel = new QLabel(tr("<b>Selected Data</b>"));
    m_selectedDataLabel->setStyleSheet(
            "QLabel { background-color: #e0e0e0; padding: 5px; border-radius: "
            "3px; }");

    // Action buttons (ParaView-style with icons)

    // Freeze button (ParaView: converts selection to a frozen representation)
    m_freezeButton = new QPushButton(QIcon(":/Resources/images/svg/pqLock.svg"),
                                     tr("Freeze"));
    m_freezeButton->setToolTip(tr(
            "Freeze the current selection (convert to independent dataset)"));
    m_freezeButton->setFixedHeight(25);
    m_freezeButton->setEnabled(false);  // Enabled when selection exists
    connect(m_freezeButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onFreezeClicked);

    // Extract button (ParaView: pqExtractSelection.svg - creates new object
    // from selection)
    QIcon extractIcon(":/Resources/images/svg/pqExtractSelection.svg");
    if (extractIcon.isNull()) {
        extractIcon = QIcon(":/Resources/images/exportCloud.png");  // Fallback
    }
    m_extractButton = new QPushButton(extractIcon, tr("Extract"));
    m_extractButton->setToolTip(
            tr("Extract selected elements to a new dataset and add to scene"));
    m_extractButton->setFixedHeight(25);
    m_extractButton->setEnabled(false);  // Enabled when selection exists
    connect(m_extractButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onExtractClicked);

    // Note: Plot Distribution button removed - feature not fully implemented
    // Can be re-added when histogram plotting is available
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupCompactStatisticsSection() {
    // Compact statistics section (replaces Statistics tab)
    // Content is placed inside m_compactStatsContainer which is controlled by
    // cvExpanderButton No QGroupBox needed - the expander handles collapsing

    QFormLayout* statsLayout = new QFormLayout(m_compactStatsContainer);
    statsLayout->setSpacing(4);
    statsLayout->setContentsMargins(8, 8, 8, 8);

    m_countLabel = new QLabel(tr("0"));
    m_countLabel->setStyleSheet("font-weight: bold;");
    statsLayout->addRow(tr("Count:"), m_countLabel);

    m_typeLabel = new QLabel(tr("None"));
    statsLayout->addRow(tr("Type:"), m_typeLabel);

    m_boundsLabel = new QLabel(tr("N/A"));
    m_boundsLabel->setWordWrap(true);
    m_boundsLabel->setStyleSheet("font-size: 9pt;");
    statsLayout->addRow(tr("Bounds:"), m_boundsLabel);

    m_centerLabel = new QLabel(tr("N/A"));
    m_centerLabel->setStyleSheet("font-size: 9pt;");
    statsLayout->addRow(tr("Center:"), m_centerLabel);

    m_volumeLabel = new QLabel(tr("N/A"));
    m_volumeLabel->setStyleSheet("font-size: 9pt;");
    statsLayout->addRow(tr("Volume:"), m_volumeLabel);

    // No need for setupCollapsibleGroupBox - handled by cvExpanderButton
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupCreateSelectionSection() {
    // Create Selection section - ParaView's "Find Data" functionality
    // Uses m_createSelectionContainer instead of QGroupBox for ParaView-style
    // collapsible behavior with cvExpanderButton
    m_createSelectionGroup = nullptr;  // Not using QGroupBox anymore

    QVBoxLayout* mainLayout = new QVBoxLayout(m_createSelectionContainer);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(8, 8, 8, 8);

    // === Selection Criteria ===
    QLabel* criteriaLabel = new QLabel(tr("<b>Selection Criteria</b>"));
    mainLayout->addWidget(criteriaLabel);

    QFormLayout* criteriaLayout = new QFormLayout();
    criteriaLayout->setSpacing(3);

    // Data Producer combo
    m_dataProducerCombo = new QComboBox();
    m_dataProducerCombo->setToolTip(tr("Select the data source"));
    connect(m_dataProducerCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &cvSelectionPropertiesWidget::onDataProducerChanged);
    criteriaLayout->addRow(tr("Data Producer"), m_dataProducerCombo);

    // Element Type combo (Point/Cell)
    m_elementTypeCombo = new QComboBox();
    QIcon pointIcon(":/Resources/images/svg/pqPointData.svg");
    m_elementTypeCombo->addItem(pointIcon, tr("Point"), 0);

    QIcon cellIcon(":/Resources/images/svg/pqCellData.svg");
    m_elementTypeCombo->addItem(cellIcon, tr("Cell"), 1);
    m_elementTypeCombo->setToolTip(tr("Select element type (Point or Cell)"));
    connect(m_elementTypeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &cvSelectionPropertiesWidget::onElementTypeChanged);
    criteriaLayout->addRow(tr("Element Type"), m_elementTypeCombo);

    mainLayout->addLayout(criteriaLayout);

    // Container for query rows (multiple conditions with +/- buttons)
    m_queriesLayout = new QVBoxLayout();
    m_queriesLayout->setSpacing(3);
    mainLayout->addLayout(m_queriesLayout);

    // Add the first query row
    addQueryRow();

    // Keep legacy pointers for compatibility (pointing to first row)
    if (!m_queryRows.isEmpty()) {
        m_attributeCombo = m_queryRows[0].attributeCombo;
        m_operatorCombo = m_queryRows[0].operatorCombo;
        m_valueEdit = m_queryRows[0].valueEdit;
    } else {
        m_attributeCombo = nullptr;
        m_operatorCombo = nullptr;
        m_valueEdit = nullptr;
    }

    // === Selection Qualifiers ===
    QLabel* qualifiersLabel = new QLabel(tr("<b>Selection Qualifiers</b>"));
    mainLayout->addWidget(qualifiersLabel);

    QFormLayout* qualifiersLayout = new QFormLayout();
    qualifiersLayout->setSpacing(3);

    // Process ID
    m_processIdSpinBox = new QSpinBox();
    m_processIdSpinBox->setRange(-1, 9999);
    m_processIdSpinBox->setValue(-1);
    m_processIdSpinBox->setToolTip(tr("Process ID (-1 for all)"));
    qualifiersLayout->addRow(tr("Process ID"), m_processIdSpinBox);

    mainLayout->addLayout(qualifiersLayout);

    // Action buttons: Find Data | Reset | Clear
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(5);

    // Use ParaView-style icons
    QIcon findIcon(":/Resources/images/svg/pqApply.svg");
    if (findIcon.isNull()) {
        findIcon = QIcon(":/Resources/images/svg/pqApply.png");
    }
    m_findDataButton = new QPushButton(findIcon, tr("Find Data"));
    m_findDataButton->setToolTip(tr("Find data using selection criteria"));
    connect(m_findDataButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onFindDataClicked);
    buttonLayout->addWidget(m_findDataButton);

    QIcon resetIcon(":/Resources/images/svg/pqCancel.svg");
    if (resetIcon.isNull()) {
        resetIcon = QIcon(":/Resources/images/svg/pqCancel.png");
    }
    m_resetButton = new QPushButton(resetIcon, tr("Reset"));
    m_resetButton->setToolTip(tr("Reset any unaccepted changes"));
    connect(m_resetButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onResetClicked);
    buttonLayout->addWidget(m_resetButton);

    QIcon clearIcon(":/Resources/images/svg/pqReset.svg");
    if (clearIcon.isNull()) {
        clearIcon = QIcon(":/Resources/images/svg/pqReset.png");
    }
    m_clearButton = new QPushButton(clearIcon, tr("Clear"));
    m_clearButton->setToolTip(tr("Clear selection criteria and qualifiers"));
    connect(m_clearButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onClearClicked);
    buttonLayout->addWidget(m_clearButton);

    mainLayout->addLayout(buttonLayout);

    // Layout is already set on m_createSelectionContainer
    // No need for setupCollapsibleGroupBox - handled by cvExpanderButton
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectionDisplaySection() {
    // Uses m_selectionDisplayContainer instead of QGroupBox for ParaView-style
    // collapsible behavior with cvExpanderButton
    m_selectionDisplayGroup = nullptr;  // Not using QGroupBox anymore

    QVBoxLayout* displayLayout = new QVBoxLayout(m_selectionDisplayContainer);
    displayLayout->setSpacing(0);  // ParaView-style: compact spacing
    displayLayout->setContentsMargins(8, 8, 8, 8);

    // === Selection Labels === (ParaView-style header with separator line)
    QVBoxLayout* labelsHeaderLayout = new QVBoxLayout();
    labelsHeaderLayout->setSpacing(0);
    QLabel* labelsHeader = new QLabel(
            tr("<html><body><p><span style=\"font-weight:600;\">Selection "
               "Labels</span></p></body></html>"));
    labelsHeaderLayout->addWidget(labelsHeader);
    QFrame* labelsSeparator = new QFrame();
    labelsSeparator->setFrameShape(QFrame::HLine);
    labelsSeparator->setFrameShadow(QFrame::Sunken);
    labelsHeaderLayout->addWidget(labelsSeparator);
    displayLayout->addLayout(labelsHeaderLayout);

    // Cell Labels and Point Labels buttons (ParaView-style: horizontal,
    // spacing=2)
    QHBoxLayout* labelsLayout = new QHBoxLayout();
    labelsLayout->setSpacing(2);

    // Cell Labels button with dropdown menu (ParaView: pqCellData.svg icon)
    m_cellLabelsButton = new QPushButton(tr("Cell Labels"));
    m_cellLabelsButton->setIcon(QIcon(":/Resources/images/svg/pqCellData.svg"));
    m_cellLabelsButton->setToolTip(
            tr("Set the array to label selected cells with"));
    m_cellLabelsMenu = new QMenu(this);
    m_cellLabelsButton->setMenu(m_cellLabelsMenu);
    // Populate menu when about to show (NOT when button clicked)
    connect(m_cellLabelsMenu, &QMenu::aboutToShow, this,
            &cvSelectionPropertiesWidget::onCellLabelsClicked);
    labelsLayout->addWidget(m_cellLabelsButton);

    // Point Labels button with dropdown menu (ParaView: pqPointData.svg icon)
    m_pointLabelsButton = new QPushButton(tr("Point Labels"));
    m_pointLabelsButton->setIcon(
            QIcon(":/Resources/images/svg/pqPointData.svg"));
    m_pointLabelsButton->setToolTip(
            tr("Set the array to label selected points with"));
    m_pointLabelsMenu = new QMenu(this);
    m_pointLabelsButton->setMenu(m_pointLabelsMenu);
    // Populate menu when about to show (NOT when button clicked)
    connect(m_pointLabelsMenu, &QMenu::aboutToShow, this,
            &cvSelectionPropertiesWidget::onPointLabelsClicked);
    labelsLayout->addWidget(m_pointLabelsButton);

    displayLayout->addLayout(labelsLayout);

    // Edit Label Properties button (ParaView: pqAdvanced.svg icon)
    m_editLabelPropertiesButton = new QPushButton(tr("Edit Label Properties"));
    m_editLabelPropertiesButton->setIcon(
            QIcon(":/Resources/images/svg/pqAdvanced.png"));
    m_editLabelPropertiesButton->setToolTip(
            tr("Edit selection label properties"));
    connect(m_editLabelPropertiesButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onEditLabelPropertiesClicked);
    displayLayout->addWidget(m_editLabelPropertiesButton);

    // === Selection Appearance === (ParaView-style header with separator line)
    QVBoxLayout* appearanceHeaderLayout = new QVBoxLayout();
    appearanceHeaderLayout->setSpacing(0);
    QLabel* appearanceHeader = new QLabel(
            tr("<html><body><p><span style=\"font-weight:600;\">Selection "
               "Appearance</span></p></body></html>"));
    appearanceHeaderLayout->addWidget(appearanceHeader);
    QFrame* appearanceSeparator = new QFrame();
    appearanceSeparator->setFrameShape(QFrame::HLine);
    appearanceSeparator->setFrameShadow(QFrame::Sunken);
    appearanceHeaderLayout->addWidget(appearanceSeparator);
    displayLayout->addLayout(appearanceHeaderLayout);

    // Selection Color button (ParaView: pqColorChooserButton style)
    // Use QToolButton like ParaView's pqColorChooserButton (which extends
    // QToolButton)
    m_selectionColorButton = new QToolButton();
    m_selectionColorButton->setText(tr("Selection Color"));
    m_selectionColorButton->setToolTip(
            tr("Set the color to use to show selected elements"));
    m_selectionColorButton->setSizePolicy(QSizePolicy::Minimum,
                                          QSizePolicy::Fixed);
    // ParaView style: TextBesideIcon - text and icon side by side
    m_selectionColorButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    // Install event filter to handle resize events (like ParaView's
    // pqColorChooserButton)
    m_selectionColorButton->installEventFilter(this);
    // ParaView style: use icon to display color (will be updated when
    // highlighter is set) Default color (magenta) - will be updated when
    // highlighter is set
    updateColorButtonIcon(m_selectionColorButton,
                          QColor(255, 0, 255));  // Magenta default
    connect(m_selectionColorButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onSelectionColorClicked);
    displayLayout->addWidget(m_selectionColorButton);

    // === Interactive Selection === (ParaView-style header with separator line)
    QVBoxLayout* interactiveHeaderLayout = new QVBoxLayout();
    interactiveHeaderLayout->setSpacing(0);
    QLabel* interactiveHeader = new QLabel(
            tr("<html><body><p><span style=\"font-weight:600;\">Interactive "
               "Selection</span></p></body></html>"));
    interactiveHeaderLayout->addWidget(interactiveHeader);
    QFrame* interactiveSeparator = new QFrame();
    interactiveSeparator->setFrameShape(QFrame::HLine);
    interactiveSeparator->setFrameShadow(QFrame::Sunken);
    interactiveHeaderLayout->addWidget(interactiveSeparator);
    displayLayout->addLayout(interactiveHeaderLayout);

    // Interactive Selection Color button (ParaView: pqColorChooserButton style)
    // Use QToolButton like ParaView's pqColorChooserButton (which extends
    // QToolButton)
    m_interactiveSelectionColorButton = new QToolButton();
    m_interactiveSelectionColorButton->setText(
            tr("Interactive Selection Color"));
    m_interactiveSelectionColorButton->setToolTip(
            tr("Set the color to use to show selected elements during "
               "interaction"));
    m_interactiveSelectionColorButton->setSizePolicy(QSizePolicy::Minimum,
                                                     QSizePolicy::Fixed);
    // ParaView style: TextBesideIcon - text and icon side by side
    m_interactiveSelectionColorButton->setToolButtonStyle(
            Qt::ToolButtonTextBesideIcon);
    // Install event filter to handle resize events (like ParaView's
    // pqColorChooserButton)
    m_interactiveSelectionColorButton->installEventFilter(this);
    // ParaView style: use icon to display color (will be updated when
    // highlighter is set) Default color (cyan) - will be updated when
    // highlighter is set
    updateColorButtonIcon(m_interactiveSelectionColorButton,
                          QColor(0, 255, 255));  // Cyan default
    connect(m_interactiveSelectionColorButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onInteractiveSelectionColorClicked);
    displayLayout->addWidget(m_interactiveSelectionColorButton);

    // Edit Interactive Label Properties button (ParaView: pqAdvanced.svg icon)
    m_editInteractiveLabelPropertiesButton =
            new QPushButton(tr("Edit Interactive Label Properties"));
    m_editInteractiveLabelPropertiesButton->setIcon(
            QIcon(":/Resources/images/svg/pqAdvanced.png"));
    m_editInteractiveLabelPropertiesButton->setToolTip(
            tr("Edit interactive selection label properties"));
    connect(m_editInteractiveLabelPropertiesButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::
                    onEditInteractiveLabelPropertiesClicked);
    displayLayout->addWidget(m_editInteractiveLabelPropertiesButton);

    // Add vertical spacer at bottom (ParaView-style)
    displayLayout->addStretch();

    // Layout is already set on m_selectionDisplayContainer
    // No need for setupCollapsibleGroupBox - handled by cvExpanderButton
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectionEditorSection() {
    // Content is placed inside m_selectionEditorContainer which is controlled
    // by cvExpanderButton No QGroupBox needed - the expander handles collapsing

    QVBoxLayout* editorLayout = new QVBoxLayout(m_selectionEditorContainer);
    editorLayout->setSpacing(5);  // ParaView-style spacing
    editorLayout->setContentsMargins(8, 8, 8, 8);

    // Data Producer row (ParaView-style: spacing=5)
    QHBoxLayout* producerLayout = new QHBoxLayout();
    producerLayout->setSpacing(5);
    m_dataProducerLabel = new QLabel(tr("Data Producer"));
    m_dataProducerLabel->setSizePolicy(QSizePolicy::Maximum,
                                       QSizePolicy::Preferred);
    m_dataProducerLabel->setToolTip(
            tr("The dataset for which selections are saved"));
    m_dataProducerValue = new QLabel();
    m_dataProducerValue->setStyleSheet(
            "QLabel { background-color: snow; border: 1px inset grey; "
            "}");  // ParaView exact style
    m_dataProducerValue->setToolTip(
            tr("The dataset for which selections are saved"));
    m_dataProducerValue->setText(
            m_dataProducerName.isEmpty() ? QString() : m_dataProducerName);
    producerLayout->addWidget(m_dataProducerLabel);
    producerLayout->addWidget(m_dataProducerValue, 1);
    editorLayout->addLayout(producerLayout);

    // Element Type row (ParaView-style: spacing=9)
    QHBoxLayout* elementTypeLayout = new QHBoxLayout();
    elementTypeLayout->setSpacing(9);
    m_elementTypeLabel = new QLabel(tr("Element Type"));
    m_elementTypeLabel->setSizePolicy(QSizePolicy::Maximum,
                                      QSizePolicy::Preferred);
    m_elementTypeLabel->setToolTip(
            tr("The element type of the saved selections"));
    m_elementTypeValue = new QLabel();
    m_elementTypeValue->setStyleSheet(
            "QLabel { background-color: snow; border: 1px inset grey; "
            "}");  // ParaView exact style
    m_elementTypeValue->setToolTip(
            tr("The element type of the saved selections"));
    elementTypeLayout->addWidget(m_elementTypeLabel);
    elementTypeLayout->addWidget(m_elementTypeValue, 1);
    editorLayout->addLayout(elementTypeLayout);

    // Expression row (ParaView-style: spacing=29)
    QHBoxLayout* expressionLayout = new QHBoxLayout();
    expressionLayout->setSpacing(29);
    m_expressionLabel = new QLabel(tr("Expression"));
    m_expressionLabel->setToolTip(
            tr("Specify the expression which defines the relation between "
               "saved selections using boolean operators: !(NOT), &(AND), "
               "|(OR), ^(XOR) and ()."));
    m_expressionEdit = new QLineEdit();
    m_expressionEdit->setPlaceholderText(
            tr("e.g., (s0|s1)&s2|(s3&s4)|s5|s6|s7"));
    m_expressionEdit->setToolTip(
            tr("Specify the expression which defines the relation between "
               "saved selections using boolean operators: !(NOT), &(AND), "
               "|(OR), ^(XOR) and ()."));
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
    m_selectionEditorTable->setHorizontalHeaderLabels(
            {tr("Name"), tr("Type"), tr("Color")});
    m_selectionEditorTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_selectionEditorTable->setSelectionMode(
            QAbstractItemView::ExtendedSelection);
    m_selectionEditorTable->setAlternatingRowColors(true);
    m_selectionEditorTable->horizontalHeader()->setStretchLastSection(true);
    m_selectionEditorTable->verticalHeader()->setVisible(false);
    m_selectionEditorTable->setMinimumHeight(120);
    connect(m_selectionEditorTable, &QTableWidget::itemSelectionChanged, this,
            &cvSelectionPropertiesWidget::
                    onSelectionEditorTableSelectionChanged);
    // Handle cell clicks for color editing and row selection highlighting
    connect(m_selectionEditorTable, &QTableWidget::cellClicked, this,
            &cvSelectionPropertiesWidget::onSelectionEditorCellClicked);
    connect(m_selectionEditorTable, &QTableWidget::cellDoubleClicked, this,
            &cvSelectionPropertiesWidget::onSelectionEditorCellDoubleClicked);
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
        addIcon = QIcon(":/Resources/images/ecvPlus.png");  // Final fallback
    }
    m_addSelectionButton->setIcon(addIcon);
    m_addSelectionButton->setToolTip(tr("Add active selection"));
    m_addSelectionButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_addSelectionButton->setEnabled(
            false);  // Initially disabled until selection exists
    connect(m_addSelectionButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onAddActiveSelectionClicked);
    toolbarLayout->addWidget(m_addSelectionButton);

    // Remove selected selection button (ParaView: pqMinus.svg/png)
    m_removeSelectionButton = new QToolButton();
    QIcon removeIcon(":/Resources/images/svg/pqMinus.svg");
    if (removeIcon.isNull()) {
        removeIcon = QIcon(":/Resources/images/ecvMinus.png");  // Fallback
    }
    m_removeSelectionButton->setIcon(removeIcon);
    m_removeSelectionButton->setToolTip(
            tr("Remove selected selection from the saved selections. Remember "
               "to edit the Expression."));
    m_removeSelectionButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_removeSelectionButton->setEnabled(false);
    connect(m_removeSelectionButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onRemoveSelectedSelectionClicked);
    toolbarLayout->addWidget(m_removeSelectionButton);

    // Vertical spacer between buttons (ParaView-style)
    toolbarLayout->addStretch();

    // Remove all selections button (ParaView: pqDelete.svg - using
    // smallTrash.png as alternative)
    m_removeAllSelectionsButton = new QToolButton();
    QIcon trashIcon(":/Resources/images/smallTrash.png");
    if (trashIcon.isNull()) {
        trashIcon = QIcon(":/Resources/images/ecvdelete.png");  // Fallback
    }
    m_removeAllSelectionsButton->setIcon(trashIcon);
    m_removeAllSelectionsButton->setToolTip(tr("Remove all saved selections"));
    m_removeAllSelectionsButton->setIconSize(
            QSize(16, 16));  // Ensure visible size
    m_removeAllSelectionsButton->setEnabled(false);
    connect(m_removeAllSelectionsButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onRemoveAllSelectionsClicked);
    toolbarLayout->addWidget(m_removeAllSelectionsButton);

    tableLayout->addLayout(toolbarLayout);
    editorLayout->addLayout(tableLayout);

    // Activate Combined Selections button (ParaView: pqApply.svg icon)
    QHBoxLayout* activateLayout = new QHBoxLayout();
    activateLayout->setSpacing(2);
    m_activateCombinedSelectionsButton =
            new QPushButton(tr("Activate Combined Selections"));
    m_activateCombinedSelectionsButton->setIcon(
            QIcon(":/Resources/images/smallValidate.png"));
    m_activateCombinedSelectionsButton->setToolTip(
            tr("Set the combined saved selections as the active selection"));
    m_activateCombinedSelectionsButton->setFocusPolicy(Qt::TabFocus);
    m_activateCombinedSelectionsButton->setDefault(true);
    m_activateCombinedSelectionsButton->setEnabled(false);
    connect(m_activateCombinedSelectionsButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onActivateCombinedSelectionsClicked);
    activateLayout->addWidget(m_activateCombinedSelectionsButton);
    editorLayout->addLayout(activateLayout);

    // No need for setupCollapsibleGroupBox - handled by cvExpanderButton
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupSelectedDataSpreadsheet() {
    // Uses m_selectedDataSpreadsheetContainer instead of QGroupBox for
    // ParaView-style collapsible behavior with cvExpanderButton
    m_selectedDataGroup = nullptr;  // Not using QGroupBox anymore

    QGridLayout* dataLayout =
            new QGridLayout(m_selectedDataSpreadsheetContainer);
    dataLayout->setSpacing(3);  // ParaView: spacing=3
    dataLayout->setContentsMargins(8, 8, 8, 8);

    // Row 0: Attribute label, combo, buttons, checkbox (ParaView:
    // columnstretch="0,1,0") Column 0: Attribute label
    QLabel* attributeLabel = new QLabel(tr(
            "<html><body><p><span "
            "style=\"font-weight:600;\">Attribute:</span></p></body></html>"));
    attributeLabel->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Preferred);
    dataLayout->addWidget(attributeLabel, 0, 0);

    // Column 1: Attribute combo box (ParaView-style with icons)
    m_attributeTypeCombo = new QComboBox();

    // Point Data with icon (ParaView: pqPointData.svg)
    QIcon pointDataIcon(":/Resources/images/svg/pqPointData.svg");
    m_attributeTypeCombo->addItem(pointDataIcon, tr("Point Data"), 0);

    // Cell Data with icon (ParaView: pqCellData.svg)
    QIcon cellDataIcon(":/Resources/images/svg/pqCellData.svg");
    m_attributeTypeCombo->addItem(cellDataIcon, tr("Cell Data"), 1);

    m_attributeTypeCombo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    m_attributeTypeCombo->setIconSize(
            QSize(16, 16));  // Ensure icons are visible
    connect(m_attributeTypeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &cvSelectionPropertiesWidget::onAttributeTypeChanged);
    dataLayout->addWidget(m_attributeTypeCombo, 0, 1);

    // Column 2: Toggle column visibility button (ParaView:
    // pqRectilinearGrid16.png)
    m_toggleColumnVisibilityButton = new QToolButton();
    QIcon colVisIcon(":/Resources/images/interactors.png");
    if (colVisIcon.isNull()) {
        colVisIcon = QIcon(":/Resources/images/settings.png");  // Fallback
    }
    m_toggleColumnVisibilityButton->setIcon(colVisIcon);
    m_toggleColumnVisibilityButton->setToolTip(tr("Toggle column visibility"));
    m_toggleColumnVisibilityButton->setStatusTip(
            tr("Toggle column visibility"));
    m_toggleColumnVisibilityButton->setIconSize(
            QSize(16, 16));  // Ensure visible size
    m_toggleColumnVisibilityButton->setPopupMode(QToolButton::InstantPopup);
    connect(m_toggleColumnVisibilityButton, &QToolButton::clicked, this,
            &cvSelectionPropertiesWidget::onToggleColumnVisibility);
    dataLayout->addWidget(m_toggleColumnVisibilityButton, 0, 2);

    // Column 3: Toggle field data button (ParaView: pqGlobalData.svg)
    m_toggleFieldDataButton = new QToolButton();
    QIcon fieldDataIcon(":/Resources/images/svg/pqGlobalData.svg");
    m_toggleFieldDataButton->setIcon(fieldDataIcon);
    m_toggleFieldDataButton->setToolTip(
            tr("Toggle field data visibility (show raw data arrays)"));
    m_toggleFieldDataButton->setIconSize(QSize(16, 16));  // Ensure visible size
    m_toggleFieldDataButton->setCheckable(true);
    connect(m_toggleFieldDataButton, &QToolButton::toggled, this,
            &cvSelectionPropertiesWidget::onToggleFieldDataClicked);
    dataLayout->addWidget(m_toggleFieldDataButton, 0, 3);

    // Column 4: Invert selection checkbox
    m_invertSelectionCheck = new QCheckBox(tr("Invert selection"));
    m_invertSelectionCheck->setToolTip(tr("Invert the selection"));
    m_invertSelectionCheck->setEnabled(
            false);  // ParaView: enabled=false by default
    connect(m_invertSelectionCheck, &QCheckBox::toggled, this,
            &cvSelectionPropertiesWidget::onInvertSelectionToggled);
    dataLayout->addWidget(m_invertSelectionCheck, 0, 4);

    // Row 2: Spreadsheet table (ParaView: pqSpreadSheetViewWidget, spans all 5
    // columns)
    m_spreadsheetTable = new QTableWidget();
    m_spreadsheetTable->setSizePolicy(QSizePolicy::Preferred,
                                      QSizePolicy::MinimumExpanding);
    m_spreadsheetTable->setMinimumHeight(120);  // ParaView: minimumHeight=120
    m_spreadsheetTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_spreadsheetTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_spreadsheetTable->setAlternatingRowColors(true);
    m_spreadsheetTable->setEditTriggers(QAbstractItemView::NoEditTriggers);

    // ParaView-style: Use cvMultiColumnHeaderView for header merging
    // This will automatically merge adjacent columns with the same DisplayRole
    // text
    cvMultiColumnHeaderView* multiColHeader =
            new cvMultiColumnHeaderView(Qt::Horizontal, m_spreadsheetTable);
    multiColHeader->setStretchLastSection(true);
    multiColHeader->setDefaultSectionSize(100);
    m_spreadsheetTable->setHorizontalHeader(multiColHeader);

    m_spreadsheetTable->verticalHeader()->setDefaultSectionSize(20);
    // ParaView-style: Remove bold from header font
    QFont headerFont = m_spreadsheetTable->horizontalHeader()->font();
    headerFont.setBold(false);
    m_spreadsheetTable->horizontalHeader()->setFont(headerFont);
    connect(m_spreadsheetTable, &QTableWidget::itemClicked, this,
            &cvSelectionPropertiesWidget::onSpreadsheetItemClicked);
    dataLayout->addWidget(m_spreadsheetTable, 2, 0, 1,
                          5);  // Row 2, spanning all 5 columns

    // Row 3: Action buttons (ParaView-style: Freeze | Extract | Plot Over Time)
    // These buttons span all 5 columns
    QHBoxLayout* actionLayout = new QHBoxLayout();
    actionLayout->setSpacing(3);

    // Freeze button (ParaView style)
    actionLayout->addWidget(m_freezeButton);

    // Extract button (ParaView style)
    actionLayout->addWidget(m_extractButton);

    // Plot Over Time button (ParaView style)
    m_plotOverTimeButton = new QPushButton(tr("Plot Over Time"));
    m_plotOverTimeButton->setToolTip(tr("Plot selection over time"));
    m_plotOverTimeButton->setEnabled(false);  // Enabled when selection exists
    connect(m_plotOverTimeButton, &QPushButton::clicked, this,
            &cvSelectionPropertiesWidget::onPlotOverTimeClicked);
    actionLayout->addWidget(m_plotOverTimeButton);

    dataLayout->addLayout(actionLayout, 3, 0, 1,
                          5);  // Row 3, spanning all 5 columns

    // Layout is already set on m_selectedDataSpreadsheetContainer
    // No need for setupCollapsibleGroupBox - handled by cvExpanderButton
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setSelectionManager(
        cvViewSelectionManager* manager) {
    m_selectionManager = manager;

    // Update bookmark combo when manager is set
    // Bookmark functionality removed - UI not implemented

    // Update data producer combo
    updateDataProducerCombo();

    // Always use the manager's shared highlighter
    // This ensures all tools (including tooltip tools) share the same
    // highlighter so color settings are automatically synchronized
    if (m_selectionManager) {
        cvSelectionHighlighter* sharedHighlighter =
                m_selectionManager->getHighlighter();
        if (sharedHighlighter && sharedHighlighter != m_highlighter) {
            setHighlighter(sharedHighlighter);
        }

        // Initialize default label properties for annotations
        // Get properties from highlighter (single source of truth)
        cvSelectionAnnotationManager* annotations =
                m_selectionManager->getAnnotations();
        if (annotations && m_highlighter) {
            // Convert SelectionLabelProperties to LabelProperties for
            // annotations
            const SelectionLabelProperties& selProps =
                    m_highlighter->getLabelProperties(false);
            cvSelectionLabelPropertiesDialog::LabelProperties labelProps;
            labelProps.opacity = selProps.opacity;
            labelProps.pointSize = selProps.pointSize;
            labelProps.lineWidth = selProps.lineWidth;
            labelProps.cellLabelFontFamily = selProps.cellLabelFontFamily;
            labelProps.cellLabelFontSize = selProps.cellLabelFontSize;
            labelProps.cellLabelColor = selProps.cellLabelColor;
            annotations->setDefaultLabelProperties(labelProps,
                                                   true);  // cell labels

            const SelectionLabelProperties& interProps =
                    m_highlighter->getLabelProperties(true);
            cvSelectionLabelPropertiesDialog::LabelProperties interLabelProps;
            interLabelProps.opacity = interProps.opacity;
            interLabelProps.pointSize = interProps.pointSize;
            interLabelProps.lineWidth = interProps.lineWidth;
            interLabelProps.pointLabelFontFamily =
                    interProps.pointLabelFontFamily;
            interLabelProps.pointLabelFontSize = interProps.pointLabelFontSize;
            interLabelProps.pointLabelColor = interProps.pointLabelColor;
            annotations->setDefaultLabelProperties(interLabelProps,
                                                   false);  // point labels
        }
    }
}

// updateBookmarkCombo removed - UI not implemented

//-----------------------------------------------------------------------------
bool cvSelectionPropertiesWidget::updateSelection(
        const cvSelectionData& selectionData, vtkPolyData* polyData) {
    m_selectionData = selectionData;

    // ParaView-style: Only clear original selection for truly NEW selections
    // A new selection is determined by comparing IDs
    bool isNewSelection = m_originalSelectionIds.isEmpty() ||
                          m_originalSelectionIds != selectionData.ids();

    if (isNewSelection) {
        // This is a brand new selection from user interaction
        // Clear original selection and reset invert checkbox
        m_originalSelectionIds.clear();
        if (m_invertSelectionCheck) {
            m_invertSelectionCheck->blockSignals(true);
            m_invertSelectionCheck->setChecked(false);
            m_invertSelectionCheck->blockSignals(false);
        }
    }
    // If not a new selection, preserve m_originalSelectionIds and checkbox
    // state

    if (m_selectionData.isEmpty()) {
        clearSelection();
        return false;
    }

    // Enable invert selection checkbox when there's an active selection
    if (m_invertSelectionCheck) {
        m_invertSelectionCheck->setEnabled(true);
    }

    // ParaView behavior: Auto-update element type combo based on selection type
    // This ensures the UI reflects the current selection's field association
    if (m_elementTypeCombo) {
        int expectedIndex =
                (m_selectionData.fieldAssociation() == cvSelectionData::CELLS)
                        ? 1
                        : 0;  // 0 = Point, 1 = Cell
        if (m_elementTypeCombo->currentIndex() != expectedIndex) {
            // Block signals to prevent recursive updates
            m_elementTypeCombo->blockSignals(true);
            m_elementTypeCombo->setCurrentIndex(expectedIndex);
            m_elementTypeCombo->blockSignals(false);
            CVLog::PrintVerbose(
                    QString("[cvSelectionPropertiesWidget] Auto-switched "
                            "element type to %1")
                            .arg(expectedIndex == 0 ? "Point" : "Cell"));
        }
    }

    // Update Selection Editor UI with current selection info (ParaView style)
    // Update element type value label
    if (m_elementTypeValue) {
        QString elementTypeStr =
                (m_selectionData.fieldAssociation() == cvSelectionData::CELLS)
                        ? tr("Cell")
                        : tr("Point");
        m_elementTypeValue->setText(elementTypeStr);
    }

    // Update Data Producer from source object name (ParaView style)
    cvViewSelectionManager* manager = cvViewSelectionManager::instance();
    if (manager) {
        ccHObject* sourceObj = manager->getSourceObject();
        if (sourceObj) {
            QString sourceName = sourceObj->getName();
            setDataProducerName(sourceName);

            // Also update the Data Producer combo selection if exists
            if (m_dataProducerCombo) {
                int comboIndex = m_dataProducerCombo->findText(sourceName);
                if (comboIndex >= 0) {
                    m_dataProducerCombo->blockSignals(true);
                    m_dataProducerCombo->setCurrentIndex(comboIndex);
                    m_dataProducerCombo->blockSignals(false);
                    // IMPORTANT: Manually trigger enable/disable logic since we
                    // blocked signals
                    onDataProducerChanged(comboIndex);
                }
            }

            CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] Updated "
                                        "Data Producer to '%1'")
                                        .arg(sourceName));
        }
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
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] polyData validation failed");
        return false;
    }

    // Update statistics and list
    updateStatistics(polyData);
    updateSelectionList(polyData);

    // Update spreadsheet data (ParaView-style: auto-update on selection change)
    updateSpreadsheetData(polyData);

    // Enable export buttons
    bool isCells =
            (m_selectionData.fieldAssociation() == cvSelectionData::CELLS);
    bool isPoints =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);

    // Enable legacy advanced tab buttons (may be nullptr in simplified UI)
    if (m_applyAlgebraButton) {
        m_applyAlgebraButton->setEnabled(m_selectionCount > 0);
    }
    if (m_extractBoundaryButton) {
        m_extractBoundaryButton->setEnabled(isCells && m_selectionCount > 0);
    }
    // Filter button removed - UI not implemented
    // Bookmark button removed - UI not implemented
    if (m_addAnnotationButton) {
        m_addAnnotationButton->setEnabled(m_selectionCount > 0);
    }

    // ParaView behavior: Enable the + button when a new selection is made
    // This allows the user to add the new selection to the saved selections
    if (m_addSelectionButton && m_selectionCount > 0) {
        m_addSelectionButton->setEnabled(true);
    }

    // Enable header action buttons (ParaView-style)
    if (m_freezeButton) {
        m_freezeButton->setEnabled(m_selectionCount > 0);
    }
    if (m_extractButton) {
        m_extractButton->setEnabled(m_selectionCount > 0);
    }
    if (m_plotOverTimeButton) {
        m_plotOverTimeButton->setEnabled(m_selectionCount > 0);
    }

    // Enable Selection Editor + button when there's an active selection
    // (ParaView behavior: can only add selection when one exists)
    if (m_addSelectionButton) {
        m_addSelectionButton->setEnabled(m_selectionCount > 0);
    }

    return true;
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::clearSelection() {
    m_selectionData.clear();
    m_selectionCount = 0;
    m_selectionType = tr("None");

    // Clear original selection IDs for invert toggle
    m_originalSelectionIds.clear();

    // Disable and reset invert selection checkbox when selection is cleared
    if (m_invertSelectionCheck) {
        m_invertSelectionCheck->blockSignals(true);
        m_invertSelectionCheck->setChecked(false);
        m_invertSelectionCheck->setEnabled(false);
        m_invertSelectionCheck->blockSignals(false);
    }

    // CRITICAL: Clear labels from the 3D scene when selection is cleared
    if (m_highlighter) {
        // Clear point labels
        if (m_highlighter->isPointLabelVisible()) {
            m_highlighter->setPointLabelArray(
                    "", false);  // Empty array name + hide
        }
        // Clear cell labels
        if (m_highlighter->isCellLabelVisible()) {
            m_highlighter->setCellLabelArray("",
                                             false);  // Empty array name + hide
        }
    }

    // Clear statistics (may be nullptr in simplified UI)
    if (m_countLabel) {
        m_countLabel->setText(QString::number(m_selectionCount));
    }
    if (m_typeLabel) {
        m_typeLabel->setText(m_selectionType);
    }
    if (m_boundsLabel) {
        m_boundsLabel->setText(tr("N/A"));
    }
    if (m_centerLabel) {
        m_centerLabel->setText(tr("N/A"));
    }
    if (m_volumeLabel) {
        m_volumeLabel->setText(tr("N/A"));
    }

    // Clear legacy table (may be nullptr in simplified UI)
    if (m_selectionTableWidget) {
        m_selectionTableWidget->clear();
        m_selectionTableWidget->setRowCount(0);
        m_selectionTableWidget->setColumnCount(0);
    }
    if (m_listInfoLabel) {
        m_listInfoLabel->setText(tr("No selection"));
        m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");
    }

    // Clear spreadsheet table (ParaView-style Selected Data)
    if (m_spreadsheetTable) {
        m_spreadsheetTable->clear();
        m_spreadsheetTable->setRowCount(0);
        m_spreadsheetTable->setColumnCount(0);
    }

    // Disable header action buttons
    if (m_freezeButton) {
        m_freezeButton->setEnabled(false);
    }
    if (m_extractButton) {
        m_extractButton->setEnabled(false);
    }
    if (m_plotOverTimeButton) {
        m_plotOverTimeButton->setEnabled(false);
    }

    // Disable Selection Editor + button when no selection
    if (m_addSelectionButton) {
        m_addSelectionButton->setEnabled(false);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateStatistics(
        vtkPolyData* polyData, const cvSelectionData* customSelection) {
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget::updateStatistics] polyData is "
                "nullptr");
        return;
    }

    // Validate polyData is still valid by checking basic properties
    // This helps catch use-after-free issues
    try {
        if (polyData->GetNumberOfPoints() < 0 ||
            polyData->GetNumberOfCells() < 0) {
            CVLog::Warning(
                    "[cvSelectionPropertiesWidget::updateStatistics] Invalid "
                    "polyData state");
            return;
        }
    } catch (...) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget::updateStatistics] polyData "
                "access failed - possible dangling pointer");
        return;
    }

    // Use custom selection if provided, otherwise use m_selectionData
    const cvSelectionData& selection =
            customSelection ? *customSelection : m_selectionData;

    m_selectionCount = selection.count();
    m_selectionType = selection.fieldTypeString();

    // Update labels (may be nullptr in simplified UI)
    if (m_countLabel) {
        m_countLabel->setText(QString::number(m_selectionCount));
    }
    if (m_typeLabel) {
        m_typeLabel->setText(m_selectionType);
    }

    // Compute bounding box
    if (m_selectionCount > 0) {
        computeBoundingBox(polyData, m_bounds);

        // Bounds
        if (m_boundsLabel) {
            m_boundsLabel->setText(formatBounds(m_bounds));
        }

        // Center
        m_center[0] = (m_bounds[0] + m_bounds[1]) / 2.0;
        m_center[1] = (m_bounds[2] + m_bounds[3]) / 2.0;
        m_center[2] = (m_bounds[4] + m_bounds[5]) / 2.0;
        if (m_centerLabel) {
            m_centerLabel->setText(QString("(%1, %2, %3)")
                                           .arg(m_center[0], 0, 'g', 6)
                                           .arg(m_center[1], 0, 'g', 6)
                                           .arg(m_center[2], 0, 'g', 6));
        }

        // Volume
        double dx = m_bounds[1] - m_bounds[0];
        double dy = m_bounds[3] - m_bounds[2];
        double dz = m_bounds[5] - m_bounds[4];
        m_volume = dx * dy * dz;
        if (m_volumeLabel) {
            m_volumeLabel->setText(QString("%1").arg(m_volume, 0, 'g', 6));
        }
    } else {
        if (m_boundsLabel) {
            m_boundsLabel->setText(tr("N/A"));
        }
        if (m_centerLabel) {
            m_centerLabel->setText(tr("N/A"));
        }
        if (m_volumeLabel) {
            m_volumeLabel->setText(tr("N/A"));
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateSelectionList(vtkPolyData* polyData) {
    // Legacy UI elements check - these may be nullptr in simplified UI
    if (!m_selectionTableWidget) {
        return;
    }

    m_selectionTableWidget->clear();
    m_selectionTableWidget->setRowCount(0);

    QVector<qint64> ids = m_selectionData.ids();
    if (ids.isEmpty()) {
        m_selectionTableWidget->setColumnCount(0);
        if (m_listInfoLabel) {
            m_listInfoLabel->setText(tr("No selection"));
            m_listInfoLabel->setStyleSheet("font-style: italic; color: gray;");
        }
        return;
    }

    bool isPoints =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);

    // Update info label (ParaView-style: "Selected Data (source.ply)")
    if (m_listInfoLabel) {
        m_listInfoLabel->setText(
                tr("Showing %1 %2")
                        .arg(ids.size())
                        .arg(m_selectionData.fieldTypeString().toLower()));
        m_listInfoLabel->setStyleSheet("font-weight: bold;");
    }

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
        idItem->setData(
                Qt::UserRole,
                QVariant::fromValue(id));  // Store ID for click handling
        m_selectionTableWidget->setItem(row, col++, idItem);

        if (isPoints && polyData) {
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                double pt[3];
                polyData->GetPoint(id, pt);

                // X, Y, Z columns
                m_selectionTableWidget->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(pt[0], 'g', 6)));
                m_selectionTableWidget->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(pt[1], 'g', 6)));
                m_selectionTableWidget->setItem(
                        row, col++,
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
                                m_selectionTableWidget->setItem(
                                        row, col++,
                                        new QTableWidgetItem(
                                                QString::number(val, 'g', 6)));
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
                    m_selectionTableWidget->setItem(
                            row, col++,
                            new QTableWidgetItem(
                                    QString::number(cell->GetCellType())));
                    // Number of points
                    m_selectionTableWidget->setItem(
                            row, col++,
                            new QTableWidgetItem(QString::number(
                                    cell->GetNumberOfPoints())));

                    // Additional cell attributes
                    if (polyData->GetCellData()) {
                        vtkCellData* cd = polyData->GetCellData();
                        for (int i = 0; i < cd->GetNumberOfArrays(); ++i) {
                            vtkDataArray* arr = cd->GetArray(i);
                            if (arr && arr->GetName()) {
                                double val = arr->GetComponent(id, 0);
                                m_selectionTableWidget->setItem(
                                        row, col++,
                                        new QTableWidgetItem(
                                                QString::number(val, 'g', 6)));
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
    if (ids.size() > maxDisplay && m_listInfoLabel) {
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
    // Get current color from highlighter (single source of truth)
    cvSelectionHighlighter::HighlightMode hlMode =
            static_cast<cvSelectionHighlighter::HighlightMode>(mode);

    QColor initialColor;
    if (m_highlighter) {
        initialColor = m_highlighter->getHighlightQColor(hlMode);
    } else {
        // Fallback if highlighter not set yet
        initialColor =
                QColor(int(currentColor[0] * 255), int(currentColor[1] * 255),
                       int(currentColor[2] * 255));
    }

    QColor newColor = QColorDialog::getColor(initialColor, this, title);

    if (newColor.isValid()) {
        // Update highlighter directly (single source of truth)
        if (m_highlighter) {
            m_highlighter->setHighlightQColor(newColor, hlMode);
        }

        // UI will be updated via colorChanged signal from highlighter
        // No need to update currentColor[] or buttons here

        // Emit signal for any additional processing
        emit highlightColorChanged(newColor.redF(), newColor.greenF(),
                                   newColor.blueF(), mode);
    }
}

//-----------------------------------------------------------------------------
// Slot implementations
//-----------------------------------------------------------------------------

void cvSelectionPropertiesWidget::onHoverColorClicked() {
    double dummy[3] = {0, 1, 1};  // Fallback value (cyan)
    showColorDialog(tr("Select Hover Highlight Color"), dummy,
                    cvSelectionHighlighter::HOVER);
}

void cvSelectionPropertiesWidget::onPreselectedColorClicked() {
    double dummy[3] = {1, 1, 0};  // Fallback value (yellow)
    showColorDialog(tr("Select Pre-selected Highlight Color"), dummy,
                    cvSelectionHighlighter::PRESELECTED);
}

void cvSelectionPropertiesWidget::onSelectedColorClicked() {
    double dummy[3] = {1, 0, 1};  // Fallback value (magenta)
    showColorDialog(tr("Select Selected Highlight Color"), dummy,
                    cvSelectionHighlighter::SELECTED);
}

void cvSelectionPropertiesWidget::onBoundaryColorClicked() {
    double dummy[3] = {1, 0.65, 0};  // Fallback value (orange)
    showColorDialog(tr("Select Boundary Highlight Color"), dummy,
                    cvSelectionHighlighter::BOUNDARY);
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
        QMessageBox::warning(this, tr("Export Failed"),
                             tr("No selection to export. Please select some "
                                "cells first."));
        return;
    }

    if (m_selectionData.fieldAssociation() != cvSelectionData::CELLS) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Can only export cells as mesh");
        QMessageBox::warning(this, tr("Export Failed"),
                             tr("Can only export cell selections as mesh. "
                                "Current selection is points."));
        return;
    }

    // First, try to use direct extraction from source ccMesh if available
    // This bypasses VTK→ccMesh conversion and preserves all attributes
    if (m_selectionManager && m_selectionManager->isSourceObjectValid()) {
        ccMesh* sourceMesh = m_selectionManager->getSourceMesh();

        if (sourceMesh) {
            // CVLog::Print(QString("[cvSelectionPropertiesWidget] Using direct
            // "
            //                      "extraction from source mesh '%1'")
            //                      .arg(sourceMesh->getName()));

            cvSelectionExporter::ExportOptions options;
            ccMesh* mesh = nullptr;

            try {
                mesh = cvSelectionExporter::exportFromSourceMesh(
                        sourceMesh, m_selectionData, options);
            } catch (const std::exception& e) {
                CVLog::Error(QString("[cvSelectionPropertiesWidget] Exception "
                                     "during direct export: %1")
                                     .arg(e.what()));
                // Fall through to VTK-based extraction
            } catch (...) {
                CVLog::Error(
                        "[cvSelectionPropertiesWidget] Unknown exception "
                        "during direct export");
                // Fall through to VTK-based extraction
            }

            if (mesh && mesh->size() > 0) {
                // ParaView-style naming: ExtractSelection1, ExtractSelection2,
                // ...
                m_extractCounter++;
                QString meshName =
                        QString("ExtractSelection%1").arg(m_extractCounter);
                mesh->setName(meshName);
                // ParaView behavior: Extract objects are hidden by default
                // This prevents them from blocking selection of the original
                // object
                mesh->setEnabled(false);
                mesh->setVisible(true);
                emit extractedObjectReady(mesh);
                emit extractSelectionRequested();
                return;
            } else {
                if (mesh) {
                    delete mesh;
                    mesh = nullptr;
                }
                CVLog::Warning(
                        "[cvSelectionPropertiesWidget] Direct "
                        "extraction failed, falling back to VTK-based "
                        "extraction");
            }
        }
    }

    // Fall back to VTK-based extraction if direct extraction not available
    CVLog::Print(
            "[cvSelectionPropertiesWidget] Using VTK-based mesh extraction");

    // Export selection to mesh
    // Get polyData (using centralized ParaView-style method)
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to get polyData from "
                "visualizer");
        QMessageBox::critical(this, tr("Export Failed"),
                              tr("Failed to get data from visualizer. Please "
                                 "ensure a mesh is loaded."));
        return;
    }

    // Validate that polyData has enough cells for the selection
    vtkIdType polyDataCells = polyData->GetNumberOfCells();
    QVector<qint64> selectionIds = m_selectionData.ids();
    bool hasValidIds = false;
    for (qint64 id : selectionIds) {
        if (id >= 0 && id < polyDataCells) {
            hasValidIds = true;
            break;
        }
    }

    if (!hasValidIds) {
        CVLog::Error(QString("[cvSelectionPropertiesWidget] Selection IDs "
                             "(%1 items) are not valid for polyData (%2 cells)")
                             .arg(selectionIds.size())
                             .arg(polyDataCells));
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("Selection IDs are not valid for the current data. "
                   "The selection may have been made on a different object."));
        return;
    }

    CVLog::Print(
            "[cvSelectionPropertiesWidget] Creating mesh from selection...");

    cvSelectionExporter::ExportOptions options;
    ccMesh* mesh = nullptr;

    try {
        mesh = cvSelectionExporter::exportToMesh(polyData, m_selectionData,
                                                 options);
    } catch (const std::exception& e) {
        CVLog::Error(QString("[cvSelectionPropertiesWidget] Exception during "
                             "export: %1")
                             .arg(e.what()));
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("An error occurred during export: %1").arg(e.what()));
        return;
    } catch (...) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Unknown exception during "
                "export");
        QMessageBox::critical(this, tr("Export Failed"),
                              tr("An unknown error occurred during export."));
        return;
    }

    if (mesh && mesh->size() > 0) {
        // ParaView-style naming: ExtractSelection1, ExtractSelection2, ...
        m_extractCounter++;
        QString meshName = QString("ExtractSelection%1").arg(m_extractCounter);
        mesh->setName(meshName);
        // ParaView behavior: Extract objects are hidden by default
        // This prevents them from blocking selection of the original object
        mesh->setEnabled(false);
        mesh->setVisible(true);

        // Emit signal for main application to add to DB Tree
        // The main application should connect to this signal and call addToDB()
        emit extractedObjectReady(mesh);
        emit extractSelectionRequested();
    } else {
        // Clean up empty mesh if created
        if (mesh) {
            delete mesh;
            mesh = nullptr;
        }
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to export selection as "
                "mesh - no cells extracted");
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("Failed to export selection. No cells could be extracted "
                   "from the selection."));
    }
}

void cvSelectionPropertiesWidget::onExportToPointCloudClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to export");
        QMessageBox::warning(this, tr("Export Failed"),
                             tr("No selection to export. Please select some "
                                "points first."));
        return;
    }

    // First, try to use direct extraction from source ccPointCloud if available
    // This bypasses VTK→ccPointCloud conversion and preserves all attributes
    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Extract point cloud check: "
                    "manager=%1, sourceValid=%2")
                    .arg(m_selectionManager != nullptr ? "yes" : "no")
                    .arg(m_selectionManager
                                 ? (m_selectionManager->isSourceObjectValid()
                                            ? "yes"
                                            : "no")
                                 : "N/A"));

    if (m_selectionManager && m_selectionManager->isSourceObjectValid()) {
        ccPointCloud* sourceCloud = m_selectionManager->getSourcePointCloud();
        CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] "
                                    "getSourcePointCloud returned: %1")
                                    .arg(sourceCloud != nullptr
                                                 ? sourceCloud->getName()
                                                 : "nullptr"));
        if (sourceCloud) {
            // For cell selection on point cloud, convert to point selection
            cvSelectionData exportSelection = m_selectionData;
            if (m_selectionData.fieldAssociation() == cvSelectionData::CELLS) {
                // Converting cell selection to point selection
                exportSelection = cvSelectionData(m_selectionData.ids(),
                                                  cvSelectionData::POINTS);
            }

            cvSelectionExporter::ExportOptions options;
            ccPointCloud* cloud = nullptr;

            try {
                cloud = cvSelectionExporter::exportFromSourceCloud(
                        sourceCloud, exportSelection, options);
            } catch (const std::exception& e) {
                CVLog::Error(QString("[cvSelectionPropertiesWidget] Exception "
                                     "during direct export: %1")
                                     .arg(e.what()));
                // Fall through to VTK-based extraction
            } catch (...) {
                CVLog::Error(
                        "[cvSelectionPropertiesWidget] Unknown exception "
                        "during direct export");
                // Fall through to VTK-based extraction
            }

            if (cloud && cloud->size() > 0) {
                // ParaView-style naming: ExtractSelection1, ExtractSelection2,
                // ...
                m_extractCounter++;
                QString cloudName =
                        QString("ExtractSelection%1").arg(m_extractCounter);
                cloud->setName(cloudName);
                // ParaView behavior: Extract objects are hidden by default
                // This prevents them from blocking selection of the original
                // object
                cloud->setEnabled(false);
                cloud->setVisible(true);

                CVLog::Print(QString("[cvSelectionPropertiesWidget] Created "
                                     "point cloud '%1' with %2 points (direct "
                                     "extraction)")
                                     .arg(cloudName)
                                     .arg(cloud->size()));

                emit extractedObjectReady(cloud);
                emit extractSelectionRequested();
                return;
            } else {
                if (cloud) {
                    delete cloud;
                    cloud = nullptr;
                }
                CVLog::Warning(
                        "[cvSelectionPropertiesWidget] Direct "
                        "extraction failed, falling back to VTK-based "
                        "extraction");
            }
        }
    }

    // Fall back to VTK-based extraction if direct extraction not available

    // ParaView behavior: Allow both POINTS and CELLS selections for point cloud
    // export For point clouds, cell IDs ARE point IDs (each vertex is a cell)
    // Check if source is a point cloud (no polygons)
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    bool isSourcePointCloud = polyData && (polyData->GetNumberOfPolys() == 0);

    if (m_selectionData.fieldAssociation() != cvSelectionData::POINTS &&
        !isSourcePointCloud) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Can only export points as point "
                "cloud");
        QMessageBox::warning(
                this, tr("Export Failed"),
                tr("Can only export point selections as point "
                   "cloud. Current selection is cells on a mesh."));
        return;
    }

    // Export selection to point cloud
    // Get polyData (using centralized ParaView-style method)

    if (!polyData) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to get polyData from "
                "visualizer");
        QMessageBox::critical(this, tr("Export Failed"),
                              tr("Failed to get data from visualizer. Please "
                                 "ensure a point cloud is loaded."));
        return;
    }

    // Determine if this is a cell selection on a point cloud
    // For point clouds, cell IDs ARE point IDs (each vertex is a cell)
    bool isCellSelectionOnPointCloud =
            (m_selectionData.fieldAssociation() == cvSelectionData::CELLS) &&
            (polyData->GetNumberOfPolys() == 0);

    // Validate that polyData has enough elements for the selection
    vtkIdType maxId = isCellSelectionOnPointCloud
                              ? polyData->GetNumberOfCells()
                              : polyData->GetNumberOfPoints();
    QVector<qint64> selectionIds = m_selectionData.ids();
    bool hasValidIds = false;
    int validCount = 0;
    int invalidCount = 0;
    for (qint64 id : selectionIds) {
        if (id >= 0 && id < maxId) {
            hasValidIds = true;
            validCount++;
        } else {
            invalidCount++;
        }
    }

    // CVLog::Print(QString("[cvSelectionPropertiesWidget] Validation: %1 valid
    // "
    //                      "IDs, %2 invalid IDs (maxId=%3, type=%4)")
    //                      .arg(validCount)
    //                      .arg(invalidCount)
    //                      .arg(maxId)
    //                      .arg(isCellSelectionOnPointCloud ? "cells->points"
    //                                                       : "points"));

    if (!hasValidIds) {
        CVLog::Error(
                QString("[cvSelectionPropertiesWidget] Selection IDs "
                        "(%1 items) are not valid for polyData (%2 elements)")
                        .arg(selectionIds.size())
                        .arg(maxId));
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("Selection IDs are not valid for the current data. "
                   "The selection may have been made on a different object."));
        return;
    }

    CVLog::Print(
            "[cvSelectionPropertiesWidget] Creating point cloud from "
            "selection...");

    // For cell selection on point cloud, convert to point selection
    // (cell IDs = point IDs for vertex cells)
    cvSelectionData exportSelection = m_selectionData;
    if (isCellSelectionOnPointCloud) {
        CVLog::Print(
                "[cvSelectionPropertiesWidget] Converting cell selection to "
                "point selection for point cloud export");
        exportSelection =
                cvSelectionData(selectionIds, cvSelectionData::POINTS);
    }

    cvSelectionExporter::ExportOptions options;
    ccPointCloud* cloud = nullptr;

    try {
        cloud = cvSelectionExporter::exportToPointCloud(
                polyData, exportSelection, options);
    } catch (const std::exception& e) {
        CVLog::Error(QString("[cvSelectionPropertiesWidget] Exception during "
                             "export: %1")
                             .arg(e.what()));
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("An error occurred during export: %1").arg(e.what()));
        return;
    } catch (...) {
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Unknown exception during "
                "export");
        QMessageBox::critical(this, tr("Export Failed"),
                              tr("An unknown error occurred during export."));
        return;
    }

    if (cloud && cloud->size() > 0) {
        // ParaView-style naming: ExtractSelection1, ExtractSelection2, ...
        m_extractCounter++;
        QString cloudName = QString("ExtractSelection%1").arg(m_extractCounter);
        cloud->setName(cloudName);
        // ParaView behavior: Extract objects are hidden by default
        // This prevents them from blocking selection of the original object
        cloud->setEnabled(false);
        cloud->setVisible(true);

        CVLog::Print(QString("[cvSelectionPropertiesWidget] Created point "
                             "cloud '%1' with %2 points")
                             .arg(cloudName)
                             .arg(cloud->size()));

        // Emit signal for main application to add to DB Tree
        // The main application should connect to this signal and call addToDB()
        emit extractedObjectReady(cloud);
        emit extractSelectionRequested();

        // ParaView behavior: Keep selection visible after extract
        // Selection is only cleared when user explicitly deletes it
        CVLog::Print(QString("[cvSelectionPropertiesWidget] Exported %1 points "
                             "as point cloud (selection preserved)")
                             .arg(cloud->size()));
    } else {
        // Clean up empty cloud if created
        if (cloud) {
            delete cloud;
            cloud = nullptr;
        }
        CVLog::Error(
                "[cvSelectionPropertiesWidget] Failed to export selection as "
                "point cloud - no points extracted");
        QMessageBox::critical(
                this, tr("Export Failed"),
                tr("Failed to export selection. No points could be extracted "
                   "from the selection."));
    }
}

void cvSelectionPropertiesWidget::onSelectionTableItemClicked(
        QTableWidgetItem* item) {
    if (!item || !m_selectionTableWidget) {
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
    bool isPointData =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);
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
    double savedColor[3] = {originalColor[0], originalColor[1],
                            originalColor[2]};

    // Set PRESELECTED mode to RED for emphasis (ParaView uses red for
    // interactive selection)
    m_highlighter->setHighlightColor(1.0, 0.0, 0.0,
                                     cvSelectionHighlighter::PRESELECTED);

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
        CVLog::PrintVerbose(
                QString("[cvSelectionPropertiesWidget] RED highlight: %1 "
                        "ID=%2 at (%3, %4, %5)")
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
            CVLog::PrintVerbose(
                    QString("[cvSelectionPropertiesWidget] RED highlight: "
                            "%1 ID=%2 "
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
            m_highlighter->setHighlightColor(
                    this->m_savedPreselectedColor[0],
                    this->m_savedPreselectedColor[1],
                    this->m_savedPreselectedColor[2],
                    cvSelectionHighlighter::PRESELECTED);

            if (!m_selectionData.isEmpty()) {
                // Restore original full selection highlight with SELECTED mode
                // (green)
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
    if (!m_selectionManager || m_selectionData.isEmpty() || !m_algebraOpCombo) {
        return;
    }

    cvSelectionAlgebra::Operation op =
            static_cast<cvSelectionAlgebra::Operation>(
                    m_algebraOpCombo->currentData().toInt());

    // For now, emit signal - actual implementation depends on having two
    // selections
    emit algebraOperationRequested(static_cast<int>(op));

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Algebra operation %1 "
                         "requested")
                         .arg(static_cast<int>(op)));
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

//-----------------------------------------------------------------------------
// onLoadBookmarkClicked removed - UI not implemented

//-----------------------------------------------------------------------------

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
    // ParaView reference: pqFindDataSelectionDisplayFrame::fillLabels
    if (!m_cellLabelsMenu) return;

    m_cellLabelsMenu->clear();

    // Add "None" option to disable labels
    QAction* noneAction = m_cellLabelsMenu->addAction(tr("None"));
    noneAction->setCheckable(true);
    noneAction->setChecked(m_currentCellLabelArray.isEmpty());
    connect(noneAction, &QAction::triggered, [this]() {
        m_currentCellLabelArray.clear();
        // Apply to highlighter
        if (m_highlighter) {
            m_highlighter->setCellLabelArray(QString(), false);
        }
        CVLog::Print("[cvSelectionPropertiesWidget] Cell labels disabled");
    });

    // Add "ID" option - ParaView uses vtkOriginalCellIds for this
    QAction* idAction = m_cellLabelsMenu->addAction(tr("ID"));
    idAction->setCheckable(true);
    idAction->setChecked(m_currentCellLabelArray == "ID");
    connect(idAction, &QAction::triggered, [this]() {
        m_currentCellLabelArray = "ID";
        // Apply to highlighter
        if (m_highlighter) {
            m_highlighter->setCellLabelArray("ID", true);
        }
        CVLog::Print("[cvSelectionPropertiesWidget] Cell labels set to ID");
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
                // Skip VTK internal arrays (ParaView does this)
                if (name.startsWith("vtk", Qt::CaseInsensitive)) {
                    continue;
                }
                QAction* action = m_cellLabelsMenu->addAction(name);
                action->setCheckable(true);
                action->setChecked(m_currentCellLabelArray == name);
                connect(action, &QAction::triggered, [this, name]() {
                    m_currentCellLabelArray = name;
                    // Apply to highlighter
                    if (m_highlighter) {
                        m_highlighter->setCellLabelArray(name, true);
                    }
                    CVLog::Print(QString("[cvSelectionPropertiesWidget] "
                                         "Cell labels set to %1")
                                         .arg(name));
                });
            }
        }
    }

    // Menu will be shown automatically by Qt since this is connected to
    // aboutToShow
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onPointLabelsClicked() {
    // Dynamically populate the menu with available point data arrays
    // ParaView reference: pqFindDataSelectionDisplayFrame::fillLabels
    if (!m_pointLabelsMenu) return;

    m_pointLabelsMenu->clear();

    // Add "None" option to disable labels
    QAction* noneAction = m_pointLabelsMenu->addAction(tr("None"));
    noneAction->setCheckable(true);
    noneAction->setChecked(m_currentPointLabelArray.isEmpty());
    connect(noneAction, &QAction::triggered, [this]() {
        m_currentPointLabelArray.clear();
        // Apply to highlighter
        if (m_highlighter) {
            m_highlighter->setPointLabelArray(QString(), false);
        }
        CVLog::Print("[cvSelectionPropertiesWidget] Point labels disabled");
    });

    // Add "ID" option - ParaView uses vtkOriginalPointIds for this
    QAction* idAction = m_pointLabelsMenu->addAction(tr("ID"));
    idAction->setCheckable(true);
    idAction->setChecked(m_currentPointLabelArray == "ID");
    connect(idAction, &QAction::triggered, [this]() {
        m_currentPointLabelArray = "ID";
        // Apply to highlighter
        if (m_highlighter) {
            m_highlighter->setPointLabelArray("ID", true);
        }
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
                // Skip VTK internal arrays (ParaView does this)
                if (name.startsWith("vtk", Qt::CaseInsensitive)) {
                    continue;
                }
                // Allow Normals and other arrays
                QAction* action = m_pointLabelsMenu->addAction(name);
                action->setCheckable(true);
                action->setChecked(m_currentPointLabelArray == name);
                connect(action, &QAction::triggered, [this, name]() {
                    m_currentPointLabelArray = name;
                    // Apply to highlighter
                    if (m_highlighter) {
                        m_highlighter->setPointLabelArray(name, true);
                    }
                    CVLog::Print(QString("[cvSelectionPropertiesWidget] "
                                         "Point labels set to %1")
                                         .arg(name));
                });
            }
        }
    }

    // Menu will be shown automatically by Qt since this is connected to
    // aboutToShow
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onEditLabelPropertiesClicked() {
    cvSelectionLabelPropertiesDialog dialog(this, false);

    // Convert SelectionLabelProperties to LabelProperties for dialog
    cvSelectionLabelPropertiesDialog::LabelProperties dialogProps;
    if (m_highlighter) {
        const SelectionLabelProperties& hlProps =
                m_highlighter->getLabelProperties(false);
        dialogProps.opacity = hlProps.opacity;
        dialogProps.pointSize = hlProps.pointSize;
        dialogProps.lineWidth = hlProps.lineWidth;
        dialogProps.cellLabelFontFamily = hlProps.cellLabelFontFamily;
        dialogProps.cellLabelFontSize = hlProps.cellLabelFontSize;
        dialogProps.cellLabelColor = hlProps.cellLabelColor;
        dialogProps.cellLabelBold = hlProps.cellLabelBold;
        dialogProps.cellLabelItalic = hlProps.cellLabelItalic;
        dialogProps.cellLabelShadow = hlProps.cellLabelShadow;
        dialogProps.cellLabelOpacity = hlProps.cellLabelOpacity;
        dialogProps.cellLabelFormat = hlProps.cellLabelFormat;
        dialogProps.pointLabelFontFamily = hlProps.pointLabelFontFamily;
        dialogProps.pointLabelFontSize = hlProps.pointLabelFontSize;
        dialogProps.pointLabelColor = hlProps.pointLabelColor;
        dialogProps.pointLabelBold = hlProps.pointLabelBold;
        dialogProps.pointLabelItalic = hlProps.pointLabelItalic;
        dialogProps.pointLabelShadow = hlProps.pointLabelShadow;
        dialogProps.pointLabelOpacity = hlProps.pointLabelOpacity;
        dialogProps.pointLabelFormat = hlProps.pointLabelFormat;
        dialogProps.showTooltips = hlProps.showTooltips;
        dialogProps.maxTooltipAttributes = hlProps.maxTooltipAttributes;
    }

    dialog.setProperties(dialogProps);
    connect(&dialog, &cvSelectionLabelPropertiesDialog::propertiesApplied, this,
            &cvSelectionPropertiesWidget::onLabelPropertiesApplied);
    dialog.exec();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionColorClicked() {
    QColor currentColor = getSelectionColor();
    QColor color = QColorDialog::getColor(currentColor, this, tr("Set Color"));
    if (color.isValid() && m_highlighter) {
        // Set color directly on highlighter (single source of truth)
        // This will trigger colorChanged signal which updates UI
        m_highlighter->setHighlightQColor(color,
                                          cvSelectionHighlighter::SELECTED);

        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }

        CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] Selection "
                                    "color changed to %1")
                                    .arg(color.name()));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onInteractiveSelectionColorClicked() {
    QColor currentColor = getInteractiveSelectionColor();
    QColor color = QColorDialog::getColor(currentColor, this, tr("Set Color"));
    if (color.isValid() && m_highlighter) {
        // Set color directly on highlighter (single source of truth)
        // This will trigger colorChanged signal which updates UI
        m_highlighter->setHighlightQColor(color, cvSelectionHighlighter::HOVER);

        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }

        CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] Interactive "
                                    "selection color changed to %1")
                                    .arg(color.name()));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onEditInteractiveLabelPropertiesClicked() {
    cvSelectionLabelPropertiesDialog dialog(this, true);

    // Convert SelectionLabelProperties to LabelProperties for dialog
    cvSelectionLabelPropertiesDialog::LabelProperties dialogProps;
    if (m_highlighter) {
        const SelectionLabelProperties& hlProps =
                m_highlighter->getLabelProperties(true);
        dialogProps.opacity = hlProps.opacity;
        dialogProps.pointSize = hlProps.pointSize;
        dialogProps.lineWidth = hlProps.lineWidth;
        dialogProps.cellLabelFontFamily = hlProps.cellLabelFontFamily;
        dialogProps.cellLabelFontSize = hlProps.cellLabelFontSize;
        dialogProps.cellLabelColor = hlProps.cellLabelColor;
        dialogProps.cellLabelBold = hlProps.cellLabelBold;
        dialogProps.cellLabelItalic = hlProps.cellLabelItalic;
        dialogProps.cellLabelShadow = hlProps.cellLabelShadow;
        dialogProps.cellLabelOpacity = hlProps.cellLabelOpacity;
        dialogProps.cellLabelFormat = hlProps.cellLabelFormat;
        dialogProps.pointLabelFontFamily = hlProps.pointLabelFontFamily;
        dialogProps.pointLabelFontSize = hlProps.pointLabelFontSize;
        dialogProps.pointLabelColor = hlProps.pointLabelColor;
        dialogProps.pointLabelBold = hlProps.pointLabelBold;
        dialogProps.pointLabelItalic = hlProps.pointLabelItalic;
        dialogProps.pointLabelShadow = hlProps.pointLabelShadow;
        dialogProps.pointLabelOpacity = hlProps.pointLabelOpacity;
        dialogProps.pointLabelFormat = hlProps.pointLabelFormat;
        dialogProps.showTooltips = hlProps.showTooltips;
        dialogProps.maxTooltipAttributes = hlProps.maxTooltipAttributes;
    }

    dialog.setProperties(dialogProps);
    connect(&dialog, &cvSelectionLabelPropertiesDialog::propertiesApplied, this,
            &cvSelectionPropertiesWidget::onInteractiveLabelPropertiesApplied);
    dialog.exec();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onLabelPropertiesApplied(
        const cvSelectionLabelPropertiesDialog::LabelProperties& props) {
    // Convert dialog properties to SelectionLabelProperties
    SelectionLabelProperties hlProps;
    hlProps.opacity = props.opacity;
    hlProps.pointSize = props.pointSize;
    hlProps.lineWidth = props.lineWidth;
    hlProps.cellLabelFontFamily = props.cellLabelFontFamily;
    hlProps.cellLabelFontSize = props.cellLabelFontSize;
    hlProps.cellLabelColor = props.cellLabelColor;
    hlProps.cellLabelBold = props.cellLabelBold;
    hlProps.cellLabelItalic = props.cellLabelItalic;
    hlProps.cellLabelShadow = props.cellLabelShadow;
    hlProps.cellLabelOpacity = props.cellLabelOpacity;
    hlProps.cellLabelFormat = props.cellLabelFormat;
    hlProps.pointLabelFontFamily = props.pointLabelFontFamily;
    hlProps.pointLabelFontSize = props.pointLabelFontSize;
    hlProps.pointLabelColor = props.pointLabelColor;
    hlProps.pointLabelBold = props.pointLabelBold;
    hlProps.pointLabelItalic = props.pointLabelItalic;
    hlProps.pointLabelShadow = props.pointLabelShadow;
    hlProps.pointLabelOpacity = props.pointLabelOpacity;
    hlProps.pointLabelFormat = props.pointLabelFormat;
    hlProps.showTooltips = props.showTooltips;
    hlProps.maxTooltipAttributes = props.maxTooltipAttributes;

    // Store to highlighter (single source of truth)
    if (m_highlighter) {
        m_highlighter->setLabelProperties(hlProps,
                                          false);  // false = SELECTED mode

        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }
    }

    // Apply font properties to annotations (cell labels)
    if (m_selectionManager) {
        cvSelectionAnnotationManager* annotations =
                m_selectionManager->getAnnotations();
        if (annotations) {
            // Set default properties for new annotations
            annotations->setDefaultLabelProperties(props,
                                                   true);  // true = cell labels

            // Apply to all existing annotations
            annotations->applyLabelProperties(props,
                                              true);  // true = cell labels
        }
    }

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Label properties applied: "
                    "opacity=%1, pointSize=%2, lineWidth=%3")
                    .arg(props.opacity)
                    .arg(props.pointSize)
                    .arg(props.lineWidth));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onInteractiveLabelPropertiesApplied(
        const cvSelectionLabelPropertiesDialog::LabelProperties& props) {
    // Convert dialog properties to SelectionLabelProperties
    SelectionLabelProperties hlProps;
    hlProps.opacity = props.opacity;
    hlProps.pointSize = props.pointSize;
    hlProps.lineWidth = props.lineWidth;
    hlProps.cellLabelFontFamily = props.cellLabelFontFamily;
    hlProps.cellLabelFontSize = props.cellLabelFontSize;
    hlProps.cellLabelColor = props.cellLabelColor;
    hlProps.cellLabelBold = props.cellLabelBold;
    hlProps.cellLabelItalic = props.cellLabelItalic;
    hlProps.cellLabelShadow = props.cellLabelShadow;
    hlProps.cellLabelOpacity = props.cellLabelOpacity;
    hlProps.cellLabelFormat = props.cellLabelFormat;
    hlProps.pointLabelFontFamily = props.pointLabelFontFamily;
    hlProps.pointLabelFontSize = props.pointLabelFontSize;
    hlProps.pointLabelColor = props.pointLabelColor;
    hlProps.pointLabelBold = props.pointLabelBold;
    hlProps.pointLabelItalic = props.pointLabelItalic;
    hlProps.pointLabelShadow = props.pointLabelShadow;
    hlProps.pointLabelOpacity = props.pointLabelOpacity;
    hlProps.pointLabelFormat = props.pointLabelFormat;
    hlProps.showTooltips = props.showTooltips;
    hlProps.maxTooltipAttributes = props.maxTooltipAttributes;

    // Apply default properties to annotation manager for point labels
    // (interactive)
    if (m_selectionManager) {
        cvSelectionAnnotationManager* annotations =
                m_selectionManager->getAnnotations();
        if (annotations) {
            // Set default properties for new annotations (point labels)
            annotations->setDefaultLabelProperties(
                    props, false);  // false = point labels

            // Apply to all existing annotations (point labels)
            annotations->applyLabelProperties(props,
                                              false);  // false = point labels
        }
    }

    // Store to highlighter (single source of truth)
    if (m_highlighter) {
        m_highlighter->setLabelProperties(
                hlProps, true);  // true = interactive/HOVER mode

        // Refresh display
        PclUtils::PCLVis* pclVis = getPCLVis();
        if (pclVis) {
            pclVis->UpdateScreen();
        }
    }

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Interactive label "
                         "properties applied: "
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

    // Update expression with new selection (ParaView-style)
    // ParaView wraps existing expression in parentheses when there are multiple
    // inputs Format: s0 -> s0|s1 -> (s0|s1)|s2 -> ((s0|s1)|s2)|s3 etc.
    QString expr = m_expressionEdit->text();
    if (!expr.isEmpty()) {
        // Wrap existing expression in parentheses if we have more than one
        // selection
        if (m_savedSelections.size() > 2) {
            expr = QString("(%1)").arg(expr);
        }
        expr += "|" + saved.name;
    } else {
        expr = saved.name;
    }
    m_expressionEdit->setText(expr);

    // Enable buttons
    m_removeAllSelectionsButton->setEnabled(true);
    m_activateCombinedSelectionsButton->setEnabled(
            !m_expressionEdit->text().isEmpty());

    // ParaView behavior: Disable the + button after adding a selection.
    // The button is re-enabled when a NEW selection is made.
    // This prevents adding the same selection multiple times.
    if (m_addSelectionButton) {
        m_addSelectionButton->setEnabled(false);
    }

    // Clear the current selection data to indicate it has been "consumed"
    // A new selection must be made to enable the + button again
    m_selectionData.clear();

    emit selectionAdded(saved.data);

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Added selection: %1 "
                    "(+ button disabled until new selection)")
                    .arg(saved.name));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onRemoveSelectedSelectionClicked() {
    QList<QTableWidgetItem*> selectedItems =
            m_selectionEditorTable->selectedItems();
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
            CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] Removed "
                                        "selection: %1")
                                        .arg(name));
        }
    }

    updateSelectionEditorTable();

    // Update button states
    m_removeAllSelectionsButton->setEnabled(!m_savedSelections.isEmpty());
    m_activateCombinedSelectionsButton->setEnabled(
            !m_expressionEdit->text().isEmpty() &&
            !m_savedSelections.isEmpty());
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
    QString expression = m_expressionEdit->text().trimmed();
    if (expression.isEmpty()) {
        QMessageBox::warning(this, tr("Activate Combined Selections"),
                             tr("Expression is empty. Please enter an "
                                "expression like: s0 & s1"));
        return;
    }

    if (m_savedSelections.isEmpty()) {
        QMessageBox::warning(
                this, tr("Activate Combined Selections"),
                tr("No saved selections. Please add selections first."));
        return;
    }

    CVLog::Print(
            QString("[cvSelectionPropertiesWidget] Evaluating expression: '%1'")
                    .arg(expression));

    // Evaluate the expression with error handling
    cvSelectionData result;
    try {
        result = evaluateExpression(expression);
    } catch (const std::exception& e) {
        CVLog::Error(
                QString("[cvSelectionPropertiesWidget] Expression evaluation "
                        "exception: %1")
                        .arg(e.what()));
        QMessageBox::warning(
                this, tr("Activate Combined Selections"),
                tr("Expression evaluation failed: %1").arg(e.what()));
        return;
    }

    if (result.isEmpty()) {
        QMessageBox::warning(
                this, tr("Activate Combined Selections"),
                tr("Expression evaluation resulted in empty selection.\n"
                   "Please check your expression syntax."));
        return;
    }

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Expression evaluated: "
                         "%1 elements")
                         .arg(result.count()));

    // Set the combined selection as the current selection
    if (m_selectionManager) {
        m_selectionManager->setCurrentSelection(result);

        // Update our own display - get polyData safely
        vtkPolyData* polyData = getPolyDataForSelection(&result);
        if (polyData) {
            updateSelection(result, polyData);

            CVLog::Print(
                    QString("[cvSelectionPropertiesWidget] Activated combined "
                            "selection: %1 elements from expression '%2'")
                            .arg(result.count())
                            .arg(expression));
        } else {
            CVLog::Warning(
                    "[cvSelectionPropertiesWidget] Could not get polyData "
                    "for combined selection");
            // Still update the selection data without polyData
            m_selectionData = result;
            m_selectionCount = result.count();
            syncUIWithHighlighter();
        }
    } else {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No selection manager available");
    }

    emit activateCombinedSelectionsRequested();
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::evaluateExpression(
        const QString& expression) {
    // ParaView expression syntax:
    // - Selection names: s0, s1, s2, ...
    // - NOT operator: ! (prefix)
    // - AND operator: &
    // - OR operator: |
    // - XOR operator: ^
    // - Parentheses: ( )
    // Example: "(s0 & s1) | !s2"

    QString expr = expression.simplified();
    if (expr.isEmpty()) {
        return cvSelectionData();
    }

    // Tokenize the expression
    QStringList tokens = tokenizeExpression(expr);
    if (tokens.isEmpty()) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Failed to tokenize expression");
        return cvSelectionData();
    }

    // Parse and evaluate
    int pos = 0;
    cvSelectionData result = parseOrExpression(tokens, pos);

    if (pos < tokens.size()) {
        CVLog::Warning(QString("[cvSelectionPropertiesWidget] Unexpected token "
                               "at position %1: %2")
                               .arg(pos)
                               .arg(tokens[pos]));
    }

    return result;
}

//-----------------------------------------------------------------------------
QStringList cvSelectionPropertiesWidget::tokenizeExpression(
        const QString& expression) {
    QStringList tokens;
    QString current;

    for (int i = 0; i < expression.length(); ++i) {
        QChar c = expression[i];

        if (c.isSpace()) {
            if (!current.isEmpty()) {
                tokens.append(current);
                current.clear();
            }
        } else if (c == '(' || c == ')' || c == '!' || c == '&' || c == '|' ||
                   c == '^') {
            if (!current.isEmpty()) {
                tokens.append(current);
                current.clear();
            }
            tokens.append(QString(c));
        } else {
            current.append(c);
        }
    }

    if (!current.isEmpty()) {
        tokens.append(current);
    }

    return tokens;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::parseOrExpression(
        const QStringList& tokens, int& pos) {
    // OR has lowest precedence: expr = xor_expr (| xor_expr)*
    cvSelectionData left = parseXorExpression(tokens, pos);

    while (pos < tokens.size() && tokens[pos] == "|") {
        pos++;  // consume '|'
        cvSelectionData right = parseXorExpression(tokens, pos);
        left = cvSelectionAlgebra::unionOf(left, right);
    }

    return left;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::parseXorExpression(
        const QStringList& tokens, int& pos) {
    // XOR: expr = and_expr (^ and_expr)*
    cvSelectionData left = parseAndExpression(tokens, pos);

    while (pos < tokens.size() && tokens[pos] == "^") {
        pos++;  // consume '^'
        cvSelectionData right = parseAndExpression(tokens, pos);
        left = cvSelectionAlgebra::symmetricDifferenceOf(left, right);
    }

    return left;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::parseAndExpression(
        const QStringList& tokens, int& pos) {
    // AND: expr = unary_expr (& unary_expr)*
    cvSelectionData left = parseUnaryExpression(tokens, pos);

    while (pos < tokens.size() && tokens[pos] == "&") {
        pos++;  // consume '&'
        cvSelectionData right = parseUnaryExpression(tokens, pos);
        left = cvSelectionAlgebra::intersectionOf(left, right);
    }

    return left;
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::parseUnaryExpression(
        const QStringList& tokens, int& pos) {
    // Unary: expr = !expr | primary
    if (pos < tokens.size() && tokens[pos] == "!") {
        pos++;  // consume '!'
        cvSelectionData operand = parseUnaryExpression(tokens, pos);

        // Complement needs polyData
        vtkPolyData* polyData = getPolyDataForSelection(&operand);
        if (polyData) {
            return cvSelectionAlgebra::complementOf(polyData, operand);
        } else {
            CVLog::Warning(
                    "[cvSelectionPropertiesWidget] Cannot compute complement: "
                    "no polyData");
            return cvSelectionData();
        }
    }

    return parsePrimaryExpression(tokens, pos);
}

//-----------------------------------------------------------------------------
cvSelectionData cvSelectionPropertiesWidget::parsePrimaryExpression(
        const QStringList& tokens, int& pos) {
    // Primary: ( expr ) | selection_name
    if (pos >= tokens.size()) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Unexpected end of expression");
        return cvSelectionData();
    }

    if (tokens[pos] == "(") {
        pos++;  // consume '('
        cvSelectionData result = parseOrExpression(tokens, pos);

        if (pos < tokens.size() && tokens[pos] == ")") {
            pos++;  // consume ')'
        } else {
            CVLog::Warning(
                    "[cvSelectionPropertiesWidget] Missing closing "
                    "parenthesis");
        }

        return result;
    }

    // Must be a selection name (e.g., "s0", "s1")
    QString name = tokens[pos];
    pos++;

    // Find the saved selection by name
    for (const SavedSelection& saved : m_savedSelections) {
        if (saved.name == name) {
            return saved.data;
        }
    }

    CVLog::Warning(
            QString("[cvSelectionPropertiesWidget] Unknown selection: %1")
                    .arg(name));
    return cvSelectionData();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionEditorTableSelectionChanged() {
    bool hasSelection = !m_selectionEditorTable->selectedItems().isEmpty();
    m_removeSelectionButton->setEnabled(hasSelection);

    // Highlight the corresponding selection when row is selected
    QList<QTableWidgetItem*> selectedItems =
            m_selectionEditorTable->selectedItems();
    if (!selectedItems.isEmpty()) {
        int row = selectedItems.first()->row();
        if (row >= 0 && row < m_savedSelections.size()) {
            const SavedSelection& sel = m_savedSelections[row];

            // Highlight this selection's data
            if (m_highlighter && !sel.data.isEmpty()) {
                m_highlighter->highlightSelection(
                        sel.data, cvSelectionHighlighter::PRESELECTED);

                // Update the viewer
                PclUtils::PCLVis* pclVis = getPCLVis();
                if (pclVis) {
                    pclVis->UpdateScreen();
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionEditorCellClicked(int row,
                                                               int column) {
    // When clicking on Color column, open color picker
    if (column == 2 && row >= 0 && row < m_savedSelections.size()) {
        // Get current color
        QColor currentColor = m_savedSelections[row].color;

        // Open color dialog
        QColor newColor = QColorDialog::getColor(
                currentColor, this,
                tr("Select Color for %1").arg(m_savedSelections[row].name));

        if (newColor.isValid() && newColor != currentColor) {
            // Update the saved selection color
            m_savedSelections[row].color = newColor;

            // Update the table display
            QTableWidgetItem* colorItem = m_selectionEditorTable->item(row, 2);
            if (colorItem) {
                colorItem->setText(newColor.name());
                colorItem->setBackground(newColor);
                colorItem->setForeground(
                        newColor.lightness() > 128 ? Qt::black : Qt::white);
            }

            CVLog::Print(QString("[cvSelectionPropertiesWidget] Changed color "
                                 "for %1 to %2")
                                 .arg(m_savedSelections[row].name)
                                 .arg(newColor.name()));
        }
    }

    // Highlight the selection for any cell click
    if (row >= 0 && row < m_savedSelections.size()) {
        const SavedSelection& sel = m_savedSelections[row];
        if (m_highlighter && !sel.data.isEmpty()) {
            m_highlighter->highlightSelection(
                    sel.data, cvSelectionHighlighter::PRESELECTED);

            PclUtils::PCLVis* pclVis = getPCLVis();
            if (pclVis) {
                pclVis->UpdateScreen();
            }
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSelectionEditorCellDoubleClicked(
        int row, int column) {
    // Double-click on Color column also opens color picker
    if (column == 2) {
        onSelectionEditorCellClicked(row, column);
    }
    // Double-click on Name column could allow editing (future feature)
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
    // ParaView-style invert selection implementation
    // Reference: ParaView/Qt/Components/pqFindDataCurrentSelectionFrame.cxx
    // line 377-397
    //
    // Key principle: NEVER modify m_originalSelectionIds during invert toggle
    // Only modify display (highlight + spreadsheet), not the underlying
    // selection state

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Invert selection: %1")
                    .arg(checked ? "ON" : "OFF"));

    // Store original selection on first invert (if not already stored)
    if (m_originalSelectionIds.isEmpty()) {
        m_originalSelectionIds = m_selectionData.ids();
    }

    if (m_originalSelectionIds.isEmpty()) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No original selection for "
                "inversion");
        return;
    }

    // Get polyData
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] No polyData for inversion");
        return;
    }

    // Determine which IDs to display
    QVector<qint64> displayIds;

    bool isCellSelection =
            (m_selectionData.fieldAssociation() == cvSelectionData::CELLS);
    vtkIdType totalCount = isCellSelection ? polyData->GetNumberOfCells()
                                           : polyData->GetNumberOfPoints();

    if (checked) {
        // Invert: show all IDs NOT in original selection
        QSet<qint64> originalIdSet = qSetFromVector(m_originalSelectionIds);

        displayIds.reserve(static_cast<int>(totalCount) -
                           m_originalSelectionIds.size());
        for (vtkIdType i = 0; i < totalCount; ++i) {
            if (!originalIdSet.contains(static_cast<qint64>(i))) {
                displayIds.append(static_cast<qint64>(i));
            }
        }
    } else {
        // Not inverted: show original selection
        displayIds = m_originalSelectionIds;
    }

    // Update 3D highlight
    if (m_highlighter) {
        vtkSmartPointer<vtkIdTypeArray> idArray =
                vtkSmartPointer<vtkIdTypeArray>::New();
        idArray->SetNumberOfTuples(displayIds.size());
        for (int i = 0; i < displayIds.size(); ++i) {
            idArray->SetValue(i, static_cast<vtkIdType>(displayIds[i]));
        }

        int fieldAssoc =
                (m_selectionData.fieldAssociation() == cvSelectionData::CELLS)
                        ? 0
                        : 1;
        m_highlighter->highlightSelection(polyData, idArray, fieldAssoc,
                                          cvSelectionHighlighter::SELECTED);

        // Update labels
        QString pointLabelArray = m_highlighter->getPointLabelArrayName();
        QString cellLabelArray = m_highlighter->getCellLabelArrayName();
        if (!pointLabelArray.isEmpty() &&
            m_highlighter->isPointLabelVisible()) {
            m_highlighter->setPointLabelArray(pointLabelArray, true);
        }
        if (!cellLabelArray.isEmpty() && m_highlighter->isCellLabelVisible()) {
            m_highlighter->setCellLabelArray(cellLabelArray, true);
        }
    }

    // Update spreadsheet display ONLY (not full updateSelection)
    // Create temporary display data without modifying m_selectionData
    cvSelectionData displaySelection(displayIds,
                                     m_selectionData.fieldAssociation());
    if (m_selectionData.hasActorInfo()) {
        displaySelection.setActorInfo(m_selectionData.primaryActor(),
                                      m_selectionData.primaryPolyData());
    }

    // Update UI with custom selection (without modifying m_selectionData)
    // This is much cleaner and safer than temporarily replacing m_selectionData
    updateSpreadsheetData(polyData, &displaySelection);
    updateStatistics(polyData, &displaySelection);

    emit invertSelectionRequested();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onFreezeClicked() {
    if (m_selectionData.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No selection to freeze");
        return;
    }

    // Freeze selection: Create a static copy that won't change with new
    // selections In ParaView, this converts the selection to an
    // "AppendSelection" filter For CloudViewer, we save current selection to
    // bookmarks with "Frozen_" prefix

    QString frozenName = QString("Frozen_%1")
                                 .arg(QDateTime::currentDateTime().toString(
                                         "yyyyMMdd_HHmmss"));

    // Bookmark functionality removed - UI not implemented
    CVLog::Print(
            QString("[cvSelectionPropertiesWidget] Selection frozen as: %1")
                    .arg(frozenName));

    QMessageBox::information(this, tr("Freeze Selection"),
                             tr("Selection frozen as: %1").arg(frozenName));

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
    // ParaView behavior: Export type depends on source data type, not selection
    // type
    // - Point clouds always export as point clouds (even with cell selection)
    // - Meshes export as meshes for cell selection, point clouds for point
    // selection

    bool isCells =
            (m_selectionData.fieldAssociation() == cvSelectionData::CELLS);
    bool isPoints =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);

    // Check source data type
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    bool isSourceMesh = false;
    if (polyData) {
        // A mesh has polygons (triangles/quads), a point cloud only has
        // vertices
        isSourceMesh = (polyData->GetNumberOfPolys() > 0);
        CVLog::PrintVerbose(
                QString("[cvSelectionPropertiesWidget::onExtractClicked] "
                        "Source: %1 points, %2 cells, %3 polys -> %4")
                        .arg(polyData->GetNumberOfPoints())
                        .arg(polyData->GetNumberOfCells())
                        .arg(polyData->GetNumberOfPolys())
                        .arg(isSourceMesh ? "mesh" : "point cloud"));
    }

    if (isCells && isSourceMesh) {
        // Cell selection on mesh -> export as mesh
        onExportToMeshClicked();
    } else {
        // Point selection OR cell selection on point cloud -> export as point
        // cloud For cell selection on point cloud, the cell IDs ARE the point
        // IDs (each vertex is a cell in VTK point cloud representation)
        onExportToPointCloudClicked();
    }

    emit extractSelectionRequested();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onPlotOverTimeClicked() {
    // Plot Over Time: Show histogram/distribution of selected data
    // Reference: ParaView's SelectionPlot functionality
    if (m_selectionData.isEmpty()) {
        QMessageBox::information(this, tr("Plot Over Time"),
                                 tr("No selection data to plot."));
        return;
    }

    // Get polyData for the selection
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        QMessageBox::warning(this, tr("Plot Over Time"),
                             tr("Cannot access selection data for plotting."));
        return;
    }

    // Emit signal for external handling
    emit plotOverTimeRequested();

    // Create plot dialog
    QDialog* plotDialog = new QDialog(this);
    plotDialog->setWindowTitle(tr("Selection Data Distribution"));
    plotDialog->setMinimumSize(700, 500);
    plotDialog->setAttribute(Qt::WA_DeleteOnClose);

    QVBoxLayout* dialogLayout = new QVBoxLayout(plotDialog);

    // Attribute selector
    QHBoxLayout* controlLayout = new QHBoxLayout();
    controlLayout->addWidget(new QLabel(tr("Attribute:")));

    QComboBox* attributeCombo = new QComboBox();
    bool isPointData =
            (m_selectionData.fieldAssociation() == cvSelectionData::POINTS);

    // Add "Coordinates" option for points
    if (isPointData) {
        attributeCombo->addItem(tr("X Coordinate"),
                                QVariant::fromValue(QString("__X__")));
        attributeCombo->addItem(tr("Y Coordinate"),
                                QVariant::fromValue(QString("__Y__")));
        attributeCombo->addItem(tr("Z Coordinate"),
                                QVariant::fromValue(QString("__Z__")));
    }

    // Add data arrays
    vtkDataSetAttributes* data = isPointData
                                         ? static_cast<vtkDataSetAttributes*>(
                                                   polyData->GetPointData())
                                         : static_cast<vtkDataSetAttributes*>(
                                                   polyData->GetCellData());

    if (data) {
        for (int i = 0; i < data->GetNumberOfArrays(); ++i) {
            vtkDataArray* arr = data->GetArray(i);
            if (arr && arr->GetName()) {
                attributeCombo->addItem(
                        QString::fromUtf8(arr->GetName()),
                        QVariant::fromValue(QString::fromUtf8(arr->GetName())));
            }
        }
    }

    controlLayout->addWidget(attributeCombo);
    controlLayout->addStretch();
    dialogLayout->addLayout(controlLayout);

    // Create QCustomPlot widget
    QCustomPlot* customPlot = new QCustomPlot(plotDialog);
    customPlot->setMinimumHeight(400);
    customPlot->xAxis->setLabel(tr("Index / Timestamp"));
    customPlot->yAxis->setLabel(tr("Attribute Value"));
    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom |
                                QCP::iSelectPlottables);
    customPlot->legend->setVisible(true);
    customPlot->legend->setFont(QFont("Helvetica", 9));
    dialogLayout->addWidget(customPlot);

    // Copy selection IDs for the lambda (avoid dangling pointer)
    QVector<qint64> selectionIds = m_selectionData.ids();

    // Helper to update plot
    auto updatePlot = [customPlot, polyData, selectionIds,
                       isPointData](const QString& attrName) {
        // Clear all existing plots before drawing new ones
        while (customPlot->plottableCount() > 0) {
            customPlot->removePlottable(0);
        }
        customPlot->clearGraphs();

        if (selectionIds.isEmpty()) {
            customPlot->replot();
            return;
        }

        const QVector<qint64>& ids = selectionIds;

        // Collect X (timestamp or ID) and Y (attribute value) data
        QVector<double> xData;  // timestamps or indices
        QVector<double> yData;  // attribute values
        xData.reserve(ids.size());
        yData.reserve(ids.size());

        // Try to find timestamp array in field data
        vtkDataSetAttributes* data =
                isPointData ? static_cast<vtkDataSetAttributes*>(
                                      polyData->GetPointData())
                            : static_cast<vtkDataSetAttributes*>(
                                      polyData->GetCellData());

        vtkDataArray* timestampArray = nullptr;
        if (data) {
            // Common timestamp field names
            const char* timestampNames[] = {"TimeValue", "Time",
                                            "timestamp", "time_value",
                                            "TimeStep",  nullptr};
            for (int i = 0; timestampNames[i] != nullptr; ++i) {
                timestampArray = data->GetArray(timestampNames[i]);
                if (timestampArray) break;
            }
        }

        // Collect Y values (attribute) and X values (timestamp or index)
        for (int idx = 0; idx < ids.size(); ++idx) {
            qint64 id = ids[idx];
            if (id < 0) continue;

            double yVal = 0.0;
            if (attrName == "__X__" || attrName == "__Y__" ||
                attrName == "__Z__") {
                if (id < polyData->GetNumberOfPoints()) {
                    double pt[3];
                    polyData->GetPoint(id, pt);
                    yVal = (attrName == "__X__")   ? pt[0]
                           : (attrName == "__Y__") ? pt[1]
                                                   : pt[2];
                }
            } else {
                if (data) {
                    vtkDataArray* arr =
                            data->GetArray(attrName.toUtf8().constData());
                    if (arr && id >= 0 && id < arr->GetNumberOfTuples()) {
                        yVal = arr->GetTuple1(id);
                    }
                }
            }

            // Get X value (timestamp if available, otherwise index)
            double xVal = idx;  // default to index
            if (timestampArray && id >= 0 &&
                id < timestampArray->GetNumberOfTuples()) {
                xVal = timestampArray->GetTuple1(id);
            }

            xData.append(xVal);
            yData.append(yVal);
        }

        if (yData.isEmpty()) return;

        // Create line graph
        QCPGraph* graph = customPlot->addGraph();
        graph->setData(xData, yData);
        graph->setPen(QPen(QColor(0, 100, 180), 2));
        graph->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5));
        graph->setName(attrName.startsWith("__") ? attrName.mid(2, 1) + " Coord"
                                                 : attrName);

        customPlot->rescaleAxes();
        customPlot->xAxis->setLabel(timestampArray ? tr("Timestamp")
                                                   : tr("Index"));
        customPlot->yAxis->setLabel(attrName.startsWith("__")
                                            ? attrName.mid(2, 1) + " Coordinate"
                                            : attrName);
        customPlot->replot();
    };

    // Connect attribute change
    connect(attributeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [updatePlot, attributeCombo]() {
                QString attrName = attributeCombo->currentData().toString();
                updatePlot(attrName);
            });

    // Initial plot
    if (attributeCombo->count() > 0) {
        QString attrName = attributeCombo->currentData().toString();
        updatePlot(attrName);
    }

    // Close button
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(buttonBox, &QDialogButtonBox::rejected, plotDialog,
            &QDialog::close);
    dialogLayout->addWidget(buttonBox);

    plotDialog->show();

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Plot Over Time: "
                         "Showing distribution for %1 %2")
                         .arg(m_selectionData.count())
                         .arg(m_selectionData.fieldTypeString()));
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
void cvSelectionPropertiesWidget::onToggleFieldDataClicked(bool checked) {
    // Toggle between showing Point/Cell data and Field data
    // ParaView: When enabled, shows vtkFieldData arrays instead of selection
    // IDs

    if (!m_spreadsheetTable) {
        return;
    }

    // Get current polyData
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        return;
    }

    m_spreadsheetTable->clear();
    m_spreadsheetTable->setRowCount(0);

    if (checked) {
        // Show field data arrays (global data, not per-point/cell)
        vtkFieldData* fieldData = polyData->GetFieldData();
        if (!fieldData || fieldData->GetNumberOfArrays() == 0) {
            // No field data available
            m_spreadsheetTable->setColumnCount(1);
            m_spreadsheetTable->setHorizontalHeaderLabels(QStringList()
                                                          << tr("Field Data"));
            m_spreadsheetTable->setRowCount(1);
            m_spreadsheetTable->setItem(
                    0, 0, new QTableWidgetItem(tr("No field data available")));
            return;
        }

        // Build column headers from field data array names
        QStringList headers;
        headers << tr("Index");
        for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* arr = fieldData->GetAbstractArray(i);
            if (arr && arr->GetName()) {
                headers << QString::fromStdString(arr->GetName());
            }
        }

        m_spreadsheetTable->setColumnCount(headers.size());
        m_spreadsheetTable->setHorizontalHeaderLabels(headers);

        // Find max number of tuples across all arrays
        vtkIdType maxTuples = 0;
        for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
            vtkAbstractArray* arr = fieldData->GetAbstractArray(i);
            if (arr) {
                maxTuples = std::max(maxTuples, arr->GetNumberOfTuples());
            }
        }

        // Limit rows
        int rowCount = std::min(1000, static_cast<int>(maxTuples));
        m_spreadsheetTable->setRowCount(rowCount);

        // Populate rows
        for (int row = 0; row < rowCount; ++row) {
            int col = 0;

            // Index column
            m_spreadsheetTable->setItem(
                    row, col++, new QTableWidgetItem(QString::number(row)));

            // Data columns
            for (int i = 0; i < fieldData->GetNumberOfArrays(); ++i) {
                vtkAbstractArray* arr = fieldData->GetAbstractArray(i);
                if (arr && arr->GetName()) {
                    QString valueStr;
                    if (row < arr->GetNumberOfTuples()) {
                        vtkDataArray* dataArr = vtkDataArray::SafeDownCast(arr);
                        if (dataArr) {
                            valueStr = QString::number(dataArr->GetTuple1(row),
                                                       'g', 6);
                        } else {
                            // String array or other type
                            vtkVariant v = arr->GetVariantValue(row);
                            valueStr = QString::fromStdString(v.ToString());
                        }
                    }
                    m_spreadsheetTable->setItem(row, col++,
                                                new QTableWidgetItem(valueStr));
                }
            }
        }
    } else {
        // Show normal selection data (Point/Cell data)
        updateSpreadsheetData(polyData);
    }
}

// ============================================================================
// Create Selection (Find Data) slots
// ============================================================================

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onDataProducerChanged(int index) {
    // ParaView behavior: When Data Producer is "None" (index 0), disable
    // the rest of Create Selection controls
    bool hasProducer = (index > 0);

    // Enable/disable all Create Selection controls except Data Producer combo
    if (m_elementTypeCombo) m_elementTypeCombo->setEnabled(hasProducer);
    if (m_attributeCombo) m_attributeCombo->setEnabled(hasProducer);
    if (m_operatorCombo) m_operatorCombo->setEnabled(hasProducer);
    if (m_valueEdit) m_valueEdit->setEnabled(hasProducer);

    // Update available attributes based on selected data producer
    updateAttributeCombo();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onElementTypeChanged(int index) {
    Q_UNUSED(index);
    // Update available attributes based on element type (Point/Cell)
    updateAttributeCombo();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onFindDataClicked() {
    // Execute the query with support for multiple query rows
    if (m_queryRows.isEmpty()) {
        CVLog::Warning("[cvSelectionPropertiesWidget] No query rows available");
        return;
    }

    QString dataProducer = m_dataProducerCombo
                                   ? m_dataProducerCombo->currentText()
                                   : QString();
    QString elementType = m_elementTypeCombo ? m_elementTypeCombo->currentText()
                                             : tr("Point");
    bool isCell = (elementType == tr("Cell"));

    // Special operators that don't need a value
    QStringList noValueOps = {tr("is min"), tr("is max"), tr("is <= mean"),
                              tr("is >= mean")};

    // Validate all query rows and collect conditions
    QVector<QPair<QString, QString>>
            queries;  // attribute, operator, value stored as pair
    QStringList queryDescriptions;

    for (int i = 0; i < m_queryRows.size(); ++i) {
        const QueryRow& row = m_queryRows[i];
        QString attribute = row.attributeCombo->currentText();
        QString op = row.operatorCombo->currentText();
        QString value = row.valueEdit->text();

        if (attribute.isEmpty()) {
            QMessageBox::warning(
                    this, tr("Find Data"),
                    tr("Please select an attribute in query row %1.")
                            .arg(i + 1));
            return;
        }

        if (value.isEmpty() && !noValueOps.contains(op)) {
            QMessageBox::warning(
                    this, tr("Find Data"),
                    tr("Please enter a value in query row %1.").arg(i + 1));
            return;
        }

        queryDescriptions.append(
                QString("%1 %2 %3").arg(attribute).arg(op).arg(value));
    }

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Find Data with %1 "
                         "condition(s): %2 (Element: %3)")
                         .arg(m_queryRows.size())
                         .arg(queryDescriptions.join(" AND "))
                         .arg(elementType));

    // For backward compatibility, emit signal for the first condition
    if (!m_queryRows.isEmpty()) {
        const QueryRow& firstRow = m_queryRows[0];
        emit findDataRequested(dataProducer, elementType,
                               firstRow.attributeCombo->currentText(),
                               firstRow.operatorCombo->currentText(),
                               firstRow.valueEdit->text());
    }

    // Perform the query with all conditions (combined with AND logic)
    // Start with first query
    if (!m_queryRows.isEmpty()) {
        const QueryRow& firstRow = m_queryRows[0];
        performFindData(firstRow.attributeCombo->currentText(),
                        firstRow.operatorCombo->currentText(),
                        firstRow.valueEdit->text(), isCell);

        // For additional rows, we would need to perform intersection with
        // current selection This requires more complex logic to combine
        // multiple selection queries For now, we apply only the first condition
        // as a starting implementation
        // TODO: Implement AND logic for multiple query conditions
        if (m_queryRows.size() > 1) {
            CVLog::Warning(
                    QString("[cvSelectionPropertiesWidget] Multiple query "
                            "conditions detected (%1 rows). "
                            "Currently only the first condition is applied. "
                            "Full AND logic implementation is pending.")
                            .arg(m_queryRows.size()));
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onResetClicked() {
    // Reset all query rows to default state
    for (auto& row : m_queryRows) {
        if (row.attributeCombo && row.attributeCombo->count() > 0) {
            row.attributeCombo->setCurrentIndex(0);
        }
        if (row.operatorCombo) {
            row.operatorCombo->setCurrentIndex(0);
        }
        if (row.valueEdit) {
            row.valueEdit->clear();
        }
    }

    if (m_processIdSpinBox) {
        m_processIdSpinBox->setValue(-1);
    }

    // Remove additional query rows (keep only the first one)
    while (m_queriesLayout && m_queriesLayout->count() > 0) {
        QLayoutItem* item = m_queriesLayout->takeAt(0);
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }

    CVLog::Print("[cvSelectionPropertiesWidget] Query reset to default.");
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onClearClicked() {
    // Clear current selection
    clearSelection();

    // Also clear the query
    onResetClicked();

    CVLog::Print("[cvSelectionPropertiesWidget] Selection and query cleared.");
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::addQueryRow(int index,
                                              const QString& attribute,
                                              const QString& op,
                                              const QString& value) {
    if (index == -1) {
        index = m_queryRows.size();
    }

    QueryRow row;

    // Create container widget for this row
    row.container = new QWidget(m_createSelectionContainer);
    QHBoxLayout* rowLayout = new QHBoxLayout(row.container);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(3);

    // Attribute combo
    row.attributeCombo = new QComboBox(row.container);
    row.attributeCombo->setMinimumWidth(80);
    row.attributeCombo->setToolTip(tr("Select attribute to query"));
    rowLayout->addWidget(row.attributeCombo);

    // Operator combo
    row.operatorCombo = new QComboBox(row.container);
    row.operatorCombo->addItems({tr("is"), tr(">="), tr("<="), tr(">"), tr("<"),
                                 tr("!="), tr("is min"), tr("is max"),
                                 tr("is <= mean"), tr("is >= mean")});
    row.operatorCombo->setToolTip(tr("Select comparison operator"));
    if (!op.isEmpty()) {
        int opIndex = row.operatorCombo->findText(op);
        if (opIndex >= 0) {
            row.operatorCombo->setCurrentIndex(opIndex);
        }
    }
    rowLayout->addWidget(row.operatorCombo);

    // Value input
    row.valueEdit = new QLineEdit(row.container);
    row.valueEdit->setPlaceholderText(tr("value"));
    row.valueEdit->setToolTip(tr("Enter comparison value"));
    if (!value.isEmpty()) {
        row.valueEdit->setText(value);
    }
    rowLayout->addWidget(row.valueEdit, 1);  // stretch factor 1

    // Plus button (add row after this one)
    row.plusButton = new QPushButton(row.container);
    QIcon plusIcon(":/Resources/images/svg/pqPlus.svg");
    if (plusIcon.isNull()) {
        row.plusButton->setText("+");
    } else {
        row.plusButton->setIcon(plusIcon);
    }
    row.plusButton->setToolTip(tr("Add query condition"));
    row.plusButton->setMaximumWidth(32);
    rowLayout->addWidget(row.plusButton);

    // Minus button (remove this row)
    row.minusButton = new QPushButton(row.container);
    QIcon minusIcon(":/Resources/images/svg/pqMinus.svg");
    if (minusIcon.isNull()) {
        row.minusButton->setText("-");
    } else {
        row.minusButton->setIcon(minusIcon);
    }
    row.minusButton->setToolTip(tr("Remove query condition"));
    row.minusButton->setMaximumWidth(32);
    rowLayout->addWidget(row.minusButton);

    row.container->setLayout(rowLayout);

    // Insert into layout and list
    m_queriesLayout->insertWidget(index, row.container);
    m_queryRows.insert(index, row);

    // Connect signals
    connect(row.plusButton, &QPushButton::clicked, [this, row]() {
        int idx = m_queryRows.indexOf(row);
        if (idx >= 0) {
            addQueryRow(idx + 1);
        }
    });

    connect(row.minusButton, &QPushButton::clicked, [this, row]() {
        int idx = m_queryRows.indexOf(row);
        if (idx >= 0) {
            removeQueryRow(idx);
        }
    });

    // Update attribute combo if first row
    if (index == 0 && !attribute.isEmpty()) {
        int attrIndex = row.attributeCombo->findText(attribute);
        if (attrIndex >= 0) {
            row.attributeCombo->setCurrentIndex(attrIndex);
        }
    }

    // Update button states (disable minus if only one row)
    updateQueryRowButtons();

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Added query row at index %1")
                    .arg(index));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::removeQueryRow(int index) {
    if (index < 0 || index >= m_queryRows.size()) {
        return;
    }

    // Can't remove if only one row
    if (m_queryRows.size() <= 1) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] Cannot remove the last query "
                "row");
        return;
    }

    QueryRow row = m_queryRows.takeAt(index);

    // Remove from layout and delete widgets
    m_queriesLayout->removeWidget(row.container);
    delete row.container;  // This will delete all child widgets

    // Update legacy pointers if we removed the first row
    if (index == 0 && !m_queryRows.isEmpty()) {
        m_attributeCombo = m_queryRows[0].attributeCombo;
        m_operatorCombo = m_queryRows[0].operatorCombo;
        m_valueEdit = m_queryRows[0].valueEdit;
    }

    // Update button states
    updateQueryRowButtons();

    CVLog::PrintVerbose(
            QString("[cvSelectionPropertiesWidget] Removed query row "
                    "at index %1")
                    .arg(index));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateQueryRowButtons() {
    bool canRemove = (m_queryRows.size() > 1);
    for (auto& row : m_queryRows) {
        row.minusButton->setEnabled(canRemove);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateDataProducerCombo() {
    if (!m_dataProducerCombo) {
        CVLog::Warning(
                "[cvSelectionPropertiesWidget] m_dataProducerCombo is null!");
        return;
    }

    m_dataProducerCombo->clear();

    // Add "(none)" option as default like ParaView
    m_dataProducerCombo->addItem(tr("(none)"));

    // Get list of available data sources from all renderers
    // Use "DatasetName" field data like PCLVis stores entity names
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        vtkRendererCollection* renderers = pclVis->getRendererCollection();
        if (renderers) {
            QSet<QString> addedNames;

            // Iterate through ALL renderers to get all data producers
            renderers->InitTraversal();
            vtkRenderer* renderer;
            while ((renderer = renderers->GetNextItem()) != nullptr) {
                vtkActorCollection* actors = renderer->GetActors();
                if (!actors) continue;

                actors->InitTraversal();
                vtkActor* actor;
                while ((actor = actors->GetNextActor()) != nullptr) {
                    // Include visible OR pickable actors (more inclusive)
                    if (!actor->GetVisibility()) {
                        continue;
                    }

                    QString name;

                    // Get the entity name from "DatasetName" field data
                    // This is how PCLVis stores the ccHObject name
                    vtkPolyData* polyData = vtkPolyData::SafeDownCast(
                            actor->GetMapper() ? actor->GetMapper()->GetInput()
                                               : nullptr);
                    if (polyData) {
                        vtkFieldData* fieldData = polyData->GetFieldData();
                        if (fieldData) {
                            // Primary: check "DatasetName" (used by PCLVis)
                            vtkStringArray* datasetNameArray =
                                    vtkStringArray::SafeDownCast(
                                            fieldData->GetAbstractArray(
                                                    "DatasetName"));
                            if (datasetNameArray &&
                                datasetNameArray->GetNumberOfTuples() > 0) {
                                name = QString::fromStdString(
                                        datasetNameArray->GetValue(0));
                            }

                            // Fallback: check "Name" array
                            if (name.isEmpty()) {
                                vtkAbstractArray* nameArray =
                                        fieldData->GetAbstractArray("Name");
                                if (nameArray &&
                                    nameArray->GetNumberOfTuples() > 0) {
                                    vtkVariant v =
                                            nameArray->GetVariantValue(0);
                                    name = QString::fromStdString(v.ToString());
                                }
                            }
                        }
                    }

                    // Skip if no valid name found
                    if (name.isEmpty()) {
                        continue;
                    }

                    // Add only unique names
                    if (!addedNames.contains(name)) {
                        m_dataProducerCombo->addItem(name);
                        addedNames.insert(name);
                    }
                }
            }

            CVLog::PrintVerbose(
                    QString("[cvSelectionPropertiesWidget] Found %1 "
                            "data producers")
                            .arg(addedNames.size()));
        }
    }

    // Also update attribute combo with first data producer
    updateAttributeCombo();

    // IMPORTANT: Manually trigger the enable/disable logic for Create Selection
    // controls When items are added to combo, the currentIndex may still be 0
    // (None), but we need to re-evaluate the enable state based on the new
    // state
    onDataProducerChanged(m_dataProducerCombo->currentIndex());
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateAttributeCombo() {
    // Update all query row attribute combos
    if (m_queryRows.isEmpty()) {
        return;
    }

    // Store current selections to preserve them if possible
    QStringList currentAttributes;
    for (const auto& row : m_queryRows) {
        if (row.attributeCombo && row.attributeCombo->count() > 0) {
            currentAttributes.append(row.attributeCombo->currentText());
        } else {
            currentAttributes.append(QString());
        }
    }

    // Clear all attribute combos (but keep the legacy pointer updated)
    for (auto& row : m_queryRows) {
        if (row.attributeCombo) {
            row.attributeCombo->clear();
        }
    }

    // ParaView behavior: ID fields should be associated with the selected Data
    // Producer IMPORTANT: Only use polyData from the SELECTED Data Producer,
    // NOT from other objects
    vtkPolyData* polyData = nullptr;
    PclUtils::PCLVis* pclVis = getPCLVis();
    bool hasExplicitProducer =
            m_dataProducerCombo && m_dataProducerCombo->currentIndex() > 0;

    // First, try to get from selected Data Producer (if any)
    if (pclVis && hasExplicitProducer) {
        QString producerName = m_dataProducerCombo->currentText();
        vtkRenderer* renderer =
                pclVis->getRendererCollection()->GetFirstRenderer();
        if (renderer) {
            vtkActorCollection* actors = renderer->GetActors();
            actors->InitTraversal();
            vtkActor* actor;
            while ((actor = actors->GetNextActor()) != nullptr) {
                if (!actor->GetVisibility() || !actor->GetPickable()) {
                    continue;
                }
                vtkPolyData* actorPolyData = vtkPolyData::SafeDownCast(
                        actor->GetMapper() ? actor->GetMapper()->GetInput()
                                           : nullptr);
                if (actorPolyData) {
                    vtkFieldData* fieldData = actorPolyData->GetFieldData();
                    if (fieldData) {
                        vtkStringArray* nameArray =
                                vtkStringArray::SafeDownCast(
                                        fieldData->GetAbstractArray(
                                                "DatasetName"));
                        if (nameArray && nameArray->GetNumberOfTuples() > 0) {
                            QString name = QString::fromStdString(
                                    nameArray->GetValue(0));
                            if (name == producerName) {
                                polyData = actorPolyData;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // If explicit producer selected but not found, don't fallback to other
        // objects
        if (!polyData) {
            CVLog::PrintVerbose(QString("[updateAttributeCombo] Data Producer "
                                        "'%1' not found")
                                        .arg(producerName));
            // Just add ID field, no other attributes
            m_attributeCombo->addItem(tr("ID"));
            return;
        }
    }

    // Fallback: Get from current selection ONLY if no explicit Data Producer
    // selected
    if (!polyData && !hasExplicitProducer) {
        polyData = getPolyDataForSelection(&m_selectionData);
    }

    if (!polyData) {
        // No polyData available - just show ID field
        m_attributeCombo->addItem(tr("ID"));
        return;
    }

    bool isCell = m_elementTypeCombo && m_elementTypeCombo->currentIndex() == 1;
    vtkDataSetAttributes* attrData =
            isCell ? static_cast<vtkDataSetAttributes*>(polyData->GetCellData())
                   : static_cast<vtkDataSetAttributes*>(
                             polyData->GetPointData());

    // Add ID first (like ParaView - just "ID", not "PointID")
    m_attributeCombo->addItem(tr("ID"));

    // Add "Points" array (coordinates) - ParaView style with Magnitude, X, Y, Z
    // This represents the 3D position of each point/cell center
    if (polyData && polyData->GetPoints()) {
        m_attributeCombo->addItem(tr("Points (Magnitude)"));
        m_attributeCombo->addItem(tr("Points (X)"));
        m_attributeCombo->addItem(tr("Points (Y)"));
        m_attributeCombo->addItem(tr("Points (Z)"));
    }

    // Helper lambda to add multi-component arrays in ParaView format
    // For color arrays (RGB/RGBA), we add the name without "(magnitude)"
    // For vector arrays (Normals, etc.), we add "(Magnitude)" then components
    auto addArrayToCombo = [this](vtkDataArray* array, const QString& name,
                                  const char* compNames[], int numCompNames,
                                  bool isColor) {
        if (!array) return;
        int numComponents = array->GetNumberOfComponents();
        if (numComponents == 1) {
            m_attributeCombo->addItem(name);
        } else if (numComponents > 1) {
            if (isColor) {
                // For color arrays, add the name first (not magnitude)
                // ParaView shows "Colors" or "RGB", not "Colors (magnitude)"
                m_attributeCombo->addItem(name);
            } else {
                // For vector arrays, add magnitude first
                m_attributeCombo->addItem(QString("%1 (Magnitude)").arg(name));
            }
            // Add individual components
            for (int c = 0; c < numComponents; ++c) {
                QString compName;
                const char* vtkCompName = array->GetComponentName(c);
                if (vtkCompName && strlen(vtkCompName) > 0) {
                    compName = QString::fromUtf8(vtkCompName);
                } else if (c < numCompNames && compNames) {
                    compName = compNames[c];
                } else {
                    compName = QString::number(c);
                }
                m_attributeCombo->addItem(
                        QString("%1 (%2)").arg(name, compName));
            }
        }
    };

    // Track added array names to avoid duplicates
    QSet<QString> addedArrays;

    // Track color arrays to avoid duplicates (RGB and Colors are the same
    // thing) PCLVis.cpp uses "RGB" for point clouds and cc2sm.cpp uses "Colors"
    // for meshes
    QSet<QString> colorArrayVariants = {"RGB", "Colors", "rgba", "rgb", "RGBA"};

    // Add VTK "active" arrays first (Normals, TCoords, Scalars/Colors)
    // These are set in PCLVis draw functions via SetNormals(), SetTCoords(),
    // SetScalars() Field names in PCLVis.cpp:
    // - Normals: set via polydata->GetPointData()->SetNormals() (only when
    // showNorms=true)
    // - Colors: named "Colors", set via polydata->GetPointData()->SetScalars()
    // - TCoords: named "TCoords0", "TCoords1", etc., first set via SetTCoords()
    if (attrData) {
        // Add Normals (3 components: X, Y, Z)
        // Set by updateShadingMode() and addTextureMesh() in PCLVis.cpp
        // NOTE: Normals are only added to VTK when "show normals" is enabled!
        // Check both active normals AND PCL-style separate arrays
        vtkDataArray* normals = attrData->GetNormals();
        bool hasNormals = (normals != nullptr);

        // If no active normals, check for PCL-style separate normal arrays
        // PCL stores normals as normal_x, normal_y, normal_z in PCLCloud
        // These might be present as 3-component arrays or separate 1-component
        // arrays
        if (!hasNormals) {
            // Check for a 3-component array named "Normals" or similar
            for (int i = 0; i < attrData->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = attrData->GetArray(i);
                if (arr && arr->GetNumberOfComponents() == 3) {
                    const char* name = arr->GetName();
                    if (name) {
                        QString qname = QString::fromUtf8(name).toLower();
                        if (qname.contains("normal")) {
                            normals = arr;
                            hasNormals = true;
                            break;
                        }
                    }
                }
            }
        }

        // Also check for separate normal_x, normal_y, normal_z arrays
        // (PCL style when not combined)
        if (!hasNormals) {
            vtkDataArray* nx = attrData->GetArray("normal_x");
            vtkDataArray* ny = attrData->GetArray("normal_y");
            vtkDataArray* nz = attrData->GetArray("normal_z");
            if (nx && ny && nz) {
                hasNormals = true;
                // Mark these as processed
                addedArrays.insert("normal_x");
                addedArrays.insert("normal_y");
                addedArrays.insert("normal_z");
                CVLog::PrintVerbose(
                        "[updateAttributeCombo] Found PCL-style separate "
                        "normals");
            }
        }

        if (hasNormals) {
            QString normalsName =
                    (normals && normals->GetName() &&
                     strlen(normals->GetName()))
                            ? QString::fromUtf8(normals->GetName())
                            : tr("Normals");
            // Add Normals entry with components (Magnitude, X, Y, Z)
            m_attributeCombo->addItem(
                    QString("%1 (Magnitude)").arg(normalsName));
            m_attributeCombo->addItem(QString("%1 (X)").arg(normalsName));
            m_attributeCombo->addItem(QString("%1 (Y)").arg(normalsName));
            m_attributeCombo->addItem(QString("%1 (Z)").arg(normalsName));
            addedArrays.insert(normalsName);
            addedArrays.insert("Normals");
            if (normals && normals->GetName()) {
                addedArrays.insert(QString::fromUtf8(normals->GetName()));
            }
        }

        // Add TCoords/Texture Coordinates (2 or 3 components: U, V, [W])
        // Set by addTextureMesh() in PCLVis.cpp, named "TCoords0", "TCoords1"
        vtkDataArray* tcoords = attrData->GetTCoords();
        if (tcoords) {
            QString tcoordsName =
                    (tcoords->GetName() && strlen(tcoords->GetName()))
                            ? QString::fromUtf8(tcoords->GetName())
                            : tr("TCoords");
            static const char* tcoordsComps[] = {"U", "V", "W"};
            addArrayToCombo(tcoords, tcoordsName, tcoordsComps, 3, false);
            addedArrays.insert(tcoordsName);
        }

        // Add Scalars/Colors (RGB: 3 components, RGBA: 4 components)
        // Set by addTextureMesh() in PCLVis.cpp
        // ParaView convention: always use "RGB" for color arrays, not "Colors"
        vtkDataArray* scalars = attrData->GetScalars();
        if (scalars) {
            QString scalarsName =
                    (scalars->GetName() && strlen(scalars->GetName()))
                            ? QString::fromUtf8(scalars->GetName())
                            : tr("RGB");
            // // Normalize "Colors" to "RGB" for consistency
            // if (scalarsName.compare("Colors", Qt::CaseInsensitive) == 0) {
            //     scalarsName = tr("RGB");
            // }
            int numComp = scalars->GetNumberOfComponents();
            if (numComp == 3) {
                static const char* rgbComps[] = {"R", "G", "B"};
                addArrayToCombo(scalars, scalarsName, rgbComps, 3, true);
            } else if (numComp == 4) {
                static const char* rgbaComps[] = {"R", "G", "B", "A"};
                addArrayToCombo(scalars, scalarsName, rgbaComps, 4, true);
            } else {
                addArrayToCombo(scalars, scalarsName, nullptr, 0, false);
            }
            addedArrays.insert(scalarsName);

            // Mark all color array variants as added to avoid duplicates
            // RGB and Colors are the same thing - just different names used by
            // different code paths (PCLVis uses "RGB", cc2sm uses "Colors")
            for (const QString& variant : colorArrayVariants) {
                addedArrays.insert(variant);
            }
        }
    }

    // Add all other arrays with component handling - ParaView format:
    // - Single component: just array name
    // - Multi-component: "{name} (Magnitude)" then "{name} ({componentName})"
    if (attrData) {
        for (int i = 0; i < attrData->GetNumberOfArrays(); ++i) {
            vtkDataArray* array = attrData->GetArray(i);
            if (!array) continue;

            const char* arrayName = array->GetName();
            if (!arrayName || strlen(arrayName) == 0) continue;

            QString name = QString::fromUtf8(arrayName);

            // Skip VTK internal arrays - ParaView filters these out
            // Reference: vtkSMTooltipSelectionPipeline.cxx
            if (name.startsWith("vtk", Qt::CaseInsensitive) ||
                name == "vtkOriginalPointIds" || name == "vtkOriginalCellIds" ||
                name == "vtkCompositeIndex" || name == "vtkBlockColors" ||
                name == "vtkGhostType") {
                continue;
            }

            // Skip if already added as a special array
            if (addedArrays.contains(name)) continue;
            addedArrays.insert(name);

            int numComponents = array->GetNumberOfComponents();

            // Check if this looks like a color array (RGB/RGBA unsigned char)
            bool isColorArray = false;
            QString lowerName = name.toLower();
            if ((numComponents == 3 || numComponents == 4) &&
                (lowerName.contains("color") || lowerName.contains("rgb") ||
                 lowerName.contains("rgba"))) {
                isColorArray = true;

                // If this is a color array and we've already added any color
                // array, skip it to avoid duplicates (RGB and Colors are the
                // same thing)
                bool alreadyHasColors = false;
                for (const QString& variant : colorArrayVariants) {
                    if (addedArrays.contains(variant)) {
                        alreadyHasColors = true;
                        break;
                    }
                }
                if (alreadyHasColors) {
                    // Mark this as added to avoid future duplicates
                    addedArrays.insert(name);
                    continue;  // Skip this array
                }

                // Mark all color variants as added
                for (const QString& variant : colorArrayVariants) {
                    addedArrays.insert(variant);
                }
            }

            if (numComponents == 1) {
                // Single component - add as is (like ParaView)
                m_attributeCombo->addItem(name);
            } else if (numComponents > 1) {
                if (isColorArray) {
                    // Color array - add name without magnitude
                    m_attributeCombo->addItem(name);
                } else {
                    // Multi-component - ParaView format: magnitude first, then
                    // individual components
                    m_attributeCombo->addItem(
                            QString("%1 (Magnitude)").arg(name));
                }

                // Add individual components with their names
                for (int c = 0; c < numComponents; ++c) {
                    QString compName;
                    const char* vtkCompName = array->GetComponentName(c);
                    if (vtkCompName && strlen(vtkCompName) > 0) {
                        compName = QString::fromUtf8(vtkCompName);
                    } else {
                        // Default component names based on component count
                        if (isColorArray && numComponents >= 3) {
                            static const char* rgbaComps[] = {"R", "G", "B",
                                                              "A"};
                            compName = rgbaComps[c];
                        } else if (numComponents == 3) {
                            static const char* xyz[] = {"X", "Y", "Z"};
                            compName = xyz[c];
                        } else if (numComponents == 4) {
                            static const char* xyzw[] = {"X", "Y", "Z", "W"};
                            compName = xyzw[c];
                        } else if (numComponents == 2) {
                            static const char* xy[] = {"X", "Y"};
                            compName = xy[c];
                        } else {
                            compName = QString::number(c);
                        }
                    }
                    m_attributeCombo->addItem(
                            QString("%1 (%2)").arg(name, compName));
                }
            }
        }
    }

    // Add Point/Cell at the end (like ParaView)
    if (!isCell && polyData->GetPoints()) {
        m_attributeCombo->addItem(tr("Point"));
    } else if (isCell) {
        m_attributeCombo->addItem(tr("Cell"));
    }

    // Update all query rows with the same attributes
    // First, collect all items from the first combo (which we just populated)
    QStringList attributes;
    if (!m_queryRows.isEmpty() && m_queryRows[0].attributeCombo) {
        QComboBox* firstCombo = m_queryRows[0].attributeCombo;
        for (int i = 0; i < firstCombo->count(); ++i) {
            attributes.append(firstCombo->itemText(i));
        }

        // Apply to all other combos
        for (int rowIdx = 1; rowIdx < m_queryRows.size(); ++rowIdx) {
            if (m_queryRows[rowIdx].attributeCombo) {
                QString current =
                        m_queryRows[rowIdx].attributeCombo->currentText();
                m_queryRows[rowIdx].attributeCombo->clear();
                m_queryRows[rowIdx].attributeCombo->addItems(attributes);

                // Try to restore previous selection
                if (!current.isEmpty()) {
                    int idx = m_queryRows[rowIdx].attributeCombo->findText(
                            current);
                    if (idx >= 0) {
                        m_queryRows[rowIdx].attributeCombo->setCurrentIndex(
                                idx);
                    }
                }
            }
        }
    }

    // Set first item as current for all combos if they're empty
    for (auto& row : m_queryRows) {
        if (row.attributeCombo && row.attributeCombo->count() > 0 &&
            row.attributeCombo->currentIndex() < 0) {
            row.attributeCombo->setCurrentIndex(0);
        }
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::performFindData(const QString& attribute,
                                                  const QString& op,
                                                  const QString& value,
                                                  bool isCell) {
    // Perform the query and create selection
    vtkPolyData* polyData = getPolyDataForSelection(&m_selectionData);
    if (!polyData) {
        QMessageBox::warning(this, tr("Find Data"),
                             tr("No data available to query."));
        return;
    }

    // Parse attribute name and component
    // Supports both formats:
    // - ParaView-style: "NormalsX", "NormalsY", "RGB (Magnitude)"
    // - Legacy: "Normals (X)", "RGB (R)"
    QString arrayName = attribute;
    int componentIndex = -1;  // -1 means use magnitude or scalar value
    bool isMagnitude = false;
    bool isIdQuery = (attribute == tr("ID"));
    // ParaView-style attribute format: "{arrayName} ({componentName})" or
    // "{arrayName} (magnitude)"
    bool isPointQuery = (attribute == tr("Point"));
    bool isCellQuery = (attribute == tr("Cell"));

    // Parse ParaView-style format: "{arrayName} ({componentName})" or
    // "{arrayName} (magnitude)"
    static QRegularExpression componentRegex(R"((.*)\s+\((.*)\)\s*$)");
    QRegularExpressionMatch match = componentRegex.match(attribute);

    if (match.hasMatch()) {
        arrayName = match.captured(1).trimmed();
        QString componentStr = match.captured(2).trimmed();

        // Check if it's magnitude (lowercase in ParaView)
        if (componentStr.toLower() == "magnitude") {
            isMagnitude = true;
        } else if (componentStr == "X" || componentStr == "R" ||
                   componentStr == "U" || componentStr == "0") {
            componentIndex = 0;
        } else if (componentStr == "Y" || componentStr == "G" ||
                   componentStr == "V" || componentStr == "1") {
            componentIndex = 1;
        } else if (componentStr == "Z" || componentStr == "B" ||
                   componentStr == "W" || componentStr == "2") {
            componentIndex = 2;
        } else if (componentStr == "A" || componentStr == "3") {
            componentIndex = 3;
        } else {
            // Try to parse as numeric component index
            bool ok;
            componentIndex = componentStr.toInt(&ok);
            if (!ok) componentIndex = 0;
        }
    }

    // Handle Point/Cell query (position-based queries)
    vtkPoints* points = polyData->GetPoints();

    // Get the data array
    vtkDataArray* dataArray = nullptr;

    if (!isIdQuery && !isPointQuery && !isCellQuery) {
        vtkDataSetAttributes* attrData =
                isCell ? static_cast<vtkDataSetAttributes*>(
                                 polyData->GetCellData())
                       : static_cast<vtkDataSetAttributes*>(
                                 polyData->GetPointData());

        // First try to get the array by name
        dataArray = attrData->GetArray(arrayName.toUtf8().constData());

        // If not found, check if it's one of the special "active" arrays
        // These arrays may be set via SetNormals(), SetTCoords(), SetScalars()
        // without being added to the named array list
        if (!dataArray) {
            // Check for Normals array
            vtkDataArray* normals = attrData->GetNormals();
            if (normals) {
                QString normalsName =
                        normals->GetName() && strlen(normals->GetName())
                                ? QString::fromUtf8(normals->GetName())
                                : tr("Normals");
                if (arrayName == normalsName || arrayName == tr("Normals")) {
                    dataArray = normals;
                }
            }

            // Check for TCoords array
            if (!dataArray) {
                vtkDataArray* tcoords = attrData->GetTCoords();
                if (tcoords) {
                    QString tcoordsName =
                            tcoords->GetName() && strlen(tcoords->GetName())
                                    ? QString::fromUtf8(tcoords->GetName())
                                    : tr("TCoords");
                    if (arrayName == tcoordsName ||
                        arrayName == tr("TCoords")) {
                        dataArray = tcoords;
                    }
                }
            }

            // Check for Scalars/RGB array
            if (!dataArray) {
                vtkDataArray* scalars = attrData->GetScalars();
                if (scalars) {
                    QString scalarsName =
                            scalars->GetName() && strlen(scalars->GetName())
                                    ? QString::fromUtf8(scalars->GetName())
                                    : tr("RGB");
                    // Normalize "Colors" to "RGB" for matching
                    // if (scalarsName.compare("Colors", Qt::CaseInsensitive) ==
                    // 0) {
                    //     scalarsName = tr("RGB");
                    // }
                    if (arrayName == scalarsName || arrayName == tr("RGB") ||
                        arrayName == tr("RGBA") || arrayName == tr("Colors")) {
                        dataArray = scalars;
                    }
                }
            }
        }

        if (!dataArray) {
            QMessageBox::warning(
                    this, tr("Find Data"),
                    tr("Attribute '%1' not found.").arg(arrayName));
            return;
        }
    }

    // Parse value
    double queryValue = 0.0;
    if (!value.isEmpty()) {
        bool ok;
        queryValue = value.toDouble(&ok);
        if (!ok) {
            QMessageBox::warning(this, tr("Find Data"),
                                 tr("Invalid numeric value: %1").arg(value));
            return;
        }
    }

    // Lambda to get value from element
    auto getValue = [&](vtkIdType i) -> double {
        if (isIdQuery) {
            return static_cast<double>(i);
        } else if (isPointQuery && points) {
            // "Point" query - return magnitude of position vector
            double pt[3];
            points->GetPoint(i, pt);
            return std::sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]);
        } else if (isCellQuery) {
            // "Cell" query - return cell ID (for containment checks, not yet
            // implemented)
            return static_cast<double>(i);
        } else if (dataArray) {
            int numComponents = dataArray->GetNumberOfComponents();
            if (isMagnitude && numComponents > 1) {
                double* tuple = dataArray->GetTuple(i);
                double sum = 0.0;
                for (int c = 0; c < numComponents; ++c) {
                    sum += tuple[c] * tuple[c];
                }
                return std::sqrt(sum);
            } else if (componentIndex >= 0 && componentIndex < numComponents) {
                return dataArray->GetComponent(i, componentIndex);
            } else {
                return dataArray->GetTuple1(i);
            }
        }
        return 0.0;
    };

    // Calculate statistics if needed
    double minVal = 0.0, maxVal = 0.0, meanVal = 0.0;
    vtkIdType numElements = isCell ? polyData->GetNumberOfCells()
                                   : polyData->GetNumberOfPoints();

    if ((dataArray || isPointQuery || isCellQuery) &&
        (op == tr("is min") || op == tr("is max") || op == tr("is <= mean") ||
         op == tr("is >= mean"))) {
        double sum = 0.0;
        minVal = std::numeric_limits<double>::max();
        maxVal = std::numeric_limits<double>::lowest();

        for (vtkIdType i = 0; i < numElements; ++i) {
            double val = getValue(i);
            sum += val;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
        meanVal = sum / numElements;
    }

    // Find matching elements
    QVector<qint64> matchingIds;

    for (vtkIdType i = 0; i < numElements; ++i) {
        double val = getValue(i);
        bool matchResult = false;

        if (op == tr("is") || op == tr("==")) {
            matchResult = (std::abs(val - queryValue) < 1e-9);
        } else if (op == tr(">=") || op == "is >=") {
            matchResult = (val >= queryValue);
        } else if (op == tr("<=") || op == "is <=") {
            matchResult = (val <= queryValue);
        } else if (op == tr(">")) {
            matchResult = (val > queryValue);
        } else if (op == tr("<")) {
            matchResult = (val < queryValue);
        } else if (op == tr("!=")) {
            matchResult = (std::abs(val - queryValue) >= 1e-9);
        } else if (op == tr("is min")) {
            matchResult = (std::abs(val - minVal) < 1e-9);
        } else if (op == tr("is max")) {
            matchResult = (std::abs(val - maxVal) < 1e-9);
        } else if (op == tr("is <= mean")) {
            matchResult = (val <= meanVal);
        } else if (op == tr("is >= mean")) {
            matchResult = (val >= meanVal);
        }

        if (matchResult) {
            matchingIds.append(static_cast<qint64>(i));
        }
    }

    CVLog::Print(QString("[cvSelectionPropertiesWidget] Find Data: Found %1 "
                         "matching elements")
                         .arg(matchingIds.size()));

    if (matchingIds.isEmpty()) {
        QMessageBox::information(this, tr("Find Data"),
                                 tr("No elements match the query criteria."));
        return;
    }

    // Create selection from matching IDs
    cvSelectionData::FieldAssociation assoc =
            isCell ? cvSelectionData::CELLS : cvSelectionData::POINTS;
    cvSelectionData newSelection(matchingIds, assoc);

    // Update the selection
    m_selectionData = newSelection;
    updateSelection(m_selectionData, polyData);

    // Highlight the selection
    if (m_highlighter) {
        m_highlighter->highlightSelection(m_selectionData,
                                          cvSelectionHighlighter::SELECTED);
    }

    // Update viewer
    PclUtils::PCLVis* pclVis = getPCLVis();
    if (pclVis) {
        pclVis->UpdateScreen();
    }

    // Update the Selected Data spreadsheet table with the query results
    updateSpreadsheetData(polyData);

    QMessageBox::information(this, tr("Find Data"),
                             tr("Selected %1 %2(s) matching '%3 %4 %5'")
                                     .arg(matchingIds.size())
                                     .arg(isCell ? tr("cell") : tr("point"))
                                     .arg(attribute)
                                     .arg(op)
                                     .arg(value.isEmpty() ? QString() : value));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::onSpreadsheetItemClicked(
        QTableWidgetItem* item) {
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
        colorItem->setForeground(sel.color.lightness() > 128 ? Qt::black
                                                             : Qt::white);
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
                QString("<b>Selected Data (%1)</b>")
                        .arg(name.isEmpty() ? tr("none") : name));
    }
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::refreshDataProducers() {
    updateDataProducerCombo();
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::updateSpreadsheetData(
        vtkPolyData* polyData, const cvSelectionData* customSelection) {
    if (!polyData || !m_spreadsheetTable) {
        return;
    }

    bool isPointData = m_attributeTypeCombo
                               ? (m_attributeTypeCombo->currentIndex() == 0)
                               : true;

    // Clear existing data
    m_spreadsheetTable->clear();
    m_spreadsheetTable->setRowCount(0);

    // Use custom selection if provided, otherwise use m_selectionData
    const cvSelectionData& selection =
            customSelection ? *customSelection : m_selectionData;

    // Get selection IDs
    if (selection.isEmpty()) {
        return;
    }

    const QVector<qint64>& ids = selection.ids();
    if (ids.isEmpty()) {
        return;
    }

    // Helper lambda to generate column headers for multi-component arrays
    // ParaView-style: Each component gets the SAME base name for header merging
    // cvMultiColumnHeaderView will merge adjacent columns with same name
    auto getArrayHeaders = [](vtkDataArray* arr) -> QStringList {
        QStringList headers;
        if (!arr || !arr->GetName()) return headers;

        QString baseName = QString::fromStdString(arr->GetName());
        int numComponents = arr->GetNumberOfComponents();

        // Skip VTK internal arrays
        if (baseName.startsWith("vtk", Qt::CaseInsensitive)) {
            return headers;
        }

        if (numComponents == 1) {
            // Single component: just the array name
            headers << baseName;
        } else {
            // Multi-component arrays: ALL components use the SAME base name
            // This allows cvMultiColumnHeaderView to merge them visually
            // ParaView approach: adjacent columns with same DisplayRole text
            // are merged
            for (int c = 0; c < numComponents; ++c) {
                headers << baseName;  // Same name for all components
            }
            // Add magnitude column with _Magnitude suffix (won't be merged)
            headers << QString("%1_Magnitude").arg(baseName);
        }
        return headers;
    };

    // Build column headers
    QStringList headers;
    headers << (isPointData ? tr("Point ID") : tr("Cell ID"));

    if (isPointData) {
        // Add point data arrays with ParaView-style multi-component handling
        vtkPointData* pointData = polyData->GetPointData();
        if (pointData) {
            for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = pointData->GetArray(i);
                headers << getArrayHeaders(arr);
            }
        }
        // Add Points columns at the end (ParaView-style: separate X, Y, Z
        // columns + magnitude) All three components use "Points" for header
        // merging
        headers << tr("Points") << tr("Points") << tr("Points")
                << tr("Points_Magnitude");
    } else {
        headers << tr("Type") << tr("Num Points");

        // Add cell data arrays with ParaView-style multi-component handling
        vtkCellData* cellData = polyData->GetCellData();
        if (cellData) {
            for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
                vtkDataArray* arr = cellData->GetArray(i);
                headers << getArrayHeaders(arr);
            }
        }
    }

    m_spreadsheetTable->setColumnCount(headers.size());
    m_spreadsheetTable->setHorizontalHeaderLabels(headers);

    // Resize columns to fit content
    m_spreadsheetTable->horizontalHeader()->setSectionResizeMode(
            QHeaderView::ResizeToContents);

    // Populate rows
    int rowCount =
            std::min(1000, static_cast<int>(ids.size()));  // Limit to 1000 rows
    m_spreadsheetTable->setRowCount(rowCount);

    for (int row = 0; row < rowCount; ++row) {
        qint64 id = ids[row];
        int col = 0;

        // ID column
        QTableWidgetItem* idItem = new QTableWidgetItem(QString::number(id));
        idItem->setData(Qt::UserRole, static_cast<qlonglong>(id));
        m_spreadsheetTable->setItem(row, col++, idItem);

        if (isPointData) {
            // Point data arrays (ParaView-style: separate columns for
            // each component)
            if (id >= 0 && id < polyData->GetNumberOfPoints()) {
                vtkPointData* pointData = polyData->GetPointData();
                if (pointData) {
                    for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
                        vtkDataArray* arr = pointData->GetArray(i);
                        if (arr && arr->GetName() &&
                            id < arr->GetNumberOfTuples()) {
                            int numComponents = arr->GetNumberOfComponents();
                            QString baseName =
                                    QString::fromStdString(arr->GetName());

                            // Skip VTK internal arrays
                            if (baseName.startsWith("vtk",
                                                    Qt::CaseInsensitive)) {
                                continue;
                            }

                            if (numComponents == 1) {
                                double value = arr->GetTuple1(id);
                                m_spreadsheetTable->setItem(
                                        row, col++,
                                        new QTableWidgetItem(QString::number(
                                                value, 'g', 6)));
                            } else {
                                // ParaView-style: All multi-component arrays -
                                // separate column for each component
                                double* tuple = arr->GetTuple(id);

                                // Add each component as a separate column
                                for (int c = 0; c < numComponents; ++c) {
                                    m_spreadsheetTable->setItem(
                                            row, col++,
                                            new QTableWidgetItem(
                                                    QString::number(tuple[c],
                                                                    'g', 6)));
                                }

                                // Add magnitude column
                                double magnitude = 0.0;
                                for (int c = 0; c < numComponents; ++c) {
                                    magnitude += tuple[c] * tuple[c];
                                }
                                magnitude = std::sqrt(magnitude);
                                m_spreadsheetTable->setItem(
                                        row, col++,
                                        new QTableWidgetItem(QString::number(
                                                magnitude, 'g', 6)));
                            }
                        } else if (arr && arr->GetName()) {
                            QString baseName =
                                    QString::fromStdString(arr->GetName());
                            // Skip VTK internal arrays
                            if (baseName.startsWith("vtk",
                                                    Qt::CaseInsensitive)) {
                                continue;
                            }
                            // Fill N/A for all columns of this array
                            QStringList arrayHeaders = getArrayHeaders(arr);
                            for (int c = 0; c < arrayHeaders.size(); ++c) {
                                m_spreadsheetTable->setItem(
                                        row, col++,
                                        new QTableWidgetItem(tr("N/A")));
                            }
                        }
                    }
                }

                // Points columns (ParaView-style: separate X, Y, Z columns +
                // magnitude)
                double pt[3];
                polyData->GetPoint(id, pt);
                // Separate column for each coordinate component
                m_spreadsheetTable->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(pt[0], 'g', 6)));
                m_spreadsheetTable->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(pt[1], 'g', 6)));
                m_spreadsheetTable->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(pt[2], 'g', 6)));
                // Points magnitude
                double mag = std::sqrt(pt[0] * pt[0] + pt[1] * pt[1] +
                                       pt[2] * pt[2]);
                m_spreadsheetTable->setItem(
                        row, col++,
                        new QTableWidgetItem(QString::number(mag, 'g', 6)));
            }
        } else {
            // Cell data
            if (id >= 0 && id < polyData->GetNumberOfCells()) {
                vtkCell* cell = polyData->GetCell(id);
                if (cell) {
                    // Type
                    QString typeName;
                    switch (cell->GetCellType()) {
                        case VTK_TRIANGLE:
                            typeName = tr("Triangle");
                            break;
                        case VTK_QUAD:
                            typeName = tr("Quad");
                            break;
                        case VTK_POLYGON:
                            typeName = tr("Polygon");
                            break;
                        case VTK_LINE:
                            typeName = tr("Line");
                            break;
                        case VTK_VERTEX:
                            typeName = tr("Vertex");
                            break;
                        default:
                            typeName = tr("Unknown");
                            break;
                    }
                    m_spreadsheetTable->setItem(row, col++,
                                                new QTableWidgetItem(typeName));

                    // Num Points
                    m_spreadsheetTable->setItem(
                            row, col++,
                            new QTableWidgetItem(QString::number(
                                    cell->GetNumberOfPoints())));
                }

                // Cell data arrays with multi-component handling
                vtkCellData* cellData = polyData->GetCellData();
                if (cellData) {
                    for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
                        vtkDataArray* arr = cellData->GetArray(i);
                        if (arr && arr->GetName() &&
                            id < arr->GetNumberOfTuples()) {
                            int numComponents = arr->GetNumberOfComponents();
                            if (numComponents == 1) {
                                double value = arr->GetTuple1(id);
                                m_spreadsheetTable->setItem(
                                        row, col++,
                                        new QTableWidgetItem(QString::number(
                                                value, 'g', 6)));
                            } else {
                                // For multi-component arrays, show as tuple
                                // format
                                QStringList values;
                                for (int c = 0; c < numComponents; ++c) {
                                    values << QString::number(
                                            arr->GetComponent(id, c), 'g', 6);
                                }
                                m_spreadsheetTable->setItem(
                                        row, col++,
                                        new QTableWidgetItem(
                                                QString("(%1)").arg(
                                                        values.join(", "))));
                            }
                        } else if (arr && arr->GetName()) {
                            // Single N/A column since we use tuple format
                            m_spreadsheetTable->setItem(
                                    row, col++,
                                    new QTableWidgetItem(tr("N/A")));
                        }
                    }
                }
            }
        }
    }

    m_spreadsheetTable->resizeColumnsToContents();

    CVLog::PrintVerbose(QString("[cvSelectionPropertiesWidget] Updated "
                                "spreadsheet with %1 rows")
                                .arg(rowCount));
}

//-----------------------------------------------------------------------------
void cvSelectionPropertiesWidget::setupCollapsibleGroupBox(
        QGroupBox* groupBox) {
    if (!groupBox) return;

    // Connect collapsible behavior - toggle content visibility when groupbox is
    // toggled
    connect(groupBox, &QGroupBox::toggled, [groupBox](bool checked) {
        QLayout* layout = groupBox->layout();
        if (!layout) return;

        for (int i = 0; i < layout->count(); ++i) {
            QLayoutItem* item = layout->itemAt(i);
            if (item->widget()) {
                item->widget()->setVisible(checked);
            }
            if (item->layout()) {
                for (int j = 0; j < item->layout()->count(); ++j) {
                    QLayoutItem* subItem = item->layout()->itemAt(j);
                    if (subItem->widget()) {
                        subItem->widget()->setVisible(checked);
                    }
                }
            }
        }
    });
}
