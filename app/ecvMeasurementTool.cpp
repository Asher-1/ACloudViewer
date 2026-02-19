// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMeasurementTool.h"

// LOCAL
#include "MainWindow.h"

// CV_DB_LIB
#include <widgets/ecvFontPropertyWidget.h>

// ECV_CORE_LIB
#include <ecvGenericMeasurementTools.h>
#include <ecvPointCloud.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvPolyline.h>

// CV_CORE_LIB
#include <CVLog.h>

// CV_PLUGIN_API
#include <ecvPickingHub.h>

// QT
#include <QApplication>
#include <QColorDialog>
#include <QMessageBox>
#include <QScreen>
#include <QScrollArea>
#include <QSizePolicy>
#include <QVBoxLayout>

#ifdef USE_PCL_BACKEND
#include <Tools/MeasurementTools/PclMeasurementTools.h>
#endif

ecvMeasurementTool::ecvMeasurementTool(QWidget* parent)
    : ccOverlayDialog(parent),
      Ui::MeasurementToolDlg(),
      m_tool(nullptr),
      m_updatingFromTool(false),
      m_pickingHub(nullptr),
      m_pickPointMode(0),
      m_currentColor(QColor(0, 255, 0))  // Default green color
{
    setupUi(this);

    // --- Wrap parametersLayout in a QScrollArea for DPI-adaptive sizing ---
    // Remove the empty parametersLayout created by setupUi from the main layout
    verticalLayout->removeItem(parametersLayout);

    // Create scroll area for tool parameter widgets
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    m_scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_scrollArea->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    // Small minimum so the scroll area doesn't force blank space for
    // lightweight tools (Contour). Heavy tools (Distance/Protractor) will
    // naturally expand via adjustSize().
    m_scrollArea->setMinimumHeight(50);

    // Create container widget inside scroll area
    QWidget* scrollContent = new QWidget();
    m_parametersLayout = new QVBoxLayout(scrollContent);
    m_parametersLayout->setSpacing(2);
    m_parametersLayout->setContentsMargins(0, 0, 0, 0);
    m_parametersLayout->setAlignment(Qt::AlignTop);
    m_scrollArea->setWidget(scrollContent);

    // Add scroll area to the main layout (replaces parametersLayout)
    verticalLayout->addWidget(m_scrollArea, /*stretch=*/1);

    // Compute DPI-adaptive max height for the WHOLE dialog (not just scroll
    // area). The scroll area expands to fill remaining space; capping the
    // dialog prevents it from covering the entire render window.
    QScreen* screen = this->screen();
    if (!screen) screen = QApplication::primaryScreen();
    if (screen) {
        int screenHeight = screen->availableGeometry().height();
        qreal dpr = screen->devicePixelRatio();
        // On HiDPI (macOS Retina), allow up to 55%; on standard, 65%
        double fraction = (dpr > 1.5) ? 0.55 : 0.65;
        int maxDialogH = static_cast<int>(screenHeight * fraction);
        this->setMaximumHeight(maxDialogH);
    }

    // Make header sections as compact as possible so the scroll area gets
    // more space
    if (fontGroupBox)
        fontGroupBox->setSizePolicy(QSizePolicy::Preferred,
                                    QSizePolicy::Maximum);

    // Get picking hub from MainWindow
    if (MainWindow::TheInstance()) {
        m_pickingHub = MainWindow::TheInstance()->pickingHub();
    }

    connect(resetButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::reset);
    connect(closeButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::closeDialog);
    connect(exportButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::exportMeasurement);

    // Instance management
    connect(instancesComboBox,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ecvMeasurementTool::onInstanceChanged);
    connect(addInstanceButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::addInstance);
    connect(removeInstanceButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::removeInstance);

    // Color selection
    connect(colorButton, &QPushButton::clicked, this,
            &ecvMeasurementTool::onColorButtonClicked);

    // Create and add font property widget
    // Color picker is now visible in the font widget (ParaView style)
    m_fontPropertyWidget = new ecvFontPropertyWidget(this);
    m_fontPropertyWidget->setColorPickerVisible(
            true);  // Show color picker in font widget
    fontLayout->addWidget(m_fontPropertyWidget);
    connect(m_fontPropertyWidget, &ecvFontPropertyWidget::fontPropertiesChanged,
            this, &ecvMeasurementTool::onFontPropertiesChanged);

    addOverridenShortcut(Qt::Key_Escape);
    connect(this, &ccOverlayDialog::shortcutTriggered, this, [this](int key) {
        if (key == Qt::Key_Escape) {
            closeDialog();
        }
    });

    // Initially disable instance management buttons
    removeInstanceButton->setEnabled(false);

    // Set initial color button appearance
    updateColorButtonAppearance();
}

ecvMeasurementTool::~ecvMeasurementTool() {
    // Remove picking listener
    if (m_pickingHub) {
        m_pickingHub->removeListener(this, false);
    }

    // Clean up all tool instances
    for (ecvGenericMeasurementTools* tool : m_toolInstances) {
        if (tool) {
            delete tool;
        }
    }
    m_toolInstances.clear();
}

ecvGenericMeasurementTools* ecvMeasurementTool::createMeasurementTool(
        ecvGenericMeasurementTools::MeasurementType type) {
#ifdef USE_PCL_BACKEND
    ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
    if (!viewer) {
        CVLog::Error("[ecvMeasurementTool] No visualizer available!");
        return nullptr;
    }

    ecvGenericMeasurementTools* tool = new PclMeasurementTools(viewer, type);
    if (tool) {
        if (m_entityContainer.getChildrenNumber() > 0) {
            tool->setInputData(m_entityContainer.getFirstChild());
        }
        tool->start();

        // CRITICAL: New tools always use their own default values
        // Different tool types are completely isolated - they don't inherit
        // from UI "Apply all" only affects instances of the SAME tool type If
        // "Apply all" is checked and there are existing instances, apply their
        // settings
        if (applyToAllCheckBox && applyToAllCheckBox->isChecked() &&
            !m_toolInstances.isEmpty() && m_toolInstances[0]) {
            // Apply settings from first existing instance of same type
            ecvGenericMeasurementTools* firstInstance = m_toolInstances[0];
            if (firstInstance->getMeasurementType() == type) {
#ifdef USE_PCL_BACKEND
                PclMeasurementTools* firstPclTool =
                        qobject_cast<PclMeasurementTools*>(firstInstance);
                if (firstPclTool) {
                    double r = 0.0, g = 1.0, b = 0.0;
                    if (firstPclTool->getColor(r, g, b)) {
                        tool->setColor(r, g, b);
                    }
                    tool->setFontFamily(firstPclTool->getFontFamily());
                    tool->setFontSize(firstPclTool->getFontSize());
                    double fontR = 1.0, fontG = 1.0, fontB = 1.0;
                    firstPclTool->getFontColor(fontR, fontG, fontB);
                    tool->setFontColor(fontR, fontG, fontB);
                    tool->setBold(firstPclTool->getFontBold());
                    tool->setItalic(firstPclTool->getFontItalic());
                    tool->setShadow(firstPclTool->getFontShadow());
                    tool->setFontOpacity(firstPclTool->getFontOpacity());
                    tool->setHorizontalJustification(
                            firstPclTool->getHorizontalJustification());
                    tool->setVerticalJustification(
                            firstPclTool->getVerticalJustification());
                }
#endif
            }
        }
        // Otherwise, tool uses its own default values (set in constructor)

        // Connect measurement changed signal
        connect(tool, &ecvGenericMeasurementTools::measurementChanged, this,
                &ecvMeasurementTool::updateMeasurementDisplay);
    }
    return tool;
#else
    return nullptr;
#endif
}

void ecvMeasurementTool::setMeasurementTool(ecvGenericMeasurementTools* tool) {
    if (!tool) return;

    // CRITICAL: Save current tool's UI state before switching
    // This ensures UI state is preserved when switching between tool types
    if (m_tool && m_tool->getMeasurementType() == m_measurementType) {
        // Save current tool's color and font properties to the tool itself
        // (they are already stored in the tool, but ensure they're up-to-date)
        // No need to save here as tools already store their own state
    }

    m_measurementType = tool->getMeasurementType();

    // CRITICAL: When switching tool types, clean up old tools of different
    // types This ensures each tool type has its own fresh instance and UI,
    // preventing UI interference between different tool types
    QList<ecvGenericMeasurementTools*> toolsToRemove;
    for (ecvGenericMeasurementTools* existingTool : m_toolInstances) {
        if (existingTool &&
            existingTool->getMeasurementType() != m_measurementType) {
            // Mark different type tools for removal
            toolsToRemove.append(existingTool);
        }
    }

    // Remove and delete old tools of different types
    for (ecvGenericMeasurementTools* toolToRemove : toolsToRemove) {
        // Disable shortcuts before deletion
        toolToRemove->disableShortcuts();

        // Remove widget from layout
        QWidget* widget = toolToRemove->getMeasurementWidget();
        if (widget) {
            m_parametersLayout->removeWidget(widget);
            widget->setVisible(false);
        }

        // Remove from list
        m_toolInstances.removeAll(toolToRemove);

        // Delete the tool (this will also delete its UI)
        toolToRemove->clear();
        delete toolToRemove;
    }

    // Add to instances list if not already present
    if (!m_toolInstances.contains(tool)) {
        m_toolInstances.append(tool);

        // Hide the widget initially - we'll show it when it becomes active
        tool->getMeasurementWidget()->setVisible(false);

        // Lock the tool initially - it will be unlocked when it becomes active
        tool->lockInteraction();

        // CRITICAL: New tools always use their own default values
        // Different tool types are completely isolated - they don't inherit
        // from UI "Apply all" only affects instances of the SAME tool type If
        // "Apply all" is checked and there are existing instances, apply their
        // settings
        if (applyToAllCheckBox && applyToAllCheckBox->isChecked() &&
            !m_toolInstances.isEmpty() && m_toolInstances[0]) {
            // Apply settings from first existing instance of same type
            ecvGenericMeasurementTools* firstInstance = m_toolInstances[0];
            if (firstInstance->getMeasurementType() ==
                tool->getMeasurementType()) {
#ifdef USE_PCL_BACKEND
                PclMeasurementTools* firstPclTool =
                        qobject_cast<PclMeasurementTools*>(firstInstance);
                if (firstPclTool) {
                    double r = 0.0, g = 1.0, b = 0.0;
                    if (firstPclTool->getColor(r, g, b)) {
                        tool->setColor(r, g, b);
                    }
                    tool->setFontFamily(firstPclTool->getFontFamily());
                    tool->setFontSize(firstPclTool->getFontSize());
                    double fontR = 1.0, fontG = 1.0, fontB = 1.0;
                    firstPclTool->getFontColor(fontR, fontG, fontB);
                    tool->setFontColor(fontR, fontG, fontB);
                    tool->setBold(firstPclTool->getFontBold());
                    tool->setItalic(firstPclTool->getFontItalic());
                    tool->setShadow(firstPclTool->getFontShadow());
                    tool->setFontOpacity(firstPclTool->getFontOpacity());
                    tool->setHorizontalJustification(
                            firstPclTool->getHorizontalJustification());
                    tool->setVerticalJustification(
                            firstPclTool->getVerticalJustification());
                }
#endif
            }
        }
        // Otherwise, tool uses its own default values (set in constructor)

        // Connect measurement changed signal
        connect(tool, &ecvGenericMeasurementTools::measurementChanged, this,
                &ecvMeasurementTool::updateMeasurementDisplay);

        // Connect point picking signals
        connect(tool, &ecvGenericMeasurementTools::pointPickingRequested, this,
                &ecvMeasurementTool::onPointPickingRequested);
        connect(tool, &ecvGenericMeasurementTools::pointPickingCancelled, this,
                &ecvMeasurementTool::onPointPickingCancelled);
    }

    // Set as current tool
    m_tool = tool;

    // Update instances combo box
    updateInstancesComboBox();

    // Switch to the new tool's UI
    switchToToolUI(tool);

    updateUIFromTool();
}

void ecvMeasurementTool::switchToToolUI(ecvGenericMeasurementTools* tool) {
    if (!tool) return;

    // Lock non-active tools and hide their widgets
    for (ecvGenericMeasurementTools* t : m_toolInstances) {
        if (t && t->getMeasurementWidget()) {
            QWidget* widget = t->getMeasurementWidget();
            // Remove from layout if present
            m_parametersLayout->removeWidget(widget);
            widget->setVisible(false);

            // Lock non-active tools (disable interaction, shortcuts, and UI
            // controls)
            if (t != tool) {
                t->lockInteraction();
            }
        }
    }

    // Show current tool's widget and unlock it
    QWidget* currentWidget = tool->getMeasurementWidget();
    if (currentWidget) {
        m_parametersLayout->addWidget(currentWidget);
        currentWidget->setVisible(true);
        currentWidget->setSizePolicy(QSizePolicy::Preferred,
                                     QSizePolicy::Preferred);

        // Unlock the active tool (enable interaction, shortcuts, and UI
        // controls)
        tool->unlockInteraction();
    }

    // CRITICAL: Invalidate layout chain and resize dialog to fit new content.
    // This prevents layout interference: the dialog adapts to the NEW tool's
    // content instead of retaining the OLD tool's size.
    if (m_scrollArea && m_scrollArea->widget()) {
        m_scrollArea->widget()->adjustSize();
    }
    if (layout()) {
        layout()->invalidate();
        layout()->activate();
    }
    adjustSize();
    updateGeometry();

    // After adjustSize, ensure the dialog has proper height for content
    // visibility Strategy: Fixed target heights based on actual tool type
    // (reliable & stable)
    if (layout()) layout()->activate();

    if (m_scrollArea && m_scrollArea->widget() && tool) {
        // Determine target height based on actual tool type
        int targetH;
        ecvGenericMeasurementTools::MeasurementType toolType =
                tool->getMeasurementType();

        switch (toolType) {
            case ecvGenericMeasurementTools::CONTOUR_WIDGET:
                // Contour: ultra-compact (minimal components)
                targetH = 240;
                break;

            case ecvGenericMeasurementTools::DISTANCE_WIDGET:
            case ecvGenericMeasurementTools::PROTRACTOR_WIDGET:
                // Distance/Protractor: spacious (many parameters + Tips)
                targetH = 400;
                break;

            default:
                // Fallback: moderate height
                targetH = 400;
                break;
        }

        // Apply maximum height constraint
        targetH = qMin(targetH, maximumHeight());

        // Only resize if significantly different (avoid jitter)
        if (qAbs(height() - targetH) > 30) {
            resize(width(), targetH);
        }
    }

    // Update UI controls (color button and font widget) from the current tool
    updateUIFromTool();
}

void ecvMeasurementTool::updateInstancesComboBox() {
    instancesComboBox->blockSignals(true);
    instancesComboBox->clear();

    for (int i = 0; i < m_toolInstances.size(); ++i) {
        QString typeName;
        switch (m_toolInstances[i]->getMeasurementType()) {
            case ecvGenericMeasurementTools::DISTANCE_WIDGET:
                typeName = "Ruler";
                break;
            case ecvGenericMeasurementTools::PROTRACTOR_WIDGET:
                typeName = "Protractor";
                break;
            case ecvGenericMeasurementTools::CONTOUR_WIDGET:
                typeName = "Contour";
                break;
        }
        QString instanceName = QString("%1 #%2").arg(typeName).arg(i + 1);
        instancesComboBox->addItem(instanceName);

        // Update tool's instance label for display in 3D view (e.g., " #1", "
        // #2") Important: When instances are added/removed, labels are
        // automatically updated
        QString label = QString(" #%1").arg(i + 1);
        m_toolInstances[i]->setInstanceLabel(label);
    }

    // Set current index
    int currentIndex = m_toolInstances.indexOf(m_tool);
    if (currentIndex >= 0) {
        instancesComboBox->setCurrentIndex(currentIndex);
    }

    instancesComboBox->blockSignals(false);
}

void ecvMeasurementTool::onInstanceChanged(int index) {
    if (index < 0 || index >= m_toolInstances.size()) return;

    ecvGenericMeasurementTools* newTool = m_toolInstances[index];
    if (newTool == m_tool) return;  // Already showing this tool

    m_tool = newTool;
    m_measurementType = m_tool->getMeasurementType();

    // Switch to the selected tool's UI
    switchToToolUI(m_tool);

    updateUIFromTool();
}

void ecvMeasurementTool::addInstance() {
    // Allow multiple instances for all measurement types including contour
    ecvGenericMeasurementTools* newTool =
            createMeasurementTool(m_measurementType);
    if (newTool) {
        // Set up the VTK widget reference for the new tool
        // This ensures new instances can create shortcuts when unlocked
        if (m_linkedWidget) {
            // Directly set the m_vtkWidget for the new tool
            // This will be used by unlockInteraction() to create shortcuts
            CVLog::PrintDebug(QString("[ecvMeasurementTool::addInstance] "
                                      "Setting m_vtkWidget=%1 for new tool")
                                      .arg((quintptr)m_linkedWidget, 0, 16));
            // We need to access the tool's internal member
            // Since we can't access it directly (it's in
            // cvGenericMeasurementTool), we'll call setupShortcuts which will
            // save the widget reference
            newTool->setupShortcuts(m_linkedWidget);
        } else {
            CVLog::Warning(
                    "[ecvMeasurementTool::addInstance] m_linkedWidget is null, "
                    "new tool may not have shortcuts");
        }

        setMeasurementTool(newTool);
        removeInstanceButton->setEnabled(true);
    }
}

void ecvMeasurementTool::removeInstance() {
    if (m_toolInstances.size() <= 1) {
        CVLog::Warning("Cannot remove the last instance");
        return;
    }

    int index = instancesComboBox->currentIndex();
    if (index < 0 || index >= m_toolInstances.size()) return;

    ecvGenericMeasurementTools* toolToRemove = m_toolInstances[index];

    // Remove widget from layout (it should already be removed by
    // switchToToolUI, but ensure it's clean)
    QWidget* widget = toolToRemove->getMeasurementWidget();
    if (widget) {
        m_parametersLayout->removeWidget(widget);
        widget->setVisible(false);
    }

    // Remove from list
    m_toolInstances.removeAt(index);

    // Delete tool (CRITICAL: disable shortcuts first to prevent crash)
    if (toolToRemove) {
        toolToRemove->disableShortcuts();  // Disable shortcuts before deletion
        toolToRemove->clear();
        delete toolToRemove;
    }

    // Set new current tool
    if (m_toolInstances.size() > 0) {
        int newIndex = (index < m_toolInstances.size())
                               ? index
                               : m_toolInstances.size() - 1;
        m_tool = m_toolInstances[newIndex];

        // Switch to the new tool's UI
        switchToToolUI(m_tool);

        updateInstancesComboBox();
        instancesComboBox->setCurrentIndex(newIndex);
        updateUIFromTool();
    } else {
        m_tool = nullptr;
        updateInstancesComboBox();
        removeInstanceButton->setEnabled(false);
    }
    ecvDisplayTools::UpdateScreen();
}

bool ecvMeasurementTool::addAssociatedEntity(ccHObject* entity) {
    if (!entity) {
        assert(false);
        return false;
    }

    // special case
    if (entity->isGroup()) {
        for (unsigned i = 0; i < entity->getChildrenNumber(); ++i) {
            if (!addAssociatedEntity(entity->getChild(i))) {
                return false;
            }
        }
        return true;
    }

    if (!m_entityContainer.addChild(entity, ccHObject::DP_NONE)) {
        CVLog::Error("An error occurred (see Console)");
        return false;
    }

    // force visibility
    entity->setVisible(true);
    entity->setEnabled(true);

    // Set input for all tool instances
    for (ecvGenericMeasurementTools* tool : m_toolInstances) {
        if (tool) {
            tool->setInputData(entity);
        }
    }

    return true;
}

unsigned ecvMeasurementTool::getNumberOfAssociatedEntity() const {
    return m_entityContainer.getChildrenNumber();
}

bool ecvMeasurementTool::linkWith(QWidget* win) {
    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    // Save the widget reference for later use (when creating new instances)
    m_linkedWidget = win;

    // Setup keyboard shortcuts bound to the VTK widget
    for (ecvGenericMeasurementTools* tool : m_toolInstances) {
        if (tool) {
            tool->setupShortcuts(win);
        }
    }

    return true;
}

bool ecvMeasurementTool::start() {
    assert(!m_processing);
    if (m_toolInstances.empty()) return false;

    // Ensure picking hub is available
    if (!m_pickingHub && MainWindow::TheInstance()) {
        m_pickingHub = MainWindow::TheInstance()->pickingHub();
    }

    if (!m_pickingHub) {
        CVLog::Warning("[ecvMeasurementTool] Picking hub not available!");
    }

    // Start all tool instances
    for (ecvGenericMeasurementTools* tool : m_toolInstances) {
        if (tool) {
            if (m_entityContainer.getChildrenNumber() > 0) {
                tool->setInputData(m_entityContainer.getFirstChild());
            }
            tool->start();  // This initializes VTK widgets and m_interactor

            // CRITICAL: After start() initializes m_interactor, manage
            // shortcuts:
            // - Lock non-active tools (disable shortcuts if they exist)
            // - Unlock the active tool (create and enable shortcuts)
            if (tool != m_tool) {
                tool->lockInteraction();
            } else {
                // Unlock the active tool to ensure shortcuts are created
                // (m_interactor is now available after start())
                tool->unlockInteraction();
            }
        }
    }

    return ccOverlayDialog::start();
}

void ecvMeasurementTool::stop(bool state) {
    // Remove picking listener
    if (m_pickingHub) {
        m_pickingHub->removeListener(this, false);
    }

    m_pickPointMode = 0;

    // Clean up all tool instances
    for (ecvGenericMeasurementTools* tool : m_toolInstances) {
        if (tool) {
            // Disable shortcuts before deleting the tool
            // This prevents keyboard shortcuts from triggering after the tool
            // is closed
            tool->disableShortcuts();

            m_parametersLayout->removeWidget(tool->getMeasurementWidget());
            tool->clear();
            delete tool;
        }
    }
    m_toolInstances.clear();
    m_tool = nullptr;

    releaseAssociatedEntities();
    ccOverlayDialog::stop(state);
}

void ecvMeasurementTool::reset() {
    if (m_tool) {
        m_tool->reset();
    }
}

void ecvMeasurementTool::closeDialog() { stop(true); }

void ecvMeasurementTool::updateMeasurementDisplay() {
    if (m_tool == sender()) {
        updateUIFromTool();
    }
}

void ecvMeasurementTool::updateUIFromTool() {
    // UI updates are now handled by the individual tool widgets
    // No need to update removed spinboxes

    // CRITICAL: Update shared UI controls (color button and font widget) from
    // current tool This prevents UI interference between different tool
    // instances Each tool instance should have its own independent settings
    // displayed in the UI
    if (!m_tool) return;

    // Prevent recursive updates
    if (m_updatingFromTool) return;
    m_updatingFromTool = true;

    // Update color button and font widget from current tool's properties
#ifdef USE_PCL_BACKEND
    PclMeasurementTools* pclTool = qobject_cast<PclMeasurementTools*>(m_tool);
    if (pclTool) {
        // CRITICAL: Always update UI from current tool's actual color
        // Different tool types are completely isolated - UI always reflects
        // current tool's state
        double r = 0.0, g = 1.0, b = 0.0;  // Default green
        if (pclTool->getColor(r, g, b)) {
            QColor toolColor = QColor::fromRgbF(r, g, b);
            // Always update button appearance to show current tool's color
            updateColorButtonAppearance(toolColor);
        }

        // Update font widget from current tool's font properties
        if (m_fontPropertyWidget) {
            // Block signals to prevent triggering applyFontToTools during
            // update
            m_fontPropertyWidget->blockSignals(true);

            // Get font properties from tool
            ecvFontPropertyWidget::FontProperties props;
            props.family = pclTool->getFontFamily();
            props.size = pclTool->getFontSize();
            double fontR = 1.0, fontG = 1.0, fontB = 1.0;
            pclTool->getFontColor(fontR, fontG, fontB);
            props.color = QColor::fromRgbF(fontR, fontG, fontB);
            props.bold = pclTool->getFontBold();
            props.italic = pclTool->getFontItalic();
            props.shadow = pclTool->getFontShadow();
            props.opacity = pclTool->getFontOpacity();
            props.horizontalJustification =
                    pclTool->getHorizontalJustification();
            props.verticalJustification = pclTool->getVerticalJustification();

            // Update font widget using setFontProperties (sets all properties
            // at once)
            m_fontPropertyWidget->setFontProperties(props);

            m_fontPropertyWidget->blockSignals(false);
        }
    }
#endif

    m_updatingFromTool = false;
}

void ecvMeasurementTool::toggleWidget(bool state) {
    if (m_tool) {
        QWidget* widget = m_tool->getMeasurementWidget();
        if (widget) {
            widget->setVisible(state);
        }
    }
}

void ecvMeasurementTool::exportMeasurement() {
    if (!m_tool) {
        CVLog::Warning("[ecvMeasurementTool] No current tool selected");
        return;
    }

    // Get output from current tool (selected contour instance)
    ccHObject* output = m_tool->getOutput();
    if (!output) {
        CVLog::Warning(
                "[ecvMeasurementTool] No measurement result to export from "
                "current tool");
        return;
    }

    if (MainWindow::TheInstance()) {
        output->setEnabled(true);

        // For contour, the name is already set in getOutput() with ID
        // For other tools, set name based on type and instance index
        if (m_measurementType != ecvGenericMeasurementTools::CONTOUR_WIDGET) {
            QString typeName;
            switch (m_measurementType) {
                case ecvGenericMeasurementTools::DISTANCE_WIDGET:
                    typeName = "Ruler";
                    break;
                case ecvGenericMeasurementTools::PROTRACTOR_WIDGET:
                    typeName = "Protractor";
                    break;
                default:
                    typeName = "Measurement";
                    break;
            }
            output->setName(QString("%1_%2").arg(typeName).arg(
                    m_toolInstances.indexOf(m_tool) + 1));
        }

        MainWindow::TheInstance()->addToDB(output);
        m_out_entities.push_back(output);

        CVLog::Print(QString("[ecvMeasurementTool] Exported %1")
                             .arg(output->getName()));
    } else {
        CVLog::Error("[ecvMeasurementTool] MainWindow instance not available");
        delete output;
    }
}

void ecvMeasurementTool::releaseAssociatedEntities() {
    m_entityContainer.removeAllChildren();
    m_out_entities.clear();
}

void ecvMeasurementTool::onItemPicked(const PickedItem& pi) {
    // Check if processing and valid pick mode
    if (!m_processing || !m_tool || m_pickPointMode == 0) {
        CVLog::Warning(QString("[ecvMeasurementTool] Ignoring pick: "
                               "m_processing=%1, m_tool=%2, m_pickPointMode=%3")
                               .arg(m_processing)
                               .arg(m_tool ? "valid" : "null")
                               .arg(m_pickPointMode));
        return;
    }

    // Check if entity is valid
    if (!pi.entity) {
        CVLog::Warning("[ecvMeasurementTool] Picked item has no entity");
        return;
    }

    // Get picked point coordinates
    double pos[3];
    pos[0] = pi.P3D.x;
    pos[1] = pi.P3D.y;
    pos[2] = pi.P3D.z;

    CVLog::Print(
            QString("[ecvMeasurementTool] Point picked: (%1, %2, %3), mode: %4")
                    .arg(pos[0])
                    .arg(pos[1])
                    .arg(pos[2])
                    .arg(m_pickPointMode));

    // Set the point based on current pick mode
    switch (m_pickPointMode) {
        case 1:  // Point 1
            m_tool->setPoint1(pos);
            m_pickPointMode = 0;
            if (m_pickingHub) {
                m_pickingHub->removeListener(this, true);
            }
            break;
        case 2:  // Point 2
            m_tool->setPoint2(pos);
            m_pickPointMode = 0;
            if (m_pickingHub) {
                m_pickingHub->removeListener(this, true);
            }
            break;
        case 3:  // Center (Protractor only)
            if (m_measurementType ==
                ecvGenericMeasurementTools::PROTRACTOR_WIDGET) {
                m_tool->setCenter(pos);
                m_pickPointMode = 0;
                if (m_pickingHub) {
                    m_pickingHub->removeListener(this, true);
                }
            } else {
                CVLog::Warning(
                        "[ecvMeasurementTool] Center picking only available "
                        "for Protractor");
                m_pickPointMode = 0;
                if (m_pickingHub) {
                    m_pickingHub->removeListener(this, true);
                }
            }
            break;
        default:
            CVLog::Warning(QString("[ecvMeasurementTool] Unknown pick mode: %1")
                                   .arg(m_pickPointMode));
            break;
    }

    // Update UI
    updateUIFromTool();
}

void ecvMeasurementTool::onPointPickingRequested(int pointIndex) {
    // Ensure picking hub is available
    if (!m_pickingHub && MainWindow::TheInstance()) {
        m_pickingHub = MainWindow::TheInstance()->pickingHub();
    }

    m_pickPointMode = pointIndex;

    if (m_pickingHub) {
        if (!m_pickingHub->addListener(this, true, true,
                                       ecvDisplayTools::POINT_PICKING)) {
            CVLog::Warning(
                    "[ecvMeasurementTool] Failed to register picking listener");
            m_pickPointMode = 0;
        }
    } else {
        CVLog::Warning("[ecvMeasurementTool] Picking hub not available!");
        m_pickPointMode = 0;
    }
}

void ecvMeasurementTool::onPointPickingCancelled() {
    m_pickPointMode = 0;
    if (m_pickingHub) {
        m_pickingHub->removeListener(this, true);
    }
}

void ecvMeasurementTool::onColorButtonClicked() {
    // CRITICAL: Get current tool's color as the initial color for the dialog
    // This ensures the color picker shows the correct color for the current
    // tool
    QColor initialColor = QColor(0, 255, 0);  // Default green
#ifdef USE_PCL_BACKEND
    PclMeasurementTools* pclTool = qobject_cast<PclMeasurementTools*>(m_tool);
    if (pclTool) {
        double r = 0.0, g = 1.0, b = 0.0;
        if (pclTool->getColor(r, g, b)) {
            initialColor = QColor::fromRgbF(r, g, b);
        }
    }
#endif

    QColor newColor = QColorDialog::getColor(initialColor, this,
                                             tr("Select Measurement Color"));

    if (newColor.isValid()) {
        // Apply color to tools (all instances of current tool type, or just
        // current based on checkbox)
        applyColorToAllTools(newColor);

        // CRITICAL: Always update UI from current tool to reflect the actual
        // applied color This ensures UI always shows the current tool's state,
        // regardless of "Apply all" setting
        updateUIFromTool();
    }
}

void ecvMeasurementTool::updateColorButtonAppearance(const QColor& color) {
    // Use provided color, or fall back to m_currentColor if not provided
    QColor colorToDisplay = color.isValid() ? color : m_currentColor;

    if (colorButton) {
        QString styleSheet =
                QString("QPushButton { background-color: rgb(%1, %2, %3); }")
                        .arg(colorToDisplay.red())
                        .arg(colorToDisplay.green())
                        .arg(colorToDisplay.blue());
        colorButton->setStyleSheet(styleSheet);
    }
}

void ecvMeasurementTool::applyColorToAllTools(const QColor& color) {
    if (!color.isValid()) return;

    // Convert QColor to normalized RGB [0.0, 1.0]
    double r = color.redF();
    double g = color.greenF();
    double b = color.blueF();

    // CRITICAL: "Apply all" only affects instances of the SAME tool type
    // m_toolInstances already contains only instances of the current tool type
    // (different types are cleaned up in setMeasurementTool)
    if (applyToAllCheckBox && applyToAllCheckBox->isChecked()) {
        // Apply to all instances of the current tool type
        for (ecvGenericMeasurementTools* tool : m_toolInstances) {
            if (tool && tool->getMeasurementType() == m_measurementType) {
                tool->setColor(r, g, b);
            }
        }
    } else {
        // Apply only to the current active tool instance
        if (m_tool) {
            m_tool->setColor(r, g, b);
        }
    }
}

void ecvMeasurementTool::onFontPropertiesChanged() {
    applyFontToTools();

    // CRITICAL: Always update UI from current tool to reflect the actual
    // applied properties This ensures UI always shows the current tool's state,
    // regardless of "Apply all" setting Use QTimer::singleShot to avoid
    // recursive updates during signal processing
    QTimer::singleShot(0, this, [this]() { updateUIFromTool(); });
}

void ecvMeasurementTool::applyFontToTools() {
    if (!m_fontPropertyWidget) return;

    auto props = m_fontPropertyWidget->fontProperties();

    // CRITICAL: "Apply all" only affects instances of the SAME tool type
    // m_toolInstances already contains only instances of the current tool type
    // (different types are cleaned up in setMeasurementTool)
    if (applyToAllCheckBox && applyToAllCheckBox->isChecked()) {
        // Apply to all instances of the current tool type
        for (ecvGenericMeasurementTools* tool : m_toolInstances) {
            if (tool && tool->getMeasurementType() == m_measurementType) {
                tool->setFontFamily(props.family);
                tool->setFontSize(props.size);
                tool->setFontColor(props.color.redF(), props.color.greenF(),
                                   props.color.blueF());
                tool->setBold(props.bold);
                tool->setItalic(props.italic);
                tool->setShadow(props.shadow);
                tool->setFontOpacity(props.opacity);
                tool->setHorizontalJustification(props.horizontalJustification);
                tool->setVerticalJustification(props.verticalJustification);
            }
        }
    } else {
        // Apply only to the current active tool instance
        if (m_tool) {
            m_tool->setFontFamily(props.family);
            m_tool->setFontSize(props.size);
            m_tool->setFontColor(props.color.redF(), props.color.greenF(),
                                 props.color.blueF());
            m_tool->setBold(props.bold);
            m_tool->setItalic(props.italic);
            m_tool->setShadow(props.shadow);
            m_tool->setFontOpacity(props.opacity);
            m_tool->setHorizontalJustification(props.horizontalJustification);
            m_tool->setVerticalJustification(props.verticalJustification);
        }
    }
}
