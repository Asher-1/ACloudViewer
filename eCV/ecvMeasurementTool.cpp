// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMeasurementTool.h"

// LOCAL
#include "MainWindow.h"

// ECV_CORE_LIB
#include <ecvGenericMeasurementTools.h>
#include <ecvPointCloud.h>

// ECV_DB_LIB
#include <ecvPolyline.h>

// CV_CORE_LIB
#include <CVLog.h>

// CV_PLUGIN_API
#include <ecvPickingHub.h>

// QT
#include <QMessageBox>

#ifdef USE_PCL_BACKEND
#include <Tools/MeasurementTools/PclMeasurementTools.h>
#endif

ecvMeasurementTool::ecvMeasurementTool(QWidget* parent)
    : ccOverlayDialog(parent),
      Ui::MeasurementToolDlg(),
      m_tool(nullptr),
      m_updatingFromTool(false),
      m_pickingHub(nullptr),
      m_pickPointMode(0) {
    setupUi(this);

    // Get picking hub from MainWindow
    if (MainWindow::TheInstance()) {
        m_pickingHub = MainWindow::TheInstance()->pickingHub();
    }

    connect(resetButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::reset);
    connect(closeButton, &QToolButton::clicked, this,
            &ecvMeasurementTool::closeDialog);
    connect(showWidgetToolButton, &QToolButton::toggled, this,
            &ecvMeasurementTool::toggleWidget);
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

    // Initially disable instance management buttons
    removeInstanceButton->setEnabled(false);
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

    m_measurementType = tool->getMeasurementType();

    // Add to instances list if not already present
    if (!m_toolInstances.contains(tool)) {
        m_toolInstances.append(tool);

        // Hide the widget initially - we'll show it when it becomes active
        tool->getMeasurementWidget()->setVisible(false);

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

    // Hide all tool widgets
    for (ecvGenericMeasurementTools* t : m_toolInstances) {
        if (t && t->getMeasurementWidget()) {
            QWidget* widget = t->getMeasurementWidget();
            // Remove from layout if present
            if (widget->parent() == nullptr ||
                parametersLayout->indexOf(widget) >= 0) {
                parametersLayout->removeWidget(widget);
            }
            widget->setVisible(false);
        }
    }

    // Show and add the current tool's widget
    QWidget* currentWidget = tool->getMeasurementWidget();
    if (currentWidget) {
        parametersLayout->addWidget(currentWidget);
        currentWidget->setVisible(true);
    }
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
        instancesComboBox->addItem(QString("%1 #%2").arg(typeName).arg(i + 1));
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
        setMeasurementTool(newTool);
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
        parametersLayout->removeWidget(widget);
        widget->setVisible(false);
    }

    // Remove from list
    m_toolInstances.removeAt(index);

    // Delete tool
    if (toolToRemove) {
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
    }
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
            tool->start();
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
            parametersLayout->removeWidget(tool->getMeasurementWidget());
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
        CVLog::Warning(QString("[ecvMeasurementTool] Ignoring pick: m_processing=%1, m_tool=%2, m_pickPointMode=%3")
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
