// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvGenericMeasurementTool.h"

#include "Tools/PickingTools/cvPointPickingHelper.h"

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// LOCAL
#include "PclUtils/PCLVis.h"
#include "VtkUtils/vtkutils.h"

#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvHObject.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// QT
#include <QApplication>
#include <QSizePolicy>

// VTK
#include <vtkAbstractWidget.h>
#include <vtkActor.h>
#include <vtkProp.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

cvGenericMeasurementTool::cvGenericMeasurementTool(QWidget* parent)
    : QWidget(parent), m_ui(new Ui::GenericMeasurementToolDlg) {
    setWindowTitle(tr("Generic Measurement Tool"));

    // CRITICAL: Set size policy to Minimum (horizontal) to prevent unnecessary
    // expansion This ensures each tool adapts to its content width without
    // extra whitespace ParaView-style: use Minimum to prevent horizontal
    // expansion beyond content
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    // CRITICAL: Remove any fixed size constraints to allow Qt's layout system
    // to work This prevents UI size interference between different tool types
    setMinimumSize(0, 0);
    setMaximumSize(16777215, 16777215);  // QWIDGETSIZE_MAX equivalent
}

cvGenericMeasurementTool::~cvGenericMeasurementTool() {
    // Disable all picking helpers before deletion to prevent shortcuts from
    // triggering after tool is destroyed
    for (cvPointPickingHelper* helper : m_pickingHelpers) {
        if (helper) {
            helper->setEnabled(false, false);
        }
    }

    // Clean up picking helpers
    qDeleteAll(m_pickingHelpers);
    m_pickingHelpers.clear();

    clearAllActor();
    delete m_ui;
}

////////////////////Initialization///////////////////////////
void cvGenericMeasurementTool::start() {
    // create UI first (must be done before initTool)
    createUi();

    // call child function
    modelReady();

    // update data
    dataChanged();

    // according to data type
    initTool();

    // Clear selection cache when starting tool (new session)
    clearPickingCache();

    // update screen
    update();
}

void cvGenericMeasurementTool::setupShortcuts(QWidget* win) {
    if (!win) {
        CVLog::Warning(
                "[cvGenericMeasurementTool] Widget is null, shortcuts not "
                "setup");
        return;
    }

    // Save the widget reference for later use (e.g., when unlocking after being
    // locked)
    m_vtkWidget = win;

    // Only create shortcuts if they don't already exist
    if (m_pickingHelpers.isEmpty()) {
        CVLog::PrintDebug(QString("[cvGenericMeasurementTool::setupShortcuts] "
                                  "Tool=%1, creating shortcuts...")
                                  .arg((quintptr)this, 0, 16));

        // Setup keyboard shortcuts for point picking
        setupPointPickingShortcuts(win);

        // Update picking helpers with current interactor/renderer
        updatePickingHelpers();

        CVLog::PrintDebug(QString("[cvGenericMeasurementTool::setupShortcuts] "
                                  "Tool=%1, created %2 shortcuts")
                                  .arg((quintptr)this, 0, 16)
                                  .arg(m_pickingHelpers.size()));
    } else {
        CVLog::PrintDebug(
                QString("[cvGenericMeasurementTool::setupShortcuts] Tool=%1, "
                        "shortcuts already exist (%2), updating only")
                        .arg((quintptr)this, 0, 16)
                        .arg(m_pickingHelpers.size()));

        // Just update the interactor/renderer
        updatePickingHelpers();
    }

    // Enable all shortcuts and set focus to the widget
    for (cvPointPickingHelper* helper : m_pickingHelpers) {
        if (helper) {
            helper->setEnabled(true, true);  // Enable and set focus
        }
    }
}

bool cvGenericMeasurementTool::setInput(ccHObject* obj) {
    m_entity = obj;
    if (!m_entity) {
        return false;
    }
    m_id = m_entity->getViewId().toStdString();

    if (!initModel()) {
        return false;
    }

    return true;
}

bool cvGenericMeasurementTool::initModel() {
    if (!m_entity || !m_viewer) {
        return false;
    }

    // Get renderer from viewer
    m_renderer = m_viewer->getCurrentRenderer(0);
    if (!m_renderer) {
        CVLog::Error("[cvGenericMeasurementTool] Failed to get renderer!");
        return false;
    }

    return true;
}

void cvGenericMeasurementTool::modelReady() {
    // Override in derived classes if needed
}

void cvGenericMeasurementTool::update() {
    if (m_viewer && m_renderer) {
        if (m_viewer->getRenderWindow()) {
            m_viewer->getRenderWindow()->Render();
            // Clear picking cache after render as scene may have changed
            clearPickingCache();
        }
    }
}

void cvGenericMeasurementTool::reset() {
    // Override in derived classes if needed
}

ccHObject* cvGenericMeasurementTool::getOutput() {
    // Override in derived classes if needed
    return nullptr;
}

void cvGenericMeasurementTool::setUpViewer(PclUtils::PCLVis* viewer) {
    m_viewer = viewer;
    if (m_viewer) {
        m_renderer = m_viewer->getCurrentRenderer(0);
        if (m_viewer->getRenderWindow()) {
            m_interactor = m_viewer->getRenderWindow()->GetInteractor();
        }
    }
}

void cvGenericMeasurementTool::setInteractor(
        vtkRenderWindowInteractor* interactor) {
    m_interactor = interactor;
}

void cvGenericMeasurementTool::addActor(const vtkSmartPointer<vtkProp> actor) {
    if (m_renderer && actor) {
        m_renderer->AddActor(actor);
    }
}

void cvGenericMeasurementTool::removeActor(
        const vtkSmartPointer<vtkProp> actor) {
    if (m_renderer && actor) {
        m_renderer->RemoveActor(actor);
    }
}

void cvGenericMeasurementTool::clearAllActor() {
    if (m_renderer) {
        if (m_modelActor) {
            m_renderer->RemoveActor(m_modelActor);
        }
    }
}

void cvGenericMeasurementTool::safeOff(vtkAbstractWidget* widget) {
    if (widget) {
        widget->Off();
    }
}

void cvGenericMeasurementTool::updatePickingHelpers() {
    for (cvPointPickingHelper* helper : m_pickingHelpers) {
        if (helper) {
            helper->setInteractor(m_interactor);
            helper->setRenderer(m_renderer);
        }
    }
}

void cvGenericMeasurementTool::disableShortcuts() {
    CVLog::PrintDebug(QString("[cvGenericMeasurementTool::disableShortcuts] "
                              "Tool=%1, disabling %2 shortcuts")
                              .arg((quintptr)this, 0, 16)
                              .arg(m_pickingHelpers.size()));

    // Disable all picking helpers to prevent shortcuts from triggering
    for (cvPointPickingHelper* helper : m_pickingHelpers) {
        if (helper) {
            helper->setEnabled(false, false);
        }
    }
}

void cvGenericMeasurementTool::clearPickingCache() {
    // Clear selection cache in all picking helpers
    // Call this when scene changes, camera moves significantly, or data updates
    for (cvPointPickingHelper* helper : m_pickingHelpers) {
        if (helper) {
            helper->clearSelectionCache();
        }
    }
}

void cvGenericMeasurementTool::setFontFamily(const QString& family) {
    m_fontFamily = family;
    applyFontProperties();
}

void cvGenericMeasurementTool::setFontSize(int size) {
    m_fontSize = size;
    applyFontProperties();
}

void cvGenericMeasurementTool::setBold(bool bold) {
    m_fontBold = bold;
    applyFontProperties();
}

void cvGenericMeasurementTool::setItalic(bool italic) {
    m_fontItalic = italic;
    applyFontProperties();
}

void cvGenericMeasurementTool::setShadow(bool shadow) {
    m_fontShadow = shadow;
    applyFontProperties();
}

void cvGenericMeasurementTool::setFontOpacity(double opacity) {
    m_fontOpacity = opacity;
    applyFontProperties();
}

void cvGenericMeasurementTool::setFontColor(double r, double g, double b) {
    m_fontColor[0] = r;
    m_fontColor[1] = g;
    m_fontColor[2] = b;
    applyFontProperties();
}

void cvGenericMeasurementTool::setHorizontalJustification(
        const QString& justification) {
    m_horizontalJustification = justification;
    applyFontProperties();
}

void cvGenericMeasurementTool::setVerticalJustification(
        const QString& justification) {
    m_verticalJustification = justification;
    applyFontProperties();
}
