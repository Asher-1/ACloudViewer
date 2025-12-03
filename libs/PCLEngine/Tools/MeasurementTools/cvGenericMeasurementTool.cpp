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

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvHObject.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

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
}

cvGenericMeasurementTool::~cvGenericMeasurementTool() {
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

    // update screen
    update();
}

void cvGenericMeasurementTool::setupShortcuts(QWidget* win) {
    if (!win) {
        CVLog::Warning("[cvGenericMeasurementTool] Widget is null, shortcuts not setup");
        return;
    }
    
    // Setup keyboard shortcuts for point picking
    setupPointPickingShortcuts(win);
    
    // Update picking helpers with current interactor/renderer
    updatePickingHelpers();
    
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

