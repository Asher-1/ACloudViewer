// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PclMeasurementTools.h"

#include <CVLog.h>
#include <PclUtils/PCLVis.h>
#include <ecvDisplayTools.h>

#include "cvContourTool.h"
#include "cvDistanceTool.h"
#include "cvGenericMeasurementTool.h"
#include "cvProtractorTool.h"

PclMeasurementTools::PclMeasurementTools(MeasurementType type)
    : ecvGenericMeasurementTools(type), m_viewer(nullptr), m_tool(nullptr) {
    initialize();
}

PclMeasurementTools::PclMeasurementTools(ecvGenericVisualizer3D* viewer,
                                         MeasurementType type)
    : ecvGenericMeasurementTools(type), m_viewer(nullptr), m_tool(nullptr) {
    setVisualizer(viewer);
    initialize();
}

PclMeasurementTools::~PclMeasurementTools() {
    if (m_tool) {
        delete m_tool;
        m_tool = nullptr;
    }
}

void PclMeasurementTools::initialize() {
    switch (m_measurementType) {
        case MeasurementType::PROTRACTOR_WIDGET:
            m_tool = new cvProtractorTool();
            break;
        case MeasurementType::DISTANCE_WIDGET:
            m_tool = new cvDistanceTool();
            break;
        case MeasurementType::CONTOUR_WIDGET:
            m_tool = new cvContourTool();
            break;
        default:
            CVLog::Error(QString("Unknown measurement type"));
            break;
    }

    if (m_tool && m_viewer) {
        m_tool->setUpViewer(m_viewer);
        if (m_viewer->getRenderWindow() &&
            m_viewer->getRenderWindow()->GetInteractor()) {
            m_tool->setInteractor(m_viewer->getRenderWindow()->GetInteractor());
        }
        // Connect measurement value changed signal
        connect(m_tool, &cvGenericMeasurementTool::measurementValueChanged,
                this, &PclMeasurementTools::measurementChanged);

        // Connect point picking signals - forward from cvGenericMeasurementTool
        // to ecvGenericMeasurementTools
        connect(m_tool, &cvGenericMeasurementTool::pointPickingRequested, this,
                &PclMeasurementTools::pointPickingRequested);
        connect(m_tool, &cvGenericMeasurementTool::pointPickingCancelled, this,
                &PclMeasurementTools::pointPickingCancelled);
    }
}

void PclMeasurementTools::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        m_viewer = reinterpret_cast<PclUtils::PCLVis*>(viewer);
        if (!m_viewer) {
            CVLog::Warning(
                    "[PclMeasurementTools::setVisualizer] viewer is Null!");
            return;
        }
        if (m_tool) {
            m_tool->setUpViewer(m_viewer);
            if (m_viewer->getRenderWindowInteractor()) {
                m_tool->setInteractor(m_viewer->getRenderWindowInteractor());
            }
        }
    } else {
        CVLog::Warning("[PclMeasurementTools::setVisualizer] viewer is Null!");
    }
}

bool PclMeasurementTools::setInputData(ccHObject* entity) {
    m_associatedEntity = entity;
    if (!m_viewer) {
        // Try to get viewer from display tools
        ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
        if (viewer) {
            setVisualizer(viewer);
        }
    }
    if (m_tool) {
        return m_tool->setInput(entity);
    }
    return false;
}

bool PclMeasurementTools::start() {
    if (!m_viewer) {
        // Try to get viewer from display tools
        ecvGenericVisualizer3D* viewer = ecvDisplayTools::GetVisualizer3D();
        if (viewer) {
            setVisualizer(viewer);
        }
    }
    if (m_tool && m_viewer) {
        // Ensure interactor is set
        if (m_viewer->getRenderWindowInteractor()) {
            m_tool->setInteractor(m_viewer->getRenderWindowInteractor());
        }
        m_tool->start();
        return true;
    }
    return false;
}

void PclMeasurementTools::reset() {
    if (m_tool) {
        m_tool->reset();
    }
}

void PclMeasurementTools::clear() {
    if (m_tool) {
        m_tool->clearAllActor();
    }
}

QWidget* PclMeasurementTools::getMeasurementWidget() { return m_tool; }

ccHObject* PclMeasurementTools::getOutput() const {
    if (m_tool) {
        return m_tool->getOutput();
    }
    return nullptr;
}

double PclMeasurementTools::getMeasurementValue() const {
    if (m_tool) {
        return m_tool->getMeasurementValue();
    }
    return 0.0;
}

void PclMeasurementTools::getPoint1(double pos[3]) const {
    if (m_tool) {
        m_tool->getPoint1(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void PclMeasurementTools::getPoint2(double pos[3]) const {
    if (m_tool) {
        m_tool->getPoint2(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void PclMeasurementTools::getCenter(double pos[3]) const {
    if (m_tool) {
        m_tool->getCenter(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void PclMeasurementTools::setPoint1(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setPoint1(pos);
    }
}

void PclMeasurementTools::setPoint2(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setPoint2(pos);
    }
}

void PclMeasurementTools::setCenter(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setCenter(pos);
    }
}

void PclMeasurementTools::setColor(double r, double g, double b) {
    if (m_tool) {
        m_tool->setColor(r, g, b);
    }
}

void PclMeasurementTools::lockInteraction() {
    if (m_tool) {
        m_tool->lockInteraction();
    }
}

void PclMeasurementTools::unlockInteraction() {
    if (m_tool) {
        m_tool->unlockInteraction();
    }
}

void PclMeasurementTools::setInstanceLabel(const QString& label) {
    if (m_tool) {
        m_tool->setInstanceLabel(label);
    }
}

void PclMeasurementTools::setFontFamily(const QString& family) {
    if (m_tool) {
        m_tool->setFontFamily(family);
    }
}

void PclMeasurementTools::setFontSize(int size) {
    if (m_tool) {
        m_tool->setFontSize(size);
    }
}

void PclMeasurementTools::setBold(bool bold) {
    if (m_tool) {
        m_tool->setBold(bold);
    }
}

void PclMeasurementTools::setItalic(bool italic) {
    if (m_tool) {
        m_tool->setItalic(italic);
    }
}

void PclMeasurementTools::setShadow(bool shadow) {
    if (m_tool) {
        m_tool->setShadow(shadow);
    }
}

void PclMeasurementTools::setFontOpacity(double opacity) {
    if (m_tool) {
        m_tool->setFontOpacity(opacity);
    }
}

void PclMeasurementTools::setFontColor(double r, double g, double b) {
    if (m_tool) {
        m_tool->setFontColor(r, g, b);
    }
}

void PclMeasurementTools::setHorizontalJustification(const QString& justification) {
    if (m_tool) {
        m_tool->setHorizontalJustification(justification);
    }
}

void PclMeasurementTools::setVerticalJustification(const QString& justification) {
    if (m_tool) {
        m_tool->setVerticalJustification(justification);
    }
}

void PclMeasurementTools::setupShortcuts(QWidget* win) {
    if (m_tool && win) {
        m_tool->setupShortcuts(win);
    } else {
        CVLog::Warning(
                "[PclMeasurementTools::setupShortcuts] tool or win is Null!");
    }
}

void PclMeasurementTools::disableShortcuts() {
    if (m_tool) {
        m_tool->disableShortcuts();
    }
}

void PclMeasurementTools::clearPickingCache() {
    if (m_tool) {
        m_tool->clearPickingCache();
    }
}
