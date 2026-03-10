// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file VtkMeasurementTools.cpp
 * @brief Implementation of measurement tools (distance, contour, protractor).
 */

#include "VtkMeasurementTools.h"

#include <CVLog.h>
#include <Visualization/VtkVis.h>
#include <ecvDisplayTools.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include "cvContourTool.h"
#include "cvDistanceTool.h"
#include "cvGenericMeasurementTool.h"
#include "cvProtractorTool.h"

VtkMeasurementTools::VtkMeasurementTools(MeasurementType type)
    : ecvGenericMeasurementTools(type), m_viewer(nullptr), m_tool(nullptr) {
    initialize();
}

VtkMeasurementTools::VtkMeasurementTools(ecvGenericVisualizer3D* viewer,
                                         MeasurementType type)
    : ecvGenericMeasurementTools(type), m_viewer(nullptr), m_tool(nullptr) {
    setVisualizer(viewer);
    initialize();
}

VtkMeasurementTools::~VtkMeasurementTools() {
    if (m_tool) {
        delete m_tool;
        m_tool = nullptr;
    }
}

void VtkMeasurementTools::initialize() {
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
                this, &VtkMeasurementTools::measurementChanged);

        // Connect point picking signals - forward from cvGenericMeasurementTool
        // to ecvGenericMeasurementTools
        connect(m_tool, &cvGenericMeasurementTool::pointPickingRequested, this,
                &VtkMeasurementTools::pointPickingRequested);
        connect(m_tool, &cvGenericMeasurementTool::pointPickingCancelled, this,
                &VtkMeasurementTools::pointPickingCancelled);
    }
}

void VtkMeasurementTools::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        m_viewer = reinterpret_cast<Visualization::VtkVis*>(viewer);
        if (!m_viewer) {
            CVLog::Warning(
                    "[VtkMeasurementTools::setVisualizer] viewer is Null!");
            return;
        }
        if (m_tool) {
            m_tool->setUpViewer(m_viewer);
            if (m_viewer->getRenderWindowInteractor()) {
                m_tool->setInteractor(m_viewer->getRenderWindowInteractor());
            }
        }
    } else {
        CVLog::Warning("[VtkMeasurementTools::setVisualizer] viewer is Null!");
    }
}

bool VtkMeasurementTools::setInputData(ccHObject* entity) {
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

bool VtkMeasurementTools::start() {
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

void VtkMeasurementTools::reset() {
    if (m_tool) {
        m_tool->reset();
    }
}

void VtkMeasurementTools::clear() {
    if (m_tool) {
        m_tool->clearAllActor();
    }
}

QWidget* VtkMeasurementTools::getMeasurementWidget() { return m_tool; }

ccHObject* VtkMeasurementTools::getOutput() const {
    if (m_tool) {
        return m_tool->getOutput();
    }
    return nullptr;
}

double VtkMeasurementTools::getMeasurementValue() const {
    if (m_tool) {
        return m_tool->getMeasurementValue();
    }
    return 0.0;
}

void VtkMeasurementTools::getPoint1(double pos[3]) const {
    if (m_tool) {
        m_tool->getPoint1(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void VtkMeasurementTools::getPoint2(double pos[3]) const {
    if (m_tool) {
        m_tool->getPoint2(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void VtkMeasurementTools::getCenter(double pos[3]) const {
    if (m_tool) {
        m_tool->getCenter(pos);
    } else if (pos) {
        pos[0] = pos[1] = pos[2] = 0.0;
    }
}

void VtkMeasurementTools::setPoint1(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setPoint1(pos);
    }
}

void VtkMeasurementTools::setPoint2(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setPoint2(pos);
    }
}

void VtkMeasurementTools::setCenter(double pos[3]) {
    if (m_tool && pos) {
        m_tool->setCenter(pos);
    }
}

void VtkMeasurementTools::setColor(double r, double g, double b) {
    if (m_tool) {
        m_tool->setColor(r, g, b);
    }
}

bool VtkMeasurementTools::getColor(double& r, double& g, double& b) const {
    if (m_tool) {
        return m_tool->getColor(r, g, b);
    }
    return false;
}

void VtkMeasurementTools::lockInteraction() {
    if (m_tool) {
        m_tool->lockInteraction();
    }
}

void VtkMeasurementTools::unlockInteraction() {
    if (m_tool) {
        m_tool->unlockInteraction();
    }
}

void VtkMeasurementTools::setInstanceLabel(const QString& label) {
    if (m_tool) {
        m_tool->setInstanceLabel(label);
    }
}

void VtkMeasurementTools::setFontFamily(const QString& family) {
    if (m_tool) {
        m_tool->setFontFamily(family);
    }
}

void VtkMeasurementTools::setFontSize(int size) {
    if (m_tool) {
        m_tool->setFontSize(size);
    }
}

void VtkMeasurementTools::setBold(bool bold) {
    if (m_tool) {
        m_tool->setBold(bold);
    }
}

void VtkMeasurementTools::setItalic(bool italic) {
    if (m_tool) {
        m_tool->setItalic(italic);
    }
}

void VtkMeasurementTools::setShadow(bool shadow) {
    if (m_tool) {
        m_tool->setShadow(shadow);
    }
}

void VtkMeasurementTools::setFontOpacity(double opacity) {
    if (m_tool) {
        m_tool->setFontOpacity(opacity);
    }
}

void VtkMeasurementTools::setFontColor(double r, double g, double b) {
    if (m_tool) {
        m_tool->setFontColor(r, g, b);
    }
}

QString VtkMeasurementTools::getFontFamily() const {
    return m_tool ? m_tool->getFontFamily() : QString("Arial");
}

int VtkMeasurementTools::getFontSize() const {
    return m_tool ? m_tool->getFontSize() : 6;
}

void VtkMeasurementTools::getFontColor(double& r, double& g, double& b) const {
    if (m_tool) {
        m_tool->getFontColor(r, g, b);
    } else {
        r = g = b = 1.0;  // Default white
    }
}

bool VtkMeasurementTools::getFontBold() const {
    return m_tool ? m_tool->getFontBold() : false;
}

bool VtkMeasurementTools::getFontItalic() const {
    return m_tool ? m_tool->getFontItalic() : false;
}

bool VtkMeasurementTools::getFontShadow() const {
    return m_tool ? m_tool->getFontShadow() : true;
}

double VtkMeasurementTools::getFontOpacity() const {
    return m_tool ? m_tool->getFontOpacity() : 1.0;
}

QString VtkMeasurementTools::getHorizontalJustification() const {
    return m_tool ? m_tool->getHorizontalJustification() : QString("Left");
}

QString VtkMeasurementTools::getVerticalJustification() const {
    return m_tool ? m_tool->getVerticalJustification() : QString("Bottom");
}

void VtkMeasurementTools::setHorizontalJustification(
        const QString& justification) {
    if (m_tool) {
        m_tool->setHorizontalJustification(justification);
    }
}

void VtkMeasurementTools::setVerticalJustification(
        const QString& justification) {
    if (m_tool) {
        m_tool->setVerticalJustification(justification);
    }
}

void VtkMeasurementTools::setupShortcuts(QWidget* win) {
    if (m_tool && win) {
        m_tool->setupShortcuts(win);
    } else {
        CVLog::Warning(
                "[VtkMeasurementTools::setupShortcuts] tool or win is Null!");
    }
}

void VtkMeasurementTools::disableShortcuts() {
    if (m_tool) {
        m_tool->disableShortcuts();
    }
}

void VtkMeasurementTools::clearPickingCache() {
    if (m_tool) {
        m_tool->clearPickingCache();
    }
}
