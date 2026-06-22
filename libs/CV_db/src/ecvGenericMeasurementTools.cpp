// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericMeasurementTools.h"

#include <QWidget>

#include "ecvGenericGLDisplay.h"
#include "ecvViewManager.h"

ecvGenericMeasurementTools::ecvGenericMeasurementTools(MeasurementType type)
    : m_measurementType(type), m_associatedEntity(nullptr) {}

ecvGenericMeasurementTools::~ecvGenericMeasurementTools() {
    // Empty destructor - required for vtable generation
}

void ecvGenericMeasurementTools::update() {
    if (QWidget* w = ecvViewManager::instance().activeWidget()) {
        w->update();
    }
    if (ecvGenericGLDisplay* v =
                ecvViewManager::instance().getEffectiveView()) {
        v->updateScene();
    }
    if (ecvViewManager::instance().viewCount() > 1) {
        ecvViewManager::instance().refreshAll();
    }
}
