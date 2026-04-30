// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericFiltersTool.h"

#include <QWidget>

#include "ecvGenericGLDisplay.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"
#include "ecvViewManager.h"

ecvGenericFiltersTool::ecvGenericFiltersTool(FilterType type)
    : m_filterType(type), m_associatedEntity(nullptr) {}

void ecvGenericFiltersTool::update() {
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