// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewManagerSetupRelay.h"

#include <ecvViewManager.h>

#include "ecvGLView.h"

namespace Visualization {

namespace {

static void typedSingletonRelay(ecvViewManager* mgr,
                                ecvGenericGLDisplay* view) {
    auto* glView = dynamic_cast<ecvGLView*>(view);
    if (!glView) return;

    QObject::connect(glView, &ecvGLView::entitySelectionChanged, mgr,
                     &ecvViewManager::entitySelectionChanged);
    QObject::connect(glView, &ecvGLView::entitiesSelectionChanged, mgr,
                     &ecvViewManager::entitiesSelectionChanged);
    QObject::connect(glView, &ecvGLView::newLabel, mgr,
                     &ecvViewManager::newLabel);
    QObject::connect(glView, &ecvGLView::filesDropped, mgr,
                     &ecvViewManager::filesDropped);
    QObject::connect(glView, &ecvGLView::cameraParamChanged, mgr,
                     &ecvViewManager::cameraParamChanged);
    QObject::connect(glView, &ecvGLView::mousePosChanged, mgr,
                     &ecvViewManager::mousePosChanged);
    QObject::connect(glView, &ecvGLView::autoPickPivot, mgr,
                     &ecvViewManager::autoPickPivot);
    QObject::connect(glView, &ecvGLView::exclusiveFullScreenToggled, mgr,
                     &ecvViewManager::exclusiveFullScreenToggled);
    QObject::connect(glView, &ecvGLView::itemPicked, mgr,
                     &ecvViewManager::itemPicked);
    QObject::connect(glView, &ecvGLView::mouseMoved, mgr,
                     &ecvViewManager::mouseMoved);
    QObject::connect(glView, &ecvGLView::leftButtonClicked, mgr,
                     &ecvViewManager::leftButtonClicked);
    QObject::connect(glView, &ecvGLView::rightButtonClicked, mgr,
                     &ecvViewManager::rightButtonClicked);
    QObject::connect(glView, &ecvGLView::doubleButtonClicked, mgr,
                     &ecvViewManager::doubleButtonClicked);
    QObject::connect(glView, &ecvGLView::buttonReleased, mgr,
                     &ecvViewManager::buttonReleased);
    QObject::connect(glView, &ecvGLView::labelmove2D, mgr,
                     &ecvViewManager::labelmove2D);
    QObject::connect(glView, &ecvGLView::pivotPointChanged, mgr,
                     &ecvViewManager::pivotPointChanged);
    QObject::connect(glView, &ecvGLView::perspectiveStateChanged, mgr,
                     &ecvViewManager::perspectiveStateChanged);
}

}  // namespace

void registerViewManagerTypedRelay() {
    ecvViewManager::registerSingletonRelayHook(&typedSingletonRelay);
}

}  // namespace Visualization
