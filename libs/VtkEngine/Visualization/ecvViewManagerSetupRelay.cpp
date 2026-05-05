// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewManagerSetupRelay.h"

#include <ecvDisplayTools.h>
#include <ecvViewManager.h>

#include "ecvGLView.h"

namespace Visualization {

namespace {

// Per-view relay: each ecvGLView's signals → ecvViewManager
// (future path for true per-view signal attribution)
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
    QObject::connect(
            glView,
            QOverload<int, int, Qt::MouseButtons>::of(&ecvGLView::mouseMoved),
            mgr, &ecvViewManager::mouseMoved);
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

// Bridge: ecvDisplayTools "result" signals → ecvViewManager.
// ecvDisplayTools emits picking results, camera-state changes, and
// entity-selection changes via `emit primaryDT()->signal(...)`.
// ecvGLView does NOT re-emit these, so without this bridge the
// ecvViewManager consumers (MainWindow, ecvPickingHub, dialogs)
// would never receive them.  UniqueConnection prevents duplicate
// delivery if an ecvGLView relay fires the same signal in the future.
static bool s_displayToolsBridgeInstalled = false;

static void installDisplayToolsBridge(ecvViewManager* mgr) {
    if (s_displayToolsBridgeInstalled) return;
    auto* dt = mgr->displayTools();
    if (!dt) return;

    auto uc = Qt::UniqueConnection;

    // Entity / picking result signals
    QObject::connect(dt, &ecvDisplayTools::entitySelectionChanged,
                     mgr, &ecvViewManager::entitySelectionChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::entitiesSelectionChanged,
                     mgr, &ecvViewManager::entitiesSelectionChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::itemPicked,
                     mgr, &ecvViewManager::itemPicked, uc);
    QObject::connect(dt, &ecvDisplayTools::itemPickedFast,
                     mgr, &ecvViewManager::itemPickedFast, uc);
    QObject::connect(dt, &ecvDisplayTools::newLabel,
                     mgr, &ecvViewManager::newLabel, uc);
    QObject::connect(dt, &ecvDisplayTools::exclusiveFullScreenToggled,
                     mgr, &ecvViewManager::exclusiveFullScreenToggled, uc);

    // Camera / view-state signals
    QObject::connect(dt, &ecvDisplayTools::cameraParamChanged,
                     mgr, &ecvViewManager::cameraParamChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::perspectiveStateChanged,
                     mgr, &ecvViewManager::perspectiveStateChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::pivotPointChanged,
                     mgr, &ecvViewManager::pivotPointChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::baseViewMatChanged,
                     mgr, &ecvViewManager::baseViewMatChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::fovChanged,
                     mgr, &ecvViewManager::fovChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::zNearCoefChanged,
                     mgr, &ecvViewManager::zNearCoefChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::cameraPosChanged,
                     mgr, &ecvViewManager::cameraPosChanged, uc);
    QObject::connect(dt, &ecvDisplayTools::cameraDisplaced,
                     mgr, &ecvViewManager::cameraDisplaced, uc);

    s_displayToolsBridgeInstalled = true;
}

}  // namespace

void registerViewManagerTypedRelay() {
    ecvViewManager::registerSingletonRelayHook(&typedSingletonRelay);
}

void installDisplayToolsRelay() {
    installDisplayToolsBridge(&ecvViewManager::instance());
}

}  // namespace Visualization
