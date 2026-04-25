// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CV_db.h"
#include "ecvGLMatrix.h"
#include "ecvGenericGLDisplay.h"
#include "ecvViewportParameters.h"

#include <QPoint>
#include <QRect>
#include <QString>

#include <cstring>

/// Per-view state container — groups every piece of display state that
/// is logically owned by a single 3D view window.
///
/// Designed to be trivially copyable (memcpy-safe for POD sub-structs,
/// value-semantics for Qt/C++ value types) so that push/pull between
/// an ecvGLView and the ecvDisplayTools singleton is a single
/// struct assignment.
///
/// **Phase A deliverable.**  Later phases will make this the *only*
/// source of truth for per-view state, replacing direct singleton
/// member access.
///
/// Reference models:
///   CloudCompare — ccGLWindowInterface owns all per-window state.
///   ParaView     — pqView / vtkSMViewProxy owns per-view state.
struct CV_DB_LIB_API ecvViewContext {
    // ================================================================
    // Viewport / Camera
    // ================================================================

    ecvViewportParameters viewportParams;
    QRect glViewport;
    ccGLMatrixd viewMatd;
    ccGLMatrixd projMatd;
    bool validModelviewMatrix = false;
    bool validProjectionMatrix = false;
    double cameraToBBCenterDist = 1.0;
    double bbHalfDiag = 1.0;

    // ================================================================
    // Bubble view
    // ================================================================

    bool bubbleViewModeEnabled = false;
    float bubbleViewFov_deg = 90.0f;
    ecvViewportParameters preBubbleViewParameters;

    // ================================================================
    // Interaction / Picking
    // ================================================================

    ecvGenericGLDisplay::INTERACTION_FLAGS interactionFlags =
            ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA;
    ecvGenericGLDisplay::PICKING_MODE pickingMode =
            ecvGenericGLDisplay::DEFAULT_PICKING;
    bool pickingModeLocked = false;
    int pickRadius = 3;
    bool allowRectangularEntityPicking = true;

    // ================================================================
    // Picking results
    // ================================================================

    CCVector3 lastPickedPoint{0, 0, 0};
    int lastPointIndex = -1;
    QString lastPickedId;

    // ================================================================
    // Mouse / Touch
    // ================================================================

    QPoint lastMousePos;
    QPoint lastMouseMovePos;
    bool mouseMoved = false;
    bool mouseButtonPressed = false;
    bool touchInProgress = false;
    qreal touchBaseDist = 1.0;
    bool ignoreMouseReleaseEvent = false;
    bool widgetClicked = false;

    // ================================================================
    // Display flags
    // ================================================================

    bool clickableItemsVisible = false;
    bool displayOverlayEntities = true;
    bool exclusiveFullscreen = false;
    bool showCursorCoordinates = false;
    bool showDebugTraces = false;

    // ================================================================
    // Pivot
    // ================================================================

    ecvGenericGLDisplay::PivotVisibility pivotVisibility =
            ecvGenericGLDisplay::PIVOT_SHOW_ON_MOVE;
    bool pivotSymbolShown = false;
    bool autoPickPivotAtCenter = false;
    CCVector3d autoPivotCandidate{0, 0, 0};

    // ================================================================
    // Rotation lock
    // ================================================================

    bool rotationAxisLocked = false;
    CCVector3d lockedRotationAxis{0, 0, 1};

    // ================================================================
    // Light
    // ================================================================

    bool sunLightEnabled = true;
    bool customLightEnabled = false;
    float customLightPos[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ================================================================
    // Timer
    // ================================================================

    qint64 lastClickTime_ticks = 0;

    // ================================================================
    // Helpers
    // ================================================================

    void copyLightPos(const float src[4]) {
        std::memcpy(customLightPos, src, sizeof(customLightPos));
    }

    void resetInteractionState() {
        mouseMoved = false;
        mouseButtonPressed = false;
        touchInProgress = false;
        touchBaseDist = 1.0;
        ignoreMouseReleaseEvent = false;
        widgetClicked = false;
        lastClickTime_ticks = 0;
        lastMousePos = QPoint();
        lastMouseMovePos = QPoint();
        lastPickedPoint = CCVector3(0, 0, 0);
        lastPointIndex = -1;
        lastPickedId.clear();
    }
};
