// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef USE_VLD
// VLD
#include <vld.h>
#endif

// CV_CORE_LIB
#include <CVTools.h>

// LOCAL
#include "LineSet.h"
#include "ecv2DLabel.h"
#include "ecv2DViewportLabel.h"
#include "ecvBBox.h"
#include "ecvCameraSensor.h"
#include "ecvClipBox.h"
#include "ecvDisplayTools.h"
#include "ecvGenericGLDisplay.h"
#include "ecvGenericMesh.h"
#include "ecvGenericVisualizer.h"
#include "ecvGenericVisualizer2D.h"
#include "ecvGenericVisualizer3D.h"
#include "ecvHObjectCaster.h"
#include "ecvInteractor.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvRedrawScope.h"
#include "ecvRenderingTools.h"
#include "ecvSphere.h"
#include "ecvSubMesh.h"
#include "ecvViewManager.h"

// STD
#include <cstring>
#include <limits>

// QT
#include <QApplication>
#include <QElapsedTimer>
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QLayout>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QScreen>
#include <QSettings>
#include <QString>

// SYSTEM
#include <assert.h>

// Convenience accessor for the shared ecvDisplayTools instance managed by
// ecvViewManager.  Not a singleton — the instance lifecycle is owned by
// ecvViewManager.  Gap S in the architecture plan tracks migrating these
// call sites to per-view signal emission.
static ecvDisplayTools* primaryDT() {
    return ecvViewManager::instance().displayTools();
}

namespace {

/// The ecvGenericGLDisplay whose viewContext() is `ctx`, or the display-tools
/// engine as fallback.
ecvGenericGLDisplay* displayOwningContext(const ecvViewContext* ctxPtr) {
    auto* primary = primaryDT();
    if (!primary) return nullptr;
    if (!ctxPtr) return primary;
    for (auto* v : ecvViewManager::instance().getAllViews()) {
        if (v && v->viewContext() == ctxPtr) return v;
    }
    return primary;
}

/// Carries forceRedraw from RedrawDisplay(bool) into RedrawDisplay(ctx)
/// without changing the public ctx overload signature.
bool g_redrawDisplayForceRedraw = true;

}  // namespace

bool ecvDisplayTools::USE_2D = true;
bool ecvDisplayTools::USE_VTK_PICK = false;

/// Returns a non-primary active view (ecvGenericGLDisplay*) if one exists
/// and it differs from the primary ecvDisplayTools (ecvViewManager).
/// Used by static methods to delegate to the per-view implementation.
static ecvGenericGLDisplay* activeSecondaryView() {
    auto* av = ecvViewManager::instance().getEffectiveView();
    if (av && av != primaryDT()) return av;
    return nullptr;
}

static const QString DEBUG_LAYER_ID = "DEBUG_LAYER";

static int viewportHeightFor(ecvGenericGLDisplay* display) {
    if (display && display != primaryDT()) {
        auto* ctx = display->viewContext();
        if (ctx) return ctx->glViewport.height();
    }
    return ecvViewManager::instance().resolveViewContext().glViewport.height();
}

// ================================================================
// Context-aware static API overloads  (Phase A)
// ================================================================

void ecvDisplayTools::GetContext(CC_DRAW_CONTEXT& CONTEXT,
                                 const ecvViewContext& viewCtx) {
    CONTEXT.glW = viewCtx.glViewport.width();
    CONTEXT.glH = viewCtx.glViewport.height();
    CONTEXT.devicePixelRatio = 1.0f;
    CONTEXT.drawingFlags = 0;

    const ecvGui::ParamStruct& guiParams = GetDisplayParameters();

    CONTEXT.decimateCloudOnMove = guiParams.decimateCloudOnMove;
    CONTEXT.minLODPointCount = guiParams.minLoDCloudSize;
    CONTEXT.minLODTriangleCount = guiParams.minLoDMeshSize;
    CONTEXT.higherLODLevelsAvailable = false;
    CONTEXT.moreLODPointsAvailable = false;
    CONTEXT.currentLODLevel = 0;

    CONTEXT.sfColorScaleToDisplay = nullptr;

    CONTEXT.labelMarkerSize = static_cast<float>(guiParams.labelMarkerSize *
                                                 ComputeActualPixelSize());
    CONTEXT.labelMarkerTextShift_pix = 5;

    CONTEXT.dispNumberPrecision = guiParams.displayedNumPrecision;
    CONTEXT.labelOpacity = guiParams.labelOpacity;

    CONTEXT.defaultMat->setDiffuseFront(guiParams.meshFrontDiff);
    CONTEXT.defaultMat->setDiffuseBack(guiParams.meshBackDiff);
    CONTEXT.defaultMat->setAmbient(ecvColor::bright);
    CONTEXT.defaultMat->setSpecular(guiParams.meshSpecular);
    CONTEXT.defaultMat->setEmission(ecvColor::night);
    CONTEXT.defaultMat->setShininessFront(30);
    CONTEXT.defaultMat->setShininessBack(50);

    CONTEXT.pointsDefaultCol = guiParams.pointsDefaultCol;
    CONTEXT.textDefaultCol = guiParams.textDefaultCol;
    CONTEXT.labelDefaultBkgCol = guiParams.labelBackgroundCol;
    CONTEXT.labelDefaultMarkerCol = guiParams.labelMarkerCol;
    CONTEXT.bbDefaultCol = guiParams.bbDefaultCol;

    CONTEXT.defaultPointSize =
            static_cast<unsigned char>(viewCtx.viewportParams.defaultPointSize);
    CONTEXT.defaultLineWidth =
            static_cast<unsigned char>(viewCtx.viewportParams.defaultLineWidth);
    CONTEXT.currentLineWidth = CONTEXT.defaultLineWidth;
}

void ecvDisplayTools::GetGLCameraParameters(ccGLCameraParameters& params,
                                            const ecvViewContext& viewCtx) {
    params.modelViewMat = viewCtx.viewMatd;
    params.projectionMat = viewCtx.projMatd;
    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = viewCtx.glViewport.width();
    params.viewport[3] = viewCtx.glViewport.height();
    params.perspective = viewCtx.viewportParams.perspectiveView;
    params.fov_deg = viewCtx.viewportParams.fov_deg;
    params.pixelSize = viewCtx.viewportParams.pixelSize;
}

void ecvDisplayTools::SetPointSize(ecvViewContext& ctx, float size) {
    ctx.viewportParams.defaultPointSize =
            std::max(std::min(size, MAX_POINT_SIZE_F), MIN_POINT_SIZE_F);
}

void ecvDisplayTools::SetLineWidth(ecvViewContext& ctx, float width) {
    ctx.viewportParams.defaultLineWidth =
            std::max(std::min(width, MAX_LINE_WIDTH_F), MIN_LINE_WIDTH_F);
}

void ecvDisplayTools::SetCameraClip(ecvViewContext& ctx,
                                    double znear,
                                    double zfar) {
    ctx.viewportParams.zNear = znear;
    ctx.viewportParams.zFar = zfar;
}

void ecvDisplayTools::SetCameraFovy(ecvViewContext& ctx, double fovy) {
    ctx.viewportParams.fov_deg = static_cast<float>(fovy);
}

ecvDisplayTools::PivotVisibility ecvDisplayTools::GetPivotVisibility(
        const ecvViewContext& ctx) {
    return ctx.pivotVisibility;
}

void ecvDisplayTools::SetInteractionMode(ecvViewContext& ctx,
                                         INTERACTION_FLAGS flags) {
    ctx.interactionFlags = flags;
}

ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::GetInteractionMode(
        const ecvViewContext& ctx) {
    return ctx.interactionFlags;
}

void ecvDisplayTools::SetPickingMode(ecvViewContext& ctx, PICKING_MODE mode) {
    ctx.pickingMode = mode;
}

ecvDisplayTools::PICKING_MODE ecvDisplayTools::GetPickingMode(
        const ecvViewContext& ctx) {
    return ctx.pickingMode;
}

// default interaction flags
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::PAN_ONLY() {
    ecvDisplayTools::INTERACTION_FLAGS flags =
            INTERACT_PAN | INTERACT_ZOOM_CAMERA | INTERACT_2D_ITEMS |
            INTERACT_CLICKABLE_ITEMS;
    return flags;
}
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::TRANSFORM_CAMERA() {
    ecvDisplayTools::INTERACTION_FLAGS flags = INTERACT_ROTATE | PAN_ONLY();
    return flags;
}
ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::TRANSFORM_ENTITIES() {
    ecvDisplayTools::INTERACTION_FLAGS flags =
            INTERACT_ROTATE | INTERACT_PAN | INTERACT_ZOOM_CAMERA |
            INTERACT_TRANSFORM_ENTITIES | INTERACT_CLICKABLE_ITEMS;
    return flags;
}

/*** Persistent settings ***/

static const char c_ps_groupName[] = "ECVWindow";
static const char c_ps_perspectiveView[] = "perspectiveView";
static const char c_ps_objectMode[] = "objectCenteredView";
static const char c_ps_sunLight[] = "sunLightEnabled";
static const char c_ps_customLight[] = "customLightEnabled";
static const char c_ps_pivotVisibility[] = "pivotVisibility";
static const char c_ps_stereoGlassType[] = "stereoGlassType";

// Vaious overlay elements dimensions
static const double CC_DISPLAYED_PIVOT_RADIUS_PERCENT =
        0.8;  // percentage of the smallest screen dimension
static const double CC_DISPLAYED_CUSTOM_LIGHT_LENGTH = 10.0;
static const float CC_DISPLAYED_TRIHEDRON_AXES_LENGTH = 25.0f;
static const float CC_TRIHEDRON_TEXT_MARGIN = 5.0f;
static const float CC_DISPLAYED_CENTER_CROSS_LENGTH = 10.0f;

// Max click duration for enabling picking mode (in ms)
static const int CC_MAX_PICKING_CLICK_DURATION_MS = 200;

// Unique GL window ID
static int s_GlWindowNumber = 0;

void ecvDisplayTools::initializeEngine(QMainWindow* win, bool stereoMode) {
    m_timer.start();

    SetMainWindow(win);
    registerVisualizer(win, stereoMode);

    m_uniqueID = ++s_GlWindowNumber;  // GL window unique ID
    auto& ctx = ecvViewManager::instance().resolveViewContext();
    ctx.lastMousePos = QPoint(-1, -1);
    ctx.lastMouseMovePos = QPoint(-1, -1);
    ctx.validModelviewMatrix = false;
    ctx.validProjectionMatrix = false;
    ctx.cameraToBBCenterDist = 0.0;
    m_shouldBeRefreshed = false;
    ctx.mouseMoved = false;
    ctx.mouseButtonPressed = false;
    ctx.widgetClicked = false;

    ctx.bbHalfDiag = 0.0;
    ctx.interactionFlags = TRANSFORM_CAMERA();
    ctx.pickingMode = NO_PICKING;
    ctx.pickingModeLocked = false;
    ctx.lastClickTime_ticks = 0;

    ctx.sunLightEnabled = true;
    ctx.customLightEnabled = false;
    ctx.clickableItemsVisible = false;
    m_alwaysUseFBO = false;
    m_updateFBO = true;
    m_winDBRoot = nullptr;
    m_globalDBRoot = nullptr;  // external DB
    m_removeFlag = false;
    m_removeAllFlag = false;
    m_font = QFont();
    ctx.pivotVisibility = PIVOT_SHOW_ON_MOVE;
    ctx.pivotSymbolShown = false;
    ctx.allowRectangularEntityPicking = false;
    m_rectPickingPoly = nullptr;
    m_overridenDisplayParametersEnabled = false;
    ctx.displayOverlayEntities = true;
    ctx.bubbleViewModeEnabled = false;
    ctx.bubbleViewFov_deg = 90.0f;
    ctx.touchInProgress = false;
    ctx.touchBaseDist = 0.0;
    m_scheduledFullRedrawTime = 0;
    ctx.exclusiveFullscreen = false;
    ctx.showDebugTraces = false;
    ctx.pickRadius = DefaultPickRadius;
    m_autoRefresh = false;
    m_hotZone = nullptr;
    ctx.showCursorCoordinates = false;
    ctx.autoPickPivotAtCenter = false;
    ctx.ignoreMouseReleaseEvent = false;
    ctx.rotationAxisLocked = false;
    ctx.lockedRotationAxis = CCVector3d(0, 0, 1);

    // GL window own DB
    m_winDBRoot = new ccHObject(QString("DB.3DView_%1").arg(m_uniqueID));

    // matrices
    ctx.viewportParams.viewMat.toIdentity();
    ctx.viewportParams.setCameraCenter(CCVector3d(
            0.0, 0.0,
            1.0));  // don't position the camera on the pivot by default!
    ctx.viewMatd.toIdentity();
    ctx.projMatd.toIdentity();

    // default modes
    SetPickingMode(DEFAULT_PICKING);
    SetInteractionMode(TRANSFORM_CAMERA());

    // auto-load previous perspective settings
    {
        QSettings settings;
        settings.beginGroup(c_ps_groupName);

        // load parameters
        bool perspectiveView =
                settings.value(c_ps_perspectiveView, false).toBool();
        // DGM: we force object-centered view by default now, as the
        // viewer-based perspective is too dependent on what is displayed (so
        // restoring this parameter at next startup is rarely a good idea)
        bool objectCenteredView =
                /*settings.value(c_ps_objectMode, true ).toBool()*/ true;
        int pivotVisibility =
                settings.value(c_ps_pivotVisibility, PIVOT_HIDE).toInt();

        settings.endGroup();

        // pivot visibility
        switch (pivotVisibility) {
            case PIVOT_HIDE:
                SetPivotVisibility(PIVOT_HIDE);
                break;
            case PIVOT_SHOW_ON_MOVE:
                SetPivotVisibility(PIVOT_SHOW_ON_MOVE);
                break;
            case PIVOT_ALWAYS_SHOW:
                SetPivotVisibility(PIVOT_ALWAYS_SHOW);
                break;
        }

        // apply saved parameters
        SetPerspectiveState(perspectiveView, objectCenteredView);
    }

    m_deferredPickingTimer.setSingleShot(true);
    m_deferredPickingTimer.setInterval(100);

    // signal/slot connections
    connect(this, &ecvDisplayTools::itemPickedFast, this,
            &ecvDisplayTools::onItemPickedFast, Qt::DirectConnection);
    connect(GetVisualizer3D(),
            &ecvGenericVisualizer3D::interactorPointPickedEvent, this,
            &ecvDisplayTools::onPointPicking);

    connect(&m_scheduleTimer, &QTimer::timeout, this,
            &ecvDisplayTools::checkScheduledRedraw);
    connect(&m_deferredPickingTimer, &QTimer::timeout, this,
            &ecvDisplayTools::doPicking);

    // Phase M3: VtkDisplayTools is a pure engine service, NOT a view.
    // The first ecvGLView is created by MainWindow and registered as the
    // actual view. We no longer register VtkDisplayTools with ecvViewManager.
}

ecvDisplayTools::~ecvDisplayTools() {
    cancelScheduledRedraw();
    if (m_winDBRoot) {
        delete m_winDBRoot;
        m_winDBRoot = nullptr;
    }
    if (m_rectPickingPoly) {
        delete m_rectPickingPoly;
        m_rectPickingPoly = nullptr;
    }
    m_hotZone = nullptr;
}

void ecvDisplayTools::checkScheduledRedraw() {
    if (m_scheduledFullRedrawTime &&
        m_timer.elapsed() > m_scheduledFullRedrawTime) {
        // clean the outdated messages
        {
            std::list<MessageToDisplay>::iterator it =
                    m_messagesToDisplay.begin();
            qint64 currentTime_sec = m_timer.elapsed() / 1000;
            // CVLog::PrintDebug(QString("[paintGL] Current time:
            // %1.").arg(currentTime_sec));

            while (it != m_messagesToDisplay.end()) {
                // no more valid? we delete the message
                if (it->messageValidity_sec < currentTime_sec) {
                    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                    it->message));
                    it = m_messagesToDisplay.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

void ecvDisplayTools::cancelScheduledRedraw() {
    m_scheduledFullRedrawTime = 0;
    m_scheduleTimer.stop();
}

void ecvDisplayTools::scheduleFullRedraw(unsigned maxDelay_ms) {
    m_scheduledFullRedrawTime = m_timer.elapsed() + maxDelay_ms;

    if (!m_scheduleTimer.isActive()) {
        m_scheduleTimer.start(500);
    }
}

void ecvDisplayTools::onPointPicking(const CCVector3& p,
                                     int index,
                                     const std::string& id) {
    ecvViewContext* ctx = nullptr;
    if (m_pickingTargetView && m_pickingTargetView->viewContext()) {
        ctx = m_pickingTargetView->viewContext();
    }
    if (!ctx) {
        ctx = &ecvViewManager::instance().resolveViewContext();
    }
    ctx->lastPickedPoint = p;
    ctx->lastPointIndex = index;
    ctx->lastPickedId = id.c_str();

#ifdef QT_DEBUG
    CVLog::Print(QString("current selected index is %1").arg(index));
    CVLog::Print(
            QString("current selected entity id is %1").arg(ctx->lastPickedId));
    CVLog::Print(QString("current selected point coord is [%1, %2, %3]")
                         .arg(p.x)
                         .arg(p.y)
                         .arg(p.z));
#endif  // !QDEBUG

    // Data stored. Processing is deferred to the per-view timer callback
    // (ecvGLView::m_deferredPickingTimer) which sets m_pickingTargetView
    // correctly before calling doPicking(). Do NOT call doPicking() here
    // to avoid double-processing.
}

void ecvDisplayTools::doPicking() {
    ecvViewContext* ctx =
            m_pickingTargetView ? m_pickingTargetView->viewContext() : nullptr;
    ecvViewContext& fallback = ecvViewManager::instance().resolveViewContext();

    CVLog::PrintDebug(QString("[doPicking] targetView=%1 ctx=%2")
                         .arg(m_pickingTargetView != nullptr)
                         .arg(ctx != nullptr));

    bool widgetClicked = ctx ? ctx->widgetClicked : fallback.widgetClicked;
    if (widgetClicked) {
        CVLog::PrintDebug(
                "[doPicking] ABORT: widgetClicked=true");
        if (ctx)
            ctx->widgetClicked = false;
        else
            fallback.widgetClicked = false;
        m_pickingTargetView = nullptr;
        return;
    }

    const QPoint& mousePos = ctx ? ctx->lastMousePos : fallback.lastMousePos;
    int x = mousePos.x();
    int y = mousePos.y();

    if (x < 0 || y < 0) {
        CVLog::PrintDebug(QString("[doPicking] ABORT: invalid pos (%1,%2)")
                             .arg(x).arg(y));
        m_pickingTargetView = nullptr;
        return;
    }

    PICKING_MODE pickMode = ctx ? ctx->pickingMode : fallback.pickingMode;
    INTERACTION_FLAGS iFlags =
            ctx ? ctx->interactionFlags : fallback.interactionFlags;
    int pickRad = ctx ? ctx->pickRadius : fallback.pickRadius;

    CVLog::PrintDebug(QString("[doPicking] pos=(%1,%2) pickMode=%3 iFlags=0x%4 "
                         "pickRad=%5")
                         .arg(x).arg(y).arg(static_cast<int>(pickMode))
                         .arg(static_cast<int>(iFlags), 0, 16).arg(pickRad));

    if ((pickMode != NO_PICKING) || (iFlags & INTERACT_2D_ITEMS)) {
        ecvGenericGLDisplay* itemView = m_pickingTargetView;
        if (!itemView) itemView = ecvViewManager::instance().getEffectiveView();
        if (!itemView) itemView = primaryDT();
        std::list<ccInteractor*>& activeItems = itemView->activeItemsRef();

        if (iFlags & INTERACT_2D_ITEMS) {
            UpdateActiveItemsList(x, y, false);
            if (!activeItems.empty() && activeItems.size() == 1) {
                ccInteractor* pickedObj = activeItems.front();
                cc2DLabel* label = dynamic_cast<cc2DLabel*>(pickedObj);
                if (label && !label->isSelected()) {
                    emit primaryDT()->entitySelectionChanged(label);
                    QApplication::processEvents();
                }
            }
        } else {
            assert(activeItems.empty());
        }

        if (activeItems.empty() && pickMode != NO_PICKING) {
            PICKING_MODE effectiveMode = pickMode;

            if (effectiveMode == ENTITY_PICKING &&
                (QApplication::keyboardModifiers() & Qt::AltModifier)) {
                effectiveMode = LABEL_PICKING;
            } else if (effectiveMode == ENTITY_PICKING &&
                       (QApplication::keyboardModifiers() &
                        Qt::ControlModifier)) {
                effectiveMode = POINT_OR_TRIANGLE_PICKING;
            }

            CVLog::PrintDebug(QString("[doPicking] StartPicking mode=%1 at (%2,%3)")
                                 .arg(static_cast<int>(effectiveMode))
                                 .arg(x).arg(y));
            PickingParameters params(effectiveMode, x, y, pickRad, pickRad);
            ecvViewContext& pickCtx = ctx ? *ctx : fallback;
            StartPicking(pickCtx, params);
        } else {
            CVLog::PrintDebug(QString("[doPicking] NO StartPicking: "
                                 "activeItems=%1 pickMode=%2")
                                 .arg(activeItems.size())
                                 .arg(static_cast<int>(pickMode)));
        }
    } else {
        CVLog::PrintDebug(QString("[doPicking] SKIP: pickMode=%1 iFlags=0x%2 "
                             "(NO_PICKING and no INTERACT_2D_ITEMS)")
                             .arg(static_cast<int>(pickMode))
                             .arg(static_cast<int>(iFlags), 0, 16));
    }

    m_pickingTargetView = nullptr;
}

void ecvDisplayTools::onWheelEvent(float wheelDelta_deg) {
    // Phase M2.3: resolveViewContext for multi-view correctness (not a stale
    // primary-only context).
    const auto& ctx = ecvViewManager::instance().resolveViewContext();

    // in perspective mode, wheel event corresponds to 'walking'
    if (ctx.viewportParams.perspectiveView) {
        // to zoom in and out we simply change the fov in bubble-view mode!
        if (ctx.bubbleViewModeEnabled) {
            SetBubbleViewFov(ctx.bubbleViewFov_deg -
                             wheelDelta_deg / 3.6f);  // 1 turn = 100 degrees
        } else {
            const double& deg2PixConversion = GetDisplayParameters().zoomSpeed;
            double delta = deg2PixConversion *
                           static_cast<double>(wheelDelta_deg) *
                           ctx.viewportParams.pixelSize;

            if (ctx.cameraToBBCenterDist > ctx.bbHalfDiag) {
                delta *= 1.0 + std::log(ctx.cameraToBBCenterDist /
                                        ctx.bbHalfDiag);
            }
        }
    } else {
        static const float c_defaultDeg2Zoom = 20.0f;
        float zoomFactor = std::pow(1.1f, wheelDelta_deg / c_defaultDeg2Zoom);
        UpdateZoom(zoomFactor);
    }

    UpdateDisplayParameters();
}

bool ecvDisplayTools::ProcessClickableItems(ecvViewContext& ctx, int x, int y) {
    ecvGenericGLDisplay* owner = displayOwningContext(&ctx);
    if (!owner) return false;
    std::vector<ClickableItem>& clickableItems = owner->clickableItemsRef();
    if (clickableItems.empty()) {
        return false;
    }

    const int retinaScale = GetDevicePixelRatio();
    x *= retinaScale;
    y *= retinaScale;

    ClickableItem::Role clickedItem = ClickableItem::NO_ROLE;
    for (std::vector<ClickableItem>::const_iterator it = clickableItems.begin();
         it != clickableItems.end(); ++it) {
        if (it->area.contains(x, y)) {
            clickedItem = it->role;
            break;
        }
    }

    switch (clickedItem) {
        case ClickableItem::NO_ROLE: {
        } break;

        case ClickableItem::INCREASE_POINT_SIZE: {
            SetPointSize(ctx.viewportParams.defaultPointSize + 1.0f);
        }
            return true;

        case ClickableItem::DECREASE_POINT_SIZE: {
            SetPointSize(ctx.viewportParams.defaultPointSize - 1.0f);
        }
            return true;

        case ClickableItem::INCREASE_LINE_WIDTH: {
            SetLineWidth(ctx.viewportParams.defaultLineWidth + 1.0f);
        }
            return true;

        case ClickableItem::DECREASE_LINE_WIDTH: {
            SetLineWidth(ctx.viewportParams.defaultLineWidth - 1.0f);
        }
            return true;

        case ClickableItem::LEAVE_BUBBLE_VIEW_MODE: {
            SetBubbleViewMode(false);
            RedrawDisplay();
        }
            return true;

        case ClickableItem::LEAVE_FULLSCREEN_MODE: {
            if (primaryDT()->m_win) {
                emit primaryDT()->exclusiveFullScreenToggled(false);
            }
        }
            return true;

        default: {
            assert(false);
        } break;
    }

    return false;
}

bool ecvDisplayTools::ProcessClickableItems(int x, int y) {
    ecvGenericGLDisplay* v = ecvViewManager::instance().getEffectiveView();
    if (!v || !v->viewContext()) return false;
    return ProcessClickableItems(*v->viewContext(), x, y);
}

bool ecvDisplayTools::ProcessClickableItems(int x,
                                             int y,
                                             float* localPointSize,
                                             float* localLineWidth) {
    Q_UNUSED(localPointSize);
    Q_UNUSED(localLineWidth);
    return ProcessClickableItems(x, y);
}

void ecvDisplayTools::SetPointSize(float size, bool silent, int viewport) {
    float newSize =
            std::max(std::min(size, MAX_POINT_SIZE_F), MIN_POINT_SIZE_F);
    if (!silent) {
        CVLog::Print(QString("New point size: %1").arg(newSize));
    }

    // Write-through: update both ecvViewManager-resolved primary context and
    // active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        ecvViewportParameters vp = av->getViewportParameters();
        vp.defaultPointSize = newSize;
        av->setViewportParameters(vp);
    }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (ctx.viewportParams.defaultPointSize != newSize) {
        ctx.viewportParams.defaultPointSize = newSize;

        if (!silent) {
            ecvDisplayTools::DisplayNewMessage(
                    QString("New default point size: %1").arg(newSize),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 2,
                    SCREEN_SIZE_MESSAGE);
        }
    }
}

void ecvDisplayTools::SetPointSizeRecursive(int size) {
    auto& vm = ecvViewManager::instance();
    if (auto* root = vm.globalDBRoot()) {
        root->setPointSizeRecursive(size);
    }

    if (auto* dt = vm.displayTools();
        dt && dt->m_winDBRoot) {
        dt->m_winDBRoot->setPointSizeRecursive(size);
    }
}

void ecvDisplayTools::SetLineWidth(float width, bool silent, int viewport) {
    float newWidth =
            std::max(std::min(width, MAX_LINE_WIDTH_F), MIN_LINE_WIDTH_F);
    if (!silent) {
        CVLog::Print(QString("New line with: %1").arg(newWidth));
    }

    // Write-through: update both ecvViewManager-resolved primary context and
    // active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        ecvViewportParameters vp = av->getViewportParameters();
        vp.defaultLineWidth = newWidth;
        av->setViewportParameters(vp);
    }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (ctx.viewportParams.defaultLineWidth != newWidth) {
        ctx.viewportParams.defaultLineWidth = newWidth;
        if (!silent) {
            ecvDisplayTools::DisplayNewMessage(
                    QString("New default line width: %1").arg(newWidth),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 2,
                    SCREEN_SIZE_MESSAGE);
        }
    }
}

void ecvDisplayTools::SetLineWithRecursive(PointCoordinateType with) {
    auto& vm = ecvViewManager::instance();
    if (auto* root = vm.globalDBRoot()) {
        root->setLineWidthRecursive(with);
    }

    if (auto* dt = vm.displayTools();
        dt && dt->m_winDBRoot) {
        dt->m_winDBRoot->setLineWidthRecursive(with);
    }
}

void ecvDisplayTools::SetViewportDefaultPointSize(ecvViewContext& ctx,
                                                  float size) {
    ctx.viewportParams.defaultPointSize = size;
}

void ecvDisplayTools::SetViewportDefaultPointSize(float size) {
    SetViewportDefaultPointSize(ecvViewManager::instance().resolveViewContext(), size);
}

void ecvDisplayTools::SetViewportDefaultLineWidth(ecvViewContext& ctx,
                                                  float width) {
    ctx.viewportParams.defaultLineWidth = width;
}

void ecvDisplayTools::SetViewportDefaultLineWidth(float width) {
    SetViewportDefaultLineWidth(ecvViewManager::instance().resolveViewContext(), width);
}

void ecvDisplayTools::StartPicking(PickingParameters& params) {
    StartPicking(ecvViewManager::instance().resolveViewContext(), params);
}

void ecvDisplayTools::StartPicking(ecvViewContext& ctx,
                                   PickingParameters& params) {
    const int retinaScale = GetDevicePixelRatio();
    params.centerX *= retinaScale;
    params.centerY *= retinaScale;

    bool hasDB = primaryDT()->m_globalDBRoot || primaryDT()->m_winDBRoot;
    if (!hasDB && primaryDT()->m_pickingTargetView) {
        hasDB = primaryDT()->m_pickingTargetView->getSceneDB() ||
                primaryDT()->m_pickingTargetView->getOwnDB();
    }
    CVLog::PrintDebug(QString("[StartPicking] hasDB=%1 globalDB=%2 winDB=%3 "
                         "targetView=%4 sceneDB=%5 retinaScale=%6 "
                         "center=(%7,%8) viewport=(%9x%10)")
                         .arg(hasDB)
                         .arg(primaryDT()->m_globalDBRoot != nullptr)
                         .arg(primaryDT()->m_winDBRoot != nullptr)
                         .arg(primaryDT()->m_pickingTargetView != nullptr)
                         .arg(primaryDT()->m_pickingTargetView
                                      ? (primaryDT()->m_pickingTargetView
                                                 ->getSceneDB() != nullptr)
                                      : false)
                         .arg(retinaScale)
                         .arg(params.centerX).arg(params.centerY)
                         .arg(ctx.glViewport.width())
                         .arg(ctx.glViewport.height()));
    if (!hasDB) {
        CVLog::PrintDebug("[StartPicking] ABORT: no DB");
        ProcessPickingResult(params, nullptr, -1);
        return;
    }

    if (params.mode == POINT_OR_TRIANGLE_PICKING ||
        params.mode == POINT_PICKING || params.mode == TRIANGLE_PICKING ||
        params.mode == LABEL_PICKING) {
        StartCPUBasedPointPicking(ctx, params);
    } else {
        StartOpenGLPicking(ctx, params);
    }
}

void ecvDisplayTools::ProcessPickingResult(
        const PickingParameters& params,
        ccHObject* pickedEntity,
        int pickedItemIndex,
        const CCVector3* nearestPoint /*=0*/,
        const std::unordered_set<int>* selectedIDs /*=0*/) {
    // standard "entity" picking
    if (params.mode == ENTITY_PICKING) {
        emit primaryDT()->entitySelectionChanged(pickedEntity);
    }
    // rectangular "entity" picking
    else if (params.mode == ENTITY_RECT_PICKING) {
        if (selectedIDs)
            emit primaryDT()->entitiesSelectionChanged(*selectedIDs);
        else
            assert(false);
    }
    // 3D point or triangle picking
    else if (params.mode == POINT_PICKING || params.mode == TRIANGLE_PICKING ||
             params.mode == POINT_OR_TRIANGLE_PICKING) {
        assert(pickedEntity == nullptr || pickedItemIndex >= 0);
        assert(nearestPoint);

        CVLog::PrintDebug(QString("[ProcessPickingResult] POINT/TRI pick: "
                             "entity=%1 idx=%2 pos=(%3,%4) P=(%5,%6,%7)")
                             .arg(pickedEntity
                                          ? pickedEntity->getName()
                                          : "null")
                             .arg(pickedItemIndex)
                             .arg(params.centerX).arg(params.centerY)
                             .arg(nearestPoint->x).arg(nearestPoint->y)
                             .arg(nearestPoint->z));

        emit primaryDT()->itemPicked(pickedEntity,
                                 static_cast<unsigned>(pickedItemIndex),
                                 params.centerX, params.centerY, *nearestPoint);
    }
    // fast picking (labels, interactors, etc.)
    else if (params.mode == FAST_PICKING) {
        emit primaryDT()->itemPickedFast(pickedEntity, pickedItemIndex,
                                     params.centerX, params.centerY);
    } else if (params.mode == LABEL_PICKING) {
        bool hasDB = primaryDT()->m_globalDBRoot != nullptr;
        if (!hasDB && primaryDT()->m_pickingTargetView)
            hasDB = primaryDT()->m_pickingTargetView->getSceneDB() != nullptr;
        if (!hasDB) {
            auto* ev = ecvViewManager::instance().getEffectiveView();
            if (ev) hasDB = ev->getSceneDB() != nullptr;
        }
        if (hasDB && pickedEntity && pickedItemIndex >= 0) {
            // qint64 stopTime = m_timer.nsecsElapsed();
            // CVLog::Print(QString("[Picking] entity ID %1 - item #%2 (time: %3
            // ms)").arg(selectedID).arg(pickedItemIndex).arg((stopTime-startTime)
            // / 1.0e6));

            // auto spawn the right label
            cc2DLabel* label = nullptr;
            if (pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                label = new cc2DLabel();
                label->addPickedPoint(
                        ccHObjectCaster::ToGenericPointCloud(pickedEntity),
                        pickedItemIndex);
                pickedEntity->addChild(label);
            } else if (pickedEntity->isKindOf(CV_TYPES::MESH)) {
                label = new cc2DLabel();
                ccGenericMesh* mesh =
                        ccHObjectCaster::ToGenericMesh(pickedEntity);
                ccGenericPointCloud* cloud = mesh->getAssociatedCloud();
                assert(cloud);
                cloudViewer::VerticesIndexes* vertexIndexes =
                        mesh->getTriangleVertIndexes(pickedItemIndex);
                label->addPickedPoint(cloud, vertexIndexes->i1);
                label->addPickedPoint(cloud, vertexIndexes->i2);
                label->addPickedPoint(cloud, vertexIndexes->i3);
                cloud->addChild(label);
                if (!cloud->isEnabled()) {
                    cloud->setVisible(false);
                    cloud->setEnabled(true);
                }
            }

            if (label) {
                label->setVisible(true);
                ecvGenericGLDisplay* labelDisplay =
                        primaryDT()->m_pickingTargetView;
                if (!labelDisplay)
                    labelDisplay = pickedEntity->getDisplay();
                if (!labelDisplay)
                    labelDisplay = ecvViewManager::instance().getEffectiveView();
                if (!labelDisplay) labelDisplay = primaryDT();
                label->setDisplay(labelDisplay);
                label->setPosition(
                        static_cast<float>(params.centerX + 20) /
                                GlWidth(),
                        static_cast<float>(params.centerY + 20) /
                                GlHeight());
                emit primaryDT()->newLabel(static_cast<ccHObject*>(label));
                QApplication::processEvents();
                Redraw2DLabel();
            }
        }
    }
}

void ecvDisplayTools::SetZNearCoef(ecvViewContext& ctx, double coef) {
    if (coef <= 0.0 || coef >= 1.0) {
        CVLog::Warning("[ecvDisplayTools::setZNearCoef] Invalid coef. value!");
        return;
    }

    if (ctx.viewportParams.zNearCoef != coef) {
        ctx.viewportParams.zNearCoef = coef;
        if (ctx.viewportParams.perspectiveView) {
            UpdateProjectionMatrix(ctx);

            SetCameraClip(ctx.viewportParams.zNear,
                          ctx.viewportParams.zFar);

            Deprecate3DLayer();

            DisplayNewMessage(
                    QString("Near clipping = %1% of max depth (= %2)")
                            .arg(ctx.viewportParams.zNearCoef * 100.0,
                                 0, 'f', 1)
                            .arg(ctx.viewportParams.zNear),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE,
                    false, 2, SCREEN_SIZE_MESSAGE);
        }

        emit primaryDT()->zNearCoefChanged(coef);
        emit primaryDT()->cameraParamChanged();
    }
}

void ecvDisplayTools::SetZNearCoef(double coef) {
    SetZNearCoef(ecvViewManager::instance().resolveViewContext(), coef);
}

// DGM: WARNING: OpenGL picking with the picking buffer is depreacted.
// We need to get rid of this code or change it to color-based selection...
void ecvDisplayTools::StartOpenGLPicking(ecvViewContext& ctx,
                                         const PickingParameters& params) {
    if (!params.pickInLocalDB && !params.pickInSceneDB) {
        assert(false);
        return;
    }

    unsigned short flags = CC_DRAW_FOREGROUND;

    switch (params.mode) {
        case FAST_PICKING:
            flags |= CC_FAST_ENTITY_PICKING;
        case ENTITY_PICKING:
        case ENTITY_RECT_PICKING:
            flags |= CC_ENTITY_PICKING;
            break;
        default:
            assert(false);
            ProcessPickingResult(params, nullptr, -1);
            return;
    }

    assert(!primaryDT()->m_captureMode.enabled);

    std::unordered_set<int> selectedIDs;
    int pickedItemIndex = -1;
    int selectedID = -1;
    ccHObject* pickedEntity = nullptr;

    CCVector3 P(0, 0, 0);
    CCVector3* pickedPoint = nullptr;

    if (ctx.lastPointIndex >= 0) {
        pickedEntity = GetPickedEntity(ctx, params);
        if (pickedEntity) {
            selectedID = pickedEntity->getUniqueID();
            selectedIDs.insert(selectedID);
            pickedItemIndex = ctx.lastPointIndex;
        }
    }

    if (pickedEntity && pickedItemIndex >= 0 &&
        pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* tempEntity =
                ccHObjectCaster::ToGenericPointCloud(pickedEntity);
        int pNum = static_cast<int>(tempEntity->size());
        if (pickedItemIndex >= pNum) {
            P = ctx.lastPickedPoint;
            CVLog::Warning(
                    QString("[ecvDisplayTools::StartOpenGLPicking] Picking "
                            "Error, %1 is more than picked entity size %2")
                            .arg(pickedItemIndex)
                            .arg(tempEntity->size()));
            pickedItemIndex = pNum - 1;
        } else {
            P = *(static_cast<ccGenericPointCloud*>(pickedEntity)
                          ->getPoint(pickedItemIndex));
            CCVector3 temp = P - ctx.lastPickedPoint;
            if (temp.norm() > 1) {
                ProcessPickingResult(params, nullptr, -1);
#ifdef QT_DEBUG
                CVLog::Warning(
                        QString("[ecvDisplayTools::StartOpenGLPicking] droped "
                                "selected point coord is [%1, %2, %3]")
                                .arg(P.x)
                                .arg(P.y)
                                .arg(P.z));
#endif  // QT_DEBUG
                return;
            }
        }

        pickedPoint = &P;
    }

    ProcessPickingResult(params, pickedEntity, pickedItemIndex, pickedPoint,
                         &selectedIDs);
}

void ecvDisplayTools::StartOpenGLPicking(const PickingParameters& params) {
    StartOpenGLPicking(ecvViewManager::instance().resolveViewContext(), params);
}

void ecvDisplayTools::StartCPUBasedPointPicking(
        ecvViewContext& ctx, const PickingParameters& params) {
    ccHObject* nearestEntity = nullptr;
    int nearestElementIndex = -1;
    double nearestElementSquareDist = -1.0;
    CCVector3 nearestPoint(0, 0, 0);
    static const unsigned MIN_POINTS_FOR_OCTREE_COMPUTATION = 128;

    static ecvGui::ParamStruct::ComputeOctreeForPicking
            autoComputeOctreeThisSession = ecvGui::ParamStruct::ASK_USER;
    bool autoComputeOctree = false;
    bool firstCloudWithoutOctree = true;

    ccGLCameraParameters camera;
    GetGLCameraParameters(camera, ctx);

    CCVector2d clickedPos(params.centerX,
                          ctx.glViewport.height() - 1 - params.centerY);

    CVLog::PrintDebug(QString("[CPUPick] USE_VTK_PICK=%1 clickedPos=(%2,%3) "
                         "viewport=(%4x%5) validMV=%6 validProj=%7 "
                         "perspective=%8 fov=%9 pixelSize=%10")
                         .arg(USE_VTK_PICK)
                         .arg(clickedPos.x).arg(clickedPos.y)
                         .arg(ctx.glViewport.width())
                         .arg(ctx.glViewport.height())
                         .arg(ctx.validModelviewMatrix)
                         .arg(ctx.validProjectionMatrix)
                         .arg(camera.perspective)
                         .arg(camera.fov_deg)
                         .arg(camera.pixelSize));
    {
        const double* mv = camera.modelViewMat.data();
        const double* pj = camera.projectionMat.data();
        CVLog::PrintDebug(QString("[CPUPick] MV row0=[%1,%2,%3,%4]")
                             .arg(mv[0],0,'g',6).arg(mv[4],0,'g',6)
                             .arg(mv[8],0,'g',6).arg(mv[12],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] MV row1=[%1,%2,%3,%4]")
                             .arg(mv[1],0,'g',6).arg(mv[5],0,'g',6)
                             .arg(mv[9],0,'g',6).arg(mv[13],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] MV row2=[%1,%2,%3,%4]")
                             .arg(mv[2],0,'g',6).arg(mv[6],0,'g',6)
                             .arg(mv[10],0,'g',6).arg(mv[14],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] MV row3=[%1,%2,%3,%4]")
                             .arg(mv[3],0,'g',6).arg(mv[7],0,'g',6)
                             .arg(mv[11],0,'g',6).arg(mv[15],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] PJ row0=[%1,%2,%3,%4]")
                             .arg(pj[0],0,'g',6).arg(pj[4],0,'g',6)
                             .arg(pj[8],0,'g',6).arg(pj[12],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] PJ row1=[%1,%2,%3,%4]")
                             .arg(pj[1],0,'g',6).arg(pj[5],0,'g',6)
                             .arg(pj[9],0,'g',6).arg(pj[13],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] PJ row2=[%1,%2,%3,%4]")
                             .arg(pj[2],0,'g',6).arg(pj[6],0,'g',6)
                             .arg(pj[10],0,'g',6).arg(pj[14],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] PJ row3=[%1,%2,%3,%4]")
                             .arg(pj[3],0,'g',6).arg(pj[7],0,'g',6)
                             .arg(pj[11],0,'g',6).arg(pj[15],0,'g',6));
        CVLog::PrintDebug(QString("[CPUPick] VP=[%1,%2,%3,%4]")
                             .arg(camera.viewport[0]).arg(camera.viewport[1])
                             .arg(camera.viewport[2]).arg(camera.viewport[3]));
    }

    if (ecvDisplayTools::USE_VTK_PICK) {
        int pickedIndex = -1;
        ccHObject* pickedEntity = nullptr;
        if (ctx.lastPointIndex >= 0) {
            pickedIndex = ctx.lastPointIndex;
            pickedEntity = GetPickedEntity(ctx, params);
        }

        if (pickedEntity && pickedIndex >= 0) {
            ccHObject* ent = nullptr;
            bool isMesh = false;
            ccGenericMesh* pickedMesh = nullptr;
            if (pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
                ent = pickedEntity;
            } else if (pickedEntity->isKindOf(CV_TYPES::MESH) &&
                       !pickedEntity->isA(CV_TYPES::MESH_GROUP)) {
                pickedMesh = ccHObjectCaster::ToGenericMesh(pickedEntity);
                ent = pickedMesh->getAssociatedCloud();
                isMesh = true;
            } else {
                return;
            }

            ccGenericPointCloud* tempEntity =
                    ccHObjectCaster::ToGenericPointCloud(ent);
            int pNum = static_cast<int>(tempEntity->size());

            if (isMesh && pickedMesh) {
                unsigned triCount = pickedMesh->size();
                int triIdx = pickedIndex / 3;
                if (triIdx >= 0 && static_cast<unsigned>(triIdx) < triCount) {
                    nearestElementIndex = triIdx;
                    nearestPoint = ctx.lastPickedPoint;
                } else {
                    nearestElementIndex = -1;
                }
            } else {
                nearestElementIndex = pickedIndex;
                if (pickedIndex >= pNum) {
                    nearestPoint = ctx.lastPickedPoint;
                    CVLog::Warning(QString("[ecvDisplayTools::"
                                           "StartCPUBasedPointPicking] "
                                           "Picking Error, %1 is more than "
                                           "picked entity size %2")
                                           .arg(pickedIndex)
                                           .arg(tempEntity->size()));
                    nearestElementIndex = pNum - 1;
                } else {
                    nearestPoint = *(tempEntity->getPoint(pickedIndex));
                    CCVector3 temp = nearestPoint - ctx.lastPickedPoint;
                    if (temp.norm() > 1) {
                        ProcessPickingResult(params, nullptr, -1);
                        return;
                    }
                }
            }

            nearestEntity = pickedEntity;
        }
    } else {
        try {
            ccHObject::Container toProcess;
            ecvGenericGLDisplay* pickView = primaryDT()->m_pickingTargetView;
            ccHObject* sceneDB = pickView ? pickView->getSceneDB() : nullptr;
            CVLog::PrintDebug(QString("[CPUPick] CPU path: pickView=%1 sceneDB=%2 "
                                 "children=%3")
                                 .arg(pickView != nullptr)
                                 .arg(sceneDB != nullptr)
                                 .arg(sceneDB ? sceneDB->getChildrenNumber()
                                              : 0));
            if (sceneDB) {
                ccHObject::Container clouds;
                sceneDB->filterChildren(clouds, true,
                                        CV_TYPES::POINT_CLOUD, true);
                if (!clouds.empty()) {
                    auto* cloud = static_cast<ccGenericPointCloud*>(clouds[0]);
                    if (cloud->size() > 0) {
                        const CCVector3* pt = cloud->getPoint(0);
                        CCVector3d projected;
                        bool ok = camera.projectSafe(*pt, projected);
                        CVLog::PrintDebug(
                                QString("[CPUPick] TestProj: pt=(%1,%2,%3)"
                                        " -> screen=(%4,%5,%6) ok=%7"
                                        " cloudEnabled=%8 cloudSize=%9")
                                        .arg(pt->x,0,'g',6)
                                        .arg(pt->y,0,'g',6)
                                        .arg(pt->z,0,'g',6)
                                        .arg(projected.x,0,'g',6)
                                        .arg(projected.y,0,'g',6)
                                        .arg(projected.z,0,'g',6)
                                        .arg(ok)
                                        .arg(cloud->isEnabled())
                                        .arg(cloud->size()));
                    }
                } else {
                    CVLog::PrintDebug("[CPUPick] No point clouds found in sceneDB!");
                }
            }
            ccHObject* ownDB = pickView ? pickView->getOwnDB() : nullptr;
            if (!sceneDB) sceneDB = primaryDT()->m_globalDBRoot;
            if (!ownDB) ownDB = primaryDT()->m_winDBRoot;
            if (sceneDB) toProcess.push_back(sceneDB);
            if (ownDB) toProcess.push_back(ownDB);

            while (!toProcess.empty()) {
                // get next item
                ccHObject* ent = toProcess.back();
                toProcess.pop_back();

                if (!ent->isEnabled()) continue;

                bool ignoreSubmeshes = false;

                // we look for point cloud displayed in this window
                if (ent->isKindOf(CV_TYPES::POINT_CLOUD)) {
                    ccGenericPointCloud* cloud =
                            static_cast<ccGenericPointCloud*>(ent);

                    if (firstCloudWithoutOctree && !cloud->getOctree() &&
                        cloud->size() >
                                MIN_POINTS_FOR_OCTREE_COMPUTATION)  // no need
                                                                    // to use
                                                                    // the
                                                                    // octree
                                                                    // for a few
                                                                    // points!
                    {
                        // can we compute an octree for picking?
                        ecvGui::ParamStruct::ComputeOctreeForPicking behavior =
                                GetDisplayParameters().autoComputeOctree;
                        if (behavior == ecvGui::ParamStruct::ASK_USER) {
                            // we use the persistent parameter for this session
                            behavior = autoComputeOctreeThisSession;
                        }

                        switch (behavior) {
                            case ecvGui::ParamStruct::ALWAYS:
                                autoComputeOctree = true;
                                break;

                            case ecvGui::ParamStruct::ASK_USER: {
                                QMessageBox question(
                                        QMessageBox::Question,
                                        "Picking acceleration",
                                        "Automatically compute octree(s) to "
                                        "accelerate the picking "
                                        "process?\n(this behavior can be "
                                        "changed later in the Display "
                                        "Settings)",
                                        QMessageBox::NoButton,
                                        GetCurrentScreen());

                                QPushButton* yes = new QPushButton("Yes");
                                question.addButton(yes,
                                                   QMessageBox::AcceptRole);
                                QPushButton* no = new QPushButton("No");
                                question.addButton(no, QMessageBox::RejectRole);
                                QPushButton* always = new QPushButton("Always");
                                question.addButton(always,
                                                   QMessageBox::AcceptRole);
                                QPushButton* never = new QPushButton("Never");
                                question.addButton(never,
                                                   QMessageBox::RejectRole);

                                question.exec();
                                QAbstractButton* clickedButton =
                                        question.clickedButton();
                                if (clickedButton == yes) {
                                    autoComputeOctree = true;
                                    autoComputeOctreeThisSession =
                                            ecvGui::ParamStruct::ALWAYS;
                                } else if (clickedButton == no) {
                                    CVLog::Warning(
                                            "now only support octree picking, "
                                            "please don't select no!");
                                    continue;
                                    autoComputeOctree = false;
                                    autoComputeOctreeThisSession =
                                            ecvGui::ParamStruct::NEVER;
                                } else if (clickedButton == always ||
                                           clickedButton == never) {
                                    autoComputeOctree =
                                            (clickedButton == always);
                                    // update the global application parameters
                                    ecvGui::ParamStruct params =
                                            ecvGui::Parameters();
                                    params.autoComputeOctree =
                                            autoComputeOctree
                                                    ? ecvGui::ParamStruct::
                                                              ALWAYS
                                                    : ecvGui::ParamStruct::
                                                              NEVER;
                                    ecvGui::Set(params);
                                    params.toPersistentSettings();
                                }
                            } break;

                            case ecvGui::ParamStruct::NEVER:
                                autoComputeOctree = false;
                                break;
                        }

                        firstCloudWithoutOctree = false;
                    }

                    int nearestPointIndex = -1;
                    double nearestSquareDist = 0.0;
                    if (cloud->pointPicking(
                                clickedPos, camera, nearestPointIndex,
                                nearestSquareDist, params.pickWidth,
                                params.pickHeight,
                                autoComputeOctree &&
                                        cloud->size() >
                                                MIN_POINTS_FOR_OCTREE_COMPUTATION)) {
                        if (nearestElementIndex < 0 ||
                            (nearestPointIndex >= 0 &&
                             nearestSquareDist < nearestElementSquareDist)) {
                            nearestElementSquareDist = nearestSquareDist;
                            nearestElementIndex = nearestPointIndex;
                            nearestPoint =
                                    *(cloud->getPoint(nearestPointIndex));
                            nearestEntity = cloud;
                        }
                    }
                } else if (ent->isKindOf(CV_TYPES::MESH) &&
                           !ent->isA(CV_TYPES::MESH_GROUP)  // we don't need to
                                                            // process mesh
                                                            // groups as their
                                                            // children will be
                                                            // processed later
                           &&
                           !ent->isA(CV_TYPES::COORDINATESYSTEM)  // we ignore
                                                                  // coordinate
                                                                  // system
                                                                  // entities
                ) {
                    ignoreSubmeshes = true;

                    ccGenericMesh* mesh = static_cast<ccGenericMesh*>(ent);
                    if (mesh->isShownAsWire()) {
                        // skip meshes that are displayed in wireframe mode
                        continue;
                    }

                    int nearestTriIndex = -1;
                    double nearestSquareDist = 0.0;
                    CCVector3d P;
                    if (mesh->trianglePicking(clickedPos, camera,
                                              nearestTriIndex,
                                              nearestSquareDist, P)) {
                        if (nearestElementIndex < 0 ||
                            (nearestTriIndex >= 0 &&
                             nearestSquareDist < nearestElementSquareDist)) {
                            nearestElementSquareDist = nearestSquareDist;
                            nearestPoint = CCVector3::fromArray(P.u);
                            nearestEntity = mesh;

                            nearestElementIndex = nearestTriIndex;
                        }
                    }
                } else if (params.mode ==
                                   PICKING_MODE::
                                           POINT_OR_TRIANGLE_OR_LABEL_PICKING &&
                           ent->isA(CV_TYPES::LABEL_2D)) {
                    cc2DLabel* label = static_cast<cc2DLabel*>(ent);

                    int nearestPointIndex = -1;
                    double nearestSquareDist = 0.0;

                    if (label->pointPicking(clickedPos, camera,
                                            nearestPointIndex,
                                            nearestSquareDist)) {
                        if (nearestElementIndex < 0 ||
                            (nearestPointIndex >= 0 &&
                             nearestSquareDist < nearestElementSquareDist)) {
                            nearestElementSquareDist = nearestSquareDist;
                            assert(nearestPointIndex <
                                   static_cast<int>(label->size()));
                            nearestElementIndex = nearestPointIndex;
                            nearestPoint =
                                    label->getPickedPoint(nearestPointIndex)
                                            .getPointPosition();
                            nearestEntity = label;
                        }
                    }
                } else if (ent->isKindOf(CV_TYPES::SENSOR)) {
                    // only activated when ctrl and mouse pressed!
                    if (params.mode != POINT_OR_TRIANGLE_PICKING) {
                        continue;
                    }

                    if (ent->isA(CV_TYPES::CAMERA_SENSOR)) {
                        ignoreSubmeshes = true;

                        ccCameraSensor* cameraSensor =
                                static_cast<ccCameraSensor*>(ent);
                        if (!cameraSensor &&
                            cameraSensor->getNearPlane().IsEmpty()) {
                            // skip meshes that are displayed in wireframe mode
                            continue;
                        }

                        QString id = ecvDisplayTools::PickObject(clickedPos.x,
                                                                 clickedPos.y);

                        if (id.toInt() != -1 &&
                            static_cast<int>(cameraSensor->getUniqueID()) ==
                                    id.toInt()) {
                            nearestElementIndex = id.toInt();
                            nearestPoint = CCVector3();
                            nearestEntity = cameraSensor;
                            break;
                        }
                    }
                }

                // add children
                for (unsigned i = 0; i < ent->getChildrenNumber(); ++i) {
                    // we ignore the sub-meshes of the current (mesh) entity
                    // as their content is the same!
                    if (ignoreSubmeshes &&
                        ent->getChild(i)->isKindOf(CV_TYPES::SUB_MESH) &&
                        static_cast<ccSubMesh*>(ent)->getAssociatedMesh() ==
                                ent) {
                        continue;
                    }

                    toProcess.push_back(ent->getChild(i));
                }
            }
        } catch (const std::bad_alloc&) {
            // not enough memory
            CVLog::Warning("[Picking][CPU] Not enough memory!");
        }
    }
    CVLog::PrintDebug(QString("[CPUPick] Result: entity=%1 idx=%2 dist=%3")
                         .arg(nearestEntity ? nearestEntity->getName() : "null")
                         .arg(nearestElementIndex)
                         .arg(nearestElementSquareDist));

    if (!ecvDisplayTools::USE_VTK_PICK) {
        ctx.lastPointIndex = nearestElementIndex;
        ctx.lastPickedPoint = nearestPoint;
        if (nearestEntity) {
            ctx.lastPickedId = nearestEntity->getViewId();
        }
    }

    ProcessPickingResult(params, nearestEntity, nearestElementIndex,
                         &nearestPoint);
}

void ecvDisplayTools::StartCPUBasedPointPicking(
        const PickingParameters& params) {
    StartCPUBasedPointPicking(ecvViewManager::instance().resolveViewContext(), params);
}

ccHObject* ecvDisplayTools::GetPickedEntity(const ecvViewContext& ctx,
                                            const PickingParameters& params) {
    if (ctx.lastPickedId.isEmpty()) return nullptr;

    ccHObject* pickedEntity = nullptr;
    unsigned int selectedID = ctx.lastPickedId.toUInt();

    ccHObject* sceneDB = nullptr;
    ccHObject* localDB = nullptr;

    if (primaryDT()->m_pickingTargetView) {
        sceneDB = primaryDT()->m_pickingTargetView->getSceneDB();
        localDB = primaryDT()->m_pickingTargetView->getOwnDB();
    }
    if (!sceneDB) sceneDB = primaryDT()->m_globalDBRoot;
    if (!localDB) localDB = primaryDT()->m_winDBRoot;

    if (params.pickInSceneDB && sceneDB) {
        pickedEntity = sceneDB->find(selectedID);
    }
    if (!pickedEntity && params.pickInLocalDB && localDB) {
        pickedEntity = localDB->find(selectedID);
    }

    return pickedEntity;
}

ccHObject* ecvDisplayTools::GetPickedEntity(const PickingParameters& params) {
    return GetPickedEntity(ecvViewManager::instance().resolveViewContext(), params);
}

QPointF ecvDisplayTools::ToCenteredGLCoordinates(int x, int y) {
    return QPointF(x - Width() / 2,
                   Height() / 2 - y) /* * GetDevicePixelRatio()*/;
}

CCVector3d ecvDisplayTools::ToVtkCoordinates(int x, int y, int z) {
    CCVector3d p = CCVector3d(x * 1.0, y * 1.0, z * 1.0);
    ToVtkCoordinates(p);
    return p;
}

void ecvDisplayTools::ToVtkCoordinates(CCVector3d& sP) {
    sP.y = Height() - sP.y;  // for vtk coordinates
    sP *= GetDevicePixelRatio();
}

void ecvDisplayTools::ToVtkCoordinates(CCVector2i& sP) {
    sP.y = Height() - sP.y;  // for vtk coordinates
    sP *= GetDevicePixelRatio();
}

void ecvDisplayTools::SetPivotVisibility(ecvViewContext& ctx,
                                         PivotVisibility vis) {
    ctx.pivotVisibility = vis;

    if (vis == PivotVisibility::PIVOT_HIDE) {
        SetPivotVisibility(false);
    } else {
        SetPivotVisibility(true);
    }

    UpdateScreen();

    {
        QSettings settings;
        settings.beginGroup(c_ps_groupName);
        settings.setValue(c_ps_pivotVisibility, vis);
        settings.endGroup();
    }
}

void ecvDisplayTools::SetPivotVisibility(PivotVisibility vis) {
    SetPivotVisibility(ecvViewManager::instance().resolveViewContext(), vis);
}

void ecvDisplayTools::ResizeGL(ecvViewContext& ctx, int w, int h) {
    SetGLViewport(ctx, QRect(0, 0, w, h));

    InvalidateVisualization();
    Deprecate3DLayer();

    if (ecvGenericGLDisplay* owner = displayOwningContext(&ctx)) {
        ecvHotZone*& hzRef = owner->hotZonePtrRef();
        if (hzRef) hzRef->topCorner = QPoint(0, 0);
    }

    DisplayNewMessage(QString("New size = %1 * %2 (px)")
                              .arg(ctx.glViewport.width())
                              .arg(ctx.glViewport.height()),
                      LOWER_LEFT_MESSAGE, false, 2, SCREEN_SIZE_MESSAGE);
}

void ecvDisplayTools::ResizeGL(int w, int h) {
    ResizeGL(ecvViewManager::instance().resolveViewContext(), w, h);
}

void ecvDisplayTools::MoveCamera(ecvViewContext& ctx,
                                 float dx, float dy, float dz) {
    if (dx != 0.0f || dy != 0.0f) {
        emit primaryDT()->cameraDisplaced(dx, dy);
    }

    CCVector3d V(dx, dy, dz);
    if (!ctx.viewportParams.objectCenteredView) {
        ctx.viewportParams.viewMat.transposed().applyRotation(V);
    }

    SetCameraPos(ctx, ctx.viewportParams.getCameraCenter() + V);
}

void ecvDisplayTools::MoveCamera(float dx, float dy, float dz) {
    MoveCamera(ecvViewManager::instance().resolveViewContext(), dx, dy, dz);
}

void ecvDisplayTools::UpdateActiveItemsList(
        int x, int y, bool extendToSelectedLabels /*=false*/) {
    ecvGenericGLDisplay* itemView = primaryDT()->m_pickingTargetView;
    if (!itemView) itemView = ecvViewManager::instance().getEffectiveView();
    if (!itemView) itemView = primaryDT();
    std::list<ccInteractor*>& items = itemView->activeItemsRef();
    items.clear();

    {
        ccHObject::Container labels;
        FilterByEntityType(labels, CV_TYPES::LABEL_2D);
        CVLog::PrintDebug("[Label] UpdateActiveItemsList: mouse(%d,%d) labels=%zu",
                     x, y, labels.size());
        for (auto* obj : labels) {
            if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
                !obj->isVisible())
                continue;
            cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
            if (!l) continue;
            QRect roi = l->getLabelROI();
            CVLog::PrintDebug(
                    "[Label]   '%s' ROI=(%d,%d %dx%d) valid=%d contains=%d "
                    "dispIn2D=%d dispLegend=%d",
                    qPrintable(l->getName()), roi.x(), roi.y(), roi.width(),
                    roi.height(), roi.isValid(), roi.contains(x, y),
                    l->isDisplayedIn2D(), l->isPointLegendDisplayed());
            if (roi.isValid() && roi.contains(x, y)) {
                items.push_back(l);
                CVLog::PrintDebug("[Label]   >>> HIT! Added to active items");
                break;
            }
        }
    }

    if (items.empty()) {
        PickingParameters params(FAST_PICKING, x, y, 2, 2);
        StartPicking(params);
    }

    if (items.size() == 1) {
        ccInteractor* pickedObj = items.front();
        cc2DLabel* label = dynamic_cast<cc2DLabel*>(pickedObj);
        if (label) {
            if (!label->isSelected() || !extendToSelectedLabels) {
            } else {
                ccHObject::Container labels;
                FilterByEntityType(labels, CV_TYPES::LABEL_2D);

                for (auto& lbl : labels) {
                    if (lbl->isA(CV_TYPES::LABEL_2D) && lbl->isEnabled() &&
                        lbl->isVisible()) {
                        cc2DLabel* l = static_cast<cc2DLabel*>(lbl);
                        if (l != pickedObj && l->isSelected()) {
                            items.push_back(l);
                        }
                    }
                }
            }
        }
    }
}

void ecvDisplayTools::onItemPickedFast(ccHObject* pickedEntity,
                                       int pickedItemIndex,
                                       int x,
                                       int y) {
    ecvGenericGLDisplay* target = m_pickingTargetView;
    if (!target) target = ecvViewManager::instance().getEffectiveView();
    if (!target) target = primaryDT();
    std::list<ccInteractor*>& items = target->activeItemsRef();
    ecvViewContext* tctx = target->viewContext();

    if (pickedEntity) {
        if (pickedEntity->isA(CV_TYPES::LABEL_2D)) {
            cc2DLabel* label = static_cast<cc2DLabel*>(pickedEntity);
            items.push_back(label);
        } else if (pickedEntity->isA(CV_TYPES::CLIPPING_BOX)) {
            ccClipBox* cbox = static_cast<ccClipBox*>(pickedEntity);
            cbox->setActiveComponent(pickedItemIndex);
            ccGLMatrixd& viewMat =
                    tctx ? tctx->viewportParams.viewMat
                         : ecvViewManager::instance()
                                   .resolveViewContext()
                                   .viewportParams.viewMat;
            cbox->setClickedPoint(x, y, Width(), Height(), viewMat);

            items.push_back(cbox);
        }
    }

    emit fastPickingFinished();
}

void ecvDisplayTools::UpdateScreen() {
    if (QWidget* w = GetCurrentScreen()) {
        w->update();
    }
    UpdateScene();

    if (ecvViewManager::instance().viewCount() > 1) {
        ecvViewManager::instance().refreshAll();
    }
}

void ecvDisplayTools::UpdateScreenSize() { ResizeGL(Width(), Height()); }

CCVector3d ecvDisplayTools::ConvertMousePositionToOrientation(
        const ecvViewContext& ctx, int x, int y) {
    double xc = static_cast<double>(Width() / 2);
    double yc = static_cast<double>(Height() / 2);

    CCVector3d Q2D;
    if (ctx.viewportParams.objectCenteredView) {
        ccGLCameraParameters camera;
        GetGLCameraParameters(camera);

        if (!camera.project(ctx.viewportParams.getPivotPoint(), Q2D)) {
            return CCVector3d(0, 0, 1);
        }

        Q2D.x = std::min(Q2D.x, 3.0 * Width() / 4.0);
        Q2D.x = std::max(Q2D.x, Width() / 4.0);

        Q2D.y = std::min(Q2D.y, 3.0 * Height() / 4.0);
        Q2D.y = std::max(Q2D.y, Height() / 4.0);
    } else {
        Q2D.x = xc;
        Q2D.y = yc;
    }

    y = Height() - 1 - y;

    CCVector3d v(x - Q2D.x, y - Q2D.y, 0.0);

    v.x = std::max(std::min(v.x / xc, 1.0), -1.0);
    v.y = std::max(std::min(v.y / yc, 1.0), -1.0);

    double d2 = v.x * v.x + v.y * v.y;

    if (d2 > 1) {
        double d = std::sqrt(d2);
        v.x /= d;
        v.y /= d;
    } else {
        v.z = std::sqrt(1.0 - d2);
    }

    return v;
}

CCVector3d ecvDisplayTools::ConvertMousePositionToOrientation(int x, int y) {
    return ConvertMousePositionToOrientation(ecvViewManager::instance().resolveViewContext(), x, y);
}

void ecvDisplayTools::RotateBaseViewMat(ecvViewContext& ctx,
                                        const ccGLMatrixd& rotMat) {
    ecvViewportParameters viewParams = GetViewportParameters(ctx);
    viewParams.viewMat = rotMat * viewParams.viewMat;

    CCVector3d camC = viewParams.viewMat.getTranslationAsVec3D();
    viewParams.setCameraCenter(camC);

    CCVector3d upDir = viewParams.viewMat.getColumnAsVec3D(1);
    upDir.normalize();
    viewParams.up = upDir;

    CCVector3d viewDir = viewParams.viewMat.getColumnAsVec3D(2);
    viewParams.focal = camC - viewDir;
    viewParams.setPivotPoint(viewParams.focal, true);

    ecvDisplayTools::SetViewportParameters(viewParams);

    emit primaryDT()->baseViewMatChanged(ctx.viewportParams.viewMat);
}

void ecvDisplayTools::RotateBaseViewMat(const ccGLMatrixd& rotMat) {
    RotateBaseViewMat(ecvViewManager::instance().resolveViewContext(), rotMat);
}

ccGLMatrixd ecvDisplayTools::GenerateViewMat(CC_VIEW_ORIENTATION orientation) {
    CCVector3d eye(0, 0, 0);
    CCVector3d center(0, 0, 0);
    CCVector3d top(0, 0, 0);

    // we look at (0,0,0) by default
    switch (orientation) {
        case CC_TOP_VIEW:
            eye.z = 1.0;
            top.y = 1.0;
            break;
        case CC_BOTTOM_VIEW:
            eye.z = -1.0;
            top.y = 1.0;
            break;
        case CC_FRONT_VIEW:
            eye.y = -1.0;
            top.z = 1.0;
            break;
        case CC_BACK_VIEW:
            eye.y = 1.0;
            top.z = 1.0;
            break;
        case CC_LEFT_VIEW:
            eye.x = -1.0;
            top.z = 1.0;
            break;
        case CC_RIGHT_VIEW:
            eye.x = 1.0;
            top.z = 1.0;
            break;
        case CC_ISO_VIEW_1:
            eye.x = -1.0;
            eye.y = -1.0;
            eye.z = 1.0;
            top.x = 1.0;
            top.y = 1.0;
            top.z = 1.0;
            break;
        case CC_ISO_VIEW_2:
            eye.x = 1.0;
            eye.y = 1.0;
            eye.z = 1.0;
            top.x = -1.0;
            top.y = -1.0;
            top.z = 1.0;
            break;
    }

    return ccGLMatrixd::FromViewDirAndUpDir(center - eye, top);
}

void ecvDisplayTools::SetView(CC_VIEW_ORIENTATION orientation, ccBBox* bbox) {
    switch (orientation) {
        case CC_TOP_VIEW:
            SetCameraPosition(0, 0, 1, 0, 0, 0, 0, 1, 0);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_BOTTOM_VIEW:
            SetCameraPosition(0, 0, -1, 0, 0, 0, 0, 1, 0);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_FRONT_VIEW:
            SetCameraPosition(0, 1, 0, 0, 0, 0, 0, 0, 1);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_BACK_VIEW:
            SetCameraPosition(0, -1, 0, 0, 0, 0, 0, 0, 1);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_LEFT_VIEW:
            SetCameraPosition(-1, 0, 0, 0, 0, 0, 0, 0, 1);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_RIGHT_VIEW:
            SetCameraPosition(1.0, 0.0, 0, 0, 0, 0, 0, 0, 1);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_ISO_VIEW_1:
            SetCameraPosition(-1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        case CC_ISO_VIEW_2:
            SetCameraPosition(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0);
            UpdateConstellationCenterAndZoom(bbox, false);
            break;
        default:
            break;
    }
}

void ecvDisplayTools::SetView(ecvViewContext& ctx,
                              CC_VIEW_ORIENTATION orientation,
                              bool forceRedraw /*=false*/) {
    bool wasViewerBased = !ctx.viewportParams.objectCenteredView;
    if (wasViewerBased) {
        SetPerspectiveState(ctx.viewportParams.perspectiveView, true);
    }
    ctx.viewportParams.viewMat = GenerateViewMat(orientation);
    if (wasViewerBased) {
        SetPerspectiveState(ctx.viewportParams.perspectiveView, false);
    }

    emit primaryDT()->baseViewMatChanged(ctx.viewportParams.viewMat);
    emit primaryDT()->cameraParamChanged();
    // may be useless

    // Get current camera parameters to preserve zoom/distance
    double currentPos[3], currentFocal[3];
    GetCameraPos(currentPos);
    GetCameraFocal(currentFocal);

    CCVector3d currentCameraPos(currentPos[0], currentPos[1], currentPos[2]);
    CCVector3d currentFocalPoint(currentFocal[0], currentFocal[1],
                                 currentFocal[2]);

    // Calculate current distance from camera to focal point (this represents
    // the zoom level)
    CCVector3d currentViewDir = currentFocalPoint - currentCameraPos;
    double currentDistance = currentViewDir.norm();

    // If distance is too small or invalid, use a default distance
    if (currentDistance < 1e-6) {
        currentDistance = 1.0;
    }

    // Generate view direction and up vector for the new orientation
    CCVector3d eye(0, 0, 0);
    CCVector3d center(0, 0, 0);
    CCVector3d top(0, 0, 0);

    switch (orientation) {
        case CC_TOP_VIEW:
            eye.z = 1.0;
            top.y = 1.0;
            break;
        case CC_BOTTOM_VIEW:
            eye.z = -1.0;
            top.y = 1.0;
            break;
        case CC_FRONT_VIEW:
            eye.y = -1.0;
            top.z = 1.0;
            break;
        case CC_BACK_VIEW:
            eye.y = 1.0;
            top.z = 1.0;
            break;
        case CC_LEFT_VIEW:
            eye.x = -1.0;
            top.z = 1.0;
            break;
        case CC_RIGHT_VIEW:
            eye.x = 1.0;
            top.z = 1.0;
            break;
        case CC_ISO_VIEW_1:
            eye.x = -1.0;
            eye.y = -1.0;
            eye.z = 1.0;
            top.x = 1.0;
            top.y = 1.0;
            top.z = 1.0;
            break;
        case CC_ISO_VIEW_2:
            eye.x = 1.0;
            eye.y = 1.0;
            eye.z = 1.0;
            top.x = -1.0;
            top.y = -1.0;
            top.z = 1.0;
            break;
        default:
            return;
    }

    // Normalize the view direction
    CCVector3d newViewDir = center - eye;
    newViewDir.normalize();

    // Normalize the up vector
    top.normalize();

    // Calculate new camera position: keep the focal point, move camera along
    // new view direction The new camera position = focal point - (view
    // direction * distance) This preserves the zoom level (distance) while
    // changing only the orientation
    CCVector3d newCameraPos = currentFocalPoint - newViewDir * currentDistance;

    // Set camera with new orientation but preserved distance and focal point
    SetCameraPosition(newCameraPos.x, newCameraPos.y, newCameraPos.z,
                      currentFocalPoint.x, currentFocalPoint.y,
                      currentFocalPoint.z, top.x, top.y, top.z);

    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();
    if (forceRedraw) {
        RedrawDisplay();
    } else {
        UpdateScreen();
    }
}

void ecvDisplayTools::SetView(CC_VIEW_ORIENTATION orientation,
                              bool forceRedraw /*=false*/) {
    SetView(ecvViewManager::instance().resolveViewContext(), orientation, forceRedraw);
}

inline float RoundScale(float equivalentWidth) {
    // we compute the scale granularity (to avoid width values with a lot of
    // decimals)
    int k = static_cast<int>(
            std::floor(std::log(equivalentWidth) / std::log(10.0f)));
    float granularity = std::pow(10.0f, static_cast<float>(k)) / 2.0f;
    // we choose the value closest to equivalentWidth with the right granularity
    return std::floor(std::max(equivalentWidth / granularity, 1.0f)) *
           granularity;
}

float ecvDisplayTools::ComputePerspectiveZoom(const ecvViewContext& ctx) {
    float currentFov_deg = GetFov();
    if (currentFov_deg < FLT_EPSILON) return 1.0f;

    double zoomEquivalentDist =
            (ctx.viewportParams.getCameraCenter() -
             ctx.viewportParams.getPivotPoint())
                    .norm();
    if (cloudViewer::LessThanEpsilon(zoomEquivalentDist)) return 1.0f;

    float screenSize = std::min(ctx.glViewport.width(),
                                ctx.glViewport.height()) *
                       ctx.viewportParams.pixelSize;
    return screenSize /
           static_cast<float>(
                   zoomEquivalentDist *
                   std::tan(cloudViewer::DegreesToRadians(currentFov_deg)));
}

float ecvDisplayTools::ComputePerspectiveZoom() {
    return ComputePerspectiveZoom(ecvViewManager::instance().resolveViewContext());
}

ccGLMatrixd& ecvDisplayTools::GetModelViewMatrix(ecvViewContext& ctx) {
    if (!ctx.validModelviewMatrix) UpdateModelViewMatrix(ctx);
    return ctx.viewMatd;
}

ccGLMatrixd& ecvDisplayTools::GetModelViewMatrix() {
    return GetModelViewMatrix(ecvViewManager::instance().resolveViewContext());
}

ccGLMatrixd& ecvDisplayTools::GetProjectionMatrix(ecvViewContext& ctx) {
    if (!ctx.validProjectionMatrix) UpdateProjectionMatrix();
    return ctx.projMatd;
}

ccGLMatrixd& ecvDisplayTools::GetProjectionMatrix() {
    return GetProjectionMatrix(ecvViewManager::instance().resolveViewContext());
}

ccGLMatrixd ecvDisplayTools::ComputeProjectionMatrix(
        const ecvViewContext& ctx,
        bool withGLfeatures,
        ProjectionMetrics* metrics /*=nullptr*/,
        double* eyeOffset /*=nullptr*/) {
    double bbHalfDiag = 1.0;
    CCVector3d bbCenter(0, 0, 0);

    if (primaryDT()->m_globalDBRoot || primaryDT()->m_winDBRoot) {
        ccBBox box;
        GetVisibleObjectsBB(box);
        if (box.isValid()) {
            bbCenter = CCVector3d::fromArray(box.getCenter().u);
            bbHalfDiag = box.getDiagNormd() / 2;
        }
    }

    CCVector3d cameraCenterToBBCenter =
            ctx.viewportParams.getCameraCenter() - bbCenter;
    double cameraToBBCenterDist = cameraCenterToBBCenter.normd();

    if (metrics) {
        metrics->bbHalfDiag = bbHalfDiag;
        metrics->cameraToBBCenterDist = cameraToBBCenterDist;
    }

    CCVector3d rotationCenter = ctx.viewportParams.getRotationCenter();

    double rotationCenterToFarthestObjectDist = 0.0;
    {
        rotationCenterToFarthestObjectDist =
                (bbCenter - rotationCenter).norm() + bbHalfDiag;

        if (ctx.pivotSymbolShown &&
            ctx.pivotVisibility != PIVOT_HIDE &&
            withGLfeatures &&
            ctx.viewportParams.objectCenteredView) {
            double pivotActualRadius =
                    CC_DISPLAYED_PIVOT_RADIUS_PERCENT *
                    std::min(ctx.glViewport.width(),
                             ctx.glViewport.height()) /
                    2;
            double pivotSymbolScale =
                    pivotActualRadius * ComputeActualPixelSize(ctx);
            rotationCenterToFarthestObjectDist = std::max(
                    rotationCenterToFarthestObjectDist, pivotSymbolScale);
        }

        if (withGLfeatures && ctx.customLightEnabled) {
            double distToCustomLight =
                    (rotationCenter -
                     CCVector3d::fromArray(ctx.customLightPos))
                            .norm();
            rotationCenterToFarthestObjectDist = std::max(
                    rotationCenterToFarthestObjectDist, distToCustomLight);
        }

        rotationCenterToFarthestObjectDist *= 1.01;
    }

    double cameraCenterToRotationCentertDist = 0;
    if (ctx.viewportParams.objectCenteredView) {
        cameraCenterToRotationCentertDist =
                ctx.viewportParams.getFocalDistance();
    }

    double zNear = cameraCenterToRotationCentertDist -
                   rotationCenterToFarthestObjectDist;
    double zFar = cameraCenterToRotationCentertDist +
                  rotationCenterToFarthestObjectDist;

    double ar = static_cast<double>(ctx.glViewport.height()) /
                ctx.glViewport.width();

    ccGLMatrixd projMatrix;
    if (ctx.viewportParams.perspectiveView) {
        zNear = bbHalfDiag * ctx.viewportParams.zNearCoef;
        zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

        double xMax = zNear *
                      ctx.viewportParams.computeDistanceToHalfWidthRatio();
        double yMax = xMax * ar;

        double frustumAsymmetry = 0.0;
        projMatrix = ecvGenericDisplayTools::Frustum(-xMax - frustumAsymmetry,
                                                     xMax - frustumAsymmetry,
                                                     -yMax, yMax, zNear, zFar);
    } else {
        zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

        double xMax = std::abs(cameraCenterToRotationCentertDist) *
                      ctx.viewportParams.computeDistanceToHalfWidthRatio();
        double yMax = xMax * ar;

        projMatrix = ecvGenericDisplayTools::Ortho(-xMax, xMax, -yMax, yMax,
                                                   zNear, zFar);
    }
    return projMatrix;
}

ccGLMatrixd ecvDisplayTools::ComputeProjectionMatrix(
        bool withGLfeatures,
        ProjectionMetrics* metrics /*=nullptr*/,
        double* eyeOffset /*=nullptr*/) {
    return ComputeProjectionMatrix(ecvViewManager::instance().resolveViewContext(), withGLfeatures,
                                   metrics, eyeOffset);
}

void ecvDisplayTools::UpdateProjectionMatrix(ecvViewContext& ctx) {
    ProjectionMetrics metrics;

    ctx.projMatd =
            ComputeProjectionMatrix(ctx, true, &metrics,
                                    nullptr);

    ctx.viewportParams.zNear = metrics.zNear;
    ctx.viewportParams.zFar = metrics.zFar;
    ctx.cameraToBBCenterDist = metrics.cameraToBBCenterDist;
    ctx.bbHalfDiag = metrics.bbHalfDiag;

    ctx.validProjectionMatrix = true;
}

void ecvDisplayTools::UpdateProjectionMatrix() {
    UpdateProjectionMatrix(ecvViewManager::instance().resolveViewContext());
}

CCVector3d ecvDisplayTools::GetRealCameraCenter(const ecvViewContext& ctx) {
    if (ctx.viewportParams.perspectiveView) {
        return ctx.viewportParams.getCameraCenter();
    }

    ccBBox box;
    GetVisibleObjectsBB(box);

    return CCVector3d(ctx.viewportParams.getCameraCenter().x,
                      ctx.viewportParams.getCameraCenter().y,
                      box.isValid() ? box.getCenter().z : 0.0);
}

CCVector3d ecvDisplayTools::GetRealCameraCenter() {
    return GetRealCameraCenter(ecvViewManager::instance().resolveViewContext());
}

ccGLMatrixd ecvDisplayTools::ComputeModelViewMatrix(const ecvViewContext& ctx) {
    ccGLMatrixd viewMatd = ctx.viewportParams.computeViewMatrix();
    ccGLMatrixd scaleMatd = ctx.viewportParams.computeScaleMatrix(ctx.glViewport);
    return scaleMatd * viewMatd;
}

ccGLMatrixd ecvDisplayTools::ComputeModelViewMatrix() {
    return ComputeModelViewMatrix(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::UpdateModelViewMatrix(ecvViewContext& ctx) {
    ctx.viewMatd = ComputeModelViewMatrix(ctx);
    ctx.validModelviewMatrix = true;
}

void ecvDisplayTools::UpdateModelViewMatrix() {
    UpdateModelViewMatrix(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::SetBaseViewMat(ecvViewContext& ctx, ccGLMatrixd& mat) {
    ctx.viewportParams.viewMat = mat;
    InvalidateVisualization();
    emit primaryDT()->baseViewMatChanged(ctx.viewportParams.viewMat);
    emit primaryDT()->cameraParamChanged();
}

void ecvDisplayTools::SetBaseViewMat(ccGLMatrixd& mat) {
    SetBaseViewMat(ecvViewManager::instance().resolveViewContext(), mat);
}

void ecvDisplayTools::SetPerspectiveState(ecvViewContext& ctx,
                                          bool state,
                                          bool objectCenteredView) {
    bool perspectiveWasEnabled = ctx.viewportParams.perspectiveView;
    bool viewWasObjectCentered = ctx.viewportParams.objectCenteredView;

    ctx.viewportParams.perspectiveView = state;
    ctx.viewportParams.objectCenteredView = objectCenteredView;

    CCVector3d PC = ctx.viewportParams.getCameraCenter() -
                    ctx.viewportParams.getPivotPoint();

    if (ctx.viewportParams.perspectiveView) {
        if (!perspectiveWasEnabled) {
            double currentFov_deg = static_cast<double>(
                    ctx.bubbleViewModeEnabled ? ctx.bubbleViewFov_deg
                                             : ctx.viewportParams.fov_deg);
            assert(cloudViewer::GreaterThanEpsilon(currentFov_deg));
            double screenSize =
                    std::min(ctx.glViewport.width(),
                             ctx.glViewport.height()) *
                    ctx.viewportParams.pixelSize;
            if (screenSize > 0.0) {
                PC.z = screenSize /
                       (ctx.viewportParams.zoom *
                        std::tan(
                                cloudViewer::DegreesToRadians(currentFov_deg)));
            }
        }

        DisplayNewMessage(objectCenteredView ? "Centered perspective ON"
                                             : "Viewer-based perspective ON",
                          LOWER_LEFT_MESSAGE, false, 2,
                          PERSPECTIVE_STATE_MESSAGE);
    } else {
        ctx.viewportParams.objectCenteredView = true;

        if (perspectiveWasEnabled) {
            float newZoom = ComputePerspectiveZoom();
            SetZoom(newZoom);
        }

        DisplayNewMessage("Perspective OFF", LOWER_LEFT_MESSAGE, false, 2,
                          PERSPECTIVE_STATE_MESSAGE);
    }

    if (viewWasObjectCentered &&
        !ctx.viewportParams.objectCenteredView) {
        ctx.viewportParams.viewMat.transposed().apply(PC);
    } else if (!viewWasObjectCentered &&
               ctx.viewportParams.objectCenteredView) {
        ctx.viewportParams.viewMat.apply(PC);
    }

    SetCameraPos(ctx, ctx.viewportParams.getPivotPoint() + PC);

    emit primaryDT()->perspectiveStateChanged();
    emit primaryDT()->cameraParamChanged();

    {
        QSettings settings;
        settings.beginGroup(c_ps_groupName);
        settings.setValue(c_ps_perspectiveView,
                          ctx.viewportParams.perspectiveView);
        settings.setValue(c_ps_objectMode,
                          ctx.viewportParams.objectCenteredView);
        settings.endGroup();
    }

    ctx.bubbleViewModeEnabled = false;

    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();
}

void ecvDisplayTools::SetPerspectiveState(bool state, bool objectCenteredView) {
    SetPerspectiveState(ecvViewManager::instance().resolveViewContext(), state, objectCenteredView);
}

bool ecvDisplayTools::ObjectPerspectiveEnabled(const ecvViewContext& ctx) {
    return ctx.viewportParams.perspectiveView &&
           ctx.viewportParams.objectCenteredView;
}

bool ecvDisplayTools::ObjectPerspectiveEnabled() {
    return ObjectPerspectiveEnabled(ecvViewManager::instance().resolveViewContext());
}

bool ecvDisplayTools::ViewerPerspectiveEnabled(const ecvViewContext& ctx) {
    return ctx.viewportParams.perspectiveView &&
           !ctx.viewportParams.objectCenteredView;
}

bool ecvDisplayTools::ViewerPerspectiveEnabled() {
    return ViewerPerspectiveEnabled(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::UpdateConstellationCenterAndZoom(const ccBBox* aBox,
                                                       bool redraw) {
    // Delegate to active secondary view if one exists.
    auto* av = activeSecondaryView();
    if (av) {
        av->updateConstellationCenterAndZoom(aBox);
        if (redraw) {
            av->redraw();
        }
        return;
    }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (ctx.bubbleViewModeEnabled) {
        CVLog::Warning(
                "[updateConstellationCenterAndZoom] Not when bubble-view is "
                "enabled!");
        return;
    }

    SetZoom(1.0f);

    ccBBox zoomedBox;

    if (aBox) {
        zoomedBox = (*aBox);
    } else {
        GetVisibleObjectsBB(zoomedBox, ecvViewManager::instance().displayTools());
    }

    if (!zoomedBox.isValid()) {
        return;
    }

    if (redraw) {
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
        RedrawDisplay();
    }

    // we get the bounding-box diagonal length
    double bbDiag = static_cast<double>(zoomedBox.getDiagNorm());

    if (cloudViewer::LessThanEpsilon(bbDiag)) {
        CVLog::Warning("[ecvDisplayTools] Entity/DB has a null bounding-box!");
        return;
    }

    // Add margin to bounding box to ensure objects are fully visible
    // and not clipped at the edges (default 10% margin)
    const double margin = 1.1;  // 10% margin on all sides
    if (margin > 1.0) {
        CCVector3d centerVec = CCVector3d::fromArray(zoomedBox.getCenter().u);
        Eigen::Vector3d center(centerVec.x, centerVec.y, centerVec.z);
        zoomedBox.Scale(margin, center);
    }

    ResetCamera(&zoomedBox);
    UpdateScreen();

    // we compute the pixel size (in world coordinates)
    {
        int minScreenSize =
                std::min(ctx.glViewport.width(), ctx.glViewport.height());
        SetPixelSize(minScreenSize > 0
                             ? static_cast<float>(bbDiag / minScreenSize)
                             : 1.0f);
    }

    // we set the pivot point on the box center
    CCVector3d P = CCVector3d::fromArray(zoomedBox.getCenter().u);
    SetPivotPoint(P);

    // CRITICAL: Update 2D labels after zoom operation to ensure they align with
    // their 3D anchor points. This fixes the issue where labels become detached
    // after zoom operations (zoom on selected, zoom to box, zoom to global).
    if (auto* dt = ecvViewManager::instance().displayTools()) {
        dt->Update2DLabel(true);
    }
}

void ecvDisplayTools::SetRedrawRecursive(bool redraw /* = false*/) {
    if (!ecvViewManager::instance().displayTools()) return;
    if (auto* scene = GetSceneDB()) scene->setRedrawFlagRecursive(redraw);
    if (auto* own = GetOwnDB()) own->setRedrawFlagRecursive(redraw);
}

void ecvDisplayTools::UpdateNamePoseRecursive() {
    if (!ecvViewManager::instance().displayTools()) return;
    if (auto* scene = GetSceneDB()) scene->updateNameIn3DRecursive();
    if (auto* own = GetOwnDB()) own->updateNameIn3DRecursive();

    static QElapsedTimer s_labelTimer;
    static bool s_timerStarted = false;
    if (!s_timerStarted) {
        s_labelTimer.start();
        s_timerStarted = true;
    }
    if (s_labelTimer.elapsed() >= 50) {
        Update2DLabel(true);
        s_labelTimer.restart();
    }
}

void ecvDisplayTools::SetRedrawRecursive(ccHObject* obj,
                                         bool redraw /* = false*/) {
    assert(obj);
    obj->setRedrawFlagRecursive(redraw);
}

void ecvDisplayTools::RedrawObject(ccHObject* obj,
                                   bool only2D /* = false*/,
                                   bool forceRedraw /* = true*/) {
    auto* dt = ecvViewManager::instance().displayTools();
    if (!obj || !dt) return;
    SetRedrawRecursive(false);
    obj->setRedrawFlagRecursive(true);

    // If entity lives in a secondary view, redraw that view directly
    // so property changes (e.g. SF scale visibility) take effect there.
    ecvGenericGLDisplay* disp = obj->getDisplay();
    if (disp && disp != dt) {
        disp->redraw(only2D, forceRedraw);
    }
    RedrawDisplay(only2D, forceRedraw);
}

void ecvDisplayTools::RedrawObjects(std::initializer_list<ccHObject*> objects,
                                    bool only2D /* = false*/,
                                    bool forceRedraw /* = true*/) {
    auto* dt = ecvViewManager::instance().displayTools();
    if (!dt) return;
    SetRedrawRecursive(false);
    for (auto* obj : objects) {
        if (obj) obj->setRedrawFlagRecursive(true);
    }
    // Redraw secondary views that own any of these entities.
    for (auto* obj : objects) {
        if (!obj) continue;
        ecvGenericGLDisplay* disp = obj->getDisplay();
        if (disp && disp != dt) {
            disp->redraw(only2D, forceRedraw);
        }
    }
    RedrawDisplay(only2D, forceRedraw);
}

void ecvDisplayTools::GetVisibleObjectsBB(ccBBox& box,
                                          const ecvGenericGLDisplay* display) {
    auto& vm = ecvViewManager::instance();
    if (auto* root = vm.globalDBRoot()) {
        box = root->getDisplayBB_recursive(false, display);
    }

    if (auto* dt = vm.displayTools();
        dt && dt->m_winDBRoot) {
        ccBBox ownBox =
                dt->m_winDBRoot->getDisplayBB_recursive(false, display);
        if (ownBox.isValid()) {
            box += ownBox;
        }
    }
}

ENTITY_TYPE ecvDisplayTools::ConvertToEntityType(const CV_CLASS_ENUM& type) {
    ENTITY_TYPE entityType = ENTITY_TYPE::ECV_NONE;
    switch (type) {
        case CV_TYPES::HIERARCHY_OBJECT:
            entityType = ENTITY_TYPE::ECV_HIERARCHY_OBJECT;
            break;
        case CV_TYPES::POINT_CLOUD:
            entityType = ENTITY_TYPE::ECV_POINT_CLOUD;
            break;
        case CV_TYPES::POLY_LINE:
        case CV_TYPES::LINESET:
        case (CV_TYPES::CUSTOM_H_OBJECT | CV_TYPES::POLY_LINE):
            entityType = ENTITY_TYPE::ECV_SHAPE;
            break;
        case CV_TYPES::LABEL_2D:
            entityType = ENTITY_TYPE::ECV_2DLABLE;
            break;
        case CV_TYPES::VIEWPORT_2D_LABEL:
            entityType = ENTITY_TYPE::ECV_2DLABLE_VIEWPORT;
            break;
        case CV_TYPES::POINT_OCTREE:
            entityType = ENTITY_TYPE::ECV_OCTREE;
            break;
        case CV_TYPES::POINT_KDTREE:
            entityType = ENTITY_TYPE::ECV_KDTREE;
            break;
        case CV_TYPES::FACET:
        case CV_TYPES::PRIMITIVE:
        case CV_TYPES::MESH:
        case CV_TYPES::SUB_MESH:
        case CV_TYPES::SPHERE:
        case CV_TYPES::CONE:
        case CV_TYPES::PLANE:
        case CV_TYPES::CYLINDER:
        case CV_TYPES::TORUS:
        case CV_TYPES::EXTRU:
        case CV_TYPES::DISH:
        case CV_TYPES::DISC:
        case CV_TYPES::BOX:
        case CV_TYPES::COORDINATESYSTEM:
        case CV_TYPES::QUADRIC:
            entityType = ENTITY_TYPE::ECV_MESH;
            break;
        case CV_TYPES::IMAGE:
            entityType = ENTITY_TYPE::ECV_IMAGE;
            break;
        case CV_TYPES::SENSOR:
        case CV_TYPES::GBL_SENSOR:
        case CV_TYPES::CAMERA_SENSOR:
            entityType = ENTITY_TYPE::ECV_SENSOR;
            break;
        default:
            break;
    }
    return entityType;
}

void ecvDisplayTools::DisplayOverlayEntities(ecvViewContext& ctx, bool state) {
    ctx.displayOverlayEntities = state;
    if (!state) {
        ClearBubbleView();
    }
}

void ecvDisplayTools::DisplayOverlayEntities(bool state) {
    DisplayOverlayEntities(ecvViewManager::instance().resolveViewContext(), state);
}

void ecvDisplayTools::SetSceneDB(ccHObject* root) {
    ecvViewManager::instance().setGlobalDBRoot(root);
    if (auto* dt = ecvViewManager::instance().displayTools()) {
        dt->m_globalDBRoot = root;
    }
    ZoomGlobal();
}

void ecvDisplayTools::AddToOwnDB(ccHObject* obj, bool noDependency /*=true*/) {
    if (!obj) {
        assert(false);
        return;
    }

    auto* av = activeSecondaryView();
    if (av) {
        av->addToOwnDB(obj, noDependency);
        return;
    }

    auto* dt = ecvViewManager::instance().displayTools();
    if (dt && dt->m_winDBRoot) {
        dt->m_winDBRoot->addChild(
                obj, noDependency ? ccHObject::DP_NONE
                                  : ccHObject::DP_PARENT_OF_OTHER);
    } else {
        CVLog::Error("[ecvDisplayTools::addToOwnDB] Window has no DB!");
    }
}

void ecvDisplayTools::RemoveFromOwnDB(ccHObject* obj) {
    auto* av = activeSecondaryView();
    if (av) {
        av->removeFromOwnDB(obj);
        return;
    }
    auto* dt = ecvViewManager::instance().displayTools();
    if (dt && dt->m_winDBRoot) dt->m_winDBRoot->removeChild(obj);
}

void ecvDisplayTools::SetRemoveViewIDs(std::vector<removeInfo>& removeinfos) {
    ecvViewManager::instance().setRemoveViewIds(removeinfos);
}

void ecvDisplayTools::ZoomCamera(double zoomFactor, int viewport) {
    if (auto* dt = ecvViewManager::instance().displayTools()) {
        dt->zoomCamera(zoomFactor, viewport);
    }
    if (!GetViewportParameters().perspectiveView) {
        UpdateZoom(ecvViewManager::instance().resolveViewContext(),
                   static_cast<float>(zoomFactor));
    }
    UpdateDisplayParameters();
}

void ecvDisplayTools::SetInteractionMode(INTERACTION_FLAGS flags) {
    // Write-through: update both the active secondary view AND the
    // ecvViewManager-resolved primary context.
    auto* av = activeSecondaryView();
    if (av) {
        av->setInteractionMode(flags);
        QWidget* w = av->asWidget();
        if (w) {
            w->setMouseTracking(flags & (INTERACT_CLICKABLE_ITEMS |
                                         INTERACT_SIG_MOUSE_MOVED));
        }
    }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    ctx.interactionFlags = flags;

#ifdef CV_GL_WINDOW_USE_QWINDOW
    if (m_parentWidget) {
        m_parentWidget->setMouseTracking(
                flags & (INTERACT_CLICKABLE_ITEMS | INTERACT_SIG_MOUSE_MOVED));
    }
#else
    if (QWidget* w = GetCurrentScreen()) {
        w->setMouseTracking(
                flags & (INTERACT_CLICKABLE_ITEMS | INTERACT_SIG_MOUSE_MOVED));
    }
#endif

    if ((flags & INTERACT_CLICKABLE_ITEMS) == 0) {
        ctx.clickableItemsVisible = false;
    }
}

ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::GetInteractionMode() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getInteractionMode();
    }
    return ecvViewManager::instance().resolveViewContext().interactionFlags;
}

CCVector3d ecvDisplayTools::GetCurrentViewDir(const ecvViewContext& ctx) {
    const double* M = ctx.viewportParams.viewMat.data();
    CCVector3d axis(-M[2], -M[6], -M[10]);
    axis.normalize();
    return axis;
}

CCVector3d ecvDisplayTools::GetCurrentViewDir() {
    auto* av = activeSecondaryView();
    if (av && av->viewContext())
        return GetCurrentViewDir(*av->viewContext());
    return GetCurrentViewDir(ecvViewManager::instance().resolveViewContext());
}

CCVector3d ecvDisplayTools::GetCurrentUpDir(const ecvViewContext& ctx) {
    const double* M = ctx.viewportParams.viewMat.data();
    CCVector3d axis(M[1], M[5], M[9]);
    axis.normalize();
    return axis;
}

CCVector3d ecvDisplayTools::GetCurrentUpDir() {
    auto* av = activeSecondaryView();
    if (av && av->viewContext())
        return GetCurrentUpDir(*av->viewContext());
    return GetCurrentUpDir(ecvViewManager::instance().resolveViewContext());
}

float ecvDisplayTools::GetFov() {
    auto* av = activeSecondaryView();
    if (av) {
        const auto& vp = av->getViewportParameters();
        return vp.fov_deg;
    }
    const auto& ctx = ecvViewManager::instance().resolveViewContext();
    return (ctx.bubbleViewModeEnabled ? ctx.bubbleViewFov_deg
                                      : ctx.viewportParams.fov_deg);
}

void ecvDisplayTools::SetupProjectiveViewport(
        ecvViewContext& ctx,
        const ccGLMatrixd& cameraMatrix,
        float fov_deg /*=0.0f*/,
        float ar /*=1.0f*/,
        bool viewerBasedPerspective /*=true*/,
        bool bubbleViewMode /*=false*/) {
    if (bubbleViewMode) {
        SetBubbleViewMode(true);
    } else {
        SetPerspectiveState(true, !viewerBasedPerspective);
    }

    if (fov_deg > 0.0f) {
        if (ctx.viewportParams.perspectiveView) {
            SetFov(ctx, fov_deg);
        } else {
            SetParallelScale(
                    static_cast<double>(cloudViewer::DegreesToRadians(fov_deg)),
                    0);
        }
    }

    SetAspectRatio(ctx, ar);

    CCVector3d T = cameraMatrix.getTranslationAsVec3D();
    CCVector3d UP = cameraMatrix.getColumnAsVec3D(1);
    cameraMatrix.applyRotation(UP.data());
    SetCameraPos(ctx, T);
    SetCameraPosition(T.data(), UP.data());
    if (viewerBasedPerspective && ctx.autoPickPivotAtCenter) {
        SetPivotPoint(T);
    }

    ccGLMatrixd trans = cameraMatrix;
    trans.clearTranslation();
    trans.invert();
    SetBaseViewMat(ctx, trans);

    ResetCameraClippingRange();
    UpdateScreen();
}

void ecvDisplayTools::SetupProjectiveViewport(
        const ccGLMatrixd& cameraMatrix,
        float fov_deg /*=0.0f*/,
        float ar /*=1.0f*/,
        bool viewerBasedPerspective /*=true*/,
        bool bubbleViewMode /*=false*/) {
    SetupProjectiveViewport(ecvViewManager::instance().resolveViewContext(), cameraMatrix,
                            fov_deg, ar, viewerBasedPerspective,
                            bubbleViewMode);
}

void ecvDisplayTools::SetAspectRatio(ecvViewContext& ctx, float ar) {
    if (ar < 0.0f) {
        CVLog::Warning("[ecvDisplayTools::setAspectRatio] Invalid AR value!");
        return;
    }

    if (ctx.viewportParams.cameraAspectRatio != ar) {
        ctx.viewportParams.cameraAspectRatio = ar;
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
    }
}

void ecvDisplayTools::SetAspectRatio(float ar) {
    SetAspectRatio(ecvViewManager::instance().resolveViewContext(), ar);
}

void ecvDisplayTools::SetFov(ecvViewContext& ctx, float fov_deg) {
    if (cloudViewer::LessThanEpsilon(fov_deg) || fov_deg > 180.0f) {
        CVLog::Warning("[ecvDisplayTools::setFov] Invalid FOV value!");
        return;
    }

    if (ctx.bubbleViewModeEnabled) {
        SetBubbleViewFov(ctx, fov_deg);
    } else if (ctx.viewportParams.fov_deg != fov_deg) {
        ctx.viewportParams.fov_deg = fov_deg;
        {
            SetCameraFovy(fov_deg);
            InvalidateViewport();
            InvalidateVisualization();
            Deprecate3DLayer();

            DisplayNewMessage(
                    QString("F.O.V. = %1 deg.").arg(fov_deg, 0, 'f', 1),
                    LOWER_LEFT_MESSAGE,
                    false, 2, SCREEN_SIZE_MESSAGE);
        }

        emit primaryDT()->fovChanged(ctx.viewportParams.fov_deg);
        emit primaryDT()->cameraParamChanged();
    }
}

void ecvDisplayTools::SetFov(float fov_deg) {
    SetFov(ecvViewManager::instance().resolveViewContext(), fov_deg);
}

void ecvDisplayTools::DisplayNewMessage(const QString& message,
                                        MessagePosition pos,
                                        bool append /*=false*/,
                                        int displayMaxDelay_sec /*=2*/,
                                        MessageType type /*=CUSTOM_MESSAGE*/) {
    auto* view = ecvViewManager::instance().getEffectiveView();
    if (view) {
        view->displayNewMessage(message, pos, append, displayMaxDelay_sec,
                                type);
        return;
    }
    // No views yet — drop message (startup only, before any view is created)
}

void ecvDisplayTools::SetPivotPoint(const CCVector3d& P,
                                    bool autoUpdateCameraPos /*=false*/,
                                    bool verbose /*=false*/) {
    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (autoUpdateCameraPos &&
        (!ctx.viewportParams.perspectiveView ||
         ctx.viewportParams.objectCenteredView)) {
        CCVector3d dP = ctx.viewportParams.getPivotPoint() - P;
        CCVector3d MdP = dP;
        ctx.viewportParams.viewMat.applyRotation(MdP);
        CCVector3d newCameraPos =
                ctx.viewportParams.getCameraCenter() + MdP - dP;
        SetCameraPos(ctx, newCameraPos);
    }

    ctx.viewportParams.setPivotPoint(P, true);
    SetAutoUpateCameraPos(autoUpdateCameraPos);
    SetCenterOfRotation(P);

    emit primaryDT()->pivotPointChanged(ctx.viewportParams.getPivotPoint());
    emit primaryDT()->cameraParamChanged();

    if (verbose) {
        const unsigned& precision =
                GetDisplayParameters().displayedNumPrecision;
        DisplayNewMessage(QString(), LOWER_LEFT_MESSAGE, false);
        DisplayNewMessage(QString("Point (%1 ; %2 ; %3) set as rotation center")
                                  .arg(P.x, 0, 'f', precision)
                                  .arg(P.y, 0, 'f', precision)
                                  .arg(P.z, 0, 'f', precision),
                          LOWER_LEFT_MESSAGE, true);
        RedrawDisplay(true, false);
    }

    ctx.autoPivotCandidate = P;
    InvalidateViewport();
    InvalidateVisualization();
}

void ecvDisplayTools::SetAutoPickPivotAtCenter(bool state) {
    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (ctx.autoPickPivotAtCenter != state) {
        ctx.autoPickPivotAtCenter = state;

        if (state) {
            ctx.autoPivotCandidate = CCVector3d(0, 0, 0);
        }
    }
}

void ecvDisplayTools::LockRotationAxis(ecvViewContext& ctx, bool state,
                                       const CCVector3d& axis) {
    ctx.rotationAxisLocked = state;
    ctx.lockedRotationAxis = axis;
    ctx.lockedRotationAxis.normalize();
}

void ecvDisplayTools::LockRotationAxis(bool state, const CCVector3d& axis) {
    LockRotationAxis(ecvViewManager::instance().resolveViewContext(), state, axis);
}

void ecvDisplayTools::GetContext(CC_DRAW_CONTEXT& CONTEXT) {
    const auto& ctx = ecvViewManager::instance().resolveViewContext();
    CONTEXT.glW = ctx.glViewport.width();
    CONTEXT.glH = ctx.glViewport.height();
    CONTEXT.devicePixelRatio = static_cast<float>(GetDevicePixelRatio());
    CONTEXT.drawingFlags = 0;

    const ecvGui::ParamStruct& guiParams = GetDisplayParameters();

    // decimation options
    CONTEXT.decimateCloudOnMove = guiParams.decimateCloudOnMove;
    CONTEXT.minLODPointCount = guiParams.minLoDCloudSize;
    CONTEXT.minLODTriangleCount = guiParams.minLoDMeshSize;
    CONTEXT.higherLODLevelsAvailable = false;
    CONTEXT.moreLODPointsAvailable = false;
    CONTEXT.currentLODLevel = 0;

    // scalar field color-bar
    CONTEXT.sfColorScaleToDisplay = nullptr;

    // point picking
    CONTEXT.labelMarkerSize = static_cast<float>(guiParams.labelMarkerSize *
                                                 ComputeActualPixelSize());
    CONTEXT.labelMarkerTextShift_pix = 5;  // 5 pixels shift

    // text display
    CONTEXT.dispNumberPrecision = guiParams.displayedNumPrecision;
    // label opacity
    CONTEXT.labelOpacity = guiParams.labelOpacity;

    // default materials
    CONTEXT.defaultMat->setDiffuseFront(guiParams.meshFrontDiff);
    CONTEXT.defaultMat->setDiffuseBack(guiParams.meshBackDiff);
    CONTEXT.defaultMat->setAmbient(ecvColor::bright);
    CONTEXT.defaultMat->setSpecular(guiParams.meshSpecular);
    CONTEXT.defaultMat->setEmission(ecvColor::night);
    CONTEXT.defaultMat->setShininessFront(30);
    CONTEXT.defaultMat->setShininessBack(50);

    // default colors
    CONTEXT.pointsDefaultCol = guiParams.pointsDefaultCol;
    CONTEXT.textDefaultCol = guiParams.textDefaultCol;
    CONTEXT.labelDefaultBkgCol = guiParams.labelBackgroundCol;
    CONTEXT.labelDefaultMarkerCol = guiParams.labelMarkerCol;
    CONTEXT.bbDefaultCol = guiParams.bbDefaultCol;

    // display acceleration
    CONTEXT.useVBOs = guiParams.useVBOs;

    // other options
    CONTEXT.drawRoundedPoints = guiParams.drawRoundedPoints;

    // Per-view point size / line width — delegate to active view when
    // a secondary view is active so each window gets independent values.
    auto* av = activeSecondaryView();
    if (av) {
        const auto& vp = av->getViewportParameters();
        CONTEXT.defaultPointSize =
                static_cast<unsigned char>(vp.defaultPointSize);
        CONTEXT.defaultLineWidth =
                static_cast<unsigned char>(vp.defaultLineWidth);
        CONTEXT.display = av;
    } else {
        CONTEXT.defaultPointSize = static_cast<unsigned char>(
                ctx.viewportParams.defaultPointSize);
        CONTEXT.defaultLineWidth = static_cast<unsigned char>(
                ctx.viewportParams.defaultLineWidth);
        auto* ev = ecvViewManager::instance().getEffectiveView();
        CONTEXT.display = ev ? ev : primaryDT();
    }
    CONTEXT.currentLineWidth = CONTEXT.defaultLineWidth;
}

bool ecvDisplayTools::RenderToFile(QString filename,
                                   float zoomFactor /*=1.0*/,
                                   bool dontScaleFeatures /*=false*/,
                                   bool renderOverlayItems /*=false*/) {
    if (filename.isEmpty() || zoomFactor < 1.0e-2f) {
        return false;
    }

    QImage outputImage = RenderToImage(static_cast<int>(zoomFactor),
                                       renderOverlayItems, false, 0);

    if (outputImage.isNull()) {
        // an error occurred (message should have already been issued!)
        return false;
    }

    if (GetDisplayParameters().drawRoundedPoints) {
        // convert the image to plain RGB to avoid issues with points
        // transparency when saving to PNG
        outputImage = outputImage.convertToFormat(QImage::Format_RGB32);
    }

    bool success =
            outputImage.convertToFormat(QImage::Format_RGB32).save(filename);
    if (success) {
        CVLog::Print(QString("[Snapshot] File '%1' saved! (%2 x %3 pixels)")
                             .arg(filename)
                             .arg(outputImage.width())
                             .arg(outputImage.height()));
    } else {
        CVLog::Print(
                QString("[Snapshot] Failed to save file '%1'!").arg(filename));
    }

    return success;
}

void ecvDisplayTools::SetCameraPos(ecvViewContext& ctx, const CCVector3d& P) {
    if ((ctx.viewportParams.getCameraCenter() - P).norm2d() != 0.0) {
        ctx.viewportParams.setCameraCenter(P, true);
        SetCameraPosition(P);
        emit primaryDT()->cameraPosChanged(ctx.viewportParams.getCameraCenter());
        emit primaryDT()->cameraParamChanged();
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
    }
}

void ecvDisplayTools::SetCameraPos(const CCVector3d& P) {
    SetCameraPos(ecvViewManager::instance().resolveViewContext(), P);
}

const ecvGui::ParamStruct& ecvDisplayTools::GetDisplayParameters() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getDisplayParameters();
    }
    auto& vm = ecvViewManager::instance();
    if (vm.hasOverriddenDisplayParameters()) {
        vm.prepareOverriddenDisplayParameters();
        return vm.overriddenDisplayParameters();
    }
    const ecvGui::ParamStruct& params = ecvGui::Parameters();
    ecvGui::UpdateParameters();
    return params;
}

void ecvDisplayTools::GetGLCameraParameters(ccGLCameraParameters& params) {
    auto* av = activeSecondaryView();
    if (av) {
        av->getGLCameraParameters(params);
        return;
    }

    // get/compute the modelview matrix
    { GetViewMatrix(params.modelViewMat.data()); }

    // get/compute the projection matrix
    { GetProjectionMatrix(params.projectionMat.data()); }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    ccGLMatrixd rotationMat;
    rotationMat.setRotation(
            ccGLMatrixd::ToEigenMatrix3(params.modelViewMat).data());
    ctx.viewportParams.viewMat = rotationMat;
    double nearFar[2];
    GetCameraClip(nearFar);

    CCVector3d pivot;
    GetCenterOfRotation(pivot);
    ctx.viewportParams.setPivotPoint(pivot);

    ctx.viewportParams.zNear = nearFar[0];
    ctx.viewportParams.zFar = nearFar[1];
    ctx.viewportParams.fov_deg = static_cast<float>(GetCameraFovy());
    params.fov_deg = ctx.viewportParams.fov_deg;

    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = Width() * GetDevicePixelRatio();
    params.viewport[3] = Height() * GetDevicePixelRatio();
    SetGLViewport(QRect(0, 0, Width(), Height()));

    params.perspective = ctx.viewportParams.perspectiveView;
    params.pixelSize = ctx.viewportParams.pixelSize;
}

void ecvDisplayTools::SetDisplayParameters(const ecvGui::ParamStruct& params) {
    // Write-through: also update the active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        av->setDisplayParameters(params, true);
    }

    ecvViewManager::instance().setOverriddenDisplayParameters(params);

    ecvGui::Set(params);
}

void ecvDisplayTools::UpdateDisplayParameters(ecvViewContext& ctx) {
    double nearFar[2];
    GetCameraClip(nearFar);
    ctx.viewportParams.zNear = nearFar[0];
    ctx.viewportParams.zFar = nearFar[1];

    ccGLMatrixd viewMat;
    GetViewMatrix(viewMat.data());
    ccGLMatrixd rotationMat;
    rotationMat.setRotation(ccGLMatrixd::ToEigenMatrix3(viewMat).data());
    ctx.viewportParams.viewMat = rotationMat;

    CCVector3d pivot;
    GetCenterOfRotation(pivot);
    ctx.viewportParams.setPivotPoint(pivot);

    if (ctx.viewportParams.perspectiveView) {
        ctx.viewportParams.zoom = ComputePerspectiveZoom();
        ctx.viewportParams.fov_deg =
                static_cast<float>(GetCameraFovy());
    } else {
        ctx.viewportParams.fov_deg = static_cast<float>(
                cloudViewer::RadiansToDegrees(GetParallelScale(0)));
    }

    double pos[3];
    GetCameraPos(pos);
    ctx.viewportParams.setCameraCenter(CCVector3d::fromArray(pos), true);

    double focal[3];
    GetCameraFocal(focal);
    ctx.viewportParams.focal = CCVector3d::fromArray(focal);

    double up[3];
    GetCameraUp(up);
    ctx.viewportParams.up = CCVector3d::fromArray(up);
}

void ecvDisplayTools::UpdateDisplayParameters() {
    UpdateDisplayParameters(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::SetViewportParameters(
        ecvViewContext& ctx, const ecvViewportParameters& params) {
    auto* av = activeSecondaryView();
    if (av) {
        av->setViewportParameters(params);
    }

    ecvViewportParameters oldParams = ctx.viewportParams;
    ctx.viewportParams = params;

    if (oldParams.perspectiveView == params.perspectiveView) {
        if (oldParams.perspectiveView) {
            SetFov(ctx, params.fov_deg);
        } else {
            SetParallelScale(static_cast<double>(cloudViewer::DegreesToRadians(
                                     params.fov_deg)),
                             0);
        }
    } else {
        ctx.viewportParams.perspectiveView = oldParams.perspectiveView;
    }

    SetCameraClip(params.zNear, params.zFar);
    SetPivotPoint(params.getPivotPoint(), false, false);
    SetCameraPosition(params.getCameraCenter().u, params.focal.u, params.up.u);

    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();

    emit primaryDT()->baseViewMatChanged(ctx.viewportParams.viewMat);
    emit primaryDT()->pivotPointChanged(ctx.viewportParams.getPivotPoint());
    emit primaryDT()->cameraPosChanged(ctx.viewportParams.getCameraCenter());
    emit primaryDT()->fovChanged(ctx.viewportParams.fov_deg);
    emit primaryDT()->cameraParamChanged();
}

void ecvDisplayTools::SetViewportParameters(
        const ecvViewportParameters& params) {
    SetViewportParameters(ecvViewManager::instance().resolveViewContext(), params);
}

const ecvViewportParameters& ecvDisplayTools::GetViewportParameters(
        const ecvViewContext& ctx) {
    return ctx.viewportParams;
}

const ecvViewportParameters& ecvDisplayTools::GetViewportParameters() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getViewportParameters();
    }
    UpdateDisplayParameters();
    return GetViewportParameters(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::SetBubbleViewMode(ecvViewContext& ctx, bool state) {
    bool bubbleViewModeWasEnabled = ctx.bubbleViewModeEnabled;
    if (!ctx.bubbleViewModeEnabled && state) {
        ctx.preBubbleViewParameters = ctx.viewportParams;
    }

    if (state) {
        SetPerspectiveState(true, false);

        ctx.bubbleViewModeEnabled = true;

        ctx.bubbleViewFov_deg = 0.0f;
        SetBubbleViewFov(ctx, 90.0f);
    } else if (bubbleViewModeWasEnabled) {
        ctx.bubbleViewModeEnabled = false;
        SetPerspectiveState(ctx.preBubbleViewParameters.perspectiveView,
                            ctx.preBubbleViewParameters.objectCenteredView);

        SetViewportParameters(ctx, ctx.preBubbleViewParameters);
    }
}

void ecvDisplayTools::SetBubbleViewMode(bool state) {
    SetBubbleViewMode(ecvViewManager::instance().resolveViewContext(), state);
}

void ecvDisplayTools::SetBubbleViewFov(ecvViewContext& ctx, float fov_deg) {
    if (fov_deg < FLT_EPSILON || fov_deg > 180.0f) return;

    if (fov_deg != ctx.bubbleViewFov_deg) {
        ctx.bubbleViewFov_deg = fov_deg;

        if (ctx.bubbleViewModeEnabled) {
            InvalidateViewport();
            InvalidateVisualization();
            Deprecate3DLayer();
            emit primaryDT()->fovChanged(ctx.bubbleViewFov_deg);
            emit primaryDT()->cameraParamChanged();
        }
    }
}

void ecvDisplayTools::SetBubbleViewFov(float fov_deg) {
    SetBubbleViewFov(ecvViewManager::instance().resolveViewContext(), fov_deg);
}

void ecvDisplayTools::SetPixelSize(ecvViewContext& ctx, float pixelSize) {
    if (ctx.viewportParams.pixelSize != pixelSize) {
        ctx.viewportParams.pixelSize = pixelSize;
    }
    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();
}

void ecvDisplayTools::SetPixelSize(float pixelSize) {
    SetPixelSize(ecvViewManager::instance().resolveViewContext(), pixelSize);
}

void ecvDisplayTools::SetZoom(ecvViewContext& ctx, float value) {
    assert(!ctx.bubbleViewModeEnabled);

    if (value < CC_GL_MIN_ZOOM_RATIO)
        value = CC_GL_MIN_ZOOM_RATIO;
    else if (value > CC_GL_MAX_ZOOM_RATIO)
        value = CC_GL_MAX_ZOOM_RATIO;

    if (ctx.viewportParams.zoom != value) {
        ctx.viewportParams.zoom = value;
        InvalidateViewport();
        InvalidateVisualization();
    }
}

void ecvDisplayTools::SetZoom(float value) {
    SetZoom(ecvViewManager::instance().resolveViewContext(), value);
}

void ecvDisplayTools::UpdateZoom(ecvViewContext& ctx, float zoomFactor) {
    assert(!ctx.viewportParams.perspectiveView);

    if (zoomFactor > 0.0f && zoomFactor != 1.0f) {
        SetZoom(ctx, ctx.viewportParams.zoom * zoomFactor);
    }
}

void ecvDisplayTools::UpdateZoom(float zoomFactor) {
    UpdateZoom(ecvViewManager::instance().resolveViewContext(), zoomFactor);
}

void ecvDisplayTools::SetPickingMode(PICKING_MODE mode /*=DEFAULT_PICKING*/) {
    // Write-through: update both the active secondary view AND the
    // ecvViewManager-resolved primary context so that m_tools-> direct access
    // in QVTKWidgetCustom sees the correct value.
    auto* av = activeSecondaryView();
    if (av) {
        av->setPickingMode(mode);
    }

    auto& ctx = ecvViewManager::instance().resolveViewContext();
    if (ctx.pickingModeLocked) {
        if ((mode != ctx.pickingMode) && (mode != DEFAULT_PICKING))
            CVLog::Warning(
                    "[ecvDisplayTools::setPickingMode] Picking mode is locked! "
                    "Can't change it...");
        return;
    }

    QWidget* screen = GetCurrentScreen();
    switch (mode) {
        case DEFAULT_PICKING:
            mode = ENTITY_PICKING;
        case NO_PICKING:
        case ENTITY_PICKING:
            if (screen) screen->setCursor(QCursor(Qt::ArrowCursor));
            break;
        case POINT_OR_TRIANGLE_PICKING:
        case POINT_OR_TRIANGLE_OR_LABEL_PICKING:
        case TRIANGLE_PICKING:
        case POINT_PICKING:
            if (screen) screen->setCursor(QCursor(Qt::PointingHandCursor));
            break;
        default:
            break;
    }

    ctx.pickingMode = mode;
}

ecvDisplayTools::PICKING_MODE ecvDisplayTools::GetPickingMode() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getPickingMode();
    }
    return ecvViewManager::instance().resolveViewContext().pickingMode;
}

void ecvDisplayTools::LockPickingMode(ecvViewContext& ctx, bool state) {
    ctx.pickingModeLocked = state;
}

void ecvDisplayTools::LockPickingMode(bool state) {
    LockPickingMode(ecvViewManager::instance().resolveViewContext(), state);
}

bool ecvDisplayTools::IsPickingModeLocked(const ecvViewContext& ctx) {
    return ctx.pickingModeLocked;
}

bool ecvDisplayTools::IsPickingModeLocked() {
    return IsPickingModeLocked(ecvViewManager::instance().resolveViewContext());
}

double ecvDisplayTools::ComputeActualPixelSize(const ecvViewContext& ctx) {
    if (!ctx.viewportParams.perspectiveView) {
        return static_cast<double>(ctx.viewportParams.pixelSize /
                                   ctx.viewportParams.zoom);
    }

    int minScreenDim =
            std::min(ctx.glViewport.width(), ctx.glViewport.height());
    if (minScreenDim <= 0) return 1.0;

    double zoomEquivalentDist =
            (ctx.viewportParams.getCameraCenter() -
             ctx.viewportParams.getPivotPoint())
                    .norm();

    double currentFov_deg = static_cast<double>(
            ctx.bubbleViewModeEnabled ? ctx.bubbleViewFov_deg
                                     : ctx.viewportParams.fov_deg);
    return zoomEquivalentDist *
           std::tan(cloudViewer::DegreesToRadians(
                   std::min(currentFov_deg, 75.0))) /
           minScreenDim;
}

double ecvDisplayTools::ComputeActualPixelSize() {
    return ComputeActualPixelSize(ecvViewManager::instance().resolveViewContext());
}

bool ecvDisplayTools::IsRectangularPickingAllowed(const ecvViewContext& ctx) {
    return ctx.allowRectangularEntityPicking;
}

bool ecvDisplayTools::IsRectangularPickingAllowed() {
    return IsRectangularPickingAllowed(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::SetRectangularPickingAllowed(ecvViewContext& ctx,
                                                   bool state) {
    ctx.allowRectangularEntityPicking = state;
}

void ecvDisplayTools::SetRectangularPickingAllowed(bool state) {
    SetRectangularPickingAllowed(ecvViewManager::instance().resolveViewContext(), state);
}

void ecvDisplayTools::ShowPivotSymbol(ecvViewContext& ctx, bool state) {
    if (state && !ctx.pivotSymbolShown &&
        ctx.viewportParams.objectCenteredView &&
        ctx.pivotVisibility != PIVOT_HIDE) {
        InvalidateViewport();
        Deprecate3DLayer();
    }
    ctx.pivotSymbolShown = state;
}

void ecvDisplayTools::ShowPivotSymbol(bool state) {
    ShowPivotSymbol(ecvViewManager::instance().resolveViewContext(), state);
}

int ecvDisplayTools::GetFontPointSize() {
    return GetOptimizedFontSize(
            primaryDT()->m_captureMode.enabled
                    ? FontSizeModifier(GetDisplayParameters().defaultFontSize,
                                       primaryDT()->m_captureMode.zoomFactor)
                    : GetDisplayParameters().defaultFontSize);
}

int ecvDisplayTools::GetLabelFontPointSize() {
    return GetOptimizedFontSize(
            primaryDT()->m_captureMode.enabled
                    ? FontSizeModifier(GetDisplayParameters().labelFontSize,
                                       primaryDT()->m_captureMode.zoomFactor)
                    : GetDisplayParameters().labelFontSize);
}

QFont ecvDisplayTools::GetLabelDisplayFont() {
    QFont font = primaryDT()->m_font;
    font.setPointSize(GetLabelFontPointSize());
    return font;
}

void ecvDisplayTools::SetFocusToScreen() {
#ifdef CV_WINDOWS
    POINT lpPoint;
    POINT oldlpPoint;
    QPoint globalOldPoint = QCursor::pos();
    oldlpPoint.x = globalOldPoint.x();
    oldlpPoint.y = globalOldPoint.y();

    QRect screenRect = GetScreenRect();
    QPoint clickPos = screenRect.center();
    lpPoint.x = clickPos.x();
    lpPoint.y = clickPos.y();

    SetCursorPos(lpPoint.x, lpPoint.y);
    mouse_event(
            MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, 0,
            0, 0, 0);
    Sleep(20);
    SetCursorPos(oldlpPoint.x, oldlpPoint.y);
#else
    CVLog::Warning("only supported in windows!");
#endif

    if (GetCurrentScreen()) {
        GetCurrentScreen()->setFocus();
        if (GetCurrentScreen()->parentWidget()) {
            GetCurrentScreen()->parentWidget()->setFocus();
        }
    }
}

void ecvDisplayTools::ToBeRefreshed() {
    if (auto* av = ecvViewManager::instance().getEffectiveView()) {
        av->toBeRefreshed();
    } else if (auto* p = primaryDT()) {
        p->m_shouldBeRefreshed = true;
    }

    InvalidateViewport();
    InvalidateVisualization();
}

void ecvDisplayTools::RefreshDisplay(bool only2D /*=false*/,
                                     bool forceRedraw /* = true*/) {
    (void)forceRedraw;
    auto* p = primaryDT();
    if (!p || !p->m_shouldBeRefreshed) return;
    ecvViewManager::instance().refreshAll(only2D);
    p->m_shouldBeRefreshed = false;
}

void ecvDisplayTools::RedrawDisplay(ecvViewContext& ctx,
                                    bool only2D /*=false*/) {
    auto* dt = ecvViewManager::instance().displayTools();
    if (!dt) return;

    if (ctx.showDebugTraces) {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!primaryDT()->m_diagStrings.isEmpty()) {
            QStringList::iterator it = primaryDT()->m_diagStrings.begin();
            while (it != primaryDT()->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = primaryDT()->m_diagStrings.erase(it);
            }
        }
        primaryDT()->m_diagStrings
                << QString("only2D : %1").arg(only2D ? "true" : "false");
        primaryDT()->m_diagStrings << QString("ForceRedraw : %1")
                                          .arg(g_redrawDisplayForceRedraw ? "true"
                                                                         : "false");
    } else {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!primaryDT()->m_diagStrings.isEmpty()) {
            QStringList::iterator it = primaryDT()->m_diagStrings.begin();
            while (it != primaryDT()->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = primaryDT()->m_diagStrings.erase(it);
            }
        }
    }

    CheckIfRemove();
    if (primaryDT()->m_removeAllFlag) {
        Update();
        return;
    }

    SetFontPointSize(GetFontPointSize());

    if (!only2D) {
        Deprecate3DLayer();
    }

    // Clean outdated messages (global, not per-view)
    {
        std::list<MessageToDisplay>::iterator it =
                primaryDT()->m_messagesToDisplay.begin();
        qint64 currentTime_sec = primaryDT()->m_timer.elapsed() / 1000;
        while (it != primaryDT()->m_messagesToDisplay.end()) {
            if (it->messageValidity_sec < currentTime_sec) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                it->message));
                it = primaryDT()->m_messagesToDisplay.erase(it);
            } else {
                ++it;
            }
        }
    }

    ecvViewManager::instance().redrawAll(only2D, g_redrawDisplayForceRedraw);

    // === Post-render housekeeping ===
    // Phase M4: the legacy draw path through ecvViewManager's primary
    // ecvDisplayTools (background, 3D, foreground,
    // debug traces) has been removed.  Each ecvGLView::redraw() now contains
    // the full pipeline including ColorRamp, Messages, ScaleBar, and the
    // parameterized DrawClickableItems.

    if (primaryDT()->m_updateFBO || GetDisplayParameters().displayCross) {
        primaryDT()->m_updateFBO = false;
    }

    primaryDT()->m_shouldBeRefreshed = false;
    UpdateScreen();
}

void ecvDisplayTools::RedrawDisplay(bool only2D /*=false*/,
                                    bool forceRedraw /* = true*/) {
    g_redrawDisplayForceRedraw = forceRedraw;
    RedrawDisplay(ecvViewManager::instance().resolveViewContext(), only2D);
}

void ecvDisplayTools::SetGLViewport(ecvViewContext& ctx, const QRect& rect) {
    const int retinaScale = GetDevicePixelRatio();
    ctx.glViewport =
            QRect(rect.left() * retinaScale, rect.top() * retinaScale,
                  rect.width() * retinaScale, rect.height() * retinaScale);
    InvalidateViewport();
}

void ecvDisplayTools::SetGLViewport(const QRect& rect) {
    SetGLViewport(ecvViewManager::instance().resolveViewContext(), rect);
}

void ecvDisplayTools::drawCross() {}

void ecvDisplayTools::drawTrihedron() {}

void ecvDisplayTools::Draw3D(ecvViewContext& ctx,
                             CC_DRAW_CONTEXT& CONTEXT) {
    CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
    if (ctx.interactionFlags & INTERACT_TRANSFORM_ENTITIES) {
        CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    if (ctx.customLightEnabled || ctx.sunLightEnabled) {
        CONTEXT.drawingFlags |= CC_LIGHT_ENABLED;
    }

    if (primaryDT()->m_globalDBRoot) {
        primaryDT()->m_globalDBRoot->draw(CONTEXT);
    }

    if (primaryDT()->m_winDBRoot) {
        primaryDT()->m_winDBRoot->draw(CONTEXT);
    }

#if 0
	if (ctx.autoPickPivotAtCenter)
	{
		CCVector3d P;
		if (GetClick3DPos(ctx.glViewport.width() / 2, ctx.glViewport.height() / 2, P))
		{
			ctx.autoPivotCandidate = P;
		}
	}
#endif

    if (primaryDT()->m_globalDBRoot &&
        primaryDT()->m_globalDBRoot->getChildrenNumber()) {
    }
}

void ecvDisplayTools::Draw3D(CC_DRAW_CONTEXT& CONTEXT) {
    Draw3D(ecvViewManager::instance().resolveViewContext(), CONTEXT);
}

void ecvDisplayTools::HideShowEntities(const QStringList& viewIDs,
                                       ENTITY_TYPE hideShowEntityType,
                                       bool visibility) {
    CC_DRAW_CONTEXT context;
    context.hideShowEntityType = hideShowEntityType;
    context.visible = visibility;
    for (const QString& removeViewId : viewIDs) {
        context.viewID = removeViewId;
        HideShowEntities(context);
    }
}

bool ecvDisplayTools::HideShowEntities(const ccHObject* obj, bool visible) {
    if (!obj || !ecvDisplayTools::GetCurrentScreen()) {
        return false;
    }
    CC_DRAW_CONTEXT context;
    context.visible = visible;
    context.viewID = obj->getViewId();
    context.hideShowEntityType = obj->getEntityType();
    context.display = const_cast<ecvGenericGLDisplay*>(obj->getDisplay());
    ecvDisplayTools::HideShowEntities(context);

    if (!obj->getDisplay()) {

        const auto& views = ecvViewManager::instance().getAllViews();
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            for (auto* view : views) {
                if (!view || view == dt) continue;
                CC_DRAW_CONTEXT viewCtx = context;
                viewCtx.display = view;
                dt->hideShowEntities(viewCtx);
            }
        }
    }
    return true;
}

void ecvDisplayTools::RemoveEntities(const CC_DRAW_CONTEXT& CONTEXT) {
    if (auto* dt = ecvViewManager::instance().displayTools())
        dt->removeEntities(CONTEXT);
}

void ecvDisplayTools::RemoveEntities(const ccHObject* obj) {
    if (!obj || !ecvDisplayTools::GetCurrentScreen()) {
        return;
    }

    CC_DRAW_CONTEXT context;
    context.removeViewID = obj->getViewId();
    context.removeEntityType = obj->getEntityType();
    context.display = const_cast<ecvGenericGLDisplay*>(obj->getDisplay());
    ecvDisplayTools::RemoveEntities(context);

    if (!obj->getDisplay()) {
        const auto& views = ecvViewManager::instance().getAllViews();
        if (auto* rmDt = ecvViewManager::instance().displayTools()) {
            for (auto* view : views) {
                if (!view || view == rmDt) continue;
                CC_DRAW_CONTEXT viewCtx = context;
                viewCtx.display = view;
                rmDt->removeEntities(viewCtx);
            }
        }
    }
}

void ecvDisplayTools::RemoveEntities(const QStringList& viewIDs,
                                     ENTITY_TYPE removeEntityType) {
    CC_DRAW_CONTEXT context;
    context.removeEntityType = removeEntityType;
    for (const QString& removeViewId : viewIDs) {
        context.removeViewID = removeViewId;
        RemoveEntities(context);
    }
}

void ecvDisplayTools::DrawBackground(CC_DRAW_CONTEXT& CONTEXT) {
    const auto& ctx = ecvViewManager::instance().resolveViewContext();
    CONTEXT.drawingFlags = CC_DRAW_2D;
    if (ctx.interactionFlags & INTERACT_TRANSFORM_ENTITIES) {
        CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    // clear background
    {
        if (CONTEXT.clearDepthLayer) {
        }
        if (CONTEXT.clearColorLayer) {
            const ecvGui::ParamStruct& displayParams = GetDisplayParameters();
            if (displayParams.drawBackgroundGradient) {
                // draw the default gradient color background
                // we use the default background color for gradient start
                const ecvColor::Rgbub& bkgCol2 =
                        GetDisplayParameters().backgroundCol;

                // and the inverse of the text color for gradient stop
                ecvColor::Rgbub bkgCol1 = GetDisplayParameters().textDefaultCol;
                bkgCol1.r = 255 - GetDisplayParameters().textDefaultCol.r;
                bkgCol1.g = 255 - GetDisplayParameters().textDefaultCol.g;
                bkgCol1.b = 255 - GetDisplayParameters().textDefaultCol.b;
                CONTEXT.backgroundCol = bkgCol1;
                CONTEXT.backgroundCol2 = bkgCol2;
                CONTEXT.drawBackgroundGradient = true;

            } else {
                // use plain color as specified by the user
                const ecvColor::Rgbub& bkgCol = displayParams.backgroundCol;
                CONTEXT.backgroundCol = bkgCol;
                CONTEXT.backgroundCol2 = bkgCol;
                CONTEXT.drawBackgroundGradient = false;
            }

            if (auto* dt = ecvViewManager::instance().displayTools()) {
                dt->setBackgroundColor(CONTEXT);
            }
        }
    }
}

void ecvDisplayTools::DrawForeground(CC_DRAW_CONTEXT& CONTEXT) {
    /****************************************/
    /****  PASS: 2D/FOREGROUND/NO LIGHT  ****/
    /****************************************/

    const auto& fgCtx = ecvViewManager::instance().resolveViewContext();
    CONTEXT.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    if (fgCtx.interactionFlags & INTERACT_TRANSFORM_ENTITIES) {
        CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    // we draw 2D entities
    if (primaryDT()->m_globalDBRoot) primaryDT()->m_globalDBRoot->draw(CONTEXT);
    if (primaryDT()->m_winDBRoot) primaryDT()->m_winDBRoot->draw(CONTEXT);

    // current displayed scalar field color ramp (if any)
    ccRenderingTools::DrawColorRamp(CONTEXT);

    ecvGenericGLDisplay* fgView = ecvViewManager::instance().getEffectiveView();
    if (!fgView) fgView = primaryDT();
    fgView->clickableItemsRef().clear();

    /*** overlay entities ***/
    if (fgCtx.displayOverlayEntities) {
        if (!primaryDT()->m_captureMode.enabled ||
            primaryDT()->m_captureMode.renderOverlayItems) {
            // scale: only in ortho mode
            if (!fgCtx.viewportParams.perspectiveView) {
                SetScaleBarVisible(true);
            } else {
                SetScaleBarVisible(false);
            }
            UpdateScreen();
        }

        if (!primaryDT()->m_captureMode.enabled) {
            int yStart = 0;

            // current messages (if valid)
            if (!primaryDT()->m_messagesToDisplay.empty()) {
                QFont font = primaryDT()->m_font;
                QFontMetrics fm(font);
                int margin = fm.height() / 4;
                int ll_currentHeight =
                        fgCtx.glViewport.height() - 10;
                int uc_currentHeight = 10;

                for (const auto& message : primaryDT()->m_messagesToDisplay) {
                    switch (message.position) {
                        case LOWER_LEFT_MESSAGE: {
                            RenderText(10, ll_currentHeight, message.message,
                                       font);
                            int messageHeight = fm.height();
                            ll_currentHeight -= (messageHeight + margin);
                        } break;
                        case UPPER_CENTER_MESSAGE: {
                            QRect rect = fm.boundingRect(message.message);
                            int x = (fgCtx.glViewport.width() -
                                     rect.width()) /
                                    2;
                            int y = uc_currentHeight + rect.height();
                            RenderText(x, y, message.message, font);
                            uc_currentHeight += (rect.height() + margin);
                        } break;
                        case SCREEN_CENTER_MESSAGE: {
                            QFont newFont(font);
                            int fontSize = GetOptimizedFontSize(12);
                            newFont.setPointSize(fontSize);
                            QRect rect = QFontMetrics(newFont).boundingRect(
                                    message.message);
                            RenderText(
                                    (fgCtx.glViewport.width() -
                                     rect.width()) /
                                            2,
                                    (fgCtx.glViewport.height() -
                                     rect.height()) /
                                            2,
                                    message.message, newFont);
                        } break;
                    }
                }
            }

            // hot-zone
            { ecvDisplayTools::DrawClickableItems(0, yStart); }
        }
    }
}

void ecvDisplayTools::Redraw2DLabel() {
    ccHObject::Container labels;
    FilterByEntityType(labels, CV_TYPES::LABEL_2D);

    for (auto& label : labels) {
        if (!label->isA(CV_TYPES::LABEL_2D) || !label->isEnabled() ||
            !label->isVisible())
            continue;
        cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
        if (!l) continue;

        CC_DRAW_CONTEXT context;
        GetContext(context);
        auto* disp = l->getDisplay();
        auto* dt = ecvViewManager::instance().displayTools();
        if (disp && dt && disp != dt) {
            context.display = disp;
            auto* viewCtx = disp->viewContext();
            if (viewCtx) {
                context.glW = viewCtx->glViewport.width();
                context.glH = viewCtx->glViewport.height();
            }
        }
        // Sync both 3D markers and 2D overlay so selection highlight,
        // marker colors, and caption state stay consistent.
        l->update3DLabelView(context, false);
        l->update2DLabelView(context, false);
    }

    if (QWidget* w = GetCurrentScreen()) {
        w->update();
    }
    if (ecvViewManager::instance().viewCount() > 1) {
        ecvViewManager::instance().redrawAll(true, true);
    }
}

void ecvDisplayTools::Update2DLabel(bool immediateUpdate /* = false*/) {
    // Only update overlay data for visible labels.
    // Do NOT add labels to m_activeItems here — active items should only be
    // populated by explicit user interaction (click/pick), not by periodic
    // timer updates.  The old approach of clearing and refilling m_activeItems
    // every 50 ms caused cross-window pollution and unintended label movement
    // during camera rotation.
    ccHObject::Container labels;
    FilterByEntityType(labels, CV_TYPES::LABEL_2D);

    for (auto& label : labels) {
        if (label->isA(CV_TYPES::LABEL_2D) && label->isEnabled() &&
            label->isVisible()) {
            cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
            if (!l) continue;

            if (immediateUpdate) {
                CC_DRAW_CONTEXT context;
                GetContext(context);
                auto* disp = l->getDisplay();
                auto* dt = ecvViewManager::instance().displayTools();
                if (disp && dt && disp != dt) {
                    context.display = disp;
                    auto* viewCtx = disp->viewContext();
                    if (viewCtx) {
                        context.glW = viewCtx->glViewport.width();
                        context.glH = viewCtx->glViewport.height();
                    }
                }
                l->update2DLabelView(context);
            }
        } else if (label->isA(CV_TYPES::VIEWPORT_2D_LABEL)) {
            cc2DViewportLabel* l = ccHObjectCaster::To2DViewportLabel(label);
            if (!l) continue;
            l->clear2Dviews();
        }
    }
}

void ecvDisplayTools::Pick2DLabel(int x, int y) {
    auto* dt = ecvViewManager::instance().displayTools();
    QString id = dt ? dt->pick2DLabel(x, y) : QString();

    std::list<ccInteractor*>* itemsPtr = nullptr;
    if (auto* effView = ecvViewManager::instance().getEffectiveView()) {
        itemsPtr = &effView->activeItemsRef();
    } else if (auto* p = primaryDT()) {
        itemsPtr = &p->m_activeItems;
    }
    if (!itemsPtr) return;

    itemsPtr->clear();
    if (!id.isEmpty()) {
        ccHObject::Container labels;
        FilterByEntityType(labels, CV_TYPES::LABEL_2D);
        for (auto& label : labels) {
            if (label->isA(CV_TYPES::LABEL_2D) && label->isEnabled() &&
                label->isVisible()) {
                cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
                if (l->getViewId().compare(id) == 0) {
                    itemsPtr->push_back(l);
                }
            }
        }
    }
}

void ecvDisplayTools::FilterByEntityType(ccHObject::Container& labels,
                                         CV_CLASS_ENUM type) {
    auto& vm = ecvViewManager::instance();

    // Per-view architecture: prefer the picking target view or effective view's
    // databases. Fall back to ecvViewManager::globalDBRoot / singleton DBs.
    ccHObject* sceneDB = nullptr;
    ccHObject* ownDB = nullptr;

    if (primaryDT()->m_pickingTargetView) {
        sceneDB = primaryDT()->m_pickingTargetView->getSceneDB();
        ownDB = primaryDT()->m_pickingTargetView->getOwnDB();
    }
    if (!sceneDB) {
        if (auto* ev = vm.getEffectiveView()) {
            sceneDB = ev->getSceneDB();
            ownDB = ev->getOwnDB();
        }
    }
    if (!sceneDB) sceneDB = vm.globalDBRoot();
    if (!sceneDB) {
        if (auto* dt = vm.displayTools()) sceneDB = dt->m_globalDBRoot;
    }
    if (!ownDB) {
        if (auto* dt = vm.displayTools()) ownDB = dt->m_winDBRoot;
    }

    if (sceneDB) sceneDB->filterChildren(labels, true, type);
    if (ownDB) ownDB->filterChildren(labels, true, type);
}

void ecvDisplayTools::RenderText(
        int x,
        int y,
        const QString& str,
        const QFont& font /*=QFont()*/,
        const ecvColor::Rgbub& color /* = ecvColor::defaultLabelBkgColor*/,
        const QString& id,
        ecvGenericGLDisplay* display /*=nullptr*/,
        double bkgAlpha /*=0.0*/,
        const double* bkgColor /*=nullptr*/) {
    CC_DRAW_CONTEXT context;
    context.display = display;
    if (id.isEmpty()) {
        context.viewID = str;
    } else {
        context.viewID = id;
    }

    context.textParam.text = str;
    context.textParam.display3D = false;
    context.textParam.font = font;
    context.textParam.font.setPointSize(font.pointSize());
    context.textParam.bkgAlpha = bkgAlpha;
    if (bkgColor && bkgAlpha > 0) {
        context.textParam.bkgColor[0] = bkgColor[0];
        context.textParam.bkgColor[1] = bkgColor[1];
        context.textParam.bkgColor[2] = bkgColor[2];
    }

    context.textDefaultCol = color;
    int vpH = viewportHeightFor(display);
    if (context.textParam.display3D) {
        context.textParam.textScale = GetPlatformAwareDPIScale();
        CCVector3d input3D(x, vpH - y, 0);
        CCVector3d output2D;
        ToWorldPoint(input3D, output2D);
        context.textParam.textPos.x = output2D.x;
        context.textParam.textPos.y = output2D.y;
        context.textParam.textPos.z = output2D.z;
    } else {
        context.textParam.textPos.x = x;
        context.textParam.textPos.y = vpH - y;
        context.textParam.textPos.z = 0;
    }
    DisplayText(context);
}

void ecvDisplayTools::RenderText(
        double x,
        double y,
        double z,
        const QString& str,
        const QFont& font /*=QFont()*/,
        const ecvColor::Rgbub& color /* = ecvColor::defaultLabelBkgColor*/,
        const QString& id) {
    // get the actual viewport / matrices
    ccGLCameraParameters camera;
    ecvDisplayTools::GetViewerPos(camera.viewport);
    ecvDisplayTools::GetProjectionMatrix(camera.projectionMat.data());
    ecvDisplayTools::GetViewMatrix(camera.modelViewMat.data());

    CCVector3d Q2D(0, 0, 0);
    if (camera.project(CCVector3d(x, y, z), Q2D)) {
        Q2D.y = GlHeight() - 1 - Q2D.y;
        RenderText(Q2D.x, Q2D.y, str, font, color, id);
    }
}

void ecvDisplayTools::Display3DLabel(const QString& str,
                                     const CCVector3& pos3D,
                                     const ecvColor::Rgbub* color /*=nullptr*/,
                                     const QFont& font /*=QFont()*/) {
    ecvColor::Rgbub col(color ? color->rgb
                              : GetDisplayParameters().textDefaultCol.rgb);
    RenderText(pos3D.x, pos3D.y, pos3D.z, str, font, col);
}

void ecvDisplayTools::DisplayText(
        const QString& text,
        int x,
        int y,
        unsigned char align /*=ALIGN_HLEFT|ALIGN_VTOP*/,
        float bkgAlpha /*=0*/,
        const unsigned char* rgbColor /*=0*/,
        const QFont* font /*=0*/,
        const QString& id /*=""*/,
        ecvGenericGLDisplay* display /*=nullptr*/) {
    int vpH = viewportHeightFor(display);
    int x2 = x;
    int y2 = vpH - 1 - y;
    double bkgInfoForText[4] = {0, 0, 0, 0};

    // actual text color
    const unsigned char* col =
            (rgbColor ? rgbColor : GetDisplayParameters().textDefaultCol.rgb);

    QFont realFont =
            (font ? *font
                  : (ecvViewManager::instance().displayTools()
                             ? ecvViewManager::instance().displayTools()->m_font
                             : QFont()));
    QFont textFont = realFont;
    QFontMetrics fm(textFont);
    int margin = fm.height() / 4;

    if (align != ALIGN_DEFAULT || bkgAlpha != 0.0f) {
        QRect rect = fm.boundingRect(text);

        // text alignment
        if (align & ALIGN_HMIDDLE)
            x2 -= rect.width() / 2;
        else if (align & ALIGN_HRIGHT)
            x2 -= rect.width();
        if (align & ALIGN_VMIDDLE)
            y2 += rect.height() / 2;
        else if (align & ALIGN_VBOTTOM)
            y2 += rect.height();

        // background is not totally transparent
        if (bkgAlpha != 0.0f) {
            const float invertedCol[4] = {(255 - col[0]) / 255.0f,
                                          (255 - col[1]) / 255.0f,
                                          (255 - col[2]) / 255.0f, bkgAlpha};

            // Always render background via VTK text actor's BackgroundColor/
            // BackgroundOpacity. This ensures the background is always
            // co-located with the text, regardless of which view it's in.
            bkgInfoForText[0] = invertedCol[0];
            bkgInfoForText[1] = invertedCol[1];
            bkgInfoForText[2] = invertedCol[2];
            bkgInfoForText[3] = bkgAlpha;
        }
    }

    if (align & ALIGN_VBOTTOM)
        y2 -= margin;  // empirical compensation
    else if (align & ALIGN_VMIDDLE)
        y2 -= margin / 2;  // empirical compensation

    ecvColor::Rgbub textColor(col);
    RenderText(x2, y2, text, realFont, textColor, id, display,
               bkgInfoForText[3], bkgInfoForText);
}

void ecvDisplayTools::DisplayTexture2DPosition(QImage image,
                                               const QString& id,
                                               int x,
                                               int y,
                                               int w,
                                               int h,
                                               unsigned char alpha /*=255*/) {
    WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_IMAGE, id);
    param.image = image;
    param.opacity = alpha / 255.0f;
    param.rect = QRect(x, y, w, h);
    DrawWidgets(param, true);
}

void ecvDisplayTools::ClearBubbleView(ecvGenericGLDisplay* display) {
    if (!display) {
        display = ecvViewManager::instance().getEffectiveView();
    }
    if (!display) return;
    ecvHotZone* hz = display->hotZonePtrRef();
    if (!hz) return;

    auto makeParam = [display](WIDGETS_TYPE type, const QString& id) {
        WIDGETS_PARAMETER p(type, id);
        p.context.display = display;
        return p;
    };

    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, hz->bbv_label));
    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, hz->fs_label));
    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, hz->psi_label));
    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, hz->lsi_label));
    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, "Exit"));
    RemoveWidgets(makeParam(WIDGETS_TYPE::WIDGET_T2D, "clicked_items"));
    RemoveWidgets(
            makeParam(WIDGETS_TYPE::WIDGET_RECTANGLE_2D, "clicked_items"));
    RemoveWidgets(
            makeParam(WIDGETS_TYPE::WIDGET_POINTS_2D, "clicked_items"));
}

void ecvDisplayTools::DrawClickableItems(int xStart0, int& yStart) {
    ecvGenericGLDisplay* view = ecvViewManager::instance().getEffectiveView();
    if (!view) return;
    DrawClickableItems(xStart0, yStart, view->hotZonePtrRef(),
                       view->clickableItemsRef(), view);
}

void ecvDisplayTools::DrawClickableItems(
        int xStart0,
        int& yStart,
        HotZone*& hotZone,
        std::vector<ecvClickableItem>& clickableItems,
        ecvGenericGLDisplay* display) {
    const static char* CLICKED_ITEMS = "clicked_items";
    QWidget* hzWin = display && display->asWidget() ? display->asWidget()
                                                    : GetCurrentScreen();
    if (!hotZone) {
        hotZone = new HotZone(hzWin);
    } else if (GetPlatformAwareDPIScale() != hotZone->pixelDeviceRatio) {
        hotZone->updateInternalVariables(hzWin);
    }

    hotZone->topCorner =
            QPoint(xStart0, yStart) +
            QPoint(hotZone->margin, hotZone->margin);

    bool fullScreenEnabled = (display && display->viewContext())
            ? display->viewContext()->exclusiveFullscreen
            : ExclusiveFullScreen();

    ecvViewContext* dispCtx = display ? display->viewContext() : nullptr;
    const ecvViewContext& hzCtx =
            dispCtx ? *dispCtx : ecvViewManager::instance().resolveViewContext();
    if (!hzCtx.clickableItemsVisible && !hzCtx.bubbleViewModeEnabled &&
        !fullScreenEnabled) {
        ClearBubbleView(display);
        return;
    }

    int fullW = hzCtx.glViewport.width();
    int fullH = hzCtx.glViewport.height();
    (void)fullW;

    {
        WIDGETS_PARAMETER rmParam(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS);
        rmParam.context.display = display;
        RemoveWidgets(rmParam);
    }

    // draw semi-transparent background
    {
        QRect areaRect = hotZone->rect(hzCtx.clickableItemsVisible,
                                       hzCtx.bubbleViewModeEnabled,
                                       fullScreenEnabled);
        areaRect.translate(hotZone->topCorner);

        WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                CLICKED_ITEMS);
        param.context.display = display;
        param.color = ecvColor::FromRgba(ecvColor::odarkGrey);
        param.color.a = 210 / 255.0f;
        int x0 = areaRect.x();
        int y0 = fullH - areaRect.y() - areaRect.height();
        param.rect = QRect(x0, y0, areaRect.width(), areaRect.height());
        DrawWidgets(param, false);
    }

    yStart = hotZone->topCorner.y();
    int offset = 0;
#ifdef Q_OS_MAC
    offset = hotZone->margin / 3;
#endif
    int iconSize = hotZone->iconSize;

    if (fullScreenEnabled) {
        int xStart = hotZone->topCorner.x();

        RenderText(xStart,
                   yStart + offset + hotZone->yTextBottomLineShift,
                   hotZone->fs_label, hotZone->font,
                   ecvColor::defaultLabelBkgColor, CLICKED_ITEMS, display);

        xStart += hotZone->fs_labelRect.width() + hotZone->margin;

#ifdef Q_OS_MAC
        xStart += hotZone->margin * 4;
#endif
        //"full-screen" icon
        {
            int x0 = xStart;
            int y0 = fullH - (yStart + iconSize);
            WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                    CLICKED_ITEMS);
            param.context.display = display;
            param.color = ecvColor::FromRgba(ecvColor::ored);
            param.rect = QRect(x0, y0, iconSize + offset, iconSize);
            DrawWidgets(param, false);

            WIDGETS_PARAMETER texParam(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS);
            texParam.context.display = display;
            texParam.color = ecvColor::bright;
            texParam.text = "Exit";
            texParam.rect =
                    QRect(x0, fullH - (yStart + offset / 2 + 3 * iconSize / 4),
                          iconSize, iconSize);
            texParam.fontSize = hotZone->font.pointSize();
            DrawWidgets(texParam, false);
            clickableItems.emplace_back(
                    ClickableItem::LEAVE_FULLSCREEN_MODE,
                    QRect(xStart, yStart, iconSize, iconSize));
            xStart += iconSize;
        }

        yStart += iconSize;
        yStart += hotZone->margin;
    }

    if (hzCtx.bubbleViewModeEnabled) {
        int xStart = hotZone->topCorner.x();

        RenderText(xStart,
                   yStart + offset + hotZone->yTextBottomLineShift,
                   hotZone->bbv_label, hotZone->font,
                   ecvColor::defaultLabelBkgColor, "", display);

        xStart += hotZone->bbv_labelRect.width() + hotZone->margin;
#ifdef Q_OS_MAC
        xStart += hotZone->margin * 4;
#endif

        //"exit" icon
        {
            clickableItems.emplace_back(
                    ClickableItem::LEAVE_BUBBLE_VIEW_MODE,
                    QRect(xStart, yStart, hotZone->iconSize,
                          hotZone->iconSize));
            xStart += hotZone->iconSize;
        }

        yStart += hotZone->iconSize;
        yStart += hotZone->margin;
    }

    if (hzCtx.clickableItemsVisible) {
        ecvColor::Rgb textColor = ecvColor::Rgb(hotZone->color);
        WIDGETS_PARAMETER widgetParam(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                      CLICKED_ITEMS);
        widgetParam.context.display = display;
        widgetParam.color = ecvColor::FromRgba(ecvColor::ogreen);
        WIDGETS_PARAMETER sepParam(WIDGETS_TYPE::WIDGET_POINTS_2D,
                                   CLICKED_ITEMS);
        sepParam.context.display = display;
        sepParam.color = widgetParam.color;
        sepParam.color.a = 0.5f;

        // default point size
        {
            int xStart = hotZone->topCorner.x();

            RenderText(
                    xStart,
                    yStart + offset + hotZone->yTextBottomLineShift,
                    hotZone->psi_label, hotZone->font,
                    textColor, CLICKED_ITEMS, display);

            xStart += hotZone->psi_labelRect.width() + hotZone->margin;
#ifdef Q_OS_MAC
            xStart += hotZone->margin * 4;
#else
            xStart -= iconSize;
#endif
            //"minus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);
                clickableItems.emplace_back(
                        ClickableItem::DECREASE_POINT_SIZE,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            // separator
            {
                sepParam.radius =
                        hzCtx.viewportParams.defaultPointSize / 2;
                int x0 = xStart + hotZone->margin;
                int y0 = fullH - (yStart + iconSize / 2);
                sepParam.rect = QRect(x0, y0, iconSize, iconSize);
                DrawWidgets(sepParam, false);
                xStart += hotZone->margin * 2;
            }

            //"plus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);
                x0 = xStart + 3 * iconSize / 8;
                y0 = fullH - (yStart + 7 * iconSize / 8);
                widgetParam.rect = QRect(x0, y0, iconSize / 4, iconSize);
                DrawWidgets(widgetParam, false);

                clickableItems.emplace_back(
                        ClickableItem::INCREASE_POINT_SIZE,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            yStart += iconSize;
            yStart += hotZone->margin;
        }

        // default line size
        {
            int xStart = hotZone->topCorner.x();

            RenderText(
                    xStart,
                    yStart + offset + hotZone->yTextBottomLineShift,
                    hotZone->lsi_label, hotZone->font,
                    textColor, CLICKED_ITEMS, display);

            xStart += hotZone->lsi_labelRect.width() + hotZone->margin;
#ifdef Q_OS_MAC
            xStart += hotZone->margin * 4;
#else
            xStart -= iconSize;
#endif
            //"minus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);

                clickableItems.emplace_back(
                        ClickableItem::DECREASE_LINE_WIDTH,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            // separator
            {
                sepParam.radius =
                        hzCtx.viewportParams.defaultLineWidth / 2;
                int x0 = xStart + hotZone->margin;
                int y0 = fullH - (yStart + iconSize / 2);
                sepParam.rect = QRect(x0, y0, iconSize, iconSize);
                DrawWidgets(sepParam, false);
                xStart += hotZone->margin * 2;
            }

            //"plus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);
                x0 = xStart + 3 * iconSize / 8;
                y0 = fullH - (yStart + 7 * iconSize / 8);
                widgetParam.rect = QRect(x0, y0, iconSize / 4, iconSize);
                DrawWidgets(widgetParam, false);

                clickableItems.emplace_back(
                        ClickableItem::INCREASE_LINE_WIDTH,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            yStart += iconSize;
            yStart += hotZone->margin;
        }
    }
}

void ecvDisplayTools::CheckIfRemove() {
    if (primaryDT()->m_removeAllFlag) {
        CC_DRAW_CONTEXT context;
        context.removeEntityType = ENTITY_TYPE::ECV_ALL;
        RemoveEntities(context);
        SetRemoveAllFlag(false);
    } else if (primaryDT()->m_removeFlag) {
        for (const removeInfo& rmInfo : primaryDT()->m_removeInfos) {
            if (rmInfo.removeType == ENTITY_TYPE::ECV_NONE) continue;
            // octree and kdtree object has been deleted before
            if (rmInfo.removeType == ENTITY_TYPE::ECV_OCTREE) continue;
            if (rmInfo.removeType == ENTITY_TYPE::ECV_KDTREE) continue;
            if (rmInfo.removeType == ENTITY_TYPE::ECV_2DLABLE) continue;
            if (rmInfo.removeType == ENTITY_TYPE::ECV_SENSOR) continue;

            CC_DRAW_CONTEXT context;
            context.removeEntityType = rmInfo.removeType;
            context.removeViewID = rmInfo.removeId;
            RemoveEntities(context);
            RemoveBB(context);
        }
        primaryDT()->m_removeFlag = false;
    }
}

void ecvDisplayTools::RemoveBB(CC_DRAW_CONTEXT context) {
    context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
    context.removeViewID = QString("BBox-") + context.removeViewID;
    RemoveEntities(context);
}

void ecvDisplayTools::RemoveBB(const QString& viewId) {
    CC_DRAW_CONTEXT context;
    context.removeViewID = viewId;
    RemoveBB(context);
}

void ecvDisplayTools::ChangeEntityProperties(PROPERTY_PARAM& propertyParam,
                                             bool autoUpdate /* = true*/) {
    if (propertyParam.entity) {
        if (propertyParam.entity->isKindOf(CV_TYPES::PRIMITIVE)) {
            propertyParam.entityType = ConvertToEntityType(CV_TYPES::PRIMITIVE);
        } else {
            propertyParam.entityType =
                    ConvertToEntityType(propertyParam.entity->getClassID());
        }

        propertyParam.viewId = propertyParam.entity->getViewId();
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            dt->changeEntityProperties(propertyParam);
        }
        if (autoUpdate) {
            UpdateScreen();
        }
    }
}

void ecvDisplayTools::DrawWidgets(const WIDGETS_PARAMETER& param,
                                  bool update /* = false*/) {
    auto* dt = ecvViewManager::instance().displayTools();
    if (!dt) return;

    ccHObject* entity = param.entity;
    int viewport = param.viewport;
    switch (param.type) {
        case WIDGETS_TYPE::WIDGET_COORDINATE:
            ShowOrientationMarker();
            break;
        case WIDGETS_TYPE::WIDGET_BBOX:
            break;

        case WIDGETS_TYPE::WIDGET_T2D: {
            QFont textFont = dt->m_font;
            const_cast<WIDGETS_PARAMETER*>(&param)->fontSize =
                    textFont.pointSize();

            dt->drawWidgets(param);
        } break;
        case WIDGETS_TYPE::WIDGET_IMAGE:
        case WIDGETS_TYPE::WIDGET_LINE_2D:
        case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
        case WIDGETS_TYPE::WIDGET_POINTS_2D:
        case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
        case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
        case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
        case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
            dt->drawWidgets(param);
            break;
        case WIDGETS_TYPE::WIDGET_LINE_3D:
            if (param.lineWidget.valid) {
                dt->drawWidgets(param);
            }
            break;
        case WIDGETS_TYPE::WIDGET_POLYLINE: {
            // context initialization
            CC_DRAW_CONTEXT CONTEXT;
            GetContext(CONTEXT);
            ccPolyline* poly = ccHObjectCaster::ToPolyline(entity);
            if (poly->is2DMode()) {
                CONTEXT.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
            } else {
                CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
            }

            if (GetInteractionMode() & INTERACT_TRANSFORM_ENTITIES) {
                CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
            }
            CONTEXT.defaultViewPort = viewport;
            poly->draw(CONTEXT);
        } break;
        case WIDGETS_TYPE::WIDGET_SPHERE:
        case WIDGETS_TYPE::WIDGET_POINT:
            dt->drawWidgets(param);
            break;
        case WIDGETS_TYPE::WIDGET_CAPTION:
            dt->drawWidgets(param);
            break;
        case WIDGETS_TYPE::WIDGET_T3D:
            break;
        default:
            break;
    }

    if (update) {
        UpdateScreen();
    }
}

void ecvDisplayTools::removeWidgets(const WIDGETS_PARAMETER& param) {
    RemoveWidgets(param);
}

void ecvDisplayTools::RemoveWidgets(const WIDGETS_PARAMETER& param,
                                    bool update /* = false*/) {
    CC_DRAW_CONTEXT context;
    context.display = param.context.display;
    switch (param.type) {
        case WIDGETS_TYPE::WIDGET_COORDINATE:
            break;
        case WIDGETS_TYPE::WIDGET_BBOX: {
            context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
            context.viewID = QString("BBox-") + param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_POLYGONMESH: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_MESH;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_LINE_3D:
        case WIDGETS_TYPE::WIDGET_POLYLINE: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_LINES_3D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_CAPTION: {
            context.removeEntityType = ENTITY_TYPE::ECV_CAPTION;
            context.defaultViewPort = param.viewport;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_SCALAR_BAR: {
            context.removeEntityType = ENTITY_TYPE::ECV_SCALAR_BAR;
            context.defaultViewPort = param.viewport;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_SPHERE:
        case WIDGETS_TYPE::WIDGET_POINT: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_IMAGE: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_IMAGE;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_POINTS_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_MARK_POINT;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_CIRCLE_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_CIRCLE_2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_TRIANGLE_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_TRIANGLE_2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_POLYLINE_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_POLYLINE_2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_LINE_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_LINES_2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_RECTANGLE_2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_RECTANGLE_2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_T2D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_TEXT2D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        case WIDGETS_TYPE::WIDGET_T3D: {
            context.defaultViewPort = param.viewport;
            context.removeEntityType = ENTITY_TYPE::ECV_TEXT3D;
            context.removeViewID = param.viewID;
            RemoveEntities(context);
        } break;
        default:
            break;
    }

    if (update) {
        UpdateScreen();
    }
}

void ecvDisplayTools::RemoveAllWidgets(bool update /* = true*/) {
    CC_DRAW_CONTEXT context;
    context.removeEntityType = ENTITY_TYPE::ECV_ALL;
    RemoveEntities(context);

    if (update) {
        UpdateScreen();
    }
}

void ecvDisplayTools::Remove3DLabel(const QString& view_id) {
    CC_DRAW_CONTEXT context;
    context.removeViewID = view_id;
    context.removeEntityType = ENTITY_TYPE::ECV_TEXT2D;
    RemoveEntities(context);
    UpdateScreen();
}

bool ecvDisplayTools::GetClick3DPos(const ecvViewContext& ctx,
                                    int x, int y, CCVector3d& P3D) {
    ccGLCameraParameters camera;
    GetGLCameraParameters(camera);

    y = ctx.glViewport.height() - 1 - y;

    double glDepth = GetGLDepth(x, y);
    if (glDepth == 1.0) {
        return false;
    }
    CCVector3d P2D(x, y, glDepth);
    return camera.unproject(P2D, P3D);
}

bool ecvDisplayTools::GetClick3DPos(int x, int y, CCVector3d& P3D) {
    return GetClick3DPos(ecvViewManager::instance().resolveViewContext(), x, y, P3D);
}

void ecvDisplayTools::DrawPivot(const ecvViewContext& ctx) {
    if (!ctx.viewportParams.objectCenteredView ||
        (ctx.pivotVisibility == PIVOT_HIDE) ||
        (ctx.pivotVisibility == PIVOT_SHOW_ON_MOVE &&
         !ctx.pivotSymbolShown)) {
        return;
    }

    CCVector3d tranlateTartget = CCVector3d(
            ctx.viewportParams.getPivotPoint().x,
            ctx.viewportParams.getPivotPoint().y,
            ctx.viewportParams.getPivotPoint().z);

    double symbolRadius =
            CC_DISPLAYED_PIVOT_RADIUS_PERCENT *
            std::min(ctx.glViewport.width(), ctx.glViewport.height()) / 2.0;

    {
        ccSphere sphere(static_cast<PointCoordinateType>(10.0 / symbolRadius));
        sphere.setColor(ecvColor::yellow);
        sphere.showColors(true);
        sphere.setVisible(true);
        sphere.setEnabled(true);
        CC_DRAW_CONTEXT CONTEXT;
        GetContext(CONTEXT);
        CONTEXT.drawingFlags =
                CC_DRAW_3D | CC_DRAW_FOREGROUND | CC_LIGHT_ENABLED;
        sphere.draw(CONTEXT);
    }
}

void ecvDisplayTools::DrawPivot() {
    DrawPivot(ecvViewManager::instance().resolveViewContext());
}

void ecvDisplayTools::SetCurrentScreen(QWidget* widget) {
    primaryDT()->m_currentScreen = widget;
    if (widget) {
        widget->update();
    }
}

// -- ecvGenericGLDisplay overrides (primary window delegation) --

void ecvDisplayTools::redraw(bool only2D, bool forceRedraw) {
    RedrawDisplay(only2D, forceRedraw);
}

void ecvDisplayTools::refresh(bool only2D) { RefreshDisplay(only2D); }

void ecvDisplayTools::toBeRefreshed() { m_shouldBeRefreshed = true; }

const ecvViewportParameters& ecvDisplayTools::getViewportParameters() const {
    return ecvViewManager::instance().resolveViewContext().viewportParams;
}

void ecvDisplayTools::setViewportParameters(
        const ecvViewportParameters& params) {
    SetViewportParameters(params);
}

void ecvDisplayTools::setPerspectiveState(bool state, bool objectCenteredView) {
    SetPerspectiveState(state, objectCenteredView);
}

bool ecvDisplayTools::perspectiveView() const {
    return ecvViewManager::instance()
            .resolveViewContext()
            .viewportParams.perspectiveView;
}

bool ecvDisplayTools::objectCenteredView() const {
    return ecvViewManager::instance()
            .resolveViewContext()
            .viewportParams.objectCenteredView;
}

void ecvDisplayTools::setSceneDB(ccHObject* root) { SetSceneDB(root); }

ccHObject* ecvDisplayTools::getSceneDB() {
    return ecvViewManager::instance().globalDBRoot();
}

ccHObject* ecvDisplayTools::getOwnDB() { return m_winDBRoot; }

void ecvDisplayTools::addToOwnDB(ccHObject* obj, bool noDependency) {
    AddToOwnDB(obj, noDependency);
}

void ecvDisplayTools::removeFromOwnDB(ccHObject* obj) { RemoveFromOwnDB(obj); }

void ecvDisplayTools::updateConstellationCenterAndZoom(const ccBBox* aBox) {
    UpdateConstellationCenterAndZoom(aBox);
}

QWidget* ecvDisplayTools::asWidget() { return m_mainScreen; }

const QWidget* ecvDisplayTools::asWidget() const { return m_mainScreen; }

bool ecvDisplayTools::hasOverriddenDisplayParameters() const {
    return ecvViewManager::instance().hasOverriddenDisplayParameters();
}

// Phase 7a wave 2 virtual overrides
CCVector3d ecvDisplayTools::toVtkCoordinates(int x, int y, int z) {
    return ToVtkCoordinates(x, y, z);
}

bool ecvDisplayTools::getClick3DPos(int x, int y, CCVector3d& pos) {
    return GetClick3DPos(x, y, pos);
}

void ecvDisplayTools::setView(CC_VIEW_ORIENTATION orientation) {
    SetView(orientation, nullptr);
}

CCVector3d ecvDisplayTools::getCurrentViewDir() const {
    return GetCurrentViewDir();
}

void ecvDisplayTools::setPivotPoint(const CCVector3d& P,
                                    bool autoRedraw,
                                    bool verbose) {
    SetPivotPoint(P, autoRedraw, verbose);
}

void ecvDisplayTools::setPivotVisibility(PivotVisibility vis) {
    SetPivotVisibility(vis);
}

void ecvDisplayTools::setAutoPickPivotAtCenter(bool state) {
    SetAutoPickPivotAtCenter(state);
}

bool ecvDisplayTools::isRotationAxisLocked() const {
    return IsRotationAxisLocked();
}

void ecvDisplayTools::lockRotationAxis(bool state, const CCVector3d& axis) {
    LockRotationAxis(state, axis);
}

void ecvDisplayTools::toggleDebugTrace() { ToggleDebugTrace(); }

void ecvDisplayTools::update2DLabels(bool immediateUpdate /*= false*/) {
    Update2DLabel(immediateUpdate);
}

bool ecvDisplayTools::renderToFile(const QString& filename,
                                   float zoomFactor,
                                   bool dontScale) {
    return RenderToFile(filename, zoomFactor, dontScale);
}

void ecvDisplayTools::removeBB(const QString& viewId) { RemoveBB(viewId); }

void ecvDisplayTools::removeBB(const ccGLDrawContext& context) {
    RemoveBB(context);
}

void ecvDisplayTools::setExclusiveFullScreenFlag(bool state) {
    auto* view = ecvViewManager::instance().getEffectiveView();
    SetExclusiveFullScreenFlag(state, view);
}

void ecvDisplayTools::filterByEntityType(std::vector<ccHObject*>& entities,
                                         CV_CLASS_ENUM type) {
    FilterByEntityType(entities, type);
}

void ecvDisplayTools::updateActiveItemsList(int x, int y, bool centerItems) {
    UpdateActiveItemsList(x, y, centerItems);
}

double ecvDisplayTools::computeActualPixelSize() const {
    return ComputeActualPixelSize();
}

void ecvDisplayTools::updateNamePoseRecursive() { UpdateNamePoseRecursive(); }

void ecvDisplayTools::showPivotSymbol(bool state) { ShowPivotSymbol(state); }

bool ecvDisplayTools::exclusiveFullScreen() const {
    auto* view = ecvViewManager::instance().getEffectiveView();
    return ExclusiveFullScreen(view);
}

CCVector3d ecvDisplayTools::convertMousePositionToOrientation(int x, int y) {
    return ConvertMousePositionToOrientation(x, y);
}

bool ecvDisplayTools::processClickableItems(int x, int y) {
    return ProcessClickableItems(x, y);
}

void ecvDisplayTools::updateZoom(float zoomFactor) { UpdateZoom(zoomFactor); }

void ecvDisplayTools::resizeGL(int w, int h) { ResizeGL(w, h); }

void ecvDisplayTools::setViewportDefaultPointSize(float size) {
    SetViewportDefaultPointSize(size);
}

void ecvDisplayTools::setViewportDefaultLineWidth(float width) {
    SetViewportDefaultLineWidth(width);
}

void ecvDisplayTools::setZNearCoef(double coef) { SetZNearCoef(coef); }

void ecvDisplayTools::setFov(float fov_deg) { SetFov(fov_deg); }

void ecvDisplayTools::setPointSizeOnView(float size) {
    SetPointSize(ecvViewManager::instance().resolveViewContext(), size);
}

void ecvDisplayTools::rotateWithAxis(const CCVector2i& mousePos,
                                     const CCVector3d& axis,
                                     double angle_deg) {
    RotateWithAxis(mousePos, axis, angle_deg, 0);
}

void ecvDisplayTools::startPicking(
        PICKING_MODE mode, int x, int y, int w, int h) {
    PickingParameters params(mode, x, y, w, h);
    StartPicking(params);
}

void ecvDisplayTools::redraw2DLabel() { Redraw2DLabel(); }
