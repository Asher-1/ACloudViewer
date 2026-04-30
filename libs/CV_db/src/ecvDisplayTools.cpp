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

// Shared display-tools instance.
// Ownership: created by ecvViewManager::initDisplayTools(),
//            destroyed by ecvViewManager::releaseDisplayTools().
static ecvDisplayTools* s_tools = nullptr;

bool ecvDisplayTools::USE_2D = true;
bool ecvDisplayTools::USE_VTK_PICK = false;

/// Returns a non-primary active view (ecvGenericGLDisplay*) if one exists
/// and it differs from the singleton.  Used by static methods to delegate
/// to the per-view implementation.
static ecvGenericGLDisplay* activeSecondaryView() {
    auto* av = ecvViewManager::instance().getEffectiveView();
    if (av && av != s_tools) return av;
    return nullptr;
}

static const QString DEBUG_LAYER_ID = "DEBUG_LAYER";

static int viewportHeightFor(ecvGenericGLDisplay* display) {
    if (display && display != s_tools) {
        auto* ctx = display->viewContext();
        if (ctx) return ctx->glViewport.height();
    }
    return s_tools->effectiveCtx().glViewport.height();
}

// ================================================================
// Per-view context  (Phase A → Phase E)
// ================================================================

ecvViewContext& ecvDisplayTools::effectiveCtx() {
    auto* eff = ecvViewManager::instance().getEffectiveView();
    if (eff && eff->viewContext()) return *eff->viewContext();
    return m_primaryCtx;
}

const ecvViewContext& ecvDisplayTools::effectiveCtx() const {
    auto* eff = ecvViewManager::instance().getEffectiveView();
    if (eff && eff->viewContext()) return *eff->viewContext();
    return m_primaryCtx;
}

void ecvDisplayTools::copyContextFrom(const ecvViewContext& ctx) {
    m_primaryCtx = ctx;
}

ecvViewContext ecvDisplayTools::snapshotContext() const { return m_primaryCtx; }

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

void ecvDisplayTools::initializeSharedInstance(ecvDisplayTools* displayTools,
                                               QMainWindow* win,
                                               bool stereoMode) {
    if (s_tools) {
        assert(false && "Display tools already initialized");
        return;
    }

    s_tools = displayTools;
    ecvGenericDisplayTools::SetInstance(s_tools);

    // start internal timer
    s_tools->m_timer.start();

    SetMainWindow(win);
    // register current instance visualizer only once
    s_tools->registerVisualizer(win, stereoMode);

    s_tools->m_uniqueID = ++s_GlWindowNumber;  // GL window unique ID
    s_tools->effectiveCtx().lastMousePos = QPoint(-1, -1);
    s_tools->effectiveCtx().lastMouseMovePos = QPoint(-1, -1);
    s_tools->effectiveCtx().validModelviewMatrix = false;
    s_tools->effectiveCtx().validProjectionMatrix = false;
    s_tools->effectiveCtx().cameraToBBCenterDist = 0.0;
    s_tools->m_shouldBeRefreshed = false;
    s_tools->effectiveCtx().mouseMoved = false;
    s_tools->effectiveCtx().mouseButtonPressed = false;
    s_tools->effectiveCtx().widgetClicked = false;

    s_tools->effectiveCtx().bbHalfDiag = 0.0;
    s_tools->effectiveCtx().interactionFlags = TRANSFORM_CAMERA();
    s_tools->effectiveCtx().pickingMode = NO_PICKING;
    s_tools->effectiveCtx().pickingModeLocked = false;
    s_tools->effectiveCtx().lastClickTime_ticks = 0;

    s_tools->effectiveCtx().sunLightEnabled = true;
    s_tools->effectiveCtx().customLightEnabled = false;
    s_tools->effectiveCtx().clickableItemsVisible = false;
    s_tools->m_alwaysUseFBO = false;
    s_tools->m_updateFBO = true;
    s_tools->m_winDBRoot = nullptr;
    s_tools->m_globalDBRoot = nullptr;  // external DB
    s_tools->m_removeFlag = false;
    s_tools->m_font = QFont();
    s_tools->effectiveCtx().pivotVisibility = PIVOT_SHOW_ON_MOVE;
    s_tools->effectiveCtx().pivotSymbolShown = false;
    s_tools->effectiveCtx().allowRectangularEntityPicking = false;
    s_tools->m_rectPickingPoly = nullptr;
    s_tools->m_overridenDisplayParametersEnabled = false;
    s_tools->effectiveCtx().displayOverlayEntities = true;
    s_tools->effectiveCtx().bubbleViewModeEnabled = false;
    s_tools->effectiveCtx().bubbleViewFov_deg = 90.0f;
    s_tools->effectiveCtx().touchInProgress = false;
    s_tools->effectiveCtx().touchBaseDist = 0.0;
    s_tools->m_scheduledFullRedrawTime = 0;
    s_tools->effectiveCtx().exclusiveFullscreen = false;
    s_tools->effectiveCtx().showDebugTraces = false;
    s_tools->effectiveCtx().pickRadius = DefaultPickRadius;
    s_tools->m_autoRefresh = false;
    s_tools->m_hotZone = nullptr;
    s_tools->effectiveCtx().showCursorCoordinates = false;
    s_tools->effectiveCtx().autoPickPivotAtCenter = false;
    s_tools->effectiveCtx().ignoreMouseReleaseEvent = false;
    s_tools->effectiveCtx().rotationAxisLocked = false;
    s_tools->effectiveCtx().lockedRotationAxis = CCVector3d(0, 0, 1);

    // GL window own DB
    s_tools->m_winDBRoot =
            new ccHObject(QString("DB.3DView_%1").arg(s_tools->m_uniqueID));

    // matrices
    s_tools->effectiveCtx().viewportParams.viewMat.toIdentity();
    s_tools->effectiveCtx().viewportParams.setCameraCenter(CCVector3d(
            0.0, 0.0,
            1.0));  // don't position the camera on the pivot by default!
    s_tools->effectiveCtx().viewMatd.toIdentity();
    s_tools->effectiveCtx().projMatd.toIdentity();

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

    s_tools->m_deferredPickingTimer.setSingleShot(true);
    s_tools->m_deferredPickingTimer.setInterval(100);

    // signal/slot connections
    connect(s_tools, &ecvDisplayTools::itemPickedFast, s_tools,
            &ecvDisplayTools::onItemPickedFast, Qt::DirectConnection);
    connect(GetVisualizer3D(),
            &ecvGenericVisualizer3D::interactorPointPickedEvent, s_tools,
            &ecvDisplayTools::onPointPicking);

    connect(&s_tools->m_scheduleTimer, &QTimer::timeout, s_tools,
            &ecvDisplayTools::checkScheduledRedraw);
    connect(&s_tools->m_deferredPickingTimer, &QTimer::timeout, s_tools,
            &ecvDisplayTools::doPicking);

    // Register with the global view manager as the primary view
    ecvViewManager::instance().registerView(s_tools);
    ecvViewManager::instance().setActiveView(s_tools);
}

ecvDisplayTools* ecvDisplayTools::sharedTools() { return s_tools; }

void ecvDisplayTools::releaseSharedInstance() {
    if (s_tools) {
        ecvGenericDisplayTools::SetInstance(nullptr);
        delete s_tools;
        s_tools = nullptr;
    }
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
    if (m_hotZone && m_hotZoneOwnedBySingleton) {
        delete m_hotZone;
    }
    m_hotZone = nullptr;
    m_hotZoneOwnedBySingleton = false;
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
    // Sync VTK pick results into the effective view context so that
    // doPicking → StartOpenGLPicking / StartCPUBasedPointPicking can
    // read them from effectiveCtx() (Phase-A bridging).
    ecvViewContext& ctx = effectiveCtx();
    ctx.lastPickedPoint = p;
    ctx.lastPointIndex = index;
    ctx.lastPickedId = id.c_str();

#ifdef QT_DEBUG
    CVLog::Print(QString("current selected index is %1").arg(index));
    CVLog::Print(
            QString("current selected entity id is %1").arg(ctx.lastPickedId));
    CVLog::Print(QString("current selected point coord is [%1, %2, %3]")
                         .arg(p.x)
                         .arg(p.y)
                         .arg(p.z));
#endif  // !QDEBUG

    if (ctx.lastPickedId.isEmpty()) {
        PICKING_MODE pickingMode = PICKING_MODE::ENTITY_PICKING;
        PickingParameters params(pickingMode, 0, 0, ctx.pickRadius,
                                 ctx.pickRadius);
        ProcessPickingResult(params, nullptr, -1);
    } else {
        if (ecvDisplayTools::USE_VTK_PICK) {
            doPicking();
        }
    }
}

void ecvDisplayTools::doPicking() {
    ecvViewContext* ctx =
            m_pickingTargetView ? m_pickingTargetView->viewContext() : nullptr;

    bool widgetClicked = ctx ? ctx->widgetClicked : m_primaryCtx.widgetClicked;
    if (widgetClicked) {
        CVLog::PrintVerbose(
                "[ecvDisplayTools::doPicking] Skipping picking because "
                "VTK widget was clicked");
        if (ctx)
            ctx->widgetClicked = false;
        else
            m_primaryCtx.widgetClicked = false;
        m_pickingTargetView = nullptr;
        return;
    }

    const QPoint& mousePos =
            ctx ? ctx->lastMousePos : m_primaryCtx.lastMousePos;
    int x = mousePos.x();
    int y = mousePos.y();

    if (x < 0 || y < 0) {
        assert(false);
        m_pickingTargetView = nullptr;
        return;
    }

    PICKING_MODE pickMode = ctx ? ctx->pickingMode : m_primaryCtx.pickingMode;
    INTERACTION_FLAGS iFlags =
            ctx ? ctx->interactionFlags : m_primaryCtx.interactionFlags;
    int pickRad = ctx ? ctx->pickRadius : m_primaryCtx.pickRadius;

    if ((pickMode != NO_PICKING) || (iFlags & INTERACT_2D_ITEMS)) {
        if (iFlags & INTERACT_2D_ITEMS) {
            UpdateActiveItemsList(x, y, false);
            if (!m_activeItems.empty() && m_activeItems.size() == 1) {
                ccInteractor* pickedObj = m_activeItems.front();
                cc2DLabel* label = dynamic_cast<cc2DLabel*>(pickedObj);
                if (label && !label->isSelected()) {
                    emit s_tools->entitySelectionChanged(label);
                    QApplication::processEvents();
                }
            }
        } else {
            assert(m_activeItems.empty());
        }

        if (m_activeItems.empty() && pickMode != NO_PICKING) {
            PICKING_MODE effectiveMode = pickMode;

            if (effectiveMode == ENTITY_PICKING &&
                (QApplication::keyboardModifiers() & Qt::AltModifier)) {
                effectiveMode = LABEL_PICKING;
            } else if (effectiveMode == ENTITY_PICKING &&
                       (QApplication::keyboardModifiers() &
                        Qt::ControlModifier)) {
                effectiveMode = POINT_OR_TRIANGLE_PICKING;
            }

            PickingParameters params(effectiveMode, x, y, pickRad, pickRad);
            StartPicking(params);
        }
    }

    m_pickingTargetView = nullptr;
}

void ecvDisplayTools::onWheelEvent(float wheelDelta_deg) {
    // Phase M2.3: use effectiveCtx() for multi-view correctness instead
    // of m_primaryCtx (which ignores secondary views).
    const auto& ctx = effectiveCtx();

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

bool ecvDisplayTools::ProcessClickableItems(int x, int y) {
    if (s_tools->m_clickableItems.empty()) {
        return false;
    }

    // correction for HD screens
    const int retinaScale = GetDevicePixelRatio();
    x *= retinaScale;
    y *= retinaScale;

    ClickableItem::Role clickedItem = ClickableItem::NO_ROLE;
    for (std::vector<ClickableItem>::const_iterator it =
                 s_tools->m_clickableItems.begin();
         it != s_tools->m_clickableItems.end(); ++it) {
        if (it->area.contains(x, y)) {
            clickedItem = it->role;
            break;
        }
    }

    switch (clickedItem) {
        case ClickableItem::NO_ROLE: {
            // nothing to do
        } break;

        case ClickableItem::INCREASE_POINT_SIZE: {
            SetPointSize(
                    s_tools->effectiveCtx().viewportParams.defaultPointSize +
                    1.0f);
        }
            return true;

        case ClickableItem::DECREASE_POINT_SIZE: {
            SetPointSize(
                    s_tools->effectiveCtx().viewportParams.defaultPointSize -
                    1.0f);
        }
            return true;

        case ClickableItem::INCREASE_LINE_WIDTH: {
            SetLineWidth(
                    s_tools->effectiveCtx().viewportParams.defaultLineWidth +
                    1.0f);
        }
            return true;

        case ClickableItem::DECREASE_LINE_WIDTH: {
            SetLineWidth(
                    s_tools->effectiveCtx().viewportParams.defaultLineWidth -
                    1.0f);
        }
            return true;

        case ClickableItem::LEAVE_BUBBLE_VIEW_MODE: {
            SetBubbleViewMode(false);
            RedrawDisplay();
        }
            return true;

        case ClickableItem::LEAVE_FULLSCREEN_MODE: {
            if (s_tools->m_win) {
                emit s_tools->exclusiveFullScreenToggled(false);
            }
        }
            return true;

        default: {
            // unhandled item?!
            assert(false);
        } break;
    }

    return false;
}

void ecvDisplayTools::SetPointSize(float size, bool silent, int viewport) {
    float newSize =
            std::max(std::min(size, MAX_POINT_SIZE_F), MIN_POINT_SIZE_F);
    if (!silent) {
        CVLog::Print(QString("New point size: %1").arg(newSize));
    }

    // Write-through: update both singleton and active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        ecvViewportParameters vp = av->getViewportParameters();
        vp.defaultPointSize = newSize;
        av->setViewportParameters(vp);
    }

    if (s_tools->effectiveCtx().viewportParams.defaultPointSize != newSize) {
        s_tools->effectiveCtx().viewportParams.defaultPointSize = newSize;

        if (!silent) {
            ecvDisplayTools::DisplayNewMessage(
                    QString("New default point size: %1").arg(newSize),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 2,
                    SCREEN_SIZE_MESSAGE);
        }
    }
}

void ecvDisplayTools::SetPointSizeRecursive(int size) {
    // we draw 3D entities
    if (s_tools->m_globalDBRoot) {
        s_tools->m_globalDBRoot->setPointSizeRecursive(size);
    }

    if (s_tools->m_winDBRoot) {
        s_tools->m_winDBRoot->setPointSizeRecursive(size);
    }
}

void ecvDisplayTools::SetLineWidth(float width, bool silent, int viewport) {
    float newWidth =
            std::max(std::min(width, MAX_LINE_WIDTH_F), MIN_LINE_WIDTH_F);
    if (!silent) {
        CVLog::Print(QString("New line with: %1").arg(newWidth));
    }

    // Write-through: update both singleton and active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        ecvViewportParameters vp = av->getViewportParameters();
        vp.defaultLineWidth = newWidth;
        av->setViewportParameters(vp);
    }

    if (s_tools->effectiveCtx().viewportParams.defaultLineWidth != newWidth) {
        s_tools->effectiveCtx().viewportParams.defaultLineWidth = newWidth;
        if (!silent) {
            ecvDisplayTools::DisplayNewMessage(
                    QString("New default line width: %1").arg(newWidth),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 2,
                    SCREEN_SIZE_MESSAGE);
        }
    }
}

void ecvDisplayTools::SetLineWithRecursive(PointCoordinateType with) {
    // we draw 3D entities
    if (s_tools->m_globalDBRoot) {
        s_tools->m_globalDBRoot->setLineWidthRecursive(with);
    }

    if (s_tools->m_winDBRoot) {
        s_tools->m_winDBRoot->setLineWidthRecursive(with);
    }
}

void ecvDisplayTools::SetViewportDefaultPointSize(float size) {
    s_tools->effectiveCtx().viewportParams.defaultPointSize = size;
}

void ecvDisplayTools::SetViewportDefaultLineWidth(float width) {
    s_tools->effectiveCtx().viewportParams.defaultLineWidth = width;
}

void ecvDisplayTools::StartPicking(PickingParameters& params) {
    // correction for HD screens
    const int retinaScale = GetDevicePixelRatio();
    params.centerX *= retinaScale;
    params.centerY *= retinaScale;

    if (!s_tools->m_globalDBRoot && !s_tools->m_winDBRoot) {
        // we must always emit a signal!
        ProcessPickingResult(params, nullptr, -1);
        return;
    }

    if (params.mode == POINT_OR_TRIANGLE_PICKING ||
        params.mode == POINT_PICKING || params.mode == TRIANGLE_PICKING ||
        params.mode == LABEL_PICKING  // = spawn a label on the clicked point or
                                      // triangle
    ) {
        // CPU-based point picking
        StartCPUBasedPointPicking(params);
    } else {
        StartOpenGLPicking(params);
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
        emit s_tools->entitySelectionChanged(pickedEntity);
    }
    // rectangular "entity" picking
    else if (params.mode == ENTITY_RECT_PICKING) {
        if (selectedIDs)
            emit s_tools->entitiesSelectionChanged(*selectedIDs);
        else
            assert(false);
    }
    // 3D point or triangle picking
    else if (params.mode == POINT_PICKING || params.mode == TRIANGLE_PICKING ||
             params.mode == POINT_OR_TRIANGLE_PICKING) {
        assert(pickedEntity == nullptr || pickedItemIndex >= 0);
        assert(nearestPoint);

        emit s_tools->itemPicked(pickedEntity,
                                 static_cast<unsigned>(pickedItemIndex),
                                 params.centerX, params.centerY, *nearestPoint);
    }
    // fast picking (labels, interactors, etc.)
    else if (params.mode == FAST_PICKING) {
        emit s_tools->itemPickedFast(pickedEntity, pickedItemIndex,
                                     params.centerX, params.centerY);
    } else if (params.mode == LABEL_PICKING) {
        if (s_tools->m_globalDBRoot && pickedEntity && pickedItemIndex >= 0) {
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
                auto* parentDisplay = pickedEntity->getDisplay();
                label->setDisplay(parentDisplay ? parentDisplay : s_tools);
                label->setPosition(
                        static_cast<float>(params.centerX + 20) /
                                s_tools->effectiveCtx().glViewport.width(),
                        static_cast<float>(params.centerY + 20) /
                                s_tools->effectiveCtx().glViewport.height());
                emit s_tools->newLabel(static_cast<ccHObject*>(label));
                QApplication::processEvents();
            }
        }
    }
}

void ecvDisplayTools::SetZNearCoef(double coef) {
    if (coef <= 0.0 || coef >= 1.0) {
        CVLog::Warning("[ecvDisplayTools::setZNearCoef] Invalid coef. value!");
        return;
    }

    if (s_tools->effectiveCtx().viewportParams.zNearCoef != coef) {
        // update param
        s_tools->effectiveCtx().viewportParams.zNearCoef = coef;
        // and camera state (if perspective view is 'on')
        if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
            // DGM: we update the projection matrix directly so as to get an
            // up-to-date estimation of zNear
            UpdateProjectionMatrix();

            SetCameraClip(s_tools->effectiveCtx().viewportParams.zNear,
                          s_tools->effectiveCtx().viewportParams.zFar);

            Deprecate3DLayer();

            DisplayNewMessage(
                    QString("Near clipping = %1% of max depth (= %2)")
                            .arg(s_tools->effectiveCtx()
                                                 .viewportParams.zNearCoef *
                                         100.0,
                                 0, 'f', 1)
                            .arg(s_tools->effectiveCtx().viewportParams.zNear),
                    ecvDisplayTools::LOWER_LEFT_MESSAGE,  // DGM HACK: we cheat
                                                          // and use the same
                                                          // 'slot' as the
                                                          // window size
                    false, 2, SCREEN_SIZE_MESSAGE);
        }

        emit s_tools->zNearCoefChanged(coef);
        emit s_tools->cameraParamChanged();
    }
}

// DGM: WARNING: OpenGL picking with the picking buffer is depreacted.
// We need to get rid of this code or change it to color-based selection...
void ecvDisplayTools::StartOpenGLPicking(const PickingParameters& params) {
    if (!params.pickInLocalDB && !params.pickInSceneDB) {
        assert(false);
        return;
    }

    // setup rendering context
    unsigned short flags = CC_DRAW_FOREGROUND;

    switch (params.mode) {
        case FAST_PICKING:
            flags |= CC_FAST_ENTITY_PICKING;
        case ENTITY_PICKING:
        case ENTITY_RECT_PICKING:
            flags |= CC_ENTITY_PICKING;
            break;
        default:
            // unhandled mode?!
            assert(false);
            // we must always emit a signal!
            ProcessPickingResult(params, nullptr, -1);
            return;
    }

    // OpenGL picking
    assert(!s_tools->m_captureMode.enabled);

    // process hits
    std::unordered_set<int> selectedIDs;
    int pickedItemIndex = -1;
    int selectedID = -1;
    ccHObject* pickedEntity = nullptr;

    CCVector3 P(0, 0, 0);
    CCVector3* pickedPoint = nullptr;

    if (s_tools->effectiveCtx().lastPointIndex >= 0) {
        pickedEntity = GetPickedEntity(params);
        if (pickedEntity) {
            selectedID = pickedEntity->getUniqueID();
            selectedIDs.insert(selectedID);
            pickedItemIndex = s_tools->effectiveCtx().lastPointIndex;
        }
    }

    if (pickedEntity && pickedItemIndex >= 0 &&
        pickedEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* tempEntity =
                ccHObjectCaster::ToGenericPointCloud(pickedEntity);
        int pNum = static_cast<int>(tempEntity->size());
        if (pickedItemIndex >= pNum) {
            P = s_tools->effectiveCtx().lastPickedPoint;
            CVLog::Warning(
                    QString("[ecvDisplayTools::StartOpenGLPicking] Picking "
                            "Error, %1 is more than picked entity size %2")
                            .arg(pickedItemIndex)
                            .arg(tempEntity->size()));
            pickedItemIndex = pNum - 1;
        } else {
            P = *(static_cast<ccGenericPointCloud*>(pickedEntity)
                          ->getPoint(pickedItemIndex));
            // check selected point
            CCVector3 temp = P - s_tools->effectiveCtx().lastPickedPoint;
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

    // we must always emit a signal!
    ProcessPickingResult(params, pickedEntity, pickedItemIndex, pickedPoint,
                         &selectedIDs);
}

void ecvDisplayTools::StartCPUBasedPointPicking(
        const PickingParameters& params) {
    // qint64 t0 = m_timer.elapsed();

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
    GetGLCameraParameters(camera);

    // Compute clickedPos AFTER GetGLCameraParameters so m_glViewport
    // reflects the active view's actual dimensions (critical for
    // multi-window picking where views may differ in size).
    CCVector2d clickedPos(
            params.centerX,
            s_tools->effectiveCtx().glViewport.height() - 1 - params.centerY);

    if (ecvDisplayTools::USE_VTK_PICK) {
        int pickedIndex = -1;
        ccHObject* pickedEntity = nullptr;
        if (s_tools->effectiveCtx().lastPointIndex >= 0) {
            pickedIndex = s_tools->effectiveCtx().lastPointIndex;
            pickedEntity = GetPickedEntity(params);
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
                    nearestPoint = s_tools->effectiveCtx().lastPickedPoint;
                } else {
                    nearestElementIndex = -1;
                }
            } else {
                nearestElementIndex = pickedIndex;
                if (pickedIndex >= pNum) {
                    nearestPoint = s_tools->effectiveCtx().lastPickedPoint;
                    CVLog::Warning(QString("[ecvDisplayTools::"
                                           "StartCPUBasedPointPicking] "
                                           "Picking Error, %1 is more than "
                                           "picked entity size %2")
                                           .arg(pickedIndex)
                                           .arg(tempEntity->size()));
                    nearestElementIndex = pNum - 1;
                } else {
                    nearestPoint = *(tempEntity->getPoint(pickedIndex));
                    CCVector3 temp = nearestPoint -
                                     s_tools->effectiveCtx().lastPickedPoint;
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
            if (s_tools->m_globalDBRoot)
                toProcess.push_back(s_tools->m_globalDBRoot);
            if (s_tools->m_winDBRoot) toProcess.push_back(s_tools->m_winDBRoot);

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
    // qint64 dt = m_timer.elapsed() - t0;
    // CVLog::Print(QString("[Picking][CPU] Time: %1 ms").arg(dt));

    if (!ecvDisplayTools::USE_VTK_PICK) {
        s_tools->effectiveCtx().lastPointIndex = nearestElementIndex;
        s_tools->effectiveCtx().lastPickedPoint = nearestPoint;
        if (nearestEntity) {
            s_tools->effectiveCtx().lastPickedId = nearestEntity->getViewId();
        }
    }

    // we must always emit a signal!
    ProcessPickingResult(params, nearestEntity, nearestElementIndex,
                         &nearestPoint);
}

ccHObject* ecvDisplayTools::GetPickedEntity(const PickingParameters& params) {
    if (s_tools->effectiveCtx().lastPickedId.isEmpty()) return nullptr;

    ccHObject* pickedEntity = nullptr;
    unsigned int selectedID = s_tools->effectiveCtx().lastPickedId.toUInt();
    if (params.pickInSceneDB && s_tools->m_globalDBRoot) {
        pickedEntity = s_tools->m_globalDBRoot->find(selectedID);
    }
    if (!pickedEntity && params.pickInLocalDB && s_tools->m_winDBRoot) {
        pickedEntity = s_tools->m_winDBRoot->find(selectedID);
    }

    return pickedEntity;
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

void ecvDisplayTools::SetPivotVisibility(PivotVisibility vis) {
    s_tools->effectiveCtx().pivotVisibility = vis;

    if (vis == PivotVisibility::PIVOT_HIDE) {
        SetPivotVisibility(false);
    } else {
        SetPivotVisibility(true);
    }

    UpdateScreen();

    // auto-save last pivot visibility settings
    {
        QSettings settings;
        settings.beginGroup(c_ps_groupName);
        settings.setValue(c_ps_pivotVisibility, vis);
        settings.endGroup();
    }
}

void ecvDisplayTools::ResizeGL(int w, int h) {
    // update OpenGL viewport
    SetGLViewport(0, 0, w, h);

    InvalidateVisualization();
    Deprecate3DLayer();

    if (s_tools->m_hotZone) {
        s_tools->m_hotZone->topCorner = QPoint(0, 0);
    }

    DisplayNewMessage(QString("New size = %1 * %2 (px)")
                              .arg(s_tools->effectiveCtx().glViewport.width())
                              .arg(s_tools->effectiveCtx().glViewport.height()),
                      LOWER_LEFT_MESSAGE, false, 2, SCREEN_SIZE_MESSAGE);
}

void ecvDisplayTools::MoveCamera(float dx, float dy, float dz) {
    if (dx != 0.0f || dy != 0.0f)  // camera movement? (dz doesn't count as it
                                   // only corresponds to a zoom)
    {
        // feedback for echo mode
        emit s_tools->cameraDisplaced(dx, dy);
    }

    // current X, Y and Z viewing directions
    // correspond to the 'model view' matrix
    // lines.
    CCVector3d V(dx, dy, dz);
    if (!s_tools->effectiveCtx().viewportParams.objectCenteredView) {
        s_tools->effectiveCtx()
                .viewportParams.viewMat.transposed()
                .applyRotation(V);
    }

    SetCameraPos(s_tools->effectiveCtx().viewportParams.getCameraCenter() + V);
}

void ecvDisplayTools::UpdateActiveItemsList(
        int x, int y, bool extendToSelectedLabels /*=false*/) {
    // Route to the effective view's active-items list (per-window isolation).
    auto* effView = ecvViewManager::instance().getEffectiveView();
    std::list<ccInteractor*>& items =
            effView ? effView->activeItemsRef() : s_tools->m_activeItems;
    items.clear();

    {
        ccHObject::Container labels;
        FilterByEntityType(labels, CV_TYPES::LABEL_2D);
        CVLog::Print("[Label] UpdateActiveItemsList: mouse(%d,%d) labels=%zu",
                     x, y, labels.size());
        for (auto* obj : labels) {
            if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
                !obj->isVisible())
                continue;
            cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
            if (!l) continue;
            QRect roi = l->getLabelROI();
            CVLog::Print(
                    "[Label]   '%s' ROI=(%d,%d %dx%d) valid=%d contains=%d "
                    "dispIn2D=%d dispLegend=%d",
                    qPrintable(l->getName()), roi.x(), roi.y(), roi.width(),
                    roi.height(), roi.isValid(), roi.contains(x, y),
                    l->isDisplayedIn2D(), l->isPointLegendDisplayed());
            if (roi.isValid() && roi.contains(x, y)) {
                items.push_back(l);
                CVLog::Print("[Label]   >>> HIT! Added to active items");
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
                if (s_tools->m_globalDBRoot)
                    s_tools->m_globalDBRoot->filterChildren(labels, true,
                                                            CV_TYPES::LABEL_2D);
                if (s_tools->m_winDBRoot)
                    s_tools->m_winDBRoot->filterChildren(labels, true,
                                                         CV_TYPES::LABEL_2D);

                for (auto& label : labels) {
                    if (label->isA(CV_TYPES::LABEL_2D) && label->isEnabled() &&
                        label->isVisible()) {
                        cc2DLabel* l = static_cast<cc2DLabel*>(label);
                        if (l != label && l->isSelected()) {
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
    if (pickedEntity) {
        if (pickedEntity->isA(CV_TYPES::LABEL_2D)) {
            cc2DLabel* label = static_cast<cc2DLabel*>(pickedEntity);
            m_activeItems.push_back(label);
        } else if (pickedEntity->isA(CV_TYPES::CLIPPING_BOX)) {
            ccClipBox* cbox = static_cast<ccClipBox*>(pickedEntity);
            cbox->setActiveComponent(pickedItemIndex);
            cbox->setClickedPoint(x, y, Width(), Height(),
                                  m_primaryCtx.viewportParams.viewMat);

            m_activeItems.push_back(cbox);
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

CCVector3d ecvDisplayTools::ConvertMousePositionToOrientation(int x, int y) {
    double xc = static_cast<double>(Width() / 2);
    double yc = static_cast<double>(
            Height() / 2);  // DGM FIME: is it scaled coordinates or not?!

    CCVector3d Q2D;
    if (s_tools->effectiveCtx().viewportParams.objectCenteredView) {
        // project the current pivot point on screen
        ccGLCameraParameters camera;
        GetGLCameraParameters(camera);

        if (!camera.project(
                    s_tools->effectiveCtx().viewportParams.getPivotPoint(),
                    Q2D)) {
            // arbitrary direction
            return CCVector3d(0, 0, 1);
        }

        // we set the virtual rotation pivot closer to the actual one (but we
        // always stay in the central part of the screen!)
        Q2D.x = std::min(Q2D.x, 3.0 * Width() / 4.0);
        Q2D.x = std::max(Q2D.x, Width() / 4.0);

        Q2D.y = std::min(Q2D.y, 3.0 * Height() / 4.0);
        Q2D.y = std::max(Q2D.y, Height() / 4.0);
    } else {
        Q2D.x = xc;
        Q2D.y = yc;
    }

    // invert y
    y = Height() - 1 - y;

    CCVector3d v(x - Q2D.x, y - Q2D.y, 0.0);

    v.x = std::max(std::min(v.x / xc, 1.0), -1.0);
    v.y = std::max(std::min(v.y / yc, 1.0), -1.0);

    // square 'radius'
    double d2 = v.x * v.x + v.y * v.y;

    // projection on the unit sphere
    if (d2 > 1) {
        double d = std::sqrt(d2);
        v.x /= d;
        v.y /= d;
    } else {
        v.z = std::sqrt(1.0 - d2);
    }

    return v;
}

void ecvDisplayTools::RotateBaseViewMat(const ccGLMatrixd& rotMat) {
    ecvViewportParameters viewParams = ecvDisplayTools::GetViewportParameters();
    viewParams.viewMat = rotMat * viewParams.viewMat;

    // pos
    CCVector3d camC = viewParams.viewMat.getTranslationAsVec3D();
    viewParams.setCameraCenter(camC);

    // up
    CCVector3d upDir = viewParams.viewMat.getColumnAsVec3D(1);
    upDir.normalize();
    viewParams.up = upDir;

    // focal
    CCVector3d viewDir = viewParams.viewMat.getColumnAsVec3D(2);
    viewParams.focal = camC - viewDir;
    viewParams.setPivotPoint(viewParams.focal, true);

    ecvDisplayTools::SetViewportParameters(viewParams);

    // we emit the 'baseViewMatChanged' signal
    emit s_tools->baseViewMatChanged(
            s_tools->effectiveCtx().viewportParams.viewMat);
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

void ecvDisplayTools::SetView(CC_VIEW_ORIENTATION orientation,
                              bool forceRedraw /*=false*/) {
    // may be useless
    bool wasViewerBased =
            !s_tools->effectiveCtx().viewportParams.objectCenteredView;
    if (wasViewerBased) {
        SetPerspectiveState(
                s_tools->effectiveCtx().viewportParams.perspectiveView, true);
    }
    s_tools->effectiveCtx().viewportParams.viewMat =
            GenerateViewMat(orientation);
    if (wasViewerBased) {
        SetPerspectiveState(
                s_tools->effectiveCtx().viewportParams.perspectiveView, false);
    }

    emit s_tools->baseViewMatChanged(
            s_tools->effectiveCtx().viewportParams.viewMat);
    emit s_tools->cameraParamChanged();
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
    // Update screen without changing zoom
    if (forceRedraw) {
        RedrawDisplay();
    } else {
        UpdateScreen();
    }
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

float ecvDisplayTools::ComputePerspectiveZoom() {
    // DGM: in fact it can be useful to compute it even in ortho mode :)
    // if (!m_viewportParams.perspectiveView)
    //	return m_viewportParams.zoom;

    // we compute the zoom equivalent to the corresponding camera position
    // (inverse of above calculus)
    float currentFov_deg = GetFov();
    if (currentFov_deg < FLT_EPSILON) return 1.0f;

    // Camera center to pivot vector
    double zoomEquivalentDist =
            (s_tools->effectiveCtx().viewportParams.getCameraCenter() -
             s_tools->effectiveCtx().viewportParams.getPivotPoint())
                    .norm();
    if (cloudViewer::LessThanEpsilon(zoomEquivalentDist)) return 1.0f;

    float screenSize = std::min(s_tools->effectiveCtx().glViewport.width(),
                                s_tools->effectiveCtx().glViewport.height()) *
                       s_tools->effectiveCtx()
                               .viewportParams
                               .pixelSize;  // see how pixelSize is computed!
    return screenSize /
           static_cast<float>(
                   zoomEquivalentDist *
                   std::tan(cloudViewer::DegreesToRadians(currentFov_deg)));
}

ccGLMatrixd& ecvDisplayTools::GetModelViewMatrix() {
    if (!s_tools->effectiveCtx().validModelviewMatrix) UpdateModelViewMatrix();

    return s_tools->effectiveCtx().viewMatd;
}

ccGLMatrixd& ecvDisplayTools::GetProjectionMatrix() {
    if (!s_tools->effectiveCtx().validProjectionMatrix)
        UpdateProjectionMatrix();

    return s_tools->effectiveCtx().projMatd;
}

ccGLMatrixd ecvDisplayTools::ComputeProjectionMatrix(
        bool withGLfeatures,
        ProjectionMetrics* metrics /*=nullptr*/,
        double* eyeOffset /*=nullptr*/) {
    double bbHalfDiag = 1.0;
    CCVector3d bbCenter(0, 0, 0);

    // compute center of visible objects constellation
    if (s_tools->m_globalDBRoot || s_tools->m_winDBRoot) {
        // get whole bounding-box
        ccBBox box;
        GetVisibleObjectsBB(box);
        if (box.isValid()) {
            // get bbox center
            bbCenter = CCVector3d::fromArray(box.getCenter().u);
            // get half bbox diagonal length
            bbHalfDiag = box.getDiagNormd() / 2;
        }
    }

    CCVector3d cameraCenterToBBCenter =
            s_tools->effectiveCtx().viewportParams.getCameraCenter() - bbCenter;
    double cameraToBBCenterDist = cameraCenterToBBCenter.normd();

    if (metrics) {
        metrics->bbHalfDiag = bbHalfDiag;
        metrics->cameraToBBCenterDist = cameraToBBCenterDist;
    }

    // virtual pivot point (i.e. to handle viewer-based mode smoothly)
    CCVector3d rotationCenter =
            s_tools->effectiveCtx().viewportParams.getRotationCenter();

    // compute the maximum distance between the pivot point and the farthest
    // displayed object
    double rotationCenterToFarthestObjectDist = 0.0;
    {
        // maximum distance between the pivot point and the farthest corner of
        // the displayed objects bounding-box
        rotationCenterToFarthestObjectDist =
                (bbCenter - rotationCenter).norm() + bbHalfDiag;

        //(if enabled) the pivot symbol should always be visible in
        // object-centere view mode
        if (s_tools->effectiveCtx().pivotSymbolShown &&
            s_tools->effectiveCtx().pivotVisibility != PIVOT_HIDE &&
            withGLfeatures &&
            s_tools->effectiveCtx().viewportParams.objectCenteredView) {
            double pivotActualRadius =
                    CC_DISPLAYED_PIVOT_RADIUS_PERCENT *
                    std::min(s_tools->effectiveCtx().glViewport.width(),
                             s_tools->effectiveCtx().glViewport.height()) /
                    2;
            double pivotSymbolScale =
                    pivotActualRadius * ComputeActualPixelSize();
            rotationCenterToFarthestObjectDist = std::max(
                    rotationCenterToFarthestObjectDist, pivotSymbolScale);
        }

        if (withGLfeatures && s_tools->effectiveCtx().customLightEnabled) {
            // distance from custom light to pivot point
            double distToCustomLight =
                    (rotationCenter -
                     CCVector3d::fromArray(
                             s_tools->effectiveCtx().customLightPos))
                            .norm();
            rotationCenterToFarthestObjectDist = std::max(
                    rotationCenterToFarthestObjectDist, distToCustomLight);
        }

        rotationCenterToFarthestObjectDist *= 1.01;  // for round-off issues
    }

    double cameraCenterToRotationCentertDist = 0;
    if (s_tools->effectiveCtx().viewportParams.objectCenteredView) {
        cameraCenterToRotationCentertDist =
                s_tools->effectiveCtx().viewportParams.getFocalDistance();
    }

    // we deduce zFar
    double zNear = cameraCenterToRotationCentertDist -
                   rotationCenterToFarthestObjectDist;
    double zFar = cameraCenterToRotationCentertDist +
                  rotationCenterToFarthestObjectDist;

    // compute the aspect ratio
    double ar =
            static_cast<double>(s_tools->effectiveCtx().glViewport.height()) /
            s_tools->effectiveCtx().glViewport.width();

    ccGLMatrixd projMatrix;
    if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
        // DGM: the 'zNearCoef' must not be too small, otherwise the loss in
        // accuracy for the detph buffer is too high and the display is
        // jeopardized, especially for entities with large coordinates) zNear =
        // zFar * m_viewportParams.zNearCoef;
        zNear = bbHalfDiag *
                s_tools->effectiveCtx()
                        .viewportParams.zNearCoef;  // we want a stable value!
        // zNear = std::max(bbHalfDiag * m_viewportParams.zNearCoef, zNear);
        // //we want a stable value!
        zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

        double xMax = zNear *
                      s_tools->effectiveCtx()
                              .viewportParams.computeDistanceToHalfWidthRatio();
        double yMax = xMax * ar;

        // DGM: we now take 'frustumAsymmetry' into account (for stereo
        // rendering)
        double frustumAsymmetry = 0.0;
        projMatrix = ecvGenericDisplayTools::Frustum(-xMax - frustumAsymmetry,
                                                     xMax - frustumAsymmetry,
                                                     -yMax, yMax, zNear, zFar);
    } else {
        // zNear = std::max(zNear, 0.0);
        zFar = std::max(zNear + ZERO_TOLERANCE_D, zFar);

        // CVLog::Print(QString("cameraCenterToPivotDist = %0 / zNear = %1 /
        // zFar = %2").arg(cameraCenterToPivotDist).arg(zNear).arg(zFar));

        double xMax = std::abs(cameraCenterToRotationCentertDist) *
                      s_tools->effectiveCtx()
                              .viewportParams.computeDistanceToHalfWidthRatio();
        double yMax = xMax * ar;

        projMatrix = ecvGenericDisplayTools::Ortho(-xMax, xMax, -yMax, yMax,
                                                   zNear, zFar);
    }
    return projMatrix;
}

void ecvDisplayTools::UpdateProjectionMatrix() {
    ProjectionMetrics metrics;

    s_tools->effectiveCtx().projMatd =
            ComputeProjectionMatrix(true, &metrics,
                                    nullptr);  // no stereo vision by default!

    s_tools->effectiveCtx().viewportParams.zNear = metrics.zNear;
    s_tools->effectiveCtx().viewportParams.zFar = metrics.zFar;
    s_tools->effectiveCtx().cameraToBBCenterDist = metrics.cameraToBBCenterDist;
    s_tools->effectiveCtx().bbHalfDiag = metrics.bbHalfDiag;

    s_tools->effectiveCtx().validProjectionMatrix = true;
}

CCVector3d ecvDisplayTools::GetRealCameraCenter() {
    // the camera center is always defined in perspective mode
    if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
        return s_tools->effectiveCtx().viewportParams.getCameraCenter();
    }

    // in orthographic mode, we put the camera at the center of the
    // visible objects (along the viewing direction)
    ccBBox box;
    GetVisibleObjectsBB(box);

    return CCVector3d(
            s_tools->effectiveCtx().viewportParams.getCameraCenter().x,
            s_tools->effectiveCtx().viewportParams.getCameraCenter().y,
            box.isValid() ? box.getCenter().z : 0.0);
}

ccGLMatrixd ecvDisplayTools::ComputeModelViewMatrix() {
    ccGLMatrixd viewMatd =
            s_tools->effectiveCtx().viewportParams.computeViewMatrix();

    ccGLMatrixd scaleMatd =
            s_tools->effectiveCtx().viewportParams.computeScaleMatrix(
                    s_tools->effectiveCtx().glViewport);

    return scaleMatd * viewMatd;
}

void ecvDisplayTools::UpdateModelViewMatrix() {
    // we save visualization matrix
    s_tools->effectiveCtx().viewMatd = ComputeModelViewMatrix();

    s_tools->effectiveCtx().validModelviewMatrix = true;
}

void ecvDisplayTools::SetBaseViewMat(ccGLMatrixd& mat) {
    s_tools->effectiveCtx().viewportParams.viewMat = mat;

    InvalidateVisualization();

    // we emit the 'baseViewMatChanged' signal
    emit s_tools->baseViewMatChanged(
            s_tools->effectiveCtx().viewportParams.viewMat);
    emit s_tools->cameraParamChanged();
}

void ecvDisplayTools::SetPerspectiveState(bool state, bool objectCenteredView) {
    // previous state
    bool perspectiveWasEnabled =
            s_tools->effectiveCtx().viewportParams.perspectiveView;
    bool viewWasObjectCentered =
            s_tools->effectiveCtx().viewportParams.objectCenteredView;

    // new state
    s_tools->effectiveCtx().viewportParams.perspectiveView = state;
    s_tools->effectiveCtx().viewportParams.objectCenteredView =
            objectCenteredView;

    // Camera center to pivot vector
    CCVector3d PC = s_tools->effectiveCtx().viewportParams.getCameraCenter() -
                    s_tools->effectiveCtx().viewportParams.getPivotPoint();

    if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
        if (!perspectiveWasEnabled)  // from ortho. mode to perspective view
        {
            // we compute the camera position that gives 'quite' the same view
            // as the ortho one (i.e. we replace the zoom by setting the camera
            // at the right distance from the pivot point)
            double currentFov_deg = static_cast<double>(GetFov());
            assert(cloudViewer::GreaterThanEpsilon(currentFov_deg));
            // see how pixelSize is computed!
            double screenSize =
                    std::min(s_tools->effectiveCtx().glViewport.width(),
                             s_tools->effectiveCtx().glViewport.height()) *
                    s_tools->effectiveCtx().viewportParams.pixelSize;
            if (screenSize > 0.0) {
                PC.z = screenSize /
                       (s_tools->effectiveCtx().viewportParams.zoom *
                        std::tan(
                                cloudViewer::DegreesToRadians(currentFov_deg)));
            }
        }

        // display message
        DisplayNewMessage(objectCenteredView ? "Centered perspective ON"
                                             : "Viewer-based perspective ON",
                          LOWER_LEFT_MESSAGE, false, 2,
                          PERSPECTIVE_STATE_MESSAGE);
    } else {
        // object-centered mode is forced for otho. view
        s_tools->effectiveCtx().viewportParams.objectCenteredView = true;

        if (perspectiveWasEnabled)  // from perspective view to ortho. view
        {
            // we compute the zoom equivalent to the corresponding camera
            // position (inverse of above calculus)
            float newZoom = ComputePerspectiveZoom();
            SetZoom(newZoom);
        }

        DisplayNewMessage("Perspective OFF", LOWER_LEFT_MESSAGE, false, 2,
                          PERSPECTIVE_STATE_MESSAGE);
    }

    // if we change form object-based to viewer-based visualization, we must
    //'rotate' around the object (or the opposite ;)
    if (viewWasObjectCentered &&
        !s_tools->effectiveCtx().viewportParams.objectCenteredView) {
        s_tools->effectiveCtx().viewportParams.viewMat.transposed().apply(
                PC);  // inverse rotation
    } else if (!viewWasObjectCentered &&
               s_tools->effectiveCtx().viewportParams.objectCenteredView) {
        s_tools->effectiveCtx().viewportParams.viewMat.apply(PC);
    }

    SetCameraPos(s_tools->effectiveCtx().viewportParams.getPivotPoint() + PC);

    emit s_tools->perspectiveStateChanged();
    emit s_tools->cameraParamChanged();

    // auto-save last perspective settings
    {
        QSettings settings;
        settings.beginGroup(c_ps_groupName);
        // write parameters
        settings.setValue(
                c_ps_perspectiveView,
                s_tools->effectiveCtx().viewportParams.perspectiveView);
        settings.setValue(
                c_ps_objectMode,
                s_tools->effectiveCtx().viewportParams.objectCenteredView);
        settings.endGroup();
    }

    s_tools->effectiveCtx().bubbleViewModeEnabled = false;

    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();
}

bool ecvDisplayTools::ObjectPerspectiveEnabled() {
    bool perspectiveWasEnabled =
            s_tools->effectiveCtx().viewportParams.perspectiveView;
    bool viewWasObjectCentered =
            s_tools->effectiveCtx().viewportParams.objectCenteredView;
    return perspectiveWasEnabled && viewWasObjectCentered;
}

bool ecvDisplayTools::ViewerPerspectiveEnabled() {
    bool perspectiveWasEnabled =
            s_tools->effectiveCtx().viewportParams.perspectiveView;
    bool viewWasObjectCentered =
            s_tools->effectiveCtx().viewportParams.objectCenteredView;
    return perspectiveWasEnabled && !viewWasObjectCentered;
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

    if (s_tools->effectiveCtx().bubbleViewModeEnabled) {
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
        GetVisibleObjectsBB(zoomedBox, sharedTools());
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
                std::min(s_tools->effectiveCtx().glViewport.width(),
                         s_tools->effectiveCtx().glViewport.height());
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
    s_tools->Update2DLabel(true);
}

void ecvDisplayTools::SetRedrawRecursive(bool redraw /* = false*/) {
    if (!sharedTools()) return;
    GetSceneDB()->setRedrawFlagRecursive(redraw);
    GetOwnDB()->setRedrawFlagRecursive(redraw);
}

void ecvDisplayTools::UpdateNamePoseRecursive() {
    if (!sharedTools()) return;
    GetSceneDB()->updateNameIn3DRecursive();
    GetOwnDB()->updateNameIn3DRecursive();

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
    if (!obj || !sharedTools()) return;
    SetRedrawRecursive(false);
    obj->setRedrawFlagRecursive(true);

    // If entity lives in a secondary view, redraw that view directly
    // so property changes (e.g. SF scale visibility) take effect there.
    ecvGenericGLDisplay* disp = obj->getDisplay();
    if (disp && disp != sharedTools()) {
        disp->redraw(only2D, forceRedraw);
    }
    RedrawDisplay(only2D, forceRedraw);
}

void ecvDisplayTools::RedrawObjects(std::initializer_list<ccHObject*> objects,
                                    bool only2D /* = false*/,
                                    bool forceRedraw /* = true*/) {
    if (!sharedTools()) return;
    SetRedrawRecursive(false);
    for (auto* obj : objects) {
        if (obj) obj->setRedrawFlagRecursive(true);
    }
    // Redraw secondary views that own any of these entities.
    for (auto* obj : objects) {
        if (!obj) continue;
        ecvGenericGLDisplay* disp = obj->getDisplay();
        if (disp && disp != sharedTools()) {
            disp->redraw(only2D, forceRedraw);
        }
    }
    RedrawDisplay(only2D, forceRedraw);
}

void ecvDisplayTools::GetVisibleObjectsBB(ccBBox& box,
                                          const ecvGenericGLDisplay* display) {
    if (s_tools->m_globalDBRoot) {
        box = s_tools->m_globalDBRoot->getDisplayBB_recursive(false, display);
    }

    if (s_tools->m_winDBRoot) {
        ccBBox ownBox =
                s_tools->m_winDBRoot->getDisplayBB_recursive(false, display);
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

void ecvDisplayTools::DisplayOverlayEntities(bool state) {
    s_tools->effectiveCtx().displayOverlayEntities = state;
    if (!state) {
        ClearBubbleView();
    }
}

void ecvDisplayTools::SetSceneDB(ccHObject* root) {
    s_tools->m_globalDBRoot = root;
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

    if (s_tools->m_winDBRoot) {
        s_tools->m_winDBRoot->addChild(
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
    if (s_tools->m_winDBRoot) s_tools->m_winDBRoot->removeChild(obj);
}

void ecvDisplayTools::SetRemoveViewIDs(std::vector<removeInfo>& removeinfos) {
    if (removeinfos.size() > 0) {
        s_tools->m_removeInfos = removeinfos;
        s_tools->m_removeFlag = true;
    } else {
        s_tools->m_removeFlag = false;
    }
}

void ecvDisplayTools::ZoomCamera(double zoomFactor, int viewport) {
    sharedTools()->zoomCamera(zoomFactor, viewport);
    if (!GetViewportParameters().perspectiveView) {
        sharedTools()->UpdateZoom(static_cast<float>(zoomFactor));
    }
    UpdateDisplayParameters();
}

void ecvDisplayTools::SetInteractionMode(INTERACTION_FLAGS flags) {
    // Write-through: update both the active secondary view AND the singleton.
    auto* av = activeSecondaryView();
    if (av) {
        av->setInteractionMode(flags);
        QWidget* w = av->asWidget();
        if (w) {
            w->setMouseTracking(flags & (INTERACT_CLICKABLE_ITEMS |
                                         INTERACT_SIG_MOUSE_MOVED));
        }
    }

    s_tools->effectiveCtx().interactionFlags = flags;

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
        s_tools->effectiveCtx().clickableItemsVisible = false;
    }
}

ecvDisplayTools::INTERACTION_FLAGS ecvDisplayTools::GetInteractionMode() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getInteractionMode();
    }
    return s_tools->effectiveCtx().interactionFlags;
}

CCVector3d ecvDisplayTools::GetCurrentViewDir() {
    auto* av = activeSecondaryView();
    const auto& vp = av ? av->getViewportParameters()
                        : s_tools->effectiveCtx().viewportParams;
    const double* M = vp.viewMat.data();
    CCVector3d axis(-M[2], -M[6], -M[10]);
    axis.normalize();
    return axis;
}

CCVector3d ecvDisplayTools::GetCurrentUpDir() {
    auto* av = activeSecondaryView();
    const auto& vp = av ? av->getViewportParameters()
                        : s_tools->effectiveCtx().viewportParams;
    const double* M = vp.viewMat.data();
    CCVector3d axis(M[1], M[5], M[9]);
    axis.normalize();
    return axis;
}

float ecvDisplayTools::GetFov() {
    auto* av = activeSecondaryView();
    if (av) {
        const auto& vp = av->getViewportParameters();
        return vp.fov_deg;
    }
    return (s_tools->effectiveCtx().bubbleViewModeEnabled
                    ? s_tools->effectiveCtx().bubbleViewFov_deg
                    : s_tools->effectiveCtx().viewportParams.fov_deg);
}

void ecvDisplayTools::SetupProjectiveViewport(
        const ccGLMatrixd& cameraMatrix,
        float fov_deg /*=0.0f*/,
        float ar /*=1.0f*/,
        bool viewerBasedPerspective /*=true*/,
        bool bubbleViewMode /*=false*/) {
    // perspective (viewer-based by default)
    if (bubbleViewMode) {
        SetBubbleViewMode(true);
    } else {
        SetPerspectiveState(true, !viewerBasedPerspective);
    }

    // field of view (= OpenGL 'fovy' but in degrees)
    if (fov_deg > 0.0f) {
        if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
            SetFov(fov_deg);
        } else {
            SetParallelScale(
                    static_cast<double>(cloudViewer::DegreesToRadians(fov_deg)),
                    0);
        }
    }

    // aspect ratio
    SetAspectRatio(ar);

    // set the camera matrix 'translation' as OpenGL camera center
    CCVector3d T = cameraMatrix.getTranslationAsVec3D();
    CCVector3d UP = cameraMatrix.getColumnAsVec3D(1);
    cameraMatrix.applyRotation(UP.data());
    SetCameraPos(T);
    SetCameraPosition(T.data(), UP.data());
    if (viewerBasedPerspective &&
        s_tools->effectiveCtx().autoPickPivotAtCenter) {
        SetPivotPoint(T);
    }

    // apply orientation matrix
    ccGLMatrixd trans = cameraMatrix;
    trans.clearTranslation();
    trans.invert();
    SetBaseViewMat(trans);

    ResetCameraClippingRange();
    UpdateScreen();
}

void ecvDisplayTools::SetAspectRatio(float ar) {
    if (ar < 0.0f) {
        CVLog::Warning("[ecvDisplayTools::setAspectRatio] Invalid AR value!");
        return;
    }

    if (s_tools->effectiveCtx().viewportParams.cameraAspectRatio != ar) {
        // update param
        s_tools->effectiveCtx().viewportParams.cameraAspectRatio = ar;

        // and camera state
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
    }
}

void ecvDisplayTools::SetFov(float fov_deg) {
    if (cloudViewer::LessThanEpsilon(fov_deg) || fov_deg > 180.0f) {
        CVLog::Warning("[ecvDisplayTools::setFov] Invalid FOV value!");
        return;
    }

    // derivation if we are in bubble-view mode
    if (s_tools->effectiveCtx().bubbleViewModeEnabled) {
        SetBubbleViewFov(fov_deg);
    } else if (s_tools->effectiveCtx().viewportParams.fov_deg != fov_deg) {
        // update param
        s_tools->effectiveCtx().viewportParams.fov_deg = fov_deg;
        // and camera state (if perspective view is 'on')
        {
            SetCameraFovy(fov_deg);
            InvalidateViewport();
            InvalidateVisualization();
            Deprecate3DLayer();

            DisplayNewMessage(
                    QString("F.O.V. = %1 deg.").arg(fov_deg, 0, 'f', 1),
                    LOWER_LEFT_MESSAGE,  // DGM HACK: we cheat and use the same
                                         // 'slot' as the window size
                    false, 2, SCREEN_SIZE_MESSAGE);
        }

        emit s_tools->fovChanged(
                s_tools->effectiveCtx().viewportParams.fov_deg);
        emit s_tools->cameraParamChanged();
    }
}

void ecvDisplayTools::DisplayNewMessage(const QString& message,
                                        MessagePosition pos,
                                        bool append /*=false*/,
                                        int displayMaxDelay_sec /*=2*/,
                                        MessageType type /*=CUSTOM_MESSAGE*/) {
    if (message.isEmpty()) {
        if (!append) {
            std::list<MessageToDisplay>::iterator it =
                    s_tools->m_messagesToDisplay.begin();
            while (it != s_tools->m_messagesToDisplay.end()) {
                // same position? we remove the message
                if (it->position == pos) {
                    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                    it->message));
                    it = s_tools->m_messagesToDisplay.erase(it);
                } else
                    ++it;
            }
        } else {
            CVLog::Warning(
                    "[ecvDisplayTools::DisplayNewMessage] Appending an empty "
                    "message has no effect!");
        }
        return;
    }

    // shall we replace the equivalent message(if any)?
    if (!append) {
        // only if type is not 'custom'
        if (type != CUSTOM_MESSAGE) {
            for (std::list<MessageToDisplay>::iterator it =
                         s_tools->m_messagesToDisplay.begin();
                 it != s_tools->m_messagesToDisplay.end();) {
                // same type? we remove it
                if (it->type == type) {
                    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                    it->message));
                    it = s_tools->m_messagesToDisplay.erase(it);
                } else
                    ++it;
            }
        }
    } else {
        if (pos == SCREEN_CENTER_MESSAGE) {
            CVLog::Warning(
                    "[ecvDisplayTools::DisplayNewMessage] Append is not "
                    "supported for center screen messages!");
        }
    }

    MessageToDisplay mess;
    mess.message = message;
    mess.messageValidity_sec =
            s_tools->m_timer.elapsed() / 1000 + displayMaxDelay_sec;
    mess.position = pos;
    mess.type = type;
    s_tools->m_messagesToDisplay.push_back(mess);
    // CVLog::Print(QString("[DisplayNewMessage] New message valid until %1
    // s.").arg(mess.messageValidity_sec));
}

void ecvDisplayTools::SetPivotPoint(const CCVector3d& P,
                                    bool autoUpdateCameraPos /*=false*/,
                                    bool verbose /*=false*/) {
    if (autoUpdateCameraPos &&
        (!s_tools->effectiveCtx().viewportParams.perspectiveView ||
         s_tools->effectiveCtx().viewportParams.objectCenteredView)) {
        // compute the equivalent camera center
        CCVector3d dP =
                s_tools->effectiveCtx().viewportParams.getPivotPoint() - P;
        CCVector3d MdP = dP;
        s_tools->effectiveCtx().viewportParams.viewMat.applyRotation(MdP);
        CCVector3d newCameraPos =
                s_tools->effectiveCtx().viewportParams.getCameraCenter() + MdP -
                dP;
        SetCameraPos(newCameraPos);
    }

    s_tools->effectiveCtx().viewportParams.setPivotPoint(P, true);
    SetAutoUpateCameraPos(autoUpdateCameraPos);
    SetCenterOfRotation(P);

    emit s_tools->pivotPointChanged(
            s_tools->effectiveCtx().viewportParams.getPivotPoint());
    emit s_tools->cameraParamChanged();

    if (verbose) {
        const unsigned& precision =
                GetDisplayParameters().displayedNumPrecision;
        DisplayNewMessage(QString(), LOWER_LEFT_MESSAGE,
                          false);  // clear previous message
        DisplayNewMessage(QString("Point (%1 ; %2 ; %3) set as rotation center")
                                  .arg(P.x, 0, 'f', precision)
                                  .arg(P.y, 0, 'f', precision)
                                  .arg(P.z, 0, 'f', precision),
                          LOWER_LEFT_MESSAGE, true);
        RedrawDisplay(true, false);
    }

    // s_tools->effectiveCtx().autoPivotCandidate = CCVector3d(0, 0, 0);
    s_tools->effectiveCtx().autoPivotCandidate = P;
    InvalidateViewport();
    InvalidateVisualization();
}

void ecvDisplayTools::SetAutoPickPivotAtCenter(bool state) {
    if (s_tools->effectiveCtx().autoPickPivotAtCenter != state) {
        s_tools->effectiveCtx().autoPickPivotAtCenter = state;

        if (state) {
            // force 3D redraw to update the coordinates of the 'auto' pivot
            // center
            s_tools->effectiveCtx().autoPivotCandidate = CCVector3d(0, 0, 0);
            // RedrawDisplay(false);
        }
    }
}

void ecvDisplayTools::LockRotationAxis(bool state, const CCVector3d& axis) {
    s_tools->effectiveCtx().rotationAxisLocked = state;
    s_tools->effectiveCtx().lockedRotationAxis = axis;
    s_tools->effectiveCtx().lockedRotationAxis.normalize();
}

void ecvDisplayTools::GetContext(CC_DRAW_CONTEXT& CONTEXT) {
    // display size
    CONTEXT.glW = s_tools->effectiveCtx().glViewport.width();
    CONTEXT.glH = s_tools->effectiveCtx().glViewport.height();
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
                s_tools->effectiveCtx().viewportParams.defaultPointSize);
        CONTEXT.defaultLineWidth = static_cast<unsigned char>(
                s_tools->effectiveCtx().viewportParams.defaultLineWidth);
        CONTEXT.display = s_tools;
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

void ecvDisplayTools::SetCameraPos(const CCVector3d& P) {
    if ((s_tools->effectiveCtx().viewportParams.getCameraCenter() - P)
                .norm2d() != 0.0) {
        s_tools->effectiveCtx().viewportParams.setCameraCenter(P, true);
        SetCameraPosition(P);
        emit s_tools->cameraPosChanged(
                s_tools->effectiveCtx().viewportParams.getCameraCenter());
        emit s_tools->cameraParamChanged();
        InvalidateViewport();
        InvalidateVisualization();
        Deprecate3DLayer();
    }
}

const ecvGui::ParamStruct& ecvDisplayTools::GetDisplayParameters() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getDisplayParameters();
    }
    if (s_tools->m_overridenDisplayParametersEnabled) {
        s_tools->m_overridenDisplayParameters.initFontSizesIfNeeded();
        return s_tools->m_overridenDisplayParameters;
    } else {
        const ecvGui::ParamStruct& params = ecvGui::Parameters();
        ecvGui::UpdateParameters();
        return params;
    }
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

    ccGLMatrixd rotationMat;
    rotationMat.setRotation(
            ccGLMatrixd::ToEigenMatrix3(params.modelViewMat).data());
    s_tools->effectiveCtx().viewportParams.viewMat = rotationMat;
    double nearFar[2];
    GetCameraClip(nearFar);

    CCVector3d pivot;
    GetCenterOfRotation(pivot);
    s_tools->effectiveCtx().viewportParams.setPivotPoint(pivot);

    s_tools->effectiveCtx().viewportParams.zNear = nearFar[0];
    s_tools->effectiveCtx().viewportParams.zFar = nearFar[1];
    s_tools->effectiveCtx().viewportParams.fov_deg =
            static_cast<float>(GetCameraFovy());
    params.fov_deg = s_tools->effectiveCtx().viewportParams.fov_deg;

    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = Width() * GetDevicePixelRatio();
    params.viewport[3] = Height() * GetDevicePixelRatio();
    SetGLViewport(QRect(0, 0, Width(), Height()));

    params.perspective = s_tools->effectiveCtx().viewportParams.perspectiveView;
    params.pixelSize = s_tools->effectiveCtx().viewportParams.pixelSize;
}

void ecvDisplayTools::SetDisplayParameters(const ecvGui::ParamStruct& params) {
    // Write-through: also update the active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        av->setDisplayParameters(params, true);
    }

    s_tools->m_overridenDisplayParametersEnabled = true;
    s_tools->m_overridenDisplayParameters = params;
    s_tools->m_overridenDisplayParameters.initFontSizesIfNeeded();
    ecvGui::Set(params);
}

void ecvDisplayTools::UpdateDisplayParameters() {
    // set camera near and far
    double nearFar[2];
    GetCameraClip(nearFar);
    s_tools->effectiveCtx().viewportParams.zNear = nearFar[0];
    s_tools->effectiveCtx().viewportParams.zFar = nearFar[1];

    ccGLMatrixd viewMat;
    GetViewMatrix(viewMat.data());
    ccGLMatrixd rotationMat;
    rotationMat.setRotation(ccGLMatrixd::ToEigenMatrix3(viewMat).data());
    s_tools->effectiveCtx().viewportParams.viewMat = rotationMat;

    CCVector3d pivot;
    GetCenterOfRotation(pivot);
    s_tools->effectiveCtx().viewportParams.setPivotPoint(pivot);

    // set camera fov
    if (s_tools->effectiveCtx().viewportParams.perspectiveView) {
        s_tools->effectiveCtx().viewportParams.zoom = ComputePerspectiveZoom();
        s_tools->effectiveCtx().viewportParams.fov_deg =
                static_cast<float>(GetCameraFovy());
    } else {
        s_tools->effectiveCtx().viewportParams.fov_deg = static_cast<float>(
                cloudViewer::RadiansToDegrees(GetParallelScale(0)));
    }

    // set camera pos
    double pos[3];
    GetCameraPos(pos);
    s_tools->effectiveCtx().viewportParams.setCameraCenter(
            CCVector3d::fromArray(pos), true);

    // set camera focal
    double focal[3];
    GetCameraFocal(focal);
    s_tools->effectiveCtx().viewportParams.focal = CCVector3d::fromArray(focal);

    // set camera up
    double up[3];
    GetCameraUp(up);
    s_tools->effectiveCtx().viewportParams.up = CCVector3d::fromArray(up);
}

void ecvDisplayTools::SetViewportParameters(
        const ecvViewportParameters& params) {
    // Write-through: also update the active secondary view.
    auto* av = activeSecondaryView();
    if (av) {
        av->setViewportParameters(params);
    }

    ecvViewportParameters oldParams = s_tools->effectiveCtx().viewportParams;
    s_tools->effectiveCtx().viewportParams = params;

    if (oldParams.perspectiveView == params.perspectiveView) {
        if (oldParams.perspectiveView) {
            SetFov(params.fov_deg);
        } else {
            SetParallelScale(static_cast<double>(cloudViewer::DegreesToRadians(
                                     params.fov_deg)),
                             0);
        }
    } else {  // ignore fov_deg in different show mode
        // keep old show mode
        s_tools->effectiveCtx().viewportParams.perspectiveView =
                oldParams.perspectiveView;
    }

    SetCameraClip(params.zNear, params.zFar);
    SetPivotPoint(params.getPivotPoint(), false, false);
    SetCameraPosition(params.getCameraCenter().u, params.focal.u, params.up.u);

    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();

    emit s_tools->baseViewMatChanged(
            s_tools->effectiveCtx().viewportParams.viewMat);
    emit s_tools->pivotPointChanged(
            s_tools->effectiveCtx().viewportParams.getPivotPoint());
    emit s_tools->cameraPosChanged(
            s_tools->effectiveCtx().viewportParams.getCameraCenter());
    emit s_tools->fovChanged(s_tools->effectiveCtx().viewportParams.fov_deg);
    emit s_tools->cameraParamChanged();
}

const ecvViewportParameters& ecvDisplayTools::GetViewportParameters() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getViewportParameters();
    }
    UpdateDisplayParameters();
    return s_tools->effectiveCtx().viewportParams;
}

void ecvDisplayTools::SetBubbleViewMode(bool state) {
    // Backup the camera center before entering this mode!
    bool bubbleViewModeWasEnabled =
            s_tools->effectiveCtx().bubbleViewModeEnabled;
    if (!s_tools->effectiveCtx().bubbleViewModeEnabled && state) {
        s_tools->effectiveCtx().preBubbleViewParameters =
                s_tools->effectiveCtx().viewportParams;
    }

    if (state) {
        // bubble-view mode = viewer-based perspective mode
        // setPerspectiveState must be called first as it
        // automatically deactivates bubble-view mode!
        SetPerspectiveState(true, false);

        s_tools->effectiveCtx().bubbleViewModeEnabled = true;

        // when entering this mode, we reset the f.o.v.
        s_tools->effectiveCtx().bubbleViewFov_deg =
                0.0f;  // to trick the signal emission mechanism
        SetBubbleViewFov(90.0f);
    } else if (bubbleViewModeWasEnabled) {
        s_tools->effectiveCtx().bubbleViewModeEnabled = false;
        SetPerspectiveState(
                s_tools->effectiveCtx().preBubbleViewParameters.perspectiveView,
                s_tools->effectiveCtx()
                        .preBubbleViewParameters.objectCenteredView);

        // when leaving this mode, we restore the original camera center
        SetViewportParameters(s_tools->effectiveCtx().preBubbleViewParameters);
    }
}

void ecvDisplayTools::SetBubbleViewFov(float fov_deg) {
    if (fov_deg < FLT_EPSILON || fov_deg > 180.0f) return;

    if (fov_deg != s_tools->effectiveCtx().bubbleViewFov_deg) {
        s_tools->effectiveCtx().bubbleViewFov_deg = fov_deg;

        if (s_tools->effectiveCtx().bubbleViewModeEnabled) {
            InvalidateViewport();
            InvalidateVisualization();
            Deprecate3DLayer();
            emit s_tools->fovChanged(s_tools->effectiveCtx().bubbleViewFov_deg);
            emit s_tools->cameraParamChanged();
        }
    }
}

void ecvDisplayTools::SetPixelSize(float pixelSize) {
    if (s_tools->effectiveCtx().viewportParams.pixelSize != pixelSize) {
        s_tools->effectiveCtx().viewportParams.pixelSize = pixelSize;
    }
    InvalidateViewport();
    InvalidateVisualization();
    Deprecate3DLayer();
}

void ecvDisplayTools::SetZoom(float value) {
    // zoom should be avoided in bubble-view mode
    assert(!s_tools->effectiveCtx().bubbleViewModeEnabled);

    if (value < CC_GL_MIN_ZOOM_RATIO)
        value = CC_GL_MIN_ZOOM_RATIO;
    else if (value > CC_GL_MAX_ZOOM_RATIO)
        value = CC_GL_MAX_ZOOM_RATIO;

    if (s_tools->effectiveCtx().viewportParams.zoom != value) {
        s_tools->effectiveCtx().viewportParams.zoom = value;
        InvalidateViewport();
        InvalidateVisualization();
        // Deprecate3DLayer();
    }
}

void ecvDisplayTools::UpdateZoom(float zoomFactor) {
    // no 'zoom' in viewer based perspective
    assert(!s_tools->effectiveCtx().viewportParams.perspectiveView);

    if (zoomFactor > 0.0f && zoomFactor != 1.0f) {
        SetZoom(s_tools->effectiveCtx().viewportParams.zoom * zoomFactor);
    }
}

void ecvDisplayTools::SetPickingMode(PICKING_MODE mode /*=DEFAULT_PICKING*/) {
    // Write-through: update both the active secondary view AND the singleton
    // so that m_tools-> direct access in QVTKWidgetCustom sees the correct
    // value.
    auto* av = activeSecondaryView();
    if (av) {
        av->setPickingMode(mode);
    }

    if (s_tools->effectiveCtx().pickingModeLocked) {
        if ((mode != s_tools->effectiveCtx().pickingMode) &&
            (mode != DEFAULT_PICKING))
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

    s_tools->effectiveCtx().pickingMode = mode;
}

ecvDisplayTools::PICKING_MODE ecvDisplayTools::GetPickingMode() {
    auto* av = activeSecondaryView();
    if (av) {
        return av->getPickingMode();
    }
    return s_tools->effectiveCtx().pickingMode;
}

void ecvDisplayTools::LockPickingMode(bool state) {
    auto* av = activeSecondaryView();
    if (av) {
        // ecvGLView stores its own m_pickingModeLocked; keep it in sync.
        // (setPickingMode already checks the lock on the view side)
    }
    s_tools->effectiveCtx().pickingModeLocked = state;
}

bool ecvDisplayTools::IsPickingModeLocked() {
    return s_tools->effectiveCtx().pickingModeLocked;
}

double ecvDisplayTools::ComputeActualPixelSize() {
    if (!s_tools->effectiveCtx().viewportParams.perspectiveView) {
        return static_cast<double>(
                s_tools->effectiveCtx().viewportParams.pixelSize /
                s_tools->effectiveCtx().viewportParams.zoom);
    }

    int minScreenDim = std::min(s_tools->effectiveCtx().glViewport.width(),
                                s_tools->effectiveCtx().glViewport.height());
    if (minScreenDim <= 0) return 1.0;

    // Camera center to pivot vector
    double zoomEquivalentDist =
            (s_tools->effectiveCtx().viewportParams.getCameraCenter() -
             s_tools->effectiveCtx().viewportParams.getPivotPoint())
                    .norm();

    double currentFov_deg = static_cast<double>(GetFov());
    return zoomEquivalentDist *
           std::tan(cloudViewer::DegreesToRadians(
                   std::min(currentFov_deg, 75.0))) /
           minScreenDim;  // tan(75) = 3.73 (then it quickly increases!)
}

bool ecvDisplayTools::IsRectangularPickingAllowed() {
    return s_tools->effectiveCtx().allowRectangularEntityPicking;
}

void ecvDisplayTools::SetRectangularPickingAllowed(bool state) {
    s_tools->effectiveCtx().allowRectangularEntityPicking = state;
}

void ecvDisplayTools::ShowPivotSymbol(bool state) {
    // is the pivot really going to be drawn?
    if (state && !s_tools->effectiveCtx().pivotSymbolShown &&
        s_tools->effectiveCtx().viewportParams.objectCenteredView &&
        s_tools->effectiveCtx().pivotVisibility != PIVOT_HIDE) {
        InvalidateViewport();
        Deprecate3DLayer();
    }

    s_tools->effectiveCtx().pivotSymbolShown = state;
}

int ecvDisplayTools::GetFontPointSize() {
    return GetOptimizedFontSize(
            s_tools->m_captureMode.enabled
                    ? FontSizeModifier(GetDisplayParameters().defaultFontSize,
                                       s_tools->m_captureMode.zoomFactor)
                    : GetDisplayParameters().defaultFontSize);
}

int ecvDisplayTools::GetLabelFontPointSize() {
    return GetOptimizedFontSize(
            s_tools->m_captureMode.enabled
                    ? FontSizeModifier(GetDisplayParameters().labelFontSize,
                                       s_tools->m_captureMode.zoomFactor)
                    : GetDisplayParameters().labelFontSize);
}

QFont ecvDisplayTools::GetLabelDisplayFont() {
    QFont font = s_tools->m_font;
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
    s_tools->m_shouldBeRefreshed = true;

    InvalidateViewport();
    InvalidateVisualization();
}

void ecvDisplayTools::RefreshDisplay(bool only2D /*=false*/,
                                     bool forceRedraw /* = true*/) {
    if (s_tools->m_shouldBeRefreshed) {
        RedrawDisplay(only2D, forceRedraw);
    }
}

void ecvDisplayTools::RedrawDisplay(bool only2D /*=false*/,
                                    bool forceRedraw /* = true*/) {
    if (!sharedTools()) return;

    // === Global housekeeping (stays in RedrawDisplay) ===

    // Debug traces cleanup
    if (s_tools->effectiveCtx().showDebugTraces) {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!s_tools->m_diagStrings.isEmpty()) {
            QStringList::iterator it = s_tools->m_diagStrings.begin();
            while (it != s_tools->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = s_tools->m_diagStrings.erase(it);
            }
        }
        s_tools->m_diagStrings
                << QString("only2D : %1").arg(only2D ? "true" : "false");
        s_tools->m_diagStrings << QString("ForceRedraw : %1")
                                          .arg(forceRedraw ? "true" : "false");
    } else {
        RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, DEBUG_LAYER_ID));
        if (!s_tools->m_diagStrings.isEmpty()) {
            QStringList::iterator it = s_tools->m_diagStrings.begin();
            while (it != s_tools->m_diagStrings.end()) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, *it));
                it = s_tools->m_diagStrings.erase(it);
            }
        }
    }

    CheckIfRemove();
    if (s_tools->m_removeAllFlag) {
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
                s_tools->m_messagesToDisplay.begin();
        qint64 currentTime_sec = s_tools->m_timer.elapsed() / 1000;
        while (it != s_tools->m_messagesToDisplay.end()) {
            if (it->messageValidity_sec < currentTime_sec) {
                RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                                it->message));
                it = s_tools->m_messagesToDisplay.erase(it);
            } else {
                ++it;
            }
        }
    }

    // === Per-view delegation (Phase B) ===
    // Skip the shared display tools to avoid infinite recursion:
    // ecvDisplayTools inherits ecvGenericGLDisplay so it may appear in the
    // view list; calling its virtual redraw() re-enters RedrawDisplay().
    const auto& views = ecvViewManager::instance().getAllViews();
    bool hasSecondary = false;
    for (auto* viewDisplay : views) {
        if (viewDisplay && viewDisplay != s_tools) {
            ecvViewManager::ScopedRenderOverride guard(viewDisplay);
            viewDisplay->redraw(only2D, forceRedraw);
            hasSecondary = true;
        }
    }

    // === Legacy singleton draw path (fallback when no views registered) ===
    s_tools->beginPrimaryRender();

    bool drawBackground = false;
    bool draw3DPass = false;
    bool drawForeground = true;
    bool draw3DCross = GetDisplayParameters().displayCross;

    if (s_tools->m_updateFBO || s_tools->m_captureMode.enabled) {
        drawBackground = true;
        draw3DPass = true;
    }

    CC_DRAW_CONTEXT CONTEXT;
    GetContext(CONTEXT);
    CONTEXT.display = s_tools;

    QRect originViewport = s_tools->effectiveCtx().glViewport;
    bool modifiedViewport = false;

    if (drawBackground) {
        if (s_tools->effectiveCtx().showDebugTraces) {
            s_tools->m_diagStrings << "draw background";
        }
        CONTEXT.clearColorLayer = true;
        CONTEXT.clearDepthLayer = true;
        DrawBackground(CONTEXT);
    }

    if (draw3DPass) {
        if (s_tools->effectiveCtx().showDebugTraces) {
            s_tools->m_diagStrings << "draw 3D";
        }
        CONTEXT.forceRedraw = forceRedraw;
        Draw3D(CONTEXT);
    }

    // Display debug traces
    if (s_tools->effectiveCtx().showDebugTraces) {
        if (!s_tools->m_diagStrings.isEmpty()) {
            QFont font = GetTextDisplayFont();
            int font_size = font.pointSize();
            QFontMetrics fm(font);
            int x = s_tools->effectiveCtx().glViewport.width() / 2 - 100;
            int margin = font_size / 2;
            int y = margin;
            {
                int height = (s_tools->m_diagStrings.size() + 1) *
                             (fm.height() + margin);
                WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                        DEBUG_LAYER_ID);
                param.color = ecvColor::dark;
                param.color.a = 0.5f;
                param.rect = QRect(x,
                                   s_tools->effectiveCtx().glViewport.height() -
                                           y - height,
                                   200, height);
                DrawWidgets(param, true);
            }
            y += margin;
            for (const QString& str : s_tools->m_diagStrings) {
                RenderText(x + font_size, y + font_size, str, font,
                           ecvColor::yellow, DEBUG_LAYER_ID);
                y += fm.height() + margin;
            }
        }
    }

    if (modifiedViewport) {
        SetGLViewport(originViewport);
        CONTEXT.glW = originViewport.width();
        CONTEXT.glH = originViewport.height();
    }

    if (drawBackground || draw3DCross) {
        s_tools->m_updateFBO = false;
    }

    if (drawForeground) {
        DrawForeground(CONTEXT);
    }

    s_tools->m_shouldBeRefreshed = false;

    if (false && s_tools->effectiveCtx().autoPickPivotAtCenter &&
        !s_tools->effectiveCtx().mouseMoved &&
        s_tools->effectiveCtx().autoPivotCandidate.norm2d() != 0.0) {
        SetPivotPoint(s_tools->effectiveCtx().autoPivotCandidate, true, false);
    }

    UpdateScreen();
    s_tools->endPrimaryRender();
}

void ecvDisplayTools::SetGLViewport(const QRect& rect) {
    const int retinaScale = GetDevicePixelRatio();
    s_tools->effectiveCtx().glViewport =
            QRect(rect.left() * retinaScale, rect.top() * retinaScale,
                  rect.width() * retinaScale, rect.height() * retinaScale);
    InvalidateViewport();
}

void ecvDisplayTools::drawCross() {}

void ecvDisplayTools::drawTrihedron() {}

void ecvDisplayTools::Draw3D(CC_DRAW_CONTEXT& CONTEXT) {
    CONTEXT.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
    if (s_tools->effectiveCtx().interactionFlags &
        INTERACT_TRANSFORM_ENTITIES) {
        CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    /****************************************/
    /****    PASS: 3D/FOREGROUND/LIGHT   ****/
    /****************************************/
    if (s_tools->effectiveCtx().customLightEnabled ||
        s_tools->effectiveCtx().sunLightEnabled) {
        CONTEXT.drawingFlags |= CC_LIGHT_ENABLED;

        // we enable absolute sun light (if activated)
        if (s_tools->effectiveCtx().sunLightEnabled) {
            // glEnableSunLight();
        }
    }

    // we draw 3D entities
    if (s_tools->m_globalDBRoot) {
        s_tools->m_globalDBRoot->draw(CONTEXT);
    }

    if (s_tools->m_winDBRoot) {
        s_tools->m_winDBRoot->draw(CONTEXT);
    }

#if 0
	//do this before drawing the pivot!
	if (s_tools->effectiveCtx().autoPickPivotAtCenter)
	{
		CCVector3d P;
		if (GetClick3DPos(s_tools->effectiveCtx().glViewport.width() / 2, s_tools->effectiveCtx().glViewport.height() / 2, P))
		{
			s_tools->effectiveCtx().autoPivotCandidate = P;
		}
	}
#endif

    if (s_tools->m_globalDBRoot &&
        s_tools->m_globalDBRoot->getChildrenNumber()) {
        // draw pivot
        // DrawPivot();
    }
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
        for (auto* view : views) {
            if (!view || view == sharedTools()) continue;
            CC_DRAW_CONTEXT viewCtx = context;
            viewCtx.display = view;
            sharedTools()->hideShowEntities(viewCtx);
        }
    }
    return true;
}

void ecvDisplayTools::RemoveEntities(const CC_DRAW_CONTEXT& CONTEXT) {
    sharedTools()->removeEntities(CONTEXT);
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
        for (auto* view : views) {
            if (!view || view == sharedTools()) continue;
            CC_DRAW_CONTEXT viewCtx = context;
            viewCtx.display = view;
            sharedTools()->removeEntities(viewCtx);
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
    CONTEXT.drawingFlags = CC_DRAW_2D;
    if (s_tools->effectiveCtx().interactionFlags &
        INTERACT_TRANSFORM_ENTITIES) {
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

            s_tools->setBackgroundColor(CONTEXT);
        }
    }
}

void ecvDisplayTools::DrawForeground(CC_DRAW_CONTEXT& CONTEXT) {
    /****************************************/
    /****  PASS: 2D/FOREGROUND/NO LIGHT  ****/
    /****************************************/

    CONTEXT.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    if (s_tools->effectiveCtx().interactionFlags &
        INTERACT_TRANSFORM_ENTITIES) {
        CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    // we draw 2D entities
    if (s_tools->m_globalDBRoot) s_tools->m_globalDBRoot->draw(CONTEXT);
    if (s_tools->m_winDBRoot) s_tools->m_winDBRoot->draw(CONTEXT);

    // current displayed scalar field color ramp (if any)
    ccRenderingTools::DrawColorRamp(CONTEXT);

    s_tools->m_clickableItems.clear();

    /*** overlay entities ***/
    if (s_tools->effectiveCtx().displayOverlayEntities) {
        // const ecvColor::Rgbub& textCol =
        // GetDisplayParameters().textDefaultCol;

        if (!s_tools->m_captureMode.enabled ||
            s_tools->m_captureMode.renderOverlayItems) {
            // scale: only in ortho mode
            if (!s_tools->effectiveCtx().viewportParams.perspectiveView) {
                SetScaleBarVisible(true);
            } else {
                SetScaleBarVisible(false);
            }
            UpdateScreen();
        }

        if (!s_tools->m_captureMode.enabled) {
            int yStart = 0;

            // current messages (if valid)
            if (!s_tools->m_messagesToDisplay.empty()) {
                QFont font = s_tools->m_font;
                QFontMetrics fm(font);
                int margin = fm.height() / 4;
                int ll_currentHeight =
                        s_tools->effectiveCtx().glViewport.height() - 10;
                int uc_currentHeight = 10;

                for (const auto& message : s_tools->m_messagesToDisplay) {
                    switch (message.position) {
                        case LOWER_LEFT_MESSAGE: {
                            RenderText(10, ll_currentHeight, message.message,
                                       font);
                            int messageHeight = fm.height();
                            ll_currentHeight -= (messageHeight + margin);
                        } break;
                        case UPPER_CENTER_MESSAGE: {
                            QRect rect = fm.boundingRect(message.message);
                            int x = (s_tools->effectiveCtx()
                                             .glViewport.width() -
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
                            RenderText((s_tools->effectiveCtx()
                                                .glViewport.width() -
                                        rect.width()) /
                                               2,
                                       (s_tools->effectiveCtx()
                                                .glViewport.height() -
                                        rect.height()) /
                                               2,
                                       message.message, newFont);
                        } break;
                    }
                }
            }

            // hot-zone
            { s_tools->DrawClickableItems(0, yStart); }
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
        if (disp && disp != sharedTools()) {
            context.display = disp;
            auto* viewCtx = disp->viewContext();
            if (viewCtx) {
                context.glW = viewCtx->glViewport.width();
                context.glH = viewCtx->glViewport.height();
            }
        }
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
                if (disp && disp != sharedTools()) {
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
    QString id = s_tools->pick2DLabel(x, y);

    auto* effView = ecvViewManager::instance().getEffectiveView();
    std::list<ccInteractor*>& items =
            effView ? effView->activeItemsRef() : s_tools->m_activeItems;
    items.clear();
    if (!id.isEmpty()) {
        ccHObject::Container labels;
        FilterByEntityType(labels, CV_TYPES::LABEL_2D);
        for (auto& label : labels) {
            if (label->isA(CV_TYPES::LABEL_2D) && label->isEnabled() &&
                label->isVisible()) {
                cc2DLabel* l = ccHObjectCaster::To2DLabel(label);
                if (l->getViewId().compare(id) == 0) {
                    items.push_back(l);
                }
            }
        }
    }
}

void ecvDisplayTools::FilterByEntityType(ccHObject::Container& labels,
                                         CV_CLASS_ENUM type) {
    if (s_tools->m_globalDBRoot)
        s_tools->m_globalDBRoot->filterChildren(labels, true, type);
    if (s_tools->m_winDBRoot)
        s_tools->m_winDBRoot->filterChildren(labels, true, type);
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
        Q2D.y = s_tools->effectiveCtx().glViewport.height() - 1 - Q2D.y;
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

    QFont realFont = (font ? *font : s_tools->m_font);
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

void ecvDisplayTools::ClearBubbleView() {
    if (!s_tools->m_hotZone) return;
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                    s_tools->m_hotZone->bbv_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                    s_tools->m_hotZone->fs_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                    s_tools->m_hotZone->psi_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D,
                                    s_tools->m_hotZone->lsi_label));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, "Exit"));
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, "clicked_items"));
}

void ecvDisplayTools::DrawClickableItems(int xStart0, int& yStart) {
    const static char* CLICKED_ITEMS = "clicked_items";
    // we init the necessary parameters the first time we need them
    if (!s_tools->m_hotZone) {
        s_tools->m_hotZone = new HotZone(ecvDisplayTools::GetCurrentScreen());
        s_tools->m_hotZoneOwnedBySingleton = true;
    } else if (GetPlatformAwareDPIScale() !=
               s_tools->m_hotZone
                       ->pixelDeviceRatio)  // the device pixel ratio has
                                            // changed (happens when changing
                                            // screen for instance)
    {
        s_tools->m_hotZone->updateInternalVariables(
                ecvDisplayTools::GetCurrentScreen());
    }

    // remember the last position of the 'top corner'
    s_tools->m_hotZone->topCorner =
            QPoint(xStart0, yStart) +
            QPoint(s_tools->m_hotZone->margin, s_tools->m_hotZone->margin);

    bool fullScreenEnabled = ExclusiveFullScreen();

    if (!s_tools->effectiveCtx().clickableItemsVisible &&
        !s_tools->effectiveCtx().bubbleViewModeEnabled && !fullScreenEnabled) {
        ClearBubbleView();
        // nothing to do
        return;
    }

    //"exit" icon
    // static const QImage c_exitIcon =
    // QImage(":/Resources/images/ecvExit.png").mirrored();
    int fullW = s_tools->effectiveCtx().glViewport.width();
    int fullH = s_tools->effectiveCtx().glViewport.height();

    // clear history
    RemoveWidgets(WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS));

    // draw semi-transparent background
    {
        QRect areaRect = s_tools->m_hotZone->rect(
                s_tools->effectiveCtx().clickableItemsVisible,
                s_tools->effectiveCtx().bubbleViewModeEnabled,
                fullScreenEnabled);
        areaRect.translate(s_tools->m_hotZone->topCorner);

        WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                CLICKED_ITEMS);
        param.color = ecvColor::FromRgba(ecvColor::odarkGrey);
        param.color.a = 210 / 255.0f;
        int x0 = areaRect.x();
        int y0 = fullH - areaRect.y() - areaRect.height();
        param.rect = QRect(x0, y0, areaRect.width(), areaRect.height());
        DrawWidgets(param, false);
    }

    yStart = s_tools->m_hotZone->topCorner.y();
    int offset = 0;
#ifdef Q_OS_MAC
    // fix the start of text vertically on macos
    offset = s_tools->m_hotZone->margin / 3;
#endif
    int iconSize = s_tools->m_hotZone->iconSize;

    if (fullScreenEnabled) {
        int xStart = s_tools->m_hotZone->topCorner.x();

        // label
        RenderText(xStart,
                   yStart + offset + s_tools->m_hotZone->yTextBottomLineShift,
                   s_tools->m_hotZone->fs_label, s_tools->m_hotZone->font,
                   ecvColor::defaultLabelBkgColor, CLICKED_ITEMS);

        // icon
        xStart += s_tools->m_hotZone->fs_labelRect.width() +
                  s_tools->m_hotZone->margin;

#ifdef Q_OS_MAC
        // fix the start of icon on mac
        xStart += s_tools->m_hotZone->margin * 4;
#endif
        //"full-screen" icon
        {
            int x0 = xStart;
            int y0 = fullH - (yStart + iconSize);
            WIDGETS_PARAMETER param(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                    CLICKED_ITEMS);
            param.color = ecvColor::FromRgba(ecvColor::ored);
            param.rect = QRect(x0, y0, iconSize + offset, iconSize);
            DrawWidgets(param, false);

            WIDGETS_PARAMETER texParam(WIDGETS_TYPE::WIDGET_T2D, CLICKED_ITEMS);
            texParam.color = ecvColor::bright;
            texParam.text = "Exit";
            texParam.rect =
                    QRect(x0, fullH - (yStart + offset / 2 + 3 * iconSize / 4),
                          iconSize, iconSize);
            texParam.fontSize = s_tools->m_hotZone->font.pointSize();
            DrawWidgets(texParam, false);
            s_tools->m_clickableItems.emplace_back(
                    ClickableItem::LEAVE_FULLSCREEN_MODE,
                    QRect(xStart, yStart, iconSize, iconSize));
            xStart += iconSize;
        }

        yStart += iconSize;
        yStart += s_tools->m_hotZone->margin;
    }

    if (s_tools->effectiveCtx().bubbleViewModeEnabled) {
        int xStart = s_tools->m_hotZone->topCorner.x();

        // label
        RenderText(xStart,
                   yStart + offset + s_tools->m_hotZone->yTextBottomLineShift,
                   s_tools->m_hotZone->bbv_label, s_tools->m_hotZone->font);

        // icon
        xStart += s_tools->m_hotZone->bbv_labelRect.width() +
                  s_tools->m_hotZone->margin;
#ifdef Q_OS_MAC
        // fix the start of icon on mac
        xStart += s_tools->m_hotZone->margin * 4;
#endif

        //"exit" icon
        {
            s_tools->m_clickableItems.emplace_back(
                    ClickableItem::LEAVE_BUBBLE_VIEW_MODE,
                    QRect(xStart, yStart, s_tools->m_hotZone->iconSize,
                          s_tools->m_hotZone->iconSize));
            xStart += s_tools->m_hotZone->iconSize;
        }

        yStart += s_tools->m_hotZone->iconSize;
        yStart += s_tools->m_hotZone->margin;
    }

    if (s_tools->effectiveCtx().clickableItemsVisible) {
        ecvColor::Rgb textColor = ecvColor::Rgb(s_tools->m_hotZone->color);
        WIDGETS_PARAMETER widgetParam(WIDGETS_TYPE::WIDGET_RECTANGLE_2D,
                                      CLICKED_ITEMS);
        widgetParam.color = ecvColor::FromRgba(ecvColor::ogreen);
        WIDGETS_PARAMETER sepParam(WIDGETS_TYPE::WIDGET_POINTS_2D,
                                   CLICKED_ITEMS);
        sepParam.color = widgetParam.color;
        sepParam.color.a = 0.5f;

        // default point size
        {
            int xStart = s_tools->m_hotZone->topCorner.x();

            RenderText(
                    xStart,
                    yStart + offset + s_tools->m_hotZone->yTextBottomLineShift,
                    s_tools->m_hotZone->psi_label, s_tools->m_hotZone->font,
                    textColor, CLICKED_ITEMS);

            // icons
            xStart += s_tools->m_hotZone->psi_labelRect.width() +
                      s_tools->m_hotZone->margin;
#ifdef Q_OS_MAC
            // fix the start of icon on mac
            xStart += s_tools->m_hotZone->margin * 4;
#else
            // fix the start of icon on linux or windows
            xStart -= iconSize;
#endif
            //"minus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);
                s_tools->m_clickableItems.emplace_back(
                        ClickableItem::DECREASE_POINT_SIZE,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            // separator
            {
                sepParam.radius = s_tools->effectiveCtx()
                                          .viewportParams.defaultPointSize /
                                  2;
                int x0 = xStart +
                         s_tools->m_hotZone->margin /*s_tools->m_hotZone->margin
                                                       / 2*/
                        ;
                int y0 = fullH - (yStart + iconSize / 2);
                sepParam.rect = QRect(x0, y0, iconSize, iconSize);
                DrawWidgets(sepParam, false);
                xStart += s_tools->m_hotZone->margin * 2;
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

                s_tools->m_clickableItems.emplace_back(
                        ClickableItem::INCREASE_POINT_SIZE,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            yStart += iconSize;
            yStart += s_tools->m_hotZone->margin;
        }

        // default line size
        {
            int xStart = s_tools->m_hotZone->topCorner.x();

            RenderText(
                    xStart,
                    yStart + offset + s_tools->m_hotZone->yTextBottomLineShift,
                    s_tools->m_hotZone->lsi_label, s_tools->m_hotZone->font,
                    textColor, CLICKED_ITEMS);

            // icons
            xStart += s_tools->m_hotZone->lsi_labelRect.width() +
                      s_tools->m_hotZone->margin;
#ifdef Q_OS_MAC
            // fix the start of icon on mac
            xStart += s_tools->m_hotZone->margin * 4;
#else
            // fix the start of icon on linux or windows
            xStart -= iconSize;
#endif
            //"minus" icon
            {
                int x0 = xStart;
                int y0 = fullH - (yStart + iconSize / 2);
                widgetParam.rect = QRect(x0, y0, iconSize, iconSize / 4);
                DrawWidgets(widgetParam, false);

                s_tools->m_clickableItems.emplace_back(
                        ClickableItem::DECREASE_LINE_WIDTH,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            // separator
            {
                sepParam.radius = s_tools->effectiveCtx()
                                          .viewportParams.defaultLineWidth /
                                  2;
                int x0 = xStart +
                         s_tools->m_hotZone->margin /*s_tools->m_hotZone->margin
                                                       / 2*/
                        ;
                int y0 = fullH - (yStart + iconSize / 2);
                sepParam.rect = QRect(x0, y0, iconSize, iconSize);
                DrawWidgets(sepParam, false);
                xStart += s_tools->m_hotZone->margin * 2;
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

                s_tools->m_clickableItems.emplace_back(
                        ClickableItem::INCREASE_LINE_WIDTH,
                        QRect(xStart, yStart, iconSize, iconSize));
                xStart += iconSize;
            }

            yStart += iconSize;
            yStart += s_tools->m_hotZone->margin;
        }
    }
}

void ecvDisplayTools::CheckIfRemove() {
    if (s_tools->m_removeAllFlag) {
        CC_DRAW_CONTEXT context;
        context.removeEntityType = ENTITY_TYPE::ECV_ALL;
        RemoveEntities(context);
        SetRemoveAllFlag(false);
    } else if (s_tools->m_removeFlag) {
        for (const removeInfo& rmInfo : s_tools->m_removeInfos) {
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
        s_tools->m_removeFlag = false;
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
        s_tools->changeEntityProperties(propertyParam);
        if (autoUpdate) {
            UpdateScreen();
        }
    }
}

void ecvDisplayTools::DrawWidgets(const WIDGETS_PARAMETER& param,
                                  bool update /* = false*/) {
    ccHObject* entity = param.entity;
    int viewport = param.viewport;
    switch (param.type) {
        case WIDGETS_TYPE::WIDGET_COORDINATE:
            ShowOrientationMarker();
            break;
        case WIDGETS_TYPE::WIDGET_BBOX:
            break;

        case WIDGETS_TYPE::WIDGET_T2D: {
            QFont textFont = s_tools->m_font;
            const_cast<WIDGETS_PARAMETER*>(&param)->fontSize =
                    textFont.pointSize();

            s_tools->drawWidgets(param);
        } break;
        case WIDGETS_TYPE::WIDGET_IMAGE:
        case WIDGETS_TYPE::WIDGET_LINE_2D:
        case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
        case WIDGETS_TYPE::WIDGET_POINTS_2D:
        case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
        case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
        case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
        case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
            s_tools->drawWidgets(param);
            break;
        case WIDGETS_TYPE::WIDGET_LINE_3D:
            if (param.lineWidget.valid) {
                s_tools->drawWidgets(param);
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

            if (s_tools->effectiveCtx().interactionFlags &
                INTERACT_TRANSFORM_ENTITIES) {
                CONTEXT.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
            }
            CONTEXT.defaultViewPort = viewport;
            poly->draw(CONTEXT);
        } break;
        case WIDGETS_TYPE::WIDGET_SPHERE:
        case WIDGETS_TYPE::WIDGET_POINT:
            s_tools->drawWidgets(param);
            break;
        case WIDGETS_TYPE::WIDGET_CAPTION:
            s_tools->drawWidgets(param);
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
        case WIDGETS_TYPE::WIDGET_SPHERE: {
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

bool ecvDisplayTools::GetClick3DPos(int x, int y, CCVector3d& P3D) {
    ccGLCameraParameters camera;
    GetGLCameraParameters(camera);

    y = s_tools->effectiveCtx().glViewport.height() - 1 - y;

    double glDepth = GetGLDepth(x, y);
    if (glDepth == 1.0) {
        return false;
    }
    CCVector3d P2D(x, y, glDepth);
    return camera.unproject(P2D, P3D);
}

void ecvDisplayTools::DrawPivot() {
    if (!s_tools->effectiveCtx().viewportParams.objectCenteredView ||
        (s_tools->effectiveCtx().pivotVisibility == PIVOT_HIDE) ||
        (s_tools->effectiveCtx().pivotVisibility == PIVOT_SHOW_ON_MOVE &&
         !s_tools->effectiveCtx().pivotSymbolShown)) {
        return;
    }

    // place origin on pivot point
    CCVector3d tranlateTartget = CCVector3d(
            s_tools->effectiveCtx().viewportParams.getPivotPoint().x,
            s_tools->effectiveCtx().viewportParams.getPivotPoint().y,
            s_tools->effectiveCtx().viewportParams.getPivotPoint().z);

    // compute actual symbol radius
    double symbolRadius =
            CC_DISPLAYED_PIVOT_RADIUS_PERCENT *
            std::min(s_tools->effectiveCtx().glViewport.width(),
                     s_tools->effectiveCtx().glViewport.height()) /
            2.0;

    // draw a small sphere
    {
        ccSphere sphere(static_cast<PointCoordinateType>(10.0 / symbolRadius));
        sphere.setColor(ecvColor::yellow);
        sphere.showColors(true);
        sphere.setVisible(true);
        sphere.setEnabled(true);
        // force lighting for proper sphere display
        CC_DRAW_CONTEXT CONTEXT;
        GetContext(CONTEXT);
        CONTEXT.drawingFlags =
                CC_DRAW_3D | CC_DRAW_FOREGROUND | CC_LIGHT_ENABLED;
        sphere.draw(CONTEXT);
    }
}

void ecvDisplayTools::SetCurrentScreen(QWidget* widget) {
    s_tools->m_currentScreen = widget;
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
    return m_primaryCtx.viewportParams;
}

void ecvDisplayTools::setViewportParameters(
        const ecvViewportParameters& params) {
    SetViewportParameters(params);
}

void ecvDisplayTools::setPerspectiveState(bool state, bool objectCenteredView) {
    SetPerspectiveState(state, objectCenteredView);
}

bool ecvDisplayTools::perspectiveView() const {
    return m_primaryCtx.viewportParams.perspectiveView;
}

bool ecvDisplayTools::objectCenteredView() const {
    return m_primaryCtx.viewportParams.objectCenteredView;
}

void ecvDisplayTools::setSceneDB(ccHObject* root) { SetSceneDB(root); }

ccHObject* ecvDisplayTools::getSceneDB() { return m_globalDBRoot; }

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
    return m_overridenDisplayParametersEnabled;
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
    SetExclusiveFullScreenFlage(state);
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
    return ExclusiveFullScreen();
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
    SetPointSize(effectiveCtx(), size);
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
