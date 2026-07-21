// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewManager.h"

#include <QJsonArray>
#include <QJsonObject>
#include <algorithm>

#include "ecvDisplayTools.h"
#include "ecvDrawContext.h"
#include "ecvGenericGLDisplay.h"
#include "ecvHObject.h"
#include "ecvRepresentationManager.h"
#include "ecvUndoManager.h"
#include "ecvViewContext.h"
#include "ecvViewLayoutProxy.h"
#include "ecvViewRepresentation.h"

namespace {
ecvViewManager::SingletonRelayHook s_singletonRelayHook = nullptr;
}

ecvViewManager::ecvViewManager() : QObject(nullptr) {
    m_undoManager = new ecvUndoManager(this);
}

void ecvViewManager::registerSingletonRelayHook(SingletonRelayHook hook) {
    s_singletonRelayHook = hook;
}

void ecvViewManager::setupSingletonRelay(ecvGenericGLDisplay* view) {
    // Typed PMF connections are registered from the VTK layer
    // (registerViewManagerTypedRelay) because CV_db cannot include vtkGLView.
    if (s_singletonRelayHook) s_singletonRelayHook(this, view);
}

ecvViewManager& ecvViewManager::instance() {
    static ecvViewManager s_instance;
    return s_instance;
}

// ============================================================================
// Active view
// ============================================================================

ecvGenericGLDisplay* ecvViewManager::getActiveView() const {
    return m_activeView;
}

ecvGenericGLDisplay* ecvViewManager::getEffectiveView() const {
    return m_renderingView ? m_renderingView : m_activeView;
}

ecvViewContext& ecvViewManager::resolveViewContext() {
    auto* view = getEffectiveView();
    if (view && view->viewContext()) return *view->viewContext();

    auto* active = getActiveView();
    if (active && active->viewContext()) return *active->viewContext();

    if (!m_views.isEmpty()) {
        auto* first = m_views.first();
        if (first && first->viewContext()) return *first->viewContext();
    }

    Q_ASSERT_X(false, "resolveViewContext", "No view context available");
    static ecvViewContext s_emergency;
    return s_emergency;
}

const ecvViewContext& ecvViewManager::resolveViewContext() const {
    auto* view = getEffectiveView();
    if (view && view->viewContext()) return *view->viewContext();

    auto* active = getActiveView();
    if (active && active->viewContext()) return *active->viewContext();

    if (!m_views.isEmpty()) {
        auto* first = m_views.first();
        if (first && first->viewContext()) return *first->viewContext();
    }

    Q_ASSERT_X(false, "resolveViewContext", "No view context available");
    static ecvViewContext s_emergency;
    return s_emergency;
}

void ecvViewManager::setActiveView(ecvGenericGLDisplay* view) {
    if (m_activeView == view) return;

    ecvGenericGLDisplay* oldActive = m_activeView;
    ecvViewLayoutProxy* oldLayout = activeLayout();

    m_activeView = view;

    updateActiveRepresentation();
    triggerSignals();

    if (m_activeView != oldActive) {
        emit activeViewChanged(m_activeView, oldActive);
    }

    ecvViewLayoutProxy* newLayout = activeLayout();
    if (newLayout != oldLayout) {
        emit activeLayoutChanged(newLayout);
    }
}

// ============================================================================
// Active source / representation (ParaView pqActiveObjects pattern)
// ============================================================================

ccHObject* ecvViewManager::activeSource() const { return m_activeSource; }

void ecvViewManager::setActiveSource(ccHObject* source) {
    if (m_activeSource == source) return;
    m_activeSource = source;
    updateActiveRepresentation();
    triggerSignals();
}

ecvViewRepresentation* ecvViewManager::activeRepresentation() const {
    return m_activeRepresentation;
}

void ecvViewManager::updateActiveRepresentation() {
    ecvViewRepresentation* repr = nullptr;
    if (m_activeSource && m_activeView) {
        repr = ecvRepresentationManager::instance().getRepresentation(
                m_activeSource, m_activeView);
    }
    m_activeRepresentation = repr;
}

void ecvViewManager::triggerSignals() {
    if (signalsBlocked()) return;

    if (m_cachedView != m_activeView) {
        m_cachedView = m_activeView;
    }

    if (m_cachedSource != m_activeSource) {
        m_cachedSource = m_activeSource;
        emit activeSourceChanged(m_activeSource);
    }

    if (m_cachedRepresentation != m_activeRepresentation) {
        m_cachedRepresentation = m_activeRepresentation;
        emit activeRepresentationChanged(m_activeRepresentation);
    }
}

// ============================================================================
// View registration
// ============================================================================

void ecvViewManager::registerView(ecvGenericGLDisplay* view) {
    if (!view || m_views.contains(view)) return;

    setupSingletonRelay(view);

    m_views.append(view);
    emit viewRegistered(view);
    emit viewCountChanged(m_views.size());

    if (!m_activeView) {
        setActiveView(view);
    }
}

void ecvViewManager::unregisterView(ecvGenericGLDisplay* view) {
    if (!view) return;

    int idx = m_views.indexOf(view);
    if (idx < 0) return;

    detachEntitiesFromView(view);

    // Disconnect per-view signal relay.
    if (auto* src = dynamic_cast<QObject*>(view)) {
        disconnect(src, nullptr, this, nullptr);
    }

    m_views.removeAt(idx);
    emit viewUnregistered(view);
    emit viewCountChanged(m_views.size());

    if (m_activeView == view) {
        ecvGenericGLDisplay* replacement = nullptr;
        if (!m_views.isEmpty()) {
            for (auto* layout : m_layouts) {
                if (!layout) continue;
                for (auto* v : m_views) {
                    if (v != view && layout->containsView(v)) {
                        replacement = v;
                        break;
                    }
                }
                if (replacement) break;
            }
            if (!replacement) {
                replacement = m_views.first();
            }
        }
        setActiveView(replacement);
    }
}

// ============================================================================
// Layout proxy management
// ============================================================================

void ecvViewManager::registerLayout(ecvViewLayoutProxy* layout) {
    if (!layout || m_layouts.contains(layout)) return;
    layout->setUndoManager(m_undoManager);
    m_layouts.append(layout);
    emit layoutRegistered(layout);
}

void ecvViewManager::unregisterLayout(ecvViewLayoutProxy* layout) {
    if (!layout) return;
    int idx = m_layouts.indexOf(layout);
    if (idx < 0) return;
    m_layouts.removeAt(idx);
    emit layoutUnregistered(layout);
}

const QList<ecvViewLayoutProxy*>& ecvViewManager::allLayouts() const {
    return m_layouts;
}

ecvViewLayoutProxy* ecvViewManager::activeLayout() const {
    if (!m_activeView) return nullptr;
    for (auto* layout : m_layouts) {
        if (layout->containsView(m_activeView)) return layout;
    }
    return m_layouts.isEmpty() ? nullptr : m_layouts.last();
}

// ============================================================================
// Query
// ============================================================================

const QList<ecvGenericGLDisplay*>& ecvViewManager::getAllViews() const {
    return m_views;
}

int ecvViewManager::viewCount() const { return m_views.size(); }

ecvGenericGLDisplay* ecvViewManager::findView(int uniqueID) const {
    for (auto* view : m_views) {
        if (view && view->getUniqueID() == uniqueID) {
            return view;
        }
    }
    return nullptr;
}

ecvGenericGLDisplay* ecvViewManager::findViewForEntity(
        const ccHObject* entity) const {
    if (!entity) return nullptr;
    auto* display = entity->getDisplay();
    if (!display) return nullptr;
    for (auto* view : m_views) {
        if (view == display) return view;
    }
    return nullptr;
}

ecvGenericGLDisplay* ecvViewManager::getPrimaryView() const {
    return m_views.isEmpty() ? nullptr : m_views.first();
}

bool ecvViewManager::hasAnyView() const { return !m_views.isEmpty(); }

// ============================================================================
// Batch operations
// ============================================================================

void ecvViewManager::refreshAll(bool only2D) {
    if (m_shuttingDown) return;
    for (auto* view : m_views) {
        if (!view) continue;
        QWidget* w = view->asWidget();
        if (w && !w->isVisible()) continue;
        ScopedRenderOverride guard(view);
        view->refresh(only2D);
    }
}

void ecvViewManager::redrawAll(bool only2D,
                               bool forceRedraw,
                               bool /*includePrimary*/) {
    if (m_shuttingDown) return;
    for (auto* view : m_views) {
        if (!view) continue;
        // Skip views whose widget is hidden (e.g. inactive QTabWidget
        // pages) — they will be redrawn when the tab is next shown.
        QWidget* w = view->asWidget();
        if (w && !w->isVisible()) continue;
        ScopedRenderOverride guard(view);
        view->redraw(only2D, forceRedraw);
    }
}

// ============================================================================
// Active-view dispatchers (Phase 4)
// ============================================================================

QWidget* ecvViewManager::activeWidget() const {
    auto* view = getEffectiveView();
    return view ? view->asWidget() : nullptr;
}

void ecvViewManager::invalidateActiveViewport() {
    auto* view = getEffectiveView();
    if (view) view->invalidateViewport();
}

void ecvViewManager::deprecateActive3DLayer() {
    auto* view = getEffectiveView();
    if (view) view->deprecate3DLayer();
}

void ecvViewManager::displayMessageOnActiveView(
        const QString& message,
        ecvGenericGLDisplay::MessagePosition pos,
        bool append,
        int displayMaxDelay_sec,
        ecvGenericGLDisplay::MessageType type) {
    auto* view = getEffectiveView();
    if (view)
        view->displayNewMessage(message, pos, append, displayMaxDelay_sec,
                                type);
}

void ecvViewManager::notifyPickCenterOfRotation() {
    emit pickCenterOfRotation();
}

void ecvViewManager::setRemoveAllFlag(bool flag) {
    m_removeAllFlag_vm = flag;
    if (m_displayTools) {
        m_displayTools->m_removeAllFlag = flag;
    }
}

void ecvViewManager::setRedrawRecursive(bool redraw) {
    ecvDisplayTools::SetRedrawRecursive(redraw);
}

void ecvViewManager::setRemoveViewIds(std::vector<removeInfo>& removeinfos) {
    if (!removeinfos.empty()) {
        m_removeInfos = removeinfos;
        setRemoveFlag(true);
        if (m_displayTools) {
            m_displayTools->m_removeInfos = removeinfos;
            m_displayTools->m_removeFlag = true;
        }
    } else {
        setRemoveFlag(false);
        m_removeInfos.clear();
        if (m_displayTools) {
            m_displayTools->m_removeFlag = false;
        }
    }
}

// ============================================================================
// Shared-tools forwarders (for base-class fallback implementations)
// ============================================================================

void ecvViewManager::sharedMoveCamera(float dx, float dy, float dz) {
    ecvDisplayTools::MoveCamera(dx, dy, dz);
}

void ecvViewManager::sharedRotateBaseViewMat(const ccGLMatrixd& rotMat) {
    ecvDisplayTools::RotateBaseViewMat(rotMat);
}

void ecvViewManager::sharedDisplayText(const QString& text,
                                       int x,
                                       int y,
                                       unsigned char align,
                                       float bkgAlpha,
                                       const unsigned char* rgbColor,
                                       const QFont* font,
                                       const QString& id,
                                       ecvGenericGLDisplay* caller) {
    ecvDisplayTools::DisplayText(text, x, y, align, bkgAlpha, rgbColor, font,
                                 id, caller);
}

void ecvViewManager::sharedLoadCameraParameters(const std::string& file) {
    ecvDisplayTools::LoadCameraParameters(file);
}

void ecvViewManager::sharedSaveCameraParameters(const std::string& file) {
    ecvDisplayTools::SaveCameraParameters(file);
}

void ecvViewManager::sharedGetContext(CC_DRAW_CONTEXT& context,
                                      const ecvViewContext& viewCtx) {
    ecvDisplayTools::GetContext(context, viewCtx);
}

void ecvViewManager::sharedSetupProjectiveViewport(
        const ccGLMatrixd& cameraMatrix,
        float fov_deg,
        float ar,
        bool viewerBasedPerspective,
        bool bubbleViewMode) {
    ecvDisplayTools::SetupProjectiveViewport(
            cameraMatrix, fov_deg, ar, viewerBasedPerspective, bubbleViewMode);
}

int ecvViewManager::sharedGetOptimizedFontSize(int baseFontSize) {
    return ecvDisplayTools::GetOptimizedFontSize(baseFontSize);
}

bool ecvViewManager::useVtkPick() { return ecvDisplayTools::USE_VTK_PICK; }

bool ecvViewManager::use2D() { return ecvDisplayTools::USE_2D; }

// ============================================================================
// Shared display tools lifecycle
// ============================================================================

void ecvViewManager::initDisplayTools(ecvDisplayTools* tools,
                                      QMainWindow* win,
                                      bool stereoMode) {
    if (m_displayTools) {
        assert(false && "Display tools already initialized");
        return;
    }
    m_displayTools = tools;

    m_displayTools->initializeEngine(win, stereoMode);
    if (tools) {
        m_globalDBRoot = tools->m_globalDBRoot;
        m_mainWindow = tools->m_win;
        m_defaultFont = tools->m_font;
    }
}

const ecvGui::ParamStruct& ecvViewManager::overriddenDisplayParameters() const {
    if (m_overridenDisplayParametersEnabled)
        return m_overridenDisplayParameters;
    return ecvGui::Parameters();
}

void ecvViewManager::setOverriddenDisplayParameters(
        const ecvGui::ParamStruct& params) {
    m_overridenDisplayParameters = params;
    m_overridenDisplayParametersEnabled = true;
}

void ecvViewManager::prepareOverriddenDisplayParameters() {
    if (m_overridenDisplayParametersEnabled) {
        m_overridenDisplayParameters.initFontSizesIfNeeded();
    }
}

void ecvViewManager::releaseDisplayTools() {
    if (!m_displayTools) return;

    unregisterView(m_displayTools);
    delete m_displayTools;
    m_displayTools = nullptr;
}

ecvDisplayTools* ecvViewManager::displayTools() const { return m_displayTools; }

// ============================================================================
// Entity-view association
// ============================================================================

void ecvViewManager::associateToActiveView(ccHObject* obj) {
    if (!obj || !m_activeView) return;
    if (!obj->getDisplay()) {
        obj->setDisplay_recursive(m_activeView);
    }
}

void ecvViewManager::forceAssociateToView(ccHObject* obj,
                                          ecvGenericGLDisplay* view) {
    if (!obj || !view) return;
    obj->setDisplay_recursive(view);
}

void ecvViewManager::moveEntityToView(ccHObject* obj,
                                      ecvGenericGLDisplay* targetView) {
    if (!obj || !targetView) return;

    ecvGenericGLDisplay* oldView = obj->getDisplay();
    if (oldView == targetView) return;

    if (oldView) {
        std::function<void(ccHObject*)> removeRecursive =
                [&](ccHObject* entity) {
                    if (!entity) return;

                    const QString viewId = entity->getViewId();
                    const QString bboxId = QString("BBox-") + viewId;

                    CC_DRAW_CONTEXT ctx;
                    ctx.removeViewID = viewId;
                    ctx.removeEntityType = entity->getEntityType();
                    ctx.display = oldView;
                    oldView->removeEntities(ctx);

                    CC_DRAW_CONTEXT bbCtx;
                    bbCtx.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
                    bbCtx.removeViewID = bboxId;
                    bbCtx.display = oldView;
                    oldView->removeEntities(bbCtx);

                    for (unsigned i = 0; i < entity->getChildrenNumber(); ++i) {
                        removeRecursive(entity->getChild(i));
                    }
                };
        removeRecursive(obj);

        std::function<void(ccHObject*)> removeReps = [&](ccHObject* entity) {
            if (!entity) return;
            ecvRepresentationManager::instance().removeRepresentation(entity,
                                                                      oldView);
            for (unsigned i = 0; i < entity->getChildrenNumber(); ++i) {
                removeReps(entity->getChild(i));
            }
        };
        removeReps(obj);
    }

    obj->setDisplay_recursive(targetView);
    obj->setForceRedrawRecursive(true);

    if (oldView) oldView->redraw(false, true);
    targetView->redraw(false, true);
}

void ecvViewManager::detachEntityFromView(ccHObject* obj,
                                          ecvGenericGLDisplay* fromView) {
    if (!obj || !fromView) return;

    std::function<void(ccHObject*)> removeRecursive = [&](ccHObject* entity) {
        if (!entity) return;
        const QString viewId = entity->getViewId();
        CC_DRAW_CONTEXT ctx;
        ctx.removeViewID = viewId;
        ctx.removeEntityType = entity->getEntityType();
        ctx.display = fromView;
        fromView->removeEntities(ctx);

        CC_DRAW_CONTEXT bbCtx;
        bbCtx.removeEntityType = ENTITY_TYPE::ECV_SHAPE;
        bbCtx.removeViewID = QString("BBox-") + viewId;
        bbCtx.display = fromView;
        fromView->removeEntities(bbCtx);

        for (unsigned i = 0; i < entity->getChildrenNumber(); ++i) {
            removeRecursive(entity->getChild(i));
        }
    };
    removeRecursive(obj);
}

void ecvViewManager::detachEntitiesFromView(ecvGenericGLDisplay* closingView) {
    if (!closingView) return;

    // Scan scene DB if available
    ccHObject* sceneDB = closingView->getSceneDB();
    if (sceneDB) {
        reassignEntitiesFromView(sceneDB, closingView, nullptr);
    }

    // Also scan the global/shared scene DB to catch entities that reference
    // this view (e.g., comparative sub-views where sceneDB may be null but
    // entities in the main DB still hold this view in their display set).
    for (auto* otherView : m_views) {
        if (!otherView || otherView == closingView) continue;
        ccHObject* otherDB = otherView->getSceneDB();
        if (otherDB && otherDB != sceneDB) {
            reassignEntitiesFromView(otherDB, closingView, nullptr);
        }
    }

    // Also clean the ownDB of the closing view itself
    ccHObject* ownDB = closingView->getOwnDB();
    if (ownDB) {
        reassignEntitiesFromView(ownDB, closingView, nullptr);
    }

    for (auto* v : m_views) {
        if (v && v != closingView) {
            v->redraw(false, true);
        }
    }
}

void ecvViewManager::reassignEntitiesFromView(ccHObject* root,
                                              ecvGenericGLDisplay* fromView,
                                              ecvGenericGLDisplay* toView) {
    if (!root) return;

    const auto& displays = root->getDisplays();
    bool boundToFromView = std::find(displays.begin(), displays.end(),
                                     fromView) != displays.end();

    if (boundToFromView) {
        root->removeFromDisplay(fromView);
        if (toView) {
            root->addToDisplay(toView);
        }
    }

    for (unsigned i = 0; i < root->getChildrenNumber(); ++i) {
        reassignEntitiesFromView(root->getChild(i), fromView, toView);
    }
}

// ============================================================================
// Layout persistence
// ============================================================================

QJsonObject ecvViewManager::saveLayout(GeometryProvider geometryOf) const {
    QJsonArray viewsArr;
    auto* firstView = m_views.isEmpty() ? nullptr : m_views.first();

    for (auto* view : m_views) {
        if (!view) continue;
        QJsonObject vObj;
        vObj["id"] = view->getUniqueID();
        vObj["title"] = view->getTitle();
        vObj["is_primary"] = (view == firstView);

        if (geometryOf) {
            QJsonObject geom = geometryOf(view);
            if (!geom.isEmpty()) {
                vObj["geometry"] = geom;
            }
        }
        viewsArr.append(vObj);
    }

    QJsonObject layout;
    layout["views"] = viewsArr;
    layout["active_view_id"] = m_activeView ? m_activeView->getUniqueID() : -1;
    layout["view_count"] = m_views.size();

    QJsonArray layoutsArr;
    for (auto* lp : m_layouts) {
        if (lp) layoutsArr.append(lp->saveState());
    }
    layout["layout_proxies"] = layoutsArr;

    return layout;
}

void ecvViewManager::restoreLayout(const QJsonObject& layout,
                                   LayoutApplier apply) {
    if (!apply) return;

    QJsonArray viewsArr = layout["views"].toArray();
    for (const auto& val : viewsArr) {
        QJsonObject vObj = val.toObject();
        apply(vObj);
    }

    int activeId = layout["active_view_id"].toInt(-1);
    if (activeId >= 0) {
        if (auto* view = findView(activeId)) {
            setActiveView(view);
        }
    }
}

ecvUndoManager* ecvViewManager::undoManager() { return m_undoManager; }

const ecvUndoManager* ecvViewManager::undoManager() const {
    return m_undoManager;
}
