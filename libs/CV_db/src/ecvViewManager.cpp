// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewManager.h"

#include <QJsonArray>
#include <QJsonObject>

#include "ecvDisplayTools.h"
#include "ecvGenericGLDisplay.h"
#include "ecvHObject.h"
#include "ecvRepresentationManager.h"
#include "ecvViewLayoutProxy.h"
#include "ecvViewRepresentation.h"

ecvViewManager::ecvViewManager() : QObject(nullptr) {}

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

void ecvViewManager::setActiveView(ecvGenericGLDisplay* view) {
    if (m_activeView == view) return;

    ecvGenericGLDisplay* oldActive = m_activeView;
    m_activeView = view;

    updateActiveRepresentation();
    triggerSignals();

    if (m_activeView != oldActive) {
        emit activeViewChanged(m_activeView, oldActive);
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

    m_views.removeAt(idx);
    emit viewUnregistered(view);
    emit viewCountChanged(m_views.size());

    if (m_activeView == view) {
        setActiveView(m_views.isEmpty() ? nullptr : m_views.last());
    }
}

// ============================================================================
// Layout proxy management
// ============================================================================

void ecvViewManager::registerLayout(ecvViewLayoutProxy* layout) {
    if (!layout || m_layouts.contains(layout)) return;
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

// ============================================================================
// Batch operations
// ============================================================================

void ecvViewManager::refreshAll(bool only2D) {
    for (auto* view : m_views) {
        if (view) {
            ScopedRenderOverride guard(view);
            view->refresh(only2D);
        }
    }
}

void ecvViewManager::redrawAll(bool only2D,
                               bool forceRedraw,
                               bool includePrimary) {
    auto* primary = ecvDisplayTools::TheInstance();
    for (auto* view : m_views) {
        if (!view) continue;
        if (!includePrimary && view == primary) continue;
        ScopedRenderOverride guard(view);
        view->redraw(only2D, forceRedraw);
    }
}

// ============================================================================
// Entity-view association
// ============================================================================

void ecvViewManager::associateToActiveView(ccHObject* obj) {
    if (!obj || !m_activeView) return;
    if (!obj->getDisplay()) {
        obj->setDisplay_recursive(m_activeView);
    }
}

void ecvViewManager::detachEntitiesFromView(ecvGenericGLDisplay* closingView) {
    if (!closingView) return;

    ccHObject* sceneDB = ecvDisplayTools::GetSceneDB();
    if (!sceneDB) return;

    ecvGenericGLDisplay* recipient = nullptr;
    for (int i = m_views.size() - 1; i >= 0; --i) {
        if (m_views[i] != closingView) {
            recipient = m_views[i];
            break;
        }
    }

    reassignEntitiesFromView(sceneDB, closingView, recipient);
}

void ecvViewManager::reassignEntitiesFromView(
        ccHObject* root,
        ecvGenericGLDisplay* fromView,
        ecvGenericGLDisplay* toView) {
    if (!root) return;

    if (root->getDisplay() == fromView) {
        if (toView) {
            root->setDisplay(toView);
        } else {
            root->removeFromDisplay(fromView);
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
    auto* primary = ecvDisplayTools::TheInstance();

    for (auto* view : m_views) {
        if (!view) continue;
        QJsonObject vObj;
        vObj["id"] = view->getUniqueID();
        vObj["title"] = view->getTitle();
        vObj["is_primary"] = (view == primary);

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
    layout["active_view_id"] =
            m_activeView ? m_activeView->getUniqueID() : -1;
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
