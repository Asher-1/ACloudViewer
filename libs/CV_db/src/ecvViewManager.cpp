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

ecvViewManager::ecvViewManager() : QObject(nullptr) {}

ecvViewManager& ecvViewManager::instance() {
    static ecvViewManager s_instance;
    return s_instance;
}

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

    emit activeViewChanged(m_activeView, oldActive);
}

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

    // Find a surviving view to receive orphaned entities (prefer the one
    // that will become active — last registered, excluding the closing one).
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
