// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvViewManager.h"

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

    m_views.removeAt(idx);
    emit viewUnregistered(view);
    emit viewCountChanged(m_views.size());

    if (m_activeView == view) {
        setActiveView(m_views.isEmpty() ? nullptr : m_views.first());
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

void ecvViewManager::refreshAll(bool only2D) {
    for (auto* view : m_views) {
        if (view) view->refresh(only2D);
    }
}

void ecvViewManager::redrawAll(bool only2D,
                               bool forceRedraw,
                               bool includePrimary) {
    auto* primary = ecvDisplayTools::TheInstance();
    for (auto* view : m_views) {
        if (!view) continue;
        if (!includePrimary && view == primary) continue;
        view->redraw(only2D, forceRedraw);
    }
}

void ecvViewManager::associateToActiveView(ccHObject* obj) {
    if (!obj || !m_activeView) return;
    if (!obj->getDisplay()) {
        obj->setDisplay_recursive(m_activeView);
    }
}

void ecvViewManager::detachEntitiesFromView(ecvGenericGLDisplay* view) {
    // This will be called when a view is closed.
    // Entities bound to this view should have their display cleared.
    // The actual recursive walk is done by the caller passing in
    // the DB root, since ecvViewManager doesn't hold the DB root.
    Q_UNUSED(view);
}
