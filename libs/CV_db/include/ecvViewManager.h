// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QList>
#include <QObject>

#include "CV_db.h"

class ccHObject;
class ecvGenericGLDisplay;

/// Global view manager — tracks all open display views and the active one.
///
/// Replaces stacking active-window static methods on ecvDisplayTools.
/// Inspired by ParaView's pqActiveObjects but lighter-weight.
///
/// Signals enable UI components (properties panel, toolbar, status bar)
/// to react automatically when the active view changes.
class CV_DB_LIB_API ecvViewManager : public QObject {
    Q_OBJECT

public:
    static ecvViewManager& instance();

    // -- Active view --

    ecvGenericGLDisplay* getActiveView() const;
    void setActiveView(ecvGenericGLDisplay* view);

    /// RAII helper that temporarily overrides the "effective" active view
    /// during rendering.  This ensures delegation methods
    /// (GetGLCameraParameters, etc.) return the rendering view's data
    /// rather than the UI-active view's data.
    class ScopedRenderOverride {
    public:
        explicit ScopedRenderOverride(ecvGenericGLDisplay* view)
            : m_saved(ecvViewManager::instance().m_renderingView) {
            ecvViewManager::instance().m_renderingView = view;
        }
        ~ScopedRenderOverride() {
            ecvViewManager::instance().m_renderingView = m_saved;
        }
        ScopedRenderOverride(const ScopedRenderOverride&) = delete;
        ScopedRenderOverride& operator=(const ScopedRenderOverride&) = delete;

    private:
        ecvGenericGLDisplay* m_saved;
    };

    /// Returns the rendering override if set, otherwise the UI-active view.
    ecvGenericGLDisplay* getEffectiveView() const;

    // -- View registration --

    void registerView(ecvGenericGLDisplay* view);
    void unregisterView(ecvGenericGLDisplay* view);

    // -- Query --

    const QList<ecvGenericGLDisplay*>& getAllViews() const;
    int viewCount() const;
    ecvGenericGLDisplay* findView(int uniqueID) const;

    // -- Batch operations --

    void refreshAll(bool only2D = false);
    /// Redraw all views.  Pass includePrimary=false when the caller
    /// already redraws the primary (ecvDisplayTools::TheInstance) view
    /// separately, to avoid double-refresh.
    void redrawAll(bool only2D = false,
                   bool forceRedraw = true,
                   bool includePrimary = true);

    // -- Entity-view association helpers --

    /// Recursively associate entity with the active view (if unbound).
    void associateToActiveView(ccHObject* obj);

    /// Clear display association for all entities bound to the given view.
    void detachEntitiesFromView(ecvGenericGLDisplay* view);

signals:
    void activeViewChanged(ecvGenericGLDisplay* newActive,
                           ecvGenericGLDisplay* oldActive);
    void viewRegistered(ecvGenericGLDisplay* view);
    void viewUnregistered(ecvGenericGLDisplay* view);
    void viewCountChanged(int count);

private:
    ecvViewManager();

    ecvGenericGLDisplay* m_activeView = nullptr;
    ecvGenericGLDisplay* m_renderingView = nullptr;
    QList<ecvGenericGLDisplay*> m_views;
};
