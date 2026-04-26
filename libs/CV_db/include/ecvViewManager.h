// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonObject>
#include <QList>
#include <QObject>
#include <functional>

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

    /// Find which registered view displays the given entity.
    /// Returns nullptr if the entity has no display or its display is not registered.
    ecvGenericGLDisplay* findViewForEntity(const ccHObject* entity) const;

    // -- Batch operations --

    void refreshAll(bool only2D = false);
    /// Redraw all views.  Pass includePrimary=false when the caller
    /// already redraws the primary (ecvDisplayTools::TheInstance) view
    /// separately, to avoid double-refresh.
    void redrawAll(bool only2D = false,
                   bool forceRedraw = true,
                   bool includePrimary = true);

    // -- Layout persistence --

    /// Serialize the current layout (view list + active view ID).
    /// The caller supplies per-view geometry via the callback.
    using GeometryProvider =
            std::function<QJsonObject(ecvGenericGLDisplay* view)>;
    QJsonObject saveLayout(GeometryProvider geometryOf) const;

    /// Restore a previously saved layout.
    /// The caller creates views / applies geometry via the callback.
    using LayoutApplier =
            std::function<void(const QJsonObject& viewJson)>;
    void restoreLayout(const QJsonObject& layout, LayoutApplier apply);

    // -- Entity-view association helpers --

    /// Recursively associate entity with the active view (if unbound).
    void associateToActiveView(ccHObject* obj);

    /// Reassign all entities bound to the closing view to a surviving view.
    /// If no surviving view exists, clears the display association.
    void detachEntitiesFromView(ecvGenericGLDisplay* view);

    /// Recursively reassign entities from one view to another.
    void reassignEntitiesFromView(ccHObject* root,
                                  ecvGenericGLDisplay* fromView,
                                  ecvGenericGLDisplay* toView);

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
