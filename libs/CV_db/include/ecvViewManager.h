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
#include <QPointer>
#include <functional>

#include "CV_db.h"

class ccHObject;
class ecvGenericGLDisplay;
class ecvViewLayoutProxy;
class ecvViewRepresentation;

/// Global active-objects coordinator — tracks active view, source,
/// representation, and all registered views.
///
/// Mirrors ParaView's pqActiveObjects:
///   - Active view/source/representation tracking
///   - Signal batching via triggerSignals() (emit only on actual changes)
///   - Layout proxy management
///   - Event-driven: UI components subscribe to signals
///
/// Also replaces the stacking active-window static methods on ecvDisplayTools.
class CV_DB_LIB_API ecvViewManager : public QObject {
    Q_OBJECT

public:
    static ecvViewManager& instance();

    // ================================================================
    // Active view (ParaView pqActiveObjects::activeView)
    // ================================================================

    ecvGenericGLDisplay* getActiveView() const;
    void setActiveView(ecvGenericGLDisplay* view);

    /// RAII helper that temporarily overrides the "effective" active view
    /// during rendering.
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

    // ================================================================
    // Active source / representation (ParaView pqActiveObjects pattern)
    // ================================================================

    ccHObject* activeSource() const;
    void setActiveSource(ccHObject* source);

    ecvViewRepresentation* activeRepresentation() const;

    // ================================================================
    // View registration
    // ================================================================

    void registerView(ecvGenericGLDisplay* view);
    void unregisterView(ecvGenericGLDisplay* view);

    // ================================================================
    // Layout proxy management (ParaView layout registration)
    // ================================================================

    void registerLayout(ecvViewLayoutProxy* layout);
    void unregisterLayout(ecvViewLayoutProxy* layout);
    const QList<ecvViewLayoutProxy*>& allLayouts() const;
    ecvViewLayoutProxy* activeLayout() const;

    // ================================================================
    // Query
    // ================================================================

    const QList<ecvGenericGLDisplay*>& getAllViews() const;
    int viewCount() const;
    ecvGenericGLDisplay* findView(int uniqueID) const;
    ecvGenericGLDisplay* findViewForEntity(const ccHObject* entity) const;

    // ================================================================
    // Batch operations
    // ================================================================

    void refreshAll(bool only2D = false);
    void redrawAll(bool only2D = false,
                   bool forceRedraw = true,
                   bool includePrimary = true);

    // ================================================================
    // Layout persistence
    // ================================================================

    using GeometryProvider =
            std::function<QJsonObject(ecvGenericGLDisplay* view)>;
    QJsonObject saveLayout(GeometryProvider geometryOf) const;

    using LayoutApplier =
            std::function<void(const QJsonObject& viewJson)>;
    void restoreLayout(const QJsonObject& layout, LayoutApplier apply);

    // ================================================================
    // Entity-view association helpers
    // ================================================================

    void associateToActiveView(ccHObject* obj);

    /// Move an entity (and its children) from its current view to a target
    /// view. Removes VTK representations from the old view and redraws both.
    /// ParaView equivalent: pqActiveObjects + representation visibility toggle.
    void moveEntityToView(ccHObject* obj, ecvGenericGLDisplay* targetView);

    void detachEntitiesFromView(ecvGenericGLDisplay* view);
    void reassignEntitiesFromView(ccHObject* root,
                                  ecvGenericGLDisplay* fromView,
                                  ecvGenericGLDisplay* toView);

signals:
    // ParaView pqActiveObjects signal set
    void activeViewChanged(ecvGenericGLDisplay* newActive,
                           ecvGenericGLDisplay* oldActive);
    void activeSourceChanged(ccHObject* source);
    void activeRepresentationChanged(ecvViewRepresentation* repr);
    void activeLayoutChanged(ecvViewLayoutProxy* layout);

    void viewRegistered(ecvGenericGLDisplay* view);
    void viewUnregistered(ecvGenericGLDisplay* view);
    void viewCountChanged(int count);

    void layoutRegistered(ecvViewLayoutProxy* layout);
    void layoutUnregistered(ecvViewLayoutProxy* layout);

private:
    ecvViewManager();

    /// ParaView triggerSignals pattern: compare cached vs current, emit only
    /// on actual change.
    void triggerSignals();
    void updateActiveRepresentation();

    ecvGenericGLDisplay* m_activeView = nullptr;
    ecvGenericGLDisplay* m_renderingView = nullptr;
    ccHObject* m_activeSource = nullptr;
    ecvViewRepresentation* m_activeRepresentation = nullptr;

    // Cached values for triggerSignals
    ecvGenericGLDisplay* m_cachedView = nullptr;
    ccHObject* m_cachedSource = nullptr;
    ecvViewRepresentation* m_cachedRepresentation = nullptr;

    QList<ecvGenericGLDisplay*> m_views;
    QList<ecvViewLayoutProxy*> m_layouts;
};
