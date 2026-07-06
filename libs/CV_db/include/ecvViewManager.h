// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFont>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QPointer>
#include <QStringList>
#include <functional>
#include <unordered_set>
#include <vector>

#include "CV_db.h"
#include "ecvDrawContext.h"
#include "ecvGenericGLDisplay.h"
#include "ecvGuiParameters.h"
#include "ecvViewContext.h"

class ccHObject;
class ecvDisplayTools;
class QMainWindow;
struct removeInfo;
class ecvViewLayoutProxy;
class ecvViewRepresentation;
class ecvUndoManager;

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

    /// Returns a guaranteed non-null view context: effective view -> active
    /// view -> first view.
    ecvViewContext& resolveViewContext();
    const ecvViewContext& resolveViewContext() const;

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

    /// Returns the first registered view (the "primary" view created at
    /// startup).
    [[deprecated("All views are equivalent; use getActiveView() instead")]]
    ecvGenericGLDisplay* getPrimaryView() const;

    /// Returns true if at least one view is registered.
    bool hasAnyView() const;

    // ================================================================
    // Batch operations
    // ================================================================

    void refreshAll(bool only2D = false);
    void redrawAll(bool only2D = false,
                   bool forceRedraw = true,
                   bool includePrimary = true);

    void setRemoveAllFlag(bool flag);
    void setRedrawRecursive(bool redraw);

    void setShuttingDown(bool shuttingDown) { m_shuttingDown = shuttingDown; }
    bool isShuttingDown() const { return m_shuttingDown; }

    /// Increment the label-cache generation counter so that all widgets
    /// rescan the DB tree for labels on their next paint.
    void invalidateLabelCache() { ++m_labelCacheGeneration; }
    unsigned labelCacheGeneration() const { return m_labelCacheGeneration; }

    // ================================================================
    // Active-view dispatchers (Phase 4: replace ecvDisplayTools statics)
    // These dispatch to the effective view or all views as appropriate.
    // ================================================================

    QWidget* activeWidget() const;
    void invalidateActiveViewport();
    void deprecateActive3DLayer();
    void displayMessageOnActiveView(
            const QString& message,
            ecvGenericGLDisplay::MessagePosition pos,
            bool append = false,
            int displayMaxDelay_sec = 2,
            ecvGenericGLDisplay::MessageType type =
                    ecvGenericGLDisplay::CUSTOM_MESSAGE);

    /// Emits pickCenterOfRotation signal to enter one-shot pick mode.
    void notifyPickCenterOfRotation();

    void setRemoveViewIds(std::vector<removeInfo>& removeinfos);

    // ================================================================
    // Shared-tools forwarders (base-class fallback implementations)
    // Called by ecvGenericGLDisplay default virtual implementations to
    // avoid direct ecvDisplayTools:: references in the base class.
    // ================================================================

    void sharedMoveCamera(float dx, float dy, float dz);
    void sharedRotateBaseViewMat(const ccGLMatrixd& rotMat);
    void sharedDisplayText(const QString& text,
                           int x,
                           int y,
                           unsigned char align,
                           float bkgAlpha,
                           const unsigned char* rgbColor,
                           const QFont* font,
                           const QString& id,
                           ecvGenericGLDisplay* caller);
    void sharedLoadCameraParameters(const std::string& file);
    void sharedSaveCameraParameters(const std::string& file);
    void sharedGetContext(CC_DRAW_CONTEXT& context,
                          const ecvViewContext& viewCtx);
    void sharedSetupProjectiveViewport(const ccGLMatrixd& cameraMatrix,
                                       float fov_deg,
                                       float ar,
                                       bool viewerBasedPerspective,
                                       bool bubbleViewMode);
    static int sharedGetOptimizedFontSize(int baseFontSize);
    static bool useVtkPick();
    static bool use2D();

    // ================================================================
    // Shared display tools (replaces ecvDisplayTools singleton lifecycle)
    //
    // ecvViewManager owns the shared ecvDisplayTools instance that was
    // formerly accessed via ecvDisplayTools::TheInstance().  The instance
    // is created once at app startup and released at shutdown.
    // ================================================================

    /// Initialise the shared display tools instance.  Must be called once
    /// during application startup, before any vtkGLView is created.
    void initDisplayTools(ecvDisplayTools* tools,
                          QMainWindow* win,
                          bool stereoMode = false);

    /// Release and destroy the shared display tools.
    void releaseDisplayTools();

    /// Returns the shared display tools (may be nullptr before init).
    ecvDisplayTools* displayTools() const;

    /// Optional relay implementation (registered by VTK engine): typed
    /// QObject::connect from vtkGLView to ViewManager relay signals. Required
    /// for multi-window — CV_db cannot depend on vtkGLView headers.
    /// When unset, registerView is registered without per-view relay
    /// connections.
    using SingletonRelayHook = void (*)(ecvViewManager*, ecvGenericGLDisplay*);
    static void registerSingletonRelayHook(SingletonRelayHook hook);

    ecvUndoManager* undoManager();
    const ecvUndoManager* undoManager() const;

    // ================================================================
    // Global state (moved from ecvDisplayTools singleton)
    // ================================================================

    ccHObject* globalDBRoot() const { return m_globalDBRoot; }
    void setGlobalDBRoot(ccHObject* root) { m_globalDBRoot = root; }

    QMainWindow* mainWindow() const { return m_mainWindow; }
    void setMainWindow(QMainWindow* win) { m_mainWindow = win; }

    QFont defaultFont() const { return m_defaultFont; }
    void setDefaultFont(const QFont& font) { m_defaultFont = font; }

    bool removeFlag() const { return m_removeFlag; }
    void setRemoveFlag(bool state) { m_removeFlag = state; }

    std::vector<removeInfo>& removeInfos() { return m_removeInfos; }

    const ecvGui::ParamStruct& overriddenDisplayParameters() const;
    void setOverriddenDisplayParameters(const ecvGui::ParamStruct& params);
    bool hasOverriddenDisplayParameters() const {
        return m_overridenDisplayParametersEnabled;
    }
    /// Called before returning overridden display parameters (initializes
    /// font sizes when override is active).
    void prepareOverriddenDisplayParameters();

    // ================================================================
    // Layout persistence
    // ================================================================

    using GeometryProvider =
            std::function<QJsonObject(ecvGenericGLDisplay* view)>;
    QJsonObject saveLayout(GeometryProvider geometryOf) const;

    using LayoutApplier = std::function<void(const QJsonObject& viewJson)>;
    void restoreLayout(const QJsonObject& layout, LayoutApplier apply);

    // ================================================================
    // Entity-view association helpers
    // ================================================================

    void associateToActiveView(ccHObject* obj);
    void forceAssociateToView(ccHObject* obj, ecvGenericGLDisplay* view);

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

    // -- Relayed per-view signals from the active view (Phase 1) --
    // Consumers that want "active view" events connect here.
    // These are automatically reconnected when the active view changes.
    void entitySelectionChanged(ccHObject* entity);
    void entitiesSelectionChanged(std::unordered_set<int> entIDs);
    void newLabel(ccHObject* obj);
    void filesDropped(const QStringList& filenames, bool displayDialog);
    void cameraParamChanged();
    void mousePosChanged(const QPoint& pos);
    void pickCenterOfRotation();
    void exclusiveFullScreenToggled(bool exclusive);

    void itemPicked(ccHObject* entity,
                    unsigned subEntityID,
                    int x,
                    int y,
                    const CCVector3& P);
    void pointIndicesSelected(ccHObject* entity, const QSet<unsigned>& indices);
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);
    void leftButtonClicked(int x, int y);
    void rightButtonClicked(int x, int y);
    void doubleButtonClicked(int x, int y);
    void buttonReleased();
    void labelmove2D(int x, int y, int dx, int dy);
    void pivotPointChanged(const CCVector3d&);
    void perspectiveStateChanged();

    // -- Camera link / state signals (Phase 4) --
    // Bridged from ecvDisplayTools so consumers on ecvViewManager can
    // receive camera-state changes that originate in the rendering engine.
    void baseViewMatChanged(const ccGLMatrixd& newViewMat);
    void fovChanged(float fov);
    void zNearCoefChanged(float coef);
    void cameraPosChanged(const CCVector3d&);
    void cameraDisplaced(float ddx, float ddy);
    void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);

private:
    ecvViewManager();

    void setupSingletonRelay(ecvGenericGLDisplay* view);

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

    ecvDisplayTools* m_displayTools = nullptr;
    ecvUndoManager* m_undoManager = nullptr;

    ccHObject* m_globalDBRoot = nullptr;
    QMainWindow* m_mainWindow = nullptr;
    QFont m_defaultFont;
    bool m_removeFlag = false;
    bool m_removeAllFlag_vm = false;
    std::vector<removeInfo> m_removeInfos;
    ecvGui::ParamStruct m_overridenDisplayParameters;
    bool m_overridenDisplayParametersEnabled = false;
    unsigned m_labelCacheGeneration = 0;
    bool m_shuttingDown = false;
};
