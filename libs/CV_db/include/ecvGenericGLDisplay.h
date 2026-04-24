// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFlags>
#include <QRect>
#include <QString>

#include "CV_db.h"
#include "ecvGuiParameters.h"

class ccBBox;
class ccHObject;
class ccDrawableObject;
class QWidget;
class ecvViewportParameters;
struct ccGLCameraParameters;
struct ccGLDrawContext;
struct ecvViewContext;

/// Per-window display interface for multi-window rendering.
///
/// Each 3D view window implements this interface, holding its own
/// viewport parameters, camera state, and scene/window DB roots.
/// The global ecvDisplayTools singleton delegates to the "active"
/// ecvGenericGLDisplay instance so that existing static call sites
/// keep working while new code can address a specific window.
///
/// Design references:
///   CloudCompare  ccGenericGLDisplay / ccGLWindowInterface
///   ParaView      pqView / pqRenderView
///   MeshLab       GLArea
class CV_DB_LIB_API ecvGenericGLDisplay {
public:
    virtual ~ecvGenericGLDisplay() = default;

    // ================================================================
    // Enums — shared across all display implementations.
    // Previously defined in ecvDisplayTools; moved here so the
    // per-view interface can reference them without a circular
    // dependency.  Existing code using ecvDisplayTools::PICKING_MODE
    // etc. still compiles because C++ resolves inherited names.
    // ================================================================

    enum PICKING_MODE {
        NO_PICKING,
        ENTITY_PICKING,
        ENTITY_RECT_PICKING,
        FAST_PICKING,
        POINT_PICKING,
        TRIANGLE_PICKING,
        POINT_OR_TRIANGLE_PICKING,
        POINT_OR_TRIANGLE_OR_LABEL_PICKING,
        LABEL_PICKING,
        DEFAULT_PICKING,
    };

    enum INTERACTION_FLAG {
        INTERACT_NONE = 0,
        INTERACT_ROTATE = 1,
        INTERACT_PAN = 2,
        INTERACT_CTRL_PAN = 4,
        INTERACT_ZOOM_CAMERA = 8,
        INTERACT_2D_ITEMS = 16,
        INTERACT_CLICKABLE_ITEMS = 32,
        INTERACT_TRANSFORM_ENTITIES = 64,
        INTERACT_SIG_RB_CLICKED = 128,
        INTERACT_SIG_LB_CLICKED = 256,
        INTERACT_SIG_MOUSE_MOVED = 512,
        INTERACT_SIG_BUTTON_RELEASED = 1024,
        INTERACT_SIG_MB_CLICKED = 2048,
        INTERACT_SEND_ALL_SIGNALS =
            INTERACT_SIG_RB_CLICKED | INTERACT_SIG_LB_CLICKED |
            INTERACT_SIG_MB_CLICKED | INTERACT_SIG_MOUSE_MOVED |
            INTERACT_SIG_BUTTON_RELEASED,
        MODE_PAN_ONLY =
            INTERACT_PAN | INTERACT_ZOOM_CAMERA | INTERACT_2D_ITEMS |
            INTERACT_CLICKABLE_ITEMS,
        MODE_TRANSFORM_CAMERA = INTERACT_ROTATE | MODE_PAN_ONLY,
        MODE_TRANSFORM_ENTITIES =
            INTERACT_ROTATE | INTERACT_PAN | INTERACT_ZOOM_CAMERA |
            INTERACT_TRANSFORM_ENTITIES | INTERACT_CLICKABLE_ITEMS,
    };
    Q_DECLARE_FLAGS(INTERACTION_FLAGS, INTERACTION_FLAG)

    enum MessagePosition {
        LOWER_LEFT_MESSAGE,
        UPPER_CENTER_MESSAGE,
        SCREEN_CENTER_MESSAGE,
    };

    enum MessageType {
        CUSTOM_MESSAGE,
        SCREEN_SIZE_MESSAGE,
        PERSPECTIVE_STATE_MESSAGE,
        SUN_LIGHT_STATE_MESSAGE,
        CUSTOM_LIGHT_STATE_MESSAGE,
        MANUAL_TRANSFORMATION_MESSAGE,
        MANUAL_SEGMENTATION_MESSAGE,
        ROTAION_LOCK_MESSAGE,
        FULL_SCREEN_MESSAGE,
    };

    enum PivotVisibility {
        PIVOT_HIDE,
        PIVOT_SHOW_ON_MOVE,
        PIVOT_ALWAYS_SHOW,
    };

    // ================================================================
    // Identification
    // ================================================================

    virtual int getUniqueID() const = 0;
    virtual QString getTitle() const = 0;

    // ================================================================
    // Refresh / redraw
    // ================================================================

    virtual void redraw(bool only2D = false, bool forceRedraw = true) = 0;
    virtual void refresh(bool only2D = false) = 0;
    virtual void toBeRefreshed() = 0;

    // ================================================================
    // Viewport / camera
    // ================================================================

    virtual const ecvViewportParameters& getViewportParameters() const = 0;
    virtual void setViewportParameters(const ecvViewportParameters& params) = 0;
    virtual void setPerspectiveState(bool state, bool objectCenteredView) = 0;
    virtual bool perspectiveView() const = 0;
    virtual bool objectCenteredView() const = 0;

    virtual void getGLCameraParameters(ccGLCameraParameters& params) const;
    virtual void getVisibleObjectsBB(ccBBox& box) const;
    virtual void updateConstellationCenterAndZoom(const ccBBox* box = nullptr);

    virtual QRect getGLViewport() const;
    virtual int glWidth() const;
    virtual int glHeight() const;
    virtual int getDevicePixelRatio() const;

    // ================================================================
    // Scene / own DB
    // ================================================================

    virtual void setSceneDB(ccHObject* root) = 0;
    virtual ccHObject* getSceneDB() = 0;
    virtual ccHObject* getOwnDB() = 0;
    virtual void addToOwnDB(ccHObject* obj, bool noDependency = true) = 0;
    virtual void removeFromOwnDB(ccHObject* obj) = 0;

    // ================================================================
    // Qt widget bridge
    // ================================================================

    virtual QWidget* asWidget() = 0;
    virtual const QWidget* asWidget() const = 0;

    // ================================================================
    // Interaction / picking (per-view)
    // ================================================================

    virtual void setInteractionMode(INTERACTION_FLAGS flags);
    virtual INTERACTION_FLAGS getInteractionMode() const;
    virtual void setPickingMode(PICKING_MODE mode);
    virtual PICKING_MODE getPickingMode() const;

    // ================================================================
    // Display context / parameters (per-view)
    // ================================================================

    virtual void getContext(ccGLDrawContext& context) const;
    virtual const ecvGui::ParamStruct& getDisplayParameters() const;
    virtual void setDisplayParameters(const ecvGui::ParamStruct& params,
                                      bool thisWindowOnly = false);
    virtual bool hasOverriddenDisplayParameters() const = 0;

    // ================================================================
    // HotZone / clickable items
    // ================================================================

    virtual void drawClickableItems(int xStart, int& yStart);

    // ================================================================
    // Per-view state synchronization (multi-window)
    //
    // Called by ecvViewManager when the active view changes.
    // Secondary views (ecvGLView) override these to sync their
    // per-view state with the ecvDisplayTools singleton.
    // The primary view (ecvDisplayTools itself) keeps defaults (no-op).
    // ================================================================

    virtual void pushStateToSingleton() {}
    virtual void pullStateFromSingleton() {}

    /// Access the per-view state container.
    /// Returns nullptr for the primary/singleton display (state lives in
    /// ecvDisplayTools members).  ecvGLView overrides to return &m_ctx.
    virtual ecvViewContext* viewContext() { return nullptr; }
    virtual const ecvViewContext* viewContext() const { return nullptr; }

    // ================================================================
    // Lifecycle notification (ref: CloudCompare aboutToBeRemoved)
    // ================================================================

    virtual void aboutToBeRemoved(ccDrawableObject* /*obj*/) {}

    // ================================================================
    // Static registry: QWidget* -> ecvGenericGLDisplay*
    // ================================================================

    static ecvGenericGLDisplay* FromWidget(QWidget* widget);
    static void RegisterGLDisplay(QWidget* widget,
                                  ecvGenericGLDisplay* display);
    static void UnregisterGLDisplay(QWidget* widget);
};

Q_DECLARE_OPERATORS_FOR_FLAGS(ecvGenericGLDisplay::INTERACTION_FLAGS)
