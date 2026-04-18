// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

#include "CV_db.h"

class ccHObject;
class ccDrawableObject;
class QWidget;
class ecvViewportParameters;

/// Per-window display interface for multi-window rendering.
///
/// Each 3D view window implements this interface, holding its own
/// viewport parameters, camera state, and scene/window DB roots.
/// The global ecvDisplayTools singleton delegates to the "active"
/// ecvGenericGLDisplay instance so that existing static call sites
/// keep working while new code can address a specific window.
///
/// Design references:
///   CloudCompare  ccGenericGLDisplay
///   ParaView      pqView / pqRenderView
///   MeshLab       GLArea
class CV_DB_LIB_API ecvGenericGLDisplay {
public:
    virtual ~ecvGenericGLDisplay() = default;

    // -- Identification --

    virtual int getUniqueID() const = 0;
    virtual QString getTitle() const = 0;

    // -- Refresh / redraw --

    virtual void redraw(bool only2D = false, bool forceRedraw = true) = 0;
    virtual void refresh(bool only2D = false) = 0;
    virtual void toBeRefreshed() = 0;

    // -- Viewport / camera --

    virtual const ecvViewportParameters& getViewportParameters() const = 0;
    virtual void setViewportParameters(const ecvViewportParameters& params) = 0;
    virtual void setPerspectiveState(bool state, bool objectCenteredView) = 0;
    virtual bool perspectiveView() const = 0;
    virtual bool objectCenteredView() const = 0;

    // -- Scene / own DB --

    virtual void setSceneDB(ccHObject* root) = 0;
    virtual ccHObject* getSceneDB() = 0;
    virtual ccHObject* getOwnDB() = 0;
    virtual void addToOwnDB(ccHObject* obj, bool noDependency = true) = 0;
    virtual void removeFromOwnDB(ccHObject* obj) = 0;

    // -- Qt widget bridge --

    virtual QWidget* asWidget() = 0;
    virtual const QWidget* asWidget() const = 0;

    // -- Display parameters (per-window overrides) --

    virtual bool hasOverriddenDisplayParameters() const = 0;

    // -- Lifecycle notification (ref: CloudCompare aboutToBeRemoved) --

    virtual void aboutToBeRemoved(ccDrawableObject* /*obj*/) {}

    // -- Static registry: QWidget* -> ecvGenericGLDisplay* --

    /// Resolve the display associated with the given widget, or nullptr.
    static ecvGenericGLDisplay* FromWidget(QWidget* widget);

    /// Register a display instance (called during window creation).
    static void RegisterGLDisplay(QWidget* widget,
                                  ecvGenericGLDisplay* display);

    /// Unregister a display instance (called during window destruction).
    static void UnregisterGLDisplay(QWidget* widget);
};
