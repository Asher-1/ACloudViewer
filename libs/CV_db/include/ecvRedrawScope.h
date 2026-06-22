// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCoreApplication>
#include <QThread>

#include "ecvHObject.h"
#include "ecvViewManager.h"

// RAII guard for selective scene redraw.
//
// Resets all redraw flags on construction, allows marking specific objects
// as needing redraw, and triggers a display refresh on destruction.
// This replaces the error-prone manual pattern of:
//   SetRedrawRecursive(false) -> mark dirty -> refreshAll()
//
// THREAD-SAFETY: The destructor calls view->redraw() which triggers
// vtkRenderWindow::Render().  VTK/OpenGL rendering MUST happen on the
// main/GUI thread.  If constructed from a worker thread the scope
// auto-dismisses and skips the redraw to prevent SIGABRT.
//
// Usage:
//   {
//       ecvRedrawScope scope;
//       scope.markDirty(modifiedMesh);
//       scope.markDirty(modifiedCloud);
//   }  // display is refreshed here, only redrawing the marked objects
//
//   // One-liner with initializer list:
//   { ecvRedrawScope scope({mesh, cloud}); }
//
class CV_DB_LIB_API ecvRedrawScope final {
public:
    explicit ecvRedrawScope(bool only2D = false, bool forceRedraw = true)
        : m_only2D(only2D), m_forceRedraw(forceRedraw) {
        if (!isMainThread()) {
            m_dismissed = true;
            return;
        }
        if (ecvViewManager::instance().hasAnyView()) {
            ecvViewManager::instance().setRedrawRecursive(false);
        }
    }

    explicit ecvRedrawScope(std::initializer_list<ccHObject*> objects,
                            bool only2D = false,
                            bool forceRedraw = true)
        : m_only2D(only2D), m_forceRedraw(forceRedraw) {
        if (!isMainThread()) {
            m_dismissed = true;
            return;
        }
        if (ecvViewManager::instance().hasAnyView()) {
            ecvViewManager::instance().setRedrawRecursive(false);
        }
        for (auto* obj : objects) {
            if (obj) obj->setRedrawFlagRecursive(true);
        }
    }

    ~ecvRedrawScope() {
        if (!m_dismissed && ecvViewManager::instance().hasAnyView()) {
            if (auto* view = ecvViewManager::instance().getEffectiveView()) {
                view->redraw(m_only2D, m_forceRedraw);
            }
        }
    }

    ecvRedrawScope& markDirty(ccHObject* obj) {
        if (obj) obj->setRedrawFlagRecursive(true);
        return *this;
    }

    void dismiss() { m_dismissed = true; }

    ecvRedrawScope(const ecvRedrawScope&) = delete;
    ecvRedrawScope& operator=(const ecvRedrawScope&) = delete;

private:
    static bool isMainThread() {
        auto* app = QCoreApplication::instance();
        return app && QThread::currentThread() == app->thread();
    }

    bool m_only2D = false;
    bool m_forceRedraw = true;
    bool m_dismissed = false;
};
