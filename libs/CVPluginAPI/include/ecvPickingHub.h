// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVPluginAPI.h"

// Local
#include "ecvPickingListener.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>

// Qt
#include <QObject>

// system
#include <set>

class ccHObject;
class QMdiSubWindow;
class ecvMainAppInterface;

/**
 * @class ccPickingHub
 * @brief Central hub for managing entity/point/triangle picking
 *
 * Coordinates picking operations across the application and plugins.
 * Multiple picking listeners can register to receive picking events,
 * with support for exclusive picking modes.
 *
 * Features:
 * - Multiple listener registration
 * - Exclusive picking mode (single listener)
 * - Automatic picking mode management per window
 * - Support for different picking modes (point, triangle, entity, etc.)
 *
 * Picking workflow:
 * 1. Listener registers with hub via addListener()
 * 2. Hub enables picking mode in active window
 * 3. User clicks in 3D view
 * 4. Hub receives picked item and forwards to all listeners
 * 5. Listeners process the picked item
 *
 * @see ccPickingListener
 * @see ecvDisplayTools::PICKING_MODE
 */
class CVPLUGIN_LIB_API ccPickingHub : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param app Main application interface
     * @param parent Parent QObject (optional)
     */
    ccPickingHub(ecvMainAppInterface* app, QObject* parent = nullptr);

    /**
     * @brief Destructor
     */
    ~ccPickingHub() override = default;

    /**
     * @brief Get number of registered listeners
     * @return Current listener count
     */
    inline size_t listenerCount() const { return m_listeners.size(); }

    /**
     * @brief Register a picking listener
     *
     * Adds a listener that will receive picking events. Optionally
     * enables exclusive mode and/or automatically starts picking.
     *
     * @param listener Listener to register
     * @param exclusive Prevent other listeners from registering (default:
     * false)
     * @param autoStartPicking Auto-enable picking in active window (default:
     * true)
     * @param mode Picking mode to use (default: point or triangle)
     * @return true if registered successfully, false if failed
     *         (e.g., incompatible mode or exclusive lock active)
     */
    bool addListener(ccPickingListener* listener,
                     bool exclusive = false,
                     bool autoStartPicking = true,
                     ecvDisplayTools::PICKING_MODE mode =
                             ecvDisplayTools::POINT_OR_TRIANGLE_PICKING);

    /**
     * @brief Unregister a picking listener
     *
     * Removes listener from receiving picking events. Optionally
     * disables picking mode if this was the last listener.
     *
     * @param listener Listener to unregister
     * @param autoStopPickingIfLast Auto-disable picking if no listeners remain
     * (default: true)
     */
    void removeListener(ccPickingListener* listener,
                        bool autoStopPickingIfLast = true);

    /**
     * @brief Toggle picking mode on/off in active window
     * @param state true = enable picking, false = disable
     */
    void togglePickingMode(bool state);

    /**
     * @brief Get currently active window
     * @return Pointer to active window widget (or nullptr)
     */
    QWidget* activeWindow() const { return m_activeWindow; }

    /**
     * @brief Check if hub is locked by exclusive listener
     *
     * Locked means an exclusive listener is registered and
     * no other listeners can be added until it's removed.
     *
     * @return true if locked (exclusive mode active)
     */
    bool isLocked() const { return m_exclusive && !m_listeners.empty(); }

public slots:

    /**
     * @brief Handle active window change
     * @param window Newly active window
     */
    void onActiveWindowChanged(QMdiSubWindow* window);

    /**
     * @brief Handle active window deletion
     * @param window Deleted window
     */
    void onActiveWindowDeleted(QObject* window);

    /**
     * @brief Process picked item from display
     * @param entity Picked entity (or nullptr)
     * @param itemIdx Item index within entity
     * @param x Screen X coordinate
     * @param y Screen Y coordinate
     * @param P 3D picked point coordinates
     */
    void processPickedItem(ccHObject* entity,
                           unsigned itemIdx,
                           int x,
                           int y,
                           const CCVector3& P);

protected:
    std::set<ccPickingListener*> m_listeners;  ///< Registered listeners

    ecvMainAppInterface* m_app;  ///< Main application interface

    QWidget* m_activeWindow;  ///< Currently active display window

    ecvDisplayTools::PICKING_MODE m_pickingMode;  ///< Current picking mode

    bool m_autoEnableOnActivatedWindow;  ///< Auto-enable on window activation

    bool m_exclusive;  ///< Exclusive mode flag
};
