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

// ECV_DB_LIB
#include <ecvDisplayTools.h>

// Qt
#include <QObject>

// system
#include <set>

class ccHObject;
class QMdiSubWindow;
class ecvMainAppInterface;

//! Point/triangle picking hub
class CVPLUGIN_LIB_API ccPickingHub : public QObject {
    Q_OBJECT

public:
    //! Default constructor
    ccPickingHub(ecvMainAppInterface* app, QObject* parent = nullptr);
    ~ccPickingHub() override = default;

    //! Returns the number of currently registered listeners
    inline size_t listenerCount() const { return m_listeners.size(); }

    //! Adds a listener
    /** \param listener listener to be registered
            \param exclusive prevents new listeners from registering
            \param autoStartPicking automatically enables the picking mode on
    the active window (if any) \param mode sets the picking mode (warning: may
    be rejected if another listener is currently registered with another mode)
            \return success
    ***/
    bool addListener(ccPickingListener* listener,
                     bool exclusive = false,
                     bool autoStartPicking = true,
                     ecvDisplayTools::PICKING_MODE mode =
                             ecvDisplayTools::POINT_OR_TRIANGLE_PICKING);

    //! Removes a listener
    /** \param listener listener to be removed
            \param autoStopPickingIfLast automatically disables the picking mode
    on the active window (if any) if no other listener is registered
    ***/
    void removeListener(ccPickingListener* listener,
                        bool autoStopPickingIfLast = true);

    //	//! Sets the default picking mode
    //	/** \param mode picking mode
    //		\param autoEnableOnActivatedWindow whether picking mode should
    // be enabled automatically on newly activated windows (if listeners are
    // present only)
    //	**/
    // DGM: too dangerous, we can't change this behavior on the fly

    //! Manual start / stop of the picking mode on the active window
    void togglePickingMode(bool state);

    //! Returns the currently active window
    QWidget* activeWindow() const { return m_activeWindow; }

    //! Returns whether the picking mechanism is currently locked (i.e. an
    //! exclusive listener is registered)
    bool isLocked() const { return m_exclusive && !m_listeners.empty(); }

public slots:

    void onActiveWindowChanged(QMdiSubWindow*);
    void onActiveWindowDeleted(QObject*);
    void processPickedItem(ccHObject*, unsigned, int, int, const CCVector3&);

protected:
    //! Listeners
    std::set<ccPickingListener*> m_listeners;

    //! Associated application
    ecvMainAppInterface* m_app;

    //! Active window
    QWidget* m_activeWindow;

    //! Default picking mode
    ecvDisplayTools::PICKING_MODE m_pickingMode;

    //! Automatically enables the picking mechanism on activated GL windows
    bool m_autoEnableOnActivatedWindow;

    //! Exclusive mode
    bool m_exclusive;
};
