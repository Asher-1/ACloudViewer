// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// QT
#include <QActionGroup>
#include <QWidget>

// ECV_DB_LIB
#include <ecvHObject.h>

// LOCAL
#include "ecvDefaultPluginInterface.h"
#include "ecvMainAppInterface.h"

// UI Modification flags
#define CC_PLUGIN_REFRESH_ENTITY_BROWSER 0x00000002
#define CC_PLUGIN_EXPAND_DB_TREE 0x00000004

//! Standard ECV plugin interface
/** Version 1.5
        The plugin is now responsible for its own actions (QAction ;)
        and the associated ecvMainAppInterface member should give it
        access to everything it needs in the main application.
**/
class ccStdPluginInterface : public ccDefaultPluginInterface {
public:
    //! Default constructor
    ccStdPluginInterface(const QString& resourcePath = QString())
        : ccDefaultPluginInterface(resourcePath), m_app(nullptr) {}

    //! Destructor
    virtual ~ccStdPluginInterface() override = default;

    // inherited from ccPluginInterface
    virtual CC_PLUGIN_TYPE getType() const override { return ECV_STD_PLUGIN; }

    //! Sets application entry point
    /** Called just after plugin creation by qCC
     **/
    virtual void setMainAppInterface(ecvMainAppInterface* app) {
        m_app = app;

        if (m_app) {
            // we use the same 'unique ID' generator in plugins as in the main
            // application (otherwise we'll have issues with 'unique IDs'!)
            ccObject::SetUniqueIDGenerator(m_app->getUniqueIDGenerator());
        }
    }

    //! A callback pointer to the main app interface for use by plugins
    /**  Any plugin (and its tools) may need to access methods of this interface
     **/
    virtual ecvMainAppInterface* getMainAppInterface() { return m_app; }

    //! Get a list of actions for this plugin
    virtual QList<QAction*> getActions() = 0;

    //! This method is called by the main application whenever the entity
    //! selection changes
    /** Does nothing by default. Should be re-implemented by the plugin if
    necessary. \param selectedEntities currently selected entities
    **/
    virtual void onNewSelection(
            const ccHObject::Container&
                    selectedEntities) { /*ignored by default*/ }

    //! Shortcut to ecvMainAppInterface::dispToConsole
    inline virtual void dispToConsole(
            QString message,
            ecvMainAppInterface::ConsoleMessageLevel level =
                    ecvMainAppInterface::STD_CONSOLE_MESSAGE) {
        if (m_app) {
            m_app->dispToConsole(message, level);
        }
    }

protected:
    //! Main application interface
    ecvMainAppInterface* m_app;
};

Q_DECLARE_METATYPE(const ccStdPluginInterface*);

Q_DECLARE_INTERFACE(ccStdPluginInterface,
                    "edf.rd.cloudviewer.ccStdPluginInterface/1.5")
