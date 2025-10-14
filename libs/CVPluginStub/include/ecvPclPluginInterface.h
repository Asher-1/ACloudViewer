// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PCL_PLUGIN_INTERFACE_HEADER
#define ECV_PCL_PLUGIN_INTERFACE_HEADER

// Qt
#include <QActionGroup>
#include <QWidget>

// qCC_db
#include <ecvHObject.h>

// qCC
#include "ecvDefaultPluginInterface.h"
#include "ecvMainAppInterface.h"

//! Pcl ECV plugin interface
/** Version 1.4
        The plugin is now responsible for its own actions (QAction ;)
        and the associated ecvMainAppInterface member should give it
        access to everything it needs in the main application.
**/
class ccPclPluginInterface : public ccDefaultPluginInterface {
public:
    //! Default constructor
    ccPclPluginInterface(const QString& resourcePath = QString())
        : ccDefaultPluginInterface(resourcePath), m_app(nullptr) {}

    //! Destructor
    virtual ~ccPclPluginInterface() = default;

    // inherited from ccPluginInterface
    virtual CC_PLUGIN_TYPE getType() const override {
        return ECV_PCL_ALGORITHM_PLUGIN;
    }

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
    virtual QVector<QList<QAction*>> getActions() = 0;

    virtual QVector<QString> getModuleNames() = 0;

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

Q_DECLARE_METATYPE(ccPclPluginInterface*);

Q_DECLARE_INTERFACE(ccPclPluginInterface,
                    "edf.rd.cloudviewer.ccPclPluginInterface/1.4")

#endif  // ECV_PCL_PLUGIN_INTERFACE_HEADER
