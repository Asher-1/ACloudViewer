// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ExamplePlugin.h"

#include <QtGui>

// Default constructor:
//	- pass the Qt resource path to the info.json file (from
//<yourPluginName>.qrc file)
//  - constructor should mainly be used to initialize actions and other members
ExamplePlugin::ExamplePlugin(QObject *parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/ExamplePlugin/info.json"),
      m_action(nullptr) {}

// This method should enable or disable your plugin actions
// depending on the currently selected entities ('selectedEntities').
void ExamplePlugin::onNewSelection(
        const ccHObject::Container &selectedEntities) {
    if (m_action == nullptr) {
        return;
    }

    // If you need to check for a specific type of object, you can use the
    // methods in ccHObjectCaster.h or loop and check the objects' classIDs like
    // this:
    //
    //	for ( ccHObject *object : selectedEntities )
    //	{
    //		if ( object->getClassID() == CV_TYPES::VIEWPORT_2D_OBJECT )
    //		{
    //			// ... do something with the viewports
    //		}
    //	}

    // For example - only enable our action if something is selected.
    m_action->setEnabled(!selectedEntities.empty());
}

// This method returns all the 'actions' your plugin can perform.
// getActions() will be called only once, when plugin is loaded.
QList<QAction *> ExamplePlugin::getActions() {
    // default action (if it has not been already created, this is the moment to
    // do it)
    if (!m_action) {
        // Here we use the default plugin name, description, and icon,
        // but each action should have its own.
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        // Connect appropriate signal
        connect(m_action, &QAction::triggered, this, &ExamplePlugin::doAction);
    }

    return {m_action};
}

// This is an example of an action's method called when the corresponding action
// is triggered (i.e. the corresponding icon or menu entry is clicked in CC's
// main interface). You can access most of CC's components (database,
// 3D views, console, etc.) via the 'm_app' variable (see the
// ecvMainAppInterface class in ecvMainAppInterface.h).
void ExamplePlugin::doAction() {
    if (m_app == nullptr) {
        // m_app should have already been initialized by CC when plugin is
        // loaded
        Q_ASSERT(false);

        return;
    }

    /*** HERE STARTS THE ACTION ***/

    // Put your code here
    // --> you may want to start by asking for parameters (with a custom dialog,
    // etc.)

    // This is how you can output messages
    // Display a standard message in the console
    m_app->dispToConsole("[ExamplePlugin] Hello world!",
                         ecvMainAppInterface::STD_CONSOLE_MESSAGE);

    // Display a warning message in the console
    m_app->dispToConsole(
            "[ExamplePlugin] Warning: example plugin shouldn't be used as is",
            ecvMainAppInterface::WRN_CONSOLE_MESSAGE);

    // Display an error message in the console AND pop-up an error box
    m_app->dispToConsole(
            "Example plugin shouldn't be used - it doesn't do anything!",
            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);

    /*** HERE ENDS THE ACTION ***/
}
