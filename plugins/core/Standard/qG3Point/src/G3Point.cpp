// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// First:
//	Replace all occurrences of 'G3PointPlugin' by your own plugin class name
// in this file. 	This includes the resource path to info.json in the
// constructor.

// Second:
//	Open G3PointPlugin.qrc, change the "prefix" and the icon filename for
// your plugin. 	Change the name of the file to <yourPluginName>.qrc

// Third:
//	Open the info.json file and fill in the information about the plugin.
//	 "type" should be one of: "Standard", "GL", or "I/O" (required)
//	 "name" is the name of the plugin (required)
//	 "icon" is the Qt resource path to the plugin's icon (from the .qrc
// file) 	 "description" is used as a tootip if the plugin has actions and
// is displayed in the plugin dialog 	 "authors", "maintainers", and
// "references" show up in the plugin dialog as well

#include "G3Point.h"

#include <QtGui>

#include "G3PointAction.h"

ccHObject* G3PointFactory::buildObject(const QString& metaName) {
    if (metaName == "GrainsAsEllipsoids") {
        CVLog::Warning(
                "[G3PointFactory::buildObject] build a GrainsAsEllipsoids "
                "object");
        return new GrainsAsEllipsoids(m_app);
    } else {
        CVLog::Warning(
                "[G3PointFactory::buildObject] you are asking for the building "
                "of an unknown object: " +
                metaName);
        return nullptr;
    }
}

// Default constructor:
//	- pass the Qt resource path to the info.json file (from
//<yourPluginName>.qrc file)
//  - constructor should mainly be used to initialize actions and other members
G3PointPlugin::G3PointPlugin(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/G3PointPlugin/info.json"),
      m_action(nullptr) {}

// This method should enable or disable your plugin actions
// depending on the currently selected entities ('selectedEntities').
void G3PointPlugin::onNewSelection(
        const ccHObject::Container& selectedEntities) {
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
QList<QAction*> G3PointPlugin::getActions() {
    ccExternalFactory::Container::Shared container =
            ccExternalFactory::Container::GetUniqueInstance();
    G3PointFactory* g3PointFactory =
            new G3PointFactory("G3Point", getMainAppInterface());
    container->addFactory(g3PointFactory);

    // default action (if it has not been already created, this is the moment to
    // do it)
    if (!m_action) {
        // Here we use the default plugin name, description, and icon,
        // but each action should have its own.
        m_action = new QAction(getName(), this);
        m_action->setToolTip(getDescription());
        m_action->setIcon(getIcon());

        // Connect appropriate signal
        connect(m_action, &QAction::triggered, this,
                [this]() { G3Point::G3PointAction::createAction(m_app); });
    }

    return {m_action};
}
