//##########################################################################
//#                                                                        #
//#                            CLOUDVIEWER                                 #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / TELECOM ParisTech (ENST-TSI)             #
//#                                                                        #
//##########################################################################

#ifndef ECV_STD_PLUGIN_INTERFACE_HEADER
#define ECV_STD_PLUGIN_INTERFACE_HEADER

//Qt
#include <QWidget>
#include <QActionGroup>

//qCC_db
#include <ecvHObject.h>

//qCC
#include "ecvDefaultPluginInterface.h"
#include "ecvMainAppInterface.h"

//UI Modification flags
#define CC_PLUGIN_REFRESH_ENTITY_BROWSER        0x00000002
#define CC_PLUGIN_EXPAND_DB_TREE                0x00000004

//! Standard CV plugin interface
/** Version 1.5
	The plugin is now responsible for its own actions (QAction ;)
	and the associated ecvMainAppInterface member should give it
	access to everything it needs in the main application.
**/
class ccStdPluginInterface : public ccDefaultPluginInterface
{	
public:

	//! Default constructor
	ccStdPluginInterface( const QString &resourcePath = QString() ) :
		ccDefaultPluginInterface( resourcePath )
	  , m_app(nullptr)
	{
	}
	
	//! Destructor
	virtual ~ccStdPluginInterface() override = default;

	//inherited from ccPluginInterface
	virtual CC_PLUGIN_TYPE getType() const override { return ECV_STD_PLUGIN; }

	//! Sets application entry point
	/** Called just after plugin creation by qCC
	**/
	virtual void setMainAppInterface(ecvMainAppInterface* app)
	{
		m_app = app;
		
		if (m_app)
		{
			//we use the same 'unique ID' generator in plugins as in the main
			//application (otherwise we'll have issues with 'unique IDs'!)
			ccObject::SetUniqueIDGenerator(m_app->getUniqueIDGenerator());
		}
	}

	//! A callback pointer to the main app interface for use by plugins
	/**  Any plugin (and its tools) may need to access methods of this interface
	**/
	virtual ecvMainAppInterface * getMainAppInterface() { return m_app; }

	//! Get a list of actions for this plugin
	virtual QList<QAction *> getActions() = 0;

	//! This method is called by the main application whenever the entity selection changes
	/** Does nothing by default. Should be re-implemented by the plugin if necessary.
		\param selectedEntities currently selected entities
	**/
	virtual void onNewSelection(const ccHObject::Container& selectedEntities) { /*ignored by default*/ }

	//! Shortcut to ecvMainAppInterface::dispToConsole
	inline virtual void dispToConsole(QString message, ecvMainAppInterface::ConsoleMessageLevel level = ecvMainAppInterface::STD_CONSOLE_MESSAGE)
	{
		if (m_app)
		{
			m_app->dispToConsole(message, level);
		}
	}

protected:

	//! Main application interface
	ecvMainAppInterface* m_app;
};

Q_DECLARE_METATYPE(const ccStdPluginInterface *);

Q_DECLARE_INTERFACE(ccStdPluginInterface, "edf.rd.CloudViewer.ccStdPluginInterface/1.4")

#endif // ECV_STD_PLUGIN_INTERFACE_HEADER
