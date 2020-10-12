//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                        COPYRIGHT: DAHAI LU                             #
//#                                                                        #
//##########################################################################
//
#ifndef Q_PCL_PLUGIN_HEADER
#define Q_PCL_PLUGIN_HEADER

//#include "ecvStdPluginInterface.h"
#include "ecvPclPluginInterface.h"

//Qt
#include <QObject>
#include <QtGui>

class BasePclModule;

//! PCL bridge plugin
class qPCL : public QObject, public ccPclPluginInterface
{
	Q_OBJECT
	Q_INTERFACES( ccPluginInterface ccPclPluginInterface )
	Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qPCL" FILE "../info.json")

public:
	//! Default constructor
	qPCL(QObject* parent = nullptr);
	//! Destructor
	virtual ~qPCL() = default;

	// inherited from ccPclPluginInterface
	virtual void onNewSelection(const ccHObject::Container& selectedEntities) override;
	virtual QVector<QList<QAction *>> getActions() override;
	virtual QVector<QString> getModuleNames() override;

	virtual void stop() override;

public slots:
	//! Handles new entity
	void handleNewEntity(ccHObject*);

	//! Handles entity (visual) modification
	void handleEntityChange(ccHObject*);

	//! Handles new error message
	void handleErrorMessage(QString);

protected:

	//! Adds a pcl module
	int addPclModule(BasePclModule* module, QList<QAction *> &actions);

	//! Loaded modules
	std::vector<BasePclModule*> m_modules;
	QVector<QString> m_moduleNames;
};

#endif //END Q_PCL_PLUGIN_HEADER
