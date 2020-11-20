//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER PLUGIN: qMPlane                         #
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
//#         COPYRIGHT: AIT Austrian Institute of Technology GmbH           #
//#                                                                        #
//##########################################################################

#ifndef Q_MPLANE_PLUGIN_HEADER
#define Q_MPLANE_PLUGIN_HEADER

// std
#include <memory>

//CloudCompare
#include <ecvPickingListener.h>
#include <ecvPointCloud.h>
#include <ecvStdPluginInterface.h>

//qMPlane
#include "src/ccMPlaneDlgController.h"

#ifdef USE_VLD
//VLD
#include <vld.h>
#endif


class qMPlane : public QObject, public ccStdPluginInterface
{
	Q_OBJECT
	Q_INTERFACES( ccPluginInterface ccStdPluginInterface )
	Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.MPlane" FILE "info.json")
	
public:
	explicit qMPlane( QObject *parent = nullptr );
	~qMPlane() override = default;
	
	void onNewSelection( const ccHObject::Container &selectedEntities ) override;
	QList<QAction *> getActions() override;


protected slots:
	void doAction();

private:
	QAction* m_action = nullptr;
	ccPointCloud *m_selectedCloud = nullptr;
	std::unique_ptr<ccMPlaneDlgController> m_controller;
};
#endif
