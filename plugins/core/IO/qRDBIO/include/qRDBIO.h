#ifndef Q_RDB_IO_PLUGIN_HEADER
#define Q_RDB_IO_PLUGIN_HEADER

//##########################################################################
//#                                                                        #
//#                     ACloudViewer PLUGIN: qRDBIO                     #
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
//#          COPYRIGHT: RIEGL Laser Measurement Systems GmbH               #
//#                                                                        #
//##########################################################################

#include <ecvIOPluginInterface.h>

//! RDB file (3D cloud)
class qRDBIO : public QObject, public ccIOPluginInterface
{
	Q_OBJECT
	Q_INTERFACES( ccPluginInterface ccIOPluginInterface )

    Q_PLUGIN_METADATA( IID "ecvcorp.cloudviewer.plugin.qRDBIO" FILE "../info.json" )

public:
	//! Default constructor
	explicit qRDBIO( QObject *parent = nullptr );

	// inherited from ccIOPluginInterface
	FilterList getFilters() override;
};

#endif // Q_RDB_IO_PLUGIN_HEADER
