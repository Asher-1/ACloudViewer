//##########################################################################
//#                                                                        #
//#                  CLOUDVIEWER  PLUGIN: qCSVMatrixIO                     #
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
//#                  COPYRIGHT: Daniel Girardeau-Montaut                   #
//#                                                                        #
//##########################################################################

#ifndef Q_CSV_MATRIX_IO_PLUGIN_HEADER
#define Q_CSV_MATRIX_IO_PLUGIN_HEADER

#include "ecvIOPluginInterface.h"

//! CSV Matrix file (2.5D cloud)
class qCSVMatrixIO : public QObject, public ccIOPluginInterface
{
	Q_OBJECT
	Q_INTERFACES( ccPluginInterface ccIOPluginInterface )
	Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qCSVMatrixIO" FILE "../info.json")

public:
	qCSVMatrixIO(QObject* parent = nullptr);
	
	virtual ~qCSVMatrixIO() override = default;

	//inherited from ccIOPluginInterface
	FilterList getFilters() override;
};

#endif // Q_CSV_MATRIX_IO_PLUGIN_HEADER
