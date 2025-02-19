#ifndef ECVPLUGINMANAGER_H
#define ECVPLUGINMANAGER_H

//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
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
//#          COPYRIGHT: CLOUDVIEWER  project                               #
//#                                                                        #
//##########################################################################

#include "CVAppCommon.h"

#include <QObject>
#include <QVector>

class ccPluginInterface;

//! Simply a list of \see ccPluginInterface
using ccPluginInterfaceList = QVector<ccPluginInterface *>;


class CVAPPCOMMON_LIB_API ccPluginManager : public QObject
{
	Q_OBJECT
	
public:
	~ccPluginManager() override = default;
	
	static ccPluginManager& get();

	void setPaths(const QStringList& paths);
	QStringList pluginPaths();

	void loadPlugins();

	ccPluginInterfaceList& pluginList();

	bool isEnabled(const ccPluginInterface* plugin) const;
	void setPluginEnabled(const ccPluginInterface* plugin, bool enabled);

protected:
	explicit ccPluginManager(QObject *parent = nullptr);
	
private:	
	void loadFromPathsAndAddToList();

	QStringList disabledPluginIIDs() const;

	QStringList m_pluginPaths;
	ccPluginInterfaceList m_pluginList;
};

#endif // ECVPLUGINMANAGER_H