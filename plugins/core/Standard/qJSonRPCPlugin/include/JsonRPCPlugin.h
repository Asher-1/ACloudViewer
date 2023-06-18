//##########################################################################
//#                                                                        #
//#                EROWCLOUDVIEWER PLUGIN: ExamplePlugin                   #
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
//#                             COPYRIGHT: XXX                             #
//#                                                                        #
//##########################################################################

#pragma once

#include "ecvStdPluginInterface.h"
//#include <jcon/json_rpc_tcp_server.h>
#include "jsonrpcserver.h"

//! Example qCC plugin
/** Replace 'ExamplePlugin' by your own plugin class name throughout and then
	check 'ExamplePlugin.cpp' for more directions.

	Each plugin requires an info.json file to provide information about itself -
	the name, authors, maintainers, icon, etc..

	The one method you are required to implement is 'getActions'. This should
	return all actions (QAction objects) for the plugin. ErowCloudViewer will
	automatically add these with their icons in the plugin toolbar and to the
	plugin menu. If	your plugin returns	several actions, CC will create a
	dedicated toolbar and a	sub-menu for your plugin. You are responsible for
	connecting these actions to	methods in your plugin.

	Use the ccStdPluginInterface::m_app variable for access to most of the CC
	components (database, 3D views, console, etc.) - see the ecvMainAppInterface
	class in ecvMainAppInterface.h.
**/
class JsonRPCPlugin : public QObject, public ccStdPluginInterface
{
	Q_OBJECT
	Q_INTERFACES( ccPluginInterface ccStdPluginInterface )

	// Replace "Example" by your plugin name (IID should be unique - let's hope your plugin name is unique ;)
	// The info.json file provides information about the plugin to the loading system and
	// it is displayed in the plugin information dialog.
    Q_PLUGIN_METADATA( IID "ecvcorp.cloudviewer.plugin.JsonRPC" FILE "../info.json" )

public:
	explicit JsonRPCPlugin( QObject *parent = nullptr );
	~JsonRPCPlugin() override = default;

	// Inherited from ccStdPluginInterface
	QList<QAction *> getActions() override;
public slots:
    void triggered(bool checked);
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);

private:
	//! Default action
	/** You can add as many actions as you want in a plugin.
		Each action will correspond to an icon in the dedicated
		toolbar and an entry in the plugin menu.
	**/
    QAction* m_action{nullptr};
protected:
    // jcon::JsonRpcTcpServer rpc_server;
    // QTcpServer rpc_server;
    // QWebSocketServer wsrpc_server(QStringLiteral("ErowCloudViewer"), QWebSocketServer::NonSecureMode);
    JsonRPCServer rpc_server;
};
