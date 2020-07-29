#ifndef CCPLUGINUIMANAGER_H
#define CCPLUGINUIMANAGER_H

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

#include <QActionGroup>
#include <QList>
#include <QObject>

#include "ecvPluginManager.h"

class QAction;
class QMenu;
class QString;
class QToolBar;
class QWidget;

class ecvMainAppInterface;
class ccPluginInterface;
class ccStdPluginInterface;
class ccPclPluginInterface;

//! Plugin UI manager
class ccPluginUIManager : public QObject
{
	Q_OBJECT
	
public:
	ccPluginUIManager( ecvMainAppInterface *appInterface, QWidget *parent );
	~ccPluginUIManager();
	
	void	init( const ccPluginInterfaceList &plugins );
	
	QMenu	*pluginMenu() const;
	QMenu	*pclAlgorithmMenu() const;
	
	QToolBar *mainPluginToolbar();
	QList<QToolBar *> &additionalPluginToolbars();
	QAction *actionShowMainPluginToolbar();

	QToolBar *glPclToolbar();
	QAction *actionShowPCLAlgorithmToolbar();

	void	updateMenus();
	void	handleSelectionChanged();

	void	showAboutDialog() const;
	
private:
	void	setupActions();
	
	void	setupMenus();
	void	addActionsToMenu( ccStdPluginInterface *stdPlugin, const QList<QAction *> &actions );
	void	addActionsToMenu( const QString &moduleName, const QList<QAction *> &actions );

	void	setupToolbars();
	void	addActionsToToolBar( ccStdPluginInterface *stdPlugin, const QList<QAction *> &actions );
	void	addActionsToToolBar( const QString &moduleName, const QList<QAction *> &actions );
	
	void	enablePCLAlgorithm();
	void	disablePCLAlgorithm();

	QWidget	*m_parentWidget;	// unfortunately we need this when creating new menus & toolbars
	
	ecvMainAppInterface *m_appInterface;
	
	QMenu	*m_pluginMenu;
	QMenu	*m_pclAlgorithmMenu;

	QAction *m_actionRemovePCLAlgorithm;
	QActionGroup m_pclAlgorithmActions;
	
	QList<ccPluginInterface *> m_plugins;
	
	QToolBar *m_mainPluginToolbar;	// if a plugin only has one action it goes here
	QList<QToolBar *> m_additionalPluginToolbars;	// if a plugin has multiple actions it gets its own toolbar
	QAction	*m_showPluginToolbar;

	QToolBar *m_pclAlgorithmsToolbar;
	QAction	*m_showPCLAlgorithmToolbar;

};

#endif
