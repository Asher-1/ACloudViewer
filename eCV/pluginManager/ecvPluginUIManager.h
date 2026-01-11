// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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
class ccPluginUIManager : public QObject {
    Q_OBJECT

public:
    ccPluginUIManager(ecvMainAppInterface *appInterface, QWidget *parent);
    ~ccPluginUIManager();

    void init();

    QMenu *pluginMenu() const;
    QMenu *pclAlgorithmMenu() const;

    QToolBar *mainPluginToolbar();
    QList<QToolBar *> &additionalPluginToolbars();
    QAction *actionShowMainPluginToolbar();

    QToolBar *glPclToolbar();
    QAction *actionShowPCLAlgorithmToolbar();

    void updateMenus();
    void handleSelectionChanged();

    void showAboutDialog() const;

    // Helper method to check if a toolbar belongs to Python plugin
    static bool isPythonPluginToolbar(QToolBar *toolbar);

private:
    void setupActions();

    void setupMenus();
    void addActionsToMenu(ccStdPluginInterface *stdPlugin,
                          const QList<QAction *> &actions);
    void addActionsToMenu(const QString &moduleName,
                          const QList<QAction *> &actions);

    void setupToolbars();
    void addActionsToToolBar(ccStdPluginInterface *stdPlugin,
                             const QList<QAction *> &actions);
    void addActionsToToolBar(const QString &moduleName,
                             const QList<QAction *> &actions);

    void enablePCLAlgorithm();
    void disablePCLAlgorithm();

    QWidget *m_parentWidget;  // unfortunately we need this when creating new
                              // menus & toolbars

    ecvMainAppInterface *m_appInterface;

    QMenu *m_pluginMenu;
    QMenu *m_pclAlgorithmMenu;

    QAction *m_actionRemovePCLAlgorithm;
    QActionGroup m_pclAlgorithmActions;

    QList<ccPluginInterface *> m_plugins;

    QToolBar *m_mainPluginToolbar;  // if a plugin only has one action it goes
                                    // here
    QList<QToolBar *>
            m_additionalPluginToolbars;  // if a plugin has multiple actions it
                                         // gets its own toolbar
    QAction *m_showPluginToolbar;

    QToolBar *m_pclAlgorithmsToolbar;
    QAction *m_showPCLAlgorithmToolbar;
};
