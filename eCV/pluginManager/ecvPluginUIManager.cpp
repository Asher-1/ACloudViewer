// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPluginUIManager.h"

#include <QAction>
#include <QMenu>
#include <QToolBar>
#include <QWidget>

#include "ecvConsole.h"
#include "ecvDisplayTools.h"
#include "ecvIOPluginInterface.h"
#include "ecvMainAppInterface.h"
#include "ecvPclPluginInterface.h"
#include "ecvPluginInfoDlg.h"
#include "ecvPluginManager.h"
#include "ecvStdPluginInterface.h"

ccPluginUIManager::ccPluginUIManager(ecvMainAppInterface *appInterface,
                                     QWidget *parent)
    : QObject(parent),
      m_parentWidget(parent),
      m_appInterface(appInterface),
      m_pluginMenu(nullptr),
      m_pclAlgorithmMenu(nullptr),
      m_actionRemovePCLAlgorithm(nullptr),
      m_pclAlgorithmActions(this),
      m_mainPluginToolbar(nullptr),
      m_showPluginToolbar(nullptr),
      m_pclAlgorithmsToolbar(nullptr),
      m_showPCLAlgorithmToolbar(nullptr) {
    setupActions();
    setupMenus();
    setupToolbars();
}

ccPluginUIManager::~ccPluginUIManager() {}

void ccPluginUIManager::init() {
    auto plugins = ccPluginManager::get().pluginList();

    m_pluginMenu->setEnabled(false);
    m_pclAlgorithmMenu->setEnabled(false);

    m_mainPluginToolbar->setVisible(false);

    QVector<ccStdPluginInterface *> coreStdPlugins;
    QVector<ccStdPluginInterface *> thirdPartyStdPlugins;

    QVector<ccPclPluginInterface *> corePCLPlugins;
    QVector<ccPclPluginInterface *> thirdPartyPCLPlugins;

    for (ccPluginInterface *plugin : plugins) {
        if (plugin == nullptr) {
            Q_ASSERT(false);
            continue;
        }

        if (!ccPluginManager::get().isEnabled(plugin)) {
            m_plugins.push_back(plugin);

            continue;
        }

        const QString pluginName = plugin->getName();

        Q_ASSERT(!pluginName.isEmpty());

        if (pluginName.isEmpty()) {
            // should be unreachable - we have already checked for this in
            // ecvPlugins::Find()
            continue;
        }

        switch (plugin->getType()) {
            case ECV_STD_PLUGIN:  // standard plugin
            {
                ccStdPluginInterface *stdPlugin =
                        static_cast<ccStdPluginInterface *>(plugin);

                stdPlugin->setMainAppInterface(m_appInterface);

                if (stdPlugin->isCore()) {
                    coreStdPlugins.append(stdPlugin);
                } else {
                    thirdPartyStdPlugins.append(stdPlugin);
                }

                m_plugins.push_back(stdPlugin);

                break;
            }

            case ECV_PCL_ALGORITHM_PLUGIN:  // pcl algorithm
            {
                ccPclPluginInterface *pclPlugin =
                        static_cast<ccPclPluginInterface *>(plugin);

                pclPlugin->setMainAppInterface(m_appInterface);

                QAction *action = new QAction(pluginName, this);

                action->setToolTip(pclPlugin->getDescription());
                action->setIcon(pclPlugin->getIcon());
                action->setCheckable(true);

                // store the plugin's interface pointer in the QAction data so
                // we can access it in enablePCLAlgorithm()
                QVariant v;

                v.setValue(pclPlugin);

                action->setData(v);

                connect(action, &QAction::triggered, this,
                        &ccPluginUIManager::enablePCLAlgorithm);

                m_pclAlgorithmActions.addAction(action);

                if (pclPlugin->isCore()) {
                    corePCLPlugins.append(pclPlugin);
                } else {
                    thirdPartyPCLPlugins.append(pclPlugin);
                }

                m_plugins.push_back(pclPlugin);
                break;
            }

            case ECV_IO_FILTER_PLUGIN: {
                ccIOPluginInterface *ioPlugin =
                        static_cast<ccIOPluginInterface *>(plugin);

                // there are no menus or toolbars for I/O plugins
                m_plugins.push_back(ioPlugin);
                break;
            }
        }
    }

    // add core standard plugins to menu & tool bar
    for (ccStdPluginInterface *plugin : coreStdPlugins) {
        QList<QAction *> actions = plugin->getActions();

        addActionsToMenu(plugin, actions);
        addActionsToToolBar(plugin, actions);

        plugin->onNewSelection(m_appInterface->getSelectedEntities());
    }

    // add 3rd standard party plugins to menu & tool bar (if any )
    if (!thirdPartyStdPlugins.isEmpty()) {
        m_pluginMenu->addSection("3rd std Party");

        for (ccStdPluginInterface *plugin : thirdPartyStdPlugins) {
            QList<QAction *> actions = plugin->getActions();

            addActionsToMenu(plugin, actions);
            addActionsToToolBar(plugin, actions);

            plugin->onNewSelection(m_appInterface->getSelectedEntities());
        }
    }

    // add core PCL plugins to menu & tool bar
    for (ccPclPluginInterface *plugin : corePCLPlugins) {
        QVector<QList<QAction *>> allModuleActions = plugin->getActions();
        QVector<QString> moduleNames = plugin->getModuleNames();
        assert(allModuleActions.size() == moduleNames.size());

        if (!plugin->getName().isEmpty()) {
            m_pclAlgorithmMenu->setTitle(plugin->getName());
        }

        for (int i = 0; i < moduleNames.size(); ++i) {
            addActionsToMenu(moduleNames[i], allModuleActions[i]);
            addActionsToToolBar(moduleNames[i], allModuleActions[i]);
        }
        plugin->onNewSelection(m_appInterface->getSelectedEntities());
    }

    // add 3rd PCL party plugins to menu & tool bar (if any )
    if (!thirdPartyPCLPlugins.isEmpty()) {
        m_pclAlgorithmMenu->addSection("3rd pcl Party");

        for (ccPclPluginInterface *plugin : thirdPartyPCLPlugins) {
            QVector<QList<QAction *>> allModuleActions = plugin->getActions();
            QVector<QString> moduleNames = plugin->getModuleNames();
            assert(allModuleActions.size() == moduleNames.size());

            for (int i = 0; i < moduleNames.size(); ++i) {
                addActionsToMenu(moduleNames[i], allModuleActions[i]);
                addActionsToToolBar(moduleNames[i], allModuleActions[i]);
            }

            plugin->onNewSelection(m_appInterface->getSelectedEntities());
        }
    }

    m_pluginMenu->setEnabled(!m_pluginMenu->isEmpty());

    if (m_mainPluginToolbar->isEnabled()) {
        m_showPluginToolbar->setEnabled(true);
    }

    m_pclAlgorithmMenu->setEnabled(!m_pclAlgorithmMenu->isEmpty());
    m_pclAlgorithmsToolbar->setEnabled(
            !m_pclAlgorithmMenu->isEmpty());  // [sic] we have toolbar actions
                                              // if we have them in the menu

    m_showPluginToolbar->setChecked(m_mainPluginToolbar->isEnabled());

    if (m_pclAlgorithmsToolbar->isEnabled()) {
        m_showPCLAlgorithmToolbar->setEnabled(true);
    }

    m_showPCLAlgorithmToolbar->setChecked(m_pclAlgorithmsToolbar->isEnabled());
}

QMenu *ccPluginUIManager::pluginMenu() const { return m_pluginMenu; }

QMenu *ccPluginUIManager::pclAlgorithmMenu() const {
    return m_pclAlgorithmMenu;
}

QToolBar *ccPluginUIManager::mainPluginToolbar() { return m_mainPluginToolbar; }

QList<QToolBar *> &ccPluginUIManager::additionalPluginToolbars() {
    return m_additionalPluginToolbars;
}

QAction *ccPluginUIManager::actionShowMainPluginToolbar() {
    return m_showPluginToolbar;
}

QToolBar *ccPluginUIManager::glPclToolbar() { return m_pclAlgorithmsToolbar; }

QAction *ccPluginUIManager::actionShowPCLAlgorithmToolbar() {
    return m_showPCLAlgorithmToolbar;
}

bool ccPluginUIManager::isPythonPluginToolbar(QToolBar *toolbar) {
    if (!toolbar) {
        return false;
    }
    // Python plugin name is "Python Plugin" (from info.json)
    return toolbar->objectName() == "Python Plugin";
}

void ccPluginUIManager::updateMenus() {
    QWidget *active3DView = m_appInterface->getActiveWindow();
    const bool hasActiveView = (active3DView != nullptr);

    const QList<QAction *> actionList = m_pclAlgorithmActions.actions();

    for (QAction *action : actionList) {
        action->setEnabled(hasActiveView);
    }
}

void ccPluginUIManager::handleSelectionChanged() {
    const ccHObject::Container &selectedEntities =
            m_appInterface->getSelectedEntities();

    for (ccPluginInterface *plugin : m_plugins) {
        if (plugin->getType() == ECV_STD_PLUGIN) {
            ccStdPluginInterface *stdPlugin =
                    static_cast<ccStdPluginInterface *>(plugin);

            stdPlugin->onNewSelection(selectedEntities);
        } else if (plugin->getType() == ECV_PCL_ALGORITHM_PLUGIN) {
            ccPclPluginInterface *pclPlugin =
                    static_cast<ccPclPluginInterface *>(plugin);

            pclPlugin->onNewSelection(selectedEntities);
        }
    }
}

void ccPluginUIManager::showAboutDialog() const {
    ccPluginInfoDlg about;

    about.setPluginPaths(ccPluginManager::get().pluginPaths());
    about.setPluginList(m_plugins);

    about.exec();
}

void ccPluginUIManager::setupActions() {
    // m_actionRemovePCLAlgorithm = new
    // QAction(QIcon(":/CC/pluginManager/images/noAlgorithm.png"), tr("Remove
    // Algorithm"), this); m_actionRemovePCLAlgorithm->setEnabled(false);
    // connect(m_actionRemovePCLAlgorithm, &QAction::triggered, this,
    // &ccPluginUIManager::disablePCLAlgorithm);

    m_showPluginToolbar = new QAction(tr("Plugins"), this);
    m_showPluginToolbar->setCheckable(true);
    m_showPluginToolbar->setEnabled(false);

    m_showPCLAlgorithmToolbar = new QAction(tr("PCL Algorithm"), this);
    m_showPCLAlgorithmToolbar->setCheckable(true);
    m_showPCLAlgorithmToolbar->setEnabled(false);
}

void ccPluginUIManager::setupMenus() {
    m_pluginMenu = new QMenu(tr("Plugins"), m_parentWidget);
    m_pclAlgorithmMenu = new QMenu(tr("PCL ALgorithms"), m_parentWidget);
}

void ccPluginUIManager::addActionsToMenu(ccStdPluginInterface *stdPlugin,
                                         const QList<QAction *> &actions) {
    // If the plugin has more than one action we create its own menu
    if (actions.size() > 1) {
        QMenu *menu = new QMenu(stdPlugin->getName(), m_parentWidget);

        menu->setIcon(stdPlugin->getIcon());
        menu->setEnabled(true);

        for (QAction *action : actions) {
            menu->addAction(action);
        }

        m_pluginMenu->addMenu(menu);
    } else  // otherwise we just add it to the main menu
    {
        Q_ASSERT(actions.count() == 1);

        m_pluginMenu->addAction(actions.at(0));
    }
}

void ccPluginUIManager::addActionsToToolBar(ccStdPluginInterface *stdPlugin,
                                            const QList<QAction *> &actions) {
    const QString pluginName = stdPlugin->getName();

    // If the plugin has more than one action we create its own tool bar
    if (actions.size() > 1) {
        QToolBar *toolBar = new QToolBar(
                pluginName + QStringLiteral(" toolbar"), m_parentWidget);

        if (toolBar != nullptr) {
            m_additionalPluginToolbars.push_back(toolBar);

            toolBar->setObjectName(pluginName);
            toolBar->setEnabled(true);

            for (QAction *action : actions) {
                toolBar->addAction(action);
            }
        }
    } else  // otherwise we just add it to the main tool bar
    {
        Q_ASSERT(actions.count() == 1);

        m_mainPluginToolbar->addAction(actions.at(0));
    }
}

void ccPluginUIManager::addActionsToMenu(const QString &moduleName,
                                         const QList<QAction *> &actions) {
    // If the plugin has more than one action we create its own menu
    if (actions.size() > 1) {
        QMenu *menu = new QMenu(moduleName, m_parentWidget);
        menu->setEnabled(true);

        for (QAction *action : actions) {
            menu->addAction(action);
        }

        m_pclAlgorithmMenu->addMenu(menu);
    } else  // otherwise we just add it to the main menu
    {
        Q_ASSERT(actions.count() == 1);

        m_pclAlgorithmMenu->addAction(actions.at(0));
    }
}

void ccPluginUIManager::addActionsToToolBar(const QString &moduleName,
                                            const QList<QAction *> &actions) {
    // If the plugin has more than one action we create its own tool bar
    if (actions.size() > 1) {
        if (m_pclAlgorithmsToolbar != nullptr) {
            for (QAction *action : actions) {
                m_pclAlgorithmsToolbar->addAction(action);
            }
        }
    } else  // otherwise we just add it to the main tool bar
    {
        Q_ASSERT(actions.count() == 1);

        m_pclAlgorithmsToolbar->addAction(actions.at(0));
    }
}

void ccPluginUIManager::setupToolbars() {
    m_mainPluginToolbar = new QToolBar(tr("Plugins"), m_parentWidget);

    m_mainPluginToolbar->setObjectName(QStringLiteral("Main Plugin Toolbar"));

    connect(m_showPluginToolbar, &QAction::toggled, m_mainPluginToolbar,
            &QToolBar::setVisible);

    m_pclAlgorithmsToolbar = new QToolBar(tr("PCL ALgorithms"), m_parentWidget);

    m_pclAlgorithmsToolbar->setObjectName(QStringLiteral("PCL Plugin Toolbar"));

    connect(m_showPCLAlgorithmToolbar, &QAction::toggled,
            m_pclAlgorithmsToolbar, &QToolBar::setVisible);
}

void ccPluginUIManager::enablePCLAlgorithm() {
    QWidget *win = m_appInterface->getActiveWindow();

    if (win == nullptr) {
        CVLog::Warning("[PCL ALGORITHM] No active 3D view");
        return;
    }

    QAction *action = qobject_cast<QAction *>(sender());

    Q_ASSERT(action != nullptr);

    ccPclPluginInterface *plugin =
            action->data().value<ccPclPluginInterface *>();

    if (plugin == nullptr) {
        return;
    }

    m_actionRemovePCLAlgorithm->setEnabled(true);

    ecvConsole::Print(
            "Note: go to << PCL ALgorithms >> to disable PCL ALGORITHM");
}

void ccPluginUIManager::disablePCLAlgorithm() {
    if (m_appInterface->getActiveWindow() != nullptr) {
        m_actionRemovePCLAlgorithm->setEnabled(false);
        m_pclAlgorithmActions.checkedAction()->setChecked(false);
    }
}
