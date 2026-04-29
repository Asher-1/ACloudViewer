// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QObject>

#include "ecvStdPluginInterface.h"

class QAction;
class ccCommandLineInterface;

//! Mes Boolean Operations (CSG) plugin
/** This plugin is based on Cork: https://github.com/gilbo/cork
        Required implementation is CC's dedicated fork:
https://github.com/cloudcompare/cork
**/
class qCork : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qCork" FILE
                          "../info.json")

public:
    //! Default constructor
    explicit qCork(QObject* parent = nullptr);

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(const ccHObject::Container& selectedEntities);
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:

    //! Starts main action
    void doAction();

protected:
    //! Associated action
    QAction* m_action;
};
