// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvStdPluginInterface.h"

// qCC_db
#include <ecvHObject.h>

class ccCommandLineInterface;

class qVoxFall : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qVoxFall" FILE
                          "../info.json")

public:
    explicit qVoxFall(QObject *parent = nullptr);
    ~qVoxFall() override = default;

    // Inherited from ccStdPluginInterface
    void onNewSelection(const ccHObject::Container &selectedEntities) override;
    QList<QAction *> getActions() override;
    virtual void registerCommands(ccCommandLineInterface *cmd) override;

private:
    void doAction();

    //! Default action
    QAction *m_action;

    //! Currently selected entities
    ccHObject::Container m_selectedEntities;
};
