// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvStdPluginInterface.h"

//! A point-clouds filtering algorithm utilize cloth simulation process.
class qCSF : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "ecvcorp.cloudviewer.plugin.qCSF" FILE "../info.json")

public:
    //! Default constructor
    explicit qCSF(QObject* parent = nullptr);

    virtual ~qCSF() = default;

    // inherited from ccStdPluginInterface
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;
    virtual QList<QAction*> getActions() override;
    virtual void registerCommands(ccCommandLineInterface* cmd) override;

protected slots:

    //! Slot called when associated ation is triggered
    void doAction();

protected:
    //! Associated action
    QAction* m_action;
};
