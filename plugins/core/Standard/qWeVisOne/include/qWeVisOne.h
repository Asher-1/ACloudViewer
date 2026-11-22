// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvStdPluginInterface.h>

#include <QAction>

#include "WeVisOneDialog.h"
#include "WeVisOneWorker.h"

class ccImage;

class qWeVisOne : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qWeVisOne" FILE
                          "../info.json")

public:
    explicit qWeVisOne(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void onResultReady(const WeVisOneRunResult& result);

private:
    void addResultToDb(const WeVisOneRunResult& result,
                       const WeVisOneWorker::Settings& settings);

    QAction* m_action = nullptr;
    WeVisOneDialog* m_dialog = nullptr;
    WeVisOneWorker::Settings m_lastSettings;
};