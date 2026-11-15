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

#include "VLXSeekDialog.h"
#include "VLXSeekWorker.h"

class ccImage;

class qVLXSeek : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qVLXSeek" FILE
                          "../info.json")

public:
    explicit qVLXSeek(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void onResultReady(const VLXSeekRunResult& result);

private:
    void addResultToDb(const VLXSeekRunResult& result,
                       const VLXSeekWorker::Settings& settings);

    QAction* m_action = nullptr;
    VLXSeekDialog* m_dialog = nullptr;
    VLXSeekWorker::Settings m_lastSettings;
};
