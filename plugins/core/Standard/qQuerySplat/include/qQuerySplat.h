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

#include "QuerySplatDialog.h"
#include "QuerySplatWorker.h"

class ccCommandLineInterface;

class qQuerySplat : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qQuerySplat" FILE
                          "../info.json")

public:
    explicit qQuerySplat(QObject* parent = nullptr);
    ~qQuerySplat() override;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    /** Headless CLI: registers -QUERY_SPLAT (image-to-3DGS prediction +
     *  RESULT_JSON). */
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void executeTask(const QuerySplatDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const QuerySplatResult& result);
    void onTaskFinished();

private:
    /** Imports the typed result into the DB tree: gaussian centers + colors
     *  as a ccPointCloud, predicted cameras as ccCameraSensor, depth /
     *  render rasters as ccImage children. */
    void addResultToDb(const QuerySplatResult& result);

    QAction* m_action = nullptr;
    QuerySplatDialog* m_dialog = nullptr;
    QuerySplatWorker* m_worker = nullptr;
    QuerySplatDialog::Settings m_currentSettings;
    QuerySplatResult m_lastResult;
    ccHObject::Container m_selectedEntities;
};
