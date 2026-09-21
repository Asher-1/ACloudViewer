// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>

#include "Sam3dDialog.h"
#include "Sam3dWorker.h"

class ccMesh;
class ecvMainAppInterface;

class qSAM3D : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qSAM3D" FILE
                          "../info.json")

public:
    explicit qSAM3D(QObject* parent = nullptr);

    QList<QAction*> getActions() override;
    /** Headless CLI: registers -SAM3D_GENERATE (batch Gaussian PLY /
     *  mesh generation over N images, single model load, RESULT_JSON
     *  manifest output). */
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void executeTask(const Sam3dDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const Sam3dRunResult& result);
    void onTaskFinished(bool success);
    void onWorkerStage(int stage, int step, int total);

private:
    void addResultToDb(const Sam3dRunResult& result);

    QAction* m_action = nullptr;
    Sam3dDialog* m_dialog = nullptr;
    Sam3dWorker* m_worker = nullptr;
};
