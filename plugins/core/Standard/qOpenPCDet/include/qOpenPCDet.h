// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvPointCloud.h>
#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QTimer>

#include "OpenPCDetDialog.h"
#include "OpenPCDetWorker.h"

class qOpenPCDet : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qOpenPCDet" FILE
                          "../info.json")

public:
    explicit qOpenPCDet(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;
    void registerCommands(ccCommandLineInterface* cmd) override;

private slots:
    void showDialog();
    void executeTask(const OpenPCDetDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const OpenPCDetRunResult& result);
    void onTaskFinished(bool success);
    void onWorkerProgress(int current, int total);

private:
    ccPointCloud* firstSelectedCloud() const;
    void addBoxesToDb(const OpenPCDetRunResult& result);
    void exportAnnotatedCloud(const OpenPCDetRunResult& result);

    QAction* m_action = nullptr;
    OpenPCDetDialog* m_dialog = nullptr;
    OpenPCDetWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    OpenPCDetDialog::Settings m_currentSettings;
    ccHObject::Container m_selectedEntities;
    /** Source cloud of the running task (cached in executeTask so the
     *  result callback annotates the exact input, not the user's current
     *  selection; same DB-tree stability window as the wireframe export
     *  that runs in the same callback). */
    ccPointCloud* m_runSourceCloud = nullptr;
};
