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
    /** Value snapshot (clone) of the source cloud taken in executeTask so
     *  the result callback annotates the exact Run-time input even if the
     *  user deletes or replaces the DB entity while the run is in flight. */
    ccPointCloud* m_runSourceSnapshot = nullptr;
    /** Set by cancelTask: late results after a cancel are logged but never
     *  exported into the DB tree. */
    bool m_runCancelled = false;
};
