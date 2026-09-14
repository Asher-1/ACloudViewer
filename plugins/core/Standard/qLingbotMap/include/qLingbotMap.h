// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvMainAppInterface.h>
#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QTimer>

#include "LingbotMapDialog.h"
#include "LingbotMapWorker.h"

class qLingbotMap : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qLingbotMap" FILE
                          "../info.json")

public:
    explicit qLingbotMap(QObject* parent = nullptr);

    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const LingbotMapWorker::Settings& settings);
    void cancelTask();
    void onResultReady(const LingbotRunResult& result);
    void onTaskFinished(bool success);

private:
    bool addResultToDb(const LingbotRunResult& result,
                       const LingbotMapWorker::Settings& settings);

    QAction* m_action = nullptr;
    LingbotMapDialog* m_dialog = nullptr;
    LingbotMapWorker* m_worker = nullptr;
    LingbotRunResult m_pendingResult;
    LingbotMapWorker::Settings m_lastSettings;
    QTimer* m_inferenceHeartbeat = nullptr;
    qint64 m_inferenceElapsedSeconds = 0;
};
