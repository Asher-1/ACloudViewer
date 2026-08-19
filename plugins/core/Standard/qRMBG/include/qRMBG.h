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
#include <QTimer>

#include "RMBGDialog.h"
#include "RMBGWorker.h"

class ccImage;

class qRMBG : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qRMBG" FILE "../info.json")

public:
    explicit qRMBG(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const RMBGDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const RMBGRunResult& result);
    void onTaskFinished(bool success);
    void onWorkerProgress(int current, int total);

private:
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    bool resolveInputPath(const QString& rawPath,
                          QString& outPath,
                          QString* errorMsg);
    void clearStagedInputFiles();
    void refreshDbImages();
    void addResultToDb(const RMBGRunResult& result,
                       const RMBGDialog::Settings& settings,
                       const QString& sourceLabel);
    void saveResultPng(const RMBGRunResult& result, const QString& sourceLabel);

    QAction* m_action = nullptr;
    RMBGDialog* m_dialog = nullptr;
    RMBGWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    RMBGDialog::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
