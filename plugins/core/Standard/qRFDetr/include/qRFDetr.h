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

#include "RFDetrDialog.h"
#include "RFDetrWorker.h"

class ccImage;

class qRFDetr : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qRFDetr" FILE
                          "../info.json")

public:
    explicit qRFDetr(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const RFDetrDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const RFDetrRunResult& result);
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
    void addResultToDb(const RFDetrRunResult& result,
                       const RFDetrDialog::Settings& settings,
                       const QString& sourceLabel);

    QAction* m_action = nullptr;
    RFDetrDialog* m_dialog = nullptr;
    RFDetrWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    RFDetrDialog::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
