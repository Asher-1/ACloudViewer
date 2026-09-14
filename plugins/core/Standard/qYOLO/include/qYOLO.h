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

#include "YOLODialog.h"
#include "YOLOWorker.h"

class ccImage;

class qYOLO : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qYOLO" FILE "../info.json")

public:
    explicit qYOLO(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const YOLODialog::Settings& settings);
    void cancelTask();
    void onResultReady(const YOLORunResult& result);
    void onDepthResultReady(const YOLODepthResult& result);
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
    void addResultToDb(const YOLORunResult& result,
                       const YOLODialog::Settings& settings,
                       const QString& sourceLabel);
    void addDepthResultToDb(const YOLODepthResult& result,
                            const YOLODialog::Settings& settings,
                            const QString& sourceLabel);

    QAction* m_action = nullptr;
    YOLODialog* m_dialog = nullptr;
    YOLOWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    YOLODialog::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
