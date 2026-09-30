// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qOCR: PP-OCR text detection/recognition over the AICore `ocr` task.
// The plugin owns the worker lifecycle (device task lock, cancel token,
// context create on the worker thread / release on the main thread), the
// heartbeat log, and the DB export of the annotated recognition result —
// the dialog stays a pure view (same split as qRMBG).

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QList>
#include <QTimer>

#include "OCRDialog.h"

class OCRWorker;
class ecvMainAppInterface;

class qOCR final : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qOCR" FILE "../info.json")

public:
    explicit qOCR(QObject* parent = nullptr);

    QList<QAction*> getActions() override;
    void setMainAppInterface(ecvMainAppInterface* app) override;

private slots:
    void showDialog();
    void executeTask(const OCRRunSettings& settings);
    void cancelTask();
    void onResultReady(const OCRRunResult& result);
    void onTaskFinished(bool success);

private:
    void addResultToDb(const OCRRunResult& result,
                       const OCRRunSettings& settings);

    QAction* m_workspaceAction = nullptr;
    OCRDialog* m_dialog = nullptr;
    OCRWorker* m_worker = nullptr;
    OCRRunSettings m_currentSettings;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
};
