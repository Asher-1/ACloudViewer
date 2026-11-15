// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qOCR: PP-OCR text detection/recognition over the AICore `ocr` task.
// The plugin owns the worker lifecycle (device task lock, cancel token,
// context create/release on the worker thread), the heartbeat log, and the
// DB export of the annotated recognition result —
// the dialog stays a pure view (same split as qRMBG).

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QHash>
#include <QList>
#include <QTimer>

#include "OCRDialog.h"

class OCRWorker;
class ccHObject;
class ecvMainAppInterface;

class qOCR final : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qOCR" FILE "../info.json")

public:
    explicit qOCR(QObject* parent = nullptr);

    QList<QAction*> getActions() override;

    void registerCommands(ccCommandLineInterface* cmd) override;
    void setMainAppInterface(ecvMainAppInterface* app) override;

private slots:
    void showDialog();
    void executeTask(const OCRRunSettings& settings);
    void cancelTask();
    void onResultReady(const OCRRunResult& result);
    void onTaskFinished(bool success);
    /** The dialog's result selector moved to \p imagePath: select the
     *  matching exported overlay in the DB tree (no-op when nothing was
     *  exported for that source or the entity was deleted). */
    void selectDbResultImage(const QString& imagePath);

private:
    void addResultToDb(const OCRRunResult& result,
                       const OCRRunSettings& settings);
    /** Depth-first DB tree lookup of the uniquely named overlay entity. */
    ccHObject* findDbEntity(const QString& name) const;

    QAction* m_workspaceAction = nullptr;
    OCRDialog* m_dialog = nullptr;
    OCRWorker* m_worker = nullptr;
    OCRRunSettings m_currentSettings;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    /** DB names of this session's exported overlays keyed by source image
     *  path (the last overlay per path); the owner of the DB identity the
     *  dialog's result selector syncs with. Names (not pointers): the
     *  entity is resolved in the DB tree at selection time, so a deleted
     *  entity simply drops out of the sync. */
    QHash<QString, QString> m_dbResultImages;
};
