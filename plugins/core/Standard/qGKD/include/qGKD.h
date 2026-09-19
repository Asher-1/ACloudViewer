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

#include "GKDDialog.h"
#include "GKDWorker.h"

class ccImage;

class qGKD : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qGKD" FILE "../info.json")

public:
    explicit qGKD(QObject* parent = nullptr);
    ~qGKD() override;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const GKDWorker::Settings& settings);
    void cancelTask();
    void onResultReady(const GKDRunResult& result);
    void onTaskFinished(bool success);

private:
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    bool resolveInputPath(const QString& rawPath,
                          QString* outPath,
                          QString* errorMsg);
    void clearStagedInputFiles();
    void refreshDbImages();
    void addResultToDb(const GKDRunResult& result,
                       const GKDWorker::Settings& settings);
    void saveResultPng(const GKDRunResult& result);
    /** Free contexts replaced by a newer load (main thread only). */
    void drainRetiredContexts();
    /** Free the resident GKD / YOLO-World contexts (main thread only).
     *  Bound to aboutToQuit: freeing at exit-time library finalization
     *  hits the already-torn-down ggml CUDA runtime and aborts. */
    void releaseResidentContexts();

    QAction* m_action = nullptr;
    GKDDialog* m_dialog = nullptr;
    GKDWorker* m_worker = nullptr;
    /** Resident GKD / YOLO-World contexts reused across runs (the fix for
     *  per-run reload lag); see GKDContextCache. */
    GKDContextCache m_ctxCache;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    GKDWorker::Settings m_currentSettings;
    QStringList m_stagedInputFiles;
    ccHObject::Container m_selectedEntities;
};
