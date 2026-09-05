// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QTimer>

#include "TrellisDialog.h"
#include "TrellisWorker.h"

class ccHObject;
class ccMesh;
class ecvMainAppInterface;

class qTrellis : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qTrellis" FILE
                          "../info.json")

public:
    explicit qTrellis(QObject* parent = nullptr);

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const TrellisDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const TrellisRunResult& result);
    void onTaskFinished(bool success);
    void onWorkerProgress(int stage, int step, int total);
    /** Export page: re-bake the textured GLB from the last generation. */
    void onExportRequested();

private:
    bool resolveInputPath(const QString& rawPath,
                          QString& outPath,
                          QString* errorMsg);
    void addResultToDb(const TrellisRunResult& result,
                       const TrellisDialog::Settings& settings);
    void addRmbgImageToDb(const TrellisRunResult& result,
                          const TrellisDialog::Settings& settings);
    void saveResultGlb(const TrellisRunResult& result,
                       const TrellisDialog::Settings& settings,
                       const QString& sourceLabel);
    /** saveResultGlb with explicit bake settings (export page). */
    void saveResultGlbEx(const TrellisRunResult& result,
                         const TrellisDialog::Settings& settings,
                         const QString& sourceLabel,
                         int textureSize,
                         int componentFilter);
    /** Bake the UV-atlas GLB from a typed result with explicit settings
     *  (AICore call; logs and returns empty on failure). */
    QByteArray bakeResultGlb(const TrellisRunResult& result,
                             int textureSize,
                             int componentFilter);
    /** Timestamped GLB file path for a source label. */
    QString glbFilePath(const QString& dir, const QString& sourceLabel) const;
    /** Write GLB bytes to path (logs the destination on success). */
    bool writeGlbFile(const QByteArray& glb, const QString& path);
    /** Import GLB bytes through the shared file filters into a named,
     *  metadata-tagged entity (full PBR material display). Null on failure
     *  (logged). */
    ccHObject* importGlbEntity(const QByteArray& glb,
                               const TrellisRunResult& result,
                               const QString& entityName);

    QAction* m_action = nullptr;
    TrellisDialog* m_dialog = nullptr;
    TrellisWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    TrellisDialog::Settings m_currentSettings;
};
