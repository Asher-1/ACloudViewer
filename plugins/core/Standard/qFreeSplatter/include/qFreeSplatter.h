// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvStdPluginInterface.h>

#include <QByteArray>
#include <memory>

#include "FreeSplatterDialog.h"
#include "FreeSplatterWorker.h"

Q_DECLARE_METATYPE(FreeSplatterDialog::Settings)

class ccPointCloud;
class ccImage;
class QTimer;

class qFreeSplatter : public QObject, public ccStdPluginInterface {
    Q_OBJECT
    Q_INTERFACES(ccPluginInterface ccStdPluginInterface)
    Q_PLUGIN_METADATA(IID "cvcorp.cloudviewer.plugin.qFreeSplatter" FILE
                          "../info.json")

public:
    explicit qFreeSplatter(QObject* parent = nullptr);
    ~qFreeSplatter() override = default;

    void onNewSelection(const ccHObject::Container& selectedEntities) override;
    QList<QAction*> getActions() override;

private slots:
    void showDialog();
    void executeTask(const FreeSplatterDialog::Settings& settings);
    void cancelTask();
    void onResultReady(const FreeSplatterResult& result);
    void onModelInfo(const QString& info);
    void onTaskFinished(bool success);
#ifdef HAS_QSIBR
    void onVisualizeRequested();
#endif
    void onExportPlyRequested();
    void onWorkerProgress(int current, int total);

private:
    ccPointCloud* buildResultPointCloud(
            const FreeSplatterResult& result,
            float opacityThreshold,
            FreeSplatterDialog::ExportFieldMode exportFieldMode,
            const QString& cloudName,
            int* validCountOut) const;
    void addResultToDb(const FreeSplatterResult& result);
    void addCameraPosesToDb(const FreeSplatterResult& result,
                            const QString& baseName,
                            ccPointCloud* cloud);
    void refreshDbImages();
    ccImage* findDbImage(const QString& name) const;
    QStringList selectedDbImageNames() const;
    bool resolveInputPaths(const QStringList& rawPaths,
                           QStringList& outPaths,
                           QString* errorMsg) const;
    QByteArray buildSibrCamerasJson(const FreeSplatterResult& result,
                                    float opacityThreshold = 0.05f);
    bool warmupInferenceBackend(const QString& device, QString* logMsg) const;
    bool launchSibrGaussianViewer(const QByteArray& plyBytes,
                                  const QByteArray& camerasJson,
                                  int shDegree);

    QAction* m_action = nullptr;
    FreeSplatterDialog* m_dialog = nullptr;
    FreeSplatterWorker* m_worker = nullptr;
    QTimer* m_inferenceHeartbeat = nullptr;
    int m_inferenceElapsedSeconds = 0;
    FreeSplatterDialog::Settings m_currentSettings;
    ccHObject::Container m_selectedEntities;

    FreeSplatterResult m_lastResult;
    QByteArray m_lastPlyBytes;
#ifdef HAS_QSIBR
    QByteArray m_lastSibrCamerasJson;
#endif
    ccPointCloud* m_lastDbCloud = nullptr;
    ccHObject* m_lastDbCameraGroup = nullptr;
};
