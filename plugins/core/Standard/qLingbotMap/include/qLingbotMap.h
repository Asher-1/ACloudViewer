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
#include <QElapsedTimer>
#include <QTimer>
#include <vector>

#include "LingbotMapDialog.h"
#include "LingbotMapWorker.h"

class ccHObject;
class ccPointCloud;

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
    /** Live online-reconstruction preview (stride-subsampled points
     *  APPEND into one growing cloud per window + COLMAP-style camera
     *  frustum per completed frame). */
    void onFramePreview(const LingbotFramePreview& preview);
    /** Dialog playback controls (official viewer Playing/FPS semantics). */
    void onPlaybackSettingsChanged(bool enabled,
                                   int fps,
                                   bool currentFrameOnly);
    void onPlaybackTick();

private:
    bool addResultToDb(const LingbotRunResult& result,
                       const LingbotMapWorker::Settings& settings);
    /** Removes the transient online-preview group from the DB. */
    void disposeOnlineGroup();
    /** Window subgroup of the online group (streaming: the group itself). */
    ccHObject* onlineWindowGroup(int windowIndex, int windowCount);
    /** The per-window accumulating preview cloud ("LingbotMap_points"),
     *  created on first use; streaming frames APPEND into it so the DB
     *  tree holds a single map entity while the engine streams. */
    ccPointCloud* onlineCloud(ccHObject* parent);
    /** COLMAP-style camera sensor at the preview pose (size from the
     *  running median camera baseline, official viewer semantics). */
    void addOnlineCamera(ccHObject* parent, const LingbotFramePreview& p);
    void startPlayback();
    void stopPlayback();

    QAction* m_action = nullptr;
    LingbotMapDialog* m_dialog = nullptr;
    LingbotMapWorker* m_worker = nullptr;
    LingbotRunResult m_pendingResult;
    LingbotMapWorker::Settings m_lastSettings;
    QTimer* m_inferenceHeartbeat = nullptr;
    qint64 m_inferenceElapsedSeconds = 0;

    // ---- online preview (transient; replaced by the final result) ----
    ccHObject* m_onlineGroup = nullptr;
    ccHObject* m_onlineCurrentWindow = nullptr;
    int m_onlineWindowCount = 0;
    std::vector<float> m_onlineBaselines; /**< inter-camera distances */
    QVector<float> m_lastPreviewC2w;      /**< previous frame pose */
    QElapsedTimer m_onlineRefreshThrottle;

    // ---- loop playback over the final result (camera-frustum cycle) ----
    QTimer* m_playbackTimer = nullptr;
    bool m_playbackEnabled = false;
    bool m_playbackCurrentFrameOnly = true;
    int m_playbackFps = 20;
    int m_playbackFrame = 0;
    unsigned m_resultGroupId = 0;
    /** Per-frame camera sensors of the final result (playback cycles
     *  their visibility; the map itself is a single merged cloud). */
    std::vector<unsigned> m_frameCameraIds;
};
