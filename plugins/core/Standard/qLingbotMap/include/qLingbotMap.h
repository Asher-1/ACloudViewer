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
    /** ALIGNED per-window batches: appended into the ONE persistent
     *  "LingbotMap_aligned_points" cloud (first window's coordinate frame),
     *  so the live map accumulates across window boundaries like the
     *  official viewer instead of being hidden with the finished window. */
    void onAlignedPreview(const LingbotAlignedPreview& preview);
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
    /** Creates the transient online group on first use. */
    ccHObject* ensureOnlineGroup();
    /** The "LingbotMap_cameras" subfolder of the online group: every online
     *  camera frustum lives here, never mixed with the point clouds. */
    ccHObject* ensureCamerasGroup();
    /** The persistent ALIGNED map cloud of the online group (created on
     *  first use; every aligned window batch appends into it). */
    ccPointCloud* onlineAlignedCloud();
    /** The per-stream accumulating preview cloud ("LingbotMap_points",
     *  streaming runs; windowed runs accumulate the aligned cloud instead). */
    ccPointCloud* onlineCloud(ccHObject* parent);
    /** COLMAP-style camera frustum at \p c2w (size from the running median
     *  camera baseline, official viewer semantics), appended into the
     *  cameras subfolder. */
    void addOnlineCamera(const float* c2w,
                         const QVector<float>& intrinsics,
                         int width,
                         int height,
                         int globalIndex);
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
    std::vector<float> m_onlineBaselines; /**< inter-camera distances */
    QVector<float> m_lastPreviewC2w;      /**< previous frame pose */
    QElapsedTimer m_onlineRefreshThrottle;
    /** Point count of the accumulating preview cloud; stretches the DB
     *  refresh interval as the per-refresh VTK re-conversion grows. */
    unsigned m_onlineCloudPoints = 0;
    /** Persistent ALIGNED map cloud (windowed runs): every aligned window
     *  batch appends into it, so history stays visible across windows. */
    ccPointCloud* m_onlineAlignedCloud = nullptr;
    unsigned m_alignedCloudPoints = 0;

    // ---- loop playback over the final result (camera-frustum cycle) ----
    QTimer* m_playbackTimer = nullptr;
    bool m_playbackEnabled = false;
    bool m_playbackCurrentFrameOnly = true;
    int m_playbackFps = 20;
    int m_playbackFrame = 0;
    unsigned m_resultGroupId = 0;
    /** Per-frame camera sensors of the final result (playback cycles
     *  their visibility; the map itself is a single merged cloud).
     *  Cached pointers: a per-tick ccHObject::find over the whole group is
     *  O(cameras x tree) at playback FPS — the pointers die together with
     *  the group, which onPlaybackTick re-checks by id every tick. */
    std::vector<ccHObject*> m_frameCameras;
};
