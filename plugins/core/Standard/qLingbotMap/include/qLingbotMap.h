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

    /** RPC/test driving (agent integration): thin Q_INVOKABLE forwards to
     *  the private replay/display-state entry points so the JSON-RPC
     *  driver can exercise the EXACT user-visible paths (start/pause/stop
     *  replay, seek, slider-commit semantics) without exposing them to
     *  normal UI code. Each wrapper adds no logic of its own. */
    Q_INVOKABLE void rpcStartPlayback();
    Q_INVOKABLE void rpcPausePlayback();
    Q_INVOKABLE void rpcStopPlayback();
    Q_INVOKABLE void rpcSeekFrame(const QVariant& frame);
    /** Net effect of a Result-display slider commit (single writer). */
    Q_INVOKABLE void rpcApplyCameraDisplay(const QVariant& showCameras,
                                           const QVariant& camStride,
                                           const QVariant& camSizePercent);
    /** Opens the dialog (if needed) and forwards to LingbotMapDialog. */
    Q_INVOKABLE void rpcStartTestScene(const QVariant& scene);

private slots:
    void showDialog();
    void executeTask(const LingbotMapWorker::Settings& settings);
    void cancelTask();
    void onResultReady(const LingbotRunResult& result);
    void onTaskFinished(bool success);
    /** Live online-reconstruction preview: ONE IMMUTABLE point cloud per
     *  frame ("LingbotMap_frames/LingbotMap_frame_%06d", the official
     *  fused-viewer structure — the same shape as the final result) plus a
     *  COLMAP-style camera frustum per completed frame. Immutability is
     *  what keeps the live refresh O(new points): stable clouds never
     *  re-enter the engine's per-entity geometry rebuild, while a single
     *  growing cloud forced a full O(total) CPU conversion plus a full
     *  vertex-buffer re-upload on every refresh. */
    void onFramePreview(const LingbotFramePreview& preview);
    /** ALIGNED per-window batches: each window's re-based points become
     *  one immutable cloud in "LingbotMap_aligned_frames/" (first window's
     *  coordinate frame), so the live map accumulates across window
     *  boundaries like the official viewer at O(batch) refresh cost. */
    void onAlignedPreview(const LingbotAlignedPreview& preview);
    /** Replay control (official viewer Playing semantics, explicit
     *  three-state machine): Play starts/resumes the loop, Pause freezes
     *  on the current frame (official "Playing unchecked" behavior —
     *  visibility is NOT restored), Stop restores every frame and rewinds.
     *  Seek/step apply the current mode's visibility for one frame while
     *  stopped/paused. */
    void onReplayPlay();
    void onReplayPause();
    void onReplayStop();
    void onReplaySeek(int frame);
    /** Dialog playback controls (FPS / replay mode / auto-play preset). */
    void onPlaybackSettingsChanged(bool enabled,
                                   int fps,
                                   bool currentFrameOnly);
    /** Reset the replay: stop, restore every frame, rewind. */
    void onPlaybackReset();
    /** Hot-swap of the live camera policy while a reconstruction runs:
     *  updates the running settings and applies the new mode immediately
     *  from the cached latest frame. */
    void onLiveViewModeChanged(int mode);
    /** Result-display adjustments (official viewer sliders parity), applied
     *  to the LAST materialized result group: visibility-threshold
     *  re-filtering, point size, camera size/visibility/stride. */
    void onResultDisplayChanged(double threshold,
                                int pointSize,
                                int camSizePercent,
                                bool showCameras,
                                int camStride);
    /** Result view shortcuts: 0 = move to the first camera, 1 = fit all. */
    void onResultViewRequested(int which);
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
    /** The "LingbotMap_aligned_frames" subfolder (created on first use). */
    ccHObject* ensureAlignedFramesFolder();
    /** The "LingbotMap_aligned_frames" subfolder cloud for one window
     *  batch (created on first use; immutable afterwards). */
    ccPointCloud* onlineAlignedBatchCloud(int windowIndex);
    /** The "LingbotMap_frames" subfolder cloud of one preview frame
     *  ("LingbotMap_frame_%06d", created on first use; immutable
     *  afterwards) — mirrors the final result's per-frame structure. */
    ccPointCloud* onlineFrameCloud(ccHObject* parent, int globalIndex);
    /** Measured adaptive DB-refresh interval: ~2x the worst sampled
     *  refresh cost, clamped to [30fps floor, 250ms burst ceiling]. */
    qint64 onlineRefreshIntervalMs() const;
    /** Live-preview camera behavior (settings.liveViewMode): Global =
     *  never touch the camera; Fit = the official StreamingViewer
     *  _fit_view (mean center of the newest frame's points, fixed
     *  diagonal eye, re-fit at frames 1/10/50/120); Follow = the view
     *  rides the newest camera frustum (OpenCV convention: +Z forward,
     *  -Y up). \p frameCount1Based is the official success counter. */
    void updateLiveViewCamera(const float* c2w,
                              const float* xyz,
                              unsigned n,
                              int frameCount1Based);
    /** Applies one live-camera pose to the active view (geometry only —
     *  the CALLER must trigger a Render afterwards). \p pullback moves the
     *  eye BACK along the view direction (Follow branch) so a "go to
     *  camera" view does not sit inside the ridden frustum's near plane. */
    void applyLiveCameraPose(int mode,
                             const float* c2w,
                             const CCVector3d& center,
                             double radius,
                             double pullback = 0.0);
    /** Ends the Follow-mode camera ride: un-hides the ridden online sensor
     *  and clears the view's apply-viewport preview state (armed by
     *  ccCameraSensor::applyViewport, which suppresses the ridden sensor's
     *  gizmo draw AND removes its actors). Without this the last ridden
     *  frustum never renders again after leaving Follow mode. */
    void endLiveCameraRide();
    /** COLMAP-style camera frustum at \p c2w (size from the running median
     *  camera baseline, official viewer semantics), appended into the
     *  cameras subfolder with its first-draw forceRedraw latch set. */
    void addOnlineCamera(const float* c2w,
                         const QVector<float>& intrinsics,
                         int width,
                         int height,
                         int globalIndex);
    /** Replay state machine (Stopped/Playing/Paused): Play enters
     *  Playing (from frame 0 when coming from Stopped, with the official
     *  global-view refit), Pause freezes on the current frame WITHOUT
     *  touching visibility (official "Playing unchecked" semantics),
     *  Stop restores every frame and rewinds to 0. seekPlayback applies
     *  the current mode's visibility for one arbitrary frame. Every
     *  transition reports back through the dialog's setReplayState. */
    enum class ReplayState { Stopped, Playing, Paused };
    void startPlayback();
    /** SINGLE WRITER for the Result-display state over the GOVERNED entity
     *  set: the materialized result cameras/clouds when present, else the
     *  streaming online-preview trail (the sliders stay live mid-run).
     *  Applies, in one pass:
     *    camera visible = Show cameras && (ordinal % stride == 0) &&
     *                     frameGate (official replay rule while a replay
     *                     runs, always true when stopped)
     *    camera scale   = baseScale * Camera size% (per-sensor base)
     *    cloud visible  = frameGate
     *  plus the engine reconcile (composite blocks + "#plane" faces).
     *  Tick, seek, slider changes and playback restore all route here —
     *  there is no second visibility owner to fight the sliders. */
    void applyDisplayState();
    /** Re-derives the replay frame caches from the LIVE result folders
     *  (LingbotMap_cameras / LingbotMap_frames children). Called on every
     *  applyDisplayState pass so external DB edits (user deleting cameras
     *  or clouds while a replay is paused/running) can never leave
     *  dangling cached pointers. */
    void rebuildFrameCachesFromResult(ccHObject* group);
    /** Replay-owned camera/cloud visibility: official frame rule (3D
     *  incremental / 4D current) COMBINED with the result-display density
     *  (Show cameras / Camera stride). Thin wrapper over applyDisplayState
     *  kept for the tick/seek call-site readability. */
    void applyReplayVisibility();
    void pausePlayback();
    void stopPlaybackAndRestore();
    void seekPlayback(int frame);
    void syncReplayStateToDialog();
    /** Mirrors the replayed frame's input image into the dialog preview
     *  (reconstruction ↔ input-stream lockstep, official frame panel). */
    void syncReplayPreviewToDialog();

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
    /** Total preview points streamed so far (reported in the finish log). */
    unsigned m_onlineCloudPoints = 0;
    /** Number of aligned window batches materialized so far. */
    int m_alignedBatchCount = 0;
    /** Measured main-thread refresh costs (apply + redraw scope, ms) for
     *  the adaptive interval and the end-of-run latency report. */
    std::vector<qint64> m_refreshCostSamples;
    qint64 m_refreshCostTotalMs = 0;

    /** Latest streamed frame's extent/pose cache — lets a mid-run Live
     *  camera mode switch apply immediately (no next-frame wait). */
    CCVector3d m_lastFrameCenter{0, 0, 0};
    double m_lastFrameRadius = 0.0;
    float m_lastFrameC2w[16] = {};
    bool m_lastFrameHasPoints = false;
    /** World up, DERIVED from the streamed cameras' own up vectors (EMA
     *  smoothed). The engine world frame is the first camera's frame
     *  (OpenCV: +Z forward, +Y down), so no world axis letter is "up" by
     *  construction — hard-coding one rolled the Fit view upside down. */
    CCVector3d m_worldUp{0, 0, 0};
    /** Last point size actually pushed to the clouds (-1 = never). Gates
     *  the per-commit cloud walk so moving ANY other slider does not
     *  restomp the clouds with the point-size slider's current value. */
    int m_appliedPointSize = -1;

    // ---- loop playback over the final result (camera-frustum cycle) ----
    QTimer* m_playbackTimer = nullptr;
    ReplayState m_replayState = ReplayState::Stopped;
    /** Auto-play preset (the dialog checkbox): start the replay as soon
     *  as a result is materialized. */
    bool m_autoPlayAfterRun = false;
    bool m_playbackCurrentFrameOnly = true;
    int m_playbackFps = 20;
    int m_playbackFrame = 0;
    unsigned m_resultGroupId = 0;
    /** Replay index → m_pendingResult.frames index (the camera loop skips
     *  degenerate frames, so the two index spaces can diverge). */
    std::vector<int> m_replayResultIndex;
    /** The online sensor currently ridden by Follow-camera mode (hidden
     *  while ridden: the eye sits on its pose; restored on mode change). */
    ccCameraSensor* m_lastOnlineSensor = nullptr;
    /** Per-frame camera sensors of the final result (playback cycles
     *  their visibility; the map itself is a single merged cloud).
     *  Cached pointers: a per-tick ccHObject::find over the whole group is
     *  O(cameras x tree) at playback FPS — the pointers die together with
     *  the group, which onPlaybackTick re-checks by id every tick. */
    std::vector<ccHObject*> m_frameCameras;
    /** Per-frame point clouds (official viewer structure, one cloud per
     *  frame) in the same global order as m_frameCameras; used by the 4D
     *  playback to show the current frame's points. */
    std::vector<ccPointCloud*> m_frameClouds;

    // ---- result-display state (official viewer sliders parity) ----
    /** The materialization-time confidence threshold: the lower bound of
     *  the live threshold slider (points below it were never materialized
     *  — re-filtering can only tighten the display, never loosen it). */
    double m_resultConfMin = 1.5;
    /** Currently applied display threshold (dedupes re-filter passes). */
    double m_resultConfApplied = 1.5;
    /** Base graphic scale captured at materialization (the camera-size
     *  slider is a percentage of it). */
    double m_cameraBaseScale = 1.0;
    /** The "LingbotMap_frames" folder of the last result (re-filtering
     *  replaces its children; pointer dies with the group — re-checked by
     *  unique id like the playback state). */
    ccHObject* m_resultFramesFolder = nullptr;
    unsigned m_resultFramesFolderId = 0;
    /** Last applied result-display state (onPlaybackReset restores it on
     *  top of the all-visible baseline). */
    bool m_resultShowCameras = true;
    int m_resultCamStride = 1;
    int m_resultCamSizePercent = 100;
    /** Rebuilds every per-frame cloud from its LingbotMap_conf scalar
     *  field, keeping only points above \p threshold (official viewer
     *  Visibility-Threshold semantics; can only tighten the display). */
    void refilterResultClouds(double threshold);
    /** Applies show/stride/scale camera display state (no redraw). */
    void applyCameraDisplay(bool showCameras,
                            int camStride,
                            int camSizePercent);
};
