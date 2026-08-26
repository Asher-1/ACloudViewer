// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Video segmentation & tracking tab for qSAM3.
//
// Qt re-implementation of the upstream sam3-ggml ImGui demo
// (examples/main_video.cpp). Layout:
//   Row 1: Mode (Text | Box | Points) + text prompt + Open video +
//          Play / Pause / Step / Reset
//   Row 2: Model combo (all catalog families) + Load + Device + Backend
//   Canvas: video frame with per-instance mask tints; click a mask to
//           refine, drag a box / click points to add an instance
//   Timeline: scrubber bar + per-instance presence bands (click to seek)
//   Bottom: Show masks, playback speed, Export frame masks, instance list,
//           status line

#pragma once

#include <QTimer>
#include <QWidget>

#include <ecvTestDataRepository.h>

#include "VideoCanvas.h"
#include "VideoTimeline.h"
#include "VideoWorker.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSlider;
class QTextBrowser;
class ecvModelDownloader;

class VideoFrameReader;
class ecvMainAppInterface;

namespace cv {
class Mat;
}

class VideoTab : public QWidget {
    Q_OBJECT
public:
    explicit VideoTab(QWidget* parent = nullptr);
    ~VideoTab() override;

    void setDevice(const QString& device);  // propagate shared device combo
    /** Release the tracker/model while preserving the opened video. */
    void releaseModel();
    /** Pass the app interface for DB-tree export. */
    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }

signals:
    /** Keep the dialog's shared backend indicator in sync with video mode. */
    void backendChanged(const QString& backendName);

private slots:
    void onOpenVideo();
    void onLoadModel();
    void onPlayPause();
    void onStep();
    void onReset();
    void onModelReady(const QString& backend, bool visualOnly);
    void onTrackerReset(const QString& textPrompt, bool ok);
    void onFrameReady(const cv::Mat& rgbFrame, int frameIndex);
    void onFrameResult(const SAM3WorkerResult& result, int frameIndex);
    void onInstanceAdded(int instanceId);
    void onInstanceRefined(int instanceId, bool ok);
    void onBusyChanged(bool busy);
    void onLog(const QString& msg);
    void onSeek(int frame);
    void onModeChanged();
    void onCanvasBox();
    void onCanvasInstanceClicked(int id, const QPointF& imagePos);
    void onCanvasPosPoint(const QPointF& p);
    void onCanvasNegPoint(const QPointF& p);
    void onExportMasks();
    /** Hot-swap: re-load the model on the newly selected device. */
    void onDeviceChanged();
    /** One-click test: download (cached) the SAM3 test dataset and open the
     *  video selected in the test-video picker. */
    void onUseTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);

protected:
    /** Space = play/pause, Right = step, Left = previous frame (upstream
     *  main_video.cpp keyboard shortcuts). */
    void keyPressEvent(QKeyEvent* e) override;

private:
    void setupUi();
    void populateModelCombo();
    /** Rebuild the model combo keeping the current selection (cache status
     *  suffixes may have changed after a download). */
    void refreshModelCombo();
    QString modelPath() const;
    /** Download the GGUF currently selected in the model combo into the
     *  shared AICore model cache (sam3_models). Naming mirrors the
     *  qDA3/qYOLO/qDeepLSD startDownload convention. When \p thenRun is
     *  true the pending operation is re-executed once the download finishes. */
    void startDownload(bool thenRun);
    /** Refresh the Download button state (cached / missing). */
    void updateDownloadButton();
    void openVideoFile(const QString& path);
    /** Ensure a model matching the current combo selection is loaded.
     *  Starts a lazy load when needed (the pending action is re-run once
     *  the load finishes, mirroring upstream main_video.cpp where the
     *  model loads automatically on first use — there is no manual Load
     *  step). Returns true when ready to run. */
    bool ensureModelReady();
    /** Action to re-run once a lazy model load completes. */
    enum class PendingAction { None, Play, Step, Seek, Reset, Annotate };
    QString desiredTextPrompt() const;
    /** Make the tracker prompt/mode match the UI. Returns true when no
     *  asynchronous reset is needed. */
    bool syncTrackerPrompt(bool retrack);
    void requestTrackerReset(const QString& textPrompt, bool retrack);
    void clearTrackingVisualization();
    void trackNextFrame();
    void schedulePlayback();
    void addInstanceFromPrompts();
    void refineInstance(int id,
                        const QVector<QPointF>& pos,
                        const QVector<QPointF>& neg);
    void updateCanvasInstances();
    void updateTimeline(int frameIndex, const SAM3WorkerResult& result);
    void appendLog(const QString& msg);
    void setStatus(const QString& msg);
    QColor instanceColor(int id) const;
    void resetPrompts();
    /** Export the current frame result to the DB tree as a ccImage. */
    void exportCurrentFrameToDb();
    /** (Re)fill the test-video picker from the extracted SAM3 dataset. */
    void populateTestVideoCombo();
    /** Open the video currently selected in the test-video picker. */
    bool loadRequestedTestVideo();
    void setTestDataControlsEnabled(bool enabled);

    VideoWorker* m_worker = nullptr;
    VideoFrameReader* m_reader = nullptr;
    QTimer m_playTimer;

    // Controls
    QRadioButton* m_modeText = nullptr;
    QRadioButton* m_modeBox = nullptr;
    QRadioButton* m_modePoints = nullptr;
    QLineEdit* m_textPrompt = nullptr;
    QPushButton* m_openBtn = nullptr;
    QPushButton* m_playBtn = nullptr;
    QPushButton* m_stepBtn = nullptr;
    QPushButton* m_resetBtn = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QPushButton* m_downloadBtn = nullptr;  // downloads the selected GGUF
    QPushButton* m_loadBtn = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QLabel* m_backendLabel = nullptr;
    QComboBox* m_testVideoCombo = nullptr;  // test-video picker (SAM3 dataset)
    QPushButton* m_testDataBtn = nullptr;
    VideoCanvas* m_canvas = nullptr;
    VideoTimeline* m_timeline = nullptr;
    QCheckBox* m_showMasks = nullptr;
    QCheckBox* m_exportToDbCheckBox = nullptr;
    QDoubleSpinBox* m_scoreSpin = nullptr;
    QSlider* m_speedSlider = nullptr;
    QLabel* m_speedLabel = nullptr;
    QPushButton* m_exportBtn = nullptr;
    QPushButton* m_exportFrameToDbBtn = nullptr;
    QTextBrowser* m_instanceLabel = nullptr;
    QLabel* m_statusLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;  // test-data download/extract status
    QProgressBar* m_progress = nullptr;

    // Model downloader (shared ecvModelDownloader, qDA3-style)
    ecvModelDownloader* m_modelDownloader = nullptr;
    bool m_downloadInProgress = false;
    /** Re-run the pending operation once the download completes. */
    bool m_downloadThenRun = false;
    QString m_downloadTargetFilename;
    /** Prevent re-prompting after the user declined the download dialog. */
    bool m_downloadPrompted = false;

    // State
    QString m_videoPath;
    int m_totalFrames = 0;
    int m_currentFrame = 0;
    int m_processedMax = -1;
    double m_fps = 0.0;
    bool m_playing = false;
    bool m_busy = false;
    bool m_trackerActive = false;
    bool m_visualOnly = false;
    QString m_activeTextPrompt;
    QString m_loadingTextPrompt;
    QString m_loadedModelPath;
    QString m_loadingModelPath;
    QString m_loadedDevice;
    QString m_loadingDevice;
    bool m_trackerPromptDirty = false;
    bool m_promptResetInFlight = false;
    bool m_reloadWhenIdle = false;
    bool m_testDataDownloadInProgress = false;
    /** Operation queued while a lazy model load runs; re-executed from
     *  onModelReady once the load finishes. */
    PendingAction m_pendingAction = PendingAction::None;
    /** A trackNextFrame() call arrived while the worker was busy (model
     *  loading / ResetTracker); re-run it once busy clears. Without this,
     *  the frame request is silently dropped because modelReady / reset
     *  completion signals arrive before busyChanged(false). */
    bool m_pendingRetrack = false;

    SAM3WorkerResult m_lastResult;
    QVector<VideoTimelineEntry> m_timelineEntries;
    QVector<int> m_timelineInstanceIds;

    // DB-tree export state
    ecvMainAppInterface* m_app = nullptr;
    QImage m_currentFrameImage;
    /** Decoded frame waiting for its matching tracker result. Consumer-driven
     *  playback guarantees at most one pending inference frame. */
    QImage m_pendingFrameImage;
    int m_pendingFrameIndex = -1;
};
