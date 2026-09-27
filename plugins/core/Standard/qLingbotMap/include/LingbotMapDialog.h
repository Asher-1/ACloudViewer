// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTextEdit>
#include <QVector>

#include "LingbotMapWorker.h"
#include "ecvTestDataRepository.h"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLineEdit;
class QDoubleSpinBox;
class QPushButton;
class QLabel;
class QSlider;
class QSpinBox;

class ecvModelDownloader;
class ecvMainAppInterface;

/** qLingbotMap main dialog: catalog-backed model selection (with automatic
 *  digest-anchored download), image-sequence input, streaming options, and
 *  the log/progress section shared by all AICore plugins. */
class LingbotMapDialog : public QDialog {
    Q_OBJECT

public:
    explicit LingbotMapDialog(QWidget* parent = nullptr);
    ~LingbotMapDialog() override;

    void setAppInterface(ecvMainAppInterface* app) { m_app = app; }
    void appendLog(const QString& msg);
    /** Run/cancel button + progress visibility while a task is active. */
    void setTaskRunning(bool running);
    /** Worker progress in percent (-1 = indeterminate); hidden when idle. */
    void setProgress(int percent);

    LingbotMapWorker::Settings collectSettings() const;

protected:
    /** Uniform plugin-close semantics: a running reconstruction is
     *  cancelled cooperatively; its finished handler frees the context.
     *  Esc/reject follows via the plugin's rejected() connection. */
    void closeEvent(QCloseEvent* event) override;
    /** Esc intercept: confirm before closing when a task is running. */
    void keyPressEvent(QKeyEvent* event) override;

signals:
    void runRequested(const LingbotMapWorker::Settings& settings);
    void cancelRequested();
    /** Live pause/resume of the streaming reconstruction (official viewer
     *  pause semantics: the render window freezes with it). */
    void pauseRequested(bool paused);
    /** Loop-playback controls changed: "enabled" is the AUTO-PLAY preset
     *  (start the replay right after a result lands); fps/mode apply
     *  immediately to a running replay. */
    void playbackSettingsChanged(bool enabled, int fps, bool currentFrameOnly);
    /** Replay transport (official viewer Playing semantics): explicit
     *  Play/Pause/Stop + timeline seek/step. */
    void replayPlayRequested();
    void replayPauseRequested();
    void replayStopRequested();
    void replaySeekRequested(int frame);
    /** Reset the replay (stop, restore every frame, rewind to frame 0). */
    void playbackReset();
    /** Live camera mode changed (applies to the RUNNING reconstruction:
     *  the worker's camera policy is hot-swappable mid-stream). */
    void liveViewModeChanged(int mode);
    /** Result-display adjustments (official viewer sliders parity), applied
     *  to the LAST result group: visibility threshold re-filter, point
     *  size, camera size/visibility/stride. */
    void resultDisplaySettingsChanged(double threshold,
                                      int pointSize,
                                      int camSizePercent,
                                      bool showCameras,
                                      int camStride);
    /** Result view shortcuts: 0 = move to first camera, 1 = fit all. */
    void resultViewRequested(int which);

public slots:
    /** Mirrors the input image of the frame just reconstructed (official
     *  viewer's frame panel). */
    void updatePreviewImage(const QImage& image, int globalIndex);
    /** Enables the result-display group once a result is materialized;
     *  \p minThreshold is the materialization threshold (the live slider
     *  can only tighten the display beyond it). */
    void setResultDisplayReady(bool ready, double minThreshold);
    /** Mirrors the replay state machine from the plugin: drives the
     *  Play/Pause/Stop/step enables and the timeline slider
     *  (state: 0 = stopped, 1 = playing, 2 = paused; count <= 0 = no
     *  result yet, everything disabled). */
    void setReplayState(int state, int frame, int count);
    /** Publishes the current auto-play/FPS/replay-mode values (the
     *  plugin calls this after wiring, since the constructor's initial
     *  emission had no receiver yet). */
    void publishPlaybackSettings();
    /** Pause button state (driven by the worker's transitions). */
    void setPaused(bool paused);
    /** Automatic download wiring shared with the AICore plugin family. */
    void onModelComboChanged();
    void onBrowseCustomModel();
    void onBrowseImageFolder();
    void onBrowseSkyMaskDir();
    void onBrowseVideoFile();

private:
    enum class PendingAction { None, Run };

    void loadSettings();
    void saveSettings() const;
    void populateCatalogs();
    void populateSkysegCombo(const QString& filename);
    void refreshInputRows();
    QString resolveModelPath() const;
    QString resolveSkysegPath() const;
    bool ensureModelAvailable(PendingAction action);
    void startDownload(const QString& filename);
    LingbotMapWorker::Settings::Mode currentMode() const;
    void onRun();
    void onCancel();

    ecvMainAppInterface* m_app = nullptr;

    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_skysegCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_inputModeCombo = nullptr;
    QWidget* m_folderRow = nullptr;
    QWidget* m_videoRow = nullptr;
    QWidget* m_videoFpsRow = nullptr;
    QLineEdit* m_videoPath = nullptr;
    QSpinBox* m_videoFps = nullptr;
    QLineEdit* m_imageExt = nullptr;
    QLineEdit* m_imageFolder = nullptr;
    QSpinBox* m_maxFrames = nullptr;
    QSpinBox* m_imageSize = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_confThreshold = nullptr;
    QSpinBox* m_downsampleFactor = nullptr;
    /** Live-preview camera behavior (Global / Fit latest / Follow). */
    QComboBox* m_liveViewCombo = nullptr;
    QComboBox* m_skySourceCombo = nullptr;
    QWidget* m_skysegRow = nullptr;
    QLineEdit* m_skyMaskDir = nullptr;
    QWidget* m_skyMaskRow = nullptr;
    QLabel* m_skyHint = nullptr;
    /** Original custom-data reminder text (the label is reused for the
     *  official-outdoor-None warning and restored when that clears). */
    QString m_customSkyHintText;
    /** Dataset → model/pipeline guidance (tips label, refreshed on scene
     *  selection; per the upstream long_real campaign). */
    QLabel* m_datasetHint = nullptr;
    QGroupBox* m_advancedBox = nullptr;
    QWidget* m_advancedContainer = nullptr;
    QSpinBox* m_kvScale = nullptr;
    QSpinBox* m_kvWindow = nullptr;
    QSpinBox* m_keyframeInterval = nullptr;
    QSpinBox* m_frameStride = nullptr;
    QCheckBox* m_rotate90 = nullptr;
    QCheckBox* m_addDbCheck = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_pauseButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPushButton* m_downloadButton = nullptr;
    /** Input-image mirror of the frame just reconstructed. */
    QLabel* m_previewImageLabel = nullptr;
    QLabel* m_previewFrameLabel = nullptr;
    QPushButton* m_playbackResetButton = nullptr;
    /** Replay transport (Play/Pause/Stop + step + timeline). */
    QPushButton* m_replayPlayButton = nullptr;
    QPushButton* m_replayPauseButton = nullptr;
    QPushButton* m_replayStopButton = nullptr;
    QPushButton* m_replayPrevButton = nullptr;
    QPushButton* m_replayNextButton = nullptr;
    QSlider* m_replaySlider = nullptr;
    QLabel* m_replayFrameLabel = nullptr;
    /** Official-viewer display sliders over the LAST result (disabled
     *  until a result is materialized). */
    QGroupBox* m_resultDisplayBox = nullptr;
    QSlider* m_resultThresholdSlider = nullptr;
    QLabel* m_resultThresholdLabel = nullptr;
    QSlider* m_resultPointSizeSlider = nullptr;
    QSlider* m_resultCamSizeSlider = nullptr;
    QCheckBox* m_resultShowCamsCheck = nullptr;
    QSlider* m_resultCamStrideSlider = nullptr;
    QPushButton* m_resultFirstCamButton = nullptr;
    QPushButton* m_resultFitAllButton = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_downloadLabel = nullptr;

    // ---- reconstruction mode (official long-sequence pipeline) ----
    QComboBox* m_modeCombo = nullptr;
    QWidget* m_windowRow = nullptr;
    QSpinBox* m_windowSize = nullptr;
    QSpinBox* m_overlap = nullptr;

    // ---- loop playback (official viewer Playing/FPS semantics) ----
    QCheckBox* m_playbackCheck = nullptr;
    QWidget* m_playbackRow = nullptr;
    QSpinBox* m_playbackFps = nullptr;
    QComboBox* m_playbackMode = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    bool m_downloadInProgress = false;
    bool m_taskRunning = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_modelExplicit = false;
    /** Sky-source auto-switch bookkeeping: outdoor test scenes enable sky
     *  masking on selection, indoor resets it to none, custom folders only
     *  get a hint. A user-explicit combo choice always wins afterwards. */
    bool m_skySourceUserSet = false;
    bool m_skyAutoSwitching = false;
    bool m_customDataSelected = false;

    // ---- test data (shared ecvTestDataRepository, lingbot_map_data) ----
    void populateTestDataScenes();
    void applyTestSceneSelection();
    void requestTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);
    void finishTestDataFlow();
    void refreshSkySourceOptions();
    /** Refreshes the dataset → model/pipeline guidance hint. */
    void updateDatasetHint();
    QComboBox* m_testSceneCombo = nullptr;
    QPushButton* m_testDataBtn = nullptr;
    bool m_testDataInProgress = false;
    ecvTestDataRepository::Dataset m_pendingMaskDownload =
            ecvTestDataRepository::Dataset::LingbotMapCourthouse;
    bool m_maskDownloadPending = false;
    /** Run clicked while the selected test video was still uncached: the
     *  extraction-finished handler starts the run automatically. */
    bool m_pendingRunAfterTestData = false;
};
