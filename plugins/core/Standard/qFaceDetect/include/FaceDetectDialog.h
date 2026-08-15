// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QHash>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QToolButton>
#include <QWidget>

#include "FaceDetectTestData.h"
#include "FaceDetectWorker.h"
#include "aicore/facedetect_capi.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

class FaceLiveDetectWidget;
class FaceRegistryWidget;
class FaceDetectTestDataWorker;

class FaceDetectDialog : public QDialog {
    Q_OBJECT

public:
    enum class Mode { Detect, Analyze, Verify, DenseLandmarks };

    struct Settings {
        QString modelPath;
        QString landmarkModelPath;
        QString inputPath;
        QString secondInputPath;
        int threads = 0;
        QString device = "auto";
        Mode mode = Mode::Detect;
        float verifyThreshold = 0.65f;
        float minDetectionScore = 0.5f;
        bool antiSpoof = false;
        bool addAnnotatedImageToDb = true;
    };

    struct DbImageEntry {
        QString name;
        QImage preview;
    };

    explicit FaceDetectDialog(QWidget* parent = nullptr);
    ~FaceDetectDialog() override;

    Settings getSettings() const;
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setRunning(bool running);
    void setDbImages(const QList<DbImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);
    void refreshModelList();

    static QString modelCacheDir();
    static QString registryPath();

signals:
    void runRequested(const FaceDetectDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void liveCaptureReady(const FaceDetectRunResult& result);
    void authVisualizationReady(const QImage& annotated,
                                const QString& summary);

private slots:
    void onBrowseImage();
    void onBrowseSecondImage();
    void onBrowseCustomModel();
    void onBrowseCustomLandmarkModel();
    void onModelComboChanged(int index);
    void onModeChanged(int index);
    void onLandmarkModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);
    void onLiveStart();
    void onLiveStop();
    void onLiveRestart();
    void onLiveCapture(const FaceDetectRunResult& result);
    void onAuthResultImageReady(const QImage& annotated,
                                const QString& summary);
    void onLiveStreamModeChanged(int streamMode, bool showUserPrompt = true);
    void validateLiveRecognizeModeFromSettings();
    void syncLiveConfig();
    void tryAutoDiscoverRegistryDb();
    void applyMatchThresholdToAllTabs(double value);
    void setupMatchThresholdLinks();
    void applyMinDetectionScoreToAllTabs(double value);
    void setupMinScoreLinks();
    void cacheTabViewportHeights();

protected:
    void closeEvent(QCloseEvent* event) override;
    // Re-measure tab viewport heights when the dialog moves to a screen
    // with a different DPI (Windows per-monitor scaling) — the cached
    // minimumSizeHint values and hardcoded clamps must be recomputed.
    void changeEvent(QEvent* event) override;

private:
    void syncRegistryConfig();
    void syncRegistryModelControlsFromBatch();
    void ensureFriendsTestData(bool fillRegistry,
                               bool fillLiveVideo,
                               bool fillBatchImage);
    void applyFriendsTestBundle(const FaceDetectFriendsBundle& bundle,
                                bool fillRegistry,
                                bool fillLiveVideo,
                                bool fillBatchImage);
    /** Fill Live/Batch sample paths only — never opens registry or registers
     * faces. */
    void applyFriendsTestDataPaths(const FaceDetectFriendsBundle& bundle,
                                   bool fillLiveVideo,
                                   bool fillBatchImage);
    void startTestDataPostProcess(const FaceDetectFriendsBundle& bundle,
                                  bool fillRegistry,
                                  bool fillLiveVideo,
                                  bool fillBatchImage,
                                  bool extractZipFirst = false,
                                  const QString& zipPath = QString(),
                                  bool clearExistingEntries = true);
    void setTestDataBusy(bool busy);
    void updateTestDataProgress(int current, int total, const QString& label);
    bool tryResolveFriendsTestBundle(FaceDetectFriendsBundle* out);
    void startFriendsTestDataDownload(bool fillRegistry,
                                      bool fillLiveVideo,
                                      bool fillBatchImage);
    /** Append / Overwrite / Cancel when registry already has entries. */
    enum class TestDataRegistryMode { Cancel, Append, Overwrite };
    TestDataRegistryMode confirmTestDataRegistryMode() const;
    void setupBatchTab(QWidget* batchTab);
    void populateModelCombo(const QString& keepFilename = QString());
    void populateLandmarkModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(QComboBox* combo, const QString& filename);
    void updateModeUi();
    void syncLandmarkPathFromCombo();
    void selectDefaultLandmarkModel();
    void ensureLandmarkModelPathFilled();
    /** First existing landmark GGUF under model cache (catalog + known
     * filenames). */
    QString defaultLandmarkModelPathOnDisk() const;
    QString resolveModelPath() const;
    QString resolveLandmarkModelPath() const;
    /** Resolve landmark filename from combo / default — never returns empty
     *  when catalog has an entry (used to trigger auto-download). */
    QString resolveLandmarkModelFilename() const;
    QString defaultLandmarkModelFilename() const;
    bool ensureModelAvailable();
    /** Check only the face detector model is downloaded and ready.
     *  Skips the Dense-Landmarks landmark-model check — used by the Live tab
     *  which only supports Detect / Recognize, never DenseLandmarks. */
    bool ensureDetectorAvailable();
    void startDownload(const aicore_facedetect_model_entry* model);
    void cancelDownload();
    void updateImagePreview();
    void updateSecondImagePreview();
    void updateActiveTabViewportHeight();
    void loadBatchSettings();
    void saveBatchSettings() const;

    QTabWidget* m_tabWidget = nullptr;
    QWidget* m_batchTab = nullptr;
    FaceLiveDetectWidget* m_liveWidget = nullptr;
    FaceRegistryWidget* m_registryWidget = nullptr;
    QPushButton* m_liveStartBtn = nullptr;
    QPushButton* m_liveStopBtn = nullptr;
    QPushButton* m_liveRestartBtn = nullptr;

    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_landmarkModelCombo = nullptr;
    QWidget* m_landmarkModelRow = nullptr;
    QLineEdit* m_customLandmarkModelPath = nullptr;
    QWidget* m_customLandmarkModelRow = nullptr;
    QComboBox* m_modeCombo = nullptr;
    QLabel* m_variantHintLabel = nullptr;
    QLineEdit* m_imagePath = nullptr;
    QLineEdit* m_secondImagePath = nullptr;
    QWidget* m_secondImageRow = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;
    QDoubleSpinBox* m_verifyThreshold = nullptr;
    QDoubleSpinBox* m_minDetectionScore = nullptr;
    QDoubleSpinBox* m_verifyMinDetectionScore = nullptr;
    QLabel* m_minScoreLabel = nullptr;
    QWidget* m_batchMinScoreRow = nullptr;
    QCheckBox* m_antiSpoofCheck = nullptr;
    QWidget* m_verifyOptionsRow = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    ecvClickableImageLabel* m_previewLabelB = nullptr;
    QWidget* m_previewRow = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QCheckBox* m_addAnnotatedCheck = nullptr;
    QCheckBox* m_linkMatchThresholdsCheck = nullptr;
    QPushButton* m_applyMatchThresholdBtn = nullptr;
    bool m_syncingMatchThresholds = false;
    bool m_syncingMinScores = false;
    bool m_batchImagePathUserChosen = false;
    QToolButton* m_dbToggleBtn = nullptr;
    QWidget* m_dbContentWidget = nullptr;
    QListWidget* m_dbImageList = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    ecvModelDownloader* m_testDataDownloader = nullptr;
    FaceDetectTestDataWorker* m_testDataWorker = nullptr;
    bool m_downloadInProgress = false;
    bool m_testDataDownloadInProgress = false;
    bool m_testDataProcessing = false;
    bool m_testFillRegistry = false;
    bool m_testFillLiveVideo = false;
    bool m_testFillBatchImage = false;
    bool m_testPostFillRegistry = false;
    bool m_testPostFillLiveVideo = false;
    bool m_testPostFillBatchImage = false;
    bool m_testClearExistingEntries = true;
    bool m_autoRunAfterDownload = false;
    int m_activeTabHeight = -1;
    QHash<const QWidget*, int> m_tabViewportHeights;

    static constexpr int kTestDataOverallMax = 1000;
    static constexpr int kTestDataDownloadShare =
            300;  // 30% of bar when post-process follows
    int m_testDataPostProgressBase = 0;
};
