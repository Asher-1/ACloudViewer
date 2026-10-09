// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// qMapAnything dialog — the qYOLO layout contract applied to the
// MapAnything model families:
//   - global Device / Threads row rendered once above the model list;
//   - left-hand model list driving a QStackedWidget (ONE page per published
//     model VARIANT, first page = vggt-omega-512, the accuracy pick);
//   - all pages share the common components: the multi-view image input
//     block, the Run/Cancel row, the Use-test-data controls, the progress
//     bar and the log.

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>
#include <QToolButton>

#include "MapAnythingModelCatalog.h"
#include "MapAnythingWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;
class QShowEvent;
class QSplitter;

/** One model-variant panel: its own model combo (the variant's four
 *  quantizations), custom-path row, preprocessing params and output
 *  toggles. Everything input-related lives in the shared block below the
 *  stack. */
struct MapAnythingPanel {
    QString variant;  // catalog variant key ("" prefix of the GGUF names)

    QWidget* page = nullptr;
    QComboBox* modelCombo = nullptr;
    bool explicitModelChoice = false;
    QLineEdit* customModelPath = nullptr;
    QWidget* customModelRow = nullptr;
    QSpinBox* imageSizeSpin = nullptr;  // 0 = checkpoint default
    QComboBox* resizeModeCombo = nullptr;
    QCheckBox* addPointCloudCheck = nullptr;
    QCheckBox* addDepthCheck = nullptr;
    QCheckBox* addPerViewCheck = nullptr;
    QCheckBox* responsiveCheck = nullptr;
    QDoubleSpinBox* minConfSpin = nullptr;
};

class MapAnythingDialog : public QDialog {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QStringList imagePaths;
        int threads = 0;
        QString device = QStringLiteral("auto");
        int imageSize = 0;
        QString resizeMode = QStringLiteral("balanced");
        bool addPointCloud = true;
        bool addDepthRasters = true;
        bool addPerViewClouds = false;
        bool responsiveStreaming = false;
        float minConf = 0.5f;
    };

    explicit MapAnythingDialog(QWidget* parent = nullptr);
    ~MapAnythingDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    Settings getSettings();
    void appendLog(const QString& msg);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void refreshModelList();
    /** Appends DB image entity names to the shared input list. */
    void applyDbTreeSelection(const QStringList& imageNames);

    static QString modelCacheDir();

signals:
    void runRequested(const MapAnythingDialog::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();
    void importDbImagesRequested();

private slots:
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onAddImages();
    void onRemoveSelectedImages();
    void onClearImages();
    void onInputSelectionChanged();
    void onBrowseCustomModel();
    void onUseTestData();

    void onModelDownloadProgress(qint64 received, qint64 total);
    void onModelDownloadLog(const QString& msg);
    void onModelDownloadFinished(bool ok, const QString& destPath);
    void cancelDownload();

    // Shared multi-view preview transport (Play / Pause / Prev / Reset;
    // the reconstruction takes the preview over while it runs).
    void onPreviewPlay();
    void onPreviewPause();
    void onPreviewPrev();
    void onPreviewReset();
    void onPlayTick();

    void onTestDataLog(const QString& message);
    void onTestDataProgress(int percent, const QString& statusText);
    /** Settles the shared ensureDatasetReady chain (download→extract). */
    void onTestDataReady(bool success, ecvTestDataRepository::Dataset kind);

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void populatePanels();
    void populateModelCombo(MapAnythingPanel& panel,
                            const QString& keepFilename = QString());
    /** Show/hide the custom-GGUF row of one panel (custom combo entry). */
    void applyModelRowVisibility(MapAnythingPanel& panel);
    void updatePreview();
    /** Shows shared-input frame \p index (no-op when out of range); the
     *  single preview funnel used by the transport, the list selection and
     *  the reconstruction progress sync. */
    void showFrame(int index);
    /** True when the shared input list has at least one frame. */
    bool ensureModelAvailable();
    void startDownload(const MapAnythingModelEntry& model);
    /** Resolve the shared input list into absolute file paths (entries may
     *  be db:// image names — resolved by the plugin through
     *  importDbImagesRequested handling). */
    QStringList resolveInputPaths() const;
    /** Fill the shared input list from a MapAnything test-data scene. */
    bool loadTestDataScene(ecvTestDataRepository::Dataset kind);

    MapAnythingPanel* currentPanel() const;
    MapAnythingPanel* panelForVariant(const QString& variant) const;
    MapAnythingPanel* panelForFilename(const QString& filename) const;

    // Global runtime parameters rendered once above the model list.
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // Left-hand model navigation (qYOLO layout): grouped list + stack.
    QListWidget* m_modelList = nullptr;
    QStackedWidget* m_modelStack = nullptr;
    QSplitter* m_bodySplitter = nullptr;
    bool m_splitterRestored = false;
    bool m_splitterSized = false;

    QVector<MapAnythingPanel> m_panels;

    // Shared input block.
    QListWidget* m_imageList = nullptr;
    QPushButton* m_addImagesBtn = nullptr;
    QPushButton* m_removeImageBtn = nullptr;
    QPushButton* m_clearImagesBtn = nullptr;
    QPushButton* m_importDbBtn = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;
    // Preview transport row (qLingbotMap replay semantics: Play / Pause /
    // Prev / Reset plus a frame counter; the worker's progress events take
    // the preview over during a reconstruction run).
    QPushButton* m_previewPlayBtn = nullptr;
    QPushButton* m_previewPauseBtn = nullptr;
    QPushButton* m_previewPrevBtn = nullptr;
    QPushButton* m_previewResetBtn = nullptr;
    QLabel* m_frameLabel = nullptr;
    QTimer* m_playTimer = nullptr;
    int m_playbackIndex = 0;
    bool m_playbackActive = false;
    bool m_reconstructionOwnsPreview = false;

    // Shared run row + test data controls.
    QPushButton* m_runBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;
    QComboBox* m_testSceneCombo = nullptr;
    QPushButton* m_testDataBtn = nullptr;

    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;

    bool m_downloadInProgress = false;
    bool m_taskRunning = false;
    QString m_pendingAfterDownload;  // model path to run once downloaded
    QString m_lastTaskError;

    // Test-data request state (the repository is a process-wide singleton;
    // one download chain at a time, requests while busy are rejected with a
    // log line — same bounded behavior as the qYOLO sample flow).
    bool m_testDataBusy = false;
    ecvTestDataRepository::Dataset m_pendingScene =
            ecvTestDataRepository::Dataset::MapAnythingCourtyard;
};
