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
#include <QStackedWidget>
#include <QTextEdit>
#include <QToolButton>

#include "GKDModelCatalog.h"
#include "GKDWorker.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

class ecvMainAppInterface;
class QSplitter;

/** One mode panel of the qGKD dialog (qYOLO-style task pages): the
 *  mode-specific prompt rows, the image input with preview, the per-keypoint
 *  score threshold and the action row. A dialog owns one panel per mode
 *  (text / visual / multimodal / multi) shown in a left-hand mode list;
 *  the GKD model and Device / Threads are global controls rendered once
 *  above the list (the same GGUF serves every mode). */
struct GKDModePanel {
    QString mode;  // GKDHelpers::promptModes() id

    QWidget* page = nullptr;
    // Mode-specific prompt rows; hidden when the mode does not use them.
    QWidget* kpsRow = nullptr;
    QLineEdit* kpsTexts = nullptr;
    QWidget* supportImageRow = nullptr;
    QLineEdit* supportImagePath = nullptr;
    QWidget* supportKpsRow = nullptr;
    QLineEdit* supportKps = nullptr;
    QWidget* roiRow = nullptr;
    QLineEdit* roi = nullptr;
    QWidget* classesRow = nullptr;
    QLineEdit* objectClasses = nullptr;
    QWidget* detectorRow = nullptr;
    QComboBox* yoloModelCombo = nullptr;
    /** Text-encoder tower (CLIP/MobileCLIP) that encodes the
     *  open-vocabulary class names; text-conditioned WORLD detectors
     *  reject the run without it. */
    QWidget* textModelRow = nullptr;
    QComboBox* yoloTextModelCombo = nullptr;
    QDoubleSpinBox* yoloConf = nullptr;
    // Shared per-panel run state.
    QLineEdit* imagePath = nullptr;
    ecvClickableImageLabel* previewLabel = nullptr;
    QDoubleSpinBox* minScore = nullptr;
    /** Skeleton of the last-filled preset ("1-2 1-3 ...", empty for
     *  manual runs); passed through to the worker for bone rendering. */
    QString skeleton;
    QPushButton* runBtn = nullptr;
    QPushButton* cancelBtn = nullptr;
    QPushButton* testDataBtn = nullptr;
    QToolButton* dbToggleBtn = nullptr;
    QWidget* dbContentWidget = nullptr;
    QListWidget* dbImageList = nullptr;
};

/** qGKD dialog: GKDT general keypoint detection on a still or DB image.
 *  The four prompt modes (text / 1-shot visual / multimodal / multi-object
 *  YOLO-World composition) each get their own page; the per-page "Use test
 *  data" button one-click fills every field of the mode with the official
 *  upstream demo configuration (scenario-matched sample images). */
class GKDDialog : public QDialog {
    Q_OBJECT

public:
    explicit GKDDialog(QWidget* parent = nullptr);
    ~GKDDialog() override;

    void setAppInterface(ecvMainAppInterface* app);
    void appendLog(const QString& msg);
    /** Snapshot of the latest successful run (COCO JSON export source;
     *  called by qGKD when the worker reports a result). */
    void setLastRun(const GKDRunResult& result);
    void setProgress(int current, int total);
    void setTaskStage(const QString& stage, int percent = -1);
    void setRunning(bool running);
    void setDbImages(const QList<GKDImageEntry>& images);
    void applyDbTreeSelection(const QStringList& imageNames);
    void refreshModelList();
    GKDWorker::Settings workerSettings() const;

    static QString modelCacheDir();

signals:
    void runRequested(const GKDWorker::Settings& settings);
    void cancelRequested();
    void refreshDbImagesRequested();

private slots:
    void onBrowseImage();
    void onBrowseSupportImage();
    void onBrowseCustomModel();
    void onBrowseSaveDir();
    void onModelComboChanged(int index);
    void onRun();
    void onCancel();
    void onDbListActivated(QListWidgetItem* item);
    void requestTestData();
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);

protected:
    void closeEvent(QCloseEvent* event) override;
    /** Esc intercept: confirm before closing when a task is running. */
    void keyPressEvent(QKeyEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    enum class PendingAction { None, Run };

    void setupUi();
    void loadSettings();
    void saveSettings() const;
    void populateModelCombo(const QString& keepFilename = QString());
    void populateYoloModelCombo(const QString& keepFilename = QString());
    bool selectModelByFilename(const QString& filename);
    QString resolveModelPath() const;
    /** Resolved cache path of a yolo-catalog combo's selection (detector
     *  or text-encoder tower; both live in the yolo model cache dir). */
    QString resolveYoloModelPath(const QComboBox* combo) const;
    bool ensureModelAvailable(PendingAction action);
    bool ensureYoloModelAvailable();
    void startDownload(const GKDModelEntry& model,
                       const QString& destDir = QString());
    void cancelDownload();
    void updateImagePreview();

    /** Content-driven (font / DPI aware) ideal width of the mode list; see
     *  qYOLO's taskListIdealWidth() for the rationale. */
    int modeListIdealWidth() const;
    /** First-show allocation of the splitter's left/right panes (skipped
     *  when loadSettings() restored a user-saved splitter state). */
    void applyModeListDefaultWidth();

    /** The panel of the currently active mode-list entry. */
    GKDModePanel* currentModePanel() const;
    /** Find the panel whose mode id is `mode` (may be nullptr). */
    GKDModePanel* panelForMode(const QString& mode) const;

    // ---- Sample data (shared ecvTestDataRepository; one dataset per
    // mode: the official GKDT demos, or the shared objects-detection cache
    // for the multi-object composition) ----
    /** Resume/queue driver for a sample-data request: fill the panel when
     *  the file is already cached, wait while the shared repository serves
     *  another download, or start the mode's dataset download/extract
     *  chain. Requests made while the repository is busy are QUEUED here
     *  instead of dropped. */
    void advancePendingTestData();
    /** Load the pending + follow-up queued requests (after a chain's
     *  extraction made the archive available) and clear both slots. */
    void servePendingTestData();
    /** Load one mode's official-demo preset into its panel. Returns false
     *  when the sample file is not cached yet. */
    bool loadTestDataFor(const QString& mode);
    void setTestDataControlsEnabled(bool enabled);

    // Global model + runtime controls (every mode shares the GKD GGUF).
    QComboBox* m_modelCombo = nullptr;
    QLineEdit* m_customModelPath = nullptr;
    QWidget* m_customModelRow = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // Left-hand mode navigation driving the stack (qYOLO layout): a
    // splitter so users can drag the list wider/narrower (state persisted
    // in QSettings).
    QListWidget* m_modeList = nullptr;
    QStackedWidget* m_modeStack = nullptr;
    QSplitter* m_bodySplitter = nullptr;
    bool m_splitterRestored = false;
    bool m_splitterSized = false;

    // One panel per mode (stack order; GKDHelpers::promptModes()).
    QVector<GKDModePanel> m_panels;

    // Active-panel aliases so the shared browse / preview handlers act on
    // the visible mode's widgets (same pattern as qYOLO).
    QLineEdit* m_imagePath = nullptr;
    QLineEdit* m_supportImagePath = nullptr;
    ecvClickableImageLabel* m_previewLabel = nullptr;

    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;

    QCheckBox* m_addDbCheck = nullptr;
    QCheckBox* m_savePngCheck = nullptr;
    /** Per-keypoint "prompt score" labels (default OFF — multi-object
     *  label backgrounds hide the objects; opt-in per run). */
    QCheckBox* m_pointLabelsCheck = nullptr;
    QPushButton* m_exportCocoBtn = nullptr;
    /** Most recent successful run (COCO JSON export source). */
    GKDRunResult m_lastRun;
    void onExportCocoJson();
    QLineEdit* m_savePngDir = nullptr;
    QTextEdit* m_log = nullptr;
    QLabel* m_taskStatusLabel = nullptr;

    bool m_testDataDownloadInProgress = false;
    // Two queued sample-data request slots: pending is served by the
    // running chain's completion, followup (a request made while a chain
    // was already serving pending) right after it. Later clicks overwrite
    // the followup slot (latest wins; bounded queue).
    QString m_pendingTestDataMode;
    QString m_followupTestDataMode;
    // Next scenario preset to serve per mode ("Try sample data" cycles
    // through GKDHelpers::modePresets(mode) on repeated clicks).
    QHash<QString, int> m_presetRotation;

    ecvModelDownloader* m_downloader = nullptr;
    ecvMainAppInterface* m_app = nullptr;
    bool m_downloadInProgress = false;
    bool m_modelExplicit = false;
    PendingAction m_pendingActionAfterDownload = PendingAction::None;
    bool m_taskRunning = false;
    bool m_firstShow = true;
};
