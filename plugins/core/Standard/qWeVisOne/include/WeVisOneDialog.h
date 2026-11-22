// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// WeVisOne-VL perception dialog. Configuration area scrolls; the image
// input, actions and progress stay visible at the minimum window size.

#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

#include "WeVisOneModelCatalog.h"
#include "WeVisOneWorker.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QScrollArea;
class QSpinBox;

class ecvClickableImageLabel;
class ecvModelDownloader;

class WeVisOneDialog : public QDialog {
    Q_OBJECT

public:
    explicit WeVisOneDialog(QWidget* parent = nullptr);
    ~WeVisOneDialog() override;

    /** Pre-select an input image path (DB selection or test data). */
    void setInputPath(const QString& path);

    /** Whether the last Run asked for the annotated result to be exported
     *  to the DB tree (the dialog's checkbox state at run time). */
    bool exportToDbRequested() const;

signals:
    /** Emitted when Run is accepted with a fully resolved settings set. */
    void runRequested(const WeVisOneWorker::Settings& settings);
    /** Passthrough of the worker's typed result (still-valid busy state). */
    void resultReady(const WeVisOneRunResult& result);

private slots:
    void onUseTestData();
    void onBrowse();
    void onRun();
    void onCancel();
    void onDownloadProgress(qint64 received, qint64 total);
    void onDownloadFinished(bool ok, const QString& path);
    /** Stages the result headline and re-emits it to the plugin (the
     *  annotated image itself is visualized by the DB tree export). */
    void onWorkerResult(const WeVisOneRunResult& result);
    void onWorkerFinished(bool success);

private:
    void populateCatalogCombos();
    /** Refills the VLM precision combo for the selected checkpoint family
     *  (2b/4b), keeping the current quantization when the family has it. */
    void repopulatePrecisionCombo();
    /** Fills the demo image plus the task-specific query (and the
     *  visual_prompt reference boxes) — a complete runnable demo state for
     *  one click. \p query / \p refBoxes come from the chosen manifest
     *  sample (or the offline fallback). */
    void applyDemoScene(const QString& demo,
                        const QString& query,
                        const QString& refBoxes);
    /** One-click demo fill: prefer the dataset's own manifest.json sample
     *  (WeVisOne-specific case-study image first), else the built-in offline
     *  fallback. Returns true only when a demo was actually applied, so the
     *  warm path can fall through to shared preparation when the verified
     *  zip has not been extracted yet. */
    bool fillDemoForCurrentTask();
    /** Manifest-driven fill; returns false when no manifest/sample/member is
     *  available so the caller can fall back. */
    bool applyManifestDemo();
    /** Offline fallback fill (synthetic caches without a manifest). */
    bool applyFallbackDemo();
    /** Returns the optimal demo image filename for the current task mode,
     *  the first sample of that task in the upstream test_data manifest. */
    QString demoImageForCurrentTask() const;
    /** Offline fallback query text for the current task (ocr clears). */
    QString fallbackQueryForCurrentTask() const;
    /** Resolves the three model paths from the selected family/precision,
     *  starting the shared downloader for missing files. Returns false
     *  (with the pending action recorded) when a download must finish
     *  first. */
    bool ensureModelsAvailable();
    void startDownload(const WeVisOneModelEntry& entry);
    void updateRunState();
    /** Refreshes the input thumbnail from the current path (qDA3 pattern:
     *  plain QImage load, click-to-enlarge via ecvClickableImageLabel). */
    void updateInputPreview();
    /** Shows or hides the reference-boxes row: only visual_prompt consumes
     *  the <ref>/<box> reference quadruples. */
    void updateRefBoxesVisibility();

    QLabel* m_inputLabel = nullptr;
    QLineEdit* m_inputEdit = nullptr;
    QPushButton* m_browseButton = nullptr;
    QPushButton* m_testDataButton = nullptr;
    ecvClickableImageLabel* m_inputPreview = nullptr;
    QCheckBox* m_exportDb = nullptr;
    QComboBox* m_taskCombo = nullptr;
    QPlainTextEdit* m_queryEdit = nullptr;
    QWidget* m_refBoxesRow = nullptr;
    QLineEdit* m_refBoxesEdit = nullptr;
    QComboBox* m_sizeCombo = nullptr;
    QComboBox* m_precisionCombo = nullptr;
    QComboBox* m_decoderPrecisionCombo = nullptr;
    QCheckBox* m_decodeMasks = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QSpinBox* m_tokensSpin = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QProgressBar* m_progress = nullptr;
    QLabel* m_statusLabel = nullptr;
    QScrollArea* m_configScroll = nullptr;

    ecvModelDownloader* m_downloader = nullptr;
    bool m_downloadInProgress = false;
    bool m_testDataBusy = false;
    QString m_downloadTargetFilename;
    enum class PendingAction {
        None,
        Run
    } m_pendingAction = PendingAction::None;
    QVector<WeVisOneModelEntry> m_catalog;
    WeVisOneWorker::Settings m_settings;  // staged paths between prepare/run
    WeVisOneWorker* m_worker = nullptr;
    QString m_lastSummary;  // result headline shown when the worker settles
    // Task of the run in flight (the summary must not follow combo edits
    // made while inference runs).
    QString m_runningTask;
};