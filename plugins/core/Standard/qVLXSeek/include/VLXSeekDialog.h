// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// VLX-Seek grounding dialog. Configuration area scrolls; the image input,
// actions and progress stay visible at the minimum window size.

#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>

#include "VLXSeekModelCatalog.h"
#include "VLXSeekWorker.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QScrollArea;
class QSpinBox;

class ecvClickableImageLabel;
class ecvModelDownloader;

class VLXSeekDialog : public QDialog {
    Q_OBJECT

public:
    explicit VLXSeekDialog(QWidget* parent = nullptr);
    ~VLXSeekDialog() override;

    /** Pre-select an input image path (DB selection or test data). */
    void setInputPath(const QString& path);

    /** Whether the last Run asked for the annotated result to be exported
     *  to the DB tree (the dialog's checkbox state at run time). */
    bool exportToDbRequested() const;

signals:
    /** Emitted when Run is accepted with a fully resolved settings set. */
    void runRequested(const VLXSeekWorker::Settings& settings);
    /** Passthrough of the worker's typed result (still-valid busy state). */
    void resultReady(const VLXSeekRunResult& result);

private slots:
    void onUseTestData();
    void onBrowse();
    void onRun();
    void onCancel();
    void onDownloadProgress(qint64 received, qint64 total);
    void onDownloadFinished(bool ok, const QString& path);
    /** Stages the result headline and re-emits it to the plugin (the
     *  annotated image itself is visualized by the DB tree export). */
    void onWorkerResult(const VLXSeekRunResult& result);
    void onWorkerFinished(bool success);

private:
    void populateCatalogCombos();
    /** Fills the demo image plus the showcase prompt (only into an empty
     *  Text field) — a complete runnable demo state for one click. The
     *  image and prompt are selected per the current task mode so each of
     *  the seven tasks gets its optimal upstream demo scene. */
    void applyDemoScene(const QString& demo);
    /** Returns the optimal demo image filename for the current task mode,
     *  derived from the upstream VLX-Seek inference guide examples. */
    QString demoImageForCurrentTask() const;
    /** Resolves the three model paths from the selected precision, starting
     *  the shared downloader for missing files. Returns false (with the
     *  pending action recorded) when a download must finish first. */
    bool ensureModelsAvailable();
    void startDownload(const VLXSeekModelEntry& entry);
    /** Download from an explicit catalog URL with the pinned-digest anchor
     *  (the OCR catalog entry struct carries no digest of its own). */
    void startDownload(const QString& url,
                       const QString& destPath,
                       const QString& pinnedFilename);
    void updateRunState();
    /** Refreshes the input thumbnail from the current path (qDA3 pattern:
     *  plain QImage load, click-to-enlarge via ecvClickableImageLabel). */
    void updateInputPreview();
    /** Shows or hides the target-region-indexes row: only the
     *  brief_region_caption / region_ocr tasks consume proposal indexes. */
    void updateTargetsRowVisibility();
    /** Enables the OCR text-detector proposal source for region_ocr only:
     *  the other tasks ground their own regions or ignore proposals. */
    void updateProposalSourceAvailability();

    QLabel* m_inputLabel = nullptr;
    QLineEdit* m_inputEdit = nullptr;
    QPushButton* m_browseButton = nullptr;
    QPushButton* m_testDataButton = nullptr;
    ecvClickableImageLabel* m_inputPreview = nullptr;
    QCheckBox* m_exportDb = nullptr;
    QWidget* m_targetsRow = nullptr;
    QLineEdit* m_targetsEdit = nullptr;
    QComboBox* m_proposalSourceCombo = nullptr;
    QWidget* m_boxesRow = nullptr;
    QLineEdit* m_boxesEdit = nullptr;
    QPushButton* m_boxesBrowseButton = nullptr;
    QComboBox* m_taskCombo = nullptr;
    QComboBox* m_langCombo = nullptr;
    QLineEdit* m_textEdit = nullptr;
    QComboBox* m_precisionCombo = nullptr;
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
    QString m_downloadTargetFilename;
    enum class PendingAction {
        None,
        Run
    } m_pendingAction = PendingAction::None;
    QVector<VLXSeekModelEntry> m_catalog;
    VLXSeekWorker::Settings m_settings;  // staged paths between prepare/run
    VLXSeekWorker* m_worker = nullptr;
    QString m_lastSummary;  // result headline shown when the worker settles
    // Task of the run in flight (the summary must not follow combo edits
    // made while inference runs).
    QString m_runningTask;
};
