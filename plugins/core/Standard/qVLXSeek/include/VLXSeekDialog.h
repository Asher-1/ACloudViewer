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
    void onResultPreview(const VLXSeekRunResult& result);
    void onWorkerFinished(bool success);

private:
    void populateCatalogCombos();
    /** Resolves the three model paths from the selected precision, starting
     *  the shared downloader for missing files. Returns false (with the
     *  pending action recorded) when a download must finish first. */
    bool ensureModelsAvailable();
    void startDownload(const VLXSeekModelEntry& entry);
    void updateRunState();
    /** Refreshes the input thumbnail from the current path (qDA3 pattern:
     *  plain QImage load, click-to-enlarge via ecvClickableImageLabel). */
    void updateInputPreview();
    /** Shows or hides the target-region-indexes row: only the
     *  brief_region_caption / region_ocr tasks consume proposal indexes. */
    void updateTargetsRowVisibility();

    QLabel* m_inputLabel = nullptr;
    QLineEdit* m_inputEdit = nullptr;
    QPushButton* m_browseButton = nullptr;
    QPushButton* m_testDataButton = nullptr;
    ecvClickableImageLabel* m_inputPreview = nullptr;
    ecvClickableImageLabel* m_resultPreview = nullptr;
    QCheckBox* m_exportDb = nullptr;
    QWidget* m_targetsRow = nullptr;
    QLineEdit* m_targetsEdit = nullptr;
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
};
