// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// qOCR workspace dialog: a pure view in the shared AICore plugin shape.
// Bilingual (Chinese/English) PP-OCR model selection, device/threads runtime
// row, shared-cache downloads through the unified ecvModelDownloader pipeline
// (digest-verified), one-click official test data, and the DB-export switch.
// All log output routes through the application console via
// aicore_inference_log (no private log widget); progress uses the shared
// download/inference progress section; the heavy run itself is executed by
// OCRWorker through the plugin (runRequested / cancelRequested).

#include <QDialog>

#include "ecvTestDataRepository.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class ecvModelDownloader;

struct OCRRunResult;

/** Everything the worker needs for one run; resolved by the dialog view. */
struct OCRRunSettings {
    QString detPath;
    QString recPath;
    QString layoutPath;  // empty = layout stage off
    QString imagePath;
    QString device = QStringLiteral("auto");
    int threads = 0;  // 0 = backend default
    bool addToDb = true;
    /** Overlay the annotated result image on the active render window. */
    bool showInWindow = true;
};

Q_DECLARE_METATYPE(OCRRunSettings)

class OCRDialog final : public QDialog {
    Q_OBJECT
public:
    explicit OCRDialog(QWidget* parent = nullptr);
    ~OCRDialog() override;

    /// Plugin-facing view API (same surface as RMBGDialog).
    void appendLog(const QString& msg);
    void setRunning(bool running);
    void setTaskStage(const QString& stage, int percent);
    void setProgress(int current, int total);
    /// Fill the results table from the worker outcome (GUI thread only).
    void showResult(const OCRRunResult& result);

    OCRRunSettings runSettings() const;

signals:
    void runRequested(const OCRRunSettings& settings);
    void cancelRequested();

private slots:
    void onDownloadMissing();
    void onUseTestData();
    void onBrowseImage();
    void onRun();
    void onCancel();
    void onDownloadFinished(bool ok, const QString& destPath);
    void onTestDataDownloadFinished(bool success,
                                    ecvTestDataRepository::Dataset kind);
    void onTestDataExtractionFinished(bool success,
                                      ecvTestDataRepository::Dataset kind);

private:
    QString cacheDir() const;
    QString detFileName() const;
    QString recFileName() const;
    bool modelPresent(const QString& filename) const;
    void startDownload(const QString& url,
                       const QString& destPath,
                       bool verifyDigest,
                       qint64 minBytes);
    void pumpQueue();
    void updateRunState();
    void fillTestData();
    void loadSettings();
    void saveSettings() const;

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

    // ---- model selection -------------------------------------------------
    QComboBox* m_language = nullptr;  // Chinese / English
    QComboBox* m_quant = nullptr;     // F32 / F16 / Q8_0

    // ---- runtime row -------------------------------------------------------
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threads = nullptr;

    // ---- input -------------------------------------------------------------
    QLineEdit* m_imagePath = nullptr;
    QPushButton* m_browseButton = nullptr;
    QPushButton* m_testDataButton = nullptr;
    bool m_testDataInProgress = false;

    // ---- output ------------------------------------------------------------
    QCheckBox* m_layoutCheck = nullptr;
    QCheckBox* m_addDbCheck = nullptr;
    QCheckBox* m_showInWindowCheck = nullptr;
    QTableWidget* m_results = nullptr;

    // ---- actions / state ---------------------------------------------------
    QPushButton* m_downloadButton = nullptr;
    QPushButton* m_runButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QLabel* m_status = nullptr;
    QLabel* m_taskStatusLabel = nullptr;
    QLabel* m_downloadLabel = nullptr;
    QProgressBar* m_progress = nullptr;
    bool m_runInProgress = false;
    bool m_downloadInProgress = false;
    QString m_activeDownloadKey;
    QString m_lastTaskError;

    ecvModelDownloader* m_downloader = nullptr;
};
