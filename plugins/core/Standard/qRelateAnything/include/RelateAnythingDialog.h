// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QDialog>
#include <QTemporaryFile>
#include <QThread>
#include <memory>

#include "RelateAnythingWorker.h"
class QComboBox;
class QFormLayout;
class QLineEdit;
class QCheckBox;
class QSpinBox;
class QDoubleSpinBox;
class QPushButton;
class QListWidget;
class QTableWidget;
class QLabel;
class QPlainTextEdit;
class QTabWidget;
class QProgressBar;
class ecvClickableImageLabel;
class ecvModelDownloader;
class RelateAnythingVideoWidget;
class RelateAnythingDialog : public QDialog {
    Q_OBJECT
public:
    explicit RelateAnythingDialog(QWidget* parent = nullptr);
    ~RelateAnythingDialog() override;
    void setDbImages(const QList<QPair<QString, QImage>>& images);
    bool saveResults(const QString& path);
    void shutdown();
signals:
    void addToDbRequested(const RelateAnythingResult&);

private:
    void addFiles();
    void addInput(const QString& path,
                  const QImage& image = {},
                  bool checked = true);
    void useTestData();
    void run();
    void runNext();
    bool ensureModel(bool runAfter);
    void submit(QImage image, const QString& source, bool frame);
    void completed(const RelateAnythingResult& result);
    void showResult();
    void cancel();
    void exportResults();
    RelateAnythingSettings settings() const;
    QString modelPath() const;
    QString relationAssetPath(int index) const;
    QString detectorPath() const;
    QString detectorTextModelPath() const;
    void log(const QString&);
    void setBusy(bool);
    void saveSettings();
    void resetResults();
    void recordResult(const QJsonObject& result);
    void closeEvent(QCloseEvent*) override;
    void showEvent(QShowEvent*) override;
    QComboBox *m_model = nullptr, *m_device = nullptr, *m_detector = nullptr,
              *m_objectSource = nullptr, *m_arithmetic = nullptr;
    QSpinBox *m_threads = nullptr, *m_maxRelations = nullptr,
             *m_maxBoxes = nullptr;
    QDoubleSpinBox *m_threshold = nullptr, *m_relThreshold = nullptr,
                   *m_spatialThreshold = nullptr, *m_detIou = nullptr,
                   *m_maxBoxFraction = nullptr;
    QSpinBox* m_maxSpatial = nullptr;
    QComboBox *m_profile = nullptr, *m_predicatePreset = nullptr;
    bool m_applyingPreset = false;
    // Video tracking / smoothing (upstream render_video.py flags). Thresholds
    // shared with the image path (relation floors, edge caps) reuse the
    // widgets above; the rest live in these tables, keyed by objectName.
    QCheckBox *m_stabilize = nullptr, *m_belief = nullptr,
              *m_boxScoreWeight = nullptr;
    double defaultConfidence(const QString& detector, int profile) const;
    std::vector<std::pair<float RelateTemporal::Params::*, QDoubleSpinBox*>>
            m_trackD;
    std::vector<std::pair<int RelateTemporal::Params::*, QSpinBox*>> m_trackI;
    void buildTrackingUi(QFormLayout* form);
    RelateTemporal::Params trackingParams() const;
    void resetTrackingDefaults();
    void applyProfile(int profile);
    void markCustomProfile();
    QCheckBox* m_decompose = nullptr;
    QLineEdit *m_predicates = nullptr, *m_filter = nullptr;
    QLineEdit *m_regions = nullptr, *m_masks = nullptr, *m_vocabulary = nullptr,
              *m_vocabularyNames = nullptr, *m_predicateTypes = nullptr;
    QListWidget* m_inputs = nullptr;
    QTableWidget* m_table = nullptr;
    ecvClickableImageLabel* m_preview = nullptr;
    QLabel *m_summary = nullptr, *m_timing = nullptr;

    QTabWidget* m_tabs = nullptr;
    QProgressBar* m_progress = nullptr;
    QPushButton *m_run = nullptr, *m_cancelButton = nullptr,
                *m_download = nullptr, *m_sample = nullptr;
    RelateAnythingVideoWidget* m_video = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    QThread m_thread;
    RelateAnythingWorker* m_worker = nullptr;
    QList<QPair<QString, QImage>> m_db;
    QList<QPair<QString, QImage>> m_queue;
    QImage m_heldFrame;
    QString m_heldFrameSource;
    std::unique_ptr<QTemporaryFile> m_resultSpool;
    qint64 m_resultCount = 0;
    int m_batchTotal = 0, m_batchCompleted = 0;
    RelateAnythingResult m_last;
    RelateAnythingSettings m_jobSettings;
    quint64 m_generation = 1;
    bool m_busy = false, m_framePending = false, m_runAfterDownload = false,
         m_batch = false, m_closed = false;
    bool m_waitingForSample = false;
};
