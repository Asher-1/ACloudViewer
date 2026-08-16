// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QAtomicInt>
#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QImage>
#include <QString>
#include <QThread>
#include <QVector>
#include <QWidget>

#include "RFDetrModelCatalog.h"
#include "VideoPlaybackWidget.h"

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QProgressBar;
class QSpinBox;
struct aicore_cancel_token;
struct aicore_rfdetr_ctx;

/** Live camera / video preview with throttled RF-DETR inference.
 *  The playback panel (preview, source selection, seek/speed controls and
 *  the background decode pipeline) is inherited from VideoPlaybackWidget;
 *  this widget only adds the model controls, inference thread and overlays. */
class RFDetrLiveWidget : public VideoPlaybackWidget {
    Q_OBJECT

public:
    struct Config {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        float threshold = 0.5f;
        uint32_t topK = 300;
    };

    explicit RFDetrLiveWidget(QWidget* parent = nullptr);
    ~RFDetrLiveWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    using VideoPlaybackWidget::setVideoFilePath;
    void setVideoFilePath(const QString& path, bool userChosen);

    bool hasSnapshot() const { return m_hasSnapshot; }
    RFDetrRunResult lastSnapshot() const { return m_lastSnapshot; }

    void syncModelControlsFrom(const QComboBox* modelCombo,
                               const QComboBox* deviceCombo,
                               const QSpinBox* threadsSpin);
    void rebuildModelCombo(const QStringList& labels,
                           const QStringList& filenames,
                           const QString& currentFilename);
    void rebuildDeviceCombo(const QComboBox* sourceDeviceCombo);
    void setModelPath(const QString& path);
    void setDevice(const QString& device);
    void setThreads(int threads);
    QString modelFilename() const;
    QString deviceId() const;
    int threadCount() const;
    QString resolveModelPath() const;

    void loadSettings();
    void saveSettings() const;

    static bool isAvailable();

signals:
    void logMessage(const QString& msg);
    void snapshotUpdated(const RFDetrRunResult& result);
    void captureToDbRequested(const RFDetrRunResult& result);
    void modelSelectionChanged(const QString& modelFilename);
    void deviceSelectionChanged(const QString& deviceId);
    void threadCountChanged(int threads);

public slots:
    void captureSnapshotToDb();

private slots:
    void onInferComplete(const RFDetrRunResult& result);

protected:
    // ---- video_base hooks -------------------------------------------------
    void onFrameDecoded(cv::Mat& frame, int frameIndex) override;
    void onDisplayFrame(QImage& display, int frameIndex) override;
    void onVideoLooped() override;
    void onStreamReset() override;
    void onStreamResumed() override;
    void onStreamStopping() override;
    bool onPrepareStream() override;
    void onSourceChanged(InputSource source) override;

private:
    void setupUi();
    void updateModelPathFromCombo();
    void submitInferJob(const QImage& inferRgb, float inferScale);
    void shutdownInferThread();
    void drawLiveOverlay(QImage& frame);

    Config m_config;
    ecvClickableImageLabel* m_previewLabel = nullptr;  // cached base accessor
    QLabel* m_statusLabel = nullptr;                   // cached base accessor
    QProgressBar* m_preloadProgress = nullptr;
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QDoubleSpinBox* m_thresholdSpin = nullptr;
    QLabel* m_thresholdLabel = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_syncingModelControls = false;
    bool m_inferBusy = false;
    bool m_preloadingModel = false;
    quint64 m_streamGeneration = 0;

    QThread* m_inferThread = nullptr;
    struct InferJob;
    InferJob* m_inferJob = nullptr;
    QFutureWatcher<RFDetrRunResult>* m_inferWatcher = nullptr;

    RFDetrRunResult m_lastSnapshot;
    bool m_hasSnapshot = false;

    // Cached overlay data — drawn on every frame to prevent flicker.
    QVector<RFDetrDetection> m_overlayDetections;
    QSize m_overlayInferSize;
    qint64 m_overlayFrameNum = 0;  // video frame when overlay was generated
    qint64 m_lastSubmitFrameNum = 0;
    QSize m_lastFrameSize;  // original frame size of the last decode
    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = 0;
};
