// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QString>
#include <QThread>
#include <QWidget>

#include "VideoPlaybackWidget.h"
#include "YOLOLiveInferWorker.h"
#include "YOLOModelCatalog.h"

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSpinBox;

/** Live camera / video preview with inference-paced YOLO rendering. Detect
 *  models overlay boxes; metric-depth models blend a turbo colorized depth
 *  layer over the frame (the task follows the selected model). The playback
 *  panel (preview, source selection, seek/speed controls and the background
 *  decode pipeline) is inherited from VideoPlaybackWidget; this widget only
 *  adds the model controls, inference thread and overlays. */
class YOLOLiveWidget : public VideoPlaybackWidget {
    Q_OBJECT

public:
    struct Config {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
    };

    explicit YOLOLiveWidget(QWidget* parent = nullptr);
    ~YOLOLiveWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    using VideoPlaybackWidget::setVideoFilePath;
    void setVideoFilePath(const QString& path, bool userChosen);

    bool hasSnapshot() const { return m_hasSnapshot; }
    /** Task of the last completed snapshot ("detect" | "depth"). */
    QString lastTask() const { return m_lastTask; }
    YOLORunResult lastSnapshot() const { return m_lastSnapshot; }
    YOLODepthResult lastDepthSnapshot() const { return m_lastDepth; }

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
    void snapshotUpdated(const YOLORunResult& result);
    void depthSnapshotUpdated(const YOLODepthResult& result);
    void captureToDbRequested(const YOLORunResult& result);
    void depthCaptureToDbRequested(const YOLODepthResult& result);
    void modelSelectionChanged(const QString& modelFilename);
    void deviceSelectionChanged(const QString& deviceId);
    void threadCountChanged(int threads);

public slots:
    void captureSnapshotToDb();

private slots:
    void onInferComplete(YOLOLiveInferWorker::Result result);

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
    void submitInferJob(const QImage& rgb);
    void rebuildOverlayLayer(const QSize& displaySize);
    void drawLiveOverlay(QImage& frame);
    void repaintLivePreview();
    void clearLiveOverlay();
    void shutdownInferThread();

    Config m_config;
    QLabel* m_statusLabel = nullptr;  // cached base accessor
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QDoubleSpinBox* m_confSpin = nullptr;
    QDoubleSpinBox* m_iouSpin = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_syncingModelControls = false;
    bool m_inferBusy = false;
    quint64 m_streamGeneration = 0;

    QThread* m_inferThread = nullptr;
    YOLOLiveInferWorker* m_inferWorker = nullptr;

    QString m_lastTask;  // "detect" | "depth" of the last snapshot
    YOLORunResult m_lastSnapshot;
    YOLODepthResult m_lastDepth;
    bool m_hasSnapshot = false;

    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = -1;
    // Last backend-RESOLVED device reported by the worker ("CUDA0", "cpu",
    // ...); a change logs once so silent CPU fallbacks are visible.
    QString m_lastResolvedDevice;

    // ---- live overlay state (ClockDriven decoupling) ----------------------
    // The display tick paints the newest frame plus a cached overlay layer
    // (detect boxes at preview resolution, or the blended depth layer);
    // inference completions only bump m_overlayGeneration and trigger a
    // repaint. Inference never paces the display, and the full-resolution
    // annotated image is rendered once at capture time from
    // m_lastSourceFrame.
    QImage m_lastDisplayFrame;  // preview-size frame from the display tick
    QImage m_lastSourceFrame;   // full-res frame of the last submitted job
    QVector<YOLODetection> m_overlayDetections;
    QSize m_overlaySourceSize;   // pixel space of m_overlayDetections coords
    QImage m_overlayDepthImage;  // colorized depth at source resolution
    quint64 m_overlayGeneration = 0;          // bumped on new results
    quint64 m_overlayRenderedGeneration = 0;  // layer's results generation
    QSize m_overlayLayerSize;
    QImage m_overlayLayer;  // preview-size transparent overlay cache
};
