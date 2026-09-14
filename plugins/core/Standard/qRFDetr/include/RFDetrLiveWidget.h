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
#include <QVector>
#include <QWidget>
#include <cstdint>

#include "RFDetrLiveInferWorker.h"
#include "RFDetrModelCatalog.h"
#include "VideoPlaybackWidget.h"

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSpinBox;

/** Live camera / video preview with inference-paced RF-DETR rendering.
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
        /** Class allowlist (empty = detect all classes). Mirrored from the
         *  dialog's class filter; changes reload the live inference ctx. */
        QVector<uint32_t> classFilter;
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
    void setClassFilter(const QVector<uint32_t>& classFilter);
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
    /** Forwarded from the inference worker after a model (re)load — the
     *  model-info JSON envelope (variant / class names) for the dialog's
     *  class-filter list. */
    void modelInfoReady(const QString& info);

public slots:
    void captureSnapshotToDb();

private slots:
    void onInferComplete(RFDetrLiveInferWorker::Result result);

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
    QDoubleSpinBox* m_thresholdSpin = nullptr;
    QLabel* m_thresholdLabel = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_syncingModelControls = false;
    bool m_inferBusy = false;
    quint64 m_streamGeneration = 0;

    QThread* m_inferThread = nullptr;
    RFDetrLiveInferWorker* m_inferWorker = nullptr;

    RFDetrRunResult m_lastSnapshot;
    bool m_hasSnapshot = false;

    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = -1;
    // Last backend-RESOLVED device reported by the worker ("CUDA0", "cpu",
    // ...); a change logs once so silent CPU fallbacks are visible.
    QString m_lastResolvedDevice;

    // ---- live overlay state (ClockDriven decoupling) ----------------------
    // The display tick paints the newest frame plus a cached overlay layer
    // (boxes/masks at preview resolution); inference completions only bump
    // m_overlayGeneration and trigger a repaint. Inference no longer paces
    // the display, and the full-resolution annotated image is rendered once
    // at capture time from m_lastSourceFrame.
    QImage m_lastDisplayFrame;  // preview-size frame from the display tick
    QImage m_lastSourceFrame;   // full-res frame of the last submitted job
    QVector<RFDetrDetection> m_overlayDetections;
    QSize m_overlaySourceSize;  // pixel space of m_overlayDetections coords
    quint64 m_overlayGeneration = 0;          // bumped on new detections
    quint64 m_overlayRenderedGeneration = 0;  // layer's detections generation
    QSize m_overlayLayerSize;
    QImage m_overlayLayer;  // preview-size transparent overlay cache
};
