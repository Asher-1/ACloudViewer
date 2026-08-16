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
#include <QWidget>

#include "RMBGModelCatalog.h"
#include "VideoPlaybackWidget.h"

class QComboBox;
class QLabel;
class QSpinBox;
struct aicore_cancel_token;
struct aicore_rmbg_ctx;

/** Live camera / video preview with throttled RMBG-2.0 background removal.
 *  The playback panel (preview, source selection, seek/speed controls and
 *  the background decode pipeline) is inherited from VideoPlaybackWidget;
 *  this widget only adds the model controls, inference thread and the
 *  transparent preview rendering. */
class RMBGLiveWidget : public VideoPlaybackWidget {
    Q_OBJECT

public:
    struct Config {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
    };

    explicit RMBGLiveWidget(QWidget* parent = nullptr);
    ~RMBGLiveWidget() override;

    void setConfig(const Config& config);
    Config config() const { return m_config; }

    using VideoPlaybackWidget::setVideoFilePath;
    void setVideoFilePath(const QString& path, bool userChosen);

    bool hasSnapshot() const { return m_hasSnapshot; }
    RMBGRunResult lastSnapshot() const { return m_lastSnapshot; }

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
    void snapshotUpdated(const RMBGRunResult& result);
    void captureToDbRequested(const RMBGRunResult& result);
    void modelSelectionChanged(const QString& modelFilename);
    void deviceSelectionChanged(const QString& deviceId);
    void threadCountChanged(int threads);

public slots:
    void captureSnapshotToDb();

private slots:
    void onInferComplete(const RMBGRunResult& result);

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
    void submitInferJob(const QImage& inferRgb);
    void shutdownInferThread();

    Config m_config;
    ecvClickableImageLabel* m_previewLabel = nullptr;  // cached base accessor
    QLabel* m_statusLabel = nullptr;                   // cached base accessor
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;

    bool m_videoPathUserChosen = false;
    bool m_syncingModelControls = false;
    bool m_inferBusy = false;
    quint64 m_streamGeneration = 0;

    struct InferJob;
    QFutureWatcher<RMBGRunResult>* m_inferWatcher = nullptr;

    RMBGRunResult m_lastSnapshot;
    bool m_hasSnapshot = false;

    // Cached result — composited over the display frame to prevent flicker.
    QImage m_overlayRgba;
    qint64 m_overlayFrameNum = 0;  // video frame when the result was generated
    qint64 m_lastSubmitFrameNum = 0;
    QElapsedTimer m_inferSubmitTime;
    qint64 m_lastInferLatencyMs = -1;
};
