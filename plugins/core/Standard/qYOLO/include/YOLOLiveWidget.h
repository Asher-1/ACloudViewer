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
#include <QStringList>
#include <QThread>
#include <QWidget>

#include "VideoPlaybackWidget.h"
#include "YOLOLiveInferWorker.h"
#include "YOLOModelCatalog.h"

class QCheckBox;
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
        // Open-vocabulary families (world/yoloe): class list encoded by the
        // text tower at context load time; empty for closed-set tasks.
        QStringList classes;
        QString textModelPath;
        // Multi-object tracking (detect/segment/pose/obb models).
        // trackerType empty disables tracking; defaults mirror the official
        // ultralytics/cfg/trackers YAMLs.
        QString trackerType;  // "" |
                              // bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack
        QString gmcMethod = QStringLiteral("sparseOptFlow");
        // Official with_reid (model="auto" detector-feature path): engages
        // the tracker's ReID cosine term; end2end heads degrade silently.
        bool withReid = false;
        float trackHighThresh = 0.25f;
        float trackLowThresh = 0.1f;
        float newTrackThresh = 0.25f;
        int trackBuffer = 30;
        float matchThresh = 0.8f;
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
    /** Populate the model combo with ALL catalog models (every task), for
     *  the Live tab that must run any model type. */
    void populateAllModels(const QString& keepFilename = QString());
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

    /** Tear the async infer thread down (releasing the resident model).
     *  Called when the owning dialog closes for good; the thread is
     *  rebuilt lazily on the next live start. Idempotent. */
    void releaseGpuResources();

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

    // Official trackzone interaction: Ctrl+drag on the preview draws the
    // tracking region (source-pixel rect, white border like the solutions
    // trackzone annotator); a Ctrl+click without drag clears it. Consumes
    // the events so the label's click-to-enlarge stays off during selection.
    bool onPreviewMousePress(QMouseEvent* event) override;
    bool onPreviewMouseMove(QMouseEvent* event) override;
    bool onPreviewMouseRelease(QMouseEvent* event) override;

private:
    void setupUi();
    void updateModelPathFromCombo();
    /** Show/hide the detection-threshold controls (Conf/IoU/Top-K) based on
     *  the selected model's task: depth models have no thresholds, detect /
     *  segment models do. Called on every model change. */
    void updateThresholdVisibility();
    /** Show/hide the tracking rows: hidden for depth (no detections) and
     *  while tracking is unchecked; the parameter area additionally hides
     *  with the row. Called on every model change and Track toggle. */
    void updateTrackVisibility();
    /** Current tracker-type / GMC combo selections (data ids). */
    QString trackerTypeId() const;
    QString gmcMethodId() const;
    /** Enable the GMC combo only for tracker types that consume camera
     *  motion (botsort / deepocsort / tracktrack); disabled elsewhere so
     *  an irrelevant control is never editable. */
    void updateGmcEnabled();
    /** Reset the exposed tracking controls (the five threshold spins and
     *  the GMC combo) to the official per-type defaults of
     *  ultralytics/cfg/trackers/<type>.yaml — the upstream
     *  tracker=<yaml> selection semantics. Called on tracker type
     *  changes and once for the initial selection. */
    void applyOfficialTrackDefaults(const QString& type);
    void submitInferJob(const QImage& rgb);
    void rebuildOverlayLayer(const QSize& displaySize);
    void drawLiveOverlay(QImage& frame);
    void repaintLivePreview();
    void clearLiveOverlay();
    void shutdownInferThread();
    /** Rebuild the infer thread after releaseGpuResources(); no-op while
     *  it is already running. */
    void ensureInferThread();

    Config m_config;
    QLabel* m_statusLabel = nullptr;  // cached base accessor
    QComboBox* m_modelCombo = nullptr;
    QComboBox* m_deviceCombo = nullptr;
    QSpinBox* m_threadsSpin = nullptr;
    QDoubleSpinBox* m_confSpin = nullptr;
    QDoubleSpinBox* m_iouSpin = nullptr;
    QSpinBox* m_topKSpin = nullptr;
    // Threshold row labels (hidden for depth models, which have no
    // detection thresholds).
    QList<QWidget*> m_thresholdWidgets;
    // Tracking row + parameter area (hidden for depth models / unchecked).
    QCheckBox* m_trackCheck = nullptr;
    QCheckBox* m_reidCheck = nullptr;
    QCheckBox* m_trailsCheck = nullptr;
    QComboBox* m_trackerCombo = nullptr;
    QComboBox* m_gmcCombo = nullptr;
    QDoubleSpinBox* m_trackHighSpin = nullptr;
    QDoubleSpinBox* m_trackLowSpin = nullptr;
    QDoubleSpinBox* m_newTrackSpin = nullptr;
    QSpinBox* m_trackBufferSpin = nullptr;
    QDoubleSpinBox* m_matchSpin = nullptr;
    QList<QWidget*> m_trackWidgets;       // row-4 controls + labels
    QList<QWidget*> m_trackParamWidgets;  // parameter-area controls + labels

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
    // Last tracking warning surfaced to the log; a change logs once (the
    // worker repeats the warning every frame while parked on a rejected
    // config, the log must not).
    QString m_lastTrackWarning;

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
    QVector<YOLOSegMask> m_overlayMasks;  // instance masks (segment only)
    QVector<YOLOObbBox> m_overlayObbs;    // oriented boxes (obb only)
    QVector<YOLOKeypointSet> m_overlayKeypointSets;  // pose skeletons
    QVector<int> m_overlayTrackIds;  // index-aligned with the overlay rows
    // Official solutions-style track trails: per-id center path in source
    // pixels (capped deque); ids absent a frame keep their history so a
    // re-found target resumes its line. Cleared when tracking turns off or
    // the source restarts.
    QHash<int, QVector<QPointF>> m_trails;
    // Trails overlay toggle (default off — the official model.track preview
    // draws boxes+ids only); a pinned target always shows its trail.
    bool m_showTrails = false;
    // "Track this one" pin: Ctrl+click on a tracked row highlights only that
    // identity (its box, banner and trail) and demotes the rest to thin gray
    // boxes; Ctrl+click on empty space or the same id releases the pin.
    int m_pinnedTrackId = -1;
    // Official trackzone region (source-pixel rect; invalid = whole frame).
    // Ctrl+drag on the preview redraws it; Ctrl+click without drag clears.
    QRectF m_trackZone;
    QRectF m_trackZoneDraft;  // live rubber band while dragging (source px)
    QPointF m_zoneDragStartLabel;  // press point in label coordinates
    bool m_zoneDragging = false;
    QSize m_overlaySourceSize;   // pixel space of m_overlayDetections coords
    QImage m_overlayDepthImage;  // colorized depth at source resolution
    quint64 m_overlayGeneration = 0;          // bumped on new results
    quint64 m_overlayRenderedGeneration = 0;  // layer's results generation
    QSize m_overlayLayerSize;
    QImage m_overlayLayer;  // preview-size transparent overlay cache
};
