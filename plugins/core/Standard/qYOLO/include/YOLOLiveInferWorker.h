// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QtGlobal>
#include <cstdint>
#include <memory>

#include "YOLOModelCatalog.h"

namespace qyolo::track {
struct Tracker;
struct TrackConfig;
}  // namespace qyolo::track

struct aicore_yolo_ctx;

/** Serialized live YOLO inference with a context reused across frames. The
 *  model decides the path per load (detect or metric depth); Result::task
 *  tells the widget which payload is valid. */
class YOLOLiveInferWorker : public QObject {
    Q_OBJECT

public:
    struct Job {
        QImage rgb;
        QString modelPath;
        QString device;
        int threads = 0;
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
        // Open-vocabulary families (world/yoloe): class list encoded by the
        // text tower once per context load; a change reloads the context.
        QStringList classes;
        QString textModelPath;
        quint64 generation = 0;
        // Multi-object tracking (qyolo::track port of the ultralytics
        // trackers; detect/segment/pose/obb models). trackerType empty
        // disables tracking. Defaults mirror the official
        // ultralytics/cfg/trackers YAMLs.
        QString trackerType;  // "" |
                              // bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack
        QString gmcMethod = QStringLiteral("sparseOptFlow");
        // Official with_reid (model="auto" detector-feature path): the
        // tracker consumes the per-detection feature rows exported by the
        // context (aicore_yolo_features_view); a stream without features
        // degrades to motion-only association.
        bool withReid = false;
        float trackHighThresh = 0.25f;
        float trackLowThresh = 0.1f;
        float newTrackThresh = 0.25f;
        int trackBuffer = 30;
        float matchThresh = 0.8f;
        // Official trackzone semantics: only detections whose center lies
        // inside this source-pixel rect join the tracker (and receive ids);
        // an invalid rect means the whole frame. Detection rows outside are
        // still reported without ids so the user sees what is excluded.
        QRectF trackZone;
    };

    struct Result {
        YOLORunResult detect;   // valid when task == "detect"/"segment"/
                                // "pose"/"obb"
        YOLODepthResult depth;  // valid when task == "depth"
        QString task;
        QString error;
        /** Non-fatal tracking diagnostics (e.g. the tracker was rejected
         *  because this build lacks OpenCV): inference results are still
         *  delivered, but without track ids. Empty = no warning. */
        QString warning;
        bool ok = false;
        quint64 generation = 0;
        /** Stable track ids, index-aligned with detect.detections /
         *  detect.keypointSets / detect.obbBoxes; 0 = untracked. Empty
         *  when tracking is disabled or the task is not trackable. */
        QVector<int> trackIds;
    };

    explicit YOLOLiveInferWorker(QObject* parent = nullptr);
    ~YOLOLiveInferWorker() override;

public slots:
    void runJob(YOLOLiveInferWorker::Job job);
    void releaseModel();

signals:
    void inferComplete(YOLOLiveInferWorker::Result result);

private:
    void runJobImpl(YOLOLiveInferWorker::Job job);
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job, QString* error);
    /** Feeds the frame's detections to the tracking state machine (creating
     *  / resetting it on generation or config changes) and fills
     *  result.trackIds. task: "detect"|"segment"|"pose"|"obb". */
    void applyTracking(const YOLOLiveInferWorker::Job& job,
                       const QString& task,
                       YOLOLiveInferWorker::Result& result);

    aicore_yolo_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
    QStringList m_loadedClasses;
    QString m_loadedTextModelPath;
    /* Task of the loaded context ("detect" | "segment" | "depth" | "pose"
     * | "obb"), cached at load time so each Result knows which payload it
     * carries. */
    QString m_loadedTask;
    /* Backend-RESOLVED device of the loaded context ("CUDA0", "cpu", ...);
     * differs from m_loadedDevice when the GPU lease failed. */
    QString m_resolvedDevice;

    /* Tracking state, worker-thread only (runJob is serialized here). One
     * tracker instance per stream: recreated when the config fingerprint or
     * the stream generation changes (source switch / seek / stop), mirroring
     * the plugin-worker contract for stale-state resets. */
    std::unique_ptr<qyolo::track::Tracker> m_tracker;
    std::unique_ptr<qyolo::track::TrackConfig> m_trackCfg;
    bool m_trackCfgValid = false;
    quint64 m_trackGeneration = 0;
    // Object-feature export state last requested from the context (the
    // toggle rebuilds the graph plan, so it only flips on change).
    bool m_objFeatEnabled = false;
    /** Last tracker-creation failure reason surfaced to the UI (cleared
     *  once a tracker is created again). */
    QString m_trackWarning;
#endif
};

Q_DECLARE_METATYPE(YOLOLiveInferWorker::Job)
Q_DECLARE_METATYPE(YOLOLiveInferWorker::Result)
