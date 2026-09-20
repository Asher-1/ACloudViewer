// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOLiveInferWorker.h"

#include <QtCompat.h>

#include <QByteArray>
#include <QElapsedTimer>
#include <QFileInfo>
#include <cmath>
#include <cstring>
#include <new>
#include <utility>

#ifdef AICore_ENABLED
#include "aicore/reid_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "tracking/tracker.hpp"
#endif

namespace {

#ifdef AICore_ENABLED
// Device-task serialization moved to ecvAICoreRuntimeHelpers.h

aicore_image_view imageView(const QImage& image) {
    return ecvAICoreRuntime::makeImageView(image);
}

// True when the tracker already holds this exact configuration. Any change
// (tracker type, GMC method or the exposed thresholds) rebuilds the state
// machine, which is equivalent to a reset.
bool sameTrackConfig(const qyolo::track::TrackConfig& cfg,
                     const YOLOLiveInferWorker::Job& job) {
    return cfg.tracker_type == job.trackerType.toStdString() &&
           cfg.gmc_method == job.gmcMethod.toStdString() &&
           cfg.with_reid == job.withReid &&
           cfg.track_high_thresh == job.trackHighThresh &&
           cfg.track_low_thresh == job.trackLowThresh &&
           cfg.new_track_thresh == job.newTrackThresh &&
           cfg.track_buffer == job.trackBuffer &&
           cfg.match_thresh == job.matchThresh;
}
#endif

}  // namespace

YOLOLiveInferWorker::YOLOLiveInferWorker(QObject* parent) : QObject(parent) {
    qRegisterMetaType<YOLOLiveInferWorker::Job>("YOLOLiveInferWorker::Job");
    qRegisterMetaType<YOLOLiveInferWorker::Result>(
            "YOLOLiveInferWorker::Result");
}

YOLOLiveInferWorker::~YOLOLiveInferWorker() { releaseModel(); }

void YOLOLiveInferWorker::releaseModel() {
#ifdef AICore_ENABLED
    if (m_ctx) {
        aicore_yolo_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_reidCtx) {
        aicore_reid_free(m_reidCtx);
        m_reidCtx = nullptr;
    }
    m_loadedReidPath.clear();
    m_loadedModelPath.clear();
    m_loadedDevice.clear();
    m_loadedThreads = 0;
    m_loadedClasses.clear();
    m_loadedTextModelPath.clear();
    m_loadedTask.clear();
    m_resolvedDevice.clear();
    // A model reload also drops the tracking state: the new stream starts
    // from a clean track table (the generation binding below covers mere
    // restarts of the same model).
    m_tracker.reset();
    m_trackCfg.reset();
    m_trackCfgValid = false;
    m_trackGeneration = 0;
    m_objFeatEnabled = false;
#endif
}

#ifdef AICore_ENABLED
bool YOLOLiveInferWorker::ensureModel(const Job& job, QString* error) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath)) {
        if (error) *error = tr("Model file does not exist.");
        return false;
    }
    if (m_ctx && aicore_yolo_is_ready(m_ctx) &&
        m_loadedModelPath == job.modelPath && m_loadedDevice == job.device &&
        m_loadedThreads == job.threads && m_loadedClasses == job.classes &&
        m_loadedTextModelPath == job.textModelPath) {
        return true;
    }

    releaseModel();
    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (!opts) {
        if (error) *error = tr("Failed to allocate model options.");
        return false;
    }
    aicore_yolo_options_set_device(opts, job.device.toUtf8().constData());
    aicore_yolo_options_set_threads(opts, job.threads);
    // Pin the initial graph plan to the ACTUAL letterbox canvas of the
    // first frame (Ultralytics LetterBox auto=True, stride=32 — the same
    // formula as yolo_image.cpp letterbox_image): otherwise create_session
    // builds a square-imgsz plan that the first inference immediately
    // rebuilds, one extra full graph construction per load (~8 s on cuda
    // for a 162-op model).
    if (!job.rgb.isNull()) {
        constexpr int kLetterboxBase = 640;  // YOLO letterbox base size
        const float r =
                std::min(static_cast<float>(kLetterboxBase) / job.rgb.width(),
                         static_cast<float>(kLetterboxBase) / job.rgb.height());
        const int new_w = static_cast<int>(std::nearbyint(job.rgb.width() * r));
        const int new_h =
                static_cast<int>(std::nearbyint(job.rgb.height() * r));
        const int dw = (kLetterboxBase - new_w) % 32;
        const int dh = (kLetterboxBase - new_h) % 32;
        aicore_yolo_options_set_input_size(opts, new_w + dw, new_h + dh);
    }
    // Open-vocabulary families (world/yoloe): the class list rides into
    // load and the text tower encodes it once per context — same contract
    // as the still-image worker. The list is pre-trimmed by the dialog.
    if (!job.classes.isEmpty()) {
        std::vector<const char*> classPtrs;
        classPtrs.reserve(static_cast<size_t>(job.classes.size()));
        QList<QByteArray> utf8;
        utf8.reserve(job.classes.size());
        for (const QString& c : job.classes) {
            utf8.append(c.toUtf8());
        }
        for (const QByteArray& c : utf8) {
            classPtrs.push_back(c.constData());
        }
        aicore_yolo_options_set_classes(opts, classPtrs.data(),
                                        static_cast<int32_t>(classPtrs.size()));
        if (!job.textModelPath.isEmpty()) {
            aicore_yolo_options_set_text_model(
                    opts, job.textModelPath.toUtf8().constData());
        }
    }
    m_ctx = aicore_yolo_load_opts(job.modelPath.toUtf8().constData(), opts);
    aicore_yolo_options_free(opts);
    if (!m_ctx || !aicore_yolo_is_ready(m_ctx)) {
        const char* message = m_ctx ? aicore_yolo_last_error(m_ctx) : nullptr;
        if (error) {
            *error = message ? QString::fromUtf8(message)
                             : tr("Failed to create model context.");
        }
        releaseModel();
        return false;
    }
    m_loadedModelPath = job.modelPath;
    m_loadedDevice = job.device;
    m_loadedThreads = job.threads;
    m_loadedClasses = job.classes;
    m_loadedTextModelPath = job.textModelPath;
    m_loadedTask = QString::fromUtf8(aicore_yolo_context_task(m_ctx));
    /* The backend-resolved device ("CUDA0", "cpu", ...), captured at load
     * time so every Result reports what actually ran — a requested GPU that
     * silently fell back to CPU shows up here. */
    const char* resolved = aicore_yolo_context_device(m_ctx);
    m_resolvedDevice = (resolved && resolved[0]) ? QString::fromUtf8(resolved)
                                                 : job.device;
    return true;
}
#endif

void YOLOLiveInferWorker::runJob(YOLOLiveInferWorker::Job job) {
    // The queued slot boundary: an uncaught allocation failure inside the
    // worker would unwind through the worker thread's event loop and
    // terminate the whole process (SIGABRT). Surface it as a per-frame error
    // instead so video inference keeps running.
    const quint64 generation = job.generation;
    const int frameIndex = job.frameIndex;
    try {
        runJobImpl(std::move(job));
    } catch (const std::bad_alloc&) {
        Result result;
        result.generation = generation;
        result.frameIndex = frameIndex;
        result.error = tr("Out of memory while processing the frame.");
        emit inferComplete(result);
    }
}

void YOLOLiveInferWorker::runJobImpl(YOLOLiveInferWorker::Job job) {
    Result result;
    result.generation = job.generation;
    result.frameIndex = job.frameIndex;

#ifndef AICore_ENABLED
    result.error = tr("AICore is not enabled.");
    emit inferComplete(result);
    return;
#else
    auto taskGuard = ecvAICoreRuntime::makeDeviceTaskLock(job.device);
    if (!taskGuard.isLocked()) {
        result.error = tr("Failed to acquire the inference device.");
        emit inferComplete(result);
        return;
    }
    if (!ensureModel(job, &result.error)) {
        emit inferComplete(result);
        return;
    }
    result.task = m_loadedTask;

    // TrackTrack loose-NMS recovery (upstream want_recovered semantics):
    // enabled only for the tracktrack tracker on the box tasks; zero cost
    // for every other combination. Runtime setter — no model reload.
    const bool wantRecovery = job.trackerType == QStringLiteral("tracktrack") &&
                              (m_loadedTask == QStringLiteral("detect") ||
                               m_loadedTask == QStringLiteral("obb"));
    aicore_yolo_set_track_recovery(m_ctx, wantRecovery ? 1 : 0);

    const aicore_image_view image = imageView(job.rgb);

    if (result.task == QStringLiteral("depth")) {
        // Metric depth: typed float map + statistics envelope. The colorized
        // image is NOT built here — the live preview only needs a downscaled
        // layer (drawn by the widget), so the full-resolution colorize pass
        // runs once at capture time.
        QElapsedTimer timer;
        timer.start();
        int32_t dw = 0, dh = 0;
        float* depth = aicore_yolo_depth_image(m_ctx, &image, &dw, &dh);
        result.depth.runtimeMs = static_cast<double>(timer.elapsed());
        if (!depth || dw <= 0 || dh <= 0) {
            if (depth) aicore_yolo_free_buffer(depth);
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO depth inference failed.");
            emit inferComplete(result);
            return;
        }
        result.depth.depthMap = qtCompatQVectorFromRange<float>(
                depth, depth + static_cast<size_t>(dw) * dh);
        aicore_yolo_free_buffer(depth);
        result.depth.width = dw;
        result.depth.height = dh;
        aicore_yolo_depth_stats stats{};
        if (aicore_yolo_last_depth_stats(m_ctx, &stats) == 0) {
            result.depth.stats.width = stats.depth_width;
            result.depth.stats.height = stats.depth_height;
            result.depth.stats.minDepth = stats.min_depth;
            result.depth.stats.maxDepth = stats.max_depth;
            result.depth.stats.meanDepth = stats.mean_depth;
            result.depth.stats.p95Depth = stats.p95_depth;
            result.depth.stats.validPixels =
                    static_cast<long long>(stats.valid_pixels);
        }
        result.depth.modelPath = job.modelPath;
        result.depth.resolvedDevice = m_resolvedDevice;
        result.depth.imageName = QStringLiteral("live");
        result.ok = true;
        emit inferComplete(result);
        return;
    }

    if (result.task == QStringLiteral("segment")) {
        // Instance segmentation: typed detections + per-instance masks.
        QElapsedTimer timer;
        timer.start();
        aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                          job.topK);
        aicore_yolo_segment_result* seg = aicore_yolo_seg_image(m_ctx, &image);
        result.detect.runtimeMs = static_cast<double>(timer.elapsed());
        if (!seg) {
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO segmentation failed.");
            emit inferComplete(result);
            return;
        }

        const int n = aicore_yolo_seg_det_count(seg);
        result.detect.detections.reserve(n > 0 ? n : 0);
        result.detect.masks.reserve(n > 0 ? n : 0);
        for (int i = 0; i < n; ++i) {
            const aicore_yolo_detection det = aicore_yolo_seg_det_at(seg, i);
            YOLODetection d;
            d.classId = det.class_id;
            d.x1 = det.x1;
            d.y1 = det.y1;
            d.x2 = det.x2;
            d.y2 = det.y2;
            d.score = det.score;
            result.detect.detections.append(d);

            const aicore_yolo_plane_view view = aicore_yolo_seg_mask_at(seg, i);
            if (view.data != nullptr && view.width > 0 && view.height > 0) {
                YOLOSegMask mask;
                mask.w = view.width;
                mask.h = view.height;
                mask.bits = QByteArray(
                        static_cast<const char*>(view.data),
                        static_cast<int>(view.row_stride_bytes) * view.height);
                result.detect.masks.append(mask);
            }
        }
        for (int i = 0; i < result.detect.detections.size(); ++i) {
            const char* name = aicore_yolo_seg_det_class_name(seg, i);
            result.detect.detections[i].className =
                    (name != nullptr && name[0] != '\0')
                            ? QString::fromUtf8(name)
                            : QStringLiteral("class %1")
                                      .arg(result.detect.detections[i].classId);
        }
        result.detect.totalDetected = n;
        result.detect.task = QStringLiteral("segment");
        aicore_yolo_seg_result_free(seg);

        result.detect.modelPath = job.modelPath;
        result.detect.resolvedDevice = m_resolvedDevice;
        result.detect.imageName = QStringLiteral("live");
        result.ok = true;
        applyTracking(job, result.task, result);
        emit inferComplete(result);
        return;
    }

    if (result.task == QStringLiteral("pose")) {
        // Pose estimation: typed detections + COCO-17 keypoints.
        QElapsedTimer timer;
        timer.start();
        aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                          job.topK);
        aicore_yolo_pose_result* pose = aicore_yolo_pose_image(m_ctx, &image);
        result.detect.runtimeMs = static_cast<double>(timer.elapsed());
        if (!pose) {
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO pose inference failed.");
            emit inferComplete(result);
            return;
        }

        const int n = aicore_yolo_pose_det_count(pose);
        const int kpts = aicore_yolo_pose_kpt_count(pose);
        result.detect.keypointSets.reserve(n > 0 ? n : 0);
        // The det-box copies keep the generic overlay / tracking pipeline
        // working for pose runs (index-aligned with keypointSets).
        result.detect.detections.reserve(n > 0 ? n : 0);
        for (int i = 0; i < n; ++i) {
            const aicore_yolo_detection det = aicore_yolo_pose_det_at(pose, i);
            YOLOKeypointSet ks;
            ks.det.classId = static_cast<uint32_t>(det.class_id);
            ks.det.x1 = det.x1;
            ks.det.y1 = det.y1;
            ks.det.x2 = det.x2;
            ks.det.y2 = det.y2;
            ks.det.score = det.score;
            for (int k = 0; k < kpts; ++k) {
                const aicore_yolo_keypoint kp =
                        aicore_yolo_pose_kpt_at(pose, i, k);
                ks.kpts.append({kp.x, kp.y, kp.visibility});
            }
            const char* name = aicore_yolo_pose_det_class_name(pose, i);
            ks.det.className =
                    (name != nullptr && name[0] != '\0')
                            ? QString::fromUtf8(name)
                            : QStringLiteral("class %1").arg(det.class_id);
            result.detect.keypointSets.append(ks);
            YOLODetection row;
            row.classId = ks.det.classId;
            row.className = ks.det.className;
            row.score = ks.det.score;
            row.x1 = ks.det.x1;
            row.y1 = ks.det.y1;
            row.x2 = ks.det.x2;
            row.y2 = ks.det.y2;
            result.detect.detections.append(row);
        }
        result.detect.kptCount = kpts;
        result.detect.totalDetected = n;
        result.detect.task = QStringLiteral("pose");
        aicore_yolo_pose_result_free(pose);

        result.detect.modelPath = job.modelPath;
        result.detect.resolvedDevice = m_resolvedDevice;
        result.detect.imageName = QStringLiteral("live");
        result.ok = true;
        applyTracking(job, result.task, result);
        emit inferComplete(result);
        return;
    }

    if (result.task == QStringLiteral("obb")) {
        // Oriented boxes: typed rotated detections.
        QElapsedTimer timer;
        timer.start();
        aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                          job.topK);
        aicore_yolo_obb_result* obb = aicore_yolo_obb_image(m_ctx, &image);
        result.detect.runtimeMs = static_cast<double>(timer.elapsed());
        if (!obb) {
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO OBB inference failed.");
            emit inferComplete(result);
            return;
        }

        const int n = aicore_yolo_obb_count(obb);
        result.detect.obbBoxes.reserve(n > 0 ? n : 0);
        for (int i = 0; i < n; ++i) {
            const aicore_yolo_obb_box b = aicore_yolo_obb_at(obb, i);
            YOLOObbBox out;
            out.cx = b.cx;
            out.cy = b.cy;
            out.w = b.w;
            out.h = b.h;
            out.angle = b.angle;
            out.score = b.score;
            out.classId = static_cast<uint32_t>(b.class_id);
            const char* name = aicore_yolo_obb_class_name(obb, i);
            out.className =
                    (name != nullptr && name[0] != '\0')
                            ? QString::fromUtf8(name)
                            : QStringLiteral("class %1").arg(b.class_id);
            result.detect.obbBoxes.append(out);
        }
        result.detect.totalDetected = n;
        result.detect.task = QStringLiteral("obb");
        aicore_yolo_obb_result_free(obb);

        result.detect.modelPath = job.modelPath;
        result.detect.resolvedDevice = m_resolvedDevice;
        result.detect.imageName = QStringLiteral("live");
        result.ok = true;
        applyTracking(job, result.task, result);
        emit inferComplete(result);
        return;
    }
    QElapsedTimer timer;
    timer.start();
    aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                      job.topK);
    const int detectRc = aicore_yolo_detect_image(m_ctx, &image);
    result.detect.runtimeMs = static_cast<double>(timer.elapsed());
    if (detectRc != 0) {
        const char* message = aicore_yolo_last_error(m_ctx);
        result.error = message ? QString::fromUtf8(message)
                               : tr("YOLO inference failed.");
        emit inferComplete(result);
        return;
    }

    const int count = aicore_yolo_detection_count(m_ctx);
    result.detect.detections.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        const aicore_yolo_detection det = aicore_yolo_detection_at(m_ctx, i);
        YOLODetection out;
        out.classId = static_cast<uint32_t>(det.class_id);
        // Backend class table (open-vocabulary class-list override or the
        // GGUF metadata); fall back to the deterministic label only when the
        // model declares no name for this class.
        const char* name = aicore_yolo_detection_class_name(m_ctx, i);
        out.className = (name != nullptr && name[0] != '\0')
                                ? QString::fromUtf8(name)
                                : QStringLiteral("class %1").arg(det.class_id);
        out.x1 = det.x1;
        out.y1 = det.y1;
        out.x2 = det.x2;
        out.y2 = det.y2;
        out.score = det.score;
        result.detect.detections.append(out);
    }
    result.detect.totalDetected = count;

    // No annotated rendering here: the live preview only needs a downscaled
    // overlay (drawn by the widget on the display image), so a per-frame
    // full-resolution drawDetections pass would be pure waste. The widget
    // caches the submitted frame and renders annotatedImage once at capture
    // time from the cached detections / depth map.

    result.detect.modelPath = job.modelPath;
    /* The device the backend actually resolved to (may differ from
     * job.device when the GPU lease failed and yolo fell back to CPU). */
    result.detect.resolvedDevice = m_resolvedDevice;
    result.detect.imageName = QStringLiteral("live");
    result.ok = true;
    applyTracking(job, result.task, result);
    emit inferComplete(result);
#endif
}

#ifdef AICore_ENABLED
void YOLOLiveInferWorker::applyTracking(const Job& job,
                                        const QString& task,
                                        Result& result) {
    // Tracking is a frame-sequence postprocess over the trackable
    // detection families (the same task set as the upstream track mode);
    // depth/classify/semantic runs are never fed to the tracker.
    if (job.trackerType.isEmpty()) return;
    if (task != QStringLiteral("detect") && task != QStringLiteral("segment") &&
        task != QStringLiteral("pose") && task != QStringLiteral("obb")) {
        return;
    }
    // Same condition as the aicore_yolo_set_track_recovery call in
    // runJobImpl (upstream want_recovered).
    const bool wantRecovery =
            job.trackerType == QStringLiteral("tracktrack") &&
            (task == QStringLiteral("detect") || task == QStringLiteral("obb"));

    // Config fingerprint: a changed tracker type / GMC method / exposed
    // threshold rebuilds the state machine (equivalent to a reset).
    if (m_trackCfgValid &&
        (!m_trackCfg || !sameTrackConfig(*m_trackCfg, job))) {
        m_trackCfgValid = false;
    }
    // Generation binding: a new stream generation (source switch, seek,
    // loop, restart, stop) invalidates the track table.
    if (m_trackCfgValid && job.generation != m_trackGeneration) {
        m_trackCfgValid = false;
    }
    if (!m_trackCfgValid) {
        m_trackWarning.clear();
        auto cfg = std::make_unique<qyolo::track::TrackConfig>();
        if (!qyolo::track::default_tracker_config(job.trackerType.toStdString(),
                                                  *cfg)) {
            // Unknown type: surface once through the result warning, then
            // park on this config (no per-frame retry spam until the
            // config changes).
            m_trackWarning = QStringLiteral("Unknown tracker type '%1'")
                                     .arg(job.trackerType);
            m_trackCfg = std::move(cfg);
            m_trackCfgValid = true;
            m_trackGeneration = job.generation;
            result.warning = m_trackWarning;
            return;
        }
        cfg->tracker_type = job.trackerType.toStdString();
        cfg->gmc_method = job.gmcMethod.toStdString();
        cfg->with_reid = job.withReid;
        cfg->track_high_thresh = job.trackHighThresh;
        cfg->track_low_thresh = job.trackLowThresh;
        cfg->new_track_thresh = job.newTrackThresh;
        cfg->track_buffer = job.trackBuffer;
        cfg->match_thresh = job.matchThresh;
        auto tracker = qyolo::track::create_tracker(*cfg);
        if (!tracker) {
            // Rejected (e.g. with an OpenCV-less build asking for an
            // OpenCV-only GMC method): surface once, then park as above.
            m_trackWarning = QStringLiteral(
                                     "Tracker rejected: type '%1' with GMC "
                                     "'%2' (orb/sift/ecc need the plugin's "
                                     "OpenCV build)")
                                     .arg(job.trackerType, job.gmcMethod);
            m_trackCfg = std::move(cfg);
            m_trackCfgValid = true;
            m_trackGeneration = job.generation;
            result.warning = m_trackWarning;
            return;
        }
        m_tracker = std::move(tracker);
        m_trackCfg = std::move(cfg);
        m_trackCfgValid = true;
        m_trackGeneration = job.generation;
    } else if (!m_trackWarning.isEmpty()) {
        // Parked on a failed config: keep surfacing the reason until the
        // user changes something (the widget logs it once per change).
        result.warning = m_trackWarning;
        return;
    }
    if (!m_tracker) return;

    // Build the tracker frame input: detections in original-image pixel
    // coordinates, center format (obb carries its angle; axis-aligned
    // boxes use the documented sentinel). FrameInput.idx joins the track
    // id back to the result row it came from. Official trackzone
    // semantics: rows whose center lies outside job.trackZone keep their
    // full-set row index but never join the association pool, so they are
    // reported without ids (drawn untracked, like the masked-out region).
    const QRectF zone = job.trackZone;
    const bool zoneActive =
            zone.isValid() && zone.width() > 0.0 && zone.height() > 0.0;
    const int n = static_cast<int>(result.detect.detections.size());
    std::vector<qyolo::track::TrackDet> dets;
    if (task == QStringLiteral("obb")) {
        dets.reserve(static_cast<size_t>(result.detect.obbBoxes.size()));
        int idx = 0;
        for (const YOLOObbBox& b : result.detect.obbBoxes) {
            const int row = idx++;
            if (zoneActive && !zone.contains(b.cx, b.cy)) continue;
            qyolo::track::TrackDet t;
            t.cx = b.cx;
            t.cy = b.cy;
            t.w = b.w;
            t.h = b.h;
            t.angle = b.angle;
            t.score = b.score;
            t.class_id = static_cast<int>(b.classId);
            t.idx = row;
            dets.push_back(t);
        }
    } else if (task == QStringLiteral("pose")) {
        dets.reserve(static_cast<size_t>(result.detect.keypointSets.size()));
        int idx = 0;
        for (const YOLOKeypointSet& ks : result.detect.keypointSets) {
            const int row = idx++;
            const float cx = (ks.det.x1 + ks.det.x2) * 0.5f;
            const float cy = (ks.det.y1 + ks.det.y2) * 0.5f;
            if (zoneActive && !zone.contains(cx, cy)) continue;
            qyolo::track::TrackDet t;
            t.cx = cx;
            t.cy = cy;
            t.w = ks.det.x2 - ks.det.x1;
            t.h = ks.det.y2 - ks.det.y1;
            t.score = ks.det.score;
            t.class_id = static_cast<int>(ks.det.classId);
            t.idx = row;
            dets.push_back(t);
        }
    } else {
        dets.reserve(static_cast<size_t>(n));
        int idx = 0;
        for (const YOLODetection& d : result.detect.detections) {
            const int row = idx++;
            const float cx = (d.x1 + d.x2) * 0.5f;
            const float cy = (d.y1 + d.y2) * 0.5f;
            if (zoneActive && !zone.contains(cx, cy)) continue;
            qyolo::track::TrackDet t;
            t.cx = cx;
            t.cy = cy;
            t.w = d.x2 - d.x1;
            t.h = d.y2 - d.y1;
            t.score = d.score;
            t.class_id = static_cast<int>(d.classId);
            t.idx = row;
            dets.push_back(t);
        }
    }

    // GMC input: a tightly-packed RGB8 view of the frame. QImage scanlines
    // are 32-bit aligned, so a RGB888 image whose width is not a multiple
    // of 4 owns padded rows — copy those to a compact buffer once; when
    // the rows are already compact the tracker borrows the QImage storage
    // synchronously inside update().
    qyolo::track::RgbFrame frame;
    std::vector<uint8_t> tightRgb;
    const QImage& img = job.rgb;
    if (!img.isNull() && img.format() == QImage::Format_RGB888) {
        const int rowBytes = img.width() * 3;
        if (img.bytesPerLine() == rowBytes) {
            frame.w = img.width();
            frame.h = img.height();
            frame.rgb = img.constBits();
        } else {
            tightRgb.resize(static_cast<size_t>(img.width()) *
                            static_cast<size_t>(img.height()) * 3);
            for (int y = 0; y < img.height(); ++y) {
                std::memcpy(tightRgb.data() + static_cast<size_t>(y) * rowBytes,
                            img.constScanLine(y),
                            static_cast<size_t>(rowBytes));
            }
            frame.w = img.width();
            frame.h = img.height();
            frame.rgb = tightRgb.data();
        }
    }

    qyolo::track::FrameInput input;
    input.dets = dets;
    input.frame = frame.rgb != nullptr ? &frame : nullptr;
    // Object-feature export toggle: flips the context-side switch only on
    // change (the toggle rebuilds the graph plan on the next inference).
    // A model reload resets the context, so the requested state is tracked
    // alongside. Not needed when an explicit ReID encoder owns appearance
    // features (the detector graph stays tap-free and lighter).
    const bool wantFeatExport = job.withReid && job.reidModelPath.isEmpty();
    if (wantFeatExport != m_objFeatEnabled) {
        aicore_yolo_set_detector_features(m_ctx, wantFeatExport ? 1 : 0);
        m_objFeatEnabled = wantFeatExport;
    }
    // Appearance features for the tracker, two mutually exclusive sources:
    //  - explicit encoder (official model=<path>): the picked
    //    reid-yolo26{n..x}-{f32,f16,q8_0}.gguf runs per-detection embeds
    //    on this worker (aicore_reid_embed_image) over the SAME tight RGB
    //    frame the tracker receives;
    //  - model="auto" detector features (aicore_yolo_features_view):
    //    per-detection rows index-aligned with the FULL detection set;
    //    empty for end2end heads or when the export is off — the tracker
    //    then runs motion-only association (upstream "feats missing"
    //    semantics).
    if (job.withReid && m_reidCtx != nullptr) {
        if (!dets.empty() && frame.rgb != nullptr) {
            std::vector<float> boxes;
            boxes.reserve(dets.size() * 4);
            for (const auto& d : dets) {
                boxes.push_back(d.cx - d.w * 0.5f);
                boxes.push_back(d.cy - d.h * 0.5f);
                boxes.push_back(d.cx + d.w * 0.5f);
                boxes.push_back(d.cy + d.h * 0.5f);
            }
            aicore_image_view view{};
            view.data = frame.rgb;
            view.width = frame.w;
            view.height = frame.h;
            view.row_stride_bytes = static_cast<size_t>(frame.w) * 3;
            view.format = AICORE_IMAGE_RGB8;
            float* emb = nullptr;
            int32_t ecount = 0, edim = 0;
            if (aicore_reid_embed_image(m_reidCtx, &view, boxes.data(),
                                        static_cast<int32_t>(dets.size()), &emb,
                                        &ecount, &edim) == 0 &&
                emb != nullptr && edim > 0) {
                input.feats.resize(dets.size());
                for (size_t i = 0; i < dets.size(); ++i) {
                    input.feats[i].assign(emb + (size_t)i * edim,
                                          emb + (size_t)(i + 1) * edim);
                }
            }
        }
    } else if (job.withReid && m_objFeatEnabled) {
        const float* fdata = nullptr;
        int32_t fcount = 0, fdim = 0;
        aicore_yolo_features_view(m_ctx, &fdata, &fcount, &fdim);
        if (fdata != nullptr && fdim > 0) {
            input.feats.resize(dets.size());
            const int rows = static_cast<int>(dets.size());
            for (int i = 0; i < rows && i < fcount; ++i) {
                // Feature rows are indexed by the FULL detection set; dets
                // rows carry their original row in idx (a trackzone filter
                // can make it differ from i).
                const int srcRow = dets[static_cast<size_t>(i)].idx;
                if (srcRow < 0 || srcRow >= fcount) continue;
                input.feats[static_cast<size_t>(i)].assign(
                        fdata + (size_t)srcRow * fdim,
                        fdata + (size_t)(srcRow + 1) * fdim);
            }
        }
    }
    // TrackTrack loose-NMS recovery: rows the tight NMS suppressed, read
    // straight from the context (idx = -1, upstream semantics — they do
    // not join the rendered detection rows); the tracker filters them by
    // track_high_thresh before they join the association pool.
    if (wantRecovery) {
        const int rn = aicore_yolo_recovery_count(m_ctx);
        input.dets_del.reserve(rn > 0 ? static_cast<size_t>(rn) : 0);
        for (int i = 0; i < rn; ++i) {
            const aicore_yolo_detection r = aicore_yolo_recovery_at(m_ctx, i);
            qyolo::track::TrackDet t;
            t.cx = (r.x1 + r.x2) * 0.5f;
            t.cy = (r.y1 + r.y2) * 0.5f;
            t.w = r.x2 - r.x1;
            t.h = r.y2 - r.y1;
            t.angle = -10.0f;  // axis-aligned sentinel
            t.score = r.score;
            t.class_id = r.class_id;
            t.idx = -1;
            input.dets_del.push_back(t);
        }
    }
    const std::vector<qyolo::track::TrackedBox> tracks =
            m_tracker->update(input);

    QVector<int> ids;
    const int rows = task == QStringLiteral("obb")
                             ? result.detect.obbBoxes.size()
                             : (task == QStringLiteral("pose")
                                        ? result.detect.keypointSets.size()
                                        : n);
    ids.resize(rows > 0 ? rows : 0);
    for (const qyolo::track::TrackedBox& tb : tracks) {
        if (tb.det.idx >= 0 && tb.det.idx < ids.size() && tb.track_id > 0) {
            ids[tb.det.idx] = tb.track_id;
        }
    }
    result.trackIds = ids;
}
#endif
