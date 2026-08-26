// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoWorker.h"

#include <CVLog.h>

#include <QElapsedTimer>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>

#ifdef AICore_ENABLED
#include <aicore/runtime_capi.h>
#endif

namespace {

#ifdef AICore_ENABLED
class DeviceTaskGuard {
public:
    explicit DeviceTaskGuard(const QString& device)
        : m_locked(aicore_device_task_lock(device.toUtf8().constData()) == 0) {}
    ~DeviceTaskGuard() {
        if (m_locked) aicore_device_task_unlock();
    }
    bool isLocked() const { return m_locked; }

private:
    bool m_locked = false;
};
#endif

}  // namespace

VideoWorker::VideoWorker(QObject* parent) : QThread(parent) {}

VideoWorker::~VideoWorker() {
    requestCancel();
    // Deleting a QThread that is still running aborts the process; give a
    // running track/load task a generous window to finish.
    if (isRunning()) wait();
#ifdef AICore_ENABLED
    DeviceTaskGuard taskGuard(m_device);
#endif
    if (m_tracker) aicore_sam3_tracker_free(m_tracker);
    if (m_ctx) aicore_sam3_free(m_ctx);
}

void VideoWorker::post(const TrackRequest& req) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_cancel.load()) return;
        if (req.action == Action::TrackFrame) {
            // Consumer-driven playback should normally keep one frame in
            // flight. Also enforce latest-wins here so seeks/speed changes
            // cannot build an unbounded stale frame backlog.
            for (int i = m_queue.size() - 1; i >= 0; --i) {
                if (m_queue[i].action == Action::TrackFrame) {
                    m_queue.removeAt(i);
                }
            }
        }
        m_queue.append(req);
        m_condition.wakeOne();
    }
    if (!isRunning()) start();
}

void VideoWorker::requestCancel() {
    {
        QMutexLocker lock(&m_mutex);
        m_cancel.store(true);
        m_queue.clear();
        m_condition.wakeAll();
    }
}

void VideoWorker::run() {
    for (;;) {
        TrackRequest req;
        {
            QMutexLocker lock(&m_mutex);
            while (m_queue.isEmpty() && !m_cancel.load()) {
                m_condition.wait(&m_mutex);
            }
            if (m_cancel.load()) break;
            req = m_queue.takeFirst();
        }
        if (req.cancel) break;
        try {
            process(req);
        } catch (const std::exception& e) {
            // Never let an exception escape a QThread (std::terminate →
            // process abort); reset the busy flag so the UI stays usable.
            m_busy = false;
            emit busyChanged(false);
            emit logMessage(QString("[SAM3] track error: %1")
                                    .arg(QString::fromUtf8(e.what())));
        } catch (...) {
            m_busy = false;
            emit busyChanged(false);
            emit logMessage(tr("[SAM3] unknown track error"));
        }
    }
}

void VideoWorker::process(const TrackRequest& req) {
    m_busy = true;
    emit busyChanged(true);

    if (req.action == Action::LoadModel) {
        // A device switch must release the old context under the old device's
        // queue before taking the new queue for model load. Otherwise a
        // CUDA -> Vulkan switch, for example, could free CUDA buffers while
        // another CUDA task is computing.
        m_hasModel.store(false);
        if (m_tracker || m_ctx) {
#ifdef AICore_ENABLED
            DeviceTaskGuard releaseGuard(m_device);
            if (!releaseGuard.isLocked()) {
                emit logMessage(tr("[SAM3] Failed to acquire the previous "
                                   "inference device for model release."));
                m_busy = false;
                emit busyChanged(false);
                return;
            }
#endif
            if (m_tracker) {
                aicore_sam3_tracker_free(m_tracker);
                m_tracker = nullptr;
            }
            if (m_ctx) {
                aicore_sam3_free(m_ctx);
                m_ctx = nullptr;
            }
        }
        m_encodedFrameIndex = -1;
        m_device = req.device;
    }
#ifdef AICore_ENABLED
    DeviceTaskGuard taskGuard(m_device);
    if (!taskGuard.isLocked()) {
        CVLog::Warning(
                "[qSAM3][VideoWorker] failed to acquire inference device (%s)",
                m_device.toUtf8().constData());
        emit logMessage(
                tr("[SAM3] Failed to acquire the inference device; another "
                   "task is running."));
        m_busy = false;
        emit busyChanged(false);
        return;
    }
#endif

    QElapsedTimer timer;
    timer.start();

    switch (req.action) {
        case Action::LoadModel: {
            aicore_sam3_options* opts = aicore_sam3_options_new();
            if (opts) {
                aicore_sam3_options_set_device(opts,
                                               req.device.toUtf8().constData());
                aicore_sam3_options_set_threads(opts, 4);
                aicore_sam3_options_set_score_threshold(opts,
                                                         req.scoreThreshold);
                m_ctx = aicore_sam3_load_opts(
                        req.modelPath.toUtf8().constData(), opts);
                aicore_sam3_options_free(opts);
            }
            if (!m_ctx || !aicore_sam3_is_ready(m_ctx)) {
                const char* err = m_ctx ? aicore_sam3_last_error(m_ctx)
                                        : aicore_sam3_last_load_error();
                CVLog::Warning("[qSAM3][VideoWorker] LoadModel failed: %s",
                               err ? err : "unknown error");
                emit logMessage(tr("Failed to load model: %1")
                                        .arg(err ? QString::fromUtf8(err)
                                                 : "unknown error"));
                break;
            }
            m_visualOnly = aicore_sam3_context_visual_only(m_ctx) != 0;
            m_tracker = aicore_sam3_tracker_create(m_ctx);
            if (!m_tracker) {
                CVLog::Warning("[qSAM3][VideoWorker] tracker_create failed: %s",
                               aicore_sam3_last_error(m_ctx));
                emit logMessage(tr("Failed to create tracker: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
                break;
            }
            m_hasModel.store(true);
            if (!m_visualOnly && !req.textPrompt.isEmpty()) {
                aicore_sam3_tracker_set_text_prompt(
                        m_tracker, req.textPrompt.toUtf8().constData());
            }
            emit modelReady(
                    QString::fromUtf8(aicore_sam3_context_backend_name(m_ctx)),
                    m_visualOnly);
            CVLog::Print(
                    "[qSAM3][VideoWorker] model+tracker ready on %s "
                    "(visual_only=%d) in %.0f ms",
                    aicore_sam3_context_backend_name(m_ctx), m_visualOnly ? 1 : 0,
                    static_cast<double>(timer.elapsed()));
            emit logMessage(
                    tr("Model + tracker ready on %1 (visual-only: %2)")
                            .arg(aicore_sam3_context_backend_name(m_ctx))
                            .arg(m_visualOnly ? "yes" : "no"));
            break;
        }
        case Action::TrackFrame: {
            if (!m_ctx || !m_tracker || req.frame.isNull()) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] TrackFrame skipped: ctx=%d "
                        "tracker=%d frame_null=%d",
                        m_ctx ? 1 : 0, m_tracker ? 1 : 0, req.frame.isNull() ? 1 : 0);
                break;
            }
            QImage rgb = req.frame.convertToFormat(QImage::Format_RGB888);
            aicore_sam3_seg_result* res =
                    m_visualOnly
                            ? aicore_sam3_propagate_frame(
                                      m_tracker, rgb.constBits(), rgb.width(),
                                      rgb.height(),
                                      static_cast<size_t>(rgb.bytesPerLine()))
                            : aicore_sam3_track_frame(
                                      m_tracker, rgb.constBits(), rgb.width(),
                                      rgb.height(),
                                      static_cast<size_t>(rgb.bytesPerLine()));
            if (!res) {
                const QString error = QString::fromUtf8(
                        aicore_sam3_tracker_last_error(m_tracker));
                CVLog::Warning("[qSAM3][VideoWorker] TrackFrame(%d) failed: %s",
                               req.frameIndex,
                               aicore_sam3_tracker_last_error(m_tracker));
                emit logMessage(tr("Track frame failed: %1").arg(error));
                SAM3WorkerResult failed;
                failed.errorMsg = tr("Track frame failed: %1").arg(error);
                emit frameResultReady(failed, req.frameIndex);
                break;
            }
            m_lastFrameResult = buildResult(res, req.frame);
            m_encodedFrameIndex = req.frameIndex;
            CVLog::Print(
                    "[qSAM3][VideoWorker] TrackFrame(%d) ok: %d det(s), "
                    "e2e=%.1f ms",
                    req.frameIndex, m_lastFrameResult.detCount,
                    static_cast<double>(m_lastFrameResult.timings.e2e_ms));
            emit frameResultReady(m_lastFrameResult, req.frameIndex);
            aicore_sam3_seg_result_free(res);
            break;
        }
        case Action::AddInstance: {
            if (!m_ctx || !m_tracker) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] AddInstance skipped: no model/"
                        "tracker");
                break;
            }
            // Upstream main_video.cpp requires frame_encoded before
            // add_instance: tracker_add_instance runs PVS internally and
            // sam3_segment_pvs bails on an unencoded state (log:
            // "image not encoded — call sam3_encode_image first"). If the
            // current frame has not been tracked yet (user drew a box right
            // after opening the video without pressing Play), run one
            // track/propagate pass first so the encoded features are valid.
            if (req.frameIndex != m_encodedFrameIndex) {
                if (req.frame.isNull()) {
                    CVLog::Warning(
                            "[qSAM3][VideoWorker] AddInstance(%d) has no frame "
                            "to encode; state is stale (encoded=%d)",
                            req.frameIndex, m_encodedFrameIndex);
                } else {
                    CVLog::Print(
                            "[qSAM3][VideoWorker] AddInstance(%d): state not "
                            "encoded (last=%d), running one track pass first",
                            req.frameIndex, m_encodedFrameIndex);
                    QImage rgb = req.frame.convertToFormat(
                            QImage::Format_RGB888);
                    aicore_sam3_seg_result* r =
                            m_visualOnly
                                    ? aicore_sam3_propagate_frame(
                                              m_tracker, rgb.constBits(),
                                              rgb.width(), rgb.height(),
                                              static_cast<size_t>(
                                                      rgb.bytesPerLine()))
                                    : aicore_sam3_track_frame(
                                              m_tracker, rgb.constBits(),
                                              rgb.width(), rgb.height(),
                                              static_cast<size_t>(
                                                      rgb.bytesPerLine()));
                    if (!r) {
                        CVLog::Warning(
                                "[qSAM3][VideoWorker] AddInstance(%d) "
                                "pre-encode failed: %s",
                                req.frameIndex,
                                aicore_sam3_tracker_last_error(m_tracker));
                        emit logMessage(tr("Cannot annotate this frame: %1")
                                                .arg(aicore_sam3_tracker_last_error(
                                                        m_tracker)));
                        break;
                    }
                    aicore_sam3_seg_result_free(r);
                    m_encodedFrameIndex = req.frameIndex;
                }
            }
            std::vector<aicore_sam3_point> pos;
            std::vector<aicore_sam3_point> neg;
            for (const auto& pt : req.prompt.posPoints) {
                pos.push_back({pt.x, pt.y});
            }
            for (const auto& pt : req.prompt.negPoints) {
                neg.push_back({pt.x, pt.y});
            }
            aicore_sam3_pvs_prompt prompt{};
            prompt.pos_points = pos.empty() ? nullptr : pos.data();
            prompt.n_pos_points = static_cast<int>(pos.size());
            prompt.neg_points = neg.empty() ? nullptr : neg.data();
            prompt.n_neg_points = static_cast<int>(neg.size());
            if (req.prompt.usePvsBox) {
                prompt.box = req.prompt.pvsBox;
                prompt.use_box = 1;
            }
            prompt.multimask = req.prompt.multimask ? 1 : 0;
            const int newId =
                    aicore_sam3_tracker_add_instance(m_tracker, &prompt);
            if (newId >= 0) {
                CVLog::Print(
                        "[qSAM3][VideoWorker] AddInstance(%d) ok: id=%d "
                        "(pts=%d box=%d)",
                        req.frameIndex, newId,
                        static_cast<int>(pos.size()),
                        prompt.use_box ? 1 : 0);
                emit instanceAdded(newId);
                // Upstream main_video.cpp re-runs PVS on the encoded frame
                // right after add_instance to display the new instance's mask
                // immediately (without advancing the tracker). Merge it into
                // the last tracked result so existing instances stay visible
                // and the timeline keeps both.
                if (aicore_sam3_seg_result* maskRes =
                            aicore_sam3_tracker_segment_pvs(m_tracker,
                                                            &prompt)) {
                    SAM3WorkerResult maskResult = buildResult(maskRes, req.frame);
                    for (int i = 0; i < maskResult.detCount; ++i) {
                        maskResult.instanceIds[i] = newId;
                    }
                    aicore_sam3_seg_result_free(maskRes);
                    if (!maskResult.instanceMasks.isEmpty()) {
                        SAM3WorkerResult merged = m_lastFrameResult;
                        merged.boxes.append(maskResult.boxes);
                        merged.scores.append(maskResult.scores);
                        merged.ious.append(maskResult.ious);
                        merged.instanceIds.append(maskResult.instanceIds);
                        merged.instanceMasks.append(maskResult.instanceMasks);
                        merged.detCount = merged.boxes.size();
                        merged.valid = true;
                        merged.timings = maskResult.timings;
                        m_lastFrameResult = merged;
                        emit frameResultReady(m_lastFrameResult,
                                              req.frameIndex);
                    }
                }
            } else {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] AddInstance(%d) failed: %s",
                        req.frameIndex,
                        aicore_sam3_tracker_last_error(m_tracker));
                emit logMessage(tr("Add instance failed: %1")
                                        .arg(aicore_sam3_tracker_last_error(
                                                m_tracker)));
            }
            break;
        }
        case Action::RefineInstance: {
            if (!m_ctx || !m_tracker) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] RefineInstance skipped: no "
                        "model/tracker");
                break;
            }
            // Refine runs PVS internally too; encode the current frame first
            // when the state is stale (same rule as AddInstance).
            if (req.frameIndex != m_encodedFrameIndex) {
                if (req.frame.isNull()) {
                    CVLog::Warning(
                            "[qSAM3][VideoWorker] RefineInstance(%d) has no "
                            "frame to encode; state is stale (encoded=%d)",
                            req.frameIndex, m_encodedFrameIndex);
                } else {
                    CVLog::Print(
                            "[qSAM3][VideoWorker] RefineInstance(%d): state "
                            "not encoded (last=%d), running one track pass "
                            "first",
                            req.frameIndex, m_encodedFrameIndex);
                    QImage rgb = req.frame.convertToFormat(
                            QImage::Format_RGB888);
                    aicore_sam3_seg_result* r =
                            m_visualOnly
                                    ? aicore_sam3_propagate_frame(
                                              m_tracker, rgb.constBits(),
                                              rgb.width(), rgb.height(),
                                              static_cast<size_t>(
                                                      rgb.bytesPerLine()))
                                    : aicore_sam3_track_frame(
                                              m_tracker, rgb.constBits(),
                                              rgb.width(), rgb.height(),
                                              static_cast<size_t>(
                                                      rgb.bytesPerLine()));
                    if (!r) {
                        CVLog::Warning(
                                "[qSAM3][VideoWorker] RefineInstance(%d) "
                                "pre-encode failed: %s",
                                req.frameIndex,
                                aicore_sam3_tracker_last_error(m_tracker));
                        emit logMessage(tr("Cannot refine this frame: %1")
                                                .arg(aicore_sam3_tracker_last_error(
                                                        m_tracker)));
                        emit instanceRefined(req.instanceId, false);
                        break;
                    }
                    aicore_sam3_seg_result_free(r);
                    m_encodedFrameIndex = req.frameIndex;
                }
            }
            std::vector<aicore_sam3_point> pos;
            std::vector<aicore_sam3_point> neg;
            for (const auto& pt : req.prompt.posPoints) {
                pos.push_back({pt.x, pt.y});
            }
            for (const auto& pt : req.prompt.negPoints) {
                neg.push_back({pt.x, pt.y});
            }
            aicore_sam3_pvs_prompt prompt{};
            prompt.pos_points = pos.empty() ? nullptr : pos.data();
            prompt.n_pos_points = static_cast<int>(pos.size());
            prompt.neg_points = neg.empty() ? nullptr : neg.data();
            prompt.n_neg_points = static_cast<int>(neg.size());
            prompt.multimask = 0;
            const int ok = aicore_sam3_refine_instance(
                    m_tracker, req.instanceId, pos.data(),
                    static_cast<int>(pos.size()), neg.data(),
                    static_cast<int>(neg.size()));
            if (ok != 0) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] RefineInstance(%d, id=%d) "
                        "failed: %s",
                        req.frameIndex, req.instanceId,
                        aicore_sam3_tracker_last_error(m_tracker));
            } else {
                CVLog::Print(
                        "[qSAM3][VideoWorker] RefineInstance(%d, id=%d) ok",
                        req.frameIndex, req.instanceId);
                if (aicore_sam3_seg_result* maskRes =
                            aicore_sam3_tracker_segment_pvs(m_tracker,
                                                            &prompt)) {
                    SAM3WorkerResult refined = buildResult(maskRes, req.frame);
                    aicore_sam3_seg_result_free(maskRes);
                    for (int i = 0; i < refined.instanceIds.size(); ++i) {
                        refined.instanceIds[i] = req.instanceId;
                    }

                    SAM3WorkerResult merged;
                    merged.valid = true;
                    for (int i = 0; i < m_lastFrameResult.detCount; ++i) {
                        if (m_lastFrameResult.instanceIds.value(i, -1) ==
                            req.instanceId) {
                            continue;
                        }
                        merged.boxes.append(m_lastFrameResult.boxes.value(i));
                        merged.scores.append(
                                m_lastFrameResult.scores.value(i));
                        merged.ious.append(m_lastFrameResult.ious.value(i));
                        merged.instanceIds.append(
                                m_lastFrameResult.instanceIds.value(i, -1));
                        merged.instanceMasks.append(
                                m_lastFrameResult.instanceMasks.value(i));
                    }
                    merged.boxes.append(refined.boxes);
                    merged.scores.append(refined.scores);
                    merged.ious.append(refined.ious);
                    merged.instanceIds.append(refined.instanceIds);
                    merged.instanceMasks.append(refined.instanceMasks);
                    merged.detCount = merged.boxes.size();
                    merged.timings = refined.timings;
                    m_lastFrameResult = merged;
                    emit frameResultReady(m_lastFrameResult,
                                          req.frameIndex);
                }
            }
            emit instanceRefined(req.instanceId, ok == 0);
            break;
        }
        case Action::ResetTracker: {
            if (!m_ctx) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] ResetTracker skipped: no model");
                emit trackerReset(req.textPrompt, false);
                break;
            }
            m_lastFrameResult = SAM3WorkerResult{};
            m_encodedFrameIndex = -1;
            if (aicore_sam3_set_score_threshold(m_ctx, req.scoreThreshold) !=
                0) {
                emit logMessage(tr("Invalid score threshold: %1")
                                        .arg(req.scoreThreshold));
                emit trackerReset(req.textPrompt, false);
                break;
            }
            m_hasModel.store(false);
            if (m_tracker) aicore_sam3_tracker_free(m_tracker);
            m_tracker = aicore_sam3_tracker_create(m_ctx);
            if (!m_tracker) {
                CVLog::Warning(
                        "[qSAM3][VideoWorker] ResetTracker re-create failed: %s",
                        aicore_sam3_last_error(m_ctx));
                emit logMessage(tr("Failed to re-create tracker: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
                emit trackerReset(req.textPrompt, false);
                break;
            }
            m_hasModel.store(true);
            if (!m_visualOnly && !req.textPrompt.isEmpty()) {
                aicore_sam3_tracker_set_text_prompt(
                        m_tracker, req.textPrompt.toUtf8().constData());
            }
            CVLog::Print("[qSAM3][VideoWorker] ResetTracker ok");
            emit logMessage(tr("Tracker reset."));
            emit trackerReset(req.textPrompt, true);
            break;
        }
        case Action::None:
            break;
    }

    m_busy = false;
    emit busyChanged(false);
}

SAM3WorkerResult VideoWorker::buildResult(aicore_sam3_seg_result* segRes,
                                          const QImage& frame) {
    SAM3WorkerResult r;
    if (!segRes) return r;
    const int n = aicore_sam3_seg_det_count(segRes);
    r.detCount = n;
    r.valid = true;
    qint64 totalNonZero = 0;
    for (int i = 0; i < n; ++i) {
        r.boxes.append(aicore_sam3_seg_det_box_at(segRes, i));
        r.scores.append(aicore_sam3_seg_det_score_at(segRes, i));
        r.ious.append(aicore_sam3_seg_det_iou_at(segRes, i));
        r.instanceIds.append(aicore_sam3_seg_det_instance_id_at(segRes, i));
        const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(segRes, i);
        QImage m(mask.width, mask.height, QImage::Format_Grayscale8);
        if (mask.data && !m.isNull()) {
            for (int y = 0; y < mask.height; ++y) {
                memcpy(m.scanLine(y),
                       static_cast<const uint8_t*>(mask.data) +
                               static_cast<size_t>(y) * mask.row_stride_bytes,
                       static_cast<size_t>(mask.width));
                const uint8_t* row = m.constScanLine(y);
                for (int x = 0; x < mask.width; ++x) {
                    totalNonZero += row[x] > 127 ? 1 : 0;
                }
            }
        }
        r.instanceMasks.append(m);
    }
    aicore_sam3_timings t{};
    if (m_tracker && aicore_sam3_tracker_last_timings(m_tracker, &t) == 0) {
        r.timings = t;
    }
    if (n > 0 && totalNonZero == 0) {
        r.errorMsg = tr("Decoder returned detections with empty masks.");
    }
    return r;
}
