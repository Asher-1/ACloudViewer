// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoWorker.h"

#include <QWaitCondition>

#include <algorithm>
#include <cstring>

VideoWorker::VideoWorker(QObject* parent) : QThread(parent) {}

VideoWorker::~VideoWorker() {
    requestCancel();
    if (isRunning()) wait(5000);
    if (m_tracker) aicore_sam3_tracker_free(m_tracker);
    if (m_ctx) aicore_sam3_free(m_ctx);
}

void VideoWorker::post(const TrackRequest& req) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_cancel) return;
        m_queue.append(req);
    }
    if (!isRunning()) start();
}

void VideoWorker::requestCancel() {
    TrackRequest req;
    req.cancel = true;
    {
        QMutexLocker lock(&m_mutex);
        m_cancel = true;
        m_queue.clear();
        m_queue.append(req);
    }
}

void VideoWorker::run() {
    for (;;) {
        TrackRequest req;
        {
            // Wait for work with a 50 ms poll so a pending cancel is picked
            // up promptly even if the queue is empty.
            QMutexLocker lock(&m_mutex);
            if (m_queue.isEmpty()) {
                lock.unlock();
                msleep(50);
                lock.relock();
                if (m_queue.isEmpty()) continue;
            }
            req = m_queue.takeFirst();
        }
        if (req.cancel) break;
        process(req);
    }
}

void VideoWorker::process(const TrackRequest& req) {
    m_busy = true;
    emit busyChanged(true);

    switch (req.action) {
        case Action::LoadModel: {
            // Release any previous session.
            if (m_tracker) {
                aicore_sam3_tracker_free(m_tracker);
                m_tracker = nullptr;
            }
            if (m_ctx) {
                aicore_sam3_free(m_ctx);
                m_ctx = nullptr;
            }
            aicore_sam3_options* opts = aicore_sam3_options_new();
            if (opts) {
                aicore_sam3_options_set_device(opts,
                                               req.device.toUtf8().constData());
                aicore_sam3_options_set_threads(opts, 4);
                m_ctx = aicore_sam3_load_opts(req.modelPath.toUtf8().constData(),
                                              opts);
                aicore_sam3_options_free(opts);
            }
            if (!m_ctx || !aicore_sam3_is_ready(m_ctx)) {
                emit logMessage(tr("Failed to load model: %1")
                                        .arg(m_ctx
                                                     ? aicore_sam3_last_error(m_ctx)
                                                     : "invalid options"));
                break;
            }
            m_visualOnly = aicore_sam3_context_visual_only(m_ctx) != 0;
            m_tracker = aicore_sam3_tracker_create(m_ctx);
            if (!m_tracker) {
                emit logMessage(tr("Failed to create tracker: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
                break;
            }
            if (!m_visualOnly && !req.textPrompt.isEmpty()) {
                aicore_sam3_tracker_set_text_prompt(
                        m_tracker, req.textPrompt.toUtf8().constData());
            }
            emit modelReady(
                    QString::fromUtf8(aicore_sam3_context_backend_name(m_ctx)),
                    m_visualOnly);
            emit logMessage(tr("Model + tracker ready on %1 (visual-only: %2)")
                                    .arg(aicore_sam3_context_backend_name(m_ctx))
                                    .arg(m_visualOnly ? "yes" : "no"));
            break;
        }
        case Action::TrackFrame: {
            if (!m_ctx || !m_tracker || req.frame.isNull()) break;
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
                emit logMessage(tr("Track frame failed: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
                break;
            }
            emit frameResultReady(buildResult(res, req.frame), req.frameIndex);
            aicore_sam3_seg_result_free(res);
            break;
        }
        case Action::AddInstance: {
            if (!m_ctx || !m_tracker) break;
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
            const int newId = aicore_sam3_tracker_add_instance(m_tracker, &prompt);
            if (newId >= 0) {
                emit instanceAdded(newId);
            } else {
                emit logMessage(tr("Add instance failed: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
            }
            break;
        }
        case Action::RefineInstance: {
            if (!m_ctx || !m_tracker) break;
            std::vector<aicore_sam3_point> pos;
            std::vector<aicore_sam3_point> neg;
            for (const auto& pt : req.prompt.posPoints) {
                pos.push_back({pt.x, pt.y});
            }
            for (const auto& pt : req.prompt.negPoints) {
                neg.push_back({pt.x, pt.y});
            }
            const int ok = aicore_sam3_refine_instance(
                    m_tracker, req.instanceId, pos.data(),
                    static_cast<int>(pos.size()), neg.data(),
                    static_cast<int>(neg.size()));
            emit instanceRefined(req.instanceId, ok == 0);
            break;
        }
        case Action::ResetTracker: {
            if (!m_ctx) break;
            if (m_tracker) aicore_sam3_tracker_free(m_tracker);
            m_tracker = aicore_sam3_tracker_create(m_ctx);
            if (!m_tracker) {
                emit logMessage(tr("Failed to re-create tracker: %1")
                                        .arg(aicore_sam3_last_error(m_ctx)));
                break;
            }
            if (!m_visualOnly && !req.textPrompt.isEmpty()) {
                aicore_sam3_tracker_set_text_prompt(
                        m_tracker, req.textPrompt.toUtf8().constData());
            }
            emit logMessage(tr("Tracker reset."));
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
    for (int i = 0; i < n; ++i) {
        r.boxes.append(aicore_sam3_seg_det_box_at(segRes, i));
        r.scores.append(aicore_sam3_seg_det_score_at(segRes, i));
        r.ious.append(aicore_sam3_seg_det_iou_at(segRes, i));
        r.instanceIds.append(aicore_sam3_seg_det_instance_id_at(segRes, i));
        const aicore_sam3_plane_view mask =
                aicore_sam3_seg_mask_at(segRes, i);
        QImage m(mask.width, mask.height, QImage::Format_Grayscale8);
        if (mask.data && !m.isNull()) {
            for (int y = 0; y < mask.height; ++y) {
                memcpy(m.scanLine(y),
                       mask.data +
                               static_cast<int64_t>(y) * mask.row_stride_bytes,
                       static_cast<size_t>(mask.width));
            }
        }
        r.instanceMasks.append(m);
    }
    aicore_sam3_timings t{};
    if (m_ctx && aicore_sam3_last_timings(m_ctx, &t) == 0) {
        r.timings = t;
    }
    return r;
}
