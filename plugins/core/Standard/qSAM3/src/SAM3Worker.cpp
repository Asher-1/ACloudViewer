// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SAM3Worker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <cstring>

#ifdef AICore_ENABLED
#include "aicore/runtime_capi.h"
#endif

static QImage blendMasksImpl(aicore_sam3_seg_result* res, int imgW, int imgH) {
    const int n = aicore_sam3_seg_det_count(res);
    if (n <= 0) return QImage();

    // Static instance colors matching the ImGui original
    static const QRgb kColors[] = {
            qRgb(255, 51, 51),  qRgb(51, 153, 255),  qRgb(51, 230, 76),
            qRgb(255, 204, 26), qRgb(204, 76, 230),  qRgb(255, 128, 26),
            qRgb(26, 230, 230), qRgb(230, 102, 153), qRgb(128, 204, 51),
            qRgb(76, 76, 255),  qRgb(255, 153, 179), qRgb(153, 255, 128),
    };
    static constexpr int kNColors = sizeof(kColors) / sizeof(kColors[0]);

    QImage overlay(imgW, imgH, QImage::Format_ARGB32);
    overlay.fill(qRgba(0, 0, 0, 0));

    for (int d = 0; d < n; ++d) {
        const aicore_sam3_plane_view m = aicore_sam3_seg_mask_at(res, d);
        if (!m.data || m.width <= 0 || m.height <= 0) continue;
        const QRgb color = kColors[d % kNColors];
        const int alpha = 102;  // ~0.4 * 255
        const uint8_t* src = static_cast<const uint8_t*>(m.data);

        for (int y = 0; y < m.height && y < imgH; ++y) {
            uint32_t* dstLine =
                    reinterpret_cast<uint32_t*>(overlay.scanLine(y));
            for (int x = 0; x < m.width && x < imgW; ++x) {
                if (src[static_cast<size_t>(y) * m.row_stride_bytes + x] >
                    127) {
                    const QRgb existing = dstLine[x];
                    const int a = alpha;
                    const int invA = 255 - a;
                    dstLine[x] = qRgba(
                            (qRed(color) * a + qRed(existing) * invA) / 255,
                            (qGreen(color) * a + qGreen(existing) * invA) / 255,
                            (qBlue(color) * a + qBlue(existing) * invA) / 255,
                            qAlpha(existing) +
                                    ((255 - qAlpha(existing)) * a) / 255);
                }
            }
        }
    }
    return overlay;
}

SAM3Worker::SAM3Worker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {}

SAM3Worker::~SAM3Worker() {
    requestCancel();
    if (m_pendingCtx) {
        aicore_sam3_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
}

void SAM3Worker::requestCancel() { m_cancelled = true; }

void SAM3Worker::run() {
    const bool ok = runInference();
    if (!m_cancelled) {
        emit progressUpdate(0, 0);
    }
}

bool SAM3Worker::runInference() {
    if (m_cancelled) return false;

    // Load model if needed
    if (m_action == SAM3WorkerAction::LoadModel ||
        m_action == SAM3WorkerAction::EncodeAndSegmentPVS ||
        m_action == SAM3WorkerAction::EncodeAndSegmentPCS) {
        if (!m_ctx) {
            aicore_sam3_options* opts = aicore_sam3_options_new();
            if (!opts) {
                emit logMessage("SAM3: Failed to allocate options.");
                return false;
            }
            aicore_sam3_options_set_device(
                    opts, m_settings.device.toUtf8().constData());
            aicore_sam3_options_set_threads(opts, m_settings.threads);
            aicore_sam3_options_set_encode_img_size(opts,
                                                    m_settings.encodeImgSize);
            aicore_sam3_options_set_score_threshold(opts,
                                                    m_settings.scoreThreshold);
            aicore_sam3_options_set_nms_threshold(opts,
                                                  m_settings.nmsThreshold);

            emit logMessage(
                    QString("Loading model: %1 (device=%2)")
                            .arg(QFileInfo(m_settings.modelPath).fileName(),
                                 m_settings.device));
            emit progressUpdate(0, 1);

            m_pendingCtx = aicore_sam3_load_opts(
                    m_settings.modelPath.toUtf8().constData(), opts);
            aicore_sam3_options_free(opts);

            if (!m_pendingCtx || !aicore_sam3_is_ready(m_pendingCtx)) {
                const char* err = m_pendingCtx
                                          ? aicore_sam3_last_error(m_pendingCtx)
                                          : "context allocation failed";
                emit logMessage(QString("Model load failed: %1")
                                        .arg(err ? QString::fromUtf8(err)
                                                 : "unknown error"));
                return false;
            }
            m_ctx = m_pendingCtx;
            m_pendingCtx = nullptr;

            const int modelType = aicore_sam3_context_model_type(m_ctx);
            const int visualOnly = aicore_sam3_context_visual_only(m_ctx);
            const char* backend = aicore_sam3_context_backend_name(m_ctx);
            emit logMessage(
                    QString("Model loaded: type=%1 visual_only=%2 backend=%3")
                            .arg(modelType)
                            .arg(visualOnly)
                            .arg(backend));
            emit modelReady(QString::fromUtf8(backend), modelType,
                            visualOnly != 0);
        }

        if (m_cancelled) return false;
    }

    if (m_action == SAM3WorkerAction::LoadModel) {
        SAM3WorkerResult ok;
        ok.valid = true;
        emit resultReady(ok);
        return true;
    }

    if (m_image.isNull()) {
        emit logMessage("No image loaded.");
        return false;
    }

    const QImage rgb = m_image.convertToFormat(QImage::Format_RGB888);
    const int w = rgb.width();
    const int h = rgb.height();
    const size_t stride = static_cast<size_t>(rgb.bytesPerLine());

    // Run encode + segment
    aicore_sam3_seg_result* segRes = nullptr;
    if (m_action == SAM3WorkerAction::EncodeAndSegmentPVS ||
        m_action == SAM3WorkerAction::SegmentOnly) {
        segRes = runPVS();
    } else if (m_action == SAM3WorkerAction::EncodeAndSegmentPCS) {
        segRes = runPCS();
    }

    if (!segRes && !m_cancelled) {
        emit logMessage("Segmentation returned no results.");
        return false;
    }

    if (m_cancelled) {
        if (segRes) aicore_sam3_seg_result_free(segRes);
        return false;
    }

    SAM3WorkerResult result = buildResult(segRes, rgb);
    if (segRes) aicore_sam3_seg_result_free(segRes);

    emit resultReady(result);
    return true;
}

aicore_sam3_seg_result* SAM3Worker::runPVS() {
    // Encode image
    const QImage rgb = m_image.convertToFormat(QImage::Format_RGB888);
    const int ok = aicore_sam3_encode_rgb(
            m_ctx, rgb.constBits(), rgb.width(), rgb.height(),
            static_cast<size_t>(rgb.bytesPerLine()), 1);
    if (ok != 0) {
        emit logMessage(QString("Encode failed: %1")
                                .arg(aicore_sam3_last_error(m_ctx)));
        return nullptr;
    }

    // Build prompt
    aicore_sam3_pvs_prompt prompt{};
    if (!m_prompt.posPoints.isEmpty()) {
        prompt.pos_points = reinterpret_cast<const aicore_sam3_point*>(
                m_prompt.posPoints.constData());
        prompt.n_pos_points = m_prompt.posPoints.size();
    }
    if (!m_prompt.negPoints.isEmpty()) {
        prompt.neg_points = reinterpret_cast<const aicore_sam3_point*>(
                m_prompt.negPoints.constData());
        prompt.n_neg_points = m_prompt.negPoints.size();
    }
    if (m_prompt.usePvsBox) {
        prompt.box = m_prompt.pvsBox;
        prompt.use_box = 1;
    }
    prompt.multimask = m_prompt.multimask ? 1 : 0;

    return aicore_sam3_segment_pvs_rgb(m_ctx, &prompt, rgb.constBits(),
                                       rgb.width(), rgb.height(),
                                       static_cast<size_t>(rgb.bytesPerLine()));
}

aicore_sam3_seg_result* SAM3Worker::runPCS() {
    // Encode image (with detector neck)
    const QImage rgb = m_image.convertToFormat(QImage::Format_RGB888);
    const int ok = aicore_sam3_encode_rgb(
            m_ctx, rgb.constBits(), rgb.width(), rgb.height(),
            static_cast<size_t>(rgb.bytesPerLine()), 0);
    if (ok != 0) {
        emit logMessage(QString("Encode (full) failed: %1")
                                .arg(aicore_sam3_last_error(m_ctx)));
        return nullptr;
    }

    aicore_sam3_pcs_prompt prompt{};
    prompt.text = m_prompt.text;
    if (!m_prompt.posExemplars.empty()) {
        prompt.pos_exemplars = m_prompt.posExemplars.data();
        prompt.n_pos_exemplars = static_cast<int>(m_prompt.posExemplars.size());
    }
    if (!m_prompt.negExemplars.empty()) {
        prompt.neg_exemplars = m_prompt.negExemplars.data();
        prompt.n_neg_exemplars = static_cast<int>(m_prompt.negExemplars.size());
    }
    prompt.score_threshold = m_prompt.scoreThreshold;
    prompt.nms_threshold = m_prompt.nmsThreshold;

    return aicore_sam3_segment_pcs_rgb(m_ctx, &prompt, rgb.constBits(),
                                       rgb.width(), rgb.height(),
                                       static_cast<size_t>(rgb.bytesPerLine()));
}

SAM3WorkerResult SAM3Worker::buildResult(aicore_sam3_seg_result* segRes,
                                         const QImage& img) {
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
        // Per-instance 0/255 mask at original image resolution, for
        // per-instance coloring (video timeline / overlay).
        const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(segRes, i);
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
    if (aicore_sam3_last_timings(m_ctx, &t) == 0) {
        r.timings = t;
    }

    r.maskComposite = blendMasksImpl(segRes, img.width(), img.height());
    return r;
}