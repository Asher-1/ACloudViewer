// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDWorker.h"

#include <QByteArray>
#include <QDateTime>
#include <QFileInfo>
#include <QImage>
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/runtime_raii.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

#ifdef AICore_ENABLED

/** Serializes inference per device for the duration of one task. The RAII
 *  guard is shared now (aicore/runtime_raii.h, same pattern as the other
 *  AICore plugin workers qRFDetr/qYOLO/qSAM3 …): without the lock, two
 *  dialogs inferring on the same GPU interleave ggml work on one device
 *  queue. */
inline aicore::runtime::DeviceTaskLock makeDeviceTaskLock(
        const QString& device) {
    const QByteArray bytes = device.toUtf8();
    return aicore::runtime::DeviceTaskLock(bytes.constData());
}

/** Converts one per-ROI engine result into a plugin keypoint set.
 *  Keypoints below minScore are dropped; the official-demo skeleton
 *  (when present) is resolved into flat segments over the shown points
 *  only, so bones never reference hidden endpoints. */
void collectRoi(const aicore_gkd_ctx* ctx,
                int roi,
                const QStringList& prompts,
                const QVector<QPair<int, int>>& skeleton,
                const QString& label,
                float bboxScore,
                float minScore,
                GKDKeypointSet* set) {
    set->label = label;
    set->hasBox = true;
    set->bboxScore = bboxScore;
    float box[4] = {0, 0, 0, 0};
    if (aicore_gkd_result_roi_bbox_at(ctx, roi, box) == 0) {
        set->x1 = box[0];
        set->y1 = box[1];
        set->x2 = box[2];
        set->y2 = box[3];
    }
    const int n = aicore_gkd_result_keypoint_count_at(ctx, roi);
    set->keypoints.reserve(n);
    // Full result list (no display filtering here): the COCO export
    // mirrors the official writer and emits every keypoint, while
    // rendering and bones apply minScore at their own layer.
    for (int i = 0; i < n; ++i) {
        const aicore_gkd_keypoint kp =
                aicore_gkd_result_keypoint_at_roi(ctx, roi, i);
        GKDKeypoint out;
        out.x = kp.x;
        out.y = kp.y;
        out.score = kp.score;
        const char* prompt = aicore_gkd_result_prompt_at_roi(ctx, roi, i);
        if (prompt != nullptr) out.prompt = QString::fromUtf8(prompt);
        set->keypoints.append(out);
    }
    set->bones =
            GKDHelpers::buildBones(prompts, set->keypoints, skeleton, minScore);
}

void fillTimings(const aicore_gkd_ctx* ctx, GKDRunResult* result) {
    aicore_gkd_timings t{};
    if (aicore_gkd_last_timings(ctx, &t) == 0) {
        result->preprocessMs = t.preprocess_ms;
        result->postprocessMs = t.decode_ms;
        result->runtimeMs = t.vision_ms + t.text_ms + t.detect_ms;
        result->totalRuntimeMs = t.e2e_ms;
    }
}

#endif  // AICore_ENABLED

}  // namespace

GKDWorker::GKDWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

GKDWorker::~GKDWorker() {
    releaseContextOnMainThread();
#ifdef AICore_ENABLED
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
#endif
}

void GKDWorker::releaseContextOnMainThread() {
    // The contexts are created on the worker thread; destroy them here
    // (main thread) so GPU teardown never races the render thread.
    // Handles borrowed from the shared cache are NOT freed here — the
    // cache owns them and the plugin frees them on the main thread.
#ifdef AICore_ENABLED
    if (m_pendingGkdCtx && m_ownsGkdCtx) {
        aicore_gkd_free(static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx));
    }
    m_pendingGkdCtx = nullptr;
    m_ownsGkdCtx = false;
    if (m_pendingYoloCtx && m_ownsYoloCtx) {
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx));
    }
    m_pendingYoloCtx = nullptr;
    m_ownsYoloCtx = false;
#endif
}

void GKDWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
#endif
}

void GKDWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(tr("[GKD] AICore is not enabled in this build."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED

bool GKDWorker::ensureGkdContext(QString* error) {
    if (m_pendingGkdCtx) return true;
    const QString key = QStringLiteral("%1|%2|%3")
                                .arg(m_settings.modelPath, m_settings.device)
                                .arg(m_settings.threads);
    if (m_settings.cache && m_settings.cache->gkdCtx &&
        m_settings.cache->gkdKey == key) {
        // Borrow the resident context: no disk read, no weight upload —
        // this is what makes repeat runs (and the multi-object rotation)
        // fast instead of reloading 483 MiB every click.
        m_pendingGkdCtx = m_settings.cache->gkdCtx;
        m_ownsGkdCtx = false;
        emit logMessage(
                tr("[GKD] Reusing loaded model: %1")
                        .arg(QFileInfo(m_settings.modelPath).fileName()));
        return true;
    }
    aicore_gkd_options* opts = aicore_gkd_options_new();
    if (!opts) {
        if (error) *error = tr("Failed to allocate GKD options.");
        return false;
    }
    aicore_gkd_options_set_device(opts, m_settings.device.toUtf8().constData());
    aicore_gkd_options_set_threads(opts, m_settings.threads);
    m_pendingGkdCtx = aicore_gkd_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_gkd_options_free(opts);
    auto* ctx = static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx);
    if (!ctx || aicore_gkd_is_ready(ctx) != 1) {
        const char* err =
                ctx ? aicore_gkd_last_error(ctx) : "context allocation failed";
        if (error) {
            *error = tr("GKD model load failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        }
        // A failed load leaves a non-ready handle: keep it owned so the
        // release path frees it (never cached — only ready contexts are).
        m_ownsGkdCtx = m_pendingGkdCtx != nullptr;
        return false;
    }
    emit logMessage(
            tr("[GKD] Model loaded: %1 (device=%2, threads=%3, input=%4)")
                    .arg(QFileInfo(m_settings.modelPath).fileName(),
                         aicore_gkd_context_device(ctx))
                    .arg(aicore_gkd_context_threads(ctx))
                    .arg(aicore_gkd_context_image_size(ctx)));
    if (m_settings.cache) {
        if (m_settings.cache->gkdCtx) {
            m_settings.cache->retiredGkd.push_back(m_settings.cache->gkdCtx);
        }
        m_settings.cache->gkdCtx = m_pendingGkdCtx;
        m_settings.cache->gkdKey = key;
        m_ownsGkdCtx = false;  // the cache owns it now
    } else {
        m_ownsGkdCtx = true;
    }
    return true;
}

bool GKDWorker::detectSingleObject(QVector<GKDKeypointSet>* sets,
                                   QString* error) {
    if (!ensureGkdContext(error)) return false;
    auto* ctx = static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx);

    emit taskStage(tr("Loading query image..."), 10);
    QImage source(m_settings.inputPath);
    if (source.isNull()) {
        if (error)
            *error = tr("Failed to load image: %1").arg(m_settings.inputPath);
        return false;
    }

    aicore_image_view view{};
    if (!GKDHelpers::imageView(&source, &view)) {
        if (error) *error = tr("Failed to create an image view.");
        return false;
    }

    aicore_gkd_detect_request req{};
    req.struct_size = sizeof(req);
    const std::vector<std::string> prompts = [this] {
        std::vector<std::string> out;
        for (const QString& p : m_settings.kpsTexts) {
            out.push_back(p.toStdString());
        }
        return out;
    }();
    std::vector<const char*> promptPtrs;
    promptPtrs.reserve(prompts.size());
    for (const std::string& p : prompts) promptPtrs.push_back(p.c_str());

    // Multimodal support image (views borrow the QImages for the call).
    QImage support;
    aicore_image_view supportView{};
    std::vector<float> supportKps;
    if (!m_settings.supportImagePath.isEmpty()) {
        support = QImage(m_settings.supportImagePath);
        if (support.isNull()) {
            if (error)
                *error = tr("Failed to load support image: %1")
                                 .arg(m_settings.supportImagePath);
            return false;
        }
        supportKps.reserve(static_cast<size_t>(m_settings.supportKps.size()) *
                           2);
        for (const QPointF& p : m_settings.supportKps) {
            supportKps.push_back(static_cast<float>(p.x()));
            supportKps.push_back(static_cast<float>(p.y()));
        }
        if (!GKDHelpers::imageView(&support, &supportView)) {
            if (error) *error = tr("Failed to create a support image view.");
            return false;
        }
        req.support_image = &supportView;
        req.support_kps_xy = supportKps.data();
        req.n_support_kps = static_cast<int32_t>(supportKps.size() / 2);
    }

    float bboxStorage[4] = {0, 0, 0, 0};
    (void)bboxStorage;

    const int textCount = static_cast<int>(promptPtrs.size());
    if (textCount > 0 && req.n_support_kps > 0 &&
        textCount != req.n_support_kps) {
        if (error)
            *error = tr("Multimodal prompts require matching counts: %1 texts "
                        "vs %2 support keypoints")
                             .arg(textCount)
                             .arg(req.n_support_kps);
        return false;
    }
    req.kps_texts = textCount > 0 ? promptPtrs.data() : nullptr;
    req.n_kps_texts = textCount;
    if (textCount == 0 && req.n_support_kps == 0) {
        if (error)
            *error =
                    tr("No prompts: enter keypoint texts and/or a support "
                       "image with keypoints.");
        return false;
    }

    emit taskStage(tr("Running GKD inference..."), 30);
    const auto skeleton = GKDHelpers::parseSkeleton(m_settings.skeleton);
    sets->clear();
    // Serialize per device (see makeDeviceTaskLock above); constructed after
    // the CPU fallback resolved the final device, per attempt.
    auto deviceGuard = makeDeviceTaskLock(m_settings.device);
    // Multi-ROI batch (official --bbox_on_input_im semantics): 2+ xyxy
    // pairs in the ROI row go through one batched forward; otherwise the
    // legacy single-box / whole-image path runs.
    const int nRoi = m_settings.roiPoints.size() / 2;
    if (nRoi >= 2) {
        std::vector<float> boxes;
        boxes.reserve((size_t)nRoi * 4);
        for (int i = 0; i < nRoi; ++i) {
            const QPointF& a = m_settings.roiPoints.at(i * 2);
            const QPointF& b = m_settings.roiPoints.at(i * 2 + 1);
            boxes.insert(boxes.end(), {(float)a.x(), (float)a.y(), (float)b.x(),
                                       (float)b.y()});
        }
        aicore_cancel_scope_begin(m_cancelToken);
        const int rc = aicore_gkd_detect_image_multi(ctx, &view, &req,
                                                     boxes.data(), nRoi);
        aicore_cancel_scope_end(m_cancelToken);
        if (rc != 0) {
            const char* err = aicore_gkd_last_error(ctx);
            if (error)
                *error = tr("GKD inference failed: %1")
                                 .arg(err ? QString::fromUtf8(err)
                                          : tr("unknown error"));
            return false;
        }
        // A cancel request during the batched forward discards the batch.
        if (aicore_cancel_token_requested(m_cancelToken)) {
            if (error) *error = tr("Cancelled.");
            return false;
        }
        for (int roi = 0; roi < nRoi; ++roi) {
            GKDKeypointSet set;
            collectRoi(ctx, roi, m_settings.kpsTexts, skeleton,
                       tr("object %1").arg(roi + 1), 1.0f, m_settings.minScore,
                       &set);
            sets->append(set);
        }
        return true;
    }

    float singleBox[4] = {0, 0, 0, 0};
    if (m_settings.hasBbox) {
        std::memcpy(singleBox, m_settings.bbox, sizeof(singleBox));
        req.bbox_xyxy = singleBox;
    }
    aicore_cancel_scope_begin(m_cancelToken);
    const int rc = aicore_gkd_detect_image(ctx, &view, &req);
    aicore_cancel_scope_end(m_cancelToken);
    if (rc != 0) {
        const char* err = aicore_gkd_last_error(ctx);
        if (error)
            *error = tr("GKD inference failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        return false;
    }

    // Stage timings for the console log: the line the user reads to
    // locate where a slow run spent its time.
    {
        aicore_gkd_timings t{};
        if (aicore_gkd_last_timings(ctx, &t) == 0) {
            const GKDHelpers::GkdStageTimings s{
                    t.preprocess_ms, t.vision_ms, t.text_ms, t.prompt_prep_ms,
                    t.detect_ms,     t.decode_ms, t.e2e_ms};
            emit logMessage(tr("[GKD] inference timings: %1")
                                    .arg(GKDHelpers::formatGkdTimings(s, 1)));
        }
    }

    GKDKeypointSet set;
    collectRoi(ctx, 0, m_settings.kpsTexts, skeleton, QString(), 1.0f,
               m_settings.minScore, &set);
    sets->append(set);
    return true;
}

bool GKDWorker::ensureYoloContext(QString* error) {
    if (m_pendingYoloCtx) {
        // A stale handle from a failed load inside this run: drop it and
        // retry the load below (this worker owns that handle).
        if (m_ownsYoloCtx) {
            aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx));
            m_ownsYoloCtx = false;
        }
        m_pendingYoloCtx = nullptr;
    }
    // Reload key: everything that forces a new context. The class COUNT
    // fixes the text-input shape / graph topology (world_nc), so it stays
    // in the key and a count change reloads once; class TEXTS and the
    // confidence are runtime-switchable (per-call classes API + live
    // thresholds), so switching between same-count vocabularies keeps the
    // resident detector.
    const QString key =
            QStringList{m_settings.yoloModelPath, m_settings.yoloTextModelPath,
                        m_settings.device, QString::number(m_settings.threads),
                        QString::number(m_settings.objectClasses.size())}
                    .join(QLatin1Char('|'));
    if (m_settings.cache && m_settings.cache->yoloCtx &&
        m_settings.cache->yoloKey == key) {
        m_pendingYoloCtx = m_settings.cache->yoloCtx;
        m_ownsYoloCtx = false;
        // Thresholds ride on the context, not the reload key: a scene
        // with a different conf re-tunes the resident detector in place.
        aicore_yolo_set_detect_thresholds(
                static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx),
                m_settings.yoloConf, 0.6f, /*top_k*/ 0);
        emit logMessage(
                tr("[GKD] Reusing loaded detector: %1")
                        .arg(QFileInfo(m_settings.yoloModelPath).fileName()));
        return true;
    }

    aicore_yolo_options* yoloOpts = aicore_yolo_options_new();
    if (!yoloOpts) {
        if (error) *error = tr("Failed to allocate YOLO options.");
        return false;
    }
    aicore_yolo_options_set_device(yoloOpts,
                                   m_settings.device.toUtf8().constData());
    aicore_yolo_options_set_threads(yoloOpts, m_settings.threads);
    std::vector<std::string> classStorage;
    std::vector<const char*> classPtrs;
    // Reserve FIRST: push_back reallocation would invalidate the c_str
    // pointers already collected in classPtrs (dangling reads inside
    // aicore_yolo_options_set_classes for 2+ classes).
    classStorage.reserve(static_cast<size_t>(m_settings.objectClasses.size()));
    for (const QString& c : m_settings.objectClasses) {
        classStorage.push_back(c.toStdString());
        classPtrs.push_back(classStorage.back().c_str());
    }
    aicore_yolo_options_set_classes(yoloOpts, classPtrs.data(),
                                    static_cast<int32_t>(classPtrs.size()));
    aicore_yolo_options_set_conf_thres(yoloOpts, m_settings.yoloConf);
    // Class-aware NMS at 0.6 (backend default 0.7): measured on the demo
    // scenes, near-duplicate wide boxes overlap at IoU ~0.64 and both
    // survive 0.7 once the confidence cut drops — each survivor becomes a
    // full GKD pass and a second overlapping keypoint set in the result.
    aicore_yolo_options_set_iou_thres(yoloOpts, 0.6f);
    if (m_settings.yoloTextModelPath.isEmpty()) {
        if (error)
            *error =
                    tr("Multi-object mode needs a text-encoder GGUF (CLIP) "
                       "— pick one in the Text encoder row.");
        return false;
    }
    aicore_yolo_options_set_text_model(
            yoloOpts, m_settings.yoloTextModelPath.toUtf8().constData());
    m_pendingYoloCtx = aicore_yolo_load_opts(
            m_settings.yoloModelPath.toUtf8().constData(), yoloOpts);
    aicore_yolo_options_free(yoloOpts);
    auto* yoloCtx = static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx);
    if (!yoloCtx || aicore_yolo_is_ready(yoloCtx) != 1) {
        const char* err = yoloCtx ? aicore_yolo_last_error(yoloCtx)
                                  : "context allocation failed";
        if (error)
            *error = tr("YOLO-World model load failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        m_ownsYoloCtx = m_pendingYoloCtx != nullptr;
        return false;
    }
    if (m_settings.cache) {
        if (m_settings.cache->yoloCtx) {
            m_settings.cache->retiredYolo.push_back(m_settings.cache->yoloCtx);
        }
        m_settings.cache->yoloCtx = m_pendingYoloCtx;
        m_settings.cache->yoloKey = key;
        m_settings.cache->yoloClasses = m_settings.objectClasses;
        m_ownsYoloCtx = false;  // the cache owns it now
    } else {
        m_ownsYoloCtx = true;
    }
    return true;
}

bool GKDWorker::detectMultiObject(QVector<GKDKeypointSet>* sets,
                                  QString* error) {
    if (!ensureGkdContext(error)) return false;
    if (m_settings.yoloModelPath.isEmpty() ||
        m_settings.objectClasses.isEmpty()) {
        if (error)
            *error =
                    tr("Multi-object mode needs object classes and a "
                       "YOLO-World model.");
        return false;
    }

    // ---- stage 1: open-vocabulary detection (existing yolo task) ----
    emit taskStage(tr("Running YOLO-World detection..."), 10);
    if (!ensureYoloContext(error)) return false;
    auto* yoloCtx = static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx);

    QImage source(m_settings.inputPath);
    if (source.isNull()) {
        if (error)
            *error = tr("Failed to load image: %1").arg(m_settings.inputPath);
        return false;
    }
    aicore_image_view view{};
    if (!GKDHelpers::imageView(&source, &view)) {
        if (error) *error = tr("Failed to create an image view.");
        return false;
    }
    // Vocabulary switch on the resident detector: the per-call classes
    // API re-queues the text embedding (process-cached, so repeated
    // scene switches are free) instead of reloading the 168-op model.
    int detectRc;
    const bool classesSwitched =
            m_settings.cache && m_settings.cache->yoloCtx == yoloCtx &&
            m_settings.cache->yoloClasses != m_settings.objectClasses;
    if (classesSwitched) {
        // Storage must outlive the call: QString::toUtf8 temporaries
        // would dangle the c_str pointers (same trap as the load-path
        // classStorage.reserve note below).
        std::vector<QByteArray> classStorage;
        std::vector<const char*> classPtrs;
        classStorage.reserve(
                static_cast<size_t>(m_settings.objectClasses.size()));
        for (const QString& c : m_settings.objectClasses) {
            classStorage.push_back(c.toUtf8());
            classPtrs.push_back(classStorage.back().constData());
        }
        detectRc = aicore_yolo_detect_image_with_classes(
                yoloCtx, &view, classPtrs.data(),
                static_cast<int32_t>(classPtrs.size()));
        if (detectRc == 0) {
            m_settings.cache->yoloClasses = m_settings.objectClasses;
        }
    } else {
        detectRc = aicore_yolo_detect_image(yoloCtx, &view);
    }
    if (detectRc != 0) {
        const char* err = aicore_yolo_last_error(yoloCtx);
        if (error)
            *error = tr("YOLO-World detection failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        return false;
    }
    {
        aicore_yolo_timings t{};
        if (aicore_yolo_last_timings(yoloCtx, &t) == 0) {
            const GKDHelpers::YoloStageTimings s{t.preprocess_ms,
                                                 t.inference_ms,
                                                 t.postprocess_ms, t.e2e_ms};
            emit logMessage(tr("[GKD] YOLO-World timings: %1")
                                    .arg(GKDHelpers::formatYoloTimings(s)));
        }
    }

    // ---- stage 2: one batched GKD forward over every detection box (the
    // box IS the GKD coordinate system, exactly like the official
    // multi-object pipeline's N_bbox batch) ----
    const int detCount = aicore_yolo_detection_count(yoloCtx);
    emit logMessage(tr("[GKD] YOLO-World found %1 object(s); running batched "
                       "GKD...")
                            .arg(detCount));
    aicore_gkd_ctx* ctx = static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx);
    if (detCount <= 0) {
        emit logMessage(tr("[GKD] 0 keypoint(s) on 0 object(s)"));
        return true;
    }
    std::vector<float> boxes;
    std::vector<float> detScores;
    std::vector<QString> detLabels;
    boxes.reserve((size_t)detCount * 4);
    for (int i = 0; i < detCount; ++i) {
        const aicore_yolo_detection det = aicore_yolo_detection_at(yoloCtx, i);
        boxes.insert(boxes.end(), {det.x1, det.y1, det.x2, det.y2});
        detScores.push_back(det.score);
        const char* className = aicore_yolo_detection_class_name(yoloCtx, i);
        detLabels.push_back(className ? QString::fromUtf8(className)
                                      : tr("object %1").arg(i));
    }

    aicore_gkd_detect_request req{};
    req.struct_size = sizeof(req);
    const std::vector<std::string> prompts = [this] {
        std::vector<std::string> out;
        for (const QString& p : m_settings.kpsTexts)
            out.push_back(p.toStdString());
        return out;
    }();
    std::vector<const char*> promptPtrs;
    for (const std::string& p : prompts) promptPtrs.push_back(p.c_str());
    req.kps_texts = promptPtrs.empty() ? nullptr : promptPtrs.data();
    req.n_kps_texts = static_cast<int32_t>(promptPtrs.size());
    if (req.n_kps_texts == 0) {
        if (error) *error = tr("Multi-object mode requires keypoint texts.");
        return false;
    }

    emit taskStage(tr("GKD on %1 object(s)...").arg(detCount), 50);
    auto deviceGuard = makeDeviceTaskLock(m_settings.device);
    aicore_cancel_scope_begin(m_cancelToken);
    const int rc = aicore_gkd_detect_image_multi(ctx, &view, &req, boxes.data(),
                                                 detCount);
    aicore_cancel_scope_end(m_cancelToken);
    if (rc != 0) {
        const char* err = aicore_gkd_last_error(ctx);
        if (error)
            *error = tr("GKD inference failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        return false;
    }
    // A cancel request during the batched forward discards the batch.
    if (aicore_cancel_token_requested(m_cancelToken)) {
        if (error) *error = tr("Cancelled.");
        return false;
    }
    {
        aicore_gkd_timings t{};
        if (aicore_gkd_last_timings(ctx, &t) == 0) {
            const GKDHelpers::GkdStageTimings s{
                    t.preprocess_ms, t.vision_ms, t.text_ms, t.prompt_prep_ms,
                    t.detect_ms,     t.decode_ms, t.e2e_ms};
            emit logMessage(
                    tr("[GKD] batched GKD timings: %1")
                            .arg(GKDHelpers::formatGkdTimings(s, detCount)));
        }
    }
    const auto skeleton = GKDHelpers::parseSkeleton(m_settings.skeleton);
    for (int roi = 0; roi < detCount; ++roi) {
        GKDKeypointSet set;
        collectRoi(ctx, roi, m_settings.kpsTexts, skeleton, detLabels[roi],
                   detScores[roi], m_settings.minScore, &set);
        sets->append(set);
    }
    return true;
}

bool GKDWorker::runInference() {
    if (aicore_gkd_warmup_backend(m_settings.device.toUtf8().constData()) !=
        0) {
        if (aicore_is_gpu_device(m_settings.device.toUtf8().constData())) {
            m_settings.device = QStringLiteral("cpu");
            emit logMessage(tr(
                    "[GKD] GPU backend unavailable — using CPU for this run."));
        }
    }

    QVector<GKDKeypointSet> sets;
    QString error;
    bool ok = m_settings.multiObject ? detectMultiObject(&sets, &error)
                                     : detectSingleObject(&sets, &error);
    // VRAM admission failures are actionable: the GKD model (and for
    // multi-object the detector too) does not fit next to the resident
    // models on this device. Retry once on CPU instead of failing the run
    // (same policy as the warmup fallback above). The release is required:
    // a failed load leaves a non-ready context handle cached.
    if (!ok && m_settings.device != QLatin1String("cpu") &&
        error.contains(QLatin1String("headroom insufficient"))) {
        emit logMessage(tr(
                "[GKD] GPU memory is short — retrying on CPU for this run."));
        releaseContextOnMainThread();
        m_settings.device = QStringLiteral("cpu");
        sets.clear();
        error.clear();
        ok = m_settings.multiObject ? detectMultiObject(&sets, &error)
                                    : detectSingleObject(&sets, &error);
    }
    if (!ok) {
        if (!error.isEmpty()) emit logMessage(tr("[GKD] %1").arg(error));
        return false;
    }

    auto* ctx = static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx);
    GKDRunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.mode = m_settings.multiObject
                          ? QStringLiteral("multi-object")
                          : (!m_settings.supportImagePath.isEmpty() &&
                                             !m_settings.kpsTexts.isEmpty()
                                     ? QStringLiteral("multimodal")
                                     : (!m_settings.supportImagePath.isEmpty()
                                                ? QStringLiteral("visual")
                                                : QStringLiteral("text")));
    result.sets = sets;
    for (const GKDKeypointSet& set : sets) {
        result.totalKeypoints += set.keypoints.size();
    }
    // Official visualize semantics: only points above the display cut
    // are shown; the COCO export still carries the full list.
    for (const GKDKeypointSet& set : sets) {
        for (const GKDKeypoint& kp : set.keypoints) {
            if (kp.score >= m_settings.minScore) ++result.shownKeypoints;
        }
    }
    fillTimings(ctx, &result);
    result.resolvedDevice = QString::fromUtf8(aicore_gkd_context_device(ctx));
    result.modelPath = m_settings.modelPath;
    if (char* json = aicore_gkd_info_json(ctx)) {
        result.infoJson = QByteArray(json);
        aicore_gkd_free_buffer(json);
    }

    QImage source(m_settings.inputPath);
    result.imageWidth = source.width();
    result.imageHeight = source.height();
    result.kpsTexts = m_settings.kpsTexts;
    result.skeleton = m_settings.skeleton;
    result.renderedImage = GKDHelpers::renderResult(
            source, sets, m_settings.minScore, m_settings.pointLabels);
    emit taskStage(tr("Done"), 100);
    emit logMessage(tr("[GKD] %1 keypoint(s) on %2 object(s) (mode=%3, %4 ms)")
                            .arg(result.totalKeypoints)
                            .arg(sets.size())
                            .arg(result.mode)
                            .arg(result.totalRuntimeMs, 0, 'f', 1));
    emit resultReady(result);
    return true;
}

#endif  // AICore_ENABLED
