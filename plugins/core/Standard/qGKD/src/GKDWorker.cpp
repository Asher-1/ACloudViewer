// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDWorker.h"

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
#include "aicore/yolo_capi.h"
#endif

namespace {

#ifdef AICore_ENABLED

/** Serializes inference per device for the duration of one task. Same
 *  pattern as the other AICore plugin workers. */
class DeviceTaskGuard {
public:
    explicit DeviceTaskGuard(const QString& device) {}
    bool isLocked() const { return true; }
};

/** Converts one context-owned GKD result into a plugin keypoint set. */
void collectSet(const aicore_gkd_ctx* ctx,
                const QString& label,
                bool hasBox,
                const float* box,
                float minScore,
                GKDKeypointSet* set) {
    set->label = label;
    set->hasBox = hasBox;
    if (hasBox && box != nullptr) {
        set->x1 = box[0];
        set->y1 = box[1];
        set->x2 = box[2];
        set->y2 = box[3];
    }
    const int n = aicore_gkd_result_keypoint_count(ctx);
    set->keypoints.reserve(n);
    for (int i = 0; i < n; ++i) {
        const aicore_gkd_keypoint kp = aicore_gkd_result_keypoint_at(ctx, i);
        if (kp.score < minScore) continue;
        GKDKeypoint out;
        out.x = kp.x;
        out.y = kp.y;
        out.score = kp.score;
        const char* prompt = aicore_gkd_result_prompt_at(ctx, i);
        if (prompt != nullptr) out.prompt = QString::fromUtf8(prompt);
        set->keypoints.append(out);
    }
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
#ifdef AICore_ENABLED
    if (m_pendingGkdCtx) {
        aicore_gkd_free(static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx));
        m_pendingGkdCtx = nullptr;
    }
    if (m_pendingYoloCtx) {
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_pendingYoloCtx));
        m_pendingYoloCtx = nullptr;
    }
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
        return false;
    }
    emit logMessage(
            tr("[GKD] Model loaded: %1 (device=%2, threads=%3, input=%4)")
                    .arg(QFileInfo(m_settings.modelPath).fileName(),
                         aicore_gkd_context_device(ctx))
                    .arg(aicore_gkd_context_threads(ctx))
                    .arg(aicore_gkd_context_image_size(ctx)));
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
    if (m_settings.hasBbox) {
        std::memcpy(bboxStorage, m_settings.bbox, sizeof(bboxStorage));
        req.bbox_xyxy = bboxStorage;
    }

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

    GKDKeypointSet set;
    collectSet(ctx, QString(), m_settings.hasBbox, bboxStorage,
               m_settings.minScore, &set);
    sets->clear();
    sets->append(set);
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
    for (const QString& c : m_settings.objectClasses) {
        classStorage.push_back(c.toStdString());
        classPtrs.push_back(classStorage.back().c_str());
    }
    aicore_yolo_options_set_classes(yoloOpts, classPtrs.data(),
                                    static_cast<int32_t>(classPtrs.size()));
    aicore_yolo_options_set_conf_thres(yoloOpts, m_settings.yoloConf);
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
        return false;
    }

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
    if (aicore_yolo_detect_image(yoloCtx, &view) != 0) {
        const char* err = aicore_yolo_last_error(yoloCtx);
        if (error)
            *error = tr("YOLO-World detection failed: %1")
                             .arg(err ? QString::fromUtf8(err)
                                      : tr("unknown error"));
        return false;
    }

    // ---- stage 2: GKD per detection box (the box IS the GKD coordinate
    // system, exactly like the official multi-object pipeline) ----
    const int detCount = aicore_yolo_detection_count(yoloCtx);
    emit logMessage(tr("[GKD] YOLO-World found %1 object(s); running GKD per "
                       "box...")
                            .arg(detCount));
    aicore_gkd_ctx* ctx = static_cast<aicore_gkd_ctx*>(m_pendingGkdCtx);
    for (int i = 0; i < detCount; ++i) {
        if (aicore_cancel_token_requested(m_cancelToken)) {
            if (error) *error = tr("Cancelled.");
            return false;
        }
        const aicore_yolo_detection det = aicore_yolo_detection_at(yoloCtx, i);
        const char* className = aicore_yolo_detection_class_name(yoloCtx, i);
        const QString label = className ? QString::fromUtf8(className)
                                        : tr("object %1").arg(i);

        aicore_gkd_detect_request req{};
        req.struct_size = sizeof(req);
        float box[4] = {det.x1, det.y1, det.x2, det.y2};
        req.bbox_xyxy = box;
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
            if (error)
                *error = tr("Multi-object mode requires keypoint texts for "
                            "class '%1'.")
                                 .arg(label);
            return false;
        }

        emit taskStage(
                tr("GKD on '%1' (%2/%3)...")
                        .arg(label)
                        .arg(i + 1)
                        .arg(detCount),
                20 + static_cast<int>(70 * (i + 1) / std::max(1, detCount)));
        aicore_cancel_scope_begin(m_cancelToken);
        const int rc = aicore_gkd_detect_image(ctx, &view, &req);
        aicore_cancel_scope_end(m_cancelToken);
        if (rc != 0) {
            const char* err = aicore_gkd_last_error(ctx);
            if (error)
                *error = tr("GKD inference failed on '%1': %2")
                                 .arg(label)
                                 .arg(err ? QString::fromUtf8(err)
                                          : tr("unknown error"));
            return false;
        }
        GKDKeypointSet set;
        collectSet(ctx, label, true, box, m_settings.minScore, &set);
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
    const bool ok = m_settings.multiObject ? detectMultiObject(&sets, &error)
                                           : detectSingleObject(&sets, &error);
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
    result.shownKeypoints = result.totalKeypoints;
    fillTimings(ctx, &result);
    result.resolvedDevice = QString::fromUtf8(aicore_gkd_context_device(ctx));
    result.modelPath = m_settings.modelPath;
    if (char* json = aicore_gkd_info_json(ctx)) {
        result.infoJson = QByteArray(json);
        aicore_gkd_free_buffer(json);
    }

    QImage source(m_settings.inputPath);
    result.renderedImage =
            GKDHelpers::renderResult(source, sets, m_settings.minScore);
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
