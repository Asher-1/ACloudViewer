// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// LightGlue worker — native feature extraction + AICore GGML matching.

#include "LightGlueWorker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>
#include <cmath>
#include <future>

#ifdef AICore_ENABLED
#include <QJsonDocument>
#include <QJsonObject>
#include <cstring>
#include <vector>

#include "aicore/aliked_capi.h"
#include "aicore/backend_capi.h"
#include "aicore/eloftr_capi.h"
#include "aicore/lightglue_capi.h"
#include "feature_extractor.h"
#endif

LightGlueWorker::LightGlueWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    static bool registered = false;
    if (!registered) {
        qRegisterMetaType<LightGlueRunResult>("LightGlueRunResult");
        registered = true;
    }
}

void LightGlueWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_lightglue_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
    if (m_pendingElOftrCtx) {
        aicore_eloftr_free(m_pendingElOftrCtx);
        m_pendingElOftrCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED

namespace {

QString formatDeviceLog(const LightGlueWorker::Settings& settings) {
    const QString req = settings.device.trimmed();
    if (!req.isEmpty() &&
        req.compare(QLatin1String("auto"), Qt::CaseInsensitive) != 0) {
        return QStringLiteral("[LG] Using device: %1").arg(req);
    }
    return QStringLiteral("[LG] Using device: auto (%1)")
            .arg(QString::fromUtf8(aicore_auto_device_order()));
}

QString formatResolvedDeviceLog(aicore_lightglue_ctx* ctx) {
    if (!ctx) return {};
    char* info = aicore_lightglue_info_json(ctx);
    if (!info) return {};
    const QJsonObject obj = QJsonDocument::fromJson(QByteArray(info)).object();
    aicore_lightglue_free_string(info);
    const QString resolved = obj.value(QStringLiteral("device")).toString();
    if (resolved.isEmpty()) return {};
    return QStringLiteral("[LG] ggml backend ready on device: %1")
            .arg(resolved);
}

QString resolve_aliked_extractor_gguf(const QString& matcher_model_path) {
    QFileInfo matcher(matcher_model_path);
    QString stem = matcher.fileName();
    stem.replace(QStringLiteral("aliked-lightglue"),
                 QStringLiteral("aliked-n16rot"));
    char* cache = aicore_aliked_model_cache_dir();
    QString base;
    if (cache) {
        base = QString::fromUtf8(cache);
        aicore_aliked_free_string(cache);
    } else {
        base = QDir::homePath() +
               QStringLiteral("/cloudViewer_data/extract/aliked_models");
    }
    const QString cached = QDir(base).filePath(stem);
    if (QFileInfo(cached).isFile()) {
        return cached;
    }
    const QString sibling = matcher.absoluteDir().filePath(stem);
    if (QFileInfo(sibling).isFile()) {
        return sibling;
    }
    return cached;
}

bool extract_feature_pair(const LightGlueWorker::Settings& settings,
                          lightglue_plugin::OwnedFeatures* f0,
                          lightglue_plugin::OwnedFeatures* f1,
                          QString* log) {
    if (settings.inputPaths.size() != 2) {
        return false;
    }
    const QString p0 = settings.inputPaths[0];
    const QString p1 = settings.inputPaths[1];
    std::string err;

    if (settings.matcherType != 1) {
        const QString extractor =
                resolve_aliked_extractor_gguf(settings.modelPath);
        if (log) {
            *log = QStringLiteral("[LG] ALIKED extractor: %1").arg(extractor);
        }
        if (!lightglue_plugin::extract_aliked_ggml(
                    p0, extractor, settings.device, settings.maxKeypoints,
                    settings.maxResize, settings.threads, f0, &err)) {
            if (log) *log = QString::fromStdString(err);
            return false;
        }
        if (!lightglue_plugin::extract_aliked_ggml(
                    p1, extractor, settings.device, settings.maxKeypoints,
                    settings.maxResize, settings.threads, f1, &err)) {
            if (log) *log = QString::fromStdString(err);
            return false;
        }
        return true;
    }

    if (!lightglue_plugin::extract_sift_opencv(p0, settings.maxKeypoints,
                                               settings.maxResize, f0, &err)) {
        if (log) *log = QString::fromStdString(err);
        return false;
    }
    if (!lightglue_plugin::extract_sift_opencv(p1, settings.maxKeypoints,
                                               settings.maxResize, f1, &err)) {
        if (log) *log = QString::fromStdString(err);
        return false;
    }
    return true;
}

QVector<QPointF> keypoints_to_qt(const aicore_lightglue_features& f) {
    QVector<QPointF> out;
    if (!f.keypoints || f.n_keypoints <= 0) return out;
    out.reserve(f.n_keypoints);
    for (int32_t i = 0; i < f.n_keypoints; ++i) {
        out.append(QPointF(f.keypoints[i].x, f.keypoints[i].y));
    }
    return out;
}

bool load_gray_same_size(const QString& p0,
                         const QString& p1,
                         std::vector<uint8_t>* g0,
                         std::vector<uint8_t>* g1,
                         int* width,
                         int* height,
                         QString* log) {
    const QImage img0 = lightglue_plugin::load_oriented_qimage(p0);
    const QImage img1 = lightglue_plugin::load_oriented_qimage(p1);
    if (img0.isNull() || img1.isNull()) {
        if (log) *log = QStringLiteral("failed to load input image(s)");
        return false;
    }
    QImage gray0 = img0.convertToFormat(QImage::Format_Grayscale8);
    QImage gray1 = img1.convertToFormat(QImage::Format_Grayscale8);
    if (gray0.size() != gray1.size()) {
        gray1 = gray1.scaled(gray0.size(), Qt::IgnoreAspectRatio,
                             Qt::SmoothTransformation);
    }
    *width = gray0.width();
    *height = gray0.height();
    g0->resize(static_cast<size_t>(*width) * *height);
    g1->resize(static_cast<size_t>(*width) * *height);
    std::memcpy(g0->data(), gray0.constBits(), g0->size());
    std::memcpy(g1->data(), gray1.constBits(), g1->size());
    return true;
}

}  // namespace

bool LightGlueWorker::runModelInfo() {
    emit logMessage(formatDeviceLog(m_settings));
    emit logMessage("[LG] Loading model: " + m_settings.modelPath);
    emit progressUpdate(20, 100);

    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_lightglue_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_lightglue_options_set_threads(opts, m_settings.threads);
    aicore_lightglue_options_set_matcher_type(opts, m_settings.matcherType);

    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_lightglue_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to allocate LightGlue context.");
        return false;
    }
    if (const char* err = aicore_lightglue_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        m_pendingCtx = ctx;
        return false;
    }
    emit logMessage(formatResolvedDeviceLog(ctx));

    char* json = aicore_lightglue_info_json(ctx);
    if (json) {
        emit modelInfoReady(QString::fromUtf8(json));
        aicore_lightglue_free_string(json);
    }
    m_pendingCtx = ctx;
    emit progressUpdate(100, 100);
    return true;
}

bool LightGlueWorker::runMatch() {
    if (m_settings.inputPaths.size() != 2) {
        emit logMessage("[Error] LightGlue requires exactly two input images.");
        return false;
    }

    QElapsedTimer timer;
    timer.start();

    emit progressUpdate(5, 100);
    emit logMessage(formatDeviceLog(m_settings));
    emit logMessage("[LG] Loading model: " + m_settings.modelPath);

    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_lightglue_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_lightglue_options_set_threads(opts, m_settings.threads);
    aicore_lightglue_options_set_min_score(opts, m_settings.minScore);
    aicore_lightglue_options_set_matcher_type(opts, m_settings.matcherType);

    aicore_lightglue_ctx* ctx = aicore_lightglue_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_lightglue_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to allocate LightGlue context.");
        return false;
    }
    if (const char* err = aicore_lightglue_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        m_pendingCtx = ctx;
        return false;
    }
    emit logMessage(formatResolvedDeviceLog(ctx));

    emit progressUpdate(20, 100);
    emit logMessage("[LG] Extracting RootSIFT features (OpenCV)...");

    lightglue_plugin::OwnedFeatures f0;
    lightglue_plugin::OwnedFeatures f1;
    QString extractLog;
    if (!extract_feature_pair(m_settings, &f0, &f1, &extractLog)) {
        emit logMessage(QString("[Error] Feature extraction failed: %1")
                                .arg(extractLog));
        m_pendingCtx = ctx;
        return false;
    }

    emit progressUpdate(45, 100);
    emit logMessage(QString("[LG] Matching %1 x %2 keypoints (GGML)...")
                            .arg(f0.view.n_keypoints)
                            .arg(f1.view.n_keypoints));

    aicore_lightglue_match* matches = nullptr;
    int32_t n_matches = 0;
    int matchRet = 0;
    auto matchFut = std::async(std::launch::async, [&]() {
        return aicore_lightglue_run_match(ctx, &f0.view, &f1.view, &matches,
                                          &n_matches);
    });
    {
        QElapsedTimer elapsed;
        elapsed.start();
        const int64_t expectedMs = 15000;
        while (matchFut.wait_for(std::chrono::milliseconds(1500)) !=
               std::future_status::ready) {
            const double t =
                    elapsed.elapsed() / static_cast<double>(expectedMs);
            const int pct =
                    45 + static_cast<int>(50.0 * (1.0 - std::exp(-2.5 * t)));
            emit progressUpdate(qMin(pct, 94), 100);
        }
    }
    matchRet = matchFut.get();
    if (matchRet != 0) {
        const char* matchErr = aicore_lightglue_last_error(ctx);
        emit logMessage(QString("[Error] Matching failed: %1")
                                .arg(matchErr ? matchErr : "unknown"));
        m_pendingCtx = ctx;
        return false;
    }

    LightGlueRunResult result;
    result.imagePath0 = m_settings.inputPaths[0];
    result.imagePath1 = m_settings.inputPaths[1];
    result.imageName0 =
            result.imagePath0.startsWith("db://")
                    ? result.imagePath0.mid(5)
                    : QFileInfo(result.imagePath0).completeBaseName();
    result.imageName1 =
            result.imagePath1.startsWith("db://")
                    ? result.imagePath1.mid(5)
                    : QFileInfo(result.imagePath1).completeBaseName();
    result.sourceName = result.imageName0 + "_x_" + result.imageName1;
    result.nKeypoints0 = f0.view.n_keypoints;
    result.nKeypoints1 = f1.view.n_keypoints;
    result.imageWidth0 = f0.view.image_width;
    result.imageHeight0 = f0.view.image_height;
    result.imageWidth1 = f1.view.image_width;
    result.imageHeight1 = f1.view.image_height;
    result.keypoints0 = keypoints_to_qt(f0.view);
    result.keypoints1 = keypoints_to_qt(f1.view);
    result.runtimeMs = timer.elapsed();
    result.matches.reserve(n_matches);
    for (int32_t i = 0; i < n_matches; ++i) {
        result.matches.append(
                {matches[i].idx1, matches[i].idx2, matches[i].score});
    }

    aicore_lightglue_free_matches(matches);
    m_pendingCtx = ctx;

    emit progressUpdate(100, 100);
    emit logMessage(QString("[LG] Found %1 mutual matches in %2 ms.")
                            .arg(n_matches)
                            .arg(result.runtimeMs, 0, 'f', 1));
    emit resultReady(result);
    return true;
}

bool LightGlueWorker::runMatchElOftr() {
    if (m_settings.inputPaths.size() != 2) {
        emit logMessage("[Error] ELoFTR requires exactly two input images.");
        return false;
    }

    QElapsedTimer timer;
    timer.start();

    emit progressUpdate(5, 100);
    emit logMessage(formatDeviceLog(m_settings));
    emit logMessage("[ELoFTR] Loading model: " + m_settings.modelPath);

    aicore_eloftr_options* opts = aicore_eloftr_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_eloftr_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_eloftr_options_set_threads(opts, m_settings.threads);

    aicore_eloftr_ctx* ctx = aicore_eloftr_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_eloftr_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to allocate ELoFTR context.");
        return false;
    }
    if (const char* err = aicore_eloftr_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        m_pendingElOftrCtx = ctx;
        return false;
    }

    emit progressUpdate(20, 100);
    std::vector<uint8_t> gray0;
    std::vector<uint8_t> gray1;
    int width = 0;
    int height = 0;
    QString loadLog;
    if (!load_gray_same_size(m_settings.inputPaths[0], m_settings.inputPaths[1],
                             &gray0, &gray1, &width, &height, &loadLog)) {
        emit logMessage(QString("[Error] %1").arg(loadLog));
        m_pendingElOftrCtx = ctx;
        return false;
    }

    emit logMessage(QString("[ELoFTR] Matching %1×%2 grayscale pair (GGML)...")
                            .arg(width)
                            .arg(height));
    emit progressUpdate(45, 100);

    aicore_eloftr_match* matches = nullptr;
    int32_t n_matches = 0;
    if (aicore_eloftr_match_gray(ctx, gray0.data(), gray1.data(), width, height,
                                 width, &matches, &n_matches) != 0) {
        const char* matchErr = aicore_eloftr_last_error(ctx);
        emit logMessage(QString("[Error] ELoFTR matching failed: %1")
                                .arg(matchErr ? matchErr : "unknown"));
        m_pendingElOftrCtx = ctx;
        return false;
    }

    LightGlueRunResult result;
    result.imagePath0 = m_settings.inputPaths[0];
    result.imagePath1 = m_settings.inputPaths[1];
    result.imageName0 =
            result.imagePath0.startsWith("db://")
                    ? result.imagePath0.mid(5)
                    : QFileInfo(result.imagePath0).completeBaseName();
    result.imageName1 =
            result.imagePath1.startsWith("db://")
                    ? result.imagePath1.mid(5)
                    : QFileInfo(result.imagePath1).completeBaseName();
    result.sourceName = result.imageName0 + "_x_" + result.imageName1;
    result.imageWidth0 = width;
    result.imageHeight0 = height;
    result.imageWidth1 = width;
    result.imageHeight1 = height;
    result.keypoints0.reserve(n_matches);
    result.keypoints1.reserve(n_matches);
    result.runtimeMs = timer.elapsed();
    result.matches.reserve(n_matches);
    for (int32_t i = 0; i < n_matches; ++i) {
        result.keypoints0.append(QPointF(matches[i].x0, matches[i].y0));
        result.keypoints1.append(QPointF(matches[i].x1, matches[i].y1));
        if (matches[i].score >= m_settings.minScore) {
            result.matches.append({i, i, matches[i].score});
        }
    }
    result.nKeypoints0 = result.keypoints0.size();
    result.nKeypoints1 = result.keypoints1.size();

    aicore_eloftr_free_matches(matches);
    m_pendingElOftrCtx = ctx;

    emit progressUpdate(100, 100);
    emit logMessage(QString("[ELoFTR] Found %1 matches in %2 ms.")
                            .arg(result.matches.size())
                            .arg(result.runtimeMs, 0, 'f', 1));
    emit resultReady(result);
    return true;
}

#endif

void LightGlueWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] LightGlue not enabled at build time.");
    emit taskFinished(false);
    return;
#else
    if (isInterruptionRequested()) {
        emit taskFinished(false);
        return;
    }
    bool ok = false;
    switch (m_settings.mode) {
        case Mode::Match:
            if (m_settings.pipelineType == 1) {
                ok = runMatchElOftr();
            } else {
                ok = runMatch();
            }
            break;
        case Mode::ModelInfo:
            ok = runModelInfo();
            break;
    }
    emit taskFinished(ok);
#endif
}
