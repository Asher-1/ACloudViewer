// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FreeSplatterWorker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>
#include <QUuid>
#include <algorithm>
#include <cmath>

#ifdef AICore_ENABLED
#include "aicore/gaussian_capi.h"
#include "aicore/runtime_capi.h"
#endif
#include "aicore/inference_log.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

#ifdef AICore_ENABLED
class AICoreInferenceGuard {
public:
    AICoreInferenceGuard(aicore_cancel_token* token, const QString& device)
        : m_token(token) {
        m_locked = aicore_device_task_lock_cancelable(
                           device.toUtf8().constData(), m_token) == 0;
        if (!m_locked) return;
        aicore_cancel_scope_begin(m_token);
    }
    ~AICoreInferenceGuard() {
        if (!m_locked) return;
        aicore_cancel_scope_end(m_token);
        aicore_device_task_unlock();
    }

    bool locked() const { return m_locked; }

private:
    aicore_cancel_token* m_token = nullptr;
    bool m_locked = false;
};
#endif

class TemporaryDirGuard {
public:
    void setPath(const QString& path) { m_path = path; }
    ~TemporaryDirGuard() {
        if (!m_path.isEmpty()) QDir(m_path).removeRecursively();
    }

private:
    QString m_path;
};

}  // namespace

FreeSplatterWorker::FreeSplatterWorker(const Settings& settings,
                                       QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
    static bool registered = false;
    if (!registered) {
        qRegisterMetaType<FreeSplatterResult>("FreeSplatterResult");
        registered = true;
    }
}

FreeSplatterWorker::~FreeSplatterWorker() {
#ifdef AICore_ENABLED
    aicore_cancel_token_free(m_cancelToken);
#endif
}

void FreeSplatterWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    aicore_cancel_token_request(m_cancelToken);
#endif
    requestInterruption();
}

void FreeSplatterWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] FreeSplatter not enabled at build time.");
    emit taskFinished(false);
    return;
#else
    AICoreInferenceGuard inferenceGuard(m_cancelToken, m_settings.device);
    if (!inferenceGuard.locked()) {
        emit taskFinished(false);
        return;
    }
    bool ok = false;
    switch (m_settings.mode) {
        case Mode::Reconstruct:
            ok = runReconstruct();
            break;
        case Mode::ModelInfo:
            ok = runModelInfo();
            break;
        default:
            emit logMessage("[Error] Unknown mode.");
            break;
    }
    emit taskFinished(ok);
#endif
}

#ifdef AICore_ENABLED

void FreeSplatterWorker::stashContext(aicore_gaussian_ctx* ctx) {
    m_pendingCtx = ctx;
}

void FreeSplatterWorker::releaseContextOnMainThread() {
    if (m_pendingCtx) {
        aicore_gaussian_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
}

aicore_gaussian_ctx* FreeSplatterWorker::loadModel() {
    {
        QFileInfo fi(m_settings.modelPath);
        const double sizeMB = fi.size() / (1024.0 * 1024.0);
        emit logMessage(QString("[FS] Loading: %1 (%2 MB)")
                                .arg(fi.fileName())
                                .arg(sizeMB, 0, 'f', 1));
    }

    aicore_gaussian_options* opts = aicore_gaussian_options_new();
    if (!opts) {
        emit logMessage("[Error] Failed to allocate model options.");
        return nullptr;
    }
    if (!m_settings.device.isEmpty()) {
        aicore_gaussian_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_inference_log::log_device_request(QStringLiteral("FS"),
                                             m_settings.device);
    aicore_gaussian_options_set_threads(opts, m_settings.threads);

    const std::string modelPath = m_settings.modelPath.toStdString();
    aicore_gaussian_ctx* ctx =
            aicore_gaussian_load_opts(modelPath.c_str(), opts);
    aicore_gaussian_options_free(opts);

    if (!ctx) {
        emit logMessage(QString("[Error] Failed to load model: %1")
                                .arg("out of memory"));
        return nullptr;
    }
    if (const char* err = aicore_gaussian_last_error(ctx)) {
        emit logMessage(QString("[Error] Failed to load model: %1").arg(err));
        stashContext(ctx);
        return nullptr;
    }
    if (char* infoJ = aicore_gaussian_info_json(ctx)) {
        const QJsonObject mi =
                QJsonDocument::fromJson(QByteArray(infoJ)).object();
        aicore_gaussian_free_buffer(infoJ);
        const QString resolved = mi.value(QStringLiteral("device")).toString();
        aicore_inference_log::log_device_resolved(QStringLiteral("FS"),
                                                  resolved);
    }
    return ctx;
}

bool FreeSplatterWorker::runReconstruct() {
    if (m_settings.inputPaths.isEmpty()) {
        emit logMessage("[Error] No input images selected.");
        return false;
    }

    emit progressUpdate(5, 100);
    emit logMessage("[FS] [1/4] Loading model...");
    aicore_gaussian_ctx* ctx = loadModel();
    if (!ctx) {
        return false;
    }

    emit progressUpdate(15, 100);
    emit logMessage("[FS] [2/4] Preparing inference...");

    aicore_gaussian_geometry geom{};
    if (aicore_gaussian_geometry_of(ctx, &geom) != 0) {
        const char* err = aicore_gaussian_last_error(ctx);
        emit logMessage(QString("[Error] Failed to get model geometry: %1")
                                .arg(err ? err : "unknown"));
        stashContext(ctx);
        return false;
    }

    if (char* infoJ = aicore_gaussian_info_json(ctx)) {
        const QJsonObject mi =
                QJsonDocument::fromJson(QByteArray(infoJ)).object();
        const bool use2dgs = mi.value(QStringLiteral("use_2dgs")).toBool();
        const bool shRes = mi.value(QStringLiteral("sh_residual")).toBool();
        emit logMessage(QString("[FS] Model: %1x%2, %3ch (%4), SH%5 %6")
                                .arg(geom.image_width)
                                .arg(geom.image_height)
                                .arg(geom.gaussian_channels)
                                .arg(use2dgs ? "2DGS" : "3DGS")
                                .arg(geom.sh_degree)
                                .arg(shRes ? "+residual" : ""));
        aicore_gaussian_free_buffer(infoJ);
    } else {
        emit logMessage(
                QString("[FS] Model: %1x%2, %3 gaussian channels, SH degree %4")
                        .arg(geom.image_width)
                        .arg(geom.image_height)
                        .arg(geom.gaussian_channels)
                        .arg(geom.sh_degree));
    }

    QStringList effectivePaths = m_settings.inputPaths;
    QString bgTmpDir;
    TemporaryDirGuard bgTmpCleanup;

#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_settings.removeBackground) {
        emit logMessage("[FS] Removing backgrounds (GrabCut)...");
        bgTmpDir =
                QStandardPaths::writableLocation(QStandardPaths::TempLocation) +
                "/freesplatter_bg_" +
                QUuid::createUuid().toString(QUuid::Id128);
        QDir().mkpath(bgTmpDir);
        bgTmpCleanup.setPath(bgTmpDir);
        QStringList processed;
        for (int i = 0; i < effectivePaths.size(); ++i) {
            cv::Mat img = cv::imread(effectivePaths[i].toStdString());
            if (img.empty()) {
                processed.append(effectivePaths[i]);
                continue;
            }
            cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
            const int mx = std::max(1, img.cols / 10);
            const int my = std::max(1, img.rows / 10);
            cv::Rect roi(mx, my, img.cols - 2 * mx, img.rows - 2 * my);
            mask(roi).setTo(cv::Scalar(cv::GC_PR_FGD));
            cv::Mat bgModel, fgModel;
            cv::grabCut(img, mask, roi, bgModel, fgModel, 5,
                        cv::GC_INIT_WITH_MASK);
            cv::Mat fg = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
            cv::Mat result(img.size(), img.type(), cv::Scalar(255, 255, 255));
            img.copyTo(result, fg);
            const QString outPath =
                    bgTmpDir + "/" + QFileInfo(effectivePaths[i]).fileName();
            cv::imwrite(outPath.toStdString(), result);
            processed.append(outPath);
            emit logMessage(QString("[FS]   bg-removed %1/%2")
                                    .arg(i + 1)
                                    .arg(effectivePaths.size()));
        }
        effectivePaths = processed;
    }
#endif

    {
        const bool isObject =
                m_settings.modelPath.contains("object", Qt::CaseInsensitive);
        const bool is2dgs =
                m_settings.modelPath.contains("2dgs", Qt::CaseInsensitive);
        const int autoMax = isObject ? (is2dgs ? 24 : 16) : 2;
        const int maxViews =
                (m_settings.maxViews > 0) ? m_settings.maxViews : autoMax;
        if (effectivePaths.size() > maxViews) {
            const int hardMax = 64;
            const int cap = qMin(maxViews, hardMax);
            QStringList sampled;
            sampled.reserve(cap);
            for (int i = 0; i < cap; ++i) {
                int src = i * effectivePaths.size() / cap;
                sampled.append(effectivePaths[src]);
            }
            emit logMessage(QString("[FS] %1 input images exceed model limit "
                                    "\u2014 uniformly subsampled to %2 "
                                    "(cap %3, hard max %4).")
                                    .arg(effectivePaths.size())
                                    .arg(cap)
                                    .arg(maxViews)
                                    .arg(hardMax));
            effectivePaths = sampled;
        }
    }

    const int n = effectivePaths.size();
    emit progressUpdate(25, 100);
    QString devLabel = m_settings.device.isEmpty() ? QStringLiteral("auto")
                                                   : m_settings.device;
    if (char* info = aicore_gaussian_info_json(ctx)) {
        const QJsonObject modelInfo =
                QJsonDocument::fromJson(QByteArray(info)).object();
        const QString resolvedDevice =
                modelInfo.value(QStringLiteral("device")).toString();
        if (!resolvedDevice.isEmpty()) devLabel = resolvedDevice;
        aicore_gaussian_free_buffer(info);
    }
    emit logMessage(
            QString("[FS] [3/4] Running inference on %1 image(s) [%2]...")
                    .arg(n)
                    .arg(devLabel));

    std::vector<std::string> paths(n);
    std::vector<const char*> cpaths(n);
    for (int i = 0; i < n; ++i) {
        paths[i] = effectivePaths[i].toStdString();
        cpaths[i] = paths[i].c_str();
    }

    float* gaussians = nullptr;
    size_t n_out = 0;

    QElapsedTimer inferTimer;
    inferTimer.start();
    emit progressUpdate(30, 100);

    // Run on this worker thread only — ggml CUDA is not safe across std::async
    // threads and can hang when combined with other plugin GPU contexts.
    if (isInterruptionRequested()) {
        stashContext(ctx);
        return false;
    }
    int ret = aicore_gaussian_run_paths(ctx, cpaths.data(), n, &gaussians,
                                        &n_out);
    if (ret != 0 || !gaussians) {
        const char* err = aicore_gaussian_last_error(ctx);
        emit logMessage(QString("[Error] Inference failed: %1")
                                .arg(err ? err : "unknown"));
        if (gaussians) {
            aicore_gaussian_free_buffer(gaussians);
        }
        stashContext(ctx);
        return false;
    }

    emit progressUpdate(75, 100);
    emit logMessage("[FS] [4/4] Building result for DB display...");

    const int H = geom.image_height;
    const int W = geom.image_width;
    const int gc = geom.gaussian_channels;

    aicore_inference_log::log_inference_done(
            QStringLiteral("FS"), devLabel, inferTimer.elapsed(),
            QStringLiteral("%1 gaussians, %2 views, %3×%4")
                    .arg(n_out)
                    .arg(n)
                    .arg(W)
                    .arg(H));

    FreeSplatterResult result;
    result.sourceName = QFileInfo(m_settings.inputPaths[0]).baseName();
    result.nViews = n;
    result.height = H;
    result.width = W;
    result.gaussianChannels = gc;
    result.shDegree = geom.sh_degree;
    // Transfer ownership of the AICore output buffer directly — no second
    // full-size copy (~350 MB for a 24-view 2DGS object run). shared_ptr
    // refcounts across the queued signal; the buffer must NOT be freed here.
    result.gaussianCount = n_out;
    result.gaussians =
            std::shared_ptr<float>(gaussians, aicore_gaussian_free_buffer);
    result.resolvedDevice = devLabel;
    result.runtimeMs = inferTimer.elapsed();

    if (m_settings.estimatePoses && n >= 2) {
        emit logMessage("[FS] Estimating camera poses...");
        result.cam2world.resize(n * 16);
        float focal = 0.0f;
        aicore_gaussian_geometry geom{};
        geom.image_height = H;
        geom.image_width = W;
        geom.gaussian_channels = gc;
        ret = aicore_gaussian_estimate_poses(&geom, result.gaussians.get(), n,
                                             m_settings.opacityThreshold,
                                             result.cam2world.data(), &focal);
        if (ret == 0) {
            result.hasPoses = true;
            result.focal = focal;
            emit logMessage(QString("[FS] Pose estimation: focal=%1")
                                    .arg(focal, 0, 'f', 2));
        } else {
            emit logMessage("[Warning] Pose estimation failed (non-fatal).");
        }
    }

    emit progressUpdate(100, 100);
    emit resultReady(result);
    emit logMessage("[FS] Reconstruction complete.");

    stashContext(ctx);
    return true;
}

bool FreeSplatterWorker::runModelInfo() {
    emit progressUpdate(10, 100);
    emit logMessage("[FS] Loading model...");
    aicore_gaussian_ctx* ctx = loadModel();
    if (!ctx) {
        return false;
    }

    emit progressUpdate(50, 100);

    char* info = aicore_gaussian_info_json(ctx);
    if (info) {
        emit modelInfoReady(QString::fromUtf8(info));
        aicore_gaussian_free_buffer(info);
        emit progressUpdate(100, 100);
        stashContext(ctx);
        return true;
    }
    emit logMessage("[FS] No info available.");
    stashContext(ctx);
    return false;
}

#else  // !AICore_ENABLED

void FreeSplatterWorker::releaseContextOnMainThread() {}
void FreeSplatterWorker::stashContext(aicore_gaussian_ctx*) {}
aicore_gaussian_ctx* FreeSplatterWorker::loadModel() { return nullptr; }
bool FreeSplatterWorker::runReconstruct() { return false; }
bool FreeSplatterWorker::runModelInfo() { return false; }

#endif  // AICore_ENABLED
