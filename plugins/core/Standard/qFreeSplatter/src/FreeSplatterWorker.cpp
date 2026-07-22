// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FreeSplatterWorker.h"

#include <QDir>
#include <QFileInfo>

#ifdef AICore_ENABLED
#include "aicore/gaussian_capi.h"
#endif

FreeSplatterWorker::FreeSplatterWorker(const Settings& settings,
                                       QObject* parent)
    : QThread(parent), m_settings(settings) {
    static bool registered = false;
    if (!registered) {
        qRegisterMetaType<FreeSplatterResult>("FreeSplatterResult");
        registered = true;
    }
}

void FreeSplatterWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] FreeSplatter not enabled at build time.");
    emit taskFinished(false);
    return;
#else
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
    emit logMessage("[FS] Loading model: " + m_settings.modelPath);

    aicore_gaussian_options* opts = aicore_gaussian_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_gaussian_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
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

    emit logMessage(
            QString("[FS] Model: %1x%2, %3 gaussian channels, SH degree %4")
                    .arg(geom.image_width)
                    .arg(geom.image_height)
                    .arg(geom.gaussian_channels)
                    .arg(geom.sh_degree));

    const int n = m_settings.inputPaths.size();
    emit progressUpdate(25, 100);
    const QString devLabel = m_settings.device.isEmpty()
                                     ? QStringLiteral("auto")
                                     : m_settings.device;
    emit logMessage(
            QString("[FS] [3/4] Running inference on %1 image(s) [%2]...")
                    .arg(n)
                    .arg(devLabel));

    std::vector<std::string> paths(n);
    std::vector<const char*> cpaths(n);
    for (int i = 0; i < n; ++i) {
        paths[i] = m_settings.inputPaths[i].toStdString();
        cpaths[i] = paths[i].c_str();
    }

    float* gaussians = nullptr;
    size_t n_out = 0;
    int ret = aicore_gaussian_run_paths(ctx, cpaths.data(), n, &gaussians,
                                        &n_out);
    if (ret != 0 || !gaussians) {
        const char* err = aicore_gaussian_last_error(ctx);
        emit logMessage(QString("[Error] Inference failed: %1")
                                .arg(err ? err : "unknown"));
        if (gaussians) {
            aicore_gaussian_free_floats(gaussians);
        }
        stashContext(ctx);
        return false;
    }

    emit progressUpdate(75, 100);
    emit logMessage("[FS] [4/4] Building result for DB display...");

    const int H = geom.image_height;
    const int W = geom.image_width;
    const int gc = geom.gaussian_channels;

    emit logMessage(
            QString("[FS] Inference complete: %1 gaussians (%2 views, %3x%4)")
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
    result.gaussians = QVector<float>(gaussians, gaussians + n_out);

    if (m_settings.estimatePoses && n >= 2) {
        emit logMessage("[FS] Estimating camera poses...");
        result.cam2world.resize(n * 16);
        float focal = 0.0f;
        ret = aicore_gaussian_estimate_poses(gaussians, n, H, W, gc,
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

    aicore_gaussian_free_floats(gaussians);

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
        aicore_gaussian_free_string(info);
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
