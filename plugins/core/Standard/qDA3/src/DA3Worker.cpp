// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DA3Worker.h"

#include <QFile>
#include <QFileInfo>
#include <algorithm>

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#endif

#include <algorithm>

DA3Worker::DA3Worker(const DA3Dialog::Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    static bool registered = false;
    if (!registered) {
        qRegisterMetaType<DA3DepthResult>("DA3DepthResult");
        qRegisterMetaType<DA3ReconResult>("DA3ReconResult");
        registered = true;
    }
}

void DA3Worker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] DA3 not enabled at build time.");
    emit taskFinished(false);
    return;
#else
    bool ok = false;
    emit progressUpdate(0, 100);
    switch (m_settings.mode) {
        case DA3Dialog::Mode::DepthSingle:
            ok = runDepthSingle();
            break;
        case DA3Dialog::Mode::DepthPose:
            ok = runDepthPose();
            break;
        case DA3Dialog::Mode::DepthMultiView:
            ok = runDepthMultiView();
            break;
        case DA3Dialog::Mode::Reconstruct:
            ok = runReconstruct();
            break;
        case DA3Dialog::Mode::ExportGLB:
            ok = runExportGLB();
            break;
        case DA3Dialog::Mode::ExportCOLMAP:
            ok = runExportCOLMAP();
            break;
        case DA3Dialog::Mode::Quantize:
            ok = runQuantize();
            break;
        case DA3Dialog::Mode::ModelInfo:
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

DA3Worker::CtxGuard::~CtxGuard() {
    if (ctx) {
        aicore_depth_free(ctx);
        ctx = nullptr;
    }
}

DA3Worker::CtxGuard DA3Worker::loadModel() {
    CtxGuard g;
    const QString modelPath = m_settings.modelPath;
    if (modelPath.isEmpty()) {
        emit logMessage("[Error] No GGUF model path configured.");
        return g;
    }
    if (!QFile::exists(modelPath)) {
        emit logMessage(
                QString("[Error] Model file not found: %1").arg(modelPath));
        return g;
    }
    if (QFileInfo(modelPath).size() < 1024) {
        emit logMessage(
                QString("[Error] Model file looks invalid (too small): %1")
                        .arg(modelPath));
        return g;
    }

    emit progressUpdate(5, 100);
    const QByteArray device = m_settings.device.trimmed().isEmpty()
                                      ? QByteArray("auto")
                                      : m_settings.device.trimmed().toUtf8();
    if (!m_settings.device.isEmpty() &&
        m_settings.device.trimmed().toLower() != QLatin1String("auto")) {
        emit logMessage(
                QString("[DA3] Using device: %1").arg(m_settings.device));
    } else {
        emit logMessage(
                QString("[DA3] Using device: auto (%1)")
                        .arg(QString::fromUtf8(aicore_auto_device_order())));
    }
    if (!m_settings.metricModelPath.isEmpty()) {
        if (!QFile::exists(m_settings.metricModelPath)) {
            emit logMessage(QString("[Error] Metric model file not found: %1")
                                    .arg(m_settings.metricModelPath));
            return g;
        }
        emit logMessage("[DA3] Loading nested model (anyview + metric)...");
        g.ctx = aicore_depth_load_nested_device(
                modelPath.toStdString().c_str(),
                m_settings.metricModelPath.toStdString().c_str(),
                m_settings.threads, device.constData());
    } else {
        emit logMessage("[DA3] Loading model: " + modelPath);
        g.ctx = aicore_depth_load_device(modelPath.toStdString().c_str(),
                                         m_settings.threads,
                                         device.constData());
    }
    if (!g.ctx) {
        emit logMessage(QString("[Error] Failed to load model (GGUF parse or "
                                "GPU offload failed): %1")
                                .arg(modelPath));
        return g;
    }
    emit progressUpdate(10, 100);
    return g;
}

bool DA3Worker::runDepthSingle() {
    if (m_settings.inputPaths.isEmpty()) {
        emit logMessage("[Error] No input image selected.");
        return false;
    }

    auto guard = loadModel();
    if (!guard) return false;

    int total = m_settings.inputPaths.size();
    int okCount = 0;
    for (int i = 0; i < total; ++i) {
        if (isInterruptionRequested()) {
            emit logMessage("[DA3] Cancelled.");
            break;
        }
        emit progressUpdate(10 + (i * 80) / std::max(total, 1), 100);
        emit logMessage("[DA3] Processing: " + m_settings.inputPaths[i]);

        int h = 0, w = 0;
        float* depth = aicore_depth_depth_path(
                guard.ctx, m_settings.inputPaths[i].toStdString().c_str(), &h,
                &w);
        if (!depth) {
            const char* err = aicore_depth_last_error(guard.ctx);
            emit logMessage(QString("[Error] Depth estimation failed for: %1%2")
                                    .arg(m_settings.inputPaths[i])
                                    .arg(err && err[0]
                                                 ? QString(" — %1").arg(err)
                                                 : QString()));
            continue;
        }
        ++okCount;

        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName();
        res.width = w;
        res.height = h;
        res.depth.resize(h * w);
        std::copy(depth, depth + h * w, res.depth.begin());
        res.hasPose = false;
        aicore_depth_free_floats(depth);

        float dmin = *std::min_element(res.depth.begin(), res.depth.end());
        float dmax = *std::max_element(res.depth.begin(), res.depth.end());
        emit logMessage(QString("[DA3] Depth %1x%2 min=%3 max=%4")
                                .arg(w)
                                .arg(h)
                                .arg(dmin, 0, 'f', 4)
                                .arg(dmax, 0, 'f', 4));

        emit depthResultReady(res);
    }

    emit progressUpdate(okCount > 0 ? 100 : 10, 100);
    if (okCount == 0) {
        emit logMessage("[Error] No depth maps produced.");
        return false;
    }
    emit logMessage(QString("[DA3] Depth estimation complete (%1/%2).")
                            .arg(okCount)
                            .arg(total));
    return true;
}

bool DA3Worker::runDepthPose() {
    if (m_settings.inputPaths.isEmpty()) {
        emit logMessage("[Error] No input image selected.");
        return false;
    }

    auto guard = loadModel();
    if (!guard) return false;

    int total = m_settings.inputPaths.size();
    int okCount = 0;
    for (int i = 0; i < total; ++i) {
        if (isInterruptionRequested()) break;
        emit progressUpdate(10 + (i * 80) / std::max(total, 1), 100);

        int h = 0, w = 0, is_metric = 0;
        float* depth_ptr = nullptr;
        float* conf_ptr = nullptr;
        float* sky_ptr = nullptr;
        float ext[12] = {}, intr[9] = {};

        int ret = aicore_depth_depth_dense(
                guard.ctx, m_settings.inputPaths[i].toStdString().c_str(), &h,
                &w, &depth_ptr, &conf_ptr, &sky_ptr, ext, intr, &is_metric);
        if (ret != 0 || !depth_ptr) {
            const char* err = aicore_depth_last_error(guard.ctx);
            emit logMessage(QString("[Error] Depth+pose failed for: %1%2")
                                    .arg(m_settings.inputPaths[i])
                                    .arg(err && err[0]
                                                 ? QString(" — %1").arg(err)
                                                 : QString()));
            if (depth_ptr) aicore_depth_free_floats(depth_ptr);
            if (conf_ptr) aicore_depth_free_floats(conf_ptr);
            if (sky_ptr) aicore_depth_free_floats(sky_ptr);
            continue;
        }
        ++okCount;

        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName();
        res.width = w;
        res.height = h;
        res.depth.resize(h * w);
        std::copy(depth_ptr, depth_ptr + h * w, res.depth.begin());
        if (conf_ptr) {
            res.confidence.resize(h * w);
            std::copy(conf_ptr, conf_ptr + h * w, res.confidence.begin());
        }
        res.hasPose = true;
        std::copy(ext, ext + 12, res.extrinsics);
        std::copy(intr, intr + 9, res.intrinsics);

        emit logMessage(QString("[DA3] %1: %2x%3 fx=%4 fy=%5")
                                .arg(res.sourceName)
                                .arg(w)
                                .arg(h)
                                .arg(intr[0], 0, 'f', 2)
                                .arg(intr[4], 0, 'f', 2));

        aicore_depth_free_floats(depth_ptr);
        if (conf_ptr) aicore_depth_free_floats(conf_ptr);
        if (sky_ptr) aicore_depth_free_floats(sky_ptr);

        emit depthResultReady(res);
    }

    emit progressUpdate(okCount > 0 ? 100 : 10, 100);
    if (okCount == 0) {
        emit logMessage("[Error] No depth+pose results produced.");
        return false;
    }
    emit logMessage(QString("[DA3] Depth + pose estimation complete (%1/%2).")
                            .arg(okCount)
                            .arg(total));
    return true;
}

bool DA3Worker::runDepthMultiView() {
    if (m_settings.inputPaths.size() < 2) {
        emit logMessage("[Error] Multi-view requires at least 2 images.");
        return false;
    }
    emit logMessage(QString("[DA3] Multi-view: %1 images")
                            .arg(m_settings.inputPaths.size()));

    auto guard = loadModel();
    if (!guard) return false;

    int n = m_settings.inputPaths.size();
    std::vector<const char*> cpaths(n);
    std::vector<std::string> paths(n);
    for (int i = 0; i < n; ++i) {
        paths[i] = m_settings.inputPaths[i].toStdString();
        cpaths[i] = paths[i].c_str();
    }

    int h = 0, w = 0, out_n = 0;
    std::vector<float> ext(n * 12), intr(n * 9);
    float* depth =
            aicore_depth_depth_pose_multi(guard.ctx, cpaths.data(), n, &h, &w,
                                          &out_n, ext.data(), intr.data());
    if (!depth) {
        emit logMessage("[Error] Multi-view failed.");
        return false;
    }

    emit logMessage(QString("[DA3] Multi-view: %1 views, %2x%3")
                            .arg(out_n)
                            .arg(w)
                            .arg(h));

    for (int i = 0; i < out_n && i < n; ++i) {
        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName() + "_mv";
        res.width = w;
        res.height = h;
        res.depth.resize(h * w);
        std::copy(depth + i * h * w, depth + (i + 1) * h * w,
                  res.depth.begin());
        res.hasPose = true;
        std::copy(ext.data() + i * 12, ext.data() + (i + 1) * 12,
                  res.extrinsics);
        std::copy(intr.data() + i * 9, intr.data() + (i + 1) * 9,
                  res.intrinsics);
        emit depthResultReady(res);
    }

    aicore_depth_free_floats(depth);
    emit logMessage("[DA3] Multi-view complete.");
    return true;
}

bool DA3Worker::runReconstruct() {
    if (m_settings.inputPaths.isEmpty()) {
        emit logMessage("[Error] No input image selected.");
        return false;
    }
    emit logMessage("[DA3] Reconstructing 3D Gaussians...");

    int H = 0;
    int W = 0;
    int N = 0;
    float* means = nullptr;
    float* scales = nullptr;
    float* harmonics = nullptr;
    float* opacities = nullptr;
    const int ret = aicore_depth_reconstruct_path(
            m_settings.modelPath.toStdString().c_str(), m_settings.threads,
            m_settings.inputPaths[0].toStdString().c_str(), &H, &W, &N, &means,
            &scales, &harmonics, &opacities);
    if (ret != 0) {
        emit logMessage("[Error] Reconstruction failed.");
        return false;
    }

    emit logMessage(QString("[DA3] Reconstructed %1 gaussians (%2x%3)")
                            .arg(N)
                            .arg(W)
                            .arg(H));

    DA3ReconResult res;
    res.sourceName = QFileInfo(m_settings.inputPaths[0]).baseName();
    res.count = N;
    if (N > 0 && means && scales && harmonics && opacities) {
        res.positions.resize(N * 3);
        res.colors.resize(N * 3);
        res.scales.resize(N * 3);
        res.opacities.resize(N);

        for (int i = 0; i < N; ++i) {
            res.positions[i * 3 + 0] = means[i * 3 + 0];
            res.positions[i * 3 + 1] = means[i * 3 + 1];
            res.positions[i * 3 + 2] = means[i * 3 + 2];
            if (harmonics) {
                const int shBase = i * 3 * 9;
                res.colors[i * 3 + 0] = harmonics[shBase + 0];
                res.colors[i * 3 + 1] = harmonics[shBase + 1];
                res.colors[i * 3 + 2] = harmonics[shBase + 2];
            }
            res.scales[i * 3 + 0] = scales[i * 3 + 0];
            res.scales[i * 3 + 1] = scales[i * 3 + 1];
            res.scales[i * 3 + 2] = scales[i * 3 + 2];
            res.opacities[i] = opacities[i];
        }
    }

    aicore_depth_free_floats(means);
    aicore_depth_free_floats(scales);
    aicore_depth_free_floats(harmonics);
    aicore_depth_free_floats(opacities);

    emit reconResultReady(res);
    emit logMessage("[DA3] Reconstruction complete.");
    return true;
}

bool DA3Worker::runExportGLB() {
    if (m_settings.inputPaths.isEmpty() || m_settings.outputDir.isEmpty()) {
        emit logMessage("[Error] Input image and output directory required.");
        return false;
    }
    emit logMessage("[DA3] Exporting GLB...");

    auto guard = loadModel();
    if (!guard) return false;

    QString outPath = m_settings.outputDir + "/" +
                      QFileInfo(m_settings.inputPaths[0]).baseName() + ".glb";
    int ret = aicore_depth_export_glb(
            guard.ctx, m_settings.inputPaths[0].toStdString().c_str(),
            outPath.toStdString().c_str());

    if (ret == 0) {
        emit logMessage("[DA3] GLB exported: " + outPath);
        return true;
    }
    emit logMessage("[Error] GLB export failed.");
    return false;
}

bool DA3Worker::runExportCOLMAP() {
    if (m_settings.inputPaths.isEmpty() || m_settings.outputDir.isEmpty()) {
        emit logMessage("[Error] Input image and output directory required.");
        return false;
    }
    emit logMessage("[DA3] Exporting COLMAP model...");

    auto guard = loadModel();
    if (!guard) return false;

    int ret = aicore_depth_export_colmap(
            guard.ctx, m_settings.inputPaths[0].toStdString().c_str(),
            m_settings.outputDir.toStdString().c_str(),
            m_settings.colmapBinary ? 1 : 0);

    if (ret == 0) {
        emit logMessage(
                QString("[DA3] COLMAP model exported (%1) to: %2")
                        .arg(m_settings.colmapBinary ? "binary" : "text")
                        .arg(m_settings.outputDir));
        return true;
    }
    emit logMessage("[Error] COLMAP export failed.");
    return false;
}

bool DA3Worker::runQuantize() {
    if (m_settings.quantInputPath.isEmpty() ||
        m_settings.quantOutputPath.isEmpty()) {
        emit logMessage(
                "[Error] Quantization input and output paths required.");
        return false;
    }
    emit logMessage(QString("[DA3] Quantizing: %1 -> %2 (%3)")
                            .arg(m_settings.quantInputPath)
                            .arg(m_settings.quantOutputPath)
                            .arg(m_settings.quantType));

    const int ret = aicore_depth_quantize_gguf(
            m_settings.quantInputPath.toStdString().c_str(),
            m_settings.quantOutputPath.toStdString().c_str(),
            m_settings.quantType.toStdString().c_str());
    if (ret == 0) {
        emit logMessage("[DA3] Quantization complete: " +
                        m_settings.quantOutputPath);
        return true;
    }
    emit logMessage("[Error] Quantization failed.");
    return false;
}

bool DA3Worker::runModelInfo() {
    auto guard = loadModel();
    if (!guard) return false;

    char* info = aicore_depth_info_json(guard.ctx);
    if (info) {
        emit modelInfoReady(QString::fromUtf8(info));
        aicore_depth_free_string(info);
        return true;
    }
    emit logMessage("[DA3] No info available.");
    return false;
}

#else  // !AICore_ENABLED

DA3Worker::CtxGuard::~CtxGuard() {}
DA3Worker::CtxGuard DA3Worker::loadModel() { return CtxGuard{}; }
bool DA3Worker::runDepthSingle() { return false; }
bool DA3Worker::runDepthPose() { return false; }
bool DA3Worker::runDepthMultiView() { return false; }
bool DA3Worker::runReconstruct() { return false; }
bool DA3Worker::runExportGLB() { return false; }
bool DA3Worker::runExportCOLMAP() { return false; }
bool DA3Worker::runQuantize() { return false; }
bool DA3Worker::runModelInfo() { return false; }

#endif  // AICore_ENABLED
