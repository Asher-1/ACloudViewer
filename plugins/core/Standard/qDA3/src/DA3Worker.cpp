#include "DA3Worker.h"
#include <QFileInfo>

#ifdef DA3_ENABLED
#include "da_capi.h"
#include "engine.hpp"
#include "quantize.hpp"
#endif

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
#ifndef DA3_ENABLED
    emit logMessage("[Error] DA3 not enabled at build time.");
    emit taskFinished(false);
    return;
#else
    bool ok = false;
    switch (m_settings.mode) {
        case DA3Dialog::Mode::DepthSingle:    ok = runDepthSingle(); break;
        case DA3Dialog::Mode::DepthPose:      ok = runDepthPose(); break;
        case DA3Dialog::Mode::DepthMultiView: ok = runDepthMultiView(); break;
        case DA3Dialog::Mode::Reconstruct:    ok = runReconstruct(); break;
        case DA3Dialog::Mode::ExportGLB:      ok = runExportGLB(); break;
        case DA3Dialog::Mode::ExportCOLMAP:   ok = runExportCOLMAP(); break;
        case DA3Dialog::Mode::Quantize:       ok = runQuantize(); break;
        case DA3Dialog::Mode::ModelInfo:       ok = runModelInfo(); break;
        default: emit logMessage("[Error] Unknown mode."); break;
    }
    emit taskFinished(ok);
#endif
}

#ifdef DA3_ENABLED

DA3Worker::CtxGuard::~CtxGuard() {
    if (ctx) { da_capi_free(ctx); ctx = nullptr; }
}

DA3Worker::CtxGuard DA3Worker::loadModel() {
    CtxGuard g;
    if (!m_settings.metricModelPath.isEmpty()) {
        emit logMessage("[DA3] Loading nested model (anyview + metric)...");
        g.ctx = da_capi_load_nested(m_settings.modelPath.toStdString().c_str(),
                                    m_settings.metricModelPath.toStdString().c_str(),
                                    m_settings.threads);
    } else {
        emit logMessage("[DA3] Loading model: " + m_settings.modelPath);
        g.ctx = da_capi_load(m_settings.modelPath.toStdString().c_str(), m_settings.threads);
    }
    if (!g) emit logMessage("[Error] Failed to load model.");
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
    for (int i = 0; i < total; ++i) {
        if (isInterruptionRequested()) {
            emit logMessage("[DA3] Cancelled.");
            break;
        }
        emit progressUpdate(i, total);
        emit logMessage("[DA3] Processing: " + m_settings.inputPaths[i]);

        int h = 0, w = 0;
        float* depth = da_capi_depth_path(guard.ctx, m_settings.inputPaths[i].toStdString().c_str(), &h, &w);
        if (!depth) {
            emit logMessage("[Error] Depth estimation failed for: " + m_settings.inputPaths[i]);
            continue;
        }

        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName();
        res.width = w;
        res.height = h;
        res.depth = QVector<float>(depth, depth + h * w);
        res.hasPose = false;
        da_capi_free_floats(depth);

        float dmin = *std::min_element(res.depth.begin(), res.depth.end());
        float dmax = *std::max_element(res.depth.begin(), res.depth.end());
        emit logMessage(QString("[DA3] Depth %1x%2 min=%3 max=%4")
                        .arg(w).arg(h).arg(dmin, 0, 'f', 4).arg(dmax, 0, 'f', 4));

        emit depthResultReady(res);
    }

    emit progressUpdate(total, total);
    emit logMessage("[DA3] Depth estimation complete.");
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
    for (int i = 0; i < total; ++i) {
        if (isInterruptionRequested()) break;
        emit progressUpdate(i, total);

        int h = 0, w = 0, is_metric = 0;
        float* depth_ptr = nullptr;
        float* conf_ptr = nullptr;
        float* sky_ptr = nullptr;
        float ext[12] = {}, intr[9] = {};

        int ret = da_capi_depth_dense(guard.ctx, m_settings.inputPaths[i].toStdString().c_str(),
                                      &h, &w, &depth_ptr, &conf_ptr, &sky_ptr,
                                      ext, intr, &is_metric);
        if (ret != 0 || !depth_ptr) {
            emit logMessage("[Error] Depth+pose failed for: " + m_settings.inputPaths[i]);
            if (depth_ptr) da_capi_free_floats(depth_ptr);
            if (conf_ptr) da_capi_free_floats(conf_ptr);
            if (sky_ptr) da_capi_free_floats(sky_ptr);
            continue;
        }

        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName();
        res.width = w;
        res.height = h;
        res.depth = QVector<float>(depth_ptr, depth_ptr + h * w);
        if (conf_ptr)
            res.confidence = QVector<float>(conf_ptr, conf_ptr + h * w);
        res.hasPose = true;
        std::copy(ext, ext + 12, res.extrinsics);
        std::copy(intr, intr + 9, res.intrinsics);

        emit logMessage(QString("[DA3] %1: %2x%3 fx=%4 fy=%5")
                        .arg(res.sourceName).arg(w).arg(h)
                        .arg(intr[0], 0, 'f', 2).arg(intr[4], 0, 'f', 2));

        da_capi_free_floats(depth_ptr);
        if (conf_ptr) da_capi_free_floats(conf_ptr);
        if (sky_ptr) da_capi_free_floats(sky_ptr);

        emit depthResultReady(res);
    }

    emit progressUpdate(total, total);
    emit logMessage("[DA3] Depth + pose estimation complete.");
    return true;
}

bool DA3Worker::runDepthMultiView() {
    if (m_settings.inputPaths.size() < 2) {
        emit logMessage("[Error] Multi-view requires at least 2 images.");
        return false;
    }
    emit logMessage(QString("[DA3] Multi-view: %1 images").arg(m_settings.inputPaths.size()));

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
    float* depth = da_capi_depth_pose_multi(guard.ctx, cpaths.data(), n,
                                             &h, &w, &out_n,
                                             ext.data(), intr.data());
    if (!depth) {
        emit logMessage("[Error] Multi-view failed.");
        return false;
    }

    emit logMessage(QString("[DA3] Multi-view: %1 views, %2x%3").arg(out_n).arg(w).arg(h));

    for (int i = 0; i < out_n && i < n; ++i) {
        DA3DepthResult res;
        res.sourceName = QFileInfo(m_settings.inputPaths[i]).baseName() + "_mv";
        res.width = w;
        res.height = h;
        res.depth = QVector<float>(depth + i * h * w, depth + (i + 1) * h * w);
        res.hasPose = true;
        std::copy(ext.data() + i * 12, ext.data() + (i + 1) * 12, res.extrinsics);
        std::copy(intr.data() + i * 9, intr.data() + (i + 1) * 9, res.intrinsics);
        emit depthResultReady(res);
    }

    da_capi_free_floats(depth);
    emit logMessage("[DA3] Multi-view complete.");
    return true;
}

bool DA3Worker::runReconstruct() {
    if (m_settings.inputPaths.isEmpty()) {
        emit logMessage("[Error] No input image selected.");
        return false;
    }
    emit logMessage("[DA3] Reconstructing 3D Gaussians...");

    // Gaussian reconstruction only uses the anyview (GIANT) branch — skip
    // loading the metric model even if one is configured.
    std::unique_ptr<da::Engine> eng =
        da::Engine::load(m_settings.modelPath.toStdString(), m_settings.threads);
    if (!eng) {
        emit logMessage("[Error] Failed to load model.");
        return false;
    }

    da::Gaussians g;
    int H, W;
    if (!eng->reconstruct_path(m_settings.inputPaths[0].toStdString(), g, H, W)) {
        emit logMessage("[Error] Reconstruction failed.");
        return false;
    }

    emit logMessage(QString("[DA3] Reconstructed %1 gaussians (%2x%3)")
                    .arg(g.N).arg(W).arg(H));

    DA3ReconResult res;
    res.sourceName = QFileInfo(m_settings.inputPaths[0]).baseName();
    res.count = g.N;
    res.positions.resize(g.N * 3);
    res.colors.resize(g.N * 3);
    res.scales.resize(g.N * 3);
    res.opacities.resize(g.N);

    for (int i = 0; i < g.N; ++i) {
        res.positions[i * 3 + 0] = g.means[i * 3 + 0];
        res.positions[i * 3 + 1] = g.means[i * 3 + 1];
        res.positions[i * 3 + 2] = g.means[i * 3 + 2];
        if (g.harmonics.size() >= static_cast<size_t>((i + 1) * 3 * 9)) {
            res.colors[i * 3 + 0] = g.harmonics[i * 3 * 9 + 0];
            res.colors[i * 3 + 1] = g.harmonics[i * 3 * 9 + 1];
            res.colors[i * 3 + 2] = g.harmonics[i * 3 * 9 + 2];
        }
        res.scales[i * 3 + 0] = g.scales[i * 3 + 0];
        res.scales[i * 3 + 1] = g.scales[i * 3 + 1];
        res.scales[i * 3 + 2] = g.scales[i * 3 + 2];
        res.opacities[i] = g.opacities[i];
    }

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
    int ret = da_capi_export_glb(guard.ctx, m_settings.inputPaths[0].toStdString().c_str(),
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

    int ret = da_capi_export_colmap(guard.ctx, m_settings.inputPaths[0].toStdString().c_str(),
                                     m_settings.outputDir.toStdString().c_str(),
                                     m_settings.colmapBinary ? 1 : 0);

    if (ret == 0) {
        emit logMessage(QString("[DA3] COLMAP model exported (%1) to: %2")
                        .arg(m_settings.colmapBinary ? "binary" : "text")
                        .arg(m_settings.outputDir));
        return true;
    }
    emit logMessage("[Error] COLMAP export failed.");
    return false;
}

bool DA3Worker::runQuantize() {
    if (m_settings.quantInputPath.isEmpty() || m_settings.quantOutputPath.isEmpty()) {
        emit logMessage("[Error] Quantization input and output paths required.");
        return false;
    }
    emit logMessage(QString("[DA3] Quantizing: %1 -> %2 (%3)")
                    .arg(m_settings.quantInputPath)
                    .arg(m_settings.quantOutputPath)
                    .arg(m_settings.quantType));

    if (da::quantize_gguf(m_settings.quantInputPath.toStdString(),
                          m_settings.quantOutputPath.toStdString(),
                          m_settings.quantType.toStdString())) {
        emit logMessage("[DA3] Quantization complete: " + m_settings.quantOutputPath);
        return true;
    }
    emit logMessage("[Error] Quantization failed.");
    return false;
}

bool DA3Worker::runModelInfo() {
    auto guard = loadModel();
    if (!guard) return false;

    char* info = da_capi_info_json(guard.ctx);
    if (info) {
        emit modelInfoReady(QString::fromUtf8(info));
        da_capi_free_string(info);
        return true;
    }
    emit logMessage("[DA3] No info available.");
    return false;
}

#else  // !DA3_ENABLED

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

#endif  // DA3_ENABLED
