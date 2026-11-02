// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "QuerySplatWorker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>

#include <algorithm>
#include <cstring>

#include <aicore/image_view.h>
#include <aicore/inference_log.h>
#include <aicore/querysplat_capi.h>

namespace {

inline QRgb turboColor(float t) {
    // Compact turbo-like ramp (matches the MapAnything depth view palette).
    t = std::min(std::max(t, 0.0f), 1.0f);
    const int r = int(255 * std::min(std::max(1.5f - std::abs(4.0f * t - 3.0f), 0.0f), 1.0f));
    const int g = int(255 * std::min(std::max(1.5f - std::abs(4.0f * t - 2.0f), 0.0f), 1.0f));
    const int b = int(255 * std::min(std::max(1.5f - std::abs(4.0f * t - 1.0f), 0.0f), 1.0f));
    return qRgb(r, g, b);
}

}  // namespace

QuerySplatWorker::QuerySplatWorker(QObject* parent) : QThread(parent) {}

QuerySplatWorker::~QuerySplatWorker() {
    // onTaskFinished may deleteLater() the worker from a queued slot while
    // run() is still returning; ~QThread aborts on a running thread and the
    // AICore context below must outlive every run() access.
    if (isRunning()) {
        requestInterruption();
        wait();
    }
}

void QuerySplatWorker::setSettings(const Settings& settings) {
    m_settings = settings;
}

bool QuerySplatWorker::loadContext() {
    aicore_querysplat_options* opts = aicore_querysplat_options_new();
    if (!opts) {
        emit logMessage(QStringLiteral("[Error] Out of memory creating the "
                                       "QuerySplat options."));
        return false;
    }
    aicore_querysplat_options_set_device(
            opts, m_settings.device.toUtf8().constData());
    aicore_querysplat_options_set_threads(opts, m_settings.threads);
    aicore_querysplat_options_set_f32_accuracy(opts, 1);  // torch parity
    if (m_settings.imageSize > 0) {
        aicore_querysplat_options_set_image_size(opts, m_settings.imageSize,
                                                 m_settings.imageSize);
    }
    aicore_querysplat_options_set_tto_lr(opts, m_settings.ttoLr);
    aicore_querysplat_options_set_tto_lpips_weight(
            opts, m_settings.ttoLpipsWeight);
    aicore_querysplat_options_set_opacity_threshold(
            opts, m_settings.opacityThreshold);

    const QByteArray dec = m_settings.decoderPath.toUtf8();
    const QByteArray enc =
            m_settings.encoderPath.isEmpty()
                    ? QByteArray()
                    : m_settings.encoderPath.toUtf8();
    m_ctx = aicore_querysplat_load_opts(
            enc.isEmpty() ? nullptr : enc.constData(), dec.constData(), opts);
    aicore_querysplat_options_free(opts);
    if (!m_ctx || !aicore_querysplat_is_ready(m_ctx)) {
        const char* err = m_ctx ? aicore_querysplat_last_error(m_ctx)
                                : nullptr;
        emit logMessage(QStringLiteral("[Error] Model load failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : QStringLiteral("unknown")));
        if (m_ctx) {
            aicore_querysplat_free(m_ctx);
            m_ctx = nullptr;
        }
        return false;
    }
    return true;
}

QImage QuerySplatWorker::colorizeDepth(const float* depth, const float* conf,
                                       int w, int h, int view) const {
    if (!depth) return QImage();
    const size_t plane = size_t(w) * h;
    const float* d = depth + size_t(view) * plane;
    const float* c = conf ? conf + size_t(view) * plane : nullptr;
    double dmin = 1e30, dmax = -1e30;
    for (int i = 0; i < w * h; ++i) {
        dmin = std::min(dmin, double(d[i]));
        dmax = std::max(dmax, double(d[i]));
    }
    const double span = std::max(dmax - dmin, 1e-9);
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            float t = float((d[i] - dmin) / span);
            if (c) {
                // confidence alpha-fade: low-confidence pixels desaturate
                const float a = std::min(std::max(c[i] - 1.0f, 0.0f), 1.0f);
                t = t * (0.35f + 0.65f * a);
            }
            img.setPixel(x, y, turboColor(t));
        }
    }
    return img;
}

QImage QuerySplatWorker::rgb8ToImage(const unsigned char* rgb8, int w, int h) {
    if (!rgb8 || w <= 0 || h <= 0) return QImage();
    QImage img(w, h, QImage::Format_RGB32);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned char* p = rgb8 + (size_t(y) * w + x) * 3;
            img.setPixel(x, y, qRgb(p[0], p[1], p[2]));
        }
    }
    return img;
}

void QuerySplatWorker::run() {
    const int total = 100;
    emit progressUpdate(0, total);
    emit stageUpdate(tr("Loading models"));

    if (!loadContext()) return;

    // staged decoded images -> aicore_image_view borrowed views
    struct Staged {
        std::vector<uint8_t> rgb;
        int32_t w = 0, h = 0;
    };
    std::vector<Staged> staged(m_settings.imagePaths.size());
    std::vector<aicore_image_view> views(m_settings.imagePaths.size());
    for (int i = 0; i < (int)m_settings.imagePaths.size(); ++i) {
        QImage img(m_settings.imagePaths[i]);
        if (img.isNull()) {
            emit logMessage(QStringLiteral("[Error] Cannot decode image: %1")
                                    .arg(m_settings.imagePaths[i]));
            aicore_querysplat_free(m_ctx);
            m_ctx = nullptr;
            return;
        }
        QImage rgb = img.convertToFormat(QImage::Format_RGB888);
        Staged& s = staged[i];
        s.w = rgb.width();
        s.h = rgb.height();
        s.rgb.resize(size_t(s.w) * s.h * 3);
        std::memcpy(s.rgb.data(), rgb.constBits(), s.rgb.size());
        views[i].data = s.rgb.data();
        views[i].width = s.w;
        views[i].height = s.h;
        views[i].row_stride_bytes = size_t(s.w) * 3;
        views[i].format = AICORE_IMAGE_RGB8;
    }

    emit stageUpdate(tr("Running inference"));
    QElapsedTimer inferTimer;
    inferTimer.start();
    const int rc = aicore_querysplat_run_images(
            m_ctx, views.data(), int32_t(views.size()));
    if (rc != 0) {
        const char* err = aicore_querysplat_last_error(m_ctx);
        emit logMessage(QStringLiteral("[Error] Inference failed (%1): %2")
                                .arg(rc)
                                .arg(err ? QString::fromUtf8(err)
                                         : QStringLiteral("unknown")));
        aicore_querysplat_free(m_ctx);
        m_ctx = nullptr;
        return;
    }

    QuerySplatResult result;
    result.sourceLabel =
            QFileInfo(m_settings.imagePaths[0]).dir().dirName();
    if (result.sourceLabel.isEmpty())
        result.sourceLabel = QStringLiteral("scene");
    result.decoderPath = m_settings.decoderPath;
    result.encoderPath = m_settings.encoderPath;
    result.viewCount = aicore_querysplat_result_view_count(m_ctx);
    result.width = aicore_querysplat_result_width(m_ctx);
    result.height = aicore_querysplat_result_height(m_ctx);
    result.shDegree = aicore_querysplat_result_sh_degree(m_ctx);
    result.resolvedDevice = [this] {
        if (char* j = aicore_querysplat_info_json(m_ctx)) {
            const QString s = QString::fromUtf8(j);
            aicore_querysplat_free_buffer(j);
            const int k = s.indexOf(QStringLiteral("\"device\":\""));
            if (k >= 0) {
                const int a = k + int(strlen("\"device\":\""));
                const int b = s.indexOf('"', a);
                if (b > a) return s.mid(a, b - a);
            }
        }
        return m_settings.device;
    }();

    // optional TTO (KV-memory optimization against the input views)
    if (m_settings.ttoEnabled && m_settings.ttoSteps > 0) {
        emit stageUpdate(tr("Test-time optimization"));
        aicore_querysplat_tto_step* losses = nullptr;
        int32_t n_losses = 0;
        const float* tto_g = nullptr;
        int64_t tto_n = 0;
        const int trc = aicore_querysplat_run_tto(
                m_ctx, m_settings.ttoSteps, &losses, &n_losses, &tto_g,
                &tto_n);
        if (trc != 0) {
            const char* err = aicore_querysplat_last_error(m_ctx);
            emit logMessage(
                    QStringLiteral("[Error] TTO failed (%1): %2")
                            .arg(trc)
                            .arg(err ? QString::fromUtf8(err)
                                     : QStringLiteral("unknown")));
            aicore_querysplat_free(m_ctx);
            m_ctx = nullptr;
            return;
        }
        for (int32_t i = 0; i < n_losses; ++i) {
            result.ttoSteps.push_back({losses[i].l1, losses[i].ssim,
                                       losses[i].lpips});
        }
        emit logMessage(
                QStringLiteral("[QS] TTO %1 steps, final L1=%2 SSIM=%3")
                        .arg(n_losses)
                        .arg(losses[n_losses - 1].l1, 0, 'f', 4)
                        .arg(losses[n_losses - 1].ssim, 0, 'f', 4));
    }

    // gaussian buffer ownership transfer (borrowed view -> shared_ptr with
    // the AICore free function; the context stays alive until the copies
    // below finish, then the result owns its own store)
    result.gaussianCount = aicore_querysplat_result_gaussian_count(m_ctx);
    result.gaussianStride = aicore_querysplat_result_gaussian_stride(m_ctx);
    {
        const float* g = aicore_querysplat_result_gaussians(m_ctx);
        const size_t bytes = size_t(result.gaussianCount) *
                             size_t(result.gaussianStride) * sizeof(float);
        auto* copy = new float[bytes / sizeof(float)];
        std::memcpy(copy, g, bytes);
        result.gaussians.reset(copy, [](float* p) { delete[] p; });
    }

    // depth rasters + cameras
    const int S = result.viewCount, H = result.height, W = result.width;
    {
        const float* w2c = aicore_querysplat_result_w2c(m_ctx);
        const float* c2w = aicore_querysplat_result_cam2world(m_ctx);
        const float* K = aicore_querysplat_result_intrinsics(m_ctx);
        if (c2w && K) {
            result.camPoses16.resize(size_t(S) * 16);
            std::memcpy(result.camPoses16.data(), c2w,
                        sizeof(float) * size_t(S) * 16);
            result.camIntrinsics.resize(size_t(S) * 4);
            std::memcpy(result.camIntrinsics.data(), K,
                        sizeof(float) * size_t(S) * 4);
        }
        (void)w2c;
    }

    emit progressUpdate(60, total);
    emit stageUpdate(tr("Rendering views"));
    if (m_settings.addDepthViews) {
        const float* depth = aicore_querysplat_result_depth(m_ctx);
        const float* conf = aicore_querysplat_result_depth_conf(m_ctx);
        for (int v = 0; v < S; ++v) {
            result.depthViews.push_back(
                    colorizeDepth(depth, conf, W, H, v));
        }
    }
    if (m_settings.addRenderViews) {
        const float* w2c = aicore_querysplat_result_w2c(m_ctx);
        const float* K = aicore_querysplat_result_intrinsics(m_ctx);
        if (w2c && K) {
            const float* rgb = nullptr;
            unsigned char* rgb8 = nullptr;
            if (aicore_querysplat_render_views(m_ctx, w2c, K, S, &rgb,
                                               &rgb8) == 0) {
                const size_t plane = size_t(W) * H;
                for (int v = 0; v < S; ++v) {
                    result.renderViews.push_back(rgb8ToImage(
                            rgb8 + size_t(v) * plane * 3, W, H));
                }
            } else {
                emit logMessage(
                        QStringLiteral("[Warning] Render failed "
                                       "(non-fatal)."));
            }
        }
    }

    emit progressUpdate(80, total);
    emit stageUpdate(tr("Depth point cloud"));
    if (m_settings.addDepthPointCloud) {
        const float* xyz = nullptr;
        unsigned char* col = nullptr;
        int32_t n = 0;
        if (aicore_querysplat_depth_pointcloud(m_ctx, 200000, &xyz, &col,
                                               &n) == 0 &&
            n > 0) {
            result.points.resize(size_t(n) * 3);
            std::memcpy(result.points.data(), xyz,
                        sizeof(float) * size_t(n) * 3);
            result.colors.resize(size_t(n) * 3);
            std::memcpy(result.colors.data(), col, size_t(n) * 3);
            result.pointCount = n;
        }
    }

    // PLY export (compatibility boundary; the DB cloud below is the typed
    // path). Written to the output dir or beside the first input.
    if (m_settings.exportPly) {
        const QString dir = m_settings.outputDir.isEmpty()
                                    ? QFileInfo(m_settings.imagePaths[0])
                                              .absolutePath()
                                    : m_settings.outputDir;
        QDir().mkpath(dir);
        result.plyPath = dir + QStringLiteral("/") + result.sourceLabel +
                         QStringLiteral("_gaussians.ply");
        if (aicore_querysplat_export_gaussians_ply(
                    m_ctx, result.plyPath.toUtf8().constData(),
                    m_settings.opacityThreshold) != 0) {
            emit logMessage(
                    QStringLiteral("[Warning] PLY export failed."));
            result.plyPath.clear();
        }
    }

    if (char* j = aicore_querysplat_info_json(m_ctx)) {
        result.resultJson = QByteArray(j);
        aicore_querysplat_free_buffer(j);
    }

    aicore_inference_log::log_inference_done(
            QStringLiteral("QS"), result.resolvedDevice,
            qint64(inferTimer.elapsed()),
            QStringLiteral("%1 gaussians, %2 views, %3x%4")
                    .arg(result.gaussianCount)
                    .arg(S)
                    .arg(W)
                    .arg(H));

    aicore_querysplat_free(m_ctx);
    m_ctx = nullptr;

    emit progressUpdate(100, total);
    emit resultReady(result);
    emit logMessage(QStringLiteral("[QS] Reconstruction complete."));
}
