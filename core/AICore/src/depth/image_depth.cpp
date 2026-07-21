// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QTemporaryFile>
#include <cstring>

#include "aicore/depth_capi.h"
#include "aicore/depth_image.h"

namespace aicore {
namespace depth {
namespace {

bool loadContext(const QString& model_path,
                 const QString& metric_model_path,
                 int n_threads,
                 aicore_depth_ctx*& ctx) {
    const int threads = n_threads > 0 ? n_threads : 1;
    ctx = metric_model_path.isEmpty()
                  ? aicore_depth_load(model_path.toUtf8().constData(), threads)
                  : aicore_depth_load_nested(
                            model_path.toUtf8().constData(),
                            metric_model_path.toUtf8().constData(), threads);
    return ctx != nullptr;
}

bool writeTempPng(const QImage& image, QTemporaryFile& tmp) {
    if (image.isNull()) return false;
    tmp.setFileTemplate(QDir::tempPath() + "/aicore_depth_image_XXXXXX.png");
    if (!tmp.open()) return false;
    if (!image.save(&tmp, "PNG")) return false;
    tmp.close();
    return true;
}

}  // namespace

bool ImageDepth::isAvailable() { return true; }

bool ImageDepth::estimateDepth(const QImage& image,
                               const QString& model_path,
                               int n_threads,
                               ImageDepthResult& out,
                               const QString& metric_model_path) {
    QTemporaryFile tmp;
    if (!writeTempPng(image, tmp)) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, ctx))
        return false;

    int h = 0, w = 0;
    float* depth = aicore_depth_depth_path(
            ctx, tmp.fileName().toUtf8().constData(), &h, &w);
    if (!depth) {
        aicore_depth_free(ctx);
        return false;
    }

    out.width = w;
    out.height = h;
    out.depth.assign(depth, depth + h * w);
    out.has_pose = false;
    aicore_depth_free_floats(depth);
    aicore_depth_free(ctx);
    return true;
}

bool ImageDepth::estimateDepthAndPose(const QImage& image,
                                      const QString& model_path,
                                      int n_threads,
                                      ImageDepthResult& out,
                                      const QString& metric_model_path) {
    QTemporaryFile tmp;
    if (!writeTempPng(image, tmp)) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, ctx))
        return false;

    int h = 0, w = 0, is_metric = 0;
    float* depth_ptr = nullptr;
    float* conf_ptr = nullptr;
    float* sky_ptr = nullptr;
    float ext[12] = {}, intr[9] = {};

    const int ret = aicore_depth_depth_dense(
            ctx, tmp.fileName().toUtf8().constData(), &h, &w, &depth_ptr,
            &conf_ptr, &sky_ptr, ext, intr, &is_metric);
    if (ret != 0 || !depth_ptr) {
        aicore_depth_free(ctx);
        return false;
    }

    out.width = w;
    out.height = h;
    out.depth.assign(depth_ptr, depth_ptr + h * w);
    if (conf_ptr) {
        out.confidence.assign(conf_ptr, conf_ptr + h * w);
        aicore_depth_free_floats(conf_ptr);
    }
    if (sky_ptr) aicore_depth_free_floats(sky_ptr);

    out.has_pose = true;
    std::memcpy(out.extrinsics, ext, sizeof(ext));
    std::memcpy(out.intrinsics, intr, sizeof(intr));

    aicore_depth_free_floats(depth_ptr);
    aicore_depth_free(ctx);
    return true;
}

}  // namespace depth
}  // namespace aicore
