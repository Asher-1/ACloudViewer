// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QStandardPaths>
#include <QTemporaryFile>
#include <cstring>

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/depth_image.h"
#include "aicore/runtime_capi.h"

namespace aicore {
namespace depth {
namespace {

bool loadContext(const QString& model_path,
                 const QString& metric_model_path,
                 int n_threads,
                 const QString& device,
                 aicore_depth_ctx*& ctx) {
    const int threads = n_threads > 0 ? n_threads : 1;
    aicore_depth_options* opts = aicore_depth_options_new();
    if (!opts) return false;
    aicore_depth_options_set_threads(opts, threads);
    if (!device.isEmpty())
        aicore_depth_options_set_device(opts, device.toUtf8().constData());
    if (metric_model_path.isEmpty()) {
        ctx = aicore_depth_load_opts(model_path.toUtf8().constData(), opts);
    } else {
        ctx = aicore_depth_load_nested_opts(
                model_path.toUtf8().constData(),
                metric_model_path.toUtf8().constData(), opts);
    }
    aicore_depth_options_free(opts);
    return ctx != nullptr;
}

bool writeTempPng(const QImage& image, QTemporaryFile& tmp) {
    if (image.isNull()) return false;
    QString cache =
            QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (cache.isEmpty()) cache = QDir::tempPath();
    QDir().mkpath(cache + QStringLiteral("/aicore_depth"));
    tmp.setFileTemplate(cache + "/aicore_depth/image_XXXXXX.png");
    if (!tmp.open()) return false;
    if (!image.save(&tmp, "PNG")) return false;
    tmp.close();
    return true;
}

}  // namespace

class ImageDepthTaskScope {
public:
    ImageDepthTaskScope(const QString& device, aicore_cancel_token* external)
        : token_(external), owns_(external == nullptr) {
        if (owns_) token_ = aicore_cancel_token_new();
        if (!token_) return;
        locked_ = aicore_device_task_lock_cancelable(
                          device.toUtf8().constData(), token_) == 0;
        if (locked_) aicore_cancel_scope_begin(token_);
    }
    ~ImageDepthTaskScope() {
        if (locked_) {
            aicore_cancel_scope_end(token_);
            aicore_device_task_unlock();
        }
        if (owns_) aicore_cancel_token_free(token_);
    }
    bool active() const { return locked_; }

private:
    aicore_cancel_token* token_ = nullptr;
    bool owns_ = false;
    bool locked_ = false;
};

bool ImageDepth::isAvailable(const QString& device) {
    return aicore_device_available(device.toUtf8().constData()) != 0;
}

bool ImageDepth::estimateDepth(const QImage& image,
                               const QString& model_path,
                               int n_threads,
                               ImageDepthResult& out,
                               const QString& metric_model_path,
                               const QString& device,
                               aicore_cancel_token* cancel_token) {
    ImageDepthTaskScope task(device, cancel_token);
    if (!task.active()) return false;
    QTemporaryFile tmp;
    if (!writeTempPng(image, tmp)) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, device, ctx))
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
    std::free(depth);
    aicore_depth_free(ctx);
    return true;
}

bool ImageDepth::estimateDepthAndPose(const QImage& image,
                                      const QString& model_path,
                                      int n_threads,
                                      ImageDepthResult& out,
                                      const QString& metric_model_path,
                                      const QString& device,
                                      aicore_cancel_token* cancel_token) {
    ImageDepthTaskScope task(device, cancel_token);
    if (!task.active()) return false;
    QTemporaryFile tmp;
    if (!writeTempPng(image, tmp)) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, device, ctx))
        return false;

    aicore_depth_dense_result dense{};
    const int ret = aicore_depth_depth_dense(
            ctx, tmp.fileName().toUtf8().constData(), &dense);
    if (ret != 0 || !dense.depth) {
        aicore_depth_dense_result_free(&dense);
        aicore_depth_free(ctx);
        return false;
    }

    out.width = dense.width;
    out.height = dense.height;
    out.depth.assign(dense.depth, dense.depth + dense.height * dense.width);
    if (dense.conf) {
        out.confidence.assign(dense.conf,
                              dense.conf + dense.height * dense.width);
    }

    out.has_pose = true;
    std::memcpy(out.extrinsics, dense.ext, sizeof(out.extrinsics));
    std::memcpy(out.intrinsics, dense.intr, sizeof(out.intrinsics));

    aicore_depth_dense_result_free(&dense);
    aicore_depth_free(ctx);
    return true;
}

}  // namespace depth
}  // namespace aicore
