// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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

aicore_image_format image_format(const QImage& image) {
    switch (image.format()) {
        case QImage::Format_RGB888:
            return AICORE_IMAGE_RGB8;
        case QImage::Format_RGBA8888:
            return AICORE_IMAGE_RGBA8;
        case QImage::Format_Grayscale8:
            return AICORE_IMAGE_GRAY8;
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        case QImage::Format_BGR888:
            return AICORE_IMAGE_BGR8;
#endif
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
            return AICORE_IMAGE_BGRA8;
#endif
        default:
            return static_cast<aicore_image_format>(0);
    }
}

QImage inference_view_image(const QImage& source) {
    if (source.isNull()) return {};
    return image_format(source) != 0
                   ? source
                   : source.convertToFormat(QImage::Format_RGB888);
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
    const QImage rgb = inference_view_image(image);
    if (rgb.isNull()) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, device, ctx))
        return false;

    aicore_image_view view{reinterpret_cast<const uint8_t*>(rgb.constBits()),
                           rgb.width(), rgb.height(),
                           static_cast<size_t>(rgb.bytesPerLine()),
                           image_format(rgb)};
    aicore_depth_dense_result dense{};
    if (aicore_depth_depth_image(ctx, &view, &dense) != 0 || !dense.depth) {
        aicore_depth_free(ctx);
        return false;
    }

    out.width = dense.width;
    out.height = dense.height;
    out.depth.assign(dense.depth, dense.depth + dense.width * dense.height);
    out.has_pose = false;
    aicore_depth_dense_result_free(&dense);
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
    const QImage rgb = inference_view_image(image);
    if (rgb.isNull()) return false;

    aicore_depth_ctx* ctx = nullptr;
    if (!loadContext(model_path, metric_model_path, n_threads, device, ctx))
        return false;

    aicore_image_view view{reinterpret_cast<const uint8_t*>(rgb.constBits()),
                           rgb.width(), rgb.height(),
                           static_cast<size_t>(rgb.bytesPerLine()),
                           image_format(rgb)};
    aicore_depth_dense_result dense{};
    const int ret = aicore_depth_depth_image(ctx, &view, &dense);
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
