// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/facedetect/image_io.hpp"

#include <QImage>
#include <cstring>
#include <limits>
#include <utility>

#include "common/capi_utils.hpp"
#include "tasks/facedetect/common.hpp"

namespace fd {
namespace {

bool rgbSize(int width, int height, size_t* size) {
    if (!size || width <= 0 || height <= 0) return false;
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::numeric_limits<size_t>::max() / 3 / h) return false;
    *size = w * h * 3;
    return true;
}

}  // namespace

bool load_image_rgb(const std::string& path, Image& out) {
    QImage decoded(QString::fromUtf8(path.c_str()));
    if (decoded.isNull()) {
        FD_LOG("load_image_rgb: failed to decode %s", path.c_str());
        return false;
    }

    aicore::capi::PackedRgb packed =
            aicore::capi::qimage_to_packed_rgb(decoded);
    if (packed.data == nullptr) {
        return false;
    }
    const size_t byteCount =
            static_cast<size_t>(packed.width) * packed.height * 3;

    Image result;
    result.width = packed.width;
    result.height = packed.height;
    result.rgb.assign(packed.data, packed.data + byteCount);
    result.row_stride_bytes = static_cast<size_t>(result.width) * 3;
    std::free(packed.data);
    out = std::move(result);
    return true;
}

bool image_from_rgb(const uint8_t* rgb, int width, int height, Image& out) {
    size_t byteCount = 0;
    if (!rgb || !rgbSize(width, height, &byteCount)) return false;

    Image result;
    result.width = width;
    result.height = height;
    result.rgb.assign(rgb, rgb + byteCount);
    result.row_stride_bytes = static_cast<size_t>(width) * 3;
    out = std::move(result);
    return true;
}

bool image_from_view(const aicore_image_view& view, Image& out) {
    if (view.data == nullptr || view.width <= 0 || view.height <= 0) {
        return false;
    }
    size_t channels = 0;
    bool bgr = false;
    switch (view.format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            bgr = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            bgr = true;
            break;
        default:
            return false;
    }
    const size_t row_bytes = static_cast<size_t>(view.width) * channels;
    if (view.row_stride_bytes < row_bytes ||
        static_cast<size_t>(view.height) >
                std::numeric_limits<size_t>::max() / row_bytes) {
        return false;
    }
    Image result;
    result.width = view.width;
    result.height = view.height;
    result.borrowed_data = view.data;
    result.row_stride_bytes = view.row_stride_bytes;
    result.channels = static_cast<int>(channels);
    result.bgr = bgr;
    out = std::move(result);
    return true;
}

}  // namespace fd
