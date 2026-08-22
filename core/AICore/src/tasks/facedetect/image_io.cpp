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
    out = std::move(result);
    return true;
}

}  // namespace fd
