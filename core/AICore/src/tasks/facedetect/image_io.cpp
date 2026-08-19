// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "image_io.hpp"

#include <QImage>
#include <cstring>
#include <limits>
#include <utility>

#include "common.hpp"

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

    const QImage rgb = decoded.convertToFormat(QImage::Format_RGB888);
    size_t byteCount = 0;
    if (rgb.isNull() || !rgbSize(rgb.width(), rgb.height(), &byteCount)) {
        return false;
    }

    Image result;
    result.width = rgb.width();
    result.height = rgb.height();
    result.rgb.resize(byteCount);
    const size_t rowBytes = static_cast<size_t>(result.width) * 3;
    for (int y = 0; y < result.height; ++y) {
        std::memcpy(result.rgb.data() + static_cast<size_t>(y) * rowBytes,
                    rgb.constScanLine(y), rowBytes);
    }
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
