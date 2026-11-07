// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/depth/image_io.hpp"

#include <QByteArray>
#include <QImage>
#include <QImageReader>

#include "CVTools.h"
#include "common/capi_utils.hpp"

namespace aicore {
namespace depth {

bool borrow_image_view(const uint8_t* data,
                       int w,
                       int h,
                       int channels,
                       bool bgr,
                       size_t row_stride_bytes,
                       Image& out) {
    if (!data || w <= 0 || h <= 0 ||
        (channels != 1 && channels != 3 && channels != 4) ||
        row_stride_bytes < static_cast<size_t>(w) * channels) {
        return false;
    }
    out.w = w;
    out.h = h;
    out.channels = channels;
    out.bgr = bgr;
    out.row_stride_bytes = row_stride_bytes;
    out.borrowed = data;
    out.rgb.clear();
    return true;
}

bool load_image_rgb(const std::string& path, Image& out) {
    QImageReader reader(CVTools::ToQString(path));
    reader.setAutoTransform(true);
    QImage img = reader.read();
    if (img.isNull()) return false;
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) return false;
    out.w = packed.width;
    out.h = packed.height;
    out.rgb.assign(packed.data,
                   packed.data + (size_t)packed.width * packed.height * 3);
    out.reset_owned_layout();
    std::free(packed.data);
    return true;
}

bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out) {
    QByteArray ba(reinterpret_cast<const char*>(bytes), static_cast<int>(len));
    QImage img;
    if (!img.loadFromData(ba)) return false;
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) return false;
    out.w = packed.width;
    out.h = packed.height;
    out.rgb.assign(packed.data,
                   packed.data + (size_t)packed.width * packed.height * 3);
    out.reset_owned_layout();
    std::free(packed.data);
    return true;
}

}  // namespace depth
}  // namespace aicore
