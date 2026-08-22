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
#include <cstring>

#include "CVTools.h"
#include "common/capi_utils.hpp"

namespace aicore {
namespace depth {

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
    std::free(packed.data);
    return true;
}

}  // namespace depth
}  // namespace aicore
