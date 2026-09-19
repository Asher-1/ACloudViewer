// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/capi_utils.hpp"

#include <QImage>
#include <QString>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace aicore {
namespace capi {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char ch : s) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if ((unsigned char)ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x",
                                  (unsigned)(unsigned char)ch);
                    out += buf;
                } else {
                    out += ch;
                }
                break;
        }
    }
    return out;
}

PackedRgb qimage_to_packed_rgb(const QImage& image) {
    PackedRgb out;
    if (image.isNull()) return out;
    QImage rgb = image.convertToFormat(QImage::Format_RGB888);
    const int w = rgb.width(), h = rgb.height();
    if (w <= 0 || h <= 0) return out;
    const size_t nbytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(nbytes));
    if (buf == nullptr) return out;
    if (rgb.bytesPerLine() == w * 3) {
        std::memcpy(buf, rgb.constBits(), nbytes);
    } else {
        for (int y = 0; y < h; ++y) {
            std::memcpy(buf + static_cast<size_t>(y) * w * 3,
                        rgb.constScanLine(y), static_cast<size_t>(w) * 3);
        }
    }
    out.data = buf;
    out.width = w;
    out.height = h;
    return out;
}

bool image_view_to_packed_rgb(const aicore_image_view& view,
                              std::vector<uint8_t>& out) {
    if (!view.data || view.width <= 0 || view.height <= 0) return false;
    size_t channels = 0;
    if (view.format == AICORE_IMAGE_RGB8)
        channels = 3;
    else if (view.format == AICORE_IMAGE_RGBA8)
        channels = 4;
    else if (view.format == AICORE_IMAGE_GRAY8)
        channels = 1;
    else if (view.format == AICORE_IMAGE_BGR8)
        channels = 3;
    else if (view.format == AICORE_IMAGE_BGRA8)
        channels = 4;
    else
        return false;
    const size_t row = static_cast<size_t>(view.width) * channels;
    if (view.row_stride_bytes < row) return false;
    out.resize(static_cast<size_t>(view.width) * view.height * 3);
    for (int y = 0; y < view.height; ++y) {
        const uint8_t* src =
                view.data + static_cast<size_t>(y) * view.row_stride_bytes;
        uint8_t* dst = out.data() + static_cast<size_t>(y) * view.width * 3;
        if (channels == 3 && view.format == AICORE_IMAGE_RGB8)
            std::memcpy(dst, src, row);
        else
            for (int x = 0; x < view.width; ++x) {
                if (channels == 1)
                    dst[3 * x] = dst[3 * x + 1] = dst[3 * x + 2] = src[x];
                else {
                    const int step = channels;
                    const bool bgr = view.format == AICORE_IMAGE_BGR8 ||
                                     view.format == AICORE_IMAGE_BGRA8;
                    dst[3 * x] = src[step * x + (bgr ? 2 : 0)];
                    dst[3 * x + 1] = src[step * x + 1];
                    dst[3 * x + 2] = src[step * x + (bgr ? 0 : 2)];
                }
            }
    }
    return true;
}

}  // namespace capi
}  // namespace aicore
