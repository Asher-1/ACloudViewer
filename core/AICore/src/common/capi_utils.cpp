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

}  // namespace capi
}  // namespace aicore
