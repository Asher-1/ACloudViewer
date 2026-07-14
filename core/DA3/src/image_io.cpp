#include "image_io.hpp"

#include "CVTools.h"

#include <QByteArray>
#include <QImage>
#include <QImageReader>
#include <cstring>

namespace da {

bool load_image_rgb(const std::string& path, Image& out) {
    QImageReader reader(CVTools::ToQString(path));
    reader.setAutoTransform(true);
    QImage img = reader.read();
    if (img.isNull()) return false;
    img = img.convertToFormat(QImage::Format_RGB888);
    out.w = img.width();
    out.h = img.height();
    const int stride = img.bytesPerLine();
    const int row_bytes = out.w * 3;
    out.rgb.resize(static_cast<size_t>(out.w) * out.h * 3);
    for (int y = 0; y < out.h; ++y) {
        const uchar* line = img.constScanLine(y);
        std::memcpy(out.rgb.data() + y * row_bytes, line, row_bytes);
    }
    return true;
}

bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out) {
    QByteArray ba(reinterpret_cast<const char*>(bytes), static_cast<int>(len));
    QImage img;
    if (!img.loadFromData(ba)) return false;
    img = img.convertToFormat(QImage::Format_RGB888);
    out.w = img.width();
    out.h = img.height();
    const int row_bytes = out.w * 3;
    out.rgb.resize(static_cast<size_t>(out.w) * out.h * 3);
    for (int y = 0; y < out.h; ++y) {
        const uchar* line = img.constScanLine(y);
        std::memcpy(out.rgb.data() + y * row_bytes, line, row_bytes);
    }
    return true;
}

}  // namespace da
