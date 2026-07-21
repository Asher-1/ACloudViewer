// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Qt-based image I/O implementation for FreeSplatter.
// Replaces stb_image with QImage for loading, center-cropping, and resizing.
#include "image_io.hpp"

#include <QImage>
#include <QImageReader>
#include <algorithm>
#include <cstring>

#ifdef AICore_HAS_CVLOG
#include "CVLog.h"
#include "CVTools.h"
#else
#define CVLog_warning(...)
#endif

namespace aicore {
namespace gaussian {

bool load_image_chw(const std::string& path,
                    int size,
                    std::vector<float>& out,
                    std::string& err) {
    QImageReader reader(QString::fromStdString(path));
    reader.setAutoTransform(true);
    QImage img = reader.read();
    if (img.isNull()) {
        err = "failed to decode image: " + path + " (" +
              reader.errorString().toStdString() + ")";
        return false;
    }

    img = img.convertToFormat(QImage::Format_RGB888);

    // Center-crop to square
    const int w = img.width();
    const int h = img.height();
    const int s = std::min(w, h);
    const int left = (w - s) / 2;
    const int top = (h - s) / 2;
    QImage cropped = img.copy(left, top, s, s);

    // Resize to model resolution
    QImage resized = cropped.scaled(size, size, Qt::KeepAspectRatioByExpanding,
                                    Qt::SmoothTransformation);
    // scaled with KeepAspectRatioByExpanding should give exactly size x size
    // since the input is already square, but ensure it:
    if (resized.width() != size || resized.height() != size) {
        resized = resized.scaled(size, size, Qt::IgnoreAspectRatio,
                                 Qt::SmoothTransformation);
    }
    resized = resized.convertToFormat(QImage::Format_RGB888);

    // Extract to CHW float[0,1]
    const size_t base = out.size();
    out.resize(base + (size_t)3 * size * size);
    const int stride = resized.bytesPerLine();
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < size; y++) {
            const uchar* line = resized.constScanLine(y);
            for (int x = 0; x < size; x++) {
                out[base + (size_t)c * size * size + y * size + x] =
                        line[x * 3 + c] / 255.0f;
            }
        }
    }
    return true;
}

bool load_images_chw(const std::vector<std::string>& paths,
                     int size,
                     std::vector<float>& out,
                     std::string& err) {
    out.clear();
    out.reserve((size_t)paths.size() * 3 * size * size);
    for (size_t i = 0; i < paths.size(); i++) {
        if (!load_image_chw(paths[i], size, out, err)) {
            return false;
        }
    }
    return true;
}

}  // namespace gaussian
}  // namespace aicore
