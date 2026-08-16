#include "image_io.hpp"
#include "common.hpp"
#include "visualize.hpp"

// Qt-based image I/O (matching the gaussian task pattern).
// Replaces stb_image / stb_image_write / stb_image_resize with QImage.
#include <QImage>
#include <QBuffer>
#include <QIODevice>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <vector>

namespace {

/// Convert a decoded QImage (any format) to an rfdetr_image (RGB888, HWC).
rfdetr_image* qimage_to_rfdetr(const QImage& img, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (img.isNull()) {
        set(RFDETR_ERR_DECODE);
        return nullptr;
    }
    const int w = img.width();
    const int h = img.height();
    QImage rgb = img.convertToFormat(QImage::Format_RGB888);

    rfdetr_image* out = new (std::nothrow) rfdetr_image();
    if (!out) { set(RFDETR_ERR_OUT_OF_MEMORY); return nullptr; }
    try {
        out->width    = w;
        out->height   = h;
        out->channels = 3;
        out->rgb.resize((size_t)w * h * 3);
        for (int y = 0; y < h; ++y) {
            memcpy(out->rgb.data() + (size_t)y * w * 3,
                   rgb.constScanLine(y), w * 3);
        }
    } catch (const std::bad_alloc&) {
        delete out;
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
    set(RFDETR_OK);
    return out;
}

}  // anonymous namespace

extern "C" rfdetr_image* rfdetr_image_load_buffer(const uint8_t* bytes, size_t len, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!bytes || len == 0) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    QImage img;
    if (!img.loadFromData(bytes, (int)len)) {
        rfdetr_logf(RFDETR_LOG_ERROR, "QImage::loadFromData failed");
        set(RFDETR_ERR_DECODE);
        return nullptr;
    }
    return qimage_to_rfdetr(img, out_status);
}

extern "C" rfdetr_image* rfdetr_image_from_rgb_buffer(const uint8_t* rgb,
                                                      int width, int height,
                                                      rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!rgb || width <= 0 || height <= 0) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    /* Guard against overflow on (w * h * 3). On a 64-bit host this is mainly
     * theoretical, but it's free to check and lets callers feeding untrusted
     * stream metadata fail cleanly. */
    const size_t w = (size_t)width;
    const size_t h = (size_t)height;
    if (w > (size_t)-1 / 3 / (h ? h : 1)) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }
    const size_t nbytes = w * h * 3;

    auto* img = new (std::nothrow) rfdetr_image();
    if (!img) { set(RFDETR_ERR_OUT_OF_MEMORY); return nullptr; }
    try {
        img->width    = width;
        img->height   = height;
        img->channels = 3;
        img->rgb.assign(rgb, rgb + nbytes);
    } catch (const std::bad_alloc&) {
        delete img;
        set(RFDETR_ERR_OUT_OF_MEMORY);
        return nullptr;
    }
    set(RFDETR_OK);
    return img;
}

extern "C" rfdetr_image* rfdetr_image_load_file(const char* path, rfdetr_status* out_status) {
    auto set = [&](rfdetr_status s) { if (out_status) *out_status = s; };
    if (!path) { set(RFDETR_ERR_INVALID_ARG); return nullptr; }

    QImage img(QString::fromUtf8(path));
    if (img.isNull()) {
        rfdetr_logf(RFDETR_LOG_ERROR, "image_load_file: cannot decode '%s'", path);
        set(RFDETR_ERR_FILE_NOT_FOUND);
        return nullptr;
    }
    return qimage_to_rfdetr(img, out_status);
}

extern "C" void rfdetr_image_free(rfdetr_image* img) {
    delete img;
}

extern "C" int rfdetr_image_width(const rfdetr_image* img) {
    return img ? img->width : 0;
}

extern "C" int rfdetr_image_height(const rfdetr_image* img) {
    return img ? img->height : 0;
}

extern "C" const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img) {
    return img ? img->rgb.data() : nullptr;
}

extern "C" rfdetr_status rfdetr_render(const rfdetr_image* img,
                                       const rfdetr_detection* detections, size_t n,
                                       const char* out_path) {
    if (!img || !out_path) return RFDETR_ERR_INVALID_ARG;

    /* Copy so we don't mutate the caller's image. */
    rfdetr_image copy;
    try {
        copy = *img;  /* deep copy of pixel buffer */
    } catch (const std::bad_alloc&) {
        return RFDETR_ERR_OUT_OF_MEMORY;
    }
    /* First pass: tint each masked region with the detection's class color
     * (segmentation models only — no-op for detection-only models). Doing
     * this in a separate pass before the boxes ensures bbox strokes and
     * label backgrounds always sit on top of the mask shading. */
    for (size_t i = 0; i < n; ++i) {
        rfdetr_visualize_overlay_mask(&copy, detections[i], /*alpha*/ 0.4f);
    }
    /* Second pass: draw the box outline and label for each detection. */
    for (size_t i = 0; i < n; ++i) {
        rfdetr_visualize_draw_box(&copy, detections[i], /*thickness*/ 3);
    }

    QImage qimg(copy.rgb.data(), copy.width, copy.height, copy.width * 3,
                QImage::Format_RGB888);
    if (!qimg.save(QString::fromUtf8(out_path), "PNG")) {
        rfdetr_logf(RFDETR_LOG_ERROR, "QImage::save failed for '%s'", out_path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}

extern "C" rfdetr_status rfdetr_write_gray_png(const char* path,
                                               const uint8_t* data,
                                               int width, int height) {
    if (!path || !data || width <= 0 || height <= 0) return RFDETR_ERR_INVALID_ARG;
    QImage qimg(data, width, height, width, QImage::Format_Grayscale8);
    if (!qimg.save(QString::fromUtf8(path), "PNG")) {
        rfdetr_logf(RFDETR_LOG_ERROR, "QImage::save (gray) failed for '%s'", path);
        return RFDETR_ERR_IO;
    }
    return RFDETR_OK;
}

bool rfdetr_encode_gray_png(const uint8_t* data, int width, int height,
                            std::vector<uint8_t>& out) {
    out.clear();
    if (!data || width <= 0 || height <= 0) return false;
    QImage qimg(data, width, height, width, QImage::Format_Grayscale8);
    QByteArray ba;
    QBuffer buf(&ba);
    if (!qimg.save(&buf, "PNG")) return false;
    out.assign(reinterpret_cast<const uint8_t*>(ba.constData()),
               reinterpret_cast<const uint8_t*>(ba.constData()) + ba.size());
    return !out.empty();
}

extern "C" rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                           int target_w, int target_h,
                                           const float mean[3], const float std_[3],
                                           float** out_data, int* out_w, int* out_h) {
    if (!img || !out_data || !out_w || !out_h || !mean || !std_) {
        return RFDETR_ERR_INVALID_ARG;
    }
    if (target_w <= 0 || target_h <= 0) return RFDETR_ERR_INVALID_ARG;
    if (img->width <= 0 || img->height <= 0 || img->rgb.empty()) {
        return RFDETR_ERR_INVALID_ARG;
    }

    /* 1. Resize via QImage::scaled (SmoothTransformation = bilinear).
     * Input is uint8 RGB packed HWC. */
    QImage src(img->rgb.data(), img->width, img->height, img->width * 3,
               QImage::Format_RGB888);
    QImage resized = src.scaled(target_w, target_h, Qt::IgnoreAspectRatio,
                                Qt::SmoothTransformation);
    resized = resized.convertToFormat(QImage::Format_RGB888);

    /* 2-3. Allocate output F32 buffer and write NCHW row-major.
     *
     * ggml ne = (W, H, 3, 1): ne[0]=W fastest-varying. Memory order:
     *   offset(c, h, w) = c*H*W + h*W + w
     * That's NCHW row-major where w is fastest. */
    const size_t n_elems = (size_t)target_w * (size_t)target_h * 3;
    float* buf = (float*)std::malloc(n_elems * sizeof(float));
    if (!buf) return RFDETR_ERR_OUT_OF_MEMORY;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            const uchar* line = resized.constScanLine(h);
            for (int w = 0; w < target_w; ++w) {
                float v = line[w * 3 + c] / 255.0f;
                v = (v - mean[c]) / std_[c];
                buf[(size_t)c * target_h * target_w + (size_t)h * target_w + w] = v;
            }
        }
    }

    *out_data = buf;
    *out_w = target_w;
    *out_h = target_h;
    return RFDETR_OK;
}
