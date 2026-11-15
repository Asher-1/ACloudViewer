// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// Instance-mask preparation and blending for the annotated preview. Qt Gui
// only (no widgets, no AICore), so it is unit-tested in isolation.
#include <QColor>
#include <QImage>
#include <QPoint>
#include <QRect>
#include <QRectF>
#include <QSize>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace RelateMask {

struct ObjectMask {
    QRect rect;     // image-space region covered by the alpha plane
    QImage alpha;   // Format_Grayscale8, rect.size(); 0..255 coverage
    QRectF source;  // box the mask was cut for; fitMask() moves it elsewhere
};

// Separable Gaussian on a binary plane followed by a x2 ramp around 50%: an
// anti-aliased edge about two pixels wide.
//  * sigmaPx > 0: blur at full resolution with that sigma. Used for masks that
//    already have pixel-level contours (bilinearly upsampled logits, supplied
//    masks).
//  * otherwise `block` is the staircase step in image pixels (the detector's
//    prototype cell): the blur runs on a crop reduced by f = block/2 (<= 4) so
//    the cost follows the object's area, with sigma ~ one cell.
//  * block <= 1 and sigmaPx == 0 returns the plane unchanged.
inline QImage softenAlpha(const QImage& gray, int block, double sigmaPx = 0) {
    if (block <= 1 && sigmaPx <= 0) return gray;
    const int w = gray.width(), h = gray.height();
    const int f = sigmaPx > 0 ? 1 : std::clamp(block / 2, 1, 4);
    const int lw = (w + f - 1) / f, lh = (h + f - 1) / f;
    std::vector<float> a(size_t(lw) * lh, 0.f), tmp(a.size());
    std::vector<int> count(a.size(), 0);
    for (int y = 0; y < h; ++y) {
        const uchar* p = gray.constScanLine(y);
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y / f) * lw + x / f;
            a[i] += p[x];
            ++count[i];
        }
    }
    for (size_t i = 0; i < a.size(); ++i) a[i] /= std::max(1, count[i]) * 255.f;
    const float sigma =
            sigmaPx > 0 ? float(sigmaPx) : std::max(.5f, float(block) / f);
    const int radius = int(std::ceil(3 * sigma));
    std::vector<float> kernel(size_t(2 * radius + 1));
    float sum = 0;
    for (int k = -radius; k <= radius; ++k)
        sum += kernel[size_t(k + radius)] =
                std::exp(-k * k / (2 * sigma * sigma));
    for (auto& v : kernel) v /= sum;
    for (int y = 0; y < lh; ++y)  // horizontal, edge-clamped
        for (int x = 0; x < lw; ++x) {
            float s = 0;
            for (int k = -radius; k <= radius; ++k)
                s += kernel[size_t(k + radius)] *
                     a[size_t(y) * lw + std::clamp(x + k, 0, lw - 1)];
            tmp[size_t(y) * lw + x] = s;
        }
    for (int y = 0; y < lh; ++y)  // vertical
        for (int x = 0; x < lw; ++x) {
            float s = 0;
            for (int k = -radius; k <= radius; ++k)
                s += kernel[size_t(k + radius)] *
                     tmp[size_t(std::clamp(y + k, 0, lh - 1)) * lw + x];
            a[size_t(y) * lw + x] = s;
        }
    QImage out(w, h, QImage::Format_Grayscale8);
    for (int y = 0; y < h; ++y) {  // bilinear upsample + ramp
        const float fy = std::clamp((y + .5f) / f - .5f, 0.f, lh - 1.f);
        const int y0 = int(fy), y1 = std::min(y0 + 1, lh - 1);
        const float ty = fy - y0;
        uchar* o = out.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const float fx = std::clamp((x + .5f) / f - .5f, 0.f, lw - 1.f);
            const int x0 = int(fx), x1 = std::min(x0 + 1, lw - 1);
            const float tx = fx - x0;
            const float v = (a[size_t(y0) * lw + x0] * (1 - tx) +
                             a[size_t(y0) * lw + x1] * tx) *
                                    (1 - ty) +
                            (a[size_t(y1) * lw + x0] * (1 - tx) +
                             a[size_t(y1) * lw + x1] * tx) *
                                    ty;
            o[x] = uchar(std::clamp((v - .5f) * 2.f + .5f, 0.f, 1.f) * 255.f +
                         .5f);
        }
    }
    return out;
}
// Crop one object's plane (any resolution, non-zero = inside) around its box
// and map it onto image pixels. `block` is the plane's staircase step in image
// pixels (<= 1: pixel-exact plane); `sigmaPx` adds a pixel-level blur.
inline ObjectMask buildMask(const uint8_t* plane,
                            int mw,
                            int mh,
                            size_t stride,
                            const float* box,
                            const QSize& image,
                            int block,
                            double sigmaPx = 0) {
    ObjectMask out;
    if (!plane || mw <= 0 || mh <= 0 || image.isEmpty()) return out;
    const double sx = double(mw) / image.width(),
                 sy = double(mh) / image.height();
    // Margin covers the blur radius so the edge is not cut at the crop border.
    const int margin =
            std::max(3 * std::max(1, block), int(std::ceil(3 * sigmaPx))) + 2;
    QRect box_px(QPoint(int(std::floor(box[0])) - margin,
                        int(std::floor(box[1])) - margin),
                 QPoint(int(std::ceil(box[2])) + margin,
                        int(std::ceil(box[3])) + margin));
    box_px = box_px.intersected(QRect(QPoint(0, 0), image));
    QRect mr(QPoint(int(std::floor(box_px.left() * sx)),
                    int(std::floor(box_px.top() * sy))),
             QPoint(int(std::ceil((box_px.right() + 1) * sx)) - 1,
                    int(std::ceil((box_px.bottom() + 1) * sy)) - 1));
    mr = mr.intersected(QRect(0, 0, mw, mh));
    if (mr.isEmpty()) return out;
    QImage gray(mr.size(), QImage::Format_Grayscale8);
    for (int y = 0; y < mr.height(); ++y) {
        const uint8_t* src = plane + size_t(mr.top() + y) * stride + mr.left();
        uchar* dst = gray.scanLine(y);
        for (int x = 0; x < mr.width(); ++x) dst[x] = src[x] ? 255 : 0;
    }
    QRect ir(QPoint(int(std::lround(mr.left() / sx)),
                    int(std::lround(mr.top() / sy))),
             QSize(std::max(1, int(std::lround(mr.width() / sx))),
                   std::max(1, int(std::lround(mr.height() / sy)))));
    ir = ir.intersected(QRect(QPoint(0, 0), image));
    if (ir.isEmpty()) return out;
    if (gray.size() != ir.size())
        gray = gray.scaled(ir.size(), Qt::IgnoreAspectRatio,
                           Qt::SmoothTransformation);
    out.rect = ir;
    out.source = QRectF(QPointF(box[0], box[1]), QPointF(box[2], box[3]));
    out.alpha = softenAlpha(gray, block, sigmaPx);
    return out;
}
// Move a mask cut for one box onto another (video: the mask comes from the
// latest raw detection while the box drawn is the Kalman-smoothed track box).
// The plane is translated and scaled with the box so contour and box agree.
inline ObjectMask fitMask(const ObjectMask& m,
                          const QRectF& target,
                          const QSize& image) {
    if (m.alpha.isNull() || m.source.isEmpty() || target.isEmpty()) return m;
    const double sx = target.width() / m.source.width(),
                 sy = target.height() / m.source.height();
    const QRectF moved(target.left() + (m.rect.left() - m.source.left()) * sx,
                       target.top() + (m.rect.top() - m.source.top()) * sy,
                       m.rect.width() * sx, m.rect.height() * sy);
    QRect ir(QPoint(int(std::lround(moved.left())),
                    int(std::lround(moved.top()))),
             QSize(std::max(1, int(std::lround(moved.width()))),
                   std::max(1, int(std::lround(moved.height())))));
    ObjectMask out;
    out.source = target;
    QImage alpha = ir.size() == m.alpha.size()
                           ? m.alpha
                           : m.alpha.scaled(ir.size(), Qt::IgnoreAspectRatio,
                                            Qt::SmoothTransformation);
    const QRect visible = ir.intersected(QRect(QPoint(0, 0), image));
    if (visible.isEmpty()) return ObjectMask{};
    out.rect = visible;
    out.alpha = visible == ir ? alpha
                              : alpha.copy(visible.translated(-ir.topLeft()));
    return out;
}
// Soft translucent fill plus a brighter band where coverage crosses 50%, which
// is what the official renderer's contour outline looks like.
inline void blendMask(QImage& canvas,
                      const ObjectMask& m,
                      const QColor& c,
                      double strength = 1.0) {
    if (m.alpha.isNull() || strength <= 0) return;
    for (int y = 0; y < m.alpha.height(); ++y) {
        const int cy = m.rect.top() + y;
        if (cy < 0 || cy >= canvas.height()) continue;
        auto* line = reinterpret_cast<QRgb*>(canvas.scanLine(cy));
        const uchar* a = m.alpha.constScanLine(y);
        for (int x = 0; x < m.alpha.width(); ++x) {
            if (!a[x]) continue;
            const int cx = m.rect.left() + x;
            if (cx < 0 || cx >= canvas.width()) continue;
            const double cover = a[x] / 255.0;
            const double edge = 1.0 - std::abs(cover - .5) * 2.0;
            const double w =
                    std::min(.9, cover * .38 + edge * edge * .5) * strength;
            line[cx] = qRgb(int(qRed(line[cx]) * (1 - w) + c.red() * w),
                            int(qGreen(line[cx]) * (1 - w) + c.green() * w),
                            int(qBlue(line[cx]) * (1 - w) + c.blue() * w));
        }
    }
}
}  // namespace RelateMask
