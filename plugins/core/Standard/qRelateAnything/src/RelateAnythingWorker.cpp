// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RelateAnythingWorker.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QFontMetrics>
#include <QJsonArray>
#include <QJsonDocument>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QThread>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "RelateAnythingDecode.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvAssetIntegrity.h"
namespace {
void check(bool condition, const QString& error) {
    if (!condition) throw std::runtime_error(error.toStdString());
}
QPair<qint64, qint64> fileGeneration(const QString& path) {
    const QFileInfo info(path);
    return {info.isFile() ? info.size() : -1,
            info.lastModified().toMSecsSinceEpoch()};
}
void verify(const QString& path,
            const ecvAssetIntegrity::Anchor& anchor,
            qint64 size = 0,
            bool gguf = true) {
    check(ecvAssetIntegrity::isVerified(path, anchor, 65536, gguf,
                                        ecvAssetIntegrity::OnMiss::DeepVerify,
                                        size),
          QObject::tr("Asset verification failed: %1").arg(path));
}
struct ObjectMask {
    QRect rect;    // image-space region covered by the alpha plane
    QImage alpha;  // Format_Grayscale8, rect.size(); 0..255 coverage
};
struct Detected {
    std::vector<float> boxes, scores;
    QStringList labels;
    std::vector<ObjectMask> masks;  // empty entries for objects without one
};
// Instance masks arrive as binary planes at prototype resolution, so their
// edges are staircases one cell (`block` image pixels) wide. A Gaussian with
// sigma ~ 1 cell followed by a x2 ramp around 50% rounds the corners and gives
// a ~2 px anti-aliased edge (measured on a synthetic staircase: about half the
// misclassified pixels of the previous box filter, contour length back to the
// true shape). The blur runs on a crop reduced by f = block/2 (<= 4), so the
// cost is proportional to the object's area.
QImage softenAlpha(const QImage& gray, int block) {
    if (block <= 1) return gray;  // supplied masks: exact edges
    const int w = gray.width(), h = gray.height();
    const int f = std::clamp(block / 2, 1, 4);
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
    const float sigma = std::max(.5f, float(block) / f);
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
// and map it onto image pixels.
ObjectMask buildMask(const uint8_t* plane,
                     int mw,
                     int mh,
                     size_t stride,
                     const float* box,
                     const QSize& image,
                     int block) {
    ObjectMask out;
    if (!plane || mw <= 0 || mh <= 0 || image.isEmpty()) return out;
    const double sx = double(mw) / image.width(),
                 sy = double(mh) / image.height();
    const int margin = 3 * std::max(1, block) + 2;  // covers the blur radius
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
    out.alpha = softenAlpha(gray, block);
    return out;
}
using RelateDecode::Candidate;
using RelateDecode::keepBest;
QString nameOr(const char* n) {
    return n && n[0] ? QString::fromUtf8(n) : QObject::tr("object");
}
Detected detectRfdetr(aicore_rfdetr_ctx* rf,
                      const aicore_image_view& view,
                      const QSize& size,
                      const RelateAnythingSettings& s) {
    check(aicore_rfdetr_detect_image(rf, &view, s.threshold,
                                     uint32_t(std::max(s.maxBoxes * 4, 100))) ==
                  0,
          QString::fromUtf8(aicore_rfdetr_last_error(rf)));
    std::vector<Candidate> found;
    for (int i = 0; i < aicore_rfdetr_detection_count(rf); ++i) {
        aicore_rfdetr_detection d{};
        if (aicore_rfdetr_detection_at(rf, i, &d) != 0) continue;
        found.push_back(
                {{d.x1, d.y1, d.x2, d.y2}, d.score, nameOr(d.class_name), i});
    }
    found = keepBest(std::move(found), s.maxBoxes, size,
                     s.temporal ? 1.f : s.maxBoxFraction,
                     s.temporal ? 2.f : .7f);
    const bool seg = aicore_rfdetr_context_has_segmentation(rf) != 0;
    Detected out;
    for (const auto& c : found) {
        out.boxes.insert(out.boxes.end(), c.box, c.box + 4);
        out.scores.push_back(c.score);
        out.labels << c.name;
        ObjectMask mask;
        int32_t mw = 0, mh = 0;
        const int len = seg ? aicore_rfdetr_detection_mask(rf, c.source,
                                                           nullptr, 0, &mw, &mh)
                            : 0;
        if (len > 0 && mw > 0 && mh > 0 && size_t(len) >= size_t(mw) * mh) {
            std::vector<unsigned char> raw;
            raw.resize(size_t(len));
            if (aicore_rfdetr_detection_mask(rf, c.source, raw.data(), len, &mw,
                                             &mh) == len) {
                // The mask is stretched over the whole frame (no letterbox).
                const int block = std::max(
                        1, int(std::lround(double(std::max(size.width(),
                                                           size.height())) /
                                           std::max(mw, mh))));
                mask = buildMask(raw.data(), mw, mh, size_t(mw), c.box, size,
                                 block);
            }
        }
        out.masks.push_back(std::move(mask));
    }
    return out;
}
// Runs the loaded YOLO model (detection, instance segmentation, YOLOE or
// YOLO-World) and keeps the best maxBoxes objects, as the upstream demo does
// before the relation head. Segmenters also give a soft mask per object.
Detected detectYolo(aicore_yolo_ctx* yolo,
                    const aicore_image_view& view,
                    const QSize& size,
                    const RelateAnythingSettings& s) {
    std::unique_ptr<aicore_yolo_segment_result,
                    decltype(&aicore_yolo_seg_result_free)>
            seg(nullptr, aicore_yolo_seg_result_free);
    std::vector<Candidate> found;
    if (QByteArray(aicore_yolo_context_task(yolo)) == "segment") {
        seg.reset(aicore_yolo_seg_image(yolo, &view));
        check(bool(seg), QString::fromUtf8(aicore_yolo_last_error(yolo)));
        for (int i = 0; i < aicore_yolo_seg_det_count(seg.get()); ++i) {
            const auto d = aicore_yolo_seg_det_at(seg.get(), i);
            found.push_back(
                    {{d.x1, d.y1, d.x2, d.y2},
                     d.score,
                     nameOr(aicore_yolo_seg_det_class_name(seg.get(), i)),
                     i});
        }
    } else {
        check(aicore_yolo_detect_image(yolo, &view) == 0,
              QString::fromUtf8(aicore_yolo_last_error(yolo)));
        for (int i = 0; i < aicore_yolo_detection_count(yolo); ++i) {
            const auto d = aicore_yolo_detection_at(yolo, i);
            found.push_back({{d.x1, d.y1, d.x2, d.y2},
                             d.score,
                             nameOr(aicore_yolo_detection_class_name(yolo, i)),
                             i});
        }
    }
    found = keepBest(std::move(found), s.maxBoxes, size,
                     s.temporal ? 1.f : s.maxBoxFraction,
                     s.temporal ? 2.f : .7f);
    // Prototype masks are 1/4 of the network input; that cell is the block to
    // smooth over once they are unscaled to the source image.
    const double proto =
            std::max(1.0, double(aicore_yolo_context_image_size(yolo)) / 4.0);
    const int block = std::max(
            1, int(std::lround(std::max(size.width(), size.height()) / proto)));
    Detected out;
    for (const auto& c : found) {
        out.boxes.insert(out.boxes.end(), c.box, c.box + 4);
        out.scores.push_back(c.score);
        out.labels << c.name;
        ObjectMask mask;
        if (seg) {
            const auto m = aicore_yolo_seg_mask_at(seg.get(), c.source);
            if (m.data && m.width > 0 && m.height > 0)
                mask = buildMask(static_cast<const uint8_t*>(m.data), m.width,
                                 m.height, m.row_stride_bytes, c.box, size,
                                 block);
        }
        out.masks.push_back(std::move(mask));
    }
    return out;
}
using RelateDecode::PickedEdge;
// Selection logic lives in RelateAnythingDecode.h so the qAnyLabeling
// workspace picks objects and edges exactly like this plugin.
std::vector<PickedEdge> pickEdges(const aicore_relation_result& r,
                                  const RelateAnythingSettings& s,
                                  bool decomposed,
                                  float calibrationA,
                                  float calibrationB,
                                  const std::vector<float>& confs) {
    RelateDecode::Params params;
    params.relationThreshold = s.relationThreshold;
    params.spatialThreshold = s.spatialThreshold;
    params.boxScoreWeight = s.boxScoreWeight;
    params.birthTopk = s.tracking.birthTopk;
    params.maxRelations = s.maxRelations;
    params.maxSpatial = s.maxSpatial;
    return RelateDecode::pickEdges(r, params, decomposed, calibrationA,
                                   calibrationB, confs);
}
const QColor kPalette[] = {{64, 156, 255}, {255, 99, 88},   {52, 199, 123},
                           {255, 184, 48}, {191, 122, 255}, {0, 199, 208},
                           {255, 133, 71}, {145, 200, 60}};
QColor paletteColor(int index) { return kPalette[index % 8]; }
// Soft translucent fill plus a brighter band where coverage crosses 50%, which
// is what the official renderer's contour outline looks like.
void blendMask(QImage& canvas,
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
struct ChipLayout {
    QPainter& p;
    QSize bounds;
    std::vector<QRectF> used;
    // Place a rounded text chip near `anchor`, sliding it vertically until it
    // stops overlapping chips already drawn, and keep it inside the image.
    void chip(const QString& text,
              QPointF anchor,
              QColor fill,
              double alpha = 1.0) {
        fill.setAlphaF(fill.alphaF() * std::clamp(alpha, 0.0, 1.0));
        const QFontMetrics fm(p.font());
        const QSizeF size(fm.horizontalAdvance(text) + 10, fm.height() + 4);
        QRectF r(anchor, size);
        for (int tries = 0; tries < 12; ++tries) {
            r.moveLeft(
                    std::clamp(r.left(), 0.0,
                               std::max(0.0, bounds.width() - size.width())));
            r.moveTop(
                    std::clamp(r.top(), 0.0,
                               std::max(0.0, bounds.height() - size.height())));
            bool hit = false;
            for (const auto& u : used)
                if (u.intersects(r)) {
                    r.moveTop(u.bottom() + 1);
                    hit = true;
                    break;
                }
            if (!hit) break;
        }
        used.push_back(r);
        p.setPen(Qt::NoPen);
        p.setBrush(fill);
        p.drawRoundedRect(r, 4, 4);
        QColor ink(Qt::white);
        ink.setAlphaF(std::clamp(alpha, 0.0, 1.0));
        p.setPen(ink);
        p.drawText(r, Qt::AlignCenter, text);
    }
};
// Optional per-object styling for video: track fade, stable track colour, and
// whether a box is drawn even though a mask exists.
struct SceneStyle {
    std::vector<float> alpha;
    std::vector<int> colorId;
    std::vector<char> forceBox;
};
QImage renderScene(const QImage& image,
                   const std::vector<float>& boxes,
                   const QStringList& labels,
                   const std::vector<ObjectMask>& masks,
                   const QJsonArray& relations,
                   const SceneStyle& style = {}) {
    QImage out = image.convertToFormat(QImage::Format_RGB32);
    const int count = labels.size();
    const auto objAlpha = [&](int i) {
        return i < int(style.alpha.size()) ? double(style.alpha[size_t(i)])
                                           : 1.0;
    };
    const auto objColor = [&](int i) {
        return paletteColor(
                i < int(style.colorId.size()) ? style.colorId[size_t(i)] : i);
    };
    for (int i = 0; i < count && i < int(masks.size()); ++i)
        blendMask(out, masks[size_t(i)], objColor(i), objAlpha(i));
    const int shortSide = std::min(out.width(), out.height());
    const double lineWidth = std::max(1.5, shortSide / 300.0);
    QPainter painter(&out);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont font = painter.font();
    font.setPixelSize(std::max(11, shortSide / 40));
    font.setBold(true);
    painter.setFont(font);
    ChipLayout layout{painter, out.size(), {}};
    const int chipHeight = QFontMetrics(font).height() + 4;
    for (int i = 0; i < count; ++i) {
        const float* b = boxes.data() + i * 4;
        QColor c = objColor(i);
        const bool hasMask =
                i < int(masks.size()) && !masks[size_t(i)].alpha.isNull();
        const bool forceBox =
                i < int(style.forceBox.size()) && style.forceBox[size_t(i)];
        if (!hasMask && forceBox) {
            QColor pen = c;
            pen.setAlphaF(std::clamp(objAlpha(i), 0.0, 1.0));
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(pen, lineWidth));
            painter.drawRoundedRect(
                    QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3])), 4, 4);
        }
        layout.chip(labels[i], QPointF(b[0], b[1] - chipHeight - 1),
                    c.darker(135), objAlpha(i));
    }
    // Curved, subject-coloured arrows; each edge bows by a rank-dependent
    // offset and puts its chip at a different point along the curve, so edges
    // that share a subject neither overprint nor stack their labels.
    static const double kChipAt[] = {.35, .5, .65, .42, .58, .46, .54, .38};
    for (int i = 0; i < relations.size(); ++i) {
        const auto e = relations[i].toObject();
        const int si = e["subject"].toInt(), oi = e["object"].toInt();
        if (si < 0 || oi < 0 || si >= count || oi >= count) continue;
        const float* s = boxes.data() + si * 4;
        const float* o = boxes.data() + oi * 4;
        const QPointF a((s[0] + s[2]) * .5, (s[1] + s[3]) * .5),
                t((o[0] + o[2]) * .5, (o[1] + o[3]) * .5);
        QPointF d = t - a;
        const double len = std::hypot(d.x(), d.y());
        if (len < 1) continue;
        const QPointF n(-d.y() / len, d.x() / len);
        const double bow = (shortSide / 40.0 + shortSide / 30.0 * (i % 3)) *
                           (i % 2 == 0 ? 1 : -1);
        const QPointF ctrl = (a + t) * .5 + n * bow;
        const auto at = [&](double u) {
            return (1 - u) * (1 - u) * a + 2 * (1 - u) * u * ctrl + u * u * t;
        };
        QPainterPath path(a);
        for (int k = 1; k <= 24; ++k) path.lineTo(at(k / 24.0));
        const double edgeAlpha = std::clamp(
                e.contains("alpha") ? e["alpha"].toDouble() : 1.0, 0.0, 1.0);
        QColor c = objColor(si);
        c.setAlphaF(edgeAlpha);
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(QColor(0, 0, 0, int(150 * edgeAlpha)),
                            lineWidth + 2.5, Qt::SolidLine, Qt::RoundCap));
        painter.drawPath(path);
        painter.setPen(QPen(c, lineWidth + .5, Qt::SolidLine, Qt::RoundCap));
        painter.drawPath(path);
        QPointF v = at(1.0) - at(.9);
        v /= std::max(1e-6, std::hypot(v.x(), v.y()));
        const QPointF w(-v.y(), v.x());
        const double head = std::max(10.0, shortSide / 32.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(c);
        painter.drawPolygon(QPolygonF({t, t - v * head + w * head * .45,
                                       t - v * head - w * head * .45}));
        const QString text = e["predicate"].toString() + "  " +
                             QString::number(e["score"].toDouble(), 'f', 2);
        const QPointF mid = at(kChipAt[i % 8]);
        layout.chip(
                text,
                mid - QPointF(QFontMetrics(font).horizontalAdvance(text) / 2.0 +
                                      5,
                              chipHeight * .5),
                QColor(22, 24, 28, 225), edgeAlpha);
    }
    painter.end();
    return out;
}
// Binary planes supplied by the user, shown with hard edges (no blur needed).
std::vector<ObjectMask> suppliedMasks(const QByteArray& planes,
                                      const std::vector<float>& boxes,
                                      const QSize& size,
                                      int count) {
    std::vector<ObjectMask> out;
    out.resize(size_t(count));
    const size_t plane = size_t(size.width()) * size_t(size.height());
    if (!plane || size_t(planes.size()) != plane * size_t(count)) return out;
    for (int i = 0; i < count; ++i)
        out[i] =
                buildMask(reinterpret_cast<const uint8_t*>(planes.constData()) +
                                  size_t(i) * plane,
                          size.width(), size.height(), size_t(size.width()),
                          boxes.data() + i * 4, size, 1);
    return out;
}
// Per-video state: persistent tracks and relation beliefs (upstream
// render_video.py). Recreated when the tracking parameters or frame size
// change, and dropped by release() / resetTemporal().
struct TemporalState {
    RelateTemporal::Params params;
    int width, height;
    RelateTemporal::Tracker tracker;
    std::unique_ptr<RelateTemporal::EdgeBook> book;
    std::map<int, ObjectMask> masks;  // latest mask per track id
    std::set<int> maskSeen;           // tracks that ever carried a mask
    TemporalState(const RelateTemporal::Params& p, int w, int h)
        : params(p), width(w), height(h), tracker(p) {}
};
struct TemporalOut {
    QJsonArray objects, relations, spatial, semantic;
    QImage annotated;
    aicore_pipeline_timings timings{};
    double forwardMs = 0, renderMs = 0;
    int tracks = 0;
};
// One video frame: detect -> relate -> track -> filter beliefs -> draw the
// smoothed scene. Same order as upstream render_video.py (observe, observe,
// step, step). Every delivered frame is inferred (consumer-driven pacing), so
// upstream's --stride is effectively 1 here.
TemporalOut temporalFrame(std::shared_ptr<void>& slot,
                          aicore_relation_ctx* ctx,
                          const aicore_image_view& view,
                          const QImage& image,
                          const Detected& det,
                          const RelateAnythingSettings& s) {
    auto* state = static_cast<TemporalState*>(slot.get());
    if (!state || !state->params.same(s.tracking) ||
        state->width != image.width() || state->height != image.height()) {
        slot = std::make_shared<TemporalState>(s.tracking, image.width(),
                                               image.height());
        state = static_cast<TemporalState*>(slot.get());
    }
    TemporalOut out;
    const int n = det.labels.size();
    std::vector<RelateTemporal::Box> boxes;
    std::vector<std::string> labels;
    for (int i = 0; i < n; ++i) {
        const float* b = det.boxes.data() + i * 4;
        boxes.push_back({b[0], b[1], b[2], b[3]});
        labels.push_back(det.labels[i].toStdString());
    }
    const std::vector<float>& confs = det.scores;

    std::unique_ptr<aicore_relation_result,
                    decltype(&aicore_relation_result_free)>
            result(nullptr, aicore_relation_result_free);
    RelateTemporal::RawFrame frame;
    if (n >= 2) {
        auto request = aicore_relation_default_request();
        request.boxes_xyxy = det.boxes.data();
        request.object_count = size_t(n);
        request.box_scores = nullptr;
        aicore_relation_result* raw = nullptr;
        const int rc = aicore_relation_infer_image(ctx, &view, &request, &raw);
        result.reset(raw);
        check(rc == 0, QString::fromUtf8(aicore_relation_last_error(ctx)));
        out.timings = raw->timings;
        out.forwardMs = raw->forward_ms;
        float calA = 1.f, calB = 0.f;
        aicore_relation_calibration(ctx, &calA, &calB);
        frame.predLogits = raw->predicate_logits;
        frame.pairLogits = raw->pair_logits;
        frame.sub = raw->subject_indices;
        frame.obj = raw->object_indices;
        frame.valid = raw->valid_mask;
        frame.K = raw->pair_count;
        frame.V = raw->predicate_count;
        frame.calA = calA;
        frame.calB = calB;
        if (!state->book && raw->predicate_names) {
            std::vector<std::string> names;
            std::vector<std::uint8_t> spatial;
            for (size_t i = 0; i < raw->predicate_count; ++i) {
                names.emplace_back(raw->predicate_names[i]);
                spatial.push_back(
                        raw->predicate_spatial ? raw->predicate_spatial[i] : 0);
            }
            state->book = std::make_unique<RelateTemporal::EdgeBook>(
                    state->params, std::move(names), std::move(spatial));
        }
    }
    // Detections are never compacted: relation indices address the detector's
    // own output order, which is also the order the tracker sees.
    // The tracker owns --merge_iou and --max_box_frac (render_video.py);
    // rejected rows stay in the relation head's input and are only skipped
    // when mapping relations onto tracks, so detection indices never shift.
    const auto det2track = state->tracker.observe(
            boxes, labels, confs, double(image.width()) * image.height(),
            s.maxBoxFraction);
    if (state->book) state->book->observe(det2track, confs, frame);
    std::set<int> matched;
    for (const auto& kv : det2track) {
        matched.insert(kv.second);
        if (kv.first < int(det.masks.size()) &&
            !det.masks[size_t(kv.first)].alpha.isNull()) {
            state->masks[kv.second] = det.masks[size_t(kv.first)];
            state->maskSeen.insert(kv.second);
        }
    }
    // A coasting track has no detection this frame; a stale mask drifts off
    // the object, so drop it (the track keeps fading on its own alpha).
    for (auto it = state->masks.begin(); it != state->masks.end();)
        it = matched.count(it->first) ? std::next(it) : state->masks.erase(it);
    state->tracker.step(image.width(), image.height());
    std::set<int> live;
    for (const auto& t : state->tracker.tracks) live.insert(t.tid);
    if (state->book) state->book->step(live);
    out.tracks = int(state->tracker.tracks.size());

    // Draw only regions that carry a visible relation, at their smoothed box.
    std::vector<RelateTemporal::Edge> visible;
    if (state->book) visible = state->book->visible();
    std::set<int> carrying;
    for (const auto& e : visible) {
        carrying.insert(e.sub);
        carrying.insert(e.obj);
    }
    std::map<int, int> indexOf;
    std::vector<float> dBoxes;
    QStringList dLabels;
    std::vector<ObjectMask> dMasks;
    SceneStyle style;
    for (const auto& t : state->tracker.tracks) {
        if (!carrying.count(t.tid)) continue;
        const auto b = t.box();
        indexOf[t.tid] = int(dLabels.size());
        dBoxes.insert(dBoxes.end(), b.begin(), b.end());
        dLabels << QString::fromStdString(t.label());
        const auto m = state->masks.find(t.tid);
        dMasks.push_back(m != state->masks.end() ? m->second : ObjectMask{});
        style.alpha.push_back(t.alpha);
        style.colorId.push_back(t.tid);
        style.forceBox.push_back(!state->maskSeen.count(t.tid));
        out.objects.append(
                QJsonObject{{"index", int(dLabels.size()) - 1},
                            {"track", t.tid},
                            {"name", dLabels.back()},
                            {"score", t.conf},
                            {"box", QJsonArray{b[0], b[1], b[2], b[3]}}});
    }
    for (const auto& e : visible) {
        const auto si = indexOf.find(e.sub), oi = indexOf.find(e.obj);
        if (si == indexOf.end() || oi == indexOf.end()) continue;
        const QString pred = QString::fromStdString(state->book->name(e.pred));
        QJsonObject edge;
        edge["subject"] = int(si->second);
        edge["object"] = int(oi->second);
        edge["subject_track"] = int(e.sub);
        edge["object_track"] = int(e.obj);
        edge["predicate_index"] = int(e.pred);
        edge["subject_name"] =
                QString(dLabels[si->second] + QStringLiteral(" #") +
                        QString::number(si->second));
        edge["object_name"] =
                QString(dLabels[oi->second] + QStringLiteral(" #") +
                        QString::number(oi->second));
        edge["predicate"] = pred;
        edge["type"] = QString::fromLatin1(e.spatial ? "spatial" : "semantic");
        edge["alpha"] = double(e.alpha);
        edge["score"] = double(std::max(e.shown, e.score));
        out.relations.append(edge);
        (e.spatial ? out.spatial : out.semantic).append(edge);
    }
    QElapsedTimer render;
    render.start();
    out.annotated =
            renderScene(image, dBoxes, dLabels, dMasks, out.relations, style);
    out.renderMs = render.nsecsElapsed() / 1e6;
    return out;
}
}  // namespace
bool RelateAnythingSettings::sameSession(
        const RelateAnythingSettings& s) const {
    return model == s.model && detector == s.detector && device == s.device &&
           threads == s.threads && maxBoxes == s.maxBoxes && topk == s.topk &&
           threshold == s.threshold && detIou == s.detIou &&
           detectorClasses == s.detectorClasses &&
           detectorTextModel == s.detectorTextModel &&
           detectorIsRfdetr == s.detectorIsRfdetr && strict == s.strict &&
           (decompose || temporal) == (s.decompose || s.temporal) &&
           predicates == s.predicates && student == s.student &&
           tokenizer == s.tokenizer && regions == s.regions &&
           masks == s.masks && vocabulary == s.vocabulary &&
           vocabularyNames == s.vocabularyNames &&
           predicateTypes == s.predicateTypes;
}
RelateAnythingWorker::RelateAnythingWorker()
    : m_cancel(aicore_cancel_token_new()) {}
RelateAnythingWorker::~RelateAnythingWorker() {
    release();
    aicore_cancel_token_free(m_cancel);
}
void RelateAnythingWorker::resetTemporal() { m_temporal.reset(); }
void RelateAnythingWorker::release() {
    m_temporal.reset();
    aicore_relation_free(m_context);
    m_context = nullptr;
    aicore_yolo_free(m_detector);
    m_detector = nullptr;
    aicore_rfdetr_free(m_rfdetr);
    m_rfdetr = nullptr;
    m_regionBoxes.clear();
    m_regionScores.clear();
    m_regionLabels.clear();
    m_regionMasks.clear();
    m_regionSize = {};
    m_sessionFiles.clear();
}
void RelateAnythingWorker::requestCancel() {
    aicore_cancel_token_request(m_cancel);
}
void RelateAnythingWorker::resetCancel() {
    aicore_cancel_token_reset(m_cancel);
}
void RelateAnythingWorker::process(const RelateAnythingSettings& settings,
                                   QImage image,
                                   const QString& source,
                                   quint64 generation) {
    Q_ASSERT(QThread::currentThread() == thread());
    RelateAnythingResult out;
    out.source = source;
    out.generation = generation;
    QElapsedTimer wall;
    wall.start();
    aicore::runtime::CancelScope scope(m_cancel);
    const auto device = settings.device.toUtf8();
    const bool locked = aicore_device_task_lock_cancelable(device.constData(),
                                                           m_cancel) == 0;
    const double waitMs = wall.nsecsElapsed() / 1e6;
    struct Unlock {
        bool locked;
        ~Unlock() {
            if (locked) aicore_device_task_unlock();
        }
    } unlock{locked};
    try {
        check(locked && !aicore_cancel_requested(), tr("Cancelled."));
        const bool supplied = !settings.regions.isEmpty();
        bool reused = m_context && (supplied || m_detector || m_rfdetr) &&
                      m_settings.sameSession(settings);
        if (reused) {
            for (auto file = m_sessionFiles.cbegin();
                 file != m_sessionFiles.cend(); ++file) {
                if (fileGeneration(file.key()) != file.value()) {
                    reused = false;
                    break;
                }
            }
        }
        QElapsedTimer timer;
        timer.start();
        if (!reused) {
            release();
            auto remember = [&](const QString& path) {
                if (!path.isEmpty())
                    m_sessionFiles.insert(path, fileGeneration(path));
            };
            remember(settings.model);
            remember(supplied ? settings.regions : settings.detector);
            if (!supplied) remember(settings.detectorTextModel);
            remember(settings.vocabulary);
            remember(settings.vocabularyNames);
            remember(settings.predicateTypes);
            if (settings.vocabulary.isEmpty() &&
                !settings.predicates.trimmed().isEmpty()) {
                remember(settings.student);
                remember(settings.tokenizer);
            }
            if (supplied) {
                QFile file(settings.regions);
                check(file.open(QIODevice::ReadOnly),
                      tr("Cannot open object boxes: %1").arg(settings.regions));
                QJsonParseError error;
                const auto document =
                        QJsonDocument::fromJson(file.readAll(), &error);
                check(error.error == QJsonParseError::NoError,
                      tr("Invalid object-box JSON: %1")
                              .arg(error.errorString()));
                const auto metadata = document.object();
                const auto boxes = document.isArray()
                                           ? document.array()
                                           : metadata["boxes"].toArray();
                const auto labels = metadata["labels"].toArray();
                const auto scores = metadata["scores"].toArray();
                check(boxes.size() >= 2 && boxes.size() <= settings.maxBoxes,
                      tr("Supply between two and %1 object boxes.")
                              .arg(settings.maxBoxes));
                check((labels.isEmpty() || labels.size() == boxes.size()) &&
                              (scores.isEmpty() ||
                               scores.size() == boxes.size()),
                      tr("Object labels and scores must match the box count."));
                for (int i = 0; i < boxes.size(); ++i) {
                    const auto box = boxes[i].toArray();
                    check(box.size() == 4,
                          tr("Every box must contain four pixel coordinates."));
                    for (const auto& value : box) {
                        check(value.isDouble() &&
                                      std::isfinite(value.toDouble()),
                              tr("Box coordinates must be finite numbers."));
                        m_regionBoxes.push_back(float(value.toDouble()));
                    }
                    check(m_regionBoxes[i * 4 + 2] > m_regionBoxes[i * 4] &&
                                  m_regionBoxes[i * 4 + 3] >
                                          m_regionBoxes[i * 4 + 1],
                          tr("Box corners must be ordered left, top, right, "
                             "bottom."));
                    m_regionLabels << (labels.isEmpty() ? tr("Object")
                                                        : labels[i].toString());
                    if (!scores.isEmpty()) {
                        check(scores[i].isDouble() &&
                                      std::isfinite(scores[i].toDouble()) &&
                                      scores[i].toDouble() >= 0 &&
                                      scores[i].toDouble() <= 1,
                              tr("Object confidence must be between zero and "
                                 "one."));
                        m_regionScores.push_back(float(scores[i].toDouble()));
                    }
                }
                const auto size = metadata["image_size"].toArray();
                if (!size.isEmpty()) {
                    check(size.size() == 2 && size[0].toInt() > 0 &&
                                  size[1].toInt() > 0,
                          tr("image_size must contain positive width and "
                             "height."));
                    m_regionSize = QSize(size[0].toInt(), size[1].toInt());
                }
                auto maskPath = settings.masks;
                if (maskPath.isEmpty() && metadata["masks"].isString())
                    maskPath = QFileInfo(settings.regions)
                                       .dir()
                                       .filePath(metadata["masks"].toString());
                if (!maskPath.isEmpty()) {
                    remember(maskPath);
                    QFile masks(maskPath);
                    check(masks.open(QIODevice::ReadOnly),
                          tr("Cannot open object masks: %1").arg(maskPath));
                    check(masks.size() > 0 &&
                                  masks.size() <=
                                          std::numeric_limits<int>::max(),
                          tr("Object mask file is empty or too large."));
                    m_regionMasks = masks.readAll();
                    check(m_regionMasks.size() == masks.size(),
                          tr("Could not read all object masks."));
                }
            }
            for (int i = 0; i < aicore_relation_model_count(); ++i) {
                const auto* e = aicore_relation_model_at(i);
                for (const auto& path : {settings.model,
                                         settings.predicates.trimmed().isEmpty()
                                                 ? QString()
                                                 : settings.student,
                                         settings.predicates.trimmed().isEmpty()
                                                 ? QString()
                                                 : settings.tokenizer}) {
                    if (!path.isEmpty() &&
                        QFileInfo(path).fileName() ==
                                QString::fromUtf8(e->filename))
                        verify(path,
                               {QCryptographicHash::Sha256,
                                QByteArray(e->sha256)},
                               e->size_bytes,
                               e->role != AICORE_RELATION_MODEL_TOKENIZER);
                }
            }
            if (!supplied) {
                verify(settings.detector,
                       {QCryptographicHash::Sha256,
                        ecvAssetIntegrity::PinnedDigest(
                                QFileInfo(settings.detector).fileName())});
            }
            if (!supplied && settings.detectorIsRfdetr) {
                std::unique_ptr<aicore_rfdetr_options,
                                decltype(&aicore_rfdetr_options_free)>
                        rfOpts(aicore_rfdetr_options_new(),
                               aicore_rfdetr_options_free);
                check(bool(rfOpts),
                      tr("Could not allocate detection options."));
                aicore_rfdetr_options_set_device(rfOpts.get(),
                                                 device.constData());
                aicore_rfdetr_options_set_threads(rfOpts.get(),
                                                  settings.threads);
                auto detectorPath = settings.detector.toUtf8();
                m_rfdetr = aicore_rfdetr_load_opts(detectorPath.constData(),
                                                   rfOpts.get());
                check(aicore_rfdetr_is_ready(m_rfdetr),
                      m_rfdetr ? QString::fromUtf8(
                                         aicore_rfdetr_last_error(m_rfdetr))
                               : tr("Could not load the detector."));
            } else if (!supplied) {
                std::unique_ptr<aicore_yolo_options,
                                decltype(&aicore_yolo_options_free)>
                        detectorOpts(aicore_yolo_options_new(),
                                     aicore_yolo_options_free);
                check(bool(detectorOpts),
                      tr("Could not allocate detection options."));
                aicore_yolo_options_set_device(detectorOpts.get(),
                                               device.constData());
                aicore_yolo_options_set_threads(detectorOpts.get(),
                                                settings.threads);
                aicore_yolo_options_set_conf_thres(detectorOpts.get(),
                                                   settings.threshold);
                aicore_yolo_options_set_iou_thres(detectorOpts.get(),
                                                  settings.detIou);
                // Leave headroom above maxBoxes: the best maxBoxes are kept
                // by score below, matching upstream's max_boxes semantics.
                aicore_yolo_options_set_top_k(
                        detectorOpts.get(),
                        uint32_t(std::max(settings.maxBoxes * 4, 100)));
                // Enable bilinear logit upsampling for smooth mask edges.
                aicore_yolo_options_set_mask_upsample(detectorOpts.get(), 1);
                // Open-vocabulary detectors (YOLO-World, YOLOE with a class
                // list) bake the vocabulary through their text encoder.
                QList<QByteArray> classNames;
                std::vector<const char*> classPointers;
                QByteArray textModel;
                if (!settings.detectorClasses.isEmpty()) {
                    verify(settings.detectorTextModel,
                           {QCryptographicHash::Sha256,
                            ecvAssetIntegrity::PinnedDigest(
                                    QFileInfo(settings.detectorTextModel)
                                            .fileName())});
                    for (const auto& name : settings.detectorClasses)
                        classNames << name.toUtf8();
                    for (const auto& name : classNames)
                        classPointers.push_back(name.constData());
                    textModel = settings.detectorTextModel.toUtf8();
                    aicore_yolo_options_set_classes(
                            detectorOpts.get(), classPointers.data(),
                            int32_t(classPointers.size()));
                    aicore_yolo_options_set_text_model(detectorOpts.get(),
                                                       textModel.constData());
                }
                auto detectorPath = settings.detector.toUtf8();
                m_detector = aicore_yolo_load_opts(detectorPath.constData(),
                                                   detectorOpts.get());
                check(aicore_yolo_is_ready(m_detector),
                      QString::fromUtf8(aicore_yolo_last_error(m_detector)));
            }
            auto model = settings.model.toUtf8(),
                 student = settings.student.toUtf8(),
                 tokenizer = settings.tokenizer.toUtf8(),
                 vocabulary = settings.vocabulary.toUtf8(),
                 vocabularyNames = settings.vocabularyNames.toUtf8(),
                 predicateTypes = settings.predicateTypes.toUtf8();
            QList<QByteArray> names;
            for (const auto& name :
                 settings.predicates.split('|', Qt::SkipEmptyParts))
                if (!name.trimmed().isEmpty()) names << name.trimmed().toUtf8();
            std::vector<const char*> pointers;
            for (const auto& name : names) pointers.push_back(name.constData());
            auto o = aicore_relation_default_options();
            o.model_path = model.constData();
            o.device = device.constData();
            o.threads = settings.threads;
            o.max_boxes = settings.maxBoxes;
            o.topk = settings.topk;
            o.strict_f32 = settings.strict;
            // Temporal mode needs the decomposed streams (spatial flags) and
            // the raw logits for the belief filter; both are cheap.
            o.decompose = settings.decompose || settings.temporal;
            o.raw_logits = 1;
            if (!vocabulary.isEmpty())
                o.vocabulary_path = vocabulary.constData();
            if (!vocabularyNames.isEmpty())
                o.vocabulary_names_path = vocabularyNames.constData();
            if (!predicateTypes.isEmpty())
                o.predicate_type_map_path = predicateTypes.constData();
            if (!pointers.empty()) {
                o.text_student_path = student.constData();
                o.tokenizer_path = tokenizer.constData();
                o.predicate_texts = pointers.data();
                o.predicate_text_count = pointers.size();
            }
            m_context = aicore_relation_create(&o);
            check(aicore_relation_ready(m_context),
                  QString::fromUtf8(aicore_relation_last_error(m_context)));
            m_settings = settings;
        }
        const double initMs = reused ? 0 : timer.nsecsElapsed() / 1e6;
        check(!aicore_cancel_requested(), tr("Cancelled."));
        timer.restart();
        if (image.isNull()) image.load(source);
        auto view = ecvAICoreRuntime::makeImageView(image);
        if (!view.data && !image.isNull()) {
            image = image.convertToFormat(QImage::Format_RGB888);
            view = ecvAICoreRuntime::makeImageView(image);
        }
        check(view.data, tr("Cannot decode image: %1").arg(source));
        const double prepareMs = timer.nsecsElapsed() / 1e6;
        timer.restart();
        double detectMs = 0;
        Detected detected;
        if (!supplied) {
            detected = m_rfdetr ? detectRfdetr(m_rfdetr, view, image.size(),
                                               settings)
                                : detectYolo(m_detector, view, image.size(),
                                             settings);
            detectMs = timer.nsecsElapsed() / 1e6;
        } else {
            check(m_regionSize.isEmpty() || m_regionSize == image.size(),
                  tr("Object-box image size does not match this image."));
            check(m_regionMasks.isEmpty() ||
                          size_t(m_regionMasks.size()) ==
                                  m_regionLabels.size() *
                                          size_t(image.width()) *
                                          size_t(image.height()),
                  tr("Masks must contain one width × height byte plane per "
                     "object."));
        }
        const std::vector<float>& boxes =
                supplied ? m_regionBoxes : detected.boxes;
        const std::vector<float>& scores =
                supplied ? m_regionScores : detected.scores;
        const QStringList& labels = supplied ? m_regionLabels : detected.labels;
        // Video / camera frames of a detector run use the ported upstream
        // tracker + relation-belief filter; images and batches stay
        // independent.
        const bool temporalMode = settings.temporal && !supplied;
        TemporalOut temporalOut;
        if (temporalMode)
            temporalOut = temporalFrame(m_temporal, m_context, view, image,
                                        detected, settings);
        else
            m_temporal.reset();
        QJsonArray objects;
        const int count = labels.size();
        const uint8_t* maskData =
                reinterpret_cast<const uint8_t*>(m_regionMasks.constData());
        const size_t maskBytes = supplied ? size_t(m_regionMasks.size()) : 0;
        const std::vector<ObjectMask> displayMasks =
                supplied ? suppliedMasks(m_regionMasks, boxes, image.size(),
                                         count)
                         : std::move(detected.masks);
        for (int i = 0; i < count && !temporalMode; ++i) {
            const float* b = boxes.data() + i * 4;
            objects.append(QJsonObject{
                    {"index", i},
                    {"name", labels[i]},
                    {"score",
                     scores.empty() ? QJsonValue() : QJsonValue(scores[i])},
                    {"box", QJsonArray{b[0], b[1], b[2], b[3]}}});
        }
        QJsonArray relations, spatial, semantic;
        aicore_pipeline_timings timings{};
        double forwardMs = 0;
        if (temporalMode) {
            objects = temporalOut.objects;
            relations = temporalOut.relations;
            spatial = temporalOut.spatial;
            semantic = temporalOut.semantic;
            timings = temporalOut.timings;
            forwardMs = temporalOut.forwardMs;
        }
        if (!temporalMode && count >= 2) {
            auto request = aicore_relation_default_request();
            request.boxes_xyxy = boxes.data();
            request.object_count = count;
            // No box_scores: the C++ path multiplies them into the reported
            // score, but upstream keeps detector confidence out of the score
            // and the threshold (it is a ranking hint only).
            request.box_scores = nullptr;
            // The official graph reads image + boxes only; masks detected here
            // are for display. User-supplied masks still reach the head.
            if (supplied && maskBytes) {
                request.masks = maskData;
                request.mask_bytes = maskBytes;
            }
            aicore_relation_result* raw = nullptr;
            const int rc = aicore_relation_infer_image(m_context, &view,
                                                       &request, &raw);
            std::unique_ptr<aicore_relation_result,
                            decltype(&aicore_relation_result_free)>
                    result(raw, aicore_relation_result_free);
            check(rc == 0,
                  QString::fromUtf8(aicore_relation_last_error(m_context)));
            const bool decomposed =
                    raw->spatial_count > 0 || raw->semantic_count > 0;
            float calA = 1.0f, calB = 0.0f;
            if (decomposed)
                aicore_relation_calibration(m_context, &calA, &calB);
            for (const auto& p :
                 pickEdges(*raw, settings, decomposed, calA, calB, scores)) {
                const auto& e = p.edge;
                check(e.subject < size_t(labels.size()) &&
                              e.object < size_t(labels.size()) &&
                              e.predicate < raw->predicate_count,
                      tr("Invalid relation indices."));
                const QJsonObject edge{
                        {"subject", int(e.subject)},
                        {"object", int(e.object)},
                        {"predicate_index", int(e.predicate)},
                        {"subject_name", QString(labels[int(e.subject)] + " #" +
                                                 QString::number(e.subject))},
                        {"object_name", QString(labels[int(e.object)] + " #" +
                                                QString::number(e.object))},
                        {"predicate",
                         QString::fromUtf8(raw->predicate_names[e.predicate])},
                        {"type", !decomposed ? "relation"
                                 : p.spatial ? "spatial"
                                             : "semantic"},
                        {"score", e.score}};
                relations.append(edge);
                if (decomposed) (p.spatial ? spatial : semantic).append(edge);
            }
            timings = raw->timings;
            forwardMs = raw->forward_ms;
        }
        check(!aicore_cancel_requested(), tr("Cancelled."));
        out.image = image;
        out.device =
                QString::fromUtf8(aicore_relation_context_device(m_context));
        timer.restart();
        if (temporalMode)
            out.annotated = temporalOut.annotated;
        else
            out.annotated =
                    renderScene(image, boxes, labels, displayMasks, relations);
        const double renderMs = temporalMode ? temporalOut.renderMs
                                             : timer.nsecsElapsed() / 1e6;
        out.e2eMs = wall.nsecsElapsed() / 1e6;
        out.json = {{"source", source},
                    {"model", QFileInfo(settings.model).fileName()},
                    {"object_source", supplied ? "supplied" : "detection"},
                    {"temporal", temporalMode},
                    {"tracks", temporalOut.tracks},
                    {"detector",
                     supplied ? QString()
                              : QFileInfo(settings.detector).fileName()},
                    {"device", out.device},
                    {"objects", objects},
                    {"relations", relations},
                    {"spatial", spatial},
                    {"semantic", semantic},
                    {"timings",
                     QJsonObject{{"valid_fields", int(timings.valid_fields)},
                                 {"preprocess_ms", timings.preprocess_ms},
                                 {"relation_e2e_ms", timings.e2e_ms},
                                 {"relation_forward_ms", forwardMs},
                                 {"detection_ms", detectMs},
                                 {"device_wait_ms", waitMs},
                                 {"context_init_ms", initMs},
                                 {"context_reused", reused},
                                 {"image_prepare_ms", prepareMs},
                                 {"visualization_ms", renderMs},
                                 {"worker_wall_ms", out.e2eMs}}}};
        out.ok = true;
    } catch (const std::exception& e) {
        out.ok = false;
        out.error = QString::fromUtf8(e.what());
        if (out.error.isEmpty())
            out.error = tr("Inference failed with empty error message.");
        const bool sessionInvalid = !aicore_relation_ready(m_context) ||
                                    (settings.regions.isEmpty() && !m_rfdetr &&
                                     !aicore_yolo_is_ready(m_detector));
        out.fatal = sessionInvalid;
        if (sessionInvalid) release();
    } catch (...) {
        out.ok = false;
        out.fatal = true;
        out.error = tr("Unexpected relation processing error.");
        release();
    }
    emit completed(out);
}
