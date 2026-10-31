// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationCanvas.h"

#include <QBuffer>
#include <QHash>
#include <QJsonArray>
#include <QKeyEvent>
#include <QLineF>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPainterPathStroker>
#include <QPolygonF>
#include <QStringList>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "AnnotationConfig.h"
#include "MaskPolygon.h"
#include "QtCompat.h"

namespace {
bool isCuboid(const AnnotationShape& shape) {
    return shape.type == QStringLiteral("cuboid") && shape.points.size() == 8;
}

void syncCuboidDepth(AnnotationShape& shape) {
    if (!isCuboid(shape)) return;
    QJsonObject metadata =
            shape.extra.value(QStringLiteral("cuboid3d")).toObject();
    const QPointF depth = shape.points[4] - shape.points[0];
    metadata.insert(QStringLiteral("depth_vector"),
                    QJsonArray{depth.x(), depth.y()});
    shape.extra.insert(QStringLiteral("cuboid3d"), metadata);
}

void drawCuboid(QPainter& painter, const QVector<QPointF>& points) {
    painter.save();
    const QPen frontPen = painter.pen();
    painter.drawPath(
            annotationCuboid::facePath(points, annotationCuboid::Face::Front));
    painter.setBrush(Qt::NoBrush);
    for (int i = 0; i < 4; ++i) painter.drawLine(points[i], points[i + 4]);
    QPen backPen = frontPen;
    backPen.setStyle(Qt::DashLine);
    painter.setPen(backPen);
    painter.drawPath(
            annotationCuboid::facePath(points, annotationCuboid::Face::Back));
    QPen orientationPen = frontPen;
    orientationPen.setColor(QColor(255, 165, 0, 220));
    painter.setPen(orientationPen);
    painter.drawLine(points[0], points[1]);
    painter.restore();
}

QRectF circleBounds(const AnnotationShape& shape) {
    if (shape.points.size() < 2) return {};
    const double radius = QLineF(shape.points[0], shape.points[1]).length();
    return QRectF(shape.points[0] - QPointF(radius, radius),
                  QSizeF(radius * 2, radius * 2));
}

QRectF shapeBounds(const AnnotationShape& shape) {
    return shape.type == QStringLiteral("circle")
                   ? circleBounds(shape)
                   : QPolygonF(shape.points).boundingRect();
}

QPen annotationOutlinePen(const AnnotationShape& shape,
                          const QColor& color,
                          double width) {
    QPen pen(color, width);
    if (shape.type != QStringLiteral("point") &&
        shape.extra.value(QStringLiteral("difficult")).toBool())
        pen.setStyle(Qt::DashLine);
    return pen;
}

void drawAnnotationVertex(QPainter& painter,
                          const AnnotationShape& shape,
                          const QPointF& point,
                          double radius) {
    QPainterPath path;
    if (shape.type == QStringLiteral("point") &&
        shape.extra.value(QStringLiteral("difficult")).toBool()) {
        const double extent = radius * 1.5;
        path.moveTo(point + QPointF(0, -extent));
        path.lineTo(point + QPointF(-extent, extent));
        path.lineTo(point + QPointF(extent, extent));
        path.closeSubpath();
        // Source round difficult vertices append the triangle twice.
        const QPainterPath triangle = path;
        path.addPath(triangle);
    } else {
        path.addEllipse(point, radius, radius);
    }
    const QBrush fill = painter.brush();
    painter.save();
    painter.setBrush(Qt::NoBrush);
    painter.drawPath(path);
    painter.fillPath(path, fill);
    painter.restore();
}

void drawGeometryMask(QPainter& painter,
                      const AnnotationShape& shape,
                      QColor color,
                      double width,
                      int opacity,
                      bool fill) {
    if (!shape.mask.isNull() || shape.points.size() < 2) return;
    QPainterPath path;
    if (shape.type == QStringLiteral("circle")) {
        path.addEllipse(circleBounds(shape));
    } else if (shape.type == QStringLiteral("polygon") ||
               shape.type == QStringLiteral("rectangle") ||
               shape.type == QStringLiteral("rotation") ||
               shape.type == QStringLiteral("quadrilateral")) {
        if (shape.points.size() < 3) return;
        path.addPolygon(QPolygonF(shape.points));
        path.closeSubpath();
    } else
        return;
    painter.save();
    if (fill) {
        QColor maskColor = color;
        maskColor.setAlpha(opacity);
        painter.fillPath(path, maskColor);
    }
    QPen pen(color, width);
    if (shape.extra.value(QStringLiteral("difficult")).toBool())
        pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawPath(path);
    painter.restore();
}

void drawRotationDirection(QPainter& painter,
                           const AnnotationShape& shape,
                           double scale,
                           int pointSize,
                           bool degrees) {
    if (shape.type != QStringLiteral("rotation") || shape.points.size() != 4)
        return;
    const QPointF center = (shape.points[0] + shape.points[2]) / 2;
    const double size = pointSize / scale;
    painter.save();
    if (degrees) {
        double angle = std::atan2(shape.points[1].y() - shape.points[0].y(),
                                  shape.points[1].x() - shape.points[0].x()) *
                       180 / 3.141592653589793;
        angle = std::fmod(angle + 360.0, 360.0);
        if (std::abs(angle - 360.0) < .1) angle = 0;
        const QString text = QString::number(angle, 'f', 2) + QChar(0x00B0);
        painter.setFont(QFont(QStringLiteral("Arial"),
                              std::max(6, int(std::nearbyint(8.0 / scale)))));
        const QRect bounds = QFontMetrics(painter.font()).boundingRect(text);
        painter.fillRect(QRect(int(bounds.x() + center.x() - size),
                               int(bounds.y() + center.y() + size),
                               bounds.width(), bounds.height()),
                         QColor(255, 153, 0));
        painter.setPen(Qt::white);
        painter.drawText(QPoint(int(center.x() - size), int(center.y() + size)),
                         text);
    } else {
        QPainterPath path;
        path.addRect(QRectF(int(center.x() - size / 2),
                            int(center.y() - size / 2), int(size), int(size)));
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
        painter.fillPath(path, QColor(255, 153, 0));
    }
    painter.restore();
}

// Export and interaction share device-pixel layout; draw all leaders before
// labels. Scratch is O(visible labels), anchor lookup O(total points); no
// pairwise collision pass.
void drawAnnotationLabels(QPainter& painter,
                          const AnnotationCanvas& canvas,
                          const QVector<AnnotationShape>& shapes,
                          const AnnotationCanvas::VisualizationOptions& options,
                          const QRect& bounds,
                          int excludedIndex = -1) {
    if (!options.showLabels && !options.showTexts) return;
    struct Label {
        QColor color;
        QRect box;
        QPoint textPosition;
        QString text;
        QPolygonF leader;
    };
    QVector<Label> labels;
    labels.reserve(shapes.size());
    const QTransform transform = painter.transform();
    painter.save();
    painter.setOpacity(1.0);
    painter.resetTransform();
    painter.setFont(
            QFont(QStringLiteral("Arial"),
                  annotationConfig::value("canvas/label_font_size").toInt()));
    const QFontMetrics metrics(painter.font());
    for (int index = 0; index < shapes.size(); ++index) {
        const auto& shape = shapes[index];
        if (index == excludedIndex || !shape.visible ||
            shape.points.isEmpty() ||
            shape.label == QStringLiteral("AUTOLABEL_OBJECT") ||
            shape.label == QStringLiteral("AUTOLABEL_ADD") ||
            shape.label == QStringLiteral("AUTOLABEL_REMOVE"))
            continue;
        QStringList fields;
        if (options.showLabels) {
            QString title = shape.label;
            const auto group = shape.extra.value(QStringLiteral("group_id"));
            if (group.isDouble()) {
                title += QStringLiteral(" (%1)").arg(group.toInt());
                title = title.trimmed();
            }
            if (!title.isEmpty()) fields.append(title);
            const auto score = shape.extra.value(QStringLiteral("score"));
            if (options.showScores && score.isDouble())
                fields.append(QString::number(score.toDouble(), 'f', 2));
        }
        const QString description =
                shape.extra.value(QStringLiteral("description")).toString();
        if (options.showTexts && !description.isEmpty())
            fields.append(description);
        const QString text = fields.join(QStringLiteral(" · "));
        if (text.isEmpty()) continue;
        const int width = metrics.tightBoundingRect(text).width() + 8;
        const int height = metrics.height() + 4;
        Label label;
        label.text = text;
        label.color = canvas.shapeColor(shape);
        if (shape.type == QStringLiteral("rectangle") ||
            shape.type == QStringLiteral("rotation")) {
            QPointF point = transform.map(shapeBounds(shape).topLeft());
            if (shape.type == QStringLiteral("rectangle")) point.ry() -= height;
            label.box = QRect(int(point.x()), int(point.y()), width, height);
            label.textPosition =
                    QPoint(int(point.x() + 4),
                           int(point.y() + height - 2 - metrics.descent()));
        } else if (shape.type == QStringLiteral("polygon") ||
                   shape.type == QStringLiteral("quadrilateral") ||
                   shape.type == QStringLiteral("cuboid") ||
                   shape.type == QStringLiteral("circle") ||
                   shape.type == QStringLiteral("line") ||
                   shape.type == QStringLiteral("linestrip") ||
                   shape.type == QStringLiteral("point")) {
            QPointF anchor = *std::min_element(
                    shape.points.cbegin(), shape.points.cend(),
                    [](const QPointF& a, const QPointF& b) {
                        return a.y() < b.y() ||
                               (a.y() == b.y() && a.x() < b.x());
                    });
            if (shape.type == QStringLiteral("circle")) {
                if (shape.points.size() < 2) continue;
                anchor = shape.points[0] -
                         QPointF(0, QLineF(shape.points[0], shape.points[1])
                                            .length());
            }
            const QPointF point = transform.map(anchor);
            double left = point.x() + 24;
            if (left + width > bounds.right()) left = point.x() - 24 - width;
            double top = point.y() - 24 - height;
            if (top < bounds.top()) top = point.y() + 24;
            left = std::max(double(bounds.left()),
                            std::min(left, double(bounds.right() - width + 1)));
            top = std::max(double(bounds.top()),
                           std::min(top, double(bounds.bottom() - height + 1)));
            label.box = QRect(int(left), int(top), width, height);
            label.textPosition =
                    QPoint(label.box.left() + 4,
                           label.box.bottom() + 1 - 2 - metrics.descent());
            const int endX = point.x() <= label.box.center().x()
                                     ? label.box.left()
                                     : label.box.right();
            const QPointF end(endX, label.box.center().y());
            const QPointF elbow(end.x() + (endX == label.box.left() ? -8 : 8),
                                end.y());
            label.leader << point << elbow << end;
        } else {
            continue;
        }
        labels.append(label);
    }
    for (const auto& label : labels) {
        if (label.leader.isEmpty()) continue;
        QColor color = label.color;
        color.setAlphaF(0.5);
        QPen pen(color, 1, Qt::DashLine);
        pen.setCosmetic(true);
        painter.setPen(pen);
        painter.drawPolyline(label.leader);
    }
    painter.setPen(Qt::NoPen);
    for (const auto& label : labels) {
        painter.setBrush(label.color);
        painter.drawRoundedRect(QRectF(label.box), 4.0, 4.0);
    }
    painter.setPen(QPen(Qt::black, 8, Qt::SolidLine));
    for (const auto& label : labels)
        painter.drawText(label.textPosition, label.text);
    painter.restore();
}

void drawAnnotationAttributes(QPainter& painter,
                              const AnnotationShape& shape,
                              double scale,
                              int pointSize) {
    const QJsonObject attributes =
            shape.extra.value(QStringLiteral("attributes")).toObject();
    if (attributes.isEmpty() || shape.points.isEmpty()) return;
    painter.save();
    painter.setOpacity(1.0);
    painter.setFont(QFont(QStringLiteral("Arial"),
                          std::max(8, int(std::nearbyint(10.0 / scale))),
                          QFont::Bold));
    QStringList lines;
    for (auto it = attributes.constBegin(); it != attributes.constEnd(); ++it) {
        QString value = it.value().toVariant().toString();
        if (it.value().isBool())
            value = it.value().toBool() ? QStringLiteral("True")
                                        : QStringLiteral("False");
        lines.append(it.key() + QStringLiteral(": ") + value);
    }
    const QFontMetrics metrics(painter.font());
    int width = 0;
    for (const QString& line : lines)
        width = std::max(width, metrics.tightBoundingRect(line).width());
    QPointF anchor;
    if (shape.type == QStringLiteral("circle") ||
        shape.type == QStringLiteral("point") ||
        shape.type == QStringLiteral("line") ||
        shape.type == QStringLiteral("linestrip")) {
        anchor = shape.points.first() + QPointF(pointSize / scale, 1);
    } else {
        anchor = QPolygonF(shape.points).boundingRect().bottomLeft() +
                 QPointF(0, 1);
    }
    const QRect box(int(anchor.x()), int(anchor.y()), width + 16,
                    lines.size() * metrics.height() + 4);
    painter.fillRect(
            box, annotationConfig::color("canvas/attributes/background_color"));
    painter.setPen(
            QPen(annotationConfig::color("canvas/attributes/border_color"), 1));
    painter.drawRect(box);
    painter.setPen(annotationConfig::color("canvas/attributes/text_color"));
    for (int i = 0; i < lines.size(); ++i)
        painter.drawText(
                QPoint(box.x() + 8,
                       box.y() + 2 + i * metrics.height() + metrics.ascent()),
                lines[i]);
    painter.restore();
}

bool closedType(const QString& type) {
    return type == QStringLiteral("polygon") ||
           type == QStringLiteral("rectangle") ||
           type == QStringLiteral("rotation") ||
           type == QStringLiteral("quadrilateral") ||
           type == QStringLiteral("circle") || type == QStringLiteral("cuboid");
}

QImage movedMask(const QImage& source, const QPointF& delta) {
    if (source.isNull()) return {};
    QImage shifted(source.size(), QImage::Format_ARGB32_Premultiplied);
    shifted.fill(Qt::transparent);
    QPainter painter(&shifted);
    painter.drawImage(QPoint(qRound(delta.x()), qRound(delta.y())), source);
    return shifted;
}

QVector<QPointF> rectanglePoints(const QPointF& a, const QPointF& b) {
    const QRectF r(a, b);
    const QRectF rect = r.normalized();
    return {rect.topLeft(), rect.topRight(), rect.bottomRight(),
            rect.bottomLeft()};
}

double normalizedAngle(double angle) {
    constexpr double fullTurn = 6.283185307179586;
    angle = std::fmod(angle, fullTurn);
    return angle < 0.0 ? angle + fullTurn : angle;
}

QPointF rotated(const QPointF& point, const QPointF& center, double radians) {
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    const QPointF delta = point - center;
    return center + QPointF(c * delta.x() - s * delta.y(),
                            s * delta.x() + c * delta.y());
}

double dot(const QPointF& a, const QPointF& b) {
    return a.x() * b.x() + a.y() * b.y();
}

double pointSegmentDistanceSquared(const QPointF& point,
                                   const QPointF& first,
                                   const QPointF& last) {
    const QPointF segment = last - first;
    const double lengthSquared = dot(segment, segment);
    if (lengthSquared <= std::numeric_limits<double>::epsilon())
        return dot(point - first, point - first);
    const double t =
            std::clamp(dot(point - first, segment) / lengthSquared, 0.0, 1.0);
    const QPointF projection = first + t * segment;
    return dot(point - projection, point - projection);
}

void simplifyRdp(const QVector<QPointF>& points,
                 int first,
                 int last,
                 double epsilonSquared,
                 QVector<bool>& keep) {
    if (last <= first + 1) return;
    double maximum = epsilonSquared;
    int split = -1;
    for (int index = first + 1; index < last; ++index) {
        const double distance = pointSegmentDistanceSquared(
                points[index], points[first], points[last]);
        if (distance > maximum) {
            maximum = distance;
            split = index;
        }
    }
    if (split < 0) return;
    keep[split] = true;
    simplifyRdp(points, first, split, epsilonSquared, keep);
    simplifyRdp(points, split, last, epsilonSquared, keep);
}

QVector<QPointF> simplifyPolygon(const QVector<QPointF>& points,
                                 double epsilon) {
    if (points.size() < 4 || epsilon <= 0.0) return points;
    QVector<bool> keep(points.size(), false);
    keep.front() = true;
    keep.back() = true;
    simplifyRdp(points, 0, points.size() - 1, epsilon * epsilon, keep);
    QVector<QPointF> simplified;
    simplified.reserve(points.size());
    for (int index = 0; index < points.size(); ++index)
        if (keep[index]) simplified.append(points[index]);
    return simplified.size() >= 3 ? simplified : points;
}

QVector<QPointF> maskOutline(const QImage& mask) {
    QVector<uint8_t> binary(mask.width() * mask.height());
    int left = mask.width(), top = mask.height(), right = -1, bottom = -1;
    for (int y = 0; y < mask.height(); ++y) {
        const auto* row = reinterpret_cast<const QRgb*>(mask.constScanLine(y));
        for (int x = 0; x < mask.width(); ++x) {
            if (qAlpha(row[x]) == 0) continue;
            binary[y * mask.width() + x] = 1;
            left = std::min(left, x);
            top = std::min(top, y);
            right = std::max(right, x);
            bottom = std::max(bottom, y);
        }
    }
    if (right < left) return {};
    QVector<QPointF> polygon =
            largestMaskPolygon(binary.constData(), mask.width(), mask.height(),
                               size_t(mask.width()));
    if (polygon.size() < 3)
        polygon = {QPointF(left, top), QPointF(right + 1, top),
                   QPointF(right + 1, bottom + 1), QPointF(left, bottom + 1)};
    return polygon;
}

struct LabColor {
    double l, a, b;
};

LabColor toLab(const uchar* rgb) {
    static const std::array<double, 256> linear = [] {
        std::array<double, 256> values{};
        for (int i = 0; i < 256; ++i) {
            const double c = i / 255.0;
            values[i] = c <= 0.04045 ? c / 12.92
                                     : std::pow((c + 0.055) / 1.055, 2.4);
        }
        return values;
    }();
    const double r = linear[rgb[0]], g = linear[rgb[1]], b = linear[rgb[2]];
    const auto transform = [](double value) {
        return value > 0.008856451679 ? std::cbrt(value)
                                      : 7.787037037 * value + 16.0 / 116.0;
    };
    const double x = transform((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) /
                               0.95047);
    const double y = transform(0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
    const double z = transform((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) /
                               1.08883);
    return {116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)};
}
}  // namespace

// X-AnyLabeling utils/colormap.py (GPL-3.0-only), palette index 0 is black.
QColor annotationLabelColor(int labelId) {
    static const QRgb palette[] = {
            qRgb(170, 170, 255), qRgb(255, 170, 170), qRgb(170, 255, 170),
            qRgb(255, 255, 170), qRgb(170, 255, 255), qRgb(255, 170, 255),
            qRgb(85, 170, 255),  qRgb(255, 170, 0),   qRgb(40, 255, 255),
            qRgb(0, 255, 127),   qRgb(255, 105, 180), qRgb(127, 255, 212),
            qRgb(255, 215, 0),   qRgb(100, 149, 237), qRgb(255, 182, 193),
            qRgb(64, 224, 208),  qRgb(255, 223, 186), qRgb(147, 112, 219),
            qRgb(0, 191, 255),   qRgb(240, 128, 128), qRgb(152, 251, 152),
            qRgb(173, 216, 230), qRgb(255, 192, 203), qRgb(221, 160, 221),
            qRgb(135, 206, 250), qRgb(255, 250, 205), qRgb(175, 238, 238),
            qRgb(250, 128, 114), qRgb(154, 205, 50),  qRgb(32, 178, 170),
            qRgb(255, 160, 122), qRgb(176, 224, 230)};
    const int index = ((labelId % 256) + 256) % 256;
    return index == 0 ? QColor(Qt::black)
                      : QColor::fromRgb(palette[(index - 1) % 32]);
}

QColor annotationShapeColor(const QString& label, const QJsonObject& extra) {
    const QJsonArray rgba =
            extra.value(QStringLiteral("acloudviewer_color")).toArray();
    if (rgba.size() >= 3 && rgba[0].isDouble() && rgba[1].isDouble() &&
        rgba[2].isDouble()) {
        return QColor(rgba[0].toInt(), rgba[1].toInt(), rgba[2].toInt(),
                      rgba.size() >= 4 ? rgba[3].toInt(255) : 255);
    }
    quint32 crc = 0xffffffffu;
    for (const char byte : label.toUtf8()) {
        crc ^= static_cast<quint8>(byte);
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
    }
    crc = ~crc;
    return QColor::fromHsv(int(crc % 360u), 110 + int((crc >> 9) % 66u),
                           225 + int((crc >> 16) % 26u));
}

AnnotationShape AnnotationShape::fromJson(const QJsonObject& object) {
    AnnotationShape shape;
    shape.extra = object;
    shape.label = object.value(QStringLiteral("label")).toString();
    shape.type = object.value(QStringLiteral("shape_type"))
                         .toString(QStringLiteral("polygon"));
    shape.locked = object.value(QStringLiteral("locked")).toBool();
    shape.visible = object.value(QStringLiteral("visible")).toBool(true);
    const QByteArray encoded =
            object.value(QStringLiteral("acloudviewer_mask_png"))
                    .toString()
                    .toLatin1();
    if (!encoded.isEmpty())
        shape.mask =
                QImage::fromData(QByteArray::fromBase64(encoded), "PNG")
                        .convertToFormat(QImage::Format_ARGB32_Premultiplied);
    const QJsonArray coords = object.value(QStringLiteral("points")).toArray();
    for (const QJsonValue& value : coords) {
        const QJsonArray pair = value.toArray();
        if (pair.size() == 2 && pair[0].isDouble() && pair[1].isDouble())
            shape.points.append(
                    QPointF(pair[0].toDouble(), pair[1].toDouble()));
    }
    if (shape.type == QStringLiteral("rectangle") && shape.points.size() == 2)
        shape.points = rectanglePoints(shape.points[0], shape.points[1]);
    return shape;
}

QJsonObject AnnotationShape::toJson() const {
    QJsonObject object = extra;
    object.insert(QStringLiteral("label"), label);
    object.insert(QStringLiteral("shape_type"), type);
    object.insert(QStringLiteral("locked"), locked);
    object.insert(QStringLiteral("visible"), visible);
    if (!mask.isNull()) {
        QByteArray bytes;
        QBuffer buffer(&bytes);
        if (buffer.open(QIODevice::WriteOnly) && mask.save(&buffer, "PNG"))
            object.insert(QStringLiteral("acloudviewer_mask_png"),
                          QString::fromLatin1(bytes.toBase64()));
    }
    if (type == QStringLiteral("rotation") && points.size() == 4)
        object.insert(
                QStringLiteral("direction"),
                normalizedAngle(std::atan2(points[1].y() - points[0].y(),
                                           points[1].x() - points[0].x())));
    if (type == QStringLiteral("cuboid") && points.size() == 8) {
        QJsonObject cuboid =
                object.value(QStringLiteral("cuboid3d")).toObject();
        if (!cuboid.contains(QStringLiteral("version")))
            cuboid.insert(QStringLiteral("version"), 1);
        if (!cuboid.contains(QStringLiteral("mode")))
            cuboid.insert(QStringLiteral("mode"),
                          QStringLiteral("from_rectangle"));
        if (!cuboid.contains(QStringLiteral("source")))
            cuboid.insert(QStringLiteral("source"), QStringLiteral("manual"));
        if (!cuboid.contains(QStringLiteral("vertices2d_order"))) {
            cuboid.insert(QStringLiteral("vertices2d_order"),
                          QJsonArray{QStringLiteral("front_tl"),
                                     QStringLiteral("front_tr"),
                                     QStringLiteral("front_br"),
                                     QStringLiteral("front_bl"),
                                     QStringLiteral("back_tl"),
                                     QStringLiteral("back_tr"),
                                     QStringLiteral("back_br"),
                                     QStringLiteral("back_bl")});
        }
        cuboid.insert(QStringLiteral("depth_vector"),
                      QJsonArray{points[4].x() - points[0].x(),
                                 points[4].y() - points[0].y()});
        object.insert(QStringLiteral("cuboid3d"), cuboid);
    }
    QJsonArray coords;
    for (const QPointF& point : points)
        coords.append(QJsonArray{point.x(), point.y()});
    object.insert(QStringLiteral("points"), coords);
    if (!object.contains(QStringLiteral("flags")))
        object.insert(QStringLiteral("flags"), QJsonObject());
    if (!object.contains(QStringLiteral("group_id")))
        object.insert(QStringLiteral("group_id"), QJsonValue::Null);
    if (!object.contains(QStringLiteral("description")))
        object.insert(QStringLiteral("description"), QString());
    if (!object.contains(QStringLiteral("difficult")))
        object.insert(QStringLiteral("difficult"), false);
    return object;
}

AnnotationCanvas::AnnotationCanvas(QWidget* parent) : QWidget(parent) {
    QPalette colors = palette();
    colors.setColor(QPalette::Window, Qt::white);
    setPalette(colors);
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setCursor(Qt::ArrowCursor);
    m_magicDefaultThreshold =
            annotationConfig::value("canvas/magic_wand/default_threshold")
                    .toInt();
    m_magicDragSensitivity =
            annotationConfig::value("canvas/magic_wand/drag_sensitivity")
                    .toDouble();
    m_magicLuminanceWeight =
            annotationConfig::value("canvas/magic_wand/luminance_weight")
                    .toDouble();
    m_magicOpacity =
            annotationConfig::value("canvas/magic_wand/opacity").toDouble();
    m_magicSimplifyEpsilon =
            annotationConfig::value("canvas/magic_wand/simplify_epsilon")
                    .toDouble();
    m_shapePointSize = annotationConfig::value("shape/point_size").toInt();
    m_shapeLineWidth = annotationConfig::value("shape/line_width").toDouble();
    m_hitEpsilon = annotationConfig::value("canvas/epsilon").toDouble();
    m_crosshairVisible =
            annotationConfig::value("canvas/crosshair/show").toBool();
    m_crosshairWidth =
            annotationConfig::value("canvas/crosshair/width").toDouble();
    m_crosshairOpacity =
            annotationConfig::value("canvas/crosshair/opacity").toDouble();
    m_doubleClickEditLabel =
            annotationConfig::value("canvas/double_click_edit_label").toBool();
    m_wheelRectangleEditing =
            annotationConfig::value("canvas/wheel_rectangle_editing/enable")
                    .toBool();
    m_wheelAdjustStep = annotationConfig::value(
                                "canvas/wheel_rectangle_editing/adjust_step")
                                .toDouble();
    m_wheelScaleStep =
            annotationConfig::value("canvas/wheel_rectangle_editing/scale_step")
                    .toDouble();
    m_brushPointDistance =
            annotationConfig::value("canvas/brush/point_distance").toDouble();
    m_brushMaxUndoSteps =
            annotationConfig::value("canvas/brush/max_undo_steps").toInt();
    m_brushMaxUndoMemoryMb =
            annotationConfig::value("canvas/brush/max_undo_memory_mb").toInt();
    m_rotationLargeIncrement =
            annotationConfig::value("canvas/rotation/large_increment")
                    .toDouble();
    m_rotationSmallIncrement =
            annotationConfig::value("canvas/rotation/small_increment")
                    .toDouble();
    m_cuboidMinDepth =
            annotationConfig::value("canvas/cuboid/min_depth").toDouble();
    m_maskOpacity = annotationConfig::value("canvas/mask/opacity").toInt();
    m_shiftAutoShapeColor =
            annotationConfig::value("shift_auto_shape_color").toInt();
    m_showShapes = annotationConfig::value("show_shapes").toBool();
    m_showMasks = annotationConfig::value("show_masks").toBool();
    m_showLabels = annotationConfig::value("show_labels").toBool();
    m_showScores = annotationConfig::value("show_scores").toBool();
    m_showGroups = annotationConfig::value("show_groups").toBool();
    m_showTexts = annotationConfig::value("show_texts").toBool();
    m_showAttributes = annotationConfig::value("show_attributes").toBool();
    m_showDegrees = annotationConfig::value("show_degrees").toBool();
    m_showLinking = annotationConfig::value("show_linking").toBool();
    m_crosshairColor = annotationConfig::color("canvas/crosshair/color");
    m_defaultShapeColor = annotationConfig::color("default_shape_color");
    m_lineColor = annotationConfig::color("shape/line_color");
    m_fillColor = annotationConfig::color("shape/fill_color");
    m_vertexFillColor = annotationConfig::color("shape/vertex_fill_color");
    m_selectLineColor = annotationConfig::color("shape/select_line_color");
    m_selectFillColor = annotationConfig::color("shape/select_fill_color");
    m_hoverVertexFillColor =
            annotationConfig::color("shape/hvertex_fill_color");
    m_doubleClickClose =
            annotationConfig::value("canvas/double_click").toString() ==
            QStringLiteral("close");
    const QJsonArray depth =
            annotationConfig::jsonValue(
                    QStringLiteral("canvas/cuboid/default_depth_vector"))
                    .toArray();
    m_cuboidDepthVector =
            QPointF(depth.at(0).toDouble(), depth.at(1).toDouble());
}

void AnnotationCanvas::rebuildDisplayImage() {
    if (m_image.isNull()) {
        m_displayImage = QImage();
        return;
    }

    const double brightness = m_brightness / 50.0;
    const double contrast = m_contrast / 50.0;
    if (brightness == 1.0 && contrast == 1.0) {
        m_displayImage = m_image;
        return;
    }

    // Null images were rejected above; older Qt cannot carry this format.
    if (m_image.format() == qtCompatQImageFormatGrayscale16()) {
        const int width = m_image.width();
        const int height = m_image.height();
        double mean = 0.0;
        for (int y = 0; y < height; ++y) {
            const auto* row =
                    reinterpret_cast<const quint16*>(m_image.constScanLine(y));
            for (int x = 0; x < width; ++x) mean += row[x];
        }
        mean /= std::max(1, width * height);
        m_displayImage =
                QImage(m_image.size(), qtCompatQImageFormatGrayscale16());
        for (int y = 0; y < height; ++y) {
            const auto* source =
                    reinterpret_cast<const quint16*>(m_image.constScanLine(y));
            auto* target =
                    reinterpret_cast<quint16*>(m_displayImage.scanLine(y));
            for (int x = 0; x < width; ++x) {
                float value = static_cast<float>(source[x]);
                value *= static_cast<float>(brightness * contrast);
                value += static_cast<float>(mean * brightness *
                                            (1.0 - contrast));
                target[x] = static_cast<quint16>(
                        std::max(0.0f, std::min(65535.0f, value)));
            }
        }
        return;
    }

    const QImage source = m_image.convertToFormat(QImage::Format_ARGB32);
    m_displayImage = QImage(source.size(), QImage::Format_ARGB32);
    // Pillow's RGB -> L conversion uses these fixed-point coefficients.
    // Keeping the conversion integer based is important at high contrast:
    // the display cache must agree with the source pixel values, not only
    // look visually similar.
    constexpr int redCoeff = 19595;
    constexpr int greenCoeff = 38470;
    constexpr int blueCoeff = 7471;
    qint64 graySum = 0;
    for (int y = 0; y < source.height(); ++y) {
        const auto* row =
                reinterpret_cast<const QRgb*>(source.constScanLine(y));
        for (int x = 0; x < source.width(); ++x) {
            const QRgb pixel = row[x];
            const auto brighten = [brightness](int channel) {
                return std::clamp(static_cast<int>(channel * brightness), 0,
                                  255);
            };
            const int r = brighten(qRed(pixel));
            const int g = brighten(qGreen(pixel));
            const int b = brighten(qBlue(pixel));
            graySum +=
                    (redCoeff * r + greenCoeff * g + blueCoeff * b + 32768) >>
                    16;
        }
    }
    const double mean = static_cast<double>(graySum) /
                        std::max(1, source.width() * source.height());
    const int grayValue =
            std::clamp(static_cast<int>(std::floor(mean + 0.5)), 0, 255);
    for (int y = 0; y < source.height(); ++y) {
        const auto* input =
                reinterpret_cast<const QRgb*>(source.constScanLine(y));
        auto* output = reinterpret_cast<QRgb*>(m_displayImage.scanLine(y));
        for (int x = 0; x < source.width(); ++x) {
            const QRgb pixel = input[x];
            const auto adjust = [brightness, contrast, grayValue](int channel) {
                const int bright = std::clamp(
                        static_cast<int>(channel * brightness), 0, 255);
                return static_cast<int>(std::floor(std::clamp(
                        bright * contrast + grayValue * (1.0 - contrast), 0.0,
                        255.0)));
            };
            output[x] = qRgba(adjust(qRed(pixel)), adjust(qGreen(pixel)),
                              adjust(qBlue(pixel)), qAlpha(pixel));
        }
    }
}

void AnnotationCanvas::setImage(const QImage& image) {
    clearRelations();
    clearMagicWand();
    clearPrompt();
    m_brushEditing = false;
    m_brushMask = QImage();
    m_brushHistory.clear();
    m_brushRedo.clear();
    m_image = image;
    m_brightness = 50;
    m_contrast = 50;
    rebuildDisplayImage();
    m_compareImage = QImage();
    m_shapes.clear();
    m_selected = -1;
    m_selectedIndices.clear();
    m_creating = false;
    m_dragging = m_rotating = false;
    m_dragFace = annotationCuboid::Face::None;
    m_hoverShape = m_hoverVertex = -1;
    m_pan = QPointF();
    fitImage();
}

void AnnotationCanvas::setShapeOpacity(int value) {
    const int clamped = std::max(0, std::min(100, value));
    if (m_shapeOpacity == clamped) return;
    m_shapeOpacity = clamped;
    update();
    emit displayAdjustmentChanged();
}

void AnnotationCanvas::setMaskOpacity(int value) {
    const int clamped = std::clamp(value, 0, 255);
    if (m_maskOpacity == clamped) return;
    m_maskOpacity = clamped;
    update();
}

void AnnotationCanvas::setMagicWandOptions(int threshold,
                                           double dragSensitivity,
                                           double luminanceWeight) {
    m_magicDefaultThreshold = std::clamp(threshold, 0, 255);
    m_magicDragSensitivity = std::max(0.1, dragSensitivity);
    m_magicLuminanceWeight = std::clamp(luminanceWeight, 0.0, 1.0);
}

void AnnotationCanvas::setMagicWandOpacity(double opacity) {
    m_magicOpacity = std::clamp(opacity, 0.0, 1.0);
    update();
}

void AnnotationCanvas::setMagicWandSimplifyEpsilon(double epsilon) {
    m_magicSimplifyEpsilon = std::clamp(epsilon, 0.0, 100.0);
}

void AnnotationCanvas::setWheelRectangleEditing(bool enabled,
                                                double adjustStep,
                                                double scaleStep) {
    m_wheelRectangleEditing = enabled;
    m_wheelAdjustStep = std::clamp(adjustStep, 0.1, 50.0);
    m_wheelScaleStep = std::clamp(scaleStep, 0.001, 1.0);
}

void AnnotationCanvas::setBrushPointDistance(double distance) {
    m_brushPointDistance = std::max(0.0, distance);
}

void AnnotationCanvas::setBrushHistoryLimits(int maxSteps, int maxMemoryMb) {
    m_brushMaxUndoSteps = std::clamp(maxSteps, 1, 1000);
    m_brushMaxUndoMemoryMb = std::clamp(maxMemoryMb, 1, 4096);
}

void AnnotationCanvas::setRotationIncrements(double largeDegrees,
                                             double smallDegrees) {
    m_rotationLargeIncrement = std::clamp(largeDegrees, 0.01, 45.0);
    m_rotationSmallIncrement = std::clamp(smallDegrees, 0.01, 10.0);
}

void AnnotationCanvas::setCuboidDefaults(const QPointF& depthVector,
                                         double minDepth) {
    m_cuboidDepthVector = depthVector;
    m_cuboidMinDepth = std::max(0.0, minDepth);
}

void AnnotationCanvas::setShapeStyle(int pointSize, double lineWidth) {
    m_shapePointSize = std::clamp(pointSize, 1, 64);
    m_shapeLineWidth = std::clamp(lineWidth, 0.5, 32.0);
    update();
}

void AnnotationCanvas::setHitEpsilon(double epsilon) {
    m_hitEpsilon = std::clamp(epsilon, 0.5, 100.0);
}

void AnnotationCanvas::setShapeColorMode(ShapeColorMode mode) {
    m_shapeColorMode = mode;
    update();
}

void AnnotationCanvas::setDefaultShapeColor(const QColor& color) {
    if (color.isValid()) m_defaultShapeColor = color;
    update();
}

void AnnotationCanvas::setShiftAutoShapeColor(int shift) {
    m_shiftAutoShapeColor = shift;
    update();
}

void AnnotationCanvas::setLabelOrder(const QStringList& labels) {
    m_labelColorIds.clear();
    for (const QString& label : labels)
        if (!m_labelColorIds.contains(label))
            m_labelColorIds.insert(label, m_labelColorIds.size() + 1);
    update();
}

void AnnotationCanvas::setShapeStyleColors(const QColor& line,
                                           const QColor& fill,
                                           const QColor& vertex,
                                           const QColor& selectLine,
                                           const QColor& selectFill,
                                           const QColor& hoverVertex) {
    if (line.isValid()) m_lineColor = line;
    if (fill.isValid()) m_fillColor = fill;
    if (vertex.isValid()) m_vertexFillColor = vertex;
    if (selectLine.isValid()) m_selectLineColor = selectLine;
    if (selectFill.isValid()) m_selectFillColor = selectFill;
    if (hoverVertex.isValid()) m_hoverVertexFillColor = hoverVertex;
    update();
}

QImage AnnotationCanvas::tintedSilhouette(const QImage& mask,
                                          const QColor& color,
                                          QImage& cache,
                                          quint64& sourceKey,
                                          QColor& cacheColor) const {
    if (!cache.isNull() && sourceKey == mask.cacheKey() && cacheColor == color)
        return cache;
    const QImage alpha = mask.convertToFormat(QImage::Format_Alpha8);
    if (alpha.isNull()) return QImage();
    QImage binarized = alpha;
    // The source renders (mask > 0) * 255 and fills the silhouette with
    // the shape line color at mask_opacity; stored semi-transparent pixels
    // do not survive compositing.
    for (int y = 0; y < binarized.height(); ++y) {
        uchar* line = binarized.scanLine(y);
        for (int x = 0; x < binarized.width(); ++x) line[x] = line[x] ? 255 : 0;
    }
    QImage out(mask.size(), QImage::Format_ARGB32_Premultiplied);
    QColor fill = color;
    fill.setAlpha(255);
    out.fill(fill);
    QPainter painter(&out);
    painter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
    painter.drawImage(0, 0, binarized);
    painter.end();
    cache = out;
    sourceKey = mask.cacheKey();
    cacheColor = color;
    return out;
}

QImage AnnotationCanvas::shapeSilhouette(const AnnotationShape& shape,
                                         const QColor& color) const {
    return tintedSilhouette(shape.mask, color, shape.displaySilhouette,
                            shape.displaySilhouetteSource,
                            shape.displaySilhouetteColor);
}

QColor AnnotationCanvas::effectiveLineColor(const AnnotationShape& shape,
                                            bool selected) const {
    if (selected) return m_selectLineColor;
    if (m_lineColor != annotationConfig::color("shape/line_color"))
        return m_lineColor;
    if (shape.label == QStringLiteral("AUTOLABEL_ADD"))
        return QColor(144, 238, 144);
    if (shape.label == QStringLiteral("AUTOLABEL_REMOVE"))
        return QColor(255, 182, 193);
    if (shape.extra.value(QStringLiteral("acloudviewer_color")).isArray())
        return annotationShapeColor(shape.label, shape.extra);
    if (m_shapeColorMode == ShapeColorMode::Manual) return m_defaultShapeColor;
    const int index = m_labelColorIds.value(shape.label, 1);
    return annotationLabelColor(index + m_shiftAutoShapeColor);
}

void AnnotationCanvas::setCrosshair(bool visible,
                                    double width,
                                    const QColor& color,
                                    double opacity) {
    m_crosshairVisible = visible;
    m_crosshairWidth = std::clamp(width, 0.5, 16.0);
    if (color.isValid()) m_crosshairColor = color;
    m_crosshairOpacity = std::clamp(opacity, 0.0, 1.0);
    update();
}

void AnnotationCanvas::setBrightness(int value) {
    setBrightnessContrast(value, m_contrast);
}

void AnnotationCanvas::setContrast(int value) {
    setBrightnessContrast(m_brightness, value);
}

void AnnotationCanvas::setBrightnessContrast(int brightness, int contrast) {
    const int clampedBrightness = std::max(0, std::min(150, brightness));
    const int clampedContrast = std::max(0, std::min(150, contrast));
    if (m_brightness == clampedBrightness && m_contrast == clampedContrast)
        return;
    m_brightness = clampedBrightness;
    m_contrast = clampedContrast;
    rebuildDisplayImage();
    update();
    emit displayAdjustmentChanged();
}

void AnnotationCanvas::setCompareImage(const QImage& image) {
    m_compareImage = image;
    m_compareSplit = 0.5;
    update();
}

void AnnotationCanvas::clearCompareImage() {
    m_compareImage = QImage();
    m_compareSplit = 0.5;
    update();
}

void AnnotationCanvas::setCompareSplit(double split) {
    m_compareSplit = std::max(0.0, std::min(1.0, split));
    update();
}

void AnnotationCanvas::setRelations(const QJsonArray& relations) {
    m_relations = relations;
    m_highlightedRelation = -1;
    update();
}

void AnnotationCanvas::clearRelations() {
    if (m_relations.isEmpty() && m_highlightedRelation < 0) return;
    m_relations = QJsonArray();
    m_highlightedRelation = -1;
    update();
}

void AnnotationCanvas::setHighlightedRelation(int row) {
    m_highlightedRelation = row;
    update();
}

void AnnotationCanvas::setShapes(const QVector<AnnotationShape>& shapes) {
    m_relations = QJsonArray();
    m_highlightedRelation = -1;
    m_brushEditing = false;
    m_brushMask = QImage();
    m_brushHistory.clear();
    m_brushRedo.clear();
    m_shapes = shapes;
    for (const AnnotationShape& shape : shapes)
        if (!m_labelColorIds.contains(shape.label))
            m_labelColorIds.insert(shape.label, m_labelColorIds.size() + 1);
    m_selected = -1;
    m_selectedIndices.clear();
    m_creating = false;
    m_dragging = m_rotating = false;
    m_dragFace = annotationCuboid::Face::None;
    m_hoverShape = m_hoverVertex = -1;
    update();
}

QImage AnnotationCanvas::renderVisualizationImage(
        const QVector<AnnotationShape>& shapes) const {
    const QImage source = m_displayImage.isNull() ? m_image : m_displayImage;
    return renderVisualizationImage(source, shapes, visualizationOptions());
}

AnnotationCanvas::VisualizationOptions AnnotationCanvas::visualizationOptions()
        const {
    return {m_showLabels,  m_showScores,  m_showGroups,
            m_showTexts,   m_showMasks,   m_showShapes,
            m_showLinking, m_showDegrees, m_showAttributes};
}

QImage AnnotationCanvas::renderVisualizationImage(
        const QImage& source,
        const QVector<AnnotationShape>& shapes,
        const VisualizationOptions& options) const {
    if (source.isNull()) return {};
    QImage result = source.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&result);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    QHash<int, QRectF> groupBounds;
    QHash<int, int> groupCounts;
    if (options.showGroups || options.showLinking) {
        for (const AnnotationShape& shape : shapes) {
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (!group.isDouble() || !shape.visible || shape.points.isEmpty())
                continue;
            const int groupId = group.toInt();
            const QRectF bounds = shapeBounds(shape);
            groupBounds[groupId] =
                    groupBounds.contains(groupId)
                            ? groupBounds.value(groupId).united(bounds)
                            : bounds;
            ++groupCounts[groupId];
        }
    }
    if (options.showLinking) {
        painter.setPen(QPen(QColor(255, 165, 0, 220), 2.0));
        for (const AnnotationShape& shape : shapes) {
            if (!shape.visible) continue;
            const QJsonArray links =
                    shape.extra.value(QStringLiteral("kie_linking")).toArray();
            for (const QJsonValue& linkValue : links) {
                const QJsonArray link = linkValue.toArray();
                if (link.size() != 2 || !link[0].isDouble() ||
                    !link[1].isDouble() ||
                    !groupBounds.contains(link[0].toInt()) ||
                    !groupBounds.contains(link[1].toInt()))
                    continue;
                const QPointF from =
                        groupBounds.value(link[0].toInt()).center();
                const QPointF to = groupBounds.value(link[1].toInt()).center();
                painter.drawLine(from, to);
                const QLineF line(from, to);
                if (line.length() < 1.0) continue;
                const QPointF unit =
                        line.unitVector().p2() - line.unitVector().p1();
                const QPointF normal(-unit.y(), unit.x());
                QPolygonF arrow;
                arrow << to << to - unit * 10.0 + normal * 4.0
                      << to - unit * 10.0 - normal * 4.0;
                painter.setBrush(QColor(255, 165, 0, 220));
                painter.drawPolygon(arrow);
                painter.setBrush(Qt::NoBrush);
            }
        }
    }
    if (options.showGroups) {
        painter.setPen(QPen(QColor(255, 165, 0, 220), 1.5, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        for (auto it = groupBounds.constBegin(); it != groupBounds.constEnd();
             ++it) {
            const QRectF bounds = it.value();
            painter.drawRect(bounds.adjusted(-4.0, -4.0, 4.0, 4.0));
            const QString title = QStringLiteral("G%1 · S%2")
                                          .arg(it.key())
                                          .arg(groupCounts.value(it.key()));
            const QRectF textRect(
                    bounds.left(), std::max(0.0, bounds.top() - 18.0),
                    QTCOMPAT_FONTMETRICS_WIDTH(painter.fontMetrics(), title) +
                            8.0,
                    17.0);
            painter.fillRect(textRect, QColor(120, 70, 0, 220));
            painter.setPen(Qt::white);
            painter.drawText(textRect.left() + 4.0, textRect.bottom() - 5.0,
                             title);
            painter.setPen(QPen(QColor(255, 165, 0, 220), 1.5, Qt::DashLine));
        }
    }

    if (options.showMasks) {
        painter.save();
        painter.setOpacity(m_shapeOpacity / 100.0);
        for (const AnnotationShape& shape : shapes)
            if (shape.visible)
                drawGeometryMask(painter, shape,
                                 effectiveLineColor(shape, false),
                                 m_shapeLineWidth, m_maskOpacity, true);
        painter.restore();
    }
    for (const AnnotationShape& shape : shapes) {
        if (!shape.visible || (shape.points.isEmpty() && shape.mask.isNull()))
            continue;
        if (options.showMasks && !shape.mask.isNull()) {
            painter.save();
            // Source compositing: binary silhouette in the shape color at
            // the mask opacity, not a multiplication of stored alpha.
            painter.setOpacity(m_shapeOpacity / 100.0 * m_maskOpacity / 255.0);
            painter.drawImage(
                    QPointF(0, 0),
                    shapeSilhouette(shape, effectiveLineColor(shape, false)));
            painter.restore();
        }
        if (options.showShapes && !shape.points.isEmpty()) {
            const QColor color = effectiveLineColor(shape, false);
            painter.setPen(
                    annotationOutlinePen(shape, color, m_shapeLineWidth));
            // Export has no selected or hovered shapes, as upstream does.
            painter.setBrush(Qt::NoBrush);
            QPainterPath path;
            path.moveTo(shape.points.first());
            if (shape.type == QStringLiteral("circle") &&
                shape.points.size() >= 2) {
                painter.drawEllipse(circleBounds(shape));
            } else if (shape.type == QStringLiteral("point")) {
                // The vertex marker is painted below with its fill color.
            } else if (isCuboid(shape)) {
                drawCuboid(painter, shape.points);
            } else {
                for (int i = 1; i < shape.points.size(); ++i)
                    path.lineTo(shape.points[i]);
                if (closedType(shape.type) && shape.points.size() > 2)
                    path.closeSubpath();
                painter.drawPath(path);
            }
        }
        if (shape.points.isEmpty()) continue;
        if (options.showShapes &&
            (shape.type == QStringLiteral("point") ||
             shape.type == QStringLiteral("quadrilateral") ||
             shape.type == QStringLiteral("line") ||
             shape.type == QStringLiteral("linestrip"))) {
            painter.save();
            const QColor color = effectiveLineColor(shape, false);
            painter.setPen(
                    annotationOutlinePen(shape, color, m_shapeLineWidth));
            painter.setBrush(color);
            const int count = shape.type == QStringLiteral("linestrip")
                                      ? shape.points.size()
                                      : 1;
            for (int i = 0; i < count; ++i)
                drawAnnotationVertex(painter, shape, shape.points[i],
                                     m_shapePointSize / 2.0);
            painter.restore();
        }
        if (options.showShapes || options.showDegrees)
            drawRotationDirection(painter, shape, 1.0, m_shapePointSize,
                                  options.showDegrees);
    }
    drawAnnotationLabels(painter, *this, shapes, options, result.rect());
    if (options.showAttributes)
        for (const auto& shape : shapes)
            if (shape.visible)
                drawAnnotationAttributes(painter, shape, 1.0, m_shapePointSize);
    painter.end();
    return result;
}

void AnnotationCanvas::setShapeDescription(int index,
                                           const QString& description) {
    if (index < 0 || index >= m_shapes.size()) return;
    m_shapes[index].extra.insert(QStringLiteral("description"), description);
}

void AnnotationCanvas::setShapeRecognition(int index,
                                           const QString& description,
                                           double score,
                                           const QString& language) {
    if (index < 0 || index >= m_shapes.size()) return;
    m_shapes[index].extra.insert(QStringLiteral("description"), description);
    m_shapes[index].extra.insert(QStringLiteral("score"), score);
    m_shapes[index].extra.insert(QStringLiteral("language"), language);
    update();
}

void AnnotationCanvas::setTool(const QString& type) {
    if (m_brushEditing) setBrushEditing(false);
    if (m_creating) cancelShape();
    clearMagicWand();
    m_tool = type;
    setCursor(type == QStringLiteral("edit") ? Qt::ArrowCursor
                                             : Qt::CrossCursor);
    update();
}

AnnotationPrompt AnnotationCanvas::prompt() const {
    AnnotationPrompt result = m_prompt;
    result.label = m_label.isEmpty() ? tr("object") : m_label;
    return result;
}

void AnnotationCanvas::clearPrompt() {
    m_prompt = AnnotationPrompt();
    m_promptBoxDragging = false;
    emit promptChanged();
    update();
}

void AnnotationCanvas::undoLastPoint() {
    if (!m_creating || m_shapes.isEmpty()) return;
    AnnotationShape& shape = m_shapes.last();
    if (shape.points.size() <= 1) {
        cancelShape();
        return;
    }
    shape.points.removeLast();
    m_last = shape.points.last();
    emit shapesChanged();
    update();
}

void AnnotationCanvas::addPointToEdge() {
    const int index = m_selected;
    if (index < 0 || index >= m_shapes.size() ||
        m_shapes[index].points.size() < 2 || m_shapes[index].locked ||
        m_shapes[index].type == QStringLiteral("cuboid"))
        return;
    AnnotationShape& shape = m_shapes[index];
    int bestEdge = -1;
    qreal bestDistance = std::numeric_limits<qreal>::max();
    const int edgeCount = shape.points.size() > 2 ? shape.points.size()
                                                  : shape.points.size() - 1;
    for (int i = 0; i < edgeCount; ++i) {
        const QPointF a = shape.points.at(i);
        const QPointF b = shape.points.at((i + 1) % shape.points.size());
        const QLineF edge(a, b);
        const QPointF v = b - a;
        const qreal len2 = QPointF::dotProduct(v, v);
        const qreal t =
                len2 > 0.0
                        ? std::clamp(QPointF::dotProduct(m_hover - a, v) / len2,
                                     qreal(0), qreal(1))
                        : 0.0;
        const QPointF projection = a + t * v;
        const qreal distance = QLineF(m_hover, projection).length();
        if (distance < bestDistance) {
            bestDistance = distance;
            bestEdge = i;
        }
    }
    if (bestEdge < 0) return;
    shape.points.insert(bestEdge + 1, m_hover);
    emit shapesChanged();
    update();
}

void AnnotationCanvas::removeSelectedPoint() {
    if (m_selected < 0 || m_selected >= m_shapes.size() ||
        m_shapes[m_selected].locked ||
        m_shapes[m_selected].type == QStringLiteral("cuboid"))
        return;
    AnnotationShape& shape = m_shapes[m_selected];
    if (shape.points.size() <=
        (shape.type == QStringLiteral("polygon") ? 3 : 1))
        return;
    int best = -1;
    qreal distance = std::numeric_limits<qreal>::max();
    for (int i = 0; i < shape.points.size(); ++i) {
        const qreal d = QLineF(shape.points.at(i), m_hover).length();
        if (d < distance) {
            distance = d;
            best = i;
        }
    }
    if (best < 0) return;
    shape.points.removeAt(best);
    emit shapesChanged();
    update();
}

void AnnotationCanvas::removeLastPromptPoint(bool negative) {
    auto& points = negative ? m_prompt.negative : m_prompt.positive;
    if (!points.isEmpty()) points.removeLast();
    emit promptChanged();
    update();
}

void AnnotationCanvas::selectShape(int index) {
    setSelectedShapes(index >= 0 ? QVector<int>{index} : QVector<int>{});
}

void AnnotationCanvas::setSelectedShapes(const QVector<int>& indices) {
    QSet<int> next;
    int primary = -1;
    for (int index : indices) {
        if (index < 0 || index >= m_shapes.size()) continue;
        next.insert(index);
        primary = index;
    }
    if (m_brushEditing && (primary != m_brushTarget || next.size() != 1))
        return;
    if (m_selectedIndices == next && m_selected == primary) return;
    m_selectedIndices = next;
    m_selected = primary;
    emit shapeSelected(m_selected);
    update();
}

void AnnotationCanvas::deleteSelected() {
    if (m_selectedIndices.isEmpty()) return;
    QVector<int> rows;
    for (int row : m_selectedIndices) {
        if (row < 0 || row >= m_shapes.size() || m_shapes[row].locked) return;
        rows.append(row);
    }
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    emit editStarted();
    for (int row : rows) m_shapes.removeAt(row);
    m_selected = -1;
    m_selectedIndices.clear();
    emit shapeSelected(-1);
    emit shapesChanged();
    update();
}

void AnnotationCanvas::discardLastShape() {
    if (m_shapes.isEmpty()) return;
    m_shapes.removeLast();
    m_creating = false;
    selectShape(-1);
    update();
}

void AnnotationCanvas::fitImage() {
    m_zoom = 1.0;
    m_pan = QPointF();
    emit viewChanged();
    update();
}

void AnnotationCanvas::fitWidth() {
    if (m_image.isNull()) return;
    const double base =
            std::min(std::max(1, width() - 24) / double(m_image.width()),
                     std::max(1, height() - 24) / double(m_image.height()));
    const double widthScale =
            std::max(1, width() - 24) / double(m_image.width());
    m_zoom = std::max(0.1, std::min(20.0, widthScale / std::max(base, 1e-9)));
    m_pan = QPointF();
    emit viewChanged();
    update();
}

void AnnotationCanvas::zoomOriginal() {
    if (m_image.isNull()) return;
    const double base =
            std::min(std::max(1, width() - 24) / double(m_image.width()),
                     std::max(1, height() - 24) / double(m_image.height()));
    m_zoom = std::max(0.1, std::min(20.0, 1.0 / std::max(base, 1e-9)));
    m_pan = QPointF();
    emit viewChanged();
    update();
}

void AnnotationCanvas::zoomBy(double factor) {
    m_zoom = std::max(0.1, std::min(20.0, m_zoom * factor));
    emit viewChanged();
    update();
}

void AnnotationCanvas::setZoomFactor(double factor) {
    const double clamped = std::max(0.1, std::min(20.0, factor));
    if (std::abs(m_zoom - clamped) < 1e-9) return;
    m_zoom = clamped;
    emit viewChanged();
    update();
}

void AnnotationCanvas::zoomToRect(const QRectF& imageRect,
                                  double widthDivisor) {
    if (m_image.isNull() || widthDivisor <= 0.0) return;
    updateTransform();
    // Source review zoom: min(1000%, 100 * viewportWidth /
    // (zoom_scale * label_width)), with a 30px minimum label width so
    // tiny shapes and points stay visible.
    const double labelWidth = std::max(imageRect.width(), 30.0);
    const double pixelScale =
            std::min(10.0, (width() - 2.0) / (widthDivisor * labelWidth));
    const double sx = std::max(1, width() - 24) / double(m_image.width());
    const double sy = std::max(1, height() - 24) / double(m_image.height());
    const double base = std::max(std::min(sx, sy), 1e-9);
    setZoomFactor(pixelScale / base);
    navigateTo(QPointF(imageRect.center().x() / m_image.width(),
                       imageRect.center().y() / m_image.height()));
}

void AnnotationCanvas::setFillDrawing(bool enabled) {
    if (m_fillDrawing == enabled) return;
    m_fillDrawing = enabled;
    update();
}

void AnnotationCanvas::setShowShapes(bool visible) {
    if (m_showShapes == visible) return;
    m_showShapes = visible;
    update();
}

void AnnotationCanvas::setShowMasks(bool visible) {
    if (m_showMasks == visible) return;
    m_showMasks = visible;
    update();
}

void AnnotationCanvas::setShowLabels(bool visible) {
    if (m_showLabels == visible) return;
    m_showLabels = visible;
    update();
}

void AnnotationCanvas::setShowScores(bool visible) {
    if (m_showScores == visible) return;
    m_showScores = visible;
    update();
}

void AnnotationCanvas::setShowGroups(bool visible) {
    if (m_showGroups == visible) return;
    m_showGroups = visible;
    update();
}

void AnnotationCanvas::setShowTexts(bool visible) {
    if (m_showTexts == visible) return;
    m_showTexts = visible;
    update();
}

void AnnotationCanvas::setShowAttributes(bool visible) {
    if (m_showAttributes == visible) return;
    m_showAttributes = visible;
    update();
}

void AnnotationCanvas::setShowDegrees(bool visible) {
    if (m_showDegrees == visible) return;
    m_showDegrees = visible;
    update();
}

void AnnotationCanvas::setShowLinking(bool visible) {
    if (m_showLinking == visible) return;
    m_showLinking = visible;
    update();
}

QRectF AnnotationCanvas::viewportImageRect() {
    if (m_image.isNull()) return QRectF();
    updateTransform();
    if (m_scale <= 0.0) return QRectF();
    const QRectF viewport = QRectF(toImage(QPointF(0.0, 0.0)),
                                   toImage(QPointF(width(), height())))
                                    .normalized();
    return viewport.intersected(
            QRectF(0.0, 0.0, m_image.width(), m_image.height()));
}

void AnnotationCanvas::navigateTo(const QPointF& imageRatio) {
    if (m_image.isNull()) return;
    updateTransform();
    const QPointF ratio(std::max(0.0, std::min(1.0, imageRatio.x())),
                        std::max(0.0, std::min(1.0, imageRatio.y())));
    const QPointF imagePoint(ratio.x() * m_image.width(),
                             ratio.y() * m_image.height());
    m_pan += QPointF(width() / 2.0, height() / 2.0) -
             (m_origin + imagePoint * m_scale);
    emit viewChanged();
    update();
}

void AnnotationCanvas::updateTransform() {
    if (m_image.isNull()) return;
    const double sx = std::max(1, width() - 24) / double(m_image.width());
    const double sy = std::max(1, height() - 24) / double(m_image.height());
    m_scale = std::min(sx, sy) * m_zoom;
    m_origin = QPointF((width() - m_image.width() * m_scale) / 2.0,
                       (height() - m_image.height() * m_scale) / 2.0) +
               m_pan;
}

QPointF AnnotationCanvas::rotationHandle(const AnnotationShape& shape) const {
    if (shape.type != QStringLiteral("rotation") || shape.points.size() != 4)
        return QPointF();
    const QPointF edge = shape.points[1] - shape.points[0];
    const double length = std::hypot(edge.x(), edge.y());
    if (length < 1e-6) return QPointF();
    return (shape.points[0] + shape.points[1]) / 2.0 +
           QPointF(edge.y(), -edge.x()) * (32.0 / (length * m_scale));
}

QPointF AnnotationCanvas::toImage(const QPointF& point) const {
    return (point - m_origin) / m_scale;
}

QPointF AnnotationCanvas::clamp(const QPointF& point) const {
    return QPointF(
            std::max(0.0, std::min(double(m_image.width()), point.x())),
            std::max(0.0, std::min(double(m_image.height()), point.y())));
}

void AnnotationCanvas::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), palette().color(QPalette::Window));
    if (m_image.isNull()) return;
    updateTransform();
    painter.setRenderHint(QPainter::Antialiasing);
    painter.drawImage(QRectF(m_origin, QSizeF(m_image.width() * m_scale,
                                              m_image.height() * m_scale)),
                      m_displayImage.isNull() ? m_image : m_displayImage);
    if (!m_compareImage.isNull() && m_compareImage.size() == m_image.size()) {
        const QRectF imageRect(m_origin, QSizeF(m_image.width() * m_scale,
                                                m_image.height() * m_scale));
        const QRectF compareRect(
                imageRect.left() + imageRect.width() * m_compareSplit,
                imageRect.top(), imageRect.width() * (1.0 - m_compareSplit),
                imageRect.height());
        painter.save();
        painter.setClipRect(compareRect);
        painter.drawImage(imageRect, m_compareImage);
        painter.restore();
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2.0));
        const double divider =
                imageRect.left() + imageRect.width() * m_compareSplit;
        painter.drawLine(QPointF(divider, imageRect.top()),
                         QPointF(divider, imageRect.bottom()));
    }
    painter.save();
    painter.translate(m_origin);
    painter.scale(m_scale, m_scale);
    painter.setOpacity(m_shapeOpacity / 100.0);
    const auto drawMask = [this, &painter](
                                  const QImage& mask, const QColor& color,
                                  const AnnotationShape* shape = nullptr) {
        painter.save();
        painter.setOpacity(m_shapeOpacity / 100.0 * m_maskOpacity / 255.0);
        const QImage silhouette =
                shape ? shapeSilhouette(*shape, color)
                      : tintedSilhouette(mask, color, m_brushSilhouetteCache,
                                         m_brushSilhouetteKey,
                                         m_brushSilhouetteColor);
        painter.drawImage(QPointF(0, 0), silhouette);
        painter.restore();
    };

    // These overlays intentionally consume the source-compatible JSON
    // metadata kept in AnnotationShape::extra. They remain display-only and
    // therefore do not alter the annotation or AICore contracts.
    QHash<int, QRectF> groupBounds;
    QHash<int, int> groupCounts;
    if (m_showGroups || m_showLinking) {
        for (const AnnotationShape& shape : m_shapes) {
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (!group.isDouble() || !shape.visible || shape.points.isEmpty())
                continue;
            const int groupId = group.toInt();
            const QRectF bounds = shapeBounds(shape);
            if (groupBounds.contains(groupId))
                groupBounds[groupId] =
                        groupBounds.value(groupId).united(bounds);
            else
                groupBounds.insert(groupId, bounds);
            ++groupCounts[groupId];
        }
    }
    if (m_showLinking) {
        painter.setPen(QPen(QColor(255, 165, 0, 220), 2.0 / m_scale));
        for (const AnnotationShape& shape : m_shapes) {
            if (!shape.visible) continue;
            const QJsonArray links =
                    shape.extra.value(QStringLiteral("kie_linking")).toArray();
            for (const QJsonValue& linkValue : links) {
                const QJsonArray link = linkValue.toArray();
                if (link.size() != 2 || !link[0].isDouble() ||
                    !link[1].isDouble() ||
                    !groupBounds.contains(link[0].toInt()) ||
                    !groupBounds.contains(link[1].toInt()))
                    continue;
                const QPointF from =
                        groupBounds.value(link[0].toInt()).center();
                const QPointF to = groupBounds.value(link[1].toInt()).center();
                painter.drawLine(from, to);
                const QLineF line(from, to);
                if (line.length() < 1.0) continue;
                const QPointF tip = line.p2();
                const QPointF unit =
                        line.unitVector().p2() - line.unitVector().p1();
                const QPointF normal(-unit.y(), unit.x());
                painter.setBrush(QColor(255, 165, 0, 220));
                QPolygonF arrow;
                arrow << tip << tip - unit * 10.0 + normal * 4.0
                      << tip - unit * 10.0 - normal * 4.0;
                painter.drawPolygon(arrow);
                painter.setBrush(Qt::NoBrush);
            }
        }
    }
    if (m_showGroups) {
        painter.setPen(
                QPen(QColor(255, 165, 0, 220), 1.5 / m_scale, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        for (auto it = groupBounds.constBegin(); it != groupBounds.constEnd();
             ++it) {
            const QRectF bounds = it.value();
            painter.drawRect(bounds.adjusted(-4.0 / m_scale, -4.0 / m_scale,
                                             4.0 / m_scale, 4.0 / m_scale));
            painter.save();
            painter.resetTransform();
            const QPointF anchor = m_origin + bounds.topLeft() * m_scale;
            const QString title = QStringLiteral("G%1 · S%2")
                                          .arg(it.key())
                                          .arg(groupCounts.value(it.key()));
            painter.setPen(Qt::white);
            painter.fillRect(QRectF(anchor.x(), anchor.y() - 18,
                                    QTCOMPAT_FONTMETRICS_WIDTH(
                                            painter.fontMetrics(), title) +
                                            8,
                                    17),
                             QColor(120, 70, 0, 220));
            painter.drawText(anchor + QPointF(4, -5), title);
            painter.restore();
        }
    }
    if (m_showMasks) {
        for (int index = 0; index < m_shapes.size(); ++index) {
            const AnnotationShape& shape = m_shapes[index];
            const bool selected = m_selectedIndices.contains(index);
            if (!shape.visible || (m_brushEditing && index == m_brushTarget))
                continue;
            drawGeometryMask(painter, shape,
                             effectiveLineColor(shape, selected),
                             m_shapeLineWidth / m_scale, m_maskOpacity,
                             !(m_dragging && selected));
        }
    }
    for (int index = 0; index < m_shapes.size(); ++index) {
        const AnnotationShape& shape = m_shapes[index];
        if (!shape.visible || (shape.points.isEmpty() && shape.mask.isNull()))
            continue;
        const bool selected = m_selectedIndices.contains(index);
        if (m_brushEditing && index == m_brushTarget) {
            if (m_showMasks)
                drawMask(m_brushMask, effectiveLineColor(shape, selected));
            continue;
        }
        if (m_showMasks && !shape.mask.isNull())
            drawMask(shape.mask, effectiveLineColor(shape, selected), &shape);
        if (m_showShapes && !shape.points.isEmpty()) {
            const QColor line = effectiveLineColor(shape, selected);
            QPen pen = annotationOutlinePen(shape, line, m_shapeLineWidth);
            pen.setWidthF((selected ? std::max(2.5, m_shapeLineWidth)
                                    : m_shapeLineWidth) /
                          m_scale);
            painter.setPen(pen);
            QColor fill = effectiveLineColor(shape, false);
            fill.setAlpha(selected ? 155 : 128);
            if (selected &&
                m_selectFillColor !=
                        annotationConfig::color("shape/select_fill_color"))
                fill = m_selectFillColor;
            if (!selected &&
                m_fillColor != annotationConfig::color("shape/fill_color"))
                fill = m_fillColor;
            const bool isInProgress =
                    m_creating && index == m_shapes.size() - 1;
            const bool fillShape =
                    closedType(shape.type) && shape.mask.isNull() &&
                    m_fillDrawing &&
                    (isInProgress ||
                     ((selected || index == m_hoverShape) && !m_dragging));
            painter.setBrush(fillShape ? QBrush(fill) : QBrush(Qt::NoBrush));
            QPainterPath path;
            path.moveTo(shape.points.first());
            if (shape.type == QStringLiteral("circle") &&
                shape.points.size() >= 2) {
                const QRectF bounds = circleBounds(shape);
                painter.drawEllipse(bounds.normalized());
            } else if (shape.type == QStringLiteral("point")) {
                // The vertex marker is painted below with its fill color.
            } else if (isCuboid(shape)) {
                drawCuboid(painter, shape.points);
            } else {
                for (int i = 1; i < shape.points.size(); ++i)
                    path.lineTo(shape.points[i]);
                if (closedType(shape.type) && shape.points.size() > 2)
                    path.closeSubpath();
                painter.drawPath(path);
            }
            if (!selected && shape.mask.isNull() &&
                (shape.type == QStringLiteral("point") ||
                 (shape.type == QStringLiteral("polygon") &&
                  (isInProgress ||
                   (index == m_hoverShape && m_hoverVertex == 0))) ||
                 shape.type == QStringLiteral("quadrilateral") ||
                 shape.type == QStringLiteral("line") ||
                 shape.type == QStringLiteral("linestrip"))) {
                painter.setBrush(effectiveLineColor(shape, false));
                const int count = shape.type == QStringLiteral("linestrip")
                                          ? shape.points.size()
                                          : 1;
                for (int i = 0; i < count; ++i)
                    drawAnnotationVertex(painter, shape, shape.points[i],
                                         m_shapePointSize / (2.0 * m_scale));
            }
            if (selected || (isCuboid(shape) && index == m_hoverShape)) {
                if (shape.mask.isNull()) {
                    QVector<int> controls;
                    if (isCuboid(shape))
                        controls =
                                annotationCuboid::visibleControls(shape.points);
                    else
                        for (int i = 0; i < shape.points.size(); ++i)
                            controls.append(i);
                    for (int control : controls) {
                        const QPointF point =
                                isCuboid(shape)
                                        ? annotationCuboid::controlPoint(
                                                  shape.points, control)
                                        : shape.points[control];
                        painter.setBrush(
                                index == m_hoverShape &&
                                                control == m_hoverVertex
                                        ? m_hoverVertexFillColor
                                        : (m_vertexFillColor ==
                                                           annotationConfig::
                                                                   color("shape"
                                                                         "/vert"
                                                                         "ex_"
                                                                         "fill_"
                                                                         "colo"
                                                                         "r")
                                                   ? effectiveLineColor(shape,
                                                                        false)
                                                   : m_vertexFillColor));
                        const double radius =
                                m_shapePointSize / (2.0 * m_scale);
                        drawAnnotationVertex(painter, shape, point, radius);
                    }
                }
                if (shape.type == QStringLiteral("rotation") &&
                    shape.points.size() == 4 && !shape.locked) {
                    const QPointF handle = rotationHandle(shape);
                    const QPointF midpoint =
                            (shape.points[0] + shape.points[1]) / 2.0;
                    painter.drawLine(midpoint, handle);
                    painter.setBrush(QColor(20, 25, 28));
                    painter.drawEllipse(handle, 5.0 / m_scale, 5.0 / m_scale);
                }
            }
        }
        if (m_showShapes || m_showDegrees)
            drawRotationDirection(painter, shape, m_scale, m_shapePointSize,
                                  m_showDegrees);
    }
    QRect labelBounds = visibleRegion().boundingRect();
    if (labelBounds.isEmpty()) labelBounds = rect();
    drawAnnotationLabels(painter, *this, m_shapes, visualizationOptions(),
                         labelBounds, m_brushEditing ? m_brushTarget : -1);
    if (m_showAttributes)
        for (int index = 0; index < m_shapes.size(); ++index)
            if (m_shapes[index].visible &&
                !(m_brushEditing && index == m_brushTarget))
                drawAnnotationAttributes(painter, m_shapes[index], m_scale,
                                         m_shapePointSize);
    // RelateAnything edges: curved arrow from the subject box centre to the
    // object box centre, coloured by the subject, with a predicate chip.
    if (m_showShapes && !m_relations.isEmpty()) {
        for (int row = 0; row < m_relations.size(); ++row) {
            const QJsonObject edge = m_relations.at(row).toObject();
            const int from =
                    edge.value(QStringLiteral("subject_index")).toInt(-1);
            const int to = edge.value(QStringLiteral("object_index")).toInt(-1);
            if (from < 0 || to < 0 || from >= m_shapes.size() ||
                to >= m_shapes.size() || from == to ||
                !m_shapes[from].visible || !m_shapes[to].visible ||
                m_shapes[from].points.isEmpty() ||
                m_shapes[to].points.isEmpty())
                continue;
            const QPointF start =
                    QPolygonF(m_shapes[from].points).boundingRect().center();
            const QPointF end =
                    QPolygonF(m_shapes[to].points).boundingRect().center();
            const QPointF delta = end - start;
            const double length = std::hypot(delta.x(), delta.y());
            if (length < 1.0) continue;
            const bool highlighted = row == m_highlightedRelation;
            const QColor color = effectiveLineColor(m_shapes[from], false);
            // Bow each edge by a row-dependent amount so parallel edges
            // between the same objects stay distinguishable.
            const QPointF normal(-delta.y() / length, delta.x() / length);
            const QPointF control =
                    (start + end) * 0.5 +
                    normal * ((18.0 + 8.0 * (row % 3)) / m_scale);
            QPainterPath path(start);
            path.quadTo(control, end);
            painter.save();
            painter.setOpacity(1.0);
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(QColor(0, 0, 0, 160),
                                (highlighted ? 4.5 : 3.0) / m_scale));
            painter.drawPath(path);
            painter.setPen(QPen(color, (highlighted ? 3.0 : 1.8) / m_scale));
            painter.drawPath(path);
            const QPointF tangent = end - control;
            const double tangentLength = std::hypot(tangent.x(), tangent.y());
            if (tangentLength > 1e-3) {
                const QPointF unit = tangent / tangentLength;
                const QPointF side(-unit.y(), unit.x());
                const double size = 11.0 / m_scale;
                painter.setPen(Qt::NoPen);
                painter.setBrush(color);
                painter.drawPolygon(
                        QPolygonF({end, end - unit * size + side * size * 0.45,
                                   end - unit * size - side * size * 0.45}));
            }
            painter.restore();
            if (m_showLabels) {
                const QPointF mid = 0.25 * start + 0.5 * control + 0.25 * end;
                const QString text =
                        edge.value(QStringLiteral("predicate")).toString() +
                        QStringLiteral(" ") +
                        QString::number(
                                edge.value(QStringLiteral("score")).toDouble(),
                                'f', 2);
                painter.save();
                painter.setOpacity(1.0);
                painter.resetTransform();
                const QPointF anchor = m_origin + mid * m_scale;
                const QRectF box =
                        QRectF(painter.fontMetrics().boundingRect(text))
                                .adjusted(-5, -3, 5, 3);
                const QRectF chip(anchor.x() - box.width() / 2,
                                  anchor.y() - box.height() / 2, box.width(),
                                  box.height());
                painter.setPen(Qt::NoPen);
                painter.setBrush(QColor(22, 24, 28, 225));
                painter.drawRoundedRect(chip, 4, 4);
                painter.setPen(Qt::white);
                painter.drawText(chip, Qt::AlignCenter, text);
                painter.restore();
            }
        }
    }
    if (m_showMasks && m_brushEditing && m_brushTarget < 0)
        drawMask(m_brushMask, m_lineColor);
    if (m_creating && !m_shapes.isEmpty()) {
        const AnnotationShape& shape = m_shapes.last();
        painter.setPen(
                QPen(QColor(0, 255, 0, 180), 1.5 / m_scale, Qt::DashLine));
        QColor previewFill(0, 255, 0, 35);
        painter.setBrush(m_fillDrawing && closedType(shape.type)
                                 ? QBrush(previewFill)
                                 : QBrush(Qt::NoBrush));
        if (shape.type == QStringLiteral("polygon") ||
            shape.type == QStringLiteral("linestrip")) {
            if (m_fillDrawing && shape.points.size() >= 2) {
                QPainterPath preview;
                preview.moveTo(shape.points.first());
                for (int i = 1; i < shape.points.size(); ++i)
                    preview.lineTo(shape.points[i]);
                preview.lineTo(m_hover);
                if (shape.type == QStringLiteral("polygon"))
                    preview.closeSubpath();
                painter.drawPath(preview);
            } else {
                painter.drawLine(shape.points.last(), m_hover);
            }
        }
    }
    if (m_magicActive && m_magicPolygon.size() >= 3) {
        QPolygonF polygon;
        for (const QPointF& point : m_magicPolygon) polygon << point;
        painter.setPen(QPen(Qt::white, 2.0 / m_scale));
        QColor magicColor(0, 180, 255);
        magicColor.setAlphaF(m_magicOpacity);
        painter.setBrush(magicColor);
        painter.drawPolygon(polygon);
    }
    if (m_prompt.useBox) {
        painter.setPen(QPen(QColor(0, 210, 245), 2.0 / m_scale, Qt::DashLine));
        painter.setBrush(QColor(0, 210, 245, 35));
        painter.drawRect(m_prompt.box);
    }
    for (const QPointF& point : m_prompt.positive) {
        painter.setPen(QPen(Qt::white, 1.5 / m_scale));
        painter.setBrush(QColor(0, 200, 90));
        painter.drawEllipse(point, 5.0 / m_scale, 5.0 / m_scale);
    }
    for (const QPointF& point : m_prompt.negative) {
        painter.setPen(QPen(Qt::white, 1.5 / m_scale));
        painter.setBrush(QColor(235, 70, 65));
        painter.drawEllipse(point, 5.0 / m_scale, 5.0 / m_scale);
    }
    if (m_brushEditing) {
        painter.setBrush(Qt::NoBrush);
        const QColor cursor = m_brushOperation == 1   ? QColor(255, 100, 100)
                              : m_brushOperation == 2 ? QColor(255, 210, 65)
                                                      : QColor(Qt::white);
        painter.setPen(QPen(cursor, 1.5 / m_scale, Qt::DashLine));
        painter.drawEllipse(m_hover, m_brushRadius, m_brushRadius);
    }
    if (m_crosshairVisible && m_tool == QStringLiteral("edit")) {
        const QPointF cursor = m_origin + m_hover * m_scale;
        if (rect().contains(cursor.toPoint())) {
            painter.save();
            painter.resetTransform();
            QColor crosshair = m_crosshairColor;
            crosshair.setAlphaF(m_crosshairOpacity);
            painter.setPen(QPen(crosshair, m_crosshairWidth));
            painter.drawLine(QPointF(cursor.x() - 8.0, cursor.y()),
                             QPointF(cursor.x() + 8.0, cursor.y()));
            painter.drawLine(QPointF(cursor.x(), cursor.y() - 8.0),
                             QPointF(cursor.x(), cursor.y() + 8.0));
            painter.restore();
        }
    }
    painter.restore();
}

void AnnotationCanvas::setBrushRadius(int radius) {
    const int next = std::max(1, std::min(200, radius));
    if (next == m_brushRadius) return;
    m_brushRadius = next;
    emit brushRadiusChanged(next);
    update();
}

bool AnnotationCanvas::setBrushEditing(bool enabled, bool cancel) {
    if (enabled) {
        if (m_brushEditing) return true;
        if (m_image.isNull() ||
            (m_selected < 0 && m_brushLabel.trimmed().isEmpty()) ||
            (m_selected >= 0 &&
             (m_selected >= m_shapes.size() ||
              m_shapes[m_selected].type != QStringLiteral("polygon") ||
              m_shapes[m_selected].locked)))
            return false;
        m_brushTarget = m_selected;
        if (m_brushTarget >= 0 && m_brushClassId < 0 &&
            m_brushLabel == m_shapes[m_brushTarget].label)
            m_brushClassId = m_shapes[m_brushTarget]
                                     .extra.value(QStringLiteral("class_id"))
                                     .toInt(-1);
        m_brushMask =
                m_brushTarget >= 0 ? m_shapes[m_brushTarget].mask : QImage();
        if (!m_brushMask.isNull() && m_brushMask.size() != m_image.size())
            m_brushMask =
                    m_brushMask.scaled(m_image.size(), Qt::IgnoreAspectRatio,
                                       Qt::FastTransformation);
        if (!m_brushMask.isNull() &&
            m_brushMask.format() != QImage::Format_ARGB32_Premultiplied)
            m_brushMask = m_brushMask.convertToFormat(
                    QImage::Format_ARGB32_Premultiplied);
        if (m_brushMask.isNull()) {
            m_brushMask =
                    QImage(m_image.size(), QImage::Format_ARGB32_Premultiplied);
            m_brushMask.fill(Qt::transparent);
            if (m_brushTarget >= 0) {
                QPainter painter(&m_brushMask);
                painter.setPen(Qt::NoPen);
                painter.setBrush(QColor(0, 180, 255, 110));
                QPolygonF polygon;
                for (const QPointF& point : m_shapes[m_brushTarget].points)
                    polygon << point;
                painter.drawPolygon(polygon);
            }
        }
        m_brushHistory = {m_brushMask};
        m_brushRedo.clear();
        m_brushEditing = true;
        m_brushStrokeActive = false;
        emit brushEditingChanged(true);
        emit brushHistoryChanged(false, false);
        update();
        return true;
    }
    if (!m_brushEditing) return true;
    finishBrushStroke();
    const bool validTarget =
            m_brushTarget >= 0 && m_brushTarget < m_shapes.size();
    const bool newTarget = m_brushTarget < 0;
    const bool labelChanged = validTarget && !m_brushLabel.isEmpty() &&
                              m_shapes[m_brushTarget].label != m_brushLabel;
    const QJsonValue oldClassId = validTarget
                                          ? m_shapes[m_brushTarget].extra.value(
                                                    QStringLiteral("class_id"))
                                          : QJsonValue();
    const bool classChanged =
            !newTarget &&
            (oldClassId.isDouble() ? oldClassId.toInt() != m_brushClassId
                                   : m_brushClassId >= 0);
    const bool changed = !cancel && (validTarget || newTarget) &&
                         !m_brushHistory.isEmpty() &&
                         (m_brushMask != m_brushHistory.first() ||
                          labelChanged || classChanged);
    if (changed) {
        const QVector<QPointF> polygon = maskOutline(m_brushMask);
        if (newTarget && polygon.size() < 3) {
            m_brushEditing = false;
            m_brushTarget = -1;
            m_brushClassId = -1;
            m_brushMask = QImage();
            m_brushHistory.clear();
            m_brushRedo.clear();
            emit brushEditingChanged(false);
            emit brushHistoryChanged(false, false);
            update();
            return true;
        }
        emit editStarted();
        if (newTarget) {
            AnnotationShape shape;
            shape.type = QStringLiteral("polygon");
            shape.label = m_brushLabel.trimmed();
            shape.points = polygon;
            shape.mask = m_brushMask;
            if (m_brushClassId >= 0)
                shape.extra.insert(QStringLiteral("class_id"), m_brushClassId);
            m_brushTarget = m_shapes.size();
            m_shapes.append(shape);
        }
        if (polygon.size() >= 3) {
            m_shapes[m_brushTarget].points = polygon;
            m_shapes[m_brushTarget].mask = m_brushMask;
            if (!m_brushLabel.isEmpty())
                m_shapes[m_brushTarget].label = m_brushLabel;
            if (m_brushClassId >= 0)
                m_shapes[m_brushTarget].extra.insert(QStringLiteral("class_id"),
                                                     m_brushClassId);
            else
                m_shapes[m_brushTarget].extra.remove(
                        QStringLiteral("class_id"));
        }
        if (m_brushClassId >= 0) {
            int left = m_brushMask.width(), top = m_brushMask.height();
            int right = -1, bottom = -1;
            const QImage& originalMask = m_brushHistory.first();
            for (int y = 0; y < m_brushMask.height(); ++y) {
                const QRgb* painted = reinterpret_cast<const QRgb*>(
                        m_brushMask.constScanLine(y));
                const QRgb* original = reinterpret_cast<const QRgb*>(
                        originalMask.constScanLine(y));
                for (int x = 0; x < m_brushMask.width(); ++x) {
                    if ((qAlpha(painted[x]) > 0) == (qAlpha(original[x]) > 0))
                        continue;
                    left = std::min(left, x);
                    top = std::min(top, y);
                    right = std::max(right, x);
                    bottom = std::max(bottom, y);
                }
            }
            QVector<int> emptyClasses;
            for (int i = 0; i < m_shapes.size(); ++i) {
                if (right < left || i == m_brushTarget || m_shapes[i].locked ||
                    !m_shapes[i]
                             .extra.value(QStringLiteral("class_id"))
                             .isDouble() ||
                    m_shapes[i].mask.size() != m_brushMask.size())
                    continue;
                const int otherClassId =
                        m_shapes[i]
                                .extra.value(QStringLiteral("class_id"))
                                .toInt();
                if (otherClassId == m_brushClassId) continue;
                bool altered = false;
                QImage& other = m_shapes[i].mask;
                if (other.format() != QImage::Format_ARGB32_Premultiplied)
                    other = other.convertToFormat(
                            QImage::Format_ARGB32_Premultiplied);
                for (int y = top; y <= bottom; ++y) {
                    QRgb* pixels = reinterpret_cast<QRgb*>(other.scanLine(y));
                    const QRgb* painted = reinterpret_cast<const QRgb*>(
                            m_brushMask.constScanLine(y));
                    const QRgb* original = reinterpret_cast<const QRgb*>(
                            originalMask.constScanLine(y));
                    for (int x = left; x <= right; ++x) {
                        if (qAlpha(original[x]) == 0 &&
                            qAlpha(painted[x]) > 0 && qAlpha(pixels[x]) > 0) {
                            pixels[x] = 0;
                            altered = true;
                        } else if (otherClassId == 0 &&
                                   m_shapes[i].label.compare(
                                           QStringLiteral("background"),
                                           Qt::CaseInsensitive) == 0 &&
                                   qAlpha(original[x]) > 0 &&
                                   qAlpha(painted[x]) == 0 &&
                                   qAlpha(pixels[x]) == 0) {
                            pixels[x] = qPremultiply(qRgba(0, 180, 255, 110));
                            altered = true;
                        }
                    }
                }
                if (!altered) continue;
                m_shapes[i].points = maskOutline(other);
                if (m_shapes[i].points.isEmpty()) emptyClasses.append(i);
            }
            std::sort(emptyClasses.begin(), emptyClasses.end(),
                      std::greater<int>());
            for (int index : emptyClasses) {
                m_shapes.removeAt(index);
                if (index < m_brushTarget) --m_brushTarget;
            }
        }
        if (polygon.size() < 3) {
            m_shapes.removeAt(m_brushTarget);
            m_selected = -1;
            m_selectedIndices.clear();
            emit shapeSelected(-1);
        } else {
            m_selected = m_brushTarget;
            m_selectedIndices = {m_brushTarget};
            emit shapeSelected(m_selected);
        }
        emit shapesChanged();
    }
    m_brushEditing = false;
    m_brushTarget = -1;
    m_brushClassId = -1;
    m_brushMask = QImage();
    m_brushHistory.clear();
    m_brushRedo.clear();
    emit brushEditingChanged(false);
    emit brushHistoryChanged(false, false);
    update();
    return true;
}

int AnnotationCanvas::effectiveBrushOperation(
        Qt::KeyboardModifiers modifiers) const {
    return modifiers & Qt::ControlModifier ? 1 : m_brushOperation;
}

void AnnotationCanvas::paintBrush(const QPointF& point, int operation) {
    if (!m_brushEditing) return;
    if (operation == 2) {
        QPainter coverage(&m_brushStrokeCoverage);
        coverage.setRenderHint(QPainter::Antialiasing, false);
        coverage.setPen(QPen(Qt::white, 2.0 * m_brushRadius, Qt::SolidLine,
                             Qt::RoundCap, Qt::RoundJoin));
        coverage.drawLine(m_brushLast, point);
        coverage.setPen(Qt::NoPen);
        coverage.setBrush(Qt::white);
        coverage.drawEllipse(point, m_brushRadius, m_brushRadius);
        coverage.end();
        const int left = std::max(
                0, int(std::floor(std::min(m_brushLast.x(), point.x()) -
                                  m_brushRadius - 1)));
        const int top = std::max(
                0, int(std::floor(std::min(m_brushLast.y(), point.y()) -
                                  m_brushRadius - 1)));
        const int right =
                std::min(m_brushMask.width() - 1,
                         int(std::ceil(std::max(m_brushLast.x(), point.x()) +
                                       m_brushRadius + 1)));
        const int bottom =
                std::min(m_brushMask.height() - 1,
                         int(std::ceil(std::max(m_brushLast.y(), point.y()) +
                                       m_brushRadius + 1)));
        const QRgb painted = qPremultiply(qRgba(0, 180, 255, 110));
        for (int y = top; y <= bottom; ++y) {
            const uchar* coverageRow = m_brushStrokeCoverage.constScanLine(y);
            const QRgb* baseline = reinterpret_cast<const QRgb*>(
                    m_brushStrokeBaseline.constScanLine(y));
            QRgb* mask = reinterpret_cast<QRgb*>(m_brushMask.scanLine(y));
            for (int x = left; x <= right; ++x)
                if (coverageRow[x] != 0)
                    mask[x] = qAlpha(baseline[x]) > 0 ? 0 : painted;
        }
        m_brushLast = point;
        update();
        return;
    }
    const bool erase = operation == 1;
    QPainter painter(&m_brushMask);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setCompositionMode(erase ? QPainter::CompositionMode_Clear
                                     : QPainter::CompositionMode_Source);
    const QColor color =
            erase ? QColor(0, 0, 0, 255) : QColor(0, 180, 255, 110);
    painter.setPen(QPen(color, 2.0 * m_brushRadius, Qt::SolidLine, Qt::RoundCap,
                        Qt::RoundJoin));
    painter.drawLine(m_brushLast, point);
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawEllipse(point, m_brushRadius, m_brushRadius);
    m_brushLast = point;
    update();
}

void AnnotationCanvas::finishBrushStroke() {
    if (!m_brushEditing || !m_brushStrokeActive) return;
    m_brushStrokeActive = false;
    m_brushStrokeBaseline = QImage();
    m_brushStrokeCoverage = QImage();
    if (m_brushHistory.isEmpty() || m_brushMask == m_brushHistory.last())
        return;
    m_brushHistory.append(m_brushMask);
    m_brushRedo.clear();
    while (m_brushHistory.size() > m_brushMaxUndoSteps + 1 ||
           (m_brushHistory.size() > 2 &&
            quint64(m_brushHistory.size()) * m_brushMask.sizeInBytes() >
                    quint64(m_brushMaxUndoMemoryMb) * 1024 * 1024))
        m_brushHistory.removeAt(1);
    emit brushHistoryChanged(m_brushHistory.size() > 1, false);
}

void AnnotationCanvas::brushUndo() {
    if (!m_brushEditing || m_brushHistory.size() < 2) return;
    m_brushRedo.append(m_brushHistory.takeLast());
    m_brushMask = m_brushHistory.last();
    emit brushHistoryChanged(m_brushHistory.size() > 1, true);
    update();
}

void AnnotationCanvas::brushRedo() {
    if (!m_brushEditing || m_brushRedo.isEmpty()) return;
    m_brushMask = m_brushRedo.takeLast();
    m_brushHistory.append(m_brushMask);
    emit brushHistoryChanged(true, !m_brushRedo.isEmpty());
    update();
}

void AnnotationCanvas::startMagicWand(const QPointF& imagePoint,
                                      const QPointF& screenPoint) {
    clearMagicWand();
    const int seedX = int(std::round(imagePoint.x()));
    const int seedY = int(std::round(imagePoint.y()));
    if (seedX < 0 || seedY < 0 || seedX >= m_image.width() ||
        seedY >= m_image.height())
        return;
    const QImage rgb = m_image.convertToFormat(QImage::Format_RGB888);
    const LabColor seed = toLab(rgb.constScanLine(seedY) + seedX * 3);
    m_magicDistances.resize(rgb.width() * rgb.height());
    for (int y = 0; y < rgb.height(); ++y) {
        const uchar* row = rgb.constScanLine(y);
        for (int x = 0; x < rgb.width(); ++x) {
            const LabColor color = toLab(row + x * 3);
            const double dl = (color.l - seed.l) * m_magicLuminanceWeight;
            m_magicDistances[y * rgb.width() + x] = float(std::sqrt(
                    dl * dl + (color.a - seed.a) * (color.a - seed.a) +
                    (color.b - seed.b) * (color.b - seed.b)));
        }
    }
    m_magicActive = true;
    m_magicAnchor = screenPoint;
    m_start = QPointF(seedX, seedY);
    updateMagicWand(m_magicDefaultThreshold);
}

void AnnotationCanvas::updateMagicWand(int threshold) {
    if (!m_magicActive) return;
    m_magicThreshold = std::max(0, std::min(255, threshold));
    const int width = m_image.width(), height = m_image.height();
    m_magicMask.fill(0, width * height);
    QVector<int> queue;
    const int seed = int(m_start.y()) * width + int(m_start.x());
    queue.append(seed);
    m_magicMask[seed] = 1;
    for (int head = 0; head < queue.size(); ++head) {
        const int index = queue[head];
        const int x = index % width, y = index / width;
        const auto enqueue = [this, &queue](int next) {
            if (m_magicMask[next] == 0 &&
                m_magicDistances[next] <= m_magicThreshold) {
                m_magicMask[next] = 1;
                queue.append(next);
            }
        };
        if (x > 0) enqueue(index - 1);
        if (x + 1 < width) enqueue(index + 1);
        if (y > 0) enqueue(index - width);
        if (y + 1 < height) enqueue(index + width);
    }
    m_magicPolygon = largestMaskPolygon(m_magicMask.constData(), width, height,
                                        size_t(width));
    update();
}

void AnnotationCanvas::clearMagicWand() {
    m_magicActive = false;
    m_magicDistances.clear();
    m_magicMask.clear();
    m_magicPolygon.clear();
    update();
}

void AnnotationCanvas::finishMagicWand() {
    if (!m_magicActive) return;
    const QVector<QPointF> polygon =
            simplifyPolygon(m_magicPolygon, m_magicSimplifyEpsilon);
    clearMagicWand();
    if (polygon.size() < 3) return;
    emit editStarted();
    AnnotationShape shape;
    shape.type = QStringLiteral("polygon");
    shape.label = m_label;
    shape.points = polygon;
    m_shapes.append(shape);
    selectShape(m_shapes.size() - 1);
    emit shapeFinished(m_shapes.size() - 1);
    emit shapesChanged();
    update();
}

int AnnotationCanvas::hitShape(const QPointF& point,
                               int* vertex,
                               annotationCuboid::Face* face) const {
    *vertex = -1;
    if (face) *face = annotationCuboid::Face::None;
    const double radius = std::max(0.5, m_hitEpsilon) / m_scale;
    for (int i = m_shapes.size() - 1; i >= 0; --i) {
        const AnnotationShape& shape = m_shapes[i];
        if (!shape.visible) continue;
        if (isCuboid(shape) && shape.mask.isNull()) {
            if (!shape.locked) {
                *vertex = annotationCuboid::nearestControl(shape.points, point,
                                                           radius);
                if (*vertex >= 0) return i;
            }
            const auto hit = annotationCuboid::hitFace(shape.points, point);
            if (hit != annotationCuboid::Face::None) {
                if (face && !shape.locked) *face = hit;
                return i;
            }
            continue;
        }
        for (int j = 0; shape.mask.isNull() && j < shape.points.size(); ++j) {
            if (QLineF(point, shape.points[j]).length() <= radius) {
                *vertex = j;
                return i;
            }
        }
        if (!shape.mask.isNull()) {
            const int x = int(point.x()), y = int(point.y());
            if (x >= 0 && y >= 0 && x < shape.mask.width() &&
                y < shape.mask.height() && qAlpha(shape.mask.pixel(x, y)) > 0)
                return i;
            continue;
        }
        QPainterPath path;
        if (shape.points.isEmpty()) continue;
        path.moveTo(shape.points.first());
        for (int j = 1; j < shape.points.size(); ++j)
            path.lineTo(shape.points[j]);
        if (shape.type == QStringLiteral("circle") &&
            shape.points.size() >= 2) {
            path = QPainterPath();
            path.addEllipse(circleBounds(shape));
        } else if (closedType(shape.type))
            path.closeSubpath();
        QPainterPathStroker stroker;
        stroker.setWidth(2.0 * radius);
        if (path.contains(point) || stroker.createStroke(path).contains(point))
            return i;
    }
    return -1;
}

int AnnotationCanvas::hitGroup(const QPointF& point) const {
    QHash<int, QRectF> bounds;
    for (const AnnotationShape& shape : m_shapes) {
        if (!shape.visible || shape.points.isEmpty()) continue;
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        if (!group.isDouble()) continue;
        const int groupId = group.toInt();
        const QRectF shapeBounds = QPolygonF(shape.points).boundingRect();
        bounds[groupId] = bounds.contains(groupId)
                                  ? bounds.value(groupId).united(shapeBounds)
                                  : shapeBounds;
    }
    int hit = -1;
    double hitArea = std::numeric_limits<double>::max();
    const double tolerance = std::max(0.5, m_hitEpsilon / m_scale);
    for (auto it = bounds.constBegin(); it != bounds.constEnd(); ++it) {
        const QRectF outer = it.value().adjusted(-tolerance, -tolerance,
                                                 tolerance, tolerance);
        if (!outer.contains(point)) continue;
        const double area =
                std::max(1.0, it.value().width() * it.value().height());
        if (area < hitArea) {
            hit = it.key();
            hitArea = area;
        }
    }
    return hit;
}

void AnnotationCanvas::moveVertex(AnnotationShape& shape,
                                  int vertex,
                                  const QPointF& point) {
    if (shape.locked) return;
    if (isCuboid(shape)) {
        if (annotationCuboid::moveControl(shape.points, vertex, point,
                                          m_cuboidMinDepth))
            syncCuboidDepth(shape);
        return;
    }
    if (shape.type == QStringLiteral("rectangle") && shape.points.size() == 4) {
        const QPointF opposite = shape.points[(vertex + 2) % 4];
        shape.points[(vertex + 1) % 4] =
                vertex % 2 == 0 ? QPointF(opposite.x(), point.y())
                                : QPointF(point.x(), opposite.y());
        shape.points[(vertex + 3) % 4] =
                vertex % 2 == 0 ? QPointF(point.x(), opposite.y())
                                : QPointF(opposite.x(), point.y());
        shape.points[vertex] = point;
        return;
    }
    if (shape.type == QStringLiteral("rotation") && shape.points.size() == 4) {
        const QPointF u = shape.points[(vertex + 1) % 4] - shape.points[vertex];
        const QPointF v = shape.points[(vertex + 3) % 4] - shape.points[vertex];
        const double uLength = std::hypot(u.x(), u.y());
        const double vLength = std::hypot(v.x(), v.y());
        if (uLength > 1e-6 && vLength > 1e-6) {
            const QPointF diagonal = shape.points[(vertex + 2) % 4] - point;
            shape.points[(vertex + 1) % 4] =
                    point + u * (dot(diagonal, u) / (uLength * uLength));
            shape.points[(vertex + 3) % 4] =
                    point + v * (dot(diagonal, v) / (vLength * vLength));
        }
    }
    shape.points[vertex] = point;
}

void AnnotationCanvas::cancelShape() {
    if (!m_creating) return;
    m_shapes.removeLast();
    m_creating = false;
    selectShape(-1);
    emit creationCanceled();
    update();
}

void AnnotationCanvas::mousePressEvent(QMouseEvent* event) {
    if (m_image.isNull()) return;
    updateTransform();
    const QPointF screenPoint = qtCompatMouseEventPos(event);
    if (event->button() == Qt::MiddleButton ||
        (event->button() == Qt::LeftButton && m_spaceHeld)) {
        m_panning = true;
        m_panStart = screenPoint;
        setCursor(Qt::ClosedHandCursor);
        return;
    }
    const QPointF rawPoint = toImage(screenPoint);
    const QPointF point = clamp(rawPoint);
    if (m_tool == QStringLiteral("sam_point") &&
        (event->button() == Qt::LeftButton ||
         event->button() == Qt::RightButton)) {
        if (!QRectF(QPointF(), m_image.size()).contains(rawPoint)) return;
        if (event->button() == Qt::LeftButton)
            m_prompt.positive.append(point);
        else
            m_prompt.negative.append(point);
        emit promptChanged();
        update();
        return;
    }
    if (m_tool == QStringLiteral("sam_box") &&
        event->button() == Qt::LeftButton) {
        if (!QRectF(QPointF(), m_image.size()).contains(rawPoint)) return;
        m_promptBoxDragging = true;
        m_promptBoxStart = point;
        m_prompt.box = QRectF(point, point);
        m_prompt.useBox = true;
        update();
        return;
    }
    if (m_brushEditing && event->button() == Qt::RightButton) {
        setBrushEditing(false);
        return;
    }
    if (m_brushEditing) {
        if (event->button() == Qt::LeftButton) {
            m_brushStrokeActive = true;
            m_brushStrokeOperation =
                    effectiveBrushOperation(event->modifiers());
            if (m_brushStrokeOperation == 2) {
                m_brushStrokeBaseline = m_brushMask;
                m_brushStrokeCoverage =
                        QImage(m_brushMask.size(), QImage::Format_Grayscale8);
                m_brushStrokeCoverage.fill(0);
            }
            m_brushLast = point;
            m_hover = point;
            paintBrush(point, m_brushStrokeOperation);
        }
        return;
    }
    if (event->button() == Qt::RightButton && m_magicActive) {
        finishMagicWand();
        return;
    }
    if (event->button() == Qt::RightButton && m_creating) {
        finishShape();
        return;
    }
    if (event->button() != Qt::LeftButton) return;
    setFocus();
    if (m_tool == QStringLiteral("magic_wand")) {
        startMagicWand(rawPoint, screenPoint);
        return;
    }
    if (m_tool == QStringLiteral("edit")) {
        if (m_selected >= 0 && m_selected < m_shapes.size() &&
            !m_shapes[m_selected].locked &&
            m_shapes[m_selected].type == QStringLiteral("rotation")) {
            const QPointF handle = rotationHandle(m_shapes[m_selected]);
            if (QLineF(rawPoint, handle).length() <= 10.0 / m_scale) {
                m_rotating = true;
                m_dragChanged = false;
                m_rotationPoints = m_shapes[m_selected].points;
                const QPointF center =
                        (m_rotationPoints[0] + m_rotationPoints[2]) / 2.0;
                m_rotationStart = std::atan2(rawPoint.y() - center.y(),
                                             rawPoint.x() - center.x());
                return;
            }
        }
        int vertex = -1;
        annotationCuboid::Face face = annotationCuboid::Face::None;
        int index = hitShape(rawPoint, &vertex, &face);
        if (event->modifiers() & Qt::ControlModifier) {
            QVector<int> selected;
            selected.reserve(m_selectedIndices.size());
            for (int value : m_selectedIndices) selected.append(value);
            if (index >= 0) {
                if (m_selectedIndices.contains(index))
                    selected.removeAll(index);
                else
                    selected.append(index);
            }
            setSelectedShapes(selected);
            return;
        }
        if (index < 0 && m_showGroups) {
            const int groupId = hitGroup(rawPoint);
            if (groupId >= 0) {
                QVector<int> selected;
                for (int row = 0; row < m_shapes.size(); ++row) {
                    const QJsonValue memberGroup = m_shapes[row].extra.value(
                            QStringLiteral("group_id"));
                    if (memberGroup.isDouble() &&
                        memberGroup.toInt() == groupId)
                        selected.append(row);
                }
                setSelectedShapes(selected);
                if (!selected.isEmpty()) index = selected.last();
                vertex = -1;
            }
        }
        // Dragging empty canvas space pans the view.
        // It must stay behind the group-selection branch above so a click
        // between grouped shapes still selects the whole group.
        if (index < 0 && m_selectedIndices.isEmpty() &&
            m_tool == QStringLiteral("edit")) {
            m_panning = true;
            m_panStart = qtCompatMouseEventPos(event);
            return;
        }
        // Keep an existing multi-selection when the user presses one of its
        // members.  X-AnyLabeling moves the selected group as one unit; a
        // plain click on a selected member must therefore not collapse the
        // selection before the drag starts.
        if (index >= 0 && !(m_selectedIndices.size() > 1 &&
                            m_selectedIndices.contains(index)))
            selectShape(index);
        if (index >= 0 && !m_shapes[index].locked) {
            const bool groupDrag = m_selectedIndices.size() > 1 &&
                                   m_selectedIndices.contains(index);
            if (groupDrag) {
                for (int selected : m_selectedIndices) {
                    if (selected < 0 || selected >= m_shapes.size() ||
                        m_shapes[selected].locked)
                        return;
                }
                m_dragIndices = m_selectedIndices.values().toVector();
                std::sort(m_dragIndices.begin(), m_dragIndices.end());
                m_dragOriginalPoints.clear();
                m_dragOriginalMasks.clear();
                m_dragOriginalPoints.reserve(m_dragIndices.size());
                for (int selected : m_dragIndices) {
                    m_dragOriginalPoints.append(m_shapes[selected].points);
                    m_dragOriginalMasks.append(m_shapes[selected].mask);
                }
                m_dragStartPoint = point;
            } else {
                m_dragIndices = {index};
                m_dragOriginalPoints = {m_shapes[index].points};
                m_dragOriginalMasks = {m_shapes[index].mask};
                m_dragStartPoint = point;
            }
            m_dragging = true;
            m_dragChanged = false;
            m_dragVertex =
                    groupDrag || !m_shapes[index].mask.isNull() ? -1 : vertex;
            m_dragFace = groupDrag ? annotationCuboid::Face::None : face;
            m_last = isCuboid(m_shapes[index]) ? rawPoint : point;
        }
        return;
    }
    if (!m_creating) {
        emit editStarted();
        AnnotationShape shape;
        shape.type = m_tool;
        shape.label = m_label;
        shape.points.append(point);
        m_shapes.append(shape);
        m_creating = true;
        m_start = point;
        selectShape(m_shapes.size() - 1);
        if (m_tool == QStringLiteral("point")) finishShape();
    } else if (m_tool == QStringLiteral("polygon") ||
               m_tool == QStringLiteral("linestrip") ||
               m_tool == QStringLiteral("quadrilateral")) {
        m_shapes.last().points.append(point);
        if (m_tool == QStringLiteral("quadrilateral") &&
            m_shapes.last().points.size() == 4)
            finishShape();
        update();
    }
}

void AnnotationCanvas::mouseMoveEvent(QMouseEvent* event) {
    if (m_image.isNull()) return;
    if (m_panning) {
        m_pan += qtCompatMouseEventPos(event) - m_panStart;
        m_panStart = qtCompatMouseEventPos(event);
        emit viewChanged();
        update();
        return;
    }
    if (m_magicActive) {
        const QPointF delta = qtCompatMouseEventPos(event) - m_magicAnchor;
        const int threshold =
                m_magicDefaultThreshold +
                int(std::hypot(delta.x(), delta.y()) / m_magicDragSensitivity);
        if (threshold != m_magicThreshold) updateMagicWand(threshold);
        return;
    }
    updateTransform();
    const QPointF rawPoint = toImage(qtCompatMouseEventPos(event));
    const QPointF point = clamp(rawPoint);
    m_hover = point;
    // Match the source vertex_selected hover signal: it drives the
    // Remove-selected-point action without consuming the move event.
    if (!m_promptBoxDragging && !m_brushEditing && !m_rotating && !m_dragging) {
        int hoverVertex = -1;
        const int hoverShape = hitShape(rawPoint, &hoverVertex);
        if (hoverVertex != m_hoverVertex || hoverShape != m_hoverShape) {
            m_hoverVertex = hoverVertex;
            m_hoverShape = hoverShape;
            update();
        }
        emit vertexHovered(hoverVertex >= 0 &&
                           (hoverShape < 0 || !isCuboid(m_shapes[hoverShape])));
    } else if (m_hoverVertex != -1) {
        m_hoverVertex = -1;
    }
    if (m_promptBoxDragging) {
        m_prompt.box = QRectF(m_promptBoxStart, point).normalized();
        update();
        return;
    }
    if (m_brushEditing) {
        if (m_brushStrokeActive)
            paintBrush(point, m_brushStrokeOperation);
        else
            update();
        return;
    }
    if (m_rotating && m_selected >= 0) {
        const QPointF center =
                (m_rotationPoints[0] + m_rotationPoints[2]) / 2.0;
        const QPointF rawPoint = toImage(qtCompatMouseEventPos(event));
        double angle = std::atan2(rawPoint.y() - center.y(),
                                  rawPoint.x() - center.x()) -
                       m_rotationStart;
        if (event->modifiers() & Qt::ShiftModifier) {
            constexpr double step = 0.2617993877991494;
            angle = std::round(angle / step) * step;
        }
        if (std::abs(angle) > 1e-8) {
            if (!m_dragChanged) emit editStarted();
            m_dragChanged = true;
            AnnotationShape& shape = m_shapes[m_selected];
            for (int i = 0; i < 4; ++i)
                shape.points[i] = rotated(m_rotationPoints[i], center, angle);
        }
    } else if (m_dragging && m_selected >= 0) {
        const AnnotationShape& current = qAsConst(m_shapes)[m_selected];
        if (isCuboid(current) && m_dragIndices.size() == 1) {
            auto next = current.points;
            bool changed = false;
            if (m_dragVertex >= 0)
                changed = annotationCuboid::moveControl(
                        next, m_dragVertex, rawPoint, m_cuboidMinDepth);
            else if (m_dragFace == annotationCuboid::Face::Front) {
                const QPointF delta = rawPoint - m_last;
                changed = !delta.isNull();
                for (QPointF& item : next) item += delta;
            } else
                changed = annotationCuboid::moveFace(
                        next, m_dragFace, rawPoint - m_last, m_cuboidMinDepth);
            if (changed) {
                if (!m_dragChanged) emit editStarted();
                // snapshot() shares the outer QVector until the next mutable
                // access. Acquire the reference after the signal so editing
                // cannot mutate the undo snapshot through an old reference.
                AnnotationShape& edited = m_shapes[m_selected];
                edited.points = next;
                syncCuboidDepth(edited);
                m_dragChanged = true;
            }
            m_last = rawPoint;
            update();
            return;
        }
        if (QLineF(point, m_last).length() < 1e-8) return;
        if (!m_dragChanged) emit editStarted();
        m_dragChanged = true;
        AnnotationShape& shape = m_shapes[m_selected];
        if (m_dragVertex >= 0)
            moveVertex(shape, m_dragVertex, point);
        else if (m_dragIndices.size() > 1 &&
                 m_dragOriginalPoints.size() == m_dragIndices.size()) {
            QPointF delta = point - m_dragStartPoint;
            double minX = double(m_image.width()),
                   minY = double(m_image.height());
            double maxX = 0.0, maxY = 0.0;
            bool hasPoint = false;
            for (const QVector<QPointF>& points : m_dragOriginalPoints) {
                for (const QPointF& item : points) {
                    minX = std::min(minX, item.x());
                    minY = std::min(minY, item.y());
                    maxX = std::max(maxX, item.x());
                    maxY = std::max(maxY, item.y());
                    hasPoint = true;
                }
            }
            if (hasPoint) {
                delta.setX(std::max(
                        -minX,
                        std::min(double(m_image.width()) - maxX, delta.x())));
                delta.setY(std::max(
                        -minY,
                        std::min(double(m_image.height()) - maxY, delta.y())));
                for (int group = 0; group < m_dragIndices.size(); ++group) {
                    const int selected = m_dragIndices[group];
                    if (selected < 0 || selected >= m_shapes.size()) continue;
                    m_shapes[selected].points = m_dragOriginalPoints[group];
                    for (QPointF& item : m_shapes[selected].points)
                        item += delta;
                    if (group < m_dragOriginalMasks.size())
                        m_shapes[selected].mask =
                                movedMask(m_dragOriginalMasks[group], delta);
                }
            }
        } else {
            QPointF delta = point - m_last;
            double minX = shape.points.first().x(), maxX = minX;
            double minY = shape.points.first().y(), maxY = minY;
            for (const QPointF& item : shape.points) {
                minX = std::min(minX, item.x());
                minY = std::min(minY, item.y());
                maxX = std::max(maxX, item.x());
                maxY = std::max(maxY, item.y());
            }
            delta.setX(std::max(-minX, std::min(double(m_image.width()) - maxX,
                                                delta.x())));
            delta.setY(std::max(-minY, std::min(double(m_image.height()) - maxY,
                                                delta.y())));
            for (QPointF& item : shape.points) item += delta;
            if (!m_dragOriginalMasks.isEmpty())
                shape.mask = movedMask(m_dragOriginalMasks.first(),
                                       point - m_dragStartPoint);
        }
        m_last = point;
    } else if (m_creating && m_tool != QStringLiteral("polygon") &&
               m_tool != QStringLiteral("linestrip") &&
               m_tool != QStringLiteral("quadrilateral")) {
        AnnotationShape& shape = m_shapes.last();
        if (shape.type == QStringLiteral("rectangle") ||
            shape.type == QStringLiteral("rotation") ||
            shape.type == QStringLiteral("cuboid")) {
            shape.points = rectanglePoints(m_start, point);
            if (shape.type == QStringLiteral("cuboid")) {
                QPointF offset = m_cuboidDepthVector;
                const double depth = std::hypot(offset.x(), offset.y());
                if (depth < m_cuboidMinDepth && depth > 1e-6)
                    offset *= m_cuboidMinDepth / depth;
                else if (depth <= 1e-6)
                    offset = QPointF(std::max(0.0, m_cuboidMinDepth), 0.0);
                for (int i = 0; i < 4; ++i)
                    shape.points.append(shape.points[i] + offset);
            }
        } else
            shape.points = {m_start, point};
    }
    update();
}

void AnnotationCanvas::mouseReleaseEvent(QMouseEvent* event) {
    if (m_promptBoxDragging && event->button() == Qt::LeftButton) {
        updateTransform();
        m_prompt.box = QRectF(m_promptBoxStart,
                              clamp(toImage(qtCompatMouseEventPos(event))))
                               .normalized();
        m_promptBoxDragging = false;
        if (m_prompt.box.width() < 2 || m_prompt.box.height() < 2)
            m_prompt.useBox = false;
        emit promptChanged();
        update();
        return;
    }
    if (m_brushEditing) {
        if (event->button() == Qt::LeftButton)
            finishBrushStroke();
        else if (event->button() == Qt::RightButton)
            setBrushEditing(false);
        return;
    }
    if (m_panning && (event->button() == Qt::MiddleButton ||
                      event->button() == Qt::LeftButton)) {
        m_panning = false;
        setCursor(Qt::ArrowCursor);
        return;
    }
    if (event->button() != Qt::LeftButton) return;
    if (m_rotating) {
        m_rotating = false;
        if (m_dragChanged) emit shapesChanged();
    } else if (m_dragging) {
        m_dragging = false;
        m_dragFace = annotationCuboid::Face::None;
        if (m_dragChanged) emit shapesChanged();
        m_dragIndices.clear();
        m_dragOriginalPoints.clear();
        m_dragOriginalMasks.clear();
    } else if (m_creating && m_tool != QStringLiteral("polygon") &&
               m_tool != QStringLiteral("linestrip") &&
               m_tool != QStringLiteral("quadrilateral")) {
        finishShape();
    }
}

void AnnotationCanvas::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && m_creating && m_doubleClickClose &&
        (m_tool == QStringLiteral("polygon") ||
         m_tool == QStringLiteral("linestrip")))
        finishShape();
    else if (event->button() == Qt::LeftButton && m_doubleClickEditLabel &&
             m_tool == QStringLiteral("edit")) {
        updateTransform();
        int vertex = -1;
        const int index =
                hitShape(clamp(toImage(qtCompatMouseEventPos(event))), &vertex);
        if (index >= 0) {
            selectShape(index);
            emit shapeDoubleClicked(index);
        }
    }
}

void AnnotationCanvas::finishShape() {
    if (!m_creating || m_shapes.isEmpty()) return;
    AnnotationShape& shape = m_shapes.last();
    const int minimum = shape.type == QStringLiteral("polygon") ? 3
                        : shape.type == QStringLiteral("point") ? 1
                                                                : 2;
    if (shape.points.size() < minimum ||
        (shape.points.size() > 1 &&
         QLineF(shape.points.first(), shape.points.last()).length() < 1.0)) {
        cancelShape();
        return;
    } else {
        emit shapeFinished(m_shapes.size() - 1);
    }
    m_creating = false;
    emit shapesChanged();
    update();
}

void AnnotationCanvas::wheelEvent(QWheelEvent* event) {
    if (m_brushEditing) {
        if (event->angleDelta().y() != 0)
            setBrushRadius(m_brushRadius +
                           (event->angleDelta().y() > 0 ? 1 : -1));
        event->accept();
        return;
    }
    updateTransform();
    const QPointF cursor = qtCompatWheelEventPos(event);
    const QPointF anchor = toImage(cursor);
    if (m_wheelRectangleEditing && m_tool == QStringLiteral("edit") &&
        m_selected >= 0 && m_selected < m_shapes.size() &&
        m_shapes[m_selected].type == QStringLiteral("rectangle") &&
        m_shapes[m_selected].points.size() == 4 &&
        !(event->modifiers() & Qt::ControlModifier) &&
        !m_shapes[m_selected].locked) {
        AnnotationShape& shape = m_shapes[m_selected];
        const QRectF bounds =
                QRectF(shape.points[0], shape.points[2]).normalized();
        const bool outward = event->angleDelta().y() > 0;
        const double factor = std::max(
                0.1, outward ? 1.0 + m_wheelScaleStep : 1.0 - m_wheelScaleStep);
        if (bounds.contains(anchor)) {
            const QPointF center = bounds.center();
            QVector<QPointF> next;
            next.reserve(4);
            bool valid = true;
            for (const QPointF& point : shape.points) {
                const QPointF candidate = center + (point - center) * factor;
                if (!QRectF(QPointF(), m_image.size()).contains(candidate)) {
                    valid = false;
                    break;
                }
                next.append(candidate);
            }
            if (valid) shape.points = next;
        } else {
            const double step =
                    outward ? m_wheelAdjustStep : -m_wheelAdjustStep;
            const bool horizontal =
                    anchor.x() < bounds.left() || anchor.x() > bounds.right();
            if (horizontal) {
                const bool adjustLeft = anchor.x() < bounds.center().x();
                for (QPointF& point : shape.points) {
                    if (adjustLeft &&
                        std::abs(point.x() - bounds.left()) < 1e-6)
                        point.setX(std::clamp(point.x() - step, 0.0,
                                              double(m_image.width() - 1)));
                    else if (!adjustLeft &&
                             std::abs(point.x() - bounds.right()) < 1e-6)
                        point.setX(std::clamp(point.x() + step, 0.0,
                                              double(m_image.width() - 1)));
                }
            } else {
                const bool adjustTop = anchor.y() < bounds.center().y();
                for (QPointF& point : shape.points) {
                    if (adjustTop && std::abs(point.y() - bounds.top()) < 1e-6)
                        point.setY(std::clamp(point.y() - step, 0.0,
                                              double(m_image.height() - 1)));
                    else if (!adjustTop &&
                             std::abs(point.y() - bounds.bottom()) < 1e-6)
                        point.setY(std::clamp(point.y() + step, 0.0,
                                              double(m_image.height() - 1)));
                }
            }
        }
        emit shapesChanged();
        update();
        event->accept();
        return;
    }
    zoomBy(event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15);
    updateTransform();
    m_pan += cursor - (m_origin + anchor * m_scale);
    emit viewChanged();
    update();
    event->accept();
}

void AnnotationCanvas::keyPressEvent(QKeyEvent* event) {
    if (m_brushEditing && event->key() == Qt::Key_R &&
        event->modifiers() == Qt::NoModifier) {
        setBrushOperation(m_brushOperation == 1 ? 0 : 1);
        event->accept();
        return;
    }
    if (m_brushEditing && event->key() == Qt::Key_Escape) {
        setBrushEditing(false, true);
        event->accept();
        return;
    }
    if (m_brushEditing && (event->modifiers() & Qt::ControlModifier) &&
        (event->key() == Qt::Key_Z || event->key() == Qt::Key_Y)) {
        if (event->key() == Qt::Key_Y ||
            (event->modifiers() & Qt::ShiftModifier))
            brushRedo();
        else
            brushUndo();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        m_spaceHeld = true;
        setCursor(Qt::OpenHandCursor);
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Escape) {
        if (m_creating)
            cancelShape();
        else if (m_magicActive)
            clearMagicWand();
        else if (m_tool.startsWith(QStringLiteral("sam_")))
            clearPrompt();
        else
            setTool(QStringLiteral("edit"));
        event->accept();
        return;
    }
    if (m_tool == QStringLiteral("edit") && !m_selectedIndices.isEmpty() &&
        event->modifiers() == Qt::NoModifier) {
        double degrees = 0.0;
        if (event->key() == Qt::Key_Z)
            degrees = m_rotationLargeIncrement;
        else if (event->key() == Qt::Key_X)
            degrees = m_rotationSmallIncrement;
        else if (event->key() == Qt::Key_C)
            degrees = -m_rotationSmallIncrement;
        else if (event->key() == Qt::Key_V)
            degrees = -m_rotationLargeIncrement;
        if (degrees != 0.0) {
            rotateSelectedBy(degrees);
            event->accept();
            return;
        }
    }
    if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) &&
        m_creating) {
        finishShape();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Delete && m_tool == QStringLiteral("edit")) {
        deleteSelected();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
}

void AnnotationCanvas::rotateSelectedBy(double degrees) {
    const double radians = degrees * 3.14159265358979323846 / 180.0;
    bool changed = false;
    for (int index : m_selectedIndices) {
        if (index < 0 || index >= m_shapes.size() || m_shapes[index].locked ||
            m_shapes[index].type != QStringLiteral("rotation") ||
            m_shapes[index].points.size() != 4)
            continue;
        const QPointF center =
                (m_shapes[index].points[0] + m_shapes[index].points[2]) / 2.0;
        for (QPointF& point : m_shapes[index].points)
            point = rotated(point, center, radians);
        changed = true;
    }
    if (changed) {
        emit editStarted();
        emit shapesChanged();
        update();
    }
}

void AnnotationCanvas::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        m_spaceHeld = false;
        if (!m_panning)
            setCursor(m_tool == QStringLiteral("edit") ? Qt::ArrowCursor
                                                       : Qt::CrossCursor);
        event->accept();
        return;
    }
    QWidget::keyReleaseEvent(event);
}
