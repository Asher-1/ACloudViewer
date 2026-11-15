// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MaskPolygon.h"

#include <QHash>
#include <algorithm>
#include <cmath>

namespace {
quint64 vertex(int x, int y) {
    return (quint64(quint32(y)) << 32) | quint32(x);
}

QPointF point(quint64 value) {
    return QPointF(quint32(value), quint32(value >> 32));
}

int direction(quint64 from, quint64 to) {
    const QPointF a = point(from), b = point(to);
    if (b.x() > a.x()) return 0;
    if (b.y() > a.y()) return 1;
    if (b.x() < a.x()) return 2;
    return 3;
}

double signedArea(const QVector<QPointF>& polygon) {
    double twiceArea = 0;
    for (int i = 0; i < polygon.size(); ++i) {
        const QPointF& a = polygon[i];
        const QPointF& b = polygon[(i + 1) % polygon.size()];
        twiceArea += a.x() * b.y() - b.x() * a.y();
    }
    return twiceArea / 2.0;
}
}  // namespace

QVector<QVector<QPointF>> maskPolygons(const uint8_t* data,
                                       int width,
                                       int height,
                                       size_t rowStride) {
    if (!data || width <= 0 || height <= 0 || rowStride < size_t(width))
        return {};
    const auto filled = [=](int x, int y) {
        return x >= 0 && y >= 0 && x < width && y < height &&
               data[size_t(y) * rowStride + size_t(x)] != 0;
    };
    QHash<quint64, QVector<quint64>> edges;
    int edgeCount = 0;
    const auto addEdge = [&edges, &edgeCount](int x1, int y1, int x2, int y2) {
        edges[vertex(x1, y1)].append(vertex(x2, y2));
        ++edgeCount;
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!filled(x, y)) continue;
            if (!filled(x, y - 1)) addEdge(x, y, x + 1, y);
            if (!filled(x + 1, y)) addEdge(x + 1, y, x + 1, y + 1);
            if (!filled(x, y + 1)) addEdge(x + 1, y + 1, x, y + 1);
            if (!filled(x - 1, y)) addEdge(x, y + 1, x, y);
        }
    }

    struct Contour {
        double area;
        QVector<QPointF> points;
    };
    QVector<Contour> contours;
    while (!edges.isEmpty()) {
        const quint64 start = edges.constBegin().key();
        quint64 current = start;
        int incoming = -1;
        QVector<QPointF> loop;
        const int maxSteps = edgeCount;
        bool closed = false;
        for (int step = 0; step < maxSteps; ++step) {
            auto it = edges.find(current);
            if (it == edges.end()) break;
            loop.append(point(current));
            int chosen = 0;
            if (incoming >= 0 && it.value().size() > 1) {
                // A right turn keeps diagonal contacts on separate external
                // loops.
                int bestPriority = 4;
                for (int i = 0; i < it.value().size(); ++i) {
                    const int turn =
                            (direction(current, it.value()[i]) - incoming + 4) %
                            4;
                    const int priority = turn == 1   ? 0
                                         : turn == 0 ? 1
                                         : turn == 3 ? 2
                                                     : 3;
                    if (priority < bestPriority) {
                        bestPriority = priority;
                        chosen = i;
                    }
                }
            }
            const quint64 next = it.value().takeAt(chosen);
            if (it.value().isEmpty()) edges.erase(it);
            --edgeCount;
            incoming = direction(current, next);
            current = next;
            if (current == start) {
                closed = true;
                break;
            }
        }
        if (!closed || loop.size() < 3) continue;
        QVector<QPointF> corners;
        corners.reserve(loop.size());
        for (int i = 0; i < loop.size(); ++i) {
            const QPointF& prev = loop[(i + loop.size() - 1) % loop.size()];
            const QPointF& now = loop[i];
            const QPointF& next = loop[(i + 1) % loop.size()];
            const QPointF a = now - prev, b = next - now;
            if (std::abs(a.x() * b.y() - a.y() * b.x()) > 1e-8) {
                QPointF pixel = now;
                if (a.y() > 0 || b.y() > 0) pixel.rx() -= 1;
                if (a.x() < 0 || b.x() < 0) pixel.ry() -= 1;
                corners.append(pixel);
            }
        }
        const double area = signedArea(corners);
        if (area > 0 && corners.size() >= 3) contours.append({area, corners});
    }
    std::sort(
            contours.begin(), contours.end(),
            [](const Contour& a, const Contour& b) { return a.area > b.area; });
    QVector<QVector<QPointF>> polygons;
    for (const Contour& contour : contours) polygons.append(contour.points);
    return polygons;
}

QVector<QPointF> largestMaskPolygon(const uint8_t* data,
                                    int width,
                                    int height,
                                    size_t rowStride) {
    const QVector<QVector<QPointF>> polygons =
            maskPolygons(data, width, height, rowStride);
    return polygons.isEmpty() ? QVector<QPointF>() : polygons.first();
}
