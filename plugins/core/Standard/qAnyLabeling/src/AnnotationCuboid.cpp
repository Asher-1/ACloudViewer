// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationCuboid.h"

#include <QLineF>
#include <algorithm>
#include <cmath>
#include <limits>

namespace annotationCuboid {
namespace {
double length(const QPointF& value) { return std::hypot(value.x(), value.y()); }

bool rearOnRight(const QVector<QPointF>& points) {
    return points[4].x() >= points[0].x();
}

void setFront(QVector<QPointF>& points, const QVector<QPointF>& front) {
    for (int i = 0; i < 4; ++i) {
        const QPointF offset = points[i + 4] - points[i];
        points[i] = front[i];
        points[i + 4] = front[i] + offset;
    }
}
}  // namespace

QVector<int> visibleControls(const QVector<QPointF>& points) {
    if (points.size() != 8) return {};
    return {0,
            1,
            2,
            3,
            rearOnRight(points) ? 5 : 4,
            rearOnRight(points) ? 6 : 7,
            FrontLeftCenter,
            FrontRightCenter,
            FrontTopCenter,
            FrontBottomCenter,
            rearOnRight(points) ? BackRightCenter : BackLeftCenter};
}

QPointF controlPoint(const QVector<QPointF>& points, int control) {
    if (points.size() != 8 || control < 0 || control > BackRightCenter)
        return {};
    if (control < 8) return points[control];
    static constexpr int edges[][2] = {{0, 3}, {1, 2}, {0, 1},
                                       {3, 2}, {4, 7}, {5, 6}};
    const auto& edge = edges[control - FrontLeftCenter];
    return (points[edge[0]] + points[edge[1]]) / 2.0;
}

int nearestControl(const QVector<QPointF>& points,
                   const QPointF& position,
                   double radius) {
    int nearest = -1;
    double best = std::numeric_limits<double>::max();
    for (int control : visibleControls(points)) {
        const double distance =
                QLineF(position, controlPoint(points, control)).length();
        if (distance <= radius && distance < best) {
            best = distance;
            nearest = control;
        }
    }
    return nearest;
}

QPainterPath facePath(const QVector<QPointF>& points, Face face) {
    QPainterPath path;
    if (points.size() != 8 || face == Face::None) return path;
    QVector<int> indices;
    switch (face) {
        case Face::Front:
            indices = {0, 1, 2, 3};
            break;
        case Face::Left:
            indices = {0, 4, 7, 3};
            break;
        case Face::Right:
            indices = {1, 2, 6, 5};
            break;
        case Face::Back:
            indices = {4, 5, 6, 7};
            break;
        default:
            return path;
    }
    path.moveTo(points[indices.first()]);
    for (int i = 1; i < indices.size(); ++i) path.lineTo(points[indices[i]]);
    path.closeSubpath();
    return path;
}

Face hitFace(const QVector<QPointF>& points, const QPointF& position) {
    if (points.size() != 8) return Face::None;
    // The front always moves the whole object, including in overlapping faces.
    const Face nearSide = rearOnRight(points) ? Face::Right : Face::Left;
    const Face farSide = rearOnRight(points) ? Face::Left : Face::Right;
    for (Face face : {Face::Front, nearSide, farSide, Face::Back})
        if (facePath(points, face).contains(position)) return face;
    return Face::None;
}

bool moveControl(QVector<QPointF>& points,
                 int control,
                 const QPointF& position,
                 double minDepth) {
    if (points.size() != 8 || !visibleControls(points).contains(control))
        return false;
    const QVector<QPointF> original = points;
    const double margin = std::max(2.0, minDepth * 0.2);
    if (control < 4) {
        // Solve in the front face's own basis so imported rotated/skewed faces
        // retain their axes and the diagonally opposite corner remains fixed.
        const int next = (control + 1) % 4, opposite = (control + 2) % 4;
        const int previous = (control + 3) % 4;
        QPointF u = points[next] - points[control];
        QPointF v = points[previous] - points[control];
        const double lenU = length(u), lenV = length(v);
        if (lenU <= 1e-6 || lenV <= 1e-6) return false;
        u /= lenU;
        v /= lenV;
        const double det = u.x() * v.y() - u.y() * v.x();
        if (std::abs(det) <= 1e-6) return false;
        const QPointF diagonal = points[opposite] - position;
        const double sizeU = std::max(
                margin, (diagonal.x() * v.y() - diagonal.y() * v.x()) / det);
        const double sizeV = std::max(
                margin, (u.x() * diagonal.y() - u.y() * diagonal.x()) / det);
        QVector<QPointF> front = points.mid(0, 4);
        front[control] = points[opposite] - u * sizeU - v * sizeV;
        front[next] = points[opposite] - v * sizeV;
        front[previous] = points[opposite] - u * sizeU;
        setFront(points, front);
    } else if (control < 8) {
        // A visible rear endpoint changes the height of both faces.
        static constexpr int top[] = {0, 1, 4, 5};
        // The source constrains top/bottom sets in this order, including
        // non-axis-aligned faces imported from existing annotations.
        static constexpr int bottom[] = {2, 3, 6, 7};
        const bool isTop = control == 4 || control == 5;
        double dy = position.y() - points[control].y();
        for (int i = 0; i < 4; ++i) {
            if (isTop)
                dy = std::min(dy, points[bottom[i]].y() - margin -
                                          points[top[i]].y());
            else
                dy = std::max(dy, points[top[i]].y() + margin -
                                          points[bottom[i]].y());
        }
        for (int i = 0; i < 4; ++i)
            points[isTop ? top[i] : bottom[i]].ry() += dy;
    } else if (control <= FrontBottomCenter) {
        static constexpr int edges[][3] = {
                {0, 3, 1}, {1, 2, 0}, {0, 1, 3}, {3, 2, 0}};
        const auto& edge = edges[control - FrontLeftCenter];
        const QPointF along = points[edge[1]] - points[edge[0]];
        const double edgeLength = length(along);
        if (edgeLength <= 1e-6) return false;
        QPointF normal(-along.y() / edgeLength, along.x() / edgeLength);
        double width =
                QPointF::dotProduct(points[edge[2]] - points[edge[0]], normal);
        if (width < 0.0) {
            width = -width;
            normal = -normal;
        }
        const double shift = std::min(
                width - margin,
                QPointF::dotProduct(position - controlPoint(points, control),
                                    normal));
        QVector<QPointF> front = points.mid(0, 4);
        front[edge[0]] += normal * shift;
        front[edge[1]] += normal * shift;
        setFront(points, front);
    } else {
        // Only the outward rear center is exposed. Its x movement follows
        // the current front-to-back ray; vertical mouse movement is ignored.
        const bool right = control == BackRightCenter;
        const QPointF center = controlPoint(points, control);
        const QPointF front = controlPoint(
                points, right ? FrontRightCenter : FrontLeftCenter);
        const double limit =
                right ? std::max(points[1].x(), points[2].x()) + margin
                      : std::min(points[0].x(), points[3].x()) - margin;
        const double target = right ? std::max(position.x(), limit)
                                    : std::min(position.x(), limit);
        const double dx = target - center.x();
        const double dy = std::abs(center.x() - front.x()) > 1e-6
                                  ? dx * (center.y() - front.y()) /
                                            (center.x() - front.x())
                                  : 0.0;
        for (int i = 4; i < 8; ++i) points[i] += QPointF(dx, dy);
    }
    return points != original;
}

bool moveFace(QVector<QPointF>& points,
              Face face,
              const QPointF& offset,
              double minDepth) {
    if (points.size() != 8 ||
        (face != Face::Left && face != Face::Right && face != Face::Back))
        return false;
    const QVector<QPointF> original = points;
    if (face == Face::Back) {
        for (int i = 4; i < 8; ++i) points[i] += offset;
    } else {
        const double margin = std::max(2.0, minDepth * 0.2);
        const bool left = face == Face::Left;
        const int frontTop = left ? 1 : 0, frontBottom = left ? 2 : 3;
        const QPointF frontCenter =
                (points[frontTop] + points[frontBottom]) / 2.0;
        const QPointF backCenter =
                (points[frontTop + 4] + points[frontBottom + 4]) / 2.0;
        const bool useBack = left ? backCenter.x() > frontCenter.x()
                                  : backCenter.x() < frontCenter.x();
        const int top = frontTop + (useBack ? 4 : 0),
                  bottom = frontBottom + (useBack ? 4 : 0);
        const double limit =
                left ? std::min(points[top].x(), points[bottom].x()) - margin
                     : std::max(points[top].x(), points[bottom].x()) + margin;
        const QVector<int> indices =
                left ? QVector<int>{0, 3, 4, 7} : QVector<int>{1, 2, 5, 6};
        double extreme = points[indices.first()].x();
        for (int i : indices)
            extreme = left ? std::max(extreme, points[i].x())
                           : std::min(extreme, points[i].x());
        double dx = offset.x();
        if (left && dx > 0.0) dx = std::min(dx, limit - extreme);
        if (!left && dx < 0.0) dx = std::max(dx, limit - extreme);
        for (int i : indices) points[i] += QPointF(dx, offset.y());
    }
    return points != original;
}
}  // namespace annotationCuboid
