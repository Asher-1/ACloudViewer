// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMatrix4x4>
#include <QPointF>
#include <QSize>
#include <QVector4D>
#include <QVector>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "PointCloudCuboid.h"

// Z-up camera state used by both rendering and screen-space selection.
// Its fit, basis and matrix follow the upstream PointCloudViewport contract.
struct PointCloudViewTransform {
    std::array<double, 3> center{{0, 0, 0}};
    double scale = 10;
    double extent = 10;
    double yaw = -45;
    double pitch = 35;
    double minimumScale = 1e-5;
    double maximumScale = 1e12;
    bool perspective = false;

    void setScale(double value) {
        scale = std::clamp(value, minimumScale, maximumScale);
    }

    void setView(int view) {
        yaw = view == 2 ? 90 : 0;
        pitch = view == 0 ? 90 : 0;
    }

    // O(n), constant extra memory. Bounds suffice; no sorted/copied cloud.
    void fit(const QVector<QVector4D>& points, const QSize& size) {
        if (points.isEmpty()) return;
        std::array<double, 3> low{
                {points[0].x(), points[0].y(), points[0].z()}};
        auto high = low;
        for (const auto& point : points)
            for (int axis = 0; axis < 3; ++axis) {
                low[axis] = std::min(low[axis], double(point[axis]));
                high[axis] = std::max(high[axis], double(point[axis]));
            }
        double squared = 0;
        double maximum = 0;
        for (int axis = 0; axis < 3; ++axis) {
            center[axis] = (low[axis] + high[axis]) / 2;
            squared += std::pow(high[axis] - low[axis], 2);
            maximum = std::max(maximum, (high[axis] - low[axis]) / 2);
        }
        const double radius = std::max(std::sqrt(squared) / 2, 0.1);
        const double aspect =
                double(std::max(1, size.width())) / std::max(1, size.height());
        double fitted = radius * 1.1 / std::min(1.0, aspect);
        if (perspective) fitted *= std::sqrt(1 + std::pow(tangent(), 2));
        setScale(fitted);
        extent = std::max(maximum, radius) * 4;
    }

    // Rows are right, up and forward in world coordinates.
    std::array<std::array<double, 3>, 3> basis() const {
        const double y = yaw * radians();
        const double p = pitch * radians();
        return {{{{std::cos(y), std::sin(y), 0}},
                 {{-std::sin(y) * std::sin(p), std::cos(y) * std::sin(p),
                   std::cos(p)}},
                 {{-std::sin(y) * std::cos(p), std::cos(y) * std::cos(p),
                   -std::sin(p)}}}};
    }

    QMatrix4x4 matrix(const QSize& size) const {
        const auto axes = basis();
        const double aspect =
                double(std::max(1, size.width())) / std::max(1, size.height());
        QMatrix4x4 result;
        const double distance = scale / tangent();
        const double near = 0.1;
        const double far = std::max(500.0, distance + extent);
        const double depth = (far + near) / (far - near);
        for (int axis = 0; axis < 3; ++axis) {
            result(0, axis) =
                    float(axes[0][axis] /
                          ((perspective ? tangent() : scale) * aspect));
            result(1, axis) =
                    float(axes[1][axis] / (perspective ? tangent() : scale));
            result(2, axis) =
                    float(axes[2][axis] *
                          (perspective ? depth : 1 / std::max(extent, 0.1)));
            result(3, axis) = perspective ? float(axes[2][axis]) : 0;
        }
        // The source first quantizes matrix rows to float32, then translates
        // using its float64 center. Keep this order for large coordinates.
        for (int row = 0; row < 3; ++row) {
            double offset = 0;
            for (int axis = 0; axis < 3; ++axis)
                offset -= double(result(row, axis)) * center[axis];
            result(row, 3) = float(offset);
        }
        if (perspective) {
            result(2, 3) = float(double(result(2, 3)) +
                                 (distance * (far + near) - 2 * far * near) /
                                         (far - near));
            double offset = distance;
            for (int axis = 0; axis < 3; ++axis)
                offset -= axes[2][axis] * center[axis];
            result(3, 3) = float(offset);
        }
        return result;
    }

    void pan(double dx, double dy, int height) {
        const auto axes = basis();
        for (int axis = 0; axis < 3; ++axis)
            center[axis] += (axes[1][axis] * dy - axes[0][axis] * dx) * 2 *
                            scale / std::max(1, height);
    }
    void orbit(double dx, double dy) {
        yaw -= dx * 0.4;
        pitch = std::clamp(pitch + dy * 0.4, -90.0, 90.0);
    }
    void zoom(double delta) { setScale(scale * std::exp(-delta / 900)); }

    // Final state of CuboidViewport.move_camera; no frame-sized allocation.
    bool moveCamera(const QString& key) {
        const auto axes = basis();
        if (key == "U" || key == "O" || key == "J" || key == "L") {
            const int axis = key == "U" || key == "O" ? 1 : 0;
            const double direction = key == "U" || key == "J" ? -2 : 2;
            for (int i = 0; i < 3; ++i) center[i] += axes[axis][i] * direction;
        } else if (key == "I" || key == "K") {
            setScale(scale + (key == "I" ? -5 : 5) * tangent());
        } else if (key == "Left" || key == "Right") {
            yaw += key == "Left" ? -20 : 20;
        } else if (key == "Up" || key == "Down") {
            const double limit = 90 - 1e-6 / radians();
            pitch = std::clamp(pitch + (key == "Up" ? 10 : -10), -limit, limit);
        } else {
            return false;
        }
        return true;
    }

    PointCloudCuboid nudgedCuboid(const PointCloudCuboid& box,
                                  int x,
                                  int y) const {
        auto result = box;
        const auto axes = basis();
        for (int i = 0; i < 3; ++i)
            result.center[i] += (axes[0][i] * x + axes[1][i] * y) * 0.1;
        return result;
    }

    CuboidVector unproject(const QPointF& point, const QSize& size) const {
        const auto axes = basis();
        auto result = center;
        const double factor = 2 * scale / std::max(1, size.height());
        for (int axis = 0; axis < 3; ++axis)
            result[axis] += factor *
                            ((point.x() - size.width() / 2.0) * axes[0][axis] -
                             (point.y() - size.height() / 2.0) * axes[1][axis]);
        return result;
    }

    // O(n), no point copy: the upstream main-view preview chooses the
    // nearest positive ray depth within a one-metre-radius cylinder.
    bool creationPreview(const QPointF& point,
                         const QSize& size,
                         const QVector<QVector4D>& points,
                         const QVector<int>& visible,
                         PointCloudCuboid* box) const {
        if (!box || point.x() < 0 || point.y() < 0 ||
            point.x() > size.width() || point.y() > size.height())
            return false;
        const auto forward = basis()[2];
        auto origin = center;
        const double distance = scale / tangent();
        for (int axis = 0; axis < 3; ++axis)
            origin[axis] -= forward[axis] * distance;
        auto direction = unproject(point, size);
        double length = 0;
        for (int axis = 0; axis < 3; ++axis) {
            direction[axis] -= origin[axis];
            length += direction[axis] * direction[axis];
        }
        length = std::sqrt(length);
        if (length == 0) return false;
        for (auto& value : direction) value /= length;
        double nearest = std::numeric_limits<double>::infinity();
        for (int index : visible) {
            if (index < 0 || index >= points.size()) continue;
            double depth = 0, squared = 0;
            for (int axis = 0; axis < 3; ++axis) {
                const double offset =
                        double(points[index][axis]) - origin[axis];
                depth += offset * direction[axis];
                squared += offset * offset;
            }
            if (depth >= .1 && squared - depth * depth < 1)
                nearest = std::min(nearest, depth);
        }
        if (!std::isfinite(nearest)) return false;
        *box = PointCloudCuboid();
        for (int axis = 0; axis < 3; ++axis)
            box->center[axis] = origin[axis] + direction[axis] * nearest;
        box->size = {{1, 1, 1}};
        return true;
    }

    void focus(const PointCloudCuboid& box, const QSize& size) {
        center = box.center;
        const auto axes = basis();
        const auto corners = box.corners();
        std::array<double, 2> low{{std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()}};
        std::array<double, 2> high{{-low[0], -low[1]}};
        for (const auto& corner : corners)
            for (int row = 0; row < 2; ++row) {
                double value = 0;
                for (int axis = 0; axis < 3; ++axis)
                    value += (corner[axis] - center[axis]) * axes[row][axis];
                low[row] = std::min(low[row], value);
                high[row] = std::max(high[row], value);
            }
        const double aspect =
                double(std::max(1, size.width())) / std::max(1, size.height());
        setScale(std::max(high[1] - low[1], (high[0] - low[0]) / aspect) *
                 0.75);
    }

    // O(boxes): intersect the viewing ray with each oriented box. A projected
    // bounding rectangle admits clicks outside rotated boxes and loses depth.
    int hit(const QPointF& point,
            const QSize& size,
            const QVector<PointCloudCuboid>& boxes) const {
        const auto forward = basis()[2];
        auto origin = center;
        auto direction = forward;
        const double distance = perspective ? scale / tangent() : extent;
        const auto target = unproject(point, size);
        if (!perspective) origin = target;
        for (int axis = 0; axis < 3; ++axis) {
            origin[axis] -= forward[axis] * distance;
            if (perspective)
                direction[axis] = (target[axis] - origin[axis]) / distance;
        }
        int selected = 0;
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto& box : boxes) {
            auto offset = origin;
            for (int axis = 0; axis < 3; ++axis)
                offset[axis] -= box.center[axis];
            const auto rotation = cuboidRotationMatrix(box.rotation);
            const auto local = cuboidTransform(rotation, offset, true);
            const auto ray = cuboidTransform(rotation, direction, true);
            double entry = perspective ? 0.1 : 0;
            double exit = std::numeric_limits<double>::infinity();
            for (int axis = 0; axis < 3; ++axis) {
                const double half = box.size[axis] / 2;
                if (std::abs(ray[axis]) < 1e-10) {
                    if (std::abs(local[axis]) > half) exit = -1;
                    continue;
                }
                const double first = (-half - local[axis]) / ray[axis];
                const double last = (half - local[axis]) / ray[axis];
                entry = std::max(entry, std::min(first, last));
                exit = std::min(exit, std::max(first, last));
            }
            if (entry <= exit &&
                (entry < nearest || (entry == nearest && box.id < selected))) {
                nearest = entry;
                selected = box.id;
            }
        }
        return selected;
    }

private:
    static double radians() { return std::acos(-1.0) / 180; }
    static double tangent() { return std::tan(25 * radians()); }
};
