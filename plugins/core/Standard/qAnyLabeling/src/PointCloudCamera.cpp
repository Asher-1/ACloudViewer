// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCamera.h"

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QPainter>
#include <QPair>
#include <cmath>

namespace {

// Upstream CORNER_SIGNS order: bit 2 = z sign, bit 1 = y sign, bit 0 =
// x sign (corners 0-3 are the z=-1 face, 4-7 the z=+1 face).
constexpr int kCornerCount = 8;

double matrixAt(const QJsonArray& rows, int row, int column, bool* ok) {
    // Row length is validated by the caller (camera_matrix rows are 3-wide,
    // transform rows 4-wide); out-of-range reads yield Undefined and fail
    // the isDouble check below.
    const QJsonArray values = rows.at(row).toArray();
    const QJsonValue value = values.at(column);
    if (!value.isDouble() || !qIsFinite(value.toDouble())) {
        *ok = false;
        return 0.0;
    }
    return value.toDouble();
}

// Distort one normalized pinhole coordinate (upstream _project_camera_points
// inner step). Returns false when the coordinate is non-finite.
bool distortPixel(const PointCloudCalibration& calibration,
                  double x,
                  double y,
                  double* outX,
                  double* outY) {
    if (calibration.opencv5) {
        const double k1 = calibration.distortion[0];
        const double k2 = calibration.distortion[1];
        const double p1 = calibration.distortion[2];
        const double p2 = calibration.distortion[3];
        const double k3 = calibration.distortion[4];
        const double radius2 = x * x + y * y;
        const double radial =
                1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3));
        const double distortedX =
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x);
        const double distortedY =
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y;
        x = distortedX;
        y = distortedY;
    }
    *outX = x * calibration.fx + calibration.cx;
    *outY = y * calibration.fy + calibration.cy;
    return qIsFinite(*outX) && qIsFinite(*outY);
}

// Project camera-space points to pixels; returns false for non-finite or
// non-positive depth entries (their output stays (0,0)).
void projectCameraPoints(const QVector<QVector3D>& camera,
                         const PointCloudCalibration& calibration,
                         QVector<QVector2D>* pixels,
                         QVector<bool>* inFront) {
    pixels->resize(camera.size());
    inFront->resize(camera.size());
    for (int i = 0; i < camera.size(); ++i) {
        const QVector3D point = camera.at(i);
        const bool valid = qIsFinite(point.x()) && qIsFinite(point.y()) &&
                           qIsFinite(point.z()) && point.z() > 0.0;
        (*inFront)[i] = valid;
        if (!valid) {
            (*pixels)[i] = QVector2D(0.0f, 0.0f);
            continue;
        }
        double x = 0.0;
        double y = 0.0;
        distortPixel(calibration, double(point.x()) / point.z(),
                     double(point.y()) / point.z(), &x, &y);
        (*pixels)[i] = QVector2D(float(x), float(y));
    }
}

// Liang-Barsky-style clipping of one segment against the image rect
// (upstream _clip_image_segments per-segment logic). Returns false when
// the segment lies fully outside.
bool clipSegment(const QVector2D& start,
                 const QVector2D& end,
                 int width,
                 int height,
                 QVector2D* clippedStart,
                 QVector2D* clippedEnd) {
    const double dx = double(end.x()) - start.x();
    const double dy = double(end.y()) - start.y();
    double lower = 0.0;
    double upper = 1.0;
    const double direction[4] = {-dx, dx, -dy, dy};
    const double distance[4] = {start.x(), width - 1.0 - start.x(), start.y(),
                                height - 1.0 - start.y()};
    for (int axis = 0; axis < 4; ++axis) {
        if (direction[axis] == 0.0) {
            if (distance[axis] < 0) return false;
            continue;
        }
        const double ratio = distance[axis] / direction[axis];
        if (direction[axis] < 0)
            lower = std::max(lower, ratio);
        else
            upper = std::min(upper, ratio);
    }
    if (lower > upper) return false;
    *clippedStart = QVector2D(float(start.x() + lower * dx),
                              float(start.y() + lower * dy));
    *clippedEnd = QVector2D(float(start.x() + upper * dx),
                            float(start.y() + upper * dy));
    return true;
}

}  // namespace

QVector<QPair<int, int>> pointCloudCuboidEdges() {
    return {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
}

bool loadPointCloudCalibration(const QString& path,
                               PointCloudCalibration* calibration,
                               QString* error) {
    if (!calibration || !error) return false;
    error->clear();
    if (QFileInfo(path).suffix().compare(QStringLiteral("json"),
                                         Qt::CaseInsensitive) != 0) {
        *error = QStringLiteral("Calibration must be a JSON file.");
        return false;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = file.errorString();
        return false;
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (document.isNull() || !document.isObject()) {
        *error = QStringLiteral("Calibration must be a JSON object.");
        return false;
    }
    const QJsonObject data = document.object();
    static const char* kRequired[] = {
            "schema_version",         "image_size",
            "camera_model",           "camera_matrix",
            "T_pointcloud_to_camera", "distortion_model",
            "distortion_coefficients"};
    for (const char* field : kRequired) {
        if (!data.contains(QLatin1String(field))) {
            *error = QStringLiteral("Missing calibration field: %1").arg(field);
            return false;
        }
    }
    if (data.size() != 7) {
        *error = QStringLiteral("Unknown calibration fields present.");
        return false;
    }
    if (!data.value(QStringLiteral("schema_version")).isDouble() ||
        data.value(QStringLiteral("schema_version")).toInt() != 1) {
        *error = QStringLiteral("schema_version must be 1.");
        return false;
    }
    if (data.value(QStringLiteral("camera_model")).toString() !=
        QStringLiteral("pinhole")) {
        *error = QStringLiteral("camera_model must be pinhole.");
        return false;
    }
    const QJsonArray size = data.value(QStringLiteral("image_size")).toArray();
    if (size.size() != 2 || !size.at(0).isDouble() || !size.at(1).isDouble() ||
        size.at(0).toInt() <= 0 || size.at(1).toInt() <= 0 ||
        size.at(0).toDouble() != size.at(0).toInt() ||
        size.at(1).toDouble() != size.at(1).toInt()) {
        *error = QStringLiteral(
                "image_size must contain positive integer width and "
                "height.");
        return false;
    }
    const QString model =
            data.value(QStringLiteral("distortion_model")).toString();
    if (model != QLatin1String("none") && model != QLatin1String("opencv5")) {
        *error = QStringLiteral("distortion_model must be none or opencv5.");
        return false;
    }
    const QJsonArray cameraMatrix =
            data.value(QStringLiteral("camera_matrix")).toArray();
    const QJsonArray transform =
            data.value(QStringLiteral("T_pointcloud_to_camera")).toArray();
    const QJsonArray distortion =
            data.value(QStringLiteral("distortion_coefficients")).toArray();
    if (cameraMatrix.size() != 3 || transform.size() != 4) {
        *error = QStringLiteral(
                "camera_matrix must be 3x3 and "
                "T_pointcloud_to_camera must be 4x4.");
        return false;
    }
    const int expectedDistortion = model == QLatin1String("opencv5") ? 5 : 0;
    if (distortion.size() != expectedDistortion) {
        *error = QStringLiteral("distortion_coefficients must hold %1 values.")
                         .arg(expectedDistortion);
        return false;
    }
    for (const QJsonValue& value : distortion) {
        if (!value.isDouble() || !qIsFinite(value.toDouble())) {
            *error = QStringLiteral(
                    "distortion_coefficients must be finite numbers.");
            return false;
        }
    }

    PointCloudCalibration parsed;
    bool ok = true;
    // Intrinsic: strict [[fx,0,cx],[0,fy,cy],[0,0,1]] shape with positive
    // focal lengths (upstream rejects any deviation).
    const double m[3][3] = {{matrixAt(cameraMatrix, 0, 0, &ok),
                             matrixAt(cameraMatrix, 0, 1, &ok),
                             matrixAt(cameraMatrix, 0, 2, &ok)},
                            {matrixAt(cameraMatrix, 1, 0, &ok),
                             matrixAt(cameraMatrix, 1, 1, &ok),
                             matrixAt(cameraMatrix, 1, 2, &ok)},
                            {matrixAt(cameraMatrix, 2, 0, &ok),
                             matrixAt(cameraMatrix, 2, 1, &ok),
                             matrixAt(cameraMatrix, 2, 2, &ok)}};
    if (!ok) {
        *error = QStringLiteral("camera_matrix must contain finite numbers.");
        return false;
    }
    if (m[0][0] <= 0 || m[1][1] <= 0 || m[0][1] != 0 || m[1][0] != 0 ||
        m[2][0] != 0 || m[2][1] != 0 || m[2][2] != 1) {
        *error = QStringLiteral(
                "camera_matrix must be [[fx,0,cx],[0,fy,cy],[0,0,1]] "
                "with positive focal lengths.");
        return false;
    }
    parsed.fx = m[0][0];
    parsed.fy = m[1][1];
    parsed.cx = m[0][2];
    parsed.cy = m[1][2];

    // Rigid transform: 4x4, rotation orthonormal with det=+1, last row
    // exactly [0,0,0,1] (upstream tolerance: atol 1e-5).
    double rotation[9] = {};
    double translation[3] = {};
    const QJsonArray lastRow = transform.at(3).toArray();
    if (lastRow.size() != 4 || lastRow.at(0).toDouble() != 0.0 ||
        lastRow.at(1).toDouble() != 0.0 || lastRow.at(2).toDouble() != 0.0 ||
        lastRow.at(3).toDouble() != 1.0) {
        *error = QStringLiteral(
                "T_pointcloud_to_camera must end with [0, 0, 0, 1].");
        return false;
    }
    for (int row = 0; row < 3; ++row) {
        const QJsonArray transformRow = transform.at(row).toArray();
        if (transformRow.size() != 4) {
            *error = QStringLiteral(
                    "T_pointcloud_to_camera rows must hold 4 values.");
            return false;
        }
        for (int column = 0; column < 3; ++column) {
            const double value = matrixAt(transform, row, column, &ok);
            if (!ok) {
                *error = QStringLiteral(
                        "T_pointcloud_to_camera must contain finite "
                        "numbers.");
                return false;
            }
            rotation[row * 3 + column] = value;
        }
        translation[row] = transformRow.at(3).toDouble();
        if (!qIsFinite(translation[row])) {
            *error = QStringLiteral(
                    "T_pointcloud_to_camera must contain finite "
                    "numbers.");
            return false;
        }
    }
    double determinant = 0.0;
    for (int column = 0; column < 3; ++column) {
        const double a = rotation[column];
        const double b = rotation[3 + column];
        const double c = rotation[6 + column];
        determinant += a * (rotation[3 + (column + 1) % 3] *
                                    rotation[6 + (column + 2) % 3] -
                            rotation[3 + (column + 2) % 3] *
                                    rotation[6 + (column + 1) % 3]);
        // Orthonormality: every column must be a unit vector.
        if (std::abs(std::sqrt(a * a + b * b + c * c) - 1.0) > 1e-5) {
            *error = QStringLiteral(
                    "T_pointcloud_to_camera must be a rigid "
                    "transform.");
            return false;
        }
    }
    if (std::abs(determinant - 1.0) > 1e-5) {
        *error = QStringLiteral(
                "T_pointcloud_to_camera must be a rigid transform "
                "with a proper rotation.");
        return false;
    }
    for (int i = 0; i < 9; ++i) parsed.rotation[i] = rotation[i];
    for (int i = 0; i < 3; ++i) parsed.translation[i] = translation[i];

    parsed.imageSize = QSize(size.at(0).toInt(), size.at(1).toInt());
    parsed.opencv5 = model == QLatin1String("opencv5");
    for (int i = 0; i < expectedDistortion; ++i)
        parsed.distortion[i] = distortion.at(i).toDouble();

    *calibration = parsed;
    return true;
}

void projectPointCloudPoints(const QVector<QVector3D>& points,
                             const PointCloudCalibration& calibration,
                             QVector<int>* insideIndices,
                             QVector<QVector2D>* pixels,
                             QVector<double>* depths) {
    insideIndices->clear();
    pixels->clear();
    depths->clear();
    // Transform into camera space first (rotation is row-major [R|t]).
    QVector<QVector3D> camera(points.size());
    for (int i = 0; i < points.size(); ++i) {
        const QVector3D& point = points.at(i);
        camera[i] = QVector3D(float(calibration.rotation[0] * point.x() +
                                    calibration.rotation[1] * point.y() +
                                    calibration.rotation[2] * point.z() +
                                    calibration.translation[0]),
                              float(calibration.rotation[3] * point.x() +
                                    calibration.rotation[4] * point.y() +
                                    calibration.rotation[5] * point.z() +
                                    calibration.translation[1]),
                              float(calibration.rotation[6] * point.x() +
                                    calibration.rotation[7] * point.y() +
                                    calibration.rotation[8] * point.z() +
                                    calibration.translation[2]));
    }
    QVector<QVector2D> projected;
    QVector<bool> inFront;
    projectCameraPoints(camera, calibration, &projected, &inFront);
    // Far-to-near stable order (upstream argsort of -z): collect the valid
    // in-bounds candidates, then sort by decreasing depth.
    QVector<QPair<double, int>> candidates;
    candidates.reserve(points.size());
    for (int i = 0; i < points.size(); ++i) {
        if (!inFront.at(i)) continue;
        const QVector2D pixel = projected.at(i);
        if (pixel.x() < 0 || pixel.x() >= calibration.imageSize.width() ||
            pixel.y() < 0 || pixel.y() >= calibration.imageSize.height())
            continue;
        candidates.append({double(camera.at(i).z()), i});
    }
    std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const QPair<double, int>& a, const QPair<double, int>& b) {
                return a.first > b.first;
            });
    insideIndices->reserve(candidates.size());
    pixels->reserve(candidates.size());
    depths->reserve(candidates.size());
    for (const auto& candidate : candidates) {
        insideIndices->append(candidate.second);
        pixels->append(projected.at(candidate.second));
        depths->append(candidate.first);
    }
}

void projectPointCloudCuboid(const QVector<QVector3D>& corners,
                             const PointCloudCalibration& calibration,
                             QVector<QPair<QVector2D, QVector2D>>* segments) {
    segments->clear();
    if (corners.size() != kCornerCount) return;
    constexpr double kNear = 1e-6;
    // Transform the corners into camera space.
    QVector<QVector3D> camera(kCornerCount);
    for (int i = 0; i < kCornerCount; ++i) {
        const QVector3D& point = corners.at(i);
        camera[i] = QVector3D(float(calibration.rotation[0] * point.x() +
                                    calibration.rotation[1] * point.y() +
                                    calibration.rotation[2] * point.z() +
                                    calibration.translation[0]),
                              float(calibration.rotation[3] * point.x() +
                                    calibration.rotation[4] * point.y() +
                                    calibration.rotation[5] * point.z() +
                                    calibration.translation[1]),
                              float(calibration.rotation[6] * point.x() +
                                    calibration.rotation[7] * point.y() +
                                    calibration.rotation[8] * point.z() +
                                    calibration.translation[2]));
    }
    // Per upstream project_cuboid: distorted edges are sampled at 33
    // fractions so the curved projection stays smooth; undistorted edges
    // project to straight lines and need only their endpoints.
    const int samples = calibration.opencv5 ? 33 : 2;
    for (const auto& edge : pointCloudCuboidEdges()) {
        QVector3D start = camera.at(edge.first);
        QVector3D end = camera.at(edge.second);
        // Near-plane clip: pull a behind endpoint forward along the edge.
        if (start.z() < kNear && end.z() < kNear) continue;
        if (start.z() < kNear) {
            const double fraction = (kNear - start.z()) / (end.z() - start.z());
            start = start + (end - start) * float(fraction);
        } else if (end.z() < kNear) {
            const double fraction = (kNear - end.z()) / (start.z() - end.z());
            end = end + (start - end) * float(fraction);
        }
        QVector<QVector2D> chain;
        chain.reserve(samples);
        for (int sample = 0; sample < samples; ++sample) {
            const double fraction = double(sample) / (samples - 1);
            const QVector3D point = start + (end - start) * float(fraction);
            if (point.z() <= 0) continue;
            double x = 0.0;
            double y = 0.0;
            if (!distortPixel(calibration, double(point.x()) / point.z(),
                              double(point.y()) / point.z(), &x, &y))
                continue;
            chain.append(QVector2D(float(x), float(y)));
        }
        for (int i = 0; i + 1 < chain.size(); ++i) {
            QVector2D clippedStart;
            QVector2D clippedEnd;
            if (clipSegment(chain.at(i), chain.at(i + 1),
                            calibration.imageSize.width(),
                            calibration.imageSize.height(), &clippedStart,
                            &clippedEnd))
                segments->append({clippedStart, clippedEnd});
        }
    }
}

QImage renderPointCloudCameraOverlay(
        const QImage& image,
        const PointCloudCalibration& calibration,
        const QVector<QVector3D>& points,
        const QVector<quint32>& labels,
        const QHash<quint16, QColor>& semanticColors,
        int pointSize) {
    QImage canvas = image.convertToFormat(QImage::Format_ARGB32);
    if (canvas.isNull() || points.isEmpty()) return canvas;
    // Far-to-near projection: painter draws in that order so near points
    // overlay far ones (upstream project_points ordering semantics).
    QVector<int> indices;
    QVector<QVector2D> pixels;
    QVector<double> depths;
    projectPointCloudPoints(points, calibration, &indices, &pixels, &depths);
    QPainter painter(&canvas);
    const int size = std::max(1, pointSize);
    for (int i = 0; i < indices.size(); ++i) {
        const int source = indices.at(i);
        const quint16 semantic = quint16(labels.value(source) & 0xffffu);
        const QColor color =
                semanticColors.value(semantic, QColor(128, 128, 128));
        painter.fillRect(int(pixels.at(i).x()), int(pixels.at(i).y()), size,
                         size, color);
    }
    painter.end();
    return canvas;
}
