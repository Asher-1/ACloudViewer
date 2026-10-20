// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCamera.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLineF>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
struct Vec3 {
    double x = 0;
    double y = 0;
    double z = 0;
};

bool fail(QString* error, const QString& message) {
    if (error) *error = message;
    return false;
}

bool finiteNumber(const QJsonValue& value, double* result) {
    if (!value.isDouble()) return false;
    const double number = value.toDouble();
    if (!std::isfinite(number)) return false;
    if (result) *result = number;
    return true;
}

bool matrix(const QJsonValue& value, int rows, int columns, double* output) {
    if (!value.isArray()) return false;
    const QJsonArray outer = value.toArray();
    if (outer.size() != rows) return false;
    for (int row = 0; row < rows; ++row) {
        if (!outer[row].isArray()) return false;
        const QJsonArray inner = outer[row].toArray();
        if (inner.size() != columns) return false;
        for (int column = 0; column < columns; ++column)
            if (!finiteNumber(inner[column], &output[row * columns + column]))
                return false;
    }
    return true;
}

double determinant3(const std::array<double, 16>& m) {
    return m[0] * (m[5] * m[10] - m[6] * m[9]) -
           m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

Vec3 cameraPoint(const Vec3& point, const PointCloudCalibration& calibration) {
    const auto& m = calibration.pointCloudToCamera;
    return {m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3],
            m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7],
            m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11]};
}

QPointF projectCamera(const Vec3& camera,
                      const PointCloudCalibration& calibration) {
    const double z = camera.z;
    double x = camera.x / z;
    double y = camera.y / z;
    if (calibration.distortionModel == QStringLiteral("opencv5")) {
        const double radius2 = x * x + y * y;
        const double k1 = calibration.distortion[0];
        const double k2 = calibration.distortion[1];
        const double p1 = calibration.distortion[2];
        const double p2 = calibration.distortion[3];
        const double k3 = calibration.distortion[4];
        const double radial =
                1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3));
        const double distortedX =
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x);
        const double distortedY =
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y;
        x = distortedX;
        y = distortedY;
    }
    return {x * calibration.cameraMatrix[0] + calibration.cameraMatrix[2],
            y * calibration.cameraMatrix[4] + calibration.cameraMatrix[5]};
}

bool validCalibration(const PointCloudCalibration& c, QString* error) {
    if (c.imageSize.width() <= 0 || c.imageSize.height() <= 0)
        return fail(error,
                    QStringLiteral("Calibration image_size is invalid."));
    const auto& k = c.cameraMatrix;
    if (!(k[0] > 0 && k[4] > 0) || k[1] != 0 || k[3] != 0 || k[6] != 0 ||
        k[7] != 0 || k[8] != 1)
        return fail(error,
                    QStringLiteral("Calibration camera_matrix is invalid."));
    const auto& m = c.pointCloudToCamera;
    if (m[12] != 0 || m[13] != 0 || m[14] != 0 || m[15] != 1)
        return fail(error,
                    QStringLiteral("Calibration transform must be rigid."));
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            double dot = 0;
            for (int axis = 0; axis < 3; ++axis)
                dot += m[row * 4 + axis] * m[column * 4 + axis];
            if (std::abs(dot - (row == column ? 1.0 : 0.0)) > 1e-5)
                return fail(error,
                            QStringLiteral("Calibration rotation is invalid."));
        }
    if (std::abs(determinant3(m) - 1.0) > 1e-5)
        return fail(error,
                    QStringLiteral("Calibration rotation is not proper."));
    if (c.distortionModel != QStringLiteral("none") &&
        c.distortionModel != QStringLiteral("opencv5"))
        return fail(error, QStringLiteral("Unsupported distortion model."));
    return true;
}

bool clipSegment(QPointF start,
                 QPointF end,
                 double width,
                 double height,
                 QLineF* result) {
    const double dx = end.x() - start.x();
    const double dy = end.y() - start.y();
    double lower = 0;
    double upper = 1;
    const std::array<double, 4> p{{-dx, dx, -dy, dy}};
    const std::array<double, 4> q{{start.x(), width - 1 - start.x(), start.y(),
                                   height - 1 - start.y()}};
    for (int i = 0; i < 4; ++i) {
        if (p[i] == 0) {
            if (q[i] < 0) return false;
            continue;
        }
        const double ratio = q[i] / p[i];
        if (p[i] < 0)
            lower = std::max(lower, ratio);
        else
            upper = std::min(upper, ratio);
        if (lower > upper) return false;
    }
    *result = QLineF(start + (end - start) * lower,
                     start + (end - start) * upper);
    return true;
}
}  // namespace

bool loadPointCloudCalibration(const QString& path,
                               PointCloudCalibration* calibration,
                               QString* error) {
    if (!calibration || !error) return false;
    error->clear();
    QFile file(path);
    if (QFileInfo(path).suffix().compare(QStringLiteral("json"),
                                         Qt::CaseInsensitive) != 0)
        return fail(error, QStringLiteral("Calibration must be a JSON file."));
    if (!file.open(QIODevice::ReadOnly)) return fail(error, file.errorString());
    QJsonParseError parse;
    const QJsonDocument document =
            QJsonDocument::fromJson(file.readAll(), &parse);
    if (parse.error != QJsonParseError::NoError || !document.isObject())
        return fail(error,
                    QStringLiteral("Calibration must be a JSON object."));
    const QJsonObject object = document.object();
    const QSet<QString> fields{QStringLiteral("schema_version"),
                               QStringLiteral("image_size"),
                               QStringLiteral("camera_model"),
                               QStringLiteral("camera_matrix"),
                               QStringLiteral("T_pointcloud_to_camera"),
                               QStringLiteral("distortion_model"),
                               QStringLiteral("distortion_coefficients")};
    for (const QString& field : fields)
        if (!object.contains(field))
            return fail(
                    error,
                    QStringLiteral("Missing calibration field: %1").arg(field));
    for (auto it = object.begin(); it != object.end(); ++it)
        if (!fields.contains(it.key()))
            return fail(error, QStringLiteral("Unknown calibration field: %1")
                                       .arg(it.key()));
    if (!object[QStringLiteral("schema_version")].isDouble() ||
        object[QStringLiteral("schema_version")].toInt() != 1)
        return fail(error, QStringLiteral("schema_version must be 1."));
    if (object[QStringLiteral("camera_model")] !=
        QJsonValue(QStringLiteral("pinhole")))
        return fail(error, QStringLiteral("camera_model must be pinhole."));
    PointCloudCalibration parsed;
    const QJsonArray size = object[QStringLiteral("image_size")].toArray();
    if (size.size() != 2 || !size[0].isDouble() || !size[1].isDouble() ||
        size[0].toInt() <= 0 || size[1].toInt() <= 0)
        return fail(error,
                    QStringLiteral("image_size must be positive integers."));
    parsed.imageSize = QSize(size[0].toInt(), size[1].toInt());
    if (!matrix(object[QStringLiteral("camera_matrix")], 3, 3,
                parsed.cameraMatrix.data()) ||
        !matrix(object[QStringLiteral("T_pointcloud_to_camera")], 4, 4,
                parsed.pointCloudToCamera.data()))
        return fail(error, QStringLiteral("Calibration matrices are invalid."));
    if (!object[QStringLiteral("distortion_model")].isString())
        return fail(error, QStringLiteral("distortion_model is invalid."));
    parsed.distortionModel =
            object[QStringLiteral("distortion_model")].toString();
    const QJsonArray distortion =
            object[QStringLiteral("distortion_coefficients")].toArray();
    const int expected =
            parsed.distortionModel == QStringLiteral("none") ? 0 : 5;
    if (distortion.size() != expected)
        return fail(
                error,
                QStringLiteral("distortion_coefficients has invalid length."));
    for (int i = 0; i < distortion.size(); ++i)
        if (!finiteNumber(distortion[i], &parsed.distortion[i]))
            return fail(error,
                        QStringLiteral("distortion_coefficients is invalid."));
    if (!validCalibration(parsed, error)) return false;
    *calibration = parsed;
    return true;
}

bool projectPointCloud(const QVector<QVector4D>& points,
                       const PointCloudCalibration& calibration,
                       QVector<int>* indices,
                       QVector<QPoint>* pixels,
                       QString* error) {
    if (!indices || !pixels || !validCalibration(calibration, error))
        return false;
    indices->clear();
    pixels->clear();
    struct Candidate {
        int index;
        double depth;
        QPoint pixel;
    };
    QVector<Candidate> candidates;
    candidates.reserve(points.size());
    for (int i = 0; i < points.size(); ++i) {
        const QVector4D point = points[i];
        const Vec3 camera =
                cameraPoint({point.x(), point.y(), point.z()}, calibration);
        if (!std::isfinite(camera.x) || !std::isfinite(camera.y) ||
            !std::isfinite(camera.z) || camera.z <= 0)
            continue;
        const QPointF projected = projectCamera(camera, calibration);
        if (!std::isfinite(projected.x()) || !std::isfinite(projected.y()) ||
            projected.x() < 0 ||
            projected.x() >= calibration.imageSize.width() ||
            projected.y() < 0 ||
            projected.y() >= calibration.imageSize.height())
            continue;
        candidates.append(
                {i, camera.z, QPoint(int(projected.x()), int(projected.y()))});
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& left, const Candidate& right) {
                         return left.depth > right.depth;
                     });
    for (const Candidate& candidate : candidates) {
        indices->append(candidate.index);
        pixels->append(candidate.pixel);
    }
    return true;
}

bool projectCuboid(const PointCloudCuboid& cuboid,
                   const PointCloudCalibration& calibration,
                   QVector<QLineF>* segments,
                   QString* error) {
    if (!segments || !validCalibration(calibration, error)) return false;
    segments->clear();
    const auto corners = cuboid.corners();
    constexpr int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (const auto& edge : edges) {
        Vec3 a = cameraPoint(
                {corners[edge[0]][0], corners[edge[0]][1], corners[edge[0]][2]},
                calibration);
        Vec3 b = cameraPoint(
                {corners[edge[1]][0], corners[edge[1]][1], corners[edge[1]][2]},
                calibration);
        constexpr double near = 1e-6;
        if (a.z < near && b.z < near) continue;
        if (a.z < near || b.z < near) {
            const double fraction = (near - a.z) / (b.z - a.z);
            if (a.z < near) {
                a = {a.x + fraction * (b.x - a.x), a.y + fraction * (b.y - a.y),
                     near};
            } else {
                b = {b.x + fraction * (a.x - b.x), b.y + fraction * (a.y - b.y),
                     near};
            }
        }
        const int samples =
                calibration.distortionModel == QStringLiteral("opencv5") ? 33
                                                                         : 2;
        for (int sample = 0; sample < samples - 1; ++sample) {
            const double first = double(sample) / double(samples - 1);
            const double second = double(sample + 1) / double(samples - 1);
            const Vec3 firstPoint{a.x + first * (b.x - a.x),
                                  a.y + first * (b.y - a.y),
                                  a.z + first * (b.z - a.z)};
            const Vec3 secondPoint{a.x + second * (b.x - a.x),
                                   a.y + second * (b.y - a.y),
                                   a.z + second * (b.z - a.z)};
            const QPointF start = projectCamera(firstPoint, calibration);
            const QPointF end = projectCamera(secondPoint, calibration);
            QLineF clipped;
            if (clipSegment(start, end, calibration.imageSize.width(),
                            calibration.imageSize.height(), &clipped))
                segments->append(clipped);
        }
    }
    return true;
}

bool listPointCloudCameraImages(const QString& directory,
                                QHash<QString, QString>* imagePaths,
                                QString* error) {
    if (!imagePaths || !error) return false;
    imagePaths->clear();
    error->clear();
    const QDir source(directory);
    if (!source.exists())
        return fail(error,
                    QStringLiteral("Camera image directory does not exist."));
    QSet<QString> formats;
    for (const QByteArray& format : QImageReader::supportedImageFormats())
        formats.insert(QString::fromLatin1(format).toLower());
    const QFileInfoList entries = source.entryInfoList(
            QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    for (const QFileInfo& entry : entries) {
        if (!formats.contains(entry.suffix().toLower())) continue;
        const QString stem = entry.completeBaseName();
        if (imagePaths->contains(stem))
            return fail(
                    error,
                    QStringLiteral("Multiple images have the same basename: %1")
                            .arg(stem));
        imagePaths->insert(stem, entry.absoluteFilePath());
    }
    if (imagePaths->isEmpty())
        return fail(error,
                    QStringLiteral("No supported images in this directory."));
    return true;
}

bool loadPointCloudCameraSource(const QString& directory,
                                const QString& calibrationPath,
                                PointCloudCameraSource* source,
                                QString* error) {
    if (!source || !error) return false;
    error->clear();
    PointCloudCameraSource loaded;
    loaded.directory = QFileInfo(directory).absoluteFilePath();
    if (!listPointCloudCameraImages(loaded.directory, &loaded.imagePaths,
                                    error))
        return false;
    const QString calibration = calibrationPath.trimmed();
    if (!calibration.isEmpty()) {
        if (!loadPointCloudCalibration(calibration, &loaded.calibration, error))
            return false;
        loaded.calibrationPath = QFileInfo(calibration).absoluteFilePath();
        loaded.hasCalibration = true;
    }
    *source = std::move(loaded);
    return true;
}

bool pointCloudCameraDepthColors(const QVector<QVector4D>& points,
                                 const PointCloudCalibration& calibration,
                                 const QVector<int>& indices,
                                 QVector<QColor>* colors,
                                 QString* error,
                                 double* nearDepth,
                                 double* farDepth) {
    if (!colors)
        return fail(error, QStringLiteral("Missing depth color output."));
    QVector<double> depths;
    const auto& transform = calibration.pointCloudToCamera;
    for (int index : indices) {
        if (index < 0 || index >= points.size())
            return fail(error,
                        QStringLiteral(
                                "Projected point index is outside the frame."));
        const auto& point = points[index];
        const double depth = transform[8] * point.x() +
                             transform[9] * point.y() +
                             transform[10] * point.z() + transform[11];
        if (!std::isfinite(depth))
            return fail(error,
                        QStringLiteral("Projected depth must be finite."));
        depths.append(depth);
    }
    QVector<QColor> result;
    if (nearDepth) *nearDepth = 0;
    if (farDepth) *farDepth = 0;
    if (!depths.isEmpty()) {
        QVector<double> ranked = depths;
        // Only four order statistics are needed. nth_element keeps expected
        // O(n) work instead of sorting every projected point for two quantiles.
        const auto percentile = [&](double quantile) {
            const double rank = quantile * (ranked.size() - 1);
            const int low = int(std::floor(rank)), high = int(std::ceil(rank));
            std::nth_element(ranked.begin(), ranked.begin() + low,
                             ranked.end());
            const double lower = ranked[low];
            std::nth_element(ranked.begin(), ranked.begin() + high,
                             ranked.end());
            return lower + (rank - low) * (ranked[high] - lower);
        };
        const double near = percentile(0.02), far = percentile(0.98);
        if (nearDepth) *nearDepth = near;
        if (farDepth) *farDepth = far;
        const int stops[5][3] = {{255, 100, 65},
                                 {255, 215, 64},
                                 {103, 220, 120},
                                 {50, 190, 230},
                                 {90, 100, 235}};
        for (double depth : depths) {
            const double position =
                    std::clamp((depth - near) / std::max(far - near, 1e-6), 0.0,
                               1.0) *
                    4;
            const int low = std::min(3, int(position));
            const double fraction = position - low;
            int channels[3];
            for (int channel = 0; channel < 3; ++channel)
                channels[channel] = int(stops[low][channel] +
                                        fraction * (stops[low + 1][channel] -
                                                    stops[low][channel]));
            result.append(QColor(channels[0], channels[1], channels[2]));
        }
    }
    *colors = std::move(result);
    return true;
}
