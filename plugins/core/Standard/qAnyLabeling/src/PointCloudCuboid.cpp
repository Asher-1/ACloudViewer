// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCuboid.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <cmath>

namespace {

constexpr double kMinSize = 0.01;

bool finiteTriple(const double values[3]) {
    return std::isfinite(values[0]) && std::isfinite(values[1]) &&
           std::isfinite(values[2]);
}

}  // namespace

void pointCloudCuboidRotation(const double angles[3], double rotation[9]) {
    // Upstream rotation_matrix(angles) row-major layout.
    const double a = std::cos(angles[0]), b = std::sin(angles[0]);
    const double c = std::cos(angles[1]), d = std::sin(angles[1]);
    const double e = std::cos(angles[2]), f = std::sin(angles[2]);
    rotation[0] = c * e;
    rotation[1] = -c * f;
    rotation[2] = d;
    rotation[3] = a * f + b * e * d;
    rotation[4] = a * e - b * f * d;
    rotation[5] = -b * c;
    rotation[6] = b * f - a * e * d;
    rotation[7] = b * e + a * f * d;
    rotation[8] = a * c;
}

bool PointCloudCuboid::valid(QString* error) const {
    if (id < 1 || id > 65535 || classId < 1 || classId > 65535) {
        if (error)
            *error = QStringLiteral("Cuboid IDs and classes must be 1-65535.");
        return false;
    }
    if (!finiteTriple(center) || !finiteTriple(size) ||
        !finiteTriple(rotation)) {
        if (error)
            *error = QStringLiteral("Cuboid fields require finite numbers.");
        return false;
    }
    if (size[0] < kMinSize || size[1] < kMinSize || size[2] < kMinSize) {
        if (error)
            *error = QStringLiteral("Cuboid dimensions must be at least %1.")
                             .arg(kMinSize);
        return false;
    }
    return true;
}

QVector<QVector3D> PointCloudCuboid::corners() const {
    // Distinct buffers: the rotation helper reads angles before writing the
    // matrix, but aliasing the two would make that ordering load-bearing.
    const double angles[3] = {rotation[0], rotation[1], rotation[2]};
    double matrix[9];
    pointCloudCuboidRotation(angles, matrix);
    static const double signs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1},
                                       {-1, 1, -1},  {-1, -1, 1}, {1, -1, 1},
                                       {1, 1, 1},    {-1, 1, 1}};
    QVector<QVector3D> result;
    result.reserve(8);
    for (const auto& sign : signs) {
        // corners = CORNER_SIGNS * (size/2) @ R.T + center
        const double local[3] = {sign[0] * size[0] * 0.5,
                                 sign[1] * size[1] * 0.5,
                                 sign[2] * size[2] * 0.5};
        double world[3] = {0.0, 0.0, 0.0};
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                world[row] += matrix[column * 3 + row] * local[column];
        result.append(QVector3D(float(world[0] + center[0]),
                                float(world[1] + center[1]),
                                float(world[2] + center[2])));
    }
    return result;
}

bool PointCloudCuboid::operator==(const PointCloudCuboid& other) const {
    if (id != other.id || classId != other.classId ||
        occluded != other.occluded || locked != other.locked)
        return false;
    for (int i = 0; i < 3; ++i) {
        if (center[i] != other.center[i] || size[i] != other.size[i] ||
            rotation[i] != other.rotation[i])
            return false;
    }
    return true;
}

bool loadPointCloudCuboids(const QString& path,
                           const QString& pointCloudName,
                           QVector<PointCloudCuboid>* cuboids,
                           QString* error) {
    if (!cuboids || !error) return false;
    error->clear();
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = file.errorString();
        return false;
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject()) {
        *error = QStringLiteral("Expected a JSON object.");
        return false;
    }
    const QJsonObject data = document.object();
    if (!data.value(QStringLiteral("version")).isDouble() ||
        data.value(QStringLiteral("version")).toInt() != 1 ||
        data.value(QStringLiteral("type")).toString() !=
                QLatin1String("pointcloud-cuboids") ||
        data.value(QStringLiteral("point_cloud")).toString() !=
                pointCloudName ||
        !data.value(QStringLiteral("cuboids")).isArray()) {
        *error = QStringLiteral(
                "Expected version 1 cuboids for this point cloud.");
        return false;
    }
    QVector<PointCloudCuboid> parsed;
    const QJsonArray entries = data.value(QStringLiteral("cuboids")).toArray();
    for (const QJsonValue& value : entries) {
        if (!value.isObject()) {
            *error = QStringLiteral("Each cuboid must be an object.");
            return false;
        }
        const QJsonObject entry = value.toObject();
        PointCloudCuboid box;
        box.id = entry.value(QStringLiteral("id")).toInt();
        box.classId = entry.value(QStringLiteral("class_id")).toInt();
        const QJsonArray center =
                entry.value(QStringLiteral("center")).toArray();
        const QJsonArray size = entry.value(QStringLiteral("size")).toArray();
        const QJsonArray rotation =
                entry.value(QStringLiteral("rotation")).toArray();
        if (center.size() != 3 || size.size() != 3 || rotation.size() != 3) {
            *error = QStringLiteral("Invalid cuboid fields.");
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            box.center[i] = center.at(i).toDouble();
            box.size[i] = size.at(i).toDouble();
            box.rotation[i] = rotation.at(i).toDouble();
        }
        box.occluded = entry.value(QStringLiteral("occluded")).toBool(false);
        box.locked = entry.value(QStringLiteral("locked")).toBool(false);
        QString boxError;
        if (!box.valid(&boxError)) {
            *error = boxError;
            return false;
        }
        parsed.append(box);
    }
    // Duplicate IDs are rejected (upstream parity).
    for (int i = 0; i < parsed.size(); ++i)
        for (int j = i + 1; j < parsed.size(); ++j)
            if (parsed.at(i).id == parsed.at(j).id) {
                *error = QStringLiteral("Duplicate cuboid IDs.");
                return false;
            }
    *cuboids = parsed;
    return true;
}

bool savePointCloudCuboids(const QString& path,
                           const QString& pointCloudName,
                           const QVector<PointCloudCuboid>& cuboids,
                           QString* error) {
    if (!error) return false;
    error->clear();
    if (!path.endsWith(QLatin1String(".cuboids.json"))) {
        *error = QStringLiteral(
                "Cuboid output must use the .cuboids.json extension.");
        return false;
    }
    for (int i = 0; i < cuboids.size(); ++i)
        for (int j = i + 1; j < cuboids.size(); ++j)
            if (cuboids.at(i).id == cuboids.at(j).id) {
                *error = QStringLiteral("Duplicate cuboid IDs.");
                return false;
            }
    QJsonObject data;
    data.insert(QStringLiteral("version"), 1);
    data.insert(QStringLiteral("type"), QStringLiteral("pointcloud-cuboids"));
    data.insert(QStringLiteral("point_cloud"), pointCloudName);
    QJsonArray entries;
    for (const PointCloudCuboid& box : cuboids) {
        QJsonObject entry;
        entry.insert(QStringLiteral("id"), box.id);
        entry.insert(QStringLiteral("class_id"), box.classId);
        QJsonArray center;
        for (double value : box.center) center.append(value);
        entry.insert(QStringLiteral("center"), center);
        QJsonArray size;
        for (double value : box.size) size.append(value);
        entry.insert(QStringLiteral("size"), size);
        QJsonArray rotation;
        for (double value : box.rotation) rotation.append(value);
        entry.insert(QStringLiteral("rotation"), rotation);
        entry.insert(QStringLiteral("occluded"), box.occluded);
        entry.insert(QStringLiteral("locked"), box.locked);
        entries.append(entry);
    }
    data.insert(QStringLiteral("cuboids"), entries);
    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(QJsonDocument(data).toJson(QJsonDocument::Indented)) < 0 ||
        !output.commit()) {
        *error = output.errorString();
        return false;
    }
    return true;
}
