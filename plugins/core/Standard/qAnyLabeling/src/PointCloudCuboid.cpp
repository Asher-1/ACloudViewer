// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCuboid.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr double kMinimum = 0.01;
bool fail(QString* error, const QString& message) {
    if (error) *error = message;
    return false;
}
QJsonArray array(const CuboidVector& v) { return {v[0], v[1], v[2]}; }
bool vector(const QJsonValue& value, CuboidVector* out) {
    if (!value.isArray() || value.toArray().size() != 3) return false;
    const auto a = value.toArray();
    for (int i = 0; i < 3; ++i) {
        if (!a[i].isDouble() || !std::isfinite(a[i].toDouble())) return false;
        (*out)[i] = a[i].toDouble();
    }
    return true;
}
}  // namespace

bool pointCloudCuboidMesh(const QVector<PointCloudCuboid>& boxes,
                          int selected,
                          const QHash<quint16, QColor>& colors,
                          const PointCloudCuboid* preview,
                          bool orthographic,
                          PointCloudCuboidMesh* output,
                          QString* error) {
    if (!output)
        return fail(error, QStringLiteral("Missing cuboid mesh output."));
    QVector<float> faces, edges;
    constexpr int faceIndices[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7},
                                       {0, 1, 5, 4}, {2, 3, 7, 6},
                                       {0, 3, 7, 4}, {1, 2, 6, 5}};
    constexpr int edgeIndices[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                        {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                        {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    auto append = [](QVector<float>& target, const CuboidVector& point,
                     const CuboidVector& rgb, double alpha) {
        for (double value : point) target.append(float(value));
        for (double value : rgb) target.append(float(value));
        target.append(float(alpha));
    };
    // O(boxes + total dashed-edge length / 0.1), O(output vertices) memory.
    // Metric dash spacing is part of the source contract, not screen pixels.
    for (int n = 0; n < boxes.size() + (preview ? 1 : 0); ++n) {
        const bool creating = n == boxes.size();
        const auto& box = creating ? *preview : boxes[n];
        if (!creating && !box.valid(error)) return false;
        const auto corners = box.corners();
        const QColor color = creating ? QColor("#ffd54f")
                                      : colors.value(quint16(box.classId),
                                                     QColor("#64b5f6"));
        const CuboidVector rgb{{color.redF(), color.greenF(), color.blueF()}};
        if (creating || box.id == selected) {
            for (const auto& indices : faceIndices) {
                std::array<CuboidVector, 4> vertices;
                CuboidVector a{}, b{}, middle{};
                for (int i = 0; i < 4; ++i) vertices[i] = corners[indices[i]];
                for (int axis = 0; axis < 3; ++axis) {
                    a[axis] = vertices[1][axis] - vertices[0][axis];
                    b[axis] = vertices[2][axis] - vertices[0][axis];
                    for (const auto& vertex : vertices)
                        middle[axis] += vertex[axis] / 4;
                    middle[axis] -= box.center[axis];
                }
                const CuboidVector normal{{a[1] * b[2] - a[2] * b[1],
                                           a[2] * b[0] - a[0] * b[2],
                                           a[0] * b[1] - a[1] * b[0]}};
                double dot = 0;
                for (int axis = 0; axis < 3; ++axis)
                    dot += normal[axis] * middle[axis];
                if (dot < 0) std::reverse(vertices.begin(), vertices.end());
                for (int index : {0, 1, 2, 0, 2, 3})
                    append(faces, vertices[index], rgb, creating ? .4 : .3);
            }
        }
        const CuboidVector outline = creating ? CuboidVector{{1, .93, .65}}
                                     : orthographic ? CuboidVector{{1, 1, 1}}
                                                    : rgb;
        for (const auto& indices : edgeIndices) {
            const auto& first = corners[indices[0]];
            const auto& last = corners[indices[1]];
            if (box.occluded && !orthographic) {
                CuboidVector direction{};
                double squared = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    direction[axis] = last[axis] - first[axis];
                    squared += direction[axis] * direction[axis];
                }
                const double length = std::sqrt(squared),
                             count = std::ceil(length / .1);
                // QVector/QOpenGLBuffer sizes are signed ints. Reject an
                // unrepresentable mesh before allocating or iterating it.
                if (!std::isfinite(count) ||
                    count > (std::numeric_limits<int>::max() / sizeof(float) -
                             faces.size() - edges.size()) /
                                    14)
                    return fail(error,
                                QStringLiteral("Cuboid mesh exceeds the "
                                               "renderer buffer limit."));
                for (int i = 0; i < int(count); ++i)
                    for (double distance :
                         {i * .1, std::min(i * .1 + .05, length)}) {
                        auto point = first;
                        for (int axis = 0; axis < 3; ++axis)
                            point[axis] += distance * direction[axis] / length;
                        append(edges, point, outline, 1);
                    }
            } else {
                append(edges, first, outline, 1);
                append(edges, last, outline, 1);
            }
        }
    }
    PointCloudCuboidMesh result;
    result.faceVertices = faces.size() / 7;
    result.vertices = std::move(faces);
    result.vertices += edges;
    *output = std::move(result);
    return true;
}

CuboidMatrix cuboidRotationMatrix(const CuboidVector& angles) {
    const double a = std::cos(angles[0]), b = std::sin(angles[0]);
    const double c = std::cos(angles[1]), d = std::sin(angles[1]);
    const double e = std::cos(angles[2]), f = std::sin(angles[2]);
    return {{{c * e, -c * f, d},
             {a * f + b * e * d, a * e - b * f * d, -b * c},
             {b * f - a * e * d, b * e + a * f * d, a * c}}};
}
CuboidVector cuboidRotationAngles(const CuboidMatrix& m) {
    const double y = std::asin(std::clamp(m[0][2], -1.0, 1.0));
    if (std::abs(m[0][2]) < 0.9999999)
        return {{std::atan2(-m[1][2], m[2][2]), y,
                 std::atan2(-m[0][1], m[0][0])}};
    return {{std::atan2(m[2][1], m[1][1]), y, 0}};
}
CuboidVector cuboidTransform(const CuboidMatrix& m,
                             const CuboidVector& p,
                             bool inverse) {
    CuboidVector r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i] += (inverse ? m[j][i] : m[i][j]) * p[j];
    return r;
}
bool PointCloudCuboid::valid(QString* error) const {
    if (id < 1 || id > 65535 || classId < 1 || classId > 65535)
        return fail(
                error,
                QStringLiteral("Cuboid IDs and classes must be in 1-65535."));
    for (const auto* v : {&center, &size, &rotation})
        for (double n : *v)
            if (!std::isfinite(n))
                return fail(
                        error,
                        QStringLiteral("Cuboid requires finite coordinates."));
    for (double n : size)
        if (n < kMinimum)
            return fail(
                    error,
                    QStringLiteral("Cuboid dimensions must be at least 0.01."));
    return true;
}
bool PointCloudCuboid::operator==(const PointCloudCuboid& o) const {
    return id == o.id && classId == o.classId && center == o.center &&
           size == o.size && rotation == o.rotation && occluded == o.occluded &&
           locked == o.locked;
}
std::array<CuboidVector, 8> PointCloudCuboid::corners() const {
    const int signs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                             {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
    const auto m = cuboidRotationMatrix(rotation);
    std::array<CuboidVector, 8> result{};
    for (int i = 0; i < 8; ++i) {
        auto p = cuboidTransform(
                m, {{signs[i][0] * size[0] / 2, signs[i][1] * size[1] / 2,
                     signs[i][2] * size[2] / 2}});
        for (int j = 0; j < 3; ++j) result[i][j] = center[j] + p[j];
    }
    return result;
}
bool PointCloudCuboid::contains(const CuboidVector& point) const {
    CuboidVector p{};
    for (int i = 0; i < 3; ++i) p[i] = point[i] - center[i];
    p = cuboidTransform(cuboidRotationMatrix(rotation), p, true);
    for (int i = 0; i < 3; ++i)
        if (std::abs(p[i]) > size[i] / 2 + 1e-6) return false;
    return true;
}
PointCloudCuboid PointCloudCuboid::resized(const std::array<int, 3>& signs,
                                           const CuboidVector& delta) const {
    auto result = *this;
    CuboidVector shift{};
    for (int i = 0; i < 3; ++i)
        if (signs[i]) {
            result.size[i] = std::max(kMinimum, size[i] + signs[i] * delta[i]);
            shift[i] = signs[i] * (result.size[i] - size[i]) / 2;
        }
    shift = cuboidTransform(cuboidRotationMatrix(rotation), shift);
    for (int i = 0; i < 3; ++i) result.center[i] += shift[i];
    return result;
}
PointCloudCuboid PointCloudCuboid::rotated(int axis, double angle) const {
    if (axis < 0 || axis > 2) return *this;
    CuboidVector angles{};
    angles[axis] = angle;
    const auto a = cuboidRotationMatrix(rotation),
               b = cuboidRotationMatrix(angles);
    CuboidMatrix c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) c[i][j] += a[i][k] * b[k][j];
    auto result = *this;
    result.rotation = cuboidRotationAngles(c);
    return result;
}
bool PointCloudCuboid::fitted(const QVector<QVector4D>& points,
                              PointCloudCuboid* result,
                              QString* error,
                              bool insideOnly) const {
    if (!result) return false;
    CuboidVector low, high;
    low.fill(std::numeric_limits<double>::infinity());
    high.fill(-std::numeric_limits<double>::infinity());
    const auto m = cuboidRotationMatrix(rotation);
    bool found = false;
    // One pass over borrowed frame storage, O(n) time and O(1) extra memory.
    for (const auto& p : points) {
        const CuboidVector world{{p.x(), p.y(), p.z()}};
        CuboidVector local{};
        for (int i = 0; i < 3; ++i) local[i] = world[i] - center[i];
        local = cuboidTransform(m, local, true);
        if (insideOnly && (std::abs(local[0]) > size[0] / 2 + 1e-6 ||
                           std::abs(local[1]) > size[1] / 2 + 1e-6 ||
                           std::abs(local[2]) > size[2] / 2 + 1e-6))
            continue;
        for (int i = 0; i < 3; ++i) {
            low[i] = std::min(low[i], local[i]);
            high[i] = std::max(high[i], local[i]);
        }
        found = true;
    }
    if (!found)
        return fail(error,
                    QStringLiteral(
                            "There are no points inside this cuboid to fit."));
    *result = *this;
    CuboidVector shift{};
    for (int i = 0; i < 3; ++i) {
        shift[i] = (low[i] + high[i]) / 2;
        result->size[i] = std::max(kMinimum, high[i] - low[i]);
    }
    shift = cuboidTransform(m, shift);
    for (int i = 0; i < 3; ++i) result->center[i] += shift[i];
    return true;
}
QJsonObject PointCloudCuboid::toJson() const {
    return {{"id", id},
            {"class_id", classId},
            {"center", array(center)},
            {"size", array(size)},
            {"rotation", array(rotation)},
            {"occluded", occluded},
            {"locked", locked}};
}
bool PointCloudCuboid::fromJson(const QJsonObject& j,
                                PointCloudCuboid* result,
                                QString* error) {
    if (!result) return false;
    const QSet<QString> keys{"id",       "class_id", "center", "size",
                             "rotation", "occluded", "locked"};
    for (auto it = j.begin(); it != j.end(); ++it)
        if (!keys.contains(it.key()))
            return fail(error, QStringLiteral("Invalid cuboid fields."));
    PointCloudCuboid b;
    for (const auto& key : {QStringLiteral("id"), QStringLiteral("class_id")}) {
        const auto v = j.value(key);
        if (!v.isDouble() || v.toDouble() < 1 || v.toDouble() > 65535 ||
            std::floor(v.toDouble()) != v.toDouble())
            return fail(error,
                        QStringLiteral(
                                "Cuboid IDs and classes must be in 1-65535."));
    }
    b.id = j["id"].toInt();
    b.classId = j["class_id"].toInt();
    if (!vector(j["center"], &b.center) || !vector(j["size"], &b.size) ||
        (j.contains("rotation") && !vector(j["rotation"], &b.rotation)))
        return fail(
                error,
                QStringLiteral("Cuboid requires three finite coordinates."));
    for (const auto& key :
         {QStringLiteral("occluded"), QStringLiteral("locked")})
        if (j.contains(key) && !j[key].isBool())
            return fail(error, QStringLiteral("Cuboid flags must be boolean."));
    b.occluded = j["occluded"].toBool();
    b.locked = j["locked"].toBool();
    if (!b.valid(error)) return false;
    *result = b;
    return true;
}
QString pointCloudCuboidPath(const QString& labelPath) {
    const QFileInfo info(labelPath);
    return info.dir().filePath(info.completeBaseName() +
                               QStringLiteral(".cuboids.json"));
}
bool loadPointCloudCuboids(const QString& path,
                           const QString& source,
                           QVector<PointCloudCuboid>* boxes,
                           QString* error) {
    if (!boxes) return false;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return fail(error, file.errorString());
    QJsonParseError parse;
    const auto doc = QJsonDocument::fromJson(file.readAll(), &parse);
    const auto j = doc.object();
    if (parse.error != QJsonParseError::NoError || !doc.isObject() ||
        !j["version"].isDouble() || j["version"].toDouble() != 1 ||
        j["type"] != QJsonValue("pointcloud-cuboids") ||
        j["point_cloud"] != QJsonValue(QFileInfo(source).fileName()) ||
        !j["cuboids"].isArray())
        return fail(
                error,
                QStringLiteral(
                        "Expected version 1 cuboids for this point cloud."));
    QVector<PointCloudCuboid> loaded;
    QSet<int> ids;
    for (const auto& value : j["cuboids"].toArray()) {
        PointCloudCuboid b;
        if (!value.isObject() ||
            !PointCloudCuboid::fromJson(value.toObject(), &b, error))
            return false;
        if (ids.contains(b.id))
            return fail(error, QStringLiteral("Duplicate cuboid IDs."));
        ids.insert(b.id);
        loaded.append(b);
    }
    *boxes = std::move(loaded);
    return true;
}
bool savePointCloudCuboids(const QString& path,
                           const QString& source,
                           const QVector<PointCloudCuboid>& boxes,
                           QString* error) {
    const QFileInfo target(path), input(source);
    if (!path.endsWith(QStringLiteral(".cuboids.json")) ||
        target.absoluteFilePath() == input.absoluteFilePath() ||
        (!target.canonicalFilePath().isEmpty() &&
         target.canonicalFilePath() == input.canonicalFilePath()))
        return fail(error, QStringLiteral("Invalid cuboid output path."));
    QJsonArray entries;
    QSet<int> ids;
    for (const auto& b : boxes) {
        if (!b.valid(error)) return false;
        if (ids.contains(b.id))
            return fail(error, QStringLiteral("Duplicate cuboid IDs."));
        ids.insert(b.id);
        entries.append(b.toJson());
    }
    const QJsonObject j{{"version", 1},
                        {"type", "pointcloud-cuboids"},
                        {"point_cloud", input.fileName()},
                        {"cuboids", entries}};
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return fail(error, file.errorString());
    const auto bytes = QJsonDocument(j).toJson();
    if (file.write(bytes) != bytes.size() || !file.commit())
        return fail(error, file.errorString());
    return true;
}
