// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QHash>
#include <QJsonObject>
#include <QString>
#include <QVector4D>
#include <QVector>
#include <array>

using CuboidVector = std::array<double, 3>;
using CuboidMatrix = std::array<CuboidVector, 3>;
CuboidMatrix cuboidRotationMatrix(const CuboidVector& angles);
CuboidVector cuboidRotationAngles(const CuboidMatrix& matrix);
CuboidVector cuboidTransform(const CuboidMatrix& matrix,
                             const CuboidVector& point,
                             bool inverse = false);

struct PointCloudCuboid {
    int id = 1;
    int classId = 1;
    CuboidVector center{{0, 0, 0}};
    CuboidVector size{{4.5, 1.8, 1.6}};
    CuboidVector rotation{{0, 0, 0}};
    bool occluded = false;
    bool locked = false;

    bool valid(QString* error = nullptr) const;
    bool operator==(const PointCloudCuboid& other) const;
    bool operator!=(const PointCloudCuboid& other) const {
        return !(*this == other);
    }
    std::array<CuboidVector, 8> corners() const;
    bool contains(const CuboidVector& point) const;
    PointCloudCuboid resized(const std::array<int, 3>& signs,
                             const CuboidVector& delta) const;
    PointCloudCuboid rotated(int axis, double angle) const;
    bool fitted(const QVector<QVector4D>& points,
                PointCloudCuboid* result,
                QString* error = nullptr,
                bool insideOnly = true) const;
    QJsonObject toJson() const;
    static bool fromJson(const QJsonObject& json,
                         PointCloudCuboid* result,
                         QString* error);
};

struct PointCloudCuboidMesh {
    // Interleaved float32 x,y,z,r,g,b,a; triangles precede line vertices.
    QVector<float> vertices;
    int faceVertices = 0;
};
bool pointCloudCuboidMesh(const QVector<PointCloudCuboid>& boxes,
                          int selected,
                          const QHash<quint16, QColor>& colors,
                          const PointCloudCuboid* preview,
                          bool orthographic,
                          PointCloudCuboidMesh* output,
                          QString* error = nullptr);

QString pointCloudCuboidPath(const QString& labelPath);
bool loadPointCloudCuboids(const QString& path,
                           const QString& source,
                           QVector<PointCloudCuboid>* boxes,
                           QString* error);
bool savePointCloudCuboids(const QString& path,
                           const QString& source,
                           const QVector<PointCloudCuboid>& boxes,
                           QString* error);
