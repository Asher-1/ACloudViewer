// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QVector3D>
#include <QVector>

// Cuboid annotation model, ported from the upstream pointcloud/cuboid.py.
// A cuboid is an oriented box over a point cloud frame, persisted as a
// `<label>.cuboids.json` sidecar with the strict upstream JSON contract
// (version 1, type "pointcloud-cuboids", per-frame point_cloud name).

struct PointCloudCuboid {
    int id = 0;       // 1..65535, unique per frame
    int classId = 0;  // 1..65535
    double center[3] = {0.0, 0.0, 0.0};
    double size[3] = {0.0, 0.0, 0.0};
    // XYZ Euler angles in radians, matching the upstream rotation_matrix
    // convention (see pointCloudCuboidRotation).
    double rotation[3] = {0.0, 0.0, 0.0};
    bool occluded = false;
    bool locked = false;

    /** Upstream Cuboid.__post_init__ parity: IDs/classes in 1..65535,
     *  finite center/size/rotation, size at least 0.01 per axis. */
    bool valid(QString* error = nullptr) const;

    /** The 8 corners in the upstream CORNER_SIGNS order (index bit 2 = z
     *  sign, bit 1 = y sign, bit 0 = x sign), i.e. exactly what
     *  projectPointCloudCuboid expects. */
    QVector<QVector3D> corners() const;

    bool operator==(const PointCloudCuboid& other) const;
};

/** Upstream rotation_matrix(angles): XYZ Euler angles to the row-major 3x3
 *  rotation used by corners(). */
void pointCloudCuboidRotation(const double angles[3], double rotation[9]);

/** Load cuboids from a .cuboids.json sidecar (strict upstream
 *  load_cuboids parity: version/type/point_cloud checks, per-entry field
 *  validation, duplicate-ID rejection). */
bool loadPointCloudCuboids(const QString& path,
                           const QString& pointCloudName,
                           QVector<PointCloudCuboid>* cuboids,
                           QString* error);

/** Write cuboids to a .cuboids.json sidecar (upstream save_cuboids
 *  parity: rejects a non-.cuboids.json target and duplicate IDs). */
bool savePointCloudCuboids(const QString& path,
                           const QString& pointCloudName,
                           const QVector<PointCloudCuboid>& cuboids,
                           QString* error);
