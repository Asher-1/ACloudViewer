// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QHash>
#include <QImage>
#include <QMatrix3x3>
#include <QSize>
#include <QString>
#include <QVector2D>
#include <QVector3D>
#include <QVector>

// Camera-calibration sidecar kernel, ported from the upstream
// pointcloud/camera.py contract. A calibration sidecar is a strict JSON
// document (schema_version 1, pinhole model, opencv5 or no distortion)
// describing how the annotated point cloud projects into one camera image:
//
// {
//   "schema_version": 1,
//   "image_size": [1280, 720],
//   "camera_model": "pinhole",
//   "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
//   "T_pointcloud_to_camera": <4x4 rigid transform>,
//   "distortion_model": "none" | "opencv5",
//   "distortion_coefficients": [] | [k1,k2,p1,p2,k3]
// }

struct PointCloudCalibration {
    QSize imageSize;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    // T_pointcloud_to_camera as [R|t]: R is row-major 3x3, t is the
    // translation column.
    double rotation[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double translation[3] = {0.0, 0.0, 0.0};
    bool opencv5 = false;
    double distortion[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // k1,k2,p1,p2,k3
};

/** Parse and fully validate a calibration sidecar (upstream
 *  load_calibration parity: missing/unknown fields, non-pinhole model,
 *  non-rigid transform, wrong matrix shapes and non-positive focal
 *  lengths are rejected). Returns false with *error set. */
bool loadPointCloudCalibration(const QString& path,
                               PointCloudCalibration* calibration,
                               QString* error);

/** Project 3D points into calibration's image: returns the point indices
 *  that fall inside the image bounds, their integer pixel coordinates and
 *  their camera-space depth, sorted far-to-near (upstream project_points
 *  parity — the caller draws them back-to-front). */
void projectPointCloudPoints(const QVector<QVector3D>& points,
                             const PointCloudCalibration& calibration,
                             QVector<int>* insideIndices,
                             QVector<QVector2D>* pixels,
                             QVector<double>* depths);

/** Project a cuboid (8 corners in the upstream CORNER_SIGNS order:
 *  index bit 2 = z sign, bit 1 = y sign, bit 0 = x sign) into the image
 *  and return the visible edge segments in pixel coordinates, clipped to
 *  the image rect (upstream project_cuboid parity: near-plane clipping at
 *  z=1e-6 with interpolation, 33-sample subdivision under opencv5
 *  distortion so curved edges stay smooth, Liang-Barsky-style segment
 *  clipping). An empty result means nothing is visible. */
void projectPointCloudCuboid(const QVector<QVector3D>& corners,
                             const PointCloudCalibration& calibration,
                             QVector<QPair<QVector2D, QVector2D>>* segments);

/** The 12 cube edges as index pairs into the corner list. */
QVector<QPair<int, int>> pointCloudCuboidEdges();

/** Render a camera image with the point cloud projected onto it (the
 *  qAnyLabeling camera panel): points that fall inside the image are drawn
 *  far-to-near as square pixels colored by their semantic class (labels
 *  encode semantic in the low 16 bits, mirroring PointCloudFrame). The
 *  returned image is a copy of \p image with the overlay applied. */
QImage renderPointCloudCameraOverlay(
        const QImage& image,
        const PointCloudCalibration& calibration,
        const QVector<QVector3D>& points,
        const QVector<quint32>& labels,
        const QHash<quint16, QColor>& semanticColors,
        int pointSize);
