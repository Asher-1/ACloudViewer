// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QHash>
#include <QLineF>
#include <QPoint>
#include <QPointF>
#include <QSize>
#include <QString>
#include <QVector4D>
#include <QVector>
#include <array>

#include "PointCloudCuboid.h"

// Camera calibration contract shared by the point-cloud overlay and tests.
// It mirrors X-AnyLabeling's schema_version=1 pinhole calibration without
// exposing OpenCV or a third-party matrix type to the plugin API.
struct PointCloudCalibration {
    QSize imageSize;
    std::array<double, 9> cameraMatrix{};
    std::array<double, 16> pointCloudToCamera{};
    QString distortionModel = QStringLiteral("none");
    std::array<double, 5> distortion{};
};

// A camera source follows the upstream frame contract: image basenames are
// matched to point-cloud basenames and calibration is optional.
struct PointCloudCameraSource {
    QString directory;
    QHash<QString, QString> imagePaths;
    QString calibrationPath;
    PointCloudCalibration calibration;
    bool hasCalibration = false;
};

bool listPointCloudCameraImages(const QString& directory,
                                QHash<QString, QString>* imagePaths,
                                QString* error);

bool loadPointCloudCameraSource(const QString& directory,
                                const QString& calibrationPath,
                                PointCloudCameraSource* source,
                                QString* error);

bool loadPointCloudCalibration(const QString& path,
                               PointCloudCalibration* calibration,
                               QString* error);

// Returns visible, in-image points ordered from farthest to nearest, matching
// the upstream stable depth ordering used for the camera overlay.
bool projectPointCloud(const QVector<QVector4D>& points,
                       const PointCloudCalibration& calibration,
                       QVector<int>* indices,
                       QVector<QPoint>* pixels,
                       QString* error = nullptr);

// Projects the 12 cuboid edges into image space. Edges crossing the near
// plane or image boundary are clipped so callers can draw the result safely.
bool projectCuboid(const PointCloudCuboid& cuboid,
                   const PointCloudCalibration& calibration,
                   QVector<QLineF>* segments,
                   QString* error = nullptr);

// Upstream camera-overlay palette: clip to this frame's 2nd/98th depth
// percentiles, then interpolate five RGB stops. Output follows index order.
bool pointCloudCameraDepthColors(const QVector<QVector4D>& points,
                                 const PointCloudCalibration& calibration,
                                 const QVector<int>& indices,
                                 QVector<QColor>* colors,
                                 QString* error = nullptr,
                                 double* nearDepth = nullptr,
                                 double* farDepth = nullptr);
