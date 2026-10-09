// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Math-kernel tests for the camera-calibration sidecar (upstream
// pointcloud/camera.py parity): strict schema validation, pinhole
// projection, opencv5 distortion, near-plane clipping and image-rect
// segment clipping.

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <cmath>
#include <cstdio>

#include "PointCloudCamera.h"
#include "PointCloudCuboid.h"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}

QString writeCalibration(const QTemporaryDir& dir, const QByteArray& body) {
    const QString path = dir.filePath(QStringLiteral("calib.json"));
    QFile file(path);
    file.open(QIODevice::WriteOnly);
    file.write(body);
    return path;
}

QByteArray validBody() {
    return QStringLiteral(
                   "{"
                   "\"schema_version\":1,"
                   "\"image_size\":[200,100],"
                   "\"camera_model\":\"pinhole\","
                   "\"camera_matrix\":[[100,0,50],[0,100,50],[0,0,1]],"
                   "\"T_pointcloud_to_camera\":"
                   "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"
                   "\"distortion_model\":\"none\","
                   "\"distortion_coefficients\":[]"
                   "}")
            .toUtf8();
}

}  // namespace

int main() {
    QTemporaryDir dir;
    if (!dir.isValid()) return 1;

    // Valid calibration loads and carries the strict fields.
    PointCloudCalibration calibration;
    QString error;
    const QString path = writeCalibration(dir, validBody());
    expect(loadPointCloudCalibration(path, &calibration, &error),
           "valid calibration must load");
    expect(error.isEmpty(), "valid calibration must not report an error");
    expect(calibration.imageSize == QSize(200, 100),
           "image size must round-trip");
    expect(calibration.fx == 100.0 && calibration.cx == 50.0,
           "intrinsics must round-trip");

    // Schema rejections (upstream load_calibration parity).
    {
        QJsonObject body = QJsonDocument::fromJson(validBody()).object();
        body.remove(QStringLiteral("distortion_model"));
        const QString bad = writeCalibration(dir, QJsonDocument(body).toJson());
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "missing field must be rejected");
    }
    {
        QJsonObject body = QJsonDocument::fromJson(validBody()).object();
        body.insert(QStringLiteral("extra"), 1);
        const QString bad = writeCalibration(dir, QJsonDocument(body).toJson());
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "unknown field must be rejected");
    }
    {
        QJsonObject body = QJsonDocument::fromJson(validBody()).object();
        body.insert(QStringLiteral("schema_version"), 2);
        const QString bad = writeCalibration(dir, QJsonDocument(body).toJson());
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "schema_version != 1 must be rejected");
    }
    {
        QJsonObject body = QJsonDocument::fromJson(validBody()).object();
        body.insert(QStringLiteral("camera_model"), QStringLiteral("fisheye"));
        const QString bad = writeCalibration(dir, QJsonDocument(body).toJson());
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "non-pinhole model must be rejected");
    }
    {
        // Non-rigid transform (scaled rotation) must be rejected.
        const QByteArray scaled =
                QByteArray(validBody())
                        .replace("[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
                                 "[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]");
        const QString bad = writeCalibration(dir, scaled);
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "non-rigid transform must be rejected");
    }
    {
        // Skewed intrinsic (non-zero off-diagonal) must be rejected.
        const QByteArray skewed =
                QByteArray(validBody())
                        .replace("[[100,0,50],[0,100,50],[0,0,1]]",
                                 "[[100,1,50],[0,100,50],[0,0,1]]");
        const QString bad = writeCalibration(dir, skewed);
        expect(!loadPointCloudCalibration(bad, &calibration, &error),
               "skewed intrinsic must be rejected");
    }

    // Identity calibration: the point (0,0,2) projects to the principal
    // point (50,50); the point behind the camera is dropped.
    {
        QVector<QVector3D> points{{0, 0, 2}, {0, 0, -1}};
        QVector<int> indices;
        QVector<QVector2D> pixels;
        QVector<double> depths;
        projectPointCloudPoints(points, calibration, &indices, &pixels,
                                &depths);
        expect(indices.size() == 1, "only the front point stays in bounds");
        if (indices.size() == 1) {
            expect(pixels.first() == QVector2D(50.0f, 50.0f),
                   "the front point must hit the principal point");
            expect(std::abs(depths.first() - 2.0) < 1e-9,
                   "depth must equal the camera-space z");
        }
    }

    // Sorting: far-to-near output order (upstream argsort of -z).
    {
        QVector<QVector3D> points{{0, 0, 1}, {0, 0, 3}, {0, 0, 2}};
        QVector<int> indices;
        QVector<QVector2D> pixels;
        QVector<double> depths;
        projectPointCloudPoints(points, calibration, &indices, &pixels,
                                &depths);
        expect(indices == (QVector<int>{1, 2, 0}),
               "points must sort far-to-near");
    }

    // opencv5 distortion pushes a center point outward radially.
    {
        PointCloudCalibration distorted = calibration;
        distorted.opencv5 = true;
        distorted.distortion[0] = 0.1;  // k1
        QVector<QVector3D> points{{0.1, 0, 1}};
        QVector<int> indices;
        QVector<QVector2D> pixels;
        QVector<double> depths;
        projectPointCloudPoints(points, distorted, &indices, &pixels, &depths);
        expect(indices.size() == 1, "distorted point stays in bounds");
        if (indices.size() == 1)
            expect(pixels.first().x() > 50.0f,
                   "positive k1 must push the point outward");
    }

    // Cuboid edges: 12 segments, near-plane and image-rect clipping.
    {
        QVector<QVector3D> corners;
        for (int corner = 0; corner < 8; ++corner) {
            const double sx = corner & 1 ? 1 : -1;
            const double sy = corner & 2 ? 1 : -1;
            const double sz = corner & 4 ? 1 : -1;
            corners.append(
                    QVector3D(float(sx), float(sy), float(sz * 0.1 + 3)));
        }
        expect(pointCloudCuboidEdges().size() == 12, "a cube has 12 edges");
        QVector<QPair<QVector2D, QVector2D>> segments;
        projectPointCloudCuboid(corners, calibration, &segments);
        expect(!segments.isEmpty(), "front cuboid must produce segments");
        for (const auto& segment : segments) {
            const bool inside =
                    segment.first.x() >= 0 && segment.first.x() <= 199 &&
                    segment.second.x() >= 0 && segment.second.x() <= 199 &&
                    segment.first.y() >= 0 && segment.first.y() <= 99 &&
                    segment.second.y() >= 0 && segment.second.y() <= 99;
            expect(inside, "segments must stay inside the image rect");
        }
        // A cuboid fully behind the camera produces nothing.
        QVector<QVector3D> behind;
        for (const QVector3D& corner : corners)
            behind.append(QVector3D(corner.x(), corner.y(), -5));
        projectPointCloudCuboid(behind, calibration, &segments);
        expect(segments.isEmpty(),
               "a cuboid behind the camera must not project");
    }

    // Overlay rendering: a point at (0,0,2) paints the principal-point
    // pixel with its semantic color (far-to-near painter order).
    {
        QImage base(200, 100, QImage::Format_RGB32);
        base.fill(Qt::black);
        QHash<quint16, QColor> colors;
        colors.insert(7, QColor(255, 0, 0));
        QVector<QVector3D> points{{0, 0, 2}, {0, 0, -1}};
        QVector<quint32> labels{7u, 7u};
        const QImage overlay = renderPointCloudCameraOverlay(
                base, calibration, points, labels, colors, 2);
        expect(overlay.pixelColor(50, 50) == QColor(255, 0, 0),
               "the projected point must paint the principal point pixel");
        expect(overlay.pixelColor(5, 5) == QColor(Qt::black),
               "unrelated pixels must stay untouched");
        // Occlusion: far-to-near draw order — the farther blue point is
        // painted first, the nearer red point covers it.
        QVector<QVector3D> occluded{{0, 0, 1}, {0, 0, 2}};
        QHash<quint16, QColor> two;
        two.insert(7, QColor(255, 0, 0));
        two.insert(3, QColor(0, 0, 255));
        QVector<quint32> mixed{7u, 3u};  // index 0 near red, index 1 far blue
        const QImage occludedImage = renderPointCloudCameraOverlay(
                base, calibration, occluded, mixed, two, 2);
        expect(occludedImage.pixelColor(50, 50) == QColor(255, 0, 0),
               "the nearer point must paint over the farther one");
    }

    // Cuboid model: axis-aligned corners, JSON round trip, validation.
    {
        PointCloudCuboid box;
        box.id = 3;
        box.classId = 5;
        box.center[0] = 1.0;
        box.center[1] = 2.0;
        box.center[2] = 3.0;
        box.size[0] = 2.0;
        box.size[1] = 4.0;
        box.size[2] = 6.0;
        const auto corners = box.corners();
        expect(corners.size() == 8, "a cuboid has 8 corners");
        // Upstream CORNER_SIGNS[0] = (-,-,-) and [7] = (-,+,+): with size
        // (2,4,6) around center (1,2,3) the corners span x=0..2, y=0..4,
        // z=0..6 with corner 7 on the -x side.
        expect(corners.at(0) == QVector3D(0.0f, 0.0f, 0.0f),
               "corner 0 is center - size/2 at identity rotation");
        expect(corners.at(7) == QVector3D(0.0f, 4.0f, 6.0f),
               "corner 7 carries the -x half-extent at identity rotation");
        expect(corners.at(6) == QVector3D(2.0f, 4.0f, 6.0f),
               "corner 6 is center + size/2 at identity rotation");
        // Rotation by yaw pi/2: the x/z extents swap through the matrix.
        PointCloudCuboid rotated = box;
        rotated.id = 4;
        rotated.rotation[1] = 3.14159265358979 / 2.0;
        const auto rotatedCorners = rotated.corners();
        const double extentX =
                rotatedCorners.at(6).x() - rotatedCorners.at(0).x();
        const double extentZ =
                rotatedCorners.at(6).z() - rotatedCorners.at(0).z();
        expect(std::abs(extentX + 6.0) < 1e-6,
               "a yaw quarter-turn maps size-z onto x (negated)");
        expect(std::abs(extentZ - 2.0) < 1e-6,
               "a yaw quarter-turn keeps size-x magnitude on z");
        // JSON sidecar round trip preserves every field.
        const QString sidecar = dir.filePath(QStringLiteral("f0.cuboids.json"));
        expect(savePointCloudCuboids(sidecar, QStringLiteral("frame0.bin"),
                                     {box, rotated}, &error),
               "cuboid sidecar must save");
        QVector<PointCloudCuboid> loaded;
        expect(loadPointCloudCuboids(sidecar, QStringLiteral("frame0.bin"),
                                     &loaded, &error),
               "cuboid sidecar must load");
        expect(loaded.size() == 2 && loaded.first() == box,
               "cuboid fields must round-trip");
        // Wrong point_cloud name must be rejected.
        QVector<PointCloudCuboid> ignored;
        expect(!loadPointCloudCuboids(sidecar, QStringLiteral("other.bin"),
                                      &ignored, &error),
               "a foreign point_cloud name must be rejected");
        // Size below the upstream minimum must be rejected.
        PointCloudCuboid tiny = box;
        tiny.size[1] = 0.001;
        expect(!tiny.valid(), "sub-minimum size must be rejected");
    }

    if (failures != 0) {
        std::fprintf(stderr, "test_pointcloud_camera: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_pointcloud_camera: ok\n");
    return 0;
}
