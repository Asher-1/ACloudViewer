// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>
#include <QTemporaryDir>
#include <cstdio>
#include <limits>

#include "PointCloudCamera.h"
#include "PointCloudCuboid.h"
#include "PointCloudViewTransform.h"

#define REQUIRE(x)                                               \
    do {                                                         \
        if (!(x)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
            return 1;                                            \
        }                                                        \
    } while (false)

namespace {
QJsonArray array(std::initializer_list<double> values) {
    QJsonArray result;
    for (const double value : values) result.append(value);
    return result;
}

QJsonObject calibrationJson(
        const QString& distortionModel = QStringLiteral("none")) {
    QJsonArray distortion;
    if (distortionModel == QStringLiteral("opencv5"))
        distortion = array({0.04, -0.01, 0.001, -0.002, 0.0005});
    return QJsonObject{
            {"schema_version", 1},
            {"image_size", QJsonArray{100, 80}},
            {"camera_model", "pinhole"},
            {"camera_matrix", QJsonArray{array({50, 0, 50}), array({0, 50, 40}),
                                         array({0, 0, 1})}},
            {"T_pointcloud_to_camera",
             QJsonArray{array({1, 0, 0, 0}), array({0, 1, 0, 0}),
                        array({0, 0, 1, 0}), array({0, 0, 0, 1})}},
            {"distortion_model", distortionModel},
            {"distortion_coefficients", distortion}};
}

bool writeJson(const QString& path, const QJsonObject& object) {
    QFile file(path);
    return file.open(QIODevice::WriteOnly) &&
           file.write(QJsonDocument(object).toJson()) >= 0;
}

QJsonArray point(const QPointF& p) { return {p.x(), p.y()}; }

QJsonObject probe(const PointCloudCalibration& calibration,
                  const QVector<PointCloudCuboid>& cuboids) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const QVector<QVector4D> points{
            {0, 0, 5, 1}, {0, 0, 10, 1},    {1, 0, 5, 1},  {-1, 0, 5, 1},
            {0, 1, 5, 1}, {100, 100, 1, 1}, {0, 0, -1, 1}, {nan, 0, 5, 1}};
    QVector<int> indices;
    QVector<QPoint> pixels;
    QString error;
    QJsonArray pointIndices;
    QJsonArray pointPixels;
    QJsonArray depthColors;
    double nearDepth = 0, farDepth = 0;
    if (projectPointCloud(points, calibration, &indices, &pixels, &error)) {
        for (const int index : indices) pointIndices.append(index);
        for (const QPoint& pixel : pixels)
            pointPixels.append(QJsonArray{pixel.x(), pixel.y()});
        QVector<QColor> colors;
        if (pointCloudCameraDepthColors(points, calibration, indices, &colors,
                                        &error, &nearDepth, &farDepth))
            for (const auto& color : colors)
                depthColors.append(
                        QJsonArray{color.red(), color.green(), color.blue()});
    }
    QJsonArray projectedCuboids;
    for (const PointCloudCuboid& cuboid : cuboids) {
        QVector<QLineF> segments;
        QJsonArray lines;
        if (projectCuboid(cuboid, calibration, &segments, &error)) {
            for (const QLineF& line : segments)
                lines.append(QJsonArray{point(line.p1()), point(line.p2())});
        }
        projectedCuboids.append(
                QJsonObject{{"count", lines.size()}, {"segments", lines}});
    }
    return QJsonObject{{"indices", pointIndices},
                       {"pixels", pointPixels},
                       {"depth_colors", depthColors},
                       {"depth_range", QJsonArray{nearDepth, farDepth}},
                       {"cuboids", projectedCuboids}};
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QString error;
    if (argc == 3 &&
        QString::fromLocal8Bit(argv[1]) == QStringLiteral("--viewport-probe")) {
        QFile input(QString::fromLocal8Bit(argv[2]));
        REQUIRE(input.open(QIODevice::ReadOnly));
        const auto cases = QJsonDocument::fromJson(input.readAll()).array();
        QJsonArray results;
        for (const auto& entry : cases) {
            const auto item = entry.toObject();
            QVector<QVector4D> points;
            for (const auto& value : item["points"].toArray()) {
                const auto p = value.toArray();
                points.append(QVector4D(float(p[0].toDouble()),
                                        float(p[1].toDouble()),
                                        float(p[2].toDouble()), 1));
            }
            const auto dimensions = item["size"].toArray();
            const QSize size(dimensions[0].toInt(), dimensions[1].toInt());
            PointCloudViewTransform state;
            const auto limits = item["limits"].toArray();
            if (!limits.isEmpty()) {
                state.minimumScale = limits[0].toDouble();
                state.maximumScale = limits[1].toDouble();
            }
            state.perspective = item["perspective"].toBool();
            state.fit(points, size);
            if (item.contains("view")) state.setView(item["view"].toInt());
            const auto pan = item["pan"].toArray();
            if (!pan.isEmpty())
                state.pan(pan[0].toDouble(), pan[1].toDouble(), size.height());
            const auto orbit = item["orbit"].toArray();
            if (!orbit.isEmpty())
                state.orbit(orbit[0].toDouble(), orbit[1].toDouble());
            if (item.contains("wheel")) state.zoom(item["wheel"].toDouble());
            for (const auto& key : item["camera_keys"].toArray())
                if (!points.isEmpty())
                    REQUIRE(state.moveCamera(key.toString()));
            QVector<PointCloudCuboid> boxes;
            for (const auto& value : item["boxes"].toArray()) {
                PointCloudCuboid box;
                REQUIRE(PointCloudCuboid::fromJson(value.toObject(), &box,
                                                   &error));
                boxes.append(box);
            }
            const int focused = item["focus"].toInt();
            for (const auto& box : boxes)
                if (box.id == focused) state.focus(box, size);
            QJsonArray hits;
            for (const auto& value : item["hits"].toArray()) {
                const auto p = value.toArray();
                hits.append(state.hit(QPointF(p[0].toDouble(), p[1].toDouble()),
                                      size, boxes));
            }
            QVector<int> visible;
            for (int i = 0; i < points.size(); ++i)
                if (!item.contains("visible") ||
                    item["visible"].toArray().contains(i))
                    visible.append(i);
            QJsonArray previews;
            for (const auto& value : item["previews"].toArray()) {
                const auto p = value.toArray();
                PointCloudCuboid box;
                if (state.creationPreview(
                            QPointF(p[0].toDouble(), p[1].toDouble()), size,
                            points, visible, &box))
                    previews.append(array(
                            {box.center[0], box.center[1], box.center[2]}));
                else
                    previews.append(QJsonValue::Null);
            }
            const auto transform = state.matrix(size);
            QJsonArray rows;
            for (int row = 0; row < 4; ++row)
                rows.append(array({transform(row, 0), transform(row, 1),
                                   transform(row, 2), transform(row, 3)}));
            results.append(QJsonObject{
                    {"matrix", rows},
                    {"center", array({state.center[0], state.center[1],
                                      state.center[2]})},
                    {"scale", state.scale},
                    {"extent", state.extent},
                    {"yaw", state.yaw},
                    {"pitch", state.pitch},
                    {"hits", hits},
                    {"previews", previews}});
        }
        std::puts(QJsonDocument(results)
                          .toJson(QJsonDocument::Compact)
                          .constData());
        return 0;
    }
    // Z-up axis presets retain their exact 90-degree top direction, including
    // the degenerate single-point fit and portrait-window scale correction.
    PointCloudViewTransform view;
    view.fit({QVector4D(2, 3, 4, 1)}, QSize(100, 200));
    REQUIRE(std::abs(view.scale - 0.22) < 1e-12);
    view.setView(0);
    const auto top = view.matrix(QSize(100, 200));
    REQUIRE(std::abs(top(0, 0) - 1 / 0.11) < 1e-6);
    REQUIRE(std::abs(top(1, 1) - 1 / 0.22) < 1e-6);
    REQUIRE(std::abs(top(1, 2)) < 1e-12);
    const auto previous = view.scale;
    view.zoom(120);
    REQUIRE(view.scale < previous);
    view.orbit(0, 1000);
    REQUIRE(view.pitch == 90);
    if (argc == 4 &&
        QString::fromLocal8Bit(argv[1]) == QStringLiteral("--probe")) {
        PointCloudCalibration calibration;
        REQUIRE(loadPointCloudCalibration(QString::fromLocal8Bit(argv[2]),
                                          &calibration, &error));
        QFile sidecar(QString::fromLocal8Bit(argv[3]));
        REQUIRE(sidecar.open(QIODevice::ReadOnly));
        const QJsonDocument document =
                QJsonDocument::fromJson(sidecar.readAll());
        REQUIRE(document.isObject());
        QVector<PointCloudCuboid> cuboids;
        const QJsonArray entries =
                document.object()[QStringLiteral("cuboids")].toArray();
        for (const QJsonValue& value : entries) {
            PointCloudCuboid cuboid;
            REQUIRE(PointCloudCuboid::fromJson(value.toObject(), &cuboid,
                                               &error));
            cuboids.append(cuboid);
        }
        std::puts(QJsonDocument(probe(calibration, cuboids))
                          .toJson(QJsonDocument::Compact)
                          .constData());
        return 0;
    }

    QTemporaryDir directory;
    REQUIRE(directory.isValid());
    const QString calibrationPath =
            directory.filePath(QStringLiteral("calibration.json"));
    REQUIRE(writeJson(calibrationPath, calibrationJson()));
    PointCloudCalibration calibration;
    REQUIRE(loadPointCloudCalibration(calibrationPath, &calibration, &error));
    REQUIRE(calibration.imageSize == QSize(100, 80));

    QVector<QVector4D> points{{0, 0, 5, 0},  {0, 0, 10, 0}, {1, 0, 5, 0},
                              {-1, 0, 5, 0}, {0, 1, 5, 0},  {100, 100, 1, 0},
                              {0, 0, -1, 0}};
    QVector<int> indices;
    QVector<QPoint> pixels;
    REQUIRE(projectPointCloud(points, calibration, &indices, &pixels, &error));
    REQUIRE(indices == QVector<int>({1, 0, 2, 3, 4}));
    REQUIRE(pixels ==
            QVector<QPoint>({QPoint(50, 40), QPoint(50, 40), QPoint(60, 40),
                             QPoint(40, 40), QPoint(50, 50)}));

    PointCloudCuboid cuboid;
    cuboid.center = {{0, 0, 5}};
    cuboid.size = {{2, 2, 2}};
    QVector<QLineF> segments;
    REQUIRE(projectCuboid(cuboid, calibration, &segments, &error));
    REQUIRE(segments.size() == 12);
    for (const QLineF& line : segments) {
        REQUIRE(line.x1() >= 0 && line.x1() <= 99 && line.y1() >= 0 &&
                line.y1() <= 79);
        REQUIRE(line.x2() >= 0 && line.x2() <= 99 && line.y2() >= 0 &&
                line.y2() <= 79);
    }

    const QString distortedPath =
            directory.filePath(QStringLiteral("distorted.json"));
    REQUIRE(writeJson(distortedPath,
                      calibrationJson(QStringLiteral("opencv5"))));
    PointCloudCalibration distorted;
    REQUIRE(loadPointCloudCalibration(distortedPath, &distorted, &error));
    REQUIRE(projectCuboid(cuboid, distorted, &segments, &error));
    REQUIRE(segments.size() == 12 * 32);

    QJsonObject invalid = calibrationJson();
    invalid[QStringLiteral("camera_model")] = QStringLiteral("fisheye");
    REQUIRE(writeJson(directory.filePath(QStringLiteral("invalid.json")),
                      invalid));
    REQUIRE(!loadPointCloudCalibration(
            directory.filePath(QStringLiteral("invalid.json")), &calibration,
            &error));

    const QString imageDirectory = directory.filePath(QStringLiteral("images"));
    REQUIRE(QDir().mkpath(imageDirectory));
    REQUIRE(QImage(100, 80, QImage::Format_RGB32)
                    .save(QDir(imageDirectory)
                                  .filePath(QStringLiteral("000001.png"))));
    REQUIRE(QImage(100, 80, QImage::Format_RGB32)
                    .save(QDir(imageDirectory)
                                  .filePath(QStringLiteral("000002.jpg"))));
    QHash<QString, QString> imagePaths;
    REQUIRE(listPointCloudCameraImages(imageDirectory, &imagePaths, &error));
    const QSet<QString> expectedStems = QSet<QString>()
                                        << QStringLiteral("000001")
                                        << QStringLiteral("000002");
    REQUIRE(imagePaths.keys().toSet() == expectedStems);
    PointCloudCameraSource source;
    REQUIRE(loadPointCloudCameraSource(imageDirectory, calibrationPath, &source,
                                       &error));
    REQUIRE(source.hasCalibration && source.imagePaths == imagePaths);
    const QString duplicateDirectory =
            directory.filePath(QStringLiteral("duplicate-images"));
    REQUIRE(QDir().mkpath(duplicateDirectory));
    REQUIRE(QImage(2, 2, QImage::Format_RGB32)
                    .save(QDir(duplicateDirectory)
                                  .filePath(QStringLiteral("frame.png"))));
    REQUIRE(QImage(2, 2, QImage::Format_RGB32)
                    .save(QDir(duplicateDirectory)
                                  .filePath(QStringLiteral("frame.jpg"))));
    REQUIRE(!listPointCloudCameraImages(duplicateDirectory, &imagePaths,
                                        &error));
    REQUIRE(error.contains(QStringLiteral("same basename")));
    std::puts(
            "PASS point-cloud camera calibration, projection, clipping and "
            "distortion sampling");
    return 0;
}
