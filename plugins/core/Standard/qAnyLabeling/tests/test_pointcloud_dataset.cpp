// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QFile>
#include <QProcess>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QtEndian>
#include <cstdio>

#include "PointCloudDatasetExporter.h"

namespace {
bool check(bool value, const QString& message) {
    if (!value) std::fprintf(stderr, "%s\n", qPrintable(message));
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir temporary;
    if (!check(temporary.isValid(),
               QStringLiteral("temporary folder unavailable")))
        return 1;

    PointCloudFrame frame;
    frame.path = temporary.filePath(QStringLiteral("frame_01.bin"));
    frame.points = {QVector4D(1.0f, 2.0f, 3.0f, 0.5f),
                    QVector4D(4.0f, 5.0f, 6.0f, 0.75f)};
    PointCloudCuboid cuboid;
    cuboid.id = 7;
    cuboid.classId = 1;
    cuboid.center = {{1.0, 2.0, 3.0}};
    cuboid.size = {{4.0, 2.0, 1.5}};
    cuboid.rotation = {{0.1, 0.2, 0.3}};
    cuboid.occluded = true;
    frame.cuboids = {cuboid};

    PointCloudDatasetExportOptions options;
    options.frames = {frame};
    options.classNames.insert(1, QStringLiteral("car"));
    options.classColors.insert(1, QColor(QStringLiteral("#123456")));
    const QString unzip =
            QStandardPaths::findExecutable(QStringLiteral("unzip"));
    for (const QString& format :
         {QStringLiteral("datumaro"), QStringLiteral("kitti_raw"),
          QStringLiteral("sly_pointcloud")}) {
        options.format = format;
        options.outputPath =
                temporary.filePath(format + QStringLiteral(".zip"));
        int progress = -1;
        QString error;
        if (!check(exportPointCloudDataset(
                           options,
                           [&](int current, int total) {
                               if (total != 1 || current < progress ||
                                   current > total)
                                   return false;
                               progress = current;
                               return true;
                           },
                           &error),
                   QStringLiteral("%1 export failed: %2").arg(format, error)) ||
            !check(progress == 1,
                   QStringLiteral("%1 export progress missing").arg(format)) ||
            !check(QFile::exists(options.outputPath),
                   QStringLiteral("%1 archive missing").arg(format)))
            return 1;
        if (!unzip.isEmpty() &&
            !check(QProcess::execute(unzip, {QStringLiteral("-tqq"),
                                             options.outputPath}) == 0,
                   QStringLiteral("%1 ZIP integrity failed").arg(format)))
            return 1;
        if (!unzip.isEmpty()) {
            QProcess listing;
            listing.start(unzip, {QStringLiteral("-l"), options.outputPath});
            if (!check(listing.waitForFinished() && listing.exitCode() == 0,
                       QStringLiteral("%1 ZIP listing failed").arg(format)))
                return 1;
            const QString contents =
                    QString::fromLocal8Bit(listing.readAllStandardOutput());
            const QStringList expected =
                    format == QStringLiteral("datumaro")
                            ? QStringList{QStringLiteral(
                                                  "annotations/default.json"),
                                          QStringLiteral("point_clouds/default/"
                                                         "frame_01.pcd")}
                    : format == QStringLiteral("kitti_raw")
                            ? QStringList{QStringLiteral("tracklet_labels.xml"),
                                          QStringLiteral("frame_list.txt"),
                                          QStringLiteral("velodyne_points/data/"
                                                         "frame_01.pcd")}
                            : QStringList{
                                      QStringLiteral("meta.json"),
                                      QStringLiteral("key_id_map.json"),
                                      QStringLiteral(
                                              "ds0/ann/frame_01.pcd.json"),
                                      QStringLiteral(
                                              "ds0/pointcloud/frame_01.pcd")};
            for (const QString& entry : expected) {
                if (!check(contents.contains(entry),
                           QStringLiteral("%1 ZIP missing %2")
                                   .arg(format, entry)))
                    return 1;
            }
        }
    }

    options.format = QStringLiteral("datumaro");
    options.outputPath = temporary.filePath(QStringLiteral("cancelled.zip"));
    QString error;
    if (!check(!exportPointCloudDataset(
                       options, [](int, int) { return false; }, &error),
               QStringLiteral("cancellation must fail")) ||
        !check(!QFile::exists(options.outputPath),
               QStringLiteral("cancelled archive was installed")))
        return 1;
    QFile previous(options.outputPath);
    const QByteArray original("previous archive must survive");
    if (!check(previous.open(QIODevice::WriteOnly) &&
                       previous.write(original) == original.size(),
               QStringLiteral("existing archive fixture failed")))
        return 1;
    previous.close();
    int callbacks = 0;
    if (!check(!exportPointCloudDataset(
                       options, [&](int, int) { return ++callbacks < 3; },
                       &error),
               QStringLiteral("ZIP writing cancellation must fail")) ||
        !check(previous.open(QIODevice::ReadOnly) &&
                       previous.readAll() == original,
               QStringLiteral("cancellation must preserve the previous ZIP")))
        return 1;
    previous.close();
    auto invalid = options;
    invalid.classNames.insert(2, QStringLiteral("car"));
    if (!check(!exportPointCloudDataset(invalid, {}, &error),
               QStringLiteral("duplicate class names must be rejected")))
        return 1;
    invalid = options;
    invalid.frames[0].cuboids.append(cuboid);
    if (!check(!exportPointCloudDataset(invalid, {}, &error),
               QStringLiteral("duplicate cuboid IDs must be rejected")))
        return 1;
    invalid = options;
    invalid.frames[0].path = options.outputPath;
    if (!check(!exportPointCloudDataset(invalid, {}, &error),
               QStringLiteral("source overwrite must be rejected")))
        return 1;
    if (!check(previous.open(QIODevice::ReadOnly) &&
                       previous.readAll() == original,
               QStringLiteral("invalid export must preserve the previous ZIP")))
        return 1;
    previous.close();
    // PLY permits intensity and RGB together. Both fields must survive PCD
    // export in the upstream order, with explicit little-endian records.
    options.frames[0].rgb = {QColor(1, 2, 3), QColor(4, 5, 6)};
    if (!check(exportPointCloudDataset(options, {}, &error),
               QStringLiteral("RGB/intensity export failed: %1").arg(error)))
        return 1;
    if (!unzip.isEmpty()) {
        QProcess payload;
        payload.start(unzip,
                      {QStringLiteral("-p"), options.outputPath,
                       QStringLiteral("point_clouds/default/frame_01.pcd")});
        if (!check(payload.waitForFinished() && payload.exitCode() == 0,
                   QStringLiteral("could not read RGB PCD payload")))
            return 1;
        const QByteArray pcd = payload.readAllStandardOutput();
        const QByteArray marker("DATA binary\n");
        const int start = pcd.indexOf(marker) + marker.size();
        if (!check(pcd.contains("FIELDS x y z intensity rgb\n") &&
                           pcd.contains("TYPE F F F F U\n") &&
                           start >= marker.size() && pcd.size() - start == 40 &&
                           qFromLittleEndian<quint32>(
                                   reinterpret_cast<const uchar*>(
                                           pcd.constData() + start)) ==
                                   0x3f800000u &&
                           qFromLittleEndian<quint32>(
                                   reinterpret_cast<const uchar*>(
                                           pcd.constData() + start + 12)) ==
                                   0x3f000000u &&
                           qFromLittleEndian<quint32>(
                                   reinterpret_cast<const uchar*>(
                                           pcd.constData() + start + 16)) ==
                                   0x010203u,
                   QStringLiteral(
                           "RGB/intensity PCD binary contract mismatch")))
            return 1;
    }
    std::puts("PASS point-cloud Datumaro/KITTI/Sly dataset export");
    return 0;
}
