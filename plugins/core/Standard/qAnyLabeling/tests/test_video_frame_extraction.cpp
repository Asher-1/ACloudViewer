// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "VideoFrameExtraction.h"
#include "VideoMaskArchive.h"

int main() {
    QTemporaryDir root;
    if (!root.isValid()) return 1;
    const QString source = root.filePath(QStringLiteral("source.avi"));
    cv::VideoWriter writer(source.toStdString(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0,
                           cv::Size(64, 64));
    if (!writer.isOpened()) return 77;
    for (int i = 0; i < 3; ++i)
        writer.write(cv::Mat(64, 64, CV_8UC3, cv::Scalar(i * 40, 80, 200)));
    writer.release();
    const QString archive = root.filePath(QStringLiteral("archive"));
    if (!QDir().mkpath(archive)) return 1;
    QImage mask(64, 64, QImage::Format_Grayscale8);
    mask.fill(0);
    for (int y = 10; y < 30; ++y)
        for (int x = 10; x < 30; ++x)
            if (x != 20 || y != 20) mask.scanLine(y)[x] = 255;
    if (!mask.save(QDir(archive).filePath(QStringLiteral("mask.png")), "PNG"))
        return 1;
    QJsonObject instance;
    instance.insert(QStringLiteral("id"), 7);
    instance.insert(QStringLiteral("mask"), QStringLiteral("mask.png"));
    QJsonObject frame;
    frame.insert(QStringLiteral("frame"), 1);
    frame.insert(QStringLiteral("instances"), QJsonArray{instance});
    QJsonObject manifest;
    manifest.insert(QStringLiteral("format"),
                    QStringLiteral("acloudviewer_sam3_tracking_v1"));
    manifest.insert(QStringLiteral("source"), source);
    manifest.insert(QStringLiteral("width"), 64);
    manifest.insert(QStringLiteral("height"), 64);
    manifest.insert(QStringLiteral("total_frames"), 3);
    manifest.insert(QStringLiteral("start_frame"), 1);
    manifest.insert(QStringLiteral("frames"), QJsonArray{frame});
    QString digestError;
    const QString sourceDigest = videoMaskArchiveSha256(source, &digestError);
    if (sourceDigest.isEmpty() || !digestError.isEmpty()) {
        std::fprintf(stderr, "source digest failed: %s\n",
                     digestError.toUtf8().constData());
        return 1;
    }
    manifest.insert(QStringLiteral("source_sha256"), sourceDigest);
    const QByteArray bytes = QJsonDocument(manifest).toJson();
    const QString destination = root.filePath(QStringLiteral("frames"));
    QString error;
    if (!extractSam3FrameWorkspace(bytes, archive, source, destination, {},
                                   &error)) {
        std::fprintf(stderr, "extract failed: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    for (int i = 0; i < 3; ++i) {
        const QString name =
                QStringLiteral("frame_%1").arg(i, 8, 10, QLatin1Char('0'));
        const QImage image(
                QDir(destination).filePath(name + QStringLiteral(".png")));
        if (image.size() != QSize(64, 64)) return 1;
        const bool annotated = QFile::exists(
                QDir(destination).filePath(name + QStringLiteral(".json")));
        if (annotated != (i == 1)) return 1;
    }
    QFile annotation(
            QDir(destination).filePath(QStringLiteral("frame_00000001.json")));
    if (!annotation.open(QIODevice::ReadOnly)) return 1;
    const QJsonArray shapes = QJsonDocument::fromJson(annotation.readAll())
                                      .object()
                                      .value(QStringLiteral("shapes"))
                                      .toArray();
    if (shapes.size() != 1) return 1;
    const QImage exact = QImage::fromData(
            QByteArray::fromBase64(
                    shapes.first()
                            .toObject()
                            .value(QStringLiteral("acloudviewer_mask_png"))
                            .toString()
                            .toLatin1()),
            "PNG");
    if (exact.isNull() ||
        shapes.first().toObject().value(QStringLiteral("group_id")).toInt() !=
                7 ||
        exact.pixelColor(20, 20).alpha() != 0 ||
        exact.pixelColor(15, 15).alpha() != 255)
        return 1;
    const QString cancelled = root.filePath(QStringLiteral("cancelled"));
    if (extractSam3FrameWorkspace(
                bytes, archive, source, cancelled,
                [](int index, int) { return index < 1; }, &error) ||
        QFileInfo::exists(cancelled))
        return 1;
    QJsonObject emptyFrame = frame;
    emptyFrame.insert(QStringLiteral("instances"), QJsonArray());
    QJsonObject emptyManifest = manifest;
    emptyManifest.insert(QStringLiteral("frames"), QJsonArray{emptyFrame});
    const QString emptyWorkspace = root.filePath(QStringLiteral("empty"));
    if (!extractSam3FrameWorkspace(QJsonDocument(emptyManifest).toJson(),
                                   archive, source, emptyWorkspace, {},
                                   &error) ||
        !QFileInfo::exists(
                QDir(emptyWorkspace)
                        .filePath(QStringLiteral("frame_00000000.png"))) ||
        QFileInfo::exists(
                QDir(emptyWorkspace)
                        .filePath(QStringLiteral("frame_00000001.json"))))
        return 1;
    QJsonObject lateFrame = emptyFrame;
    lateFrame.insert(QStringLiteral("frame"), 1);
    QJsonObject lateManifest = manifest;
    lateManifest.insert(QStringLiteral("frames"), QJsonArray{frame, lateFrame});
    const QString malformed = root.filePath(QStringLiteral("malformed"));
    if (extractSam3FrameWorkspace(QJsonDocument(lateManifest).toJson(), archive,
                                  source, malformed, {}, &error) ||
        QFileInfo::exists(malformed))
        return 1;
    // A replacement with the same dimensions and frame count must not receive
    // the archived masks. Rewriting the source path keeps the path/metadata
    // checks valid and exercises the content binding specifically.
    cv::VideoWriter replacement(source.toStdString(),
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                5.0, cv::Size(64, 64));
    if (!replacement.isOpened()) {
        std::fprintf(stderr, "replacement video encoder unavailable\n");
        return 77;
    }
    for (int i = 0; i < 3; ++i)
        replacement.write(
                cv::Mat(64, 64, CV_8UC3, cv::Scalar(200 - i * 30, 10, 10)));
    replacement.release();
    const QString replaced = root.filePath(QStringLiteral("replaced"));
    if (extractSam3FrameWorkspace(bytes, archive, source, replaced, {},
                                  &error) ||
        QFileInfo::exists(replaced) ||
        !error.contains(QStringLiteral("content"), Qt::CaseInsensitive)) {
        std::fprintf(stderr, "replacement not rejected: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    QJsonObject malformedDigest = manifest;
    malformedDigest.insert(QStringLiteral("source_sha256"),
                           QStringLiteral("bad"));
    const QString malformedDigestDir =
            root.filePath(QStringLiteral("malformed_digest"));
    if (extractSam3FrameWorkspace(QJsonDocument(malformedDigest).toJson(),
                                  archive, source, malformedDigestDir, {},
                                  &error) ||
        QFileInfo::exists(malformedDigestDir) ||
        !error.contains(QStringLiteral("SHA-256"), Qt::CaseInsensitive)) {
        std::fprintf(stderr, "malformed digest not rejected: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    // Legacy manifests without a digest stay readable when the source still
    // satisfies their structural checks.
    QJsonObject legacy = manifest;
    legacy.remove(QStringLiteral("source_sha256"));
    const QString legacyDir = root.filePath(QStringLiteral("legacy"));
    if (!extractSam3FrameWorkspace(QJsonDocument(legacy).toJson(), archive,
                                   source, legacyDir, {}, &error) ||
        !QFileInfo::exists(QDir(legacyDir).filePath(
                QStringLiteral("frame_00000000.png")))) {
        std::fprintf(stderr, "legacy manifest rejected: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    manifest.insert(QStringLiteral("total_frames"), 4);
    const QString mismatch = root.filePath(QStringLiteral("mismatch"));
    return extractSam3FrameWorkspace(QJsonDocument(manifest).toJson(), archive,
                                     source, mismatch, {}, &error) ||
                           QFileInfo::exists(mismatch)
                   ? 1
                   : 0;
}
