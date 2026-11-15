// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "VideoMaskArchive.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir folder;
    if (!folder.isValid()) return 1;
    QFile source(folder.filePath(QStringLiteral("source.bin")));
    if (!source.open(QIODevice::WriteOnly) || source.write("abc", 3) != 3)
        return 1;
    source.close();
    QString digestError;
    if (videoMaskArchiveSha256(source.fileName(), &digestError) !=
                QStringLiteral("ba7816bf8f01cfea414140de5dae2223b00361a396177a9"
                               "cb410ff61f20015ad") ||
        !digestError.isEmpty() ||
        !videoMaskArchiveSha256(folder.filePath(QStringLiteral("missing")),
                                &digestError)
                 .isEmpty() ||
        digestError.isEmpty()) {
        std::fprintf(stderr, "source video digest failed\n");
        return 1;
    }

    SAM3WorkerResult result;
    result.valid = true;
    result.detCount = 2;
    result.instanceIds = {7, 19};
    result.scores = {0.9f, 0.8f};
    result.boxes = {{0, 0, 4, 4}, {1, 1, 3, 3}};
    QImage ring(5, 5, QImage::Format_Grayscale8);
    ring.fill(255);
    ring.setPixel(2, 2, 0);
    QImage dot(5, 5, QImage::Format_Grayscale8);
    dot.fill(0);
    dot.setPixel(2, 2, 255);
    result.instanceMasks = {ring, dot};

    QJsonObject frame;
    if (!writeVideoMaskFrame(folder.path(), 42, result, &frame) ||
        frame.value(QStringLiteral("frame")).toInt() != 42) {
        std::fprintf(stderr, "frame export failed\n");
        return 1;
    }
    const QJsonArray instances =
            frame.value(QStringLiteral("instances")).toArray();
    if (instances.size() != 2 ||
        instances.at(0).toObject().value(QStringLiteral("id")).toInt() != 7 ||
        instances.at(1).toObject().value(QStringLiteral("id")).toInt() != 19) {
        std::fprintf(stderr, "instance IDs were not preserved\n");
        return 1;
    }
    const QString firstFile =
            instances.at(0).toObject().value(QStringLiteral("mask")).toString();
    const QImage decoded(folder.filePath(firstFile));
    if (decoded.isNull() || qGray(decoded.pixel(2, 2)) != 0 ||
        qGray(decoded.pixel(0, 0)) != 255) {
        std::fprintf(stderr, "mask hole was not preserved\n");
        return 1;
    }
    result.instanceIds = {7, 7};
    QJsonObject repeated;
    if (!writeVideoMaskFrame(folder.path(), 43, result, &repeated)) return 1;
    const QJsonArray repeatedInstances =
            repeated.value(QStringLiteral("instances")).toArray();
    const QString secondFile = repeatedInstances.at(1)
                                       .toObject()
                                       .value(QStringLiteral("mask"))
                                       .toString();
    const QString repeatedFirstFile = repeatedInstances.at(0)
                                              .toObject()
                                              .value(QStringLiteral("mask"))
                                              .toString();
    if (repeatedFirstFile == secondFile ||
        QImage(folder.filePath(repeatedFirstFile)).isNull() ||
        QImage(folder.filePath(secondFile)).isNull()) {
        std::fprintf(stderr, "repeated IDs overwrote a mask\n");
        return 1;
    }
    const QJsonObject manifest{{QStringLiteral("complete"), false},
                               {QStringLiteral("frames"), QJsonArray{frame}}};
    if (!writeVideoMaskManifest(folder.path(), manifest)) return 1;
    QFile file(folder.filePath(QStringLiteral("manifest.json")));
    if (!file.open(QIODevice::ReadOnly) ||
        QJsonDocument::fromJson(file.readAll()).object() != manifest) {
        std::fprintf(stderr, "manifest round trip failed\n");
        return 1;
    }
    QJsonObject rejected{{QStringLiteral("stale"), true}};
    if (writeVideoMaskFrame(folder.filePath(QStringLiteral("missing")), 43,
                            result, &rejected) ||
        !rejected.isEmpty()) {
        std::fprintf(stderr, "invalid export directory was accepted\n");
        return 1;
    }
    return 0;
}
