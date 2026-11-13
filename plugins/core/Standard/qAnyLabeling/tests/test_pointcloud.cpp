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
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtEndian>
#include <cstdio>
#include <cstring>

#include "PointCloudColors.h"
#include "PointCloudDocument.h"

static void appendFloat(QByteArray* data, float value) {
    quint32 bits;
    std::memcpy(&bits, &value, 4);
    char raw[4];
    qToLittleEndian<quint32>(bits, reinterpret_cast<uchar*>(raw));
    data->append(raw, 4);
}

static bool writeFile(const QString& path, const QByteArray& data) {
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(data) == data.size();
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc == 4 && QString::fromLocal8Bit(argv[1]) == "--colors-probe") {
        QFile input(QString::fromLocal8Bit(argv[2]));
        if (!input.open(QIODevice::ReadOnly)) return 1;
        QJsonArray results;
        for (const auto& value :
             QJsonDocument::fromJson(input.readAll()).array()) {
            const auto item = value.toObject();
            QHash<quint16, QColor> palette;
            const auto definitions = item["palette"].toObject();
            for (auto it = definitions.begin(); it != definitions.end(); ++it)
                palette.insert(quint16(it.key().toInt()),
                               QColor(it.value().toString()));
            QColor rgb;
            const auto channels = item["rgb"].toArray();
            if (!channels.isEmpty())
                rgb = QColor(channels[0].toInt(), channels[1].toInt(),
                             channels[2].toInt());
            const auto color = pointCloudDisplayColor(
                    quint32(item["label"].toDouble()),
                    float(item["intensity"].toDouble()),
                    channels.isEmpty() ? nullptr : &rgb,
                    PointCloudColorMode(item["mode"].toInt()), palette,
                    item["low"].toDouble(), item["high"].toDouble(),
                    item["selected"].toBool());
            results.append(QJsonArray{double(color.x()), double(color.y()),
                                      double(color.z())});
        }
        QFile output(QString::fromLocal8Bit(argv[3]));
        return output.open(QIODevice::WriteOnly) &&
                               output.write(QJsonDocument(results).toJson()) > 0
                       ? 0
                       : 1;
    }
    QTemporaryDir temporary;
    if (!temporary.isValid()) return 1;
    const QString binary = temporary.path() + QStringLiteral("/00001.bin");
    QByteArray points;
    for (float value : {1.0f, 2.0f, 3.0f, 0.2f, 4.0f, 5.0f, 6.0f, 0.8f})
        appendFloat(&points, value);
    if (!writeFile(binary, points)) return 1;
    PointCloudFrame frame;
    QString error;
    if (!loadPointCloudFrame(binary, &frame, &error) ||
        frame.points.size() != 2 || frame.points[1].z() != 6 ||
        frame.labels != QVector<quint32>{0, 0}) {
        std::fprintf(stderr, "BIN load failed: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    PointCloudDocument document;
    document.setFrame(frame);
    if (!document.assign({0, 1}, 12, 7) || !document.dirty() ||
        document.frame().labels != QVector<quint32>{0x0007000c, 0x0007000c} ||
        !document.undo() || document.dirty() || !document.redo())
        return 1;
    if (!savePointCloudLabels(document.frame(), &error)) return 1;
    document.markSaved();
    if (document.dirty() || !loadPointCloudFrame(binary, &frame, &error) ||
        frame.labels != QVector<quint32>{0x0007000c, 0x0007000c})
        return 1;
    if (document.assign({2}, 1, 0) || !document.assign({0}, 3, 0) ||
        !document.undo() || document.dirty())
        return 1;
    if (savePointCloudLabels(frame, &error, binary) || error.isEmpty())
        return 1;
    // A cancelled candidate must never overwrite a previously loaded frame.
    PointCloudFrame preserved = frame;
    if (loadPointCloudFrame(binary, &frame, &error, {}, [] { return true; }) ||
        !error.contains("cancelled") || frame.points != preserved.points ||
        frame.labels != preserved.labels || frame.path != preserved.path)
        return 1;
    const QString large = temporary.filePath("large.bin");
    QByteArray many;
    for (int i = 0; i < 8193; ++i) many.append(points.left(16));
    if (!writeFile(large, many)) return 1;
    int checks = 0;
    if (loadPointCloudFrame(large, &frame, &error, {},
                            [&] { return ++checks == 3; }) ||
        checks != 3 || !error.contains("cancelled") ||
        frame.path != preserved.path || frame.points != preserved.points)
        return 1;

    document.setFrame(frame);
    if (!document.operate({0, 1}, 12, 0, PointCloudDocument::AssignSemantic,
                          &error) ||
        !document.operate({0, 1}, 12, 0, PointCloudDocument::CreateInstance,
                          &error) ||
        document.frame().labels != QVector<quint32>{0x0001000c, 0x0001000c}) {
        if (!error.isEmpty())
            std::fprintf(stderr, "%s\n", error.toUtf8().constData());
        return 1;
    }
    if (!document.operate({0}, 12, 1, PointCloudDocument::RemoveFromInstance,
                          &error) ||
        document.frame().labels[0] != 12 ||
        !document.operate({0}, 12, 1, PointCloudDocument::AddToInstance,
                          &error) ||
        document.frame().labels[0] != 0x0001000c ||
        !document.operate({0}, 12, 1, PointCloudDocument::SplitInstance,
                          &error) ||
        document.frame().labels[0] != 0x0002000c)
        return 1;

    const QString ascii = temporary.path() + QStringLiteral("/frame.ply");
    const QByteArray ply =
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float "
            "x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar "
            "green\nproperty uchar blue\n"
            "element face 1\nproperty list uchar int "
            "vertex_indices\nend_header\n"
            "1 2 3 255 0 0\n4 5 6 0 255 0\n3 0 1 1\n";
    if (!writeFile(ascii, ply) || !loadPointCloudFrame(ascii, &frame, &error) ||
        frame.points.size() != 2 || frame.hasIntensity ||
        frame.rgb.size() != 2 || frame.rgb[0] != QColor(255, 0, 0)) {
        std::fprintf(stderr, "PLY load failed: %s\n",
                     error.toUtf8().constData());
        return 1;
    }
    preserved = frame;
    checks = 0;
    if (loadPointCloudFrame(ascii, &frame, &error, {},
                            [&] { return ++checks == 5; }) ||
        checks != 5 || !error.contains("cancelled") ||
        frame.points != preserved.points || frame.rgb != preserved.rgb)
        return 1;
    if (!writeFile(temporary.path() + QStringLiteral("/frame.label"),
                   QByteArray(4, '\0')) ||
        loadPointCloudFrame(ascii, &frame, &error) || error.isEmpty())
        return 1;
    return 0;
}
