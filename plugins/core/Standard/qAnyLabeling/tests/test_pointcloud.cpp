// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

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

int runColorsProbe(int argc, char** argv) {
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
    return 0;
}

class PointCloudDocumentTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(temporary.isValid());
        binary = temporary.filePath("00001.bin");
        for (float value : {1.0f, 2.0f, 3.0f, 0.2f, 4.0f, 5.0f, 6.0f, 0.8f})
            appendFloat(&points, value);
        ASSERT_TRUE(writeFile(binary, points));
        ASSERT_TRUE(loadPointCloudFrame(binary, &frame, &error))
                << error.toStdString();
        ASSERT_EQ(frame.points.size(), 2);
    }
    QTemporaryDir temporary;
    QString binary, error;
    QByteArray points;
    PointCloudFrame frame;
};

TEST_F(PointCloudDocumentTest, BinLabelsRoundTripAndMixedUndo) {
    EXPECT_EQ(frame.points[1].z(), 6);
    EXPECT_EQ(frame.labels, (QVector<quint32>{0, 0}));
    PointCloudDocument document;
    document.setFrame(frame);
    ASSERT_TRUE(document.assign({0, 1}, 12, 7));
    EXPECT_TRUE(document.dirty());
    EXPECT_EQ(document.frame().labels,
              (QVector<quint32>{0x0007000c, 0x0007000c}));
    ASSERT_TRUE(document.undo());
    EXPECT_FALSE(document.dirty());
    ASSERT_TRUE(document.redo());
    ASSERT_TRUE(savePointCloudLabels(document.frame(), &error));
    document.markSaved();
    EXPECT_FALSE(document.dirty());
    ASSERT_TRUE(loadPointCloudFrame(binary, &frame, &error));
    EXPECT_EQ(frame.labels, (QVector<quint32>{0x0007000c, 0x0007000c}));
    EXPECT_FALSE(document.assign({2}, 1, 0));
    ASSERT_TRUE(document.assign({0}, 3, 0));
    ASSERT_TRUE(document.undo());
    EXPECT_FALSE(document.dirty());
    EXPECT_FALSE(savePointCloudLabels(frame, &error, binary));
    EXPECT_FALSE(error.isEmpty());
}

TEST_F(PointCloudDocumentTest, CancelledBinLoadPreservesExistingFrame) {
    const auto preserved = frame;
    EXPECT_FALSE(loadPointCloudFrame(binary, &frame, &error, {},
                                     [] { return true; }));
    EXPECT_TRUE(error.contains("cancelled"));
    EXPECT_EQ(frame.points, preserved.points);
    EXPECT_EQ(frame.labels, preserved.labels);
    EXPECT_EQ(frame.path, preserved.path);
    const auto large = temporary.filePath("large.bin");
    QByteArray many;
    for (int i = 0; i < 8193; ++i) many.append(points.left(16));
    ASSERT_TRUE(writeFile(large, many));
    int checks = 0;
    EXPECT_FALSE(loadPointCloudFrame(large, &frame, &error, {},
                                     [&] { return ++checks == 3; }));
    EXPECT_EQ(checks, 3);
    EXPECT_TRUE(error.contains("cancelled"));
    EXPECT_EQ(frame.path, preserved.path);
    EXPECT_EQ(frame.points, preserved.points);
}

TEST_F(PointCloudDocumentTest, SemanticAndInstanceOperationsRetainExactLabels) {
    PointCloudDocument document;
    document.setFrame(frame);
    ASSERT_TRUE(document.operate({0, 1}, 12, 0,
                                 PointCloudDocument::AssignSemantic, &error));
    ASSERT_TRUE(document.operate({0, 1}, 12, 0,
                                 PointCloudDocument::CreateInstance, &error));
    EXPECT_EQ(document.frame().labels,
              (QVector<quint32>{0x0001000c, 0x0001000c}));
    ASSERT_TRUE(document.operate(
            {0}, 12, 1, PointCloudDocument::RemoveFromInstance, &error));
    EXPECT_EQ(document.frame().labels[0], 12u);
    ASSERT_TRUE(document.operate({0}, 12, 1, PointCloudDocument::AddToInstance,
                                 &error));
    EXPECT_EQ(document.frame().labels[0], 0x0001000cu);
    ASSERT_TRUE(document.operate({0}, 12, 1, PointCloudDocument::SplitInstance,
                                 &error));
    EXPECT_EQ(document.frame().labels[0], 0x0002000cu);
}

TEST_F(PointCloudDocumentTest, PlyLoadCancellationAndLabelSizeValidation) {
    const auto ascii = temporary.filePath("frame.ply");
    const QByteArray ply =
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float "
            "x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar "
            "green\nproperty uchar blue\n"
            "element face 1\nproperty list uchar int "
            "vertex_indices\nend_header\n"
            "1 2 3 255 0 0\n4 5 6 0 255 0\n3 0 1 1\n";
    ASSERT_TRUE(writeFile(ascii, ply));
    ASSERT_TRUE(loadPointCloudFrame(ascii, &frame, &error))
            << error.toStdString();
    ASSERT_EQ(frame.points.size(), 2);
    EXPECT_FALSE(frame.hasIntensity);
    ASSERT_EQ(frame.rgb.size(), 2);
    EXPECT_EQ(frame.rgb[0], QColor(255, 0, 0));
    const auto preserved = frame;
    int checks = 0;
    EXPECT_FALSE(loadPointCloudFrame(ascii, &frame, &error, {},
                                     [&] { return ++checks == 5; }));
    EXPECT_EQ(checks, 5);
    EXPECT_TRUE(error.contains("cancelled"));
    EXPECT_EQ(frame.points, preserved.points);
    EXPECT_EQ(frame.rgb, preserved.rgb);
    ASSERT_TRUE(
            writeFile(temporary.filePath("frame.label"), QByteArray(4, '\0')));
    EXPECT_FALSE(loadPointCloudFrame(ascii, &frame, &error));
    EXPECT_FALSE(error.isEmpty());
}

int main(int argc, char** argv) {
    if (argc == 4 && QString::fromLocal8Bit(argv[1]) == "--colors-probe")
        return runColorsProbe(argc, argv);
    QCoreApplication application(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
