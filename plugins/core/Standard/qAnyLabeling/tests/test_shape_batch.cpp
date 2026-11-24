// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <cstdio>

#include "AnnotationShapeBatch.h"

namespace {
bool save(const QString& path, const QJsonObject& object) {
    QFile file(path);
    const auto bytes = QJsonDocument(object).toJson();
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}
QJsonObject read(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
}  // namespace

class AnnotationShapeBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(work.isValid());
        root = QDir(work.path());
        ASSERT_TRUE(root.mkdir("images"));
        ASSERT_TRUE(root.mkdir("labels"));
        pixels = QImage(80, 60, QImage::Format_RGB32);
        pixels.fill(Qt::white);
        for (int i = 0; i < 3; ++i) {
            images.append(
                    root.filePath(QStringLiteral("images/%1.png").arg(i)));
            ASSERT_TRUE(pixels.save(images.last()));
        }
        labels = root.filePath("labels");
        a = root.filePath("labels/0.json");
        b = root.filePath("labels/1.json");
        shape = {{"label", "box"},
                 {"shape_type", "rectangle"},
                 {"points", QJsonArray{QJsonArray{0, 0}, QJsonArray{80, 60}}},
                 {"custom", 17}};
        selected = {shape};
    }
    QTemporaryDir work;
    QDir root;
    QStringList images;
    QImage pixels;
    QString labels, a, b;
    QJsonObject shape;
    QJsonArray selected;
};

TEST_F(AnnotationShapeBatchTest, CancellationStopsBeforeNextImage) {
    const auto result = applyAnnotationShapeBatch(
            images, labels, 1, 3, AnnotationShapeBatchOperation::AddSelected,
            selected, [](int done, int) { return done < 1; });
    EXPECT_TRUE(result.canceled);
    EXPECT_EQ(result.shapes, 1);
    EXPECT_TRUE(QFile::exists(a));
    EXPECT_FALSE(QFile::exists(b));
}

TEST_F(AnnotationShapeBatchTest,
       RemovalPreservesLockedShapeAndUnknownMetadata) {
    auto locked = shape;
    locked.insert("locked", true);
    QJsonObject document{{"unknown_document", 71},
                         {"shapes", QJsonArray{shape, locked}}};
    ASSERT_TRUE(save(a, document));
    const auto result = applyAnnotationShapeBatch(
            images, labels, 1, 1, AnnotationShapeBatchOperation::RemoveSelected,
            selected);
    const auto retained = read(a);
    EXPECT_EQ(result.shapes, 1);
    EXPECT_EQ(retained.value("unknown_document").toInt(), 71);
    EXPECT_EQ(retained.value("shapes").toArray(), QJsonArray{locked});
}

TEST_F(AnnotationShapeBatchTest, MalformedSidecarRemainsUntouched) {
    QFile malformed(b);
    ASSERT_TRUE(malformed.open(QIODevice::WriteOnly));
    ASSERT_EQ(malformed.write("{"), 1);
    malformed.close();
    const auto result = applyAnnotationShapeBatch(
            images, labels, 2, 2, AnnotationShapeBatchOperation::AddSelected,
            selected);
    EXPECT_EQ(result.errors.size(), 1);
    ASSERT_TRUE(malformed.open(QIODevice::ReadOnly));
    EXPECT_EQ(malformed.readAll(), QByteArray("{"));
}

TEST_F(AnnotationShapeBatchTest, TrashCollisionPreservesImageAndSidecar) {
    ASSERT_TRUE(save(a, QJsonObject{{"shapes", selected}}));
    ASSERT_TRUE(root.mkdir("_delete_"));
    ASSERT_TRUE(pixels.save(root.filePath("_delete_/0.png")));
    const auto result = applyAnnotationShapeBatch(
            images, labels, 1, 1, AnnotationShapeBatchOperation::DeleteImages,
            {});
    EXPECT_EQ(result.errors.size(), 1);
    EXPECT_TRUE(QFile::exists(images.first()));
    EXPECT_TRUE(QFile::exists(a));
}

TEST_F(AnnotationShapeBatchTest, InvalidRangeDoesNotMutateFiles) {
    ASSERT_TRUE(save(a, QJsonObject{{"shapes", selected}}));
    const auto result = applyAnnotationShapeBatch(
            images, labels, 2, 1,
            AnnotationShapeBatchOperation::DeleteAnnotations, {});
    EXPECT_FALSE(result.errors.isEmpty());
    EXPECT_TRUE(QFile::exists(a));
}

int main(int argc, char** argv) {
    QCoreApplication application(argc, argv);
    if (argc == 2 && !QString::fromLocal8Bit(argv[1]).startsWith("--gtest")) {
        const auto request = read(QString::fromLocal8Bit(argv[1]));
        QStringList images;
        for (const auto& image : request.value("images").toArray())
            images.append(image.toString());
        const auto result = applyAnnotationShapeBatch(
                images, request.value("output").toString(),
                request.value("first").toInt(), request.value("last").toInt(),
                static_cast<AnnotationShapeBatchOperation>(
                        request.value("operation").toInt()),
                request.value("selected").toArray());
        std::puts(
                QJsonDocument(QJsonObject{{"files", result.files},
                                          {"shapes", result.shapes},
                                          {"skipped", result.skipped},
                                          {"errors", QJsonArray::fromStringList(
                                                             result.errors)}})
                        .toJson(QJsonDocument::Compact)
                        .constData());
        return result.errors.isEmpty() ? 0 : 1;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
