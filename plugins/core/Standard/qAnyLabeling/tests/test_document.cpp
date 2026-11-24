// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <QBuffer>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTemporaryDir>
#include <cstdio>

#include "AnnotationDocument.h"

namespace {
bool write(const QString& path, const QJsonObject& json) {
    QFile file(path);
    const auto bytes = QJsonDocument(json).toJson();
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}
}  // namespace

class AnnotationDocumentTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(temporary.isValid());
        root = QDir(temporary.path());
        ASSERT_TRUE(root.mkdir("images"));
        ASSERT_TRUE(root.mkdir("labels"));
        imagePath = root.filePath("images/scene.png");
        labels = root.filePath("labels/scene.json");
        image = QImage(37, 19, QImage::Format_RGB32);
        image.fill(Qt::red);
        ASSERT_TRUE(image.save(imagePath));
        json = {{"shapes", QJsonArray()},
                {"imagePath", "../images/scene.png"},
                {"imageData", QJsonValue::Null},
                {"custom", 42}};
    }
    QTemporaryDir temporary;
    QDir root;
    QString imagePath, labels;
    QImage image;
    QJsonObject json;
};

TEST_F(AnnotationDocumentTest, ResolvesRelativeAndAbsolutePaths) {
    ASSERT_TRUE(write(labels, json));
    const auto relative = loadAnnotationDocument(labels);
    EXPECT_TRUE(relative.error.isEmpty()) << relative.error.toStdString();
    EXPECT_EQ(relative.imagePath, imagePath);
    EXPECT_EQ(relative.image.size(), image.size());
    EXPECT_EQ(relative.json, json);
    json.insert("imagePath", imagePath);
    ASSERT_TRUE(write(labels, json));
    EXPECT_TRUE(loadAnnotationDocument(labels).error.isEmpty());
}

TEST_F(AnnotationDocumentTest,
       SupportsRelocatedImagesAndExplicitLabelDirectory) {
    ASSERT_TRUE(image.save(root.filePath("labels/scene.png")));
    json.insert("imagePath", "/old/machine/scene.png");
    ASSERT_TRUE(write(labels, json));
    const auto fallback = loadAnnotationDocument(labels);
    EXPECT_TRUE(fallback.error.isEmpty());
    EXPECT_EQ(fallback.imagePath, root.filePath("labels/scene.png"));
    json.insert("imagePath", "/missing/original.png");
    ASSERT_TRUE(write(labels, json));
    const auto explicitImage =
            loadAnnotationDocument(imagePath, root.filePath("labels"));
    EXPECT_TRUE(explicitImage.error.isEmpty());
    EXPECT_EQ(explicitImage.imagePath, imagePath);
    EXPECT_EQ(explicitImage.labelPath, labels);
    EXPECT_EQ(explicitImage.json, json);
}

TEST_F(AnnotationDocumentTest, EmbeddedPixelsTakePrecedenceAndRetainMetadata) {
    QImage embeddedImage(11, 7, QImage::Format_RGB32);
    embeddedImage.fill(Qt::blue);
    QByteArray encoded;
    QBuffer buffer(&encoded);
    ASSERT_TRUE(buffer.open(QIODevice::WriteOnly));
    ASSERT_TRUE(embeddedImage.save(&buffer, "PNG"));
    json.insert("imageData", QString::fromLatin1(encoded.toBase64()));
    ASSERT_TRUE(write(labels, json));
    const auto embedded = loadAnnotationDocument(labels);
    EXPECT_TRUE(embedded.error.isEmpty());
    EXPECT_EQ(embedded.image.size(), embeddedImage.size());
    EXPECT_EQ(embedded.json.value("custom").toInt(), 42);
}

TEST_F(AnnotationDocumentTest, RejectsMalformedImageDataShapesAndJson) {
    json.insert("imageData", "broken pixels");
    ASSERT_TRUE(write(labels, json));
    EXPECT_FALSE(loadAnnotationDocument(labels).error.isEmpty());
    json.insert("imageData", 123);
    ASSERT_TRUE(write(labels, json));
    EXPECT_FALSE(loadAnnotationDocument(labels).error.isEmpty());
    json.insert("shapes", QJsonObject());
    ASSERT_TRUE(write(labels, json));
    EXPECT_FALSE(loadAnnotationDocument(labels).error.isEmpty());
    QFile corrupt(labels);
    ASSERT_TRUE(corrupt.open(QIODevice::WriteOnly));
    ASSERT_EQ(corrupt.write("{"), 1);
    corrupt.close();
    EXPECT_FALSE(loadAnnotationDocument(imagePath, root.filePath("labels"))
                         .error.isEmpty());
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc >= 2 && !QString::fromLocal8Bit(argv[1]).startsWith("--gtest")) {
        const auto loaded = loadAnnotationDocument(
                QString::fromLocal8Bit(argv[1]),
                argc >= 3 ? QString::fromLocal8Bit(argv[2]) : QString());
        const QJsonObject result{{"ok", loaded.error.isEmpty()},
                                 {"path", loaded.imagePath},
                                 {"size", QJsonArray{loaded.image.width(),
                                                     loaded.image.height()}},
                                 {"error", loaded.error}};
        std::puts(QJsonDocument(result)
                          .toJson(QJsonDocument::Compact)
                          .constData());
        return 0;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
