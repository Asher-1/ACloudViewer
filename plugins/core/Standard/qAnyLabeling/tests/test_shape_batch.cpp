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
bool check(bool passed, const char* message) {
    if (!passed) std::fprintf(stderr, "FAIL: %s\n", message);
    return passed;
}
}  // namespace

int main(int argc, char** argv) {
    QCoreApplication application(argc, argv);
    if (argc == 2) {
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
    QTemporaryDir work;
    if (!work.isValid()) return 1;
    QDir root(work.path());
    if (!root.mkdir("images") || !root.mkdir("labels")) return 1;
    QStringList images;
    QImage pixels(80, 60, QImage::Format_RGB32);
    pixels.fill(Qt::white);
    for (int i = 0; i < 3; ++i) {
        images.append(root.filePath(QStringLiteral("images/%1.png").arg(i)));
        if (!pixels.save(images.last())) return 1;
    }
    const auto labels = root.filePath("labels");
    const auto a = root.filePath("labels/0.json"),
               b = root.filePath("labels/1.json");
    const QJsonObject shape{
            {"label", "box"},
            {"shape_type", "rectangle"},
            {"points", QJsonArray{QJsonArray{0, 0}, QJsonArray{80, 60}}},
            {"custom", 17}};
    const QJsonArray selected{shape};
    auto result = applyAnnotationShapeBatch(
            images, labels, 1, 3, AnnotationShapeBatchOperation::AddSelected,
            selected, [](int done, int) { return done < 1; });
    if (!check(result.canceled && result.shapes == 1 && QFile::exists(a) &&
                       !QFile::exists(b),
               "cancel must stop before changing the next image"))
        return 1;
    auto document = read(a);
    document.insert("unknown_document", 71);
    auto locked = shape;
    locked.insert("locked", true);
    document.insert("shapes", QJsonArray{shape, locked});
    if (!save(a, document)) return 1;
    result = applyAnnotationShapeBatch(
            images, labels, 1, 1, AnnotationShapeBatchOperation::RemoveSelected,
            selected);
    const auto retained = read(a);
    if (!check(result.shapes == 1 &&
                       retained.value("unknown_document").toInt() == 71 &&
                       retained.value("shapes").toArray() == QJsonArray{locked},
               "remove must preserve locked shape and unknown document data"))
        return 1;
    QFile malformed(b);
    if (!malformed.open(QIODevice::WriteOnly) || malformed.write("{") != 1)
        return 1;
    malformed.close();
    result = applyAnnotationShapeBatch(
            images, labels, 2, 2, AnnotationShapeBatchOperation::AddSelected,
            selected);
    if (!check(result.errors.size() == 1 &&
                       malformed.open(QIODevice::ReadOnly) &&
                       malformed.readAll() == "{",
               "malformed sidecar must remain untouched"))
        return 1;
    malformed.close();
    if (!root.mkdir("_delete_") ||
        !pixels.save(root.filePath("_delete_/0.png")))
        return 1;
    result = applyAnnotationShapeBatch(
            images, labels, 1, 1, AnnotationShapeBatchOperation::DeleteImages,
            {});
    if (!check(result.errors.size() == 1 && QFile::exists(images.first()) &&
                       QFile::exists(a),
               "trash collision must preserve both image and sidecar"))
        return 1;
    result = applyAnnotationShapeBatch(
            images, labels, 2, 1,
            AnnotationShapeBatchOperation::DeleteAnnotations, {});
    if (!check(!result.errors.isEmpty() && QFile::exists(a),
               "invalid range must not mutate files"))
        return 1;
    std::puts(
            "Shape batch cancellation/locks/metadata/corruption/collision: "
            "PASS");
    return 0;
}
