// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationShapeBatch.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <cmath>

namespace {
bool identical(const QJsonObject& a, const QJsonObject& b) {
    if (a.value("label") != b.value("label") ||
        a.value("shape_type").toString("polygon") !=
                b.value("shape_type").toString("polygon"))
        return false;
    const auto ap = a.value("points").toArray();
    const auto bp = b.value("points").toArray();
    if (ap.size() != bp.size()) return false;
    for (int i = 0; i < ap.size(); ++i) {
        const auto p = ap[i].toArray(), q = bp[i].toArray();
        if (p.size() != 2 || q.size() != 2 ||
            std::abs(p[0].toDouble() - q[0].toDouble()) > 1.0 ||
            std::abs(p[1].toDouble() - q[1].toDouble()) > 1.0)
            return false;
    }
    return true;
}

bool withinBounds(const QJsonObject& shape, const QSize& size) {
    for (const auto& value : shape.value("points").toArray()) {
        const auto point = value.toArray();
        if (point.size() != 2 || !point[0].isDouble() || !point[1].isDouble())
            return false;
        const double x = point[0].toDouble(), y = point[1].toDouble();
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 ||
            x > size.width() || y > size.height())
            return false;
    }
    return true;
}
}  // namespace

AnnotationShapeBatchResult applyAnnotationShapeBatch(
        const QStringList& images,
        const QString& outputDirectory,
        int first,
        int last,
        AnnotationShapeBatchOperation operation,
        const QJsonArray& selected,
        const std::function<bool(int, int)>& progress) {
    AnnotationShapeBatchResult result;
    if (first < 1 || last < first || last > images.size()) {
        result.errors.append(QStringLiteral("Invalid frame range"));
        return result;
    }
    if ((operation == AnnotationShapeBatchOperation::AddSelected ||
         operation == AnnotationShapeBatchOperation::RemoveSelected) &&
        selected.isEmpty())
        return result;
    const int total = last - first + 1;
    // Upstream uses the parent of the first image's folder for batch deletion.
    const QDir trash(QDir(QFileInfo(images.first()).absolutePath())
                             .absoluteFilePath(QStringLiteral("../_delete_")));
    for (int row = first - 1; row < last; ++row) {
        if (progress && !progress(row - first + 1, total)) {
            result.canceled = true;
            break;
        }
        const QFileInfo image(images[row]);
        const QString label =
                QDir(outputDirectory.isEmpty() ? image.absolutePath()
                                               : outputDirectory)
                        .filePath(image.completeBaseName() + ".json");
        if (operation == AnnotationShapeBatchOperation::DeleteAnnotations) {
            if (QFile::exists(label)) {
                if (QFile::remove(label))
                    ++result.files;
                else
                    result.errors.append(label);
            }
            continue;
        }
        if (operation == AnnotationShapeBatchOperation::DeleteImages) {
            if (!image.isFile()) continue;
            // Preserve an earlier deleted image rather than overwrite it.
            // QFile::rename also handles a cross-filesystem move by
            // copy/remove.
            const QString destination = trash.filePath(image.fileName());
            if (!QDir().mkpath(trash.absolutePath()) ||
                !QFile::rename(image.absoluteFilePath(), destination)) {
                result.errors.append(image.absoluteFilePath());
                continue;
            }
            ++result.files;
            if (QFile::exists(label) && !QFile::remove(label))
                result.errors.append(label);
            continue;
        }
        const bool existed = QFile::exists(label);
        if (!existed &&
            operation == AnnotationShapeBatchOperation::RemoveSelected)
            continue;
        QJsonObject document;
        if (existed) {
            QFile input(label);
            if (!input.open(QIODevice::ReadOnly)) {
                result.errors.append(label);
                continue;
            }
            QJsonParseError error;
            const auto json = QJsonDocument::fromJson(input.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !json.isObject() ||
                (json.object().contains("shapes") &&
                 !json.object().value("shapes").isArray())) {
                result.errors.append(label);
                continue;
            }
            document = json.object();
        }
        auto shapes = document.value("shapes").toArray();
        int changed = 0;
        if (operation == AnnotationShapeBatchOperation::RemoveSelected) {
            if (shapes.isEmpty()) continue;
            QJsonArray retained;
            for (const auto& value : shapes) {
                const auto shape = value.toObject();
                bool remove = false;
                if (!shape.value("locked").toBool())
                    for (const auto& selection : selected)
                        if (identical(shape, selection.toObject())) {
                            remove = true;
                            break;
                        }
                if (remove)
                    ++changed;
                else
                    retained.append(value);
            }
            if (!changed) continue;
            if (retained.isEmpty()) {
                if (QFile::remove(label)) {
                    ++result.files;
                    result.shapes += changed;
                } else
                    result.errors.append(label);
                continue;
            }
            shapes = retained;
        } else {
            const QImage pixels(image.absoluteFilePath());
            if (pixels.isNull()) continue;
            if (!existed)
                document = QJsonObject{{"version", "5.5.0"},
                                       {"flags", QJsonObject()},
                                       {"imagePath", image.fileName()},
                                       {"imageData", QJsonValue::Null},
                                       {"imageHeight", pixels.height()},
                                       {"imageWidth", pixels.width()}};
            // Identity is label + type + ordered coordinates within 1 pixel.
            // Pairwise comparison intentionally matches the upstream contract;
            // a hash would lose the non-transitive coordinate tolerance.
            for (const auto& selection : selected) {
                if (!withinBounds(selection.toObject(), pixels.size())) {
                    ++result.skipped;
                    continue;
                }
                bool duplicate = false;
                for (const auto& value : shapes)
                    if (identical(value.toObject(), selection.toObject())) {
                        duplicate = true;
                        break;
                    }
                if (!duplicate) {
                    shapes.append(selection);
                    ++changed;
                }
            }
        }
        document.insert("shapes", shapes);
        QSaveFile output(label);
        const auto bytes =
                QJsonDocument(document).toJson(QJsonDocument::Indented);
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(bytes) != bytes.size() || !output.commit()) {
            result.errors.append(label);
            continue;
        }
        ++result.files;
        result.shapes += changed;
    }
    return result;
}
