// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QPainter>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QTemporaryDir>
#include <QTextDocument>
#include <cmath>

#include "AnnotationArchive.h"
#include "PpocrDocument.h"
#include "QtCompat.h"

namespace PpocrDocument {
namespace {
QString richMarkdown(const QString& text) {
    static const QRegularExpression html(
            "</"
            "?(?:p|div|span|h[1-6]|strong|b|em|i|u|s|strike|del|img|br|ul|ol|"
            "li)\\b",
            QRegularExpression::CaseInsensitiveOption);
    if (!html.match(text).hasMatch()) return text;
    QTextDocument document;
    document.setHtml(text);
    return qtCompatToMarkdown(document).trimmed();
}
QImage readImage(const QString& path) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    return reader.read();
}
QImage flatten(const QImage& image) {
    QImage result(image.size(), QImage::Format_RGB32);
    result.fill(Qt::white);
    QPainter painter(&result);
    painter.drawImage(0, 0, image);
    return result;
}
}  // namespace

bool exportBundle(const QJsonObject& document,
                  const QString& rootDirectory,
                  const QStringList& pagePaths,
                  const QString& zipPath,
                  QString* error) {
    if (error) error->clear();
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    const auto pages = document.value("layoutParsingResults").toArray();
    if (pages.isEmpty()) return fail(QStringLiteral("No parsed document"));
    QTemporaryDir temporary;
    if (!temporary.isValid()) return fail(temporary.errorString());
    const QString package = temporary.path() + "/package";
    if (!QDir().mkpath(package + "/imgs"))
        return fail(QStringLiteral("Cannot create document export folder"));
    const auto imagePaths = document.value("_ppocr_meta")
                                    .toObject()
                                    .value("block_image_paths")
                                    .toObject();
    const QDir root(rootDirectory);
    const auto rootPath = QFileInfo(rootDirectory).canonicalFilePath();
    QVector<QVector<Block>> pageBlocks(pages.size());
    for (const auto& block : blocks(document))
        pageBlocks[block.page].append(block);
    QSet<QString> usedNames;
    QStringList parts;
    // Decode one page at a time. Retaining all rendered pages would multiply
    // memory by document length; the ZIP writer also streams 64 KiB chunks.
    for (int p = 0; p < pages.size(); ++p) {
        QImage pageImage;
        if (p < pagePaths.size() && !pagePaths[p].isEmpty()) {
            pageImage = readImage(pagePaths[p]);
            if (pageImage.isNull())
                return fail(QStringLiteral("Cannot decode page %1").arg(p + 1));
        }
        QImage layout = pageImage.isNull() ? QImage() : flatten(pageImage);
        const auto dimensions =
                pages[p].toObject().value("prunedResult").toObject();
        const double width =
                dimensions.value("width").toDouble(pageImage.width());
        const double height =
                dimensions.value("height").toDouble(pageImage.height());
        for (const auto& block : pageBlocks[p]) {
            auto polygon = block.polygon;
            if (!pageImage.isNull() && width > 0 && height > 0)
                for (auto& point : polygon)
                    point = QPointF(point.x() * pageImage.width() / width,
                                    point.y() * pageImage.height() / height);
            if (!layout.isNull() && polygon.size() >= 2) {
                QPainter painter(&layout);
                const auto color = categoryColor(block.label);
                auto fill = color;
                fill.setAlpha(64);
                painter.setPen(QPen(color, 2));
                painter.setBrush(fill);
                painter.drawPolygon(polygon);
                const auto box = polygon.boundingRect();
                const auto text = block.label;
                const auto size =
                        painter.fontMetrics().boundingRect(text).size() +
                        QSize(16, 8);
                const QPointF top(
                        qBound(0.0, box.left(),
                               double(qMax(0, layout.width() - size.width()))),
                        qMax(0.0, box.top() - size.height()));
                painter.fillRect(QRectF(top, size), color);
                painter.setPen(Qt::white);
                painter.drawText(QRectF(top, size), Qt::AlignCenter, text);
            }
            QImage asset;
            const bool imageBlock = QStringList{"image", "chart",
                                                "header_image", "footer_image"}
                                            .contains(block.label);
            auto relative = imageBlock ? imagePaths.value(block.key).toString()
                                       : QString();
            if (imageBlock && relative.isEmpty() &&
                block.key.contains("__dup_"))
                relative = imagePaths.value(block.key.section("__dup_", 0, 0))
                                   .toString();
            if (!relative.isEmpty()) {
                const auto path =
                        QFileInfo(root.filePath(relative)).canonicalFilePath();
                const auto local = QDir(rootPath).relativeFilePath(path);
                if (rootPath.isEmpty() || path.isEmpty() ||
                    QDir::isAbsolutePath(local) || local == ".." ||
                    local.startsWith("../"))
                    return fail(
                            QStringLiteral("Block image is missing or outside "
                                           "the document workspace"));
                asset = readImage(path);
                if (asset.isNull())
                    return fail(QStringLiteral("Cannot decode block image: %1")
                                        .arg(relative));
            } else if (QStringList{"image", "header_image", "footer_image"}
                               .contains(block.label) &&
                       !pageImage.isNull() && !polygon.isEmpty()) {
                const auto crop =
                        polygon.boundingRect().toAlignedRect().intersected(
                                pageImage.rect());
                if (!crop.isEmpty()) asset = pageImage.copy(crop);
            }
            QStringList chunks;
            const auto text = richMarkdown(block.text).trimmed();
            if (!text.isEmpty()) chunks.append(text);
            if (!asset.isNull()) {
                const auto box = block.polygon.boundingRect();
                const auto stem =
                        QStringLiteral("img_in_image_box_%1_%2_%3_%4")
                                .arg(qMax(0LL, std::llround(box.left())))
                                .arg(qMax(0LL, std::llround(box.top())))
                                .arg(qMax(0LL, std::llround(box.right())))
                                .arg(qMax(0LL, std::llround(box.bottom())));
                QString name = stem + ".jpg";
                int copy = 1;
                while (usedNames.contains(name))
                    name = stem + QStringLiteral("_%1.jpg").arg(copy++);
                usedNames.insert(name);
                if (!flatten(asset).save(package + "/imgs/" + name, "JPEG", 95))
                    return fail(QStringLiteral("Cannot save block image"));
                const int percent =
                        width > 0 && box.width() > 0
                                ? qBound(1,
                                         int(std::lround(box.width() * 100 /
                                                         width)),
                                         100)
                                : 100;
                chunks.append(
                        QStringLiteral("<div style=\"text-align: "
                                       "center;\"><img src=\"imgs/%1\" "
                                       "alt=\"Image\" width=\"%2%\" /></div>")
                                .arg(name)
                                .arg(percent));
            }
            if (!chunks.isEmpty()) parts.append(chunks.join("\n\n"));
        }
        if (!layout.isNull() &&
            !layout.save(
                    package + QStringLiteral("/layout_det_res_%1.jpg").arg(p),
                    "JPEG", 95))
            return fail(QStringLiteral("Cannot save page layout"));
    }
    QFile markdownFile(package + "/doc_0.md");
    const auto bytes = parts.join("\n\n\n").trimmed().toUtf8();
    if (!markdownFile.open(QIODevice::WriteOnly) ||
        markdownFile.write(bytes) != bytes.size())
        return fail(QStringLiteral("Cannot write document Markdown"));
    markdownFile.close();
    const QString archive = temporary.path() + "/document.zip";
    if (!writeZipDirectory(package, archive, error, false)) return false;
    QFile source(archive);
    QSaveFile destination(zipPath);
    if (!source.open(QIODevice::ReadOnly) ||
        !destination.open(QIODevice::WriteOnly))
        return fail(destination.errorString());
    while (!source.atEnd()) {
        const auto chunk = source.read(65536);
        if (chunk.isEmpty() || destination.write(chunk) != chunk.size())
            return fail(QStringLiteral("Cannot copy document ZIP"));
    }
    if (!destination.commit()) return fail(destination.errorString());
    return true;
}
}  // namespace PpocrDocument
