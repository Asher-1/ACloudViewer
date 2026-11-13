// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PpocrDocument.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <algorithm>

namespace PpocrDocument {
QJsonObject parse(const QByteArray& input, QString* error) {
    if (error) error->clear();
    QVector<QJsonValue> records;
    QJsonParseError parseError;
    const auto parsed = QJsonDocument::fromJson(input, &parseError);
    if (parseError.error == QJsonParseError::NoError) {
        if (parsed.isObject()) records.append(parsed.object());
        if (parsed.isArray())
            for (const auto& record : parsed.array()) records.append(record);
    } else {
        int number = 0;
        for (const auto& line : input.split('\n')) {
            ++number;
            if (line.trimmed().isEmpty()) continue;
            const auto row = QJsonDocument::fromJson(line, &parseError);
            if (parseError.error != QJsonParseError::NoError ||
                !row.isObject()) {
                if (error)
                    *error = QStringLiteral("Invalid OCR JSONL line %1")
                                     .arg(number);
                return {};
            }
            records.append(row.object());
        }
    }
    QJsonArray pages, images;
    QJsonObject result, info;
    for (const auto& record : records) {
        auto row = record.toObject();
        if (row.contains("result")) row = row.value("result").toObject();
        if (records.size() == 1)
            result = row;  // Preserve unknown document fields.
        for (const auto& page : row.value("layoutParsingResults").toArray())
            if (page.isObject()) pages.append(page);
        if (row.value("prunedResult").isObject()) pages.append(row);
        for (const auto& image : row.value("preprocessedImages").toArray())
            images.append(image);
        const auto data = row.value("dataInfo").toObject();
        for (auto it = data.begin(); it != data.end(); ++it)
            info.insert(it.key(), it.value());
    }
    if (pages.isEmpty()) {
        if (error)
            *error = QStringLiteral("OCR result contains no page results");
        return {};
    }
    result.insert("layoutParsingResults", pages);
    result.insert("preprocessedImages", images);
    result.insert("dataInfo", info);
    return result;
}

QVector<Block> blocks(const QJsonObject& document) {
    QVector<Block> result;
    const auto imagePaths = document.value("_ppocr_meta")
                                    .toObject()
                                    .value("block_image_paths")
                                    .toObject();
    QSet<QString> edited;
    for (const auto& value : document.value("_ppocr_meta")
                                     .toObject()
                                     .value("edited_blocks")
                                     .toArray())
        edited.insert(value.toString());
    const auto pages = document.value("layoutParsingResults").toArray();
    for (int p = 0; p < pages.size(); ++p) {
        const int begin = result.size();
        QHash<QString, int> seen;
        const auto rows = pages[p].toObject()
                                  .value("prunedResult")
                                  .toObject()
                                  .value("parsing_res_list")
                                  .toArray();
        for (int i = 0; i < rows.size(); ++i) {
            const auto row = rows[i].toObject();
            auto id = row.value("global_block_id");
            QString stem = QStringLiteral("global_block_");
            if (id.isNull() || id.isUndefined()) {
                stem = QStringLiteral("block_");
                id = row.value("block_id");
            }
            const auto token =
                    id.isString() ? id.toString()
                                  : QString::number(id.isDouble() ? id.toInt()
                                                                  : i + 1);
            QString key =
                    QStringLiteral("page_%1:%2%3").arg(p + 1).arg(stem, token);
            const auto baseKey = key;
            const int occurrence = ++seen[key];
            if (occurrence > 1)
                key += QStringLiteral("__dup_%1").arg(occurrence);
            Block block;
            block.key = key;
            block.page = p;
            block.index = i;
            block.order = row.value("block_order").toInt(i + 1);
            if (!block.order) block.order = i + 1;
            block.label =
                    row.value("block_label").toString().trimmed().toLower();
            block.label.replace(QRegularExpression(QStringLiteral("[\\s-]+")),
                                "_");
            if (block.label.isEmpty()) block.label = QStringLiteral("text");
            block.text = row.value("block_content").toString();
            if (QStringList{"image", "chart", "header_image", "footer_image"}
                        .contains(block.label)) {
                block.imagePath = imagePaths.value(key).toString();
                if (block.imagePath.isEmpty())
                    block.imagePath = imagePaths.value(baseKey).toString();
            }
            block.edited = edited.contains(key);
            auto polygon = row.value("block_polygon_points").toArray();
            if (polygon.isEmpty())
                polygon = row.value("block_polygon").toArray();
            if (polygon.isEmpty()) polygon = row.value("polygon").toArray();
            for (const auto& point : polygon) {
                const auto xy = point.toArray();
                if (xy.size() >= 2)
                    block.polygon.append(
                            QPointF(xy[0].toDouble(), xy[1].toDouble()));
            }
            if (block.polygon.isEmpty()) {
                const auto box = row.value("block_bbox").toArray();
                if (box.size() >= 4) {
                    const double x = box[0].toDouble(), y = box[1].toDouble();
                    const double right = box[2].toDouble(),
                                 bottom = box[3].toDouble();
                    block.polygon << QPointF(x, y) << QPointF(right, y)
                                  << QPointF(right, bottom)
                                  << QPointF(x, bottom);
                }
            }
            result.append(block);
        }
        // Upstream block_order is reading order and may differ from JSON order.
        // Stable O(n log n) sorting preserves ties and each block's source
        // index.
        std::stable_sort(result.begin() + begin, result.end(),
                         [](const Block& a, const Block& b) {
                             return a.order < b.order;
                         });
    }
    return result;
}

QColor categoryColor(const QString& label) {
    if (label == "table") return QColor(47, 189, 113);
    if (QStringList{"image", "chart", "header_image", "footer_image"}.contains(
                label))
        return QColor(189, 76, 255);
    if (label == "header") return QColor(182, 178, 241);
    if (QStringList{"display_formula", "formula_number", "algorithm", "formula"}
                .contains(label))
        return QColor(250, 219, 20);
    return QColor(70, 88, 255);
}

bool edit(QJsonObject* document, const QString& key, const QString& text) {
    for (const auto& block : blocks(*document)) {
        if (block.key != key) continue;
        auto pages = document->value("layoutParsingResults").toArray();
        auto page = pages[block.page].toObject();
        auto pruned = page.value("prunedResult").toObject();
        auto rows = pruned.value("parsing_res_list").toArray();
        auto row = rows[block.index].toObject();
        row.insert("block_content", text);
        rows[block.index] = row;
        pruned.insert("parsing_res_list", rows);
        page.insert("prunedResult", pruned);
        pages[block.page] = page;
        document->insert("layoutParsingResults", pages);
        // A service markdown string predates the correction. Rebuild it from
        // current blocks rather than exporting silently stale text.
        auto md = page.value("markdown").toObject();
        QStringList texts;
        for (const auto& current : blocks(*document))
            if (current.page == block.page && !current.text.isEmpty())
                texts.append(current.text);
        md.insert("text", texts.join("\n\n"));
        page.insert("markdown", md);
        pages[block.page] = page;
        document->insert("layoutParsingResults", pages);
        auto meta = document->value("_ppocr_meta").toObject();
        auto edited = meta.value("edited_blocks").toArray();
        if (!edited.contains(key)) edited.append(key);
        meta.insert("edited_blocks", edited);
        document->insert("_ppocr_meta", meta);
        return true;
    }
    return false;
}

QString markdown(const QJsonObject& document) {
    QStringList pages;
    for (const auto& value : document.value("layoutParsingResults").toArray()) {
        const auto page = value.toObject();
        QString text =
                page.value("markdown").toObject().value("text").toString();
        if (text.isEmpty()) {
            QStringList contents;
            for (const auto& block : page.value("prunedResult")
                                             .toObject()
                                             .value("parsing_res_list")
                                             .toArray()) {
                const auto content =
                        block.toObject().value("block_content").toString();
                if (!content.isEmpty()) contents.append(content);
            }
            text = contents.join("\n\n");
        }
        pages.append(text);
    }
    return pages.join("\n\n");
}

bool save(const QString& path, const QJsonObject& document, QString* error) {
    QSaveFile file(path);
    const auto bytes = QJsonDocument(document).toJson();
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit()) {
        if (error) *error = file.errorString();
        return false;
    }
    return true;
}
}  // namespace PpocrDocument
