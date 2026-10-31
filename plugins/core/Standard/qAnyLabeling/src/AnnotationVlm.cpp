// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationVlm.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSet>
#include <algorithm>
#include <cmath>

#include "AnnotationDocument.h"

namespace {
QString answerText(const QString& text) {
    QString answer = text.trimmed();
    const QRegularExpression tag(
            QStringLiteral("<answer>(.*?)</answer>"),
            QRegularExpression::DotMatchesEverythingOption);
    QRegularExpressionMatchIterator matches = tag.globalMatch(answer);
    while (matches.hasNext()) answer = matches.next().captured(1).trimmed();
    const int fence = answer.indexOf(QStringLiteral("```"));
    if (fence >= 0) {
        const int start = answer.indexOf(QLatin1Char('\n'), fence);
        const int end = answer.indexOf(QStringLiteral("```"), start + 1);
        if (start >= 0 && end > start)
            answer = answer.mid(start + 1, end - start - 1).trimmed();
    }
    return answer;
}

QByteArray repairedArrayJson(const QString& text) {
    QString answer = answerText(text).trimmed();
    if (answer.compare(QStringLiteral("None"), Qt::CaseInsensitive) == 0)
        return {};
    const int first = answer.indexOf(QLatin1Char('['));
    const int last = answer.lastIndexOf(QLatin1Char(']'));
    if (first < 0 || last <= first) return {};
    answer = answer.mid(first, last - first + 1);
    answer.replace(QLatin1Char('\''), QLatin1Char('"'));
    static const QRegularExpression key(
            QStringLiteral("([\\{,]\\s*)([A-Za-z_][A-Za-z0-9_]*)\\s*:"));
    answer.replace(key, QStringLiteral(R"(\1"\2":)"));
    static const QRegularExpression trailing(QStringLiteral(",\\s*([}\\]])"));
    answer.replace(trailing, QStringLiteral("\\1"));
    return answer.toUtf8();
}
}  // namespace

VlmImport importVlm(const QByteArray& jsonLines) {
    VlmImport result;
    const QList<QByteArray> lines = jsonLines.split('\n');
    for (int lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        const QByteArray line = lines[lineIndex].trimmed();
        if (line.isEmpty()) continue;
        QJsonParseError parse;
        const QJsonDocument record = QJsonDocument::fromJson(line, &parse);
        if (parse.error != QJsonParseError::NoError || !record.isObject()) {
            result.error =
                    QStringLiteral("Invalid VLM-R1 JSONL record at line %1")
                            .arg(lineIndex + 1);
            return result;
        }
        const QJsonObject object = record.object();
        const QString name =
                QFileInfo(object.value(QStringLiteral("image")).toString())
                        .fileName();
        const QJsonArray conversation =
                object.value(QStringLiteral("conversations")).toArray();
        if (name.isEmpty() || result.documents.contains(name) ||
            conversation.size() < 2 || !conversation[1].isObject() ||
            !conversation[1]
                     .toObject()
                     .value(QStringLiteral("value"))
                     .isString()) {
            result.error =
                    QStringLiteral(
                            "Invalid or duplicate VLM-R1 image at line %1")
                            .arg(lineIndex + 1);
            return result;
        }
        const QString answer = conversation[1]
                                       .toObject()
                                       .value(QStringLiteral("value"))
                                       .toString();
        const QByteArray decodedAnswer = repairedArrayJson(answer);
        QJsonArray boxes;
        if (!decodedAnswer.isEmpty()) {
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(decodedAnswer, &parse);
            if (parse.error == QJsonParseError::NoError && parsed.isArray())
                boxes = parsed.array();
        }
        QJsonArray shapes;
        for (const QJsonValue& value : boxes) {
            const QJsonObject box = value.toObject();
            const QJsonArray xyxy =
                    box.value(QStringLiteral("bbox_2d")).toArray();
            const QString label =
                    box.value(QStringLiteral("label")).toString().trimmed();
            if (xyxy.size() != 4 || label.isEmpty()) {
                result.error = QStringLiteral("Invalid VLM-R1 box at line %1")
                                       .arg(lineIndex + 1);
                return result;
            }
            double xy[4];
            for (int i = 0; i < 4; ++i) {
                if (!xyxy[i].isDouble() || !std::isfinite(xyxy[i].toDouble())) {
                    result.error =
                            QStringLiteral(
                                    "Invalid VLM-R1 coordinates at line %1")
                                    .arg(lineIndex + 1);
                    return result;
                }
                xy[i] = xyxy[i].toDouble();
            }
            if (xy[2] <= xy[0] || xy[3] <= xy[1]) {
                result.error =
                        QStringLiteral("Invalid VLM-R1 box size at line %1")
                                .arg(lineIndex + 1);
                return result;
            }
            shapes.append(QJsonObject{
                    {QStringLiteral("label"), label},
                    {QStringLiteral("shape_type"), QStringLiteral("rectangle")},
                    {QStringLiteral("group_id"), QJsonValue::Null},
                    {QStringLiteral("description"), QJsonValue::Null},
                    {QStringLiteral("points"),
                     QJsonArray{QJsonArray{xy[0], xy[1]},
                                QJsonArray{xy[2], xy[1]},
                                QJsonArray{xy[2], xy[3]},
                                QJsonArray{xy[0], xy[3]}}}});
            ++result.count;
        }
        result.documents.insert(
                name,
                QJsonObject{
                        {QStringLiteral("version"), annotationFormatVersion()},
                        {QStringLiteral("flags"), QJsonObject()},
                        {QStringLiteral("shapes"), shapes},
                        {QStringLiteral("imagePath"), name},
                        {QStringLiteral("imageData"), QJsonValue::Null}});
    }
    return result;
}

VlmConversion convertVlm(const QVector<CocoImage>& images,
                         const QStringList& classes,
                         const QString& prefix) {
    VlmConversion result;
    QSet<QString> names;
    for (int index = 0; index < images.size(); ++index) {
        const CocoImage& image = images[index];
        const int width =
                image.document.value(QStringLiteral("imageWidth")).toInt();
        const int height =
                image.document.value(QStringLiteral("imageHeight")).toInt();
        if (image.fileName.isEmpty() || names.contains(image.fileName) ||
            width <= 0 || height <= 0) {
            result.error = QStringLiteral("Invalid VLM-R1 image: %1")
                                   .arg(image.fileName);
            return result;
        }
        names.insert(image.fileName);
        QJsonArray boxes;
        QStringList labels = classes;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            const QJsonObject shape = value.toObject();
            if (shape.value(QStringLiteral("shape_type")).toString() !=
                QStringLiteral("rectangle"))
                continue;
            const QString label =
                    shape.value(QStringLiteral("label")).toString();
            if (label.isEmpty() ||
                (!classes.isEmpty() && !classes.contains(label)))
                continue;
            const QJsonArray points =
                    shape.value(QStringLiteral("points")).toArray();
            if (points.size() != 2 && points.size() != 4) {
                result.error = QStringLiteral("Invalid VLM-R1 rectangle in %1")
                                       .arg(image.fileName);
                return result;
            }
            const QJsonArray first = points[0].toArray();
            const QJsonArray opposite =
                    points[points.size() == 2 ? 1 : 2].toArray();
            if (first.size() != 2 || opposite.size() != 2 ||
                !first[0].isDouble() || !first[1].isDouble() ||
                !opposite[0].isDouble() || !opposite[1].isDouble() ||
                !std::isfinite(first[0].toDouble()) ||
                !std::isfinite(first[1].toDouble()) ||
                !std::isfinite(opposite[0].toDouble()) ||
                !std::isfinite(opposite[1].toDouble())) {
                result.error = QStringLiteral("Invalid VLM-R1 rectangle in %1")
                                       .arg(image.fileName);
                return result;
            }
            const int x0 = int(
                    std::clamp(first[0].toDouble(), 0.0, double(width - 1)));
            const int y0 = int(
                    std::clamp(first[1].toDouble(), 0.0, double(height - 1)));
            const int x1 = int(
                    std::clamp(opposite[0].toDouble(), 0.0, double(width - 1)));
            const int y1 = int(std::clamp(opposite[1].toDouble(), 0.0,
                                          double(height - 1)));
            if (x1 <= x0 || y1 <= y0) {
                result.error = QStringLiteral("Invalid VLM-R1 box size in %1")
                                       .arg(image.fileName);
                return result;
            }
            boxes.append(QJsonObject{
                    {QStringLiteral("bbox_2d"), QJsonArray{x0, y0, x1, y1}},
                    {QStringLiteral("label"), label}});
            if (classes.isEmpty() && !labels.contains(label))
                labels.append(label);
            ++result.count;
        }
        if (boxes.isEmpty() && classes.isEmpty()) continue;
        const QString answer =
                boxes.isEmpty()
                        ? QStringLiteral("None")
                        : QStringLiteral("```json\n%1\n```")
                                  .arg(QString::fromUtf8(
                                               QJsonDocument(boxes).toJson(
                                                       QJsonDocument::Indented))
                                               .trimmed());
        const QString labelList = QStringLiteral("['%1']").arg(
                labels.join(QStringLiteral("', '")));
        const QString question =
                QStringLiteral(
                        "Please carefully check the image and detect the "
                        "following objects: %1. "
                        "Output the bbox coordinates of detected objects in "
                        "JSON format in <answer></answer>. "
                        "The bbox coordinate format should be: \n```json\n"
                        "[{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"object "
                        "name\"},\n"
                        "{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"object "
                        "name\"},\n"
                        "...\n]\n```\n. If no targets are detected in the "
                        "image, simply respond with \"None\".")
                        .arg(labelList);
        const QJsonObject human{
                {QStringLiteral("from"), QStringLiteral("human")},
                {QStringLiteral("value"), question}};
        const QJsonObject assistant{
                {QStringLiteral("from"), QStringLiteral("assistant")},
                {QStringLiteral("value"), answer}};
        QJsonArray conversation;
        conversation.append(human);
        conversation.append(assistant);
        QJsonObject record;
        record.insert(QStringLiteral("id"), index + 1);
        record.insert(QStringLiteral("image"),
                      QString(prefix + image.fileName));
        record.insert(QStringLiteral("conversations"), conversation);
        result.annotations +=
                QJsonDocument(record).toJson(QJsonDocument::Compact);
        result.annotations += '\n';
    }
    return result;
}
