// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QHash>
#include <QJsonArray>
#include <QPointF>
#include <algorithm>
#include <cmath>

#include "AnnotationExporter.h"
#include "AnnotationYaml.h"

namespace {
QString scalar(QString value) {
    value = value.trimmed();
    if (value.size() >= 2 && ((value.startsWith(QLatin1Char('\'')) &&
                               value.endsWith(QLatin1Char('\''))) ||
                              (value.startsWith(QLatin1Char('"')) &&
                               value.endsWith(QLatin1Char('"')))))
        value = value.mid(1, value.size() - 2);
    return value;
}

QString coordinate(double value) {
    return QString::number(std::round(value * 1000000.0) / 1000000.0, 'g', 15);
}

QVector<QPointF> pointsFor(const QJsonObject& shape) {
    QVector<QPointF> points;
    for (const QJsonValue& value :
         shape.value(QStringLiteral("points")).toArray()) {
        const QJsonArray pair = value.toArray();
        if (pair.size() != 2 || !pair[0].isDouble() || !pair[1].isDouble())
            return {};
        const double x = pair[0].toDouble(), y = pair[1].toDouble();
        if (!std::isfinite(x) || !std::isfinite(y)) return {};
        points.append(QPointF(x, y));
    }
    return points;
}

struct PoseGroup {
    int id = 0;
    QString boxLabel;
    QVector<QPointF> box;
    bool difficult = false;
    QHash<QString, QPointF> keypoints;
    QHash<QString, int> visibility;
};

QString collectPoseGroups(const QJsonObject& document,
                          QVector<PoseGroup>* groups) {
    QHash<int, int> groupIndex;
    for (const QJsonValue& value :
         document.value(QStringLiteral("shapes")).toArray()) {
        const QJsonObject shape = value.toObject();
        const QString type =
                shape.value(QStringLiteral("shape_type")).toString();
        if (type != QStringLiteral("rectangle") &&
            type != QStringLiteral("point"))
            continue;
        const QJsonValue idValue = shape.value(QStringLiteral("group_id"));
        bool validId = idValue.isDouble();
        int id = idValue.toInt();
        if (idValue.isString()) id = idValue.toString().toInt(&validId);
        if (!validId || (idValue.isDouble() && idValue.toDouble() != id))
            return QStringLiteral(
                    "Pose rectangle/point needs an integer group_id");
        const QVector<QPointF> points = pointsFor(shape);
        if ((type == QStringLiteral("point") && points.size() != 1) ||
            (type == QStringLiteral("rectangle") && points.size() != 2 &&
             points.size() != 4))
            return QStringLiteral("Invalid pose geometry in group %1").arg(id);
        if (!groupIndex.contains(id)) {
            groupIndex.insert(id, groups->size());
            groups->append({id, {}, {}, false, {}, {}});
        }
        PoseGroup& group = (*groups)[groupIndex.value(id)];
        const QString label = shape.value(QStringLiteral("label")).toString();
        if (type == QStringLiteral("rectangle")) {
            group.box = points;
            group.boxLabel = label;
            group.difficult = shape.value(QStringLiteral("difficult")).toBool();
        } else {
            group.keypoints.insert(label, points[0]);
            group.visibility.insert(
                    label,
                    shape.value(QStringLiteral("difficult")).toBool() ? 1 : 2);
        }
    }
    return {};
}
}  // namespace

PoseConfig parsePoseConfig(const QByteArray& yaml) {
    PoseConfig config;
    // The source reads pose configs with yaml.safe_load and only touches
    // `has_visible` (default true) and the `classes` mapping; every other
    // root key is ignored. Mirrors that contract, including flow style,
    // anchors/aliases, comments and any consistent indentation.
    const AnnotationYamlResult parsed =
            parseAnnotationYaml(QString::fromUtf8(yaml));
    if (!parsed.error.isEmpty()) {
        config.error = parsed.error;
        return config;
    }
    const AnnotationYamlNode& root = parsed.root;
    if (root.type != AnnotationYamlNode::Mapping) {
        config.error = QStringLiteral(
                "Invalid pose config: root node must be a mapping");
        return config;
    }
    const AnnotationYamlNode* classes = nullptr;
    for (const auto& entry : root.mapping) {
        if (entry.first == QStringLiteral("has_visible")) {
            if (!entry.second.isBool()) {
                config.error =
                        QStringLiteral("has_visible must be a YAML boolean");
                return config;
            }
            config.hasVisible = entry.second.boolValue();
        } else if (entry.first == QStringLiteral("classes")) {
            classes = &entry.second;
        }
        // Extra root keys (name, kpt_shape, ...) are ignored like the
        // source dictionary access does.
    }
    if (!classes || classes->type != AnnotationYamlNode::Mapping ||
        classes->mapping.isEmpty()) {
        config.error = QStringLiteral(
                "Pose configuration needs classes and "
                "keypoints");
        return config;
    }
    for (const auto& entry : classes->mapping) {
        const AnnotationYamlNode& keypoints = entry.second;
        if (entry.first.isEmpty() ||
            keypoints.type != AnnotationYamlNode::Sequence ||
            keypoints.sequence.isEmpty()) {
            config.error = QStringLiteral(
                    "Pose configuration needs classes and keypoints");
            return config;
        }
        // Duplicate class names follow dict semantics: the last value wins
        // but the first position is kept. Keypoint duplicates are preserved
        // like Python lists are under yaml.safe_load.
        PoseClass* target = nullptr;
        for (PoseClass& item : config.classes)
            if (item.name == entry.first) target = &item;
        if (!target) {
            config.classes.append({entry.first, {}});
            target = &config.classes.last();
        }
        for (const AnnotationYamlNode& item : keypoints.sequence) {
            if (item.type != AnnotationYamlNode::Scalar ||
                item.scalar.isEmpty()) {
                config.error = QStringLiteral(
                        "Pose configuration needs classes and keypoints");
                return config;
            }
            target->keypoints.append(item.scalar);
        }
    }
    return config;
}

YoloConversion convertYoloPose(const QJsonObject& document,
                               const PoseConfig& config) {
    YoloConversion output;
    if (!config.error.isEmpty() || config.classes.isEmpty()) {
        output.error = config.error.isEmpty()
                               ? QStringLiteral("Invalid pose classes")
                               : config.error;
        return output;
    }
    const int width = document.value(QStringLiteral("imageWidth")).toInt();
    const int height = document.value(QStringLiteral("imageHeight")).toInt();
    if (width <= 0 || height <= 0) {
        output.error = QStringLiteral("Invalid image dimensions");
        return output;
    }
    QVector<PoseGroup> groups;
    output.error = collectPoseGroups(document, &groups);
    if (!output.error.isEmpty()) return output;
    int maxKeypoints = 0;
    for (const PoseClass& item : config.classes)
        maxKeypoints = std::max(maxKeypoints, item.keypoints.size());
    for (const PoseGroup& group : groups) {
        if (group.box.isEmpty()) {
            output.error = QStringLiteral("Missing pose rectangle in group %1")
                                   .arg(group.id);
            return output;
        }
        int classIndex = -1;
        for (int i = 0; i < config.classes.size(); ++i)
            if (config.classes[i].name == group.boxLabel) {
                classIndex = i;
                break;
            }
        if (classIndex < 0) {
            output.error = QStringLiteral("Unknown pose class in group %1: %2")
                                   .arg(group.id)
                                   .arg(group.boxLabel);
            return output;
        }
        const QPointF first(std::clamp(group.box[0].x(), 0.0, double(width)),
                            std::clamp(group.box[0].y(), 0.0, double(height)));
        const QPointF opposite =
                group.box.size() == 2 ? group.box[1] : group.box[2];
        const QPointF third(std::clamp(opposite.x(), 0.0, double(width)),
                            std::clamp(opposite.y(), 0.0, double(height)));
        QStringList fields{
                QString::number(classIndex),
                coordinate((first.x() + third.x()) / (2 * width)),
                coordinate((first.y() + third.y()) / (2 * height)),
                coordinate(std::abs(third.x() - first.x()) / width),
                coordinate(std::abs(third.y() - first.y()) / height)};
        for (const QString& name : config.classes[classIndex].keypoints) {
            if (!group.keypoints.contains(name)) {
                fields << QStringLiteral("0") << QStringLiteral("0");
                if (config.hasVisible) fields << QStringLiteral("0");
                continue;
            }
            const QPointF point = group.keypoints.value(name);
            fields << coordinate(
                              int(std::clamp(point.x(), 0.0, double(width))) /
                              double(width))
                   << coordinate(
                              int(std::clamp(point.y(), 0.0, double(height))) /
                              double(height));
            if (config.hasVisible)
                fields << QString::number(group.visibility.value(name));
        }
        for (int i = config.classes[classIndex].keypoints.size();
             i < maxKeypoints; ++i) {
            fields << QStringLiteral("0") << QStringLiteral("0");
            if (config.hasVisible) fields << QStringLiteral("0");
        }
        output.text += fields.join(QLatin1Char(' ')).toUtf8() + '\n';
        ++output.count;
    }
    return output;
}

CocoConversion convertCocoPose(const QVector<CocoImage>& images,
                               const PoseConfig& config) {
    CocoConversion output;
    if (!config.error.isEmpty() || config.classes.isEmpty()) {
        output.error = config.error.isEmpty()
                               ? QStringLiteral("Invalid pose classes")
                               : config.error;
        return output;
    }
    QStringList classNames;
    QJsonArray categories;
    for (int i = 0; i < config.classes.size(); ++i) {
        const PoseClass& item = config.classes[i];
        classNames.append(item.name);
        QJsonArray keypoints;
        for (const QString& name : item.keypoints) keypoints.append(name);
        categories.append(
                QJsonObject{{QStringLiteral("id"), i + 1},
                            {QStringLiteral("name"), item.name},
                            {QStringLiteral("supercategory"), QString()},
                            {QStringLiteral("keypoints"), keypoints},
                            {QStringLiteral("skeleton"), QJsonArray()}});
    }
    output = convertCoco(images, classNames, false);
    if (!output.error.isEmpty()) return output;
    QJsonArray annotations;
    for (int imageId = 0; imageId < images.size(); ++imageId) {
        const QJsonObject& document = images[imageId].document;
        const int width = document.value(QStringLiteral("imageWidth")).toInt();
        const int height =
                document.value(QStringLiteral("imageHeight")).toInt();
        QVector<PoseGroup> groups;
        output.error = collectPoseGroups(document, &groups);
        if (!output.error.isEmpty()) {
            output.error = images[imageId].fileName + QStringLiteral(": ") +
                           output.error;
            return output;
        }
        for (const PoseGroup& group : groups) {
            if (group.box.isEmpty()) {
                output.error =
                        QStringLiteral("%1: missing pose rectangle in group %2")
                                .arg(images[imageId].fileName)
                                .arg(group.id);
                return output;
            }
            const int classIndex = classNames.indexOf(group.boxLabel);
            if (classIndex < 0) {
                output.error =
                        QStringLiteral("%1: unknown pose class %2")
                                .arg(images[imageId].fileName, group.boxLabel);
                return output;
            }
            const QPointF first = group.box[0];
            const QPointF opposite =
                    group.box.size() == 2 ? group.box[1] : group.box[2];
            const double x0 = std::clamp(first.x(), 0.0, double(width));
            const double y0 = std::clamp(first.y(), 0.0, double(height));
            const double x1 = std::clamp(opposite.x(), 0.0, double(width));
            const double y1 = std::clamp(opposite.y(), 0.0, double(height));
            const double left = std::min(x0, x1), top = std::min(y0, y1);
            const double boxWidth = std::abs(x1 - x0),
                         boxHeight = std::abs(y1 - y0);
            QJsonArray keypoints;
            int numKeypoints = 0;
            for (const QString& name : config.classes[classIndex].keypoints) {
                if (!group.keypoints.contains(name)) {
                    keypoints.append(0);
                    keypoints.append(0);
                    if (config.hasVisible) keypoints.append(0);
                    continue;
                }
                const QPointF point = group.keypoints.value(name);
                keypoints.append(
                        int(std::clamp(point.x(), 0.0, double(width))));
                keypoints.append(
                        int(std::clamp(point.y(), 0.0, double(height))));
                if (config.hasVisible)
                    keypoints.append(group.visibility.value(name));
                ++numKeypoints;
            }
            annotations.append(QJsonObject{
                    {QStringLiteral("id"), annotations.size()},
                    {QStringLiteral("image_id"), imageId},
                    {QStringLiteral("category_id"), classIndex + 1},
                    {QStringLiteral("bbox"),
                     QJsonArray{left, top, boxWidth, boxHeight}},
                    {QStringLiteral("area"), boxWidth * boxHeight},
                    {QStringLiteral("iscrowd"), 0},
                    {QStringLiteral("keypoints"), keypoints},
                    {QStringLiteral("num_keypoints"), numKeypoints},
                    {QStringLiteral("ignore"), int(group.difficult)},
                    {QStringLiteral("segmentation"), QJsonArray()}});
        }
    }
    output.document.insert(QStringLiteral("categories"), categories);
    output.document.insert(QStringLiteral("annotations"), annotations);
    output.count = annotations.size();
    return output;
}
