// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationPpocr.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLineF>
#include <QPainter>
#include <QPolygonF>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTemporaryDir>
#include <QTransform>
#include <QVector>
#include <cmath>
#include <limits>

namespace {
bool point(const QJsonValue& value, double* x, double* y) {
    const QJsonArray pair = value.toArray();
    if (pair.size() != 2 || !pair[0].isDouble() || !pair[1].isDouble())
        return false;
    *x = pair[0].toDouble();
    *y = pair[1].toDouble();
    return std::isfinite(*x) && std::isfinite(*y);
}

QString shapeType(const QJsonArray& points) {
    if (points.size() != 4)
        return points.size() >= 3 ? QStringLiteral("polygon") : QString();
    double x0, y0, x1, y1, x2, y2, x3, y3;
    if (!point(points[0], &x0, &y0) || !point(points[1], &x1, &y1) ||
        !point(points[2], &x2, &y2) || !point(points[3], &x3, &y3))
        return {};
    const bool rectangle = qFuzzyCompare(x0, x3) && qFuzzyCompare(x1, x2) &&
                           qFuzzyCompare(y0, y1) && qFuzzyCompare(y2, y3);
    return rectangle ? QStringLiteral("rectangle")
                     : QStringLiteral("quadrilateral");
}

bool annotation(const QJsonValue& value, PpocrMode mode, QJsonObject* shape) {
    if (!value.isObject()) return false;
    const QJsonObject source = value.toObject();
    const QJsonArray points = source.value(QStringLiteral("points")).toArray();
    const QString type = shapeType(points);
    if (type.isEmpty()) return false;
    QString label = source.value(QStringLiteral("label")).toString().trimmed();
    if (label.isEmpty()) label = QStringLiteral("text");
    const QString transcription =
            source.value(QStringLiteral("transcription")).toString();
    shape->insert(QStringLiteral("label"), label);
    shape->insert(QStringLiteral("description"), transcription);
    shape->insert(QStringLiteral("points"), points);
    shape->insert(QStringLiteral("shape_type"), type);
    shape->insert(QStringLiteral("group_id"),
                  source.value(QStringLiteral("id")).isDouble()
                          ? source.value(QStringLiteral("id"))
                          : QJsonValue::Null);
    shape->insert(QStringLiteral("difficult"),
                  source.value(QStringLiteral("difficult")).toBool(false));
    if (mode == PpocrMode::Kie)
        shape->insert(QStringLiteral("kie_linking"),
                      source.value(QStringLiteral("linking")).isArray()
                              ? source.value(QStringLiteral("linking"))
                              : QJsonArray());
    return true;
}

QVector<QPointF> pointsFromJson(const QJsonValue& value) {
    QVector<QPointF> points;
    for (const QJsonValue& item : value.toArray()) {
        const QJsonArray pair = item.toArray();
        if (pair.size() != 2 || !pair[0].isDouble() || !pair[1].isDouble())
            return {};
        const QPointF point(pair[0].toDouble(), pair[1].toDouble());
        if (!std::isfinite(point.x()) || !std::isfinite(point.y())) return {};
        points.push_back(point);
    }
    return points;
}

QVector<QPointF> minimumAreaQuad(const QVector<QPointF>& points) {
    if (points.size() <= 4) return points;
    qreal bestArea = std::numeric_limits<qreal>::max();
    QPolygonF best;
    for (int edge = 0; edge < points.size(); ++edge) {
        const QPointF delta = points[(edge + 1) % points.size()] - points[edge];
        if (qFuzzyIsNull(delta.x()) && qFuzzyIsNull(delta.y())) continue;
        const qreal angle = -std::atan2(delta.y(), delta.x());
        const QTransform rotate = QTransform().rotateRadians(angle);
        const QTransform inverse = QTransform().rotateRadians(-angle);
        qreal minX = std::numeric_limits<qreal>::max();
        qreal minY = std::numeric_limits<qreal>::max();
        qreal maxX = -std::numeric_limits<qreal>::max();
        qreal maxY = -std::numeric_limits<qreal>::max();
        for (const QPointF& point : points) {
            const QPointF rotated = rotate.map(point);
            minX = qMin(minX, rotated.x());
            minY = qMin(minY, rotated.y());
            maxX = qMax(maxX, rotated.x());
            maxY = qMax(maxY, rotated.y());
        }
        const qreal area = (maxX - minX) * (maxY - minY);
        if (area >= bestArea) continue;
        bestArea = area;
        best = QPolygonF(QVector<QPointF>{inverse.map(QPointF(minX, minY)),
                                          inverse.map(QPointF(maxX, minY)),
                                          inverse.map(QPointF(maxX, maxY)),
                                          inverse.map(QPointF(minX, maxY))});
    }
    // QPolygonF is a QVector<QPointF> (Qt 5) / QList<QPointF> (Qt 6); copy
    // the base rather than use the iterator-range ctor (Qt 5.14+).
    return best.isEmpty() ? QVector<QPointF>() : QVector<QPointF>(best);
}

QVector<QPointF> normalizeQuad(QVector<QPointF> points,
                               const QSize& imageSize) {
    for (QPointF& point : points) {
        point.setX(qBound<qreal>(0.0, point.x(), imageSize.width() - 1.0));
        point.setY(qBound<qreal>(0.0, point.y(), imageSize.height() - 1.0));
    }
    if (points.size() == 2) {
        const QRectF bounds(QPointF(qMin(points[0].x(), points[1].x()),
                                    qMin(points[0].y(), points[1].y())),
                            QPointF(qMax(points[0].x(), points[1].x()),
                                    qMax(points[0].y(), points[1].y())));
        points = {bounds.topLeft(), bounds.topRight(), bounds.bottomRight(),
                  bounds.bottomLeft()};
    } else if (points.size() > 4) {
        points = minimumAreaQuad(points);
    }
    return points;
}

QImage rotateCrop(const QImage& source, const QVector<QPointF>& points) {
    if (points.size() != 4) return {};
    QVector<QPointF> ordered = points;
    qreal orientation = 0.0;
    for (int index = -1; index < 3; ++index) {
        const QPointF& current = ordered[index < 0 ? 3 : index];
        const QPointF& next = ordered[index + 1];
        orientation +=
                -0.5 * (next.y() + current.y()) * (next.x() - current.x());
    }
    if (orientation < 0.0) std::swap(ordered[1], ordered[3]);
    const qreal width = qMax(QLineF(ordered[0], ordered[1]).length(),
                             QLineF(ordered[2], ordered[3]).length());
    const qreal height = qMax(QLineF(ordered[0], ordered[3]).length(),
                              QLineF(ordered[1], ordered[2]).length());
    const int cropWidth = qMax(1, qRound(width));
    const int cropHeight = qMax(1, qRound(height));
    QPolygonF target(QVector<QPointF>{QPointF(0, 0), QPointF(cropWidth, 0),
                                      QPointF(cropWidth, cropHeight),
                                      QPointF(0, cropHeight)});
    QTransform transform;
    if (!QTransform::quadToQuad(QPolygonF(ordered), target, transform))
        return {};
    QImage crop(cropWidth, cropHeight, QImage::Format_ARGB32);
    crop.fill(Qt::white);
    QPainter painter(&crop);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    painter.setTransform(transform);
    painter.drawImage(QPointF(0, 0), source);
    painter.end();
    if (height / qMax<qreal>(width, 1.0) >= 1.5)
        crop = crop.transformed(QTransform().rotate(90),
                                Qt::SmoothTransformation);
    return crop;
}

QByteArray recAnnotations(const QJsonArray& annotations) {
    return QJsonDocument(annotations).toJson(QJsonDocument::Compact);
}

bool writeAtomic(const QString& path, const QByteArray& bytes, QString* error) {
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit()) {
        if (error) *error = file.errorString();
        return false;
    }
    return true;
}

bool movePath(const QString& from, const QString& to) {
    return QDir().rename(from, to);
}

bool removePath(const QString& path) {
    const QFileInfo info(path);
    if (!info.exists()) return true;
    return info.isDir() ? QDir(path).removeRecursively() : QFile::remove(path);
}

bool commitRecArtifacts(const QString& stagedDirectory,
                        const QString& outputDirectory,
                        QString* error) {
    const QStringList names{QStringLiteral("crop_img"),
                            QStringLiteral("Label.txt"),
                            QStringLiteral("rec_gt.txt")};
    const QString backupDirectory =
            QDir(stagedDirectory).filePath(QStringLiteral("previous"));
    if (!QDir().mkpath(backupDirectory)) {
        *error = QStringLiteral("Cannot stage existing PPOCR artifacts");
        return false;
    }
    QVector<QString> backedUp;
    QVector<QString> committed;
    for (const QString& name : names) {
        const QString destination = QDir(outputDirectory).filePath(name);
        if (!QFileInfo(destination).exists()) continue;
        if (!movePath(destination, QDir(backupDirectory).filePath(name))) {
            *error = QStringLiteral("Cannot stage existing PPOCR artifact: %1")
                             .arg(name);
            for (const QString& old : backedUp)
                movePath(QDir(backupDirectory).filePath(old),
                         QDir(outputDirectory).filePath(old));
            return false;
        }
        backedUp.push_back(name);
    }
    for (const QString& name : names) {
        if (movePath(QDir(stagedDirectory).filePath(name),
                     QDir(outputDirectory).filePath(name))) {
            committed.push_back(name);
            continue;
        }
        *error = QStringLiteral("Cannot commit PPOCR artifact: %1").arg(name);
        for (const QString& written : committed)
            removePath(QDir(outputDirectory).filePath(written));
        for (const QString& old : backedUp)
            movePath(QDir(backupDirectory).filePath(old),
                     QDir(outputDirectory).filePath(old));
        return false;
    }
    return true;
}
}  // namespace

PpocrImport importPpocr(const QByteArray& labels, PpocrMode mode) {
    PpocrImport result;
    const QList<QByteArray> lines = labels.split('\n');
    for (int index = 0; index < lines.size(); ++index) {
        const QByteArray line = lines[index].trimmed();
        if (line.isEmpty()) continue;
        const int tab = line.indexOf('\t');
        if (tab <= 0) {
            result.error =
                    QStringLiteral("PPOCR line %1 needs filename and JSON")
                            .arg(index + 1);
            return result;
        }
        const QString name =
                QFileInfo(QString::fromUtf8(line.left(tab))).fileName();
        if (name.isEmpty() || result.documents.contains(name)) {
            result.error = QStringLiteral("Duplicate PPOCR image at line %1")
                                   .arg(index + 1);
            return result;
        }
        QJsonParseError parse;
        const QJsonDocument data =
                QJsonDocument::fromJson(line.mid(tab + 1), &parse);
        if (parse.error != QJsonParseError::NoError || !data.isArray()) {
            result.error = QStringLiteral("Invalid PPOCR JSON at line %1")
                                   .arg(index + 1);
            return result;
        }
        QJsonArray shapes;
        for (const QJsonValue& item : data.array()) {
            QJsonObject shape;
            if (!annotation(item, mode, &shape)) {
                result.error =
                        QStringLiteral("Invalid PPOCR annotation at line %1")
                                .arg(index + 1);
                return result;
            }
            shapes.append(shape);
            ++result.count;
        }
        QJsonObject document{
                {QStringLiteral("version"), QStringLiteral("4.0.6")},
                {QStringLiteral("flags"), QJsonObject()},
                {QStringLiteral("shapes"), shapes},
                {QStringLiteral("imagePath"), name},
                {QStringLiteral("imageData"), QJsonValue::Null}};
        result.documents.insert(name, document);
    }
    return result;
}

PpocrConversion convertPpocr(const QVector<CocoImage>& images, PpocrMode mode) {
    PpocrConversion result;
    for (const CocoImage& image : images) {
        if (image.fileName.isEmpty()) {
            result.error = QStringLiteral("PPOCR image has no filename");
            return result;
        }
        QJsonArray annotations;
        for (const QJsonValue& value :
             image.document.value(QStringLiteral("shapes")).toArray()) {
            const QJsonObject shape = value.toObject();
            const QString type =
                    shape.value(QStringLiteral("shape_type")).toString();
            if (type != QStringLiteral("rectangle") &&
                type != QStringLiteral("rotation") &&
                type != QStringLiteral("polygon") &&
                type != QStringLiteral("quadrilateral"))
                continue;
            const QJsonArray points =
                    shape.value(QStringLiteral("points")).toArray();
            if (points.size() < 3) {
                result.error = QStringLiteral("Invalid PPOCR shape in %1")
                                       .arg(image.fileName);
                return result;
            }
            QJsonObject annotation;
            annotation.insert(
                    QStringLiteral("transcription"),
                    shape.value(QStringLiteral("description")).toString());
            annotation.insert(QStringLiteral("points"), points);
            annotation.insert(
                    QStringLiteral("difficult"),
                    shape.value(QStringLiteral("difficult")).toBool(false));
            if (mode == PpocrMode::Kie) {
                annotation.insert(
                        QStringLiteral("label"),
                        shape.value(QStringLiteral("label")).toString());
                annotation.insert(
                        QStringLiteral("id"),
                        shape.value(QStringLiteral("group_id")).isDouble()
                                ? shape.value(QStringLiteral("group_id"))
                                : QJsonValue(0));
                annotation.insert(
                        QStringLiteral("linking"),
                        shape.value(QStringLiteral("kie_linking")).isArray()
                                ? shape.value(QStringLiteral("kie_linking"))
                                : QJsonArray());
            }
            annotations.append(annotation);
            ++result.count;
        }
        if (annotations.isEmpty()) continue;
        result.text += image.fileName.toUtf8();
        result.text += '\t';
        result.text +=
                QJsonDocument(annotations).toJson(QJsonDocument::Compact);
        result.text += '\n';
    }
    return result;
}

PpocrRecExport exportPpocrRecDataset(const QVector<PpocrDatasetImage>& images,
                                     const QString& outputDirectory) {
    PpocrRecExport result;
    if (outputDirectory.isEmpty()) {
        result.error = QStringLiteral("PPOCR output directory is empty");
        return result;
    }
    QDir output(outputDirectory);
    if (!output.exists() && !QDir().mkpath(outputDirectory)) {
        result.error = QStringLiteral("Cannot create PPOCR output directory");
        return result;
    }
    QTemporaryDir staged(output.filePath(QStringLiteral(".ppocr-rec-XXXXXX")));
    if (!staged.isValid() ||
        !QDir(staged.path()).mkpath(QStringLiteral("crop_img"))) {
        result.error = QStringLiteral("Cannot stage PPOCR crop directory");
        return result;
    }

    QByteArray labels;
    QByteArray recGroundTruth;
    for (const PpocrDatasetImage& input : images) {
        QImage image(input.imagePath);
        if (image.isNull()) {
            result.error = QStringLiteral("Cannot read PPOCR image: %1")
                                   .arg(input.imagePath);
            return result;
        }
        const QFileInfo info(input.imagePath);
        const QJsonArray sourceShapes =
                input.document.value(QStringLiteral("shapes")).toArray();
        QJsonArray annotations;
        int imageCropIndex = 0;
        for (const QJsonValue& value : sourceShapes) {
            const QJsonObject shape = value.toObject();
            const QString type =
                    shape.value(QStringLiteral("shape_type")).toString();
            if (type != QStringLiteral("rectangle") &&
                type != QStringLiteral("rotation") &&
                type != QStringLiteral("polygon") &&
                type != QStringLiteral("quadrilateral"))
                continue;
            QVector<QPointF> points = normalizeQuad(
                    pointsFromJson(shape.value(QStringLiteral("points"))),
                    image.size());
            if (points.size() != 4) {
                result.error = QStringLiteral("Invalid PPOCR shape in %1")
                                       .arg(input.imagePath);
                return result;
            }
            QJsonArray jsonPoints;
            for (const QPointF& point : points)
                jsonPoints.append(
                        QJsonArray{qRound(point.x()), qRound(point.y())});
            annotations.append(QJsonObject{
                    {QStringLiteral("transcription"),
                     shape.value(QStringLiteral("description")).toString()},
                    {QStringLiteral("points"), jsonPoints},
                    {QStringLiteral("difficult"),
                     shape.value(QStringLiteral("difficult")).toBool(false)}});
            const QImage crop = rotateCrop(image, points);
            if (crop.isNull()) continue;
            const QString cropName = QStringLiteral("%1_crop_%2.jpg")
                                             .arg(info.completeBaseName())
                                             .arg(imageCropIndex++);
            QSaveFile cropFile(QDir(staged.path())
                                       .filePath(QStringLiteral("crop_img/%1")
                                                         .arg(cropName)));
            if (!cropFile.open(QIODevice::WriteOnly) ||
                !crop.save(&cropFile, "JPG", 95) || !cropFile.commit()) {
                result.error = QStringLiteral("Cannot write PPOCR crop: %1")
                                       .arg(cropName);
                return result;
            }
            recGroundTruth += QByteArray("crop_img/") + cropName.toUtf8() +
                              '\t' +
                              shape.value(QStringLiteral("description"))
                                      .toString()
                                      .toUtf8() +
                              '\n';
            ++result.cropCount;
            ++result.annotationCount;
        }
        if (!annotations.isEmpty()) {
            const QString relativeDirectory = info.dir().dirName();
            labels += relativeDirectory.toUtf8() + '/' +
                      info.fileName().toUtf8() + '\t' +
                      recAnnotations(annotations) + '\n';
        }
    }
    if (!writeAtomic(QDir(staged.path()).filePath(QStringLiteral("Label.txt")),
                     labels, &result.error) ||
        !writeAtomic(QDir(staged.path()).filePath(QStringLiteral("rec_gt.txt")),
                     recGroundTruth, &result.error))
        return result;
    if (!commitRecArtifacts(staged.path(), outputDirectory, &result.error)) {
        return result;
    }
    return result;
}
