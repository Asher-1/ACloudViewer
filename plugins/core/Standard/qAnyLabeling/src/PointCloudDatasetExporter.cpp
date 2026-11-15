// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudDatasetExporter.h"

#include <zlib.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QXmlStreamWriter>
#include <QtEndian>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
struct ExportClass {
    int id = 0;
    QString name;
    QString color;
};

struct Entry {
    PointCloudFrame frame;
    QString name;
    QVector<PointCloudCuboid> cuboids;
};

QJsonArray vectorJson(const CuboidVector& value) {
    QJsonArray result;
    for (double item : value) result.append(item);
    return result;
}

QJsonObject xyzObject(const CuboidVector& value) {
    return QJsonObject{{QStringLiteral("x"), value[0]},
                       {QStringLiteral("y"), value[1]},
                       {QStringLiteral("z"), value[2]}};
}

QString safeName(const QString& path, int index, QSet<QString>* used) {
    QString name = QFileInfo(path).completeBaseName();
    name.replace(QRegularExpression(QStringLiteral("[\\r\\n\\\\/]")),
                 QStringLiteral("_"));
    name = name.trimmed();
    if (name.isEmpty() || name == QStringLiteral(".") ||
        name == QStringLiteral(".."))
        name = QString::number(index);
    if (used->contains(name))
        name = QStringLiteral("%1_%2").arg(index).arg(name);
    while (used->contains(name)) name.prepend(QLatin1Char('_'));
    used->insert(name);
    return name;
}

bool cancelled(const std::function<bool(int, int)>& progress,
               int current,
               int total) {
    return progress && !progress(current, total);
}

struct ZipPayload {
    QByteArray name;
    QByteArray data;
};

struct ZipEntry {
    QByteArray name;
    quint32 offset = 0;
    quint32 size = 0;
    quint32 compressed = 0;
    quint32 crc = 0;
};

bool write16(QIODevice& file, quint16 value) {
    const char bytes[] = {char(value), char(value >> 8)};
    return file.write(bytes, 2) == 2;
}

bool write32(QIODevice& file, quint32 value) {
    return write16(file, quint16(value)) && write16(file, quint16(value >> 16));
}

QByteArray deflateRaw(const QByteArray& input, bool* success) {
    *success = false;
    if (input.size() > std::numeric_limits<uInt>::max()) return {};
    z_stream stream{};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
        return {};
    stream.next_in =
            reinterpret_cast<Bytef*>(const_cast<char*>(input.constData()));
    stream.avail_in = static_cast<uInt>(input.size());
    QByteArray result;
    char output[65536];
    int status = Z_OK;
    while (status != Z_STREAM_END) {
        stream.next_out = reinterpret_cast<Bytef*>(output);
        stream.avail_out = sizeof(output);
        status = deflate(&stream, Z_FINISH);
        if (status != Z_OK && status != Z_STREAM_END) break;
        const int produced = int(sizeof(output) - stream.avail_out);
        if (produced > 0) result.append(output, produced);
    }
    const bool complete = status == Z_STREAM_END &&
                          stream.total_in == static_cast<uLong>(input.size());
    deflateEnd(&stream);
    if (!complete) return {};
    *success = true;
    return result;
}

bool writeZipArchive(const QString& path,
                     const QVector<ZipPayload>& payloads,
                     const std::function<bool()>& keepWriting,
                     QString* error) {
    QSaveFile zip(path);
    zip.setDirectWriteFallback(false);
    if (!zip.open(QIODevice::WriteOnly)) {
        if (error) *error = zip.errorString();
        return false;
    }
    QVector<ZipEntry> entries;
    bool success = true;
    for (const ZipPayload& payload : payloads) {
        if (keepWriting && !keepWriting()) {
            if (error) *error = QStringLiteral("Point-cloud export cancelled");
            success = false;
            break;
        }
        if (payload.name.isEmpty() ||
            payload.name.size() > std::numeric_limits<quint16>::max() ||
            payload.data.size() > std::numeric_limits<quint32>::max() ||
            entries.size() >= std::numeric_limits<quint16>::max() ||
            zip.pos() > std::numeric_limits<quint32>::max()) {
            success = false;
            break;
        }
        bool compressedOk = false;
        const QByteArray compressed = deflateRaw(payload.data, &compressedOk);
        if (!compressedOk ||
            compressed.size() > std::numeric_limits<quint32>::max()) {
            success = false;
            break;
        }
        ZipEntry entry;
        entry.name = payload.name;
        entry.offset = quint32(zip.pos());
        entry.size = quint32(payload.data.size());
        entry.compressed = quint32(compressed.size());
        entry.crc = crc32(
                0L, reinterpret_cast<const Bytef*>(payload.data.constData()),
                static_cast<uInt>(payload.data.size()));
        success = write32(zip, 0x04034b50) && write16(zip, 20) &&
                  write16(zip, 0x0800) && write16(zip, 8) && write16(zip, 0) &&
                  write16(zip, 0) && write32(zip, entry.crc) &&
                  write32(zip, entry.compressed) && write32(zip, entry.size) &&
                  write16(zip, quint16(entry.name.size())) && write16(zip, 0) &&
                  zip.write(entry.name) == entry.name.size() &&
                  zip.write(compressed) == compressed.size();
        if (!success) break;
        entries.append(entry);
    }
    const qint64 centralOffset = zip.pos();
    if (success) {
        for (const ZipEntry& entry : entries) {
            success = write32(zip, 0x02014b50) && write16(zip, 20) &&
                      write16(zip, 20) && write16(zip, 0x0800) &&
                      write16(zip, 8) && write16(zip, 0) && write16(zip, 0) &&
                      write32(zip, entry.crc) &&
                      write32(zip, entry.compressed) &&
                      write32(zip, entry.size) &&
                      write16(zip, quint16(entry.name.size())) &&
                      write16(zip, 0) && write16(zip, 0) && write16(zip, 0) &&
                      write16(zip, 0) && write32(zip, 0) &&
                      write32(zip, entry.offset) &&
                      zip.write(entry.name) == entry.name.size();
            if (!success) break;
        }
    }
    const qint64 centralSize = zip.pos() - centralOffset;
    success = success &&
              entries.size() <= std::numeric_limits<quint16>::max() &&
              centralOffset <= std::numeric_limits<quint32>::max() &&
              centralSize <= std::numeric_limits<quint32>::max() &&
              write32(zip, 0x06054b50) && write16(zip, 0) && write16(zip, 0) &&
              write16(zip, quint16(entries.size())) &&
              write16(zip, quint16(entries.size())) &&
              write32(zip, quint32(centralSize)) &&
              write32(zip, quint32(centralOffset)) && write16(zip, 0);
    if (success && keepWriting && !keepWriting()) {
        if (error) *error = QStringLiteral("Point-cloud export cancelled");
        success = false;
    }
    success = success && zip.commit();
    if (!success) {
        if (error && error->isEmpty())
            *error = QStringLiteral("Could not create ZIP archive");
    }
    return success;
}

QByteArray pointCloudPcd(const PointCloudFrame& frame, QString* error) {
    const bool hasRgb =
            frame.rgb.size() == frame.points.size() && !frame.rgb.isEmpty();
    const bool hasIntensity = frame.hasIntensity;
    QByteArray result;
    QStringList fields{QStringLiteral("x"), QStringLiteral("y"),
                       QStringLiteral("z")};
    if (hasIntensity) fields.append(QStringLiteral("intensity"));
    if (hasRgb) fields.append(QStringLiteral("rgb"));
    QStringList sizes;
    QStringList types;
    QStringList counts;
    for (int i = 0; i < fields.size(); ++i) {
        sizes.append(QStringLiteral("4"));
        types.append(fields.at(i) == QStringLiteral("rgb")
                             ? QStringLiteral("U")
                             : QStringLiteral("F"));
        counts.append(QStringLiteral("1"));
    }
    result += QByteArray("VERSION .7\nFIELDS ") +
              fields.join(QLatin1Char(' ')).toUtf8() + QByteArray("\nSIZE ") +
              sizes.join(QLatin1Char(' ')).toUtf8() + QByteArray("\nTYPE ") +
              types.join(QLatin1Char(' ')).toUtf8() + QByteArray("\nCOUNT ") +
              counts.join(QLatin1Char(' ')).toUtf8() + QByteArray("\nWIDTH ") +
              QByteArray::number(frame.points.size()) +
              QByteArray("\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS ") +
              QByteArray::number(frame.points.size()) +
              QByteArray("\nDATA binary\n");
    result.reserve(result.size() + frame.points.size() * fields.size() * 4);
    for (int i = 0; i < frame.points.size(); ++i) {
        const QVector4D point = frame.points.at(i);
        if (!std::isfinite(point.x()) || !std::isfinite(point.y()) ||
            !std::isfinite(point.z())) {
            if (error)
                *error = QStringLiteral("Point coordinates must be finite");
            return {};
        }
        const auto appendFloat = [&](float value) {
            quint32 bits;
            std::memcpy(&bits, &value, sizeof(bits));
            const quint32 little = qToLittleEndian(bits);
            result.append(reinterpret_cast<const char*>(&little),
                          sizeof(little));
        };
        appendFloat(point.x());
        appendFloat(point.y());
        appendFloat(point.z());
        if (hasIntensity) appendFloat(point.w());
        if (hasRgb) {
            const QColor color = frame.rgb.at(i);
            const quint32 packed = (quint32(color.red()) << 16) |
                                   (quint32(color.green()) << 8) |
                                   quint32(color.blue());
            const quint32 little = qToLittleEndian(packed);
            result.append(reinterpret_cast<const char*>(&little),
                          sizeof(little));
        }
    }
    return result;
}

QJsonObject cuboidJson(const PointCloudCuboid& cuboid,
                       int labelId,
                       int annotationId) {
    return QJsonObject{
            {QStringLiteral("id"), annotationId},
            {QStringLiteral("type"), QStringLiteral("cuboid_3d")},
            {QStringLiteral("attributes"),
             QJsonObject{{QStringLiteral("occluded"), cuboid.occluded},
                         {QStringLiteral("locked"), cuboid.locked}}},
            {QStringLiteral("group"), 0},
            {QStringLiteral("label_id"), labelId},
            {QStringLiteral("position"), vectorJson(cuboid.center)},
            {QStringLiteral("rotation"), vectorJson(cuboid.rotation)},
            {QStringLiteral("scale"), vectorJson(cuboid.size)}};
}

QByteArray datumaro(const QVector<Entry>& entries,
                    const QVector<ExportClass>& classes) {
    QJsonArray items;
    QMap<int, int> labelIds;
    for (int i = 0; i < classes.size(); ++i) labelIds[classes.at(i).id] = i;
    int annotationId = 0;
    for (int index = 0; index < entries.size(); ++index) {
        const Entry& entry = entries.at(index);
        QJsonArray annotations;
        for (const PointCloudCuboid& cuboid : entry.cuboids) {
            annotations.append(cuboidJson(
                    cuboid, labelIds.value(cuboid.classId), ++annotationId));
        }
        QJsonObject attr;
        attr.insert(QStringLiteral("frame"), index);
        QJsonObject pointCloud;
        pointCloud.insert(QStringLiteral("path"),
                          QStringLiteral("%1.pcd").arg(entry.name));
        QJsonObject item;
        item.insert(QStringLiteral("id"), entry.name);
        item.insert(QStringLiteral("attr"), attr);
        item.insert(QStringLiteral("point_cloud"), pointCloud);
        item.insert(QStringLiteral("annotations"), annotations);
        items.append(item);
    }
    QJsonArray labels;
    for (const ExportClass& item : classes)
        labels.append(
                QJsonObject{{QStringLiteral("name"), item.name},
                            {QStringLiteral("parent"), QString()},
                            {QStringLiteral("attributes"), QJsonArray{}}});
    QJsonObject labelCategory;
    labelCategory.insert(QStringLiteral("labels"), labels);
    labelCategory.insert(QStringLiteral("label_groups"), QJsonArray{});
    labelCategory.insert(
            QStringLiteral("attributes"),
            QJsonArray{QStringLiteral("occluded"), QStringLiteral("locked")});
    QJsonObject categories;
    categories.insert(QStringLiteral("label"), labelCategory);
    QJsonObject root;
    root.insert(QStringLiteral("dm_format_version"), QStringLiteral("1.0"));
    root.insert(QStringLiteral("media_type"), 6);
    root.insert(QStringLiteral("infos"), QJsonObject{});
    root.insert(QStringLiteral("categories"), categories);
    root.insert(QStringLiteral("items"), items);
    return QJsonDocument(root).toJson(QJsonDocument::Indented);
}

QByteArray kitti(const QVector<Entry>& entries,
                 const QVector<ExportClass>& classes) {
    QMap<int, QString> names;
    for (const ExportClass& item : classes) names[item.id] = item.name;
    QByteArray output;
    QXmlStreamWriter xml(&output);
    xml.setAutoFormatting(true);
    xml.writeStartDocument(QStringLiteral("1.0"), true);
    xml.writeDTD(QStringLiteral("<!DOCTYPE boost_serialization>"));
    xml.writeStartElement(QStringLiteral("boost_serialization"));
    xml.writeAttribute(QStringLiteral("version"), QStringLiteral("9"));
    xml.writeAttribute(QStringLiteral("signature"),
                       QStringLiteral("serialization::archive"));
    xml.writeStartElement(QStringLiteral("tracklets"));
    xml.writeAttribute(QStringLiteral("version"), QStringLiteral("0"));
    xml.writeAttribute(QStringLiteral("tracking_level"), QStringLiteral("0"));
    xml.writeAttribute(QStringLiteral("class_id"), QStringLiteral("0"));
    int count = 0;
    for (const Entry& entry : entries) count += entry.cuboids.size();
    xml.writeTextElement(QStringLiteral("count"), QString::number(count));
    xml.writeTextElement(QStringLiteral("item_version"), QStringLiteral("1"));
    for (int index = 0; index < entries.size(); ++index) {
        for (const PointCloudCuboid& cuboid : entries.at(index).cuboids) {
            xml.writeStartElement(QStringLiteral("item"));
            xml.writeTextElement(QStringLiteral("objectType"),
                                 names.value(cuboid.classId));
            xml.writeTextElement(QStringLiteral("h"),
                                 QString::number(cuboid.size[0], 'g', 17));
            xml.writeTextElement(QStringLiteral("w"),
                                 QString::number(cuboid.size[1], 'g', 17));
            xml.writeTextElement(QStringLiteral("l"),
                                 QString::number(cuboid.size[2], 'g', 17));
            xml.writeTextElement(QStringLiteral("first_frame"),
                                 QString::number(index));
            xml.writeStartElement(QStringLiteral("poses"));
            xml.writeTextElement(QStringLiteral("count"), QStringLiteral("1"));
            xml.writeTextElement(QStringLiteral("item_version"),
                                 QStringLiteral("0"));
            xml.writeStartElement(QStringLiteral("item"));
            const QStringList keys{QStringLiteral("tx"), QStringLiteral("ty"),
                                   QStringLiteral("tz"), QStringLiteral("rx"),
                                   QStringLiteral("ry"), QStringLiteral("rz")};
            const CuboidVector values{
                    {cuboid.center[0], cuboid.center[1], cuboid.center[2]}};
            const CuboidVector angles{{cuboid.rotation[0], cuboid.rotation[1],
                                       cuboid.rotation[2]}};
            const CuboidVector all{{values[0], values[1], values[2]}};
            for (int i = 0; i < 3; ++i)
                xml.writeTextElement(keys.at(i),
                                     QString::number(all[i], 'g', 17));
            for (int i = 0; i < 3; ++i)
                xml.writeTextElement(keys.at(i + 3),
                                     QString::number(angles[i], 'g', 17));
            xml.writeTextElement(QStringLiteral("state"), QStringLiteral("2"));
            xml.writeTextElement(QStringLiteral("occlusion"),
                                 cuboid.occluded ? QStringLiteral("1")
                                                 : QStringLiteral("0"));
            xml.writeTextElement(QStringLiteral("occlusion_kf"),
                                 QStringLiteral("0"));
            xml.writeTextElement(QStringLiteral("truncation"),
                                 QStringLiteral("0"));
            xml.writeTextElement(QStringLiteral("amt_occlusion"),
                                 QStringLiteral("-1"));
            xml.writeTextElement(QStringLiteral("amt_border_l"),
                                 QStringLiteral("-1"));
            xml.writeTextElement(QStringLiteral("amt_border_r"),
                                 QStringLiteral("-1"));
            xml.writeTextElement(QStringLiteral("amt_occlusion_kf"),
                                 QStringLiteral("-1"));
            xml.writeTextElement(QStringLiteral("amt_border_kf"),
                                 QStringLiteral("-1"));
            xml.writeEndElement();
            xml.writeEndElement();
            xml.writeTextElement(QStringLiteral("finished"),
                                 QStringLiteral("1"));
            xml.writeEndElement();
        }
    }
    xml.writeEndElement();
    xml.writeEndElement();
    xml.writeEndDocument();
    return output;
}

}  // namespace

bool exportPointCloudDataset(const PointCloudDatasetExportOptions& options,
                             const std::function<bool(int, int)>& progress,
                             QString* error) {
    const auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    if (error) error->clear();
    if (options.outputPath.isEmpty() ||
        !options.outputPath.endsWith(QStringLiteral(".zip"),
                                     Qt::CaseInsensitive))
        return fail(QStringLiteral("Point-cloud export requires a ZIP path"));
    if (options.frames.isEmpty())
        return fail(QStringLiteral("No point-cloud frames to export"));
    if (options.format != QStringLiteral("datumaro") &&
        options.format != QStringLiteral("kitti_raw") &&
        options.format != QStringLiteral("sly_pointcloud"))
        return fail(QStringLiteral("Unknown point-cloud export format"));
    QDir parent = QFileInfo(options.outputPath).absoluteDir();
    if (!parent.exists())
        return fail(QStringLiteral("Export parent folder does not exist"));
    if (QFileInfo(options.outputPath).isSymLink())
        return fail(
                QStringLiteral("Export destination must be a regular file"));

    QVector<Entry> entries;
    QSet<QString> used;
    QMap<int, ExportClass> classMap;
    for (auto it = options.classNames.cbegin(); it != options.classNames.cend();
         ++it) {
        if (it.key() == 0) continue;
        if (it.key() < 1 || it.key() > 65535 || it.value().trimmed().isEmpty())
            return fail(QStringLiteral(
                    "Export classes require valid IDs and nonempty names"));
        classMap.insert(
                it.key(),
                {it.key(), it.value(),
                 options.classColors
                         .value(it.key(), QColor(QStringLiteral("#808080")))
                         .name(QColor::HexRgb)});
    }
    for (int index = 0; index < options.frames.size(); ++index) {
        const PointCloudFrame& frame = options.frames.at(index);
        const QFileInfo destination(options.outputPath);
        for (const QString& path :
             {frame.path, frame.labelPath, frame.cuboidPath}) {
            if (path.isEmpty()) continue;
            const QFileInfo source(path);
            if (destination.absoluteFilePath() == source.absoluteFilePath() ||
                (!destination.canonicalFilePath().isEmpty() &&
                 destination.canonicalFilePath() == source.canonicalFilePath()))
                return fail(QStringLiteral(
                        "The export must not overwrite a source file"));
        }
        if (options.includePointCloud &&
            (frame.points.isEmpty() ||
             (!frame.rgb.isEmpty() && frame.rgb.size() != frame.points.size())))
            return fail(
                    QStringLiteral("Point-cloud payload requires points and "
                                   "matching RGB records"));
        Entry entry{frame, safeName(frame.path, index, &used), frame.cuboids};
        QSet<int> cuboidIds;
        for (const PointCloudCuboid& cuboid : entry.cuboids) {
            if (!cuboid.valid(error)) return false;
            if (cuboidIds.contains(cuboid.id))
                return fail(QStringLiteral("Duplicate cuboid IDs"));
            cuboidIds.insert(cuboid.id);
            if (!classMap.contains(cuboid.classId))
                classMap.insert(cuboid.classId,
                                {cuboid.classId,
                                 QStringLiteral("Class %1").arg(cuboid.classId),
                                 QStringLiteral("#808080")});
        }
        entries.append(entry);
    }
    QVector<ExportClass> classes = classMap.values().toVector();
    if (classes.isEmpty())
        classes.append(
                {1, QStringLiteral("Object"), QStringLiteral("#808080")});
    QSet<QString> classNames;
    for (const auto& item : classes) {
        if (classNames.contains(item.name))
            return fail(QStringLiteral(
                    "Class names must be unique for dataset export"));
        classNames.insert(item.name);
    }
    if (cancelled(progress, 0, entries.size()))
        return fail(QStringLiteral("Point-cloud export cancelled"));

    QVector<ZipPayload> payloads;
    bool ok = true;
    auto add = [&](const QString& name, const QByteArray& data) {
        if (ok) payloads.append({name.toUtf8(), data});
    };
    if (options.format == QStringLiteral("datumaro")) {
        add(QStringLiteral("annotations/default.json"),
            datumaro(entries, classes));
    } else if (options.format == QStringLiteral("kitti_raw")) {
        add(QStringLiteral("tracklet_labels.xml"), kitti(entries, classes));
        QByteArray frameList;
        for (int i = 0; i < entries.size(); ++i)
            frameList += QByteArray::number(i) + ' ' +
                         entries.at(i).name.toUtf8() + '\n';
        add(QStringLiteral("frame_list.txt"), frameList);
        QJsonArray labels;
        for (const ExportClass& item : classes) labels.append(item.name);
        add(QStringLiteral("dataset_meta.json"),
            QJsonDocument(QJsonObject{{QStringLiteral("labels"), labels}})
                    .toJson(QJsonDocument::Indented));
    } else {
        QJsonArray classArray;
        for (int i = 0; i < classes.size(); ++i)
            classArray.append(QJsonObject{
                    {QStringLiteral("id"), i + 1},
                    {QStringLiteral("title"), classes.at(i).name},
                    {QStringLiteral("color"), classes.at(i).color},
                    {QStringLiteral("shape"), QStringLiteral("cuboid_3d")},
                    {QStringLiteral("geometry_config"), QJsonObject{}}});
        QJsonArray tags;
        for (int i = 0; i < 2; ++i) {
            const QString name = i == 0 ? QStringLiteral("occluded")
                                        : QStringLiteral("locked");
            tags.append(QJsonObject{{QStringLiteral("id"), i + 1},
                                    {QStringLiteral("name"), name},
                                    {QStringLiteral("value_type"),
                                     QStringLiteral("any_string")},
                                    {QStringLiteral("color"), QString()},
                                    {QStringLiteral("hotkey"), QString()},
                                    {QStringLiteral("applicable_type"),
                                     QStringLiteral("objectsOnly")},
                                    {QStringLiteral("classes"), QJsonArray{}}});
        }
        add(QStringLiteral("meta.json"),
            QJsonDocument(QJsonObject{{QStringLiteral("classes"), classArray},
                                      {QStringLiteral("tags"), tags},
                                      {QStringLiteral("projectType"),
                                       QStringLiteral("point_clouds")}})
                    .toJson(QJsonDocument::Indented));
    }
    QMap<int, int> labelIds;
    for (int i = 0; i < classes.size(); ++i) labelIds[classes.at(i).id] = i;
    QJsonObject slyObjectMap;
    QJsonObject slyFigureMap;
    QJsonObject slyVideoMap;
    for (int i = 0; ok && i < entries.size(); ++i) {
        const Entry& entry = entries.at(i);
        if (options.format == QStringLiteral("sly_pointcloud")) {
            const QString videoKey = QString::number(i + 1);
            slyVideoMap.insert(videoKey, i);
            QJsonArray objects;
            QJsonArray figures;
            for (int j = 0; j < entry.cuboids.size(); ++j) {
                const PointCloudCuboid& cuboid = entry.cuboids.at(j);
                const QString objectKey =
                        QStringLiteral("object_%1_%2").arg(i).arg(j);
                const QString figureKey =
                        QStringLiteral("figure_%1_%2").arg(i).arg(j);
                const int objectId = slyObjectMap.size() + 1;
                slyObjectMap.insert(objectKey, objectId);
                slyFigureMap.insert(figureKey, objectId);
                QJsonArray objectTags;
                for (const QString& tag :
                     {QStringLiteral("occluded"), QStringLiteral("locked")}) {
                    const bool value = tag == QStringLiteral("occluded")
                                               ? cuboid.occluded
                                               : cuboid.locked;
                    objectTags.append(QJsonObject{
                            {QStringLiteral("key"),
                             QStringLiteral("%1_%2").arg(tag, objectKey)},
                            {QStringLiteral("name"), tag},
                            {QStringLiteral("value"),
                             value ? QStringLiteral("true")
                                   : QStringLiteral("false")}});
                }
                objects.append(QJsonObject{
                        {QStringLiteral("key"), objectKey},
                        {QStringLiteral("classTitle"),
                         classes.at(labelIds.value(cuboid.classId)).name},
                        {QStringLiteral("tags"), objectTags}});
                figures.append(
                        QJsonObject{{QStringLiteral("key"), figureKey},
                                    {QStringLiteral("objectKey"), objectKey},
                                    {QStringLiteral("geometryType"),
                                     QStringLiteral("cuboid_3d")},
                                    {QStringLiteral("geometry"),
                                     QJsonObject{{QStringLiteral("position"),
                                                  xyzObject(cuboid.center)},
                                                 {QStringLiteral("rotation"),
                                                  xyzObject(cuboid.rotation)},
                                                 {QStringLiteral("dimensions"),
                                                  xyzObject(cuboid.size)}}}});
            }
            add(QStringLiteral("ds0/ann/%1.pcd.json").arg(entry.name),
                QJsonDocument(
                        QJsonObject{
                                {QStringLiteral("description"), QString()},
                                {QStringLiteral("key"), QString::number(i + 1)},
                                {QStringLiteral("tags"), QJsonArray{}},
                                {QStringLiteral("objects"), objects},
                                {QStringLiteral("figures"), figures}})
                        .toJson(QJsonDocument::Indented));
        }
        if (options.includePointCloud) {
            QString localError;
            const QByteArray pcd = pointCloudPcd(entry.frame, &localError);
            if (pcd.isEmpty() && !entry.frame.points.isEmpty()) {
                if (error) *error = localError;
                ok = false;
                break;
            }
            const QString path =
                    options.format == QStringLiteral("datumaro")
                            ? QStringLiteral("point_clouds/default/%1.pcd")
                                      .arg(entry.name)
                    : options.format == QStringLiteral("kitti_raw")
                            ? QStringLiteral("velodyne_points/data/%1.pcd")
                                      .arg(entry.name)
                            : QStringLiteral("ds0/pointcloud/%1.pcd")
                                      .arg(entry.name);
            add(path, pcd);
        }
        if (cancelled(progress, i + 1, entries.size())) {
            if (error) *error = QStringLiteral("Point-cloud export cancelled");
            ok = false;
        }
    }
    if (ok && options.format == QStringLiteral("sly_pointcloud")) {
        add(QStringLiteral("key_id_map.json"),
            QJsonDocument(QJsonObject{{QStringLiteral("tags"), QJsonObject{}},
                                      {QStringLiteral("objects"), slyObjectMap},
                                      {QStringLiteral("figures"), slyFigureMap},
                                      {QStringLiteral("videos"), slyVideoMap}})
                    .toJson(QJsonDocument::Indented));
    }
    if (ok)
        ok = writeZipArchive(
                options.outputPath, payloads,
                [&] {
                    return !cancelled(progress, entries.size(), entries.size());
                },
                error);
    if (!ok) {
        if (error && error->isEmpty())
            *error = QStringLiteral("Could not finish ZIP archive");
        return false;
    }
    return true;
}
