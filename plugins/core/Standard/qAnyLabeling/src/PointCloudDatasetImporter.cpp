// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudDatasetImporter.h"
#include "QtCompat.h"

#include <zlib.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>
#include <QRegularExpression>
#include <QSet>
#include <QTemporaryFile>
#include <QXmlStreamReader>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef Q_OS_WIN
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {
constexpr qint64 kMaxMemberBytes = 128ll * 1024 * 1024;
constexpr qint64 kMaxAnnotationBytes = 512ll * 1024 * 1024;

struct ZipMember {
    QString name;
    quint16 method = 0;
    quint32 compressed = 0;
    quint32 size = 0;
    quint32 offset = 0;
    quint32 crc = 0;
};

bool fail(QString* error, const QString& message) {
    if (error) *error = message;
    return false;
}

quint16 u16(const QByteArray& data, qint64 at, bool* ok) {
    if (at < 0 || at + 2 > data.size()) {
        *ok = false;
        return 0;
    }
    *ok = true;
    return quint16(quint8(data.at(int(at)))) |
           (quint16(quint8(data.at(int(at + 1)))) << 8);
}

quint32 u32(const QByteArray& data, qint64 at, bool* ok) {
    if (at < 0 || at + 4 > data.size()) {
        *ok = false;
        return 0;
    }
    *ok = true;
    return quint32(quint8(data.at(int(at)))) |
           (quint32(quint8(data.at(int(at + 1)))) << 8) |
           (quint32(quint8(data.at(int(at + 2)))) << 16) |
           (quint32(quint8(data.at(int(at + 3)))) << 24);
}

bool safeName(const QString& name) {
    QString normalized = name;
    normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));
    if (normalized.isEmpty() || normalized.startsWith(QLatin1Char('/')) ||
        QRegularExpression(QStringLiteral("^[A-Za-z]:"))
                .match(normalized)
                .hasMatch())
        return false;
    const QStringList parts =
            normalized.split(QLatin1Char('/'), QtCompat::KeepEmptyParts);
    for (int i = 0; i < parts.size(); ++i)
        if (parts.at(i).isEmpty() &&
            (i != parts.size() - 1 || !normalized.endsWith(QLatin1Char('/'))))
            return false;
        else if (parts.at(i) == QStringLiteral("..") ||
                 parts.at(i) == QStringLiteral("."))
            return false;
    return true;
}

bool inflateRaw(const QByteArray& input,
                quint32 expected,
                QByteArray* output,
                QString* error) {
    if (expected > kMaxMemberBytes || input.size() > kMaxMemberBytes)
        return fail(
                error,
                QStringLiteral(
                        "Annotation member exceeds the import size limit."));
    z_stream stream{};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        return fail(error,
                    QStringLiteral("Could not initialize ZIP decompressor."));
    QByteArray result;
    result.resize(std::max(1, int(expected)));
    stream.next_in =
            reinterpret_cast<Bytef*>(const_cast<char*>(input.constData()));
    stream.avail_in = static_cast<uInt>(input.size());
    stream.next_out = reinterpret_cast<Bytef*>(result.data());
    stream.avail_out = static_cast<uInt>(result.size());
    const int status = inflate(&stream, Z_FINISH);
    const bool complete = status == Z_STREAM_END &&
                          stream.total_out == expected &&
                          stream.total_in == static_cast<uLong>(input.size());
    inflateEnd(&stream);
    if (!complete)
        return fail(error,
                    QStringLiteral("Invalid deflated ZIP annotation member."));
    result.resize(int(expected));
    *output = result;
    return true;
}

class Archive {
public:
    bool open(const QString& path, QString* error) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly))
            return fail(error, file.errorString());
        if (file.size() > std::numeric_limits<int>::max())
            return fail(error,
                        QStringLiteral("Archives larger than 2 GiB require the "
                                       "streaming ZIP importer."));
        m_data = file.readAll();
        if (file.error() != QFileDevice::NoError)
            return fail(error, file.errorString());
        if (m_data.size() < 22)
            return fail(error, QStringLiteral("Invalid ZIP archive."));
        const qint64 begin = std::max<qint64>(0, m_data.size() - 65557);
        qint64 eocd = -1;
        for (qint64 at = m_data.size() - 22; at >= begin; --at) {
            bool ok = false;
            if (u32(m_data, at, &ok) == 0x06054b50u &&
                at + 22 + u16(m_data, at + 20, &ok) == m_data.size()) {
                eocd = at;
                break;
            }
        }
        if (eocd < 0)
            return fail(error, QStringLiteral("ZIP end record is missing."));
        bool ok = false;
        const quint16 disk = u16(m_data, eocd + 4, &ok);
        const quint16 centralDisk = u16(m_data, eocd + 6, &ok);
        const quint16 count = u16(m_data, eocd + 10, &ok);
        const quint16 total = u16(m_data, eocd + 8, &ok);
        const quint32 centralSize = u32(m_data, eocd + 12, &ok);
        const quint32 centralOffset = u32(m_data, eocd + 16, &ok);
        if (!ok || disk != 0 || centralDisk != 0 || total != count ||
            qint64(centralOffset) + centralSize != eocd)
            return fail(
                    error,
                    QStringLiteral(
                            "Unsupported or malformed ZIP central directory."));
        qint64 at = centralOffset;
        for (quint16 i = 0; i < count; ++i) {
            if (at + 46 > m_data.size() || u32(m_data, at, &ok) != 0x02014b50u)
                return fail(error,
                            QStringLiteral(
                                    "Malformed ZIP central directory entry."));
            const quint16 flags = u16(m_data, at + 8, &ok);
            const quint16 method = u16(m_data, at + 10, &ok);
            const quint32 crc = u32(m_data, at + 16, &ok);
            const quint32 compressed = u32(m_data, at + 20, &ok);
            const quint32 size = u32(m_data, at + 24, &ok);
            const quint16 nameLength = u16(m_data, at + 28, &ok);
            const quint16 extraLength = u16(m_data, at + 30, &ok);
            const quint16 commentLength = u16(m_data, at + 32, &ok);
            const quint32 offset = u32(m_data, at + 42, &ok);
            const qint64 recordSize =
                    46ll + nameLength + extraLength + commentLength;
            QString name =
                    QString::fromUtf8(m_data.mid(int(at + 46), nameLength));
            name.replace(QLatin1Char('\\'), QLatin1Char('/'));
            if (!ok || at + recordSize > m_data.size() || (flags & 1) ||
                (method != 0 && method != 8) || !safeName(name))
                return fail(
                        error,
                        QStringLiteral("Unsafe or unsupported ZIP member."));
            if (name.endsWith(QLatin1Char('/'))) {
                at += recordSize;
                continue;
            }
            if (m_members.contains(name))
                return fail(
                        error,
                        QStringLiteral("Duplicate ZIP member: %1").arg(name));
            m_members.insert(name,
                             {name, method, compressed, size, offset, crc});
            at += recordSize;
        }
        return at == qint64(centralOffset) + centralSize ||
               fail(error, QStringLiteral("ZIP central directory size does not "
                                          "match entries."));
    }
    QStringList names() const { return m_members.keys(); }
    bool contains(const QString& name) const {
        return m_members.contains(name);
    }
    bool read(const QString& name, QByteArray* output, QString* error) const {
        if (!m_members.contains(name))
            return fail(error,
                        QStringLiteral("Missing ZIP member: %1").arg(name));
        const ZipMember member = m_members.value(name);
        if (member.size > kMaxMemberBytes ||
            member.compressed > kMaxMemberBytes)
            return fail(error, QStringLiteral("Annotation member exceeds the "
                                              "import size limit."));
        bool ok = false;
        if (qint64(member.offset) + 30 > m_data.size() ||
            u32(m_data, member.offset, &ok) != 0x04034b50u)
            return fail(error, QStringLiteral("Invalid ZIP local header."));
        const quint16 nameLength = u16(m_data, member.offset + 26, &ok);
        const quint16 extraLength = u16(m_data, member.offset + 28, &ok);
        const quint16 flags = u16(m_data, member.offset + 6, &ok);
        const quint16 method = u16(m_data, member.offset + 8, &ok);
        const qint64 dataOffset =
                qint64(member.offset) + 30 + nameLength + extraLength;
        if (!ok || dataOffset < 0 ||
            dataOffset + member.compressed > m_data.size())
            return fail(error,
                        QStringLiteral("Invalid ZIP member data bounds."));
        QString localName = QString::fromUtf8(
                m_data.mid(int(member.offset + 30), nameLength));
        localName.replace(QLatin1Char('\\'), QLatin1Char('/'));
        if ((flags & 1) || method != member.method || localName != member.name)
            return fail(error, QStringLiteral("ZIP local header does not match "
                                              "its directory entry."));
        const QByteArray compressed =
                m_data.mid(int(dataOffset), int(member.compressed));
        if (member.method == 0) {
            if (member.size > kMaxMemberBytes ||
                compressed.size() != int(member.size))
                return fail(error,
                            QStringLiteral("Invalid stored ZIP member."));
            *output = compressed;
        } else if (!inflateRaw(compressed, member.size, output, error)) {
            return false;
        }
        return crc32(0L, reinterpret_cast<const Bytef*>(output->constData()),
                     static_cast<uInt>(output->size())) == member.crc ||
               fail(error, QStringLiteral("ZIP annotation checksum mismatch."));
    }

private:
    QByteArray m_data;
    QMap<QString, ZipMember> m_members;
};

bool progress(const std::function<bool(int, int)>& callback,
              int current,
              int total,
              QString* error) {
    return !callback || callback(current, total) ||
           fail(error, QStringLiteral("Point-cloud dataset import cancelled."));
}

bool readJson(const Archive& archive,
              const QString& name,
              QJsonDocument* doc,
              qint64* total,
              QString* error) {
    QByteArray bytes;
    if (!archive.read(name, &bytes, error)) return false;
    *total += bytes.size();
    if (*total > kMaxAnnotationBytes)
        return fail(error,
                    QStringLiteral(
                            "Annotation data exceeds the import size limit."));
    QJsonParseError parse;
    *doc = QJsonDocument::fromJson(bytes, &parse);
    if (parse.error != QJsonParseError::NoError || doc->isNull())
        return fail(error, QStringLiteral("Invalid JSON annotation: %1")
                                   .arg(parse.errorString()));
    return true;
}

bool classesFrom(const QJsonArray& values,
                 QVector<PointCloudDatasetImportClass>* classes,
                 QString* error) {
    if (values.size() > 65535)
        return fail(error, QStringLiteral("Too many imported classes."));
    QSet<QString> names;
    QVector<PointCloudDatasetImportClass> parsed;
    for (int i = 0; i < values.size(); ++i) {
        const QJsonObject object = values.at(i).toObject();
        const QString name = object.value(QStringLiteral("name")).toString();
        if (name.trimmed().isEmpty() || names.contains(name))
            return fail(error, QStringLiteral("Imported class names must be "
                                              "nonempty and unique."));
        QColor color(QStringLiteral("#AAAAFF"));
        const QString text = object.value(QStringLiteral("color")).toString();
        if (QRegularExpression(QStringLiteral("^#[0-9a-fA-F]{6}$"))
                    .match(text)
                    .hasMatch())
            color = QColor(text);
        parsed.append({i + 1, name, color});
        names.insert(name);
    }
    *classes = parsed;
    return true;
}

bool vector(const QJsonValue& value, CuboidVector* output, QString* error) {
    const QJsonArray values = value.toArray();
    if (!value.isArray() || values.size() != 3)
        return fail(
                error,
                QStringLiteral("Cuboid vectors must contain three values."));
    for (int i = 0; i < 3; ++i) {
        if (!values.at(i).isDouble() || !std::isfinite(values.at(i).toDouble()))
            return fail(
                    error,
                    QStringLiteral("Cuboid vectors must be finite numbers."));
        (*output)[i] = values.at(i).toDouble();
    }
    return true;
}

bool boolean(const QJsonValue& value, bool* output, QString* error) {
    if (value.isUndefined() || value.isNull()) {
        *output = false;
        return true;
    }
    if (value.isBool()) {
        *output = value.toBool();
        return true;
    }
    if (value.isDouble() && (value.toDouble() == 0 || value.toDouble() == 1)) {
        *output = value.toInt() != 0;
        return true;
    }
    if (value.isString()) {
        const QString text = value.toString();
        if (text == QStringLiteral("true") || text == QStringLiteral("True") ||
            text == QStringLiteral("1")) {
            *output = true;
            return true;
        }
        if (text.isEmpty() || text == QStringLiteral("false") ||
            text == QStringLiteral("False") || text == QStringLiteral("0")) {
            *output = false;
            return true;
        }
    }
    return fail(error, QStringLiteral("Invalid cuboid boolean attribute."));
}

bool box(int id,
         int classId,
         const QJsonValue& position,
         const QJsonValue& size,
         const QJsonValue& rotation,
         const QJsonObject& attributes,
         PointCloudCuboid* output,
         QString* error) {
    PointCloudCuboid value;
    value.id = id;
    value.classId = classId;
    if (!vector(position, &value.center, error) ||
        !vector(size, &value.size, error) ||
        !vector(rotation, &value.rotation, error) ||
        !boolean(attributes.value(QStringLiteral("occluded")), &value.occluded,
                 error) ||
        !boolean(attributes.value(QStringLiteral("locked")), &value.locked,
                 error) ||
        !value.valid(error))
        return false;
    *output = value;
    return true;
}

QStringList annotationFiles(const QStringList& names, const QString& suffix) {
    QStringList result;
    for (const QString& name : names)
        if (name.endsWith(suffix, Qt::CaseInsensitive)) result.append(name);
    std::sort(result.begin(), result.end());
    return result;
}

QString memberFileName(const QString& name) {
    return QFileInfo(name).fileName();
}

bool readText(const Archive& archive,
              const QString& name,
              QByteArray* bytes,
              qint64* total,
              QString* error) {
    if (!archive.read(name, bytes, error)) return false;
    *total += bytes->size();
    return *total <= kMaxAnnotationBytes ||
           fail(error,
                QStringLiteral(
                        "Annotation data exceeds the import size limit."));
}

bool parseDatumaro(const Archive& archive,
                   const QStringList& names,
                   PointCloudDatasetImportResult* result,
                   const std::function<bool(int, int)>& callback,
                   QString* error) {
    QStringList files;
    for (const QString& name : names) {
        if ((name.startsWith(QStringLiteral("annotations/")) ||
             name.contains(QStringLiteral("/annotations/"))) &&
            name.endsWith(QStringLiteral(".json"), Qt::CaseInsensitive))
            files.append(name);
    }
    if (files.isEmpty())
        return fail(error,
                    QStringLiteral(
                            "Datumaro annotations/default.json is missing."));
    std::sort(files.begin(), files.end());
    QVector<PointCloudDatasetImportClass> classes;
    QSet<QString> frameIds;
    qint64 totalBytes = 0;
    for (int fileIndex = 0; fileIndex < files.size(); ++fileIndex) {
        if (!progress(callback, fileIndex, files.size(), error)) return false;
        QJsonDocument document;
        if (!readJson(archive, files.at(fileIndex), &document, &totalBytes,
                      error))
            return false;
        const QJsonObject root = document.object();
        if (root.value(QStringLiteral("dm_format_version"))
                            .toString(QStringLiteral("1.0")) !=
                    QStringLiteral("1.0") ||
            root.value(QStringLiteral("media_type")).toInt(6) != 6)
            return fail(error,
                        QStringLiteral("Unsupported Datumaro point-cloud "
                                       "annotation metadata."));
        const QJsonObject category = root.value(QStringLiteral("categories"))
                                             .toObject()
                                             .value(QStringLiteral("label"))
                                             .toObject();
        QVector<PointCloudDatasetImportClass> parsedClasses;
        if (!classesFrom(category.value(QStringLiteral("labels")).toArray(),
                         &parsedClasses, error))
            return false;
        if (classes.isEmpty())
            classes = parsedClasses;
        else if (classes != parsedClasses)
            return fail(error, QStringLiteral("Datumaro subsets have different "
                                              "class definitions."));
        const QJsonArray items = root.value(QStringLiteral("items")).toArray();
        for (int itemIndex = 0; itemIndex < items.size(); ++itemIndex) {
            const QJsonObject item = items.at(itemIndex).toObject();
            const QJsonValue idValue = item.value(QStringLiteral("id"));
            const QString id =
                    idValue.isDouble()
                            ? QString::number(idValue.toDouble(), 'g', 17)
                            : idValue.toString();
            if (id.isEmpty() || frameIds.contains(id))
                return fail(error, QStringLiteral("Datumaro frame ids must be "
                                                  "unique and nonempty."));
            if (item.value(QStringLiteral("point_cloud"))
                        .toObject()
                        .value(QStringLiteral("path"))
                        .toString()
                        .isEmpty())
                return fail(error,
                            QStringLiteral(
                                    "Datumaro item has no point-cloud path."));
            PointCloudDatasetImportedFrame frame;
            frame.name = id;
            frame.index = item.value(QStringLiteral("attr"))
                                  .toObject()
                                  .value(QStringLiteral("frame"))
                                  .toInt(itemIndex);
            if (frame.index < 0)
                return fail(error,
                            QStringLiteral("Datumaro frame index is invalid."));
            const QJsonArray annotations =
                    item.value(QStringLiteral("annotations")).toArray();
            for (int annotationIndex = 0; annotationIndex < annotations.size();
                 ++annotationIndex) {
                const QJsonObject annotation =
                        annotations.at(annotationIndex).toObject();
                if (annotation.value(QStringLiteral("type")).toString() !=
                    QStringLiteral("cuboid_3d"))
                    return fail(error,
                                QStringLiteral("Only cuboid_3d annotations can "
                                               "be imported."));
                const QJsonValue labelValue =
                        annotation.value(QStringLiteral("label_id"));
                const double labelNumber = labelValue.toDouble(-1);
                if (!labelValue.isDouble() || labelNumber < 0 ||
                    labelNumber >= classes.size() ||
                    std::floor(labelNumber) != labelNumber)
                    return fail(error, QStringLiteral("Datumaro cuboid label "
                                                      "id is out of range."));
                const int label = int(labelNumber);
                QJsonValue rotation =
                        annotation.value(QStringLiteral("rotation"));
                if (rotation.isUndefined())
                    rotation = QJsonArray{0.0, 0.0, 0.0};
                PointCloudCuboid cuboid;
                if (!box(frame.cuboids.size() + 1, classes.at(label).id,
                         annotation.value(QStringLiteral("position")),
                         annotation.value(QStringLiteral("scale")), rotation,
                         annotation.value(QStringLiteral("attributes"))
                                 .toObject(),
                         &cuboid, error))
                    return false;
                frame.cuboids.append(cuboid);
            }
            result->frames.append(frame);
            frameIds.insert(id);
        }
    }
    result->classes = classes;
    return progress(callback, files.size(), files.size(), error);
}

struct XmlNode {
    QString name;
    QString text;
    QVector<XmlNode> children;

    const XmlNode* first(const QString& childName) const {
        for (const XmlNode& child : children)
            if (child.name == childName) return &child;
        return nullptr;
    }
    QVector<const XmlNode*> all(const QString& childName) const {
        QVector<const XmlNode*> values;
        for (const XmlNode& child : children)
            if (child.name == childName) values.append(&child);
        return values;
    }
};

bool parseXmlNode(QXmlStreamReader* reader, XmlNode* node, QString* error) {
    if (!reader->isStartElement())
        return fail(error, QStringLiteral("Invalid KITTI XML node."));
    node->name = reader->name().toString();
    while (!reader->atEnd()) {
        const QXmlStreamReader::TokenType token = reader->readNext();
        if (token == QXmlStreamReader::StartElement) {
            XmlNode child;
            if (!parseXmlNode(reader, &child, error)) return false;
            node->children.append(child);
        } else if (token == QXmlStreamReader::Characters ||
                   token == QXmlStreamReader::EntityReference) {
            node->text += reader->text().toString();
        } else if (token == QXmlStreamReader::EndElement) {
            return true;
        }
    }
    return fail(error, QStringLiteral("Unterminated KITTI XML annotation."));
}

QString xmlText(const XmlNode* node,
                const QString& child,
                const QString& fallback = {}) {
    const XmlNode* value = node ? node->first(child) : nullptr;
    return value ? value->text.trimmed() : fallback;
}

bool xmlDouble(const XmlNode* node,
               const QString& child,
               double* value,
               QString* error) {
    bool ok = false;
    const QString text = xmlText(node, child);
    *value = text.toDouble(&ok);
    if (!ok || !std::isfinite(*value))
        return fail(error, QStringLiteral("Invalid KITTI numeric value for %1.")
                                   .arg(child));
    return true;
}

bool xmlInt(const XmlNode* node,
            const QString& child,
            int* value,
            QString* error) {
    bool ok = false;
    const QString text = xmlText(node, child);
    *value = text.toInt(&ok);
    return ok ||
           fail(error, QStringLiteral("Invalid KITTI integer value for %1.")
                               .arg(child));
}

bool parseKitti(const Archive& archive,
                const QStringList& names,
                PointCloudDatasetImportResult* result,
                const std::function<bool(int, int)>& callback,
                QString* error) {
    QString xmlName;
    for (const QString& name : names)
        if (memberFileName(name) == QStringLiteral("tracklet_labels.xml")) {
            xmlName = name;
            break;
        }
    if (xmlName.isEmpty())
        return fail(error,
                    QStringLiteral("KITTI tracklet_labels.xml is missing."));
    qint64 totalBytes = 0;
    QByteArray xmlBytes;
    if (!readText(archive, xmlName, &xmlBytes, &totalBytes, error))
        return false;
    if (xmlBytes.toUpper().contains("<!ENTITY"))
        return fail(error,
                    QStringLiteral("KITTI XML entities are not allowed."));
    QXmlStreamReader reader(xmlBytes);
    XmlNode root;
    while (!reader.atEnd() && !reader.isStartElement()) reader.readNext();
    if (reader.atEnd() || !parseXmlNode(&reader, &root, error) ||
        reader.hasError())
        return fail(error, QStringLiteral("Invalid KITTI tracklet XML: %1")
                                   .arg(reader.errorString()));
    const XmlNode* tracklets =
            root.name == QStringLiteral("boost_serialization")
                    ? root.first(QStringLiteral("tracklets"))
                    : nullptr;
    if (!tracklets)
        return fail(error, QStringLiteral("KITTI tracklets node is missing."));

    QVector<PointCloudDatasetImportClass> classes;
    for (const QString& name : names) {
        if (memberFileName(name) != QStringLiteral("dataset_meta.json"))
            continue;
        QJsonDocument metadata;
        if (!readJson(archive, name, &metadata, &totalBytes, error))
            return false;
        QJsonArray labels;
        for (const QJsonValue& value :
             metadata.object().value(QStringLiteral("labels")).toArray())
            labels.append(
                    QJsonObject{{QStringLiteral("name"), value.toString()}});
        if (!labels.isEmpty() && !classesFrom(labels, &classes, error))
            return false;
        break;
    }
    QMap<QString, int> classIds;
    for (const auto& item : classes) classIds.insert(item.name, item.id);
    QMap<int, QVector<PointCloudCuboid>> byFrame;
    const QVector<const XmlNode*> tracks =
            tracklets->all(QStringLiteral("item"));
    for (const XmlNode* track : tracks) {
        const QString objectType = xmlText(track, QStringLiteral("objectType"));
        if (objectType.isEmpty())
            return fail(error,
                        QStringLiteral("KITTI track has no object type."));
        if (!classIds.contains(objectType)) {
            if (classes.size() >= 65535)
                return fail(error,
                            QStringLiteral("Too many imported classes."));
            const int id = classes.size() + 1;
            classes.append({id, objectType, QColor(QStringLiteral("#AAAAFF"))});
            classIds.insert(objectType, id);
        }
        CuboidVector size{{0, 0, 0}};
        const QStringList sizeKeys{QStringLiteral("h"), QStringLiteral("w"),
                                   QStringLiteral("l")};
        for (int i = 0; i < 3; ++i)
            if (!xmlDouble(track, sizeKeys.at(i), &size[i], error) ||
                size[i] <= 0)
                return fail(
                        error,
                        QStringLiteral(
                                "KITTI cuboid dimensions must be positive."));
        int firstFrame = -1;
        if (!xmlInt(track, QStringLiteral("first_frame"), &firstFrame, error) ||
            firstFrame < 0)
            return fail(error, QStringLiteral("KITTI first_frame is invalid."));
        const XmlNode* poses = track->first(QStringLiteral("poses"));
        if (!poses)
            return fail(error, QStringLiteral("KITTI track has no poses."));
        const QVector<const XmlNode*> poseItems =
                poses->all(QStringLiteral("item"));
        int poseCount = 0;
        if (!xmlInt(poses, QStringLiteral("count"), &poseCount, error) ||
            poseCount != poseItems.size())
            return fail(error,
                        QStringLiteral(
                                "KITTI pose count does not match XML items."));
        if (firstFrame > std::numeric_limits<int>::max() - poseItems.size())
            return fail(
                    error,
                    QStringLiteral(
                            "KITTI frame indices exceed the supported range."));
        const QStringList centerKeys{QStringLiteral("tx"), QStringLiteral("ty"),
                                     QStringLiteral("tz")};
        const QStringList rotationKeys{QStringLiteral("rx"),
                                       QStringLiteral("ry"),
                                       QStringLiteral("rz")};
        bool previousOccluded = false;
        for (int poseIndex = 0; poseIndex < poseItems.size(); ++poseIndex) {
            const XmlNode* pose = poseItems.at(poseIndex);
            const int frameIndex = firstFrame + poseIndex;
            byFrame[frameIndex];
            int occlusion = -1;
            if ((pose->first(QStringLiteral("occlusion")) &&
                 !xmlInt(pose, QStringLiteral("occlusion"), &occlusion,
                         error)) ||
                occlusion < -1 || occlusion > 2)
                return fail(
                        error,
                        QStringLiteral("KITTI occlusion state is invalid."));
            const bool occluded =
                    occlusion == -1 ? previousOccluded : occlusion > 0;
            if (xmlText(pose, QStringLiteral("occlusion_kf"),
                        QStringLiteral("0")) == QStringLiteral("1"))
                previousOccluded = occluded;
            int truncation = -1;
            if (pose->first(QStringLiteral("truncation")) &&
                !xmlInt(pose, QStringLiteral("truncation"), &truncation, error))
                return false;
            if (truncation < -1 || (truncation > 2 && truncation != 99))
                return fail(
                        error,
                        QStringLiteral("KITTI truncation state is invalid."));
            if (truncation == 2 || truncation == 99) continue;
            PointCloudCuboid cuboid;
            cuboid.id = byFrame[frameIndex].size() + 1;
            cuboid.classId = classIds.value(objectType);
            cuboid.size = size;
            for (int i = 0; i < 3; ++i) {
                if (!xmlDouble(pose, centerKeys.at(i), &cuboid.center[i],
                               error) ||
                    (pose->first(rotationKeys.at(i)) &&
                     !xmlDouble(pose, rotationKeys.at(i), &cuboid.rotation[i],
                                error)))
                    return false;
            }
            cuboid.occluded = occluded;
            if (!cuboid.valid(error)) return false;
            byFrame[frameIndex].append(cuboid);
        }
    }
    const int expectedCount =
            xmlText(tracklets, QStringLiteral("count")).toInt();
    if (expectedCount != tracks.size())
        return fail(error, QStringLiteral("KITTI track count does not match "
                                          "parsed annotations."));

    QMap<int, QString> frameNames;
    for (const QString& name : names) {
        if (memberFileName(name) != QStringLiteral("frame_list.txt")) continue;
        QByteArray bytes;
        if (!readText(archive, name, &bytes, &totalBytes, error)) return false;
        for (const QByteArray& lineBytes : bytes.split('\n')) {
            const QString line = QString::fromUtf8(lineBytes).trimmed();
            if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
            const QRegularExpressionMatch match =
                    QRegularExpression(QStringLiteral("^(\\d+)\\s+(.+)$"))
                            .match(line);
            if (!match.hasMatch())
                return fail(
                        error,
                        QStringLiteral("Invalid KITTI frame_list.txt row."));
            const int index = match.captured(1).toInt();
            const QString frameName = match.captured(2).trimmed();
            if (frameNames.contains(index) ||
                frameNames.values().contains(frameName))
                return fail(
                        error,
                        QStringLiteral(
                                "KITTI frame_list entries must be unique."));
            frameNames.insert(index, frameName);
        }
        break;
    }
    for (auto it = byFrame.cbegin(); it != byFrame.cend(); ++it)
        if (!frameNames.isEmpty() && !frameNames.contains(it.key()))
            return fail(error, QStringLiteral("KITTI annotation frame is "
                                              "missing from frame_list.txt."));
    if (frameNames.isEmpty()) {
        for (auto it = byFrame.cbegin(); it != byFrame.cend(); ++it)
            result->frames.append({QString(), it.key(), it.value()});
    } else {
        for (auto it = frameNames.cbegin(); it != frameNames.cend(); ++it)
            result->frames.append(
                    {it.value(), it.key(), byFrame.value(it.key())});
    }
    result->classes = classes;
    return progress(callback, 1, 1, error);
}

bool jsonPoint(const QJsonValue& value, CuboidVector* output, QString* error) {
    if (!value.isObject())
        return fail(error,
                    QStringLiteral("Cuboid geometry must be an object."));
    const QJsonObject object = value.toObject();
    const QStringList keys{QStringLiteral("x"), QStringLiteral("y"),
                           QStringLiteral("z")};
    for (int i = 0; i < 3; ++i) {
        const QJsonValue component = object.value(keys.at(i));
        if (!component.isDouble() || !std::isfinite(component.toDouble()))
            return fail(
                    error,
                    QStringLiteral(
                            "Cuboid geometry must contain finite numbers."));
        (*output)[i] = component.toDouble();
    }
    return true;
}

bool parseSly(const Archive& archive,
              const QStringList& names,
              PointCloudDatasetImportResult* result,
              const std::function<bool(int, int)>& callback,
              QString* error) {
    QString metaName;
    for (const QString& name : names)
        if (memberFileName(name) == QStringLiteral("meta.json")) {
            metaName = name;
            break;
        }
    if (metaName.isEmpty())
        return fail(error, QStringLiteral("Supervisely meta.json is missing."));
    qint64 totalBytes = 0;
    QJsonDocument meta;
    if (!readJson(archive, metaName, &meta, &totalBytes, error)) return false;
    const QJsonObject root = meta.object();
    if (root.value(QStringLiteral("projectType")).toString() !=
        QStringLiteral("point_clouds"))
        return fail(
                error,
                QStringLiteral(
                        "Supervisely project is not a point-cloud project."));
    QJsonArray labels;
    for (const QJsonValue& value :
         root.value(QStringLiteral("classes")).toArray()) {
        const QJsonObject object = value.toObject();
        labels.append(QJsonObject{
                {QStringLiteral("name"),
                 object.value(QStringLiteral("title")).toString()},
                {QStringLiteral("color"),
                 object.value(QStringLiteral("color")).toString()}});
    }
    if (!classesFrom(labels, &result->classes, error)) return false;
    QMap<QString, int> classIds;
    for (const auto& item : result->classes)
        classIds.insert(item.name, item.id);
    QStringList files;
    for (const QString& name : names)
        if (name.contains(QStringLiteral("/ann/")) &&
            name.endsWith(QStringLiteral(".pcd.json"), Qt::CaseInsensitive))
            files.append(name);
    std::sort(files.begin(), files.end());
    if (files.isEmpty())
        return fail(
                error,
                QStringLiteral(
                        "Supervisely point-cloud annotations are missing."));
    QSet<QString> frameNames;
    QSet<QString> figureKeys;
    for (int index = 0; index < files.size(); ++index) {
        if (!progress(callback, index, files.size(), error)) return false;
        QJsonDocument document;
        if (!readJson(archive, files.at(index), &document, &totalBytes, error))
            return false;
        const QJsonObject annotation = document.object();
        QMap<QString, QPair<int, QPair<bool, bool>>> objectMap;
        for (const QJsonValue& value :
             annotation.value(QStringLiteral("objects")).toArray()) {
            const QJsonObject object = value.toObject();
            const QString key = object.value(QStringLiteral("key")).toString();
            const QString className =
                    object.value(QStringLiteral("classTitle")).toString();
            if (key.isEmpty() || objectMap.contains(key) ||
                !classIds.contains(className))
                return fail(error,
                            QStringLiteral(
                                    "Invalid Supervisely object annotation."));
            bool occluded = false;
            bool locked = false;
            for (const QJsonValue& tagValue :
                 object.value(QStringLiteral("tags")).toArray()) {
                const QJsonObject tag = tagValue.toObject();
                bool valueBool = false;
                if (!boolean(tag.value(QStringLiteral("value")), &valueBool,
                             error))
                    return false;
                if (tag.value(QStringLiteral("name")).toString() ==
                    QStringLiteral("occluded"))
                    occluded = valueBool;
                else if (tag.value(QStringLiteral("name")).toString() ==
                         QStringLiteral("locked"))
                    locked = valueBool;
            }
            objectMap.insert(key, qMakePair(classIds.value(className),
                                            qMakePair(occluded, locked)));
        }
        QString frameName = memberFileName(files.at(index));
        frameName.chop(QStringLiteral(".pcd.json").size());
        if (frameName.isEmpty() || frameNames.contains(frameName))
            return fail(
                    error,
                    QStringLiteral("Supervisely frame names must be unique."));
        PointCloudDatasetImportedFrame frame{frameName, index, {}};
        for (const QJsonValue& value :
             annotation.value(QStringLiteral("figures")).toArray()) {
            const QJsonObject figure = value.toObject();
            const QString figureKey =
                    figure.value(QStringLiteral("key")).toString();
            const QString objectKey =
                    figure.value(QStringLiteral("objectKey")).toString();
            if (figureKey.isEmpty() || figureKeys.contains(figureKey) ||
                !objectMap.contains(objectKey) ||
                figure.value(QStringLiteral("geometryType")).toString() !=
                        QStringLiteral("cuboid_3d"))
                return fail(
                        error,
                        QStringLiteral("Invalid Supervisely cuboid figure."));
            const QJsonObject geometry =
                    figure.value(QStringLiteral("geometry")).toObject();
            PointCloudCuboid cuboid;
            cuboid.id = frame.cuboids.size() + 1;
            cuboid.classId = objectMap.value(objectKey).first;
            if (!jsonPoint(geometry.value(QStringLiteral("position")),
                           &cuboid.center, error) ||
                !jsonPoint(geometry.value(QStringLiteral("dimensions")),
                           &cuboid.size, error))
                return false;
            if (geometry.value(QStringLiteral("rotation")).isUndefined())
                cuboid.rotation = {{0, 0, 0}};
            else if (!jsonPoint(geometry.value(QStringLiteral("rotation")),
                                &cuboid.rotation, error))
                return false;
            cuboid.occluded = objectMap.value(objectKey).second.first;
            cuboid.locked = objectMap.value(objectKey).second.second;
            if (!cuboid.valid(error)) return false;
            frame.cuboids.append(cuboid);
            figureKeys.insert(figureKey);
        }
        result->frames.append(frame);
        frameNames.insert(frameName);
    }
    return progress(callback, files.size(), files.size(), error);
}

}  // namespace

bool importPointCloudDataset(const QString& archivePath,
                             const QString& format,
                             PointCloudDatasetImportResult* result,
                             const std::function<bool(int, int)>& callback,
                             QString* error) {
    if (error) error->clear();
    if (!result) return fail(error, QStringLiteral("Import result is null."));
    if (archivePath.isEmpty())
        return fail(error,
                    QStringLiteral("Point-cloud import requires a ZIP path."));
    if (format != QStringLiteral("datumaro") &&
        format != QStringLiteral("kitti_raw") &&
        format != QStringLiteral("sly_pointcloud"))
        return fail(error,
                    QStringLiteral("Unknown point-cloud import format."));
    PointCloudDatasetImportResult parsed;
    Archive archive;
    if (!archive.open(archivePath, error)) return false;
    const QStringList names = archive.names();
    bool ok = false;
    if (format == QStringLiteral("datumaro"))
        ok = parseDatumaro(archive, names, &parsed, callback, error);
    else if (format == QStringLiteral("kitti_raw"))
        ok = parseKitti(archive, names, &parsed, callback, error);
    else
        ok = parseSly(archive, names, &parsed, callback, error);
    if (!ok) return false;
    if (parsed.classes.isEmpty() || parsed.frames.isEmpty())
        return fail(
                error,
                QStringLiteral("Point-cloud dataset contains no annotations."));
    *result = parsed;
    return true;
}

namespace {
QString resolvedPath(const QString& path) {
    const QFileInfo info(path);
    const QString canonical = info.canonicalFilePath();
    if (!canonical.isEmpty()) return canonical;
    const QString parent = info.absoluteDir().canonicalPath();
    return parent.isEmpty() ? info.absoluteFilePath()
                            : QDir(parent).filePath(info.fileName());
}

bool readExisting(const QString& path,
                  QByteArray* bytes,
                  bool* exists,
                  QString* error) {
    const QFileInfo info(path);
    if (info.isSymLink() || (info.exists() && !info.isFile()))
        return fail(error,
                    QStringLiteral("Choose a regular import destination: %1")
                            .arg(path));
    *exists = info.exists();
    bytes->clear();
    if (!*exists) return true;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return fail(error, file.errorString());
    *bytes = file.readAll();
    return file.error() == QFileDevice::NoError ||
           fail(error, file.errorString());
}

QString frameKey(QString name) {
    name.replace(QLatin1Char('\\'), QLatin1Char('/'));
    name = name.section(QLatin1Char('/'), -1);
    const QString extension = QFileInfo(name).suffix().toLower();
    if (extension == QStringLiteral("bin") ||
        extension == QStringLiteral("ply") ||
        extension == QStringLiteral("pcd"))
        name = QFileInfo(name).completeBaseName();
    return name;
}

QStringList exportedFrameNames(
        const QVector<PointCloudDatasetImportTarget>& targets) {
    QStringList names;
    QSet<QString> used;
    for (int index = 0; index < targets.size(); ++index) {
        QString name = QFileInfo(targets.at(index).path).completeBaseName();
        name.replace(QLatin1Char('\\'), QLatin1Char('_'));
        name.replace(QLatin1Char('\n'), QLatin1Char('_'));
        name.replace(QLatin1Char('\r'), QLatin1Char('_'));
        name = name.trimmed();
        if (name.isEmpty() || name == QStringLiteral(".") ||
            name == QStringLiteral(".."))
            name = QString::number(index);
        if (used.contains(name))
            name = QString::number(index) + QLatin1Char('_') + name;
        while (used.contains(name)) name.prepend(QLatin1Char('_'));
        names.append(name);
        used.insert(name);
    }
    return names;
}

bool targetCuboids(const PointCloudDatasetImportTarget& target,
                   QVector<PointCloudCuboid>* boxes,
                   QString* error) {
    if (target.cuboidsProvided) {
        *boxes = target.cuboids;
        return true;
    }
    // The active output folder may differ from the source's sidecar folder.
    QStringList candidates{
            target.cuboidPath, pointCloudCuboidPath(target.labelPath),
            QFileInfo(target.path)
                    .dir()
                    .filePath(QFileInfo(target.path).completeBaseName() +
                              QStringLiteral(".cuboids.json"))};
    for (const QString& candidate : candidates) {
        if (candidate.isEmpty()) continue;
        const QFileInfo info(candidate);
        if (info.exists() || info.isSymLink())
            return loadPointCloudCuboids(candidate, target.path, boxes, error);
    }
    boxes->clear();
    return true;
}

bool validateClasses(const QVector<PointCloudDatasetImportClass>& classes,
                     bool segmentation,
                     QString* error) {
    QSet<int> ids;
    QSet<QString> names;
    for (const auto& item : classes) {
        if (item.id < (segmentation ? 0 : 1) || item.id > 65535 ||
            ids.contains(item.id) || item.name.trimmed().isEmpty() ||
            !item.color.isValid() ||
            (!segmentation && names.contains(item.name)))
            return fail(error,
                        QStringLiteral("Class IDs, names, and colors must be "
                                       "valid and unique before importing."));
        ids.insert(item.id);
        names.insert(item.name);
    }
    return !segmentation || ids.contains(0) ||
           fail(error, QStringLiteral("Segmentation classes require ID 0."));
}

QJsonArray classJson(const QVector<PointCloudDatasetImportClass>& classes) {
    QJsonArray array;
    for (const auto& item : classes)
        array.append(QJsonObject{
                {QStringLiteral("id"), item.id},
                {QStringLiteral("name"), item.name},
                {QStringLiteral("color"), item.color.name().toUpper()}});
    return array;
}

bool segmentationClasses(const QByteArray& bytes,
                         bool existed,
                         const QVector<PointCloudDatasetImportClass>& current,
                         QVector<PointCloudDatasetImportClass>* classes,
                         QString* error) {
    *classes = current;
    bool explicitSection = false;
    if (existed) {
        QJsonParseError parse;
        const QJsonDocument doc = QJsonDocument::fromJson(bytes, &parse);
        const QJsonObject root = doc.object();
        if (parse.error != QJsonParseError::NoError || !doc.isObject() ||
            root.value(QStringLiteral("version")).toDouble(-1) != 1)
            return fail(
                    error,
                    QStringLiteral("Invalid existing class configuration."));
        QJsonValue entries;
        if (root.contains(QStringLiteral("segmentation")))
            entries = root.value(QStringLiteral("segmentation"))
                              .toObject()
                              .value(QStringLiteral("classes"));
        else if (root.contains(QStringLiteral("classes")))
            entries = root.value(QStringLiteral("classes"));
        else if (!root.contains(QStringLiteral("detection")))
            return fail(
                    error,
                    QStringLiteral(
                            "Class configuration has no task definitions."));
        if (!entries.isUndefined()) {
            explicitSection = true;
            if (!entries.isArray())
                return fail(error,
                            QStringLiteral(
                                    "Segmentation classes must be an array."));
            classes->clear();
            for (const QJsonValue& value : entries.toArray()) {
                const QJsonObject entry = value.toObject();
                const double id =
                        entry.value(QStringLiteral("id")).toDouble(-1);
                if (!std::isfinite(id) || std::floor(id) != id || id < 0 ||
                    id > 65535)
                    return fail(
                            error,
                            QStringLiteral("Invalid segmentation class ID."));
                classes->append({int(id),
                                 entry.value(QStringLiteral("name")).toString(),
                                 QColor(entry.value(QStringLiteral("color"))
                                                .toString())});
            }
        }
    }
    const bool hasZero =
            std::any_of(classes->cbegin(), classes->cend(),
                        [](const PointCloudDatasetImportClass& entry) {
                            return entry.id == 0;
                        });
    if (!explicitSection && !hasZero)
        *classes = {{0, QStringLiteral("Unlabeled"),
                     QColor(QStringLiteral("#808080"))}};
    return validateClasses(*classes, true, error);
}

bool unchanged(const PointCloudDatasetImportFileChange& change,
               QString* error) {
    QByteArray bytes;
    bool existed = false;
    if (!readExisting(change.path, &bytes, &existed, error)) return false;
    return (existed == change.existed && bytes == change.before) ||
           fail(error, QStringLiteral("File changed after import preview: %1")
                               .arg(change.path));
}

bool syncFile(QFile& file, QString* error) {
    if (!file.flush()) return fail(error, file.errorString());
#ifdef Q_OS_WIN
    const int result = _commit(int(file.handle()));
#else
    const int result = fsync(int(file.handle()));
#endif
    return result == 0 ||
           fail(error, QStringLiteral("Could not sync staged import file: %1")
                               .arg(file.fileName()));
}
}  // namespace

bool preparePointCloudDatasetImport(
        const QString& archivePath,
        const QString& format,
        const QVector<PointCloudDatasetImportTarget>& targets,
        const QVector<PointCloudDatasetImportClass>& currentClasses,
        const QString& configPath,
        PointCloudDatasetImportPlan* output,
        const std::function<bool(int, int)>& callback,
        QString* error,
        const QVector<PointCloudDatasetImportClass>* segmentationOverride) {
    if (error) error->clear();
    if (!output || targets.isEmpty())
        return fail(error, QStringLiteral("Point-cloud import requires targets "
                                          "and a plan output."));
    if (!progress(callback, 0, targets.size() + 1, error)) return false;
    PointCloudDatasetImportResult imported;
    if (!importPointCloudDataset(
                archivePath, format, &imported,
                [&](int, int) {
                    return progress(callback, 0, targets.size() + 1, error);
                },
                error))
        return false;
    const int total = targets.size() + imported.frames.size() + 1;
    int completed = 0;
    if (!progress(callback, completed, total, error)) return false;
    const QStringList names = exportedFrameNames(targets);
    QSet<QString> importedNames;
    for (const auto& frame : imported.frames) importedNames.insert(frame.name);
    const QSet<QString> exportedNames =
                qtCompatQSetFromVector(names.toVector());
    const bool exact = importedNames == exportedNames;
    QMap<QString, QVector<int>> byName;
    QSet<QString> protectedPaths{resolvedPath(archivePath)};
    for (int index = 0; index < targets.size(); ++index) {
        const auto& target = targets.at(index);
        if (target.path.isEmpty() || target.labelPath.isEmpty() ||
            target.cuboidPath.isEmpty())
            return fail(error, QStringLiteral("Import targets require cloud, "
                                              "label, and cuboid paths."));
        byName[QFileInfo(target.path).completeBaseName()].append(index);
        protectedPaths.insert(resolvedPath(target.path));
        protectedPaths.insert(resolvedPath(target.labelPath));
    }
    QVector<int> matched;
    QSet<int> seen;
    for (const auto& frame : imported.frames) {
        int index = frame.index;
        if (!frame.name.isEmpty()) {
            if (exact)
                index = names.indexOf(frame.name);
            else {
                const QVector<int> candidates =
                        byName.value(frameKey(frame.name));
                if (candidates.size() != 1)
                    return fail(
                            error,
                            QStringLiteral(
                                    "Cannot uniquely match imported frame: %1")
                                    .arg(frame.name));
                index = candidates.front();
            }
        }
        if (index < 0 || index >= targets.size() || seen.contains(index))
            return fail(error,
                        QStringLiteral("Imported frame index is duplicate or "
                                       "outside the current sequence."));
        matched.append(index);
        seen.insert(index);
    }
    PointCloudDatasetImportPlan plan;
    QVector<PointCloudDatasetImportClass> segmentation;
    for (const auto& item : currentClasses) {
        segmentation.append(item);
        if (item.id) plan.classes.append(item);
    }
    if (!validateClasses(plan.classes, false, error)) return false;
    QMap<QString, int> byClassName;
    QSet<int> used;
    for (const auto& item : plan.classes) {
        byClassName.insert(item.name, item.id);
        used.insert(item.id);
    }
    QVector<QVector<PointCloudCuboid>> existing;
    for (const auto& target : targets) {
        QVector<PointCloudCuboid> boxes;
        if (!targetCuboids(target, &boxes, error)) return false;
        for (const auto& box : boxes) used.insert(box.classId);
        existing.append(boxes);
        if (!progress(callback, ++completed, total, error)) return false;
    }
    QMap<int, int> mapping;
    const int previousClasses = plan.classes.size();
    for (const auto& item : imported.classes) {
        if (!byClassName.contains(item.name)) {
            int id =
                    used.isEmpty()
                            ? 1
                            : *std::max_element(used.cbegin(), used.cend()) + 1;
            if (id > 65535) {
                id = 1;
                while (id <= 65535 && used.contains(id)) ++id;
            }
            if (id > 65535)
                return fail(
                        error,
                        QStringLiteral(
                                "No class IDs remain for imported classes."));
            plan.classes.append({id, item.name, item.color});
            byClassName.insert(item.name, id);
            used.insert(id);
        }
        mapping.insert(item.id, byClassName.value(item.name));
    }
    plan.newClasses = plan.classes.size() - previousClasses;
    QSet<QString> destinations;
    const auto addChange = [&](const QString& path, const QByteArray& data) {
        const QString resolved = resolvedPath(path);
        if (protectedPaths.contains(resolved) ||
            destinations.contains(resolved))
            return fail(error,
                        QStringLiteral("Conflicting import destination: %1")
                                .arg(path));
        PointCloudDatasetImportFileChange change;
        change.path = QFileInfo(path).absoluteFilePath();
        change.after = data;
        if (!readExisting(change.path, &change.before, &change.existed, error))
            return false;
        destinations.insert(resolved);
        plan.changes.append(change);
        return true;
    };
    for (int frameIndex = 0; frameIndex < imported.frames.size();
         ++frameIndex) {
        const int index = matched.at(frameIndex);
        auto target = targets.at(index);
        if (!target.cuboidPath.endsWith(QStringLiteral(".cuboids.json")))
            return fail(error, QStringLiteral("Cuboid output must use the "
                                              ".cuboids.json extension."));
        plan.previousObjects += existing.at(index).size();
        target.cuboids = imported.frames.at(frameIndex).cuboids;
        target.cuboidsProvided = true;
        QJsonArray boxes;
        QSet<int> ids;
        for (auto& box : target.cuboids) {
            if (!mapping.contains(box.classId))
                return fail(error,
                            QStringLiteral("Unknown imported cuboid class."));
            box.classId = mapping.value(box.classId);
            if (!box.valid(error) || ids.contains(box.id))
                return fail(
                        error,
                        QStringLiteral(
                                "Invalid or duplicate imported cuboid ID."));
            ids.insert(box.id);
            boxes.append(box.toJson());
        }
        const QJsonObject sidecar{
                {QStringLiteral("version"), 1},
                {QStringLiteral("type"), QStringLiteral("pointcloud-cuboids")},
                {QStringLiteral("point_cloud"),
                 QFileInfo(target.path).fileName()},
                {QStringLiteral("cuboids"), boxes}};
        if (!addChange(target.cuboidPath, QJsonDocument(sidecar).toJson()))
            return false;
        plan.targets.append(target);
        if (!progress(callback, ++completed, total, error)) return false;
    }
    if (!configPath.isEmpty()) {
        if (QFileInfo(configPath)
                    .suffix()
                    .compare(QStringLiteral("json"), Qt::CaseInsensitive) != 0)
            return fail(error, QStringLiteral("Class configuration must use "
                                              "the .json extension."));
        QByteArray bytes;
        bool existed = false;
        if (!readExisting(configPath, &bytes, &existed, error)) return false;
        if (segmentationOverride) {
            segmentation = *segmentationOverride;
            if (!validateClasses(segmentation, true, error)) return false;
        } else if (!segmentationClasses(bytes, existed, segmentation,
                                        &segmentation, error))
            return false;
        const QJsonObject config{{QStringLiteral("version"), 1},
                                 {QStringLiteral("detection"),
                                  QJsonObject{{QStringLiteral("classes"),
                                               classJson(plan.classes)}}},
                                 {QStringLiteral("segmentation"),
                                  QJsonObject{{QStringLiteral("classes"),
                                               classJson(segmentation)}}}};
        if (!addChange(configPath, QJsonDocument(config).toJson()))
            return false;
        if (plan.changes.last().before != bytes ||
            plan.changes.last().existed != existed)
            return fail(error, QStringLiteral("Class configuration changed "
                                              "during import preparation."));
    }
    if (!progress(callback, total, total, error)) return false;
    *output = plan;
    return true;
}

bool applyPointCloudDatasetImport(const PointCloudDatasetImportPlan& plan,
                                  const std::function<bool(int, int)>& callback,
                                  QString* error) {
    if (error) error->clear();
    if (plan.changes.isEmpty())
        return fail(error,
                    QStringLiteral("Import plan contains no file changes."));
    QStringList staged;
    QVector<int> committed;
    QMap<int, QString> backups;
    QSet<QString> destinations;
    const auto cleanup = [&] {
        for (const auto& path : staged) QFile::remove(path);
    };
    const auto rollback = [&] {
        QStringList failures;
        for (auto it = committed.crbegin(); it != committed.crend(); ++it) {
            const QString path = plan.changes.at(*it).path;
            if ((QFile::exists(path) && !QFile::remove(path)) ||
                (backups.contains(*it) &&
                 !QFile::rename(backups.value(*it), path))) {
                failures.append(path);
            } else
                backups.remove(*it);
        }
        cleanup();
        if (!failures.isEmpty()) {
            const QString detail =
                    QStringLiteral(
                            "Import rollback failed for %1. Originals retained "
                            "at: %2")
                            .arg(failures.join(QStringLiteral(", ")),
                                 backups.values().join(QStringLiteral(", ")));
            if (error) *error += QLatin1Char('\n') + detail;
        }
        return false;
    };
    for (const auto& change : plan.changes) {
        if (change.path.isEmpty() ||
            destinations.contains(resolvedPath(change.path))) {
            fail(error,
                 QStringLiteral("Duplicate or empty import destination."));
            return rollback();
        }
        destinations.insert(resolvedPath(change.path));
        if (!progress(callback, 0, plan.changes.size(), error) ||
            !unchanged(change, error))
            return rollback();
        const QDir directory = QFileInfo(change.path).absoluteDir();
        if (!QDir().mkpath(directory.absolutePath())) {
            fail(error,
                 QStringLiteral("Could not create import destination folder."));
            return rollback();
        }
        QTemporaryFile temporary(directory.filePath(
                QStringLiteral(".%1.import.XXXXXX")
                        .arg(QFileInfo(change.path).fileName())));
        if (!temporary.open() ||
            temporary.write(change.after) != change.after.size() ||
            !syncFile(temporary, error)) {
            if (error && error->isEmpty()) *error = temporary.errorString();
            return rollback();
        }
        staged.append(temporary.fileName());
        temporary.setAutoRemove(false);
    }
    for (int index = 0; index < plan.changes.size(); ++index) {
        const auto& change = plan.changes.at(index);
        if (!progress(callback, index, plan.changes.size(), error) ||
            !unchanged(change, error))
            return rollback();
        if (change.existed) {
            QTemporaryFile backup(
                    QFileInfo(change.path)
                            .absoluteDir()
                            .filePath(QStringLiteral(".%1.backup.XXXXXX")
                                              .arg(QFileInfo(change.path)
                                                           .fileName())));
            if (!backup.open()) {
                fail(error, backup.errorString());
                return rollback();
            }
            const QString backupPath = backup.fileName();
            backup.close();
            if (!backup.remove() || !QFile::rename(change.path, backupPath)) {
                fail(error,
                     QStringLiteral("Could not back up import destination: %1")
                             .arg(change.path));
                return rollback();
            }
            backup.setAutoRemove(false);
            backups.insert(index, backupPath);
        }
        committed.append(index);
        if (!QFile::rename(staged.at(index), change.path)) {
            fail(error,
                 QStringLiteral("Could not install imported annotations: %1")
                         .arg(change.path));
            return rollback();
        }
    }
    if (!progress(callback, plan.changes.size(), plan.changes.size(), error))
        return rollback();
    cleanup();
    for (const auto& path : backups) QFile::remove(path);
    return true;
}
