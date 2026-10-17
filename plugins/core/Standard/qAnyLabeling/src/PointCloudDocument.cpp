// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudDocument.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>
#include <QSet>
#include <QtEndian>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
struct Property {
    QByteArray name;
    QByteArray type;
    QByteArray countType;
};
struct Element {
    QByteArray name;
    qint64 count = 0;
    QVector<Property> properties;
};

int scalarSize(const QByteArray& type) {
    if (type == "char" || type == "int8" || type == "uchar" || type == "uint8")
        return 1;
    if (type == "short" || type == "int16" || type == "ushort" ||
        type == "uint16")
        return 2;
    if (type == "int" || type == "int32" || type == "uint" ||
        type == "uint32" || type == "float" || type == "float32")
        return 4;
    if (type == "double" || type == "float64") return 8;
    return 0;
}

bool scalar(QFile& file, const QByteArray& type, bool little, double* value) {
    const int size = scalarSize(type);
    char bytes[8] = {};
    if (!size || file.read(bytes, size) != size) return false;
    quint64 bits = 0;
    for (int i = 0; i < size; ++i) {
        const int shift = little ? 8 * i : 8 * (size - i - 1);
        bits |= quint64(quint8(bytes[i])) << shift;
    }
    if (type == "float" || type == "float32") {
        const quint32 raw = quint32(bits);
        float number;
        std::memcpy(&number, &raw, sizeof(number));
        *value = number;
    } else if (type == "double" || type == "float64") {
        std::memcpy(value, &bits, sizeof(bits));
    } else if (type == "char" || type == "int8")
        *value = qint8(bits);
    else if (type == "short" || type == "int16")
        *value = qint16(bits);
    else if (type == "int" || type == "int32")
        *value = qint32(bits);
    else
        *value = double(bits);
    return true;
}

bool readPly(QFile& file, PointCloudFrame* frame, QString* error) {
    if (file.readLine().trimmed() != "ply") {
        *error = QStringLiteral("Missing PLY header");
        return false;
    }
    QByteArray encoding;
    QVector<Element> elements;
    bool ended = false;
    while (file.pos() < 1024 * 1024 && !file.atEnd()) {
        const QList<QByteArray> fields =
                file.readLine(65536).simplified().split(' ');
        if (fields.isEmpty() || fields[0] == "comment" ||
            fields[0] == "obj_info")
            continue;
        if (fields[0] == "end_header") {
            ended = true;
            break;
        }
        if (fields[0] == "format" && fields.size() == 3 && fields[2] == "1.0" &&
            encoding.isEmpty())
            encoding = fields[1];
        else if (fields[0] == "element" && fields.size() == 3) {
            bool ok = false;
            const qint64 count = fields[2].toLongLong(&ok);
            if (!ok || count < 0 || count > std::numeric_limits<int>::max())
                break;
            Element element;
            element.name = fields[1];
            element.count = count;
            elements.append(element);
        } else if (fields[0] == "property" && !elements.isEmpty()) {
            Property property;
            if (fields.size() == 3) {
                property.type = fields[1];
                property.name = fields[2];
            } else if (fields.size() == 5 && fields[1] == "list") {
                property.countType = fields[2];
                property.type = fields[3];
                property.name = fields[4];
            } else
                break;
            if (!scalarSize(property.type) || (!property.countType.isEmpty() &&
                                               !scalarSize(property.countType)))
                break;
            elements.last().properties.append(property);
        } else
            break;
    }
    if (!ended || (encoding != "ascii" && encoding != "binary_little_endian" &&
                   encoding != "binary_big_endian")) {
        *error = QStringLiteral("Unsupported PLY header");
        return false;
    }
    const bool ascii = encoding == "ascii";
    const bool little = encoding == "binary_little_endian";
    bool vertices = false;
    for (const Element& element : elements) {
        if (element.name == "vertex") {
            if (vertices || !element.count) {
                *error = QStringLiteral("Invalid PLY vertices");
                return false;
            }
            vertices = true;
            QSet<QByteArray> names;
            for (const Property& property : element.properties) {
                if (!property.countType.isEmpty() ||
                    names.contains(property.name)) {
                    *error = QStringLiteral("Invalid PLY vertex property");
                    return false;
                }
                names.insert(property.name);
            }
            if (!names.contains("x") || !names.contains("y") ||
                !names.contains("z")) {
                *error = QStringLiteral("PLY vertices need x, y and z");
                return false;
            }
            frame->hasIntensity = names.contains("intensity");
            frame->points.reserve(int(element.count));
            if (names.contains("red") && names.contains("green") &&
                names.contains("blue"))
                frame->rgb.reserve(int(element.count));
        }
        for (qint64 row = 0; row < element.count; ++row) {
            const QList<QByteArray> tokens =
                    ascii ? file.readLine().simplified().split(' ')
                          : QList<QByteArray>{};
            int cursor = 0;
            double x = 0, y = 0, z = 0, intensity = 0, red = 0, green = 0,
                   blue = 0;
            for (const Property& property : element.properties) {
                double countValue = 1;
                if (!property.countType.isEmpty()) {
                    if (ascii) {
                        if (cursor >= tokens.size()) {
                            *error = QStringLiteral("Truncated PLY list");
                            return false;
                        }
                        bool ok = false;
                        countValue = tokens[cursor++].toDouble(&ok);
                        if (!ok) {
                            *error = QStringLiteral("Invalid PLY list");
                            return false;
                        }
                    } else if (!scalar(file, property.countType, little,
                                       &countValue)) {
                        *error = QStringLiteral("Truncated PLY list");
                        return false;
                    }
                }
                if (!std::isfinite(countValue) || countValue < 0 ||
                    countValue > file.size() ||
                    std::floor(countValue) != countValue) {
                    *error = QStringLiteral("Invalid PLY list length");
                    return false;
                }
                const qint64 count = qint64(countValue);
                if (!ascii && property.countType.isEmpty() &&
                    file.bytesAvailable() < scalarSize(property.type)) {
                    *error = QStringLiteral("Truncated PLY data");
                    return false;
                }
                for (qint64 item = 0; item < count; ++item) {
                    double value = 0;
                    if (ascii) {
                        if (cursor >= tokens.size()) {
                            *error = QStringLiteral("Truncated PLY data");
                            return false;
                        }
                        bool ok = false;
                        value = tokens[cursor++].toDouble(&ok);
                        if (!ok) {
                            *error = QStringLiteral("Invalid PLY scalar");
                            return false;
                        }
                    } else if (!scalar(file, property.type, little, &value)) {
                        *error = QStringLiteral("Truncated PLY data");
                        return false;
                    }
                    if (element.name != "vertex") continue;
                    if (property.name == "x")
                        x = value;
                    else if (property.name == "y")
                        y = value;
                    else if (property.name == "z")
                        z = value;
                    else if (property.name == "intensity")
                        intensity = value;
                    else if (property.name == "red")
                        red = value;
                    else if (property.name == "green")
                        green = value;
                    else if (property.name == "blue")
                        blue = value;
                }
            }
            if (ascii && cursor != tokens.size()) {
                *error = QStringLiteral("Extra PLY fields");
                return false;
            }
            if (element.name == "vertex") {
                if (!std::isfinite(x) || !std::isfinite(y) ||
                    !std::isfinite(z) ||
                    std::abs(x) > std::numeric_limits<float>::max() ||
                    std::abs(y) > std::numeric_limits<float>::max() ||
                    std::abs(z) > std::numeric_limits<float>::max()) {
                    *error = QStringLiteral("Invalid PLY point coordinate");
                    return false;
                }
                frame->points.append(QVector4D(float(x), float(y), float(z),
                                               float(intensity)));
                if (frame->rgb.capacity())
                    frame->rgb.append(
                            QColor(int(std::clamp(red, 0.0, 255.0)),
                                   int(std::clamp(green, 0.0, 255.0)),
                                   int(std::clamp(blue, 0.0, 255.0))));
            }
        }
    }
    if (!vertices || !file.atEnd()) {
        *error = QStringLiteral("Incomplete PLY records");
        return false;
    }
    return true;
}
}  // namespace

bool loadPointCloudFrame(const QString& path,
                         PointCloudFrame* frame,
                         QString* error,
                         const QString& labelPath) {
    if (!frame || !error) return false;
    error->clear();
    PointCloudFrame loaded;
    loaded.path = QFileInfo(path).absoluteFilePath();
    QFile file(loaded.path);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = file.errorString();
        return false;
    }
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == QStringLiteral("bin")) {
        if (!file.size() || file.size() % 16 ||
            file.size() / 16 > std::numeric_limits<int>::max()) {
            *error = QStringLiteral(
                    "BIN requires nonempty 16-byte XYZI records");
            return false;
        }
        loaded.points.reserve(int(file.size() / 16));
        while (!file.atEnd()) {
            const QByteArray row = file.read(16);
            if (row.size() != 16) {
                *error = QStringLiteral("Truncated BIN record");
                return false;
            }
            float values[4];
            for (int k = 0; k < 4; ++k) {
                const quint32 bits = qFromLittleEndian<quint32>(
                        reinterpret_cast<const uchar*>(row.constData() +
                                                       k * 4));
                std::memcpy(&values[k], &bits, 4);
            }
            if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
                !std::isfinite(values[2])) {
                *error = QStringLiteral("BIN has nonfinite point coordinates");
                return false;
            }
            loaded.points.append(
                    QVector4D(values[0], values[1], values[2], values[3]));
        }
    } else if (suffix == QStringLiteral("ply")) {
        if (!readPly(file, &loaded, error)) return false;
    } else {
        *error = QStringLiteral("Expected .bin or .ply point cloud");
        return false;
    }
    const QFileInfo source(loaded.path);
    const QDir base =
            source.dir().dirName() == QStringLiteral("velodyne")
                    ? QDir(source.dir().absolutePath() + QStringLiteral("/.."))
                    : source.dir();
    const QString direct = source.dir().filePath(source.completeBaseName() +
                                                 QStringLiteral(".label"));
    const QString sibling =
            base.filePath(QStringLiteral("labels/") +
                          source.completeBaseName() + QStringLiteral(".label"));
    loaded.labelPath = labelPath.isEmpty() ? (QFile::exists(direct)    ? direct
                                              : QFile::exists(sibling) ? sibling
                                                                       : direct)
                                           : labelPath;
    loaded.labels.fill(0, loaded.points.size());
    QFile labels(loaded.labelPath);
    if (labels.exists()) {
        if (!labels.open(QIODevice::ReadOnly)) {
            *error = labels.errorString();
            return false;
        }
        if (labels.size() != qint64(loaded.points.size()) * 4) {
            *error = QStringLiteral("Label count does not match point count");
            return false;
        }
        for (int i = 0; i < loaded.labels.size(); ++i) {
            char raw[4];
            if (labels.read(raw, 4) != 4) {
                *error = QStringLiteral("Truncated labels");
                return false;
            }
            loaded.labels[i] = qFromLittleEndian<quint32>(
                    reinterpret_cast<const uchar*>(raw));
        }
        loaded.labelExists = true;
    }
    // Cuboid sidecar: optional, best-effort (upstream parity — a missing
    // or unparseable sidecar leaves an empty list; a hard parse error is
    // surfaced like the label read above).
    loaded.cuboids.clear();
    const QString cuboidSidecar =
            QFileInfo(loaded.labelPath)
                    .dir()
                    .filePath(QFileInfo(loaded.labelPath).completeBaseName() +
                              QStringLiteral(".cuboids.json"));
    QFile cuboidFile(cuboidSidecar);
    if (cuboidFile.exists()) {
        QString cuboidError;
        QVector<PointCloudCuboid> boxes;
        if (!loadPointCloudCuboids(cuboidSidecar,
                                   QFileInfo(loaded.path).fileName(), &boxes,
                                   &cuboidError)) {
            *error = cuboidError;
            return false;
        }
        loaded.cuboids = boxes;
    }
    *frame = std::move(loaded);
    return true;
}

bool savePointCloudLabels(const PointCloudFrame& frame,
                          QString* error,
                          const QString& outputPath) {
    if (!error) return false;
    error->clear();
    const QString path = outputPath.isEmpty() ? frame.labelPath : outputPath;
    if (QFileInfo(path).suffix().compare(QStringLiteral("label"),
                                         Qt::CaseInsensitive) != 0 ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(frame.path).absoluteFilePath() ||
        frame.labels.size() != frame.points.size()) {
        *error = QStringLiteral("Invalid point label output");
        return false;
    }
    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        *error = output.errorString();
        return false;
    }
    for (quint32 label : frame.labels) {
        char raw[4];
        qToLittleEndian<quint32>(label, reinterpret_cast<uchar*>(raw));
        if (output.write(raw, 4) != 4) {
            *error = output.errorString();
            return false;
        }
    }
    if (!output.commit()) {
        *error = output.errorString();
        return false;
    }
    // Cuboid sidecar: written alongside the labels whenever the frame
    // carries any cuboids (upstream save_cuboids parity).
    if (!frame.cuboids.isEmpty()) {
        const QString cuboidSidecar = QFileInfo(path).dir().filePath(
                QFileInfo(path).completeBaseName() +
                QStringLiteral(".cuboids.json"));
        QString cuboidError;
        if (!savePointCloudCuboids(cuboidSidecar,
                                   QFileInfo(frame.path).fileName(),
                                   frame.cuboids, &cuboidError)) {
            *error = cuboidError;
            return false;
        }
    }
    return true;
}

void PointCloudDocument::setFrame(PointCloudFrame frame) {
    m_frame = std::move(frame);
    m_baseline = m_frame.labels;
    m_undo.clear();
    m_redo.clear();
    m_dirty = false;
}

void PointCloudDocument::markSaved() {
    m_baseline = m_frame.labels;
    m_dirty = false;
}

bool PointCloudDocument::assign(const QVector<int>& indices,
                                quint16 semantic,
                                quint16 instance) {
    Edit edit;
    QSet<int> seen;
    const quint32 value = quint32(semantic) | (quint32(instance) << 16);
    for (int index : indices) {
        if (index < 0 || index >= m_frame.labels.size()) return false;
        if (seen.contains(index) || m_frame.labels[index] == value) continue;
        seen.insert(index);
        edit.indices.append(index);
        edit.before.append(m_frame.labels[index]);
        edit.after.append(value);
    }
    if (edit.indices.isEmpty()) return false;
    apply(edit, true);
    m_undo.append(std::move(edit));
    if (m_undo.size() > 50) m_undo.removeFirst();
    m_redo.clear();
    return true;
}

bool PointCloudDocument::operate(const QVector<int>& indices,
                                 quint16 semantic,
                                 quint16 instance,
                                 Operation operation,
                                 QString* error) {
    if (error) error->clear();
    if (semantic == 0 && operation != AssignSemantic) {
        if (error)
            *error = QStringLiteral("Instances cannot use the unlabeled class");
        return false;
    }
    QSet<int> seen;
    QVector<int> selected;
    for (int index : indices) {
        if (index < 0 || index >= m_frame.labels.size()) {
            if (error)
                *error = QStringLiteral("Point selection is outside the frame");
            return false;
        }
        if (!seen.contains(index)) {
            seen.insert(index);
            selected.append(index);
        }
    }
    if (selected.isEmpty()) return false;
    quint16 targetInstance = instance;
    if (operation == CreateInstance || operation == SplitInstance) {
        targetInstance = 0;
        QSet<quint16> used;
        for (quint32 label : m_frame.labels)
            if ((label & 0xffffu) == semantic && (label >> 16))
                used.insert(quint16(label >> 16));
        for (quint32 candidate = 1; candidate <= 65535; ++candidate)
            if (!used.contains(quint16(candidate))) {
                targetInstance = quint16(candidate);
                break;
            }
        if (!targetInstance) {
            if (error)
                *error =
                        QStringLiteral("No instance ID remains for this class");
            return false;
        }
    }
    if (operation == RemoveFromInstance) targetInstance = 0;
    if (operation == SplitInstance) {
        const quint32 source = quint32(instance) << 16 | semantic;
        int total = 0;
        for (quint32 label : m_frame.labels) total += label == source;
        int selectedSource = 0;
        for (int index : selected)
            selectedSource += m_frame.labels[index] == source;
        if (!instance || selectedSource == 0 || selectedSource >= total) {
            if (error)
                *error = QStringLiteral(
                        "Split requires a proper subset of an instance");
            return false;
        }
    }
    Edit edit;
    const quint32 source = (quint32(instance) << 16) | semantic;
    for (int index : selected) {
        const quint32 before = m_frame.labels[index];
        quint32 after = before;
        if (operation == AssignSemantic)
            after = semantic;
        else if (operation == CreateInstance) {
            if ((before & 0xffffu) != semantic) continue;
            after = (quint32(targetInstance) << 16) | semantic;
        } else if (operation == AddToInstance) {
            if ((before & 0xffffu) != semantic || !instance) continue;
            after = source;
        } else if (operation == RemoveFromInstance) {
            if (before != source) continue;
            after = semantic;
        } else if (operation == SplitInstance) {
            if (before != source) continue;
            after = (quint32(targetInstance) << 16) | semantic;
        }
        if (after == before) continue;
        edit.indices.append(index);
        edit.before.append(before);
        edit.after.append(after);
    }
    if (edit.indices.isEmpty()) return false;
    apply(edit, true);
    m_undo.append(std::move(edit));
    if (m_undo.size() > 50) m_undo.removeFirst();
    m_redo.clear();
    return true;
}

void PointCloudDocument::apply(const Edit& edit, bool forward) {
    for (int i = 0; i < edit.indices.size(); ++i)
        m_frame.labels[edit.indices[i]] =
                forward ? edit.after[i] : edit.before[i];
    m_dirty = m_frame.labels != m_baseline;
}

bool PointCloudDocument::undo() {
    if (m_undo.isEmpty()) return false;
    Edit edit = m_undo.takeLast();
    apply(edit, false);
    m_redo.append(std::move(edit));
    return true;
}

bool PointCloudDocument::redo() {
    if (m_redo.isEmpty()) return false;
    Edit edit = m_redo.takeLast();
    apply(edit, true);
    m_undo.append(std::move(edit));
    return true;
}
