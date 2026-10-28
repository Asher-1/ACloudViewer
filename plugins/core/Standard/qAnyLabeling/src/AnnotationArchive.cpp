// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationArchive.h"

#include <zlib.h>

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QVector>
#include <limits>

#include "QtCompat.h"

namespace {
struct ZipEntry {
    QByteArray name;
    quint32 offset = 0;
    quint32 size = 0;
    quint32 compressed = 0;
    quint32 crc = 0;
};

bool write16(QFile& file, quint16 value) {
    const char bytes[] = {char(value), char(value >> 8)};
    return file.write(bytes, 2) == 2;
}

bool write32(QFile& file, quint32 value) {
    return write16(file, quint16(value)) && write16(file, quint16(value >> 16));
}

}  // namespace

bool writeZipDirectory(const QString& directory,
                       const QString& zipPath,
                       QString* error,
                       bool includeRootDirectory) {
    QFile zip(zipPath);
    if (!qtCompatOpenNewFile(zip, error)) return false;
    QVector<ZipEntry> entries;
    QDirIterator iterator(directory, QDir::Files | QDir::NoSymLinks,
                          QDirIterator::Subdirectories);
    const QDir root(directory);
    while (iterator.hasNext()) {
        const QString path = iterator.next();
        QFile input(path);
        const qint64 size = QFileInfo(path).size();
        if (size > std::numeric_limits<quint32>::max() ||
            zip.pos() > std::numeric_limits<quint32>::max() ||
            entries.size() >= std::numeric_limits<quint16>::max() ||
            !input.open(QIODevice::ReadOnly)) {
            if (error)
                *error = QStringLiteral(
                                 "ZIP64 is required or a file cannot be read: "
                                 "%1")
                                 .arg(path);
            zip.close();
            QFile::remove(zipPath);
            return false;
        }
        ZipEntry entry;
        entry.name = ((includeRootDirectory ? QFileInfo(directory).fileName() +
                                                      QLatin1Char('/')
                                            : QString()) +
                      root.relativeFilePath(path).replace(QLatin1Char('\\'),
                                                          QLatin1Char('/')))
                             .toUtf8();
        if (entry.name.size() > std::numeric_limits<quint16>::max()) break;
        entry.offset = quint32(zip.pos());
        entry.size = quint32(size);
        // Bit 3 allows CRC and sizes to be written after a streaming deflate.
        if (!write32(zip, 0x04034b50) || !write16(zip, 20) ||
            !write16(zip, 0x0808) || !write16(zip, 8) || !write16(zip, 0) ||
            !write16(zip, 0) || !write32(zip, 0) || !write32(zip, 0) ||
            !write32(zip, 0) || !write16(zip, quint16(entry.name.size())) ||
            !write16(zip, 0) || zip.write(entry.name) != entry.name.size())
            break;

        z_stream stream{};
        if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS,
                         8, Z_DEFAULT_STRATEGY) != Z_OK)
            break;
        bool success = true;
        int result = Z_OK;
        char output[65536];
        while (success && result != Z_STREAM_END) {
            const QByteArray chunk = input.read(65536);
            if (chunk.isEmpty() && !input.atEnd()) {
                success = false;
                break;
            }
            entry.crc = crc32(entry.crc,
                              reinterpret_cast<const Bytef*>(chunk.constData()),
                              uInt(chunk.size()));
            stream.next_in = reinterpret_cast<Bytef*>(
                    const_cast<char*>(chunk.constData()));
            stream.avail_in = uInt(chunk.size());
            const int flush = input.atEnd() ? Z_FINISH : Z_NO_FLUSH;
            do {
                stream.next_out = reinterpret_cast<Bytef*>(output);
                stream.avail_out = sizeof(output);
                result = deflate(&stream, flush);
                const qint64 produced = sizeof(output) - stream.avail_out;
                if (result == Z_STREAM_ERROR ||
                    zip.write(output, produced) != produced) {
                    success = false;
                    break;
                }
            } while (stream.avail_in > 0 || stream.avail_out == 0 ||
                     (flush == Z_FINISH && result != Z_STREAM_END));
        }
        const uLong totalIn = stream.total_in;
        const uLong totalOut = stream.total_out;
        deflateEnd(&stream);
        if (!success || totalIn != quint64(size) ||
            totalOut > std::numeric_limits<quint32>::max() ||
            zip.pos() > std::numeric_limits<quint32>::max() ||
            !write32(zip, 0x08074b50) || !write32(zip, entry.crc) ||
            !write32(zip, quint32(totalOut)) || !write32(zip, entry.size))
            break;
        entry.compressed = quint32(totalOut);
        entries.append(entry);
    }
    // A premature loop exit leaves fewer entries than files in the directory.
    QDirIterator countIterator(directory, QDir::Files | QDir::NoSymLinks,
                               QDirIterator::Subdirectories);
    int fileCount = 0;
    while (countIterator.hasNext()) {
        countIterator.next();
        ++fileCount;
    }
    bool success = entries.size() == fileCount && fileCount > 0;
    const qint64 centralOffset = zip.pos();
    for (const ZipEntry& entry : entries) {
        success = success && write32(zip, 0x02014b50) && write16(zip, 20) &&
                  write16(zip, 20) && write16(zip, 0x0808) && write16(zip, 8) &&
                  write16(zip, 0) && write16(zip, 0) &&
                  write32(zip, entry.crc) && write32(zip, entry.compressed) &&
                  write32(zip, entry.size) &&
                  write16(zip, quint16(entry.name.size())) && write16(zip, 0) &&
                  write16(zip, 0) && write16(zip, 0) && write16(zip, 0) &&
                  write32(zip, 0) && write32(zip, entry.offset) &&
                  zip.write(entry.name) == entry.name.size();
    }
    const qint64 centralSize = zip.pos() - centralOffset;
    success = success && centralOffset <= std::numeric_limits<quint32>::max() &&
              centralSize <= std::numeric_limits<quint32>::max() &&
              write32(zip, 0x06054b50) && write16(zip, 0) && write16(zip, 0) &&
              write16(zip, quint16(entries.size())) &&
              write16(zip, quint16(entries.size())) &&
              write32(zip, quint32(centralSize)) &&
              write32(zip, quint32(centralOffset)) && write16(zip, 0);
    if (!zip.flush()) success = false;
    zip.close();
    if (!success) {
        QFile::remove(zipPath);
        if (error) *error = QStringLiteral("Could not create ZIP archive");
    }
    return success;
}
