// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAssetIntegrity.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QSaveFile>
#include <cstring>

// GGUF file format magic. The first 4 bytes of every valid GGUF file are
// the ASCII characters "GGUF" (0x46475547 in little-endian). Checking it
// rejects HTML error pages and empty/truncated junk without any hashing.
static constexpr const char kGgufMagic[4] = {'G', 'G', 'U', 'F'};

namespace {

constexpr int kLedgerVersion = 1;

QString ledgerPath(const QString& path) {
    return path + QStringLiteral(".cvintegrity");
}

QString algoName(QCryptographicHash::Algorithm algo) {
    switch (algo) {
        case QCryptographicHash::Md5:
            return QStringLiteral("md5");
        case QCryptographicHash::Sha1:
            return QStringLiteral("sha1");
        case QCryptographicHash::Sha256:
            return QStringLiteral("sha256");
        default:
            return QStringLiteral("algo-%1").arg(static_cast<int>(algo));
    }
}

bool algoFromName(const QString& name, QCryptographicHash::Algorithm* out) {
    if (name == QLatin1String("md5")) {
        *out = QCryptographicHash::Md5;
        return true;
    }
    if (name == QLatin1String("sha1")) {
        *out = QCryptographicHash::Sha1;
        return true;
    }
    if (name == QLatin1String("sha256")) {
        *out = QCryptographicHash::Sha256;
        return true;
    }
    return false;
}

bool hasGgufMagic(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    char head[4] = {0, 0, 0, 0};
    const qint64 read = f.read(head, sizeof(head));
    f.close();
    if (read != sizeof(head)) return false;
    return std::memcmp(head, kGgufMagic, sizeof(kGgufMagic)) == 0;
}

struct LedgerEntry {
    QCryptographicHash::Algorithm algo = QCryptographicHash::Sha256;
    QByteArray digestHex;
    qint64 size = -1;
    qint64 mtimeMs = -1;
};

bool readLedger(const QString& path, LedgerEntry* out) {
    QFile f(ledgerPath(path));
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    f.close();
    if (!doc.isObject()) return false;
    const QJsonObject obj = doc.object();
    if (obj.value(QStringLiteral("version")).toInt() != kLedgerVersion) {
        return false;
    }
    if (!algoFromName(obj.value(QStringLiteral("algo")).toString(),
                      &out->algo)) {
        return false;
    }
    out->digestHex = obj.value(QStringLiteral("digest")).toString().toLatin1();
    out->size = static_cast<qint64>(
            obj.value(QStringLiteral("size")).toDouble());
    out->mtimeMs = static_cast<qint64>(
            obj.value(QStringLiteral("mtime_ms")).toDouble());
    return out->size >= 0 && out->mtimeMs >= 0;
}

bool writeLedger(const QString& path, const LedgerEntry& entry) {
    QJsonObject obj;
    obj.insert(QStringLiteral("version"), kLedgerVersion);
    obj.insert(QStringLiteral("algo"), algoName(entry.algo));
    obj.insert(QStringLiteral("digest"),
               QString::fromLatin1(entry.digestHex));
    obj.insert(QStringLiteral("size"), static_cast<double>(entry.size));
    obj.insert(QStringLiteral("mtime_ms"),
               static_cast<double>(entry.mtimeMs));

    QSaveFile file(ledgerPath(path));
    if (!file.open(QIODevice::WriteOnly)) return false;
    file.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    return file.commit();
}

}  // namespace

bool ecvAssetIntegrity::passesCheapChecks(const QString& path,
                                          qint64 minBytes,
                                          bool requireGgufMagic) {
    const QFileInfo fi(path);
    if (!fi.isFile()) return false;
    if (minBytes > 0 && fi.size() < minBytes) return false;
    if (requireGgufMagic && !hasGgufMagic(path)) return false;
    return true;
}

bool ecvAssetIntegrity::verifyNow(const QString& path, const Anchor& anchor) {
    if (anchor.digestHex.isEmpty()) return false;  // nothing to compare
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return false;
    QCryptographicHash hash(anchor.algo);
    if (!hash.addData(&file)) return false;
    file.close();
    const QByteArray actual = hash.result().toHex();
    return actual.compare(anchor.digestHex, Qt::CaseInsensitive) == 0;
}

bool ecvAssetIntegrity::isVerified(const QString& path,
                                   const Anchor& anchor,
                                   qint64 minBytes,
                                   bool requireGgufMagic,
                                   OnMiss onMiss,
                                   qint64 noLedgerExactSize) {
    if (!passesCheapChecks(path, minBytes, requireGgufMagic)) {
        // Artifact missing or junk: any recorded evidence describes bytes
        // that are no longer there — drop it so orphan sidecars left by
        // manual deletions self-clean on the next check.
        invalidate(path);
        return false;
    }

    const QFileInfo fi(path);
    const qint64 mtimeMs = fi.lastModified().toMSecsSinceEpoch();

    LedgerEntry entry;
    if (readLedger(path, &entry)) {
        // An empty-pin query accepts any ledger evidence: a pinned entry is
        // strictly stronger than what the query itself could establish, so
        // trusting it never weakens the check (and must not destroy it).
        const bool digestCompatible = anchor.digestHex.isEmpty() ||
                entry.digestHex.compare(anchor.digestHex,
                                        Qt::CaseInsensitive) == 0;
        const bool algoCompatible =
                anchor.digestHex.isEmpty() || entry.algo == anchor.algo;
        const bool statMatches =
                entry.size == fi.size() && entry.mtimeMs == mtimeMs;
        // A matching entry means the exact bytes were verified once; the
        // file is trusted while size+mtime stay unchanged.
        if (digestCompatible && algoCompatible && statMatches) {
            return true;
        }
        if (!digestCompatible && !entry.digestHex.isEmpty() &&
            !anchor.digestHex.isEmpty() && statMatches) {
            // The current file generation was verified against a DIFFERENT
            // pin: those bytes cannot match the new pin, so reject without
            // spending a hash pass — a source-side pin bump must never be
            // silently absorbed by the cheap-check path.
            return false;
        }
        // Stale entry (the file changed, or an empty-pin entry superseded
        // by a pinned query): drop it so the miss policy decides and a
        // fresh state gets recorded.
        invalidate(path);
    }

    if (anchor.digestHex.isEmpty()) {
        // No content pin: fall back to the legacy exact-size guard (when
        // provided) — same trust level as the per-access checks it replaces.
        return noLedgerExactSize > 0 ? fi.size() == noLedgerExactSize : true;
    }

    if (onMiss == OnMiss::DeepVerify) {
        if (verifyNow(path, anchor)) {
            markVerified(path, anchor);
            return true;
        }
        return false;
    }

    // CheapChecksOnly: never hash (multi-GB models on dialog-open paths).
    return noLedgerExactSize > 0 ? fi.size() == noLedgerExactSize : true;
}

bool ecvAssetIntegrity::markVerified(const QString& path,
                                     const Anchor& anchor) {
    const QFileInfo fi(path);
    if (!fi.isFile()) return false;
    LedgerEntry entry;
    entry.algo = anchor.algo;
    entry.digestHex = anchor.digestHex;  // may be empty (size-only entry)
    entry.size = fi.size();
    entry.mtimeMs = fi.lastModified().toMSecsSinceEpoch();
    return writeLedger(path, entry);
}

void ecvAssetIntegrity::removeIfNotVerified(const QString& path,
                                            const Anchor& anchor,
                                            qint64 minBytes,
                                            bool requireGgufMagic,
                                            OnMiss onMiss,
                                            qint64 noLedgerExactSize) {
    if (!isVerified(path, anchor, minBytes, requireGgufMagic, onMiss,
                    noLedgerExactSize)) {
        QFile::remove(path);
        invalidate(path);
    }
}

void ecvAssetIntegrity::invalidate(const QString& path) {
    QFile::remove(ledgerPath(path));
}

QByteArray ecvAssetIntegrity::PinnedDigest(const QString& fileName) {
    const char* digest = aicore::AssetDigestForFile(fileName.toUtf8().constData());
    return digest ? QByteArray(digest) : QByteArray();
}
