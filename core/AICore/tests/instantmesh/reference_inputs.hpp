// SPDX-License-Identifier: MIT
// Test-only: authenticate reference identity and all input digests before running production components.
#pragma once

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace instantmesh_reference {
using Files = std::vector<std::pair<const char*, size_t>>;

inline int authenticate(const std::filesystem::path& root,
                        const QJsonObject& identity, const Files& files) {
    const QDir directory(QString::fromUtf8(root.u8string().c_str()));
    QFile receipt(directory.filePath("provenance.json"));
    if (!receipt.exists()) {
        std::fprintf(stderr, "INCOMPLETE: missing reference provenance.json\n");
        return 77;
    }
    if (!receipt.open(QIODevice::ReadOnly)) return 1;
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(receipt.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        std::fprintf(stderr, "FAIL: invalid reference provenance JSON\n");
        return 1;
    }
    const auto metadata = document.object();
    for (auto it = identity.begin(); it != identity.end(); ++it) {
        if (metadata.value(it.key()) != it.value()) {
            std::fprintf(stderr, "FAIL: reference identity mismatch: %s\n",
                         it.key().toUtf8().constData());
            return 1;
        }
    }
    const auto digests = metadata.value("files_sha256").toObject();
    for (const auto& entry : files) {
        QFile file(directory.filePath(QString::fromUtf8(entry.first)));
        if (!file.exists()) {
            std::fprintf(stderr, "INCOMPLETE: missing reference %s\n", entry.first);
            return 77;
        }
        const auto expected = digests.value(QString::fromUtf8(entry.first)).toString().toLatin1();
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (expected.size() != 64 || file.size() != qint64(entry.second * sizeof(float)) ||
            !file.open(QIODevice::ReadOnly) || !hash.addData(&file) ||
            hash.result().toHex() != expected) {
            std::fprintf(stderr, "FAIL: reference size/digest mismatch: %s\n", entry.first);
            return 1;
        }
    }
    return 0;
}

inline QJsonObject identity(const char* generator, const char* generatorSha) {
    return {{"upstream_revision", "32b14574edc9caba95f03672bfa1eeddd13c3af7"},
            {"generator", QString::fromUtf8(generator)},
            {"generator_sha256", QString::fromUtf8(generatorSha)}, {"seed", 42}};
}
} // namespace instantmesh_reference
