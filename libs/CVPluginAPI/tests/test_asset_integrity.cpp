// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <cstdio>

#include "ecvAssetIntegrity.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    const QString path = temporary.filePath("model.gguf");
    const QByteArray bytes = QByteArray("GGUF") + QByteArray(4096, 'x');
    const ecvAssetIntegrity::Anchor pin{
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(bytes, QCryptographicHash::Sha256)
                    .toHex()};
    const auto write = [&path](const QByteArray& payload) {
        QFile file(path);
        return file.open(QIODevice::WriteOnly | QIODevice::Truncate) &&
               file.write(payload) == payload.size();
    };
    REQUIRE(write(bytes));
    REQUIRE(ecvAssetIntegrity::isVerified(
            path, pin, 64, true, ecvAssetIntegrity::OnMiss::DeepVerify));
    REQUIRE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    REQUIRE(ecvAssetIntegrity::isVerified(path, {}, 64, true));

    // A changed downloaded file must stay rejected across repeated GUI
    // presence checks and a failed explicit revalidation.
    REQUIRE(write(bytes + "corrupt"));
    REQUIRE(!ecvAssetIntegrity::isVerified(path, pin, 64, true));
    REQUIRE(!ecvAssetIntegrity::isVerified(path, pin, 64, true));
    REQUIRE(!ecvAssetIntegrity::isVerified(path, {}, 64, true));
    REQUIRE(!ecvAssetIntegrity::isVerified(
            path, pin, 64, true, ecvAssetIntegrity::OnMiss::DeepVerify));
    REQUIRE(!ecvAssetIntegrity::isVerified(path, pin, 64, true));
    REQUIRE(QFileInfo::exists(path + ".cvintegrity"));

    // Restoring the original payload allows an explicit hash to recover it.
    REQUIRE(write(bytes));
    REQUIRE(ecvAssetIntegrity::isVerified(
            path, pin, 64, true, ecvAssetIntegrity::OnMiss::DeepVerify));
    REQUIRE(ecvAssetIntegrity::isVerified(path, pin, 64, true));

    // A source pin change must keep rejecting the old payload and evidence.
    const ecvAssetIntegrity::Anchor otherPin{QCryptographicHash::Sha256,
                                             QByteArray("11").repeated(32)};
    REQUIRE(!ecvAssetIntegrity::isVerified(path, otherPin, 64, true));
    REQUIRE(!ecvAssetIntegrity::isVerified(path, otherPin, 64, true));
    REQUIRE(ecvAssetIntegrity::isVerified(path, pin, 64, true));

    // Preserve legacy ingestion policy for files without prior evidence.
    ecvAssetIntegrity::invalidate(path);
    REQUIRE(ecvAssetIntegrity::isVerified(path, pin, 64, true));
    REQUIRE(!ecvAssetIntegrity::isVerified(
            path, {}, 64, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            bytes.size() + 1));
    REQUIRE(ecvAssetIntegrity::isVerified(
            path, {}, 64, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            bytes.size()));
    std::puts("shared asset integrity cache checks passed");
    return 0;
}
