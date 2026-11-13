// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QNetworkProxy>
#include <QTemporaryDir>
#include <cstdio>

#include "ecvAssetIntegrity.h"
#include "ecvTestDataRepository.h"

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
    const QString cacheRoot =
            QDir(temporary.path())
                    .filePath(QString::fromUtf8(
                            u8"cache \u6a21\u578b-\U0001f9e0"));
    REQUIRE(QDir().mkpath(cacheRoot));
    REQUIRE(cloudViewer::utility::filesystem::SetEnvVar(
            "CLOUDVIEWER_DATA_ROOT", cacheRoot.toUtf8().toStdString()));
    using Repository = ecvTestDataRepository;
    using Dataset = Repository::Dataset;
    auto& repository = Repository::instance();
    const auto fixture = [](Dataset kind, const QByteArray& bytes) {
        const QString path = Repository::zipPath(kind);
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly) ||
            file.write(bytes) != bytes.size())
            return false;
        file.close();
        return ecvAssetIntegrity::markVerified(
                path, Repository::getDatasetInfo(kind).anchor);
    };
    REQUIRE(fixture(Dataset::Monstree,
                    QByteArray::fromHex("504b0304140000000000d664695d6fc811d914"
                                        "0000001400000020000000646174617365"
                                        "745f6d6f6e73747265652f6d696e69332f6672"
                                        "616d652e6a706773796e74686574696320"
                                        "696d6167652070617468504b01021403140000"
                                        "000000d664695d6fc811d9140000001400"
                                        "00002000000000000000000000008001000000"
                                        "00646174617365745f6d6f6e7374726565"
                                        "2f6d696e69332f6672616d652e6a7067504b05"
                                        "0600000000010001004e00000052000000"
                                        "0000")));
    // Store real PNG contents: Qt 5.12 does not decode PPM bytes through a
    // .png path, unlike newer Qt versions that sniff the contents again.
    REQUIRE(fixture(Dataset::Image2Mesh,
                    QByteArray::fromHex("504b0304140000000000d664495d510fda0245"
                                        "000000450000002d000000696d6167655f"
                                        "746f5f6d6573685f646174612f6578616d706c"
                                        "65735f696d616765732f73616d706c652e"
                                        "706e6789504e470d0a1a0a0000000d49484452"
                                        "00000001000000010802000000907753de"
                                        "0000000c49444154789c63f8cfc00000030101"
                                        "00c9fe92ef0000000049454e44ae426082"
                                        "504b0304140000000000d664495d510fda0245"
                                        "000000450000002f000000696d6167655f"
                                        "746f5f6d6573685f646174612f73616d33645f"
                                        "696d616765732f7363656e652f696d6167"
                                        "652e706e6789504e470d0a1a0a0000000d4948"
                                        "4452000000010000000108020000009077"
                                        "53de0000000c49444154789c63f8cfc0000003"
                                        "010100c9fe92ef0000000049454e44ae42"
                                        "6082504b01021403140000000000d664495d51"
                                        "0fda0245000000450000002d0000000000"
                                        "000000000000800100000000696d6167655f74"
                                        "6f5f6d6573685f646174612f6578616d70"
                                        "6c65735f696d616765732f73616d706c652e70"
                                        "6e67504b01021403140000000000d66449"
                                        "5d510fda0245000000450000002f0000000000"
                                        "000000000000800190000000696d616765"
                                        "5f746f5f6d6573685f646174612f73616d3364"
                                        "5f696d616765732f7363656e652f696d61"
                                        "67652e706e67504b05060000000002000200b8"
                                        "000000220100000000")));
    int completions = 0;
    int failures = 0;
    int extracts = 0;
    QObject::connect(&repository, &Repository::downloadLogMessage, &app,
                     [](const QString& message) {
                         std::fprintf(stderr, "%s\n",
                                      message.toUtf8().constData());
                     });
    QObject::connect(&repository, &Repository::extractionFinished, &app,
                     [&](bool, Dataset) { ++extracts; });
    QObject::connect(
            &repository, &Repository::datasetReady, &app,
            [&](bool ok, Dataset kind) {
                ++completions;
                if (!ok) ++failures;
                std::fprintf(stderr,
                             "dataset %d: ready=%d, completions=%d, "
                             "failures=%d, extracts=%d\n",
                             static_cast<int>(kind), ok, completions, failures,
                             extracts);
                if (ok && kind == Dataset::Monstree && completions == 1)
                    repository.ensureDatasetReady(Dataset::Image2Mesh);
            });
    // A cached ZIP prepares without transport; its callback may start the
    // next dataset immediately, exactly as a multi-stage plugin does.
    repository.ensureDatasetReady(Dataset::Monstree);
    REQUIRE(completions == 2 && failures == 0 && extracts == 2);
    REQUIRE(!Repository::getMonstreeImages(
                     Repository::extractPath(Dataset::Monstree))
                     .isEmpty());
    repository.ensureDatasetReady(Dataset::Monstree);
    repository.ensureDatasetReady(Dataset::Image2Mesh);
    REQUIRE(completions == 4 && extracts == 2);
    // An interrupted extract missing one consumer's subtree is recovered
    // from the cached ZIP, without another download.
    REQUIRE(QDir(QDir(Repository::extractPath(Dataset::Image2Mesh))
                         .filePath("sam3d_images"))
                    .removeRecursively());
    repository.ensureDatasetReady(Dataset::Image2Mesh);
    REQUIRE(completions == 5 && extracts == 3 && failures == 0);
    REQUIRE(!Repository::getSam3dObjectImages(
                     Repository::extractPath(Dataset::Image2Mesh))
                     .isEmpty());
    // Extraction failure settles once and releases the service for retry.
    REQUIRE(QDir(Repository::extractPath(Dataset::Monstree))
                    .removeRecursively());
    REQUIRE(fixture(Dataset::Monstree, "invalid zip fixture"));
    repository.ensureDatasetReady(Dataset::Monstree);
    REQUIRE(completions == 6 && extracts == 4 && failures == 1);
    REQUIRE(fixture(Dataset::Monstree,
                    QByteArray::fromHex("504b0304140000000000d664695d6fc811d914"
                                        "0000001400000020000000646174617365"
                                        "745f6d6f6e73747265652f6d696e69332f6672"
                                        "616d652e6a706773796e74686574696320"
                                        "696d6167652070617468504b01021403140000"
                                        "000000d664695d6fc811d9140000001400"
                                        "00002000000000000000000000008001000000"
                                        "00646174617365745f6d6f6e7374726565"
                                        "2f6d696e69332f6672616d652e6a7067504b05"
                                        "0600000000010001004e00000052000000"
                                        "0000")));
    repository.ensureDatasetReady(Dataset::Monstree);
    REQUIRE(completions == 7 && extracts == 5 && failures == 1);
    REQUIRE(!repository.isDownloadInProgress());
    // Single-file assets materialize at extractPath(), not inside it.
    const auto video = Dataset::LingbotMapDriveVideo;
    REQUIRE(fixture(video, "synthetic verified video"));
    REQUIRE(Repository::findDatasetFile(
                    video, Repository::getDatasetInfo(video).zipFileName)
                    .isEmpty());
    repository.ensureDatasetReady(video);
    REQUIRE(completions == 8 && extracts == 6 && failures == 1);
    REQUIRE(!Repository::findDatasetFile(
                     video, Repository::getDatasetInfo(video).zipFileName)
                     .isEmpty());
    repository.ensureDatasetReady(video);
    REQUIRE(completions == 9 && extracts == 6);
    REQUIRE(QFile::remove(Repository::extractPath(video)));
    repository.ensureDatasetReady(video);
    REQUIRE(completions == 10 && extracts == 7 && failures == 1);
    // Cancel may synchronously start another preparation. Force transport
    // through loopback; no external connection or event loop is needed.
    const QNetworkProxy savedProxy = QNetworkProxy::applicationProxy();
    QNetworkProxy::setApplicationProxy(QNetworkProxy(
            QNetworkProxy::HttpProxy, QStringLiteral("127.0.0.1"), 9));
    const auto retry = QObject::connect(
            &repository, &Repository::datasetReady, &app,
            [&](bool ok, Dataset kind) {
                if (!ok && kind == Dataset::RAM)
                    repository.ensureDatasetReady(Dataset::RelateAnything);
            });
    repository.ensureDatasetReady(Dataset::RAM);
    REQUIRE(repository.isDownloadInProgress());
    repository.cancelDownload();
    REQUIRE(completions == 11 && failures == 2);
    REQUIRE(repository.isDownloadInProgress());
    QObject::disconnect(retry);
    repository.cancelDownload();
    REQUIRE(completions == 12 && failures == 3);
    REQUIRE(!repository.isDownloadInProgress());
    QNetworkProxy::setApplicationProxy(savedProxy);
    std::puts(
            "dataset repository: cached preparation, reentry, repair and retry "
            "passed");
    return 0;
}
