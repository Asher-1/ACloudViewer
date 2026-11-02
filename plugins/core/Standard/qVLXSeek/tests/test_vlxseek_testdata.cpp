// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Cold-start evidence for the qVLXSeek "Use test data" action: a previously
// absent cache must go through the shared repository's download -> digest
// verification -> extraction chain and expose the pinned demo member. Warm
// reuse (already-extracted / verified-zip) is a separate, cheaper check.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QTimer>

#include "ecvTestDataRepository.h"

namespace {

constexpr auto kDataset = ecvTestDataRepository::Dataset::VlxSeek;

QString zipPath() { return ecvTestDataRepository::zipPath(kDataset); }
QString extractPath() { return ecvTestDataRepository::extractPath(kDataset); }

/** Removes ONLY the VlxSeek dataset artifacts (zip + extracted tree). Other
 *  datasets' caches are untouched. */
void makeCold() {
    QFile::remove(zipPath());
    QDir(extractPath()).removeRecursively();
}

}  // namespace

TEST(VLXSeekTestData, MetadataAndPinnedDigest) {
    const ecvTestDataRepository::DatasetInfo info =
            ecvTestDataRepository::getDatasetInfo(kDataset);
    EXPECT_EQ(info.zipFileName, QStringLiteral("vlx_seek_test_images.zip"));
    EXPECT_EQ(info.extractDirName,
              QStringLiteral("vlx_seek_test_images"));
    EXPECT_TRUE(info.downloadUrl.startsWith(QStringLiteral("https://")));
    // The anchor must resolve to the pinned digest (asset_digests.h), i.e.
    // 64 hex characters — an empty digest would mean an unpinned asset.
    EXPECT_EQ(info.anchor.digestHex.size(), 64);
}

TEST(VLXSeekTestData, ColdDownloadExtractAndDemoFill) {
    if (qEnvironmentVariableIsEmpty("https_proxy") &&
            qEnvironmentVariableIsEmpty("HTTPS_PROXY")) {
        GTEST_SKIP()
                << "cold download needs the site proxy; set https_proxy";
    }
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    makeCold();
    ASSERT_FALSE(repo.isDatasetAvailable(kDataset));

    int readyCalls = 0;
    bool ok = false;
    // ensureDatasetReady completes SYNCHRONOUSLY when the verified zip is
    // already cached (the signal fires before any event loop starts), so the
    // wait only spins when an actual download is in flight.
    QEventLoop loop;
    QObject::connect(&repo, &ecvTestDataRepository::datasetReady, &loop,
                     [&](bool success, ecvTestDataRepository::Dataset kind) {
                         if (kind != kDataset) return;
                         ++readyCalls;
                         ok = success;
                         loop.quit();
                     });
    QTimer::singleShot(10 * 60 * 1000, &loop, &QEventLoop::quit);
    repo.ensureDatasetReady(kDataset);
    if (!readyCalls) loop.exec();

    ASSERT_TRUE(ok) << "cold download/extract chain failed";
    // Exactly one datasetReady(ok) emission for the accepted call.
    ASSERT_EQ(readyCalls, 1);

    // The pinned demo member the dialog fills.
    const QString demo = ecvTestDataRepository::findDatasetFile(
            kDataset, QStringLiteral("demo_image.jpg"));
    EXPECT_FALSE(demo.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(demo));
    // The retained zip stays cached for offline re-extraction.
    EXPECT_TRUE(QFileInfo::exists(zipPath()));

    // Warm reuse: a second ensureDatasetReady resolves immediately without
    // another download (still exactly one ready emission per call).
    int warmCalls = 0;
    QEventLoop warmLoop;
    QObject::connect(&repo, &ecvTestDataRepository::datasetReady, &warmLoop,
                     [&](bool success, ecvTestDataRepository::Dataset kind) {
                         if (kind != kDataset) return;
                         ++warmCalls;
                         EXPECT_TRUE(success);
                         warmLoop.quit();
                     });
    repo.ensureDatasetReady(kDataset);
    if (!warmCalls) warmLoop.exec();
    EXPECT_EQ(warmCalls, 1);
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
