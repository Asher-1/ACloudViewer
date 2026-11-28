// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Cold-start evidence for the qWeVisOne "Use test data" action: a previously
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

constexpr auto kDataset = ecvTestDataRepository::Dataset::WeVisOne;

QString zipPath() { return ecvTestDataRepository::zipPath(kDataset); }
QString extractPath() { return ecvTestDataRepository::extractPath(kDataset); }

/** Removes ONLY the WeVisOne dataset artifacts (zip + extracted tree).
 *  Other datasets' caches are untouched. */
void makeCold() {
    QFile::remove(zipPath());
    QDir(extractPath()).removeRecursively();
}

}  // namespace

TEST(WeVisOneTestData, MetadataAndPinnedDigest) {
    const ecvTestDataRepository::DatasetInfo info =
            ecvTestDataRepository::getDatasetInfo(kDataset);
    EXPECT_EQ(info.zipFileName, QStringLiteral("wevisone_vl_test_data.zip"));
    EXPECT_EQ(info.extractDirName, QStringLiteral("wevisone_vl_test_data"));
    // Revision-pinned resolver URL: the task-sample manifest.json only
    // exists from the pinned commit onward.
    EXPECT_TRUE(info.downloadUrl.startsWith(QStringLiteral(
            "https://huggingface.co/Asher-1/WeVisOne-VL-GGML/resolve/"
            "ec7bf6528100a17ac97269e1a7d94d35d9e0c792/")));
    // The anchor must resolve to the pinned digest (asset_digests.h), i.e.
    // 64 hex characters — an empty digest would mean an unpinned asset.
    EXPECT_EQ(info.anchor.digestHex.size(), 64);
}

TEST(WeVisOneTestData, ColdDownloadExtractAndDemoFill) {
    // The shared downloader tries huggingface.co first and rotates to its
    // hf-mirror.com fallback when the primary is unreachable (blocked on
    // some networks); either endpoint serves the identical digest-pinned
    // archive. An unavailable network fails this test clearly instead of
    // silently passing.
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

    // The pinned demo member the dialog fills for the default detection
    // task (the multi-object living room) plus the manifest that carries
    // the task-sample queries.
    const QString demo = ecvTestDataRepository::findDatasetFile(
            kDataset, QStringLiteral("livingroom.jpg"));
    EXPECT_FALSE(demo.isEmpty());
    EXPECT_TRUE(QFileInfo::exists(demo));
    EXPECT_FALSE(ecvTestDataRepository::findDatasetFile(
                         kDataset, QStringLiteral("manifest.json"))
                         .isEmpty());
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