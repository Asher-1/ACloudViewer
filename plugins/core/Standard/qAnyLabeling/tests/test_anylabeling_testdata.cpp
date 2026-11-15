// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Cold-start evidence for the qAnyLabeling "Use Test Data" action: a
// previously absent cache must go through the shared repository's download
// -> digest verification -> extraction chain and expose the pinned demo
// members. Warm reuse (already-extracted / verified-zip) is a separate,
// cheaper check. The archive itself is assembled by the plugin's
// scripts/build_test_data.py from the official X-AnyLabeling-GGML demo
// assets; the manifest row (anylabeling-test-data) pins the archive and
// member digests.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTimer>
#include <QtGlobal>

#include "ecvTestDataRepository.h"

#ifndef QANY_VALIDATION_MANIFEST
#define QANY_VALIDATION_MANIFEST ""
#endif

namespace {

constexpr auto kDataset = ecvTestDataRepository::Dataset::AnyLabeling;

QString zipPath() { return ecvTestDataRepository::zipPath(kDataset); }
QString extractPath() { return ecvTestDataRepository::extractPath(kDataset); }

/** Removes ONLY the AnyLabeling dataset artifacts (zip + extracted tree).
 *  Other datasets' caches are untouched. */
void makeCold() {
    QFile::remove(zipPath());
    QDir(extractPath()).removeRecursively();
}

/** The manifest row for this archive: {} when unavailable. */
QJsonObject manifestEntry() {
    const QString path = QStringLiteral(QANY_VALIDATION_MANIFEST);
    if (path.isEmpty()) return {};
    QFile manifest(path);
    if (!manifest.open(QIODevice::ReadOnly)) return {};
    const QJsonArray assets = QJsonDocument::fromJson(manifest.readAll())
                                      .object()
                                      .value(QStringLiteral("input_assets"))
                                      .toArray();
    for (const QJsonValue& value : assets) {
        const QJsonObject entry = value.toObject();
        if (entry.value(QStringLiteral("id")).toString() ==
            QStringLiteral("anylabeling-test-data"))
            return entry;
    }
    return {};
}

/** ensureDatasetReady resolves synchronously when the verified zip is
 *  already cached (the signal fires before any event loop starts), so the
 *  wait only spins when an actual download is in flight. */
bool ensureReady(int* readyCalls) {
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    *readyCalls = 0;
    bool ok = false;
    QEventLoop loop;
    QObject::connect(&repo, &ecvTestDataRepository::datasetReady, &loop,
                     [&](bool success, ecvTestDataRepository::Dataset kind) {
                         if (kind != kDataset) return;
                         ++(*readyCalls);
                         ok = success;
                         loop.quit();
                     });
    QTimer::singleShot(15 * 60 * 1000, &loop, &QEventLoop::quit);
    repo.ensureDatasetReady(kDataset);
    if (!*readyCalls) loop.exec();
    return ok;
}

}  // namespace

TEST(AnyLabelingTestData, MetadataAndPinnedDigest) {
    const ecvTestDataRepository::DatasetInfo info =
            ecvTestDataRepository::getDatasetInfo(kDataset);
    EXPECT_EQ(info.zipFileName, QStringLiteral("anylabeling_test_data.zip"));
    EXPECT_EQ(info.extractDirName, QStringLiteral("anylabeling_test_data"));
    EXPECT_TRUE(info.downloadUrl.startsWith(QStringLiteral("https://")));
    // The anchor must resolve to the pinned digest, i.e. 64 hex characters
    // — an empty digest would mean an unpinned asset.
    EXPECT_EQ(info.anchor.digestHex.size(), 64);

    // The C++ pin and the validation-manifest pin must never drift.
    const QJsonObject entry = manifestEntry();
    ASSERT_FALSE(entry.isEmpty())
            << "validation_manifest.json lacks the anylabeling-test-data row";
    EXPECT_EQ(entry.value(QStringLiteral("sha256")).toString(),
              info.anchor.digestHex);
    EXPECT_EQ(entry.value(QStringLiteral("size_bytes")).toInt(), 8673063);
    EXPECT_TRUE(entry.value(QStringLiteral("archive")).toBool());
    // Plugin UI demo data: no AICore probe consumes it, so the manifest
    // tasks field (gate download scope) must stay empty.
    EXPECT_EQ(entry.value(QStringLiteral("tasks")).toArray().size(), 0);
    EXPECT_FALSE(
            entry.value(QStringLiteral("extracted_files")).toArray().isEmpty());
}

TEST(AnyLabelingTestData, WarmCachePinnedMembersAndReuse) {
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    if (!QFileInfo::exists(zipPath()) && !repo.isDatasetAvailable(kDataset)) {
        GTEST_SKIP() << "no cached archive on this runner; the cold-download "
                        "test owns the transport chain";
    }
    int readyCalls = 0;
    ASSERT_TRUE(ensureReady(&readyCalls))
            << "warm cache failed to verify/extract";
    EXPECT_EQ(readyCalls, 1);
    ASSERT_TRUE(repo.isDatasetAvailable(kDataset));

    // The deterministic first-fill demo pair (upstream canonical rectangle
    // demo), the guide, and every pinned representative member of the
    // 14-mode collection.
    const QStringList members = {QStringLiteral("README.md"),
                                 QStringLiteral("01_rectangle_demo.jpg"),
                                 QStringLiteral("01_rectangle_demo.json"),
                                 QStringLiteral("02_polygon_cat_dog.json"),
                                 QStringLiteral("03_rotation_harbor.json"),
                                 QStringLiteral("04_line_lane.json"),
                                 QStringLiteral("05_point_pose.json"),
                                 QStringLiteral("13_mot_group_bangkok.json"),
                                 QStringLiteral("14_cuboid_street.json"),
                                 QStringLiteral("15_kie_linking.jpg"),
                                 QStringLiteral("15_kie_linking.json"),
                                 QStringLiteral("16_circle_pandas.json"),
                                 QStringLiteral("17_quadrilateral_car.json"),
                                 QStringLiteral("od.json"),
                                 QStringLiteral("ppocr_kie.json")};
    for (const QString& member : members) {
        const QString path =
                ecvTestDataRepository::findDatasetFile(kDataset, member);
        EXPECT_FALSE(path.isEmpty()) << member.toUtf8().constData();
        EXPECT_TRUE(QFileInfo::exists(path)) << member.toUtf8().constData();
    }
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

TEST(AnyLabelingTestData, ColdDownloadExtractAndDemoFill) {
    if (qEnvironmentVariableIsEmpty("https_proxy") &&
        qEnvironmentVariableIsEmpty("HTTPS_PROXY")) {
        GTEST_SKIP() << "cold download needs the site proxy; set https_proxy";
    }
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    makeCold();
    ASSERT_FALSE(repo.isDatasetAvailable(kDataset));

    int readyCalls = 0;
    const bool ok = ensureReady(&readyCalls);
    ASSERT_TRUE(ok) << "cold download/extract chain failed";
    // Exactly one datasetReady(ok) emission for the accepted call.
    ASSERT_EQ(readyCalls, 1);

    // The first-fill demo pair the action opens (image_modes/ prefix).
    for (const QString& demo : {QStringLiteral("01_rectangle_demo.jpg"),
                                QStringLiteral("01_rectangle_demo.json")}) {
        const QString path =
                ecvTestDataRepository::findDatasetFile(kDataset, demo);
        EXPECT_FALSE(path.isEmpty()) << demo.toUtf8().constData();
        EXPECT_TRUE(QFileInfo::exists(path));
    }
    // The retained zip stays cached for offline re-extraction.
    EXPECT_TRUE(QFileInfo::exists(zipPath()));
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
