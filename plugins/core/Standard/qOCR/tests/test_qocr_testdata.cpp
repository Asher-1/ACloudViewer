// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Cold-start evidence for the qOCR tables-mode "Try sample data" action: a
// previously absent cache must go through the shared repository's
// download -> digest verification -> extraction chain and expose the pinned
// demo members (one Chinese and one Japanese real-document table page) that
// OCRDialog::recommendedTestImages(true, ...) fills. Warm reuse
// (already-extracted / verified-zip) is a separate, cheaper check.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QTimer>

#include "OCRDialog.h"
#include "ecvTestDataRepository.h"

namespace {

constexpr auto kDataset = ecvTestDataRepository::Dataset::PpocrTables;

QString zipPath() { return ecvTestDataRepository::zipPath(kDataset); }
QString extractPath() { return ecvTestDataRepository::extractPath(kDataset); }

/** Removes ONLY the PpocrTables dataset artifacts (zip + extracted tree).
 *  Other datasets' caches are untouched. */
void makeCold() {
    QFile::remove(zipPath());
    QDir(extractPath()).removeRecursively();
}

/** Runs one ensureDatasetReady chain to settlement (exactly one
 *  datasetReady emission for the accepted call). */
bool ensureReady(int* readyCalls) {
    ecvTestDataRepository& repo = ecvTestDataRepository::instance();
    *readyCalls = 0;
    bool ok = false;
    // ensureDatasetReady completes SYNCHRONOUSLY when the verified zip is
    // already cached (the signal fires before any event loop starts), so the
    // wait only spins when an actual download is in flight.
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

TEST(QOCRTablesTestData, MetadataAndPinnedDigest) {
    const ecvTestDataRepository::DatasetInfo info =
            ecvTestDataRepository::getDatasetInfo(kDataset);
    EXPECT_EQ(info.zipFileName, QStringLiteral("ocr_tables_data.zip"));
    EXPECT_EQ(info.extractDirName, QStringLiteral("ocr_tables_data"));
    EXPECT_EQ(info.downloadUrl,
              QStringLiteral("https://github.com/Asher-1/cloudViewer_downloads/"
                             "releases/download/PP-OCR/ocr_tables_data.zip"));
    // The archive digest must be pinned in source (64 hex characters) — an
    // empty digest would mean an unpinned asset.
    EXPECT_EQ(info.anchor.digestHex.size(), 64);
    // Tables mode maps to this dataset; every other mode keeps the demo
    // bundle.
    EXPECT_EQ(OCRDialog::datasetForMode(true), kDataset);
    EXPECT_EQ(OCRDialog::datasetForMode(false),
              ecvTestDataRepository::Dataset::Ppocr);
    // The deterministic first-fill recommendations are exactly the pinned
    // demo members checked below.
    const QStringList rec = OCRDialog::recommendedTestImages(true, false, true);
    EXPECT_EQ(rec, (QStringList{QStringLiteral("zh_val_0.jpg"),
                                QStringLiteral("ja_val_0.jpg")}));
}

TEST(QOCRTablesTestData, ColdDownloadExtractAndDemoFill) {
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

    // The pinned demo members the dialog's tables mode checks (one Chinese,
    // one Japanese real-document table page).
    for (const QString& demo :
         {QStringLiteral("zh_val_0.jpg"), QStringLiteral("ja_val_0.jpg")}) {
        const QString path =
                ecvTestDataRepository::findDatasetFile(kDataset, demo);
        EXPECT_FALSE(path.isEmpty()) << demo.toUtf8().constData();
        EXPECT_TRUE(QFileInfo::exists(path));
    }
    // The full validation bundle: 100 real-document table pages.
    const QStringList images =
            OCRDialog::testDataImages(extractPath(), true, false, true);
    EXPECT_EQ(images.size(), 100);
    // The recommendations are listed first, in recommendation order.
    const QStringList rec = OCRDialog::recommendedTestImages(true, false, true);
    for (int i = 0; i < rec.size(); ++i)
        EXPECT_EQ(images.value(i),
                  QDir(extractPath()).absoluteFilePath(rec[i]));
    // The retained zip stays cached for offline re-extraction.
    EXPECT_TRUE(QFileInfo::exists(zipPath()));

    // Warm reuse: a second ensureDatasetReady resolves immediately without
    // another download (still exactly one ready emission per call).
    int warmCalls = 0;
    const bool warmOk = ensureReady(&warmCalls);
    EXPECT_TRUE(warmOk);
    EXPECT_EQ(warmCalls, 1);
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QCoreApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
