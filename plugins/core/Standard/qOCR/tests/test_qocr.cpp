// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Workspace, sample-selection and optional real-model contracts for qOCR.
// Pass --require-e2e to enable the cached f32 CPU table/worker cases. Missing
// required assets produce CTest's exit-77 skip after all selected tests finish.

#include <gtest/gtest.h>

#include <QApplication>
#include <QAtomicInt>
#include <QBuffer>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <memory>

#include "OCRDialog.h"
#include "OCRWorker.h"
#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"
#include "ecvTestDataRepository.h"

namespace {

bool requireE2E = false;
bool missingE2EAssets = false;

QString cachedModel(const QString& name) {
    char* dir = aicore_ocr_model_cache_dir();
    if (!dir) return QString();
    const QString path = QDir(QString::fromUtf8(dir)).filePath(name);
    aicore_ocr_free_buffer(dir);
    return QFileInfo(path).isFile() ? path : QString();
}

class QOCRWorkspace : public ::testing::Test {
protected:
    void SetUp() override {
        // Each case starts without images or options persisted by another case
        // or by a GUI session on the developer's machine.
        QSettings settings;
        settings.beginGroup(QStringLiteral("qOCR"));
        settings.remove(QStringLiteral(""));
        settings.endGroup();
        dialog = std::make_unique<OCRDialog>();
        model = dialog->findChild<QComboBox*>(QStringLiteral("qOCRModel"));
        quant = dialog->findChild<QComboBox*>(QStringLiteral("qOCRQuant"));
        device = dialog->findChild<QComboBox*>(QStringLiteral("qOCRDevice"));
        threads = dialog->findChild<QSpinBox*>(QStringLiteral("qOCRThreads"));
        imageList = dialog->findChild<QListWidget*>(
                QStringLiteral("qOCRImageList"));
        testData = dialog->findChild<QPushButton*>(
                QStringLiteral("qOCRUseTestData"));
        download =
                dialog->findChild<QPushButton*>(QStringLiteral("qOCRDownload"));
        run = dialog->findChild<QPushButton*>(QStringLiteral("qOCRRun"));
        cancel = dialog->findChild<QPushButton*>(QStringLiteral("qOCRCancel"));
        addDb = dialog->findChild<QCheckBox*>(QStringLiteral("qOCRAddToDb"));
        results =
                dialog->findChild<QTableWidget*>(QStringLiteral("qOCRResults"));
        textFilter =
                dialog->findChild<QLineEdit*>(QStringLiteral("qOCRTextFilter"));
        copyText =
                dialog->findChild<QPushButton*>(QStringLiteral("qOCRTextCopy"));
        saveText =
                dialog->findChild<QPushButton*>(QStringLiteral("qOCRTextSave"));
        progress = dialog->findChild<QProgressBar*>(
                QStringLiteral("qOCRProgress"));
        ASSERT_NE(model, nullptr);
        ASSERT_NE(quant, nullptr);
        ASSERT_NE(device, nullptr);
        ASSERT_NE(threads, nullptr);
        ASSERT_NE(imageList, nullptr);
        ASSERT_NE(testData, nullptr);
        ASSERT_NE(download, nullptr);
        ASSERT_NE(run, nullptr);
        ASSERT_NE(cancel, nullptr);
        ASSERT_NE(addDb, nullptr);
        ASSERT_NE(results, nullptr);
        ASSERT_NE(textFilter, nullptr);
        ASSERT_NE(copyText, nullptr);
        ASSERT_NE(saveText, nullptr);
        ASSERT_NE(progress, nullptr);
    }

    std::unique_ptr<OCRDialog> dialog;
    QComboBox *model = nullptr, *quant = nullptr, *device = nullptr;
    QSpinBox* threads = nullptr;
    QListWidget* imageList = nullptr;
    QPushButton *testData = nullptr, *download = nullptr, *run = nullptr,
                *cancel = nullptr, *copyText = nullptr, *saveText = nullptr;
    QCheckBox* addDb = nullptr;
    QTableWidget* results = nullptr;
    QLineEdit* textFilter = nullptr;
    QProgressBar* progress = nullptr;
};

TEST_F(QOCRWorkspace, UsesSharedConsoleAndSingleOutputChoice) {
    EXPECT_EQ(dialog->findChild<QPlainTextEdit*>(QStringLiteral("qOCRLog")),
              nullptr);
    EXPECT_EQ(dialog->findChild<QCheckBox*>(QStringLiteral("qOCRShowInWindow")),
              nullptr);
}

TEST_F(QOCRWorkspace, RecognitionCatalogOwnsLanguageAndQuantization) {
    EXPECT_EQ(dialog->findChild<QComboBox*>(QStringLiteral("qOCRLanguage")),
              nullptr);
    ASSERT_GE(model->count(), 2);
    EXPECT_EQ(model->itemData(0).toString(), QStringLiteral("ppocr-v4-ch"));
    EXPECT_EQ(model->itemData(1).toString(), QStringLiteral("ppocr-v4-en"));
    EXPECT_EQ(quant->count(), 3);
}

TEST_F(QOCRWorkspace, ResultsExposeLayoutAndClipboardExport) {
    ASSERT_EQ(results->columnCount(), 6);
    ASSERT_NE(results->horizontalHeaderItem(0), nullptr);
    ASSERT_NE(results->horizontalHeaderItem(5), nullptr);
    EXPECT_EQ(results->horizontalHeaderItem(0)->text(), QStringLiteral("#"));
    EXPECT_EQ(results->horizontalHeaderItem(5)->text(),
              QStringLiteral("Layout"));
    EXPECT_EQ(results->contextMenuPolicy(), Qt::CustomContextMenu);
    EXPECT_FALSE(copyText->isEnabled());
    EXPECT_FALSE(saveText->isEnabled());
}

TEST_F(QOCRWorkspace, RuntimeDefaultsAndActionsRequireStagedInput) {
    ASSERT_GE(device->count(), 1);
    EXPECT_FALSE(device->itemData(0).toString().isEmpty());
    EXPECT_EQ(threads->value(), 0);
    EXPECT_FALSE(run->isEnabled());
    EXPECT_FALSE(cancel->isEnabled());
}

TEST_F(QOCRWorkspace, RecognitionSwitchReportsDetectionAndRecognitionState) {
    model->setCurrentIndex(1);
    auto* status = dialog->findChild<QLabel*>(QStringLiteral("qOCRStatus"));
    ASSERT_NE(status, nullptr);
    EXPECT_TRUE(status->text().contains(QStringLiteral("det:")));
    EXPECT_TRUE(status->text().contains(QStringLiteral("rec:")));
}

TEST_F(QOCRWorkspace, ImageListDeduplicatesAndTracksChecks) {
    const QString own = QDir::temp().filePath(QStringLiteral("qocr-own.png"));
    EXPECT_EQ(dialog->addImages({own, own}, true), 1);
    EXPECT_EQ(dialog->addImages(
                      {QDir::temp().filePath(QStringLiteral("qocr-b.png"))},
                      false),
              1);
    EXPECT_EQ(imageList->count(), 2);
    EXPECT_EQ(dialog->checkedImages(),
              QStringList{QFileInfo(own).absoluteFilePath()});
}

TEST(QOCRSamples, ModesAndFamiliesRecommendOfficialImagesFromOwningBundle) {
    const QString root = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::Ppocr);
    const QString tablesRoot = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::PpocrTables);
    const QStringList families = {QStringLiteral("zh"), QStringLiteral("en"),
                                  QStringLiteral("ja")};
    for (const bool table : {false, true}) {
        for (const bool layout : {false, true}) {
            for (const QString& family : families) {
                SCOPED_TRACE(::testing::Message()
                             << "table=" << table << ", layout=" << layout
                             << ", family=" << family.toStdString());
                const QStringList recommended =
                        OCRDialog::recommendedTestImages(table, layout, family);
                ASSERT_FALSE(recommended.isEmpty());
                const auto expectedKind =
                        (table && family != QStringLiteral("en"))
                                ? ecvTestDataRepository::Dataset::PpocrTables
                                : ecvTestDataRepository::Dataset::Ppocr;
                EXPECT_EQ(OCRDialog::datasetForMode(table, family),
                          expectedKind);
                for (const QString& relative : recommended) {
                    EXPECT_FALSE(OCRDialog::isSyntheticTestImage(relative))
                            << relative.toStdString();
                }
                const QString& modeRoot =
                        expectedKind == ecvTestDataRepository::Dataset::Ppocr
                                ? root
                                : tablesRoot;
                if (!QDir(modeRoot).exists()) continue;
                const QStringList all = OCRDialog::testDataImages(
                        modeRoot, table, layout, family);
                for (int i = 0; i < recommended.size(); ++i) {
                    EXPECT_EQ(all.value(i),
                              QDir(modeRoot).absoluteFilePath(recommended[i]));
                }
            }
        }
    }
}

TEST_F(QOCRWorkspace, TableStageIsOptInAndResolvesSelectedTier) {
    auto* table = dialog->findChild<QCheckBox*>(QStringLiteral("qOCRTable"));
    auto* tabs =
            dialog->findChild<QTabWidget*>(QStringLiteral("qOCRResultTabs"));
    auto* selector =
            dialog->findChild<QComboBox*>(QStringLiteral("qOCRTableSelect"));
    auto* grid =
            dialog->findChild<QTableWidget*>(QStringLiteral("qOCRTableGrid"));
    auto* saveHtml = dialog->findChild<QPushButton*>(
            QStringLiteral("qOCRTableSaveHtml"));
    auto* saveCsv =
            dialog->findChild<QPushButton*>(QStringLiteral("qOCRTableSaveCsv"));
    auto* status = dialog->findChild<QLabel*>(QStringLiteral("qOCRStatus"));
    ASSERT_NE(table, nullptr);
    ASSERT_NE(tabs, nullptr);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(grid, nullptr);
    ASSERT_NE(saveHtml, nullptr);
    ASSERT_NE(saveCsv, nullptr);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(tabs->count(), 2);
    EXPECT_FALSE(table->isChecked());
    EXPECT_FALSE(saveHtml->isEnabled());
    EXPECT_FALSE(saveCsv->isEnabled());
    quant->setCurrentIndex(1);
    table->setChecked(true);
    EXPECT_TRUE(status->text().contains(QStringLiteral("table:")));
    EXPECT_TRUE(dialog->runSettings().tablePath.endsWith(
            QStringLiteral("slanet-plus-table-f16.gguf")));
    table->setChecked(false);
    EXPECT_TRUE(dialog->runSettings().tablePath.isEmpty());
}

TEST_F(QOCRWorkspace, CachedSampleFillRetargetsChecksAndPreservesUserRows) {
    auto* table = dialog->findChild<QCheckBox*>(QStringLiteral("qOCRTable"));
    auto* layout = dialog->findChild<QCheckBox*>(QStringLiteral("qOCRLayout"));
    ASSERT_NE(table, nullptr);
    ASSERT_NE(layout, nullptr);
    const QString root = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::Ppocr);
    const QString tablesRoot = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::PpocrTables);
    if (!QDir(root).exists() || !QDir(tablesRoot).exists()) {
        GTEST_SKIP()
                << "PaddleOCR demo and real-document table caches required";
    }
    const QString own = QDir::temp().filePath(QStringLiteral("qocr-own.png"));
    ASSERT_EQ(dialog->addImages({own, own}, true), 1);
    ASSERT_EQ(dialog->addImages(
                      {QDir::temp().filePath(QStringLiteral("qocr-b.png"))},
                      false),
              1);
    const QStringList zhTableRecs =
            OCRDialog::recommendedTestImages(true, false, QStringLiteral("zh"));
    const int tablesCount = OCRDialog::testDataImages(tablesRoot, true, false,
                                                      QStringLiteral("zh"))
                                    .size();
    const int demoCount =
            OCRDialog::testDataImages(root, false, false, QStringLiteral("zh"))
                    .size();
    table->setChecked(true);
    layout->setChecked(false);
    testData->click();
    QStringList checked = dialog->checkedImages();
    EXPECT_EQ(imageList->count(), tablesCount + 2);
    ASSERT_EQ(checked.size(), 1 + zhTableRecs.size());
    EXPECT_EQ(checked.first(), QFileInfo(own).absoluteFilePath());
    for (const QString& relative : zhTableRecs) {
        EXPECT_TRUE(
                checked.contains(QDir(tablesRoot).absoluteFilePath(relative)))
                << relative.toStdString();
    }
    // A user check on a carried table row must survive recommendation changes.
    QListWidgetItem* userRow = nullptr;
    for (int i = 0; i < imageList->count() && !userRow; ++i) {
        QListWidgetItem* item = imageList->item(i);
        if (item->data(Qt::UserRole)
                    .toString()
                    .endsWith(QStringLiteral("zh_val_30.jpg")))
            userRow = item;
    }
    ASSERT_NE(userRow, nullptr);
    userRow->setCheckState(Qt::Checked);
    table->setChecked(false);
    layout->setChecked(true);
    testData->click();
    checked = dialog->checkedImages();
    const QStringList layoutRecs =
            OCRDialog::recommendedTestImages(false, true, QStringLiteral("zh"));
    EXPECT_EQ(imageList->count(), tablesCount + demoCount + 2);
    EXPECT_EQ(checked.size(), 1 + 1 + layoutRecs.size());
    for (const QString& relative : layoutRecs) {
        EXPECT_TRUE(checked.contains(QDir(root).absoluteFilePath(relative)))
                << relative.toStdString();
    }
    for (const QString& relative : zhTableRecs) {
        EXPECT_FALSE(
                checked.contains(QDir(tablesRoot).absoluteFilePath(relative)))
                << relative.toStdString();
    }
    EXPECT_TRUE(checked.contains(
            QDir(tablesRoot)
                    .absoluteFilePath(QStringLiteral("zh_val_30.jpg"))));
    model->setCurrentIndex(1);
    layout->setChecked(false);
    testData->click();
    checked = dialog->checkedImages();
    const QStringList enRecs = OCRDialog::recommendedTestImages(
            false, false, QStringLiteral("en"));
    EXPECT_EQ(checked.size(), 1 + 1 + enRecs.size());
    for (const QString& relative : enRecs) {
        EXPECT_TRUE(checked.contains(QDir(root).absoluteFilePath(relative)))
                << relative.toStdString();
    }
}

TEST(QOCRTableGrid, PlacesCellsWithColumnAndRowSpans) {
    OCRTableResult table;
    table.rows = 2;
    table.cols = 3;
    OCRTableCell head;
    head.colSpan = 2;
    head.text = QStringLiteral("head");
    OCRTableCell tall;
    tall.col = 2;
    tall.rowSpan = 2;
    tall.text = QStringLiteral("tall");
    OCRTableCell a, b;
    a.row = b.row = 1;
    b.col = 1;
    a.text = QStringLiteral("a");
    b.text = QStringLiteral("b");
    table.cells = {head, tall, a, b};
    QTableWidget grid;
    OCRDialog::fillTableGrid(&grid, table);
    EXPECT_EQ(grid.rowCount(), 2);
    EXPECT_EQ(grid.columnCount(), 3);
    EXPECT_EQ(grid.columnSpan(0, 0), 2);
    EXPECT_EQ(grid.rowSpan(0, 2), 2);
    ASSERT_NE(grid.item(1, 1), nullptr);
    EXPECT_EQ(grid.item(1, 1)->text(), QStringLiteral("b"));
    ASSERT_NE(grid.item(0, 2), nullptr);
    EXPECT_EQ(grid.item(0, 2)->text(), QStringLiteral("tall"));
}

// JPEG copy tagged EXIF Orientation 6 (rotate 90 CW): an APP1 segment
// (big-endian TIFF, one SHORT entry) spliced in after SOI.
bool writeRotatedExifJpeg(const QString& source, const QString& target) {
    QByteArray jpeg;
    QBuffer buffer(&jpeg);
    if (!buffer.open(QIODevice::WriteOnly) ||
        !QImage(source).save(&buffer, "JPG", 95) ||
        !jpeg.startsWith("\xFF\xD8"))
        return false;
    static const char kApp1[] =
            "\xFF\xE1\x00\x22"                  // APP1, length 34
            "Exif\x00\x00"                      // EXIF header
            "MM\x00\x2A\x00\x00\x00\x08"        // TIFF, IFD0 at 8
            "\x00\x01"                          // one entry
            "\x01\x12\x00\x03\x00\x00\x00\x01"  // Orientation, SHORT, 1
            "\x00\x06\x00\x00"                  // value 6
            "\x00\x00\x00\x00";                 // no next IFD
    jpeg.insert(2, QByteArray(kApp1, sizeof(kApp1) - 1));
    QFile file(target);
    return file.open(QIODevice::WriteOnly) && file.write(jpeg) == jpeg.size();
}

class QOCRE2E : public ::testing::Test {
protected:
    void SetUp() override {
        if (!requireE2E) GTEST_SKIP() << "enable with --require-e2e";
        det = cachedModel(QStringLiteral("ppocr-det-ch-f32.gguf"));
        rec = cachedModel(QStringLiteral("ppocr-rec-ch-f32.gguf"));
        tableModel = cachedModel(QStringLiteral("slanet-plus-table-f32.gguf"));
        image = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::Ppocr,
                QStringLiteral("table.jpg"));
        if (det.isEmpty() || rec.isEmpty() || tableModel.isEmpty() ||
            image.isEmpty()) {
            missingE2EAssets = true;
            GTEST_SKIP() << "f32 models/PaddleOCR table test data not cached";
        }
        settings.detPath = det;
        settings.recPath = rec;
        settings.tablePath = tableModel;
        settings.imagePath = image;
    }

    bool createContexts() {
        const QByteArray detBytes = QFile::encodeName(det);
        const QByteArray recBytes = QFile::encodeName(rec);
        const QByteArray tableBytes = QFile::encodeName(tableModel);
        aicore_ocr_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = AICORE_OCR_ABI_VERSION;
        options.det_model_path = detBytes.constData();
        options.rec_model_path = recBytes.constData();
        options.device = "cpu";
        options.det_limit_type = 1;
        options.det_max_side = 960;
        aicore_table_options tableOptions{};
        tableOptions.struct_size = sizeof(tableOptions);
        tableOptions.abi_version = AICORE_TABLE_ABI_VERSION;
        tableOptions.model_path = tableBytes.constData();
        tableOptions.device = "cpu";
        ocr.reset(aicore_ocr_context_create(&options));
        table.reset(aicore_table_context_create(&tableOptions));
        return ocr && table;
    }

    bool setBatchImages() {
        const QString crop = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::Ppocr,
                QStringLiteral("border_bottom_18508_NC47FH2P0Z0HWA248FL3.jpg"));
        if (crop.isEmpty()) {
            missingE2EAssets = true;
            return false;
        }
        settings.device = QStringLiteral("cpu");
        settings.imagePaths = QStringList{image, crop};
        return true;
    }

    QString det, rec, tableModel, image;
    OCRWorker::Settings settings;
    std::unique_ptr<aicore_ocr_ctx, decltype(&aicore_ocr_context_destroy)> ocr{
            nullptr, aicore_ocr_context_destroy};
    std::unique_ptr<aicore_table_ctx, decltype(&aicore_table_context_destroy)>
            table{nullptr, aicore_table_context_destroy};
};

TEST_F(QOCRE2E, ReconstructsReferenceTableAndRendersSourceAndOfficialPreviews) {
    ASSERT_TRUE(createContexts()) << "OCR/table context creation failed";
    OCRRunResult result;
    QString error;
    ASSERT_TRUE(OCRWorker::recognizeImage(ocr.get(), table.get(), settings,
                                          nullptr, &result, &error))
            << error.toStdString();
    ASSERT_EQ(result.tables.size(), 1);
    const OCRTableResult& reconstructed = result.tables[0];
    EXPECT_EQ(reconstructed.rows, 4);
    EXPECT_EQ(reconstructed.cols, 4);
    ASSERT_EQ(reconstructed.cells.size(), 13);
    EXPECT_EQ(reconstructed.cells[0].colSpan, 4);
    EXPECT_EQ(reconstructed.cells[0].text, QStringLiteral("CRuncover"));
    EXPECT_TRUE(reconstructed.csv.startsWith("CRuncover,,,\r\n"));
    EXPECT_FALSE(result.tableModel.isEmpty());
    EXPECT_GT(result.tableMs, 0.0);
    // DB overlays preserve source geometry; official exports append a
    // 600-pixel text panel using PaddleOCR's draw_ocr layout.
    const QImage overlay = OCRRender::overlayVisualization(result);
    const QImage official = OCRRender::officialVisualization(result);
    EXPECT_EQ(overlay.size(), result.sourceImage.size());
    EXPECT_FALSE(official.isNull());
    EXPECT_GT(official.width(), 600);
    EXPECT_LE(official.height(), 600);
    EXPECT_NE(official.size(), result.sourceImage.size());
}

TEST_F(QOCRE2E, AppliesExifOrientationBeforeRecognition) {
    ASSERT_TRUE(createContexts()) << "OCR/table context creation failed";
    QTemporaryDir temporary;
    ASSERT_TRUE(temporary.isValid());
    const QString rotated =
            QDir(temporary.path()).filePath(QStringLiteral("rot90.jpg"));
    ASSERT_TRUE(writeRotatedExifJpeg(image, rotated));
    OCRRunResult source;
    QString error;
    ASSERT_TRUE(OCRWorker::recognizeImage(ocr.get(), nullptr, settings, nullptr,
                                          &source, &error))
            << error.toStdString();
    settings.imagePath = rotated;
    OCRRunResult upright;
    ASSERT_TRUE(OCRWorker::recognizeImage(ocr.get(), nullptr, settings, nullptr,
                                          &upright, &error))
            << error.toStdString();
    EXPECT_EQ(upright.sourceImage.size(),
              source.sourceImage.size().transposed());
    EXPECT_FALSE(upright.boxes.isEmpty());
}

TEST_F(QOCRE2E, WorkerReturnsOneTablePerImageInInputOrder) {
    if (!setBatchImages()) GTEST_SKIP() << "worker crop test image not cached";
    OCRWorker worker(settings);
    QStringList order;
    QVector<int> tables;
    QAtomicInt finished(-1);
    QObject::connect(
            &worker, &OCRWorker::resultReady, &worker,
            [&](const OCRRunResult& result) {
                order << result.imagePath;
                tables << result.tables.size();
            },
            Qt::DirectConnection);
    QObject::connect(
            &worker, &OCRWorker::taskFinished, &worker,
            [&](bool ok) { finished = ok ? 1 : 0; }, Qt::DirectConnection);
    worker.start();
    ASSERT_TRUE(worker.wait());
    EXPECT_EQ(finished.loadAcquire(), 1);
    EXPECT_EQ(order, settings.imagePaths);
    EXPECT_EQ(tables, (QVector<int>{1, 1}));
}

TEST_F(QOCRE2E, CancellationBeforeStartReportsUnsuccessfulCompletionOnce) {
    if (!setBatchImages()) GTEST_SKIP() << "worker crop test image not cached";
    QAtomicInt finished(-1);
    QAtomicInt completions(0);
    OCRWorker worker(settings);
    QObject::connect(
            &worker, &OCRWorker::taskFinished, &worker,
            [&](bool ok) {
                finished = ok ? 1 : 0;
                completions.fetchAndAddOrdered(1);
            },
            Qt::DirectConnection);
    worker.requestTaskCancel();
    worker.start();
    ASSERT_TRUE(worker.wait());
    EXPECT_EQ(finished.loadAcquire(), 0);
    EXPECT_EQ(completions.loadAcquire(), 1);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (QString::fromLocal8Bit(argv[i]) ==
            QStringLiteral("--require-e2e")) {
            requireE2E = true;
            for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
            --argc;
            --i;
        }
    }
    argv[argc] = nullptr;
    ::testing::InitGoogleTest(&argc, argv);
    QTemporaryDir settingsDirectory;
    if (!settingsDirectory.isValid()) return 1;
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("ACloudViewerTests"));
    QCoreApplication::setApplicationName(QStringLiteral("qOCR"));
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       settingsDirectory.path());
    QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope,
                       settingsDirectory.path());
    const int result = RUN_ALL_TESTS();
    if (result != 0) return result;
    return requireE2E && missingE2EAssets ? 77 : 0;
}
