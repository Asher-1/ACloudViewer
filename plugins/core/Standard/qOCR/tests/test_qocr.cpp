// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Focused qOCR workspace contract test: bilingual model selection, the
// device/threads runtime row, the shared progress section, the DB-export
// switch, the unified download entry, the checkable multi-image input list
// (one-click official test data with the mode's recommended images checked,
// user images preserved), the shared-console logging contract (no private
// log widget), the table view (span-aware grid, HTML/CSV export), and - when
// the f32 models and the PaddleOCR test data are cached - real headless
// table reconstruction through recognizeImage and a two-image OCRWorker run
// (one model load, one result per image).

#include <QApplication>
#include <QAtomicInt>
#include <QBuffer>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLabel>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <cstdio>

#include "OCRDialog.h"
#include "OCRWorker.h"
#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"

namespace {

QString cachedModel(const QString& name) {
    char* dir = aicore_ocr_model_cache_dir();
    if (!dir) return QString();
    const QString path = QDir(QString::fromUtf8(dir)).filePath(name);
    aicore_ocr_free_buffer(dir);
    return QFileInfo(path).isFile() ? path : QString();
}

// 2 x 3 grid: a colspan-2 header and a rowspan-2 cell (the span layout the
// AICore post-process emits for a colspan + rowspan structure).
int checkTableGrid() {
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
    if (grid.rowCount() != 2 || grid.columnCount() != 3 ||
        grid.columnSpan(0, 0) != 2 || grid.rowSpan(0, 2) != 2 ||
        !grid.item(1, 1) || grid.item(1, 1)->text() != QStringLiteral("b") ||
        !grid.item(0, 2) || grid.item(0, 2)->text() != QStringLiteral("tall")) {
        std::fprintf(stderr, "qOCR table grid must place cells with spans\n");
        return 1;
    }
    return 0;
}

// JPEG copy of \p source tagged EXIF Orientation 6 (rotate 90 CW): an APP1
// segment (big-endian TIFF, one SHORT entry) spliced in after SOI.
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

// Real models: PaddleOCR table.jpg through det/rec + SLANet_plus on CPU,
// whole image as one table (the AICore e2e anchor: 4 x 4, 13 cells,
// colspan-4 header "CRuncover").
int checkHeadlessTable() {
    const QString det = cachedModel(QStringLiteral("ppocr-det-ch-f32.gguf"));
    const QString rec = cachedModel(QStringLiteral("ppocr-rec-ch-f32.gguf"));
    const QString tableModel =
            cachedModel(QStringLiteral("slanet-plus-table-f32.gguf"));
    const QString image = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::Ppocr, QStringLiteral("table.jpg"));
    if (det.isEmpty() || rec.isEmpty() || tableModel.isEmpty() ||
        image.isEmpty()) {
        std::printf(
                "qOCR headless table: models/test data not cached, "
                "skipped\n");
        return 77;
    }
    const QByteArray detBytes = QFile::encodeName(det);
    const QByteArray recBytes = QFile::encodeName(rec);
    const QByteArray tableBytes = QFile::encodeName(tableModel);
    aicore_ocr_options o{};
    o.struct_size = sizeof(o);
    o.abi_version = AICORE_OCR_ABI_VERSION;
    o.det_model_path = detBytes.constData();
    o.rec_model_path = recBytes.constData();
    o.device = "cpu";
    o.det_limit_type = 1;
    o.det_max_side = 960;
    aicore_table_options t{};
    t.struct_size = sizeof(t);
    t.abi_version = AICORE_TABLE_ABI_VERSION;
    t.model_path = tableBytes.constData();
    t.device = "cpu";
    aicore_ocr_ctx* ocr = aicore_ocr_context_create(&o);
    aicore_table_ctx* tableCtx = aicore_table_context_create(&t);
    int failures = 0;
    if (!ocr || !tableCtx) {
        std::fprintf(stderr, "qOCR headless table: context create failed\n");
        failures = 1;
    } else {
        OCRWorker::Settings settings;
        settings.detPath = det;
        settings.recPath = rec;
        settings.tablePath = tableModel;
        settings.imagePath = image;
        OCRRunResult result;
        QString error;
        if (!OCRWorker::recognizeImage(ocr, tableCtx, settings, nullptr,
                                       &result, &error)) {
            std::fprintf(stderr, "qOCR headless table: %s\n",
                         error.toUtf8().constData());
            failures = 1;
        } else if (result.tables.size() != 1 || result.tables[0].rows != 4 ||
                   result.tables[0].cols != 4 ||
                   result.tables[0].cells.size() != 13 ||
                   result.tables[0].cells[0].colSpan != 4 ||
                   result.tables[0].cells[0].text !=
                           QStringLiteral("CRuncover") ||
                   !result.tables[0].csv.startsWith("CRuncover,,,\r\n") ||
                   result.tableModel.isEmpty() || result.tableMs <= 0.0) {
            std::fprintf(stderr,
                         "qOCR headless table: unexpected reconstruction\n");
            failures = 1;
        } else {
            // The DB overlay preserves source geometry; the exported preview
            // follows PaddleOCR draw_ocr and appends a 600-px text panel.
            const QImage overlay = OCRRender::overlayVisualization(result);
            const QImage official = OCRRender::officialVisualization(result);
            if (overlay.size() != result.sourceImage.size() ||
                official.isNull() || official.width() <= 600 ||
                official.height() > 600 ||
                official.size() == result.sourceImage.size())
                failures = 1;
            std::printf("qOCR headless table: 4 x 4, 13 cells, %.0f ms\n",
                        result.tableMs);
        }
        // EXIF orientation: OCR runs on the upright image ecvImage shows,
        // so box coordinates match the DB image.
        const QString rotated =
                QDir::temp().filePath(QStringLiteral("qocr-exif-rot90.jpg"));
        OCRWorker::Settings exif = settings;
        exif.imagePath = rotated;
        OCRRunResult upright;
        if (!failures &&
            (!writeRotatedExifJpeg(image, rotated) ||
             !OCRWorker::recognizeImage(ocr, nullptr, exif, nullptr, &upright,
                                        &error) ||
             upright.sourceImage.size() !=
                     result.sourceImage.size().transposed() ||
             upright.boxes.isEmpty())) {
            std::fprintf(stderr,
                         "qOCR must apply EXIF orientation before OCR\n");
            failures = 1;
        }
        QFile::remove(rotated);
    }
    if (tableCtx) aicore_table_context_destroy(tableCtx);
    if (ocr) aicore_ocr_context_destroy(ocr);
    if (failures) return failures;

    // The dialog's batch path: one OCRWorker, two images, one model load,
    // one result per image in list order.
    const QString crop = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::Ppocr,
            QStringLiteral("border_bottom_18508_NC47FH2P0Z0HWA248FL3.jpg"));
    if (crop.isEmpty()) {
        std::printf("qOCR worker batch: crop test image not cached, skipped\n");
        return 77;
    }
    OCRWorker::Settings batch;
    batch.detPath = det;
    batch.recPath = rec;
    batch.tablePath = tableModel;
    batch.device = QStringLiteral("cpu");
    batch.imagePaths = QStringList{image, crop};
    OCRWorker worker(batch);
    QStringList order;
    QVector<int> tables;
    QAtomicInt finished(-1);
    QObject::connect(
            &worker, &OCRWorker::resultReady, &worker,
            [&](const OCRRunResult& r) {
                order << r.imagePath;
                tables << r.tables.size();
            },
            Qt::DirectConnection);
    QObject::connect(
            &worker, &OCRWorker::taskFinished, &worker,
            [&](bool ok) { finished = ok ? 1 : 0; }, Qt::DirectConnection);
    worker.start();
    worker.wait();
    if (finished.loadAcquire() != 1 || order != batch.imagePaths ||
        tables != QVector<int>{1, 1}) {
        std::fprintf(stderr,
                     "qOCR worker batch: expected 2 ordered results with one "
                     "table each (finished=%d, results=%d)\n",
                     finished.loadAcquire(), static_cast<int>(order.size()));
        return 1;
    }
    std::printf("qOCR worker batch: 2 images, 2 tables, one model load\n");

    // Cancellation requested before start still loads and tears down contexts
    // on the worker thread, then reports a non-successful completion once.
    QAtomicInt cancelledFinished(-1);
    OCRWorker cancelledWorker(batch);
    QObject::connect(
            &cancelledWorker, &OCRWorker::taskFinished, &cancelledWorker,
            [&](bool ok) { cancelledFinished = ok ? 1 : 0; },
            Qt::DirectConnection);
    cancelledWorker.requestTaskCancel();
    cancelledWorker.start();
    cancelledWorker.wait();
    if (cancelledFinished.loadAcquire() != 0) {
        std::fprintf(stderr,
                     "qOCR worker cancellation must finish exactly once as "
                     "unsuccessful\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    // Isolate from the developer machine's persisted qOCR state (a staged
    // image from a GUI session would enable Run before the test stages it).
    {
        QSettings settings;
        settings.beginGroup(QStringLiteral("qOCR"));
        settings.remove(QStringLiteral(""));
        settings.endGroup();
    }

    OCRDialog dialog;
    auto* language =
            dialog.findChild<QComboBox*>(QStringLiteral("qOCRLanguage"));
    auto* quant = dialog.findChild<QComboBox*>(QStringLiteral("qOCRQuant"));
    auto* device = dialog.findChild<QComboBox*>(QStringLiteral("qOCRDevice"));
    auto* threads = dialog.findChild<QSpinBox*>(QStringLiteral("qOCRThreads"));
    auto* imageList =
            dialog.findChild<QListWidget*>(QStringLiteral("qOCRImageList"));
    auto* testData =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRUseTestData"));
    auto* download =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRDownload"));
    auto* run = dialog.findChild<QPushButton*>(QStringLiteral("qOCRRun"));
    auto* cancel = dialog.findChild<QPushButton*>(QStringLiteral("qOCRCancel"));
    auto* addDb = dialog.findChild<QCheckBox*>(QStringLiteral("qOCRAddToDb"));
    auto* results =
            dialog.findChild<QTableWidget*>(QStringLiteral("qOCRResults"));
    auto* progress =
            dialog.findChild<QProgressBar*>(QStringLiteral("qOCRProgress"));
    if (!language || !quant || !device || !threads || !imageList || !testData ||
        !download || !run || !cancel || !addDb || !results || !progress) {
        std::fprintf(stderr, "qOCR workspace is missing a core widget\n");
        return 1;
    }

    // Shared-console logging contract: the private log widget is gone; all
    // messages route through aicore_inference_log -> the application console.
    if (dialog.findChild<QPlainTextEdit*>(QStringLiteral("qOCRLog"))) {
        std::fprintf(stderr,
                     "qOCR must not embed a private log widget (use the "
                     "shared console)\n");
        return 1;
    }

    // Bilingual catalog: Chinese and English rec models over the shared
    // script-agnostic detector.
    if (language->count() != 2 ||
        language->itemData(0).toString() != QStringLiteral("ch") ||
        language->itemData(1).toString() != QStringLiteral("en")) {
        std::fprintf(stderr, "qOCR language selection must offer ch and en\n");
        return 1;
    }
    if (quant->count() != 3) {
        std::fprintf(stderr, "qOCR must expose the F32/F16/Q8_0 tiers\n");
        return 1;
    }
    if (results->columnCount() != 6 ||
        results->horizontalHeaderItem(0)->text() != QStringLiteral("#") ||
        results->horizontalHeaderItem(5)->text() != QStringLiteral("Layout")) {
        std::fprintf(
                stderr,
                "qOCR results table must expose index and layout columns\n");
        return 1;
    }
    if (dialog.findChild<QCheckBox*>(QStringLiteral("qOCRShowInWindow"))) {
        std::fprintf(stderr,
                     "qOCR must expose one DB/output choice, not a duplicate "
                     "render-window checkbox\n");
        return 1;
    }

    // Device selection follows the shared runtime row: at least the
    // synthetic "auto" entry, plus a threads spinner defaulting to auto.
    if (device->count() < 1 || device->itemData(0).toString().isEmpty()) {
        std::fprintf(stderr, "qOCR device combo must offer the auto entry\n");
        return 1;
    }
    if (threads->value() != 0) {
        std::fprintf(stderr, "qOCR threads must default to 0 (= auto)\n");
        return 1;
    }

    // The run starts disabled until models and an image are staged; the
    // cancel button mirrors the running state.
    if (run->isEnabled()) {
        std::fprintf(stderr,
                     "qOCR Run must start disabled until models and an "
                     "image are staged\n");
        return 1;
    }
    if (cancel->isEnabled()) {
        std::fprintf(stderr, "qOCR Cancel must start disabled\n");
        return 1;
    }

    // Switching the language retargets the rec model and keeps the state
    // line informative (the det file is shared and language-agnostic).
    language->setCurrentIndex(1);
    auto* status = dialog.findChild<QLabel*>(QStringLiteral("qOCRStatus"));
    if (!status || !status->text().contains(QStringLiteral("det:")) ||
        !status->text().contains(QStringLiteral("rec:"))) {
        std::fprintf(stderr, "qOCR status line missing model state\n");
        return 1;
    }

    // The input is a checkable list: duplicates are skipped and only checked
    // rows run.
    const QString own = QDir::temp().filePath(QStringLiteral("qocr-own.png"));
    if (dialog.addImages({own, own}, true) != 1 ||
        dialog.addImages({QDir::temp().filePath(QStringLiteral("qocr-b.png"))},
                         false) != 1 ||
        imageList->count() != 2 ||
        dialog.checkedImages() !=
                QStringList{QFileInfo(own).absoluteFilePath()}) {
        std::fprintf(stderr, "qOCR image list must dedupe and track checks\n");
        return 1;
    }
    // Every mode recommends official images, and the recommendations of
    // every mode exist in the published bundle once it is extracted.
    const QString root = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::Ppocr);
    for (const bool t : {false, true}) {
        for (const bool l : {false, true}) {
            for (const bool c : {false, true}) {
                const QStringList rec =
                        OCRDialog::recommendedTestImages(t, l, c);
                if (rec.isEmpty()) {
                    std::fprintf(stderr, "qOCR mode without test images\n");
                    return 1;
                }
                for (const QString& rel : rec) {
                    if (OCRDialog::isSyntheticTestImage(rel)) {
                        std::fprintf(stderr,
                                     "qOCR recommends a synthetic image: %s\n",
                                     rel.toUtf8().constData());
                        return 1;
                    }
                }
                if (!QDir(root).exists()) continue;
                const QStringList all =
                        OCRDialog::testDataImages(root, t, l, c);
                for (int i = 0; i < rec.size(); ++i) {
                    if (all.value(i) != QDir(root).absoluteFilePath(rec[i])) {
                        std::fprintf(stderr,
                                     "qOCR recommended test image missing or "
                                     "not listed first: %s\n",
                                     rec[i].toUtf8().constData());
                        return 1;
                    }
                }
            }
        }
    }

    // Table mode: an opt-in stage with its own status entry and a Tables
    // tab (selector, span grid, HTML/CSV export) disabled until a result.
    auto* table = dialog.findChild<QCheckBox*>(QStringLiteral("qOCRTable"));
    auto* tabs =
            dialog.findChild<QTabWidget*>(QStringLiteral("qOCRResultTabs"));
    auto* tableSelect =
            dialog.findChild<QComboBox*>(QStringLiteral("qOCRTableSelect"));
    auto* grid =
            dialog.findChild<QTableWidget*>(QStringLiteral("qOCRTableGrid"));
    auto* saveHtml =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRTableSaveHtml"));
    auto* saveCsv =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRTableSaveCsv"));
    if (!table || !tabs || !tableSelect || !grid || !saveHtml || !saveCsv ||
        tabs->count() != 2 || table->isChecked() || saveHtml->isEnabled() ||
        saveCsv->isEnabled()) {
        std::fprintf(stderr, "qOCR table view contract violated\n");
        return 1;
    }
    quant->setCurrentIndex(1);
    table->setChecked(true);
    if (!status->text().contains(QStringLiteral("table:")) ||
        !dialog.runSettings().tablePath.endsWith(
                QStringLiteral("slanet-plus-table-f16.gguf"))) {
        std::fprintf(stderr,
                     "qOCR table mode must resolve the catalog pack of the "
                     "selected tier\n");
        return 1;
    }
    table->setChecked(false);
    if (!dialog.runSettings().tablePath.isEmpty()) {
        std::fprintf(stderr, "qOCR table stage must be opt-in\n");
        return 1;
    }

    // Use test data (bundle extracted): every test image listed after the
    // user's rows, the mode's recommendation checked; pressing again after a
    // mode switch re-targets the checks and keeps the user's rows/state.
    auto* layoutCheck =
            dialog.findChild<QCheckBox*>(QStringLiteral("qOCRLayout"));
    if (QDir(root).exists() && layoutCheck) {
        const int bundle =
                OCRDialog::testDataImages(root, true, false, true).size();
        table->setChecked(true);
        layoutCheck->setChecked(false);
        testData->click();
        QStringList checked = dialog.checkedImages();
        if (imageList->count() != bundle + 2 ||
            checked.size() !=
                    1 + OCRDialog::recommendedTestImages(true, false, true)
                                    .size() ||
            checked.first() != QFileInfo(own).absoluteFilePath()) {
            std::fprintf(stderr,
                         "qOCR test data (table crops): %d rows, %d checked\n",
                         imageList->count(), static_cast<int>(checked.size()));
            return 1;
        }
        layoutCheck->setChecked(true);
        testData->click();
        checked = dialog.checkedImages();
        if (imageList->count() != bundle + 2 ||
            checked.size() !=
                    1 + OCRDialog::recommendedTestImages(true, true, true)
                                    .size() ||
            !checked.last().endsWith(
                    QStringLiteral("layout/table_page_layout.jpg"))) {
            std::fprintf(stderr, "qOCR test data must follow the mode\n");
            return 1;
        }
        std::printf("qOCR test data: %d images listed, mode checks OK\n",
                    bundle);
        table->setChecked(false);
        layoutCheck->setChecked(false);
    }
    if (checkTableGrid() != 0) return 1;
    if (app.arguments().contains(QStringLiteral("--require-e2e"))) {
        const int status = checkHeadlessTable();
        if (status != 0) return status;
    }

    std::printf("qOCR workspace contract OK\n");
    return 0;
}
