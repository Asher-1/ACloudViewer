// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OCRDialog.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QSettings>
#include <QSizePolicy>
#include <QSpinBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QVBoxLayout>

#include "OCRWorker.h"
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {
// Image suffixes listed from the shared Ppocr dataset bundle (the dataset
// registration owns the URL, the zip cache, the integrity ledger, and
// extraction).
const QStringList kImageFilters = {
        QStringLiteral("*.png"), QStringLiteral("*.jpg"),
        QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")};

QString ppocrRoot() {
    return ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::Ppocr);
}

bool writeTableFile(const QString& path,
                    const OCRTableResult& table,
                    bool html) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    if (html) {
        file.write(table.html.toUtf8());
    } else {
        // UTF-8 BOM: spreadsheet applications otherwise decode the CJK cell
        // text with the system code page.
        file.write("\xEF\xBB\xBF");
        file.write(table.csv);
    }
    return true;
}
}  // namespace

OCRDialog::OCRDialog(QWidget* parent) : QDialog(parent) {
    setObjectName(QStringLiteral("qOCRWorkspace"));
    setWindowTitle(tr("OCR Text Recognition (PP-OCRv4)"));
    setMinimumSize(ecvAICoreUi::dpiScaled(680), ecvAICoreUi::dpiScaled(520));

    auto* layout = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(layout);
    layout->setSizeConstraint(QLayout::SetNoConstraint);

    // ---- model selection ----------------------------------------------------
    auto* form = new QFormLayout;
    form->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    form->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    layout->addLayout(form);

    m_language = new QComboBox(this);
    m_language->setObjectName(QStringLiteral("qOCRLanguage"));
    m_language->addItem(tr("Chinese"), QStringLiteral("ch"));
    m_language->addItem(tr("English"), QStringLiteral("en"));
    form->addRow(ecvAICoreUi::makeLabel(tr("Recognition language:")),
                 m_language);

    m_preset = new QComboBox(this);
    m_preset->setObjectName(QStringLiteral("qOCRPreset"));
    for (int i = 0; i < aicore_ocr_preset_count(); ++i) {
        const aicore_ocr_preset_entry* p = aicore_ocr_preset_at(i);
        if (p && p->display_name && p->id)
            m_preset->addItem(QString::fromUtf8(p->display_name),
                              QString::fromUtf8(p->id));
    }
    form->addRow(ecvAICoreUi::makeLabel(tr("Model recipe:")), m_preset);

    m_quant = new QComboBox(this);
    m_quant->setObjectName(QStringLiteral("qOCRQuant"));
    m_quant->addItem(QStringLiteral("F32"), QStringLiteral("f32"));
    m_quant->addItem(QStringLiteral("F16"), QStringLiteral("f16"));
    m_quant->addItem(QStringLiteral("Q8_0"), QStringLiteral("q8_0"));
    m_quant->setCurrentIndex(1);  // F16 is the recommended trade-off
    form->addRow(ecvAICoreUi::makeLabel(tr("Quantization:")), m_quant);

    // ---- runtime row (device + threads; shared AICore layout) --------------
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->setObjectName(QStringLiteral("qOCRDevice"));
    {
        const int nDev = aicore_device_count();
        for (int i = 0; i < nDev; ++i) {
            const aicore_device_info* dev = aicore_device_at(i);
            if (!dev || !dev->id) continue;
            m_deviceCombo->addItem(QString::fromUtf8(dev->label),
                                   QString::fromUtf8(dev->id));
            if (dev->is_default) m_deviceCombo->setCurrentIndex(i);
        }
    }
    m_threads = new QSpinBox(this);
    m_threads->setObjectName(QStringLiteral("qOCRThreads"));
    m_threads->setRange(0, 64);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = auto"));
    layout->addWidget(
            ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads, this));

    // ---- input: checkable image list (run = every checked image) ----------
    m_imageList = new QListWidget(this);
    m_imageList->setObjectName(QStringLiteral("qOCRImageList"));
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_imageList->setMinimumHeight(ecvAICoreUi::dpiScaled(96));
    m_imageList->setMaximumHeight(ecvAICoreUi::dpiScaled(150));
    m_imageList->setToolTip(
            tr("Checked images are recognized in list order with one model "
               "load. Add any files, or list the official test data."));
    layout->addWidget(m_imageList);
    auto* imageRow = new QHBoxLayout;
    imageRow->setSpacing(ecvAICoreUi::hSpacing());
    m_testDataButton = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataButton->setObjectName(QStringLiteral("qOCRUseTestData"));
    m_testDataButton->setToolTip(
            tr("List every official PaddleOCR test image and check the ones "
               "recommended for the current mode (tables / layout / "
               "language). Your own images keep their state."));
    // Not makeBrowseBtn: its fixed width is sized for "Browse…".
    m_browseButton = new QPushButton(tr("Add images…"), this);
    m_browseButton->setObjectName(QStringLiteral("qOCRAddImages"));
    m_removeImagesButton = new QPushButton(tr("Remove"), this);
    m_removeImagesButton->setObjectName(QStringLiteral("qOCRRemoveImages"));
    m_removeImagesButton->setToolTip(tr("Remove the selected rows"));
    m_checkAllButton = new QPushButton(tr("Check all"), this);
    m_checkAllButton->setObjectName(QStringLiteral("qOCRCheckAll"));
    m_checkNoneButton = new QPushButton(tr("Uncheck all"), this);
    m_checkNoneButton->setObjectName(QStringLiteral("qOCRCheckNone"));
    imageRow->addWidget(m_testDataButton);
    imageRow->addWidget(m_browseButton);
    imageRow->addWidget(m_removeImagesButton);
    imageRow->addStretch();
    imageRow->addWidget(m_checkAllButton);
    imageRow->addWidget(m_checkNoneButton);
    layout->addLayout(imageRow);

    // ---- output
    // ---------------------------------------------------------------
    auto* outputRow = new QHBoxLayout;
    outputRow->setSpacing(ecvAICoreUi::hSpacing());
    m_layoutCheck = new QCheckBox(tr("Layout regions (PP-Structure)"), this);
    m_layoutCheck->setObjectName(QStringLiteral("qOCRLayout"));
    m_layoutCheck->setToolTip(
            tr("Also run the PicoDet layout model (when cached) and export "
               "each region as a rectangle with its class and reading "
               "order. The pack downloads on first use."));
    outputRow->addWidget(m_layoutCheck);
    m_tableCheck = new QCheckBox(tr("Tables (SLANet_plus)"), this);
    m_tableCheck->setObjectName(QStringLiteral("qOCRTable"));
    m_tableCheck->setToolTip(
            tr("Reconstruct table structure (cells, row/column spans) and "
               "fill the cells with the recognized text. With layout regions "
               "every detected table is rebuilt; otherwise the whole image is "
               "one table. View and export the result as HTML or CSV."));
    outputRow->addWidget(m_tableCheck);
    m_addDbCheck = new QCheckBox(tr("Add result to DB"), this);
    m_addDbCheck->setObjectName(QStringLiteral("qOCRAddToDb"));
    m_addDbCheck->setChecked(true);
    m_addDbCheck->setToolTip(
            tr("Add the numbered OCR overlay to the DB tree and active render "
               "window after each run"));
    outputRow->addWidget(m_addDbCheck);
    outputRow->addStretch();
    layout->addLayout(outputRow);

    // ---- download / run actions
    // ------------------------------------------------
    m_downloadButton = new QPushButton(tr("Download missing models"), this);
    m_downloadButton->setObjectName(QStringLiteral("qOCRDownload"));
    m_downloadButton->setToolTip(
            tr("Fetch the selected det/rec GGUF pair (plus the layout and "
               "table packs when enabled) into the shared AICore model cache "
               "(SHA-256 verified against the pinned digests)"));
    layout->addWidget(m_downloadButton);

    auto* actionRow = new QHBoxLayout;
    actionRow->setSpacing(ecvAICoreUi::hSpacing());
    actionRow->addStretch();
    m_runButton = new QPushButton(tr("Run OCR"), this);
    m_runButton->setObjectName(QStringLiteral("qOCRRun"));
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_runButton->setToolTip(
            tr("Run det + rec (+ layout / tables) over every checked image "
               "(worker thread, one model load)"));
    actionRow->addWidget(m_runButton);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setObjectName(QStringLiteral("qOCRCancel"));
    m_cancelButton->setEnabled(false);
    actionRow->addWidget(m_cancelButton);
    layout->addLayout(actionRow);

    // ---- status / progress (shared AICore section)
    // -----------------------------
    m_status = new QLabel(this);
    m_status->setObjectName(QStringLiteral("qOCRStatus"));
    m_status->setWordWrap(true);
    layout->addWidget(m_status);

    m_taskStatusLabel = new QLabel(this);
    m_taskStatusLabel->setObjectName(QStringLiteral("qOCRTaskStatus"));
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    layout->addWidget(m_taskStatusLabel);

    ecvAICoreUi::setupProgressSection(layout, m_downloadLabel, m_progress);
    m_progress->setObjectName(QStringLiteral("qOCRProgress"));

    // ---- results ------------------------------------------------------------
    auto* resultBar = new QHBoxLayout;
    resultBar->setSpacing(ecvAICoreUi::hSpacing());
    m_resultImage = new QComboBox(this);
    m_resultImage->setObjectName(QStringLiteral("qOCRResultImage"));
    m_resultImage->setToolTip(tr("Processed images of the last run"));
    m_resultImage->setEnabled(false);
    m_saveAllButton = new QPushButton(tr("Save all tables…"), this);
    m_saveAllButton->setObjectName(QStringLiteral("qOCRTableSaveAll"));
    m_saveAllButton->setToolTip(
            tr("Write <image>_table<i>.html and .csv for every table of "
               "every processed image into one folder"));
    m_saveAllButton->setEnabled(false);
    resultBar->addWidget(ecvAICoreUi::makeLabel(tr("Result:")));
    resultBar->addWidget(m_resultImage, 1);
    resultBar->addWidget(m_saveAllButton);
    layout->addLayout(resultBar);
    m_resultSummary = new QLabel(this);
    m_resultSummary->setObjectName(QStringLiteral("qOCRResultSummary"));
    m_resultSummary->setWordWrap(true);
    layout->addWidget(m_resultSummary);
    m_resultTabs = new QTabWidget(this);
    m_resultTabs->setObjectName(QStringLiteral("qOCRResultTabs"));
    m_results = new QTableWidget(0, 6, this);
    m_results->setObjectName(QStringLiteral("qOCRResults"));
    m_results->setHorizontalHeaderLabels({tr("#"), tr("Text"), tr("Score"),
                                          tr("Language"), tr("Box"),
                                          tr("Layout")});
    m_results->horizontalHeader()->setStretchLastSection(true);
    m_results->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultTabs->addTab(m_results, tr("Text"));

    auto* tablePage = new QWidget(this);
    auto* tableLayout = new QVBoxLayout(tablePage);
    tableLayout->setContentsMargins(0, 0, 0, 0);
    auto* tableBar = new QHBoxLayout;
    tableBar->setSpacing(ecvAICoreUi::hSpacing());
    m_tableSelect = new QComboBox(tablePage);
    m_tableSelect->setObjectName(QStringLiteral("qOCRTableSelect"));
    m_saveHtmlButton = new QPushButton(tr("Save HTML…"), tablePage);
    m_saveHtmlButton->setObjectName(QStringLiteral("qOCRTableSaveHtml"));
    m_saveCsvButton = new QPushButton(tr("Save CSV…"), tablePage);
    m_saveCsvButton->setObjectName(QStringLiteral("qOCRTableSaveCsv"));
    tableBar->addWidget(m_tableSelect, 1);
    tableBar->addWidget(m_saveHtmlButton);
    tableBar->addWidget(m_saveCsvButton);
    tableLayout->addLayout(tableBar);
    m_tableGrid = new QTableWidget(0, 0, tablePage);
    m_tableGrid->setObjectName(QStringLiteral("qOCRTableGrid"));
    m_tableGrid->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tableGrid->setWordWrap(true);
    tableLayout->addWidget(m_tableGrid, 1);
    m_resultTabs->addTab(tablePage, tr("Tables"));
    m_tableSelect->setEnabled(false);
    m_saveHtmlButton->setEnabled(false);
    m_saveCsvButton->setEnabled(false);
    layout->addWidget(m_resultTabs, 1);

    // ---- wiring ------------------------------------------------------------
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                m_progress->setVisible(true);
                if (total > 0) {
                    m_progress->setRange(0, 100);
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                    m_downloadLabel->setText(
                            tr("Downloading %1: %2")
                                    .arg(m_activeDownloadKey,
                                         ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                    m_downloadLabel->setVisible(true);
                }
            });
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &OCRDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &OCRDialog::onDownloadFinished);

    connect(m_downloadButton, &QPushButton::clicked, this,
            &OCRDialog::onDownloadMissing);
    // Test data flows through the shared ecvTestDataRepository (the
    // Dataset::Ppocr registration owns the URL, the zip cache, the
    // integrity ledger, and extraction) — this dialog only filters its
    // broadcasts by dataset kind.
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataInProgress) return;
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_progress->setVisible(true);
                m_downloadLabel->setText(statusText);
                m_downloadLabel->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataInProgress) appendLog(message);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            &OCRDialog::onTestDataDownloadFinished);
    connect(&testDataRepo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataInProgress || total <= 0) return;
                m_progress->setRange(0, total);
                m_progress->setValue(current);
                m_progress->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            &OCRDialog::onTestDataExtractionFinished);
    connect(m_testDataButton, &QPushButton::clicked, this,
            &OCRDialog::onUseTestData);
    connect(m_runButton, &QPushButton::clicked, this, &OCRDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this, &OCRDialog::onCancel);
    connect(m_browseButton, &QPushButton::clicked, this,
            &OCRDialog::onAddImages);
    connect(m_removeImagesButton, &QPushButton::clicked, this, [this]() {
        for (QListWidgetItem* item : m_imageList->selectedItems()) delete item;
        updateRunState();
    });
    auto checkAll = [this](Qt::CheckState state) {
        for (int i = 0; i < m_imageList->count(); ++i)
            m_imageList->item(i)->setCheckState(state);
        updateRunState();
    };
    connect(m_checkAllButton, &QPushButton::clicked, this,
            [checkAll]() { checkAll(Qt::Checked); });
    connect(m_checkNoneButton, &QPushButton::clicked, this,
            [checkAll]() { checkAll(Qt::Unchecked); });
    connect(m_imageList, &QListWidget::itemChanged, this,
            [this](QListWidgetItem*) { updateRunState(); });
    connect(m_resultImage, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &OCRDialog::showRunResult);
    connect(m_saveAllButton, &QPushButton::clicked, this,
            &OCRDialog::exportAllTables);
    connect(m_preset, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                // Keep the language combo (test-data chooser) consistent with
                // the selected recipe's recognition family.
                const aicore_ocr_preset_entry* p = aicore_ocr_preset_at(index);
                if (!p || !p->rec_stem) return;
                const QString stem = QString::fromUtf8(p->rec_stem);
                const int lang = stem.contains("-en") ? 1 : 0;
                if (m_language->currentIndex() != lang)
                    m_language->setCurrentIndex(lang);
                updateRunState();
            });
    connect(m_language, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateRunState(); });
    connect(m_quant, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { updateRunState(); });
    connect(m_layoutCheck, &QCheckBox::toggled, this,
            [this](bool) { updateRunState(); });
    connect(m_tableCheck, &QCheckBox::toggled, this,
            [this](bool) { updateRunState(); });
    connect(m_tableSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &OCRDialog::showTable);
    connect(m_saveHtmlButton, &QPushButton::clicked, this,
            [this]() { exportTable(true); });
    connect(m_saveCsvButton, &QPushButton::clicked, this,
            [this]() { exportTable(false); });

    appendLog(tr("[qOCR] PP-OCRv4 catalog: %1 entries published.")
                      .arg(aicore_ocr_model_count()));
    loadSettings();
    updateRunState();
}

OCRDialog::~OCRDialog() { saveSettings(); }

QString OCRDialog::cacheDir() const {
    char* dir = aicore_ocr_model_cache_dir();
    if (!dir) return QString();
    const QString out = QString::fromUtf8(dir);
    aicore_ocr_free_buffer(dir);
    return out;
}

const aicore_ocr_preset_entry* OCRDialog::currentPreset() const {
    const QString id = m_preset->currentData().toString();
    for (int i = 0; i < aicore_ocr_preset_count(); ++i) {
        const aicore_ocr_preset_entry* p = aicore_ocr_preset_at(i);
        if (p && p->id && id == QString::fromUtf8(p->id)) return p;
    }
    return aicore_ocr_preset_at(0);
}

QString OCRDialog::detFileName() const {
    // The recipe owns the det/rec/cls stem pairing; the quantization tier is
    // the only user-facing axis. The DB detector is script-agnostic.
    const aicore_ocr_preset_entry* p = currentPreset();
    return QStringLiteral("%1-%2.gguf")
            .arg(QString::fromUtf8(p->det_stem),
                 m_quant->currentData().toString());
}

QString OCRDialog::recFileName() const {
    const aicore_ocr_preset_entry* p = currentPreset();
    return QStringLiteral("%1-%2.gguf")
            .arg(QString::fromUtf8(p->rec_stem),
                 m_quant->currentData().toString());
}

QString OCRDialog::layoutFileName() const {
    // Same quantization tier as det/rec (the layout graph is script-agnostic).
    return QStringLiteral("pplcnet-layout-%1.gguf")
            .arg(m_quant->currentData().toString());
}

QString OCRDialog::tableFileName() const {
    // The table catalog publishes one SLANet_plus pack per tier.
    const QString suffix =
            QStringLiteral("-%1.gguf").arg(m_quant->currentData().toString());
    for (int i = 0; i < aicore_table_model_count(); ++i) {
        const aicore_table_model_entry* e = aicore_table_model_at(i);
        if (!e || !e->filename) continue;
        const QString fn = QString::fromUtf8(e->filename);
        if (fn.endsWith(suffix)) return fn;
    }
    return QString();
}

bool OCRDialog::modelPresent(const QString& filename) const {
    if (filename.isEmpty()) return false;
    const QString path = QDir(cacheDir()).filePath(filename);
    const ecvAssetIntegrity::Anchor anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(filename)};
    return ecvAssetIntegrity::isVerified(path, anchor, 1024 * 1024, true);
}

void OCRDialog::onDownloadMissing() {
    if (m_downloadInProgress) {
        appendLog(tr("[qOCR] A download is already running."));
        return;
    }
    QDir().mkpath(cacheDir());
    pumpQueue();
}

void OCRDialog::pumpQueue() {
    // The OCR and table catalogs share one cache folder; each file
    // downloads from its own catalog base URL.
    QStringList missing;
    if (!modelPresent(detFileName())) missing << detFileName();
    if (!modelPresent(recFileName())) missing << recFileName();
    if (m_layoutCheck->isChecked() && !modelPresent(layoutFileName()))
        missing << layoutFileName();
    const QString table = tableFileName();
    if (m_tableCheck->isChecked() && !table.isEmpty() && !modelPresent(table))
        missing << table;
    if (missing.isEmpty()) {
        appendLog(tr("[qOCR] Selected models are already cached."));
        updateRunState();
        return;
    }
    const QString filename = missing.first();
    const char* base = filename == table ? aicore_table_model_download_base()
                                         : aicore_ocr_model_download_base();
    const QString url =
            QStringLiteral("%1%2").arg(QString::fromUtf8(base), filename);
    m_downloadInProgress = true;
    m_downloadButton->setEnabled(false);
    startDownload(url, QDir(cacheDir()).filePath(filename),
                  /*verifyDigest=*/true, 1024 * 1024);
}

void OCRDialog::startDownload(const QString& url,
                              const QString& destPath,
                              bool verifyDigest,
                              qint64 minBytes) {
    m_activeDownloadKey = QFileInfo(destPath).fileName();
    appendLog(tr("[qOCR] Downloading %1…").arg(url));
    ecvModelDownloader::Request req;
    req.url = url;
    req.destPath = destPath;
    req.minBytes = minBytes;
    req.requireGgufMagic = verifyDigest;
    if (verifyDigest) {
        // Content identity from the pinned digest registry — streamed
        // SHA-256 check at ingestion (truncation and corruption caught).
        req.contentAnchor = {
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(m_activeDownloadKey)};
    }
    m_downloader->download(req);
}

void OCRDialog::onDownloadFinished(bool ok, const QString& destPath) {
    m_downloadInProgress = false;
    m_downloadButton->setEnabled(true);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
    const QString filename = QFileInfo(destPath).fileName();
    if (ok) {
        appendLog(tr("[qOCR] Downloaded %1.").arg(filename));
        m_status->setText(tr("Downloaded %1.").arg(filename));
        // Continue the model queue; the test-image download never queues.
        pumpQueue();
        return;
    }
    appendLog(tr("[qOCR] Download failed for %1.").arg(filename));
    m_status->setText(tr("Download failed."));
    updateRunState();
}

QStringList OCRDialog::recommendedTestImages(bool table,
                                             bool layout,
                                             bool chinese) {
    if (table && layout) {
        // PP-Structure pages: layout finds the tables, SLANet rebuilds them.
        return {QStringLiteral("layout/table_page_1.png"),
                QStringLiteral("layout/table_page_layout.jpg")};
    }
    if (table) {
        // Whole image = one real-document table: the PaddleOCR medal table,
        // the PP-Structure docs table and the PaddleX v2 demo form (merged
        // cells). The synthetic table_rec/ GT crops stay listed unchecked.
        return {QStringLiteral("test_files/medal_table.png"),
                QStringLiteral("tables/ppstructure_table.jpg"),
                QStringLiteral("tables/table_recognition_v2.jpg")};
    }
    if (layout) {
        return {QStringLiteral("layout/table_page_layout.jpg"),
                QStringLiteral("layout/table_page_1.png")};
    }
    if (chinese) {
        return {QStringLiteral("scenes/ppocrv4.png"),
                QStringLiteral("imgs/00006737.jpg"),
                QStringLiteral("imgs_words_ch/word_1.jpg")};
    }
    return {QStringLiteral("test_files/textline.png"),
            QStringLiteral("words_en/word_1.png")};
}

bool OCRDialog::isSyntheticTestImage(const QString& relativePath) {
    return relativePath.startsWith(QStringLiteral("table_rec/")) ||
           relativePath == QStringLiteral("test_files/table.jpg") ||
           relativePath == QStringLiteral("layout_doc_test.png");
}

QStringList OCRDialog::testDataImages(const QString& root,
                                      bool table,
                                      bool layout,
                                      bool chinese) {
    const QDir dir(root);
    if (root.isEmpty() || !dir.exists()) return {};
    QStringList rest;
    // O(files) walk of the extracted bundle (~165 entries).
    QDirIterator it(root, kImageFilters, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) rest << dir.relativeFilePath(it.next());
    rest.sort();
    QStringList out;
    for (const QString& rel : recommendedTestImages(table, layout, chinese)) {
        if (rest.removeOne(rel)) out << dir.absoluteFilePath(rel);
    }
    for (const QString& rel : rest) out << dir.absoluteFilePath(rel);
    return out;
}

QString OCRDialog::imageLabel(const QString& path) const {
    const QString root = ppocrRoot();
    const QString rel = QDir(root).relativeFilePath(path);
    if (!root.isEmpty() && !rel.startsWith(QStringLiteral("..")) &&
        !QDir::isAbsolutePath(rel)) {
        return isSyntheticTestImage(rel) ? tr("[test, synthetic] %1").arg(rel)
                                         : tr("[test] %1").arg(rel);
    }
    return QFileInfo(path).fileName();
}

int OCRDialog::addImages(const QStringList& paths, bool checked) {
    QSet<QString> present;
    for (int i = 0; i < m_imageList->count(); ++i)
        present.insert(m_imageList->item(i)->data(Qt::UserRole).toString());
    int added = 0;
    m_imageList->blockSignals(true);
    for (const QString& raw : paths) {
        const QString path = QFileInfo(raw).absoluteFilePath();
        if (raw.isEmpty() || present.contains(path)) continue;
        present.insert(path);
        auto* item = new QListWidgetItem(imageLabel(path), m_imageList);
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
        ++added;
    }
    m_imageList->blockSignals(false);
    updateRunState();
    return added;
}

QStringList OCRDialog::checkedImages() const {
    QStringList out;
    for (int i = 0; i < m_imageList->count(); ++i) {
        const QListWidgetItem* item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked)
            out << item->data(Qt::UserRole).toString();
    }
    return out;
}

void OCRDialog::fillTestData() {
    m_testDataInProgress = false;
    m_testDataButton->setEnabled(true);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
    const bool table = m_tableCheck->isChecked();
    const bool layout = m_layoutCheck->isChecked();
    const bool chinese =
            m_language->currentData().toString() == QStringLiteral("ch");
    const QString root = QFileInfo(ppocrRoot()).absoluteFilePath();
    const QStringList images = testDataImages(root, table, layout, chinese);
    if (images.isEmpty()) {
        appendLog(tr("[Test data] PaddleOCR test images unavailable."));
        updateRunState();
        return;
    }
    // Rebuild the order: the user's own images keep their rows and state on
    // top, then every test image with the current mode's recommendation
    // checked (re-pressing after a mode switch re-targets the checks).
    QStringList own;
    QVector<Qt::CheckState> ownState;
    for (int i = 0; i < m_imageList->count(); ++i) {
        const QListWidgetItem* item = m_imageList->item(i);
        const QString path = item->data(Qt::UserRole).toString();
        if (path.startsWith(root + QLatin1Char('/'))) continue;
        own << path;
        ownState << item->checkState();
    }
    m_imageList->clear();
    addImages(own, false);
    for (int i = 0; i < ownState.size(); ++i)
        m_imageList->item(i)->setCheckState(ownState[i]);
    QSet<QString> recommended;
    for (const QString& rel : recommendedTestImages(table, layout, chinese))
        recommended.insert(QDir(root).absoluteFilePath(rel));
    addImages(images, false);
    int checked = 0;
    for (int i = own.size(); i < m_imageList->count(); ++i) {
        QListWidgetItem* item = m_imageList->item(i);
        if (!recommended.contains(item->data(Qt::UserRole).toString()))
            continue;
        item->setCheckState(Qt::Checked);
        ++checked;
    }
    if (QListWidgetItem* first = m_imageList->item(own.size()))
        m_imageList->scrollToItem(first);
    appendLog(tr("[Test data] %1 PaddleOCR test image(s) listed, %2 "
                 "recommended checked (%3).")
                      .arg(images.size())
                      .arg(checked)
                      .arg(table     ? (layout ? tr("table pages")
                                               : tr("table crops"))
                           : layout  ? tr("layout pages")
                           : chinese ? tr("Chinese")
                                     : tr("English")));
    updateRunState();
}

void OCRDialog::onUseTestData() {
    if (m_downloadInProgress || m_testDataInProgress) {
        appendLog(tr("[qOCR] Another operation is in progress."));
        return;
    }
    auto& repo = ecvTestDataRepository::instance();
    const auto kind = ecvTestDataRepository::Dataset::Ppocr;
    // An extraction from an older bundle lacks the newer images: every
    // recommended file of every mode must be on disk, else (re)extract or
    // download the pinned bundle.
    bool complete = QDir(ppocrRoot()).exists();
    for (const bool table : {false, true}) {
        for (const bool layout : {false, true}) {
            for (const bool chinese : {false, true}) {
                for (const QString& rel :
                     recommendedTestImages(table, layout, chinese)) {
                    complete =
                            complete &&
                            QFileInfo(QDir(ppocrRoot()).filePath(rel)).isFile();
                }
            }
        }
    }
    if (complete) {
        fillTestData();
        return;
    }
    m_testDataInProgress = true;
    m_testDataButton->setEnabled(false);
    m_progress->setVisible(true);
    m_downloadLabel->setVisible(true);
    m_downloadLabel->setText(tr("Preparing PaddleOCR test data..."));
    if (repo.isDatasetAvailable(kind)) {
        m_progress->setRange(0, 0);
        appendLog(
                tr("[Test data] Extracting the cached PaddleOCR "
                   "images..."));
        repo.extractDataset(kind);
        return;
    }
    if (repo.isDownloadInProgress()) {
        appendLog(
                tr("A shared dataset download is running; please retry "
                   "in a moment."));
        m_testDataInProgress = false;
        m_testDataButton->setEnabled(true);
        m_progress->setVisible(false);
        m_downloadLabel->setVisible(false);
        return;
    }
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    appendLog(
            tr("[Test data] Downloading the official PaddleOCR demo "
               "images..."));
    repo.startDownload(kind);
}

void OCRDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (kind != ecvTestDataRepository::Dataset::Ppocr) return;
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        fillTestData();
        return;
    }
    m_progress->setRange(0, 0);
    ecvTestDataRepository::instance().extractDataset(
            ecvTestDataRepository::Dataset::Ppocr);
}

void OCRDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (kind != ecvTestDataRepository::Dataset::Ppocr) return;
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Extraction failed."));
        fillTestData();
        return;
    }
    fillTestData();
}

void OCRDialog::onAddImages() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qOCR"),
                                             QStringLiteral("lastImageFileDir"),
                                             QDir::homePath());
    const QStringList paths = QFileDialog::getOpenFileNames(
            this, tr("Add images"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (paths.isEmpty()) return;
    const int added = addImages(paths, true);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qOCR"),
                         QStringLiteral("lastImageFileDir"), paths.first());
    appendLog(tr("[qOCR] %1 image(s) added.").arg(added));
}

void OCRDialog::onRun() {
    if (m_runInProgress || m_downloadInProgress || m_testDataInProgress) {
        appendLog(tr("[qOCR] Another operation is in progress."));
        return;
    }
    if (runSettings().imagePaths.isEmpty()) {
        appendLog(tr("[qOCR] Check at least one existing image first."));
        return;
    }
    if (!modelPresent(detFileName()) || !modelPresent(recFileName())) {
        appendLog(
                tr("[qOCR] Download the det/rec models first (or press "
                   "\"Download missing models\")."));
        return;
    }
    if ((m_layoutCheck->isChecked() && !modelPresent(layoutFileName())) ||
        (m_tableCheck->isChecked() && !modelPresent(tableFileName()))) {
        appendLog(
                tr("[qOCR] Download the layout/table packs first (press "
                   "\"Download missing models\")."));
        return;
    }
    emit runRequested(runSettings());
}

void OCRDialog::onCancel() { emit cancelRequested(); }

OCRRunSettings OCRDialog::runSettings() const {
    OCRRunSettings settings;
    settings.detPath = QDir(cacheDir()).filePath(detFileName());
    settings.recPath = QDir(cacheDir()).filePath(recFileName());
    // Layout pack: same cache directory, same quantization tier as the det
    // model (the layout graph is script-agnostic); enabled via the check.
    if (m_layoutCheck->isChecked()) {
        settings.layoutPath = QDir(cacheDir()).filePath(layoutFileName());
    }
    if (m_tableCheck->isChecked() && !tableFileName().isEmpty()) {
        settings.tablePath = QDir(cacheDir()).filePath(tableFileName());
    }
    for (const QString& path : checkedImages()) {
        if (QFileInfo(path).isFile()) settings.imagePaths << path;
    }
    settings.device = m_deviceCombo->currentData().toString();
    if (settings.device.isEmpty()) settings.device = QStringLiteral("auto");
    settings.threads = m_threads->value();
    settings.addToDb = m_addDbCheck->isChecked();
    return settings;
}

void OCRDialog::appendLog(const QString& msg) {
    // Single application log owner: the shared console (CVLog) — no private
    // dialog log widget. Errors surface through the Error channel.
    aicore_inference_log::log(msg);
    if (msg.startsWith(QStringLiteral("[Error]")) &&
        m_lastTaskError.isEmpty()) {
        m_lastTaskError = msg.mid(QStringLiteral("[Error]").size()).trimmed();
        if (m_taskStatusLabel) {
            m_taskStatusLabel->setText(m_lastTaskError);
            m_taskStatusLabel->setStyleSheet(
                    "font-weight: bold; color: #b91c1c;");
            m_taskStatusLabel->setVisible(true);
        }
    }
}

void OCRDialog::setRunning(bool running) {
    m_runInProgress = running;
    if (running) {
        m_lastTaskError.clear();
        m_taskStatusLabel->setText(tr("Starting..."));
        m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
        m_taskStatusLabel->setVisible(true);
        m_progress->setVisible(true);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
        m_runResults.clear();
        m_resultImage->blockSignals(true);
        m_resultImage->clear();
        m_resultImage->blockSignals(false);
        m_resultImage->setEnabled(false);
        m_saveAllButton->setEnabled(false);
        showRunResult(-1);
    } else {
        if (m_lastTaskError.isEmpty()) {
            m_taskStatusLabel->clear();
            m_taskStatusLabel->setVisible(false);
        }
        m_progress->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    }
    m_runButton->setEnabled(!running);
    m_cancelButton->setEnabled(running);
    m_downloadButton->setEnabled(!running);
    m_testDataButton->setEnabled(!running);
    m_browseButton->setEnabled(!running);
    m_removeImagesButton->setEnabled(!running);
    updateRunState();
}

void OCRDialog::setTaskStage(const QString& stage, int percent) {
    if (!m_taskStatusLabel) return;
    m_taskStatusLabel->setText(stage);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    m_taskStatusLabel->setVisible(true);
    m_progress->setVisible(true);
    if (percent >= 0) {
        m_progress->setRange(0, 100);
        m_progress->setValue(percent);
    } else {
        // Busy (indeterminate) mode — the aligned fallback for backends
        // without per-node progress.
        m_progress->setRange(0, 0);
    }
}

void OCRDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void OCRDialog::showResult(const OCRRunResult& result) {
    // Keep text + tables only: the DB overlay already owns the pixels, so a
    // long batch does not hold every decoded source image.
    OCRRunResult kept = result;
    kept.sourceImage = QImage();
    m_runResults.push_back(kept);
    m_resultImage->addItem(tr("%1. %2 — %3 text, %4 table(s)")
                                   .arg(m_runResults.size())
                                   .arg(QFileInfo(result.imagePath).fileName())
                                   .arg(result.boxes.size())
                                   .arg(result.tables.size()),
                           result.imagePath);
    m_resultImage->setEnabled(true);
    bool anyTable = false;
    for (const OCRRunResult& r : m_runResults) anyTable |= !r.tables.isEmpty();
    m_saveAllButton->setEnabled(anyTable);
    // Follow the newest image while the run streams results.
    m_resultImage->setCurrentIndex(m_runResults.size() - 1);
}

void OCRDialog::showRunResult(int index) {
    if (index < 0 || index >= m_runResults.size()) {
        m_results->setRowCount(0);
        m_tables.clear();
        m_tablesSource.clear();
        m_tableSelect->clear();
        m_tableSelect->setEnabled(false);
        m_saveHtmlButton->setEnabled(false);
        m_saveCsvButton->setEnabled(false);
        m_resultSummary->clear();
        showTable(-1);
        return;
    }
    const OCRRunResult& result = m_runResults[index];
    // A stable index binds the scrollable text list to the same number painted
    // next to its source quad. It avoids encoding unbounded text in image
    // width.
    const int count = result.boxes.size();
    m_results->setRowCount(count);
    for (int i = 0; i < count; ++i) {
        const OCRBoxResult& box = result.boxes[i];
        float minX = box.quad[0][0], maxX = minX;
        float minY = box.quad[0][1], maxY = minY;
        for (int c = 1; c < 4; ++c) {
            minX = std::min(minX, box.quad[c][0]);
            maxX = std::max(maxX, box.quad[c][0]);
            minY = std::min(minY, box.quad[c][1]);
            maxY = std::max(maxY, box.quad[c][1]);
        }
        const QString extents = QStringLiteral("(%1,%2)-(%3,%4)")
                                        .arg(minX, 0, 'f', 0)
                                        .arg(minY, 0, 'f', 0)
                                        .arg(maxX, 0, 'f', 0)
                                        .arg(maxY, 0, 'f', 0);
        QString layoutName = tr("—");
        for (const OCRRegionResult& region : result.regions) {
            if (minX >= region.quad[0][0] && minY >= region.quad[0][1] &&
                maxX <= region.quad[2][0] && maxY <= region.quad[2][1]) {
                layoutName = region.className;
                break;
            }
        }
        m_results->setItem(i, 0, new QTableWidgetItem(QString::number(i + 1)));
        m_results->setItem(i, 1, new QTableWidgetItem(box.text));
        m_results->setItem(
                i, 2, new QTableWidgetItem(QString::number(box.score, 'f', 3)));
        m_results->setItem(i, 3, new QTableWidgetItem(box.language));
        m_results->setItem(i, 4, new QTableWidgetItem(extents));
        m_results->setItem(i, 5, new QTableWidgetItem(layoutName));
    }
    const int layoutCount = result.regions.size();
    QString summary =
            tr("%1 text region(s), %2 layout region(s), inference %3 ms, "
               "post %4 ms")
                    .arg(count)
                    .arg(layoutCount)
                    .arg(result.inferenceMs, 0, 'f', 0)
                    .arg(result.postprocessMs, 0, 'f', 0);

    m_tables = result.tables;
    m_tablesSource = result.imagePath;
    m_tableSelect->blockSignals(true);
    m_tableSelect->clear();
    for (int t = 0; t < m_tables.size(); ++t) {
        const OCRTableResult& table = m_tables[t];
        m_tableSelect->addItem(tr("Table %1: %2 x %3, %4 cell(s)")
                                       .arg(t + 1)
                                       .arg(table.rows)
                                       .arg(table.cols)
                                       .arg(table.cells.size()));
    }
    m_tableSelect->blockSignals(false);
    const bool haveTables = !m_tables.isEmpty();
    m_tableSelect->setEnabled(haveTables);
    m_saveHtmlButton->setEnabled(haveTables);
    m_saveCsvButton->setEnabled(haveTables);
    showTable(haveTables ? 0 : -1);
    if (!result.tableModel.isEmpty()) {
        summary += tr("; %1 table(s) in %2 ms")
                           .arg(m_tables.size())
                           .arg(result.tableMs, 0, 'f', 0);
        if (haveTables) m_resultTabs->setCurrentIndex(1);
    }
    m_resultSummary->setText(summary);
}

void OCRDialog::fillTableGrid(QTableWidget* grid, const OCRTableResult& table) {
    grid->clear();
    grid->clearSpans();
    grid->setRowCount(table.rows);
    grid->setColumnCount(table.cols);
    for (const OCRTableCell& cell : table.cells) {
        if (cell.row < 0 || cell.col < 0 || cell.row >= table.rows ||
            cell.col >= table.cols) {
            continue;
        }
        grid->setItem(cell.row, cell.col, new QTableWidgetItem(cell.text));
        const int rowSpan = qMin(cell.rowSpan, table.rows - cell.row);
        const int colSpan = qMin(cell.colSpan, table.cols - cell.col);
        if (rowSpan > 1 || colSpan > 1)
            grid->setSpan(cell.row, cell.col, rowSpan, colSpan);
    }
    grid->resizeColumnsToContents();
    grid->resizeRowsToContents();
}

void OCRDialog::showTable(int index) {
    if (index < 0 || index >= m_tables.size()) {
        m_tableGrid->clear();
        m_tableGrid->clearSpans();
        m_tableGrid->setRowCount(0);
        m_tableGrid->setColumnCount(0);
        return;
    }
    fillTableGrid(m_tableGrid, m_tables[index]);
}

void OCRDialog::exportTable(bool html) {
    const int index = m_tableSelect->currentIndex();
    if (index < 0 || index >= m_tables.size()) return;
    const QFileInfo source(m_tablesSource);
    const QString suggested =
            QDir(source.absolutePath())
                    .filePath(
                            QStringLiteral("%1_table%2.%3")
                                    .arg(source.completeBaseName())
                                    .arg(index + 1)
                                    .arg(QLatin1String(html ? "html" : "csv")));
    const QString path = QFileDialog::getSaveFileName(
            this, html ? tr("Save table as HTML") : tr("Save table as CSV"),
            suggested, html ? tr("HTML (*.html *.htm)") : tr("CSV (*.csv)"));
    if (path.isEmpty()) return;
    if (!writeTableFile(path, m_tables[index], html)) {
        appendLog(tr("[Error] Cannot write %1").arg(path));
        return;
    }
    appendLog(tr("[qOCR] Table %1 saved to %2").arg(index + 1).arg(path));
}

void OCRDialog::exportAllTables() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qOCR"),
                             QStringLiteral("lastTableDir"), QDir::homePath());
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Save all tables to folder"), lastDir);
    if (dir.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qOCR"),
                         QStringLiteral("lastTableDir"),
                         QDir(dir).filePath(QStringLiteral("x")));
    // Same names as -OCR_RECOGNIZE (<stem>_table<i>.html/.csv); a repeated
    // stem (imgs/1.jpg vs scenes/1.jpg) gets a _2, _3... suffix.
    QSet<QString> stems;
    int written = 0, failed = 0;
    for (const OCRRunResult& r : m_runResults) {
        if (r.tables.isEmpty()) continue;
        const QString base = QFileInfo(r.imagePath).completeBaseName();
        QString stem = base;
        for (int n = 2; stems.contains(stem); ++n)
            stem = QStringLiteral("%1_%2").arg(base).arg(n);
        stems.insert(stem);
        for (int t = 0; t < r.tables.size(); ++t) {
            for (const bool html : {true, false}) {
                const QString path = QDir(dir).filePath(
                        QStringLiteral("%1_table%2.%3")
                                .arg(stem)
                                .arg(t + 1)
                                .arg(QLatin1String(html ? "html" : "csv")));
                writeTableFile(path, r.tables[t], html) ? ++written : ++failed;
            }
        }
    }
    appendLog(
            tr("[qOCR] %1 table file(s) written to %2%3")
                    .arg(written)
                    .arg(dir)
                    .arg(failed ? tr(" (%1 failed)").arg(failed) : QString()));
}

void OCRDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qOCR"));
    const int language = settings.value(QStringLiteral("language"), 0).toInt();
    if (language >= 0 && language < m_language->count()) {
        m_language->setCurrentIndex(language);
    }
    const int quant = settings.value(QStringLiteral("quantization"), 1).toInt();
    if (quant >= 0 && quant < m_quant->count()) m_quant->setCurrentIndex(quant);
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    // Restore the image list (files that vanished are dropped).
    const QStringList images =
            settings.value(QStringLiteral("images")).toStringList();
    const QStringList checked =
            settings.value(QStringLiteral("checkedImages")).toStringList();
    for (const QString& path : images) {
        if (QFileInfo(path).isFile()) addImages({path}, checked.contains(path));
    }
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_layoutCheck->setChecked(
            settings.value(QStringLiteral("layout"), false).toBool());
    m_tableCheck->setChecked(
            settings.value(QStringLiteral("table"), false).toBool());
    settings.endGroup();
}

void OCRDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qOCR"));
    settings.setValue(QStringLiteral("language"), m_language->currentIndex());
    settings.setValue(QStringLiteral("quantization"), m_quant->currentIndex());
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    QStringList images;
    for (int i = 0; i < m_imageList->count(); ++i)
        images << m_imageList->item(i)->data(Qt::UserRole).toString();
    settings.setValue(QStringLiteral("images"), images);
    settings.setValue(QStringLiteral("checkedImages"), checkedImages());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("layout"), m_layoutCheck->isChecked());
    settings.setValue(QStringLiteral("table"), m_tableCheck->isChecked());
    settings.endGroup();
}

void OCRDialog::updateRunState() {
    const bool modelsReady =
            modelPresent(detFileName()) && modelPresent(recFileName());
    const int checkedCount = checkedImages().size();
    const bool imageReady = checkedCount > 0;
    const bool layoutReady = modelPresent(layoutFileName());
    const bool tableReady = modelPresent(tableFileName());
    if (!m_runInProgress) {
        m_status->setText(
                tr("det: %1 | rec: %2 | layout: %3 | table: %4 | images: "
                   "%5/%6 checked | device: %7")
                        .arg(modelsReady ? tr("cached") : tr("missing"),
                             modelsReady ? tr("cached") : tr("missing"),
                             layoutReady ? tr("cached") : tr("missing"),
                             tableReady ? tr("cached") : tr("missing"),
                             QString::number(checkedCount),
                             QString::number(m_imageList->count()),
                             m_deviceCombo->currentData().toString()));
    }
    // The ocr runtime is live: run is enabled when the staged models (and
    // the enabled layout/table packs) and a selected image line up and
    // nothing else is in flight.
    const bool stagesReady = (!m_layoutCheck->isChecked() || layoutReady) &&
                             (!m_tableCheck->isChecked() || tableReady);
    m_runButton->setEnabled(modelsReady && stagesReady && imageReady &&
                            !m_runInProgress && !m_downloadInProgress &&
                            !m_testDataInProgress);
}

void OCRDialog::closeEvent(QCloseEvent* event) {
    // Uniform plugin-close semantics: if a task, download, or test-data
    // transfer is active, ask for confirmation before closing.
    if (m_runInProgress || m_downloadInProgress || m_testDataInProgress) {
        if (QMessageBox::question(this, tr("Task running"),
                                  tr("An OCR task is running. Close anyway?"),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes) {
            event->ignore();
            return;
        }
    }
    emit cancelRequested();
    saveSettings();
    event->accept();
}

void OCRDialog::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape &&
        (m_runInProgress || m_downloadInProgress || m_testDataInProgress)) {
        if (QMessageBox::question(this, tr("Task running"),
                                  tr("An OCR task is running. Close anyway?"),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes) {
            return;
        }
    }
    QDialog::keyPressEvent(event);
}
