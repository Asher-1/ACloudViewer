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
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSizePolicy>
#include <QSpinBox>
#include <QTableWidget>
#include <QVBoxLayout>

#include "OCRWorker.h"
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/ocr_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {
// Official demo images inside the shared Ppocr dataset bundle (one per
// supported script). The dataset registration owns the URL, the zip
// cache, the integrity ledger, and extraction.
constexpr const char* kDemoImageCh = "ppocrv4.png";
constexpr const char* kDemoImageEn = "word_1.png";
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

    // ---- input
    // ---------------------------------------------------------------
    auto* imageRow = new QHBoxLayout;
    imageRow->setSpacing(ecvAICoreUi::hSpacing());
    m_imagePath = new QLineEdit(this);
    m_imagePath->setObjectName(QStringLiteral("qOCRImagePath"));
    m_imagePath->setPlaceholderText(tr("Image to recognize"));
    m_testDataButton = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataButton->setObjectName(QStringLiteral("qOCRUseTestData"));
    m_testDataButton->setToolTip(
            tr("Load the official PaddleOCR demo image for the selected "
               "language"));
    m_browseButton = ecvAICoreUi::makeBrowseBtn(tr("Browse…"), this);
    imageRow->addWidget(m_imagePath, 1);
    imageRow->addWidget(m_testDataButton);
    imageRow->addWidget(m_browseButton);
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
    m_addDbCheck = new QCheckBox(tr("Add result to DB"), this);
    m_addDbCheck->setObjectName(QStringLiteral("qOCRAddToDb"));
    m_addDbCheck->setChecked(true);
    m_addDbCheck->setToolTip(
            tr("Export an annotated image (boxes + transcriptions) to the DB "
               "tree after each run"));
    outputRow->addWidget(m_addDbCheck);
    m_showInWindowCheck = new QCheckBox(tr("Show in render window"), this);
    m_showInWindowCheck->setObjectName(QStringLiteral("qOCRShowInWindow"));
    m_showInWindowCheck->setChecked(true);
    m_showInWindowCheck->setToolTip(
            tr("Overlay the annotated result image on the active render "
               "window (the image stays in the DB tree)"));
    outputRow->addWidget(m_showInWindowCheck);
    outputRow->addStretch();
    layout->addLayout(outputRow);

    // ---- download / run actions
    // ------------------------------------------------
    m_downloadButton = new QPushButton(tr("Download missing models"), this);
    m_downloadButton->setObjectName(QStringLiteral("qOCRDownload"));
    m_downloadButton->setToolTip(
            tr("Fetch the selected det/rec GGUF pair into the shared AICore "
               "model cache (SHA-256 verified against the pinned digests)"));
    layout->addWidget(m_downloadButton);

    auto* actionRow = new QHBoxLayout;
    actionRow->setSpacing(ecvAICoreUi::hSpacing());
    actionRow->addStretch();
    m_runButton = new QPushButton(tr("Run OCR"), this);
    m_runButton->setObjectName(QStringLiteral("qOCRRun"));
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_runButton->setToolTip(
            tr("Run det + rec over the selected image (worker thread)"));
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
    m_results = new QTableWidget(0, 4, this);
    m_results->setObjectName(QStringLiteral("qOCRResults"));
    m_results->setHorizontalHeaderLabels(
            {tr("Text"), tr("Score"), tr("Language"), tr("Box")});
    m_results->horizontalHeader()->setStretchLastSection(true);
    m_results->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(m_results, 1);

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
            &OCRDialog::onBrowseImage);
    connect(m_language, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { updateRunState(); });
    connect(m_quant, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { updateRunState(); });

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

QString OCRDialog::detFileName() const {
    // The DB detector is script-agnostic; the catalog publishes it under
    // the Chinese stem and both languages share it.
    const QString quant = m_quant->currentData().toString();
    return QStringLiteral("ppocr-det-ch-%1.gguf").arg(quant);
}

QString OCRDialog::recFileName() const {
    const QString lang = m_language->currentData().toString();
    const QString quant = m_quant->currentData().toString();
    return QStringLiteral("ppocr-rec-%1-%2.gguf").arg(lang, quant);
}

bool OCRDialog::modelPresent(const QString& filename) const {
    if (filename.isEmpty()) return false;
    QFileInfo info(QDir(cacheDir()).filePath(filename));
    return info.isFile() && info.size() > 1024 * 1024;
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
    QStringList missing;
    if (!modelPresent(detFileName())) missing << detFileName();
    if (!modelPresent(recFileName())) missing << recFileName();
    if (missing.isEmpty()) {
        appendLog(tr("[qOCR] Selected det/rec models are already cached."));
        updateRunState();
        return;
    }
    const QString filename = missing.first();
    const QString url = QStringLiteral("%1%2").arg(
            QString::fromUtf8(aicore_ocr_model_download_base()), filename);
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

void OCRDialog::fillTestData() {
    m_testDataInProgress = false;
    m_testDataButton->setEnabled(true);
    const bool chinese =
            m_language->currentData().toString() == QStringLiteral("ch");
    const QString name =
            QString::fromUtf8(chinese ? kDemoImageCh : kDemoImageEn);
    QString path = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::Ppocr, name);
    if (path.isEmpty()) {
        // Fall back to any image from the same script group.
        path = ecvTestDataRepository::findDatasetFile(
                ecvTestDataRepository::Dataset::Ppocr,
                chinese ? QStringLiteral("general_ocr_02.png")
                        : QStringLiteral("word_2.png"));
    }
    if (path.isEmpty()) {
        appendLog(tr("[Test data] PaddleOCR demo images unavailable."));
        return;
    }
    m_imagePath->setText(path);
    appendLog(tr("[Test data] Image ready: %1").arg(path));
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
    updateRunState();
}

void OCRDialog::onUseTestData() {
    if (m_downloadInProgress || m_testDataInProgress) {
        appendLog(tr("[qOCR] Another operation is in progress."));
        return;
    }
    auto& repo = ecvTestDataRepository::instance();
    const auto kind = ecvTestDataRepository::Dataset::Ppocr;
    const bool chinese =
            m_language->currentData().toString() == QStringLiteral("ch");
    const QString name =
            QString::fromUtf8(chinese ? kDemoImageCh : kDemoImageEn);
    if (!ecvTestDataRepository::findDatasetFile(kind, name).isEmpty()) {
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

void OCRDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qOCR"),
                                             QStringLiteral("lastImageFileDir"),
                                             QDir::homePath());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qOCR"),
                         QStringLiteral("lastImageFileDir"), path);
    updateRunState();
}

void OCRDialog::onRun() {
    if (m_runInProgress || m_downloadInProgress || m_testDataInProgress) {
        appendLog(tr("[qOCR] Another operation is in progress."));
        return;
    }
    const QString image = m_imagePath->text().trimmed();
    if (image.isEmpty() || !QFileInfo::exists(image)) {
        appendLog(tr("[qOCR] Select an image first."));
        return;
    }
    if (!modelPresent(detFileName()) || !modelPresent(recFileName())) {
        appendLog(
                tr("[qOCR] Download the det/rec models first (or press "
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
        settings.layoutPath =
                QDir(cacheDir())
                        .filePath(QStringLiteral("pplcnet-layout-%1.gguf")
                                          .arg(m_quant->currentData()
                                                       .toString()));
    }
    settings.imagePath = m_imagePath->text().trimmed();
    settings.device = m_deviceCombo->currentData().toString();
    if (settings.device.isEmpty()) settings.device = QStringLiteral("auto");
    settings.threads = m_threads->value();
    settings.addToDb = m_addDbCheck->isChecked();
    settings.showInWindow = m_showInWindowCheck->isChecked();
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
        m_results->setRowCount(0);
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
    // Header order: Text | Score | Language | Box (the box column shows the
    // axis-aligned extent of the ordered quad).
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
        m_results->setItem(i, 0, new QTableWidgetItem(box.text));
        m_results->setItem(
                i, 1, new QTableWidgetItem(QString::number(box.score, 'f', 3)));
        m_results->setItem(i, 2, new QTableWidgetItem(box.language));
        m_results->setItem(i, 3, new QTableWidgetItem(extents));
    }
    const int layoutCount = result.regions.size();
    m_status->setText(
            tr("OCR done: %1 text region(s), %2 layout region(s), inference "
               "%3 ms, post %4 ms")
                    .arg(count)
                    .arg(layoutCount)
                    .arg(result.inferenceMs, 0, 'f', 0)
                    .arg(result.postprocessMs, 0, 'f', 0));
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
    const QString imagePath =
            settings.value(QStringLiteral("imagePath")).toString();
    if (!imagePath.isEmpty()) m_imagePath->setText(imagePath);
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_showInWindowCheck->setChecked(
            settings.value(QStringLiteral("showInWindow"), true).toBool());
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
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("showInWindow"),
                      m_showInWindowCheck->isChecked());
    settings.endGroup();
}

void OCRDialog::updateRunState() {
    const bool modelsReady =
            modelPresent(detFileName()) && modelPresent(recFileName());
    const bool imageReady = !m_imagePath->text().trimmed().isEmpty();
    const bool layoutReady =
            modelPresent(QStringLiteral("pplcnet-layout-%1.gguf")
                                 .arg(m_quant->currentData().toString()));
    if (!m_runInProgress) {
        m_status->setText(
                tr("det: %1 | rec: %2 | layout: %3 | image: %4 | device: %5")
                        .arg(modelsReady ? tr("cached") : tr("missing"),
                             modelsReady ? tr("cached") : tr("missing"),
                             layoutReady ? tr("cached") : tr("missing"),
                             imageReady ? tr("set") : tr("unset"),
                             m_deviceCombo->currentData().toString()));
    }
    // The ocr runtime is live: run is enabled when the staged models and a
    // selected image line up and nothing else is in flight.
    m_runButton->setEnabled(modelsReady && imageReady && !m_runInProgress &&
                            !m_downloadInProgress && !m_testDataInProgress);
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
