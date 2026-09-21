// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Sam3dDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include "aicore/sam3d_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

namespace {

// The model files the current dtype needs, in download order (the MoGe
// conditioning model is shared and f16-pinned).
QStringList sam3dRequiredModels(const QString& dtype, bool generateMesh) {
    QStringList names = {QStringLiteral("ss_generator-%1.gguf").arg(dtype),
                         QStringLiteral("ss_decoder-%1.gguf").arg(dtype),
                         QStringLiteral("slat_generator-%1.gguf").arg(dtype),
                         QStringLiteral("slat_decoder_gs-%1.gguf").arg(dtype)};
    if (generateMesh) {
        names << QStringLiteral("slat_decoder_mesh-%1.gguf").arg(dtype);
    }
    names << QStringLiteral("moge_vitl-f16.gguf");
    return names;
}

QString sam3dModelCacheDir() {
    const char* cache = aicore_sam3d_model_cache_dir();
    return cache ? QString::fromUtf8(cache) : QString();
}

}  // namespace

Sam3dDialog::Sam3dDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("SAM 3D Objects — Image to 3D (GGML)"));
    setMinimumWidth(560);

    auto* layout = new QVBoxLayout(this);
    auto* form = new QFormLayout;

    // Image source -----------------------------------------------------------
    auto* imageRow = new QHBoxLayout;
    m_imageEdit = new QLineEdit(this);
    m_imageEdit->setPlaceholderText(
            tr("Source image (PNG/JPEG; alpha = mask)"));
    auto* browseImage = new QPushButton(tr("Browse..."), this);
    connect(browseImage, &QPushButton::clicked, this,
            &Sam3dDialog::browseImage);
    imageRow->addWidget(m_imageEdit, 1);
    imageRow->addWidget(browseImage);
    form->addRow(tr("Image:"), imageRow);

    // Sample data (official scene set; downloads on first use) --------------
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Load the official SAM 3D Objects demo scenes.\n"
               "Downloads on first use, then cached locally."));
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &Sam3dDialog::onTestDataClicked);
    form->addRow(QString(), m_testDataBtn);

    // Output directory -------------------------------------------------------
    auto* outRow = new QHBoxLayout;
    m_outputDirEdit = new QLineEdit(this);
    m_outputDirEdit->setPlaceholderText(tr("Gaussian PLY export directory"));
    auto* browseOut = new QPushButton(tr("Browse..."), this);
    connect(browseOut, &QPushButton::clicked, this,
            &Sam3dDialog::browseOutputDir);
    outRow->addWidget(m_outputDirEdit, 1);
    outRow->addWidget(browseOut);
    form->addRow(tr("Output:"), outRow);

    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->addItems({tr("auto"), QStringLiteral("cpu"),
                             QStringLiteral("cuda"), QStringLiteral("vulkan")});
    form->addRow(tr("Device:"), m_deviceCombo);

    m_dtypeCombo = new QComboBox(this);
    m_dtypeCombo->addItems({QStringLiteral("f16"), QStringLiteral("q8_0"),
                            QStringLiteral("q4_k (light)")});
    m_dtypeCombo->setCurrentIndex(2);
    auto* dtypeRow = new QHBoxLayout;
    dtypeRow->addWidget(m_dtypeCombo, 1);
    m_downloadBtn = new QPushButton(tr("Download models"), this);
    m_downloadBtn->setToolTip(tr(
            "Fetch the missing GGUF models for the selected quantization from\n"
            "huggingface.co/Asher-1/SAM_3D_OBJECTS_GGUF into the shared "
            "cache."));
    connect(m_downloadBtn, &QPushButton::clicked, this,
            &Sam3dDialog::onDownloadModels);
    dtypeRow->addWidget(m_downloadBtn);
    form->addRow(tr("Quantization:"), dtypeRow);

    m_stepsSpin = new QSpinBox(this);
    m_stepsSpin->setRange(1, 50);
    m_stepsSpin->setValue(25);
    m_stepsSpin->setToolTip(
            tr("Diffusion steps for the SS and SLat samplers (official gate: "
               "25)"));
    form->addRow(tr("Steps:"), m_stepsSpin);

    m_seedSpin = new QSpinBox(this);
    m_seedSpin->setRange(0, 999999);
    m_seedSpin->setValue(42);
    form->addRow(tr("Seed:"), m_seedSpin);

    m_rmbgCheck = new QCheckBox(
            tr("Remove background (RMBG; builds the object mask)"), this);
    m_rmbgCheck->setChecked(true);
    form->addRow(QString(), m_rmbgCheck);

    m_meshCheck = new QCheckBox(
            tr("Also generate the FlexiCubes mesh (needs mesh decoder)"), this);
    m_meshCheck->setChecked(true);
    form->addRow(QString(), m_meshCheck);

    layout->addLayout(form);

    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 1);
    m_progress->setValue(0);
    layout->addWidget(m_progress);

    m_status = new QLabel(this);
    layout->addWidget(m_status);

    m_log = new QPlainTextEdit(this);
    m_log->setReadOnly(true);
    m_log->setMaximumHeight(160);
    layout->addWidget(m_log);

    auto* buttons = new QHBoxLayout;
    m_runButton = new QPushButton(tr("Generate"), this);
    m_runButton->setDefault(true);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setEnabled(false);
    buttons->addStretch(1);
    buttons->addWidget(m_cancelButton);
    buttons->addWidget(m_runButton);
    layout->addLayout(buttons);

    connect(m_runButton, &QPushButton::clicked, this, &Sam3dDialog::emitRun);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &Sam3dDialog::cancelRequested);

    // Shared model downloader (AICore catalog URLs / sizes / SHA-256).
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0) {
                    m_progress->setRange(0, 100);
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                }
            });
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &Sam3dDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                if (!ok) {
                    m_pendingDownloads.clear();
                    appendLog(tr("[SAM3D] Download failed: %1 — retry, or "
                                 "fetch the file manually from the URL shown "
                                 "in the log and place it at that path.")
                                      .arg(path));
                    return;
                }
                appendLog(tr("[SAM3D] Model downloaded: %1").arg(path));
                downloadNextModel();
            });

    // Sample-data repository (official SAM 3D Objects scene set).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_status->setText(statusText);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            &Sam3dDialog::appendLog);
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset dataset) {
                if (dataset != ecvTestDataRepository::Dataset::Image2Mesh) {
                    return;
                }
                if (success) {
                    appendLog(tr("[SAM3D] Sample data downloaded"));
                    m_status->setText(tr("Extracting sample data..."));
                    if (ecvTestDataRepository::instance().extractDataset(
                                ecvTestDataRepository::Dataset::Image2Mesh)) {
                        return;  // extractionFinished continues the chain
                    }
                }
                m_status->setText(tr("Sample data download failed."));
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset dataset) {
                if (dataset != ecvTestDataRepository::Dataset::Image2Mesh) {
                    return;
                }
                if (success) {
                    appendLog(tr("[SAM3D] Sample data ready"));
                    m_status->clear();
                    populateTestImage();
                } else {
                    m_status->setText(tr("Sample data extraction failed."));
                }
            });

    restoreDefaults();
}

Sam3dDialog::~Sam3dDialog() = default;

void Sam3dDialog::restoreDefaults() {
    // Suggest the shared AICore data root output directory.
    m_outputDirEdit->setText(QDir::homePath());
}

void Sam3dDialog::browseImage() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select the source image"), m_imageEdit->text(),
            tr("Images (*.png *.jpg *.jpeg *.bmp *.webp)"));
    if (!path.isEmpty()) m_imageEdit->setText(path);
}

void Sam3dDialog::browseOutputDir() {
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Output directory"), m_outputDirEdit->text());
    if (!dir.isEmpty()) m_outputDirEdit->setText(dir);
}

Sam3dDialog::Settings Sam3dDialog::settings() const {
    Settings s;
    s.imagePath = m_imageEdit->text().trimmed();
    s.outputDir = m_outputDirEdit->text().trimmed();
    s.device = m_deviceCombo->currentText();
    s.dtypeIndex = m_dtypeCombo->currentIndex();
    s.steps = m_stepsSpin->value();
    s.seed = m_seedSpin->value();
    s.useRmbg = m_rmbgCheck->isChecked();
    s.generateMesh = m_meshCheck->isChecked();
    return s;
}

void Sam3dDialog::appendLog(const QString& message) {
    m_log->appendPlainText(message);
}

void Sam3dDialog::setProgress(int percent) {
    m_progress->setRange(0, 100);
    m_progress->setValue(percent);
}

void Sam3dDialog::setRunning(bool running) {
    m_runButton->setEnabled(!running);
    m_cancelButton->setEnabled(running);
    m_progress->setRange(0, running ? 0 : 1);
    m_progress->setValue(running ? 0 : 1);
    m_status->setText(running ? tr("Running the SAM 3D pipeline...")
                              : QString());
}

void Sam3dDialog::emitRun() {
    const Settings s = settings();
    if (s.imagePath.isEmpty()) {
        m_status->setText(tr("Select a source image first."));
        return;
    }
    if (s.outputDir.isEmpty()) {
        m_status->setText(tr("Select an output directory first."));
        return;
    }
    emit runRequested(s);
}

void Sam3dDialog::populateTestImage() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const QStringList images = ecvTestDataRepository::getSam3dObjectImages(
            ecvTestDataRepository::extractPath(TestDataset::Image2Mesh));
    if (images.isEmpty()) {
        m_status->setText(tr("No scene images found in the sample bundle."));
        return;
    }
    m_imageEdit->setText(images.first());
    appendLog(tr("[SAM3D] %1 official scene(s) available in sam3d_images; "
                 "loaded '%2'. Pick another one with Browse.")
                      .arg(images.size())
                      .arg(QFileInfo(images.first()).fileName()));
}

void Sam3dDialog::onTestDataClicked() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    // The official scene set ships inside the shared Image2Mesh bundle
    // (image_to_mesh_data.zip, sam3d_images/ folder).
    const TestDataset kind = TestDataset::Image2Mesh;

    // 1. Already extracted: fill the image field directly.
    const QStringList images = ecvTestDataRepository::getSam3dObjectImages(
            ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImage();
        return;
    }

    // 2. Zip cached and intact: extract, then populate (signal chain).
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        m_progress->setRange(0, 100);
        m_status->setText(tr("Extracting sample data..."));
        repo.extractDataset(kind);
        return;
    }

    // 3. Nothing cached: download (downloadFinished -> extraction ->
    //    populateTestImage).
    if (repo.isDownloadInProgress()) return;
    m_status->setText(tr("Downloading sample data..."));
    repo.startDownload(kind);
}

void Sam3dDialog::onDownloadModels() {
    const QString cacheDir = sam3dModelCacheDir();
    if (cacheDir.isEmpty()) {
        m_status->setText(tr("SAM 3D model cache is unavailable."));
        return;
    }
    QDir().mkpath(cacheDir);
    static const char* kQuant[] = {"f16", "q8_0", "q4_k"};
    const QString dtype =
            QString::fromLatin1(kQuant[m_dtypeCombo->currentIndex()]);

    m_pendingDownloads.clear();
    for (const QString& name :
         sam3dRequiredModels(dtype, m_meshCheck->isChecked())) {
        const QString path = QDir(cacheDir).filePath(name);
        const aicore_sam3d_model_entry* entry =
                aicore_sam3d_model_by_filename(name.toUtf8().constData());
        if (QFileInfo::exists(path)) {
            // Cheap ledger/stat re-check; a corrupt file re-queues it.
            const ecvAssetIntegrity::Anchor anchor{
                    QCryptographicHash::Sha256,
                    entry ? QByteArray(entry->sha256) : QByteArray()};
            if (ecvAssetIntegrity::isVerified(
                        path, anchor, 1024 * 1024, true,
                        ecvAssetIntegrity::OnMiss::CheapChecksOnly)) {
                continue;
            }
        }
        m_pendingDownloads << name;
    }

    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[SAM3D] Nothing to download (%1 / %2 cache).")
                          .arg(dtype, cacheDir));
        return;
    }
    qint64 totalBytes = 0;
    for (const QString& name : m_pendingDownloads) {
        const aicore_sam3d_model_entry* entry =
                aicore_sam3d_model_by_filename(name.toUtf8().constData());
        if (entry) totalBytes += entry->size_bytes;
    }
    appendLog(tr("[SAM3D] %1 model(s) to download (%2 total).")
                      .arg(m_pendingDownloads.size())
                      .arg(ecvModelDownloader::formatFileSize(totalBytes)));
    m_downloadBtn->setEnabled(false);
    downloadNextModel();
}

void Sam3dDialog::downloadNextModel() {
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[SAM3D] All model downloads finished."));
        m_downloadBtn->setEnabled(true);
        m_progress->setRange(0, 1);
        m_progress->setValue(1);
        return;
    }
    const QString filename = m_pendingDownloads.takeFirst();
    // The AICore catalog supplies the HF URL, size, and SHA-256 for every
    // published model; qSAM3D deliberately owns no duplicate model table.
    const aicore_sam3d_model_entry* entry =
            aicore_sam3d_model_by_filename(filename.toUtf8().constData());
    if (!entry) {
        appendLog(tr("[SAM3D] Model is absent from the AICore catalog: %1")
                          .arg(filename));
        downloadNextModel();
        return;
    }
    const QString url = QString::fromUtf8(entry->download_url);
    const QString dest = QDir(sam3dModelCacheDir()).filePath(filename);
    appendLog(tr("[SAM3D] Downloading %1...\n  URL: %2\n  Dest: %3")
                      .arg(filename, url, dest));
    m_downloadInProgress = true;
    ecvModelDownloader::Request req;
    req.url = url;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;
    req.contentAnchor = {QCryptographicHash::Sha256,
                         QByteArray(entry->sha256)};  // streamed content check
    m_downloader->download(req);
}
