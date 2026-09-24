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

    // Output options (DB import by default; file export is opt-in) ----------
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
    m_rmbgDtypeCombo = new QComboBox(this);
    m_rmbgDtypeCombo->addItem(QStringLiteral("q8_0"));
    m_rmbgDtypeCombo->addItem(QStringLiteral("f16"));
    m_rmbgDtypeCombo->setToolTip(
            tr("RMBG model quantization (rmbg_q8.gguf / rmbg_f16.gguf; the\n"
               "file is shared with the qRMBG plugin's model cache)."));
    auto* rmbgRow = new QHBoxLayout;
    rmbgRow->addWidget(m_rmbgCheck, 1);
    rmbgRow->addWidget(new QLabel(tr("Quantization:"), this));
    rmbgRow->addWidget(m_rmbgDtypeCombo);
    form->addRow(QString(), rmbgRow);

    // Output artifacts = pipeline gating. The colored point cloud is the
    // mandatory generation product; the textured mesh extends the pipeline
    // with the FlexiCubes decode + UV-atlas bake stages.
    m_pointCloudCheck = new QCheckBox(
            tr("Colored gaussian point cloud (generation product)"), this);
    m_pointCloudCheck->setChecked(true);
    m_pointCloudCheck->setToolTip(
            tr("Gaussian splats with their display colors, delivered into\n"
               "the DB tree. This is the direct generation output — the\n"
               "pipeline always produces it."));
    form->addRow(QString(), m_pointCloudCheck);

    m_texturedMeshCheck = new QCheckBox(
            tr("Textured mesh (FlexiCubes decode + UV-atlas bake)"), this);
    m_texturedMeshCheck->setChecked(true);
    m_texturedMeshCheck->setToolTip(
            tr("Extends the pipeline past the point cloud: FlexiCubes mesh\n"
               "decode, then a textured GLB bake (xatlas UV unwrap →\n"
               "per-texel bake → gutter inpaint) through the shared AICore\n"
               "bake API. Uncheck to stop the pipeline at the point cloud\n"
               "and skip both stages (~105 s faster)."));
    form->addRow(QString(), m_texturedMeshCheck);

    m_importDbCheck = new QCheckBox(tr("Import results into the DB"), this);
    m_importDbCheck->setChecked(true);
    form->addRow(QString(), m_importDbCheck);

    m_exportPlyCheck =
            new QCheckBox(tr("Export Gaussian PLY to file (optional)"), this);
    m_exportPlyCheck->setChecked(false);
    form->addRow(QString(), m_exportPlyCheck);

    // Scene mode: per-object binary masks compose a full scene (the official
    // demo_multi_object flow). Masks ship with the sample bundles as
    // <n>.png at the image resolution.
    m_sceneCheck = new QCheckBox(
            tr("Multi-object scene mode (per-object mask directory)"), this);
    m_sceneCheck->setChecked(false);
    m_sceneCheck->setToolTip(
            tr("Reconstruct every <n>.png mask in the directory as an\n"
               "independent single-object run, then compose the objects\n"
               "into one scene with the official make_scene pose semantics\n"
               "(upstream scene-assemble flow). Each object becomes its own\n"
               "textured GLB entity. RMBG is not used in this mode."));
    form->addRow(QString(), m_sceneCheck);
    auto* masksRow = new QHBoxLayout;
    m_masksDirEdit = new QLineEdit(this);
    m_masksDirEdit->setPlaceholderText(
            tr("Directory of per-object masks (0.png, 1.png, ...)"));
    m_browseMasksBtn = new QPushButton(tr("Browse..."), this);
    connect(m_browseMasksBtn, &QPushButton::clicked, this, [this]() {
        const QString dir = QFileDialog::getExistingDirectory(
                this, tr("Mask directory"), m_masksDirEdit->text());
        if (!dir.isEmpty()) m_masksDirEdit->setText(dir);
    });
    masksRow->addWidget(m_masksDirEdit, 1);
    masksRow->addWidget(m_browseMasksBtn);
    form->addRow(tr("Masks:"), masksRow);
    auto updateMasksEnabled = [this]() {
        const bool scene = m_sceneCheck->isChecked();
        m_masksDirEdit->setEnabled(scene);
        m_browseMasksBtn->setEnabled(scene);
    };
    connect(m_sceneCheck, &QCheckBox::toggled, this, updateMasksEnabled);
    updateMasksEnabled();

    // Output directory (only used by the opt-in file exports) --------------
    auto* outRow = new QHBoxLayout;
    m_outputDirEdit = new QLineEdit(this);
    m_outputDirEdit->setPlaceholderText(
            tr("GLB / Gaussian PLY export directory"));
    auto* browseOut = new QPushButton(tr("Browse..."), this);
    connect(browseOut, &QPushButton::clicked, this,
            &Sam3dDialog::browseOutputDir);
    outRow->addWidget(m_outputDirEdit, 1);
    outRow->addWidget(browseOut);
    form->addRow(tr("Output:"), outRow);
    auto updateOutputEnabled = [this, browseOut]() {
        const bool enabled = m_exportPlyCheck->isChecked() ||
                             m_texturedMeshCheck->isChecked();
        m_outputDirEdit->setEnabled(enabled);
        browseOut->setEnabled(enabled);
    };
    connect(m_exportPlyCheck, &QCheckBox::toggled, this, updateOutputEnabled);
    connect(m_texturedMeshCheck, &QCheckBox::toggled, this,
            updateOutputEnabled);
    updateOutputEnabled();

    layout->addLayout(form);

    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 1);
    m_progress->setValue(0);
    layout->addWidget(m_progress);

    m_status = new QLabel(this);
    layout->addWidget(m_status);

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
            [this](const QString& message) { appendLog(message); });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                if (!ok) {
                    m_pendingDownloads.clear();
                    appendLog(tr("[SAM3D] Download failed: %1 — retry, or "
                                 "fetch the file manually from the URL shown "
                                 "in the log and place it at that path.")
                                      .arg(path),
                              ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
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
            [this](const QString& message) { appendLog(message); });
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
    s.masksDir = m_sceneCheck->isChecked() ? m_masksDirEdit->text().trimmed()
                                           : QString();
    s.device = m_deviceCombo->currentText();
    s.dtypeIndex = m_dtypeCombo->currentIndex();
    s.rmbgDtypeIndex = m_rmbgDtypeCombo->currentIndex();
    s.steps = m_stepsSpin->value();
    s.seed = m_seedSpin->value();
    s.useRmbg = m_rmbgCheck->isChecked();
    s.outputPointCloud = m_pointCloudCheck->isChecked();
    s.outputTexturedMesh = m_texturedMeshCheck->isChecked();
    s.importToDb = m_importDbCheck->isChecked();
    s.exportPly = m_exportPlyCheck->isChecked();
    return s;
}

void Sam3dDialog::appendLog(const QString& message,
                            ecvMainAppInterface::ConsoleMessageLevel level) {
    // Reuse the application console (same channel as every other plugin);
    // the dialog itself only keeps the single-line status label.
    if (m_app) {
        m_app->dispToConsole(message, level);
    }
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
    if (!s.outputPointCloud && !s.outputTexturedMesh) {
        m_status->setText(tr("Select at least one output artifact."));
        return;
    }
    if (!s.importToDb && !s.exportPly && !s.outputTexturedMesh) {
        m_status->setText(
                tr("Select an output: DB import and/or a file "
                   "export."));
        return;
    }
    if ((s.exportPly || s.outputTexturedMesh) && s.outputDir.isEmpty()) {
        m_status->setText(
                tr("Select an output directory for the file "
                   "exports."));
        return;
    }
    if (!s.masksDir.isEmpty()) {
        // Scene mode: every object bakes into its own GLB file (released
        // from memory right after the write), so the textured bake needs a
        // writable output directory.
        if (s.outputTexturedMesh && s.outputDir.isEmpty()) {
            m_status->setText(
                    tr("Scene mode with textured mesh requires an "
                       "output directory."));
            return;
        }
        QFileInfoList masks =
                QDir(s.masksDir)
                        .entryInfoList(QStringList{"*.png"}, QDir::Files);
        if (masks.isEmpty()) {
            m_status->setText(tr("The mask directory has no PNG masks."));
            return;
        }
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
         sam3dRequiredModels(dtype, m_texturedMeshCheck->isChecked())) {
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
