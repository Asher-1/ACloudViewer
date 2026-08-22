// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisDialog.h"

#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageReader>
#include <QMessageBox>
#include <QSettings>
#include <QVBoxLayout>

#include "ecvPersistentSettings.h"
#include "aicore/inference_log.h"
#include "ecvAICoreUiHelper.h"

namespace {

const char* kSettingsGroup = "qTrellis";

}  // namespace

TrellisDialog::TrellisDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle(tr("TRELLIS.2 Image to 3D"));
    setupUi();
    loadSettings();
    updateModelStatus();
}

TrellisDialog::~TrellisDialog() = default;

void TrellisDialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
}

void TrellisDialog::setupUi() {
    auto* root = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(root);
    // Lock the dialog size after the first layout pass: DB sections / status
    // text changes must not inflate the window (shared AICore UI spec §13.4).
    root->setSizeConstraint(QLayout::SetNoConstraint);

    // ── Input image ──────────────────────────────────────────────────────
    auto* inputGroup = new QGroupBox(tr("Input image"), this);
    ecvAICoreUi::tightenGroupBox(inputGroup);
    auto* inputLayout = new QGridLayout(inputGroup);
    inputLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    inputLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    m_imagePath = new QLineEdit(inputGroup);
    m_imagePath->setPlaceholderText(tr("PNG / JPEG / WebP image..."));
    m_browseImageBtn = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), inputGroup);
    m_useTestDataBtn = ecvAICoreUi::makeSampleDataBtn(inputGroup);
    m_useTestDataBtn->setToolTip(
            "Load sample images for generation.\n"
            "Downloads on first use, then cached locally.");
    inputLayout->addWidget(m_imagePath, 0, 0, 1, 2);
    inputLayout->addWidget(m_browseImageBtn, 0, 2);
    inputLayout->addWidget(m_useTestDataBtn, 0, 3);
    // Sample-image picker: populated from the Image2Mesh dataset once it is
    // downloaded/extracted; "Browse..." stays available for local files.
    m_testImageCombo = new QComboBox(inputGroup);
    m_testImageCombo->setEnabled(false);
    m_testImageCombo->setToolTip(
            tr("Pick one of the bundled sample images (Image2Mesh dataset)."));
    inputLayout->addWidget(m_testImageCombo, 1, 0, 1, 3);
    m_imagePreview = new ecvClickableImageLabel(inputGroup);
    m_imagePreview->setMinimumSize(ecvAICoreUi::dpiScaled(200),
                                   ecvAICoreUi::dpiScaled(150));
    m_imagePreview->setAlignment(Qt::AlignCenter);
    m_imagePreview->setStyleSheet("border: 1px dashed #999;");
    inputLayout->addWidget(m_imagePreview, 2, 0, 1, 4);
    root->addWidget(inputGroup);

    connect(m_useTestDataBtn, &QPushButton::clicked, this,
            &TrellisDialog::onTestDataClicked);
    connect(m_testImageCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &TrellisDialog::onTestImageSelected);

    // ── Runtime parameters (shared row) ─────────────────────────────────
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->addItems({tr("auto"), tr("cpu"), tr("vulkan"), tr("cuda")});
    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 128);
    m_threads->setValue(0);
    m_threads->setSpecialValueText(tr("default"));
    root->addWidget(ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads));

    // ── Model preset ─────────────────────────────────────────────────────
    auto* modelGroup = new QGroupBox(tr("Model"), this);
    ecvAICoreUi::tightenGroupBox(modelGroup);
    auto* modelLayout = new QGridLayout(modelGroup);
    modelLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    modelLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    modelLayout->addWidget(new QLabel(tr("Quality:"), modelGroup), 0, 0);
    m_presetCombo = new QComboBox(modelGroup);
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    for (const TrellisPreset& p : presets) {
        m_presetCombo->addItem(p.name, p.description);
    }
    m_presetCombo->setCurrentIndex(1);  // Standard 512 + PBR
    modelLayout->addWidget(m_presetCombo, 0, 1, 1, 3);

    modelLayout->addWidget(new QLabel(tr("DINOv3:"), modelGroup), 1, 0);
    m_dinoCombo = new QComboBox(modelGroup);
    m_dinoCombo->addItem(tr("q8 (smaller, recommended)"), QStringLiteral("dino_q8"));
    m_dinoCombo->addItem(tr("f16 (reference)"), QStringLiteral("dino_f16"));
    modelLayout->addWidget(m_dinoCombo, 1, 1);

    modelLayout->addWidget(new QLabel(tr("Occupancy decoder:"), modelGroup), 1, 2);
    m_decCombo = new QComboBox(modelGroup);
    m_decCombo->addItem(tr("f16 (recommended)"), QStringLiteral("ss_dec_f16"));
    m_decCombo->addItem(tr("q8"), QStringLiteral("ss_dec_q8"));
    modelLayout->addWidget(m_decCombo, 1, 3);

    m_textureCheck = new QCheckBox(tr("PBR textures"), modelGroup);
    m_textureCheck->setChecked(true);
    m_textureCheck->setToolTip(tr("Shape encoder + texture decoder + texture "
                                  "flow (~3 GB extra); per-vertex base color / "
                                  "metallic / roughness"));
    modelLayout->addWidget(m_textureCheck, 2, 0, 1, 2);

    m_rmbgCheck = new QCheckBox(tr("AI background removal (RMBG-2.0)"), modelGroup);
    m_rmbgCheck->setChecked(true);
    m_rmbgCheck->setToolTip(tr("Remove the background with the in-tree "
                               "RMBG-2.0 model (rmbg_f16.gguf) before "
                               "generation; falls back to the solid-color "
                               "heuristic when the model is absent."));
    modelLayout->addWidget(m_rmbgCheck, 2, 2, 1, 2);

    m_modelStatus = new QLabel(modelGroup);
    m_modelStatus->setWordWrap(true);
    modelLayout->addWidget(m_modelStatus, 3, 0, 1, 3);
    m_downloadBtn = ecvAICoreUi::makeBrowseBtn(tr("Download missing models..."), modelGroup);
    m_downloadBtn->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    m_downloadBtn->setFixedWidth(ecvAICoreUi::dpiScaled(168));
    modelLayout->addWidget(m_downloadBtn, 3, 3);
    root->addWidget(modelGroup);

    // ── Parameters ───────────────────────────────────────────────────────
    auto* paramGroup = new QGroupBox(tr("Parameters"), this);
    ecvAICoreUi::tightenGroupBox(paramGroup);
    auto* paramLayout = new QGridLayout(paramGroup);
    paramLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    paramLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    paramLayout->addWidget(new QLabel(tr("Steps:"), paramGroup), 0, 0);
    m_steps = new QSpinBox(paramGroup);
    m_steps->setRange(0, 50);
    m_steps->setValue(0);
    m_steps->setSpecialValueText(tr("default (12)"));
    ecvAICoreUi::setCompactSpin(m_steps);
    paramLayout->addWidget(m_steps, 0, 1);

    paramLayout->addWidget(new QLabel(tr("Guidance:"), paramGroup), 0, 2);
    m_guidance = new QDoubleSpinBox(paramGroup);
    m_guidance->setRange(-1.0, 30.0);
    m_guidance->setDecimals(1);
    m_guidance->setSingleStep(0.5);
    m_guidance->setValue(-1.0);
    m_guidance->setSpecialValueText(tr("default (7.5)"));
    ecvAICoreUi::setCompactDoubleSpin(m_guidance);
    paramLayout->addWidget(m_guidance, 0, 3);

    paramLayout->addWidget(new QLabel(tr("Texture steps:"), paramGroup), 1, 0);
    m_textureSteps = new QSpinBox(paramGroup);
    m_textureSteps->setRange(0, 50);
    m_textureSteps->setValue(0);
    m_textureSteps->setSpecialValueText(tr("default (12)"));
    ecvAICoreUi::setCompactSpin(m_textureSteps);
    paramLayout->addWidget(m_textureSteps, 1, 1);

    paramLayout->addWidget(new QLabel(tr("Seed:"), paramGroup), 1, 2);
    m_seed = new QSpinBox(paramGroup);
    m_seed->setRange(0, 999999);
    m_seed->setValue(0);
    m_seed->setSpecialValueText(tr("random"));
    ecvAICoreUi::setCompactSpin(m_seed);
    paramLayout->addWidget(m_seed, 1, 3);
    root->addWidget(paramGroup);

    // ── Output ───────────────────────────────────────────────────────────
    auto* outputGroup = new QGroupBox(tr("Output"), this);
    ecvAICoreUi::tightenGroupBox(outputGroup);
    auto* outputLayout = new QGridLayout(outputGroup);
    outputLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    outputLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    m_addToDbCheck = new QCheckBox(tr("Add mesh to DB"), outputGroup);
    m_addToDbCheck->setChecked(true);
    outputLayout->addWidget(m_addToDbCheck, 0, 0, 1, 2);
    outputLayout->addWidget(new QLabel(tr("Save GLB:"), outputGroup), 1, 0);
    m_saveGlbDir = new QLineEdit(outputGroup);
    m_saveGlbDir->setPlaceholderText(tr("(empty = skip GLB export)"));
    auto* browseGlb = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), outputGroup);
    outputLayout->addWidget(m_saveGlbDir, 1, 1);
    outputLayout->addWidget(browseGlb, 1, 2);
    root->addWidget(outputGroup);

    // ── Progress / log ───────────────────────────────────────────────────
    ecvAICoreUi::setupProgressSection(root, m_stageLabel, m_progress);
    auto* runBtn = new QPushButton(tr("Generate 3D"), this);
    runBtn->setDefault(true);
    auto* cancelBtn = new QPushButton(tr("Cancel"), this);
    cancelBtn->setEnabled(false);
    root->addLayout(ecvAICoreUi::makeActionRow(runBtn, cancelBtn));

    m_log = new QLabel(this);
    m_log->setWordWrap(true);
    m_log->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_log->setMaximumHeight(ecvAICoreUi::dpiScaled(90));
    root->addWidget(m_log);

    connect(m_browseImageBtn, &QPushButton::clicked, this,
            &TrellisDialog::onBrowseImage);
    connect(m_useTestDataBtn, &QPushButton::clicked, this,
            &TrellisDialog::onTestDataClicked);
    connect(m_presetCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &TrellisDialog::onPresetChanged);
    connect(m_dinoCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &TrellisDialog::updateModelStatus);
    connect(m_decCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &TrellisDialog::updateModelStatus);
    connect(m_textureCheck, &QCheckBox::toggled, this,
            &TrellisDialog::updateModelStatus);
    connect(m_downloadBtn, &QPushButton::clicked, this,
            &TrellisDialog::onDownloadModels);
    connect(runBtn, &QPushButton::clicked, this, &TrellisDialog::onRun);
    connect(cancelBtn, &QPushButton::clicked, this, &TrellisDialog::onCancel);
    connect(browseGlb, &QPushButton::clicked, this,
            &TrellisDialog::onBrowseSaveDir);
    connect(m_imagePath, &QLineEdit::textChanged, this,
            &TrellisDialog::updateImagePreview);

    // Downloader.
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                m_progress->setVisible(true);
                if (total > 0) {
                    m_progress->setRange(0, 100);
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                }
            });
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &TrellisDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                if (!ok) {
                    m_pendingDownloads.clear();
                    appendLog(tr("[TRELLIS] Download failed: %1").arg(path));
                    updateModelStatus();
                    return;
                }
                appendLog(tr("[TRELLIS] Model downloaded: %1").arg(path));
                downloadNextModel();
            });

    // Sample-data repository (Image2Mesh dataset).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                m_progress->setVisible(true);
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_stageLabel->setVisible(true);
                m_stageLabel->setText(statusText);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            [this](bool success,
                   ecvTestDataRepository::Dataset dataset) {
                if (dataset != ecvTestDataRepository::Dataset::Image2Mesh) {
                    return;
                }
                if (success) {
                    appendLog(tr("[TRELLIS] Sample data downloaded"));
                    // Fall through to extraction; the extracted() signal
                    // populates the picker.
                    const auto info =
                            ecvTestDataRepository::getDatasetInfo(dataset);
                    if (ecvTestDataRepository::verifyZipIntegrity(
                                ecvTestDataRepository::zipPath(dataset),
                                info.expectedMd5, info.expectedSize)) {
                        m_progress->setRange(0, 100);
                        m_progress->setValue(0);
                        m_stageLabel->setText(tr("Extracting sample data..."));
                        ecvTestDataRepository::instance().extractDataset(
                                dataset);
                    } else {
                        appendLog(tr(
                                "[TRELLIS] Sample data integrity check failed"));
                        m_progress->setVisible(false);
                        m_stageLabel->setVisible(false);
                    }
                } else {
                    appendLog(tr("[TRELLIS] Sample data download failed"));
                    m_progress->setVisible(false);
                    m_stageLabel->setVisible(false);
                }
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success,
                   ecvTestDataRepository::Dataset dataset) {
                if (dataset != ecvTestDataRepository::Dataset::Image2Mesh) {
                    return;
                }
                if (success) {
                    onTestDataExtracted();
                } else {
                    appendLog(tr("[TRELLIS] Sample data extraction failed"));
                    m_progress->setVisible(false);
                    m_stageLabel->setVisible(false);
                }
            });

    // Populate the sample picker when the dataset is already cached.
    ensureImage2MeshDataset();

    resize(ecvAICoreUi::dpiScaled(560), ecvAICoreUi::dpiScaled(640));
}

void TrellisDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(kSettingsGroup);
    m_imagePath->setText(settings.value("inputPath").toString());
    const int presetIdx = settings.value("presetIndex", 1).toInt();
    if (presetIdx >= 0 && presetIdx < m_presetCombo->count()) {
        m_presetCombo->setCurrentIndex(presetIdx);
    }
    m_textureCheck->setChecked(
            settings.value("textureEnabled", true).toBool());
    m_rmbgCheck->setChecked(settings.value("useRmbg", true).toBool());
    m_steps->setValue(settings.value("steps", 0).toInt());
    m_guidance->setValue(settings.value("guidance", -1.0).toDouble());
    m_textureSteps->setValue(settings.value("textureSteps", 0).toInt());
    m_seed->setValue(settings.value("seed", 0).toInt());
    const QString device = settings.value("device", QStringLiteral("auto")).toString();
    const int di = m_deviceCombo->findText(device);
    if (di >= 0) m_deviceCombo->setCurrentIndex(di);
    m_threads->setValue(settings.value("threads", 0).toInt());
    m_addToDbCheck->setChecked(settings.value("addToDb", true).toBool());
    m_saveGlbDir->setText(settings.value("saveGlbDir").toString());
    settings.endGroup();
}

void TrellisDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(kSettingsGroup);
    settings.setValue("inputPath", m_imagePath->text());
    settings.setValue("presetIndex", m_presetCombo->currentIndex());
    settings.setValue("textureEnabled", m_textureCheck->isChecked());
    settings.setValue("useRmbg", m_rmbgCheck->isChecked());
    settings.setValue("steps", m_steps->value());
    settings.setValue("guidance", m_guidance->value());
    settings.setValue("textureSteps", m_textureSteps->value());
    settings.setValue("seed", m_seed->value());
    settings.setValue("device", m_deviceCombo->currentText());
    settings.setValue("threads", m_threads->value());
    settings.setValue("addToDb", m_addToDbCheck->isChecked());
    settings.setValue("saveGlbDir", m_saveGlbDir->text());
    settings.endGroup();
}

void TrellisDialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    QDialog::closeEvent(event);
}

TrellisDialog::Settings TrellisDialog::getSettings() const {
    Settings s;
    s.presetName = m_presetCombo->currentText();
    s.inputPath = m_imagePath->text().trimmed();
    s.steps = m_steps->value();
    s.guidance = m_guidance->value();
    s.textureSteps = m_textureSteps->value();
    s.seed = static_cast<uint64_t>(m_seed->value());
    s.device = m_deviceCombo->currentText();
    s.threads = m_threads->value();
    s.useRmbg = m_rmbgCheck->isChecked();
    s.textureEnabled = m_textureCheck->isChecked();
    s.addResultToDb = m_addToDbCheck->isChecked();
    s.saveGlbDir = m_saveGlbDir->text().trimmed();
    s.pipelineType = m_presetCombo->currentIndex() == 0
                             ? 1 /* coarse */
                             : (m_presetCombo->currentIndex() == 2 ? 3 /* 1024 */
                                                                   : 0 /* auto */);

    // Resolve the absolute model paths for this preset/variant selection.
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx >= 0 && idx < presets.size()) {
        s.modelPaths = TrellisHelpers::resolvePresetFiles(
                presets[idx], TrellisHelpers::modelCacheDir(),
                m_dinoCombo->currentData().toString(),
                m_decCombo->currentData().toString());
    }
    return s;
}

void TrellisDialog::appendLog(const QString& msg) {
    // Mirror into the AICore log so [TRELLIS] messages reach the Console and
    // the on-disk log file (the dialog's own text box alone is invisible to
    // log-file based troubleshooting).
#ifdef AICore_ENABLED
    aicore_inference_log::log(msg);
#endif
    m_log->setText(msg);
}

void TrellisDialog::setProgressStage(const QString& stage, int step, int total) {
    m_stageLabel->setVisible(true);
    if (total > 0) {
        m_stageLabel->setText(tr("Stage: %1 (%2/%3)").arg(stage).arg(step).arg(total));
        m_progress->setVisible(true);
        m_progress->setRange(0, total);
        m_progress->setValue(step);
    } else {
        m_stageLabel->setText(tr("Stage: %1").arg(stage));
        m_progress->setVisible(true);
        m_progress->setRange(0, 1);
        m_progress->setValue(0);
    }
}

void TrellisDialog::setRunning(bool running) {
    m_browseImageBtn->setEnabled(!running);
    m_useTestDataBtn->setEnabled(!running);
    m_downloadBtn->setEnabled(!running);
    m_presetCombo->setEnabled(!running);
    m_dinoCombo->setEnabled(!running);
    m_decCombo->setEnabled(!running);
    m_textureCheck->setEnabled(!running);
    m_rmbgCheck->setEnabled(!running);
    m_steps->setEnabled(!running);
    m_guidance->setEnabled(!running);
    m_textureSteps->setEnabled(!running);
    m_seed->setEnabled(!running);
    m_deviceCombo->setEnabled(!running);
    m_threads->setEnabled(!running);
    m_addToDbCheck->setEnabled(!running);
    m_saveGlbDir->setEnabled(!running);
}

void TrellisDialog::setImagePreview(const QImage& image) {
    if (image.isNull()) {
        m_imagePreview->setPreviewImage(QImage(), ecvAICoreUi::previewSize());
        return;
    }
    // ecvClickableImageLabel keeps the full-resolution copy internally, so
    // clicking the preview opens the original image (shared UI spec §13.3).
    m_imagePreview->setPreviewImage(image, ecvAICoreUi::previewSize());
}

void TrellisDialog::updateImagePreview() {
    const QString path = m_imagePath->text().trimmed();
    if (path.isEmpty()) {
        m_imagePreview->setPreviewImage(QImage(), ecvAICoreUi::previewSize());
        return;
    }
    QImageReader reader(path);
    if (!reader.canRead()) return;
    const QImage img = reader.read();
    if (!img.isNull()) setImagePreview(img);
}

void TrellisDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    if (m_firstShow) {
        // Qt has finished the layout pass by the time Show is sent; with
        // SetNoConstraint this pins the window to its first settled size so
        // later content changes (status text, progress) do not inflate it.
        m_firstShow = false;
        adjustSize();
    }
}

void TrellisDialog::onBrowseImage() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select image"), QDir::homePath(),
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    updateImagePreview();
}

void TrellisDialog::onBrowseSaveDir() {
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select output directory"), QDir::homePath());
    if (!dir.isEmpty()) m_saveGlbDir->setText(dir);
}

void TrellisDialog::onPresetChanged(int index) {
    Q_UNUSED(index);
    updateModelStatus();
}

void TrellisDialog::updateModelStatus() {
    const QStringList missing = missingPresetFiles();
    const QString cacheDir = TrellisHelpers::modelCacheDir();
    if (missing.isEmpty()) {
        m_modelStatus->setText(tr("\u2705 All models present in %1")
                                       .arg(cacheDir));
        m_downloadBtn->setEnabled(false);
    } else {
        m_modelStatus->setText(tr("\u26a0 Missing %1 model(s): %2")
                                       .arg(missing.size())
                                       .arg(missing.join(", ")));
        m_downloadBtn->setEnabled(true);
    }
}

QStringList TrellisDialog::missingPresetFiles() const {
    const QString cacheDir = TrellisHelpers::modelCacheDir();
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return {};
    const QString dinoVariant = m_dinoCombo->currentData().toString();
    const QString decVariant = m_decCombo->currentData().toString();
    QStringList needed = TrellisHelpers::resolvePresetFiles(
            presets[idx], cacheDir, dinoVariant, decVariant);
    if (m_textureCheck->isChecked() && idx == 0) {
        // Texture requires a fine preset; ignore for coarse.
    }
    QStringList missing;
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;  // preset placeholder (model not in set)
        if (!QFile::exists(p)) missing << QFileInfo(p).fileName();
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbg = cacheDir + QLatin1Char('/') +
                             QStringLiteral("rmbg_f16.gguf");
        if (!QFile::exists(rmbg)) missing << QStringLiteral("rmbg_f16.gguf");
    }
    return missing;
}

void TrellisDialog::onDownloadModels() {
    const QString cacheDir = TrellisHelpers::modelCacheDir();
    QDir().mkpath(cacheDir);
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return;
    const QString dinoVariant = m_dinoCombo->currentData().toString();
    const QString decVariant = m_decCombo->currentData().toString();
    const QStringList needed = TrellisHelpers::resolvePresetFiles(
            presets[idx], cacheDir, dinoVariant, decVariant);

    m_pendingDownloads.clear();
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;  // preset placeholder (model not in set)
        if (!QFile::exists(p)) {
            m_pendingDownloads << QFileInfo(p).fileName();
        }
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbg = QStringLiteral("rmbg_f16.gguf");
        if (!QFile::exists(cacheDir + QLatin1Char('/') + rmbg)) {
            m_pendingDownloads << rmbg;
        }
    }
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[TRELLIS] Nothing to download."));
        updateModelStatus();
        return;
    }
    appendLog(tr("[TRELLIS] %1 model(s) to download (~%2 GB total).")
                      .arg(m_pendingDownloads.size())
                      .arg(m_pendingDownloads.size() > 6 ? 9 : 5));
    m_downloadInProgress = false;
    downloadNextModel();
}

void TrellisDialog::downloadNextModel() {
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[TRELLIS] All model downloads finished."));
        updateModelStatus();
        if (m_pendingActionAfterDownload == PendingAction::Run) {
            m_pendingActionAfterDownload = PendingAction::None;
            onRun();
        }
        return;
    }
    const QString filename = m_pendingDownloads.takeFirst();
    TrellisModelEntry entry;
    if (!TrellisHelpers::findModelByFilename(filename, &entry)) {
        appendLog(tr("[TRELLIS] Unknown model in catalog: %1").arg(filename));
        downloadNextModel();
        return;
    }
    const QString dest = TrellisHelpers::modelCacheDir() +
                         QDir::separator() + entry.filename;
    appendLog(tr("[TRELLIS] Downloading %1...").arg(entry.filename));
    m_downloadInProgress = true;
    ecvModelDownloader::Request req;
    req.url = entry.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;
    m_downloader->download(req);
}

void TrellisDialog::onTestDataClicked() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::Image2Mesh;

    // 1. Already extracted: fill the picker and select the first image.
    const QStringList images =
            ecvTestDataRepository::getImage2MeshImages(
                    ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImages();
        return;
    }

    // 2. Zip cached and intact: extract, then populate (signal chain).
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvTestDataRepository::verifyZipIntegrity(
                ecvTestDataRepository::zipPath(kind), info.expectedMd5,
                info.expectedSize)) {
        m_progress->setRange(0, 100);
        m_progress->setVisible(true);
        m_stageLabel->setVisible(true);
        m_stageLabel->setText(tr("Extracting sample data..."));
        repo.extractDataset(kind);
        return;
    }

    // 3. Nothing cached: download (downloadFinished -> extraction -> populate).
    m_progress->setRange(0, 100);
    m_progress->setVisible(true);
    m_stageLabel->setVisible(true);
    m_stageLabel->setText(tr("Downloading sample data..."));
    repo.startDownload(kind);
}

void TrellisDialog::ensureImage2MeshDataset() {
    // Populate lazily if the dataset is already extracted (e.g. on dialog
    // open after a previous run); otherwise leave the picker disabled until
    // the user clicks "Try sample data".
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    const QStringList images =
            ecvTestDataRepository::getImage2MeshImages(
                    ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImages();
    }
}

void TrellisDialog::populateTestImages() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    const QStringList images =
            ecvTestDataRepository::getImage2MeshImages(
                    ecvTestDataRepository::extractPath(kind));
    if (images.isEmpty()) {
        m_testImageCombo->clear();
        m_testImageCombo->setEnabled(false);
        return;
    }
    m_testImageCombo->blockSignals(true);
    m_testImageCombo->clear();
    for (const QString& path : images) {
        m_testImageCombo->addItem(QFileInfo(path).fileName(), path);
    }
    m_testImageCombo->blockSignals(false);
    m_testImageCombo->setEnabled(true);
    m_testImageCombo->setCurrentIndex(0);
    onTestImageSelected(0);
    appendLog(tr("[TRELLIS] %1 sample image(s) ready — pick one above.")
                      .arg(images.size()));
}

void TrellisDialog::onTestImageSelected(int index) {
    if (index < 0 || !m_testImageCombo) return;
    const QString path = m_testImageCombo->itemData(index).toString();
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    updateImagePreview();
}

void TrellisDialog::onTestDataExtracted() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    m_progress->setVisible(false);
    m_stageLabel->setVisible(false);
    if (!ecvTestDataRepository::getImage2MeshImages(
                 ecvTestDataRepository::extractPath(kind))
                 .isEmpty()) {
        populateTestImages();
    } else {
        appendLog(tr("[TRELLIS] Sample data extracted, but no images found."));
    }
}

void TrellisDialog::onRun() {
    if (m_imagePath->text().trimmed().isEmpty()) {
        appendLog(tr("[TRELLIS] Select an input image first."));
        return;
    }
    // Ensure every required model is present; download what is missing.
    const QStringList missing = missingPresetFiles();
    if (!missing.isEmpty()) {
        appendLog(tr("[TRELLIS] Missing models, downloading: %1")
                          .arg(missing.join(", ")));
        m_pendingActionAfterDownload = PendingAction::Run;
        onDownloadModels();
        return;
    }
    saveSettings();
    emit runRequested(getSettings());
}

void TrellisDialog::onCancel() { emit cancelRequested(); }

void TrellisDialog::refreshModelState() { updateModelStatus(); }
