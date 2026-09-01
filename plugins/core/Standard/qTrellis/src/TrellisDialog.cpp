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
#include <QPixmap>
#include <QSettings>
#include <QStandardPaths>
#include <QVBoxLayout>

#ifdef AICore_ENABLED
#include "aicore/inference_log.h"
#include "aicore/trellis_capi.h"
#endif
#include "ecvAICoreUiHelper.h"
#include "ecvPersistentSettings.h"

namespace {

const char* kSettingsGroup = "qTrellis";

// Backend defaults shown while a parameter row is in auto mode (the C-API
// sentinels 0 / -1.0 / 0 mean "use the backend default").
constexpr int kDefaultSteps = 12;
constexpr double kDefaultGuidance = 7.5;
constexpr int kDefaultTextureSteps = 12;

}  // namespace

TrellisDialog::TrellisDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("TRELLIS.2 Image to 3D"));
    setupUi();
    loadSettings();
    updateModelStatus();
}

TrellisDialog::~TrellisDialog() = default;

void TrellisDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void TrellisDialog::setupUi() {
    // qYOLO-style two-page shell: a left-hand page list driving a stacked
    // content area, so the generate flow and the export/print tools each get
    // a dedicated, uncluttered page.
    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);

    m_pageList = new QListWidget(this);
    m_pageList->setFixedWidth(ecvAICoreUi::dpiScaled(132));
    m_pageList->setSpacing(ecvAICoreUi::dpiScaled(2));
    m_pageList->addItem(tr("Generate"));
    m_pageList->addItem(tr("Export / Print"));
    m_pageList->setCurrentRow(0);

    m_pageStack = new QStackedWidget(this);
    auto* generatePage = new QWidget(m_pageStack);
    auto* exportPage = new QWidget(m_pageStack);
    buildGeneratePage(generatePage);
    buildExportPage(exportPage);
    m_pageStack->addWidget(generatePage);
    m_pageStack->addWidget(exportPage);

    root->addWidget(m_pageList);
    root->addWidget(m_pageStack, 1);

    connect(m_pageList, &QListWidget::currentRowChanged, this,
            &TrellisDialog::onPageChanged);

    resize(ecvAICoreUi::dpiScaled(640), ecvAICoreUi::dpiScaled(680));
}

void TrellisDialog::onPageChanged(int row) {
    if (m_pageStack) m_pageStack->setCurrentIndex(row);
}

void TrellisDialog::buildGeneratePage(QWidget* page) {
    auto* root = new QVBoxLayout(page);
    ecvAICoreUi::setupTabLayout(root);

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
    m_deviceCombo->setToolTip(
            tr("Inference backend. 'auto' picks the best device whose free "
               "VRAM fits this preset (CUDA \u2192 Vulkan \u2192 CPU); a "
               "low-VRAM card falls back gracefully instead of failing."));
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

    modelLayout->addWidget(new QLabel(tr("Quantization:"), modelGroup), 1, 0);
    m_quantCombo = new QComboBox(modelGroup);
    // q8 is the default weight-precision chain: every model that publishes a
    // q8 variant uses it, halving the VRAM/RAM footprint (the
    // precision-sensitive decoders shape_dec / shape_enc / tex_dec have no
    // q8 variant and always stay f16). f16 is the full reference chain.
    m_quantCombo->addItem(tr("q8 (recommended, low VRAM)"),
                          QStringLiteral("q8"));
    m_quantCombo->addItem(tr("f16 (reference)"), QStringLiteral("f16"));
    // The exact chain upgrades dino, flows, ss_dec, and shape_dec to the
    // published f32 weights. Texture-only files remain on f16.
    m_quantCombo->addItem(tr("f32 (exact reference)"), QStringLiteral("f32"));
    m_quantCombo->setToolTip(
            tr("Weight precision of the whole chain. q8 halves the memory "
               "footprint (fits small GPUs); the sensitive decoders always "
               "stay f16. Use f16 for the numerically exact reference run."));
    modelLayout->addWidget(m_quantCombo, 1, 1, 1, 3);

    m_textureCheck = new QCheckBox(tr("PBR textures"), modelGroup);
    m_textureCheck->setChecked(true);
    m_textureCheck->setToolTip(
            tr("Shape encoder + texture decoder + texture "
               "flow (~3 GB extra); per-vertex base color / "
               "metallic / roughness"));
    modelLayout->addWidget(m_textureCheck, 2, 0, 1, 2);

    m_rmbgCheck =
            new QCheckBox(tr("AI background removal (RMBG-2.0)"), modelGroup);
    m_rmbgCheck->setChecked(true);
    m_rmbgCheck->setToolTip(
            tr("Remove the background with the in-tree "
               "RMBG-2.0 model (rmbg_f16.gguf) before "
               "generation; falls back to the solid-color "
               "heuristic when the model is absent."));
    modelLayout->addWidget(m_rmbgCheck, 2, 2, 1, 2);

    m_modelStatus = new QLabel(modelGroup);
    m_modelStatus->setWordWrap(true);
    modelLayout->addWidget(m_modelStatus, 3, 0, 1, 3);
    m_downloadBtn = ecvAICoreUi::makeBrowseBtn(tr("Download missing models..."),
                                               modelGroup);
    // The fixed 168 px clips the long label on 96-dpi screens; let the
    // button size itself to the text, keeping 168 px as a floor.
    m_downloadBtn->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    m_downloadBtn->setMinimumWidth(ecvAICoreUi::dpiScaled(168));
    m_downloadBtn->setMaximumWidth(QWIDGETSIZE_MAX);
    modelLayout->addWidget(m_downloadBtn, 3, 3);
    root->addWidget(modelGroup);

    // ── Parameters ───────────────────────────────────────────────────────
    // Each parameter is one "auto / manual" row: while the checkbox is
    // checked the spinbox is disabled and shows the backend default, so the
    // user always sees what value will be used; unchecking unlocks manual
    // entry.  Auto rows map to the C-API sentinels (0 / -1.0 / 0) in
    // getSettings(), keeping the inference contract unchanged.
    auto* paramGroup = new QGroupBox(tr("Parameters"), this);
    ecvAICoreUi::tightenGroupBox(paramGroup);
    auto* paramLayout = new QGridLayout(paramGroup);
    paramLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    paramLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    paramLayout->setColumnStretch(0, 1);

    m_stepsAuto = new QCheckBox(tr("Steps (default)"), paramGroup);
    m_stepsAuto->setChecked(true);
    m_stepsAuto->setToolTip(
            tr("Flow-matching steps for the shape (default 12). "
               "More steps = finer detail, slower generation."));
    m_steps = new QSpinBox(paramGroup);
    m_steps->setRange(1, 50);
    m_steps->setValue(kDefaultSteps);
    m_steps->setEnabled(false);
    ecvAICoreUi::setCompactSpin(m_steps);
    paramLayout->addWidget(m_stepsAuto, 0, 0);
    paramLayout->addWidget(m_steps, 0, 1, Qt::AlignLeft);
    connect(m_stepsAuto, &QCheckBox::toggled, this, [this](bool autoOn) {
        m_steps->setEnabled(!autoOn);
        if (autoOn) m_steps->setValue(kDefaultSteps);
    });

    m_guidanceAuto = new QCheckBox(tr("Guidance (default)"), paramGroup);
    m_guidanceAuto->setChecked(true);
    m_guidanceAuto->setToolTip(
            tr("Classifier-free guidance scale (default 7.5). "
               "Higher = output follows the input image more closely."));
    m_guidance = new QDoubleSpinBox(paramGroup);
    m_guidance->setRange(0.0, 30.0);
    m_guidance->setDecimals(1);
    m_guidance->setSingleStep(0.5);
    m_guidance->setValue(kDefaultGuidance);
    m_guidance->setEnabled(false);
    ecvAICoreUi::setCompactDoubleSpin(m_guidance);
    paramLayout->addWidget(m_guidanceAuto, 1, 0);
    paramLayout->addWidget(m_guidance, 1, 1, Qt::AlignLeft);
    connect(m_guidanceAuto, &QCheckBox::toggled, this, [this](bool autoOn) {
        m_guidance->setEnabled(!autoOn);
        if (autoOn) m_guidance->setValue(kDefaultGuidance);
    });

    m_textureStepsAuto =
            new QCheckBox(tr("Texture steps (default)"), paramGroup);
    m_textureStepsAuto->setChecked(true);
    m_textureStepsAuto->setToolTip(
            tr("Flow-matching steps for the PBR texture (default 12). "
               "More steps = finer texture, slower generation."));
    m_textureSteps = new QSpinBox(paramGroup);
    m_textureSteps->setRange(1, 50);
    m_textureSteps->setValue(kDefaultTextureSteps);
    m_textureSteps->setEnabled(false);
    ecvAICoreUi::setCompactSpin(m_textureSteps);
    paramLayout->addWidget(m_textureStepsAuto, 2, 0);
    paramLayout->addWidget(m_textureSteps, 2, 1, Qt::AlignLeft);
    connect(m_textureStepsAuto, &QCheckBox::toggled, this, [this](bool autoOn) {
        m_textureSteps->setEnabled(!autoOn);
        if (autoOn) m_textureSteps->setValue(kDefaultTextureSteps);
    });

    m_seedRandom = new QCheckBox(tr("Random seed"), paramGroup);
    m_seedRandom->setChecked(true);
    m_seedRandom->setToolTip(
            tr("Use a random seed for the RNG. Uncheck to enter a fixed "
               "seed and reproduce a previous run."));
    m_seed = new QSpinBox(paramGroup);
    m_seed->setRange(0, 999999);
    m_seed->setValue(0);
    m_seed->setSpecialValueText(tr("random"));
    m_seed->setEnabled(false);
    ecvAICoreUi::setCompactSpin(m_seed);
    // "random" / "999999" need more room than the shared compact width.
    m_seed->setFixedWidth(ecvAICoreUi::dpiScaled(96));
    paramLayout->addWidget(m_seedRandom, 3, 0);
    paramLayout->addWidget(m_seed, 3, 1, Qt::AlignLeft);
    connect(m_seedRandom, &QCheckBox::toggled, this, [this](bool randomOn) {
        m_seed->setEnabled(!randomOn);
        if (randomOn) {
            m_seed->setValue(0);  // 0 renders as "random"
        } else if (m_seed->value() == 0) {
            m_seed->setValue(1);
        }
    });
    root->addWidget(paramGroup);

    // End-to-end step strip: one chip per core pipeline step, each showing a
    // live thumbnail (source image / RMBG+preprocess / SS voxel set / mesh
    // keyframe / PBR texture / GLB file) so the user can watch every stage's
    // effect without leaving the dialog.
    auto* stripGroup = new QGroupBox(tr("Pipeline steps"), this);
    ecvAICoreUi::tightenGroupBox(stripGroup);
    auto* stripLayout = new QHBoxLayout(stripGroup);
    stripLayout->setSpacing(ecvAICoreUi::dpiScaled(6));
    const QStringList kStepNames = {tr("Source"),  tr("Preprocess"),
                                    tr("Voxels"),  tr("Mesh"),
                                    tr("Texture"), tr("GLB")};
    const int thumb = ecvAICoreUi::dpiScaled(72);
    for (const QString& name : kStepNames) {
        auto* cell = new QVBoxLayout();
        cell->setSpacing(1);
        auto* thumbLabel = new QLabel(stripGroup);
        thumbLabel->setFixedSize(thumb, thumb);
        thumbLabel->setAlignment(Qt::AlignCenter);
        thumbLabel->setStyleSheet(
                "border: 1px solid #B8C4D0; border-radius: 3px; "
                "background: #F4F7FA; color: #98A4B0;");
        thumbLabel->setText(QStringLiteral("-"));
        auto* caption = new QLabel(name, stripGroup);
        caption->setAlignment(Qt::AlignCenter);
        caption->setStyleSheet("color: #5A6672; font-size: 10px;");
        cell->addWidget(thumbLabel, 0, Qt::AlignHCenter);
        cell->addWidget(caption, 0, Qt::AlignHCenter);
        stripLayout->addLayout(cell);
        m_stageThumbs << thumbLabel;
        m_stageCaptions << caption;
    }
    stripLayout->addStretch(1);
    root->addWidget(stripGroup);

    // ── Output ───────────────────────────────────────────────────────────
    auto* outputGroup = new QGroupBox(tr("Output"), this);
    ecvAICoreUi::tightenGroupBox(outputGroup);
    auto* outputLayout = new QGridLayout(outputGroup);
    outputLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    outputLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    m_addToDbCheck = new QCheckBox(tr("Add mesh to DB"), outputGroup);
    m_addToDbCheck->setChecked(true);
    outputLayout->addWidget(m_addToDbCheck, 0, 0, 1, 2);
    m_addRmbgToDbCheck = new QCheckBox(tr("Add RMBG image to DB"), outputGroup);
    m_addRmbgToDbCheck->setChecked(false);
    m_addRmbgToDbCheck->setToolTip(
            tr("Add the AI background-removed image as a ccImage entity to "
               "the DB tree. Requires AI background removal above."));
    outputLayout->addWidget(m_addRmbgToDbCheck, 1, 0, 1, 2);
    outputLayout->addWidget(new QLabel(tr("Save GLB:"), outputGroup), 2, 0);
    m_saveGlbDir = new QLineEdit(outputGroup);
    m_saveGlbDir->setPlaceholderText(tr("(empty = skip GLB export)"));
    auto* browseGlb = ecvAICoreUi::makeBrowseBtn(tr("Browse..."), outputGroup);
    outputLayout->addWidget(m_saveGlbDir, 2, 1);
    outputLayout->addWidget(browseGlb, 2, 2);
    root->addWidget(outputGroup);

    // ── Progress / log ───────────────────────────────────────────────────
    ecvAICoreUi::setupProgressSection(root, m_stageLabel, m_progress);
    auto* runBtn = new QPushButton(tr("Generate 3D"), this);
    runBtn->setDefault(true);
    // Keep a floor width so the label never clips on narrow windows or
    // translated strings.
    runBtn->setMinimumWidth(ecvAICoreUi::dpiScaled(120));
    // One-click end-to-end: generate with the current settings AND write the
    // textured GLB into the Save-GLB directory (defaulted to ~/Downloads /
    // TRELLIS when empty), so the whole image -> portable-asset flow is one
    // button for non-expert users.
    auto* oneClickBtn = new QPushButton(tr("Generate + GLB"), this);
    oneClickBtn->setToolTip(
            tr("One-click end-to-end: run generation and save the textured "
               "GLB to the Save GLB directory (defaults to Downloads/"
               "TRELLIS)."));
    oneClickBtn->setMinimumWidth(ecvAICoreUi::dpiScaled(120));
    auto* cancelBtn = new QPushButton(tr("Cancel"), this);
    cancelBtn->setEnabled(false);
    root->addLayout(ecvAICoreUi::makeActionRow(runBtn, oneClickBtn, cancelBtn));
    connect(oneClickBtn, &QPushButton::clicked, this,
            &TrellisDialog::onRunOneClick);

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
    connect(m_quantCombo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &TrellisDialog::onQuantChanged);
    connect(m_textureCheck, &QCheckBox::toggled, this,
            &TrellisDialog::updateModelStatus);
    // The RMBG-image output depends on the AI matting actually running; keep
    // it disabled while the RMBG switch is off.
    connect(m_rmbgCheck, &QCheckBox::toggled, this,
            [this](bool on) { m_addRmbgToDbCheck->setEnabled(on); });
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
            [this](bool success, ecvTestDataRepository::Dataset dataset) {
                if (dataset != ecvTestDataRepository::Dataset::Image2Mesh) {
                    return;
                }
                if (success) {
                    appendLog(tr("[TRELLIS] Sample data downloaded"));
                    // Fall through to extraction; the extracted() signal
                    // populates the picker.
                    const auto info =
                            ecvTestDataRepository::getDatasetInfo(dataset);
                    if (ecvAssetIntegrity::isVerified(
                                ecvTestDataRepository::zipPath(dataset),
                                info.anchor, 0, false,
                                ecvAssetIntegrity::OnMiss::DeepVerify)) {
                        m_progress->setRange(0, 100);
                        m_progress->setValue(0);
                        m_stageLabel->setText(tr("Extracting sample data..."));
                        ecvTestDataRepository::instance().extractDataset(
                                dataset);
                    } else {
                        appendLog(
                                tr("[TRELLIS] Sample data integrity check "
                                   "failed"));
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
            [this](bool success, ecvTestDataRepository::Dataset dataset) {
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
}

void TrellisDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(kSettingsGroup);
    m_imagePath->setText(settings.value("inputPath").toString());
    const int presetIdx = settings.value("presetIndex", 1).toInt();
    if (presetIdx >= 0 && presetIdx < m_presetCombo->count()) {
        m_presetCombo->setCurrentIndex(presetIdx);
    }
    m_textureCheck->setChecked(settings.value("textureEnabled", true).toBool());
    m_rmbgCheck->setChecked(settings.value("useRmbg", true).toBool());
    // Restore after useRmbg so the toggled-handler keeps the output checkbox
    // enabled only when AI matting is on.
    m_addRmbgToDbCheck->setChecked(
            settings.value("addRmbgImageToDb", false).toBool());
    // 0 / -1.0 / 0 stored by saveSettings (and by earlier versions) mean
    // "use the backend default"; re-arm the matching auto checkbox.
    const int steps = settings.value("steps", 0).toInt();
    m_stepsAuto->setChecked(steps == 0);
    m_steps->setValue(steps == 0 ? kDefaultSteps : steps);
    m_steps->setEnabled(steps != 0);
    const double guidance = settings.value("guidance", -1.0).toDouble();
    m_guidanceAuto->setChecked(guidance < 0.0);
    m_guidance->setValue(guidance < 0.0 ? kDefaultGuidance : guidance);
    m_guidance->setEnabled(guidance >= 0.0);
    const int textureSteps = settings.value("textureSteps", 0).toInt();
    m_textureStepsAuto->setChecked(textureSteps == 0);
    m_textureSteps->setValue(textureSteps == 0 ? kDefaultTextureSteps
                                               : textureSteps);
    m_textureSteps->setEnabled(textureSteps != 0);
    const int seed = settings.value("seed", 0).toInt();
    m_seedRandom->setChecked(seed == 0);
    m_seed->setValue(seed);
    m_seed->setEnabled(seed != 0);
    const QString device =
            settings.value("device", QStringLiteral("auto")).toString();
    const int di = m_deviceCombo->findText(device);
    if (di >= 0) m_deviceCombo->setCurrentIndex(di);
    m_threads->setValue(settings.value("threads", 0).toInt());
    m_addToDbCheck->setChecked(settings.value("addToDb", true).toBool());
    m_saveGlbDir->setText(settings.value("saveGlbDir").toString());
    const QString quant =
            settings.value("quantization", QStringLiteral("q8")).toString();
    const int qi = m_quantCombo->findData(quant);
    if (qi >= 0) m_quantCombo->setCurrentIndex(qi);
    m_exportTextureSize->setValue(
            settings.value("exportTextureSize", 2048).toInt());
    m_exportComponentFilter->setCurrentIndex(
            settings.value("exportComponentFilter", 0).toInt());
    settings.endGroup();
}

void TrellisDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(kSettingsGroup);
    settings.setValue("inputPath", m_imagePath->text());
    settings.setValue("presetIndex", m_presetCombo->currentIndex());
    settings.setValue("textureEnabled", m_textureCheck->isChecked());
    settings.setValue("useRmbg", m_rmbgCheck->isChecked());
    settings.setValue("addRmbgImageToDb", m_addRmbgToDbCheck->isChecked());
    // Auto rows persist the C-API sentinels (0 / -1.0 / 0) so stored values
    // keep meaning "backend default" across versions.
    settings.setValue("steps", m_stepsAuto->isChecked() ? 0 : m_steps->value());
    settings.setValue("guidance",
                      m_guidanceAuto->isChecked() ? -1.0 : m_guidance->value());
    settings.setValue("textureSteps", m_textureStepsAuto->isChecked()
                                              ? 0
                                              : m_textureSteps->value());
    settings.setValue("seed", m_seedRandom->isChecked() ? 0 : m_seed->value());
    settings.setValue("device", m_deviceCombo->currentText());
    settings.setValue("threads", m_threads->value());
    settings.setValue("addToDb", m_addToDbCheck->isChecked());
    settings.setValue("saveGlbDir", m_saveGlbDir->text());
    settings.setValue("quantization", quantization());
    settings.setValue("exportTextureSize", m_exportTextureSize->value());
    settings.setValue("exportComponentFilter",
                      m_exportComponentFilter->currentIndex());
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
    // Auto rows map to the backend default sentinels (0 / -1.0 / 0).
    s.steps = m_stepsAuto->isChecked() ? 0 : m_steps->value();
    s.guidance = m_guidanceAuto->isChecked() ? -1.0 : m_guidance->value();
    s.textureSteps =
            m_textureStepsAuto->isChecked() ? 0 : m_textureSteps->value();
    s.seed = m_seedRandom->isChecked() ? 0
                                       : static_cast<uint64_t>(m_seed->value());
    s.device = m_deviceCombo->currentText();
    s.threads = m_threads->value();
    s.useRmbg = m_rmbgCheck->isChecked();
    s.textureEnabled = m_textureCheck->isChecked();
    s.addResultToDb = m_addToDbCheck->isChecked();
    s.addRmbgImageToDb = m_addRmbgToDbCheck->isChecked();
    s.saveGlbDir = m_saveGlbDir->text().trimmed();
    s.pipelineType =
            m_presetCombo->currentIndex() == 0
                    ? 1                                       /* coarse */
                    : (m_presetCombo->currentIndex() == 2 ? 3 /* 1024 */
                                                          : 0 /* auto */);

    // Resolve the absolute model paths for this preset/variant selection.
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx >= 0 && idx < presets.size()) {
        s.quantization = quantization();
        s.modelPaths = TrellisHelpers::resolvePresetFiles(
                presets[idx], TrellisHelpers::modelCacheDir(), s.quantization);
    }
    return s;
}

void TrellisDialog::onQuantChanged(int index) {
    Q_UNUSED(index);
    updateModelStatus();
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

void TrellisDialog::setProgressStage(int stageId,
                                     const QString& stage,
                                     int step,
                                     int total) {
    m_stageLabel->setVisible(true);
    m_progress->setVisible(true);
    // Map the 11 pipeline stages onto a single 0..100 sweep so the bar
    // advances monotonically with the real inference work (the C API reports
    // per-stage step counts, which alone would make the bar jump back to 0
    // between stages and look frozen on long stages).
    static const int kRange[11][2] = {
            {0, 2},    // PREPROCESS
            {2, 6},    // DINO
            {6, 26},   // SS_FLOW
            {26, 29},  // SS_DEC
            {29, 49},  // SLAT_FLOW
            {49, 52},  // SHAPE_DEC
            {52, 55},  // MESH
            {55, 56},  // UPSAMPLE
            {56, 57},  // SLAT_FLOW_HR
            {57, 60},  // SHAPE_DEC_HR
            {60, 100}  // TEXTURE
    };
    int pct = 0;
    if (stageId >= 0 && stageId < 11) {
        const int lo = kRange[stageId][0];
        const int hi = kRange[stageId][1];
        pct = (total > 0) ? lo + (hi - lo) * step / total : lo;
        if (pct > hi) pct = hi;
    }
    m_progress->setRange(0, 100);
    m_progress->setValue(pct);
    if (total > 0) {
        m_stageLabel->setText(
                tr("Stage: %1 (%2/%3)").arg(stage).arg(step).arg(total));
    } else {
        m_stageLabel->setText(tr("Stage: %1").arg(stage));
    }
}

void TrellisDialog::setRunning(bool running) {
    m_browseImageBtn->setEnabled(!running);
    m_useTestDataBtn->setEnabled(!running);
    m_downloadBtn->setEnabled(!running);
    m_presetCombo->setEnabled(!running);
    m_quantCombo->setEnabled(!running);
    m_textureCheck->setEnabled(!running);
    m_rmbgCheck->setEnabled(!running);
    // Parameter rows: the checkboxes toggle with running state, the
    // spinboxes stay disabled while their row is in auto mode.
    m_stepsAuto->setEnabled(!running);
    m_steps->setEnabled(!running && !m_stepsAuto->isChecked());
    m_guidanceAuto->setEnabled(!running);
    m_guidance->setEnabled(!running && !m_guidanceAuto->isChecked());
    m_textureStepsAuto->setEnabled(!running);
    m_textureSteps->setEnabled(!running && !m_textureStepsAuto->isChecked());
    m_seedRandom->setEnabled(!running);
    m_seed->setEnabled(!running && !m_seedRandom->isChecked());
    m_deviceCombo->setEnabled(!running);
    m_threads->setEnabled(!running);
    m_addToDbCheck->setEnabled(!running);
    m_addRmbgToDbCheck->setEnabled(!running && m_rmbgCheck->isChecked());
    m_saveGlbDir->setEnabled(!running);
    if (running) {
        // Reset the inference sweep so the first stage callback is visible
        // instead of leaving the bar at a stale download/previous-run value.
        m_progress->setVisible(true);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
        m_stageLabel->setVisible(true);
        m_stageLabel->setText(tr("Stage: starting..."));
    }
}

void TrellisDialog::setImagePreview(const QImage& image) {
    if (image.isNull()) {
        m_imagePreview->setPreviewImage(QImage(), ecvAICoreUi::previewSize());
        return;
    }
    // ecvClickableImageLabel keeps the full-resolution copy internally, so
    // clicking the preview opens the original image (shared UI spec §13.3).
    m_lastPreviewImage = image;
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
    if (missing.isEmpty()) {
        m_modelStatus->setText(tr("\u2705 All models present."));
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
    const QString rmbgCacheDir = TrellisHelpers::rmbgModelCacheDir();
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return {};
    QStringList needed = TrellisHelpers::resolvePresetFiles(
            presets[idx], cacheDir, quantization());
    if (m_textureCheck->isChecked() && idx == 0) {
        // Texture requires a fine preset; ignore for coarse.
    }
    QStringList missing;
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;  // preset placeholder (model not in set)
        // Validate the deployed file, not just its existence: a truncated,
        // stale, or wrong-size GGUF (e.g. a failed manual HF download) must
        // be treated as missing and re-fetched.
        const QString name = QFileInfo(p).fileName();
        if (!TrellisHelpers::isValidModelFile(p, name)) missing << name;
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbgName = TrellisHelpers::isValidModelFile(
                                         rmbgCacheDir + QLatin1Char('/') +
                                                 QStringLiteral("rmbg_q8.gguf"),
                                         QStringLiteral("rmbg_q8.gguf"))
                                         ? QStringLiteral("rmbg_q8.gguf")
                                         : QStringLiteral("rmbg_f16.gguf");
        const QString rmbg = rmbgCacheDir + QLatin1Char('/') + rmbgName;
        if (!TrellisHelpers::isValidModelFile(rmbg, rmbgName)) {
            missing << rmbgName;
        }
    }
    return missing;
}

void TrellisDialog::onDownloadModels() {
    const QString cacheDir = TrellisHelpers::modelCacheDir();
    const QString rmbgCacheDir = TrellisHelpers::rmbgModelCacheDir();
    QDir().mkpath(cacheDir);
    QDir().mkpath(rmbgCacheDir);
    const QVector<TrellisPreset> presets = TrellisHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return;
    const QStringList needed = TrellisHelpers::resolvePresetFiles(
            presets[idx], cacheDir, quantization());

    m_pendingDownloads.clear();
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;  // preset placeholder (model not in set)
        const QString name = QFileInfo(p).fileName();
        if (!TrellisHelpers::isValidModelFile(p, name)) {
            m_pendingDownloads << name;
        }
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbgName = TrellisHelpers::isValidModelFile(
                                         rmbgCacheDir + QLatin1Char('/') +
                                                 QStringLiteral("rmbg_q8.gguf"),
                                         QStringLiteral("rmbg_q8.gguf"))
                                         ? QStringLiteral("rmbg_q8.gguf")
                                         : QStringLiteral("rmbg_f16.gguf");
        if (!TrellisHelpers::isValidModelFile(
                    rmbgCacheDir + QLatin1Char('/') + rmbgName, rmbgName)) {
            m_pendingDownloads << rmbgName;
        }
    }
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[TRELLIS] Nothing to download."));
        updateModelStatus();
        return;
    }
    // Report the exact total from the HF mirror instead of a rough guess.
    qint64 totalBytes = 0;
    for (const QString& name : m_pendingDownloads) {
        HfModelInfo info;
        if (TrellisHelpers::hfModelInfo(name, &info)) {
            totalBytes += info.sizeBytes;
        }
    }
    appendLog(tr("[TRELLIS] %1 model(s) to download (%2 total).")
                      .arg(m_pendingDownloads.size())
                      .arg(ecvModelDownloader::formatFileSize(totalBytes)));
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
    // The AICore catalog supplies the HF URL, size, and LFS SHA-256 for every
    // supported model; qTrellis deliberately owns no duplicate model table.
    HfModelInfo hfInfo;
    if (!TrellisHelpers::hfModelInfo(filename, &hfInfo)) {
        appendLog(tr("[TRELLIS] Model is absent from the AICore catalog: %1")
                          .arg(filename));
        downloadNextModel();
        return;
    }
    const QString url = TrellisHelpers::hfDownloadUrl(filename);
    const QString dest = TrellisHelpers::modelCacheDirFor(filename) +
                         QDir::separator() + filename;
    appendLog(tr("[TRELLIS] Downloading %1...").arg(filename));
    m_downloadInProgress = true;
    ecvModelDownloader::Request req;
    req.url = url;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;
    req.contentAnchor = {QCryptographicHash::Sha256,
                         hfInfo.sha256.toLatin1()};  // streamed content check
    m_downloader->download(req);
}

void TrellisDialog::onTestDataClicked() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::Image2Mesh;

    // 1. Already extracted: fill the picker and select the first image.
    const QStringList images = ecvTestDataRepository::getImage2MeshImages(
            ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImages();
        return;
    }

    // 2. Zip cached and intact: extract, then populate (signal chain).
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
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
    const QStringList images = ecvTestDataRepository::getImage2MeshImages(
            ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImages();
    }
}

void TrellisDialog::populateTestImages() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    const QStringList images = ecvTestDataRepository::getImage2MeshImages(
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
    // Prefer the curated "T" sample (the representative single-image-to-3D
    // demo from the bundle); fall back to the first image otherwise.
    int pickIndex = 0;
    const int tIndex = m_testImageCombo->findText(QStringLiteral("T.png"));
    if (tIndex >= 0) {
        pickIndex = tIndex;
    }
    m_testImageCombo->setCurrentIndex(pickIndex);
    onTestImageSelected(pickIndex);
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
    resetStageStrip(m_lastPreviewImage);
    emit runRequested(getSettings());
}

void TrellisDialog::onRunOneClick() {
    // One-click end-to-end: guarantee a GLB destination, then run. The GLB
    // write itself happens in the plugin when the result arrives.
    if (m_saveGlbDir->text().trimmed().isEmpty()) {
        const QString downloads = QStandardPaths::writableLocation(
                QStandardPaths::DownloadLocation);
        m_saveGlbDir->setText(downloads + QStringLiteral("/TRELLIS"));
    }
    onRun();
}

void TrellisDialog::onCancel() { emit cancelRequested(); }

void TrellisDialog::refreshModelState() { updateModelStatus(); }

void TrellisDialog::buildExportPage(QWidget* page) {
    auto* root = new QVBoxLayout(page);
    ecvAICoreUi::setupTabLayout(root);

    auto* infoGroup = new QGroupBox(tr("Last generation result"), page);
    ecvAICoreUi::tightenGroupBox(infoGroup);
    auto* infoLayout = new QVBoxLayout(infoGroup);
    m_exportInfo = new QLabel(tr("Run a generation first."), infoGroup);
    m_exportInfo->setWordWrap(true);
    m_exportInfo->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    infoLayout->addWidget(m_exportInfo);
    root->addWidget(infoGroup);

    auto* glbGroup = new QGroupBox(tr("Textured GLB export"), page);
    ecvAICoreUi::tightenGroupBox(glbGroup);
    auto* glbLayout = new QGridLayout(glbGroup);
    glbLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    glbLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    glbLayout->addWidget(new QLabel(tr("Texture size:"), glbGroup), 0, 0);
    m_exportTextureSize = new QSpinBox(glbGroup);
    m_exportTextureSize->setRange(256, 8192);
    m_exportTextureSize->setSingleStep(256);
    m_exportTextureSize->setValue(2048);
    glbLayout->addWidget(m_exportTextureSize, 0, 1);
    glbLayout->addWidget(new QLabel(tr("Components:"), glbGroup), 1, 0);
    m_exportComponentFilter = new QComboBox(glbGroup);
    m_exportComponentFilter->addItem(tr("Remove tiny islands"));
    m_exportComponentFilter->addItem(tr("Largest component only"));
    m_exportComponentFilter->addItem(tr("Keep all"));
    glbLayout->addWidget(m_exportComponentFilter, 1, 1);
    root->addWidget(glbGroup);

    auto* actions = new QHBoxLayout();
    m_rebakeBtn = new QPushButton(tr("Re-bake GLB..."), page);
    m_rebakeBtn->setToolTip(
            tr("Re-run the UV-unwrap + PBR atlas bake on the last generated "
               "mesh with the settings above, and save it into the Save GLB "
               "directory."));
    m_printWrapBtn = new QPushButton(tr("Print wrap (CGAL)"), page);
    m_printWrapBtn->setToolTip(
            tr("Watertight Alpha-Wrap print mesh preview of the last "
               "generation (requires a CGAL build)."));
    actions->addWidget(m_rebakeBtn);
    actions->addWidget(m_printWrapBtn);
    actions->addStretch(1);
    root->addLayout(actions);
    root->addStretch(1);

#ifdef AICore_ENABLED
    const bool printable = aicore_trellis_print_remesh_available() != 0;
    m_printWrapBtn->setEnabled(printable);
    if (!printable) {
        m_printWrapBtn->setToolTip(
                tr("Unavailable: rebuild ACloudViewer with CGAL >= 5.5 to "
                   "enable the print wrap."));
    }
#else
    m_printWrapBtn->setEnabled(false);
    m_rebakeBtn->setEnabled(false);
#endif

    connect(m_rebakeBtn, &QPushButton::clicked, this,
            [this]() { emit exportRequested(); });
}

QString TrellisDialog::quantization() const {
    return m_quantCombo ? m_quantCombo->currentData().toString()
                        : QStringLiteral("q8");
}

void TrellisDialog::resetStageStrip(const QImage& input) {
    const QStringList names = {tr("Source"), tr("Preprocess"), tr("Voxels"),
                               tr("Mesh"),   tr("Texture"),    tr("GLB")};
    for (int i = 0; i < m_stageThumbs.size() && i < names.size(); ++i) {
        m_stageThumbs[i]->setStyleSheet(
                "border: 1px solid #B8C4D0; border-radius: 3px; "
                "background: #F4F7FA; color: #98A4B0;");
        m_stageThumbs[i]->setText(QStringLiteral("-"));
        m_stageThumbs[i]->setPixmap(QPixmap());
        m_stageCaptions[i]->setText(names[i]);
        m_stageCaptions[i]->setStyleSheet("color: #5A6672; font-size: 10px;");
    }
    if (!input.isNull() && !m_stageThumbs.isEmpty()) {
        m_stageThumbs[0]->setPixmap(QPixmap::fromImage(
                input.scaled(m_stageThumbs[0]->size(), Qt::KeepAspectRatio,
                             Qt::SmoothTransformation)));
    }
}

void TrellisDialog::updateExportInfo(const TrellisRunResult& result) {
    if (!m_exportInfo) return;
    if (result.verts.isEmpty()) {
        m_exportInfo->setText(tr("Run a generation first."));
        return;
    }
    const int nv = result.verts.size() / 3;
    const int nt = result.tris.size() / 3;
    m_exportInfo->setText(
            tr("Mesh: %1 verts / %2 tris\nBackend: %3 (%4 chain, %5 ms)\n"
               "PBR: %6\nGrid: %7\nSource: %8")
                    .arg(nv)
                    .arg(nt)
                    .arg(result.backend)
                    .arg(result.quantization)
                    .arg(result.totalRuntimeMs, 0, 'f', 0)
                    .arg(result.hasPbr ? tr("textured") : tr("untextured"))
                    .arg(result.gridRes > 0
                                 ? tr("%1\u00b3 dual grid").arg(result.gridRes)
                                 : tr("(coarse)"))
                    .arg(QFileInfo(result.sourceImage).fileName()));
}

void TrellisDialog::setStageState(int slot, int state) {
    if (slot < 0 || slot >= m_stageThumbs.size()) return;
    const QString border =
            state == kStageDone
                    ? QStringLiteral("2px solid #3D9A50")
                    : (state == kStageActive
                               ? QStringLiteral("2px solid #2E7BD0")
                               : QStringLiteral("1px solid #B8C4D0"));
    m_stageThumbs[slot]->setStyleSheet(
            QStringLiteral("border: %1; border-radius: 3px; "
                           "background: #F4F7FA;")
                    .arg(border));
    if (state == kStageDone) {
        m_stageCaptions[slot]->setStyleSheet(
                "color: #3D9A50; font-size: 10px; font-weight: bold;");
    } else if (state == kStageActive) {
        m_stageCaptions[slot]->setStyleSheet(
                "color: #2E7BD0; font-size: 10px; font-weight: bold;");
    }
}

void TrellisDialog::setStagePreview(const TrellisStagePreview& preview) {
    // Map the AICore stage onto the strip slot: voxel sets (SS_FLOW /
    // SS_DEC / UPSAMPLE) light the Voxels chip, mesh keyframes and decodes
    // (SLAT_FLOW* / SHAPE_DEC*) the Mesh chip.
    int slot = -1;
    switch (preview.stage) {
        case AICORE_TRELLIS_STAGE_SS_FLOW:
        case AICORE_TRELLIS_STAGE_SS_DEC:
        case AICORE_TRELLIS_STAGE_UPSAMPLE:
            slot = 2;
            break;
        case AICORE_TRELLIS_STAGE_SLAT_FLOW:
        case AICORE_TRELLIS_STAGE_SLAT_FLOW_HR:
        case AICORE_TRELLIS_STAGE_SHAPE_DEC:
        case AICORE_TRELLIS_STAGE_SHAPE_DEC_HR:
            slot = 3;
            break;
        default:
            break;
    }
    if (slot < 0 || slot >= m_stageThumbs.size()) return;
    if (!preview.image.isNull()) {
        m_stageThumbs[slot]->setPixmap(QPixmap::fromImage(preview.image.scaled(
                m_stageThumbs[slot]->size(), Qt::KeepAspectRatio,
                Qt::SmoothTransformation)));
    }
    if (!preview.label.isEmpty()) {
        m_stageCaptions[slot]->setText(preview.label);
    }
    setStageState(slot, kStageActive);
}
