// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "InstantMeshDialog.h"

#include <QCloseEvent>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QMouseEvent>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QStandardPaths>
#include <QVBoxLayout>
#include <functional>

#include "aicore/inference_log.h"
#include "ecvAICoreMeshViewer.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"

#ifdef AICore_ENABLED
#include <aicore/rmbg_capi.h>
#endif

#ifdef AICore_ENABLED
#include <aicore/backend_capi.h>
#include <aicore/instantmesh_capi.h>
#endif

namespace {

constexpr int kDefaultSteps = 75;
constexpr double kDefaultGuidance = 4.0;
constexpr int kDefaultGridRes = 64;
constexpr double kDefaultGridScale = 2.1;
constexpr int kDefaultTextureRes = 2048;

}  // namespace

namespace {

// Pipeline-strip chip for the mesh-bearing stages (Mesh / GLB): the click
// opens the shared 3D orbit viewer when a result mesh is available and
// falls back to the shared image zoom otherwise (the qTrellis
// MeshStageChip rule).
class MeshStageChip : public ecvClickableImageLabel {
public:
    using ecvClickableImageLabel::ecvClickableImageLabel;

    std::function<void()> openViewer;

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (openViewer && event->button() == Qt::LeftButton) {
            openViewer();
            return;
        }
        ecvClickableImageLabel::mousePressEvent(event);
    }
};

}  // namespace

InstantMeshDialog::InstantMeshDialog(QWidget* parent) : QDialog(parent) {
    setAttribute(Qt::WA_DeleteOnClose, false);
    setupUi();
    loadSettings();
    connect(&ecvTestDataRepository::instance(),
            &ecvTestDataRepository::datasetReady, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                if (kind != ecvTestDataRepository::Dataset::Image2Mesh) return;
                m_samplePreparing = false;
                if (!m_acceptDataResults) return;
                m_useTestDataBtn->setEnabled(true);
                if (success)
                    onTestDataExtracted();
                else {
                    setProgressIdle();
                    appendLog(tr(
                            "[InstantMesh] Sample data preparation failed."));
                }
            });
    auto& repo = ecvTestDataRepository::instance();
    connect(&repo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_samplePreparing && m_acceptDataResults)
                    appendLog(message);
            });
    connect(&repo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& status) {
                if (m_samplePreparing && m_acceptDataResults)
                    setProgressStage(0, status, percent, 100);
            });
    refreshModelState();
    ensureImage2MeshDataset();
    constrainGeometry();
}

InstantMeshDialog::~InstantMeshDialog() { saveSettings(); }

void InstantMeshDialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
}

void InstantMeshDialog::setupUi() {
    setWindowTitle(tr("InstantMesh Image to 3D"));
    buildGeneratePage(this);
}

QSize InstantMeshDialog::pinnedMinimumSize() const {
    // The fixed blocks' minimum alone lets the window shrink below the
    // width the configuration content needs without horizontal scrolling
    // (a scroll area's minimum ignores its content); widen the floor to
    // the content minimum plus a vertical scrollbar.
    QSize minSize = minimumSizeHint();
    if (auto* scroll = findChild<QScrollArea*>()) {
        const int contentWidth =
                scroll->widget()->minimumSizeHint().width() +
                scroll->verticalScrollBar()->sizeHint().width() +
                ecvAICoreUi::dpiScaled(8);
        minSize.setWidth(qMax(minSize.width(), contentWidth));
    }
    return minSize;
}

void InstantMeshDialog::setProgressIdle() {
    // The row stays laid out (geometry conservation); an idle row is an
    // empty determinate bar. reset() alone would keep a busy (0,0) range
    // animating forever.
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_stageLabel->clear();
}

void InstantMeshDialog::constrainGeometry() {
    // Geometry conservation (the qQuerySplat rule): measure the layout
    // minimum after the settings and catalogs filled the form, pin the
    // window minimum to it (the user can never shrink into clipping the
    // fixed run/progress block), and keep the initial size inside the
    // screen so window managers on scaled or small displays cannot clip
    // the bottom.
    adjustSize();
    const QSize minSize = pinnedMinimumSize();
    setMinimumSize(minSize);
    QSize target(ecvAICoreUi::dpiScaled(720), ecvAICoreUi::dpiScaled(660));
    target = target.expandedTo(minSize);
    if (const QScreen* screen = QGuiApplication::primaryScreen()) {
        target = target.boundedTo(screen->availableSize() - QSize(16, 16));
    }
    resize(target);
}

void InstantMeshDialog::buildGeneratePage(QWidget* page) {
    auto* root = new QVBoxLayout(page);
    ecvAICoreUi::setupTabLayout(root);
    auto* outerRoot = root;

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
    m_testImageCombo = new QComboBox(inputGroup);
    m_testImageCombo->setEnabled(false);
    m_testImageCombo->setToolTip(
            tr("Pick one of the bundled sample images (Image2Mesh dataset)."));
    inputLayout->addWidget(m_testImageCombo, 1, 0, 1, 4);
    m_imagePreview = new ecvClickableImageLabel(inputGroup);
    m_imagePreview->setFixedSize(ecvAICoreUi::dpiScaled(96),
                                 ecvAICoreUi::dpiScaled(96));
    m_imagePreview->setAlignment(Qt::AlignCenter);
    m_imagePreview->setStyleSheet("border: 1px dashed #999;");
    inputLayout->addWidget(m_imagePreview, 0, 4, 2, 1);
    inputLayout->setColumnStretch(0, 1);
    root->addWidget(inputGroup);

    connect(m_imagePath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateImagePreview(); });
    connect(m_browseImageBtn, &QPushButton::clicked, this,
            &InstantMeshDialog::onBrowseImage);
    connect(m_useTestDataBtn, &QPushButton::clicked, this,
            &InstantMeshDialog::onTestDataClicked);
    connect(m_testImageCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &InstantMeshDialog::onTestImageSelected);

    // Only configuration scrolls; input and execution feedback remain fixed.
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setMinimumHeight(ecvAICoreUi::dpiScaled(80));
    auto* controls = new QWidget(scroll);
    scroll->setWidget(controls);
    outerRoot->addWidget(scroll, 1);
    root = new QVBoxLayout(controls);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(ecvAICoreUi::tightVSpacing());

    // ── Runtime parameters (shared row) ─────────────────────────────────
    m_deviceCombo = new QComboBox(this);
#ifdef AICore_ENABLED
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev || !dev->id) continue;
        m_deviceCombo->addItem(QString::fromUtf8(dev->label),
                               QString::fromUtf8(dev->id));
        if (dev->is_default) m_deviceCombo->setCurrentIndex(i);
    }
    m_deviceCombo->setToolTip(
            tr("Inference backend. 'auto' picks the best registered device; "
               "a low-VRAM card falls back gracefully instead of failing."));
#else
    m_deviceCombo->addItem(tr("auto"), QStringLiteral("auto"));
    m_deviceCombo->addItem(tr("cpu"), QStringLiteral("cpu"));
#endif
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
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    for (const InstantMeshPreset& p : presets) {
        m_presetCombo->addItem(p.name, p.description);
    }
    m_presetCombo->setCurrentIndex(0);  // Full multi-view
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InstantMeshDialog::onPresetChanged);
    modelLayout->addWidget(m_presetCombo, 0, 1);

    modelLayout->addWidget(new QLabel(tr("Quantization:"), modelGroup), 0, 2);
    m_quantCombo = new QComboBox(modelGroup);
    // f16 is the recommended chain (every model that publishes f16 uses it;
    // the Zero123++ cond constants only publish f32). The q8 chain swaps the
    // InstantMesh trio to q8 (the Zero123++ UNet/VAE have no q8 variant and
    // stay f16). f32 is the full exact-reference chain.
    m_quantCombo->addItem(tr("f16 (recommended)"), QStringLiteral("f16"));
    m_quantCombo->addItem(tr("q8 (low memory, mixed)"), QStringLiteral("q8"));
    m_quantCombo->addItem(tr("f32 (exact reference)"), QStringLiteral("f32"));
    m_quantCombo->setToolTip(
            tr("Weight precision of the chain. f16 is the published "
               "recommended set; q8 shrinks the InstantMesh trio (Zero123++ "
               "has no q8 release and stays f16); f32 is the exact "
               "reference."));
    connect(m_quantCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &InstantMeshDialog::onQuantChanged);
    modelLayout->addWidget(m_quantCombo, 0, 3);

    m_bakeTextureCheck =
            new QCheckBox(tr("Bake UV texture (2048)"), modelGroup);
    m_bakeTextureCheck->setChecked(true);
    m_bakeTextureCheck->setToolTip(
            tr("Unwrap the mesh with xatlas and bake the net_rgb appearance "
               "into a texture atlas (slower; untick for the fast vertex-"
               "colour-only path)."));
    modelLayout->addWidget(m_bakeTextureCheck, 2, 0, 1, 2);

    m_rmbgCheck = new QCheckBox(tr("AI background removal"), modelGroup);
    m_rmbgCheck->setChecked(true);
    m_rmbgCheck->setToolTip(
            tr("Remove the background with the in-tree RMBG-2.0 model before "
               "generation (matches the official rembg preprocessing). "
               "The selected model is required when enabled."));
    modelLayout->addWidget(m_rmbgCheck, 2, 2, 1, 2);

    modelLayout->addWidget(new QLabel(tr("RMBG precision:"), modelGroup), 3, 0);
    m_rmbgQuantCombo = new QComboBox(modelGroup);
#ifdef AICore_ENABLED
    for (int i = 0; i < aicore_rmbg_model_count(); ++i) {
        const auto* entry = aicore_rmbg_model_at(i);
        if (entry && entry->filename)
            m_rmbgQuantCombo->addItem(QString::fromUtf8(entry->display_name),
                                      QString::fromUtf8(entry->filename));
    }
    m_rmbgQuantCombo->setCurrentIndex(aicore_rmbg_model_default_index());
#endif
    m_rmbgQuantCombo->setToolTip(
            tr("RMBG-2.0 GGUF quantization (shared with the qRMBG plugin "
               "cache)."));
    modelLayout->addWidget(m_rmbgQuantCombo, 3, 1, 1, 3);

    // Long catalog labels belong in the popup, not the form's minimum width.
    for (auto* combo :
         {m_deviceCombo, m_presetCombo, m_quantCombo, m_rmbgQuantCombo}) {
        combo->setSizeAdjustPolicy(
                QComboBox::AdjustToMinimumContentsLengthWithIcon);
        combo->setMinimumContentsLength(8);
        combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }
    modelLayout->setColumnStretch(1, 1);
    modelLayout->setColumnStretch(3, 1);

    m_modelStatus = new QLabel(modelGroup);
    m_modelStatus->setWordWrap(true);
    modelLayout->addWidget(m_modelStatus, 4, 0, 1, 3);
    m_downloadBtn = ecvAICoreUi::makeBrowseBtn(tr("Download missing models..."),
                                               modelGroup);
    m_downloadBtn->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    m_downloadBtn->setMinimumWidth(ecvAICoreUi::dpiScaled(168));
    m_downloadBtn->setMaximumWidth(QWIDGETSIZE_MAX);
    modelLayout->addWidget(m_downloadBtn, 4, 3);
    root->addWidget(modelGroup);

    connect(m_downloadBtn, &QPushButton::clicked, this,
            &InstantMeshDialog::onDownloadModels);
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &InstantMeshDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (m_acceptDataResults)
                    setProgressStage(
                            0,
                            ecvModelDownloader::formatDownloadProgress(received,
                                                                       total),
                            total > 0 ? int(100.0 * received / total) : 0,
                            total > 0 ? 100 : 0);
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& destPath) {
                m_downloadInProgress = false;
                if (!ok) {
                    appendLog(tr("[InstantMesh] Download failed: %1")
                                      .arg(destPath));
                    m_pendingDownloads.clear();
                    m_pendingActionAfterDownload = PendingAction::None;
                    updateModelStatus();
                    return;
                }
                QFileInfo fi(destPath);
                appendLog(
                        tr("[InstantMesh] Downloaded %1.").arg(fi.fileName()));
                downloadNextModel();
            });

    // ── Parameters ───────────────────────────────────────────────────────
    auto* paramGroup = new QGroupBox(tr("Parameters"), this);
    ecvAICoreUi::tightenGroupBox(paramGroup);
    auto* paramLayout = new QGridLayout(paramGroup);
    paramLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());
    paramLayout->setVerticalSpacing(ecvAICoreUi::tightVSpacing());
    paramLayout->setColumnStretch(0, 1);

    m_stepsAuto = new QCheckBox(tr("Default steps"), paramGroup);
    m_stepsAuto->setChecked(true);
    m_stepsAuto->setToolTip(
            tr("Zero123++ denoising steps (default 75). More steps = "
               "cleaner views, slower generation."));
    m_steps = new QSpinBox(paramGroup);
    m_steps->setRange(2, 1000);
    m_steps->setValue(kDefaultSteps);
    m_steps->setEnabled(false);
    ecvAICoreUi::setCompactSpin(m_steps);
    paramLayout->addWidget(m_stepsAuto, 0, 0);
    paramLayout->addWidget(m_steps, 0, 1, Qt::AlignLeft);
    connect(m_stepsAuto, &QCheckBox::toggled, this, [this](bool autoOn) {
        m_steps->setEnabled(!autoOn);
        if (autoOn) m_steps->setValue(kDefaultSteps);
    });

    m_guidanceAuto = new QCheckBox(tr("Default guidance"), paramGroup);
    m_guidanceAuto->setChecked(true);
    m_guidanceAuto->setToolTip(
            tr("Classifier-free guidance scale (default 4.0)."));
    m_guidance = new QDoubleSpinBox(paramGroup);
    m_guidance->setRange(1.0, 15.0);
    m_guidance->setDecimals(1);
    m_guidance->setSingleStep(0.5);
    m_guidance->setValue(kDefaultGuidance);
    m_guidance->setEnabled(false);
    ecvAICoreUi::setCompactDoubleSpin(m_guidance);
    paramLayout->addWidget(m_guidanceAuto, 0, 2);
    paramLayout->addWidget(m_guidance, 0, 3, Qt::AlignLeft);
    connect(m_guidanceAuto, &QCheckBox::toggled, this, [this](bool autoOn) {
        m_guidance->setEnabled(!autoOn);
        if (autoOn) m_guidance->setValue(kDefaultGuidance);
    });

    m_seedRandom = new QCheckBox(tr("Random seed"), paramGroup);
    m_seedRandom->setChecked(false);
    m_seedRandom->setToolTip(
            tr("Use a random seed for the diffusion noise. Uncheck to enter "
               "a fixed seed and reproduce a previous run."));
    m_seed = new QSpinBox(paramGroup);
    m_seed->setRange(0, 999999);
    m_seed->setValue(42);
    m_seed->setEnabled(true);
    ecvAICoreUi::setCompactSpin(m_seed);
    m_seed->setFixedWidth(ecvAICoreUi::dpiScaled(96));
    paramLayout->addWidget(m_seedRandom, 1, 0);
    paramLayout->addWidget(m_seed, 1, 1, Qt::AlignLeft);
    connect(m_seedRandom, &QCheckBox::toggled, this,
            [this](bool randomOn) { m_seed->setEnabled(!randomOn); });

    paramLayout->addWidget(new QLabel(tr("Grid resolution:"), paramGroup), 1,
                           2);
    m_gridRes = new QSpinBox(paramGroup);
    m_gridRes->setRange(16, 256);
    m_gridRes->setValue(kDefaultGridRes);
    m_gridRes->setToolTip(
            tr("FlexiCubes cubes per axis (default 64; the upstream benchmark "
               "uses 88 for the highest quality at quadratic cost)."));
    ecvAICoreUi::setCompactSpin(m_gridRes);
    paramLayout->addWidget(m_gridRes, 1, 3, Qt::AlignLeft);

    paramLayout->addWidget(new QLabel(tr("Texture size:"), paramGroup), 2, 0);
    m_textureRes = new QSpinBox(paramGroup);
    m_textureRes->setRange(256, 4096);
    m_textureRes->setSingleStep(256);
    m_textureRes->setValue(kDefaultTextureRes);
    m_textureRes->setToolTip(
            tr("UV atlas bake resolution (default 2048). Higher = less bake "
               "aliasing, slower bake."));
    ecvAICoreUi::setCompactSpin(m_textureRes);
    paramLayout->addWidget(m_textureRes, 2, 1, Qt::AlignLeft);
    root->addWidget(paramGroup);

    // ── Pipeline step strip ───────────────────────────────────────────────
    auto* stripGroup = new QGroupBox(tr("Pipeline steps"), this);
    ecvAICoreUi::tightenGroupBox(stripGroup);
    auto* stripLayout = new QHBoxLayout(stripGroup);
    stripLayout->setSpacing(ecvAICoreUi::dpiScaled(6));
    const QStringList kStepNames = {tr("Source"), tr("RMBG"), tr("Views"),
                                    tr("Mesh"), tr("GLB")};
    const int thumb = ecvAICoreUi::dpiScaled(48);
    for (int slot = 0; slot < kStepNames.size(); ++slot) {
        auto* cell = new QVBoxLayout();
        cell->setSpacing(1);
        ecvClickableImageLabel* thumbLabel = nullptr;
        if (slot >= 3) {
            // Mesh / GLB chips carry the result mesh: the click opens the
            // shared 3D orbit viewer (the qTrellis MeshStageChip rule).
            auto* meshChip = new MeshStageChip(stripGroup);
            meshChip->openViewer = [this, slot]() { openMesh3D(slot); };
            thumbLabel = meshChip;
        } else {
            thumbLabel = new ecvClickableImageLabel(stripGroup);
        }
        thumbLabel->setFixedSize(thumb, thumb);
        thumbLabel->setAlignment(Qt::AlignCenter);
        thumbLabel->setStyleSheet(
                "border: 1px solid #bbb; background: #fafafa;");
        auto* caption = new QLabel(kStepNames[slot], stripGroup);
        caption->setAlignment(Qt::AlignCenter);
        caption->setStyleSheet("color: #666; font-size: 8pt;");
        cell->addWidget(thumbLabel, 0, Qt::AlignHCenter);
        cell->addWidget(caption, 0, Qt::AlignHCenter);
        stripLayout->addLayout(cell);
        m_stageThumbs.append(thumbLabel);
        m_stageCaptions.append(caption);
    }
    stripLayout->addStretch(1);
    // Fixed below the configuration scroll: the strip is run feedback (its
    // chips light up while stages advance), never scroll it out of view.
    outerRoot->addWidget(stripGroup);

    root->addStretch();

    // ── Fixed run / progress ─────────────────────────────────────────────
    auto* runGroup = new QGroupBox(this);
    ecvAICoreUi::tightenGroupBox(runGroup);
    auto* runLayout = new QGridLayout(runGroup);
    runLayout->setHorizontalSpacing(ecvAICoreUi::hSpacing());

    auto* runBtn = m_runBtn = new QPushButton(tr("Generate 3D Mesh"), runGroup);
    connect(runBtn, &QPushButton::clicked, this, &InstantMeshDialog::onRun);

    auto* cancelBtn = m_cancelBtn = new QPushButton(tr("Cancel"), runGroup);
    cancelBtn->setEnabled(false);
    connect(cancelBtn, &QPushButton::clicked, this,
            &InstantMeshDialog::onCancel);

    m_addToDbCheck = new QCheckBox(tr("Add result to DB tree"), runGroup);
    m_addToDbCheck->setChecked(true);

    runLayout->addWidget(runBtn, 0, 0);
    runLayout->addWidget(cancelBtn, 0, 1);
    runLayout->addWidget(m_addToDbCheck, 0, 2, Qt::AlignRight);
    runLayout->setColumnStretch(2, 1);

    // The stage label and progress bar stay laid out permanently (empty
    // when idle): geometry conservation keeps the window minimum constant
    // so showing them mid-run can never push the fixed bottom block out of
    // the window.
    m_stageLabel = new QLabel(runGroup);
    m_progress = new QProgressBar(runGroup);
    runLayout->addWidget(m_stageLabel, 1, 0, 1, 3);
    runLayout->addWidget(m_progress, 2, 0, 1, 3);
    outerRoot->addWidget(runGroup);
}

void InstantMeshDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qInstantMesh"));
    m_imagePath->setText(
            settings.value(QStringLiteral("imagePath")).toString());
    m_presetCombo->setCurrentIndex(
            settings.value(QStringLiteral("preset"), 0).toInt());
    m_quantCombo->setCurrentIndex(
            settings.value(QStringLiteral("quant"), 0).toInt());
    QString selectedRmbg =
            settings.value(QStringLiteral("rmbgModel")).toString();
    if (selectedRmbg.isEmpty() &&
        settings.contains(QStringLiteral("rmbgQuant")))
        selectedRmbg = settings.value(QStringLiteral("rmbgQuant")).toInt() == 1
                               ? QStringLiteral("rmbg_f16.gguf")
                               : QStringLiteral("rmbg_q8.gguf");
    const int rmbgIndex = m_rmbgQuantCombo->findData(selectedRmbg);
    if (rmbgIndex >= 0) m_rmbgQuantCombo->setCurrentIndex(rmbgIndex);
    m_rmbgCheck->setChecked(
            settings.value(QStringLiteral("useRmbg"), true).toBool());
    m_bakeTextureCheck->setChecked(
            settings.value(QStringLiteral("bakeTexture"), true).toBool());
    m_addToDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int devIdx = m_deviceCombo->findData(device);
    if (devIdx >= 0) m_deviceCombo->setCurrentIndex(devIdx);
    settings.endGroup();
    updateImagePreview();
}

void InstantMeshDialog::saveSettings() const {
    if (!m_imagePath) return;
    QSettings settings;
    settings.beginGroup(QStringLiteral("qInstantMesh"));
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("preset"), m_presetCombo->currentIndex());
    settings.setValue(QStringLiteral("quant"), m_quantCombo->currentIndex());
    settings.setValue(QStringLiteral("rmbgModel"),
                      m_rmbgQuantCombo->currentData());
    settings.setValue(QStringLiteral("useRmbg"), m_rmbgCheck->isChecked());
    settings.setValue(QStringLiteral("bakeTexture"),
                      m_bakeTextureCheck->isChecked());
    settings.setValue(QStringLiteral("addToDb"), m_addToDbCheck->isChecked());
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.endGroup();
}

InstantMeshDialog::Settings InstantMeshDialog::getSettings() const {
    Settings s;
    const QString cacheDir = InstantMeshHelpers::modelCacheDir();
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx >= 0 && idx < presets.size()) {
        s.modelPaths = InstantMeshHelpers::resolvePresetFiles(
                presets[idx], cacheDir, quantization());
        s.presetName = presets[idx].name;
    }
    s.inputPath = m_imagePath->text().trimmed();
    s.steps = m_stepsAuto->isChecked() ? 0 : m_steps->value();
    s.seed =
            m_seedRandom->isChecked()
                    ? static_cast<uint64_t>(QDateTime::currentMSecsSinceEpoch())
                    : static_cast<uint64_t>(m_seed->value());
    s.guidance = m_guidanceAuto->isChecked() ? -1.0 : m_guidance->value();
    s.gridRes = m_gridRes->value();
    s.gridScale = kDefaultGridScale;
    s.textureRes = m_textureRes->value();
    s.bakeTexture = m_bakeTextureCheck->isChecked();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.useRmbg = m_rmbgCheck->isChecked();
    if (s.useRmbg)
        s.rmbgModelPath = InstantMeshHelpers::rmbgModelCacheDir() +
                          QDir::separator() +
                          m_rmbgQuantCombo->currentData().toString();
    s.addResultToDb = m_addToDbCheck->isChecked();
    s.quantization = quantization();
    return s;
}

QString InstantMeshDialog::quantization() const {
    return m_quantCombo->currentData().toString();
}

void InstantMeshDialog::onPresetChanged(int index) {
    Q_UNUSED(index);
    updateModelStatus();
}

void InstantMeshDialog::onQuantChanged(int index) {
    Q_UNUSED(index);
    updateModelStatus();
}

void InstantMeshDialog::onBrowseImage() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select input image"),
            m_imagePath->text().trimmed().isEmpty()
                    ? QStandardPaths::writableLocation(
                              QStandardPaths::PicturesLocation)
                    : QFileInfo(m_imagePath->text()).absolutePath(),
            tr("Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    updateImagePreview();
}

void InstantMeshDialog::updateImagePreview() {
    const QString path = m_imagePath->text().trimmed();
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        m_imagePreview->clearPreview();
        return;
    }
    QImage img(path);
    if (img.isNull()) {
        m_imagePreview->clearPreview();
        return;
    }
    m_lastPreviewImage = img;
    // setPreviewImage keeps the full-resolution image as the
    // click-to-enlarge target (qTrellis behavior); a raw setPixmap leaves
    // the label inert.
    m_imagePreview->setPreviewImage(img, m_imagePreview->size());
}

void InstantMeshDialog::refreshModelState() { updateModelStatus(); }

QStringList InstantMeshDialog::missingPresetFiles() const {
    const QString cacheDir = InstantMeshHelpers::modelCacheDir();
    const QString rmbgCacheDir = InstantMeshHelpers::rmbgModelCacheDir();
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return {};
    const QStringList needed = InstantMeshHelpers::resolvePresetFiles(
            presets[idx], cacheDir, quantization());
    QStringList missing;
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;
        const QString name = QFileInfo(p).fileName();
        if (!InstantMeshHelpers::isValidModelFile(p, name)) missing << name;
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbgName = m_rmbgQuantCombo->currentData().toString();
        const QString rmbg = rmbgCacheDir + QLatin1Char('/') + rmbgName;
        if (!InstantMeshHelpers::isValidModelFile(rmbg, rmbgName)) {
            missing << rmbgName;
        }
    }
    return missing;
}

void InstantMeshDialog::updateModelStatus() {
    const QStringList missing = missingPresetFiles();
    m_modelStatus->setTextInteractionFlags(Qt::TextSelectableByMouse);
    if (missing.isEmpty()) {
        m_modelStatus->setText(tr("\u2705 All models present."));
        m_modelStatus->setToolTip(
                tr("InstantMesh models: %1\nRMBG models: %2")
                        .arg(InstantMeshHelpers::modelCacheDir(),
                             InstantMeshHelpers::rmbgModelCacheDir()));
        m_downloadBtn->setEnabled(false);
    } else {
        m_modelStatus->setText(tr("\u26a0 Missing %1 model(s): %2")
                                       .arg(missing.size())
                                       .arg(missing.join(", ")));
        QStringList rows;
        for (const QString& name : missing) {
            rows << QStringLiteral("%1 \u2192 %2/%3")
                            .arg(name,
                                 name.startsWith(QStringLiteral("rmbg"))
                                         ? InstantMeshHelpers::
                                                   rmbgModelCacheDir()
                                         : InstantMeshHelpers::modelCacheDir(),
                                 name);
        }
        const QString mirror = InstantMeshHelpers::hfMirrorUrl();
        m_modelStatus->setToolTip(
                tr("Manual download (if the button keeps failing):\n%1%2")
                        .arg(rows.join(QLatin1Char('\n')),
                             mirror.isEmpty() ? QString()
                                              : QStringLiteral("\nMirror: %1")
                                                        .arg(mirror)));
        m_downloadBtn->setEnabled(true);
    }
}

void InstantMeshDialog::onDownloadModels() {
    const QString cacheDir = InstantMeshHelpers::modelCacheDir();
    const QString rmbgCacheDir = InstantMeshHelpers::rmbgModelCacheDir();
    QDir().mkpath(cacheDir);
    QDir().mkpath(rmbgCacheDir);
    const QVector<InstantMeshPreset> presets = InstantMeshHelpers::presets();
    const int idx = m_presetCombo->currentIndex();
    if (idx < 0 || idx >= presets.size()) return;
    const QStringList needed = InstantMeshHelpers::resolvePresetFiles(
            presets[idx], cacheDir, quantization());

    m_pendingDownloads.clear();
    for (const QString& p : needed) {
        if (p.isEmpty()) continue;
        const QString name = QFileInfo(p).fileName();
        if (!InstantMeshHelpers::isValidModelFile(p, name)) {
            m_pendingDownloads << name;
        }
    }
    if (m_rmbgCheck->isChecked()) {
        const QString rmbgName = m_rmbgQuantCombo->currentData().toString();
        if (!InstantMeshHelpers::isValidModelFile(
                    rmbgCacheDir + QLatin1Char('/') + rmbgName, rmbgName)) {
            m_pendingDownloads << rmbgName;
        }
    }
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[InstantMesh] Nothing to download."));
        updateModelStatus();
        return;
    }
    qint64 totalBytes = 0;
    for (const QString& name : m_pendingDownloads) {
        HfInstantMeshInfo info;
        if (name.startsWith(QStringLiteral("rmbg"))) {
            // RMBG lives in the shared qRMBG catalog (same LFS objects); its
            // download goes through the qRMBG/qTrellis shared size table via
            // the pinned digest only (size shown as unknown).
            totalBytes += 0;
            continue;
        }
        if (InstantMeshHelpers::hfModelInfo(name, &info)) {
            totalBytes += info.sizeBytes;
        }
    }
    appendLog(tr("[InstantMesh] %1 model(s) to download (%2 total).")
                      .arg(m_pendingDownloads.size())
                      .arg(totalBytes > 0
                                   ? ecvModelDownloader::formatFileSize(
                                             totalBytes)
                                   : QStringLiteral("RMBG shared cache")));
    m_downloadInProgress = false;
    downloadNextModel();
}

void InstantMeshDialog::downloadNextModel() {
    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[InstantMesh] All model downloads finished."));
        updateModelStatus();
        if (m_pendingActionAfterDownload == PendingAction::Run) {
            m_pendingActionAfterDownload = PendingAction::None;
            onRun();
        }
        return;
    }
    const QString filename = m_pendingDownloads.takeFirst();
    QString url;
    QByteArray sha;
    if (filename.startsWith(QStringLiteral("rmbg"))) {
        // Shared RMBG dependency: URL and SHA-256 come from the in-tree
        // RMBG task catalog (single owner of those LFS objects).
        url.clear();
        sha.clear();
#ifdef AICore_ENABLED
        const int n = aicore_rmbg_model_count();
        for (int i = 0; i < n; ++i) {
            const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(i);
            if (e && e->filename &&
                filename == QString::fromUtf8(e->filename)) {
                url = QString::fromUtf8(e->download_url);
                break;
            }
        }
#endif
        if (url.isEmpty()) {
            appendLog(tr("[InstantMesh] RMBG model is absent from the AICore "
                         "RMBG catalog: %1")
                              .arg(filename));
            downloadNextModel();
            return;
        }
        // Content anchor from the global asset digest table (the RMBG rows
        // live there; the RMBG catalog struct carries no digest field).
        const QByteArray pinned = ecvAssetIntegrity::PinnedDigest(filename);
        if (!pinned.isEmpty()) sha = pinned;
    } else {
        HfInstantMeshInfo hfInfo;
        if (!InstantMeshHelpers::hfModelInfo(filename, &hfInfo)) {
            appendLog(tr("[InstantMesh] Model is absent from the AICore "
                         "catalog: %1")
                              .arg(filename));
            downloadNextModel();
            return;
        }
        url = InstantMeshHelpers::hfDownloadUrl(filename);
        sha = hfInfo.sha256.toLatin1();
    }
    const QString destDir = filename.startsWith(QStringLiteral("rmbg"))
                                    ? InstantMeshHelpers::rmbgModelCacheDir()
                                    : InstantMeshHelpers::modelCacheDir();
    const QString dest = destDir + QDir::separator() + filename;
    appendLog(tr("[InstantMesh] Downloading %1...\n  URL: %2\n  Dest: %3")
                      .arg(filename, url, dest));
    m_downloadInProgress = true;
    ecvModelDownloader::Request req;
    req.url = url;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;
    if (!sha.isEmpty()) {
        req.contentAnchor = {QCryptographicHash::Sha256, sha};
    } else {
        req.minBytes = 16 * 1024 * 1024;  // RMBG GGUFs are >250 MB
    }
    m_downloader->download(req);
}

void InstantMeshDialog::onTestDataClicked() {
    if (m_samplePreparing || m_taskRunning || m_downloadInProgress) return;
    m_acceptDataResults = true;
    m_samplePreparing = true;
    m_useTestDataBtn->setEnabled(false);
    m_progress->setRange(0, 0);
    m_progress->setVisible(true);
    m_stageLabel->setVisible(true);
    m_stageLabel->setText(tr("Preparing sample data..."));
    ecvTestDataRepository::instance().ensureDatasetReady(
            ecvTestDataRepository::Dataset::Image2Mesh);
}

void InstantMeshDialog::ensureImage2MeshDataset() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    const QStringList images = ecvTestDataRepository::getImage2MeshImages(
            ecvTestDataRepository::extractPath(kind));
    if (!images.isEmpty()) {
        populateTestImages();
    }
}

void InstantMeshDialog::populateTestImages() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    QStringList images = ecvTestDataRepository::getImage2MeshImages(
            ecvTestDataRepository::extractPath(kind));
    for (auto it = images.begin(); it != images.end();) {
        if (QImage(*it).isNull())
            it = images.erase(it);
        else
            ++it;
    }
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
    // The default comes from the shared release ZIP, never a machine-local
    // asset.
    int pickIndex = 0;
    const int pandaIndex =
            m_testImageCombo->findText(QStringLiteral("cartoon_panda.png"));
    if (pandaIndex >= 0) pickIndex = pandaIndex;
    m_testImageCombo->setCurrentIndex(pickIndex);
    onTestImageSelected(pickIndex);
    appendLog(tr("[InstantMesh] %1 sample image(s) ready — pick one above.")
                      .arg(images.size()));
}

void InstantMeshDialog::onTestImageSelected(int index) {
    if (index < 0 || !m_testImageCombo) return;
    const QString path = m_testImageCombo->itemData(index).toString();
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    updateImagePreview();
}

void InstantMeshDialog::onTestDataExtracted() {
    using TestDataset = ecvTestDataRepository::Dataset;
    const TestDataset kind = TestDataset::Image2Mesh;
    setProgressIdle();
    if (!ecvTestDataRepository::getImage2MeshImages(
                 ecvTestDataRepository::extractPath(kind))
                 .isEmpty()) {
        populateTestImages();
    } else {
        appendLog(
                tr("[InstantMesh] Sample data extracted, but no images "
                   "found."));
    }
}

void InstantMeshDialog::onRun() {
    if (m_taskRunning || m_samplePreparing || m_downloadInProgress) return;
    if (m_imagePath->text().trimmed().isEmpty()) {
        appendLog(tr("[InstantMesh] Select an input image first."));
        return;
    }
    const QStringList missing = missingPresetFiles();
    if (!missing.isEmpty()) {
        appendLog(tr("[InstantMesh] Missing models, downloading: %1")
                          .arg(missing.join(", ")));
        m_pendingActionAfterDownload = PendingAction::Run;
        onDownloadModels();
        return;
    }
    saveSettings();
    resetStageStrip(m_lastPreviewImage);
    emit runRequested(getSettings());
}

void InstantMeshDialog::onCancel() { emit cancelRequested(); }

void InstantMeshDialog::appendLog(const QString& msg) {
    aicore_inference_log::log(msg);
}

void InstantMeshDialog::setProgressStage(int stageId,
                                         const QString& stage,
                                         int step,
                                         int total) {
    Q_UNUSED(stageId);
    if (!m_progress) return;
    m_stageLabel->setVisible(true);
    m_progress->setVisible(true);
    if (total > 0) {
        m_progress->setRange(0, 100);
        m_progress->setValue(int(100.0 * step / total));
        m_stageLabel->setText(tr("%1 (%2/%3)").arg(stage).arg(step).arg(total));
    } else {
        m_progress->setRange(0, 0);  // busy indicator
        m_stageLabel->setText(stage);
    }
}

void InstantMeshDialog::setRunning(bool running) {
    m_taskRunning = running;
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    m_useTestDataBtn->setEnabled(!running && !m_samplePreparing);
    if (!running) {
        setProgressIdle();
    }
}

void InstantMeshDialog::setImagePreview(const QImage& image) {
    if (image.isNull() || !m_imagePreview) return;
    m_imagePreview->setPreviewImage(image, m_imagePreview->size());
}

void InstantMeshDialog::resetStageStrip(const QImage& input) {
    const QStringList kNames = {tr("Source"), tr("RMBG"), tr("Views"),
                                tr("Mesh"), tr("GLB")};
    for (int i = 0; i < m_stageThumbs.size(); ++i) {
        if (m_stageCaptions[i]) m_stageCaptions[i]->setText(kNames[i]);
        // Drop the previous run's full image (stale-result leak) and show
        // the qTrellis "-" placeholder while the stage has not run yet.
        if (m_stageThumbs[i]) {
            m_stageThumbs[i]->clearPreview();
            m_stageThumbs[i]->setText(QStringLiteral("-"));
        }
        setStageState(i, kStagePending);
    }
    m_activeSlot = -1;
    if (!input.isNull() && m_stageThumbs[0]) {
        setImagePreviewToSlot(0, input);
    }
}

void InstantMeshDialog::setImagePreviewToSlot(int slot, const QImage& image) {
    if (slot < 0 || slot >= m_stageThumbs.size() || image.isNull()) return;
    // setPreviewImage stores the full-resolution image as the
    // click-to-enlarge target; a raw setPixmap leaves the chip inert.
    m_stageThumbs[slot]->setPreviewImage(image, m_stageThumbs[slot]->size());
}

void InstantMeshDialog::setStageState(int slot, int state) {
    if (slot < 0 || slot >= m_stageThumbs.size()) return;
    const QColor border =
            state == kStageDone
                    ? QColor(0x2e, 0x7d, 0x32)
                    : (state == kStageActive ? QColor(0xf5, 0x7f, 0x17)
                                             : QColor(0xbb, 0xbb, 0xbb));
    QString style = QStringLiteral("border: 2px solid %1; background: #fafafa;")
                            .arg(border.name());
    // The pending "-" placeholder reads dimmed; image chips carry no text.
    if (state == kStagePending) style += QStringLiteral(" color: #999;");
    m_stageThumbs[slot]->setStyleSheet(style);
}

void InstantMeshDialog::setStageActive(int stage) {
    // aicore_instantmesh_stage ids (the progress-callback contract, indexed
    // numerically like the qInstantMesh stage-name table): preprocess(0) /
    // rmbg(1) own the RMBG chip, multiview(2) / vae_decode(3) the Views
    // chip, encode(4)..mesh(7) the Mesh chip and texture(8) the GLB chip.
    int slot = -1;
    switch (stage) {
        case 0:
        case 1:
            slot = 1;
            break;
        case 2:
        case 3:
            slot = 2;
            break;
        case 4:
        case 5:
        case 6:
        case 7:
            slot = 3;
            break;
        case 8:
            slot = 4;
            break;
        default:
            return;
    }
    if (slot == m_activeSlot) return;  // per-step callbacks repeat a stage
    m_activeSlot = slot;
    setStageState(slot, kStageActive);
}

void InstantMeshDialog::applyResultToStrip(const InstantMeshRunResult& result) {
    if (m_stageThumbs.size() < 5) return;
    setLastResult(result);
    // A successful run must leave NO chip blank (the qTrellis rule): a
    // stage whose image the AICore result does not carry shows the closest
    // truthful image with an explicit state caption instead, so a blank
    // slot can never read as a broken stage.
    const QImage source = m_stageThumbs[0]->fullImage();
    // RMBG chip: the condition image the pipeline actually consumed — the
    // matting (click-zoom shows whether the background is cleanly gone)
    // when RMBG ran, the input itself otherwise.
    if (!result.preprocessImage.isNull()) {
        setImagePreviewToSlot(1, result.preprocessImage);
        m_stageCaptions[1]->setText(
                result.rmbgApplied ? tr("RMBG") : tr("Preprocess · no RMBG"));
    } else if (!source.isNull()) {
        setImagePreviewToSlot(1, source);
        m_stageCaptions[1]->setText(tr("Preprocess · no RMBG"));
    }
    setStageState(1, kStageDone);
    // Views chip: the real 6-view Zero123++ mosaic when the diffusion ran.
    if (!result.viewsImage.isNull()) {
        setImagePreviewToSlot(2, result.viewsImage);
        m_stageCaptions[2]->setText(tr("Views · Zero123++"));
    } else if (!result.multiview) {
        // Replicate mode: the input image IS the actual 6-view condition.
        if (!source.isNull()) setImagePreviewToSlot(2, source);
        m_stageCaptions[2]->setText(tr("Views · replicated input"));
    } else {
        // Diffusion ran but the mosaic export is absent: explicit caption
        // instead of a blank chip.
        if (!source.isNull()) setImagePreviewToSlot(2, source);
        m_stageCaptions[2]->setText(tr("Views · grid not exported"));
    }
    setStageState(2, kStageDone);
    if (!result.previewImage.isNull()) {
        setImagePreviewToSlot(3, result.previewImage);
        setStageState(3, kStageDone);
    } else {
        m_stageCaptions[3]->setText(tr("Mesh · render failed"));
    }
    if (!result.glb.isEmpty()) {
        if (!result.texRgba.isEmpty() && result.texW > 0 && result.texH > 0) {
            // The baked atlas is the GLB's actual texture content; zoom it
            // to inspect bake quality (row 0 = top matches QImage).
            const QImage atlas(
                    reinterpret_cast<const uchar*>(result.texRgba.constData()),
                    result.texW, result.texH, result.texW * 4,
                    QImage::Format_RGBA8888);
            setImagePreviewToSlot(4, atlas.copy());  // own the pixels
        } else if (!result.previewImage.isNull()) {
            setImagePreviewToSlot(4, result.previewImage);
        }
        m_stageCaptions[4]->setText(
                result.texRgba.isEmpty()
                        ? tr("GLB · vertex colours")
                        : tr("GLB · %1 MB")
                                  .arg(result.glb.size() / (1024.0 * 1024.0), 0,
                                       'f', 1));
        setStageState(4, kStageDone);
    } else if (!result.previewImage.isNull()) {
        // GLB bake failed: the mesh render with an explicit caption states
        // what happened instead of a blank chip.
        setImagePreviewToSlot(4, result.previewImage);
        m_stageCaptions[4]->setText(tr("GLB · none"));
    }
}

void InstantMeshDialog::openMesh3D(int slot) {
    if (slot < 0 || slot >= m_stageThumbs.size()) return;
    const QImage chip = m_stageThumbs[slot]->fullImage();
    if (m_lastResult.verts.isEmpty() || m_lastResult.tris.isEmpty()) {
        // No result mesh yet (mid-run / failed generation): keep the shared
        // image zoom as the fallback so the chip never feels dead.
        if (!chip.isNull()) {
            ecvClickableImageLabel::showEnlargedImage(
                    this, chip, m_stageCaptions[slot]->text());
        }
        return;
    }
    // InstantMesh meshes are Z-up (the Zero123++ cameras orbit Z); the
    // shared viewer orbits a Y-up scene, so rotate once about X:
    // (x, y, z) -> (x, z, -y).
    const int nv = m_lastResult.verts.size() / 3;
    QVector<float> verts(m_lastResult.verts);
    for (int i = 0; i < nv; ++i) {
        const float y = verts[3 * i + 1];
        const float z = verts[3 * i + 2];
        verts[3 * i + 1] = z;
        verts[3 * i + 2] = -y;
    }
    const QVector<float> normals =
            InstantMeshWorker::vertexNormals(verts, m_lastResult.tris);
    // Per-vertex base colour in the viewer's 6-float layout (rgb + unused
    // metallic/roughness/alpha); empty renders the amber geometry shading.
    QVector<float> colors;
    if (m_lastResult.vcolors.size() == m_lastResult.verts.size()) {
        colors.resize(nv * 6);
        for (int i = 0; i < nv; ++i) {
            for (int c = 0; c < 3; ++c)
                colors[6 * i + c] = m_lastResult.vcolors[3 * i + c];
        }
    }
    ecvAICoreMeshViewerDialog viewer(
            verts, normals, m_lastResult.tris, colors, !colors.isEmpty(),
            QStringLiteral("InstantMesh — %1")
                    .arg(m_stageCaptions[slot]->text()),
            this);
    viewer.exec();
}

void InstantMeshDialog::closeEvent(QCloseEvent* event) {
    m_acceptDataResults = false;
    m_pendingActionAfterDownload = PendingAction::None;
    if (m_taskRunning) emit cancelRequested();
    saveSettings();
    QDialog::closeEvent(event);
}

void InstantMeshDialog::reject() {
    // closeEvent already detached and requested cancellation before Qt rejects.
    const bool cancel = m_acceptDataResults && m_taskRunning;
    m_acceptDataResults = false;
    m_pendingActionAfterDownload = PendingAction::None;
    if (cancel) emit cancelRequested();
    QDialog::reject();
}

void InstantMeshDialog::keyPressEvent(QKeyEvent* event) {
    if (event->matches(QKeySequence::Cancel) && m_taskRunning) {
        onCancel();
        event->accept();
        return;
    }
    QDialog::keyPressEvent(event);
}

void InstantMeshDialog::showEvent(QShowEvent* event) {
    m_acceptDataResults = true;
    QDialog::showEvent(event);
    m_useTestDataBtn->setEnabled(!m_samplePreparing && !m_taskRunning);
    if (!m_samplePreparing && !m_taskRunning && !m_downloadInProgress) {
        setProgressIdle();
    }
    if (m_firstShow) {
        m_firstShow = false;
        refreshModelState();
        ensureImage2MeshDataset();
        // Polish-time font metrics can grow the layout minimum by a pixel
        // and widen the content; re-pin so the floor matches the shown
        // layout exactly.
        setMinimumSize(pinnedMinimumSize().expandedTo(minimumSize()));
    }
}
