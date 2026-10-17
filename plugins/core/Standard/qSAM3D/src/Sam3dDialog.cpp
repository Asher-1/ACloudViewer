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
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <cstring>

#include "aicore/sam3_capi.h"
#include "aicore/sam3d_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

namespace {

// Shared SAM3 model cache root (aicore_sam3_model_cache_dir), used by the
// segmentation-model combobox annotations and the download queue.
QString sam3ModelCacheDir() {
    const char* cache = aicore_sam3_model_cache_dir();
    return cache ? QString::fromUtf8(cache) : QString();
}

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

// Build capability probe: the official PBR chain is a compile-scoped AICore
// property (the same gate as the runtime AICORE_SAM3D_CAP_PBR bit a
// CUDA-session context reports), so the dialog can default its control from
// the static info JSON without constructing a context or resolving a
// backend. A machine without a CUDA device still fails the request
// explicitly at session run start.
bool sam3dBuildHasPbrChain() {
    const char* info = aicore_sam3d_info_json();
    return info && std::strstr(info, "\"pbr\":true") != nullptr;
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

    // DB tree output selection (pipeline gating): the colored point cloud
    // is the mandatory generation product; the textured mesh extends the
    // pipeline with the FlexiCubes decode + UV-atlas bake stages; the
    // official PBR bake replaces that projection bake when the AICore build
    // ships it.
    form->addRow(QString(), new QLabel(tr("DB tree output:"), this));
    m_pointCloudCheck =
            new QCheckBox(tr("Gaussian point cloud (colored splats)"), this);
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

    m_pbrCheck = new QCheckBox(
            tr("Official PBR texture bake (CUDA, ~10 s/object)"), this);
    m_pbrCheck->setChecked(sam3dBuildHasPbrChain());
    m_pbrCheck->setToolTip(
            tr("Replaces the projection bake with the official native PBR\n"
               "chain inside AICore: VTK quadric simplify → visibility\n"
               "min-cut cleanup → pymeshfix repair → xatlas UV → 100-view\n"
               "Gaussian render → 2500-step optimized base-color atlas →\n"
               "Telea inpaint → Y-up GLB. On by default when this AICore\n"
               "build ships the official chain; disabled entirely when it\n"
               "does not. Requires a CUDA device at run time — otherwise\n"
               "the run fails with an explicit error."));
    form->addRow(QString(), m_pbrCheck);
    // PBR bake quality knobs (reconstruction side — deliberately separate
    // from the SAM3 segmentation filters): 0 keeps the AICore defaults
    // (library-chosen texture size, 2500 optimization steps). Lower steps
    // trade bake precision for speed (~20 ms/step per object on a
    // mid-range GPU); a smaller texture trades sharpness for VRAM/time.
    auto* pbrParamsRow = new QHBoxLayout;
    m_pbrTextureCombo = new QComboBox(this);
    m_pbrTextureCombo->addItem(tr("Default"), 0);
    m_pbrTextureCombo->addItem(QStringLiteral("512"), 512);
    m_pbrTextureCombo->addItem(QStringLiteral("1024"), 1024);
    m_pbrTextureCombo->addItem(QStringLiteral("2048"), 2048);
    m_pbrTextureCombo->setToolTip(
            tr("Atlas texture resolution; Default lets AICore choose."));
    m_pbrStepsSpin = new QSpinBox(this);
    m_pbrStepsSpin->setRange(0, 5000);
    m_pbrStepsSpin->setSingleStep(250);
    m_pbrStepsSpin->setValue(0);
    m_pbrStepsSpin->setSpecialValueText(tr("Default (2500)"));
    m_pbrStepsSpin->setToolTip(
            tr("Optimization steps of the official base-color bake: fewer\n"
               "steps finish proportionally faster with slightly softer\n"
               "textures; Default (2500) matches the upstream quality."));
    pbrParamsRow->addWidget(new QLabel(tr("Texture:"), this));
    pbrParamsRow->addWidget(m_pbrTextureCombo);
    pbrParamsRow->addWidget(new QLabel(tr("Bake steps:"), this));
    pbrParamsRow->addWidget(m_pbrStepsSpin);
    pbrParamsRow->addStretch(1);
    form->addRow(tr("PBR bake:"), pbrParamsRow);
    // The knobs track the PBR checkbox's usable state (build carries the
    // chain AND the textured-mesh stage is selected AND PBR is checked).
    auto updatePbrParamsEnabled = [this]() {
        const bool pbrUsable =
                m_pbrCheck->isEnabled() && m_pbrCheck->isChecked();
        m_pbrTextureCombo->setEnabled(pbrUsable);
        m_pbrStepsSpin->setEnabled(pbrUsable);
    };
    connect(m_pbrCheck, &QCheckBox::toggled, this, updatePbrParamsEnabled);
    connect(m_texturedMeshCheck, &QCheckBox::toggled, this,
            updatePbrParamsEnabled);
    updatePbrParamsEnabled();
    // PBR availability + dependency wiring: without an AICore build that
    // carries the license-gated official chain the PBR option stays off and
    // disabled — the run then produces the point cloud plus (when selected)
    // the CPU-baked textured mesh. The PBR bake is a mesh producer, so it
    // also needs the textured-mesh stage: clearing that stage disables it,
    // and enabling it re-selects the mesh.
    if (!sam3dBuildHasPbrChain()) {
        m_pbrCheck->setChecked(false);
        m_pbrCheck->setEnabled(false);
    }
    connect(m_texturedMeshCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_pbrCheck->setEnabled(on && sam3dBuildHasPbrChain());
        if (!on) m_pbrCheck->setChecked(false);
    });
    connect(m_pbrCheck, &QCheckBox::toggled, this, [this](bool on) {
        if (on) m_texturedMeshCheck->setChecked(true);
    });

    // Scene-mode diagnostic: the masks-over-image preview tells the user
    // which objects the segmentation accepted before the (much longer)
    // per-object runs start. Meaningless outside scene mode → disabled.
    m_segPreviewCheck = new QCheckBox(
            tr("SAM3 segmentation preview (masks over image)"), this);
    m_segPreviewCheck->setChecked(true);
    m_segPreviewCheck->setToolTip(
            tr("Scene mode only: composites every accepted object mask over\n"
               "the source image (one color + label per object) and imports\n"
               "the preview into the DB tree before the per-object runs\n"
               "start. Use it to check which objects the prompt set actually\n"
               "detected — objects missing from the reconstruction are\n"
               "almost always segmentation misses (widen the prompt set or\n"
               "lower the score threshold), not reconstruction failures."));
    form->addRow(QString(), m_segPreviewCheck);

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
    // The segmentation preview is a scene-mode artifact: masks exist only
    // when the run segments or loads per-object masks.
    m_segPreviewCheck->setEnabled(m_sceneCheck->isChecked());
    connect(m_sceneCheck, &QCheckBox::toggled, m_segPreviewCheck,
            &QCheckBox::setEnabled);

    // Concept prompts (automatic segmentation): when non-empty, the AICore
    // sam3 task segments every prompt's instances directly from the image —
    // no hand-made masks (fully end-to-end). The default set covers the
    // common indoor/kidsroom concepts of the official scene bundle. The
    // explicit mask directory stays as the manual/interop fallback.
    auto* promptsRow = new QHBoxLayout;
    m_promptsEdit = new QLineEdit(this);
    m_promptsEdit->setPlaceholderText(
            tr("Comma-separated SAM3 prompts (empty = use the mask "
               "directory)"));
    m_promptsEdit->setText(
            QStringLiteral("table, chair, bed, shelf, lamp, pillow, "
                           "cushion, pouf, rug, picture frame, wall hanging, "
                           "potted plant, toy, stuffed animal, doll, "
                           "toy bird, bird, ball, block, box, basket, bag, "
                           "laundry bag, shoe"));
    m_promptsEdit->setToolTip(
            tr("One instance set per prompt: every prompt is segmented on\n"
               "the source image with the AICore sam3 task (text-prompted\n"
               "multi-instance detection). Detections are deduplicated by\n"
               "greedy IoU and score order, then each instance mask drives\n"
               "its own single-object reconstruction. Each prompt costs one\n"
               "detection pass — keep the list scene-specific. The default\n"
               "set was probe-tuned on the official kidsroom demo scene:\n"
               "all 27 demo objects (bag, laundry bag, pouf, shoe, toy\n"
               "bird, block…) are covered by at least one prompt. For\n"
               "exact parity with a set of pre-made masks, point the\n"
               "Masks field at their directory instead."));
    promptsRow->addWidget(m_promptsEdit, 1);
    form->addRow(tr("Prompts:"), promptsRow);

    // Detection filters — the SAM3 segmentation chain's own knobs, kept
    // separate from the sam3d reconstruction parameters (quantization /
    // steps / seed, which have no score/IoU concept): score floors the
    // per-detection confidence (recall filter), NMS IoU suppresses
    // overlapping boxes within one prompt, and dedup IoU rejects a
    // detection from a DIFFERENT prompt that covers an already accepted
    // object. Scene mode only.
    m_samScoreSpin = new QDoubleSpinBox(this);
    m_samScoreSpin->setRange(0.0, 1.0);
    m_samScoreSpin->setSingleStep(0.05);
    m_samScoreSpin->setDecimals(2);
    m_samScoreSpin->setValue(0.0);
    m_samScoreSpin->setToolTip(
            tr("Minimum SAM3 detection score (recall filter): raise it to "
               "drop low-confidence objects; 0 = the AICore default (0.5)."));
    m_samNmsSpin = new QDoubleSpinBox(this);
    m_samNmsSpin->setRange(0.0, 1.0);
    m_samNmsSpin->setSingleStep(0.05);
    m_samNmsSpin->setDecimals(2);
    m_samNmsSpin->setValue(0.0);
    m_samNmsSpin->setToolTip(
            tr("NMS IoU within ONE prompt: detections overlapping more than "
               "this are merged; 0 = the AICore default."));
    m_samDedupSpin = new QDoubleSpinBox(this);
    m_samDedupSpin->setRange(0.0, 1.0);
    m_samDedupSpin->setSingleStep(0.05);
    m_samDedupSpin->setDecimals(2);
    m_samDedupSpin->setValue(0.50);
    m_samDedupSpin->setToolTip(
            tr("Dedup IoU ACROSS prompts: a detection overlapping an already "
               "accepted object more than this is dropped as its duplicate "
               "(different prompts hitting the same object). Lower it to "
               "keep tightly packed neighbours (e.g. felt toys on one "
               "mobile) as separate objects."));
    auto* segRow = new QHBoxLayout;
    segRow->addWidget(new QLabel(tr("Score:"), this));
    segRow->addWidget(m_samScoreSpin);
    segRow->addWidget(new QLabel(tr("NMS IoU:"), this));
    segRow->addWidget(m_samNmsSpin);
    segRow->addWidget(new QLabel(tr("Dedup IoU:"), this));
    segRow->addWidget(m_samDedupSpin);
    segRow->addStretch(1);
    form->addRow(tr("SAM3 filters:"), segRow);

    // SAM3 segmentation model (AICore sam3 catalog, text-prompt capable
    // family only — the visual-only variants have no tokenizer, so they
    // cannot drive text prompts). Best default = q8_0 (catalog note: best
    // accuracy/size trade). Cached entries are annotated in-place.
    auto* sam3ModelRow = new QHBoxLayout;
    m_sam3ModelCombo = new QComboBox(this);
    const QString sam3Cache = sam3ModelCacheDir();
    for (int i = 0; i < aicore_sam3_model_count(); ++i) {
        const aicore_sam3_model_entry* e = aicore_sam3_model_at(i);
        if (e == nullptr || e->visual_only) continue;
        const QString filename = QString::fromUtf8(e->filename);
        const QString quant = filename.section('-', 1).section('.', 0, 0);
        const bool cached =
                QFileInfo::exists(QDir(sam3Cache).filePath(filename));
        m_sam3ModelCombo->addItem(
                QStringLiteral("%1 — %2%3")
                        .arg(quant.toUpper(),
                             ecvModelDownloader::formatFileSize(e->size_bytes),
                             cached ? QStringLiteral("  (cached)") : QString()),
                filename);
    }
    const int q8Index =
            m_sam3ModelCombo->findData(QStringLiteral("sam3-q8_0.gguf"));
    m_sam3ModelCombo->setCurrentIndex(q8Index >= 0 ? q8Index : 0);
    m_sam3ModelCombo->setToolTip(
            tr("SAM3 model quantization for the text-prompt segmentation.\n"
               "q8_0 (default) is the catalog's best accuracy/size trade;\n"
               "q4_0/q4_1 are lighter and faster; f16 is the reference.\n"
               "Entries marked (cached) are already in the shared model\n"
               "cache; the rest download on demand through Download models."));
    sam3ModelRow->addWidget(m_sam3ModelCombo, 1);
    form->addRow(tr("SAM3 model:"), sam3ModelRow);

    auto updateMasksEnabled = [this]() {
        const bool scene = m_sceneCheck->isChecked();
        m_masksDirEdit->setEnabled(scene);
        m_browseMasksBtn->setEnabled(scene);
        m_promptsEdit->setEnabled(scene);
        m_samScoreSpin->setEnabled(scene);
        m_samNmsSpin->setEnabled(scene);
        m_samDedupSpin->setEnabled(scene);
        m_sam3ModelCombo->setEnabled(scene);
        // The preview button MUST follow the scene checkbox here, not only
        // setRunning(): it is created disabled, and setRunning() runs only
        // when a task starts/ends — without this line the button stayed
        // grey forever and the two-step workflow was unreachable.
        m_previewMasksBtn->setEnabled(scene && !m_running);
    };
    connect(m_sceneCheck, &QCheckBox::toggled, this, updateMasksEnabled);

    // Output directory (only used by the opt-in file exports) --------------
    auto* outRow = new QHBoxLayout;
    m_outputDirEdit = new QLineEdit(this);
    m_outputDirEdit->setPlaceholderText(
            tr("GLB / Gaussian PLY export directory (empty = no file "
               "export)"));
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

    // Two-step scene workflow, step 1: segmentation + preview only. Sits
    // above the run buttons so the iterate-on-prompts loop is obvious:
    // preview → adjust prompts → preview → … → Generate.
    m_previewMasksBtn = new QPushButton(
            tr("Generate mask preview only (no reconstruction)"), this);
    m_previewMasksBtn->setToolTip(
            tr("Scene mode: runs only the SAM3 segmentation and imports the\n"
               "masks-over-image preview into the DB tree — no per-object\n"
               "reconstruction. Adjust the prompts until the preview covers\n"
               "every object you want, then press Generate for the full\n"
               "pipeline."));
    m_previewMasksBtn->setEnabled(false);
    layout->addWidget(m_previewMasksBtn);

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
    connect(m_previewMasksBtn, &QPushButton::clicked, this,
            &Sam3dDialog::emitPreviewMasks);
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

    // Initial enabled/disabled state. This touches EVERY widget the lambda
    // references — including m_previewMasksBtn, created above the run
    // buttons — so it must run only after all of them are constructed:
    // calling it next to the lambda definition dereferenced a
    // not-yet-created m_previewMasksBtn and segfaulted the dialog on open.
    updateMasksEnabled();
    restoreDefaults();
}

Sam3dDialog::~Sam3dDialog() = default;

void Sam3dDialog::restoreDefaults() {
    // No silent default: file exports are opt-in and require the user to
    // pick the directory explicitly (an empty field = no file export).
    m_outputDirEdit->clear();
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
    // One instance set per comma-separated prompt; empties dropped.
    s.conceptPrompts.clear();
    if (m_sceneCheck->isChecked()) {
        const QStringList parts = m_promptsEdit->text().split(',');
        for (QString part : parts) {
            part = part.trimmed();
            if (!part.isEmpty()) s.conceptPrompts << part;
        }
    }
    s.samScoreThreshold = static_cast<float>(m_samScoreSpin->value());
    s.samNmsThreshold = static_cast<float>(m_samNmsSpin->value());
    s.samDedupIouThreshold = static_cast<float>(m_samDedupSpin->value());
    s.device = m_deviceCombo->currentText();
    s.dtypeIndex = m_dtypeCombo->currentIndex();
    s.rmbgDtypeIndex = m_rmbgDtypeCombo->currentIndex();
    s.steps = m_stepsSpin->value();
    s.seed = m_seedSpin->value();
    s.useRmbg = m_rmbgCheck->isChecked();
    s.outputPointCloud = m_pointCloudCheck->isChecked();
    s.pbrBake = m_pbrCheck->isChecked();
    s.pbrTextureSize = m_pbrTextureCombo->currentData().toInt();
    s.pbrBakeSteps = m_pbrStepsSpin->value();
    // The PBR bake is a textured-mesh producer: checking it implies the
    // textured-mesh stage in the pipeline gating below.
    s.outputTexturedMesh =
            m_texturedMeshCheck->isChecked() || m_pbrCheck->isChecked();
    s.segPreview = m_segPreviewCheck->isChecked();
    s.sam3ModelFile = m_sam3ModelCombo->currentData().toString();
    s.importToDb = m_importDbCheck->isChecked();
    s.exportPly = m_exportPlyCheck->isChecked();
    return s;
}

void Sam3dDialog::appendLog(const QString& message,
                            ecvMainAppInterface::ConsoleMessageLevel level) {
    // Reuse the application console (same channel as every other plugin);
    // the dialog itself only keeps the single-line status label. A missing
    // app interface must never swallow messages silently — failure
    // reporting rides on this channel.
    if (m_app) {
        m_app->dispToConsole(message, level);
    } else {
        qWarning("%s", message.toUtf8().constData());
    }
}

void Sam3dDialog::setProgress(int percent) {
    m_progress->setRange(0, 100);
    m_progress->setValue(percent);
}

void Sam3dDialog::emitPreviewMasks() {
    const Settings s = settings();
    if (s.imagePath.isEmpty()) {
        m_status->setText(tr("Select a source image first."));
        return;
    }
    if (s.conceptPrompts.isEmpty() && s.masksDir.isEmpty()) {
        m_status->setText(tr("Scene mode needs prompts or a mask directory."));
        return;
    }
    Settings preview = s;
    preview.previewOnly = true;
    preview.segPreview = true;
    emit previewMasksRequested(preview);
}

void Sam3dDialog::setRunning(bool running, const QString& statusText) {
    m_running = running;
    m_runButton->setEnabled(!running);
    m_previewMasksBtn->setEnabled(!running && m_sceneCheck->isChecked());
    m_cancelButton->setEnabled(running);
    m_progress->setRange(0, running ? 0 : 1);
    m_progress->setValue(running ? 0 : 1);
    m_status->setText(running ? (statusText.isEmpty() ? tr("Running the SAM 3D "
                                                           "pipeline...")
                                                      : statusText)
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
    if (s.exportPly && s.outputDir.isEmpty()) {
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
    // The selected SAM3 segmentation model (scene-mode dependency) rides
    // the same queue: URL/size come from the AICore sam3 catalog.
    const QString sam3File = m_sam3ModelCombo->currentData().toString();
    if (!sam3File.isEmpty() &&
        !QFileInfo::exists(QDir(sam3ModelCacheDir()).filePath(sam3File))) {
        m_pendingDownloads << sam3File;
    }

    if (m_pendingDownloads.isEmpty()) {
        appendLog(tr("[SAM3D] Nothing to download (%1 / %2 cache).")
                          .arg(dtype, cacheDir));
        return;
    }
    qint64 totalBytes = 0;
    for (const QString& name : m_pendingDownloads) {
        if (const aicore_sam3d_model_entry* entry =
                    aicore_sam3d_model_by_filename(name.toUtf8().constData())) {
            totalBytes += entry->size_bytes;
        } else if (const aicore_sam3_model_entry* sam3Entry =
                           aicore_sam3_model_by_filename(
                                   name.toUtf8().constData())) {
            totalBytes += sam3Entry->size_bytes;
        }
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
    // The AICore catalogs supply the download URL and size for every
    // published model; qSAM3D deliberately owns no duplicate model table.
    // The SAM3 segmentation catalog carries no per-file SHA-256, so its
    // finalize gates on the GGUF magic + the catalog size instead.
    QString url;
    QString dest;
    qint64 minBytes = 1024 * 1024;
    ecvAssetIntegrity::Anchor anchor{QCryptographicHash::Sha256, QByteArray()};
    if (const aicore_sam3d_model_entry* entry =
                aicore_sam3d_model_by_filename(filename.toUtf8().constData())) {
        url = QString::fromUtf8(entry->download_url);
        dest = QDir(sam3dModelCacheDir()).filePath(filename);
        anchor = {QCryptographicHash::Sha256, QByteArray(entry->sha256)};
    } else if (const aicore_sam3_model_entry* sam3Entry =
                       aicore_sam3_model_by_filename(
                               filename.toUtf8().constData())) {
        url = QString::fromUtf8(sam3Entry->download_url);
        dest = QDir(sam3ModelCacheDir()).filePath(filename);
        minBytes = std::max<qint64>(minBytes, sam3Entry->size_bytes);
    } else {
        appendLog(tr("[SAM3D] Model is absent from the AICore catalogs: %1")
                          .arg(filename));
        downloadNextModel();
        return;
    }
    appendLog(tr("[SAM3D] Downloading %1...\n  URL: %2\n  Dest: %3")
                      .arg(filename, url, dest));
    m_downloadInProgress = true;
    ecvModelDownloader::Request req;
    req.url = url;
    req.destPath = dest;
    req.minBytes = minBytes;
    req.contentAnchor = anchor;  // streamed content check (sam3d models)
    m_downloader->download(req);
}
