// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LingbotMapDialog.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMessageBox>
#include <QSettings>
#include <QStandardItemModel>
#include <QVBoxLayout>

#include "aicore/asset_digests.h"
#include "aicore/lingbot_capi.h"
#include "cvFileDialog.h"
#include "ecvAICoreUiHelper.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {

struct LingbotCatalogEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    QString role;
    qint64 sizeBytes = 0;
};

// Role-filtered catalog view over the AICore LingBot-Map catalog.
QVector<LingbotCatalogEntry> catalogByRole(const char* role) {
    QVector<LingbotCatalogEntry> out;
    const int n = aicore_lingbot_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_lingbot_model_entry* e = aicore_lingbot_model_at(i);
        if (!e || !e->filename || !e->role || std::strcmp(e->role, role) != 0) {
            continue;
        }
        LingbotCatalogEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.role = QString::fromUtf8(e->role);
        entry.sizeBytes = static_cast<qint64>(e->size_bytes);
        out.append(entry);
    }
    return out;
}

bool findEntryByFilename(const QVector<LingbotCatalogEntry>& catalog,
                         const QString& filename,
                         LingbotCatalogEntry* out) {
    for (const LingbotCatalogEntry& e : catalog) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString lingbotModelCacheDir() {
    char* raw = aicore_lingbot_model_cache_dir();
    if (!raw) return QString();
    const QString dir = QString::fromUtf8(raw);
    aicore_lingbot_free_buffer(raw);
    return dir;
}

// First-run defaults keyed by the total memory of the best GPU on the
// machine. Large cards get the upstream GUI default: the f16 GGUF
// (full-alignment format, pose 1.72e-04 / depth 4.72e-04 vs the official
// fp32 checkpoint over 286 frames) with the official release KV profile.
// Smaller cards keep the official memory-saving q8 deployment format and
// downscale the profile instead (q8 trades one order of magnitude of
// accuracy for half the weight memory). 0 bytes (no accelerator or a
// CPU-only run) follows the upstream default on host memory. Users can
// override anything; the choice persists and the auto tier never fires
// again.
struct LingbotKvTier {
    int minTotalGiB;
    int scale;
    int window;
    const char* model;  // catalog filename of the tier's default map GGUF
};
constexpr LingbotKvTier kLingbotKvTiers[] = {
        // Thresholds sit below the nominal card size (a "12 GB" card reports
        // ~11.7 GiB, a "24 GB" card ~23.5 GiB) so nominal tiers land right.
        {22, 8, 64, "lingbot-map-f16.gguf"},  // ≥22 GiB: official release
                                              // profile + full-alignment model
        {10, 4, 32, "lingbot-map-q8.gguf"},   // 12-GB class: memory default
        {6, 2, 16, "lingbot-map-q8.gguf"},    // 8-GB class
        {0, 2, 16, "lingbot-map-q8.gguf"},    // smaller accelerators
};

void lingbotAutoKvProfile(uint64_t gpuTotalBytes,
                          int* scale,
                          int* window,
                          const char** model) {
    if (gpuTotalBytes == 0) {
        // No accelerator resolved (CPU-only run): host memory is not the
        // binding constraint — follow the upstream GUI default (f16) with
        // the official release profile.
        *scale = 8;
        *window = 64;
        *model = "lingbot-map-f16.gguf";
        return;
    }
    const int gib = static_cast<int>(gpuTotalBytes / (1024ull * 1024 * 1024));
    for (const LingbotKvTier& tier : kLingbotKvTiers) {
        if (gib >= tier.minTotalGiB) {
            *scale = tier.scale;
            *window = tier.window;
            *model = tier.model;
            return;
        }
    }
    *scale = 8;
    *window = 64;
    *model = "lingbot-map-f16.gguf";
}

}  // namespace

LingbotMapDialog::LingbotMapDialog(QWidget* parent)
    : QDialog(parent, Qt::Window) {
    setWindowTitle(tr("LingBot-Map Streaming 3D Reconstruction"));
    // Application-modal dialog mode: while open, the dialog stays above the
    // main window and clicking elsewhere can no longer raise the main window
    // over it (a plain modeless Qt::Window gets buried and appears hidden).
    // Esc or the close button releases the modality; the streaming task
    // itself keeps running in its worker thread.
    setWindowModality(Qt::ApplicationModal);
    resize(640, 780);

    auto* rootLayout = new QVBoxLayout(this);
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    // --- model catalog (map role) ---
    m_modelCombo = new QComboBox(this);
    form->addRow(tr("Model"), m_modelCombo);

    // Hidden rows keep their controls inside the row widget with an
    // in-row label: a plain QFormLayout label would stay visible when only
    // the field widget is hidden.
    m_customModelRow = new QWidget(this);
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    customLayout->setContentsMargins(0, 0, 0, 0);
    customLayout->addWidget(
            new QLabel(tr("Custom GGUF model:"), m_customModelRow), 0);
    m_customModelPath = new QLineEdit(m_customModelRow);
    auto* browseModelBtn = new QPushButton(tr("Browse…"), m_customModelRow);
    customLayout->addWidget(m_customModelPath, 1);
    customLayout->addWidget(browseModelBtn);
    m_customModelRow->setVisible(false);
    form->addRow(QString(), m_customModelRow);

    // --- sky masking source (none / native skyseg / cached test masks) ---
    m_skySourceCombo = new QComboBox(this);
    m_skySourceCombo->addItem(tr("None"), QStringLiteral("none"));
    m_skySourceCombo->addItem(tr("Native skyseg (--mask_sky)"),
                              QStringLiteral("native"));
    m_skySourceCombo->addItem(tr("Cached test masks"),
                              QStringLiteral("cached"));
    // SkySeg GGUF picker: the published catalog carries f16 (recommended,
    // bit-exact vs the official onnxruntime path), q8_0 and f32; missing
    // files auto-download with pinned SHA-256 ingestion.
    m_skysegCombo = new QComboBox(this);
    m_skysegCombo->setToolTip(
            tr("SkySeg GGUF variant: f16 is the recommended release profile; "
               "q8_0 halves the download size, f32 is the exact reference."));
    form->addRow(tr("Sky masking"), m_skySourceCombo);
    m_skysegRow = new QWidget(this);
    auto* skysegLayout = new QHBoxLayout(m_skysegRow);
    skysegLayout->setContentsMargins(0, 0, 0, 0);
    skysegLayout->addWidget(new QLabel(tr("Skyseg model:"), m_skysegRow), 0);
    skysegLayout->addWidget(m_skysegCombo, 1);
    form->addRow(QString(), m_skysegRow);

    // --- test data row (cached download + auto-fill, lingbot_map_data) ---
    m_testSceneCombo = new QComboBox(this);
    connect(m_testSceneCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { applyTestSceneSelection(); });
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Download (cached) the selected official LingBot-Map demo "
               "sequence and auto-fill the image folder"));
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::requestTestData);
    connect(m_skySourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) {
                if (!m_skyAutoSwitching) m_skySourceUserSet = true;
                refreshSkySourceOptions();
            });
    auto* testRow = new QHBoxLayout;
    testRow->addWidget(m_testSceneCombo, 1);
    testRow->addWidget(m_testDataBtn);
    form->addRow(tr("Test data:"), testRow);

    // Cached sky-mask directory (auto-filled by the test-data flow; a
    // custom directory works too — masks must be <frame-stem>.png with
    // 255 = keep at the processed resolution).
    m_skyMaskRow = new QWidget(this);
    auto* maskRowLayout = new QHBoxLayout(m_skyMaskRow);
    maskRowLayout->setContentsMargins(0, 0, 0, 0);
    m_skyMaskDir = new QLineEdit(m_skyMaskRow);
    auto* browseMaskBtn = new QPushButton(tr("Browse…"), m_skyMaskRow);
    maskRowLayout->addWidget(m_skyMaskDir, 1);
    maskRowLayout->addWidget(browseMaskBtn);
    connect(browseMaskBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseSkyMaskDir);
    form->addRow(tr("Mask folder"), m_skyMaskRow);
    m_skyMaskRow->setVisible(false);

    // Custom-data reminder: outdoor sequences with sky in view should enable
    // sky masking (the official test scenes toggle it automatically).
    m_customSkyHintText =
            tr("Tip: for custom outdoor data with sky in view, enable Sky "
               "masking (Native skyseg or cached masks) to filter sky points.");
    m_skyHint = ecvAICoreUi::makeHintLabel(m_customSkyHintText, this);
    m_skyHint->setVisible(false);
    form->addRow(QString(), m_skyHint);

    // Dataset → model guidance (tips/label): tells the user which model
    // and pipeline each dataset is adapted for, per the upstream long_real
    // campaign.
    m_datasetHint = ecvAICoreUi::makeHintLabel(QString(), this);
    m_datasetHint->setWordWrap(true);
    form->addRow(QString(), m_datasetHint);

    // --- input source: image folder (default) or video file ---
    m_inputModeCombo = new QComboBox(this);
    m_inputModeCombo->addItem(tr("Image folder (ordered sequence)"),
                              QStringLiteral("folder"));
#ifdef HAS_OPENCV_FACE_CAPTURE
    m_inputModeCombo->addItem(tr("Video file (sampled at fps)"),
                              QStringLiteral("video"));
#endif
    m_inputModeCombo->setToolTip(
            tr("Upstream --image_folder / --video_path parity: video frames "
               "are sampled at the fps below (interval = round(source fps / "
               "fps), source fps falls back to 30)."));
    form->addRow(tr("Input source"), m_inputModeCombo);
    connect(m_inputModeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshInputRows(); });

    // --- input ---
    m_folderRow = new QWidget(this);
    auto* folderLayout = new QHBoxLayout(m_folderRow);
    folderLayout->setContentsMargins(0, 0, 0, 0);
    m_imageFolder = new QLineEdit(m_folderRow);
    auto* browseFolderBtn = new QPushButton(tr("Browse…"), m_folderRow);
    folderLayout->addWidget(m_imageFolder, 1);
    folderLayout->addWidget(browseFolderBtn);
    form->addRow(tr("Image folder"), m_folderRow);

#ifdef HAS_OPENCV_FACE_CAPTURE
    m_videoRow = new QWidget(this);
    auto* videoLayout = new QHBoxLayout(m_videoRow);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    videoLayout->addWidget(new QLabel(tr("Video file:"), m_videoRow), 0);
    m_videoPath = new QLineEdit(m_videoRow);
    auto* browseVideoBtn = new QPushButton(tr("Browse…"), m_videoRow);
    videoLayout->addWidget(m_videoPath, 1);
    videoLayout->addWidget(browseVideoBtn);
    m_videoRow->setVisible(false);
    form->addRow(QString(), m_videoRow);

    m_videoFpsRow = new QWidget(this);
    auto* fpsLayout = new QHBoxLayout(m_videoFpsRow);
    fpsLayout->setContentsMargins(0, 0, 0, 0);
    fpsLayout->addWidget(new QLabel(tr("Sampling fps:"), m_videoFpsRow), 0);
    m_videoFps = new QSpinBox(m_videoFpsRow);
    m_videoFps->setRange(1, 60);
    m_videoFps->setValue(10);
    m_videoFps->setToolTip(
            tr("Sampling rate for the video input (upstream --fps; default "
               "10)."));
    fpsLayout->addWidget(m_videoFps, 1);
    m_videoFpsRow->setVisible(false);
    form->addRow(QString(), m_videoFpsRow);
    connect(browseVideoBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseVideoFile);
#endif

    m_maxFrames = new QSpinBox(this);
    m_maxFrames->setRange(0, 1000000);
    m_maxFrames->setValue(0);
    m_maxFrames->setSpecialValueText(tr("all"));
    m_maxFrames->setToolTip(
            tr("Limit the stream to the first N frames (upstream --frames; "
               "default: all frames in the folder)."));
    form->addRow(tr("Max frames"), m_maxFrames);

    m_imageSize = new QSpinBox(this);
    m_imageSize->setRange(224, 1022);
    m_imageSize->setValue(518);
    m_imageSize->setSingleStep(14);
    m_imageSize->setToolTip(
            tr("Official crop width (upstream --image_size; default 518, "
               "snapped to the 14-px patch grid). Height follows the "
               "aspect ratio."));
    form->addRow(tr("Processing width"), m_imageSize);

    // --- backend ---
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->addItem(tr("Auto"), QStringLiteral("auto"));
    m_deviceCombo->addItem(QStringLiteral("CPU"), QStringLiteral("cpu"));
    m_deviceCombo->addItem(QStringLiteral("GPU"), QStringLiteral("gpu"));
    m_deviceCombo->setToolTip(
            tr("Auto follows the AICore runtime device order (CUDA → Vulkan "
               "→ CPU on Linux); upstream --backend."));
    form->addRow(tr("Device"), m_deviceCombo);

    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 256);
    m_threads->setValue(0);
    m_threads->setSpecialValueText(tr("default"));
    m_threads->setToolTip(
            tr("CPU thread count (upstream --threads; default: backend "
               "auto)."));
    form->addRow(tr("CPU threads"), m_threads);

    m_confThreshold = new QDoubleSpinBox(this);
    m_confThreshold->setRange(0.0, 100.0);
    m_confThreshold->setSingleStep(0.1);
    m_confThreshold->setValue(1.5);
    m_confThreshold->setToolTip(
            tr("Visibility confidence filter (upstream --conf_threshold; "
               "default 1.5)."));
    form->addRow(tr("Confidence threshold"), m_confThreshold);

    // --- advanced engine options (upstream ggml_demo parity) ---
    // First-run defaults: auto-profile the KV cache AND the map-model format
    // to the machine's GPU memory (skipped once anything has been
    // persisted). Large cards follow the upstream GUI default (f16,
    // full-alignment); small cards keep the official memory-saving q8 and
    // downscale the profile instead.
    bool kvAutoProfiled = false;
    uint64_t gpuTotalBytes = 0;
    int kvDefaultScale = 8, kvDefaultWindow = 64;
    QString modelDefault;
    {
        QSettings probe;
        probe.beginGroup(QStringLiteral("qLingbotMap"));
        const bool kvPersisted = probe.contains(QStringLiteral("kvScale")) &&
                                 probe.contains(QStringLiteral("kvWindow"));
        const bool modelPersisted =
                probe.contains(QStringLiteral("modelFilename"));
        probe.endGroup();
        if (!kvPersisted || !modelPersisted) {
            gpuTotalBytes = aicore_lingbot_device_total_memory("auto");
            const char* tierModel = nullptr;
            lingbotAutoKvProfile(gpuTotalBytes, &kvDefaultScale,
                                 &kvDefaultWindow, &tierModel);
            kvAutoProfiled = gpuTotalBytes > 0;
            if (!modelPersisted && tierModel) {
                modelDefault = QString::fromUtf8(tierModel);
            }
        }
    }
    m_advancedBox = new QGroupBox(tr("Advanced (KV cache / sampling)"), this);
    m_advancedBox->setCheckable(true);
    m_advancedBox->setChecked(true);
    m_advancedBox->setToolTip(
            tr("Persistent KV-cache profile and stream sampling flags, "
               "matching upstream ggml_demo.py --kv_cache_scale / "
               "--kv_cache_window / --stride / --rotate_clockwise_90."));
    m_advancedContainer = new QWidget(m_advancedBox);
    auto* advForm = new QFormLayout(m_advancedContainer);
    advForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    m_kvScale = new QSpinBox(m_advancedContainer);
    m_kvScale->setRange(1, 32);
    m_kvScale->setValue(kvDefaultScale);
    m_kvScale->setToolTip(
            tr("Persistent scale frames of the GCT KV cache (upstream "
               "--kv_cache_scale; official release default 8). The first-run "
               "default is auto-profiled to the detected GPU memory."));
    advForm->addRow(tr("KV cache scale"), m_kvScale);
    m_kvWindow = new QSpinBox(m_advancedContainer);
    m_kvWindow->setRange(1, 256);
    m_kvWindow->setValue(kvDefaultWindow);
    m_kvWindow->setToolTip(
            tr("Sliding-window frames of the persistent KV cache (upstream "
               "--kv_cache_window; official release default 64). The "
               "first-run default is auto-profiled to the detected GPU "
               "memory."));
    advForm->addRow(tr("KV cache window"), m_kvWindow);
    m_keyframeInterval = new QSpinBox(m_advancedContainer);
    m_keyframeInterval->setRange(0, 64);
    m_keyframeInterval->setValue(0);
    m_keyframeInterval->setSpecialValueText(tr("Auto"));
    m_keyframeInterval->setToolTip(
            tr("Official long-stream keyframe policy (upstream "
               "--keyframe_interval): every N-th streaming frame persists "
               "its KV; non-keyframes attend but do not persist. Auto "
               "resolves to ceil(N/320) per the official streaming rule "
               "(1 when the stream is shorter). Windowed mode always runs "
               "per-window interval 1."));
    advForm->addRow(tr("Keyframe interval"), m_keyframeInterval);
    if (kvAutoProfiled) {
        advForm->addRow(
                QString(),
                ecvAICoreUi::makeHintLabel(
                        tr("Auto-profiled for the detected GPU (%1 GiB): KV "
                           "scale=%2, window=%3, model=%4. Adjust freely — "
                           "your choice persists.")
                                .arg(static_cast<int>(gpuTotalBytes /
                                                      (1024ull * 1024 * 1024)))
                                .arg(kvDefaultScale)
                                .arg(kvDefaultWindow)
                                .arg(modelDefault.isEmpty()
                                             ? QStringLiteral("f16")
                                             : QFileInfo(modelDefault)
                                                       .baseName()),
                        m_advancedContainer));
    }
    m_frameStride = new QSpinBox(m_advancedContainer);
    m_frameStride->setRange(1, 100);
    m_frameStride->setValue(1);
    m_frameStride->setToolTip(
            tr("Process every Nth frame of the sequence (upstream --stride; "
               "1 = all frames). Applied after Max frames, as upstream."));
    advForm->addRow(tr("Frame stride"), m_frameStride);
    m_imageExt = new QLineEdit(m_advancedContainer);
    m_imageExt->setText(QStringLiteral(".jpg,.png,.jpeg,.bmp,.tif,.tiff"));
    m_imageExt->setToolTip(
            tr("Comma-separated extensions for the folder input (upstream "
               "--image_ext); matching is case-insensitive, so .JPG counts "
               "as .jpg."));
    advForm->addRow(tr("Image extensions"), m_imageExt);
    m_rotate90 = new QCheckBox(tr("Rotate frames 90° clockwise"),
                               m_advancedContainer);
    m_rotate90->setToolTip(
            tr("Upstream --rotate_clockwise_90; for portrait phone "
               "sequences stored sideways."));
    advForm->addRow(QString(), m_rotate90);
    auto* advLayout = new QVBoxLayout(m_advancedBox);
    advLayout->setContentsMargins(0, 0, 0, 0);
    advLayout->addWidget(m_advancedContainer);
    // The group starts expanded (all options at their defaults); the
    // toggled connection shows/hides the container when the user folds it.
    connect(m_advancedBox, &QGroupBox::toggled, m_advancedContainer,
            &QWidget::setVisible);
    form->addRow(QString(), m_advancedBox);
    // --- reconstruction mode (official long-sequence pipeline) ---
    m_modeCombo = new QComboBox(this);
    m_modeCombo->addItem(tr("Streaming (single KV cache)"),
                         QStringLiteral("streaming"));
    m_modeCombo->addItem(tr("Windowed (long sequences)"),
                         QStringLiteral("windowed"));
    form->addRow(tr("Mode"), m_modeCombo);
    m_windowSize = new QSpinBox(this);
    m_windowSize->setRange(8, 512);
    m_windowSize->setValue(64);
    m_windowSize->setSingleStep(8);
    m_overlap = new QSpinBox(this);
    m_overlap->setRange(1, 256);
    m_overlap->setValue(16);
    m_windowRow = new QWidget(this);
    auto* windowLayout = new QHBoxLayout(m_windowRow);
    windowLayout->setContentsMargins(0, 0, 0, 0);
    windowLayout->addWidget(new QLabel(tr("Window size:"), m_windowRow), 0);
    windowLayout->addWidget(m_windowSize, 1);
    windowLayout->addWidget(new QLabel(tr("Overlap:"), m_windowRow), 0);
    windowLayout->addWidget(m_overlap, 1);
    form->addRow(QString(), m_windowRow);
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() {
                m_windowRow->setVisible(
                        currentMode() ==
                        LingbotMapWorker::Settings::Mode::Windowed);
            });

    // --- loop playback (official viewer Playing/FPS semantics) ---
    m_playbackCheck =
            new QCheckBox(tr("Loop playback of reconstruction frames"), this);
    form->addRow(QString(), m_playbackCheck);
    m_playbackFps = new QSpinBox(this);
    m_playbackFps->setRange(1, 60);
    m_playbackFps->setValue(20);
    m_playbackMode = new QComboBox(this);
    m_playbackMode->addItem(tr("3D (all frames)"), QStringLiteral("all"));
    m_playbackMode->addItem(tr("4D (current frame)"),
                            QStringLiteral("current"));
    m_playbackRow = new QWidget(this);
    auto* playbackLayout = new QHBoxLayout(m_playbackRow);
    playbackLayout->setContentsMargins(0, 0, 0, 0);
    playbackLayout->addWidget(new QLabel(tr("FPS:"), m_playbackRow), 0);
    playbackLayout->addWidget(m_playbackFps, 1);
    playbackLayout->addWidget(new QLabel(tr("Mode:"), m_playbackRow), 0);
    playbackLayout->addWidget(m_playbackMode, 1);
    form->addRow(QString(), m_playbackRow);
    auto emitPlaybackSettings = [this]() {
        emit playbackSettingsChanged(m_playbackCheck->isChecked(),
                                     m_playbackFps->value(),
                                     m_playbackMode->currentData().toString() ==
                                             QStringLiteral("current"));
    };
    connect(m_playbackCheck, &QCheckBox::toggled, this,
            [this, emitPlaybackSettings]() {
                m_playbackRow->setVisible(m_playbackCheck->isChecked());
                emitPlaybackSettings();
            });
    connect(m_playbackFps, qOverload<int>(&QSpinBox::valueChanged), this,
            emitPlaybackSettings);
    connect(m_playbackMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, emitPlaybackSettings);

    m_addDbCheck = new QCheckBox(tr("Add reconstruction to DB tree"), this);
    m_addDbCheck->setChecked(true);
    form->addRow(QString(), m_addDbCheck);

    rootLayout->addLayout(form);

    // --- run / cancel ---
    auto* buttons = new QHBoxLayout;
    m_runButton = new QPushButton(tr("Run"), this);
    m_runButton->setDefault(true);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setEnabled(false);
    m_downloadButton = new QPushButton(tr("Download model"), this);
    buttons->addWidget(m_runButton);
    buttons->addWidget(m_cancelButton);
    buttons->addWidget(m_downloadButton);
    buttons->addStretch(1);
    rootLayout->addLayout(buttons);

    // --- progress + log (shared AICore plugin layout) ---
    m_downloadLabel = new QLabel(this);
    m_downloadLabel->setVisible(false);
    m_progress = new QProgressBar(this);
    m_progress->setVisible(false);
    rootLayout->addWidget(m_downloadLabel);
    rootLayout->addWidget(m_progress);
    m_log = new QTextEdit(this);
    m_log->setReadOnly(true);
    rootLayout->addWidget(m_log, 1);

    connect(m_runButton, &QPushButton::clicked, this, &LingbotMapDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &LingbotMapDialog::onCancel);
    connect(m_downloadButton, &QPushButton::clicked, this, [this]() {
        startDownload(m_modelCombo->currentData().toString());
    });
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                // Any combo change outside the blocked population path is a
                // user choice worth persisting over the VRAM-tier default.
                m_modelExplicit = true;
                onModelComboChanged();
            });
    connect(browseModelBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseCustomModel);
    connect(browseFolderBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseImageFolder);

    // Test-data repository wiring (shared download/extract pipeline).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& text) {
                Q_UNUSED(text);
                m_progress->setVisible(true);
                m_progress->setValue(percent);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            &LingbotMapDialog::appendLog);
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            &LingbotMapDialog::onTestDataDownloadFinished);
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            &LingbotMapDialog::onTestDataExtractionFinished);

    // Downloader (HF resolve-main URLs; digest-anchored ingestion).
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                m_progress->setVisible(true);
                if (total > 0) {
                    m_progress->setRange(0, 100);
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                    m_downloadLabel->setText(
                            ecvModelDownloader::formatDownloadProgress(received,
                                                                       total));
                }
            });
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &LingbotMapDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[LingbotMap] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[LingbotMap] Model downloaded: %1").arg(path));
                if (m_pendingActionAfterDownload == PendingAction::Run) {
                    m_pendingActionAfterDownload = PendingAction::None;
                    onRun();
                }
            });

    populateCatalogs();
    // VRAM-tier model default (first run only; a persisted choice always
    // wins via loadSettings below).
    if (!modelDefault.isEmpty()) {
        const int modelIdx = m_modelCombo->findData(modelDefault);
        if (modelIdx >= 0) m_modelCombo->setCurrentIndex(modelIdx);
    }
    // Sync the custom-model row visibility for the initial selection (the
    // catalog population runs with blocked signals, so the combo signal
    // handler never fired for it).
    onModelComboChanged();
    populateTestDataScenes();
    loadSettings();
}

LingbotMapDialog::~LingbotMapDialog() {
    if (m_downloader && m_downloadInProgress) m_downloader->cancel();
    saveSettings();
}

void LingbotMapDialog::populateCatalogs() {
    const QString previous = m_modelCombo->currentData().toString();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    const QVector<LingbotCatalogEntry> maps = catalogByRole("map");
    const int defaultIndex = aicore_lingbot_model_default_index();
    int currentDefault = 0;
    for (int i = 0; i < maps.size(); ++i) {
        const LingbotCatalogEntry& e = maps[i];
        m_modelCombo->addItem(tr("%1 (%2, %3 MB)")
                                      .arg(e.displayName, e.quantNote)
                                      .arg(e.sizeBytes / (1024 * 1024)),
                              e.filename);
        if (aicore_lingbot_model_by_filename(e.filename.toUtf8().constData()) ==
            aicore_lingbot_model_at(defaultIndex)) {
            currentDefault = i;
        }
    }
    m_modelCombo->setToolTip(
            tr("Dataset → model guidance:\n"
               "• Official long datasets (Drive, Lingbo World — video, "
               "N>320): use a long-* model (long-f16 recommended) with "
               "the streaming auto keyframe policy.\n"
               "• Short demo streams (Courthouse, Oxford, University, "
               "Loop): use a balanced model (f16 recommended, q8 for "
               "smaller VRAM).\n"
               "f16 is the upstream GUI default; q8 trades accuracy for "
               "half the weight memory; f32 is the exact reference."));
    m_modelCombo->setCurrentIndex(currentDefault);
    if (!previous.isEmpty()) {
        const int idx = m_modelCombo->findData(previous);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    m_modelCombo->blockSignals(false);

    m_skysegCombo->blockSignals(true);
    m_skysegCombo->clear();
    const QVector<LingbotCatalogEntry> skysegs = catalogByRole("skyseg");
    const int skysegDefault = aicore_lingbot_skyseg_model_default_index();
    int skysegCurrent = 0;
    for (int i = 0; i < skysegs.size(); ++i) {
        const LingbotCatalogEntry& e = skysegs[i];
        m_skysegCombo->addItem(tr("%1 (%2 MB)")
                                       .arg(e.displayName)
                                       .arg(e.sizeBytes / (1024 * 1024)),
                               e.filename);
        if (aicore_lingbot_model_by_filename(e.filename.toUtf8().constData()) ==
            aicore_lingbot_model_at(skysegDefault)) {
            skysegCurrent = i;
        }
    }
    m_skysegCombo->setCurrentIndex(skysegCurrent);
    m_skysegCombo->blockSignals(false);
}

void LingbotMapDialog::populateSkysegCombo(const QString& filename) {
    const int idx = m_skysegCombo->findData(filename);
    if (idx >= 0) m_skysegCombo->setCurrentIndex(idx);
}

void LingbotMapDialog::appendLog(const QString& msg) {
    m_log->append(QStringLiteral("[%1] %2").arg(
            QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")),
            msg));
}

void LingbotMapDialog::onModelComboChanged() {
    const QString filename = m_modelCombo->currentData().toString();
    const bool isCustom =
            filename.isEmpty() ||
            (filename.endsWith(QStringLiteral(".gguf")) &&
             !findEntryByFilename(catalogByRole("map"), filename, nullptr));
    m_customModelRow->setVisible(isCustom);
}

void LingbotMapDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qLingbotMap"),
            QStringLiteral("lastModelDir"), lingbotModelCacheDir());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select LingBot-Map GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    m_customModelPath->setText(path);
    m_customModelRow->setVisible(true);
    m_modelCombo->setCurrentIndex(-1);
    m_modelCombo->addItem(QFileInfo(path).fileName(), path);
    m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
}

void LingbotMapDialog::onBrowseImageFolder() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qLingbotMap"),
                             QStringLiteral("lastImageDir"), QDir::homePath());
    const QString path = cvFileDialog::getExistingDirectory(
            this, tr("Select ordered image sequence folder"), lastDir);
    if (path.isEmpty()) return;
    m_imageFolder->setText(path);
    m_customDataSelected = true;
    refreshSkySourceOptions();
}

void LingbotMapDialog::onBrowseSkyMaskDir() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qLingbotMap"),
                             QStringLiteral("lastMaskDir"), QDir::homePath());
    const QString path = cvFileDialog::getExistingDirectory(
            this, tr("Select sky mask folder (<frame-stem>.png, 255 = keep)"),
            lastDir);
    if (path.isEmpty()) return;
    m_skyMaskDir->setText(path);
}

void LingbotMapDialog::refreshInputRows() {
    const bool video =
            m_inputModeCombo && m_inputModeCombo->currentData().toString() ==
                                        QStringLiteral("video");
    if (m_folderRow) m_folderRow->setVisible(!video);
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (m_videoRow) m_videoRow->setVisible(video);
    if (m_videoFpsRow) m_videoFpsRow->setVisible(video);
#endif
}

#ifdef HAS_OPENCV_FACE_CAPTURE
void LingbotMapDialog::onBrowseVideoFile() {
    // Same convention as VideoPlaybackWidget::browseVideoFile: remembers
    // the last directory under <qLingbotMap>/lastVideoDir.
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qLingbotMap"),
                             QStringLiteral("lastVideoDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select input video"), lastDir,
            tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.ts "
               "*.mpg *.mpeg);;All files (*.*)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qLingbotMap"),
                         QStringLiteral("lastVideoDir"), path);
    m_videoPath->setText(path);
}
#endif

QString LingbotMapDialog::resolveModelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = lingbotModelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

QString LingbotMapDialog::resolveSkysegPath() const {
    const QString filename = m_skysegCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = lingbotModelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

bool LingbotMapDialog::ensureModelAvailable(PendingAction action) {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[LingbotMap] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath())) {
        LingbotCatalogEntry entry;
        if (!findEntryByFilename(catalogByRole("map"), filename, &entry)) {
            appendLog(
                    tr("[LingbotMap] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[LingbotMap] Model missing — downloading %1; the stream "
                     "starts automatically when ready.")
                          .arg(filename));
        startDownload(filename);
        return false;
    }
    return true;
}

void LingbotMapDialog::startDownload(const QString& filename) {
    if (m_downloadInProgress) {
        appendLog(tr("[LingbotMap] A download is already running."));
        return;
    }
    LingbotCatalogEntry entry;
    if (!findEntryByFilename(catalogByRole("map"), filename, &entry) &&
        !findEntryByFilename(catalogByRole("skyseg"), filename, &entry)) {
        appendLog(tr("[LingbotMap] Unknown model: %1").arg(filename));
        return;
    }
    QDir().mkpath(lingbotModelCacheDir());
    const QString dest =
            lingbotModelCacheDir() + QDir::separator() + entry.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[LingbotMap] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[LingbotMap] Downloading %1 (%2)…")
                      .arg(entry.filename, entry.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = entry.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // LingBot-Map GGUFs are hundreds of MB
    // Content identity from the pinned digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught).
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(entry.filename)};
    m_downloader->download(req);
}

LingbotMapWorker::Settings LingbotMapDialog::collectSettings() const {
    LingbotMapWorker::Settings s;
    s.modelPath = resolveModelPath();
    s.imageFolder = m_imageFolder->text().trimmed();
#ifdef HAS_OPENCV_FACE_CAPTURE
    const bool videoMode =
            m_inputModeCombo && m_inputModeCombo->currentData().toString() ==
                                        QStringLiteral("video");
    s.videoPath = videoMode ? m_videoPath->text().trimmed() : QString();
    s.videoFps = m_videoFps->value();
#endif
    s.imageExt = m_imageExt->text().trimmed();
    s.maxFrames = m_maxFrames->value();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.image_size = m_imageSize->value();
    s.confThreshold = static_cast<float>(m_confThreshold->value());
    s.kvScale = m_kvScale->value();
    s.kvWindow = m_kvWindow->value();
    s.keyframeInterval = m_keyframeInterval->value();
    s.frameStride = m_frameStride->value();
    s.rotateClockwise90 = m_rotate90->isChecked();
    const QString skySource = m_skySourceCombo->currentData().toString();
    if (skySource == QStringLiteral("native")) {
        s.skySource = LingbotMapWorker::Settings::SkySource::Native;
        s.skysegModelPath = resolveSkysegPath();
    } else if (skySource == QStringLiteral("cached")) {
        s.skySource = LingbotMapWorker::Settings::SkySource::CachedMasks;
        s.skyMaskDir = m_skyMaskDir->text().trimmed();
    } else {
        s.skySource = LingbotMapWorker::Settings::SkySource::None;
    }
    s.addResultToDb = m_addDbCheck->isChecked();
    s.mode = currentMode();
    s.windowSize = m_windowSize->value();
    s.overlap = m_overlap->value();
    return s;
}

LingbotMapWorker::Settings::Mode LingbotMapDialog::currentMode() const {
    return m_modeCombo && m_modeCombo->currentData().toString() ==
                                   QStringLiteral("windowed")
                   ? LingbotMapWorker::Settings::Mode::Windowed
                   : LingbotMapWorker::Settings::Mode::Streaming;
}

// Forward declaration: the definition lives with the other file-local
// helpers below (onRun needs it before that point).
namespace {
bool lingbotDatasetFromName(const QString& name,
                            ecvTestDataRepository::Dataset* out);
}

void LingbotMapDialog::onRun() {
    if (!ensureModelAvailable(PendingAction::Run)) return;
    const LingbotMapWorker::Settings s = collectSettings();
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!s.videoPath.isEmpty()) {
        if (!QFileInfo::exists(s.videoPath)) {
            QMessageBox::warning(this, tr("LingBot-Map"),
                                 tr("Select an existing video file first."));
            return;
        }
    } else if (m_inputModeCombo->currentData().toString() ==
               QStringLiteral("video")) {
        // Video mode with no file resolved: pull it from the selected
        // official test video (downloading + caching it on first use) and
        // start the run when ready. A custom path always wins (it was
        // already validated above by its existence check).
        ecvTestDataRepository::Dataset scene;
        const bool isTestVideo =
                lingbotDatasetFromName(
                        m_testSceneCombo->currentData().toString(), &scene) &&
                ecvTestDataRepository::isSingleFileDataset(scene);
        if (isTestVideo) {
            if (ecvTestDataRepository::instance().isDatasetAvailable(scene)) {
                applyTestSceneSelection();  // fills the video input
                onRun();                    // re-collect settings and run
            } else {
                m_pendingRunAfterTestData = true;
                appendLog(
                        tr("[LingbotMap] Test video not cached yet — "
                           "downloading it; the run starts "
                           "automatically when ready."));
                requestTestData();
            }
            return;
        }
        QMessageBox::warning(this, tr("LingBot-Map"),
                             tr("Select an existing video file first."));
        return;
    } else
#endif
            if (s.imageFolder.isEmpty() || !QFileInfo::exists(s.imageFolder)) {
        QMessageBox::warning(this, tr("LingBot-Map"),
                             tr("Select an existing image folder first."));
        return;
    }
    if (s.skySource == LingbotMapWorker::Settings::SkySource::Native &&
        !QFileInfo::exists(s.skysegModelPath)) {
        // Auto-download the selected skyseg model when missing.
        const QString skysegName = m_skysegCombo->currentData().toString();
        if (!skysegName.isEmpty() &&
            findEntryByFilename(catalogByRole("skyseg"), skysegName, nullptr)) {
            appendLog(tr("[LingbotMap] Sky model missing — downloading %1…")
                              .arg(skysegName));
            startDownload(skysegName);
            return;
        }
        appendLog(
                tr("[LingbotMap] Sky model unavailable; continuing without "
                   "sky masking."));
        m_skySourceCombo->setCurrentIndex(0);
    }
    if (s.skySource == LingbotMapWorker::Settings::SkySource::CachedMasks &&
        s.skyMaskDir.isEmpty()) {
        appendLog(
                tr("[LingbotMap] Cached mask folder is empty — continuing "
                   "without sky masking."));
        m_skySourceCombo->setCurrentIndex(0);
    }
    saveSettings();
    emit runRequested(s);
}

void LingbotMapDialog::onCancel() { emit cancelRequested(); }

void LingbotMapDialog::closeEvent(QCloseEvent* event) {
    // Uniform plugin-close semantics: if a reconstruction, download, or
    // test-data operation is active, ask for confirmation.
    if (m_taskRunning || m_downloadInProgress || m_testDataInProgress ||
        m_maskDownloadPending) {
        if (QMessageBox::question(
                    this, tr("Task running"),
                    tr("A LingbotMap task is running. Close anyway?"),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No) != QMessageBox::Yes) {
            event->ignore();
            return;
        }
    }
    emit cancelRequested();
    saveSettings();
    QDialog::closeEvent(event);
}

void LingbotMapDialog::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape &&
        (m_taskRunning || m_downloadInProgress || m_testDataInProgress ||
         m_maskDownloadPending)) {
        if (QMessageBox::question(
                    this, tr("Task running"),
                    tr("A LingbotMap task is running. Close anyway?"),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No) != QMessageBox::Yes) {
            return;
        }
    }
    QDialog::keyPressEvent(event);
}

void LingbotMapDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qLingbotMap"));
    // Restore any persisted model choice (catalog entry or custom GGUF).
    // Without one the first-run VRAM-tier default stays active.
    if (settings.contains(QStringLiteral("modelFilename"))) {
        const QString modelFilename =
                settings.value(QStringLiteral("modelFilename")).toString();
        const int modelIdx = m_modelCombo->findData(modelFilename);
        if (modelIdx >= 0) m_modelCombo->setCurrentIndex(modelIdx);
    }
    m_modelExplicit =
            settings.value(QStringLiteral("modelFilenameExplicit"), false)
                    .toBool();
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    m_imageFolder->setText(
            settings.value(QStringLiteral("imageFolder")).toString());
    m_maxFrames->setValue(
            settings.value(QStringLiteral("maxFrames"), 0).toInt());
    m_imageSize->setValue(
            settings.value(QStringLiteral("imageSize"), 518).toInt());
    m_confThreshold->setValue(
            settings.value(QStringLiteral("confThreshold"), 1.5).toDouble());
#ifdef HAS_OPENCV_FACE_CAPTURE
    {
        const QString inputMode = settings.value(QStringLiteral("inputMode"),
                                                 QStringLiteral("folder"))
                                          .toString();
        const int modeIdx = m_inputModeCombo->findData(inputMode);
        if (modeIdx >= 0) m_inputModeCombo->setCurrentIndex(modeIdx);
        m_videoPath->setText(
                settings.value(QStringLiteral("videoPath")).toString());
        m_videoFps->setValue(
                settings.value(QStringLiteral("videoFps"), 10).toInt());
        refreshInputRows();
    }
#endif
    m_imageExt->setText(
            settings.value(QStringLiteral("imageExt"),
                           QStringLiteral(".jpg,.png,.jpeg,.bmp,.tif,.tiff"))
                    .toString());
    // Restore a persisted sky source programmatically (never counted as a
    // user-explicit choice, so the outdoor-scene auto switch keeps working).
    if (settings.contains(QStringLiteral("skySource"))) {
        const QString skySource =
                settings.value(QStringLiteral("skySource")).toString();
        m_skyAutoSwitching = true;
        const int skyIdx = m_skySourceCombo->findData(skySource);
        if (skyIdx >= 0) m_skySourceCombo->setCurrentIndex(skyIdx);
        m_skyAutoSwitching = false;
    }
    populateSkysegCombo(
            settings.value(QStringLiteral("skysegModel")).toString());
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_advancedBox->setChecked(
            settings.value(QStringLiteral("advancedVisible"), true).toBool());
    // The KV profile restores only what was explicitly persisted: without a
    // saved choice the VRAM-auto-profiled first-run default stays active.
    if (settings.contains(QStringLiteral("kvScale"))) {
        m_kvScale->setValue(settings.value(QStringLiteral("kvScale")).toInt());
    }
    if (settings.contains(QStringLiteral("kvWindow"))) {
        m_kvWindow->setValue(
                settings.value(QStringLiteral("kvWindow")).toInt());
    }
    if (settings.contains(QStringLiteral("keyframeInterval"))) {
        m_keyframeInterval->setValue(
                settings.value(QStringLiteral("keyframeInterval")).toInt());
    }
    m_frameStride->setValue(
            settings.value(QStringLiteral("frameStride"), 1).toInt());
    m_rotate90->setChecked(
            settings.value(QStringLiteral("rotateClockwise90"), false)
                    .toBool());
    const QString mode =
            settings.value(QStringLiteral("mode"), QStringLiteral("streaming"))
                    .toString();
    const int modeIdx = m_modeCombo->findData(mode);
    if (modeIdx >= 0) m_modeCombo->setCurrentIndex(modeIdx);
    m_windowRow->setVisible(currentMode() ==
                            LingbotMapWorker::Settings::Mode::Windowed);
    m_windowSize->setValue(
            settings.value(QStringLiteral("windowSize"), 64).toInt());
    m_overlap->setValue(settings.value(QStringLiteral("overlap"), 16).toInt());
    m_playbackCheck->setChecked(
            settings.value(QStringLiteral("playback"), false).toBool());
    m_playbackRow->setVisible(m_playbackCheck->isChecked());
    m_playbackFps->setValue(
            settings.value(QStringLiteral("playbackFps"), 20).toInt());
    const QString playbackMode = settings.value(QStringLiteral("playbackMode"),
                                                QStringLiteral("current"))
                                         .toString();
    const int pbIdx = m_playbackMode->findData(playbackMode);
    if (pbIdx >= 0) m_playbackMode->setCurrentIndex(pbIdx);
    settings.endGroup();

    // Re-apply the test-scene defaults AFTER the persisted values land.
    // loadSettings runs after populateTestDataScenes() (whose automatic
    // sky-mask selection had already picked the scene-appropriate source);
    // restoring skySource here would otherwise silently override that with
    // a stale persisted value (e.g. "None" chosen for an unrelated scene
    // in a previous session). The programmatic restore is deliberately NOT
    // a user-explicit choice, so the scene auto-switch still wins.
    applyTestSceneSelection();
}

void LingbotMapDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qLingbotMap"));
    settings.setValue(QStringLiteral("modelFilename"),
                      m_modelCombo->currentData().toString());
    settings.setValue(QStringLiteral("modelFilenameExplicit"), m_modelExplicit);
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.setValue(QStringLiteral("imageFolder"), m_imageFolder->text());
    settings.setValue(QStringLiteral("maxFrames"), m_maxFrames->value());
    settings.setValue(QStringLiteral("imageSize"), m_imageSize->value());
    settings.setValue(QStringLiteral("confThreshold"),
                      m_confThreshold->value());
#ifdef HAS_OPENCV_FACE_CAPTURE
    settings.setValue(QStringLiteral("inputMode"),
                      m_inputModeCombo->currentData().toString());
    settings.setValue(QStringLiteral("videoPath"), m_videoPath->text());
    settings.setValue(QStringLiteral("videoFps"), m_videoFps->value());
#endif
    settings.setValue(QStringLiteral("imageExt"), m_imageExt->text());
    settings.setValue(QStringLiteral("skySource"),
                      m_skySourceCombo->currentData().toString());
    settings.setValue(QStringLiteral("skysegModel"),
                      m_skysegCombo->currentData().toString());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("advancedVisible"),
                      m_advancedBox->isChecked());
    settings.setValue(QStringLiteral("kvScale"), m_kvScale->value());
    settings.setValue(QStringLiteral("kvWindow"), m_kvWindow->value());
    settings.setValue(QStringLiteral("keyframeInterval"),
                      m_keyframeInterval->value());
    settings.setValue(QStringLiteral("frameStride"), m_frameStride->value());
    settings.setValue(QStringLiteral("rotateClockwise90"),
                      m_rotate90->isChecked());
    settings.setValue(
            QStringLiteral("mode"),
            currentMode() == LingbotMapWorker::Settings::Mode::Windowed
                    ? QStringLiteral("windowed")
                    : QStringLiteral("streaming"));
    settings.setValue(QStringLiteral("windowSize"), m_windowSize->value());
    settings.setValue(QStringLiteral("overlap"), m_overlap->value());
    settings.setValue(QStringLiteral("playback"), m_playbackCheck->isChecked());
    settings.setValue(QStringLiteral("playbackFps"), m_playbackFps->value());
    settings.setValue(QStringLiteral("playbackMode"),
                      m_playbackMode->currentData().toString());
    settings.endGroup();
}

void LingbotMapDialog::setTaskRunning(bool running) {
    m_taskRunning = running;
    m_runButton->setEnabled(!running);
    m_cancelButton->setEnabled(running);
    if (!running) {
        m_progress->setVisible(false);
        m_downloadLabel->setVisible(false);
    }
}

void LingbotMapDialog::setProgress(int percent) {
    if (percent < 0) {
        m_progress->setVisible(false);
        return;
    }
    m_progress->setVisible(true);
    m_progress->setRange(0, 100);
    m_progress->setValue(percent);
}

// ---------------------------------------------------------------------------
// Test data (shared ecvTestDataRepository, lingbot_map_data release)
// ---------------------------------------------------------------------------

namespace {

bool lingbotDatasetFromName(const QString& name,
                            ecvTestDataRepository::Dataset* out) {
    if (name == QStringLiteral("LingbotMapCourthouse")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapCourthouse;
    } else if (name == QStringLiteral("LingbotMapLoop")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapLoop;
    } else if (name == QStringLiteral("LingbotMapOxford")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapOxford;
    } else if (name == QStringLiteral("LingbotMapUniversity")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapUniversity;
    } else if (name == QStringLiteral("LingbotMapOxfordSkyMasks")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks;
    } else if (name == QStringLiteral("LingbotMapUniversitySkyMasks")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapUniversitySkyMasks;
    } else if (name == QStringLiteral("LingbotMapDriveVideo")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapDriveVideo;
    } else if (name == QStringLiteral("LingbotMapLingboWorldVideo")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapLingboWorldVideo;
    } else {
        return false;
    }
    return true;
}

// Returns the directory that directly holds the sequence frames below an
// extracted bundle root (the archives carry a top-level scene directory).
QString lingbotFrameDirOf(const QString& extractRoot) {
    QDirIterator it(
            extractRoot,
            QStringList{QStringLiteral("*.png"), QStringLiteral("*.jpg")},
            QDir::Files, QDirIterator::Subdirectories);
    if (it.hasNext()) {
        return QFileInfo(it.next()).absolutePath();
    }
    return extractRoot;
}

// Sky-mask dataset paired with a scene (none for courthouse/loop).
bool lingbotMaskDatasetFor(ecvTestDataRepository::Dataset scene,
                           ecvTestDataRepository::Dataset* maskOut) {
    if (scene == ecvTestDataRepository::Dataset::LingbotMapOxford) {
        *maskOut = ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks;
        return true;
    }
    if (scene == ecvTestDataRepository::Dataset::LingbotMapUniversity) {
        *maskOut = ecvTestDataRepository::Dataset::LingbotMapUniversitySkyMasks;
        return true;
    }
    return false;
}

}  // namespace

void LingbotMapDialog::populateTestDataScenes() {
    QComboBox* combo = m_testSceneCombo;
    combo->blockSignals(true);
    combo->clear();
    combo->addItem(tr("Courthouse (outdoor, short)"),
                   QStringLiteral("LingbotMapCourthouse"));
    combo->addItem(tr("Loop (indoor loop closure, short)"),
                   QStringLiteral("LingbotMapLoop"));
    combo->addItem(tr("Oxford Spires (outdoor, short)"),
                   QStringLiteral("LingbotMapOxford"));
    combo->addItem(tr("University (outdoor, short)"),
                   QStringLiteral("LingbotMapUniversity"));
    combo->addItem(tr("Drive (long, car-mounted camera)"),
                   QStringLiteral("LingbotMapDriveVideo"));
    combo->addItem(tr("Lingbo World (long, walkthrough)"),
                   QStringLiteral("LingbotMapLingboWorldVideo"));
    combo->addItem(tr("Custom (own image folder)"), QStringLiteral("Custom"));
    combo->setItemData(
            0,
            tr("Official courthouse demo stream (~286 frames): use a "
               "balanced model (f16 recommended / q8)."),
            Qt::ToolTipRole);
    combo->setItemData(
            1,
            tr("Indoor loop-closure demo stream: use a balanced model; "
               "no sky masking needed."),
            Qt::ToolTipRole);
    combo->setItemData(
            4,
            tr("Official long-model dataset (car-mounted, ~1050 frames @ "
               "10 fps): use a long-* model with the streaming auto "
               "keyframe policy."),
            Qt::ToolTipRole);
    combo->setItemData(
            5,
            tr("Official long-model dataset (walkthrough, ~667 frames @ "
               "10 fps): use a long-* model with the streaming auto "
               "keyframe policy."),
            Qt::ToolTipRole);
    combo->blockSignals(false);
    applyTestSceneSelection();
}

void LingbotMapDialog::applyTestSceneSelection() {
    const QString sceneName = m_testSceneCombo->currentData().toString();
    if (sceneName == QStringLiteral("Custom")) {
        // User-provided folder: no auto-fill; the sky-masking reminder
        // (hint label) is driven by refreshSkySourceOptions instead.
        m_customDataSelected = true;
        updateDatasetHint();
        refreshSkySourceOptions();
        return;
    }
    ecvTestDataRepository::Dataset scene;
    if (!lingbotDatasetFromName(sceneName, &scene)) {
        return;
    }
    m_customDataSelected = false;
    // Official long-model video datasets fill the VIDEO input (not the
    // image folder) once downloaded; sky masking is left untouched (the
    // upstream long_real campaign runs them unmasked).
    if (ecvTestDataRepository::isSingleFileDataset(scene)) {
#ifdef HAS_OPENCV_FACE_CAPTURE
        if (m_inputModeCombo) {
            const int videoIdx =
                    m_inputModeCombo->findData(QStringLiteral("video"));
            if (videoIdx >= 0) m_inputModeCombo->setCurrentIndex(videoIdx);
        }
        const QString video = ecvTestDataRepository::findDatasetFile(
                scene,
                ecvTestDataRepository::getDatasetInfo(scene).zipFileName);
        if (!video.isEmpty() && m_videoPath) {
            m_videoPath->setText(video);
        }
#endif
        updateDatasetHint();
        refreshSkySourceOptions();
        return;
    }
    // Auto sky masking: outdoor demo scenes enable it (the paired cached
    // mask bundle when already available, native skyseg otherwise); the
    // indoor loop scene has no sky. A user-explicit combo choice always
    // wins afterwards.
    if (!m_skySourceUserSet) {
        QString want;
        if (scene == ecvTestDataRepository::Dataset::LingbotMapLoop) {
            want = QStringLiteral("none");
        } else {
            ecvTestDataRepository::Dataset maskKind;
            if (lingbotMaskDatasetFor(scene, &maskKind) &&
                ecvTestDataRepository::instance().isDatasetAvailable(
                        maskKind)) {
                if (m_skyMaskDir->text().trimmed().isEmpty()) {
                    m_skyMaskDir->setText(lingbotFrameDirOf(
                            ecvTestDataRepository::extractPath(maskKind)));
                }
                want = QStringLiteral("cached");
            } else {
                want = QStringLiteral("native");
            }
        }
        if (m_skySourceCombo->currentData().toString() != want) {
            m_skyAutoSwitching = true;
            const int wantIdx = m_skySourceCombo->findData(want);
            if (wantIdx >= 0) m_skySourceCombo->setCurrentIndex(wantIdx);
            m_skyAutoSwitching = false;
        }
    }
    // Auto-fill the image folder when the scene is already extracted.
    const QString extract = ecvTestDataRepository::extractPath(scene);
    if (!ecvTestDataRepository::getLingbotMapImages(extract).isEmpty()) {
        m_imageFolder->setText(lingbotFrameDirOf(extract));
    }
    updateDatasetHint();
    refreshSkySourceOptions();
}

void LingbotMapDialog::updateDatasetHint() {
    if (!m_datasetHint) return;
    const QString sceneName = m_testSceneCombo->currentData().toString();
    QString hint;
    if (sceneName == QStringLiteral("LingbotMapDriveVideo")) {
        hint =
                tr("Drive dataset: official long-model sequence (car-mounted, "
                   "~1050 frames @ 10 fps, auto keyframe interval = 4). "
                   "Recommended model: long-f16 (or long-q8 for smaller VRAM). "
                   "Run in Streaming mode — the auto keyframe policy bounds "
                   "the KV cache for the long run.");
    } else if (sceneName == QStringLiteral("LingbotMapLingboWorldVideo")) {
        hint =
                tr("Lingbo World dataset: official long-model sequence "
                   "(walkthrough, ~667 frames @ 10 fps, auto keyframe "
                   "interval = 3). Recommended model: long-f16 (or long-q8 "
                   "for smaller VRAM). Run in Streaming mode with the auto "
                   "keyframe policy.");
    } else if (sceneName.startsWith(QStringLiteral("LingbotMap")) &&
               sceneName != QStringLiteral("LingbotMapLoop")) {
        hint =
                tr("Short outdoor demo sequence: use a balanced model (f16 "
                   "recommended, q8 for smaller VRAM) in Streaming mode. "
                   "Outdoor scenes benefit from sky masking.");
    } else if (sceneName == QStringLiteral("LingbotMapLoop")) {
        hint =
                tr("Indoor loop-closure sequence: use a balanced model (f16 "
                   "recommended). No sky masking needed.");
    } else if (sceneName == QStringLiteral("Custom")) {
        hint =
                tr("Model guidance: sequences above ~320 frames are long runs "
                   "— prefer a long-* model with the auto keyframe policy "
                   "(Streaming); shorter sequences match the balanced f16/q8 "
                   "models.");
    }
    m_datasetHint->setText(hint);
    m_datasetHint->setVisible(!hint.isEmpty());
}

void LingbotMapDialog::refreshSkySourceOptions() {
    const QString sceneName = m_testSceneCombo->currentData().toString();
    ecvTestDataRepository::Dataset scene;
    const bool knownScene = lingbotDatasetFromName(sceneName, &scene);
    if (!knownScene && sceneName != QStringLiteral("Custom")) {
        return;
    }
    ecvTestDataRepository::Dataset maskKind;
    const bool hasMasks = knownScene && lingbotMaskDatasetFor(scene, &maskKind);
    const int cachedIdx = m_skySourceCombo->findData(QStringLiteral("cached"));
    if (cachedIdx >= 0) {
        // Cached masks exist only for the scenes that ship them, and only
        // once the mask bundle is downloaded/extracted.
        const bool enabled =
                hasMasks &&
                ecvTestDataRepository::instance().isDatasetAvailable(maskKind);
        auto* itemModel =
                qobject_cast<QStandardItemModel*>(m_skySourceCombo->model());
        if (itemModel && itemModel->item(cachedIdx)) {
            itemModel->item(cachedIdx)->setEnabled(enabled);
        }
    }
    const bool native = m_skySourceCombo->currentData().toString() ==
                        QStringLiteral("native");
    const bool cached = m_skySourceCombo->currentData().toString() ==
                        QStringLiteral("cached");
    m_skysegRow->setVisible(native);
    m_skyMaskRow->setVisible(cached);
    // Sky-effect warnings for the log panel reader:
    //  - custom data with masking off: the original reminder;
    //  - an OFFICIAL outdoor scene with masking explicitly set to None:
    //    outdoor reconstruction quality degrades without sky masking
    //    (sky pixels look like ground planes to the depth head).
    if (m_skyHint) {
        const bool outdoorScene =
                knownScene &&
                scene != ecvTestDataRepository::Dataset::LingbotMapLoop;
        const bool videoScene =
                knownScene && ecvTestDataRepository::isSingleFileDataset(scene);
        const bool warnNone = outdoorScene && !videoScene && !native && !cached;
        const bool remindCustom = m_customDataSelected && !native && !cached;
        const bool show = remindCustom || warnNone;
        m_skyHint->setVisible(show);
        if (show) {
            m_skyHint->setText(warnNone ? tr("Sky masking is OFF for this "
                                             "outdoor sequence — sky pixels "
                                             "will be treated as geometry "
                                             "and reconstruction quality may "
                                             "be noticeably worse. Switch "
                                             "Sky masking to Native skyseg "
                                             "for best results.")
                                        : m_customSkyHintText);
        }
    }
}

void LingbotMapDialog::requestTestData() {
    if (m_testDataInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    ecvTestDataRepository::Dataset scene;
    if (!lingbotDatasetFromName(m_testSceneCombo->currentData().toString(),
                                &scene)) {
        return;
    }
    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        return;
    }
    m_testDataInProgress = true;
    m_testDataBtn->setEnabled(false);
    m_runButton->setEnabled(false);
    m_progress->setVisible(true);
    m_progress->setValue(0);
    m_downloadLabel->setVisible(true);

    if (!repo.isDatasetAvailable(scene)) {
        appendLog(tr("[Test data] Downloading %1…")
                          .arg(ecvTestDataRepository::getDatasetInfo(scene)
                                       .displayName));
        repo.startDownload(scene);
        return;
    }
    if (ecvTestDataRepository::isSingleFileDataset(scene)) {
        // Video datasets: apply directly when the mp4 is materialized.
        if (ecvTestDataRepository::instance().isDatasetAvailable(scene)) {
            applyTestSceneSelection();
            finishTestDataFlow();
        } else {
            // Not cached even though the earlier availability check said
            // so — fall through to the download path above is impossible
            // here; report and bail out.
            appendLog(
                    tr("[Test data] Video asset missing — retry the "
                       "download."));
            finishTestDataFlow();
        }
        return;
    }
    if (!ecvTestDataRepository::getLingbotMapImages(
                 ecvTestDataRepository::extractPath(scene))
                 .isEmpty()) {
        // Already extracted — apply the selection directly.
        applyTestSceneSelection();
        // Queue the paired mask bundle when the cached-mask source needs it.
        if (m_skySourceCombo->currentData().toString() ==
            QStringLiteral("cached")) {
            ecvTestDataRepository::Dataset maskKind;
            if (lingbotMaskDatasetFor(scene, &maskKind) &&
                !ecvTestDataRepository::instance().isDatasetAvailable(
                        maskKind)) {
                m_maskDownloadPending = true;
                m_pendingMaskDownload = maskKind;
                repo.startDownload(maskKind);
                return;
            }
        }
        finishTestDataFlow();
        return;
    }
    // Verified zip cached — extract it.
    appendLog(tr("[Test data] Extracting cached archive..."));
    repo.extractDataset(scene);
}

void LingbotMapDialog::finishTestDataFlow() {
    m_testDataInProgress = false;
    m_testDataBtn->setEnabled(true);
    m_runButton->setEnabled(true);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
}

void LingbotMapDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        m_pendingRunAfterTestData = false;
        finishTestDataFlow();
        return;
    }
    appendLog(tr("[Test data] Extracting…"));
    ecvTestDataRepository::instance().extractDataset(kind);
}

void LingbotMapDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        m_pendingRunAfterTestData = false;
        finishTestDataFlow();
        return;
    }
    // Mask bundles just complete the cached-mask flow; scene bundles fill
    // the image folder. Video datasets fill the video input instead.
    const bool isMask =
            kind == ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks ||
            kind == ecvTestDataRepository::Dataset::
                            LingbotMapUniversitySkyMasks;
    if (ecvTestDataRepository::isSingleFileDataset(kind)) {
        applyTestSceneSelection();
        appendLog(
                tr("[Test data] Long-model video ready — video input "
                   "auto-filled (sampled at 10 fps). Recommended model: "
                   "a long-* GGUF (long-f16); Streaming mode with the "
                   "auto keyframe policy."));
        const bool runNow = m_pendingRunAfterTestData;
        m_pendingRunAfterTestData = false;
        finishTestDataFlow();
        if (runNow) {
            // Run was clicked while the video was still uncached; the
            // video input is filled now, so continue the run.
            onRun();
        }
        return;
    }
    if (isMask) {
        appendLog(tr("[Test data] Sky masks ready."));
        m_skyMaskDir->setText(
                lingbotFrameDirOf(ecvTestDataRepository::extractPath(kind)));
        refreshSkySourceOptions();
        finishTestDataFlow();
        return;
    }
    applyTestSceneSelection();
    appendLog(tr("[Test data] Scene ready — image folder auto-filled."));
    // Queue the paired mask bundle when the cached-mask source needs it.
    if (m_skySourceCombo->currentData().toString() ==
        QStringLiteral("cached")) {
        ecvTestDataRepository::Dataset maskKind;
        if (lingbotMaskDatasetFor(kind, &maskKind) &&
            !ecvTestDataRepository::instance().isDatasetAvailable(maskKind)) {
            m_maskDownloadPending = true;
            m_pendingMaskDownload = maskKind;
            ecvTestDataRepository::instance().startDownload(maskKind);
            return;
        }
    }
    finishTestDataFlow();
}
