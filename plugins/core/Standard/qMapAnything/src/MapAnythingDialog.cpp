// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MapAnythingDialog.h"

#include <QCloseEvent>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QKeyEvent>
#include <QMessageBox>
#include <QProgressBar>
#include <QSettings>
#include <QShowEvent>
#include <QSplitter>
#include <QVBoxLayout>

#include "aicore/inference_log.h"
#include "ecvAICoreUiHelper.h"
#include "ecvPersistentSettings.h"

namespace {

// C++17: a using-declaration cannot import a scoped-enum TYPE name, so
// alias it instead.
using Dataset = ecvTestDataRepository::Dataset;

// Single-pass budget: the upstream VRAM post-mortem measures ~118 MiB per
// frame (f16 @512, global attention), 38 frames peaking at 8.75 GiB on a
// 24 GB card - 64 frames fit comfortably; beyond it the worker streams
// with overlapping chunks (slight quality downgrade, official chain mode).
constexpr int kMaxViews = 64;

// Scene combo index <-> Dataset kind (order matches kMapAnythingScenes).
const Dataset kSceneKinds[] = {
        Dataset::MapAnythingCourtyard,  Dataset::MapAnythingDeliveryArea,
        Dataset::MapAnythingElectro,    Dataset::MapAnythingFacade,
        Dataset::MapAnythingKicker,     Dataset::MapAnythingMeadow,
        Dataset::MapAnythingOffice,     Dataset::MapAnythingPipes,
        Dataset::MapAnythingPlayground, Dataset::MapAnythingRelief,
        Dataset::MapAnythingRelief2,    Dataset::MapAnythingTerrace,
        Dataset::MapAnythingTerrains};

}  // namespace

MapAnythingDialog::MapAnythingDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(
            tr("MapAnything — Universal Feed-Forward Metric 3D "
               "Reconstruction"));
    setupUi();
    populatePanels();
    loadSettings();
}

MapAnythingDialog::~MapAnythingDialog() { saveSettings(); }

void MapAnythingDialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
}

QString MapAnythingDialog::modelCacheDir() {
    return MapAnythingHelpers::modelCacheDir();
}

void MapAnythingDialog::setupUi() {
    resize(1080, 720);
    m_downloader = new ecvModelDownloader(this);

    auto* rootLayout = new QVBoxLayout(this);

    // Global runtime parameters (qYOLO layout: one row above the list).
    auto* globalRow = new QHBoxLayout();
    globalRow->addWidget(new QLabel(tr("Device:"), this));
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->addItem(QStringLiteral("auto"));
    m_deviceCombo->addItem(QStringLiteral("cpu"));
    m_deviceCombo->addItem(QStringLiteral("cuda"));
    m_deviceCombo->addItem(QStringLiteral("vulkan"));
    globalRow->addWidget(m_deviceCombo);
    globalRow->addWidget(new QLabel(tr("Threads:"), this));
    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 256);
    m_threads->setValue(0);
    m_threads->setSpecialValueText(tr("auto"));
    globalRow->addWidget(m_threads);
    globalRow->addStretch(1);
    m_downloadLabel = new QLabel(this);
    globalRow->addWidget(m_downloadLabel);
    rootLayout->addLayout(globalRow);

    // Body splitter: model list | stacked pages (qYOLO layout).
    m_bodySplitter = new QSplitter(Qt::Horizontal, this);
    m_modelList = new QListWidget(m_bodySplitter);
    m_modelList->setObjectName(QStringLiteral("mapAnythingModelList"));
    m_modelStack = new QStackedWidget(m_bodySplitter);
    m_bodySplitter->addWidget(m_modelList);
    m_bodySplitter->addWidget(m_modelStack);
    m_bodySplitter->setStretchFactor(0, 0);
    m_bodySplitter->setStretchFactor(1, 1);
    rootLayout->addWidget(m_bodySplitter, 1);

    connect(m_modelList, &QListWidget::currentRowChanged, m_modelStack,
            &QStackedWidget::setCurrentIndex);

    // Shared input block (common to every model page).
    auto* inputBox = new QWidget(this);
    auto* inputLayout = new QHBoxLayout(inputBox);
    inputLayout->setContentsMargins(0, 0, 0, 0);
    auto* listColumn = new QVBoxLayout();
    listColumn->addWidget(new QLabel(
            tr("Multi-view input (%1 views max):").arg(kMaxViews), inputBox));
    m_imageList = new QListWidget(inputBox);
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_imageList->setMaximumHeight(140);
    listColumn->addWidget(m_imageList);
    auto* listBtnRow = new QHBoxLayout();
    m_addImagesBtn = new QPushButton(tr("Add images..."), inputBox);
    m_removeImageBtn = new QPushButton(tr("Remove"), inputBox);
    m_clearImagesBtn = new QPushButton(tr("Clear"), inputBox);
    m_importDbBtn = new QPushButton(tr("Import from DB..."), inputBox);
    listBtnRow->addWidget(m_addImagesBtn);
    listBtnRow->addWidget(m_removeImageBtn);
    listBtnRow->addWidget(m_clearImagesBtn);
    listBtnRow->addWidget(m_importDbBtn);
    listBtnRow->addStretch(1);
    listColumn->addLayout(listBtnRow);
    inputLayout->addLayout(listColumn, 2);
    auto* previewColumn = new QVBoxLayout();
    m_previewLabel = new ecvClickableImageLabel(inputBox);
    m_previewLabel->setMinimumSize(180, 120);
    previewColumn->addWidget(m_previewLabel, 1);
    // Preview transport (qLingbotMap replay semantics): Prev / Play /
    // Pause / Reset plus a frame counter. The reconstruction takes the
    // preview over while it runs (progress events step the frames).
    auto* transportRow = new QHBoxLayout();
    m_previewPrevBtn = new QPushButton(QStringLiteral("<"), inputBox);
    m_previewPrevBtn->setToolTip(tr("Previous frame"));
    m_previewPrevBtn->setMaximumWidth(32);
    m_previewPlayBtn = new QPushButton(tr("Play"), inputBox);
    m_previewPauseBtn = new QPushButton(tr("Pause"), inputBox);
    m_previewResetBtn = new QPushButton(tr("Reset"), inputBox);
    m_frameLabel = new QLabel(QStringLiteral("0 / 0"), inputBox);
    transportRow->addWidget(m_previewPrevBtn);
    transportRow->addWidget(m_previewPlayBtn);
    transportRow->addWidget(m_previewPauseBtn);
    transportRow->addWidget(m_previewResetBtn);
    transportRow->addStretch(1);
    transportRow->addWidget(m_frameLabel);
    previewColumn->addLayout(transportRow);
    m_playTimer = new QTimer(this);
    m_playTimer->setInterval(700);  // ~1.4 fps flip-through of the views
    inputLayout->addLayout(previewColumn, 1);
    rootLayout->addWidget(inputBox);

    connect(m_previewPrevBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onPreviewPrev);
    connect(m_previewPlayBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onPreviewPlay);
    connect(m_previewPauseBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onPreviewPause);
    connect(m_previewResetBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onPreviewReset);
    connect(m_playTimer, &QTimer::timeout, this,
            &MapAnythingDialog::onPlayTick);

    connect(m_addImagesBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onAddImages);
    connect(m_removeImageBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onRemoveSelectedImages);
    connect(m_clearImagesBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onClearImages);
    connect(m_importDbBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::importDbImagesRequested);
    connect(m_imageList, &QListWidget::itemSelectionChanged, this,
            &MapAnythingDialog::onInputSelectionChanged);

    // Shared run row + Use-test-data controls.
    auto* runRow = new QHBoxLayout();
    m_runBtn = new QPushButton(tr("Run"), this);
    m_runBtn->setDefault(true);
    m_cancelBtn = new QPushButton(tr("Cancel"), this);
    m_cancelBtn->setEnabled(false);
    m_testSceneCombo = new QComboBox(this);
    // No "Use test data:" prefix — the button next to the combo already
    // names the action (matches the other AI plugin sample rows).
    for (const auto* label :
         {"courtyard (ETH3D)", "delivery_area (ETH3D)", "electro (ETH3D)",
          "facade (ETH3D)", "kicker (ETH3D)", "meadow (ETH3D)",
          "office (ETH3D)", "pipes (ETH3D)", "playground (ETH3D)",
          "relief (ETH3D)", "relief_2 (ETH3D)", "terrace (ETH3D)",
          "terrains (ETH3D)"}) {
        m_testSceneCombo->addItem(QString::fromLatin1(label));
    }
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Download and load an official MapAnything ETH3D test scene "
               "into the shared multi-view input list"));
    runRow->addWidget(m_runBtn);
    runRow->addWidget(m_cancelBtn);
    runRow->addWidget(m_testSceneCombo, 1);
    runRow->addWidget(m_testDataBtn);
    rootLayout->addLayout(runRow);

    connect(m_runBtn, &QPushButton::clicked, this, &MapAnythingDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onCancel);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &MapAnythingDialog::onUseTestData);

    // Progress + status. Logging routes through the shared AICore log
    // (CVLog → main-window console + session log file); plugins must not
    // host their own log widgets.
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    rootLayout->addWidget(m_progress);
    m_taskStatusLabel = new QLabel(this);
    rootLayout->addWidget(m_taskStatusLabel);

    // Repository signals: connected ONCE here — the repository is a
    // process-wide singleton and re-connecting per click would multiply
    // the slots. The chain itself (download→extract) lives in the shared
    // ensureDatasetReady API; the dialog only consumes datasetReady.
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            &MapAnythingDialog::onTestDataProgress);
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            &MapAnythingDialog::onTestDataLog);
    connect(&testDataRepo, &ecvTestDataRepository::datasetReady, this,
            &MapAnythingDialog::onTestDataReady);
}

void MapAnythingDialog::populatePanels() {
    const QStringList variants = MapAnythingHelpers::catalogVariants();
    int panelIndex = 0;
    for (const QString& variant : variants) {
        MapAnythingPanel panel;
        panel.variant = variant;
        panel.page = new QWidget(m_modelStack);
        // RPC-automation handles: unique per-panel objectNames (qt.invokeMethod
        // resolves targets by class+objectName; duplicate names across the
        // per-variant pages would be ambiguous).
        const QString panelTag = QString::number(panelIndex++);
        auto* form = new QFormLayout(panel.page);

        panel.modelCombo = new QComboBox(panel.page);
        form->addRow(tr("Checkpoint:"), panel.modelCombo);
        panel.customModelPath = new QLineEdit(panel.page);
        auto* customBrowse = new QPushButton(tr("..."), panel.page);
        customBrowse->setMaximumWidth(32);
        auto* customRow = new QWidget(panel.page);
        auto* customLayout = new QHBoxLayout(customRow);
        customLayout->setContentsMargins(0, 0, 0, 0);
        customLayout->addWidget(panel.customModelPath, 1);
        customLayout->addWidget(customBrowse);
        panel.customModelRow = customRow;
        form->addRow(tr("Custom GGUF:"), customRow);

        panel.imageSizeSpin = new QSpinBox(panel.page);
        panel.imageSizeSpin->setRange(0, 2048);
        panel.imageSizeSpin->setSingleStep(14);
        panel.imageSizeSpin->setSpecialValueText(tr("checkpoint default"));
        form->addRow(tr("Image size:"), panel.imageSizeSpin);
        panel.resizeModeCombo = new QComboBox(panel.page);
        panel.resizeModeCombo->addItem(QStringLiteral("balanced"));
        panel.resizeModeCombo->addItem(QStringLiteral("max_size"));
        panel.resizeModeCombo->addItem(QStringLiteral("square"));
        form->addRow(tr("Resize mode:"), panel.resizeModeCombo);

        panel.addPointCloudCheck = new QCheckBox(
                tr("Add merged world-frame point cloud to DB"), panel.page);
        panel.addPointCloudCheck->setChecked(true);
        panel.addDepthCheck = new QCheckBox(
                tr("Add per-view depth rasters to DB"), panel.page);
        panel.addDepthCheck->setChecked(true);
        panel.addPerViewCheck = new QCheckBox(
                tr("Add per-view point clouds to DB (large)"), panel.page);
        panel.addPerViewCheck->setChecked(false);
        panel.addPerViewCheck->setObjectName(
                QStringLiteral("mapPerViewCheck%1").arg(panelTag));
        panel.responsiveCheck = new QCheckBox(
                tr("Responsive streaming (chunked forwards: progress + "
                   "cancellable, slightly lower quality)"),
                panel.page);
        panel.responsiveCheck->setChecked(false);
        panel.responsiveCheck->setObjectName(
                QStringLiteral("mapResponsiveCheck%1").arg(panelTag));
        form->addRow(QString(), panel.addPointCloudCheck);
        form->addRow(QString(), panel.addDepthCheck);
        form->addRow(QString(), panel.addPerViewCheck);
        form->addRow(QString(), panel.responsiveCheck);
        panel.minConfSpin = new QDoubleSpinBox(panel.page);
        panel.minConfSpin->setRange(0.0, 0.99);
        panel.minConfSpin->setSingleStep(0.05);
        // 0.67 aligns with the upstream demo default (raw conf >= 3.0 on
        // the 1+exp scale <=> (conf-1)/conf >= 0.67) — keeps the sky /
        // occlusion sheets out of the fused cloud.
        panel.minConfSpin->setValue(0.67);
        panel.minConfSpin->setToolTip(
                tr("Normalized confidence floor in [0,1), family-unified: "
                   "sigmoid(logit) for the pi3 family, (conf-1)/conf for the "
                   "1+exp families (0.5 <=> raw conf >= 2.0; 0.67 <=> raw "
                   "conf >= 3.0, the upstream demo default). Raise it to "
                   "drop floating sheets from sky/occlusion regions."));
        form->addRow(tr("Min confidence:"), panel.minConfSpin);

        const QString display = variant;
        new QListWidgetItem(display, m_modelList);
        m_modelStack->addWidget(panel.page);
        m_panels.append(panel);

        connect(panel.modelCombo,
                qOverload<int>(&QComboBox::currentIndexChanged), this,
                &MapAnythingDialog::onModelComboChanged);
        connect(customBrowse, &QPushButton::clicked, this,
                &MapAnythingDialog::onBrowseCustomModel);
    }
    if (m_modelList->count() > 0) m_modelList->setCurrentRow(0);
    populateModelCombo(m_panels.first());
}

void MapAnythingDialog::populateModelCombo(MapAnythingPanel& panel,
                                           const QString& keepFilename) {
    if (panel.modelCombo == nullptr) return;
    panel.modelCombo->blockSignals(true);
    panel.modelCombo->clear();
    const auto models = MapAnythingHelpers::modelsForVariant(panel.variant);
    int defaultRow = -1;
    int q8FallbackRow = -1;
    for (const auto& model : models) {
        // Item text MUST carry the quantization note: displayName is
        // constant across the quantization rows of one variant, so the
        // bare displayName renders 4 indistinguishable entries (qYOLO's
        // modelDisplayLabel contract).
        panel.modelCombo->addItem(MapAnythingHelpers::modelDisplayLabel(model),
                                  model.filename);
        if (model.quantNote.contains(QStringLiteral("(recommended)"))) {
            defaultRow = panel.modelCombo->count() - 1;
        } else if (q8FallbackRow < 0 &&
                   model.filename.contains(QStringLiteral("q8_0"))) {
            // Belt-and-braces fallback if a future catalog row loses the
            // marker: the smallest fast default stays the pick.
            q8FallbackRow = panel.modelCombo->count() - 1;
        }
        if (!keepFilename.isEmpty() && model.filename == keepFilename) {
            defaultRow = panel.modelCombo->count() - 1;
        }
    }
    panel.modelCombo->addItem(tr("Custom path..."), QString());
    if (defaultRow < 0) defaultRow = q8FallbackRow;
    if (defaultRow >= 0) {
        panel.modelCombo->setCurrentIndex(defaultRow);
    }
    panel.modelCombo->blockSignals(false);
    applyModelRowVisibility(panel);
}

void MapAnythingDialog::applyModelRowVisibility(MapAnythingPanel& panel) {
    const bool custom = panel.modelCombo != nullptr &&
                        panel.modelCombo->currentData().toString().isEmpty();
    panel.customModelRow->setVisible(custom);
}

MapAnythingPanel* MapAnythingDialog::currentPanel() const {
    const int row = m_modelList != nullptr ? m_modelList->currentRow() : -1;
    if (row < 0 || row >= m_panels.size()) return nullptr;
    return const_cast<MapAnythingPanel*>(&m_panels[row]);
}

MapAnythingPanel* MapAnythingDialog::panelForVariant(
        const QString& variant) const {
    for (int i = 0; i < m_panels.size(); ++i) {
        if (m_panels[i].variant == variant) {
            return const_cast<MapAnythingPanel*>(&m_panels[i]);
        }
    }
    return nullptr;
}

MapAnythingPanel* MapAnythingDialog::panelForFilename(
        const QString& filename) const {
    for (int i = 0; i < m_panels.size(); ++i) {
        auto* panel = const_cast<MapAnythingPanel*>(&m_panels[i]);
        if (panel->modelCombo == nullptr) continue;
        const int idx = panel->modelCombo->findData(filename);
        if (idx >= 0) return panel;
    }
    return nullptr;
}

void MapAnythingDialog::onModelComboChanged(int index) {
    Q_UNUSED(index);
    auto* panel = currentPanel();
    if (panel == nullptr) return;
    panel->explicitModelChoice = true;
    applyModelRowVisibility(*panel);

    // Official-parity inference defaults for the selected checkpoint
    // (sources documented on aicore_map_anything_model_entry): the resize
    // mode and the confidence-filter semantics follow the upstream repo's
    // own reconstruction recipe.
    MapAnythingModelEntry entry;
    if (MapAnythingHelpers::findModelByFilename(
                panel->modelCombo->currentData().toString(), &entry) &&
        !entry.defaultResizeMode.isEmpty()) {
        const int modeIdx =
                panel->resizeModeCombo->findText(entry.defaultResizeMode);
        if (modeIdx >= 0) panel->resizeModeCombo->setCurrentIndex(modeIdx);
        if (entry.confFilter == QStringLiteral("percentile")) {
            // Official MapAnything gradio semantics: drop the lowest N
            // percent of confidences.
            panel->minConfSpin->setRange(1.0, 99.0);
            panel->minConfSpin->setSingleStep(1.0);
            panel->minConfSpin->setDecimals(0);
            panel->minConfSpin->setValue(entry.defaultMinConf);
            panel->minConfSpin->setSuffix(QStringLiteral("%"));
            panel->minConfSpin->setToolTip(
                    tr("Confidence percentile (official MapAnything "
                       "semantics): drops the lowest N percent of "
                       "confidences before fusion."));
        } else {
            panel->minConfSpin->setRange(0.0, 0.99);
            panel->minConfSpin->setSingleStep(0.05);
            panel->minConfSpin->setDecimals(2);
            panel->minConfSpin->setValue(entry.defaultMinConf);
            panel->minConfSpin->setSuffix(QString());
            panel->minConfSpin->setToolTip(
                    tr("Normalized confidence floor in [0,1), "
                       "family-unified: sigmoid(logit) for the pi3 family, "
                       "(conf-1)/conf for the 1+exp families."));
        }
    }
}

void MapAnythingDialog::onBrowseCustomModel() {
    auto* panel = currentPanel();
    if (panel == nullptr) return;
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select a MapAnything GGUF"), modelCacheDir(),
            tr("GGUF models (*.gguf)"));
    if (path.isEmpty()) return;
    panel->customModelPath->setText(path);
    panel->modelCombo->setCurrentIndex(panel->modelCombo->count() - 1);
    panel->explicitModelChoice = true;
}

void MapAnythingDialog::refreshModelList() {
    for (auto& panel : m_panels) {
        populateModelCombo(panel, panel.modelCombo->currentData().toString());
    }
}

QStringList MapAnythingDialog::resolveInputPaths() const {
    QStringList paths;
    for (int i = 0; i < m_imageList->count(); ++i) {
        const QString text = m_imageList->item(i)->text();
        if (!text.startsWith(QStringLiteral("db://"))) {
            paths.append(text);
        }
    }
    return paths;
}

void MapAnythingDialog::onRun() {
    if (m_taskRunning) return;
    if (!ensureModelAvailable()) return;
    QStringList paths = resolveInputPaths();
    if (paths.isEmpty()) {
        // Chain completion: a selected test scene that is already on disk
        // (downloaded + extracted) fills the view list automatically
        // instead of demanding another manual click.
        const int sceneIndex = m_testSceneCombo->currentIndex();
        if (sceneIndex >= 0 &&
            sceneIndex < (int)(sizeof(kSceneKinds) / sizeof(kSceneKinds[0])) &&
            ecvTestDataRepository::instance().isDatasetAvailable(
                    kSceneKinds[sceneIndex]) &&
            loadTestDataScene(kSceneKinds[sceneIndex])) {
            paths = resolveInputPaths();
        }
    }
    if (paths.isEmpty()) {
        appendLog(
                tr("[Error] No input images — add files, import from the "
                   "DB, or click Use test data."));
        return;
    }
    if (paths.size() > kMaxViews) {
        appendLog(tr("[Error] Too many views (%1 > %2) — the inter-frame "
                     "attention cost grows quadratically with the view "
                     "count.")
                          .arg(paths.size())
                          .arg(kMaxViews));
        return;
    }
    auto* panel = currentPanel();
    Settings s;
    s.modelPath = panel->modelCombo->currentData().toString().isEmpty()
                          ? panel->customModelPath->text()
                          : modelCacheDir() + QStringLiteral("/") +
                                    panel->modelCombo->currentData().toString();
    if (panel->modelCombo->currentData().toString().isEmpty() &&
        !QFileInfo::exists(s.modelPath)) {
        appendLog(tr("[Error] Custom model not found: %1").arg(s.modelPath));
        return;
    }
    s.imagePaths = paths;
    s.device = m_deviceCombo->currentText();
    s.threads = m_threads->value();
    s.imageSize = panel->imageSizeSpin->value();
    s.resizeMode = panel->resizeModeCombo->currentText();
    s.addPointCloud = panel->addPointCloudCheck->isChecked();
    s.addDepthRasters = panel->addDepthCheck->isChecked();
    s.addPerViewClouds = panel->addPerViewCheck->isChecked();
    s.responsiveStreaming = panel->responsiveCheck->isChecked();
    s.minConf = (float)panel->minConfSpin->value();
    saveSettings();
    emit runRequested(s);
}

void MapAnythingDialog::onCancel() {
    if (m_downloadInProgress) {
        cancelDownload();
        return;
    }
    emit cancelRequested();
}

void MapAnythingDialog::onAddImages() {
    const QStringList files = QFileDialog::getOpenFileNames(
            this, tr("Select the multi-view image set"), QString(),
            tr("Images (*.jpg *.jpeg *.png *.bmp)"));
    for (const QString& file : files) {
        if (m_imageList->count() >= kMaxViews) {
            appendLog(tr("[Warning] View cap (%1) reached; extra files "
                         "ignored.")
                              .arg(kMaxViews));
            break;
        }
        m_imageList->addItem(file);
    }
    updatePreview();
}

void MapAnythingDialog::onRemoveSelectedImages() {
    for (QListWidgetItem* item : m_imageList->selectedItems()) {
        delete m_imageList->takeItem(m_imageList->row(item));
    }
    updatePreview();
}

void MapAnythingDialog::onClearImages() {
    m_imageList->clear();
    updatePreview();
}

void MapAnythingDialog::onInputSelectionChanged() { updatePreview(); }

void MapAnythingDialog::updatePreview() {
    QListWidgetItem* item = m_imageList->currentItem();
    if (item == nullptr && m_imageList->count() > 0) {
        item = m_imageList->item(0);
    }
    if (item == nullptr) {
        m_previewLabel->clearPreview();
        m_playbackIndex = 0;
        m_frameLabel->setText(QStringLiteral("0 / 0"));
        onPreviewPause();
        return;
    }
    m_playbackIndex = m_imageList->row(item);
    showFrame(m_playbackIndex);
}

// ---- shared multi-view preview transport ---------------------------------
// Single preview funnel: every path (transport buttons, auto-play tick,
// list selection, reconstruction progress sync) lands here, so the frame
// counter and the list selection can never disagree.
void MapAnythingDialog::showFrame(int index) {
    if (index < 0 || index >= m_imageList->count()) return;
    m_playbackIndex = index;
    QListWidgetItem* item = m_imageList->item(index);
    const QString text = item->text();
    if (text.startsWith(QStringLiteral("db://"))) {
        m_previewLabel->clearPreview();  // DB previews come via the plugin
    } else {
        m_previewLabel->setPreviewImage(QImage(text), m_previewLabel->size());
    }
    m_frameLabel->setText(
            QStringLiteral("%1 / %2").arg(index + 1).arg(m_imageList->count()));
    // Keep the list selection on the shown frame (visual feedback without
    // triggering onInputSelectionChanged re-entry: blockSignals guard).
    m_imageList->blockSignals(true);
    m_imageList->setCurrentRow(index);
    m_imageList->blockSignals(false);
}

void MapAnythingDialog::onPreviewPlay() {
    if (m_imageList->count() == 0) return;
    m_playbackActive = true;
    m_reconstructionOwnsPreview = false;
    m_playTimer->start();
}

void MapAnythingDialog::onPreviewPause() {
    m_playbackActive = false;
    m_playTimer->stop();
}

void MapAnythingDialog::onPreviewPrev() {
    onPreviewPause();
    showFrame((m_playbackIndex - 1 + m_imageList->count()) %
              std::max(1, m_imageList->count()));
}

void MapAnythingDialog::onPreviewReset() {
    onPreviewPause();
    showFrame(0);
}

void MapAnythingDialog::onPlayTick() {
    if (m_imageList->count() == 0) {
        onPreviewPause();
        return;
    }
    // Wrap-around flip-through (a view sequence has no inherent end).
    showFrame((m_playbackIndex + 1) % m_imageList->count());
}

bool MapAnythingDialog::ensureModelAvailable() {
    auto* panel = currentPanel();
    if (panel == nullptr) return false;
    const QString filename = panel->modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        return QFileInfo::exists(panel->customModelPath->text());
    }
    const QString path = modelCacheDir() + QStringLiteral("/") + filename;
    MapAnythingModelEntry entry;
    MapAnythingHelpers::findModelByFilename(filename, &entry);
    const auto anchor = ecvAssetIntegrity::Anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(filename)};
    if (ecvAssetIntegrity::isVerified(
                path, anchor, 64 * 1024, /*requireGgufMagic=*/true,
                ecvAssetIntegrity::OnMiss::CheapChecksOnly, entry.sizeBytes)) {
        return true;
    }
    appendLog(tr("[MapAnything] Model not cached: %1 — starting the "
                 "download (%2).")
                      .arg(filename, ecvModelDownloader::formatFileSize(
                                             entry.sizeBytes)));
    if (m_downloadInProgress) {
        appendLog(tr("[Warning] A download is already in progress."));
        return false;
    }
    m_pendingAfterDownload = QStringLiteral("__run__");
    startDownload(entry);
    return false;
}

void MapAnythingDialog::startDownload(const MapAnythingModelEntry& model) {
    m_downloadInProgress = true;
    m_runBtn->setEnabled(false);
    ecvModelDownloader::Request request;
    request.url = model.downloadUrl;
    request.destPath = modelCacheDir() + QStringLiteral("/") + model.filename;
    request.requireGgufMagic = true;
    request.contentAnchor = ecvAssetIntegrity::Anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(model.filename)};
    request.ingestExactSize =
            request.contentAnchor.digestHex.isEmpty() ? model.sizeBytes : 0;
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &MapAnythingDialog::onModelDownloadProgress);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &MapAnythingDialog::onModelDownloadLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &MapAnythingDialog::onModelDownloadFinished);
    m_downloader->download(request);
}

void MapAnythingDialog::cancelDownload() {
    if (m_downloader != nullptr) m_downloader->cancel();
}

void MapAnythingDialog::onModelDownloadProgress(qint64 received, qint64 total) {
    m_downloadLabel->setText(
            ecvModelDownloader::formatDownloadProgress(received, total));
    const int percent = total > 0 ? int(received * 100 / total) : 0;
    setProgress(percent, 100);
}

void MapAnythingDialog::onModelDownloadLog(const QString& msg) {
    appendLog(msg);
}

void MapAnythingDialog::onModelDownloadFinished(bool ok,
                                                const QString& destPath) {
    m_downloadInProgress = false;
    m_runBtn->setEnabled(true);
    m_downloadLabel->clear();
    disconnect(m_downloader, nullptr, this, nullptr);
    if (!ok) {
        appendLog(tr("[Error] Model download failed: %1").arg(destPath));
        m_pendingAfterDownload.clear();
        return;
    }
    appendLog(tr("[MapAnything] Model cached: %1").arg(destPath));
    if (m_pendingAfterDownload == QStringLiteral("__run__")) {
        m_pendingAfterDownload.clear();
        onRun();
    }
}

void MapAnythingDialog::onUseTestData() {
    if (m_testDataBusy) {
        appendLog(
                tr("[Warning] A test-data download is already in "
                   "progress."));
        return;
    }
    const int index = m_testSceneCombo->currentIndex();
    if (index < 0 ||
        index >= (int)(sizeof(kSceneKinds) / sizeof(kSceneKinds[0]))) {
        return;
    }
    const Dataset kind = kSceneKinds[index];
    auto& repo = ecvTestDataRepository::instance();
    // Single-call chain: the shared repository owns the download→extract
    // state machine and settles through datasetReady (onTestDataReady then
    // fills the view list). No per-plugin ordering to get wrong.
    m_testDataBusy = true;
    m_pendingScene = kind;
    m_testDataBtn->setEnabled(false);
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    appendLog(tr("[MapAnything] Preparing the official test scene '%1' "
                 "(download/extract as needed)...")
                      .arg(info.displayName));
    repo.ensureDatasetReady(kind);
}

bool MapAnythingDialog::loadTestDataScene(Dataset kind) {
    const QString root = ecvTestDataRepository::extractPath(kind);
    const QStringList images =
            ecvTestDataRepository::getMapAnythingSceneImages(root);
    if (images.isEmpty()) {
        appendLog(tr("[Error] No images found under the extracted scene: %1")
                          .arg(root));
        return false;
    }
    m_imageList->clear();
    int added = 0;
    for (const QString& image : images) {
        if (added >= kMaxViews) {
            appendLog(tr("[Warning] Scene has %1 views; keeping the first "
                         "%2 (view cap).")
                              .arg(images.size())
                              .arg(kMaxViews));
            break;
        }
        m_imageList->addItem(image);
        ++added;
    }
    appendLog(tr("[MapAnything] Loaded %1 views from the test scene (%2).")
                      .arg(added)
                      .arg(ecvTestDataRepository::getDatasetInfo(kind)
                                   .displayName));
    updatePreview();
    return true;
}

void MapAnythingDialog::onTestDataProgress(int percent,
                                           const QString& statusText) {
    setProgress(percent, 100);
    m_taskStatusLabel->setText(statusText);
}

void MapAnythingDialog::onTestDataLog(const QString& message) {
    appendLog(message);
}

void MapAnythingDialog::onTestDataReady(bool success,
                                        ecvTestDataRepository::Dataset kind) {
    if (kind != m_pendingScene) return;
    m_testDataBusy = false;
    m_testDataBtn->setEnabled(true);
    setProgress(0, 100);
    m_taskStatusLabel->clear();
    if (!success) {
        appendLog(
                tr("[Error] Test-data preparation failed (see the console "
                   "log)."));
        return;
    }
    loadTestDataScene(kind);
}

void MapAnythingDialog::appendLog(const QString& msg) {
    // Shared AICore log route (CVLog → the MAIN-WINDOW console + the
    // session log file): plugins must not host their own log widgets —
    // the same routing qDA3 / qLingbotMap use.
    aicore_inference_log::log(msg);
    if (msg.startsWith(QStringLiteral("[Error]"))) {
        m_lastTaskError = msg;
        m_taskStatusLabel->setText(msg);
    }
}

void MapAnythingDialog::setProgress(int current, int total) {
    const int percent = total > 0 ? current * 100 / total : 0;
    m_progress->setValue(std::min(100, std::max(0, percent)));
    // Reconstruction-driven preview sync (qLingbotMap replay semantics:
    // the run owns the transport while it processes the view set — frame
    // i of the run shows shared-input view i, pausing manual playback).
    // The worker emits (0..N+2) where 1..N are the per-view decode steps;
    // clamp the shown frame to the actual list range.
    if (m_reconstructionOwnsPreview && total > 0 && current > 0) {
        onPreviewPause();
        showFrame(std::min(current - 1, m_imageList->count() - 1));
    }
}

void MapAnythingDialog::setTaskStage(const QString& stage, int percent) {
    m_taskStatusLabel->setText(stage);
    if (percent >= 0) m_progress->setValue(percent);
}

void MapAnythingDialog::setRunning(bool running) {
    m_taskRunning = running;
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    m_testDataBtn->setEnabled(!running && !m_testDataBusy);
    // Run takes the preview over; the transport buttons follow the run
    // state (manual playback resumes only after the task finishes).
    m_reconstructionOwnsPreview = running;
    m_previewPlayBtn->setEnabled(!running);
    m_previewPauseBtn->setEnabled(true);
    m_previewPrevBtn->setEnabled(!running);
    m_previewResetBtn->setEnabled(true);
    if (running) {
        onPreviewPause();
        showFrame(0);  // deterministic start: the run walks views in order
        // Busy indicator: the single forward is one blocking GPU call with
        // no per-frame progress - a pulsing bar reads as "working" instead
        // of a frozen percentage that invites killing the app.
        m_progress->setRange(0, 0);
    } else {
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    }
}

void MapAnythingDialog::applyDbTreeSelection(const QStringList& imageNames) {
    for (const QString& name : imageNames) {
        const QString entry = QStringLiteral("db://") + name;
        if (m_imageList->count() >= kMaxViews) break;
        if (!m_imageList->findItems(entry, Qt::MatchExactly).isEmpty()) {
            continue;
        }
        m_imageList->addItem(entry);
    }
    updatePreview();
}

void MapAnythingDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qMapAnything"));
    m_deviceCombo->setCurrentText(
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString());
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    const QByteArray splitterState =
            settings.value(QStringLiteral("splitterState")).toByteArray();
    if (!splitterState.isEmpty()) {
        m_bodySplitter->restoreState(splitterState);
        m_splitterRestored = true;
    }
    const QStringList variants = MapAnythingHelpers::catalogVariants();
    const QString variant =
            settings.value(QStringLiteral("variant"), variants.value(0))
                    .toString();
    for (int i = 0; i < m_modelList->count(); ++i) {
        if (m_modelList->item(i)->text() == variant) {
            m_modelList->setCurrentRow(i);
            break;
        }
    }
    settings.endGroup();
}

void MapAnythingDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qMapAnything"));
    settings.setValue(QStringLiteral("device"), m_deviceCombo->currentText());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.setValue(QStringLiteral("splitterState"),
                      m_bodySplitter->saveState());
    if (m_modelList->currentItem() != nullptr) {
        settings.setValue(QStringLiteral("variant"),
                          m_modelList->currentItem()->text());
    }
    settings.endGroup();
}

void MapAnythingDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    if (!m_splitterSized) {
        m_splitterSized = true;
        if (!m_splitterRestored && m_modelList != nullptr) {
            // Content-driven default: the widest variant label + padding.
            int width = m_modelList->sizeHintForColumn(0) + 48;
            width = std::min(width, this->width() / 3);
            m_bodySplitter->setSizes({width, this->width() - width});
        }
    }
}

void MapAnythingDialog::closeEvent(QCloseEvent* event) {
    if (m_taskRunning) {
        const auto answer = QMessageBox::question(
                this, tr("Task running"),
                tr("A reconstruction task is running. Close anyway?"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        emit cancelRequested();
    }
    saveSettings();
    QDialog::closeEvent(event);
}

void MapAnythingDialog::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape && m_taskRunning) {
        // Esc intercept: confirm before closing when a task is running
        // (same contract as the qYOLO dialog).
        event->accept();
        return;
    }
    QDialog::keyPressEvent(event);
}
