// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "QuerySplatDialog.h"

#include <aicore/inference_log.h>
#include <aicore/map_anything_capi.h>
#include <ecvAICoreUiHelper.h>
#include <ecvAssetIntegrity.h>
#include <ecvClickableImageLabel.h>
#include <ecvMainAppInterface.h>
#include <ecvModelDownloader.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

namespace {

QString deviceLabel(const QString& id) {
    if (id == QStringLiteral("auto"))
        return QObject::tr("Auto (CUDA > Vulkan > Metal > CPU)");
    return id.toUpper();
}

// Scene combo index <-> shared multi-view scene dataset. These are the SAME
// map_anything_vggt_test_data fixtures the MapAnything plugin consumes —
// QuerySplat runs on any multi-view set, so both plugins share the whole
// scene catalog (courtyard's first four frames are byte-identical to the
// upstream QuerySplat eval inputs).
const ecvTestDataRepository::Dataset kSceneKinds[] = {
        ecvTestDataRepository::Dataset::MapAnythingCourtyard,
        ecvTestDataRepository::Dataset::MapAnythingDeliveryArea,
        ecvTestDataRepository::Dataset::MapAnythingElectro,
        ecvTestDataRepository::Dataset::MapAnythingFacade,
        ecvTestDataRepository::Dataset::MapAnythingKicker,
        ecvTestDataRepository::Dataset::MapAnythingMeadow,
        ecvTestDataRepository::Dataset::MapAnythingOffice,
        ecvTestDataRepository::Dataset::MapAnythingPipes,
        ecvTestDataRepository::Dataset::MapAnythingPlayground,
        ecvTestDataRepository::Dataset::MapAnythingRelief,
        ecvTestDataRepository::Dataset::MapAnythingRelief2,
        ecvTestDataRepository::Dataset::MapAnythingTerrace,
        ecvTestDataRepository::Dataset::MapAnythingTerrains};

// Demo fill keeps the upstream benchmark geometry (the first frames of the
// selected scene); the rest stay available through Add Images.
constexpr int kDemoFillMaxViews = 4;

// Short combo labels, matching the MapAnything dialog's scene rows (the
// repository's display name keeps the "MapAnything" prefix for logs).
const char* const kSceneLabels[] = {
        "courtyard (ETH3D)", "delivery_area (ETH3D)", "electro (ETH3D)",
        "facade (ETH3D)",    "kicker (ETH3D)",        "meadow (ETH3D)",
        "office (ETH3D)",    "pipes (ETH3D)",         "playground (ETH3D)",
        "relief (ETH3D)",    "relief_2 (ETH3D)",      "terrace (ETH3D)",
        "terrains (ETH3D)"};
static_assert(sizeof(kSceneLabels) / sizeof(kSceneLabels[0]) ==
                      sizeof(kSceneKinds) / sizeof(kSceneKinds[0]),
              "scene label row must match the dataset row");

}  // namespace

QuerySplatDialog::QuerySplatDialog(QWidget* parent) : QDialog(parent) {
    connect(this, &QDialog::finished, this, &QuerySplatDialog::onCancel);
    buildUi();
    refreshModelList();
}

QuerySplatDialog::~QuerySplatDialog() = default;

void QuerySplatDialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
}

QString QuerySplatDialog::decoderCacheDir() const {
    return QuerySplatHelpers::decoderCacheDir();
}

void QuerySplatDialog::buildUi() {
    setWindowTitle(tr("QuerySplat Gaussian Prediction"));
    // Fixed-area height budget (the geometry contract: input/actions/
    // progress stay visible outside the config scroll):
    //   input page ~ 150 (list/preview capped at 140 + two button rows)
    //   action row ~ 32, progress ~ 22, status label ~ 20, margins ~ 40
    //   => ~ 264 px; the minimum window height reserves ~ 280 px of config
    //   scroll viewport on top of that so the lowest checkbox row is
    //   reachable without resizing. The config section itself scrolls.
    // The two-row button split keeps every label fully visible at the
    // minimum width on all DPI scales (six controls on one row force text
    // elision at 640-700 px widths).
    setMinimumSize(700, 660);
    resize(760, 800);

    auto* rootLayout = new QVBoxLayout(this);

    // ---- fixed input area ----
    rootLayout->addWidget(buildInputPage());

    // ---- configuration scroll area ----
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidget(buildConfigPage());
    rootLayout->addWidget(scroll, /*stretch=*/1);

    // ---- fixed action/progress area ----
    auto* actionRow = new QHBoxLayout();
    m_runBtn = new QPushButton(tr("Run"), this);
    m_runBtn->setDefault(true);
    m_cancelBtn = new QPushButton(tr("Cancel"), this);
    m_cancelBtn->setEnabled(false);
    actionRow->addWidget(m_runBtn);
    actionRow->addWidget(m_cancelBtn);
    actionRow->addStretch(1);
    m_downloadLabel = new QLabel(this);
    actionRow->addWidget(m_downloadLabel);
    rootLayout->addLayout(actionRow);

    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    rootLayout->addWidget(m_progress);

    m_taskStatusLabel = new QLabel(this);
    rootLayout->addWidget(m_taskStatusLabel);

    // Shared HTTPS model downloader. startModelDownload() wires its
    // progress/log/finished signals per request and disconnects on finish, so
    // only the object itself has to live here as a child of the dialog.
    // Without this the pointer stayed null and clicking Run on a cold cache
    // dereferenced it in ecvModelDownloader::download (macOS-only repro: the
    // Linux dev boxes already had the decoder GGUF cached, so the download
    // branch was never reached).
    m_downloader = new ecvModelDownloader(this);
    connect(this, &QDialog::finished, m_downloader,
            &ecvModelDownloader::cancel);

    connect(m_runBtn, &QPushButton::clicked, this, &QuerySplatDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this,
            &QuerySplatDialog::onCancel);
    connect(m_addBtn, &QPushButton::clicked, this,
            &QuerySplatDialog::onAddFiles);
    connect(m_removeBtn, &QPushButton::clicked, this,
            &QuerySplatDialog::onRemoveSelected);
    connect(m_clearBtn, &QPushButton::clicked, this,
            &QuerySplatDialog::onClearImages);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &QuerySplatDialog::onUseTestData);
    connect(m_imageList, &QListWidget::itemSelectionChanged, this,
            &QuerySplatDialog::updatePreview);
    connect(&ecvTestDataRepository::instance(),
            &ecvTestDataRepository::datasetReady, this,
            &QuerySplatDialog::onTestDataReady);
    // Download/extract feedback for the shared test-data preparation (the
    // same repository signals the MapAnything dialog consumes).
    connect(&ecvTestDataRepository::instance(),
            &ecvTestDataRepository::downloadProgress, this,
            &QuerySplatDialog::onTestDataProgress);
    connect(&ecvTestDataRepository::instance(),
            &ecvTestDataRepository::downloadLogMessage, this,
            &QuerySplatDialog::onTestDataLog);
}

QWidget* QuerySplatDialog::buildInputPage() {
    auto* box = new QWidget(this);
    auto* layout = new QHBoxLayout(box);
    layout->setContentsMargins(0, 0, 0, 0);

    // list column — same 2:1 split as the MapAnything input area
    auto* listColumn = new QVBoxLayout();
    m_imageList = new QListWidget(box);
    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    // All listed images form ONE multi-view scene (the upstream --images
    // semantics): the worker issues a single run_images(S views) call, so
    // 1 image = single-view prediction and N images = an S=N scene.
    m_imageList->setToolTip(
            tr("Every listed image is a view of the SAME scene: the run "
               "feeds all of them to the network together (multi-view by "
               "default, S = list size). Add 2+ overlapping views of one "
               "scene for the best reconstruction."));
    // Same cap as the MapAnything input list: the fixed input area must not
    // squeeze the configuration scroll below its usable height.
    m_imageList->setMinimumHeight(96);
    m_imageList->setMaximumHeight(140);
    listColumn->addWidget(m_imageList, 1);

    auto* btnRow = new QHBoxLayout();
    m_addBtn = new QPushButton(tr("Add Images..."), box);
    m_removeBtn = new QPushButton(tr("Remove Selected"), box);
    m_clearBtn = new QPushButton(tr("Clear"), box);
    btnRow->addWidget(m_addBtn);
    btnRow->addWidget(m_removeBtn);
    btnRow->addWidget(m_clearBtn);
    btnRow->addStretch(1);
    listColumn->addLayout(btnRow);

    // sample-data row: scene combo + shared teal button get their own row —
    // sharing the action row squeezed both below their size hints and
    // elided their labels ("Remove Selecte…", "ry sample da…").
    m_testSceneCombo = new QComboBox(box);
    for (size_t i = 0; i < sizeof(kSceneLabels) / sizeof(kSceneLabels[0]);
         ++i) {
        m_testSceneCombo->addItem(QString::fromLatin1(kSceneLabels[i]));
    }
    // Shared teal "Try sample data" button (ecvAICoreUi) — the same control
    // every AICore plugin uses for its sample-data row.
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(box);
    m_testDataBtn->setToolTip(
            tr("Download and load an official MapAnything ETH3D test scene "
               "into the shared multi-view input list"));
    m_testDataBtn->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
    auto* sampleRow = new QHBoxLayout();
    sampleRow->addWidget(m_testSceneCombo, /*stretch=*/1);
    sampleRow->addWidget(m_testDataBtn);
    listColumn->addLayout(sampleRow);
    layout->addLayout(listColumn, 2);

    // preview column — the shared ecvClickableImageLabel the MapAnything
    // dialog uses (no per-plugin preview implementation); capped to the list
    // height so the preview column cannot stretch the fixed input area and
    // squeeze the config scroll below its usable height.
    auto* previewColumn = new QVBoxLayout();
    m_previewLabel = new ecvClickableImageLabel(box);
    m_previewLabel->setMinimumSize(150, 96);
    m_previewLabel->setMaximumHeight(140);
    previewColumn->addWidget(m_previewLabel, 1);
    layout->addLayout(previewColumn, 1);
    return box;
}

QWidget* QuerySplatDialog::buildConfigPage() {
    auto* box = new QWidget(this);
    auto* form = new QFormLayout(box);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    m_modelCombo = new QComboBox(box);
    // Long catalog labels live in the popup only: the combo elides so the
    // form's minimum width stays fixed (no horizontal overflow).
    m_modelCombo->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_modelCombo->setMinimumContentsLength(12);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) { m_downloadLabel->clear(); });
    form->addRow(tr("Decoder model"), m_modelCombo);

    m_deviceCombo = new QComboBox(box);
    for (const char* d : {"auto", "cuda", "vulkan", "metal", "cpu"})
        m_deviceCombo->addItem(deviceLabel(QString::fromLatin1(d)),
                               QString::fromLatin1(d));
    form->addRow(tr("Device"), m_deviceCombo);

    m_threadsSpin = new QSpinBox(box);
    m_threadsSpin->setRange(0, 256);
    // The sentinel keeps the number in view: 0 IS the value (backend-
    // picked threads); a bare "Backend default" read as a broken text
    // field (user report 2026-11-03).
    m_threadsSpin->setSpecialValueText(tr("Auto (0)"));
    m_threadsSpin->setToolTip(
            tr("CPU worker threads for CPU inference. 0 = backend default\n"
               "(ggml picks the thread count); 1-256 pins an explicit\n"
               "limit. Ignored by GPU backends."));
    form->addRow(tr("CPU threads"), m_threadsSpin);

    m_imageSizeSpin = new QSpinBox(box);
    m_imageSizeSpin->setRange(0, 2048);
    m_imageSizeSpin->setSingleStep(32);
    m_imageSizeSpin->setSpecialValueText(tr("Auto (512)"));
    m_imageSizeSpin->setToolTip(
            tr("Square processing resolution fed to the network. 0 = the\n"
               "resolution stored in the decoder checkpoint — 512 for the\n"
               "released paper checkpoints (the number shown here); a\n"
               "future checkpoint with a different stored size wins over\n"
               "this label. Other values override it explicitly."));
    form->addRow(tr("Image size"), m_imageSizeSpin);

    m_opacitySpin = new QDoubleSpinBox(box);
    m_opacitySpin->setRange(0.0, 1.0);
    m_opacitySpin->setSingleStep(0.01);
    m_opacitySpin->setDecimals(3);
    m_opacitySpin->setValue(0.05);
    form->addRow(tr("Opacity threshold"), m_opacitySpin);

    m_bgSpin = new QDoubleSpinBox(box);
    m_bgSpin->setRange(0.0, 1.0);
    m_bgSpin->setSingleStep(0.1);
    m_bgSpin->setDecimals(2);
    m_bgSpin->setValue(0.0);
    m_bgSpin->setToolTip(
            tr("Grayscale background of the rendered views (gsplat "
               "rasterizer); 0 = black, 1 = white. Official default: 0."));
    form->addRow(tr("Render background"), m_bgSpin);

    m_ttoCheck = new QCheckBox(tr("Test-time optimization (TTO)"), box);
    m_ttoCheck->setToolTip(
            tr("Optional refinement after the feed-forward prediction: the\n"
               "gaussians are rendered with the gsplat rasterizer and\n"
               "compared against the input views with an L1 + SSIM + LPIPS\n"
               "loss, then updated by gradient descent for the given\n"
               "number of steps (needs the lpips-vgg-f32 model). Sharper\n"
               "results at roughly one render + backward pass per step.\n"
               "The backward graph is sized by the checkpoint's 8192\n"
               "gaussian tokens (resolution-independent) and can exceed a\n"
               "24 GB GPU at 512x512 — CPU or a larger device then. Upstream\n"
               "default is off; the refined gaussians replace the\n"
               "feed-forward ones in every output."));
    form->addRow(QString(), m_ttoCheck);
    m_ttoStepsSpin = new QSpinBox(box);
    m_ttoStepsSpin->setRange(1, 500);
    m_ttoStepsSpin->setValue(20);
    form->addRow(tr("TTO steps"), m_ttoStepsSpin);
    m_ttoLrSpin = new QDoubleSpinBox(box);
    m_ttoLrSpin->setRange(1e-5, 1e-1);
    m_ttoLrSpin->setDecimals(5);
    m_ttoLrSpin->setSingleStep(1e-3);
    m_ttoLrSpin->setValue(5e-3);
    form->addRow(tr("TTO learning rate"), m_ttoLrSpin);
    m_ttoLpipsSpin = new QDoubleSpinBox(box);
    m_ttoLpipsSpin->setRange(0.0, 1.0);
    m_ttoLpipsSpin->setSingleStep(0.01);
    m_ttoLpipsSpin->setValue(0.05);
    form->addRow(tr("TTO LPIPS weight"), m_ttoLpipsSpin);
    m_ttoStepsSpin->setEnabled(false);
    m_ttoLrSpin->setEnabled(false);
    m_ttoLpipsSpin->setEnabled(false);
    connect(m_ttoCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_ttoStepsSpin->setEnabled(on);
        m_ttoLrSpin->setEnabled(on);
        m_ttoLpipsSpin->setEnabled(on);
    });

    m_camerasCheck = new QCheckBox(tr("Predicted cameras (per view)"), box);
    m_camerasCheck->setChecked(true);
    m_depthViewsCheck = new QCheckBox(tr("Depth rasters (per view)"), box);
    m_depthViewsCheck->setChecked(true);
    m_renderViewsCheck =
            new QCheckBox(tr("Rendered views (gsplat rasterizer)"), box);
    m_renderViewsCheck->setChecked(true);
    m_pointCloudCheck = new QCheckBox(tr("Depth point cloud into DB"), box);
    m_pointCloudCheck->setChecked(true);
    m_plyCheck = new QCheckBox(tr("Export 3DGS PLY"), box);
    m_plyCheck->setChecked(true);
    // The five export switches share a two-column grid: five QFormLayout
    // rows would dominate the config scroll viewport and hide the lowest
    // rows behind the fold even at the minimum window height.
    auto* outputsGrid = new QGridLayout();
    outputsGrid->setContentsMargins(0, 0, 0, 0);
    outputsGrid->addWidget(m_camerasCheck, 0, 0);
    outputsGrid->addWidget(m_depthViewsCheck, 0, 1);
    outputsGrid->addWidget(m_renderViewsCheck, 1, 0);
    outputsGrid->addWidget(m_pointCloudCheck, 1, 1);
    outputsGrid->addWidget(m_plyCheck, 2, 0);
    auto* outputsHolder = new QWidget(box);
    outputsHolder->setLayout(outputsGrid);
    form->addRow(tr("Outputs"), outputsHolder);
    return box;
}

void QuerySplatDialog::refreshModelList() {
    if (!m_modelCombo) return;
    const QString previous = m_modelCombo->currentData().toString();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    const auto decoders = QuerySplatHelpers::decoderModels();
    for (const auto& m : decoders) {
        m_modelCombo->addItem(tr("%1 (%2)").arg(m.displayName, m.quantNote),
                              m.filename);
    }
    if (!previous.isEmpty()) {
        const int idx = m_modelCombo->findData(previous);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    m_modelCombo->blockSignals(false);
}

void QuerySplatDialog::fillImages(const QStringList& paths) {
    for (const QString& p : paths) {
        if (!m_imageList->findItems(p, Qt::MatchExactly).isEmpty()) continue;
        m_imageList->addItem(p);
    }
    updatePreview();
}

void QuerySplatDialog::updatePreview() {
    // Same funnel semantics as the MapAnything preview: the selected item
    // drives the frame, falling back to the first row when nothing is
    // selected; empty list clears.
    QListWidgetItem* item = m_imageList->currentItem();
    if (!item && m_imageList->count() > 0) item = m_imageList->item(0);
    if (!item) {
        m_previewLabel->clearPreview();
        return;
    }
    const QImage img(item->text());
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        return;
    }
    m_previewLabel->setPreviewImage(img, m_previewLabel->size());
}

void QuerySplatDialog::onAddFiles() {
    // Continue in the directory of the already-listed images.
    const QString current =
            m_imageList->count() ? m_imageList->item(0)->text() : QString();
    const QStringList files = ecvAICoreUi::browseInputImages(
            this, current, tr("Select input images"));
    fillImages(files);
}

void QuerySplatDialog::onRemoveSelected() {
    for (QListWidgetItem* item : m_imageList->selectedItems()) delete item;
    updatePreview();
}

void QuerySplatDialog::onClearImages() {
    m_imageList->clear();
    updatePreview();
}

void QuerySplatDialog::onUseTestData() {
    if (m_testDataBusy) {
        appendLog(
                tr("[Warning] A test-data download is already in "
                   "progress."));
        return;
    }
    // absence of the cache is a recoverable state: shared preparation, not
    // an Error (an Error dialog here would be a modal the user must
    // dismiss before anything happens)
    const int scene =
            std::min(std::max(m_testSceneCombo->currentIndex(), 0),
                     int(sizeof(kSceneKinds) / sizeof(kSceneKinds[0])) - 1);
    m_pendingTestDataset = kSceneKinds[scene];
    const auto info =
            ecvTestDataRepository::getDatasetInfo(m_pendingTestDataset);
    m_testDataBusy = true;
    m_testDataBtn->setEnabled(false);
    appendLog(tr("[QS] Preparing the shared multi-view test scene (%1)...")
                      .arg(info.displayName));
    ecvTestDataRepository::instance().ensureDatasetReady(m_pendingTestDataset);
}

void QuerySplatDialog::onTestDataReady(bool ok,
                                       ecvTestDataRepository::Dataset kind) {
    if (!m_testDataBusy || kind != m_pendingTestDataset) return;
    m_testDataBusy = false;
    m_testDataBtn->setEnabled(true);
    setProgress(0, 100);
    m_taskStatusLabel->clear();
    if (!ok) {
        appendLog(tr("[Error] Test-data preparation failed."));
        return;
    }
    // The shared map_anything_vggt_test_data bundles keep their frames under
    // the scene directory; the repository's recursive scan returns them in
    // deterministic name order (the same helper qMapAnything consumes).
    const QString root = ecvTestDataRepository::extractPath(kind);
    const QStringList images =
            ecvTestDataRepository::getMapAnythingSceneImages(root);
    if (images.isEmpty()) {
        appendLog(tr("[Error] No images found under %1").arg(root));
        return;
    }
    // Demo fill keeps the upstream benchmark geometry: the first frames of
    // the scene (courtyard's first four are byte-identical to the upstream
    // QuerySplat eval inputs); the remaining frames stay available through
    // Add Images.
    const int fill = std::min(int(images.size()), kDemoFillMaxViews);
    fillImages(images.mid(0, fill));
    appendLog(tr("[QS] Filled %1 test image(s) from %2 (%3 available).")
                      .arg(fill)
                      .arg(ecvTestDataRepository::getDatasetInfo(kind)
                                   .displayName)
                      .arg(images.size()));
}

void QuerySplatDialog::onTestDataProgress(int percent,
                                          const QString& statusText) {
    setProgress(percent, 100);
    m_taskStatusLabel->setText(statusText);
}

void QuerySplatDialog::onTestDataLog(const QString& message) {
    appendLog(message);
}

bool QuerySplatDialog::ensureEncoderAvailable() {
    // The encoder is the map_anything-shared VGGT-Omega GGUF. Verify it the
    // same way (integrity anchor from the pinned digest table) but route the
    // download through this dialog's shared ecvModelDownloader when absent.
    const QString encPath = QuerySplatHelpers::encoderDefaultPath();
    if (encPath.isEmpty()) {
        appendLog(
                tr("[Error] The VGGT-Omega encoder catalog is "
                   "unavailable (AICore map_anything catalog missing)."));
        return false;
    }
    if (QFileInfo::exists(encPath)) {
        const QString filename = QFileInfo(encPath).fileName();
        const auto anchor = ecvAssetIntegrity::Anchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(filename)};
        if (ecvAssetIntegrity::isVerified(
                    encPath, anchor, 64 * 1024, /*requireGgufMagic=*/true,
                    ecvAssetIntegrity::OnMiss::CheapChecksOnly, 0)) {
            return true;
        }
        appendLog(tr("[Warning] Encoder digest mismatch, re-downloading."));
    }
    // resolve the download URL from the map_anything catalog through the
    // shared AICore entries
    const QString filename = QFileInfo(encPath).fileName();
    // the map_anything catalog lives behind the AICore C API
    const int32_t n = aicore_map_anything_model_count();
    for (int32_t i = 0; i < n; ++i) {
        const auto* e = aicore_map_anything_model_at(i);
        if (e && filename == QString::fromUtf8(e->filename)) {
            QuerySplatModelEntry row;
            row.filename = filename;
            row.downloadUrl = QString::fromUtf8(e->download_url);
            row.sizeBytes = e->size_bytes;
            m_pendingAfterDownload = QStringLiteral("__run__");
            startModelDownload(row);
            return false;
        }
    }
    appendLog(tr("[Error] Encoder model '%1' is not in the map_anything "
                 "catalog.")
                      .arg(filename));
    return false;
}

bool QuerySplatDialog::ensureModelsAvailable() {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return false;
    QuerySplatModelEntry entry;
    QuerySplatHelpers::findModelByFilename(filename, &entry);
    const QString path = decoderCacheDir() + QStringLiteral("/") + filename;
    const auto anchor = ecvAssetIntegrity::Anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(filename)};
    if (!ecvAssetIntegrity::isVerified(
                path, anchor, 64 * 1024, /*requireGgufMagic=*/true,
                ecvAssetIntegrity::OnMiss::CheapChecksOnly, entry.sizeBytes)) {
        appendLog(tr("[QS] Decoder model not cached: %1 — starting the "
                     "download (%2).")
                          .arg(filename, ecvModelDownloader::formatFileSize(
                                                 entry.sizeBytes)));
        m_pendingDownloads.prepend(entry);
    }
    // LPIPS weights (TTO only, small): check but never block the
    // feed-forward path on them.
    if (m_ttoCheck->isChecked()) {
        const QString lp =
                decoderCacheDir() + QStringLiteral("/lpips-vgg-f32.gguf");
        if (!ecvAssetIntegrity::isModelVerified(lp)) {
            QuerySplatModelEntry lpEntry;
            if (!QuerySplatHelpers::findModelByFilename(
                         QStringLiteral("lpips-vgg-f32.gguf"), &lpEntry)
                         .isEmpty()) {
                m_pendingDownloads.append(lpEntry);
            }
        }
    }
    if (!m_pendingDownloads.isEmpty()) {
        m_pendingAfterDownload = QStringLiteral("__run__");
        queueNextDownload();
        return false;
    }
    return ensureEncoderAvailable();
}

void QuerySplatDialog::queueNextDownload() {
    if (m_pendingDownloads.isEmpty()) {
        // all decoder-side models present: continue the chain
        if (m_pendingAfterDownload == QStringLiteral("__run__") &&
            ensureEncoderAvailable()) {
            m_pendingAfterDownload.clear();
            onRun();
        }
        return;
    }
    startModelDownload(m_pendingDownloads.takeFirst());
}

void QuerySplatDialog::startModelDownload(const QuerySplatModelEntry& model) {
    m_downloadInProgress = true;
    m_cancelBtn->setEnabled(true);
    m_runBtn->setEnabled(false);
    ecvModelDownloader::Request request;
    request.url = model.downloadUrl;
    request.destPath = decoderCacheDir() + QStringLiteral("/") + model.filename;
    request.requireGgufMagic = model.role != QStringLiteral("lpips");
    request.contentAnchor = ecvAssetIntegrity::Anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(model.filename)};
    request.ingestExactSize =
            request.contentAnchor.digestHex.isEmpty() ? model.sizeBytes : 0;
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &QuerySplatDialog::onModelDownloadProgress);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &QuerySplatDialog::onModelDownloadLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &QuerySplatDialog::onModelDownloadFinished);
    m_downloader->download(request);
}

void QuerySplatDialog::onModelDownloadProgress(qint64 received, qint64 total) {
    m_downloadLabel->setText(
            ecvModelDownloader::formatDownloadProgress(received, total));
    const int percent = total > 0 ? int(received * 100 / total) : 0;
    setProgress(percent, 100);
}

void QuerySplatDialog::onModelDownloadLog(const QString& msg) {
    appendLog(msg);
}

void QuerySplatDialog::onModelDownloadFinished(bool ok,
                                               const QString& destPath) {
    m_downloadInProgress = false;
    m_cancelBtn->setEnabled(false);
    m_runBtn->setEnabled(true);
    m_downloadLabel->clear();
    disconnect(m_downloader, nullptr, this, nullptr);
    if (!ok) {
        appendLog(tr("[Error] Model download failed: %1").arg(destPath));
        m_pendingAfterDownload.clear();
        m_pendingDownloads.clear();
        return;
    }
    appendLog(tr("[QS] Model cached: %1").arg(destPath));
    queueNextDownload();
}

QuerySplatDialog::Settings QuerySplatDialog::getSettings() const {
    Settings s;
    s.decoderPath =
            decoderCacheDir() + QStringLiteral("/") +
            (m_modelCombo ? m_modelCombo->currentData().toString() : QString());
    s.encoderPath = QuerySplatHelpers::encoderDefaultPath();
    for (int i = 0; i < m_imageList->count(); ++i)
        s.imagePaths << m_imageList->item(i)->text();
    s.threads = m_threadsSpin ? m_threadsSpin->value() : 0;
    s.device = m_deviceCombo ? m_deviceCombo->currentData().toString()
                             : QStringLiteral("auto");
    s.imageSize = m_imageSizeSpin ? m_imageSizeSpin->value() : 0;
    s.ttoEnabled = m_ttoCheck && m_ttoCheck->isChecked();
    s.ttoSteps = m_ttoStepsSpin ? m_ttoStepsSpin->value() : 20;
    s.ttoLr = m_ttoLrSpin ? m_ttoLrSpin->value() : 5e-3;
    s.ttoLpipsWeight = m_ttoLpipsSpin ? m_ttoLpipsSpin->value() : 0.05;
    s.opacityThreshold = m_opacitySpin ? float(m_opacitySpin->value()) : 0.05f;
    s.renderBackground = m_bgSpin ? float(m_bgSpin->value()) : 0.0f;
    s.addDepthViews = m_depthViewsCheck && m_depthViewsCheck->isChecked();
    s.addRenderViews = m_renderViewsCheck && m_renderViewsCheck->isChecked();
    s.addCameras = m_camerasCheck && m_camerasCheck->isChecked();
    s.addDepthPointCloud = m_pointCloudCheck && m_pointCloudCheck->isChecked();
    s.exportPly = m_plyCheck && m_plyCheck->isChecked();
    return s;
}

void QuerySplatDialog::appendLog(const QString& msg) {
    // Shared AICore log route (CVLog -> the MAIN-WINDOW console + session
    // log): plugins must not host their own log widgets.
    aicore_inference_log::log(msg);
    if (msg.startsWith(QStringLiteral("[Error]"))) {
        m_lastTaskError = msg;
        m_taskStatusLabel->setText(msg);
    }
}

void QuerySplatDialog::setProgress(int current, int total) {
    m_progress->setMaximum(total);
    m_progress->setValue(current);
}

void QuerySplatDialog::setTaskStage(const QString& stage, int percent) {
    m_taskStatusLabel->setText(stage);
    if (percent >= 0) setProgress(percent, 100);
}

void QuerySplatDialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    m_addBtn->setEnabled(!running);
    m_testDataBtn->setEnabled(!running && !m_testDataBusy);
    if (!running && !m_lastTaskError.isEmpty()) {
        m_taskStatusLabel->setText(m_lastTaskError);
    }
}

void QuerySplatDialog::onRun() {
    m_lastTaskError.clear();
    if (m_imageList->count() < 1) {
        appendLog(tr("[Warning] Add at least one input image first."));
        return;
    }
    if (!ensureModelsAvailable()) return;
    emit runRequested(getSettings());
}

void QuerySplatDialog::onCancel() {
    m_pendingAfterDownload.clear();
    m_pendingDownloads.clear();
    m_testDataBusy = false;
    m_testDataBtn->setEnabled(true);
    if (m_downloadInProgress) m_downloader->cancel();
    emit cancelRequested();
}
