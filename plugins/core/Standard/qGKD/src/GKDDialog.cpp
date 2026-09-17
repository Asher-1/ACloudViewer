// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDDialog.h"

#include <QtCompat.h>

#include <QCloseEvent>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSettings>
#include <QSplitter>
#include <QTimer>
#include <QVBoxLayout>

#include "ecvAICoreUiHelper.h"
#include "ecvAssetIntegrity.h"
#include "ecvMainAppInterface.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/gkd_capi.h"
#include "aicore/inference_log.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

// QListWidgetItem data role carrying the QStackedWidget page index behind
// a mode-list entry (section headers are disabled and carry no data).
constexpr int kModeStackIndexRole = Qt::UserRole + 10;
// QListWidgetItem data role carrying the full-resolution image for the
// click-to-enlarge preview (the list icon is only a scaled thumbnail).
constexpr int kDbFullImageRole = Qt::UserRole + 1;

// Test-data bundle for every mode: the official GKDT demo bundle covers
// the single-object prompt modes AND the multi-object quadruped demo
// (alpaca_150.jpg), so one cached archive serves the whole dialog.
ecvTestDataRepository::Dataset datasetForMode(const QString& mode) {
    (void)mode;
    return ecvTestDataRepository::Dataset::GeneralKeypointDetection;
}

}  // namespace

GKDDialog::GKDDialog(QWidget* parent) : QDialog(parent) {
    setupUi();
    // Populate first, then restore: the persisted selections are looked up
    // in populated combos (the old single-page dialog restored into an
    // empty combo and lost the choice when populate ran afterwards).
    populateYoloModelCombo();
    populateModelCombo();
    loadSettings();
    // Content-driven minimum (font / DPI aware) instead of hard-coded
    // pixels, so the dialog adapts to any platform and screen resolution.
    const QSize hint = minimumSizeHint();
    setMinimumSize(ecvAICoreUi::dpiScaled(hint.width()),
                   ecvAICoreUi::dpiScaled(hint.height()));
}

GKDDialog::~GKDDialog() {
    if (m_downloader && m_downloadInProgress) m_downloader->cancel();
    saveSettings();
}

void GKDDialog::setupUi() {
    setWindowTitle(tr("GKD — General Keypoint Detection (GKDT)"));

    auto* rootLayout = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(rootLayout);

    // ---- global model group (the same GKD GGUF serves every mode) ----
    auto* modelGroup = new QGroupBox(tr("Model"), this);
    auto* modelGrid = new QGridLayout(modelGroup);
    ecvAICoreUi::setupFormGrid(modelGrid);
    m_modelCombo = new QComboBox(modelGroup);
    modelGrid->addWidget(ecvAICoreUi::makeLabel(tr("GKD model:"), 110), 0, 0);
    modelGrid->addWidget(m_modelCombo, 0, 1);
    auto* browseModelBtn =
            ecvAICoreUi::makeBrowseBtn(tr("Custom…"), modelGroup);
    modelGrid->addWidget(browseModelBtn, 0, 2);
    m_customModelPath = new QLineEdit(modelGroup);
    m_customModelPath->setReadOnly(true);
    m_customModelRow = new QWidget(modelGroup);
    {
        auto* row = new QHBoxLayout(m_customModelRow);
        row->setContentsMargins(ecvAICoreUi::rowMargins());
        row->addWidget(new QLabel(tr("Custom model:"), m_customModelRow));
        row->addWidget(m_customModelPath, 1);
        m_customModelRow->setVisible(false);
        modelGrid->addWidget(m_customModelRow, 1, 0, 1, 3);
    }
    rootLayout->addWidget(modelGroup);

    // ---- global runtime row: rendered once above the mode list; the
    // device / threads configure every mode panel alike (qYOLO layout) ----
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_deviceCombo->setMaximumWidth(ecvAICoreUi::dpiScaled(220));
    const int deviceCount =
#ifdef AICore_ENABLED
            aicore_device_count();
#else
            0;
#endif
    for (int i = 0; i < deviceCount; ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev) continue;
        m_deviceCombo->addItem(QString::fromUtf8(dev->label),
                               QString::fromUtf8(dev->id));
        if (dev->is_default) m_deviceCombo->setCurrentIndex(i);
    }
    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 128);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = backend default"));
    ecvAICoreUi::setCompactSpin(m_threads);
    auto* runtimeRow =
            ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads, this);
    rootLayout->addWidget(runtimeRow);

    // ---- body: mode list on the left, one panel per mode on the right
    // (qYOLO layout). A splitter (not a fixed-width list) lets users drag
    // the pane wider for localized texts; the default split is
    // content-driven (applyModeListDefaultWidth) and the dragged state is
    // persisted in QSettings. ----
    m_bodySplitter = new QSplitter(Qt::Horizontal, this);
    m_bodySplitter->setChildrenCollapsible(false);
    m_modeList = new QListWidget(this);
    m_modeList->setMinimumWidth(ecvAICoreUi::dpiScaled(120));
    m_modeList->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_modeList->setStyleSheet(
            QStringLiteral("QListWidget { background: palette(base); border: "
                           "1px solid palette(mid); border-radius: 3px; "
                           "padding: 2px; }"
                           "QListWidget::item { padding: 4px 6px; "
                           "border-radius: 3px; }"
                           "QListWidget::item:selected { background: "
                           "palette(highlight); color: "
                           "palette(highlighted-text); }"));
    m_modeStack = new QStackedWidget(this);
    m_bodySplitter->addWidget(m_modeList);
    m_bodySplitter->addWidget(m_modeStack);
    // Window resizes grow the right-hand panels only; the mode list keeps
    // its (user-adjustable) width.
    m_bodySplitter->setStretchFactor(0, 0);
    m_bodySplitter->setStretchFactor(1, 1);
    rootLayout->addWidget(m_bodySplitter, 1);

    // ---- per-mode panels ----
    for (const QString& mode : GKDHelpers::promptModes()) {
        GKDModePanel panel;
        panel.mode = mode;
        panel.page = new QWidget(this);
        auto* layout = new QVBoxLayout(panel.page);
        ecvAICoreUi::setupTabLayout(layout);

        // Two-column body: config controls on the left, preview on the
        // right, so the dialog stays compact along both axes.
        auto* contentRow = new QHBoxLayout;
        contentRow->setSpacing(ecvAICoreUi::hSpacing());
        auto* configCol = new QVBoxLayout;
        configCol->setSpacing(ecvAICoreUi::vSpacing());

        const bool hasText = mode != QStringLiteral("visual");
        const bool hasSupport = mode == QStringLiteral("visual") ||
                                mode == QStringLiteral("multimodal");
        const bool hasRoi = mode == QStringLiteral("text");
        const bool hasYolo = GKDHelpers::modeUsesYolo(mode);

        // Keypoint texts (text / multimodal / multi modes).
        if (hasText) {
            panel.kpsRow = new QWidget(panel.page);
            auto* row = new QHBoxLayout(panel.kpsRow);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("Keypoint texts:")));
            panel.kpsTexts = new QLineEdit(panel.kpsRow);
            panel.kpsTexts->setPlaceholderText(
                    tr("nose, left eye, right eye, left ear, right ear"));
            row->addWidget(panel.kpsTexts, 1);
            configCol->addWidget(panel.kpsRow);
        }

        // 1-shot support image (visual / multimodal modes).
        if (hasSupport) {
            panel.supportImageRow = new QWidget(panel.page);
            auto* row = new QHBoxLayout(panel.supportImageRow);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("Support image:")));
            panel.supportImagePath = new QLineEdit(panel.supportImageRow);
            row->addWidget(panel.supportImagePath, 1);
            auto* browseSupportBtn = ecvAICoreUi::makeBrowseBtn(
                    tr("Browse…"), panel.supportImageRow);
            connect(browseSupportBtn, &QPushButton::clicked, this,
                    [this, edit = panel.supportImagePath]() {
                        m_supportImagePath = edit;
                        onBrowseSupportImage();
                    });
            row->addWidget(browseSupportBtn);
            configCol->addWidget(panel.supportImageRow);

            panel.supportKpsRow = new QWidget(panel.page);
            auto* kpsRow = new QHBoxLayout(panel.supportKpsRow);
            kpsRow->setContentsMargins(0, 0, 0, 0);
            kpsRow->setSpacing(ecvAICoreUi::hSpacing());
            kpsRow->addWidget(ecvAICoreUi::makeLabel(tr("Support kps:")));
            panel.supportKps = new QLineEdit(panel.supportKpsRow);
            panel.supportKps->setPlaceholderText(
                    tr("x1,y1 x2,y2 … (support-image pixels)"));
            kpsRow->addWidget(panel.supportKps, 1);
            configCol->addWidget(panel.supportKpsRow);
        }

        // ROI window (text mode only; the official bbox demo).
        if (hasRoi) {
            panel.roiRow = new QWidget(panel.page);
            auto* row = new QHBoxLayout(panel.roiRow);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("ROI bbox:")));
            panel.roi = new QLineEdit(panel.roiRow);
            panel.roi->setPlaceholderText(
                    tr("x1 y1 x2 y2 (empty = whole image)"));
            panel.roi->setToolTip(
                    tr("Optional ROI. One box: x1 y1 x2 y2. Multiple boxes "
                       "(4+ coordinates, comma or space separated) run as "
                       "one batched GKD forward and label object1…N."));
            row->addWidget(panel.roi, 1);
            configCol->addWidget(panel.roiRow);
        }

        // Multi-object composition rows (multi mode only): open-vocabulary
        // object classes + the YOLO-World detector that produces the boxes.
        if (hasYolo) {
            panel.classesRow = new QWidget(panel.page);
            auto* row = new QHBoxLayout(panel.classesRow);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("Classes:")));
            panel.objectClasses = new QLineEdit(panel.classesRow);
            panel.objectClasses->setPlaceholderText(tr("cat, dog, human"));
            row->addWidget(panel.objectClasses, 1);
            configCol->addWidget(panel.classesRow);

            panel.detectorRow = new QWidget(panel.page);
            auto* detRow = new QHBoxLayout(panel.detectorRow);
            detRow->setContentsMargins(0, 0, 0, 0);
            detRow->setSpacing(ecvAICoreUi::hSpacing());
            detRow->addWidget(ecvAICoreUi::makeLabel(tr("Detector:")));
            panel.yoloModelCombo = new QComboBox(panel.detectorRow);
            panel.yoloModelCombo->setMinimumContentsLength(16);
            panel.yoloModelCombo->setSizeAdjustPolicy(
                    QComboBox::AdjustToMinimumContentsLengthWithIcon);
            panel.yoloModelCombo->setSizePolicy(QSizePolicy::Expanding,
                                                QSizePolicy::Fixed);
            detRow->addWidget(panel.yoloModelCombo, 1);
            detRow->addWidget(ecvAICoreUi::makeLabel(tr("Conf:")));
            panel.yoloConf = new QDoubleSpinBox(panel.detectorRow);
            panel.yoloConf->setRange(0.05, 0.95);
            panel.yoloConf->setSingleStep(0.05);
            panel.yoloConf->setValue(0.25);
            ecvAICoreUi::setCompactDoubleSpin(panel.yoloConf);
            detRow->addWidget(panel.yoloConf);
            configCol->addWidget(panel.detectorRow);

            // Text-encoder tower: text-conditioned WORLD detectors encode
            // the open-vocabulary class names through a CLIP/MobileCLIP
            // GGUF; without it the run is rejected by the backend
            // (aicore_yolo_options_set_text_model).
            panel.textModelRow = new QWidget(panel.page);
            auto* textRow = new QHBoxLayout(panel.textModelRow);
            textRow->setContentsMargins(0, 0, 0, 0);
            textRow->setSpacing(ecvAICoreUi::hSpacing());
            textRow->addWidget(ecvAICoreUi::makeLabel(tr("Text encoder:")));
            panel.yoloTextModelCombo = new QComboBox(panel.textModelRow);
            panel.yoloTextModelCombo->setMinimumContentsLength(16);
            panel.yoloTextModelCombo->setSizeAdjustPolicy(
                    QComboBox::AdjustToMinimumContentsLengthWithIcon);
            panel.yoloTextModelCombo->setSizePolicy(QSizePolicy::Expanding,
                                                    QSizePolicy::Fixed);
            panel.yoloTextModelCombo->setToolTip(
                    tr("Encodes the class names for the YOLO-World detector "
                       "(same catalog as qYOLO; downloaded on demand)"));
            textRow->addWidget(panel.yoloTextModelCombo, 1);
            configCol->addWidget(panel.textModelRow);
        }

        // Input row: image path + browse (every mode).
        {
            auto* row = new QHBoxLayout;
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("Image (or db://):")));
            panel.imagePath = new QLineEdit(panel.page);
            row->addWidget(panel.imagePath, 1);
            auto* browseBtn =
                    ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.page);
            connect(browseBtn, &QPushButton::clicked, this,
                    [this, edit = panel.imagePath]() {
                        m_imagePath = edit;
                        onBrowseImage();
                    });
            row->addWidget(browseBtn);
            configCol->addLayout(row);
        }

        // Min-score row (per-keypoint display threshold).
        {
            auto* row = new QHBoxLayout;
            row->setSpacing(ecvAICoreUi::hSpacing());
            row->addWidget(ecvAICoreUi::makeLabel(tr("Min score:")));
            panel.minScore = new QDoubleSpinBox(panel.page);
            panel.minScore->setRange(0.0, 1.0);
            panel.minScore->setSingleStep(0.05);
            panel.minScore->setValue(0.10);
            ecvAICoreUi::setCompactDoubleSpin(panel.minScore);
            row->addWidget(panel.minScore);
            row->addStretch();
            configCol->addLayout(row);
        }

        // DB image picker (collapsible, one per panel).
        panel.dbContentWidget = new QWidget(panel.page);
        {
            auto* dbLayout = new QVBoxLayout(panel.dbContentWidget);
            dbLayout->setContentsMargins(0, 0, 0, 0);
            dbLayout->setSpacing(ecvAICoreUi::vSpacing());
            panel.dbImageList = new QListWidget(panel.dbContentWidget);
            panel.dbImageList->setViewMode(QListWidget::IconMode);
            panel.dbImageList->setIconSize(QSize(48, 48));
            panel.dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
            panel.dbImageList->setSelectionMode(
                    QAbstractItemView::SingleSelection);
            dbLayout->addWidget(panel.dbImageList);
            auto* dbBtnRow = new QHBoxLayout;
            auto* refreshDbBtn =
                    new QPushButton(tr("Refresh"), panel.dbContentWidget);
            refreshDbBtn->setToolTip(
                    tr("Reload the ccImage list from the DB tree"));
            connect(refreshDbBtn, &QPushButton::clicked, this,
                    [this]() { emit refreshDbImagesRequested(); });
            dbBtnRow->addWidget(refreshDbBtn);
            dbBtnRow->addStretch();
            dbLayout->addLayout(dbBtnRow);
        }
        panel.dbToggleBtn = ecvAICoreUi::makeDbSection(panel.dbContentWidget);
        ecvAICoreUi::connectDbToggle(panel.dbToggleBtn, panel.dbContentWidget);
        configCol->addWidget(panel.dbToggleBtn, 0, Qt::AlignLeft);
        configCol->addWidget(panel.dbContentWidget);
        configCol->addStretch();

        connect(panel.dbImageList, &QListWidget::itemActivated, this,
                &GKDDialog::onDbListActivated);
        connect(panel.dbImageList, &QListWidget::itemClicked, this,
                &GKDDialog::onDbListActivated);

        contentRow->addLayout(configCol, 1);

        // Right column: preview thumbnail (top-aligned, DPI aware).
        auto* previewCol = new QVBoxLayout;
        previewCol->setSpacing(ecvAICoreUi::vSpacing());
        panel.previewLabel = new ecvClickableImageLabel(panel.page);
        const int ps = ecvAICoreUi::previewSize();
        panel.previewLabel->setFixedSize(ps, ps);
        panel.previewLabel->setStyleSheet(
                "border: 1px solid palette(mid); background: palette(base);");
        panel.previewLabel->setText(tr("Preview"));
        previewCol->addWidget(panel.previewLabel);
        previewCol->addStretch();
        contentRow->addLayout(previewCol);

        layout->addLayout(contentRow);

        // Now that the preview label exists, wire the alias-first preview
        // refresh for this panel's image edits.
        connect(panel.imagePath, &QLineEdit::textChanged, this,
                [this, edit = panel.imagePath, preview = panel.previewLabel]() {
                    m_imagePath = edit;
                    m_previewLabel = preview;
                    updateImagePreview();
                });

        // Action row: sample data + Run / Cancel (per mode).
        auto* actionRow = new QHBoxLayout;
        actionRow->setSpacing(ecvAICoreUi::hSpacing());
        actionRow->addStretch();
        panel.testDataBtn = ecvAICoreUi::makeSampleDataBtn(panel.page);
        QStringList tip = {
                tr("One-click fill of every field of this mode with the "
                   "bundled demo scenarios (cached download); repeated "
                   "clicks cycle through all of them")};
        if (mode == QStringLiteral("text")) {
            tip << tr(
                    "(cat-face 5-point demo, hand X-ray, penguin, chair "
                    "parts, tiger — each click advances)");
        } else if (mode == QStringLiteral("visual")) {
            tip << tr(
                    "(pug / tiger / cat-&-dog lineup / penguin queries "
                    "with the official 1-shot cat-face support keypoints "
                    "— each click advances)");
        } else if (mode == QStringLiteral("multimodal")) {
            tip << tr(
                    "(official cat pair, then pug / cat-&-dog lineup "
                    "queries with the same fused support keypoints — "
                    "each click advances)");
        } else if (mode == QStringLiteral("multi")) {
            tip << tr(
                    "(8 bundled multi-target scenes: alpaca herd, bronze "
                    "statues, pig farm, fish school, cat-&-dog lineup, "
                    "traffic intersection, egocentric hands, bird row — "
                    "each click advances)");
        }
        panel.testDataBtn->setToolTip(tip.join(QChar(' ')));
        connect(panel.testDataBtn, &QPushButton::clicked, this,
                &GKDDialog::requestTestData);
        actionRow->addWidget(panel.testDataBtn);
        panel.runBtn = new QPushButton(tr("Run"), panel.page);
        panel.runBtn->setDefault(mode == QStringLiteral("text"));
        actionRow->addWidget(panel.runBtn);
        panel.cancelBtn = new QPushButton(tr("Cancel"), panel.page);
        panel.cancelBtn->setEnabled(false);
        actionRow->addWidget(panel.cancelBtn);
        layout->addLayout(actionRow);

        connect(panel.runBtn, &QPushButton::clicked, this, &GKDDialog::onRun);
        connect(panel.cancelBtn, &QPushButton::clicked, this,
                &GKDDialog::onCancel);

        m_panels.append(panel);
        m_modeStack->addWidget(panel.page);
    }

    // ---- Mode list entries ------------------------------------------------
    // Grouped navigation (section headers are disabled items): the
    // single-object prompt modes, then the multi-object composition.
    auto addSectionHeader = [this](const QString& title) {
        auto* header = new QListWidgetItem(title, m_modeList);
        header->setFlags(Qt::NoItemFlags);
        QFont bold = header->font();
        bold.setBold(true);
        header->setFont(bold);
        header->setForeground(palette().mid());
        header->setToolTip(title);
    };
    auto addModeItem = [this](const QString& title, int stackIndex) {
        auto* item = new QListWidgetItem(title, m_modeList);
        item->setData(kModeStackIndexRole, stackIndex);
        item->setToolTip(title);
    };
    bool promptHeaderAdded = false;
    bool compositionHeaderAdded = false;
    const QStringList modeTitles = {tr("Text Prompts"), tr("Visual (1-shot)"),
                                    tr("Multimodal"),
                                    tr("Multi-object (YOLO-World)")};
    const QStringList modes = GKDHelpers::promptModes();
    for (int i = 0; i < modes.size(); ++i) {
        if (GKDHelpers::modeUsesYolo(modes[i])) {
            if (!compositionHeaderAdded) {
                addSectionHeader(tr("Composition"));
                compositionHeaderAdded = true;
            }
        } else if (!promptHeaderAdded) {
            addSectionHeader(tr("Prompt modes"));
            promptHeaderAdded = true;
        }
        addModeItem(modeTitles[i], i);
    }
    connect(m_modeList, &QListWidget::currentRowChanged, this, [this](int row) {
        QListWidgetItem* item = m_modeList->item(row);
        if (!item) return;
        const QVariant page = item->data(kModeStackIndexRole);
        if (!page.isValid()) return;
        m_modeStack->setCurrentIndex(page.toInt());
    });
    // Select the first selectable entry (row 0 is a section header).
    for (int row = 0; row < m_modeList->count(); ++row) {
        if (m_modeList->item(row)->data(kModeStackIndexRole).isValid()) {
            m_modeList->setCurrentRow(row);
            break;
        }
    }

    // ---- output group (shared by every mode) ----
    auto* outGroup = new QGroupBox(tr("Output"), this);
    auto* outGrid = new QGridLayout(outGroup);
    ecvAICoreUi::setupFormGrid(outGrid);
    m_addDbCheck = new QCheckBox(tr("Add rendered result to DB"), outGroup);
    m_savePngCheck = new QCheckBox(tr("Save PNG"), outGroup);
    // Per-keypoint "prompt score" labels, default OFF: in multi-object
    // scenes the label backgrounds alone cover the objects. Opting in
    // also gets the de-overlap layout with thin leader arrows.
    m_pointLabelsCheck = new QCheckBox(tr("Show keypoint labels"), outGroup);
    m_savePngDir = new QLineEdit(outGroup);
    auto* browseSaveBtn = ecvAICoreUi::makeBrowseBtn(tr("Browse…"), outGroup);
    connect(browseSaveBtn, &QPushButton::clicked, this,
            &GKDDialog::onBrowseSaveDir);
    outGrid->addWidget(m_addDbCheck, 0, 0, 1, 3);
    outGrid->addWidget(m_savePngCheck, 1, 0);
    outGrid->addWidget(m_savePngDir, 1, 1);
    outGrid->addWidget(browseSaveBtn, 1, 2);
    outGrid->addWidget(m_pointLabelsCheck, 2, 0, 1, 3);
    // Official-parity COCO prediction export (categories with keypoint
    // names + skeleton, images, xywh boxes with detector scores, and
    // x,y,score triplets); enabled once a run succeeds.
    m_exportCocoBtn = new QPushButton(tr("Export COCO JSON…"), outGroup);
    m_exportCocoBtn->setEnabled(false);
    connect(m_exportCocoBtn, &QPushButton::clicked, this,
            &GKDDialog::onExportCocoJson);
    outGrid->addWidget(m_exportCocoBtn, 3, 0, 1, 3);
    rootLayout->addWidget(outGroup);

    // ---- progress + task status + log (shared by every mode) ----
    ecvAICoreUi::setupProgressSection(rootLayout, m_downloadLabel, m_progress);

    m_taskStatusLabel = new QLabel(this);
    m_taskStatusLabel->setVisible(false);
    rootLayout->addWidget(m_taskStatusLabel);

    m_log = new QTextEdit(this);
    m_log->setReadOnly(true);
    m_log->setMaximumHeight(ecvAICoreUi::dpiScaled(120));
    rootLayout->addWidget(m_log);

    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GKDDialog::onModelComboChanged);
    connect(browseModelBtn, &QPushButton::clicked, this,
            &GKDDialog::onBrowseCustomModel);

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
            &GKDDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[GKD] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[GKD] Model downloaded: %1").arg(path));
                if (m_pendingActionAfterDownload == PendingAction::Run) {
                    m_pendingActionAfterDownload = PendingAction::None;
                    onRun();
                }
            });

    // Sample-data repository (shared cache/download/extract state machine;
    // the downloadProgress / logMessage handlers only react while OUR
    // chain drives the repository, matching the qYOLO contract).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataDownloadInProgress) return;
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_progress->setVisible(true);
                m_downloadLabel->setText(statusText);
                m_downloadLabel->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataDownloadInProgress) appendLog(message);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            &GKDDialog::onTestDataDownloadFinished);
    connect(&testDataRepo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                m_progress->setRange(0, total);
                m_progress->setValue(current);
                m_progress->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            &GKDDialog::onTestDataExtractionFinished);
}

void GKDDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void GKDDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qGKD"));
    const QString modelFilename =
            settings.value(QStringLiteral("modelFilename")).toString();
    m_modelExplicit =
            settings.value(QStringLiteral("modelFilenameExplicit"), false)
                    .toBool();
    if (m_modelExplicit) selectModelByFilename(modelFilename);
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());

    // Per-mode prompt state (one settings subkey per mode panel).
    const QStringList modes = GKDHelpers::promptModes();
    for (int i = 0; i < m_panels.size() && i < modes.size(); ++i) {
        GKDModePanel& panel = m_panels[i];
        const QString key = QStringLiteral("mode/") + modes[i];
        if (panel.kpsTexts) {
            panel.kpsTexts->setText(
                    settings.value(key + QStringLiteral("/kpsTexts"))
                            .toString());
        }
        if (panel.supportImagePath) {
            panel.supportImagePath->setText(
                    settings.value(key + QStringLiteral("/supportPath"))
                            .toString());
        }
        if (panel.supportKps) {
            panel.supportKps->setText(
                    settings.value(key + QStringLiteral("/supportKps"))
                            .toString());
        }
        if (panel.roi) {
            panel.roi->setText(
                    settings.value(key + QStringLiteral("/roi")).toString());
        }
        if (panel.objectClasses) {
            panel.objectClasses->setText(
                    settings.value(key + QStringLiteral("/objectClasses"))
                            .toString());
        }
        if (panel.yoloConf) {
            // Defaults were recalibrated (0.5 -> 0.25, 0.30 -> 0.10) after
            // measured score distributions showed the old values hid valid
            // results. A persisted OLD default means "never touched by the
            // user": migrate it so existing installs pick the new default.
            double conf =
                    settings.value(key + QStringLiteral("/yoloConf"), 0.25)
                            .toDouble();
            if (qFuzzyCompare(conf, 0.5)) conf = 0.25;
            panel.yoloConf->setValue(conf);
        }
        double minScore =
                settings.value(key + QStringLiteral("/minScore"), 0.10)
                        .toDouble();
        if (qFuzzyCompare(minScore, 0.30)) minScore = 0.10;
        panel.minScore->setValue(minScore);
        const QString imagePath =
                settings.value(key + QStringLiteral("/imagePath")).toString();
        if (!imagePath.isEmpty()) {
            m_imagePath = panel.imagePath;
            m_previewLabel = panel.previewLabel;
            panel.imagePath->setText(imagePath);
        }
    }
    // Legacy single-page-dialog fallback: migrate the old flat text-mode
    // keys once so an update does not silently clear the user's prompts.
    if (!settings.contains(QStringLiteral("mode/text/kpsTexts"))) {
        GKDModePanel* text = panelForMode(QStringLiteral("text"));
        if (text && text->kpsTexts) {
            const QString legacyKps =
                    settings.value(QStringLiteral("kpsTexts")).toString();
            text->kpsTexts->setText(
                    legacyKps.isEmpty()
                            ? QStringLiteral("nose, left eye, right eye")
                            : legacyKps);
        }
    }
    if (!settings.contains(QStringLiteral("mode/text/imagePath"))) {
        const QString legacyImage =
                settings.value(QStringLiteral("imagePath")).toString();
        GKDModePanel* text = panelForMode(QStringLiteral("text"));
        if (text && !legacyImage.isEmpty()) {
            m_imagePath = text->imagePath;
            m_previewLabel = text->previewLabel;
            text->imagePath->setText(legacyImage);
        }
    }
    const QString yoloModel =
            settings.value(QStringLiteral("yoloModel")).toString();
    const QString yoloTextModel =
            settings.value(QStringLiteral("yoloTextModel")).toString();
    const GKDModePanel* multi = panelForMode(QStringLiteral("multi"));
    if (multi) {
        if (!yoloModel.isEmpty() && multi->yoloModelCombo) {
            // "yolov8s-world-f16" was the pre-recalibration detector
            // default: a persisted match means "never picked by the user",
            // so keep the populate-time default (now yolov8l-world) instead
            // of restoring the stale choice.
            if (yoloModel != QStringLiteral("yolov8s-world-f16.gguf")) {
                const int idx = multi->yoloModelCombo->findData(yoloModel);
                if (idx >= 0) multi->yoloModelCombo->setCurrentIndex(idx);
            }
        }
        if (!yoloTextModel.isEmpty() && multi->yoloTextModelCombo) {
            const int idx = multi->yoloTextModelCombo->findData(yoloTextModel);
            if (idx >= 0) multi->yoloTextModelCombo->setCurrentIndex(idx);
        }
    }
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_savePngCheck->setChecked(
            settings.value(QStringLiteral("savePng"), false).toBool());
    m_pointLabelsCheck->setChecked(
            settings.value(QStringLiteral("pointLabels"), false).toBool());
    m_savePngDir->setText(
            settings.value(QStringLiteral("savePngDir")).toString());
    // Restore the last active mode (empty → the list's default selection).
    const QString lastMode =
            settings.value(QStringLiteral("lastMode")).toString();
    if (!lastMode.isEmpty()) {
        for (int row = 0; row < m_modeList->count(); ++row) {
            QListWidgetItem* item = m_modeList->item(row);
            const QVariant page =
                    item ? item->data(kModeStackIndexRole) : QVariant();
            if (!page.isValid()) continue;
            const int stackIndex = page.toInt();
            if (stackIndex >= 0 && stackIndex < m_panels.size() &&
                m_panels[stackIndex].mode == lastMode) {
                m_modeList->setCurrentRow(row);
                break;
            }
        }
    }
    // Splitter geometry: restore the user's last left/right drag (empty
    // key / state mismatch → fall back to the content-driven default,
    // applied on first show).
    const QByteArray splitterState =
            settings.value(QStringLiteral("bodySplitterState")).toByteArray();
    if (!splitterState.isEmpty() && m_bodySplitter &&
        m_bodySplitter->restoreState(splitterState)) {
        m_splitterRestored = true;
    }
    settings.endGroup();
}

void GKDDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qGKD"));
    settings.setValue(QStringLiteral("modelFilename"),
                      m_modelCombo->currentData().toString());
    settings.setValue(QStringLiteral("modelFilenameExplicit"), m_modelExplicit);
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    const QStringList modes = GKDHelpers::promptModes();
    for (int i = 0; i < m_panels.size() && i < modes.size(); ++i) {
        const GKDModePanel& panel = m_panels[i];
        const QString key = QStringLiteral("mode/") + modes[i];
        if (panel.kpsTexts) {
            settings.setValue(key + QStringLiteral("/kpsTexts"),
                              panel.kpsTexts->text());
        }
        if (panel.supportImagePath) {
            settings.setValue(key + QStringLiteral("/supportPath"),
                              panel.supportImagePath->text());
        }
        if (panel.supportKps) {
            settings.setValue(key + QStringLiteral("/supportKps"),
                              panel.supportKps->text());
        }
        if (panel.roi) {
            settings.setValue(key + QStringLiteral("/roi"), panel.roi->text());
        }
        if (panel.objectClasses) {
            settings.setValue(key + QStringLiteral("/objectClasses"),
                              panel.objectClasses->text());
        }
        if (panel.yoloConf) {
            settings.setValue(key + QStringLiteral("/yoloConf"),
                              panel.yoloConf->value());
        }
        settings.setValue(key + QStringLiteral("/minScore"),
                          panel.minScore->value());
        settings.setValue(key + QStringLiteral("/imagePath"),
                          panel.imagePath->text());
    }
    const GKDModePanel* multi = panelForMode(QStringLiteral("multi"));
    settings.setValue(QStringLiteral("yoloModel"),
                      multi && multi->yoloModelCombo
                              ? multi->yoloModelCombo->currentData().toString()
                              : QString());
    settings.setValue(
            QStringLiteral("yoloTextModel"),
            multi && multi->yoloTextModelCombo
                    ? multi->yoloTextModelCombo->currentData().toString()
                    : QString());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("savePng"), m_savePngCheck->isChecked());
    settings.setValue(QStringLiteral("pointLabels"),
                      m_pointLabelsCheck->isChecked());
    settings.setValue(QStringLiteral("savePngDir"), m_savePngDir->text());
    const GKDModePanel* active = currentModePanel();
    settings.setValue(QStringLiteral("lastMode"),
                      active ? active->mode : QString());
    // Splitter geometry: persist the user's last left/right drag.
    if (m_bodySplitter) {
        settings.setValue(QStringLiteral("bodySplitterState"),
                          m_bodySplitter->saveState());
    }
    settings.endGroup();
}

QString GKDDialog::modelCacheDir() { return GKDHelpers::modelCacheDir(); }

void GKDDialog::populateModelCombo(const QString& keepFilename) {
    const QVector<GKDModelEntry> models = GKDHelpers::catalogModels();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const GKDModelEntry& e : models) {
        m_modelCombo->addItem(GKDHelpers::modelDisplayLabel(e), e.filename);
    }
    ecvAICoreUi::selectModelRow(m_modelCombo, keepFilename,
                                GKDHelpers::catalogDefaultIndex());
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());
}

void GKDDialog::populateYoloModelCombo(const QString& keepFilename) {
    // Multi-object detector + its text tower come from the existing yolo
    // task catalog (no second model table): the WORLD role lists the
    // open-vocabulary detectors, the TEXT role the CLIP/MobileCLIP towers
    // that encode the class names.
    GKDModePanel* multi = panelForMode(QStringLiteral("multi"));
    if (!multi || !multi->yoloModelCombo) return;
    multi->yoloModelCombo->clear();
    for (const GKDModelEntry& e : GKDHelpers::yoloWorldModels()) {
        multi->yoloModelCombo->addItem(GKDHelpers::modelDisplayLabel(e),
                                       e.filename);
    }
    ecvAICoreUi::selectModelRow(multi->yoloModelCombo, keepFilename,
                                GKDHelpers::yoloWorldDefaultIndex());
    if (!multi->yoloTextModelCombo) return;
    // Keep the current text-tower choice across catalog refreshes (empty
    // on the first populate → the catalog-declared default).
    const QString keepText =
            multi->yoloTextModelCombo->currentData().toString();
    multi->yoloTextModelCombo->clear();
    for (const GKDModelEntry& e : GKDHelpers::yoloTextModels()) {
        multi->yoloTextModelCombo->addItem(GKDHelpers::modelDisplayLabel(e),
                                           e.filename);
    }
    ecvAICoreUi::selectModelRow(multi->yoloTextModelCombo, keepText,
                                GKDHelpers::yoloTextDefaultIndex());
}

bool GKDDialog::selectModelByFilename(const QString& filename) {
    if (filename.isEmpty()) return false;
    const int idx = m_modelCombo->findData(filename);
    if (idx < 0) return false;
    m_modelCombo->setCurrentIndex(idx);
    return true;
}

void GKDDialog::refreshModelList() {
    const GKDModePanel* multi = panelForMode(QStringLiteral("multi"));
    populateYoloModelCombo(
            multi && multi->yoloModelCombo
                    ? multi->yoloModelCombo->currentData().toString()
                    : QString());
    populateModelCombo(m_modelCombo->currentData().toString());
}

void GKDDialog::onModelComboChanged(int index) {
    if (sender() == m_modelCombo) m_modelExplicit = true;
    const QString filename = m_modelCombo->itemData(index).toString();
    const bool isCustom = filename.isEmpty() ||
                          (filename.endsWith(QStringLiteral(".gguf")) &&
                           !GKDHelpers::findModelByFilename(filename, nullptr));
    m_customModelRow->setVisible(isCustom);
}

QString GKDDialog::resolveModelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = GKDHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

QString GKDDialog::resolveYoloModelPath(const QComboBox* combo) const {
    const QString filename =
            combo ? combo->currentData().toString() : QString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    char* raw =
#ifdef AICore_ENABLED
            aicore_yolo_model_cache_dir();
#else
            nullptr;
#endif
    if (!raw) return QString();
    const QString dir = QString::fromUtf8(raw);
    aicore_yolo_free_buffer(raw);
    return dir + QDir::separator() + filename;
}

bool GKDDialog::ensureModelAvailable(PendingAction action) {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[GKD] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath())) {
        GKDModelEntry entry;
        if (!GKDHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[GKD] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[GKD] Model missing — downloading %1; it will start "
                     "automatically when ready.")
                          .arg(filename));
        startDownload(entry);
        return false;
    }
    return true;
}

void GKDDialog::startDownload(const GKDModelEntry& model,
                              const QString& destDir) {
    if (m_downloadInProgress) {
        appendLog(tr("[GKD] A download is already running."));
        return;
    }
    const QString dir =
            destDir.isEmpty() ? GKDHelpers::modelCacheDir() : destDir;
    QDir().mkpath(dir);
    const QString dest = dir + QDir::separator() + model.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[GKD] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[GKD] Downloading %1 (%2)…")
                      .arg(model.filename, model.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // GKDT GGUFs are hundreds of MB
    // Content identity from the pinned digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught). Catalog
    // rows without a published baseline (later yolo additions) fall back to
    // the size/magic checks — same policy as qYOLO.
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(model.filename)};
    m_downloader->download(req);
}

bool GKDDialog::ensureYoloModelAvailable() {
    GKDModePanel* panel = currentModePanel();
    if (!panel || !GKDHelpers::modeUsesYolo(panel->mode)) return true;
    // The detector AND its text tower are both required: text-conditioned
    // WORLD models reject the run without the encoder.
    const QComboBox* required[] = {panel->yoloModelCombo,
                                   panel->yoloTextModelCombo};
    for (const QComboBox* combo : required) {
        if (!combo) continue;
        const QString filename = combo->currentData().toString();
        if (filename.isEmpty()) {
            appendLog(
                    tr("[GKD] Multi-object mode needs a YOLO-World "
                       "detector and a text-encoder GGUF."));
            return false;
        }
        if (QFileInfo::exists(resolveYoloModelPath(combo))) continue;
#ifdef AICore_ENABLED
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_by_filename(filename.toUtf8().constData());
        if (e == nullptr) {
            appendLog(tr("[GKD] Model not found in the yolo catalog: %1")
                              .arg(filename));
            return false;
        }
        GKDModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        m_pendingActionAfterDownload = PendingAction::Run;
        appendLog(tr("[GKD] %1 missing — downloading %2; the run starts "
                     "automatically when ready.")
                          .arg(combo == panel->yoloModelCombo
                                       ? tr("Detector model")
                                       : tr("Text encoder"),
                               filename));
        char* raw = aicore_yolo_model_cache_dir();
        const QString destDir =
                raw ? QString::fromUtf8(raw) : GKDHelpers::modelCacheDir();
        aicore_yolo_free_buffer(raw);
        startDownload(entry, destDir);
        return false;
#else
        appendLog(tr("[GKD] Model file not found: %1").arg(filename));
        return false;
#endif
    }
    return true;
}

void GKDDialog::cancelDownload() {
    if (m_downloadInProgress) m_downloader->cancel();
}

void GKDDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qGKD"),
                                             QStringLiteral("lastModelDir"),
                                             GKDHelpers::modelCacheDir());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select GKD GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    m_customModelPath->setText(path);
    m_customModelRow->setVisible(true);
    m_modelCombo->setCurrentIndex(-1);
    m_modelCombo->addItem(QFileInfo(path).fileName(), path);
    m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
}

void GKDDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qGKD"),
                                             QStringLiteral("lastImageFileDir"),
                                             QDir::homePath());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select query image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qGKD"),
                         QStringLiteral("lastImageFileDir"), path);
}

void GKDDialog::onBrowseSupportImage() {
    if (!m_supportImagePath) return;
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qGKD"),
                                             QStringLiteral("lastSupportDir"),
                                             QDir::homePath());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select 1-shot support image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_supportImagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qGKD"),
                         QStringLiteral("lastSupportDir"), path);
}

void GKDDialog::onBrowseSaveDir() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qGKD"),
                             QStringLiteral("lastSaveDir"), QDir::homePath());
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select output directory"), lastDir);
    if (dir.isEmpty()) return;
    m_savePngDir->setText(dir);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qGKD"),
                         QStringLiteral("lastSaveDir"), dir);
}

void GKDDialog::updateImagePreview() {
    if (!m_imagePath || !m_previewLabel) return;
    QImage img;
    const QString path = m_imagePath->text().trimmed();
    if (path.startsWith(QStringLiteral("db://"))) {
        // DB-tree entity: look up the stored full-resolution image so the
        // click-to-enlarge preview works for DB inputs too.
        const QString name = path.mid(5);
        if (GKDModePanel* panel = currentModePanel()) {
            for (int i = 0; i < panel->dbImageList->count(); ++i) {
                QListWidgetItem* item = panel->dbImageList->item(i);
                if (item && item->data(Qt::UserRole).toString() == name) {
                    img = item->data(kDbFullImageRole).value<QImage>();
                    break;
                }
            }
        }
    } else {
        img = QImage(path);
    }
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        return;
    }
    m_previewLabel->setPreviewImage(img, ecvAICoreUi::previewSize());
}

int GKDDialog::modeListIdealWidth() const {
    if (!m_modeList) return 0;
    // Content-driven, font/DPI aware: measure the widest entry text
    // (section headers render bold) instead of hard-coding pixels, so the
    // default fits every translation, platform style and screen
    // resolution without eliding entries.
    const QFont base = m_modeList->font();
    const QFontMetrics fm(base);
    QFont boldFont = base;
    boldFont.setBold(true);
    const QFontMetrics fmBold(boldFont);
    int textWidth = 0;
    for (int i = 0; i < m_modeList->count(); ++i) {
        const QListWidgetItem* item = m_modeList->item(i);
        if (!item) continue;
        const bool header = !(item->flags() & Qt::ItemIsEnabled);
        const QFontMetrics& itemFm = header ? fmBold : fm;
        textWidth = qMax(textWidth, itemFm.horizontalAdvance(item->text()));
    }
    // Room for the QSS item padding (4px 6px), list frame + padding, the
    // vertical scrollbar small windows may show, and platform style
    // margins (e.g. macOS focus rings).
    return textWidth + ecvAICoreUi::dpiScaled(34);
}

void GKDDialog::applyModeListDefaultWidth() {
    if (!m_bodySplitter || !m_modeList || m_modeList->count() == 0) return;
    const int ideal = modeListIdealWidth();
    const int total = m_bodySplitter->width();
    if (total <= 0) return;
    // Left pane = content width, right pane = what remains (the splitter
    // clamps both to the panes' minimum size hints on tiny screens).
    m_bodySplitter->setSizes({ideal, qMax(total - ideal, 1)});
}

GKDModePanel* GKDDialog::currentModePanel() const {
    if (!m_modeStack) return nullptr;
    QWidget* page = m_modeStack->currentWidget();
    for (const GKDModePanel& panel : m_panels) {
        if (panel.page == page) {
            // const_cast: callers expect a mutable panel (they set controls).
            return const_cast<GKDModePanel*>(&panel);
        }
    }
    return nullptr;
}

GKDModePanel* GKDDialog::panelForMode(const QString& mode) const {
    for (const GKDModePanel& panel : m_panels) {
        if (panel.mode == mode) {
            return const_cast<GKDModePanel*>(&panel);
        }
    }
    return nullptr;
}

void GKDDialog::onRun() {
    GKDModePanel* panel = currentModePanel();
    if (!panel) return;
    // Validate the input BEFORE the model chain: a missing image would
    // otherwise trigger the model download first and then fail with a
    // confusing "Input file not found: <empty>" after the wait.
    if (panel->imagePath->text().trimmed().isEmpty()) {
        appendLog(
                tr("[Error] Select an image first — pick a file, a DB "
                   "image, or click Use test data."));
        return;
    }
    if (!ensureModelAvailable(PendingAction::Run)) return;
    if (!ensureYoloModelAvailable()) return;
    emit runRequested(workerSettings());
}

void GKDDialog::onCancel() {
    cancelDownload();
    if (m_testDataDownloadInProgress) {
        ecvTestDataRepository::instance().cancelDownload();
    }
    emit cancelRequested();
}

void GKDDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    GKDModePanel* panel = currentModePanel();
    if (!panel) return;
    panel->imagePath->setText(QStringLiteral("db://") +
                              item->data(Qt::UserRole).toString());
}

void GKDDialog::appendLog(const QString& msg) {
#ifdef AICore_ENABLED
    aicore_inference_log::log(msg);
#endif
    m_log->append(msg);
}

void GKDDialog::setProgress(int current, int total) {
    if (total > 0) {
        m_progress->setRange(0, total);
        m_progress->setValue(current);
        m_progress->setVisible(true);
    } else {
        m_progress->setVisible(false);
    }
}

void GKDDialog::setTaskStage(const QString& stage, int percent) {
    m_taskStatusLabel->setVisible(!stage.isEmpty());
    m_taskStatusLabel->setText(
            percent >= 0 ? QStringLiteral("%1 (%2%)").arg(stage).arg(percent)
                         : stage);
}

void GKDDialog::setRunning(bool running) {
    m_taskRunning = running;
    for (const GKDModePanel& panel : m_panels) {
        panel.runBtn->setEnabled(!running);
        panel.cancelBtn->setEnabled(running);
    }
}

void GKDDialog::setDbImages(const QList<GKDImageEntry>& images) {
    for (const GKDModePanel& panel : m_panels) {
        panel.dbImageList->clear();
        for (const GKDImageEntry& entry : images) {
            auto* item = new QListWidgetItem(
                    entry.preview.isNull() ? QIcon()
                                           : QPixmap::fromImage(entry.preview),
                    entry.name, panel.dbImageList);
            item->setData(Qt::UserRole, entry.name);
            // Full-resolution image for the click-to-enlarge preview (the
            // icon above is only a scaled thumbnail).
            item->setData(kDbFullImageRole, entry.preview);
        }
    }
}

void GKDDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    GKDModePanel* panel = currentModePanel();
    if (!panel) return;
    panel->imagePath->setText(QStringLiteral("db://") + imageNames.first());
}

GKDWorker::Settings GKDDialog::workerSettings() const {
    GKDWorker::Settings s;
    GKDModePanel* panel = const_cast<GKDDialog*>(this)->currentModePanel();
    if (!panel) return s;
    s.modelPath = resolveModelPath();
    s.inputPath = panel->imagePath->text().trimmed();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    if (panel->kpsTexts) {
        s.kpsTexts = GKDHelpers::splitPrompts(panel->kpsTexts->text());
    }
    s.skeleton = panel->skeleton;
    if (panel->supportImagePath) {
        s.supportImagePath = panel->supportImagePath->text().trimmed();
    }
    if (panel->supportKps) {
        QVector<QPointF> supportKps;
        GKDHelpers::parseCoordinatePairs(panel->supportKps->text(),
                                         &supportKps);
        s.supportKps = supportKps;
    }
    if (panel->roi) {
        s.hasBbox = false;
        QVector<QPointF> roi;
        if (GKDHelpers::parseCoordinatePairs(
                    panel->roi->text().replace(QLatin1Char(','),
                                               QLatin1Char(' ')),
                    &roi) &&
            roi.size() == 2) {
            s.hasBbox = true;
            s.bbox[0] = static_cast<float>(roi[0].x());
            s.bbox[1] = static_cast<float>(roi[0].y());
            s.bbox[2] = static_cast<float>(roi[1].x());
            s.bbox[3] = static_cast<float>(roi[1].y());
        }
        // Multi-ROI batch (official --bbox_on_input_im semantics): 4+
        // points = 2+ boxes go through one batched forward.
        s.roiPoints = roi;
    }
    s.minScore = static_cast<float>(panel->minScore->value());
    s.pointLabels = m_pointLabelsCheck->isChecked();
    s.multiObject = GKDHelpers::modeUsesYolo(panel->mode);
    if (s.multiObject) {
        s.objectClasses = panel->objectClasses->text().split(
                QLatin1Char(','), QtCompat::SkipEmptyParts);
        for (QString& c : s.objectClasses) c = c.trimmed();
        s.yoloModelPath = resolveYoloModelPath(panel->yoloModelCombo);
        s.yoloTextModelPath = resolveYoloModelPath(panel->yoloTextModelCombo);
        s.yoloConf = static_cast<float>(panel->yoloConf->value());
    }
    s.addResultToDb = m_addDbCheck->isChecked();
    if (m_savePngCheck->isChecked())
        s.savePngDir = m_savePngDir->text().trimmed();
    return s;
}

void GKDDialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    if (m_taskRunning) {
        const auto answer = QMessageBox::question(
                this, tr("Task running"),
                tr("A GKD task is running. Close anyway?"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        emit cancelRequested();
    }
    QDialog::closeEvent(event);
}

void GKDDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    // First show: the splitter only knows its final size after layout, so
    // give the mode list its content-driven width here. A splitter state
    // restored from QSettings (the user's last drag) wins over the
    // default. The DB image list is refreshed once for every panel.
    if (!m_splitterSized) {
        m_splitterSized = true;
        if (!m_splitterRestored) applyModeListDefaultWidth();
    }
    if (m_firstShow) {
        m_firstShow = false;
        emit refreshDbImagesRequested();
    }
}

// ---- Sample data (shared ecvTestDataRepository; one dataset per mode) ----

void GKDDialog::requestTestData() {
    GKDModePanel* panel = currentModePanel();
    if (!panel) return;
    if (m_downloadInProgress) {
        appendLog(tr("[Test data] Wait for model download to finish first."));
        return;
    }
    if (m_taskRunning) {
        appendLog(tr("[Test data] Wait for the current task to finish."));
        return;
    }
    // Capture the requesting panel's mode: the async path must fill THIS
    // panel even if the user switches modes meanwhile.
    const QString mode = panel->mode;
    if (loadTestDataFor(mode)) return;

    auto& repo = ecvTestDataRepository::instance();
    if (m_testDataDownloadInProgress || repo.isDownloadInProgress()) {
        // The shared repository serves one download at a time (single
        // downloader slot, shared by every plugin). Queue this request
        // instead of dropping it: the repository's downloadFinished /
        // extractionFinished broadcasts resume the queued slots.
        if (m_pendingTestDataMode.isEmpty()) {
            m_pendingTestDataMode = mode;
        } else {
            m_followupTestDataMode = mode;
        }
        appendLog(
                tr("[Test data] Queued — will load when the current "
                   "test-data download finishes."));
        return;
    }

    // No chain is running: this request drives it.
    m_pendingTestDataMode = mode;
    advancePendingTestData();
}

void GKDDialog::advancePendingTestData() {
    if (m_pendingTestDataMode.isEmpty()) return;
    if (loadTestDataFor(m_pendingTestDataMode)) {
        // Clear the slot before serving: servePendingTestData() loads the
        // pending mode again otherwise (double load of the same file).
        m_pendingTestDataMode.clear();
        servePendingTestData();  // only a queued follow-up remains to serve
        return;
    }

    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        // Another request is being served right now; keep waiting — the
        // repository's finished signals resume the queued slots.
        appendLog(
                tr("[Test data] Queued — will load when the current "
                   "test-data download finishes."));
        return;
    }

    const auto kind = datasetForMode(m_pendingTestDataMode);
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    setTestDataControlsEnabled(false);
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        appendLog(tr("[Test data] Extracting cached archive..."));
        m_progress->setRange(0, 0);  // indeterminate / busy
        m_progress->setValue(0);
        m_progress->setVisible(true);
        m_downloadLabel->setText(tr("Extracting test data..."));
        m_downloadLabel->setVisible(true);
        repo.extractDataset(kind);
        return;
    }
    appendLog(tr("[Test data] Downloading the mode's official demo data..."));
    m_downloadLabel->setText(tr("Downloading test data..."));
    m_downloadLabel->setVisible(true);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(true);
    repo.startDownload(kind);
}

void GKDDialog::servePendingTestData() {
    if (!m_pendingTestDataMode.isEmpty()) {
        if (!loadTestDataFor(m_pendingTestDataMode)) {
            appendLog(
                    tr("[Test data] Requested file was not found in the "
                       "archive."));
        }
        m_pendingTestDataMode.clear();
    }
    if (!m_followupTestDataMode.isEmpty()) {
        if (!loadTestDataFor(m_followupTestDataMode)) {
            appendLog(
                    tr("[Test data] Requested file was not found in the "
                       "archive."));
        }
        m_followupTestDataMode.clear();
    }
}

bool GKDDialog::loadTestDataFor(const QString& mode) {
    GKDModePanel* panel = panelForMode(mode);
    if (!panel) return false;
    // "Try sample data" cycles through the mode's bundled scenarios on
    // repeated clicks (the multi-object mode bundles the whole dataset's
    // multi-target scenes).
    const QVector<GKDModePreset> presets = GKDHelpers::modePresets(mode);
    if (presets.isEmpty()) return false;
    const int rotation = m_presetRotation.value(mode, 0);
    m_presetRotation.insert(mode, (rotation + 1) % presets.size());
    const GKDModePreset preset = presets.at(rotation % presets.size());
    if (preset.queryImage.isEmpty()) return false;
    const auto kind = datasetForMode(mode);
    const QString queryPath =
            ecvTestDataRepository::findDatasetFile(kind, preset.queryImage);
    if (queryPath.isEmpty()) return false;

    // Same-bundle support image (1-shot demos): when it is missing the
    // extraction is partial — report "not cached" so the chain re-fetches.
    QString supportPath;
    if (!preset.supportImage.isEmpty()) {
        supportPath = ecvTestDataRepository::findDatasetFile(
                kind, preset.supportImage);
        if (supportPath.isEmpty()) return false;
    }

    // Alias the shared handlers to the target panel FIRST: the setText
    // calls below fire textChanged synchronously.
    m_imagePath = panel->imagePath;
    m_previewLabel = panel->previewLabel;

    QImage img(queryPath);
    if (img.isNull()) {
        appendLog(tr("[Test data] Failed to decode sample image: %1")
                          .arg(queryPath));
        return true;  // cached file exists but is unusable; don't re-download
    }
    panel->imagePath->setText(queryPath);
    if (panel->kpsTexts) panel->kpsTexts->setText(preset.kpsTexts);
    if (panel->supportImagePath) {
        panel->supportImagePath->setText(supportPath);
    }
    if (panel->supportKps) panel->supportKps->setText(preset.supportKps);
    if (panel->roi) panel->roi->clear();
    if (panel->objectClasses) {
        panel->objectClasses->setText(preset.objectClasses);
    }
    if (panel->yoloConf && preset.yoloConf > 0.0f) {
        // Scene-specific recall cut (dense/small-target scenes measure
        // better recall at lower cuts); other scenes leave the spin at
        // its persisted/global value.
        panel->yoloConf->setValue(preset.yoloConf);
    }
    panel->skeleton = preset.skeleton;
    updateImagePreview();
    appendLog(tr("[Test data] Loaded official demo image '%1' — every field "
                 "of this mode was auto-filled.")
                      .arg(QFileInfo(queryPath).fileName()));
    return true;
}

void GKDDialog::setLastRun(const GKDRunResult& result) {
    m_lastRun = result;
    if (m_exportCocoBtn) m_exportCocoBtn->setEnabled(true);
}

void GKDDialog::onExportCocoJson() {
    if (m_lastRun.sets.isEmpty()) return;
    const QString path = QFileDialog::getSaveFileName(
            this, tr("Export COCO predictions"),
            QStringLiteral("gkd_result.json"), tr("COCO JSON (*.json)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        appendLog(tr("[GKD] Failed to write COCO JSON: %1").arg(path));
        return;
    }
    file.write(GKDHelpers::buildCocoJson(m_lastRun).toUtf8());
    file.close();
    appendLog(tr("[GKD] COCO predictions written: %1").arg(path));
}

void GKDDialog::setTestDataControlsEnabled(bool enabled) {
    for (const GKDModePanel& panel : m_panels) {
        if (panel.testDataBtn) panel.testDataBtn->setEnabled(enabled);
    }
}

void GKDDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    const bool ours = m_testDataDownloadInProgress;
    const bool haveQueued = !m_pendingTestDataMode.isEmpty() ||
                            !m_followupTestDataMode.isEmpty();
    if (!ours && !haveQueued) return;  // a broadcast we did not ask for

    if (!m_pendingTestDataMode.isEmpty() &&
        kind != datasetForMode(m_pendingTestDataMode)) {
        // A foreign download of ANOTHER dataset finished: the repository
        // slot is free now (the repo clears its busy flag before this
        // signal) — resume our queued chain, which was waiting for the
        // slot, not for this dataset. Deferred so the repo's slot state
        // is settled regardless of slot-invocation order.
        if (!ours) {
            QTimer::singleShot(0, this, [this]() { advancePendingTestData(); });
        }
        return;
    }

    if (!success) {
        m_testDataDownloadInProgress = false;
        m_downloadLabel->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setVisible(false);
        setTestDataControlsEnabled(true);
        appendLog(tr("[Test data] Download failed."));
        // Drop the failed request; a queued follow-up retries once from
        // our side (the repo slot is free now — bounded retry).
        m_pendingTestDataMode.clear();
        if (!m_followupTestDataMode.isEmpty()) {
            m_pendingTestDataMode = m_followupTestDataMode;
            m_followupTestDataMode.clear();
            advancePendingTestData();
        }
        return;
    }

    if (ours) {
        appendLog(tr("[Test data] Extracting..."));
        m_downloadLabel->setText(tr("Extracting test data..."));
        m_progress->setRange(0, 0);  // indeterminate / busy
        m_progress->setVisible(true);
        ecvTestDataRepository::instance().extractDataset(kind);
        return;
    }
    // Queued on a foreign chain for OUR dataset: its starter extracts next
    // and the extractionFinished broadcast serves the queued slots — do
    // not extract the same archive twice on the GUI thread.
    appendLog(
            tr("[Test data] Download finished — loading after "
               "extraction..."));
}

void GKDDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    const bool ours = m_testDataDownloadInProgress;
    const bool haveQueued = !m_pendingTestDataMode.isEmpty() ||
                            !m_followupTestDataMode.isEmpty();
    if (!ours && !haveQueued) return;  // a broadcast we did not ask for
    if (!m_pendingTestDataMode.isEmpty() &&
        kind != datasetForMode(m_pendingTestDataMode)) {
        return;  // a foreign dataset we are not queued on
    }

    m_testDataDownloadInProgress = false;
    m_downloadLabel->setVisible(false);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    setTestDataControlsEnabled(true);

    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        // Same bounded-retry contract as the download-failure path: the
        // queued follow-up takes over as the pending request.
        m_pendingTestDataMode.clear();
        if (!m_followupTestDataMode.isEmpty()) {
            m_pendingTestDataMode = m_followupTestDataMode;
            m_followupTestDataMode.clear();
            advancePendingTestData();
        }
        return;
    }
    servePendingTestData();
}
