// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OpenPCDetDialog.h"

#include <cvFileDialog.h>
#include <ecvPluginDbNaming.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QHash>
#include <QHeaderView>
#include <QKeyEvent>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QScrollArea>
#include <QSettings>
#include <QShowEvent>
#include <QSizePolicy>
#include <QSplitter>
#include <QStackedWidget>
#include <QTextStream>
#include <QVBoxLayout>
#include <vector>

#include "QtCompat.h"
#include "ecvAICoreUiHelper.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#endif

namespace {
// QListWidgetItem data role carrying the QStackedWidget page index behind
// a task-list entry (section headers are disabled and carry no data).
constexpr int kTaskStackIndexRole = Qt::UserRole + 10;

// Task-list grouping of the 18 checkpoints: KITTI / nuScenes point-cloud
// checkpoints first (dataset grouping), then the multi-modal scene
// checkpoints. Order inside each group follows the AICore catalog.
enum CheckpointGroup { GroupKitti = 0, GroupNuScenes = 1, GroupScene = 2 };

CheckpointGroup groupForCheckpoint(
        const OpenPCDetHelpers::OpenPCDetCheckpoint& cp) {
    if (OpenPCDetHelpers::inputKindForFilename(cp.stem +
                                               QStringLiteral(".f16.gguf")) ==
        OpenPCDetHelpers::InputKind::Scene) {
        return GroupScene;
    }
    // Dataset grouping follows the AICore family note ("... KITTI (...)"
    // / "... nuScenes (...)") — the stem prefix alone is ambiguous
    // ("second_7862" is a KITTI checkpoint without the kitti_ prefix).
    return cp.familyNote.contains(QStringLiteral("KITTI")) ? GroupKitti
                                                           : GroupNuScenes;
}

// Human label of the panel: the checkpoint's display name (quant note
// stripped) minus the redundant "OpenPCDet " prefix keeps the task list
// compact.
QString panelLabel(const OpenPCDetHelpers::OpenPCDetCheckpoint& cp) {
    QString label = cp.displayName;
    if (label.startsWith(QStringLiteral("OpenPCDet "))) {
        label.remove(0, QStringLiteral("OpenPCDet ").size());
    }
    return label;
}
}  // namespace

QString OpenPCDetTaskPanel::modelFilename() const {
    if (quantCombo == nullptr) return QString();
    const QString filename = quantCombo->currentData().toString();
    // A quant entry is "stem.f32.gguf" style data; anything else (e.g. a
    // custom-path placeholder) is not a catalog quantization.
    return filename.endsWith(QStringLiteral(".gguf")) ? filename : QString();
}

QString OpenPCDetTaskPanel::modelPath() const {
    // Custom GGUF override wins over the catalog selection.
    if (customModelPath) {
        const QString custom = customModelPath->text().trimmed();
        if (!custom.isEmpty()) return custom;
    }
    const QString filename = modelFilename();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = OpenPCDetHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

OpenPCDetDialog::OpenPCDetDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("OpenPCDet 3D Object Detection"));
    setupUi();
    populatePanels();
    loadSettings();
    // The configuration body scrolls; do not DPI-scale Qt's already scaled
    // hint.
    resize(ecvAICoreUi::dpiScaled(960), ecvAICoreUi::dpiScaled(600));
}

OpenPCDetDialog::~OpenPCDetDialog() { saveSettings(); }

void OpenPCDetDialog::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(rootLayout);

    // Global runtime row: device / threads render once above the task
    // list — they configure every checkpoint panel alike.
    auto* runtimeRow = new QHBoxLayout;
    runtimeRow->setSpacing(ecvAICoreUi::hSpacing());
    runtimeRow->addWidget(ecvAICoreUi::makeLabel(tr("Device:")));
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_deviceCombo->setMaximumWidth(ecvAICoreUi::dpiScaled(220));
#ifdef AICore_ENABLED
    const int nDev = aicore_device_count();
    for (int i = 0; i < nDev; ++i) {
        const aicore_device_info* dev = aicore_device_at(i);
        if (!dev || !dev->id) continue;
        m_deviceCombo->addItem(QString::fromUtf8(dev->label),
                               QString::fromUtf8(dev->id));
        if (dev->is_default) m_deviceCombo->setCurrentIndex(i);
    }
#endif
    runtimeRow->addWidget(m_deviceCombo);
    runtimeRow->addWidget(ecvAICoreUi::makeLabel(tr("Threads:")));
    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 64);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = auto"));
    runtimeRow->addWidget(m_threads);
    runtimeRow->addStretch();
    rootLayout->addLayout(runtimeRow);

    // Body: grouped checkpoint list on the left, one panel per checkpoint
    // on the right — the qYOLO list/stack navigation keeps all 18
    // checkpoints visible and reachable without a tab-bar overflow. A
    // splitter (not a fixed-width list) lets users drag the pane wider;
    // the default split is content-driven, see applyTaskListDefaultWidth().
    m_bodySplitter = new QSplitter(Qt::Horizontal, this);
    m_bodySplitter->setChildrenCollapsible(false);
    m_taskList = new QListWidget(this);
    m_taskList->setMinimumWidth(ecvAICoreUi::dpiScaled(120));
    m_taskList->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_taskList->setStyleSheet(
            QStringLiteral("QListWidget { background: palette(base); border: "
                           "1px solid palette(mid); border-radius: 3px; "
                           "padding: 2px; }"
                           "QListWidget::item { padding: 4px 6px; "
                           "border-radius: 3px; }"
                           "QListWidget::item:selected { background: "
                           "palette(highlight); color: "
                           "palette(highlighted-text); }"));

    m_taskStack = new QStackedWidget(this);
    m_bodySplitter->addWidget(m_taskList);
    m_bodySplitter->addWidget(m_taskStack);
    // Window resizes grow the right-hand panels only; the task list keeps
    // its (user-adjustable) width.
    m_bodySplitter->setStretchFactor(0, 0);
    m_bodySplitter->setStretchFactor(1, 1);
    rootLayout->addWidget(m_bodySplitter, 1);

    // ---- Shared detection result table ------------------------------------
    m_resultTable = new QTableWidget(this);
    m_resultTable->setColumnCount(9);
    m_resultTable->setHorizontalHeaderLabels(
            {tr("Class"), tr("Score"), tr("X"), tr("Y"), tr("Z"), tr("Dx"),
             tr("Dy"), tr("Dz"), tr("Heading")});
    m_resultTable->verticalHeader()->setVisible(false);
    m_resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_resultTable->setAlternatingRowColors(true);
    m_resultTable->setMaximumHeight(ecvAICoreUi::dpiScaled(140));
    m_resultTable->setToolTip(
            tr("Detections of the last run (KITTI LiDAR frame): class, "
               "confidence, box center (x,y,z), size (dx,dy,dz) and yaw"));
    rootLayout->addWidget(m_resultTable);

    // ---- Summary ----------------------------------------------------------
    m_summaryLabel = new QLabel(this);
    m_summaryLabel->setWordWrap(true);
    m_summaryLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_summaryLabel->setVisible(false);
    rootLayout->addWidget(m_summaryLabel);

    // ---- Progress section -------------------------------------------------
    ecvAICoreUi::setupProgressSection(rootLayout, m_downloadLabel, m_progress);

    // ---- Shared action row ------------------------------------------------
    auto* actionRow = new QHBoxLayout;
    actionRow->setSpacing(ecvAICoreUi::hSpacing());
    m_ensureModelBtn = new QPushButton(tr("Download/Ensure Model"), this);
    m_ensureModelBtn->setToolTip(
            tr("Download the active checkpoint's selected quantization into "
               "the shared AICore cache (SHA-256 verified)"));
    connect(m_ensureModelBtn, &QPushButton::clicked, this,
            &OpenPCDetDialog::onEnsureModel);
    actionRow->addWidget(m_ensureModelBtn);
    m_exportCsvBtn = new QPushButton(tr("Export CSV"), this);
    m_exportCsvBtn->setEnabled(false);
    m_exportCsvBtn->setToolTip(
            tr("Export the last detection result as x,y,z,dx,dy,dz,heading,"
               "score,label CSV"));
    connect(m_exportCsvBtn, &QPushButton::clicked, this,
            &OpenPCDetDialog::onExportCsv);
    actionRow->addWidget(m_exportCsvBtn);
    actionRow->addStretch();
    rootLayout->addLayout(actionRow);

    // Downloader.
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
            &OpenPCDetDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[OpenPCDet] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[OpenPCDet] Model downloaded: %1").arg(path));
                if (m_pendingActionAfterDownload == PendingAction::Run) {
                    m_pendingActionAfterDownload = PendingAction::None;
                    onRun();
                }
            });

    // Shared test-data repository (openpcdet_test_data): connected ONCE
    // here — the repository is a process-wide singleton and re-connecting
    // per click would multiply the slots. The download→extract chain lives
    // in the shared ensureDatasetReady API; the dialog only consumes
    // datasetReady (plus progress/log feedback while our chain runs).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            &OpenPCDetDialog::onTestDataProgress);
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            &OpenPCDetDialog::onTestDataLog);
    connect(&testDataRepo, &ecvTestDataRepository::datasetReady, this,
            &OpenPCDetDialog::onTestDataReady);
}

void OpenPCDetDialog::populatePanels() {
    QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints;
    const auto catalog = OpenPCDetHelpers::catalogCheckpoints();
    // Three fixed groups: O(n), preserving catalog order within each group.
    // Interleaved catalog rows must not produce repeated navigation headers.
    for (int group = GroupKitti; group <= GroupScene; ++group)
        for (const auto& checkpoint : catalog)
            if (int(groupForCheckpoint(checkpoint)) == group)
                checkpoints.append(checkpoint);
    const QStringList groupTitles = {tr("KITTI point cloud"),
                                     tr("nuScenes point cloud"),
                                     tr("Multi-modal scene")};

    auto addSectionHeader = [this](const QString& title) {
        auto* header = new QListWidgetItem(title, m_taskList);
        header->setFlags(Qt::NoItemFlags);
        QFont bold = header->font();
        bold.setBold(true);
        header->setFont(bold);
        header->setForeground(palette().mid());
        header->setToolTip(title);
    };
    auto addTaskItem = [this](const QString& title, int stackIndex) {
        auto* item = new QListWidgetItem(title, m_taskList);
        item->setData(kTaskStackIndexRole, stackIndex);
        item->setToolTip(title);
    };

    int lastGroup = -1;
    for (const OpenPCDetHelpers::OpenPCDetCheckpoint& cp : checkpoints) {
        const CheckpointGroup group = groupForCheckpoint(cp);
        if (group != lastGroup) {
            addSectionHeader(groupTitles[static_cast<int>(group)]);
            lastGroup = group;
        }

        OpenPCDetTaskPanel panel;
        panel.checkpoint = cp.stem;
        panel.tab = new QWidget(this);
        auto* layout = new QVBoxLayout(panel.tab);
        ecvAICoreUi::setupTabLayout(layout);

        // Two-column body: config controls on the left, scene preview on
        // the right (scene panels only).
        auto* contentRow = new QHBoxLayout;
        contentRow->setSpacing(ecvAICoreUi::hSpacing());
        auto* inputRow = new QHBoxLayout;
        inputRow->setSpacing(ecvAICoreUi::hSpacing());
        auto* inputCol = new QVBoxLayout;
        inputCol->setSpacing(ecvAICoreUi::tightVSpacing());
        auto* configCol = new QVBoxLayout;
        configCol->setSpacing(ecvAICoreUi::vSpacing());

        // Quantization row: the three published quantizations of this
        // checkpoint, straight from the AICore catalog rows.
        auto* quantRow = new QHBoxLayout;
        quantRow->setSpacing(ecvAICoreUi::hSpacing());
        quantRow->addWidget(ecvAICoreUi::makeLabel(tr("Quantization:")));
        panel.quantCombo = new QComboBox(panel.tab);
        // Keep full checkpoint labels in the popup without widening the form.
        panel.quantCombo->setSizeAdjustPolicy(
                QComboBox::AdjustToMinimumContentsLengthWithIcon);
        panel.quantCombo->setMinimumContentsLength(8);
        panel.quantCombo->setSizePolicy(QSizePolicy::Expanding,
                                        QSizePolicy::Fixed);
        for (const OpenPCDetModelEntry& e : cp.quants) {
            panel.quantCombo->addItem(OpenPCDetHelpers::modelDisplayLabel(e),
                                      e.filename);
        }
        // The published f16 rows carry the "(recommended)" marker — the
        // shared selection policy picks it as the default (defaultIndex is
        // left unset: the catalog-wide default row index does not map onto
        // this panel's 3-row view).
        ecvAICoreUi::selectModelRow(panel.quantCombo, QString(), -1);
        quantRow->addWidget(panel.quantCombo, 1);
        configCol->addLayout(quantRow);

        // Model family / dataset notes from the AICore catalog.
        panel.modelInfoLabel = new QLabel(cp.familyNote, panel.tab);
        panel.modelInfoLabel->setWordWrap(true);
        panel.modelInfoLabel->setStyleSheet(
                QStringLiteral("color: palette(placeholder-text);"));
        panel.modelInfoLabel->setVisible(!cp.familyNote.isEmpty());
        configCol->addWidget(panel.modelInfoLabel);

        // Threshold row (0 = follow the GGUF-declared value).
        auto* thresholdRow = new QWidget(panel.tab);
        auto* thresholdLayout = new QHBoxLayout(thresholdRow);
        thresholdLayout->setContentsMargins(0, 0, 0, 0);
        thresholdLayout->setSpacing(ecvAICoreUi::tightHSpacing());
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("Score:")));
        panel.score = new QDoubleSpinBox(thresholdRow);
        panel.score->setRange(0.0, 1.0);
        panel.score->setSingleStep(0.05);
        panel.score->setDecimals(2);
        panel.score->setValue(0.0);
        panel.score->setSpecialValueText(tr("model"));
        panel.score->setToolTip(
                tr("Detection score threshold override; 0 follows the "
                   "GGUF-declared value"));
        ecvAICoreUi::setCompactDoubleSpin(panel.score);
        thresholdLayout->addWidget(panel.score);
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("NMS:")));
        panel.nms = new QDoubleSpinBox(thresholdRow);
        panel.nms->setRange(0.0, 1.0);
        panel.nms->setSingleStep(0.05);
        panel.nms->setDecimals(2);
        panel.nms->setValue(0.0);
        panel.nms->setSpecialValueText(tr("model"));
        panel.nms->setToolTip(
                tr("Rotated NMS IoU override; 0 follows the "
                   "GGUF-declared value"));
        ecvAICoreUi::setCompactDoubleSpin(panel.nms);
        thresholdLayout->addWidget(panel.nms);
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("Max boxes:")));
        panel.maxBoxes = new QSpinBox(thresholdRow);
        panel.maxBoxes->setRange(0, 500);
        panel.maxBoxes->setValue(0);
        panel.maxBoxes->setSpecialValueText(tr("model"));
        panel.maxBoxes->setToolTip(
                tr("Maximum returned boxes; 0 follows "
                   "the GGUF-declared value"));
        ecvAICoreUi::setCompactSpin(panel.maxBoxes);
        thresholdLayout->addWidget(panel.maxBoxes);
        thresholdLayout->addStretch();
        configCol->addWidget(thresholdRow);

        // Custom GGUF row (optional override of the catalog selection).
        panel.customModelRow = new QWidget(panel.tab);
        auto* customRow = new QHBoxLayout(panel.customModelRow);
        customRow->setContentsMargins(0, 0, 0, 0);
        customRow->setSpacing(ecvAICoreUi::hSpacing());
        customRow->addWidget(ecvAICoreUi::makeLabel(tr("Custom GGUF:")));
        panel.customModelPath = new QLineEdit(panel.customModelRow);
        panel.customModelPath->setPlaceholderText(
                tr("Optional — overrides the checkpoint selection"));
        customRow->addWidget(panel.customModelPath, 1);
        auto* browseCustomBtn =
                ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.customModelRow);
        connect(browseCustomBtn, &QPushButton::clicked, this,
                [this]() { onBrowseCustomModel(); });
        customRow->addWidget(browseCustomBtn);
        configCol->addWidget(panel.customModelRow);

        const bool scenePanel = group == GroupScene;

        // Active cloud row (every panel: point inputs and the LiDAR branch
        // of scene runs share it).
        auto* cloudRow = new QHBoxLayout;
        cloudRow->setSpacing(ecvAICoreUi::hSpacing());
        cloudRow->addWidget(ecvAICoreUi::makeLabel(tr("Cloud:")));
        panel.cloudPath = new QLineEdit(panel.tab);
        panel.cloudPath->setReadOnly(true);
        panel.cloudPath->setPlaceholderText(
                tr("Select a point cloud in the DB tree, then reopen this "
                   "dialog"));
        cloudRow->addWidget(panel.cloudPath, 1);
        inputCol->addLayout(cloudRow);

        // Optional raw point file (takes precedence over the active cloud).
        panel.pointsFileRow = new QWidget(panel.tab);
        auto* pointsRow = new QHBoxLayout(panel.pointsFileRow);
        pointsRow->setContentsMargins(0, 0, 0, 0);
        pointsRow->setSpacing(ecvAICoreUi::hSpacing());
        pointsRow->addWidget(ecvAICoreUi::makeLabel(tr("Point file:")));
        panel.pointsFileEdit = new QLineEdit(panel.pointsFileRow);
        panel.pointsFileEdit->setPlaceholderText(
                tr("Optional KITTI .bin / ASCII .txt — overrides the cloud"));
        pointsRow->addWidget(panel.pointsFileEdit, 1);
        auto* browsePointsBtn =
                ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.pointsFileRow);
        connect(browsePointsBtn, &QPushButton::clicked, this, [this]() {
            OpenPCDetTaskPanel* active = currentPanel();
            if (!active) return;
            QSettings settings;
            const QString lastDir = ecvPS::browseDir(
                    settings, QStringLiteral("qOpenPCDet"),
                    QStringLiteral("lastPointsDir"), QDir::homePath());
            const QString path = cvFileDialog::getOpenFileName(
                    this, tr("Select a point cloud file"), lastDir,
                    tr("Point files (*.bin *.txt *.pts *.xyz);;"
                       "All files (*)"));
            if (path.isEmpty()) return;
            active->pointsFileEdit->setText(path);
            ecvPS::saveBrowseDir(settings, QStringLiteral("qOpenPCDet"),
                                 QStringLiteral("lastPointsDir"), path);
        });
        pointsRow->addWidget(browsePointsBtn);
        inputCol->addWidget(panel.pointsFileRow);

        // Scene rows (multi-modal panels only).
        panel.sceneDirRow = new QWidget(panel.tab);
        auto* sceneRow = new QHBoxLayout(panel.sceneDirRow);
        sceneRow->setContentsMargins(0, 0, 0, 0);
        sceneRow->setSpacing(ecvAICoreUi::hSpacing());
        sceneRow->addWidget(ecvAICoreUi::makeLabel(tr("Scene dir:")));
        panel.sceneDirEdit = new QLineEdit(panel.sceneDirRow);
        panel.sceneDirEdit->setPlaceholderText(
                tr("Directory with images.bin + calibration + scene.json"));
        connect(panel.sceneDirEdit, &QLineEdit::textChanged, this,
                [this](const QString& text) {
                    refreshScenePreview(currentPanel(), text.trimmed());
                });
        sceneRow->addWidget(panel.sceneDirEdit, 1);
        auto* browseSceneBtn =
                ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.sceneDirRow);
        connect(browseSceneBtn, &QPushButton::clicked, this, [this]() {
            OpenPCDetTaskPanel* active = currentPanel();
            if (!active) return;
            QSettings settings;
            const QString lastDir = ecvPS::browseDir(
                    settings, QStringLiteral("qOpenPCDet"),
                    QStringLiteral("lastSceneDir"), QDir::homePath());
            const QString path = cvFileDialog::getExistingDirectory(
                    this, tr("Select a scene directory"), lastDir);
            if (path.isEmpty()) return;
            active->sceneDirEdit->setText(path);
            ecvPS::saveBrowseDir(settings, QStringLiteral("qOpenPCDet"),
                                 QStringLiteral("lastSceneDir"), path);
        });
        sceneRow->addWidget(browseSceneBtn);
        inputCol->addWidget(panel.sceneDirRow);

        // LiDAR branch point source of a scene run: the active cloud or an
        // extra point file (BEVFusion requires points; CaDDN ignores them).
        panel.scenePointsFileRow = new QWidget(panel.tab);
        auto* scenePointsRow = new QHBoxLayout(panel.scenePointsFileRow);
        scenePointsRow->setContentsMargins(0, 0, 0, 0);
        scenePointsRow->setSpacing(ecvAICoreUi::hSpacing());
        scenePointsRow->addWidget(ecvAICoreUi::makeLabel(tr("LiDAR file:")));
        panel.scenePointsFileEdit = new QLineEdit(panel.scenePointsFileRow);
        panel.scenePointsFileEdit->setPlaceholderText(
                tr("LiDAR .bin for the fusion branch (required by BEVFusion "
                   "and the focal VoxelRCNN, ignored by CaDDN)"));
        scenePointsRow->addWidget(panel.scenePointsFileEdit, 1);
        auto* browseScenePointsBtn = ecvAICoreUi::makeBrowseBtn(
                tr("Browse…"), panel.scenePointsFileRow);
        connect(browseScenePointsBtn, &QPushButton::clicked, this, [this]() {
            OpenPCDetTaskPanel* active = currentPanel();
            if (!active) return;
            QSettings settings;
            const QString lastDir = ecvPS::browseDir(
                    settings, QStringLiteral("qOpenPCDet"),
                    QStringLiteral("lastPointsDir"), QDir::homePath());
            const QString path = cvFileDialog::getOpenFileName(
                    this, tr("Select a LiDAR point file"), lastDir,
                    tr("Point files (*.bin *.txt *.pts *.xyz);;"
                       "All files (*)"));
            if (path.isEmpty()) return;
            active->scenePointsFileEdit->setText(path);
            ecvPS::saveBrowseDir(settings, QStringLiteral("qOpenPCDet"),
                                 QStringLiteral("lastPointsDir"), path);
        });
        scenePointsRow->addWidget(browseScenePointsBtn);
        inputCol->addWidget(panel.scenePointsFileRow);
        panel.scenePointsFileRow->setVisible(scenePanel);
        panel.sceneDirRow->setVisible(scenePanel);
        panel.pointsFileRow->setVisible(!scenePanel);

        inputRow->addLayout(inputCol, 1);
        configCol->addStretch();
        contentRow->addLayout(configCol, 1);

        // Right column: scene image preview (scene panels only; keeps the
        // CaDDN / BEVFusion input visible before running).
        if (scenePanel) {
            auto* previewCol = new QVBoxLayout;
            previewCol->setSpacing(ecvAICoreUi::vSpacing());
            panel.previewLabel = new ecvClickableImageLabel(panel.tab);
            const int ps = ecvAICoreUi::previewSize();
            panel.previewLabel->setFixedSize(ps, ps);
            panel.previewLabel->setStyleSheet(
                    "border: 1px solid palette(mid); "
                    "background: palette(base);");
            panel.previewLabel->setText(tr("Preview"));
            previewCol->addWidget(panel.previewLabel);
            auto* previewHint =
                    new QLabel(tr("Camera 0 of the scene"), panel.tab);
            previewHint->setAlignment(Qt::AlignCenter);
            previewHint->setStyleSheet(
                    QStringLiteral("color: palette(mid); font-size: 11px;"));
            previewCol->addWidget(previewHint);
            previewCol->addStretch();
            inputRow->addLayout(previewCol);
        }
        layout->addLayout(inputRow);
        auto* configPage = new QWidget(panel.tab);
        auto* configLayout = new QVBoxLayout(configPage);
        configLayout->setContentsMargins(0, 0, 0, 0);
        configLayout->addLayout(contentRow);
        auto* configScroll = new QScrollArea(panel.tab);
        configScroll->setWidgetResizable(true);
        configScroll->setFrameShape(QFrame::NoFrame);
        configScroll->setWidget(configPage);
        configScroll->setMinimumHeight(ecvAICoreUi::dpiScaled(80));
        layout->addWidget(configScroll, 1);

        // Action row: add-to-DB + Run / Cancel (per panel, like qYOLO).
        auto* panelActionRow = new QHBoxLayout;
        panelActionRow->setSpacing(ecvAICoreUi::hSpacing());
        panel.addBoxesCheck =
                new QCheckBox(tr("Add wireframe boxes to DB"), panel.tab);
        panel.addBoxesCheck->setChecked(true);
        panel.addBoxesCheck->setToolTip(
                tr("Write the detections as per-class colored wireframe "
                   "polylines into the DB tree"));
        auto* outputRow = new QHBoxLayout;
        outputRow->addWidget(panel.addBoxesCheck);
        panel.annotatedCheck =
                new QCheckBox(tr("Export annotated cloud"), panel.tab);
        panel.annotatedCheck->setChecked(true);
        panel.annotatedCheck->setToolTip(
                tr("Export a copy of the source point cloud into the DB "
                   "tree with in-box points tinted by detection class "
                   "color (matches the OpenPCDet Python demo styling)"));
        outputRow->addWidget(panel.annotatedCheck);
        outputRow->addStretch();
        layout->addLayout(outputRow);
        panelActionRow->addStretch();
        panel.testDataBtn = ecvAICoreUi::makeSampleDataBtn(panel.tab);
        panel.testDataBtn->setToolTip(
                tr("Fill this panel from the shared OpenPCDet detection "
                   "sample (downloaded once into the shared test-data cache, "
                   "SHA-256 verified): KITTI panels take the raw 000008 "
                   "frame (PointRCNN its pre-sampled 16384-point variant), "
                   "nuScenes panels the n015 sweep, CaDDN the single-camera "
                   "scene and BEVFusion the 6-camera scene"));
        connect(panel.testDataBtn, &QPushButton::clicked, this,
                &OpenPCDetDialog::onUseTestData);
        panelActionRow->addWidget(panel.testDataBtn);
        panel.runBtn = new QPushButton(tr("Run"), panel.tab);
        panel.runBtn->setDefault(m_panels.isEmpty());
        panelActionRow->addWidget(panel.runBtn);
        panel.cancelBtn = new QPushButton(tr("Cancel"), panel.tab);
        panel.cancelBtn->setEnabled(false);
        panelActionRow->addWidget(panel.cancelBtn);
        layout->addLayout(panelActionRow);

        connect(panel.runBtn, &QPushButton::clicked, this,
                &OpenPCDetDialog::onRun);
        connect(panel.cancelBtn, &QPushButton::clicked, this,
                &OpenPCDetDialog::onCancel);

        m_panels.append(panel);
        m_taskStack->addWidget(panel.tab);
        addTaskItem(panelLabel(cp), m_taskStack->count() - 1);
    }

    connect(m_taskList, &QListWidget::currentRowChanged, this, [this](int row) {
        QListWidgetItem* item = m_taskList->item(row);
        if (!item) return;
        const QVariant page = item->data(kTaskStackIndexRole);
        if (!page.isValid()) return;
        m_taskStack->setCurrentIndex(page.toInt());
    });
    // Select the first selectable entry (row 0 is a section header).
    for (int row = 0; row < m_taskList->count(); ++row) {
        if (m_taskList->item(row)->data(kTaskStackIndexRole).isValid()) {
            m_taskList->setCurrentRow(row);
            break;
        }
    }
}

int OpenPCDetDialog::taskListIdealWidth() const {
    if (!m_taskList) return 0;
    // Content-driven, font/DPI aware: measure the widest entry text
    // (section headers render bold) plus item padding, frame and scrollbar
    // room, so no entry is elided by default.
    int textWidth = 0;
    const QFontMetrics metrics(m_taskList->font());
    for (int i = 0; i < m_taskList->count(); ++i) {
        textWidth =
                qMax(textWidth,
                     metrics.horizontalAdvance(m_taskList->item(i)->text()));
    }
    return textWidth + ecvAICoreUi::dpiScaled(36);
}

void OpenPCDetDialog::applyTaskListDefaultWidth() {
    if (!m_bodySplitter || !m_taskList) return;
    const int total = m_bodySplitter->width();
    if (total <= 0) return;
    const int listWidth = qBound(ecvAICoreUi::dpiScaled(120),
                                 taskListIdealWidth(), total / 2);
    m_bodySplitter->setSizes({listWidth, total - listWidth});
}

void OpenPCDetDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    for (const auto& panel : m_panels)
        panel.testDataBtn->setEnabled(!m_testDataBusy && !m_taskRunning);
    if (!m_testDataBusy && !m_taskRunning && !m_downloadInProgress) {
        m_progress->hide();
        m_downloadLabel->hide();
    }
    // First-show allocation of the splitter panes (a user-restored splitter
    // state wins).
    if (!m_splitterSized) {
        m_splitterSized = true;
        if (!m_splitterRestored) applyTaskListDefaultWidth();
    }
}

void OpenPCDetDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void OpenPCDetDialog::setActiveCloud(ccPointCloud* cloud, const QString& name) {
    m_cloud = cloud;
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        if (panel.cloudPath) {
            panel.cloudPath->setText(cloud ? name : QString());
        }
    }
}

void OpenPCDetDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qOpenPCDet"));
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        const QString key = panel.checkpoint;
        const QString quant =
                settings.value(QStringLiteral("quant/") + key).toString();
        QString filename;
        if (OpenPCDetHelpers::buildModelFilename(panel.checkpoint, quant,
                                                 &filename)) {
            const int idx = panel.quantCombo->findData(filename);
            if (idx >= 0) panel.quantCombo->setCurrentIndex(idx);
        }
        panel.score->setValue(
                settings.value(QStringLiteral("score/") + key, 0.0).toDouble());
        panel.nms->setValue(
                settings.value(QStringLiteral("nms/") + key, 0.0).toDouble());
        panel.maxBoxes->setValue(
                settings.value(QStringLiteral("maxBoxes/") + key, 0).toInt());
        panel.pointsFileEdit->setText(
                settings.value(QStringLiteral("pointsFile/") + key).toString());
        panel.sceneDirEdit->setText(
                settings.value(QStringLiteral("sceneDir/") + key).toString());
        panel.scenePointsFileEdit->setText(
                settings.value(QStringLiteral("scenePointsFile/") + key)
                        .toString());
        panel.customModelPath->setText(
                settings.value(QStringLiteral("customModel/") + key)
                        .toString());
        panel.addBoxesCheck->setChecked(
                settings.value(QStringLiteral("addBoxes/") + key, true)
                        .toBool());
        panel.annotatedCheck->setChecked(
                settings.value(QStringLiteral("annotatedCloud/") + key, true)
                        .toBool());
    }
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    const QByteArray splitterState =
            settings.value(QStringLiteral("bodySplitterState")).toByteArray();
    if (!splitterState.isEmpty() && m_bodySplitter &&
        m_bodySplitter->restoreState(splitterState)) {
        m_splitterRestored = true;
    }
    QString activeCheckpoint =
            settings.value(QStringLiteral("activeCheckpoint")).toString();
    const int activeRow =
            settings.value(QStringLiteral("activeRow"), -1).toInt();
    settings.endGroup();
    // Migrate the legacy interleaved row index to the stable catalog identity.
    if (activeCheckpoint.isEmpty() && activeRow >= 0) {
        int row = 0, lastGroup = -1;
        for (const auto& checkpoint : OpenPCDetHelpers::catalogCheckpoints()) {
            const int group = int(groupForCheckpoint(checkpoint));
            if (group != lastGroup) {
                ++row;
                lastGroup = group;
            }
            if (row == activeRow) {
                activeCheckpoint = checkpoint.stem;
                break;
            }
            ++row;
        }
    }
    for (int row = 0; row < m_taskList->count(); ++row) {
        const QVariant page = m_taskList->item(row)->data(kTaskStackIndexRole);
        if (page.isValid() && page.toInt() >= 0 &&
            page.toInt() < m_panels.size() &&
            m_panels[page.toInt()].checkpoint == activeCheckpoint) {
            m_taskList->setCurrentRow(row);
            break;
        }
    }
}

void OpenPCDetDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qOpenPCDet"));
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        const QString key = panel.checkpoint;
        const QString filename = panel.quantCombo->currentData().toString();
        // Persist the quant suffix only (catalog rows may change between
        // builds; the stem + suffix stays the stable user intent).
        QString quant;
        for (const QString& q : OpenPCDetHelpers::quantSuffixes()) {
            if (filename == panel.checkpoint + QStringLiteral(".") + q +
                                    QStringLiteral(".gguf")) {
                quant = q;
            }
        }
        settings.setValue(QStringLiteral("quant/") + key, quant);
        settings.setValue(QStringLiteral("score/") + key, panel.score->value());
        settings.setValue(QStringLiteral("nms/") + key, panel.nms->value());
        settings.setValue(QStringLiteral("maxBoxes/") + key,
                          panel.maxBoxes->value());
        settings.setValue(QStringLiteral("pointsFile/") + key,
                          panel.pointsFileEdit->text());
        settings.setValue(QStringLiteral("sceneDir/") + key,
                          panel.sceneDirEdit->text());
        settings.setValue(QStringLiteral("scenePointsFile/") + key,
                          panel.scenePointsFileEdit->text());
        settings.setValue(QStringLiteral("customModel/") + key,
                          panel.customModelPath->text());
        settings.setValue(QStringLiteral("addBoxes/") + key,
                          panel.addBoxesCheck->isChecked());
        settings.setValue(QStringLiteral("annotatedCloud/") + key,
                          panel.annotatedCheck->isChecked());
    }
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    if (m_bodySplitter) {
        settings.setValue(QStringLiteral("bodySplitterState"),
                          m_bodySplitter->saveState());
    }
    settings.setValue(QStringLiteral("activeRow"), m_taskList->currentRow());
    if (const auto* panel = currentPanel())
        settings.setValue(QStringLiteral("activeCheckpoint"),
                          panel->checkpoint);
    settings.endGroup();
}

OpenPCDetTaskPanel* OpenPCDetDialog::currentPanel() const {
    if (!m_taskStack) return nullptr;
    QWidget* page = m_taskStack->currentWidget();
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        if (panel.tab == page) {
            return const_cast<OpenPCDetTaskPanel*>(&panel);
        }
    }
    return nullptr;
}

OpenPCDetTaskPanel* OpenPCDetDialog::panelForFilename(
        const QString& filename) const {
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        if (panel.quantCombo && panel.quantCombo->findData(filename) >= 0) {
            return const_cast<OpenPCDetTaskPanel*>(&panel);
        }
    }
    return nullptr;
}

void OpenPCDetDialog::refreshModelList() {
    // The panels read the catalog live at construction; a refresh only
    // needs to re-sync the quantization combos (e.g. after an AICore
    // catalog update) while keeping each panel's explicit choice.
    const QVector<OpenPCDetHelpers::OpenPCDetCheckpoint> checkpoints =
            OpenPCDetHelpers::catalogCheckpoints();
    if (checkpoints.size() != m_panels.size()) return;
    QHash<QString, const OpenPCDetHelpers::OpenPCDetCheckpoint*> byCheckpoint;
    for (const auto& checkpoint : checkpoints)
        byCheckpoint.insert(checkpoint.stem, &checkpoint);
    for (OpenPCDetTaskPanel& panel : m_panels) {
        const auto found = byCheckpoint.constFind(panel.checkpoint);
        if (found == byCheckpoint.constEnd()) continue;
        const QString keep = panel.quantCombo->currentData().toString();
        panel.quantCombo->blockSignals(true);
        panel.quantCombo->clear();
        for (const OpenPCDetModelEntry& e : found.value()->quants) {
            panel.quantCombo->addItem(OpenPCDetHelpers::modelDisplayLabel(e),
                                      e.filename);
        }
        ecvAICoreUi::selectModelRow(panel.quantCombo, keep, -1);
        panel.quantCombo->blockSignals(false);
    }
}

QString OpenPCDetDialog::resolveModelPath(const QString& filename) const {
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = OpenPCDetHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

void OpenPCDetDialog::refreshScenePreview(OpenPCDetTaskPanel* panel,
                                          const QString& sceneDir) {
    if (!panel || !panel->previewLabel) return;
    QImage preview;
    QString error;
    if (!sceneDir.isEmpty() &&
        OpenPCDetHelpers::readScenePreview(sceneDir, &preview, &error)) {
        panel->previewLabel->setPreviewImage(preview,
                                             ecvAICoreUi::previewSize());
    } else {
        panel->previewLabel->clearPreview();
        if (!sceneDir.isEmpty() && !error.isEmpty()) {
            appendLog(tr("[OpenPCDet] Scene preview: %1").arg(error));
        }
    }
}

bool OpenPCDetDialog::ensureModelAvailable(PendingAction action) {
    OpenPCDetTaskPanel* panel = currentPanel();
    if (!panel) return false;
    // Custom GGUF override: only an existing file is accepted.
    const QString custom = panel->customModelPath
                                   ? panel->customModelPath->text().trimmed()
                                   : QString();
    if (!custom.isEmpty()) {
        if (!QFileInfo::exists(custom)) {
            appendLog(tr("[OpenPCDet] Custom GGUF not found: %1").arg(custom));
            return false;
        }
        return true;
    }
    const QString filename = panel->modelFilename();
    if (filename.isEmpty()) {
        appendLog(tr("[OpenPCDet] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath(filename))) {
        OpenPCDetModelEntry entry;
        if (!OpenPCDetHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[OpenPCDet] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[OpenPCDet] Model missing — downloading %1; it will "
                     "start automatically when ready.")
                          .arg(filename));
        startDownload(entry);
        return false;
    }
    return true;
}

void OpenPCDetDialog::startDownload(const OpenPCDetModelEntry& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[OpenPCDet] A download is already running."));
        return;
    }
    QDir().mkpath(OpenPCDetHelpers::modelCacheDir());
    const QString dest = OpenPCDetHelpers::modelCacheDir() + QDir::separator() +
                         model.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[OpenPCDet] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[OpenPCDet] Downloading %1 (%2)…")
                      .arg(model.filename, model.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // OpenPCDet GGUFs are tens of MB
    // Content identity from the release digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught).
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(model.filename)};
    m_downloader->download(req);
}

void OpenPCDetDialog::onEnsureModel() {
    OpenPCDetTaskPanel* panel = currentPanel();
    if (!panel) return;
    const QString custom = panel->customModelPath
                                   ? panel->customModelPath->text().trimmed()
                                   : QString();
    if (!custom.isEmpty()) {
        appendLog(
                QFileInfo::exists(custom)
                        ? tr("[OpenPCDet] Custom model present: %1").arg(custom)
                        : tr("[OpenPCDet] Custom GGUF not found: %1")
                                  .arg(custom));
        return;
    }
    const QString filename = panel->modelFilename();
    if (filename.isEmpty()) {
        appendLog(tr("[OpenPCDet] Select a model first."));
        return;
    }
    if (QFileInfo::exists(resolveModelPath(filename))) {
        appendLog(tr("[OpenPCDet] Model already present: %1")
                          .arg(resolveModelPath(filename)));
        return;
    }
    OpenPCDetModelEntry entry;
    if (!OpenPCDetHelpers::findModelByFilename(filename, &entry)) {
        appendLog(tr("[OpenPCDet] '%1' is not a catalog model — point "
                     "Custom GGUF at an existing file instead.")
                          .arg(filename));
        return;
    }
    startDownload(entry);
}

void OpenPCDetDialog::onBrowseCustomModel() {
    OpenPCDetTaskPanel* active = currentPanel();
    if (!active || !active->customModelPath) return;
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qOpenPCDet"),
            QStringLiteral("lastModelDir"), OpenPCDetHelpers::modelCacheDir());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select OpenPCDet GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    active->customModelPath->setText(path);
}

// ---------------------------------------------------------------------------
// Test data — via the shared ecvTestDataRepository
// ---------------------------------------------------------------------------

void OpenPCDetDialog::onUseTestData() {
    OpenPCDetTaskPanel* panel = currentPanel();
    if (!panel) return;
    int panelIndex = -1;
    for (int i = 0; i < m_panels.size(); ++i) {
        if (m_panels[i].tab == panel->tab) {
            panelIndex = i;
            break;
        }
    }
    // Readiness owns cache probes, verified archive reuse and cold preparation.
    // A cache miss is recoverable, not a detection error.
    if (m_taskRunning || m_downloadInProgress || m_testDataBusy ||
        ecvTestDataRepository::instance().isDownloadInProgress()) {
        appendLog(
                tr("[Test data] Wait for the current test-data download "
                   "to finish first."));
        return;
    }
    const auto kind = ecvTestDataRepository::Dataset::OpenPCDetSample;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataBusy = true;
    m_pendingTestDataPanel = panelIndex;
    if (panel->testDataBtn) panel->testDataBtn->setEnabled(false);
    appendLog(tr("[Test data] Preparing %1 (download / SHA-256 verify / "
                 "extract as needed)...")
                      .arg(info.displayName));
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(true);
    m_downloadLabel->setText(tr("Downloading %1...").arg(info.displayName));
    m_downloadLabel->setVisible(true);
    ecvTestDataRepository::instance().ensureDatasetReady(kind);
}

void OpenPCDetDialog::onTestDataProgress(int percent,
                                         const QString& statusText) {
    if (!m_testDataBusy) return;
    m_progress->setRange(0, 100);
    m_progress->setValue(percent);
    m_progress->setVisible(true);
    m_downloadLabel->setText(statusText);
    m_downloadLabel->setVisible(true);
}

void OpenPCDetDialog::onTestDataLog(const QString& message) {
    if (m_testDataBusy) appendLog(message);
}

void OpenPCDetDialog::onTestDataReady(bool success,
                                      ecvTestDataRepository::Dataset kind) {
    if (kind != ecvTestDataRepository::Dataset::OpenPCDetSample) return;
    if (!m_testDataBusy) return;  // a foreign chain we did not start
    m_testDataBusy = false;
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        if (panel.testDataBtn) panel.testDataBtn->setEnabled(true);
    }
    const int panelIndex = m_pendingTestDataPanel;
    m_pendingTestDataPanel = -1;
    if (!success) {
        appendLog(
                tr("[Error] Test-data preparation failed (download / "
                   "SHA-256 verification / extraction)."));
        return;
    }
    fillTestDataPanel(panelIndex);
}

bool OpenPCDetDialog::fillTestDataPanel(int panelIndex) {
    if (panelIndex < 0 || panelIndex >= m_panels.size()) return false;
    OpenPCDetTaskPanel& panel = m_panels[panelIndex];
    const QString bundleRoot = ecvTestDataRepository::extractPath(
            ecvTestDataRepository::Dataset::OpenPCDetSample);
    OpenPCDetHelpers::OpenPCDetTestDataFiles files;
    QString error;
    if (!OpenPCDetHelpers::resolveTestDataPaths(bundleRoot, panel.checkpoint,
                                                &files, &error)) {
        appendLog(tr("[Error] %1").arg(error));
        return false;
    }
    if (!files.pointsFile.isEmpty()) {
        panel.pointsFileEdit->setText(files.pointsFile);
    }
    if (!files.sceneDirName.isEmpty()) {
        // Signal blocked: the textChanged hook refreshes the CURRENT
        // panel's preview, which would cross-fill another panel's preview
        // when the user switched tabs while the download ran.
        panel.sceneDirEdit->blockSignals(true);
        panel.sceneDirEdit->setText(files.sceneDirName);
        panel.sceneDirEdit->blockSignals(false);
        refreshScenePreview(&panel, files.sceneDirName);
    }
    if (!files.scenePointsFile.isEmpty()) {
        panel.scenePointsFileEdit->setText(files.scenePointsFile);
    }
    appendLog(tr("[Test data] Filled '%1' from the shared detection sample.")
                      .arg(panel.checkpoint));
    // Same showcase cloud for every panel: the KITTI 000008 frame lands in
    // the DB tree and gets selected, so the active-cloud Run path works
    // without touching the point file again.
    loadSampleCloudIntoDb(ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::OpenPCDetSample,
            QStringLiteral("kitti_000008.bin")));
    return true;
}

void OpenPCDetDialog::loadSampleCloudIntoDb(const QString& binPath) {
    if (!m_app) return;  // no DB tree (tests / headless)
    if (binPath.isEmpty()) {
        appendLog(
                tr("[Error] Sample frame kitti_000008.bin not found in "
                   "the test-data cache."));
        return;
    }
    QFile file(binPath);
    if (!file.open(QIODevice::ReadOnly)) {
        appendLog(tr("[Error] Cannot read %1").arg(binPath));
        return;
    }
    const QByteArray bytes = file.readAll();
    file.close();

    // Inverse of the worker's cloud input path (extractCloudPoints): the
    // 4th KITTI feature becomes the "intensity" scalar field the worker
    // picks again on Run.
    std::vector<float> points;
    int count = 0;
    QString error;
    if (!OpenPCDetHelpers::parsePointsBin(bytes, 4, &points, &count, &error)) {
        appendLog(tr("[Error] Sample frame parse failed: %1").arg(error));
        return;
    }
    auto* cloud = new ccPointCloud;
    if (!cloud->reserve(static_cast<unsigned>(count))) {
        delete cloud;
        appendLog(tr("[Error] Not enough memory for the sample frame (%1 "
                     "points).")
                          .arg(count));
        return;
    }
    for (int i = 0; i < count; ++i) {
        const float* row = points.data() + static_cast<size_t>(i) * 4;
        cloud->addPoint(CCVector3(static_cast<PointCoordinateType>(row[0]),
                                  static_cast<PointCoordinateType>(row[1]),
                                  static_cast<PointCoordinateType>(row[2])));
    }
    auto* intensity = new ccScalarField("intensity");
    bool sfAdded = false;
    if (intensity->reserveSafe(static_cast<unsigned>(count))) {
        for (int i = 0; i < count; ++i) {
            intensity->addElement(points[static_cast<size_t>(i) * 4 + 3]);
        }
        intensity->computeMinAndMax();
        sfAdded = cloud->addScalarField(intensity) >= 0;
    }
    if (!sfAdded) {
        intensity->release();
    }

    const QString cloudName = ecvPluginDbNaming::makeUnique(
            QStringLiteral("KITTI 000008 (sample)"), m_app);
    cloud->setName(cloudName);
    cloud->setMetaData(QStringLiteral("OpenPCDet"), true);
    cloud->setMetaData(QStringLiteral("OpenPCDet/Sample"),
                       QStringLiteral("kitti_000008.bin"));
    m_app->addToDB(cloud, true, true, false, true);
    m_app->setSelectedInDB(cloud, true);
    m_app->updateUI();
    setActiveCloud(cloud, cloud->getName());
    appendLog(tr("[Test data] Loaded sample cloud '%1' (%2 points) into the "
                 "DB tree.")
                      .arg(cloudName)
                      .arg(count));
}

OpenPCDetDialog::Settings OpenPCDetDialog::getSettings() const {
    Settings s;
    const OpenPCDetTaskPanel* panel = currentPanel();
    if (!panel) return s;
    s.modelPath = panel->modelPath();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    s.scoreThresh = static_cast<float>(panel->score->value());
    s.nmsThresh = static_cast<float>(panel->nms->value());
    s.maxBoxes = panel->maxBoxes->value();
    s.addBoxesToDb = panel->addBoxesCheck->isChecked();
    s.annotatedCloudToDb = panel->annotatedCheck->isChecked();
    if (!panel->sceneDirEdit->text().trimmed().isEmpty()) {
        s.sceneDir = panel->sceneDirEdit->text().trimmed();
        // The LiDAR branch of a scene run: an explicit file wins, else the
        // active cloud (resolved by the plugin) is used.
        s.pointsFile = panel->scenePointsFileEdit->text().trimmed();
    } else {
        s.pointsFile = panel->pointsFileEdit->text().trimmed();
    }
    return s;
}

void OpenPCDetDialog::onRun() {
    OpenPCDetTaskPanel* panel = currentPanel();
    if (!panel) return;
    // Validate the input BEFORE the model chain: a missing input would
    // otherwise trigger the model download first and then fail after the
    // wait.
    const bool sceneRun =
            OpenPCDetHelpers::inputKindForFilename(panel->modelFilename()) ==
                    OpenPCDetHelpers::InputKind::Scene &&
            panel->customModelPath->text().trimmed().isEmpty();
    if (sceneRun && panel->sceneDirEdit->text().trimmed().isEmpty()) {
        appendLog(
                tr("[Error] Select a scene directory first (images + "
                   "calibration)."));
        return;
    }
    if (!sceneRun) {
        const bool hasFile =
                !panel->pointsFileEdit->text().trimmed().isEmpty() ||
                !panel->scenePointsFileEdit->text().trimmed().isEmpty();
        if (!hasFile && (m_cloud == nullptr || m_cloud->size() == 0)) {
            appendLog(
                    tr("[Error] Select a non-empty point cloud in the DB "
                       "tree or a point file first."));
            return;
        }
    }
    if (!ensureModelAvailable(PendingAction::Run)) return;
    emit runRequested(getSettings());
}

void OpenPCDetDialog::onCancel() {
    if (m_downloadInProgress) m_downloader->cancel();
    emit cancelRequested();
}

void OpenPCDetDialog::showDetectionSummary(const OpenPCDetRunResult& result) {
    m_lastResult = result;
    m_exportCsvBtn->setEnabled(!result.detections.isEmpty());
    updateResultTable(result);

    // Per-class counts, ordered by label.
    const std::map<int32_t, int> counts =
            OpenPCDetHelpers::perClassCounts(result.detections);
    QStringList parts;
    for (const auto& kv : counts) {
        parts << QStringLiteral("%1 ×%2")
                         .arg(OpenPCDetHelpers::className(kv.first,
                                                          result.numClass))
                         .arg(kv.second);
    }
    QString prefix =
            result.fromScene
                    ? tr("<b>%1 box(es)</b> from scene %2 (%3 "
                         "camera(s)) — %4")
                              .arg(result.detections.size())
                              .arg(QFileInfo(result.sceneDir).fileName())
                              .arg(result.numCameras)
                              .arg(parts.isEmpty()
                                           ? tr("no detections")
                                           : parts.join(QStringLiteral(", ")))
                    : tr("<b>%1 box(es)</b> — %2")
                              .arg(result.detections.size())
                              .arg(parts.isEmpty()
                                           ? tr("no detections")
                                           : parts.join(QStringLiteral(", ")));
    m_summaryLabel->setText(prefix);
    m_summaryLabel->setVisible(true);
}

void OpenPCDetDialog::updateResultTable(const OpenPCDetRunResult& result) {
    if (!m_resultTable) return;
    m_resultTable->setRowCount(result.detections.size());
    for (int i = 0; i < result.detections.size(); ++i) {
        const OpenPCDetDetection& d = result.detections[i];
        const QStringList cells = {
                d.className,
                QStringLiteral("%1").arg(d.score, 0, 'f', 3),
                QStringLiteral("%1").arg(d.x, 0, 'f', 3),
                QStringLiteral("%1").arg(d.y, 0, 'f', 3),
                QStringLiteral("%1").arg(d.z, 0, 'f', 3),
                QStringLiteral("%1").arg(d.dx, 0, 'f', 3),
                QStringLiteral("%1").arg(d.dy, 0, 'f', 3),
                QStringLiteral("%1").arg(d.dz, 0, 'f', 3),
                QStringLiteral("%1").arg(d.heading, 0, 'f', 3)};
        for (int c = 0; c < cells.size() && c < m_resultTable->columnCount();
             ++c) {
            m_resultTable->setItem(i, c, new QTableWidgetItem(cells.at(c)));
        }
    }
    m_resultTable->resizeColumnsToContents();
}

void OpenPCDetDialog::onExportCsv() {
    if (m_lastResult.detections.isEmpty()) return;
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qOpenPCDet"),
                             QStringLiteral("lastCsvDir"), QDir::homePath());
    QString path = cvFileDialog::getSaveFileName(
            this, tr("Export detections CSV"),
            QDir(lastDir).filePath(
                    QFileInfo(m_lastResult.modelPath).completeBaseName() +
                    QStringLiteral("_detections.csv")),
            tr("CSV files (*.csv);;All files (*)"));
    if (path.isEmpty()) return;
    if (!path.endsWith(QStringLiteral(".csv"), Qt::CaseInsensitive)) {
        path += QStringLiteral(".csv");
    }
    ecvPS::saveBrowseDir(settings, QStringLiteral("qOpenPCDet"),
                         QStringLiteral("lastCsvDir"), path);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        appendLog(tr("[Error] Cannot write %1").arg(path));
        return;
    }
    QTextStream stream(&file);
    qtCompatSetTextStreamUtf8(stream);
    stream << OpenPCDetHelpers::csvHeader() << QtCompat::endl;
    for (const OpenPCDetDetection& d : m_lastResult.detections) {
        stream << OpenPCDetHelpers::detectionToCsvRow(d) << QtCompat::endl;
    }
    file.close();
    appendLog(tr("[OpenPCDet] Exported %1 detection(s) to %2")
                      .arg(m_lastResult.detections.size())
                      .arg(path));
}

void OpenPCDetDialog::appendLog(const QString& msg) {
#ifdef AICore_ENABLED
    aicore_inference_log::log(msg);
#endif
}

void OpenPCDetDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void OpenPCDetDialog::setRunning(bool running) {
    m_taskRunning = running;
    for (const OpenPCDetTaskPanel& panel : m_panels) {
        panel.runBtn->setEnabled(!running);
        panel.cancelBtn->setEnabled(running);
    }
    if (running) {
        m_progress->setVisible(true);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    } else {
        m_progress->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    }
    m_ensureModelBtn->setEnabled(!running && !m_downloadInProgress);
}

void OpenPCDetDialog::closeEvent(QCloseEvent* event) {
    // Uniform plugin-close semantics: if a task or download is active, ask
    // for confirmation before closing.
    if (m_taskRunning || m_downloadInProgress) {
        if (QMessageBox::question(this, tr("Task running"),
                                  tr("An OpenPCDet task is running. Close "
                                     "anyway?"),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes) {
            event->ignore();
            return;
        }
    }
    m_testDataBusy = false;
    m_pendingTestDataPanel = -1;
    m_pendingActionAfterDownload = PendingAction::None;
    onCancel();
    saveSettings();
    event->accept();
}

void OpenPCDetDialog::reject() {
    m_testDataBusy = false;
    m_pendingTestDataPanel = -1;
    m_pendingActionAfterDownload = PendingAction::None;
    onCancel();
    QDialog::reject();
}

void OpenPCDetDialog::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape &&
        (m_taskRunning || m_downloadInProgress)) {
        if (QMessageBox::question(this, tr("Task running"),
                                  tr("An OpenPCDet task is running. Close "
                                     "anyway?"),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes) {
            return;
        }
    }
    QDialog::keyPressEvent(event);
}
