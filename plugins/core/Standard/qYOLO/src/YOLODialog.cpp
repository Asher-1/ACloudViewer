// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLODialog.h"

#include <cvFileDialog.h>

#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QScrollArea>
#include <QSet>
#include <QSettings>
#include <QShowEvent>
#include <QSizePolicy>
#include <QSplitter>
#include <QTimer>
#include <QVBoxLayout>

#include "ecvAICoreUiHelper.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/yolo_capi.h"
#endif

namespace {
// QListWidgetItem data role carrying the QStackedWidget page index behind
// a task-list entry (section headers are disabled and carry no data).
constexpr int kTaskStackIndexRole = Qt::UserRole + 10;
// QListWidgetItem data role carrying the full-resolution ccImage for the
// click-to-enlarge preview (the list icon is only a scaled thumbnail).
constexpr int kDbFullImageRole = Qt::UserRole + 1;
// One task panel per family, in tab order. taskModels() maps each id onto
// its catalog role, so every tab only offers its own family's models.
QStringList kPanelTasks() {
    return {QStringLiteral("detect"),   QStringLiteral("segment"),
            QStringLiteral("depth"),    QStringLiteral("pose"),
            QStringLiteral("obb"),      QStringLiteral("classify"),
            QStringLiteral("semantic"), QStringLiteral("world"),
            QStringLiteral("yoloe")};
}
// Tabs whose models carry detection thresholds (Conf/IoU/Top-K).
bool taskHasThresholds(const QString& task) {
    return task != QStringLiteral("depth") &&
           task != QStringLiteral("classify") &&
           task != QStringLiteral("semantic");
}
// Open-vocabulary tabs (class list + text-encoder model).
bool taskHasText(const QString& task) {
    return task == QStringLiteral("world") || task == QStringLiteral("yoloe");
}
// Default text-encoder GGUF per open-vocabulary family (world detects with
// CLIP embeddings; yoloe segments with MobileCLIP embeddings).
QString defaultTextModelForTask(const QString& task) {
    return task == QStringLiteral("yoloe")
                   ? QStringLiteral("mobileclip2_b-f16.gguf")
                   : QStringLiteral("clip-ViT-B-32-f16.gguf");
}
}  // namespace

YOLODialog::YOLODialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("YOLO Inference"));
    setupUi();
    populateModelCombo();
    loadSettings();
    m_liveWidget->loadSettings();
    // Content-driven minimum (font / DPI aware) instead of hard-coded
    // pixels, so the dialog adapts to any platform and screen resolution.
    const QSize hint = minimumSizeHint();
    setMinimumSize(ecvAICoreUi::dpiScaled(hint.width()),
                   ecvAICoreUi::dpiScaled(hint.height()));
}

YOLODialog::~YOLODialog() {
    saveSettings();
    m_liveWidget->saveSettings();
}

QString YOLOTaskPanel::modelPath() const {
    const QString filename =
            modelCombo ? modelCombo->currentData().toString() : QString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = YOLOHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

QString YOLOTaskPanel::textModelPath() const {
    const QString filename = textModelCombo
                                     ? textModelCombo->currentData().toString()
                                     : QString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = YOLOHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

void YOLODialog::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(rootLayout);

    // Global runtime row: device / threads render once above the task
    // list — they configure every task panel and the Live widget alike.
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

    // Body: grouped task list on the left, one panel per task on the
    // right — the list/stack navigation keeps every task visible and
    // reachable without a tab-bar overflow. A splitter (not a fixed-width
    // list) lets users drag the pane wider for long entries or localized
    // texts; the default split is content-driven, see
    // applyTaskListDefaultWidth(), and the dragged state is persisted in
    // QSettings.
    m_bodySplitter = new QSplitter(Qt::Horizontal, this);
    m_bodySplitter->setChildrenCollapsible(false);
    m_taskList = new QListWidget(this);
    // Draggable lower bound — wide enough to stay usable (horizontal
    // scrolling kicks in below it), narrow enough to leave room for the
    // task panels on small screens.
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

    // ---- Per-task panels (each with its own model combo + thresholds) ---
    const QStringList taskOrder = kPanelTasks();
    const QStringList tabTitles = {tr("Object Detection"),
                                   tr("Instance Segmentation"),
                                   tr("Metric Depth"),
                                   tr("Pose (Keypoints)"),
                                   tr("Oriented Boxes"),
                                   tr("Classification"),
                                   tr("Semantic Segmentation"),
                                   tr("Open-Vocab Detect (World)"),
                                   tr("Open-Vocab Segment (YOLOE)")};

    for (int i = 0; i < taskOrder.size(); ++i) {
        YOLOTaskPanel panel;
        panel.task = taskOrder[i];
        panel.tab = new QWidget(this);
        auto* layout = new QVBoxLayout(panel.tab);
        ecvAICoreUi::setupTabLayout(layout);

        // Two-column body: config controls on the left, preview on the
        // right, so the dialog stays compact along both axes.
        auto* contentRow = new QHBoxLayout;
        contentRow->setSpacing(ecvAICoreUi::hSpacing());
        auto* configCol = new QVBoxLayout;
        configCol->setSpacing(ecvAICoreUi::vSpacing());

        // Model row: label + combo (filtered to this task's catalog).
        auto* modelRow = new QHBoxLayout;
        modelRow->setSpacing(ecvAICoreUi::hSpacing());
        modelRow->addWidget(ecvAICoreUi::makeLabel(tr("Model:")));
        panel.modelCombo = new QComboBox(panel.tab);
        panel.modelCombo->setMinimumContentsLength(16);
        panel.modelCombo->setSizeAdjustPolicy(
                QComboBox::AdjustToMinimumContentsLengthWithIcon);
        panel.modelCombo->setSizePolicy(QSizePolicy::Expanding,
                                        QSizePolicy::Fixed);
        modelRow->addWidget(panel.modelCombo, 1);
        configCol->addLayout(modelRow);

        // Threshold row: Conf / IoU / Top-K (hidden for metric-depth models,
        // which have no detection thresholds).
        panel.thresholdRow = new QWidget(panel.tab);
        auto* thresholdLayout = new QHBoxLayout(panel.thresholdRow);
        thresholdLayout->setContentsMargins(0, 0, 0, 0);
        thresholdLayout->setSpacing(ecvAICoreUi::tightHSpacing());
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("Conf:")));
        panel.conf = new QDoubleSpinBox(panel.tab);
        panel.conf->setRange(0.01, 1.0);
        panel.conf->setSingleStep(0.05);
        panel.conf->setValue(0.25);
        panel.conf->setToolTip(
                tr("Confidence threshold (detect/segment models)"));
        ecvAICoreUi::setCompactDoubleSpin(panel.conf);
        thresholdLayout->addWidget(panel.conf);
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("IoU:")));
        panel.iou = new QDoubleSpinBox(panel.tab);
        panel.iou->setRange(0.1, 1.0);
        panel.iou->setSingleStep(0.05);
        panel.iou->setValue(0.7);
        panel.iou->setToolTip(tr("NMS IoU threshold (detect/segment models)"));
        ecvAICoreUi::setCompactDoubleSpin(panel.iou);
        thresholdLayout->addWidget(panel.iou);
        thresholdLayout->addWidget(ecvAICoreUi::makeLabel(tr("Top-K:")));
        panel.topK = new QSpinBox(panel.tab);
        panel.topK->setRange(1, 1000);
        panel.topK->setValue(300);
        thresholdLayout->addWidget(panel.topK);
        thresholdLayout->addStretch();
        configCol->addWidget(panel.thresholdRow);

        // Open-vocabulary row (world/yoloe tabs): class list + text-model
        // combo, following the qSAM3 text-prompt interaction — the detector
        // and the text encoder are picked separately, and the text model
        // defaults to the family's tower (CLIP for World, MobileCLIP for
        // YOLOE). Classes are plain text, comma-separated.
        panel.textRow = new QWidget(panel.tab);
        auto* textLayout = new QVBoxLayout(panel.textRow);
        textLayout->setContentsMargins(0, 0, 0, 0);
        textLayout->setSpacing(ecvAICoreUi::tightHSpacing());
        auto* classesRow = new QHBoxLayout;
        classesRow->setSpacing(ecvAICoreUi::hSpacing());
        classesRow->addWidget(ecvAICoreUi::makeLabel(tr("Classes:")));
        panel.classesEdit = new QLineEdit(panel.textRow);
        panel.classesEdit->setPlaceholderText(tr("person, bus, car"));
        panel.classesEdit->setToolTip(
                tr("Comma-separated open-vocabulary class names (leave empty "
                   "to use the vocabulary stored in the checkpoint). Use "
                   "short category nouns — each entry becomes one category "
                   "vector. Descriptive phrases (e.g. \"female in yellow "
                   "hat\") score far lower and need Confidence ~0.02 or "
                   "below."));
        classesRow->addWidget(panel.classesEdit, 1);
        textLayout->addLayout(classesRow);
        auto* textModelRow = new QHBoxLayout;
        textModelRow->setSpacing(ecvAICoreUi::hSpacing());
        textModelRow->addWidget(ecvAICoreUi::makeLabel(tr("Text model:")));
        panel.textModelCombo = new QComboBox(panel.textRow);
        panel.textModelCombo->setMinimumContentsLength(16);
        panel.textModelCombo->setSizeAdjustPolicy(
                QComboBox::AdjustToMinimumContentsLengthWithIcon);
        panel.textModelCombo->setSizePolicy(QSizePolicy::Expanding,
                                            QSizePolicy::Fixed);
        for (const YOLOModelEntry& e : YOLOHelpers::textModels()) {
            // Family lock: the detector GGUF's text tower is fixed at
            // conversion time (YOLO-World embeddings live in CLIP space,
            // YOLOE in MobileCLIP2 space — docs.ultralytics.com; mixing
            // them fails at encode time). Only offer the matching tower.
            // The mclip bridge projects into the CLIP space, so it is
            // also offered on the World tab (multilingual prompts).
            const bool clip_ok =
                    panel.task == QStringLiteral("world") &&
                    e.filename.startsWith(QStringLiteral("clip-ViT-B-32"));
            const bool bridge_ok =
                    panel.task == QStringLiteral("world") &&
                    e.filename.startsWith(QStringLiteral("mclip-labse"));
            const bool mc_ok =
                    panel.task == QStringLiteral("yoloe") &&
                    e.filename.startsWith(QStringLiteral("mobileclip2_b"));
            if (clip_ok || bridge_ok || mc_ok) {
                panel.textModelCombo->addItem(YOLOHelpers::modelDisplayLabel(e),
                                              e.filename);
            }
        }
        selectDefaultTextModel(panel);
        textModelRow->addWidget(panel.textModelCombo, 1);
        textLayout->addLayout(textModelRow);
        // Confidence guidance while the multilingual bridge tower is
        // selected: bridged prompts score below native-English ones, so the
        // threshold has to come down for reliable detections.
        panel.bridgeHint =
                new QLabel(tr("Multilingual bridge — Conf auto-lowered to 0.03 "
                              "(bridged prompts score ~4x lower than English)"),
                           panel.textRow);
        panel.bridgeHint->setStyleSheet(
                QStringLiteral("color: palette(mid); font-size: 11px;"));
        panel.bridgeHint->setWordWrap(true);
        panel.bridgeHint->setVisible(false);
        textLayout->addWidget(panel.bridgeHint);
        connect(panel.textModelCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this](int) { updateBridgeHints(); });
        // The bridge hint also reflects the prompt language (English
        // prompts belong on the native tower); refresh it as the user
        // edits the class list. Recalibration is guarded by the tower-
        // change check inside updateBridgeHints, so typing never moves
        // the threshold.
        connect(panel.classesEdit, &QLineEdit::textChanged, this,
                [this](const QString&) { updateBridgeHints(); });
        configCol->addWidget(panel.textRow);

        // YOLOE prompt-mode row: text prompt (class list + MobileCLIP tower)
        // or visual prompt (draw example boxes; the SAVPE encoder derives
        // the class embeddings, official semantics object0..objectN-1).
        panel.promptModeRow = new QWidget(panel.tab);
        auto* modeLayout = new QVBoxLayout(panel.promptModeRow);
        modeLayout->setContentsMargins(0, 0, 0, 0);
        modeLayout->setSpacing(ecvAICoreUi::tightHSpacing());
        auto* modeRow = new QHBoxLayout;
        modeRow->setSpacing(ecvAICoreUi::hSpacing());
        modeRow->addWidget(ecvAICoreUi::makeLabel(tr("Prompt mode:")));
        panel.promptModeCombo = new QComboBox(panel.promptModeRow);
        panel.promptModeCombo->addItem(tr("Text prompt"));
        panel.promptModeCombo->addItem(tr("Visual prompt"));
        panel.promptModeCombo->setToolTip(
                tr("Text prompt: comma-separated class names encoded by the "
                   "text tower. Visual prompt: draw one example box per "
                   "target on the preview; the checkpoint's SAVPE encoder "
                   "derives the categories (results labeled object0, "
                   "object1, ...). Requires the non-prompt-free checkpoint; "
                   "prompt-free variants reject visual prompts."));
        modeRow->addWidget(panel.promptModeCombo, 1);
        modeLayout->addLayout(modeRow);
        configCol->addWidget(panel.promptModeRow);
        connect(panel.promptModeCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                [this](int) {
                    // Swap the text row / drawing canvas of the owning panel
                    // (sender mapping, like onModelComboChanged).
                    QComboBox* combo = qobject_cast<QComboBox*>(sender());
                    for (YOLOTaskPanel& p : m_panels) {
                        if (p.promptModeCombo == combo) {
                            applyPanelVisibility(p);
                            if (panelUsesVisualPrompts(p)) {
                                // Seed the canvas with the panel's current
                                // image so boxes can be drawn right away.
                                const QString path =
                                        p.imagePath
                                                ? p.imagePath->text().trimmed()
                                                : QString();
                                QImage img;
                                if (path.startsWith(QStringLiteral("db://"))) {
                                    const QString name = path.mid(5);
                                    for (int i = 0; i < p.dbImageList->count();
                                         ++i) {
                                        QListWidgetItem* item =
                                                p.dbImageList->item(i);
                                        if (item &&
                                            item->data(Qt::UserRole)
                                                            .toString() ==
                                                    name) {
                                            img = item->data(kDbFullImageRole)
                                                          .value<QImage>();
                                            break;
                                        }
                                    }
                                } else {
                                    img = QImage(path);
                                }
                                if (!img.isNull()) {
                                    p.vpLabel->setPromptImage(
                                            img,
                                            QSize(ecvAICoreUi::previewSize(),
                                                  ecvAICoreUi::previewSize()));
                                }
                            }
                            return;
                        }
                    }
                });

        // Custom GGUF row (shown only when a non-catalog file is picked).
        panel.customModelRow = new QWidget(panel.tab);
        auto* customRow = new QHBoxLayout(panel.customModelRow);
        customRow->setContentsMargins(0, 0, 0, 0);
        customRow->setSpacing(ecvAICoreUi::hSpacing());
        customRow->addWidget(ecvAICoreUi::makeLabel(tr("Custom GGUF:")));
        panel.customModelPath = new QLineEdit(panel.customModelRow);
        customRow->addWidget(panel.customModelPath, 1);
        auto* browseCustomBtn =
                ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.customModelRow);
        connect(browseCustomBtn, &QPushButton::clicked, this, [this]() {
            // The browse dialog stores into the ACTIVE panel's line edit.
            YOLOTaskPanel* active = currentTaskPanel();
            if (!active) return;
            m_customModelPath = active->customModelPath;
            m_customModelRow = active->customModelRow;
            onBrowseCustomModel();
        });
        customRow->addWidget(browseCustomBtn);
        configCol->addWidget(panel.customModelRow);

        // Input row: image path + browse.
        auto* inputRow = new QHBoxLayout;
        inputRow->setSpacing(ecvAICoreUi::hSpacing());
        inputRow->addWidget(ecvAICoreUi::makeLabel(tr("Image:")));
        panel.imagePath = new QLineEdit(panel.tab);
        inputRow->addWidget(panel.imagePath, 1);
        auto* browseBtn = ecvAICoreUi::makeBrowseBtn(tr("Browse…"), panel.tab);
        connect(browseBtn, &QPushButton::clicked, this, [this]() {
            YOLOTaskPanel* active = currentTaskPanel();
            if (!active) return;
            m_imagePath = active->imagePath;
            m_previewLabel = active->previewLabel;
            onBrowseImage();
        });
        inputRow->addWidget(browseBtn);
        configCol->addLayout(inputRow);

        // DB image picker (collapsible).
        panel.dbContentWidget = new QWidget(panel.tab);
        auto* dbLayout = new QVBoxLayout(panel.dbContentWidget);
        dbLayout->setContentsMargins(0, 0, 0, 0);
        dbLayout->setSpacing(ecvAICoreUi::vSpacing());
        panel.dbImageList = new QListWidget(panel.dbContentWidget);
        panel.dbImageList->setIconSize(QSize(48, 48));
        panel.dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
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
        panel.dbContentWidget->setVisible(false);
        panel.dbToggleBtn = ecvAICoreUi::makeDbSection(panel.dbContentWidget);
        ecvAICoreUi::connectDbToggle(panel.dbToggleBtn, panel.dbContentWidget);
        configCol->addWidget(panel.dbToggleBtn, 0, Qt::AlignLeft);
        configCol->addWidget(panel.dbContentWidget);
        configCol->addStretch();
        connect(panel.dbImageList, &QListWidget::itemActivated, this,
                &YOLODialog::onDbListActivated);
        connect(panel.dbImageList, &QListWidget::itemClicked, this,
                &YOLODialog::onDbListActivated);

        contentRow->addLayout(configCol, 1);

        // Right column: preview thumbnail (top-aligned). The thumbnail size
        // is DPI-aware via ecvAICoreUi::previewSize().
        auto* previewCol = new QVBoxLayout;
        previewCol->setSpacing(ecvAICoreUi::vSpacing());
        panel.previewLabel = new ecvClickableImageLabel(panel.tab);
        const int ps = ecvAICoreUi::previewSize();
        panel.previewLabel->setFixedSize(ps, ps);
        panel.previewLabel->setStyleSheet(
                "border: 1px solid palette(mid); background: palette(base);");
        panel.previewLabel->setText(tr("Preview"));
        previewCol->addWidget(panel.previewLabel);
        // Visual-prompt canvas (yoloe tab): same size as the preview,
        // swapped in while "Visual prompt" mode is active.
        panel.vpLabel = new YOLOVisualPromptLabel(panel.tab);
        panel.vpLabel->setFixedSize(ps, ps);
        panel.vpLabel->setStyleSheet(
                "border: 1px solid palette(mid); background: palette(base);");
        panel.vpLabel->setText(tr("Preview"));
        panel.vpLabel->setVisible(false);
        previewCol->addWidget(panel.vpLabel);
        auto* vpBtnRow = new QHBoxLayout;
        vpBtnRow->setSpacing(ecvAICoreUi::hSpacing());
        panel.vpClearBtn = new QPushButton(tr("Clear boxes"), panel.tab);
        panel.vpClearBtn->setToolTip(tr("Remove all drawn example boxes"));
        panel.vpClearBtn->setVisible(false);
        connect(panel.vpClearBtn, &QPushButton::clicked, panel.vpLabel,
                &YOLOVisualPromptLabel::clearBoxes);
        vpBtnRow->addWidget(panel.vpClearBtn);
        vpBtnRow->addStretch();
        previewCol->addLayout(vpBtnRow);
        auto* previewHint = new QLabel(tr("Tap to preview"), panel.tab);
        previewHint->setAlignment(Qt::AlignCenter);
        previewHint->setStyleSheet(
                QStringLiteral("color: palette(mid); font-size: 11px;"));
        previewCol->addWidget(previewHint);
        previewCol->addStretch();
        contentRow->addLayout(previewCol);

        layout->addLayout(contentRow);

        // Action row: add-to-DB + sample data + Run / Cancel.
        auto* actionRow = new QHBoxLayout;
        actionRow->setSpacing(ecvAICoreUi::hSpacing());
        panel.addAnnotatedCheck =
                new QCheckBox(tr("Add annotated image to DB"), panel.tab);
        panel.addAnnotatedCheck->setChecked(true);
        actionRow->addWidget(panel.addAnnotatedCheck);
        actionRow->addStretch();
        panel.testDataBtn = ecvAICoreUi::makeSampleDataBtn(panel.tab);
        // Per-task sample image: classification loads the single-subject
        // cat.jpg, OBB loads the DOTA-style aerial_airport.jpg, and every
        // other task loads the COCO street scene 000000397133.jpg.
        QStringList tip = {
                tr("Load this task's default sample image from the shared "
                   "test-data cache")};
        if (panel.task == QStringLiteral("obb")) {
            tip << tr(
                    "(aerial_airport.jpg — OBB models are trained on DOTA "
                    "aerial imagery)");
        } else if (panel.task == QStringLiteral("classify")) {
            tip << tr(
                    "(cat.jpg — a single-subject photo classifies more "
                    "meaningfully than a multi-object street scene)");
        } else if (panel.task == QStringLiteral("pose")) {
            tip << tr(
                    "(000000087038.jpg — multiple people in dynamic poses "
                    "for keypoint detection)");
        } else if (panel.task == QStringLiteral("world") ||
                   panel.task == QStringLiteral("yoloe")) {
            tip << tr(
                    "(party_hats.jpg — one differently colored party hat "
                    "per person; try prompts like \"adult with red hat\" to "
                    "see text-prompt selectivity)");
        } else {
            tip << tr("(000000397133.jpg — a multi-object street scene)");
        }
        panel.testDataBtn->setToolTip(tip.join(QChar(' ')));
        connect(panel.testDataBtn, &QPushButton::clicked, this,
                [this]() { requestTestData(TestDataTarget::Image); });
        actionRow->addWidget(panel.testDataBtn);
        panel.runBtn = new QPushButton(tr("Run"), panel.tab);
        panel.runBtn->setDefault(i == 0);
        actionRow->addWidget(panel.runBtn);
        panel.cancelBtn = new QPushButton(tr("Cancel"), panel.tab);
        panel.cancelBtn->setEnabled(false);
        actionRow->addWidget(panel.cancelBtn);
        layout->addLayout(actionRow);

        connect(panel.modelCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                &YOLODialog::onModelComboChanged);
        connect(panel.runBtn, &QPushButton::clicked, this, &YOLODialog::onRun);
        connect(panel.cancelBtn, &QPushButton::clicked, this,
                &YOLODialog::onCancel);

        m_panels.append(panel);
        m_taskStack->addWidget(panel.tab);
    }

    // ---- Live (camera / video) tab ----------------------------------------
    m_liveTab = new QWidget(this);
    auto* liveLayout = new QVBoxLayout(m_liveTab);
    ecvAICoreUi::setupTabLayout(liveLayout);
    m_liveWidget = new YOLOLiveWidget(m_liveTab);
    liveLayout->addWidget(m_liveWidget, 1);

    // Playback controls live in the Live tab itself (mirrors qFaceDetect).
    auto* liveBtnRow = new QHBoxLayout;
    liveBtnRow->setSpacing(ecvAICoreUi::hSpacing());
    m_testVideoCombo = new QComboBox(m_liveTab);
    m_testVideoCombo->addItem(QStringLiteral("traffic.mp4"),
                              QStringLiteral("traffic.mp4"));
    m_testVideoCombo->addItem(QStringLiteral("supervision_demo.mp4"),
                              QStringLiteral("supervision_demo.mp4"));
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(m_liveTab);
    m_testDataBtn->setToolTip(
            tr("Load the selected video from the shared test-data cache"));
    m_liveStartBtn = new QPushButton(tr("Start"), m_liveTab);
    m_liveStopBtn = new QPushButton(tr("Stop"), m_liveTab);
    m_liveRestartBtn = new QPushButton(tr("Restart"), m_liveTab);
    m_liveStopBtn->setEnabled(false);
    m_liveRestartBtn->setEnabled(false);
    liveBtnRow->addWidget(m_testVideoCombo);
    liveBtnRow->addWidget(m_testDataBtn);
    liveBtnRow->addWidget(m_liveStartBtn);
    liveBtnRow->addWidget(m_liveStopBtn);
    liveBtnRow->addWidget(m_liveRestartBtn);
    liveBtnRow->addStretch();
    liveLayout->addLayout(liveBtnRow);

    m_taskStack->addWidget(m_liveTab);

    connect(m_liveStartBtn, &QPushButton::clicked, this,
            &YOLODialog::onLiveStart);
    connect(m_liveStopBtn, &QPushButton::clicked, this,
            &YOLODialog::onLiveStop);
    connect(m_liveRestartBtn, &QPushButton::clicked, this,
            &YOLODialog::onLiveRestart);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            [this]() { requestTestData(TestDataTarget::Video); });

    // Keep the live button states in sync with the stream lifecycle.
    connect(m_liveWidget, &YOLOLiveWidget::streamStarted, this, [this]() {
        m_liveStartBtn->setEnabled(false);
        m_liveStopBtn->setEnabled(true);
        m_liveRestartBtn->setEnabled(m_liveWidget->inputSource() ==
                                     YOLOLiveWidget::InputSource::VideoFile);
    });
    connect(m_liveWidget, &YOLOLiveWidget::streamStopped, this, [this]() {
        m_liveStartBtn->setEnabled(true);
        m_liveStopBtn->setEnabled(false);
        if (m_liveWidget->inputSource() !=
            YOLOLiveWidget::InputSource::VideoFile) {
            m_liveRestartBtn->setEnabled(false);
        }
    });

    // The Live page lists ALL catalog models (any task); device / threads
    // live in the global runtime row above the task list (one shared
    // instance — no cross-panel sync needed).
    m_liveWidget->populateAllModels();
    m_liveWidget->rebuildDeviceCombo(m_deviceCombo);
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                m_liveWidget->setDevice(
                        m_deviceCombo->currentData().toString());
            });
    connect(m_threads, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int v) { m_liveWidget->setThreads(v); });
    connect(m_liveWidget, &YOLOLiveWidget::modelSelectionChanged, this,
            [this](const QString& filename) {
                // Keep the matching batch tab's model in sync so the two
                // surfaces don't drift, but do NOT force a tab switch.
                // The mirror is programmatic: block signals so it is not
                // recorded as an explicit user model choice — that used to
                // pin the tab to the mirrored model across restarts.
                YOLOTaskPanel* panel = panelForFilename(filename);
                if (panel && panel->modelCombo) {
                    panel->modelCombo->blockSignals(true);
                    const int idx = panel->modelCombo->findData(filename);
                    if (idx >= 0) panel->modelCombo->setCurrentIndex(idx);
                    panel->modelCombo->blockSignals(false);
                    applyPanelVisibility(*panel);
                }
            });
    connect(m_liveWidget, &YOLOLiveWidget::deviceSelectionChanged, this,
            [this](const QString& device) {
                const int index = m_deviceCombo->findData(device);
                if (index >= 0) m_deviceCombo->setCurrentIndex(index);
            });
    connect(m_liveWidget, &YOLOLiveWidget::threadCountChanged, this,
            [this](int threads) {
                if (m_threads->value() != threads) m_threads->setValue(threads);
            });
    connect(m_liveWidget, &YOLOLiveWidget::captureToDbRequested, this,
            &YOLODialog::onLiveCapture);

    // ---- Task list entries ------------------------------------------------
    // Grouped navigation (section headers are disabled items): closed-set
    // tasks, open-vocabulary families, then the capture page.
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
    addSectionHeader(tr("Closed-set tasks"));
    bool openVocabHeaderAdded = false;
    for (int i = 0; i < taskOrder.size(); ++i) {
        if (taskHasText(taskOrder[i]) && !openVocabHeaderAdded) {
            addSectionHeader(tr("Open-vocabulary"));
            openVocabHeaderAdded = true;
        }
        addTaskItem(tabTitles[i], i);
    }
    addSectionHeader(tr("Capture"));
    addTaskItem(tr("Live (camera / video)"), m_taskStack->count() - 1);

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
    connect(m_liveWidget, &YOLOLiveWidget::depthCaptureToDbRequested, this,
            &YOLODialog::onLiveDepthCapture);

    // Download / task progress — shared by all tabs so a model fetch started
    // from any tab stays visible.
    ecvAICoreUi::setupProgressSection(rootLayout, m_downloadLabel, m_progress);

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
            &YOLODialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[YOLO] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[YOLO] Model downloaded: %1").arg(path));
                if (m_pendingActionAfterDownload != PendingAction::None) {
                    const PendingAction action = m_pendingActionAfterDownload;
                    m_pendingActionAfterDownload = PendingAction::None;
                    if (action == PendingAction::Run) {
                        onRun();
                    } else if (action == PendingAction::LiveStart) {
                        startLiveStream();
                    }
                }
            });

    m_taskStatusLabel = new QLabel(this);
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    rootLayout->addWidget(m_taskStatusLabel);

    // Shared test data repository.
    auto& repo = ecvTestDataRepository::instance();
    connect(&repo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataDownloadInProgress) return;
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_progress->setVisible(true);
                m_downloadLabel->setText(statusText);
                m_downloadLabel->setVisible(true);
            });
    connect(&repo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataDownloadInProgress) appendLog(message);
            });
    connect(&repo, &ecvTestDataRepository::downloadFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataDownloadFinished(success, kind);
            });
    connect(&repo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                m_progress->setRange(0, total);
                m_progress->setValue(current);
                m_progress->setVisible(true);
            });
    connect(&repo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataExtractionFinished(success, kind);
            });
}

void YOLODialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void YOLODialog::selectDefaultTextModel(YOLOTaskPanel& panel) const {
    if (!panel.textModelCombo) return;
    // Keep a valid explicit selection; only fill the default when nothing
    // (or a non-catalog entry) is selected.
    if (!panel.textModelCombo->currentData().toString().isEmpty()) return;
    const QString preferred = defaultTextModelForTask(panel.task);
    const int idx = panel.textModelCombo->findData(preferred);
    if (idx >= 0) panel.textModelCombo->setCurrentIndex(idx);
}

void YOLODialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO"));
    const QStringList tasks = kPanelTasks();
    for (int i = 0; i < m_panels.size() && i < tasks.size(); ++i) {
        YOLOTaskPanel& panel = m_panels[i];
        const QString modelFilename =
                settings.value(QStringLiteral("modelFilename/") + tasks[i])
                        .toString();
        // Legacy builds auto-persisted the index-0 (F32 reference) default
        // on close; only restore an explicit user choice and otherwise keep
        // the recommended entry picked by populateModelCombo().
        // One-shot cleanup: the Live widget used to emit its auto-populated
        // row-0 model on dialog construction and the sync path recorded it
        // as an explicit choice, pinning the detect/segment/depth tabs to
        // the F32 reference builds. Treat exactly those stale entries as
        // non-explicit so the recommended default applies again.
        static const QSet<QString> kStaleMirrorModels = {
                QStringLiteral("yolov8n-f32.gguf"),
                QStringLiteral("yolov8n-seg-f32.gguf"),
                QStringLiteral("yolo26n-depth-f32.gguf")};
        panel.explicitModelChoice =
                settings.value(QStringLiteral("modelFilenameExplicit/") +
                                       tasks[i],
                               false)
                        .toBool() &&
                !kStaleMirrorModels.contains(modelFilename);
        // Restore the bridge state BEFORE the text tower so that a conf
        // left in the bridge's low band (e.g. 0.02) is corrected on the
        // tower-restored updateBridgeHints() call even after a restart.
        panel.bridgeWasActive =
                settings.value(QStringLiteral("bridgeActive/") + tasks[i],
                               false)
                        .toBool();
        if (panel.explicitModelChoice && !modelFilename.isEmpty()) {
            const int idx = panel.modelCombo->findData(modelFilename);
            if (idx >= 0) panel.modelCombo->setCurrentIndex(idx);
        }
        panel.conf->setValue(
                settings.value(QStringLiteral("conf/") + tasks[i], 0.25)
                        .toDouble());
        panel.iou->setValue(
                settings.value(QStringLiteral("iou/") + tasks[i], 0.7)
                        .toDouble());
        panel.topK->setValue(
                settings.value(QStringLiteral("topK/") + tasks[i], 300)
                        .toInt());
        panel.addAnnotatedCheck->setChecked(
                settings.value(QStringLiteral("addAnnotated/") + tasks[i], true)
                        .toBool());
        const QString imagePath =
                settings.value(QStringLiteral("imagePath/") + tasks[i])
                        .toString();
        if (!imagePath.isEmpty()) {
            panel.imagePath->setText(imagePath);
            m_imagePath = panel.imagePath;
            m_previewLabel = panel.previewLabel;
            updateImagePreview();
        }
        if (panel.classesEdit) {
            panel.classesEdit->setText(
                    settings.value(QStringLiteral("classes/") + tasks[i])
                            .toString());
        }
        if (panel.textModelCombo) {
            const QString textModelFilename =
                    settings.value(QStringLiteral("textModel/") + tasks[i])
                            .toString();
            if (!textModelFilename.isEmpty()) {
                const int idx =
                        panel.textModelCombo->findData(textModelFilename);
                if (idx >= 0) panel.textModelCombo->setCurrentIndex(idx);
            }
            selectDefaultTextModel(panel);
        }
    }
    // Device/threads are global controls rendered once above the task
    // list.
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int threads = settings.value(QStringLiteral("threads"), 0).toInt();
    if (!device.isEmpty()) {
        const int idx = m_deviceCombo->findData(device);
        if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    }
    m_threads->setValue(threads);
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
    // Recalibrate every panel's confidence to its restored text tower's
    // score band. The tower combo may not emit when the restored entry
    // equals the populated default, so a conf left over from another tower
    // (e.g. 0.05 on a native tower) would otherwise survive the restart.
    updateBridgeHints(/*force*/ true);
}

void YOLODialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO"));
    const QStringList tasks = kPanelTasks();
    for (int i = 0; i < m_panels.size() && i < tasks.size(); ++i) {
        const YOLOTaskPanel& panel = m_panels[i];
        settings.setValue(QStringLiteral("modelFilename/") + tasks[i],
                          panel.modelCombo->currentData().toString());
        settings.setValue(QStringLiteral("modelFilenameExplicit/") + tasks[i],
                          panel.explicitModelChoice);
        settings.setValue(QStringLiteral("bridgeActive/") + tasks[i],
                          panel.bridgeWasActive);
        settings.setValue(QStringLiteral("conf/") + tasks[i],
                          panel.conf->value());
        settings.setValue(QStringLiteral("iou/") + tasks[i],
                          panel.iou->value());
        settings.setValue(QStringLiteral("topK/") + tasks[i],
                          panel.topK->value());
        settings.setValue(QStringLiteral("addAnnotated/") + tasks[i],
                          panel.addAnnotatedCheck->isChecked());
        settings.setValue(QStringLiteral("imagePath/") + tasks[i],
                          panel.imagePath->text());
        if (panel.classesEdit) {
            settings.setValue(QStringLiteral("classes/") + tasks[i],
                              panel.classesEdit->text());
        }
        if (panel.textModelCombo) {
            settings.setValue(QStringLiteral("textModel/") + tasks[i],
                              panel.textModelCombo->currentData().toString());
        }
    }
    // Device/threads are global controls; persist them directly.
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    // Splitter geometry: persist the user's last left/right drag.
    if (m_bodySplitter) {
        settings.setValue(QStringLiteral("bodySplitterState"),
                          m_bodySplitter->saveState());
    }
    settings.endGroup();
}

int YOLODialog::taskListIdealWidth() const {
    if (!m_taskList) return 0;
    // Content-driven, font/DPI aware: measure the widest entry text
    // (section headers render bold) instead of hard-coding pixels, so the
    // default fits every translation, platform style and screen
    // resolution without eliding entries.
    const QFont base = m_taskList->font();
    const QFontMetrics fm(base);
    QFont boldFont = base;
    boldFont.setBold(true);
    const QFontMetrics fmBold(boldFont);
    int textWidth = 0;
    for (int i = 0; i < m_taskList->count(); ++i) {
        const QListWidgetItem* item = m_taskList->item(i);
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

void YOLODialog::applyTaskListDefaultWidth() {
    if (!m_bodySplitter || !m_taskList || m_taskList->count() == 0) return;
    const int ideal = taskListIdealWidth();
    const int total = m_bodySplitter->width();
    if (total <= 0) return;
    // Left pane = content width, right pane = what remains (the splitter
    // clamps both to the panes' minimum size hints on tiny screens).
    m_bodySplitter->setSizes({ideal, qMax(total - ideal, 1)});
}

void YOLODialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    // First show: the splitter only knows its final size after layout, so
    // give the task list its content-driven width here. A splitter state
    // restored from QSettings (the user's last drag) wins over the
    // default.
    if (!m_splitterSized) {
        m_splitterSized = true;
        if (!m_splitterRestored) applyTaskListDefaultWidth();
    }
}

QString YOLODialog::modelCacheDir() { return YOLOHelpers::modelCacheDir(); }

void YOLODialog::populateModelCombo(const QString& keepFilename) {
    // Each task panel lists only its own task's catalog models.
    const QStringList tasks = kPanelTasks();
    for (int i = 0; i < m_panels.size() && i < tasks.size(); ++i) {
        YOLOTaskPanel& panel = m_panels[i];
        const QVector<YOLOModelEntry> models =
                YOLOHelpers::taskModels(tasks[i]);
        panel.modelCombo->blockSignals(true);
        panel.modelCombo->clear();
        for (const YOLOModelEntry& e : models) {
            panel.modelCombo->addItem(YOLOHelpers::modelDisplayLabel(e),
                                      e.filename);
        }
        // Single selection policy for every AICore dialog: keep the
        // caller's selection when valid, else the catalog-declared default
        // row for this task's role view.
        ecvAICoreUi::selectModelRow(
                panel.modelCombo, keepFilename,
                YOLOHelpers::defaultModelIndexForTask(tasks[i]));
        panel.modelCombo->blockSignals(false);
        // Signals were blocked above, so the currentIndexChanged handler
        // would not run — apply the visibility directly.
        applyPanelVisibility(panel);
    }
    if (m_liveWidget) {
        // Keep the Live tab's all-model list fresh too (it may be open).
        m_liveWidget->populateAllModels(keepFilename);
    }
}

void YOLODialog::refreshModelList() {
    const QString keep =
            currentTaskPanel() && currentTaskPanel()->modelCombo
                    ? currentTaskPanel()->modelCombo->currentData().toString()
                    : QString();
    populateModelCombo(keep);
}

YOLOTaskPanel* YOLODialog::currentTaskPanel() const {
    if (!m_taskStack) return nullptr;
    QWidget* page = m_taskStack->currentWidget();
    for (const YOLOTaskPanel& panel : m_panels) {
        if (panel.tab == page) {
            // const_cast: callers expect a mutable panel (they set controls).
            return const_cast<YOLOTaskPanel*>(&panel);
        }
    }
    return nullptr;
}

bool YOLODialog::panelUsesVisualPrompts(const YOLOTaskPanel& panel) {
    return panel.task == QStringLiteral("yoloe") && panel.promptModeCombo &&
           panel.promptModeCombo->currentIndex() == 1;
}

YOLOTaskPanel* YOLODialog::panelForTask(const QString& task) const {
    for (const YOLOTaskPanel& panel : m_panels) {
        if (panel.task == task) {
            return const_cast<YOLOTaskPanel*>(&panel);
        }
    }
    return nullptr;
}

YOLOTaskPanel* YOLODialog::panelForFilename(const QString& filename) const {
    for (const YOLOTaskPanel& panel : m_panels) {
        if (panel.modelCombo->findData(filename) >= 0) {
            return const_cast<YOLOTaskPanel*>(&panel);
        }
    }
    return nullptr;
}

void YOLODialog::onModelComboChanged(int /*index*/) {
    // The sender is one of the task panels' model combos; map back to the
    // owning panel by the signal origin. When invoked programmatically
    // (sender() == nullptr, e.g. from populateModelCombo) the caller uses
    // applyPanelVisibility() directly instead.
    QComboBox* combo = qobject_cast<QComboBox*>(sender());
    for (YOLOTaskPanel& p : m_panels) {
        if (p.modelCombo == combo) {
            // A match implies a real signal (programmatic callers pass a
            // null sender and never match) — the user picked explicitly.
            p.explicitModelChoice = true;
            applyPanelVisibility(p);
            return;
        }
    }
}

void YOLODialog::applyPanelVisibility(YOLOTaskPanel& panel) {
    const QString filename = panel.modelCombo->currentData().toString();
    const bool isCustom =
            filename.isEmpty() ||
            filename.endsWith(QStringLiteral(".gguf")) &&
                    !YOLOHelpers::findModelByFilename(filename, nullptr);
    panel.customModelRow->setVisible(isCustom);

    // YOLOE prompt-mode gating: the mode row exists on every panel (shared
    // construction) but only the yoloe tab shows it, and it swaps the text
    // row for the box-drawing canvas while "Visual prompt" is active.
    const bool visual = panelUsesVisualPrompts(panel);
    if (panel.promptModeRow) {
        panel.promptModeRow->setVisible(panel.task == QStringLiteral("yoloe"));
    }
    if (panel.vpLabel) {
        panel.vpLabel->setVisible(visual);
        panel.vpLabel->setDrawingEnabled(visual);
        panel.previewLabel->setVisible(!visual);
        if (panel.vpClearBtn) panel.vpClearBtn->setVisible(visual);
    }

    // Threshold row visible for the box/conf-driven tasks (detect, segment,
    // pose, obb, world, yoloe); depth/classify/semantic have none.
    panel.thresholdRow->setVisible(taskHasThresholds(panel.task));
    // Text row visible only for the open-vocabulary families in TEXT mode;
    // keep the family-default text tower selected.
    if (panel.textRow) {
        panel.textRow->setVisible(taskHasText(panel.task) && !visual);
        if (taskHasText(panel.task)) selectDefaultTextModel(panel);
        // Prompt-free YOLOE checkpoints match regions against the built-in
        // 4585-entry vocabulary (LRPC) and REJECT set_classes outright
        // (docs.ultralytics.com AssertionError) — disable the class list
        // and the text tower for them. Non-prompt-free YOLOE checkpoints
        // ship no stored vocabulary at all, so their class list is
        // REQUIRED (see ensureModelAvailable).
        const QString sel = panel.modelCombo->currentData().toString();
        const bool prompt_free = YOLOHelpers::isPromptFreeFilename(sel);
        const bool yoloe_needs_classes =
                panel.task == QStringLiteral("yoloe") && !prompt_free;
        if (panel.classesEdit) {
            panel.classesEdit->setEnabled(!prompt_free);
            panel.classesEdit->setToolTip(
                    prompt_free
                            ? tr("Prompt-free checkpoints use the built-in "
                                 "vocabulary and do not accept a class list")
                    : yoloe_needs_classes
                            ? tr("Required for this checkpoint: it ships no "
                                 "stored vocabulary — enter comma-separated "
                                 "class names (an empty list is rejected at "
                                 "run time). Use short category nouns; "
                                 "descriptive phrases score far lower and "
                                 "need Confidence ~0.02 or below")
                            : tr("Comma-separated open-vocabulary class names "
                                 "(leave empty to use the vocabulary stored "
                                 "in the checkpoint). Use short category "
                                 "nouns — descriptive phrases score far "
                                 "lower and need Confidence ~0.02 or "
                                 "below"));
        }
        if (panel.textModelCombo) {
            panel.textModelCombo->setEnabled(!prompt_free);
        }
    }
    updateBridgeHints();
}

void YOLODialog::updateBridgeHints(bool force) {
    for (YOLOTaskPanel& p : m_panels) {
        if (!p.bridgeHint || !p.textModelCombo) continue;
        const QString tower = p.textModelCombo->currentData().toString();
        const bool mclip = tower.startsWith(QStringLiteral("mclip-labse"));
        // Conf recalibrations only fire on a real tower change (or a forced
        // refresh after settings restore): applyPanelVisibility() runs on
        // every model-combo change and must never touch the threshold.
        const bool towerChanged = force || tower != p.lastTextTower;
        p.lastTextTower = tower;
        const bool visible = mclip && p.textRow && p.textRow->isVisible();
        p.bridgeHint->setVisible(visible);
        if (!p.conf) continue;
        if (!mclip) {
            p.conf->setToolTip(
                    tr("Confidence threshold (detect/segment models)"));
            // Symmetric restore: leaving the multilingual bridge brings the
            // threshold back out of the bridge's low band (0.02-0.05),
            // where a native tower floods the scene with low-score false
            // positives. The yoloe phrase workflow (Conf ~0.02) never sees
            // this branch: its tower is mobileclip2 and bridgeWasActive
            // stays false there.
            if (towerChanged && p.bridgeWasActive && p.conf->value() < 0.10) {
                p.conf->setValue(0.25);
                appendLog(
                        tr("[YOLO] Native text tower selected — "
                           "confidence restored to 0.25"));
            }
            if (towerChanged) p.bridgeWasActive = false;
            continue;
        }
        // Bridged multilingual prompts score ~4x lower than native-English
        // ones (measured on the party-hats scene: ZH max 0.046 vs EN 0.186),
        // so the closed-set default of 0.25 silently hides every detection.
        // Recalibrate on EVERY tower change: a leftover value from another
        // tower (e.g. 0.05-0.10) sits above the bridge's band and silently
        // hides detections. The user can still tune it afterwards; the next
        // tower switch recalibrates again (predictable, no stale state).
        if (towerChanged) {
            p.conf->setValue(0.03);
            appendLog(
                    tr("[YOLO] Multilingual bridge selected — confidence "
                       "set to 0.03 (bridged prompts score ~4x lower "
                       "than native English)"));
        }
        p.bridgeWasActive = true;
        p.conf->setToolTip(
                tr("Bridged multilingual prompts score ~4x lower than "
                   "native-English ones — keep Confidence low (0.02-0.05)"));
        // The bridge exists for the 100+ non-English languages; pure-ASCII
        // prompts belong on the native CLIP tower, whose score band is ~3x
        // higher (party-hats scene, same prompt: 0.166 vs 0.053) and whose
        // ranking discriminates color attributes better.
        if (p.classesEdit) {
            const QString text = p.classesEdit->text();
            bool asciiOnly = !text.isEmpty();
            for (const QChar& c : text) {
                if (c.unicode() > 0x7F) {
                    asciiOnly = false;
                    break;
                }
            }
            p.bridgeHint->setText(
                    asciiOnly ? tr("Multilingual bridge — Confidence auto-set "
                                   "to 0.03. English prompts score ~3x higher "
                                   "on the native CLIP tower — switch unless "
                                   "you need multilingual input")
                              : tr("Multilingual bridge — Confidence auto-set "
                                   "to 0.03 (bridged prompts score ~4x lower "
                                   "than native English)"));
        }
    }
}

bool YOLODialog::ensureModelAvailable(PendingAction action) {
    // The Live page has its own model combo (all catalog models) and is
    // not one of the task panels. Resolve against the live widget's model
    // when it is the active page — checking the batch panel here would
    // silently no-op (currentTaskPanel() == nullptr) and Start would do
    // nothing.
    if (m_taskStack && m_taskStack->currentWidget() == m_liveTab) {
        if (!m_liveWidget) return false;
        const QString filename = m_liveWidget->modelFilename();
        if (filename.isEmpty()) {
            appendLog(tr("[Error] Select a model first."));
            return false;
        }
        if (!QFileInfo::exists(m_liveWidget->resolveModelPath())) {
            YOLOModelEntry entry;
            if (!YOLOHelpers::findModelByFilename(filename, &entry)) {
                appendLog(tr("[Error] Model file not found: %1").arg(filename));
                return false;
            }
            m_pendingActionAfterDownload = action;
            appendLog(tr("[YOLO] Model missing — downloading %1; it will "
                         "start automatically when ready.")
                              .arg(filename));
            startDownload(entry);
            return false;
        }
        return true;
    }

    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return false;
    const QString filename = panel->modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[Error] Select a model first."));
        return false;
    }
    // YOLOE visual-prompt mode: SAVPE box prompts replace the class-list
    // requirement entirely (official semantics). Two guards: prompt-free
    // checkpoints REJECT visual prompts (upstream AssertionError), and the
    // canvas needs at least one example box.
    if (panel->task == QStringLiteral("yoloe") &&
        panelUsesVisualPrompts(*panel)) {
        if (YOLOHelpers::isPromptFreeFilename(filename)) {
            appendLog(
                    tr("[Error] Prompt-free YOLOE checkpoints reject visual "
                       "prompts — pick the non-prompt-free variant of the "
                       "same scale, or switch back to Text prompt mode."));
            return false;
        }
        if (!panel->vpLabel || panel->vpLabel->boxCount() == 0) {
            appendLog(
                    tr("[Error] Visual prompt mode: draw at least one "
                       "example box on the preview first."));
            return false;
        }
        return true;
    }
    // Non-prompt-free YOLOE checkpoints ship NO stored vocabulary: the
    // load rejects them with "no stored vocabulary for N classes" unless
    // a class list + text tower encode one. Fail BEFORE the model
    // download (the catalog GGUFs are 100 MB+) with an actionable hint;
    // prompt-free variants and custom GGUFs are exempt (a custom
    // checkpoint may carry txt_feats, and the backend reports it).
    if (panel->task == QStringLiteral("yoloe") &&
        !YOLOHelpers::isPromptFreeFilename(filename) &&
        YOLOHelpers::findModelByFilename(filename, nullptr)) {
        bool hasClasses = false;
        const QStringList rawClasses =
                panel->classesEdit
                        ? panel->classesEdit->text().split(QLatin1Char(','))
                        : QStringList();
        for (const QString& c : rawClasses) {
            if (!c.trimmed().isEmpty()) {
                hasClasses = true;
                break;
            }
        }
        if (!hasClasses) {
            // Official-parity no-input route: the upstream *-seg.pt reports
            // nc=80 numeric placeholder names without set_classes (its
            // zero-embedding fallback), so the sanctioned promptless route
            // is the same-scale -pf checkpoint (built-in 4585-entry LRPC
            // vocabulary). Do NOT auto-switch to it: silently re-pointing
            // the combo away from an explicit user pick made the model jump
            // back to prompt-free with no on-screen hint (the switch notice
            // only reached the internal log), and the -pf vocabulary yields
            // entirely different results than the text/visual prompts the
            // user intended. Fail with an actionable hint instead — picking
            // the -pf checkpoint stays the user's call.
            appendLog(
                    tr("[Error] This YOLOE checkpoint ships no stored "
                       "vocabulary and no class list was entered — enter "
                       "comma-separated class names (e.g. person, bus, "
                       "car), draw visual prompts (Visual prompt mode), or "
                       "select the prompt-free (-pf) checkpoint of the "
                       "same scale."));
            return false;
        }
    }
    if (!QFileInfo::exists(panel->modelPath())) {
        YOLOModelEntry entry;
        if (!YOLOHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[Error] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[YOLO] Model missing — downloading %1; it will "
                     "start automatically when ready.")
                          .arg(filename));
        startDownload(entry);
        return false;
    }
    // Open-vocabulary tabs: with a non-empty class list the text-encoder
    // GGUF is a SECOND required model — run the same availability +
    // auto-download + pending-action-rerun chain as the detector (if both
    // files are missing the two downloads chain through the pending
    // action: detector first, text on the automatic re-run).
    if (taskHasText(panel->task) && panel->classesEdit &&
        panel->textModelCombo) {
        bool hasClasses = false;
        const QStringList raw =
                panel->classesEdit->text().split(QLatin1Char(','));
        for (const QString& c : raw) {
            if (!c.trimmed().isEmpty()) {
                hasClasses = true;
                break;
            }
        }
        if (hasClasses) {
            const QString textFilename =
                    panel->textModelCombo->currentData().toString();
            if (textFilename.isEmpty()) {
                appendLog(tr("[YOLO] Select a text model first."));
                return false;
            }
            if (!QFileInfo::exists(panel->textModelPath())) {
                YOLOModelEntry entry;
                if (!YOLOHelpers::findModelByFilename(textFilename, &entry)) {
                    appendLog(tr("[YOLO] Text model file not found: %1")
                                      .arg(textFilename));
                    return false;
                }
                m_pendingActionAfterDownload = action;
                appendLog(tr("[YOLO] Text model missing — downloading %1; it "
                             "will start automatically when ready.")
                                  .arg(textFilename));
                startDownload(entry);
                return false;
            }
        }
    }
    return true;
}

void YOLODialog::startDownload(const YOLOModelEntry& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[YOLO] A download is already running."));
        return;
    }
    QDir().mkpath(YOLOHelpers::modelCacheDir());
    const QString dest =
            YOLOHelpers::modelCacheDir() + QDir::separator() + model.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[YOLO] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[YOLO] Downloading %1 (%2)…")
                      .arg(model.filename, model.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // YOLO GGUFs are tens of MB
    // Content identity from the release digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught).
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(model.filename)};
    m_downloader->download(req);
}

void YOLODialog::cancelDownload() {
    if (m_downloadInProgress) m_downloader->cancel();
}

void YOLODialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qYOLO"),
                                             QStringLiteral("lastModelDir"),
                                             YOLOHelpers::modelCacheDir());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select YOLO GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return;
    if (m_customModelPath) m_customModelPath->setText(path);
    if (m_customModelRow) m_customModelRow->setVisible(true);
    panel->modelCombo->setCurrentIndex(-1);
    panel->modelCombo->addItem(QFileInfo(path).fileName(), path);
    panel->modelCombo->setCurrentIndex(panel->modelCombo->count() - 1);
    m_liveWidget->setModelPath(path);
}

void YOLODialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qYOLO"),
                                             QStringLiteral("lastImageFileDir"),
                                             QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    if (m_imagePath) m_imagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qYOLO"),
                         QStringLiteral("lastImageFileDir"), path);
    updateImagePreview();
}

void YOLODialog::updateImagePreview() {
    if (!m_imagePath || !m_previewLabel) return;
    const QString path = m_imagePath->text().trimmed();
    QImage img;
    if (path.startsWith(QStringLiteral("db://"))) {
        // DB-tree entity: look up the stored full-resolution image so the
        // click-to-enlarge preview works for DB inputs too.
        const QString name = path.mid(5);
        if (YOLOTaskPanel* panel = currentTaskPanel()) {
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
        m_previewLabel->setText(tr("Preview"));
        if (YOLOTaskPanel* panel = currentTaskPanel();
            panel && panel->vpLabel) {
            panel->vpLabel->clearPrompt();
            panel->vpLabel->setText(tr("Preview"));
        }
        return;
    }
    m_previewLabel->setPreviewImage(
            img, QSize(ecvAICoreUi::previewSize(), ecvAICoreUi::previewSize()));
    // Keep the visual-prompt canvas in sync (yoloe tab): it shows the same
    // image so example boxes can be drawn without reloading anything.
    if (YOLOTaskPanel* panel = currentTaskPanel(); panel && panel->vpLabel) {
        panel->vpLabel->setPromptImage(img, QSize(ecvAICoreUi::previewSize(),
                                                  ecvAICoreUi::previewSize()));
    }
}

void YOLODialog::onRun() {
    // Validate the input BEFORE the model chain: a missing image would
    // otherwise trigger the model download first and then fail with a
    // confusing "Input file not found: <empty>" after the wait.
    if (currentTaskPanel() &&
        currentTaskPanel()->imagePath->text().trimmed().isEmpty()) {
        appendLog(
                tr("[Error] Select an image first — pick a file, a DB "
                   "image, or click Use test data."));
        return;
    }
    if (!ensureModelAvailable(PendingAction::Run)) return;
    emit runRequested(getSettings());
}

void YOLODialog::onCancel() {
    cancelDownload();
    emit cancelRequested();
}

YOLODialog::Settings YOLODialog::getSettings() {
    Settings s;
    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return s;
    s.modelPath = panel->modelPath();
    s.inputPath = panel->imagePath->text();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    s.confThres = static_cast<float>(panel->conf->value());
    s.iouThres = static_cast<float>(panel->iou->value());
    s.topK = static_cast<uint32_t>(panel->topK->value());
    s.addAnnotatedImageToDb = panel->addAnnotatedCheck->isChecked();
    // Prompt-free YOLOE checkpoints reject set_classes outright (upstream
    // AssertionError) and match their built-in 4585-entry LRPC vocabulary:
    // the disabled edit's leftover text must not reach the backend, where
    // it would replace the stored class-name table and degrade every
    // label to "class <id>".
    const bool promptFree = YOLOHelpers::isPromptFreeFilename(
            QFileInfo(s.modelPath).fileName());
    // YOLOE visual-prompt mode: the drawn boxes become the categories
    // (object0..objectN-1); the class list and the text tower are ignored.
    const bool visualPrompts = panelUsesVisualPrompts(*panel);
    if (visualPrompts && panel->vpLabel) {
        s.visualPrompts = panel->vpLabel->boxes();
    }
    if (panel->classesEdit && !promptFree && !visualPrompts) {
        // Split the comma-separated class list; surrounding spaces are
        // padding, but an empty field stays a real class row (the
        // background prompt semantics of the YOLO-World docs). Chinese
        // prompts are translated to English (dictionary + "wear X-hat Y"
        // templates) because the text towers are English-trained — bridged
        // Chinese vectors lose the color/age discriminative directions.
        const QStringList raw =
                panel->classesEdit->text().split(QLatin1Char(','));
        for (const QString& name : raw) {
            const QString trimmed = name.trimmed();
            if (trimmed.isEmpty()) {
                s.classes.append(trimmed);
                continue;
            }
            bool translated = false;
            s.classes.append(YOLOHelpers::translatePromptToEnglish(
                    trimmed, &translated));
        }
        while (!s.classes.isEmpty() && s.classes.last().isEmpty() &&
               s.classes.size() > 1) {
            // Drop only the trailing empty field produced by a trailing
            // comma; interior empties stay (real class rows).
            s.classes.removeLast();
        }
        if (s.classes.size() == 1 && s.classes.first().isEmpty()) {
            s.classes.clear();  // blank input = checkpoint vocabulary
        }
    }
    if (panel->textModelCombo) {
        s.textModelPath = panel->textModelPath();
    }
    return s;
}

void YOLODialog::appendLog(const QString& msg) {
#ifdef AICore_ENABLED
    aicore_inference_log::log(msg);
#endif
    if (!m_taskStatusLabel || !msg.startsWith(QStringLiteral("[Error]"))) {
        return;
    }
    // Record the latest error and show it immediately: pre-run validation
    // (onRun / ensureModelAvailable) returns before the plugin enters the
    // running state, so the label would otherwise stay untouched until a
    // later setRunning(false) flushes m_lastTaskError.
    m_lastTaskError = msg.mid(QStringLiteral("[Error]").size()).trimmed();
    m_taskStatusLabel->setText(m_lastTaskError);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #b91c1c;");
    m_taskStatusLabel->setVisible(true);
}

void YOLODialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void YOLODialog::setTaskStage(const QString& stage, int percent) {
    if (!m_taskStatusLabel) return;
    m_taskStatusLabel->setText(stage);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    m_taskStatusLabel->setVisible(true);
    m_progress->setVisible(true);
    if (percent >= 0) {
        m_progress->setRange(0, 100);
        m_progress->setValue(percent);
    } else {
        m_progress->setRange(0, 0);
    }
}

void YOLODialog::enableResultButtons(bool /*hasResult*/) {
    // Reserved for future Visualize/Export buttons (aligned with
    // qFreeSplatter).
}

void YOLODialog::setRunning(bool running) {
    for (YOLOTaskPanel& panel : m_panels) {
        panel.runBtn->setEnabled(!running);
        panel.cancelBtn->setEnabled(running);
    }
    if (running) {
        m_lastTaskError.clear();
        m_taskStatusLabel->setText(tr("Starting..."));
        m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
        m_taskStatusLabel->setVisible(true);
        m_progress->setVisible(true);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    } else {
        if (m_lastTaskError.isEmpty()) {
            m_taskStatusLabel->clear();
            m_taskStatusLabel->setVisible(false);
        } else {
            m_taskStatusLabel->setText(m_lastTaskError);
            m_taskStatusLabel->setStyleSheet(
                    "font-weight: bold; color: #b91c1c;");
            m_taskStatusLabel->setVisible(true);
        }
        m_progress->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    }
}

void YOLODialog::setDbImages(const QList<DbImageEntry>& images) {
    for (YOLOTaskPanel& panel : m_panels) {
        panel.dbImageList->clear();
        for (const DbImageEntry& e : images) {
            auto* item =
                    new QListWidgetItem(QIcon(QPixmap::fromImage(e.preview)),
                                        e.name, panel.dbImageList);
            item->setData(Qt::UserRole, e.name);
            // Full-resolution image for the click-to-enlarge preview (the
            // icon above is only a scaled thumbnail).
            item->setData(kDbFullImageRole, e.preview);
        }
    }
}

void YOLODialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return;
    panel->imagePath->setText(QStringLiteral("db://") + imageNames.first());
    m_imagePath = panel->imagePath;
    m_previewLabel = panel->previewLabel;
    updateImagePreview();
}

void YOLODialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return;
    panel->imagePath->setText(QStringLiteral("db://") +
                              item->data(Qt::UserRole).toString());
    m_imagePath = panel->imagePath;
    m_previewLabel = panel->previewLabel;
    updateImagePreview();
}

void YOLODialog::onLiveStart() {
    if (!m_liveWidget) return;
    if (!ensureModelAvailable(PendingAction::LiveStart)) return;
    startLiveStream();
}

void YOLODialog::startLiveStream() {
    if (!m_liveWidget) return;
    YOLOLiveWidget::Config config = m_liveWidget->config();
    config.modelPath = m_liveWidget->resolveModelPath();
    config.device = m_liveWidget->deviceId();
    config.threads = m_liveWidget->threadCount();
    // Thresholds are read from the Live widget's own (adaptive) controls —
    // they stay visible/hidden according to the selected model's task.
    m_liveWidget->setConfig(config);

    if (m_liveWidget->inputSource() == YOLOLiveWidget::InputSource::VideoFile) {
        const QString path = m_liveWidget->videoFilePath();
        if (path.isEmpty() || !QFile::exists(path)) {
            appendLog(tr("[YOLO] Select a valid video file first."));
            return;
        }
        if (!m_liveWidget->startVideoFile(path)) {
            appendLog(tr("[YOLO] Failed to start video."));
        }
        return;
    }
    const int camIdx = m_liveWidget->selectedCameraIndex();
    if (camIdx < 0) {
        appendLog(tr("[YOLO] No camera available."));
        return;
    }
    if (!m_liveWidget->startCamera(camIdx)) {
        appendLog(tr("[YOLO] Failed to start camera %1.").arg(camIdx));
    }
}

void YOLODialog::onLiveStop() { m_liveWidget->stopStream(); }

void YOLODialog::onLiveRestart() { m_liveWidget->restartVideoFile(); }

void YOLODialog::onLiveCapture(const YOLORunResult& result) {
    emit liveCaptureReady(result);
}

void YOLODialog::onLiveDepthCapture(const YOLODepthResult& result) {
    emit liveDepthCaptureReady(result);
}

// ---------------------------------------------------------------------------
// Test data — via shared ecvTestDataRepository
// ---------------------------------------------------------------------------

void YOLODialog::requestTestData(TestDataTarget target) {
    if (m_downloadInProgress) {
        appendLog(tr("[Test data] Wait for model download to finish first."));
        return;
    }

    // Capture the requesting panel's task: the async path must fill THIS
    // panel even if the user switches tabs meanwhile.
    QString task;
    if (target == TestDataTarget::Image) {
        const YOLOTaskPanel* panel = currentTaskPanel();
        task = panel ? panel->task : QString();
    }
    // The file may already be cached even while a download chain is
    // running — always try the immediate load before queueing.
    if (loadTestDataFor(target, task)) return;

    if (m_testDataDownloadInProgress ||
        ecvTestDataRepository::instance().isDownloadInProgress()) {
        // The shared repository serves one download at a time (single
        // downloader slot, shared by every plugin). Queue this request
        // instead of dropping it: the repository's downloadFinished /
        // extractionFinished broadcasts resume the queued slots.
        if (m_pendingTestDataTarget == TestDataTarget::None) {
            m_pendingTestDataTarget = target;
            m_pendingTestDataTask = task;
        } else {
            m_followupTestDataTarget = target;
            m_followupTestDataTask = task;
        }
        appendLog(
                tr("[Test data] Queued — will load when the current "
                   "test-data download finishes."));
        return;
    }

    // No chain is running: this request drives it.
    m_pendingTestDataTarget = target;
    m_pendingTestDataTask = task;
    advancePendingTestData();
}

void YOLODialog::advancePendingTestData() {
    if (m_pendingTestDataTarget == TestDataTarget::None) return;
    if (loadTestDataFor(m_pendingTestDataTarget, m_pendingTestDataTask)) {
        // Clear the slot before serving: servePendingTestData() loads the
        // pending target again otherwise (double load of the same file).
        m_pendingTestDataTarget = TestDataTarget::None;
        m_pendingTestDataTask.clear();
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

    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    setTestDataControlsEnabled(false);
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        appendLog(tr("[Test data] Extracting cached archive..."));
        m_progress->setRange(0, 0);
        m_progress->setValue(0);
        m_progress->setVisible(true);
        m_downloadLabel->setText(
                tr("Extracting object detection test data..."));
        m_downloadLabel->setVisible(true);
        repo.extractDataset(kind);
        return;
    }

    m_downloadLabel->setText(tr("Downloading object detection test data..."));
    m_downloadLabel->setVisible(true);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(true);
    repo.startDownload(kind);
}

void YOLODialog::servePendingTestData() {
    if (m_pendingTestDataTarget != TestDataTarget::None) {
        if (!loadTestDataFor(m_pendingTestDataTarget, m_pendingTestDataTask)) {
            appendLog(
                    tr("[Test data] Requested file was not found in the "
                       "archive."));
        }
        m_pendingTestDataTarget = TestDataTarget::None;
        m_pendingTestDataTask.clear();
    }
    if (m_followupTestDataTarget != TestDataTarget::None) {
        if (!loadTestDataFor(m_followupTestDataTarget,
                             m_followupTestDataTask)) {
            appendLog(
                    tr("[Test data] Requested file was not found in the "
                       "archive."));
        }
        m_followupTestDataTarget = TestDataTarget::None;
        m_followupTestDataTask.clear();
    }
}

bool YOLODialog::loadTestDataFor(TestDataTarget target, const QString& task) {
    if (target == TestDataTarget::None) return false;
    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    QString fileName;
    if (target == TestDataTarget::Image) {
        // Resolve by the task captured at request time (falls back to the
        // active panel); loading into whatever panel is active at
        // COMPLETION time would fill the wrong tab when the user switches
        // while the archive downloads.
        YOLOTaskPanel* panel = panelForTask(task);
        if (!panel) panel = currentTaskPanel();
        fileName =
                YOLOHelpers::testImageForTask(panel ? panel->task : QString());
    } else if (target == TestDataTarget::Video && m_testVideoCombo) {
        fileName = m_testVideoCombo->currentData().toString();
    }
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(kind, fileName);
    if (path.isEmpty()) return false;

    if (target == TestDataTarget::Image) {
        YOLOTaskPanel* panel = panelForTask(task);
        if (!panel) panel = currentTaskPanel();
        if (!panel) return false;
        panel->imagePath->setText(path);
        m_imagePath = panel->imagePath;
        m_previewLabel = panel->previewLabel;
        updateImagePreview();
        appendLog(tr("[Test data] Loaded image: %1").arg(path));
    } else if (m_liveWidget) {
        m_liveWidget->setInputSource(YOLOLiveWidget::InputSource::VideoFile);
        m_liveWidget->setVideoFilePath(path, false);
        appendLog(tr("[Test data] Loaded video: %1").arg(path));
        appendLog(tr("[Test data] Press Start to run inference on it."));
    }
    return true;
}

void YOLODialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    const bool ours = m_testDataDownloadInProgress;
    const bool haveQueued = m_pendingTestDataTarget != TestDataTarget::None ||
                            m_followupTestDataTarget != TestDataTarget::None;
    if (!ours && !haveQueued) return;  // a broadcast we did not ask for

    if (kind != ecvTestDataRepository::Dataset::ObjectsDetection) {
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
        if (ours) {
            m_testDataDownloadInProgress = false;
            m_downloadLabel->setVisible(false);
            m_progress->setRange(0, 100);
            m_progress->setVisible(false);
            setTestDataControlsEnabled(true);
            appendLog(tr("[Test data] Download failed."));
            // Drop the failed request; a queued follow-up retries once
            // from our side (the repo slot is free now — bounded retry).
            m_pendingTestDataTarget = TestDataTarget::None;
            m_pendingTestDataTask.clear();
            if (m_followupTestDataTarget != TestDataTarget::None) {
                m_pendingTestDataTarget = m_followupTestDataTarget;
                m_pendingTestDataTask = m_followupTestDataTask;
                m_followupTestDataTarget = TestDataTarget::None;
                m_followupTestDataTask.clear();
                advancePendingTestData();
            }
            return;
        }
        // Foreign chain for OUR dataset failed: retry once from our side.
        QTimer::singleShot(0, this, [this]() { advancePendingTestData(); });
        return;
    }

    if (ours) {
        appendLog(tr("[Test data] Extracting..."));
        m_downloadLabel->setText(
                tr("Extracting object detection test data..."));
        m_progress->setRange(0, 0);  // indeterminate / busy
        m_progress->setVisible(true);
        ecvTestDataRepository::instance().extractDataset(kind);
        return;
    }
    // Queued on a foreign chain for OUR dataset: its starter extracts
    // next and the extractionFinished broadcast serves the queued slots —
    // do not extract the same archive twice on the GUI thread.
    appendLog(
            tr("[Test data] Download finished — loading after "
               "extraction..."));
}

void YOLODialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    const bool ours = m_testDataDownloadInProgress;
    if (ours && kind == ecvTestDataRepository::Dataset::ObjectsDetection) {
        m_testDataDownloadInProgress = false;

        m_downloadLabel->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
        m_progress->setVisible(false);
        setTestDataControlsEnabled(true);

        if (!success) {
            appendLog(tr("[Test data] Failed to extract zip archive."));
            // Same bounded-retry contract as the download-failure path:
            // the queued follow-up takes over as the pending request.
            m_pendingTestDataTarget = TestDataTarget::None;
            m_pendingTestDataTask.clear();
            if (m_followupTestDataTarget != TestDataTarget::None) {
                m_pendingTestDataTarget = m_followupTestDataTarget;
                m_pendingTestDataTask = m_followupTestDataTask;
                m_followupTestDataTarget = TestDataTarget::None;
                m_followupTestDataTask.clear();
                advancePendingTestData();
            }
            return;
        }
        servePendingTestData();
        return;
    }
    if (!ours && kind == ecvTestDataRepository::Dataset::ObjectsDetection &&
        (m_pendingTestDataTarget != TestDataTarget::None ||
         m_followupTestDataTarget != TestDataTarget::None)) {
        // A foreign plugin extracted OUR dataset (we were queued on its
        // download): the files just landed — serve the queued requests.
        if (success) {
            servePendingTestData();
        } else {
            QTimer::singleShot(0, this, [this]() { advancePendingTestData(); });
        }
    }
}

void YOLODialog::setTestDataControlsEnabled(bool enabled) {
    for (YOLOTaskPanel& panel : m_panels) {
        if (panel.testDataBtn) panel.testDataBtn->setEnabled(enabled);
    }
    if (m_testDataBtn) m_testDataBtn->setEnabled(enabled);
    if (m_testVideoCombo) m_testVideoCombo->setEnabled(enabled);
}

void YOLODialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    m_liveWidget->saveSettings();
    event->accept();
}
