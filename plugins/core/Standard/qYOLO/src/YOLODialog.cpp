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
#include <QSettings>
#include <QSizePolicy>
#include <QTimer>
#include <QVBoxLayout>

#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/yolo_capi.h"
#endif

namespace {
const int kThumbSize = 96;
constexpr const char* kYOLOTestImage = "000000397133.jpg";

void styleSampleDataButton(QPushButton* button) {
    button->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold; border: none; border-radius: 4px; padding: 5px 12px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:pressed { background: #00695c; }"
            "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
}
}  // namespace

YOLODialog::YOLODialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("YOLO Detect, Segment & Depth"));
    setupUi();
    populateModelCombo();
    loadSettings();
    m_liveWidget->loadSettings();
    // Content-driven minimum (font / DPI aware) instead of hard-coded
    // pixels, so the dialog adapts to any platform and screen resolution.
    setMinimumSize(minimumSizeHint());
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

void YOLODialog::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(6, 6, 6, 6);
    rootLayout->setSpacing(4);
    m_tabWidget = new QTabWidget(this);
    rootLayout->addWidget(m_tabWidget);

    // Shared inference parameters (device / threads) live OUTSIDE the task
    // tabs: they are runtime properties, while each tab owns its model
    // combo + thresholds (which DO differ per task).
    auto* sharedParams = new QHBoxLayout;
    sharedParams->setSpacing(6);
    sharedParams->addWidget(new QLabel(tr("Device:"), this));
    m_deviceCombo = new QComboBox(this);
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
    sharedParams->addWidget(m_deviceCombo);
    sharedParams->addWidget(new QLabel(tr("Threads:"), this));
    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 64);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = auto"));
    sharedParams->addWidget(m_threads);
    sharedParams->addStretch();
    rootLayout->addLayout(sharedParams);

    // ---- Per-task tabs (each with its own model combo + thresholds) -----
    const QStringList taskOrder = {QStringLiteral("detect"),
                                   QStringLiteral("segment"),
                                   QStringLiteral("depth")};
    const QStringList tabTitles = {tr("Object Detection"),
                                   tr("Instance Segmentation"),
                                   tr("Metric Depth")};

    auto makeParamLabel = [this](const QString& text, QWidget* parent) {
        auto* label = new QLabel(text, parent);
        label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        return label;
    };

    for (int i = 0; i < taskOrder.size(); ++i) {
        YOLOTaskPanel panel;
        panel.task = taskOrder[i];
        panel.tab = new QWidget(this);
        auto* layout = new QVBoxLayout(panel.tab);
        layout->setContentsMargins(4, 4, 4, 4);
        layout->setSpacing(4);

        // Two-column body: config controls on the left, preview on the
        // right, so the dialog stays compact along both axes.
        auto* contentRow = new QHBoxLayout;
        contentRow->setSpacing(8);
        auto* configCol = new QVBoxLayout;
        configCol->setSpacing(4);

        // Model row: label + combo (filtered to this task's catalog).
        auto* modelRow = new QHBoxLayout;
        modelRow->setSpacing(6);
        modelRow->addWidget(makeParamLabel(tr("Model:"), panel.tab));
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
        thresholdLayout->setSpacing(4);
        thresholdLayout->addWidget(makeParamLabel(tr("Conf:"), panel.tab));
        panel.conf = new QDoubleSpinBox(panel.tab);
        panel.conf->setRange(0.01, 1.0);
        panel.conf->setSingleStep(0.05);
        panel.conf->setValue(0.25);
        panel.conf->setToolTip(
                tr("Confidence threshold (detect/segment models)"));
        thresholdLayout->addWidget(panel.conf);
        thresholdLayout->addWidget(makeParamLabel(tr("IoU:"), panel.tab));
        panel.iou = new QDoubleSpinBox(panel.tab);
        panel.iou->setRange(0.1, 1.0);
        panel.iou->setSingleStep(0.05);
        panel.iou->setValue(0.7);
        panel.iou->setToolTip(tr("NMS IoU threshold (detect/segment models)"));
        thresholdLayout->addWidget(panel.iou);
        thresholdLayout->addWidget(makeParamLabel(tr("Top-K:"), panel.tab));
        panel.topK = new QSpinBox(panel.tab);
        panel.topK->setRange(1, 1000);
        panel.topK->setValue(300);
        thresholdLayout->addWidget(panel.topK);
        thresholdLayout->addStretch();
        configCol->addWidget(panel.thresholdRow);

        // Custom GGUF row (shown only when a non-catalog file is picked).
        panel.customModelRow = new QWidget(panel.tab);
        auto* customRow = new QHBoxLayout(panel.customModelRow);
        customRow->setContentsMargins(0, 0, 0, 0);
        customRow->setSpacing(6);
        customRow->addWidget(
                new QLabel(tr("Custom GGUF:"), panel.customModelRow));
        panel.customModelPath = new QLineEdit(panel.customModelRow);
        customRow->addWidget(panel.customModelPath, 1);
        auto* browseCustomBtn =
                new QPushButton(tr("Browse…"), panel.customModelRow);
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

        // Input row: image path + browse + sample-data button.
        auto* inputRow = new QHBoxLayout;
        inputRow->setSpacing(6);
        inputRow->addWidget(makeParamLabel(tr("Image:"), panel.tab));
        panel.imagePath = new QLineEdit(panel.tab);
        inputRow->addWidget(panel.imagePath, 1);
        auto* browseBtn = new QPushButton(tr("Browse…"), panel.tab);
        connect(browseBtn, &QPushButton::clicked, this, [this]() {
            YOLOTaskPanel* active = currentTaskPanel();
            if (!active) return;
            m_imagePath = active->imagePath;
            onBrowseImage();
        });
        inputRow->addWidget(browseBtn);
        panel.testDataBtn =
                new QPushButton(tr("\U0001f9ea  Try sample data"), panel.tab);
        styleSampleDataButton(panel.testDataBtn);
        panel.testDataBtn->setToolTip(
                tr("Load images/000000397133.jpg from the shared test-data "
                   "cache"));
        connect(panel.testDataBtn, &QPushButton::clicked, this,
                [this]() { requestTestData(TestDataTarget::Image); });
        inputRow->addWidget(panel.testDataBtn);
        configCol->addLayout(inputRow);

        // DB image picker (collapsible).
        panel.dbToggleBtn = new QToolButton(panel.tab);
        panel.dbToggleBtn->setText(tr("DB images ▾"));
        panel.dbToggleBtn->setCheckable(true);
        panel.dbToggleBtn->setChecked(false);
        configCol->addWidget(panel.dbToggleBtn, 0, Qt::AlignLeft);
        panel.dbContentWidget = new QWidget(panel.tab);
        auto* dbLayout = new QVBoxLayout(panel.dbContentWidget);
        dbLayout->setContentsMargins(0, 0, 0, 0);
        dbLayout->setSpacing(4);
        panel.dbImageList = new QListWidget(panel.dbContentWidget);
        panel.dbImageList->setIconSize(QSize(48, 48));
        // ~8 rows, scaled with the dialog font instead of fixed pixels.
        panel.dbImageList->setMaximumHeight(panel.tab->fontMetrics().height() *
                                            8);
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
        configCol->addWidget(panel.dbContentWidget);
        configCol->addStretch();
        connect(panel.dbToggleBtn, &QToolButton::toggled, this,
                [this](bool on) {
                    YOLOTaskPanel* active = currentTaskPanel();
                    if (!active) return;
                    active->dbContentWidget->setVisible(on);
                    if (on) emit refreshDbImagesRequested();
                    // Auto-grow the dialog so the expanded DB list stays
                    // fully visible; never shrink a user-resized dialog.
                    QTimer::singleShot(0, this, [this]() {
                        const QSize hint = sizeHint();
                        resize(qMax(width(), hint.width()),
                               qMax(height(), hint.height()));
                    });
                });
        connect(panel.dbImageList, &QListWidget::itemActivated, this,
                &YOLODialog::onDbListActivated);
        connect(panel.dbImageList, &QListWidget::itemClicked, this,
                &YOLODialog::onDbListActivated);

        contentRow->addLayout(configCol, 1);

        // Right column: preview thumbnail (top-aligned). The thumbnail size
        // is in logical pixels (Qt scales it by devicePixelRatio), while all
        // text-adjacent sizes stay font-relative for cross-platform fit.
        auto* previewCol = new QVBoxLayout;
        previewCol->setSpacing(4);
        panel.previewLabel = new ecvClickableImageLabel(panel.tab);
        panel.previewLabel->setFixedSize(kThumbSize, kThumbSize);
        panel.previewLabel->setStyleSheet(
                "border: 1px solid palette(mid); background: palette(base);");
        panel.previewLabel->setText(tr("Preview"));
        previewCol->addWidget(panel.previewLabel);
        auto* previewHint = new QLabel(tr("Tap to preview"), panel.tab);
        previewHint->setAlignment(Qt::AlignCenter);
        previewHint->setStyleSheet(
                QStringLiteral("color: palette(mid); font-size: 11px;"));
        previewCol->addWidget(previewHint);
        previewCol->addStretch();
        contentRow->addLayout(previewCol);

        layout->addLayout(contentRow);

        // Action row: add-to-DB + Run / Cancel.
        auto* actionRow = new QHBoxLayout;
        actionRow->setSpacing(6);
        panel.addAnnotatedCheck =
                new QCheckBox(tr("Add annotated image to DB"), panel.tab);
        panel.addAnnotatedCheck->setChecked(true);
        actionRow->addWidget(panel.addAnnotatedCheck);
        actionRow->addStretch();
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
        m_tabWidget->addTab(panel.tab, tabTitles[i]);
    }

    // ---- Live (camera / video) tab ----------------------------------------
    m_liveTab = new QWidget(this);
    auto* liveLayout = new QVBoxLayout(m_liveTab);
    liveLayout->setContentsMargins(4, 4, 4, 4);
    liveLayout->setSpacing(4);
    m_liveWidget = new YOLOLiveWidget(m_liveTab);
    liveLayout->addWidget(m_liveWidget, 1);

    // Playback controls live in the Live tab itself (mirrors qFaceDetect).
    auto* liveBtnRow = new QHBoxLayout;
    liveBtnRow->setSpacing(6);
    m_testVideoCombo = new QComboBox(m_liveTab);
    m_testVideoCombo->addItem(QStringLiteral("traffic.mp4"),
                              QStringLiteral("traffic.mp4"));
    m_testVideoCombo->addItem(QStringLiteral("supervision_demo.mp4"),
                              QStringLiteral("supervision_demo.mp4"));
    m_testDataBtn =
            new QPushButton(tr("\U0001f9ea  Try sample data"), m_liveTab);
    styleSampleDataButton(m_testDataBtn);
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

    m_tabWidget->addTab(m_liveTab, tr("Live (camera / video)"));

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

    // The Live tab lists ALL catalog models (any task) and shares the
    // device/threads controls with the batch tabs.
    m_liveWidget->populateAllModels();
    m_liveWidget->rebuildDeviceCombo(m_deviceCombo);
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            m_liveWidget, [this](int) {
                m_liveWidget->setDevice(
                        m_deviceCombo->currentData().toString());
            });
    connect(m_threads, QOverload<int>::of(&QSpinBox::valueChanged),
            m_liveWidget, [this](int v) { m_liveWidget->setThreads(v); });
    connect(m_liveWidget, &YOLOLiveWidget::modelSelectionChanged, this,
            [this](const QString& filename) {
                // Keep the matching batch tab's model in sync so the two
                // surfaces don't drift, but do NOT force a tab switch.
                YOLOTaskPanel* panel = panelForFilename(filename);
                if (panel && panel->modelCombo) {
                    const int idx = panel->modelCombo->findData(filename);
                    if (idx >= 0) panel->modelCombo->setCurrentIndex(idx);
                }
            });
    connect(m_liveWidget, &YOLOLiveWidget::deviceSelectionChanged, this,
            [this](const QString& device) {
                const int index = m_deviceCombo->findData(device);
                if (index >= 0 && index != m_deviceCombo->currentIndex()) {
                    m_deviceCombo->setCurrentIndex(index);
                }
            });
    connect(m_liveWidget, &YOLOLiveWidget::threadCountChanged, this,
            [this](int threads) {
                if (m_threads->value() != threads) m_threads->setValue(threads);
            });
    connect(m_liveWidget, &YOLOLiveWidget::captureToDbRequested, this,
            &YOLODialog::onLiveCapture);
    connect(m_liveWidget, &YOLOLiveWidget::depthCaptureToDbRequested, this,
            &YOLODialog::onLiveDepthCapture);

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

    // Download / task progress — shared by all tabs so a model fetch started
    // from any tab stays visible.
    rootLayout->addWidget(m_downloadLabel = new QLabel(this));
    m_downloadLabel->setWordWrap(true);
    m_downloadLabel->setVisible(false);
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    rootLayout->addWidget(m_progress);

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

void YOLODialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO"));
    const QStringList tasks = {QStringLiteral("detect"),
                               QStringLiteral("segment"),
                               QStringLiteral("depth")};
    for (int i = 0; i < m_panels.size() && i < tasks.size(); ++i) {
        YOLOTaskPanel& panel = m_panels[i];
        const QString modelFilename =
                settings.value(QStringLiteral("modelFilename/") + tasks[i])
                        .toString();
        if (!modelFilename.isEmpty()) {
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
    }
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    settings.endGroup();
}

void YOLODialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qYOLO"));
    const QStringList tasks = {QStringLiteral("detect"),
                               QStringLiteral("segment"),
                               QStringLiteral("depth")};
    for (int i = 0; i < m_panels.size() && i < tasks.size(); ++i) {
        const YOLOTaskPanel& panel = m_panels[i];
        settings.setValue(QStringLiteral("modelFilename/") + tasks[i],
                          panel.modelCombo->currentData().toString());
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
    }
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.endGroup();
}

QString YOLODialog::modelCacheDir() { return YOLOHelpers::modelCacheDir(); }

void YOLODialog::populateModelCombo(const QString& keepFilename) {
    // Each task panel lists only its own task's catalog models.
    const QStringList tasks = {QStringLiteral("detect"),
                               QStringLiteral("segment"),
                               QStringLiteral("depth")};
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
        if (!keepFilename.isEmpty()) {
            const int idx = panel.modelCombo->findData(keepFilename);
            if (idx >= 0) panel.modelCombo->setCurrentIndex(idx);
        }
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

bool YOLODialog::selectModelByFilename(const QString& filename) {
    if (filename.isEmpty()) return false;
    for (YOLOTaskPanel& panel : m_panels) {
        const int idx = panel.modelCombo->findData(filename);
        if (idx >= 0) {
            panel.modelCombo->setCurrentIndex(idx);
            return true;
        }
    }
    return false;
}

void YOLODialog::refreshModelList() {
    const QString keep =
            currentTaskPanel() && currentTaskPanel()->modelCombo
                    ? currentTaskPanel()->modelCombo->currentData().toString()
                    : QString();
    populateModelCombo(keep);
}

YOLOTaskPanel* YOLODialog::currentTaskPanel() const {
    return panelForTab(m_tabWidget->currentWidget());
}

YOLOTaskPanel* YOLODialog::panelForTab(QWidget* tab) const {
    for (const YOLOTaskPanel& panel : m_panels) {
        if (panel.tab == tab) {
            // const_cast: callers expect a mutable panel (they set controls).
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

    // Threshold row visible for detect/segment, hidden for depth.
    panel.thresholdRow->setVisible(panel.task != QStringLiteral("depth"));
}

QString YOLODialog::resolveModelPath() const {
    const YOLOTaskPanel* panel = currentTaskPanel();
    return panel ? panel->modelPath() : QString();
}

bool YOLODialog::ensureModelAvailable(PendingAction action) {
    YOLOTaskPanel* panel = currentTaskPanel();
    if (!panel) return false;
    const QString filename = panel->modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[YOLO] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(panel->modelPath())) {
        YOLOModelEntry entry;
        if (!YOLOHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[YOLO] Model file not found: %1").arg(filename));
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
    const QImage img(m_imagePath->text());
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    m_previewLabel->setPreviewImage(img, kThumbSize);
}

void YOLODialog::onRun() {
    if (!ensureModelAvailable(PendingAction::Run)) return;
    emit runRequested(getSettings());
}

void YOLODialog::onCancel() {
    cancelDownload();
    emit cancelRequested();
}

YOLODialog::Settings YOLODialog::getSettings() const {
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
    return s;
}

void YOLODialog::appendLog(const QString& msg) {
#ifdef AICore_ENABLED
    aicore_inference_log::log(msg);
#endif
    if (!m_taskStatusLabel || !msg.startsWith(QStringLiteral("[Error]"))) {
        return;
    }
    if (m_lastTaskError.isEmpty()) {
        m_lastTaskError = msg.mid(QStringLiteral("[Error]").size()).trimmed();
    }
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
    m_taskRunning = running;
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
    if (m_testDataDownloadInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    if (m_downloadInProgress) {
        appendLog(tr("[Test data] Wait for model download to finish first."));
        return;
    }

    m_pendingTestDataTarget = target;
    if (loadRequestedTestData()) {
        m_pendingTestDataTarget = TestDataTarget::None;
        return;
    }

    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        m_pendingTestDataTarget = TestDataTarget::None;
        return;
    }

    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    setTestDataControlsEnabled(false);
    if (ecvTestDataRepository::verifyZipIntegrity(
                ecvTestDataRepository::zipPath(kind), info.expectedMd5,
                info.expectedSize)) {
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

bool YOLODialog::loadRequestedTestData() {
    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    QString fileName;
    if (m_pendingTestDataTarget == TestDataTarget::Image) {
        fileName = QString::fromLatin1(kYOLOTestImage);
    } else if (m_pendingTestDataTarget == TestDataTarget::Video &&
               m_testVideoCombo) {
        fileName = m_testVideoCombo->currentData().toString();
    }
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(kind, fileName);
    if (path.isEmpty()) return false;

    if (m_pendingTestDataTarget == TestDataTarget::Image) {
        YOLOTaskPanel* panel = currentTaskPanel();
        if (panel) {
            panel->imagePath->setText(path);
            m_imagePath = panel->imagePath;
            m_previewLabel = panel->previewLabel;
            updateImagePreview();
            appendLog(tr("[Test data] Loaded image: %1").arg(path));
        }
    } else if (m_pendingTestDataTarget == TestDataTarget::Video &&
               m_liveWidget) {
        m_liveWidget->setInputSource(YOLOLiveWidget::InputSource::VideoFile);
        m_liveWidget->setVideoFilePath(path, false);
        appendLog(tr("[Test data] Loaded video: %1").arg(path));
        appendLog(tr("[Test data] Press Start to run inference on it."));
    }
    return true;
}

void YOLODialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::ObjectsDetection) {
        return;
    }

    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        m_testDataDownloadInProgress = false;
        m_downloadLabel->setVisible(false);
        m_progress->setRange(0, 100);
        m_progress->setVisible(false);
        setTestDataControlsEnabled(true);
        m_pendingTestDataTarget = TestDataTarget::None;
        return;
    }

    appendLog(tr("[Test data] Extracting..."));
    m_downloadLabel->setText(tr("Extracting object detection test data..."));
    m_progress->setRange(0, 0);  // indeterminate / busy
    m_progress->setVisible(true);
    ecvTestDataRepository::instance().extractDataset(kind);
}

void YOLODialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::ObjectsDetection) {
        return;
    }
    m_testDataDownloadInProgress = false;

    m_downloadLabel->setVisible(false);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    setTestDataControlsEnabled(true);

    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        m_pendingTestDataTarget = TestDataTarget::None;
        return;
    }
    if (!loadRequestedTestData()) {
        appendLog(
                tr("[Test data] Requested file was not found in the archive."));
    }
    m_pendingTestDataTarget = TestDataTarget::None;
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

void YOLODialog::changeEvent(QEvent* event) {
    QDialog::changeEvent(event);
    if (event->type() == QEvent::ActivationChange) {
        adaptTabWidgetHeight();
    }
}

void YOLODialog::adaptTabWidgetHeight() {
    // Keep the dialog compact on small screens; the live tab has its own
    // fixed preview height.
    if (m_activeTabHeight < 0) {
        m_activeTabHeight = m_tabWidget->height();
    }
}
