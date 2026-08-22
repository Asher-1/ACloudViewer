// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrDialog.h"

#include <cvFileDialog.h>

#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
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
#include <QVBoxLayout>

#include "ecvAICoreUiHelper.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/rfdetr_capi.h"
#endif

namespace {
constexpr const char* kRFDetrTestImage = "000000397133.jpg";
// QListWidgetItem data role carrying the full-resolution ccImage for the
// click-to-enlarge preview (the list icon is only a scaled thumbnail).
constexpr int kDbFullImageRole = Qt::UserRole + 1;
}  // namespace

RFDetrDialog::RFDetrDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RF-DETR Object Detection"));
    setMinimumSize(ecvAICoreUi::dpiScaled(680), ecvAICoreUi::dpiScaled(560));
    setupUi();
    populateModelCombo();
    loadSettings();
    m_liveWidget->loadSettings();
}

RFDetrDialog::~RFDetrDialog() {
    saveSettings();
    m_liveWidget->saveSettings();
}

void RFDetrDialog::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(rootLayout);
    m_tabWidget = new QTabWidget(this);
    ecvAICoreUi::styleTabWidget(m_tabWidget);
    rootLayout->addWidget(m_tabWidget);

    // ---- Image tab --------------------------------------------------------
    m_imageTab = new QWidget(this);
    auto* imageLayout = new QVBoxLayout(m_imageTab);
    ecvAICoreUi::setupTabLayout(imageLayout);

    auto* modelRow = new QHBoxLayout;
    modelRow->setSpacing(ecvAICoreUi::hSpacing());
    modelRow->addWidget(ecvAICoreUi::makeLabel(tr("Model:")));
    m_modelCombo = new QComboBox(m_imageTab);
    m_modelCombo->setMinimumContentsLength(26);
    m_modelCombo->setSizeAdjustPolicy(
            QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    modelRow->addWidget(m_modelCombo, 1);
    imageLayout->addLayout(modelRow);

    m_customModelRow = new QWidget(m_imageTab);
    auto* customRow = new QHBoxLayout(m_customModelRow);
    customRow->setContentsMargins(0, 0, 0, 0);
    customRow->setSpacing(ecvAICoreUi::hSpacing());
    customRow->addWidget(ecvAICoreUi::makeLabel(tr("Custom GGUF:")));
    m_customModelPath = new QLineEdit(m_customModelRow);
    customRow->addWidget(m_customModelPath, 1);
    auto* browseCustomBtn = ecvAICoreUi::makeBrowseBtn("Browse...");
    connect(browseCustomBtn, &QPushButton::clicked, this,
            &RFDetrDialog::onBrowseCustomModel);
    customRow->addWidget(browseCustomBtn);
    imageLayout->addWidget(m_customModelRow);

    auto* runRow = new QHBoxLayout;
    runRow->setSpacing(ecvAICoreUi::hSpacing());
    m_deviceCombo = new QComboBox(m_imageTab);
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
    m_threads = new QSpinBox(m_imageTab);
    m_threads->setRange(0, 64);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = auto"));
    auto* runtimeWidget = ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads);
    runRow->addWidget(runtimeWidget);

    runRow->addWidget(ecvAICoreUi::makeLabel(tr("Threshold:")));
    m_threshold = new QDoubleSpinBox(m_imageTab);
    m_threshold->setRange(0.01, 1.0);
    m_threshold->setSingleStep(0.05);
    m_threshold->setValue(0.5);
    ecvAICoreUi::setCompactDoubleSpin(m_threshold);
    runRow->addWidget(m_threshold);

    runRow->addWidget(ecvAICoreUi::makeLabel(tr("Top-K:")));
    m_topK = new QSpinBox(m_imageTab);
    m_topK->setRange(1, 1000);
    m_topK->setValue(300);
    runRow->addWidget(m_topK);
    runRow->addStretch();
    imageLayout->addLayout(runRow);

    auto* inputRow = new QHBoxLayout;
    inputRow->setSpacing(ecvAICoreUi::hSpacing());
    inputRow->addWidget(ecvAICoreUi::makeLabel(tr("Image:")));
    m_imagePath = new QLineEdit(m_imageTab);
    inputRow->addWidget(m_imagePath, 1);
    auto* browseBtn = ecvAICoreUi::makeBrowseBtn("Browse...");
    connect(browseBtn, &QPushButton::clicked, this,
            &RFDetrDialog::onBrowseImage);
    inputRow->addWidget(browseBtn);
    imageLayout->addLayout(inputRow);

    // DB image picker (collapsible).
    m_dbToggleBtn = ecvAICoreUi::makeDbSection(nullptr);
    imageLayout->addWidget(m_dbToggleBtn, 0, Qt::AlignLeft);
    m_dbContentWidget = new QWidget(m_imageTab);
    auto* dbLayout = new QVBoxLayout(m_dbContentWidget);
    dbLayout->setContentsMargins(0, 0, 0, 0);
    dbLayout->setSpacing(ecvAICoreUi::tightVSpacing());
    m_dbImageList = new QListWidget(m_dbContentWidget);
    m_dbImageList->setIconSize(QSize(48, 48));
    m_dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
    dbLayout->addWidget(m_dbImageList);
    auto* dbBtnRow = new QHBoxLayout;
    auto* refreshDbBtn = new QPushButton(tr("Refresh"), m_dbContentWidget);
    refreshDbBtn->setToolTip(tr("Reload the ccImage list from the DB tree"));
    connect(refreshDbBtn, &QPushButton::clicked, this,
            [this]() { emit refreshDbImagesRequested(); });
    dbBtnRow->addWidget(refreshDbBtn);
    dbBtnRow->addStretch();
    dbLayout->addLayout(dbBtnRow);
    m_dbContentWidget->setVisible(false);
    imageLayout->addWidget(m_dbContentWidget);
    ecvAICoreUi::connectDbToggle(m_dbToggleBtn, m_dbContentWidget);
    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool on) {
        if (on) emit refreshDbImagesRequested();
    });
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &RFDetrDialog::onDbListActivated);
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &RFDetrDialog::onDbListActivated);

    m_previewLabel = new ecvClickableImageLabel(m_imageTab);
    m_previewLabel->setFixedSize(ecvAICoreUi::previewSize(),
                                 ecvAICoreUi::previewSize());
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabel->setText(tr("Preview"));
    imageLayout->addWidget(
            ecvClickableImageLabel::wrapWithTapToPreviewHint(m_previewLabel));

    m_taskStatusLabel = new QLabel(m_imageTab);
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    imageLayout->addWidget(m_taskStatusLabel);

    auto* actionRow = new QHBoxLayout;
    actionRow->setSpacing(ecvAICoreUi::hSpacing());
    m_addAnnotatedCheck =
            new QCheckBox(tr("Add annotated image to DB"), m_imageTab);
    m_addAnnotatedCheck->setChecked(true);
    actionRow->addWidget(m_addAnnotatedCheck);
    actionRow->addStretch();
    m_imageTestDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_imageTestDataBtn->setToolTip(
            tr("Load images/000000397133.jpg from the shared test-data cache"));
    connect(m_imageTestDataBtn, &QPushButton::clicked, this,
            [this]() { requestTestData(TestDataTarget::Image); });
    actionRow->addWidget(m_imageTestDataBtn);
    m_runBtn = new QPushButton(tr("Run"), m_imageTab);
    m_runBtn->setDefault(true);
    actionRow->addWidget(m_runBtn);
    m_cancelBtn = new QPushButton(tr("Cancel"), m_imageTab);
    m_cancelBtn->setEnabled(false);
    actionRow->addWidget(m_cancelBtn);
    imageLayout->addLayout(actionRow);

    m_tabWidget->addTab(m_imageTab, tr("Image"));

    // ---- Live (camera / video) tab ----------------------------------------
    m_liveTab = new QWidget(this);
    auto* liveLayout = new QVBoxLayout(m_liveTab);
    ecvAICoreUi::setupTabLayout(liveLayout);
    m_liveWidget = new RFDetrLiveWidget(m_liveTab);
    liveLayout->addWidget(m_liveWidget, 1);

    // Playback controls live in the Live tab itself (mirrors qFaceDetect).
    auto* liveBtnRow = new QHBoxLayout;
    liveBtnRow->setSpacing(ecvAICoreUi::hSpacing());
    m_testVideoCombo = new QComboBox(m_liveTab);
    m_testVideoCombo->addItem(QStringLiteral("traffic.mp4"),
                              QStringLiteral("traffic.mp4"));
    m_testVideoCombo->addItem(QStringLiteral("supervision_demo.mp4"),
                              QStringLiteral("supervision_demo.mp4"));
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
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

    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RFDetrDialog::onModelComboChanged);
    connect(m_runBtn, &QPushButton::clicked, this, &RFDetrDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &RFDetrDialog::onCancel);
    connect(m_liveStartBtn, &QPushButton::clicked, this,
            &RFDetrDialog::onLiveStart);
    connect(m_liveStopBtn, &QPushButton::clicked, this,
            &RFDetrDialog::onLiveStop);
    connect(m_liveRestartBtn, &QPushButton::clicked, this,
            &RFDetrDialog::onLiveRestart);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            [this]() { requestTestData(TestDataTarget::Video); });

    // Keep the live button states in sync with the stream lifecycle.
    connect(m_liveWidget, &RFDetrLiveWidget::streamStarted, this, [this]() {
        m_liveStartBtn->setEnabled(false);
        m_liveStopBtn->setEnabled(true);
        m_liveRestartBtn->setEnabled(m_liveWidget->inputSource() ==
                                     RFDetrLiveWidget::InputSource::VideoFile);
    });
    connect(m_liveWidget, &RFDetrLiveWidget::streamStopped, this, [this]() {
        m_liveStartBtn->setEnabled(true);
        m_liveStopBtn->setEnabled(false);
        if (m_liveWidget->inputSource() !=
            RFDetrLiveWidget::InputSource::VideoFile) {
            m_liveRestartBtn->setEnabled(false);
        }
    });

    // Keep the live tab's model/device/threads controls in sync.
    m_liveWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo, m_threads);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            m_liveWidget,
            [this](int) { m_liveWidget->setModelPath(resolveModelPath()); });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            m_liveWidget, [this](int) {
                m_liveWidget->setDevice(
                        m_deviceCombo->currentData().toString());
            });
    connect(m_threads, QOverload<int>::of(&QSpinBox::valueChanged),
            m_liveWidget, [this](int v) { m_liveWidget->setThreads(v); });
    connect(m_liveWidget, &RFDetrLiveWidget::modelSelectionChanged, this,
            [this](const QString& filename) {
                const int index = m_modelCombo->findData(filename);
                if (index < 0 || index == m_modelCombo->currentIndex()) return;
                const bool restartStream = m_liveWidget->isActive();
                if (restartStream) m_liveWidget->stopStream();
                m_modelCombo->setCurrentIndex(index);
                if (restartStream) onLiveStart();
            });
    connect(m_liveWidget, &RFDetrLiveWidget::deviceSelectionChanged, this,
            [this](const QString& device) {
                const int index = m_deviceCombo->findData(device);
                if (index >= 0 && index != m_deviceCombo->currentIndex()) {
                    m_deviceCombo->setCurrentIndex(index);
                }
            });
    connect(m_liveWidget, &RFDetrLiveWidget::threadCountChanged, this,
            [this](int threads) {
                if (m_threads->value() != threads) m_threads->setValue(threads);
            });
    connect(m_liveWidget, &RFDetrLiveWidget::captureToDbRequested, this,
            &RFDetrDialog::onLiveCapture);

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
            &RFDetrDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[RF-DETR] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[RF-DETR] Model downloaded: %1").arg(path));
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

    // Download / task progress — shared by both tabs so a model fetch
    // started from the Live tab stays visible.
    ecvAICoreUi::setupProgressSection(rootLayout, m_downloadLabel, m_progress);

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

void RFDetrDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void RFDetrDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr"));
    const QString modelFilename =
            settings.value(QStringLiteral("modelFilename")).toString();
    selectModelByFilename(modelFilename);
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    m_threshold->setValue(
            settings.value(QStringLiteral("threshold"), 0.5).toDouble());
    m_topK->setValue(settings.value(QStringLiteral("topK"), 300).toInt());
    const QString imagePath =
            settings.value(QStringLiteral("imagePath")).toString();
    if (!imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
        updateImagePreview();
    }
    m_addAnnotatedCheck->setChecked(
            settings.value(QStringLiteral("addAnnotated"), true).toBool());
    settings.endGroup();
}

void RFDetrDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr"));
    settings.setValue(QStringLiteral("modelFilename"),
                      m_modelCombo->currentData().toString());
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.setValue(QStringLiteral("threshold"), m_threshold->value());
    settings.setValue(QStringLiteral("topK"), m_topK->value());
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("addAnnotated"),
                      m_addAnnotatedCheck->isChecked());
    settings.endGroup();
}

QString RFDetrDialog::modelCacheDir() { return RFDetrHelpers::modelCacheDir(); }

void RFDetrDialog::populateModelCombo(const QString& keepFilename) {
    const QVector<RFDetrModelEntry> models = RFDetrHelpers::catalogModels();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const RFDetrModelEntry& e : models) {
        m_modelCombo->addItem(RFDetrHelpers::modelDisplayLabel(e), e.filename);
    }
    if (!keepFilename.isEmpty()) {
        const int idx = m_modelCombo->findData(keepFilename);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    m_modelCombo->blockSignals(false);
    if (m_liveWidget) {
        m_liveWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo,
                                            m_threads);
    }
    onModelComboChanged(m_modelCombo->currentIndex());
}

bool RFDetrDialog::selectModelByFilename(const QString& filename) {
    if (filename.isEmpty()) return false;
    const int idx = m_modelCombo->findData(filename);
    if (idx < 0) return false;
    m_modelCombo->setCurrentIndex(idx);
    return true;
}

void RFDetrDialog::refreshModelList() {
    const QString keep = m_modelCombo->currentData().toString();
    populateModelCombo(keep);
}

void RFDetrDialog::onModelComboChanged(int index) {
    const QString filename = m_modelCombo->itemData(index).toString();
    const bool isCustom =
            filename.isEmpty() ||
            filename.endsWith(QStringLiteral(".gguf")) &&
                    !RFDetrHelpers::findModelByFilename(filename, nullptr);
    m_customModelRow->setVisible(isCustom);
    m_liveWidget->setModelPath(resolveModelPath());
}

QString RFDetrDialog::resolveModelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = RFDetrHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

bool RFDetrDialog::ensureModelAvailable(PendingAction action) {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[RF-DETR] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath())) {
        RFDetrModelEntry entry;
        if (!RFDetrHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[RF-DETR] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[RF-DETR] Model missing — downloading %1; it will "
                     "start automatically when ready.")
                          .arg(filename));
        startDownload(entry);
        return false;
    }
    return true;
}

void RFDetrDialog::startDownload(const RFDetrModelEntry& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[RF-DETR] A download is already running."));
        return;
    }
    QDir().mkpath(RFDetrHelpers::modelCacheDir());
    const QString dest =
            RFDetrHelpers::modelCacheDir() + QDir::separator() + model.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[RF-DETR] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[RF-DETR] Downloading %1 (%2)…")
                      .arg(model.filename, model.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // RF-DETR GGUFs are tens of MB
    m_downloader->download(req);
}

void RFDetrDialog::cancelDownload() {
    if (m_downloadInProgress) m_downloader->cancel();
}

void RFDetrDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qRFDetr"), QStringLiteral("lastModelDir"),
            RFDetrHelpers::modelCacheDir());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select RF-DETR GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    m_customModelPath->setText(path);
    m_customModelRow->setVisible(true);
    m_modelCombo->setCurrentIndex(-1);
    m_modelCombo->addItem(QFileInfo(path).fileName(), path);
    m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
    m_liveWidget->setModelPath(path);
}

void RFDetrDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qRFDetr"),
            QStringLiteral("lastImageFileDir"), QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qRFDetr"),
                         QStringLiteral("lastImageFileDir"), path);
    updateImagePreview();
}

void RFDetrDialog::updateImagePreview() {
    const QString path = m_imagePath->text().trimmed();
    QImage img;
    if (path.startsWith(QStringLiteral("db://"))) {
        // DB-tree entity: look up the stored full-resolution image so the
        // click-to-enlarge preview works for DB inputs too.
        const QString name = path.mid(5);
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            QListWidgetItem* item = m_dbImageList->item(i);
            if (item && item->data(Qt::UserRole).toString() == name) {
                img = item->data(kDbFullImageRole).value<QImage>();
                break;
            }
        }
    } else {
        img = QImage(path);
    }
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    m_previewLabel->setPreviewImage(img, ecvAICoreUi::previewSize());
}

void RFDetrDialog::onRun() {
    if (!ensureModelAvailable(PendingAction::Run)) return;
    emit runRequested(getSettings());
}

void RFDetrDialog::onCancel() {
    cancelDownload();
    emit cancelRequested();
}

RFDetrDialog::Settings RFDetrDialog::getSettings() const {
    Settings s;
    s.modelPath = resolveModelPath();
    s.inputPath = m_imagePath->text();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    s.threshold = static_cast<float>(m_threshold->value());
    s.topK = static_cast<uint32_t>(m_topK->value());
    s.addAnnotatedImageToDb = m_addAnnotatedCheck->isChecked();
    return s;
}

void RFDetrDialog::appendLog(const QString& msg) {
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

void RFDetrDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void RFDetrDialog::setTaskStage(const QString& stage, int percent) {
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

void RFDetrDialog::enableResultButtons(bool /*hasResult*/) {
    // Reserved for future Visualize/Export buttons (aligned with
    // qFreeSplatter).
}

void RFDetrDialog::setRunning(bool running) {
    m_taskRunning = running;
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
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
}

void RFDetrDialog::setDbImages(const QList<DbImageEntry>& images) {
    m_dbImageList->clear();
    for (const DbImageEntry& e : images) {
        auto* item = new QListWidgetItem(QIcon(QPixmap::fromImage(e.preview)),
                                         e.name, m_dbImageList);
        item->setData(Qt::UserRole, e.name);
        // Full-resolution image for the click-to-enlarge preview (the icon
        // above is only a scaled thumbnail).
        item->setData(kDbFullImageRole, e.preview);
    }
}

void RFDetrDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    m_imagePath->setText(QStringLiteral("db://") + imageNames.first());
    updateImagePreview();
}

void RFDetrDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_imagePath->setText(QStringLiteral("db://") +
                         item->data(Qt::UserRole).toString());
    updateImagePreview();
}

void RFDetrDialog::onLiveStart() {
    if (!m_liveWidget) return;
    if (!ensureModelAvailable(PendingAction::LiveStart)) return;
    startLiveStream();
}

void RFDetrDialog::startLiveStream() {
    if (!m_liveWidget) return;
    RFDetrLiveWidget::Config config = m_liveWidget->config();
    config.modelPath = m_liveWidget->resolveModelPath();
    config.device = m_liveWidget->deviceId();
    config.threads = m_liveWidget->threadCount();
    config.topK = static_cast<uint32_t>(m_topK->value());
    m_liveWidget->setConfig(config);

    if (m_liveWidget->inputSource() ==
        RFDetrLiveWidget::InputSource::VideoFile) {
        const QString path = m_liveWidget->videoFilePath();
        if (path.isEmpty() || !QFile::exists(path)) {
            appendLog(tr("[RF-DETR] Select a valid video file first."));
            return;
        }
        if (!m_liveWidget->startVideoFile(path)) {
            appendLog(tr("[RF-DETR] Failed to start video."));
        }
        return;
    }
    const int camIdx = m_liveWidget->selectedCameraIndex();
    if (camIdx < 0) {
        appendLog(tr("[RF-DETR] No camera available."));
        return;
    }
    if (!m_liveWidget->startCamera(camIdx)) {
        appendLog(tr("[RF-DETR] Failed to start camera %1.").arg(camIdx));
    }
}

void RFDetrDialog::onLiveStop() { m_liveWidget->stopStream(); }

void RFDetrDialog::onLiveRestart() { m_liveWidget->restartVideoFile(); }

void RFDetrDialog::onLiveCapture(const RFDetrRunResult& result) {
    emit liveCaptureReady(result);
}

// ---------------------------------------------------------------------------
// Test data — via shared ecvTestDataRepository
// ---------------------------------------------------------------------------

void RFDetrDialog::requestTestData(TestDataTarget target) {
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

bool RFDetrDialog::loadRequestedTestData() {
    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    QString fileName;
    if (m_pendingTestDataTarget == TestDataTarget::Image) {
        fileName = QString::fromLatin1(kRFDetrTestImage);
    } else if (m_pendingTestDataTarget == TestDataTarget::Video &&
               m_testVideoCombo) {
        fileName = m_testVideoCombo->currentData().toString();
    }
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(kind, fileName);
    if (path.isEmpty()) return false;

    if (m_pendingTestDataTarget == TestDataTarget::Image) {
        m_imagePath->setText(path);
        updateImagePreview();
        appendLog(tr("[Test data] Loaded image: %1").arg(path));
    } else if (m_pendingTestDataTarget == TestDataTarget::Video &&
               m_liveWidget) {
        m_liveWidget->setInputSource(RFDetrLiveWidget::InputSource::VideoFile);
        m_liveWidget->setVideoFilePath(path, false);
        appendLog(tr("[Test data] Loaded video: %1").arg(path));
        appendLog(tr("[Test data] Press Start to run detection on it."));
    }
    return true;
}

void RFDetrDialog::onTestDataDownloadFinished(
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

void RFDetrDialog::onTestDataExtractionFinished(
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

void RFDetrDialog::setTestDataControlsEnabled(bool enabled) {
    if (m_imageTestDataBtn) m_imageTestDataBtn->setEnabled(enabled);
    if (m_testDataBtn) m_testDataBtn->setEnabled(enabled);
    if (m_testVideoCombo) m_testVideoCombo->setEnabled(enabled);
}

void RFDetrDialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    m_liveWidget->saveSettings();
    event->accept();
}

void RFDetrDialog::changeEvent(QEvent* event) {
    QDialog::changeEvent(event);
    if (event->type() == QEvent::ActivationChange) {
        adaptTabWidgetHeight();
    }
}

void RFDetrDialog::adaptTabWidgetHeight() {
    // Keep the dialog compact on small screens; the live tab has its own
    // fixed preview height.
    if (m_activeTabHeight < 0) {
        m_activeTabHeight = m_tabWidget->height();
    }
}
