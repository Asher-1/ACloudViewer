// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGDialog.h"

#include <cvFileDialog.h>

#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QScrollArea>
#include <QSettings>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/inference_log.h"
#include "aicore/rmbg_capi.h"
#endif

namespace {
const int kThumbSize = 96;
constexpr const char* kRMBGTestImage = "friends1.jpg";

void styleSampleDataButton(QPushButton* button) {
    button->setStyleSheet(
            "QPushButton { background: #00897b; color: white; font-weight: "
            "bold; border: none; border-radius: 4px; padding: 5px 12px; }"
            "QPushButton:hover { background: #00796b; }"
            "QPushButton:pressed { background: #00695c; }"
            "QPushButton:disabled { background: #b2dfdb; color: #e0f2f1; }");
}
}  // namespace

RMBGDialog::RMBGDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RMBG Background Removal"));
    setMinimumSize(680, 560);
    setupUi();
    populateModelCombo();
    loadSettings();
    m_liveWidget->loadSettings();
}

RMBGDialog::~RMBGDialog() {
    saveSettings();
    m_liveWidget->saveSettings();
}

void RMBGDialog::setupUi() {
    auto* rootLayout = new QVBoxLayout(this);
    m_tabWidget = new QTabWidget(this);
    rootLayout->addWidget(m_tabWidget);

    // ---- Image tab --------------------------------------------------------
    m_imageTab = new QWidget(this);
    auto* imageLayout = new QVBoxLayout(m_imageTab);

    auto* modelRow = new QHBoxLayout;
    modelRow->addWidget(new QLabel(tr("Model:"), m_imageTab));
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
    customRow->addWidget(new QLabel(tr("Custom GGUF:"), m_customModelRow));
    m_customModelPath = new QLineEdit(m_customModelRow);
    customRow->addWidget(m_customModelPath, 1);
    auto* browseCustomBtn = new QPushButton(tr("Browse…"), m_customModelRow);
    connect(browseCustomBtn, &QPushButton::clicked, this,
            &RMBGDialog::onBrowseCustomModel);
    customRow->addWidget(browseCustomBtn);
    imageLayout->addWidget(m_customModelRow);

    auto* runRow = new QHBoxLayout;
    runRow->addWidget(new QLabel(tr("Device:"), m_imageTab));
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
    runRow->addWidget(m_deviceCombo);

    runRow->addWidget(new QLabel(tr("Threads:"), m_imageTab));
    m_threads = new QSpinBox(m_imageTab);
    m_threads->setRange(0, 64);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = auto"));
    runRow->addWidget(m_threads);

    runRow->addWidget(new QLabel(tr("Alpha Threshold:"), m_imageTab));
    m_alphaThreshold = new QDoubleSpinBox(m_imageTab);
    m_alphaThreshold->setRange(0.0, 1.0);
    m_alphaThreshold->setSingleStep(0.05);
    m_alphaThreshold->setValue(0.5);
    m_alphaThreshold->setToolTip(
            tr("Pixels below this alpha become transparent (0 disables)"));
    runRow->addWidget(m_alphaThreshold);
    runRow->addStretch();
    imageLayout->addLayout(runRow);

    auto* inputRow = new QHBoxLayout;
    inputRow->addWidget(new QLabel(tr("Image:"), m_imageTab));
    m_imagePath = new QLineEdit(m_imageTab);
    inputRow->addWidget(m_imagePath, 1);
    auto* browseBtn = new QPushButton(tr("Browse…"), m_imageTab);
    connect(browseBtn, &QPushButton::clicked, this, &RMBGDialog::onBrowseImage);
    m_imageTestDataBtn =
            new QPushButton(tr("\U0001f9ea  Try sample data"), m_imageTab);
    styleSampleDataButton(m_imageTestDataBtn);
    m_imageTestDataBtn->setToolTip(
            tr("Load friends1.jpg from the FriendsFaces test-data cache"));
    connect(m_imageTestDataBtn, &QPushButton::clicked, this,
            [this]() { requestTestData(TestDataTarget::Image); });
    inputRow->addWidget(browseBtn);
    imageLayout->addLayout(inputRow);

    // DB image picker (collapsible).
    m_dbToggleBtn = new QToolButton(m_imageTab);
    m_dbToggleBtn->setText(tr("DB images ▾"));
    m_dbToggleBtn->setCheckable(true);
    m_dbToggleBtn->setChecked(false);
    imageLayout->addWidget(m_dbToggleBtn, 0, Qt::AlignLeft);
    m_dbContentWidget = new QWidget(m_imageTab);
    auto* dbLayout = new QVBoxLayout(m_dbContentWidget);
    dbLayout->setContentsMargins(0, 0, 0, 0);
    m_dbImageList = new QListWidget(m_dbContentWidget);
    m_dbImageList->setIconSize(QSize(48, 48));
    m_dbImageList->setMaximumHeight(140);
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
    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool on) {
        m_dbContentWidget->setVisible(on);
        if (on) emit refreshDbImagesRequested();
    });
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &RMBGDialog::onDbListActivated);
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &RMBGDialog::onDbListActivated);

    m_previewLabel = new ecvClickableImageLabel(m_imageTab);
    m_previewLabel->setFixedSize(kThumbSize, kThumbSize);
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabel->setText(tr("Preview"));
    imageLayout->addWidget(
            ecvClickableImageLabel::wrapWithTapToPreviewHint(m_previewLabel));

    m_downloadLabel = new QLabel(this);
    m_downloadLabel->setWordWrap(true);
    m_downloadLabel->setVisible(false);

    m_taskStatusLabel = new QLabel(m_imageTab);
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    imageLayout->addWidget(m_taskStatusLabel);

    auto* outputRow = new QHBoxLayout;
    m_addDbCheck = new QCheckBox(tr("Add result to DB"), m_imageTab);
    m_addDbCheck->setChecked(true);
    outputRow->addWidget(m_addDbCheck);
    m_savePngCheck = new QCheckBox(tr("Save PNG to:"), m_imageTab);
    m_savePngCheck->setChecked(false);
    outputRow->addWidget(m_savePngCheck);
    m_savePngDir = new QLineEdit(m_imageTab);
    m_savePngDir->setEnabled(false);
    m_savePngDir->setPlaceholderText(tr("output directory"));
    outputRow->addWidget(m_savePngDir, 1);
    auto* browseSaveDirBtn = new QPushButton(tr("Browse…"), m_imageTab);
    browseSaveDirBtn->setEnabled(false);
    connect(browseSaveDirBtn, &QPushButton::clicked, this,
            &RMBGDialog::onBrowseSaveDir);
    outputRow->addWidget(browseSaveDirBtn);
    connect(m_savePngCheck, &QCheckBox::toggled, this,
            [this, browseSaveDirBtn](bool on) {
                m_savePngDir->setEnabled(on);
                browseSaveDirBtn->setEnabled(on);
            });
    imageLayout->addLayout(outputRow);

    auto* actionRow = new QHBoxLayout;
    actionRow->addStretch();
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
    m_liveWidget = new RMBGLiveWidget(m_liveTab);
    liveLayout->addWidget(m_liveWidget, 1);

    // Playback controls live in the Live tab itself (mirrors qFaceDetect).
    auto* liveBtnRow = new QHBoxLayout;
    m_testVideoCombo = new QComboBox(m_liveTab);
    m_testVideoCombo->addItem(QStringLiteral("friends_demo.mp4"),
                              QStringLiteral("friends_demo.mp4"));
    m_testDataBtn =
            new QPushButton(tr("\U0001f9ea  Try sample data"), m_liveTab);
    styleSampleDataButton(m_testDataBtn);
    m_testDataBtn->setToolTip(
            tr("Load the selected video from the FriendsFaces test-data cache"));
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
            this, &RMBGDialog::onModelComboChanged);
    connect(m_runBtn, &QPushButton::clicked, this, &RMBGDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &RMBGDialog::onCancel);
    connect(m_liveStartBtn, &QPushButton::clicked, this,
            &RMBGDialog::onLiveStart);
    connect(m_liveStopBtn, &QPushButton::clicked, this,
            &RMBGDialog::onLiveStop);
    connect(m_liveRestartBtn, &QPushButton::clicked, this,
            &RMBGDialog::onLiveRestart);
    connect(m_testDataBtn, &QPushButton::clicked, this,
            [this]() { requestTestData(TestDataTarget::Video); });

    // Surface live-stream messages (model missing, inference failures…) in
    // the shared log. Without this connection the Live tab failed silently:
    // a failing inference looked exactly like "inference never runs".
    connect(m_liveWidget, &RMBGLiveWidget::logMessage, this,
            &RMBGDialog::appendLog);

    // Keep the live button states in sync with the stream lifecycle.
    connect(m_liveWidget, &RMBGLiveWidget::streamStarted, this, [this]() {
        m_liveStartBtn->setEnabled(false);
        m_liveStopBtn->setEnabled(true);
        m_liveRestartBtn->setEnabled(m_liveWidget->inputSource() ==
                                     RMBGLiveWidget::InputSource::VideoFile);
    });
    connect(m_liveWidget, &RMBGLiveWidget::streamStopped, this, [this]() {
        m_liveStartBtn->setEnabled(true);
        m_liveStopBtn->setEnabled(false);
        if (m_liveWidget->inputSource() !=
            RMBGLiveWidget::InputSource::VideoFile) {
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
    connect(m_liveWidget, &RMBGLiveWidget::modelSelectionChanged, this,
            [this](const QString& filename) {
                const int index = m_modelCombo->findData(filename);
                if (index < 0 || index == m_modelCombo->currentIndex()) return;
                const bool restartStream = m_liveWidget->isActive();
                if (restartStream) m_liveWidget->stopStream();
                m_modelCombo->setCurrentIndex(index);
                if (restartStream) onLiveStart();
            });
    connect(m_liveWidget, &RMBGLiveWidget::deviceSelectionChanged, this,
            [this](const QString& device) {
                const int index = m_deviceCombo->findData(device);
                if (index >= 0 && index != m_deviceCombo->currentIndex()) {
                    m_deviceCombo->setCurrentIndex(index);
                }
            });
    connect(m_liveWidget, &RMBGLiveWidget::threadCountChanged, this,
            [this](int threads) {
                if (m_threads->value() != threads) m_threads->setValue(threads);
            });
    connect(m_liveWidget, &RMBGLiveWidget::captureToDbRequested, this,
            &RMBGDialog::onLiveCapture);

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
            &RMBGDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[RMBG] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[RMBG] Model downloaded: %1").arg(path));
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
    rootLayout->addWidget(m_downloadLabel);
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    rootLayout->addWidget(m_progress);

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
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataDownloadFinished(success, kind);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                m_progress->setRange(0, total);
                m_progress->setValue(current);
                m_progress->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                onTestDataExtractionFinished(success, kind);
            });
}

void RMBGDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void RMBGDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRMBG"));
    const QString modelFilename =
            settings.value(QStringLiteral("modelFilename")).toString();
    selectModelByFilename(modelFilename);
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    m_alphaThreshold->setValue(
            settings.value(QStringLiteral("alphaThreshold"), 0.5).toDouble());
    const QString imagePath =
            settings.value(QStringLiteral("imagePath")).toString();
    if (!imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
        updateImagePreview();
    }
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_savePngCheck->setChecked(
            settings.value(QStringLiteral("savePng"), false).toBool());
    m_savePngDir->setText(
            settings.value(QStringLiteral("savePngDir")).toString());
    settings.endGroup();
}

void RMBGDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRMBG"));
    settings.setValue(QStringLiteral("modelFilename"),
                      m_modelCombo->currentData().toString());
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.setValue(QStringLiteral("alphaThreshold"),
                      m_alphaThreshold->value());
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("savePng"), m_savePngCheck->isChecked());
    settings.setValue(QStringLiteral("savePngDir"), m_savePngDir->text());
    settings.endGroup();
}

QString RMBGDialog::modelCacheDir() { return RMBGHelpers::modelCacheDir(); }

void RMBGDialog::populateModelCombo(const QString& keepFilename) {
    const QVector<RMBGModelEntry> models = RMBGHelpers::catalogModels();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const RMBGModelEntry& e : models) {
        m_modelCombo->addItem(RMBGHelpers::modelDisplayLabel(e), e.filename);
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

bool RMBGDialog::selectModelByFilename(const QString& filename) {
    if (filename.isEmpty()) return false;
    const int idx = m_modelCombo->findData(filename);
    if (idx < 0) return false;
    m_modelCombo->setCurrentIndex(idx);
    return true;
}

void RMBGDialog::refreshModelList() {
    const QString keep = m_modelCombo->currentData().toString();
    populateModelCombo(keep);
}

void RMBGDialog::onModelComboChanged(int index) {
    const QString filename = m_modelCombo->itemData(index).toString();
    const bool isCustom =
            filename.isEmpty() ||
            filename.endsWith(QStringLiteral(".gguf")) &&
                    !RMBGHelpers::findModelByFilename(filename, nullptr);
    m_customModelRow->setVisible(isCustom);
    m_liveWidget->setModelPath(resolveModelPath());
}

QString RMBGDialog::resolveModelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = RMBGHelpers::modelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

bool RMBGDialog::ensureModelAvailable(PendingAction action) {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[RMBG] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath())) {
        RMBGModelEntry entry;
        if (!RMBGHelpers::findModelByFilename(filename, &entry)) {
            appendLog(tr("[RMBG] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[RMBG] Model missing — downloading %1; it will start "
                     "automatically when ready.")
                          .arg(filename));
        startDownload(entry);
        return false;
    }
    return true;
}

void RMBGDialog::startDownload(const RMBGModelEntry& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[RMBG] A download is already running."));
        return;
    }
    QDir().mkpath(RMBGHelpers::modelCacheDir());
    const QString dest =
            RMBGHelpers::modelCacheDir() + QDir::separator() + model.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[RMBG] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[RMBG] Downloading %1 (%2)…")
                      .arg(model.filename, model.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // RMBG-2.0 GGUF is tens of MB
    m_downloader->download(req);
}

void RMBGDialog::cancelDownload() {
    if (m_downloadInProgress) m_downloader->cancel();
}

void RMBGDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qRMBG"),
                                             QStringLiteral("lastModelDir"),
                                             RMBGHelpers::modelCacheDir());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select RMBG GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    m_customModelPath->setText(path);
    m_customModelRow->setVisible(true);
    m_modelCombo->setCurrentIndex(-1);
    m_modelCombo->addItem(QFileInfo(path).fileName(), path);
    m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
    m_liveWidget->setModelPath(path);
}

void RMBGDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(settings, QStringLiteral("qRMBG"),
                                             QStringLiteral("lastImageFileDir"),
                                             QDir::homePath());
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All "
               "files (*)"));
    if (path.isEmpty()) return;
    m_imagePath->setText(path);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qRMBG"),
                         QStringLiteral("lastImageFileDir"), path);
    updateImagePreview();
}

void RMBGDialog::onBrowseSaveDir() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qRMBG"),
                             QStringLiteral("lastSaveDir"), QDir::homePath());
    const QString dir = cvFileDialog::getExistingDirectory(
            this, tr("Select output directory"), lastDir);
    if (dir.isEmpty()) return;
    m_savePngDir->setText(dir);
    ecvPS::saveBrowseDir(settings, QStringLiteral("qRMBG"),
                         QStringLiteral("lastSaveDir"), dir);
}

void RMBGDialog::updateImagePreview() {
    const QImage img(m_imagePath->text());
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    m_previewLabel->setPreviewImage(img, kThumbSize);
}

void RMBGDialog::onRun() {
    if (!ensureModelAvailable(PendingAction::Run)) return;
    emit runRequested(getSettings());
}

void RMBGDialog::onCancel() {
    cancelDownload();
    emit cancelRequested();
}

RMBGDialog::Settings RMBGDialog::getSettings() const {
    Settings s;
    s.modelPath = resolveModelPath();
    s.inputPath = m_imagePath->text();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    s.alphaThreshold = static_cast<float>(m_alphaThreshold->value());
    s.addResultToDb = m_addDbCheck->isChecked();
    if (m_savePngCheck->isChecked()) {
        s.savePngDir = m_savePngDir->text().trimmed();
    }
    return s;
}

void RMBGDialog::appendLog(const QString& msg) {
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

void RMBGDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void RMBGDialog::setTaskStage(const QString& stage, int percent) {
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

void RMBGDialog::enableResultButtons(bool /*hasResult*/) {
    // Reserved for future Visualize/Export buttons (aligned with
    // qFreeSplatter).
}

void RMBGDialog::setRunning(bool running) {
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

void RMBGDialog::setDbImages(const QList<DbImageEntry>& images) {
    m_dbImageList->clear();
    for (const DbImageEntry& e : images) {
        auto* item = new QListWidgetItem(QIcon(QPixmap::fromImage(e.preview)),
                                         e.name, m_dbImageList);
        item->setData(Qt::UserRole, e.name);
    }
}

void RMBGDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    m_imagePath->setText(QStringLiteral("db://") + imageNames.first());
    updateImagePreview();
}

void RMBGDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_imagePath->setText(QStringLiteral("db://") +
                         item->data(Qt::UserRole).toString());
    updateImagePreview();
}

void RMBGDialog::onLiveStart() {
    if (!m_liveWidget) return;
    if (!ensureModelAvailable(PendingAction::LiveStart)) return;
    startLiveStream();
}

void RMBGDialog::startLiveStream() {
    if (!m_liveWidget) return;
    RMBGLiveWidget::Config config = m_liveWidget->config();
    config.modelPath = m_liveWidget->resolveModelPath();
    config.device = m_liveWidget->deviceId();
    config.threads = m_liveWidget->threadCount();
    config.alphaThreshold = static_cast<float>(m_alphaThreshold->value());
    m_liveWidget->setConfig(config);

    if (m_liveWidget->inputSource() == RMBGLiveWidget::InputSource::VideoFile) {
        const QString path = m_liveWidget->videoFilePath();
        if (path.isEmpty() || !QFile::exists(path)) {
            appendLog(tr("[RMBG] Select a valid video file first."));
            return;
        }
        if (!m_liveWidget->startVideoFile(path)) {
            appendLog(tr("[RMBG] Failed to start video."));
        }
        return;
    }
    const int camIdx = m_liveWidget->selectedCameraIndex();
    if (camIdx < 0) {
        appendLog(tr("[RMBG] No camera available."));
        return;
    }
    if (!m_liveWidget->startCamera(camIdx)) {
        appendLog(tr("[RMBG] Failed to start camera %1.").arg(camIdx));
    }
}

void RMBGDialog::onLiveStop() { m_liveWidget->stopStream(); }

void RMBGDialog::onLiveRestart() { m_liveWidget->restartVideoFile(); }

void RMBGDialog::onLiveCapture(const RMBGRunResult& result) {
    emit liveCaptureReady(result);
}

void RMBGDialog::requestTestData(TestDataTarget target) {
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

    const auto kind = ecvTestDataRepository::Dataset::FriendsFaces;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    setTestDataControlsEnabled(false);
    m_downloadLabel->setVisible(true);
    m_progress->setVisible(true);
    if (ecvTestDataRepository::verifyZipIntegrity(
                ecvTestDataRepository::zipPath(kind), info.expectedMd5,
                info.expectedSize)) {
        m_downloadLabel->setText(tr("Extracting cached test data..."));
        m_progress->setRange(0, 0);
        repo.extractDataset(kind);
        return;
    }

    m_downloadLabel->setText(tr("Downloading FriendsFaces test data..."));
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    repo.startDownload(kind);
}

bool RMBGDialog::loadRequestedTestData() {
    const auto kind = ecvTestDataRepository::Dataset::FriendsFaces;
    QString fileName;
    if (m_pendingTestDataTarget == TestDataTarget::Image) {
        fileName = QString::fromLatin1(kRMBGTestImage);
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
        m_liveWidget->setInputSource(RMBGLiveWidget::InputSource::VideoFile);
        m_liveWidget->setVideoFilePath(path, false);
        appendLog(tr("[Test data] Loaded video: %1").arg(path));
        appendLog(tr("[Test data] Press Start to remove its background."));
    }
    return true;
}

void RMBGDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::FriendsFaces) {
        return;
    }
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        m_testDataDownloadInProgress = false;
        m_pendingTestDataTarget = TestDataTarget::None;
        setTestDataControlsEnabled(true);
        m_downloadLabel->setVisible(false);
        m_progress->setVisible(false);
        return;
    }

    m_downloadLabel->setText(tr("Extracting FriendsFaces test data..."));
    m_progress->setRange(0, 0);
    ecvTestDataRepository::instance().extractDataset(kind);
}

void RMBGDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::FriendsFaces) {
        return;
    }
    m_testDataDownloadInProgress = false;
    setTestDataControlsEnabled(true);
    m_downloadLabel->setVisible(false);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);

    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
    } else if (!loadRequestedTestData()) {
        appendLog(
                tr("[Test data] Requested file was not found in the archive."));
    }
    m_pendingTestDataTarget = TestDataTarget::None;
}

void RMBGDialog::setTestDataControlsEnabled(bool enabled) {
    if (m_imageTestDataBtn) m_imageTestDataBtn->setEnabled(enabled);
    if (m_testDataBtn) m_testDataBtn->setEnabled(enabled);
    if (m_testVideoCombo) m_testVideoCombo->setEnabled(enabled);
}

void RMBGDialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    m_liveWidget->saveSettings();
    event->accept();
}

void RMBGDialog::changeEvent(QEvent* event) {
    QDialog::changeEvent(event);
    if (event->type() == QEvent::ActivationChange) {
        adaptTabWidgetHeight();
    }
}

void RMBGDialog::adaptTabWidgetHeight() {
    // Keep the dialog compact on small screens; the live tab has its own
    // fixed preview height.
    if (m_activeTabHeight < 0) {
        m_activeTabHeight = m_tabWidget->height();
    }
}
