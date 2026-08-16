// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrDialog.h"

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
#include <QVBoxLayout>

#include <cvFileDialog.h>

#include "ecvPersistentSettings.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rfdetr_capi.h"
#endif

RFDetrDialog::RFDetrDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("RF-DETR Object Detection"));
    setMinimumSize(680, 560);
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
    m_tabWidget = new QTabWidget(this);
    rootLayout->addWidget(m_tabWidget);

    // ---- Image tab --------------------------------------------------------
    m_imageTab = new QWidget(this);
    auto* imageLayout = new QVBoxLayout(m_imageTab);

    auto* modelRow = new QHBoxLayout;
    modelRow->addWidget(new QLabel(tr("Model:"), m_imageTab));
    m_modelCombo = new QComboBox(m_imageTab);
    m_modelCombo->setMinimumWidth(260);
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
            &RFDetrDialog::onBrowseCustomModel);
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

    runRow->addWidget(new QLabel(tr("Threshold:"), m_imageTab));
    m_threshold = new QDoubleSpinBox(m_imageTab);
    m_threshold->setRange(0.01, 1.0);
    m_threshold->setSingleStep(0.05);
    m_threshold->setValue(0.5);
    runRow->addWidget(m_threshold);

    runRow->addWidget(new QLabel(tr("Top-K:"), m_imageTab));
    m_topK = new QSpinBox(m_imageTab);
    m_topK->setRange(1, 1000);
    m_topK->setValue(300);
    runRow->addWidget(m_topK);
    runRow->addStretch();
    imageLayout->addLayout(runRow);

    auto* inputRow = new QHBoxLayout;
    inputRow->addWidget(new QLabel(tr("Image:"), m_imageTab));
    m_imagePath = new QLineEdit(m_imageTab);
    inputRow->addWidget(m_imagePath, 1);
    auto* browseBtn = new QPushButton(tr("Browse…"), m_imageTab);
    connect(browseBtn, &QPushButton::clicked, this, &RFDetrDialog::onBrowseImage);
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
    m_dbContentWidget->setVisible(false);
    imageLayout->addWidget(m_dbContentWidget);
    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool on) {
        m_dbContentWidget->setVisible(on);
        if (on) emit refreshDbImagesRequested();
    });
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &RFDetrDialog::onDbListActivated);
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &RFDetrDialog::onDbListActivated);

    m_previewLabel = new ecvClickableImageLabel(m_imageTab);
    m_previewLabel->setMinimumHeight(200);
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setText(tr("No image selected"));
    imageLayout->addWidget(m_previewLabel, 1);

    m_hintLabel = new QLabel(m_imageTab);
    m_hintLabel->setWordWrap(true);
    m_hintLabel->setStyleSheet(QStringLiteral("color: gray;"));
    imageLayout->addWidget(m_hintLabel);

    m_downloadLabel = new QLabel(m_imageTab);
    m_downloadLabel->setWordWrap(true);
    m_downloadLabel->setVisible(false);
    imageLayout->addWidget(m_downloadLabel);

    m_progress = new QProgressBar(m_imageTab);
    m_progress->setRange(0, 100);
    m_progress->setValue(0);
    m_progress->setVisible(false);
    imageLayout->addWidget(m_progress);

    auto* actionRow = new QHBoxLayout;
    m_addAnnotatedCheck = new QCheckBox(tr("Add annotated image to DB"),
                                        m_imageTab);
    m_addAnnotatedCheck->setChecked(true);
    actionRow->addWidget(m_addAnnotatedCheck);
    actionRow->addStretch();
    m_runBtn = new QPushButton(tr("Run"), m_imageTab);
    m_runBtn->setDefault(true);
    actionRow->addWidget(m_runBtn);
    m_cancelBtn = new QPushButton(tr("Cancel"), m_imageTab);
    m_cancelBtn->setEnabled(false);
    actionRow->addWidget(m_cancelBtn);
    imageLayout->addLayout(actionRow);

    m_tabWidget->addTab(m_imageTab, tr("Image"));

    // ---- Live (camera / video) tab ----------------------------------------
    m_liveWidget = new RFDetrLiveWidget(this);
    m_tabWidget->addTab(m_liveWidget, tr("Live (camera / video)"));

    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RFDetrDialog::onModelComboChanged);
    connect(m_runBtn, &QPushButton::clicked, this, &RFDetrDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &RFDetrDialog::onCancel);

    // Keep the live tab's model/device/threads controls in sync.
    m_liveWidget->syncModelControlsFrom(m_modelCombo, m_deviceCombo, m_threads);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            m_liveWidget, [this](int) {
                m_liveWidget->setModelPath(resolveModelPath());
            });
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            m_liveWidget, [this](int) {
                m_liveWidget->setDevice(
                        m_deviceCombo->currentData().toString());
            });
    connect(m_threads, QOverload<int>::of(&QSpinBox::valueChanged),
            m_liveWidget, [this](int v) { m_liveWidget->setThreads(v); });
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
                            ecvModelDownloader::formatDownloadProgress(
                                    received, total));
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
                if (m_autoRunAfterDownload) {
                    m_autoRunAfterDownload = false;
                    onRun();
                }
            });
}

void RFDetrDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRFDetr"));
    const QString modelFilename = settings.value(
            QStringLiteral("modelFilename")).toString();
    selectModelByFilename(modelFilename);
    const QString device = settings.value(
            QStringLiteral("device"), QStringLiteral("auto")).toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    m_threshold->setValue(
            settings.value(QStringLiteral("threshold"), 0.5).toDouble());
    m_topK->setValue(settings.value(QStringLiteral("topK"), 300).toInt());
    const QString imagePath = settings.value(
            QStringLiteral("imagePath")).toString();
    if (!imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
        updateImagePreview();
    }
    m_addAnnotatedCheck->setChecked(settings.value(
            QStringLiteral("addAnnotated"), true).toBool());
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

QString RFDetrDialog::modelCacheDir() {
    return RFDetrHelpers::modelCacheDir();
}

void RFDetrDialog::populateModelCombo(const QString& keepFilename) {
    const QVector<RFDetrModelEntry> models = RFDetrHelpers::catalogModels();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const RFDetrModelEntry& e : models) {
        QString label = e.displayName;
        if (!e.quantNote.isEmpty()) {
            label += QStringLiteral(" — ") + e.quantNote;
        }
        m_modelCombo->addItem(label, e.filename);
    }
    if (!keepFilename.isEmpty()) {
        const int idx = m_modelCombo->findData(keepFilename);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    m_modelCombo->blockSignals(false);
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
    const bool isCustom = filename.isEmpty() ||
                          filename.endsWith(QStringLiteral(".gguf")) &&
                                  !RFDetrHelpers::findModelByFilename(
                                          filename, nullptr);
    m_customModelRow->setVisible(isCustom);
    RFDetrModelEntry entry;
    const bool known = RFDetrHelpers::findModelByFilename(filename, &entry);
    if (known) {
        m_hintLabel->setText(entry.licenseNote);
    } else {
        m_hintLabel->setText(tr("Select a catalog model or browse a custom "
                                "GGUF file."));
    }
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

bool RFDetrDialog::ensureModelAvailable() {
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
        m_autoRunAfterDownload = false;
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
    const QString dest = RFDetrHelpers::modelCacheDir() + QDir::separator() +
                         model.filename;
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
            settings, QStringLiteral("qRFDetr"),
            QStringLiteral("lastModelDir"), RFDetrHelpers::modelCacheDir());
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
    const QImage img(m_imagePath->text());
    if (img.isNull()) return;
    const QPixmap pix = QPixmap::fromImage(img).scaled(
            m_previewLabel->size(), Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
    m_previewLabel->setPixmap(pix);
}

void RFDetrDialog::onRun() {
    if (!ensureModelAvailable()) return;
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
    // Surface log lines in the status hint (kept short) — the DB console is
    // owned by the app; plugins route through it via ecvMainAppInterface.
    if (m_hintLabel && !msg.isEmpty()) {
        m_hintLabel->setText(msg);
    }
}

void RFDetrDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void RFDetrDialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    m_progress->setVisible(running);
    if (!running) m_progress->setValue(0);
}

void RFDetrDialog::setDbImages(const QList<DbImageEntry>& images) {
    m_dbImageList->clear();
    for (const DbImageEntry& e : images) {
        auto* item = new QListWidgetItem(QIcon(QPixmap::fromImage(e.preview)),
                                         e.name, m_dbImageList);
        item->setData(Qt::UserRole, e.name);
    }
}

void RFDetrDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    m_imagePath->setText(QStringLiteral("db://") + imageNames.first());
    updateImagePreview();
}

void RFDetrDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_imagePath->setText(QStringLiteral("db://") + item->data(Qt::UserRole).toString());
    updateImagePreview();
}

void RFDetrDialog::onLiveStart() {
    // Live tab has its own Start button inside VideoPlaybackWidget; keep the
    // dialog-level controls in sync with the batch settings.
    m_liveWidget->setConfig({resolveModelPath(),
                             m_deviceCombo->currentData().toString(),
                             m_threads->value(),
                             static_cast<float>(m_threshold->value()),
                             static_cast<uint32_t>(m_topK->value())});
}

void RFDetrDialog::onLiveStop() {
    m_liveWidget->stopStream();
}

void RFDetrDialog::onLiveCapture(const RFDetrRunResult& result) {
    emit liveCaptureReady(result);
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
