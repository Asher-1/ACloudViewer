// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGDialog.h"

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
#include <QVBoxLayout>

#include <cvFileDialog.h>

#include "ecvPersistentSettings.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/rmbg_capi.h"
#endif

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
    runRow->addStretch();
    imageLayout->addLayout(runRow);

    auto* inputRow = new QHBoxLayout;
    inputRow->addWidget(new QLabel(tr("Image:"), m_imageTab));
    m_imagePath = new QLineEdit(m_imageTab);
    inputRow->addWidget(m_imagePath, 1);
    auto* browseBtn = new QPushButton(tr("Browse…"), m_imageTab);
    connect(browseBtn, &QPushButton::clicked, this, &RMBGDialog::onBrowseImage);
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
            &RMBGDialog::onDbListActivated);
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &RMBGDialog::onDbListActivated);

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
    m_runBtn = new QPushButton(tr("Run"), m_imageTab);
    m_runBtn->setDefault(true);
    actionRow->addWidget(m_runBtn);
    m_cancelBtn = new QPushButton(tr("Cancel"), m_imageTab);
    m_cancelBtn->setEnabled(false);
    actionRow->addWidget(m_cancelBtn);
    imageLayout->addLayout(actionRow);

    m_tabWidget->addTab(m_imageTab, tr("Image"));

    // ---- Live (camera / video) tab ----------------------------------------
    m_liveWidget = new RMBGLiveWidget(this);
    m_tabWidget->addTab(m_liveWidget, tr("Live (camera / video)"));

    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &RMBGDialog::onModelComboChanged);
    connect(m_runBtn, &QPushButton::clicked, this, &RMBGDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &RMBGDialog::onCancel);

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
                            ecvModelDownloader::formatDownloadProgress(
                                    received, total));
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
                if (m_autoRunAfterDownload) {
                    m_autoRunAfterDownload = false;
                    onRun();
                }
            });
}

void RMBGDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qRMBG"));
    const QString modelFilename = settings.value(
            QStringLiteral("modelFilename")).toString();
    selectModelByFilename(modelFilename);
    const QString device = settings.value(
            QStringLiteral("device"), QStringLiteral("auto")).toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    const QString imagePath = settings.value(
            QStringLiteral("imagePath")).toString();
    if (!imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
        updateImagePreview();
    }
    m_addDbCheck->setChecked(settings.value(
            QStringLiteral("addToDb"), true).toBool());
    m_savePngCheck->setChecked(
            settings.value(QStringLiteral("savePng"), false).toBool());
    m_savePngDir->setText(settings.value(
            QStringLiteral("savePngDir")).toString());
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
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("savePng"), m_savePngCheck->isChecked());
    settings.setValue(QStringLiteral("savePngDir"), m_savePngDir->text());
    settings.endGroup();
}

QString RMBGDialog::modelCacheDir() {
    return RMBGHelpers::modelCacheDir();
}

void RMBGDialog::populateModelCombo(const QString& keepFilename) {
    const QVector<RMBGModelEntry> models = RMBGHelpers::catalogModels();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const RMBGModelEntry& e : models) {
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
    const bool isCustom = filename.isEmpty() ||
                          filename.endsWith(QStringLiteral(".gguf")) &&
                                  !RMBGHelpers::findModelByFilename(
                                          filename, nullptr);
    m_customModelRow->setVisible(isCustom);
    RMBGModelEntry entry;
    const bool known = RMBGHelpers::findModelByFilename(filename, &entry);
    if (known) {
        m_hintLabel->setText(entry.licenseNote);
    } else {
        m_hintLabel->setText(tr("Select a catalog model or browse a custom "
                                "GGUF file."));
    }
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

bool RMBGDialog::ensureModelAvailable() {
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
        m_autoRunAfterDownload = false;
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
    const QString dest = RMBGHelpers::modelCacheDir() + QDir::separator() +
                         model.filename;
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
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qRMBG"),
            QStringLiteral("lastModelDir"), RMBGHelpers::modelCacheDir());
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
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qRMBG"),
            QStringLiteral("lastImageFileDir"), QDir::homePath());
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
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qRMBG"),
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
    if (img.isNull()) return;
    const QPixmap pix = QPixmap::fromImage(img).scaled(
            m_previewLabel->size(), Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
    m_previewLabel->setPixmap(pix);
}

void RMBGDialog::onRun() {
    if (!ensureModelAvailable()) return;
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
    s.addResultToDb = m_addDbCheck->isChecked();
    if (m_savePngCheck->isChecked()) {
        s.savePngDir = m_savePngDir->text().trimmed();
    }
    return s;
}

void RMBGDialog::appendLog(const QString& msg) {
    // Surface log lines in the status hint (kept short) — the DB console is
    // owned by the app; plugins route through it via ecvMainAppInterface.
    if (m_hintLabel && !msg.isEmpty()) {
        m_hintLabel->setText(msg);
    }
}

void RMBGDialog::setProgress(int current, int total) {
    m_progress->setVisible(true);
    m_progress->setRange(0, total > 0 ? total : 1);
    m_progress->setValue(current);
}

void RMBGDialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
    m_progress->setVisible(running);
    if (!running) m_progress->setValue(0);
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
    m_imagePath->setText(QStringLiteral("db://") + item->data(Qt::UserRole).toString());
    updateImagePreview();
}

void RMBGDialog::onLiveStart() {
    // Live tab has its own Start button inside VideoPlaybackWidget; keep the
    // dialog-level controls in sync with the batch settings.
    m_liveWidget->setConfig({resolveModelPath(),
                             m_deviceCombo->currentData().toString(),
                             m_threads->value()});
}

void RMBGDialog::onLiveStop() {
    m_liveWidget->stopStream();
}

void RMBGDialog::onLiveCapture(const RMBGRunResult& result) {
    emit liveCaptureReady(result);
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
