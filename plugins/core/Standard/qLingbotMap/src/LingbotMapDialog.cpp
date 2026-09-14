// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LingbotMapDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QMessageBox>
#include <QSettings>
#include <QStandardItemModel>
#include <QVBoxLayout>

#include "aicore/asset_digests.h"
#include "aicore/lingbot_capi.h"
#include "ecvAICoreUiHelper.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"

namespace {

struct LingbotCatalogEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    QString role;
    qint64 sizeBytes = 0;
};

// Role-filtered catalog view over the AICore LingBot-Map catalog.
QVector<LingbotCatalogEntry> catalogByRole(const char* role) {
    QVector<LingbotCatalogEntry> out;
    const int n = aicore_lingbot_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_lingbot_model_entry* e = aicore_lingbot_model_at(i);
        if (!e || !e->filename || !e->role || std::strcmp(e->role, role) != 0) {
            continue;
        }
        LingbotCatalogEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.role = QString::fromUtf8(e->role);
        entry.sizeBytes = static_cast<qint64>(e->size_bytes);
        out.append(entry);
    }
    return out;
}

bool findEntryByFilename(const QVector<LingbotCatalogEntry>& catalog,
                         const QString& filename,
                         LingbotCatalogEntry* out) {
    for (const LingbotCatalogEntry& e : catalog) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString lingbotModelCacheDir() {
    char* raw = aicore_lingbot_model_cache_dir();
    if (!raw) return QString();
    const QString dir = QString::fromUtf8(raw);
    aicore_lingbot_free_buffer(raw);
    return dir;
}

}  // namespace

LingbotMapDialog::LingbotMapDialog(QWidget* parent)
    : QDialog(parent, Qt::Window) {
    setWindowTitle(tr("LingBot-Map Streaming 3D Reconstruction"));
    resize(640, 560);

    auto* rootLayout = new QVBoxLayout(this);
    auto* form = new QFormLayout;
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    // --- model catalog (map role) ---
    m_modelCombo = new QComboBox(this);
    form->addRow(tr("Model"), m_modelCombo);

    m_customModelRow = new QWidget(this);
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    customLayout->setContentsMargins(0, 0, 0, 0);
    m_customModelPath = new QLineEdit(m_customModelRow);
    auto* browseModelBtn = new QPushButton(tr("Browse…"), m_customModelRow);
    customLayout->addWidget(m_customModelPath, 1);
    customLayout->addWidget(browseModelBtn);
    m_customModelRow->setVisible(false);
    form->addRow(tr("Custom model"), m_customModelRow);

    // --- sky masking source (none / native skyseg / cached test masks) ---
    m_skySourceCombo = new QComboBox(this);
    m_skySourceCombo->addItem(tr("None"), QStringLiteral("none"));
    m_skySourceCombo->addItem(tr("Native skyseg (--mask_sky)"),
                              QStringLiteral("native"));
    m_skySourceCombo->addItem(tr("Cached test masks"),
                              QStringLiteral("cached"));
    m_skysegCombo = new QComboBox(this);
    m_skysegCombo->setEnabled(false);
    form->addRow(tr("Sky masking"), m_skySourceCombo);
    m_skysegRow = new QWidget(this);
    auto* skysegLayout = new QHBoxLayout(m_skysegRow);
    skysegLayout->setContentsMargins(0, 0, 0, 0);
    skysegLayout->addWidget(new QLabel(tr("Skyseg model:"), m_skysegRow), 0);
    skysegLayout->addWidget(m_skysegCombo, 1);
    form->addRow(QString(), m_skysegRow);

    // --- test data row (cached download + auto-fill, lingbot_map_data) ---
    m_testSceneCombo = new QComboBox(this);
    connect(m_testSceneCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { applyTestSceneSelection(); });
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Download (cached) the selected official LingBot-Map demo "
               "sequence and auto-fill the image folder"));
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::requestTestData);
    connect(m_skySourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { refreshSkySourceOptions(); });
    auto* testRow = new QHBoxLayout;
    testRow->addWidget(m_testSceneCombo, 1);
    testRow->addWidget(m_testDataBtn);
    form->addRow(tr("Test data:"), testRow);

    // Cached sky-mask directory (auto-filled by the test-data flow; a
    // custom directory works too — masks must be <frame-stem>.png with
    // 255 = keep at the processed resolution).
    m_skyMaskRow = new QWidget(this);
    auto* maskRowLayout = new QHBoxLayout(m_skyMaskRow);
    maskRowLayout->setContentsMargins(0, 0, 0, 0);
    m_skyMaskDir = new QLineEdit(m_skyMaskRow);
    maskRowLayout->addWidget(m_skyMaskDir, 1);
    form->addRow(QString(), m_skyMaskRow);
    m_skyMaskRow->setVisible(false);

    // --- input ---
    auto* folderRow = new QWidget(this);
    auto* folderLayout = new QHBoxLayout(folderRow);
    folderLayout->setContentsMargins(0, 0, 0, 0);
    m_imageFolder = new QLineEdit(folderRow);
    auto* browseFolderBtn = new QPushButton(tr("Browse…"), folderRow);
    folderLayout->addWidget(m_imageFolder, 1);
    folderLayout->addWidget(browseFolderBtn);
    form->addRow(tr("Image folder"), folderRow);

    m_maxFrames = new QSpinBox(this);
    m_maxFrames->setRange(0, 1000000);
    m_maxFrames->setValue(0);
    m_maxFrames->setSpecialValueText(tr("all"));
    form->addRow(tr("Max frames"), m_maxFrames);

    m_imageSize = new QSpinBox(this);
    m_imageSize->setRange(224, 1022);
    m_imageSize->setValue(518);
    m_imageSize->setSingleStep(14);
    form->addRow(tr("Processing width"), m_imageSize);

    // --- backend ---
    m_deviceCombo = new QComboBox(this);
    m_deviceCombo->addItem(tr("Auto"), QStringLiteral("auto"));
    m_deviceCombo->addItem(QStringLiteral("CPU"), QStringLiteral("cpu"));
    m_deviceCombo->addItem(QStringLiteral("GPU"), QStringLiteral("gpu"));
    form->addRow(tr("Device"), m_deviceCombo);

    m_threads = new QSpinBox(this);
    m_threads->setRange(0, 256);
    m_threads->setValue(0);
    m_threads->setSpecialValueText(tr("default"));
    form->addRow(tr("CPU threads"), m_threads);

    m_confThreshold = new QDoubleSpinBox(this);
    m_confThreshold->setRange(0.0, 100.0);
    m_confThreshold->setSingleStep(0.1);
    m_confThreshold->setValue(1.5);
    form->addRow(tr("Confidence threshold"), m_confThreshold);

    m_addDbCheck = new QCheckBox(tr("Add reconstruction to DB tree"), this);
    m_addDbCheck->setChecked(true);
    form->addRow(QString(), m_addDbCheck);

    rootLayout->addLayout(form);

    // --- run / cancel ---
    auto* buttons = new QHBoxLayout;
    m_runButton = new QPushButton(tr("Run"), this);
    m_runButton->setDefault(true);
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setEnabled(false);
    m_downloadButton = new QPushButton(tr("Download model"), this);
    buttons->addWidget(m_runButton);
    buttons->addWidget(m_cancelButton);
    buttons->addWidget(m_downloadButton);
    buttons->addStretch(1);
    rootLayout->addLayout(buttons);

    // --- progress + log (shared AICore plugin layout) ---
    m_downloadLabel = new QLabel(this);
    m_downloadLabel->setVisible(false);
    m_progress = new QProgressBar(this);
    m_progress->setVisible(false);
    rootLayout->addWidget(m_downloadLabel);
    rootLayout->addWidget(m_progress);
    m_log = new QTextEdit(this);
    m_log->setReadOnly(true);
    rootLayout->addWidget(m_log, 1);

    connect(m_runButton, &QPushButton::clicked, this, &LingbotMapDialog::onRun);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &LingbotMapDialog::onCancel);
    connect(m_downloadButton, &QPushButton::clicked, this, [this]() {
        startDownload(m_modelCombo->currentData().toString());
    });
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LingbotMapDialog::onModelComboChanged);
    connect(browseModelBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseCustomModel);
    connect(browseFolderBtn, &QPushButton::clicked, this,
            &LingbotMapDialog::onBrowseImageFolder);

    // Test-data repository wiring (shared download/extract pipeline).
    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& text) {
                Q_UNUSED(text);
                m_progress->setVisible(true);
                m_progress->setValue(percent);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            &LingbotMapDialog::appendLog);
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            &LingbotMapDialog::onTestDataDownloadFinished);
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            &LingbotMapDialog::onTestDataExtractionFinished);

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
            &LingbotMapDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                m_downloadInProgress = false;
                m_progress->setVisible(false);
                m_downloadLabel->setVisible(false);
                if (!ok) {
                    appendLog(tr("[LingbotMap] Download failed: %1").arg(path));
                    return;
                }
                appendLog(tr("[LingbotMap] Model downloaded: %1").arg(path));
                if (m_pendingActionAfterDownload == PendingAction::Run) {
                    m_pendingActionAfterDownload = PendingAction::None;
                    onRun();
                }
            });

    populateCatalogs();
    populateTestDataScenes();
    loadSettings();
}

LingbotMapDialog::~LingbotMapDialog() {
    if (m_downloader && m_downloadInProgress) m_downloader->cancel();
    saveSettings();
}

void LingbotMapDialog::populateCatalogs() {
    const QString previous = m_modelCombo->currentData().toString();
    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    const QVector<LingbotCatalogEntry> maps = catalogByRole("map");
    const int defaultIndex = aicore_lingbot_model_default_index();
    int currentDefault = 0;
    for (int i = 0; i < maps.size(); ++i) {
        const LingbotCatalogEntry& e = maps[i];
        m_modelCombo->addItem(tr("%1 (%2, %3 MB)")
                                      .arg(e.displayName, e.quantNote)
                                      .arg(e.sizeBytes / (1024 * 1024)),
                              e.filename);
        if (aicore_lingbot_model_by_filename(e.filename.toUtf8().constData()) ==
            aicore_lingbot_model_at(defaultIndex)) {
            currentDefault = i;
        }
    }
    m_modelCombo->setCurrentIndex(currentDefault);
    if (!previous.isEmpty()) {
        const int idx = m_modelCombo->findData(previous);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    m_modelCombo->blockSignals(false);

    m_skysegCombo->blockSignals(true);
    m_skysegCombo->clear();
    const QVector<LingbotCatalogEntry> skysegs = catalogByRole("skyseg");
    const int skysegDefault = aicore_lingbot_skyseg_model_default_index();
    int skysegCurrent = 0;
    for (int i = 0; i < skysegs.size(); ++i) {
        const LingbotCatalogEntry& e = skysegs[i];
        m_skysegCombo->addItem(tr("%1 (%2 MB)")
                                       .arg(e.displayName)
                                       .arg(e.sizeBytes / (1024 * 1024)),
                               e.filename);
        if (aicore_lingbot_model_by_filename(e.filename.toUtf8().constData()) ==
            aicore_lingbot_model_at(skysegDefault)) {
            skysegCurrent = i;
        }
    }
    m_skysegCombo->setCurrentIndex(skysegCurrent);
    m_skysegCombo->blockSignals(false);
}

void LingbotMapDialog::populateSkysegCombo(const QString& filename) {
    const int idx = m_skysegCombo->findData(filename);
    if (idx >= 0) m_skysegCombo->setCurrentIndex(idx);
}

void LingbotMapDialog::appendLog(const QString& msg) {
    m_log->append(QStringLiteral("[%1] %2").arg(
            QDateTime::currentDateTime().toString(QStringLiteral("HH:mm:ss")),
            msg));
}

void LingbotMapDialog::onModelComboChanged() {
    const QString filename = m_modelCombo->currentData().toString();
    const bool isCustom =
            filename.isEmpty() ||
            (filename.endsWith(QStringLiteral(".gguf")) &&
             !findEntryByFilename(catalogByRole("map"), filename, nullptr));
    m_customModelRow->setVisible(isCustom);
}

void LingbotMapDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir = ecvPS::browseDir(
            settings, QStringLiteral("qLingbotMap"),
            QStringLiteral("lastModelDir"), lingbotModelCacheDir());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select LingBot-Map GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    m_customModelPath->setText(path);
    m_customModelRow->setVisible(true);
    m_modelCombo->setCurrentIndex(-1);
    m_modelCombo->addItem(QFileInfo(path).fileName(), path);
    m_modelCombo->setCurrentIndex(m_modelCombo->count() - 1);
}

void LingbotMapDialog::onBrowseImageFolder() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qLingbotMap"),
                             QStringLiteral("lastImageDir"), QDir::homePath());
    const QString path = QFileDialog::getExistingDirectory(
            this, tr("Select ordered image sequence folder"), lastDir);
    if (path.isEmpty()) return;
    m_imageFolder->setText(path);
}

QString LingbotMapDialog::resolveModelPath() const {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = lingbotModelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

QString LingbotMapDialog::resolveSkysegPath() const {
    const QString filename = m_skysegCombo->currentData().toString();
    if (filename.isEmpty()) return QString();
    if (QFileInfo::exists(filename)) return filename;
    const QString dir = lingbotModelCacheDir();
    if (dir.isEmpty()) return QString();
    return dir + QDir::separator() + filename;
}

bool LingbotMapDialog::ensureModelAvailable(PendingAction action) {
    const QString filename = m_modelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[LingbotMap] Select a model first."));
        return false;
    }
    if (!QFileInfo::exists(resolveModelPath())) {
        LingbotCatalogEntry entry;
        if (!findEntryByFilename(catalogByRole("map"), filename, &entry)) {
            appendLog(
                    tr("[LingbotMap] Model file not found: %1").arg(filename));
            return false;
        }
        m_pendingActionAfterDownload = action;
        appendLog(tr("[LingbotMap] Model missing — downloading %1; the stream "
                     "starts automatically when ready.")
                          .arg(filename));
        startDownload(filename);
        return false;
    }
    return true;
}

void LingbotMapDialog::startDownload(const QString& filename) {
    if (m_downloadInProgress) {
        appendLog(tr("[LingbotMap] A download is already running."));
        return;
    }
    LingbotCatalogEntry entry;
    if (!findEntryByFilename(catalogByRole("map"), filename, &entry) &&
        !findEntryByFilename(catalogByRole("skyseg"), filename, &entry)) {
        appendLog(tr("[LingbotMap] Unknown model: %1").arg(filename));
        return;
    }
    QDir().mkpath(lingbotModelCacheDir());
    const QString dest =
            lingbotModelCacheDir() + QDir::separator() + entry.filename;
    if (QFile::exists(dest)) {
        appendLog(tr("[LingbotMap] Model already present: %1").arg(dest));
        return;
    }
    appendLog(tr("[LingbotMap] Downloading %1 (%2)…")
                      .arg(entry.filename, entry.downloadUrl));
    m_downloadInProgress = true;
    m_downloadLabel->setVisible(true);
    ecvModelDownloader::Request req;
    req.url = entry.downloadUrl;
    req.destPath = dest;
    req.minBytes = 1024 * 1024;  // LingBot-Map GGUFs are hundreds of MB
    // Content identity from the pinned digest registry — streamed SHA-256
    // check at ingestion (truncation and corruption both caught).
    req.contentAnchor = {QCryptographicHash::Sha256,
                         ecvAssetIntegrity::PinnedDigest(entry.filename)};
    m_downloader->download(req);
}

LingbotMapWorker::Settings LingbotMapDialog::collectSettings() const {
    LingbotMapWorker::Settings s;
    s.modelPath = resolveModelPath();
    s.imageFolder = m_imageFolder->text().trimmed();
    s.maxFrames = m_maxFrames->value();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.image_size = m_imageSize->value();
    s.confThreshold = static_cast<float>(m_confThreshold->value());
    const QString skySource = m_skySourceCombo->currentData().toString();
    if (skySource == QStringLiteral("native")) {
        s.skySource = LingbotMapWorker::Settings::SkySource::Native;
        s.skysegModelPath = resolveSkysegPath();
    } else if (skySource == QStringLiteral("cached")) {
        s.skySource = LingbotMapWorker::Settings::SkySource::CachedMasks;
        s.skyMaskDir = m_skyMaskDir->text().trimmed();
    } else {
        s.skySource = LingbotMapWorker::Settings::SkySource::None;
    }
    s.addResultToDb = m_addDbCheck->isChecked();
    return s;
}

void LingbotMapDialog::onRun() {
    if (!ensureModelAvailable(PendingAction::Run)) return;
    const LingbotMapWorker::Settings s = collectSettings();
    if (s.imageFolder.isEmpty() || !QFileInfo::exists(s.imageFolder)) {
        QMessageBox::warning(this, tr("LingBot-Map"),
                             tr("Select an existing image folder first."));
        return;
    }
    if (s.skySource == LingbotMapWorker::Settings::SkySource::Native &&
        !QFileInfo::exists(s.skysegModelPath)) {
        // Auto-download the selected skyseg model when missing.
        const QString skysegName = m_skysegCombo->currentData().toString();
        if (!skysegName.isEmpty() &&
            findEntryByFilename(catalogByRole("skyseg"), skysegName, nullptr)) {
            appendLog(tr("[LingbotMap] Sky model missing — downloading %1…")
                              .arg(skysegName));
            startDownload(skysegName);
            return;
        }
        appendLog(
                tr("[LingbotMap] Sky model unavailable; continuing without "
                   "sky masking."));
        m_skySourceCombo->setCurrentIndex(0);
    }
    if (s.skySource == LingbotMapWorker::Settings::SkySource::CachedMasks &&
        s.skyMaskDir.isEmpty()) {
        appendLog(
                tr("[LingbotMap] Cached mask folder is empty — continuing "
                   "without sky masking."));
        m_skySourceCombo->setCurrentIndex(0);
    }
    saveSettings();
    emit runRequested(s);
}

void LingbotMapDialog::onCancel() { emit cancelRequested(); }

void LingbotMapDialog::loadSettings() {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qLingbotMap"));
    const QString modelFilename =
            settings.value(QStringLiteral("modelFilename")).toString();
    m_modelExplicit =
            settings.value(QStringLiteral("modelFilenameExplicit"), false)
                    .toBool();
    if (m_modelExplicit) {
        const int idx = m_modelCombo->findData(modelFilename);
        if (idx >= 0) m_modelCombo->setCurrentIndex(idx);
    }
    const QString device =
            settings.value(QStringLiteral("device"), QStringLiteral("auto"))
                    .toString();
    const int idx = m_deviceCombo->findData(device);
    if (idx >= 0) m_deviceCombo->setCurrentIndex(idx);
    m_threads->setValue(settings.value(QStringLiteral("threads"), 0).toInt());
    m_imageFolder->setText(
            settings.value(QStringLiteral("imageFolder")).toString());
    m_maxFrames->setValue(
            settings.value(QStringLiteral("maxFrames"), 0).toInt());
    m_imageSize->setValue(
            settings.value(QStringLiteral("imageSize"), 518).toInt());
    m_confThreshold->setValue(
            settings.value(QStringLiteral("confThreshold"), 1.5).toDouble());
    const QString skySource =
            settings.value(QStringLiteral("skySource"), QStringLiteral("none"))
                    .toString();
    const int skyIdx = m_skySourceCombo->findData(skySource);
    if (skyIdx >= 0) m_skySourceCombo->setCurrentIndex(skyIdx);
    populateSkysegCombo(
            settings.value(QStringLiteral("skysegModel")).toString());
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    settings.endGroup();
}

void LingbotMapDialog::saveSettings() const {
    QSettings settings;
    settings.beginGroup(QStringLiteral("qLingbotMap"));
    settings.setValue(QStringLiteral("modelFilename"),
                      m_modelCombo->currentData().toString());
    settings.setValue(QStringLiteral("modelFilenameExplicit"), m_modelExplicit);
    settings.setValue(QStringLiteral("device"),
                      m_deviceCombo->currentData().toString());
    settings.setValue(QStringLiteral("threads"), m_threads->value());
    settings.setValue(QStringLiteral("imageFolder"), m_imageFolder->text());
    settings.setValue(QStringLiteral("maxFrames"), m_maxFrames->value());
    settings.setValue(QStringLiteral("imageSize"), m_imageSize->value());
    settings.setValue(QStringLiteral("confThreshold"),
                      m_confThreshold->value());
    settings.setValue(QStringLiteral("skySource"),
                      m_skySourceCombo->currentData().toString());
    settings.setValue(QStringLiteral("skysegModel"),
                      m_skysegCombo->currentData().toString());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.endGroup();
}

void LingbotMapDialog::setTaskRunning(bool running) {
    m_runButton->setEnabled(!running);
    m_cancelButton->setEnabled(running);
    if (!running) {
        m_progress->setVisible(false);
        m_downloadLabel->setVisible(false);
    }
}

void LingbotMapDialog::setProgress(int percent) {
    if (percent < 0) {
        m_progress->setVisible(false);
        return;
    }
    m_progress->setVisible(true);
    m_progress->setRange(0, 100);
    m_progress->setValue(percent);
}

// ---------------------------------------------------------------------------
// Test data (shared ecvTestDataRepository, lingbot_map_data release)
// ---------------------------------------------------------------------------

namespace {

bool lingbotDatasetFromName(const QString& name,
                            ecvTestDataRepository::Dataset* out) {
    if (name == QStringLiteral("LingbotMapCourthouse")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapCourthouse;
    } else if (name == QStringLiteral("LingbotMapLoop")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapLoop;
    } else if (name == QStringLiteral("LingbotMapOxford")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapOxford;
    } else if (name == QStringLiteral("LingbotMapUniversity")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapUniversity;
    } else if (name == QStringLiteral("LingbotMapOxfordSkyMasks")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks;
    } else if (name == QStringLiteral("LingbotMapUniversitySkyMasks")) {
        *out = ecvTestDataRepository::Dataset::LingbotMapUniversitySkyMasks;
    } else {
        return false;
    }
    return true;
}

// Returns the directory that directly holds the sequence frames below an
// extracted bundle root (the archives carry a top-level scene directory).
QString lingbotFrameDirOf(const QString& extractRoot) {
    QDirIterator it(
            extractRoot,
            QStringList{QStringLiteral("*.png"), QStringLiteral("*.jpg")},
            QDir::Files, QDirIterator::Subdirectories);
    if (it.hasNext()) {
        return QFileInfo(it.next()).absolutePath();
    }
    return extractRoot;
}

// Sky-mask dataset paired with a scene (none for courthouse/loop).
bool lingbotMaskDatasetFor(ecvTestDataRepository::Dataset scene,
                           ecvTestDataRepository::Dataset* maskOut) {
    if (scene == ecvTestDataRepository::Dataset::LingbotMapOxford) {
        *maskOut = ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks;
        return true;
    }
    if (scene == ecvTestDataRepository::Dataset::LingbotMapUniversity) {
        *maskOut = ecvTestDataRepository::Dataset::LingbotMapUniversitySkyMasks;
        return true;
    }
    return false;
}

}  // namespace

void LingbotMapDialog::populateTestDataScenes() {
    QComboBox* combo = m_testSceneCombo;
    combo->blockSignals(true);
    combo->clear();
    combo->addItem(tr("Courthouse (outdoor)"),
                   QStringLiteral("LingbotMapCourthouse"));
    combo->addItem(tr("Loop (outdoor loop closure)"),
                   QStringLiteral("LingbotMapLoop"));
    combo->addItem(tr("Oxford Spires (outdoor)"),
                   QStringLiteral("LingbotMapOxford"));
    combo->addItem(tr("University (outdoor)"),
                   QStringLiteral("LingbotMapUniversity"));
    combo->blockSignals(false);
    applyTestSceneSelection();
}

void LingbotMapDialog::applyTestSceneSelection() {
    ecvTestDataRepository::Dataset scene;
    if (!lingbotDatasetFromName(m_testSceneCombo->currentData().toString(),
                                &scene)) {
        return;
    }
    // Auto-fill the image folder when the scene is already extracted.
    const QString extract = ecvTestDataRepository::extractPath(scene);
    if (!ecvTestDataRepository::getLingbotMapImages(extract).isEmpty()) {
        m_imageFolder->setText(lingbotFrameDirOf(extract));
    }
    refreshSkySourceOptions();
}

void LingbotMapDialog::refreshSkySourceOptions() {
    ecvTestDataRepository::Dataset scene;
    if (!lingbotDatasetFromName(m_testSceneCombo->currentData().toString(),
                                &scene)) {
        return;
    }
    ecvTestDataRepository::Dataset maskKind;
    const bool hasMasks = lingbotMaskDatasetFor(scene, &maskKind);
    const int cachedIdx = m_skySourceCombo->findData(QStringLiteral("cached"));
    if (cachedIdx >= 0) {
        // Cached masks exist only for the scenes that ship them, and only
        // once the mask bundle is downloaded/extracted.
        const bool enabled =
                hasMasks &&
                ecvTestDataRepository::instance().isDatasetAvailable(maskKind);
        auto* itemModel =
                qobject_cast<QStandardItemModel*>(m_skySourceCombo->model());
        if (itemModel && itemModel->item(cachedIdx)) {
            itemModel->item(cachedIdx)->setEnabled(enabled);
        }
    }
    const bool native = m_skySourceCombo->currentData().toString() ==
                        QStringLiteral("native");
    const bool cached = m_skySourceCombo->currentData().toString() ==
                        QStringLiteral("cached");
    m_skysegRow->setVisible(native);
    m_skyMaskRow->setVisible(cached);
}

void LingbotMapDialog::requestTestData() {
    if (m_testDataInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    ecvTestDataRepository::Dataset scene;
    if (!lingbotDatasetFromName(m_testSceneCombo->currentData().toString(),
                                &scene)) {
        return;
    }
    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        return;
    }
    m_testDataInProgress = true;
    m_testDataBtn->setEnabled(false);
    m_runButton->setEnabled(false);
    m_progress->setVisible(true);
    m_progress->setValue(0);
    m_downloadLabel->setVisible(true);

    if (!repo.isDatasetAvailable(scene)) {
        appendLog(tr("[Test data] Downloading %1…")
                          .arg(ecvTestDataRepository::getDatasetInfo(scene)
                                       .displayName));
        repo.startDownload(scene);
        return;
    }
    if (!ecvTestDataRepository::getLingbotMapImages(
                 ecvTestDataRepository::extractPath(scene))
                 .isEmpty()) {
        // Already extracted — apply the selection directly.
        applyTestSceneSelection();
        // Queue the paired mask bundle when the cached-mask source needs it.
        if (m_skySourceCombo->currentData().toString() ==
            QStringLiteral("cached")) {
            ecvTestDataRepository::Dataset maskKind;
            if (lingbotMaskDatasetFor(scene, &maskKind) &&
                !ecvTestDataRepository::instance().isDatasetAvailable(
                        maskKind)) {
                m_maskDownloadPending = true;
                m_pendingMaskDownload = maskKind;
                repo.startDownload(maskKind);
                return;
            }
        }
        finishTestDataFlow();
        return;
    }
    // Verified zip cached — extract it.
    appendLog(tr("[Test data] Extracting cached archive..."));
    repo.extractDataset(scene);
}

void LingbotMapDialog::finishTestDataFlow() {
    m_testDataInProgress = false;
    m_testDataBtn->setEnabled(true);
    m_runButton->setEnabled(true);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
}

void LingbotMapDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        finishTestDataFlow();
        return;
    }
    appendLog(tr("[Test data] Extracting…"));
    ecvTestDataRepository::instance().extractDataset(kind);
}

void LingbotMapDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataInProgress) return;
    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        finishTestDataFlow();
        return;
    }
    // Mask bundles just complete the cached-mask flow; scene bundles fill
    // the image folder.
    const bool isMask =
            kind == ecvTestDataRepository::Dataset::LingbotMapOxfordSkyMasks ||
            kind == ecvTestDataRepository::Dataset::
                            LingbotMapUniversitySkyMasks;
    if (isMask) {
        appendLog(tr("[Test data] Sky masks ready."));
        m_skyMaskDir->setText(
                lingbotFrameDirOf(ecvTestDataRepository::extractPath(kind)));
        refreshSkySourceOptions();
        finishTestDataFlow();
        return;
    }
    applyTestSceneSelection();
    appendLog(tr("[Test data] Scene ready — image folder auto-filled."));
    // Queue the paired mask bundle when the cached-mask source needs it.
    if (m_skySourceCombo->currentData().toString() ==
        QStringLiteral("cached")) {
        ecvTestDataRepository::Dataset maskKind;
        if (lingbotMaskDatasetFor(kind, &maskKind) &&
            !ecvTestDataRepository::instance().isDatasetAvailable(maskKind)) {
            m_maskDownloadPending = true;
            m_pendingMaskDownload = maskKind;
            ecvTestDataRepository::instance().startDownload(maskKind);
            return;
        }
    }
    finishTestDataFlow();
}
