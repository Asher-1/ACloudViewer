// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDDialog.h"

#include <QCloseEvent>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSettings>
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

constexpr int kDbFullImageRole = Qt::UserRole + 1;

/** Official-demo presets (upstream General-Keypoint-Detection README):
 *  selecting a sample auto-fills the prompt fields with the exact official
 *  demo configuration so one click reproduces the published result. */
struct SamplePreset {
    const char* fileName;
    const char* kpsTexts; /**< keypoint texts (empty = leave) */
};
const SamplePreset kSamplePresets[] = {
        // Text-prompt demo: 5 face keypoints on the whole image.
        {"2007_007524.jpg", "nose, left eye, right eye, left ear, right ear"},
        // 1-shot visual demo: the support image IS 2007_003778.jpg; its
        // three annotated keypoints (left eye / right eye / nose).
        {"2007_003778.jpg", "left eye, right eye, nose"},
};

void applySamplePreset(const QString& fileName, QLineEdit* kpsTexts) {
    for (const auto& preset : kSamplePresets) {
        if (fileName == QLatin1String(preset.fileName)) {
            kpsTexts->setText(QString::fromUtf8(preset.kpsTexts));
            return;
        }
    }
}

}  // namespace

GKDDialog::GKDDialog(QWidget* parent) : QDialog(parent) {
    setupUi();
    loadSettings();
    populateYoloModelCombo();
    populateModelCombo();
}

GKDDialog::~GKDDialog() {
    if (m_downloader && m_downloadInProgress) m_downloader->cancel();
    saveSettings();
}

void GKDDialog::setupUi() {
    setWindowTitle(tr("GKD — General Keypoint Detection (GKDT)"));
    resize(ecvAICoreUi::dpiScaled(560), ecvAICoreUi::dpiScaled(680));

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setSpacing(ecvAICoreUi::vSpacing());

    // ---- model group ----
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
    m_deviceCombo = new QComboBox(modelGroup);
    m_threads = new QSpinBox(modelGroup);
    m_threads->setRange(0, 128);
    m_threads->setValue(0);
    m_threads->setToolTip(tr("0 = backend default"));
    ecvAICoreUi::setCompactSpin(m_threads);
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
    modelGrid->addWidget(
            ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads, modelGroup),
            2, 0, 1, 3);
    rootLayout->addWidget(modelGroup);

    // ---- input group ----
    auto* inputGroup = new QGroupBox(tr("Query image"), this);
    auto* inputGrid = new QGridLayout(inputGroup);
    ecvAICoreUi::setupFormGrid(inputGrid);
    m_imagePath = new QLineEdit(inputGroup);
    connect(m_imagePath, &QLineEdit::textChanged, this,
            &GKDDialog::updateImagePreview);
    auto* browseImageBtn =
            ecvAICoreUi::makeBrowseBtn(tr("Browse…"), inputGroup);
    connect(browseImageBtn, &QPushButton::clicked, this,
            &GKDDialog::onBrowseImage);
    inputGrid->addWidget(
            ecvAICoreUi::makeLabel(tr("Image (or db://name):"), 110), 0, 0);
    inputGrid->addWidget(m_imagePath, 0, 1);
    inputGrid->addWidget(browseImageBtn, 0, 2);
    m_previewLabel = new ecvClickableImageLabel(inputGroup);
    m_previewLabel->setFixedSize(ecvAICoreUi::previewSize(),
                                 ecvAICoreUi::previewSize());
    inputGrid->addWidget(m_previewLabel, 1, 1, Qt::AlignHCenter);

    // DB image picker (same collapsible section as the sibling dialogs).
    m_dbImageList = new QListWidget(inputGroup);
    m_dbImageList->setViewMode(QListWidget::IconMode);
    m_dbImageList->setIconSize(QSize(72, 72));
    m_dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
    m_dbImageList->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &GKDDialog::onDbListActivated);
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &GKDDialog::onDbListActivated);
    auto* dbRow = new QHBoxLayout;
    auto* dbRefreshBtn = new QPushButton(tr("Refresh DB images"), inputGroup);
    connect(dbRefreshBtn, &QPushButton::clicked, this,
            &GKDDialog::refreshDbImagesRequested);
    dbRow->addWidget(m_dbImageList, 1);
    dbRow->addWidget(dbRefreshBtn);
    inputGrid->addWidget(ecvAICoreUi::makeLabel(tr("DB images:"), 110), 2, 0,
                         Qt::AlignTop);
    inputGrid->addLayout(dbRow, 2, 1, 1, 2);

    // Sample-data row: cached download + auto-fill of the official GKDT
    // demo images (shared ecvTestDataRepository), plus custom browse.
    m_testDataCombo = new QComboBox(inputGroup);
    m_testDataCombo->setMinimumWidth(ecvAICoreUi::dpiScaled(150));
    m_testDataCombo->setToolTip(
            tr("Pick which official GKDT demo image to load (cached "
               "download)"));
    connect(m_testDataCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { loadSelectedTestData(); });
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(inputGroup);
    m_testDataBtn->setToolTip(
            tr("Download (cached) and load the selected official demo image, "
               "with its prompts auto-filled"));
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &GKDDialog::requestTestData);
    auto* sampleRow = new QHBoxLayout;
    sampleRow->addWidget(m_testDataCombo, 1);
    sampleRow->addWidget(m_testDataBtn);
    inputGrid->addWidget(ecvAICoreUi::makeLabel(tr("Sample data:"), 110), 3, 0);
    inputGrid->addLayout(sampleRow, 3, 1, 1, 2);
    rootLayout->addWidget(inputGroup);

    // ---- prompts group ----
    auto* promptGroup = new QGroupBox(tr("Prompts"), this);
    auto* promptGrid = new QGridLayout(promptGroup);
    ecvAICoreUi::setupFormGrid(promptGrid);
    m_kpsTexts = new QLineEdit(promptGroup);
    m_kpsTexts->setPlaceholderText(
            tr("nose, left eye, right eye, left ear, right ear"));
    connect(m_kpsTexts, &QLineEdit::textChanged, this,
            [this](const QString&) { saveSettings(); });
    promptGrid->addWidget(ecvAICoreUi::makeLabel(tr("Keypoint texts:"), 110), 0,
                          0);
    promptGrid->addWidget(m_kpsTexts, 0, 1, 1, 2);

    m_supportImagePath = new QLineEdit(promptGroup);
    auto* browseSupportBtn =
            ecvAICoreUi::makeBrowseBtn(tr("Browse…"), promptGroup);
    connect(browseSupportBtn, &QPushButton::clicked, this,
            &GKDDialog::onBrowseSupportImage);
    promptGrid->addWidget(ecvAICoreUi::makeLabel(tr("Support image:"), 110), 1,
                          0);
    promptGrid->addWidget(m_supportImagePath, 1, 1);
    promptGrid->addWidget(browseSupportBtn, 1, 2);

    m_supportKps = new QLineEdit(promptGroup);
    m_supportKps->setPlaceholderText(
            tr("x1,y1 x2,y2 … (support-image pixels)"));
    promptGrid->addWidget(ecvAICoreUi::makeLabel(tr("Support kps:"), 110), 2,
                          0);
    promptGrid->addWidget(m_supportKps, 2, 1, 1, 2);

    m_roi = new QLineEdit(promptGroup);
    m_roi->setPlaceholderText(tr("x1 y1 x2 y2 (empty = whole image)"));
    promptGrid->addWidget(ecvAICoreUi::makeLabel(tr("ROI bbox:"), 110), 3, 0);
    promptGrid->addWidget(m_roi, 3, 1, 1, 2);

    m_minScore = new QDoubleSpinBox(promptGroup);
    m_minScore->setRange(0.0, 1.0);
    m_minScore->setSingleStep(0.05);
    m_minScore->setValue(0.30);
    ecvAICoreUi::setCompactDoubleSpin(m_minScore);
    promptGrid->addWidget(ecvAICoreUi::makeLabel(tr("Min score:"), 110), 4, 0);
    promptGrid->addWidget(m_minScore, 4, 1);
    rootLayout->addWidget(promptGroup);

    // ---- multi-object group ----
    auto* multiGroup =
            new QGroupBox(tr("Multi-object (YOLO-World boxes)"), this);
    auto* multiGrid = new QGridLayout(multiGroup);
    ecvAICoreUi::setupFormGrid(multiGrid);
    m_multiObjectCheck = new QCheckBox(
            tr("Detect object boxes with YOLO-World first, then GKD per box"),
            multiGroup);
    multiGrid->addWidget(m_multiObjectCheck, 0, 0, 1, 3);
    m_objectClasses = new QLineEdit(multiGroup);
    m_objectClasses->setPlaceholderText(tr("cat, dog, human"));
    m_yoloModelCombo = new QComboBox(multiGroup);
    m_yoloConf = new QDoubleSpinBox(multiGroup);
    m_yoloConf->setRange(0.05, 0.95);
    m_yoloConf->setSingleStep(0.05);
    m_yoloConf->setValue(0.25);
    ecvAICoreUi::setCompactDoubleSpin(m_yoloConf);
    multiGrid->addWidget(ecvAICoreUi::makeLabel(tr("Classes:"), 110), 1, 0);
    multiGrid->addWidget(m_objectClasses, 1, 1, 1, 2);
    multiGrid->addWidget(ecvAICoreUi::makeLabel(tr("Detector:"), 110), 2, 0);
    multiGrid->addWidget(m_yoloModelCombo, 2, 1);
    multiGrid->addWidget(ecvAICoreUi::makeLabel(tr("Conf:"), 110), 2, 2);
    multiGrid->addWidget(m_yoloConf, 2, 2, Qt::AlignLeft);
    rootLayout->addWidget(multiGroup);

    // ---- output group ----
    auto* outGroup = new QGroupBox(tr("Output"), this);
    auto* outGrid = new QGridLayout(outGroup);
    ecvAICoreUi::setupFormGrid(outGrid);
    m_addDbCheck = new QCheckBox(tr("Add rendered result to DB"), outGroup);
    m_savePngCheck = new QCheckBox(tr("Save PNG"), outGroup);
    m_savePngDir = new QLineEdit(outGroup);
    auto* browseSaveBtn = ecvAICoreUi::makeBrowseBtn(tr("Browse…"), outGroup);
    connect(browseSaveBtn, &QPushButton::clicked, this,
            &GKDDialog::onBrowseSaveDir);
    outGrid->addWidget(m_addDbCheck, 0, 0, 1, 3);
    outGrid->addWidget(m_savePngCheck, 1, 0);
    outGrid->addWidget(m_savePngDir, 1, 1);
    outGrid->addWidget(browseSaveBtn, 1, 2);
    rootLayout->addWidget(outGroup);

    // ---- actions + progress + log ----
    m_runBtn = new QPushButton(tr("Run"), this);
    m_runBtn->setDefault(true);
    m_cancelBtn = new QPushButton(tr("Cancel"), this);
    m_cancelBtn->setEnabled(false);
    connect(m_runBtn, &QPushButton::clicked, this, &GKDDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &GKDDialog::onCancel);
    rootLayout->addLayout(ecvAICoreUi::makeActionRow(m_runBtn, m_cancelBtn));

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

    // Sample-data repository (shared cache/download/extract state machine).
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
    m_kpsTexts->setText(
            settings.value(QStringLiteral("kpsTexts"),
                           QStringLiteral("nose, left eye, right eye"))
                    .toString());
    m_supportImagePath->setText(
            settings.value(QStringLiteral("supportPath")).toString());
    m_supportKps->setText(
            settings.value(QStringLiteral("supportKps")).toString());
    m_roi->setText(settings.value(QStringLiteral("roi")).toString());
    m_minScore->setValue(
            settings.value(QStringLiteral("minScore"), 0.30).toDouble());
    m_multiObjectCheck->setChecked(
            settings.value(QStringLiteral("multiObject"), false).toBool());
    m_objectClasses->setText(
            settings.value(QStringLiteral("objectClasses")).toString());
    m_yoloConf->setValue(
            settings.value(QStringLiteral("yoloConf"), 0.25).toDouble());
    const QString yoloModel =
            settings.value(QStringLiteral("yoloModel")).toString();
    if (!yoloModel.isEmpty()) populateYoloModelCombo(yoloModel);
    m_imagePath->setText(
            settings.value(QStringLiteral("imagePath")).toString());
    m_addDbCheck->setChecked(
            settings.value(QStringLiteral("addToDb"), true).toBool());
    m_savePngCheck->setChecked(
            settings.value(QStringLiteral("savePng"), false).toBool());
    m_savePngDir->setText(
            settings.value(QStringLiteral("savePngDir")).toString());
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
    settings.setValue(QStringLiteral("kpsTexts"), m_kpsTexts->text());
    settings.setValue(QStringLiteral("supportPath"),
                      m_supportImagePath->text());
    settings.setValue(QStringLiteral("supportKps"), m_supportKps->text());
    settings.setValue(QStringLiteral("roi"), m_roi->text());
    settings.setValue(QStringLiteral("minScore"), m_minScore->value());
    settings.setValue(QStringLiteral("multiObject"),
                      m_multiObjectCheck->isChecked());
    settings.setValue(QStringLiteral("objectClasses"), m_objectClasses->text());
    settings.setValue(QStringLiteral("yoloConf"), m_yoloConf->value());
    settings.setValue(QStringLiteral("yoloModel"),
                      m_yoloModelCombo->currentData().toString());
    settings.setValue(QStringLiteral("imagePath"), m_imagePath->text());
    settings.setValue(QStringLiteral("addToDb"), m_addDbCheck->isChecked());
    settings.setValue(QStringLiteral("savePng"), m_savePngCheck->isChecked());
    settings.setValue(QStringLiteral("savePngDir"), m_savePngDir->text());
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
    // Multi-object detector: the open-vocabulary YOLO-World family from the
    // existing yolo task catalog (no second model table here).
    QVector<GKDModelEntry> worldModels;
    int worldDefaultIndex = -1;
#ifdef AICore_ENABLED
    const int n = aicore_yolo_model_count(AICORE_YOLO_ROLE_WORLD);
    worldDefaultIndex = aicore_yolo_model_default_index(AICORE_YOLO_ROLE_WORLD);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_WORLD);
        if (!e || !e->filename) continue;
        GKDModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        worldModels.append(entry);
    }
#else
    (void)keepFilename;
#endif
    m_yoloModelCombo->clear();
    for (const GKDModelEntry& e : worldModels) {
        m_yoloModelCombo->addItem(GKDHelpers::modelDisplayLabel(e), e.filename);
    }
    ecvAICoreUi::selectModelRow(m_yoloModelCombo, keepFilename,
                                worldDefaultIndex);
}

bool GKDDialog::selectModelByFilename(const QString& filename) {
    if (filename.isEmpty()) return false;
    const int idx = m_modelCombo->findData(filename);
    if (idx < 0) return false;
    m_modelCombo->setCurrentIndex(idx);
    return true;
}

void GKDDialog::refreshModelList() {
    populateYoloModelCombo(m_yoloModelCombo->currentData().toString());
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

QString GKDDialog::resolveYoloModelPath() const {
    const QString filename = m_yoloModelCombo->currentData().toString();
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
    if (!m_multiObjectCheck->isChecked()) return true;
    const QString filename = m_yoloModelCombo->currentData().toString();
    if (filename.isEmpty()) {
        appendLog(tr("[GKD] Multi-object mode needs a YOLO-World detector."));
        return false;
    }
    if (QFileInfo::exists(resolveYoloModelPath())) return true;
#ifdef AICore_ENABLED
    const aicore_yolo_model_entry* e =
            aicore_yolo_model_by_filename(filename.toUtf8().constData());
    if (e == nullptr) {
        appendLog(tr("[GKD] Detector model not found: %1").arg(filename));
        return false;
    }
    GKDModelEntry entry;
    entry.filename = QString::fromUtf8(e->filename);
    entry.downloadUrl = QString::fromUtf8(e->download_url);
    entry.displayName = QString::fromUtf8(e->display_name);
    entry.quantNote = QString::fromUtf8(e->quant_note);
    m_pendingActionAfterDownload = PendingAction::Run;
    appendLog(tr("[GKD] Detector model missing — downloading %1; the run "
                 "starts automatically when ready.")
                      .arg(filename));
    char* raw = aicore_yolo_model_cache_dir();
    const QString destDir =
            raw ? QString::fromUtf8(raw) : GKDHelpers::modelCacheDir();
    aicore_yolo_free_buffer(raw);
    startDownload(entry, destDir);
    return false;
#else
    appendLog(tr("[GKD] Detector model file not found: %1").arg(filename));
    return false;
#endif
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
    QImage img;
    const QString path = m_imagePath->text().trimmed();
    if (path.startsWith(QStringLiteral("db://"))) {
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
        return;
    }
    m_previewLabel->setPreviewImage(img, ecvAICoreUi::previewSize());
}

void GKDDialog::onRun() {
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
    m_imagePath->setText(QStringLiteral("db://") +
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
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
}

void GKDDialog::setDbImages(const QList<GKDImageEntry>& images) {
    const QString keep = m_imagePath->text();
    m_dbImageList->clear();
    for (const GKDImageEntry& entry : images) {
        auto* item = new QListWidgetItem(
                entry.preview.isNull() ? QIcon()
                                       : QPixmap::fromImage(entry.preview),
                entry.name);
        item->setData(Qt::UserRole, entry.name);
        item->setData(kDbFullImageRole, entry.preview);
        m_dbImageList->addItem(item);
    }
    m_imagePath->setText(keep);
}

void GKDDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    m_imagePath->setText(QStringLiteral("db://") + imageNames.first());
}

GKDWorker::Settings GKDDialog::workerSettings() const {
    GKDWorker::Settings s;
    s.modelPath = resolveModelPath();
    s.inputPath = m_imagePath->text().trimmed();
    s.device = m_deviceCombo->currentData().toString();
    s.threads = m_threads->value();
    s.kpsTexts = GKDHelpers::splitPrompts(m_kpsTexts->text());
    s.supportImagePath = m_supportImagePath->text().trimmed();
    QVector<QPointF> supportKps;
    GKDHelpers::parseCoordinatePairs(m_supportKps->text(), &supportKps);
    s.supportKps = supportKps;
    s.hasBbox = false;
    QVector<QPointF> roi;
    if (GKDHelpers::parseCoordinatePairs(
                m_roi->text().replace(QLatin1Char(','), QLatin1Char(' ')),
                &roi) &&
        roi.size() == 2) {
        s.hasBbox = true;
        s.bbox[0] = static_cast<float>(roi[0].x());
        s.bbox[1] = static_cast<float>(roi[0].y());
        s.bbox[2] = static_cast<float>(roi[1].x());
        s.bbox[3] = static_cast<float>(roi[1].y());
    }
    s.minScore = static_cast<float>(m_minScore->value());
    s.multiObject = m_multiObjectCheck->isChecked();
    s.objectClasses =
            m_objectClasses->text().split(QLatin1Char(','), Qt::SkipEmptyParts);
    for (QString& c : s.objectClasses) c = c.trimmed();
    s.yoloModelPath = resolveYoloModelPath();
    s.yoloConf = static_cast<float>(m_yoloConf->value());
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
    if (m_firstShow) {
        m_firstShow = false;
        populateTestDataCombo();
        emit refreshDbImagesRequested();
    }
}

// ---- Sample data (shared ecvTestDataRepository, GKDT demo dataset) ------

void GKDDialog::populateTestDataCombo() {
    if (!m_testDataCombo) return;
    const QStringList images =
            ecvTestDataRepository::getGeneralKeypointDetectionImages(
                    ecvTestDataRepository::extractPath(
                            ecvTestDataRepository::Dataset::
                                    GeneralKeypointDetection));
    m_testDataCombo->blockSignals(true);
    m_testDataCombo->clear();
    if (images.isEmpty()) {
        m_testDataCombo->addItem(tr("(no test data)"), QString());
    } else {
        for (const QString& path : images) {
            const QString name = QFileInfo(path).fileName();
            m_testDataCombo->addItem(name, name);
        }
        // Default to the official text-prompt demo when present.
        const int demo =
                m_testDataCombo->findData(QStringLiteral("2007_007524.jpg"));
        if (demo >= 0) m_testDataCombo->setCurrentIndex(demo);
    }
    m_testDataCombo->blockSignals(false);
}

bool GKDDialog::loadSelectedTestData() {
    if (!m_testDataCombo) return false;
    const QString fileName = m_testDataCombo->currentData().toString();
    if (fileName.isEmpty()) return false;

    const QString path = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::GeneralKeypointDetection, fileName);
    if (path.isEmpty()) return false;

    QImage img(path);
    if (img.isNull()) {
        appendLog(
                tr("[Test data] Failed to decode sample image: %1").arg(path));
        return true;  // cached file exists but is unusable; don't re-download
    }
    m_imagePath->setText(path);  // auto-fill: updates the preview too
    applySamplePreset(fileName, m_kpsTexts);
    appendLog(tr("[Test data] Loaded official demo image '%1' (prompts "
                 "auto-filled when the demo defines them).")
                      .arg(fileName));
    return true;
}

void GKDDialog::requestTestData() {
    if (m_testDataDownloadInProgress) {
        appendLog(tr("[Test data] Download already in progress."));
        return;
    }
    if (m_taskRunning) {
        appendLog(tr("[Test data] Wait for the current task to finish."));
        return;
    }
    if (loadSelectedTestData()) return;

    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        return;
    }

    const auto kind = ecvTestDataRepository::Dataset::GeneralKeypointDetection;
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    m_testDataDownloadInProgress = true;
    m_runBtn->setEnabled(false);
    m_testDataBtn->setEnabled(false);
    m_progress->setVisible(true);
    m_progress->setValue(0);
    m_downloadLabel->setVisible(true);
    if (ecvAssetIntegrity::isVerified(ecvTestDataRepository::zipPath(kind),
                                      info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        appendLog(tr("[Test data] Extracting cached archive..."));
        repo.extractDataset(kind);
        return;
    }
    appendLog(tr("[Test data] Downloading official GKDT demo data..."));
    repo.startDownload(kind);
}

void GKDDialog::onTestDataDownloadFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::GeneralKeypointDetection) {
        return;
    }
    if (!success) {
        appendLog(tr("[Test data] Download failed."));
        m_testDataDownloadInProgress = false;
        m_runBtn->setEnabled(true);
        m_testDataBtn->setEnabled(true);
        m_progress->setVisible(false);
        m_downloadLabel->setVisible(false);
        return;
    }
    appendLog(tr("[Test data] Extracting..."));
    if (m_progress) m_progress->setValue(0);
    ecvTestDataRepository::instance().extractDataset(kind);
}

void GKDDialog::onTestDataExtractionFinished(
        bool success, ecvTestDataRepository::Dataset kind) {
    if (!m_testDataDownloadInProgress ||
        kind != ecvTestDataRepository::Dataset::GeneralKeypointDetection) {
        return;
    }
    m_testDataDownloadInProgress = false;
    m_runBtn->setEnabled(true);
    m_testDataBtn->setEnabled(true);
    m_progress->setVisible(false);
    m_downloadLabel->setVisible(false);
    if (!success) {
        appendLog(tr("[Test data] Failed to extract zip archive."));
        return;
    }
    populateTestDataCombo();
    if (!loadSelectedTestData()) {
        appendLog(
                tr("[Test data] Select a sample and press the button "
                   "again to load it."));
    }
}
