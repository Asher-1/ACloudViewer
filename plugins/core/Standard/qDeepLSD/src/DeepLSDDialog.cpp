// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DeepLSDDialog.h"

#include <CVLog.h>
#include <cvFileDialog.h>
#include "ecvAICoreUiHelper.h"

#include <QCloseEvent>
#include <QDir>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSettings>
#include <QVBoxLayout>

#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/inference_log.h"
#include "ecvModelDownloader.h"
static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "DeepLSD/";

namespace {

constexpr int kDbFullImageRole = Qt::UserRole + 1;
constexpr const char* kDeepLSDTestImage = "deeplsd_examples.jpg";

bool isSupportedImageFile(const QString& filePath) {
    static const QStringList extensions = {
            QStringLiteral("png"),  QStringLiteral("jpg"),
            QStringLiteral("jpeg"), QStringLiteral("bmp"),
            QStringLiteral("tif"),  QStringLiteral("tiff"),
            QStringLiteral("webp"),
    };
    return extensions.contains(QFileInfo(filePath).suffix(),
                               Qt::CaseInsensitive);
}

bool isValidCachedGguf(const QFileInfo& fi) {
    return ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath());
}

}  // namespace

QVector<DeepLSDBuiltinModel> DeepLSDDialog::builtinModels() {
    const QString base = QString::fromLatin1(kDownloadBase);
    return {
            {tr("DeepLSD Wireframe F16 (recommended)"),
             "deeplsd_wireframe-f16.gguf", base + "deeplsd_wireframe-f16.gguf"},
            {tr("DeepLSD Wireframe Q8_0 (smaller)"),
             "deeplsd_wireframe-q8_0.gguf",
             base + "deeplsd_wireframe-q8_0.gguf"},
            {tr("DeepLSD Wireframe F32"), "deeplsd_wireframe-f32.gguf",
             base + "deeplsd_wireframe-f32.gguf"},
            {tr("DeepLSD MegaDepth F16 (outdoor)"), "deeplsd_md-f16.gguf",
             base + "deeplsd_md-f16.gguf"},
            {tr("DeepLSD MegaDepth Q8_0"), "deeplsd_md-q8_0.gguf",
             base + "deeplsd_md-q8_0.gguf"},
            {tr("DeepLSD MegaDepth F32"), "deeplsd_md-f32.gguf",
             base + "deeplsd_md-f32.gguf"},
    };
}

QString DeepLSDDialog::modelCacheDir() {
    char* dir = aicore_deeplsd_model_cache_dir();
    if (dir) {
        QString result = QString::fromUtf8(dir);
        aicore_deeplsd_free_buffer(dir);
        return result;
    }
    return QDir::homePath() +
           QStringLiteral("/cloudViewer_data/extract/deeplsd_models");
}

DeepLSDDialog::DeepLSDDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("DeepLSD Line Extraction"));
    setMinimumSize(ecvAICoreUi::dpiScaled(720), 0);
    setupUi();
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &DeepLSDDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0 && m_progress) {
                    m_progress->setValue(
                            static_cast<int>(received * 100 / total));
                }
                if (m_downloadLabel) {
                    m_downloadLabel->setText(
                            tr("Downloading %1 — %2")
                                    .arg(m_modelCombo->currentData().toString())
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                const QString finishedFilename = QFileInfo(dest).fileName();
                m_downloadInProgress = false;
                m_downloadLabel->setVisible(false);
                if (ok) {
                    appendLog(tr("[OK] Downloaded model: %1").arg(dest));
                    populateModelCombo(finishedFilename);
                    if (m_autoRunAfterDownload) {
                        m_autoRunAfterDownload = false;
                        onRun();
                    }
                } else {
                    populateModelCombo(finishedFilename);
                    m_autoRunAfterDownload = false;
                }
            });

    auto& testDataRepo = ecvTestDataRepository::instance();
    connect(&testDataRepo, &ecvTestDataRepository::downloadProgress, this,
            [this](int percent, const QString& statusText) {
                if (!m_testDataDownloadInProgress) return;
                m_progress->setRange(0, 100);
                m_progress->setValue(percent);
                m_downloadLabel->setText(statusText);
                m_downloadLabel->setVisible(true);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadLogMessage, this,
            [this](const QString& message) {
                if (m_testDataDownloadInProgress) appendLog(message);
            });
    connect(&testDataRepo, &ecvTestDataRepository::downloadFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                if (!m_testDataDownloadInProgress ||
                    kind != ecvTestDataRepository::Dataset::ObjectsDetection) {
                    return;
                }
                if (!success) {
                    m_testDataDownloadInProgress = false;
                    m_testDataBtn->setEnabled(true);
                    m_downloadLabel->setVisible(false);
                    return;
                }
                m_downloadLabel->setText(tr("Extracting test data..."));
                ecvTestDataRepository::instance().extractDataset(kind);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionProgress, this,
            [this](int current, int total) {
                if (!m_testDataDownloadInProgress || total <= 0) return;
                m_progress->setRange(0, total);
                m_progress->setValue(current);
            });
    connect(&testDataRepo, &ecvTestDataRepository::extractionFinished, this,
            [this](bool success, ecvTestDataRepository::Dataset kind) {
                if (!m_testDataDownloadInProgress ||
                    kind != ecvTestDataRepository::Dataset::ObjectsDetection) {
                    return;
                }
                m_testDataDownloadInProgress = false;
                m_testDataBtn->setEnabled(true);
                m_downloadLabel->setVisible(false);
                if (success) {
                    loadTestImage();
                } else {
                    appendLog(tr("[Test data] Extraction failed."));
                }
            });
    CVLog::Print(QString("[DeepLSD] Model cache: %1").arg(modelCacheDir()));
    aicore_inference_log::log_backend_probe(QStringLiteral("DeepLSD"));
    populateModelCombo();
    restoreSettings();
}

void DeepLSDDialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void DeepLSDDialog::setupUi() {
    auto* main = new QVBoxLayout(this);
    ecvAICoreUi::setupTabLayout(main);

    auto* modelGroup = new QGroupBox(tr("Model"));
    auto* modelLayout = new QGridLayout(modelGroup);
    ecvAICoreUi::setupFormGrid(modelLayout, 92);

    m_modelCombo = new QComboBox;
    modelLayout->addWidget(ecvAICoreUi::makeLabel(tr("GGUF:")), 0, 0);
    modelLayout->addWidget(m_modelCombo, 0, 1, 1, 3);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DeepLSDDialog::onModelComboChanged);

    m_variantHintLabel = new QLabel;
    m_variantHintLabel->setWordWrap(true);
    m_variantHintLabel->setStyleSheet(
            "color: #333; background: #eef4fb; border: 1px solid #b8d4f0; "
            "padding: 6px; border-radius: 4px; font-size: 11px;");
    modelLayout->addWidget(m_variantHintLabel, 1, 0, 1, 4);

    m_customModelRow = new QWidget;
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    customLayout->setContentsMargins(0, 0, 0, 0);
    customLayout->setSpacing(ecvAICoreUi::hSpacing());
    m_customModelPath = new QLineEdit;
    auto* browseModel = ecvAICoreUi::makeBrowseBtn(tr("Browse..."));
    connect(browseModel, &QPushButton::clicked, this,
            &DeepLSDDialog::onBrowseCustomModel);
    customLayout->addWidget(m_customModelPath, 1);
    customLayout->addWidget(browseModel);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 2, 0, 1, 4);

    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        if (const aicore_device_info* d = aicore_device_at(i)) {
            m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
            if (d->is_default) m_deviceCombo->setCurrentIndex(i);
        }
    }

    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText(tr("Auto"));

    auto* runtimeRow = ecvAICoreUi::makeRuntimeRow(m_deviceCombo, m_threads);
    modelLayout->addWidget(runtimeRow, 3, 0, 1, 4);

    m_minSegmentScore = new QDoubleSpinBox;
    m_minSegmentScore->setRange(0.0, 1.0);
    m_minSegmentScore->setSingleStep(0.05);
    m_minSegmentScore->setValue(0.15);
    ecvAICoreUi::setCompactDoubleSpin(m_minSegmentScore);
    m_minSegmentScore->setToolTip(
            tr("Filter by LSD segment quality (-log10 NFA), mapped to 0\u20131. "
               "Higher = more significant line (typical 0.1\u20130.5)."));
    modelLayout->addWidget(ecvAICoreUi::makeLabel(tr("Min segment quality:")), 4, 0);
    modelLayout->addWidget(m_minSegmentScore, 4, 1);

    ecvAICoreUi::tightenGroupBox(modelGroup);
    main->addWidget(modelGroup);

    auto* ioGroup = new QGroupBox(tr("Input"));
    auto* ioLayout = new QVBoxLayout(ioGroup);
    ioLayout->setContentsMargins(6, 4, 6, 4);
    ioLayout->setSpacing(ecvAICoreUi::vSpacing());

    auto* pathRow = new QHBoxLayout;
    pathRow->setSpacing(ecvAICoreUi::hSpacing());
    m_imagePath = new QLineEdit;
    m_imagePath->setPlaceholderText(
            tr("Local image path, or db://EntityName from DB tree"));
    m_imagePath->setToolTip(
            tr("Single-image input. Browse remembers the last folder via "
               "QSettings."));
    connect(m_imagePath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateImagePreview(); });
    auto* browseImg = ecvAICoreUi::makeBrowseBtn(tr("Browse..."));
    browseImg->setToolTip(
            tr("Pick an image file (last folder is remembered)."));
    connect(browseImg, &QPushButton::clicked, this,
            &DeepLSDDialog::onBrowseImage);
    pathRow->addWidget(m_imagePath, 1);
    pathRow->addWidget(browseImg);
    ioLayout->addLayout(pathRow);

    m_previewLabel = new ecvClickableImageLabel;
    const int ps = ecvAICoreUi::previewSize();
    m_previewLabel->setFixedSize(ps, ps);
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabel->setText(tr("Preview"));
    ioLayout->addWidget(
            ecvClickableImageLabel::wrapWithTapToPreviewHint(m_previewLabel));

    auto* dbHeader = new QHBoxLayout;
    dbHeader->setSpacing(ecvAICoreUi::hSpacing());
    m_dbToggleBtn = ecvAICoreUi::makeDbSection(nullptr);
    m_dbToggleBtn->setText(tr("DB Source Images (optional)"));
    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool checked) {
        m_dbToggleBtn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        m_dbContentWidget->setVisible(checked);
    });
    dbHeader->addWidget(m_dbToggleBtn);
    dbHeader->addStretch();
    ioLayout->addLayout(dbHeader);

    m_dbContentWidget = new QWidget;
    auto* dbLayout = new QVBoxLayout(m_dbContentWidget);
    dbLayout->setContentsMargins(0, 0, 0, 0);
    dbLayout->setSpacing(ecvAICoreUi::tightVSpacing());
    m_dbImageList = new QListWidget;
    m_dbImageList->setMaximumHeight(ecvAICoreUi::dbListMaxHeight());
    m_dbImageList->setToolTip(
            tr("Double-click a ccImage from the DB tree to use as input."));
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &DeepLSDDialog::onDbListActivated);
    // Single-click also assigns and refreshes the preview thumbnail, so the
    // shown image always follows the highlighted browser entry (same
    // behaviour as qRFDetr/qRMBG/qYOLO).
    connect(m_dbImageList, &QListWidget::itemClicked, this,
            &DeepLSDDialog::onDbListActivated);
    dbLayout->addWidget(m_dbImageList);
    auto* refreshBtn = new QPushButton(tr("Refresh DB Images"));
    refreshBtn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    connect(refreshBtn, &QPushButton::clicked, this,
            &DeepLSDDialog::refreshDbImagesRequested);
    dbLayout->addWidget(refreshBtn);
    m_dbContentWidget->setVisible(false);
    ioLayout->addWidget(m_dbContentWidget);

    m_addLineVizCheck =
            new QCheckBox(tr("Add line overlay ccImage to DB tree after run"));
    m_addLineVizCheck->setToolTip(
            tr("Raster overlay: detected segments drawn on the source image "
               "(ccImage, 2D). Good for quick visual QA."));
    m_addLineVizCheck->setChecked(true);
    ioLayout->addWidget(m_addLineVizCheck);

    m_addDistanceOverlayCheck = new QCheckBox(
            tr("Add distance-field heatmap ccImage to DB tree after run"));
    m_addDistanceOverlayCheck->setToolTip(
            tr("False-color heatmap of the DeepLSD distance field (ccImage, "
               "2D). Useful for threshold tuning and debugging."));
    m_addDistanceOverlayCheck->setChecked(false);
    ioLayout->addWidget(m_addDistanceOverlayCheck);

    m_exportPolylinesCheck =
            new QCheckBox(tr("Export detected segments as LineSet in DB tree"));
    m_exportPolylinesCheck->setToolTip(
            tr("Vector export: one 2D LineSet entity with segment endpoints "
               "(editable wireframe geometry, not ccPolyline)."));
    m_exportPolylinesCheck->setChecked(false);
    ioLayout->addWidget(m_exportPolylinesCheck);

    ecvAICoreUi::tightenGroupBox(ioGroup);
    main->addWidget(ioGroup);

    ecvAICoreUi::setupProgressSection(main, m_downloadLabel, m_progress);

    auto* btnRow = new QHBoxLayout;
    btnRow->setSpacing(ecvAICoreUi::hSpacing());
    m_testDataBtn = ecvAICoreUi::makeSampleDataBtn(this);
    m_testDataBtn->setToolTip(
            tr("Load deeplsd_examples.jpg from the shared test-data cache"));
    connect(m_testDataBtn, &QPushButton::clicked, this,
            &DeepLSDDialog::onUseTestData);
    m_runBtn = new QPushButton(tr("Run"));
    m_cancelBtn = new QPushButton(tr("Cancel"));
    m_cancelBtn->setEnabled(false);
    connect(m_runBtn, &QPushButton::clicked, this, &DeepLSDDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &DeepLSDDialog::onCancel);
    btnRow->addStretch();
    btnRow->addWidget(m_testDataBtn);
    btnRow->addWidget(m_runBtn);
    btnRow->addWidget(m_cancelBtn);
    main->addLayout(btnRow);
}void DeepLSDDialog::populateModelCombo(const QString& keepFilename) {
    const QString cache = modelCacheDir();
    QString selected = keepFilename;
    if (selected.isEmpty() && m_modelCombo && m_modelCombo->count() > 0) {
        selected = m_modelCombo->currentData().toString();
    }

    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const auto& m : builtinModels()) {
        const QFileInfo fi(cache + "/" + m.filename);
        const QString suffix =
                isValidCachedGguf(fi)
                        ? QString(" [%1] \u2713").arg(formatFileSize(fi.size()))
                        : QString(" [download]");
        m_modelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectModelByFilename(selected);
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());
}

bool DeepLSDDialog::selectModelByFilename(const QString& filename) {
    if (!m_modelCombo || filename.isEmpty()) return false;
    for (int i = 0; i < m_modelCombo->count(); ++i) {
        if (m_modelCombo->itemData(i).toString() == filename) {
            m_modelCombo->setCurrentIndex(i);
            return true;
        }
    }
    return false;
}

void DeepLSDDialog::refreshModelList() { populateModelCombo(); }

QString DeepLSDDialog::formatFileSize(qint64 bytes) {
    return ecvModelDownloader::formatFileSize(bytes);
}

DeepLSDDialog::Settings DeepLSDDialog::getSettings() const {
    Settings s;
    s.modelPath = resolveModelPath();
    s.inputPath = m_imagePath->text().trimmed();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.minSegmentScore = static_cast<float>(m_minSegmentScore->value());
    s.addLineVizToDb = m_addLineVizCheck->isChecked();
    s.addDistanceOverlayToDb = m_addDistanceOverlayCheck->isChecked();
    s.exportPolylinesToDb = m_exportPolylinesCheck->isChecked();
    return s;
}

void DeepLSDDialog::saveSettings() {
    QSettings settings;
    const QString prefix = QStringLiteral("qDeepLSD");
    settings.setValue(prefix + "/modelFilename",
                      m_modelCombo->currentData().toString());
    settings.setValue(prefix + "/customModelPath",
                      m_customModelPath->text().trimmed());
    settings.setValue(prefix + "/device",
                      m_deviceCombo->currentData().toString());
    settings.setValue(prefix + "/threads", m_threads->value());
    settings.setValue(prefix + "/minSegmentScore", m_minSegmentScore->value());
    settings.setValue(prefix + "/addLineViz", m_addLineVizCheck->isChecked());
    settings.setValue(prefix + "/addDistanceOverlay",
                      m_addDistanceOverlayCheck->isChecked());
    settings.setValue(prefix + "/exportPolylines",
                      m_exportPolylinesCheck->isChecked());
    settings.setValue(prefix + "/imagePath", m_imagePath->text().trimmed());
}

void DeepLSDDialog::restoreSettings() {
    QSettings settings;
    const QString prefix = QStringLiteral("qDeepLSD");

    const QString modelFilename =
            settings.value(prefix + "/modelFilename").toString();
    if (!modelFilename.isEmpty()) {
        selectModelByFilename(modelFilename);
    }

    const QString customModelPath =
            settings.value(prefix + "/customModelPath").toString();
    if (!customModelPath.isEmpty()) {
        m_customModelPath->setText(customModelPath);
    }

    const QString device = settings.value(prefix + "/device").toString();
    if (!device.isEmpty()) {
        for (int i = 0; i < m_deviceCombo->count(); ++i) {
            if (m_deviceCombo->itemData(i).toString() == device) {
                m_deviceCombo->setCurrentIndex(i);
                break;
            }
        }
    }

    if (settings.contains(prefix + "/threads")) {
        m_threads->setValue(settings.value(prefix + "/threads").toInt());
    }
    if (settings.contains(prefix + "/minSegmentScore")) {
        m_minSegmentScore->setValue(
                settings.value(prefix + "/minSegmentScore").toDouble());
    }
    if (settings.contains(prefix + "/addLineViz")) {
        m_addLineVizCheck->setChecked(
                settings.value(prefix + "/addLineViz").toBool());
    }
    if (settings.contains(prefix + "/addDistanceOverlay")) {
        m_addDistanceOverlayCheck->setChecked(
                settings.value(prefix + "/addDistanceOverlay").toBool());
    }
    if (settings.contains(prefix + "/exportPolylines")) {
        m_exportPolylinesCheck->setChecked(
                settings.value(prefix + "/exportPolylines").toBool());
    }

    const QString imagePath = settings.value(prefix + "/imagePath").toString();
    if (!imagePath.isEmpty()) {
        m_imagePath->setText(imagePath);
    }
}

QString DeepLSDDialog::resolveModelPath() const {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return m_customModelPath->text().trimmed();
    return modelCacheDir() + "/" + data;
}

void DeepLSDDialog::appendLog(const QString& msg) {
    aicore_inference_log::log(msg);
}

void DeepLSDDialog::setProgress(int current, int total) {
    m_progress->setMaximum(total);
    m_progress->setValue(current);
}

void DeepLSDDialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running);
    m_cancelBtn->setEnabled(running);
}

void DeepLSDDialog::setDbImages(const QList<DbImageEntry>& images) {
    m_dbImageList->clear();
    if (images.isEmpty()) {
        m_dbToggleBtn->setText(tr("DB Source Images (optional)"));
        return;
    }
    for (const auto& entry : images) {
        auto* item = new QListWidgetItem(entry.name);
        if (!entry.preview.isNull()) {
            item->setIcon(QIcon(QPixmap::fromImage(entry.preview)
                                        .scaled(48, 48, Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation)));
            // Full-resolution image for the click-to-enlarge preview.
            item->setData(kDbFullImageRole, entry.preview);
        }
        m_dbImageList->addItem(item);
    }
    m_dbToggleBtn->setText(tr("DB Source Images (%1)").arg(images.size()));
}

void DeepLSDDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    const QString name = imageNames.first();
    for (int i = 0; i < m_dbImageList->count(); ++i) {
        if (m_dbImageList->item(i)->text() == name) {
            m_dbImageList->setCurrentRow(i);
            break;
        }
    }
    m_imagePath->setText(QStringLiteral("db://") + name);
    appendLog(tr("[Info] Assigned DB image '%1'.").arg(name));
}

void DeepLSDDialog::updateImagePreview() {
    const QString path = m_imagePath->text().trimmed();
    if (path.startsWith(QStringLiteral("db://"))) {
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            if (m_dbImageList->item(i)->text() == path.mid(5)) {
                // Use the stored full-resolution image so the enlarged
                // preview shows the original pixels (not the 48 px icon).
                const QVariant full =
                        m_dbImageList->item(i)->data(kDbFullImageRole);
                if (full.canConvert<QImage>()) {
                    const QImage fullImg = full.value<QImage>();
                    if (!fullImg.isNull()) {
                        m_previewLabel->setPreviewImage(fullImg, ecvAICoreUi::previewSize());
                        return;
                    }
                }
                const QIcon icon = m_dbImageList->item(i)->icon();
                if (!icon.isNull()) {
                    m_previewLabel->setPreviewPixmap(
                            icon.pixmap(ecvAICoreUi::previewSize(), ecvAICoreUi::previewSize()), ecvAICoreUi::previewSize());
                    return;
                }
            }
        }
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("DB"));
        return;
    }
    if (path.isEmpty() || !isSupportedImageFile(path)) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    QImage img(path);
    if (img.isNull()) {
        m_previewLabel->clearPreview();
        m_previewLabel->setText(tr("?"));
        return;
    }
    m_previewLabel->setPreviewImage(img, ecvAICoreUi::previewSize());
}

void DeepLSDDialog::onBrowseImage() {
    QSettings settings;
    const QString lastDir =
            settings.value("qDeepLSD/lastImageFileDir", QDir::homePath())
                    .toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select image"), lastDir,
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"));
    if (path.isEmpty()) return;
    settings.setValue("qDeepLSD/lastImageFileDir",
                      QFileInfo(path).absolutePath());
    m_imagePath->setText(path);
}

void DeepLSDDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir =
            settings.value("qDeepLSD/lastModelDir", modelCacheDir()).toString();
    const QString path = cvFileDialog::getOpenFileName(
            this, tr("Select GGUF"), lastDir, tr("GGUF (*.gguf)"));
    if (path.isEmpty()) return;
    settings.setValue("qDeepLSD/lastModelDir", QFileInfo(path).absolutePath());
    m_customModelPath->setText(path);
    onModelComboChanged(m_modelCombo->currentIndex());
}

void DeepLSDDialog::onModelComboChanged(int index) {
    const QString data = m_modelCombo->itemData(index).toString();
    m_customModelRow->setVisible(data == "CUSTOM");

    QString variantHint;
    if (data.contains(QStringLiteral("wireframe"), Qt::CaseInsensitive)) {
        variantHint =
                tr("Wireframe model \u2014 trained on indoor/wireframe scenes "
                   "(synthetic "
                   "wireframe + ScanNet). Best for structured indoor geometry, "
                   "CAD-like "
                   "edges, and man-made environments.");
    } else if (data.contains(QStringLiteral("deeplsd_md"),
                             Qt::CaseInsensitive) ||
               data.contains(QStringLiteral("megadepth"),
                             Qt::CaseInsensitive)) {
        variantHint = tr(
                "MegaDepth (md) model \u2014 trained on outdoor phototourism "
                "(MegaDepth). "
                "Best for natural scenes, facades, and general outdoor/street "
                "photography.");
    } else if (data == "CUSTOM") {
        const QString path = m_customModelPath->text();
        if (path.contains(QStringLiteral("wireframe"), Qt::CaseInsensitive)) {
            variantHint =
                    tr("Custom wireframe checkpoint \u2014 prefer indoor/man-made "
                       "scenes.");
        } else if (path.contains(QStringLiteral("_md"), Qt::CaseInsensitive)) {
            variantHint =
                    tr("Custom MegaDepth checkpoint \u2014 prefer outdoor/natural "
                       "scenes.");
        } else {
            variantHint =
                    tr("Custom GGUF: use deeplsd_wireframe-* for indoor, "
                       "deeplsd_md-* for "
                       "outdoor. See MODEL_CARD.md.");
        }
    }
    if (m_variantHintLabel) {
        m_variantHintLabel->setText(variantHint);
        m_variantHintLabel->setVisible(!variantHint.isEmpty());
    }
}

void DeepLSDDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_imagePath->setText(QStringLiteral("db://") + item->text());
}

void DeepLSDDialog::onUseTestData() {
    if (m_testDataDownloadInProgress) return;
    if (m_downloadInProgress) {
        appendLog(tr("[Test data] Wait for the model download to finish."));
        return;
    }

    const auto kind = ecvTestDataRepository::Dataset::ObjectsDetection;
    if (!ecvTestDataRepository::findDatasetFile(
                 kind, QString::fromLatin1(kDeepLSDTestImage))
                 .isEmpty()) {
        loadTestImage();
        return;
    }

    auto& repo = ecvTestDataRepository::instance();
    if (repo.isDownloadInProgress()) {
        appendLog(tr("[Test data] Another test-data download is running."));
        return;
    }

    m_testDataDownloadInProgress = true;
    m_testDataBtn->setEnabled(false);
    m_downloadLabel->setVisible(true);
    const auto info = ecvTestDataRepository::getDatasetInfo(kind);
    if (ecvTestDataRepository::verifyZipIntegrity(
                ecvTestDataRepository::zipPath(kind), info.expectedMd5,
                info.expectedSize)) {
        m_downloadLabel->setText(tr("Extracting cached test data..."));
        repo.extractDataset(kind);
        return;
    }

    m_downloadLabel->setText(tr("Downloading shared test data..."));
    repo.startDownload(kind);
}

void DeepLSDDialog::loadTestImage() {
    const QString path = ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::ObjectsDetection,
            QString::fromLatin1(kDeepLSDTestImage));
    if (path.isEmpty()) {
        appendLog(tr("[Test data] deeplsd_examples.jpg was not found."));
        return;
    }
    m_imagePath->setText(path);
    updateImagePreview();
    appendLog(tr("[Test data] Loaded %1").arg(path));
}

bool DeepLSDDialog::ensureModelAvailable() {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return !resolveModelPath().isEmpty();
    const QString cached = modelCacheDir() + "/" + data;
    const QFileInfo cachedInfo(cached);
    if (isValidCachedGguf(cachedInfo)) return true;
    if (cachedInfo.exists()) {
        QFile::remove(cached);
        appendLog(
                tr("[Warning] Removed incomplete model cache: %1").arg(cached));
    }
    for (const auto& bm : builtinModels()) {
        if (bm.filename != data) continue;
        if (QMessageBox::question(
                    this, tr("Download Model"),
                    tr("Download '%1' now?").arg(bm.displayName)) !=
            QMessageBox::Yes) {
            return false;
        }
        m_autoRunAfterDownload = true;
        startDownload(bm);
        return false;
    }
    return false;
}

void DeepLSDDialog::startDownload(const DeepLSDBuiltinModel& model) {
    if (m_downloadInProgress || !m_downloader) {
        if (m_downloadInProgress) {
            appendLog(tr("[Warning] A download is already in progress."));
        }
        return;
    }
    QDir().mkpath(modelCacheDir());
    const QString dest = modelCacheDir() + "/" + model.filename;
    m_downloadInProgress = true;
    m_downloadLabel->setText(tr("Downloading %1 ...").arg(model.filename));
    m_downloadLabel->setVisible(true);
    m_progress->setValue(0);

    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    m_downloader->download(req);
}

void DeepLSDDialog::cancelDownload() {
    if (m_downloader) m_downloader->cancel();
    m_downloadInProgress = false;
    m_downloadLabel->setVisible(false);
}

void DeepLSDDialog::onRun() {
    if (!ensureModelAvailable()) return;
    emit runRequested(getSettings());
}

void DeepLSDDialog::onCancel() { emit cancelRequested(); }

void DeepLSDDialog::closeEvent(QCloseEvent* event) {
    saveSettings();
    onCancel();
    QDialog::closeEvent(event);
}