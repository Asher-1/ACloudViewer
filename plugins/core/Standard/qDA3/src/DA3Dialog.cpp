// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DA3Dialog.h"

#include <CVLog.h>
#include <QtCompat.h>

#include <QCloseEvent>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSettings>
#include <QStandardItemModel>
#include <QVBoxLayout>

#include "aicore/backend_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/inference_log.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"
#include "ecvPersistentSettings.h"
#include "ecvTestDataRepository.h"

namespace {

bool isSupportedImageFile(const QString& filePath) {
    static const QStringList extensions = {
            QStringLiteral("png"),  QStringLiteral("jpg"),
            QStringLiteral("jpeg"), QStringLiteral("bmp"),
            QStringLiteral("tif"),  QStringLiteral("tiff"),
            QStringLiteral("webp"), QStringLiteral("gif"),
            QStringLiteral("ppm"),  QStringLiteral("pgm"),
            QStringLiteral("pbm"),  QStringLiteral("heic"),
            QStringLiteral("heif"),
    };
    return extensions.contains(QFileInfo(filePath).suffix(),
                               Qt::CaseInsensitive);
}

QString imageFileDialogFilter() {
    return QStringLiteral(
            "Images (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP "
            "*.tif *.TIF *.tiff *.TIFF *.webp *.WEBP *.gif *.GIF "
            "*.ppm *.PPM *.pgm *.PGM *.pbm *.PBM *.heic *.HEIC *.heif *.HEIF);;"
            "All Files (*)");
}

QStringList listImageFilesInDir(const QString& dirPath) {
    QStringList files;
    QDirIterator it(dirPath, QDir::Files, QDirIterator::NoIteratorFlags);
    while (it.hasNext()) {
        const QString path = it.next();
        if (isSupportedImageFile(path)) {
            files.append(path);
        }
    }
    files.sort(Qt::CaseInsensitive);
    return files;
}

}  // namespace

static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "DA3/";

static const int kInputPreviewSize = 96;

QVector<DA3BuiltinModel> DA3Dialog::builtinModels() {
    const QString base = QString::fromLatin1(kDownloadBase);
    return {
            {tr("Base Q8_0 (recommended)"), "depth-anything-base-q8_0.gguf",
             base + "depth-anything-base-q8_0.gguf"},
            {tr("Base Q4_K (smallest)"), "depth-anything-base-q4_k.gguf",
             base + "depth-anything-base-q4_k.gguf"},
            {tr("Base F16 (half precision)"), "depth-anything-base-f16.gguf",
             base + "depth-anything-base-f16.gguf"},
            {tr("Large Q8_0 (better quality)"),
             "depth-anything-large-q8_0.gguf",
             base + "depth-anything-large-q8_0.gguf"},
            {tr("Large Q4_K (compact)"), "depth-anything-large-q4_k.gguf",
             base + "depth-anything-large-q4_k.gguf"},
            {tr("Giant Q8_0 (best quality)"), "depth-anything-giant-q8_0.gguf",
             base + "depth-anything-giant-q8_0.gguf"},
            {tr("Giant Q4_K (balanced)"), "depth-anything-giant-q4_k.gguf",
             base + "depth-anything-giant-q4_k.gguf"},
    };
}

QVector<DA3BuiltinModel> DA3Dialog::builtinMetricModels() {
    const QString base = QString::fromLatin1(kDownloadBase);
    return {
            {tr("Nested Metric F32"), "depth-anything-nested-metric.gguf",
             base + "depth-anything-nested-metric.gguf"},
            {tr("Nested AnyView Q8_0"),
             "depth-anything-nested-anyview-q8_0.gguf",
             base + "depth-anything-nested-anyview-q8_0.gguf"},
            {tr("Nested AnyView Q4_K"),
             "depth-anything-nested-anyview-q4_k.gguf",
             base + "depth-anything-nested-anyview-q4_k.gguf"},
    };
}

QString DA3Dialog::modelCacheDir() {
    char* dir = aicore_depth_model_cache_dir();
    QString result = QString::fromUtf8(dir);
    aicore_depth_free_string(dir);
    return result;
}

void DA3Dialog::updateInputPreview() {
    if (!m_inputPreview) return;
    const QString path =
            m_inputPath ? m_inputPath->text().trimmed() : QString();
    if (path.isEmpty()) {
        m_inputPreview->clearPreview();
        m_inputPreview->setText(tr("Preview"));
        return;
    }

    QImage img;
    const QFileInfo fi(path);
    if (fi.isDir()) {
        const QStringList files = listImageFilesInDir(path);
        if (!files.isEmpty()) {
            img = QImage(files.first());
        }
    } else if (isSupportedImageFile(path)) {
        img = QImage(path);
    }

    if (img.isNull()) {
        m_inputPreview->clearPreview();
        m_inputPreview->setText(tr("Preview"));
        return;
    }
    m_inputPreview->setPreviewImage(img, kInputPreviewSize);
}

DA3Dialog::DA3Dialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle("Depth Anything V3");
    setMinimumWidth(620);
    setupUi();
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &DA3Dialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0) {
                    m_progressBar->setValue(
                            static_cast<int>(received * 100 / total));
                }
                if (m_downloadLabel) {
                    m_downloadLabel->setText(
                            tr("Downloading %1 — %2")
                                    .arg(m_downloadTargetFilename)
                                    .arg(ecvModelDownloader::
                                                 formatDownloadProgress(
                                                         received, total)));
                }
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                Q_UNUSED(dest);
                const QString keepModel =
                        m_modelCombo->currentData().toString();
                const QString keepMetric =
                        m_metricModelCombo->currentData().toString();
                const bool shouldAutoRun = m_autoRunAfterDownload;
                m_downloadInProgress = false;
                m_downloadLabel->setVisible(false);
                m_progressBar->setValue(ok ? 100 : 0);
                populateModelCombos(keepModel, keepMetric);

                if (ok && !m_downloadQueue.isEmpty()) {
                    startDownload(m_downloadQueue.takeFirst());
                } else {
                    m_runBtn->setEnabled(true);
                    m_cancelBtn->setEnabled(false);
                    if (ok && shouldAutoRun && m_downloadQueue.isEmpty()) {
                        m_autoRunAfterDownload = false;
                        onRun();
                    } else if (!ok) {
                        m_downloadQueue.clear();
                        m_autoRunAfterDownload = false;
                    }
                }
            });
    CVLog::Print(QString("[DA3] Model cache: %1").arg(modelCacheDir()));
    aicore_inference_log::log_backend_probe(QStringLiteral("DA3"));
}

void DA3Dialog::setAppInterface(ecvMainAppInterface* app) { m_app = app; }

void DA3Dialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);

    // --- Mode selection ---
    auto* modeGroup = new QGroupBox("Operation Mode");
    auto* modeLayout = new QHBoxLayout(modeGroup);
    m_modeCombo = new QComboBox;
    m_modeCombo->addItem("Depth (Single Image)",
                         static_cast<int>(Mode::DepthSingle));
    m_modeCombo->addItem("Depth + Pose", static_cast<int>(Mode::DepthPose));
    m_modeCombo->addItem("Multi-View Depth + Pose",
                         static_cast<int>(Mode::DepthMultiView));
    m_modeCombo->addItem("3D Reconstruct (Gaussian)",
                         static_cast<int>(Mode::Reconstruct));
    m_modeCombo->addItem("Export GLB", static_cast<int>(Mode::ExportGLB));
    m_modeCombo->addItem("Export COLMAP", static_cast<int>(Mode::ExportCOLMAP));
    m_modeCombo->addItem("Quantize Model", static_cast<int>(Mode::Quantize));
    m_modeCombo->addItem("Model Info", static_cast<int>(Mode::ModelInfo));
    modeLayout->addWidget(new QLabel("Mode:"));
    modeLayout->addWidget(m_modeCombo, 1);
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DA3Dialog::onModeChanged);
    mainLayout->addWidget(modeGroup);

    // --- Model selection ---
    auto* modelGroup = new QGroupBox("Model");
    auto* modelLayout = new QGridLayout(modelGroup);

    modelLayout->addWidget(new QLabel("GGUF Model:"), 0, 0);
    m_modelCombo = new QComboBox;
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    modelLayout->addWidget(m_modelCombo, 0, 1, 1, 2);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DA3Dialog::onModelComboChanged);

    m_customModelRow = new QWidget;
    auto* customModelLayout = new QHBoxLayout(m_customModelRow);
    customModelLayout->setContentsMargins(0, 0, 0, 0);
    m_customModelPath = new QLineEdit;
    m_customModelPath->setPlaceholderText("Path to custom .gguf file");
    customModelLayout->addWidget(m_customModelPath, 1);
    m_browseCustomModelBtn = new QPushButton("Browse...");
    connect(m_browseCustomModelBtn, &QPushButton::clicked, this,
            &DA3Dialog::onBrowseCustomModel);
    customModelLayout->addWidget(m_browseCustomModelBtn);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 1, 0, 1, 3);

    m_metricLabel = new QLabel("Metric Model:");
    modelLayout->addWidget(m_metricLabel, 2, 0);
    m_metricModelCombo = new QComboBox;
    m_metricModelCombo->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Fixed);
    modelLayout->addWidget(m_metricModelCombo, 2, 1, 1, 2);
    connect(m_metricModelCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &DA3Dialog::onMetricModelComboChanged);

    m_customMetricRow = new QWidget;
    auto* customMetricLayout = new QHBoxLayout(m_customMetricRow);
    customMetricLayout->setContentsMargins(0, 0, 0, 0);
    m_customMetricPath = new QLineEdit;
    m_customMetricPath->setPlaceholderText("Path to custom metric .gguf file");
    customMetricLayout->addWidget(m_customMetricPath, 1);
    m_browseCustomMetricBtn = new QPushButton("Browse...");
    connect(m_browseCustomMetricBtn, &QPushButton::clicked, this,
            &DA3Dialog::onBrowseCustomMetricModel);
    customMetricLayout->addWidget(m_browseCustomMetricBtn);
    m_customMetricRow->setVisible(false);
    modelLayout->addWidget(m_customMetricRow, 3, 0, 1, 3);

    modelLayout->addWidget(new QLabel("Device:"), 4, 0);
    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* d = aicore_device_at(i);
        m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
        if (d->is_default) m_deviceCombo->setCurrentIndex(i);
    }
    m_deviceCombo->setToolTip(
            tr("Auto tries %1.").arg(aicore_auto_device_order()));
    modelLayout->addWidget(m_deviceCombo, 4, 1);

    modelLayout->addWidget(new QLabel("Threads:"), 5, 0);
    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText("Auto");
    modelLayout->addWidget(m_threads, 5, 1);

    mainLayout->addWidget(modelGroup);

    populateModelCombos();

    // --- I/O configuration ---
    auto* ioGroup = new QGroupBox("Input / Output");
    auto* ioLayout = new QGridLayout(ioGroup);

    int row = 0;
    ioLayout->addWidget(new QLabel("Input:"), row, 0);
    m_inputPath = new QLineEdit;
    m_inputPath->setPlaceholderText("Image file(s) or directory path");
    connect(m_inputPath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateInputPreview(); });
    ioLayout->addWidget(m_inputPath, row, 1);
    auto* inputBtnLayout = new QHBoxLayout;
    auto* browseFileBtn = new QPushButton("File...");
    browseFileBtn->setToolTip("Select one or more image files");
    connect(browseFileBtn, &QPushButton::clicked, this,
            &DA3Dialog::onBrowseFile);
    inputBtnLayout->addWidget(browseFileBtn);
    auto* browseFolderBtn = new QPushButton("Folder...");
    browseFolderBtn->setToolTip(
            "Select a folder for batch depth estimation on all images inside");
    connect(browseFolderBtn, &QPushButton::clicked, this,
            &DA3Dialog::onBrowseFolder);
    inputBtnLayout->addWidget(browseFolderBtn);
    inputBtnLayout->setContentsMargins(0, 0, 0, 0);
    inputBtnLayout->setSpacing(4);
    auto* inputBtnWidget = new QWidget;
    inputBtnWidget->setLayout(inputBtnLayout);
    ioLayout->addWidget(inputBtnWidget, row, 2);

    row++;
    m_inputPreview = new ecvClickableImageLabel;
    m_inputPreview->setFixedSize(kInputPreviewSize, kInputPreviewSize);
    m_inputPreview->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_inputPreview->setText(tr("Preview"));
    ioLayout->addWidget(ecvClickableImageLabel::wrapWithTapToPreviewHint(
                                m_inputPreview, this),
                        row, 0, 1, 3, Qt::AlignLeft);

    row++;
    m_dbToggleBtn = new QToolButton;
    m_dbToggleBtn->setArrowType(Qt::RightArrow);
    m_dbToggleBtn->setCheckable(true);
    m_dbToggleBtn->setChecked(false);
    m_dbToggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_dbToggleBtn->setText(tr("DB Images"));
    m_dbToggleBtn->setCursor(Qt::PointingHandCursor);
    m_dbToggleBtn->setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 4px 6px; "
            "  border-radius: 3px; color: palette(text); }"
            "QToolButton:hover { background: palette(midlight); }");
    ioLayout->addWidget(m_dbToggleBtn, row, 0);

    m_dbContentWidget = new QWidget;
    auto* dbContentLayout = new QHBoxLayout(m_dbContentWidget);
    dbContentLayout->setContentsMargins(2, 2, 2, 2);
    m_dbImageCombo = new QComboBox;
    m_dbImageCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_dbImageCombo->setToolTip(
            "Select an image already loaded in the viewer's DB tree");
    m_dbImageCombo->addItem(tr("(no images in DB)"));
    m_dbImageCombo->setEnabled(false);
    connect(m_dbImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DA3Dialog::onDbImageSelected);
    dbContentLayout->addWidget(m_dbImageCombo, 1);
    auto* refreshDbBtn = new QPushButton("Refresh");
    refreshDbBtn->setToolTip("Refresh list of images from DB tree");
    connect(refreshDbBtn, &QPushButton::clicked, this,
            &DA3Dialog::refreshDbImagesRequested);
    dbContentLayout->addWidget(refreshDbBtn);
    m_dbContentWidget->setVisible(false);
    ioLayout->addWidget(m_dbContentWidget, row, 1, 1, 2);

    connect(m_dbToggleBtn, &QToolButton::toggled, this, [this](bool checked) {
        m_dbToggleBtn->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
        m_dbContentWidget->setVisible(checked);
    });

    row++;
    ioLayout->addWidget(new QLabel("Output Dir:"), row, 0);
    m_outputDir = new QLineEdit;
    m_outputDir->setPlaceholderText("Output directory for depth maps / export");
    ioLayout->addWidget(m_outputDir, row, 1);
    auto* browseOutputBtn = new QPushButton("Browse...");
    connect(browseOutputBtn, &QPushButton::clicked, this,
            &DA3Dialog::onBrowseOutput);
    ioLayout->addWidget(browseOutputBtn, row, 2);

    row++;
    m_invertCheck = new QCheckBox("Invert depth visualization");
    m_invertCheck->setChecked(true);
    ioLayout->addWidget(m_invertCheck, row, 0, 1, 2);

    row++;
    m_unprojectCheck = new QCheckBox("3D unprojection (requires pose)");
    m_unprojectCheck->setChecked(true);
    m_unprojectCheck->setToolTip(
            "When enabled with Depth+Pose mode, generates real 3D point "
            "clouds using camera intrinsics and depth values.");
    ioLayout->addWidget(m_unprojectCheck, row, 0, 1, 2);

    row++;
    m_downsampleLabel = new QLabel("Downsample step:");
    ioLayout->addWidget(m_downsampleLabel, row, 0);
    m_downsampleStep = new QSpinBox;
    m_downsampleStep->setRange(1, 16);
    m_downsampleStep->setValue(2);
    m_downsampleStep->setToolTip(
            "Sample every Nth pixel. 1=full resolution, 2=every 2nd pixel.");
    ioLayout->addWidget(m_downsampleStep, row, 1);

    row++;
    m_colmapBinaryCheck = new QCheckBox("COLMAP binary format");
    m_colmapBinaryCheck->setChecked(true);
    ioLayout->addWidget(m_colmapBinaryCheck, row, 0, 1, 2);

    mainLayout->addWidget(ioGroup);

    // --- Quantize group ---
    m_quantGroup = new QGroupBox("Quantization");
    auto* quantLayout = new QGridLayout(m_quantGroup);
    quantLayout->addWidget(new QLabel("Input GGUF:"), 0, 0);
    m_quantInput = new QLineEdit;
    quantLayout->addWidget(m_quantInput, 0, 1);
    quantLayout->addWidget(new QLabel("Output GGUF:"), 1, 0);
    m_quantOutput = new QLineEdit;
    quantLayout->addWidget(m_quantOutput, 1, 1);
    quantLayout->addWidget(new QLabel("Type:"), 2, 0);
    m_quantType = new QComboBox;
    m_quantType->addItems({"q8_0", "q4_k", "q4_0", "q5_0", "q5_1"});
    quantLayout->addWidget(m_quantType, 2, 1);
    m_quantGroup->setVisible(false);
    mainLayout->addWidget(m_quantGroup);

    // --- Download / Progress ---
    m_downloadLabel = new QLabel;
    m_downloadLabel->setVisible(false);
    mainLayout->addWidget(m_downloadLabel);

    m_progressBar = new QProgressBar;
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    mainLayout->addWidget(m_progressBar);

    // --- Buttons ---
    auto* btnLayout = new QHBoxLayout;
    btnLayout->addStretch();

    m_useTestDataBtn = new QPushButton("Use test data");
    m_useTestDataBtn->setToolTip(
            "Load Monstree multi-view test images for depth estimation.\n"
            "Downloads on first use, then cached locally.");
    connect(m_useTestDataBtn, &QPushButton::clicked, this,
            &DA3Dialog::onUseTestData);
    btnLayout->addWidget(m_useTestDataBtn);

    m_runBtn = new QPushButton("Run");
    m_runBtn->setDefault(true);
    connect(m_runBtn, &QPushButton::clicked, this, &DA3Dialog::onRun);
    btnLayout->addWidget(m_runBtn);

    m_exportDepthBtn = new QPushButton("Export Last Depth");
    m_exportDepthBtn->setEnabled(false);
    m_exportDepthBtn->setToolTip(
            "Save the last depth estimation as a grayscale image file");
    connect(m_exportDepthBtn, &QPushButton::clicked, this,
            &DA3Dialog::exportDepthRequested);
    btnLayout->addWidget(m_exportDepthBtn);

    m_exportAllBtn = new QPushButton("Export All Depths");
    m_exportAllBtn->setEnabled(false);
    m_exportAllBtn->setToolTip(
            "Export all depth maps to the output directory as images");
    connect(m_exportAllBtn, &QPushButton::clicked, this,
            &DA3Dialog::onExportAllDepths);
    btnLayout->addWidget(m_exportAllBtn);

    m_cancelBtn = new QPushButton("Cancel");
    m_cancelBtn->setEnabled(false);
    connect(m_cancelBtn, &QPushButton::clicked, this, &DA3Dialog::onCancel);
    btnLayout->addWidget(m_cancelBtn);

    m_closeBtn = new QPushButton("Close");
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::close);
    btnLayout->addWidget(m_closeBtn);

    mainLayout->addLayout(btnLayout);

    onModeChanged(0);
}

void DA3Dialog::refreshModelList() { populateModelCombos(); }

void DA3Dialog::populateModelCombos(const QString& keepModelFilename,
                                    const QString& keepMetricFilename) {
    const QString cacheDir = modelCacheDir();

    QString selectedModel = keepModelFilename;
    if (selectedModel.isEmpty() && m_modelCombo && m_modelCombo->count() > 0) {
        selectedModel = m_modelCombo->currentData().toString();
    }
    QString selectedMetric = keepMetricFilename;
    if (selectedMetric.isEmpty() && m_metricModelCombo &&
        m_metricModelCombo->count() > 0) {
        selectedMetric = m_metricModelCombo->currentData().toString();
    }

    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const auto& m : builtinModels()) {
        QString cached = cacheDir + "/" + m.filename;
        QFileInfo fi(cached);
        QString suffix;
        if (ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath())) {
            suffix =
                    QString(" [%1] \u2713")
                            .arg(ecvModelDownloader::formatFileSize(fi.size()));
        } else {
            suffix = QString(" [download]");
        }
        m_modelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_modelCombo->insertSeparator(m_modelCombo->count());
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectComboItemByFilename(m_modelCombo, selectedModel);
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());

    m_metricModelCombo->blockSignals(true);
    m_metricModelCombo->clear();
    m_metricModelCombo->addItem(tr("(None)"), "NONE");
    for (const auto& m : builtinMetricModels()) {
        QString cached = cacheDir + "/" + m.filename;
        QFileInfo fi(cached);
        QString suffix;
        if (ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath())) {
            suffix =
                    QString(" [%1] \u2713")
                            .arg(ecvModelDownloader::formatFileSize(fi.size()));
        } else {
            suffix = QString(" [download]");
        }
        m_metricModelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_metricModelCombo->insertSeparator(m_metricModelCombo->count());
    m_metricModelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectComboItemByFilename(m_metricModelCombo, selectedMetric);
    m_metricModelCombo->blockSignals(false);
    onMetricModelComboChanged(m_metricModelCombo->currentIndex());
}

bool DA3Dialog::selectComboItemByFilename(QComboBox* combo,
                                          const QString& filename) {
    if (!combo || filename.isEmpty()) return false;
    for (int i = 0; i < combo->count(); ++i) {
        if (combo->itemData(i).toString() == filename) {
            combo->setCurrentIndex(i);
            return true;
        }
    }
    return false;
}

DA3Dialog::ModelSize DA3Dialog::modelSizeFromFilename(const QString& fn) {
    if (fn.contains("giant", Qt::CaseInsensitive)) return ModelSize::Giant;
    if (fn.contains("large", Qt::CaseInsensitive)) return ModelSize::Large;
    if (fn.contains("base", Qt::CaseInsensitive)) return ModelSize::Base;
    return ModelSize::Unknown;
}

void DA3Dialog::syncModeModelConstraints() {
    const QString filename = m_modelCombo->currentData().toString();
    const ModelSize sz = modelSizeFromFilename(filename);
    const bool isGiantOrCustom =
            (sz == ModelSize::Giant || filename == "CUSTOM" ||
             sz == ModelSize::Unknown);
    const auto mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    const bool needsGiant = (mode == Mode::Reconstruct);

    auto* modeStdModel =
            qobject_cast<QStandardItemModel*>(m_modeCombo->model());
    if (modeStdModel) {
        for (int i = 0; i < m_modeCombo->count(); ++i) {
            auto m = static_cast<Mode>(m_modeCombo->itemData(i).toInt());
            auto* item = modeStdModel->item(i);
            if (!item) continue;
            if (m == Mode::Reconstruct) item->setEnabled(isGiantOrCustom);
        }
    }

    auto* modelStdModel =
            qobject_cast<QStandardItemModel*>(m_modelCombo->model());
    if (modelStdModel) {
        for (int i = 0; i < m_modelCombo->count(); ++i) {
            auto* item = modelStdModel->item(i);
            if (!item) continue;
            QString fn = m_modelCombo->itemData(i).toString();
            if (fn.isEmpty() || fn == "CUSTOM") continue;
            if (needsGiant)
                item->setEnabled(modelSizeFromFilename(fn) == ModelSize::Giant);
            else
                item->setEnabled(true);
        }
    }

    if (needsGiant && !isGiantOrCustom) {
        m_modeCombo->blockSignals(true);
        for (int i = 0; i < m_modeCombo->count(); ++i) {
            if (static_cast<Mode>(m_modeCombo->itemData(i).toInt()) ==
                Mode::DepthSingle) {
                m_modeCombo->setCurrentIndex(i);
                break;
            }
        }
        m_modeCombo->blockSignals(false);
        onModeChanged(m_modeCombo->currentIndex());
    }
}

void DA3Dialog::onModelComboChanged(int index) {
    QString data = m_modelCombo->itemData(index).toString();
    m_customModelRow->setVisible(data == "CUSTOM");
    syncModeModelConstraints();
}

void DA3Dialog::onMetricModelComboChanged(int index) {
    QString data = m_metricModelCombo->itemData(index).toString();
    m_customMetricRow->setVisible(data == "CUSTOM");
}

void DA3Dialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qDA3"),
                             QStringLiteral("lastModelDir"), modelCacheDir());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select GGUF Model"), lastDir,
            tr("GGUF Models (*.gguf);;All Files (*)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qDA3"),
                         QStringLiteral("lastModelDir"), path);
    m_customModelPath->setText(path);
}

void DA3Dialog::onBrowseCustomMetricModel() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qDA3"),
                             QStringLiteral("lastModelDir"), modelCacheDir());
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select Metric Model"), lastDir,
            tr("GGUF Models (*.gguf);;All Files (*)"));
    if (path.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qDA3"),
                         QStringLiteral("lastModelDir"), path);
    m_customMetricPath->setText(path);
}

QString DA3Dialog::resolveModelPath(QComboBox* combo,
                                    QLineEdit* customEdit) const {
    QString data = combo->currentData().toString();
    if (data == "CUSTOM") return customEdit->text();
    if (data == "NONE") return QString();
    return modelCacheDir() + "/" + data;
}

bool DA3Dialog::ensureAllModelsAvailable() {
    const QString cacheDir = modelCacheDir();
    QVector<DA3BuiltinModel> needed;

    auto collectMissing = [&](QComboBox* combo,
                              const QVector<DA3BuiltinModel>& catalog) {
        QString data = combo->currentData().toString();
        if (data == "CUSTOM" || data == "NONE") return;
        QString cached = cacheDir + "/" + data;
        if (ecvModelDownloader::isValidCachedFile(cached)) return;
        for (const auto& bm : catalog) {
            if (bm.filename == data) {
                needed.append(bm);
                break;
            }
        }
    };

    collectMissing(m_modelCombo, builtinModels());
    if (m_metricModelCombo->isVisible())
        collectMissing(m_metricModelCombo, builtinMetricModels());

    if (needed.isEmpty()) return true;

    QStringList names;
    for (const auto& m : needed) names << m.displayName;
    auto result = QMessageBox::question(
            this, tr("Download Model(s)"),
            tr("The following model(s) are not cached locally:\n\n"
               "  %1\n\nDownload them now?")
                    .arg(names.join("\n  ")),
            QMessageBox::Yes | QMessageBox::No);
    if (result != QMessageBox::Yes) {
        appendLog(
                tr("[Info] Download declined by user. Please download or "
                   "select a cached model before running."));
        return false;
    }
    m_downloadQueue = needed;
    m_autoRunAfterDownload = true;
    startDownload(m_downloadQueue.takeFirst());
    return false;
}

void DA3Dialog::startDownload(const DA3BuiltinModel& model) {
    if (m_downloadInProgress || !m_downloader) {
        if (m_downloadInProgress) {
            appendLog(tr("[Warning] A download is already in progress."));
        }
        return;
    }

    QDir().mkpath(modelCacheDir());
    const QString dest = modelCacheDir() + "/" + model.filename;

    m_downloadInProgress = true;
    m_downloadTargetFilename = model.filename;
    m_downloadLabel->setText(tr("Downloading %1 ...").arg(model.filename));
    m_downloadLabel->setVisible(true);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_runBtn->setEnabled(false);
    m_cancelBtn->setEnabled(true);

    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    m_downloader->download(req);
}

void DA3Dialog::cancelDownload() {
    if (!m_downloadInProgress) return;
    m_autoRunAfterDownload = false;
    m_downloadQueue.clear();
    if (m_downloader) m_downloader->cancel();
    m_downloadInProgress = false;
    m_downloadLabel->setVisible(false);
    m_progressBar->setValue(0);
    m_runBtn->setEnabled(true);
    m_cancelBtn->setEnabled(false);
}

void DA3Dialog::onCancel() {
    if (m_downloadInProgress) {
        cancelDownload();
        return;
    }
    emit cancelRequested();
}

void DA3Dialog::closeEvent(QCloseEvent* event) {
    onCancel();
    QDialog::closeEvent(event);
}

void DA3Dialog::setDbImages(const QStringList& imageNames) {
    m_dbImageCombo->blockSignals(true);
    m_dbImageCombo->clear();
    if (imageNames.isEmpty()) {
        m_dbImageCombo->addItem(tr("(no images in DB)"));
        m_dbImageCombo->setEnabled(false);
        if (m_dbToggleBtn) {
            m_dbToggleBtn->setText(tr("DB Images"));
            m_dbToggleBtn->setChecked(false);
        }
    } else {
        m_dbImageCombo->addItem(
                tr("-- Select from DB (%1 images) --").arg(imageNames.size()));
        for (const auto& name : imageNames) {
            m_dbImageCombo->addItem(name, name);
        }
        m_dbImageCombo->setEnabled(true);
        if (m_dbToggleBtn) {
            m_dbToggleBtn->setText(tr("DB Images (%1)").arg(imageNames.size()));
            m_dbToggleBtn->setChecked(true);
        }
    }
    m_dbImageCombo->blockSignals(false);
}

void DA3Dialog::onDbImageSelected(int index) {
    if (index <= 0) return;
    QString name = m_dbImageCombo->itemData(index).toString();
    if (!name.isEmpty()) {
        m_inputPath->setText(QString("db://%1").arg(name));
        appendLog(tr("[Info] Selected DB image: %1").arg(name));
    }
}

void DA3Dialog::onModeChanged(int index) {
    auto mode = static_cast<Mode>(m_modeCombo->itemData(index).toInt());
    bool needsMetric = (mode == Mode::DepthSingle || mode == Mode::DepthPose);
    m_metricLabel->setVisible(needsMetric);
    m_metricModelCombo->setVisible(needsMetric);
    m_customMetricRow->setVisible(
            needsMetric &&
            m_metricModelCombo->currentData().toString() == "CUSTOM");

    bool isQuantize = (mode == Mode::Quantize);
    m_quantGroup->setVisible(isQuantize);

    bool isExport = (mode == Mode::ExportGLB || mode == Mode::ExportCOLMAP);
    bool isDepth = (mode == Mode::DepthSingle || mode == Mode::DepthPose ||
                    mode == Mode::DepthMultiView);
    m_colmapBinaryCheck->setVisible(mode == Mode::ExportCOLMAP);
    m_invertCheck->setVisible(isDepth);
    m_unprojectCheck->setVisible(isDepth);
    m_downsampleLabel->setVisible(isDepth);
    m_downsampleStep->setVisible(isDepth);

    bool showDbImages = isDepth || (mode == Mode::Reconstruct);
    m_dbToggleBtn->setVisible(showDbImages);
    m_dbContentWidget->setVisible(showDbImages && m_dbToggleBtn->isChecked());

    bool showOutput =
            isExport || isQuantize || isDepth || (mode == Mode::Reconstruct);
    m_outputDir->setVisible(showOutput);

    syncModeModelConstraints();
}

void DA3Dialog::setRunning(bool running) {
    m_runBtn->setEnabled(!running && !m_downloadInProgress);
    m_cancelBtn->setEnabled(running || m_downloadInProgress);
    m_modeCombo->setEnabled(!running && !m_downloadInProgress);
    if (running) {
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
    }
}

void DA3Dialog::enableExportButtons(bool hasResults) {
    m_exportDepthBtn->setEnabled(hasResults);
    m_exportAllBtn->setEnabled(hasResults);
}

DA3Dialog::Settings DA3Dialog::getSettings() const {
    Settings s;
    s.mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    s.modelPath = resolveModelPath(m_modelCombo, m_customModelPath);
    s.metricModelPath =
            resolveModelPath(m_metricModelCombo, m_customMetricPath);
    s.inputPaths = m_inputPath->text().split(";", QtCompat::SkipEmptyParts);
    s.outputDir = m_outputDir->text();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.invertDepth = m_invertCheck->isChecked();
    s.unproject3D = m_unprojectCheck->isChecked();
    s.downsampleStep = m_downsampleStep->value();
    s.colmapBinary = m_colmapBinaryCheck->isChecked();
    s.quantInputPath = m_quantInput->text();
    s.quantOutputPath = m_quantOutput->text();
    s.quantType = m_quantType->currentText();
    if (m_dbImageCombo->currentIndex() > 0) {
        s.dbImageName = m_dbImageCombo->currentData().toString();
    }
    return s;
}

void DA3Dialog::appendLog(const QString& msg) {
    aicore_inference_log::log(msg);
}

void DA3Dialog::setProgress(int current, int total) {
    if (total <= 0) {
        m_progressBar->setRange(0, 0);
        return;
    }
    m_progressBar->setRange(0, total);
    m_progressBar->setValue(current);
}

void DA3Dialog::onBrowseFile() {
    QSettings settings;
    const QString lastDir =
            settings.value("qDA3/lastImageFileDir", QDir::homePath())
                    .toString();

    auto mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    if (mode == Mode::DepthMultiView || mode == Mode::Reconstruct) {
        QStringList paths = QFileDialog::getOpenFileNames(
                this, "Select Image(s)", lastDir, imageFileDialogFilter());
        if (paths.isEmpty()) return;

        settings.setValue("qDA3/lastImageFileDir",
                          QFileInfo(paths.first()).absolutePath());

        QStringList accepted;
        for (const QString& path : paths) {
            if (isSupportedImageFile(path)) {
                accepted.append(path);
            } else {
                appendLog(
                        tr("[Warning] Skipped unsupported file: %1").arg(path));
            }
        }
        if (accepted.isEmpty()) {
            appendLog(tr("[Warning] No supported image files selected."));
            return;
        }
        m_inputPath->setText(accepted.join(";"));
        appendLog(tr("[Info] Added %1 file(s).").arg(accepted.size()));
    } else {
        QString path = QFileDialog::getOpenFileName(
                this, "Select Input Image", lastDir, imageFileDialogFilter());
        if (path.isEmpty()) return;
        if (!isSupportedImageFile(path)) {
            appendLog(tr("[Warning] Unsupported image file: %1").arg(path));
            return;
        }
        settings.setValue("qDA3/lastImageFileDir",
                          QFileInfo(path).absolutePath());
        m_inputPath->setText(path);
    }
}

void DA3Dialog::onBrowseFolder() {
    QSettings settings;
    const QString lastDir =
            settings.value("qDA3/lastImageFolder", QDir::homePath()).toString();
    QString dir = QFileDialog::getExistingDirectory(this, "Select Image Folder",
                                                    lastDir);
    if (dir.isEmpty()) return;

    settings.setValue("qDA3/lastImageFolder", dir);

    const QStringList files = listImageFilesInDir(dir);
    if (files.isEmpty()) {
        appendLog(tr("[Warning] No image files found in: %1").arg(dir));
        return;
    }

    m_inputPath->setText(files.join(";"));
    appendLog(tr("[Info] Found %1 image(s) in folder: %2")
                      .arg(files.size())
                      .arg(dir));
}

void DA3Dialog::onExportAllDepths() {
    QSettings settings;
    QString dir = m_outputDir->text();
    if (dir.isEmpty()) {
        const QString lastDir = ecvPS::browseDir(
                settings, QStringLiteral("qDA3"),
                QStringLiteral("lastOutputDir"), QDir::homePath());
        dir = QFileDialog::getExistingDirectory(
                this, tr("Select Output Directory for Depth Maps"), lastDir);
        if (dir.isEmpty()) return;
        ecvPS::saveBrowseDir(settings, QStringLiteral("qDA3"),
                             QStringLiteral("lastOutputDir"), dir);
        m_outputDir->setText(dir);
    }
    emit exportAllDepthsRequested(dir);
}

void DA3Dialog::onBrowseOutput() {
    QSettings settings;
    const QString lastDir =
            ecvPS::browseDir(settings, QStringLiteral("qDA3"),
                             QStringLiteral("lastOutputDir"), QDir::homePath());
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select Output Directory"), lastDir);
    if (dir.isEmpty()) return;
    ecvPS::saveBrowseDir(settings, QStringLiteral("qDA3"),
                         QStringLiteral("lastOutputDir"), dir);
    m_outputDir->setText(dir);
}

void DA3Dialog::onRun() {
    if (!ensureAllModelsAvailable()) return;
    emit runRequested(getSettings());
}

void DA3Dialog::onUseTestData() {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::Monstree;

    // Check if already extracted
    const QString bundleRoot = ecvTestDataRepository::extractPath(kind);
    if (QDir(bundleRoot).exists()) {
        const QStringList images =
                ecvTestDataRepository::getMonstreeImages(bundleRoot);
        if (!images.isEmpty()) {
            m_inputPath->setText(images.join(";"));
            appendLog(tr("[Test data] Loaded %1 images from Monstree dataset")
                              .arg(images.size()));
            return;
        }
    }

    // Check if zip is cached
    if (repo.isDatasetAvailable(kind)) {
        appendLog(tr("[Test data] Extracting cached Monstree archive..."));
        m_testDataInProgress = true;
        m_useTestDataBtn->setEnabled(false);
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
        m_progressBar->setFormat(tr("Extracting... %p%"));
        m_progressBar->setVisible(true);
        repo.extractDataset(kind);
        return;
    }

    // Need to download
    if (m_testDataInProgress || m_downloadInProgress) {
        appendLog(tr("[Test data] Another operation is in progress."));
        return;
    }

    appendLog(tr("[Test data] Downloading Monstree dataset..."));
    m_testDataInProgress = true;
    m_useTestDataBtn->setEnabled(false);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setFormat(tr("Downloading... %p%"));
    m_progressBar->setVisible(true);

    // Progress connections
    auto onDownloadProgress = [this](int percent, const QString& status) {
        if (m_testDataInProgress && m_progressBar) {
            m_progressBar->setValue(percent);
            m_progressBar->setFormat(tr("Downloading... %p%"));
        }
    };
    auto onExtractionProgress = [this](int current, int total) {
        if (m_testDataInProgress && m_progressBar && total > 0) {
            if (m_progressBar->maximum() != total) {
                m_progressBar->setRange(0, total);
            }
            m_progressBar->setValue(current);
            m_progressBar->setFormat(tr("Extracting... %v/%m"));
        }
    };

    // One-shot connections
    auto onFinished = [this, &repo, kind](bool success,
                                          ecvTestDataRepository::Dataset) {
        if (!m_testDataInProgress) return;
        if (!success) {
            appendLog(tr("[Test data] Download failed."));
            m_testDataInProgress = false;
            m_useTestDataBtn->setEnabled(true);
            m_progressBar->setVisible(false);
            return;
        }
        appendLog(tr("[Test data] Extracting..."));
        m_progressBar->setValue(0);
        m_progressBar->setFormat(tr("Extracting... %p%"));
        repo.extractDataset(kind);
    };
    auto onExtracted = [this, kind](bool success,
                                    ecvTestDataRepository::Dataset) {
        if (!m_testDataInProgress) return;
        m_testDataInProgress = false;
        m_useTestDataBtn->setEnabled(true);
        m_progressBar->setVisible(false);
        if (!success) {
            appendLog(tr("[Test data] Extraction failed."));
            return;
        }
        const QString root =
                ecvTestDataRepository::extractPath(kind);
        const QStringList images =
                ecvTestDataRepository::getMonstreeImages(root);
        if (!images.isEmpty()) {
            m_inputPath->setText(images.join(";"));
            appendLog(tr("[Test data] Loaded %1 images from Monstree dataset")
                              .arg(images.size()));
        } else {
            appendLog(tr("[Test data] No images found in bundle."));
        }
    };

    connect(&repo, &ecvTestDataRepository::downloadProgress, this,
            onDownloadProgress);
    connect(&repo, &ecvTestDataRepository::extractionProgress, this,
            onExtractionProgress);
    connect(&repo,
            &ecvTestDataRepository::downloadFinished, this, onFinished);
    connect(&repo,
            &ecvTestDataRepository::extractionFinished, this, onExtracted);
    repo.startDownload(kind);
}
