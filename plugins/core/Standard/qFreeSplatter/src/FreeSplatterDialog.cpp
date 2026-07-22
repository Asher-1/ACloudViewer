// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FreeSplatterDialog.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QSettings>
#include <QVBoxLayout>

#include "aicore/backend_capi.h"
#include "aicore/gaussian_capi.h"

static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "3dgs/";

static const int kThumbSize = 96;

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

QVector<FreeSplatterBuiltinModel> FreeSplatterDialog::builtinModels() {
    const QString base = QString::fromLatin1(kDownloadBase);
    return {
            {tr("Scene Q8_0 (recommended)"), "freesplatter-scene-q8_0.gguf",
             base + "freesplatter-scene-q8_0.gguf"},
            {tr("Scene F16"), "freesplatter-scene-f16.gguf",
             base + "freesplatter-scene-f16.gguf"},
            {tr("Scene F32 (full precision)"), "freesplatter-scene-f32.gguf",
             base + "freesplatter-scene-f32.gguf"},
            {tr("Object Q8_0"), "freesplatter-object-q8_0.gguf",
             base + "freesplatter-object-q8_0.gguf"},
            {tr("Object F16"), "freesplatter-object-f16.gguf",
             base + "freesplatter-object-f16.gguf"},
            {tr("Object F32"), "freesplatter-object-f32.gguf",
             base + "freesplatter-object-f32.gguf"},
    };
}

QString FreeSplatterDialog::modelCacheDir() {
    char* dir = aicore_gaussian_model_cache_dir();
    if (!dir) {
        return QDir::homePath() +
               QStringLiteral("/cloudViewer_data/extract/freesplatter_models");
    }
    QString result = QString::fromUtf8(dir);
    aicore_gaussian_free_string(dir);
    return result;
}

QString FreeSplatterDialog::formatFileSize(qint64 bytes) {
    if (bytes < 0) return QString();
    if (bytes < 1024) return QString("%1 B").arg(bytes);
    if (bytes < 1024LL * 1024)
        return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    if (bytes < 1024LL * 1024 * 1024)
        return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
    return QString("%1 GB").arg(bytes / (1024.0 * 1024.0 * 1024.0), 0, 'f', 2);
}

FreeSplatterDialog::FreeSplatterDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle("FreeSplatter 3D Reconstruction");
    setMinimumWidth(640);
    m_netManager = new QNetworkAccessManager(this);
    setupUi();
}

void FreeSplatterDialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);

    // --- Mode selection ---
    auto* modeGroup = new QGroupBox("Operation Mode");
    auto* modeLayout = new QHBoxLayout(modeGroup);
    m_modeCombo = new QComboBox;
    m_modeCombo->addItem("3D Reconstruct (Gaussian)",
                         static_cast<int>(Mode::Reconstruct));
    m_modeCombo->addItem("Model Info", static_cast<int>(Mode::ModelInfo));
    modeLayout->addWidget(new QLabel("Mode:"));
    modeLayout->addWidget(m_modeCombo, 1);
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FreeSplatterDialog::onModeChanged);
    mainLayout->addWidget(modeGroup);

    // --- Model selection ---
    auto* modelGroup = new QGroupBox("Model");
    auto* modelLayout = new QGridLayout(modelGroup);

    modelLayout->addWidget(new QLabel("GGUF Model:"), 0, 0);
    m_modelCombo = new QComboBox;
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    modelLayout->addWidget(m_modelCombo, 0, 1, 1, 2);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FreeSplatterDialog::onModelComboChanged);

    m_customModelRow = new QWidget;
    auto* customModelLayout = new QHBoxLayout(m_customModelRow);
    customModelLayout->setContentsMargins(0, 0, 0, 0);
    m_customModelPath = new QLineEdit;
    m_customModelPath->setPlaceholderText("Path to custom .gguf file");
    connect(m_customModelPath, &QLineEdit::textChanged, this, [this]() {
        updateImageCountStatus();
        updateRunButtonState();
    });
    customModelLayout->addWidget(m_customModelPath, 1);
    m_browseCustomModelBtn = new QPushButton("Browse...");
    connect(m_browseCustomModelBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onBrowseCustomModel);
    customModelLayout->addWidget(m_browseCustomModelBtn);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 1, 0, 1, 3);

    modelLayout->addWidget(new QLabel("Device:"), 2, 0);
    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* d = aicore_device_at(i);
        m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
        if (d->is_default) m_deviceCombo->setCurrentIndex(i);
    }
    m_deviceCombo->setToolTip(
            tr("Auto tries %1.").arg(aicore_auto_device_order()));
    modelLayout->addWidget(m_deviceCombo, 2, 1);

    modelLayout->addWidget(new QLabel("Threads:"), 3, 0);
    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText("Auto");
    modelLayout->addWidget(m_threads, 3, 1);

    mainLayout->addWidget(modelGroup);

    // --- I/O configuration ---
    auto* ioGroup = new QGroupBox("Input / Output");
    auto* ioLayout = new QGridLayout(ioGroup);

    int row = 0;
    ioLayout->addWidget(new QLabel("Input:"), row, 0, Qt::AlignTop);
    auto* inputCol = new QVBoxLayout;
    auto* inputBtnLayout = new QHBoxLayout;
    auto* browseFileBtn = new QPushButton("File...");
    browseFileBtn->setToolTip(tr("Add one or more image files"));
    connect(browseFileBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onBrowseFile);
    inputBtnLayout->addWidget(browseFileBtn);
    auto* browseFolderBtn = new QPushButton("Folder...");
    browseFolderBtn->setToolTip(tr("Load all images from a folder"));
    connect(browseFolderBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onBrowseFolder);
    inputBtnLayout->addWidget(browseFolderBtn);
    auto* clearInputBtn = new QPushButton("Clear");
    connect(clearInputBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onClearInput);
    inputBtnLayout->addWidget(clearInputBtn);
    inputBtnLayout->addStretch();
    inputCol->addLayout(inputBtnLayout);

    m_thumbScroll = new QScrollArea;
    m_thumbScroll->setWidgetResizable(true);
    m_thumbScroll->setMinimumHeight(kThumbSize + 24);
    m_thumbScroll->setMaximumHeight(kThumbSize + 24);
    m_thumbScroll->setFrameShape(QFrame::StyledPanel);
    m_thumbContainer = new QWidget;
    auto* thumbLayout = new QHBoxLayout(m_thumbContainer);
    thumbLayout->setContentsMargins(4, 4, 4, 4);
    m_thumbScroll->setWidget(m_thumbContainer);
    inputCol->addWidget(m_thumbScroll);
    ioLayout->addLayout(inputCol, row, 1, 1, 2);

    row++;
    m_dbImageLabel = new QLabel("DB Images:");
    ioLayout->addWidget(m_dbImageLabel, row, 0, Qt::AlignTop);
    auto* dbCol = new QVBoxLayout;
    m_dbImageList = new QListWidget;
    m_dbImageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_dbImageList->setMinimumHeight(100);
    m_dbImageList->setMaximumHeight(140);
    m_dbImageList->setToolTip(
            tr("ccImage entities from the DB tree — check/uncheck to add or "
               "remove from input"));
    connect(m_dbImageList, &QListWidget::itemChanged, this,
            &FreeSplatterDialog::onDbListItemChanged);
    dbCol->addWidget(m_dbImageList);
    auto* dbBtnLayout = new QHBoxLayout;
    auto* refreshDbBtn = new QPushButton("Refresh");
    refreshDbBtn->setToolTip(tr("Refresh ccImage list from DB tree"));
    connect(refreshDbBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::refreshDbImagesRequested);
    dbBtnLayout->addWidget(refreshDbBtn);
    dbBtnLayout->addStretch();
    dbCol->addLayout(dbBtnLayout);
    ioLayout->addLayout(dbCol, row, 1, 1, 2);

    row++;
    ioLayout->addWidget(new QLabel("Opacity Threshold:"), row, 0);
    m_opacityThreshold = new QDoubleSpinBox;
    m_opacityThreshold->setRange(0.0, 1.0);
    m_opacityThreshold->setSingleStep(0.01);
    m_opacityThreshold->setValue(0.05);
    m_opacityThreshold->setToolTip("Prune gaussians with opacity <= threshold");
    ioLayout->addWidget(m_opacityThreshold, row, 1);

    row++;
    m_exportFieldLabel = new QLabel("Point cloud export:");
    ioLayout->addWidget(m_exportFieldLabel, row, 0);
    m_exportFieldModeCombo = new QComboBox;
    m_exportFieldModeCombo->addItem(tr("Basic — XYZ + RGB + Opacity"),
                                    static_cast<int>(ExportFieldMode::Basic));
    m_exportFieldModeCombo->addItem(tr("Full — SH + scale + thin-axis normals"),
                                    static_cast<int>(ExportFieldMode::Full));
    m_exportFieldModeCombo->setToolTip(
#ifdef HAS_QSIBR
            tr("Applies to Add to DB tree and Export PLY.\n"
               "Basic: XYZ + RGB + Opacity (grey ramp).\n"
               "Full: also exports SH and scale as scalar fields, plus "
               "normals\n"
               "from the thinnest Gaussian axis (quat+scale). Near-spherical\n"
               "points get a default normal. Use Visualize (SIBR) for full "
               "3DGS.")
#else
            tr("Applies to Add to DB tree and Export PLY.\n"
               "Basic: XYZ + RGB + Opacity (grey ramp).\n"
               "Full: also exports SH and scale as scalar fields, plus "
               "normals\n"
               "from the thinnest Gaussian axis (quat+scale). Near-spherical\n"
               "points get a default normal.")
#endif
    );
    ioLayout->addWidget(m_exportFieldModeCombo, row, 1, 1, 2);

    row++;
    m_addToDbCheck = new QCheckBox("Add result to DB tree (main 3D view)");
    m_addToDbCheck->setChecked(true);
    m_addToDbCheck->setToolTip(
            "After inference, add a colored point cloud to the database tree "
            "and zoom the main rendering window to it.");
    ioLayout->addWidget(m_addToDbCheck, row, 0, 1, 2);

    row++;
    m_estimatePosesCheck = new QCheckBox("Estimate camera poses (multi-view)");
    m_estimatePosesCheck->setChecked(false);
    ioLayout->addWidget(m_estimatePosesCheck, row, 0, 1, 2);

    row++;
    m_imageCountLabel = new QLabel;
    m_imageCountLabel->setStyleSheet("font-weight: bold;");
    ioLayout->addWidget(m_imageCountLabel, row, 0, 1, 3);

    mainLayout->addWidget(ioGroup);

    populateModelCombo();

    // --- Download / Progress ---
    m_downloadLabel = new QLabel;
    m_downloadLabel->setVisible(false);
    mainLayout->addWidget(m_downloadLabel);

    m_taskStatusLabel = new QLabel;
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    mainLayout->addWidget(m_taskStatusLabel);

    m_progressBar = new QProgressBar;
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    mainLayout->addWidget(m_progressBar);

    // --- Log ---
    m_logOutput = new QTextEdit;
    m_logOutput->setReadOnly(true);
    m_logOutput->setMaximumHeight(150);
    mainLayout->addWidget(m_logOutput);

    // --- Buttons ---
    auto* btnLayout = new QHBoxLayout;
    btnLayout->addStretch();

    m_runBtn = new QPushButton("Run");
    m_runBtn->setDefault(true);
    m_runBtn->setEnabled(false);
    connect(m_runBtn, &QPushButton::clicked, this, &FreeSplatterDialog::onRun);
    btnLayout->addWidget(m_runBtn);

#ifdef HAS_QSIBR
    m_visualizeBtn = new QPushButton("Visualize (SIBR)");
    m_visualizeBtn->setEnabled(false);
    m_visualizeBtn->setToolTip(
            "Open interactive 3D Gaussian viewer (in-memory, no disk PLY)");
    connect(m_visualizeBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onVisualize);
    btnLayout->addWidget(m_visualizeBtn);
#endif

    m_exportPlyBtn = new QPushButton("Export PLY...");
    m_exportPlyBtn->setEnabled(false);
    m_exportPlyBtn->setToolTip(
            "Export the result point cloud to PLY using CV_io (same as File > "
            "Save)");
    connect(m_exportPlyBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onExportPly);
    btnLayout->addWidget(m_exportPlyBtn);

    m_cancelBtn = new QPushButton("Cancel");
    m_cancelBtn->setEnabled(false);
    connect(m_cancelBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onCancel);
    btnLayout->addWidget(m_cancelBtn);

    m_closeBtn = new QPushButton("Close");
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::close);
    btnLayout->addWidget(m_closeBtn);

    mainLayout->addLayout(btnLayout);

    onModeChanged(0);
    refreshThumbnailStrip();
    updateRunButtonState();
}

void FreeSplatterDialog::populateModelCombo(const QString& keepFilename) {
    const QString cacheDir = modelCacheDir();
    QString selected = keepFilename;
    if (selected.isEmpty() && m_modelCombo && m_modelCombo->count() > 0) {
        selected = m_modelCombo->currentData().toString();
    }

    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const auto& m : builtinModels()) {
        QString cached = cacheDir + "/" + m.filename;
        QFileInfo fi(cached);
        QString suffix;
        if (fi.exists()) {
            suffix = QString(" [%1] \u2713").arg(formatFileSize(fi.size()));
        } else {
            suffix = QString(" [download]");
        }
        m_modelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_modelCombo->insertSeparator(m_modelCombo->count());
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectModelByFilename(selected);
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());
}

bool FreeSplatterDialog::selectModelByFilename(const QString& filename) {
    if (!m_modelCombo || filename.isEmpty()) return false;
    for (int i = 0; i < m_modelCombo->count(); ++i) {
        if (m_modelCombo->itemData(i).toString() == filename) {
            m_modelCombo->setCurrentIndex(i);
            return true;
        }
    }
    return false;
}

void FreeSplatterDialog::onModelComboChanged(int index) {
    QString data = m_modelCombo->itemData(index).toString();
    m_customModelRow->setVisible(data == "CUSTOM");
    updateImageCountStatus();
    updateRunButtonState();
}

FreeSplatterDialog::ModelType FreeSplatterDialog::modelTypeFromFilename(
        const QString& filename) {
    if (filename.contains("scene", Qt::CaseInsensitive))
        return ModelType::Scene;
    if (filename.contains("object", Qt::CaseInsensitive))
        return ModelType::Object;
    return ModelType::Unknown;
}

FreeSplatterDialog::ModelType FreeSplatterDialog::currentModelType() const {
    QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") {
        return modelTypeFromFilename(m_customModelPath->text());
    }
    return modelTypeFromFilename(data);
}

int FreeSplatterDialog::requiredImageCount() const {
    switch (currentModelType()) {
        case ModelType::Scene:
            return 2;
        case ModelType::Object:
            return 3;
        default:
            return 1;
    }
}

int FreeSplatterDialog::currentImageCount() const {
    return m_inputPaths.size();
}

bool FreeSplatterDialog::isModelReady() const {
    QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") {
        return !m_customModelPath->text().trimmed().isEmpty() &&
               QFile::exists(m_customModelPath->text().trimmed());
    }
    if (data.isEmpty()) return false;
    if (QFile::exists(modelCacheDir() + "/" + data)) return true;
    for (const auto& m : builtinModels()) {
        if (m.filename == data) return true;
    }
    return false;
}

bool FreeSplatterDialog::isInputValid() const {
    int current = currentImageCount();
    if (current == 0) return false;
    int required = requiredImageCount();
    auto type = currentModelType();
    if (type == ModelType::Scene) return current >= required;
    if (type == ModelType::Object) return current >= required;
    return current >= 1;
}

void FreeSplatterDialog::updateImageCountStatus() {
    if (!m_imageCountLabel || !m_modeCombo) return;
    if (m_modeCombo->currentData().toInt() !=
        static_cast<int>(Mode::Reconstruct)) {
        m_imageCountLabel->clear();
        return;
    }
    int current = currentImageCount();
    int required = requiredImageCount();
    auto type = currentModelType();
    QString typeName = (type == ModelType::Scene)    ? "Scene"
                       : (type == ModelType::Object) ? "Object"
                                                     : "Unknown";
    QString reqStr =
            (type == ModelType::Scene)
                    ? QString("at least %1 (recommended %1)").arg(required)
                    : QString("at least %1").arg(required);
    QString color = "gray";
    if (type == ModelType::Scene) {
        if (current >= required && current <= required)
            color = "green";
        else if (current > required)
            color = "#b7791f";
        else if (current > 0)
            color = "orange";
    } else {
        color = (current >= required) ? "green"
                : (current > 0)       ? "orange"
                                      : "gray";
    }
    m_imageCountLabel->setStyleSheet(
            QString("font-weight: bold; color: %1;").arg(color));
    QString status = QString("%1 model: %2 images selected (need %3)")
                             .arg(typeName)
                             .arg(current)
                             .arg(reqStr);
    if (type == ModelType::Scene && current > required) {
        status += tr(" — extra views may reduce quality");
    }
    m_imageCountLabel->setText(status);
}

void FreeSplatterDialog::updateRunButtonState() {
    if (!m_runBtn || !m_modeCombo) return;
    if (m_downloadInProgress) {
        m_runBtn->setEnabled(false);
        m_cancelBtn->setEnabled(true);
        return;
    }
    if (m_taskRunning) {
        m_runBtn->setEnabled(false);
        m_cancelBtn->setEnabled(true);
        return;
    }
    m_cancelBtn->setEnabled(false);
    auto mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    bool ready = isModelReady();
    if (mode == Mode::Reconstruct) {
        ready = ready && isInputValid();
    }
    m_runBtn->setEnabled(ready);
}

QImage FreeSplatterDialog::previewForPath(const QString& path) const {
    if (path.startsWith("db://")) {
        const QString name = path.mid(5);
        return m_dbPreviews.value(name);
    }
    QImage img;
    if (img.load(path)) return img;
    return QImage();
}

void FreeSplatterDialog::refreshThumbnailStrip() {
    if (!m_thumbContainer) return;
    auto* thumbLayout = qobject_cast<QHBoxLayout*>(m_thumbContainer->layout());
    if (!thumbLayout) return;
    QLayoutItem* child;
    while ((child = thumbLayout->takeAt(0)) != nullptr) {
        if (child->widget()) child->widget()->deleteLater();
        delete child;
    }

    if (m_inputPaths.isEmpty()) {
        auto* placeholder =
                new QLabel(tr("(no images — use File/Folder or DB Images)"));
        placeholder->setAlignment(Qt::AlignCenter);
        placeholder->setStyleSheet("color: gray;");
        thumbLayout->addWidget(placeholder);
        thumbLayout->addStretch();
        return;
    }

    for (const QString& path : m_inputPaths) {
        auto* tile = new QWidget;
        auto* tileLayout = new QVBoxLayout(tile);
        tileLayout->setContentsMargins(2, 2, 2, 2);
        tileLayout->setSpacing(2);

        auto* imgLabel = new QLabel;
        QImage img = previewForPath(path);
        if (!img.isNull()) {
            imgLabel->setPixmap(QPixmap::fromImage(
                    img.scaled(kThumbSize, kThumbSize, Qt::KeepAspectRatio,
                               Qt::SmoothTransformation)));
        } else {
            imgLabel->setFixedSize(kThumbSize, kThumbSize);
            imgLabel->setAlignment(Qt::AlignCenter);
            imgLabel->setText("?");
            imgLabel->setFrameShape(QFrame::Box);
        }
        imgLabel->setToolTip(path);
        tileLayout->addWidget(imgLabel, 0, Qt::AlignHCenter);

        QString caption = path.startsWith("db://") ? path.mid(5)
                                                   : QFileInfo(path).fileName();
        auto* nameLabel = new QLabel(caption);
        nameLabel->setMaximumWidth(kThumbSize + 8);
        nameLabel->setAlignment(Qt::AlignCenter);
        nameLabel->setWordWrap(true);
        tileLayout->addWidget(nameLabel);

        auto* removeBtn = new QPushButton("×");
        removeBtn->setFixedSize(20, 20);
        removeBtn->setToolTip(tr("Remove this image"));
        removeBtn->setProperty("inputPath", path);
        connect(removeBtn, &QPushButton::clicked, this,
                &FreeSplatterDialog::onRemoveInputItem);
        tileLayout->addWidget(removeBtn, 0, Qt::AlignHCenter);

        thumbLayout->addWidget(tile);
    }
    thumbLayout->addStretch();
}

void FreeSplatterDialog::addInputPaths(const QStringList& paths, bool replace) {
    if (replace) m_inputPaths.clear();
    for (const QString& p : paths) {
        if (p.isEmpty()) continue;
        if (!m_inputPaths.contains(p)) m_inputPaths.append(p);
    }
    refreshThumbnailStrip();
    updateImageCountStatus();
    updateRunButtonState();
}

void FreeSplatterDialog::removeInputPath(const QString& path) {
    m_inputPaths.removeAll(path);
    if (m_dbImageList && path.startsWith("db://")) {
        const QString name = path.mid(5);
        m_dbImageList->blockSignals(true);
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            QListWidgetItem* item = m_dbImageList->item(i);
            if (item && item->data(Qt::UserRole).toString() == name) {
                item->setCheckState(Qt::Unchecked);
            }
        }
        m_dbImageList->blockSignals(false);
    }
    refreshThumbnailStrip();
    updateImageCountStatus();
    updateRunButtonState();
}

void FreeSplatterDialog::onRemoveInputItem() {
    auto* btn = qobject_cast<QPushButton*>(sender());
    if (!btn) return;
    removeInputPath(btn->property("inputPath").toString());
}

void FreeSplatterDialog::onClearInput() {
    m_inputPaths.clear();
    if (m_dbImageList) {
        m_dbImageList->blockSignals(true);
        for (int i = 0; i < m_dbImageList->count(); ++i) {
            if (QListWidgetItem* item = m_dbImageList->item(i)) {
                item->setCheckState(Qt::Unchecked);
            }
        }
        m_dbImageList->blockSignals(false);
    }
    refreshThumbnailStrip();
    updateImageCountStatus();
    updateRunButtonState();
}

void FreeSplatterDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFreeSplatter/lastModelDir", QDir::homePath())
                    .toString();
    QString path =
            QFileDialog::getOpenFileName(this, "Select GGUF Model", lastDir,
                                         "GGUF Models (*.gguf);;All Files (*)");
    if (path.isEmpty()) return;
    settings.setValue("qFreeSplatter/lastModelDir",
                      QFileInfo(path).absolutePath());
    m_customModelPath->setText(path);
}

QString FreeSplatterDialog::resolveModelPath() const {
    QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return m_customModelPath->text();
    return modelCacheDir() + "/" + data;
}

bool FreeSplatterDialog::ensureModelAvailable() {
    QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return true;

    QString cached = modelCacheDir() + "/" + data;
    if (QFile::exists(cached)) return true;

    for (const auto& bm : builtinModels()) {
        if (bm.filename == data) {
            auto result =
                    QMessageBox::question(this, tr("Download Model"),
                                          tr("The model '%1' is not cached "
                                             "locally.\n\nDownload it now?")
                                                  .arg(bm.displayName),
                                          QMessageBox::Yes | QMessageBox::No);
            if (result != QMessageBox::Yes) {
                appendLog(
                        tr("[Info] Download declined. Please select a cached "
                           "model."));
                return false;
            }
            m_autoRunAfterDownload = true;
            startDownload(bm);
            return false;
        }
    }
    return true;
}

void FreeSplatterDialog::startDownload(const FreeSplatterBuiltinModel& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[Warning] A download is already in progress."));
        return;
    }

    QDir().mkpath(modelCacheDir());
    QString dest = modelCacheDir() + "/" + model.filename;
    QString tmpDest = dest + ".part";

    m_downloadInProgress = true;
    m_downloadTargetFilename = model.filename;
    m_downloadTmpPath = tmpDest;
    m_downloadLabel->setText(tr("Downloading %1 ...").arg(model.filename));
    m_downloadLabel->setVisible(true);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    updateRunButtonState();

    QNetworkRequest req(QUrl(model.downloadUrl));
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    m_currentDownload = m_netManager->get(req);

    m_downloadOutFile = new QFile(tmpDest, m_currentDownload);
    if (!m_downloadOutFile->open(QIODevice::WriteOnly)) {
        appendLog(tr("[Error] Cannot write to %1").arg(tmpDest));
        cancelDownload();
        return;
    }

    connect(m_currentDownload, &QNetworkReply::readyRead, this, [this]() {
        if (m_downloadOutFile && m_currentDownload) {
            m_downloadOutFile->write(m_currentDownload->readAll());
        }
    });

    connect(m_currentDownload, &QNetworkReply::downloadProgress, this,
            [this](qint64 received, qint64 total) {
                if (total > 0) {
                    m_progressBar->setValue(
                            static_cast<int>(received * 100 / total));
                    m_downloadLabel->setText(
                            tr("Downloading... %1 / %2")
                                    .arg(formatFileSize(received))
                                    .arg(formatFileSize(total)));
                }
            });

    connect(m_currentDownload, &QNetworkReply::finished, this,
            [this, dest, tmpDest, model]() {
                if (m_downloadOutFile) {
                    m_downloadOutFile->close();
                    m_downloadOutFile->deleteLater();
                    m_downloadOutFile = nullptr;
                }

                const bool canceled =
                        m_currentDownload &&
                        m_currentDownload->error() ==
                                QNetworkReply::OperationCanceledError;
                bool ok = m_currentDownload &&
                          m_currentDownload->error() == QNetworkReply::NoError;

                if (canceled) {
                    appendLog(tr("[Info] Download canceled: %1")
                                      .arg(model.filename));
                    QFile::remove(tmpDest);
                    m_autoRunAfterDownload = false;
                } else if (ok) {
                    QFile::remove(dest);
                    QFile::rename(tmpDest, dest);
                    appendLog(tr("[OK] Model downloaded: %1").arg(dest));
                } else if (m_currentDownload) {
                    appendLog(tr("[Error] Download failed: %1")
                                      .arg(m_currentDownload->errorString()));
                    QFile::remove(tmpDest);
                    m_autoRunAfterDownload = false;
                }

                if (m_currentDownload) {
                    m_currentDownload->deleteLater();
                    m_currentDownload = nullptr;
                }

                const QString finishedFilename = m_downloadTargetFilename;
                const bool shouldAutoRun = m_autoRunAfterDownload;
                m_downloadInProgress = false;
                m_downloadTmpPath.clear();
                m_downloadLabel->setVisible(false);
                m_progressBar->setValue(ok ? 100 : 0);
                populateModelCombo(finishedFilename);
                updateRunButtonState();

                if (ok && shouldAutoRun) {
                    m_autoRunAfterDownload = false;
                    selectModelByFilename(finishedFilename);
                    onRun();
                }
            });
}

void FreeSplatterDialog::cancelDownload() {
    if (!m_downloadInProgress) return;
    m_autoRunAfterDownload = false;
    if (m_currentDownload) {
        m_currentDownload->abort();
    }
}

void FreeSplatterDialog::onCancel() {
    if (m_downloadInProgress) {
        cancelDownload();
        return;
    }
    emit cancelRequested();
}

void FreeSplatterDialog::setDbImages(const QList<DbImageEntry>& images) {
    if (!m_dbImageList) return;
    m_dbPreviews.clear();
    m_dbImageList->blockSignals(true);
    m_dbImageList->clear();
    if (images.isEmpty()) {
        m_dbImageList->addItem(tr("(no ccImage entities in DB)"));
        m_dbImageList->item(0)->setFlags(Qt::NoItemFlags);
        m_dbImageList->setEnabled(false);
    } else {
        m_dbImageList->setEnabled(true);
        for (const auto& entry : images) {
            m_dbPreviews.insert(entry.name, entry.preview);
            auto* item = new QListWidgetItem(entry.name);
            item->setData(Qt::UserRole, entry.name);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            const QString dbPath = QString("db://%1").arg(entry.name);
            item->setCheckState(m_inputPaths.contains(dbPath) ? Qt::Checked
                                                              : Qt::Unchecked);
            if (!entry.preview.isNull()) {
                item->setIcon(QIcon(QPixmap::fromImage(
                        entry.preview.scaled(32, 32, Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation))));
            }
            m_dbImageList->addItem(item);
        }
    }
    m_dbImageList->blockSignals(false);
    refreshThumbnailStrip();
}

void FreeSplatterDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    QStringList dbPaths;
    for (const QString& name : imageNames) {
        dbPaths << QString("db://%1").arg(name);
    }
    addInputPaths(dbPaths, false);
    m_dbImageList->blockSignals(true);
    for (int i = 0; i < m_dbImageList->count(); ++i) {
        QListWidgetItem* item = m_dbImageList->item(i);
        if (!item) continue;
        const QString name = item->data(Qt::UserRole).toString();
        if (imageNames.contains(name)) {
            item->setCheckState(Qt::Checked);
        }
    }
    m_dbImageList->blockSignals(false);
    appendLog(tr("[Info] Added %1 image(s) from DB tree selection.")
                      .arg(imageNames.size()));
}

void FreeSplatterDialog::onDbListItemChanged(QListWidgetItem* item) {
    if (!item || !(item->flags() & Qt::ItemIsUserCheckable)) return;
    const QString name = item->data(Qt::UserRole).toString();
    if (name.isEmpty()) return;
    const QString dbPath = QString("db://%1").arg(name);
    if (item->checkState() == Qt::Checked) {
        addInputPaths({dbPath}, false);
    } else {
        removeInputPath(dbPath);
    }
}

void FreeSplatterDialog::onModeChanged(int index) {
    auto mode = static_cast<Mode>(m_modeCombo->itemData(index).toInt());
    bool isReconstruct = (mode == Mode::Reconstruct);

    m_thumbScroll->setVisible(isReconstruct);
    m_opacityThreshold->setVisible(isReconstruct);
    if (m_exportFieldLabel) m_exportFieldLabel->setVisible(isReconstruct);
    if (m_exportFieldModeCombo)
        m_exportFieldModeCombo->setVisible(isReconstruct);
    m_addToDbCheck->setVisible(isReconstruct);
    m_estimatePosesCheck->setVisible(isReconstruct);
    m_dbImageLabel->setVisible(isReconstruct);
    m_dbImageList->setVisible(isReconstruct);
    m_imageCountLabel->setVisible(isReconstruct);
    if (isReconstruct) updateImageCountStatus();
    updateRunButtonState();
}

void FreeSplatterDialog::setRunning(bool running) {
    m_taskRunning = running;
    if (running) {
        m_taskStatusLabel->setText(tr("Starting..."));
        m_taskStatusLabel->setVisible(true);
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
    } else {
        m_taskStatusLabel->clear();
        m_taskStatusLabel->setVisible(false);
    }
    updateRunButtonState();
    m_modeCombo->setEnabled(!running && !m_downloadInProgress);
}

void FreeSplatterDialog::setTaskStage(const QString& stage, int percent) {
    if (!m_taskStatusLabel) return;
    m_taskStatusLabel->setText(stage);
    m_taskStatusLabel->setVisible(true);
    if (percent >= 0) {
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(percent);
    } else {
        m_progressBar->setRange(0, 0);
    }
}

void FreeSplatterDialog::enableResultButtons(bool hasResult) {
    m_hasResult = hasResult;
#ifdef HAS_QSIBR
    if (m_visualizeBtn) m_visualizeBtn->setEnabled(hasResult);
#endif
    if (m_exportPlyBtn) m_exportPlyBtn->setEnabled(hasResult);
}

FreeSplatterDialog::Settings FreeSplatterDialog::getSettings() const {
    Settings s;
    s.mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    s.modelPath = resolveModelPath();
    s.inputPaths = m_inputPaths;
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.opacityThreshold = (float)m_opacityThreshold->value();
    s.exportFieldMode = static_cast<ExportFieldMode>(
            m_exportFieldModeCombo
                    ? m_exportFieldModeCombo->currentData().toInt()
                    : static_cast<int>(ExportFieldMode::Basic));
    s.addToDb = m_addToDbCheck->isChecked();
    s.estimatePoses = m_estimatePosesCheck->isChecked();
    return s;
}

void FreeSplatterDialog::appendLog(const QString& msg) {
    m_logOutput->append(msg);
}

void FreeSplatterDialog::setProgress(int current, int total) {
    if (total <= 0) {
        m_progressBar->setRange(0, 0);
        return;
    }
    m_progressBar->setMaximum(total);
    m_progressBar->setValue(current);
    if (total == 100) {
        m_progressBar->setValue(current);
    }
}

void FreeSplatterDialog::onBrowseFile() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFreeSplatter/lastImageFileDir", QDir::homePath())
                    .toString();
    QStringList paths = QFileDialog::getOpenFileNames(
            this, "Select Image(s)", lastDir, imageFileDialogFilter());
    if (paths.isEmpty()) return;

    settings.setValue("qFreeSplatter/lastImageFileDir",
                      QFileInfo(paths.first()).absolutePath());

    QStringList accepted;
    for (const QString& path : paths) {
        if (isSupportedImageFile(path)) {
            accepted.append(path);
        } else {
            appendLog(tr("[Warning] Skipped unsupported file: %1").arg(path));
        }
    }
    if (accepted.isEmpty()) {
        appendLog(tr("[Warning] No supported image files selected."));
        return;
    }

    addInputPaths(accepted, false);
    appendLog(tr("[Info] Added %1 file(s).").arg(accepted.size()));
}

void FreeSplatterDialog::onBrowseFolder() {
    QSettings settings;
    const QString lastDir =
            settings.value("qFreeSplatter/lastImageFolder", QDir::homePath())
                    .toString();
    QString dir = QFileDialog::getExistingDirectory(this, "Select Image Folder",
                                                    lastDir);
    if (dir.isEmpty()) return;

    settings.setValue("qFreeSplatter/lastImageFolder", dir);

    const QStringList files = listImageFilesInDir(dir);
    if (files.isEmpty()) {
        appendLog(tr("[Warning] No image files found in: %1").arg(dir));
        return;
    }

    addInputPaths(files, true);
    appendLog(tr("[Info] Loaded %1 image(s) from folder: %2")
                      .arg(files.size())
                      .arg(dir));
}

void FreeSplatterDialog::onVisualize() {
    if (m_hasResult) emit visualizeRequested();
}

void FreeSplatterDialog::onExportPly() {
    if (m_hasResult) emit exportPlyRequested();
}

void FreeSplatterDialog::onRun() {
    if (!ensureModelAvailable()) return;
    auto mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    if (mode == Mode::Reconstruct && !isInputValid()) {
        int current = currentImageCount();
        int required = requiredImageCount();
        auto type = currentModelType();
        QString reqStr = (type == ModelType::Scene)
                                 ? tr("Scene model requires at least %1 images "
                                      "(you have %2)")
                                           .arg(required)
                                           .arg(current)
                                 : tr("Object model requires at least %1 "
                                      "images (you have %2)")
                                           .arg(required)
                                           .arg(current);
        QMessageBox::warning(this, tr("Image Count"), reqStr);
        return;
    }
    if (mode == Mode::Reconstruct && currentModelType() == ModelType::Scene &&
        currentImageCount() > requiredImageCount()) {
        appendLog(tr("[Warning] Scene model is tuned for %1 views; running "
                     "with %2 "
                     "images may reduce quality.")
                          .arg(requiredImageCount())
                          .arg(currentImageCount()));
    }
    if (mode == Mode::Reconstruct && currentImageCount() > 64) {
        appendLog(
                tr("[Warning] Inference supports at most 64 views; excess "
                   "images will be uniformly subsampled before run."));
    } else if (mode == Mode::Reconstruct &&
               currentModelType() == ModelType::Object &&
               currentImageCount() > 16) {
        appendLog(
                tr("[Warning] Object model works best with ~8–16 views; "
                   "inputs above 16 will be uniformly subsampled."));
    }
    emit runRequested(getSettings());
}
