// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LightGlueDialog.h"

#include <QDir>
#include <QDirIterator>
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
#include "aicore/lightglue_capi.h"
#include "feature_extractor.h"

static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "LightGlue/";

static const int kThumbSize = 72;

namespace {

QString imageFileDialogFilter() {
    return QStringLiteral(
            "Images (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP "
            "*.tif *.TIF *.tiff *.TIFF *.webp *.WEBP);;All Files (*)");
}

bool isSupportedImageFile(const QString& filePath) {
    static const QStringList extensions = {
            QStringLiteral("png"),  QStringLiteral("jpg"),
            QStringLiteral("jpeg"), QStringLiteral("bmp"),
            QStringLiteral("tif"),  QStringLiteral("tiff"),
            QStringLiteral("webp"), QStringLiteral("gif"),
    };
    return extensions.contains(QFileInfo(filePath).suffix(),
                               Qt::CaseInsensitive);
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

QVector<LightGlueBuiltinModel> LightGlueDialog::builtinModels() {
    const QString base = QString::fromLatin1(kDownloadBase);
    return {
            {tr("SIFT F16 (recommended)"), "sift-lightglue-f16.gguf",
             base + "sift-lightglue-f16.gguf", 1},
            {tr("SIFT Q8_0 (smaller)"), "sift-lightglue-q8_0.gguf",
             base + "sift-lightglue-q8_0.gguf", 1},
            {tr("SIFT F32"), "sift-lightglue-f32.gguf",
             base + "sift-lightglue-f32.gguf", 1},
            {tr("ALIKED F16 (matcher only)"), "aliked-lightglue-f16.gguf",
             base + "aliked-lightglue-f16.gguf", 2},
            {tr("ALIKED Q8_0 (matcher only)"), "aliked-lightglue-q8_0.gguf",
             base + "aliked-lightglue-q8_0.gguf", 2},
            {tr("ALIKED F32 (matcher only)"), "aliked-lightglue-f32.gguf",
             base + "aliked-lightglue-f32.gguf", 2},
    };
}

QString LightGlueDialog::modelCacheDir() {
    char* dir = aicore_lightglue_model_cache_dir();
    if (!dir) {
        return QDir::homePath() +
               QStringLiteral("/cloudViewer_data/extract/lightglue_models");
    }
    QString result = QString::fromUtf8(dir);
    aicore_lightglue_free_string(dir);
    return result;
}

QString LightGlueDialog::formatFileSize(qint64 bytes) {
    if (bytes < 1024) return QString("%1 B").arg(bytes);
    if (bytes < 1024LL * 1024)
        return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
}

LightGlueDialog::LightGlueDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("LightGlue Feature Matching"));
    setMinimumWidth(760);
    m_netManager = new QNetworkAccessManager(this);
    setupUi();
    populateModelCombo();
}

void LightGlueDialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);

    auto* modeGroup = new QGroupBox(tr("Operation Mode"));
    auto* modeLayout = new QHBoxLayout(modeGroup);
    m_modeCombo = new QComboBox;
    m_modeCombo->addItem(tr("Match two images"), static_cast<int>(Mode::Match));
    m_modeCombo->addItem(tr("Model Info"), static_cast<int>(Mode::ModelInfo));
    modeLayout->addWidget(new QLabel(tr("Mode:")));
    modeLayout->addWidget(m_modeCombo, 1);
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LightGlueDialog::onModeChanged);
    mainLayout->addWidget(modeGroup);

    auto* modelGroup = new QGroupBox(tr("Model"));
    auto* modelLayout = new QGridLayout(modelGroup);
    m_modelCombo = new QComboBox;
    modelLayout->addWidget(new QLabel(tr("GGUF Model:")), 0, 0);
    modelLayout->addWidget(m_modelCombo, 0, 1, 1, 2);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LightGlueDialog::onModelComboChanged);

    m_customModelRow = new QWidget;
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    customLayout->setContentsMargins(0, 0, 0, 0);
    m_customModelPath = new QLineEdit;
    m_customModelPath->setPlaceholderText(tr("Path to custom .gguf file"));
    customLayout->addWidget(m_customModelPath, 1);
    m_browseCustomModelBtn = new QPushButton(tr("Browse..."));
    connect(m_browseCustomModelBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onBrowseCustomModel);
    customLayout->addWidget(m_browseCustomModelBtn);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 1, 0, 1, 3);

    modelLayout->addWidget(new QLabel(tr("Device:")), 2, 0);
    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        if (const aicore_device_info* d = aicore_device_at(i)) {
            m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
            if (d->is_default) m_deviceCombo->setCurrentIndex(i);
        }
    }
    modelLayout->addWidget(m_deviceCombo, 2, 1);

    modelLayout->addWidget(new QLabel(tr("Threads:")), 3, 0);
    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText(tr("Auto"));
    modelLayout->addWidget(m_threads, 3, 1);

    modelLayout->addWidget(new QLabel(tr("Min score:")), 4, 0);
    m_minScore = new QDoubleSpinBox;
    m_minScore->setRange(0.0, 1.0);
    m_minScore->setSingleStep(0.05);
    m_minScore->setValue(0.1);
    modelLayout->addWidget(m_minScore, 4, 1);

    mainLayout->addWidget(modelGroup);

    auto* ioGroup = new QGroupBox(tr("Input Images"));
    m_ioGroup = ioGroup;
    auto* ioLayout = new QVBoxLayout(ioGroup);

    auto* hintLabel = new QLabel(
            tr("Pick two images above to match. "
               "Load Folder fills Image 1/2 with the first two files and lists "
               "all folder images below — select one, then → Image 1/2 or "
               "double-click."));
    hintLabel->setWordWrap(true);
    hintLabel->setStyleSheet("color: #555;");
    ioLayout->addWidget(hintLabel);

    auto* slotRow = new QHBoxLayout;
    for (int slot = 0; slot < kSlotCount; ++slot) {
        auto* slotGroup = new QGroupBox(tr("Image %1").arg(slot + 1));
        m_slotGroups[slot] = slotGroup;
        auto* slotLayout = new QVBoxLayout(slotGroup);

        m_slotPreview[slot] = new QLabel;
        m_slotPreview[slot]->setFixedSize(kThumbSize + 16, kThumbSize + 16);
        m_slotPreview[slot]->setAlignment(Qt::AlignCenter);
        m_slotPreview[slot]->setFrameShape(QFrame::StyledPanel);
        m_slotPreview[slot]->setStyleSheet("background: #f4f4f4;");
        slotLayout->addWidget(m_slotPreview[slot], 0, Qt::AlignHCenter);

        m_slotNameLabel[slot] = new QLabel(tr("(not set)"));
        m_slotNameLabel[slot]->setAlignment(Qt::AlignCenter);
        m_slotNameLabel[slot]->setWordWrap(true);
        m_slotNameLabel[slot]->setMinimumHeight(36);
        slotLayout->addWidget(m_slotNameLabel[slot]);

        auto* slotBtnRow = new QHBoxLayout;
        auto* browseSlotBtn = new QPushButton(tr("Browse..."));
        browseSlotBtn->setProperty("slotIndex", slot);
        connect(browseSlotBtn, &QPushButton::clicked, this,
                &LightGlueDialog::onBrowseSlotImage);
        slotBtnRow->addWidget(browseSlotBtn);
        auto* clearSlotBtn = new QPushButton(tr("Clear"));
        clearSlotBtn->setProperty("slotIndex", slot);
        connect(clearSlotBtn, &QPushButton::clicked, this,
                &LightGlueDialog::onClearSlot);
        slotBtnRow->addWidget(clearSlotBtn);
        slotLayout->addLayout(slotBtnRow);

        slotRow->addWidget(slotGroup, 1);
    }
    ioLayout->addLayout(slotRow);

    auto* inputBtnLayout = new QHBoxLayout;
    auto* browseFileBtn = new QPushButton(tr("Add Files..."));
    browseFileBtn->setToolTip(
            tr("Add one or more image files (first two fill the slots)"));
    connect(browseFileBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onBrowseFile);
    inputBtnLayout->addWidget(browseFileBtn);
    auto* browseFolderBtn = new QPushButton(tr("Load Folder..."));
    browseFolderBtn->setToolTip(
            tr("Load all images from a folder into the list below; "
               "the first two fill Image 1 and Image 2"));
    connect(browseFolderBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onBrowseFolder);
    inputBtnLayout->addWidget(browseFolderBtn);
    auto* clearBtn = new QPushButton(tr("Clear All"));
    connect(clearBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onClearImages);
    inputBtnLayout->addWidget(clearBtn);
    inputBtnLayout->addStretch();
    ioLayout->addLayout(inputBtnLayout);

    m_filePoolGroup = new QWidget;
    auto* filePoolLayout = new QVBoxLayout(m_filePoolGroup);
    filePoolLayout->setContentsMargins(0, 0, 0, 0);
    auto* filePoolHeader = new QHBoxLayout;
    filePoolHeader->addWidget(new QLabel(tr("Loaded files:")));
    filePoolHeader->addStretch();
    auto* assign1Btn = new QPushButton(tr("→ Image 1"));
    assign1Btn->setToolTip(tr("Assign the selected loaded file to Image 1"));
    connect(assign1Btn, &QPushButton::clicked, this,
            &LightGlueDialog::onAssignToSlot1);
    filePoolHeader->addWidget(assign1Btn);
    auto* assign2Btn = new QPushButton(tr("→ Image 2"));
    assign2Btn->setToolTip(tr("Assign the selected loaded file to Image 2"));
    connect(assign2Btn, &QPushButton::clicked, this,
            &LightGlueDialog::onAssignToSlot2);
    filePoolHeader->addWidget(assign2Btn);
    filePoolLayout->addLayout(filePoolHeader);

    m_filePoolList = new QListWidget;
    m_filePoolList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_filePoolList->setMinimumHeight(100);
    m_filePoolList->setMaximumHeight(140);
    m_filePoolList->setToolTip(tr(
            "Images from Add Files / Load Folder. "
            "Select an entry, click → Image 1/2, or double-click to assign."));
    connect(m_filePoolList, &QListWidget::itemActivated, this,
            &LightGlueDialog::onFilePoolActivated);
    filePoolLayout->addWidget(m_filePoolList);
    m_filePoolGroup->setVisible(false);
    ioLayout->addWidget(m_filePoolGroup);

    auto* dbHeader = new QHBoxLayout;
    dbHeader->addWidget(new QLabel(tr("DB source images (optional):")));
    dbHeader->addStretch();
    ioLayout->addLayout(dbHeader);

    m_dbImageList = new QListWidget;
    m_dbImageList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_dbImageList->setMinimumHeight(100);
    m_dbImageList->setMaximumHeight(140);
    m_dbImageList->setToolTip(
            tr("ccImage entities from the DB tree (match outputs are hidden). "
               "Double-click to assign to Image 1 or Image 2."));
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &LightGlueDialog::onDbListActivated);
    ioLayout->addWidget(m_dbImageList);

    auto* refreshBtn = new QPushButton(tr("Refresh DB Images"));
    connect(refreshBtn, &QPushButton::clicked, this,
            &LightGlueDialog::refreshDbImagesRequested);
    ioLayout->addWidget(refreshBtn);

    m_imageStatusLabel = new QLabel;
    m_imageStatusLabel->setStyleSheet("font-weight: bold;");
    ioLayout->addWidget(m_imageStatusLabel);

    m_addToDbCheck =
            new QCheckBox(tr("Add match visualization to DB tree after run"));
    m_addToDbCheck->setChecked(true);
    ioLayout->addWidget(m_addToDbCheck);

    mainLayout->addWidget(ioGroup);

    m_downloadLabel = new QLabel;
    m_downloadLabel->setVisible(false);
    mainLayout->addWidget(m_downloadLabel);

    m_taskStatusLabel = new QLabel;
    m_taskStatusLabel->setVisible(false);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066aa;");
    mainLayout->addWidget(m_taskStatusLabel);

    m_progress = new QProgressBar;
    m_progress->setRange(0, 100);
    mainLayout->addWidget(m_progress);

    m_log = new QTextEdit;
    m_log->setReadOnly(true);
    m_log->setMaximumHeight(160);
    mainLayout->addWidget(m_log);

    auto* actionRow = new QHBoxLayout;
    m_runBtn = new QPushButton(tr("Run"));
    connect(m_runBtn, &QPushButton::clicked, this, &LightGlueDialog::onRun);
    actionRow->addWidget(m_runBtn);
    m_cancelBtn = new QPushButton(tr("Cancel"));
    m_cancelBtn->setEnabled(false);
    connect(m_cancelBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onCancel);
    actionRow->addWidget(m_cancelBtn);
    actionRow->addStretch();
    m_exportBtn = new QPushButton(tr("Export JSON..."));
    m_exportBtn->setEnabled(false);
    connect(m_exportBtn, &QPushButton::clicked, this,
            &LightGlueDialog::onExportMatches);
    actionRow->addWidget(m_exportBtn);
    mainLayout->addLayout(actionRow);

    refreshSlotWidgets();
    updateImageStatus();
    updateRunButtonState();
    onModeChanged(m_modeCombo->currentIndex());
}

void LightGlueDialog::populateModelCombo(const QString& keepFilename) {
    const QString cacheDir = modelCacheDir();
    QString selected = keepFilename;
    if (selected.isEmpty() && m_modelCombo && m_modelCombo->count() > 0) {
        selected = m_modelCombo->currentData().toString();
    }

    m_modelCombo->blockSignals(true);
    m_modelCombo->clear();
    for (const auto& m : builtinModels()) {
        const QString cached = cacheDir + "/" + m.filename;
        const QFileInfo fi(cached);
        const QString suffix =
                fi.exists() ? QString(" [%1] ✓").arg(formatFileSize(fi.size()))
                            : QString(" [download]");
        m_modelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_modelCombo->insertSeparator(m_modelCombo->count());
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    selectModelByFilename(selected.isEmpty() ? "sift-lightglue-f16.gguf"
                                             : selected);
    m_modelCombo->blockSignals(false);
    onModelComboChanged(m_modelCombo->currentIndex());
}

bool LightGlueDialog::selectModelByFilename(const QString& filename) {
    if (!m_modelCombo || filename.isEmpty()) return false;
    for (int i = 0; i < m_modelCombo->count(); ++i) {
        if (m_modelCombo->itemData(i).toString() == filename) {
            m_modelCombo->setCurrentIndex(i);
            return true;
        }
    }
    return false;
}

void LightGlueDialog::onModelComboChanged(int index) {
    const QString data = m_modelCombo->itemData(index).toString();
    if (m_customModelRow) m_customModelRow->setVisible(data == "CUSTOM");
    updateRunButtonState();
}

LightGlueDialog::Settings LightGlueDialog::getSettings() const {
    Settings s;
    s.mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    s.modelPath = resolveModelPath();
    s.inputPaths = QStringList() << m_slotPaths[0] << m_slotPaths[1];
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.minScore = m_minScore->value();
    s.addResultToDb = m_addToDbCheck->isChecked();

    const QString data = m_modelCombo->currentData().toString();
    for (const auto& model : builtinModels()) {
        if (model.filename == data) {
            s.matcherType = model.matcherType;
            break;
        }
    }
    if (data == "CUSTOM") {
        const QString name = m_customModelPath->text().toLower();
        if (name.contains("sift"))
            s.matcherType = 1;
        else if (name.contains("aliked"))
            s.matcherType = 2;
    }
    return s;
}

QString LightGlueDialog::resolveModelPath() const {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return m_customModelPath->text().trimmed();
    if (data.isEmpty()) return QString();
    return modelCacheDir() + "/" + data;
}

bool LightGlueDialog::isModelReady() const {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") {
        const QString path = m_customModelPath->text().trimmed();
        return !path.isEmpty() && QFile::exists(path);
    }
    if (data.isEmpty()) return false;
    if (QFile::exists(modelCacheDir() + "/" + data)) return true;
    for (const auto& m : builtinModels()) {
        if (m.filename == data) return true;
    }
    return false;
}

bool LightGlueDialog::isInputValid() const {
    return !m_slotPaths[0].isEmpty() && !m_slotPaths[1].isEmpty();
}

void LightGlueDialog::updateImageStatus() {
    if (!m_imageStatusLabel) return;
    const bool slot0 = !m_slotPaths[0].isEmpty();
    const bool slot1 = !m_slotPaths[1].isEmpty();
    if (slot0 && slot1) {
        m_imageStatusLabel->setText(
                tr("Ready: %1 × %2")
                        .arg(displayNameForPath(m_slotPaths[0]),
                             displayNameForPath(m_slotPaths[1])));
        m_imageStatusLabel->setStyleSheet("font-weight: bold; color: #006600;");
    } else if (slot0) {
        m_imageStatusLabel->setText(
                tr("Image 1 set — choose Image 2 to run matching."));
        m_imageStatusLabel->setStyleSheet("font-weight: bold; color: #aa6600;");
    } else if (slot1) {
        m_imageStatusLabel->setText(
                tr("Image 2 set — choose Image 1 to run matching."));
        m_imageStatusLabel->setStyleSheet("font-weight: bold; color: #aa6600;");
    } else {
        m_imageStatusLabel->setText(
                tr("Select two images (files, folder, or DB list)."));
        m_imageStatusLabel->setStyleSheet("font-weight: bold; color: #666666;");
    }
}

void LightGlueDialog::updateRunButtonState() {
    if (!m_runBtn || !m_modeCombo) return;
    if (m_downloadInProgress) {
        m_runBtn->setEnabled(false);
        if (m_cancelBtn) m_cancelBtn->setEnabled(true);
        return;
    }
    if (m_taskRunning) {
        m_runBtn->setEnabled(false);
        if (m_cancelBtn) m_cancelBtn->setEnabled(true);
        return;
    }
    if (m_cancelBtn) m_cancelBtn->setEnabled(false);
    const auto mode = static_cast<Mode>(m_modeCombo->currentData().toInt());
    bool ready = isModelReady();
    if (mode == Mode::Match) {
        ready = ready && isInputValid();
    }
    m_runBtn->setEnabled(ready);
}

void LightGlueDialog::onModeChanged(int index) {
    const auto mode = static_cast<Mode>(m_modeCombo->itemData(index).toInt());
    const bool isMatch = (mode == Mode::Match);
    if (m_ioGroup) m_ioGroup->setVisible(isMatch);
    if (m_addToDbCheck) m_addToDbCheck->setVisible(isMatch);
    if (m_minScore) m_minScore->setEnabled(isMatch);
    if (isMatch) {
        updateImageStatus();
    } else if (m_imageStatusLabel) {
        m_imageStatusLabel->clear();
    }
    updateRunButtonState();
}

QImage LightGlueDialog::previewForPath(const QString& path) const {
    if (path.startsWith("db://")) {
        return m_dbPreviews.value(path.mid(5));
    }
    return lightglue_plugin::load_oriented_qimage(path);
}

QString LightGlueDialog::displayNameForPath(const QString& path) const {
    if (path.startsWith("db://")) {
        return path.mid(5);
    }
    return QFileInfo(path).fileName();
}

void LightGlueDialog::refreshSlotWidgets() {
    for (int slot = 0; slot < kSlotCount; ++slot) {
        if (!m_slotPreview[slot] || !m_slotNameLabel[slot]) continue;
        const QString& path = m_slotPaths[slot];
        if (path.isEmpty()) {
            m_slotPreview[slot]->clear();
            m_slotPreview[slot]->setText(tr("Drop\nor\nBrowse"));
            m_slotNameLabel[slot]->setText(tr("(not set)"));
            if (m_slotGroups[slot]) {
                m_slotGroups[slot]->setStyleSheet(QString());
            }
            continue;
        }

        const QImage img = previewForPath(path);
        if (!img.isNull()) {
            m_slotPreview[slot]->setText(QString());
            m_slotPreview[slot]->setPixmap(QPixmap::fromImage(
                    img.scaled(kThumbSize, kThumbSize, Qt::KeepAspectRatio,
                               Qt::SmoothTransformation)));
        } else {
            m_slotPreview[slot]->setPixmap(QPixmap());
            m_slotPreview[slot]->setText("?");
        }
        m_slotPreview[slot]->setToolTip(path);
        m_slotNameLabel[slot]->setText(displayNameForPath(path));
        if (m_slotGroups[slot]) {
            m_slotGroups[slot]->setStyleSheet(
                    "QGroupBox { border: 2px solid #0066aa; }");
        }
    }
    syncDbListHighlight();
    syncFilePoolHighlight();
}

void LightGlueDialog::setFilePoolPaths(const QStringList& paths) {
    m_filePoolPaths.clear();
    for (const QString& path : paths) {
        if (path.isEmpty() || m_filePoolPaths.contains(path)) continue;
        m_filePoolPaths.append(path);
    }
    refreshFilePoolList();
}

void LightGlueDialog::refreshFilePoolList() {
    if (!m_filePoolList || !m_filePoolGroup) return;
    m_filePoolList->clear();
    if (m_filePoolPaths.isEmpty()) {
        m_filePoolGroup->setVisible(false);
        return;
    }

    m_filePoolGroup->setVisible(true);
    for (const QString& path : m_filePoolPaths) {
        auto* item = new QListWidgetItem(QFileInfo(path).fileName());
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
        const QImage img = previewForPath(path);
        if (!img.isNull()) {
            item->setIcon(QIcon(QPixmap::fromImage(img.scaled(
                    48, 48, Qt::KeepAspectRatio, Qt::SmoothTransformation))));
        }
        m_filePoolList->addItem(item);
    }
    syncFilePoolHighlight();
}

void LightGlueDialog::syncFilePoolHighlight() {
    if (!m_filePoolList) return;
    for (int i = 0; i < m_filePoolList->count(); ++i) {
        QListWidgetItem* item = m_filePoolList->item(i);
        if (!item) continue;
        const QString path = item->data(Qt::UserRole).toString();
        const bool inSlot0 = (m_slotPaths[0] == path);
        const bool inSlot1 = (m_slotPaths[1] == path);
        QFont font = item->font();
        font.setBold(inSlot0 || inSlot1);
        item->setFont(font);
        QString label = QFileInfo(path).fileName();
        if (inSlot0 && inSlot1) {
            label += tr("  [Image 1 & 2]");
        } else if (inSlot0) {
            label += tr("  [Image 1]");
        } else if (inSlot1) {
            label += tr("  [Image 2]");
        }
        item->setText(label);
    }
}

bool LightGlueDialog::assignSelectedToSlot(int slot) {
    if (slot < 0 || slot >= kSlotCount) return false;

    if (m_filePoolList && m_filePoolList->currentItem()) {
        const QString path =
                m_filePoolList->currentItem()->data(Qt::UserRole).toString();
        if (!path.isEmpty()) {
            assignToSlot(slot, path);
            return true;
        }
    }

    if (m_dbImageList && m_dbImageList->currentItem() &&
        (m_dbImageList->currentItem()->flags() & Qt::ItemIsSelectable)) {
        const QString name =
                m_dbImageList->currentItem()->data(Qt::UserRole).toString();
        if (!name.isEmpty()) {
            assignToSlot(slot, QStringLiteral("db://") + name);
            return true;
        }
    }

    appendLog(tr("[Hint] Select an image in Loaded files or DB source images, "
                 "then click → Image %1.")
                      .arg(slot + 1));
    return false;
}

void LightGlueDialog::assignToSlot(int slot, const QString& path) {
    if (slot < 0 || slot >= kSlotCount || path.isEmpty()) return;
    m_slotPaths[slot] = path;
    refreshSlotWidgets();
    syncFilePoolHighlight();
    updateImageStatus();
    updateRunButtonState();
}

void LightGlueDialog::clearSlot(int slot) {
    if (slot < 0 || slot >= kSlotCount) return;
    m_slotPaths[slot].clear();
    refreshSlotWidgets();
    updateImageStatus();
    updateRunButtonState();
}

void LightGlueDialog::clearAllSlots() {
    for (int slot = 0; slot < kSlotCount; ++slot) {
        m_slotPaths[slot].clear();
    }
    m_filePoolPaths.clear();
    refreshFilePoolList();
    refreshSlotWidgets();
    updateImageStatus();
    updateRunButtonState();
}

void LightGlueDialog::autoAssignPaths(const QStringList& paths) {
    QStringList unique;
    for (const QString& path : paths) {
        if (path.isEmpty() || unique.contains(path)) continue;
        unique.append(path);
    }
    if (unique.isEmpty()) return;

    if (unique.size() == 1) {
        if (m_slotPaths[0].isEmpty()) {
            assignToSlot(0, unique[0]);
        } else if (m_slotPaths[1].isEmpty()) {
            assignToSlot(1, unique[0]);
        } else {
            assignToSlot(1, unique[0]);
        }
        return;
    }

    assignToSlot(0, unique[0]);
    assignToSlot(1, unique[1]);
    if (unique.size() > 2) {
        appendLog(tr("[Info] Using first two of %1 images for matching; "
                     "select others in Loaded files and click → Image 1/2.")
                          .arg(unique.size()));
    }
}

int LightGlueDialog::senderSlotIndex() const {
    const auto* btn = qobject_cast<const QPushButton*>(sender());
    if (!btn) return -1;
    bool ok = false;
    const int slot = btn->property("slotIndex").toInt(&ok);
    return ok ? slot : -1;
}

void LightGlueDialog::syncDbListHighlight() {
    if (!m_dbImageList) return;
    for (int i = 0; i < m_dbImageList->count(); ++i) {
        QListWidgetItem* item = m_dbImageList->item(i);
        if (!item || !(item->flags() & Qt::ItemIsSelectable)) continue;
        const QString dbPath =
                QStringLiteral("db://") + item->data(Qt::UserRole).toString();
        const bool inSlot0 = (m_slotPaths[0] == dbPath);
        const bool inSlot1 = (m_slotPaths[1] == dbPath);
        QFont font = item->font();
        font.setBold(inSlot0 || inSlot1);
        item->setFont(font);
        if (inSlot0 && inSlot1) {
            item->setText(item->data(Qt::UserRole).toString() +
                          tr("  [Image 1 & 2]"));
        } else if (inSlot0) {
            item->setText(item->data(Qt::UserRole).toString() +
                          tr("  [Image 1]"));
        } else if (inSlot1) {
            item->setText(item->data(Qt::UserRole).toString() +
                          tr("  [Image 2]"));
        } else {
            item->setText(item->data(Qt::UserRole).toString());
        }
    }
}

void LightGlueDialog::onBrowseSlotImage() {
    const int slot = senderSlotIndex();
    if (slot < 0) return;

    QSettings settings;
    const QString lastDir =
            settings.value("qLightGlue/lastImageFileDir", QDir::homePath())
                    .toString();
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select Image %1").arg(slot + 1), lastDir,
            imageFileDialogFilter());
    if (path.isEmpty()) return;
    if (!isSupportedImageFile(path)) {
        appendLog(tr("[Warning] Unsupported image file: %1").arg(path));
        return;
    }
    settings.setValue("qLightGlue/lastImageFileDir",
                      QFileInfo(path).absolutePath());
    assignToSlot(slot, path);
}

void LightGlueDialog::onClearSlot() {
    const int slot = senderSlotIndex();
    if (slot < 0) return;
    clearSlot(slot);
}

void LightGlueDialog::onFilePoolActivated(QListWidgetItem* item) {
    if (!item) return;
    const QString path = item->data(Qt::UserRole).toString();
    if (path.isEmpty()) return;
    if (m_slotPaths[0].isEmpty()) {
        assignToSlot(0, path);
    } else if (m_slotPaths[1].isEmpty()) {
        assignToSlot(1, path);
    } else {
        assignToSlot(1, path);
    }
}

void LightGlueDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item || !(item->flags() & Qt::ItemIsSelectable)) return;
    const QString name = item->data(Qt::UserRole).toString();
    if (name.isEmpty()) return;
    const QString dbPath = QStringLiteral("db://") + name;
    if (m_slotPaths[0].isEmpty()) {
        assignToSlot(0, dbPath);
    } else if (m_slotPaths[1].isEmpty()) {
        assignToSlot(1, dbPath);
    } else {
        assignToSlot(1, dbPath);
    }
}

void LightGlueDialog::onAssignToSlot1() { assignSelectedToSlot(0); }

void LightGlueDialog::onAssignToSlot2() { assignSelectedToSlot(1); }

void LightGlueDialog::setDbImages(const QList<DbImageEntry>& images) {
    if (!m_dbImageList) return;
    m_dbPreviews.clear();
    m_dbImageList->clear();
    if (images.isEmpty()) {
        auto* item =
                new QListWidgetItem(tr("(no source ccImage entities in DB)"));
        item->setFlags(Qt::NoItemFlags);
        m_dbImageList->addItem(item);
        m_dbImageList->setEnabled(false);
    } else {
        m_dbImageList->setEnabled(true);
        for (const auto& entry : images) {
            m_dbPreviews.insert(entry.name, entry.preview);
            auto* item = new QListWidgetItem(entry.name);
            item->setData(Qt::UserRole, entry.name);
            if (!entry.preview.isNull()) {
                item->setIcon(QIcon(QPixmap::fromImage(
                        entry.preview.scaled(48, 48, Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation))));
            }
            m_dbImageList->addItem(item);
        }
    }
    syncDbListHighlight();
}

void LightGlueDialog::applyDbTreeSelection(const QStringList& imageNames) {
    if (imageNames.isEmpty()) return;
    QStringList names = imageNames;
    while (names.size() > kSlotCount) {
        names.removeLast();
    }
    if (names.size() >= 1) {
        assignToSlot(0, QStringLiteral("db://") + names[0]);
    }
    if (names.size() >= 2) {
        assignToSlot(1, QStringLiteral("db://") + names[1]);
    }
    appendLog(tr("[Info] Assigned %1 image(s) from DB tree selection.")
                      .arg(names.size()));
}

void LightGlueDialog::onClearImages() { clearAllSlots(); }

void LightGlueDialog::onBrowseFile() {
    QSettings settings;
    const QString lastDir =
            settings.value("qLightGlue/lastImageFileDir", QDir::homePath())
                    .toString();
    const QStringList paths = QFileDialog::getOpenFileNames(
            this, tr("Select Image(s)"), lastDir, imageFileDialogFilter());
    if (paths.isEmpty()) return;

    settings.setValue("qLightGlue/lastImageFileDir",
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

    autoAssignPaths(accepted);
    QStringList pool = m_filePoolPaths;
    for (const QString& path : accepted) {
        if (!pool.contains(path)) pool.append(path);
    }
    setFilePoolPaths(pool);
    appendLog(tr("[Info] Added %1 file(s).").arg(accepted.size()));
}

void LightGlueDialog::onBrowseFolder() {
    QSettings settings;
    const QString lastDir =
            settings.value("qLightGlue/lastImageFolder", QDir::homePath())
                    .toString();
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Select Image Folder"), lastDir);
    if (dir.isEmpty()) return;

    settings.setValue("qLightGlue/lastImageFolder", dir);

    const QStringList files = listImageFilesInDir(dir);
    if (files.isEmpty()) {
        appendLog(tr("[Warning] No image files found in: %1").arg(dir));
        return;
    }

    clearAllSlots();
    setFilePoolPaths(files);
    autoAssignPaths(files);
    appendLog(tr("[Info] Loaded %1 image(s) from folder: %2")
                      .arg(files.size())
                      .arg(dir));
}

void LightGlueDialog::onBrowseCustomModel() {
    QSettings settings;
    const QString lastDir =
            settings.value("qLightGlue/lastModelDir", modelCacheDir())
                    .toString();
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select GGUF model"), lastDir,
            tr("GGUF models (*.gguf);;All files (*)"));
    if (path.isEmpty()) return;
    settings.setValue("qLightGlue/lastModelDir",
                      QFileInfo(path).absolutePath());
    m_customModelPath->setText(path);
}

bool LightGlueDialog::ensureModelAvailable() {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return isModelReady();

    const QString cached = modelCacheDir() + "/" + data;
    if (QFile::exists(cached)) return true;

    for (const auto& bm : builtinModels()) {
        if (bm.filename != data) continue;
        const auto answer = QMessageBox::question(
                this, tr("Download Model"),
                tr("The model '%1' is not cached locally.\n\nDownload it now?")
                        .arg(bm.displayName));
        if (answer != QMessageBox::Yes) {
            appendLog(tr("[Info] Download declined."));
            return false;
        }
        m_autoRunAfterDownload = true;
        startDownload(bm);
        return false;
    }
    return false;
}

void LightGlueDialog::startDownload(const LightGlueBuiltinModel& model) {
    if (m_downloadInProgress) {
        appendLog(tr("[Warning] A download is already in progress."));
        return;
    }

    QDir().mkpath(modelCacheDir());
    const QString dest = modelCacheDir() + "/" + model.filename;
    const QString tmpDest = dest + ".part";

    m_downloadInProgress = true;
    m_downloadTargetFilename = model.filename;
    m_downloadTmpPath = tmpDest;
    m_downloadLabel->setText(tr("Downloading %1 ...").arg(model.filename));
    m_downloadLabel->setVisible(true);
    m_progress->setValue(0);
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
                    m_progress->setValue(
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
                const bool ok =
                        m_currentDownload &&
                        m_currentDownload->error() == QNetworkReply::NoError;
                if (ok) {
                    QFile::remove(dest);
                    QFile::rename(tmpDest, dest);
                    appendLog(tr("[OK] Model downloaded: %1").arg(dest));
                } else if (m_currentDownload) {
                    appendLog(tr("[Error] Download failed: %1")
                                      .arg(m_currentDownload->errorString()));
                    QFile::remove(tmpDest);
                }
                cancelDownload();
                populateModelCombo(model.filename);
                selectModelByFilename(model.filename);
                updateRunButtonState();
                if (ok && m_autoRunAfterDownload) {
                    m_autoRunAfterDownload = false;
                    onRun();
                } else {
                    m_autoRunAfterDownload = false;
                }
            });
}

void LightGlueDialog::cancelDownload() {
    if (m_currentDownload) {
        m_currentDownload->abort();
        m_currentDownload->deleteLater();
        m_currentDownload = nullptr;
    }
    if (m_downloadOutFile) {
        m_downloadOutFile->close();
        m_downloadOutFile->deleteLater();
        m_downloadOutFile = nullptr;
    }
    if (!m_downloadTmpPath.isEmpty()) QFile::remove(m_downloadTmpPath);
    m_downloadInProgress = false;
    m_downloadLabel->setVisible(false);
    updateRunButtonState();
}

void LightGlueDialog::appendLog(const QString& msg) { m_log->append(msg); }

void LightGlueDialog::setProgress(int current, int total) {
    m_progress->setMaximum(total);
    m_progress->setValue(current);
}

void LightGlueDialog::setRunning(bool running) {
    m_taskRunning = running;
    if (running) {
        if (m_taskStatusLabel) {
            m_taskStatusLabel->setText(tr("Starting..."));
            m_taskStatusLabel->setVisible(true);
        }
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
    } else if (m_taskStatusLabel) {
        m_taskStatusLabel->clear();
        m_taskStatusLabel->setVisible(false);
    }
    updateRunButtonState();
    if (m_modeCombo) {
        m_modeCombo->setEnabled(!running && !m_downloadInProgress);
    }
}

void LightGlueDialog::setTaskStage(const QString& stage, int percent) {
    if (!m_taskStatusLabel) return;
    m_taskStatusLabel->setText(stage);
    m_taskStatusLabel->setVisible(true);
    if (percent >= 0) {
        m_progress->setRange(0, 100);
        m_progress->setValue(percent);
    } else {
        m_progress->setRange(0, 0);
    }
}

void LightGlueDialog::enableExportButton(bool enabled) {
    m_exportBtn->setEnabled(enabled);
}

void LightGlueDialog::onRun() {
    if (!ensureModelAvailable()) {
        if (m_downloadInProgress) {
            appendLog(tr("[LG] Waiting for model download..."));
        }
        return;
    }
    emit runRequested(getSettings());
}

void LightGlueDialog::onCancel() {
    if (m_downloadInProgress) {
        cancelDownload();
        appendLog(tr("[LG] Download canceled."));
    }
    emit cancelRequested();
}

void LightGlueDialog::onExportMatches() { emit exportMatchesRequested(); }
