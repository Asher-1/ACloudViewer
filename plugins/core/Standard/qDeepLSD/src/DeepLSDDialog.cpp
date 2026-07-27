// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DeepLSDDialog.h"

#include <QDir>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QVBoxLayout>

#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"

static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "DeepLSD/";

namespace {

const int kThumbSize = 96;

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

bool isQ8QuantModel(const QString& filename) {
    return filename.contains(QStringLiteral("q8_0"), Qt::CaseInsensitive);
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
        aicore_deeplsd_free_string(dir);
        return result;
    }
    return QDir::homePath() +
           QStringLiteral("/cloudViewer_data/extract/deeplsd_models");
}

DeepLSDDialog::DeepLSDDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("DeepLSD Line Extraction"));
    setMinimumWidth(720);
    m_netManager = new QNetworkAccessManager(this);
    setupUi();
    populateModelCombo();
}

void DeepLSDDialog::setupUi() {
    auto* main = new QVBoxLayout(this);

    auto* modelGroup = new QGroupBox(tr("Model"));
    auto* modelLayout = new QGridLayout(modelGroup);
    m_modelCombo = new QComboBox;
    modelLayout->addWidget(new QLabel(tr("GGUF:")), 0, 0);
    modelLayout->addWidget(m_modelCombo, 0, 1);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DeepLSDDialog::onModelComboChanged);

    m_quantWarningLabel = new QLabel;
    m_quantWarningLabel->setWordWrap(true);
    m_quantWarningLabel->setStyleSheet(
            "color: palette(button-text); background: #fff3cd; border: 1px "
            "solid "
            "#ffc107; padding: 6px; border-radius: 4px;");
    m_quantWarningLabel->setVisible(false);
    modelLayout->addWidget(m_quantWarningLabel, 1, 0, 1, 2);

    m_variantHintLabel = new QLabel;
    m_variantHintLabel->setWordWrap(true);
    m_variantHintLabel->setStyleSheet(
            "color: #333; background: #eef4fb; border: 1px solid #b8d4f0; "
            "padding: 6px; border-radius: 4px; font-size: 11px;");
    modelLayout->addWidget(m_variantHintLabel, 2, 0, 1, 2);

    m_customModelRow = new QWidget;
    auto* customLayout = new QHBoxLayout(m_customModelRow);
    m_customModelPath = new QLineEdit;
    auto* browseModel = new QPushButton(tr("Browse..."));
    connect(browseModel, &QPushButton::clicked, this,
            &DeepLSDDialog::onBrowseCustomModel);
    customLayout->addWidget(m_customModelPath, 1);
    customLayout->addWidget(browseModel);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 3, 0, 1, 2);

    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        if (const aicore_device_info* d = aicore_device_at(i)) {
            m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
            if (d->is_default) m_deviceCombo->setCurrentIndex(i);
        }
    }
    modelLayout->addWidget(new QLabel(tr("Device:")), 4, 0);
    modelLayout->addWidget(m_deviceCombo, 4, 1);

    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText(tr("Auto"));
    modelLayout->addWidget(new QLabel(tr("Threads:")), 5, 0);
    modelLayout->addWidget(m_threads, 5, 1);
    main->addWidget(modelGroup);

    auto* ioGroup = new QGroupBox(tr("Input"));
    auto* ioLayout = new QVBoxLayout(ioGroup);
    auto* pathRow = new QHBoxLayout;
    m_imagePath = new QLineEdit;
    m_imagePath->setPlaceholderText(
            tr("Image file path or db://EntityName from DB tree"));
    connect(m_imagePath, &QLineEdit::textChanged, this,
            [this](const QString&) { updateImagePreview(); });
    auto* browseImg = new QPushButton(tr("Browse..."));
    connect(browseImg, &QPushButton::clicked, this,
            &DeepLSDDialog::onBrowseImage);
    pathRow->addWidget(m_imagePath, 1);
    pathRow->addWidget(browseImg);
    ioLayout->addLayout(pathRow);

    m_previewLabel = new QLabel;
    m_previewLabel->setFixedSize(kThumbSize, kThumbSize);
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setStyleSheet(
            "border: 1px solid palette(mid); background: palette(base);");
    m_previewLabel->setText(tr("Preview"));
    ioLayout->addWidget(m_previewLabel, 0, Qt::AlignLeft);

    auto* dbHeader = new QHBoxLayout;
    m_dbToggleBtn = new QToolButton;
    m_dbToggleBtn->setArrowType(Qt::RightArrow);
    m_dbToggleBtn->setCheckable(true);
    m_dbToggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
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
    m_dbImageList = new QListWidget;
    m_dbImageList->setMinimumHeight(80);
    m_dbImageList->setMaximumHeight(140);
    m_dbImageList->setToolTip(
            tr("Double-click a ccImage from the DB tree to use as input."));
    connect(m_dbImageList, &QListWidget::itemActivated, this,
            &DeepLSDDialog::onDbListActivated);
    dbLayout->addWidget(m_dbImageList);
    auto* refreshBtn = new QPushButton(tr("Refresh DB Images"));
    connect(refreshBtn, &QPushButton::clicked, this,
            &DeepLSDDialog::refreshDbImagesRequested);
    dbLayout->addWidget(refreshBtn);
    m_dbContentWidget->setVisible(false);
    ioLayout->addWidget(m_dbContentWidget);

    m_addToDbCheck = new QCheckBox(
            tr("Add distance-field overlay to DB tree after run"));
    m_addToDbCheck->setChecked(true);
    ioLayout->addWidget(m_addToDbCheck);
    main->addWidget(ioGroup);

    m_downloadLabel = new QLabel;
    m_downloadLabel->setVisible(false);
    main->addWidget(m_downloadLabel);

    m_progress = new QProgressBar;
    main->addWidget(m_progress);

    m_log = new QTextEdit;
    m_log->setReadOnly(true);
    m_log->setMinimumHeight(160);
    main->addWidget(m_log);

    auto* btnRow = new QHBoxLayout;
    m_runBtn = new QPushButton(tr("Run"));
    m_cancelBtn = new QPushButton(tr("Cancel"));
    m_cancelBtn->setEnabled(false);
    connect(m_runBtn, &QPushButton::clicked, this, &DeepLSDDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &DeepLSDDialog::onCancel);
    btnRow->addStretch();
    btnRow->addWidget(m_runBtn);
    btnRow->addWidget(m_cancelBtn);
    main->addLayout(btnRow);
}

void DeepLSDDialog::populateModelCombo() {
    m_modelCombo->clear();
    const QString cache = modelCacheDir();
    for (const auto& m : builtinModels()) {
        const QFileInfo fi(cache + "/" + m.filename);
        const QString suffix =
                fi.exists() ? QString(" [%1] ✓").arg(formatFileSize(fi.size()))
                            : QString(" [download]");
        m_modelCombo->addItem(m.displayName + suffix, m.filename);
    }
    m_modelCombo->addItem(tr("Custom..."), "CUSTOM");
    onModelComboChanged(m_modelCombo->currentIndex());
}

void DeepLSDDialog::refreshModelList() { populateModelCombo(); }

QString DeepLSDDialog::formatFileSize(qint64 bytes) {
    if (bytes < 1024) return QString("%1 B").arg(bytes);
    if (bytes < 1024LL * 1024)
        return QString("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    return QString("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
}

DeepLSDDialog::Settings DeepLSDDialog::getSettings() const {
    Settings s;
    s.modelPath = resolveModelPath();
    s.inputPath = m_imagePath->text().trimmed();
    s.threads = m_threads->value();
    s.device = m_deviceCombo->currentData().toString();
    s.addResultToDb = m_addToDbCheck->isChecked();
    return s;
}

QString DeepLSDDialog::resolveModelPath() const {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return m_customModelPath->text().trimmed();
    return modelCacheDir() + "/" + data;
}

void DeepLSDDialog::appendLog(const QString& msg) { m_log->append(msg); }

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
                const QIcon icon = m_dbImageList->item(i)->icon();
                if (!icon.isNull()) {
                    m_previewLabel->setPixmap(
                            icon.pixmap(kThumbSize, kThumbSize));
                    return;
                }
            }
        }
        m_previewLabel->setText(tr("DB"));
        return;
    }
    if (path.isEmpty() || !isSupportedImageFile(path)) {
        m_previewLabel->clear();
        m_previewLabel->setText(tr("Preview"));
        return;
    }
    QImage img(path);
    if (img.isNull()) {
        m_previewLabel->setText(tr("?"));
        return;
    }
    m_previewLabel->setPixmap(QPixmap::fromImage(img).scaled(
            kThumbSize, kThumbSize, Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
}

void DeepLSDDialog::onBrowseImage() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select image"), QDir::homePath(),
            tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (!path.isEmpty()) m_imagePath->setText(path);
}

void DeepLSDDialog::onBrowseCustomModel() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select GGUF"), modelCacheDir(), tr("GGUF (*.gguf)"));
    if (!path.isEmpty()) {
        m_customModelPath->setText(path);
        onModelComboChanged(m_modelCombo->currentIndex());
    }
}

void DeepLSDDialog::onModelComboChanged(int index) {
    const QString data = m_modelCombo->itemData(index).toString();
    m_customModelRow->setVisible(data == "CUSTOM");

    QString variantHint;
    if (data.contains(QStringLiteral("wireframe"), Qt::CaseInsensitive)) {
        variantHint =
                tr("Wireframe model — trained on indoor/wireframe scenes "
                   "(synthetic "
                   "wireframe + ScanNet). Best for structured indoor geometry, "
                   "CAD-like "
                   "edges, and man-made environments.");
    } else if (data.contains(QStringLiteral("deeplsd_md"),
                             Qt::CaseInsensitive) ||
               data.contains(QStringLiteral("megadepth"),
                             Qt::CaseInsensitive)) {
        variantHint = tr(
                "MegaDepth (md) model — trained on outdoor phototourism "
                "(MegaDepth). "
                "Best for natural scenes, facades, and general outdoor/street "
                "photography.");
    } else if (data == "CUSTOM") {
        const QString path = m_customModelPath->text();
        if (path.contains(QStringLiteral("wireframe"), Qt::CaseInsensitive)) {
            variantHint =
                    tr("Custom wireframe checkpoint — prefer indoor/man-made "
                       "scenes.");
        } else if (path.contains(QStringLiteral("_md"), Qt::CaseInsensitive)) {
            variantHint =
                    tr("Custom MegaDepth checkpoint — prefer outdoor/natural "
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

    const bool q8 = data == "CUSTOM" ? isQ8QuantModel(m_customModelPath->text())
                                     : isQ8QuantModel(data);
    if (q8) {
        m_quantWarningLabel->setText(
                tr("Q8_0 quantization is experimental: distance/angle fields "
                   "may deviate noticeably from F32/F16 (df p99 ~0.09, angle "
                   "p99 ~0.24 vs PyTorch on reference scenes). Prefer F16 for "
                   "production use."));
        m_quantWarningLabel->setVisible(true);
    } else {
        m_quantWarningLabel->clear();
        m_quantWarningLabel->setVisible(false);
    }
}

void DeepLSDDialog::onDbListActivated(QListWidgetItem* item) {
    if (!item) return;
    m_imagePath->setText(QStringLiteral("db://") + item->text());
}

bool DeepLSDDialog::ensureModelAvailable() {
    const QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") return !resolveModelPath().isEmpty();
    const QString cached = modelCacheDir() + "/" + data;
    if (QFile::exists(cached)) return true;
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
    QDir().mkpath(modelCacheDir());
    const QString dest = modelCacheDir() + "/" + model.filename;
    const QString tmpDest = dest + ".part";
    m_downloadInProgress = true;
    m_downloadTmpPath = tmpDest;
    m_downloadLabel->setText(tr("Downloading %1 ...").arg(model.filename));
    m_downloadLabel->setVisible(true);
    m_currentDownload =
            m_netManager->get(QNetworkRequest(QUrl(model.downloadUrl)));
    m_downloadOutFile = new QFile(tmpDest, m_currentDownload);
    m_downloadOutFile->open(QIODevice::WriteOnly);
    connect(m_currentDownload, &QNetworkReply::readyRead, this, [this]() {
        if (m_downloadOutFile && m_currentDownload) {
            m_downloadOutFile->write(m_currentDownload->readAll());
        }
    });
    connect(m_currentDownload, &QNetworkReply::finished, this, [this, dest]() {
        if (m_downloadOutFile) {
            m_downloadOutFile->close();
            delete m_downloadOutFile;
            m_downloadOutFile = nullptr;
        }
        if (m_currentDownload->error() == QNetworkReply::NoError) {
            QFile::remove(dest);
            QFile::rename(m_downloadTmpPath, dest);
            appendLog(tr("[OK] Downloaded model."));
            populateModelCombo();
            if (m_autoRunAfterDownload) onRun();
        } else {
            appendLog(tr("[Error] Download failed."));
            QFile::remove(m_downloadTmpPath);
        }
        m_autoRunAfterDownload = false;
        m_downloadInProgress = false;
        m_downloadLabel->setVisible(false);
        m_currentDownload->deleteLater();
        m_currentDownload = nullptr;
    });
}

void DeepLSDDialog::cancelDownload() {
    if (m_currentDownload) m_currentDownload->abort();
}

void DeepLSDDialog::onRun() {
    if (!ensureModelAvailable()) return;
    const Settings s = getSettings();
    const QString fname = m_modelCombo->currentData().toString();
    if (fname == "CUSTOM" ? isQ8QuantModel(s.modelPath)
                          : isQ8QuantModel(fname)) {
        const QMessageBox::StandardButton btn = QMessageBox::warning(
                this, tr("Q8_0 accuracy warning"),
                tr("The selected Q8_0 model uses 8-bit weights and may produce "
                   "inaccurate wireframe distance/angle fields compared to "
                   "F16/F32 "
                   "(experimental; see MODEL_CARD.md).\n\n"
                   "Continue with Q8_0 anyway?"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (btn != QMessageBox::Yes) return;
    }
    emit runRequested(s);
}

void DeepLSDDialog::onCancel() { emit cancelRequested(); }
