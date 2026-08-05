// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FreeSplatterDialog.h"

#include <CVLog.h>

#include <QCloseEvent>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QScreen>
#include <QSettings>
#include <QStyle>
#include <QTabBar>
#include <QTimer>
#include <QVBoxLayout>

#include "FaceCaptureWidget.h"
#include "aicore/backend_capi.h"
#include "aicore/gaussian_capi.h"
#include "aicore/inference_log.h"
#include "ecvClickableImageLabel.h"
#include "ecvModelDownloader.h"

static const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "3dgs/";

static const int kThumbSize = 96;
static const int kThumbCaptionH = 18;
static const int kThumbRemoveBtnH = 20;
static const int kThumbTileSpacing = 8;
static const int kThumbStripHeight =
        kThumbSize + kThumbCaptionH + kThumbRemoveBtnH + kThumbTileSpacing;
// The capture form is deliberately scrollable.  Keeping the viewport bounded
// avoids making the reconstruction dialog taller than a typical desktop.
static const int kFaceCaptureViewportMaxHeight = 560;

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
            {tr("Object-2DGS Q8_0 (recommended)"),
             "freesplatter-object-2dgs-q8_0.gguf",
             base + "freesplatter-object-2dgs-q8_0.gguf"},
            {tr("Object-2DGS F16"), "freesplatter-object-2dgs-f16.gguf",
             base + "freesplatter-object-2dgs-f16.gguf"},
            {tr("Object-2DGS F32 (full precision)"),
             "freesplatter-object-2dgs-f32.gguf",
             base + "freesplatter-object-2dgs-f32.gguf"},
            {tr("Object-3DGS Q8_0 (deprecated)"),
             "freesplatter-object-q8_0.gguf",
             base + "freesplatter-object-q8_0.gguf"},
            {tr("Object-3DGS F16 (deprecated)"), "freesplatter-object-f16.gguf",
             base + "freesplatter-object-f16.gguf"},
            {tr("Object-3DGS F32 (deprecated)"), "freesplatter-object-f32.gguf",
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
    // The capture form has two rows of paired controls.  Keep enough width to
    // show those rows rather than introducing a horizontal scroll bar.
    setMinimumWidth(800);
    setMinimumHeight(0);
    setupUi();
    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            &FreeSplatterDialog::appendLog);
    connect(m_downloader, &ecvModelDownloader::progress, this,
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
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& dest) {
                Q_UNUSED(dest);
                const QString finishedFilename = m_downloadTargetFilename;
                const bool shouldAutoRun = m_autoRunAfterDownload;
                m_downloadInProgress = false;
                m_downloadLabel->setVisible(false);
                m_progressBar->setValue(ok ? 100 : 0);
                populateModelCombo(finishedFilename);
                updateRunButtonState();

                if (ok && shouldAutoRun) {
                    m_autoRunAfterDownload = false;
                    selectModelByFilename(finishedFilename);
                    onRun();
                } else if (!ok) {
                    m_autoRunAfterDownload = false;
                }
            });
    CVLog::Print(
            QString("[FreeSplatter] Model cache: %1").arg(modelCacheDir()));
    aicore_inference_log::log_backend_probe(QStringLiteral("FS"));
}

void FreeSplatterDialog::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
}

void FreeSplatterDialog::setupUi() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(4);
    mainLayout->setSizeConstraint(QLayout::SetNoConstraint);

    // --- Model & Mode (merged into one group) ---
    auto* modelGroup = new QGroupBox("Model");
    auto* modelLayout = new QGridLayout(modelGroup);
    modelLayout->setVerticalSpacing(4);

    auto* pipelineHint = new QLabel(tr(
            "<b>Pipeline:</b> <i>Face detect</i> → <i>Multi-view capture</i> "
            "→ <i>Gaussian 3D reconstruct</i> → <i>Export / SIBR</i>"));
    pipelineHint->setWordWrap(true);
    pipelineHint->setStyleSheet(
            "color: #334155; background: #f8fafc; border: 1px solid #cbd5e1; "
            "padding: 4px 8px; border-radius: 4px; font-size: 11px;");
    modelLayout->addWidget(pipelineHint, 0, 0, 1, 4);

    modelLayout->addWidget(new QLabel("Mode:"), 1, 0);
    m_modeCombo = new QComboBox;
    m_modeCombo->addItem("3D Reconstruct (Gaussian)",
                         static_cast<int>(Mode::Reconstruct));
    m_modeCombo->addItem("Model Info", static_cast<int>(Mode::ModelInfo));
    connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FreeSplatterDialog::onModeChanged);
    modelLayout->addWidget(m_modeCombo, 1, 1);
    modelLayout->addWidget(new QLabel("GGUF:"), 1, 2);
    m_modelCombo = new QComboBox;
    m_modelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    modelLayout->addWidget(m_modelCombo, 1, 3);
    connect(m_modelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FreeSplatterDialog::onModelComboChanged);

    m_customModelRow = new QWidget;
    auto* customModelLayout = new QHBoxLayout(m_customModelRow);
    customModelLayout->setContentsMargins(0, 0, 0, 0);
    m_customModelPath = new QLineEdit;
    m_customModelPath->setPlaceholderText("Path to custom .gguf file");
    connect(m_customModelPath, &QLineEdit::textChanged, this, [this]() {
        updateObjectModelHint();
        updateImageCountStatus();
        updateRunButtonState();
    });
    customModelLayout->addWidget(m_customModelPath, 1);
    m_browseCustomModelBtn = new QPushButton("Browse...");
    connect(m_browseCustomModelBtn, &QPushButton::clicked, this,
            &FreeSplatterDialog::onBrowseCustomModel);
    customModelLayout->addWidget(m_browseCustomModelBtn);
    m_customModelRow->setVisible(false);
    modelLayout->addWidget(m_customModelRow, 2, 0, 1, 4);

    m_objectHintLabel = new QLabel;
    m_objectHintLabel->setTextFormat(Qt::RichText);
    m_objectHintLabel->setWordWrap(true);
    m_objectHintLabel->setStyleSheet(
            "QLabel { color: #b58900; font-size: 11px; padding: 2px 4px; "
            "background: #fffde7; border-radius: 3px; }");
    m_objectHintLabel->setVisible(false);
    modelLayout->addWidget(m_objectHintLabel, 3, 0, 1, 4);

    auto* runtimeRow = new QWidget(modelGroup);
    auto* runtimeLayout = new QHBoxLayout(runtimeRow);
    runtimeLayout->setContentsMargins(0, 0, 0, 0);
    runtimeLayout->setSpacing(8);
    runtimeLayout->addWidget(new QLabel(tr("Device:")));
    m_deviceCombo = new QComboBox;
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* d = aicore_device_at(i);
        m_deviceCombo->addItem(tr(d->label), QString::fromUtf8(d->id));
        if (d->is_default) m_deviceCombo->setCurrentIndex(i);
    }
    m_deviceCombo->setToolTip(
            tr("Auto tries %1.").arg(aicore_auto_device_order()));
    runtimeLayout->addWidget(m_deviceCombo, 1);
    runtimeLayout->addWidget(new QLabel(tr("Threads:")));
    m_threads = new QSpinBox;
    m_threads->setRange(0, 128);
    m_threads->setSpecialValueText("Auto");
    runtimeLayout->addWidget(m_threads);
    runtimeLayout->addWidget(new QLabel(tr("Views:")));
    m_maxViewsSpin = new QSpinBox;
    m_maxViewsSpin->setRange(0, 64);
    m_maxViewsSpin->setSpecialValueText("Auto");
    m_maxViewsSpin->setToolTip(
            tr("Max input views for inference.\n"
               "Auto: Scene=2, Object-3DGS=16, Object-2DGS=24.\n"
               "Trained with up to 32 views; more views = better quality.\n"
               "O(N\u00b2) compute scaling; 16 views \u2248 30-60s on Metal."));
    runtimeLayout->addWidget(m_maxViewsSpin);
    modelLayout->addWidget(runtimeRow, 3, 0, 1, 4);

    mainLayout->addWidget(modelGroup);

    // --- I/O configuration ---
    auto* ioGroup = new QGroupBox("Input / Output");
    auto* ioMainLayout = new QVBoxLayout(ioGroup);
    ioMainLayout->setSpacing(2);
    ioMainLayout->setContentsMargins(4, 6, 4, 4);

    m_inputTabWidget = new QTabWidget;
    m_inputTabWidget->setDocumentMode(true);
    m_inputTabWidget->tabBar()->setDrawBase(false);

    // ---- Tab 0: Images ----
    {
        m_imagesTab = new QWidget;
        auto* imagesLayout = new QVBoxLayout(m_imagesTab);
        imagesLayout->setContentsMargins(2, 2, 2, 2);
        imagesLayout->setSpacing(2);

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
        imagesLayout->addLayout(inputBtnLayout);

        m_thumbScroll = new QScrollArea;
        m_thumbScroll->setWidgetResizable(true);
        m_thumbScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_thumbScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        m_thumbScroll->setSizePolicy(QSizePolicy::Expanding,
                                     QSizePolicy::Fixed);
        m_thumbScroll->setFixedHeight(kThumbStripHeight);
        m_thumbScroll->setFrameShape(QFrame::StyledPanel);
        m_thumbContainer = new QWidget;
        m_thumbContainer->setMinimumHeight(kThumbStripHeight - 4);
        auto* thumbLayout = new QHBoxLayout(m_thumbContainer);
        thumbLayout->setContentsMargins(4, 4, 4, 4);
        m_thumbScroll->setWidget(m_thumbContainer);
        imagesLayout->addWidget(m_thumbScroll);

        // DB Images collapsible
        auto* dbRow = new QHBoxLayout;
        m_dbToggleBtn = new QToolButton;
        m_dbToggleBtn->setArrowType(Qt::RightArrow);
        m_dbToggleBtn->setCheckable(true);
        m_dbToggleBtn->setChecked(false);
        m_dbToggleBtn->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        m_dbToggleBtn->setText(tr("DB Images"));
        m_dbToggleBtn->setCursor(Qt::PointingHandCursor);
        m_dbToggleBtn->setStyleSheet(
                "QToolButton { border: none; font-weight: bold; padding: 4px "
                "6px; "
                "  border-radius: 3px; color: palette(text); }"
                "QToolButton:hover { background: palette(midlight); }");
        dbRow->addWidget(m_dbToggleBtn);
        dbRow->addStretch();
        imagesLayout->addLayout(dbRow);

        m_dbContentWidget = new QWidget;
        m_dbContentWidget->setStyleSheet(
                "QWidget#dbContent { "
                "  border: 1px solid palette(mid); "
                "  border-radius: 4px; "
                "  background: palette(base); }");
        m_dbContentWidget->setObjectName("dbContent");
        auto* dbCol = new QVBoxLayout(m_dbContentWidget);
        dbCol->setContentsMargins(4, 4, 4, 4);
        dbCol->setSpacing(4);
        m_dbImageList = new QListWidget;
        m_dbImageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
        m_dbImageList->setMinimumHeight(72);
        m_dbImageList->setMaximumHeight(120);
        m_dbImageList->setAlternatingRowColors(true);
        m_dbImageList->setToolTip(
                tr("ccImage entities from the DB tree \u2014 check/uncheck to "
                   "add or remove from input"));
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
        m_dbContentWidget->setVisible(false);
        imagesLayout->addWidget(m_dbContentWidget);

        connect(m_dbToggleBtn, &QToolButton::toggled, this,
                [this](bool checked) {
                    m_dbToggleBtn->setArrowType(checked ? Qt::DownArrow
                                                        : Qt::RightArrow);
                    m_dbContentWidget->setVisible(checked);
                    adaptTabWidgetHeight();
                });

        m_inputTabWidget->addTab(m_imagesTab, tr("Images"));
    }

    // ---- Tab 1: Face Capture ----
    if (FaceCaptureWidget::isAvailable()) {
        auto* faceTab = new QWidget;
        auto* faceLayout = new QVBoxLayout(faceTab);
        faceLayout->setContentsMargins(0, 0, 0, 0);

        m_faceCaptureWidget = new FaceCaptureWidget(faceTab);
        m_faceCaptureWidget->setInferenceDevice(
                m_deviceCombo ? m_deviceCombo->currentData().toString()
                              : QStringLiteral("auto"));
        connect(m_deviceCombo,
                QOverload<int>::of(&QComboBox::currentIndexChanged),
                m_faceCaptureWidget, [this](int) {
                    m_faceCaptureWidget->setInferenceDevice(
                            m_deviceCombo->currentData().toString());
                });
        m_faceCaptureScroll = new QScrollArea(faceTab);
        m_faceCaptureScroll->setWidgetResizable(true);
        m_faceCaptureScroll->setFrameShape(QFrame::NoFrame);
        m_faceCaptureScroll->setHorizontalScrollBarPolicy(
                Qt::ScrollBarAlwaysOff);
        m_faceCaptureScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        m_faceCaptureScroll->setSizePolicy(QSizePolicy::Expanding,
                                           QSizePolicy::Expanding);
        // Keep the complete capture form as the scroll area's widget.  The
        // viewport may shrink on small displays, but it must never collapse
        // the form to zero height.
        m_faceCaptureWidget->setSizePolicy(QSizePolicy::Expanding,
                                           QSizePolicy::Preferred);
        m_faceCaptureScroll->setMinimumHeight(280);
        m_faceCaptureScroll->setMaximumHeight(kFaceCaptureViewportMaxHeight);
        m_faceCaptureScroll->setWidget(m_faceCaptureWidget);
        faceLayout->addWidget(m_faceCaptureScroll, 1);

        auto* faceBtnLayout = new QHBoxLayout;
        faceBtnLayout->setContentsMargins(0, 0, 0, 0);
        faceBtnLayout->setSpacing(6);
        m_faceStartBtn = new QPushButton(tr("Start Capture"));
        m_faceStopBtn = new QPushButton(tr("Stop Capture"));
        m_faceStopBtn->setEnabled(false);
        m_faceResetBtn = new QPushButton(tr("Reset"));
        m_faceResetBtn->setEnabled(false);
        for (QPushButton* btn :
             {m_faceStartBtn, m_faceStopBtn, m_faceResetBtn}) {
            btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        }
        faceBtnLayout->addWidget(m_faceStartBtn);
        faceBtnLayout->addWidget(m_faceStopBtn);
        faceBtnLayout->addWidget(m_faceResetBtn);
        faceLayout->addLayout(faceBtnLayout);

        connect(m_faceStartBtn, &QPushButton::clicked, this,
                &FreeSplatterDialog::onFaceStartCamera);
        connect(m_faceStopBtn, &QPushButton::clicked, this,
                &FreeSplatterDialog::onFaceStopCamera);
        connect(m_faceResetBtn, &QPushButton::clicked, this,
                &FreeSplatterDialog::onFaceReset);
        connect(m_faceCaptureWidget, &FaceCaptureWidget::captureComplete, this,
                &FreeSplatterDialog::onFaceCaptureComplete);
        connect(m_faceCaptureWidget, &FaceCaptureWidget::cameraStarted, this,
                [this]() {
                    m_faceStartBtn->setEnabled(false);
                    m_faceStopBtn->setEnabled(true);
                    if (m_faceCaptureWidget->inputSource() ==
                        FaceCaptureWidget::InputSource::Camera) {
                        m_faceCaptureWidget->startGuidedCapture({
                                FaceCaptureWidget::CaptureAngle::Front,
                                FaceCaptureWidget::CaptureAngle::Left45,
                                FaceCaptureWidget::CaptureAngle::Right45,
                                FaceCaptureWidget::CaptureAngle::Up15,
                                FaceCaptureWidget::CaptureAngle::Down15,
                        });
                    } else {
                        m_faceCaptureWidget->startGuidedCapture({
                                FaceCaptureWidget::CaptureAngle::Front,
                                FaceCaptureWidget::CaptureAngle::Left45,
                                FaceCaptureWidget::CaptureAngle::Right45,
                                FaceCaptureWidget::CaptureAngle::Left90,
                                FaceCaptureWidget::CaptureAngle::Right90,
                                FaceCaptureWidget::CaptureAngle::Up15,
                        });
                        m_faceResetBtn->setEnabled(true);
                    }
                });
        connect(m_faceCaptureWidget, &FaceCaptureWidget::cameraStopped, this,
                [this]() {
                    m_faceStartBtn->setEnabled(true);
                    m_faceStopBtn->setEnabled(false);
                });
        connect(m_faceCaptureWidget, &FaceCaptureWidget::frameCaptured, this,
                [this](int idx, int total) {
                    appendLog(tr("[FaceCapture] Auto-captured %1/%2")
                                      .arg(idx)
                                      .arg(total));
                });
        connect(m_faceCaptureWidget, &FaceCaptureWidget::logMessage, this,
                &FreeSplatterDialog::appendLog);

        m_inputTabWidget->addTab(faceTab, tr("Face Capture"));
    }

    connect(m_inputTabWidget, &QTabWidget::currentChanged, this, [this](int) {
        // QTabWidget updates its stacked-page geometry after this signal.
        // Deferring avoids reading the previous page's height on a quick tab
        // switch, which previously kept the Images tab at Face Capture size.
        QTimer::singleShot(0, this, [this]() { adaptTabWidgetHeight(); });
    });
    ioMainLayout->addWidget(m_inputTabWidget);

    // --- Output settings (compact dual-column) ---
    auto* outputGrid = new QGridLayout;
    outputGrid->setContentsMargins(0, 2, 0, 0);
    outputGrid->setHorizontalSpacing(8);
    outputGrid->setVerticalSpacing(2);
    outputGrid->setColumnMinimumWidth(0, 74);
    outputGrid->setColumnMinimumWidth(2, 64);
    outputGrid->setColumnStretch(1, 1);
    outputGrid->setColumnStretch(3, 1);
    int row = 0;

    auto* opacityLabel = new QLabel("Opacity:");
    opacityLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    outputGrid->addWidget(opacityLabel, row, 0);
    m_opacityThreshold = new QDoubleSpinBox;
    m_opacityThreshold->setRange(0.0, 1.0);
    m_opacityThreshold->setSingleStep(0.01);
    m_opacityThreshold->setValue(0.05);
    m_opacityThreshold->setToolTip("Prune gaussians with opacity <= threshold");
    m_opacityThreshold->setMinimumWidth(120);
    m_opacityThreshold->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Fixed);
    outputGrid->addWidget(m_opacityThreshold, row, 1);
    m_exportFieldLabel = new QLabel("Export:");
    m_exportFieldLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    outputGrid->addWidget(m_exportFieldLabel, row, 2);
    m_exportFieldModeCombo = new QComboBox;
    m_exportFieldModeCombo->addItem(tr("Basic \u2014 XYZ+RGB+Opacity"),
                                    static_cast<int>(ExportFieldMode::Basic));
    m_exportFieldModeCombo->addItem(tr("Full \u2014 SH+scale+normals"),
                                    static_cast<int>(ExportFieldMode::Full));
    m_exportFieldModeCombo->setToolTip(
            tr("Basic: XYZ + RGB + Opacity.\n"
               "Full: also SH, scale scalar fields and thin-axis normals."));
    m_exportFieldModeCombo->setMinimumWidth(190);
    m_exportFieldModeCombo->setSizePolicy(QSizePolicy::Expanding,
                                          QSizePolicy::Fixed);
    outputGrid->addWidget(m_exportFieldModeCombo, row, 3);

    row++;
    m_addToDbCheck = new QCheckBox("Add to DB tree");
    m_addToDbCheck->setChecked(true);
    m_addToDbCheck->setToolTip(
            "Add colored point cloud to the database tree after inference.");
    outputGrid->addWidget(m_addToDbCheck, row, 0, 1, 2,
                          Qt::AlignLeft | Qt::AlignVCenter);
    m_estimatePosesCheck = new QCheckBox("Estimate poses");
    m_estimatePosesCheck->setChecked(false);
    m_estimatePosesCheck->setToolTip("Estimate camera poses (multi-view)");
    outputGrid->addWidget(m_estimatePosesCheck, row, 2, 1, 2,
                          Qt::AlignLeft | Qt::AlignVCenter);

    row++;
    m_removeBgCheck = new QCheckBox("Remove background (Object model)");
    m_removeBgCheck->setChecked(false);
    m_removeBgCheck->setVisible(false);
    m_removeBgCheck->setToolTip(
            tr("Auto-remove backgrounds using GrabCut before inference.\n"
               "Recommended for Object models with non-white backgrounds."));
    outputGrid->addWidget(m_removeBgCheck, row, 0, 1, 4);

    row++;
    m_imageCountLabel = new QLabel;
    m_imageCountLabel->setStyleSheet("font-weight: bold;");
    outputGrid->addWidget(m_imageCountLabel, row, 0, 1, 4);

    ioMainLayout->addLayout(outputGrid);

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
    m_progressBar->setFixedHeight(14);
    m_progressBar->setTextVisible(false);
    m_progressBar->setVisible(false);
    mainLayout->addWidget(m_progressBar);

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
    adaptTabWidgetHeight();
}

void FreeSplatterDialog::adaptTabWidgetHeight() {
    if (!m_inputTabWidget) return;
    const int idx = m_inputTabWidget->currentIndex();
    QWidget* current = m_inputTabWidget->widget(idx);
    if (!current) return;

    const int tabChrome = m_inputTabWidget->tabBar()->sizeHint().height() +
                          2 * m_inputTabWidget->style()->pixelMetric(
                                      QStyle::PM_DefaultFrameWidth);
    int contentHeight = current->minimumSizeHint().height();
    if (current == m_imagesTab) {
        // An empty image tab needs only its commands and thumbnail strip.
        contentHeight = qBound(150, contentHeight, 210);
    } else {
        const QScreen* screen =
                QGuiApplication::screenAt(frameGeometry().center());
        const int available =
                screen ? screen->availableGeometry().height() : 800;
        // A QScrollArea reports only its viewport minimum.  Use the actual
        // form's size hint so the first capture controls stay visible, then
        // constrain the viewport to the current screen and retain scrolling
        // for the rest of the form.
        const int formHeight =
                m_faceCaptureWidget ? m_faceCaptureWidget->sizeHint().height()
                                    : contentHeight;
        const int dialogChrome = height() - m_inputTabWidget->height();
        const int viewportBudget =
                std::max(280, available - std::max(220, dialogChrome) - 32);
        contentHeight = std::min(
                formHeight,
                std::min(viewportBudget, kFaceCaptureViewportMaxHeight));
    }
    const int targetHeight = tabChrome + contentHeight;
    m_inputTabWidget->setFixedHeight(targetHeight);
    m_inputTabWidget->updateGeometry();

    if (isVisible() && m_activeInputTabHeight >= 0 &&
        targetHeight != m_activeInputTabHeight) {
        const QScreen* screen =
                QGuiApplication::screenAt(frameGeometry().center());
        const int available =
                screen ? screen->availableGeometry().height() : 800;
        const int requested = height() + targetHeight - m_activeInputTabHeight;
        resize(width(), qBound(360, requested, available - 20));
    }
    m_activeInputTabHeight = targetHeight;
}

void FreeSplatterDialog::refreshModelList() { populateModelCombo(); }

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
        if (ecvModelDownloader::isValidCachedFile(fi.absoluteFilePath())) {
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

    const bool isObject =
            data.contains("object", Qt::CaseInsensitive) ||
            (data == "CUSTOM" && m_customModelPath &&
             m_customModelPath->text().contains("object", Qt::CaseInsensitive));
    if (m_removeBgCheck) {
        m_removeBgCheck->setVisible(isObject);
        if (!isObject) m_removeBgCheck->setChecked(false);
    }
    updateObjectModelHint();

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

bool FreeSplatterDialog::isObject2dgsModel(const QString& filename) {
    return filename.contains("object", Qt::CaseInsensitive) &&
           filename.contains("2dgs", Qt::CaseInsensitive);
}

QString FreeSplatterDialog::currentModelFilename() const {
    QString data = m_modelCombo->currentData().toString();
    if (data == "CUSTOM") {
        return m_customModelPath ? m_customModelPath->text().trimmed()
                                 : QString();
    }
    return data;
}

void FreeSplatterDialog::updateObjectModelHint() {
    if (!m_objectHintLabel) return;

    const QString filename = currentModelFilename();
    const bool isObject = filename.contains("object", Qt::CaseInsensitive);
    if (!isObject) {
        m_objectHintLabel->setVisible(false);
        if (m_modelCombo) m_modelCombo->setToolTip(QString());
        return;
    }

    m_objectHintLabel->setVisible(true);
    const bool is2dgs = isObject2dgsModel(filename);
    if (is2dgs) {
        m_objectHintLabel->setText(
                tr("\u26a0 <b>Object-2DGS (recommended)</b>: use "
                   "background-removed photos (Remove BG). Oriented 2D surfels "
                   "\u2014 sharper surfaces, fewer floaters on thin edges. "
                   "Best with <b>8\u201324 views</b> (Auto=24). Ideal for "
                   "products, props, and multi-view object capture."));
        if (m_modelCombo) {
            m_modelCombo->setToolTip(
                    tr("Object-2DGS: 22-channel 2D Gaussian surfels.\n"
                       "Pros: cleaner surfaces, scales to more views.\n"
                       "Cons: slower with 16+ views (O(N\u00b2)); needs clean "
                       "background.\n"
                       "Use Object-3DGS only for legacy / quick 3\u20138 view "
                       "tests."));
        }
    } else {
        m_objectHintLabel->setText(
                tr("\u26a0 <b>Object-3DGS (deprecated)</b>: prefer "
                   "<b>Object-2DGS</b> for new work. Full 3D ellipsoid "
                   "Gaussians \u2014 OK for quick tests with "
                   "<b>3\u20138 views</b> (Auto=16) but more floaters on thin "
                   "geometry. Background removal still recommended."));
        if (m_modelCombo) {
            m_modelCombo->setToolTip(
                    tr("Object-3DGS (legacy): 23-channel full 3D Gaussians.\n"
                       "Pros: works with fewer views, slightly faster total "
                       "run.\n"
                       "Cons: blobbier thin parts, more floaters.\n"
                       "Switch to Object-2DGS when you have 8+ clean views."));
        }
    }
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
            return 2;
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
    if (ecvModelDownloader::isValidCachedFile(modelCacheDir() + "/" + data))
        return true;
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
    const QString filename = currentModelFilename();
    if (type == ModelType::Object && isObject2dgsModel(filename)) {
        typeName = "Object-2DGS";
    } else if (type == ModelType::Object) {
        typeName = "Object-3DGS";
    }
    QString reqStr =
            (type == ModelType::Scene)
                    ? QString("at least %1 (recommended %1)").arg(required)
            : (type == ModelType::Object && isObject2dgsModel(filename))
                    ? QString("at least %1 (recommended 8\u201324, "
                              "Auto=24)")
                              .arg(required)
                    : QString("at least %1 (recommended 3\u20138, "
                              "Auto=16)")
                              .arg(required);
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

        auto* imgLabel = new ecvClickableImageLabel;
        imgLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        QImage img = previewForPath(path);
        if (!img.isNull()) {
            imgLabel->setPreviewImage(img, kThumbSize);
        } else {
            imgLabel->setFixedSize(kThumbSize, kThumbSize);
            imgLabel->setText("?");
            imgLabel->setFrameShape(QFrame::Box);
        }
        imgLabel->setToolTip(path);
        tileLayout->addWidget(ecvClickableImageLabel::wrapWithTapToPreviewHint(
                                      imgLabel, tile),
                              0, Qt::AlignHCenter);

        QString caption = path.startsWith("db://") ? path.mid(5)
                                                   : QFileInfo(path).fileName();
        auto* nameLabel = new QLabel(caption);
        nameLabel->setFixedHeight(kThumbCaptionH);
        nameLabel->setMaximumWidth(kThumbSize + 8);
        nameLabel->setAlignment(Qt::AlignCenter);
        nameLabel->setWordWrap(false);
        nameLabel->setTextFormat(Qt::PlainText);
        nameLabel->setToolTip(caption);
        const QFontMetrics fm(nameLabel->font());
        nameLabel->setText(
                fm.elidedText(caption, Qt::ElideMiddle, kThumbSize + 8));

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
    if (replace) {
        m_inputPaths.clear();
        if (!m_identityInputs.isEmpty() &&
            paths != m_identityInputs.front().inputPaths) {
            m_identityInputs.clear();
        }
    }
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
    if (ecvModelDownloader::isValidCachedFile(cached)) return true;

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
    updateRunButtonState();

    ecvModelDownloader::Request req;
    req.url = model.downloadUrl;
    req.destPath = dest;
    m_downloader->download(req);
}

void FreeSplatterDialog::cancelDownload() {
    if (!m_downloadInProgress) return;
    m_autoRunAfterDownload = false;
    if (m_downloader) m_downloader->cancel();
    m_downloadInProgress = false;
    m_downloadLabel->setVisible(false);
    m_progressBar->setValue(0);
    updateRunButtonState();
}

void FreeSplatterDialog::onCancel() {
    if (m_downloadInProgress) {
        cancelDownload();
        return;
    }
    emit cancelRequested();
}

void FreeSplatterDialog::closeEvent(QCloseEvent* event) {
    onCancel();
    onFaceStopCamera();
    if (m_faceCaptureWidget) {
        m_faceCaptureWidget->releaseGpuResources();
    }
    clearFaceCaptureExportDir();
    QDialog::closeEvent(event);
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
        if (m_dbToggleBtn) {
            m_dbToggleBtn->setText(tr("DB Images"));
            m_dbToggleBtn->setChecked(false);
        }
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
        if (m_dbToggleBtn) {
            m_dbToggleBtn->setText(tr("DB Images (%1)").arg(images.size()));
            m_dbToggleBtn->setChecked(true);
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

    if (m_inputTabWidget) m_inputTabWidget->setVisible(isReconstruct);
    m_opacityThreshold->setVisible(isReconstruct);
    if (m_exportFieldLabel) m_exportFieldLabel->setVisible(isReconstruct);
    if (m_exportFieldModeCombo)
        m_exportFieldModeCombo->setVisible(isReconstruct);
    m_addToDbCheck->setVisible(isReconstruct);
    m_estimatePosesCheck->setVisible(isReconstruct);
    if (m_removeBgCheck) {
        const bool isObject = currentModelType() == ModelType::Object;
        m_removeBgCheck->setVisible(isReconstruct && isObject);
    }
    m_imageCountLabel->setVisible(isReconstruct);
    if (isReconstruct) updateImageCountStatus();
    updateRunButtonState();
}

void FreeSplatterDialog::setRunning(bool running) {
    m_taskRunning = running;
    if (running) {
        m_lastTaskError.clear();
        m_taskStatusLabel->setText(tr("Starting..."));
        m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
        m_taskStatusLabel->setVisible(true);
        m_progressBar->setVisible(true);
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
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
        m_progressBar->setVisible(false);
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
    }
    updateRunButtonState();
    m_modeCombo->setEnabled(!running && !m_downloadInProgress);
}

void FreeSplatterDialog::setTaskStage(const QString& stage, int percent) {
    if (!m_taskStatusLabel) return;
    m_taskStatusLabel->setText(stage);
    m_taskStatusLabel->setStyleSheet("font-weight: bold; color: #0066cc;");
    m_taskStatusLabel->setVisible(true);
    m_progressBar->setVisible(true);
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
    s.removeBackground = m_removeBgCheck && m_removeBgCheck->isChecked();
    s.maxViews = m_maxViewsSpin ? m_maxViewsSpin->value() : 0;
    s.identityInputs = m_identityInputs;
    if (!m_identityInputs.isEmpty()) {
        s.identityId = m_identityInputs.front().id;
        s.identityName = m_identityInputs.front().name;
    }
    return s;
}

void FreeSplatterDialog::appendLog(const QString& msg) {
    aicore_inference_log::log(msg);
    if (!m_taskStatusLabel || !msg.startsWith(QStringLiteral("[Error]"))) {
        return;
    }
    // The generic task-failed message follows the actionable backend/input
    // error. Preserve the first concrete reason until the next run starts.
    if (m_lastTaskError.isEmpty()) {
        m_lastTaskError = msg.mid(QStringLiteral("[Error]").size()).trimmed();
    }
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
               currentImageCount() > 24) {
        appendLog(
                tr("[Warning] Object model trained with up to 32 views; "
                   "inputs above 24 will be uniformly subsampled."));
    }
    emit runRequested(getSettings());
}

// ---- Face Capture integration ----

void FreeSplatterDialog::onFaceStartCamera() {
    if (!m_faceCaptureWidget) return;
    if (m_faceCaptureWidget->inputSource() ==
        FaceCaptureWidget::InputSource::VideoFile) {
        const QString path = m_faceCaptureWidget->videoFilePath();
        if (path.isEmpty() || !QFileInfo::exists(path)) {
            appendLog(tr("[FaceCapture] Select a valid video file first."));
            return;
        }
        if (!m_faceCaptureWidget->startVideoFile(path)) {
            appendLog(tr("[FaceCapture] Failed to start video playback."));
        }
        return;
    }
    const int camIdx = m_faceCaptureWidget->selectedCameraIndex();
    if (camIdx < 0) {
        appendLog(tr("[FaceCapture] No camera available."));
        return;
    }
    m_faceCaptureWidget->startCamera(camIdx);
}

void FreeSplatterDialog::onFaceStopCamera() {
    if (!m_faceCaptureWidget) return;
    m_faceCaptureWidget->requestInferenceCancel();
    m_faceCaptureWidget->stopCamera();
}

void FreeSplatterDialog::onFaceReset() {
    if (!m_faceCaptureWidget) return;
    m_faceCaptureWidget->resetCapture();
    m_identityInputs.clear();
    clearFaceCaptureExportDir();
    if (m_faceResetBtn) m_faceResetBtn->setEnabled(false);
}

void FreeSplatterDialog::onFaceCaptureComplete() {
    if (!m_faceCaptureWidget || m_faceCaptureWidget->capturedCount() == 0)
        return;

    const int minCaptures = m_faceCaptureWidget->minCapturesBeforeComplete();
    if (m_faceCaptureWidget->capturedCount() < minCaptures) {
        appendLog(tr("[FaceCapture] Need at least %1 captured faces before "
                     "reconstruction (got %2).")
                          .arg(minCaptures)
                          .arg(m_faceCaptureWidget->capturedCount()));
        return;
    }

    clearFaceCaptureExportDir();
    QString cacheRoot =
            QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
    if (cacheRoot.isEmpty()) {
        cacheRoot = QStandardPaths::writableLocation(
                QStandardPaths::AppLocalDataLocation);
    }
    m_faceCaptureExportDir = QDir(cacheRoot).filePath(
            QStringLiteral("qFreeSplatter/face_capture/") +
            QUuid::createUuid().toString(QUuid::Id128));
    const std::vector<FaceCaptureWidget::IdentityImageBatch> batches =
            m_faceCaptureWidget->exportCapturedIdentityImages(
                    m_faceCaptureExportDir);
    if (batches.empty()) {
        appendLog(tr("[Error] Failed to export captured face images"));
        return;
    }

    m_identityInputs.clear();
    for (const FaceCaptureWidget::IdentityImageBatch& batch : batches) {
        if (batch.paths.size() < minCaptures) {
            appendLog(tr("[FaceCapture] Identity %1 has only %2/%3 captures")
                              .arg(batch.name)
                              .arg(batch.paths.size())
                              .arg(minCaptures));
            return;
        }
        Settings::IdentityInput input;
        input.id = batch.id;
        input.name = batch.name;
        input.inputPaths = batch.paths;
        m_identityInputs.push_back(std::move(input));
    }

    const QStringList saved = m_identityInputs.front().inputPaths;

    addInputPaths(saved, true);
    appendLog(tr("[FaceCapture] Prepared %1 identities / %2 face images for "
                 "separate reconstruction")
                      .arg(m_identityInputs.size())
                      .arg(m_faceCaptureWidget->capturedCount()));

    m_faceCaptureWidget->stopCamera();

    if (m_inputTabWidget) m_inputTabWidget->setCurrentIndex(0);

    updateRunButtonState();

    if (isModelReady() && isInputValid()) {
        appendLog(tr("[FaceCapture] Auto-starting reconstruction..."));
        onRun();
    }
}

void FreeSplatterDialog::clearFaceCaptureExportDir() {
    if (m_faceCaptureExportDir.isEmpty()) return;
    QDir(m_faceCaptureExportDir).removeRecursively();
    m_faceCaptureExportDir.clear();
}

void FreeSplatterDialog::clearFaceCaptureTransientInputs() {
    const QString exportedRoot = m_faceCaptureExportDir;
    clearFaceCaptureExportDir();
    if (exportedRoot.isEmpty()) return;
    m_identityInputs.clear();
    for (auto it = m_inputPaths.begin(); it != m_inputPaths.end();) {
        if (it->startsWith(exportedRoot)) {
            it = m_inputPaths.erase(it);
        } else {
            ++it;
        }
    }
    refreshThumbnailStrip();
    updateImageCountStatus();
    updateRunButtonState();
}
