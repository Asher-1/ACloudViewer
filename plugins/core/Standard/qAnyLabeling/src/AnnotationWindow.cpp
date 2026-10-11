// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationWindow.h"

#include <aicore/aliked_capi.h>

#include <QAbstractItemModel>
#include <QAbstractItemView>
#include <QActionGroup>
#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QColor>
#include <QColorDialog>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontComboBox>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHash>
#include <QHeaderView>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QKeyEvent>
#include <QKeySequenceEdit>
#include <QLabel>
#include <QLineEdit>
#include <QLineF>
#include <QListWidget>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QPainterPath>
#include <QPalette>
#include <QPlainTextEdit>
#include <QPolygonF>
#include <QProgressDialog>
#include <QPushButton>
#include <QSaveFile>
#include <QScrollArea>
#include <QSet>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTableWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidgetAction>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "AnnotationLabelDialog.h"
#include "AnnotationLogging.h"
#include "AnnotationNavigator.h"
#include "AnnotationPopup.h"
#include "AnnotationYaml.h"
#include "LomaBridgeDialog.h"
#include "PointCloudWindow.h"
#include "RemoteServerClient.h"
#include "RemoteServerDialog.h"
#include "VideoAnnotationDialog.h"
#include "VideoFrameExtraction.h"
#include "VideoMaskImport.h"
#include "VideoTab.h"
#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/model_catalog_capi.h"
#include "aicore/ocr_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/yolo_capi.h"
#include "ecvAssetIntegrity.h"
#include "ecvModelDownloader.h"

namespace {
const QStringList kImageFilters = {
        QStringLiteral("*.jpg"),  QStringLiteral("*.jpeg"),
        QStringLiteral("*.png"),  QStringLiteral("*.bmp"),
        QStringLiteral("*.webp"), QStringLiteral("*.tif"),
        QStringLiteral("*.tiff")};

QString imageFilter() {
    return QObject::tr("Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)");
}

bool acceptsImageDrop(const QMimeData* mime) {
    if (!mime->hasUrls()) return false;
    for (const QUrl& url : mime->urls()) {
        const QFileInfo info(url.toLocalFile());
        if (info.isDir() ||
            (info.isFile() && kImageFilters.contains(QStringLiteral("*.%1").arg(
                                      info.suffix().toLower()))))
            return true;
    }
    return false;
}

QString iconPath(const QString& name) {
    return QStringLiteral(":/CC/plugin/qAnyLabeling/images/") + name;
}

QString quantization(const QString& filename) {
    const QString stem = QFileInfo(filename).completeBaseName();
    for (const QString& suffix :
         {QStringLiteral("q8_0"), QStringLiteral("q4_0"),
          QStringLiteral("q4_1"), QStringLiteral("q4_K"), QStringLiteral("f16"),
          QStringLiteral("f32"), QStringLiteral("q8")}) {
        if (stem.endsWith(QLatin1Char('-') + suffix, Qt::CaseInsensitive) ||
            stem.endsWith(QLatin1Char('_') + suffix, Qt::CaseInsensitive))
            return suffix;
    }
    return stem.section(QLatin1Char('-'), -1);
}

// Det GGUF paired with an OCR rec entry: same quantization tier from the
// catalog (the det graph is script-agnostic, so the ch det model serves
// both rec languages). Empty when the catalog has no matching tier.
QString ocrDetPartner(const QString& recFilename) {
    const QString quant = quantization(recFilename);
    for (int i = 0; i < aicore_ocr_model_count(); ++i) {
        const aicore_ocr_model_entry* entry = aicore_ocr_model_at(i);
        if (!entry || !entry->filename || !entry->role) continue;
        if (QString::fromLatin1(entry->role) != QStringLiteral("det")) continue;
        const QString filename = QString::fromUtf8(entry->filename);
        if (quantization(filename) == quant) return filename;
    }
    return QString();
}

QString modelFamily(const QString& filename) {
    const QString stem = QFileInfo(filename).completeBaseName();
    return stem.left(stem.size() - quantization(filename).size() - 1);
}

QString normalizeImageTag(const QString& value) {
    const QString tag = value.trimmed();
    if (tag.isEmpty() || tag.contains(QChar('\n')) || tag.contains(QChar('\r')))
        return QString();
    return tag;
}

QString nextImageTag(const QListWidget* tags) {
    QSet<QString> existing;
    for (int i = 0; i < tags->count(); ++i)
        existing.insert(tags->item(i)->text());
    if (!existing.contains(QStringLiteral("tag"))) return QStringLiteral("tag");
    int suffix = 1;
    while (existing.contains(QStringLiteral("tag%1").arg(suffix))) ++suffix;
    return QStringLiteral("tag%1").arg(suffix);
}
}  // namespace

AnnotationWindow::AnnotationWindow(QWidget* parent) : QMainWindow(parent) {
    setObjectName(QStringLiteral("qAnyLabelingWorkspace"));
    setWindowTitle(tr("X-AnyLabeling Annotation"));
    setMinimumSize(900, 600);
    resize(1320, 820);
    setAcceptDrops(true);
    qApp->installEventFilter(this);
    QSettings settings;
    settings.beginGroup(QStringLiteral("qAnyLabeling"));
    m_recentFiles =
            settings.value(QStringLiteral("recentFiles")).toStringList();
    m_directory = settings.value(QStringLiteral("lastDirectory")).toString();
    m_autoSave = settings.value(QStringLiteral("autoSave"), true).toBool();
    m_saveWithImageData =
            settings.value(QStringLiteral("storeData"), false).toBool();
    m_keepPreviousBrightness =
            settings.value(QStringLiteral("keepPreviousBrightness"), false)
                    .toBool();
    m_keepPreviousContrast =
            settings.value(QStringLiteral("keepPreviousContrast"), false)
                    .toBool();
    m_keepPreviousScale =
            settings.value(QStringLiteral("keepPreviousScale"), false).toBool();
    m_autoUseLastLabel =
            settings.value(QStringLiteral("autoUseLastLabel"), false).toBool();
    m_autoHighlightShape =
            settings.value(QStringLiteral("autoHighlightShape"), true).toBool();
    m_autoSwitchToEditMode =
            settings.value(QStringLiteral("autoSwitchToEditMode"), true)
                    .toBool();
    m_validateLabel =
            settings.value(QStringLiteral("validateLabel")).toString();
    m_loggerLevel = settings.value(QStringLiteral("loggerLevel"),
                                   QStringLiteral("info"))
                            .toString();
    m_remoteModelId =
            settings.value(QStringLiteral("remoteServerModel")).toString();
    m_exifScanEnabled =
            settings.value(QStringLiteral("exifScanEnabled"), true).toBool();
    m_displayLabelPopup =
            settings.value(QStringLiteral("displayLabelPopup"), true).toBool();
    // The source defaults this to false, but the ACloudViewer checked-image
    // workflow renders its state through the file-list checkboxes, so the
    // adapter keeps them visible by default.
    m_fileListCheckboxEditable =
            settings.value(QStringLiteral("fileListCheckboxEditable"), true)
                    .toBool();
    m_sortLabels = settings.value(QStringLiteral("sortLabels"), true).toBool();
    m_showLabelTextField =
            settings.value(QStringLiteral("showLabelTextField"), true).toBool();
    m_labelCompletion = settings.value(QStringLiteral("labelCompletion"),
                                       QStringLiteral("startswith"))
                                .toString();
    m_fitToContentColumn =
            settings.value(QStringLiteral("fitToContentColumn"), true).toBool();
    m_fitToContentRow =
            settings.value(QStringLiteral("fitToContentRow"), false).toBool();
    m_moveMode =
            settings.value(QStringLiteral("moveMode"), QStringLiteral("auto"))
                    .toString();
    m_fontFamily = settings.value(QStringLiteral("fontFamily")).toString();
    m_systemClipboard =
            settings.value(QStringLiteral("systemClipboard"), false).toBool();
    m_switchToChecked =
            settings.value(QStringLiteral("switchToChecked"), false).toBool();
    m_defaultBrushRadius = std::clamp(
            settings.value(QStringLiteral("brushRadius"), 12).toInt(), 1, 200);
    m_defaultShapeOpacity = std::clamp(
            settings.value(QStringLiteral("shapeOpacity"), 100).toInt(), 0,
            100);
    m_defaultMaskOpacity = std::clamp(
            settings.value(QStringLiteral("maskOpacity"), 80).toInt(), 0, 255);
    m_magicDefaultThreshold = std::clamp(
            settings.value(QStringLiteral("magicWand/defaultThreshold"), 15)
                    .toInt(),
            0, 255);
    m_magicDragSensitivity = std::max(
            0.1,
            settings.value(QStringLiteral("magicWand/dragSensitivity"), 3.0)
                    .toDouble());
    m_magicLuminanceWeight = std::clamp(
            settings.value(QStringLiteral("magicWand/luminanceWeight"), 0.5)
                    .toDouble(),
            0.0, 1.0);
    m_shapePointSize = std::clamp(
            settings.value(QStringLiteral("shape/pointSize"), 10).toInt(), 1,
            64);
    m_shapeLineWidth = std::clamp(
            settings.value(QStringLiteral("shape/lineWidth"), 4.0).toDouble(),
            0.5, 32.0);
    m_canvasEpsilon = std::clamp(
            settings.value(QStringLiteral("canvas/epsilon"), 10.0).toDouble(),
            0.5, 100.0);
    m_doubleClickClose =
            settings.value(QStringLiteral("canvas/doubleClickClose"), true)
                    .toBool();
    m_doubleClickEditLabel =
            settings.value(QStringLiteral("canvas/doubleClickEditLabel"), true)
                    .toBool();
    m_numBackups = std::clamp(
            settings.value(QStringLiteral("canvas/numBackups"), 10).toInt(), 1,
            1000);
    m_wheelRectangleEditing =
            settings.value(QStringLiteral("canvas/wheelRectangleEditing"), true)
                    .toBool();
    m_wheelAdjustStep = std::clamp(
            settings.value(QStringLiteral("canvas/wheelAdjustStep"), 2.0)
                    .toDouble(),
            0.1, 50.0);
    m_wheelScaleStep = std::clamp(
            settings.value(QStringLiteral("canvas/wheelScaleStep"), 0.05)
                    .toDouble(),
            0.001, 1.0);
    m_crosshairVisible =
            settings.value(QStringLiteral("canvas/crosshair/show"), true)
                    .toBool();
    m_crosshairWidth = std::clamp(
            settings.value(QStringLiteral("canvas/crosshair/width"), 2.0)
                    .toDouble(),
            0.5, 16.0);
    m_crosshairColor =
            QColor(settings.value(QStringLiteral("canvas/crosshair/color"),
                                  QStringLiteral("#00FF00"))
                           .toString());
    if (!m_crosshairColor.isValid()) m_crosshairColor = QColor("#00FF00");
    m_crosshairOpacity = std::clamp(
            settings.value(QStringLiteral("canvas/crosshair/opacity"), 0.5)
                    .toDouble(),
            0.0, 1.0);
    m_rotationLargeIncrement = std::clamp(
            settings.value(QStringLiteral("canvas/rotation/largeIncrement"),
                           1.0)
                    .toDouble(),
            0.01, 45.0);
    m_rotationSmallIncrement = std::clamp(
            settings.value(QStringLiteral("canvas/rotation/smallIncrement"),
                           0.1)
                    .toDouble(),
            0.01, 10.0);
    m_brushPointDistance = std::clamp(
            settings.value(QStringLiteral("canvas/brush/pointDistance"), 25.0)
                    .toDouble(),
            0.0, 500.0);
    m_brushSimplifyEpsilon = std::clamp(
            settings.value(QStringLiteral("canvas/brush/simplifyEpsilon"), 2.0)
                    .toDouble(),
            0.0, 100.0);
    m_brushMaxUndoSteps = std::clamp(
            settings.value(QStringLiteral("canvas/brush/maxUndoSteps"), 30)
                    .toInt(),
            1, 1000);
    m_brushMaxUndoMemoryMb = std::clamp(
            settings.value(QStringLiteral("canvas/brush/maxUndoMemoryMb"), 128)
                    .toInt(),
            1, 4096);
    m_magicSimplifyEpsilon = std::clamp(
            settings.value(QStringLiteral("canvas/magicWand/simplifyEpsilon"),
                           0.5)
                    .toDouble(),
            0.0, 100.0);
    m_magicOpacity = std::clamp(
            settings.value(QStringLiteral("canvas/magicWand/opacity"), 0.6)
                    .toDouble(),
            0.0, 1.0);
    m_cuboidMinDepth = std::clamp(
            settings.value(QStringLiteral("canvas/cuboid/minDepth"), 5.0)
                    .toDouble(),
            0.0, 10000.0);
    const QJsonDocument digitDocument = QJsonDocument::fromJson(
            settings.value(QStringLiteral("digitShortcuts")).toByteArray());
    if (digitDocument.isObject()) {
        const QJsonObject shortcuts = digitDocument.object();
        for (auto it = shortcuts.constBegin(); it != shortcuts.constEnd();
             ++it) {
            bool ok = false;
            const int digit = it.key().toInt(&ok);
            const QJsonObject value = it.value().toObject();
            if (ok && digit >= 0 && digit <= 9 &&
                value.value(QStringLiteral("mode")).isString() &&
                value.value(QStringLiteral("label")).isString())
                m_digitShortcuts.insert(
                        digit,
                        qMakePair(
                                value.value(QStringLiteral("mode")).toString(),
                                value.value(QStringLiteral("label"))
                                        .toString()));
        }
    }
    settings.endGroup();
    createUi();
    createActions();
    connect(m_canvas, &AnnotationCanvas::shapeFinished, this, [this](int) {
        if (!m_autoSwitchToEditMode || !m_tools) return;
        m_canvas->setTool(QStringLiteral("edit"));
        for (QAction* action : m_tools->actions()) {
            if (action->data().toString() == QStringLiteral("edit")) {
                action->setChecked(true);
                break;
            }
        }
        updateInstruction();
    });
    connect(m_canvas, &AnnotationCanvas::shapeDoubleClicked, this,
            [this](int) { editSelectedLabel(); });
    populateModels();
    // The upstream workspace starts with the optional AI panel collapsed.
    // Keep the toolbar action available without consuming canvas height at
    // startup.
    m_labelLoopPopup = new AnnotationPopup(this);
    m_labelLoopPopup->setObjectName(QStringLiteral("AnnotationReviewPopup"));
    {
        // Source shape.* color group and move mode (defaults follow
        // xanylabeling_config.yaml).
        const QSettings persist;
        const auto readColor = [&persist](const QString& key,
                                          const QColor& fallback) {
            const QColor parsed =
                    QColor(persist.value(QStringLiteral("qAnyLabeling/") + key)
                                   .toString());
            return parsed.isValid() ? parsed : fallback;
        };
        const QString strategy =
                persist.value(QStringLiteral("qAnyLabeling/shapeColorStrategy"),
                              QStringLiteral("auto"))
                        .toString();
        m_canvas->setShapeColorMode(
                strategy == QStringLiteral("manual")
                        ? AnnotationCanvas::ShapeColorMode::Manual
                        : AnnotationCanvas::ShapeColorMode::Auto);
        m_canvas->setDefaultShapeColor(readColor(
                QStringLiteral("defaultShapeColor"), QColor(0, 255, 0)));
        m_canvas->setShiftAutoShapeColor(
                persist.value(QStringLiteral(
                                      "qAnyLabeling/shiftAutoShapeColor"),
                              0)
                        .toInt());
        m_canvas->setShapeStyleColors(
                readColor(QStringLiteral("shapeLineColor"),
                          QColor(0, 255, 0, 128)),
                readColor(QStringLiteral("shapeFillColor"),
                          QColor(220, 220, 220, 150)),
                readColor(QStringLiteral("shapeVertexFillColor"),
                          QColor(0, 255, 0, 255)),
                readColor(QStringLiteral("shapeSelectLineColor"),
                          QColor(255, 255, 255, 255)),
                readColor(QStringLiteral("shapeSelectFillColor"),
                          QColor(0, 255, 0, 155)),
                readColor(QStringLiteral("shapeHoverVertexFillColor"),
                          QColor(255, 255, 255, 255)));
        m_canvas->setMoveMode(m_moveMode);
        if (!m_fontFamily.isEmpty()) setFont(QFont(m_fontFamily));
    }
    QTimer::singleShot(0, this, [this]() {
        if (m_modelPanel) m_modelPanel->hide();
    });

    m_thread = new QThread(this);
    m_worker = new AnnotationWorker;
    m_worker->moveToThread(m_thread);
    qRegisterMetaType<AnnotationInference>("AnnotationInference");
    qRegisterMetaType<AnnotationPrompt>("AnnotationPrompt");
    connect(this, &AnnotationWindow::inferenceRequested, m_worker,
            &AnnotationWorker::infer, Qt::QueuedConnection);
    connect(m_worker, &AnnotationWorker::finished, this,
            &AnnotationWindow::onInference, Qt::QueuedConnection);
    connect(m_thread, &QThread::finished, m_worker, &QObject::deleteLater);
    m_thread->start();

    m_downloader = new ecvModelDownloader(this);
    connect(m_downloader, &ecvModelDownloader::progress, this,
            [this](qint64 received, qint64 total) {
                m_status->setText(ecvModelDownloader::formatDownloadProgress(
                        received, total));
            });
    connect(m_downloader, &ecvModelDownloader::finished, this,
            [this](bool ok, const QString& path) {
                if (m_batchCancelRequested) {
                    finishBatch(tr("Auto Run cancelled (%1 images completed)")
                                        .arg(m_batchIndex));
                    return;
                }
                m_status->setText(ok ? tr("Model ready")
                                     : tr("Model download failed"));
                if (!ok && m_batchRunning)
                    finishBatch(tr("Auto Run stopped: model download failed"));
                if (ok && m_runAfterDownload &&
                    m_downloadGeneration == m_generation &&
                    m_downloadImagePath == m_imagePath &&
                    m_downloadFilename == m_quant->currentData().toString() &&
                    !m_image.isNull()) {
                    if (m_downloadDependencyModel) {
                        m_runAfterDownload = false;
                        m_downloadDependencyModel = false;
                        runDetection();
                        if (m_batchRunning && !m_inferenceRunning &&
                            !m_downloader->isBusy())
                            finishBatch(m_status->text());
                        return;
                    } else
                        startInference(path);
                }
                m_runAfterDownload = false;
                m_downloadDependencyModel = false;
                if (m_batchRunning && !m_inferenceRunning)
                    finishBatch(m_status->text());
            });
}

void AnnotationWindow::setAppInterface(ecvMainAppInterface* app) {
    m_app = app;
    if (m_videoDialog) {
        if (auto* tab = m_videoDialog->findChild<VideoTab*>())
            tab->setAppInterface(m_app);
    }
}

AnnotationWindow::~AnnotationWindow() {
    qApp->removeEventFilter(this);
    for (auto it = m_suspendedHostActions.constBegin();
         it != m_suspendedHostActions.constEnd(); ++it)
        it.key()->setEnabled(it.value());
    m_suspendedHostActions.clear();
    m_runAfterDownload = false;
    if (m_downloader && m_downloader->isBusy()) m_downloader->cancel();
    if (m_thread) {
        m_thread->quit();
        m_thread->wait();
    }
}

void AnnotationWindow::openImage(const QString& file) { openFile(file); }

void AnnotationWindow::createUi() {
    m_workspace = new QWidget(this);
    auto* workspaceLayout = new QHBoxLayout(m_workspace);
    workspaceLayout->setContentsMargins(0, 0, 0, 0);
    workspaceLayout->setSpacing(0);

    // Keep the tool strip inside the labeling workspace, matching the
    // upstream layout and preventing it from becoming another host toolbar.
    m_toolbarHost = new QWidget(m_workspace);
    m_toolbarHost->setObjectName(QStringLiteral("AnnotationToolbarHost"));
    m_toolbarHostLayout = new QVBoxLayout(m_toolbarHost);
    m_toolbarHostLayout->setContentsMargins(0, 0, 0, 0);
    m_toolbarHostLayout->setSpacing(0);
    m_toolbarScrollArea = new QScrollArea(m_workspace);
    m_toolbarScrollArea->setObjectName(
            QStringLiteral("AnnotationToolbarScrollArea"));
    m_toolbarScrollArea->setFrameShape(QFrame::NoFrame);
    m_toolbarScrollArea->setWidgetResizable(true);
    m_toolbarScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_toolbarScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    // Upstream leaves six pixels around the 40px tool buttons.
    m_toolbarScrollArea->setFixedWidth(46);
    m_toolbarScrollArea->setWidget(m_toolbarHost);
    workspaceLayout->addWidget(m_toolbarScrollArea);

    auto* central = new QWidget(this);
    auto* centralLayout = new QVBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->setSpacing(0);
    m_instruction = new QLabel(
            tr("Current mode: Edit | Shortcuts: Previous (A), Next (D), "
               "Rectangle (R), Polygon (P), Settings (Ctrl+0)"),
            central);
    m_instruction->setObjectName(QStringLiteral("AnnotationInstructions"));
    m_instruction->setContentsMargins(10, 4, 6, 4);
    m_instruction->setMinimumHeight(27);
    auto* instructionLayout = new QHBoxLayout;
    instructionLayout->setContentsMargins(0, 0, 0, 0);
    instructionLayout->addWidget(m_instruction, 1);
    m_sidebarToggle = new QToolButton(central);
    m_sidebarToggle->setObjectName(QStringLiteral("AnnotationSidebarToggle"));
    m_sidebarToggle->setIcon(
            QIcon(iconPath(QStringLiteral("panel-right.svg"))));
    m_sidebarToggle->setIconSize(QSize(18, 18));
    m_sidebarToggle->setFixedSize(26, 26);
    m_sidebarToggle->setFocusPolicy(Qt::NoFocus);
    m_sidebarToggle->setToolTip(tr("Toggle right panel"));
    m_sidebarToggle->setAccessibleName(m_sidebarToggle->toolTip());
    instructionLayout->addWidget(m_sidebarToggle);
    centralLayout->addLayout(instructionLayout);
    centralLayout->addSpacing(5);
    m_canvas = new AnnotationCanvas(central);
    m_canvas->setAcceptDrops(true);
    m_canvas->installEventFilter(this);
    m_canvasScrollArea = new QScrollArea(central);
    m_canvasScrollArea->setObjectName(
            QStringLiteral("AnnotationCanvasScrollArea"));
    m_canvasScrollArea->setFrameShape(QFrame::NoFrame);
    m_canvasScrollArea->setWidgetResizable(true);
    m_canvasScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_canvasScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_canvasScrollArea->setWidget(m_canvas);
    centralLayout->addWidget(m_canvasScrollArea, 1);

    m_adjustmentPanel = new QFrame(m_canvas);
    m_adjustmentPanel->setObjectName(QStringLiteral("canvas_adjustment"));
    m_adjustmentPanel->setAttribute(Qt::WA_StyledBackground);
    m_adjustmentPanel->setStyleSheet(QStringLiteral(
            "#canvas_adjustment { background: rgba(255,255,255,220); "
            "border-radius: 6px; }"));
    auto* adjustmentLayout = new QVBoxLayout(m_adjustmentPanel);
    adjustmentLayout->setContentsMargins(10, 8, 10, 8);
    adjustmentLayout->setSpacing(5);
    auto* adjustmentHeader = new QHBoxLayout;
    auto* adjustmentTitle = new QLabel(tr("Canvas Display"), m_adjustmentPanel);
    QFont titleFont = adjustmentTitle->font();
    titleFont.setBold(true);
    adjustmentTitle->setFont(titleFont);
    adjustmentHeader->addWidget(adjustmentTitle);
    adjustmentHeader->addStretch();
    m_adjustmentToggle = new QToolButton(m_adjustmentPanel);
    m_adjustmentToggle->setIcon(
            QIcon(iconPath(QStringLiteral("caret-up.svg"))));
    m_adjustmentToggle->setToolTip(tr("Collapse adjustments"));
    m_adjustmentToggle->setFixedSize(20, 20);
    m_adjustmentToggle->setFocusPolicy(Qt::NoFocus);
    adjustmentHeader->addWidget(m_adjustmentToggle);
    adjustmentLayout->addLayout(adjustmentHeader);
    auto* adjustmentContent = new QWidget(m_adjustmentPanel);
    auto* contentLayout = new QVBoxLayout(adjustmentContent);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(5);
    auto addAdjustment = [this, contentLayout](const QString& name, int maximum,
                                               int initial,
                                               QLabel*& valueLabel) {
        auto* row = new QHBoxLayout;
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(4);
        auto* nameLabel = new QLabel(name, m_adjustmentPanel);
        nameLabel->setFixedWidth(64);
        auto* slider = new QSlider(Qt::Horizontal, m_adjustmentPanel);
        slider->setRange(0, maximum);
        slider->setValue(initial);
        slider->setFixedWidth(120);
        slider->setFocusPolicy(Qt::NoFocus);
        valueLabel = new QLabel(m_adjustmentPanel);
        valueLabel->setFixedWidth(38);
        valueLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        auto* reset = new QToolButton(m_adjustmentPanel);
        reset->setIcon(QIcon(iconPath(QStringLiteral("refresh.svg"))));
        reset->setToolTip(tr("Reset to default"));
        reset->setFixedSize(22, 20);
        reset->setFocusPolicy(Qt::NoFocus);
        connect(reset, &QToolButton::clicked, slider,
                [slider, initial]() { slider->setValue(initial); });
        row->addWidget(nameLabel);
        row->addWidget(slider);
        row->addWidget(valueLabel);
        row->addWidget(reset);
        contentLayout->addLayout(row);
        return slider;
    };
    m_opacitySlider = addAdjustment(tr("Opacity"), 100, 100, m_opacityValue);
    m_brightnessSlider =
            addAdjustment(tr("Brightness"), 150, 50, m_brightnessValue);
    m_contrastSlider = addAdjustment(tr("Contrast"), 150, 50, m_contrastValue);
    m_opacitySlider->setObjectName(QStringLiteral("AnnotationOpacity"));
    m_brightnessSlider->setObjectName(QStringLiteral("AnnotationBrightness"));
    m_contrastSlider->setObjectName(QStringLiteral("AnnotationContrast"));
    adjustmentLayout->addWidget(adjustmentContent);
    m_opacityValue->setText(QStringLiteral("100%"));
    m_brightnessValue->setText(QStringLiteral("1.00"));
    m_contrastValue->setText(QStringLiteral("1.00"));
    connect(m_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        m_opacityValue->setText(QStringLiteral("%1%").arg(value));
        m_canvas->setShapeOpacity(value);
    });
    m_canvas->setMaskOpacity(m_defaultMaskOpacity);
    m_canvas->setShapeOpacity(m_defaultShapeOpacity);
    m_canvas->setMagicWandOptions(m_magicDefaultThreshold,
                                  m_magicDragSensitivity,
                                  m_magicLuminanceWeight);
    m_canvas->setMagicWandOpacity(m_magicOpacity);
    m_canvas->setMagicWandSimplifyEpsilon(m_magicSimplifyEpsilon);
    m_canvas->setShapeStyle(m_shapePointSize, m_shapeLineWidth);
    m_canvas->setHitEpsilon(m_canvasEpsilon);
    m_canvas->setCrosshair(m_crosshairVisible, m_crosshairWidth,
                           m_crosshairColor, m_crosshairOpacity);
    m_canvas->setDoubleClickClose(m_doubleClickClose);
    m_canvas->setDoubleClickEditLabel(m_doubleClickEditLabel);
    m_canvas->setWheelRectangleEditing(m_wheelRectangleEditing,
                                       m_wheelAdjustStep, m_wheelScaleStep);
    m_canvas->setBrushPointDistance(m_brushPointDistance);
    m_canvas->setBrushHistoryLimits(m_brushMaxUndoSteps,
                                    m_brushMaxUndoMemoryMb);
    m_canvas->setRotationIncrements(m_rotationLargeIncrement,
                                    m_rotationSmallIncrement);
    m_canvas->setCuboidDefaults(QPointF(24.0, -24.0), m_cuboidMinDepth);
    m_adjustmentTimer = new QTimer(this);
    m_adjustmentTimer->setSingleShot(true);
    m_adjustmentTimer->setInterval(33);
    connect(m_adjustmentTimer, &QTimer::timeout, this, [this]() {
        m_canvas->setBrightnessContrast(m_brightnessSlider->value(),
                                        m_contrastSlider->value());
    });
    connect(m_brightnessSlider, &QSlider::valueChanged, this,
            [this](int value) {
                m_brightnessValue->setText(
                        QString::number(value / 50.0, 'f', 2));
                if (!m_adjustmentTimer->isActive()) m_adjustmentTimer->start();
                if (!m_imagePath.isEmpty())
                    m_displayAdjustments.insert(
                            m_imagePath, qMakePair(m_brightnessSlider->value(),
                                                   m_contrastSlider->value()));
            });
    connect(m_contrastSlider, &QSlider::valueChanged, this, [this](int value) {
        m_contrastValue->setText(QString::number(value / 50.0, 'f', 2));
        if (!m_adjustmentTimer->isActive()) m_adjustmentTimer->start();
        if (!m_imagePath.isEmpty())
            m_displayAdjustments.insert(m_imagePath,
                                        qMakePair(m_brightnessSlider->value(),
                                                  m_contrastSlider->value()));
    });
    connect(m_adjustmentToggle, &QToolButton::clicked, this,
            [this, adjustmentContent, adjustmentTitle]() {
                const bool show = !adjustmentContent->isVisible();
                adjustmentContent->setVisible(show);
                adjustmentTitle->setVisible(show);
                m_adjustmentToggle->setIcon(QIcon(
                        iconPath(show ? QStringLiteral("caret-up.svg")
                                      : QStringLiteral("caret-down.svg"))));
                m_adjustmentToggle->setToolTip(show ? tr("Collapse adjustments")
                                                    : tr("Expand adjustments"));
                m_adjustmentPanel->adjustSize();
                m_adjustmentPanel->move(
                        10,
                        std::max(0, m_canvas->height() -
                                            m_adjustmentPanel->height() - 10));
            });
    m_adjustmentPanel->adjustSize();
    m_adjustmentPanel->hide();

    m_navigator = new AnnotationNavigator(this);
    m_navigatorDock = new QDockWidget(tr("Navigator"), this);
    m_navigatorDock->setObjectName(QStringLiteral("Navigator"));
    m_navigatorDock->setWidget(m_navigator);
    m_navigatorDock->setAllowedAreas(Qt::LeftDockWidgetArea |
                                     Qt::RightDockWidgetArea |
                                     Qt::BottomDockWidgetArea);
    addDockWidget(Qt::LeftDockWidgetArea, m_navigatorDock);
    m_navigatorDock->hide();

    m_compareBar = addToolBar(tr("Compare View"));
    m_compareBar->setObjectName(QStringLiteral("CompareToolbar"));
    m_compareSlider = new QSlider(Qt::Horizontal, m_compareBar);
    m_compareSlider->setRange(0, 100);
    m_compareSlider->setValue(50);
    m_compareSlider->setToolTip(tr("Reference image split"));
    m_compareValue = new QLabel(tr("50%"), m_compareBar);
    m_compareValue->setMinimumWidth(42);
    m_compareBar->addWidget(new QLabel(tr("Reference"), m_compareBar));
    m_compareBar->addWidget(m_compareSlider);
    m_compareBar->addWidget(m_compareValue);
    m_compareBar->setVisible(false);

    auto* filePanel = new QWidget(this);
    auto* fileLayout = new QVBoxLayout(filePanel);
    fileLayout->setContentsMargins(4, 4, 4, 4);
    m_fileFilter = new QLineEdit(filePanel);
    m_fileFilter->setPlaceholderText(tr("Search files..."));
    auto* fileSearchRow = new QHBoxLayout;
    fileSearchRow->setContentsMargins(0, 0, 0, 0);
    auto* settingsButton = new QToolButton(filePanel);
    settingsButton->setObjectName(QStringLiteral("AnnotationSettingsButton"));
    settingsButton->setIcon(QIcon(iconPath(QStringLiteral("settings.svg"))));
    settingsButton->setToolTip(tr("Settings"));
    settingsButton->setIconSize(QSize(28, 28));
    settingsButton->setFixedSize(32, 32);
    connect(settingsButton, &QToolButton::clicked, this,
            &AnnotationWindow::openSettings);
    fileSearchRow->addWidget(m_fileFilter);
    fileSearchRow->addWidget(settingsButton);
    m_files = new QListWidget(filePanel);
    m_files->setObjectName(QStringLiteral("AnnotationFiles"));
    m_files->setIconSize(QSize(12, 12));
    fileLayout->addLayout(fileSearchRow);
    fileLayout->addWidget(m_files);
    auto* filesDock = new QDockWidget(tr("Files"), this);
    filesDock->setObjectName(QStringLiteral("Files"));
    filesDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    filesDock->setWidget(filePanel);

    auto* descriptionPanel = new QWidget(this);
    auto* descriptionLayout = new QVBoxLayout(descriptionPanel);
    descriptionLayout->setContentsMargins(4, 4, 4, 4);
    m_descriptionTitle = new QLabel(tr("Image Description"), descriptionPanel);
    m_description = new QPlainTextEdit(descriptionPanel);
    m_description->setObjectName(QStringLiteral("AnnotationDescription"));
    m_description->setPlaceholderText(tr("Description"));
    m_description->setEnabled(false);
    descriptionLayout->addWidget(m_descriptionTitle);
    descriptionLayout->addWidget(m_description);
    auto* descriptionDock = new QDockWidget(tr("Description"), this);
    descriptionDock->setObjectName(QStringLiteral("Description"));
    descriptionDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    descriptionDock->setWidget(descriptionPanel);
    descriptionLayout->removeWidget(m_descriptionTitle);
    m_descriptionTitle->setParent(nullptr);
    connect(m_description, &QPlainTextEdit::textChanged, this, [this]() {
        if (m_updating || m_imagePath.isEmpty()) return;
        const QSet<int> selected = m_canvas->selectedIndices();
        const int selectedIndex =
                selected.size() == 1 ? *selected.constBegin() : -1;
        if (selectedIndex >= 0 && selectedIndex < m_canvas->shapes().size()) {
            m_canvas->setShapeDescription(selectedIndex,
                                          m_description->toPlainText());
        } else {
            m_document.insert(QStringLiteral("description"),
                              m_description->toPlainText());
        }
        setDirty(true);
    });

    m_objects = new QListWidget(this);
    m_objects->setObjectName(QStringLiteral("AnnotationObjects"));
    m_objects->setSelectionMode(QAbstractItemView::ExtendedSelection);
    auto* objectsDock = new QDockWidget(tr("Objects"), this);
    objectsDock->setObjectName(QStringLiteral("Objects"));
    objectsDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    objectsDock->setWidget(m_objects);

    m_labels = new QListWidget(this);
    auto* labelsDock = new QDockWidget(tr("Labels"), this);
    labelsDock->setObjectName(QStringLiteral("Labels"));
    labelsDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    labelsDock->setWidget(m_labels);

    m_modelPanel = new QWidget(this);
    m_modelPanel->setObjectName(QStringLiteral("AIModel"));
    auto* modelPanel = m_modelPanel;
    auto* flagsPanel = new QWidget(this);
    auto* flagsLayout = new QVBoxLayout(flagsPanel);
    flagsLayout->setContentsMargins(4, 4, 4, 4);
    flagsLayout->setSpacing(3);
    m_flags = new QListWidget(flagsPanel);
    m_flags->setObjectName(QStringLiteral("AnnotationImageFlags"));
    flagsLayout->addWidget(m_flags);
    m_flagsDock = new QDockWidget(tr("Flags"), this);
    m_flagsDock->setObjectName(QStringLiteral("ImageFlags"));
    m_flagsDock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    m_flagsDock->setWidget(flagsPanel);

    // The upstream application presents these panels as a single sidebar
    // inside the labeling workspace. Keep QDockWidget instances for the
    // existing API/tests, but use them as ordinary embedded panels here.
    m_rightSidebar = new QWidget(m_workspace);
    m_rightSidebar->setObjectName(QStringLiteral("AnnotationRightSidebar"));
    m_rightSidebarLayout = new QVBoxLayout(m_rightSidebar);
    m_rightSidebarLayout->setContentsMargins(0, 0, 0, 0);
    m_rightSidebarLayout->setSpacing(4);
    auto addSidebarSection = [this](const QString& title, QDockWidget* dock,
                                    bool checkable, int stretch,
                                    QLabel* titleLabel = nullptr,
                                    bool showHeader = true) {
        auto* section = new QFrame(m_rightSidebar);
        section->setObjectName(QStringLiteral("AnnotationSidebarSection"));
        section->setFrameShape(QFrame::NoFrame);
        auto* sectionLayout = new QVBoxLayout(section);
        sectionLayout->setContentsMargins(0, 0, 0, 0);
        sectionLayout->setSpacing(0);

        if (showHeader) {
            auto* header = new QWidget(section);
            auto* headerLayout = new QHBoxLayout(header);
            headerLayout->setContentsMargins(0, 2, 0, 2);
            headerLayout->addStretch();
            auto* label = titleLabel ? titleLabel : new QLabel(title, header);
            if (titleLabel) {
                titleLabel->setParent(header);
                titleLabel->setText(title);
            }
            label->setAlignment(Qt::AlignCenter);
            headerLayout->addWidget(label);
            headerLayout->addStretch();
            if (checkable) {
                auto* visibility = new QCheckBox(header);
                visibility->setObjectName(
                        QStringLiteral("Annotation%1Visibility")
                                .arg(dock->objectName()));
                visibility->setChecked(true);
                visibility->setToolTip(tr("Show %1 panel").arg(title));
                headerLayout->addWidget(visibility);
                connect(visibility, &QCheckBox::toggled, dock,
                        &QWidget::setVisible);
            }
            sectionLayout->addWidget(header);
        }
        auto* emptyTitleBar = new QWidget(dock);
        emptyTitleBar->setFixedHeight(0);
        dock->setTitleBarWidget(emptyTitleBar);
        sectionLayout->addWidget(dock);
        m_rightSidebarLayout->addWidget(section, stretch);
        return section;
    };
    addSidebarSection(tr("Description"), descriptionDock, true, 0,
                      m_descriptionTitle);
    // The source embeds the flag dock directly and hides the whole dock when
    // no template is configured; there is no persistent empty "Flags" header.
    addSidebarSection(tr("Flags"), m_flagsDock, false, 0, nullptr, false);
    m_flagsDock->hide();
    addSidebarSection(tr("Labels"), labelsDock, true, 0);
    addSidebarSection(tr("Objects"), objectsDock, true, 0);
    // Upstream places the file search row directly after Objects without a
    // separate Files heading.
    addSidebarSection(tr("Files"), filesDock, false, 1, nullptr, false);

    m_workspaceSplitter = new QSplitter(Qt::Horizontal, m_workspace);
    m_workspaceSplitter->setObjectName(
            QStringLiteral("AnnotationWorkspaceSplitter"));
    m_workspaceSplitter->setChildrenCollapsible(false);
    m_workspaceSplitter->addWidget(central);
    m_workspaceSplitter->addWidget(m_rightSidebar);
    m_workspaceSplitter->setStretchFactor(0, 1);
    m_workspaceSplitter->setStretchFactor(1, 0);
    m_rightSidebar->setMinimumWidth(260);
    m_rightSidebar->setMaximumWidth(560);
    m_workspaceSplitter->setSizes({800, 320});
    connect(m_sidebarToggle, &QToolButton::clicked, this, [this]() {
        if (!m_rightSidebar || !m_workspaceSplitter) return;
        const bool show = m_rightSidebar->isHidden();
        if (!show) m_rightSidebarWidth = m_rightSidebar->width();
        m_rightSidebar->setVisible(show);
        if (show) {
            const int available = m_workspaceSplitter->width() -
                                  m_workspaceSplitter->handleWidth();
            m_workspaceSplitter->setSizes(
                    {qMax(1, available - m_rightSidebarWidth),
                     m_rightSidebarWidth});
        }
        m_sidebarToggle->setIcon(
                QIcon(iconPath(show ? QStringLiteral("panel-right.svg")
                                    : QStringLiteral("panel-left.svg"))));
    });
    workspaceLayout->addWidget(m_workspaceSplitter, 1);
    setCentralWidget(m_workspace);

    auto* tagsPanel = new QWidget(central);
    tagsPanel->setObjectName(QStringLiteral("ImageTagsPanel"));
    auto* tagsLayout = new QVBoxLayout(tagsPanel);
    tagsLayout->setContentsMargins(5, 5, 5, 5);
    tagsLayout->setSpacing(4);
    m_tags = new QListWidget(tagsPanel);
    m_tags->setObjectName(QStringLiteral("ImageTags"));
    m_tags->setSelectionMode(QAbstractItemView::SingleSelection);
    m_tags->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tags->setFlow(QListView::LeftToRight);
    m_tags->setWrapping(true);
    m_tags->setResizeMode(QListView::Adjust);
    m_tags->setStyleSheet(QStringLiteral(
            "QListWidget { border: 0; background: transparent; } "
            "QListWidget::item { border-radius: 14px; padding: 3px 9px; } "
            "QListWidget::item:selected { border: 2px solid #202020; }"));
    m_tags->setDragDropMode(QAbstractItemView::InternalMove);
    m_tags->setDefaultDropAction(Qt::MoveAction);
    m_tags->setMinimumHeight(32);
    m_tags->setMaximumHeight(92);
    m_tagInput = new QLineEdit(tagsPanel);
    m_tagInput->setObjectName(QStringLiteral("ImageTagInput"));
    m_tagInput->setPlaceholderText(tr("New image tag"));
    m_tagInput->installEventFilter(this);
    m_tagInput->hide();
    auto* tagButtons = new QHBoxLayout;
    auto* addTag = new QPushButton(tr("Add"), tagsPanel);
    auto* editTag = new QPushButton(tr("Edit"), tagsPanel);
    auto* deleteTag = new QPushButton(tr("Delete"), tagsPanel);
    auto* copyTags = new QPushButton(tr("Copy all"), tagsPanel);
    auto* batchTag = new QPushButton(QStringLiteral("-"), tagsPanel);
    addTag->setObjectName(QStringLiteral("AddImageTag"));
    editTag->setObjectName(QStringLiteral("EditImageTag"));
    deleteTag->setObjectName(QStringLiteral("DeleteImageTags"));
    copyTags->setObjectName(QStringLiteral("CopyImageTags"));
    batchTag->setObjectName(QStringLiteral("BatchDeleteImageTags"));
    batchTag->setToolTip(tr("Batch delete image tags"));
    batchTag->setAccessibleName(tr("Batch delete image tags"));
    for (QPushButton* button : {addTag, editTag, deleteTag, copyTags}) {
        button->setAutoDefault(false);
        button->setMinimumHeight(28);
    }
    batchTag->setAutoDefault(false);
    batchTag->setFixedSize(28, 28);
    batchTag->setStyleSheet(QStringLiteral(
            "QPushButton { border: none; border-radius: 14px; "
            "background: rgba(128,128,128,35); font-size: 18px; "
            "font-weight: bold; padding: 0; }"
            "QPushButton:hover { background: rgba(128,128,128,65); }"
            "QPushButton:pressed { background: rgba(128,128,128,90); }"));
    tagButtons->addWidget(addTag);
    tagButtons->addWidget(editTag);
    tagButtons->addWidget(deleteTag);
    tagButtons->addWidget(copyTags);
    tagButtons->addWidget(batchTag);
    tagsLayout->addWidget(m_tags);
    tagsLayout->addWidget(m_tagInput);
    tagsLayout->addLayout(tagButtons);
    m_tagsPanel = tagsPanel;
    m_tagsPanel->setMaximumHeight(140);
    centralLayout->addWidget(m_tagsPanel);
    m_tagsPanel->hide();
    auto addTagValue = [this]() {
        if (m_imagePath.isEmpty()) return;
        if (m_tagInput->isHidden()) {
            m_tagInput->setText(nextImageTag(m_tags));
            m_tagInput->show();
            m_tagInput->setFocus();
            m_tagInput->selectAll();
            return;
        }
        commitTagInput();
    };
    addTag->setFocusPolicy(Qt::NoFocus);
    connect(addTag, &QPushButton::clicked, this, addTagValue);
    connect(m_tagInput, &QLineEdit::returnPressed, this,
            &AnnotationWindow::commitTagInput);
    connect(editTag, &QPushButton::clicked, this, [this]() {
        QListWidgetItem* item = m_tags->currentItem();
        if (!item || m_tagBatchMode) return;
        item->setFlags(item->flags() | Qt::ItemIsEditable);
        m_tags->editItem(item);
    });
    connect(m_tags, &QListWidget::itemDoubleClicked, editTag,
            &QPushButton::click);
    connect(m_tags, &QListWidget::itemChanged, this,
            [this](QListWidgetItem* item) {
                const QString original = item->data(Qt::UserRole).toString();
                const QString tag = normalizeImageTag(item->text());
                bool duplicate = false;
                for (int i = 0; i < m_tags->count(); ++i)
                    duplicate |= m_tags->item(i) != item &&
                                 m_tags->item(i)->text() == tag;
                if (tag.isEmpty() || duplicate) {
                    const QSignalBlocker blocked(m_tags);
                    item->setText(original);
                    return;
                }
                if (tag == original) return;
                {
                    const QSignalBlocker blocked(m_tags);
                    item->setText(tag);
                    item->setData(Qt::UserRole, tag);
                }
                syncTagsToDocument();
            });
    connect(deleteTag, &QPushButton::clicked, this, [this]() {
        const QList<QListWidgetItem*> selected = m_tags->selectedItems();
        if (selected.isEmpty()) return;
        if (QMessageBox::question(
                    this, tr("Delete image tags"),
                    tr("Delete %1 selected image tag(s)?").arg(selected.size()),
                    QMessageBox::Yes | QMessageBox::Cancel,
                    QMessageBox::Cancel) != QMessageBox::Yes)
            return;
        for (QListWidgetItem* item : selected) delete item;
        syncTagsToDocument();
    });
    connect(copyTags, &QPushButton::clicked, this, [this]() {
        QStringList values;
        for (int i = 0; i < m_tags->count(); ++i)
            values.append(m_tags->item(i)->text());
        QApplication::clipboard()->setText(values.join(QStringLiteral(",")));
    });
    connect(batchTag, &QPushButton::clicked, this,
            [this]() { setTagBatchMode(!m_tagBatchMode); });
    connect(m_tags, &QListWidget::itemSelectionChanged, this, [this]() {
        if (!m_tagBatchMode) return;
        setTagBatchMode(true);
    });
    auto persistTagOrder = [this]() {
        const quint64 generation = m_generation;
        QTimer::singleShot(0, this, [this, generation]() {
            if (generation == m_generation) syncTagsToDocument();
        });
    };
    connect(m_tags->model(), &QAbstractItemModel::rowsMoved, this,
            [persistTagOrder](const QModelIndex&, int, int, const QModelIndex&,
                              int) { persistTagOrder(); });
    connect(m_tags->model(), &QAbstractItemModel::rowsInserted, this,
            [persistTagOrder](const QModelIndex&, int, int) {
                persistTagOrder();
            });
    connect(m_tags->model(), &QAbstractItemModel::rowsRemoved, this,
            [persistTagOrder](const QModelIndex&, int, int) {
                persistTagOrder();
            });
    m_tags->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_tags, &QListWidget::customContextMenuRequested, this,
            [this, editTag, deleteTag](const QPoint& position) {
                QMenu menu(m_tags);
                QAction* edit = menu.addAction(tr("Edit"));
                QAction* selectAll = menu.addAction(tr("Select all"));
                QAction* remove = menu.addAction(tr("Delete selected"));
                edit->setEnabled(m_tags->currentItem() != nullptr &&
                                 !m_tagBatchMode);
                remove->setEnabled(!m_tags->selectedItems().isEmpty());
                QAction* chosen = menu.exec(m_tags->mapToGlobal(position));
                if (chosen == edit) editTag->click();
                if (chosen == selectAll) {
                    setTagBatchMode(true);
                    m_tags->selectAll();
                }
                if (chosen == remove) deleteTag->click();
            });

    auto* modelLayout = new QVBoxLayout(modelPanel);
    modelLayout->setContentsMargins(6, 4, 6, 4);
    auto* modelRow = new QHBoxLayout;
    auto* promptRow = new QHBoxLayout;
    auto* sliceRow = new QHBoxLayout;
    m_model = new QComboBox(modelPanel);
    m_model->setObjectName(QStringLiteral("AnnotationModelFamily"));
    m_quant = new QComboBox(modelPanel);
    m_quant->setObjectName(QStringLiteral("AnnotationQuantization"));
    m_backend = new QComboBox(modelPanel);
    m_backend->setObjectName(QStringLiteral("AnnotationBackend"));
    m_textModel = new QComboBox(modelPanel);
    m_textModel->setObjectName(QStringLiteral("AnnotationTextEncoder"));
    m_textModel->setToolTip(tr("Text encoder and quantization"));
    m_textModel->setVisible(false);
    m_faceLandmark = new QComboBox(modelPanel);
    m_faceLandmark->setObjectName(QStringLiteral("FaceLandmarkModel"));
    m_faceLandmark->setToolTip(tr("Face landmark model"));
    m_faceLandmark->setVisible(false);
    m_samText = new QLineEdit(modelPanel);
    m_samText->setObjectName(QStringLiteral("SamTextPrompt"));
    m_samText->setPlaceholderText(tr("SAM 3 text prompt"));
    m_samText->setClearButtonEnabled(true);
    m_samText->setMinimumWidth(180);
    m_samText->setVisible(false);
    m_gkdSupportButton = new QPushButton(tr("Use selected points"), modelPanel);
    m_gkdSupportButton->setObjectName(QStringLiteral("GkdVisualSupport"));
    m_gkdSupportButton->setToolTip(
            tr("Use selected point annotations on this image as a reusable GKD "
               "support example"));
    m_gkdSupportButton->setVisible(false);
    m_confidence = new QDoubleSpinBox(modelPanel);
    m_model->setToolTip(tr("Model family"));
    m_quant->setToolTip(tr("Weight quantization"));
    m_backend->setToolTip(tr("Inference backend"));
    m_confidence->setToolTip(tr("Confidence threshold"));
    m_confidence->setRange(0.01, 1.0);
    m_confidence->setSingleStep(0.05);
    m_confidence->setValue(0.25);
    m_confidence->setDecimals(2);
    m_slicedDetection = new QCheckBox(tr("Sliced"), modelPanel);
    m_slicedDetection->setObjectName(QStringLiteral("SlicedDetection"));
    m_slicedDetection->setToolTip(tr("SAHI tiled detection for small objects"));
    m_sliceWidth = new QSpinBox(modelPanel);
    m_sliceHeight = new QSpinBox(modelPanel);
    for (QSpinBox* spin : {m_sliceWidth, m_sliceHeight}) {
        spin->setRange(128, 4096);
        spin->setSingleStep(32);
        spin->setValue(512);
        spin->setSuffix(tr(" px"));
        spin->setVisible(false);
    }
    m_sliceWidth->setToolTip(tr("SAHI slice width"));
    m_sliceHeight->setToolTip(tr("SAHI slice height"));
    m_sliceOverlap = new QDoubleSpinBox(modelPanel);
    m_sliceOverlap->setRange(0.0, 0.9);
    m_sliceOverlap->setSingleStep(0.05);
    m_sliceOverlap->setValue(0.2);
    m_sliceOverlap->setDecimals(2);
    m_sliceOverlap->setToolTip(tr("SAHI overlap ratio"));
    m_sliceOverlap->setVisible(false);
    m_sliceMergeThreshold = new QDoubleSpinBox(modelPanel);
    m_sliceMergeThreshold->setRange(0.01, 1.0);
    m_sliceMergeThreshold->setSingleStep(0.05);
    m_sliceMergeThreshold->setValue(0.45);
    m_sliceMergeThreshold->setDecimals(2);
    m_sliceMergeThreshold->setToolTip(tr("SAHI IOS merge threshold"));
    m_sliceMergeThreshold->setVisible(false);
    auto* runButton = new QPushButton(tr("Run model"), modelPanel);
    runButton->setIcon(QIcon(iconPath(QStringLiteral("auto-run.png"))));
    runButton->setToolTip(
            tr("Selected boxes become SAM3 prompts when no point, box, or text "
               "prompt is set"));
    auto* skipDetButton = new QPushButton(tr("Skip Detection"), modelPanel);
    skipDetButton->setToolTip(
            tr("Re-recognize the selected text regions without re-running text "
               "detection"));
    modelRow->addWidget(m_model, 3);
    modelRow->addWidget(m_quant, 1);
    modelRow->addWidget(m_backend, 1);
    promptRow->addWidget(m_textModel, 2);
    promptRow->addWidget(m_faceLandmark, 2);
    promptRow->addWidget(m_samText, 3);
    promptRow->addWidget(m_gkdSupportButton);
    promptRow->addWidget(m_confidence);
    promptRow->addWidget(m_slicedDetection);
    promptRow->addWidget(runButton);
    promptRow->addWidget(skipDetButton);
    sliceRow->addWidget(m_sliceWidth);
    sliceRow->addWidget(m_sliceHeight);
    sliceRow->addWidget(m_sliceOverlap);
    sliceRow->addWidget(m_sliceMergeThreshold);
    modelLayout->addLayout(modelRow);
    modelLayout->addLayout(promptRow);
    modelLayout->addLayout(sliceRow);
    // The upstream workbench embeds the auto-labeling controls directly
    // between the instruction row and the canvas as a plain child widget:
    // no dock chrome, no floating, no title bar.  The "AIModel" object name
    // set above remains the visible/test handle.
    m_modelPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    const int canvasIndex = centralLayout->indexOf(m_canvasScrollArea);
    centralLayout->insertWidget(
            canvasIndex < 0 ? centralLayout->count() : canvasIndex,
            m_modelPanel);
    m_modelPanel->hide();

    m_checked = new QCheckBox(tr("Checked"), this);
    m_status = new QLabel(tr("Open an image or folder"), this);
    statusBar()->addWidget(m_status, 1);
    statusBar()->addPermanentWidget(m_checked);

    connect(m_fileFilter, &QLineEdit::textChanged, this,
            [this](const QString& text) {
                for (int i = 0; i < m_files->count(); ++i)
                    m_files->item(i)->setHidden(
                            !m_files->item(i)->text().contains(
                                    text, Qt::CaseInsensitive));
            });
    connect(m_files, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* item) {
                if (item && item->data(Qt::UserRole).toString() != m_imagePath)
                    openFile(item->data(Qt::UserRole).toString());
            });
    connect(m_objects, &QListWidget::itemSelectionChanged, this, [this]() {
        QVector<int> rows;
        for (QListWidgetItem* item : m_objects->selectedItems())
            rows.append(m_objects->row(item));
        m_canvas->setSelectedShapes(rows);
    });
    connect(m_objects, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { editSelectedLabel(); });
    m_objects->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_objects, &QListWidget::customContextMenuRequested, this,
            [this](const QPoint& position) {
                QListWidgetItem* item = m_objects->itemAt(position);
                if (!item) return;
                if (!item->isSelected()) {
                    m_objects->clearSelection();
                    item->setSelected(true);
                }
                const int index = m_objects->row(item);
                if (index < 0 || index >= m_canvas->shapes().size()) return;
                const AnnotationShape current = m_canvas->shapes()[index];
                QMenu menu(this);
                QAction* visibility = menu.addAction(
                        current.visible ? tr("Hide") : tr("Show"));
                QAction* lock = menu.addAction(current.locked ? tr("Unlock")
                                                              : tr("Lock"));
                QAction* remove = menu.addAction(tr("Delete"));
                menu.addSeparator();
                QAction* group = menu.addAction(tr("Set group ID"));
                QAction* ungroup = menu.addAction(tr("Ungroup"));
                remove->setEnabled(!current.locked);
                QAction* chosen = menu.exec(m_objects->mapToGlobal(position));
                if (!chosen) return;
                if (chosen == group || chosen == ungroup) {
                    int groupId = 0;
                    if (chosen == group) {
                        bool accepted = false;
                        groupId = QInputDialog::getInt(
                                this, tr("Set group ID"), tr("Group ID"),
                                current.extra.value(QStringLiteral("group_id"))
                                        .toInt(),
                                0, std::numeric_limits<int>::max(), 1,
                                &accepted);
                        if (!accepted) return;
                    }
                    snapshot();
                    QVector<AnnotationShape> shapes = m_canvas->shapes();
                    for (QListWidgetItem* selected :
                         m_objects->selectedItems()) {
                        const int row = m_objects->row(selected);
                        if (row < 0 || row >= shapes.size() ||
                            shapes[row].locked)
                            continue;
                        shapes[row].extra.insert(QStringLiteral("group_id"),
                                                 chosen == group
                                                         ? QJsonValue(groupId)
                                                         : QJsonValue::Null);
                    }
                    m_canvas->setShapes(shapes);
                    updateObjects();
                    setDirty(true);
                    return;
                }
                m_canvas->selectShape(index);
                if (chosen == remove) {
                    m_canvas->deleteSelected();
                    return;
                }
                snapshot();
                QVector<AnnotationShape> shapes = m_canvas->shapes();
                if (chosen == visibility)
                    shapes[index].visible = !shapes[index].visible;
                else if (chosen == lock)
                    shapes[index].locked = !shapes[index].locked;
                m_canvas->setShapes(shapes);
                m_canvas->selectShape(index);
                updateObjects();
                setDirty(true);
            });
    connect(m_canvas, &AnnotationCanvas::shapeSelected, this,
            [this](int index) {
                ++m_promptRevision;
                const QSignalBlocker blocked(m_objects);
                m_objects->clearSelection();
                for (int row : m_canvas->selectedIndices())
                    if (row >= 0 && row < m_objects->count())
                        m_objects->item(row)->setSelected(true);
                m_objects->setCurrentRow(index, QItemSelectionModel::NoUpdate);
                if (m_brushTarget && index >= 0 &&
                    index < m_canvas->shapes().size()) {
                    const QString label = m_canvas->shapes()[index].label;
                    const QSignalBlocker blockedTarget(m_brushTarget);
                    m_brushTarget->setEditText(label);
                    m_canvas->setBrushLabel(label);
                    m_canvas->setBrushClassId(
                            m_canvas->shapes()[index]
                                    .extra.value(QStringLiteral("class_id"))
                                    .toInt(-1));
                }
                updateDescriptionEditor();
            });
    connect(m_canvas, &AnnotationCanvas::promptChanged, this, [this]() {
        ++m_promptRevision;
        ++m_generation;
    });
    connect(m_canvas, &AnnotationCanvas::viewChanged, this,
            &AnnotationWindow::updateNavigator);
    connect(m_compareSlider, &QSlider::valueChanged, this, [this](int value) {
        m_compareValue->setText(tr("%1%").arg(value));
        m_canvas->setCompareSplit(value / 100.0);
    });
    connect(m_navigator, &AnnotationNavigator::navigationRequested, m_canvas,
            &AnnotationCanvas::navigateTo);
    connect(m_canvas, &AnnotationCanvas::editStarted, this,
            &AnnotationWindow::snapshot);
    connect(m_canvas, &AnnotationCanvas::creationCanceled, this, [this]() {
        if (!m_undo.isEmpty()) m_undo.removeLast();
        m_undoAction->setEnabled(!m_undo.isEmpty());
    });
    connect(m_canvas, &AnnotationCanvas::shapesChanged, this, [this]() {
        if (m_ignoreNextChange) {
            m_ignoreNextChange = false;
            updateObjects();
            return;
        }
        setDirty(true);
        updateObjects();
    });
    connect(m_canvas, &AnnotationCanvas::shapeFinished, this,
            [this](int index) {
                bool accepted = true;
                QString label;
                QJsonObject chosenFlags;
                if (m_autoUseLastLabel &&
                    !m_canvas->currentLabel().trimmed().isEmpty()) {
                    label = m_canvas->currentLabel().trimmed();
                } else if (!m_displayLabelPopup) {
                    // The source keeps the drawn shape unlabeled-optional when
                    // the label popup is disabled.
                    label = m_canvas->currentLabel().trimmed();
                } else {
                    label = openLabelDialog(QString(), &accepted, &chosenFlags);
                }
                if (accepted && !label.isEmpty() && index >= 0 &&
                    index < m_canvas->shapes().size()) {
                    QVector<AnnotationShape> shapes = m_canvas->shapes();
                    shapes[index].label = label;
                    if (!chosenFlags.isEmpty())
                        shapes[index].extra.insert(QStringLiteral("flags"),
                                                   chosenFlags);
                    m_canvas->setShapes(shapes);
                    m_canvas->setCurrentLabel(label);
                    updateObjects();
                } else {
                    m_canvas->discardLastShape();
                    if (!m_undo.isEmpty()) m_undo.removeLast();
                    m_undoAction->setEnabled(!m_undo.isEmpty());
                    m_ignoreNextChange = true;
                }
            });
    connect(m_labels, &QListWidget::currentTextChanged, m_canvas,
            &AnnotationCanvas::setCurrentLabel);
    connect(m_flags, &QListWidget::itemChanged, this,
            [this](QListWidgetItem* item) {
                if (m_updating) return;
                QJsonObject flags =
                        m_document.value(QStringLiteral("flags")).toObject();
                flags.insert(item->text(), item->checkState() == Qt::Checked);
                m_document.insert(QStringLiteral("flags"), flags);
                setDirty(true);
            });
    connect(m_checked, &QCheckBox::toggled, this, [this]() {
        if (!m_updating) setDirty(true);
    });
    connect(m_model, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AnnotationWindow::refreshQuantizations);
    for (QComboBox* selector :
         {m_model, m_quant, m_backend, m_textModel, m_faceLandmark}) {
        connect(selector, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this]() { ++m_generation; });
    }
    connect(m_confidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this]() { ++m_generation; });
    connect(m_slicedDetection, &QCheckBox::toggled, this, [this](bool enabled) {
        for (QWidget* widget : {static_cast<QWidget*>(m_sliceWidth),
                                static_cast<QWidget*>(m_sliceHeight),
                                static_cast<QWidget*>(m_sliceOverlap),
                                static_cast<QWidget*>(m_sliceMergeThreshold)})
            widget->setEnabled(enabled);
        ++m_generation;
    });
    connect(m_sliceWidth, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this]() { ++m_generation; });
    connect(m_sliceHeight, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this]() { ++m_generation; });
    connect(m_sliceOverlap,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this]() { ++m_generation; });
    connect(m_sliceMergeThreshold,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this]() { ++m_generation; });
    connect(m_samText, &QLineEdit::textChanged, this, [this]() {
        ++m_promptRevision;
        ++m_generation;
    });
    connect(m_samText, &QLineEdit::returnPressed, this,
            &AnnotationWindow::runDetection);
    connect(m_gkdSupportButton, &QPushButton::clicked, this, [this]() {
        if (!m_gkdSupportImage.isNull()) {
            m_gkdSupportImage = QImage();
            m_gkdSupportPoints.clear();
            m_gkdSupportButton->setText(tr("Use selected points"));
            m_status->setText(tr("GKD visual support cleared"));
        } else {
            QVector<int> selected =
                    m_canvas->selectedIndices().values().toVector();
            std::sort(selected.begin(), selected.end());
            QVector<QPointF> points;
            for (int index : selected) {
                if (index < 0 || index >= m_canvas->shapes().size()) continue;
                const AnnotationShape& shape = m_canvas->shapes().at(index);
                if (shape.visible && shape.type == QStringLiteral("point") &&
                    shape.points.size() == 1 &&
                    std::isfinite(shape.points.first().x()) &&
                    std::isfinite(shape.points.first().y()))
                    points.append(shape.points.first());
            }
            if (points.isEmpty()) {
                m_status->setText(
                        tr("Select visible point annotations for GKD visual "
                           "support"));
                return;
            }
            m_gkdSupportImage = m_image;
            m_gkdSupportPoints = points;
            m_gkdSupportButton->setText(tr("Clear visual support"));
            m_status->setText(
                    tr("GKD visual support: %1 points").arg(points.size()));
        }
        ++m_promptRevision;
        ++m_generation;
    });
    connect(runButton, &QPushButton::clicked, this,
            &AnnotationWindow::runDetection);
    connect(skipDetButton, &QPushButton::clicked, this, [this]() {
        m_ocrSkipRequested = true;
        runDetection();
    });
}

void AnnotationWindow::updateNavigator() {
    if (!m_navigator || !m_canvas) return;
    m_navigator->setImage(m_image);
    m_navigator->setShapes(m_canvas->shapes());
    m_navigator->setViewport(m_canvas->viewportImageRect());
}

void AnnotationWindow::toggleCompareView() {
    if (!m_compareDirectory.isEmpty()) {
        closeCompareView();
        return;
    }
    if (m_imagePath.isEmpty()) {
        m_status->setText(tr("Open an image before enabling compare view"));
        return;
    }
    const QString directory = QFileDialog::getExistingDirectory(
            this, tr("Open compare folder"), m_directory,
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (directory.isEmpty()) return;

    QHash<QString, QString> files;
    QDirIterator iterator(directory, kImageFilters, QDir::Files,
                          QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        const QFileInfo info(iterator.next());
        const QString key = info.completeBaseName().toLower();
        if (!files.contains(key)) files.insert(key, info.absoluteFilePath());
    }
    if (files.isEmpty()) {
        m_status->setText(tr("No matching images in compare folder"));
        return;
    }
    m_compareDirectory = directory;
    m_compareFiles = files;
    m_compareAction->setChecked(true);
    m_compareBar->setVisible(true);
    loadCompareImage();
}

void AnnotationWindow::closeCompareView() { closeCompareView(true); }

void AnnotationWindow::closeCompareView(bool confirm) {
    if (confirm && !m_compareDirectory.isEmpty() &&
        QMessageBox::question(this, tr("Close Compare View"),
                              tr("Close the compare view?"),
                              QMessageBox::Yes | QMessageBox::No,
                              QMessageBox::No) != QMessageBox::Yes) {
        return;
    }
    m_compareDirectory.clear();
    m_compareFiles.clear();
    m_canvas->clearCompareImage();
    m_compareBar->setVisible(false);
    if (m_compareAction) m_compareAction->setChecked(false);
}

void AnnotationWindow::loadCompareImage() {
    if (m_compareDirectory.isEmpty() || m_imagePath.isEmpty()) return;
    const QString key = QFileInfo(m_imagePath).completeBaseName().toLower();
    const QString path = m_compareFiles.value(key);
    if (path.isEmpty()) {
        m_canvas->clearCompareImage();
        m_status->setText(tr("No compare image for %1")
                                  .arg(QFileInfo(m_imagePath).fileName()));
        return;
    }
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage image = reader.read();
    if (image.isNull() || image.size() != m_image.size()) {
        m_canvas->clearCompareImage();
        m_status->setText(tr("Compare image size mismatch for %1")
                                  .arg(QFileInfo(m_imagePath).fileName()));
        return;
    }
    m_canvas->setCompareImage(image);
    m_status->setText(
            tr("Comparing %1").arg(QFileInfo(m_imagePath).fileName()));
}

void AnnotationWindow::configureDigitShortcuts() {
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Configure digit shortcuts"));
    dialog.setMinimumWidth(560);
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel(tr("Assign a label and drawing mode to keys "
                                    "0-9 for quick shape creation."),
                                 &dialog));
    auto* table = new QTableWidget(10, 3, &dialog);
    table->setHorizontalHeaderLabels(
            {tr("Digit"), tr("Drawing mode"), tr("Label")});
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    const QStringList modes = {
            QStringLiteral("polygon"),  QStringLiteral("rectangle"),
            QStringLiteral("rotation"), QStringLiteral("quadrilateral"),
            QStringLiteral("circle"),   QStringLiteral("line"),
            QStringLiteral("point"),    QStringLiteral("linestrip"),
            QStringLiteral("cuboid")};
    const QStringList modeLabels = {
            tr("Polygon"),       tr("Rectangle"),  tr("Rotation"),
            tr("Quadrilateral"), tr("Circle"),     tr("Line"),
            tr("Point"),         tr("Line strip"), tr("Cuboid")};
    for (int digit = 0; digit < 10; ++digit) {
        table->setItem(digit, 0, new QTableWidgetItem(QString::number(digit)));
        auto* combo = new QComboBox(table);
        combo->addItem(tr("None"));
        for (int i = 0; i < modes.size(); ++i)
            combo->addItem(modeLabels.at(i), modes.at(i));
        const auto configured = m_digitShortcuts.constFind(digit);
        if (configured != m_digitShortcuts.constEnd()) {
            const int modeIndex = combo->findData(configured.value().first);
            if (modeIndex >= 0) combo->setCurrentIndex(modeIndex);
        }
        table->setCellWidget(digit, 1, combo);
        auto* label = new QLineEdit(table);
        if (configured != m_digitShortcuts.constEnd())
            label->setText(configured.value().second);
        label->setPlaceholderText(tr("Required when a mode is selected"));
        table->setCellWidget(digit, 2, label);
        connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                label, [label](int index) { label->setEnabled(index != 0); });
        label->setEnabled(combo->currentIndex() != 0);
    }
    layout->addWidget(table);
    auto* reset = new QPushButton(tr("Reset all"), &dialog);
    layout->addWidget(reset);
    connect(reset, &QPushButton::clicked, &dialog, [&dialog, table]() {
        if (QMessageBox::question(&dialog, QObject::tr("Reset digit shortcuts"),
                                  QObject::tr("Clear all digit shortcuts?")) !=
            QMessageBox::Yes)
            return;
        for (int digit = 0; digit < 10; ++digit) {
            auto* combo = qobject_cast<QComboBox*>(table->cellWidget(digit, 1));
            auto* label = qobject_cast<QLineEdit*>(table->cellWidget(digit, 2));
            if (combo) combo->setCurrentIndex(0);
            if (label) label->clear();
        }
    });
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&dialog, table]() {
        for (int digit = 0; digit < 10; ++digit) {
            auto* combo = qobject_cast<QComboBox*>(table->cellWidget(digit, 1));
            auto* label = qobject_cast<QLineEdit*>(table->cellWidget(digit, 2));
            if (combo && label && combo->currentIndex() != 0 &&
                label->text().trimmed().isEmpty()) {
                QMessageBox::warning(
                        &dialog, QObject::tr("Digit shortcuts"),
                        QObject::tr("Key %1 needs a label.").arg(digit));
                return;
            }
        }
        dialog.accept();
    });
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;

    QHash<int, QPair<QString, QString>> updated;
    for (int digit = 0; digit < 10; ++digit) {
        auto* combo = qobject_cast<QComboBox*>(table->cellWidget(digit, 1));
        auto* label = qobject_cast<QLineEdit*>(table->cellWidget(digit, 2));
        if (!combo || !label || combo->currentIndex() == 0) continue;
        const QString text = label->text().trimmed();
        updated.insert(digit, qMakePair(combo->currentData().toString(), text));
    }
    m_digitShortcuts = updated;
    QJsonObject serialized;
    for (auto it = m_digitShortcuts.constBegin();
         it != m_digitShortcuts.constEnd(); ++it)
        serialized.insert(
                QString::number(it.key()),
                QJsonObject{{QStringLiteral("mode"), it.value().first},
                            {QStringLiteral("label"), it.value().second}});
    QSettings settings;
    settings.beginGroup(QStringLiteral("qAnyLabeling"));
    settings.setValue(QStringLiteral("digitShortcuts"),
                      QJsonDocument(serialized).toJson(QJsonDocument::Compact));
    settings.endGroup();
}

void AnnotationWindow::activateDigitShortcut(int digit) {
    const auto configured = m_digitShortcuts.constFind(digit);
    if (configured == m_digitShortcuts.constEnd()) return;
    m_canvas->setCurrentLabel(configured.value().second);
    for (QAction* action : m_tools->actions()) {
        if (action->data().toString() == configured.value().first) {
            action->trigger();
            break;
        }
    }
    m_status->setText(
            tr("Digit %1: %2 (%3)")
                    .arg(digit)
                    .arg(configured.value().second, configured.value().first));
}

void AnnotationWindow::openVideoTracking() {
    if (!m_videoDialog) {
        auto* dialog = new QDialog(this);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle(tr("SAM3 video tracking"));
        dialog->resize(1100, 780);
        auto* layout = new QVBoxLayout(dialog);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* tab = new VideoTab(dialog);
        tab->setAppInterface(m_app);
        tab->setStandaloneDeviceControlsVisible(true);
        layout->addWidget(tab);
        m_videoDialog = dialog;
    }
    m_videoDialog->show();
    m_videoDialog->raise();
    m_videoDialog->activateWindow();
}

void AnnotationWindow::openPointCloudAnnotation() {
    if (!m_pointCloudWindow) {
        m_pointCloudWindow = new PointCloudWindow(this);
        m_pointCloudWindow->setAttribute(Qt::WA_DeleteOnClose);
    }
    m_pointCloudWindow->show();
    m_pointCloudWindow->raise();
    m_pointCloudWindow->activateWindow();
}

void AnnotationWindow::saveCroppedImage() {
    if (m_image.isNull() || m_canvas->shapes().isEmpty()) return;
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Save Cropped Image"));
    auto* form = new QFormLayout(&dialog);
    auto* directory = new QLineEdit(QDir(QFileInfo(m_imagePath).absolutePath())
                                            .filePath(QStringLiteral("crops")),
                                    &dialog);
    auto* browse = new QPushButton(tr("Browse..."), &dialog);
    auto* directoryRow = new QWidget(&dialog);
    auto* directoryLayout = new QHBoxLayout(directoryRow);
    directoryLayout->setContentsMargins(0, 0, 0, 0);
    directoryLayout->addWidget(directory);
    directoryLayout->addWidget(browse);
    form->addRow(tr("Output directory"), directoryRow);
    auto* minimumWidth = new QSpinBox(&dialog);
    minimumWidth->setRange(1, 100000);
    minimumWidth->setValue(1);
    auto* minimumHeight = new QSpinBox(&dialog);
    minimumHeight->setRange(1, 100000);
    minimumHeight->setValue(1);
    form->addRow(tr("Minimum width"), minimumWidth);
    form->addRow(tr("Minimum height"), minimumHeight);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(browse, &QPushButton::clicked, &dialog, [&]() {
        const QString selected = QFileDialog::getExistingDirectory(
                &dialog, tr("Select output directory"), directory->text());
        if (!selected.isEmpty()) directory->setText(selected);
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;
    QDir output(directory->text());
    if (!output.mkpath(QStringLiteral("."))) {
        QMessageBox::warning(
                this, tr("Save Cropped Image"),
                tr("Unable to create %1").arg(output.absolutePath()));
        return;
    }
    const QRect imageRect = m_image.rect();
    const QString stem = QFileInfo(m_imagePath).completeBaseName();
    int saved = 0;
    const auto isClosedShape = [](const QString& type) {
        return type == QStringLiteral("polygon") ||
               type == QStringLiteral("rectangle") ||
               type == QStringLiteral("rotation") ||
               type == QStringLiteral("quadrilateral") ||
               type == QStringLiteral("circle") ||
               type == QStringLiteral("cuboid");
    };
    for (const AnnotationShape& shape : m_canvas->shapes()) {
        if (!shape.visible || shape.points.size() < 2 ||
            !isClosedShape(shape.type))
            continue;
        const QRect bounds = QPolygonF(shape.points)
                                     .boundingRect()
                                     .toAlignedRect()
                                     .intersected(imageRect);
        if (bounds.width() < minimumWidth->value() ||
            bounds.height() < minimumHeight->value())
            continue;
        QString label = shape.label.trimmed();
        if (label.isEmpty()) label = QStringLiteral("unlabeled");
        for (const QChar forbidden : {QChar('/'), QChar('\\'), QChar(':')})
            label.replace(forbidden, QChar('_'));
        const QDir labelDirectory(output.filePath(label));
        if (!labelDirectory.mkpath(QStringLiteral("."))) continue;
        const QString filename = QStringLiteral("%1_%2-%3.jpg")
                                         .arg(stem)
                                         .arg(++saved, 4, 10, QChar('0'))
                                         .arg(shape.type);
        if (!m_image.copy(bounds)
                     .convertToFormat(QImage::Format_RGB888)
                     .save(labelDirectory.filePath(filename), "JPG"))
            --saved;
    }
    m_status->setText(tr("Saved %1 cropped image(s)").arg(saved));
}

void AnnotationWindow::saveVisualizationImage() {
    if (m_image.isNull()) return;
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Save Visualization Image"));
    auto* form = new QFormLayout(&dialog);
    auto* output = new QLineEdit(
            QDir(QFileInfo(m_imagePath).absolutePath())
                    .filePath(QFileInfo(m_imagePath).completeBaseName() +
                              QStringLiteral("_visualization.png")),
            &dialog);
    auto* browse = new QPushButton(tr("Browse..."), &dialog);
    auto* outputRow = new QWidget(&dialog);
    auto* outputLayout = new QHBoxLayout(outputRow);
    outputLayout->setContentsMargins(0, 0, 0, 0);
    outputLayout->addWidget(output);
    outputLayout->addWidget(browse);
    form->addRow(tr("Output image"), outputRow);
    auto* saveLabels = new QCheckBox(tr("Labels"), &dialog);
    auto* saveScores = new QCheckBox(tr("Scores"), &dialog);
    auto* saveGroups = new QCheckBox(tr("Group IDs"), &dialog);
    auto* saveTexts = new QCheckBox(tr("Texts"), &dialog);
    auto* saveMasks = new QCheckBox(tr("Semi-transparent masks"), &dialog);
    saveLabels->setChecked(m_canvas->showLabels());
    saveScores->setChecked(m_canvas->showScores());
    saveGroups->setChecked(m_canvas->showGroups());
    saveTexts->setChecked(m_canvas->showTexts());
    saveMasks->setChecked(m_canvas->showMasks());
    auto* options = new QHBoxLayout;
    options->addWidget(saveLabels);
    options->addWidget(saveScores);
    options->addWidget(saveGroups);
    options->addWidget(saveTexts);
    options->addWidget(saveMasks);
    auto* optionsWidget = new QWidget(&dialog);
    optionsWidget->setLayout(options);
    form->addRow(tr("Overlay options"), optionsWidget);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(browse, &QPushButton::clicked, &dialog, [&]() {
        const QString selected = QFileDialog::getSaveFileName(
                &dialog, tr("Save visualization image"), output->text(),
                tr("PNG image (*.png);;JPEG image (*.jpg *.jpeg)"));
        if (!selected.isEmpty()) output->setText(selected);
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;
    const bool oldLabels = m_canvas->showLabels();
    const bool oldScores = m_canvas->showScores();
    const bool oldGroups = m_canvas->showGroups();
    const bool oldTexts = m_canvas->showTexts();
    const bool oldMasks = m_canvas->showMasks();
    m_canvas->setShowLabels(saveLabels->isChecked());
    m_canvas->setShowScores(saveScores->isChecked());
    m_canvas->setShowGroups(saveGroups->isChecked());
    m_canvas->setShowTexts(saveTexts->isChecked());
    m_canvas->setShowMasks(saveMasks->isChecked());
    const QImage rendered =
            m_canvas->renderVisualizationImage(m_canvas->shapes());
    m_canvas->setShowLabels(oldLabels);
    m_canvas->setShowScores(oldScores);
    m_canvas->setShowGroups(oldGroups);
    m_canvas->setShowTexts(oldTexts);
    m_canvas->setShowMasks(oldMasks);
    if (rendered.isNull() || !rendered.save(output->text())) {
        QMessageBox::warning(this, tr("Save Visualization Image"),
                             tr("Unable to write %1").arg(output->text()));
        return;
    }
    m_status->setText(tr("Saved visualization image: %1")
                              .arg(QFileInfo(output->text()).fileName()));
}

void AnnotationWindow::openShapeConverter() {
    const QStringList types = {
            QStringLiteral("polygon"),  QStringLiteral("rectangle"),
            QStringLiteral("rotation"), QStringLiteral("line"),
            QStringLiteral("circle"),   QStringLiteral("quadrilateral")};
    const QHash<QString, QStringList> targets = {
            {QStringLiteral("polygon"),
             {QStringLiteral("rectangle"), QStringLiteral("rotation")}},
            {QStringLiteral("rectangle"),
             {QStringLiteral("rotation"), QStringLiteral("polygon"),
              QStringLiteral("circle"), QStringLiteral("quadrilateral")}},
            {QStringLiteral("rotation"),
             {QStringLiteral("rectangle"), QStringLiteral("quadrilateral"),
              QStringLiteral("polygon"), QStringLiteral("circle")}},
            {QStringLiteral("line"), {QStringLiteral("linestrip")}},
            {QStringLiteral("circle"),
             {QStringLiteral("rectangle"), QStringLiteral("rotation"),
              QStringLiteral("quadrilateral"), QStringLiteral("polygon")}},
            {QStringLiteral("quadrilateral"), {QStringLiteral("polygon")}}};
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Shape Converter"));
    auto* form = new QFormLayout(&dialog);
    auto* source = new QComboBox(&dialog);
    source->addItems(types);
    auto* target = new QComboBox(&dialog);
    auto refreshTargets = [&]() {
        target->clear();
        target->addItems(targets.value(source->currentText()));
    };
    refreshTargets();
    connect(source, &QComboBox::currentTextChanged, &dialog, refreshTargets);
    form->addRow(tr("Source type"), source);
    form->addRow(tr("Target type"), target);
    auto* selectedOnly = new QCheckBox(tr("Selected shapes only"), &dialog);
    selectedOnly->setChecked(!m_canvas->selectedIndices().isEmpty());
    form->addRow(QString(), selectedOnly);
    auto* sides = new QSpinBox(&dialog);
    sides->setRange(3, 128);
    sides->setValue(32);
    form->addRow(tr("Circle polygon sides"), sides);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;
    const QString from = source->currentText();
    const QString to = target->currentText();
    if (from.isEmpty() || to.isEmpty()) return;
    const QSet<int> selected = m_canvas->selectedIndices();
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    const auto axisBox = [](const QVector<QPointF>& points) {
        if (points.isEmpty()) return QVector<QPointF>();
        QRectF bounds = QPolygonF(points).boundingRect();
        return QVector<QPointF>{bounds.topLeft(), bounds.topRight(),
                                bounds.bottomRight(), bounds.bottomLeft()};
    };
    const auto circleData = [](const QVector<QPointF>& points, QPointF* center,
                               double* radius) {
        if (points.size() != 2) return false;
        *center = points[0];
        *radius = QLineF(points[0], points[1]).length();
        return *radius > 0.0;
    };
    const auto rotateCircleBox = [](const QPointF& center, double radius) {
        const int left = qRound(center.x() - radius);
        const int right = qRound(center.x() + radius);
        const int top = qRound(center.y() - radius);
        const int bottom = qRound(center.y() + radius);
        return QVector<QPointF>{QPointF(left, top), QPointF(right, top),
                                QPointF(right, bottom), QPointF(left, bottom)};
    };
    int converted = 0;
    snapshot();
    for (int index = 0; index < shapes.size(); ++index) {
        AnnotationShape& shape = shapes[index];
        if (shape.locked || shape.type != from ||
            (selectedOnly->isChecked() && !selected.contains(index)))
            continue;
        QVector<QPointF> convertedPoints;
        if (to == QStringLiteral("rectangle") ||
            to == QStringLiteral("rotation")) {
            if (from == QStringLiteral("circle")) {
                QPointF center;
                double radius = 0.0;
                if (!circleData(shape.points, &center, &radius)) continue;
                convertedPoints = rotateCircleBox(center, radius);
            } else {
                convertedPoints = axisBox(shape.points);
            }
        } else if (to == QStringLiteral("polygon")) {
            if (from == QStringLiteral("circle")) {
                QPointF center;
                double radius = 0.0;
                if (!circleData(shape.points, &center, &radius)) continue;
                for (int i = 0; i < sides->value(); ++i) {
                    const double angle =
                            2.0 * 3.141592653589793 * i / sides->value();
                    convertedPoints.append(center +
                                           QPointF(radius * std::cos(angle),
                                                   radius * std::sin(angle)));
                }
            } else {
                convertedPoints = shape.points;
                if (from == QStringLiteral("rectangle"))
                    convertedPoints = axisBox(convertedPoints);
            }
        } else if (to == QStringLiteral("quadrilateral")) {
            convertedPoints = from == QStringLiteral("circle")
                                      ? QVector<QPointF>()
                                      : (from == QStringLiteral("rectangle")
                                                 ? axisBox(shape.points)
                                                 : shape.points);
            if (convertedPoints.isEmpty() && from == QStringLiteral("circle")) {
                QPointF center;
                double radius = 0.0;
                if (!circleData(shape.points, &center, &radius)) continue;
                convertedPoints = rotateCircleBox(center, radius);
            }
        } else if (to == QStringLiteral("circle")) {
            QPointF center;
            double radius = 0.0;
            if (from == QStringLiteral("rectangle")) {
                convertedPoints = axisBox(shape.points);
                if (convertedPoints.size() == 4) {
                    center = (convertedPoints[0] + convertedPoints[2]) / 2.0;
                    radius = std::min(QLineF(convertedPoints[0],
                                             convertedPoints[1])
                                              .length(),
                                      QLineF(convertedPoints[1],
                                             convertedPoints[2])
                                              .length()) /
                             2.0;
                }
            } else if (from == QStringLiteral("rotation") &&
                       shape.points.size() == 4) {
                center = (shape.points[0] + shape.points[1] + shape.points[2] +
                          shape.points[3]) /
                         4.0;
                radius = std::numeric_limits<double>::max();
                for (int i = 0; i < 4; ++i)
                    radius = std::min(radius, QLineF(shape.points[i],
                                                     shape.points[(i + 1) % 4])
                                                              .length() /
                                                      2.0);
            }
            if (radius <= 0.0 || radius == std::numeric_limits<double>::max())
                continue;
            convertedPoints = {center, center + QPointF(radius, 0.0)};
        } else if (to == QStringLiteral("linestrip") &&
                   from == QStringLiteral("line")) {
            convertedPoints = shape.points;
        }
        if (convertedPoints.isEmpty()) continue;
        shape.points = convertedPoints;
        shape.type = to;
        shape.extra.remove(QStringLiteral("direction"));
        if (to == QStringLiteral("rotation"))
            shape.extra.insert(QStringLiteral("direction"), 0);
        ++converted;
    }
    if (converted == 0) return;
    m_canvas->setShapes(shapes);
    m_canvas->setSelectedShapes(selected.values().toVector());
    updateObjects();
    setDirty(true);
    m_status->setText(tr("Converted %1 shape(s) from %2 to %3")
                              .arg(converted)
                              .arg(from, to));
}

void AnnotationWindow::editLabels() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;

    QStringList imageFiles;
    if (m_files) {
        for (int i = 0; i < m_files->count(); ++i) {
            const QString path =
                    m_files->item(i)->data(Qt::UserRole).toString();
            if (!path.isEmpty()) imageFiles.append(path);
        }
    }
    if (imageFiles.isEmpty()) imageFiles.append(m_imagePath);
    if (!imageFiles.contains(m_imagePath)) imageFiles.append(m_imagePath);

    const auto sidecarPath = [this](const QString& image) {
        return sidecarCandidates(image).first();
    };
    QSet<QString> labels;
    QHash<QString, QColor> labelColors;
    QSet<QString> explicitColors;
    for (const QString& image : imageFiles) {
        QFile input(sidecarPath(image));
        if (!input.exists()) continue;
        if (!input.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, tr("Label change manager"),
                                 tr("Unable to read %1").arg(input.fileName()));
            return;
        }
        QJsonParseError parseError;
        const QJsonDocument document =
                QJsonDocument::fromJson(input.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError ||
            !document.isObject()) {
            QMessageBox::warning(
                    this, tr("Label change manager"),
                    tr("Cannot parse %1: %2")
                            .arg(input.fileName(), parseError.errorString()));
            return;
        }
        if (document.object().value(QStringLiteral("type")).toString() ==
            QStringLiteral("video_classification"))
            continue;
        for (const QJsonValue& value :
             document.object().value(QStringLiteral("shapes")).toArray()) {
            if (value.isObject()) {
                const QString label = value.toObject()
                                              .value(QStringLiteral("label"))
                                              .toString();
                if (!label.isEmpty()) {
                    labels.insert(label);
                    const bool hasColor = value.toObject().contains(
                            QStringLiteral("acloudviewer_color"));
                    if (!labelColors.contains(label) ||
                        (hasColor && !explicitColors.contains(label)))
                        labelColors.insert(
                                label,
                                annotationShapeColor(label, value.toObject()));
                    if (hasColor) explicitColors.insert(label);
                }
            }
        }
    }
    if (labels.isEmpty()) {
        QMessageBox::information(this, tr("Label change manager"),
                                 tr("No labels were found in this folder."));
        return;
    }

    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("AnnotationLabelManager"));
    dialog.setWindowTitle(tr("Label Change Manager"));
    dialog.setWindowFlags(dialog.windowFlags() | Qt::WindowMinimizeButtonHint |
                          Qt::WindowMaximizeButtonHint);
    dialog.resize(700, 400);

    auto* table = new QTableWidget(&dialog);
    table->setObjectName(QStringLiteral("AnnotationLabelTable"));
    table->setColumnCount(5);
    table->setHorizontalHeaderLabels({tr("Category"), tr("Delete"),
                                      tr("New Value"), tr("Visible"),
                                      tr("Color")});
    table->setSelectionMode(QAbstractItemView::NoSelection);
    table->setFocusPolicy(Qt::NoFocus);
    table->verticalHeader()->setDefaultSectionSize(40);
    table->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(
            1, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(
            3, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(
            4, QHeaderView::ResizeToContents);

    const auto checkboxCell = [&dialog](bool checked) {
        auto* cell = new QWidget(&dialog);
        auto* layout = new QHBoxLayout(cell);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* checkbox = new QCheckBox(cell);
        checkbox->setChecked(checked);
        layout->addWidget(checkbox, 0, Qt::AlignCenter);
        return cell;
    };
    QStringList sortedLabels = labels.values();
    std::sort(sortedLabels.begin(), sortedLabels.end(),
              [](const QString& left, const QString& right) {
                  return left.toCaseFolded() < right.toCaseFolded();
              });
    table->setRowCount(sortedLabels.size());
    for (int row = 0; row < sortedLabels.size(); ++row) {
        const QString& label = sortedLabels.at(row);
        auto* category = new QTableWidgetItem(label);
        category->setFlags(category->flags() & ~Qt::ItemIsEditable);
        category->setToolTip(tr("Double-click to copy label text"));
        table->setItem(row, 0, category);
        table->setCellWidget(row, 1, checkboxCell(false));
        auto* replacement = new QLineEdit(table);
        replacement->setObjectName(
                QStringLiteral("AnnotationLabelReplacement"));
        table->setCellWidget(row, 2, replacement);
        table->setCellWidget(row, 3, checkboxCell(true));
        const QColor color =
                labelColors.value(label, annotationShapeColor(label, {}));
        auto* swatch = new QPushButton(table);
        swatch->setObjectName(QStringLiteral("AnnotationLabelColorButton"));
        swatch->setFixedSize(28, 28);
        swatch->setProperty("labelColor", color);
        swatch->setProperty("originalColor", color);
        const auto showColor = [swatch](const QColor& value) {
            swatch->setToolTip(QObject::tr("Change color: %1")
                                       .arg(value.name(QColor::HexArgb)));
            swatch->setStyleSheet(
                    QStringLiteral(
                            "background-color:%1; border:1px solid #555; "
                            "border-radius:14px;")
                            .arg(value.name(QColor::HexRgb)));
        };
        showColor(color);
        connect(swatch, &QPushButton::clicked, &dialog,
                [&dialog, swatch, showColor]() {
                    const QColor selected = QColorDialog::getColor(
                            swatch->property("labelColor").value<QColor>(),
                            &dialog, QObject::tr("Label color"),
                            QColorDialog::ShowAlphaChannel |
                                    QColorDialog::DontUseNativeDialog);
                    if (!selected.isValid()) return;
                    swatch->setProperty("labelColor", selected);
                    showColor(selected);
                });
        table->setCellWidget(row, 4, swatch);
    }
    connect(table, &QTableWidget::itemDoubleClicked, &dialog,
            [](QTableWidgetItem* item) {
                if (item && item->column() == 0)
                    QApplication::clipboard()->setText(item->text());
            });
    table->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(table, &QTableWidget::customContextMenuRequested, &dialog,
            [&dialog, table](const QPoint& point) {
                if (table->columnAt(point.x()) != 3) return;
                QMenu menu(&dialog);
                QAction* selectAll = menu.addAction(QObject::tr("Select All"));
                QAction* deselectAll =
                        menu.addAction(QObject::tr("Deselect All"));
                QAction* selected =
                        menu.exec(table->viewport()->mapToGlobal(point));
                if (selected != selectAll && selected != deselectAll) return;
                for (int row = 0; row < table->rowCount(); ++row)
                    table->cellWidget(row, 3)
                            ->findChild<QCheckBox*>()
                            ->setChecked(selected == selectAll);
            });

    auto* fromLabel = new QLabel(tr("From:"), &dialog);
    auto* from = new QSpinBox(&dialog);
    from->setObjectName(QStringLiteral("AnnotationLabelFrom"));
    from->setRange(1, std::max(1, imageFiles.size()));
    from->setValue(1);
    auto* toLabel = new QLabel(tr("To:"), &dialog);
    auto* to = new QSpinBox(&dialog);
    to->setObjectName(QStringLiteral("AnnotationLabelTo"));
    to->setRange(1, std::max(1, imageFiles.size()));
    to->setValue(std::max(1, imageFiles.size()));
    auto* go = new QPushButton(tr("Go"), &dialog);
    go->setObjectName(QStringLiteral("AnnotationLabelApplyRange"));
    auto* rangeLayout = new QHBoxLayout;
    rangeLayout->addStretch(1);
    rangeLayout->addWidget(fromLabel);
    rangeLayout->addWidget(from);
    rangeLayout->addWidget(toLabel);
    rangeLayout->addWidget(to);
    rangeLayout->addWidget(go);
    rangeLayout->addStretch(1);

    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    buttons->setObjectName(QStringLiteral("AnnotationLabelButtons"));
    auto* layout = new QVBoxLayout(&dialog);
    layout->addWidget(table);
    layout->addLayout(rangeLayout);
    layout->addWidget(buttons);

    struct LabelRule {
        QString replacement;
        bool remove = false;
        bool visible = true;
        QColor color;
        bool colorChanged = false;
    };
    const auto collectRules = [table, &sortedLabels]() {
        QHash<QString, LabelRule> rules;
        for (int row = 0; row < sortedLabels.size(); ++row) {
            const auto* deleteCell = table->cellWidget(row, 1);
            const auto* visibleCell = table->cellWidget(row, 3);
            const auto* deleteBox = deleteCell->findChild<QCheckBox*>();
            const auto* visibleBox = visibleCell->findChild<QCheckBox*>();
            const auto* replacement =
                    qobject_cast<QLineEdit*>(table->cellWidget(row, 2));
            LabelRule rule;
            rule.remove = deleteBox && deleteBox->isChecked();
            rule.visible = !visibleBox || visibleBox->isChecked();
            rule.replacement =
                    replacement ? replacement->text().trimmed() : QString();
            const auto* swatch =
                    qobject_cast<QPushButton*>(table->cellWidget(row, 4));
            if (swatch) {
                rule.color = swatch->property("labelColor").value<QColor>();
                rule.colorChanged =
                        rule.color !=
                        swatch->property("originalColor").value<QColor>();
            }
            rules.insert(sortedLabels.at(row), rule);
        }
        return rules;
    };

    const auto applyChanges = [this, &dialog, &imageFiles, &sidecarPath, from,
                               to, collectRules]() {
        if (from->value() > to->value()) {
            QMessageBox::warning(
                    &dialog, tr("Label change manager"),
                    tr("The start file must not be after the end file."));
            return false;
        }
        const QHash<QString, LabelRule> rules = collectRules();
        // Upstream validate_label parity: under "exact" every replacement
        // label must already exist in the unique label list, checked before
        // any file is touched.
        if (m_validateLabel == QStringLiteral("exact")) {
            for (auto rule = rules.constBegin(); rule != rules.constEnd();
                 ++rule) {
                if (rule->replacement.isEmpty() ||
                    validateLabel(rule->replacement))
                    continue;
                QMessageBox::warning(
                        &dialog, tr("Label change manager"),
                        tr("Label \"%1\" is not in the unique label list "
                           "(exact validation).")
                                .arg(rule->replacement));
                return false;
            }
        }
        struct PendingWrite {
            QString path;
            QByteArray data;
        };
        QVector<PendingWrite> pending;
        for (int index = from->value() - 1; index < to->value(); ++index) {
            const QString path = sidecarPath(imageFiles.at(index));
            QFile input(path);
            if (!input.exists()) continue;
            if (!input.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(&dialog, tr("Label change manager"),
                                     tr("Unable to read %1").arg(path));
                return false;
            }
            QJsonParseError parseError;
            const QJsonDocument document =
                    QJsonDocument::fromJson(input.readAll(), &parseError);
            if (parseError.error != QJsonParseError::NoError ||
                !document.isObject()) {
                QMessageBox::warning(
                        &dialog, tr("Label change manager"),
                        tr("Cannot parse %1: %2")
                                .arg(path, parseError.errorString()));
                return false;
            }
            QJsonObject object = document.object();
            QJsonArray updatedShapes;
            bool changed = false;
            for (const QJsonValue& value :
                 object.value(QStringLiteral("shapes")).toArray()) {
                if (!value.isObject()) {
                    updatedShapes.append(value);
                    continue;
                }
                QJsonObject shape = value.toObject();
                const QString label =
                        shape.value(QStringLiteral("label")).toString();
                const auto rule = rules.constFind(label);
                if (rule == rules.constEnd()) {
                    updatedShapes.append(shape);
                    continue;
                }
                const bool locked =
                        shape.value(QStringLiteral("locked")).toBool();
                if (rule->remove && !locked) {
                    changed = true;
                    continue;
                }
                if (!rule->replacement.isEmpty() &&
                    rule->replacement != label) {
                    shape.insert(QStringLiteral("label"), rule->replacement);
                    changed = true;
                }
                if (shape.value(QStringLiteral("visible")).toBool(true) !=
                    rule->visible) {
                    shape.insert(QStringLiteral("visible"), rule->visible);
                    changed = true;
                }
                if (rule->colorChanged && rule->color.isValid()) {
                    const QColor color = rule->color;
                    shape.insert(QStringLiteral("acloudviewer_color"),
                                 QJsonArray{color.red(), color.green(),
                                            color.blue(), color.alpha()});
                    const QByteArray encoded =
                            shape.value(QStringLiteral("acloudviewer_mask_png"))
                                    .toString()
                                    .toLatin1();
                    if (!encoded.isEmpty()) {
                        QImage mask =
                                QImage::fromData(
                                        QByteArray::fromBase64(encoded), "PNG")
                                        .convertToFormat(QImage::Format_ARGB32);
                        if (mask.isNull()) {
                            QMessageBox::warning(
                                    &dialog, tr("Label change manager"),
                                    tr("Cannot decode a mask in %1").arg(path));
                            return false;
                        }
                        for (int y = 0; y < mask.height(); ++y) {
                            auto* pixels =
                                    reinterpret_cast<QRgb*>(mask.scanLine(y));
                            for (int x = 0; x < mask.width(); ++x) {
                                const int alpha = qAlpha(pixels[x]);
                                if (alpha)
                                    pixels[x] =
                                            qRgba(color.red(), color.green(),
                                                  color.blue(), alpha);
                            }
                        }
                        QByteArray bytes;
                        QBuffer buffer(&bytes);
                        if (!buffer.open(QIODevice::WriteOnly) ||
                            !mask.save(&buffer, "PNG")) {
                            QMessageBox::warning(
                                    &dialog, tr("Label change manager"),
                                    tr("Cannot encode a mask in %1").arg(path));
                            return false;
                        }
                        shape.insert(QStringLiteral("acloudviewer_mask_png"),
                                     QString::fromLatin1(bytes.toBase64()));
                    }
                    changed = true;
                }
                updatedShapes.append(shape);
            }
            if (!changed) continue;
            object.insert(QStringLiteral("shapes"), updatedShapes);
            pending.append({path, QJsonDocument(object).toJson(
                                          QJsonDocument::Indented)});
        }
        for (const PendingWrite& write : pending) {
            QSaveFile output(write.path);
            if (!output.open(QIODevice::WriteOnly) ||
                output.write(write.data) != write.data.size() ||
                !output.commit()) {
                QMessageBox::warning(
                        &dialog, tr("Label change manager"),
                        tr("Unable to save %1: %2")
                                .arg(write.path, output.errorString()));
                return false;
            }
        }
        m_dirty = false;
        if (!m_imagePath.isEmpty()) openFile(m_imagePath);
        m_status->setText(tr("Labels modified successfully"));
        return true;
    };
    const auto applyAndAccept = [applyChanges, &dialog]() {
        if (applyChanges()) dialog.accept();
    };
    connect(buttons, &QDialogButtonBox::accepted, &dialog, applyAndAccept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(go, &QPushButton::clicked, &dialog, applyAndAccept);
    dialog.exec();
}

void AnnotationWindow::openSettings() {
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("AnnotationSettingsDialog"));
    dialog.setWindowTitle(tr("Settings"));
    dialog.setModal(true);
    dialog.resize(920, 600);
    dialog.setMinimumSize(720, 480);
    auto* shell = new QHBoxLayout(&dialog);
    shell->setContentsMargins(0, 0, 0, 0);
    shell->setSpacing(0);
    auto* sidebar = new QWidget(&dialog);
    sidebar->setObjectName(QStringLiteral("AnnotationSettingsSidebar"));
    sidebar->setFixedWidth(160);
    auto* sidebarLayout = new QVBoxLayout(sidebar);
    sidebarLayout->setContentsMargins(8, 12, 8, 12);
    auto* brand = new QLabel(tr("Settings"), sidebar);
    brand->setObjectName(QStringLiteral("AnnotationSettingsBrand"));
    brand->setMargin(8);
    sidebarLayout->addWidget(brand);
    auto* navigation = new QListWidget(sidebar);
    navigation->setObjectName(QStringLiteral("AnnotationSettingsNavigation"));
    navigation->setFrameShape(QFrame::NoFrame);
    navigation->setSpacing(6);
    navigation->addItem(new QListWidgetItem(
            QIcon(iconPath(QStringLiteral("shortcuts.svg"))), tr("Shortcuts")));
    navigation->addItem(new QListWidgetItem(
            QIcon(iconPath(QStringLiteral("settings.svg"))), tr("General")));
    navigation->addItem(new QListWidgetItem(
            QIcon(iconPath(QStringLiteral("polygon.png"))), tr("Shape")));
    navigation->addItem(new QListWidgetItem(
            QIcon(iconPath(QStringLiteral("compare.png"))), tr("Canvas")));
    navigation->setFixedHeight(190);
    sidebarLayout->addWidget(navigation);
    sidebarLayout->addStretch();
    shell->addWidget(sidebar);

    auto* content = new QWidget(&dialog);
    content->setObjectName(QStringLiteral("AnnotationSettingsContent"));
    auto* root = new QVBoxLayout(content);
    root->setContentsMargins(16, 12, 16, 12);
    auto* title = new QLabel(content);
    title->setObjectName(QStringLiteral("AnnotationSettingsTitle"));
    root->addWidget(title);
    auto* tabs = new QStackedWidget(content);
    tabs->setObjectName(QStringLiteral("AnnotationSettingsPages"));
    root->addWidget(tabs, 1);
    shell->addWidget(content, 1);
    dialog.setStyleSheet(QStringLiteral(
            "#AnnotationSettingsSidebar { background: rgb(225,225,225); }"
            "#AnnotationSettingsContent { background: rgb(239,238,239); }"
            "#AnnotationSettingsBrand { color: rgb(75,75,80); font-weight: "
            "600; }"
            "#AnnotationSettingsTitle { color: rgb(25,25,25); font-weight: "
            "700; "
            "font-size: 15px; min-height: 36px; }"
            "#AnnotationSettingsNavigation { background: transparent; "
            "border: none; outline: none; }"
            "#AnnotationSettingsNavigation::item { height: 28px; padding-left: "
            "8px; }"
            "#AnnotationSettingsNavigation::item:selected { "
            "background: rgb(212,212,216); color: rgb(30,30,30); }"));

    auto* general = new QWidget(tabs);
    auto* generalForm = new QFormLayout(general);
    generalForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto* keepBrightness =
            new QCheckBox(tr("Keep previous brightness"), general);
    auto* keepContrast = new QCheckBox(tr("Keep previous contrast"), general);
    auto* autoLastLabel = new QCheckBox(tr("Auto use last label"), general);
    auto* autoHighlight = new QCheckBox(tr("Auto highlight shape"), general);
    auto* autoSwitchEdit =
            new QCheckBox(tr("Switch to edit mode after drawing"), general);
    auto* exifScan = new QCheckBox(tr("Apply EXIF orientation"), general);
    auto* systemClipboard =
            new QCheckBox(tr("Use system clipboard for objects"), general);
    auto* switchChecked =
            new QCheckBox(tr("Switch to checked image after marking"), general);
    keepBrightness->setChecked(m_keepBrightnessAction &&
                               m_keepBrightnessAction->isChecked());
    keepContrast->setChecked(m_keepContrastAction &&
                             m_keepContrastAction->isChecked());
    autoLastLabel->setChecked(m_autoUseLastLabelAction &&
                              m_autoUseLastLabelAction->isChecked());
    autoHighlight->setChecked(m_autoHighlightShape);
    autoSwitchEdit->setChecked(m_autoSwitchToEditMode);
    exifScan->setChecked(m_exifScanEnabled);
    systemClipboard->setChecked(m_systemClipboard);
    switchChecked->setChecked(m_switchToChecked);
    generalForm->addRow(tr("Behavior"), keepBrightness);
    generalForm->addRow(QString(), keepContrast);
    generalForm->addRow(QString(), autoLastLabel);
    generalForm->addRow(QString(), autoHighlight);
    generalForm->addRow(QString(), autoSwitchEdit);
    generalForm->addRow(QString(), exifScan);
    generalForm->addRow(QString(), systemClipboard);
    generalForm->addRow(QString(), switchChecked);
    auto* displayLabelPopup = new QCheckBox(tr("Display label popup"), general);
    displayLabelPopup->setChecked(m_displayLabelPopup);
    auto* fileListEditable =
            new QCheckBox(tr("File list checkbox editable"), general);
    fileListEditable->setChecked(m_fileListCheckboxEditable);
    generalForm->addRow(QString(), displayLabelPopup);
    generalForm->addRow(QString(), fileListEditable);
    auto* sortLabels = new QCheckBox(tr("Sort labels"), general);
    sortLabels->setChecked(m_sortLabels);
    auto* showLabelTextField =
            new QCheckBox(tr("Show label text field"), general);
    showLabelTextField->setChecked(m_showLabelTextField);
    auto* labelCompletion = new QComboBox(general);
    labelCompletion->addItems(
            {QStringLiteral("startswith"), QStringLiteral("contains")});
    labelCompletion->setCurrentText(m_labelCompletion);
    auto* moveMode = new QComboBox(general);
    moveMode->addItems({QStringLiteral("auto"), QStringLiteral("free")});
    moveMode->setCurrentText(m_moveMode);
    auto* fitColumn = new QCheckBox(tr("Fit to content column"), general);
    fitColumn->setChecked(m_fitToContentColumn);
    auto* fitRow = new QCheckBox(tr("Fit to content row"), general);
    fitRow->setChecked(m_fitToContentRow);
    auto* fontCombo = new QFontComboBox(general);
    if (!m_fontFamily.isEmpty()) fontCombo->setCurrentFont(QFont(m_fontFamily));
    generalForm->addRow(tr("Label Dialog"), sortLabels);
    generalForm->addRow(QString(), showLabelTextField);
    generalForm->addRow(tr("Label completion"), labelCompletion);
    generalForm->addRow(tr("Move mode"), moveMode);
    generalForm->addRow(QString(), fitColumn);
    generalForm->addRow(QString(), fitRow);
    generalForm->addRow(tr("Application font"), fontCombo);
    auto* validateLabelMode = new QComboBox(general);
    validateLabelMode->setObjectName(
            QStringLiteral("AnnotationSettingsValidateLabel"));
    validateLabelMode->addItems(
            {QStringLiteral("(none)"), QStringLiteral("exact")});
    validateLabelMode->setCurrentText(m_validateLabel.isEmpty()
                                              ? QStringLiteral("(none)")
                                              : m_validateLabel);
    auto* loggerLevel = new QComboBox(general);
    loggerLevel->setObjectName(QStringLiteral("AnnotationSettingsLoggerLevel"));
    loggerLevel->addItems({QStringLiteral("info"), QStringLiteral("warning"),
                           QStringLiteral("error")});
    loggerLevel->setCurrentText(m_loggerLevel);
    generalForm->addRow(tr("Validate label"), validateLabelMode);
    generalForm->addRow(tr("Logger level"), loggerLevel);
    auto* settingsTodo = new QLabel(
            tr("Unimplemented upstream fields (contract-owned): model hub is "
               "owned by the AICore catalog, and the Qt image allocation "
               "limit needs Qt 6. See qAnyLabeling/TODO.md."),
            general);
    settingsTodo->setWordWrap(true);
    settingsTodo->setStyleSheet(QStringLiteral("color: gray;"));
    generalForm->addRow(QString(), settingsTodo);
    tabs->addWidget(general);

    auto* shape = new QWidget(tabs);
    auto* shapeForm = new QFormLayout(shape);
    auto* radius = new QSpinBox(shape);
    radius->setObjectName(QStringLiteral("AnnotationSettingsBrushRadius"));
    radius->setRange(1, 200);
    radius->setSuffix(tr(" px"));
    radius->setValue(m_brushRadius ? m_brushRadius->value()
                                   : m_defaultBrushRadius);
    auto* operation = new QComboBox(shape);
    operation->setObjectName(QStringLiteral("AnnotationSettingsBrushMode"));
    operation->addItem(tr("Paint"), 0);
    operation->addItem(tr("Erase"), 1);
    operation->addItem(tr("Invert"), 2);
    if (m_brushMode) operation->setCurrentIndex(m_brushMode->currentIndex());
    shapeForm->addRow(tr("Brush radius"), radius);
    shapeForm->addRow(tr("Brush operation"), operation);
    auto* magicThreshold = new QSpinBox(shape);
    magicThreshold->setObjectName(
            QStringLiteral("AnnotationSettingsMagicThreshold"));
    magicThreshold->setRange(0, 255);
    magicThreshold->setValue(m_magicDefaultThreshold);
    auto* magicSensitivity = new QDoubleSpinBox(shape);
    magicSensitivity->setObjectName(
            QStringLiteral("AnnotationSettingsMagicSensitivity"));
    magicSensitivity->setRange(0.1, 100.0);
    magicSensitivity->setDecimals(2);
    magicSensitivity->setValue(m_magicDragSensitivity);
    auto* magicLuminance = new QDoubleSpinBox(shape);
    magicLuminance->setObjectName(
            QStringLiteral("AnnotationSettingsMagicLuminance"));
    magicLuminance->setRange(0.0, 1.0);
    magicLuminance->setDecimals(2);
    magicLuminance->setValue(m_magicLuminanceWeight);
    shapeForm->addRow(tr("Magic wand threshold"), magicThreshold);
    shapeForm->addRow(tr("Magic wand drag sensitivity"), magicSensitivity);
    shapeForm->addRow(tr("Magic wand luminance weight"), magicLuminance);
    auto* pointSize = new QSpinBox(shape);
    pointSize->setObjectName(
            QStringLiteral("AnnotationSettingsShapePointSize"));
    pointSize->setRange(1, 64);
    pointSize->setValue(m_shapePointSize);
    auto* lineWidth = new QDoubleSpinBox(shape);
    lineWidth->setObjectName(
            QStringLiteral("AnnotationSettingsShapeLineWidth"));
    lineWidth->setRange(0.5, 32.0);
    lineWidth->setDecimals(1);
    lineWidth->setValue(m_shapeLineWidth);
    auto* magicSimplify = new QDoubleSpinBox(shape);
    magicSimplify->setRange(0.0, 100.0);
    magicSimplify->setDecimals(2);
    magicSimplify->setValue(m_magicSimplifyEpsilon);
    auto* magicOpacity = new QDoubleSpinBox(shape);
    magicOpacity->setRange(0.0, 1.0);
    magicOpacity->setDecimals(2);
    magicOpacity->setValue(m_magicOpacity);
    shapeForm->addRow(tr("Point size"), pointSize);
    shapeForm->addRow(tr("Line width"), lineWidth);
    shapeForm->addRow(tr("Magic wand simplify"), magicSimplify);
    shapeForm->addRow(tr("Magic wand preview opacity"), magicOpacity);
    auto* brushPointDistance = new QDoubleSpinBox(shape);
    brushPointDistance->setRange(0.0, 500.0);
    brushPointDistance->setDecimals(1);
    brushPointDistance->setValue(m_brushPointDistance);
    auto* brushSimplify = new QDoubleSpinBox(shape);
    brushSimplify->setRange(0.0, 100.0);
    brushSimplify->setDecimals(2);
    brushSimplify->setValue(m_brushSimplifyEpsilon);
    auto* brushUndoSteps = new QSpinBox(shape);
    brushUndoSteps->setRange(1, 1000);
    brushUndoSteps->setValue(m_brushMaxUndoSteps);
    auto* brushUndoMemory = new QSpinBox(shape);
    brushUndoMemory->setRange(1, 4096);
    brushUndoMemory->setSuffix(tr(" MB"));
    brushUndoMemory->setValue(m_brushMaxUndoMemoryMb);
    shapeForm->addRow(tr("Brush point distance"), brushPointDistance);
    shapeForm->addRow(tr("Brush simplify"), brushSimplify);
    shapeForm->addRow(tr("Brush undo steps"), brushUndoSteps);
    shapeForm->addRow(tr("Brush undo memory"), brushUndoMemory);
    // Source shape.* color group. Colors persist as hex strings and feed the
    // canvas render path through the same defaults as the source config.
    const auto colorButton = [](QWidget* parent, const QColor& initial) {
        auto* button = new QPushButton(initial.name(QColor::HexArgb), parent);
        QObject::connect(button, &QPushButton::clicked, button, [button]() {
            const QColor chosen = QColorDialog::getColor(
                    QColor(button->text()), button, QObject::tr("Shape color"));
            if (chosen.isValid()) button->setText(chosen.name(QColor::HexArgb));
        });
        return button;
    };
    const QSettings shapeStyleSettings;
    auto* strategy = new QComboBox(shape);
    strategy->addItem(tr("Auto (per label)"), QStringLiteral("auto"));
    strategy->addItem(tr("Manual"), QStringLiteral("manual"));
    strategy->setCurrentIndex(
            shapeStyleSettings.value(QStringLiteral(
                                             "qAnyLabeling/shapeColorStrategy"),
                                     QStringLiteral("auto"))
                                    .toString() == QStringLiteral("manual")
                    ? 1
                    : 0);
    auto* defaultColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral(
                                          "qAnyLabeling/defaultShapeColor"),
                                  QStringLiteral("#00ff00"))
                           .toString()));
    auto* shiftColor = new QSpinBox(shape);
    shiftColor->setRange(-5, 5);
    shiftColor->setValue(
            shapeStyleSettings
                    .value(QStringLiteral("qAnyLabeling/shiftAutoShapeColor"),
                           0)
                    .toInt());
    auto* lineColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral("qAnyLabeling/shapeLineColor"),
                                  QStringLiteral("#8000ff00"))
                           .toString()));
    auto* fillColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral("qAnyLabeling/shapeFillColor"),
                                  QStringLiteral("#96dcdcdc"))
                           .toString()));
    auto* vertexColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral(
                                          "qAnyLabeling/shapeVertexFillColor"),
                                  QStringLiteral("#ff00ff00"))
                           .toString()));
    auto* selectLineColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral(
                                          "qAnyLabeling/shapeSelectLineColor"),
                                  QStringLiteral("#ffffffff"))
                           .toString()));
    auto* selectFillColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral(
                                          "qAnyLabeling/shapeSelectFillColor"),
                                  QStringLiteral("#9b00ff00"))
                           .toString()));
    auto* hoverVertexColor = colorButton(
            shape,
            QColor(shapeStyleSettings
                           .value(QStringLiteral("qAnyLabeling/"
                                                 "shapeHoverVertexFillColor"),
                                  QStringLiteral("#ffffffff"))
                           .toString()));
    shapeForm->addRow(tr("Shape color strategy"), strategy);
    shapeForm->addRow(tr("Default shape color"), defaultColor);
    shapeForm->addRow(tr("Auto color shift"), shiftColor);
    shapeForm->addRow(tr("Line color"), lineColor);
    shapeForm->addRow(tr("Fill color"), fillColor);
    shapeForm->addRow(tr("Vertex fill color"), vertexColor);
    shapeForm->addRow(tr("Select line color"), selectLineColor);
    shapeForm->addRow(tr("Select fill color"), selectFillColor);
    shapeForm->addRow(tr("Hover vertex fill color"), hoverVertexColor);
    auto* shapeScroll = new QScrollArea(tabs);
    shapeScroll->setWidgetResizable(true);
    shapeScroll->setFrameShape(QFrame::NoFrame);
    shapeScroll->setWidget(shape);
    tabs->addWidget(shapeScroll);

    auto* canvas = new QWidget(tabs);
    auto* canvasForm = new QFormLayout(canvas);
    auto* opacity = new QSlider(Qt::Horizontal, canvas);
    opacity->setObjectName(QStringLiteral("AnnotationSettingsMaskOpacity"));
    opacity->setRange(0, 255);
    opacity->setValue(m_canvas->maskOpacity());
    auto* shapeOpacity = new QSlider(Qt::Horizontal, canvas);
    shapeOpacity->setObjectName(
            QStringLiteral("AnnotationSettingsShapeOpacity"));
    shapeOpacity->setRange(0, 100);
    shapeOpacity->setValue(m_canvas->shapeOpacity());
    auto* showShapes = new QCheckBox(tr("Show shapes"), canvas);
    auto* showMasks = new QCheckBox(tr("Show masks"), canvas);
    auto* showLabels = new QCheckBox(tr("Show labels"), canvas);
    auto* showScores = new QCheckBox(tr("Show scores"), canvas);
    showScores->setObjectName(QStringLiteral("AnnotationSettingsShowScores"));
    auto* showGroups = new QCheckBox(tr("Show groups"), canvas);
    auto* showTexts = new QCheckBox(tr("Show descriptions"), canvas);
    auto* showAttributes = new QCheckBox(tr("Show attributes"), canvas);
    auto* showDegrees = new QCheckBox(tr("Show rotation degrees"), canvas);
    auto* showLinking = new QCheckBox(tr("Show KIE linking"), canvas);
    auto* epsilon = new QDoubleSpinBox(canvas);
    epsilon->setObjectName(QStringLiteral("AnnotationSettingsCanvasEpsilon"));
    epsilon->setRange(0.5, 100.0);
    epsilon->setDecimals(1);
    epsilon->setValue(m_canvasEpsilon);
    auto* doubleClickClose =
            new QCheckBox(tr("Double-click closes polygon"), canvas);
    doubleClickClose->setChecked(m_doubleClickClose);
    auto* doubleClickLabel =
            new QCheckBox(tr("Double-click edits label"), canvas);
    doubleClickLabel->setChecked(m_doubleClickEditLabel);
    auto* backups = new QSpinBox(canvas);
    backups->setRange(1, 1000);
    backups->setValue(m_numBackups);
    auto* wheelEdit = new QCheckBox(tr("Wheel edits rectangles"), canvas);
    wheelEdit->setChecked(m_wheelRectangleEditing);
    auto* wheelAdjust = new QDoubleSpinBox(canvas);
    wheelAdjust->setRange(0.1, 50.0);
    wheelAdjust->setDecimals(2);
    wheelAdjust->setValue(m_wheelAdjustStep);
    auto* wheelScale = new QDoubleSpinBox(canvas);
    wheelScale->setRange(0.001, 1.0);
    wheelScale->setDecimals(3);
    wheelScale->setValue(m_wheelScaleStep);
    auto* crosshair = new QCheckBox(tr("Show crosshair"), canvas);
    crosshair->setChecked(m_crosshairVisible);
    auto* crosshairWidth = new QDoubleSpinBox(canvas);
    crosshairWidth->setRange(0.5, 16.0);
    crosshairWidth->setDecimals(1);
    crosshairWidth->setValue(m_crosshairWidth);
    auto* crosshairOpacity = new QDoubleSpinBox(canvas);
    crosshairOpacity->setRange(0.0, 1.0);
    crosshairOpacity->setDecimals(2);
    crosshairOpacity->setValue(m_crosshairOpacity);
    auto* crosshairColor =
            new QPushButton(m_crosshairColor.name(QColor::HexRgb), canvas);
    connect(crosshairColor, &QPushButton::clicked, crosshairColor,
            [crosshairColor]() {
                const QColor chosen = QColorDialog::getColor(
                        QColor(crosshairColor->text()), crosshairColor,
                        tr("Crosshair color"));
                if (chosen.isValid())
                    crosshairColor->setText(chosen.name(QColor::HexRgb));
            });
    auto* rotationLarge = new QDoubleSpinBox(canvas);
    rotationLarge->setRange(0.01, 45.0);
    rotationLarge->setDecimals(2);
    rotationLarge->setValue(m_rotationLargeIncrement);
    auto* rotationSmall = new QDoubleSpinBox(canvas);
    rotationSmall->setRange(0.01, 10.0);
    rotationSmall->setDecimals(2);
    rotationSmall->setValue(m_rotationSmallIncrement);
    auto* cuboidDepth = new QDoubleSpinBox(canvas);
    cuboidDepth->setRange(0.0, 10000.0);
    cuboidDepth->setDecimals(1);
    cuboidDepth->setValue(m_cuboidMinDepth);
    showShapes->setChecked(m_showShapesAction &&
                           m_showShapesAction->isChecked());
    showMasks->setChecked(m_showMasksAction && m_showMasksAction->isChecked());
    showLabels->setChecked(m_showLabelsAction &&
                           m_showLabelsAction->isChecked());
    showScores->setChecked(m_showScoresAction &&
                           m_showScoresAction->isChecked());
    showGroups->setChecked(m_showGroupsAction &&
                           m_showGroupsAction->isChecked());
    showTexts->setChecked(m_showTextsAction && m_showTextsAction->isChecked());
    showAttributes->setChecked(m_showAttributesAction &&
                               m_showAttributesAction->isChecked());
    showDegrees->setChecked(m_showDegreesAction &&
                            m_showDegreesAction->isChecked());
    showLinking->setChecked(m_showLinkingAction &&
                            m_showLinkingAction->isChecked());
    canvasForm->addRow(tr("Mask opacity"), opacity);
    canvasForm->addRow(tr("Shape opacity"), shapeOpacity);
    canvasForm->addRow(QString(), showShapes);
    canvasForm->addRow(QString(), showMasks);
    canvasForm->addRow(QString(), showLabels);
    canvasForm->addRow(QString(), showScores);
    canvasForm->addRow(QString(), showGroups);
    canvasForm->addRow(QString(), showTexts);
    canvasForm->addRow(QString(), showAttributes);
    canvasForm->addRow(QString(), showDegrees);
    canvasForm->addRow(QString(), showLinking);
    canvasForm->addRow(tr("Hit epsilon"), epsilon);
    canvasForm->addRow(QString(), doubleClickClose);
    canvasForm->addRow(QString(), doubleClickLabel);
    canvasForm->addRow(tr("Undo backups"), backups);
    canvasForm->addRow(QString(), wheelEdit);
    canvasForm->addRow(tr("Wheel adjust step"), wheelAdjust);
    canvasForm->addRow(tr("Wheel scale step"), wheelScale);
    canvasForm->addRow(QString(), crosshair);
    canvasForm->addRow(tr("Crosshair width"), crosshairWidth);
    canvasForm->addRow(tr("Crosshair opacity"), crosshairOpacity);
    canvasForm->addRow(tr("Crosshair color"), crosshairColor);
    canvasForm->addRow(tr("Rotation large increment"), rotationLarge);
    canvasForm->addRow(tr("Rotation small increment"), rotationSmall);
    canvasForm->addRow(tr("Cuboid minimum depth"), cuboidDepth);
    auto* canvasScroll = new QScrollArea(tabs);
    canvasScroll->setWidgetResizable(true);
    canvasScroll->setFrameShape(QFrame::NoFrame);
    canvasScroll->setWidget(canvas);
    tabs->addWidget(canvasScroll);

    auto* shortcuts = new QWidget(tabs);
    auto* shortcutsLayout = new QVBoxLayout(shortcuts);
    auto* shortcutFilter = new QComboBox(shortcuts);
    shortcutFilter->setObjectName(QStringLiteral("AnnotationShortcutCategory"));
    shortcutFilter->addItems({tr("All"), tr("AI"), tr("Dialog"), tr("File"),
                              tr("Shape"), tr("View")});
    shortcutsLayout->addWidget(shortcutFilter);
    auto* shortcutTable = new QTableWidget(shortcuts);
    shortcutTable->setObjectName(QStringLiteral("AnnotationShortcutTable"));
    shortcutTable->setColumnCount(3);
    shortcutTable->setHorizontalHeaderLabels(
            {tr("Action"), tr("Category"), tr("Shortcut")});
    shortcutTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    shortcutTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    shortcutTable->verticalHeader()->hide();
    shortcutTable->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::Stretch);
    shortcutTable->horizontalHeader()->setSectionResizeMode(
            1, QHeaderView::ResizeToContents);
    shortcutTable->horizontalHeader()->setSectionResizeMode(2,
                                                            QHeaderView::Fixed);
    shortcutTable->setColumnWidth(2, 190);
    QList<QAction*> actions = findChildren<QAction*>();
    std::sort(actions.begin(), actions.end(),
              [](const QAction* lhs, const QAction* rhs) {
                  return lhs->text().localeAwareCompare(rhs->text()) < 0;
              });
    QHash<QString, QKeySequenceEdit*> shortcutEditors;
    QHash<QString, QAction*> shortcutActions;
    QHash<QString, QString> shortcutCategories;
    QHash<QString, QKeySequence> shortcutDefaults;
    for (QAction* action : actions) {
        if (!action || action->text().isEmpty() || action->isSeparator() ||
            action->property("qAnyShortcutId").toString().isEmpty())
            continue;
        const QString id = action->property("qAnyShortcutId").toString();
        if (shortcutActions.contains(id)) continue;
        const int row = shortcutTable->rowCount();
        shortcutTable->insertRow(row);
        shortcutTable->setItem(row, 0, new QTableWidgetItem(action->text()));
        const QString category =
                action->property("qAnyShortcutCategory").toString();
        shortcutTable->setItem(row, 1, new QTableWidgetItem(category));
        auto* editor = new QKeySequenceEdit(action->shortcut(), shortcutTable);
        editor->setObjectName(
                QStringLiteral("AnnotationShortcutEditor_%1").arg(id));
        shortcutTable->setCellWidget(row, 2, editor);
        editor->setProperty("qAnyShortcutId", id);
        shortcutEditors.insert(id, editor);
        shortcutActions.insert(id, action);
        shortcutCategories.insert(id, category);
        shortcutDefaults.insert(
                id,
                action->property("qAnyShortcutDefault").value<QKeySequence>());
    }
    shortcutsLayout->addWidget(shortcutTable);
    auto* resetShortcuts = new QPushButton(tr("Reset to defaults"), shortcuts);
    resetShortcuts->setObjectName(QStringLiteral("AnnotationResetShortcuts"));
    shortcutsLayout->addWidget(resetShortcuts, 0, Qt::AlignRight);
    connect(shortcutFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            &dialog, [shortcutFilter, shortcutTable](int index) {
                const QString category = shortcutFilter->itemText(index);
                for (int row = 0; row < shortcutTable->rowCount(); ++row) {
                    const auto* item = shortcutTable->item(row, 1);
                    shortcutTable->setRowHidden(
                            row,
                            index > 0 && (!item || item->text() != category));
                }
            });
    connect(resetShortcuts, &QPushButton::clicked, &dialog,
            [shortcutEditors, shortcutDefaults]() {
                for (auto it = shortcutEditors.constBegin();
                     it != shortcutEditors.constEnd(); ++it)
                    it.value()->setKeySequence(
                            shortcutDefaults.value(it.key()));
            });
    tabs->insertWidget(0, shortcuts);
    connect(navigation, &QListWidget::currentRowChanged, tabs,
            &QStackedWidget::setCurrentIndex);
    connect(navigation, &QListWidget::currentRowChanged, title,
            [navigation, title](int row) {
                if (QListWidgetItem* item = navigation->item(row))
                    title->setText(item->text());
            });
    navigation->setCurrentRow(0);

    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    root->addWidget(buttons);
    QPushButton* acceptButton = buttons->button(QDialogButtonBox::Ok);
    auto updateShortcutConflicts = [shortcutEditors, shortcutActions,
                                    shortcutCategories, shortcutDefaults,
                                    acceptButton]() {
        QHash<QString, QStringList> conflicts;
        for (auto it = shortcutEditors.constBegin();
             it != shortcutEditors.constEnd(); ++it) {
            const QKeySequence sequence = it.value()->keySequence();
            if (sequence.isEmpty()) continue;
            for (auto other = shortcutEditors.constBegin();
                 other != shortcutEditors.constEnd(); ++other) {
                if (other.key() == it.key() ||
                    other.value()->keySequence() != sequence)
                    continue;
                if ((it.key() == QStringLiteral("undo") &&
                     other.key() == QStringLiteral("undo_last_point")) ||
                    (it.key() == QStringLiteral("undo_last_point") &&
                     other.key() == QStringLiteral("undo")))
                    continue;
                conflicts[it.key()].append(
                        shortcutActions.value(other.key())->text());
            }
            QList<QAction*> globalActions =
                    QApplication::instance()->findChildren<QAction*>();
            for (QWidget* topLevel : QApplication::topLevelWidgets())
                globalActions.append(topLevel->findChildren<QAction*>());
            for (QAction* other : globalActions) {
                if (!other || shortcutActions.values().contains(other) ||
                    other->shortcutContext() != Qt::ApplicationShortcut ||
                    other->isSeparator())
                    continue;
                // Official defaults can overlap ACloudViewer's application
                // shortcuts. syncHostShortcuts suspends those host actions
                // only while the annotation workspace has focus.
                if (other->shortcut() == sequence &&
                    sequence != shortcutDefaults.value(it.key())) {
                    QString name = other->text();
                    name.remove(QLatin1Char('&'));
                    conflicts[it.key()].append(
                            tr("ACloudViewer: %1").arg(name));
                }
            }
        }
        bool hasConflict = false;
        for (auto it = shortcutEditors.constBegin();
             it != shortcutEditors.constEnd(); ++it) {
            const QStringList names = conflicts.value(it.key());
            it.value()->setStyleSheet(
                    names.isEmpty()
                            ? QString()
                            : QStringLiteral("QKeySequenceEdit { border: 1px "
                                             "solid #d43f3a; }"));
            it.value()->setToolTip(
                    names.isEmpty()
                            ? tr("This shortcut is scoped to the active "
                                 "annotation workspace.")
                            : tr("Conflicts with: %1")
                                      .arg(names.join(QStringLiteral(", "))));
            hasConflict = hasConflict || !names.isEmpty();
        }
        acceptButton->setEnabled(!hasConflict);
        Q_UNUSED(shortcutCategories);
    };
    for (QKeySequenceEdit* editor : shortcutEditors)
        connect(editor, &QKeySequenceEdit::keySequenceChanged, &dialog,
                [updateShortcutConflicts](const QKeySequence&) {
                    updateShortcutConflicts();
                });
    updateShortcutConflicts();
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
        QSettings settings;
        settings.beginGroup(QStringLiteral("qAnyLabeling/shortcuts"));
        for (auto it = shortcutEditors.constBegin();
             it != shortcutEditors.constEnd(); ++it) {
            const QKeySequence sequence = it.value()->keySequence();
            QAction* action = shortcutActions.value(it.key());
            if (action) action->setShortcut(sequence);
            settings.setValue(it.key(),
                              sequence.toString(QKeySequence::PortableText));
        }
        settings.endGroup();
        syncHostShortcuts();
        if (m_keepBrightnessAction)
            m_keepBrightnessAction->setChecked(keepBrightness->isChecked());
        if (m_keepContrastAction)
            m_keepContrastAction->setChecked(keepContrast->isChecked());
        if (m_autoUseLastLabelAction)
            m_autoUseLastLabelAction->setChecked(autoLastLabel->isChecked());
        m_autoHighlightShape = autoHighlight->isChecked();
        m_autoSwitchToEditMode = autoSwitchEdit->isChecked();
        m_exifScanEnabled = exifScan->isChecked();
        m_systemClipboard = systemClipboard->isChecked();
        if (m_useSystemClipboardAction)
            m_useSystemClipboardAction->setChecked(m_systemClipboard);
        m_switchToChecked = switchChecked->isChecked();
        m_displayLabelPopup = displayLabelPopup->isChecked();
        m_fileListCheckboxEditable = fileListEditable->isChecked();
        settings.setValue(QStringLiteral("qAnyLabeling/displayLabelPopup"),
                          m_displayLabelPopup);
        settings.setValue(
                QStringLiteral("qAnyLabeling/fileListCheckboxEditable"),
                m_fileListCheckboxEditable);
        const QColor crosshairChosen(crosshairColor->text());
        if (crosshairChosen.isValid()) {
            m_crosshairColor = crosshairChosen;
            settings.setValue(QStringLiteral("qAnyLabeling/crosshairColor"),
                              m_crosshairColor.name(QColor::HexArgb));
        }
        // Source label-dialog / move-mode / shape-color groups.
        m_sortLabels = sortLabels->isChecked();
        m_showLabelTextField = showLabelTextField->isChecked();
        m_labelCompletion = labelCompletion->currentText();
        m_moveMode = moveMode->currentText();
        m_fitToContentColumn = fitColumn->isChecked();
        m_fitToContentRow = fitRow->isChecked();
        settings.setValue(QStringLiteral("qAnyLabeling/sortLabels"),
                          m_sortLabels);
        settings.setValue(QStringLiteral("qAnyLabeling/showLabelTextField"),
                          m_showLabelTextField);
        settings.setValue(QStringLiteral("qAnyLabeling/labelCompletion"),
                          m_labelCompletion);
        settings.setValue(QStringLiteral("qAnyLabeling/moveMode"), m_moveMode);
        settings.setValue(QStringLiteral("qAnyLabeling/fitToContentColumn"),
                          m_fitToContentColumn);
        settings.setValue(QStringLiteral("qAnyLabeling/fitToContentRow"),
                          m_fitToContentRow);
        m_fontFamily = fontCombo->currentFont().family();
        settings.setValue(QStringLiteral("qAnyLabeling/fontFamily"),
                          m_fontFamily);
        m_validateLabel =
                validateLabelMode->currentText() == QStringLiteral("exact")
                        ? QStringLiteral("exact")
                        : QString();
        m_loggerLevel = loggerLevel->currentText();
        settings.setValue(QStringLiteral("qAnyLabeling/validateLabel"),
                          m_validateLabel);
        settings.setValue(QStringLiteral("qAnyLabeling/loggerLevel"),
                          m_loggerLevel);
        // Immediate effect on the level-gated plugin logger.
        qany_log::refreshLevel();
        if (!m_fontFamily.isEmpty()) setFont(QFont(m_fontFamily));
        settings.setValue(QStringLiteral("qAnyLabeling/shapeColorStrategy"),
                          strategy->currentData().toString());
        m_canvas->setShapeColorMode(
                strategy->currentData().toString() == QStringLiteral("manual")
                        ? AnnotationCanvas::ShapeColorMode::Manual
                        : AnnotationCanvas::ShapeColorMode::Auto);
        const auto applyShapeColor = [this, &settings](const QString& key,
                                                       QPushButton* button) {
            const QColor parsed(button->text());
            if (!parsed.isValid()) return;
            settings.setValue(QStringLiteral("qAnyLabeling/") + key,
                              parsed.name(QColor::HexArgb));
        };
        applyShapeColor(QStringLiteral("defaultShapeColor"), defaultColor);
        m_canvas->setDefaultShapeColor(QColor(defaultColor->text()));
        m_canvas->setShiftAutoShapeColor(shiftColor->value());
        settings.setValue(QStringLiteral("qAnyLabeling/shiftAutoShapeColor"),
                          shiftColor->value());
        applyShapeColor(QStringLiteral("shapeLineColor"), lineColor);
        applyShapeColor(QStringLiteral("shapeFillColor"), fillColor);
        applyShapeColor(QStringLiteral("shapeVertexFillColor"), vertexColor);
        applyShapeColor(QStringLiteral("shapeSelectLineColor"),
                        selectLineColor);
        applyShapeColor(QStringLiteral("shapeSelectFillColor"),
                        selectFillColor);
        applyShapeColor(QStringLiteral("shapeHoverVertexFillColor"),
                        hoverVertexColor);
        m_canvas->setShapeStyleColors(
                QColor(lineColor->text()), QColor(fillColor->text()),
                QColor(vertexColor->text()), QColor(selectLineColor->text()),
                QColor(selectFillColor->text()),
                QColor(hoverVertexColor->text()));
        m_canvas->setMoveMode(m_moveMode);
        applyFileListCheckboxPolicy();
        settings.setValue(QStringLiteral("qAnyLabeling/autoHighlightShape"),
                          m_autoHighlightShape);
        settings.setValue(QStringLiteral("qAnyLabeling/autoSwitchToEditMode"),
                          m_autoSwitchToEditMode);
        settings.setValue(QStringLiteral("qAnyLabeling/exifScanEnabled"),
                          m_exifScanEnabled);
        settings.setValue(QStringLiteral("qAnyLabeling/systemClipboard"),
                          m_systemClipboard);
        settings.setValue(QStringLiteral("qAnyLabeling/switchToChecked"),
                          m_switchToChecked);
        if (m_brushRadius) m_brushRadius->setValue(radius->value());
        if (m_brushMode)
            m_brushMode->setCurrentIndex(operation->currentIndex());
        m_defaultMaskOpacity = opacity->value();
        m_canvas->setMaskOpacity(m_defaultMaskOpacity);
        settings.setValue(QStringLiteral("qAnyLabeling/maskOpacity"),
                          m_defaultMaskOpacity);
        m_defaultShapeOpacity = shapeOpacity->value();
        m_canvas->setShapeOpacity(m_defaultShapeOpacity);
        settings.setValue(QStringLiteral("qAnyLabeling/shapeOpacity"),
                          m_defaultShapeOpacity);
        m_magicDefaultThreshold = magicThreshold->value();
        m_magicDragSensitivity = magicSensitivity->value();
        m_magicLuminanceWeight = magicLuminance->value();
        m_magicOpacity = magicOpacity->value();
        m_canvas->setMagicWandOptions(m_magicDefaultThreshold,
                                      m_magicDragSensitivity,
                                      m_magicLuminanceWeight);
        m_canvas->setMagicWandOpacity(m_magicOpacity);
        settings.setValue(
                QStringLiteral("qAnyLabeling/magicWand/defaultThreshold"),
                m_magicDefaultThreshold);
        settings.setValue(
                QStringLiteral("qAnyLabeling/magicWand/dragSensitivity"),
                m_magicDragSensitivity);
        settings.setValue(
                QStringLiteral("qAnyLabeling/magicWand/luminanceWeight"),
                m_magicLuminanceWeight);
        m_shapePointSize = pointSize->value();
        m_shapeLineWidth = lineWidth->value();
        m_canvas->setShapeStyle(m_shapePointSize, m_shapeLineWidth);
        settings.setValue(QStringLiteral("qAnyLabeling/shape/pointSize"),
                          m_shapePointSize);
        settings.setValue(QStringLiteral("qAnyLabeling/shape/lineWidth"),
                          m_shapeLineWidth);
        m_magicSimplifyEpsilon = magicSimplify->value();
        m_canvas->setMagicWandSimplifyEpsilon(m_magicSimplifyEpsilon);
        m_brushPointDistance = brushPointDistance->value();
        m_brushSimplifyEpsilon = brushSimplify->value();
        m_brushMaxUndoSteps = brushUndoSteps->value();
        m_brushMaxUndoMemoryMb = brushUndoMemory->value();
        m_canvas->setBrushPointDistance(m_brushPointDistance);
        m_canvas->setBrushHistoryLimits(m_brushMaxUndoSteps,
                                        m_brushMaxUndoMemoryMb);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/magicWand/simplifyEpsilon"),
                m_magicSimplifyEpsilon);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/magicWand/opacity"),
                m_magicOpacity);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/brush/pointDistance"),
                m_brushPointDistance);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/brush/simplifyEpsilon"),
                m_brushSimplifyEpsilon);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/brush/maxUndoSteps"),
                m_brushMaxUndoSteps);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/brush/maxUndoMemoryMb"),
                m_brushMaxUndoMemoryMb);
        m_canvasEpsilon = epsilon->value();
        m_canvas->setHitEpsilon(m_canvasEpsilon);
        m_doubleClickClose = doubleClickClose->isChecked();
        m_doubleClickEditLabel = doubleClickLabel->isChecked();
        m_canvas->setDoubleClickClose(m_doubleClickClose);
        m_canvas->setDoubleClickEditLabel(m_doubleClickEditLabel);
        m_numBackups = backups->value();
        m_wheelRectangleEditing = wheelEdit->isChecked();
        m_wheelAdjustStep = wheelAdjust->value();
        m_wheelScaleStep = wheelScale->value();
        m_crosshairVisible = crosshair->isChecked();
        m_crosshairWidth = crosshairWidth->value();
        m_crosshairOpacity = crosshairOpacity->value();
        m_rotationLargeIncrement = rotationLarge->value();
        m_rotationSmallIncrement = rotationSmall->value();
        m_cuboidMinDepth = cuboidDepth->value();
        m_canvas->setWheelRectangleEditing(m_wheelRectangleEditing,
                                           m_wheelAdjustStep, m_wheelScaleStep);
        m_canvas->setBrushPointDistance(m_brushPointDistance);
        m_canvas->setRotationIncrements(m_rotationLargeIncrement,
                                        m_rotationSmallIncrement);
        m_canvas->setCuboidDefaults(QPointF(24.0, -24.0), m_cuboidMinDepth);
        m_canvas->setCrosshair(m_crosshairVisible, m_crosshairWidth,
                               m_crosshairColor, m_crosshairOpacity);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/epsilon"),
                          m_canvasEpsilon);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/doubleClickClose"),
                m_doubleClickClose);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/doubleClickEditLabel"),
                m_doubleClickEditLabel);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/numBackups"),
                          m_numBackups);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/wheelRectangleEditing"),
                m_wheelRectangleEditing);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/wheelAdjustStep"),
                          m_wheelAdjustStep);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/wheelScaleStep"),
                          m_wheelScaleStep);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/crosshair/show"),
                          m_crosshairVisible);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/crosshair/width"),
                          m_crosshairWidth);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/crosshair/color"),
                          m_crosshairColor.name(QColor::HexArgb));
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/crosshair/opacity"),
                m_crosshairOpacity);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/rotation/largeIncrement"),
                m_rotationLargeIncrement);
        settings.setValue(
                QStringLiteral("qAnyLabeling/canvas/rotation/smallIncrement"),
                m_rotationSmallIncrement);
        settings.setValue(QStringLiteral("qAnyLabeling/canvas/cuboid/minDepth"),
                          m_cuboidMinDepth);
        if (m_showShapesAction)
            m_showShapesAction->setChecked(showShapes->isChecked());
        if (m_showMasksAction)
            m_showMasksAction->setChecked(showMasks->isChecked());
        if (m_showLabelsAction)
            m_showLabelsAction->setChecked(showLabels->isChecked());
        if (m_showScoresAction)
            m_showScoresAction->setChecked(showScores->isChecked());
        if (m_showGroupsAction)
            m_showGroupsAction->setChecked(showGroups->isChecked());
        if (m_showTextsAction)
            m_showTextsAction->setChecked(showTexts->isChecked());
        if (m_showAttributesAction)
            m_showAttributesAction->setChecked(showAttributes->isChecked());
        if (m_showDegreesAction)
            m_showDegreesAction->setChecked(showDegrees->isChecked());
        if (m_showLinkingAction)
            m_showLinkingAction->setChecked(showLinking->isChecked());
        dialog.accept();
    });
    dialog.exec();
    updateInstruction();
}

void AnnotationWindow::applyTheme(const QString& mode) {
    QPalette palette = QApplication::style()->standardPalette();
    if (mode == QStringLiteral("dark")) {
        palette.setColor(QPalette::Window, QColor(45, 45, 48));
        palette.setColor(QPalette::Base, QColor(30, 30, 30));
        palette.setColor(QPalette::AlternateBase, QColor(52, 52, 55));
        palette.setColor(QPalette::Text, QColor(235, 235, 235));
        palette.setColor(QPalette::WindowText, QColor(235, 235, 235));
        palette.setColor(QPalette::Button, QColor(55, 55, 58));
        palette.setColor(QPalette::ButtonText, QColor(235, 235, 235));
        palette.setColor(QPalette::Highlight, QColor(42, 130, 218));
        palette.setColor(QPalette::HighlightedText, Qt::white);
    } else if (mode == QStringLiteral("light")) {
        palette.setColor(QPalette::Window, QColor(240, 240, 240));
        palette.setColor(QPalette::Base, Qt::white);
        palette.setColor(QPalette::AlternateBase, QColor(248, 248, 248));
        palette.setColor(QPalette::Text, QColor(32, 32, 32));
        palette.setColor(QPalette::WindowText, QColor(32, 32, 32));
        palette.setColor(QPalette::Button, QColor(240, 240, 240));
        palette.setColor(QPalette::ButtonText, QColor(32, 32, 32));
        palette.setColor(QPalette::Highlight, QColor(48, 140, 198));
        palette.setColor(QPalette::HighlightedText, Qt::white);
    }
    setPalette(palette);
    if (m_adjustmentPanel) {
        const bool dark = mode == QStringLiteral("dark");
        m_adjustmentPanel->setStyleSheet(
                dark ? QStringLiteral(
                               "QFrame { background: rgba(25,25,25,235); }")
                     : QStringLiteral("QFrame { background: "
                                      "rgba(245,245,245,235); }"));
    }
}

void AnnotationWindow::updateInstruction() {
    if (!m_instruction || !m_canvas) return;
    const auto shortcut = [this](const QString& id) {
        for (QAction* action : findChildren<QAction*>()) {
            if (action->property("qAnyShortcutId").toString() == id) {
                const QString value =
                        action->shortcut().toString(QKeySequence::NativeText);
                return value.isEmpty() ? tr("None") : value;
            }
        }
        return tr("None");
    };
    QString mode = m_canvas->tool();
    if (m_tools) {
        for (QAction* action : m_tools->actions()) {
            if (action->data().toString() == mode) {
                mode = action->text();
                break;
            }
        }
    }
    m_instruction->setTextFormat(Qt::RichText);
    m_instruction->setText(
            tr("<b>Mode:</b> %1 | <b>Shortcuts:</b> Previous(%2), "
               "Next(%3), Rectangle(%4), Polygon(%5), Rotation(%6), "
               "Quadrilateral(%7), Settings(%8)")
                    .arg(mode.toHtmlEscaped(),
                         shortcut(QStringLiteral("open_prev")),
                         shortcut(QStringLiteral("open_next")),
                         shortcut(QStringLiteral("create_rectangle")),
                         shortcut(QStringLiteral("create_polygon")),
                         shortcut(QStringLiteral("create_rotation")),
                         shortcut(QStringLiteral("create_quadrilateral")),
                         shortcut(QStringLiteral("open_settings"))));
}

void AnnotationWindow::createActions() {
    const auto registerShortcut = [this](QAction* action,
                                         const QString& sourceKey,
                                         const QString& category) {
        if (!action || sourceKey.isEmpty()) return;
        const QKeySequence defaultSequence = action->shortcut();
        if (action->objectName().isEmpty())
            action->setObjectName(
                    QStringLiteral("qAnyShortcut_%1").arg(sourceKey));
        action->setProperty("qAnyShortcutId", sourceKey);
        action->setProperty("qAnyShortcutCategory", category);
        action->setProperty("qAnyShortcutDefault",
                            QVariant::fromValue(defaultSequence));
        // Every annotation command is scoped to this top-level workspace.
        // Qt activates WindowShortcut only while this window is active, so
        // the host viewer keeps ownership of its own window shortcuts.
        action->setShortcutContext(Qt::WindowShortcut);
        QSettings settings;
        const QString key =
                QStringLiteral("qAnyLabeling/shortcuts/%1").arg(sourceKey);
        if (settings.contains(key)) {
            action->setShortcut(
                    QKeySequence::fromString(settings.value(key).toString(),
                                             QKeySequence::PortableText));
        }
    };
    QMenu* fileMenu = menuBar()->addMenu(tr("File"));
    QMenu* editMenu = menuBar()->addMenu(tr("Edit"));
    QMenu* viewMenu = menuBar()->addMenu(tr("View"));
    QMenu* themeMenu = menuBar()->addMenu(tr("Theme"));
    QMenu* languageMenu = menuBar()->addMenu(tr("Language"));
    auto* themeGroup = new QActionGroup(this);
    themeGroup->setExclusive(true);
    const QString savedTheme =
            QSettings()
                    .value(QStringLiteral("qAnyLabeling/theme"),
                           QStringLiteral("auto"))
                    .toString();
    const auto addTheme = [this, themeMenu, themeGroup, savedTheme](
                                  const QString& id, const QString& title) {
        auto* action = themeMenu->addAction(title);
        action->setCheckable(true);
        action->setChecked(savedTheme == id);
        themeGroup->addAction(action);
        connect(action, &QAction::triggered, this, [this, id]() {
            QSettings settings;
            settings.setValue(QStringLiteral("qAnyLabeling/theme"), id);
            applyTheme(id);
            m_status->setText(tr("Theme applied: %1").arg(id));
        });
    };
    addTheme(QStringLiteral("auto"), tr("Auto"));
    addTheme(QStringLiteral("light"), tr("Light"));
    addTheme(QStringLiteral("dark"), tr("Dark"));
    applyTheme(savedTheme);
    auto* languageGroup = new QActionGroup(this);
    languageGroup->setExclusive(true);
    const QString savedLanguage =
            QSettings()
                    .value(QStringLiteral("qAnyLabeling/language"),
                           QStringLiteral("en_US"))
                    .toString();
    const auto addLanguage = [this, languageMenu, languageGroup, savedLanguage](
                                     const QString& id, const QString& title) {
        auto* action = languageMenu->addAction(title);
        action->setCheckable(true);
        action->setChecked(savedLanguage == id);
        languageGroup->addAction(action);
        const bool available =
                id == QStringLiteral("en_US") ||
                (id == QStringLiteral("zh_CN") &&
                 QFile::exists(QStringLiteral(":/qAnyLabeling/translations/"
                                              "qanylabeling_zh_CN.qm")));
        if (!available) {
            action->setEnabled(false);
            action->setToolTip(tr("TODO: add a C++ translation catalog"));
        }
        connect(action, &QAction::triggered, this, [this, id]() {
            QSettings settings;
            settings.setValue(QStringLiteral("qAnyLabeling/language"), id);
            m_status->setText(
                    tr("Language selected: %1 (restart the application for "
                       "full coverage)")
                            .arg(id));
        });
    };
    addLanguage(QStringLiteral("en_US"), tr("English"));
    addLanguage(QStringLiteral("zh_CN"), tr("简体中文"));
    addLanguage(QStringLiteral("ja_JP"), tr("日本語"));
    addLanguage(QStringLiteral("ko_KR"), tr("한국어"));
    auto* fileBar = new QToolBar(tr("Tools"), m_toolbarHost);
    fileBar->setObjectName(QStringLiteral("AnnotationToolbar"));
    fileBar->setOrientation(Qt::Vertical);
    fileBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
    fileBar->setIconSize(QSize(24, 24));
    // Match the source toolbar's narrow 40px rail.  The surrounding scroll
    // area provides the additional six pixels used by the upstream layout.
    fileBar->setFixedWidth(40);
    fileBar->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::MinimumExpanding);
    fileBar->setMovable(false);
    m_toolbarHostLayout->addWidget(fileBar);
    m_toolbarHostLayout->addStretch(1);
    QHash<QString, QAction*> toolbarActions;
    auto addCommand = [this, fileMenu, &registerShortcut, &toolbarActions](
                              const QString& text, const QString& icon,
                              const QKeySequence& key, const QString& sourceKey,
                              const std::function<void()>& callback) {
        QAction* action = new QAction(QIcon(iconPath(icon)), text, this);
        action->setShortcut(key);
        registerShortcut(action, sourceKey,
                         sourceKey == QStringLiteral("auto_run") ? tr("AI")
                                                                 : tr("File"));
        action->setProperty("qAnyToolbarId", sourceKey);
        connect(action, &QAction::triggered, this, callback);
        fileMenu->addAction(action);
        if (!sourceKey.isEmpty()) toolbarActions.insert(sourceKey, action);
        return action;
    };
    QAction* openImageAction = addCommand(
            tr("Open File"), QStringLiteral("file.png"),
            QKeySequence(Qt::CTRL + Qt::Key_I), QStringLiteral("open"),
            [this]() {
                const QString file = QFileDialog::getOpenFileName(
                        this, tr("Open image"), m_directory, imageFilter());
                if (!file.isEmpty()) openFile(file);
            });
    QAction* openVideo = addCommand(
            tr("Open Video"), QStringLiteral("video.png"),
            QKeySequence(Qt::CTRL + Qt::Key_O), QStringLiteral("open_video"),
            [this]() { openVideoTracking(); });
    QAction* openDir = addCommand(tr("Open Dir"), QStringLiteral("folder.svg"),
                                  QKeySequence(Qt::CTRL + Qt::Key_U),
                                  QStringLiteral("open_dir"),
                                  [this]() { openDirectory(); });
    m_recentMenu = new QMenu(tr("Open Recent"), this);
    m_recentMenu->setObjectName(QStringLiteral("AnnotationRecentMenu"));
    m_recentMenu->setIcon(QIcon(iconPath(QStringLiteral("labels.png"))));
    connect(m_recentMenu, &QMenu::aboutToShow, this,
            &AnnotationWindow::updateRecentMenu);
    fileMenu->addMenu(m_recentMenu);
    updateRecentMenu();
    QAction* videoClassifier =
            addCommand(tr("Video Classifier"), QStringLiteral("cavy.png"),
                       QKeySequence(Qt::CTRL + Qt::Key_5),
                       QStringLiteral("open_video_classifier"), [this]() {
                           auto* dialog = new VideoAnnotationDialog(this);
                           dialog->setAttribute(Qt::WA_DeleteOnClose);
                           dialog->show();
                       });
    fileMenu->removeAction(videoClassifier);
    m_saveAction = addCommand(tr("Save"), QStringLiteral("save.png"),
                              QKeySequence::Save, QStringLiteral("save"),
                              [this]() { saveDocument(); });
    m_saveAsAction = addCommand(tr("Save As"), QStringLiteral("save-as.png"),
                                QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_S),
                                QStringLiteral("save_as"),
                                [this]() { saveDocumentAs(); });
    QAction* saveAsAction = m_saveAsAction;
    m_autoRunAction =
            addCommand(tr("Auto Run"), QStringLiteral("auto-run.png"),
                       QKeySequence(Qt::CTRL + Qt::Key_B),
                       QStringLiteral("auto_run"), [this]() { runBatch(); });
    QAction* autoRunAction = m_autoRunAction;
    fileMenu->removeAction(autoRunAction);
    m_closeAction =
            addCommand(tr("Close"), QStringLiteral("cancel.png"),
                       QKeySequence(Qt::CTRL + Qt::Key_W),
                       QStringLiteral("close"), [this]() { closeImage(); });
    QAction* closeAction = m_closeAction;
    m_deleteLabelFileAction = addCommand(
            tr("Delete File"), QStringLiteral("delete.png"),
            QKeySequence(Qt::CTRL + Qt::Key_Delete),
            QStringLiteral("delete_file"), [this]() { deleteLabelFile(); });
    QAction* deleteLabelAction = m_deleteLabelFileAction;
    QAction* deleteImageAction =
            addCommand(tr("Delete Image File"), QStringLiteral("delete.png"),
                       QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_Delete),
                       QStringLiteral("delete_image_file"),
                       [this]() { deleteImageFile(); });
    QAction* previousAction = addCommand(
            tr("Prev Image"), QStringLiteral("prev.png"),
            QKeySequence(Qt::Key_A), QStringLiteral("open_prev"), [this]() {
                const int row = m_files->currentRow();
                if (row > 0) m_files->setCurrentRow(row - 1);
            });
    m_previousAction = previousAction;
    QAction* nextAction = addCommand(
            tr("Next Image"), QStringLiteral("next.png"),
            QKeySequence(Qt::Key_D), QStringLiteral("open_next"), [this]() {
                const int row = m_files->currentRow();
                if (row + 1 < m_files->count()) m_files->setCurrentRow(row + 1);
            });
    m_nextAction = nextAction;
    QAction* previousUncheckedAction = addCommand(
            tr("Prev Unchecked Image"), QStringLiteral("prev.png"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_A),
            QStringLiteral("open_prev_unchecked"), [this]() {
                if (!confirmSave()) return;
                const int row = m_files->currentRow();
                for (int i = row - 1; i >= 0; --i) {
                    if (m_files->item(i)->checkState() != Qt::Checked) {
                        m_files->setCurrentRow(i);
                        return;
                    }
                }
            });
    m_previousUncheckedAction = previousUncheckedAction;
    QAction* nextUncheckedAction = addCommand(
            tr("Next Unchecked Image"), QStringLiteral("next.png"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_D),
            QStringLiteral("open_next_unchecked"), [this]() {
                if (!confirmSave()) return;
                const int row = m_files->currentRow();
                for (int i = row + 1; i < m_files->count(); ++i) {
                    if (m_files->item(i)->checkState() != Qt::Checked) {
                        m_files->setCurrentRow(i);
                        return;
                    }
                }
            });
    m_nextUncheckedAction = nextUncheckedAction;
    QAction* toggleChecked =
            new QAction(QIcon(iconPath(QStringLiteral("checkmark.svg"))),
                        tr("Mark as Checked"), this);
    m_toggleCheckedAction = toggleChecked;
    toggleChecked->setCheckable(true);
    toggleChecked->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_K));
    registerShortcut(toggleChecked, QStringLiteral("toggle_annotation_checked"),
                     tr("Edit"));
    connect(toggleChecked, &QAction::triggered, this,
            [this]() { m_checked->setChecked(!m_checked->isChecked()); });
    // The source exposes this through the window action/context set rather
    // than the File menu.  Register it on the workspace so the shortcut is
    // active whenever the annotation window owns focus.
    addAction(toggleChecked);
    QAction* navigator = viewMenu->addAction(tr("Navigator"));
    navigator->setCheckable(true);
    navigator->setShortcut(QKeySequence(Qt::Key_F9));
    registerShortcut(navigator, QStringLiteral("show_navigator"), tr("View"));
    connect(navigator, &QAction::toggled, m_navigatorDock,
            &QDockWidget::setVisible);
    connect(m_navigatorDock, &QDockWidget::visibilityChanged, navigator,
            &QAction::setChecked);
    m_tagsAction = viewMenu->addAction(tr("Image Tags"));
    m_tagsAction->setCheckable(true);
    m_tagsAction->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_T));
    registerShortcut(m_tagsAction, QStringLiteral("toggle_image_tags"),
                     tr("View"));
    connect(m_tagsAction, &QAction::toggled, this, [this](bool visible) {
        m_tagsVisibilityExplicit = true;
        m_tagsPanel->setVisible(visible);
    });
    // The source exposes Compare View only through the File menu; adding it
    // to a second container would duplicate the entry and its shortcut.
    m_compareAction = new QAction(tr("Compare View"), this);
    m_compareAction->setCheckable(true);
    m_compareAction->setShortcut(QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_C));
    registerShortcut(m_compareAction, QStringLiteral("toggle_compare_view"),
                     tr("View"));
    connect(m_compareAction, &QAction::triggered, this,
            &AnnotationWindow::toggleCompareView);

    QAction* fillDrawing =
            viewMenu->addAction(QIcon(iconPath(QStringLiteral("color.png"))),
                                tr("Fill Drawing Polygon"));
    fillDrawing->setCheckable(true);
    fillDrawing->setChecked(
            QSettings()
                    .value(QStringLiteral("qAnyLabeling/fillDrawing"), true)
                    .toBool());
    fillDrawing->setToolTip(tr("Fill polygon while drawing"));
    connect(fillDrawing, &QAction::toggled, this, [this](bool enabled) {
        m_canvas->setFillDrawing(enabled);
        QSettings().setValue(QStringLiteral("qAnyLabeling/fillDrawing"),
                             enabled);
    });
    m_canvas->setFillDrawing(fillDrawing->isChecked());

    // The source File menu has a stable, user-facing order that is separate
    // from the order in which QAction objects are created for the toolbar.
    // Rebuild the menu from the same actions so a shortcut still has one
    // owner while menu placement matches the upstream workspace.
    for (QAction* action : fileMenu->actions()) fileMenu->removeAction(action);
    fileMenu->addAction(openImageAction);
    fileMenu->addAction(nextAction);
    fileMenu->addAction(previousAction);
    fileMenu->addAction(nextUncheckedAction);
    fileMenu->addAction(previousUncheckedAction);
    fileMenu->addAction(openDir);
    fileMenu->addAction(openVideo);
    fileMenu->addAction(m_compareAction);
    fileMenu->addMenu(m_recentMenu);
    fileMenu->addAction(m_saveAction);
    fileMenu->addAction(saveAsAction);

    QAction* saveAutomatically = fileMenu->addAction(tr("Save Automatically"));
    saveAutomatically->setCheckable(true);
    saveAutomatically->setChecked(m_autoSave);
    saveAutomatically->setToolTip(tr("Save annotations after each edit"));
    connect(saveAutomatically, &QAction::toggled, this, [this](bool enabled) {
        m_autoSave = enabled;
        QSettings().setValue(QStringLiteral("qAnyLabeling/autoSave"), enabled);
        if (enabled && m_dirty) saveDocument();
    });

    QAction* changeOutputDirectory = fileMenu->addAction(
            QIcon(iconPath(QStringLiteral("open.png"))),
            tr("Change Output Dir"), this, &AnnotationWindow::changeOutputDir);
    changeOutputDirectory->setObjectName(
            QStringLiteral("AnnotationChangeOutputDirectory"));
    changeOutputDirectory->setToolTip(
            tr("Change where annotations are loaded/saved"));

    QAction* saveWithImageData =
            fileMenu->addAction(tr("Save With Image Data"));
    saveWithImageData->setCheckable(true);
    saveWithImageData->setChecked(m_saveWithImageData);
    saveWithImageData->setToolTip(
            tr("Embed the current image as PNG data in the JSON sidecar"));
    connect(saveWithImageData, &QAction::toggled, this, [this](bool enabled) {
        m_saveWithImageData = enabled;
        QSettings().setValue(QStringLiteral("qAnyLabeling/storeData"), enabled);
        if (m_dirty && !m_imagePath.isEmpty()) saveDocument();
    });
    fileMenu->addAction(closeAction);
    fileMenu->addAction(deleteLabelAction);
    fileMenu->addAction(deleteImageAction);
    fileMenu->addSeparator();

    // The source keeps the Auto Labeling action in the left toolbar only.
    QAction* autoLabelPanel = new QAction(tr("Auto Labeling"), this);
    autoLabelPanel->setIcon(QIcon(iconPath(QStringLiteral("brain.png"))));
    autoLabelPanel->setCheckable(true);
    autoLabelPanel->setChecked(m_modelPanel && m_modelPanel->isVisible());
    autoLabelPanel->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_A));
    registerShortcut(autoLabelPanel, QStringLiteral("auto_label"), tr("AI"));
    connect(autoLabelPanel, &QAction::toggled, m_modelPanel,
            &QWidget::setVisible);
    connect(autoLabelPanel, &QAction::toggled, this,
            [this](bool) { updateActionState(); });
    toolbarActions.insert(QStringLiteral("auto_label"), autoLabelPanel);
    m_zoomOriginalAction = viewMenu->addAction(tr("Original Size"));
    QAction* zoomOriginal = m_zoomOriginalAction;
    zoomOriginal->setIcon(QIcon(iconPath(QStringLiteral("zoom.png"))));
    zoomOriginal->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Equal));
    registerShortcut(zoomOriginal, QStringLiteral("zoom_to_original"),
                     tr("View"));
    connect(zoomOriginal, &QAction::triggered, m_canvas,
            &AnnotationCanvas::zoomOriginal);
    toolbarActions.insert(QStringLiteral("zoom_original"), zoomOriginal);
    m_fitWidthAction = viewMenu->addAction(tr("Fit Width"));
    QAction* fitWidth = m_fitWidthAction;
    fitWidth->setIcon(QIcon(iconPath(QStringLiteral("fit-width.png"))));
    fitWidth->setShortcut(QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_F));
    registerShortcut(fitWidth, QStringLiteral("fit_width"), tr("View"));
    connect(fitWidth, &QAction::triggered, m_canvas,
            &AnnotationCanvas::fitWidth);
    toolbarActions.insert(QStringLiteral("fit_width"), fitWidth);

    // X-AnyLabeling uses a compact, buttonless percentage spin box as the
    // final toolbar control. Keep the same interaction instead of exposing
    // the adapter's "Original Size" action in the source toolbar.
    m_zoomWidget = new QSpinBox(fileBar);
    m_zoomWidget->setObjectName(QStringLiteral("AnnotationZoomWidget"));
    m_zoomWidget->setButtonSymbols(QAbstractSpinBox::NoButtons);
    m_zoomWidget->setRange(1, 1000);
    m_zoomWidget->setValue(100);
    m_zoomWidget->setSuffix(QStringLiteral("%"));
    m_zoomWidget->setAlignment(Qt::AlignCenter);
    QFont zoomFont = m_zoomWidget->font();
    zoomFont.setPointSize(9);
    m_zoomWidget->setFont(zoomFont);
    m_zoomWidget->setToolTip(tr("Zoom Level"));
    m_zoomWidget->setStatusTip(m_zoomWidget->toolTip());
    m_zoomWidget->setStyleSheet(
            QStringLiteral("QSpinBox { background: transparent; border: none; "
                           "padding: 0; min-height: 0; }"));
    m_zoomWidget->setFixedWidth(40);
    m_zoomWidget->setEnabled(!m_image.isNull());
    auto* zoomWidgetAction = new QWidgetAction(fileBar);
    zoomWidgetAction->setObjectName(QStringLiteral("AnnotationZoomAction"));
    zoomWidgetAction->setDefaultWidget(m_zoomWidget);
    connect(m_zoomWidget, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                if (m_canvas) m_canvas->setZoomFactor(value / 100.0);
            });
    connect(m_canvas, &AnnotationCanvas::viewChanged, this, [this]() {
        if (!m_zoomWidget || !m_canvas) return;
        const QSignalBlocker blocker(m_zoomWidget);
        m_zoomWidget->setValue(qRound(m_canvas->zoomFactor() * 100.0));
    });
    viewMenu->addSeparator();
    m_keepBrightnessAction =
            viewMenu->addAction(tr("Keep Previous Brightness"));
    m_keepBrightnessAction->setCheckable(true);
    m_keepBrightnessAction->setChecked(m_keepPreviousBrightness);
    connect(m_keepBrightnessAction, &QAction::toggled, this,
            [this](bool enabled) {
                m_keepPreviousBrightness = enabled;
                QSettings settings;
                settings.setValue(
                        QStringLiteral("qAnyLabeling/keepPreviousBrightness"),
                        enabled);
            });
    m_keepContrastAction = viewMenu->addAction(tr("Keep Previous Contrast"));
    m_keepContrastAction->setCheckable(true);
    m_keepContrastAction->setChecked(m_keepPreviousContrast);
    connect(m_keepContrastAction, &QAction::toggled, this,
            [this](bool enabled) {
                m_keepPreviousContrast = enabled;
                QSettings settings;
                settings.setValue(
                        QStringLiteral("qAnyLabeling/keepPreviousContrast"),
                        enabled);
            });
    QAction* keepPreviousScale = viewMenu->addAction(tr("Keep Previous Scale"));
    keepPreviousScale->setCheckable(true);
    keepPreviousScale->setChecked(m_keepPreviousScale);
    keepPreviousScale->setToolTip(tr("Keep previous zoom scale"));
    connect(keepPreviousScale, &QAction::toggled, this, [this](bool enabled) {
        m_keepPreviousScale = enabled;
        QSettings().setValue(QStringLiteral("qAnyLabeling/keepPreviousScale"),
                             enabled);
    });
    QAction* toggleKeepPrevious =
            viewMenu->addAction(tr("Toggle keep previous image adjustments"));
    toggleKeepPrevious->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_P));
    registerShortcut(toggleKeepPrevious,
                     QStringLiteral("toggle_keep_prev_mode"), tr("View"));
    connect(toggleKeepPrevious, &QAction::triggered, this, [this]() {
        const bool keep = !(m_keepBrightnessAction->isChecked() &&
                            m_keepContrastAction->isChecked());
        m_keepBrightnessAction->setChecked(keep);
        m_keepContrastAction->setChecked(keep);
    });
    m_autoUseLastLabelAction = viewMenu->addAction(tr("Auto use last label"));
    m_autoUseLastLabelAction->setCheckable(true);
    m_autoUseLastLabelAction->setChecked(m_autoUseLastLabel);
    m_autoUseLastLabelAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Y));
    registerShortcut(m_autoUseLastLabelAction,
                     QStringLiteral("toggle_auto_use_last_label"), tr("View"));
    connect(m_autoUseLastLabelAction, &QAction::toggled, this,
            [this](bool enabled) {
                m_autoUseLastLabel = enabled;
                QSettings settings;
                settings.setValue(
                        QStringLiteral("qAnyLabeling/autoUseLastLabel"),
                        enabled);
            });
    m_showShapesAction = viewMenu->addAction(tr("Show Shapes"));
    m_showShapesAction->setCheckable(true);
    {
        QSettings settings;
        m_showShapesAction->setChecked(
                settings.value(QStringLiteral("qAnyLabeling/showShapes"), true)
                        .toBool());
    }
    m_showShapesAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));
    registerShortcut(m_showShapesAction,
                     QStringLiteral("toggle_visibility_shapes"), tr("View"));
    connect(m_showShapesAction, &QAction::toggled, this, [this](bool visible) {
        m_canvas->setShowShapes(visible);
        QSettings settings;
        settings.setValue(QStringLiteral("qAnyLabeling/showShapes"), visible);
    });
    m_showMasksAction = viewMenu->addAction(tr("Show Masks"));
    m_showMasksAction->setCheckable(true);
    {
        QSettings settings;
        m_showMasksAction->setChecked(
                settings.value(QStringLiteral("qAnyLabeling/showMasks"), true)
                        .toBool());
    }
    m_showMasksAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_M));
    registerShortcut(m_showMasksAction, QStringLiteral("show_masks"),
                     tr("View"));
    connect(m_showMasksAction, &QAction::toggled, this, [this](bool visible) {
        m_canvas->setShowMasks(visible);
        QSettings settings;
        settings.setValue(QStringLiteral("qAnyLabeling/showMasks"), visible);
    });
    m_showLabelsAction = viewMenu->addAction(tr("Show Labels"));
    m_showLabelsAction->setCheckable(true);
    {
        QSettings settings;
        m_showLabelsAction->setChecked(
                settings.value(QStringLiteral("qAnyLabeling/showLabels"), true)
                        .toBool());
    }
    m_showLabelsAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_L));
    registerShortcut(m_showLabelsAction, QStringLiteral("show_labels"),
                     tr("View"));
    connect(m_showLabelsAction, &QAction::toggled, this, [this](bool visible) {
        m_canvas->setShowLabels(visible);
        QSettings settings;
        settings.setValue(QStringLiteral("qAnyLabeling/showLabels"), visible);
    });
    m_canvas->setShowShapes(m_showShapesAction->isChecked());
    m_canvas->setShowMasks(m_showMasksAction->isChecked());
    m_canvas->setShowLabels(m_showLabelsAction->isChecked());
    const auto addDisplayAction =
            [this, viewMenu](const QString& label, const QString& setting,
                             bool defaultValue,
                             void (AnnotationCanvas::*setter)(bool)) {
                QAction* action = viewMenu->addAction(label);
                action->setCheckable(true);
                const QString key = QStringLiteral("qAnyLabeling/") + setting;
                action->setChecked(
                        QSettings().value(key, defaultValue).toBool());
                (m_canvas->*setter)(action->isChecked());
                connect(action, &QAction::toggled, this,
                        [this, key, setter](bool visible) {
                            (m_canvas->*setter)(visible);
                            QSettings().setValue(key, visible);
                        });
                return action;
            };
    m_showScoresAction =
            addDisplayAction(tr("Show Scores"), QStringLiteral("showScores"),
                             true, &AnnotationCanvas::setShowScores);
    m_showGroupsAction =
            addDisplayAction(tr("Show Groups"), QStringLiteral("showGroups"),
                             false, &AnnotationCanvas::setShowGroups);
    m_showTextsAction =
            addDisplayAction(tr("Show Texts"), QStringLiteral("showTexts"),
                             true, &AnnotationCanvas::setShowTexts);
    m_showTextsAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_T));
    registerShortcut(m_showTextsAction, QStringLiteral("show_texts"),
                     tr("View"));
    m_showAttributesAction = addDisplayAction(
            tr("Show Attributes"), QStringLiteral("showAttributes"), true,
            &AnnotationCanvas::setShowAttributes);
    m_showAttributesAction->setShortcut(
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_L));
    registerShortcut(m_showAttributesAction, QStringLiteral("show_attributes"),
                     tr("View"));
    m_showDegreesAction =
            addDisplayAction(tr("Show Degress"), QStringLiteral("showDegrees"),
                             false, &AnnotationCanvas::setShowDegrees);
    m_showLinkingAction = addDisplayAction(tr("Show KIE Linking"),
                                           QStringLiteral("showLinking"), true,
                                           &AnnotationCanvas::setShowLinking);
    m_showLinkingAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_K));
    registerShortcut(m_showLinkingAction, QStringLiteral("show_linking"),
                     tr("View"));
    QAction* closeCompare = m_compareBar->addAction(
            QIcon(iconPath(QStringLiteral("cancel.png"))),
            tr("Close Compare View"));
    connect(closeCompare, &QAction::triggered, this,
            [this]() { closeCompareView(); });
    m_settingsAction = viewMenu->addAction(tr("Settings"));
    m_settingsAction->setObjectName(QStringLiteral("AnnotationSettings"));
    m_settingsAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_0));
    registerShortcut(m_settingsAction, QStringLiteral("open_settings"),
                     tr("Dialog"));
    connect(m_settingsAction, &QAction::triggered, this,
            &AnnotationWindow::openSettings);

    QMenu* importMenu = menuBar()->addMenu(tr("Upload"));
    const auto addUnsupportedImport = [importMenu, this](const QString& text,
                                                         const QString& tip) {
        auto* action = importMenu->addAction(
                QIcon(iconPath(QStringLiteral("label.png"))), text);
        action->setEnabled(false);
        action->setToolTip(tip);
        action->setStatusTip(tip);
        action->setObjectName(
                QStringLiteral("AnnotationTODOImport_%1")
                        .arg(text.simplified().replace(QLatin1Char(' '),
                                                       QLatin1Char('_'))));
        return action;
    };
    QAction* imageFlags = importMenu->addAction(
            QIcon(iconPath(QStringLiteral("label.png"))), tr("Image Flags"),
            this, &AnnotationWindow::importImageFlags);
    imageFlags->setObjectName(QStringLiteral("AnnotationImportImageFlags"));
    imageFlags->setToolTip(tr("Upload Custom Image Flags File"));
    QAction* labelFlagsUpload = importMenu->addAction(
            QIcon(iconPath(QStringLiteral("label.png"))), tr("Label Flags"),
            this, &AnnotationWindow::uploadLabelFlags);
    labelFlagsUpload->setObjectName(
            QStringLiteral("AnnotationUploadLabelFlags"));
    labelFlagsUpload->setToolTip(tr("Upload Custom Label Flags File"));
    QAction* attributesUpload = importMenu->addAction(
            QIcon(iconPath(QStringLiteral("label.png"))), tr("Attributes"),
            this, &AnnotationWindow::uploadShapeAttributes);
    attributesUpload->setObjectName(
            QStringLiteral("AnnotationUploadAttributes"));
    attributesUpload->setToolTip(tr("Upload Custom Attributes File"));
    QAction* labelClassesUpload = importMenu->addAction(
            QIcon(iconPath(QStringLiteral("label.png"))), tr("Label Classes"),
            this, &AnnotationWindow::uploadLabelClasses);
    labelClassesUpload->setObjectName(
            QStringLiteral("AnnotationUploadLabelClasses"));
    labelClassesUpload->setToolTip(tr("Upload Custom Label Classes File"));
    importMenu->addSeparator();
    importMenu->addAction(tr("YOLO HBB"), this,
                          [this]() { importYoloLabels(YoloFormat::Boxes); });
    importMenu->addAction(tr("YOLO OBB"), this, [this]() {
        importYoloLabels(YoloFormat::OrientedBoxes);
    });
    importMenu->addAction(tr("YOLO Seg"), this,
                          [this]() { importYoloLabels(YoloFormat::Segments); });
    importMenu->addAction(tr("YOLO Pose"), this,
                          [this]() { importYoloLabels(YoloFormat::Pose); });
    importMenu->addSeparator();
    importMenu->addAction(tr("VOC Detection"), this,
                          [this]() { importVocDota(true); });
    importMenu->addAction(tr("VOC Segmentation"), this,
                          [this]() { importVocDota(true, true); });
    importMenu->addSeparator();
    importMenu->addAction(tr("COCO Detection"), this,
                          [this]() { importCocoLabels(CocoFormat::Boxes); });
    importMenu->addAction(tr("COCO Segmentation"), this,
                          [this]() { importCocoLabels(CocoFormat::Segments); });
    importMenu->addAction(tr("COCO Keypoints"), this,
                          [this]() { importCocoLabels(CocoFormat::Pose); });
    importMenu->addSeparator();
    importMenu->addAction(tr("DOTA"), this, [this]() { importVocDota(false); });
    importMenu->addAction(tr("MASK"), this, &AnnotationWindow::importMasks);
    importMenu->addAction(tr("MOT"), this, &AnnotationWindow::importMotLabels);
    importMenu->addAction(tr("ODVG"), this,
                          &AnnotationWindow::importOdvgLabels);
    addUnsupportedImport(tr("MM-Grounding-DINO"),
                         tr("TODO: AICore grounding text result contract"));
    importMenu->addSeparator();
    importMenu->addAction(tr("PPOCR Rec"), this,
                          [this]() { importPpocrLabels(PpocrMode::Rec); });
    importMenu->addAction(tr("PPOCR KIE"), this,
                          [this]() { importPpocrLabels(PpocrMode::Kie); });
    importMenu->addSeparator();
    importMenu->addAction(tr("VLM-R1 OVD"), this,
                          &AnnotationWindow::importVlmLabels);
    importMenu->addSeparator();
    importMenu->addAction(tr("SAM3 tracking archive"), this,
                          &AnnotationWindow::importSam3Archive);
    importMenu->addAction(tr("Open SAM3 archive workspace"), this,
                          &AnnotationWindow::openSam3ArchiveWorkspace);
    const QIcon importExportIcon(iconPath(QStringLiteral("label.png")));
    for (QAction* action : importMenu->actions()) {
        if (!action->isSeparator() && action->icon().isNull())
            action->setIcon(importExportIcon);
    }
    QMenu* exportMenu = menuBar()->addMenu(tr("Export"));
    exportMenu->addAction(tr("YOLO HBB"), this,
                          [this]() { exportYolo(YoloFormat::Boxes); });
    exportMenu->addAction(tr("YOLO OBB"), this,
                          [this]() { exportYolo(YoloFormat::OrientedBoxes); });
    exportMenu->addAction(tr("YOLO Seg"), this,
                          [this]() { exportYolo(YoloFormat::Segments); });
    exportMenu->addAction(tr("YOLO Pose"), this,
                          [this]() { exportYolo(YoloFormat::Pose); });
    exportMenu->addSeparator();
    exportMenu->addAction(tr("VOC Detection"), this,
                          [this]() { exportVocDota(true); });
    exportMenu->addAction(tr("VOC Segmentation"), this,
                          [this]() { exportVocDota(true, true); });
    exportMenu->addSeparator();
    exportMenu->addAction(tr("COCO Detection"), this,
                          [this]() { exportCoco(false); });
    exportMenu->addAction(tr("COCO Segmentation"), this,
                          [this]() { exportCoco(true); });
    exportMenu->addAction(tr("COCO Keypoints"), this,
                          [this]() { exportCoco(false, true); });
    exportMenu->addSeparator();
    exportMenu->addAction(tr("DOTA"), this, [this]() { exportVocDota(false); });
    exportMenu->addAction(tr("MASK"), this, &AnnotationWindow::exportMasks);
    exportMenu->addAction(tr("ODVG"), this, &AnnotationWindow::exportOdvg);
    exportMenu->addSeparator();
    exportMenu->addAction(tr("MOT"), this, [this]() { exportMot(); });
    exportMenu->addAction(tr("MOTS"), this, [this]() { exportMot(true); });
    exportMenu->addSeparator();
    exportMenu->addAction(tr("PPOCR Rec"), this,
                          [this]() { exportPpocr(PpocrMode::Rec); });
    exportMenu->addAction(tr("PPOCR KIE"), this,
                          [this]() { exportPpocr(PpocrMode::Kie); });
    exportMenu->addSeparator();
    exportMenu->addAction(tr("VLM-R1 OVD"), this, &AnnotationWindow::exportVlm);
    for (QAction* action : exportMenu->actions()) {
        if (!action->isSeparator() && action->icon().isNull())
            action->setIcon(importExportIcon);
    }

    // Preserve the upstream top-level order. Commands without a complete
    // local implementation remain visible as explicit TODO entries.
    QMenu* toolMenu = menuBar()->addMenu(tr("Tool"));
    QMenu* trainMenu = menuBar()->addMenu(tr("Train"));
    QMenu* helpMenu = menuBar()->addMenu(tr("Help"));
    auto* cropAction = new QAction(tr("Save Cropped Image"), this);
    connect(cropAction, &QAction::triggered, this,
            &AnnotationWindow::saveCroppedImage);
    cropAction->setToolTip(tr("Export one crop per visible closed shape"));
    auto* visualizationAction =
            new QAction(tr("Save Visualization Image"), this);
    connect(visualizationAction, &QAction::triggered, this,
            &AnnotationWindow::saveVisualizationImage);
    visualizationAction->setToolTip(
            tr("Export the current image with annotation overlays"));
    auto* visualizationVideoAction =
            new QAction(tr("Save Visualization Video"), this);
    visualizationVideoAction->setEnabled(false);
    visualizationVideoAction->setToolTip(
            tr("TODO: upstream visualization video export workflow"));
    auto* trainAction = trainMenu->addAction(tr("Ultralytics"));
    trainAction->setEnabled(false);
    trainAction->setToolTip(tr("TODO: Ultralytics training UI and contract"));
    auto* documentation = helpMenu->addAction(tr("Documentation"));
    connect(documentation, &QAction::triggered, this, []() {
        QDesktopServices::openUrl(QUrl(QStringLiteral(
                "https://github.com/Asher-1/X-AnyLabeling-GGML")));
    });
    auto* sponsor = helpMenu->addAction(tr("Sponsor"));
    connect(sponsor, &QAction::triggered, this, []() {
        QDesktopServices::openUrl(QUrl(QStringLiteral(
                "https://github.com/Asher-1/X-AnyLabeling-GGML")));
    });
    auto* about = helpMenu->addAction(tr("About"));
    connect(about, &QAction::triggered, this, [this]() {
        QMessageBox::about(this, tr("About X-AnyLabeling"),
                           tr("qAnyLabeling\nC++/Qt adapter for ACloudViewer\n"
                              "Powered by AICore GGML tasks."));
    });

    m_undoAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("undo.png"))), tr("Undo"), this,
            &AnnotationWindow::undo, QKeySequence::Undo);
    registerShortcut(m_undoAction, QStringLiteral("undo"), tr("Shape"));
    toolbarActions.insert(QStringLiteral("undo"), m_undoAction);
    m_redoAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("redo.svg"))), tr("Redo"), this,
            &AnnotationWindow::redo, QKeySequence::Redo);
    // X-AnyLabeling has one Ctrl+Z action which undoes a drawing vertex first
    // and an annotation edit otherwise. Keep redo available from the menu so
    // it does not steal the upstream Ctrl+Y action.
    m_redoAction->setShortcut(QKeySequence());
    QAction* deleteAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("delete.png"))), tr("Delete object"),
            m_canvas, &AnnotationCanvas::deleteSelected, QKeySequence::Delete);
    registerShortcut(deleteAction, QStringLiteral("delete_polygon"),
                     tr("Shape"));
    toolbarActions.insert(QStringLiteral("delete_polygon"), deleteAction);
    m_deletePolygonAction = deleteAction;
    deleteAction->setEnabled(false);
    QAction* copy = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("copy.png"))), tr("Copy objects"),
            this, &AnnotationWindow::copyShapes, QKeySequence::Copy);
    registerShortcut(copy, QStringLiteral("copy_polygon"), tr("Shape"));
    QAction* paste = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("paste.png"))), tr("Paste objects"),
            this, &AnnotationWindow::pasteShapes, QKeySequence::Paste);
    registerShortcut(paste, QStringLiteral("paste_polygon"), tr("Shape"));
    m_pasteAction = paste;
    paste->setEnabled(false);
    QAction* editLabel = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("edit.png"))), tr("Edit label"), this,
            &AnnotationWindow::editSelectedLabel,
            QKeySequence(Qt::CTRL + Qt::Key_E));
    registerShortcut(editLabel, QStringLiteral("edit_label"), tr("Dialog"));
    QAction* labelManager = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("labels.png"))), tr("Label Manager"),
            this, &AnnotationWindow::editLabels,
            QKeySequence(Qt::ALT + Qt::Key_L));
    labelManager->setObjectName(QStringLiteral("AnnotationLabelManager"));
    registerShortcut(labelManager, QStringLiteral("edit_labels"), tr("Dialog"));
    QAction* duplicate = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("copy.png"))),
            tr("Duplicate objects"), this,
            [this]() {
                copyShapes();
                pasteShapes();
            },
            QKeySequence(Qt::CTRL + Qt::Key_D));
    registerShortcut(duplicate, QStringLiteral("duplicate_polygon"),
                     tr("Shape"));
    // Source Edit-menu entries: copy the selected shape's coordinates and
    // toggle using the system clipboard for shape copy/paste.
    m_copyCoordinatesAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("copy.png"))), tr("Copy Coordinates"),
            this, &AnnotationWindow::copyShapeCoordinates);
    m_copyCoordinatesAction->setObjectName(
            QStringLiteral("AnnotationCopyCoordinates"));
    registerShortcut(m_copyCoordinatesAction,
                     QStringLiteral("copy_coordinates"), tr("Shape"));
    m_copyCoordinatesAction->setEnabled(false);
    m_canvas->addAction(m_copyCoordinatesAction);
    m_objects->addAction(m_copyCoordinatesAction);
    m_useSystemClipboardAction =
            editMenu->addAction(tr("Use System Clipboard"));
    m_useSystemClipboardAction->setCheckable(true);
    m_useSystemClipboardAction->setChecked(m_systemClipboard);
    registerShortcut(m_useSystemClipboardAction,
                     QStringLiteral("use_system_clipboard"), tr("Shape"));
    connect(m_useSystemClipboardAction, &QAction::toggled, this,
            [this](bool enabled) {
                m_systemClipboard = enabled;
                QSettings().setValue(
                        QStringLiteral("qAnyLabeling/systemClipboard"),
                        enabled);
                if (m_pasteAction)
                    m_pasteAction->setEnabled(
                            !m_imagePath.isEmpty() &&
                            (m_systemClipboard ||
                             !m_internalShapeClipboard.isEmpty()));
            });
    editMenu->addSeparator();
    auto setSelectedVisibility = [this](bool visible) {
        const QSet<int> selected = m_canvas->selectedIndices();
        if (selected.isEmpty()) return;
        QVector<AnnotationShape> shapes = m_canvas->shapes();
        bool changed = false;
        for (int row : selected) {
            if (row < 0 || row >= shapes.size() ||
                shapes[row].visible == visible)
                continue;
            shapes[row].visible = visible;
            changed = true;
        }
        if (!changed) return;
        snapshot();
        m_canvas->setShapes(shapes);
        m_canvas->setSelectedShapes(selected.values().toVector());
        updateObjects();
        setDirty(true);
    };
    QAction* hideSelected = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("hide-selected.png"))),
            tr("Hide selected polygons"),
            [setSelectedVisibility]() { setSelectedVisibility(false); },
            QKeySequence(Qt::Key_S));
    QAction* showHidden = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("show-hidden.png"))),
            tr("Show hidden polygons"),
            [this]() {
                QVector<AnnotationShape> shapes = m_canvas->shapes();
                bool changed = false;
                for (AnnotationShape& shape : shapes) {
                    if (!shape.visible) {
                        shape.visible = true;
                        changed = true;
                    }
                }
                if (!changed) return;
                const QVector<int> selected =
                        m_canvas->selectedIndices().values().toVector();
                snapshot();
                m_canvas->setShapes(shapes);
                m_canvas->setSelectedShapes(selected);
                updateObjects();
                setDirty(true);
            },
            QKeySequence(Qt::Key_W));
    registerShortcut(hideSelected, QStringLiteral("hide_selected_polygons"),
                     tr("Shape"));
    registerShortcut(showHidden, QStringLiteral("show_hidden_polygons"),
                     tr("Shape"));
    toolbarActions.insert(QStringLiteral("hide_selected_polygons"),
                          hideSelected);
    toolbarActions.insert(QStringLiteral("show_hidden_polygons"), showHidden);
    m_selectToggleShapesAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("eye.svg"))),
            tr("Toggle shapes visibility"), this, [this]() {
                const QVector<AnnotationShape> current = m_canvas->shapes();
                if (current.isEmpty()) return;
                const bool allVisible =
                        std::all_of(current.cbegin(), current.cend(),
                                    [](const AnnotationShape& shape) {
                                        return shape.visible;
                                    });
                QVector<AnnotationShape> updated = current;
                for (AnnotationShape& shape : updated)
                    shape.visible = !allVisible;
                const QVector<int> selected =
                        m_canvas->selectedIndices().values().toVector();
                snapshot();
                m_canvas->setShapes(updated);
                m_canvas->setSelectedShapes(selected);
                updateObjects();
                setDirty(true);
            });
    toolbarActions.insert(QStringLiteral("select_toggle_shapes"),
                          m_selectToggleShapesAction);
    QAction* groupAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("group-shapes.png"))),
            tr("Group selected shapes"), this,
            &AnnotationWindow::groupSelectedShapes, QKeySequence(Qt::Key_G));
    registerShortcut(groupAction, QStringLiteral("group_selected_shapes"),
                     tr("Shape"));
    QAction* ungroupAction = editMenu->addAction(
            QIcon(iconPath(QStringLiteral("ungroup-shapes.png"))),
            tr("Ungroup selected shapes"), this,
            &AnnotationWindow::ungroupSelectedShapes, QKeySequence(Qt::Key_U));
    registerShortcut(ungroupAction, QStringLiteral("ungroup_selected_shapes"),
                     tr("Shape"));
    QAction* unionAction =
            editMenu->addAction(QIcon(iconPath(QStringLiteral("union.png"))),
                                tr("Union selected objects"), this,
                                &AnnotationWindow::unionSelectedShapes,
                                QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_M));
    registerShortcut(unionAction, QStringLiteral("union_selected_shapes"),
                     tr("Shape"));
    QAction* digitSettings =
            editMenu->addAction(tr("Digit Shortcut Manager"), this,
                                &AnnotationWindow::configureDigitShortcuts);
    digitSettings->setShortcut(QKeySequence(Qt::ALT + Qt::Key_D));
    registerShortcut(digitSettings, QStringLiteral("edit_digit_shortcut"),
                     tr("Dialog"));
    QAction* overviewAction = new QAction(tr("Overview"), this);
    overviewAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_G));
    registerShortcut(overviewAction, QStringLiteral("show_overview"),
                     tr("View"));
    connect(overviewAction, &QAction::triggered, this, [this]() {
        m_navigatorDock->show();
        m_navigatorDock->raise();
        updateNavigator();
    });
    const auto addShortcutAction =
            [this, &registerShortcut](
                    QMenu* menu, const QString& text, const QString& id,
                    const QKeySequence& key, const QString& category,
                    const std::function<void()>& callback) {
                QAction* action = menu->addAction(text);
                action->setShortcut(key);
                registerShortcut(action, id, category);
                connect(action, &QAction::triggered, this, callback);
                return action;
            };
    m_undoLastPointAction = addShortcutAction(
            editMenu, tr("Undo last point"), QStringLiteral("undo_last_point"),
            QKeySequence::Undo, tr("Shape"),
            [this]() { m_canvas->undoLastPoint(); });
    m_undoLastPointAction->setEnabled(false);
    connect(m_canvas, &AnnotationCanvas::editStarted, this,
            [this]() { m_undoLastPointAction->setEnabled(true); });
    connect(m_canvas, &AnnotationCanvas::shapeFinished, this,
            [this](int) { m_undoLastPointAction->setEnabled(false); });
    connect(m_canvas, &AnnotationCanvas::creationCanceled, this,
            [this]() { m_undoLastPointAction->setEnabled(false); });
    QAction* addPointAction = addShortcutAction(
            editMenu, tr("Add point to edge"),
            QStringLiteral("add_point_to_edge"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_P), tr("Shape"),
            [this]() { m_canvas->addPointToEdge(); });
    QAction* removePointAction =
            addShortcutAction(editMenu, tr("Remove selected point"),
                              QStringLiteral("remove_selected_point"),
                              QKeySequence(Qt::Key_Backspace), tr("Shape"),
                              [this]() { m_canvas->removeSelectedPoint(); });
    removePointAction->setEnabled(false);
    // Vertex-level enablement matches the source vertex_selected signal.
    connect(m_canvas, &AnnotationCanvas::vertexHovered, removePointAction,
            &QAction::setEnabled);
    connect(m_canvas, &AnnotationCanvas::shapeSelected, removePointAction,
            [removePointAction](int) { removePointAction->setEnabled(false); });
    QAction* gidManager = addShortcutAction(
            editMenu, tr("Group ID Manager"), QStringLiteral("edit_group_id"),
            QKeySequence(Qt::ALT + Qt::Key_G), tr("Dialog"), [this]() {
                const QVector<int> selected =
                        m_canvas->selectedIndices().values().toVector();
                if (selected.isEmpty()) return;
                bool accepted = false;
                const int group = QInputDialog::getInt(
                        this, tr("Set group ID"), tr("Group ID"), 0, 0,
                        std::numeric_limits<int>::max(), 1, &accepted);
                if (!accepted) return;
                snapshot();
                QVector<AnnotationShape> shapes = m_canvas->shapes();
                for (int row : selected)
                    if (row >= 0 && row < shapes.size() && !shapes[row].locked)
                        shapes[row].extra.insert(QStringLiteral("group_id"),
                                                 group);
                m_canvas->setShapes(shapes);
                updateObjects();
                setDirty(true);
            });
    QAction* shapeManager = addShortcutAction(
            editMenu, tr("Shape Manager"), QStringLiteral("edit_shapes"),
            QKeySequence(Qt::ALT + Qt::Key_S), tr("Dialog"), []() {});
    shapeManager->setEnabled(false);
    shapeManager->setToolTip(tr("TODO: upstream shape manager dialog"));
    toolMenu->addAction(overviewAction);
    toolMenu->addSeparator();
    toolMenu->addAction(cropAction);
    toolMenu->addAction(visualizationAction);
    toolMenu->addAction(visualizationVideoAction);
    toolMenu->addSeparator();
    toolMenu->addAction(digitSettings);
    toolMenu->addAction(labelManager);
    toolMenu->addAction(gidManager);
    toolMenu->addAction(shapeManager);
    toolMenu->addSeparator();
    auto* shapeConverter = toolMenu->addAction(tr("Shape Converter"));
    connect(shapeConverter, &QAction::triggered, this,
            &AnnotationWindow::openShapeConverter);
    shapeConverter->setToolTip(tr("Convert compatible annotation geometry"));
    auto* pointCloudAction = toolMenu->addAction(
            QIcon(iconPath(QStringLiteral("cartesian.png"))),
            tr("Point cloud annotation"), this,
            &AnnotationWindow::openPointCloudAnnotation);
    pointCloudAction->setToolTip(
            tr("Open the point cloud annotation workspace"));
    pointCloudAction->setProperty("qAnyToolbarId",
                                  QStringLiteral("open_pointcloud"));
    toolbarActions.insert(QStringLiteral("open_pointcloud"), pointCloudAction);
    if (QAction* videoClassifier =
                toolbarActions.value(QStringLiteral("open_video_classifier")))
        videoClassifier->setIcon(QIcon(iconPath(QStringLiteral("cavy.png"))));
    QAction* autoUseLastGidAction = addShortcutAction(
            viewMenu, tr("Toggle auto use last group ID"),
            QStringLiteral("toggle_auto_use_last_gid"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_G), tr("View"),
            [this]() {
                QSettings settings;
                const bool enabled =
                        !settings.value(QStringLiteral(
                                                "qAnyLabeling/autoUseLastGid"),
                                        false)
                                 .toBool();
                settings.setValue(QStringLiteral("qAnyLabeling/autoUseLastGid"),
                                  enabled);
                m_status->setText(enabled ? tr("Last group ID reuse enabled")
                                          : tr("Last group ID reuse disabled"));
            });
    QAction* loopThroughLabels = addShortcutAction(
            viewMenu, tr("Loop Through Labels"),
            QStringLiteral("loop_thru_labels"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_N), tr("View"),
            [this]() {
                // Source review loop: the list is snapshotted on the
                // first trigger, each step zooms to the shape, and
                // finishing clears the selection and restores the
                // fitted view. Deletions during the loop are skipped.
                const QVector<AnnotationShape> shapes = m_canvas->shapes();
                if (!m_labelLoopActive) {
                    if (shapes.isEmpty()) {
                        m_labelLoopPopup->showMessage(
                                tr("No objects to review"));
                        return;
                    }
                    m_labelLoopActive = true;
                    m_labelLoopCount = -1;
                    m_labelLoopTotal = shapes.size();
                }
                ++m_labelLoopCount;
                while (m_labelLoopCount < m_labelLoopTotal &&
                       m_labelLoopCount >= shapes.size())
                    ++m_labelLoopCount;
                if (m_labelLoopCount >= m_labelLoopTotal) {
                    m_labelLoopActive = false;
                    m_labelLoopCount = -1;
                    m_canvas->selectShape(-1);
                    m_canvas->fitImage();
                    m_labelLoopPopup->showMessage(tr("Review complete"));
                    return;
                }
                const AnnotationShape& shape = shapes[m_labelLoopCount];
                QRectF bounds;
                for (const QPointF& point : shape.points) {
                    bounds = bounds.isNull()
                                     ? QRectF(point, point)
                                     : bounds.united(QRectF(point, point));
                }
                m_canvas->selectShape(m_labelLoopCount);
                m_canvas->zoomToRect(bounds, 4.0);
                m_labelLoopPopup->showMessage(tr("Reviewing %1 / %2 - %3")
                                                      .arg(m_labelLoopCount + 1)
                                                      .arg(m_labelLoopTotal)
                                                      .arg(shape.label));
            });
    m_loopThroughLabelsAction = loopThroughLabels;
    loopThroughLabels->setIcon(QIcon(iconPath(QStringLiteral("loop.png"))));
    toolbarActions.insert(QStringLiteral("loop_thru_labels"),
                          loopThroughLabels);
    QAction* loopSelectLabels = addShortcutAction(
            viewMenu, tr("Loop Select Labels"),
            QStringLiteral("loop_select_labels"),
            QKeySequence(Qt::CTRL + Qt::SHIFT + Qt::Key_C), tr("View"),
            [this]() {
                // Source semantics: walk the object list; reaching
                // the end clears the selection and restarts.
                const int count = m_canvas->shapes().size();
                ++m_selectLoopCount;
                if (count == 0 || m_selectLoopCount >= count) {
                    m_selectLoopCount = -1;
                    m_canvas->selectShape(-1);
                    return;
                }
                m_canvas->selectShape(m_selectLoopCount);
            });
    m_loopSelectLabelsAction = loopSelectLabels;
    loopSelectLabels->setIcon(
            QIcon(iconPath(QStringLiteral("circle-selection.png"))));
    toolbarActions.insert(QStringLiteral("loop_select_labels"),
                          loopSelectLabels);
    addShortcutAction(viewMenu, tr("Add SAM point"),
                      QStringLiteral("auto_labeling_add_point"),
                      QKeySequence(Qt::Key_Q), tr("AI"), [this]() {
                          m_canvas->setTool(QStringLiteral("sam_point"));
                          updateInstruction();
                      });
    addShortcutAction(viewMenu, tr("Remove SAM point"),
                      QStringLiteral("auto_labeling_remove_point"),
                      QKeySequence(Qt::Key_E), tr("AI"), [this]() {
                          m_canvas->removeLastPromptPoint(true);
                          updateInstruction();
                      });
    addShortcutAction(viewMenu, tr("Run auto labeling"),
                      QStringLiteral("auto_labeling_run"),
                      QKeySequence(Qt::Key_I), tr("AI"),
                      [this]() { runDetection(); });
    addShortcutAction(viewMenu, tr("Clear auto labeling prompts"),
                      QStringLiteral("auto_labeling_clear"),
                      QKeySequence(Qt::Key_B), tr("AI"),
                      [this]() { m_canvas->clearPrompt(); });
    addShortcutAction(viewMenu, tr("Finish auto labeling object"),
                      QStringLiteral("auto_labeling_finish_object"),
                      QKeySequence(Qt::Key_F), tr("AI"), [this]() {
                          runDetection();
                          m_canvas->clearPrompt();
                      });
    // Entries whose upstream features have no AICore contract yet keep their
    // official source names and icons, but are explicitly disabled instead of
    // exposing a fake behavior.
    const auto addUnsupported = [this, viewMenu, &registerShortcut](
                                        const QString& text, const QString& id,
                                        const QKeySequence& key) {
        QAction* action = new QAction(text, this);
        action->setObjectName(QStringLiteral("AnnotationTODO_") + id);
        action->setShortcut(key);
        registerShortcut(action, id, tr("Dialog"));
        action->setEnabled(false);
        const QString tip =
                tr("TODO: reserved for a future AICore contract; see "
                   "qAnyLabeling/TODO.md.");
        action->setToolTip(tip);
        action->setStatusTip(tip);
        viewMenu->addAction(action);
        return action;
    };
    const auto addUnsupportedToolbarCommand =
            [&addUnsupported, &toolbarActions, this](
                    const QString& text, const QString& id, const QString& icon,
                    const QKeySequence& key) {
                QAction* action = addUnsupported(text, id, key);
                action->setIcon(QIcon(iconPath(icon)));
                action->setProperty("qAnyToolbarId", id);
                toolbarActions.insert(id, action);
            };
    addUnsupportedToolbarCommand(tr("ChatBot"), QStringLiteral("open_chatbot"),
                                 QStringLiteral("psyduck.png"),
                                 QKeySequence(Qt::CTRL + Qt::Key_1));
    addUnsupportedToolbarCommand(tr("VQA"), QStringLiteral("open_vqa"),
                                 QStringLiteral("husky.png"),
                                 QKeySequence(Qt::CTRL + Qt::Key_2));
    addUnsupportedToolbarCommand(
            tr("Classifier"), QStringLiteral("open_image_classifier"),
            QStringLiteral("ragdoll.png"), QKeySequence(Qt::CTRL + Qt::Key_3));
    // PaddleOCR runs inside this window through the AICore ocr C API: the
    // picker entry pairs a rec GGUF with its det partner, and this action
    // selects the PP-OCR family (if not active) and runs over the current
    // image, creating one text annotation per recognized region.
    QAction* paddleOcr = viewMenu->addAction(tr("PaddleOCR"), this, [this]() {
        QString ocrFamily;
        for (const Model& model : m_models) {
            if (model.task == QStringLiteral("ocr")) {
                ocrFamily = model.family;
                break;
            }
        }
        if (ocrFamily.isEmpty()) {
            QMessageBox::information(this, tr("PaddleOCR"),
                                     tr("No PP-OCR catalog entry is "
                                        "available."));
            return;
        }
        if (m_model->currentText() != ocrFamily)
            m_model->setCurrentText(ocrFamily);
        runDetection();
    });
    paddleOcr->setObjectName(QStringLiteral("AnnotationOpenPaddleOCR"));
    paddleOcr->setIcon(QIcon(iconPath(QStringLiteral("paddlepaddle.png"))));
    paddleOcr->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_4));
    paddleOcr->setStatusTip(
            tr("Run PP-OCR text detection and recognition on the image"));
    paddleOcr->setProperty("qAnyToolbarId", QStringLiteral("open_paddleocr"));
    toolbarActions.insert(QStringLiteral("open_paddleocr"), paddleOcr);
    QAction* lomaBridge = viewMenu->addAction(tr("LoMa two-image bridge..."));
    lomaBridge->setObjectName(QStringLiteral("AnnotationLomaBridge"));
    connect(lomaBridge, &QAction::triggered, this, [this]() {
        LomaBridgeDialog bridge(this);
        if (!m_imagePath.isEmpty()) bridge.setImage0(m_imagePath);
        if (bridge.exec() != QDialog::Accepted) return;
        const QVector<LomaCorrespondence> matches = bridge.acceptedMatches();
        if (matches.isEmpty()) return;
        if (m_imagePath != bridge.image0Path()) {
            QMessageBox::information(this, tr("LoMa bridge"),
                                     tr("The active image must be Image A to "
                                        "import correspondence points."));
            return;
        }
        snapshot();
        QVector<AnnotationShape> shapes = m_canvas->shapes();
        for (const LomaCorrespondence& match : matches) {
            AnnotationShape shape;
            shape.type = QStringLiteral("point");
            shape.label = tr("LoMa correspondence");
            shape.points = {match.image0};
            shape.extra.insert(QStringLiteral("peer_image"),
                               bridge.image1Path());
            shape.extra.insert(QStringLiteral("peer_x"), match.image1.x());
            shape.extra.insert(QStringLiteral("peer_y"), match.image1.y());
            shape.extra.insert(QStringLiteral("score"), double(match.score));
            shapes.append(shape);
        }
        m_canvas->setShapes(shapes);
        updateObjects();
        setDirty(true);
        m_status->setText(
                tr("Imported %1 LoMa correspondences").arg(matches.size()));
    });
    // Keep the same WindowShortcut scope as every other qAny action.  The
    // actions are still installed on the canvas/list widgets for context-menu
    // discoverability, but their activation is owned by this annotation
    // workspace only.
    copy->setShortcutContext(Qt::WindowShortcut);
    paste->setShortcutContext(Qt::WindowShortcut);
    m_canvas->addAction(copy);
    m_canvas->addAction(paste);
    m_canvas->addAction(editLabel);
    m_canvas->addAction(duplicate);
    m_canvas->addAction(hideSelected);
    m_canvas->addAction(showHidden);
    m_objects->addAction(copy);
    m_objects->addAction(paste);
    m_objects->addAction(editLabel);
    m_objects->addAction(duplicate);
    m_objects->addAction(hideSelected);
    m_objects->addAction(showHidden);
    m_undoAction->setEnabled(false);
    m_redoAction->setEnabled(false);

    auto* toolBar = fileBar;
    m_tools = new QActionGroup(this);
    m_tools->setExclusive(true);
    struct Tool {
        const char* type;
        const char* icon;
        const char* label;
    };
    const Tool tools[] = {
            {"edit", "edit.png", "Edit"},
            {"polygon", "polygon.png", "Polygon"},
            {"rectangle", "rectangle.png", "Rectangle"},
            {"rotation", "rotation.png", "Rotation"},
            {"quadrilateral", "quadrilateral.png", "Quadrilateral"},
            {"circle", "circle.png", "Circle"},
            {"line", "line.png", "Line"},
            {"point", "point.png", "Point"},
            {"linestrip", "line-strip.png", "Line strip"},
            {"magic_wand", "magic_wand.png", "Magic wand"},
            {"sam_point", "point.png", "SAM points"},
            {"sam_box", "rectangle.png", "SAM box"},
            {"cuboid", "cuboid.png", "Cuboid"}};
    const QHash<QString, QKeySequence> toolShortcuts = {
            {QStringLiteral("edit"), QKeySequence(Qt::CTRL + Qt::Key_J)},
            {QStringLiteral("polygon"), QKeySequence(Qt::Key_P)},
            {QStringLiteral("rectangle"), QKeySequence(Qt::Key_R)},
            {QStringLiteral("rotation"), QKeySequence(Qt::Key_O)},
            {QStringLiteral("quadrilateral"), QKeySequence(Qt::Key_T)},
            {QStringLiteral("circle"), QKeySequence(Qt::SHIFT + Qt::Key_C)},
            {QStringLiteral("line"), QKeySequence(Qt::SHIFT + Qt::Key_L)},
            {QStringLiteral("point"), QKeySequence(Qt::SHIFT + Qt::Key_P)},
            {QStringLiteral("linestrip"), QKeySequence(Qt::SHIFT + Qt::Key_S)},
            {QStringLiteral("magic_wand"), QKeySequence(Qt::SHIFT + Qt::Key_W)},
            {QStringLiteral("cuboid"), QKeySequence(Qt::CTRL + Qt::Key_R)}};
    const QHash<QString, QString> toolShortcutIds = {
            {QStringLiteral("edit"), QStringLiteral("edit_polygon")},
            {QStringLiteral("polygon"), QStringLiteral("create_polygon")},
            {QStringLiteral("rectangle"), QStringLiteral("create_rectangle")},
            {QStringLiteral("rotation"), QStringLiteral("create_rotation")},
            {QStringLiteral("quadrilateral"),
             QStringLiteral("create_quadrilateral")},
            {QStringLiteral("circle"), QStringLiteral("create_circle")},
            {QStringLiteral("line"), QStringLiteral("create_line")},
            {QStringLiteral("point"), QStringLiteral("create_point")},
            {QStringLiteral("linestrip"), QStringLiteral("create_linestrip")},
            {QStringLiteral("magic_wand"), QStringLiteral("create_magic_wand")},
            {QStringLiteral("cuboid"), QStringLiteral("create_cuboid")}};
    for (const Tool& tool : tools) {
        const QString type = QString::fromLatin1(tool.type);
        QAction* action =
                new QAction(QIcon(iconPath(QString::fromLatin1(tool.icon))),
                            tr(tool.label), this);
        action->setToolTip(tr(tool.label));
        action->setData(type);
        const QString toolbarId = QStringLiteral("tool_%1").arg(type);
        action->setProperty("qAnyToolbarId", toolbarId);
        toolbarActions.insert(toolbarId, action);
        action->setCheckable(true);
        if (toolShortcuts.contains(type)) {
            action->setShortcut(toolShortcuts.value(type));
            registerShortcut(action, toolShortcutIds.value(type), tr("Shape"));
            m_canvas->addAction(action);
        }
        m_tools->addAction(action);
        if (type == QStringLiteral("edit")) action->setChecked(true);
        connect(action, &QAction::triggered, this, [this, type]() {
            m_canvas->setTool(type);
            updateInstruction();
        });
    }
    for (int digit = 0; digit <= 9; ++digit) {
        QAction* shortcut = new QAction(QString::number(digit), this);
        shortcut->setShortcut(QKeySequence(QString::number(digit)));
        shortcut->setShortcutContext(Qt::WindowShortcut);
        m_canvas->addAction(shortcut);
        connect(shortcut, &QAction::triggered, this,
                [this, digit]() { activateDigitShortcut(digit); });
    }
    QAction* clearPrompt =
            new QAction(QIcon(iconPath(QStringLiteral("cancel.png"))),
                        tr("Clear SAM prompts"), this);
    clearPrompt->setProperty("qAnyToolbarId", QStringLiteral("clear_prompt"));
    toolbarActions.insert(QStringLiteral("clear_prompt"), clearPrompt);
    clearPrompt->setToolTip(tr("Clear SAM prompts"));
    connect(clearPrompt, &QAction::triggered, m_canvas,
            &AnnotationCanvas::clearPrompt);
    QAction* brush = new QAction(QIcon(iconPath(QStringLiteral("brush.png"))),
                                 tr("Brush edit"), this);
    brush->setProperty("qAnyToolbarId", QStringLiteral("edit_brush_mode"));
    toolbarActions.insert(QStringLiteral("edit_brush_mode"), brush);
    brush->setCheckable(true);
    brush->setShortcut(QKeySequence(Qt::SHIFT + Qt::Key_B));
    registerShortcut(brush, QStringLiteral("edit_brush_mode"), tr("Shape"));
    m_canvas->addAction(brush);
    brush->setToolTip(tr("Refine a mask; Ctrl erases, R switches paint/erase"));
    auto* brushSettings = new QMenu(tr("Brush settings"), toolBar);
    auto* brushSettingsWidget = new QWidget(brushSettings);
    auto* brushSettingsLayout = new QFormLayout(brushSettingsWidget);
    brushSettingsLayout->setContentsMargins(8, 8, 8, 8);
    m_brushRadius = new QSpinBox(brushSettingsWidget);
    m_brushRadius->setRange(1, 200);
    m_brushRadius->setValue(m_defaultBrushRadius);
    m_brushRadius->setSuffix(tr(" px"));
    m_brushRadius->setObjectName(QStringLiteral("AnnotationBrushRadius"));
    m_brushRadius->setToolTip(tr("Brush radius"));
    brushSettingsLayout->addRow(tr("Radius"), m_brushRadius);
    connect(m_brushRadius, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                m_canvas->setBrushRadius(value);
                m_defaultBrushRadius = value;
                QSettings settings;
                settings.setValue(QStringLiteral("qAnyLabeling/brushRadius"),
                                  value);
            });
    connect(m_canvas, &AnnotationCanvas::brushRadiusChanged, m_brushRadius,
            &QSpinBox::setValue);
    m_canvas->setBrushRadius(m_defaultBrushRadius);
    m_brushMode = new QComboBox(brushSettingsWidget);
    m_brushMode->addItem(tr("Paint"), 0);
    m_brushMode->addItem(tr("Erase"), 1);
    m_brushMode->addItem(tr("Invert"), 2);
    {
        QSettings settings;
        const int operation =
                settings.value(QStringLiteral("qAnyLabeling/brushOperation"), 0)
                        .toInt();
        const int index = m_brushMode->findData(operation);
        m_brushMode->setCurrentIndex(index >= 0 ? index : 0);
    }
    m_brushMode->setToolTip(tr("Brush operation"));
    m_brushMode->setObjectName(QStringLiteral("AnnotationBrushMode"));
    brushSettingsLayout->addRow(tr("Operation"), m_brushMode);
    m_brushTarget = new QComboBox(brushSettingsWidget);
    m_brushTarget->setEditable(true);
    m_brushTarget->setMinimumContentsLength(10);
    m_brushTarget->setToolTip(tr("Target class for the selected instance"));
    m_brushTarget->setObjectName(QStringLiteral("AnnotationBrushTarget"));
    brushSettingsLayout->addRow(tr("Target class"), m_brushTarget);
    auto* brushSettingsAction = new QWidgetAction(brushSettings);
    brushSettingsAction->setDefaultWidget(brushSettingsWidget);
    brushSettings->addAction(brushSettingsAction);
    auto* brushSettingsButton = new QToolButton(toolBar);
    brushSettingsButton->setIcon(
            QIcon(iconPath(QStringLiteral("settings.svg"))));
    brushSettingsButton->setToolTip(tr("Brush settings"));
    brushSettingsButton->setMenu(brushSettings);
    brushSettingsButton->setPopupMode(QToolButton::InstantPopup);
    connect(m_brushMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int index) {
                const int operation = m_brushMode->itemData(index).toInt();
                m_canvas->setBrushOperation(operation);
                QSettings settings;
                settings.setValue(QStringLiteral("qAnyLabeling/brushOperation"),
                                  operation);
            });
    connect(m_canvas, &AnnotationCanvas::brushOperationChanged, m_brushMode,
            &QComboBox::setCurrentIndex);
    m_canvas->setBrushOperation(m_brushMode->currentData().toInt());
    connect(m_brushTarget, &QComboBox::currentTextChanged, this,
            [this](const QString& label) {
                m_canvas->setBrushLabel(label);
                const int row = m_brushTarget->findText(label);
                m_canvas->setBrushClassId(
                        row >= 0 && m_brushTarget->itemData(row).isValid()
                                ? m_brushTarget->itemData(row).toInt()
                                : -1);
            });
    connect(brush, &QAction::toggled, this, [this, brush](bool enabled) {
        if (enabled) {
            m_canvas->setTool(QStringLiteral("edit"));
            m_tools->actions().first()->setChecked(true);
        }
        if (!m_canvas->setBrushEditing(enabled) && enabled)
            brush->setChecked(false);
        updateInstruction();
    });
    connect(m_canvas, &AnnotationCanvas::brushEditingChanged, brush,
            &QAction::setChecked);
    connect(m_canvas, &AnnotationCanvas::brushEditingChanged, this,
            [this](bool enabled) { m_objects->setEnabled(!enabled); });
    QAction* createBrush =
            editMenu->addAction(QIcon(iconPath(QStringLiteral("brush.png"))),
                                tr("Create brush polygon"));
    createBrush->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_N));
    createBrush->setProperty("qAnyToolbarId",
                             QStringLiteral("create_brush_polygon"));
    toolbarActions.insert(QStringLiteral("create_brush_polygon"), createBrush);
    registerShortcut(createBrush, QStringLiteral("create_brush_polygon"),
                     tr("Shape"));
    m_canvas->addAction(createBrush);
    connect(createBrush, &QAction::triggered, this, [this, brush]() {
        if (m_canvas->brushEditing()) return;
        // Ctrl+N is explicitly a new mask workflow, even when an existing
        // object is selected in the object list.
        m_canvas->selectShape(-1);
        QString label = m_brushTarget->currentText().trimmed();
        if (label.isEmpty() && m_labels->currentItem())
            label = m_labels->currentItem()->text().trimmed();
        if (label.isEmpty()) {
            bool accepted = false;
            label = QInputDialog::getText(this, tr("Create brush polygon"),
                                          tr("Label"), QLineEdit::Normal,
                                          QString(), &accepted)
                            .trimmed();
            if (!accepted || label.isEmpty()) return;
        }
        m_brushTarget->setEditText(label);
        m_canvas->setBrushLabel(label);
        const int row = m_brushTarget->findText(label);
        m_canvas->setBrushClassId(
                row >= 0 && m_brushTarget->itemData(row).isValid()
                        ? m_brushTarget->itemData(row).toInt()
                        : -1);
        brush->setChecked(true);
    });
    connect(m_canvas, &AnnotationCanvas::brushHistoryChanged, this,
            [this](bool canUndo, bool canRedo) {
                m_undoAction->setEnabled(
                        m_canvas->brushEditing() ? canUndo : !m_undo.isEmpty());
                m_redoAction->setEnabled(
                        m_canvas->brushEditing() ? canRedo : !m_redo.isEmpty());
            });
    QAction* fit =
            new QAction(QIcon(iconPath(QStringLiteral("fit-window.png"))),
                        tr("Fit Window"), this);
    m_fitWindowAction = fit;
    fit->setProperty("qAnyToolbarId", QStringLiteral("fit_image"));
    toolbarActions.insert(QStringLiteral("fit_image"), fit);
    fit->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_F));
    registerShortcut(fit, QStringLiteral("fit_window"), tr("View"));
    m_canvas->addAction(fit);
    connect(fit, &QAction::triggered, m_canvas, &AnnotationCanvas::fitImage);
    QAction* zoomIn =
            new QAction(QIcon(iconPath(QStringLiteral("zoom-in.png"))),
                        tr("Zoom In"), this);
    m_zoomInAction = zoomIn;
    zoomIn->setProperty("qAnyToolbarId", QStringLiteral("zoom_in"));
    toolbarActions.insert(QStringLiteral("zoom_in"), zoomIn);
    zoomIn->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Plus));
    registerShortcut(zoomIn, QStringLiteral("zoom_in"), tr("View"));
    m_canvas->addAction(zoomIn);
    connect(zoomIn, &QAction::triggered, this,
            [this]() { m_canvas->zoomBy(1.2); });
    QAction* zoomOut =
            new QAction(QIcon(iconPath(QStringLiteral("zoom-out.png"))),
                        tr("Zoom Out"), this);
    m_zoomOutAction = zoomOut;
    zoomOut->setProperty("qAnyToolbarId", QStringLiteral("zoom_out"));
    toolbarActions.insert(QStringLiteral("zoom_out"), zoomOut);
    zoomOut->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Minus));
    registerShortcut(zoomOut, QStringLiteral("zoom_out"), tr("View"));
    m_canvas->addAction(zoomOut);
    connect(zoomOut, &QAction::triggered, this,
            [this]() { m_canvas->zoomBy(1.0 / 1.2); });

    // The source treats Fit Window and Fit Width as one mutually exclusive
    // zoom mode.  Keep the same state model even though the C++ canvas uses
    // explicit actions instead of a Python zoom widget.
    fit->setCheckable(true);
    fitWidth->setCheckable(true);
    auto* zoomModeGroup = new QActionGroup(this);
    zoomModeGroup->setExclusive(true);
    zoomModeGroup->addAction(fit);
    zoomModeGroup->addAction(fitWidth);
    fit->setChecked(true);
    connect(fit, &QAction::triggered, this, [fit, fitWidth]() {
        fit->setChecked(true);
        fitWidth->setChecked(false);
    });
    connect(fitWidth, &QAction::triggered, this, [fit, fitWidth]() {
        fit->setChecked(false);
        fitWidth->setChecked(true);
    });

    // Rebuild View in the exact order used by label_widget.py.  Actions that
    // are specific to ACloudViewer (AI, settings and LoMa) remain available
    // after the source section, while the source section itself is stable for
    // keyboard users and screenshots.
    const auto isSourceViewAction = [=](QAction* action) {
        return action == navigator || action == m_tagsAction ||
               action == fillDrawing || action == loopThroughLabels ||
               action == loopSelectLabels || action == zoomIn ||
               action == zoomOut || action == zoomOriginal ||
               action == keepPreviousScale ||
               action == m_keepBrightnessAction ||
               action == m_keepContrastAction || action == fit ||
               action == fitWidth || action == m_showMasksAction ||
               action == m_showTextsAction || action == m_showLabelsAction ||
               action == m_showScoresAction || action == m_showDegreesAction ||
               action == m_showAttributesAction ||
               action == m_showLinkingAction || action == m_showGroupsAction ||
               action == hideSelected || action == showHidden ||
               action == groupAction || action == ungroupAction;
    };
    QList<QAction*> viewExtensions;
    for (QAction* action : viewMenu->actions()) {
        if (!action->isSeparator() && !isSourceViewAction(action))
            viewExtensions.append(action);
    }
    for (QAction* action : viewMenu->actions()) viewMenu->removeAction(action);
    editMenu->removeAction(hideSelected);
    editMenu->removeAction(showHidden);
    editMenu->removeAction(groupAction);
    editMenu->removeAction(ungroupAction);
    viewMenu->addAction(navigator);
    viewMenu->addAction(m_tagsAction);
    viewMenu->addAction(fillDrawing);
    viewMenu->addAction(loopThroughLabels);
    viewMenu->addAction(loopSelectLabels);
    viewMenu->addSeparator();
    viewMenu->addAction(zoomIn);
    viewMenu->addAction(zoomOut);
    viewMenu->addAction(zoomOriginal);
    viewMenu->addSeparator();
    viewMenu->addAction(keepPreviousScale);
    viewMenu->addAction(m_keepBrightnessAction);
    viewMenu->addAction(m_keepContrastAction);
    viewMenu->addSeparator();
    viewMenu->addAction(fit);
    viewMenu->addAction(fitWidth);
    viewMenu->addSeparator();
    viewMenu->addAction(m_showMasksAction);
    viewMenu->addAction(m_showTextsAction);
    viewMenu->addAction(m_showLabelsAction);
    viewMenu->addAction(m_showScoresAction);
    viewMenu->addAction(m_showDegreesAction);
    viewMenu->addAction(m_showAttributesAction);
    viewMenu->addAction(m_showLinkingAction);
    viewMenu->addAction(m_showGroupsAction);
    viewMenu->addAction(hideSelected);
    viewMenu->addAction(showHidden);
    viewMenu->addAction(groupAction);
    viewMenu->addAction(ungroupAction);
    if (!viewExtensions.isEmpty()) viewMenu->addSeparator();
    for (QAction* action : viewExtensions) viewMenu->addAction(action);

    // Rebuild Edit in the source order: create modes, edit modes, object
    // commands, undo, coordinates/merge, then the persisted mode toggles.
    // Commands that the source keeps only in the Tool menu are removed here;
    // ACloudViewer-specific commands stay available after the source section.
    QHash<QString, QAction*> modeActions;
    for (QAction* mode : m_tools->actions())
        modeActions.insert(mode->data().toString(), mode);
    QAction* editMode = modeActions.value(QStringLiteral("edit"));
    QAction* polygonMode = modeActions.value(QStringLiteral("polygon"));
    QAction* magicWandMode = modeActions.value(QStringLiteral("magic_wand"));
    QAction* rectangleMode = modeActions.value(QStringLiteral("rectangle"));
    QAction* cuboidMode = modeActions.value(QStringLiteral("cuboid"));
    QAction* rotationMode = modeActions.value(QStringLiteral("rotation"));
    QAction* quadrilateralMode =
            modeActions.value(QStringLiteral("quadrilateral"));
    QAction* circleMode = modeActions.value(QStringLiteral("circle"));
    QAction* lineMode = modeActions.value(QStringLiteral("line"));
    QAction* pointMode = modeActions.value(QStringLiteral("point"));
    QAction* lineStripMode = modeActions.value(QStringLiteral("linestrip"));
    for (QAction* toolOnly :
         {labelManager, digitSettings, gidManager, shapeManager})
        editMenu->removeAction(toolOnly);
    const auto isSourceEditAction = [=](QAction* action) {
        return action == polygonMode || action == createBrush ||
               action == magicWandMode || action == rectangleMode ||
               action == cuboidMode || action == rotationMode ||
               action == quadrilateralMode || action == circleMode ||
               action == lineMode || action == pointMode ||
               action == lineStripMode || action == editMode ||
               action == brush || action == editLabel || action == duplicate ||
               action == deleteAction || action == copy || action == paste ||
               action == m_undoAction || action == m_undoLastPointAction ||
               action == m_copyCoordinatesAction ||
               action == removePointAction || action == unionAction ||
               action == toggleKeepPrevious ||
               action == m_autoUseLastLabelAction ||
               action == autoUseLastGidAction ||
               action == m_useSystemClipboardAction ||
               action == m_showShapesAction;
    };
    QList<QAction*> editExtensions;
    for (QAction* action : editMenu->actions())
        if (!action->isSeparator() && !isSourceEditAction(action))
            editExtensions.append(action);
    for (QAction* action : editMenu->actions()) editMenu->removeAction(action);
    editMenu->addAction(polygonMode);
    editMenu->addAction(createBrush);
    editMenu->addAction(magicWandMode);
    editMenu->addAction(rectangleMode);
    editMenu->addAction(cuboidMode);
    editMenu->addAction(rotationMode);
    editMenu->addAction(quadrilateralMode);
    editMenu->addAction(circleMode);
    editMenu->addAction(lineMode);
    editMenu->addAction(pointMode);
    editMenu->addAction(lineStripMode);
    editMenu->addSeparator();
    editMenu->addAction(editMode);
    editMenu->addAction(brush);
    editMenu->addAction(editLabel);
    editMenu->addAction(duplicate);
    editMenu->addAction(deleteAction);
    editMenu->addAction(copy);
    editMenu->addAction(paste);
    editMenu->addSeparator();
    editMenu->addAction(m_undoAction);
    editMenu->addAction(m_undoLastPointAction);
    editMenu->addSeparator();
    editMenu->addAction(m_copyCoordinatesAction);
    editMenu->addAction(removePointAction);
    editMenu->addAction(unionAction);
    editMenu->addSeparator();
    editMenu->addAction(toggleKeepPrevious);
    editMenu->addAction(m_autoUseLastLabelAction);
    editMenu->addAction(autoUseLastGidAction);
    editMenu->addAction(m_useSystemClipboardAction);
    editMenu->addAction(m_showShapesAction);
    if (!editExtensions.isEmpty()) editMenu->addSeparator();
    for (QAction* action : editExtensions) editMenu->addAction(action);

    // Selection-driven enable states follow the source
    // canvas_selection_changed slot.
    const auto updateSelectionStates = [this, copy, paste, editLabel, duplicate,
                                        unionAction, brush]() {
        const bool hasImage = !m_imagePath.isEmpty();
        const QSet<int> selected =
                hasImage ? m_canvas->selectedIndices() : QSet<int>();
        const QVector<AnnotationShape> shapes =
                hasImage ? m_canvas->shapes() : QVector<AnnotationShape>();
        QVector<const AnnotationShape*> picked;
        for (int row : selected)
            if (row >= 0 && row < shapes.size()) picked.append(&shapes[row]);
        const int n = picked.size();
        bool hasLocked = false;
        bool sameType = n > 0;
        bool mergeable = n > 1;
        for (const AnnotationShape* shape : picked) {
            hasLocked |= shape->locked;
            sameType = sameType && shape->type == picked.front()->type;
            mergeable = mergeable && !shape->locked &&
                        (shape->type == QStringLiteral("rectangle") ||
                         shape->type == QStringLiteral("polygon"));
        }
        if (m_deletePolygonAction)
            m_deletePolygonAction->setEnabled(hasImage && n > 0 && !hasLocked);
        if (m_copyCoordinatesAction)
            m_copyCoordinatesAction->setEnabled(hasImage && n == 1);
        if (m_pasteAction)
            m_pasteAction->setEnabled(
                    hasImage &&
                    (m_systemClipboard || !m_internalShapeClipboard.isEmpty()));
        copy->setEnabled(hasImage && n > 0);
        duplicate->setEnabled(hasImage && n > 0);
        editLabel->setEnabled(hasImage && n >= 1 && sameType);
        unionAction->setEnabled(hasImage && mergeable);
        // Brush editing stays available while a brush session is active even
        // though it clears the selection.
        brush->setEnabled(
                hasImage &&
                (m_canvas->brushEditing() ||
                 (n == 1 && picked.front()->type == QStringLiteral("polygon") &&
                  !picked.front()->locked)));
    };
    connect(m_canvas, &AnnotationCanvas::shapeSelected, this,
            [updateSelectionStates](int) { updateSelectionStates(); });
    connect(m_canvas, &AnnotationCanvas::shapesChanged, this,
            updateSelectionStates);
    updateSelectionStates();

    // Source canvas context menu: the two filter submenus and the checked
    // toggle lead the source actions.menu sequence. Right-click stays
    // reserved for SAM negative prompts while a SAM tool is active.
    m_canvas->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_canvas, &QWidget::customContextMenuRequested, this,
            [this, polygonMode, createBrush, magicWandMode, rectangleMode,
             cuboidMode, rotationMode, quadrilateralMode, circleMode, lineMode,
             pointMode, lineStripMode, editMode, brush, editLabel, duplicate,
             copy, paste, deleteAction, removePointAction,
             unionAction](const QPoint& position) {
                if (m_tools) {
                    if (QAction* checkedTool = m_tools->checkedAction()) {
                        const QString tool = checkedTool->data().toString();
                        if (tool == QStringLiteral("sam_point") ||
                            tool == QStringLiteral("sam_box"))
                            return;
                    }
                }
                QMenu menu(m_canvas);
                QMenu* labelMenu = menu.addMenu(tr("Filter by Label"));
                QMenu* gidMenu = menu.addMenu(tr("Filter by Group ID"));
                connect(labelMenu, &QMenu::aboutToShow, this,
                        [this, labelMenu]() {
                            populateLabelFilterMenu(labelMenu);
                        });
                connect(gidMenu, &QMenu::aboutToShow, this,
                        [this, gidMenu]() { populateGidFilterMenu(gidMenu); });
                menu.addSeparator();
                if (m_toggleCheckedAction)
                    menu.addAction(m_toggleCheckedAction);
                menu.addSeparator();
                for (QAction* mode :
                     {polygonMode, createBrush, magicWandMode, rectangleMode,
                      cuboidMode, rotationMode, quadrilateralMode, circleMode,
                      lineMode, pointMode, lineStripMode})
                    if (mode) menu.addAction(mode);
                menu.addSeparator();
                for (QAction* action : {editMode, brush, editLabel})
                    if (action) menu.addAction(action);
                const QSet<int> selectedShapes = m_canvas->selectedIndices();
                // The uploaded shape-attribute template edits the single
                // selected shape through a generated dialog.
                if (!m_attributes.isEmpty() && selectedShapes.size() == 1) {
                    const int only = *selectedShapes.cbegin();
                    if (only >= 0 && only < m_canvas->shapes().size() &&
                        m_attributes.contains(m_canvas->shapes()[only].label)) {
                        QAction* editAttributes =
                                menu.addAction(tr("Edit Attributes"));
                        connect(editAttributes, &QAction::triggered, this,
                                [this]() { editShapeAttributes(); });
                    }
                }
                bool anyUnlocked = false;
                for (int row : selectedShapes)
                    if (row >= 0 && row < m_canvas->shapes().size() &&
                        !m_canvas->shapes()[row].locked)
                        anyUnlocked = true;
                QAction* toggleLock =
                        menu.addAction(anyUnlocked ? tr("Lock") : tr("Unlock"));
                toggleLock->setEnabled(!selectedShapes.isEmpty());
                connect(toggleLock, &QAction::triggered, this, [this]() {
                    const QSet<int> selected = m_canvas->selectedIndices();
                    if (selected.isEmpty()) return;
                    QVector<AnnotationShape> shapes = m_canvas->shapes();
                    const bool allLocked = std::all_of(
                            selected.cbegin(), selected.cend(),
                            [&shapes](int row) {
                                return row >= 0 && row < shapes.size() &&
                                       shapes[row].locked;
                            });
                    for (int row : selected)
                        if (row >= 0 && row < shapes.size())
                            shapes[row].locked = !allLocked;
                    m_canvas->setShapes(shapes);
                    updateObjects();
                    setDirty(true);
                });
                menu.addSeparator();
                if (m_copyCoordinatesAction)
                    menu.addAction(m_copyCoordinatesAction);
                menu.addAction(unionAction);
                menu.addAction(duplicate);
                menu.addAction(copy);
                menu.addAction(paste);
                menu.addSeparator();
                menu.addAction(deleteAction);
                menu.addAction(m_undoAction);
                menu.addAction(m_undoLastPointAction);
                menu.addAction(removePointAction);
                menu.exec(m_canvas->mapToGlobal(position));
            });

    // The source registers Ctrl+Q as the quit shortcut; the annotation
    // workspace routes it through the confirm-on-close path.
    QAction* quitAction = new QAction(tr("Quit"), this);
    quitAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q));
    registerShortcut(quitAction, QStringLiteral("quit"), tr("File"));
    connect(quitAction, &QAction::triggered, this, [this]() { close(); });
    addAction(quitAction);

    // Match the upstream left toolbar sequence. These commands stay inside
    // the annotation workspace; the host plugin contributes one entry action.
    const auto appendToolbarAction = [fileBar,
                                      &toolbarActions](const QString& id) {
        if (QAction* action = toolbarActions.value(id)) {
            action->setProperty("qAnyToolbarId", id);
            fileBar->addAction(action);
        }
    };
    appendToolbarAction(QStringLiteral("open_dir"));
    appendToolbarAction(QStringLiteral("open_next"));
    appendToolbarAction(QStringLiteral("open_prev"));
    appendToolbarAction(QStringLiteral("save"));
    appendToolbarAction(QStringLiteral("delete_file"));
    fileBar->addSeparator();
    appendToolbarAction(QStringLiteral("tool_polygon"));
    appendToolbarAction(QStringLiteral("create_brush_polygon"));
    appendToolbarAction(QStringLiteral("tool_magic_wand"));
    appendToolbarAction(QStringLiteral("tool_rectangle"));
    appendToolbarAction(QStringLiteral("tool_cuboid"));
    appendToolbarAction(QStringLiteral("tool_rotation"));
    appendToolbarAction(QStringLiteral("tool_quadrilateral"));
    appendToolbarAction(QStringLiteral("tool_circle"));
    appendToolbarAction(QStringLiteral("tool_line"));
    appendToolbarAction(QStringLiteral("tool_point"));
    appendToolbarAction(QStringLiteral("tool_linestrip"));
    fileBar->addSeparator();
    appendToolbarAction(QStringLiteral("edit_brush_mode"));
    appendToolbarAction(QStringLiteral("tool_edit"));
    appendToolbarAction(QStringLiteral("delete_polygon"));
    appendToolbarAction(QStringLiteral("undo"));
    appendToolbarAction(QStringLiteral("loop_thru_labels"));
    appendToolbarAction(QStringLiteral("loop_select_labels"));
    appendToolbarAction(QStringLiteral("select_toggle_shapes"));
    appendToolbarAction(QStringLiteral("auto_run"));
    appendToolbarAction(QStringLiteral("auto_label"));
    fileBar->addSeparator();
    appendToolbarAction(QStringLiteral("open_chatbot"));
    appendToolbarAction(QStringLiteral("open_vqa"));
    appendToolbarAction(QStringLiteral("open_pointcloud"));
    appendToolbarAction(QStringLiteral("open_image_classifier"));
    appendToolbarAction(QStringLiteral("open_video_classifier"));
    appendToolbarAction(QStringLiteral("open_paddleocr"));
    fileBar->addSeparator();
    appendToolbarAction(QStringLiteral("fit_width"));
    // The source toolbar ends with the editable zoom percentage widget.
    fileBar->addAction(zoomWidgetAction);
    fileBar->addSeparator();
    // ACloudViewer additions follow the complete upstream toolbar sequence.
    appendToolbarAction(QStringLiteral("hide_selected_polygons"));
    appendToolbarAction(QStringLiteral("show_hidden_polygons"));
    fileBar->addWidget(brushSettingsButton);
    appendToolbarAction(QStringLiteral("open_video"));
    appendToolbarAction(QStringLiteral("open"));
    appendToolbarAction(QStringLiteral("open_prev_unchecked"));
    appendToolbarAction(QStringLiteral("open_next_unchecked"));
    appendToolbarAction(QStringLiteral("delete_image_file"));
    appendToolbarAction(QStringLiteral("clear_prompt"));
    appendToolbarAction(QStringLiteral("fit_image"));
    appendToolbarAction(QStringLiteral("zoom_in"));
    appendToolbarAction(QStringLiteral("zoom_out"));
    updateActionState();
    updateInstruction();
}

void AnnotationWindow::updateActionState() {
    // Bind image, file-list and model-panel lifecycle to the command state,
    // matching the source's on_load_active / zoom_actions / has_label_file
    // behavior: commands start disabled and become available once an image
    // (or the required context) exists.
    const bool imageLoaded = !m_imagePath.isEmpty();
    const bool multipleFiles = m_files && m_files->count() > 1;
    if (m_nextAction) m_nextAction->setEnabled(multipleFiles);
    if (m_previousAction) m_previousAction->setEnabled(multipleFiles);
    if (m_nextUncheckedAction) m_nextUncheckedAction->setEnabled(multipleFiles);
    if (m_previousUncheckedAction)
        m_previousUncheckedAction->setEnabled(multipleFiles);
    if (m_toggleCheckedAction) m_toggleCheckedAction->setEnabled(imageLoaded);
    if (m_saveAction)
        m_saveAction->setEnabled(imageLoaded && !m_sidecarBlocked);
    if (m_saveAsAction)
        m_saveAsAction->setEnabled(imageLoaded && !m_sidecarBlocked);
    if (m_deleteLabelFileAction) {
        bool sidecarExists = false;
        for (const QString& candidate : sidecarCandidates(m_imagePath))
            sidecarExists |= QFileInfo::exists(candidate);
        m_deleteLabelFileAction->setEnabled(imageLoaded && !m_sidecarBlocked &&
                                            sidecarExists);
    }
    if (m_autoRunAction)
        m_autoRunAction->setEnabled(m_modelPanel && m_modelPanel->isVisible());
    for (QAction* action :
         {m_zoomInAction, m_zoomOutAction, m_zoomOriginalAction,
          m_fitWindowAction, m_fitWidthAction}) {
        if (action) action->setEnabled(imageLoaded);
    }
    if (m_zoomWidget) m_zoomWidget->setEnabled(imageLoaded);
    const bool shapesPresent =
            imageLoaded && m_canvas && !m_canvas->shapes().isEmpty();
    if (m_loopThroughLabelsAction)
        m_loopThroughLabelsAction->setEnabled(imageLoaded);
    if (m_loopSelectLabelsAction)
        m_loopSelectLabelsAction->setEnabled(imageLoaded);
    if (m_selectToggleShapesAction)
        m_selectToggleShapesAction->setEnabled(shapesPresent);
}

void AnnotationWindow::populateModels() {
    for (int i = 0; i < aicore_model_count(AICORE_MODEL_FAMILY_DEPTH); ++i) {
        const aicore_model_entry* entry =
                aicore_model_at(AICORE_MODEL_FAMILY_DEPTH, i);
        if (!entry || !entry->filename || !entry->role ||
            (QString::fromLatin1(entry->role) != QStringLiteral("depth") &&
             !QString::fromUtf8(entry->filename)
                      .contains(QStringLiteral("nested-anyview"))))
            continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = model.filename.contains(QStringLiteral("nested-anyview"))
                             ? QStringLiteral("depth-nested")
                             : QStringLiteral("depth");
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (const aicore_model_family family :
         {AICORE_MODEL_FAMILY_DEEPLSD, AICORE_MODEL_FAMILY_ALIKED}) {
        for (int i = 0; i < aicore_model_count(family); ++i) {
            const aicore_model_entry* entry = aicore_model_at(family, i);
            if (!entry || !entry->filename) continue;
            Model model;
            model.filename = QString::fromUtf8(entry->filename);
            model.url = QString::fromUtf8(
                    entry->download_url ? entry->download_url : "");
            model.quant = quantization(model.filename);
            model.task = family == AICORE_MODEL_FAMILY_DEEPLSD
                                 ? QStringLiteral("deeplsd")
                                 : QStringLiteral("aliked");
            model.family = modelFamily(model.filename);
            m_models.append(model);
            if (m_model->findText(model.family) < 0)
                m_model->addItem(model.family);
        }
    }
    for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY); ++i) {
        const aicore_yolo_model_entry* entry =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        if (!entry || !entry->filename) continue;
        if (QString::fromUtf8(entry->filename)
                    .startsWith(QStringLiteral("reid-")))
            continue;
        const QString task =
                QString::fromLatin1(entry->task ? entry->task : "");
        if (task != QStringLiteral("detect") &&
            task != QStringLiteral("segment") &&
            task != QStringLiteral("semantic") &&
            task != QStringLiteral("pose") && task != QStringLiteral("obb") &&
            task != QStringLiteral("classify"))
            continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        const bool yoloe = model.filename.startsWith(QStringLiteral("yoloe-"));
        const bool world = model.filename.contains(QStringLiteral("-world"));
        model.task = yoloe   ? QStringLiteral("yoloe")
                     : world ? QStringLiteral("world")
                             : task;
        model.supportsText = entry->text_input &&
                             !model.filename.contains(QStringLiteral("-pf-"));
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
        const aicore_rfdetr_model_entry* entry = aicore_rfdetr_model_at(i);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = entry->segmentation_capable
                             ? QStringLiteral("rfdetr-segment")
                             : QStringLiteral("rfdetr-detect");
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_sam3_model_count(); ++i) {
        const aicore_sam3_model_entry* entry = aicore_sam3_model_at(i);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = QStringLiteral("sam3-pvs");
        model.supportsText = entry->visual_only == 0;
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_rmbg_model_count(); ++i) {
        const aicore_rmbg_model_entry* entry = aicore_rmbg_model_at(i);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = QStringLiteral("rmbg");
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_gkd_model_count(); ++i) {
        const aicore_gkd_model_entry* entry = aicore_gkd_model_at(i);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = QStringLiteral("gkd");
        model.family = modelFamily(model.filename);
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_facedetect_detector_model_count(); ++i) {
        const aicore_facedetect_model_entry* entry =
                aicore_facedetect_detector_model_at(i);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = QStringLiteral("default");
        model.task = QStringLiteral("facedetect");
        model.family = QFileInfo(model.filename).completeBaseName();
        model.confDefault = entry->conf_threshold;
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    m_faceLandmark->addItem(tr("5 landmarks"), QString());
    for (int i = 0; i < aicore_facedetect_landmark_model_count(); ++i) {
        const aicore_facedetect_model_entry* entry =
                aicore_facedetect_landmark_model_at(i);
        if (!entry || !entry->filename) continue;
        m_faceLandmark->addItem(
                QString::fromUtf8(entry->display_name ? entry->display_name
                                                      : entry->filename),
                QString::fromUtf8(entry->filename));
    }
    // PP-OCR text detection + recognition: the picker lists the rec GGUF
    // (family = script); the paired det GGUF is resolved per quantization
    // at run time through the dependency channel, exactly like the face
    // landmark model.
    for (int i = 0; i < aicore_ocr_model_count(); ++i) {
        const aicore_ocr_model_entry* entry = aicore_ocr_model_at(i);
        if (!entry || !entry->filename || !entry->role) continue;
        if (QString::fromLatin1(entry->role) != QStringLiteral("rec")) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.task = QStringLiteral("ocr");
        model.family =
                QStringLiteral("PP-OCRv4 (%1)")
                        .arg(QString::fromUtf8(entry->language ? entry->language
                                                               : "ch"));
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_TEXT); ++i) {
        const aicore_yolo_model_entry* entry =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_TEXT);
        if (!entry || !entry->filename) continue;
        Model model;
        model.filename = QString::fromUtf8(entry->filename);
        model.url = QString::fromUtf8(entry->download_url ? entry->download_url
                                                          : "");
        model.quant = quantization(model.filename);
        model.family = modelFamily(model.filename);
        m_textModels.append(model);
    }
    {
        // The remote inference service is a pseudo-entry: no local weights
        // exist, the configured server is the inference backend (see
        // RemoteServerClient).
        Model model;
        model.filename = QStringLiteral("remote-server");
        model.task = QStringLiteral("remote-server");
        model.family = QStringLiteral("Remote Server");
        m_models.append(model);
        if (m_model->findText(model.family) < 0) m_model->addItem(model.family);
    }
    refreshQuantizations();
    for (int i = 0; i < aicore_device_count(); ++i) {
        const aicore_device_info* info = aicore_device_at(i);
        if (info) {
            const QString id = QString::fromUtf8(info->id);
            m_backend->addItem(id.toUpper(), id);
            m_backend->setItemData(m_backend->count() - 1,
                                   QString::fromUtf8(info->label),
                                   Qt::ToolTipRole);
            if (info->is_default)
                m_backend->setCurrentIndex(m_backend->count() - 1);
        }
    }
}

void AnnotationWindow::refreshQuantizations() {
    // Apply the catalog interaction default for the selected model (the
    // upstream yaml conf per detector); the user can still override it via
    // the spinbox afterwards.
    const QString selectedFamily = m_model->currentText();
    for (const Model& model : m_models) {
        if (model.family == selectedFamily &&
            model.task == QStringLiteral("facedetect") &&
            model.confDefault > 0.0f) {
            m_confidence->setValue(model.confDefault);
            break;
        }
    }
    m_quant->clear();
    const QString family = m_model->currentText();
    const auto selected = std::find_if(
            m_models.cbegin(), m_models.cend(),
            [&family](const Model& model) { return model.family == family; });
    const bool gkd = selected != m_models.cend() &&
                     selected->task == QStringLiteral("gkd");
    const bool face = selected != m_models.cend() &&
                      selected->task == QStringLiteral("facedetect");
    const bool featureOnly = selected != m_models.cend() &&
                             selected->task == QStringLiteral("aliked");
    const bool openVocabulary = selected != m_models.cend() &&
                                (selected->task == QStringLiteral("world") ||
                                 selected->task == QStringLiteral("yoloe"));
    const bool sliced = selected != m_models.cend() &&
                        selected->task == QStringLiteral("detect");
    m_slicedDetection->setVisible(sliced);
    for (QWidget* widget : {static_cast<QWidget*>(m_sliceWidth),
                            static_cast<QWidget*>(m_sliceHeight),
                            static_cast<QWidget*>(m_sliceOverlap),
                            static_cast<QWidget*>(m_sliceMergeThreshold)})
        widget->setVisible(sliced);
    if (!sliced) m_slicedDetection->setChecked(false);
    m_samText->setVisible(selected != m_models.cend() &&
                          (selected->supportsText || gkd));
    m_gkdSupportButton->setVisible(gkd);
    m_faceLandmark->setVisible(face);
    m_confidence->setEnabled(!featureOnly);
    m_confidence->setToolTip(
            featureOnly ? tr("ALIKED returns its strongest keypoints")
                        : QString());
    m_samText->setPlaceholderText(gkd ? tr("Keypoints, comma separated")
                                  : openVocabulary
                                          ? tr("Classes, comma separated")
                                          : tr("SAM 3 text prompt"));
    m_textModel->clear();
    int preferredText = 0;
    if (openVocabulary && selected->supportsText) {
        const QString prefix = selected->task == QStringLiteral("world")
                                       ? QStringLiteral("clip-ViT-B-32-")
                                       : QStringLiteral("mobileclip2_b-");
        for (const Model& encoder : m_textModels) {
            if (!encoder.filename.startsWith(prefix)) continue;
            m_textModel->addItem(encoder.quant.toUpper(), encoder.filename);
            if (encoder.quant == QStringLiteral("f16"))
                preferredText = m_textModel->count() - 1;
        }
    }
    m_textModel->setCurrentIndex(preferredText);
    m_textModel->setVisible(m_textModel->count() > 0);
    m_confidence->setValue(selected != m_models.cend() &&
                                           selected->task ==
                                                   QStringLiteral("yoloe")
                                   ? 0.02
                                   : 0.25);
    int recommended = 0;
    for (const Model& model : m_models) {
        if (model.family != family) continue;
        m_quant->addItem(model.quant.toUpper(), model.filename);
        const QString preferred =
                model.task.startsWith(QStringLiteral("rfdetr-"))
                        ? QStringLiteral("f16")
                : model.task == QStringLiteral("rmbg") ? QStringLiteral("q8")
                                                       : QStringLiteral("q8_0");
        if (model.quant == preferred) recommended = m_quant->count() - 1;
    }
    m_quant->setCurrentIndex(recommended);
}

void AnnotationWindow::openDirectory() {
    const QString path = QFileDialog::getExistingDirectory(
            this, tr("Open folder"), m_directory);
    if (path.isEmpty()) return;
    openDirectoryPath(path);
}

void AnnotationWindow::openDirectoryPath(const QString& path) {
    if (path.isEmpty() || !QFileInfo(path).isDir() || !confirmSave()) return;
    m_directory = path;
    refreshFileList();
    if (m_files->count()) m_files->setCurrentRow(0);
    QSettings settings;
    settings.beginGroup(QStringLiteral("qAnyLabeling"));
    settings.setValue(QStringLiteral("lastDirectory"), m_directory);
    settings.endGroup();
    updateRecentMenu();
}

void AnnotationWindow::updateRecentMenu() {
    if (!m_recentMenu) return;
    m_recentMenu->clear();
    QStringList valid;
    for (const QString& file : m_recentFiles) {
        const QFileInfo info(file);
        const QString absolute = info.absoluteFilePath();
        if (!info.isFile() || valid.contains(absolute)) continue;
        valid.append(absolute);
    }
    m_recentFiles = valid;
    if (!m_directory.isEmpty() && QFileInfo(m_directory).isDir()) {
        QAction* folder = m_recentMenu->addAction(
                QIcon(iconPath(QStringLiteral("folder.svg"))),
                tr("Open last folder: %1")
                        .arg(QFileInfo(m_directory).fileName()));
        connect(folder, &QAction::triggered, this,
                [this]() { openDirectoryPath(m_directory); });
        if (!m_recentFiles.isEmpty()) m_recentMenu->addSeparator();
    }
    int number = 0;
    for (const QString& file : m_recentFiles) {
        if (file == m_imagePath) continue;
        QAction* action = m_recentMenu->addAction(
                QIcon(iconPath(QStringLiteral("image.svg"))),
                QStringLiteral("&%1 %2").arg(++number).arg(
                        QFileInfo(file).fileName()));
        action->setData(file);
        connect(action, &QAction::triggered, this,
                [this, file]() { openFile(file); });
    }
    m_recentMenu->setEnabled(!m_recentMenu->actions().isEmpty());
    QSettings settings;
    settings.beginGroup(QStringLiteral("qAnyLabeling"));
    settings.setValue(QStringLiteral("recentFiles"), m_recentFiles);
    settings.endGroup();
}

void AnnotationWindow::rememberRecentFile(const QString& file) {
    const QString absolute = QFileInfo(file).absoluteFilePath();
    m_recentFiles.removeAll(absolute);
    m_recentFiles.prepend(absolute);
    while (m_recentFiles.size() > 7) m_recentFiles.removeLast();
    QSettings settings;
    settings.beginGroup(QStringLiteral("qAnyLabeling"));
    settings.setValue(QStringLiteral("recentFiles"), m_recentFiles);
    settings.endGroup();
    updateRecentMenu();
}

void AnnotationWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (acceptsImageDrop(event->mimeData()))
        event->acceptProposedAction();
    else
        event->ignore();
}

void AnnotationWindow::dragMoveEvent(QDragMoveEvent* event) {
    if (acceptsImageDrop(event->mimeData()))
        event->acceptProposedAction();
    else
        event->ignore();
}

void AnnotationWindow::dropEvent(QDropEvent* event) {
    for (const QUrl& url : event->mimeData()->urls()) {
        const QString path = url.toLocalFile();
        if (QFileInfo(path).isDir()) {
            openDirectoryPath(path);
            event->acceptProposedAction();
            return;
        }
        if (!QFileInfo(path).isFile()) continue;
        const QString filter =
                QStringLiteral("*.%1").arg(QFileInfo(path).suffix().toLower());
        if (kImageFilters.contains(filter)) {
            openFile(path);
            event->acceptProposedAction();
            return;
        }
    }
    event->ignore();
}

bool AnnotationWindow::eventFilter(QObject* watched, QEvent* event) {
    if (event && (event->type() == QEvent::WindowActivate ||
                  event->type() == QEvent::WindowDeactivate))
        QTimer::singleShot(0, this, &AnnotationWindow::syncHostShortcuts);
    if (watched == m_tagInput && event) {
        if (event->type() == QEvent::KeyPress) {
            auto* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Escape) {
                cancelTagInput();
                return true;
            }
        } else if (event->type() == QEvent::FocusOut) {
            commitTagInput();
        }
    }
    if (watched == m_canvas && event) {
        if (event->type() == QEvent::Resize && m_adjustmentPanel) {
            m_adjustmentPanel->move(
                    10, std::max(0, m_canvas->height() -
                                            m_adjustmentPanel->height() - 10));
        }
        if (event->type() == QEvent::DragEnter) {
            dragEnterEvent(static_cast<QDragEnterEvent*>(event));
            return event->isAccepted();
        }
        if (event->type() == QEvent::DragMove) {
            dragMoveEvent(static_cast<QDragMoveEvent*>(event));
            return event->isAccepted();
        }
        if (event->type() == QEvent::Drop) {
            dropEvent(static_cast<QDropEvent*>(event));
            return event->isAccepted();
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void AnnotationWindow::syncHostShortcuts() {
    QWidget* active = QApplication::activeWindow();
    bool annotationActive = false;
    for (QWidget* widget = active; widget; widget = widget->parentWidget()) {
        if (widget == this) {
            annotationActive = true;
            break;
        }
    }
    if (!annotationActive) {
        for (auto it = m_suspendedHostActions.constBegin();
             it != m_suspendedHostActions.constEnd(); ++it)
            it.key()->setEnabled(it.value());
        m_suspendedHostActions.clear();
        return;
    }
    QSet<QKeySequence> pluginKeys;
    for (QAction* action : findChildren<QAction*>()) {
        if (action->isEnabled() && !action->shortcut().isEmpty() &&
            !action->property("qAnyShortcutId").toString().isEmpty())
            pluginKeys.insert(action->shortcut());
    }
    for (auto it = m_suspendedHostActions.begin();
         it != m_suspendedHostActions.end();) {
        if (pluginKeys.contains(it.key()->shortcut())) {
            ++it;
            continue;
        }
        it.key()->setEnabled(it.value());
        it = m_suspendedHostActions.erase(it);
    }
    for (QWidget* topLevel : QApplication::topLevelWidgets()) {
        if (topLevel == this || isAncestorOf(topLevel)) continue;
        for (QAction* action : topLevel->findChildren<QAction*>()) {
            if (!action ||
                action->shortcutContext() != Qt::ApplicationShortcut ||
                !pluginKeys.contains(action->shortcut()) ||
                m_suspendedHostActions.contains(action) || !action->isEnabled())
                continue;
            m_suspendedHostActions.insert(action, true);
            connect(action, &QObject::destroyed, this, [this, action]() {
                m_suspendedHostActions.remove(action);
            });
            action->setEnabled(false);
        }
    }
}

void AnnotationWindow::refreshFileList() {
    m_files->clear();
    if (m_directory.isEmpty()) return;
    const QFileInfoList entries =
            QDir(m_directory)
                    .entryInfoList(kImageFilters, QDir::Files, QDir::Name);
    for (const QFileInfo& entry : entries) {
        auto* item = new QListWidgetItem(entry.fileName(), m_files);
        item->setData(Qt::UserRole, entry.absoluteFilePath());
        item->setIcon(QIcon(iconPath(QStringLiteral("image.svg"))));
    }
    updateFileStatus();
    updateActionState();
    applyFileListCheckboxPolicy();
}

void AnnotationWindow::openFile(const QString& file) {
    if (!confirmSave()) {
        const auto found = m_files->findItems(QFileInfo(m_imagePath).fileName(),
                                              Qt::MatchExactly);
        if (!found.isEmpty()) m_files->setCurrentItem(found.first());
        return;
    }
    const auto previousAdjustment =
            qMakePair(m_brightnessSlider->value(), m_contrastSlider->value());
    const bool hadPreviousImage = !m_imagePath.isEmpty();
    const double previousZoom = m_canvas ? m_canvas->zoomFactor() : 1.0;
    if (hadPreviousImage) {
        m_displayAdjustments[m_imagePath] = qMakePair(
                m_brightnessSlider->value(), m_contrastSlider->value());
    }
    QImageReader reader(file);
    reader.setAutoTransform(m_exifScanEnabled);
    const QImage image = reader.read();
    if (image.isNull()) {
        QMessageBox::warning(this, tr("Open image"), reader.errorString());
        return;
    }
    const QString dir = QFileInfo(file).absolutePath();
    if (dir != m_directory) {
        m_directory = dir;
        refreshFileList();
        QSettings settings;
        settings.beginGroup(QStringLiteral("qAnyLabeling"));
        settings.setValue(QStringLiteral("lastDirectory"), m_directory);
        settings.endGroup();
    } else {
        const QString absoluteFile = QFileInfo(file).absoluteFilePath();
        bool listed = false;
        for (int i = 0; i < m_files->count(); ++i)
            listed |= m_files->item(i)->data(Qt::UserRole).toString() ==
                      absoluteFile;
        if (!listed) refreshFileList();
    }
    m_imagePath = QFileInfo(file).absoluteFilePath();
    ++m_generation;
    // A new image resets the source label-review loop state.
    m_labelLoopActive = false;
    m_labelLoopCount = -1;
    m_labelLoopTotal = 0;
    m_selectLoopCount = -1;
    m_image = image;
    m_canvas->setImage(image);
    if (m_zoomWidget) {
        m_zoomWidget->setEnabled(true);
        const QSignalBlocker blocker(m_zoomWidget);
        m_zoomWidget->setValue(qRound(m_canvas->zoomFactor() * 100.0));
    }
    if (hadPreviousImage && m_keepPreviousScale)
        m_canvas->setZoomFactor(previousZoom);
    m_canvas->setMaskOpacity(m_defaultMaskOpacity);
    m_adjustmentTimer->stop();
    auto adjustment =
            m_displayAdjustments.value(m_imagePath, qMakePair(50, 50));
    if (hadPreviousImage && m_keepPreviousBrightness)
        adjustment.first = previousAdjustment.first;
    if (hadPreviousImage && m_keepPreviousContrast)
        adjustment.second = previousAdjustment.second;
    m_displayAdjustments.insert(m_imagePath, adjustment);
    {
        const QSignalBlocker opacityBlock(m_opacitySlider);
        const QSignalBlocker brightnessBlock(m_brightnessSlider);
        const QSignalBlocker contrastBlock(m_contrastSlider);
        m_opacitySlider->setValue(100);
        m_brightnessSlider->setValue(adjustment.first);
        m_contrastSlider->setValue(adjustment.second);
    }
    m_opacityValue->setText(QStringLiteral("100%"));
    m_brightnessValue->setText(
            QString::number(adjustment.first / 50.0, 'f', 2));
    m_contrastValue->setText(QString::number(adjustment.second / 50.0, 'f', 2));
    m_canvas->setBrightnessContrast(adjustment.first, adjustment.second);
    m_adjustmentPanel->show();
    m_adjustmentPanel->adjustSize();
    m_adjustmentPanel->move(
            10,
            std::max(0, m_canvas->height() - m_adjustmentPanel->height() - 10));
    m_adjustmentPanel->raise();
    m_document = QJsonObject();
    m_sidecarBlocked = false;
    const QString sidecar = sidecarPathFor(file);
    QFile input(sidecar);
    QVector<AnnotationShape> shapes;
    if (input.open(QIODevice::ReadOnly)) {
        QJsonParseError error;
        const QJsonDocument parsed =
                QJsonDocument::fromJson(input.readAll(), &error);
        if (error.error == QJsonParseError::NoError && parsed.isObject() &&
            parsed.object().value(QStringLiteral("type")).toString() !=
                    QStringLiteral("video_classification")) {
            m_document = parsed.object();
            for (const QJsonValue& value :
                 m_document.value(QStringLiteral("shapes")).toArray())
                if (value.isObject())
                    shapes.append(AnnotationShape::fromJson(value.toObject()));
        } else {
            m_sidecarBlocked = true;
            QMessageBox::warning(this, tr("Annotation file"),
                                 tr("Cannot use %1 as an image annotation; "
                                    "saving is disabled until the conflicting "
                                    "JSON is moved or repaired.")
                                         .arg(sidecar));
        }
    } else if (input.exists()) {
        m_sidecarBlocked = true;
        QMessageBox::warning(this, tr("Annotation file"),
                             tr("Unable to read %1").arg(sidecar));
    }
    m_canvas->setShapes(shapes);
    updateNavigator();
    loadCompareImage();
    m_undo.clear();
    m_redo.clear();
    m_undoAction->setEnabled(false);
    m_redoAction->setEnabled(false);
    m_updating = true;
    m_checked->setChecked(m_document.value(QStringLiteral("checked")).toBool());
    m_updating = false;
    for (int i = 0; i < m_files->count(); ++i) {
        QListWidgetItem* item = m_files->item(i);
        if (item->data(Qt::UserRole).toString() == m_imagePath)
            item->setCheckState(m_checked->isChecked() ? Qt::Checked
                                                       : Qt::Unchecked);
    }
    updateFlags();
    updateTags();
    updateObjects();
    updateDescriptionEditor();
    setDirty(false);
    applyFileListCheckboxPolicy();
    for (int i = 0; i < m_files->count(); ++i)
        if (m_files->item(i)->data(Qt::UserRole).toString() == m_imagePath)
            m_files->setCurrentRow(i);
    m_status->setText(QFileInfo(file).fileName());
    rememberRecentFile(m_imagePath);
}

void AnnotationWindow::setDirty(bool dirty) {
    m_dirty = dirty;
    setWindowTitle(tr("Image Annotation - %1%2")
                           .arg(QFileInfo(m_imagePath).fileName(),
                                dirty ? QStringLiteral(" *") : QString()));
    updateActionState();
    if (dirty && m_autoSave && !m_autoSaveInProgress && !m_updating &&
        !m_imagePath.isEmpty()) {
        m_autoSaveInProgress = true;
        saveDocument();
        m_autoSaveInProgress = false;
    }
}

bool AnnotationWindow::confirmSave() {
    finishTagEditing();
    if (m_canvas->brushEditing()) m_canvas->setBrushEditing(false);
    if (!m_dirty) return true;
    const auto answer = QMessageBox::question(
            this, tr("Unsaved annotations"), tr("Save changes to this image?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);
    if (answer == QMessageBox::Cancel) return false;
    return answer == QMessageBox::Discard || saveDocument();
}

void AnnotationWindow::finishTagEditing() {
    for (QLineEdit* editor : m_tags->findChildren<QLineEdit*>()) {
        if (editor != m_tagInput && editor->isVisible()) {
            editor->clearFocus();
            break;
        }
    }
    if (m_tagInput && !m_tagInput->isHidden()) {
        if (!m_tagInput->text().trimmed().isEmpty()) commitTagInput();
        if (!m_tagInput->isHidden()) cancelTagInput();
    }
}

bool AnnotationWindow::saveDocument() {
    if (m_canvas->brushEditing()) m_canvas->setBrushEditing(false);
    if (m_imagePath.isEmpty()) return false;
    if (m_sidecarBlocked) {
        QMessageBox::warning(this, tr("Save annotations"),
                             tr("The image annotation JSON conflicts with "
                                "another or unreadable sidecar."));
        return false;
    }
    const QString sidecar = sidecarPathFor(m_imagePath);
    return saveDocumentTo(sidecar);
}

bool AnnotationWindow::saveDocumentTo(const QString& sidecar) {
    finishTagEditing();
    QJsonObject document = m_document;
    if (document.contains(QStringLiteral("tags"))) {
        QJsonArray tags;
        for (int i = 0; i < m_tags->count(); ++i)
            tags.append(m_tags->item(i)->text());
        document.insert(QStringLiteral("tags"), tags);
    }
    if (!document.contains(QStringLiteral("version")))
        document.insert(QStringLiteral("version"), QStringLiteral("4.0.6"));
    if (!document.contains(QStringLiteral("flags")))
        document.insert(QStringLiteral("flags"), QJsonObject());
    if (m_saveWithImageData && !m_image.isNull()) {
        QByteArray encoded;
        QBuffer buffer(&encoded);
        if (buffer.open(QIODevice::WriteOnly) && m_image.save(&buffer, "PNG"))
            document.insert(QStringLiteral("imageData"),
                            QString::fromLatin1(encoded.toBase64()));
        else
            document.insert(QStringLiteral("imageData"), QJsonValue::Null);
    } else {
        document.insert(QStringLiteral("imageData"), QJsonValue::Null);
    }
    document.insert(QStringLiteral("checked"), m_checked->isChecked());
    document.insert(QStringLiteral("imagePath"),
                    QFileInfo(m_imagePath).fileName());
    document.insert(QStringLiteral("imageHeight"), m_image.height());
    document.insert(QStringLiteral("imageWidth"), m_image.width());
    QJsonArray shapes;
    for (const AnnotationShape& shape : m_canvas->shapes())
        shapes.append(shape.toJson());
    document.insert(QStringLiteral("shapes"), shapes);
    QSaveFile output(sidecar);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(QJsonDocument(document).toJson(QJsonDocument::Indented)) <
                0 ||
        !output.commit()) {
        QMessageBox::warning(
                this, tr("Save annotations"),
                tr("Unable to save %1: %2").arg(sidecar, output.errorString()));
        return false;
    }
    m_document = document;
    setDirty(false);
    updateFileStatus();
    m_status->setText(tr("Saved %1").arg(QFileInfo(sidecar).fileName()));
    return true;
}

void AnnotationWindow::saveDocumentAs() {
    if (m_canvas->brushEditing()) m_canvas->setBrushEditing(false);
    if (m_imagePath.isEmpty() || m_sidecarBlocked) return;
    const QString suggested = sidecarPathFor(m_imagePath);
    const QString sidecar =
            QFileDialog::getSaveFileName(this, tr("Save annotations as"),
                                         suggested, tr("JSON files (*.json)"));
    if (!sidecar.isEmpty()) saveDocumentTo(sidecar);
}

void AnnotationWindow::closeImage() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    closeCompareView(false);
    ++m_generation;
    m_image = QImage();
    m_imagePath.clear();
    m_document = QJsonObject();
    m_sidecarBlocked = false;
    m_labelLoopActive = false;
    m_labelLoopCount = -1;
    m_labelLoopTotal = 0;
    m_selectLoopCount = -1;
    m_canvas->setImage(QImage());
    if (m_zoomWidget) m_zoomWidget->setEnabled(false);
    m_adjustmentTimer->stop();
    m_adjustmentPanel->hide();
    m_files->setCurrentRow(-1);
    m_objects->clear();
    m_labels->clear();
    updateFlags();
    updateDescriptionEditor();
    m_tagInput->clear();
    m_tagInput->hide();
    updateTags();
    m_checked->setChecked(false);
    m_undo.clear();
    m_redo.clear();
    m_undoAction->setEnabled(false);
    m_redoAction->setEnabled(false);
    setDirty(false);
    m_status->setText(tr("Open an image or folder"));
}

void AnnotationWindow::deleteLabelFile() {
    if (m_imagePath.isEmpty() || m_sidecarBlocked) return;
    QString sidecar;
    for (const QString& candidate : sidecarCandidates(m_imagePath)) {
        if (QFileInfo::exists(candidate)) {
            sidecar = candidate;
            break;
        }
    }
    if (sidecar.isEmpty()) {
        m_status->setText(tr("No label file for %1")
                                  .arg(QFileInfo(m_imagePath).fileName()));
        return;
    }
    if (QMessageBox::warning(this, tr("Delete label file"),
                             tr("Delete annotations for %1?")
                                     .arg(QFileInfo(m_imagePath).fileName()),
                             QMessageBox::Yes | QMessageBox::No,
                             QMessageBox::No) != QMessageBox::Yes) {
        return;
    }
    if (!QFile::remove(sidecar)) {
        QMessageBox::warning(this, tr("Delete label file"),
                             tr("Unable to delete %1").arg(sidecar));
        return;
    }
    m_document = QJsonObject();
    m_canvas->setShapes({});
    {
        const QSignalBlocker blocked(m_checked);
        m_checked->setChecked(false);
    }
    m_undo.clear();
    m_redo.clear();
    m_undoAction->setEnabled(false);
    m_redoAction->setEnabled(false);
    updateFlags();
    updateObjects();
    setDirty(false);
    updateFileStatus();
    m_status->setText(tr("Deleted %1").arg(QFileInfo(sidecar).fileName()));
}

void AnnotationWindow::deleteImageFile() {
    if (m_imagePath.isEmpty() || m_files->count() < 2 || !confirmSave()) return;
    const QFileInfo imageInfo(m_imagePath);
    if (QMessageBox::warning(
                this, tr("Delete image file"),
                tr("Move %1 and its label file to the _delete_ folder?")
                        .arg(imageInfo.fileName()),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes) {
        return;
    }

    const QString trashDirectory = imageInfo.absolutePath() +
                                   QDir::separator() +
                                   QStringLiteral("_delete_");
    if (!QDir().mkpath(trashDirectory)) {
        QMessageBox::warning(this, tr("Delete image file"),
                             tr("Unable to create %1").arg(trashDirectory));
        return;
    }
    QString target = QDir(trashDirectory).filePath(imageInfo.fileName());
    const QString stem = imageInfo.completeBaseName();
    const QString suffix = imageInfo.suffix();
    for (int serial = 1; QFile::exists(target); ++serial) {
        const QString name =
                suffix.isEmpty() ? QStringLiteral("%1_%2").arg(stem).arg(serial)
                                 : QStringLiteral("%1_%2.%3")
                                           .arg(stem)
                                           .arg(serial)
                                           .arg(suffix);
        target = QDir(trashDirectory).filePath(name);
    }
    if (!QFile::rename(m_imagePath, target)) {
        QMessageBox::warning(
                this, tr("Delete image file"),
                tr("Unable to move %1 to _delete_").arg(imageInfo.fileName()));
        return;
    }

    QString sidecar;
    for (const QString& candidate :
         sidecarCandidates(imageInfo.absoluteFilePath())) {
        if (QFile::exists(candidate)) {
            sidecar = candidate;
            break;
        }
    }
    const bool hadSidecar = !sidecar.isEmpty();
    const bool removedSidecar = !hadSidecar || QFile::remove(sidecar);
    const int oldRow = m_files->currentRow();
    closeImage();
    refreshFileList();
    if (m_files->count())
        m_files->setCurrentRow(std::min(oldRow, m_files->count() - 1));
    if (!removedSidecar) {
        QMessageBox::warning(this, tr("Delete image file"),
                             tr("The image was moved, but its label file could "
                                "not be removed."));
    }
    m_status->setText(tr("Moved %1 to _delete_").arg(imageInfo.fileName()));
}

void AnnotationWindow::runBatch() {
    if (m_batchRunning || m_inferenceRunning || m_downloader->isBusy() ||
        m_files->count() == 0 || m_image.isNull()) {
        return;
    }

    const QString filename = m_quant->currentData().toString();
    const auto found = std::find_if(m_models.cbegin(), m_models.cend(),
                                    [&filename](const Model& model) {
                                        return model.filename == filename;
                                    });
    if (found == m_models.cend()) return;

    const bool sam = found->task == QStringLiteral("sam3-pvs");
    const bool gkd = found->task == QStringLiteral("gkd");
    const bool openVocabulary = found->task == QStringLiteral("world") ||
                                found->task == QStringLiteral("yoloe");
    if (sam && !found->supportsText) {
        m_status->setText(tr("Auto Run requires a text-prompt SAM model"));
        return;
    }
    if ((sam || (gkd && m_gkdSupportPoints.isEmpty()) ||
         (openVocabulary && found->supportsText)) &&
        m_samText->text().trimmed().isEmpty()) {
        m_status->setText(gkd ? tr("Enter comma-separated keypoint names or "
                                   "select visual support points")
                              : tr("Enter comma-separated class names"));
        return;
    }

    QDialog range(this);
    range.setObjectName(QStringLiteral("AnnotationBatchRangeDialog"));
    range.setWindowTitle(tr("Auto Run"));
    range.setMinimumWidth(440);
    auto* layout = new QVBoxLayout(&range);
    layout->setContentsMargins(24, 24, 24, 24);
    layout->addWidget(new QLabel(tr("Image range"), &range));
    layout->addWidget(new QLabel(
            tr("Select the first and last images to process."), &range));
    auto* fields = new QGridLayout;
    auto* from = new QSpinBox(&range);
    auto* to = new QSpinBox(&range);
    from->setObjectName(QStringLiteral("AnnotationBatchFrom"));
    to->setObjectName(QStringLiteral("AnnotationBatchTo"));
    from->setRange(1, m_files->count());
    from->setValue(std::max(1, m_files->currentRow() + 1));
    to->setRange(from->value(), m_files->count());
    to->setValue(m_files->count());
    connect(from, QOverload<int>::of(&QSpinBox::valueChanged), to,
            &QSpinBox::setMinimum);
    fields->addWidget(new QLabel(tr("From"), &range), 0, 0);
    fields->addWidget(new QLabel(tr("To"), &range), 0, 1);
    fields->addWidget(from, 1, 0);
    fields->addWidget(to, 1, 1);
    auto* summary = new QLabel(&range);
    const auto updateSummary = [this, from, to, summary] {
        summary->setText(tr("%1 of %2 images selected")
                                 .arg(to->value() - from->value() + 1)
                                 .arg(m_files->count()));
    };
    connect(from, QOverload<int>::of(&QSpinBox::valueChanged), &range,
            updateSummary);
    connect(to, QOverload<int>::of(&QSpinBox::valueChanged), &range,
            updateSummary);
    updateSummary();
    fields->addWidget(summary, 2, 0);
    auto* buttons = new QDialogButtonBox(&range);
    buttons->addButton(tr("Cancel"), QDialogButtonBox::RejectRole);
    buttons->addButton(tr("Run"), QDialogButtonBox::AcceptRole);
    connect(buttons, &QDialogButtonBox::accepted, &range, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &range, &QDialog::reject);
    fields->addWidget(buttons, 2, 1);
    layout->addLayout(fields);
    if (range.exec() != QDialog::Accepted || !confirmSave()) return;

    m_batchFiles.clear();
    for (int row = from->value() - 1; row < to->value(); ++row) {
        const QString path = m_files->item(row)->data(Qt::UserRole).toString();
        if (!path.isEmpty()) m_batchFiles.append(path);
    }
    if (m_batchFiles.isEmpty()) return;

    m_batchIndex = 0;
    m_batchRunning = true;
    m_batchCancelRequested = false;
    m_batchProgress =
            new QProgressDialog(tr("Progress: 0/%1").arg(m_batchFiles.size()),
                                tr("Cancel"), 0, m_batchFiles.size(), this);
    m_batchProgress->setObjectName(QStringLiteral("AnnotationBatchProgress"));
    m_batchProgress->setWindowTitle(tr("Batch Processing"));
    m_batchProgress->setWindowModality(Qt::WindowModal);
    m_batchProgress->setMinimumWidth(400);
    m_batchProgress->setMinimumHeight(150);
    m_batchProgress->setMinimumDuration(0);
    m_batchProgress->setAutoClose(false);
    m_batchProgress->setAutoReset(false);
    connect(m_batchProgress, &QProgressDialog::canceled, this, [this] {
        if (!m_batchRunning) return;
        m_batchCancelRequested = true;
        m_runAfterDownload = false;
        if (m_downloader->isBusy()) {
            // Downloader cancellation has no finished signal to await.
            m_downloader->cancel();
            finishBatch(tr("Auto Run cancelled (%1 images completed)")
                                .arg(m_batchIndex));
            return;
        }
        if (!m_inferenceRunning)
            finishBatch(tr("Auto Run cancelled (%1 images completed)")
                                .arg(m_batchIndex));
    });
    m_batchProgress->show();
    m_status->setText(tr("Auto Run 0/%1").arg(m_batchFiles.size()));
    startBatchNext();
}

void AnnotationWindow::finishBatch(const QString& message) {
    m_batchRunning = false;
    m_batchCancelRequested = false;
    m_runAfterDownload = false;
    m_downloadDependencyModel = false;
    m_batchFiles.clear();
    m_batchIndex = -1;
    if (m_batchProgress) {
        m_batchProgress->hide();
        m_batchProgress->deleteLater();
        m_batchProgress = nullptr;
    }
    m_status->setText(message);
}

void AnnotationWindow::startBatchNext() {
    if (!m_batchRunning) return;
    if (m_batchIndex < 0 || m_batchIndex >= m_batchFiles.size()) {
        const int count = m_batchFiles.size();
        finishBatch(tr("Auto Run completed (%1 images)").arg(count));
        return;
    }

    const QString path = m_batchFiles.at(m_batchIndex);
    openFile(path);
    if (!m_batchRunning) return;
    if (m_imagePath != QFileInfo(path).absoluteFilePath() || m_sidecarBlocked) {
        finishBatch(tr("Auto Run stopped while opening %1")
                            .arg(QFileInfo(path).fileName()));
        return;
    }
    m_batchProgress->setLabelText(
            tr("Progress: %1/%2").arg(m_batchIndex).arg(m_batchFiles.size()));
    if (!m_batchRunning) return;
    m_status->setText(tr("Auto Run %1/%2: %3")
                              .arg(m_batchIndex + 1)
                              .arg(m_batchFiles.size())
                              .arg(QFileInfo(path).fileName()));
    runDetection();
    if (m_batchRunning && !m_inferenceRunning && !m_downloader->isBusy())
        finishBatch(m_status->text());
}

void AnnotationWindow::importYoloLabels(YoloFormat format) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const bool pose = format == YoloFormat::Pose;
    const QString classesPath = QFileDialog::getOpenFileName(
            this,
            pose ? tr("Select pose classes file") : tr("Select classes file"),
            m_directory,
            pose ? tr("YAML files (*.yaml *.yml)") : tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Import YOLO"),
                             classesFile.errorString());
        return;
    }
    QStringList classes;
    PoseConfig poseConfig;
    if (pose) {
        poseConfig = parsePoseConfig(classesFile.readAll());
        if (!poseConfig.error.isEmpty()) {
            QMessageBox::warning(this, tr("Import YOLO"), poseConfig.error);
            return;
        }
    } else {
        QSet<QString> uniqueClasses;
        for (const QByteArray& line : classesFile.readAll().split('\n')) {
            const QString name = QString::fromUtf8(line).trimmed();
            if (name.isEmpty()) continue;
            if (uniqueClasses.contains(name)) {
                QMessageBox::warning(this, tr("Import YOLO"),
                                     tr("Duplicate class: %1").arg(name));
                return;
            }
            classes.append(name);
            uniqueClasses.insert(name);
        }
    }
    if (!pose && classes.isEmpty()) {
        QMessageBox::warning(this, tr("Import YOLO"),
                             tr("The classes file is empty"));
        return;
    }
    importLabelFolder(
            tr("Import YOLO"), {QStringLiteral(".txt")},
            [classes, format, poseConfig](const QByteArray& labels,
                                          const QFileInfo& image) {
                QImageReader reader(image.absoluteFilePath());
                const QSize size = reader.size().isValid()
                                           ? reader.size()
                                           : reader.read().size();
                const YoloImport imported = importYolo(
                        labels, classes, format, image.fileName(), size,
                        format == YoloFormat::Pose ? &poseConfig : nullptr);
                return AnnotationImport{imported.document, imported.error,
                                        imported.count};
            });
}

void AnnotationWindow::importVocDota(bool voc, bool polygons) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    importLabelFolder(
            voc ? (polygons ? tr("Import VOC segmentation")
                            : tr("Import VOC detection"))
                : tr("Import DOTA"),
            {voc ? QStringLiteral(".xml") : QStringLiteral(".txt")},
            [voc, polygons](const QByteArray& labels, const QFileInfo& image) {
                if (voc) return importVoc(labels, image.fileName(), polygons);
                QImageReader reader(image.absoluteFilePath());
                const QSize size = reader.size().isValid()
                                           ? reader.size()
                                           : reader.read().size();
                return importDota(labels, image.fileName(), size);
            });
}

void AnnotationWindow::importLabelFolder(
        const QString& title,
        const QStringList& extensions,
        const std::function<AnnotationImport(const QByteArray&,
                                             const QFileInfo&)>& convert,
        const QString& protectedInput) {
    const QString labelsDir = QFileDialog::getExistingDirectory(
            this, tr("Select labels directory"), m_directory);
    if (labelsDir.isEmpty()) return;
    QHash<QString, QJsonObject> documents;
    QSet<QString> stems;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        const QString stem = image.completeBaseName();
        QString labelPath;
        for (const QString& extension : extensions) {
            const QString candidate =
                    QDir(labelsDir).filePath(stem + extension);
            if (QFileInfo::exists(candidate)) {
                labelPath = candidate;
                break;
            }
        }
        if (labelPath.isEmpty()) continue;
        QFile labels(labelPath);
        if (stems.contains(stem.toLower())) {
            QMessageBox::warning(
                    this, title,
                    tr("Multiple images share the label name %1").arg(stem));
            return;
        }
        stems.insert(stem.toLower());
        if (!labels.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, title, labels.errorString());
            return;
        }
        const AnnotationImport converted = convert(labels.readAll(), image);
        if (!converted.error.isEmpty()) {
            QMessageBox::warning(this, title,
                                 tr("%1: %2").arg(labelPath, converted.error));
            return;
        }
        documents.insert(image.absoluteFilePath(), converted.document);
    }
    if (documents.isEmpty()) {
        m_status->setText(tr("No matching label files found"));
        return;
    }
    writeImportedDocuments(title, documents, protectedInput);
}

void AnnotationWindow::importMasks() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Import mask images");
    const QString mappingPath = QFileDialog::getOpenFileName(
            this, tr("Select mask color map"), m_directory,
            tr("JSON files (*.json)"));
    if (mappingPath.isEmpty()) return;
    QFile mappingFile(mappingPath);
    if (!mappingFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, mappingFile.errorString());
        return;
    }
    QJsonParseError error;
    const QJsonDocument mapping =
            QJsonDocument::fromJson(mappingFile.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !mapping.isObject()) {
        QMessageBox::warning(this, title, tr("Invalid mask color map JSON"));
        return;
    }
    const QJsonObject colorMap = mapping.object();
    importLabelFolder(
            title, {QStringLiteral(".png"), QStringLiteral(".jpg")},
            [colorMap](const QByteArray& data, const QFileInfo& image) {
                QImageReader reader(image.absoluteFilePath());
                const QSize size = reader.size().isValid()
                                           ? reader.size()
                                           : reader.read().size();
                return importMask(QImage::fromData(data), colorMap,
                                  image.fileName(), size);
            },
            mappingPath);
}

void AnnotationWindow::importMotLabels() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString classesPath =
            QFileDialog::getOpenFileName(this, tr("Select MOT classes file"),
                                         m_directory, tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Import MOT"), classesFile.errorString());
        return;
    }
    QStringList classes;
    for (const QByteArray& line : classesFile.readAll().split('\n')) {
        const QString name = QString::fromUtf8(line).trimmed();
        if (!name.isEmpty()) classes.append(name);
    }
    if (classes.isEmpty()) {
        QMessageBox::warning(this, tr("Import MOT"),
                             tr("The classes file is empty"));
        return;
    }
    const QString gtPath =
            QFileDialog::getOpenFileName(this, tr("Select MOT gt.txt"),
                                         m_directory, tr("Text files (*.txt)"));
    if (gtPath.isEmpty()) return;
    QFile gtFile(gtPath);
    if (!gtFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Import MOT"), gtFile.errorString());
        return;
    }
    QVector<MotImage> images;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo info(m_files->item(i)->data(Qt::UserRole).toString());
        QImageReader reader(info.absoluteFilePath());
        const QSize size =
                reader.size().isValid() ? reader.size() : reader.read().size();
        images.append({info.fileName(), size, {}});
    }
    const MotImport converted = importMot(gtFile.readAll(), classes, images);
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, tr("Import MOT"), converted.error);
        return;
    }
    QHash<QString, QJsonObject> matched;
    for (auto it = converted.documents.cbegin();
         it != converted.documents.cend(); ++it) {
        for (const MotImage& image : images) {
            if (image.fileName == it.key()) {
                matched.insert(QDir(m_directory).filePath(it.key()),
                               it.value());
                break;
            }
        }
    }
    if (matched.isEmpty()) {
        m_status->setText(tr("No MOT frames match this folder"));
        return;
    }
    writeImportedDocuments(tr("Import MOT"), matched);
}

void AnnotationWindow::importSam3Archive() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Import SAM3 tracking archive");
    const QString manifestPath = QFileDialog::getOpenFileName(
            this, tr("Select SAM3 tracking manifest"), m_directory,
            tr("SAM3 tracking manifest (manifest.json)"));
    if (manifestPath.isEmpty()) return;
    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, manifestFile.errorString());
        return;
    }
    QVector<Sam3ArchiveImage> images;
    images.reserve(m_files->count());
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo info(m_files->item(i)->data(Qt::UserRole).toString());
        QImageReader reader(info.absoluteFilePath());
        const QSize size =
                reader.size().isValid() ? reader.size() : reader.read().size();
        images.append({info.absoluteFilePath(), size});
    }
    const Sam3ArchiveImport converted =
            ::importSam3Archive(manifestFile.readAll(),
                                QFileInfo(manifestPath).absolutePath(), images);
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, title, converted.error);
        return;
    }
    if (converted.documents.isEmpty()) {
        m_status->setText(tr("SAM3 archive has no instance masks to import"));
        return;
    }
    writeImportedDocuments(title, converted.documents, manifestPath);
}

void AnnotationWindow::openSam3ArchiveWorkspace() {
    if (!confirmSave()) return;
    const QString title = tr("Open SAM3 archive workspace");
    const QString manifestPath = QFileDialog::getOpenFileName(
            this, tr("Select SAM3 tracking manifest"), m_directory,
            tr("SAM3 tracking manifest (manifest.json)"));
    if (manifestPath.isEmpty()) return;
    QFile manifestFile(manifestPath);
    if (!manifestFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, manifestFile.errorString());
        return;
    }
    const QByteArray manifest = manifestFile.readAll();
    QJsonParseError parseError;
    const QJsonDocument parsed = QJsonDocument::fromJson(manifest, &parseError);
    if (parseError.error != QJsonParseError::NoError || !parsed.isObject()) {
        QMessageBox::warning(this, title,
                             tr("Invalid SAM3 tracking manifest JSON"));
        return;
    }
    QString videoPath =
            parsed.object().value(QStringLiteral("source")).toString();
    if (!QFileInfo(videoPath).isFile()) {
        videoPath = QFileDialog::getOpenFileName(
                this, tr("Select source video"), m_directory,
                tr("Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)"));
    }
    if (videoPath.isEmpty()) return;
    const QDir archive(QFileInfo(manifestPath).absolutePath());
    QString output = archive.filePath(QStringLiteral("frames"));
    for (int index = 2; QFileInfo(output).exists(); ++index)
        output = archive.filePath(QStringLiteral("frames_%1").arg(index));
    QProgressDialog progress(
            tr("Extracting video frames..."), tr("Cancel"), 0,
            parsed.object().value(QStringLiteral("total_frames")).toInt(0),
            this);
    progress.setWindowTitle(title);
    progress.setWindowModality(Qt::WindowModal);
    progress.setAutoClose(false);
    progress.show();
    QString error;
    const bool extracted = extractSam3FrameWorkspace(
            manifest, QFileInfo(manifestPath).absolutePath(), videoPath, output,
            [&progress](int current, int total) {
                progress.setMaximum(total);
                progress.setValue(current);
                QApplication::processEvents(QEventLoop::AllEvents, 5);
                return !progress.wasCanceled();
            },
            &error);
    progress.close();
    if (!extracted) {
        QMessageBox::warning(this, title, error);
        return;
    }
    openDirectoryPath(output);
    m_status->setText(tr("Opened SAM3 frame workspace: %1").arg(output));
}

void AnnotationWindow::importCocoLabels(CocoFormat format) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title =
            format == CocoFormat::Boxes      ? tr("Import COCO detection")
            : format == CocoFormat::Segments ? tr("Import COCO segmentation")
                                             : tr("Import COCO keypoints");
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select COCO annotations"), m_directory,
            tr("JSON files (*.json)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, file.errorString());
        return;
    }
    const CocoImport parsed = importCoco(file.readAll(), format);
    if (!parsed.error.isEmpty()) {
        QMessageBox::warning(this, title, parsed.error);
        return;
    }
    importNamedDocuments(title, parsed.documents, path);
}

void AnnotationWindow::importOdvgLabels() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Import ODVG detection");
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select ODVG annotations"), m_directory,
            tr("JSON lines (*.json *.jsonl)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, file.errorString());
        return;
    }
    const OdvgImport parsed = importOdvg(file.readAll());
    if (!parsed.error.isEmpty()) {
        QMessageBox::warning(this, title, parsed.error);
        return;
    }
    importNamedDocuments(title, parsed.documents, path);
}

void AnnotationWindow::importVlmLabels() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Import VLM-R1 OVD");
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select VLM-R1 annotations"), m_directory,
            tr("JSON lines (*.jsonl)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, file.errorString());
        return;
    }
    const VlmImport parsed = importVlm(file.readAll());
    if (!parsed.error.isEmpty()) {
        QMessageBox::warning(this, title, parsed.error);
        return;
    }
    importNamedDocuments(title, parsed.documents, path);
}

void AnnotationWindow::importPpocrLabels(PpocrMode mode) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = mode == PpocrMode::Rec ? tr("Import PPOCR Rec")
                                                 : tr("Import PPOCR KIE");
    const QString path = QFileDialog::getOpenFileName(
            this, title, m_directory, tr("PPOCR labels (*.txt *.json)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, file.errorString());
        return;
    }
    const PpocrImport parsed = importPpocr(file.readAll(), mode);
    if (!parsed.error.isEmpty()) {
        QMessageBox::warning(this, title, parsed.error);
        return;
    }
    importNamedDocuments(title, parsed.documents, path);
}

int AnnotationWindow::importNamedDocuments(
        const QString& title,
        const QHash<QString, QJsonObject>& namedDocuments,
        const QString& protectedInput) {
    QHash<QString, QString> available;
    for (int i = 0; i < m_files->count(); ++i) {
        const QString imagePath =
                m_files->item(i)->data(Qt::UserRole).toString();
        const QString key = QFileInfo(imagePath).fileName().toLower();
        if (available.contains(key)) {
            QMessageBox::warning(
                    this, title,
                    tr("Multiple images share the name %1").arg(key));
            return -1;
        }
        available.insert(key, imagePath);
    }
    QHash<QString, QJsonObject> documents;
    for (auto it = namedDocuments.cbegin(); it != namedDocuments.cend(); ++it) {
        const QString imagePath = available.value(it.key().toLower());
        if (imagePath.isEmpty()) continue;
        QImageReader reader(imagePath);
        const QSize actual =
                reader.size().isValid() ? reader.size() : reader.read().size();
        if (!actual.isValid() ||
            (it.value().contains(QStringLiteral("imageWidth")) &&
             actual.width() !=
                     it.value().value(QStringLiteral("imageWidth")).toInt()) ||
            (it.value().contains(QStringLiteral("imageHeight")) &&
             actual.height() !=
                     it.value().value(QStringLiteral("imageHeight")).toInt())) {
            QMessageBox::warning(
                    this, title,
                    tr("Annotation image size differs from %1").arg(imagePath));
            return -1;
        }
        QJsonObject document = it.value();
        document.insert(QStringLiteral("imageWidth"), actual.width());
        document.insert(QStringLiteral("imageHeight"), actual.height());
        documents.insert(imagePath, document);
    }
    if (documents.isEmpty()) {
        m_status->setText(tr("No annotation images match this folder"));
        return 0;
    }
    const int written =
            writeImportedDocuments(title, documents, protectedInput);
    if (written > 0 && namedDocuments.size() != written)
        m_status->setText(tr("Imported %1 images; %2 were not in this folder")
                                  .arg(written)
                                  .arg(namedDocuments.size() - written));
    return written;
}

int AnnotationWindow::writeImportedDocuments(
        const QString& title,
        const QHash<QString, QJsonObject>& documents,
        const QString& protectedInput) {
    struct Pending {
        QString path;
        QByteArray data;
    };
    QVector<Pending> pending;
    QSet<QString> sidecarNames;
    int existing = 0;
    for (auto it = documents.cbegin(); it != documents.cend(); ++it) {
        const QFileInfo image(it.key());
        const QString sidecar = QDir(image.absolutePath())
                                        .filePath(image.completeBaseName() +
                                                  QStringLiteral(".json"));
        const QString key = sidecar.toLower();
        if (sidecarNames.contains(key) ||
            (!protectedInput.isEmpty() &&
             QFileInfo(sidecar).absoluteFilePath() ==
                     QFileInfo(protectedInput).absoluteFilePath())) {
            QMessageBox::warning(
                    this, title,
                    tr("Conflicting annotation file: %1").arg(sidecar));
            return -1;
        }
        sidecarNames.insert(key);
        QFile old(sidecar);
        if (old.exists()) {
            if (!old.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, old.errorString());
                return -1;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(old.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(
                        this, title,
                        tr("Conflicting annotation file: %1").arg(sidecar));
                return -1;
            }
            ++existing;
        }
        pending.append({sidecar, QJsonDocument(it.value()).toJson()});
    }
    if (existing &&
        QMessageBox::question(
                this, title,
                tr("Replace %1 existing image annotation files?").arg(existing),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes)
        return 0;
    int written = 0;
    for (const Pending& item : pending) {
        QSaveFile output(item.path);
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(item.data) != item.data.size() || !output.commit()) {
            QMessageBox::warning(
                    this, title,
                    tr("Imported %1 files, then failed to write %2: %3")
                            .arg(written)
                            .arg(item.path, output.errorString()));
            return -1;
        }
        ++written;
    }
    openFile(m_imagePath);
    updateFileStatus();
    m_status->setText(tr("Imported %1 annotations").arg(written));
    return written;
}

void AnnotationWindow::exportYolo(YoloFormat format) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const bool pose = format == YoloFormat::Pose;
    const QString classesPath = QFileDialog::getOpenFileName(
            this,
            pose ? tr("Select pose classes file") : tr("Select classes file"),
            m_directory,
            pose ? tr("YAML files (*.yaml *.yml)") : tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Export YOLO"),
                             classesFile.errorString());
        return;
    }
    QStringList classes;
    PoseConfig poseConfig;
    if (pose) {
        poseConfig = parsePoseConfig(classesFile.readAll());
        if (!poseConfig.error.isEmpty()) {
            QMessageBox::warning(this, tr("Export YOLO"), poseConfig.error);
            return;
        }
        for (const PoseClass& item : poseConfig.classes)
            classes.append(item.name);
    } else {
        QSet<QString> uniqueClasses;
        for (const QByteArray& line : classesFile.readAll().split('\n')) {
            const QString name = QString::fromUtf8(line).trimmed();
            if (name.isEmpty()) continue;
            if (uniqueClasses.contains(name)) {
                QMessageBox::warning(this, tr("Export YOLO"),
                                     tr("Duplicate class: %1").arg(name));
                return;
            }
            classes.append(name);
            uniqueClasses.insert(name);
        }
    }
    if (classes.isEmpty()) {
        QMessageBox::warning(this, tr("Export YOLO"),
                             tr("The classes file is empty"));
        return;
    }
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory).filePath(QStringLiteral("../labels"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, tr("Export YOLO"),
                tr("Choose a directory outside the source image directory"));
        return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("YOLO export options"));
    QVBoxLayout options(&dialog);
    QCheckBox includeImages(tr("Save with images"), &dialog);
    QCheckBox skipEmpty(tr("Skip empty labels"), &dialog);
    QCheckBox keepObb(tr("Keep out-of-bounds oriented boxes"), &dialog);
    keepObb.setChecked(true);
    options.addWidget(&includeImages);
    options.addWidget(&skipEmpty);
    if (format == YoloFormat::OrientedBoxes) options.addWidget(&keepObb);
    QDialogButtonBox buttons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                             &dialog);
    options.addWidget(&buttons);
    connect(&buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(&buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;

    QStringList images;
    for (int i = 0; i < m_files->count(); ++i)
        images.append(m_files->item(i)->data(Qt::UserRole).toString());
    if (images.isEmpty()) images.append(m_imagePath);
    QSet<QString> labelNames;
    for (const QString& imagePath : images) {
        const QString stem = QFileInfo(imagePath).completeBaseName().toLower();
        if (labelNames.contains(stem)) {
            QMessageBox::warning(
                    this, tr("Export YOLO"),
                    tr("Multiple images share the label name %1.txt")
                            .arg(stem));
            return;
        }
        labelNames.insert(stem);
        const QString labelPath = QDir(outputDir).filePath(
                QFileInfo(imagePath).completeBaseName() +
                QStringLiteral(".txt"));
        if (QFileInfo(labelPath).absoluteFilePath() ==
            QFileInfo(classesPath).absoluteFilePath()) {
            QMessageBox::warning(
                    this, tr("Export YOLO"),
                    tr("The export would overwrite the classes file"));
            return;
        }
    }
    int exported = 0;
    QProgressDialog progress(tr("Exporting annotations"), tr("Cancel"), 0,
                             images.size(), this);
    progress.setWindowModality(Qt::WindowModal);
    for (int index = 0; index < images.size(); ++index) {
        progress.setValue(index);
        QCoreApplication::processEvents();
        if (progress.wasCanceled()) {
            m_status->setText(
                    tr("Export canceled after %1 images").arg(exported));
            return;
        }
        const QString& imagePath = images[index];
        const QFileInfo imageInfo(imagePath);
        const QString sidecar = QDir(imageInfo.absolutePath())
                                        .filePath(imageInfo.completeBaseName() +
                                                  QStringLiteral(".json"));
        QJsonObject document;
        QFile input(sidecar);
        if (input.exists()) {
            QJsonParseError error;
            if (!input.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, tr("Export YOLO"),
                                     input.errorString());
                return;
            }
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(input.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(
                        this, tr("Export YOLO"),
                        tr("Invalid annotation file: %1").arg(sidecar));
                return;
            }
            document = parsed.object();
        }
        if (document.value(QStringLiteral("imageWidth")).toInt() <= 0 ||
            document.value(QStringLiteral("imageHeight")).toInt() <= 0) {
            QImageReader reader(imagePath);
            const QSize size = reader.size().isValid() ? reader.size()
                                                       : reader.read().size();
            document.insert(QStringLiteral("imageWidth"), size.width());
            document.insert(QStringLiteral("imageHeight"), size.height());
        }
        const YoloConversion converted =
                convertYolo(document, classes, format, keepObb.isChecked(),
                            pose ? &poseConfig : nullptr);
        if (!converted.error.isEmpty()) {
            QMessageBox::warning(this, tr("Export YOLO"),
                                 tr("%1: %2").arg(imagePath, converted.error));
            return;
        }
        if (skipEmpty.isChecked() && !converted.count) continue;
        QSaveFile label(QDir(outputDir).filePath(imageInfo.completeBaseName() +
                                                 QStringLiteral(".txt")));
        if (!label.open(QIODevice::WriteOnly) ||
            label.write(converted.text) != converted.text.size() ||
            !label.commit()) {
            QMessageBox::warning(this, tr("Export YOLO"), label.errorString());
            return;
        }
        if (includeImages.isChecked()) {
            QFile sourceImage(imagePath);
            QSaveFile targetImage(
                    QDir(outputDir).filePath(imageInfo.fileName()));
            if (!sourceImage.open(QIODevice::ReadOnly) ||
                !targetImage.open(QIODevice::WriteOnly)) {
                QMessageBox::warning(this, tr("Export YOLO"),
                                     tr("Unable to copy %1").arg(imagePath));
                return;
            }
            while (!sourceImage.atEnd()) {
                const QByteArray chunk = sourceImage.read(1024 * 1024);
                if (chunk.isEmpty() ||
                    targetImage.write(chunk) != chunk.size()) {
                    QMessageBox::warning(
                            this, tr("Export YOLO"),
                            tr("Unable to copy %1").arg(imagePath));
                    return;
                }
            }
            if (!targetImage.commit()) {
                QMessageBox::warning(this, tr("Export YOLO"),
                                     targetImage.errorString());
                return;
            }
        }
        ++exported;
    }
    progress.setValue(images.size());
    m_status->setText(
            tr("Exported %1 YOLO labels to %2").arg(exported).arg(outputDir));
}

void AnnotationWindow::exportVocDota(bool voc, bool polygons) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory).filePath(QStringLiteral("../labels"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, tr("Export annotations"),
                tr("Choose a directory outside the source image directory"));
        return;
    }
    QStringList images;
    for (int i = 0; i < m_files->count(); ++i)
        images.append(m_files->item(i)->data(Qt::UserRole).toString());
    if (images.isEmpty()) images.append(m_imagePath);
    QSet<QString> stems;
    for (const QString& image : images) {
        const QString stem = QFileInfo(image).completeBaseName().toLower();
        if (stems.contains(stem)) {
            QMessageBox::warning(
                    this, tr("Export annotations"),
                    tr("Multiple images share the output name %1").arg(stem));
            return;
        }
        stems.insert(stem);
    }
    int exported = 0;
    QProgressDialog progress(tr("Exporting annotations"), tr("Cancel"), 0,
                             images.size(), this);
    progress.setWindowModality(Qt::WindowModal);
    for (int index = 0; index < images.size(); ++index) {
        progress.setValue(index);
        QCoreApplication::processEvents();
        if (progress.wasCanceled()) break;
        const QFileInfo info(images[index]);
        const QString sidecar = QDir(info.absolutePath())
                                        .filePath(info.completeBaseName() +
                                                  QStringLiteral(".json"));
        QJsonObject document;
        QFile input(sidecar);
        if (input.exists()) {
            QJsonParseError error;
            if (!input.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, tr("Export annotations"),
                                     input.errorString());
                return;
            }
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(input.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(
                        this, tr("Export annotations"),
                        tr("Invalid image annotation: %1").arg(sidecar));
                return;
            }
            document = parsed.object();
        }
        if (document.value(QStringLiteral("imageWidth")).toInt() <= 0 ||
            document.value(QStringLiteral("imageHeight")).toInt() <= 0) {
            QImageReader reader(images[index]);
            const QSize size = reader.size().isValid() ? reader.size()
                                                       : reader.read().size();
            document.insert(QStringLiteral("imageWidth"), size.width());
            document.insert(QStringLiteral("imageHeight"), size.height());
        }
        const YoloConversion converted =
                voc ? convertVoc(document, info.fileName(), outputDir, polygons)
                    : convertDota(document);
        if (!converted.error.isEmpty()) {
            QMessageBox::warning(
                    this, tr("Export annotations"),
                    tr("%1: %2").arg(images[index], converted.error));
            return;
        }
        const QString output = QDir(outputDir).filePath(
                info.completeBaseName() +
                (voc ? QStringLiteral(".xml") : QStringLiteral(".txt")));
        QSaveFile file(output);
        if (!file.open(QIODevice::WriteOnly) ||
            file.write(converted.text) != converted.text.size() ||
            !file.commit()) {
            QMessageBox::warning(this, tr("Export annotations"),
                                 file.errorString());
            return;
        }
        ++exported;
    }
    progress.setValue(images.size());
    m_status->setText(
            tr("Exported %1 annotations to %2").arg(exported).arg(outputDir));
}

void AnnotationWindow::exportCoco(bool polygons, bool pose) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString classesPath = QFileDialog::getOpenFileName(
            this,
            pose ? tr("Select pose classes file") : tr("Select classes file"),
            m_directory,
            pose ? tr("YAML files (*.yaml *.yml)") : tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Export COCO"),
                             classesFile.errorString());
        return;
    }
    QStringList classes;
    PoseConfig poseConfig;
    if (pose) {
        poseConfig = parsePoseConfig(classesFile.readAll());
        if (!poseConfig.error.isEmpty()) {
            QMessageBox::warning(this, tr("Export COCO"), poseConfig.error);
            return;
        }
        for (const PoseClass& item : poseConfig.classes)
            classes.append(item.name);
    } else {
        QSet<QString> unique;
        for (const QByteArray& line : classesFile.readAll().split('\n')) {
            const QString name = QString::fromUtf8(line).trimmed();
            if (name.isEmpty()) continue;
            if (unique.contains(name)) {
                QMessageBox::warning(this, tr("Export COCO"),
                                     tr("Duplicate class: %1").arg(name));
                return;
            }
            unique.insert(name);
            classes.append(name);
        }
    }
    if (classes.isEmpty()) {
        QMessageBox::warning(this, tr("Export COCO"),
                             tr("The classes file is empty"));
        return;
    }
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory).filePath(QStringLiteral("../labels"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, tr("Export COCO"),
                tr("Choose a directory outside the source image directory"));
        return;
    }
    const QString output = QDir(outputDir).filePath(
            pose       ? QStringLiteral("coco_keypoints.json")
            : polygons ? QStringLiteral("coco_instance_segmentation.json")
                       : QStringLiteral("coco_detection.json"));
    if (QFileInfo(output).absoluteFilePath() ==
        QFileInfo(classesPath).absoluteFilePath()) {
        QMessageBox::warning(this, tr("Export COCO"),
                             tr("The export would overwrite the classes file"));
        return;
    }
    QVector<CocoImage> images;
    QProgressDialog progress(tr("Exporting annotations"), tr("Cancel"), 0,
                             m_files->count(), this);
    progress.setWindowModality(Qt::WindowModal);
    for (int i = 0; i < m_files->count(); ++i) {
        progress.setValue(i);
        QCoreApplication::processEvents();
        if (progress.wasCanceled()) return;
        const QFileInfo info(m_files->item(i)->data(Qt::UserRole).toString());
        const QString sidecar = QDir(info.absolutePath())
                                        .filePath(info.completeBaseName() +
                                                  QStringLiteral(".json"));
        QFile input(sidecar);
        if (!input.exists()) continue;
        if (!input.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, tr("Export COCO"), input.errorString());
            return;
        }
        QJsonParseError error;
        const QJsonDocument parsed =
                QJsonDocument::fromJson(input.readAll(), &error);
        if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
            parsed.object().value(QStringLiteral("type")).toString() ==
                    QStringLiteral("video_classification")) {
            QMessageBox::warning(
                    this, tr("Export COCO"),
                    tr("Invalid image annotation: %1").arg(sidecar));
            return;
        }
        images.append({info.fileName(), parsed.object()});
    }
    progress.setValue(m_files->count());
    const CocoConversion converted =
            pose ? convertCocoPose(images, poseConfig)
                 : convertCoco(images, classes, polygons);
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, tr("Export COCO"), converted.error);
        return;
    }
    const QByteArray data =
            QJsonDocument(converted.document).toJson(QJsonDocument::Indented);
    QSaveFile file(output);
    if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size() ||
        !file.commit()) {
        QMessageBox::warning(this, tr("Export COCO"), file.errorString());
        return;
    }
    m_status->setText(tr("Exported %1 COCO annotations to %2")
                              .arg(converted.count)
                              .arg(output));
}

void AnnotationWindow::exportMasks() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString mapPath = QFileDialog::getOpenFileName(
            this, tr("Select mask color map"), m_directory,
            tr("JSON files (*.json)"));
    if (mapPath.isEmpty()) return;
    QFile mapFile(mapPath);
    if (!mapFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Export masks"), mapFile.errorString());
        return;
    }
    QJsonParseError error;
    const QJsonDocument parsedMap =
            QJsonDocument::fromJson(mapFile.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !parsedMap.isObject()) {
        QMessageBox::warning(this, tr("Export masks"),
                             tr("Invalid mask color map"));
        return;
    }
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory).filePath(QStringLiteral("../masks"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, tr("Export masks"),
                tr("Choose a directory outside the source image directory"));
        return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Mask export options"));
    QVBoxLayout options(&dialog);
    QCheckBox includeEmpty(tr("Include images without labels"), &dialog);
    QCheckBox checkedOnly(tr("Only checked images"), &dialog);
    options.addWidget(&includeEmpty);
    options.addWidget(&checkedOnly);
    QDialogButtonBox buttons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                             &dialog);
    options.addWidget(&buttons);
    connect(&buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(&buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;

    QSet<QString> stems;
    for (int i = 0; i < m_files->count(); ++i) {
        const QString stem =
                QFileInfo(m_files->item(i)->data(Qt::UserRole).toString())
                        .completeBaseName()
                        .toLower();
        if (stems.contains(stem)) {
            QMessageBox::warning(
                    this, tr("Export masks"),
                    tr("Multiple images share the mask name %1.png").arg(stem));
            return;
        }
        stems.insert(stem);
    }
    int exported = 0;
    QProgressDialog progress(tr("Exporting masks"), tr("Cancel"), 0,
                             m_files->count(), this);
    progress.setWindowModality(Qt::WindowModal);
    for (int i = 0; i < m_files->count(); ++i) {
        progress.setValue(i);
        QCoreApplication::processEvents();
        if (progress.wasCanceled()) break;
        QListWidgetItem* item = m_files->item(i);
        if (checkedOnly.isChecked() && item->checkState() != Qt::Checked)
            continue;
        const QFileInfo image(item->data(Qt::UserRole).toString());
        const QString sidecar = QDir(image.absolutePath())
                                        .filePath(image.completeBaseName() +
                                                  QStringLiteral(".json"));
        QFile input(sidecar);
        QJsonObject document;
        if (input.exists()) {
            if (!input.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, tr("Export masks"),
                                     input.errorString());
                return;
            }
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(input.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(
                        this, tr("Export masks"),
                        tr("Invalid image annotation: %1").arg(sidecar));
                return;
            }
            document = parsed.object();
        } else if (!includeEmpty.isChecked() || checkedOnly.isChecked())
            continue;
        if (document.value(QStringLiteral("imageWidth")).toInt() <= 0 ||
            document.value(QStringLiteral("imageHeight")).toInt() <= 0) {
            QImageReader reader(image.absoluteFilePath());
            const QSize size = reader.size().isValid() ? reader.size()
                                                       : reader.read().size();
            document.insert(QStringLiteral("imageWidth"), size.width());
            document.insert(QStringLiteral("imageHeight"), size.height());
        }
        const MaskConversion converted =
                convertMask(document, parsedMap.object());
        if (!converted.error.isEmpty()) {
            QMessageBox::warning(
                    this, tr("Export masks"),
                    tr("%1: %2").arg(image.fileName(), converted.error));
            return;
        }
        QSaveFile output(QDir(outputDir).filePath(image.completeBaseName() +
                                                  QStringLiteral(".png")));
        if (!output.open(QIODevice::WriteOnly) ||
            !converted.image.save(&output, "PNG") || !output.commit()) {
            QMessageBox::warning(this, tr("Export masks"),
                                 output.errorString());
            return;
        }
        ++exported;
    }
    progress.setValue(m_files->count());
    m_status->setText(
            tr("Exported %1 masks to %2").arg(exported).arg(outputDir));
}

void AnnotationWindow::exportMot(bool segmentation) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = segmentation ? tr("Export MOTS") : tr("Export MOT");
    const QString classesPath =
            QFileDialog::getOpenFileName(this, tr("Select MOT classes file"),
                                         m_directory, tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, classesFile.errorString());
        return;
    }
    QStringList classes;
    for (const QByteArray& line : classesFile.readAll().split('\n')) {
        const QString name = QString::fromUtf8(line).trimmed();
        if (!name.isEmpty()) classes.append(name);
    }
    if (classes.isEmpty() ||
        QSet<QString>(classes.cbegin(), classes.cend()).size() !=
                classes.size()) {
        QMessageBox::warning(this, title,
                             tr("MOT classes must be nonempty and unique"));
        return;
    }
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory)
                              .filePath(segmentation ? QStringLiteral("../mots")
                                                     : QStringLiteral("../mot"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, title,
                tr("Choose a directory outside the source image directory"));
        return;
    }
    QVector<MotImage> images;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        QImageReader reader(image.absoluteFilePath());
        const QSize size =
                reader.size().isValid() ? reader.size() : reader.read().size();
        QJsonObject document;
        QFile sidecar(QDir(image.absolutePath())
                              .filePath(image.completeBaseName() +
                                        QStringLiteral(".json")));
        if (sidecar.exists()) {
            if (!sidecar.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, sidecar.errorString());
                return;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(sidecar.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(this, title,
                                     tr("Invalid image annotation: %1")
                                             .arg(sidecar.fileName()));
                return;
            }
            document = parsed.object();
        }
        if (!segmentation || document.contains(QStringLiteral("shapes")))
            images.append({image.fileName(), size, document});
    }
    const MotConversion converted =
            segmentation
                    ? convertMots(images, classes, QDir(outputDir).dirName())
                    : convertMot(images, classes, QDir(outputDir).dirName());
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, title, converted.error);
        return;
    }
    QVector<QPair<QString, QByteArray>> files{
            {QStringLiteral("seqinfo.ini"), converted.sequence}};
    if (segmentation)
        files.append({QStringLiteral("custom_gt.txt"), converted.groundTruth});
    else {
        files.append({QStringLiteral("det.txt"), converted.detections});
        files.append({QStringLiteral("gt.txt"), converted.groundTruth});
    }
    int existing = 0;
    for (const auto& file : files)
        if (QFileInfo::exists(QDir(outputDir).filePath(file.first))) ++existing;
    if (existing &&
        QMessageBox::question(
                this, title,
                tr("Replace %1 existing output files?").arg(existing),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes)
        return;
    for (const auto& file : files) {
        QSaveFile output(QDir(outputDir).filePath(file.first));
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(file.second) != file.second.size() ||
            !output.commit()) {
            QMessageBox::warning(
                    this, title,
                    tr("Unable to write %1: %2")
                            .arg(output.fileName(), output.errorString()));
            return;
        }
    }
    m_status->setText(tr("Exported %1 %2 tracks to %3")
                              .arg(converted.count)
                              .arg(segmentation ? QStringLiteral("MOTS")
                                                : QStringLiteral("MOT"))
                              .arg(outputDir));
}

void AnnotationWindow::exportOdvg() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Export ODVG");
    const QString classesPath =
            QFileDialog::getOpenFileName(this, tr("Select classes file"),
                                         m_directory, tr("Text files (*.txt)"));
    if (classesPath.isEmpty()) return;
    QFile classesFile(classesPath);
    if (!classesFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, title, classesFile.errorString());
        return;
    }
    QStringList classes;
    for (const QByteArray& line : classesFile.readAll().split('\n')) {
        const QString name = QString::fromUtf8(line).trimmed();
        if (!name.isEmpty()) classes.append(name);
    }
    if (classes.isEmpty() ||
        QSet<QString>(classes.cbegin(), classes.cend()).size() !=
                classes.size()) {
        QMessageBox::warning(this, title,
                             tr("Classes must be nonempty and unique"));
        return;
    }
    const QString outputDir = QFileDialog::getExistingDirectory(
            this, tr("Export directory"),
            m_outputDir.isEmpty()
                    ? QDir(m_directory).filePath(QStringLiteral("../odvg"))
                    : m_outputDir);
    if (outputDir.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QDir(outputDir).canonicalPath();
    if (source == destination ||
        source.startsWith(destination + QDir::separator()) ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, title,
                tr("Choose a directory outside the source image directory"));
        return;
    }
    QVector<CocoImage> images;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        QImageReader reader(image.absoluteFilePath());
        const QSize size =
                reader.size().isValid() ? reader.size() : reader.read().size();
        QJsonObject document;
        QFile sidecar(QDir(image.absolutePath())
                              .filePath(image.completeBaseName() +
                                        QStringLiteral(".json")));
        if (sidecar.exists()) {
            if (!sidecar.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, sidecar.errorString());
                return;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(sidecar.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(this, title,
                                     tr("Invalid image annotation: %1")
                                             .arg(sidecar.fileName()));
                return;
            }
            document = parsed.object();
        }
        document.insert(QStringLiteral("imageWidth"), size.width());
        document.insert(QStringLiteral("imageHeight"), size.height());
        images.append({image.fileName(), document});
    }
    const OdvgConversion converted = convertOdvg(images, classes);
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, title, converted.error);
        return;
    }
    const QVector<QPair<QString, QByteArray>> files{
            {QStringLiteral("label_map.json"), converted.labelMap},
            {QStringLiteral("od.json"), converted.annotations}};
    int existing = 0;
    for (const auto& file : files)
        if (QFileInfo::exists(QDir(outputDir).filePath(file.first))) ++existing;
    if (existing &&
        QMessageBox::question(
                this, title,
                tr("Replace %1 existing ODVG output files?").arg(existing),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes)
        return;
    for (const auto& file : files) {
        QSaveFile output(QDir(outputDir).filePath(file.first));
        if (!output.open(QIODevice::WriteOnly) ||
            output.write(file.second) != file.second.size() ||
            !output.commit()) {
            QMessageBox::warning(
                    this, title,
                    tr("Unable to write %1: %2")
                            .arg(output.fileName(), output.errorString()));
            return;
        }
    }
    m_status->setText(tr("Exported %1 ODVG boxes to %2")
                              .arg(converted.count)
                              .arg(outputDir));
}

void AnnotationWindow::exportVlm() {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = tr("Export VLM-R1 OVD");
    QDialog options(this);
    options.setWindowTitle(title);
    auto* layout = new QVBoxLayout(&options);
    layout->addWidget(new QLabel(tr("Image path prefix"), &options));
    auto* prefix = new QLineEdit(&options);
    layout->addWidget(prefix);
    layout->addWidget(new QLabel(tr("Classes file (optional)"), &options));
    auto* classesPath = new QLineEdit(&options);
    auto* classesRow = new QHBoxLayout;
    classesRow->addWidget(classesPath);
    auto* browse = new QPushButton(tr("Browse"), &options);
    classesRow->addWidget(browse);
    layout->addLayout(classesRow);
    connect(browse, &QPushButton::clicked, &options, [this, classesPath]() {
        const QString path = QFileDialog::getOpenFileName(
                this, tr("Select classes file"), m_directory,
                tr("Text files (*.txt)"));
        if (!path.isEmpty()) classesPath->setText(path);
    });
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &options);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &options, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &options, &QDialog::reject);
    if (options.exec() != QDialog::Accepted) return;
    QStringList classes;
    if (!classesPath->text().trimmed().isEmpty()) {
        QFile file(classesPath->text().trimmed());
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, title, file.errorString());
            return;
        }
        for (const QByteArray& line : file.readAll().split('\n')) {
            const QString name = QString::fromUtf8(line).trimmed();
            if (!name.isEmpty()) classes.append(name);
        }
        if (classes.isEmpty() ||
            QSet<QString>(classes.cbegin(), classes.cend()).size() !=
                    classes.size()) {
            QMessageBox::warning(this, title,
                                 tr("Classes must be nonempty and unique"));
            return;
        }
    }
    const QString outputPath = QFileDialog::getSaveFileName(
            this, title,
            QDir(m_directory).filePath(QStringLiteral("../vlm_r1_ovd.jsonl")),
            tr("JSON lines (*.jsonl)"));
    if (outputPath.isEmpty()) return;
    const QString source = QDir(m_directory).canonicalPath();
    const QString destination = QFileInfo(outputPath).absolutePath();
    if (source == destination ||
        destination.startsWith(source + QDir::separator())) {
        QMessageBox::warning(
                this, title,
                tr("Choose a file outside the source image directory"));
        return;
    }
    QVector<CocoImage> images;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        QImageReader reader(image.absoluteFilePath());
        const QSize size =
                reader.size().isValid() ? reader.size() : reader.read().size();
        QJsonObject document;
        QFile sidecar(QDir(image.absolutePath())
                              .filePath(image.completeBaseName() +
                                        QStringLiteral(".json")));
        if (sidecar.exists()) {
            if (!sidecar.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, sidecar.errorString());
                return;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(sidecar.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(this, title,
                                     tr("Invalid image annotation: %1")
                                             .arg(sidecar.fileName()));
                return;
            }
            document = parsed.object();
        }
        document.insert(QStringLiteral("imageWidth"), size.width());
        document.insert(QStringLiteral("imageHeight"), size.height());
        images.append({image.fileName(), document});
    }
    const VlmConversion converted =
            convertVlm(images, classes, prefix->text().trimmed());
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, title, converted.error);
        return;
    }
    if (QFile::exists(outputPath) &&
        QMessageBox::question(this, title,
                              tr("Replace the existing VLM-R1 JSONL file?"),
                              QMessageBox::Yes | QMessageBox::No,
                              QMessageBox::No) != QMessageBox::Yes)
        return;
    QSaveFile output(outputPath);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(converted.annotations) != converted.annotations.size() ||
        !output.commit()) {
        QMessageBox::warning(this, title, output.errorString());
        return;
    }
    m_status->setText(tr("Exported %1 VLM-R1 boxes to %2")
                              .arg(converted.count)
                              .arg(outputPath));
}

void AnnotationWindow::exportPpocr(PpocrMode mode) {
    if (m_imagePath.isEmpty() || !confirmSave()) return;
    const QString title = mode == PpocrMode::Rec ? tr("Export PPOCR Rec")
                                                 : tr("Export PPOCR KIE");
    if (mode == PpocrMode::Rec) {
        const QString outputDirectory = QFileDialog::getExistingDirectory(
                this, title, m_outputDir.isEmpty() ? m_directory : m_outputDir,
                QFileDialog::ShowDirsOnly);
        if (outputDirectory.isEmpty()) return;
        if ((QFile::exists(QDir(outputDirectory)
                                   .filePath(QStringLiteral("Label.txt"))) ||
             QFile::exists(QDir(outputDirectory)
                                   .filePath(QStringLiteral("rec_gt.txt"))) ||
             QDir(QDir(outputDirectory).filePath(QStringLiteral("crop_img")))
                     .exists()) &&
            QMessageBox::question(this, title,
                                  tr("Replace the existing PPOCR Rec dataset?"),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes)
            return;
        QVector<PpocrDatasetImage> dataset;
        for (int i = 0; i < m_files->count(); ++i) {
            const QString imagePath =
                    m_files->item(i)->data(Qt::UserRole).toString();
            const QFileInfo info(imagePath);
            QFile sidecar(QDir(info.absolutePath())
                                  .filePath(info.completeBaseName() +
                                            QStringLiteral(".json")));
            if (!sidecar.exists()) continue;
            if (!sidecar.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, sidecar.errorString());
                return;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(sidecar.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject()) {
                QMessageBox::warning(this, title,
                                     tr("Invalid image annotation: %1")
                                             .arg(sidecar.fileName()));
                return;
            }
            dataset.push_back({imagePath, parsed.object()});
        }
        const PpocrRecExport exported =
                exportPpocrRecDataset(dataset, outputDirectory);
        if (!exported.error.isEmpty()) {
            QMessageBox::warning(this, title, exported.error);
            return;
        }
        m_status->setText(tr("Exported %1 PPOCR annotations and %2 crops to %3")
                                  .arg(exported.annotationCount)
                                  .arg(exported.cropCount)
                                  .arg(outputDirectory));
        return;
    }
    const QString outputPath = QFileDialog::getSaveFileName(
            this, title,
            QDir(m_directory).filePath(QStringLiteral("ppocr_kie.json")),
            tr("Text files (*.txt *.json)"));
    if (outputPath.isEmpty()) return;
    const QString expectedSuffix = QStringLiteral("json");
    if (QFileInfo(outputPath)
                .suffix()
                .compare(expectedSuffix, Qt::CaseInsensitive) != 0) {
        QMessageBox::warning(
                this, title,
                tr("Choose a .%1 output file").arg(expectedSuffix));
        return;
    }
    QVector<CocoImage> images;
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        QJsonObject document;
        QFile sidecar(QDir(image.absolutePath())
                              .filePath(image.completeBaseName() +
                                        QStringLiteral(".json")));
        if (sidecar.exists()) {
            if (!sidecar.open(QIODevice::ReadOnly)) {
                QMessageBox::warning(this, title, sidecar.errorString());
                return;
            }
            QJsonParseError error;
            const QJsonDocument parsed =
                    QJsonDocument::fromJson(sidecar.readAll(), &error);
            if (error.error != QJsonParseError::NoError || !parsed.isObject() ||
                parsed.object().value(QStringLiteral("type")).toString() ==
                        QStringLiteral("video_classification")) {
                QMessageBox::warning(this, title,
                                     tr("Invalid image annotation: %1")
                                             .arg(sidecar.fileName()));
                return;
            }
            document = parsed.object();
        }
        images.append({image.fileName(), document});
    }
    const PpocrConversion converted = convertPpocr(images, mode);
    if (!converted.error.isEmpty()) {
        QMessageBox::warning(this, title, converted.error);
        return;
    }
    for (int i = 0; i < m_files->count(); ++i) {
        const QFileInfo image(m_files->item(i)->data(Qt::UserRole).toString());
        const QString sidecar = QDir(image.absolutePath())
                                        .filePath(image.completeBaseName() +
                                                  QStringLiteral(".json"));
        if (QFileInfo(outputPath).absoluteFilePath() ==
            QFileInfo(sidecar).absoluteFilePath()) {
            QMessageBox::warning(
                    this, title,
                    tr("PPOCR output cannot overwrite an image sidecar"));
            return;
        }
    }
    if (QFile::exists(outputPath) &&
        QMessageBox::question(this, title,
                              tr("Replace the existing PPOCR file?"),
                              QMessageBox::Yes | QMessageBox::No,
                              QMessageBox::No) != QMessageBox::Yes)
        return;
    QSaveFile output(outputPath);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(converted.text) != converted.text.size() ||
        !output.commit()) {
        QMessageBox::warning(this, title, output.errorString());
        return;
    }
    m_status->setText(tr("Exported %1 PPOCR annotations to %2")
                              .arg(converted.count)
                              .arg(outputPath));
}

void AnnotationWindow::updateObjects() {
    const int selected = m_canvas->selectedIndex();
    m_objects->blockSignals(true);
    m_objects->clear();
    QStringList names;
    QHash<QString, int> classIds;
    int index = 0;
    for (const AnnotationShape& shape : m_canvas->shapes()) {
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        const QString groupText =
                group.isDouble() ? QStringLiteral(" [%1]").arg(group.toInt())
                                 : QString();
        auto* item = new QListWidgetItem(
                QStringLiteral("%1  %2%3  (%4)")
                        .arg(++index)
                        .arg(shape.label, groupText, shape.type),
                m_objects);
        item->setForeground(annotationShapeColor(shape.label, shape.extra));
        if (!shape.visible)
            item->setIcon(QIcon(iconPath(QStringLiteral("eye-off.svg"))));
        else if (shape.locked)
            item->setIcon(QIcon(iconPath(QStringLiteral("lock.svg"))));
        if (!shape.label.isEmpty() && !names.contains(shape.label))
            names.append(shape.label);
        const QJsonValue classId =
                shape.extra.value(QStringLiteral("class_id"));
        if (classId.isDouble() && !classIds.contains(shape.label))
            classIds.insert(shape.label, classId.toInt());
    }
    if (m_selectToggleShapesAction) {
        const QVector<AnnotationShape>& shapes = m_canvas->shapes();
        const bool allVisible = !shapes.isEmpty() &&
                                std::all_of(shapes.cbegin(), shapes.cend(),
                                            [](const AnnotationShape& shape) {
                                                return shape.visible;
                                            });
        const QString tooltip =
                allVisible ? tr("Hide all shapes") : tr("Show all shapes");
        m_selectToggleShapesAction->setEnabled(!shapes.isEmpty());
        m_selectToggleShapesAction->setIcon(
                QIcon(iconPath(allVisible ? QStringLiteral("eye.svg")
                                          : QStringLiteral("hidden.png"))));
        m_selectToggleShapesAction->setToolTip(tooltip);
        m_selectToggleShapesAction->setStatusTip(tooltip);
    }
    for (int row : m_canvas->selectedIndices())
        if (row >= 0 && row < m_objects->count())
            m_objects->item(row)->setSelected(true);
    m_objects->setCurrentRow(selected, QItemSelectionModel::NoUpdate);
    m_objects->blockSignals(false);
    m_labels->blockSignals(true);
    m_labels->clear();
    m_labels->addItems(names);
    m_labels->blockSignals(false);
    if (m_brushTarget) {
        const QString previous =
                selected >= 0 && selected < m_canvas->shapes().size()
                        ? m_canvas->shapes().at(selected).label
                        : m_brushTarget->currentText();
        const QSignalBlocker blocked(m_brushTarget);
        m_brushTarget->clear();
        for (const QString& name : names)
            m_brushTarget->addItem(name,
                                   classIds.contains(name)
                                           ? QVariant(classIds.value(name))
                                           : QVariant());
        m_brushTarget->setEditText(previous);
        m_canvas->setBrushLabel(previous);
        m_canvas->setBrushClassId(
                selected >= 0 && selected < m_canvas->shapes().size()
                        ? m_canvas->shapes()
                                  .at(selected)
                                  .extra.value(QStringLiteral("class_id"))
                                  .toInt(-1)
                        : -1);
    }
    updateNavigator();
    updateDescriptionEditor();
    applyObjectFilter();
}

void AnnotationWindow::updateDescriptionEditor() {
    if (!m_description) return;
    const QSignalBlocker blocked(m_description);
    if (m_imagePath.isEmpty()) {
        m_descriptionTitle->setText(tr("Description"));
        m_description->clear();
        m_description->setEnabled(false);
        return;
    }
    m_description->setEnabled(true);
    const QSet<int> selected = m_canvas->selectedIndices();
    const int selectedIndex =
            selected.size() == 1 ? *selected.constBegin() : -1;
    if (selectedIndex >= 0 && selectedIndex < m_canvas->shapes().size()) {
        m_descriptionTitle->setText(tr("Object Description"));
        m_description->setPlainText(
                m_canvas->shapes()
                        .at(selectedIndex)
                        .extra.value(QStringLiteral("description"))
                        .toString());
    } else {
        m_descriptionTitle->setText(tr("Image Description"));
        m_description->setPlainText(
                m_document.value(QStringLiteral("description")).toString());
    }
}

void AnnotationWindow::importImageFlags() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select a specific flags file"), QString(),
            tr("Image Flags Files (*.txt);;All Files (*)"));
    if (path.isEmpty()) return;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Image Flags"),
                             tr("Unable to read %1").arg(path));
        return;
    }
    const QByteArray bytes = file.readAll();
    QString contents = QString::fromUtf8(bytes.constData(), bytes.size());
    if (contents.toUtf8() != bytes) {
        QMessageBox::warning(this, tr("Image Flags"),
                             tr("The flags file is not valid UTF-8."));
        return;
    }

    contents.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    contents.replace(QChar('\r'), QChar('\n'));
    if (contents.isEmpty()) {
        m_imageFlagNames.clear();
    } else {
        m_imageFlagNames = contents.split(QChar('\n'), Qt::KeepEmptyParts);
        if (contents.endsWith(QChar('\n'))) m_imageFlagNames.removeLast();
    }
    updateFlags();
    m_status->setText(
            tr("Image flags loaded from %1").arg(QFileInfo(path).fileName()));
}

void AnnotationWindow::updateFlags() {
    m_updating = true;
    m_flags->clear();
    QJsonObject flags = m_document.value(QStringLiteral("flags")).toObject();
    QStringList names = m_imageFlagNames;
    for (const QString& name : flags.keys())
        if (!names.contains(name)) names.append(name);
    for (const QString& name : m_imageFlagNames)
        if (!flags.contains(name)) flags.insert(name, false);
    if (!m_imagePath.isEmpty() && !m_sidecarBlocked)
        m_document.insert(QStringLiteral("flags"), flags);
    for (const QString& name : names) {
        auto* item = new QListWidgetItem(name, m_flags);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(flags.value(name).toBool() ? Qt::Checked
                                                       : Qt::Unchecked);
    }
    m_flagsDock->setVisible(!m_imageFlagNames.isEmpty());
    m_updating = false;
}

void AnnotationWindow::updateTags() {
    setTagBatchMode(false);
    const QSignalBlocker blocked(m_tags->model());
    const QSignalBlocker listBlocked(m_tags);
    m_tags->clear();
    QSet<QString> seen;
    for (const QJsonValue& value :
         m_document.value(QStringLiteral("tags")).toArray()) {
        if (!value.isString()) continue;
        const QString tag = value.toString().trimmed();
        if (tag.isEmpty() || tag.contains(QChar('\r')) ||
            tag.contains(QChar('\n')) || seen.contains(tag))
            continue;
        seen.insert(tag);
        auto* item = new QListWidgetItem(tag, m_tags);
        item->setData(Qt::UserRole, tag);
    }
    if (!m_tagsVisibilityExplicit) {
        const QSignalBlocker actionBlocked(m_tagsAction);
        const bool visible = m_document.contains(QStringLiteral("tags"));
        m_tagsAction->setChecked(visible);
        m_tagsPanel->setVisible(visible);
    }
    m_tagsPanel->setEnabled(!m_imagePath.isEmpty());
    for (int i = 0; i < m_tags->count(); ++i) {
        QListWidgetItem* item = m_tags->item(i);
        QJsonObject extra;
        for (const AnnotationShape& shape : m_canvas->shapes()) {
            if (shape.label == item->text()) {
                extra = shape.extra;
                break;
            }
        }
        const QColor color = annotationShapeColor(item->text(), extra);
        item->setBackground(color);
        item->setForeground(color.lightness() > 150 ? QColor(Qt::black)
                                                    : QColor(Qt::white));
        item->setSizeHint(QSize(
                std::min(240,
                         fontMetrics().horizontalAdvance(item->text()) + 26),
                28));
    }
}

void AnnotationWindow::setTagBatchMode(bool enabled) {
    if (!m_tags) return;
    const QSignalBlocker blocked(m_tags);
    m_tagBatchMode = enabled;
    if (!enabled) m_tags->clearSelection();
    m_tags->setSelectionMode(enabled ? QAbstractItemView::MultiSelection
                                     : QAbstractItemView::SingleSelection);

    for (int i = 0; i < m_tags->count(); ++i) {
        QListWidgetItem* item = m_tags->item(i);
        QFont font = item->font();
        font.setStrikeOut(enabled && item->isSelected());
        item->setFont(font);
    }

    auto* add = findChild<QPushButton*>(QStringLiteral("AddImageTag"));
    auto* edit = findChild<QPushButton*>(QStringLiteral("EditImageTag"));
    auto* remove = findChild<QPushButton*>(QStringLiteral("DeleteImageTags"));
    auto* copy = findChild<QPushButton*>(QStringLiteral("CopyImageTags"));
    auto* batch =
            findChild<QPushButton*>(QStringLiteral("BatchDeleteImageTags"));
    if (add) add->setEnabled(!enabled);
    if (edit) edit->setEnabled(!enabled);
    if (copy) copy->setEnabled(!enabled);
    if (remove) {
        remove->setEnabled(!enabled || !m_tags->selectedItems().isEmpty());
        remove->setText(enabled ? tr("Delete selected") : tr("Delete"));
    }
    if (batch) {
        batch->setText(enabled ? QStringLiteral("x") : QStringLiteral("-"));
        batch->setToolTip(enabled ? tr("Cancel batch delete")
                                  : tr("Batch delete image tags"));
        batch->setAccessibleName(batch->toolTip());
    }
}

void AnnotationWindow::commitTagInput() {
    if (!m_tagInput || m_tagInput->isHidden() || m_imagePath.isEmpty()) return;
    const QString tag = normalizeImageTag(m_tagInput->text());
    if (tag.isEmpty()) return;
    for (int i = 0; i < m_tags->count(); ++i)
        if (m_tags->item(i)->text() == tag) return;
    auto* item = new QListWidgetItem(tag, m_tags);
    item->setData(Qt::UserRole, tag);
    m_tagInput->clear();
    m_tagInput->hide();
    syncTagsToDocument();
}

void AnnotationWindow::cancelTagInput() {
    if (!m_tagInput) return;
    m_tagInput->clear();
    m_tagInput->hide();
}

void AnnotationWindow::syncTagsToDocument() {
    if (m_imagePath.isEmpty()) return;
    QJsonArray tags;
    for (int i = 0; i < m_tags->count(); ++i)
        tags.append(m_tags->item(i)->text());
    if (m_document.value(QStringLiteral("tags")) == QJsonValue(tags)) return;
    m_document.insert(QStringLiteral("tags"), tags);
    setDirty(true);
    updateTags();
}

QString AnnotationWindow::sidecarPathFor(const QString& imagePath) const {
    // Source contract: when an output directory is active only that
    // directory is consulted for the per-image sidecar (load, save and
    // checked-state discovery); otherwise the sidecar sits beside the
    // image.
    const QFileInfo info(imagePath);
    if (!m_outputDir.isEmpty())
        return QDir(m_outputDir)
                .filePath(info.completeBaseName() + QStringLiteral(".json"));
    return QDir(info.absolutePath())
            .filePath(info.completeBaseName() + QStringLiteral(".json"));
}

QStringList AnnotationWindow::sidecarCandidates(
        const QString& imagePath) const {
    QStringList candidates;
    const QFileInfo info(imagePath);
    if (!m_outputDir.isEmpty())
        candidates.append(QDir(m_outputDir)
                                  .filePath(info.completeBaseName() +
                                            QStringLiteral(".json")));
    const QString local = QDir(info.absolutePath())
                                  .filePath(info.completeBaseName() +
                                            QStringLiteral(".json"));
    if (!candidates.contains(local)) candidates.append(local);
    return candidates;
}

void AnnotationWindow::changeOutputDir() {
    QString suggested = m_outputDir;
    if (suggested.isEmpty())
        suggested = m_imagePath.isEmpty()
                            ? m_directory
                            : QFileInfo(m_imagePath).absolutePath();
    const QString dir = QFileDialog::getExistingDirectory(
            this, tr("Change Output Dir"), suggested,
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (dir.isEmpty()) return;
    m_outputDir = dir;
    // The source re-imports the folder without reloading and keeps the
    // current selection; the in-memory document stays as-is and the next
    // save goes to the new directory.
    const QString currentImage = m_imagePath;
    refreshFileList();
    if (!currentImage.isEmpty()) {
        const auto found = m_files->findItems(
                QFileInfo(currentImage).fileName(), Qt::MatchExactly);
        if (!found.isEmpty()) m_files->setCurrentItem(found.first());
    }
    updateActionState();
    m_status->setText(tr("Annotations will be saved/loaded in %1").arg(dir));
}

void AnnotationWindow::updateFileStatus() {
    for (int i = 0; i < m_files->count(); ++i) {
        QListWidgetItem* item = m_files->item(i);
        const QStringList candidates =
                sidecarCandidates(item->data(Qt::UserRole).toString());
        const QString sidecar = candidates.first();
        item->setForeground(QFile::exists(sidecar) ? QColor(24, 130, 91)
                                                   : palette().text().color());
        bool checked = false;
        // candidates.first() already resolves to the active output directory
        // when one is set, matching _refresh_uploaded_file_items.
        QFile input(sidecar);
        if (input.open(QIODevice::ReadOnly)) {
            const QJsonDocument document =
                    QJsonDocument::fromJson(input.readAll());
            checked =
                    document.isObject() &&
                    document.object().value(QStringLiteral("checked")).toBool();
        }
        item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
        item->setFlags(item->flags() & ~Qt::ItemIsUserCheckable);
    }
}

void AnnotationWindow::snapshot() {
    m_undo.append(m_canvas->shapes());
    if (m_undo.size() > 50) m_undo.removeFirst();
    m_redo.clear();
    m_undoAction->setEnabled(true);
    m_redoAction->setEnabled(false);
}

void AnnotationWindow::undo() {
    if (m_canvas->brushEditing()) {
        m_canvas->brushUndo();
        return;
    }
    if (m_undo.isEmpty()) return;
    m_redo.append(m_canvas->shapes());
    m_canvas->setShapes(m_undo.takeLast());
    m_undoAction->setEnabled(!m_undo.isEmpty());
    m_redoAction->setEnabled(true);
    updateObjects();
    setDirty(true);
}

void AnnotationWindow::redo() {
    if (m_canvas->brushEditing()) {
        m_canvas->brushRedo();
        return;
    }
    if (m_redo.isEmpty()) return;
    m_undo.append(m_canvas->shapes());
    m_canvas->setShapes(m_redo.takeLast());
    m_undoAction->setEnabled(true);
    m_redoAction->setEnabled(!m_redo.isEmpty());
    updateObjects();
    setDirty(true);
}

void AnnotationWindow::groupSelectedShapes() {
    const QList<QListWidgetItem*> selected = m_objects->selectedItems();
    if (selected.isEmpty()) return;
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    QSet<int> groupIds;
    QVector<int> rows;
    bool hasUngrouped = false;
    int newGroup = 0;
    for (QListWidgetItem* item : selected) {
        const int row = m_objects->row(item);
        if (row < 0 || row >= shapes.size()) continue;
        rows.append(row);
        const QJsonValue group =
                shapes[row].extra.value(QStringLiteral("group_id"));
        if (group.isDouble())
            groupIds.insert(group.toInt());
        else
            hasUngrouped = true;
    }
    if (rows.isEmpty()) return;
    for (int row : rows)
        if (shapes[row].locked) return;
    if (!groupIds.isEmpty()) {
        newGroup = *std::min_element(groupIds.cbegin(), groupIds.cend());
    } else {
        for (const AnnotationShape& shape : shapes) {
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (group.isDouble())
                newGroup = std::max(newGroup, group.toInt() + 1);
        }
    }
    if (!hasUngrouped && groupIds.size() == 1) return;
    snapshot();
    for (int row : rows) {
        shapes[row].extra.insert(QStringLiteral("group_id"), newGroup);
    }
    for (AnnotationShape& shape : shapes) {
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        if (group.isDouble() && groupIds.contains(group.toInt()) &&
            group.toInt() != newGroup)
            shape.extra.insert(QStringLiteral("group_id"), newGroup);
    }
    m_canvas->setShapes(shapes);
    updateObjects();
    setDirty(true);
}

void AnnotationWindow::ungroupSelectedShapes() {
    const QList<QListWidgetItem*> selected = m_objects->selectedItems();
    if (selected.isEmpty()) return;
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    QSet<int> groupIds;
    for (QListWidgetItem* item : selected) {
        const int row = m_objects->row(item);
        if (row < 0 || row >= shapes.size()) continue;
        const QJsonValue group =
                shapes[row].extra.value(QStringLiteral("group_id"));
        if (group.isDouble()) groupIds.insert(group.toInt());
    }
    if (groupIds.isEmpty()) return;
    for (QListWidgetItem* item : selected) {
        const int row = m_objects->row(item);
        if (row >= 0 && row < shapes.size() && shapes[row].locked) return;
    }
    snapshot();
    for (AnnotationShape& shape : shapes) {
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        if (!shape.locked && group.isDouble() &&
            groupIds.contains(group.toInt()))
            shape.extra.insert(QStringLiteral("group_id"), QJsonValue::Null);
    }
    m_canvas->setShapes(shapes);
    updateObjects();
    setDirty(true);
}

void AnnotationWindow::unionSelectedShapes() {
    const QList<QListWidgetItem*> selected = m_objects->selectedItems();
    if (selected.size() < 2) return;
    const QVector<AnnotationShape> source = m_canvas->shapes();
    QVector<int> rows;
    rows.reserve(selected.size());
    for (QListWidgetItem* item : selected) {
        const int row = m_objects->row(item);
        if (row >= 0 && row < source.size()) rows.append(row);
    }
    if (rows.size() < 2) return;
    for (int row : rows)
        if (source[row].locked) return;
    AnnotationShape merged = source[rows.last()];
    bool hasRectangle = false;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (int row : rows) {
        const AnnotationShape& shape = source[row];
        if (shape.type == QStringLiteral("rectangle") &&
            shape.points.size() >= 4) {
            hasRectangle = true;
            for (const QPointF& point : shape.points) {
                minX = std::min(minX, point.x());
                minY = std::min(minY, point.y());
                maxX = std::max(maxX, point.x());
                maxY = std::max(maxY, point.y());
            }
        }
    }
    if (hasRectangle) {
        merged.type = QStringLiteral("rectangle");
        merged.points = {QPointF(minX, minY), QPointF(maxX, minY),
                         QPointF(maxX, maxY), QPointF(minX, maxY)};
    } else {
        QPainterPath path;
        bool first = true;
        for (int row : rows) {
            const QVector<QPointF>& points = source[row].points;
            if (points.size() < 3) continue;
            QPolygonF polygon;
            for (const QPointF& point : points) polygon.append(point);
            QPainterPath part;
            part.addPolygon(polygon);
            if (first) {
                path = part;
                first = false;
            } else
                path = path.united(part);
        }
        const QList<QPolygonF> polygons = path.toSubpathPolygons();
        if (polygons.isEmpty()) return;
        auto area = [](const QPolygonF& polygon) {
            double value = 0;
            for (int i = 0; i < polygon.size(); ++i) {
                const QPointF& a = polygon[i];
                const QPointF& b = polygon[(i + 1) % polygon.size()];
                value += a.x() * b.y() - b.x() * a.y();
            }
            return std::abs(value);
        };
        const QPolygonF* largest = &polygons.first();
        for (const QPolygonF& polygon : polygons)
            if (area(polygon) > area(*largest)) largest = &polygon;
        merged.type = QStringLiteral("polygon");
        merged.points.clear();
        for (const QPointF& point : *largest) merged.points.append(point);
    }
    snapshot();
    QVector<AnnotationShape> shapes;
    shapes.reserve(source.size() - rows.size() + 1);
    QSet<int> removed(rows.cbegin(), rows.cend());
    for (int i = 0; i < source.size(); ++i)
        if (!removed.contains(i)) shapes.append(source[i]);
    shapes.append(merged);
    m_canvas->setShapes(shapes);
    m_canvas->selectShape(shapes.size() - 1);
    updateObjects();
    setDirty(true);
}

void AnnotationWindow::copyShapes() {
    QJsonArray items;
    for (QListWidgetItem* item : m_objects->selectedItems()) {
        const int index = m_objects->row(item);
        if (index >= 0 && index < m_canvas->shapes().size())
            items.append(m_canvas->shapes()[index].toJson());
    }
    if (items.isEmpty() && m_canvas->selectedIndex() >= 0)
        items.append(m_canvas->shapes()[m_canvas->selectedIndex()].toJson());
    if (items.isEmpty()) return;
    m_internalShapeClipboard = items;
    if (m_systemClipboard) {
        auto* payload = new QMimeData;
        payload->setData("application/x-xanylabeling-shapes+json",
                         QJsonDocument(items).toJson(QJsonDocument::Compact));
        QApplication::clipboard()->setMimeData(payload);
    }
    m_pasteCount = 0;
    if (m_pasteAction) m_pasteAction->setEnabled(true);
    m_status->setText(tr("Copied %1 objects").arg(items.size()));
}

void AnnotationWindow::copyShapeCoordinates() {
    if (m_imagePath.isEmpty()) return;
    const int index = m_canvas->selectedIndex();
    if (index < 0 || index >= m_canvas->shapes().size()) return;
    const AnnotationShape& shape = m_canvas->shapes()[index];
    QStringList parts;
    if (shape.type == QStringLiteral("rectangle")) {
        // Source format: integer [x1, y1, x2, y2] from the outer corners.
        if (shape.points.size() < 3) return;
        parts << QString::number(int(shape.points[0].x()))
              << QString::number(int(shape.points[0].y()))
              << QString::number(int(shape.points[2].x()))
              << QString::number(int(shape.points[2].y()));
    } else {
        for (const QPointF& point : shape.points) {
            parts << QString::number(point.x()) << QString::number(point.y());
        }
    }
    QApplication::clipboard()->setText(
            QStringLiteral("[%1]").arg(parts.join(", ")));
    m_status->setText(tr("Copied shape coordinates"));
}

void AnnotationWindow::populateLabelFilterMenu(QMenu* menu) {
    menu->clear();
    auto* group = new QActionGroup(menu);
    group->setExclusive(true);
    QSet<QString> labels;
    if (m_canvas)
        for (const AnnotationShape& shape : m_canvas->shapes())
            if (!shape.label.isEmpty()) labels.insert(shape.label);
    const auto addOption = [menu, group](const QString& text, bool checked,
                                         const std::function<void()>& run) {
        QAction* action = menu->addAction(text);
        action->setCheckable(true);
        action->setChecked(checked);
        QObject::connect(action, &QAction::triggered, menu, [run]() { run(); });
        group->addAction(action);
    };
    addOption(tr("All Labels"), m_labelFilter.isEmpty(), [this]() {
        m_labelFilter.clear();
        applyObjectFilter();
    });
    for (const QString& label : labels.values()) {
        addOption(label, m_labelFilter == label, [this, label]() {
            m_labelFilter = label;
            applyObjectFilter();
        });
    }
}

void AnnotationWindow::populateGidFilterMenu(QMenu* menu) {
    menu->clear();
    auto* group = new QActionGroup(menu);
    group->setExclusive(true);
    QSet<int> gids;
    if (m_canvas)
        for (const AnnotationShape& shape : m_canvas->shapes()) {
            const QJsonValue value =
                    shape.extra.value(QStringLiteral("group_id"));
            if (value.isDouble()) gids.insert(value.toInt());
            // Shapes without a group id are only visible under
            // "All Group IDs", matching the source filter semantics, so
            // they get no dedicated entry.
        }
    const auto addOption = [menu, group](const QString& text, bool checked,
                                         const std::function<void()>& run) {
        QAction* action = menu->addAction(text);
        action->setCheckable(true);
        action->setChecked(checked);
        QObject::connect(action, &QAction::triggered, menu, [run]() { run(); });
        group->addAction(action);
    };
    addOption(tr("All Group IDs"), m_gidFilter < 0, [this]() {
        m_gidFilter = -1;
        applyObjectFilter();
    });
    QList<int> sorted = gids.values();
    std::sort(sorted.begin(), sorted.end());
    for (int gid : sorted) {
        addOption(QString::number(gid), m_gidFilter == gid, [this, gid]() {
            m_gidFilter = gid;
            applyObjectFilter();
        });
    }
}

void AnnotationWindow::applyObjectFilter() {
    if (!m_objects) return;
    const QVector<AnnotationShape> shapes =
            m_canvas ? m_canvas->shapes() : QVector<AnnotationShape>();
    for (int row = 0; row < m_objects->count(); ++row) {
        bool visible = true;
        if (row < shapes.size()) {
            const AnnotationShape& shape = shapes[row];
            if (!m_labelFilter.isEmpty() && shape.label != m_labelFilter)
                visible = false;
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (m_gidFilter >= 0 &&
                !(group.isDouble() && group.toInt() == m_gidFilter))
                visible = false;
        }
        m_objects->item(row)->setHidden(!visible);
    }
}

void AnnotationWindow::applyFileListCheckboxPolicy() {
    if (!m_files) return;
    for (int row = 0; row < m_files->count(); ++row) {
        QListWidgetItem* item = m_files->item(row);
        Qt::ItemFlags flags = item->flags();
        flags.setFlag(Qt::ItemIsUserCheckable, m_fileListCheckboxEditable);
        item->setFlags(flags);
    }
}

QString AnnotationWindow::openLabelDialog(const QString& initial,
                                          bool* accepted,
                                          QJsonObject* flagsOut,
                                          const QJsonObject& currentFlags) {
    QStringList labels;
    for (int i = 0; i < m_labels->count(); ++i)
        labels.append(m_labels->item(i)->text());
    if (!initial.isEmpty() && !labels.contains(initial)) labels.append(initial);
    AnnotationLabelDialog dialog(labels, m_sortLabels, m_showLabelTextField,
                                 m_labelCompletion, m_fitToContentColumn,
                                 m_fitToContentRow, initial, this);
    // The Label Flags template renders checkbox fields that follow the
    // label being chosen; the edited shape's current values apply only
    // while its label is still selected.
    if (flagsOut) {
        QString appliedLabel;
        const auto syncFlags = [this, &dialog, &appliedLabel, initial,
                                &currentFlags](const QString& label) {
            if (label.isEmpty() || label == appliedLabel) return;
            appliedLabel = label;
            dialog.setFlagFields(
                    m_labelFlags.value(label),
                    label == initial ? currentFlags : QJsonObject{});
        };
        syncFlags(initial);
        connect(dialog.findChild<QListWidget*>(
                        QStringLiteral("AnnotationLabelDialogList")),
                &QListWidget::currentTextChanged, this,
                [&syncFlags](const QString& label) { syncFlags(label); });
    }
    const int code = dialog.exec();
    if (accepted) *accepted = code == QDialog::Accepted;
    if (flagsOut) *flagsOut = dialog.selectedFlags();
    return code == QDialog::Accepted ? dialog.selectedLabel() : QString();
}

void AnnotationWindow::uploadLabelFlags() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select a specific flags file"), m_directory,
            tr("Label Flags Files (*.yaml);;All Files (*)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Label Flags"), file.errorString());
        return;
    }
    const AnnotationYamlResult parsed =
            parseAnnotationYaml(QString::fromUtf8(file.readAll()));
    if (!parsed.error.isEmpty() ||
        parsed.root.type != AnnotationYamlNode::Mapping) {
        QMessageBox::warning(this, tr("Label Flags"),
                             parsed.error.isEmpty()
                                     ? tr("The flags file must be a mapping "
                                          "of label to flag list")
                                     : parsed.error);
        return;
    }
    QHash<QString, QStringList> flags;
    for (const auto& entry : parsed.root.mapping) {
        if (entry.second.type != AnnotationYamlNode::Sequence ||
            entry.second.sequence.isEmpty()) {
            QMessageBox::warning(this, tr("Label Flags"),
                                 tr("Label %1 needs a non-empty list of flags")
                                         .arg(entry.first));
            return;
        }
        QStringList list;
        for (const AnnotationYamlNode& item : entry.second.sequence) {
            if (item.type != AnnotationYamlNode::Scalar ||
                item.scalar.isEmpty()) {
                QMessageBox::warning(
                        this, tr("Label Flags"),
                        tr("Label %1 needs a non-empty list of flags")
                                .arg(entry.first));
                return;
            }
            list.append(item.scalar);
        }
        flags.insert(entry.first, list);
        // Missing labels join the label list like the source upload does.
        bool exists = false;
        for (int i = 0; i < m_labels->count(); ++i)
            exists |= m_labels->item(i)->text() == entry.first;
        if (!exists) m_labels->addItem(entry.first);
    }
    m_labelFlags = flags;
    m_status->setText(tr("Uploading flags file successfully!"));
}

void AnnotationWindow::uploadShapeAttributes() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select a specific shape attributes file"), m_directory,
            tr("Shape Attributes Files (*.json);;All Files (*)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Attributes"), file.errorString());
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument parsed =
            QJsonDocument::fromJson(file.readAll(), &parseError);
    const auto invalid = [this](const QString& detail) {
        QMessageBox::warning(
                this, tr("Attributes"),
                tr("Invalid shape attribute configuration: %1").arg(detail));
    };
    if (parseError.error != QJsonParseError::NoError || !parsed.isObject()) {
        invalid(QStringLiteral("the root must be a JSON object"));
        return;
    }
    // Mirrors the source validate_shape_attributes_config contract.
    static const QSet<QString> kWidgetTypes{
            QStringLiteral("combobox"), QStringLiteral("radiobutton"),
            QStringLiteral("group_id"), QStringLiteral("lineedit")};
    const QJsonObject root = parsed.object();
    QHash<QString, QJsonObject> attributes;
    for (auto it = root.constBegin(); it != root.constEnd(); ++it) {
        if (it.key().startsWith(QStringLiteral("__"))) continue;
        if (!it.value().isObject()) {
            invalid(QStringLiteral("label %1 needs a property object")
                            .arg(it.key()));
            return;
        }
        attributes.insert(it.key(), it.value().toObject());
    }
    const QJsonObject widgetTypes =
            root.value(QStringLiteral("__widget_types__")).toObject();
    if (!root.value(QStringLiteral("__widget_types__")).isUndefined() &&
        !root.value(QStringLiteral("__widget_types__")).isObject()) {
        invalid(QStringLiteral("__widget_types__ must be a mapping"));
        return;
    }
    for (auto labelIt = widgetTypes.constBegin();
         labelIt != widgetTypes.constEnd(); ++labelIt) {
        if (!attributes.contains(labelIt.key())) {
            invalid(QStringLiteral("widget type label %1 is not defined")
                            .arg(labelIt.key()));
            return;
        }
        const QJsonObject properties = labelIt.value().toObject();
        for (auto propIt = properties.constBegin();
             propIt != properties.constEnd(); ++propIt) {
            if (!attributes.value(labelIt.key()).contains(propIt.key())) {
                invalid(QStringLiteral("widget type property %1 is not "
                                       "defined for label %2")
                                .arg(propIt.key(), labelIt.key()));
                return;
            }
            if (!kWidgetTypes.contains(propIt.value().toString())) {
                invalid(QStringLiteral("property %1 needs one of combobox, "
                                       "radiobutton, group_id or lineedit")
                                .arg(propIt.key()));
                return;
            }
        }
    }
    for (auto labelIt = attributes.constBegin();
         labelIt != attributes.constEnd(); ++labelIt) {
        const QJsonObject types = widgetTypes.value(labelIt.key()).toObject();
        for (auto propIt = labelIt.value().constBegin();
             propIt != labelIt.value().constEnd(); ++propIt) {
            const QString type = types.value(propIt.key())
                                         .toString(QStringLiteral("combobox"));
            const QJsonValue value = propIt.value();
            if ((type == QStringLiteral("combobox") ||
                 type == QStringLiteral("radiobutton"))) {
                if (!value.isArray()) {
                    invalid(QStringLiteral("%1/%2 needs a list of options")
                                    .arg(labelIt.key(), propIt.key()));
                    return;
                }
                for (const QJsonValue& option : value.toArray())
                    if (!option.isString() || option.toString().isEmpty()) {
                        invalid(QStringLiteral("%1/%2 needs a list of "
                                               "option strings")
                                        .arg(labelIt.key(), propIt.key()));
                        return;
                    }
                if (value.toArray().isEmpty()) {
                    invalid(QStringLiteral("%1/%2 needs a non-empty list")
                                    .arg(labelIt.key(), propIt.key()));
                    return;
                }
            } else if (type == QStringLiteral("lineedit")) {
                if (!value.isString()) {
                    invalid(QStringLiteral("%1/%2 needs a string default")
                                    .arg(labelIt.key(), propIt.key()));
                    return;
                }
            } else if (type == QStringLiteral("group_id")) {
                if (!value.isArray() || !value.toArray().isEmpty()) {
                    invalid(QStringLiteral("%1/%2 must be an empty list")
                                    .arg(labelIt.key(), propIt.key()));
                    return;
                }
            }
        }
        // Attribute labels join the label list like the source upload.
        bool exists = false;
        for (int i = 0; i < m_labels->count(); ++i)
            exists |= m_labels->item(i)->text() == labelIt.key();
        if (!exists) m_labels->addItem(labelIt.key());
    }
    m_attributes = attributes;
    m_attributeWidgetTypes.clear();
    for (auto it = widgetTypes.constBegin(); it != widgetTypes.constEnd(); ++it)
        m_attributeWidgetTypes.insert(it.key(), it.value().toObject());
    m_status->setText(tr("Uploading shape attributes file successfully!"));
}

void AnnotationWindow::uploadLabelClasses() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Select a specific label classes file"), m_directory,
            tr("Label Files (*.txt);;All Files (*)"));
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Label Classes"), file.errorString());
        return;
    }
    QStringList labels;
    QSet<QString> unique;
    for (const QByteArray& line : file.readAll().split('\n')) {
        const QString name = QString::fromUtf8(line).trimmed();
        if (name.isEmpty() || unique.contains(name)) continue;
        unique.insert(name);
        labels.append(name);
    }
    if (labels.isEmpty()) {
        QMessageBox::warning(this, tr("Label Classes"),
                             tr("No labels found in the file!"));
        return;
    }
    if (QMessageBox::warning(this, tr("Warning"),
                             tr("Current labels will be lost"),
                             QMessageBox::Ok | QMessageBox::Cancel,
                             QMessageBox::Cancel) != QMessageBox::Ok)
        return;
    m_labels->clear();
    m_labels->addItems(labels);
    m_status->setText(tr("Successfully loaded %1 labels!").arg(labels.size()));
}

void AnnotationWindow::editShapeAttributes() {
    int index = m_canvas->selectedIndex();
    const QList<QListWidgetItem*> selected = m_objects->selectedItems();
    if (!selected.isEmpty()) index = m_objects->row(selected.first());
    if (index < 0 || index >= m_canvas->shapes().size()) return;
    const AnnotationShape current = m_canvas->shapes().at(index);
    const QJsonObject definitions = m_attributes.value(current.label);
    const QJsonObject types = m_attributeWidgetTypes.value(current.label);
    const QJsonObject existing =
            current.extra.value(QStringLiteral("attributes")).toObject();
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Shape attributes - %1").arg(current.label));
    auto* form = new QFormLayout(&dialog);
    struct Field {
        QString property;
        QString type;
        QWidget* widget;
    };
    QVector<Field> fields;
    QSpinBox* gidBox = nullptr;
    for (auto it = definitions.constBegin(); it != definitions.constEnd();
         ++it) {
        const QString type =
                types.value(it.key()).toString(QStringLiteral("combobox"));
        if (type == QStringLiteral("group_id")) {
            gidBox = new QSpinBox(&dialog);
            gidBox->setRange(0, 999999);
            gidBox->setValue(
                    current.extra.value(QStringLiteral("group_id")).toInt());
            form->addRow(it.key(), gidBox);
            fields.append({it.key(), type, gidBox});
            continue;
        }
        if (type == QStringLiteral("lineedit")) {
            auto* edit =
                    new QLineEdit(existing.value(it.key()).toString(), &dialog);
            form->addRow(it.key(), edit);
            fields.append({it.key(), type, edit});
            continue;
        }
        auto* combo = new QComboBox(&dialog);
        for (const QJsonValue& option : it.value().toArray())
            combo->addItem(option.toString());
        combo->setCurrentText(existing.value(it.key()).toString());
        form->addRow(it.key(), combo);
        fields.append({it.key(), type, combo});
    }
    if (fields.isEmpty()) return;
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted) return;
    snapshot();
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    QJsonObject attributes =
            shapes[index].extra.value(QStringLiteral("attributes")).toObject();
    for (const Field& field : fields) {
        if (field.type == QStringLiteral("group_id")) {
            shapes[index].extra.insert(QStringLiteral("group_id"),
                                       gidBox->value());
            continue;
        }
        if (field.type == QStringLiteral("lineedit"))
            attributes.insert(field.property,
                              static_cast<QLineEdit*>(field.widget)->text());
        else
            attributes.insert(
                    field.property,
                    static_cast<QComboBox*>(field.widget)->currentText());
    }
    shapes[index].extra.insert(QStringLiteral("attributes"), attributes);
    m_canvas->setShapes(shapes);
    m_canvas->selectShape(index);
    updateObjects();
    setDirty(true);
    m_status->setText(tr("Shape attributes updated"));
}

void AnnotationWindow::pasteShapes() {
    if (m_image.isNull()) return;
    QJsonArray items;
    if (m_systemClipboard) {
        const QMimeData* payload = QApplication::clipboard()->mimeData();
        if (!payload ||
            !payload->hasFormat("application/x-xanylabeling-shapes+json"))
            return;
        QJsonParseError error;
        const QJsonDocument document = QJsonDocument::fromJson(
                payload->data("application/x-xanylabeling-shapes+json"),
                &error);
        if (error.error != QJsonParseError::NoError || !document.isArray())
            return;
        items = document.array();
    } else {
        items = m_internalShapeClipboard;
    }
    if (items.isEmpty()) return;
    QVector<AnnotationShape> copied;
    QRectF bounds;
    bool hasPoint = false;
    for (const QJsonValue& value : items) {
        if (!value.isObject()) return;
        AnnotationShape shape = AnnotationShape::fromJson(value.toObject());
        if (shape.points.isEmpty()) return;
        for (const QPointF& point : shape.points) {
            if (!hasPoint) {
                bounds = QRectF(point, QSizeF(0, 0));
                hasPoint = true;
            } else
                bounds |= QRectF(point, QSizeF(0, 0));
        }
        copied.append(shape);
    }
    if (copied.isEmpty()) return;
    if (bounds.width() > m_image.width() ||
        bounds.height() > m_image.height()) {
        m_status->setText(tr("Copied objects exceed image bounds"));
        return;
    }
    const double requested = 10.0 * (++m_pasteCount);
    const QPointF delta(std::clamp(requested, -bounds.left(),
                                   double(m_image.width()) - bounds.right()),
                        std::clamp(requested, -bounds.top(),
                                   double(m_image.height()) - bounds.bottom()));
    snapshot();
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    int nextGroup = 0;
    for (const AnnotationShape& shape : shapes) {
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        if (group.isDouble())
            nextGroup = std::max(nextGroup, group.toInt() + 1);
    }
    QHash<int, int> groups;
    for (AnnotationShape& shape : copied) {
        for (QPointF& point : shape.points) point += delta;
        const QJsonValue group = shape.extra.value(QStringLiteral("group_id"));
        if (group.isDouble()) {
            const int oldId = group.toInt();
            if (!groups.contains(oldId)) groups.insert(oldId, nextGroup++);
            shape.extra.insert(QStringLiteral("group_id"), groups.value(oldId));
        }
        shapes.append(shape);
    }
    m_canvas->setShapes(shapes);
    m_canvas->selectShape(shapes.size() - 1);
    updateObjects();
    setDirty(true);
    m_status->setText(tr("Pasted %1 objects").arg(copied.size()));
}

void AnnotationWindow::editSelectedLabel() {
    int index = m_canvas->selectedIndex();
    const QList<QListWidgetItem*> selected = m_objects->selectedItems();
    if (!selected.isEmpty()) index = m_objects->row(selected.first());
    if (index < 0 || index >= m_canvas->shapes().size()) return;

    const AnnotationShape current = m_canvas->shapes().at(index);
    bool accepted = false;
    QJsonObject chosenFlags;
    const QString label = openLabelDialog(
            current.label, &accepted, &chosenFlags,
            current.extra.value(QStringLiteral("flags")).toObject());
    if (!accepted || label.isEmpty()) return;
    if (label == current.label &&
        chosenFlags == current.extra.value(QStringLiteral("flags")).toObject())
        return;

    snapshot();
    QVector<AnnotationShape> shapes = m_canvas->shapes();
    if (label != current.label) shapes[index].label = label;
    if (!chosenFlags.isEmpty() ||
        shapes[index].extra.contains(QStringLiteral("flags")))
        shapes[index].extra.insert(QStringLiteral("flags"), chosenFlags);
    m_canvas->setShapes(shapes);
    m_canvas->selectShape(index);
    updateObjects();
    setDirty(true);
}

QVector<AnnotationShape> AnnotationWindow::remoteShapes(
        const QJsonArray& shapes, const QSize& imageSize) const {
    Q_UNUSED(imageSize);
    QVector<AnnotationShape> result;
    for (const QJsonValue& value : shapes) {
        const QJsonObject item = value.toObject();
        AnnotationShape shape;
        shape.label = item.value(QStringLiteral("label")).toString();
        shape.type = item.value(QStringLiteral("shape_type")).toString();
        if (shape.type.isEmpty()) shape.type = QStringLiteral("rectangle");
        const QJsonArray points =
                item.value(QStringLiteral("points")).toArray();
        for (const QJsonValue& point : points) {
            const QJsonArray xy = point.toArray();
            if (xy.size() >= 2)
                shape.points.append(
                        QPointF(xy.at(0).toDouble(), xy.at(1).toDouble()));
        }
        if (shape.points.isEmpty()) continue;
        const QJsonValue score = item.value(QStringLiteral("score"));
        if (score.isDouble())
            shape.extra.insert(QStringLiteral("score"), score);
        const QJsonValue difficult = item.value(QStringLiteral("difficult"));
        if (difficult.isBool())
            shape.extra.insert(QStringLiteral("difficult"), difficult);
        const QJsonValue description =
                item.value(QStringLiteral("description"));
        if (description.isString() && !description.toString().isEmpty())
            shape.extra.insert(QStringLiteral("description"), description);
        const QJsonValue groupId = item.value(QStringLiteral("group_id"));
        if (groupId.isDouble())
            shape.extra.insert(QStringLiteral("group_id"), groupId);
        const QJsonValue attributes = item.value(QStringLiteral("attributes"));
        if (attributes.isObject())
            shape.extra.insert(QStringLiteral("attributes"), attributes);
        result.append(shape);
    }
    return result;
}

void AnnotationWindow::startRemoteInference() {
    if (m_image.isNull() || m_remoteInferenceRunning || m_remoteVideoRunning)
        return;
    if (!m_remoteClient) {
        m_remoteClient = new RemoteServerClient(this);
        connect(m_remoteClient, &RemoteServerClient::predictFinished, this,
                [this](const QJsonArray& shapes, const QString& error) {
                    m_remoteInferenceRunning = false;
                    // Stale results (image or prompt changed mid-flight) are
                    // discarded without touching the canvas.
                    const bool stale =
                            m_generation != m_remoteGeneration ||
                            m_promptRevision != m_remotePromptRevision;
                    if (!error.isEmpty()) {
                        qany_log::log(
                                QStringLiteral("[Error] Remote inference: %1")
                                        .arg(error));
                        m_status->setText(error);
                        return;
                    }
                    if (stale) return;
                    AnnotationInference result;
                    result.generation = m_generation;
                    result.promptRevision = m_promptRevision;
                    result.imagePath = m_imagePath;
                    result.modelPath = QStringLiteral("remote-server");
                    result.task = QStringLiteral("remote-server");
                    result.shapes = remoteShapes(shapes, m_image.size());
                    onInference(result);
                });
        connectRemoteVideoSignals();
    }
    QSettings settings;
    RemoteServerClient::Settings remote;
    remote.serverUrl =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerUrl"),
                           QStringLiteral("http://127.0.0.1:8000"))
                    .toString();
    remote.apiKey =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerApiKey"))
                    .toString();
    remote.timeoutSec =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerTimeout"),
                           30)
                    .toInt();
    m_remoteClient->applySettings(remote);
    if (remote.serverUrl.isEmpty() || m_remoteModelId.isEmpty()) {
        openRemoteServerSettings();
        return;
    }
    m_remoteInferenceRunning = true;
    m_remoteGeneration = m_generation;
    m_remotePromptRevision = m_promptRevision;
    // The batch mode is a server-side contract declared in /v1/models
    // (upstream get_batch_processing_mode parity), so every run starts with
    // a cheap probe that routes between the plain predict and the
    // video-session flow.
    m_status->setText(
            tr("Resolving remote model (%1)...").arg(m_remoteModelId));
    m_remoteClient->fetchModels();
}

void AnnotationWindow::onRemoteModelsFetched(const QJsonObject& data,
                                             const QString& error) {
    m_remoteInferenceRunning = false;
    if (!error.isEmpty()) {
        qany_log::log(
                QStringLiteral("[Error] Remote model list: %1").arg(error));
        m_status->setText(error);
        return;
    }
    const QJsonObject model = data.value(m_remoteModelId).toObject();
    // Upstream falls back to the model-level batch_processing_mode; a
    // task-level mode is adopted only when the task list is unambiguous
    // (this plugin has no per-task selector).
    m_remoteBatchMode = model.value(QStringLiteral("batch_processing_mode"))
                                .toString(QStringLiteral("default"));
    const QJsonArray tasks =
            model.value(QStringLiteral("available_tasks")).toArray();
    if (m_remoteBatchMode == QLatin1String("default") && tasks.size() == 1) {
        const QJsonObject task = tasks.at(0).toObject();
        if (task.contains(QStringLiteral("batch_processing_mode")))
            m_remoteBatchMode =
                    task.value(QStringLiteral("batch_processing_mode"))
                            .toString();
    }
    if (m_remoteBatchMode == QLatin1String("video")) {
        startRemoteVideoFlow();
        return;
    }
    runRemotePredict();
}

void AnnotationWindow::runRemotePredict() {
    m_remoteInferenceRunning = true;
    m_remoteGeneration = m_generation;
    m_remotePromptRevision = m_promptRevision;
    QVariantMap params;
    params.insert(QStringLiteral("conf_threshold"), m_confidence->value());
    params.insert(QStringLiteral("epsilon_factor"), 0.001);
    const QString textPrompt = m_samText->text().trimmed();
    if (!textPrompt.isEmpty())
        params.insert(QStringLiteral("text_prompt"), textPrompt);
    m_status->setText(
            tr("Running remote inference (%1)...").arg(m_remoteModelId));
    m_remoteClient->predict(m_image, m_imagePath, m_remoteModelId, params);
}

void AnnotationWindow::connectRemoteVideoSignals() {
    connect(m_remoteClient, &RemoteServerClient::modelsFetched, this,
            &AnnotationWindow::onRemoteModelsFetched);

    // Session init finished: prompt the current frame with the text box or
    // the canvas points (converted to relative coords, upstream parity).
    connect(m_remoteClient, &RemoteServerClient::videoInitFinished, this,
            [this](const QString& sessionId, const QString& error) {
                Q_UNUSED(sessionId);
                m_remoteInferenceRunning = false;
                if (!error.isEmpty()) {
                    m_remoteVideoRunning = false;
                    qany_log::log(
                            QStringLiteral("[Error] Remote video init: %1")
                                    .arg(error));
                    m_status->setText(error);
                    return;
                }
                m_remoteInferenceRunning = true;
                m_remoteGeneration = m_generation;
                m_remotePromptRevision = m_promptRevision;
                QVariantMap params;
                params.insert(QStringLiteral("conf_threshold"),
                              m_confidence->value());
                params.insert(QStringLiteral("epsilon_factor"), 0.001);
                const QString textPrompt = m_samText->text().trimmed();
                if (!textPrompt.isEmpty()) {
                    m_status->setText(tr("Prompting frame %1 with text...")
                                              .arg(m_remotePromptFrame));
                    m_remoteClient->videoPromptText(textPrompt,
                                                    m_remotePromptFrame,
                                                    m_remoteModelId, params);
                    return;
                }
                const AnnotationPrompt prompt = m_canvas->prompt();
                if (!prompt.positive.isEmpty() || !prompt.negative.isEmpty()) {
                    const QSize size = m_image.size();
                    QVariantList points;
                    QVariantList labels;
                    for (const QPointF& point : prompt.positive) {
                        points.append(QVariantList{point.x() / size.width(),
                                                   point.y() / size.height()});
                        labels.append(1);
                    }
                    for (const QPointF& point : prompt.negative) {
                        points.append(QVariantList{point.x() / size.width(),
                                                   point.y() / size.height()});
                        labels.append(0);
                    }
                    m_status->setText(tr("Prompting frame %1 with points...")
                                              .arg(m_remotePromptFrame));
                    m_remoteClient->videoPromptPoints(points, labels,
                                                      m_remotePromptFrame,
                                                      m_remoteModelId, params);
                    return;
                }
                m_remoteVideoRunning = false;
                m_status->setText(
                        tr("Add a text or point prompt for the remote video "
                           "model"));
            });

    // Prompt finished: the prompt-frame shapes go through the regular
    // inference path, then propagation fills the remaining frames.
    connect(m_remoteClient, &RemoteServerClient::videoPromptFinished, this,
            [this](const QJsonArray& masks, const QString& error) {
                if (!error.isEmpty()) {
                    m_remoteVideoRunning = false;
                    m_remoteInferenceRunning = false;
                    qany_log::log(
                            QStringLiteral("[Error] Remote video prompt: %1")
                                    .arg(error));
                    m_status->setText(error);
                    return;
                }
                if (m_generation != m_remoteGeneration ||
                    m_promptRevision != m_remotePromptRevision) {
                    m_remoteVideoRunning = false;
                    m_remoteInferenceRunning = false;
                    m_status->setText(
                            tr("Inference inputs changed; run again"));
                    return;
                }
                m_remoteInferenceRunning = false;
                AnnotationInference result;
                result.generation = m_generation;
                result.promptRevision = m_promptRevision;
                result.imagePath = m_imagePath;
                result.modelPath = QStringLiteral("remote-server");
                result.task = QStringLiteral("remote-server");
                result.shapes = remoteShapes(masks, m_image.size());
                onInference(result);
                m_status->setText(tr("Propagating masks across frames..."));
                m_remoteClient->videoPropagate(m_remoteModelId, 0, -1);
            });

    connect(m_remoteClient, &RemoteServerClient::videoPropagateStarted, this,
            [this](int totalFrames, int startFrameIndex) {
                m_status->setText(tr("Propagating %1 frames from %2...")
                                          .arg(totalFrames)
                                          .arg(startFrameIndex));
            });
    connect(m_remoteClient, &RemoteServerClient::videoPropagateProgress, this,
            [this](int frame) {
                m_status->setText(tr("Propagating frame %1...").arg(frame + 1));
            });

    // Propagation finished: save every frame's masks into its sidecar
    // (upstream save_auto_labeling_result parity); the prompt frame is
    // already on the canvas, and the session is released either way.
    connect(m_remoteClient, &RemoteServerClient::videoPropagateFinished, this,
            [this](const QJsonObject& results, const QString& error) {
                m_remoteVideoRunning = false;
                m_remoteInferenceRunning = false;
                if (!error.isEmpty()) {
                    qany_log::log(
                            QStringLiteral("[Error] Remote propagation: %1")
                                    .arg(error));
                    m_status->setText(error);
                    return;
                }
                int saved = 0;
                for (auto it = results.constBegin(); it != results.constEnd();
                     ++it) {
                    bool ok = false;
                    const int frameIndex = it.key().toInt(&ok);
                    if (!ok || frameIndex < 0 ||
                        frameIndex >= m_remoteFramePaths.size() ||
                        frameIndex == m_remotePromptFrame)
                        continue;
                    const QJsonArray masks =
                            it.value()
                                    .toObject()
                                    .value(QStringLiteral("masks"))
                                    .toArray();
                    if (masks.isEmpty()) continue;
                    saveRemoteShapesToSidecar(m_remoteFramePaths.at(frameIndex),
                                              remoteShapes(masks, QSize()));
                    ++saved;
                }
                m_status->setText(
                        tr("Remote propagation saved %1 frames").arg(saved));
            });
}

void AnnotationWindow::startRemoteVideoFlow() {
    if (m_remoteVideoRunning) return;
    // Frame set: the file list mirrors the upstream image_list; a lone
    // image degrades to a single-frame session.
    QStringList frames;
    if (m_files) {
        for (int i = 0; i < m_files->count(); ++i) {
            const QString path =
                    m_files->item(i)->data(Qt::UserRole).toString();
            if (!path.isEmpty()) frames.append(path);
        }
    }
    if (frames.isEmpty()) frames.append(m_imagePath);
    if (!frames.contains(m_imagePath)) frames.append(m_imagePath);
    // Safety bound: /v1/video/init posts every frame's file bytes at once
    // (upstream parity), so cap the request before the base64 payload can
    // exhaust the process on very large folders.
    if (frames.size() > 2000) {
        m_remoteVideoRunning = false;
        m_status->setText(tr("Remote video sessions support at most 2000 "
                             "frames (got %1)")
                                  .arg(frames.size()));
        return;
    }
    m_remotePromptFrame = frames.indexOf(m_imagePath);
    m_remoteFramePaths = frames;
    m_remoteVideoRunning = true;
    m_status->setText(tr("Uploading %1 frames to the remote session...")
                              .arg(frames.size()));
    const QStringList uris = m_remoteClient->frameDataUris(frames);
    m_remoteClient->videoInit(uris, m_remoteModelId);
}

void AnnotationWindow::saveRemoteShapesToSidecar(
        const QString& imagePath, const QVector<AnnotationShape>& shapes) {
    const QString sidecar = sidecarPathFor(imagePath);
    QJsonObject document;
    QFile input(sidecar);
    if (input.exists() && input.open(QIODevice::ReadOnly)) {
        const QJsonDocument parsed = QJsonDocument::fromJson(input.readAll());
        if (parsed.isObject()) document = parsed.object();
    }
    int imageHeight = document.value(QStringLiteral("imageHeight")).toInt();
    int imageWidth = document.value(QStringLiteral("imageWidth")).toInt();
    if (imageHeight <= 0 || imageWidth <= 0) {
        // Header fields only — QImageReader avoids a full decode.
        QImageReader reader(imagePath);
        const QSize size = reader.size();
        imageHeight = size.height();
        imageWidth = size.width();
    }
    QJsonArray shapeArray;
    for (const AnnotationShape& shape : shapes)
        shapeArray.append(shape.toJson());
    document.insert(QStringLiteral("version"), QStringLiteral("4.0.6"));
    if (!document.contains(QStringLiteral("flags")))
        document.insert(QStringLiteral("flags"), QJsonObject());
    document.insert(QStringLiteral("imageData"), QJsonValue::Null);
    document.insert(QStringLiteral("imagePath"),
                    QFileInfo(imagePath).fileName());
    document.insert(QStringLiteral("imageHeight"), imageHeight);
    document.insert(QStringLiteral("imageWidth"), imageWidth);
    document.insert(QStringLiteral("shapes"), shapeArray);
    QSaveFile output(sidecar);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(QJsonDocument(document).toJson(QJsonDocument::Indented)) <
                0 ||
        !output.commit()) {
        qany_log::log(
                QStringLiteral("[Error] Unable to save remote frame %1: %2")
                        .arg(sidecar, output.errorString()));
    }
}

void AnnotationWindow::openRemoteServerSettings() {
    if (!m_remoteDialog) {
        m_remoteDialog = new RemoteServerDialog(this);
        connect(m_remoteDialog, &QDialog::accepted, this, [this]() {
            const RemoteServerClient::Settings remote =
                    m_remoteDialog->settings();
            QSettings settings;
            settings.setValue(QStringLiteral("qAnyLabeling/remoteServerUrl"),
                              remote.serverUrl);
            settings.setValue(QStringLiteral("qAnyLabeling/remoteServerApiKey"),
                              remote.apiKey);
            settings.setValue(
                    QStringLiteral("qAnyLabeling/remoteServerTimeout"),
                    remote.timeoutSec);
            m_remoteModelId = m_remoteDialog->selectedModelId();
            settings.setValue(QStringLiteral("qAnyLabeling/remoteServerModel"),
                              m_remoteModelId);
        });
    }
    QSettings settings;
    RemoteServerClient::Settings remote;
    remote.serverUrl =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerUrl"),
                           QStringLiteral("http://127.0.0.1:8000"))
                    .toString();
    remote.apiKey =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerApiKey"))
                    .toString();
    remote.timeoutSec =
            settings.value(QStringLiteral("qAnyLabeling/remoteServerTimeout"),
                           30)
                    .toInt();
    m_remoteDialog->setSettings(remote);
    m_remoteDialog->setSelectedModelId(m_remoteModelId);
    m_remoteDialog->exec();
}

bool AnnotationWindow::validateLabel(const QString& label) const {
    // Upstream validate_label semantics: null (empty here) accepts every
    // label; "exact" requires an exact match against the unique label
    // list. Wired into the batch-edit dialog (editLabels) apply path and
    // available to future label-entry sites.
    if (m_validateLabel != QStringLiteral("exact")) return true;
    if (!m_labels) return true;
    for (int i = 0; i < m_labels->count(); ++i)
        if (m_labels->item(i)->data(Qt::UserRole).toString() == label)
            return true;
    return false;
}

void AnnotationWindow::runDetection() {
    if (m_image.isNull() || m_inferenceRunning || m_downloader->isBusy())
        return;
    const QString filename = m_quant->currentData().toString();
    const auto found = std::find_if(m_models.cbegin(), m_models.cend(),
                                    [&filename](const Model& model) {
                                        return model.filename == filename;
                                    });
    if (found == m_models.cend()) return;
    if (found->task == QStringLiteral("remote-server")) {
        // No local weights: the configured service is the backend, so skip
        // the catalog download path and dispatch straight to the HTTP client.
        startInference(QStringLiteral("remote-server"));
        return;
    }
    const bool sam = found->task == QStringLiteral("sam3-pvs");
    const bool depthModel = found->task.startsWith(QStringLiteral("depth"));
    const bool gkd = found->task == QStringLiteral("gkd");
    const bool face = found->task == QStringLiteral("facedetect");
    const bool deeplsd = found->task == QStringLiteral("deeplsd");
    const bool aliked = found->task == QStringLiteral("aliked");
    const bool textPrompt = sam && found->supportsText &&
                            !m_samText->text().trimmed().isEmpty();
    const bool samSelectedBoxes = sam && !m_batchRunning && !textPrompt &&
                                  m_canvas->prompt().positive.isEmpty() &&
                                  !m_canvas->prompt().useBox &&
                                  !m_canvas->selectedIndices().isEmpty();
    if (sam && !textPrompt && !samSelectedBoxes &&
        m_canvas->prompt().positive.isEmpty() && !m_canvas->prompt().useBox) {
        m_status->setText(
                tr("Add a SAM point or box, or select detection boxes"));
        return;
    }
    if (gkd && m_samText->text().trimmed().isEmpty() &&
        m_gkdSupportPoints.isEmpty()) {
        m_status->setText(
                tr("Enter comma-separated keypoint names or select visual "
                   "support points"));
        return;
    }
    const bool openVocabulary = found->task == QStringLiteral("world") ||
                                found->task == QStringLiteral("yoloe");
    const bool visualPrompt = !m_batchRunning &&
                              found->task == QStringLiteral("yoloe") &&
                              !m_canvas->selectedIndices().isEmpty();
    if (openVocabulary && found->supportsText && !visualPrompt &&
        m_samText->text().trimmed().isEmpty()) {
        m_status->setText(tr(
                "Enter comma-separated class names or select a visual prompt"));
        return;
    }
    const bool rfdetr = found->task.startsWith(QStringLiteral("rfdetr-"));
    const bool rmbg = found->task == QStringLiteral("rmbg");
    const bool ocr = found->task == QStringLiteral("ocr");
    char* cache = depthModel ? aicore_depth_model_cache_dir()
                  : deeplsd  ? aicore_deeplsd_model_cache_dir()
                  : aliked   ? aicore_aliked_model_cache_dir()
                  : sam      ? aicore_sam3_model_cache_dir()
                  : rfdetr   ? aicore_rfdetr_model_cache_dir()
                  : rmbg     ? aicore_rmbg_model_cache_dir()
                  : gkd      ? aicore_gkd_model_cache_dir()
                  : face     ? aicore_facedetect_model_cache_dir()
                  : ocr      ? aicore_ocr_model_cache_dir()
                             : aicore_yolo_model_cache_dir();
    if (!cache) {
        m_status->setText(tr("Model cache is unavailable"));
        return;
    }
    const QString path = QDir(QString::fromUtf8(cache)).filePath(filename);
    if (depthModel)
        aicore_depth_free_buffer(cache);
    else if (deeplsd)
        aicore_deeplsd_free_buffer(cache);
    else if (aliked)
        aicore_aliked_free_buffer(cache);
    else if (sam)
        aicore_sam3_free_buffer(cache);
    else if (rfdetr)
        aicore_rfdetr_free_buffer(cache);
    else if (rmbg)
        aicore_rmbg_free_buffer(cache);
    else if (gkd)
        aicore_gkd_free_buffer(cache);
    else if (face)
        aicore_facedetect_free_buffer(cache);
    else if (ocr)
        aicore_ocr_free_buffer(cache);
    else
        aicore_yolo_free_buffer(cache);
    if (found->task == QStringLiteral("depth-nested")) {
        const char* metricName = "depth-anything-nested-metric.gguf";
        const aicore_model_entry* metric =
                aicore_model_by_filename(AICORE_MODEL_FAMILY_DEPTH, metricName);
        if (!metric || !metric->download_url) {
            m_status->setText(tr("Nested metric model unavailable"));
            return;
        }
        const QString metricPath =
                QDir(QFileInfo(path).absolutePath())
                        .filePath(QString::fromLatin1(metricName));
        const ecvAssetIntegrity::Anchor metricAnchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(
                        QString::fromLatin1(metricName))};
        if (!ecvAssetIntegrity::isVerified(metricPath, metricAnchor,
                                           1024 * 1024, true)) {
            QDir().mkpath(QFileInfo(metricPath).absolutePath());
            ecvModelDownloader::Request request;
            request.url = QString::fromUtf8(metric->download_url);
            request.destPath = metricPath;
            request.minBytes = 1024 * 1024;
            request.contentAnchor = metricAnchor;
            m_runAfterDownload = true;
            m_downloadDependencyModel = true;
            m_downloadGeneration = m_generation;
            m_downloadImagePath = m_imagePath;
            m_downloadFilename = filename;
            m_status->setText(
                    tr("Downloading %1").arg(QString::fromLatin1(metricName)));
            m_downloader->download(request);
            return;
        }
    }
    if (face && !m_faceLandmark->currentData().toString().isEmpty()) {
        const QString landmarkFilename =
                m_faceLandmark->currentData().toString();
        const QByteArray nameBytes = landmarkFilename.toUtf8();
        const aicore_facedetect_model_entry* landmark =
                aicore_facedetect_model_by_filename(nameBytes.constData());
        if (!landmark || !landmark->download_url) {
            m_status->setText(tr("Face landmark model unavailable"));
            return;
        }
        const QString landmarkPath =
                QDir(QFileInfo(path).absolutePath()).filePath(landmarkFilename);
        const ecvAssetIntegrity::Anchor landmarkAnchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(landmarkFilename)};
        if (!ecvAssetIntegrity::isVerified(landmarkPath, landmarkAnchor,
                                           1024 * 1024, true)) {
            QDir().mkpath(QFileInfo(landmarkPath).absolutePath());
            ecvModelDownloader::Request request;
            request.url = QString::fromUtf8(landmark->download_url);
            request.destPath = landmarkPath;
            request.minBytes = 1024 * 1024;
            request.contentAnchor = landmarkAnchor;
            m_runAfterDownload = true;
            m_downloadDependencyModel = true;
            m_downloadGeneration = m_generation;
            m_downloadImagePath = m_imagePath;
            m_downloadFilename = filename;
            m_status->setText(tr("Downloading %1").arg(landmarkFilename));
            m_downloader->download(request);
            return;
        }
    }
    if (ocr) {
        // Det partner: same quantization tier, verified/downloaded into the
        // same cache directory before the rec model runs.
        const QString detFilename = ocrDetPartner(filename);
        if (detFilename.isEmpty()) {
            m_status->setText(tr("OCR det model unavailable"));
            return;
        }
        const QString detPath =
                QDir(QFileInfo(path).absolutePath()).filePath(detFilename);
        const ecvAssetIntegrity::Anchor detAnchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(detFilename)};
        if (!ecvAssetIntegrity::isVerified(detPath, detAnchor, 1024 * 1024,
                                           true)) {
            const aicore_ocr_model_entry* det = aicore_ocr_model_by_filename(
                    detFilename.toUtf8().constData());
            if (!det || !det->download_url) {
                m_status->setText(tr("OCR det model unavailable"));
                return;
            }
            QDir().mkpath(QFileInfo(detPath).absolutePath());
            ecvModelDownloader::Request request;
            request.url = QString::fromUtf8(det->download_url);
            request.destPath = detPath;
            request.minBytes = 1024 * 1024;
            request.contentAnchor = detAnchor;
            m_runAfterDownload = true;
            m_downloadDependencyModel = true;
            m_downloadGeneration = m_generation;
            m_downloadImagePath = m_imagePath;
            m_downloadFilename = filename;
            m_status->setText(tr("Downloading %1").arg(detFilename));
            m_downloader->download(request);
            return;
        }
    }
    if (openVocabulary && found->supportsText && !visualPrompt) {
        const QString textFilename = m_textModel->currentData().toString();
        const auto encoder =
                std::find_if(m_textModels.cbegin(), m_textModels.cend(),
                             [&textFilename](const Model& item) {
                                 return item.filename == textFilename;
                             });
        if (encoder == m_textModels.cend()) {
            m_status->setText(tr("Text encoder unavailable"));
            return;
        }
        const QString textPath =
                QDir(QFileInfo(path).absolutePath()).filePath(textFilename);
        const ecvAssetIntegrity::Anchor textAnchor{
                QCryptographicHash::Sha256,
                ecvAssetIntegrity::PinnedDigest(textFilename)};
        if (!ecvAssetIntegrity::isVerified(textPath, textAnchor, 1024 * 1024,
                                           true)) {
            if (encoder->url.isEmpty()) {
                m_status->setText(tr("Text encoder download URL unavailable"));
                return;
            }
            ecvModelDownloader::Request request;
            request.url = encoder->url;
            request.destPath = textPath;
            request.minBytes = 1024 * 1024;
            request.contentAnchor = textAnchor;
            m_runAfterDownload = true;
            m_downloadDependencyModel = true;
            m_downloadGeneration = m_generation;
            m_downloadImagePath = m_imagePath;
            m_downloadFilename = filename;
            m_status->setText(tr("Downloading %1").arg(textFilename));
            m_downloader->download(request);
            return;
        }
    }
    const ecvAssetIntegrity::Anchor anchor{
            QCryptographicHash::Sha256,
            ecvAssetIntegrity::PinnedDigest(filename)};
    if (ecvAssetIntegrity::isVerified(path, anchor, 1024 * 1024, true)) {
        startInference(path);
        return;
    }
    if (found->url.isEmpty()) {
        m_status->setText(tr("Model download URL is unavailable"));
        return;
    }
    QDir().mkpath(QFileInfo(path).absolutePath());
    ecvModelDownloader::Request request;
    request.url = found->url;
    request.destPath = path;
    request.minBytes = 1024 * 1024;
    request.contentAnchor = anchor;
    m_runAfterDownload = true;
    m_downloadDependencyModel = false;
    m_downloadGeneration = m_generation;
    m_downloadImagePath = m_imagePath;
    m_downloadFilename = filename;
    m_status->setText(tr("Downloading %1").arg(filename));
    m_downloader->download(request);
}

void AnnotationWindow::startInference(const QString& modelPath) {
    const QString filename = QFileInfo(modelPath).fileName();
    const auto model = std::find_if(m_models.cbegin(), m_models.cend(),
                                    [&filename](const Model& item) {
                                        return item.filename == filename;
                                    });
    if (model == m_models.cend()) return;
    if (model->task == QStringLiteral("remote-server")) {
        startRemoteInference();
        return;
    }
    const bool textPrompt = model->task == QStringLiteral("sam3-pvs") &&
                            model->supportsText &&
                            !m_samText->text().trimmed().isEmpty();
    const bool samSelectedBoxes = model->task == QStringLiteral("sam3-pvs") &&
                                  !m_batchRunning && !textPrompt &&
                                  m_canvas->prompt().positive.isEmpty() &&
                                  !m_canvas->prompt().useBox &&
                                  !m_canvas->selectedIndices().isEmpty();
    if (model->task == QStringLiteral("sam3-pvs") && !textPrompt &&
        !samSelectedBoxes && m_canvas->prompt().positive.isEmpty() &&
        !m_canvas->prompt().useBox) {
        m_status->setText(
                tr("Add a SAM point or box, or select detection boxes"));
        return;
    }
    if (model->task == QStringLiteral("gkd") &&
        m_samText->text().trimmed().isEmpty() && m_gkdSupportPoints.isEmpty()) {
        m_status->setText(
                tr("Enter comma-separated keypoint names or select visual "
                   "support points"));
        return;
    }
    const bool openVocabulary = model->task == QStringLiteral("world") ||
                                model->task == QStringLiteral("yoloe");
    const bool visualPrompt = !m_batchRunning &&
                              model->task == QStringLiteral("yoloe") &&
                              !m_canvas->selectedIndices().isEmpty();
    if (openVocabulary && model->supportsText && !visualPrompt &&
        m_samText->text().trimmed().isEmpty()) {
        m_status->setText(tr(
                "Enter comma-separated class names or select a visual prompt"));
        return;
    }
    AnnotationPrompt prompt = m_canvas->prompt();
    if (m_ocrSkipRequested && model->task != QStringLiteral("ocr")) {
        m_ocrSkipRequested = false;
        m_status->setText(tr("Skip detection applies to the OCR family"));
        return;
    }
    if (m_ocrSkipRequested) {
        QVector<int> selected = m_canvas->selectedIndices().values().toVector();
        std::sort(selected.begin(), selected.end());
        for (int index : selected) {
            if (index < 0 || index >= m_canvas->shapes().size()) continue;
            const AnnotationShape& shape = m_canvas->shapes().at(index);
            if (!shape.visible || shape.label != QStringLiteral("text") ||
                shape.points.size() < 2)
                continue;
            const QRectF bounds = QPolygonF(shape.points).boundingRect();
            if (bounds.width() < 1.0 || bounds.height() < 1.0) continue;
            prompt.rois.append(bounds);
            prompt.roiSourceIndices.append(index);
        }
        if (prompt.rois.isEmpty()) {
            m_ocrSkipRequested = false;
            m_status->setText(tr("Select at least one visible text region"));
            return;
        }
    }
    if (samSelectedBoxes) {
        QVector<int> selected = m_canvas->selectedIndices().values().toVector();
        std::sort(selected.begin(), selected.end());
        for (int index : selected) {
            if (index < 0 || index >= m_canvas->shapes().size()) continue;
            const AnnotationShape& shape = m_canvas->shapes().at(index);
            if (!shape.visible || shape.points.size() < 2 ||
                (shape.type != QStringLiteral("rectangle") &&
                 shape.type != QStringLiteral("rotation")))
                continue;
            const QRectF bounds = QPolygonF(shape.points).boundingRect();
            if (bounds.width() < 1.0 || bounds.height() < 1.0) continue;
            prompt.rois.append(bounds);
            prompt.roiSourceIndices.append(index);
            prompt.roiLabels.append(shape.label);
        }
        if (prompt.rois.isEmpty()) {
            m_status->setText(tr("Select at least one visible detection box"));
            return;
        }
    }
    if (visualPrompt) {
        prompt.text.clear();
        prompt.textModelPath.clear();
    }
    if (textPrompt || model->task == QStringLiteral("gkd") ||
        (openVocabulary && model->supportsText && !visualPrompt))
        prompt.text = m_samText->text().trimmed();
    if (model->task == QStringLiteral("gkd")) {
        prompt.visualSupportImage = m_gkdSupportImage;
        prompt.visualSupportPoints = m_gkdSupportPoints;
    }
    if (model->task == QStringLiteral("detect") &&
        m_slicedDetection->isChecked()) {
        prompt.slicedDetection = true;
        prompt.sliceWidth = m_sliceWidth->value();
        prompt.sliceHeight = m_sliceHeight->value();
        prompt.sliceOverlap = m_sliceOverlap->value();
        prompt.sliceMergeThreshold = m_sliceMergeThreshold->value();
    }
    if (visualPrompt) {
        QVector<int> selected = m_canvas->selectedIndices().values().toVector();
        std::sort(selected.begin(), selected.end());
        for (int index : selected) {
            if (index < 0 || index >= m_canvas->shapes().size()) continue;
            const AnnotationShape& shape = m_canvas->shapes().at(index);
            if (!shape.visible || shape.points.size() < 2 ||
                (shape.type != QStringLiteral("rectangle") &&
                 shape.type != QStringLiteral("rotation")))
                continue;
            const QRectF bounds = QPolygonF(shape.points).boundingRect();
            if (bounds.width() < 1.0 || bounds.height() < 1.0) continue;
            prompt.visualPrompts.append(bounds);
        }
        if (prompt.visualPrompts.isEmpty()) {
            m_status->setText(
                    tr("Select at least one visible rectangle for a YOLOE "
                       "visual prompt"));
            return;
        }
    }
    if (model->task == QStringLiteral("gkd") && !m_batchRunning &&
        !prompt.useBox) {
        // Selected rectangles are GKD's top-down proposals. Keep their order
        // deterministic because QSet iteration order is unspecified, and
        // carry an existing group ID when the user already grouped the ROI.
        QVector<int> selected = m_canvas->selectedIndices().values().toVector();
        std::sort(selected.begin(), selected.end());
        int nextGroup = 0;
        for (const AnnotationShape& shape : m_canvas->shapes()) {
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (group.isDouble())
                nextGroup = std::max(nextGroup, group.toInt() + 1);
        }
        for (int index : selected) {
            if (index < 0 || index >= m_canvas->shapes().size()) continue;
            const AnnotationShape& shape = m_canvas->shapes().at(index);
            if (!shape.visible || shape.points.size() < 2 ||
                (shape.type != QStringLiteral("rectangle") &&
                 shape.type != QStringLiteral("rotation")))
                continue;
            const QRectF bounds = QPolygonF(shape.points).boundingRect();
            if (bounds.width() < 1.0 || bounds.height() < 1.0) continue;
            prompt.rois.append(bounds);
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            prompt.roiGroupIds.append(group.isDouble() ? group.toInt()
                                                       : nextGroup++);
            prompt.roiSourceIndices.append(index);
        }
    }
    if (openVocabulary && model->supportsText && !visualPrompt) {
        const QString filename = m_textModel->currentData().toString();
        if (filename.isEmpty()) return;
        prompt.textModelPath =
                QDir(QFileInfo(modelPath).absolutePath()).filePath(filename);
    }
    const QString task =
            textPrompt ? QStringLiteral("sam3-pcs")
            : model->task == QStringLiteral("facedetect") &&
                            !m_faceLandmark->currentData().toString().isEmpty()
                    ? QStringLiteral("facedetect-dense")
                    : model->task;
    QString metricModelPath;
    if (task == QStringLiteral("depth-nested")) {
        metricModelPath = QDir(QFileInfo(modelPath).absolutePath())
                                  .filePath(QStringLiteral(
                                          "depth-anything-nested-metric.gguf"));
    }
    if (task == QStringLiteral("facedetect-dense")) {
        metricModelPath =
                QDir(QFileInfo(modelPath).absolutePath())
                        .filePath(m_faceLandmark->currentData().toString());
    }
    if (task == QStringLiteral("ocr")) {
        const QString detFilename =
                ocrDetPartner(QFileInfo(modelPath).fileName());
        if (detFilename.isEmpty()) {
            m_status->setText(tr("OCR det model unavailable"));
            return;
        }
        metricModelPath =
                QDir(QFileInfo(modelPath).absolutePath()).filePath(detFilename);
    }
    const QString device = m_backend->currentData().toString();
    m_inferenceRunning = true;
    m_status->setText(tr("Running %1...").arg(task));
    emit inferenceRequested(modelPath, metricModelPath, device, task,
                            m_imagePath, m_image, m_generation,
                            m_confidence->value(), m_promptRevision, prompt);
}

void AnnotationWindow::onInference(const AnnotationInference& result) {
    m_inferenceRunning = false;
    if (m_batchCancelRequested) {
        finishBatch(tr("Auto Run cancelled (%1 images completed)")
                            .arg(m_batchIndex));
        return;
    }
    if (result.generation != m_generation || result.imagePath != m_imagePath ||
        QFileInfo(result.modelPath).fileName() !=
                m_quant->currentData().toString()) {
        const QString message = tr("Inference inputs changed; run again");
        if (m_batchRunning)
            finishBatch(message);
        else
            m_status->setText(message);
        return;
    }
    if ((result.task.startsWith(QStringLiteral("sam3-")) ||
         result.task == QStringLiteral("gkd") ||
         result.task == QStringLiteral("world") ||
         result.task == QStringLiteral("yoloe")) &&
        result.promptRevision != m_promptRevision) {
        const QString message = tr("Inference prompt changed; run again");
        if (m_batchRunning)
            finishBatch(message);
        else
            m_status->setText(message);
        return;
    }
    if (!result.sourceIndices.isEmpty() &&
        result.sourceIndices.size() != result.sourceRois.size()) {
        m_status->setText(tr("Box proposal changed; run again"));
        return;
    }
    if (!result.sourceIndices.isEmpty()) {
        for (int roi = 0; roi < result.sourceIndices.size(); ++roi) {
            const int index = result.sourceIndices.at(roi);
            if (index < 0 || index >= m_canvas->shapes().size()) {
                m_status->setText(tr("Box proposal changed; run again"));
                return;
            }
            const QRectF current =
                    QPolygonF(m_canvas->shapes().at(index).points)
                            .boundingRect();
            const QRectF source = result.sourceRois.at(roi);
            if (qAbs(current.x() - source.x()) > 0.01 ||
                qAbs(current.y() - source.y()) > 0.01 ||
                qAbs(current.width() - source.width()) > 0.01 ||
                qAbs(current.height() - source.height()) > 0.01) {
                m_status->setText(tr("Box proposal changed; run again"));
                return;
            }
        }
    }
    if (!result.error.isEmpty()) {
        if (m_batchRunning)
            finishBatch(result.error);
        else
            m_status->setText(result.error);
        return;
    }
    // Skip-detection OCR: update the selected text shapes in place instead
    // of appending new ones (upstream replace=true semantics on the same
    // roi source indices).
    if (result.task == QStringLiteral("ocr") &&
        !result.sourceIndices.isEmpty() &&
        result.sourceIndices.size() == result.shapes.size()) {
        for (int i = 0; i < result.sourceIndices.size(); ++i) {
            const int index = result.sourceIndices.at(i);
            if (index < 0 || index >= m_canvas->shapes().size()) continue;
            const AnnotationShape& updated = result.shapes.at(i);
            m_canvas->setShapeRecognition(
                    index,
                    updated.extra.value(QStringLiteral("description"))
                            .toString(),
                    updated.extra.value(QStringLiteral("score")).toDouble(),
                    updated.extra.value(QStringLiteral("language")).toString());
        }
        m_status->setText(result.summary);
        return;
    }
    if (!result.depthPreview.isNull()) {
        const QFileInfo input(result.imagePath);
        const QString directory = QDir(input.absolutePath())
                                          .absoluteFilePath(QStringLiteral(
                                                  "../x-anylabeling-depth"));
        if (!QDir().mkpath(directory)) {
            const QString message = tr("Unable to create depth output folder");
            if (m_batchRunning)
                finishBatch(message);
            else
                m_status->setText(message);
            return;
        }
        const QString output = QDir(directory).filePath(
                input.completeBaseName() + QStringLiteral(".png"));
        QSaveFile file(output);
        if (!file.open(QIODevice::WriteOnly) ||
            !result.depthPreview.save(&file, "PNG") || !file.commit()) {
            const QString message =
                    tr("Unable to save depth image: %1").arg(output);
            if (m_batchRunning)
                finishBatch(message);
            else
                m_status->setText(message);
            return;
        }
    }
    if (m_canvas->brushEditing()) m_canvas->setBrushEditing(false);
    if (m_batchRunning) {
        // Auto Run replaces each sidecar's generated annotations. This keeps
        // reruns deterministic and matches the source batch workflow.
        snapshot();
        m_canvas->setShapes(result.shapes);
        updateObjects();
        setDirty(true);
    } else if (!result.shapes.isEmpty()) {
        snapshot();
        QVector<AnnotationShape> shapes = m_canvas->shapes();
        int nextGroup = 0;
        for (const AnnotationShape& shape : shapes) {
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (group.isDouble())
                nextGroup = std::max(nextGroup, group.toInt() + 1);
        }
        QVector<int> roiGroups;
        for (int index : result.sourceIndices) {
            const QJsonValue current =
                    shapes[index].extra.value(QStringLiteral("group_id"));
            const int groupId =
                    current.isDouble() ? current.toInt() : nextGroup++;
            shapes[index].extra.insert(QStringLiteral("group_id"), groupId);
            roiGroups.append(groupId);
        }
        QHash<int, int> groups;
        for (AnnotationShape shape : result.shapes) {
            const QJsonValue source =
                    shape.extra.take(QStringLiteral("source_roi"));
            if (source.isDouble() && source.toInt(-1) >= 0 &&
                source.toInt() < roiGroups.size()) {
                shape.extra.insert(QStringLiteral("group_id"),
                                   roiGroups.at(source.toInt()));
                shapes.append(shape);
                continue;
            }
            const QJsonValue group =
                    shape.extra.value(QStringLiteral("group_id"));
            if (group.isDouble()) {
                const int oldId = group.toInt();
                if (!groups.contains(oldId)) groups.insert(oldId, nextGroup++);
                shape.extra.insert(QStringLiteral("group_id"),
                                   groups.value(oldId));
            }
            shapes.append(shape);
        }
        m_canvas->setShapes(shapes);
        updateObjects();
        setDirty(true);
    }
    if (result.hasFlags) {
        QJsonObject flags =
                m_document.value(QStringLiteral("flags")).toObject();
        for (auto it = result.flags.constBegin(); it != result.flags.constEnd();
             ++it)
            flags.insert(it.key(), it.value());
        m_document.insert(QStringLiteral("flags"), flags);
        updateFlags();
        setDirty(true);
    }
    if (result.task == QStringLiteral("sam3-pvs") && !result.shapes.isEmpty())
        m_canvas->clearPrompt();
    m_status->setText(
            result.summary.isEmpty()
                    ? tr("%1 annotations on %2")
                              .arg(result.shapes.size())
                              .arg(result.device)
                    : tr("%1 on %2").arg(result.summary, result.device));
    if (m_batchRunning) {
        if (!saveDocument()) {
            finishBatch(tr("Auto Run stopped: unable to save annotations"));
            return;
        }
        ++m_batchIndex;
        m_batchProgress->setValue(m_batchIndex);
        if (!m_batchRunning) return;
        m_batchProgress->setLabelText(tr("Progress: %1/%2")
                                              .arg(m_batchIndex)
                                              .arg(m_batchFiles.size()));
        QTimer::singleShot(0, this, &AnnotationWindow::startBatchNext);
    }
}

void AnnotationWindow::closeEvent(QCloseEvent* event) {
    if (confirmSave())
        event->accept();
    else
        event->ignore();
}
