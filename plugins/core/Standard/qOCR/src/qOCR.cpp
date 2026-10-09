// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qOCR.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QMainWindow>
#include <QPainter>
#include <QPolygonF>

#include "OCRWorker.h"
#include "QtCompat.h"
#include "ecvGenericGLDisplay.h"
#include "ecvImage.h"
#include "ecvMainAppInterface.h"
#include "ecvPersistentSettings.h"
#include "ecvPluginDbNaming.h"

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#endif

qOCR::qOCR(QObject* parent)
    : QObject(parent), ccStdPluginInterface(":/CC/plugin/qOCR/info.json") {
    ecvPS::registerSettingsGroup(QStringLiteral("qOCR"));
    qRegisterMetaType<OCRRunSettings>("OCRRunSettings");
    qRegisterMetaType<OCRRunResult>("OCRRunResult");
    qRegisterMetaType<OCRBoxResult>("OCRBoxResult");
    m_workspaceAction = new QAction(QIcon(":/CC/plugin/qOCR/images/qOCR.svg"),
                                    tr("OCR Workspace"), this);
    m_workspaceAction->setObjectName(QStringLiteral("qOCRWorkspaceAction"));
    m_workspaceAction->setToolTip(
            tr("PP-OCR text recognition: stage models, pick a demo image, "
               "run det+rec, export the annotated result to the DB tree"));
    connect(m_workspaceAction, &QAction::triggered, this, &qOCR::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(tr("[qOCR] Task is running (%1 s elapsed)...")
                                    .arg(m_inferenceElapsedSeconds));
    });
}

QList<QAction*> qOCR::getActions() { return {m_workspaceAction}; }

void qOCR::setMainAppInterface(ecvMainAppInterface* app) {
    ccStdPluginInterface::setMainAppInterface(app);
}

void qOCR::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new OCRDialog(static_cast<QWidget*>(m_app->getMainWindow()));
        connect(m_dialog, &OCRDialog::runRequested, this, &qOCR::executeTask);
        connect(m_dialog, &OCRDialog::cancelRequested, this, &qOCR::cancelTask);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qOCR::executeTask(const OCRRunSettings& settings) {
    if (m_worker && m_worker->isRunning()) return;

    // A fresh worker per run (the shared AICore plugin lifecycle): the old
    // worker's context is released on the main thread before replacement.
    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    if (!QFileInfo::exists(settings.detPath) ||
        !QFileInfo::exists(settings.recPath)) {
        m_dialog->appendLog(tr("[Error] OCR models missing on disk."));
        return;
    }

    QString workerDevice = settings.device;
#ifdef AICore_ENABLED
    if (workerDevice != QStringLiteral("cpu") &&
        aicore_warmup_backend(workerDevice.toUtf8().constData()) != 0) {
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(
                    tr("[qOCR] GPU backend unavailable — using CPU for this "
                       "run."));
        }
    }
#endif

    m_currentSettings = settings;
    m_currentSettings.device = workerDevice;

    OCRWorker::Settings ws;
    ws.detPath = settings.detPath;
    ws.recPath = settings.recPath;
    ws.layoutPath = settings.layoutPath;
    ws.imagePath = settings.imagePath;
    ws.device = workerDevice;
    ws.threads = settings.threads;

    m_worker = new OCRWorker(ws, this);
    connect(m_worker, &OCRWorker::logMessage, m_dialog, &OCRDialog::appendLog,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::taskStage, m_dialog, &OCRDialog::setTaskStage,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::resultReady, this, &qOCR::onResultReady,
            Qt::QueuedConnection);
    connect(m_worker, &OCRWorker::taskFinished, this, &qOCR::onTaskFinished,
            Qt::QueuedConnection);
    m_dialog->setRunning(true);
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qOCR::cancelTask() {
    if (m_worker && m_worker->isRunning()) m_worker->requestTaskCancel();
}

void qOCR::onResultReady(const OCRRunResult& result) {
    m_dialog->showResult(result);
    if (m_currentSettings.addToDb || m_currentSettings.showInWindow) {
        addResultToDb(result, m_currentSettings);
    } else {
        m_dialog->appendLog(
                tr("[qOCR] Result display only (DB export and "
                   "render-window overlay are off)."));
    }
}

namespace {
// Official PaddleOCR draw_ocr visualization constants (tools/infer/utility.py):
// red 2-px closed polygon outlines (cv2 polylines color (255,0,0) on an RGB
// array), boxes below drop_score 0.5 are skipped, the annotated image is
// resized to a 600-px long side and a white 600-px text panel (font size 20,
// 25-px line gap, wrap at 26 CJK width units, panels continue horizontally)
// is concatenated on the right.
constexpr double kOfficialDropScore = 0.5;
constexpr int kOfficialPanelWidth = 600;
constexpr int kOfficialFontPx = 20;
constexpr int kOfficialLineGap = 25;
constexpr int kOfficialLongSide = 600;
constexpr int kOfficialWrapUnits = kOfficialPanelWidth / kOfficialFontPx - 4;

// str_count: CJK/fullwidth characters count 1, ASCII 0.5 (visual width).
double officialVisualWidth(const QString& text) {
    double w = 0.0;
    for (const QChar c : text) w += (c.unicode() > 127) ? 1.0 : 0.5;
    return w;
}

// The longest prefix whose visual width stays within `units` (the official
// wrap slices by str_count units, not by character count).
QString officialLeftByUnits(const QString& text, int units) {
    double w = 0.0;
    for (int i = 0; i < text.size(); ++i) {
        w += (text.at(i).unicode() > 127) ? 1.0 : 0.5;
        if (w > units) return text.left(i);
    }
    return text;
}

// A CJK-capable font family when one exists; Qt falls back gracefully.
QStringList officialFontFamilies() {
    return {
            QStringLiteral("Noto Sans CJK SC"),
            QStringLiteral("WenQuanYi Zen Hei"),
            QStringLiteral("WenQuanYi Micro Hei"),
            QStringLiteral("SimFang"),
            QStringLiteral("Microsoft YaHei"),
    };
}
}  // namespace

void qOCR::addResultToDb(const OCRRunResult& result,
                         const OCRRunSettings& settings) {
    if (!m_app || result.sourceImage.isNull()) return;

    // Official draw_ocr visualization: red 2-px polygon outlines for boxes
    // above drop_score, then the image is resized to a 600-px long side and
    // a white text panel ("index: text score" lines, wrapped, continuing
    // horizontally when full) is concatenated on the right.
    QImage annotated = result.sourceImage.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&annotated);
    QPen boxPen(QColor(255, 0, 0), 2);
    boxPen.setJoinStyle(Qt::MiterJoin);
    painter.setPen(boxPen);
    painter.setBrush(Qt::NoBrush);
    for (const OCRBoxResult& box : result.boxes) {
        if (box.score < kOfficialDropScore) continue;  // official drop_score
        QPolygonF polygon;
        for (int c = 0; c < 4; ++c)
            polygon << QPointF(box.quad[c][0], box.quad[c][1]);
        polygon << polygon.front();  // cv2.polylines(isClosed=True)
        painter.drawPolyline(polygon);
    }
    painter.end();

    // resize_img: limit the longest side to 600 (upscales small images too).
    const int longSide = qMax(annotated.width(), annotated.height());
    if (longSide > 0 && longSide != kOfficialLongSide) {
        const qreal scale = static_cast<qreal>(kOfficialLongSide) / longSide;
        annotated = annotated.scaled(qRound(annotated.width() * scale),
                                     qRound(annotated.height() * scale),
                                     Qt::IgnoreAspectRatio,
                                     Qt::SmoothTransformation);
    }

    // text_visual panel: 600 px wide, image height tall, font 20, gap 25.
    const int panelH = annotated.height();
    QFont textFont;
    textFont.setPixelSize(kOfficialFontPx);
    // Qt5/Qt6 + Qt5.12-safe family fallback (QtCompat.h feature 16)
    qtCompatSetFontFamilies(textFont, officialFontFamilies());
    QStringList lines;  // pre-wrapped "index: text score" lines
    int index = 0;
    for (const OCRBoxResult& box : result.boxes) {
        if (box.score < kOfficialDropScore) continue;
        ++index;
        QString txt = box.text;
        bool firstLine = true;
        while (officialVisualWidth(txt) >= kOfficialWrapUnits) {
            const QString head = officialLeftByUnits(txt, kOfficialWrapUnits);
            lines << (firstLine ? QStringLiteral("%1: %2").arg(index).arg(head)
                                : QStringLiteral("    %1").arg(head));
            txt = txt.mid(head.size());
            firstLine = false;
        }
        lines << (firstLine ? QStringLiteral("%1: %2   %3")
                                      .arg(index)
                                      .arg(txt)
                                      .arg(box.score, 0, 'f', 3)
                            : QStringLiteral("  %1   %2")
                                      .arg(txt)
                                      .arg(box.score, 0, 'f', 3));
    }
    // Lay the lines out into 600-px panels; full panels continue to the
    // right (np.concatenate axis=1).
    const int linesPerPanel = qMax(1, panelH / kOfficialLineGap - 1);
    const int panelCount =
            qMax(1, (lines.size() + linesPerPanel - 1) / linesPerPanel);
    QImage textPanel(kOfficialPanelWidth * panelCount, panelH,
                     QImage::Format_RGB32);
    textPanel.fill(QColor(255, 255, 255));
    QPainter panelPainter(&textPanel);
    panelPainter.setFont(textFont);
    panelPainter.setPen(QColor(0, 0, 0));
    for (int p = 0; p < panelCount; ++p) {
        // black 1-px right border per panel (blank_img[:, img_w-1:] = 0)
        panelPainter.fillRect(kOfficialPanelWidth * (p + 1) - 1, 0, 1, panelH,
                              QColor(0, 0, 0));
    }
    for (int i = 0; i < lines.size(); ++i) {
        const int panel = i / linesPerPanel;
        const int row = i % linesPerPanel;
        panelPainter.drawText(0, kOfficialLineGap * (row + 1) + 2,
                              kOfficialPanelWidth - 4, kOfficialLineGap,
                              Qt::AlignLeft | Qt::AlignTop, lines[i]);
    }
    panelPainter.end();

    // np.concatenate([img, txt_img], axis=1)
    QImage combined(annotated.width() + textPanel.width(),
                    qMax(annotated.height(), panelH), QImage::Format_RGB32);
    combined.fill(QColor(255, 255, 255));
    QPainter combiner(&combined);
    combiner.drawImage(0, 0, annotated);
    combiner.drawImage(annotated.width(), 0, textPanel);
    combiner.end();
    annotated = combined;

    const QString deviceTag = ecvPluginDbNaming::deviceTagFromName(
            result.resolvedDevice.isEmpty() ? settings.device
                                            : result.resolvedDevice);
    const QString modelTag =
            ecvPluginDbNaming::modelTagFromFilename(settings.recPath);
    const QString name = ecvPluginDbNaming::makeUnique(
            QStringLiteral("OCR_%1_%2").arg(modelTag, deviceTag), m_app);

    auto* img = new ccImage(annotated, name);
    img->setMetaData(QStringLiteral("OCR"), true);
    img->setMetaData(QStringLiteral("OCR/Count"), result.boxes.size());
    QStringList texts;
    for (const OCRBoxResult& box : result.boxes) {
        texts << QStringLiteral("%1 (%2)").arg(box.text).arg(box.score, 0, 'f',
                                                             3);
    }
    img->setMetaData(QStringLiteral("OCR/Texts"), texts.join('\n'));
    img->setMetaData(QStringLiteral("OCR/DetModel"), result.detModel);
    img->setMetaData(QStringLiteral("OCR/RecModel"), result.recModel);
    img->setMetaData(QStringLiteral("Runtime (ms)"),
                     static_cast<double>(result.e2eMs));
    img->setMetaData(QStringLiteral("OCR/Preprocess (ms)"),
                     static_cast<double>(result.preprocessMs));
    img->setMetaData(QStringLiteral("OCR/Inference (ms)"),
                     static_cast<double>(result.inferenceMs));
    img->setMetaData(QStringLiteral("OCR/Postprocess (ms)"),
                     static_cast<double>(result.postprocessMs));
    if (!result.resolvedDevice.isEmpty()) {
        img->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    img->setMetaData(QStringLiteral("Source"), result.imagePath);
    img->setMetaData(QStringLiteral("Model"),
                     QFileInfo(settings.recPath).fileName());

    // Render-window overlay: a visible ccImage draws as a pixel-space
    // foreground slice in its attached display. Attach to the currently
    // active 3D view so the user sees the end-to-end annotated result
    // without digging through the DB tree.
    if (settings.showInWindow) {
        if (ecvGenericGLDisplay* display = m_app->getActiveGLDisplay()) {
            img->setDisplay(display);
            img->setVisible(true);
            img->setEnabled(true);
        } else {
            m_dialog->appendLog(
                    tr("[qOCR] No active render window — the "
                       "result stays in the DB tree only."));
        }
    }

    m_app->addToDB(img, true, true, false, true);
    m_app->setSelectedInDB(img, true);
    if (settings.addToDb) {
        m_dialog->appendLog(tr("[qOCR] Added annotated result '%1' to the DB "
                               "tree.")
                                    .arg(name));
    } else {
        m_dialog->appendLog(tr("[qOCR] Annotated result '%1' kept out of the "
                               "DB tree (overlay only).")
                                    .arg(name));
    }
}

void qOCR::onTaskFinished(bool success) {
    m_inferenceHeartbeat->stop();
    m_dialog->setRunning(false);
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    if (!success) m_dialog->appendLog(tr("[Error] OCR task failed."));
}
