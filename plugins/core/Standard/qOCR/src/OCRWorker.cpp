// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OCRWorker.h"

#include <QtCompat.h>

#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QPainter>
#include <cstring>

#include "aicore/ocr_capi.h"
#include "ecvAICoreRuntimeHelpers.h"

OCRWorker::OCRWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    m_cancelToken = aicore_cancel_token_new();
}

OCRWorker::~OCRWorker() {
    releaseContextOnMainThread();
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
}

void OCRWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
    ecvAICoreRuntime::releasePending(m_pendingCtx, &aicore_ocr_context_destroy);
}

void OCRWorker::requestTaskCancel() {
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
}

void OCRWorker::run() {
    // Serialize against every other AICore inference task on the same
    // device (live video loops, other plugin workers): concurrent graph
    // compute on one backend races its command queues and poisons the
    // backend for the rest of the process.
    auto taskGuard = ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[qOCR] Failed to acquire the inference device; another "
                   "task is running."));
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Loading models…"), -1);
    emit logMessage(
            tr("[qOCR] Loading det/rec models (device=%1, threads=%2)...")
                    .arg(m_settings.device)
                    .arg(m_settings.threads));

    aicore_ocr_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = AICORE_OCR_ABI_VERSION;
    const QByteArray detBytes = QFile::encodeName(m_settings.detPath);
    const QByteArray recBytes = QFile::encodeName(m_settings.recPath);
    const QByteArray deviceBytes = m_settings.device.toUtf8();
    options.det_model_path = detBytes.constData();
    options.rec_model_path = recBytes.constData();
    // Optional layout stage: the pack travels through the same settings
    // channel; an empty path keeps the regions array empty.
    QByteArray layoutBytes;
    if (!m_settings.layoutPath.isEmpty()) {
        layoutBytes = QFile::encodeName(m_settings.layoutPath);
        options.layout_model_path = layoutBytes.constData();
    }
    options.device = deviceBytes.constData();
    options.n_threads = m_settings.threads;
    m_pendingCtx = aicore_ocr_context_create(&options);
    if (!m_pendingCtx) {
        emit logMessage(
                tr("[qOCR] Model load failed: %1")
                        .arg(QString::fromUtf8(aicore_ocr_last_error())));
        emit taskFinished(false);
        return;
    }
    if (!aicore_ocr_context_ready(m_pendingCtx)) {
        emit logMessage(tr("[qOCR] Model context is not ready."));
        emit taskFinished(false);
        return;
    }
    const QString resolvedDevice =
            QString::fromUtf8(aicore_ocr_context_device(m_pendingCtx));
    emit logMessage(tr("[qOCR] Models loaded on device: %1")
                            .arg(resolvedDevice.isEmpty() ? m_settings.device
                                                          : resolvedDevice));

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[qOCR] Cancelled before inference."));
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Detecting and recognizing text…"), -1);
    OCRRunResult runResult;
    QString error;
    if (!recognizeImage(m_pendingCtx, m_settings, m_cancelToken, &runResult,
                        &error)) {
        emit logMessage(tr("[qOCR] %1").arg(error));
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Done"), 100);
    emit logMessage(tr("[qOCR] %1 region(s) recognized in %2 ms [%3]")
                            .arg(runResult.boxes.size())
                            .arg(runResult.e2eMs, 0, 'f', 0)
                            .arg(runResult.resolvedDevice));
    emit resultReady(runResult);
    emit taskFinished(true);
}

bool OCRWorker::recognizeImage(aicore_ocr_ctx* ctx,
                               const Settings& settings,
                               aicore_cancel_token* cancel,
                               OCRRunResult* out,
                               QString* error) {
    // Borrowed, stride-aware image view (ecvAICoreRuntimeHelpers). The
    // converted QImage must outlive the aicore_ocr_run call below.
    QImage source(settings.imagePath);
    if (source.isNull()) {
        *error = tr("Failed to load image: %1").arg(settings.imagePath);
        return false;
    }
    aicore_image_view view = ecvAICoreRuntime::makeImageView(source);
    if (!view.data) {
        // Unsupported source format: convert once, keep it alive for the run.
        source = source.convertToFormat(QImage::Format_RGB888);
        view = ecvAICoreRuntime::makeImageView(source);
    }
    if (!view.data) {
        *error = tr("Unsupported image format.");
        return false;
    }

    aicore_cancel_scope_begin(cancel);
    aicore_ocr_result* result = aicore_ocr_run(ctx, &view);
    aicore_cancel_scope_end(cancel);
    if (!result) {
        *error = tr("Inference failed: %1")
                         .arg(QString::fromUtf8(aicore_ocr_last_error()));
        return false;
    }

    OCRRunResult& runResult = *out;
    runResult.sourceImage = source;
    runResult.imagePath = settings.imagePath;
    runResult.detModel = QFileInfo(settings.detPath).fileName();
    runResult.recModel = QFileInfo(settings.recPath).fileName();
    runResult.layoutModel = settings.layoutPath.isEmpty()
                                    ? QString()
                                    : QFileInfo(settings.layoutPath).fileName();
    runResult.resolvedDevice =
            QString::fromUtf8(aicore_ocr_context_device(ctx));
    const int count = aicore_ocr_box_count(result);
    runResult.boxes.reserve(count);
    for (int i = 0; i < count; ++i) {
        const aicore_ocr_box* box = aicore_ocr_box_at(result, i);
        if (!box) continue;
        OCRBoxResult entry;
        std::memcpy(entry.quad, box->quad, sizeof(entry.quad));
        entry.text = QString::fromUtf8(box->text);
        entry.score = box->score;
        entry.language = QString::fromLatin1(box->language);
        runResult.boxes.push_back(entry);
    }
    const int regionCount = aicore_ocr_region_count(result);
    runResult.regions.reserve(regionCount);
    for (int r = 0; r < regionCount; ++r) {
        const aicore_ocr_region* region = aicore_ocr_region_at(result, r);
        if (!region) continue;
        OCRRegionResult entry;
        entry.classId = region->class_id;
        entry.readingOrder = region->reading_order;
        std::memcpy(entry.quad, region->quad, sizeof(entry.quad));
        entry.score = region->score;
        const char* className = aicore_ocr_label_at(ctx, region->class_id);
        entry.className = className ? QString::fromUtf8(className)
                                    : QString::number(region->class_id);
        runResult.regions.push_back(entry);
    }
    const aicore_pipeline_timings timings = aicore_ocr_timings(result);
    if (timings.valid_fields & AICORE_TIMING_PREPROCESS) {
        runResult.preprocessMs = timings.preprocess_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_INFERENCE) {
        runResult.inferenceMs = timings.inference_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_POSTPROCESS) {
        runResult.postprocessMs = timings.postprocess_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_E2E) {
        runResult.e2eMs = timings.e2e_ms;
    }
    aicore_ocr_result_release(result);
    return true;
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

QImage OCRRender::officialVisualization(const OCRRunResult& result) {
    if (result.sourceImage.isNull()) return QImage();
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
        panelPainter.drawText(kOfficialPanelWidth * panel,
                              kOfficialLineGap * (row + 1) + 2,
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
    return annotated;
}
