// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OCRWorker.h"

#include <QtCompat.h>

#include <QColor>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QPainter>
#include <QPolygonF>
#include <cstring>

#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"
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
    ecvAICoreRuntime::releasePending(m_pendingTableCtx,
                                     &aicore_table_context_destroy);
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
    if (!m_settings.tablePath.isEmpty()) {
        // Table mode runs the text detection of the official PaddleX
        // table_recognition pipeline (limit_type max, 960 long side), the
        // configuration its TEDS gate is measured with.
        options.det_limit_type = 1;
        options.det_max_side = 960;
    }
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

    if (!m_settings.tablePath.isEmpty()) {
        // Table structure context on the same worker thread and device.
        aicore_table_options tableOptions{};
        tableOptions.struct_size = sizeof(tableOptions);
        tableOptions.abi_version = AICORE_TABLE_ABI_VERSION;
        const QByteArray tableBytes = QFile::encodeName(m_settings.tablePath);
        tableOptions.model_path = tableBytes.constData();
        tableOptions.device = deviceBytes.constData();
        tableOptions.n_threads = m_settings.threads;
        m_pendingTableCtx = aicore_table_context_create(&tableOptions);
        if (!m_pendingTableCtx ||
            !aicore_table_context_ready(m_pendingTableCtx)) {
            emit logMessage(
                    tr("[qOCR] Table model load failed: %1")
                            .arg(QString::fromUtf8(aicore_table_last_error())));
            emit taskFinished(false);
            return;
        }
    }

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[qOCR] Cancelled before inference."));
        emit taskFinished(false);
        return;
    }

    const QStringList images = m_settings.imagePaths.isEmpty()
                                       ? QStringList{m_settings.imagePath}
                                       : m_settings.imagePaths;
    const int total = images.size();
    int done = 0, failed = 0;
    bool cancelled = false;
    for (int i = 0; i < total; ++i) {
        if (aicore_cancel_token_requested(m_cancelToken)) {
            cancelled = true;
            break;
        }
        const QString name = QFileInfo(images[i]).fileName();
        emit taskStage((m_pendingTableCtx ? tr("Text + tables %1/%2: %3")
                                          : tr("Text %1/%2: %3"))
                               .arg(i + 1)
                               .arg(total)
                               .arg(name),
                       total > 1 ? i * 100 / total : -1);
        Settings one = m_settings;
        one.imagePath = images[i];
        OCRRunResult runResult;
        QString error;
        if (!recognizeImage(m_pendingCtx, m_pendingTableCtx, one, m_cancelToken,
                            &runResult, &error)) {
            if (aicore_cancel_token_requested(m_cancelToken)) {
                cancelled = true;
                break;
            }
            ++failed;
            emit logMessage(tr("[qOCR] %1: %2").arg(name, error));
            continue;
        }
        ++done;
        QString line = tr("[qOCR] %1/%2 %3: %4 region(s) in %5 ms [%6]")
                               .arg(i + 1)
                               .arg(total)
                               .arg(name)
                               .arg(runResult.boxes.size())
                               .arg(runResult.e2eMs, 0, 'f', 0)
                               .arg(runResult.resolvedDevice);
        if (m_pendingTableCtx) {
            line += tr(", %1 table(s) in %2 ms")
                            .arg(runResult.tables.size())
                            .arg(runResult.tableMs, 0, 'f', 0);
        }
        emit logMessage(line);
        emit resultReady(runResult);
    }

    emit taskStage(tr("Done"), 100);
    if (total > 1 || cancelled || failed) {
        emit logMessage(
                tr("[qOCR] %1 of %2 image(s) recognized, %3 failed%4")
                        .arg(done)
                        .arg(total)
                        .arg(failed)
                        .arg(cancelled ? tr(", cancelled") : QString()));
    }
    emit taskFinished(done > 0 && failed == 0 && !cancelled);
}

bool OCRWorker::recognizeImage(aicore_ocr_ctx* ctx,
                               aicore_table_ctx* tableCtx,
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
    if (!tableCtx) {
        aicore_ocr_result_release(result);
        return true;
    }

    // Table stage over the same borrowed view and OCR result: layout
    // "table" regions when the layout stage ran, else the whole image.
    aicore_table_run_options tableRun{};
    tableRun.struct_size = sizeof(tableRun);
    tableRun.abi_version = AICORE_TABLE_ABI_VERSION;
    tableRun.use_layout_regions = settings.layoutPath.isEmpty() ? 0 : 1;
    aicore_cancel_scope_begin(cancel);
    aicore_table_result* tables =
            aicore_table_run(tableCtx, &view, ctx, result, &tableRun);
    aicore_cancel_scope_end(cancel);
    aicore_ocr_result_release(result);
    if (!tables) {
        *error = tr("Table reconstruction failed: %1")
                         .arg(QString::fromUtf8(aicore_table_last_error()));
        return false;
    }
    runResult.tableModel = QFileInfo(settings.tablePath).fileName();
    const int tableCount = aicore_table_count(tables);
    runResult.tables.reserve(tableCount);
    for (int t = 0; t < tableCount; ++t) {
        const aicore_table* table = aicore_table_at(tables, t);
        if (!table) continue;
        OCRTableResult entry;
        std::memcpy(entry.box, table->box, sizeof(entry.box));
        entry.layoutRegion = table->layout_region;
        entry.rows = table->rows;
        entry.cols = table->cols;
        entry.structureScore = table->structure_score;
        entry.html = QString::fromUtf8(table->html);
        entry.csv = QByteArray(table->csv);
        entry.cells.reserve(table->cell_count);
        for (int c = 0; c < table->cell_count; ++c) {
            const aicore_table_cell* cell =
                    aicore_table_cell_at(tables, table->cell_offset + c);
            if (!cell) continue;
            OCRTableCell out;
            std::memcpy(out.quad, cell->quad, sizeof(out.quad));
            out.row = cell->row;
            out.col = cell->col;
            out.rowSpan = cell->row_span;
            out.colSpan = cell->col_span;
            out.text = QString::fromUtf8(cell->text);
            entry.cells.push_back(out);
        }
        runResult.tables.push_back(entry);
    }
    const aicore_pipeline_timings tableTimings = aicore_table_timings(tables);
    if (tableTimings.valid_fields & AICORE_TIMING_E2E) {
        runResult.tableMs = tableTimings.e2e_ms;
    }
    aicore_table_result_release(tables);
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

    // Keep the source geometry intact. A stable index beside each quad joins
    // this compact overlay to the scrollable result table without growing the
    // image for every recognized line.
    QImage annotated = result.sourceImage.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&annotated);
    QPen boxPen(QColor(220, 38, 38),
                qMax(2, qMax(annotated.width(), annotated.height()) / 1200));
    boxPen.setJoinStyle(Qt::RoundJoin);
    painter.setPen(boxPen);
    painter.setBrush(Qt::NoBrush);

    const int fontPx =
            qBound(12, qMax(annotated.width(), annotated.height()) / 90, 28);
    QFont labelFont;
    labelFont.setBold(true);
    labelFont.setPixelSize(fontPx);
    qtCompatSetFontFamilies(labelFont, officialFontFamilies());
    painter.setFont(labelFont);

    int index = 0;
    for (const OCRBoxResult& box : result.boxes) {
        if (box.score < kOfficialDropScore) continue;
        ++index;
        QPolygonF polygon;
        for (int c = 0; c < 4; ++c)
            polygon << QPointF(box.quad[c][0], box.quad[c][1]);
        painter.drawPolygon(polygon);

        const QPointF anchor(box.quad[0][0], box.quad[0][1]);
        const QRectF labelRect(anchor.x(), anchor.y() - fontPx - 6,
                               qMax(fontPx * 2, 30), fontPx + 6);
        painter.fillRect(labelRect, QColor(220, 38, 38, 220));
        painter.setPen(Qt::white);
        painter.drawText(labelRect, Qt::AlignCenter, QString::number(index));
        painter.setPen(boxPen);
    }

    // Table cells: thin blue polygons (the PaddleX table_recognition cell
    // visualization), drawn after the text so the indices stay readable.
    QPen cellPen(QColor(37, 99, 235), qMax(1, boxPen.width() / 2));
    painter.setPen(cellPen);
    for (const OCRTableResult& table : result.tables) {
        for (const OCRTableCell& cell : table.cells) {
            QPolygonF polygon;
            for (int c = 0; c < 4; ++c)
                polygon << QPointF(cell.quad[c][0], cell.quad[c][1]);
            painter.drawPolygon(polygon);
        }
    }
    painter.end();
    return annotated;
}
