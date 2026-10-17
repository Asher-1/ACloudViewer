// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QVector>

struct aicore_cancel_token;
struct aicore_ocr_ctx;
struct aicore_table_ctx;

/** One recognized text region (typed mirror of aicore_ocr_box). */
struct OCRBoxResult {
    float quad[4][2] = {};  ///< corner polygon, source-image pixel coordinates
    QString text;
    float score = 0.0f;
    QString language;
};

Q_DECLARE_METATYPE(OCRBoxResult)

/** One layout region (typed mirror of aicore_ocr_region). */
struct OCRRegionResult {
    int classId = -1;
    int readingOrder = -1;
    float quad[4][2] = {};  ///< axis-aligned rect corners, source pixels
    float score = 0.0f;
    QString className;
};

Q_DECLARE_METATYPE(OCRRegionResult)

/** One table cell (typed mirror of aicore_table_cell). */
struct OCRTableCell {
    float quad[4][2] = {};  ///< predicted cell polygon, source pixels
    int row = 0, col = 0, rowSpan = 1, colSpan = 1;
    QString text;
};

/** One reconstructed table (typed mirror of aicore_table). */
struct OCRTableResult {
    float box[4] = {};      ///< table region [xmin, ymin, xmax, ymax]
    int layoutRegion = -1;  ///< source layout region, -1 = whole image
    int rows = 0, cols = 0;
    float structureScore = 0.0f;
    QVector<OCRTableCell> cells;
    QString html;
    QByteArray csv;  ///< RFC 4180, UTF-8
};

/** Full single-image OCR outcome handed to the plugin over a queued
 *  connection. The source image travels along so the GUI thread can render
 *  the annotated DB entity without re-reading the file. */
struct OCRRunResult {
    QImage sourceImage;
    QString imagePath;
    QVector<OCRBoxResult> boxes;
    QVector<OCRRegionResult> regions;  // empty without a layout model
    QVector<OCRTableResult> tables;    // empty without a table model
    // Common stage timings (ms) from aicore_ocr_timings.
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double e2eMs = 0.0;
    QString resolvedDevice;
    QString detModel;
    QString recModel;
    QString layoutModel;   // empty without a layout model
    QString tableModel;    // empty without a table model
    double tableMs = 0.0;  // table stage e2e (aicore_table_timings)
};

Q_DECLARE_METATYPE(OCRRunResult)

/** Background PP-OCR det+rec(+layout/table) worker over a list of images.
 *
 *  Lifecycle contract shared by every AICore plugin worker: the model
 *  contexts are created and destroyed inside run() on the worker thread while
 *  the device task queue is held. Each image emits resultReady as soon as it
 *  is done (a
 *  failing image is logged and skipped); cancellation is checked between
 *  images and inside each run. Progress is stage-based per image: the ocr C
 *  API exposes no per-node callback. */
class OCRWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString detPath;
        QString recPath;
        QString layoutPath;  // empty = layout stage off
        /** SLANet table GGUF; empty = table stage off. With a layout model
         *  every "table" region is reconstructed, otherwise the whole
         *  image is one table. */
        QString tablePath;
        /** Image of one recognizeImage call. */
        QString imagePath;
        /** run(): images processed in order (empty = {imagePath}). */
        QStringList imagePaths;
        QString device = QStringLiteral("auto");
        int threads = 0;  // 0 = backend default
    };

    explicit OCRWorker(const Settings& settings, QObject* parent = nullptr);
    ~OCRWorker() override;

    void requestTaskCancel();

    /** One image (settings.imagePath) through a loaded det+rec(+layout)
     *  context and, when \p tableCtx is set, the table stage: typed boxes,
     *  layout regions, tables and stage timings. Shared by run() and the
     *  headless -OCR_RECOGNIZE batch command so both paths produce
     *  identical results. \p cancel and \p tableCtx may be null. */
    static bool recognizeImage(aicore_ocr_ctx* ctx,
                               aicore_table_ctx* tableCtx,
                               const Settings& settings,
                               aicore_cancel_token* cancel,
                               OCRRunResult* result,
                               QString* error);

signals:
    void logMessage(const QString& msg);
    /** Stage progress; percent < 0 puts the progress bar into busy
     *  (indeterminate) mode. */
    void taskStage(const QString& stage, int percent);
    void resultReady(const OCRRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
};

namespace OCRRender {
/** Source-size in-app overlay with indexed text quads and table cells. */
QImage overlayVisualization(const OCRRunResult& result);
/** PaddleOCR draw_ocr-style export: red quads, 600-px long-side image and
 *  one or more white 600-px text panels with CJK-aware wrapping. */
QImage officialVisualization(const OCRRunResult& result);
}  // namespace OCRRender
