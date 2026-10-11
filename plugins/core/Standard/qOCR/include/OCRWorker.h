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
#include <QThread>
#include <QVector>

struct aicore_cancel_token;
struct aicore_ocr_ctx;

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

/** Full single-image OCR outcome handed to the plugin over a queued
 *  connection. The source image travels along so the GUI thread can render
 *  the annotated DB entity without re-reading the file. */
struct OCRRunResult {
    QImage sourceImage;
    QString imagePath;
    QVector<OCRBoxResult> boxes;
    QVector<OCRRegionResult> regions;  // empty without a layout model
    // Common stage timings (ms) from aicore_ocr_timings.
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double e2eMs = 0.0;
    QString resolvedDevice;
    QString detModel;
    QString recModel;
    QString layoutModel;  // empty without a layout model
};

Q_DECLARE_METATYPE(OCRRunResult)

/** Background PP-OCR det+rec worker (single image).
 *
 *  Lifecycle contract shared by every AICore plugin worker: the model
 *  context is created inside run() on the worker thread (after acquiring
 *  the device task queue) and released on the main thread via
 *  releaseContextOnMainThread() so GPU teardown never races the render
 *  thread. Progress is stage-based: the ocr C API exposes no per-node
 *  callback, so the dialog shows the aligned busy bar with stage labels
 *  (same fallback every other AICore plugin uses on CPU/Vulkan/CUDA). */
class OCRWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString detPath;
        QString recPath;
        QString layoutPath;  // empty = layout stage off
        QString imagePath;
        QString device = QStringLiteral("auto");
        int threads = 0;  // 0 = backend default
    };

    explicit OCRWorker(const Settings& settings, QObject* parent = nullptr);
    ~OCRWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

    /** One image (settings.imagePath) through a loaded det+rec(+layout)
     *  context: typed boxes, layout regions and stage timings. Shared by
     *  run() and the headless -OCR_RECOGNIZE batch command so both paths
     *  produce identical results. \p cancel may be null. */
    static bool recognizeImage(aicore_ocr_ctx* ctx,
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
    struct aicore_ocr_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};

namespace OCRRender {
/** Official PaddleOCR draw_ocr visualization (polygons over a 600-px
 *  long-side image + wrapped "index: text score" side panel). Shared by
 *  the DB export and the batch command. Null for an empty source. */
QImage officialVisualization(const OCRRunResult& result);
}  // namespace OCRRender
