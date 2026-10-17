// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QRectF>
#include <QString>
#include <QThread>
#include <QVector>

#include "YOLOModelCatalog.h"

struct aicore_cancel_token;
struct aicore_yolo_ctx;

/** Background YOLO inference worker (single image). The task follows the
 *  model: a detect GGUF emits resultReady (boxes), a depth GGUF emits
 *  depthResultReady (metric depth map). The model context is created inside
 *  run() on the worker thread; it is released on the main thread via
 *  releaseContextOnMainThread() so GPU teardown never races the render
 *  thread. */
class YOLOWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
        // Open-vocabulary knobs (world/yoloe tabs): the class list is
        // encoded through the text model GGUF at load time.
        QStringList classes;
        QString textModelPath;
        // YOLOE visual prompts (SAVPE): full-image pixel boxes; non-empty
        // switches the run to the visual-prompt path (classes are ignored).
        QList<QRectF> visualPrompts;
        // Optional user-assigned names, index-aligned with visualPrompts;
        // empty entries keep the backend's positional objectN label.
        QStringList visualPromptNames;
    };

    explicit YOLOWorker(const Settings& settings, QObject* parent = nullptr);
    ~YOLOWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

#ifdef AICore_ENABLED
    /** Typed output of one image: run (all non-depth tasks) or depth, plus
     *  the human-readable summary/hint lines the dialog logs. */
    struct ImageResult {
        bool isDepth = false;
        YOLORunResult run;
        YOLODepthResult depth;
        QString summary;
        QString hint;
    };

    /** Build options from \p settings (thresholds, class list + text model,
     *  visual prompts) and load the context. Returns the context even when
     *  it is not ready (caller frees it); \p error is set on failure. */
    static aicore_yolo_ctx* loadContext(const Settings& settings,
                                        QString* error);
    /** One image (settings.inputPath) through a loaded context; dispatches
     *  on the model's task. Shared by run() and the headless -YOLO_PREDICT
     *  batch command so both paths produce identical results. \p cancel
     *  may be null. */
    static bool inferImage(aicore_yolo_ctx* ctx,
                           const Settings& settings,
                           aicore_cancel_token* cancel,
                           ImageResult* out,
                           QString* error);
#endif

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const YOLORunResult& result);
    void depthResultReady(const YOLODepthResult& result);
    void taskFinished(bool success);
    void modelInfoReady(const QString& info);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    static bool runDetect(aicore_yolo_ctx* ctx,
                          const Settings& settings,
                          aicore_cancel_token* cancel,
                          const QImage& rgb,
                          ImageResult* out,
                          QString* error);
    static bool runSegment(aicore_yolo_ctx* ctx,
                           const Settings& settings,
                           aicore_cancel_token* cancel,
                           const QImage& rgb,
                           ImageResult* out,
                           QString* error);
    static bool runDepth(aicore_yolo_ctx* ctx,
                         const Settings& settings,
                         aicore_cancel_token* cancel,
                         const QImage& rgb,
                         ImageResult* out,
                         QString* error);
    static bool runPose(aicore_yolo_ctx* ctx,
                        const Settings& settings,
                        aicore_cancel_token* cancel,
                        const QImage& rgb,
                        ImageResult* out,
                        QString* error);
    static bool runObb(aicore_yolo_ctx* ctx,
                       const Settings& settings,
                       aicore_cancel_token* cancel,
                       const QImage& rgb,
                       ImageResult* out,
                       QString* error);
    static bool runSemantic(aicore_yolo_ctx* ctx,
                            const Settings& settings,
                            aicore_cancel_token* cancel,
                            const QImage& rgb,
                            ImageResult* out,
                            QString* error);
    static bool runClassify(aicore_yolo_ctx* ctx,
                            const Settings& settings,
                            aicore_cancel_token* cancel,
                            const QImage& rgb,
                            ImageResult* out,
                            QString* error);
    /** User-assigned name for prompt index \p classId (empty = keep the
     *  backend's positional objectN label); shared by the detect and
     *  segment result paths. */
    static QString visualPromptName(const Settings& settings, uint32_t classId);
#endif

    Settings m_settings;
    struct aicore_yolo_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};
