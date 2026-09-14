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
#include <vector>

#include "GKDModelCatalog.h"

struct aicore_cancel_token;
struct aicore_gkd_ctx;

/** Cross-run inference-context cache owned by the plugin (main thread).
 *
 *  Reloading the GKD GGUF (483 MiB) — and for multi-object mode the
 *  YOLO-World + CLIP text towers — dominated every run (measured:
 *  "Model loaded" logged on every Run click). Workers borrow handles
 *  from this cache and populate it on first use; frees happen on the
 *  main thread only (the plugin drains `retiredGkd/retiredYolo` in
 *  onTaskFinished and frees the live entries in its destructor). Keys
 *  encode every option that forces a reload: weights file, device,
 *  threads, and for YOLO also the class list + cuts (the C API only
 *  accepts classes at load time). Runs are serialized by the plugin
 *  (one worker at a time), so no extra locking is needed. */
struct GKDContextCache {
    void* gkdCtx = nullptr; /**< aicore_gkd_ctx* (opaque) */
    QString gkdKey;
    void* yoloCtx = nullptr; /**< aicore_yolo_ctx* (opaque) */
    QString yoloKey;
    /** Handles replaced by a newer load; freed on the main thread so GPU
     *  teardown never races the render thread. */
    std::vector<void*> retiredGkd;
    std::vector<void*> retiredYolo;
};

/** Background GKDT keypoint detection worker.
 *
 *  The GKD context (and the optional YOLO-World detector context for
 *  multi-object mode) is created inside run() on the worker thread. When
 *  Settings::cache is set, the context is stored there and reused across
 *  runs; without a cache the worker owns it and releases it on the main
 *  thread via releaseContextOnMainThread() so GPU teardown never races
 *  the render thread. Cancellation is cooperative via a caller-owned
 *  aicore_cancel_token. */
class GKDWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath; /**< GKD GGUF */
        QString inputPath; /**< query image (file path) */
        int threads = 0;
        QString device = QStringLiteral("auto");
        /** Text prompts ("nose", "left eye", ...). */
        QStringList kpsTexts;
        /** Official-demo skeleton "1-2 1-3 ..." (1-based keypoint pairs,
         *  empty for manual runs: bones render only from presets that
         *  carry an upstream skeleton). */
        QString skeleton;
        /** Optional 1-shot visual prompt. */
        QString supportImagePath;
        QVector<QPointF> supportKps; /**< support-image pixel coords */
        /** Optional ROI [x1,y1,x2,y2] on the query image (single-object). */
        bool hasBbox = false;
        float bbox[4] = {0, 0, 0, 0};
        /** Parsed ROI-row point pairs: 2 points = the legacy single box,
         *  4+ points = a multi-ROI batch (xyxy per box, official
         *  --bbox_on_input_im semantics). */
        QVector<QPointF> roiPoints;
        /** Multi-object mode: run YOLO-World first, then GKD per box. */
        bool multiObject = false;
        QStringList objectClasses; /**< open-vocabulary class names */
        QString yoloModelPath;     /**< YOLO-World GGUF (WOLD family) */
        /** Text-encoder GGUF (CLIP/MobileCLIP) that encodes the class
         *  names; text-conditioned WORLD detectors reject the run without
         *  it (aicore_yolo_options_set_text_model). */
        QString yoloTextModelPath;
        /** YOLO-World confidence cut. 0.25 = the qYOLO / ultralytics /
         *  backend default. Measured via the public C ABI on the bundled
         *  demo scenes: custom open-vocabulary prompts (pig, fish,
         *  statue) score systematically below closed-set ones, so higher
         *  cuts drop real objects (fish school: 7 vs 23 at 0.25). */
        float yoloConf = 0.25f;
        /** Keypoints below this score are dropped from the result. GKDT
         *  scores are cosine similarities (uncalibrated, domain-shifting):
         *  measured invisible/wrong peaks sit <= 0.08 while visible ones
         *  sit >= 0.2, so 0.10 sits in the empirical gap (0.30 hid valid
         *  open-world keypoints entirely). */
        float minScore = 0.10f;
        /** Draw per-keypoint "prompt score" labels over the result.
         *  Default OFF: in multi-object scenes the label backgrounds
         *  alone cover the objects (63 labels buried the cat_dog
         *  lineup); the dialog checkbox opts in per run, and labeled
         *  runs get a de-overlap layout with thin leader arrows. */
        bool pointLabels = false;
        bool addResultToDb = true;
        QString savePngDir;
        /** Optional cross-run context cache (owned by the plugin). When
         *  null the worker owns its contexts for this run only. */
        GKDContextCache* cache = nullptr;
    };

    explicit GKDWorker(const Settings& settings, QObject* parent = nullptr);
    ~GKDWorker() override;

    /** Move the pending model contexts back to the main thread and free
     *  them. Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void taskStage(const QString& stage, int percent = -1);
    void resultReady(const GKDRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool detectSingleObject(QVector<GKDKeypointSet>* sets, QString* error);
    bool detectMultiObject(QVector<GKDKeypointSet>* sets, QString* error);
    bool ensureGkdContext(QString* error);
    bool ensureYoloContext(QString* error);
#endif

    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
    void* m_pendingGkdCtx = nullptr;  /**< aicore_gkd_ctx* (opaque) */
    void* m_pendingYoloCtx = nullptr; /**< aicore_yolo_ctx* (opaque) */
    /** True when this worker loaded the handle itself (must free it);
     *  false for handles borrowed from the shared cache. */
    bool m_ownsGkdCtx = false;
    bool m_ownsYoloCtx = false;
};
