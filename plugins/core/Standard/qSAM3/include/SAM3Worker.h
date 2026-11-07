// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <aicore/sam3_capi.h>

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QString>
#include <QThread>
#include <QVector>
#include <atomic>

#include "SAM3Types.h"

// Actions the worker can perform
enum class SAM3WorkerAction {
    None,
    LoadModel,
    EncodeAndSegmentPVS,
    EncodeAndSegmentPCS,
    SegmentOnly,  // re-segment on already-encoded image
};

class SAM3Worker : public QThread {
    Q_OBJECT
public:
    struct Settings {
        QString modelPath;
        QString device = "auto";
        int threads = 4;
        int encodeImgSize = 0;
        float scoreThreshold = 0.5f;
        float nmsThreshold = 0.1f;
        float assocIouThreshold = 0.1f;
        int hotstartDelay = 15;
        int maxKeepAlive = 30;
        int reconditionEvery = 16;
        int fillHoleArea = 16;
    };

    using Prompt = SAM3Prompt;

    explicit SAM3Worker(const Settings& settings, QObject* parent = nullptr);
    ~SAM3Worker() override;

    void setAction(SAM3WorkerAction action) { m_action = action; }
    SAM3WorkerAction action() const { return m_action; }
    void setImage(const QImage& img) { m_image = img; }
    void setPrompt(const Prompt& prompt) { m_prompt = prompt; }
    void requestCancel();
    aicore_sam3_ctx* context() const { return m_ctx; }

    /** Optional encode + one PVS (pcs=false) or PCS decode on an RGB888
     *  image: the single per-image path of the dialog and -SAM3_SEGMENT.
     *  NULL + \p error on failure; free with aicore_sam3_seg_result_free. */
    static aicore_sam3_seg_result* segment(aicore_sam3_ctx* ctx,
                                           const QImage& rgb,
                                           const Prompt& prompt,
                                           bool pcs,
                                           bool encode,
                                           QString* error);
    /** Typed copy of \p segRes (boxes, scores, masks, overlay, timings). */
    static SAM3WorkerResult buildResult(aicore_sam3_ctx* ctx,
                                        aicore_sam3_seg_result* segRes,
                                        const QImage& img);

signals:
    void progressUpdate(int current, int total);
    void logMessage(const QString& msg);
    void resultReady(const SAM3WorkerResult& result);
    void modelReady(const QString& backendName, int modelType, bool visualOnly);

protected:
    void run() override;

private:
    bool runInference();
    aicore_sam3_seg_result* runPVS();
    aicore_sam3_seg_result* runPCS();

    Settings m_settings;
    SAM3WorkerAction m_action = SAM3WorkerAction::None;
    QImage m_image;
    Prompt m_prompt;
    aicore_sam3_ctx* m_ctx = nullptr;
    aicore_sam3_ctx* m_pendingCtx = nullptr;
    std::atomic_bool m_cancelled{false};

    // Encoding cache: the C-API caches the encoded features by image size;
    // we gate re-encoding on the actual image + pvs_only mode so repeated
    // point/box clicks on the same picture only run the decoder, matching
    // upstream examples/main_image.cpp (encode once, segment per click).
    qint64 m_encodedImageKey = -1;
    int m_encodedWidth = 0;
    int m_encodedHeight = 0;
    bool m_encodedPvsOnly = false;
};
