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

// Actions the worker can perform
enum class SAM3WorkerAction {
    None,
    LoadModel,
    EncodeAndSegmentPVS,
    EncodeAndSegmentPCS,
    SegmentOnly,  // re-segment on already-encoded image
};

// Result data transferred from worker back to UI thread
struct SAM3WorkerResult {
    bool valid = false;
    int detCount = 0;
    QVector<aicore_sam3_box> boxes;
    QVector<float> scores;
    QVector<float> ious;
    QVector<int> instanceIds;
    QVector<QImage> instanceMasks;  // per-detection 0/255 mask at original size
    QImage maskComposite;  // masks blended on the source image (0.4 alpha, like upstream)
    aicore_sam3_timings timings{};
    QString errorMsg;
};

// The worker emits resultReady() / frameResultReady() from its own thread;
// without this registration Qt silently drops queued (cross-thread) signals
// of this custom type, so the UI never sees the segmentation output. qYOLO
// / qRFDetr follow the same pattern (Q_DECLARE_METATYPE after the struct).
Q_DECLARE_METATYPE(SAM3WorkerResult)

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

    struct Prompt {
        // PVS mode
        QVector<aicore_sam3_point> posPoints;
        QVector<aicore_sam3_point> negPoints;
        std::vector<aicore_sam3_box> posExemplars;
        std::vector<aicore_sam3_box> negExemplars;
        aicore_sam3_box pvsBox{};
        bool usePvsBox = false;
        bool multimask = false;

        // PCS mode
        char text[256] = {};
        float scoreThreshold = 0.5f;
        float nmsThreshold = 0.1f;
    };

    explicit SAM3Worker(const Settings& settings, QObject* parent = nullptr);
    ~SAM3Worker() override;

    void setAction(SAM3WorkerAction action) { m_action = action; }
    SAM3WorkerAction action() const { return m_action; }
    void setImage(const QImage& img) { m_image = img; }
    void setPrompt(const Prompt& prompt) { m_prompt = prompt; }
    void requestCancel();
    aicore_sam3_ctx* context() const { return m_ctx; }

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
    SAM3WorkerResult buildResult(aicore_sam3_seg_result* segRes,
                                 const QImage& img);

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
