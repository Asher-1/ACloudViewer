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
    QImage maskComposite;           // all masks blended on a black background
    aicore_sam3_timings timings{};
    QString errorMsg;
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
    QImage blendMasks(aicore_sam3_seg_result* res, int imgW, int imgH);

    Settings m_settings;
    SAM3WorkerAction m_action = SAM3WorkerAction::None;
    QImage m_image;
    Prompt m_prompt;
    aicore_sam3_ctx* m_ctx = nullptr;
    aicore_sam3_ctx* m_pendingCtx = nullptr;
    bool m_cancelled = false;
};