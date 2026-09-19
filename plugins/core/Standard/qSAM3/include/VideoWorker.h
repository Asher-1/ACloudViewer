// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Video tracking worker for the qSAM3 video tab.
//
// Owns the model context and the tracker for the whole video session (unlike
// the image-mode SAM3Worker, which creates a fresh context per action). All
// C-API calls happen on this thread; the UI thread posts requests through a
// mutex-protected queue and receives results via queued signals.

#pragma once

#include <aicore/sam3_capi.h>

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QThread>
#include <QVector>
#include <QWaitCondition>
#include <atomic>

#include "SAM3Worker.h"  // SAM3WorkerResult / Prompt types

class VideoWorker : public QThread {
    Q_OBJECT
public:
    enum class Action {
        None,
        LoadModel,    // load ctx + create tracker (with optional text prompt)
        TrackFrame,   // track_frame / propagate_frame on the given frame
        AddInstance,  // tracker_add_instance from PVS prompts
        RefineInstance,  // refine_instance with pos/neg points
        ResetTracker,    // clear tracker state and re-create it
    };

    struct TrackRequest {
        Action action = Action::None;
        QString modelPath;
        QString device = "auto";
        QString textPrompt;
        float scoreThreshold = 0.5f;
        QImage frame;               // TrackFrame
        int frameIndex = -1;        // TrackFrame: source frame number
        SAM3Worker::Prompt prompt;  // AddInstance / RefineInstance
        int instanceId = -1;        // RefineInstance
        bool cancel = false;
    };

    explicit VideoWorker(QObject* parent = nullptr);
    ~VideoWorker() override;

    void post(const TrackRequest& req);
    void requestCancel();

    bool hasModel() const { return m_hasModel.load(); }
    bool isBusy() const { return m_busy.load(); }

signals:
    void logMessage(const QString& msg);
    void modelReady(const QString& backendName, bool visualOnly);
    void trackerReset(const QString& textPrompt, bool ok);
    void frameResultReady(const SAM3WorkerResult& result, int frameIndex);
    void instanceAdded(int instanceId);
    void instanceRefined(int instanceId, bool ok);
    void busyChanged(bool busy);

protected:
    void run() override;

private:
    void process(const TrackRequest& req);
    SAM3WorkerResult buildResult(aicore_sam3_seg_result* segRes,
                                 const QImage& frame);

    mutable QMutex m_mutex;
    QWaitCondition m_condition;
    QVector<TrackRequest> m_queue;
    std::atomic_bool m_cancel{false};
    std::atomic_bool m_busy{false};
    std::atomic_bool m_hasModel{false};

    aicore_sam3_ctx* m_ctx = nullptr;
    aicore_sam3_tracker_ctx* m_tracker = nullptr;
    QString m_device = QStringLiteral("auto");
    bool m_visualOnly = false;
    /** Result of the most recent TrackFrame; merged with the mask of a new
     *  instance so the UI shows old + new instances at once (upstream
     *  main_video.cpp appends the PVS mask to the current result). */
    SAM3WorkerResult m_lastFrameResult;
    /** Frame index whose features are currently encoded in the tracker
     *  state; -1 = none. Mirrors upstream main_video.cpp frame_encoded:
     *  add_instance / refine run PVS internally and need valid encoded
     *  features of the current frame (sam3_segment_pvs bails out with
     *  "image not encoded" otherwise). */
    int m_encodedFrameIndex = -1;
};
