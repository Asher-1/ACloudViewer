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

    bool hasModel() const { return m_ctx != nullptr; }
    bool isBusy() const { return m_busy; }

signals:
    void logMessage(const QString& msg);
    void modelReady(const QString& backendName, bool visualOnly);
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
    QVector<TrackRequest> m_queue;
    bool m_cancel = false;
    bool m_busy = false;

    aicore_sam3_ctx* m_ctx = nullptr;
    aicore_sam3_tracker_ctx* m_tracker = nullptr;
    bool m_visualOnly = false;
};
