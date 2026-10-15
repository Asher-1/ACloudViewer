// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include "VideoPlaybackWidget.h"
class RelateAnythingVideoWidget : public VideoPlaybackWidget {
    Q_OBJECT
public:
    explicit RelateAnythingVideoWidget(QWidget* p = nullptr)
        : VideoPlaybackWidget(p) {
        setFrameAdvanceMode(FrameAdvanceMode::ConsumerDriven);
    }
    void acknowledge(const QImage& image = QImage()) {
        completeFrameProcessing(image);
    }
signals:
    void frameReady(const QImage&, int);
    void invalidated();
    // A video file reached its last frame and wrapped to the first one.
    void looped();

protected:
#ifdef HAS_OPENCV_FACE_CAPTURE
    void onFrameDecoded(cv::Mat& frame, int index) override {
        emit frameReady(cvMatToQImage(frame), index);
    }
#endif
    void onVideoLooped() override {
        // Tracks and relation beliefs of the last frame must not carry over
        // to the first frame of the next pass.
        emit invalidated();
        emit looped();
    }
    void onStreamReset() override { emit invalidated(); }
    void onStreamStopping() override { emit invalidated(); }
    void onSourceChanged(InputSource) override { emit invalidated(); }
};
