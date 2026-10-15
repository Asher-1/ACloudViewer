// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include "VideoPlaybackWidget.h"
class RAMVideoWidget : public VideoPlaybackWidget {
    Q_OBJECT
public:
    explicit RAMVideoWidget(QWidget* p = nullptr) : VideoPlaybackWidget(p) {
        setFrameAdvanceMode(FrameAdvanceMode::ConsumerDriven);
    }
    // ConsumerDriven mode only paints the image handed back here.
    void acknowledge(const QImage& image = QImage()) {
        completeFrameProcessing(image);
    }
signals:
    void frameReady(const QImage&, int);
    void invalidated();

protected:
#ifdef HAS_OPENCV_FACE_CAPTURE
    void onFrameDecoded(cv::Mat& frame, int index) override {
        emit frameReady(cvMatToQImage(frame), index);
    }
#endif
    void onStreamReset() override { emit invalidated(); }
    void onStreamStopping() override { emit invalidated(); }
    void onSourceChanged(InputSource) override { emit invalidated(); }
};
