// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Modified from: WebRTC src/pc/video_track_source.h
//
// Copyright 2016 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// ----------------------------------------------------------------------------
//
// This is a private header. It shall be hidden from CloudViewer's public API.
// Do not put this in CloudViewer.h.in.

#pragma once

#include <Logging.h>
#include <absl/types/optional.h>
#include <api/notifier.h>
#include <api/sequence_checker.h>
#include <api/video/recordable_encoded_frame.h>
#include <api/video/video_frame.h>
#include <api/video/video_sink_interface.h>
#include <api/video/video_source_interface.h>
#include <media/base/media_channel.h>

#include "core/Tensor.h"

namespace cloudViewer {
namespace visualization {
namespace webrtc_server {

/// [Related classes]
/// - VideoTrackSourceInterface: WebRTC expects a custom implementation
///   of this class.
/// - BitmapTrackSourceInterface: Used as the primary interface in
///   PeerConnectionManager. It is almost the same as VideoTrackSourceInterface
///   with an additional OnFrame function for triggering frame handlers.
/// - BitmapTrackSource: Abstract class for bit map tracks.
/// - ImageTrackSource: Captures frames from CloudViewer visualizer.
/// - VideoFilter: Video frame processing, e.g. scaling.
///
/// [Class hierarchy]
/// BitmapTrackSourceInterface --inherits--> webrtc::VideoTrackSourceInterface
/// BitmapTrackSource --inherits--> webrtc::Notifier<BitmapTrackSourceInterface>
/// ImageTrackSource  --inherits--> BitmapTrackSource
/// ImageCapturer     --owned by--> ImageTrackSource
/// VideoFilter       --inherits--> BitmapTrackSource
class BitmapTrackSourceInterface : public webrtc::VideoTrackSourceInterface {
public:
    virtual void OnFrame(const std::shared_ptr<core::Tensor>& frame) = 0;
};

class BitmapTrackSource : public webrtc::Notifier<BitmapTrackSourceInterface> {
public:
    explicit BitmapTrackSource(bool remote);
    void SetState(webrtc::MediaSourceInterface::SourceState new_state);
    webrtc::MediaSourceInterface::SourceState state() const override {
        return state_;
    }
    bool remote() const override { return remote_; }
    bool is_screencast() const override { return false; }
    absl::optional<bool> needs_denoising() const override {
        return absl::nullopt;
    }
    bool GetStats(Stats* stats) override { return false; }
    void AddOrUpdateSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
                         const rtc::VideoSinkWants& wants) override;
    void RemoveSink(rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) override;
    bool SupportsEncodedOutput() const override { return false; }
    void GenerateKeyFrame() override {}
    void AddEncodedSink(rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>*
                                sink) override {}
    void RemoveEncodedSink(
            rtc::VideoSinkInterface<webrtc::RecordableEncodedFrame>* sink)
            override {}

    virtual void OnFrame(const std::shared_ptr<core::Tensor>& frame) override {
        // Shall be implemented by child class.
        utility::LogError("BitmapTrackSource::OnFrame called");
    }

protected:
    virtual rtc::VideoSourceInterface<webrtc::VideoFrame>* source() = 0;

private:
    webrtc::SequenceChecker worker_thread_checker_;
    webrtc::MediaSourceInterface::SourceState state_;
    const bool remote_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace cloudViewer
