// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace mcalib {

class RosBagReader;
struct BagMessage;

/// FFmpeg-backed H.264 / HEVC decoder with per-topic state for bag playback.
class VideoDecodeCache {
public:
    VideoDecodeCache();
    ~VideoDecodeCache();

    VideoDecodeCache(const VideoDecodeCache&) = delete;
    VideoDecodeCache& operator=(const VideoDecodeCache&) = delete;

    void clear();

    /// Decode one compressed image message, feeding prior bag messages when
    /// needed.
    bool decodeMessage(RosBagReader& reader,
                       const BagMessage& msg,
                       const std::string& format,
                       cv::Mat& image,
                       double& timestamp_sec);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class VideoCodecDecoder {
public:
    enum class Codec { H264, HEVC };

    explicit VideoCodecDecoder(Codec codec);
    ~VideoCodecDecoder();

    VideoCodecDecoder(const VideoCodecDecoder&) = delete;
    VideoCodecDecoder& operator=(const VideoCodecDecoder&) = delete;

    void reset();
    cv::Mat decodePacket(const uint8_t* data,
                         size_t size,
                         int* pict_type_out = nullptr);

    static bool isVideoFormat(const std::string& format);
    static Codec codecFromFormat(const std::string& format);
    static bool looksLikeVideoBitstream(const uint8_t* data, size_t size);
    /// True when the Annex-B buffer starts an IDR/CRA (H.264 type 5 / HEVC
    /// 19-21).
    static bool bufferContainsSyncFrame(Codec codec,
                                        const uint8_t* data,
                                        size_t size);
    static bool bufferContainsSyncFrame(const std::string& format,
                                        const std::string& buffer);

private:
    Codec codec_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mcalib
