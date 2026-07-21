// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoCodecDecoder.h"

#include <CVLog.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <set>

#include "ProtoDecoder.h"
#include "RosBagReader.h"

#if defined(MCALIB_HAS_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

namespace mcalib {

namespace {

bool isCompressedImageTopic(const std::string& topic) {
    return topic.find("/sensors/camera/") != std::string::npos &&
           (topic.find("_raw_data/compressed") != std::string::npos ||
            topic.find("/compressed_proto") != std::string::npos);
}

bool extractFormatAndBuffer(const std::string& bag_msg_data,
                            std::string& format,
                            std::string& image_buffer,
                            double& timestamp_sec) {
    format.clear();
    image_buffer.clear();
    timestamp_sec = 0;
    if (bag_msg_data.size() < 4) return false;

    uint32_t str_len = 0;
    std::memcpy(&str_len, bag_msg_data.data(), 4);
    if (4 + str_len > bag_msg_data.size()) return false;

    const std::string proto_data =
            ProtoDecoder::stripChurchHeader(bag_msg_data.substr(4, str_len));
    for (const auto& f : ProtoDecoder::parseFields(proto_data)) {
        if (f.field_number == 1 &&
            f.wire_type == ProtoDecoder::LENGTH_DELIMITED) {
            timestamp_sec = ProtoDecoder::decodeHeaderTimestamp(f.bytes_val);
        } else if (f.field_number == 2 &&
                   f.wire_type == ProtoDecoder::LENGTH_DELIMITED) {
            format = f.bytes_val;
        } else if (f.field_number == 3 &&
                   f.wire_type == ProtoDecoder::LENGTH_DELIMITED) {
            image_buffer = f.bytes_val;
        }
    }
    return !image_buffer.empty();
}

uint64_t findVideoSyncStartNs(RosBagReader& reader,
                              const std::string& topic,
                              uint64_t target_ns) {
    if (target_ns == 0) return reader.getBeginTime();

    const uint64_t bag_begin = reader.getBeginTime();
    constexpr uint64_t kLookbackNs = 6000000000ULL;
    const uint64_t start_ns =
            target_ns > kLookbackNs ? target_ns - kLookbackNs : bag_begin;

    std::set<std::string> topic_filter{topic};
    const auto prior_msgs = reader.readAllMessagesParallel(
            topic_filter, nullptr, start_ns, target_ns);

    uint64_t sync_ns = bag_begin;
    for (const auto& prior : prior_msgs) {
        std::string prior_format;
        std::string prior_buffer;
        double prior_ts = 0;
        if (!extractFormatAndBuffer(prior.data, prior_format, prior_buffer,
                                    prior_ts)) {
            continue;
        }
        if (VideoCodecDecoder::bufferContainsSyncFrame(prior_format,
                                                       prior_buffer)) {
            sync_ns = prior.timestamp_ns;
        }
    }
    return sync_ns;
}

// Reject only fully uniform frames (FFmpeg error-concealment gray fill).
bool isFrameUniform(const cv::Mat& frame) {
    if (frame.empty() || frame.rows < 2 || frame.cols < 2) return true;

    const cv::Vec3b first = frame.at<cv::Vec3b>(0, 0);
    const int step_y = std::max(1, frame.rows / 10);
    const int step_x = std::max(1, frame.cols / 10);
    for (int y = 0; y < frame.rows; y += step_y) {
        for (int x = 0; x < frame.cols; x += step_x) {
            if (frame.at<cv::Vec3b>(y, x) != first) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

#if defined(MCALIB_HAS_FFMPEG)

struct VideoCodecDecoder::Impl {
    Codec codec = Codec::H264;
    AVCodecContext* ctx = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* sws = nullptr;
    int last_width = 0;
    int last_height = 0;
    AVPixelFormat last_pix_fmt = AV_PIX_FMT_NONE;

    ~Impl() { reset(); }

    void reset() {
        if (sws) {
            sws_freeContext(sws);
            sws = nullptr;
        }
        if (frame) {
            av_frame_free(&frame);
            frame = nullptr;
        }
        if (ctx) {
            avcodec_flush_buffers(ctx);
            avcodec_free_context(&ctx);
            ctx = nullptr;
        }
        last_width = 0;
        last_height = 0;
        last_pix_fmt = AV_PIX_FMT_NONE;
    }

    bool ensureOpen() {
        if (ctx) return true;

        const AVCodecID codec_id =
                codec == Codec::HEVC ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
        const AVCodec* av_codec = avcodec_find_decoder(codec_id);
        if (!av_codec) {
            CVLog::Warning("[VideoCodecDecoder] avcodec_find_decoder failed");
            return false;
        }

        ctx = avcodec_alloc_context3(av_codec);
        if (!ctx) return false;

        ctx->thread_count = 1;
        if (avcodec_open2(ctx, av_codec, nullptr) < 0) {
            CVLog::Warning("[VideoCodecDecoder] avcodec_open2 failed");
            reset();
            return false;
        }

        frame = av_frame_alloc();
        return frame != nullptr;
    }

    cv::Mat decodePacket(const uint8_t* data, size_t size, int* pict_type_out) {
        if (pict_type_out) *pict_type_out = 0;
        if (!ensureOpen() || data == nullptr || size == 0) return {};

        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = const_cast<uint8_t*>(data);
        pkt.size = static_cast<int>(size);

        avcodec_send_packet(ctx, &pkt);

        cv::Mat result;
        int last_pict_type = 0;
        while (true) {
            const int recv_ret = avcodec_receive_frame(ctx, frame);
            if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                break;
            }
            if (recv_ret < 0) {
                break;
            }

            if (frame->width <= 0 || frame->height <= 0) continue;

            if (!sws || frame->width != last_width ||
                frame->height != last_height || frame->format != last_pix_fmt) {
                if (sws) {
                    sws_freeContext(sws);
                    sws = nullptr;
                }
                sws = sws_getContext(frame->width, frame->height,
                                     static_cast<AVPixelFormat>(frame->format),
                                     frame->width, frame->height,
                                     AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr,
                                     nullptr, nullptr);
                if (!sws) return result;
                last_width = frame->width;
                last_height = frame->height;
                last_pix_fmt = static_cast<AVPixelFormat>(frame->format);
            }

            cv::Mat bgr(frame->height, frame->width, CV_8UC3);
            uint8_t* dst_data[4] = {bgr.data, nullptr, nullptr, nullptr};
            int dst_linesize[4] = {static_cast<int>(bgr.step[0]), 0, 0, 0};
            sws_scale(sws, frame->data, frame->linesize, 0, frame->height,
                      dst_data, dst_linesize);
            result = bgr.clone();
            last_pict_type = frame->pict_type;
        }
        if (pict_type_out && !result.empty()) {
            *pict_type_out = last_pict_type;
        }
        return result;
    }
};

VideoCodecDecoder::VideoCodecDecoder(Codec codec)
    : codec_(codec), impl_(std::make_unique<Impl>()) {
    impl_->codec = codec;
}

VideoCodecDecoder::~VideoCodecDecoder() = default;

void VideoCodecDecoder::reset() {
    if (impl_) impl_->reset();
}

cv::Mat VideoCodecDecoder::decodePacket(const uint8_t* data,
                                        size_t size,
                                        int* pict_type_out) {
    return impl_ ? impl_->decodePacket(data, size, pict_type_out) : cv::Mat{};
}

#else  // !MCALIB_HAS_FFMPEG

struct VideoCodecDecoder::Impl {};

VideoCodecDecoder::VideoCodecDecoder(Codec) : impl_(std::make_unique<Impl>()) {}

VideoCodecDecoder::~VideoCodecDecoder() = default;

void VideoCodecDecoder::reset() {}

cv::Mat VideoCodecDecoder::decodePacket(const uint8_t*,
                                        size_t,
                                        int* pict_type_out) {
    if (pict_type_out) *pict_type_out = 0;
    CVLog::Warning(
            "[VideoCodecDecoder] FFmpeg not available; cannot decode "
            "H.264/HEVC");
    return {};
}

#endif  // MCALIB_HAS_FFMPEG

bool VideoCodecDecoder::isVideoFormat(const std::string& format) {
    return format == "h264" || format == "hevc" || format == "h265" ||
           format == "H264" || format == "HEVC" || format == "H265";
}

VideoCodecDecoder::Codec VideoCodecDecoder::codecFromFormat(
        const std::string& format) {
    if (format == "hevc" || format == "h265" || format == "HEVC" ||
        format == "H265") {
        return Codec::HEVC;
    }
    return Codec::H264;
}

bool VideoCodecDecoder::looksLikeVideoBitstream(const uint8_t* data,
                                                size_t size) {
    if (data == nullptr || size < 4) return false;
    return data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 &&
           data[3] == 0x01;
}

bool VideoCodecDecoder::bufferContainsSyncFrame(Codec codec,
                                                const uint8_t* data,
                                                size_t size) {
    if (data == nullptr || size < 5) return false;

    size_t i = 0;
    while (i + 5 <= size) {
        if (data[i] == 0x00 && data[i + 1] == 0x00 &&
            (data[i + 2] == 0x01 ||
             (data[i + 2] == 0x00 && i + 6 <= size && data[i + 3] == 0x01))) {
            size_t hdr = i + 3;
            if (data[i + 2] == 0x00) hdr = i + 4;
            const uint8_t nal = data[hdr];
            if (codec == Codec::HEVC) {
                const int type = (nal >> 1) & 0x3F;
                if (type == 19 || type == 20 || type == 21) return true;
            } else {
                const int type = nal & 0x1F;
                if (type == 5) return true;
            }
            i = hdr + 1;
        } else {
            ++i;
        }
    }
    return false;
}

bool VideoCodecDecoder::bufferContainsSyncFrame(const std::string& format,
                                                const std::string& buffer) {
    if (buffer.empty()) return false;
    const Codec codec =
            isVideoFormat(format) ? codecFromFormat(format) : Codec::HEVC;
    return bufferContainsSyncFrame(
            codec, reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size());
}

constexpr size_t kMaxCachedVideoFramesPerTopic = 32;

struct VideoDecodeCache::Impl {
    struct TopicState {
        std::mutex mutex;
        std::unique_ptr<VideoCodecDecoder> decoder;
        VideoCodecDecoder::Codec codec = VideoCodecDecoder::Codec::H264;
        uint64_t decoded_upto_ns = 0;
        uint64_t last_fed_bag_ns = 0;
        bool begin_to_capture = false;
        std::map<int64_t, cv::Mat> frames;

        void resetDecoder() {
            decoded_upto_ns = 0;
            last_fed_bag_ns = 0;
            begin_to_capture = false;
            frames.clear();
            if (decoder) decoder->reset();
        }

        void trimFrameCache() {
            while (frames.size() > kMaxCachedVideoFramesPerTopic) {
                frames.erase(frames.begin());
            }
        }
    };

    std::mutex topics_mutex;
    std::map<std::string, std::unique_ptr<TopicState>> topics;
};

VideoDecodeCache::VideoDecodeCache() = default;

VideoDecodeCache::~VideoDecodeCache() = default;

void VideoDecodeCache::clear() {
    if (impl_) impl_->topics.clear();
}

bool VideoDecodeCache::decodeMessage(RosBagReader& reader,
                                     const BagMessage& msg,
                                     const std::string& format,
                                     cv::Mat& image,
                                     double& timestamp_sec) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    image.release();
    timestamp_sec = 0;

    std::string resolved_format = format;
    std::string image_buffer;
    if (!extractFormatAndBuffer(msg.data, resolved_format, image_buffer,
                                timestamp_sec)) {
        return false;
    }

    const bool video =
            VideoCodecDecoder::isVideoFormat(resolved_format) ||
            (resolved_format.empty() &&
             VideoCodecDecoder::looksLikeVideoBitstream(
                     reinterpret_cast<const uint8_t*>(image_buffer.data()),
                     image_buffer.size()));

    if (!video) {
        image = ProtoDecoder::decodeImageBuffer(image_buffer);
        return !image.empty();
    }

#if !defined(MCALIB_HAS_FFMPEG)
    CVLog::Warning("[VideoDecodeCache] video format '%s' requires FFmpeg",
                   resolved_format.c_str());
    return false;
#else
    Impl::TopicState* state_ptr = nullptr;
    {
        std::lock_guard<std::mutex> topics_lock(impl_->topics_mutex);
        auto& state = impl_->topics[msg.topic];
        if (!state) {
            state = std::make_unique<Impl::TopicState>();
        }
        state_ptr = state.get();
    }
    auto& state = *state_ptr;
    std::lock_guard<std::mutex> topic_lock(state.mutex);

    const VideoCodecDecoder::Codec codec =
            VideoCodecDecoder::codecFromFormat(resolved_format);
    if (!state.decoder || state.codec != codec) {
        state.codec = codec;
        state.decoder = std::make_unique<VideoCodecDecoder>(codec);
        state.resetDecoder();
    }

    const uint64_t target_bag_ns = msg.timestamp_ns;
    if (target_bag_ns > 0) {
        auto cached = state.frames.find(static_cast<int64_t>(target_bag_ns));
        if (cached != state.frames.end() && !cached->second.empty()) {
            image = cached->second.clone();
            return true;
        }
    }

    const bool seek_backward =
            target_bag_ns > 0 && target_bag_ns < state.decoded_upto_ns;
    if (seek_backward) {
        state.resetDecoder();
    }

    const uint64_t bag_begin = reader.getBeginTime();
    const uint64_t end_ns =
            target_bag_ns > 0 ? target_bag_ns : msg.timestamp_ns;

    uint64_t start_ns = bag_begin;
    if (state.last_fed_bag_ns > 0 && !seek_backward) {
        start_ns = state.last_fed_bag_ns + 1;
    } else if (end_ns > 0) {
        start_ns = findVideoSyncStartNs(reader, msg.topic, end_ns);
    }

    if (end_ns > 0 && end_ns < start_ns) {
        start_ns = bag_begin;
        state.resetDecoder();
        start_ns = findVideoSyncStartNs(reader, msg.topic, end_ns);
    }

    auto decode_range = [&](uint64_t range_start, uint64_t range_end,
                            bool reset_decoder) -> cv::Mat {
        if (reset_decoder) {
            state.resetDecoder();
        }

        std::map<int64_t, std::string> stamp_buffers;
        if (range_start <= range_end) {
            std::set<std::string> topic_filter{msg.topic};
            const auto prior_msgs = reader.readAllMessagesParallel(
                    topic_filter, nullptr, range_start, range_end);
            for (const auto& prior : prior_msgs) {
                if (prior.timestamp_ns <= state.last_fed_bag_ns) continue;
                std::string prior_format;
                std::string prior_buffer;
                double prior_ts = 0;
                if (!extractFormatAndBuffer(prior.data, prior_format,
                                            prior_buffer, prior_ts)) {
                    continue;
                }
                stamp_buffers[static_cast<int64_t>(prior.timestamp_ns)] =
                        std::move(prior_buffer);
            }
        }
        if (target_bag_ns > 0) {
            stamp_buffers[static_cast<int64_t>(target_bag_ns)] = image_buffer;
        } else if (stamp_buffers.empty()) {
            stamp_buffers[static_cast<int64_t>(msg.timestamp_ns)] =
                    image_buffer;
        }

        cv::Mat decoded_target;
        for (const auto& [stamp, buffer] : stamp_buffers) {
            if (static_cast<uint64_t>(stamp) <= state.last_fed_bag_ns) {
                continue;
            }

            int pict_type = 0;
            cv::Mat frame = state.decoder->decodePacket(
                    reinterpret_cast<const uint8_t*>(buffer.data()),
                    buffer.size(), &pict_type);
            state.last_fed_bag_ns = std::max(state.last_fed_bag_ns,
                                             static_cast<uint64_t>(stamp));

            if (frame.empty()) {
                continue;
            }
#if defined(MCALIB_HAS_FFMPEG)
            if (pict_type == AV_PICTURE_TYPE_I) {
                state.begin_to_capture = true;
            }
#endif
            if (!state.begin_to_capture || isFrameUniform(frame)) {
                continue;
            }

            state.frames[stamp] = frame.clone();
            if (target_bag_ns > 0 &&
                static_cast<uint64_t>(stamp) == target_bag_ns) {
                decoded_target = frame;
            }
        }

        state.decoded_upto_ns = std::max(state.decoded_upto_ns, range_end);
        state.trimFrameCache();
        return decoded_target;
    };

    cv::Mat target_frame = decode_range(start_ns, end_ns, false);
    if (target_frame.empty() && start_ns > bag_begin && end_ns > 0) {
        target_frame = decode_range(bag_begin, end_ns, true);
    }

    if (!target_frame.empty()) {
        image = target_frame.clone();
        return true;
    }
    CVLog::Print("[VideoDecodeCache] decode failed topic=%s stamp=%llu",
                 msg.topic.c_str(),
                 static_cast<unsigned long long>(target_bag_ns));
    return false;
#endif
}

}  // namespace mcalib
