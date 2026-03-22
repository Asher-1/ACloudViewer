// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FFmpegVideoEncoder.hpp"

#ifndef HEADLESS
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#endif

#define QQ(rat) (rat.num / (double)rat.den)

// Disable ffmpeg deprecation warning.
#pragma warning(disable : 4996)

namespace sibr {

bool FFVideoEncoder::ffmpegInitDone = false;

FFVideoEncoder::FFVideoEncoder(const std::string &_filepath,
                               double _fps,
                               const sibr::Vector2i &size,
                               bool forceResize)
    : filepath(_filepath), fps(_fps), _forceResize(forceResize) {
#ifndef HEADLESS
    /** Init FFMPEG, registering available codec plugins. */
    if (!ffmpegInitDone) {
        SIBR_LOG << "[FFMPEG] Registering all." << std::endl;
        // av_register_all() is deprecated in FFmpeg 4.0+
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
        av_register_all();
#endif
        ffmpegInitDone = true;
    }

    sibr::Vector2i sizeFix = size;
    bool hadToFix = false;
    if (sizeFix[0] % 2 != 0) {
        sizeFix[0] -= 1;
        hadToFix = true;
    }
    if (sizeFix[1] % 2 != 0) {
        sizeFix[1] -= 1;
        hadToFix = true;
    }
    if (hadToFix) {
        SIBR_WRG << "Non-even video dimensions, resized to " << sizeFix[0]
                 << "x" << sizeFix[1] << "." << std::endl;
        _forceResize = true;
    }

    init(sizeFix);
#endif
}

bool FFVideoEncoder::isFine() const { return initWasFine; }

void FFVideoEncoder::close() {
#ifndef HEADLESS
    if (av_write_trailer(pFormatCtx) < 0) {
        SIBR_WRG << "[FFMPEG] Can not av_write_trailer " << std::endl;
    }

    if (video_st) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 48, 101)
        avcodec_close(video_st->codec);
#else
        if (pCodecCtx) {
            avcodec_free_context(&pCodecCtx);
        }
#endif
        av_free(frameYUV);
    }
    avio_close(pFormatCtx->pb);
    avformat_free_context(pFormatCtx);

    needFree = false;
#endif
}

FFVideoEncoder::~FFVideoEncoder() {
    if (needFree) {
        close();
    }
}

void FFVideoEncoder::init(const sibr::Vector2i &size) {
#ifndef HEADLESS
    w = size[0];
    h = size[1];

    auto out_file = filepath.c_str();

    pFormatCtx = avformat_alloc_context();

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(59, 0, 100)
    fmt = const_cast<AVOutputFormat*>(av_guess_format(NULL, out_file, NULL));
#else
    fmt = const_cast<AVOutputFormat*>(av_guess_format(NULL, out_file, NULL));
#endif
    pFormatCtx->oformat = fmt;

    const bool isH264 = pFormatCtx->oformat->video_codec == AV_CODEC_ID_H264;
    if (isH264) {
        SIBR_LOG << "[FFMPEG] Found H264 codec." << std::endl;
    } else {
        SIBR_LOG << "[FFMPEG] Found codec with ID "
                 << pFormatCtx->oformat->video_codec << " (not H264)."
                 << std::endl;
    }

    if (avio_open(&pFormatCtx->pb, out_file, AVIO_FLAG_READ_WRITE) < 0) {
        SIBR_WRG << "[FFMPEG] Could not open file " << filepath << std::endl;
        return;
    }

    pCodec = const_cast<AVCodec*>(avcodec_find_encoder(pFormatCtx->oformat->video_codec));
    if (!pCodec) {
        SIBR_WRG << "[FFMPEG] Could not find codec." << std::endl;
        return;
    }

    video_st = avformat_new_stream(pFormatCtx, pCodec);

    if (video_st == NULL) {
        SIBR_WRG << "[FFMPEG] Could not create stream." << std::endl;
        return;
    }

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 48, 101)
    pCodecCtx = video_st->codec;
#else
    pCodecCtx = avcodec_alloc_context3(pCodec);
    if (!pCodecCtx) {
        SIBR_WRG << "[FFMPEG] Could not allocate codec context." << std::endl;
        return;
    }
#endif
    pCodecCtx->codec_id = fmt->video_codec;
    pCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
    pCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    pCodecCtx->width = w;
    pCodecCtx->height = h;
    pCodecCtx->gop_size = 10;
    pCodecCtx->time_base.num = 1;
    pCodecCtx->time_base.den = (int)std::round(fps);

    // Required for the header to be well-formed and compatible with
    // Powerpoint/MediaPlayer/...
    if (pFormatCtx->oformat->flags & AVFMT_GLOBALHEADER) {
        pCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // H.264 specific options.
    AVDictionary *param = 0;
    if (pCodecCtx->codec_id == AV_CODEC_ID_H264) {
        av_dict_set(&param, "preset", "slow", 0);
        av_dict_set(&param, "tune", "zerolatency", 0);
    }

    av_dump_format(pFormatCtx, 0, out_file, 1);

    int res = avcodec_open2(pCodecCtx, pCodec, &param);
    if (res < 0) {
        SIBR_WRG << "[FFMPEG] Failed to open encoder, error: " << res
                 << std::endl;
        return;
    }
    
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 48, 101)
    // In newer FFmpeg, copy codec parameters from context to stream
    res = avcodec_parameters_from_context(video_st->codecpar, pCodecCtx);
    if (res < 0) {
        SIBR_WRG << "[FFMPEG] Failed to copy codec parameters, error: " << res << std::endl;
        return;
    }
#endif
    
    // Write the file header.
    res = avformat_write_header(pFormatCtx, NULL);
    if (res < 0) {
        SIBR_WRG << "[FFMPEG] Failed to write header, error: " << res << std::endl;
        return;
    }

    // Prepare the scratch frame.
    frameYUV = av_frame_alloc();
    frameYUV->format = (int)pCodecCtx->pix_fmt;
    frameYUV->width = w;
    frameYUV->height = h;
    frameYUV->linesize[0] = w;
    frameYUV->linesize[1] = w / 2;
    frameYUV->linesize[2] = w / 2;

    yuSize[0] = frameYUV->linesize[0] * h;
    yuSize[1] = frameYUV->linesize[1] * h / 2;

    pkt = av_packet_alloc();

    initWasFine = true;
    needFree = true;
#endif
}

bool FFVideoEncoder::operator<<(cv::Mat frame) {
#ifndef HEADLESS
    if (!video_st) {
        return false;
    }
    cv::Mat local;
    if (frame.cols != w || frame.rows != h) {
        if (_forceResize) {
            cv::resize(frame, local, cv::Size(w, h));
        } else {
            SIBR_WRG << "[FFMPEG] Frame doesn't have the same dimensions as "
                        "the video."
                     << std::endl;
            return false;
        }
    } else {
        local = frame;
    }

    cv::cvtColor(local, cvFrameYUV, cv::COLOR_BGR2YUV_I420);
    frameYUV->data[0] = cvFrameYUV.data;
    frameYUV->data[1] = frameYUV->data[0] + yuSize[0];
    frameYUV->data[2] = frameYUV->data[1] + yuSize[1];

    // frameYUV->pts = (1.0 / std::round(fps)) *frameCount * 90;
    frameYUV->pts = (int)(frameCount * (video_st->time_base.den) /
                          ((video_st->time_base.num) * std::round(fps)));
    ++frameCount;

    return encode(frameYUV);
#else
    SIBR_ERR << "Not supported in headless" << std::endl;
    return false;
#endif
}

bool FFVideoEncoder::operator<<(const sibr::ImageRGB &frame) {
    return (*this) << (frame.toOpenCVBGR());
}

#ifndef HEADLESS
bool FFVideoEncoder::encode(AVFrame *frame) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 48, 101)
    // Old API (FFmpeg < 3.1)
    int got_picture = 0;
    int ret = avcodec_encode_video2(pCodecCtx, pkt, frameYUV, &got_picture);
    if (ret < 0) {
        SIBR_WRG << "[FFMPEG] Failed to encode frame." << std::endl;
        return false;
    }
    if (got_picture == 1) {
        pkt->stream_index = video_st->index;
        ret = av_write_frame(pFormatCtx, pkt);
        av_packet_unref(pkt);
    }
#else
    // New API (FFmpeg >= 3.1)
    int ret = avcodec_send_frame(pCodecCtx, frameYUV);
    if (ret < 0) {
        SIBR_WRG << "[FFMPEG] Failed to send frame for encoding." << std::endl;
        return false;
    }
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(pCodecCtx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            SIBR_WRG << "[FFMPEG] Error during encoding." << std::endl;
            return false;
        }
        
        pkt->stream_index = video_st->index;
        ret = av_write_frame(pFormatCtx, pkt);
        av_packet_unref(pkt);
        
        if (ret < 0) {
            SIBR_WRG << "[FFMPEG] Error writing frame." << std::endl;
            return false;
        }
    }
#endif

    return true;
}
#endif

}  // namespace sibr
