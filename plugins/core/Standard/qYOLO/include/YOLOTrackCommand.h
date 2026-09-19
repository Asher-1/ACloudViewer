// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless multi-object tracking command (-YOLO_TRACK), mirroring the
// upstream ultralytics-ggml "yolo-cli track" subcommand: a frame/video
// source runs through the AICore YOLO C API and the six official tracker
// modes (the same qyolo::track port the Live tab uses), producing a
// tracks JSONL file. Supports the detect / segment / pose / obb tasks,
// exactly like the upstream track mode. Recovery rows (dets_del) follow
// the upstream want_recovered semantics (tracktrack + box tasks).

#pragma once

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QObject>
#include <QStringList>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/image_view.h>
#include <aicore/yolo_capi.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "tracking/tracker.hpp"

#ifdef QYOLO_WITH_OPENCV
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif
#endif  // AICore_ENABLED

// Command keyword (the -SILENT CLI token users type after the binary).
#define COMMAND_YOLO_TRACK "YOLO_TRACK"

struct CommandYoloTrack : public ccCommandLineInterface::Command {
    CommandYoloTrack()
        : ccCommandLineInterface::Command(QObject::tr("YOLO tracking"),
                                          COMMAND_YOLO_TRACK) {}

#ifdef AICore_ENABLED

    static QStringList listFrameFiles(const QDir& dir) {
        QStringList files;
        for (const QString& pattern :
             {QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
              QStringLiteral("*.png"), QStringLiteral("*.bmp"),
              QStringLiteral("*.tif"), QStringLiteral("*.tiff")}) {
            files << dir.entryList(QStringList{pattern}, QDir::Files,
                                   QDir::Name);
        }
        files.removeDuplicates();
        QStringList absolute;
        absolute.reserve(files.size());
        for (const QString& f : files) absolute << dir.absoluteFilePath(f);
        return absolute;
    }

    static qyolo::track::TrackDet aabbToTrack(float x1,
                                              float y1,
                                              float x2,
                                              float y2,
                                              float score,
                                              int cls,
                                              int idx) {
        qyolo::track::TrackDet t;
        t.cx = (x1 + x2) * 0.5f;
        t.cy = (y1 + y2) * 0.5f;
        t.w = x2 - x1;
        t.h = y2 - y1;
        t.angle = -10.0f;  // axis-aligned sentinel (same as upstream)
        t.score = score;
        t.class_id = cls;
        t.idx = idx;
        return t;
    }

    // Same flat schema as the upstream --tracks-json entries: coordinates
    // in original-image pixels, center format, angle/id optional.
    static void writeTrackEntry(FILE* f,
                                const qyolo::track::TrackDet& t,
                                int track_id) {
        std::fprintf(f, "{\"cx\":%.6f,\"cy\":%.6f,\"w\":%.6f,\"h\":%.6f", t.cx,
                     t.cy, t.w, t.h);
        if (t.angled()) std::fprintf(f, ",\"angle\":%.6f", t.angle);
        if (track_id >= 0) std::fprintf(f, ",\"id\":%d", track_id);
        std::fprintf(f, ",\"score\":%.6f,\"cls\":%d,\"idx\":%d", t.score,
                     t.class_id, t.idx);
        std::fputs("}", f);
    }

    bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[YOLO_TRACK]");

        QString modelPath, videoPath, framesDir, tracksJson;
        QString trackerType = QStringLiteral("tracktrack");
        QString gmcMethod = QStringLiteral("sparseOptFlow");
        QString device = QStringLiteral("auto");
        float conf = 0.1f;  // upstream track-mode default (ByteTrack family)
        float iou = 0.7f;
        int maxDet = 0;
        int threads = 0;
        // Official with_reid (model="auto" detector-feature path): 0|1.
        bool reid = false;

        auto take = [&cmd]() -> QString {
            return cmd.arguments().isEmpty() ? QString()
                                             : cmd.arguments().takeFirst();
        };
        while (!cmd.arguments().isEmpty()) {
            const QString key = cmd.arguments().takeFirst().toUpper();
            if (key == QStringLiteral("MODEL")) {
                modelPath = take();
            } else if (key == QStringLiteral("VIDEO")) {
                videoPath = take();
            } else if (key == QStringLiteral("FRAMES_DIR")) {
                framesDir = take();
            } else if (key == QStringLiteral("TRACKER")) {
                trackerType = take();
            } else if (key == QStringLiteral("GMC")) {
                gmcMethod = take();
            } else if (key == QStringLiteral("CONF")) {
                conf = take().toFloat();
            } else if (key == QStringLiteral("IOU")) {
                iou = take().toFloat();
            } else if (key == QStringLiteral("MAX_DET")) {
                maxDet = take().toInt();
            } else if (key == QStringLiteral("DEVICE")) {
                device = take();
            } else if (key == QStringLiteral("THREADS")) {
                threads = take().toInt();
            } else if (key == QStringLiteral("REID")) {
                reid = take().toInt() != 0;
            } else if (key == QStringLiteral("TRACKS_JSON")) {
                tracksJson = take();
            } else {
                return cmd.error(QObject::tr("Unknown argument after -%1: %2")
                                         .arg(COMMAND_YOLO_TRACK, key));
            }
        }

        if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
            return cmd.error(QObject::tr(
                    "MODEL <model.gguf> is required and must exist"));
        }
        if (videoPath.isEmpty() == framesDir.isEmpty()) {
            return cmd.error(QObject::tr(
                    "Provide exactly one source: VIDEO <file> or FRAMES_DIR "
                    "<dir>"));
        }
        if (conf <= 0.0f || conf >= 1.0f || iou < 0.0f || iou > 1.0f) {
            return cmd.error(
                    QObject::tr("CONF must be in (0,1) and IOU in [0,1]"));
        }

        // Tracker (official YAML defaults; the OpenCV-only GMC methods on
        // OpenCV-less builds are rejected, mirroring the upstream
        // create_tracker contract. with_reid engages the official
        // model="auto" detector-feature ReID path).
        qyolo::track::TrackConfig tcfg;
        if (!qyolo::track::default_tracker_config(trackerType.toStdString(),
                                                  tcfg)) {
            return cmd.error(
                    QObject::tr("Unknown tracker type '%1'").arg(trackerType));
        }
        tcfg.tracker_type = trackerType.toStdString();
        tcfg.gmc_method = gmcMethod.toStdString();
        tcfg.with_reid = reid;
        auto tracker = qyolo::track::create_tracker(tcfg);
        if (!tracker) {
            return cmd.error(QObject::tr("Tracker rejected: type '%1' with "
                                         "GMC '%2'")
                                     .arg(trackerType, gmcMethod));
        }
        // Upstream want_recovered: loose-NMS recovery rows exist only for
        // the tracktrack tracker on the box tasks.
        const bool wantRecovery = trackerType == QStringLiteral("tracktrack");

        // Model context (headless: no UI, direct C API).
        aicore_yolo_options* opts = aicore_yolo_options_new();
        if (opts == nullptr) {
            return cmd.error(QObject::tr("Out of memory for YOLO options"));
        }
        aicore_yolo_options_set_device(opts, device.toUtf8().constData());
        aicore_yolo_options_set_threads(opts, threads);
        aicore_yolo_ctx* ctx =
                aicore_yolo_load_opts(modelPath.toUtf8().constData(), opts);
        aicore_yolo_options_free(opts);
        if (ctx == nullptr || !aicore_yolo_is_ready(ctx)) {
            const char* message =
                    ctx != nullptr ? aicore_yolo_last_error(ctx) : nullptr;
            const bool freed = ctx != nullptr;
            if (freed) aicore_yolo_free(ctx);
            return cmd.error(
                    QObject::tr("Model load failed: %1")
                            .arg(message != nullptr
                                         ? QString::fromUtf8(message)
                                         : QStringLiteral("unknown error")));
        }
        const QString task = QString::fromUtf8(aicore_yolo_context_task(ctx));
        if (task != QStringLiteral("detect") &&
            task != QStringLiteral("segment") &&
            task != QStringLiteral("pose") && task != QStringLiteral("obb")) {
            aicore_yolo_free(ctx);
            return cmd.error(
                    QObject::tr("task '%1' doesn't support mode=track, valid "
                                "tasks are detect, segment, pose, obb")
                            .arg(task));
        }
        aicore_yolo_set_detect_thresholds(
                ctx, conf, iou, maxDet > 0 ? static_cast<uint32_t>(maxDet) : 0);
        aicore_yolo_set_track_recovery(
                ctx, (wantRecovery && (task == QStringLiteral("detect") ||
                                       task == QStringLiteral("obb")))
                             ? 1
                             : 0);

        FILE* jf = nullptr;
        if (!tracksJson.isEmpty()) {
            jf = std::fopen(tracksJson.toUtf8().constData(), "wb");
            if (jf == nullptr) {
                aicore_yolo_free(ctx);
                return cmd.error(
                        QObject::tr("Failed to write %1").arg(tracksJson));
            }
        }

#ifdef QYOLO_WITH_OPENCV
        std::unique_ptr<cv::VideoCapture> videoCapture;
        if (!videoPath.isEmpty()) {
            videoCapture = std::make_unique<cv::VideoCapture>(
                    videoPath.toUtf8().constData());
            if (!videoCapture->isOpened()) {
                videoCapture.reset();
                if (jf != nullptr) std::fclose(jf);
                aicore_yolo_free(ctx);
                return cmd.error(
                        QObject::tr("Failed to open video %1").arg(videoPath));
            }
            cmd.print(QObject::tr("Video: %1").arg(videoPath));
        }
#else
        if (!videoPath.isEmpty()) {
            aicore_yolo_free(ctx);
            return cmd.error(QObject::tr(
                    "VIDEO sources need the plugin's OpenCV build; use "
                    "FRAMES_DIR instead"));
        }
#endif

        if (!framesDir.isEmpty()) {
            const QStringList files = listFrameFiles(QDir(framesDir));
            if (files.isEmpty()) {
                if (jf != nullptr) std::fclose(jf);
                aicore_yolo_free(ctx);
                return cmd.error(
                        QObject::tr("No frames found in %1").arg(framesDir));
            }
            cmd.print(QObject::tr("Frames: %1 (from %2)")
                              .arg(files.size())
                              .arg(framesDir));
            frameFiles = files;
        }

        bool ok = true;
        size_t frameIndex = 0;
        long totalTracks = 0;
        int firstW = 0, firstH = 0;
        while ((ok = decodeNextFrame())) {
            // Tracking (and the cv2 LK/ECC GMC kernels exactly like the
            // upstream Python) requires consecutive frames of one size —
            // fail with a clear message instead of a deep OpenCV assert.
            if (firstW == 0) {
                firstW = width;
                firstH = height;
            } else if (width != firstW || height != firstH) {
                if (jf != nullptr) std::fclose(jf);
                aicore_yolo_free(ctx);
                return cmd.error(
                        QObject::tr("frame %1 is %2x%3; tracking requires "
                                    "every frame to share the first frame's "
                                    "size (%4x%5)")
                                .arg(static_cast<qulonglong>(frameIndex))
                                .arg(width)
                                .arg(height)
                                .arg(firstW)
                                .arg(firstH));
            }
            // rgb now holds the tightly-packed RGB8 frame every stage
            // borrows (AICore view + tracker GMC input).
            const aicore_image_view view{rgb.data(), width, height,
                                         static_cast<size_t>(width) * 3,
                                         AICORE_IMAGE_RGB8};
            std::vector<qyolo::track::TrackDet> frameDets;
            if (task == QStringLiteral("detect")) {
                if (aicore_yolo_detect_image(ctx, &view) == 0) {
                    const int n = aicore_yolo_detection_count(ctx);
                    for (int i = 0; i < n; ++i) {
                        const aicore_yolo_detection d =
                                aicore_yolo_detection_at(ctx, i);
                        frameDets.push_back(aabbToTrack(d.x1, d.y1, d.x2, d.y2,
                                                        d.score, d.class_id,
                                                        i));
                    }
                }
            } else if (task == QStringLiteral("segment")) {
                aicore_yolo_segment_result* seg =
                        aicore_yolo_seg_image(ctx, &view);
                if (seg != nullptr) {
                    const int n = aicore_yolo_seg_det_count(seg);
                    for (int i = 0; i < n; ++i) {
                        const aicore_yolo_detection d =
                                aicore_yolo_seg_det_at(seg, i);
                        frameDets.push_back(aabbToTrack(d.x1, d.y1, d.x2, d.y2,
                                                        d.score, d.class_id,
                                                        i));
                    }
                    aicore_yolo_seg_result_free(seg);
                }
            } else if (task == QStringLiteral("pose")) {
                aicore_yolo_pose_result* pose =
                        aicore_yolo_pose_image(ctx, &view);
                if (pose != nullptr) {
                    const int n = aicore_yolo_pose_det_count(pose);
                    for (int i = 0; i < n; ++i) {
                        const aicore_yolo_detection d =
                                aicore_yolo_pose_det_at(pose, i);
                        frameDets.push_back(aabbToTrack(d.x1, d.y1, d.x2, d.y2,
                                                        d.score, d.class_id,
                                                        i));
                    }
                    aicore_yolo_pose_result_free(pose);
                }
            } else {  // obb
                aicore_yolo_obb_result* obb = aicore_yolo_obb_image(ctx, &view);
                if (obb != nullptr) {
                    const int n = aicore_yolo_obb_count(obb);
                    for (int i = 0; i < n; ++i) {
                        const aicore_yolo_obb_box b =
                                aicore_yolo_obb_at(obb, i);
                        qyolo::track::TrackDet t;
                        t.cx = b.cx;
                        t.cy = b.cy;
                        t.w = b.w;
                        t.h = b.h;
                        t.angle = b.angle;
                        t.score = b.score;
                        t.class_id = b.class_id;
                        t.idx = i;
                        frameDets.push_back(t);
                    }
                    aicore_yolo_obb_result_free(obb);
                }
            }

            qyolo::track::FrameInput input;
            input.dets = frameDets;
            const qyolo::track::RgbFrame gmcFrame{width, height, rgb.data()};
            input.frame = gmcFrame.rgb != nullptr ? &gmcFrame : nullptr;
            if (wantRecovery && (task == QStringLiteral("detect") ||
                                 task == QStringLiteral("obb"))) {
                const int rn = aicore_yolo_recovery_count(ctx);
                input.dets_del.reserve(rn > 0 ? static_cast<size_t>(rn) : 0);
                for (int i = 0; i < rn; ++i) {
                    const aicore_yolo_detection r =
                            aicore_yolo_recovery_at(ctx, i);
                    input.dets_del.push_back(aabbToTrack(
                            r.x1, r.y1, r.x2, r.y2, r.score, r.class_id, -1));
                }
            }
            const std::vector<qyolo::track::TrackedBox> tracks =
                    tracker->update(input);
            totalTracks += static_cast<long>(tracks.size());

            cmd.print(QObject::tr("frame %1: %2 track(s)")
                              .arg(static_cast<qulonglong>(frameIndex))
                              .arg(tracks.size()));
            if (jf != nullptr) {
                std::fprintf(jf, "{\"frame\":%zu,\"detections\":[", frameIndex);
                for (size_t i = 0; i < input.dets.size(); ++i) {
                    if (i != 0) std::fputs(",", jf);
                    writeTrackEntry(jf, input.dets[i], -1);
                }
                std::fputs("],\"detections_del\":[", jf);
                for (size_t i = 0; i < input.dets_del.size(); ++i) {
                    if (i != 0) std::fputs(",", jf);
                    writeTrackEntry(jf, input.dets_del[i], -1);
                }
                std::fputs("],\"tracks\":[", jf);
                for (size_t i = 0; i < tracks.size(); ++i) {
                    if (i != 0) std::fputs(",", jf);
                    writeTrackEntry(jf, tracks[i].det, tracks[i].track_id);
                }
                std::fputs("]}\n", jf);
            }
            ++frameIndex;
        }
        (void)ok;

        if (jf != nullptr) std::fclose(jf);
        aicore_yolo_free(ctx);
        cmd.print(QObject::tr("[YOLO_TRACK] done: %1 frame(s), %2 track "
                              "assignment(s)%3")
                          .arg(static_cast<qulonglong>(frameIndex))
                          .arg(totalTracks)
                          .arg(tracksJson.isEmpty()
                                       ? QString()
                                       : QStringLiteral(", tracks JSONL: %1")
                                                 .arg(tracksJson)));
        return true;
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_YOLO_TRACK));
    }
#endif  // AICore_ENABLED

private:
#ifdef AICore_ENABLED
    QStringList frameFiles;
    std::vector<uint8_t> rgb;
    int width = 0;
    int height = 0;
    size_t nextFileIndex = 0;

    /** Decodes the next frame into `rgb` (tightly-packed RGB8). Returns
     *  false when the source is exhausted. FRAMES_DIR decodes through Qt
     *  (no OpenCV required, unreadable files are skipped); VIDEO reads
     *  through cv::VideoCapture (RGB conversion keeps the single compact
     *  layout every stage borrows). */
    bool decodeNextFrame() {
#ifdef QYOLO_WITH_OPENCV
        if (videoCapture != nullptr) {
            cv::Mat bgr;
            if (!videoCapture->read(bgr) || bgr.empty()) return false;
            width = bgr.cols;
            height = bgr.rows;
            cv::Mat continuous = bgr.isContinuous() ? bgr : bgr.clone();
            if (continuous.type() != CV_8UC3) {
                cv::Mat converted;
                cv::cvtColor(continuous, converted, cv::COLOR_BGR2RGB);
                continuous = converted;
            }
            rgb.assign(continuous.datastart, continuous.dataend);
            return true;
        }
#endif
        while (nextFileIndex < static_cast<size_t>(frameFiles.size())) {
            QImage frame(frameFiles[static_cast<int>(nextFileIndex)]);
            ++nextFileIndex;
            if (frame.isNull()) continue;  // skip unreadable files
            if (frame.format() != QImage::Format_RGB888 ||
                frame.bytesPerLine() != frame.width() * 3) {
                frame = frame.convertToFormat(QImage::Format_RGB888);
            }
            width = frame.width();
            height = frame.height();
            const int rowBytes = width * 3;
            rgb.resize(static_cast<size_t>(width) *
                       static_cast<size_t>(height) * 3);
            for (int y = 0; y < height; ++y) {
                std::memcpy(rgb.data() + static_cast<size_t>(y) * rowBytes,
                            frame.constScanLine(y),
                            static_cast<size_t>(rowBytes));
            }
            return true;
        }
        return false;
    }

#ifdef QYOLO_WITH_OPENCV
    std::unique_ptr<cv::VideoCapture> videoCapture;
#endif
#endif  // AICore_ENABLED
};
