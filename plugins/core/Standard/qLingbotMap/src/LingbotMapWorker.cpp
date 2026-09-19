// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LingbotMapWorker.h"

#include <QtCompat.h>

// Qt (including Q_OBJECT meta-object compilation, which happens with
// AICore_ENABLED undefined for the moc input as well).
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>
#include <QTransform>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <cmath>

#include "OpenCVFrameSource.h"
#include "VideoPlaybackWidget.h"
#endif

// AICore public C ABI.
#include "aicore/lingbot_capi.h"

// Official windowed long-sequence math (pure, Qt-free).
#include "LingbotWindowStitcher.h"

namespace {

// Image extensions accepted for the folder input (upstream --image_ext;
// comma/space separated, leading dots optional, matching is
// case-insensitive so upstream's ".jpg,.png,.JPG" style lists all hit).
QStringList parseImageExtensions(const QString& spec) {
    QStringList exts;
    const QStringList parts =
            spec.split(QLatin1Char(','), QtCompat::SkipEmptyParts);
    for (const QString& part : parts) {
        QString ext = part.trimmed();
        while (ext.startsWith(QLatin1Char('.'))) ext.remove(0, 1);
        if (!ext.isEmpty()) exts.append(ext.toLower());
    }
    if (exts.isEmpty()) {
        // Defensive fallback = the full default set.
        exts << QStringLiteral("jpg") << QStringLiteral("png")
             << QStringLiteral("jpeg") << QStringLiteral("bmp")
             << QStringLiteral("tif") << QStringLiteral("tiff");
    }
    return exts;
}

// Ordered image sequence of the folder (official demo folders are
// zero-padded frame names; a natural sort keeps them in stream order).
// Upstream ggml_demo semantics: --frames truncates first, then --stride
// samples every Nth frame of the truncated list.
QStringList listFrameFiles(const QString& folder,
                           const QStringList& exts,
                           int maxFrames,
                           int frameStride) {
    QDir dir(folder);
    QStringList files;
    const QStringList entries = dir.entryList(QDir::Files, QDir::Name);
    files.reserve(entries.size());
    for (const QString& entry : entries) {
        if (exts.contains(QFileInfo(entry).suffix().toLower())) {
            files.append(entry);
        }
    }
    if (maxFrames > 0 && files.size() > maxFrames) {
        files = files.mid(0, maxFrames);
    }
    if (frameStride > 1) {
        QStringList sampled;
        sampled.reserve(files.size() / frameStride + 1);
        for (int i = 0; i < files.size(); i += frameStride) {
            sampled.append(files[i]);
        }
        files = sampled;
    }
    QStringList full;
    full.reserve(files.size());
    for (const QString& f : files) {
        full.append(dir.filePath(f));
    }
    return full;
}

// Video decode via the shared video_base infrastructure (OpenCVFrameSource:
// cv::VideoCapture with best-effort hardware decode). Mirrors the upstream
// extract_video_frames: sample at --fps with interval = round(src_fps / fps)
// (src_fps falls back to 30 when the container does not report it); frames
// are converted BGR→RGB by VideoPlaybackWidget::cvMatToQImage and named
// 000000/000001/… like the upstream temp JPEG stems.
#ifdef HAS_OPENCV_FACE_CAPTURE
bool decodeVideoFrames(const QString& videoPath,
                       int fps,
                       std::vector<QImage>& frames,
                       QStringList& names,
                       int* totalFrames,
                       int* interval,
                       QString* error) {
    OpenCVFrameSource source;
    if (!source.openVideo(videoPath.toStdString())) {
        *error = QStringLiteral("cannot open video: %1").arg(videoPath);
        return false;
    }
    double srcFps = source.fps();
    if (srcFps <= 0.0 || !std::isfinite(srcFps)) srcFps = 30.0;
    *interval = std::max(
            1, static_cast<int>(std::lround(srcFps / std::max(1, fps))));
    *totalFrames = static_cast<int>(source.frameCount());
    while (true) {
        cv::Mat bgr;
        int64_t index = 0;
        const auto result = source.read(bgr, &index);
        if (result == IFrameSource::ReadResult::Eof) break;
        if (result != IFrameSource::ReadResult::Ok || bgr.empty()) continue;
        if (static_cast<int>(index) % *interval != 0) continue;
        frames.push_back(VideoPlaybackWidget::cvMatToQImage(bgr));
        names.append(QStringLiteral("%1").arg(names.size(), 6, 10,
                                              QLatin1Char('0')));
    }
    source.release();
    return true;
}
#endif  // HAS_OPENCV_FACE_CAPTURE

/** Wrap a QImage (RGB888) as a borrowed, stride-aware AICore image view. */
aicore_image_view viewFromImage(const QImage& img) {
    aicore_image_view view{};
    view.data = const_cast<uint8_t*>(img.constBits());
    view.width = img.width();
    view.height = img.height();
    view.row_stride_bytes = img.bytesPerLine();
    view.format = AICORE_IMAGE_RGB8;
    return view;
}

/** Engine scale-pass profile (official defaults; the plugin never overrides
 *  the KV profile, so this is the scale-frame count used by window planning).
 */
constexpr int kOfficialScaleFrames = 8;

/** Compact stride-subsampled live preview of one engine frame (the same
 *  confidence-filtered geometry the official StreamingViewer renders). */
LingbotFramePreview buildPreview(const aicore_lingbot_result* r,
                                 int globalIndex,
                                 int windowIndex,
                                 int windowCount,
                                 const QImage& rgb,
                                 int stride,
                                 float confThreshold,
                                 const unsigned char* skyKeep) {
    LingbotFramePreview preview;
    preview.globalIndex = globalIndex;
    preview.windowIndex = windowIndex;
    preview.windowCount = windowCount;
    preview.width = r->width;
    preview.height = r->height;
    preview.c2w.resize(16);
    preview.intrinsics.resize(4);
    std::memcpy(preview.c2w.data(), r->c2w, 16 * sizeof(float));
    std::memcpy(preview.intrinsics.data(), r->intrinsics, 4 * sizeof(float));
    const float fx = r->intrinsics[0];
    const float fy = r->intrinsics[1];
    const float cx = r->intrinsics[2];
    const float cy = r->intrinsics[3];
    const bool finitePose =
            std::isfinite(fx) && std::isfinite(fy) && fx > 0.f && fy > 0.f &&
            std::isfinite(r->c2w[0]) && std::isfinite(r->c2w[3]) &&
            std::isfinite(r->c2w[7]) && std::isfinite(r->c2w[11]);
    const int step = std::max(1, stride);
    for (int y = 0; y < r->height; y += step) {
        for (int x = 0; x < r->width; x += step) {
            const size_t i = static_cast<size_t>(y) * r->width + x;
            const float d = r->depth[i];
            if (!std::isfinite(d) || d <= 0.f) continue;
            if (!finitePose) continue;
            if (r->depth_conf[i] < confThreshold) continue;
            if (skyKeep && skyKeep[i] == 0) continue;
            // OpenCV camera frame (x right, y down, z forward), then c2w.
            const float X = (x - cx) / fx * d;
            const float Y = -(y - cy) / fy * d;
            const float Z = d;
            const float* R = r->c2w;
            preview.points.append(R[0] * X + R[1] * Y + R[2] * Z + R[3]);
            preview.points.append(R[4] * X + R[5] * Y + R[6] * Z + R[7]);
            preview.points.append(R[8] * X + R[9] * Y + R[10] * Z + R[11]);
            if (!rgb.isNull()) {
                const QRgb c = rgb.pixel(x, y);
                preview.colors.append(static_cast<uint8_t>(qRed(c)));
                preview.colors.append(static_cast<uint8_t>(qGreen(c)));
                preview.colors.append(static_cast<uint8_t>(qBlue(c)));
            } else {
                preview.colors.append(200);
                preview.colors.append(200);
                preview.colors.append(200);
            }
        }
    }
    // A frame whose pose/depth diverged renders as a frustum only (official
    // StreamingViewer degenerate-frame semantics).
    preview.degenerate = !finitePose;
    return preview;
}

/** Fill one LingbotFrameResult from an engine result (all buffers copied). */
void fillFrameResult(LingbotFrameResult& out,
                     const aicore_lingbot_result* r,
                     int globalIndex,
                     int windowIndex) {
    out.width = r->width;
    out.height = r->height;
    out.globalIndex = globalIndex;
    out.windowIndex = windowIndex;
    const size_t n = static_cast<size_t>(r->width) * r->height;
    out.depth.resize(static_cast<int>(n));
    out.depthConf.resize(static_cast<int>(n));
    out.c2w.resize(16);
    out.intrinsics.resize(4);
    out.poseEnc.resize(lingbot_stitch::kPoseEncSize);
    std::memcpy(out.depth.data(), r->depth, n * sizeof(float));
    std::memcpy(out.depthConf.data(), r->depth_conf, n * sizeof(float));
    std::memcpy(out.c2w.data(), r->c2w, 16 * sizeof(float));
    std::memcpy(out.intrinsics.data(), r->intrinsics, 4 * sizeof(float));
    std::memcpy(out.poseEnc.data(), r->pose_enc,
                lingbot_stitch::kPoseEncSize * sizeof(float));
}

}  // namespace

int LingbotMapWorker::streamCallbackEntry(void* user,
                                          const aicore_lingbot_result* r) {
    auto* s = static_cast<StreamState*>(user);
    const int idx = s->worker->m_delivered;
    if (idx >= s->total || !r) return 1;  // abort
    const int globalIndex = s->globalStart + idx;
    LingbotFrameResult& out = (*s->frames)[static_cast<size_t>(idx)];
    fillFrameResult(out, r, globalIndex, s->windowIndex);
    if (s->worker->m_skyReady) {
        const size_t n = static_cast<size_t>(r->width) * r->height;
        out.skyKeep.resize(static_cast<int>(n));
        if (aicore_lingbot_last_sky_mask(
                    static_cast<aicore_lingbot_ctx*>(s->worker->m_pendingCtx),
                    out.skyKeep.data(),
                    static_cast<int>(n)) != static_cast<int>(n)) {
            out.skyKeep.clear();
        }
    }
    // Live preview (window-local coordinates for wi > 0 until the final
    // aligned result replaces it; mirrors the official window-tagged keys).
    if (!s->rgb->empty() && globalIndex >= 0 &&
        globalIndex < static_cast<int>(s->rgb->size())) {
        emit s->worker->framePreviewReady(buildPreview(
                r, globalIndex, s->windowIndex, s->windowCount,
                (*s->rgb)[static_cast<size_t>(globalIndex)],
                s->worker->m_settings.previewStride,
                s->worker->m_settings.confThreshold,
                out.skyKeep.isEmpty() ? nullptr : out.skyKeep.constData()));
    }
    emit s->worker->taskStage(
            s->windowCount > 1
                    ? QObject::tr("Window %1/%2 streaming")
                              .arg(s->windowIndex + 1)
                              .arg(s->windowCount)
                    : QObject::tr("Streaming reconstruction (%1 frames)")
                              .arg(s->total),
            static_cast<int>((globalIndex + 1) * 100 /
                             std::max(s->globalStart + s->total, 1)));
    ++s->worker->m_delivered;
    // Cooperative cancellation: the engine stops and reports the
    // "frame callback aborted" contract error.
    return s->worker->m_cancelRequested.load() ? 1 : 0;
}

LingbotMapWorker::LingbotMapWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    qRegisterMetaType<LingbotRunResult>("LingbotRunResult");
    qRegisterMetaType<LingbotFrameResult>("LingbotFrameResult");
    qRegisterMetaType<LingbotFramePreview>("LingbotFramePreview");
}

LingbotMapWorker::~LingbotMapWorker() { releaseContextOnMainThread(); }

void LingbotMapWorker::releaseContextOnMainThread() {
    if (m_pendingCtx) {
        aicore_lingbot_free(static_cast<aicore_lingbot_ctx*>(m_pendingCtx));
        m_pendingCtx = nullptr;
    }
}

void LingbotMapWorker::run() {
    QElapsedTimer timer;
    timer.start();
    emit taskStage(tr("Loading model"), -1);

    aicore_lingbot_options* opts = aicore_lingbot_options_new();
    if (!opts) {
        emit taskFinished(false);
        return;
    }
    aicore_lingbot_options_set_device(opts,
                                      m_settings.device.toUtf8().constData());
    aicore_lingbot_options_set_threads(opts, m_settings.threads);
    aicore_lingbot_options_set_image_size(opts, m_settings.image_size);
    // Persistent KV-cache profile (upstream --kv_cache_scale/--kv_cache_window;
    // official release default 8/64). The stream-capacity hint sizes the
    // resident-KV F32 special segment to (bound + 2) x 6 tokens per layer, so
    // it must reflect the ACTUAL stream length: passing the raw Max-frames
    // spinbox value instead preallocates gigabytes of never-used capacity
    // whenever the folder holds far fewer frames than the cap.
    aicore_lingbot_options_set_kv_profile(opts, m_settings.kvScale,
                                          m_settings.kvWindow);
    int streamBound = 0;
    if (!m_settings.videoPath.isEmpty()) {
#ifdef HAS_OPENCV_FACE_CAPTURE
        // Container metadata probe (header read only — no decode).
        OpenCVFrameSource probe;
        if (probe.openVideo(m_settings.videoPath.toStdString())) {
            double srcFps = probe.fps();
            if (srcFps <= 0.0 || !std::isfinite(srcFps)) srcFps = 30.0;
            const int interval = std::max(
                    1, static_cast<int>(std::lround(
                               srcFps / std::max(1, m_settings.videoFps))));
            const int64_t containerFrames = probe.frameCount();
            if (containerFrames > 0) {
                streamBound = static_cast<int>(
                        (containerFrames + interval - 1) / interval);
            }
            probe.release();
        }
#endif
    } else {
        streamBound = static_cast<int>(
                listFrameFiles(m_settings.imageFolder,
                               parseImageExtensions(m_settings.imageExt), 0, 1)
                        .size());
    }
    // Same truncation/sampling order as runInference, applied to the bound.
    if (m_settings.maxFrames > 0 && streamBound > m_settings.maxFrames) {
        streamBound = m_settings.maxFrames;
    }
    if (m_settings.frameStride > 1 && streamBound > 0) {
        streamBound = (streamBound + m_settings.frameStride - 1) /
                      m_settings.frameStride;
    }
    // Official windowed semantics (long-sequence pipeline): every window
    // owns a fresh KV cache holding only the window's keyframes
    // (inference_windowed's per-window clean_kv_cache), so the profile is
    // sized to the window and the full-stream capacity hint is skipped —
    // a per-window cache makes a resident full-stream segment pure waste.
    if (m_settings.mode == Settings::Mode::Windowed) {
        aicore_lingbot_options_set_kv_profile(
                opts, kOfficialScaleFrames,
                std::max(m_settings.windowSize, kOfficialScaleFrames));
    } else if (streamBound > 0) {
        aicore_lingbot_options_set_stream_capacity(opts, streamBound);
    }
    // Official long-stream keyframe policy: auto (0) resolves to
    // ceil(N/320) once the real stream length is known; windowed mode runs
    // each window as an independent stream (< 320 frames) with kf=1.
    if (m_settings.mode == Settings::Mode::Windowed) {
        aicore_lingbot_options_set_keyframe_interval(opts, 1);
    } else {
        int kf = 1;
        if (m_settings.keyframeInterval > 0) {
            kf = m_settings.keyframeInterval;
        } else if (streamBound > 0) {
            kf = std::max(1, (streamBound + 319) / 320);
        }
        aicore_lingbot_options_set_keyframe_interval(opts, kf);
        if (kf > 1) {
            emit logMessage(tr("[LingbotMap] Official keyframe policy: "
                               "keyframe interval = %1 (stream ~%2 frames; "
                               "non-keyframes attend but do not persist KV, "
                               "bounding the cache for long runs)")
                                    .arg(kf)
                                    .arg(streamBound));
        }
    }

    // VRAM guidance for GPU OOM paths: the official 8/64 release profile's
    // scale pass dominates the compute workspace and cannot fit small GPUs.
    auto logVramGuidance = [this]() {
        if (m_settings.device == QStringLiteral("cpu")) return;
        emit logMessage(
                tr("[LingbotMap] VRAM hint: the KV-cache profile "
                   "(scale=%1/window=%2 at width %3) needs a large compute "
                   "workspace — the official 8/64 release profile measures "
                   "~21 GiB at 518x294. On smaller GPUs, lower KV cache "
                   "scale/window in Advanced (e.g. 4/32 or 2/16), reduce the "
                   "Processing width, or switch Device to Vulkan (its "
                   "system-memory fallback keeps the run alive on "
                   "overflow).")
                        .arg(m_settings.kvScale)
                        .arg(m_settings.kvWindow)
                        .arg(m_settings.image_size));
    };

    aicore_lingbot_ctx* ctx = aicore_lingbot_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_lingbot_options_free(opts);
    m_pendingCtx = ctx;
    if (!ctx || aicore_lingbot_is_ready(ctx) != 1) {
        emit logMessage(tr("[LingbotMap] Model load failed: %1")
                                .arg(QString::fromUtf8(
                                        aicore_lingbot_last_error(ctx))));
        logVramGuidance();
        aicore_lingbot_free(ctx);
        m_pendingCtx = nullptr;
        emit taskFinished(false);
        return;
    }

    char* deviceName = nullptr;
    {
        // aicore_lingbot_context_device returns a borrowed string owned by
        // the context; copy through info-free accessors only.
        const char* dev = aicore_lingbot_context_device(ctx);
        deviceName = dev ? strdup(dev) : nullptr;
    }
    emit logMessage(tr("[LingbotMap] Model ready (device: %1)")
                            .arg(deviceName ? QString::fromUtf8(deviceName)
                                            : QStringLiteral("?")));
    free(deviceName);

    if (m_settings.skySource == Settings::SkySource::Native) {
        emit taskStage(tr("Loading sky segmentation"), -1);
        if (aicore_lingbot_skyseg_load(
                    ctx, m_settings.skysegModelPath.toUtf8().constData()) !=
            0) {
            emit logMessage(
                    tr("[LingbotMap] Sky segmentation unavailable (%1) — "
                       "continuing without it.")
                            .arg(QString::fromUtf8(
                                    aicore_lingbot_last_error(ctx))));
        }
    } else if (m_settings.skySource == Settings::SkySource::CachedMasks) {
        emit logMessage(tr("[LingbotMap] Using cached sky masks from %1")
                                .arg(m_settings.skyMaskDir));
    }

    m_elapsed.restart();
    if (!runInference(nullptr)) {
        const QString reason =
                QString::fromUtf8(aicore_lingbot_last_error(ctx));
        emit logMessage(tr("[LingbotMap] Inference failed: %1").arg(reason));
        if (reason.contains(QStringLiteral("allocation"))) {
            emit logMessage(
                    tr("[LingbotMap] Device memory was insufficient for this "
                       "graph. Lower the window size (windowed mode), the "
                       "processing width, or pick another device."));
        }
        emit taskFinished(false);
        return;
    }

    emit taskStage(m_settings.mode == Settings::Mode::Windowed
                           ? tr("Windowed reconstruction")
                           : tr("Streaming reconstruction"),
                   100);
    emit taskFinished(true);
}

bool LingbotMapWorker::runInference(QString* error) {
    auto* ctx = static_cast<aicore_lingbot_ctx*>(m_pendingCtx);
    if (!ctx) return false;

    // Input selection (upstream precedence: --video_path replaces
    // --image_folder); both paths share --frames truncation then --stride
    // sampling, applied in that order.
    QStringList files;            // folder paths / video frame stems
    std::vector<QImage> decoded;  // video mode: pre-decoded RGB frames
#ifdef HAS_OPENCV_FACE_CAPTURE
    if (!m_settings.videoPath.isEmpty()) {
        int total = 0, interval = 1;
        QString decodeError;
        if (!decodeVideoFrames(m_settings.videoPath, m_settings.videoFps,
                               decoded, files, &total, &interval,
                               &decodeError)) {
            emit logMessage(
                    tr("[LingbotMap] Video input failed: %1").arg(decodeError));
            return false;
        }
        emit logMessage(
                tr("[LingbotMap] Extracted %1 frame(s) from video (%2 total, "
                   "interval %3)")
                        .arg(files.size())
                        .arg(total)
                        .arg(interval));
        if (m_settings.maxFrames > 0 && files.size() > m_settings.maxFrames) {
            files = files.mid(0, m_settings.maxFrames);
            decoded.resize(static_cast<size_t>(files.size()));
        }
        if (m_settings.frameStride > 1) {
            QStringList sampledNames;
            std::vector<QImage> sampled;
            sampledNames.reserve(files.size() / m_settings.frameStride + 1);
            for (int i = 0; i < files.size(); i += m_settings.frameStride) {
                sampledNames.append(files[i]);
                sampled.push_back(std::move(decoded[static_cast<size_t>(i)]));
            }
            files = sampledNames;
            decoded = std::move(sampled);
        }
    } else {
#endif
        files = listFrameFiles(m_settings.imageFolder,
                               parseImageExtensions(m_settings.imageExt),
                               m_settings.maxFrames, m_settings.frameStride);
#ifdef HAS_OPENCV_FACE_CAPTURE
    }
#else
    if (!m_settings.videoPath.isEmpty()) {
        emit logMessage(
                tr("[LingbotMap] Video input requires a build with OpenCV "
                   "(BUILD_OPENCV=ON) — falling back to the image folder."));
    }
#endif
    if (files.isEmpty()) {
        emit logMessage(tr("[LingbotMap] No input images found in %1")
                                .arg(m_settings.videoPath.isEmpty()
                                             ? m_settings.imageFolder
                                             : m_settings.videoPath));
        return false;
    }
    emit framesDecoded(files.size());
    emit taskStage(tr("Preprocessing %1 frames").arg(files.size()), 0);

    // Preprocess every frame straight into the contiguous stream buffer
    // [N, 3, H, W] (official crop + bicubic + [0,1] NCHW). Preprocessing
    // belongs to AICore; the worker keeps the processed-resolution RGB
    // around for point-cloud colors. No per-frame staging copy: frames
    // land directly at their stream offset.
    std::vector<float> stream;
    std::vector<QImage> rgb;
    QStringList rgbFiles;  // source path per frame (mask lookup)
    rgb.reserve(files.size());
    rgbFiles.reserve(files.size());
    int procW = 0, procH = 0;
    for (int i = 0; i < files.size(); ++i) {
        if (m_cancelRequested.load()) {
            emit logMessage(tr("[LingbotMap] Cancelled during preprocessing."));
            return false;
        }
        // Video mode reads the pre-decoded RGB frame; folder mode loads
        // from disk.
        QImage img = (static_cast<int>(decoded.size()) > i)
                             ? decoded[static_cast<size_t>(i)]
                             : QImage(files[i]);
        if (img.isNull()) {
            emit logMessage(tr("[LingbotMap] Skipping unreadable frame: %1")
                                    .arg(files[i]));
            continue;
        }
        if (m_settings.rotateClockwise90) {
            // Upstream --rotate_clockwise_90 (phone portrait sequences).
            img = img.transformed(QTransform().rotate(90.0));
            if (img.isNull()) {
                emit logMessage(tr("[LingbotMap] Skipping unreadable frame: %1")
                                        .arg(files[i]));
                continue;
            }
        }
        QImage rgb888 = img.convertToFormat(QImage::Format_RGB888);
        aicore_image_view view = viewFromImage(rgb888);
        const int needed = aicore_lingbot_preprocess_image(
                &view, m_settings.image_size, nullptr, 0, nullptr, nullptr);
        if (needed <= 0) {
            emit logMessage(tr("[LingbotMap] Preprocess sizing failed: %1")
                                    .arg(files[i]));
            continue;
        }
        std::vector<float> nchw(static_cast<size_t>(needed));
        int32_t w = 0, h = 0;
        if (aicore_lingbot_preprocess_image(&view, m_settings.image_size,
                                            nchw.data(), needed, &w,
                                            &h) != needed) {
            emit logMessage(
                    tr("[LingbotMap] Preprocess failed: %1").arg(files[i]));
            continue;
        }
        if (stream.empty()) {
            procW = w;
            procH = h;
        } else if (w != procW || h != procH) {
            // The stream contract needs a uniform [N,3,H,W] buffer; one
            // folder must preprocess to one resolution.
            emit logMessage(tr("[LingbotMap] Skipping frame with mismatched "
                               "processed size: %1")
                                    .arg(files[i]));
            continue;
        }
        // Keep the processed-resolution RGB for colors: crop/resize the
        // source view through the same geometry (scale mapping at
        // materialization time uses the two resolutions directly).
        rgb.push_back(rgb888.scaled(w, h, Qt::IgnoreAspectRatio,
                                    Qt::SmoothTransformation));
        rgbFiles.push_back(files[i]);
        stream.insert(stream.end(), nchw.begin(), nchw.end());
        if (i % 20 == 0) {
            emit taskStage(tr("Preprocessing %1 frames").arg(files.size()),
                           static_cast<int>(i * 100 / files.size()));
        }
    }
    if (stream.empty()) return false;

    const int frameCount = static_cast<int>(rgb.size());
    const size_t plane = static_cast<size_t>(procW) * procH * 3;

    LingbotRunResult result;
    result.modelFile = m_settings.modelPath;
    result.device = QString::fromUtf8(aicore_lingbot_context_device(ctx));
    result.windowed = m_settings.mode == Settings::Mode::Windowed;

    // CachedMasks injects per stream call (single call in streaming mode,
    // per window in windowed mode); Native skyseg generates masks inside
    // each stream call and keeps none across calls.
    m_skyReady = aicore_lingbot_skyseg_ready(ctx) == 1 ||
                 m_settings.skySource == Settings::SkySource::CachedMasks;

    const QString streamingLabel =
            tr("Streaming reconstruction (%1 frames, %2x%3)")
                    .arg(frameCount)
                    .arg(procW)
                    .arg(procH);
    const QString windowedLabel =
            tr("Windowed reconstruction (%1 frames, %2x%3)")
                    .arg(frameCount)
                    .arg(procW)
                    .arg(procH);
    emit taskStage(result.windowed ? windowedLabel : streamingLabel, 0);

    const bool ok = result.windowed
                            ? runWindowed(ctx, stream, frameCount, procW, procH,
                                          rgbFiles, rgb, result, error)
                            : runStreaming(ctx, stream, frameCount, procW,
                                           procH, rgbFiles, rgb, result, error);
    if (!ok) return false;

    aicore_pipeline_timings timings{};
    if (aicore_lingbot_last_pipeline_timings(ctx, &timings) == 0) {
        emit logMessage(tr("[LingbotMap] Stream e2e: %1 ms")
                                .arg(timings.e2e_ms, 0, 'f', 1));
    }
    result.elapsedMs = m_elapsed.elapsed();
    emit resultReady(result);
    return true;
}

bool LingbotMapWorker::runStreaming(aicore_lingbot_ctx* ctx,
                                    const std::vector<float>& stream,
                                    int frameCount,
                                    int procW,
                                    int procH,
                                    const QStringList& files,
                                    const std::vector<QImage>& rgb,
                                    LingbotRunResult& result,
                                    QString* error) {
    result.windowCount = 1;
    result.frames.resize(frameCount);
    std::vector<LingbotFrameResult> localFrames(
            static_cast<size_t>(frameCount));

    if (m_settings.skySource == Settings::SkySource::CachedMasks &&
        !injectCachedMasks(ctx, files, 0, frameCount, procW, procH)) {
        emit logMessage(
                tr("[LingbotMap] Failed to inject cached sky "
                   "masks — continuing without them."));
        m_skyReady = aicore_lingbot_skyseg_ready(ctx) == 1;
    }

    StreamState state{this, &localFrames, &rgb, frameCount, 0, 0, 1};
    m_delivered = 0;
    if (aicore_lingbot_infer_stream(ctx, stream.data(), frameCount, procW,
                                    procH, &streamCallbackEntry, &state) != 0) {
        if (m_cancelRequested.load()) {
            emit logMessage(tr("[LingbotMap] Stream cancelled by user."));
            return false;
        }
        if (error) *error = QString::fromUtf8(aicore_lingbot_last_error(ctx));
        return false;
    }
    result.frames.resize(m_delivered);
    // Attach the processed-resolution RGB for point-cloud colors.
    for (int i = 0; i < result.frames.size(); ++i) {
        LingbotFrameResult& out = result.frames[i];
        out = std::move(localFrames[static_cast<size_t>(i)]);
        if (out.globalIndex >= 0 && out.globalIndex < files.size()) {
            out.sourceFile = files[static_cast<size_t>(out.globalIndex)];
        }
        if (out.globalIndex >= 0 &&
            out.globalIndex < static_cast<int>(rgb.size())) {
            out.frameRgb = rgb[static_cast<size_t>(out.globalIndex)];
        }
    }
    return true;
}

bool LingbotMapWorker::injectCachedMasks(aicore_lingbot_ctx* ctx,
                                         const QStringList& files,
                                         int start,
                                         int frameCount,
                                         int procW,
                                         int procH) {
    if (!ctx || frameCount <= 0) return false;
    const size_t maskPlane = static_cast<size_t>(procW) * procH;
    std::vector<unsigned char> masks(maskPlane *
                                     static_cast<size_t>(frameCount));
    int missing = 0;
    for (int f = 0; f < frameCount; ++f) {
        const int global = start + f;
        if (global < 0 || global >= files.size()) {
            ++missing;
            continue;
        }
        unsigned char* dst = masks.data() + static_cast<size_t>(f) * maskPlane;
        std::fill(dst, dst + maskPlane, 255);  // keep by default
        const QString stem = QFileInfo(files[static_cast<size_t>(global)])
                                     .completeBaseName();
        QImage mask(QDir(m_settings.skyMaskDir).filePath(stem + ".png"));
        if (mask.isNull()) {
            ++missing;
            continue;
        }
        if (mask.size() != QSize(procW, procH)) {
            mask = mask.scaled(procW, procH, Qt::IgnoreAspectRatio,
                               Qt::SmoothTransformation);
        }
        const QImage gray = mask.convertToFormat(QImage::Format_Grayscale8);
        for (int y = 0; y < procH; ++y) {
            const uchar* row = gray.constScanLine(y);
            for (int x = 0; x < procW; ++x) {
                dst[static_cast<size_t>(y) * procW + x] = row[x] != 0 ? 255 : 0;
            }
        }
    }
    if (aicore_lingbot_set_external_sky_masks(ctx, masks.data(), frameCount,
                                              procW, procH) != 0) {
        return false;
    }
    if (missing > 0) {
        emit logMessage(tr("[LingbotMap] %1 frame(s) without a cached mask "
                           "were kept unmasked.")
                                .arg(missing));
    }
    return true;
}

bool LingbotMapWorker::runWindowed(aicore_lingbot_ctx* ctx,
                                   const std::vector<float>& stream,
                                   int frameCount,
                                   int procW,
                                   int procH,
                                   const QStringList& files,
                                   const std::vector<QImage>& rgb,
                                   LingbotRunResult& result,
                                   QString* error) {
    const lingbot_stitch::WindowPlan plan = lingbot_stitch::planWindows(
            frameCount, m_settings.windowSize, m_settings.overlap,
            kOfficialScaleFrames);
    result.windowCount = static_cast<int>(plan.windows.size());
    emit logMessage(
            tr("[LingbotMap] Windowed inference: %1 window(s), window=%2, "
               "overlap=%3 (official defaults)")
                    .arg(result.windowCount)
                    .arg(plan.effWindow)
                    .arg(plan.effOverlap));

    const size_t plane = static_cast<size_t>(procW) * procH;
    // Window numeric results stay resident until the stitch (same order of
    // memory as the streaming result); per-window metadata keeps only the
    // sky state (source RGB is resolved from the shared per-frame vector).
    std::vector<lingbot_stitch::WindowData> warped;
    std::vector<std::vector<LingbotFrameResult>> windowMeta;
    warped.reserve(plan.windows.size());
    windowMeta.reserve(plan.windows.size());

    for (int wi = 0; wi < static_cast<int>(plan.windows.size()); ++wi) {
        const int start = plan.windows[static_cast<size_t>(wi)].first;
        const int end = plan.windows[static_cast<size_t>(wi)].second;
        if (m_cancelRequested.load()) {
            emit logMessage(tr("[LingbotMap] Cancelled between windows."));
            return false;
        }
        emit taskStage(tr("Window %1/%2: frames [%3, %4)")
                               .arg(wi + 1)
                               .arg(result.windowCount)
                               .arg(start)
                               .arg(end),
                       static_cast<int>(start * 100 / std::max(frameCount, 1)));

        // Fresh KV cache: the next stream call re-runs the official scale
        // pass (inference_windowed's per-window clean_kv_cache).
        aicore_lingbot_stream_reset(ctx);

        if (m_settings.skySource == Settings::SkySource::CachedMasks &&
            !injectCachedMasks(ctx, files, start, end - start, procW, procH)) {
            emit logMessage(tr("[LingbotMap] Failed to inject cached sky "
                               "masks for window %1 — continuing without "
                               "them.")
                                    .arg(wi + 1));
        }

        std::vector<LingbotFrameResult> local(static_cast<size_t>(end - start));
        StreamState state{
                this, &local, &rgb, end - start, start, wi, result.windowCount};
        m_delivered = 0;
        if (aicore_lingbot_infer_stream(
                    ctx, stream.data() + static_cast<size_t>(start) * plane * 3,
                    end - start, procW, procH, &streamCallbackEntry,
                    &state) != 0) {
            if (m_cancelRequested.load()) {
                emit logMessage(tr("[LingbotMap] Stream cancelled by user."));
                return false;
            }
            if (error) {
                *error = QString::fromUtf8(aicore_lingbot_last_error(ctx));
            }
            return false;
        }

        // WindowData for the official align/warp/stitch math.
        lingbot_stitch::WindowData win;
        win.start = start;
        win.frames = end - start;
        win.width = procW;
        win.height = procH;
        win.poseEnc.resize(static_cast<size_t>(win.frames) *
                           lingbot_stitch::kPoseEncSize);
        win.depth.resize(static_cast<size_t>(win.frames) * plane);
        win.depthConf.resize(static_cast<size_t>(win.frames) * plane);
        win.c2w.resize(static_cast<size_t>(win.frames) *
                       lingbot_stitch::kC2WSize);
        win.intrinsics.resize(static_cast<size_t>(win.frames) *
                              lingbot_stitch::kIntrinsicsSize);
        for (int f = 0; f < win.frames; ++f) {
            const LingbotFrameResult& src = local[static_cast<size_t>(f)];
            std::copy(
                    src.poseEnc.cbegin(), src.poseEnc.cend(),
                    win.poseEnc.begin() + static_cast<ptrdiff_t>(f) *
                                                  lingbot_stitch::kPoseEncSize);
            std::copy(src.depth.cbegin(), src.depth.cend(),
                      win.depth.begin() + static_cast<ptrdiff_t>(f) * plane);
            std::copy(
                    src.depthConf.cbegin(), src.depthConf.cend(),
                    win.depthConf.begin() + static_cast<ptrdiff_t>(f) * plane);
            std::copy(src.c2w.cbegin(), src.c2w.cend(),
                      win.c2w.begin() + static_cast<ptrdiff_t>(f) *
                                                lingbot_stitch::kC2WSize);
            std::copy(src.intrinsics.cbegin(), src.intrinsics.cend(),
                      win.intrinsics.begin() +
                              static_cast<ptrdiff_t>(f) *
                                      lingbot_stitch::kIntrinsicsSize);
        }
        if (wi > 0) {
            const lingbot_stitch::Similarity sim =
                    lingbot_stitch::pairwiseAlignment(
                            warped[static_cast<size_t>(wi - 1)], win,
                            plan.effOverlap);
            emit logMessage(tr("[LingbotMap] Window %1 alignment: scale=%2 "
                               "t=[%3 %4 %5]")
                                    .arg(wi + 1)
                                    .arg(sim.s, 0, 'f', 6)
                                    .arg(sim.t[0], 0, 'f', 4)
                                    .arg(sim.t[1], 0, 'f', 4)
                                    .arg(sim.t[2], 0, 'f', 4));
            lingbot_stitch::warpWindow(win, sim);
        }
        // Keep only the sky state + source path per window frame; the
        // numeric arrays now live in the warped WindowData.
        for (LingbotFrameResult& meta : local) {
            meta.depth.clear();
            meta.depthConf.clear();
            meta.c2w.clear();
            meta.intrinsics.clear();
            meta.poseEnc.clear();
        }
        warped.push_back(std::move(win));
        windowMeta.push_back(std::move(local));
    }

    // Official stitch: non-final windows contribute [0, frames - overlap).
    const lingbot_stitch::WindowData merged =
            lingbot_stitch::stitchWindows(warped, plan.effOverlap);
    const std::vector<std::pair<int, int>> contrib =
            lingbot_stitch::contributionTable(warped, plan.effOverlap);

    result.frames.resize(merged.frames);
    for (int k = 0; k < merged.frames; ++k) {
        const auto [wi, localIdx] = contrib[static_cast<size_t>(k)];
        const LingbotFrameResult& meta =
                windowMeta[static_cast<size_t>(wi)]
                          [static_cast<size_t>(localIdx)];
        LingbotFrameResult& out = result.frames[k];
        out.width = procW;
        out.height = procH;
        out.globalIndex = warped[static_cast<size_t>(wi)].start + localIdx;
        out.windowIndex = wi;
        out.sourceFile = meta.sourceFile;
        out.skyKeep = meta.skyKeep;
        const size_t n = plane;
        out.depth.resize(static_cast<int>(n));
        out.depthConf.resize(static_cast<int>(n));
        out.c2w.resize(16);
        out.intrinsics.resize(4);
        out.poseEnc.resize(lingbot_stitch::kPoseEncSize);
        std::memcpy(out.depth.data(),
                    merged.depth.data() + static_cast<size_t>(k) * n,
                    n * sizeof(float));
        std::memcpy(out.depthConf.data(),
                    merged.depthConf.data() + static_cast<size_t>(k) * n,
                    n * sizeof(float));
        std::memcpy(out.c2w.data(),
                    merged.c2w.data() +
                            static_cast<size_t>(k) * lingbot_stitch::kC2WSize,
                    lingbot_stitch::kC2WSize * sizeof(float));
        std::memcpy(out.intrinsics.data(),
                    merged.intrinsics.data() +
                            static_cast<size_t>(k) *
                                    lingbot_stitch::kIntrinsicsSize,
                    lingbot_stitch::kIntrinsicsSize * sizeof(float));
        std::memcpy(
                out.poseEnc.data(),
                merged.poseEnc.data() +
                        static_cast<size_t>(k) * lingbot_stitch::kPoseEncSize,
                lingbot_stitch::kPoseEncSize * sizeof(float));
        if (out.globalIndex >= 0 &&
            out.globalIndex < static_cast<int>(rgb.size())) {
            out.frameRgb = rgb[static_cast<size_t>(out.globalIndex)];
        }
    }
    emit taskStage(tr("Stitched %1 frames from %2 windows")
                           .arg(result.frames.size())
                           .arg(result.windowCount),
                   100);
    return true;
}
