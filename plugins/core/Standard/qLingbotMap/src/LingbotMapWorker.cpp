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
#include <cstring>

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <cmath>

#include "OpenCVFrameSource.h"
#include "VideoPlaybackWidget.h"
#endif

// AICore public C ABI.
#include "aicore/lingbot_capi.h"

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

}  // namespace

LingbotMapWorker::LingbotMapWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    qRegisterMetaType<LingbotRunResult>("LingbotRunResult");
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
    if (streamBound > 0) {
        aicore_lingbot_options_set_stream_capacity(opts, streamBound);
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
        emit logMessage(tr("[LingbotMap] Inference failed: %1")
                                .arg(QString::fromUtf8(
                                        aicore_lingbot_last_error(ctx))));
        logVramGuidance();
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Streaming reconstruction"), 100);
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

    // Preprocess every frame up-front (official crop + bicubic + [0,1]
    // NCHW). Preprocessing belongs to AICore; the worker keeps the
    // processed-resolution RGB around for point-cloud colors.
    std::vector<std::vector<float>> frames;
    std::vector<QImage> rgb;
    std::vector<QString> rgbFiles;  // source path per frame (mask lookup)
    frames.reserve(files.size());
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
        procW = w;
        procH = h;
        // Keep the processed-resolution RGB for colors: crop/resize the
        // source view through the same geometry (scale mapping at
        // materialization time uses the two resolutions directly).
        rgb.push_back(rgb888.scaled(w, h, Qt::IgnoreAspectRatio,
                                    Qt::SmoothTransformation));
        rgbFiles.push_back(files[i]);
        frames.push_back(std::move(nchw));
        if (i % 20 == 0) {
            emit taskStage(tr("Preprocessing %1 frames").arg(files.size()),
                           static_cast<int>(i * 100 / files.size()));
        }
    }
    if (frames.empty()) return false;

    // Tight, contiguous frame buffer [N, 3, H, W].
    const int frameCount = static_cast<int>(frames.size());
    const size_t plane = static_cast<size_t>(procW) * procH * 3;
    std::vector<float> stream(static_cast<size_t>(frameCount) * plane);
    for (int f = 0; f < frameCount; ++f) {
        std::copy(frames[f].begin(), frames[f].end(),
                  stream.begin() + static_cast<size_t>(f) * plane);
    }
    frames.clear();

    emit taskStage(tr("Streaming reconstruction (%1 frames, %2x%3)")
                           .arg(frameCount)
                           .arg(procW)
                           .arg(procH),
                   0);

    LingbotRunResult result;
    result.modelFile = m_settings.modelPath;
    result.device = QString::fromUtf8(aicore_lingbot_context_device(ctx));
    result.frames.resize(static_cast<int>(rgb.size()));

    struct StreamState {
        LingbotMapWorker* worker;
        LingbotRunResult* result;
        const QStringList* files;
        int total;
    } state{this, &result, &files, static_cast<int>(rgb.size())};

    auto cb = [](void* user, const aicore_lingbot_result* r) -> int {
        auto* s = static_cast<StreamState*>(user);
        const int idx = s->worker->m_delivered;
        if (idx >= s->total || !r) return 1;  // abort
        LingbotFrameResult& out = s->result->frames[idx];
        out.width = r->width;
        out.height = r->height;
        out.sourceFile = (*s->files)[idx];
        const size_t n = static_cast<size_t>(r->width) * r->height;
        out.depth.resize(static_cast<int>(n));
        out.depthConf.resize(static_cast<int>(n));
        out.c2w.resize(16);
        out.intrinsics.resize(4);
        std::memcpy(out.depth.data(), r->depth, n * sizeof(float));
        std::memcpy(out.depthConf.data(), r->depth_conf, n * sizeof(float));
        std::memcpy(out.c2w.data(), r->c2w, 16 * sizeof(float));
        std::memcpy(out.intrinsics.data(), r->intrinsics, 4 * sizeof(float));
        if (s->worker->m_skyReady) {
            out.skyKeep.resize(static_cast<int>(n));
            if (aicore_lingbot_last_sky_mask(static_cast<aicore_lingbot_ctx*>(
                                                     s->worker->m_pendingCtx),
                                             out.skyKeep.data(),
                                             static_cast<int>(n)) != n) {
                out.skyKeep.clear();
            }
        }
        emit s->worker->taskStage(
                tr("Streaming reconstruction (%1 frames)").arg(s->total),
                static_cast<int>((idx + 1) * 100 / s->total));
        ++s->worker->m_delivered;
        // Cooperative cancellation: the engine stops and reports the
        // "frame callback aborted" contract error.
        return s->worker->m_cancelRequested.load() ? 1 : 0;
    };
    m_delivered = 0;
    m_skyReady = false;
    if (m_settings.skySource == Settings::SkySource::CachedMasks) {
        // Cached-mask mode (official <scene>_sky_masks semantics): load the
        // per-frame PNG (255 = keep, named after the source frame stem) at
        // the processed resolution and hand them to the engine in one
        // injection — the native skyseg pass is skipped entirely.
        const size_t maskPlane = static_cast<size_t>(procW) * procH;
        std::vector<unsigned char> masks(maskPlane *
                                         static_cast<size_t>(frameCount));
        int missing = 0;
        for (int f = 0; f < frameCount; ++f) {
            unsigned char* dst =
                    masks.data() + static_cast<size_t>(f) * maskPlane;
            std::fill(dst, dst + maskPlane, 255);  // keep by default
            const QString stem = QFileInfo(rgbFiles[static_cast<size_t>(f)])
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
                    dst[static_cast<size_t>(y) * procW + x] =
                            row[x] != 0 ? 255 : 0;
                }
            }
        }
        if (aicore_lingbot_set_external_sky_masks(ctx, masks.data(), frameCount,
                                                  procW, procH) != 0) {
            emit logMessage(
                    tr("[LingbotMap] Failed to inject cached sky "
                       "masks — continuing without them."));
        } else {
            m_skyReady = true;
            if (missing > 0) {
                emit logMessage(tr("[LingbotMap] %1 frame(s) without a cached "
                                   "mask were kept unmasked.")
                                        .arg(missing));
            }
        }
    } else {
        m_skyReady = aicore_lingbot_skyseg_ready(ctx) == 1;
    }

    if (aicore_lingbot_infer_stream(ctx, stream.data(), frameCount, procW,
                                    procH, cb, &state) != 0) {
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
        result.frames[i].frameRgb = rgb[i];
    }

    aicore_pipeline_timings timings{};
    if (aicore_lingbot_last_pipeline_timings(ctx, &timings) == 0) {
        emit logMessage(tr("[LingbotMap] Stream e2e: %1 ms")
                                .arg(timings.e2e_ms, 0, 'f', 1));
    }
    result.elapsedMs = m_elapsed.elapsed();
    emit resultReady(result);
    return true;
}
