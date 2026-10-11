// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// MapAnything background worker: loads the GGUF, decodes the multi-view set,
// runs the typed AICore inference and assembles the world-frame colored
// cloud + per-view depth rasters. The context lives on the worker thread and
// is freed on the main thread (releaseContextOnMainThread) so GPU teardown
// never races the render thread — same contract as qYOLO.

#include "MapAnythingWorker.h"

#include "MapAnythingGeometry.h"
#include "MapAnythingModelCatalog.h"
#include "MapAnythingReconstruction.h"
#include "MapAnythingSceneMeta.h"
#include "ecvAICoreRuntimeHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/map_anything_capi.h"
#include "aicore/runtime_capi.h"
#endif

#include <QFileInfo>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

#ifdef AICore_ENABLED

// Turbo-style colorization (near = blue, far = red), the same visual
// convention the YOLO depth panels use. Values outside [min,max] clamp.
QImage colorizeDepth(
        const float* depth, int width, int height, float minD, float maxD) {
    QImage image(width, height, QImage::Format_RGB32);
    const float range = maxD > minD ? maxD - minD : 1.0f;
    for (int y = 0; y < height; ++y) {
        QRgb* line = reinterpret_cast<QRgb*>(image.scanLine(y));
        for (int x = 0; x < width; ++x) {
            const float d = depth[(size_t)y * width + x];
            float t = (d - minD) / range;
            t = std::min(1.0f, std::max(0.0f, t));
            // 5th-order Turbo polynomial (standard approximation).
            const float r = std::min(
                    1.0f,
                    std::max(0.0f,
                             0.13572138f +
                                     t * (4.61539260f +
                                          t * (-42.66032258f +
                                               t * (132.13108234f +
                                                    t * (-152.94239396f +
                                                         t * 59.28637943f))))));
            const float g = std::min(
                    1.0f,
                    std::max(0.0f,
                             0.09140261f +
                                     t * (2.19418839f +
                                          t * (4.84296658f +
                                               t * (-14.18503333f +
                                                    t * (4.27729857f +
                                                         t * 2.82956604f))))));
            const float b = std::min(
                    1.0f,
                    std::max(0.0f,
                             0.10667330f +
                                     t * (12.64194608f +
                                          t * (-60.58204836f +
                                               t * (110.36276771f +
                                                    t * (-89.90310912f +
                                                         t * 27.34824973f))))));
            line[x] = qRgb(int(r * 255), int(g * 255), int(b * 255));
        }
    }
    return image;
}

#endif  // AICore_ENABLED

}  // namespace

MapAnythingWorker::MapAnythingWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

MapAnythingWorker::~MapAnythingWorker() {
#ifdef AICore_ENABLED
    aicore_cancel_token_free(m_cancelToken);
#endif
}

void MapAnythingWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    aicore_cancel_token_request(m_cancelToken);
#endif
}

void MapAnythingWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx != nullptr) {
        aicore_map_anything_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED

void MapAnythingWorker::appendViewRasters(aicore_map_anything_ctx* ctx,
                                          int i,
                                          int g) {
    const float* depthArr = aicore_map_anything_result_depth(ctx);
    if (depthArr == nullptr) return;
    const float* confArr = aicore_map_anything_result_conf(ctx);
    const float* maskArr = aicore_map_anything_result_mask(ctx);
    const int height = aicore_map_anything_result_height(ctx);
    const int width = aicore_map_anything_result_width(ctx);
    const size_t plane = (size_t)height * width;
    const QString arch =
            QString::fromUtf8(aicore_map_anything_result_architecture(ctx));
    const bool rawLogitArch =
            arch == QStringLiteral("pi3") || arch == QStringLiteral("pi3x");

    int32_t vw = width, vh = height;
    aicore_map_anything_result_view_valid_size(ctx, i, &vw, &vh);
    vw = std::min(vw, width);
    vh = std::min(vh, height);
    const int ox = (width - vw) / 2, oy = (height - vh) / 2;
    const float* vd = depthArr + (size_t)i * plane;
    const float* vm = maskArr ? maskArr + (size_t)i * plane : nullptr;
    std::vector<float> cropD, cropC;
    cropD.reserve((size_t)vw * vh);
    cropC.reserve((size_t)vw * vh);
    float minD = 0, maxD = 0;
    for (int y = 0; y < vh; ++y) {
        for (int x = 0; x < vw; ++x) {
            const float d = vd[(size_t)(oy + y) * width + (ox + x)];
            cropD.push_back(d);
            if (std::isfinite(d) && d > 0) {
                minD = minD > 0 ? std::min(minD, d) : d;
                maxD = std::max(maxD, d);
            }
            if (confArr != nullptr) {
                const float c = confArr[(size_t)i * plane +
                                        (size_t)(oy + y) * width + (ox + x)];
                cropC.push_back(rawLogitArch ? MapAnythingGeometry::sigmoid(c)
                                : c > 1.0f   ? (c - 1.0f) / c
                                             : 0.0f);
            }
        }
    }
    MapAnythingViewResult view;
    view.width = vw;
    view.height = vh;
    for (float v : cropD) view.depthMap.append(v);
    view.depthImage = colorizeDepth(cropD.data(), vw, vh, minD, maxD);
    if (confArr != nullptr) {
        view.confImage = colorizeDepth(cropC.data(), vw, vh, 0.0f, 1.0f);
    }
    if (vm != nullptr) {
        QImage m(vw, vh, QImage::Format_Grayscale8);
        for (int y = 0; y < vh; ++y) {
            uchar* line = m.scanLine(y);
            for (int x = 0; x < vw; ++x) {
                const float mk = vm[(size_t)(oy + y) * width + (ox + x)];
                line[x] = mk >= 0.5f ? uchar(255) : uchar(0);
            }
        }
        view.maskImage = m;
    }
    view.sourcePath = m_settings.imagePaths.at(g);
    m_result.views.append(view);
}

bool MapAnythingWorker::runInference() {
    const int viewCount = m_settings.imagePaths.size();
    if (viewCount <= 0) {
        emit logMessage(tr("[Error] No input images."));
        return false;
    }
    // Official DUSt3R semantics are pair-wise: the model consumes exactly
    // 2 views and its heads live in view1's frame; >2 views would require
    // the (unported) global alignment optimizer.
    {
        MapAnythingModelEntry entry;
        if (MapAnythingHelpers::findModelByFilename(
                    QFileInfo(m_settings.modelPath).fileName(), &entry) &&
            entry.variant == QStringLiteral("dust3r") && viewCount != 2) {
            emit logMessage(
                    tr("[Error] DUSt3R is a pair-wise model: exactly 2 "
                       "input views are required (got %1). Select 2 images "
                       "or switch to an N-view architecture (VGGT-Omega / "
                       "MapAnything / Pi3).")
                            .arg(viewCount));
            return false;
        }
    }
    // Reference cameras are read BEFORE the model: a bad file must fail in
    // milliseconds, not after minutes of inference. The user's model_aligner
    // file wins over a discovered benchmark scene_meta.json.
    MapAnythingSceneMeta::ReferenceCenters reference;
    QString referencePath;
    if (!m_settings.refImagesPath.isEmpty()) {
        referencePath = m_settings.refImagesPath;
        QString refError;
        if (!MapAnythingSceneMeta::loadColmapReferences(
                    referencePath, m_settings.imagePaths, m_settings.refIsGps,
                    &reference, &refError)) {
            emit logMessage(tr("[Error] Reference cameras: %1").arg(refError));
            return false;
        }
    } else if (!m_settings.gtMetaPath.isEmpty()) {
        referencePath = m_settings.gtMetaPath;
        QString metaError;
        if (!MapAnythingSceneMeta::loadCenters(
                    referencePath, m_settings.imagePaths, &reference.centers,
                    &metaError)) {
            emit logMessage(tr("[Error] Reference cameras: %1").arg(metaError));
            return false;
        }
    }

    // Steps: 1 model load, 1 decode, one per view (emitted by the shared
    // driver's first-cover hook, so overlap frames never count twice), 1
    // scene filter.
    const int totalSteps = viewCount + 3;
    int step = 0;

    aicore_map_anything_options* opts = aicore_map_anything_options_new();
    aicore_map_anything_options_set_device(
            opts, m_settings.device.toUtf8().constData());
    aicore_map_anything_options_set_threads(opts, m_settings.threads);
    aicore_map_anything_options_set_image_size(opts, m_settings.imageSize);
    aicore_map_anything_options_set_resize_mode(
            opts, m_settings.resizeMode.toUtf8().constData());

    emit progressUpdate(++step, totalSteps);
    m_pendingCtx = aicore_map_anything_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_map_anything_options_free(opts);
    if (m_pendingCtx == nullptr ||
        aicore_map_anything_is_ready(m_pendingCtx) == 0) {
        const char* err = m_pendingCtx
                                  ? aicore_map_anything_last_error(m_pendingCtx)
                                  : nullptr;
        emit logMessage(tr("[Error] Model load failed: %1")
                                .arg(err != nullptr
                                             ? QString::fromUtf8(err)
                                             : QStringLiteral("unknown")));
        return false;
    }
    emit modelInfoReady(
            tr("[MapAnything] Model loaded (%1, device %2, "
               "%3 threads).")
                    .arg(QString::fromUtf8(
                                 aicore_map_anything_result_architecture(
                                         m_pendingCtx)),
                         QString::fromUtf8(aicore_map_anything_context_device(
                                 m_pendingCtx)))
                    .arg(aicore_map_anything_context_threads(m_pendingCtx)));

    // Decode + borrow every view. Courtyard-class scenes are 38 x 6201x4135
    // PNGs and serial decoding alone costs tens of seconds of wall time
    // that reads as a hang; the images are independent, so the pool maps
    // cleanly over views.
    QVector<QImage> sources;
    sources.reserve(viewCount);
    std::vector<aicore_image_view> views;
    views.reserve(viewCount);
    {
        emit logMessage(tr("[MapAnything] Decoding %1 views (parallel)...")
                                .arg(viewCount));
        std::vector<QImage> decoded((size_t)viewCount);
        std::vector<int> decodeIdx((size_t)viewCount);
        for (int i = 0; i < viewCount; ++i) decodeIdx[(size_t)i] = i;
        // Read-only access only: the list is implicitly shared with the
        // dialog's Settings copy, so a non-const operator[] from the pool
        // threads would race on detach() and corrupt the heap.
        const QStringList& paths = m_settings.imagePaths;
        QtConcurrent::blockingMap(decodeIdx, [&](int i) {
            decoded[(size_t)i] =
                    QImage(paths.at(i)).convertToFormat(QImage::Format_RGB888);
        });
        for (int i = 0; i < viewCount; ++i) {
            if (decoded[(size_t)i].isNull()) {
                emit logMessage(tr("[Error] Failed to decode image: %1")
                                        .arg(paths.at(i)));
                return false;
            }
            sources.append(decoded[(size_t)i]);
        }
    }
    for (int i = 0; i < viewCount; ++i) {
        // Shared AICore QImage->borrowed-view mapping (stride-aware, format
        // checked) - no per-worker reimplementation.
        const aicore_image_view view =
                ecvAICoreRuntime::makeImageView(sources[i]);
        if (view.data == nullptr) {
            emit logMessage(tr("[Error] Unsupported image format: %1")
                                    .arg(m_settings.imagePaths.at(i)));
            return false;
        }
        views.push_back(view);
    }
    emit progressUpdate(++step, totalSteps);

    // ---- one shared driver for GUI and headless ---------------------------
    // Single pass is the default and the best-quality path (~118 MiB/frame
    // f16@512, upstream VRAM-measured). Larger sets, or the responsive
    // toggle, stream sliding windows (overlap 2) aligned by a dense
    // per-pixel Sim3 on the shared frames.
    MapAnythingReconstruction::Options ro;
    ro.maxFramesPerPass = m_settings.responsiveStreaming
                                  ? 16
                                  : std::max(2, m_settings.maxFramesPerPass);
    ro.overlap = 2;
    ro.buildCloud = m_settings.addPointCloud || m_settings.addPerViewClouds;
    ro.depthEdgeRtol = m_settings.depthEdgeRtol;
    const int K = std::min(viewCount, ro.maxFramesPerPass);
    const size_t passes =
            MapAnythingGeometry::plan_chunk_starts(viewCount, K, ro.overlap)
                    .size();
    if (passes > 1) {
        emit logMessage(tr("[Warning] %1 frames exceed the %2-frame "
                           "single-pass budget; streaming %3 chunks "
                           "(overlap 2, dense shared-frame Sim3 alignment; "
                           "quality slightly below single-pass).")
                                .arg(viewCount)
                                .arg(K)
                                .arg(passes));
    }
    emit logMessage(
            tr("[MapAnything] Running %1 forward(s) over %2 frames - each "
               "forward is one blocking GPU call (~118 MiB/frame).")
                    .arg(passes)
                    .arg(viewCount));

    MapAnythingReconstruction::Result rec;
    std::string error;
    const int rc = MapAnythingReconstruction::run(
            m_pendingCtx, views.data(), viewCount, ro, &rec, &error,
            [&](const MapAnythingReconstruction::ChunkView& cv, int i, int g) {
                if (m_settings.addDepthRasters) appendViewRasters(cv.ctx, i, g);
                emit progressUpdate(++step, totalSteps);
            },
            [](int, int) { return aicore_cancel_requested() == 0; });
    if (rc == MapAnythingReconstruction::kCancelled) {
        emit logMessage(tr("[MapAnything] Cancelled between chunks."));
        return false;
    }
    if (rc != MapAnythingReconstruction::kOk) {
        emit logMessage(tr("[Error] Inference failed: %1")
                                .arg(QString::fromStdString(error)));
        return false;
    }
    for (const std::string& note : rec.memoryNotes) {
        emit logMessage(tr("[MapAnything] Device memory: %1 -> halving the "
                           "pass size.")
                                .arg(QString::fromStdString(note)));
    }
    if (rec.framesPerPass != rec.requestedFramesPerPass) {
        emit logMessage(tr("[MapAnything] Ran %1 views per pass (%2 chunks) "
                           "instead of %3.")
                                .arg(rec.framesPerPass)
                                .arg(rec.chunks.size())
                                .arg(rec.requestedFramesPerPass));
    }
    for (size_t ci = 1; ci < rec.chunks.size(); ++ci) {
        const MapAnythingReconstruction::Chunk& c = rec.chunks[ci];
        emit logMessage(tr("[MapAnything] Chunk %1 (frames %2..%3): Sim3 "
                           "scale %4 from %5 shared pixels, median "
                           "residual %6.")
                                .arg(ci)
                                .arg(c.start)
                                .arg(c.start + c.count - 1)
                                .arg(c.toGlobal.s, 0, 'f', 4)
                                .arg(c.correspondences)
                                .arg(c.residualMedian, 0, 'g', 4));
    }

    // Reference frame (benchmark scene_meta.json: upstream e2e_full_scene.py
    // --align gt, the frame of the published upstream e2e PLYs; or the
    // user's model_aligner file). Before the cameras and the cloud are
    // copied out, so every output shares it.
    bool anchored = false;
    if (!referencePath.isEmpty()) {
        MapAnythingReconstruction::ReferenceAnchor anchor;
        std::string anchorError;
        if (!MapAnythingReconstruction::anchor_to_reference(
                    rec, reference.centers, &anchor, &anchorError)) {
            emit logMessage(tr("[Error] Reference-frame anchor (%1): %2")
                                    .arg(referencePath,
                                         QString::fromStdString(anchorError)));
            return false;
        }
        anchored = true;
        m_result.referenceFrame = referencePath;
        m_result.referenceScale = anchor.sim3.s;
        for (int k = 0; k < 3; ++k)
            m_result.referenceOffset[k] = reference.offset[k];
        emit logMessage(tr("[MapAnything] Anchored to the reference frame "
                           "(%1): Sim3 scale %2 from %3 cameras, median "
                           "camera-center residual %4 (spread across the "
                           "camera path %5).")
                                .arg(referencePath)
                                .arg(anchor.sim3.s, 0, 'f', 4)
                                .arg(anchor.pairs)
                                .arg(anchor.residualMedian, 0, 'g', 4)
                                .arg(anchor.crossSpread, 0, 'g', 4));
        if (reference.gps) {
            emit logMessage(tr("[MapAnything] GPS references -> local "
                               "East-North-Up metres, origin lat %1 lon %2 "
                               "alt %3 (the file's first line).")
                                    .arg(reference.enuOrigin[0], 0, 'f', 8)
                                    .arg(reference.enuOrigin[1], 0, 'f', 8)
                                    .arg(reference.enuOrigin[2], 0, 'f', 3));
        }
        if (reference.offset[0] != 0.0 || reference.offset[1] != 0.0 ||
            reference.offset[2] != 0.0) {
            emit logMessage(tr("[MapAnything] Large reference coordinates: "
                               "stored relative to (%1, %2, %3), carried as "
                               "the cloud's global shift.")
                                    .arg(reference.offset[0], 0, 'f', 1)
                                    .arg(reference.offset[1], 0, 'f', 1)
                                    .arg(reference.offset[2], 0, 'f', 1));
        }
    }

    m_result.viewCount = viewCount;
    m_result.architecture = QString::fromStdString(rec.architecture);
    m_result.resolvedDevice = QString::fromStdString(rec.device);
    m_result.e2eMs = rec.e2eMs;
    m_result.hasMetricScale = rec.hasMetricScale;
    m_result.metricScale = rec.metricScale;
    if (!rec.textEmbedding.empty()) {
        m_result.textEmbedding = QByteArray(
                reinterpret_cast<const char*>(rec.textEmbedding.data()),
                (int)(rec.textEmbedding.size() * sizeof(float)));
        m_result.textEmbeddingDim = (int)rec.textEmbedding.size();
    }
    // Cameras need intrinsics; only the vggt-family pose_enc carries them.
    for (float f : rec.focalPx) {
        if (std::isfinite(f) && f > 0.0f) m_result.hasCameras = true;
    }
    if (m_result.hasCameras) {
        m_result.camWidth = rec.width;
        m_result.camHeight = rec.height;
        for (float v : rec.c2w) m_result.camPoses16.append(v);
        for (float f : rec.focalPx) {
            m_result.camFocalPx.append(std::isfinite(f) ? f : 0.0f);
        }
    }

    // ---- scene-wide official filter stack (shared with headless) --------
    MapAnythingReconstruction::Cloud& cloud = rec.cloud;
    MapAnythingReconstruction::FilterOptions fo;
    fo.minConf = m_settings.minConf;
    fo.confPercentile = m_settings.confPercentile;
    fo.voxelSize = m_settings.voxelSize;
    const MapAnythingReconstruction::FilterReport fr =
            MapAnythingReconstruction::filter_scene(
                    cloud, rec.architecture, fo,
                    rec.hasMetricScale || anchored);
    if (!fr.ok) {
        emit logMessage(tr("[Error] Scene filter: %1")
                                .arg(QString::fromStdString(fr.rule)));
        return false;
    }
    emit logMessage(tr("[MapAnything] Filter %1%2: %3 -> %4 points%5.")
                            .arg(QString::fromStdString(fr.rule))
                            .arg(std::isfinite(fr.confFloor)
                                         ? tr(" (floor %1)")
                                                   .arg(fr.confFloor, 0, 'f', 3)
                                         : QString())
                            .arg(fr.input)
                            .arg(fr.kept)
                            .arg(fr.voxelApplied
                                         ? tr(", metric voxel %1 -> %2")
                                                   .arg(fo.voxelSize, 0, 'f', 3)
                                                   .arg(fr.fused.size())
                                         : QString()));
    emit progressUpdate(++step, totalSteps);

    if (m_settings.addPointCloud) {
        const size_t count = fr.voxelApplied ? fr.fused.size() : cloud.size();
        m_result.points.reserve((int)(count * 3));
        m_result.colors.reserve((int)(count * 3));
        for (size_t k = 0; k < count; ++k) {
            const size_t i = fr.voxelApplied ? fr.fused[k] : k;
            for (int c = 0; c < 3; ++c) {
                m_result.points.append(cloud.xyz[i * 3 + c]);
                m_result.colors.append(cloud.rgb[i * 3 + c]);
            }
        }
        m_result.pointCount = (int)count;
    }
    // Opt-in per-view entities: the confidence-kept population sliced back
    // per view (no voxel merge - that is a fused-cloud stage). cloud.view is
    // ascending, so one linear pass slices it.
    if (m_settings.addPerViewClouds) {
        m_result.perViewPoints.resize(viewCount);
        m_result.perViewColors.resize(viewCount);
        for (size_t i = 0; i < cloud.size(); ++i) {
            const int g = cloud.view[i];
            for (int c = 0; c < 3; ++c) {
                m_result.perViewPoints[g].append(cloud.xyz[i * 3 + c]);
                m_result.perViewColors[g].append(cloud.rgb[i * 3 + c]);
            }
        }
    }

    // Compact stats in the JSON envelope (compat/export boundary only - the
    // typed arrays above are the hot path).
    QString chunksJson;
    for (const MapAnythingReconstruction::Chunk& c : rec.chunks) {
        if (!chunksJson.isEmpty()) chunksJson += QLatin1Char(',');
        chunksJson += QStringLiteral("{\"start\":%1,\"count\":%2,\"scale\":%3}")
                              .arg(c.start)
                              .arg(c.count)
                              .arg(c.toGlobal.s, 0, 'g', 7);
    }
    m_result.resultJson =
            QString(QStringLiteral("{\"architecture\":\"%1\",\"views\":%2,"
                                   "\"height\":%3,\"width\":%4,"
                                   "\"e2e_ms\":%5,\"points\":%6,"
                                   "\"chunks\":[%7],\"frames_per_pass\":%8}"))
                    .arg(m_result.architecture)
                    .arg(m_result.viewCount)
                    .arg(rec.height)
                    .arg(rec.width)
                    .arg(m_result.e2eMs, 0, 'f', 1)
                    .arg(m_result.pointCount)
                    .arg(chunksJson)
                    .arg(rec.framesPerPass)
                    .toUtf8();
    return true;
}

#endif  // AICore_ENABLED

void MapAnythingWorker::run() {
#ifdef AICore_ENABLED
    m_result = MapAnythingResult();
    // Bind the cancel token for this thread: the AICore run path checks
    // aicore_cancel_requested() at the run boundary (run-granularity
    // cancellation; the graph itself is one monolithic compute call).
    aicore_cancel_scope_begin(m_cancelToken);
    const bool ok = runInference();
    aicore_cancel_scope_end(m_cancelToken);
    if (ok) emit resultReady(m_result);
    emit taskFinished(ok);
#else
    emit logMessage(tr("[Error] Built without AICore."));
    emit taskFinished(false);
#endif
}
