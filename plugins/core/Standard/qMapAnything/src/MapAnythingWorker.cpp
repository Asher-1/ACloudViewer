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
#include "ecvAICoreRuntimeHelpers.h"

#ifdef AICore_ENABLED
#include "aicore/map_anything_capi.h"
#include "aicore/runtime_capi.h"
#endif

#include <QElapsedTimer>
#include <QFileInfo>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

bool MapAnythingWorker::appendViewCloud(int chunkViewIdx,
                                        int globalViewIdx,
                                        int height,
                                        int width,
                                        const QVector<QImage>& sources,
                                        std::vector<float>& outPts,
                                        std::vector<quint8>& outCols,
                                        std::vector<float>& outConf,
                                        const uint8_t* edgeMask) {
    const float* depth = aicore_map_anything_result_depth(m_pendingCtx);
    const float* conf = aicore_map_anything_result_conf(m_pendingCtx);
    const float* world = aicore_map_anything_result_points(m_pendingCtx);
    const float* local = aicore_map_anything_result_local_points(m_pendingCtx);
    const float* pose = aicore_map_anything_result_pose(m_pendingCtx);
    const int32_t poseStride =
            aicore_map_anything_result_pose_stride(m_pendingCtx);
    const QString arch = QString::fromUtf8(
            aicore_map_anything_result_architecture(m_pendingCtx));

    const size_t plane = (size_t)height * width;
    const bool pi3Family = (poseStride == 16);
    const bool vggtFamily = (poseStride == 9);

    // Cloud source decision per architecture (see MapAnythingGeometry.h):
    //   dust3r          -> local points only (both views share view1's frame)
    //   mapanything,
    //   vggt-1b         -> world/common-frame points when published
    //   pi3/pi3x        -> local points lifted by the c2w pose
    //   vggt-omega/vggt -> depth unprojected through the FoV intrinsics and
    //                      the pose_enc extrinsic
    const float* points = nullptr;
    float w2cR[9], w2cT[3], k[4];
    bool useUnproject = false;
    if (arch == QStringLiteral("dust3r")) {
        points = local;
    } else if (world != nullptr) {
        points = world;
    } else if (local != nullptr && pi3Family && pose != nullptr) {
        points = local;
    } else if (depth != nullptr && vggtFamily && pose != nullptr) {
        useUnproject = true;
    } else {
        return false;
    }

    const bool rawLogit = pi3Family && arch != QStringLiteral("dust3r");
    // Per-view valid (pre-pad) rect; vx0 < 0 = unavailable (no cropping).
    int vw = -1, vh = -1, vx0 = -1, vy0 = -1;
    if (m_pendingCtx != nullptr) {
        int32_t rw = 0, rh = 0;
        if (aicore_map_anything_result_view_valid_size(
                    m_pendingCtx, chunkViewIdx, &rw, &rh) == 0 &&
            rw > 0 && rh > 0 && rw <= width && rh <= height) {
            vw = rw;
            vh = rh;
            vx0 = (width - vw) / 2;
            vy0 = (height - vh) / 2;
        }
    }

    const float* viewConf =
            conf != nullptr ? conf + (size_t)chunkViewIdx * plane : nullptr;
    const QImage& src = sources[globalViewIdx];
    const int srcW = src.width(), srcH = src.height();

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            const size_t idx = (size_t)v * width + u;
            // Padded-border skip: pixels outside the view's pre-pad valid
            // rect are white filler; the model's predictions there are not
            // meaningful and unproject into the frustum-crossing sheets.
            if (vx0 >= 0 &&
                (u < vx0 || u >= vx0 + vw || v < vy0 || v >= vy0 + vh)) {
                continue;
            }
            // Official depth-edge filter: 3x3 depth-jump discontinuities
            // (flying pixels at object boundaries) never enter the cloud.
            if (edgeMask != nullptr && edgeMask[idx]) {
                continue;
            }
            float world3[3];
            if (useUnproject) {
                const float* poseRow = pose + (size_t)chunkViewIdx * 9;
                MapAnythingGeometry::pose_enc_to_w2c(poseRow, w2cR, w2cT);
                const float fovH = poseRow[7], fovW = poseRow[8];
                MapAnythingGeometry::fov_to_intrinsics(fovH, fovW, width,
                                                       height, k);
                // PER-VIEW plane offset: depth is the (S,H,W) stack —
                // idx alone addresses view 0's plane, which silently
                // composited every view from view 0's depth map (the
                // folded-cookie defect).
                MapAnythingGeometry::unproject_depth_pixel(
                        w2cR, w2cT, k, u, v,
                        depth[(size_t)chunkViewIdx * plane + idx], world3);
            } else {
                const float* p = points + idx * 3;
                if (pi3Family && points == local && pose != nullptr) {
                    MapAnythingGeometry::transform_by_c2w(
                            pose + (size_t)chunkViewIdx * 16, p, world3);
                } else {
                    world3[0] = p[0];
                    world3[1] = p[1];
                    world3[2] = p[2];
                }
            }
            if (!std::isfinite(world3[0]) || !std::isfinite(world3[1]) ||
                !std::isfinite(world3[2])) {
                continue;
            }
            // RAW confidence travels with the point; the SCENE-WIDE
            // percentile floor (max(2.0, p20)) is applied after all chunks
            // land - official predictions_to_glb semantics.
            outConf.push_back(viewConf ? viewConf[idx] : -1.0f);
            outPts.push_back(world3[0]);
            outPts.push_back(world3[1]);
            outPts.push_back(world3[2]);
            // Nearest source-image color sample (the dense grid may differ
            // from the source resolution for non-omega builders).
            if (!src.isNull()) {
                const int su =
                        std::min(srcW - 1, (u - (vx0 > 0 ? vx0 : 0)) * srcW /
                                                   std::max(1, vw));
                const int sv =
                        std::min(srcH - 1, (v - (vy0 > 0 ? vy0 : 0)) * srcH /
                                                   std::max(1, vh));
                const QRgb rgb = src.pixel(su, sv);
                outCols.push_back(quint8(qRed(rgb)));
                outCols.push_back(quint8(qGreen(rgb)));
                outCols.push_back(quint8(qBlue(rgb)));
            } else {
                outCols.push_back(200);
                outCols.push_back(200);
                outCols.push_back(200);
            }
        }
    }
    return true;
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
    const int totalSteps = viewCount * 3 + 1;
    int step = 0;

    aicore_map_anything_options* opts = aicore_map_anything_options_new();
    aicore_map_anything_options_set_device(
            opts, m_settings.device.toUtf8().constData());
    aicore_map_anything_options_set_threads(opts, m_settings.threads);
    aicore_map_anything_options_set_image_size(opts, m_settings.imageSize);
    aicore_map_anything_options_set_resize_mode(
            opts, m_settings.resizeMode.toUtf8().constData());

    emit progressUpdate(++step, totalSteps);  // model load + inference slot
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

    // Decode + borrow every view (packed RGB staging, one owned buffer per
    // view; released at scope exit after the inference consumed them).
    QVector<QImage> sources;
    sources.reserve(viewCount);
    std::vector<aicore_image_view> views;
    views.reserve(viewCount);
    std::vector<std::vector<uint8_t>> staging;
    staging.reserve(viewCount);
    // Parallel decode pass: courtyard-class scenes are 38 x 6201x4135
    // PNGs and serial decoding alone costs tens of seconds of wall time
    // that reads as a hang. The decoded images are independent and QImage
    // is thread-safe for this use, so the pool maps cleanly over views.
    {
        emit logMessage(tr("[MapAnything] Decoding %1 views (parallel)...")
                                .arg(viewCount));
        std::vector<QImage> decoded((size_t)viewCount);
        std::vector<int> decodeIdx((size_t)viewCount);
        for (int i = 0; i < viewCount; ++i) decodeIdx[(size_t)i] = i;
        QtConcurrent::blockingMap(decodeIdx, [&](int i) {
            decoded[(size_t)i] =
                    QImage(m_settings.imagePaths[i])
                            .convertToFormat(QImage::Format_RGB888);
        });
        for (int i = 0; i < viewCount; ++i) {
            if (decoded[(size_t)i].isNull()) {
                emit logMessage(tr("[Error] Failed to decode image: %1")
                                        .arg(m_settings.imagePaths[i]));
                return false;
            }
            sources.append(decoded[(size_t)i]);
        }
        emit progressUpdate(++step, totalSteps);
    }
    for (int i = 0; i < viewCount; ++i) {
        // Shared AICore QImage→borrowed-view mapping (stride-aware, format
        // checked) — no per-worker reimplementation.
        const aicore_image_view view =
                ecvAICoreRuntime::makeImageView(sources[i]);
        if (view.data == nullptr) {
            emit logMessage(tr("[Error] Unsupported image format: %1")
                                    .arg(m_settings.imagePaths[i]));
            return false;
        }
        views.push_back(view);
        emit progressUpdate(++step, totalSteps);
    }

    QElapsedTimer timer;
    timer.start();

    // ---- chunked streaming (official e2e_full_scene.py semantics) -------
    // chunk=0/single-pass is the DEFAULT and the best-quality path (~118
    // MiB/frame f16@512, upstream VRAM-measured). Frames beyond
    // maxFramesPerPass fall back to a sliding window with overlap=2 and a
    // per-boundary robust Horn Sim3 (the upstream "chain" mode, documented
    // as a quality downgrade vs single-pass).
    const int K = m_settings.responsiveStreaming
                          ? std::min(viewCount, 16)
                          : std::min(viewCount,
                                     std::max(2, m_settings.maxFramesPerPass));
    std::vector<int> chunkStarts;
    if (K >= viewCount) {
        chunkStarts = {0};
    } else {
        emit logMessage(tr("[Warning] %1 frames exceed the %2-frame "
                           "single-pass budget; streaming with chunk=%3, "
                           "overlap=2 and per-boundary Sim3 alignment "
                           "(quality slightly below single-pass).")
                                .arg(viewCount)
                                .arg(K)
                                .arg(K));
        const int step = std::max(1, K - 2);
        for (int st = 0; st < viewCount; st += step) {
            chunkStarts.push_back(std::min(st, viewCount - K));
        }
        chunkStarts.push_back(viewCount - K);
        std::sort(chunkStarts.begin(), chunkStarts.end());
        chunkStarts.erase(std::unique(chunkStarts.begin(), chunkStarts.end()),
                          chunkStarts.end());
    }

    // Global snapshot header (the architecture/device/grid are identical
    // across chunks - same model, same preprocessing).
    m_result.viewCount = viewCount;

    // Global accumulation pool. ~19 B/px at the 512 grid => ~5 MB/frame, a
    // 38-frame scene peaks at ~190 MB of RAM - comfortably RAM-bound.
    std::vector<float> gPts, gConf;
    std::vector<quint8> gCols;
    std::vector<size_t> viewBegin((size_t)viewCount, SIZE_MAX);
    std::vector<size_t> viewEnd((size_t)viewCount, 0);
    // Raw per-view camera-TO-world poses (chunk frame) + the chunk each
    // view belongs to; aligned into chunk 0's frame after the loop.
    std::vector<float> rawCamPoses16;
    std::vector<int> camChunkOf;
    // Global-frame camera centers per view (-1 = not covered yet); the
    // chunk-boundary Sim3 fits on these.
    std::vector<float> centersGlobal((size_t)viewCount * 3, -1.0f);
    // Per-chunk Sim3 (s, R 3x3 row-major, t 3) into chunk 0's frame.
    std::vector<std::vector<float>> chunkSim3(chunkStarts.size());
    int height = 0, width = 0;

    emit logMessage(
            tr("[MapAnything] Running %1 forward(s) over %2 frames (%3) - "
               "each forward is one blocking GPU call (~118 MiB/frame); "
               "expect a while without per-frame progress.")
                    .arg(chunkStarts.size())
                    .arg(viewCount)
                    .arg(m_settings.responsiveStreaming
                                 ? QStringLiteral("responsive streaming")
                                 : QStringLiteral("official single-pass")));
    for (size_t ci = 0; ci < chunkStarts.size(); ++ci) {
        if (ci > 0 && aicore_cancel_requested()) {
            emit logMessage(tr("[MapAnything] Cancelled between chunks."));
            return false;
        }
        const int st = chunkStarts[ci];
        const int count = std::min(K, viewCount - st);
        const int rc = aicore_map_anything_run_images(m_pendingCtx,
                                                      views.data() + st, count);
        if (rc != 0) {
            const char* err = aicore_map_anything_last_error(m_pendingCtx);
            emit logMessage(tr("[Error] Inference failed: %1")
                                    .arg(err != nullptr
                                                 ? QString::fromUtf8(err)
                                                 : QStringLiteral("unknown")));
            return false;
        }
        if (height == 0) {
            m_result.architecture = QString::fromUtf8(
                    aicore_map_anything_result_architecture(m_pendingCtx));
            m_result.resolvedDevice = QString::fromUtf8(
                    aicore_map_anything_context_device(m_pendingCtx));
            height = aicore_map_anything_result_height(m_pendingCtx);
            width = aicore_map_anything_result_width(m_pendingCtx);
        }
        aicore_pipeline_timings pipeline{};
        if (aicore_map_anything_last_pipeline_timings(m_pendingCtx,
                                                      &pipeline) == 0) {
            m_result.e2eMs += pipeline.e2e_ms;
        }

        // ---- chunk pose/centers FIRST: the Sim3 into the global frame
        // must be fitted BEFORE any point enters the pool, or chunk 1..N
        // land unaligned (the mis-registration blur).
        const float* depthArr = aicore_map_anything_result_depth(m_pendingCtx);
        const float* poseArr = aicore_map_anything_result_pose(m_pendingCtx);
        const int32_t poseStride =
                aicore_map_anything_result_pose_stride(m_pendingCtx);
        const size_t plane = (size_t)height * width;
        std::vector<float> centers((size_t)count * 3, 0.0f);
        if (poseArr != nullptr && poseStride == 9) {
            // pose_enc is camera-from-world. Its camera center is -R^T*T.
            for (int i = 0; i < count; ++i) {
                float R[9], t[3];
                MapAnythingGeometry::pose_enc_to_w2c(poseArr + (size_t)i * 9, R,
                                                     t);
                for (int r = 0; r < 3; ++r) {
                    centers[(size_t)i * 3 + r] =
                            -(R[r] * t[0] + R[3 + r] * t[1] + R[6 + r] * t[2]);
                }
            }
        } else if (poseArr != nullptr && poseStride == 16) {
            for (int i = 0; i < count; ++i) {
                // c2w row-major: translation in the flat 3/7/11 slots.
                centers[(size_t)i * 3 + 0] = poseArr[(size_t)i * 16 + 3];
                centers[(size_t)i * 3 + 1] = poseArr[(size_t)i * 16 + 7];
                centers[(size_t)i * 3 + 2] = poseArr[(size_t)i * 16 + 11];
            }
        }

        // Robust Horn Sim3 on the OVERLAP camera centers (frames this chunk
        // shares with an earlier chunk): maps this chunk into chunk 0's
        // frame (upstream "chain" mode). Chunk 0 IS the global frame.
        float sG = 1.0f;
        std::vector<float> RG = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> tG = {0, 0, 0};
        if (ci > 0) {
            const int overlap = std::min(2, count);
            std::vector<float> src, dst;
            for (int i = 0; i < overlap; ++i) {
                const int g = st + i;
                if (centersGlobal[(size_t)g * 3] >= 0.0f) {
                    src.insert(src.end(), centers.begin() + (size_t)i * 3,
                               centers.begin() + (size_t)(i + 1) * 3);
                    dst.insert(dst.end(), centersGlobal.begin() + (size_t)g * 3,
                               centersGlobal.begin() + (size_t)(g + 1) * 3);
                }
            }
            if (src.size() >= 9) {
                float sO = 1.0f, RO[9], tO[3];
                if (MapAnythingGeometry::robust_sim3(src.data(), dst.data(),
                                                     src.size() / 3, &sO, RO,
                                                     tO) == 0) {
                    sG = sO;
                    RG.assign(RO, RO + 9);
                    tG.assign(tO, tO + 3);
                }
            }
            // Align this chunk's centers so later chunks anchor on them.
            MapAnythingGeometry::apply_sim3(RG.data(), tG.data(), sG,
                                            centers.data(), count,
                                            centers.data());
        }
        chunkSim3[ci] = {sG,    RG[0], RG[1], RG[2], RG[3], RG[4], RG[5],
                         RG[6], RG[7], RG[8], tG[0], tG[1], tG[2]};

        // Official depth-edge masks (3x3 depth-jump discontinuities) for
        // every view of this chunk.
        std::vector<std::vector<uint8_t>> edgeMasks;
        if (m_settings.depthEdgeRtol > 0.0f && depthArr != nullptr) {
            edgeMasks.resize((size_t)count);
            for (int i = 0; i < count; ++i) {
                edgeMasks[(size_t)i].assign(plane, 0);
                MapAnythingGeometry::depth_edge_mask(
                        depthArr + (size_t)i * plane, height, width,
                        m_settings.depthEdgeRtol, edgeMasks[(size_t)i].data());
            }
        }

        // ---- per-view outputs: ONLY the first chunk covering a frame
        // produces it (overlap frames anchor the Sim3 but must not yield a
        // second copy of their points/raster/camera - the duplicates were
        // the 48-raster / 117%-progress / double-image defects).
        const float* confArr = aicore_map_anything_result_conf(m_pendingCtx);
        const float* maskArr = aicore_map_anything_result_mask(m_pendingCtx);
        const bool rawLogitArch =
                m_result.architecture == QStringLiteral("pi3") ||
                m_result.architecture == QStringLiteral("pi3x");
        for (int i = 0; i < count; ++i) {
            const int g = st + i;
            if (viewBegin[(size_t)g] != SIZE_MAX) continue;  // already owned

            // cloud (Sim3-aligned)
            if (m_settings.addPointCloud) {
                std::vector<float> pts, cf;
                std::vector<quint8> cols;
                appendViewCloud(i, g, height, width, sources, pts, cols, cf,
                                edgeMasks.empty()
                                        ? nullptr
                                        : edgeMasks[(size_t)i].data());
                if (sG != 1.0f || tG[0] != 0.0f || tG[1] != 0.0f ||
                    tG[2] != 0.0f) {
                    MapAnythingGeometry::apply_sim3(RG.data(), tG.data(), sG,
                                                    pts.data(), pts.size() / 3,
                                                    pts.data());
                }
                viewBegin[(size_t)g] = gPts.size() / 3;
                gPts.insert(gPts.end(), pts.begin(), pts.end());
                gCols.insert(gCols.end(), cols.begin(), cols.end());
                gConf.insert(gConf.end(), cf.begin(), cf.end());
                viewEnd[(size_t)g] = gPts.size() / 3;
            }
            emit progressUpdate(++step, totalSteps);

            // rasters (same crop contract as before)
            if (m_settings.addDepthRasters && depthArr != nullptr) {
                int32_t vw = width, vh = height;
                aicore_map_anything_result_view_valid_size(m_pendingCtx, i, &vw,
                                                           &vh);
                vw = std::min(vw, width);
                vh = std::min(vh, height);
                const int ox = (width - vw) / 2, oy = (height - vh) / 2;
                const float* vd = depthArr + (size_t)i * plane;
                const float* vm =
                        maskArr ? maskArr + (size_t)i * plane : nullptr;
                std::vector<float> cropD, cropC;
                cropD.reserve((size_t)vw * vh);
                cropC.reserve((size_t)vw * vh);
                float minD = 0, maxD = 0;
                for (int y = 0; y < vh; ++y) {
                    for (int x = 0; x < vw; ++x) {
                        const float d = vd[(size_t)(oy + y) * width + (ox + x)];
                        cropD.push_back(d);
                        if (std::isfinite(d) && d > 0) {
                            minD = std::isfinite(minD) && minD > 0
                                           ? std::min(minD, d)
                                           : d;
                            maxD = std::max(maxD, d);
                        }
                        if (confArr != nullptr) {
                            const float c = confArr[(size_t)i * plane +
                                                    (size_t)(oy + y) * width +
                                                    (ox + x)];
                            cropC.push_back(
                                    rawLogitArch
                                            ? MapAnythingGeometry::sigmoid(c)
                                    : c > 1.0f ? (c - 1.0f) / c
                                               : 0.0f);
                        }
                    }
                }
                MapAnythingViewResult view;
                view.width = vw;
                view.height = vh;
                for (float v : cropD) view.depthMap.append(v);
                view.depthImage =
                        colorizeDepth(cropD.data(), vw, vh, minD, maxD);
                if (confArr != nullptr) {
                    view.confImage =
                            colorizeDepth(cropC.data(), vw, vh, 0.0f, 1.0f);
                }
                if (vm != nullptr) {
                    QImage m(vw, vh, QImage::Format_Grayscale8);
                    for (int y = 0; y < vh; ++y) {
                        uchar* line = m.scanLine(y);
                        for (int x = 0; x < vw; ++x) {
                            const float mk =
                                    vm[(size_t)(oy + y) * width + (ox + x)];
                            line[x] = mk >= 0.5f ? uchar(255) : uchar(0);
                        }
                    }
                    view.maskImage = m;
                }
                view.sourcePath = m_settings.imagePaths[g];
                m_result.views.append(view);
            }

            // Camera actors consume c2w. Invert the upstream pose_enc
            // camera-from-world matrix [R|T] before applying chunk alignment.
            if (poseArr != nullptr && poseStride == 9) {
                const float* pp = poseArr + (size_t)i * 9;
                float R[9], t[3], k[4];
                MapAnythingGeometry::pose_enc_to_w2c(pp, R, t);
                MapAnythingGeometry::fov_to_intrinsics(pp[7], pp[8], width,
                                                       height, k);
                float c2w[16] = {};
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) c2w[r * 4 + c] = R[c * 3 + r];
                    c2w[r * 4 + 3] =
                            -(R[r] * t[0] + R[3 + r] * t[1] + R[6 + r] * t[2]);
                }
                c2w[15] = 1.0f;
                for (int e = 0; e < 16; ++e) rawCamPoses16.push_back(c2w[e]);
                m_result.camFocalPx.append(k[0]);
                camChunkOf.push_back((int)ci);
                m_result.camWidth = width;
                m_result.camHeight = height;
                m_result.hasCameras = true;
            }
        }

        // mapanything metric scale / text embedding: identical across
        // chunks; take them from the last pass.
        if (ci + 1 == chunkStarts.size()) {
            if (const float* scale =
                        aicore_map_anything_result_scale(m_pendingCtx)) {
                m_result.hasMetricScale = true;
                m_result.metricScale = scale[0];
            }
            int32_t embDim = 0;
            const float* emb = aicore_map_anything_result_text_embedding(
                    m_pendingCtx, &embDim);
            if (emb != nullptr && embDim > 0) {
                m_result.textEmbedding =
                        QByteArray(reinterpret_cast<const char*>(emb),
                                   (int)embDim * (int)sizeof(float));
                m_result.textEmbeddingDim = embDim;
            }
        }
    }

    // ---- cameras: apply each chunk's Sim3 to the raw chunk-frame poses ----
    if (m_result.hasCameras) {
        const int nCam = (int)(rawCamPoses16.size() / 16);
        m_result.camPoses16.reserve((size_t)nCam * 16);
        for (int v = 0; v < nCam; ++v) {
            const int cid = camChunkOf[(size_t)v];
            const std::vector<float>& sim = chunkSim3[(size_t)cid];
            const float sG = sim[0];
            const float* RG = &sim[1];
            const float* tG = &sim[10];
            float c2w[16];
            std::memcpy(c2w, &rawCamPoses16[(size_t)v * 16], sizeof(c2w));
            // c2w' = [s*Rg*R | Rg*(s*t) + tG]  (scale acts on the position)
            float newR[9], newT[3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    newR[r * 3 + c] = 0.0f;
                    for (int k2 = 0; k2 < 3; ++k2)
                        newR[r * 3 + c] += RG[r * 3 + k2] * c2w[k2 * 4 + c];
                }
                newT[r] = tG[r];
                for (int k2 = 0; k2 < 3; ++k2)
                    newT[r] += RG[r * 3 + k2] * (sG * c2w[k2 * 4 + 3]);
            }
            float out16[16] = {};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) out16[r * 4 + c] = newR[r * 3 + c];
                out16[r * 4 + 3] = newT[r];
            }
            out16[15] = 1.0f;
            for (int e = 0; e < 16; ++e) m_result.camPoses16.append(out16[e]);
        }
    }

    // ---- scene-wide official filter stack --------------------------------
    // floor = max(2.0, p<confPercentile>) over ALL raw confidences
    // (official predictions_to_glb); the UI minConf stays as an extra AND
    // gate on the family-unified [0,1) scale.
    float confFloor = -1.0f;
    if (!gConf.empty()) {
        confFloor = MapAnythingGeometry::official_conf_floor(
                gConf.data(), gConf.size(), m_settings.confPercentile);
        emit logMessage(tr("[MapAnything] Official filter: conf floor = %1 "
                           "(scene-wide p%2, min 2.0).")
                                .arg(confFloor, 0, 'f', 3)
                                .arg(m_settings.confPercentile));
    }
    const bool rawLogitArch = m_result.architecture == QStringLiteral("pi3") ||
                              m_result.architecture == QStringLiteral("pi3x");

    std::vector<float> fPts;
    std::vector<quint8> fCols;
    std::vector<float> fConf;
    fPts.reserve(gPts.size());
    fCols.reserve(gCols.size());
    fConf.reserve(gConf.size());
    for (size_t i = 0; i < gConf.size(); ++i) {
        const float c = gConf[i];
        if (c >= 0.0f && c < confFloor) continue;
        if (m_settings.minConf > 0.0f) {
            const float cn = rawLogitArch ? MapAnythingGeometry::sigmoid(c)
                             : c > 1.0f   ? (c - 1.0f) / c
                                          : 0.0f;
            if (c >= 0.0f && !(cn >= m_settings.minConf)) continue;
        }
        fPts.push_back(gPts[3 * i + 0]);
        fPts.push_back(gPts[3 * i + 1]);
        fPts.push_back(gPts[3 * i + 2]);
        fCols.push_back(gCols[3 * i + 0]);
        fCols.push_back(gCols[3 * i + 1]);
        fCols.push_back(gCols[3 * i + 2]);
        fConf.push_back(c);
    }

    // A voxel width has physical meaning only in a metric frame.  Applying
    // a scene-diagonal ratio to learned-frame VGGT/Pi3 coordinates silently
    // changes the exported geometry and makes the GUI much sparser than the
    // headless/upstream raw reconstruction.  Keep that geometry dense unless
    // this model actually publishes the metric scale needed by the 0.02 m
    // contract; this matches the headless export path.
    if (m_settings.voxelSize > 0.0f && m_result.hasMetricScale &&
        !fPts.empty()) {
        const float voxel = std::max(m_settings.voxelSize, 1e-6f);
        const size_t before = fPts.size() / 3;
        const size_t after = MapAnythingGeometry::conf_priority_voxel(
                fPts.data(), fCols.data(), fConf.data(), fPts.size() / 3,
                voxel);
        fPts.resize(3 * after);
        fCols.resize(3 * after);
        fConf.resize(after);
        emit logMessage(tr("[MapAnything] Metric voxel merge (size %1): %2 "
                           "-> %3 points.")
                                .arg(voxel, 0, 'f', 3)
                                .arg(before)
                                .arg(after));
    } else if (m_settings.voxelSize > 0.0f && !fPts.empty()) {
        emit logMessage(
                tr("[MapAnything] Voxel merge skipped: this model has "
                   "no metric scale; preserving dense learned-frame "
                   "geometry."));
    }

    m_result.points.reserve(fPts.size());
    m_result.colors.reserve(fCols.size());
    for (size_t i = 0; i < fPts.size(); ++i) {
        m_result.points.append(fPts[i]);
        m_result.colors.append(fCols[i]);
    }
    m_result.pointCount = (int)(fPts.size() / 3);

    // Opt-in per-view entities: the same filtered population sliced back
    // per view (no voxel merge - the merge is a fused-cloud stage).
    if (m_settings.addPerViewClouds) {
        for (int g = 0; g < viewCount; ++g) {
            QVector<float> pts;
            QVector<quint8> cols;
            if (viewBegin[(size_t)g] == SIZE_MAX) {
                m_result.perViewPoints.append(pts);
                m_result.perViewColors.append(cols);
                continue;
            }
            for (size_t i = viewBegin[(size_t)g]; i < viewEnd[(size_t)g]; ++i) {
                const float c = gConf[i];
                bool pass = c < 0.0f || c >= confFloor;
                if (pass && m_settings.minConf > 0.0f && c >= 0.0f) {
                    const float cn = rawLogitArch
                                             ? MapAnythingGeometry::sigmoid(c)
                                     : c > 1.0f ? (c - 1.0f) / c
                                                : 0.0f;
                    pass = cn >= m_settings.minConf;
                }
                if (!pass) continue;
                pts.append(gPts[3 * i + 0]);
                pts.append(gPts[3 * i + 1]);
                pts.append(gPts[3 * i + 2]);
                cols.append(gCols[3 * i + 0]);
                cols.append(gCols[3 * i + 1]);
                cols.append(gCols[3 * i + 2]);
            }
            m_result.perViewPoints.append(pts);
            m_result.perViewColors.append(cols);
        }
    }

    // ---- cameras: apply each chunk's Sim3 to the raw chunk-frame poses ----
    if (m_result.hasCameras) {
        const int nCam = (int)(rawCamPoses16.size() / 16);
        m_result.camPoses16.reserve((size_t)nCam * 16);
        for (int v = 0; v < nCam; ++v) {
            const int ci = camChunkOf[(size_t)v];
            const std::vector<float>& sim = chunkSim3[(size_t)ci];
            const float sG = sim[0];
            const float* RG = &sim[1];
            const float* tG = &sim[10];
            float c2w[16];
            std::memcpy(c2w, &rawCamPoses16[(size_t)v * 16], sizeof(c2w));
            // c2w' = [s*Rg*R | Rg*(s*t) + tG]  (scale acts on the position)
            float newR[9], newT[3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    newR[r * 3 + c] = 0.0f;
                    for (int k2 = 0; k2 < 3; ++k2)
                        newR[r * 3 + c] += RG[r * 3 + k2] * c2w[k2 * 4 + c];
                }
                newT[r] = tG[r];
                for (int k2 = 0; k2 < 3; ++k2)
                    newT[r] += RG[r * 3 + k2] * (sG * c2w[k2 * 4 + 3]);
            }
            float out16[16] = {};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) out16[r * 4 + c] = newR[r * 3 + c];
                out16[r * 4 + 3] = newT[r];
            }
            out16[15] = 1.0f;
            for (int e = 0; e < 16; ++e) m_result.camPoses16.append(out16[e]);
        }
    }

    // Percentile-free stats kept compact in the JSON envelope (compat/export
    // boundary only — the typed arrays above are the hot path).
    m_result.resultJson = QString(QStringLiteral("{\"architecture\":\"%1\","
                                                 "\"views\":%2,\"height\":%3,"
                                                 "\"width\":%4,\"e2e_ms\":%5,"
                                                 "\"points\":%6}"))
                                  .arg(m_result.architecture)
                                  .arg(m_result.viewCount)
                                  .arg(height)
                                  .arg(width)
                                  .arg(m_result.e2eMs, 0, 'f', 1)
                                  .arg(m_result.pointCount)
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
