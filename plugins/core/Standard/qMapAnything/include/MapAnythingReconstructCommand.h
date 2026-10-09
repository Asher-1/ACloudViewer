// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless multi-view reconstruction command (-MAP_ANYTHING_RECONSTRUCT),
// mirroring the SAM3D_GENERATE offscreen pattern: one child process loads
// the model once and runs the AICore map_anything pipeline over 2..16
// input images, exporting a merged world-frame colored point cloud (binary
// PLY) plus a RESULT_JSON manifest. Model resolution: an existing file
// path wins; otherwise the token is looked up in the published catalog and
// the shared cache dir (…/extract/map_anything_models). Headless runs do
// NOT trigger downloads — the JSON-RPC layer and the dialog own that flow.

#pragma once

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <QtConcurrent>

#include "MapAnythingGeometry.h"
#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/map_anything_capi.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#endif  // AICore_ENABLED

// Command keyword (the -SILENT CLI token users type after the binary).
#define COMMAND_MAP_ANYTHING_RECONSTRUCT "MAP_ANYTHING_RECONSTRUCT"

struct CommandMapAnythingReconstruct : public ccCommandLineInterface::Command {
    CommandMapAnythingReconstruct()
        : ccCommandLineInterface::Command(
                  QObject::tr("MapAnything multi-view reconstruction"),
                  COMMAND_MAP_ANYTHING_RECONSTRUCT) {}

#ifdef AICore_ENABLED

    bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[MAP_ANYTHING_RECONSTRUCT]");

        QStringList images;
        QString outputDir;
        QString modelToken;
        QString device = QStringLiteral("auto");
        int imageSize = 0;
        QString resizeMode = QStringLiteral("balanced");
        float minConf = -1.0f;  // <0 = not given: seeded from the catalog
                                // entry default (its own conf_filter scale)
        bool perView = false;   // PER_VIEW 1: one PLY per input view (large)
        QString cloudPly = QStringLiteral("map_anything_cloud.ply");
        QString resultJson;
        QString rawOutputDir;  // RAW_OUTPUT_DIR: parity tensors/camera exports
        QString stageDumpDir;  // STAGE_DUMP_DIR: debug graph stage tensors
        QString gtMetaPath;    // GT_META: optional benchmark metric anchor
        int threads = 0;

        // ---- argument walk -------------------------------------------------
        auto take = [&cmd]() -> QString {
            return cmd.arguments().isEmpty() ? QString()
                                             : cmd.arguments().takeFirst();
        };
        while (!cmd.arguments().isEmpty()) {
            const QString token = cmd.arguments().takeFirst().toUpper();
            if (token == QStringLiteral("IMAGE")) {
                const QString path = take();
                if (path.isEmpty()) {
                    return cmd.error(QObject::tr("IMAGE requires a path"));
                }
                images << path;
            } else if (token == QStringLiteral("OUT_DIR")) {
                outputDir = take();
            } else if (token == QStringLiteral("MODEL")) {
                modelToken = take();
            } else if (token == QStringLiteral("DEVICE")) {
                device = take();
            } else if (token == QStringLiteral("IMAGE_SIZE")) {
                imageSize = take().toInt();
            } else if (token == QStringLiteral("RESIZE_MODE")) {
                resizeMode = take();
            } else if (token == QStringLiteral("MIN_CONF")) {
                minConf = take().toFloat();
            } else if (token == QStringLiteral("CLOUD_PLY")) {
                cloudPly = take();
            } else if (token == QStringLiteral("THREADS")) {
                threads = take().toInt();
            } else if (token == QStringLiteral("PER_VIEW")) {
                perView = take().toInt() != 0;
            } else if (token == QStringLiteral("RESULT_JSON")) {
                resultJson = take();
            } else if (token == QStringLiteral("RAW_OUTPUT_DIR")) {
                rawOutputDir = take();
            } else if (token == QStringLiteral("STAGE_DUMP_DIR")) {
                stageDumpDir = take();
            } else if (token == QStringLiteral("GT_META")) {
                gtMetaPath = take();
            } else {
                return cmd.error(
                        QObject::tr("Unknown argument after -%1: %2")
                                .arg(COMMAND_MAP_ANYTHING_RECONSTRUCT, token));
            }
        }
        if (images.size() < 2 || images.size() > 64 || outputDir.isEmpty()) {
            return cmd.error(
                    QObject::tr("MAP_ANYTHING_RECONSTRUCT needs 2..64 IMAGE "
                                "entries and an OUT_DIR (single-pass budget: "
                                "~118 MiB/frame f16@512, 38 frames = 8.75 "
                                "GiB on a 24 GB card)"));
        }
        if (!QDir().mkpath(outputDir)) {
            return cmd.error(QObject::tr("Cannot create output directory: %1")
                                     .arg(outputDir));
        }
        if (resultJson.isEmpty()) {
            resultJson = QDir(outputDir).filePath(
                    QStringLiteral("_map_anything_result.json"));
        } else if (QFileInfo(resultJson).isRelative()) {
            // A bare RESULT_JSON token anchors to OUT_DIR (like the PLY);
            // it must never land in the process CWD.
            resultJson = QDir(outputDir).filePath(resultJson);
        }
        if (!rawOutputDir.isEmpty() && QFileInfo(rawOutputDir).isRelative()) {
            rawOutputDir = QDir(outputDir).filePath(rawOutputDir);
        }
        if (!rawOutputDir.isEmpty() && !QDir().mkpath(rawOutputDir)) {
            return cmd.error(QObject::tr("Cannot create RAW_OUTPUT_DIR: %1")
                                     .arg(rawOutputDir));
        }
        if (!stageDumpDir.isEmpty() && QFileInfo(stageDumpDir).isRelative()) {
            stageDumpDir = QDir(outputDir).filePath(stageDumpDir);
        }
        if (!stageDumpDir.isEmpty() && !QDir().mkpath(stageDumpDir)) {
            return cmd.error(QObject::tr("Cannot create STAGE_DUMP_DIR: %1")
                                     .arg(stageDumpDir));
        }
        if (!gtMetaPath.isEmpty() && QFileInfo(gtMetaPath).isRelative()) {
            gtMetaPath = QDir(outputDir).filePath(gtMetaPath);
        }

        // ---- model resolution ----------------------------------------------

        // Official-pipeline guards/config from the catalog entry.

        MapAnythingModelEntry modelEntry;

        MapAnythingHelpers::findModelByFilename(modelToken, &modelEntry);
        if (minConf < 0.0f) minConf = modelEntry.defaultMinConf;

        if (modelEntry.variant == QStringLiteral("dust3r") &&
            images.size() != 2) {
            return cmd.error(QObject::tr(

                                     "DUSt3R is a pair-wise model: exactly 2 "
                                     "IMAGE tokens are "

                                     "required (got %1).")
                                     .arg(images.size()));
        }
        QString modelPath = modelToken;
        if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
            // Catalog lookup: a published filename resolves against the
            // shared AICore cache dir (…/extract/map_anything_models).
            QString filename = modelToken;
            if (filename.isEmpty()) {
                filename = QStringLiteral("vggt-omega-1b-512-q8_0.gguf");
            }
            char* cache = aicore_map_anything_model_cache_dir();
            const QString cacheDir =
                    QString::fromUtf8(cache != nullptr ? cache : "");
            std::free(cache);
            modelPath = QDir(cacheDir).filePath(filename);
        }
        if (!QFileInfo::exists(modelPath)) {
            return cmd.error(
                    QObject::tr("Model not cached: %1 (download it "
                                "through the qMapAnything dialog or pass "
                                "an explicit MODEL path)")
                            .arg(modelPath));
        }

        // ---- decode + borrow the views --------------------------------------
        struct PackedImage {
            QImage image;
            aicore_image_view view{};
        };
        std::vector<PackedImage> packed;
        packed.reserve(images.size());
        // Parallel decode: full-scene batches are tens of 6201x4135 PNGs;
        // serial decoding costs tens of seconds of wall time.
        {
            std::vector<QImage> decoded(images.size());
            std::vector<int> decodeIdx(images.size());
            for (int i = 0; i < (int)images.size(); ++i)
                decodeIdx[(size_t)i] = i;
            QtConcurrent::blockingMap(decodeIdx, [&](int i) {
                decoded[(size_t)i] = QImage(images[i]).convertToFormat(
                        QImage::Format_RGB888);
            });
            for (size_t i = 0; i < (size_t)images.size(); ++i) {
                if (decoded[i].isNull()) {
                    return cmd.error(QObject::tr("Failed to decode image: %1")
                                             .arg(images[(int)i]));
                }
                PackedImage p;
                p.image = decoded[i];
                p.view.data = p.image.constBits();
                p.view.width = p.image.width();
                p.view.height = p.image.height();
                p.view.row_stride_bytes = (size_t)p.image.bytesPerLine();
                p.view.format = AICORE_IMAGE_RGB8;
                packed.push_back(p);
            }
        }
        std::vector<aicore_image_view> views;
        views.reserve(packed.size());
        for (const PackedImage& p : packed) views.push_back(p.view);

        // ---- load + run
        // ------------------------------------------------------
        aicore_map_anything_options* opts = aicore_map_anything_options_new();
        aicore_map_anything_options_set_device(opts,
                                               device.toUtf8().constData());
        aicore_map_anything_options_set_threads(opts, threads);
        aicore_map_anything_options_set_image_size(opts, imageSize);
        aicore_map_anything_options_set_resize_mode(
                opts, resizeMode.toUtf8().constData());
        if (!stageDumpDir.isEmpty()) {
            aicore_map_anything_options_set_dump_dir(
                    opts, stageDumpDir.toUtf8().constData());
        }
        aicore_map_anything_ctx* ctx = aicore_map_anything_load_opts(
                modelPath.toUtf8().constData(), opts);
        aicore_map_anything_options_free(opts);
        if (ctx == nullptr || aicore_map_anything_is_ready(ctx) == 0) {
            const char* message = ctx != nullptr
                                          ? aicore_map_anything_last_error(ctx)
                                          : nullptr;
            return cmd.error(
                    QObject::tr("Model load failed: %1")
                            .arg(QString::fromUtf8(message != nullptr ? message
                                                                      : "?")));
        }

        const int rc = aicore_map_anything_run_images(ctx, views.data(),
                                                      (int32_t)views.size());
        if (rc != 0) {
            const char* message = aicore_map_anything_last_error(ctx);
            aicore_map_anything_free(ctx);
            return cmd.error(
                    QObject::tr("Inference failed: %1")
                            .arg(QString::fromUtf8(message != nullptr ? message
                                                                      : "?")));
        }

        const int32_t viewCount = aicore_map_anything_result_view_count(ctx);
        const int32_t height = aicore_map_anything_result_height(ctx);
        const int32_t width = aicore_map_anything_result_width(ctx);
        const QString architecture =
                QString::fromUtf8(aicore_map_anything_result_architecture(ctx));
        const float* depth = aicore_map_anything_result_depth(ctx);
        const float* conf = aicore_map_anything_result_conf(ctx);
        const float* maskArr = aicore_map_anything_result_mask(ctx);
        const float* world = aicore_map_anything_result_points(ctx);
        const float* local = aicore_map_anything_result_local_points(ctx);
        const float* pose = aicore_map_anything_result_pose(ctx);
        const int32_t poseStride = aicore_map_anything_result_pose_stride(ctx);

        // Optional raw contract export: these files retain the exact AICore
        // result layout and are directly comparable to upstream vggt-cli bins.
        // Geometry/DB materialization never feeds back into this export.
        if (!rawOutputDir.isEmpty()) {
            auto writeRaw = [&](const QString& name, const void* data,
                                size_t bytes) -> bool {
                if (data == nullptr || bytes == 0) return true;
                QFile file(QDir(rawOutputDir).filePath(name));
                return file.open(QIODevice::WriteOnly | QIODevice::Truncate) &&
                       file.write(static_cast<const char*>(data),
                                  static_cast<qint64>(bytes)) ==
                               static_cast<qint64>(bytes);
            };
            const size_t denseCount = static_cast<size_t>(viewCount) *
                                      static_cast<size_t>(height) * width;
            int32_t rawTextEmbeddingDim = 0;
            const float* rawTextEmbedding =
                    aicore_map_anything_result_text_embedding(
                            ctx, &rawTextEmbeddingDim);
            const float* rawMetricScale = aicore_map_anything_result_scale(ctx);
            // Export every typed family output under the exact vggt-cli file
            // names. Missing heads intentionally produce no file; parity must
            // therefore check presence per architecture rather than treating
            // a synthesized zero tensor as a model result.
            if (!writeRaw(QStringLiteral("pose.bin"), pose,
                          static_cast<size_t>(viewCount) * poseStride *
                                  sizeof(float)) ||
                !writeRaw(QStringLiteral("depth.bin"), depth,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("depth_conf.bin"), conf,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("conf.bin"), conf,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("points.bin"), world,
                          denseCount * 3 * sizeof(float)) ||
                !writeRaw(QStringLiteral("local_points.bin"), local,
                          denseCount * 3 * sizeof(float)) ||
                !writeRaw(QStringLiteral("mask.bin"), maskArr,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("scale.bin"), rawMetricScale,
                          rawMetricScale != nullptr ? sizeof(float) : 0) ||
                !writeRaw(QStringLiteral("text_embedding.bin"),
                          rawTextEmbedding,
                          rawTextEmbedding != nullptr
                                  ? static_cast<size_t>(rawTextEmbeddingDim) *
                                            sizeof(float)
                                  : 0)) {
                aicore_map_anything_free(ctx);
                return cmd.error(
                        QObject::tr("Cannot write raw parity outputs: %1")
                                .arg(rawOutputDir));
            }
            if (pose != nullptr && poseStride == 9) {
                std::vector<float> c2w(static_cast<size_t>(viewCount) * 16,
                                       0.0f);
                for (int32_t v = 0; v < viewCount; ++v) {
                    float R[9], t[3];
                    MapAnythingGeometry::pose_enc_to_w2c(
                            pose + static_cast<size_t>(v) * 9, R, t);
                    float* M = c2w.data() + static_cast<size_t>(v) * 16;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) M[r * 4 + c] = R[c * 3 + r];
                        M[r * 4 + 3] = -(R[r] * t[0] + R[3 + r] * t[1] +
                                         R[6 + r] * t[2]);
                    }
                    M[15] = 1.0f;
                }
                if (!writeRaw(QStringLiteral("camera_to_world.bin"), c2w.data(),
                              c2w.size() * sizeof(float))) {
                    aicore_map_anything_free(ctx);
                    return cmd.error(QObject::tr("Cannot write c2w exports: %1")
                                             .arg(rawOutputDir));
                }
            }
        }

        // ---- world-frame cloud assembly (same contract as the worker) -------
        struct CloudPoint {
            float xyz[3];
            unsigned char rgb[3];
            float confidence;
            int32_t viewIndex;
        };
        std::vector<CloudPoint> cloud;
        bool hasGtAlignment = false;
        float gtScale = 1.0f;
        float gtRotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        float gtTranslation[3] = {0, 0, 0};
        const bool pi3Family = (poseStride == 16);
        const bool vggtFamily = (poseStride == 9);
        const bool dust3r = (architecture == QStringLiteral("dust3r"));
        const size_t plane = (size_t)height * width;

        // Official end-to-end filter stack: per-view depth-edge masks and a
        // SCENE-WIDE raw-confidence pool (floor = max(2.0, p20) over the
        // whole pass, applied after assembly - predictions_to_glb).
        constexpr float kDepthEdgeRtol = 0.03f;
        std::vector<std::vector<uint8_t>> edgeMasks;
        if (depth != nullptr) {
            edgeMasks.resize((size_t)viewCount);
            for (int32_t v = 0; v < viewCount; ++v) {
                edgeMasks[(size_t)v].assign((size_t)height * width, 0);
                MapAnythingGeometry::depth_edge_mask(
                        depth + (size_t)v * height * width, height, width,
                        kDepthEdgeRtol, edgeMasks[(size_t)v].data());
            }
        }
        std::vector<float> allConf;
        for (int32_t v = 0; v < viewCount; ++v) {
            // Cloud source decision per architecture — SAME contract as the
            // GUI worker (MapAnythingWorker::appendViewCloud): dust3r keeps
            // its shared-frame local points, mapanything/vggt-1b use the
            // published world points, pi3/pi3x lift local points by the c2w
            // pose, and the vggt family UNPROJECTS the depth map through the
            // per-view FoV intrinsics + pose_enc extrinsic. Never treat a
            // camera-frame local pointmap as a world-frame cloud: every view
            // would then be stacked around its own camera origin (the
            // frustum-crossing sheet defect).
            const float* points = dust3r ? local
                                         : (world       ? world
                                            : pi3Family ? local
                                                        : nullptr);
            float w2cR[9], w2cT[3], k[4];
            const bool unproject =
                    depth != nullptr && vggtFamily && pose != nullptr;
            const bool lift = pi3Family && local != nullptr &&
                              pose != nullptr && points == local;
            const float* poseRow = pose + (size_t)v * 9;
            if (unproject) {
                MapAnythingGeometry::pose_enc_to_w2c(poseRow, w2cR, w2cT);
                MapAnythingGeometry::fov_to_intrinsics(poseRow[7], poseRow[8],
                                                       width, height, k);
            }
            int32_t vw = width, vh = height;
            aicore_map_anything_result_view_valid_size(ctx, v, &vw, &vh);
            vw = std::min(vw, width);
            vh = std::min(vh, height);
            const int vpx0 = (width - vw) / 2, vpy0 = (height - vh) / 2;
            // Per-view raw-confidence percentile threshold (official
            // MapAnything semantics; only used by percentile variants).
            float pctThreshold = 0.0f;
            if (modelEntry.confFilter == QStringLiteral("percentile") &&
                conf != nullptr) {
                std::vector<float> vals;
                vals.reserve((size_t)vw * vh);
                for (int y2 = vpy0; y2 < vpy0 + vh; ++y2)
                    for (int x2 = vpx0; x2 < vpx0 + vw; ++x2)
                        vals.push_back(conf[(size_t)v * height * width +
                                            (size_t)y2 * width + x2]);
                if (!vals.empty()) {
                    const size_t drop =
                            (size_t)std::min(99.0f, std::max(0.0f, minConf)) /
                            100.0f * vals.size();
                    auto it = vals.begin() + std::min(drop, vals.size() - 1);
                    std::nth_element(vals.begin(), it, vals.end());
                    pctThreshold = *it;
                }
            }
            for (int y = vpy0; y < vpy0 + vh; ++y) {
                for (int x = vpx0; x < vpx0 + vw; ++x) {
                    const size_t idx = (size_t)y * width + x;
                    if (maskArr != nullptr &&
                        maskArr[(size_t)v * plane + (size_t)y * width + x] <
                                0.5f) {
                        continue;  // mapanything validity mask: out-of-mask
                    }
                    if (!edgeMasks.empty() &&
                        edgeMasks[(size_t)v][(size_t)y * width + x]) {
                        continue;  // official depth-edge discontinuity
                    }
                    // RAW confidence rides with the point; the scene-wide
                    // floor + UI gate are applied after the whole pass.
                    const float rawConfidence =
                            conf != nullptr ? conf[(size_t)v * plane + idx]
                                            : -1.0f;
                    float world3[3];
                    if (unproject) {
                        MapAnythingGeometry::unproject_depth_pixel(
                                w2cR, w2cT, k, x, y,
                                depth[(size_t)v * plane + idx], world3);
                    } else if (lift) {
                        MapAnythingGeometry::transform_by_c2w(
                                pose + (size_t)v * 16,
                                points + ((size_t)v * plane + idx) * 3, world3);
                    } else if (points != nullptr) {
                        const size_t pidx = (size_t)v * plane + idx;
                        world3[0] = points[pidx * 3 + 0];
                        world3[1] = points[pidx * 3 + 1];
                        world3[2] = points[pidx * 3 + 2];
                    } else {
                        continue;
                    }
                    if (!std::isfinite(world3[0]) ||
                        !std::isfinite(world3[1]) ||
                        !std::isfinite(world3[2])) {
                        continue;
                    }
                    CloudPoint p;
                    p.xyz[0] = world3[0];
                    p.xyz[1] = world3[1];
                    p.xyz[2] = world3[2];
                    // Nearest source-image color sample (through the
                    // view's valid rect, matching the unproject space).
                    const QImage& src = packed[v].image;
                    const int su =
                            std::min(src.width() - 1, (x - vpx0) * src.width() /
                                                              std::max(1, vw));
                    const int sv = std::min(
                            src.height() - 1,
                            (y - vpy0) * src.height() / std::max(1, vh));
                    const QRgb rgb = src.pixel(su, sv);
                    p.rgb[0] = (unsigned char)qRed(rgb);
                    p.rgb[1] = (unsigned char)qGreen(rgb);
                    p.rgb[2] = (unsigned char)qBlue(rgb);
                    p.confidence = rawConfidence;
                    p.viewIndex = v;
                    cloud.push_back(p);
                    allConf.push_back(rawConfidence);
                }
            }
        }

        // Optional benchmark-only metric anchor. Model outputs remain in
        // their native learned frame by default; a 2 cm voxel is meaningful
        // only after a caller supplies metric reference camera centers.
        if (!gtMetaPath.isEmpty()) {
            QFile metaFile(gtMetaPath);
            if (!metaFile.open(QIODevice::ReadOnly)) {
                aicore_map_anything_free(ctx);
                return cmd.error(
                        QObject::tr("Cannot read GT_META: %1").arg(gtMetaPath));
            }
            const QJsonDocument metaDoc =
                    QJsonDocument::fromJson(metaFile.readAll());
            const QJsonArray frames =
                    metaDoc.object().value("frames").toArray();
            QHash<QString, QVector<float>> gtCenters;
            for (const QJsonValue& value : frames) {
                const QJsonObject frame = value.toObject();
                const QJsonArray matrix =
                        frame.value("transform_matrix").toArray();
                if (matrix.size() < 3) continue;
                QVector<float> center(3);
                bool valid = true;
                for (int r = 0; r < 3; ++r) {
                    const QJsonArray row = matrix[r].toArray();
                    if (row.size() < 4) {
                        valid = false;
                        break;
                    }
                    center[r] = static_cast<float>(row[3].toDouble());
                }
                if (valid)
                    gtCenters.insert(frame.value("frame_name").toString(),
                                     center);
            }
            std::vector<float> sourceCenters, targetCenters;
            if (pose != nullptr && poseStride == 9) {
                for (int32_t v = 0; v < viewCount; ++v) {
                    const QVector<float> target = gtCenters.value(
                            QFileInfo(images[v]).completeBaseName());
                    if (target.size() != 3) continue;
                    float R[9], t[3];
                    MapAnythingGeometry::pose_enc_to_w2c(
                            pose + static_cast<size_t>(v) * 9, R, t);
                    for (int r = 0; r < 3; ++r) {
                        sourceCenters.push_back(-(R[r] * t[0] +
                                                  R[3 + r] * t[1] +
                                                  R[6 + r] * t[2]));
                        targetCenters.push_back(target[r]);
                    }
                }
            }
            if (sourceCenters.size() < 9 ||
                MapAnythingGeometry::robust_sim3(
                        sourceCenters.data(), targetCenters.data(),
                        sourceCenters.size() / 3, &gtScale, gtRotation,
                        gtTranslation) != 0) {
                aicore_map_anything_free(ctx);
                return cmd.error(QObject::tr("GT_META has insufficient usable "
                                             "camera centers: %1")
                                         .arg(gtMetaPath));
            }
            for (CloudPoint& point : cloud) {
                float aligned[3];
                MapAnythingGeometry::apply_sim3(gtRotation, gtTranslation,
                                                gtScale, point.xyz, 1, aligned);
                std::memcpy(point.xyz, aligned, sizeof(aligned));
            }
            hasGtAlignment = true;
        }

        // ---- scene-wide official filter stack --------------------------
        // floor = max(2.0, p20) over the whole pass (predictions_to_glb);
        // the catalog/UI minConf stays as an extra AND gate on the
        // family-unified [0,1) scale.
        if (!allConf.empty()) {
            const float confFloor = MapAnythingGeometry::official_conf_floor(
                    allConf.data(), allConf.size(), 20.0f);
            const bool rawLogit = pi3Family && !dust3r;
            std::vector<CloudPoint> filtered;
            filtered.reserve(cloud.size());
            for (size_t i = 0; i < cloud.size(); ++i) {
                const float c = allConf[i];
                if (c >= 0.0f && c < confFloor) continue;
                if (minConf > 0.0f && c >= 0.0f) {
                    const float cn = rawLogit ? MapAnythingGeometry::sigmoid(c)
                                     : c > 1.0f ? (c - 1.0f) / c
                                                : 0.0f;
                    if (!(cn >= minConf)) continue;
                }
                filtered.push_back(cloud[i]);
            }
            const size_t before = filtered.size();
            // Conf-priority voxel merge (official: per voxel keep the
            // highest-conf observation) - removes the multi-view
            // depth-noise band that reads as blur.
            {
                // The upstream 0.02 is a metric-world voxel. Without GT_META
                // this command has no legitimate model-to-metric scale, so do
                // not silently apply a dimensionally invalid metric threshold.
                const float voxel = hasGtAlignment ? 0.02f : 0.0f;
                std::vector<float> pts, cf;
                std::vector<uint8_t> cols;
                pts.reserve(filtered.size() * 3);
                cols.reserve(filtered.size() * 3);
                cf.reserve(filtered.size());
                for (const CloudPoint& p : filtered) {
                    pts.push_back(p.xyz[0]);
                    pts.push_back(p.xyz[1]);
                    pts.push_back(p.xyz[2]);
                    cols.push_back(p.rgb[0]);
                    cols.push_back(p.rgb[1]);
                    cols.push_back(p.rgb[2]);
                    cf.push_back(p.confidence);
                }
                std::vector<size_t> survivorIndices;
                MapAnythingGeometry::conf_priority_voxel(
                        pts.data(), cols.data(), cf.data(), filtered.size(),
                        voxel, &survivorIndices);
                std::vector<CloudPoint> voxelFiltered;
                voxelFiltered.reserve(survivorIndices.size());
                for (size_t index : survivorIndices) {
                    voxelFiltered.push_back(filtered[index]);
                }
                filtered.swap(voxelFiltered);
            }
            cmd.print(QObject::tr("official filter: conf floor %.3f, voxel "
                                  "0.02: %1 -> %2 points")
                              .arg(before)
                              .arg(filtered.size())
                              .arg(confFloor, 0, 'f', 3));
            cloud.swap(filtered);
        }

        // ---- export: binary PLY (XYZ + uchar RGB)
        // ----------------------------
        const QString plyPath = QDir(outputDir).absoluteFilePath(cloudPly);
        QFile plyFile(plyPath);
        if (!plyFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            aicore_map_anything_free(ctx);
            return cmd.error(QObject::tr("Cannot write %1").arg(plyPath));
        }
        {
            const QByteArray header =
                    QString("ply\n"
                            "format binary_little_endian 1.0\n"
                            "element vertex %1\n"
                            "property float x\n"
                            "property float y\n"
                            "property float z\n"
                            "property uchar red\n"
                            "property uchar green\n"
                            "property uchar blue\n"
                            "end_header\n")
                            .arg((quint64)cloud.size())
                            .toUtf8();
            plyFile.write(header);
            for (const CloudPoint& p : cloud) {
                plyFile.write(reinterpret_cast<const char*>(p.xyz),
                              sizeof(p.xyz));
                plyFile.write(reinterpret_cast<const char*>(p.rgb),
                              sizeof(p.rgb));
            }
        }
        plyFile.close();

        aicore_pipeline_timings timings{};
        double e2eMs = 0.0;
        if (aicore_map_anything_last_pipeline_timings(ctx, &timings) == 0) {
            e2eMs = timings.e2e_ms;
        }
        const QString resolvedDevice =
                QString::fromUtf8(aicore_map_anything_context_device(ctx));
        // Captured before the context is freed: mapanything metric scale
        // and the omega-256-text language embedding (sidecar artifact).
        float metricScale = 0.0f;
        const bool hasMetricScale =
                aicore_map_anything_result_scale(ctx) != nullptr;
        if (hasMetricScale) {
            metricScale = aicore_map_anything_result_scale(ctx)[0];
        }
        int32_t embDim = 0;
        const float* embPtr =
                aicore_map_anything_result_text_embedding(ctx, &embDim);
        const QByteArray embBytes(
                embPtr ? reinterpret_cast<const char*>(embPtr) : nullptr,
                embPtr ? (int)embDim * (int)sizeof(float) : 0);
        aicore_map_anything_free(ctx);

        // ---- RESULT_JSON manifest -------------------------------------------
        QJsonObject report;
        report["task"] = QStringLiteral("map_anything");
        report["model"] = modelPath;
        report["architecture"] = architecture;
        report["backend"] = resolvedDevice;
        report["views"] = static_cast<qint64>(viewCount);
        report["height"] = static_cast<qint64>(height);
        report["width"] = static_cast<qint64>(width);
        report["points"] = static_cast<qint64>(cloud.size());
        report["e2e_ms"] = e2eMs;
        report["ply"] = plyPath;
        report["min_conf"] = minConf;
        if (!stageDumpDir.isEmpty()) report["stage_dump_dir"] = stageDumpDir;
        if (hasGtAlignment) {
            report["gt_meta"] = gtMetaPath;
            report["gt_sim3"] = QJsonObject{{"scale", gtScale},
                                            {"rotation",
                                             QJsonArray {
                                                 gtRotation[0],
                                                 gtRotation[1],
                                                 gtRotation[2],
                                                 gtRotation[3],
                                                 gtRotation[4],
                                                 gtRotation[5],
                                                 gtRotation[6],
                                                 gtRotation[7],
                                                 gtRotation[8]
                                             }},
                                            { "translation",
                                              QJsonArray {
                                                  gtTranslation[0],
                                                  gtTranslation[1],
                                                  gtTranslation[2]
                                              } }};
        }
        if (hasMetricScale) {
            report["metric_scale"] = metricScale;
        }
        if (embDim > 0) {
            const double l2 = [](const QByteArray& b) {
                const float* e = reinterpret_cast<const float*>(b.constData());
                double acc = 0.0;
                for (int i = 0; i < b.size() / (int)sizeof(float); ++i)
                    acc += double(e[i]) * double(e[i]);
                return std::sqrt(acc);
            }(embBytes);
            report["text_embedding"] = QJsonObject{{"dim", embDim},
                                                   { "l2",
                                                     l2 }};
            const QString embPath = QDir(outputDir).absoluteFilePath(
                    QStringLiteral("text_embedding.bin"));
            QFile embFile(embPath);
            if (embFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                embFile.write(embBytes);
                embFile.close();
                report["text_embedding_bin"] = embPath;
            }
        }
        if (perView) {
            // One PLY per input view from the same final filtered/voxelized
            // population as the merged cloud. Point ownership must travel
            // through both filtering stages; filtering-time offsets are stale.
            QJsonArray perViewList;
            for (int32_t v = 0; v < viewCount; ++v) {
                std::vector<const CloudPoint*> viewPoints;
                for (const CloudPoint& point : cloud) {
                    if (point.viewIndex == v) viewPoints.push_back(&point);
                }
                const QString vp = QDir(outputDir).absoluteFilePath(
                        QStringLiteral("map_anything_view_%1.ply").arg(v + 1));
                QFile vf(vp);
                if (!vf.open(QIODevice::WriteOnly | QIODevice::Truncate))
                    continue;
                const QByteArray vh =
                        QString("ply\n"
                                "format binary_little_endian 1.0\n"
                                "element vertex %1\n"
                                "property float x\n"
                                "property float y\n"
                                "property float z\n"
                                "property uchar red\n"
                                "property uchar green\n"
                                "property uchar blue\n"
                                "end_header\n")
                                .arg((quint64)viewPoints.size())
                                .toUtf8();
                vf.write(vh);
                for (const CloudPoint* point : viewPoints) {
                    vf.write(reinterpret_cast<const char*>(point->xyz),
                             sizeof(point->xyz));
                    vf.write(reinterpret_cast<const char*>(point->rgb),
                             sizeof(point->rgb));
                }
                vf.close();
                perViewList.append(vp);
                cmd.print(QObject::tr("view %1 -> %2 (%3 points)")
                                  .arg(v + 1)
                                  .arg(vp)
                                  .arg(viewPoints.size()));
            }
            report["per_view_ply"] = perViewList;
        }
        const QByteArray reportData =
                QJsonDocument(report).toJson(QJsonDocument::Indented);
        QFile jsonFile(resultJson);
        if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            return cmd.error(QObject::tr("Cannot write %1").arg(resultJson));
        }
        jsonFile.write(reportData);

        cmd.print(QObject::tr("reconstruction done: %1 points -> %2 (%3 ms)")
                          .arg(cloud.size())
                          .arg(plyPath)
                          .arg(e2eMs, 0, 'f', 0));
        cmd.print(QObject::tr("Result JSON: %1").arg(resultJson));
        return true;
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_MAP_ANYTHING_RECONSTRUCT));
    }
#endif  // AICore_ENABLED
};
