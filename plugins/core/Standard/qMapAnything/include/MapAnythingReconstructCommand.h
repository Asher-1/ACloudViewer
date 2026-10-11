// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless multi-view reconstruction command (-MAP_ANYTHING_RECONSTRUCT),
// mirroring the SAM3D_GENERATE offscreen pattern: one child process loads
// the model once and runs the AICore map_anything pipeline over >= 2
// input images, exporting a merged world-frame colored point cloud (binary
// PLY) plus a RESULT_JSON manifest. Chunking, Sim3 alignment, assembly and
// the scene filter are the shared MapAnythingReconstruction driver - the
// GUI worker runs the identical code, so MAX_FRAMES_PER_PASS K here equals
// the dialog's "max frames per pass" K. Model resolution: an existing file
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
#include "MapAnythingModelCatalog.h"
#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/map_anything_capi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "MapAnythingReconstruction.h"
#include "MapAnythingSceneMeta.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

// Command keyword (the -SILENT CLI token users type after the binary).
#define COMMAND_MAP_ANYTHING_RECONSTRUCT "MAP_ANYTHING_RECONSTRUCT"

struct CommandMapAnythingReconstruct : public ccCommandLineInterface::Command {
    CommandMapAnythingReconstruct()
        : ccCommandLineInterface::Command(
                  QObject::tr("MapAnything multi-view reconstruction"),
                  COMMAND_MAP_ANYTHING_RECONSTRUCT) {}

#ifdef AICore_ENABLED

    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"IMAGE", "list", "", "Input image; repeat at least twice"},
                {"OUT_DIR", "path", "", "Output directory (required)"},
                {"MODEL", "string", "vggt-omega-1b-512-q8_0.gguf",
                 "GGUF path or published catalog filename"},
                {"DEVICE", "string", "auto", "auto|cpu|cuda|vulkan|metal[:N]"},
                {"IMAGE_SIZE", "int", "0", "0 uses the catalog default"},
                {"RESIZE_MODE", "choice:balanced|max_size|square", "",
                 "Empty uses the catalog default"},
                {"MIN_CONF", "float", "-1",
                 "Negative uses the catalog confidence default"},
                {"CLOUD_PLY", "path", "map_anything_cloud.ply",
                 "Merged point-cloud path relative to OUT_DIR"},
                {"THREADS", "int", "0", "CPU threads (0 = backend default)"},
                {"PER_VIEW", "bool", "0", "Export one PLY per input view"},
                {"RESULT_JSON", "path", "_map_anything_result.json",
                 "Result manifest path"},
                {"RAW_OUTPUT_DIR", "path", "",
                 "Optional raw tensor and camera export directory"},
                {"STAGE_DUMP_DIR", "path", "",
                 "Optional debug graph-stage tensor directory"},
                {"GT_META", "path", "",
                 "Optional metric benchmark camera metadata"},
                {"REF_IMAGES_PATH", "path", "",
                 "Optional known camera positions, COLMAP model_aligner "
                 "text: '<image_name> X Y Z' per line"},
                {"REF_IS_GPS", "bool", "",
                 "Required with REF_IMAGES_PATH: 1 = WGS84 lat lon alt (to "
                 "local ENU, model_aligner --alignment_type enu), 0 = "
                 "cartesian X Y Z"},
                {"MAX_FRAMES_PER_PASS", "int", "64",
                 "Views per forward before overlap-aligned chunking"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        using ecvAICoreBatch::SchemaHelp;
        const SchemaHelp help = ecvAICoreBatch::handleSchemaHelp(
                cmd, COMMAND_MAP_ANYTHING_RECONSTRUCT,
                QStringLiteral("map_anything"),
                QStringLiteral("MapAnything multi-view reconstruction"),
                options());
        if (help != SchemaHelp::NotRequested) return help == SchemaHelp::Done;

        cmd.print("[MAP_ANYTHING_RECONSTRUCT]");

        QStringList images;
        QString outputDir;
        QString modelToken;
        QString device = QStringLiteral("auto");
        int imageSize = 0;
        QString resizeMode;     // empty = not given: catalog entry default
        float minConf = -1.0f;  // <0 = not given: seeded from the catalog
                                // entry default (its own conf_filter scale)
        bool perView = false;   // PER_VIEW 1: one PLY per input view (large)
        QString cloudPly = QStringLiteral("map_anything_cloud.ply");
        QString resultJson;
        QString rawOutputDir;   // RAW_OUTPUT_DIR: parity tensors/camera exports
        QString stageDumpDir;   // STAGE_DUMP_DIR: debug graph stage tensors
        QString gtMetaPath;     // GT_META: optional benchmark metric anchor
        QString refImagesPath;  // REF_IMAGES_PATH: user model_aligner file
        QString refIsGpsToken;  // REF_IS_GPS: required with REF_IMAGES_PATH
        int threads = 0;
        int maxFramesPerPass = 64;  // same default as the GUI worker

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
            } else if (token == QStringLiteral("REF_IMAGES_PATH")) {
                refImagesPath = take();
            } else if (token == QStringLiteral("REF_IS_GPS")) {
                refIsGpsToken = take();
            } else if (token == QStringLiteral("MAX_FRAMES_PER_PASS")) {
                maxFramesPerPass = take().toInt();
                if (maxFramesPerPass < 2) {
                    return cmd.error(
                            QObject::tr("MAX_FRAMES_PER_PASS must be >= 2"));
                }
            } else {
                return cmd.error(
                        QObject::tr("Unknown argument after -%1: %2")
                                .arg(COMMAND_MAP_ANYTHING_RECONSTRUCT, token));
            }
        }
        // No upper view bound: past MAX_FRAMES_PER_PASS the shared driver
        // streams overlapping chunks, exactly like the GUI worker.
        if (images.size() < 2 || outputDir.isEmpty()) {
            return cmd.error(
                    QObject::tr("MAP_ANYTHING_RECONSTRUCT needs >= 2 IMAGE "
                                "entries and an OUT_DIR (single-pass budget: "
                                "~118 MiB/frame f16@512; larger sets stream "
                                "in MAX_FRAMES_PER_PASS chunks)"));
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
        if (!refImagesPath.isEmpty() && QFileInfo(refImagesPath).isRelative()) {
            refImagesPath = QDir(outputDir).filePath(refImagesPath);
        }

        // ---- reference cameras (read before the model: fail fast) ---------
        // Same readers as the GUI worker. REF_IS_GPS has no default: COLMAP
        // defaults --ref_is_gps to 1 (and ECEF), so silently assuming either
        // value would misread a file copied from a COLMAP command line.
        MapAnythingSceneMeta::ReferenceCenters reference;
        bool refIsGps = false;
        if (!refImagesPath.isEmpty()) {
            if (!gtMetaPath.isEmpty()) {
                return cmd.error(QObject::tr(
                        "GT_META and REF_IMAGES_PATH are exclusive: give "
                        "one reference frame"));
            }
            if (!ecvAICoreBatch::parseBool(refIsGpsToken, &refIsGps)) {
                return cmd.error(QObject::tr(
                        "REF_IMAGES_PATH needs REF_IS_GPS 0 (cartesian X Y "
                        "Z) or 1 (WGS84 lat lon alt)"));
            }
            QString refError;
            if (!MapAnythingSceneMeta::loadColmapReferences(
                        refImagesPath, images, refIsGps, &reference,
                        &refError)) {
                return cmd.error(QObject::tr("Cannot use REF_IMAGES_PATH: %1")
                                         .arg(refError));
            }
        } else if (!refIsGpsToken.isEmpty()) {
            return cmd.error(QObject::tr("REF_IS_GPS needs REF_IMAGES_PATH"));
        } else if (!gtMetaPath.isEmpty()) {
            QString metaError;
            if (!MapAnythingSceneMeta::loadCenters(
                        gtMetaPath, images, &reference.centers, &metaError)) {
                return cmd.error(
                        QObject::tr("Cannot read GT_META: %1").arg(metaError));
            }
        }
        const bool shifted = reference.offset[0] != 0.0 ||
                             reference.offset[1] != 0.0 ||
                             reference.offset[2] != 0.0;
        // PLYs carry true coordinates in double when the frame is large.
        const double* plyOffset = shifted ? reference.offset : nullptr;

        // ---- model resolution ----------------------------------------------

        if (modelToken.isEmpty()) {
            modelToken = QStringLiteral("vggt-omega-1b-512-q8_0.gguf");
        }
        // Official-pipeline guards/config from the catalog entry, keyed by
        // file name so an explicit MODEL path gets the same defaults as the
        // dialog (which always resolves a catalog entry).
        MapAnythingModelEntry modelEntry;
        MapAnythingHelpers::findModelByFilename(
                QFileInfo(modelToken).fileName(), &modelEntry);
        if (minConf < 0.0f) minConf = modelEntry.defaultMinConf;
        if (resizeMode.isEmpty()) resizeMode = modelEntry.defaultResizeMode;
        if (resizeMode.isEmpty()) resizeMode = QStringLiteral("balanced");

        if (modelEntry.variant == QStringLiteral("dust3r") &&
            images.size() != 2) {
            return cmd.error(QObject::tr(

                                     "DUSt3R is a pair-wise model: exactly 2 "
                                     "IMAGE tokens are "

                                     "required (got %1).")
                                     .arg(images.size()));
        }
        QString modelPath = modelToken;
        if (!QFileInfo::exists(modelPath)) {
            // Catalog lookup: a published filename resolves against the
            // shared AICore cache dir (…/extract/map_anything_models).
            const QString filename = modelToken;
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
            // Const access only: a non-const QList::operator[] from the
            // pool threads may detach() concurrently.
            const QStringList& paths = images;
            QtConcurrent::blockingMap(decodeIdx, [&](int i) {
                decoded[(size_t)i] =
                        QImage(paths.at(i))
                                .convertToFormat(QImage::Format_RGB888);
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

        // ---- load + run (the SAME shared driver as the GUI worker) ----------
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
            if (ctx != nullptr) aicore_map_anything_free(ctx);
            return cmd.error(
                    QObject::tr("Model load failed: %1")
                            .arg(QString::fromUtf8(message != nullptr ? message
                                                                      : "?")));
        }
        // The context is freed on every exit path below.
        std::unique_ptr<aicore_map_anything_ctx,
                        void (*)(aicore_map_anything_ctx*)>
                ctxGuard(ctx, aicore_map_anything_free);

        MapAnythingReconstruction::Options ro;
        ro.maxFramesPerPass = maxFramesPerPass;
        ro.overlap = 2;
        ro.buildCloud = true;
        const int passes =
                (int)MapAnythingGeometry::plan_chunk_starts(
                        (int)views.size(),
                        std::min((int)views.size(), ro.maxFramesPerPass),
                        ro.overlap)
                        .size();
        // RAW_OUTPUT_DIR is the frames-contract export of ONE forward; a
        // streamed run has no single model output to compare.
        if (!rawOutputDir.isEmpty() && passes > 1) {
            return cmd.error(
                    QObject::tr("RAW_OUTPUT_DIR needs a single forward: %1 "
                                "views stream in %2 chunks at "
                                "MAX_FRAMES_PER_PASS %3")
                            .arg(views.size())
                            .arg(passes)
                            .arg(maxFramesPerPass));
        }
        MapAnythingReconstruction::Result rec;
        std::string runError;
        if (MapAnythingReconstruction::run(ctx, views.data(), (int)views.size(),
                                           ro, &rec, &runError) !=
            MapAnythingReconstruction::kOk) {
            return cmd.error(QObject::tr("Inference failed: %1")
                                     .arg(QString::fromStdString(runError)));
        }
        for (const std::string& note : rec.memoryNotes) {
            cmd.print(QObject::tr("device memory: %1 -> halving the pass")
                              .arg(QString::fromStdString(note)));
        }
        if (!rawOutputDir.isEmpty() && rec.chunks.size() > 1) {
            return cmd.error(
                    QObject::tr("RAW_OUTPUT_DIR needs a single forward, but "
                                "the device admitted only %1 views per pass")
                            .arg(rec.framesPerPass));
        }
        const QString architecture = QString::fromStdString(rec.architecture);

        // Optional raw contract export: these files retain the exact AICore
        // result layout and are directly comparable to upstream vggt-cli bins.
        // Single pass only (checked above), so ctx still holds that forward.
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
            const size_t denseCount = static_cast<size_t>(rec.views) *
                                      static_cast<size_t>(rec.height) *
                                      rec.width;
            const float* depth = aicore_map_anything_result_depth(ctx);
            const float* conf = aicore_map_anything_result_conf(ctx);
            int32_t rawTextEmbeddingDim = 0;
            const float* rawTextEmbedding =
                    aicore_map_anything_result_text_embedding(
                            ctx, &rawTextEmbeddingDim);
            const float* rawMetricScale = aicore_map_anything_result_scale(ctx);
            // Export every typed family output under the exact vggt-cli file
            // names. Missing heads intentionally produce no file; parity must
            // therefore check presence per architecture rather than treating
            // a synthesized zero tensor as a model result.
            if (!writeRaw(QStringLiteral("pose.bin"),
                          aicore_map_anything_result_pose(ctx),
                          static_cast<size_t>(rec.views) * rec.poseStride *
                                  sizeof(float)) ||
                !writeRaw(QStringLiteral("depth.bin"), depth,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("depth_conf.bin"), conf,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("conf.bin"), conf,
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("points.bin"),
                          aicore_map_anything_result_points(ctx),
                          denseCount * 3 * sizeof(float)) ||
                !writeRaw(QStringLiteral("local_points.bin"),
                          aicore_map_anything_result_local_points(ctx),
                          denseCount * 3 * sizeof(float)) ||
                !writeRaw(QStringLiteral("mask.bin"),
                          aicore_map_anything_result_mask(ctx),
                          denseCount * sizeof(float)) ||
                !writeRaw(QStringLiteral("scale.bin"), rawMetricScale,
                          rawMetricScale != nullptr ? sizeof(float) : 0) ||
                !writeRaw(QStringLiteral("text_embedding.bin"),
                          rawTextEmbedding,
                          rawTextEmbedding != nullptr
                                  ? static_cast<size_t>(rawTextEmbeddingDim) *
                                            sizeof(float)
                                  : 0)) {
                return cmd.error(
                        QObject::tr("Cannot write raw parity outputs: %1")
                                .arg(rawOutputDir));
            }
            // Driver c2w (stride 9 and 16 alike; chunk 0 frame == model
            // frame). dust3r publishes no pose: its rows are NaN, skip.
            const bool posed = !rec.c2w.empty() && std::isfinite(rec.c2w[0]);
            if (posed &&
                !writeRaw(QStringLiteral("camera_to_world.bin"), rec.c2w.data(),
                          rec.c2w.size() * sizeof(float))) {
                return cmd.error(QObject::tr("Cannot write c2w exports: %1")
                                         .arg(rawOutputDir));
            }
        }

        // Optional reference frame. Model outputs remain in their native
        // learned frame by default; a 2 cm voxel is meaningful only after a
        // caller supplies metric reference camera centers. Same anchor as
        // the GUI worker.
        const QString referencePath =
                !refImagesPath.isEmpty() ? refImagesPath : gtMetaPath;
        bool anchored = false;
        MapAnythingReconstruction::ReferenceAnchor anchor;
        if (!referencePath.isEmpty()) {
            std::string anchorError;
            if (!MapAnythingReconstruction::anchor_to_reference(
                        rec, reference.centers, &anchor, &anchorError)) {
                return cmd.error(
                        QObject::tr("Reference-frame anchor (%1): %2")
                                .arg(referencePath,
                                     QString::fromStdString(anchorError)));
            }
            anchored = true;
        }

        // ---- scene-wide official filter stack (shared with the GUI) --------
        MapAnythingReconstruction::Cloud& cloud = rec.cloud;
        MapAnythingReconstruction::FilterOptions fo;
        fo.minConf = minConf;
        const MapAnythingReconstruction::FilterReport fr =
                MapAnythingReconstruction::filter_scene(
                        cloud, rec.architecture, fo,
                        anchored || rec.hasMetricScale);
        if (!fr.ok) {
            return cmd.error(QObject::tr("Scene filter: %1")
                                     .arg(QString::fromStdString(fr.rule)));
        }
        const size_t exported =
                fr.voxelApplied ? fr.fused.size() : cloud.size();
        cmd.print(QObject::tr("official filter %1: %2 -> %3 points, voxel "
                              "%4: -> %5 points")
                          .arg(QString::fromStdString(fr.rule))
                          .arg(fr.input)
                          .arg(fr.kept)
                          .arg(fr.voxelApplied ? QString::number(fo.voxelSize)
                                               : QStringLiteral("off"))
                          .arg(exported));

        // ---- export: binary PLY (XYZ + uchar RGB) --------------------------
        const QString plyPath = QDir(outputDir).absoluteFilePath(cloudPly);
        if (!MapAnythingReconstruction::write_ply(
                    plyPath.toStdString(), cloud,
                    fr.voxelApplied ? &fr.fused : nullptr, plyOffset)) {
            return cmd.error(QObject::tr("Cannot write %1").arg(plyPath));
        }

        // ---- RESULT_JSON manifest -------------------------------------------
        QJsonObject report;
        report["schema"] = QStringLiteral("acloudviewer.aicore.batch/1");
        report["keyword"] = QStringLiteral(COMMAND_MAP_ANYTHING_RECONSTRUCT);
        report["task"] = QStringLiteral("map_anything");
        report["succeeded"] = 1;
        report["failed"] = 0;
        report["model"] = modelPath;
        report["architecture"] = architecture;
        report["backend"] = QString::fromStdString(rec.device);
        report["views"] = static_cast<qint64>(rec.views);
        report["height"] = static_cast<qint64>(rec.height);
        report["width"] = static_cast<qint64>(rec.width);
        report["points"] = static_cast<qint64>(exported);
        report["e2e_ms"] = rec.e2eMs;
        report["ply"] = plyPath;
        report["min_conf"] = minConf;
        report["resize_mode"] = resizeMode;
        report["max_frames_per_pass"] = maxFramesPerPass;
        report["frames_per_pass"] = rec.framesPerPass;
        if (!rec.memoryNotes.empty()) {
            QJsonArray notes;
            for (const std::string& note : rec.memoryNotes)
                notes.append(QString::fromStdString(note));
            report["memory_downshifts"] = notes;
        }
        QJsonArray chunkList;
        for (const MapAnythingReconstruction::Chunk& c : rec.chunks) {
            chunkList.append(QJsonObject{
                    {"start", c.start},
                    {"count", c.count},
                    {"scale", c.toGlobal.s},
                    {"correspondences", static_cast<qint64>(c.correspondences)},
                    {"residual_median", c.residualMedian},
                    { "e2e_ms",
                      c.e2eMs }});
        }
        report["chunks"] = chunkList;
        report["filter"] = QJsonObject{
                {"rule", QString::fromStdString(fr.rule)},
                {"floor", std::isfinite(fr.confFloor) ? QJsonValue(fr.confFloor)
                                                      : QJsonValue()},
                {"input", static_cast<qint64>(fr.input)},
                {"kept", static_cast<qint64>(fr.kept)},
                { "voxel",
                  fr.voxelApplied ? QJsonValue(fo.voxelSize) : QJsonValue() }};
        if (!stageDumpDir.isEmpty()) report["stage_dump_dir"] = stageDumpDir;
        if (anchored) {
            const MapAnythingGeometry::Sim3& gt = anchor.sim3;
            QJsonArray rot, trans;
            for (float v : gt.R) rot.append(v);
            for (float v : gt.t) trans.append(v);
            const QJsonObject sim3{
                    {"scale", gt.s},
                    {"rotation", rot},
                    {"translation", trans},
                    {"cameras", anchor.pairs},
                    {"center_residual_median", anchor.residualMedian},
                    { "cross_spread",
                      anchor.crossSpread }};
            if (refImagesPath.isEmpty()) {
                report["gt_meta"] = gtMetaPath;
                report["gt_sim3"] = sim3;
            } else {
                // Translation is in the local frame: true = local + offset.
                QJsonObject ref = sim3;
                ref["path"] = refImagesPath;
                ref["coordinates"] = refIsGps ? QStringLiteral("gps_enu")
                                              : QStringLiteral("xyz");
                ref["matched_images"] = reference.matched;
                ref["offset"] =
                        QJsonArray{reference.offset[0], reference.offset[1],
                                   reference.offset[2]};
                if (refIsGps) {
                    ref["enu_origin_lat_lon_alt"] = QJsonArray{
                            reference.enuOrigin[0], reference.enuOrigin[1],
                            reference.enuOrigin[2]};
                }
                report["reference_sim3"] = ref;
            }
            report["ply_xyz"] = shifted ? QStringLiteral("double")
                                        : QStringLiteral("float");
        }
        if (rec.hasMetricScale) report["metric_scale"] = rec.metricScale;
        if (!rec.textEmbedding.empty()) {
            double acc = 0.0;
            for (float e : rec.textEmbedding) acc += double(e) * double(e);
            report["text_embedding"] = QJsonObject{
                    {"dim", static_cast<qint64>(rec.textEmbedding.size())},
                    { "l2",
                      std::sqrt(acc) }};
            const QString embPath = QDir(outputDir).absoluteFilePath(
                    QStringLiteral("text_embedding.bin"));
            QFile embFile(embPath);
            const qint64 bytes =
                    (qint64)(rec.textEmbedding.size() * sizeof(float));
            if (embFile.open(QIODevice::WriteOnly | QIODevice::Truncate) &&
                embFile.write(
                        reinterpret_cast<const char*>(rec.textEmbedding.data()),
                        bytes) == bytes) {
                report["text_embedding_bin"] = embPath;
            }
        }
        if (perView) {
            // One PLY per input view from the confidence-kept population
            // (no voxel merge: that is a fused-cloud stage), exactly the GUI
            // per-view entities. cloud.view is ascending -> one linear pass.
            QJsonArray perViewList;
            size_t begin = 0;
            for (int v = 0; v < rec.views; ++v) {
                std::vector<size_t> idx;
                while (begin < cloud.size() && cloud.view[begin] == v) {
                    idx.push_back(begin++);
                }
                const QString vp = QDir(outputDir).absoluteFilePath(
                        QStringLiteral("map_anything_view_%1.ply").arg(v + 1));
                if (!MapAnythingReconstruction::write_ply(
                            vp.toStdString(), cloud, &idx, plyOffset)) {
                    return cmd.error(QObject::tr("Cannot write %1").arg(vp));
                }
                perViewList.append(vp);
                cmd.print(QObject::tr("view %1 -> %2 (%3 points)")
                                  .arg(v + 1)
                                  .arg(vp)
                                  .arg(idx.size()));
            }
            report["per_view_ply"] = perViewList;
        }
        const QByteArray reportData =
                QJsonDocument(report).toJson(QJsonDocument::Indented);
        QFile jsonFile(resultJson);
        if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate) ||
            jsonFile.write(reportData) != reportData.size()) {
            return cmd.error(QObject::tr("Cannot write %1").arg(resultJson));
        }

        cmd.print(QObject::tr("reconstruction done: %1 points -> %2 (%3 ms, "
                              "%4 forward(s))")
                          .arg(exported)
                          .arg(plyPath)
                          .arg(rec.e2eMs, 0, 'f', 0)
                          .arg(rec.chunks.size()));
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
