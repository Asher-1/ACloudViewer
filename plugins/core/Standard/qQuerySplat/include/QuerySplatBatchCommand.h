// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless QuerySplat batch command (-QUERY_SPLAT), built on the shared
// ecvAICoreBatchCommand skeleton: one model load x N image items -> per-item
// outputs (3DGS PLY + RGB renders + cameras JSON) plus one RESULT_JSON
// manifest (schema "acloudviewer.aicore.batch/1"). MULTIVIEW 1 switches to
// the upstream --images semantics: all images form ONE multi-view scene with
// a single run and export set. Model resolution follows
// the shared grammar: an existing path wins; otherwise the token is a
// published catalog filename under the shared cache dirs. Headless runs do
// NOT trigger downloads unless DOWNLOAD 1 is passed.
#pragma once

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "QtCompat.h"
#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/map_anything_capi.h>
#include <aicore/querysplat_capi.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "QuerySplatModelCatalog.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

// Command keyword (the -SILENT CLI token after the binary).
#define COMMAND_QUERY_SPLAT "QUERY_SPLAT"

struct CommandQuerySplatPredict : public ccCommandLineInterface::Command {
    CommandQuerySplatPredict()
        : ccCommandLineInterface::Command(
                  QObject::tr("QuerySplat image-to-3DGS prediction"),
                  COMMAND_QUERY_SPLAT) {}

#ifdef AICore_ENABLED

    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"IMAGE", "list", "", "Input image; repeat for a view set"},
                {"INPUT_DIR", "path", "", "Directory of input images"},
                {"RECURSIVE", "bool", "0", "INPUT_DIR: recurse"},
                {"MULTIVIEW", "bool", "0",
                 "Treat all images as ONE multi-view scene (upstream "
                 "--images semantics) instead of one scene per image"},
                {"OUT_DIR", "path", "", "Output directory (required)"},
                {"MODEL", "string", "querysplat-paper-q8_0.gguf",
                 "Decoder GGUF path or published catalog filename"},
                {"ENCODER", "string", "",
                 "VGGT-Omega encoder GGUF; empty = catalog default (the "
                 "map_anything shared cache)"},
                {"DEVICE", "string", "auto", "auto|cpu|cuda|vulkan|metal[:N]"},
                {"IMAGE_SIZE", "int", "0", "0 uses the decoder GGUF metadata"},
                {"TTO", "bool", "0", "Run test-time optimization"},
                {"TTO_STEPS", "int", "20", "TTO optimization steps"},
                {"TTO_LR", "float", "0.005", "TTO AdamW learning rate"},
                {"TTO_LPIPS", "float", "0.05",
                 "TTO LPIPS term weight (0 disables the LPIPS graph)"},
                {"TTO_SAVE_STEPS", "string", "",
                 "Comma-separated TTO steps whose gaussians are also saved "
                 "as PLYs (0 = feed-forward seed; needs TTO_SNAPSHOT_DIR)"},
                {"TTO_SNAPSHOT_DIR", "path", "",
                 "Directory for the TTO step-checkpoint PLYs"},
                {"OPACITY_THRESHOLD", "float", "0.05",
                 "Gaussian PLY export opacity threshold"},
                {"EXPORT_PLY", "bool", "1", "Write the 3DGS PLY per item"},
                {"EXPORT_RENDER", "bool", "1",
                 "Write gsplat renders of the predicted input views"},
                {"EXPORT_CAMERAS", "bool", "1",
                 "Write the predicted input cameras JSON+NPZ per item"},
                {"EXPORT_INPUT_DEPTHS", "bool", "0",
                 "Write the raw VGGT depth+confidence NPZ per item"},
                {"EXPORT_DEPTH_PLY", "bool", "0",
                 "Write the VGGT depth point cloud PLY per item"},
                {"THREADS", "int", "0", "CPU threads (0 = backend default)"},
                {"STAGE_DUMP_DIR", "path", "",
                 "Optional debug graph-stage tensor directory"},
                {"RESULT_JSON", "path", "_querysplat_result.json",
                 "Result manifest path"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        using namespace ecvAICoreBatch;
        Session s;
        const Prep prep =
                prepare(cmd, COMMAND_QUERY_SPLAT, QStringLiteral("querysplat"),
                        QStringLiteral("QuerySplat "
                                       "image-to-3DGS prediction"),
                        {&options()}, &s);
        if (prep != Prep::Run) return prep == Prep::Done;

        QStringList images = s.args.list("IMAGE");
        if (const QString dir = s.args.str("INPUT_DIR"); !dir.isEmpty()) {
            images << scanDir(dir, imagePatterns(),
                              s.args.toBool("RECURSIVE", false));
        }
        naturalSort(images);
        images.removeDuplicates();
        if (images.isEmpty()) {
            return cmd.error(QStringLiteral(
                    "-QUERY_SPLAT needs IMAGE <path> (repeatable) or "
                    "INPUT_DIR <dir>"));
        }

        const QString decoderCache = QuerySplatHelpers::decoderCacheDir();
        const QString decoderPath =
                resolveModelPath(s.args.str("MODEL"), decoderCache,
                                 QStringLiteral("querysplat-paper-q8_0.gguf"));
        QString encoderPath = resolveModelPath(
                s.args.str("ENCODER"), QuerySplatHelpers::decoderCacheDir());
        if (encoderPath.isEmpty()) {
            encoderPath = QuerySplatHelpers::encoderDefaultPath();
        }
        const QString decoderName = QFileInfo(decoderPath).fileName();
        const QString encoderName = QFileInfo(encoderPath).fileName();

        // resolve + optionally download the model chain
        QString error;
        {
            QString url;
            QuerySplatModelEntry entry;
            if (!QuerySplatHelpers::findModelByFilename(decoderName, &entry)
                         .isEmpty()) {
                url = entry.downloadUrl;
            }
            if (!ensureModel(cmd, decoderPath, {url, QString()}, s.download,
                             &error)) {
                return cmd.error(error);
            }
        }
        {
            QString url;
            const int32_t n = aicore_map_anything_model_count();
            for (int32_t i = 0; i < n; ++i) {
                const auto* e = aicore_map_anything_model_at(i);
                if (e && encoderName == QString::fromUtf8(e->filename)) {
                    url = QString::fromUtf8(e->download_url);
                    break;
                }
            }
            if (!ensureModel(cmd, encoderPath, {url, QString()}, s.download,
                             &error)) {
                return cmd.error(error);
            }
        }
        // LPIPS weights when TTO is enabled
        if (s.args.toBool("TTO", false)) {
            const QString lpipsPath =
                    decoderCache + QStringLiteral("/lpips-vgg-f32.gguf");
            QuerySplatModelEntry lpEntry;
            QString url;
            if (!QuerySplatHelpers::findModelByFilename(
                         QStringLiteral("lpips-vgg-f32.gguf"), &lpEntry)
                         .isEmpty()) {
                url = lpEntry.downloadUrl;
            }
            if (!ensureModel(cmd, lpipsPath, {url, QString()}, s.download,
                             &error)) {
                return cmd.error(error);
            }
        }

        // options
        aicore_querysplat_options* opts = aicore_querysplat_options_new();
        if (!opts) return cmd.error(QStringLiteral("Out of memory"));
        aicore_querysplat_options_set_device(opts,
                                             s.device.toUtf8().constData());
        aicore_querysplat_options_set_threads(opts, s.threads);
        aicore_querysplat_options_set_f32_accuracy(opts, 1);
        const int imageSize = s.args.toInt("IMAGE_SIZE", 0);
        if (imageSize > 0)
            aicore_querysplat_options_set_image_size(opts, imageSize,
                                                     imageSize);
        aicore_querysplat_options_set_tto_lr(opts,
                                             s.args.toFloat("TTO_LR", 5e-3f));
        aicore_querysplat_options_set_tto_lpips_weight(
                opts, s.args.toFloat("TTO_LPIPS", 0.05f));
        // --tto_save_step parity: PLY checkpoints at the requested steps
        {
            const QString stepsRaw = s.args.str("TTO_SAVE_STEPS");
            const QString snapDir = s.args.str("TTO_SNAPSHOT_DIR");
            if (!snapDir.isEmpty() && !stepsRaw.isEmpty()) {
                std::vector<int32_t> steps;
                for (const QString& tok :
                     stepsRaw.split(',', QtCompat::SkipEmptyParts)) {
                    bool ok = false;
                    const int v = tok.trimmed().toInt(&ok);
                    if (ok && v >= 0) steps.push_back(v);
                }
                if (!steps.empty()) {
                    aicore_querysplat_options_set_tto_snapshot_steps(
                            opts, steps.data(), int32_t(steps.size()));
                    aicore_querysplat_options_set_tto_snapshot_dir(
                            opts, snapDir.toUtf8().constData());
                }
            }
        }
        // record the configured step count on the context (info JSON) even
        // though run_tto takes the authoritative count per call
        if (s.args.toBool("TTO", false)) {
            aicore_querysplat_options_set_tto_steps(
                    opts, s.args.toInt("TTO_STEPS", 20));
        }
        aicore_querysplat_options_set_opacity_threshold(
                opts, s.args.toFloat("OPACITY_THRESHOLD", 0.05f));
        if (!s.args.str("STAGE_DUMP_DIR").isEmpty()) {
            aicore_querysplat_options_set_dump_dir(
                    opts, s.args.str("STAGE_DUMP_DIR").toUtf8().constData());
        }

        aicore_querysplat_ctx* ctx = aicore_querysplat_load_opts(
                encoderPath.toUtf8().constData(),
                decoderPath.toUtf8().constData(), opts);
        aicore_querysplat_options_free(opts);
        if (!ctx || !aicore_querysplat_is_ready(ctx)) {
            const char* err = ctx ? aicore_querysplat_last_error(ctx) : nullptr;
            const QString message =
                    err ? QString::fromUtf8(err) : QStringLiteral("unknown");
            if (ctx) aicore_querysplat_free(ctx);
            return cmd.error(
                    QStringLiteral("Model load failed: %1").arg(message));
        }

        QJsonObject root;
        root["schema"] = QStringLiteral("acloudviewer.aicore.batch/1");
        root["task"] = QStringLiteral("querysplat");
        root["keyword"] = COMMAND_QUERY_SPLAT;
        root["decoder"] = decoderPath;
        root["encoder"] = encoderPath;
        root["device"] = s.device;
        QJsonArray rows;
        int failed = 0;

        // Scene grouping: the upstream C++ CLI consumes ONE multi-view scene
        // per invocation (--images <folder>), while the default batch mode
        // keeps the one-scene-per-image item semantics. MULTIVIEW 1 switches
        // to the upstream grouping: a single run over all views and one
        // export set for the scene.
        const bool multiview = s.args.toBool("MULTIVIEW", false);
        QList<QStringList> sceneGroups;
        if (multiview) {
            sceneGroups.append(images);
        } else {
            for (const QString& image : images) {
                sceneGroups.append(QStringList{image});
            }
        }

        static QSet<QString> usedStems;  // per-invocation, unique anyway
        for (const QStringList& scene : sceneGroups) {
            QElapsedTimer itemTimer;
            itemTimer.start();

            // decode every view; single-view scenes keep the per-item
            // decode_failed row, multi-view scenes skip unreadable files
            // with a console warning
            std::vector<QImage> owned;
            owned.reserve(size_t(scene.size()));
            QStringList readable;
            bool aborted = false;
            for (const QString& image : scene) {
                QImage img(image);
                if (img.isNull()) {
                    if (scene.size() == 1) {
                        ++failed;
                        QJsonObject row;
                        row["input"] = image;
                        row["status"] = QStringLiteral("decode_failed");
                        rows.append(row);
                        if (!s.continueOnError) aborted = true;
                    } else {
                        cmd.print(
                                QStringLiteral("[QUERY_SPLAT] unreadable image "
                                               "skipped: %1")
                                        .arg(image));
                    }
                    continue;
                }
                readable.append(image);
                owned.push_back(img.convertToFormat(QImage::Format_RGB888));
            }
            if (aborted) break;
            if (owned.empty()) {
                if (scene.size() > 1) {
                    ++failed;
                    QJsonObject row;
                    row["input"] = scene.first();
                    row["status"] = QStringLiteral("decode_failed");
                    rows.append(row);
                    if (!s.continueOnError) break;
                }
                continue;  // single-view row already recorded above
            }

            const QString inputLabel =
                    readable.size() == 1 ? readable.first()
                                         : QStringLiteral("%1 (+%2 views)")
                                                   .arg(readable.first())
                                                   .arg(readable.size() - 1);

            // the QImage buffers back the borrowed views for the call
            std::vector<aicore_image_view> views(owned.size());
            for (size_t i = 0; i < owned.size(); ++i) {
                views[i].data = owned[i].constBits();
                views[i].width = owned[i].width();
                views[i].height = owned[i].height();
                views[i].row_stride_bytes = size_t(owned[i].bytesPerLine());
                views[i].format = AICORE_IMAGE_RGB8;
            }

            const int rc = aicore_querysplat_run_images(ctx, views.data(),
                                                        int(views.size()));
            if (rc != 0) {
                ++failed;
                const char* err = aicore_querysplat_last_error(ctx);
                QJsonObject row;
                row["input"] = inputLabel;
                row["status"] = QStringLiteral("inference_failed");
                row["error"] = err ? QString::fromUtf8(err) : QString();
                rows.append(row);
                if (!s.continueOnError) break;
                continue;
            }

            const QString stem = uniqueStem(
                    QFileInfo(readable.first()).completeBaseName(), &usedStems);
            const int S = aicore_querysplat_result_view_count(ctx);
            const int W = aicore_querysplat_result_width(ctx);
            const int H = aicore_querysplat_result_height(ctx);

            QJsonObject row;
            row["input"] = inputLabel;
            row["status"] = QStringLiteral("ok");
            row["views"] = S;
            row["width"] = W;
            row["height"] = H;
            row["n_gaussians"] =
                    double(aicore_querysplat_result_gaussian_count(ctx));
            row["gaussian_stride"] =
                    aicore_querysplat_result_gaussian_stride(ctx);

            // optional TTO refinement against the predicted input views; the
            // refined gaussians replace the feed-forward ones inside the ctx,
            // so every export below reflects the optimization (the gaussian
            // count itself is unchanged)
            if (s.args.toBool("TTO", false)) {
                const int ttoSteps = s.args.toInt("TTO_STEPS", 20);
                aicore_querysplat_tto_step* losses = nullptr;
                int32_t n_losses = 0;
                const float* tto_g = nullptr;
                int64_t tto_n = 0;
                const int trc = aicore_querysplat_run_tto(
                        ctx, ttoSteps, &losses, &n_losses, &tto_g, &tto_n);
                if (trc != 0) {
                    ++failed;
                    const char* err = aicore_querysplat_last_error(ctx);
                    row["status"] = QStringLiteral("tto_failed");
                    row["error"] = err ? QString::fromUtf8(err) : QString();
                    rows.append(row);
                    if (!s.continueOnError) break;
                    continue;
                }
                QJsonArray lossRows;
                for (int32_t i = 0; i < n_losses; ++i) {
                    QJsonObject l;
                    l["step"] = i + 1;
                    l["l1"] = double(losses[i].l1);
                    l["ssim"] = double(losses[i].ssim);
                    l["lpips"] = double(losses[i].lpips);
                    lossRows.append(l);
                }
                row["tto_steps"] = ttoSteps;
                row["tto_losses"] = lossRows;
            }

            if (s.args.toBool("EXPORT_PLY", true)) {
                const QString ply = QDir(s.outDir).filePath(
                        stem + QStringLiteral("_gaussians.ply"));
                row["gaussians_ply"] =
                        aicore_querysplat_export_gaussians_ply(
                                ctx, ply.toUtf8().constData(),
                                s.args.toFloat("OPACITY_THRESHOLD", 0.05f)) == 0
                                ? ply
                                : QString();
            }
            if (s.args.toBool("EXPORT_RENDER", true)) {
                const float* w2c = aicore_querysplat_result_w2c(ctx);
                const float* K = aicore_querysplat_result_intrinsics(ctx);
                const float* rgbF = nullptr;
                unsigned char* rgb8 = nullptr;
                if (w2c && K &&
                    aicore_querysplat_render_views(ctx, w2c, K, S, &rgbF,
                                                   &rgb8) == 0) {
                    const size_t plane = size_t(W) * H;
                    QJsonArray renderFiles;
                    for (int v = 0; v < S; ++v) {
                        QImage out(rgb8 + size_t(v) * plane * 3, W, H, W * 3,
                                   QImage::Format_RGB888);
                        const QString path = QDir(s.outDir).filePath(
                                QStringLiteral("%1_render%2.png")
                                        .arg(stem)
                                        .arg(v));
                        out.save(path);
                        renderFiles.append(path);
                    }
                    row["render_views"] = renderFiles;
                }
            }
            if (s.args.toBool("EXPORT_CAMERAS", true)) {
                if (char* j = aicore_querysplat_export_cameras_json(ctx)) {
                    const QString path = QDir(s.outDir).filePath(
                            stem + QStringLiteral("_cameras.json"));
                    QFile f(path);
                    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                        f.write(j);
                        row["cameras_json"] = path;
                    }
                    aicore_querysplat_free_buffer(j);
                }
                // tensor-format companion (the official .npz payload)
                const QString npzPath = QDir(s.outDir).filePath(
                        stem + QStringLiteral("_cameras.npz"));
                if (aicore_querysplat_export_cameras_npz(
                            ctx, npzPath.toUtf8().constData()) == 0) {
                    row["cameras_npz"] = npzPath;
                }
            }
            if (s.args.toBool("EXPORT_INPUT_DEPTHS", false)) {
                const QString path = QDir(s.outDir).filePath(
                        stem + QStringLiteral("_input_depths.npz"));
                if (aicore_querysplat_export_input_depths_npz(
                            ctx, path.toUtf8().constData()) == 0) {
                    row["input_depths_npz"] = path;
                }
            }
            if (s.args.toBool("EXPORT_DEPTH_PLY", false)) {
                const float* xyz = nullptr;
                unsigned char* col = nullptr;
                int32_t n = 0;
                if (aicore_querysplat_depth_pointcloud(ctx, 200000, &xyz, &col,
                                                       &n) == 0 &&
                    n > 0) {
                    const QString path = QDir(s.outDir).filePath(
                            stem + QStringLiteral("_depth_cloud.ply"));
                    QFile f(path);
                    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                        QTextStream ts(&f);
                        ts << "ply\nformat ascii 1.0\nelement vertex " << n
                           << "\nproperty float x\nproperty float y\n"
                              "property float z\nproperty uchar red\n"
                              "property uchar green\nproperty uchar blue\n"
                              "end_header\n";
                        for (int32_t i = 0; i < n; ++i) {
                            ts << xyz[size_t(i) * 3] << ' '
                               << xyz[size_t(i) * 3 + 1] << ' '
                               << xyz[size_t(i) * 3 + 2] << ' '
                               << int(col[size_t(i) * 3]) << ' '
                               << int(col[size_t(i) * 3 + 1]) << ' '
                               << int(col[size_t(i) * 3 + 2]) << '\n';
                        }
                        row["depth_pointcloud_ply"] = path;
                    }
                }
            }
            aicore_pipeline_timings t{};
            if (aicore_querysplat_last_pipeline_timings(ctx, &t) == 0) {
                row["timings"] = timingsJson(t);
            }
            row["elapsed_ms"] = double(itemTimer.elapsed());
            rows.append(row);
        }

        root["items"] = rows;
        root["failed"] = failed;
        saveJson(root, s.resultJson, nullptr);
        cmd.print(
                QStringLiteral("[QUERY_SPLAT] manifest: %1").arg(s.resultJson));

        aicore_querysplat_free(ctx);
        return failed == 0;
    }

#endif  // AICore_ENABLED
};
