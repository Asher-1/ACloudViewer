// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch generation command (-SAM3D_GENERATE), mirroring the
// YOLO_TRACK offscreen pattern: one child process loads the model set once
// and runs the AICore sam3d pipeline over N input images, exporting a
// Gaussian PLY (and optional FlexiCubes mesh stats) per image plus a single
// RESULT_JSON manifest. Images may carry their object mask either as an
// alpha channel or as an ordered MASK <path> companion (official
// `mask > 0` semantics, same as the C API mask view).

#pragma once

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/image_view.h>
#include <aicore/pipeline_timing.h>
#include <aicore/rmbg_capi.h>
#include <aicore/sam3d_capi.h>

#include <cstring>
#include <memory>
#include <vector>
#endif  // AICore_ENABLED

// Command keyword (the -SILENT CLI token users type after the binary).
#define COMMAND_SAM3D_GENERATE "SAM3D_GENERATE"

struct CommandSam3dGenerate : public ccCommandLineInterface::Command {
    CommandSam3dGenerate()
        : ccCommandLineInterface::Command(
                  QObject::tr("SAM 3D batch generation"),
                  COMMAND_SAM3D_GENERATE) {}

#ifdef AICore_ENABLED

    bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[SAM3D_GENERATE]");

        QStringList images;
        QStringList masks;  // ordered companions; may stay empty
        QString outputDir;
        QString modelsDir;  // empty -> AICore cache dir
        QString device = QStringLiteral("auto");
        QString dtype = QStringLiteral("q4_k");
        QString resultJson;
        int steps = 25;
        int seed = 42;
        bool generateMesh = true;
        bool useRmbg = true;
        int threads = 8;
        QString conditionsOut;  // accuracy-diagnostic: dump ss_input_*.samt
        // 0 = disable the MoGe point-map cache (matches the acceptance
        // probe / upstream reference bit-for-bit); 1 reuses cached point
        // maps for repeat runs (tiny numeric drift is possible).
        bool useMogeCache = false;

        auto take = [&cmd]() -> QString {
            return cmd.arguments().isEmpty() ? QString()
                                             : cmd.arguments().takeFirst();
        };
        while (!cmd.arguments().isEmpty()) {
            const QString key = cmd.arguments().takeFirst().toUpper();
            if (key == QStringLiteral("IMAGE")) {
                images << take();
            } else if (key == QStringLiteral("MASK")) {
                masks << take();
            } else if (key == QStringLiteral("OUT_DIR")) {
                outputDir = take();
            } else if (key == QStringLiteral("MODELS_DIR")) {
                modelsDir = take();
            } else if (key == QStringLiteral("DEVICE")) {
                device = take();
            } else if (key == QStringLiteral("DTYPE")) {
                dtype = take().toLower();
            } else if (key == QStringLiteral("STEPS")) {
                steps = take().toInt();
            } else if (key == QStringLiteral("SEED")) {
                seed = take().toInt();
            } else if (key == QStringLiteral("MESH")) {
                generateMesh = take().toInt() != 0;
            } else if (key == QStringLiteral("RMBG")) {
                useRmbg = take().toInt() != 0;
            } else if (key == QStringLiteral("MOGE_CACHE")) {
                useMogeCache = take().toInt() != 0;
            } else if (key == QStringLiteral("THREADS")) {
                threads = take().toInt();
            } else if (key == QStringLiteral("RESULT_JSON")) {
                resultJson = take();
            } else if (key == QStringLiteral("COND_OUT")) {
                conditionsOut = take();
            } else {
                return cmd.error(QObject::tr("Unknown argument after -%1: %2")
                                         .arg(COMMAND_SAM3D_GENERATE, key));
            }
        }

        if (images.isEmpty()) {
            return cmd.error(QObject::tr(
                    "IMAGE <path> is required (repeat for batch runs)"));
        }
        if (outputDir.isEmpty()) {
            return cmd.error(QObject::tr("OUT_DIR <dir> is required"));
        }
        if (dtype != QStringLiteral("f16") && dtype != QStringLiteral("q8_0") &&
            dtype != QStringLiteral("q4_k")) {
            return cmd.error(QObject::tr("DTYPE must be f16, q8_0 or q4_k"));
        }
        if (steps < 1 || steps > 100) {
            return cmd.error(QObject::tr("STEPS must be in [1, 100]"));
        }
        if (!QDir().mkpath(outputDir)) {
            return cmd.error(QObject::tr("Cannot create output directory %1")
                                     .arg(outputDir));
        }

        // Model resolution: explicit MODELS_DIR wins, otherwise the shared
        // AICore cache (same layout the GUI worker checks).
        if (modelsDir.isEmpty()) {
            const char* cache = aicore_sam3d_model_cache_dir();
            modelsDir = cache ? QString::fromUtf8(cache) : QString();
        }
        const QDir modelDir(modelsDir);
        if (!modelDir.exists()) {
            return cmd.error(QObject::tr("SAM 3D model cache not found: %1")
                                     .arg(modelsDir));
        }
        QStringList stageModels = {
                QStringLiteral("ss_generator-%1.gguf").arg(dtype),
                QStringLiteral("ss_decoder-%1.gguf").arg(dtype),
                QStringLiteral("slat_generator-%1.gguf").arg(dtype),
                QStringLiteral("slat_decoder_gs-%1.gguf").arg(dtype),
                QStringLiteral("moge_vitl-f16.gguf")};
        if (generateMesh) {
            stageModels
                    << QStringLiteral("slat_decoder_mesh-%1.gguf").arg(dtype);
        }
        for (const QString& name : stageModels) {
            if (!modelDir.exists(name)) {
                return cmd.error(QObject::tr("missing model '%1' in %2")
                                         .arg(name, modelsDir));
            }
        }

        // Optional shared RMBG model (borrowed from the rmbg task cache —
        // never downloaded here). Load once for the whole batch.
        aicore_rmbg_ctx* rmbgCtx = nullptr;
        if (useRmbg) {
            char* rmbg_cache = aicore_rmbg_model_cache_dir();
            const QDir rmbgDir(rmbg_cache ? QString::fromUtf8(rmbg_cache)
                                          : QString());
            if (rmbg_cache) aicore_rmbg_free_buffer(rmbg_cache);
            QString rmbgModel;
            for (const QString& name : {QStringLiteral("rmbg_q8.gguf"),
                                        QStringLiteral("rmbg_f16.gguf")}) {
                const QString candidate = rmbgDir.filePath(name);
                if (QFileInfo::exists(candidate)) {
                    rmbgModel = candidate;
                    break;
                }
            }
            if (rmbgModel.isEmpty()) {
                cmd.print(QObject::tr("RMBG model not found under %1 — images "
                                      "are used "
                                      "as-is (alpha channel still applies).")
                                  .arg(rmbgDir.absolutePath()));
            } else {
                const QByteArray rmbgUtf8 = rmbgModel.toUtf8();
                aicore_rmbg_options* rmbgOpts = aicore_rmbg_options_new();
                aicore_rmbg_options_set_device(rmbgOpts,
                                               device.toUtf8().constData());
                aicore_rmbg_options_set_threads(rmbgOpts, threads);
                rmbgCtx = aicore_rmbg_load_opts(rmbgUtf8.constData(), rmbgOpts);
                aicore_rmbg_options_free(rmbgOpts);
                if (rmbgCtx == nullptr) {
                    cmd.print(QObject::tr(
                            "RMBG load failed — images are used as-is."));
                }
            }
        }

        // One sam3d context for the whole batch: the model set is loaded
        // once and reused across images (the batch saving vs one process
        // per image).
        const QByteArray modelsDirUtf8 = modelsDir.toUtf8();
        aicore_sam3d_dtype dtypeEnum = AICORE_SAM3D_DTYPE_Q4_K;
        if (dtype == QStringLiteral("f16")) {
            dtypeEnum = AICORE_SAM3D_DTYPE_F16;
        } else if (dtype == QStringLiteral("q8_0")) {
            dtypeEnum = AICORE_SAM3D_DTYPE_Q8_0;
        }
        aicore_sam3d_options* opts = aicore_sam3d_options_new();
        aicore_sam3d_options_set_models_dir(opts, modelsDirUtf8.constData());
        aicore_sam3d_options_set_dtype(opts, dtypeEnum);
        aicore_sam3d_options_set_device(opts, device.toUtf8().constData());
        aicore_sam3d_options_set_threads(opts, threads);
        aicore_sam3d_options_set_seed(opts, seed);
        aicore_sam3d_options_set_steps(opts, steps, steps);
        aicore_sam3d_options_set_disable_moge_cache(opts, useMogeCache ? 0 : 1);
        if (!conditionsOut.isEmpty()) {
            const QByteArray condUtf8 = conditionsOut.toUtf8();
            aicore_sam3d_options_set_conditions_out(opts, condUtf8.constData());
            cmd.print(QObject::tr("Condition dump: %1").arg(conditionsOut));
        }
        char err[512] = {0};
        aicore_sam3d_ctx* ctx = aicore_sam3d_load_opts(opts, err, sizeof(err));
        aicore_sam3d_options_free(opts);
        if (ctx == nullptr) {
            if (rmbgCtx) aicore_rmbg_free(rmbgCtx);
            return cmd.error(QObject::tr("AICore sam3d load failed: %1")
                                     .arg(QString::fromUtf8(err)));
        }
        const QString backend = QString::fromUtf8(aicore_sam3d_backend(ctx));
        cmd.print(QObject::tr("Backend resolved: %1 (%2, steps=%3, seed=%4)")
                          .arg(backend, dtype)
                          .arg(steps)
                          .arg(seed));

        QJsonArray items;
        int okCount = 0;
        int failedCount = 0;
        for (int i = 0; i < images.size(); ++i) {
            const QString imagePath = images[i];
            const QString maskPath = i < masks.size() ? masks[i] : QString();

            QJsonObject item;
            item["image"] = imagePath;
            item["mask"] = maskPath;

            // Decode once to the non-premultiplied Format_ARGB32: on
            // little-endian systems its memory order is BGRA8, which the
            // image-view contract accepts natively — the QImage storage maps
            // straight into the view with its real bytesPerLine stride.
            QImage image(imagePath);
            if (image.isNull()) {
                item["status"] = QStringLiteral("decode_failed");
                item["error"] = QStringLiteral("cannot decode image");
                items.append(item);
                ++failedCount;
                continue;
            }
            image = image.format() == QImage::Format_ARGB32
                            ? image
                            : image.convertToFormat(QImage::Format_ARGB32);
            const int w = image.width();
            const int h = image.height();

            // Optional ordered mask file (official `mask > 0` semantics);
            // grayscale rows keep their real QImage stride.
            QImage maskImage;
            aicore_image_view maskView{};
            aicore_image_view* maskPtr = nullptr;
            if (!maskPath.isEmpty()) {
                maskImage = QImage(maskPath);
                if (maskImage.isNull()) {
                    item["status"] = QStringLiteral("decode_failed");
                    item["error"] = QStringLiteral("cannot decode mask");
                    items.append(item);
                    ++failedCount;
                    continue;
                }
                maskImage = maskImage.format() == QImage::Format_Grayscale8
                                    ? maskImage
                                    : maskImage.convertToFormat(
                                              QImage::Format_Grayscale8);
                if (maskImage.width() != w || maskImage.height() != h) {
                    item["status"] = QStringLiteral("mask_mismatch");
                    item["error"] =
                            QStringLiteral(
                                    "mask size %1x%2 != image size %3x%4")
                                    .arg(maskImage.width())
                                    .arg(maskImage.height())
                                    .arg(w)
                                    .arg(h);
                    items.append(item);
                    ++failedCount;
                    continue;
                }
                maskView.data = const_cast<uint8_t*>(maskImage.constBits());
                maskView.width = w;
                maskView.height = h;
                maskView.row_stride_bytes =
                        static_cast<size_t>(maskImage.bytesPerLine());
                maskView.format = AICORE_IMAGE_GRAY8;
                maskPtr = &maskView;
            } else if (rmbgCtx != nullptr) {
                // Borrow the shared RMBG model: binary alpha in place
                // (alpha > 0 keeps the pixel), same as the GUI worker.
                aicore_image_view rgbaView{
                        image.bits(), w, h,
                        static_cast<size_t>(image.bytesPerLine()),
                        AICORE_IMAGE_BGRA8};
                uint8_t* alpha = nullptr;
                int32_t alphaW = 0;
                int32_t alphaH = 0;
                if (aicore_rmbg_alpha_mat_image_view(rmbgCtx, &rgbaView, &alpha,
                                                     &alphaW, &alphaH) == 0 &&
                    alpha != nullptr && alphaW == w && alphaH == h) {
                    for (int y = 0; y < h; ++y) {
                        uint8_t* row = image.scanLine(y);
                        const uint8_t* maskRow =
                                alpha + static_cast<size_t>(y) * alphaW;
                        for (int x = 0; x < w; ++x) {
                            row[static_cast<size_t>(x) * 4 + 3] =
                                    maskRow[x] == 0 ? 0 : 255;
                        }
                    }
                } else {
                    cmd.print(
                            QObject::tr(
                                    "image %1: RMBG matting failed — using the "
                                    "image as-is")
                                    .arg(imagePath));
                }
                if (alpha) aicore_rmbg_free_buffer(alpha);
            }

            const QString plyName =
                    QStringLiteral("%1_sam3d.ply")
                            .arg(QFileInfo(imagePath).completeBaseName());
            const QByteArray plyUtf8 =
                    QDir(outputDir).filePath(plyName).toUtf8();
            const aicore_image_view view{
                    image.bits(), w, h,
                    static_cast<size_t>(image.bytesPerLine()),
                    AICORE_IMAGE_BGRA8};
            aicore_sam3d_result* sam3d = aicore_sam3d_generate(
                    ctx, &view, maskPtr, plyUtf8.constData(),
                    generateMesh ? 1 : 0, nullptr, nullptr);
            if (sam3d == nullptr) {
                const char* message = aicore_sam3d_last_error(ctx);
                item["status"] = QStringLiteral("generate_failed");
                item["error"] =
                        QString::fromUtf8(message != nullptr ? message : "?");
                items.append(item);
                ++failedCount;
                continue;
            }

            aicore_pipeline_timings timings{};
            double e2eMs = 0.0;
            if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0) {
                e2eMs = timings.e2e_ms;
            }
            item["status"] = QStringLiteral("ok");
            item["ply"] = QString::fromUtf8(plyUtf8);
            item["gaussians"] = static_cast<qint64>(
                    aicore_sam3d_result_gaussian_count(sam3d));
            item["mesh_vertices"] =
                    aicore_sam3d_result_mesh_vertex_count(sam3d);
            item["mesh_triangles"] =
                    aicore_sam3d_result_mesh_triangle_count(sam3d);
            item["e2e_ms"] = e2eMs;
            aicore_sam3d_result_free(sam3d);
            items.append(item);
            ++okCount;
            cmd.print(QObject::tr("image %1/%2 done: %3 (%4 gaussians, %5 ms)")
                              .arg(i + 1)
                              .arg(images.size())
                              .arg(plyName)
                              .arg(item["gaussians"].toVariant().toString())
                              .arg(e2eMs, 0, 'f', 0));
        }

        if (rmbgCtx) aicore_rmbg_free(rmbgCtx);
        aicore_sam3d_free(ctx);

        QJsonObject report;
        report["task"] = QStringLiteral("sam3d");
        report["backend"] = backend;
        report["dtype"] = dtype;
        report["steps"] = steps;
        report["seed"] = seed;
        report["ok"] = okCount;
        report["failed"] = failedCount;
        report["items"] = items;
        const QByteArray reportData =
                QJsonDocument(report).toJson(QJsonDocument::Indented);
        if (!resultJson.isEmpty()) {
            QFile jsonFile(resultJson);
            if (!jsonFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
                return cmd.error(
                        QObject::tr("Cannot write %1").arg(resultJson));
            }
            jsonFile.write(reportData);
            cmd.print(QObject::tr("Result JSON: %1").arg(resultJson));
        } else {
            cmd.print(QString::fromUtf8(reportData));
        }

        return failedCount == 0;
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_SAM3D_GENERATE));
    }
#endif  // AICore_ENABLED
};
