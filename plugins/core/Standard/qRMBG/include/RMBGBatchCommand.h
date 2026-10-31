// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch background removal (-RMBG_REMOVE): one RMBG-2.0 model load,
// N images -> <stem>_rgba.png cutout + <stem>_alpha.png matte each, plus the
// shared ecvAICoreBatch RESULT_JSON manifest. Same C API call, threshold
// pass and statistics as the dialog worker (RMBGWorker::runInference).

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/rmbg_capi.h>

#include <cstdlib>

#include "RMBGModelCatalog.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_RMBG_REMOVE "RMBG_REMOVE"

struct CommandRmbgRemove : public ccCommandLineInterface::Command {
    CommandRmbgRemove()
        : ccCommandLineInterface::Command(
                  QObject::tr("RMBG batch background removal"),
                  COMMAND_RMBG_REMOVE) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"ALPHA_THRESHOLD", "float", "0.5",
                 "Alpha below this becomes fully transparent (0 = off)"},
                {"SAVE_CUTOUT", "bool", "1", "Write <stem>_rgba.png"},
                {"SAVE_MATTE", "bool", "1", "Write <stem>_alpha.png"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_RMBG_REMOVE, QStringLiteral("rmbg"),
                           QStringLiteral("Background removal (RMBG-2.0)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = RMBGHelpers::modelCacheDir();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const RMBGModelEntry& e : RMBGHelpers::catalogModels()) {
                rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                QDir(cacheDir).filePath(e.filename),
                                QString()});
            }
            return B::listModels(cmd, s, rows);
        }
        const float alphaThreshold = s.args.toFloat("ALPHA_THRESHOLD", 0.5f);
        const bool saveCutout = s.args.toBool("SAVE_CUTOUT", true);
        const bool saveMatte = s.args.toBool("SAVE_MATTE", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        const QVector<RMBGModelEntry> catalog = RMBGHelpers::catalogModels();
        const int defIdx = RMBGHelpers::catalogDefaultIndex();
        const QString defaultName = defIdx >= 0 && defIdx < catalog.size()
                                            ? catalog[defIdx].filename
                                            : QString();
        const QString modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir, defaultName);
        RMBGModelEntry entry;
        RMBGHelpers::findModelByFilename(QFileInfo(modelPath).fileName(),
                                         &entry);
        if (!B::ensureModel(cmd, modelPath, {entry.downloadUrl, QString()},
                            s.download, &error)) {
            return cmd.error(error);
        }

        aicore_rmbg_options* opts = aicore_rmbg_options_new();
        aicore_rmbg_options_set_device(opts, s.device.toUtf8().constData());
        aicore_rmbg_options_set_threads(opts, s.threads);
        aicore_rmbg_ctx* ctx =
                aicore_rmbg_load_opts(modelPath.toUtf8().constData(), opts);
        aicore_rmbg_options_free(opts);
        if (ctx == nullptr || !aicore_rmbg_is_ready(ctx)) {
            const char* msg = ctx ? aicore_rmbg_last_error(ctx) : nullptr;
            if (ctx) aicore_rmbg_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg ? QString::fromUtf8(msg)
                                              : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_rmbg_ctx, void (*)(aicore_rmbg_ctx*)> guard(
                ctx, aicore_rmbg_free);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        if (char* info = aicore_rmbg_info_json(ctx)) {
            manifest.set("model_info",
                         QJsonDocument::fromJson(QByteArray(info)).object());
            aicore_rmbg_free_buffer(info);
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    QImage source = B::loadImage(item.path,
                                                 QImage::Format_Invalid, &err);
                    if (source.isNull()) return false;
                    aicore_image_view view{};
                    if (!RMBGHelpers::imageView(&source, &view)) {
                        err = QStringLiteral("Unsupported image format");
                        return false;
                    }
                    uint8_t* rgba = nullptr;
                    int32_t w = 0, h = 0;
                    int len = 0;
                    if (aicore_rmbg_remove_background_rgba_image_view(
                                ctx, &view, &rgba, &w, &h, &len) != 0 ||
                        rgba == nullptr || len <= 0) {
                        if (rgba) aicore_rmbg_free_buffer(rgba);
                        err = QString::fromUtf8(aicore_rmbg_last_error(ctx));
                        return false;
                    }
                    // Deep copy, then return the buffer to libAICore's
                    // allocator (cross-CRT safe on Windows).
                    QImage result =
                            QImage(rgba, w, h, w * 4, QImage::Format_RGBA8888)
                                    .convertToFormat(QImage::Format_ARGB32);
                    aicore_rmbg_free_buffer(rgba);
                    if (alphaThreshold > 0.0f) {
                        result = RMBGHelpers::applyAlphaThreshold(
                                result, alphaThreshold);
                    }
                    double alphaMean = 0.0, fgRatio = 0.0;
                    RMBGHelpers::computeAlphaStats(result, &alphaMean,
                                                   &fgRatio);
                    QJsonObject outputs;
                    if (saveCutout) {
                        const QString p = B::outPath(s, item.stem, "_rgba.png");
                        if (!B::saveImage(result, p, &err)) return false;
                        outputs["cutout"] = p;
                    }
                    if (saveMatte) {
                        const QString p =
                                B::outPath(s, item.stem, "_alpha.png");
                        // Alpha8 and Grayscale8 share one byte per
                        // pixel: relabel instead of converting.
                        QImage matte =
                                result.convertToFormat(QImage::Format_Alpha8);
                        matte.reinterpretAsFormat(QImage::Format_Grayscale8);
                        if (!B::saveImage(matte, p, &err)) return false;
                        outputs["matte"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_rmbg_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["width"] = w;
                    rec["height"] = h;
                    rec["alpha_mean"] = alphaMean;
                    rec["foreground_ratio"] = fgRatio;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_RMBG_REMOVE));
    }
#endif  // AICore_ENABLED
};
