// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch line-segment extraction (-DEEPLSD_LINES): one DeepLSD model
// load, N images -> <stem>_lines.json (+ optional overlays) each, plus the
// shared ecvAICoreBatch RESULT_JSON manifest. The per-image body is
// DeepLSDWorker::extractImage, the exact code path of the dialog.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/deeplsd_capi.h>

#include "DeepLSDWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_DEEPLSD_LINES "DEEPLSD_LINES"

struct CommandDeepLsdLines : public ccCommandLineInterface::Command {
    CommandDeepLsdLines()
        : ccCommandLineInterface::Command(
                  QObject::tr("DeepLSD batch line extraction"),
                  COMMAND_DEEPLSD_LINES) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MIN_SCORE", "float", "0.15",
                 "Drop segments below this score (0 = keep all)"},
                {"SAVE_VIS", "bool", "1", "Write <stem>_lines.png overlay"},
                {"DISTANCE_OVERLAY", "bool", "0",
                 "Write <stem>_distance.png (distance-field heat map)"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_DEEPLSD_LINES,
                           QStringLiteral("deeplsd"),
                           QStringLiteral("Line segment detection (DeepLSD)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = B::takeCString(
                aicore_deeplsd_model_cache_dir(), &aicore_deeplsd_free_buffer);
        const std::vector<B::CatalogRow> rows =
                B::familyRows(AICORE_MODEL_FAMILY_DEEPLSD, cacheDir);
        if (s.listModels) return B::listModels(cmd, s, rows);

        DeepLSDWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.minSegmentScore = s.args.toFloat("MIN_SCORE", 0.15f);
        settings.computeDistanceOverlay =
                s.args.toBool("DISTANCE_OVERLAY", false);
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        settings.modelPath = B::resolveModelPath(
                s.args.str("MODEL"), cacheDir,
                rows.empty() ? QString() : rows.front().filename);
        if (!B::ensureModel(
                    cmd, settings.modelPath,
                    B::familySource(AICORE_MODEL_FAMILY_DEEPLSD,
                                    QFileInfo(settings.modelPath).fileName()),
                    s.download, &error)) {
            return cmd.error(error);
        }

        aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
        aicore_deeplsd_options_set_device(opts, s.device.toUtf8().constData());
        aicore_deeplsd_options_set_threads(opts, s.threads);
        aicore_deeplsd_ctx* ctx = aicore_deeplsd_load_opts(
                settings.modelPath.toUtf8().constData(), opts);
        aicore_deeplsd_options_free(opts);
        if (ctx == nullptr || !aicore_deeplsd_is_ready(ctx)) {
            const char* msg = ctx ? aicore_deeplsd_last_error(ctx) : nullptr;
            if (ctx) aicore_deeplsd_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg ? QString::fromUtf8(msg)
                                              : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_deeplsd_ctx, void (*)(aicore_deeplsd_ctx*)>
                guard(ctx, aicore_deeplsd_free);

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        manifest.set("min_score", settings.minSegmentScore);

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    DeepLSDWorker::Settings one = settings;
                    one.inputPath = item.path;
                    DeepLSDRunResult result;
                    if (!DeepLSDWorker::extractImage(ctx, one, &result, &err)) {
                        return false;
                    }
                    QJsonArray segments;
                    for (const DeepLSDLineSegment& seg : result.segments) {
                        segments.append(QJsonArray{seg.x1, seg.y1, seg.x2,
                                                   seg.y2, seg.score});
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_lines.json");
                    if (!B::saveJson(
                                QJsonObject{{"image", item.path},
                                            {"width", result.originalWidth},
                                            {"height", result.originalHeight},
                                            {"format", "x1,y1,x2,y2,score"},
                                            { "segments",
                                              segments }},
                                jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "lines", jsonPath }};
                    if (saveVis) {
                        const QString p =
                                B::outPath(s, item.stem, "_lines.png");
                        if (!B::saveImage(result.lineVisualization, p, &err))
                            return false;
                        outputs["visualization"] = p;
                    }
                    if (one.computeDistanceOverlay &&
                        !result.distanceOverlay.isNull()) {
                        const QString p =
                                B::outPath(s, item.stem, "_distance.png");
                        if (!B::saveImage(result.distanceOverlay, p, &err))
                            return false;
                        outputs["distance"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_deeplsd_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["segments"] = static_cast<int>(result.segments.size());
                    rec["device"] = result.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_DEEPLSD_LINES));
    }
#endif  // AICore_ENABLED
};
