// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch VLX-Seek grounding (-VLXSEEK_RUN): one three-model context,
// N images -> <stem>_vlxseek.json (official inference.py JSON fields plus the
// shared ecvAICoreBatch RESULT_JSON manifest) each, plus an optional Qt-free
// PNG visualization. The per-image body is VLXSeekWorker::runImage, the exact
// code path of the dialog.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/vlx_seek_capi.h>

#include "VLXSeekModelCatalog.h"
#include "VLXSeekWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_VLXSEEK_RUN "VLXSEEK_RUN"

struct CommandVlxSeekRun : public ccCommandLineInterface::Command {
    CommandVlxSeekRun()
        : ccCommandLineInterface::Command(
                  QObject::tr("VLX-Seek grounding batch"),
                  COMMAND_VLXSEEK_RUN) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"TASK", "enum", "detection",
                 "detection|grounding_single|counting|reasoning_detection|"
                 "brief_region_caption|region_ocr|vqa"},
                {"TEXT", "string", "",
                 "Target text (categories / phrase / "
                 "question)"},
                {"LANG", "enum", "en", "en|zh (task template language)"},
                {"TARGET_INDEXES", "string", "",
                 "Comma-separated zero-based proposal indexes "
                 "(required for brief_region_caption|region_ocr, e.g. 2,3)"},
                {"PRECISION", "enum", "q8_0",
                 "f16|q8_0 (vision+LLM rows of "
                 "the catalog; detector stays "
                 "f32)"},
                {"MAX_NEW_TOKENS", "int", "1024", "Greedy generation cap"},
                {"DEVICE", "device", "", "Device request (empty = auto)"},
                {"THREADS", "int", "0", "CPU threads (0 = default)"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_VLXSEEK_RUN, QStringLiteral("vlxseek"),
                           QStringLiteral("Fine-grained grounding (VLX-Seek)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = VLXSeekHelpers::modelCacheDir();
        const QVector<VLXSeekModelEntry> catalog =
                VLXSeekHelpers::catalogModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const VLXSeekModelEntry& e : catalog) {
                rows.push_back(
                        {e.filename, e.displayName, e.downloadUrl,
                         QDir(cacheDir).filePath(e.filename),
                         e.role == 0 ? QStringLiteral("detector")
                                     : (e.role == 1 ? QStringLiteral("vision")
                                                    : QStringLiteral("llm"))});
            }
            return B::listModels(cmd, s, rows);
        }

        VLXSeekWorker::Settings settings;
        settings.task = s.args.str("TASK", "detection");
        settings.text = s.args.str("TEXT", QString());
        settings.lang = s.args.str("LANG", "en");
        settings.targetIndexes = s.args.str("TARGET_INDEXES", "");
        const QString precision = s.args.str("PRECISION", "q8_0");
        settings.device = s.args.str("DEVICE", s.device);
        settings.threads = s.threads;
        settings.maxNewTokens = s.args.toInt("MAX_NEW_TOKENS", 1024);
        if (settings.text.isEmpty() && settings.task != "vqa") {
            return cmd.error(QStringLiteral("TEXT is required for task %1")
                                     .arg(settings.task));
        }
        {
            // Mirror the runtime contract up front: the region tasks
            // hard-reject empty target lists, so fail before model load.
            QString targetsError;
            const QVector<int32_t> targets = VLXSeekHelpers::parseTargetIndexes(
                    settings.targetIndexes, &targetsError);
            const bool regionTask =
                    settings.task == QStringLiteral("brief_region_caption") ||
                    settings.task == QStringLiteral("region_ocr");
            if (!targetsError.isEmpty()) return cmd.error(targetsError);
            if (regionTask && targets.isEmpty()) {
                return cmd.error(
                        QStringLiteral("TARGET_INDEXES is required for task "
                                       "%1 (e.g. -TARGET_INDEXES 2,3)")
                                .arg(settings.task));
            }
        }
        if (!s.args.ok()) return cmd.error(s.args.error());

        // Resolve the three model paths from the catalog + precision; the
        // detector stays f32 (the official .pth dtype).
        VLXSeekModelEntry vision, wedetect, llm;
        for (const VLXSeekModelEntry& e : catalog) {
            if (e.role == 1 && e.quantization == precision) vision = e;
            if (e.role == 0 && e.quantization == "f32") wedetect = e;
            if (e.role == 2 && e.quantization == precision) llm = e;
        }
        struct Need {
            VLXSeekModelEntry entry;
            const char* what;
        };
        const std::vector<Need> needs = {{wedetect, "detector"},
                                         {vision, "vision"},
                                         { llm,
                                           "LLM" }};
        QString error;
        for (const Need& need : needs) {
            if (need.entry.filename.isEmpty()) {
                return cmd.error(
                        QStringLiteral("catalog has no %1 row for precision "
                                       "%2")
                                .arg(need.what, precision));
            }
            const QString path = QDir(cacheDir).filePath(need.entry.filename);
            if (!B::ensureModel(cmd, path, {need.entry.downloadUrl, QString()},
                                s.download, &error)) {
                return cmd.error(error);
            }
        }
        settings.wedetectPath = QDir(cacheDir).filePath(wedetect.filename);
        settings.visionPath = QDir(cacheDir).filePath(vision.filename);
        settings.llmPath = QDir(cacheDir).filePath(llm.filename);

        aicore_vlx_seek_options* opts = aicore_vlx_seek_options_new();
        aicore_vlx_seek_options_set_device(
                opts, settings.device.toUtf8().constData());
        aicore_vlx_seek_options_set_threads(opts, settings.threads);
        aicore_vlx_seek_options_set_max_new_tokens(opts, settings.maxNewTokens);
        // Paths are the loading truth; the precision string keeps the
        // reported model_info metadata consistent with the loaded rows.
        aicore_vlx_seek_options_set_precision(opts,
                                              precision.toUtf8().constData());
        aicore_vlx_seek_options_set_wedetect_gguf(
                opts, settings.wedetectPath.toUtf8().constData());
        aicore_vlx_seek_options_set_vision_gguf(
                opts, settings.visionPath.toUtf8().constData());
        aicore_vlx_seek_options_set_llm_gguf(
                opts, settings.llmPath.toUtf8().constData());
        aicore_vlx_seek_ctx* ctx = aicore_vlx_seek_load_opts(opts);
        aicore_vlx_seek_options_free(opts);
        if (ctx == nullptr || !aicore_vlx_seek_is_ready(ctx)) {
            const char* msg = ctx ? aicore_vlx_seek_last_error(ctx) : nullptr;
            if (ctx) aicore_vlx_seek_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg ? QString::fromUtf8(msg)
                                              : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_vlx_seek_ctx, void (*)(aicore_vlx_seek_ctx*)>
                guard(ctx, aicore_vlx_seek_free);

        QString error2;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error2)) return cmd.error(error2);

        B::Manifest manifest(s);
        manifest.set("task", settings.task);
        manifest.set("text", settings.text);
        manifest.set("lang", settings.lang);
        manifest.set("precision", precision);
        {
            char* info = aicore_vlx_seek_info_json(ctx);
            if (info) {
                manifest.set(
                        "model_info",
                        QJsonDocument::fromJson(QByteArray(info)).object());
                aicore_vlx_seek_free_buffer(info);
            }
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec, QString& err) {
                    rec["input"] = item.path;
                    VLXSeekWorker::Settings one = settings;
                    one.inputPath = item.path;
                    VLXSeekRunResult result;
                    QImage annotated;
                    if (!VLXSeekWorker::runImage(ctx, one, &result, &annotated,
                                                 &err)) {
                        return false;
                    }
                    QJsonArray boxes;
                    for (const VLXSeekRunResult::Box& b : result.boxes) {
                        boxes.append(QJsonObject{{"xmin", b.rect.left()},
                                                 {"ymin", b.rect.top()},
                                                 {"xmax", b.rect.right()},
                                                 {"ymax", b.rect.bottom()},
                                                 { "label",
                                                   b.label }});
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_vlxseek.json");
                    if (!B::saveJson(QJsonObject{{"image", item.path},
                                                 {"task", settings.task},
                                                 {"answer", result.answer},
                                                 {"bbox_format", "x1,y1,x2,y2"},
                                                 {"boxes", boxes},
                                                 {"prompt_tokens",
                                                  result.promptTokens},
                                                 { "completion_tokens",
                                                   result.completionTokens }},
                                     jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "vlxseek", jsonPath }};
                    const QString visPath =
                            B::outPath(s, item.stem, "_vlxseek_vis.png");
                    if (!annotated.isNull() &&
                        !B::saveImage(annotated, visPath, &err)) {
                        return false;
                    }
                    outputs["visualization"] = visPath;
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_vlx_seek_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["answer"] = result.answer;
                    rec["boxes"] = result.boxes.size();
                    rec["proposals"] = result.proposalCount;
                    rec["device"] = result.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_VLXSEEK_RUN));
    }
#endif  // AICore_ENABLED
};
