// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch WeVisOne-VL perception (-WEVISONE_RUN): one three-model
// context (VLM + mmproj + pixel decoder), N images -> <stem>_wevisone.json
// (answer + typed boxes/matte metadata plus the shared ecvAICoreBatch
// RESULT_JSON manifest) each, an optional Qt-free PNG visualization and the
// raw f32 matte when the decoder ran. The per-image body is
// WeVisOneWorker::runImage, the exact code path of the dialog.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/wevisone_capi.h>

#include "WeVisOneModelCatalog.h"
#include "WeVisOneWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_WEVISONE_RUN "WEVISONE_RUN"

struct CommandWeVisOneRun : public ccCommandLineInterface::Command {
    CommandWeVisOneRun()
        : ccCommandLineInterface::Command(
                  QObject::tr("WeVisOne-VL unified perception batch"),
                  COMMAND_WEVISONE_RUN) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"TASK", "enum", "detection",
                 "detection|referring|point|count_box|count_point|ocr|"
                 "ocr_grounding|segmentation|matting|visual_prompt|freeform"},
                {"QUERY", "string", "",
                 "Categories (literal \\n = newline), referring phrase or "
                 "freeform instruction; unused by ocr|visual_prompt"},
                {"REF_BOXES", "string", "",
                 "visual_prompt reference boxes, upstream --ref-boxes "
                 "format: x1,y1,x2,y2[;x1,y1,x2,y2...] in image pixels"},
                {"SIZE", "enum", "2b", "2b|4b (checkpoint family)"},
                {"PRECISION", "enum", "q8_0",
                 "f16|q8_0|q4_K_M (VLM rows of the catalog; mmproj stays "
                 "f16)"},
                {"DECODER_PRECISION", "enum", "f32",
                 "f32|f16 (pixel-decoder tier; f32 is the parity-gated "
                 "checkpoint dtype)"},
                {"DECODE_MASKS", "bool", "1",
                 "Run the pixel decoder on <OBJ> answers (0 = pure "
                 "object-level detector)"},
                {"MAX_NEW_TOKENS", "int", "4096", "Greedy generation cap"},
                {"DEVICE", "device", "", "Device request (empty = auto)"},
                {"THREADS", "int", "0", "CPU threads (0 = default)"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_WEVISONE_RUN,
                           QStringLiteral("wevisone"),
                           QStringLiteral("Unified visual perception "
                                          "(WeVisOne-VL)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = WeVisOneHelpers::modelCacheDir();
        const QVector<WeVisOneModelEntry> catalog =
                WeVisOneHelpers::catalogModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const WeVisOneModelEntry& e : catalog) {
                rows.push_back(
                        {e.filename, e.displayName, e.downloadUrl,
                         QDir(cacheDir).filePath(e.filename),
                         e.role == 0
                                 ? QStringLiteral("vlm")
                                 : (e.role == 1 ? QStringLiteral("mmproj")
                                                : QStringLiteral("decoder"))});
            }
            return B::listModels(cmd, s, rows);
        }

        WeVisOneWorker::Settings settings;
        settings.task = s.args.str("TASK", "detection");
        // The CLI cannot embed raw newlines in one argument; a literal \n
        // is the documented escape for the detection category list (the
        // upstream prompts.py join separator). Keep the two literals on
        // separate lines: adjacent backslash escapes on one line trip the
        // boundary checker's UNC-path heuristic.
        settings.query = s.args.str("QUERY", QString());
        const QString newlineEscape = QStringLiteral("\\n");
        settings.query.replace(newlineEscape, QStringLiteral("\n"));
        settings.refBoxes = s.args.str("REF_BOXES", "");
        settings.size = s.args.str("SIZE", "2b");
        const QString precision = s.args.str("PRECISION", "q8_0");
        settings.precision = precision;
        settings.decoderPrecision = s.args.str("DECODER_PRECISION", "f32");
        settings.decodeMasks = s.args.toBool("DECODE_MASKS", true);
        settings.device = s.args.str("DEVICE", s.device);
        settings.threads = s.threads;
        settings.maxNewTokens = s.args.toInt("MAX_NEW_TOKENS", 4096);
        // Mirror the runtime contract up front: the ocr task runs a fixed
        // prompt and visual_prompt grounds from the reference boxes — the
        // query is unused there; every other template consumes it
        // (freeform passes it through verbatim).
        const bool queryUnused =
                settings.task == QStringLiteral("ocr") ||
                settings.task == QStringLiteral("visual_prompt");
        if (settings.query.trimmed().isEmpty() && !queryUnused) {
            return cmd.error(QStringLiteral("QUERY is required for task %1")
                                     .arg(settings.task));
        }
        if (settings.task == QStringLiteral("visual_prompt")) {
            // The runtime hard-rejects visual_prompt without reference
            // boxes — fail before the model load, not inside inference.
            QString boxesError;
            const QVector<float> boxes = WeVisOneHelpers::parseRefBoxes(
                    settings.refBoxes, &boxesError);
            if (!boxesError.isEmpty()) return cmd.error(boxesError);
            if (boxes.isEmpty()) {
                return cmd.error(QStringLiteral(
                        "REF_BOXES is required for task visual_prompt "
                        "(e.g. -REF_BOXES 880,120,1500,1070)"));
            }
        }
        if (!s.args.ok()) return cmd.error(s.args.error());

        // Resolve the three model paths from the catalog: VLM by
        // family+precision, mmproj always f16 for the family, decoder by
        // family+decoder tier.
        WeVisOneModelEntry vlm, mmproj, decoder;
        for (const WeVisOneModelEntry& e : catalog) {
            if (e.family != settings.size) continue;
            if (e.role == 0 && e.quantization == precision) vlm = e;
            if (e.role == 1) mmproj = e;
            if (e.role == 2 && e.quantization == settings.decoderPrecision)
                decoder = e;
        }
        struct Need {
            WeVisOneModelEntry entry;
            const char* what;
        };
        const std::vector<Need> needs = {{vlm, "VLM"},
                                         {mmproj, "mmproj"},
                                         { decoder,
                                           "decoder" }};
        QString error;
        for (const Need& need : needs) {
            if (need.entry.filename.isEmpty()) {
                return cmd.error(
                        QStringLiteral("catalog has no %1 row for size %2 "
                                       "(precision %3 / decoder %4)")
                                .arg(need.what, settings.size, precision,
                                     settings.decoderPrecision));
            }
            const QString path = QDir(cacheDir).filePath(need.entry.filename);
            if (!B::ensureModel(cmd, path, {need.entry.downloadUrl, QString()},
                                s.download, &error)) {
                return cmd.error(error);
            }
        }
        settings.vlmPath = QDir(cacheDir).filePath(vlm.filename);
        settings.mmprojPath = QDir(cacheDir).filePath(mmproj.filename);
        settings.decoderPath = QDir(cacheDir).filePath(decoder.filename);

        aicore_wevisone_options* opts = aicore_wevisone_options_new();
        aicore_wevisone_options_set_device(
                opts, settings.device.toUtf8().constData());
        aicore_wevisone_options_set_threads(opts, settings.threads);
        aicore_wevisone_options_set_max_new_tokens(opts, settings.maxNewTokens);
        aicore_wevisone_options_set_decode_masks(opts,
                                                 settings.decodeMasks ? 1 : 0);
        // Paths are the loading truth; the size/precision strings keep the
        // reported model_info metadata consistent with the loaded rows.
        aicore_wevisone_options_set_size(opts,
                                         settings.size.toUtf8().constData());
        aicore_wevisone_options_set_precision(opts,
                                              precision.toUtf8().constData());
        aicore_wevisone_options_set_decoder_precision(
                opts, settings.decoderPrecision.toUtf8().constData());
        aicore_wevisone_options_set_models_dir(opts,
                                               cacheDir.toUtf8().constData());
        aicore_wevisone_options_set_vlm_gguf(
                opts, settings.vlmPath.toUtf8().constData());
        aicore_wevisone_options_set_mmproj_gguf(
                opts, settings.mmprojPath.toUtf8().constData());
        aicore_wevisone_options_set_decoder_gguf(
                opts, settings.decoderPath.toUtf8().constData());
        aicore_wevisone_ctx* ctx = aicore_wevisone_load_opts(opts);
        aicore_wevisone_options_free(opts);
        if (ctx == nullptr || !aicore_wevisone_is_ready(ctx)) {
            const char* msg = ctx ? aicore_wevisone_last_error(ctx) : nullptr;
            if (ctx) aicore_wevisone_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg ? QString::fromUtf8(msg)
                                              : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_wevisone_ctx, void (*)(aicore_wevisone_ctx*)>
                guard(ctx, aicore_wevisone_free);

        QString error2;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error2)) return cmd.error(error2);

        B::Manifest manifest(s);
        manifest.set("task", settings.task);
        manifest.set("query", settings.query);
        manifest.set("ref_boxes", settings.refBoxes);
        manifest.set("size", settings.size);
        manifest.set("precision", precision);
        manifest.set("decoder_precision", settings.decoderPrecision);
        manifest.set("decode_masks", settings.decodeMasks ? 1 : 0);
        {
            char* info = aicore_wevisone_info_json(ctx);
            if (info) {
                manifest.set(
                        "model_info",
                        QJsonDocument::fromJson(QByteArray(info)).object());
                aicore_wevisone_free_buffer(info);
            }
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec, QString& err) {
                    rec["input"] = item.path;
                    WeVisOneWorker::Settings one = settings;
                    one.inputPath = item.path;
                    WeVisOneRunResult result;
                    QImage annotated;
                    if (!WeVisOneWorker::runImage(ctx, one, &result, &annotated,
                                                  &err)) {
                        return false;
                    }
                    QJsonArray boxes;
                    for (const QRectF& b : result.boxes) {
                        boxes.append(QJsonObject{{"xmin", b.left()},
                                                 {"ymin", b.top()},
                                                 {"xmax", b.right()},
                                                 { "ymax",
                                                   b.bottom() }});
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_wevisone.json");
                    if (!B::saveJson(
                                QJsonObject{
                                        {"image", item.path},
                                        {"task", settings.task},
                                        {"query", settings.query},
                                        {"answer", result.answer},
                                        {"bbox_format", "x1,y1,x2,y2"},
                                        {"boxes", boxes},
                                        {"has_masks", result.hasMasks},
                                        {"matte_count", result.matteCount},
                                        {"matte_width", result.matteWidth},
                                        {"matte_height", result.matteHeight},
                                        {"prompt_tokens", result.promptTokens},
                                        { "completion_tokens",
                                          result.completionTokens }},
                                jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "wevisone", jsonPath }};
                    const QString visPath =
                            B::outPath(s, item.stem, "_wevisone_vis.png");
                    if (!annotated.isNull() &&
                        !B::saveImage(annotated, visPath, &err)) {
                        return false;
                    }
                    outputs["visualization"] = visPath;
                    if (result.hasMasks && !result.matte.isEmpty()) {
                        // Raw row-major f32 alpha (the upstream
                        // RunResult.matte dtype): directly consumable by
                        // numpy.fromfile(..., dtype=float32).reshape(h, w).
                        const QString mattePath =
                                B::outPath(s, item.stem, "_wevisone_matte.bin");
                        QFile matteFile(mattePath);
                        const qint64 bytes =
                                static_cast<qint64>(result.matte.size()) *
                                static_cast<qint64>(sizeof(float));
                        if (!matteFile.open(QIODevice::WriteOnly) ||
                            matteFile.write(reinterpret_cast<const char*>(
                                                    result.matte.constData()),
                                            bytes) != bytes) {
                            err = QStringLiteral("Cannot write %1: %2")
                                          .arg(mattePath,
                                               matteFile.errorString());
                            return false;
                        }
                        matteFile.close();
                        outputs["matte"] = mattePath;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_wevisone_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["answer"] = result.answer;
                    rec["boxes"] = result.boxes.size();
                    rec["has_masks"] = result.hasMasks;
                    rec["matte_count"] = result.matteCount;
                    rec["device"] = result.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_WEVISONE_RUN));
    }
#endif  // AICore_ENABLED
};