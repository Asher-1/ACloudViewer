// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch segmentation (-SAM3_SEGMENT): one SAM3/SAM2 context, N
// images -> <stem>_sam3.json (+ blended overlay and per-instance 0/255 masks)
// each, plus the shared ecvAICoreBatch RESULT_JSON manifest. Prompts are
// either global (TEXT / POINT / BOX / EXEMPLAR ...) or per image through
// PROMPT_FILE. The per-image body is SAM3Worker::segment + buildResult, the
// dialog code path.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/sam3_capi.h>

#include <QFile>
#include <QJsonArray>

#include "SAM3Worker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_SAM3_SEGMENT "SAM3_SEGMENT"

struct CommandSam3Segment : public ccCommandLineInterface::Command {
    CommandSam3Segment()
        : ccCommandLineInterface::Command(
                  QObject::tr("SAM3 batch segmentation"),
                  COMMAND_SAM3_SEGMENT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MODE", "choice:auto|pcs|pvs", "auto",
                 "pcs = text/exemplar concept segmentation, pvs = point/box; "
                 "auto picks pcs when TEXT or EXEMPLAR is given"},
                {"TEXT", "string", "", "PCS text prompt, e.g. 'person'"},
                {"EXEMPLAR", "list", "",
                 "PCS positive exemplar box x0,y0,x1,y1 (repeatable)"},
                {"NEG_EXEMPLAR", "list", "",
                 "PCS negative exemplar box x0,y0,x1,y1 (repeatable)"},
                {"POINT", "list", "", "PVS positive point x,y (repeatable)"},
                {"NEG_POINT", "list", "",
                 "PVS negative point x,y (repeatable)"},
                {"BOX", "string", "", "PVS box x0,y0,x1,y1"},
                {"MULTIMASK", "bool", "0", "PVS: best of 3 candidate masks"},
                {"PROMPT_FILE", "string", "",
                 "JSON {\"<file name|stem>\": {text, points, neg_points, box, "
                 "exemplars, neg_exemplars}} per-image prompts (override the "
                 "global ones; images without an entry use the global ones)"},
                {"SCORE_THRESH", "float", "0.5", "PCS score threshold"},
                {"NMS_THRESH", "float", "0.1", "PCS NMS IoU threshold"},
                {"ENCODE_SIZE", "int", "0",
                 "Encoder input size (0 = model default)"},
                {"SAVE_VIS", "bool", "1", "Write <stem>_vis.png overlay"},
                {"SAVE_MASKS", "bool", "1", "Write <stem>_mask_<i>.png"},
        };
        return table;
    }

    /** Comma/space separated floats; exactly \p n values or false. */
    static bool parseFloats(const QString& text, int n, float* out) {
        const QStringList parts = QString(text).replace(',', ' ').split(
                ' ', QtCompat::SkipEmptyParts);
        if (parts.size() != n) return false;
        for (int i = 0; i < n; ++i) {
            bool ok = false;
            out[i] = parts[i].toFloat(&ok);
            if (!ok) return false;
        }
        return true;
    }

    static bool jsonFloats(const QJsonValue& v, int n, float* out) {
        const QJsonArray a = v.toArray();
        if (a.size() != n) return false;
        for (int i = 0; i < n; ++i) {
            if (!a[i].isDouble()) return false;
            out[i] = static_cast<float>(a[i].toDouble());
        }
        return true;
    }

    /** Fills \p p from one PROMPT_FILE entry. */
    static bool promptFromJson(const QJsonObject& o,
                               SAM3Prompt* p,
                               QString* error) {
        if (o.contains("text")) {
            const QByteArray t = o["text"].toString().toUtf8();
            qstrncpy(p->text, t.constData(), sizeof(p->text));
        }
        auto points = [&](const char* key, QVector<aicore_sam3_point>* dst) {
            if (!o.contains(key)) return true;
            dst->clear();
            for (const QJsonValue& v : o[key].toArray()) {
                float xy[2];
                if (!jsonFloats(v, 2, xy)) return false;
                dst->push_back({xy[0], xy[1]});
            }
            return true;
        };
        auto boxes = [&](const char* key, std::vector<aicore_sam3_box>* dst) {
            if (!o.contains(key)) return true;
            dst->clear();
            for (const QJsonValue& v : o[key].toArray()) {
                float b[4];
                if (!jsonFloats(v, 4, b)) return false;
                dst->push_back({b[0], b[1], b[2], b[3]});
            }
            return true;
        };
        if (!points("points", &p->posPoints) ||
            !points("neg_points", &p->negPoints)) {
            *error = QStringLiteral("points must be [[x,y],...]");
            return false;
        }
        if (!boxes("exemplars", &p->posExemplars) ||
            !boxes("neg_exemplars", &p->negExemplars)) {
            *error = QStringLiteral("exemplars must be [[x0,y0,x1,y1],...]");
            return false;
        }
        if (o.contains("box")) {
            float b[4];
            if (!jsonFloats(o["box"], 4, b)) {
                *error = QStringLiteral("box must be [x0,y0,x1,y1]");
                return false;
            }
            p->pvsBox = {b[0], b[1], b[2], b[3]};
            p->usePvsBox = true;
        }
        return true;
    }

    static ecvAICoreBatch::ModelSource source(const QString& path) {
        const aicore_sam3_model_entry* e = aicore_sam3_model_by_filename(
                QFileInfo(path).fileName().toUtf8().constData());
        return {e && e->download_url ? QString::fromUtf8(e->download_url)
                                     : QString(),
                QString()};
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_SAM3_SEGMENT, QStringLiteral("sam3"),
                           QStringLiteral("Promptable segmentation (SAM3 "
                                          "text/exemplar, SAM3/SAM2 "
                                          "point/box)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir =
                B::takeCString(aicore_sam3_model_cache_dir(), [](void* p) {
                    aicore_sam3_free_buffer(static_cast<char*>(p));
                });
        QString defaultModel;  // first full SAM3 row (text + visual prompts)
        std::vector<B::CatalogRow> rows;
        for (int i = 0; i < aicore_sam3_model_count(); ++i) {
            const aicore_sam3_model_entry* e = aicore_sam3_model_at(i);
            if (!e || !e->filename) continue;
            const QString fn = QString::fromUtf8(e->filename);
            const QString family =
                    QString::fromUtf8(e->model_family ? e->model_family : "");
            if (defaultModel.isEmpty() && family == QLatin1String("sam3"))
                defaultModel = fn;
            rows.push_back({fn, QString::fromUtf8(e->display_name),
                            QString::fromUtf8(e->download_url),
                            QDir(cacheDir).filePath(fn), family});
        }
        if (s.listModels) return B::listModels(cmd, s, rows);

        SAM3Prompt global;
        global.multimask = s.args.toBool("MULTIMASK", false);
        global.scoreThreshold = s.args.toFloat("SCORE_THRESH", 0.5f);
        global.nmsThreshold = s.args.toFloat("NMS_THRESH", 0.1f);
        const int encodeSize = s.args.toInt("ENCODE_SIZE", 0);
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        const bool saveMasks = s.args.toBool("SAVE_MASKS", true);
        const QString mode = s.args.choice("MODE", QStringLiteral("auto"),
                                           {"auto", "pcs", "pvs"});
        if (!s.args.ok()) return cmd.error(s.args.error());
        {
            const QByteArray t = s.args.str("TEXT").toUtf8();
            qstrncpy(global.text, t.constData(), sizeof(global.text));
        }
        for (const char* key : {"POINT", "NEG_POINT"}) {
            for (const QString& v : s.args.list(key)) {
                float xy[2];
                if (!parseFloats(v, 2, xy))
                    return cmd.error(QStringLiteral("%1 expects x,y, got '%2'")
                                             .arg(key, v));
                (qstrcmp(key, "POINT") == 0 ? global.posPoints
                                            : global.negPoints)
                        .push_back({xy[0], xy[1]});
            }
        }
        for (const char* key : {"EXEMPLAR", "NEG_EXEMPLAR"}) {
            for (const QString& v : s.args.list(key)) {
                float b[4];
                if (!parseFloats(v, 4, b))
                    return cmd.error(
                            QStringLiteral("%1 expects x0,y0,x1,y1, got '%2'")
                                    .arg(key, v));
                (qstrcmp(key, "EXEMPLAR") == 0 ? global.posExemplars
                                               : global.negExemplars)
                        .push_back({b[0], b[1], b[2], b[3]});
            }
        }
        if (s.args.has("BOX")) {
            float b[4];
            if (!parseFloats(s.args.str("BOX"), 4, b))
                return cmd.error(QStringLiteral("BOX expects x0,y0,x1,y1"));
            global.pvsBox = {b[0], b[1], b[2], b[3]};
            global.usePvsBox = true;
        }
        QJsonObject promptFile;
        if (s.args.has("PROMPT_FILE")) {
            QFile f(s.args.str("PROMPT_FILE"));
            QJsonParseError pe{};
            const QJsonDocument doc =
                    f.open(QIODevice::ReadOnly)
                            ? QJsonDocument::fromJson(f.readAll(), &pe)
                            : QJsonDocument();
            if (!doc.isObject()) {
                return cmd.error(QStringLiteral("PROMPT_FILE %1: not a JSON "
                                                "object (%2)")
                                         .arg(f.fileName(), pe.errorString()));
            }
            promptFile = doc.object();
        }

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        const QString modelPath = B::resolveModelPath(s.args.str("MODEL"),
                                                      cacheDir, defaultModel);
        if (!B::ensureModel(cmd, modelPath, source(modelPath), s.download,
                            &error)) {
            return cmd.error(error);
        }

        aicore_sam3_options* opts = aicore_sam3_options_new();
        aicore_sam3_options_set_device(opts, s.device.toUtf8().constData());
        aicore_sam3_options_set_threads(opts, s.threads > 0 ? s.threads : 4);
        aicore_sam3_options_set_encode_img_size(opts, encodeSize);
        aicore_sam3_options_set_score_threshold(opts, global.scoreThreshold);
        aicore_sam3_options_set_nms_threshold(opts, global.nmsThreshold);
        aicore_sam3_ctx* ctx = aicore_sam3_load_opts(
                QFile::encodeName(modelPath).constData(), opts);
        aicore_sam3_options_free(opts);
        if (ctx == nullptr || !aicore_sam3_is_ready(ctx)) {
            const char* msg = ctx ? aicore_sam3_last_error(ctx)
                                  : aicore_sam3_last_load_error();
            const QString text = QString::fromUtf8(msg ? msg : "unknown");
            if (ctx) aicore_sam3_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1").arg(text));
        }
        std::unique_ptr<aicore_sam3_ctx, void (*)(aicore_sam3_ctx*)> guard(
                ctx, aicore_sam3_free);
        const bool visualOnly = aicore_sam3_context_visual_only(ctx) != 0;
        const char* backend = aicore_sam3_context_backend_name(ctx);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        manifest.set("model_type", aicore_sam3_context_model_type(ctx));
        manifest.set("visual_only", visualOnly);
        manifest.set("backend", QString::fromUtf8(backend ? backend : ""));

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    SAM3Prompt prompt = global;
                    const QFileInfo fi(item.path);
                    for (const QString& key :
                         {fi.fileName(), fi.completeBaseName(), item.path}) {
                        if (!promptFile.contains(key)) continue;
                        if (!promptFromJson(promptFile[key].toObject(), &prompt,
                                            &err)) {
                            err = QStringLiteral("PROMPT_FILE[%1]: %2")
                                          .arg(key, err);
                            return false;
                        }
                        break;
                    }
                    const bool hasConcept = prompt.text[0] != '\0' ||
                                            !prompt.posExemplars.empty();
                    const bool pcs =
                            mode == QLatin1String("pcs") ||
                            (mode == QLatin1String("auto") && hasConcept);
                    if (pcs && visualOnly) {
                        err = QStringLiteral(
                                "visual-only model: PCS (TEXT/EXEMPLAR) is "
                                "unsupported, use POINT/BOX");
                        return false;
                    }
                    if (pcs ? !hasConcept
                            : (prompt.posPoints.isEmpty() &&
                               prompt.negPoints.isEmpty() &&
                               !prompt.usePvsBox)) {
                        err = pcs ? QStringLiteral("PCS needs TEXT or EXEMPLAR")
                                  : QStringLiteral("PVS needs POINT or BOX");
                        return false;
                    }
                    const QImage rgb = B::loadImage(
                            item.path, QImage::Format_RGB888, &err);
                    if (rgb.isNull()) return false;
                    aicore_sam3_seg_result* res = SAM3Worker::segment(
                            ctx, rgb, prompt, pcs, true, &err);
                    if (res == nullptr) {
                        if (err.isEmpty())
                            err = QString::fromUtf8(
                                    aicore_sam3_last_error(ctx));
                        return false;
                    }
                    const SAM3WorkerResult r =
                            SAM3Worker::buildResult(ctx, res, rgb);
                    aicore_sam3_seg_result_free(res);

                    QJsonArray detections;
                    for (int i = 0; i < r.detCount; ++i) {
                        const aicore_sam3_box& b = r.boxes[i];
                        QJsonObject o{{"instance_id", r.instanceIds[i]},
                                      {"score", r.scores[i]},
                                      {"iou", r.ious[i]},
                                      { "bbox",
                                        QJsonArray {
                                            b.x0,
                                            b.y0,
                                            b.x1,
                                            b.y1
                                        } }};
                        if (saveMasks && !r.instanceMasks[i].isNull()) {
                            const QString p = B::outPath(
                                    s, item.stem,
                                    QStringLiteral("_mask_%1.png").arg(i));
                            if (!B::saveImage(r.instanceMasks[i], p, &err))
                                return false;
                            o["mask"] = p;
                        }
                        detections.append(o);
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_sam3.json");
                    if (!B::saveJson(QJsonObject{{"image", item.path},
                                                 {"width", rgb.width()},
                                                 {"height", rgb.height()},
                                                 {"mode", pcs ? "pcs" : "pvs"},
                                                 {"text", QString::fromUtf8(
                                                                  prompt.text)},
                                                 {"bbox_format", "x0,y0,x1,y1"},
                                                 { "detections",
                                                   detections }},
                                     jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "result", jsonPath }};
                    if (saveVis && !r.maskComposite.isNull()) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(r.maskComposite, p, &err))
                            return false;
                        outputs["visualization"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_sam3_last_pipeline_timings(ctx, &t) == 0)
                        rec["timings"] = B::timingsJson(t);
                    rec["mode"] = pcs ? "pcs" : "pvs";
                    rec["detections"] = r.detCount;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_SAM3_SEGMENT));
    }
#endif  // AICore_ENABLED
};
