// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch YOLO prediction (-YOLO_PREDICT): one model load, N still
// images -> <stem>_<task>.json (+ previews, masks, dense maps) each, plus
// the shared ecvAICoreBatch RESULT_JSON manifest. Covers every model task the
// dialog runs (detect / segment / pose / obb / classify / semantic / depth,
// and the open-vocabulary world / yoloe families); the GGUF decides the
// task. Load and per-image bodies are YOLOWorker::loadContext/inferImage,
// the exact code path of the dialog. Video tracking stays on -YOLO_TRACK.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/yolo_capi.h>

#include <QRectF>

#include "YOLOModelCatalog.h"
#include "YOLOWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_YOLO_PREDICT "YOLO_PREDICT"

struct CommandYoloPredict : public ccCommandLineInterface::Command {
    CommandYoloPredict()
        : ccCommandLineInterface::Command(QObject::tr("YOLO batch prediction"),
                                          COMMAND_YOLO_PREDICT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"TASK",
                 "choice:detect|segment|pose|obb|classify|semantic|depth|"
                 "world|yoloe",
                 "detect",
                 "Picks the catalog default model when MODEL is omitted (the "
                 "loaded GGUF always decides the actual task)"},
                {"CONF", "float", "0.25", "Confidence threshold"},
                {"IOU", "float", "0.7", "NMS IoU threshold"},
                {"TOP_K", "int", "300", "Maximum detections per image"},
                {"CLASSES", "list", "",
                 "world/yoloe: comma-separated class prompts (Chinese is "
                 "translated like the dialog)"},
                {"TEXT_MODEL", "string", "",
                 "world/yoloe text tower (default: family CLIP/MobileCLIP)"},
                {"VISUAL_PROMPT", "list", "",
                 "yoloe: x1,y1,x2,y2 example box in pixels; repeatable, "
                 "applied to every image (fixed-camera batches)"},
                {"VISUAL_PROMPT_NAMES", "list", "",
                 "Names index-aligned with VISUAL_PROMPT"},
                {"SAVE_VIS", "bool", "1", "Write <stem>_vis.png"},
                {"SAVE_MASKS", "bool", "1",
                 "segment: <stem>_mask_<i>.png; semantic: "
                 "<stem>_classmap.png"},
                {"SAVE_DEPTH", "bool", "1",
                 "depth: <stem>_depth.npy (float32 meters)"},
                {"CLASSIFY_TOP_K", "int", "5",
                 "classify: classes kept in the result JSON"},
        };
        return table;
    }

    static QJsonObject detJson(const YOLODetection& d) {
        return QJsonObject{{"class_id", static_cast<int>(d.classId)},
                           {"class_name", d.className},
                           {"score", d.score},
                           { "bbox",
                             QJsonArray {
                                 d.x1,
                                 d.y1,
                                 d.x2,
                                 d.y2
                             } }};
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_YOLO_PREDICT, QStringLiteral("yolo"),
                           QStringLiteral("YOLO detect/segment/pose/obb/"
                                          "classify/semantic/depth/world/"
                                          "yoloe"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = YOLOHelpers::modelCacheDir();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const YOLOModelEntry& e : YOLOHelpers::catalogModels()) {
                rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                QDir(cacheDir).filePath(e.filename), e.task});
            }
            return B::listModels(cmd, s, rows);
        }

        const QString task =
                s.args.choice("TASK", QStringLiteral("detect"),
                              {"detect", "segment", "pose", "obb", "classify",
                               "semantic", "depth", "world", "yoloe"});
        YOLOWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.confThres = s.args.toFloat("CONF", 0.25f);
        settings.iouThres = s.args.toFloat("IOU", 0.7f);
        settings.topK = static_cast<uint32_t>(s.args.toInt("TOP_K", 300));
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        const bool saveMasks = s.args.toBool("SAVE_MASKS", true);
        const bool saveDepth = s.args.toBool("SAVE_DEPTH", true);
        const int classifyTopK = s.args.toInt("CLASSIFY_TOP_K", 5);
        for (const QString& box : s.args.list("VISUAL_PROMPT")) {
            const QStringList v = box.split(',');
            bool ok = v.size() == 4;
            float c[4] = {0, 0, 0, 0};
            for (int i = 0; ok && i < 4; ++i) c[i] = v[i].toFloat(&ok);
            if (!ok || c[2] <= c[0] || c[3] <= c[1]) {
                return cmd.error(QStringLiteral("VISUAL_PROMPT expects "
                                                "x1,y1,x2,y2 with x2>x1, "
                                                "y2>y1, got '%1'")
                                         .arg(box));
            }
            settings.visualPrompts.append(
                    QRectF(QPointF(c[0], c[1]), QPointF(c[2], c[3])));
        }
        settings.visualPromptNames = s.args.tokens("VISUAL_PROMPT_NAMES");
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        // Model: explicit path/filename, else the catalog default of TASK.
        const QVector<YOLOModelEntry> taskRows = YOLOHelpers::taskModels(task);
        const int defIdx = YOLOHelpers::defaultModelIndexForTask(task);
        settings.modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir,
                                    defIdx >= 0 && defIdx < taskRows.size()
                                            ? taskRows[defIdx].filename
                                            : QString());
        const QString modelFile = QFileInfo(settings.modelPath).fileName();
        YOLOModelEntry entry;
        YOLOHelpers::findModelByFilename(modelFile, &entry);
        if (!B::ensureModel(cmd, settings.modelPath,
                            {entry.downloadUrl, QString()}, s.download,
                            &error)) {
            return cmd.error(error);
        }

        // Open vocabulary: same class-list rules as YOLODialog::settings().
        const QString family = YOLOHelpers::openVocabularyFamily(modelFile);
        const bool promptFree = YOLOHelpers::isPromptFreeFilename(modelFile);
        if (!family.isEmpty() && !promptFree &&
            settings.visualPrompts.isEmpty()) {
            for (const QString& name : s.args.tokens("CLASSES")) {
                bool translated = false;
                settings.classes.append(YOLOHelpers::translatePromptToEnglish(
                        name, &translated));
            }
            if (!settings.classes.isEmpty()) {
                settings.textModelPath = B::resolveModelPath(
                        s.args.str("TEXT_MODEL"), cacheDir,
                        YOLOHelpers::defaultTextModelForTask(family));
                YOLOModelEntry textEntry;
                YOLOHelpers::findModelByFilename(
                        QFileInfo(settings.textModelPath).fileName(),
                        &textEntry);
                if (!B::ensureModel(cmd, settings.textModelPath,
                                    {textEntry.downloadUrl, QString()},
                                    s.download, &error)) {
                    return cmd.error(error);
                }
            }
        } else if (s.args.has("CLASSES")) {
            cmd.warning(QStringLiteral("[%1] CLASSES ignored: %2 is not a "
                                       "text-prompted open-vocabulary model")
                                .arg(COMMAND_YOLO_PREDICT, modelFile));
        }

        aicore_yolo_ctx* ctx = YOLOWorker::loadContext(settings, &error);
        if (!error.isEmpty()) {
            if (ctx) aicore_yolo_free(ctx);
            return cmd.error(error);
        }
        std::unique_ptr<aicore_yolo_ctx, void (*)(aicore_yolo_ctx*)> guard(
                ctx, aicore_yolo_free);
        const QString modelTask =
                QString::fromUtf8(aicore_yolo_context_task(ctx));

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        manifest.set("model_task", modelTask);
        manifest.set("model_name",
                     QString::fromUtf8(aicore_yolo_context_model_name(ctx)));
        manifest.set("num_classes",
                     static_cast<int>(aicore_yolo_context_num_classes(ctx)));
        if (!settings.classes.isEmpty()) {
            manifest.set("classes",
                         QJsonArray::fromStringList(settings.classes));
            manifest.set("text_model", settings.textModelPath);
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    YOLOWorker::Settings one = settings;
                    one.inputPath = item.path;
                    YOLOWorker::ImageResult out;
                    if (!YOLOWorker::inferImage(ctx, one, nullptr, &out,
                                                &err)) {
                        return false;
                    }
                    QJsonObject json{{"image", item.path},
                                     { "task",
                                       modelTask }};
                    QJsonObject outputs;
                    QImage preview;
                    if (out.isDepth) {
                        const YOLODepthResult& d = out.depth;
                        json["width"] = d.width;
                        json["height"] = d.height;
                        json["stats"] = QJsonObject{
                                {"min", d.stats.minDepth},
                                {"max", d.stats.maxDepth},
                                {"mean", d.stats.meanDepth},
                                {"p95", d.stats.p95Depth},
                                { "valid_pixels",
                                  static_cast<double>(d.stats.validPixels) }};
                        if (saveDepth) {
                            const QString p =
                                    B::outPath(s, item.stem, "_depth.npy");
                            if (!B::saveNpyFloat32(d.depthMap.constData(),
                                                   d.height, d.width, 1, p,
                                                   &err)) {
                                return false;
                            }
                            outputs["depth"] = p;
                        }
                        preview = d.annotatedImage;
                        rec["device"] = d.resolvedDevice;
                    } else {
                        const YOLORunResult& r = out.run;
                        json["width"] = r.annotatedImage.width();
                        json["height"] = r.annotatedImage.height();
                        json["bbox_format"] = "x1,y1,x2,y2";
                        QJsonArray dets;
                        for (int i = 0; i < r.detections.size(); ++i) {
                            QJsonObject o = detJson(r.detections[i]);
                            if (saveMasks && i < r.masks.size() &&
                                r.masks[i].h > 0) {
                                const YOLOSegMask& m = r.masks[i];
                                // 0/1 bytes, stride = bits/h -> 0/255 PNG.
                                QImage mask(m.w, m.h,
                                            QImage::Format_Grayscale8);
                                const int stride = m.bits.size() / m.h;
                                for (int y = 0; y < m.h; ++y) {
                                    const uchar* src =
                                            reinterpret_cast<const uchar*>(
                                                    m.bits.constData()) +
                                            y * stride;
                                    uchar* dst = mask.scanLine(y);
                                    for (int x = 0; x < m.w; ++x)
                                        dst[x] = src[x] ? 255 : 0;
                                }
                                const QString p = B::outPath(
                                        s, item.stem,
                                        QStringLiteral("_mask_%1.png").arg(i));
                                if (!B::saveImage(mask, p, &err)) return false;
                                o["mask"] = p;
                            }
                            dets.append(o);
                        }
                        for (const YOLOKeypointSet& k : r.keypointSets) {
                            QJsonObject o = detJson(k.det);
                            QJsonArray kpts;
                            for (const YOLOKeypoint& p : k.kpts)
                                kpts.append(QJsonArray{p.x, p.y, p.visibility});
                            o["keypoints"] = kpts;
                            dets.append(o);
                        }
                        for (const YOLOObbBox& b : r.obbBoxes) {
                            dets.append(QJsonObject{
                                    {"class_id", static_cast<int>(b.classId)},
                                    {"class_name", b.className},
                                    {"score", b.score},
                                    { "obb",
                                      QJsonArray {
                                          b.cx,
                                          b.cy,
                                          b.w,
                                          b.h,
                                          b.angle
                                      } }});
                        }
                        if (!r.obbBoxes.isEmpty())
                            json["obb_format"] = "cx,cy,w,h,angle_rad";
                        if (!r.classifications.isEmpty()) {
                            QVector<YOLOClassProb> sorted = r.classifications;
                            std::sort(sorted.begin(), sorted.end(),
                                      [](const YOLOClassProb& a,
                                         const YOLOClassProb& b) {
                                          return a.prob > b.prob;
                                      });
                            QJsonArray top;
                            for (int i = 0;
                                 i < sorted.size() && i < classifyTopK; ++i) {
                                top.append(QJsonObject{
                                        {"class_id",
                                         static_cast<int>(sorted[i].classId)},
                                        {"class_name", sorted[i].className},
                                        { "prob",
                                          sorted[i].prob }});
                            }
                            json["classifications"] = top;
                            rec["top1"] = sorted.front().className;
                        }
                        if (!r.semanticClassMap.isEmpty()) {
                            json["num_classes"] = r.semanticNumClasses;
                            if (saveMasks) {
                                // One class id per pixel (row-major w*h).
                                const QImage map(
                                        reinterpret_cast<const uchar*>(
                                                r.semanticClassMap.constData()),
                                        r.semanticWidth, r.semanticHeight,
                                        r.semanticClassMap.size() /
                                                r.semanticHeight,
                                        QImage::Format_Grayscale8);
                                const QString p = B::outPath(s, item.stem,
                                                             "_classmap.png");
                                if (!B::saveImage(map, p, &err)) return false;
                                outputs["class_map"] = p;
                            }
                        }
                        if (!dets.isEmpty() || r.classifications.isEmpty())
                            json["detections"] = dets;
                        rec["detections"] = dets.size();
                        preview = r.annotatedImage;
                        rec["device"] = r.resolvedDevice;
                    }
                    const QString jsonPath = B::outPath(
                            s, item.stem,
                            QStringLiteral("_%1.json").arg(modelTask));
                    if (!B::saveJson(json, jsonPath, &err)) return false;
                    outputs["result"] = jsonPath;
                    if (saveVis && !preview.isNull()) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(preview, p, &err)) return false;
                        outputs["visualization"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_yolo_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["summary"] = out.summary;
                    if (!out.hint.isEmpty()) rec["hint"] = out.hint;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_YOLO_PREDICT));
    }
#endif  // AICore_ENABLED
};
