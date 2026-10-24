// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch object detection / instance segmentation (-RFDETR_DETECT):
// one RF-DETR model load, N images -> <stem>_detections.json (+ annotated
// overlay and per-instance masks for segmentation checkpoints) each, plus the
// shared ecvAICoreBatch RESULT_JSON manifest. The per-image body is
// RFDetrWorker::detectImage, the exact code path of the dialog.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/rfdetr_capi.h>

#include "RFDetrModelCatalog.h"
#include "RFDetrWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_RFDETR_DETECT "RFDETR_DETECT"

struct CommandRfDetrDetect : public ccCommandLineInterface::Command {
    CommandRfDetrDetect()
        : ccCommandLineInterface::Command(
                  QObject::tr("RF-DETR batch detection"),
                  COMMAND_RFDETR_DETECT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"THRESHOLD", "float", "0.5", "Score threshold"},
                {"TOP_K", "int", "300", "Maximum detections per image"},
                {"CLASSES", "list", "",
                 "Class-id allowlist, e.g. 0,2,5 (empty = all classes)"},
                {"SAVE_VIS", "bool", "1",
                 "Write <stem>_vis.png (boxes + mask tint)"},
                {"SAVE_MASKS", "bool", "1",
                 "Segmentation models: write <stem>_mask_<i>.png"},
        };
        return table;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_RFDETR_DETECT, QStringLiteral("rfdetr"),
                           QStringLiteral("Object detection / instance "
                                          "segmentation (RF-DETR)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = RFDetrHelpers::modelCacheDir();
        const QVector<RFDetrModelEntry> catalog =
                RFDetrHelpers::catalogModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const RFDetrModelEntry& e : catalog) {
                rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                QDir(cacheDir).filePath(e.filename),
                                e.segmentationCapable
                                        ? QStringLiteral("segmentation")
                                        : QStringLiteral("detection")});
            }
            return B::listModels(cmd, s, rows);
        }

        RFDetrWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.threshold = s.args.toFloat("THRESHOLD", 0.5f);
        settings.topK = static_cast<uint32_t>(s.args.toInt("TOP_K", 300));
        for (const QString& token : s.args.tokens("CLASSES")) {
            bool ok = false;
            const uint id = token.toUInt(&ok);
            if (!ok) {
                return cmd.error(QStringLiteral("CLASSES expects class ids, "
                                                "got '%1'")
                                         .arg(token));
            }
            settings.classFilter.append(id);
        }
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        const bool saveMasks = s.args.toBool("SAVE_MASKS", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        const int defIdx = RFDetrHelpers::catalogDefaultIndex();
        settings.modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir,
                                    defIdx >= 0 && defIdx < catalog.size()
                                            ? catalog[defIdx].filename
                                            : QString());
        RFDetrModelEntry entry;
        RFDetrHelpers::findModelByFilename(
                QFileInfo(settings.modelPath).fileName(), &entry);
        if (!B::ensureModel(cmd, settings.modelPath,
                            {entry.downloadUrl, QString()}, s.download,
                            &error)) {
            return cmd.error(error);
        }

        aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
        aicore_rfdetr_options_set_device(opts, s.device.toUtf8().constData());
        aicore_rfdetr_options_set_threads(opts, s.threads);
        if (!settings.classFilter.isEmpty()) {
            aicore_rfdetr_options_set_class_filter(
                    opts, settings.classFilter.constData(),
                    static_cast<size_t>(settings.classFilter.size()));
        }
        aicore_rfdetr_ctx* ctx = aicore_rfdetr_load_opts(
                settings.modelPath.toUtf8().constData(), opts);
        aicore_rfdetr_options_free(opts);
        if (ctx == nullptr || !aicore_rfdetr_is_ready(ctx)) {
            const char* msg = ctx ? aicore_rfdetr_last_error(ctx) : nullptr;
            if (ctx) aicore_rfdetr_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg ? QString::fromUtf8(msg)
                                              : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_rfdetr_ctx, void (*)(aicore_rfdetr_ctx*)> guard(
                ctx, aicore_rfdetr_free);

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        manifest.set("model_info",
                     QJsonDocument::fromJson(
                             RFDetrHelpers::modelInfoJsonFromCtx(ctx).toUtf8())
                             .object());
        manifest.set("threshold", settings.threshold);

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    RFDetrWorker::Settings one = settings;
                    one.inputPath = item.path;
                    RFDetrRunResult result;
                    if (!RFDetrWorker::detectImage(ctx, one, nullptr, &result,
                                                   &err)) {
                        return false;
                    }
                    QJsonArray detections;
                    for (int i = 0; i < result.detections.size(); ++i) {
                        const RFDetrDetection& d = result.detections[i];
                        QJsonObject o{{"class_id", static_cast<int>(d.classId)},
                                      {"class_name", d.className},
                                      {"score", d.score},
                                      { "bbox",
                                        QJsonArray {
                                            d.x1,
                                            d.y1,
                                            d.x2,
                                            d.y2
                                        } }};
                        if (saveMasks && !d.maskRaw.isEmpty()) {
                            // Raw 0/255 bytes at model resolution; wrap
                            // (no copy) as Grayscale8 for PNG encoding.
                            const QImage mask(reinterpret_cast<const uchar*>(
                                                      d.maskRaw.constData()),
                                              d.maskWidth, d.maskHeight,
                                              d.maskWidth,
                                              QImage::Format_Grayscale8);
                            const QString p = B::outPath(
                                    s, item.stem,
                                    QStringLiteral("_mask_%1.png").arg(i));
                            if (!B::saveImage(mask, p, &err)) return false;
                            o["mask"] = p;
                        }
                        detections.append(o);
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_detections.json");
                    if (!B::saveJson(
                                QJsonObject{
                                        {"image", item.path},
                                        {"width",
                                         result.annotatedImage.width()},
                                        {"height",
                                         result.annotatedImage.height()},
                                        {"variant", result.modelVariant},
                                        {"segmentation", result.segmentation},
                                        {"bbox_format", "x1,y1,x2,y2"},
                                        { "detections",
                                          detections }},
                                jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "detections", jsonPath }};
                    if (saveVis) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(result.annotatedImage, p, &err))
                            return false;
                        outputs["visualization"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_rfdetr_last_pipeline_timings(ctx, &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["detections"] = result.detections.size();
                    rec["device"] = result.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_RFDETR_DETECT));
    }
#endif  // AICore_ENABLED
};
