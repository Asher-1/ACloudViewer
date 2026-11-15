// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch general keypoint detection (-GKD_KEYPOINTS): text / visual
// (1-shot support image) / multimodal / multi-object (YOLO-World boxes)
// prompting over N query images -> <stem>_coco.json (official-style COCO
// prediction) + <stem>_vis.png each, plus the shared ecvAICoreBatch
// RESULT_JSON manifest. Every image runs the dialog's GKDWorker unchanged
// (synchronously on the calling thread); one GKDContextCache keeps the GKD
// and YOLO-World contexts resident so the models load once per batch.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/gkd_capi.h>
#include <aicore/yolo_capi.h>

#include "GKDModelCatalog.h"
#include "GKDWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_GKD_KEYPOINTS "GKD_KEYPOINTS"

struct CommandGkdKeypoints : public ccCommandLineInterface::Command {
    CommandGkdKeypoints()
        : ccCommandLineInterface::Command(
                  QObject::tr("GKD batch keypoint detection"),
                  COMMAND_GKD_KEYPOINTS) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"PROMPTS", "list", "",
                 "Keypoint text prompts, comma-separated (\"nose, left "
                 "eye\"); quoted segments keep commas"},
                {"SKELETON", "string", "",
                 "1-based bone pairs \"1-2 1-3 ...\" (rendering + COCO)"},
                {"SUPPORT_IMAGE", "path", "",
                 "1-shot visual prompt image (visual / multimodal)"},
                {"SUPPORT_KPS", "string", "",
                 "Support keypoints \"x1,y1 x2,y2 ...\" in support pixels"},
                {"ROI", "string", "",
                 "Query ROI(s) \"x1,y1 x2,y2 [...]\" applied to every image"},
                {"MULTI_OBJECT", "bool", "0",
                 "Detect CLASSES with YOLO-World first, then GKD per box"},
                {"CLASSES", "list", "", "Multi-object class names"},
                {"YOLO_MODEL", "string", "",
                 "YOLO-World GGUF (default: catalog default)"},
                {"YOLO_TEXT_MODEL", "string", "",
                 "Class-name text tower (default: catalog default)"},
                {"YOLO_CONF", "float", "0.25", "YOLO-World confidence cut"},
                {"MIN_SCORE", "float", "0.10", "Drop keypoints below score"},
                {"POINT_LABELS", "bool", "0", "Label points in the overlay"},
                {"SAVE_VIS", "bool", "1", "Write <stem>_vis.png"},
        };
        return table;
    }

    /** Runs the dialog worker body on the calling thread (no event loop,
     *  no second thread: the CLI has no GUI to keep responsive). */
    struct SyncWorker : public GKDWorker {
        using GKDWorker::GKDWorker;
        void runHere() { run(); }
    };

    static QString defaultOf(const QVector<GKDModelEntry>& rows, int index) {
        return index >= 0 && index < rows.size() ? rows[index].filename
                                                 : QString();
    }

    static QString urlOf(const QVector<GKDModelEntry>& rows,
                         const QString& path) {
        const QString fn = QFileInfo(path).fileName();
        for (const GKDModelEntry& e : rows) {
            if (e.filename == fn) return e.downloadUrl;
        }
        return QString();
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_GKD_KEYPOINTS, QStringLiteral("gkd"),
                           QStringLiteral("Open-vocabulary keypoint "
                                          "detection (GKD)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = GKDHelpers::modelCacheDir();
        const QString yoloDir = B::takeCString(aicore_yolo_model_cache_dir(),
                                               &aicore_yolo_free_buffer);
        const QVector<GKDModelEntry> gkdRows = GKDHelpers::catalogModels();
        const QVector<GKDModelEntry> worldRows = GKDHelpers::yoloWorldModels();
        const QVector<GKDModelEntry> textRows = GKDHelpers::yoloTextModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            auto add = [&](const QVector<GKDModelEntry>& v, const QString& dir,
                           const char* role) {
                for (const GKDModelEntry& e : v) {
                    rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                    QDir(dir).filePath(e.filename),
                                    QLatin1String(role)});
                }
            };
            add(gkdRows, cacheDir, "gkd");
            add(worldRows, yoloDir, "yolo_world");
            add(textRows, yoloDir, "yolo_text");
            return B::listModels(cmd, s, rows);
        }

        GKDWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.kpsTexts =
                GKDHelpers::splitPrompts(s.args.list("PROMPTS").join(','));
        settings.skeleton = s.args.str("SKELETON");
        settings.supportImagePath = s.args.str("SUPPORT_IMAGE");
        if (!GKDHelpers::parseCoordinatePairs(s.args.str("SUPPORT_KPS"),
                                              &settings.supportKps)) {
            return cmd.error(
                    QStringLiteral("SUPPORT_KPS: expected "
                                   "\"x1,y1 x2,y2 ...\""));
        }
        QVector<QPointF> roi;
        if (!GKDHelpers::parseCoordinatePairs(
                    s.args.str("ROI").replace(QLatin1Char(','),
                                              QLatin1Char(' ')),
                    &roi) ||
            roi.size() % 2 != 0) {
            return cmd.error(
                    QStringLiteral("ROI: expected box corner pairs "
                                   "\"x1,y1 x2,y2 ...\""));
        }
        // Same ROI rules as GKDDialog::settings(): 2 points = one box,
        // 4+ = batched multi-ROI.
        settings.roiPoints = roi;
        if (roi.size() == 2) {
            settings.hasBbox = true;
            settings.bbox[0] = static_cast<float>(roi[0].x());
            settings.bbox[1] = static_cast<float>(roi[0].y());
            settings.bbox[2] = static_cast<float>(roi[1].x());
            settings.bbox[3] = static_cast<float>(roi[1].y());
        }
        settings.multiObject = s.args.toBool("MULTI_OBJECT", false);
        settings.objectClasses = s.args.tokens("CLASSES");
        settings.yoloConf = s.args.toFloat("YOLO_CONF", 0.25f);
        settings.minScore = s.args.toFloat("MIN_SCORE", 0.10f);
        settings.pointLabels = s.args.toBool("POINT_LABELS", false);
        settings.addResultToDb = false;
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        if (!s.args.ok()) return cmd.error(s.args.error());
        if (settings.kpsTexts.isEmpty() && settings.supportImagePath.isEmpty())
            return cmd.error(QStringLiteral(
                    "Give PROMPTS and/or SUPPORT_IMAGE + SUPPORT_KPS"));
        if (settings.multiObject && settings.objectClasses.isEmpty())
            return cmd.error(QStringLiteral("MULTI_OBJECT 1 needs CLASSES"));

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        settings.modelPath = B::resolveModelPath(
                s.args.str("MODEL"), cacheDir,
                defaultOf(gkdRows, GKDHelpers::catalogDefaultIndex()));
        if (!B::ensureModel(cmd, settings.modelPath,
                            {urlOf(gkdRows, settings.modelPath), QString()},
                            s.download, &error)) {
            return cmd.error(error);
        }
        if (settings.multiObject) {
            settings.yoloModelPath = B::resolveModelPath(
                    s.args.str("YOLO_MODEL"), yoloDir,
                    defaultOf(worldRows, GKDHelpers::yoloWorldDefaultIndex()));
            settings.yoloTextModelPath = B::resolveModelPath(
                    s.args.str("YOLO_TEXT_MODEL"), yoloDir,
                    defaultOf(textRows, GKDHelpers::yoloTextDefaultIndex()));
            for (const QString& p :
                 {settings.yoloModelPath, settings.yoloTextModelPath}) {
                const QString url = urlOf(worldRows, p).isEmpty()
                                            ? urlOf(textRows, p)
                                            : urlOf(worldRows, p);
                if (!B::ensureModel(cmd, p, {url, QString()}, s.download,
                                    &error)) {
                    return cmd.error(error);
                }
            }
        }

        GKDContextCache cache;
        settings.cache = &cache;
        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        if (settings.multiObject) {
            manifest.set("yolo_model", settings.yoloModelPath);
            manifest.set("yolo_text_model", settings.yoloTextModelPath);
            manifest.set("classes",
                         QJsonArray::fromStringList(settings.objectClasses));
        }
        manifest.set("prompts", QJsonArray::fromStringList(settings.kpsTexts));

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    GKDWorker::Settings one = settings;
                    one.inputPath = item.path;
                    SyncWorker worker(one);
                    GKDRunResult result;
                    bool got = false;
                    QString lastLog;
                    QObject::connect(&worker, &GKDWorker::resultReady,
                                     [&](const GKDRunResult& r) {
                                         result = r;
                                         got = true;
                                     });
                    QObject::connect(&worker, &GKDWorker::logMessage,
                                     [&](const QString& m) { lastLog = m; });
                    worker.runHere();
                    if (!got) {
                        err = lastLog.isEmpty() ? QStringLiteral("GKD failed")
                                                : lastLog;
                        return false;
                    }
                    const QString coco = B::outPath(s, item.stem, "_coco.json");
                    if (!B::saveBytes(
                                GKDHelpers::buildCocoJson(result).toUtf8(),
                                coco, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{ "coco", coco }};
                    if (saveVis && !result.renderedImage.isNull()) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(result.renderedImage, p, &err))
                            return false;
                        outputs["visualization"] = p;
                    }
                    rec["mode"] = result.mode;
                    rec["objects"] = result.sets.size();
                    rec["keypoints"] = result.totalKeypoints;
                    rec["shown_keypoints"] = result.shownKeypoints;
                    rec["elapsed_inference_ms"] = result.totalRuntimeMs;
                    rec["device"] = result.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });

        // Cache-owned handles (the plugin frees these in its destructor).
        if (cache.gkdCtx)
            aicore_gkd_free(static_cast<aicore_gkd_ctx*>(cache.gkdCtx));
        if (cache.yoloCtx)
            aicore_yolo_free(static_cast<aicore_yolo_ctx*>(cache.yoloCtx));
        for (void* h : cache.retiredGkd)
            aicore_gkd_free(static_cast<aicore_gkd_ctx*>(h));
        for (void* h : cache.retiredYolo)
            aicore_yolo_free(static_cast<aicore_yolo_ctx*>(h));
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_GKD_KEYPOINTS));
    }
#endif  // AICore_ENABLED
};
