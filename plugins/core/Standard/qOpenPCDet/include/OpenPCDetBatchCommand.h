// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch LiDAR 3D object detection (-OPENPCDET_DETECT): one
// OpenPCDet model load, N point-cloud files -> <stem>_detections.json each,
// plus the shared ecvAICoreBatch RESULT_JSON manifest. The per-item body is
// OpenPCDetWorker::detectBuffer, the exact code path of the dialog; raw
// inputs reuse OpenPCDetHelpers::parsePointsBin / parsePointsTxt so CLI and
// GUI see identical point buffers.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/openpcdet_capi.h>

#include <QFile>

#include "OpenPCDetModelCatalog.h"
#include "OpenPCDetWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_OPENPCDET_DETECT "OPENPCDET_DETECT"

struct CommandOpenPCDetDetect : public ccCommandLineInterface::Command {
    CommandOpenPCDetDetect()
        : ccCommandLineInterface::Command(
                  QObject::tr("OpenPCDet batch 3D detection"),
                  COMMAND_OPENPCDET_DETECT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"SCORE_THRESH", "float", "0",
                 "Score threshold override (0 = keep GGUF value)"},
                {"NMS_THRESH", "float", "0",
                 "Rotated NMS IoU override (0 = keep GGUF value)"},
                {"MAX_BOXES", "int", "0",
                 "Maximum returned boxes (0 = keep GGUF value)"},
        };
        return table;
    }

    /** Parses one raw point file (.bin KITTI float32 or whitespace-separated
     *  ASCII) into a row-major [n, pointDim] buffer. */
    static bool loadPoints(const QString& path,
                           int pointDim,
                           std::vector<float>* outPoints,
                           int* outCount,
                           QString* error) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            *error = QStringLiteral("Cannot open point file: %1").arg(path);
            return false;
        }
        const QString suffix =
                QFileInfo(path).suffix().toLower();
        if (suffix == QLatin1String("bin")) {
            return OpenPCDetHelpers::parsePointsBin(file.readAll(), pointDim,
                                                    outPoints, outCount,
                                                    error);
        }
        return OpenPCDetHelpers::parsePointsTxt(
                QString::fromUtf8(file.readAll()), pointDim, outPoints,
                outCount, error);
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_OPENPCDET_DETECT,
                           QStringLiteral("openpcdet"),
                           QStringLiteral("LiDAR point cloud 3D object "
                                          "detection (PointPillars / "
                                          "SECOND, GGML)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = OpenPCDetHelpers::modelCacheDir();
        const QVector<OpenPCDetModelEntry> catalog =
                OpenPCDetHelpers::catalogModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const OpenPCDetModelEntry& e : catalog) {
                rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                QDir(cacheDir).filePath(e.filename),
                                e.familyNote});
            }
            return B::listModels(cmd, s, rows);
        }

        OpenPCDetWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.scoreThresh = s.args.toFloat("SCORE_THRESH", 0.0f);
        settings.nmsThresh = s.args.toFloat("NMS_THRESH", 0.0f);
        settings.maxBoxes = s.args.toInt("MAX_BOXES", 0);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(
                    s.args, &items, &error,
                    {"*.bin", "*.pts", "*.xyz", "*.txt"})) {
            return cmd.error(error);
        }

        const int defIdx = OpenPCDetHelpers::catalogDefaultIndex();
        settings.modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir,
                                    defIdx >= 0 && defIdx < catalog.size()
                                            ? catalog[defIdx].filename
                                            : QString());
        OpenPCDetModelEntry entry;
        OpenPCDetHelpers::findModelByFilename(
                QFileInfo(settings.modelPath).fileName(), &entry);
        if (!B::ensureModel(cmd, settings.modelPath,
                            {entry.downloadUrl, QString()}, s.download,
                            &error)) {
            return cmd.error(error);
        }

        aicore_openpcdet_options* opts = aicore_openpcdet_options_new();
        aicore_openpcdet_options_set_device(opts,
                                            s.device.toUtf8().constData());
        aicore_openpcdet_options_set_threads(opts, s.threads);
        if (settings.scoreThresh > 0.0f) {
            aicore_openpcdet_options_set_score_thresh(
                    opts, settings.scoreThresh);
        }
        if (settings.nmsThresh > 0.0f) {
            aicore_openpcdet_options_set_nms_thresh(opts, settings.nmsThresh);
        }
        if (settings.maxBoxes > 0) {
            aicore_openpcdet_options_set_max_boxes(opts, settings.maxBoxes);
        }
        aicore_openpcdet_ctx* ctx = aicore_openpcdet_load_opts(
                settings.modelPath.toUtf8().constData(), opts);
        aicore_openpcdet_options_free(opts);
        if (ctx == nullptr || !aicore_openpcdet_is_ready(ctx)) {
            const char* msg = ctx ? aicore_openpcdet_last_error(ctx) : nullptr;
            if (ctx) aicore_openpcdet_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg && msg[0]
                                                  ? QString::fromUtf8(msg)
                                                  : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_openpcdet_ctx, void (*)(aicore_openpcdet_ctx*)>
                guard(ctx, aicore_openpcdet_free);

        aicore_openpcdet_geometry geometry{};
        aicore_openpcdet_geometry_of(ctx, &geometry);
        const int pointDim = geometry.num_point_features;
        if (pointDim <= 0) {
            return cmd.error(QStringLiteral(
                    "Model geometry declares no point features"));
        }

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        if (char* info = aicore_openpcdet_info_json(ctx)) {
            manifest.set("model_info",
                         QJsonDocument::fromJson(QByteArray(info)).object());
            aicore_openpcdet_free_buffer(info);
        }
        manifest.set("point_dim", pointDim);
        manifest.set("arch", QString::fromLatin1(geometry.arch));

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    std::vector<float> points;
                    int count = 0;
                    if (!loadPoints(item.path, pointDim, &points, &count,
                                    &err)) {
                        return false;
                    }
                    OpenPCDetRunResult result;
                    if (!OpenPCDetWorker::detectBuffer(ctx, points.data(),
                                                       count, pointDim,
                                                       &result, &err)) {
                        return false;
                    }
                    result.modelPath = settings.modelPath;
                    QJsonArray detections;
                    for (const OpenPCDetDetection& d : result.detections) {
                        detections.append(
                                QJsonObject{
                                        {"class_id", d.label},
                                        {"class_name", d.className},
                                        {"score", d.score},
                                        {"center",
                                         QJsonArray{d.x, d.y, d.z}},
                                        {"size",
                                         QJsonArray{d.dx, d.dy, d.dz}},
                                        {"heading", d.heading}});
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem,
                                       QStringLiteral("_detections.json"));
                    if (!B::saveJson(
                                QJsonObject{
                                        {"input", item.path},
                                        {"model", result.modelPath},
                                        {"num_points", result.numPoints},
                                        {"point_dim", result.pointDim},
                                        {"bbox_format",
                                         "center(x,y,z),size(dx,dy,dz),"
                                         "heading"},
                                        {"detections", detections}},
                                jsonPath, &err)) {
                        return false;
                    }
                    QJsonObject outputs{{"detections", jsonPath}};
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_openpcdet_last_pipeline_timings(ctx, &t) ==
                        0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["points"] = result.numPoints;
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
                                 .arg(COMMAND_OPENPCDET_DETECT));
    }
#endif  // AICore_ENABLED
};
