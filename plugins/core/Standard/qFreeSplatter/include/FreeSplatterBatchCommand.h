// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless multiview Gaussian reconstruction (-FREESPLATTER_RECONSTRUCT).

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/gaussian_capi.h>
#include <aicore/model_catalog_capi.h>

#include <QFile>
#include <QJsonArray>
#include <QSaveFile>

#include "FreeSplatterWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_FREESPLATTER_RECONSTRUCT "FREESPLATTER_RECONSTRUCT"

struct CommandFreeSplatterReconstruct : public ccCommandLineInterface::Command {
    CommandFreeSplatterReconstruct()
        : ccCommandLineInterface::Command(
                  QObject::tr("FreeSplatter batch reconstruction"),
                  COMMAND_FREESPLATTER_RECONSTRUCT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MODEL_TYPE", "choice:scene|object|object_2dgs", "scene",
                 "Default catalog model family"},
                {"QUANT", "choice:f16|q8_0|f32", "f16",
                 "Default catalog quantization"},
                {"OPACITY_THRESHOLD", "float", "0.05",
                 "PLY/splat pruning and pose threshold"},
                {"ESTIMATE_POSES", "bool", "1",
                 "Estimate camera-to-world matrices for >=2 views"},
                {"REMOVE_BACKGROUND", "bool", "0",
                 "Apply the plugin GrabCut preprocessing when available"},
                {"MAX_VIEWS", "int", "0",
                 "Uniform view cap (0 = scene 2, object 16, object_2dgs 24)"},
                {"MAX_SPLATS", "int", "0",
                 "Splat importance cap (0 = all passing opacity)"},
                {"SAVE_PLY", "bool", "1", "Write <set>.ply"},
                {"SAVE_SPLAT", "bool", "1", "Write <set>.splat"},
                {"SAVE_RAW", "bool", "0",
                 "Write activated [views,H,W,channels] float32 as .f32"},
        };
        return table;
    }

    static bool saveRaw(const FreeSplatterResult& r,
                        const QString& path,
                        QString* error) {
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly)) {
            *error = QStringLiteral("Cannot write %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        const qint64 bytes = static_cast<qint64>(r.gaussianCount) *
                             static_cast<qint64>(sizeof(float));
        if (file.write(reinterpret_cast<const char*>(r.gaussians.get()),
                       bytes) != bytes ||
            !file.commit()) {
            *error = QStringLiteral("Cannot commit %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        return true;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(
                cmd, COMMAND_FREESPLATTER_RECONSTRUCT,
                QStringLiteral("freesplatter"),
                QStringLiteral("Multiview 3D Gaussian reconstruction"),
                {&options(), &B::setOptions()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir =
                B::takeCString(aicore_gaussian_model_cache_dir(),
                               [](void* p) { aicore_gaussian_free_buffer(p); });
        if (s.listModels) {
            return B::listModels(
                    cmd, s,
                    B::familyRows(AICORE_MODEL_FAMILY_GAUSSIAN, cacheDir));
        }

        const QString type =
                s.args.choice("MODEL_TYPE", QStringLiteral("scene"),
                              {"scene", "object", "object_2dgs"});
        const QString quant = s.args.choice("QUANT", QStringLiteral("f16"),
                                            {"f16", "q8_0", "f32"});
        const float opacity = s.args.toFloat("OPACITY_THRESHOLD", 0.05f);
        const bool estimatePoses = s.args.toBool("ESTIMATE_POSES", true);
        const bool removeBackground = s.args.toBool("REMOVE_BACKGROUND", false);
        const int maxViews = s.args.toInt("MAX_VIEWS", 0);
        const int maxSplats = s.args.toInt("MAX_SPLATS", 0);
        const bool savePly = s.args.toBool("SAVE_PLY", true);
        const bool saveSplat = s.args.toBool("SAVE_SPLAT", true);
        const bool saveRawOutput = s.args.toBool("SAVE_RAW", false);
        if (!s.args.ok()) return cmd.error(s.args.error());
        if (maxSplats < 0)
            return cmd.error(QStringLiteral("MAX_SPLATS must be >= 0"));

        QString modelStem = type;
        modelStem.replace('_', '-');
        const QString defaultModel =
                QStringLiteral("freesplatter-%1-%2.gguf").arg(modelStem, quant);
        const QString modelPath = B::resolveModelPath(s.args.str("MODEL"),
                                                      cacheDir, defaultModel);
        QString error;
        if (!B::ensureModel(cmd, modelPath,
                            B::familySource(AICORE_MODEL_FAMILY_GAUSSIAN,
                                            QFileInfo(modelPath).fileName()),
                            s.download, &error)) {
            return cmd.error(error);
        }
        std::vector<B::SetItem> sets;
        if (!B::collectSets(s.args, &sets, &error)) return cmd.error(error);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        manifest.set("model_type", type);
        manifest.set("opacity_threshold", opacity);
        B::runBatch(
                cmd, s, sets, manifest,
                [&](const B::SetItem& set, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["inputs"] = QJsonArray::fromStringList(set.paths);
                    FreeSplatterWorker::Settings settings;
                    settings.modelPath = modelPath;
                    settings.inputPaths = set.paths;
                    settings.device = s.device;
                    settings.threads = s.threads;
                    settings.opacityThreshold = opacity;
                    settings.estimatePoses = estimatePoses;
                    settings.removeBackground = removeBackground;
                    settings.maxViews = maxViews;
                    FreeSplatterResult result;
                    bool finished = false;
                    bool success = false;
                    QStringList logs;
                    FreeSplatterWorker worker(settings);
                    QObject::connect(
                            &worker, &FreeSplatterWorker::resultReady, &worker,
                            [&](const FreeSplatterResult& r) { result = r; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &FreeSplatterWorker::logMessage, &worker,
                            [&](const QString& m) { logs << m; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &FreeSplatterWorker::taskFinished, &worker,
                            [&](bool ok) {
                                finished = true;
                                success = ok;
                            },
                            Qt::DirectConnection);
                    worker.run();
                    worker.releaseContextOnMainThread();
                    if (!finished || !success || !result.gaussians) {
                        err = logs.isEmpty()
                                      ? QStringLiteral("Inference failed")
                                      : logs.last();
                        return false;
                    }
                    aicore_gaussian_geometry geom{};
                    geom.image_height = result.height;
                    geom.image_width = result.width;
                    geom.gaussian_channels = result.gaussianChannels;
                    geom.sh_degree = result.shDegree;
                    QJsonObject outputs;
                    if (savePly) {
                        const QString p = B::outPath(s, set.stem, ".ply");
                        if (aicore_gaussian_export_ply(
                                    &geom, result.gaussians.get(),
                                    result.nViews, opacity,
                                    QFile::encodeName(p).constData()) != 0) {
                            err = QStringLiteral("PLY export failed: %1")
                                          .arg(p);
                            return false;
                        }
                        outputs["ply"] = p;
                    }
                    if (saveSplat) {
                        const QString p = B::outPath(s, set.stem, ".splat");
                        if (aicore_gaussian_export_splat(
                                    result.gaussians.get(),
                                    result.gaussianCount /
                                            static_cast<size_t>(
                                                    result.gaussianChannels),
                                    result.gaussianChannels, opacity,
                                    static_cast<size_t>(maxSplats),
                                    QFile::encodeName(p).constData()) != 0) {
                            err = QStringLiteral("Splat export failed: %1")
                                          .arg(p);
                            return false;
                        }
                        outputs["splat"] = p;
                    }
                    if (saveRawOutput) {
                        const QString p = B::outPath(s, set.stem, ".f32");
                        if (!saveRaw(result, p, &err)) return false;
                        outputs["raw"] = p;
                    }
                    if (result.hasPoses) {
                        QJsonArray cameras;
                        for (int v = 0; v < result.nViews; ++v) {
                            QJsonArray c2w;
                            for (int k = 0; k < 16; ++k)
                                c2w.append(result.cam2world[v * 16 + k]);
                            cameras.append(QJsonObject{{"index", v},
                                                       { "cam2world",
                                                         c2w }});
                        }
                        const QString p =
                                B::outPath(s, set.stem, "_cameras.json");
                        if (!B::saveJson(QJsonObject{{"focal", result.focal},
                                                     { "cameras",
                                                       cameras }},
                                         p, &err))
                            return false;
                        outputs["cameras"] = p;
                    }
                    rec["views"] = result.nViews;
                    rec["resolution"] = QJsonArray{result.width, result.height};
                    rec["gaussian_channels"] = result.gaussianChannels;
                    rec["gaussian_records"] = static_cast<double>(
                            result.gaussianCount / result.gaussianChannels);
                    rec["device"] = result.resolvedDevice;
                    rec["inference_ms"] = result.runtimeMs;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_FREESPLATTER_RECONSTRUCT));
    }
#endif
};
