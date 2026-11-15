// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless image-to-3D generation (-TRELLIS_GENERATE).

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/trellis_capi.h>

#include <QFile>
#include <QJsonArray>
#include <QSaveFile>
#include <QTextStream>

#include "TrellisModelCatalog.h"
#include "TrellisWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_TRELLIS_GENERATE "TRELLIS_GENERATE"

struct CommandTrellisGenerate : public ccCommandLineInterface::Command {
    CommandTrellisGenerate()
        : ccCommandLineInterface::Command(
                  QObject::tr("TRELLIS batch image-to-3D"),
                  COMMAND_TRELLIS_GENERATE) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"PRESET", "choice:coarse|512|1024", "512",
                 "Model set / pipeline quality"},
                {"QUANT", "choice:q8|f16|f32", "q8", "Weight precision chain"},
                {"MODE", "choice:full|geometry", "full",
                 "Full includes PBR GLB bake; geometry skips GLB bake"},
                {"BACKGROUND", "choice:auto|keep|black|white", "auto",
                 "Solid-background handling"},
                {"USE_RMBG", "bool", "0",
                 "Use the shared RMBG model instead of solid-color removal"},
                {"RMBG_MODEL", "string", "",
                 "Optional RMBG GGUF override (requires USE_RMBG 1)"},
                {"STEPS", "int", "0", "Shape-flow steps (0 = default 12)"},
                {"GUIDANCE", "float", "-1", "CFG guidance (<0 = default 7.5)"},
                {"TEXTURE_STEPS", "int", "0",
                 "Texture-flow steps (0 = default 12)"},
                {"SEED", "int", "0", "Deterministic generation seed"},
                {"SHAPE_DEC_PLACEMENT", "choice:auto|cpu|device", "auto",
                 "Shape decoder placement"},
                {"SAVE_PREVIEW", "bool", "1", "Write <stem>_preview.png"},
                {"SAVE_RMBG", "bool", "1",
                 "Write <stem>_rmbg.png when RMBG ran"},
        };
        return table;
    }

    struct SyncWorker : TrellisWorker {
        using TrellisWorker::TrellisWorker;
        void runHere() { run(); }
    };

    static bool saveObj(const TrellisRunResult& r,
                        const QString& path,
                        QString* error) {
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            *error = QStringLiteral("Cannot write %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        QTextStream out(&file);
        out.setRealNumberNotation(QTextStream::FixedNotation);
        out.setRealNumberPrecision(7);
        for (int i = 0; i + 2 < r.verts.size(); i += 3)
            out << "v " << r.verts[i] << ' ' << r.verts[i + 1] << ' '
                << r.verts[i + 2] << '\n';
        for (int i = 0; i + 2 < r.normals.size(); i += 3)
            out << "vn " << r.normals[i] << ' ' << r.normals[i + 1] << ' '
                << r.normals[i + 2] << '\n';
        const bool normals = r.normals.size() == r.verts.size();
        for (int i = 0; i + 2 < r.tris.size(); i += 3) {
            const int a = r.tris[i] + 1, b = r.tris[i + 1] + 1,
                      c = r.tris[i + 2] + 1;
            if (normals)
                out << "f " << a << "//" << a << ' ' << b << "//" << b << ' '
                    << c << "//" << c << '\n';
            else
                out << "f " << a << ' ' << b << ' ' << c << '\n';
        }
        out.flush();
        if (out.status() != QTextStream::Ok || !file.commit()) {
            *error = QStringLiteral("Cannot commit %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        return true;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_TRELLIS_GENERATE,
                           QStringLiteral("trellis"),
                           QStringLiteral("TRELLIS.2 image-to-3D mesh/PBR GLB"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = TrellisHelpers::modelCacheDir();
        const QVector<TrellisModelEntry> catalog =
                TrellisHelpers::catalogModels();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const TrellisModelEntry& e : catalog)
                rows.push_back(
                        {e.filename, e.displayName, e.downloadUrl,
                         QDir(TrellisHelpers::modelCacheDirFor(e.filename))
                                 .filePath(e.filename),
                         e.role});
            return B::listModels(cmd, s, rows);
        }

        const QString presetToken = s.args.choice(
                "PRESET", QStringLiteral("512"), {"coarse", "512", "1024"});
        const QString quant = s.args.choice("QUANT", QStringLiteral("q8"),
                                            {"q8", "f16", "f32"});
        const QString mode = s.args.choice("MODE", QStringLiteral("full"),
                                           {"full", "geometry"});
        const QString bg = s.args.choice("BACKGROUND", QStringLiteral("auto"),
                                         {"auto", "keep", "black", "white"});
        const bool useRmbg = s.args.toBool("USE_RMBG", false);
        const bool savePreview = s.args.toBool("SAVE_PREVIEW", true);
        const bool saveRmbg = s.args.toBool("SAVE_RMBG", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        const int presetIndex =
                presetToken == QLatin1String("coarse")
                        ? 0
                        : (presetToken == QLatin1String("512") ? 1 : 2);
        const QVector<TrellisPreset> presets = TrellisHelpers::presets();
        if (presetIndex >= presets.size())
            return cmd.error(
                    QStringLiteral("TRELLIS preset catalog incomplete"));
        const TrellisPreset preset = presets[presetIndex];
        const QStringList modelPaths =
                TrellisHelpers::resolvePresetFiles(preset, cacheDir, quant);
        QString error;
        for (const QString& path : modelPaths) {
            if (path.isEmpty()) continue;
            TrellisModelEntry e;
            TrellisHelpers::findModelByFilename(QFileInfo(path).fileName(), &e);
            if (!B::ensureModel(cmd, path, {e.downloadUrl, e.sha256},
                                s.download, &error))
                return cmd.error(error);
        }
        QString rmbgPath;
        if (useRmbg) {
            rmbgPath = B::resolveModelPath(s.args.str("RMBG_MODEL"),
                                           TrellisHelpers::rmbgModelCacheDir(),
                                           QStringLiteral("rmbg_q8.gguf"));
            // The RMBG owner catalog is consumed by qRMBG. Here an absent
            // optional model is left to the worker's documented solid-color
            // fallback unless the caller explicitly supplied RMBG_MODEL.
            if (s.args.has("RMBG_MODEL") && !QFileInfo(rmbgPath).isFile())
                return cmd.error(QStringLiteral("RMBG model not found: %1")
                                         .arg(rmbgPath));
        }
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        const int pipeline = presetIndex == 0   ? AICORE_TRELLIS_PIPE_COARSE
                             : presetIndex == 1 ? AICORE_TRELLIS_PIPE_512
                                                : AICORE_TRELLIS_PIPE_1024;
        const QHash<QString, int> bgModes{{"auto", AICORE_TRELLIS_BG_AUTO},
                                          {"keep", AICORE_TRELLIS_BG_KEEP},
                                          {"black", AICORE_TRELLIS_BG_BLACK},
                                          { "white",
                                            AICORE_TRELLIS_BG_WHITE }};
        B::Manifest manifest(s);
        manifest.set("preset", preset.name);
        manifest.set("quantization", quant);
        manifest.set("models", QJsonArray::fromStringList(modelPaths));
        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    TrellisWorker::Settings settings;
                    settings.modelPaths = modelPaths;
                    settings.inputPath = item.path;
                    settings.presetName = preset.name;
                    settings.pipelineType = pipeline;
                    settings.backgroundMode = bgModes.value(bg);
                    settings.steps = s.args.toInt("STEPS", 0);
                    settings.guidance = s.args.toFloat("GUIDANCE", -1.0f);
                    settings.textureSteps = s.args.toInt("TEXTURE_STEPS", 0);
                    settings.seed = static_cast<uint64_t>(
                            qMax(0, s.args.toInt("SEED", 0)));
                    settings.threads = s.threads;
                    settings.device = s.device;
                    settings.shapeDecPlacement = s.args.choice(
                            "SHAPE_DEC_PLACEMENT", QStringLiteral("auto"),
                            {"auto", "cpu", "device"});
                    settings.useRmbg = useRmbg;
                    settings.rmbgModelPath = rmbgPath;
                    settings.textureEnabled = presetIndex > 0;
                    settings.quantization = quant;
                    settings.livePreview = false;
                    settings.runMode = mode == QLatin1String("full")
                                               ? TrellisRunMode::Full
                                               : TrellisRunMode::GeometryOnly;
                    if (!s.args.ok()) {
                        err = s.args.error();
                        return false;
                    }
                    TrellisRunResult result;
                    bool success = false;
                    QStringList logs;
                    SyncWorker worker(settings);
                    QObject::connect(
                            &worker, &TrellisWorker::resultReady, &worker,
                            [&](const TrellisRunResult& r) { result = r; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &TrellisWorker::logMessage, &worker,
                            [&](const QString& m) { logs << m; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &TrellisWorker::taskFinished, &worker,
                            [&](bool ok) { success = ok; },
                            Qt::DirectConnection);
                    worker.runHere();
                    if (!success || result.verts.isEmpty()) {
                        err = logs.isEmpty()
                                      ? QStringLiteral("Generation failed")
                                      : logs.last();
                        return false;
                    }
                    QJsonObject outputs;
                    const QString obj = B::outPath(s, item.stem, ".obj");
                    if (!saveObj(result, obj, &err)) return false;
                    outputs["mesh"] = obj;
                    if (!result.glb.isEmpty()) {
                        const QString p = B::outPath(s, item.stem, ".glb");
                        if (!B::saveBytes(result.glb, p, &err)) return false;
                        outputs["glb"] = p;
                    }
                    if (savePreview && !result.previewImage.isNull()) {
                        const QString p =
                                B::outPath(s, item.stem, "_preview.png");
                        if (!B::saveImage(result.previewImage, p, &err))
                            return false;
                        outputs["preview"] = p;
                    }
                    if (saveRmbg && !result.rmbgImage.isNull()) {
                        const QString p = B::outPath(s, item.stem, "_rmbg.png");
                        if (!B::saveImage(result.rmbgImage, p, &err))
                            return false;
                        outputs["rmbg"] = p;
                    }
                    QJsonObject stages;
                    for (auto it = result.stageMs.cbegin();
                         it != result.stageMs.cend(); ++it)
                        stages[QString::number(it.key())] = it.value();
                    rec["vertices"] = result.verts.size() / 3;
                    rec["triangles"] = result.tris.size() / 3;
                    rec["has_pbr"] = result.hasPbr;
                    rec["backend"] = result.backend;
                    rec["runtime_ms"] = result.totalRuntimeMs;
                    rec["stage_ms"] = stages;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_TRELLIS_GENERATE));
    }
#endif
};
