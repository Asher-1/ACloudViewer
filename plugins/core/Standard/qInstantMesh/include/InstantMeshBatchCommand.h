// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless image-to-3D generation (-INSTANTMESH_GENERATE).

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/instantmesh_capi.h>
#include <aicore/rmbg_capi.h>

#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QSaveFile>
#include <QTextStream>
#include <cmath>

#include "InstantMeshModelCatalog.h"
#include "InstantMeshWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_INSTANTMESH_GENERATE "INSTANTMESH_GENERATE"

struct CommandInstantMeshGenerate : public ccCommandLineInterface::Command {
    CommandInstantMeshGenerate()
        : ccCommandLineInterface::Command(
                  QObject::tr("InstantMesh batch image-to-3D"),
                  COMMAND_INSTANTMESH_GENERATE) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"PRESET", "choice:full|geometry", "full",
                 "Model set: full (Zero123++ 6-view diffusion) or geometry "
                 "(replicate the input as the 6 views)"},
                {"QUANT", "choice:f16|q8|f32", "f16",
                 "Weight precision chain (Zero123++ stays f16 on q8)"},
                {"USE_RMBG", "bool", "1",
                 "Use the shared RMBG model before the diffusion input"},
                {"RMBG_MODEL", "string", "",
                 "Optional RMBG GGUF override (requires USE_RMBG 1)"},
                {"STEPS", "int", "0", "Diffusion steps (0 = default 75)"},
                {"GUIDANCE", "float", "-1", "CFG guidance (<0 = default 4.0)"},
                {"SEED", "int", "42", "Deterministic diffusion seed"},
                {"GRID_RES", "int", "0",
                 "FlexiCubes cubes per axis (0 = default 64)"},
                {"GRID_SCALE", "float", "-1", "Grid scale (<=0 = default 2.1)"},
                {"TEXTURE", "bool", "1", "Bake the UV texture atlas"},
                {"TEXTURE_RES", "int", "0",
                 "Atlas resolution (0 = default 2048)"},
        };
        return table;
    }

    struct SyncWorker : InstantMeshWorker {
        using InstantMeshWorker::InstantMeshWorker;
        void runHere() { run(); }
    };

    static bool saveObj(const InstantMeshRunResult& r,
                        const QString& path,
                        QString* error) {
        if (r.verts.isEmpty() || r.verts.size() % 3 || r.tris.isEmpty() ||
            r.tris.size() % 3) {
            *error = QStringLiteral("Invalid OBJ mesh layout");
            return false;
        }
        for (int index : r.tris) {
            if (index < 0 || index >= r.verts.size() / 3) {
                *error = QStringLiteral("Invalid OBJ triangle index");
                return false;
            }
        }
        for (float value : r.verts) {
            if (!std::isfinite(value)) {
                *error = QStringLiteral("Non-finite OBJ position");
                return false;
            }
        }
        const bool textured = !r.texRgba.isEmpty() || !r.uvs.isEmpty();
        const QString base = QFileInfo(path).completeBaseName();
        const QString mtlName = base + QStringLiteral(".mtl");
        if (textured) {
            if (r.texW <= 0 || r.texH <= 0 ||
                uint64_t(r.texW) * r.texH * 4 != uint64_t(r.texRgba.size()) ||
                uint64_t(r.uvs.size()) != uint64_t(r.tris.size()) * 2) {
                *error = QStringLiteral("Invalid OBJ texture layout");
                return false;
            }
            for (float value : r.uvs) {
                if (!std::isfinite(value)) {
                    *error = QStringLiteral("Non-finite OBJ UV");
                    return false;
                }
            }
            const QString textureName = base + QStringLiteral("_texture.png");
            const QDir directory = QFileInfo(path).absoluteDir();
            const QImage image(r.texRgba.constData(), r.texW, r.texH,
                               r.texW * 4, QImage::Format_RGBA8888);
            if (!ecvAICoreBatch::saveImage(
                        image, directory.filePath(textureName), error) ||
                !ecvAICoreBatch::saveBytes(
                        QByteArray(
                                "newmtl InstantMesh_atlas\nKd 1 1 1\nmap_Kd ") +
                                textureName.toUtf8() + '\n',
                        directory.filePath(mtlName), error))
                return false;
        }
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            *error = QStringLiteral("Cannot write %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        QTextStream out(&file);
        out.setRealNumberNotation(QTextStream::FixedNotation);
        out.setRealNumberPrecision(7);
        if (textured) out << "mtllib " << mtlName << '\n';
        const bool colored = r.vcolors.size() == r.verts.size();
        for (int i = 0; i + 2 < r.verts.size(); i += 3) {
            out << "v " << r.verts[i] << ' ' << r.verts[i + 1] << ' '
                << r.verts[i + 2];
            if (colored) {
                out << ' ' << r.vcolors[i] << ' ' << r.vcolors[i + 1] << ' '
                    << r.vcolors[i + 2];
            }
            out << '\n';
        }
        if (textured) {
            for (int i = 0; i < r.uvs.size(); i += 2)
                out << "vt " << r.uvs[i] << ' ' << r.uvs[i + 1] << '\n';
            out << "usemtl InstantMesh_atlas\n";
        }
        for (int i = 0; i < r.tris.size(); i += 3) {
            out << "f";
            for (int k = 0; k < 3; ++k) {
                out << ' ' << r.tris[i + k] + 1;
                if (textured) out << '/' << i + k + 1;
            }
            out << '\n';
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
        InstantMeshWorker::Settings baseSettings;
        QString presetToken, quant;
        bool useRmbg = false, bakeTexture = false;
        const auto validateArgs = [&](B::Args& args) {
            presetToken = args.choice("PRESET", QStringLiteral("full"),
                                      {"full", "geometry"});
            quant = args.choice("QUANT", QStringLiteral("f16"),
                                {"f16", "q8", "f32"});
            useRmbg = args.toBool("USE_RMBG", true);
            bakeTexture = args.toBool("TEXTURE", true);
            baseSettings.steps = args.toInt("STEPS", 0);
            baseSettings.guidance = args.toFloat("GUIDANCE", -1.0f);
            baseSettings.gridRes = args.toInt("GRID_RES", 0);
            baseSettings.gridScale = args.toFloat("GRID_SCALE", -1.0f);
            baseSettings.textureRes = args.toInt("TEXTURE_RES", 0);
            bool seedOk = false;
            const QString seedToken =
                    args.str("SEED", QStringLiteral("42")).trimmed();
            baseSettings.seed = seedToken.toULongLong(&seedOk);
            if (!seedOk || seedToken.startsWith('-'))
                args.fail(QStringLiteral(
                        "SEED expects an unsigned 64-bit integer"));
            if (baseSettings.steps != 0 &&
                (baseSettings.steps < 2 || baseSettings.steps > 1000))
                args.fail(QStringLiteral("STEPS must be 0 or [2,1000]"));
            if (!std::isfinite(baseSettings.guidance) ||
                baseSettings.guidance > 15)
                args.fail(QStringLiteral("GUIDANCE must be finite and <=15"));
            if (baseSettings.gridRes != 0 &&
                (baseSettings.gridRes < 2 || baseSettings.gridRes > 256))
                args.fail(QStringLiteral("GRID_RES must be 0 or [2,256]"));
            if (!std::isfinite(baseSettings.gridScale))
                args.fail(QStringLiteral("GRID_SCALE must be finite"));
            if (baseSettings.textureRes != 0 &&
                (baseSettings.textureRes < 1 || baseSettings.textureRes > 4096))
                args.fail(QStringLiteral("TEXTURE_RES must be 0 or [1,4096]"));
        };
        B::Session s;
        switch (B::prepare(cmd, COMMAND_INSTANTMESH_GENERATE,
                           QStringLiteral("instantmesh"),
                           QStringLiteral("InstantMesh image-to-3D textured "
                                          "mesh/GLB"),
                           {&options()}, &s, validateArgs)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = InstantMeshHelpers::modelCacheDir();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (const InstantMeshModelEntry& e :
                 InstantMeshHelpers::catalogModels())
                rows.push_back({e.filename, e.displayName, e.downloadUrl,
                                QDir(cacheDir).filePath(e.filename), e.role});
            return B::listModels(cmd, s, rows);
        }

        baseSettings.bakeTexture = bakeTexture;
        baseSettings.threads = s.threads;
        baseSettings.device = s.device;
        baseSettings.useRmbg = useRmbg;
        baseSettings.quantization = quant;

        const int presetIndex = presetToken == QLatin1String("full") ? 0 : 1;
        const QVector<InstantMeshPreset> presets =
                InstantMeshHelpers::presets();
        if (presetIndex >= presets.size())
            return cmd.error(
                    QStringLiteral("InstantMesh preset catalog incomplete"));
        const InstantMeshPreset preset = presets[presetIndex];
        const QStringList modelPaths =
                InstantMeshHelpers::resolvePresetFiles(preset, cacheDir, quant);
        QString error;
        for (const QString& path : modelPaths) {
            if (path.isEmpty()) continue;
            InstantMeshModelEntry e;
            InstantMeshHelpers::findModelByFilename(QFileInfo(path).fileName(),
                                                    &e);
            if (!B::ensureModel(cmd, path, {e.downloadUrl, e.sha256},
                                s.download, &error))
                return cmd.error(error);
        }
        QString rmbgPath;
        if (useRmbg) {
            const auto* recommended =
                    aicore_rmbg_model_at(aicore_rmbg_model_default_index());
            if (!recommended || !recommended->filename)
                return cmd.error(
                        QStringLiteral("Shared RMBG catalog incomplete"));
            rmbgPath = B::resolveModelPath(
                    s.args.str("RMBG_MODEL"),
                    InstantMeshHelpers::rmbgModelCacheDir(),
                    QString::fromUtf8(recommended->filename));
            const QByteArray name = QFileInfo(rmbgPath).fileName().toUtf8();
            const auto* entry = aicore_rmbg_model_by_filename(name.constData());
            if (!entry)
                return cmd.error(QStringLiteral("Unknown shared RMBG model: %1")
                                         .arg(rmbgPath));
            if (!B::ensureModel(
                        cmd, rmbgPath,
                        {QString::fromUtf8(entry->download_url),
                         QString::fromLatin1(ecvAssetIntegrity::PinnedDigest(
                                 QString::fromUtf8(name)))},
                        s.download, &error))
                return cmd.error(error);
        }
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        B::Manifest manifest(s);
        manifest.set("preset", preset.name);
        manifest.set("quantization", quant);
        manifest.set("models", QJsonArray::fromStringList(modelPaths));
        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    InstantMeshWorker::Settings settings = baseSettings;
                    settings.modelPaths = modelPaths;
                    settings.inputPath = item.path;
                    settings.presetName = preset.name;
                    settings.rmbgModelPath = rmbgPath;
                    InstantMeshRunResult result;
                    bool success = false;
                    QStringList logs;
                    SyncWorker worker(settings);
                    QObject::connect(
                            &worker, &InstantMeshWorker::resultReady, &worker,
                            [&](const InstantMeshRunResult& r) { result = r; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &InstantMeshWorker::logMessage, &worker,
                            [&](const QString& m) { logs << m; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &InstantMeshWorker::taskFinished, &worker,
                            [&](bool ok) { success = ok; },
                            Qt::DirectConnection);
                    worker.runHere();
                    if (!success || result.verts.isEmpty()) {
                        err = logs.isEmpty()
                                      ? QStringLiteral("Generation failed")
                                      : logs.last();
                        return false;
                    }
                    QElapsedTimer fileExportTimer;
                    fileExportTimer.start();
                    QJsonObject outputs;
                    const QString obj = B::outPath(s, item.stem, ".obj");
                    if (!saveObj(result, obj, &err)) return false;
                    outputs["mesh"] = obj;
                    if (!result.texRgba.isEmpty()) {
                        const QFileInfo meshPath(obj);
                        outputs["material"] = meshPath.absoluteDir().filePath(
                                meshPath.completeBaseName() +
                                QStringLiteral(".mtl"));
                        outputs["texture"] = meshPath.absoluteDir().filePath(
                                meshPath.completeBaseName() +
                                QStringLiteral("_texture.png"));
                    }
                    if (!result.glb.isEmpty()) {
                        const QString p = B::outPath(s, item.stem, ".glb");
                        if (!B::saveBytes(result.glb, p, &err)) return false;
                        outputs["glb"] = p;
                    }
                    if (!result.previewImage.isNull()) {
                        const QString p =
                                B::outPath(s, item.stem, "_preview.png");
                        if (!B::saveImage(result.previewImage, p, &err))
                            return false;
                        outputs["preview"] = p;
                    }
                    const double fileExportMs =
                            fileExportTimer.nsecsElapsed() / 1e6;
                    QJsonObject stages;
                    for (auto it = result.stageMs.cbegin();
                         it != result.stageMs.cend(); ++it)
                        stages[QString::number(it.key())] = it.value();
                    rec["vertices"] = result.verts.size() / 3;
                    rec["triangles"] = result.tris.size() / 3;
                    rec["has_texture"] = !result.texRgba.isEmpty();
                    rec["multiview"] = result.multiview;
                    rec["backend"] = result.backend;
                    rec["runtime_ms"] = result.totalRuntimeMs;
                    rec["timings"] = B::timingsJson(result.timings);
                    rec["load_ms"] = result.loadRuntimeMs;
                    rec["glb_bake_ms"] = result.exportRuntimeMs;
                    rec["file_export_ms"] = fileExportMs;
                    rec["export_ms"] = result.exportRuntimeMs + fileExportMs;
                    rec["stage_ms"] = stages;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_INSTANTMESH_GENERATE));
    }
#endif
};
