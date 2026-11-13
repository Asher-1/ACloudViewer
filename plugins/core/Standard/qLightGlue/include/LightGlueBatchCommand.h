// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch feature matching (-LIGHTGLUE_MATCH): one LightGlue matcher,
// N image pairs -> <a>_x_<b>_matches.json (+ side-by-side match preview)
// each, plus the shared ecvAICoreBatch RESULT_JSON manifest. Pairs come from
// PAIRS_FILE (COLMAP-style "a b" lines) or from the collected images through
// PAIR_MODE. Per-image features are extracted once and reused by every pair
// the image takes part in. The per-step bodies are the LightGlueWorker
// statics and renderMatchVisualization, the dialog code paths.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/lightglue_capi.h>
#include <aicore/model_catalog_capi.h>

#include <QFile>
#include <QHash>
#include <QRegularExpression>
#include <QTextStream>
#include <memory>

#include "LightGlueWorker.h"
#include "ecvAICoreBatchCommand.h"
#include "feature_extractor.h"
#include "match_visualization.h"
#endif  // AICore_ENABLED

#define COMMAND_LIGHTGLUE_MATCH "LIGHTGLUE_MATCH"

struct CommandLightGlueMatch : public ccCommandLineInterface::Command {
    CommandLightGlueMatch()
        : ccCommandLineInterface::Command(
                  QObject::tr("LightGlue batch feature matching"),
                  COMMAND_LIGHTGLUE_MATCH) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MATCHER", "choice:sift|aliked", "sift",
                 "Feature type selecting the default catalog matcher (an "
                 "explicit MODEL decides by its catalog row / file name)"},
                {"QUANT", "choice:f16|q8_0|f32", "f16",
                 "Catalog quantization of matcher (+ ALIKED extractor)"},
                {"PAIRS_FILE", "string", "",
                 "Text file, one pair per line: '<a> <b>' (tab-separated "
                 "when names contain spaces; relative to INPUT_DIR or the "
                 "file's folder; '#' comments)"},
                {"PAIR_MODE", "choice:sequential|exhaustive|reference",
                 "sequential",
                 "Pairs from the collected images: i with the next WINDOW "
                 "images / every pair / the first image with every other"},
                {"WINDOW", "int", "1", "sequential: neighbours per image"},
                {"MIN_SCORE", "float", "0.5", "Match confidence threshold"},
                {"MAX_KEYPOINTS", "int", "2048", "Keypoints per image"},
                {"MAX_RESIZE", "int", "640",
                 "Longest image edge for extraction (0 = native)"},
                {"SAVE_VIS", "bool", "1",
                 "Write <a>_x_<b>_vis.png (side-by-side matches)"},
        };
        return table;
    }

    struct Pair {
        QString path0;
        QString path1;
        QString stem;
    };

    static bool readPairsFile(const QString& file,
                              const QString& baseDir,
                              QList<QPair<QString, QString>>* out,
                              QString* error) {
        QFile f(file);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            *error = QStringLiteral("Cannot read PAIRS_FILE %1").arg(file);
            return false;
        }
        const QDir base(baseDir.isEmpty() ? QFileInfo(file).absolutePath()
                                          : baseDir);
        QTextStream in(&f);
        int lineNo = 0;
        while (!in.atEnd()) {
            ++lineNo;
            const QString line = in.readLine().trimmed();
            if (line.isEmpty() || line.startsWith('#')) continue;
            const QStringList parts =
                    line.contains('\t')
                            ? line.split('\t', QtCompat::SkipEmptyParts)
                            : line.split(QRegularExpression(
                                                 QStringLiteral("\\s+")),
                                         QtCompat::SkipEmptyParts);
            if (parts.size() < 2) {
                *error = QStringLiteral(
                                 "PAIRS_FILE line %1: expected two "
                                 "image names")
                                 .arg(lineNo);
                return false;
            }
            out->append(
                    {QFileInfo(base, parts[0].trimmed()).absoluteFilePath(),
                     QFileInfo(base, parts[1].trimmed()).absoluteFilePath()});
        }
        return true;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_LIGHTGLUE_MATCH,
                           QStringLiteral("lightglue"),
                           QStringLiteral("Image-pair feature matching "
                                          "(SIFT/ALIKED + LightGlue)"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir =
                B::takeCString(aicore_lightglue_model_cache_dir(), [](void* p) {
                    aicore_lightglue_free_buffer(static_cast<char*>(p));
                });
        if (s.listModels) {
            // Matchers and the ALIKED extractors share one cache folder.
            std::vector<B::CatalogRow> rows =
                    B::familyRows(AICORE_MODEL_FAMILY_LIGHTGLUE, cacheDir);
            for (B::CatalogRow& row : rows)
                row.role = QStringLiteral("matcher");
            for (B::CatalogRow row :
                 B::familyRows(AICORE_MODEL_FAMILY_ALIKED, cacheDir)) {
                row.role = QStringLiteral("extractor");
                rows.push_back(row);
            }
            return B::listModels(cmd, s, rows);
        }

        const QString matcher = s.args.choice("MATCHER", QStringLiteral("sift"),
                                              {"sift", "aliked"});
        const QString quant = s.args.choice("QUANT", QStringLiteral("f16"),
                                            {"f16", "q8_0", "f32"});
        const QString pairMode =
                s.args.choice("PAIR_MODE", QStringLiteral("sequential"),
                              {"sequential", "exhaustive", "reference"});
        const int window = qMax(1, s.args.toInt("WINDOW", 1));
        LightGlueWorker::Settings settings;
        settings.minScore = s.args.toFloat("MIN_SCORE", 0.5f);
        settings.maxKeypoints = s.args.toInt("MAX_KEYPOINTS", 2048);
        settings.maxResize = s.args.toInt("MAX_RESIZE", 640);
        settings.threads = s.threads;
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        // ---- pairs ------------------------------------------------------
        QList<QPair<QString, QString>> rawPairs;
        if (s.args.has("PAIRS_FILE")) {
            if (!readPairsFile(s.args.str("PAIRS_FILE"),
                               s.args.str("INPUT_DIR"), &rawPairs, &error)) {
                return cmd.error(error);
            }
        } else {
            std::vector<B::Item> items;
            if (!B::collectItems(s.args, &items, &error))
                return cmd.error(error);
            const int n = static_cast<int>(items.size());
            // exhaustive is O(n^2) pairs by definition (n(n-1)/2); the
            // feature cache below keeps extraction at O(n).
            for (int i = 0; i < n; ++i) {
                if (pairMode == QLatin1String("reference")) {
                    if (i > 0) rawPairs.append({items[0].path, items[i].path});
                    continue;
                }
                const int last = pairMode == QLatin1String("exhaustive")
                                         ? n - 1
                                         : qMin(n - 1, i + window);
                for (int j = i + 1; j <= last; ++j)
                    rawPairs.append({items[i].path, items[j].path});
            }
        }
        if (rawPairs.isEmpty()) {
            return cmd.error(QStringLiteral(
                    "No image pair (give >= 2 images or PAIRS_FILE)"));
        }
        std::vector<Pair> pairs;
        QSet<QString> stems;
        for (const auto& p : rawPairs) {
            const QString stem = B::uniqueStem(
                    B::sanitizeStem(QFileInfo(p.first).completeBaseName() +
                                    QStringLiteral("_x_") +
                                    QFileInfo(p.second).completeBaseName()),
                    &stems);
            pairs.push_back({p.first, p.second, stem});
        }

        // ---- models -----------------------------------------------------
        const QString defaultFile =
                QStringLiteral("%1-lightglue-%2.gguf").arg(matcher, quant);
        settings.modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir, defaultFile);
        const QString modelName = QFileInfo(settings.modelPath).fileName();
        const aicore_model_entry* entry = aicore_model_by_filename(
                AICORE_MODEL_FAMILY_LIGHTGLUE, modelName.toUtf8().constData());
        if (entry != nullptr) {
            settings.matcherType = entry->matcher_type;
        } else {
            // Custom matcher: same file-name heuristic as the dialog.
            settings.matcherType =
                    modelName.toLower().contains(QLatin1String("sift")) ? 1 : 2;
        }
        if (!B::ensureModel(
                    cmd, settings.modelPath,
                    B::familySource(AICORE_MODEL_FAMILY_LIGHTGLUE, modelName),
                    s.download, &error)) {
            return cmd.error(error);
        }
        QString extractorPath;
        if (settings.matcherType != 1) {
            // Same-quant extractor next to the matcher (worker contract).
            QString extractorName = modelName;
            extractorName.replace(QStringLiteral("aliked-lightglue"),
                                  QStringLiteral("aliked-n16rot"));
            extractorPath = QFileInfo(settings.modelPath)
                                    .absoluteDir()
                                    .filePath(extractorName);
            if (!B::ensureModel(cmd, extractorPath,
                                B::familySource(AICORE_MODEL_FAMILY_ALIKED,
                                                extractorName),
                                s.download, &error)) {
                return cmd.error(error);
            }
        }

        aicore_lightglue_ctx* ctx =
                LightGlueWorker::loadContext(settings, s.device, &error);
        if (!error.isEmpty()) {
            if (ctx) aicore_lightglue_free(ctx);
            return cmd.error(error);
        }
        aicore_lightglue_ctx* cpuCtx = nullptr;
        struct Guard {
            aicore_lightglue_ctx** a;
            aicore_lightglue_ctx** b;
            ~Guard() {
                if (*a) aicore_lightglue_free(*a);
                if (*b) aicore_lightglue_free(*b);
            }
        } guard{&ctx, &cpuCtx};
        settings.device = LightGlueWorker::canonicalDevice(ctx, s.device);

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        if (!extractorPath.isEmpty()) manifest.set("extractor", extractorPath);
        manifest.set("matcher_type", settings.matcherType == 1
                                             ? QStringLiteral("sift")
                                             : QStringLiteral("aliked"));
        manifest.set("pairs", static_cast<int>(pairs.size()));

        // Features per image path, extracted on first use. Memory is about
        // MAX_KEYPOINTS * (descriptor_dim + 4) floats per image (~1.1 MB at
        // 2048 x 128-D), kept for the whole run so exhaustive / reference
        // pairs never re-extract.
        QHash<QString, std::shared_ptr<lightglue_plugin::OwnedFeatures>>
                features;
        auto featuresOf = [&](const QString& path, QString* err)
                -> std::shared_ptr<lightglue_plugin::OwnedFeatures> {
            auto it = features.constFind(path);
            if (it != features.constEnd()) return it.value();
            auto f = std::make_shared<lightglue_plugin::OwnedFeatures>();
            if (!LightGlueWorker::extractFeatures(settings, path, f.get(),
                                                  err)) {
                *err = QStringLiteral("%1: %2").arg(path, *err);
                return nullptr;
            }
            features.insert(path, f);
            return f;
        };

        B::runBatch(
                cmd, s, pairs, manifest,
                [&](const Pair& pair, QJsonObject& rec, QString& err) -> bool {
                    rec["image0"] = pair.path0;
                    rec["image1"] = pair.path1;
                    const auto f0 = featuresOf(pair.path0, &err);
                    if (!f0) return false;
                    const auto f1 = featuresOf(pair.path1, &err);
                    if (!f1) return false;
                    LightGlueWorker::Settings one = settings;
                    one.inputPaths = QStringList{pair.path0, pair.path1};
                    LightGlueRunResult r;
                    QString log;
                    if (!LightGlueWorker::matchFeatures(ctx, &cpuCtx, one, *f0,
                                                        *f1, &r, &log, &err)) {
                        return false;
                    }
                    if (!log.isEmpty()) cmd.print(log);
                    QJsonArray matches;
                    for (const LightGlueMatchResult& m : r.matches) {
                        QJsonObject o{{"idx1", m.idx1},
                                      {"idx2", m.idx2},
                                      { "score",
                                        static_cast<double>(m.score) }};
                        if (m.idx1 >= 0 && m.idx1 < r.keypoints0.size()) {
                            o["x1"] = r.keypoints0[m.idx1].x();
                            o["y1"] = r.keypoints0[m.idx1].y();
                        }
                        if (m.idx2 >= 0 && m.idx2 < r.keypoints1.size()) {
                            o["x2"] = r.keypoints1[m.idx2].x();
                            o["y2"] = r.keypoints1[m.idx2].y();
                        }
                        matches.append(o);
                    }
                    // Same schema as the dialog's "Export matches" + the
                    // feature-image sizes the coordinates refer to.
                    const QJsonObject json{{"source", r.sourceName},
                                           {"image0", r.imageName0},
                                           {"image1", r.imageName1},
                                           {"image0_path", pair.path0},
                                           {"image1_path", pair.path1},
                                           {"feature_size0",
                                            QJsonArray {
                                                r.imageWidth0,
                                                r.imageHeight0
                                            }},
                                           {"feature_size1",
                                            QJsonArray {
                                                r.imageWidth1,
                                                r.imageHeight1
                                            }},
                                           {"runtime_ms", r.runtimeMs},
                                           {"keypoints0", r.nKeypoints0},
                                           {"keypoints1", r.nKeypoints1},
                                           { "matches",
                                             matches }};
                    const QString jsonPath =
                            B::outPath(s, pair.stem, "_matches.json");
                    if (!B::saveJson(json, jsonPath, &err)) return false;
                    QJsonObject outputs{{ "matches", jsonPath }};
                    if (saveVis) {
                        const QImage viz = renderMatchVisualization(
                                lightglue_plugin::load_oriented_qimage(
                                        pair.path0),
                                lightglue_plugin::load_oriented_qimage(
                                        pair.path1),
                                r.keypoints0, r.keypoints1, r.matches,
                                r.imageWidth0, r.imageHeight0, r.imageWidth1,
                                r.imageHeight1);
                        const QString p = B::outPath(s, pair.stem, "_vis.png");
                        if (!viz.isNull()) {
                            if (!B::saveImage(viz, p, &err)) return false;
                            outputs["visualization"] = p;
                        }
                    }
                    rec["matches"] = r.matches.size();
                    rec["keypoints0"] = r.nKeypoints0;
                    rec["keypoints1"] = r.nKeypoints1;
                    rec["device"] = r.resolvedDevice;
                    rec["match_ms"] = r.runtimeMs;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_LIGHTGLUE_MATCH));
    }
#endif  // AICore_ENABLED
};
