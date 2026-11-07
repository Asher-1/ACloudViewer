// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch face analysis (-FACE_DETECT): one detector load (+ landmark
// head), N images -> <stem>_faces.json (+ annotated overlay) each, plus the
// shared ecvAICoreBatch RESULT_JSON manifest. MODE verify compares every
// input against one REFERENCE image ("is this person in each photo").
// The per-image body is FaceDetectWorker::processImage, the dialog path.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/facedetect_capi.h>

#include "FaceDetectEmbedHelpers.h"
#include "FaceDetectWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_FACE_DETECT "FACE_DETECT"

struct CommandFaceDetect : public ccCommandLineInterface::Command {
    CommandFaceDetect()
        : ccCommandLineInterface::Command(
                  QObject::tr("FaceDetect batch face analysis"),
                  COMMAND_FACE_DETECT) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MODE", "choice:detect|analyze|landmarks|verify", "detect",
                 "detect: boxes + 5 points; analyze: + age/gender; "
                 "landmarks: dense 2D106/3D68; verify: match vs REFERENCE"},
                {"MIN_SCORE", "float", "0.5", "Minimum detection score"},
                {"LANDMARK_MODEL", "string", "",
                 "landmarks: landmark GGUF (default: catalog landmark head)"},
                {"REFERENCE", "path", "", "verify: reference face image"},
                {"VERIFY_THRESHOLD", "float", "0.65",
                 "verify: cosine-distance match threshold"},
                {"ANTI_SPOOF", "bool", "0", "verify: require liveness pass"},
                {"SAVE_VIS", "bool", "1", "Write <stem>_vis.png"},
        };
        return table;
    }

    static QString firstFilename(
            int count, const aicore_facedetect_model_entry* (*at)(int)) {
        const aicore_facedetect_model_entry* e = count > 0 ? at(0) : nullptr;
        return e && e->filename ? QString::fromUtf8(e->filename) : QString();
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_FACE_DETECT,
                           QStringLiteral("facedetect"),
                           QStringLiteral("Face detection / attributes / "
                                          "dense landmarks / verification"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = FaceDetectEmbed::modelCacheDir();
        if (s.listModels) {
            std::vector<B::CatalogRow> rows;
            for (int i = 0; i < aicore_facedetect_model_count(); ++i) {
                const aicore_facedetect_model_entry* e =
                        aicore_facedetect_model_at(i);
                if (!e || !e->filename) continue;
                const QString fn = QString::fromUtf8(e->filename);
                rows.push_back({fn, QString::fromUtf8(e->display_name),
                                QString::fromUtf8(e->download_url),
                                QDir(cacheDir).filePath(fn),
                                e->detector_capable
                                        ? QStringLiteral("detector")
                                        : QStringLiteral("landmark")});
            }
            return B::listModels(cmd, s, rows);
        }

        const QString mode =
                s.args.choice("MODE", QStringLiteral("detect"),
                              {"detect", "analyze", "landmarks", "verify"});
        FaceDetectWorker::Settings settings;
        settings.device = s.device;
        settings.threads = s.threads;
        settings.mode = mode == QStringLiteral("analyze")
                                ? FaceDetectWorker::Mode::Analyze
                        : mode == QStringLiteral("landmarks")
                                ? FaceDetectWorker::Mode::DenseLandmarks
                        : mode == QStringLiteral("verify")
                                ? FaceDetectWorker::Mode::Verify
                                : FaceDetectWorker::Mode::Detect;
        settings.minDetectionScore = s.args.toFloat("MIN_SCORE", 0.5f);
        settings.verifyThreshold = s.args.toFloat("VERIFY_THRESHOLD", 0.65f);
        settings.antiSpoof = s.args.toBool("ANTI_SPOOF", false);
        settings.secondInputPath = s.args.str("REFERENCE");
        const bool saveVis = s.args.toBool("SAVE_VIS", true);
        if (!s.args.ok()) return cmd.error(s.args.error());
        if (settings.mode == FaceDetectWorker::Mode::Verify &&
            !QFileInfo(settings.secondInputPath).isFile()) {
            return cmd.error(
                    QStringLiteral("MODE verify needs REFERENCE <image>"));
        }

        QString error;
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        auto source = [](const QString& path) -> B::ModelSource {
            const aicore_facedetect_model_entry* e =
                    aicore_facedetect_model_by_filename(
                            QFileInfo(path).fileName().toUtf8().constData());
            return {e && e->download_url ? QString::fromUtf8(e->download_url)
                                         : QString(),
                    QString()};
        };
        settings.modelPath = B::resolveModelPath(
                s.args.str("MODEL"), cacheDir,
                firstFilename(aicore_facedetect_detector_model_count(),
                              &aicore_facedetect_detector_model_at));
        if (!B::ensureModel(cmd, settings.modelPath, source(settings.modelPath),
                            s.download, &error)) {
            return cmd.error(error);
        }
        if (settings.mode == FaceDetectWorker::Mode::DenseLandmarks) {
            settings.landmarkModelPath = B::resolveModelPath(
                    s.args.str("LANDMARK_MODEL"), cacheDir,
                    firstFilename(aicore_facedetect_landmark_model_count(),
                                  &aicore_facedetect_landmark_model_at));
            if (!B::ensureModel(cmd, settings.landmarkModelPath,
                                source(settings.landmarkModelPath), s.download,
                                &error)) {
                return cmd.error(error);
            }
        }

        using Ctx = std::unique_ptr<aicore_facedetect_ctx,
                                    void (*)(aicore_facedetect_ctx*)>;
        auto load = [&](const QString& path, Ctx* out) -> bool {
            aicore_facedetect_options* opts = aicore_facedetect_options_new();
            aicore_facedetect_options_set_device(opts,
                                                 s.device.toUtf8().constData());
            aicore_facedetect_options_set_threads(opts, s.threads);
            aicore_facedetect_ctx* ctx = aicore_facedetect_load_opts(
                    path.toUtf8().constData(), opts);
            aicore_facedetect_options_free(opts);
            out->reset(ctx);
            if (ctx == nullptr || !aicore_facedetect_is_ready(ctx)) {
                const char* msg =
                        ctx ? aicore_facedetect_last_error(ctx) : nullptr;
                error = QStringLiteral("Model load failed (%1): %2")
                                .arg(path, msg && msg[0]
                                                   ? QString::fromUtf8(msg)
                                                   : QStringLiteral("unknown"));
                return false;
            }
            return true;
        };
        Ctx ctx(nullptr, aicore_facedetect_free);
        Ctx landmarkCtx(nullptr, aicore_facedetect_free);
        if (!load(settings.modelPath, &ctx)) return cmd.error(error);
        if (!settings.landmarkModelPath.isEmpty() &&
            !load(settings.landmarkModelPath, &landmarkCtx)) {
            return cmd.error(error);
        }

        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        if (!settings.landmarkModelPath.isEmpty())
            manifest.set("landmark_model", settings.landmarkModelPath);
        manifest.set("mode", mode);
        if (char* info = aicore_facedetect_info_json(ctx.get())) {
            manifest.set("model_info",
                         QJsonDocument::fromJson(QByteArray(info)).object());
            aicore_facedetect_free_buffer(info);
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    FaceDetectWorker::Settings one = settings;
                    one.inputPath = item.path;
                    FaceDetectRunResult r;
                    QString message;
                    if (!FaceDetectWorker::processImage(ctx.get(),
                                                        landmarkCtx.get(), one,
                                                        &r, &message, &err)) {
                        return false;
                    }
                    QJsonObject json{{"image", item.path}, { "mode", mode }};
                    if (settings.mode == FaceDetectWorker::Mode::Verify) {
                        json = QJsonDocument::fromJson(r.resultJson).object();
                        rec["verified"] = r.verifyMatched == 1;
                        rec["distance"] = r.verifyDistance;
                    } else {
                        QJsonArray faces;
                        for (const FaceDetectBox& f : r.faces) {
                            QJsonArray lm;
                            for (const auto& p : f.landmarks)
                                lm.append(QJsonArray{p[0], p[1]});
                            QJsonObject o{{"bbox",
                                           QJsonArray {
                                               f.x1,
                                               f.y1,
                                               f.x2,
                                               f.y2
                                           }},
                                          {"score", f.score},
                                          { "landmarks5",
                                            lm }};
                            if (f.age >= 0) o["age"] = f.age;
                            if (f.gender != '?')
                                o["gender"] = QString(QChar(f.gender));
                            if (!f.denseLandmarks2d.empty()) {
                                QJsonArray d2;
                                for (const QPointF& p : f.denseLandmarks2d)
                                    d2.append(QJsonArray{p.x(), p.y()});
                                o["landmarks2d"] = d2;
                            }
                            if (!f.denseLandmarks3d.empty()) {
                                QJsonArray d3;
                                for (const QVector3D& p : f.denseLandmarks3d)
                                    d3.append(QJsonArray{p.x(), p.y(), p.z()});
                                o["landmarks3d"] = d3;
                            }
                            faces.append(o);
                        }
                        json["faces"] = faces;
                        json["bbox_format"] = "x1,y1,x2,y2";
                        json["total_detected"] = r.totalDetected;
                        json["rejected_by_score"] = r.rejectedByScore;
                        rec["faces"] = static_cast<int>(r.faces.size());
                    }
                    const QString jsonPath =
                            B::outPath(s, item.stem, "_faces.json");
                    if (!B::saveJson(json, jsonPath, &err)) return false;
                    QJsonObject outputs{{ "result", jsonPath }};
                    if (saveVis && !r.annotatedImage.isNull()) {
                        const QString p = B::outPath(s, item.stem, "_vis.png");
                        if (!B::saveImage(r.annotatedImage, p, &err))
                            return false;
                        outputs["visualization"] = p;
                    }
                    aicore_pipeline_timings t = B::emptyTimings();
                    if (aicore_facedetect_last_pipeline_timings(ctx.get(),
                                                                &t) == 0) {
                        rec["timings"] = B::timingsJson(t);
                    }
                    rec["summary"] = message;
                    rec["device"] = r.resolvedDevice;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_FACE_DETECT));
    }
#endif  // AICore_ENABLED
};
