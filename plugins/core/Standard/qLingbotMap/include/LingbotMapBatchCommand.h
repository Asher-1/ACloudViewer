// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless RGB-sequence reconstruction (-LINGBOT_MAP_RECONSTRUCT).

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/lingbot_capi.h>

#include <QDataStream>
#include <QFileInfo>
#include <QJsonArray>
#include <QSaveFile>
#include <cmath>

#include "LingbotMapWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_LINGBOT_MAP_RECONSTRUCT "LINGBOT_MAP_RECONSTRUCT"

struct CommandLingbotMapReconstruct : public ccCommandLineInterface::Command {
    CommandLingbotMapReconstruct()
        : ccCommandLineInterface::Command(
                  QObject::tr("LingbotMap batch reconstruction"),
                  COMMAND_LINGBOT_MAP_RECONSTRUCT) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"VIDEO", "string", "",
                 "Video input (takes precedence over IMAGE/INPUT_DIR)"},
                {"VIDEO_FPS", "int", "10", "Video sampling FPS"},
                {"MODE", "choice:streaming|windowed", "streaming",
                 "Persistent KV stream or overlap-aligned windows"},
                {"IMAGE_SIZE", "int", "518", "Official processing width"},
                {"MAX_FRAMES", "int", "0", "Frame cap (0 = all)"},
                {"FRAME_STRIDE", "int", "1", "Keep every Nth input frame"},
                {"ROTATE_90", "bool", "0", "Rotate frames clockwise 90°"},
                {"CONF_THRESHOLD", "float", "1.5",
                 "Point visibility threshold (strictly greater)"},
                {"DOWNSAMPLE", "int", "10",
                 "Sequential point-cloud stride after confidence filtering"},
                {"KV_SCALE", "int", "8", "Persistent scale frames"},
                {"KV_WINDOW", "int", "64", "Sliding KV window"},
                {"KEYFRAME_INTERVAL", "int", "0",
                 "Persistent keyframe interval (0 = official auto)"},
                {"WINDOW_SIZE", "int", "64", "Windowed frames per window"},
                {"OVERLAP", "int", "16", "Windowed overlap frames"},
                {"SKY", "choice:none|native|cached", "none",
                 "Sky masking source"},
                {"SKYSEG_MODEL", "string", "",
                 "Native skyseg catalog model override"},
                {"SKY_MASK_DIR", "string", "",
                 "Cached mode: frame-stem PNG masks"},
                {"SAVE_PLY", "bool", "1", "Write merged colored binary PLY"},
                {"SAVE_DEPTH", "bool", "1", "Write per-frame depth .npy"},
                {"SAVE_CONF", "bool", "0", "Write per-frame confidence .npy"},
        };
        return table;
    }

    struct SyncWorker : LingbotMapWorker {
        using LingbotMapWorker::LingbotMapWorker;
        void runHere() { run(); }
    };

    template <typename Fn>
    static quint64 forEachPoint(const LingbotFrameResult& f,
                                float threshold,
                                int downsample,
                                Fn fn) {
        if (f.width <= 0 || f.height <= 0 || f.depth.isEmpty() ||
            f.depthConf.size() != f.depth.size() || f.c2w.size() < 16 ||
            f.intrinsics.size() < 4)
            return 0;
        const float fx = f.intrinsics[0], fy = f.intrinsics[1];
        const float cx = f.intrinsics[2], cy = f.intrinsics[3];
        if (fx <= 0 || fy <= 0) return 0;
        QVector<int> kept;
        kept.reserve(f.depth.size() / 2);
        for (int i = 0; i < f.depth.size(); ++i) {
            if (!std::isfinite(f.depth[i]) || f.depthConf[i] <= threshold)
                continue;
            if (!f.skyKeep.isEmpty() && f.skyKeep[i] == 0) continue;
            kept.push_back(i);
        }
        // Match the GUI / official parse_pc_data behavior exactly: floor
        // count followed by indices 0,N,2N,... (not ceil).
        const int count = kept.size() / qMax(1, downsample);
        const float* R = f.c2w.constData();
        const bool rgb = !f.frameRgb.isNull() &&
                         f.frameRgb.size() == QSize(f.width, f.height);
        quint64 written = 0;
        for (int k = 0; k < count; ++k) {
            const int i = kept[k * qMax(1, downsample)];
            const int x = i % f.width, y = i / f.width;
            const float d = f.depth[i];
            const float X = (x - cx) / fx * d;
            const float Y = (y - cy) / fy * d;
            const float wx = R[0] * X + R[1] * Y + R[2] * d + R[3];
            const float wy = R[4] * X + R[5] * Y + R[6] * d + R[7];
            const float wz = R[8] * X + R[9] * Y + R[10] * d + R[11];
            if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz))
                continue;
            const QRgb c = rgb ? f.frameRgb.pixel(x, y) : qRgb(255, 255, 255);
            fn(wx, wy, wz, static_cast<quint8>(qRed(c)),
               static_cast<quint8>(qGreen(c)), static_cast<quint8>(qBlue(c)));
            ++written;
        }
        return written;
    }

    static bool savePly(const LingbotRunResult& r,
                        float threshold,
                        int downsample,
                        const QString& path,
                        quint64* countOut,
                        QString* error) {
        quint64 count = 0;
        for (const LingbotFrameResult& f : r.frames)
            count += forEachPoint(
                    f, threshold, downsample,
                    [](float, float, float, quint8, quint8, quint8){});
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly)) {
            *error = QStringLiteral("Cannot write %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        const QByteArray header =
                QStringLiteral(
                        "ply\nformat binary_little_endian 1.0\n"
                        "element vertex %1\n"
                        "property float x\nproperty float y\n"
                        "property float z\nproperty uchar red\n"
                        "property uchar green\nproperty uchar blue\n"
                        "end_header\n")
                        .arg(count)
                        .toLatin1();
        if (file.write(header) != header.size()) {
            *error = QStringLiteral("Cannot write PLY header: %1").arg(path);
            return false;
        }
        QDataStream out(&file);
        out.setByteOrder(QDataStream::LittleEndian);
        out.setFloatingPointPrecision(QDataStream::SinglePrecision);
        for (const LingbotFrameResult& f : r.frames) {
            forEachPoint(f, threshold, downsample,
                         [&](float x, float y, float z, quint8 red,
                             quint8 green, quint8 blue) {
                             out << x << y << z << red << green << blue;
                         });
        }
        if (out.status() != QDataStream::Ok || !file.commit()) {
            *error = QStringLiteral("Cannot commit %1: %2")
                             .arg(path, file.errorString());
            return false;
        }
        *countOut = count;
        return true;
    }

    static ecvAICoreBatch::ModelSource source(const QString& path) {
        const aicore_lingbot_model_entry* e = aicore_lingbot_model_by_filename(
                QFileInfo(path).fileName().toUtf8().constData());
        return {e && e->download_url ? QString::fromUtf8(e->download_url)
                                     : QString(),
                e && e->sha256 ? QString::fromLatin1(e->sha256) : QString()};
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_LINGBOT_MAP_RECONSTRUCT,
                           QStringLiteral("lingbot_map"),
                           QStringLiteral("LingBot RGB sequence/video mapping"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir =
                B::takeCString(aicore_lingbot_model_cache_dir(),
                               [](void* p) { aicore_lingbot_free_buffer(p); });
        std::vector<B::CatalogRow> rows;
        QString defaultMap, defaultSky;
        const int mapDefault = aicore_lingbot_model_default_index();
        const int skyDefault = aicore_lingbot_skyseg_model_default_index();
        for (int i = 0; i < aicore_lingbot_model_count(); ++i) {
            const aicore_lingbot_model_entry* e = aicore_lingbot_model_at(i);
            if (!e || !e->filename) continue;
            const QString fn = QString::fromUtf8(e->filename);
            const QString role = QString::fromUtf8(e->role ? e->role : "");
            rows.push_back({fn, QString::fromUtf8(e->display_name),
                            QString::fromUtf8(e->download_url),
                            QDir(cacheDir).filePath(fn), role});
            if (i == mapDefault) defaultMap = fn;
            if (i == skyDefault) defaultSky = fn;
        }
        if (s.listModels) return B::listModels(cmd, s, rows);

        LingbotMapWorker::Settings settings;
        settings.modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cacheDir, defaultMap);
        settings.videoPath = s.args.str("VIDEO");
        settings.videoFps = s.args.toInt("VIDEO_FPS", 10);
        settings.maxFrames = s.args.toInt("MAX_FRAMES", 0);
        settings.frameStride = s.args.toInt("FRAME_STRIDE", 1);
        settings.rotateClockwise90 = s.args.toBool("ROTATE_90", false);
        settings.image_size = s.args.toInt("IMAGE_SIZE", 518);
        settings.confThreshold = s.args.toFloat("CONF_THRESHOLD", 1.5f);
        settings.downsampleFactor = s.args.toInt("DOWNSAMPLE", 10);
        settings.kvScale = s.args.toInt("KV_SCALE", 8);
        settings.kvWindow = s.args.toInt("KV_WINDOW", 64);
        settings.keyframeInterval = s.args.toInt("KEYFRAME_INTERVAL", 0);
        settings.windowSize = s.args.toInt("WINDOW_SIZE", 64);
        settings.overlap = s.args.toInt("OVERLAP", 16);
        settings.mode = s.args.choice("MODE", QStringLiteral("streaming"),
                                      {"streaming", "windowed"}) ==
                                        QLatin1String("windowed")
                                ? LingbotMapWorker::Settings::Mode::Windowed
                                : LingbotMapWorker::Settings::Mode::Streaming;
        const QString sky = s.args.choice("SKY", QStringLiteral("none"),
                                          {"none", "native", "cached"});
        settings.skySource =
                sky == QLatin1String("native")
                        ? LingbotMapWorker::Settings::SkySource::Native
                : sky == QLatin1String("cached")
                        ? LingbotMapWorker::Settings::SkySource::CachedMasks
                        : LingbotMapWorker::Settings::SkySource::None;
        settings.skyMaskDir = s.args.str("SKY_MASK_DIR");
        settings.threads = s.threads;
        settings.device = s.device;
        settings.liveViewMode = 0;
        settings.previewStride = qMax(1, settings.downsampleFactor);
        settings.addResultToDb = false;
        const bool savePlyOutput = s.args.toBool("SAVE_PLY", true);
        const bool saveDepth = s.args.toBool("SAVE_DEPTH", true);
        const bool saveConf = s.args.toBool("SAVE_CONF", false);
        if (!s.args.ok()) return cmd.error(s.args.error());
        if (settings.frameStride < 1 || settings.downsampleFactor < 1 ||
            settings.image_size < 32)
            return cmd.error(QStringLiteral(
                    "FRAME_STRIDE/DOWNSAMPLE must be >=1 and IMAGE_SIZE >=32"));

        QString error;
        if (!B::ensureModel(cmd, settings.modelPath, source(settings.modelPath),
                            s.download, &error))
            return cmd.error(error);
        if (settings.skySource ==
            LingbotMapWorker::Settings::SkySource::Native) {
            settings.skysegModelPath = B::resolveModelPath(
                    s.args.str("SKYSEG_MODEL"), cacheDir, defaultSky);
            if (!B::ensureModel(cmd, settings.skysegModelPath,
                                source(settings.skysegModelPath), s.download,
                                &error))
                return cmd.error(error);
        } else if (settings.skySource ==
                           LingbotMapWorker::Settings::SkySource::CachedMasks &&
                   !QDir(settings.skyMaskDir).exists()) {
            return cmd.error(QStringLiteral("SKY cached needs SKY_MASK_DIR"));
        }
        if (settings.videoPath.isEmpty()) {
            std::vector<B::Item> items;
            if (!B::collectItems(s.args, &items, &error))
                return cmd.error(error);
            for (const B::Item& item : items) settings.inputPaths << item.path;
            if (s.args.has("INPUT_DIR"))
                settings.imageFolder = s.args.str("INPUT_DIR");
        } else if (!QFileInfo(settings.videoPath).isFile()) {
            return cmd.error(QStringLiteral("Video not found: %1")
                                     .arg(settings.videoPath));
        }

        struct Sequence {
            QString stem;
        };
        const QString stem =
                settings.videoPath.isEmpty()
                        ? (s.args.has("INPUT_DIR")
                                   ? B::sanitizeStem(
                                             QDir(s.args.str("INPUT_DIR"))
                                                     .dirName())
                                   : QStringLiteral("sequence"))
                        : B::sanitizeStem(QFileInfo(settings.videoPath)
                                                  .completeBaseName());
        std::vector<Sequence> sequences{{ stem }};
        B::Manifest manifest(s);
        manifest.set("model", settings.modelPath);
        manifest.set("mode",
                     settings.mode == LingbotMapWorker::Settings::Mode::Windowed
                             ? QStringLiteral("windowed")
                             : QStringLiteral("streaming"));
        B::runBatch(
                cmd, s, sequences, manifest,
                [&](const Sequence& sequence, QJsonObject& rec,
                    QString& err) -> bool {
                    LingbotRunResult result;
                    bool success = false;
                    QStringList logs;
                    SyncWorker worker(settings);
                    QObject::connect(
                            &worker, &LingbotMapWorker::resultReady, &worker,
                            [&](const LingbotRunResult& r) { result = r; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &LingbotMapWorker::logMessage, &worker,
                            [&](const QString& m) { logs << m; },
                            Qt::DirectConnection);
                    QObject::connect(
                            &worker, &LingbotMapWorker::taskFinished, &worker,
                            [&](bool ok) { success = ok; },
                            Qt::DirectConnection);
                    worker.runHere();
                    worker.releaseContextOnMainThread();
                    if (!success || result.frames.isEmpty()) {
                        err = logs.isEmpty()
                                      ? QStringLiteral("Inference failed")
                                      : logs.last();
                        return false;
                    }
                    QJsonObject outputs;
                    QJsonArray cameras, depthFiles, confFiles;
                    for (int i = 0; i < result.frames.size(); ++i) {
                        const LingbotFrameResult& f = result.frames[i];
                        const QString frameStem =
                                QStringLiteral("%1_%2")
                                        .arg(sequence.stem)
                                        .arg(f.globalIndex, 6, 10,
                                             QLatin1Char('0'));
                        if (saveDepth) {
                            const QString p =
                                    B::outPath(s, frameStem, "_depth.npy");
                            if (!B::saveNpyFloat32(f.depth.constData(),
                                                   f.height, f.width, 1, p,
                                                   &err))
                                return false;
                            depthFiles.append(p);
                        }
                        if (saveConf) {
                            const QString p =
                                    B::outPath(s, frameStem, "_conf.npy");
                            if (!B::saveNpyFloat32(f.depthConf.constData(),
                                                   f.height, f.width, 1, p,
                                                   &err))
                                return false;
                            confFiles.append(p);
                        }
                        QJsonArray c2w, intr;
                        for (float v : f.c2w) c2w.append(v);
                        for (float v : f.intrinsics) intr.append(v);
                        cameras.append(QJsonObject{{"index", f.globalIndex},
                                                   {"source", f.sourceFile},
                                                   {"width", f.width},
                                                   {"height", f.height},
                                                   {"intrinsics", intr},
                                                   { "cam2world",
                                                     c2w }});
                    }
                    const QString cameraPath =
                            B::outPath(s, sequence.stem, "_cameras.json");
                    if (!B::saveJson(
                                QJsonObject{
                                        { "cameras",
                                          cameras }},
                                cameraPath, &err))
                        return false;
                    outputs["cameras"] = cameraPath;
                    if (saveDepth) outputs["depth"] = depthFiles;
                    if (saveConf) outputs["confidence"] = confFiles;
                    quint64 points = 0;
                    if (savePlyOutput) {
                        const QString p = B::outPath(s, sequence.stem, ".ply");
                        if (!savePly(result, settings.confThreshold,
                                     settings.downsampleFactor, p, &points,
                                     &err))
                            return false;
                        outputs["point_cloud"] = p;
                    }
                    rec["frames"] = result.frames.size();
                    rec["windows"] = result.windowCount;
                    rec["windowed"] = result.windowed;
                    rec["points"] = static_cast<double>(points);
                    rec["device"] = result.device;
                    rec["elapsed_ms_engine"] =
                            static_cast<double>(result.elapsedMs);
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_LINGBOT_MAP_RECONSTRUCT));
    }
#endif
};
