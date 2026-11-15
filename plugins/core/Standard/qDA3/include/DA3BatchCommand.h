// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless batch Depth-Anything-3 (-DA3_DEPTH): one model load (plain or
// nested anyview + metric), N images or N view-sets through the same
// path-based C ABI calls as DA3Worker (depth_dense, points, export_glb,
// export_colmap, depth_pose_multi [+ export_colmap_multi]), plus the
// shared ecvAICoreBatch RESULT_JSON manifest. Dense maps are lossless
// float32 .npy; PNGs are previews only.

#pragma once

#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <aicore/depth_capi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "ecvAICoreBatchCommand.h"
#endif  // AICore_ENABLED

#define COMMAND_DA3_DEPTH "DA3_DEPTH"

struct CommandDa3Depth : public ccCommandLineInterface::Command {
    CommandDa3Depth()
        : ccCommandLineInterface::Command(
                  QObject::tr("Depth-Anything-3 batch depth"),
                  COMMAND_DA3_DEPTH) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"MODE", "choice:depth|points|glb|colmap|multiview", "depth",
                 "depth: npy maps + camera; points: PLY; glb / colmap: 3D "
                 "export per image; multiview: one joint solve per set"},
                {"METRIC_MODEL", "string", "",
                 "Nested metric branch GGUF (MODEL is then the anyview "
                 "branch)"},
                {"RESIZE", "int", "0",
                 "Preprocess long-side target (0 = model default)"},
                {"SAVE_CONF", "bool", "1", "depth: write <stem>_conf.npy"},
                {"SAVE_PREVIEW", "bool", "1",
                 "depth/multiview: write <stem>_depth.png (8-bit preview)"},
                {"CONF_THRESH", "float", "0",
                 "points: keep pixels with confidence >= value"},
                {"COLMAP_BINARY", "bool", "1",
                 "colmap/multiview: .bin (1) or .txt (0) sparse model"},
                {"JOINT", "bool", "1",
                 "multiview: one joint pass at the RESIZE target (VRAM grows "
                 "with views); 0 = AICore sequential per-view pass, which "
                 "runs at native resolution (small inputs only)"},
                {"COLMAP_EXPORT", "bool", "0",
                 "multiview: also write <set>/sparse (export_colmap_multi; "
                 "runs the solve a second time)"},
        };
        return table;
    }

    /** Min-max normalized 8-bit preview, near = bright. O(h*w). */
    static QImage preview(const float* d, int w, int h) {
        QImage img(w, h, QImage::Format_Grayscale8);
        float lo = std::numeric_limits<float>::max();
        float hi = std::numeric_limits<float>::lowest();
        for (int i = 0; i < w * h; ++i) {
            if (std::isfinite(d[i]) && d[i] > 0) {
                lo = std::min(lo, d[i]);
                hi = std::max(hi, d[i]);
            }
        }
        const float span = hi > lo ? hi - lo : 1.0f;
        for (int y = 0; y < h; ++y) {
            uchar* row = img.scanLine(y);
            for (int x = 0; x < w; ++x) {
                const float v = d[y * w + x];
                row[x] = (std::isfinite(v) && v > 0)
                                 ? static_cast<uchar>(255.0f *
                                                      (1.0f - (v - lo) / span))
                                 : 0;
            }
        }
        return img;
    }

    static QJsonArray floats(const float* v, int n) {
        QJsonArray a;
        for (int i = 0; i < n; ++i) a.append(v[i]);
        return a;
    }

    /** Binary little-endian PLY (float xyz + uchar rgb). */
    static bool savePly(const float* xyz,
                        const unsigned char* rgb,
                        int n,
                        const QString& path,
                        QString* err) {
        QByteArray bytes = QStringLiteral(
                                   "ply\nformat binary_little_endian 1.0\n"
                                   "element vertex %1\nproperty float x\n"
                                   "property float y\nproperty float z\n"
                                   "property uchar red\nproperty uchar "
                                   "green\nproperty uchar blue\nend_header\n")
                                   .arg(n)
                                   .toLatin1();
        bytes.reserve(bytes.size() + n * 15);
        for (int i = 0; i < n; ++i) {
            bytes.append(reinterpret_cast<const char*>(xyz + 3 * i), 12);
            bytes.append(reinterpret_cast<const char*>(rgb + 3 * i), 3);
        }
        return ecvAICoreBatch::saveBytes(bytes, path, err);
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_DA3_DEPTH, QStringLiteral("da3"),
                           QStringLiteral("Monocular / multi-view depth + "
                                          "camera (Depth-Anything-3)"),
                           {&options(), &B::setOptions()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cacheDir = B::takeCString(aicore_depth_model_cache_dir(),
                                                &aicore_depth_free_buffer);
        const std::vector<B::CatalogRow> rows =
                B::familyRows(AICORE_MODEL_FAMILY_DEPTH, cacheDir);
        if (s.listModels) return B::listModels(cmd, s, rows);

        const QString mode = s.args.choice(
                "MODE", QStringLiteral("depth"),
                {"depth", "points", "glb", "colmap", "multiview"});
        const int resize = s.args.toInt("RESIZE", 0);
        const bool saveConf = s.args.toBool("SAVE_CONF", true);
        const bool savePreview = s.args.toBool("SAVE_PREVIEW", true);
        const float confThresh = s.args.toFloat("CONF_THRESH", 0.0f);
        const int colmapBinary = s.args.toBool("COLMAP_BINARY", true) ? 1 : 0;
        const bool colmapExport = s.args.toBool("COLMAP_EXPORT", false);
        const bool joint = s.args.toBool("JOINT", true);
        if (!s.args.ok()) return cmd.error(s.args.error());

        QString error;
        std::vector<B::Item> items;
        std::vector<B::SetItem> sets;
        if (mode == QStringLiteral("multiview")) {
            if (!B::collectSets(s.args, &sets, &error, 2))
                return cmd.error(error);
            if (sets.empty())
                return cmd.error(
                        QStringLiteral("multiview needs >= 2 images per set"));
        } else if (!B::collectItems(s.args, &items, &error)) {
            return cmd.error(error);
        }

        QString defaultModel;
        for (const B::CatalogRow& r : rows) {
            if (r.role == QStringLiteral("depth")) {
                defaultModel = r.filename;
                break;
            }
        }
        const QString modelPath = B::resolveModelPath(s.args.str("MODEL"),
                                                      cacheDir, defaultModel);
        if (!B::ensureModel(cmd, modelPath,
                            B::familySource(AICORE_MODEL_FAMILY_DEPTH,
                                            QFileInfo(modelPath).fileName()),
                            s.download, &error)) {
            return cmd.error(error);
        }
        QString metricPath;
        if (s.args.has("METRIC_MODEL")) {
            metricPath = B::resolveModelPath(s.args.str("METRIC_MODEL"),
                                             cacheDir, QString());
            if (!B::ensureModel(
                        cmd, metricPath,
                        B::familySource(AICORE_MODEL_FAMILY_DEPTH,
                                        QFileInfo(metricPath).fileName()),
                        s.download, &error)) {
                return cmd.error(error);
            }
        }

        aicore_depth_options* opts = aicore_depth_options_new();
        aicore_depth_options_set_device(opts, s.device.toUtf8().constData());
        aicore_depth_options_set_threads(opts, s.threads);
        aicore_depth_options_set_force_joint_multiview(opts, joint ? 1 : 0);
        aicore_depth_ctx* ctx =
                metricPath.isEmpty()
                        ? aicore_depth_load_opts(modelPath.toUtf8().constData(),
                                                 opts)
                        : aicore_depth_load_nested_opts(
                                  modelPath.toUtf8().constData(),
                                  metricPath.toUtf8().constData(), opts);
        aicore_depth_options_free(opts);
        if (ctx == nullptr || !aicore_depth_is_ready(ctx)) {
            const char* msg = ctx ? aicore_depth_last_error(ctx) : nullptr;
            if (ctx) aicore_depth_free(ctx);
            return cmd.error(QStringLiteral("Model load failed: %1")
                                     .arg(msg && msg[0]
                                                  ? QString::fromUtf8(msg)
                                                  : QStringLiteral("unknown")));
        }
        std::unique_ptr<aicore_depth_ctx, void (*)(aicore_depth_ctx*)> guard(
                ctx, aicore_depth_free);
        if (resize > 0) aicore_depth_set_img_resize_target(ctx, resize);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        if (!metricPath.isEmpty()) manifest.set("metric_model", metricPath);
        manifest.set("mode", mode);
        if (mode == QStringLiteral("multiview")) manifest.set("joint", joint);
        if (char* info = aicore_depth_info_json(ctx)) {
            manifest.set("model_info",
                         QJsonDocument::fromJson(QByteArray(info)).object());
            aicore_depth_free_buffer(info);
        }
        const QString device = QString::fromUtf8(aicore_depth_device_name(ctx));
        auto lastError = [ctx]() {
            const char* e = aicore_depth_last_error(ctx);
            return e && e[0] ? QString::fromUtf8(e)
                             : QStringLiteral("inference failed");
        };
        auto addTimings = [ctx](QJsonObject& rec) {
            aicore_pipeline_timings t = B::emptyTimings();
            if (aicore_depth_last_pipeline_timings(ctx, &t) == 0)
                rec["timings"] = B::timingsJson(t);
        };

        if (mode == QStringLiteral("multiview")) {
            B::runBatch(
                    cmd, s, sets, manifest,
                    [&](const B::SetItem& set, QJsonObject& rec,
                        QString& err) -> bool {
                        rec["inputs"] = QJsonArray::fromStringList(set.paths);
                        std::vector<std::string> paths;
                        for (const QString& p : set.paths)
                            paths.push_back(p.toStdString());
                        std::vector<const char*> cpaths;
                        for (const std::string& p : paths)
                            cpaths.push_back(p.c_str());
                        aicore_depth_multiview_result mv{};
                        if (aicore_depth_depth_pose_multi(
                                    ctx, cpaths.data(),
                                    static_cast<int>(cpaths.size()),
                                    &mv) != 0 ||
                            mv.depth == nullptr) {
                            aicore_depth_multiview_result_free(&mv);
                            err = lastError();
                            return false;
                        }
                        const QString dir = B::outPath(s, set.stem, "");
                        QDir().mkpath(dir);
                        const int hw = mv.height * mv.width;
                        QJsonArray views;
                        bool ok = true;
                        for (int v = 0;
                             ok && v < mv.n_views && v < set.paths.size();
                             ++v) {
                            const QString base =
                                    QFileInfo(set.paths[v]).completeBaseName();
                            const QString npy = QDir(dir).filePath(
                                    base + QStringLiteral("_depth.npy"));
                            ok = B::saveNpyFloat32(mv.depth + v * hw, mv.height,
                                                   mv.width, 1, npy, &err);
                            QJsonObject view{{"image", set.paths[v]},
                                             {"depth", npy},
                                             {"extrinsics_3x4",
                                              floats(mv.ext + 12 * v, 12)},
                                             { "intrinsics_3x3",
                                               floats(mv.intr + 9 * v, 9) }};
                            if (ok && savePreview) {
                                const QString png = QDir(dir).filePath(
                                        base + QStringLiteral("_depth.png"));
                                ok = B::saveImage(preview(mv.depth + v * hw,
                                                          mv.width, mv.height),
                                                  png, &err);
                                view["preview"] = png;
                            }
                            views.append(view);
                        }
                        QJsonObject outputs{{ "dir", dir }};
                        if (ok) {
                            const QString cams =
                                    QDir(dir).filePath("cameras.json");
                            ok = B::saveJson(QJsonObject{{"width", mv.width},
                                                         {"height", mv.height},
                                                         { "views",
                                                           views }},
                                             cams, &err);
                            outputs["cameras"] = cams;
                        }
                        if (ok && colmapExport) {
                            const QString sparse = QDir(dir).filePath("sparse");
                            QDir().mkpath(sparse);
                            // Engine-owned export (consistent processed
                            // colors); write_colmap_from_multiview re-derives
                            // them with a different preprocess.
                            if (aicore_depth_export_colmap_multi(
                                        ctx, cpaths.data(),
                                        static_cast<int>(cpaths.size()),
                                        sparse.toUtf8().constData(),
                                        colmapBinary) != 0) {
                                err = lastError();
                                ok = false;
                            }
                            outputs["colmap"] = sparse;
                        }
                        rec["views"] = mv.n_views;
                        rec["width"] = mv.width;
                        rec["height"] = mv.height;
                        aicore_depth_multiview_result_free(&mv);
                        addTimings(rec);
                        rec["device"] = device;
                        rec["outputs"] = outputs;
                        return ok;
                    });
            return B::finish(cmd, s, manifest);
        }

        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    rec["input"] = item.path;
                    const QByteArray path = item.path.toUtf8();
                    QJsonObject outputs;
                    if (mode == QStringLiteral("glb")) {
                        const QString p = B::outPath(s, item.stem, ".glb");
                        if (aicore_depth_export_glb(ctx, path.constData(),
                                                    p.toUtf8().constData()) !=
                            0) {
                            err = lastError();
                            return false;
                        }
                        outputs["glb"] = p;
                    } else if (mode == QStringLiteral("colmap")) {
                        const QString p = B::outPath(s, item.stem, "_colmap");
                        QDir().mkpath(p);
                        if (aicore_depth_export_colmap(ctx, path.constData(),
                                                       p.toUtf8().constData(),
                                                       colmapBinary) != 0) {
                            err = lastError();
                            return false;
                        }
                        outputs["colmap"] = p;
                    } else if (mode == QStringLiteral("points")) {
                        int n = 0;
                        float* xyz = nullptr;
                        unsigned char* rgb = nullptr;
                        if (aicore_depth_points(ctx, path.constData(),
                                                confThresh, &n, &xyz,
                                                &rgb) != 0) {
                            aicore_depth_free_buffer(xyz);
                            aicore_depth_free_buffer(rgb);
                            err = lastError();
                            return false;
                        }
                        const QString p = B::outPath(s, item.stem, ".ply");
                        const bool ok = savePly(xyz, rgb, n, p, &err);
                        aicore_depth_free_buffer(xyz);
                        aicore_depth_free_buffer(rgb);
                        if (!ok) return false;
                        rec["points"] = n;
                        outputs["ply"] = p;
                    } else {
                        aicore_depth_dense_result dense{};
                        if (aicore_depth_depth_dense(ctx, path.constData(),
                                                     &dense) != 0 ||
                            dense.depth == nullptr) {
                            aicore_depth_dense_result_free(&dense);
                            err = lastError();
                            return false;
                        }
                        const int w = dense.width, h = dense.height;
                        bool ok = true;
                        const QString npy =
                                B::outPath(s, item.stem, "_depth.npy");
                        ok = B::saveNpyFloat32(dense.depth, h, w, 1, npy, &err);
                        outputs["depth"] = npy;
                        if (ok && saveConf && dense.conf) {
                            const QString p =
                                    B::outPath(s, item.stem, "_conf.npy");
                            ok = B::saveNpyFloat32(dense.conf, h, w, 1, p,
                                                   &err);
                            outputs["confidence"] = p;
                        }
                        if (ok && savePreview) {
                            const QString p =
                                    B::outPath(s, item.stem, "_depth.png");
                            ok = B::saveImage(preview(dense.depth, w, h), p,
                                              &err);
                            outputs["preview"] = p;
                        }
                        const bool hasPose =
                                std::any_of(dense.intr, dense.intr + 9,
                                            [](float v) { return v != 0.0f; });
                        if (ok) {
                            const QString p =
                                    B::outPath(s, item.stem, "_camera.json");
                            QJsonObject camera{
                                    {"width", w},
                                    {"height", h},
                                    {"is_metric", dense.is_metric != 0},
                                    { "has_pose",
                                      hasPose }};
                            if (hasPose) {
                                camera["extrinsics_3x4"] =
                                        floats(dense.ext, 12);
                                camera["intrinsics_3x3"] =
                                        floats(dense.intr, 9);
                            }
                            ok = B::saveJson(camera, p, &err);
                            outputs["camera"] = p;
                        }
                        rec["width"] = w;
                        rec["height"] = h;
                        rec["is_metric"] = dense.is_metric != 0;
                        aicore_depth_dense_result_free(&dense);
                        if (!ok) return false;
                    }
                    addTimings(rec);
                    rec["device"] = device;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else   // !AICore_ENABLED
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_DA3_DEPTH));
    }
#endif  // AICore_ENABLED
};
