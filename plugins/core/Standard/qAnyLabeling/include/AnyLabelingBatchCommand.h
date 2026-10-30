// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Generic qAnyLabeling auto-annotation batch command (-ANYLABEL_INFER).

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/aliked_capi.h>
#include <aicore/attribute_capi.h>
#include <aicore/deeplsd_capi.h>
#include <aicore/depth_capi.h>
#include <aicore/facedetect_capi.h>
#include <aicore/gkd_capi.h>
#include <aicore/model_catalog_capi.h>
#include <aicore/ocr_capi.h>
#include <aicore/rfdetr_capi.h>
#include <aicore/rmbg_capi.h>
#include <aicore/sam3_capi.h>
#include <aicore/yolo_capi.h>

#include <QFile>
#include <QJsonArray>

#include "AnnotationWorker.h"
#include "ecvAICoreBatchCommand.h"
#endif

#define COMMAND_ANYLABEL_INFER "ANYLABEL_INFER"

struct CommandAnyLabelInfer : public ccCommandLineInterface::Command {
    CommandAnyLabelInfer()
        : ccCommandLineInterface::Command(
                  QObject::tr("AnyLabeling batch auto-annotation"),
                  COMMAND_ANYLABEL_INFER) {}
#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"TASK",
                 "choice:detect|segment|semantic|pose|obb|classify|"
                 "world|yoloe|rfdetr-detect|rfdetr-segment|sam3-pvs|"
                 "sam3-pcs|rmbg|gkd|facedetect|facedetect-dense|ocr|"
                 "depth|depth-nested|deeplsd|aliked|attribute",
                 "detect", "qAnyLabeling inference workflow"},
                {"METRIC_MODEL", "string", "",
                 "Dependency: nested metric / face landmark / OCR detector"},
                {"TEXT_MODEL", "string", "", "YOLO-World/YOLOE text encoder"},
                {"CONFIDENCE", "float", "0.5", "Generic score threshold"},
                {"TEXT", "string", "", "Text/classes/keypoint names"},
                {"LABEL", "string", "object", "SAM prompt result label"},
                {"POINT", "list", "", "Positive point x,y (repeatable)"},
                {"NEG_POINT", "list", "", "Negative point x,y (repeatable)"},
                {"BOX", "string", "", "Prompt/ROI box x0,y0,x1,y1"},
                {"ROI", "list", "", "Additional ROI x0,y0,x1,y1"},
                {"VISUAL_PROMPT", "list", "", "YOLOE visual box x0,y0,x1,y1"},
                {"SUPPORT_IMAGE", "string", "", "GKD one-shot support image"},
                {"SUPPORT_POINT", "list", "",
                 "GKD support keypoint x,y (repeatable)"},
                {"PROMPT_FILE", "string", "",
                 "Per-image JSON keyed by file name/stem; fields text,label,"
                 "points,neg_points,box,rois,visual_prompts"},
                {"SLICED", "bool", "0", "YOLO detect SAHI tiling"},
                {"SLICE_WIDTH", "int", "512", "SAHI tile width"},
                {"SLICE_HEIGHT", "int", "512", "SAHI tile height"},
                {"SLICE_OVERLAP", "float", "0.2", "SAHI fractional overlap"},
                {"SLICE_MERGE", "float", "0.45", "SAHI merge IoU"},
                {"SAVE_MASKS", "bool", "1", "Write masks as binary PNG"},
                {"SAVE_DEPTH_PREVIEW", "bool", "1",
                 "Depth tasks: write qAnyLabeling 8-bit preview"},
        };
        return table;
    }

    static QString task(const ecvAICoreBatch::Args& args) {
        const QString value = args.str("TASK", QStringLiteral("detect"));
        static const QSet<QString> allowed = {
                "detect",        "segment",
                "semantic",      "pose",
                "obb",           "classify",
                "world",         "yoloe",
                "rfdetr-detect", "rfdetr-segment",
                "sam3-pvs",      "sam3-pcs",
                "rmbg",          "gkd",
                "facedetect",    "facedetect-dense",
                "ocr",           "depth",
                "depth-nested",  "deeplsd",
                "aliked",        "attribute"};
        return allowed.contains(value) ? value : QString();
    }

    static bool coords(const QString& text, int n, float* out) {
        const QStringList parts = QString(text).replace(',', ' ').split(
                ' ', QtCompat::SkipEmptyParts);
        if (parts.size() != n) return false;
        for (int i = 0; i < n; ++i) {
            bool ok = false;
            out[i] = parts[i].toFloat(&ok);
            if (!ok) return false;
        }
        return true;
    }

    static bool jsonCoords(const QJsonValue& value, int n, float* out) {
        const QJsonArray a = value.toArray();
        if (a.size() != n) return false;
        for (int i = 0; i < n; ++i) {
            if (!a[i].isDouble()) return false;
            out[i] = static_cast<float>(a[i].toDouble());
        }
        return true;
    }

    static bool applyPromptJson(const QJsonObject& o,
                                AnnotationPrompt* p,
                                QString* error) {
        if (o.contains("text")) p->text = o["text"].toString();
        if (o.contains("label")) p->label = o["label"].toString();
        auto points = [&](const char* key, QVector<QPointF>* dst) {
            if (!o.contains(key)) return true;
            dst->clear();
            for (const QJsonValue& v : o[key].toArray()) {
                float xy[2];
                if (!jsonCoords(v, 2, xy)) return false;
                dst->append(QPointF(xy[0], xy[1]));
            }
            return true;
        };
        auto boxes = [&](const char* key, QVector<QRectF>* dst) {
            if (!o.contains(key)) return true;
            dst->clear();
            for (const QJsonValue& v : o[key].toArray()) {
                float b[4];
                if (!jsonCoords(v, 4, b)) return false;
                dst->append(QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3]))
                                    .normalized());
            }
            return true;
        };
        if (!points("points", &p->positive) ||
            !points("neg_points", &p->negative)) {
            *error = QStringLiteral("points must be [[x,y],...]");
            return false;
        }
        if (!boxes("rois", &p->rois) ||
            !boxes("visual_prompts", &p->visualPrompts)) {
            *error = QStringLiteral("boxes must be [[x0,y0,x1,y1],...]");
            return false;
        }
        if (o.contains("box")) {
            float b[4];
            if (!jsonCoords(o["box"], 4, b)) {
                *error = QStringLiteral("box must be [x0,y0,x1,y1]");
                return false;
            }
            p->box = QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3]))
                             .normalized();
            p->useBox = true;
        }
        return true;
    }

    static QString cacheDir(const QString& taskName) {
        namespace B = ecvAICoreBatch;
        if (taskName == QLatin1String("attribute"))
            return B::takeCString(aicore_attribute_model_cache_dir(), [](void* p) {
                aicore_attribute_free_buffer(static_cast<char*>(p));
            });
        if (taskName.startsWith("depth"))
            return B::takeCString(aicore_depth_model_cache_dir(),
                                  aicore_depth_free_buffer);
        if (taskName == QLatin1String("deeplsd"))
            return B::takeCString(aicore_deeplsd_model_cache_dir(),
                                  aicore_deeplsd_free_buffer);
        if (taskName == QLatin1String("aliked"))
            return B::takeCString(aicore_aliked_model_cache_dir(),
                                  aicore_aliked_free_buffer);
        if (taskName.startsWith("rfdetr"))
            return B::takeCString(aicore_rfdetr_model_cache_dir(),
                                  aicore_rfdetr_free_buffer);
        if (taskName.startsWith("sam3"))
            return B::takeCString(aicore_sam3_model_cache_dir(),
                                  aicore_sam3_free_buffer);
        if (taskName == QLatin1String("rmbg"))
            return B::takeCString(aicore_rmbg_model_cache_dir(),
                                  aicore_rmbg_free_buffer);
        if (taskName == QLatin1String("gkd"))
            return B::takeCString(aicore_gkd_model_cache_dir(),
                                  aicore_gkd_free_buffer);
        if (taskName.startsWith("facedetect"))
            return B::takeCString(aicore_facedetect_model_cache_dir(),
                                  aicore_facedetect_free_buffer);
        if (taskName == QLatin1String("ocr"))
            return B::takeCString(aicore_ocr_model_cache_dir(), [](void* p) {
                aicore_ocr_free_buffer(static_cast<char*>(p));
            });
        return B::takeCString(aicore_yolo_model_cache_dir(),
                              aicore_yolo_free_buffer);
    }

    static ecvAICoreBatch::ModelSource source(const QString& taskName,
                                              const QString& path) {
        const QByteArray name = QFileInfo(path).fileName().toUtf8();
        // Local recipes have no published URL/digest; never fall back to the YOLO catalog or invent downloads.
        if (taskName == QLatin1String("attribute")) return {};
        if (taskName.startsWith("depth"))
            return ecvAICoreBatch::familySource(AICORE_MODEL_FAMILY_DEPTH,
                                                QString::fromUtf8(name));
        if (taskName == QLatin1String("deeplsd"))
            return ecvAICoreBatch::familySource(AICORE_MODEL_FAMILY_DEEPLSD,
                                                QString::fromUtf8(name));
        if (taskName == QLatin1String("aliked"))
            return ecvAICoreBatch::familySource(AICORE_MODEL_FAMILY_ALIKED,
                                                QString::fromUtf8(name));
#define ANYLABEL_SOURCE(API)                                           \
    if (const auto* e = API(name.constData())) return {                \
            QString::fromUtf8(e->download_url ? e->download_url : ""), \
                    QString()                                          \
        }
        if (taskName.startsWith("rfdetr")) {
            ANYLABEL_SOURCE(aicore_rfdetr_model_by_filename);
        } else if (taskName.startsWith("sam3")) {
            ANYLABEL_SOURCE(aicore_sam3_model_by_filename);
        } else if (taskName == QLatin1String("rmbg")) {
            ANYLABEL_SOURCE(aicore_rmbg_model_by_filename);
        } else if (taskName == QLatin1String("gkd")) {
            ANYLABEL_SOURCE(aicore_gkd_model_by_filename);
        } else if (taskName.startsWith("facedetect")) {
            ANYLABEL_SOURCE(aicore_facedetect_model_by_filename);
        } else if (taskName == QLatin1String("ocr")) {
            ANYLABEL_SOURCE(aicore_ocr_model_by_filename);
        } else {
            ANYLABEL_SOURCE(aicore_yolo_model_by_filename);
        }
#undef ANYLABEL_SOURCE
        return {};
    }

    static std::vector<ecvAICoreBatch::CatalogRow> rows(const QString& taskName,
                                                        const QString& cache) {
        namespace B = ecvAICoreBatch;
        if (taskName.startsWith("depth"))
            return B::familyRows(AICORE_MODEL_FAMILY_DEPTH, cache, "depth");
        if (taskName == QLatin1String("deeplsd"))
            return B::familyRows(AICORE_MODEL_FAMILY_DEEPLSD, cache);
        if (taskName == QLatin1String("aliked"))
            return B::familyRows(AICORE_MODEL_FAMILY_ALIKED, cache);
        std::vector<B::CatalogRow> out;
        auto append = [&](const char* filename, const char* display,
                          const char* url, const QString& role) {
            if (!filename) return;
            const QString fn = QString::fromUtf8(filename);
            out.push_back({fn, QString::fromUtf8(display ? display : filename),
                           QString::fromUtf8(url ? url : ""),
                           QDir(cache).filePath(fn), role});
        };
        if (taskName == QLatin1String("attribute")) {
            for (int i = 0; i < aicore_attribute_preset_count(); ++i) {
                const auto* e = aicore_attribute_preset_at(i);
                if (e) append(e->filename, e->display_name, nullptr, "attribute");
            }
        } else if (taskName.startsWith("rfdetr")) {
            for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
                const auto* e = aicore_rfdetr_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           e->segmentation_capable ? "rfdetr-segment"
                                                   : "rfdetr-detect");
            }
        } else if (taskName.startsWith("sam3")) {
            for (int i = 0; i < aicore_sam3_model_count(); ++i) {
                const auto* e = aicore_sam3_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           QString::fromUtf8(e->model_family));
            }
        } else if (taskName == QLatin1String("rmbg")) {
            for (int i = 0; i < aicore_rmbg_model_count(); ++i) {
                const auto* e = aicore_rmbg_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           "rmbg");
            }
        } else if (taskName == QLatin1String("gkd")) {
            for (int i = 0; i < aicore_gkd_model_count(); ++i) {
                const auto* e = aicore_gkd_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           "gkd");
            }
        } else if (taskName.startsWith("facedetect")) {
            for (int i = 0; i < aicore_facedetect_detector_model_count(); ++i) {
                const auto* e = aicore_facedetect_detector_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           "detector");
            }
            for (int i = 0; i < aicore_facedetect_landmark_model_count(); ++i) {
                const auto* e = aicore_facedetect_landmark_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           "landmark");
            }
        } else if (taskName == QLatin1String("ocr")) {
            for (int i = 0; i < aicore_ocr_model_count(); ++i) {
                const auto* e = aicore_ocr_model_at(i);
                if (e)
                    append(e->filename, e->display_name, e->download_url,
                           QString::fromUtf8(e->role ? e->role : ""));
            }
        } else {
            for (int i = 0; i < aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
                 ++i) {
                const auto* e = aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
                if (!e || !e->filename) continue;
                const QString fn = QString::fromUtf8(e->filename);
                const QString rowTask =
                        fn.startsWith("yoloe-") ? QStringLiteral("yoloe")
                        : fn.contains("-world")
                                ? QStringLiteral("world")
                                : QString::fromUtf8(e->task ? e->task : "");
                append(e->filename, e->display_name, e->download_url, rowTask);
            }
        }
        return out;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(
                cmd, COMMAND_ANYLABEL_INFER, QStringLiteral("anylabel"),
                QStringLiteral("qAnyLabeling generic auto-annotation"),
                {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString taskName = task(s.args);
        if (taskName.isEmpty())
            return cmd.error(QStringLiteral("Invalid TASK"));
        const QString cache = cacheDir(taskName);
        const std::vector<B::CatalogRow> catalog = rows(taskName, cache);
        if (s.listModels) return B::listModels(cmd, s, catalog);

        QString defaultModel;
        for (const B::CatalogRow& row : catalog) {
            if (taskName.startsWith("rfdetr") && row.role != taskName) continue;
            if ((taskName == QLatin1String("facedetect") ||
                 taskName == QLatin1String("facedetect-dense")) &&
                row.role != QLatin1String("detector"))
                continue;
            if (taskName == QLatin1String("ocr") &&
                row.role != QLatin1String("rec"))
                continue;
            if (!taskName.startsWith("rfdetr") &&
                !taskName.startsWith("facedetect") &&
                taskName != QLatin1String("ocr") &&
                (taskName == QLatin1String("detect") ||
                 taskName == QLatin1String("segment") ||
                 taskName == QLatin1String("semantic") ||
                 taskName == QLatin1String("pose") ||
                 taskName == QLatin1String("obb") ||
                 taskName == QLatin1String("classify") ||
                 taskName == QLatin1String("world") ||
                 taskName == QLatin1String("yoloe")) &&
                row.role != taskName)
                continue;
            defaultModel = row.filename;
            break;
        }
        const QString modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cache, defaultModel);
        QString error;
        if (!B::ensureModel(cmd, modelPath, source(taskName, modelPath),
                            s.download, &error))
            return cmd.error(error);
        const QString metricPath =
                B::resolveModelPath(s.args.str("METRIC_MODEL"), cache);
        if (!metricPath.isEmpty() &&
            !B::ensureModel(cmd, metricPath, source(taskName, metricPath),
                            s.download, &error))
            return cmd.error(error);
        const QString textPath = B::resolveModelPath(
                s.args.str("TEXT_MODEL"), cacheDir(QStringLiteral("detect")));
        if (!textPath.isEmpty() &&
            !B::ensureModel(cmd, textPath,
                            source(QStringLiteral("detect"), textPath),
                            s.download, &error))
            return cmd.error(error);
        if ((taskName == QLatin1String("ocr") ||
             taskName == QLatin1String("depth-nested") ||
             taskName == QLatin1String("facedetect-dense")) &&
            metricPath.isEmpty())
            return cmd.error(
                    QStringLiteral("TASK %1 needs METRIC_MODEL").arg(taskName));

        AnnotationPrompt global;
        global.text = s.args.str("TEXT");
        global.label = s.args.str("LABEL", QStringLiteral("object"));
        global.textModelPath = textPath;
        global.slicedDetection = s.args.toBool("SLICED", false);
        global.sliceWidth = s.args.toInt("SLICE_WIDTH", 512);
        global.sliceHeight = s.args.toInt("SLICE_HEIGHT", 512);
        global.sliceOverlap = s.args.toFloat("SLICE_OVERLAP", 0.2f);
        global.sliceMergeThreshold = s.args.toFloat("SLICE_MERGE", 0.45f);
        const double confidence = s.args.toFloat("CONFIDENCE", 0.5f);
        const bool saveMasks = s.args.toBool("SAVE_MASKS", true);
        const bool saveDepth = s.args.toBool("SAVE_DEPTH_PREVIEW", true);
        if (!s.args.ok()) return cmd.error(s.args.error());
        auto parsePoints = [&](const char* key, QVector<QPointF>* dst) {
            for (const QString& value : s.args.list(key)) {
                float xy[2];
                if (!coords(value, 2, xy)) {
                    error = QStringLiteral("%1 expects x,y: %2")
                                    .arg(key, value);
                    return false;
                }
                dst->append(QPointF(xy[0], xy[1]));
            }
            return true;
        };
        auto parseBoxes = [&](const char* key, QVector<QRectF>* dst) {
            for (const QString& value : s.args.list(key)) {
                float b[4];
                if (!coords(value, 4, b)) {
                    error = QStringLiteral("%1 expects x0,y0,x1,y1: %2")
                                    .arg(key, value);
                    return false;
                }
                dst->append(QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3]))
                                    .normalized());
            }
            return true;
        };
        if (!parsePoints("POINT", &global.positive) ||
            !parsePoints("NEG_POINT", &global.negative) ||
            !parsePoints("SUPPORT_POINT", &global.visualSupportPoints) ||
            !parseBoxes("ROI", &global.rois) ||
            !parseBoxes("VISUAL_PROMPT", &global.visualPrompts))
            return cmd.error(error);
        if (s.args.has("BOX")) {
            float b[4];
            if (!coords(s.args.str("BOX"), 4, b))
                return cmd.error(QStringLiteral("BOX expects x0,y0,x1,y1"));
            global.box = QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3]))
                                 .normalized();
            global.useBox = true;
        }
        if (s.args.has("SUPPORT_IMAGE")) {
            global.visualSupportImage = QImage(s.args.str("SUPPORT_IMAGE"));
            if (global.visualSupportImage.isNull())
                return cmd.error(QStringLiteral("Cannot load SUPPORT_IMAGE"));
        }
        QJsonObject promptFile;
        if (s.args.has("PROMPT_FILE")) {
            QFile f(s.args.str("PROMPT_FILE"));
            if (!f.open(QIODevice::ReadOnly))
                return cmd.error(QStringLiteral("Cannot read PROMPT_FILE"));
            const QJsonDocument d = QJsonDocument::fromJson(f.readAll());
            if (!d.isObject())
                return cmd.error(
                        QStringLiteral("PROMPT_FILE must be an object"));
            promptFile = d.object();
        }
        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        B::Manifest manifest(s);
        manifest.set("task_name", taskName);
        manifest.set("model", modelPath);
        if (!metricPath.isEmpty()) manifest.set("dependency_model", metricPath);
        AnnotationWorker worker;  // caches the model across every image
        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    const QImage image(item.path);
                    if (image.isNull()) {
                        err = QStringLiteral("Cannot decode %1").arg(item.path);
                        return false;
                    }
                    AnnotationPrompt prompt = global;
                    const QFileInfo fi(item.path);
                    for (const QString& key :
                         {fi.fileName(), fi.completeBaseName(), item.path}) {
                        if (!promptFile.contains(key)) continue;
                        if (!applyPromptJson(promptFile[key].toObject(),
                                             &prompt, &err))
                            return false;
                        break;
                    }
                    AnnotationInference result;
                    bool received = false;
                    const QMetaObject::Connection connection = QObject::connect(
                            &worker, &AnnotationWorker::finished, &worker,
                            [&](const AnnotationInference& r) {
                                result = r;
                                received = true;
                            },
                            Qt::DirectConnection);
                    worker.infer(modelPath, metricPath, s.device, taskName,
                                 item.path, image, 1, confidence, 1, prompt);
                    QObject::disconnect(connection);
                    if (!received || !result.error.isEmpty()) {
                        err = result.error.isEmpty()
                                      ? QStringLiteral("No inference result")
                                      : result.error;
                        return false;
                    }
                    QJsonArray shapes;
                    QJsonObject outputs;
                    for (int i = 0; i < result.shapes.size(); ++i) {
                        const AnnotationShape& shape = result.shapes[i];
                        QJsonObject object = shape.toJson();
                        object.remove("acloudviewer_mask_png");
                        if (saveMasks && !shape.mask.isNull()) {
                            QImage binary(shape.mask.size(),
                                          QImage::Format_Grayscale8);
                            for (int y = 0; y < binary.height(); ++y) {
                                uchar* dst = binary.scanLine(y);
                                for (int x = 0; x < binary.width(); ++x)
                                    dst[x] = qAlpha(shape.mask.pixel(x, y)) > 0
                                                     ? 255
                                                     : 0;
                            }
                            const QString p = B::outPath(
                                    s, item.stem,
                                    QStringLiteral("_mask_%1.png").arg(i));
                            if (!B::saveImage(binary, p, &err)) return false;
                            object["mask"] = p;
                        }
                        shapes.append(object);
                    }
                    if (saveDepth && !result.depthPreview.isNull()) {
                        const QString p =
                                B::outPath(s, item.stem, "_depth.png");
                        if (!B::saveImage(result.depthPreview, p, &err))
                            return false;
                        outputs["depth_preview"] = p;
                    }
                    const QString annotation =
                            B::outPath(s, item.stem, ".json");
                    QJsonObject json{{"version", "ACloudViewer"},
                                     {"flags", result.flags},
                                     {"shapes", shapes},
                                     {"imagePath", item.path},
                                     {"imageData", QJsonValue::Null},
                                     {"imageHeight", image.height()},
                                     {"imageWidth", image.width()},
                                     {"task", taskName},
                                     {"model", modelPath},
                                     {"device", result.device},
                                     { "summary",
                                       result.summary }};
                    if (!B::saveJson(json, annotation, &err)) return false;
                    outputs["annotation"] = annotation;
                    rec["input"] = item.path;
                    rec["shapes"] = result.shapes.size();
                    rec["flags"] = result.flags;
                    rec["device"] = result.device;
                    rec["outputs"] = outputs;
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_ANYLABEL_INFER));
    }
#endif
};
