// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless RelateAnything batch command (-RELATE_ANYTHING_INFER): one child
// process loads the relation model once and scores object relations for
// every input image. Object boxes come from REGIONS (the same JSON contract
// as the qRelateAnything dialog's supplied-regions path: {boxes, labels,
// scores, image_size}; a single object or a per-image map keyed by file
// name/stem), so the detection stage stays composable with -ANYLABEL_INFER
// (rfdetr-detect / detect) instead of being duplicated here. REGIONS files
// are resolved per image by file name, completeBaseName or full path, exactly
// like AnyLabeling's PROMPT_FILE. Model resolution follows the shared
// grammar: an existing path wins; otherwise the token is a published catalog
// filename under the relation cache dir; DOWNLOAD 1 fetches missing catalog
// files through the shared ecvModelDownloader.

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/relation_capi.h>

#include <QFileInfo>
#include <QJsonArray>

#include "ecvAICoreBatchCommand.h"
#include "ecvAICoreRuntimeHelpers.h"
#endif

#define COMMAND_RELATE_ANYTHING_INFER "RELATE_ANYTHING_INFER"

struct CommandRelateAnythingInfer : public ccCommandLineInterface::Command {
    CommandRelateAnythingInfer()
        : ccCommandLineInterface::Command(
                  QObject::tr("RelateAnything scene-relation batch"),
                  COMMAND_RELATE_ANYTHING_INFER) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"REGIONS", "path", "",
                 "Object boxes JSON: {boxes,labels,scores,image_size} for a "
                 "single image, or a map keyed by file name/stem (>= 2 "
                 "boxes); produce it with -ANYLABEL_INFER"},
                {"RELATION_MODEL", "string", "",
                 "GGUF path or published catalog filename"},
                {"MAX_BOXES", "int", "20", "Maximum objects per image"},
                {"TOPK", "int", "0", "0 = backend default top-k pairs"},
                {"STRICT", "choice:auto|opt|strict", "auto",
                 "auto = upstream precision policy; opt = optimized "
                 "arithmetic; strict = always-strict F32 matrices"},
                {"PREDICATES", "string", "",
                 "Comma-separated predicate texts (uses the text student)"},
                {"STUDENT", "path", "", "Native text student GGUF"},
                {"TOKENIZER", "path", "", "Official CLIP tokenizer GGUF"},
                {"VOCABULARY", "path", "", "Predicate vocabulary file"},
                {"VOCAB_NAMES", "path", "", "Predicate vocabulary names"},
                {"PREDICATE_TYPES", "path", "", "Predicate type map"},
        };
        return table;
    }

    static QString cacheDir() {
        return ecvAICoreBatch::takeCString(
                aicore_relation_model_cache_dir(),
                [](void* p) { aicore_relation_free_buffer(p); });
    }

    static ecvAICoreBatch::ModelSource source(const QString& path) {
        const QByteArray name = QFileInfo(path).fileName().toUtf8();
        for (int i = 0; i < aicore_relation_model_count(); ++i) {
            const auto* e = aicore_relation_model_at(i);
            if (e && name == QString::fromUtf8(e->filename) &&
                e->role == AICORE_RELATION_MODEL_RELATIONS) {
                return {QString::fromUtf8(e->download_url ? e->download_url
                                                          : ""),
                        QString::fromUtf8(e->sha256 ? e->sha256 : "")};
            }
        }
        return {};
    }

    static std::vector<ecvAICoreBatch::CatalogRow> rows(const QString& cache) {
        std::vector<ecvAICoreBatch::CatalogRow> out;
        for (int i = 0; i < aicore_relation_model_count(); ++i) {
            const auto* e = aicore_relation_model_at(i);
            if (!e || !e->filename) continue;
            const QString fn = QString::fromUtf8(e->filename);
            out.push_back(
                    {fn,
                     QString::fromUtf8(e->display_name ? e->display_name
                                                       : e->filename),
                     QString::fromUtf8(e->download_url ? e->download_url : ""),
                     QDir(cache).filePath(fn),
                     e->role == AICORE_RELATION_MODEL_TEXT
                             ? QStringLiteral("text")
                     : e->role == AICORE_RELATION_MODEL_TOKENIZER
                             ? QStringLiteral("tokenizer")
                             : QStringLiteral("relations")});
        }
        return out;
    }

    /** One image's object boxes, parsed with the dialog's supplied-regions
     *  contract (order, limits and finiteness checks included). */
    struct Regions {
        std::vector<float> boxes;
        QStringList labels;
        std::vector<float> scores;
        QSize imageSize;
    };

    static bool parseRegions(const QJsonObject& object,
                             const QString& source,
                             int maxBoxes,
                             Regions* out,
                             QString* error) {
        const QJsonArray boxes = object.contains("boxes")
                                         ? object["boxes"].toArray()
                                         : QJsonArray();
        if (boxes.size() < 2 || boxes.size() > maxBoxes) {
            *error = QStringLiteral(
                             "%1: supply between two and %2 object "
                             "boxes (got %3)")
                             .arg(source, QString::number(maxBoxes),
                                  QString::number(boxes.size()));
            return false;
        }
        const QJsonArray labels = object["labels"].toArray();
        const QJsonArray scores = object["scores"].toArray();
        if ((!labels.isEmpty() && labels.size() != boxes.size()) ||
            (!scores.isEmpty() && scores.size() != boxes.size())) {
            *error = QStringLiteral(
                             "%1: object labels and scores must match "
                             "the box count")
                             .arg(source);
            return false;
        }
        for (int i = 0; i < boxes.size(); ++i) {
            const QJsonArray box = boxes[i].toArray();
            if (box.size() != 4) {
                *error = QStringLiteral(
                                 "%1: every box must contain four "
                                 "pixel coordinates")
                                 .arg(source);
                return false;
            }
            for (const auto& value : box) {
                if (!value.isDouble() || !qIsFinite(value.toDouble())) {
                    *error = QStringLiteral(
                                     "%1: box coordinates must be "
                                     "finite numbers")
                                     .arg(source);
                    return false;
                }
                out->boxes.push_back(float(value.toDouble()));
            }
            if (!(out->boxes[i * 4 + 2] > out->boxes[i * 4] &&
                  out->boxes[i * 4 + 3] > out->boxes[i * 4 + 1])) {
                *error = QStringLiteral(
                                 "%1: box corners must be ordered "
                                 "left, top, right, bottom")
                                 .arg(source);
                return false;
            }
            out->labels << (labels.isEmpty() ? QStringLiteral("object")
                                             : labels[i].toString());
            if (!scores.isEmpty()) {
                const double score = scores[i].toDouble();
                if (!qIsFinite(score) || score < 0 || score > 1) {
                    *error = QStringLiteral(
                                     "%1: object confidence must be "
                                     "between zero and one")
                                     .arg(source);
                    return false;
                }
                out->scores.push_back(float(score));
            }
        }
        const QJsonArray size = object["image_size"].toArray();
        if (!size.isEmpty()) {
            if (size.size() != 2 || size[0].toInt() <= 0 ||
                size[1].toInt() <= 0) {
                *error = QStringLiteral(
                                 "%1: image_size must contain positive "
                                 "width and height")
                                 .arg(source);
                return false;
            }
            out->imageSize = QSize(size[0].toInt(), size[1].toInt());
        }
        return true;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_RELATE_ANYTHING_INFER,
                           QStringLiteral("relate_anything"),
                           QStringLiteral("RelateAnything scene relations"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString regionsPath = s.args.str("REGIONS");
        if (regionsPath.isEmpty())
            return cmd.error(QStringLiteral(
                    "-RELATE_ANYTHING_INFER needs REGIONS <boxes.json>"));
        QFile regionsFile(regionsPath);
        if (!regionsFile.open(QIODevice::ReadOnly))
            return cmd.error(
                    QStringLiteral("Cannot read REGIONS: %1").arg(regionsPath));
        QJsonParseError parseError;
        const QJsonDocument regionsDoc =
                QJsonDocument::fromJson(regionsFile.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError)
            return cmd.error(QStringLiteral("Invalid REGIONS JSON: %1")
                                     .arg(parseError.errorString()));
        const QJsonObject regionsRoot = regionsDoc.object();
        const bool perImage =
                regionsRoot.contains("boxes") && regionsRoot["boxes"].isArray();
        if (!perImage && regionsRoot.isEmpty())
            return cmd.error(QStringLiteral("REGIONS must be an object"));

        const QString cache = cacheDir();
        const std::vector<B::CatalogRow> catalog = rows(cache);
        if (s.listModels) return B::listModels(cmd, s, catalog);

        QString defaultModel;
        for (const B::CatalogRow& row : catalog) {
            if (row.role == QLatin1String("relations")) {
                defaultModel = row.filename;
                break;
            }
        }
        const QString modelPath = B::resolveModelPath(
                s.args.str("RELATION_MODEL"), cache, defaultModel);
        QString error;
        if (!B::ensureModel(cmd, modelPath, source(modelPath), s.download,
                            &error))
            return cmd.error(error);

        const QString strict = s.args.choice("STRICT", QStringLiteral("auto"),
                                             {"auto", "opt", "strict"});
        auto opts = aicore_relation_default_options();
        const auto model = modelPath.toUtf8();
        const auto vocabulary = s.args.str("VOCABULARY").toUtf8();
        const auto vocabNames = s.args.str("VOCAB_NAMES").toUtf8();
        const auto predicateTypes = s.args.str("PREDICATE_TYPES").toUtf8();
        const auto student = s.args.str("STUDENT").toUtf8();
        const auto tokenizer = s.args.str("TOKENIZER").toUtf8();
        QStringList predicates = s.args.str("PREDICATES")
                                         .trimmed()
                                         .split(',', QtCompat::SkipEmptyParts);
        for (QString& text : predicates) text = text.trimmed();
        predicates.removeAll(QString());
        std::vector<QByteArray> predicateBytes;
        std::vector<const char*> predicateTexts;
        for (const QString& text : predicates) {
            predicateBytes.push_back(text.toUtf8());
            predicateTexts.push_back(predicateBytes.back().constData());
        }
        opts.model_path = model.constData();
        opts.device = s.device.toUtf8().constData();
        opts.threads = s.threads;
        opts.max_boxes = uint32_t(s.args.toInt("MAX_BOXES", 20));
        opts.topk = uint32_t(s.args.toInt("TOPK", 0));
        opts.strict_f32 = strict == QStringLiteral("strict") ? 1
                          : strict == QStringLiteral("opt")  ? 0
                                                             : -1;
        opts.vocabulary_path =
                vocabulary.isEmpty() ? nullptr : vocabulary.constData();
        opts.vocabulary_names_path =
                vocabNames.isEmpty() ? nullptr : vocabNames.constData();
        opts.predicate_type_map_path =
                predicateTypes.isEmpty() ? nullptr : predicateTypes.constData();
        if (!predicates.empty()) {
            opts.text_student_path =
                    student.isEmpty() ? nullptr : student.constData();
            opts.tokenizer_path =
                    tokenizer.isEmpty() ? nullptr : tokenizer.constData();
            opts.predicate_texts = predicateTexts.data();
            opts.predicate_text_count = predicateTexts.size();
        }
        if (!s.args.ok()) return cmd.error(s.args.error());

        auto* context = aicore_relation_create(&opts);
        if (context == nullptr || !aicore_relation_ready(context)) {
            const QString message =
                    context ? QString::fromUtf8(
                                      aicore_relation_last_error(context))
                            : QStringLiteral("allocation failure");
            aicore_relation_free(context);
            return cmd.error(
                    QStringLiteral("Model load failed: %1").arg(message));
        }
        struct ContextGuard {
            aicore_relation_ctx* ctx;
            ~ContextGuard() { aicore_relation_free(ctx); }
        } guard{context};
        float calibrationA = 1.0f;
        float calibrationB = 0.0f;
        aicore_relation_calibration(context, &calibrationA, &calibrationB);
        const QString deviceName =
                QString::fromUtf8(aicore_relation_context_device(context));

        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        manifest.set("regions", regionsPath);
        manifest.set("calibration", QJsonObject{{"a", calibrationA},
                                                { "b",
                                                  calibrationB }});
        B::runBatch(
                cmd, s, items, manifest,
                [&](const B::Item& item, QJsonObject& rec,
                    QString& err) -> bool {
                    const QFileInfo fi(item.path);
                    Regions regions;
                    if (perImage) {
                        QJsonObject entry;
                        for (const QString& key :
                             {fi.fileName(), fi.completeBaseName(),
                              item.path}) {
                            if (!regionsRoot.contains(key)) continue;
                            entry = regionsRoot[key].toObject();
                            break;
                        }
                        if (entry.isEmpty()) {
                            err = QStringLiteral(
                                          "REGIONS has no entry for "
                                          "%1")
                                          .arg(fi.fileName());
                            return false;
                        }
                        if (!parseRegions(entry, fi.fileName(),
                                          int(opts.max_boxes), &regions, &err))
                            return false;
                    } else if (!parseRegions(regionsRoot,
                                             QFileInfo(regionsPath).fileName(),
                                             int(opts.max_boxes), &regions,
                                             &err)) {
                        return false;
                    }
                    QString decodeError;
                    const QImage image = B::loadImage(
                            item.path, QImage::Format_RGB888, &decodeError);
                    if (image.isNull()) {
                        err = decodeError;
                        return false;
                    }
                    if (regions.imageSize.isEmpty())
                        regions.imageSize = image.size();
                    if (regions.imageSize != image.size()) {
                        err = QStringLiteral(
                                      "REGIONS image_size %1x%2 does not "
                                      "match %3 (%4x%5)")
                                      .arg(regions.imageSize.width())
                                      .arg(regions.imageSize.height())
                                      .arg(fi.fileName())
                                      .arg(image.width())
                                      .arg(image.height());
                        return false;
                    }
                    auto view = ecvAICoreRuntime::makeImageView(image);
                    aicore_relation_request request =
                            aicore_relation_default_request();
                    request.boxes_xyxy = regions.boxes.data();
                    request.object_count = regions.labels.size();
                    request.box_scores = regions.scores.empty()
                                                 ? nullptr
                                                 : regions.scores.data();
                    aicore_relation_result* raw = nullptr;
                    const int rc = aicore_relation_infer_image(context, &view,
                                                               &request, &raw);
                    std::unique_ptr<aicore_relation_result,
                                    decltype(&aicore_relation_result_free)>
                            result(raw, aicore_relation_result_free);
                    if (rc) {
                        err = QString::fromUtf8(
                                aicore_relation_last_error(context));
                        return false;
                    }
                    QJsonArray objects;
                    for (int i = 0; i < int(request.object_count); ++i) {
                        QJsonObject object{
                                {"index", i},
                                {"name", regions.labels[i]},
                                { "box",
                                  QJsonArray {
                                      regions.boxes[size_t(i * 4)],
                                      regions.boxes[size_t(i * 4 + 1)],
                                      regions.boxes[size_t(i * 4 + 2)],
                                      regions.boxes[size_t(i * 4 + 3)]
                                  } }};
                        if (!regions.scores.empty())
                            object["score"] = regions.scores[size_t(i)];
                        objects.append(object);
                    }
                    QJsonArray relations;
                    QJsonArray spatial;
                    QJsonArray semantic;
                    for (size_t i = 0; i < raw->relation_count; ++i) {
                        const auto& edge = raw->relations[i];
                        const int predicate = int(edge.predicate);
                        const QString name =
                                raw->predicate_names &&
                                                predicate <
                                                        int(raw->predicate_count)
                                        ? QString::fromUtf8(raw->predicate_names
                                                                    [predicate])
                                        : QString::number(predicate);
                        QJsonObject json{{"subject", int(edge.subject)},
                                         {"object", int(edge.object)},
                                         {"subject_name",
                                          regions.labels[size_t(edge.subject)]},
                                         {"object_name",
                                          regions.labels[size_t(edge.object)]},
                                         {"predicate", name},
                                         {"predicate_index", predicate},
                                         { "score",
                                           edge.score }};
                        relations.append(json);
                        const bool isSpatial =
                                raw->predicate_spatial &&
                                raw->predicate_spatial[size_t(predicate)];
                        (isSpatial ? spatial : semantic).append(json);
                    }
                    QJsonObject json{
                            {"source", item.path},
                            {"model", QFileInfo(modelPath).fileName()},
                            {"device", deviceName},
                            {"objects", objects},
                            {"relations", relations},
                            {"spatial_relations", spatial},
                            {"semantic_relations", semantic},
                            {"forward_ms", raw->forward_ms},
                            { "timings",
                              ecvAICoreBatch::timingsJson(raw->timings) }};
                    const QString outJson = B::outPath(s, item.stem, ".json");
                    if (!B::saveJson(json, outJson, &err)) return false;
                    rec["input"] = item.path;
                    rec["objects"] = int(request.object_count);
                    rec["relations"] = int(raw->relation_count);
                    rec["spatial"] = spatial.size();
                    rec["semantic"] = semantic.size();
                    rec["device"] = deviceName;
                    rec["outputs"] = QJsonObject{{ "relations", outJson }};
                    return true;
                });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_RELATE_ANYTHING_INFER));
    }
#endif
};
