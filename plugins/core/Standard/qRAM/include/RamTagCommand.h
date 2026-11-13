// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Headless RAM / RAM++ tagging batch command (-RAM_TAG): one child process
// loads the tagging model once and runs the AICore tagging pipeline over the
// input images, writing one tags JSON per item plus a RESULT_JSON manifest.
// Same model resolution grammar as the other batch commands: an existing
// path wins; otherwise the token is a published catalog filename under the
// tagging cache dir. DOWNLOAD 1 fetches a missing catalog file through the
// shared ecvModelDownloader. Model selection, threshold semantics and the
// result fields mirror the qRAM dialog worker exactly.

#pragma once
#include <QObject>

#include "ecvCommandLineInterface.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/tagging_capi.h>

#include <QCryptographicHash>
#include <QFileInfo>
#include <QJsonArray>

#include "ecvAICoreBatchCommand.h"
#include "ecvAICoreRuntimeHelpers.h"
#endif

#define COMMAND_RAM_TAG "RAM_TAG"

struct CommandRamTag : public ccCommandLineInterface::Command {
    CommandRamTag()
        : ccCommandLineInterface::Command(
                  QObject::tr("RAM / RAM++ image tagging batch"),
                  COMMAND_RAM_TAG) {}

#ifdef AICore_ENABLED
    static const ecvAICoreBatch::OptionTable& options() {
        static const ecvAICoreBatch::OptionTable table = {
                {"THRESHOLD", "float", "-1",
                 "-1 uses the GGUF per-class thresholds; else 0..1"},
                {"THRESHOLD_EPSILON", "float", "0",
                 "Tag2Text only: subtract 0..1 from the class threshold"},
                {"SPECIFIED_TAGS", "string", "",
                 "Tag2Text caption conditioning tags"},
                {"CAPTION", "bool", "0",
                 "Tag2Text only; loads the text decoder weights"},
                {"MAX_LENGTH", "int", "30", "Caption token budget (10..512)"},
                {"BEAM_BATCH", "int", "0", "0=backend default, 1 or 3"},
                {"TAG_LIST", "path", "",
                 "Optional upstream label-space adapter"},
                {"THRESHOLDS", "path", "Optional per-class threshold file"},
                {"LABEL_EMBEDDING", "path", "",
                 "Optional label-embedding adapter"},
                {"CPU_PRECISION", "choice:auto|upstream", "auto",
                 "auto preserves the official fast path; upstream keeps the "
                 "original CPU activation quantization"},
        };
        return table;
    }

    static QString cacheDir() {
        return ecvAICoreBatch::takeCString(
                aicore_tagging_model_cache_dir(),
                [](void* p) { aicore_tagging_free_buffer(p); });
    }

    static ecvAICoreBatch::ModelSource source(const QString& path) {
        const QByteArray name = QFileInfo(path).fileName().toUtf8();
        for (int i = 0; i < aicore_tagging_model_count(); ++i) {
            const auto* e = aicore_tagging_model_at(i);
            if (e && name == QString::fromUtf8(e->filename)) {
                return {QString::fromUtf8(e->download_url ? e->download_url
                                                          : ""),
                        QString::fromUtf8(e->sha256 ? e->sha256 : "")};
            }
        }
        return {};
    }

    static std::vector<ecvAICoreBatch::CatalogRow> rows(const QString& cache) {
        std::vector<ecvAICoreBatch::CatalogRow> out;
        for (int i = 0; i < aicore_tagging_model_count(); ++i) {
            const auto* e = aicore_tagging_model_at(i);
            if (!e || !e->filename) continue;
            const QString fn = QString::fromUtf8(e->filename);
            out.push_back(
                    {fn,
                     QString::fromUtf8(e->display_name ? e->display_name
                                                       : e->filename),
                     QString::fromUtf8(e->download_url ? e->download_url : ""),
                     QDir(cache).filePath(fn),
                     e->supports_caption ? QStringLiteral("tag2text")
                                         : QStringLiteral("ram")});
        }
        return out;
    }

    bool process(ccCommandLineInterface& cmd) override {
        namespace B = ecvAICoreBatch;
        B::Session s;
        switch (B::prepare(cmd, COMMAND_RAM_TAG, QStringLiteral("ram_tag"),
                           QStringLiteral("RAM / RAM++ image tagging"),
                           {&options()}, &s)) {
            case B::Prep::Done:
                return true;
            case B::Prep::Failed:
                return false;
            case B::Prep::Run:
                break;
        }
        const QString cache = cacheDir();
        const std::vector<B::CatalogRow> catalog = rows(cache);
        if (s.listModels) return B::listModels(cmd, s, catalog);

        const QString defaultModel =
                catalog.empty() ? QString() : catalog.front().filename;
        const QString modelPath =
                B::resolveModelPath(s.args.str("MODEL"), cache, defaultModel);
        QString error;
        if (!B::ensureModel(cmd, modelPath, source(modelPath), s.download,
                            &error))
            return cmd.error(error);

        auto opts = aicore_tagging_default_options();
        const auto model = modelPath.toUtf8();
        const auto tagList = s.args.str("TAG_LIST").toUtf8();
        const auto thresholds = s.args.str("THRESHOLDS").toUtf8();
        const auto embedding = s.args.str("LABEL_EMBEDDING").toUtf8();
        opts.model_path = model.constData();
        opts.device = s.device.toUtf8().constData();
        opts.threads = s.threads;
        opts.generate_caption = s.args.toBool("CAPTION", false) ? 1 : 0;
        opts.max_length = s.args.toInt("MAX_LENGTH", 30);
        opts.beam_batch = s.args.toInt("BEAM_BATCH", 0);
        opts.tag_list_path = tagList.isEmpty() ? nullptr : tagList.constData();
        opts.threshold_path =
                thresholds.isEmpty() ? nullptr : thresholds.constData();
        opts.label_embedding_path =
                embedding.isEmpty() ? nullptr : embedding.constData();
        opts.cpu_precision =
                s.args.choice("CPU_PRECISION", QStringLiteral("auto"),
                              {"auto", "upstream"}) ==
                                QStringLiteral("upstream")
                        ? AICORE_TAGGING_CPU_UPSTREAM
                        : AICORE_TAGGING_CPU_AUTO;
        if (!s.args.ok()) return cmd.error(s.args.error());

        auto* context = aicore_tagging_create(&opts);
        if (context == nullptr || !aicore_tagging_ready(context)) {
            const QString message =
                    context ? QString::fromUtf8(
                                      aicore_tagging_last_error(context))
                            : QStringLiteral("allocation failure");
            aicore_tagging_free(context);
            return cmd.error(
                    QStringLiteral("Model load failed: %1").arg(message));
        }
        struct ContextGuard {
            aicore_tagging_ctx* ctx;
            ~ContextGuard() { aicore_tagging_free(ctx); }
        } guard{context};

        aicore_tagging_request request = aicore_tagging_default_request();
        request.threshold = s.args.toFloat("THRESHOLD", -1.0f);
        request.threshold_epsilon = s.args.toFloat("THRESHOLD_EPSILON", 0.0f);
        const auto specified = s.args.str("SPECIFIED_TAGS").toUtf8();
        request.specified_tags =
                specified.isEmpty() ? nullptr : specified.constData();
        if (!s.args.ok()) return cmd.error(s.args.error());

        std::vector<B::Item> items;
        if (!B::collectItems(s.args, &items, &error)) return cmd.error(error);

        B::Manifest manifest(s);
        manifest.set("model", modelPath);
        manifest.set("cpu_precision",
                     opts.cpu_precision == AICORE_TAGGING_CPU_UPSTREAM
                             ? QStringLiteral("upstream")
                             : QStringLiteral("auto"));
        const QString deviceName =
                QString::fromUtf8(aicore_tagging_context_device(context));
        B::runBatch(cmd, s, items, manifest,
                    [&](const B::Item& item, QJsonObject& rec,
                        QString& err) -> bool {
                        QString decodeError;
                        const QImage image = B::loadImage(
                                item.path, QImage::Format_RGB888, &decodeError);
                        if (image.isNull()) {
                            err = decodeError;
                            return false;
                        }
                        auto view = ecvAICoreRuntime::makeImageView(image);
                        aicore_tagging_result* raw = nullptr;
                        const int rc = aicore_tagging_infer_image(
                                context, &view, &request, &raw);
                        std::unique_ptr<aicore_tagging_result,
                                        decltype(&aicore_tagging_result_free)>
                                result(raw, aicore_tagging_result_free);
                        if (rc) {
                            err = QString::fromUtf8(
                                    aicore_tagging_last_error(context));
                            return false;
                        }
                        QJsonArray tags, selected;
                        for (size_t i = 0; i < raw->tag_count; ++i) {
                            const auto& t = raw->tags[i];
                            tags.append(QJsonObject{
                                    {"index", t.index},
                                    {"name", QString::fromUtf8(t.name)},
                                    {"logit", t.logit},
                                    {"probability", t.probability},
                                    {"threshold", t.threshold},
                                    { "selected",
                                      bool(t.selected) }});
                            if (t.selected)
                                selected.append(QString::fromUtf8(t.name));
                        }
                        QJsonObject json{
                                {"source", item.path},
                                {"model", QFileInfo(modelPath).fileName()},
                                {"device", deviceName},
                                {"caption", QString::fromUtf8(raw->caption)},
                                {"selected_tags", selected},
                                {"tags", tags},
                                { "timings",
                                  ecvAICoreBatch::timingsJson(raw->timings) }};
                        const QString tagsPath =
                                B::outPath(s, item.stem, ".json");
                        if (!B::saveJson(json, tagsPath, &err)) return false;
                        rec["input"] = item.path;
                        rec["selected_tags"] = selected;
                        rec["caption"] = json.value("caption");
                        rec["device"] = deviceName;
                        rec["outputs"] = QJsonObject{{ "tags", tagsPath }};
                        return true;
                    });
        return B::finish(cmd, s, manifest);
    }
#else
    bool process(ccCommandLineInterface& cmd) override {
        return cmd.error(QObject::tr("-%1 requires AICore_ENABLED")
                                 .arg(COMMAND_RAM_TAG));
    }
#endif
};
