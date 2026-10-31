// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LLMBatchCommand.h"

#ifdef AICore_ENABLED
#include <cmath>
#include <limits>
#include <memory>

#include "aicore/llm_capi.h"
#include "ecvAICoreRuntimeHelpers.h"
#endif

bool CommandLLMGenerate::process(ccCommandLineInterface& cmd) {
#ifndef AICore_ENABLED
    return cmd.error(QStringLiteral("LLM_GENERATE requires AICore"));
#else
    namespace B = ecvAICoreBatch;
    B::Session session;
    const auto prep =
            B::prepare(cmd, COMMAND_LLM_GENERATE, "llm",
                       "Local LLM/VLM text generation", {&options()}, &session);
    if (prep != B::Prep::Run) return prep == B::Prep::Done;
    auto& args = session.args;
    const QString cache = B::takeCString(aicore_llm_model_cache_dir(),
                                         aicore_llm_free_buffer);
    if (cache.isEmpty()) return cmd.error("LLM model cache is unavailable");
    if (session.listModels) {
        std::vector<B::CatalogRow> rows;
        for (int i = 0; i < aicore_llm_model_count(); ++i) {
            const auto* entry = aicore_llm_model_at(i);
            if (!entry || !entry->filename) continue;
            const QString name = QString::fromUtf8(entry->filename);
            rows.push_back({name, QString::fromUtf8(entry->display_name),
                            QString::fromUtf8(entry->download_url),
                            QDir(cache).filePath(name),
                            entry->role == AICORE_LLM_ROLE_MODEL ? "model"
                                                                 : "mmproj"});
        }
        return B::listModels(cmd, session, rows);
    }

    const QByteArray prompt = args.str("PROMPT").toUtf8();
    const QByteArray system = args.str("SYSTEM").toUtf8();
    if (prompt.trimmed().isEmpty()) return cmd.error("PROMPT is required");
    auto request = aicore_llm_default_request();
    const int maxTokens =
            args.toInt("MAX_TOKENS", static_cast<int>(request.max_tokens));
    const float temperature = args.toFloat("TEMPERATURE", request.temperature);
    args.toBool("RECURSIVE", false);
    if (!args.ok()) return cmd.error(args.error());
    if (maxTokens <= 0) return cmd.error("MAX_TOKENS must be positive");
    if (!std::isfinite(temperature) || temperature < 0)
        return cmd.error("TEMPERATURE must be finite and nonnegative");
    request.max_tokens = static_cast<uint32_t>(maxTokens);
    request.temperature = temperature;

    const bool imagesRequested = args.has("IMAGE") || args.has("INPUT_DIR") ||
                                 args.has("INPUT_LIST");
    std::vector<B::Item> inputs;
    QString error;
    if (imagesRequested) {
        if (!B::collectItems(args, &inputs, &error)) return cmd.error(error);
    } else {
        inputs.push_back({QString(), QStringLiteral("response")});
    }
    const auto* defaultModel =
            aicore_llm_model_at(aicore_llm_model_default_index());
    const QString model = B::resolveModelPath(
            args.str("MODEL"), cache,
            defaultModel ? QString::fromUtf8(defaultModel->filename)
                         : QString());
    auto entryFor = [](const QString& path) {
        const QByteArray name = QFileInfo(path).fileName().toUtf8();
        return aicore_llm_model_by_filename(name.constData());
    };
    // Metadata comes from the public catalog; stream validation without copying
    // the GGUF or writing original asset receipts.
    auto ensureAsset = [&](const QString& path, int role) {
        const auto* entry = entryFor(path);
        if (entry && entry->role != role) {
            error = QStringLiteral("Wrong LLM asset role: %1").arg(path);
            return false;
        }
        const B::ModelSource source =
                entry ? B::ModelSource{QString::fromUtf8(entry->download_url),
                                       QString::fromLatin1(entry->sha256)}
                      : B::ModelSource{};
        if (!B::ensureModel(cmd, path, source, session.download, &error))
            return false;
        if (entry) {
            if (QFileInfo(path).size() != entry->size_bytes ||
                !ecvAssetIntegrity::verifyNow(path,
                                              {QCryptographicHash::Sha256,
                                               QByteArray(entry->sha256)})) {
                error = QStringLiteral("LLM asset SHA-256/size mismatch: %1")
                                .arg(path);
                return false;
            }
        } else if (!ecvAssetIntegrity::passesCheapChecks(path, 0, true)) {
            error = QStringLiteral("Invalid LLM GGUF: %1").arg(path);
            return false;
        }
        return true;
    };
    if (!ensureAsset(model, AICORE_LLM_ROLE_MODEL)) return cmd.error(error);
    const auto* modelEntry = entryFor(model);
    const QString recommended =
            imagesRequested && modelEntry && modelEntry->mmproj_filename
                    ? QString::fromUtf8(modelEntry->mmproj_filename)
                    : QString();
    const QString projectorToken =
            args.has("MMPROJ") ? args.str("MMPROJ") : recommended;
    const QString projector =
            projectorToken.isEmpty()
                    ? QString()
                    : B::resolveModelPath(projectorToken, cache);
    if (imagesRequested && projector.isEmpty())
        return cmd.error(
                "Selected LLM model does not accept images without MMPROJ");
    if (!projector.isEmpty() && !ensureAsset(projector, AICORE_LLM_ROLE_MMPROJ))
        return cmd.error(error);

    auto deviceLock = ecvAICoreRuntime::makeDeviceTaskLock(session.device);
    if (!deviceLock.isLocked())
        return cmd.error("Cannot acquire LLM device task lock");
    auto config = aicore_llm_default_options();
    const QByteArray modelBytes = model.toUtf8(),
                     projectorBytes = projector.toUtf8();
    const QByteArray device = session.device.toUtf8();
    config.model_path = modelBytes.constData();
    config.mmproj_path =
            projector.isEmpty() ? nullptr : projectorBytes.constData();
    config.device = device.constData();
    config.threads = session.threads;
    // Retain one context; release results and the context before the runtime
    // guard.
    std::unique_ptr<aicore_llm_ctx, decltype(&aicore_llm_free)> context(
            aicore_llm_create(&config), aicore_llm_free);
    if (!aicore_llm_ready(context.get()))
        return cmd.error(
                QString::fromUtf8(aicore_llm_last_error(context.get())));
    if (imagesRequested && !aicore_llm_supports_vision(context.get()))
        return cmd.error("Selected LLM model does not accept images");
    std::vector<aicore_llm_message> messages;
    if (!system.isEmpty()) messages.push_back({"system", system.constData()});
    messages.push_back({"user", prompt.constData()});
    request.messages = messages.data();
    request.message_count = messages.size();
    B::Manifest manifest(session);
    manifest.set("model", model);
    manifest.set("mmproj", projector);
    manifest.set("resolved_device",
                 QString::fromUtf8(aicore_llm_context_device(context.get())));
    manifest.set("max_tokens", maxTokens);
    manifest.set("temperature", temperature);
    B::runBatch(
            cmd, session, inputs, manifest,
            [&](const B::Item& input, QJsonObject& record, QString& why) {
                // Decode one item at a time; retain one RGB image across the
                // ABI. Conversion may temporarily retain the source plus RGB,
                // not all items.
                QImage image;
                aicore_image_view view{};
                auto current = request;
                if (!input.path.isEmpty()) {
                    image = B::loadImage(input.path, QImage::Format_RGB888,
                                         &why);
                    if (image.isNull()) return false;
                    view = ecvAICoreRuntime::makeImageView(image);
                    current.images = &view;
                    current.image_count = 1;
                    record["input"] = input.path;
                }
                aicore_llm_result* output = nullptr;
                const int rc =
                        aicore_llm_generate(context.get(), &current, &output);
                std::unique_ptr<aicore_llm_result,
                                decltype(&aicore_llm_result_free)>
                        result(output, aicore_llm_result_free);
                if (!result) {
                    why = QString::fromUtf8(
                            aicore_llm_last_error(context.get()));
                    return false;
                }
                if (result->text_bytes >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                    why = "LLM text exceeds Qt export size";
                    return false;
                }
                record["text"] = QString::fromUtf8(
                        result->text, static_cast<int>(result->text_bytes));
                record["finish_reason"] =
                        static_cast<int>(result->finish_reason);
                record["prompt_tokens"] =
                        static_cast<int>(result->prompt_tokens);
                record["generated_tokens"] =
                        static_cast<int>(result->generated_tokens);
                record["prefill_ms"] = result->prefill_ms;
                record["decode_ms"] = result->decode_ms;
                record["timings"] = B::timingsJson(result->timings);
                if (rc != 0 ||
                    result->finish_reason == AICORE_LLM_FINISH_CANCELLED) {
                    why = rc == 1 ? QStringLiteral("LLM generation cancelled")
                                  : QString::fromUtf8(aicore_llm_last_error(
                                            context.get()));
                    return false;
                }
                return true;
            });
    return B::finish(cmd, session, manifest);
#endif
}
