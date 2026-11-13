// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QSet>
#include <QSettings>
#include <QVector>
#include <optional>
#include <tuple>

namespace AnnotationChatConfig {
// Process-local call tracking only; models.json remains the model-list owner.
// Match provider.get_models_data + EventTracker, including reset(timer=0)
// after the first 300-second interval in the pinned source.
struct ModelRefreshTracker {
    struct Call {
        int count = 0;
        double first = 0;
    };
    QHash<QString, Call> providers;
    bool shouldFetch(const QString& provider, bool cached, double now) {
        auto it = providers.find(provider);
        if (it == providers.end()) it = providers.insert(provider, {0, now});
        ++it->count;
        if (now - it->first > 300) {
            it->count = 1;
            it->first = 0;
        }
        return !(it->count > 1 && provider.toLower() != "ollama" && cached);
    }
};

// difflib.SequenceMatcher(None, query, name).ratio(), including automatic
// popular-character exclusion. Worst-case O(n*m): preserve greedy matching
// blocks and tie order; edit-distance/LCS scores change popup search results.
inline double modelNameSimilarity(const QString& query, const QString& name) {
    const auto a = query.toUcs4(), b = name.toUcs4();
    if (a.isEmpty() && b.isEmpty()) return 1;
    QHash<uint, QVector<int>> positions;
    for (int j = 0; j < b.size(); ++j) positions[b[j]].append(j);
    if (b.size() >= 200)
        for (auto it = positions.begin(); it != positions.end();)
            if (it.value().size() > b.size() / 100 + 1)
                it = positions.erase(it);
            else
                ++it;
    QVector<std::tuple<int, int, int, int>> pending{
            std::make_tuple(0, a.size(), 0, b.size())};
    int matched = 0;
    while (!pending.isEmpty()) {
        const auto range = pending.takeLast();
        const int alo = std::get<0>(range), ahi = std::get<1>(range),
                  blo = std::get<2>(range), bhi = std::get<3>(range);
        int bestA = alo, bestB = blo, size = 0;
        QVector<int> previous(b.size() + 1), current(b.size() + 1);
        for (int i = alo; i < ahi; ++i) {
            current.fill(0);
            for (int j : positions.value(a[i])) {
                if (j < blo) continue;
                if (j >= bhi) break;
                const int length = previous[j] + 1;
                current[j + 1] = length;
                if (length > size) {
                    bestA = i - length + 1;
                    bestB = j - length + 1;
                    size = length;
                }
            }
            previous.swap(current);
        }
        while (bestA > alo && bestB > blo && a[bestA - 1] == b[bestB - 1]) {
            --bestA;
            --bestB;
            ++size;
        }
        while (bestA + size < ahi && bestB + size < bhi &&
               a[bestA + size] == b[bestB + size])
            ++size;
        if (!size) continue;
        matched += size;
        if (alo < bestA && blo < bestB)
            pending.append(std::make_tuple(alo, bestA, blo, bestB));
        if (bestA + size < ahi && bestB + size < bhi)
            pending.append(
                    std::make_tuple(bestA + size, ahi, bestB + size, bhi));
    }
    return 2.0 * matched / (a.size() + b.size());
}

inline bool modelNameMatches(const QString& query, const QString& name) {
    const auto search = query.toLower(), candidate = name.toLower();
    return search.isEmpty() || candidate.contains(search) ||
           modelNameSimilarity(search, candidate) >= 0.4;
}

inline QString path(const QString& filename) {
    const auto work =
            QSettings()
                    .value("qAnyLabeling/workDirectory", QDir::homePath())
                    .toString();
    return QDir(work).filePath("xanylabeling_data/chatbot/" + filename);
}
inline QJsonObject defaultProviders() {
    static const auto value = QJsonDocument::fromJson(R"config({
  "custom": {
    "api_address": "",
    "api_key": "",
    "model_name": "",
    "api_key_url": null,
    "api_docs_url": null,
    "model_docs_url": null
  },
  "anthropic": {
    "api_address": "https://api.anthropic.com/v1/",
    "api_key": null,
    "api_key_url": "https://console.anthropic.com/settings/keys",
    "api_docs_url": "https://docs.anthropic.com/en/docs",
    "model_docs_url": "https://docs.anthropic.com/en/docs/about-claude/models/all-models"
  },
  "deepseek": {
    "api_address": "https://api.deepseek.com/v1",
    "api_key": null,
    "api_key_url": "https://platform.deepseek.com/api_keys",
    "api_docs_url": "https://platform.deepseek.com/docs",
    "model_docs_url": "https://platform.deepseek.com/models"
  },
  "google": {
    "api_address": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key": null,
    "api_key_url": "https://aistudio.google.com/app/apikey",
    "api_docs_url": "https://ai.google.dev/gemini-api/docs",
    "model_docs_url": "https://ai.google.dev/gemini-api/docs/models"
  },
  "local": {
    "api_address": "aicore://llm",
    "api_key": "",
    "api_key_url": null,
    "api_docs_url": null,
    "model_docs_url": null
  },
  "ollama": {
    "api_address": "http://localhost:11434/v1",
    "api_key": "ollama",
    "api_key_url": null,
    "api_docs_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
    "model_docs_url": "https://ollama.com/search"
  },
  "openai": {
    "api_address": "https://api.openai.com/v1",
    "api_key": null,
    "api_key_url": "https://platform.openai.com/api-keys",
    "api_docs_url": "https://platform.openai.com/docs",
    "model_docs_url": "https://platform.openai.com/docs/models"
  },
  "openrouter": {
    "api_address": "https://openrouter.ai/api/v1",
    "api_key": null,
    "api_key_url": "https://openrouter.ai/settings/keys",
    "api_docs_url": "https://openrouter.ai/docs/quick-start",
    "model_docs_url": "https://openrouter.ai/models"
  },
  "qwen": {
    "api_address": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": null,
    "api_key_url": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
    "api_docs_url": "https://help.aliyun.com/document_detail/2590237.html",
    "model_docs_url": "https://help.aliyun.com/zh/model-studio/developer-reference/what-is-qwen-llm"
  }
})config")
                                      .object();
    return value;
}
inline QJsonObject defaultModels() {
    static const auto value = QJsonDocument::fromJson(R"config({
  "settings": {
    "provider": "ollama",
    "model_id": null,
    "temperature": 10,
    "max_length": null,
    "system_prompt": null
  },
  "models_data": {},
  "supported_vision_models": [
    "claude-3",
    "claude-haiku-4",
    "claude-opus-4",
    "claude-sonnet-4",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-3.1-flash",
    "gemini-3.1-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "bakllava",
    "gemma3",
    "granite3.2-vision",
    "llama3.2-vision",
    "llama4",
    "llava",
    "minicpm-v",
    "moondream",
    "qwen2.5-vl",
    "qwen2.5vl",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "o1",
    "o3",
    "o4-mini",
    "qvq",
    "qwen-vl",
    "qwen3-omni",
    "qwen3-vl",
    "qwen3.5-",
    "qwen3.6-",
    "ernie-4.5-vl",
    "glm-4.5v",
    "glm-4.6v",
    "glm-5v",
    "grok-4",
    "kimi-k2",
    "llama-3.2-11b-vision",
    "llama-4-maverick",
    "llama-4-scout",
    "mistral-medium-3",
    "mistral-small-3.1",
    "mistral-small-3.2",
    "nemotron-nano-12b-v2-vl",
    "nova-2-lite",
    "nova-lite",
    "nova-premier",
    "nova-pro",
    "pixtral",
    "qianfan-ocr",
    "seed-1.6",
    "seed-2.0",
    "ui-tars"
  ]
})config")
                                      .object();
    return value;
}
// Preserve the source precedence: explicit metadata, architecture, modalities,
// configured vision-name substrings, then fallback names. Explicit false wins.
inline std::optional<bool> metadataVision(const QJsonObject& metadata) {
    for (const auto& key : {"vision", "supports_vision", "multimodal"})
        if (metadata.value(key).isBool()) return metadata.value(key).toBool();
    const auto capabilities = metadata.value("capabilities").toObject();
    for (const auto& key : {"vision", "image", "images", "video", "multimodal"})
        if (capabilities.value(key).isBool())
            return capabilities.value(key).toBool();
    if (metadata.value("architecture").isObject()) {
        const auto result =
                metadataVision(metadata.value("architecture").toObject());
        if (result.has_value()) return result;
    }
    const QSet<QString> tokens{"image", "images", "vision", "visual", "video"};
    auto modalities = [&](const QJsonValue& value) -> std::optional<bool> {
        if (value.isUndefined() || value.isNull()) return {};
        if (value.isString())
            return tokens.contains(value.toString().toLower());
        if (value.isArray()) {
            for (const auto& item : value.toArray())
                if (item.isString() &&
                    tokens.contains(item.toString().toLower()))
                    return true;
            return false;
        }
        if (value.isObject()) {
            const auto object = value.toObject();
            for (auto it = object.begin(); it != object.end(); ++it)
                if (tokens.contains(it.key().toLower())) return true;
            return false;
        }
        return {};
    };
    for (const auto& key : {"input_modalities", "input_types", "inputs"}) {
        const auto result = modalities(metadata.value(key));
        if (result.has_value()) return result;
    }
    const auto value = metadata.value("modalities");
    if (value.isObject()) {
        const auto object = value.toObject();
        for (const auto& key : {"input", "inputs", "input_modalities"}) {
            const auto result = modalities(object.value(key));
            if (result.has_value()) return result;
        }
    } else {
        const auto result = modalities(value);
        if (result.has_value()) return result;
    }
    const auto modality = metadata.value("modality").toString().toLower();
    if (modality.contains("->")) {
        const auto input = modality.section("->", 0, 0);
        for (const auto& token : tokens)
            if (input.contains(token)) return true;
        return false;
    }
    return {};
}
inline bool supportsVision(const QString& id,
                           const QJsonObject& metadata,
                           const QJsonArray& configured) {
    const auto explicitResult = metadataVision(metadata);
    if (explicitResult.has_value()) return *explicitResult;
    const auto name = id.toLower();
    for (const auto& value : configured)
        if (name.contains(value.toString().toLower())) return true;
    for (const auto& token :
         {"asr", "audio-preview", "dall-e", "embedding", "gpt-image",
          "image-edit", "image-generation", "rerank", "speech",
          "text-embedding", "tts", "whisper"})
        if (name.contains(token)) return false;
    for (const auto& token : {"bakllava",
                              "claude-3",
                              "claude-haiku-4",
                              "claude-opus-4",
                              "claude-sonnet-4",
                              "gemini",
                              "gemma3",
                              "gpt-4.1",
                              "gpt-4o",
                              "granite-vision",
                              "llama3.2-vision",
                              "llava",
                              "minicpm-v",
                              "mistral-small-3.2",
                              "mistral-medium-3.1",
                              "mistral-medium-2508",
                              "mistral-small-2506",
                              "mistral-small-2503",
                              "mistral-small-latest",
                              "mistral-medium-latest",
                              "mistral-small",
                              "moondream",
                              "multimodal",
                              "nemotron-nano-12b-v2-vl",
                              "pixtral",
                              "qwen-vl",
                              "qwen2-vl",
                              "qwen2.5-vl",
                              "qwen3-vl",
                              "vision"})
        if (name.contains(token)) return true;
    return false;
}
inline QJsonObject read(const QString& filename, QString* error) {
    QFile file(path(filename));
    if (!file.open(QIODevice::ReadOnly)) {
        *error = file.errorString() + ": " + file.fileName();
        return {};
    }
    QJsonParseError parse;
    const auto json = QJsonDocument::fromJson(file.readAll(), &parse);
    const auto object = json.object();
    bool valid = parse.error == QJsonParseError::NoError && json.isObject();
    if (filename == "models.json") {
        const auto settings = object.value("settings").toObject();
        valid = valid && object.value("settings").isObject() &&
                object.value("models_data").isObject() &&
                settings.value("provider").isString() &&
                (settings.value("model_id").isString() ||
                 settings.value("model_id").isNull()) &&
                settings.value("temperature").isDouble() &&
                (settings.value("max_length").isDouble() ||
                 settings.value("max_length").isNull());
        const auto providers = object.value("models_data").toObject();
        for (auto p = providers.begin(); valid && p != providers.end(); ++p) {
            valid = p.value().isObject();
            const auto models = p.value().toObject();
            for (auto m = models.begin(); valid && m != models.end(); ++m)
                valid = m.value().isObject();
        }
    } else {
        for (auto it = object.begin(); valid && it != object.end(); ++it) {
            const auto entry = it.value().toObject();
            valid = it.value().isObject() &&
                    entry.value("api_address").isString() &&
                    (entry.value("api_key").isString() ||
                     entry.value("api_key").isNull());
        }
    }
    if (!valid) {
        *error = "Invalid chatbot configuration; original file preserved: " +
                 file.fileName();
        return {};
    }
    return object;
}
inline bool write(const QString& filename,
                  const QJsonObject& object,
                  QString* error) {
    const auto target = path(filename);
    if (!QDir().mkpath(QFileInfo(target).absolutePath())) {
        *error = "Cannot create chatbot configuration directory: " + target;
        return false;
    }
    QSaveFile file(target);
    const auto bytes = QJsonDocument(object).toJson();
    if (!file.open(QIODevice::WriteOnly) ||
        !file.setPermissions(QFileDevice::ReadOwner |
                             QFileDevice::WriteOwner) ||
        file.write(bytes) != bytes.size() || !file.commit()) {
        *error = "Cannot save chatbot configuration: " + target + ": " +
                 file.errorString();
        return false;
    }
    return true;
}
// QSettings is a one-time migration input. Existing upstream JSON wins.
inline bool initialize(QString* error) {
    QSettings legacy;
    if (!QFileInfo::exists(path("providers.json"))) {
        auto providers = defaultProviders();
        for (auto it = providers.begin(); it != providers.end(); ++it) {
            auto entry = it.value().toObject();
            const QString prefix =
                    "qAnyLabeling/chat/providers/" + it.key() + '/';
            if (legacy.contains(prefix + "address"))
                entry.insert("api_address",
                             legacy.value(prefix + "address").toString());
            if (legacy.contains(prefix + "key"))
                entry.insert("api_key",
                             legacy.value(prefix + "key").toString());
            if (it.key() == "custom" && legacy.contains(prefix + "model"))
                entry.insert("model_name",
                             legacy.value(prefix + "model").toString());
            it.value() = entry;
        }
        if (!write("providers.json", providers, error)) return false;
    }
    if (!QFileInfo::exists(path("models.json"))) {
        auto document = defaultModels();
        auto settings = document.value("settings").toObject();
        const auto provider =
                legacy.value("qAnyLabeling/chat/provider", "ollama").toString();
        settings.insert("provider", provider);
        const auto model = legacy.value("qAnyLabeling/chat/providers/" +
                                        provider + "/model")
                                   .toString();
        if (!model.isEmpty()) settings.insert("model_id", model);
        if (legacy.contains("qAnyLabeling/chat/temperature"))
            settings.insert(
                    "temperature",
                    qRound(10 * legacy.value("qAnyLabeling/chat/temperature")
                                        .toDouble()));
        const auto max = legacy.value("qAnyLabeling/chat/maxTokens").toInt();
        if (max > 0) settings.insert("max_length", max);
        const auto prompt =
                legacy.value("qAnyLabeling/chat/systemPrompt").toString();
        if (!prompt.isEmpty()) settings.insert("system_prompt", prompt);
        document.insert("settings", settings);
        if (!model.isEmpty())
            document.insert(
                    "models_data",
                    QJsonObject{
                            {provider,
                             QJsonObject{
                                     {model,
                                      QJsonObject{
                                              {"vision",
                                               supportsVision(
                                                       model, {},
                                                       document.value("supporte"
                                                                      "d_"
                                                                      "vision_"
                                                                      "models")
                                                               .toArray())},
                                              {"selected", true},
                                              {"favorite", false}}}}}});
        if (!write("models.json", document, error)) return false;
    }
    return true;
}
}  // namespace AnnotationChatConfig
