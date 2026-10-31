// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvCommandLineInterface.h>
#include <ecvStdPluginInterface.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPluginLoader>
#include <QSet>
#include <QTemporaryDir>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {
// Unused point-cloud interfaces must not silently take part in image batching;
// inference and file writes are all performed by the real command.
class CommandHost final : public ccCommandLineInterface {
public:
    explicit CommandHost(QString keyword = QStringLiteral("ANYLABEL_INFER"))
        : selectedKeyword(std::move(keyword)) {}
    QString selectedKeyword;
    Command::Shared command;
    QStringList argv;
    mutable QStringList messages, errors;
    bool registerCommand(Command::Shared value) override {
        if (value->m_keyword == selectedKeyword) command = value;
        return true;
    }
    QStringList& arguments() override { return argv; }
    const QStringList& arguments() const override { return argv; }
    void print(const QString& text) const override { messages.append(text); }
    void warning(const QString& text) const override { messages.append(text); }
    bool error(const QString& text) const override {
        errors.append(text);
        return false;
    }
    QString getExportFilename(const CLEntityDesc&,
                              QString,
                              QString,
                              QString*,
                              bool) const override {
        std::abort();
    }
    QString exportEntity(CLEntityDesc&,
                         const QString&,
                         QString*,
                         ExportOptions) override {
        std::abort();
    }
    bool saveClouds(QString, bool, const QString*) override { std::abort(); }
    bool saveMeshes(QString, bool, const QString*) override { std::abort(); }
    void removeClouds(bool) override { std::abort(); }
    void removeMeshes(bool) override { std::abort(); }
    bool importFile(QString,
                    const GlobalShiftOptions&,
                    FileIOFilter::Shared) override {
        std::abort();
    }
    QString cloudExportFormat() const override { std::abort(); }
    QString cloudExportExt() const override { std::abort(); }
    QString meshExportFormat() const override { std::abort(); }
    QString meshExportExt() const override { std::abort(); }
    QString hierarchyExportFormat() const override { std::abort(); }
    QString hierarchyExportExt() const override { std::abort(); }
    void setCloudExportFormat(QString, QString) override { std::abort(); }
    void setMeshExportFormat(QString, QString) override { std::abort(); }
    void setHierarchyExportFormat(QString, QString) override { std::abort(); }
    bool run(const QStringList& values) {
        argv = values;
        messages.clear();
        errors.clear();
        return command->process(*this);
    }
};

bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}
QJsonObject readJson(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
QString sha256(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&file)) return {};
    return QString::fromLatin1(hash.result().toHex());
}
// Missing native commands, annotation misrouting or validation after model
// loading fail this test.
int llmSchema(const QString& pluginPath) {
    QTemporaryDir work;
    if (!check(work.isValid(), "LLM schema temporary directory")) return 1;
    QPluginLoader loader(pluginPath);
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!check(plugin != nullptr, "Load the real LLM consumer plugin")) {
        std::fprintf(stderr, "%s\n", loader.errorString().toUtf8().constData());
        return 1;
    }
    CommandHost host(QStringLiteral("LLM_GENERATE"));
    plugin->registerCommands(&host);
    if (!check(!host.command.isNull(),
               "The real plugin must register LLM_GENERATE"))
        return 1;
    const QString path = work.filePath("llm-schema.json");
    if (!check(host.run({"HELP", "RESULT_JSON", path}),
               "LLM HELP without model or image"))
        return 1;
    const auto schema = readJson(path);
    QSet<QString> keys;
    for (const auto value : schema["options"].toArray())
        keys.insert(value.toObject()["key"].toString());
    if (!check(schema["schema"] == "acloudviewer.aicore.batch/1" &&
                       schema["task"] == "llm" && keys.contains("PROMPT") &&
                       keys.contains("SYSTEM") && keys.contains("MMPROJ") &&
                       keys.contains("MAX_TOKENS") &&
                       keys.contains("TEMPERATURE") && keys.contains("MODEL") &&
                       keys.contains("IMAGE") && !keys.contains("TASK"),
               "LLM native schema must describe generation, not annotation"))
        return 1;
    const QString missing = work.filePath("absent.gguf");
    const QString out = work.filePath("out");
    if (!check(!host.run({"MODEL", missing, "OUT_DIR", out}) &&
                       host.errors.join(' ').contains("PROMPT"),
               "Missing PROMPT must fail before model loading"))
        return 1;
    for (const QString& value :
         {QStringLiteral("0"), QStringLiteral("-1"), QStringLiteral("abc")}) {
        if (!check(!host.run({"PROMPT", "Paris?", "MODEL", missing, "OUT_DIR",
                              out, "MAX_TOKENS", value}) &&
                           host.errors.join(' ').contains("MAX_TOKENS"),
                   "Invalid MAX_TOKENS must fail before model loading"))
            return 1;
    }
    if (!check(!host.run({"PROMPT", "Paris?", "MODEL", missing, "OUT_DIR", out,
                          "TEMPERATURE", "-1"}) &&
                       host.errors.join(' ').contains("TEMPERATURE"),
               "Negative TEMPERATURE must fail before model loading"))
        return 1;
    if (!check(!host.run({"PROMPT", "Paris?", "MODEL", missing, "OUT_DIR",
                          out}) &&
                       host.errors.join(' ').contains(missing) &&
                       !host.errors.join(' ').contains("No inputs"),
               "Text input must reach model validation without requiring "
               "IMAGE"))
        return 1;
    std::fprintf(stdout,
                 "LLM native registration/schema/preload-validation PASS; no "
                 "model inference\n");
    return 0;
}

// Use real pinned assets; missing assets fail rather than skip or substitute
// synthetic generation.
int llmModels(const QString& pluginPath, const QString& device) {
    QTemporaryDir work;
    if (!check(work.isValid(), "LLM model temporary directory")) return 1;
    const QString textModel =
            QString::fromLocal8Bit(qgetenv("AICORE_TEST_LLM_TEXT_MODEL"));
    const QString visionModel =
            QString::fromLocal8Bit(qgetenv("AICORE_TEST_LLM_VISION_MODEL"));
    const QString projector =
            QString::fromLocal8Bit(qgetenv("AICORE_TEST_LLM_MMPROJ"));
    QJsonObject digests;
    for (const QString& path :
         {textModel, visionModel, projector, pluginPath}) {
        const auto digest = sha256(path);
        if (!check(!digest.isEmpty(),
                   "Every real LLM asset and plugin must exist"))
            return 1;
        digests[path] = digest;
    }
    QPluginLoader loader(pluginPath);
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!check(plugin != nullptr, "Load real LLM batch plugin")) return 1;
    CommandHost host(QStringLiteral("LLM_GENERATE"));
    plugin->registerCommands(&host);
    if (!check(!host.command.isNull(), "Register real LLM generation command"))
        return 1;
    for (int run = 0; run < 2; ++run) {
        const QString out = work.filePath(QStringLiteral("text-%1").arg(run));
        if (!check(host.run({"PROMPT",
                             "What is the capital of France? Answer in one "
                             "word.",
                             "MODEL", textModel, "DEVICE", device, "THREADS",
                             "4", "MAX_TOKENS", "32", "TEMPERATURE", "0",
                             "OUT_DIR", out}),
                   "Real text generation without IMAGE")) {
            std::fprintf(stderr, "%s\n",
                         host.errors.join('\n').toUtf8().constData());
            return 1;
        }
        const auto manifest = readJson(out + "/_llm_result.json");
        const auto items = manifest["items"].toArray();
        const auto item = items.isEmpty() ? QJsonObject{} : items[0].toObject();
        if (!check(manifest["schema"] == "acloudviewer.aicore.batch/1" &&
                           manifest["task"] == "llm" &&
                           manifest["command"] == "LLM_GENERATE" &&
                           manifest["ok"] == 1 && manifest["failed"] == 0 &&
                           manifest["total"] == 1 && item["status"] == "ok" &&
                           item["text"].toString().contains("Paris") &&
                           item["generated_tokens"].toInt() > 0 &&
                           item["generated_tokens"].toInt() <= 32 &&
                           item["prompt_tokens"].toInt() > 0 &&
                           (item["finish_reason"].toInt() == 1 ||
                            item["finish_reason"].toInt() == 2) &&
                           item["timings"].isObject(),
                   "Real text batch exposes typed text/tokens/finish/timing on "
                   "repeated commands"))
            return 1;
    }
    QImage red(65, 48, QImage::Format_RGB888);
    red.fill(Qt::red);
    const QString first = work.filePath("red1.png"),
                  second = work.filePath("red2.png");
    if (!check(red.save(first) && red.save(second),
               "Save row-padded RGB image fixtures"))
        return 1;
    const QString out = work.filePath("vision");
    if (!check(host.run({"PROMPT",
                         "What color is this image? Answer in one word.",
                         "MODEL",
                         visionModel,
                         "MMPROJ",
                         projector,
                         "DEVICE",
                         device,
                         "THREADS",
                         "4",
                         "MAX_TOKENS",
                         "32",
                         "TEMPERATURE",
                         "0",
                         "IMAGE",
                         first,
                         "IMAGE",
                         second,
                         "OUT_DIR",
                         out}),
               "Real serial VLM batch with two images")) {
        std::fprintf(stderr, "%s\n",
                     host.errors.join('\n').toUtf8().constData());
        return 1;
    }
    const auto vision = readJson(out + "/_llm_result.json");
    if (!check(vision["ok"] == 2 && vision["failed"] == 0 &&
                       vision["total"] == 2,
               "One manifest row per input image"))
        return 1;
    for (const auto value : vision["items"].toArray()) {
        const auto item = value.toObject();
        if (!check(item["status"] == "ok" &&
                           item["text"].toString().contains(
                                   "red", Qt::CaseInsensitive),
                   "Both decoded RGB inputs reach the real VLM"))
            return 1;
    }
    if (!check(!host.run({"PROMPT", "Describe.", "MODEL", textModel, "IMAGE",
                          first, "OUT_DIR", work.filePath("text-image"),
                          "DEVICE", device}) &&
                       host.errors.join(' ').contains("image",
                                                      Qt::CaseInsensitive),
               "Text-only model rejects an image rather than silently ignoring "
               "it"))
        return 1;
    QFile broken(work.filePath("broken.png"));
    if (!check(broken.open(QIODevice::WriteOnly), "Stage corrupt image"))
        return 1;
    broken.write("invalid image");
    broken.close();
    const QString failedOut = work.filePath("failed-image");
    if (!check(!host.run({"PROMPT", "Describe.", "MODEL", visionModel, "MMPROJ",
                          projector, "IMAGE", broken.fileName(), "IMAGE", first,
                          "OUT_DIR", failedOut, "DEVICE", device,
                          "CONTINUE_ON_ERROR", "0"}),
               "Corrupt image fails the native batch"))
        return 1;
    const auto failed = readJson(failedOut + "/_llm_result.json");
    if (!check(failed["failed"] == 1 && failed["ok"] == 0 &&
                       failed["total"] == 1,
               "Corrupt image is recorded and the remaining image is not "
               "processed"))
        return 1;
    for (auto it = digests.constBegin(); it != digests.constEnd(); ++it)
        if (!check(sha256(it.key()) == it.value().toString(),
                   "Original LLM assets/plugin unchanged"))
            return 1;
    std::fprintf(stdout,
                 "LLM %s real text/repeat/two-image/rejection/error batch "
                 "PASS\n%s\n",
                 device.toUtf8().constData(),
                 QJsonDocument(QJsonObject{{"inputs_sha256", digests},
                                           {"device", device},
                                           {"release_qualified", false}})
                         .toJson(QJsonDocument::Compact)
                         .constData());
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc == 3 && QString::fromLocal8Bit(argv[2]) == "--llm-schema")
        return llmSchema(QString::fromLocal8Bit(argv[1]));
    if ((argc == 3 || argc == 4) &&
        QString::fromLocal8Bit(argv[2]) == "--llm-model")
        return llmModels(QString::fromLocal8Bit(argv[1]),
                         argc == 4 ? QString::fromLocal8Bit(argv[3]) : "cpu");
    if (argc != 2 && argc != 3) return 2;
    const QString device = argc == 3 ? QString::fromLocal8Bit(argv[2]) : "cpu";
    QTemporaryDir work;
    if (!check(work.isValid(), "Temporary batch directory")) return 1;
    const QString root = work.path();
    const QString dataRoot =
            QString::fromLocal8Bit(qgetenv("CLOUDVIEWER_DATA_ROOT"));
    const QString imageSource =
            QString::fromLocal8Bit(qgetenv("AICORE_TEST_ATTRIBUTE_IMAGE"));
    QJsonObject inputDigests;
    inputDigests[imageSource] = sha256(imageSource);
    if (!check(!inputDigests[imageSource].toString().isEmpty(),
               "Real input is missing"))
        return 1;
    const QString image = work.filePath("bus.jpg"),
                  second = work.filePath("bus2.jpg");
    if (!check(QFile::copy(imageSource, image) &&
                       QFile::copy(imageSource, second),
               "Copy the real image"))
        return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!check(plugin != nullptr, "Load the real plugin")) {
        std::fprintf(stderr, "%s\n", loader.errorString().toUtf8().constData());
        return 1;
    }
    CommandHost host;
    plugin->registerCommands(&host);
    if (!check(!host.command.isNull(),
               "The real plugin must register ANYLABEL_INFER"))
        return 1;
    const QString schemaPath = work.filePath("schema.json");
    if (!check(host.run({"HELP", "RESULT_JSON", schemaPath}),
               "Native HELP must succeed"))
        return 1;
    const auto schema = readJson(schemaPath);
    bool attributeChoice = false;
    for (const auto option : schema["options"].toArray()) {
        const auto row = option.toObject();
        if (row["key"] == "TASK")
            attributeChoice =
                    row["type"].toString().split('|').contains("attribute");
    }
    if (!check(schema["schema"] == "acloudviewer.aicore.batch/1" &&
                       schema["task"] == "anylabel" && attributeChoice,
               "The native schema must expose the attribute TASK and keep the "
               "anylabel identity"))
        return 1;
    const QString list = work.filePath("models.json");
    if (!check(host.run({"TASK", "attribute", "LIST_MODELS", "1", "RESULT_JSON",
                         list}),
               "The local recipe list must be queryable"))
        return 1;
    const auto recipes = readJson(list)["models"].toArray();
    if (!check(recipes.size() == 2,
               "Exactly the two owning local F32 recipes must be listed"))
        return 1;
    for (const auto value : recipes) {
        const auto row = value.toObject();
        if (!check(row["url"].toString().isEmpty() && !row["cached"].toBool() &&
                           row["display_name"].toString().contains("local") &&
                           row["path"].toString().startsWith(
                                   root + "/extract/attribute_models/"),
                   "A local recipe must not masquerade as a download catalog"))
            return 1;
    }
    const QString missing =
            root + "/extract/attribute_models/pulc-person-f32.gguf";
    if (!check(!host.run({"TASK", "attribute", "IMAGE", image, "OUT_DIR",
                          work.filePath("missing"), "DOWNLOAD", "1"}) &&
                       host.errors.join(' ').contains(missing) &&
                       !host.messages.join(' ').contains("Downloading") &&
                       !QFileInfo::exists(missing),
               "DOWNLOAD must not invent assets for a missing local recipe"))
        return 1;
    QDir().mkpath(root + "/extract/attribute_models");
    for (const QString& family :
         {QStringLiteral("person"), QStringLiteral("vehicle")}) {
        const QByteArray key =
                "AICORE_TEST_ATTRIBUTE_" + family.toUpper().toUtf8() + "_GGUF";
        const QString source = QString::fromLocal8Bit(qgetenv(key.constData()));
        inputDigests[source] = sha256(source);
        const QString filename = "pulc-" + family + "-f32.gguf";
        if (!check(!inputDigests[source].toString().isEmpty() &&
                           QFile::link(source,
                                       root + "/extract/attribute_models/" +
                                               filename),
                   "Stage the real model"))
            return 1;
        QJsonObject previous;
        for (const QString& confidence :
             {QStringLiteral("0"), QStringLiteral("1")}) {
            const QString out = work.filePath(family + confidence);
            if (!check(host.run({"TASK", "attribute", "MODEL", filename,
                                 "DEVICE", device, "THREADS", "4", "CONFIDENCE",
                                 confidence, "IMAGE", image, "IMAGE", second,
                                 "OUT_DIR", out}),
                       "Native real attribute batch run")) {
                std::fprintf(stderr, "%s\n",
                             host.errors.join('\n').toUtf8().constData());
                return 1;
            }
            const auto result = readJson(out + "/_anylabel_result.json");
            if (!check(result["schema"] == "acloudviewer.aicore.batch/1" &&
                               result["task_name"] == "attribute" &&
                               result["ok"] == 2 && result["failed"] == 0 &&
                               result["total"] == 2,
                       "The native manifest keeps the per-image success state"))
                return 1;
            for (const QString& stem :
                 {QStringLiteral("bus"), QStringLiteral("bus2")}) {
                const auto doc = readJson(out + '/' + stem + ".json");
                const auto shapes = doc["shapes"].toArray();
                const auto shape =
                        shapes.isEmpty() ? QJsonObject{} : shapes[0].toObject();
                const auto attrs = shape["attributes"].toObject();
                if (!check(doc["task"] == "attribute" &&
                                   doc["device"].toString().contains(
                                           device, Qt::CaseInsensitive) &&
                                   shapes.size() == 1 &&
                                   shape["label"] == family &&
                                   shape["shape_type"] == "rectangle" &&
                                   doc["flags"].toObject().isEmpty() &&
                                   attrs.size() ==
                                           (family == "person" ? 21 : 2) &&
                                   (family == "person"
                                            ? attrs["Gender"] == "Male" &&
                                                      attrs["Direction"] ==
                                                              "Back"
                                            : attrs["Color"] == "white" &&
                                                      attrs["Type"] == "truck"),
                           "The batch run saves the complete original "
                           "attributes and must not degrade to classification "
                           "flags"))
                    return 1;
                if (!previous.isEmpty() &&
                    !check(attrs == previous,
                           "Detection confidence must not change the task "
                           "attribute threshold"))
                    return 1;
                previous = attrs;
            }
        }
    }
    const QString broken = work.filePath("broken.jpg");
    QFile invalid(broken);
    if (!check(invalid.open(QIODevice::WriteOnly), "Stage the invalid input"))
        return 1;
    invalid.write("invalid image");
    invalid.close();
    const QString errorOut = work.filePath("error");
    if (!check(!host.run({"TASK", "attribute", "DEVICE", device, "IMAGE",
                          broken, "IMAGE", image, "OUT_DIR", errorOut,
                          "CONTINUE_ON_ERROR", "0"}),
               "An invalid input must make the command fail"))
        return 1;
    const auto failed = readJson(errorOut + "/_anylabel_result.json");
    if (!check(failed["failed"] == 1 && failed["ok"] == 0 &&
                       failed["total"] == 1,
               "The error must be written to the manifest and stop the "
               "remaining images per the option"))
        return 1;
    for (auto it = inputDigests.begin(); it != inputDigests.end(); ++it)
        if (!check(sha256(it.key()) == it.value().toString(),
                   "Inputs/models must not be modified"))
            return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", dataRoot.toUtf8());
    std::fprintf(stdout,
                 "attribute %s native batch: "
                 "schema/local-only/missing/two-model/two-image/threshold/"
                 "error PASS\n",
                 device.toUtf8().constData());
    std::fprintf(
            stdout, "%s\n",
            QJsonDocument(QJsonObject{{"inputs_sha256", inputDigests},
                                      {"plugin_sha256",
                                       sha256(QString::fromLocal8Bit(argv[1]))},
                                      {"device", device},
                                      {"release_qualified", false}})
                    .toJson(QJsonDocument::Compact)
                    .constData());
    return 0;
}
