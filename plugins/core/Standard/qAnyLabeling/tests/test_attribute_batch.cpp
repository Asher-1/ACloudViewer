// SPDX-License-Identifier: MIT
// 真实插件注册的 ANYLABEL_INFER：测试宿主只提供参数/日志，不替换推理、模型或导出。
#include <ecvCommandLineInterface.h>
#include <ecvStdPluginInterface.h>

#include <QApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPluginLoader>
#include <QTemporaryDir>
#include <cstdio>
#include <cstdlib>

namespace {
// 未使用的点云接口不能悄悄参与图像批处理；推理与文件写入均由真实 command 执行。
class CommandHost final : public ccCommandLineInterface {
public:
    Command::Shared command;
    QStringList argv;
    mutable QStringList messages, errors;
    bool registerCommand(Command::Shared value) override {
        if (value->m_keyword == "ANYLABEL_INFER") command = value;
        return true;
    }
    QStringList& arguments() override { return argv; }
    const QStringList& arguments() const override { return argv; }
    void print(const QString& text) const override { messages.append(text); }
    void warning(const QString& text) const override { messages.append(text); }
    bool error(const QString& text) const override { errors.append(text); return false; }
    QString getExportFilename(const CLEntityDesc&, QString, QString, QString*, bool) const override { std::abort(); }
    QString exportEntity(CLEntityDesc&, const QString&, QString*, ExportOptions) override { std::abort(); }
    bool saveClouds(QString, bool, const QString*) override { std::abort(); }
    bool saveMeshes(QString, bool, const QString*) override { std::abort(); }
    void removeClouds(bool) override { std::abort(); }
    void removeMeshes(bool) override { std::abort(); }
    bool importFile(QString, const GlobalShiftOptions&, FileIOFilter::Shared) override { std::abort(); }
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
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (argc != 2 && argc != 3) return 2;
    const QString device = argc == 3 ? QString::fromLocal8Bit(argv[2]) : "cpu";
    QTemporaryDir work;
    if (!check(work.isValid(), "临时批处理目录")) return 1;
    const QString root = work.path();
    const QString dataRoot = QString::fromLocal8Bit(qgetenv("CLOUDVIEWER_DATA_ROOT"));
    const QString imageSource = QString::fromLocal8Bit(qgetenv("AICORE_TEST_ATTRIBUTE_IMAGE"));
    QJsonObject inputDigests;
    inputDigests[imageSource] = sha256(imageSource);
    if (!check(!inputDigests[imageSource].toString().isEmpty(), "真实输入缺失")) return 1;
    const QString image = work.filePath("bus.jpg"), second = work.filePath("bus2.jpg");
    if (!check(QFile::copy(imageSource, image) && QFile::copy(imageSource, second), "复制真实图片")) return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", root.toUtf8());
    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    if (!check(plugin != nullptr, "加载真实插件")) {
        std::fprintf(stderr, "%s\n", loader.errorString().toUtf8().constData());
        return 1;
    }
    CommandHost host;
    plugin->registerCommands(&host);
    if (!check(!host.command.isNull(), "真实插件必须注册 ANYLABEL_INFER")) return 1;
    const QString schemaPath = work.filePath("schema.json");
    if (!check(host.run({"HELP", "RESULT_JSON", schemaPath}), "原生 HELP 必须成功")) return 1;
    const auto schema = readJson(schemaPath);
    bool attributeChoice = false;
    for (const auto option : schema["options"].toArray()) {
        const auto row = option.toObject();
        if (row["key"] == "TASK")
            attributeChoice = row["type"].toString().split('|').contains("attribute");
    }
    if (!check(schema["schema"] == "acloudviewer.aicore.batch/1" &&
               schema["task"] == "anylabel" && attributeChoice,
               "原生 schema 必须公开 attribute TASK，保持 anylabel 身份")) return 1;
    const QString list = work.filePath("models.json");
    if (!check(host.run({"TASK", "attribute", "LIST_MODELS", "1", "RESULT_JSON", list}),
               "本地 recipe 列表必须可查询")) return 1;
    const auto recipes = readJson(list)["models"].toArray();
    if (!check(recipes.size() == 2, "必须只列两个 owning 本地 F32 recipe")) return 1;
    for (const auto value : recipes) {
        const auto row = value.toObject();
        if (!check(row["url"].toString().isEmpty() && !row["cached"].toBool() &&
                   row["display_name"].toString().contains("local") &&
                   row["path"].toString().startsWith(root + "/extract/attribute_models/"),
                   "本地 recipe 不能冒充下载 catalog")) return 1;
    }
    const QString missing = root + "/extract/attribute_models/pulc-person-f32.gguf";
    if (!check(!host.run({"TASK", "attribute", "IMAGE", image, "OUT_DIR", work.filePath("missing"),
                         "DOWNLOAD", "1"}) && host.errors.join(' ').contains(missing) &&
               !host.messages.join(' ').contains("Downloading") && !QFileInfo::exists(missing),
               "DOWNLOAD 不能为缺失本地 recipe 虚构资产")) return 1;
    QDir().mkpath(root + "/extract/attribute_models");
    for (const QString& family : {QStringLiteral("person"), QStringLiteral("vehicle")}) {
        const QByteArray key = "AICORE_TEST_ATTRIBUTE_" + family.toUpper().toUtf8() + "_GGUF";
        const QString source = QString::fromLocal8Bit(qgetenv(key.constData()));
        inputDigests[source] = sha256(source);
        const QString filename = "pulc-" + family + "-f32.gguf";
        if (!check(!inputDigests[source].toString().isEmpty() &&
                   QFile::link(source, root + "/extract/attribute_models/" + filename), "装配真实模型")) return 1;
        QJsonObject previous;
        for (const QString& confidence : {QStringLiteral("0"), QStringLiteral("1")}) {
            const QString out = work.filePath(family + confidence);
            if (!check(host.run({"TASK", "attribute", "MODEL", filename, "DEVICE", device,
                                 "THREADS", "4", "CONFIDENCE", confidence, "IMAGE", image,
                                 "IMAGE", second, "OUT_DIR", out}), "原生真实属性批处理")) {
                std::fprintf(stderr, "%s\n", host.errors.join('\n').toUtf8().constData());
                return 1;
            }
            const auto result = readJson(out + "/_anylabel_result.json");
            if (!check(result["schema"] == "acloudviewer.aicore.batch/1" && result["task_name"] == "attribute" &&
                       result["ok"] == 2 && result["failed"] == 0 && result["total"] == 2,
                       "原生 manifest 保留逐图成功状态")) return 1;
            for (const QString& stem : {QStringLiteral("bus"), QStringLiteral("bus2")}) {
                const auto doc = readJson(out + '/' + stem + ".json");
                const auto shapes = doc["shapes"].toArray();
                const auto shape = shapes.isEmpty() ? QJsonObject{} : shapes[0].toObject();
                const auto attrs = shape["attributes"].toObject();
                if (!check(doc["task"] == "attribute" && doc["device"].toString().contains(device, Qt::CaseInsensitive) &&
                           shapes.size() == 1 && shape["label"] == family && shape["shape_type"] == "rectangle" &&
                           doc["flags"].toObject().isEmpty() && attrs.size() == (family == "person" ? 21 : 2) &&
                           (family == "person" ? attrs["Gender"] == "Male" && attrs["Direction"] == "Back"
                                               : attrs["Color"] == "white" && attrs["Type"] == "truck"),
                           "批处理保存完整原属性，不能降为分类 flags")) return 1;
                if (!previous.isEmpty() && !check(attrs == previous, "检测 confidence 不得改变 task 属性阈值")) return 1;
                previous = attrs;
            }
        }
    }
    const QString broken = work.filePath("broken.jpg");
    QFile invalid(broken);
    if (!check(invalid.open(QIODevice::WriteOnly), "装配错误输入")) return 1;
    invalid.write("invalid image");
    invalid.close();
    const QString errorOut = work.filePath("error");
    if (!check(!host.run({"TASK", "attribute", "DEVICE", device, "IMAGE", broken, "IMAGE", image,
                         "OUT_DIR", errorOut, "CONTINUE_ON_ERROR", "0"}), "错误输入必须使命令失败")) return 1;
    const auto failed = readJson(errorOut + "/_anylabel_result.json");
    if (!check(failed["failed"] == 1 && failed["ok"] == 0 && failed["total"] == 1,
               "错误必须写入 manifest 且按选项停止后续图")) return 1;
    for (auto it = inputDigests.begin(); it != inputDigests.end(); ++it)
        if (!check(sha256(it.key()) == it.value().toString(), "输入/模型不能被修改")) return 1;
    qputenv("CLOUDVIEWER_DATA_ROOT", dataRoot.toUtf8());
    std::fprintf(stdout, "attribute %s native batch: schema/local-only/missing/two-model/two-image/threshold/error PASS\n",
                 device.toUtf8().constData());
    std::fprintf(stdout, "%s\n", QJsonDocument(QJsonObject{{"inputs_sha256", inputDigests},
        {"plugin_sha256", sha256(QString::fromLocal8Bit(argv[1]))}, {"device", device},
        {"release_qualified", false}}).toJson(QJsonDocument::Compact).constData());
    return 0;
}
