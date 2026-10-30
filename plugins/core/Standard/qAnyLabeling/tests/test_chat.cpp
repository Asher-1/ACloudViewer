// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvStdPluginInterface.h>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QImage>
#include <QJsonDocument>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPluginLoader>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStyle>
#include <QTabWidget>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QTextEdit>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cstdio>
#include <functional>

#include "AnnotationChatConfig.h"
#include "ChatServiceClient.h"
#include "LocalChatEngine.h"

#define REQUIRE(x)                                               \
    do {                                                         \
        if (!(x)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); \
            return 1;                                            \
        }                                                        \
    } while (false)
namespace {
bool waitFor(const std::function<bool()>& predicate) {
    QElapsedTimer time;
    time.start();
    while (!predicate() && time.elapsed() < 5000)
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    return predicate();
}
QJsonObject readJson(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
}  // namespace
int main(int argc, char** argv) {
    QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    QTemporaryDir temp;
    REQUIRE(temp.isValid() && (argc == 2 || argc == 3));
    QCoreApplication::setOrganizationName("ACloudViewerTests");
    QCoreApplication::setApplicationName("qAnyLabelingChat");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temp.path());
    AnnotationChatConfig::ModelRefreshTracker refreshTracker;
    QJsonArray refreshCases;
    const auto refreshCase = [&](const QString& provider, bool cached,
                                 double now, bool expected) {
        const bool fetch = refreshTracker.shouldFetch(provider, cached, now);
        refreshCases.append(QJsonObject{{"provider", provider},
                                        {"cached", cached},
                                        {"time", now},
                                        {"fetch", fetch}});
        return fetch == expected;
    };
    REQUIRE(refreshCase("openai", true, 1000, true) &&
            refreshCase("openai", true, 1001, false) &&
            refreshCase("openai", true, 1300, false) &&
            refreshCase("openai", true, 1300.001, true) &&
            refreshCase("openai", true, 1300.002, true) &&
            refreshCase("openai", true, 1301, true) &&
            refreshCase("ollama", true, 1000, true) &&
            refreshCase("ollama", true, 1001, true) &&
            refreshCase("OLLAMA", true, 1001, true) &&
            refreshCase("anthropic", false, 1000, true) &&
            refreshCase("anthropic", false, 1001, true) &&
            refreshCase("anthropic", true, 1002, false) &&
            refreshCase("qwen", true, 0, true) &&
            refreshCase("qwen", true, 300, false) &&
            refreshCase("qwen", true, 301, true));
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-model-refresh-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly) &&
                fixture.write(QJsonDocument(refreshCases).toJson()) > 0);
    }
    // Local provider: aicore://llm lists the AICore llm catalog (models, not
    // projectors) asynchronously and runs generation on the shared worker.
    {
        ChatServiceClient::Settings local;
        local.baseUrl = QUrl(AnnotationChatConfig::defaultProviders()
                                     .value("local")
                                     .toObject()
                                     .value("api_address")
                                     .toString());
        REQUIRE(LocalChatEngine::isLocal(local.baseUrl));
        ChatServiceClient client;
        QJsonArray models;
        QString error = "unset";
        int lists = 0;
        QObject::connect(&client, &ChatServiceClient::modelsReady,
                         [&](const QJsonArray& value, const QString& why) {
                             models = value;
                             error = why;
                             ++lists;
                         });
        client.fetchModels(local);
        REQUIRE(lists == 0);  // asynchronous like the HTTP transport
        client.cancel();      // a cancelled listing is never delivered
        QCoreApplication::processEvents();
        REQUIRE(lists == 0);
        client.fetchModels(local);
        REQUIRE(waitFor([&]() { return lists == 1; }) && error.isEmpty());
        QStringList ids;
        for (const auto& value : models) {
            const auto object = value.toObject();
            ids << object.value("id").toString();
            REQUIRE(!ids.last().startsWith("mmproj-"));
            REQUIRE(object.value("vision").toBool() ==
                    ids.last().startsWith("SmolVLM"));
        }
        REQUIRE(ids.contains("SmolVLM-500M-Instruct-Q8_0.gguf") &&
                ids.contains("qwen2.5-0.5b-instruct-q8_0.gguf"));

        QString text, finalError, streamed;
        int done = 0;
        QObject::connect(&client, &ChatServiceClient::textReceived,
                         [&](const QString& delta) { streamed += delta; });
        QObject::connect(&client, &ChatServiceClient::finished,
                         [&](const QString& value, const QString& why) {
                             text = value;
                             finalError = why;
                             ++done;
                         });
        local.model = "not-a-catalog-model.gguf";
        client.complete(local, QJsonArray{QJsonObject{{"role", "user"},
                                                      {"content", "hi"}}});
        REQUIRE(done == 1 && finalError.contains("Unknown local model"));

        // Real generation only when the catalog files are already verified
        // in the data root: a unit test never downloads hundreds of MB.
        const auto waitLong = [](const std::function<bool()>& predicate) {
            QElapsedTimer time;
            time.start();
            while (!predicate() && time.elapsed() < 300000)
                QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
            return predicate();
        };
        QVector<LocalChatEngine::Asset> missing;
        local.model = "qwen2.5-0.5b-instruct-q8_0.gguf";
        local.temperature = 0;
        if (LocalChatEngine::missingAssets(local.model, &missing, &error) &&
            missing.isEmpty()) {
            const QJsonArray question{QJsonObject{
                    {"role", "user"},
                    {"content",
                     "What is the capital of France? Answer in one word."}}};
            done = 0;
            streamed.clear();
            client.complete(local, question);
            REQUIRE(client.busy());
            REQUIRE(waitLong([&]() { return done == 1; }));
            REQUIRE(finalError.isEmpty() && text.contains("Paris") &&
                    streamed == text && !client.busy());
            // Cancel after the first fragment: nothing is delivered later,
            // and the shared engine serves the next request normally.
            local.maxTokens = 256;
            done = 0;
            streamed.clear();
            QMetaObject::Connection stop = QObject::connect(
                    &client, &ChatServiceClient::textReceived,
                    [&](const QString&) { client.cancel(); });
            client.complete(local, QJsonArray{QJsonObject{
                                           {"role", "user"},
                                           {"content",
                                            "Write a long story about the "
                                            "sea."}}});
            REQUIRE(waitLong([&]() { return !streamed.isEmpty(); }));
            QObject::disconnect(stop);
            const QString partial = streamed;
            QElapsedTimer settle;
            settle.start();
            while (settle.elapsed() < 1500)
                QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
            REQUIRE(done == 0 && streamed == partial && !client.busy());
            local.maxTokens = 0;
            client.complete(local, question);
            REQUIRE(waitLong([&]() { return done == 1; }) &&
                    finalError.isEmpty() && text.contains("Paris"));
            // Destroying a client mid-request must not deliver into it.
            {
                ChatServiceClient doomed;
                doomed.complete(local, question);
            }
            QElapsedTimer drain;
            drain.start();
            while (drain.elapsed() < 1500)
                QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        } else {
            std::puts("local provider generation: catalog model not cached, "
                      "skipped");
        }
        local.model = "SmolVLM-500M-Instruct-Q8_0.gguf";
        if (LocalChatEngine::missingAssets(local.model, &missing, &error) &&
            missing.isEmpty()) {
            // Image parts arrive as OpenAI data URLs (userMessage) and are
            // decoded for the projector inside the engine.
            QImage image(64, 48, QImage::Format_RGB888);
            image.fill(Qt::red);
            // Kept out of temp.path(): later VQA cases list that folder.
            REQUIRE(QDir(temp.path()).mkpath("local-llm"));
            const QString path = temp.filePath("local-llm/red.png");
            REQUIRE(image.save(path));
            QString imageError;
            const auto message = ChatServiceClient::userMessage(
                    "What color is this image? Answer in one word.", path,
                    &imageError);
            REQUIRE(imageError.isEmpty());
            done = 0;
            client.complete(local, QJsonArray{message});
            REQUIRE(waitLong([&]() { return done == 1; }));
            REQUIRE(finalError.isEmpty() &&
                    text.toLower().contains("red"));
            // A text-only model rejects images with a typed error.
            local.model = "qwen2.5-0.5b-instruct-q8_0.gguf";
            done = 0;
            client.complete(local, QJsonArray{message});
            REQUIRE(waitLong([&]() { return done == 1; }) &&
                    finalError.contains("does not accept images"));
        }
    }
    QTcpServer server;
    REQUIRE(server.listen(QHostAddress::LocalHost));
    QJsonObject received;
    QVector<QJsonObject> receivedRequests;
    QByteArray headers;
    int response = 0, requests = 0, modelRequests = 0;
    QObject::connect(&server, &QTcpServer::newConnection, &server, [&]() {
        auto* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [&, socket]() {
            QByteArray buffer = socket->property("request").toByteArray() +
                                socket->readAll();
            socket->setProperty("request", buffer);
            const int split = buffer.indexOf("\r\n\r\n");
            if (split < 0 || socket->property("answered").toBool()) return;
            int length = 0;
            for (const auto& line : buffer.left(split).split('\n'))
                if (line.toLower().startsWith("content-length:"))
                    length = line.mid(15).trimmed().toInt();
            if (buffer.size() < split + 4 + length) return;
            socket->setProperty("answered", true);
            headers = buffer.left(split);
            if (headers.startsWith("GET")) {
                ++modelRequests;
            } else {
                ++requests;  // Inference count excludes background discovery.
                received =
                        QJsonDocument::fromJson(buffer.mid(split + 4)).object();
                receivedRequests.append(received);
            }
            if (response == 4) return;  // timeout / cancellation.
            if (response == 7) {
                QTimer::singleShot(150, socket, [socket]() {
                    if (socket->state() != QAbstractSocket::ConnectedState)
                        return;
                    const QByteArray body =
                            "{\"choices\":[{\"message\":{\"content\":"
                            "\"Late hidden answer\"}}]}";
                    socket->write(
                            "HTTP/1.1 200 OK\r\nContent-Type: "
                            "application/json\r\nContent-Length: " +
                            QByteArray::number(body.size()) +
                            "\r\nConnection: close\r\n\r\n" + body);
                    socket->disconnectFromHost();
                });
                return;
            }
            QByteArray body;
            if (headers.startsWith("GET"))
                body = "{\"data\":[{\"id\":\"vision-test\"}]}";
            else if (response == 1)
                body = "{\"error\":{\"message\":\"bad credentials\"}}";
            else if (response == 2)
                body = "invalid json";
            else if (response == 5 || response == 6) {
                const QString flags =
                        response == 5
                                ? QStringLiteral(
                                          "```json\n{\"cat\":true,\"dog\":"
                                          "false,\"unknown\":true}\n```")
                                : QStringLiteral(
                                          "{\"cat\":\"true\",\"dog\":false}");
                body = QJsonDocument(
                               QJsonObject{{"choices",
                                            QJsonArray{QJsonObject{
                                                    {"message",
                                                     QJsonObject{{"content",
                                                                  flags}}}}}}})
                               .toJson(QJsonDocument::Compact);
            } else if (received.value("stream").toBool()) {
                const QByteArray stream =
                        QByteArray("data: ") +
                        QJsonDocument(
                                QJsonObject{
                                        {"choices",
                                         QJsonArray{QJsonObject{
                                                 {"delta",
                                                  QJsonObject{
                                                          {"content",
                                                           QString::fromUtf8(
                                                                   "图像答"
                                                                   "案")}}}}}}})
                                .toJson(QJsonDocument::Compact) +
                        "\r\n\r\ndata: [DONE]\n\n";
                socket->write(
                        "HTTP/1.1 200 OK\r\nContent-Type: "
                        "text/event-stream\r\nConnection: close\r\n\r\n");
                const int cut = stream.indexOf(char(0xe5)) + 1;
                socket->write(stream.left(cut));
                QTimer::singleShot(10, socket, [socket, stream, cut]() {
                    socket->write(stream.mid(cut));
                    socket->disconnectFromHost();
                });
                return;
            } else
                body = "{\"choices\":[{\"message\":{\"content\":\"Generated "
                       "answer\"}}]}";
            socket->write((response == 1 ? "HTTP/1.1 401 Unauthorized\r\n"
                                         : "HTTP/1.1 200 OK\r\n") +
                          QByteArray("Content-Type: "
                                     "application/json\r\nContent-Length: ") +
                          QByteArray::number(body.size()) +
                          "\r\nConnection: close\r\n\r\n" + body);
            socket->disconnectFromHost();
        });
        QObject::connect(socket, &QTcpSocket::disconnected, socket,
                         &QObject::deleteLater);
    });
    ChatServiceClient client;
    ChatServiceClient::Settings settings;
    settings.baseUrl = QUrl(
            QStringLiteral("http://127.0.0.1:%1/v1/").arg(server.serverPort()));
    settings.model = "vision-test";
    settings.apiKey = "test-key";
    QString answer, error;
    int finished = 0, models = 0;
    QObject::connect(&client, &ChatServiceClient::finished, &app,
                     [&](const QString& text, const QString& failure) {
                         answer = text;
                         error = failure;
                         ++finished;
                     });
    QObject::connect(&client, &ChatServiceClient::modelsReady, &app,
                     [&](const QJsonArray& values, const QString& failure) {
                         models = values.size();
                         error = failure;
                     });
    client.fetchModels(settings);
    REQUIRE(waitFor([&]() { return models == 1; }) && error.isEmpty());
    REQUIRE(headers.startsWith("GET /v1/models ") &&
            headers.contains("Bearer test-key"));
    models = 0;
    settings.baseUrl.setPath("/anthropic/v1/");
    settings.apiKey = " test-key ";
    client.fetchModels(settings);
    REQUIRE(waitFor([&]() { return models == 1; }) && error.isEmpty());
    REQUIRE(headers.startsWith("GET /anthropic/v1/models ") &&
            headers.contains("x-api-key: test-key") &&
            headers.contains("anthropic-version: 2023-06-01") &&
            !headers.toLower().contains("authorization:"));
    settings.baseUrl.setPath("/v1/");
    settings.apiKey = "test-key";
    QImage image(80, 60, QImage::Format_RGB888);
    image.fill(Qt::green);
    const QString imagePath = temp.path() + "/first.png";
    const QString secondPath = temp.path() + "/second.png";
    REQUIRE(image.save(imagePath) && image.save(secondPath));
    const auto message =
            ChatServiceClient::userMessage("Describe", imagePath, &error);
    REQUIRE(error.isEmpty());
    const auto content = message.value("content").toArray();
    const auto uri = content.first()
                             .toObject()
                             .value("image_url")
                             .toObject()
                             .value("url")
                             .toString();
    QFile original(imagePath);
    REQUIRE(original.open(QIODevice::ReadOnly));
    REQUIRE(uri.startsWith("data:image/png;base64,") &&
            QByteArray::fromBase64(uri.section(',', 1).toLatin1()) ==
                    original.readAll());
    client.complete(settings, QJsonArray{message});
    REQUIRE(waitFor([&]() { return finished == 1; }) && error.isEmpty() &&
            answer == QString::fromUtf8("图像答案"));
    REQUIRE(headers.startsWith("POST /v1/chat/completions ") &&
            !received.contains("max_tokens") &&
            received.value("temperature").toDouble() == 1);
    settings.maxTokens = 123;
    client.complete(settings, QJsonArray{message}, false);
    REQUIRE(waitFor([&]() { return finished == 2; }) && error.isEmpty() &&
            answer == "Generated answer" &&
            received.value("max_tokens").toInt() == 123);
    response = 1;
    client.complete(settings, QJsonArray{message}, false);
    REQUIRE(waitFor([&]() { return finished == 3; }) &&
            error == "bad credentials");
    response = 2;
    client.complete(settings, QJsonArray{message}, false);
    REQUIRE(waitFor([&]() { return finished == 4; }) && !error.isEmpty());
    response = 4;
    settings.timeoutMs = 30;
    client.complete(settings, QJsonArray{message}, false);
    REQUIRE(waitFor([&]() { return finished == 5; }) &&
            error.contains("timed out"));
    settings.timeoutMs = 120000;
    int before = requests;
    client.complete(settings, QJsonArray{message});
    REQUIRE(waitFor([&]() { return requests > before; }));
    client.cancel();
    REQUIRE(!client.busy());
    response = 0;
    client.complete(settings, QJsonArray{message});
    REQUIRE(waitFor([&]() { return finished == 6; }) && error.isEmpty());
    REQUIRE(ChatServiceClient::userMessage("test", temp.path() + "/missing.png",
                                           &error)
                    .isEmpty() &&
            !error.isEmpty());

    // Real plugin workflow: all four source component types and metadata IO,
    // AI preview/apply, streamed chat, cancel and source-generation changes.
    const QJsonArray components{
            QJsonObject{{"title", "Caption"},
                        {"type", "QLineEdit"},
                        {"options", QJsonArray()}},
            QJsonObject{{"title", "Radio"},
                        {"type", "QRadioButton"},
                        {"options", QJsonArray{"yes", "no"}}},
            QJsonObject{{"title", "Choice"},
                        {"type", "QComboBox"},
                        {"options", QJsonArray{"red", "green"}}},
            QJsonObject{{"title", "Multi"},
                        {"type", "QCheckBox"},
                        {"options", QJsonArray{"car", "road"}}}};
    QSettings saved;
    saved.setValue("qAnyLabeling/workDirectory", temp.path());
    saved.setValue(
            "qAnyLabeling/vqa/components",
            QJsonDocument(QJsonObject{{"components", components}}).toJson());
    saved.setValue("qAnyLabeling/chat/providers/ollama/address",
                   settings.baseUrl.toString());
    saved.setValue("qAnyLabeling/chat/providers/ollama/model", settings.model);
    QJsonObject document{{"version", "4.1.3"},
                         {"imagePath", "first.png"},
                         {"imageWidth", 80},
                         {"imageHeight", 60},
                         {"shapes", QJsonArray()},
                         {"vendor", "keep"},
                         {"flags", QJsonObject{{"cat", false}, {"dog", false}}},
                         {"vqaData", QJsonObject{{"Unknown", "preserve"}}}};
    QFile sidecar(temp.path() + "/first.json");
    REQUIRE(sidecar.open(QIODevice::WriteOnly));
    sidecar.write(QJsonDocument(document).toJson());
    sidecar.close();
    QPluginLoader loader(QString::fromLocal8Bit(argv[1]));
    auto* plugin = qobject_cast<ccStdPluginInterface*>(loader.instance());
    REQUIRE(plugin);
    plugin->getActions().first()->trigger();
    QMainWindow* window = nullptr;
    for (auto* widget : QApplication::topLevelWidgets())
        if (widget->objectName() == "qAnyLabelingWorkspace")
            window = qobject_cast<QMainWindow*>(widget);
    REQUIRE(window &&
            QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath)));
    auto* vqaAction = window->findChild<QAction*>("AnnotationOpenVQA");
    REQUIRE(vqaAction);
    vqaAction->trigger();
    auto* vqa = window->findChild<QDialog*>("AnnotationVQA");
    REQUIRE(vqa);
    const auto providerConfig = AnnotationChatConfig::path("providers.json");
    const auto modelsConfig = AnnotationChatConfig::path("models.json");
    REQUIRE(readJson(providerConfig)
                            .value("ollama")
                            .toObject()
                            .value("api_address") ==
                    settings.baseUrl.toString() &&
            readJson(modelsConfig)
                            .value("settings")
                            .toObject()
                            .value("model_id") == settings.model &&
            readJson(modelsConfig)
                            .value("settings")
                            .toObject()
                            .value("temperature") == 10);
    REQUIRE((QFileInfo(providerConfig).permissions() &
             (QFileDevice::ReadGroup | QFileDevice::WriteGroup |
              QFileDevice::ReadOther | QFileDevice::WriteOther)) == 0);
    if (argc == 3) {
        QFile defaults(QDir(QString::fromLocal8Bit(argv[2]))
                               .filePath("chat-config-defaults-native.json"));
        REQUIRE(defaults.open(QIODevice::WriteOnly));
        defaults.write(
                QJsonDocument(
                        QJsonObject{{"providers",
                                     AnnotationChatConfig::defaultProviders()},
                                    {"models",
                                     AnnotationChatConfig::defaultModels()}})
                        .toJson());
    }
    // Compare capability precedence against the actual source provider methods.
    QJsonArray capabilityCases;
    for (const auto& id :
         QStringList{"vision-test", "gpt-image-vision", "llava", "custom-text",
                     "gemini-2.5-pro"}) {
        const QVector<QJsonObject> metadataCases{
                {},
                {{"vision", false}},
                {{"supports_vision", true}},
                {{"capabilities", QJsonObject{{"image", false}}}},
                {{"architecture", QJsonObject{{"input_modalities",
                                               QJsonArray{"text", "image"}}}}},
                {{"inputs", QJsonArray{"text"}}},
                {{"modalities", QJsonObject{{"input", QJsonArray{"video"}}}}},
                {{"modality", "text->image"}},
                {{"modality", "text+image->text"}}};
        for (const auto& metadata : metadataCases) {
            const auto configured = AnnotationChatConfig::defaultModels()
                                            .value("supported_vision_models")
                                            .toArray();
            capabilityCases.append(
                    QJsonObject{{"id", id},
                                {"metadata", metadata},
                                {"vision", AnnotationChatConfig::supportsVision(
                                                   id, metadata, configured)}});
        }
    }
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-capabilities-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(QJsonDocument(capabilityCases).toJson());
    }
    QJsonArray searchCases;
    auto searchCase = [&](const QString& query, const QString& name) {
        searchCases.append(QJsonObject{
                {"query", query},
                {"name", name},
                {"ratio",
                 AnnotationChatConfig::modelNameSimilarity(query, name)},
                {"match",
                 AnnotationChatConfig::modelNameMatches(query, name)}});
    };
    for (const auto& query : QStringList{"", "tide", "diet", "vision", "VSON",
                                         QString::fromUtf8("中😀İΣ")})
        for (const auto& name : QStringList{"", "tide", "diet", "vision-test",
                                            QString::fromUtf8("😀中iΣ")})
            searchCase(query, name);
    unsigned seed = 1827;
    const QStringList alphabet{"a",
                               "b",
                               "c",
                               "-",
                               "v",
                               "X",
                               QString::fromUtf8("😀"),
                               QString::fromUtf8("中")};
    auto randomText = [&]() {
        seed = seed * 1664525u + 1013904223u;
        const int length = (seed >> 16) % 40;
        QString text;
        for (int i = 0; i < length; ++i) {
            seed = seed * 1664525u + 1013904223u;
            text += alphabet[(seed >> 16) % alphabet.size()];
        }
        return text;
    };
    for (int i = 0; i < 1000; ++i) {
        const QString query = randomText();
        const QString name = randomText();
        searchCase(query, name);
    }
    for (int i = 0; i < 15; ++i) {
        searchCase(QString(i * 19, 'a') + "b" + QString(205 - i, 'a'),
                   QString(210, 'a') + "b" + QString(i, 'a'));
        searchCase("b" + QString(i + 200, 'a'), QString(i + 200, 'a') + "b");
    }
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-model-search-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(QJsonDocument(searchCases).toJson());
    }
    QPlainTextEdit* caption = nullptr;
    for (auto* edit : vqa->findChildren<QPlainTextEdit*>("VQAComponent"))
        if (edit->property("title") == "Caption") caption = edit;
    REQUIRE(caption);
    caption->setPlainText(" manual caption ");
    auto* combo = vqa->findChild<QComboBox*>("VQAComponent");
    REQUIRE(combo);
    combo->setCurrentIndex(2);
    auto checkboxes = vqa->findChildren<QCheckBox*>();
    for (auto* box : checkboxes)
        if (box->text() == "road") box->setChecked(true);
    auto values = readJson(sidecar.fileName()).value("vqaData").toObject();
    REQUIRE(values.value("Caption") == "manual caption" &&
            values.value("Radio") == "yes" &&
            values.value("Choice") == "green" &&
            values.value("Multi").toArray() == QJsonArray{"road"} &&
            values.value("Unknown") == "preserve");
    REQUIRE(readJson(sidecar.fileName()).value("vendor") == "keep" &&
            readJson(sidecar.fileName()).value("version") == "4.1.3");
    const auto beforeVqaClear = readJson(sidecar.fileName());
    const auto clearVqa = [&](QMessageBox::StandardButton answer) {
        bool defaultNo = false;
        QTimer::singleShot(0, vqa, [&]() {
            auto* question = qobject_cast<QMessageBox*>(
                    QApplication::activeModalWidget());
            if (question) {
                defaultNo = question->defaultButton() ==
                            question->button(QMessageBox::No);
                question->button(answer)->click();
            }
        });
        vqa->findChild<QPushButton*>("VQAClearCurrent")->click();
        return defaultNo;
    };
    REQUIRE(clearVqa(QMessageBox::No) &&
            readJson(sidecar.fileName()) == beforeVqaClear);
    REQUIRE(clearVqa(QMessageBox::Yes));
    const auto clearedVqa =
            readJson(sidecar.fileName()).value("vqaData").toObject();
    REQUIRE(clearedVqa.value("Caption") == "" &&
            clearedVqa.value("Radio").isNull() &&
            clearedVqa.value("Choice").isNull() &&
            clearedVqa.value("Multi").toArray().isEmpty() &&
            clearedVqa.value("Unknown") == "preserve" &&
            readJson(sidecar.fileName()).value("vendor") == "keep");
    caption->setPlainText(" manual caption ");
    combo->setCurrentIndex(2);
    for (auto* radio : vqa->findChildren<QRadioButton*>())
        if (radio->text() == "yes") radio->setChecked(true);
    for (auto* box : checkboxes)
        if (box->text() == "road") box->setChecked(true);
    REQUIRE(readJson(sidecar.fileName()) == beforeVqaClear);
    // Actual export dialogs: whole image list, field selection/rename and
    // defaults for an unlabelled second image. No navigation or sidecar writes.
    auto* openFolder = vqa->findChild<QPushButton*>("VQAOpenFolder");
    REQUIRE(openFolder);
    bool folderSelected = false;
    QTimer folderChooser;
    QObject::connect(&folderChooser, &QTimer::timeout, &folderChooser, [&]() {
        auto* file =
                qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        if (!file || folderSelected) return;
        folderSelected = true;
        file->setDirectory(temp.path());
        QMetaObject::invokeMethod(file, "accept", Qt::QueuedConnection);
    });
    folderChooser.start(10);
    openFolder->click();
    folderChooser.stop();
    REQUIRE(folderSelected);
    QCoreApplication::processEvents();
    auto* vqaFiles = window->findChild<QListWidget*>("AnnotationFiles");
    REQUIRE(vqaFiles && vqaFiles->count() == 2 &&
            vqaFiles->item(0)->data(Qt::UserRole).toString() == imagePath &&
            vqaFiles->item(1)->data(Qt::UserRole).toString() == secondPath);
    auto* vqaSplitter = vqa->findChild<QSplitter*>("VQAMainSplitter");
    auto* vqaToggle = vqa->findChild<QPushButton*>("VQAToggleSidebar");
    auto* vqaRestore = vqa->findChild<QPushButton*>("VQAToggleSidebarRight");
    auto* vqaRefresh = vqa->findChild<QPushButton*>("VQARefresh");
    auto* vqaRefreshRight = vqa->findChild<QPushButton*>("VQARefreshRight");
    auto* vqaPage = vqa->findChild<QLineEdit*>("VQAPageInput");
    auto* vqaPrevious = vqa->findChild<QPushButton*>("VQAPrevious");
    auto* vqaNext = vqa->findChild<QPushButton*>("VQANext");
    REQUIRE(vqaSplitter && vqaToggle && vqaRestore && vqaRefresh &&
            vqaRefreshRight && !vqaRefreshRight->isVisible() && vqaPage &&
            vqaPrevious && vqaNext && !vqaPrevious->isEnabled() &&
            vqaNext->isEnabled() && vqaPage->text() == "1" &&
            vqaPage->size() == QSize(68, 32) && !vqaRestore->isVisible());
    vqaToggle->click();
    REQUIRE(vqaSplitter->sizes().first() == 0 && vqaRestore->isVisible() &&
            vqaRefreshRight->isVisible());
    QKeyEvent vqaEnter(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    vqaPage->setText("2");
    QCoreApplication::sendEvent(vqaPage, &vqaEnter);
    REQUIRE(vqaPage->text() == "2" && vqaPrevious->isEnabled() &&
            !vqaNext->isEnabled() && caption->toPlainText().isEmpty() &&
            vqaSplitter->sizes().first() == 0);
    vqaPage->setText("99");
    REQUIRE(vqaPage->text() == "2");
    vqaPage->clear();
    QCoreApplication::sendEvent(vqaPage, &vqaEnter);
    REQUIRE(vqaPage->text() == "2");
    vqaPrevious->click();
    REQUIRE(vqaPage->text() == "1" && !vqaPrevious->isEnabled() &&
            vqaNext->isEnabled() &&
            caption->toPlainText() == "manual caption" &&
            vqaSplitter->sizes().first() == 0);
    // The current path is unchanged: refresh must reload disk data through
    // the owning workspace rather than reuse the VQA dialog's snapshot.
    auto externalVqa = readJson(sidecar.fileName());
    auto externalValues = externalVqa.value("vqaData").toObject();
    externalValues.insert("Caption", "externally edited caption");
    externalVqa.insert("vqaData", externalValues);
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write(QJsonDocument(externalVqa).toJson());
    sidecar.close();
    vqaRefreshRight->click();
    REQUIRE(caption->toPlainText() == "externally edited caption" &&
            vqaSplitter->sizes().first() == 0 &&
            readJson(sidecar.fileName()) == externalVqa);
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write(QJsonDocument(beforeVqaClear).toJson());
    sidecar.close();
    vqaRefresh->click();
    REQUIRE(caption->toPlainText() == "manual caption" &&
            readJson(sidecar.fileName()) == beforeVqaClear);
    vqaRestore->click();
    REQUIRE(vqaSplitter->sizes().first() > 0 && !vqaRestore->isVisible() &&
            !vqaRefreshRight->isVisible());
    QJsonArray vqaThemes;
    for (const auto& mode : {QStringLiteral("dark"), QStringLiteral("light")}) {
        auto* action = window->findChild<QAction*>("AnnotationTheme_" + mode);
        REQUIRE(action);
        auto* modal = QApplication::activeModalWidget();
        std::fprintf(stderr,
                     "VQA theme action %s: enabled=%d checked=%d modal=%s "
                     "workspace_enabled=%d\n",
                     qPrintable(mode), action->isEnabled(), action->isChecked(),
                     modal ? qPrintable(modal->objectName()) : "none",
                     window->isEnabled());
        bool themeTriggered = false;
        const auto themeConnection = QObject::connect(
                action, &QAction::triggered, action,
                [&themeTriggered]() { themeTriggered = true; });
        action->trigger();
        QObject::disconnect(themeConnection);
        REQUIRE(themeTriggered);
        QCoreApplication::processEvents();
        auto* imageContainer = vqa->findChild<QWidget*>("VQAImageContainer");
        auto* filename = vqa->findChild<QLabel*>("VQAFilename");
        auto* exportButton = vqa->findChild<QPushButton*>("VQAExport");
        REQUIRE(imageContainer && filename && exportButton);
        const auto surface = imageContainer->grab().toImage().pixelColor(5, 5);
        const auto text = filename->palette().color(QPalette::WindowText);
        const auto exportColor = exportButton->grab().toImage().pixelColor(
                exportButton->width() / 2, 5);
        std::fprintf(
                stderr,
                "VQA theme %s: surface=%s text=%s export=%s workspace=%s "
                "application=%s standard=%s stylesheet_empty=%d\n",
                qPrintable(mode), qPrintable(surface.name()),
                qPrintable(text.name()), qPrintable(exportColor.name()),
                qPrintable(window->palette().color(QPalette::Window).name()),
                qPrintable(QApplication::palette().color(QPalette::Window).name()),
                qPrintable(QApplication::style()->standardPalette()
                                   .color(QPalette::Window).name()),
                window->styleSheet().isEmpty());
        REQUIRE(surface == QColor(mode == "dark" ? "#2c2c2e" : "#F9F9F9") &&
                text == QColor(mode == "dark" ? "#f5f5f7" : "#1d1d1f") &&
                exportColor == QColor("#10b981"));
        QJsonObject styles;
        for (const char* name :
             {"VQAMainSplitter", "VQAFilename", "VQAImageContainer",
              "VQAImagePreview", "VQAPageInput", "VQARefresh",
              "VQARefreshRight", "VQAToggleSidebar", "VQAToggleSidebarRight",
              "VQAPrevious", "VQANext", "VQAAddComponent", "VQAExport",
              "VQAClearCurrent"}) {
            auto* widget = vqa->findChild<QWidget*>(name);
            REQUIRE(widget);
            styles.insert(name, widget->styleSheet());
        }
        vqaThemes.append(
                QJsonObject{{"mode", mode},
                            {"styles", styles},
                            {"image_container_surface", surface.name()},
                            {"filename_text", text.name()},
                            {"export_background", exportColor.name()}});
        if (argc == 3)
            REQUIRE(vqa->grab().save(
                    QDir(QString::fromLocal8Bit(argv[2]))
                            .filePath("vqa-window-" + mode + ".png")));
    }
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("vqa-styles-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(QJsonDocument(vqaThemes).toJson());
    }
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("vqa-navigation-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(
                QJsonDocument(
                        QJsonObject{
                                {"page_size", QJsonArray{vqaPage->width(),
                                                         vqaPage->height()}},
                                {"splitter_handle", vqaSplitter->handleWidth()},
                                {"panel_button_size",
                                 QJsonArray{vqaToggle->width(),
                                            vqaToggle->height()}},
                                {"nav_button_size",
                                 QJsonArray{vqaPrevious->width(),
                                            vqaPrevious->height()}},
                                {"cleared_vqa", clearedVqa},
                                {"first_page", QJsonArray{false, true}},
                                {"last_page", QJsonArray{true, false}},
                                {"empty_restored", "2"},
                                {"over_maximum", "2"}})
                        .toJson());
    }
    auto* exportLabels = vqa->findChild<QPushButton*>("VQAExport");
    REQUIRE(exportLabels);
    auto exportTo = [&](const QString& path) {
        bool configured = false, chosen = false;
        QTimer choose;
        QObject::connect(&choose, &QTimer::timeout, &choose, [&]() {
            auto* modal = QApplication::activeModalWidget();
            if (!modal) return;
            if (modal->objectName() == "VQAExportFields" && !configured) {
                auto* fields =
                        modal->findChild<QTableWidget*>("VQAExportTable");
                if (!fields) return;
                configured = true;
                // Keep image, Caption and Multi; rename Caption to answer.
                for (int row = 0; row < fields->rowCount(); ++row) {
                    const auto key = fields->item(row, 1)->text();
                    static_cast<QCheckBox*>(fields->cellWidget(row, 3))
                            ->setChecked(key == "image" || key == "Caption" ||
                                         key == "Multi");
                    if (key == "Caption")
                        static_cast<QLineEdit*>(fields->cellWidget(row, 2))
                                ->setText("answer");
                }
                modal->findChild<QDialogButtonBox*>()
                        ->button(QDialogButtonBox::Save)
                        ->click();
            } else if (auto* file = qobject_cast<QFileDialog*>(modal)) {
                if (chosen) return;
                chosen = true;
                // Drive the visible filename edit. selectFile alone can retain
                // the prior default while QFileSystemModel resolves a folder.
                file->setDirectory(QFileInfo(path).absolutePath());
                file->findChild<QLineEdit*>("fileNameEdit")->setText(path);
                QMetaObject::invokeMethod(file, "accept", Qt::QueuedConnection);
            } else if (auto* message = qobject_cast<QMessageBox*>(modal)) {
                if (message->button(QMessageBox::Yes))
                    message->button(QMessageBox::Yes)->click();
            }
        });
        choose.start(10);
        exportLabels->click();
        choose.stop();
        return configured && chosen;
    };
    const QString exportedPath = temp.path() + "/vqa-selected.jsonl";
    REQUIRE(exportTo(exportedPath));
    if (!QFileInfo::exists(exportedPath))
        std::fprintf(stderr, "VQA export status: %s\n",
                     qPrintable(vqa->findChild<QLabel*>("ChatStatus")->text()));
    QFile exported(exportedPath);
    REQUIRE(exported.open(QIODevice::ReadOnly));
    const auto exportedRows = exported.readAll().trimmed().split('\n');
    exported.close();
    if (exportedRows.size() != 2) {
        const auto* list = window->findChild<QListWidget*>("AnnotationFiles");
        std::fprintf(stderr, "VQA image list: %d; output: %s\n",
                     list ? list->count() : -1,
                     exportedRows.join('\n').constData());
        if (list)
            for (int i = 0; i < list->count(); ++i)
                std::fprintf(
                        stderr, "  %s\n",
                        qPrintable(
                                list->item(i)->data(Qt::UserRole).toString()));
    }
    REQUIRE(exportedRows.size() == 2);
    const auto firstExport = QJsonDocument::fromJson(exportedRows[0]).object();
    const auto secondExport = QJsonDocument::fromJson(exportedRows[1]).object();
    REQUIRE(firstExport == QJsonObject({{"image", "first.png"},
                                        {"answer", "manual caption"},
                                        {"Multi", QJsonArray{"road"}}}));
    REQUIRE(secondExport == QJsonObject({{"image", "second.png"},
                                         {"answer", ""},
                                         {"Multi", QJsonArray()}}));
    REQUIRE(!QFileInfo::exists(temp.path() + "/second.json"));
    if (argc == 3) {
        const QString evidence = QString::fromLocal8Bit(argv[2]);
        REQUIRE(QDir().mkpath(evidence));
        for (const auto& path :
             {imagePath, secondPath, sidecar.fileName(), exportedPath})
            REQUIRE(QFile::copy(
                    path, QDir(evidence).filePath(QFileInfo(path).fileName())));
        QFile config(QDir(evidence).filePath("export-config.json"));
        REQUIRE(config.open(QIODevice::WriteOnly));
        config.write(
                QJsonDocument(QJsonObject{{"components", components},
                                          {"mapping",
                                           QJsonObject{{"image", "image"},
                                                       {"Caption", "answer"},
                                                       {"Multi", "Multi"}}}})
                        .toJson());
    }
    // A malformed later sidecar cancels atomic export and preserves old bytes.
    QFile bad(temp.path() + "/second.json");
    REQUIRE(bad.open(QIODevice::WriteOnly));
    bad.write("{broken");
    bad.close();
    QFile sentinel(temp.path() + "/vqa-preserved.jsonl");
    REQUIRE(sentinel.open(QIODevice::WriteOnly));
    sentinel.write("previous output\n");
    sentinel.close();
    REQUIRE(exportTo(sentinel.fileName()));
    REQUIRE(sentinel.open(QIODevice::ReadOnly));
    REQUIRE(sentinel.readAll() == "previous output\n");
    sentinel.close();
    REQUIRE(bad.open(QIODevice::ReadOnly) && bad.readAll() == "{broken");
    bad.close();
    REQUIRE(bad.remove());
    auto* confidence =
            window->findChild<QDoubleSpinBox*>("AnnotationConfidence");
    REQUIRE(confidence);
    confidence->setValue(confidence->value() > 0.5 ? 0.25 : 0.75);
    caption->setPlainText("manual caption after model revision");
    REQUIRE(readJson(sidecar.fileName())
                    .value("vqaData")
                    .toObject()
                    .value("Caption") == "manual caption after model revision");
    auto assistants = vqa->findChildren<QPushButton*>("VQAAssistant");
    REQUIRE(!assistants.isEmpty());
    QString chosenPrompt;
    bool promptContract = true;
    auto visitTemplates = [&](const std::function<void(QDialog*)>& visit) {
        QTimer::singleShot(0, vqa, [&]() {
            auto* dialog =
                    qobject_cast<QDialog*>(QApplication::activeModalWidget());
            if (!dialog || dialog->objectName() != "VQAAiPromptDialog") return;
            auto* edit = dialog->findChild<QTextEdit*>("VQAPrompt");
            promptContract &= edit && edit->toPlainText().isEmpty() &&
                              dialog->minimumWidth() == 600 &&
                              edit->minimumHeight() == 120 &&
                              edit->maximumHeight() == 160;
            QTimer::singleShot(0, dialog, [&]() {
                auto* gallery = qobject_cast<QDialog*>(
                        QApplication::activeModalWidget());
                if (gallery && gallery->objectName() == "VQATemplateGallery")
                    visit(gallery);
            });
            dialog->findChild<QPushButton*>("VQATemplates")->click();
            chosenPrompt = edit->toPlainText();
            dialog->reject();
        });
        assistants.first()->click();
    };
    auto submitPrompt = [&](const QString& text) {
        QTimer::singleShot(0, vqa, [&]() {
            auto* dialog =
                    qobject_cast<QDialog*>(QApplication::activeModalWidget());
            if (!dialog || dialog->objectName() != "VQAAiPromptDialog") return;
            dialog->findChild<QTextEdit*>("VQAPrompt")->setPlainText(text);
            dialog->findChild<QPushButton*>("VQAPromptGenerate")->click();
        });
        assistants.first()->click();
    };
    // Source defaults and user-template CRUD use the source JSON shape/path.
    const QString promptsPath =
            QDir(temp.path()).filePath("xanylabeling_data/vqa/prompts.json");
    QJsonArray systemTemplates;
    bool templatesValid = false;
    visitTemplates([&](QDialog* gallery) {
        auto* table = gallery->findChild<QTableWidget*>("VQATemplateTable");
        templatesValid = table && table->rowCount() == 11;
        if (templatesValid) {
            for (int row = 0; row < table->rowCount(); ++row) {
                systemTemplates.append(QJsonObject{
                        {"name", table->item(row, 1)->text()},
                        {"content", table->item(row, 1)->toolTip()}});
                templatesValid &= !table->cellWidget(row, 2)->isEnabled();
            }
            static_cast<QCheckBox*>(table->cellWidget(8, 0))->setChecked(true);
        }
        gallery->accept();
    });
    REQUIRE(templatesValid &&
            chosenPrompt ==
                    "@image Read all texts in the image, output in lines.");
    auto editTemplate = [](const QString& name, const QString& content) {
        auto* editor =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!editor || editor->objectName() != "VQATemplateEditor") return;
        editor->findChild<QLineEdit*>("VQATemplateName")->setText(name);
        editor->findChild<QPlainTextEdit*>("VQATemplateContent")
                ->setPlainText(content);
        editor->findChild<QDialogButtonBox*>()
                ->button(QDialogButtonBox::Ok)
                ->click();
    };
    visitTemplates([&](QDialog* gallery) {
        QTimer::singleShot(0, gallery, [&]() {
            editTemplate("My prompt", "@image 自定义");
        });
        gallery->findChild<QPushButton*>("VQAAddTemplate")->click();
        auto* table = gallery->findChild<QTableWidget*>("VQATemplateTable");
        templatesValid = table->rowCount() == 12 &&
                         table->item(0, 1)->text() == "My prompt";
        auto* user = static_cast<QCheckBox*>(table->cellWidget(0, 0));
        auto* system = static_cast<QCheckBox*>(table->cellWidget(9, 0));
        user->setChecked(true);
        system->setChecked(true);
        templatesValid &= !user->isChecked();
        user->setChecked(true);
        templatesValid &= !system->isChecked();
        gallery->accept();
    });
    REQUIRE(templatesValid && chosenPrompt == "@image 自定义");
    QFile promptsFile(promptsPath);
    REQUIRE(promptsFile.open(QIODevice::ReadOnly));
    REQUIRE((QJsonDocument::fromJson(promptsFile.readAll()).array() ==
             QJsonArray{QJsonObject{{"name", "My prompt"},
                                    {"content", "@image 自定义"}}}));
    promptsFile.close();
    visitTemplates([&](QDialog* gallery) {
        auto* table = gallery->findChild<QTableWidget*>("VQATemplateTable");
        QTimer::singleShot(0, gallery,
                           [&]() { editTemplate("Revised", "@text revised"); });
        table->itemDoubleClicked(table->item(0, 1));
        templatesValid = table->item(0, 1)->text() == "Revised";
        QTimer::singleShot(0, gallery, []() {
            auto* question = qobject_cast<QMessageBox*>(
                    QApplication::activeModalWidget());
            if (question) question->button(QMessageBox::Yes)->click();
        });
        static_cast<QPushButton*>(table->cellWidget(0, 2))->click();
        templatesValid &= waitFor([&]() { return table->rowCount() == 11; });
        gallery->reject();
    });
    REQUIRE(templatesValid);
    REQUIRE(promptsFile.open(QIODevice::ReadOnly));
    REQUIRE(QJsonDocument::fromJson(promptsFile.readAll()).array().isEmpty());
    promptsFile.close();
    REQUIRE(promptsFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    promptsFile.write("damaged template configuration");
    promptsFile.close();
    visitTemplates([&](QDialog* gallery) {
        templatesValid = !gallery->findChild<QPushButton*>("VQAAddTemplate")
                                  ->isEnabled();
        gallery->reject();
    });
    REQUIRE(templatesValid && promptsFile.open(QIODevice::ReadOnly));
    REQUIRE(promptsFile.readAll() == "damaged template configuration");
    promptsFile.close();
    if (argc == 3) {
        QFile defaults(QDir(QString::fromLocal8Bit(argv[2]))
                               .filePath("default-templates.json"));
        REQUIRE(defaults.open(QIODevice::WriteOnly));
        defaults.write(QJsonDocument(systemTemplates).toJson());
    }
    REQUIRE(promptContract);
    const int beforeCancelledPrompt = requests;
    QTimer::singleShot(0, vqa, []() {
        auto* dialog =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (dialog) dialog->reject();
    });
    assistants.first()->click();
    REQUIRE(requests == beforeCancelledPrompt);
    // Use a deliberately different disk document, with insertion order,
    // float/int distinction, Python literals, and quotes inside nested values.
    const QByteArray referenceLabels =
            R"JSON({"shapes":[{"z":false,"points":[[1.0,-0.0],[0.00001,1000000000000000.0]],"label":"don't\n\u4e2d\u6587"}],"imagePath":"disk.png","imageHeight":99,"imageWidth":123,"flags":{"z":true,"a":null},"vqaData":{"X":[true,null,"don't"],"XLong":"longer","Empty":false}})JSON";
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    REQUIRE(sidecar.write(referenceLabels) == referenceLabels.size());
    sidecar.close();
    const QString referencePrompt =
            "  @image\n\t @text | @widget.Caption | @label.shapes | "
            "@label.imagePath | @label.imageHeight | @label.imageWidth | "
            "@label.flags | @label.XLong | @label.Empty | @label.vendor  ";
    submitPrompt(referencePrompt);
    REQUIRE(waitFor([&]() {
        return vqa->findChild<QDialog*>("VQAAnswerPreview") != nullptr;
    }));
    const auto referenceMessages = received.value("messages").toArray();
    const auto referenceContent =
            referenceMessages.last().toObject().value("content").toArray();
    const QString referenceText =
            referenceContent.last().toObject().value("text").toString();
    REQUIRE(referenceText.contains("123") && referenceText.contains("True") &&
            referenceText.contains("None") && referenceText.contains("1.0") &&
            referenceText.contains("-0.0") &&
            referenceText.contains("@label.vendor"));
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("prompt-references.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(
                QJsonDocument(
                        QJsonObject{{"labels_raw",
                                     QString::fromUtf8(referenceLabels)},
                                    {"prompt", referencePrompt.trimmed()},
                                    {"current_text", caption->toPlainText()},
                                    {"widget_text", caption->toPlainText()},
                                    {"actual", referenceText},
                                    {"has_image", !referenceContent.first()
                                                           .toObject()
                                                           .value("image_url")
                                                           .isUndefined()}})
                        .toJson());
    }
    // Restore the original sidecar before checking preview/apply ownership.
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write(QJsonDocument(document).toJson());
    sidecar.close();
    auto* preview = vqa->findChild<QDialog*>("VQAAnswerPreview");
    preview->findChild<QDialogButtonBox*>()
            ->button(QDialogButtonBox::Apply)
            ->click();
    REQUIRE(readJson(sidecar.fileName())
                    .value("vqaData")
                    .toObject()
                    .value("Caption") == "Generated answer");
    response = 4;
    before = requests;
    submitPrompt("@image cancel this request");
    REQUIRE(waitFor([&]() { return requests > before; }));
    auto* loading = vqa->findChild<QDialog*>("VQALoadingDialog");
    REQUIRE(loading && loading->isModal() &&
            loading->size() == QSize(420, 180));
    loading->findChild<QPushButton*>("VQALoadingCancel")->click();
    REQUIRE(waitFor([&]() {
        auto* active = vqa->findChild<QDialog*>("VQALoadingDialog");
        return !active || !active->isVisible();
    }));
    REQUIRE(readJson(sidecar.fileName())
                    .value("vqaData")
                    .toObject()
                    .value("Caption") == "Generated answer");
    REQUIRE(sidecar.open(QIODevice::ReadOnly));
    const auto savedLabelBytes = sidecar.readAll();
    sidecar.close();
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write("{broken");
    sidecar.close();
    before = requests;
    submitPrompt("@label.flags");
    REQUIRE(requests == before);
    REQUIRE(sidecar.open(QIODevice::ReadOnly) &&
            sidecar.readAll() == "{broken");
    sidecar.close();
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write(savedLabelBytes);
    sidecar.close();
    before = requests;
    submitPrompt("@image @text");
    REQUIRE(waitFor([&]() { return requests > before; }));
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, secondPath)));
    REQUIRE(caption->toPlainText().isEmpty() &&
            !readJson(temp.path() + "/second.json").contains("vqaData"));
    // Dataset schema changes operate on current and other sidecars, preserving
    // unrelated fields. A malformed later file must fail before any writes.
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath)));
    const QString otherSidecar = temp.path() + "/second.json";
    QJsonObject otherDocument{
            {"imagePath", "second.png"},
            {"vendor", "other"},
            {"shapes", QJsonArray()},
            {"vqaData", QJsonObject{{"Caption", "other caption"},
                                    {"Radio", "no"},
                                    {"Choice", "green"},
                                    {"Multi", QJsonArray{"road"}}}}};
    auto writeOther = [&](const QByteArray& bytes) {
        QFile file(otherSidecar);
        return file.open(QIODevice::WriteOnly) &&
               file.write(bytes) == bytes.size();
    };
    auto componentButton = [&](const char* action, const QString& title) {
        for (auto* button : vqa->findChildren<QPushButton*>(action))
            if (button->property("title").toString() == title) return button;
        return static_cast<QPushButton*>(nullptr);
    };
    auto schema = [&]() {
        return QJsonDocument::fromJson(
                       saved.value("qAnyLabeling/vqa/components").toByteArray())
                .object()
                .value("components")
                .toArray();
    };
    bool typeLocked = false;
    auto editComponent = [&](const QString& oldTitle, const QString& newTitle,
                             const QString& options) {
        auto* button = componentButton("VQAEditComponent", oldTitle);
        if (!button) return false;
        bool edited = false;
        QTimer edit;
        QObject::connect(&edit, &QTimer::timeout, &edit, [&]() {
            auto* modal = QApplication::activeModalWidget();
            if (!modal) return;
            if (modal->objectName() == "VQAComponentEditor" && !edited) {
                edited = true;
                typeLocked = !modal->findChild<QComboBox*>("VQAComponentType")
                                      ->isEnabled();
                modal->findChild<QLineEdit*>("VQAComponentTitle")
                        ->setText(newTitle);
                modal->findChild<QPlainTextEdit*>("VQAComponentOptions")
                        ->setPlainText(options);
                modal->findChild<QDialogButtonBox*>()
                        ->button(QDialogButtonBox::Ok)
                        ->click();
            } else if (auto* message = qobject_cast<QMessageBox*>(modal)) {
                if (message->button(QMessageBox::Yes))
                    message->button(QMessageBox::Yes)->click();
            }
        });
        edit.start(10);
        button->click();
        edit.stop();
        QCoreApplication::processEvents();
        return edited && typeLocked;
    };
    QFile firstBefore(sidecar.fileName());
    REQUIRE(firstBefore.open(QIODevice::ReadOnly));
    const auto firstBytes = firstBefore.readAll();
    firstBefore.close();
    REQUIRE(writeOther("{broken"));
    REQUIRE(editComponent("Caption", "Renamed", {}));
    REQUIRE(schema().first().toObject().value("title") == "Caption");
    REQUIRE(firstBefore.open(QIODevice::ReadOnly) &&
            firstBefore.readAll() == firstBytes);
    firstBefore.close();
    REQUIRE(vqa->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("Invalid annotation"));
    REQUIRE(writeOther(QJsonDocument(otherDocument).toJson()));
    REQUIRE(editComponent("Caption", "Renamed", {}));
    REQUIRE(waitFor(
            [&]() { return componentButton("VQAEditComponent", "Renamed"); }));
    auto firstVqa = readJson(sidecar.fileName()).value("vqaData").toObject();
    auto otherVqa = readJson(otherSidecar).value("vqaData").toObject();
    REQUIRE(!firstVqa.contains("Caption") &&
            firstVqa.value("Renamed") == "Generated answer" &&
            !otherVqa.contains("Caption") &&
            otherVqa.value("Renamed") == "other caption");
    REQUIRE(editComponent("Radio", "Radio", "affirmative\nnegative"));
    REQUIRE(waitFor([&]() {
        return readJson(otherSidecar)
                       .value("vqaData")
                       .toObject()
                       .value("Radio") == "negative";
    }));
    REQUIRE(readJson(sidecar.fileName())
                    .value("vqaData")
                    .toObject()
                    .value("Radio") == "affirmative");
    REQUIRE(editComponent("Multi", "Multi", "car"));
    REQUIRE(readJson(sidecar.fileName())
                    .value("vqaData")
                    .toObject()
                    .value("Multi")
                    .toArray()
                    .isEmpty() &&
            readJson(otherSidecar)
                    .value("vqaData")
                    .toObject()
                    .value("Multi")
                    .toArray()
                    .isEmpty());
    auto* down = componentButton("VQAComponentDown", "Renamed");
    REQUIRE(down && down->isEnabled());
    down->click();
    REQUIRE(waitFor([&]() {
        return schema().first().toObject().value("title") == "Radio";
    }));
    auto configFile = [&](const char* action, const QString& path) {
        auto* button = vqa->findChild<QPushButton*>(action);
        if (!button) return false;
        bool chosen = false;
        QTimer chooser;
        QObject::connect(&chooser, &QTimer::timeout, &chooser, [&]() {
            auto* file = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget());
            if (!file || chosen) return;
            chosen = true;
            file->setDirectory(QFileInfo(path).absolutePath());
            file->findChild<QLineEdit*>("fileNameEdit")->setText(path);
            QMetaObject::invokeMethod(file, "accept", Qt::QueuedConnection);
        });
        chooser.start(10);
        button->click();
        chooser.stop();
        return chosen;
    };
    const QString componentPath = temp.path() + "/components-export.json";
    REQUIRE(configFile("VQAExportComponents", componentPath));
    REQUIRE(readJson(componentPath).value("components").toArray() == schema());
    REQUIRE(configFile("VQAImportComponents", componentPath));
    auto* remove = componentButton("VQADeleteComponent", "Renamed");
    REQUIRE(remove);
    QTimer::singleShot(0, vqa, []() {
        auto* question =
                qobject_cast<QMessageBox*>(QApplication::activeModalWidget());
        if (question && question->button(QMessageBox::Yes))
            question->button(QMessageBox::Yes)->click();
    });
    remove->click();
    REQUIRE(waitFor([&]() { return schema().size() == 3; }));
    REQUIRE(!readJson(sidecar.fileName())
                     .value("vqaData")
                     .toObject()
                     .contains("Renamed") &&
            !readJson(otherSidecar)
                     .value("vqaData")
                     .toObject()
                     .contains("Renamed"));
    REQUIRE(readJson(sidecar.fileName()).value("vendor") == "keep" &&
            readJson(otherSidecar).value("vendor") == "other");
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("migration-fixture.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        const QJsonObject data{
                {"before",
                 QJsonArray{QJsonDocument::fromJson(firstBytes).object(),
                            otherDocument}},
                {"after", QJsonArray{readJson(sidecar.fileName()),
                                     readJson(otherSidecar)}}};
        fixture.write(QJsonDocument(data).toJson());
    }
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, secondPath)));
    QKeyEvent escape(QEvent::KeyPress, Qt::Key_Escape, Qt::NoModifier);
    QApplication::sendEvent(vqa, &escape);
    REQUIRE(!vqa->isVisible());
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(!window->findChild<QDialog*>("AnnotationVQA"));
    const QString automaticConfig =
            QDir(temp.path()).filePath("xanylabeling_data/vqa/components.json");
    const auto persistedComponents =
            readJson(automaticConfig).value("components");
    REQUIRE(persistedComponents.toArray().size() == 3);
    saved.remove("qAnyLabeling/vqa/components");
    vqaAction->trigger();
    vqa = window->findChild<QDialog*>("AnnotationVQA");
    REQUIRE(vqa && schema() == persistedComponents.toArray() &&
            vqa->findChildren<QWidget*>("VQAComponent").size() == 3);
    vqa->close();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QFile automaticFile(automaticConfig);
    REQUIRE(automaticFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    automaticFile.write("damaged component configuration");
    automaticFile.close();
    vqaAction->trigger();
    vqa = window->findChild<QDialog*>("AnnotationVQA");
    REQUIRE(vqa &&
            !vqa->findChild<QPushButton*>("VQAAddComponent")->isEnabled());
    for (auto* button : vqa->findChildren<QPushButton*>("VQAEditComponent"))
        REQUIRE(!button->isEnabled());
    REQUIRE(automaticFile.open(QIODevice::ReadOnly));
    REQUIRE(automaticFile.readAll() == "damaged component configuration");
    automaticFile.close();
    REQUIRE(configFile("VQAImportComponents", componentPath));
    REQUIRE(vqa->findChild<QPushButton*>("VQAAddComponent")->isEnabled() &&
            readJson(automaticConfig).value("components") ==
                    readJson(componentPath).value("components"));
    vqa->close();
    // Existing upstream JSON wins over old plugin settings. Keep unknown fields
    // and cached favorite/capability data through a full window round trip.
    auto providerDocument = readJson(providerConfig);
    auto ollama = providerDocument.value("ollama").toObject();
    ollama.insert("api_key", "json-secret");
    ollama.insert("vendor", "preserve-provider");
    providerDocument.insert("ollama", ollama);
    auto openai = providerDocument.value("openai").toObject();
    openai.insert("api_address", settings.baseUrl.toString());
    openai.insert("api_key", "mock-openai-secret");
    providerDocument.insert("openai", openai);
    auto modelDocument = readJson(modelsConfig);
    auto parameters = modelDocument.value("settings").toObject();
    parameters.insert("temperature", 13);
    parameters.insert("max_length", 9876543);
    parameters.insert("system_prompt", "Loaded system instruction");
    parameters.insert("vendor", "preserve-setting");
    modelDocument.insert("settings", parameters);
    modelDocument.insert("vendor", "preserve-model-document");
    modelDocument.insert(
            "models_data",
            QJsonObject{
                    {"ollama",
                     QJsonObject{{settings.model,
                                  QJsonObject{{"selected", true},
                                              {"favorite", true},
                                              {"vision", true},
                                              {"vendor", "preserve-model"}}},
                                 {"obsolete-model",
                                  QJsonObject{{"selected", false},
                                              {"favorite", false},
                                              {"vision", false}}}}}});
    QString configError;
    REQUIRE(AnnotationChatConfig::write("providers.json", providerDocument,
                                        &configError) &&
            AnnotationChatConfig::write("models.json", modelDocument,
                                        &configError));
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(readJson(modelsConfig) == modelDocument &&
            readJson(providerConfig) == providerDocument);
    saved.setValue("qAnyLabeling/chat/providers/ollama/address",
                   "http://invalid.localhost:1/v1");
    saved.setValue("qAnyLabeling/chat/providers/ollama/model", "stale-model");
    response = 0;
    auto* chatAction = window->findChild<QAction*>("AnnotationOpenChatbot");
    REQUIRE(chatAction);
    chatAction->trigger();
    auto* chat = window->findChild<QDialog*>("AnnotationChatbot");
    REQUIRE(chat);
    auto* chatProvider = chat->findChild<QComboBox*>("ChatProvider");
    auto* chatModel = chat->findChild<QComboBox*>("ChatModel");
    auto* chatAddress = chat->findChild<QLineEdit*>("ChatAddress");
    auto* chatKey = chat->findChild<QLineEdit*>("ChatApiKey");
    auto* chatSystem = chat->findChild<QLineEdit*>("ChatSystemPrompt");
    auto* chatTemperature = chat->findChild<QDoubleSpinBox*>("ChatTemperature");
    auto* chatTokens = chat->findChild<QSpinBox*>("ChatMaxTokens");
    REQUIRE(chatProvider && chatModel && chatAddress && chatKey && chatSystem &&
            chatTemperature && chatTokens &&
            chatProvider->currentText() == "ollama" &&
            chatModel->currentText() == settings.model &&
            chatAddress->text() == settings.baseUrl.toString() &&
            chatKey->text() == "json-secret" &&
            chatSystem->text() == "Loaded system instruction" &&
            chatTemperature->value() == 1.3 && chatTokens->value() == 9876543 &&
            chatTokens->maximum() == 9999999);
    QCoreApplication::processEvents();
    auto* chatSplitter = chat->findChild<QSplitter*>("ChatMainSplitter");
    auto* rightSplitter = chat->findChild<QSplitter*>("ChatRightSplitter");
    auto* settingsTabs = chat->findChild<QTabWidget*>("ChatSettingsTabs");
    auto* chatPreview = chat->findChild<QLabel*>("ChatImagePreview");
    REQUIRE(chatSplitter && chatSplitter->count() == 3 && rightSplitter &&
            rightSplitter->count() == 2 && settingsTabs &&
            settingsTabs->count() == 2 && chatPreview &&
            !chatPreview->property("pixmap").value<QPixmap>().isNull());
    auto* providersPanel = chatSplitter->widget(0);
    auto* conversationPanel = chatSplitter->widget(1);
    auto* previewSettingsPanel = chatSplitter->widget(2);
    const auto initialSizes = chatSplitter->sizes();
    chat->resize(chat->width() + 200, chat->height());
    QCoreApplication::processEvents();
    const auto expandedSizes = chatSplitter->sizes();
    REQUIRE(providersPanel->minimumWidth() == 200 &&
            providersPanel->maximumWidth() == 250 &&
            previewSettingsPanel->minimumWidth() == 360 &&
            previewSettingsPanel->maximumWidth() == 360 &&
            expandedSizes[0] == initialSizes[0] &&
            expandedSizes[2] == initialSizes[2] &&
            expandedSizes[1] == initialSizes[1] + 200);
    auto* providerButton =
            chat->findChild<QPushButton*>("ChatProviderButton_custom");
    auto* ollamaButton =
            chat->findChild<QPushButton*>("ChatProviderButton_ollama");
    REQUIRE(providerButton && ollamaButton && ollamaButton->isChecked());
    providerButton->click();
    REQUIRE(chatProvider->currentText() == "custom" &&
            providerButton->isChecked() && !ollamaButton->isChecked() &&
            chatModel->isVisible());
    chatProvider->setCurrentText("ollama");
    REQUIRE(ollamaButton->isChecked() && !providerButton->isChecked() &&
            !chatModel->isVisible());
    auto* temperatureSlider =
            chat->findChild<QSlider*>("ChatTemperatureSlider");
    REQUIRE(temperatureSlider && temperatureSlider->minimum() == 0 &&
            temperatureSlider->maximum() == 20 &&
            temperatureSlider->value() == 13);
    temperatureSlider->setValue(17);
    REQUIRE(chatTemperature->value() == 1.7 &&
            readJson(modelsConfig)
                            .value("settings")
                            .toObject()
                            .value("temperature") == 17);
    temperatureSlider->setValue(13);
    auto* visibility = chat->findChild<QPushButton*>("ChatApiKeyVisibility");
    REQUIRE(visibility && chatKey->echoMode() == QLineEdit::Password);
    visibility->click();
    REQUIRE(chatKey->echoMode() == QLineEdit::Normal);
    visibility->click();
    REQUIRE(chatKey->echoMode() == QLineEdit::Password);
    // Exercise the actual workspace theme actions while the child window is
    // open, including the workspace's subsequent inherited stylesheet update.
    QJsonArray themeSamples;
    for (const auto& mode : {QStringLiteral("dark"), QStringLiteral("light")}) {
        auto* action = window->findChild<QAction*>("AnnotationTheme_" + mode);
        REQUIRE(action);
        action->trigger();
        QCoreApplication::processEvents();
        const auto providerColor =
                providersPanel->grab().toImage().pixelColor(5, 5);
        const auto middleColor =
                conversationPanel->grab().toImage().pixelColor(5, 5);
        const auto providerText =
                ollamaButton->palette().color(QPalette::ButtonText);
        std::fprintf(stderr, "Chat theme %s: provider=%s middle=%s\n",
                     qPrintable(mode), qPrintable(providerColor.name()),
                     qPrintable(middleColor.name()));
        REQUIRE(providerColor ==
                        QColor(mode == "dark" ? "#2c2c2e" : "#f5f5f7") &&
                middleColor == QColor(mode == "dark" ? "#1c1c1e" : "#ffffff"));
        if (providerText != QColor(mode == "dark" ? "#f5f5f7" : "#1d1d1f"))
            std::fprintf(stderr, "Provider foreground %s; stylesheet=%s\n",
                         qPrintable(providerText.name()),
                         qPrintable(ollamaButton->styleSheet()));
        REQUIRE(providerText == QColor(mode == "dark" ? "#f5f5f7" : "#1d1d1f"));
        themeSamples.append(
                QJsonObject{{"mode", mode},
                            {"provider_surface", providerColor.name()},
                            {"provider_text", providerText.name()},
                            {"conversation_background", middleColor.name()}});
        if (argc == 3)
            REQUIRE(chat->grab().save(
                    QDir(QString::fromLocal8Bit(argv[2]))
                            .filePath("chat-window-" + mode + ".png")));
    }
    if (argc > 2) {
        QJsonArray columns;
        for (auto* panel :
             {providersPanel, conversationPanel, previewSettingsPanel}) {
            columns.append(
                    QJsonObject{{"name", panel->objectName()},
                                {"minimum_width", panel->minimumWidth()},
                                {"maximum_width", panel->maximumWidth()}});
        }
        const auto margins = chat->layout()->contentsMargins();
        QJsonObject layout{
                {"columns", columns},
                {"main_handle_width", chatSplitter->handleWidth()},
                {"right_handle_width", rightSplitter->handleWidth()},
                {"root_margins", QJsonArray{margins.left(), margins.top(),
                                            margins.right(), margins.bottom()}},
                {"root_spacing", chat->layout()->spacing()},
                {"settings_tabs", QJsonArray{settingsTabs->tabText(0),
                                             settingsTabs->tabText(1)}},
                {"theme_samples", themeSamples},
                {"preview_minimum_height", chatPreview->minimumHeight()},
                {"temperature_range",
                 QJsonArray{temperatureSlider->minimum(),
                            temperatureSlider->maximum()}}};
        QFile output(QDir(QString::fromLocal8Bit(argv[2]))
                             .filePath("chat-layout-native.json"));
        REQUIRE(output.open(QIODevice::WriteOnly));
        output.write(QJsonDocument(layout).toJson());
        REQUIRE(chat->grab().save(QDir(QString::fromLocal8Bit(argv[2]))
                                          .filePath("chat-window.png")));
    }
    chat->resize(chat->width() - 200, chat->height());
    QCoreApplication::processEvents();
    chatSystem->clear();
    QMetaObject::invokeMethod(chatSystem, "editingFinished",
                              Qt::DirectConnection);
    const auto savedParameters =
            readJson(modelsConfig).value("settings").toObject();
    REQUIRE(savedParameters.value("system_prompt").isNull() &&
            savedParameters.value("temperature") == 13 &&
            savedParameters.value("max_length") == 9876543 &&
            savedParameters.value("vendor") == "preserve-setting");
    chatProvider->setCurrentText("custom");
    chatAddress->setText(settings.baseUrl.toString());
    chatKey->setText("custom-secret");
    chatModel->setCurrentText("my-private-model");
    chatProvider->setCurrentText("ollama");
    // save_model_selection clears selected flags across all providers.
    REQUIRE(chatModel->currentText().isEmpty() &&
            chatAddress->text() == settings.baseUrl.toString() &&
            chatKey->text() == "json-secret" &&
            readJson(providerConfig)
                            .value("custom")
                            .toObject()
                            .value("model_name") == "my-private-model" &&
            readJson(providerConfig)
                            .value("ollama")
                            .toObject()
                            .value("vendor") == "preserve-provider");
    chatModel->setCurrentText(settings.model);
    QMetaObject::invokeMethod(chatModel->lineEdit(), "editingFinished",
                              Qt::DirectConnection);
    auto* refreshModels = chat->findChild<QPushButton*>("ChatRefreshModels");
    REQUIRE(refreshModels);
    const int beforeRefresh = modelRequests;
    refreshModels->click();
    REQUIRE(waitFor([&]() {
        return modelRequests > beforeRefresh &&
               chat->findChild<QLabel*>("ChatStatus")
                       ->text()
                       .contains("1 models");
    }));
    const auto cachedModels = readJson(modelsConfig)
                                      .value("models_data")
                                      .toObject()
                                      .value("ollama")
                                      .toObject();
    REQUIRE(cachedModels.size() == 1 &&
            !cachedModels.contains("obsolete-model") &&
            cachedModels.value(settings.model).toObject().value("favorite") ==
                    true &&
            cachedModels.value(settings.model).toObject().value("vendor") ==
                    "preserve-model" &&
            cachedModels.value(settings.model).toObject().value("selected") ==
                    true &&
            headers.contains("Authorization: Bearer json-secret"));
    auto* browseModels = chat->findChild<QPushButton*>("ChatBrowseModels");
    REQUIRE(browseModels);
    const int beforeBrowseModels = modelRequests;
    browseModels->click();
    auto* popup = chat->findChild<QWidget*>("ChatModelPopup");
    REQUIRE(popup);
    auto* settingsPanel = chat->findChild<QWidget*>("ChatPreviewSettingsPanel");
    REQUIRE(settingsPanel);
    const auto buttonPosition = browseModels->mapToGlobal(QPoint(0, 0));
    const auto panelPosition = settingsPanel->mapToGlobal(QPoint(0, 0));
    REQUIRE(popup->width() == settingsPanel->width() &&
            popup->height() == buttonPosition.y() - panelPosition.y() &&
            popup->y() == panelPosition.y());
    if (argc == 3) {
        const QJsonObject geometry{{"button_x", buttonPosition.x()},
                                   {"button_y", buttonPosition.y()},
                                   {"button_width", browseModels->width()},
                                   {"panel_top", panelPosition.y()},
                                   {"panel_width", settingsPanel->width()},
                                   {"initial_width", 360},
                                   {"popup_x", popup->x()},
                                   {"popup_y", popup->y()},
                                   {"popup_width", popup->width()},
                                   {"popup_height", popup->height()}};
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-model-geometry-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly) &&
                fixture.write(QJsonDocument(geometry).toJson()) > 0);
    }
    auto* search = popup->findChild<QLineEdit*>("ChatModelSearch");
    REQUIRE(search && search->height() == 32 &&
            search->font().pixelSize() == 11 &&
            search->placeholderText() == "Search models" &&
            popup->findChildren<QWidget*>("ChatModelRow").size() == 2 &&
            popup->findChildren<QLabel*>("ChatModelVision").size() == 2);
    // Ollama refreshes on every open, while the menu immediately shows its
    // existing cached flags. Discovery remains asynchronous.
    REQUIRE(waitFor([&]() {
        return modelRequests > beforeBrowseModels &&
               chat->findChild<QLabel*>("ChatStatus")
                       ->text()
                       .contains("1 models");
    }));
    auto* emptyModels = popup->findChild<QLabel*>("ChatModelEmpty");
    search->setText("zzzzzzzzzzzzzzzzzz");
    REQUIRE(emptyModels && !emptyModels->isHidden());
    for (auto* row : popup->findChildren<QWidget*>("ChatModelRow"))
        REQUIRE(row->isHidden());
    search->setText("VSON-TEST");
    REQUIRE(emptyModels->isHidden());
    auto* section = popup->findChild<QWidget*>("ChatModelProviderSection");
    REQUIRE(section && !section->isHidden());
    auto* favorite = section->findChild<QPushButton*>("ChatModelFavorite");
    REQUIRE(favorite && !favorite->icon().isNull());
    const auto beforeFavorite = readJson(modelsConfig);
    favorite->click();
    REQUIRE(waitFor([&]() {
        return !readJson(modelsConfig)
                        .value("models_data")
                        .toObject()
                        .value("ollama")
                        .toObject()
                        .value(settings.model)
                        .toObject()
                        .value("favorite")
                        .toBool();
    }));
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-model-favorite-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(
                QJsonDocument(QJsonObject{{"before", beforeFavorite},
                                          {"after", readJson(modelsConfig)},
                                          {"provider", "ollama"},
                                          {"model", settings.model}})
                        .toJson());
    }
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QCoreApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(popup->findChildren<QWidget*>("ChatModelRow").size() == 1);
    section = popup->findChild<QWidget*>("ChatModelProviderSection");
    favorite = section->findChild<QPushButton*>("ChatModelFavorite");
    favorite->click();
    QCoreApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(readJson(modelsConfig) == beforeFavorite);
    search->clear();
    if (argc == 3)
        REQUIRE(popup->grab().save(QDir(QString::fromLocal8Bit(argv[2]))
                                           .filePath("chat-model-popup.png")));
    section = popup->findChild<QWidget*>("ChatModelProviderSection");
    auto* modelRow = section->findChild<QWidget*>("ChatModelRow");
    REQUIRE(modelRow);
    QPointer<QWidget> popupGuard(popup);
    QMouseEvent chooseModel(QEvent::MouseButtonPress, QPointF(5, 5),
                            Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QCoreApplication::sendEvent(modelRow, &chooseModel);
    REQUIRE(chatModel->currentText() == settings.model && !popup->isVisible());
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(popupGuard.isNull());
    window->findChild<QAction*>("AnnotationTheme_dark")->trigger();
    QCoreApplication::processEvents();
    browseModels->click();
    auto* darkPopup = chat->findChild<QWidget*>("ChatModelPopup");
    REQUIRE(darkPopup && darkPopup->width() == settingsPanel->width() &&
            darkPopup->height() ==
                    browseModels->mapToGlobal(QPoint(0, 0)).y() -
                            settingsPanel->mapToGlobal(QPoint(0, 0)).y());
    QCoreApplication::processEvents();
    const auto darkShot = darkPopup->grab();
    REQUIRE(darkShot.toImage().pixelColor(darkShot.width() / 2,
                                          darkShot.height() * 9 / 10) ==
            QColor("#2c2c2e"));
    if (argc == 3)
        REQUIRE(darkShot.save(QDir(QString::fromLocal8Bit(argv[2]))
                                      .filePath("chat-model-popup-dark.png")));
    darkPopup->close();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    window->findChild<QAction*>("AnnotationTheme_light")->trigger();
    QCoreApplication::processEvents();
    REQUIRE(readJson(modelsConfig) == beforeFavorite);
    if (argc == 3) {
        QFile config(QDir(QString::fromLocal8Bit(argv[2]))
                             .filePath("chat-config-roundtrip-native.json"));
        REQUIRE(config.open(QIODevice::WriteOnly));
        config.write(
                QJsonDocument(
                        QJsonObject{{"providers", readJson(providerConfig)},
                                    {"models", readJson(modelsConfig)}})
                        .toJson());
    }
    const int beforeOpenai = modelRequests;
    chatProvider->setCurrentText("openai");
    browseModels->click();
    auto* firstOpenaiPopup = chat->findChild<QWidget*>("ChatModelPopup");
    REQUIRE(firstOpenaiPopup &&
            firstOpenaiPopup->findChildren<QWidget*>("ChatModelRow").isEmpty());
    REQUIRE(waitFor([&]() {
        return modelRequests > beforeOpenai &&
               firstOpenaiPopup->findChildren<QWidget*>("ChatModelRow")
                               .size() == 1 &&
               chat->findChild<QLabel*>("ChatStatus")
                       ->text()
                       .contains("1 models");
    }));
    firstOpenaiPopup->close();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    for (int i = 0; i < 2; ++i) {
        const int beforeCachedOpen = modelRequests;
        browseModels->click();
        auto* cachedPopup = chat->findChild<QWidget*>("ChatModelPopup");
        REQUIRE(cachedPopup);
        QCoreApplication::processEvents();
        REQUIRE(modelRequests == beforeCachedOpen);
        cachedPopup->close();
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    }
    const int beforeForcedRefresh = modelRequests;
    refreshModels->click();
    REQUIRE(waitFor([&]() {
        return modelRequests > beforeForcedRefresh &&
               chat->findChild<QLabel*>("ChatStatus")
                       ->text()
                       .contains("1 models");
    }));
    const int beforeOllamaReturn = modelRequests;
    chatProvider->setCurrentText("ollama");
    REQUIRE(waitFor([&]() {
        return modelRequests > beforeOllamaReturn &&
               chat->findChild<QLabel*>("ChatStatus")
                       ->text()
                       .contains("1 models");
    }));
    chatModel->setCurrentText(settings.model);
    QMetaObject::invokeMethod(chatModel->lineEdit(), "editingFinished",
                              Qt::DirectConnection);
    auto messageWidgets = [&]() {
        QVector<QWidget*> result;
        auto* view = chat->findChild<QScrollArea*>("ChatHistory");
        if (!view) return result;
        auto* layout = view->widget()->layout();
        for (int i = 0; i < layout->count(); ++i)
            if (auto* widget = layout->itemAt(i)->widget())
                if (widget->objectName() == "ChatMessage")
                    result.append(widget);
        return result;
    };
    auto historyText = [&]() {
        QString text;
        for (auto* widget : messageWidgets())
            text += widget->property("content").toString() + '\n';
        return text;
    };
    auto* input = chat->findChild<QTextEdit*>("ChatInput");
    auto* send = chat->findChild<QPushButton*>("ChatSend");
    REQUIRE(input && send && !send->isEnabled() && !input->acceptRichText() &&
            input->document()->documentMargin() == 0 &&
            input->minimumHeight() >= 20 && input->maximumHeight() <= 300 &&
            !chat->findChild<QPushButton*>("ChatStop") &&
            !chat->findChild<QCheckBox*>("ChatAttachImage"));
    input->setPlainText(QString(80, '\n'));
    REQUIRE(input->height() == 300 &&
            input->verticalScrollBarPolicy() == Qt::ScrollBarAlwaysOn &&
            !send->isEnabled());
    input->setPlainText("Short message");
    REQUIRE(input->height() < 300 && send->isEnabled() &&
            input->verticalScrollBarPolicy() == Qt::ScrollBarAlwaysOff);
    input->setPlainText("line one");
    input->moveCursor(QTextCursor::End);
    const int beforeNewline = requests;
    QKeyEvent newline(QEvent::KeyPress, Qt::Key_Return, Qt::ShiftModifier);
    QCoreApplication::sendEvent(input, &newline);
    REQUIRE(input->toPlainText() == "line one\n" && requests == beforeNewline);
    input->setPlainText("@image Describe");
    auto tagCursor = input->textCursor();
    tagCursor.setPosition(2);
    REQUIRE(tagCursor.charFormat().background().color() == QColor("#E3F2FD") &&
            tagCursor.charFormat().foreground().color() == QColor("#1976D2"));
    QKeyEvent enter(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
    QCoreApplication::sendEvent(input, &enter);
    REQUIRE(waitFor([&]() { return input->isEnabled(); }));
    REQUIRE(!send->isEnabled() && send->toolTip() == "Send");
    REQUIRE(historyText().contains(QString::fromUtf8("图像答案")));
    auto history = readJson(temp.path() + "/second.json")
                           .value("chat_history")
                           .toArray();
    REQUIRE(history.size() == 2 &&
            history.first().toObject().value("content").isString());
    REQUIRE(history.first().toObject().value("image") == secondPath &&
            history.last().toObject().value("image").isNull());
    input->setPlainText("Follow up");
    const int beforeSettingsEnter = requests;
    QCoreApplication::sendEvent(chatKey, &enter);
    REQUIRE(requests == beforeSettingsEnter &&
            input->toPlainText() == "Follow up");
    send->click();
    REQUIRE(waitFor([&]() { return input->isEnabled(); }) &&
            received.value("messages").toArray().size() == 3);
    auto* exportDataset = chat->findChild<QPushButton*>("ChatExportDataset");
    REQUIRE(exportDataset);
    const QString datasetDirectory =
            QDir(temp.path()).filePath("dataset-export");
    REQUIRE(QDir().mkpath(datasetDirectory));
    auto exportShareGpt = [&]() {
        bool chosen = false;
        QTimer choose;
        QObject::connect(&choose, &QTimer::timeout, &choose, [&]() {
            auto* file = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget());
            if (!file || chosen) return;
            chosen = true;
            file->setDirectory(datasetDirectory);
            file->findChild<QLineEdit*>("fileNameEdit")
                    ->setText(datasetDirectory);
            QMetaObject::invokeMethod(file, "accept", Qt::QueuedConnection);
        });
        choose.start(10);
        exportDataset->click();
        return chosen;
    };
    REQUIRE(exportShareGpt());
    const auto archives =
            QDir(datasetDirectory).entryList({"*.zip"}, QDir::Files);
    REQUIRE(archives.size() == 1 && chat->findChild<QLabel*>("ChatStatus")
                                            ->text()
                                            .contains("successfully"));
    if (argc == 3) {
        const QDir evidence(QString::fromLocal8Bit(argv[2]));
        const QString fixtures = evidence.filePath("sharegpt-input");
        REQUIRE(QDir().mkpath(fixtures));
        for (const auto& name : QStringList{"first.png", "second.png",
                                            "first.json", "second.json"})
            REQUIRE(QFile::copy(QDir(temp.path()).filePath(name),
                                QDir(fixtures).filePath(name)));
        REQUIRE(QFile::copy(QDir(datasetDirectory).filePath(archives.first()),
                            evidence.filePath("sharegpt-native.zip")));
    }
    REQUIRE(sidecar.open(QIODevice::ReadOnly));
    const auto beforeInvalidExport = sidecar.readAll();
    sidecar.close();
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write("invalid annotation");
    sidecar.close();
    REQUIRE(exportShareGpt());
    REQUIRE(chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("Invalid annotation") &&
            QDir(datasetDirectory).entryList({"*.zip"}, QDir::Files) ==
                    archives);
    REQUIRE(sidecar.open(QIODevice::WriteOnly | QIODevice::Truncate));
    sidecar.write(beforeInvalidExport);
    sidecar.close();
    const auto initialMessageHistory = readJson(temp.path() + "/second.json")
                                               .value("chat_history")
                                               .toArray();
    REQUIRE(messageWidgets().size() == 4);
    QJsonObject messageFixture{{"initial", initialMessageHistory}};
    auto messageButton = [&](int index, const char* name) {
        return messageWidgets()[index]->findChild<QPushButton*>(name);
    };
    REQUIRE(!messageWidgets()[0]->findChild<QLabel*>("ChatMessageRoleIcon") &&
            messageWidgets()[1]->findChild<QLabel*>("ChatMessageRoleIcon") &&
            messageWidgets()[1]
                            ->findChild<QLabel*>("ChatMessageRoleIcon")
                            ->property("provider") == "ollama");
    QJsonArray messageThemes;
    for (const auto& mode : {QStringLiteral("dark"), QStringLiteral("light")}) {
        window->findChild<QAction*>("AnnotationTheme_" + mode)->trigger();
        QCoreApplication::processEvents();
        auto* userBubble =
                messageWidgets()[0]->findChild<QWidget*>("messageBubble");
        const auto surface = userBubble->grab().toImage().pixelColor(
                userBubble->width() / 2, userBubble->height() - 8);
        const auto textColor =
                messageWidgets()[0]
                        ->findChild<QLabel*>("ChatMessageContent")
                        ->palette()
                        .color(QPalette::WindowText);
        REQUIRE(surface == QColor(mode == "dark" ? "#2c2c2e" : "#f5f5f7") &&
                textColor == QColor(mode == "dark" ? "#f5f5f7" : "#1d1d1f"));
        messageThemes.append(QJsonObject{{"mode", mode},
                                         {"surface", surface.name()},
                                         {"text", textColor.name()}});
        if (argc == 3)
            REQUIRE(chat->grab().save(
                    QDir(QString::fromLocal8Bit(argv[2]))
                            .filePath("chat-messages-" + mode + ".png")));
    }
    messageFixture.insert("themes", messageThemes);
    QJsonArray messageMenus;
    auto messageMenu = [&](int index, const char* actionName) {
        QJsonArray items;
        bool selected = false;
        auto* widget = messageWidgets()[index];
        QTimer::singleShot(0, widget, [&]() {
            auto* menu =
                    qobject_cast<QMenu*>(QApplication::activePopupWidget());
            if (!menu || menu->objectName() != "ChatMessageMenu") return;
            for (auto* action : menu->actions()) {
                if (!action->isSeparator()) items.append(action->text());
                if (action->objectName() == actionName) {
                    selected = action->isEnabled();
                    action->trigger();
                }
            }
            menu->close();
        });
        QMetaObject::invokeMethod(widget, "customContextMenuRequested",
                                  Qt::DirectConnection,
                                  Q_ARG(QPoint, QPoint(20, 20)));
        messageMenus.append(
                QJsonObject{{"role", widget->property("role").toString()},
                            {"items", items}});
        return selected;
    };
    REQUIRE(messageMenu(1, "ChatMessageMenuCopy") &&
            QApplication::clipboard()->text() ==
                    messageWidgets()[1]->property("content").toString() &&
            !messageWidgets()[1]->property("copyFeedback").toBool());
    REQUIRE(messageMenu(0, "ChatMessageMenuEdit") &&
            messageWidgets()[0]
                    ->findChild<QPlainTextEdit*>("ChatMessageEditor")
                    ->isVisible());
    messageButton(0, "ChatMessageCancel")->click();
    messageButton(0, "ChatMessageCopy")->click();
    REQUIRE(messageWidgets()[0]->property("copyFeedback").toBool());
    const auto* copyReset =
            messageWidgets()[0]->findChild<QTimer*>("ChatMessageCopyReset");
    REQUIRE(copyReset && copyReset->interval() == 1000 &&
            copyReset->isSingleShot());
    REQUIRE(waitFor([&]() {
        return !messageWidgets()[0]->property("copyFeedback").toBool();
    }));
    messageFixture.insert("menus", messageMenus);
    messageFixture.insert("copy_feedback_ms", copyReset->interval());
    REQUIRE(QApplication::clipboard()->text() == "<image> Describe");
    const int beforeEdit = requests;
    const QString editedAnswer = "**Edited answer**\n\n```cpp\nint x = 1;\n```";
    messageButton(1, "ChatMessageEdit")->click();
    auto* messageEditor = messageWidgets()[1]->findChild<QPlainTextEdit*>(
            "ChatMessageEditor");
    REQUIRE(messageEditor->isVisible() &&
            !messageButton(1, "ChatMessageResend")->isVisible());
    messageEditor->setPlainText("discard this edit");
    messageButton(1, "ChatMessageCancel")->click();
    REQUIRE(readJson(temp.path() + "/second.json").value("chat_history") ==
            initialMessageHistory);
    messageButton(1, "ChatMessageEdit")->click();
    messageEditor->setPlainText(editedAnswer);
    messageButton(1, "ChatMessageSave")->click();
    auto afterEdit = readJson(temp.path() + "/second.json")
                             .value("chat_history")
                             .toArray();
    REQUIRE(afterEdit.size() == 4 && requests == beforeEdit &&
            afterEdit[1].toObject().value("content") == editedAnswer &&
            afterEdit[1].toObject().value("image").isNull() &&
            afterEdit[2] == initialMessageHistory[2]);
    auto* renderedAnswer =
            messageWidgets()[1]->findChild<QLabel*>("ChatMessageContent");
    REQUIRE(renderedAnswer->textFormat() == Qt::RichText &&
            renderedAnswer->text().contains("font-weight:600") &&
            renderedAnswer->text().contains("int x = 1;"));
    messageButton(1, "ChatMessageCopy")->click();
    REQUIRE(QApplication::clipboard()->text() == editedAnswer);
    messageFixture.insert("saved", afterEdit);
    messageFixture.insert("edited_content", editedAnswer);
    auto confirmDelete = [&](int index, bool yes) {
        QTimer::singleShot(0, chat, [yes]() {
            auto* question = qobject_cast<QMessageBox*>(
                    QApplication::activeModalWidget());
            if (question)
                question->button(yes ? QMessageBox::Yes : QMessageBox::No)
                        ->click();
        });
        messageButton(index, "ChatMessageDelete")->click();
    };
    confirmDelete(3, false);
    REQUIRE(readJson(temp.path() + "/second.json").value("chat_history") ==
            afterEdit);
    confirmDelete(3, true);
    const auto afterDelete = readJson(temp.path() + "/second.json")
                                     .value("chat_history")
                                     .toArray();
    REQUIRE(afterDelete.size() == 3 && afterDelete[0] == afterEdit[0] &&
            afterDelete[1] == afterEdit[1] && afterDelete[2] == afterEdit[2]);
    messageFixture.insert("deleted", afterDelete);
    before = requests;
    messageButton(1, "ChatMessageRegenerate")->click();
    REQUIRE(waitFor([&]() { return input->isEnabled(); }) &&
            requests == before + 1 &&
            received.value("messages").toArray().size() == 1);
    auto regenerated = readJson(temp.path() + "/second.json")
                               .value("chat_history")
                               .toArray();
    REQUIRE(regenerated.size() == 2 &&
            regenerated[0] == initialMessageHistory[0] &&
            regenerated[1].toObject().value("content") ==
                    QString::fromUtf8("图像答案"));
    messageFixture.insert("regenerated", regenerated);
    messageFixture.insert("regenerate_request", received);
    before = requests;
    messageButton(0, "ChatMessageEdit")->click();
    REQUIRE(messageButton(0, "ChatMessageResend")->isVisible());
    messageButton(0, "ChatMessageResend")->click();
    REQUIRE(requests == before);
    messageButton(0, "ChatMessageEdit")->click();
    messageWidgets()[0]
            ->findChild<QPlainTextEdit*>("ChatMessageEditor")
            ->setPlainText("@image Edited request");
    messageButton(0, "ChatMessageResend")->click();
    REQUIRE(waitFor([&]() { return input->isEnabled(); }) &&
            requests == before + 1 &&
            received.value("messages").toArray().size() == 1);
    const auto resent = readJson(temp.path() + "/second.json")
                                .value("chat_history")
                                .toArray();
    REQUIRE(resent.size() == 2 &&
            resent[0].toObject().value("content") == "<image> Edited request" &&
            resent[0].toObject().value("image") == secondPath);
    messageFixture.insert("resent", resent);
    messageFixture.insert("resend_request", received);
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-message-controls-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(QJsonDocument(messageFixture).toJson());
    }
    response = 4;
    input->setPlainText("Stop me");
    before = requests;
    send->click();
    REQUIRE(waitFor([&]() { return requests > before; }));
    REQUIRE(!input->isEnabled() && send->isEnabled() &&
            send->toolTip() == "Stop");
    send->click();
    REQUIRE(input->isEnabled() && !send->isEnabled());
    input->setPlainText("Stale answer");
    before = requests;
    send->click();
    REQUIRE(waitFor([&]() { return requests > before; }));
    QPointer<QWidget> staleMessage = messageWidgets().first();
    bool menuSourceChanged = false;
    QTimer::singleShot(0, chat, [&]() {
        menuSourceChanged = QMetaObject::invokeMethod(
                window, "openImage", Qt::DirectConnection,
                Q_ARG(QString, imagePath));
        QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    });
    QMetaObject::invokeMethod(staleMessage, "customContextMenuRequested",
                              Qt::DirectConnection,
                              Q_ARG(QPoint, QPoint(20, 20)));
    REQUIRE(menuSourceChanged && !staleMessage);
    REQUIRE(input->isEnabled() && historyText().isEmpty());
    REQUIRE(!readJson(sidecar.fileName()).contains("chat_history"));
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, secondPath)));
    REQUIRE(historyText().contains(QString::fromUtf8("图像答案")));
    auto* autoRun = chat->findChild<QPushButton*>("ChatAutoRun");
    auto* batchProgress =
            chat->findChild<QProgressDialog*>("ChatBatchProgress");
    REQUIRE(autoRun && batchProgress);
    auto launchChatBatch = [&]() {
        bool selected = false;
        QTimer::singleShot(0, chat, [&]() {
            auto* options =
                    qobject_cast<QDialog*>(QApplication::activeModalWidget());
            if (!options || options->objectName() != "ChatBatchOptions") return;
            auto* concurrency =
                    options->findChild<QSpinBox*>("ChatBatchConcurrency");
            const int cores = std::max(1, QThread::idealThreadCount());
            selected =
                    concurrency->maximum() == std::max(1, int(cores * .95)) &&
                    concurrency->value() == std::max(1, int(cores * .8));
            concurrency->setValue(2);
            options->findChild<QPlainTextEdit*>("ChatBatchPrompt")
                    ->setPlainText("@image  Describe this image");
            options->findChild<QDialogButtonBox*>()
                    ->button(QDialogButtonBox::Ok)
                    ->click();
        });
        autoRun->click();
        return selected;
    };
    response = 0;
    before = requests;
    const int previousRequestCount = receivedRequests.size();
    REQUIRE(launchChatBatch());
    REQUIRE(waitFor([&]() { return autoRun->isEnabled(); }) &&
            requests == before + 2);
    REQUIRE(!batchProgress->isVisible() && input->isEnabled() &&
            chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("0 images failed"));
    for (const auto& path : QStringList{sidecar.fileName(), otherSidecar}) {
        const auto result = readJson(path);
        const auto history = result.value("chat_history").toArray();
        REQUIRE(history.size() == 2 &&
                history.first().toObject().value("content") ==
                        "<image>Describe this image" &&
                history.last().toObject().value("content") ==
                        "Generated answer" &&
                history.last().toObject().value("image").isNull());
        REQUIRE(result.value("vendor") ==
                (path == sidecar.fileName() ? "keep" : "other"));
    }
    for (int i = previousRequestCount; i < receivedRequests.size(); ++i) {
        const auto request = receivedRequests[i];
        REQUIRE(!request.value("stream").toBool());
        const auto messages = request.value("messages").toArray();
        REQUIRE(messages.size() == 1 &&
                messages.first().toObject().value("content").toArray().size() ==
                        2);
    }
    const auto firstBatch = readJson(sidecar.fileName());
    const auto secondBatch = readJson(otherSidecar);
    response = 1;
    REQUIRE(launchChatBatch() &&
            waitFor([&]() { return autoRun->isEnabled(); }));
    REQUIRE(chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("2 images failed") &&
            readJson(sidecar.fileName()) == firstBatch &&
            readJson(otherSidecar) == secondBatch);
    response = 4;
    before = requests;
    REQUIRE(launchChatBatch() &&
            waitFor([&]() { return requests == before + 2; }));
    batchProgress->findChild<QPushButton*>()->click();
    REQUIRE(autoRun->isEnabled() && input->isEnabled() &&
            readJson(sidecar.fileName()) == firstBatch &&
            readJson(otherSidecar) == secondBatch);
    before = requests;
    REQUIRE(launchChatBatch() &&
            waitFor([&]() { return requests == before + 2; }));
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath)));
    REQUIRE(autoRun->isEnabled() && !batchProgress->isVisible() &&
            readJson(sidecar.fileName()) == firstBatch &&
            readJson(otherSidecar) == secondBatch);
    if (argc == 3) {
        QFile fixture(QDir(QString::fromLocal8Bit(argv[2]))
                              .filePath("chat-batch-native.json"));
        REQUIRE(fixture.open(QIODevice::WriteOnly));
        fixture.write(
                QJsonDocument(QJsonArray{firstBatch, secondBatch}).toJson());
    }
    // ShareGPT import replaces annotation documents, as the source importer
    // does, and copies only the first available image of each record.
    const QString importRoot = temp.path() + "/sharegpt-import";
    const QString importInput = importRoot + "/input";
    const QString importTarget = importRoot + "/native";
    REQUIRE(QDir().mkpath(importInput + "/images") &&
            QDir().mkpath(importTarget));
    REQUIRE(QFile::copy(imagePath, importTarget + "/anchor.png") &&
            QFile::copy(secondPath, importInput + "/images/added.png"));
    QFile oldImported(importTarget + "/anchor.json");
    REQUIRE(oldImported.open(QIODevice::WriteOnly));
    oldImported.write(
            QJsonDocument(QJsonObject{{"imagePath", "anchor.png"},
                                      {"vendor", "replace on import"},
                                      {"flags", QJsonObject{{"old", true}}},
                                      {"shapes", QJsonArray()}})
                    .toJson());
    oldImported.close();
    const QJsonArray importMessages{
            QJsonObject{{"role", "system"}, {"content", "System instruction"}},
            QJsonObject{{"role", "user"}, {"content", "<image> Describe"}},
            QJsonObject{{"role", "assistant"},
                        {"content", QString::fromUtf8("图像答案")}},
            QJsonObject{{"role", "user"}, {"content", "Follow up"}}};
    const QJsonArray importRecords{
            QJsonObject{{"messages", importMessages},
                        {"images", QJsonArray{"images/anchor.png"}}},
            QJsonObject{{"messages", importMessages},
                        {"images",
                         QJsonArray{"images/missing.png", "images/added.png"}}},
            QJsonObject{{"messages", importMessages},
                        {"images", QJsonArray()}}};
    QFile importFile(importInput + "/dataset.json");
    REQUIRE(importFile.open(QIODevice::WriteOnly));
    importFile.write(QJsonDocument(importRecords).toJson());
    importFile.close();
    const QString anchor = importTarget + "/anchor.png";
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, anchor)));
    auto* importDataset = chat->findChild<QPushButton*>("ChatImportDataset");
    REQUIRE(importDataset);
    auto importShareGpt = [&]() {
        bool chosen = false;
        QTimer choose;
        QObject::connect(&choose, &QTimer::timeout, &choose, [&]() {
            auto* dialog = qobject_cast<QFileDialog*>(
                    QApplication::activeModalWidget());
            if (!dialog || chosen) return;
            chosen = true;
            dialog->setDirectory(importInput);
            dialog->findChild<QLineEdit*>("fileNameEdit")
                    ->setText(importFile.fileName());
            QMetaObject::invokeMethod(dialog, "accept", Qt::QueuedConnection);
        });
        choose.start(10);
        importDataset->click();
        return chosen;
    };
    REQUIRE(importShareGpt());
    const auto anchorDocument = readJson(importTarget + "/anchor.json");
    const auto addedDocument = readJson(importTarget + "/added.json");
    REQUIRE(chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("Successfully imported 2") &&
            anchorDocument.value("version") == "4.2.1" &&
            addedDocument.value("version") == "4.2.1" &&
            readJson(sidecar.fileName()).value("version") == "4.1.3" &&
            !anchorDocument.contains("vendor") &&
            anchorDocument.value("flags").toObject().isEmpty() &&
            addedDocument.value("imagePath") == "added.png" &&
            addedDocument.value("imageWidth") == 80 &&
            addedDocument.value("imageHeight") == 60 &&
            addedDocument.value("chat_history").toArray().size() == 4 &&
            historyText().contains("System instruction"));
    REQUIRE(importFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    auto invalidRecords = importRecords;
    invalidRecords.append(QJsonObject{{"messages", "invalid"},
                                      {"images", QJsonArray{"added.png"}}});
    importFile.write(QJsonDocument(invalidRecords).toJson());
    importFile.close();
    REQUIRE(importShareGpt() &&
            chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("Invalid ShareGPT") &&
            readJson(importTarget + "/anchor.json") == anchorDocument &&
            readJson(importTarget + "/added.json") == addedDocument);
    REQUIRE(importFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    importFile.write(
            QJsonDocument(
                    QJsonArray{QJsonObject{
                            {"messages", importMessages},
                            {"images", QJsonArray{"../native/anchor.png"}}}})
                    .toJson());
    importFile.close();
    REQUIRE(importShareGpt() &&
            chat->findChild<QLabel*>("ChatStatus")
                    ->text()
                    .contains("Invalid dataset image") &&
            readJson(importTarget + "/anchor.json") == anchorDocument);
    REQUIRE(importFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    importFile.write(QJsonDocument(importRecords).toJson());
    importFile.close();
    if (argc == 3) {
        const QDir evidence(QString::fromLocal8Bit(argv[2]));
        const QString target = evidence.filePath("sharegpt-import");
        REQUIRE(QDir().mkpath(target + "/input/images") &&
                QDir().mkpath(target + "/native"));
        for (const auto& relative :
             QStringList{"input/dataset.json", "input/images/added.png",
                         "native/anchor.png", "native/added.png",
                         "native/anchor.json", "native/added.json"})
            REQUIRE(QFile::copy(importRoot + '/' + relative,
                                target + '/' + relative));
    }
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath)));
    response = 7;
    before = requests;
    input->setPlainText("Close before response");
    send->click();
    REQUIRE(waitFor([&]() { return requests > before; }));
    const auto beforeEscape = readJson(sidecar.fileName());
    // The test and loaded plugin each link this QObject class. Match the
    // actual plugin object's meta-object name across the DSO boundary.
    ChatServiceClient* activeChatClient = nullptr;
    for (auto* object : chat->findChildren<QObject*>())
        if (object->objectName() == "ChatCompletionClient" &&
            QByteArray(object->metaObject()->className()) ==
                    "ChatServiceClient")
            activeChatClient = static_cast<ChatServiceClient*>(object);
    REQUIRE(activeChatClient && activeChatClient->busy());
    const int beforeConcurrentDiscovery = modelRequests;
    refreshModels->click();
    REQUIRE(waitFor([&]() {
                return modelRequests > beforeConcurrentDiscovery;
            }) &&
            activeChatClient->busy());
    QApplication::sendEvent(chat, &escape);
    REQUIRE(!chat->isVisible() && !activeChatClient->busy());
    QPointer<QDialog> closedChat(chat);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(!closedChat);
    QElapsedTimer lateResponse;
    lateResponse.start();
    REQUIRE(waitFor([&]() { return lateResponse.elapsed() > 250; }) &&
            readJson(sidecar.fileName()) == beforeEscape);
    REQUIRE(QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                      Q_ARG(QString, imagePath)));
    auto* classifierAction =
            window->findChild<QAction*>("AnnotationImageClassifierAction");
    REQUIRE(classifierAction);
    bool classifierValid = false;
    QTimer::singleShot(0, window, [&]() {
        auto* dialog =
                qobject_cast<QDialog*>(QApplication::activeModalWidget());
        if (!dialog) return;
        auto* ai = dialog->findChild<QPushButton*>("ClassifierAI");
        auto* autoRun = dialog->findChild<QPushButton*>("ClassifierAutoRun");
        auto* status = dialog->findChild<QLabel*>("ClassifierAIStatus");
        auto* progress =
                dialog->findChild<QProgressDialog*>("ClassifierVLMProgress");
        auto launch = [&](bool batch) {
            bool selected = false;
            QTimer choose;
            choose.setInterval(5);
            QObject::connect(&choose, &QTimer::timeout, [&]() {
                auto* active = qobject_cast<QDialog*>(
                        QApplication::activeModalWidget());
                if (active &&
                    active->objectName() == "AnnotationClassifierPrompt") {
                    selected = true;
                    active->findChild<QDialogButtonBox*>()
                            ->button(QDialogButtonBox::Ok)
                            ->click();
                } else if (auto* question = qobject_cast<QMessageBox*>(active))
                    question->button(QMessageBox::Yes)->click();
            });
            choose.start();
            (batch ? autoRun : ai)->click();
            return selected;
        };
        auto check = [&](bool value, const char* message) {
            if (!value) std::fprintf(stderr, "Classifier VLM: %s\n", message);
            return value;
        };
        if (!check(ai && autoRun && status && progress, "controls")) {
            dialog->reject();
            return;
        }
        response = 5;
        if (!check(launch(false) && waitFor([&]() { return ai->isEnabled(); }),
                   "single image completion")) {
            dialog->reject();
            return;
        }
        const auto flags =
                readJson(sidecar.fileName()).value("flags").toObject();
        if (!check(flags == QJsonObject{{"cat", true}, {"dog", false}} &&
                           readJson(sidecar.fileName()).value("vendor") ==
                                   "keep",
                   "custom categories and metadata")) {
            dialog->reject();
            return;
        }
        const auto messages = received.value("messages").toArray();
        if (!check(!received.value("stream").toBool() &&
                           messages.last().toObject()
                                           .value("content")
                                           .toArray()
                                           .size() == 2 &&
                           messages.last()
                                   .toObject()
                                   .value("content")
                                   .toArray()
                                   .last()
                                   .toObject()
                                   .value("text")
                                   .toString()
                                   .contains("multi-class"),
                   "VLM image and source prompt")) {
            dialog->reject();
            return;
        }
        response = 6;
        if (!check(launch(false) &&
                           waitFor([&]() { return ai->isEnabled(); }) &&
                           readJson(sidecar.fileName())
                                           .value("flags")
                                           .toObject() == flags &&
                           status->text().contains("parse"),
                   "invalid AI flags preserve annotation")) {
            dialog->reject();
            return;
        }
        response = 4;
        before = requests;
        if (!check(launch(false) &&
                           waitFor([&]() { return requests > before; }),
                   "cancellable request")) {
            dialog->reject();
            return;
        }
        progress->findChild<QPushButton*>()->click();
        if (!check(ai->isEnabled() && readJson(sidecar.fileName())
                                                      .value("flags")
                                                      .toObject() == flags,
                   "cancel preserves flags")) {
            dialog->reject();
            return;
        }
        before = requests;
        if (!check(launch(false) &&
                           waitFor([&]() { return requests > before; }),
                   "stale source request")) {
            dialog->reject();
            return;
        }
        QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                  Q_ARG(QString, secondPath));
        if (!check(ai->isEnabled(), "source change cancels")) {
            dialog->reject();
            return;
        }
        QMetaObject::invokeMethod(window, "openImage", Qt::DirectConnection,
                                  Q_ARG(QString, imagePath));
        response = 5;
        dialog->findChild<QPushButton*>("ClassifierMultiLabel")->click();
        before = requests;
        if (!check(launch(true) && waitFor([&]() { return ai->isEnabled(); }) &&
                           requests - before == 2,
                   "two image batch completion")) {
            dialog->reject();
            return;
        }
        if (!check(readJson(temp.path() + "/second.json")
                                           .value("flags")
                                           .toObject() == flags &&
                           readJson(temp.path() + "/second.json")
                                           .value("chat_history")
                                           .toArray() ==
                                   secondBatch.value("chat_history")
                                           .toArray() &&
                           status->text().contains("0 failed"),
                   "batch preserves chat and sidecar fields")) {
            dialog->reject();
            return;
        }
        // A malformed existing sidecar is an explicit failed batch row.
        QFile broken(temp.path() + "/second.json");
        if (!check(broken.open(QIODevice::WriteOnly), "prepare bad sidecar")) {
            dialog->reject();
            return;
        }
        broken.write("broken sidecar");
        broken.close();
        if (!check(launch(true) && waitFor([&]() { return ai->isEnabled(); }) &&
                           status->text().contains("1 failed"),
                   "batch reports invalid sidecar")) {
            dialog->reject();
            return;
        }
        if (!check(broken.open(QIODevice::ReadOnly) &&
                           broken.readAll() == "broken sidecar",
                   "batch never overwrites malformed sidecar")) {
            dialog->reject();
            return;
        }
        classifierValid = true;
        dialog->reject();
    });
    classifierAction->trigger();
    REQUIRE(classifierValid);
    chatAction->trigger();
    auto* clearChat = window->findChild<QDialog*>("AnnotationChatbot");
    REQUIRE(clearChat);
    const auto beforeClear = readJson(sidecar.fileName());
    REQUIRE(!beforeClear.value("chat_history").toArray().isEmpty());
    const auto confirmClear = [&](QMessageBox::StandardButton answer) {
        bool valid = false;
        QTimer::singleShot(0, clearChat, [&]() {
            auto* confirm = qobject_cast<QMessageBox*>(
                    QApplication::activeModalWidget());
            if (!confirm) return;
            valid = confirm->defaultButton() ==
                            confirm->button(QMessageBox::No) &&
                    confirm->icon() == QMessageBox::Warning;
            confirm->button(answer)->click();
        });
        clearChat->findChild<QPushButton*>("ChatClear")->click();
        return valid;
    };
    REQUIRE(confirmClear(QMessageBox::No) &&
            readJson(sidecar.fileName()) == beforeClear);
    REQUIRE(confirmClear(QMessageBox::Yes));
    auto afterClear = readJson(sidecar.fileName());
    REQUIRE(afterClear.value("chat_history").toArray().isEmpty());
    auto remainingBeforeClear = beforeClear;
    remainingBeforeClear.remove("chat_history");
    afterClear.remove("chat_history");
    REQUIRE(afterClear == remainingBeforeClear);
    clearChat->close();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    // A malformed upstream file must survive initialization, settings edits
    // and dialog destruction byte-for-byte; old QSettings cannot replace it.
    QFile brokenModels(modelsConfig);
    REQUIRE(brokenModels.open(QIODevice::WriteOnly | QIODevice::Truncate));
    brokenModels.write("{broken model config");
    brokenModels.close();
    chat = nullptr;  // Escape already deleted the WA_DeleteOnClose dialog.
    chatAction->trigger();
    auto* brokenChat = window->findChild<QDialog*>("AnnotationChatbot");
    REQUIRE(brokenChat && brokenChat->findChild<QLabel*>("ChatStatus")
                                  ->text()
                                  .contains("original file preserved"));
    brokenChat->findChild<QLineEdit*>("ChatApiKey")->setText("must-not-write");
    QPointer<QDialog> closedBrokenChat(brokenChat);
    brokenChat->close();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    REQUIRE(!closedBrokenChat);
    REQUIRE(brokenModels.open(QIODevice::ReadOnly) &&
            brokenModels.readAll() == "{broken model config");
    brokenModels.close();
    window->close();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    std::puts(
            "Chat/VQA: wire protocol, UTF-8 SSE, images, errors, timeout, "
            "cancel, four component types, sidecar preservation, preview/apply "
            "and stale source; VLM custom categories, source prompts, single/"
            "batch, invalid JSON, cancellation and sidecar preservation; "
            "Chat bounded AutoRun, failed requests, cancellation and stale "
            "batch preservation; ShareGPT import/export, invalid-record "
            "preflight and path confinement; upstream provider/models JSON, "
            "legacy migration, selected/favorite flags, private permissions "
            "and "
            "malformed configuration preservation PASS");
    return 0;
}
