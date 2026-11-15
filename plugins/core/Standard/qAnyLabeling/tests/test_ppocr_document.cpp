// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QGraphicsView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QTableWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTextEdit>
#include <QTimer>
#include <QToolButton>
#include <cstdio>
#include <functional>

#include "AnnotationArchive.h"
#include "PpocrDialog.h"
#include "PpocrDocument.h"
#include "PpocrRichTextEditor.h"
#include "PpocrServiceClient.h"
#include "PpocrTableEditor.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)
namespace {
bool waitFor(const std::function<bool()>& predicate, int timeout = 4000) {
    QElapsedTimer elapsed;
    elapsed.start();
    while (!predicate() && elapsed.elapsed() < timeout)
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    return predicate();
}
struct Request {
    QByteArray headers, body;
};
}  // namespace

int main(int argc, char** argv) {
    QApplication application(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    const QJsonObject block{{"block_id", 4},
                            {"block_label", "Text"},
                            {"block_content", "old text"},
                            {"block_bbox", QJsonArray{10, 20, 50, 80}},
                            {"unrecognized_attribute", 17}};
    const QJsonObject page{
            {"prunedResult",
             QJsonObject{{"width", 100},
                         {"height", 120},
                         {"parsing_res_list", QJsonArray{block, block}}}},
            {"markdown", QJsonObject{{"text", "old text"}}}};
    const QJsonObject original{{"layoutParsingResults", QJsonArray{page}},
                               {"future_schema", "preserved"}};
    QString error;
    auto document =
            PpocrDocument::parse(QJsonDocument(original).toJson(), &error);
    REQUIRE(error.isEmpty() && document.value("future_schema") == "preserved");
    const auto blocks = PpocrDocument::blocks(document);
    REQUIRE(blocks.size() == 2 && blocks[0].key == "page_1:block_4" &&
            blocks[1].key == "page_1:block_4__dup_2");
    REQUIRE(blocks[0].polygon.size() == 4 &&
            blocks[0].polygon[2] == QPointF(50, 80));
    auto imageDocument = document;
    auto imagePage = page;
    auto imagePruned = page.value("prunedResult").toObject();
    auto chartBlock = block;
    chartBlock.insert("block_label", "chart");
    imagePruned.insert("parsing_res_list",
                       QJsonArray{chartBlock, chartBlock, block});
    imagePage.insert("prunedResult", imagePruned);
    imageDocument.insert("layoutParsingResults", QJsonArray{imagePage});
    imageDocument.insert(
            "_ppocr_meta",
            QJsonObject{{"block_image_paths",
                         QJsonObject{{"page_1:block_4", "files/crop.png"},
                                     {"page_1:block_4__dup_2",
                                      "files/second.png"}}}});
    auto imageBlocks = PpocrDocument::blocks(imageDocument);
    REQUIRE(imageBlocks[0].imagePath == "files/crop.png" &&
            imageBlocks[1].imagePath == "files/second.png" &&
            imageBlocks[2].imagePath.isEmpty());
    imageDocument.insert("_ppocr_meta",
                         QJsonObject{{"block_image_paths",
                                      QJsonObject{{"page_1:block_4",
                                                   "files/fallback.png"}}}});
    REQUIRE(PpocrDocument::blocks(imageDocument)[1].imagePath ==
            "files/fallback.png");
    REQUIRE(PpocrDocument::edit(&document, blocks[1].key,
                                QString::fromUtf8("修订内容")));
    REQUIRE(!PpocrDocument::edit(&document, "missing", "bad"));
    const auto edited = PpocrDocument::blocks(document);
    REQUIRE(edited[0].text == "old text" &&
            edited[1].text == QString::fromUtf8("修订内容") &&
            edited[1].edited);
    REQUIRE(PpocrDocument::markdown(document).contains(
            QString::fromUtf8("修订内容")));
    const auto unchanged = document.value("layoutParsingResults")
                                   .toArray()[0]
                                   .toObject()
                                   .value("prunedResult")
                                   .toObject()
                                   .value("parsing_res_list")
                                   .toArray()[1]
                                   .toObject();
    REQUIRE(unchanged.value("unrecognized_attribute") == 17);
    REQUIRE(PpocrDocument::save(temporary.path() + "/document.json", document,
                                &error));
    QFile saved(temporary.path() + "/document.json");
    REQUIRE(saved.open(QIODevice::ReadOnly));
    REQUIRE(PpocrDocument::parse(saved.readAll(), &error) == document);
    REQUIRE(!PpocrDocument::save(temporary.path() + "/missing/invalid.json",
                                 document, &error));
    REQUIRE(!error.isEmpty());
    const QByteArray jsonl = QJsonDocument(QJsonObject{{"result", page}})
                                     .toJson(QJsonDocument::Compact) +
                             '\n' +
                             QJsonDocument(QJsonObject{{"result", original}})
                                     .toJson(QJsonDocument::Compact);
    REQUIRE(PpocrDocument::parse(jsonl, &error)
                            .value("layoutParsingResults")
                            .toArray()
                            .size() == 2 &&
            error.isEmpty());
    REQUIRE(PpocrDocument::parse(jsonl + "\n{bad}", &error).isEmpty() &&
            error.contains("3"));
    REQUIRE(PpocrDocument::parse("{}", &error).isEmpty() && !error.isEmpty());
    auto polygonBlock = block;
    polygonBlock.insert("block_polygon_points",
                        QJsonArray{QJsonArray{1, 2}, QJsonArray{90, 3},
                                   QJsonArray{88, 100}});
    polygonBlock.insert("block_order", 3);
    auto firstBlock = block;
    firstBlock.insert("block_order", 1);
    auto orderedPage = page;
    auto orderedPruned = orderedPage.value("prunedResult").toObject();
    orderedPruned.insert("parsing_res_list",
                         QJsonArray{polygonBlock, firstBlock});
    orderedPage.insert("prunedResult", orderedPruned);
    const auto ordered = PpocrDocument::blocks(
            QJsonObject{{"layoutParsingResults", QJsonArray{orderedPage}}});
    REQUIRE(ordered.size() == 2 && ordered[0].index == 1 &&
            ordered[1].index == 0 && ordered[1].polygon.size() == 3 &&
            ordered[1].polygon[0] == QPointF(1, 2));

    QTcpServer server;
    REQUIRE(server.listen(QHostAddress::LocalHost));
    int mode = 0, polls = 0;
    QVector<Request> requests;
    const auto base =
            QStringLiteral("http://127.0.0.1:%1").arg(server.serverPort());
    QObject::connect(&server, &QTcpServer::newConnection, &server, [&]() {
        auto* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::disconnected, socket,
                         &QObject::deleteLater);
        QObject::connect(socket, &QTcpSocket::readyRead, socket, [&, socket]() {
            QByteArray buffer = socket->property("buffer").toByteArray() +
                                socket->readAll();
            socket->setProperty("buffer", buffer);
            const int split = buffer.indexOf("\r\n\r\n");
            if (split < 0 || socket->property("answered").toBool()) return;
            int length = 0;
            for (const auto& line : buffer.left(split).split('\n'))
                if (line.toLower().startsWith("content-length:"))
                    length = line.mid(15).trimmed().toInt();
            if (buffer.size() < split + 4 + length) return;
            socket->setProperty("answered", true);
            Request request{buffer.left(split), buffer.mid(split + 4, length)};
            requests.append(request);
            if (mode == 2)
                return;  // Server never answers: cancellation/timeout.
            QByteArray body;
            if (request.headers.startsWith("POST")) {
                if (mode == 1)
                    body = "{\"errorCode\":401,\"errorMsg\":\"wrong key\"}";
                else if (mode == 3)
                    body = "{\"errorCode\":\"invalid\"}";
                else
                    body = "{\"data\":{\"job_id\":\"test-job\"}}";
            } else if (request.headers.startsWith("GET /jobs/test-job")) {
                ++polls;
                if (mode == 5 || (mode == 6 && polls == 1))
                    body = "{\"data\":{\"status\":\"queued\"}}";
                else if (mode == 4)
                    body = "{\"job\":{\"state\":\"failed\",\"error\":{"
                           "\"message\":\"bad document\"}}}";
                else
                    body = QJsonDocument(
                                   QJsonObject{
                                           {"job",
                                            QJsonObject{
                                                    {"state", "completed"},
                                                    {"resultUrl",
                                                     QJsonObject{
                                                             {"jsonUrl",
                                                              QStringLiteral(
                                                                      "%1/"
                                                                      "result")
                                                                      .arg(base)}}}}}})
                                   .toJson();
            } else if (request.headers.startsWith("GET /result"))
                body = jsonl;
            else
                body = "invalid json";
            socket->write(
                    "HTTP/1.1 200 OK\r\nContent-Type: "
                    "application/json\r\nContent-Length: " +
                    QByteArray::number(body.size()) +
                    "\r\nConnection: close\r\n\r\n" + body);
            socket->disconnectFromHost();
        });
    });
    const QString imagePath = temporary.path() + "/input.png";
    QImage image(100, 120, QImage::Format_RGB32);
    image.fill(Qt::white);
    REQUIRE(image.save(imagePath));
    auto imageBlock = block;
    imageBlock.insert("block_label", "image");
    imageBlock.insert("block_content", "");
    auto exportPage = page;
    auto exportPruned = exportPage.value("prunedResult").toObject();
    exportPruned.insert("parsing_res_list", QJsonArray{imageBlock, imageBlock});
    exportPage.insert("prunedResult", exportPruned);
    auto exportDocument = document;
    auto exportPages = exportDocument.value("layoutParsingResults").toArray();
    exportPages.append(exportPage);
    exportDocument.insert("layoutParsingResults", exportPages);
    const QString bundlePath =
            temporary.path() + QString::fromUtf8("/文档.zip");
    REQUIRE(PpocrDocument::exportBundle(exportDocument, temporary.path(),
                                        {imagePath, imagePath}, bundlePath,
                                        &error));
    QFile bundle(bundlePath);
    REQUIRE(bundle.open(QIODevice::ReadOnly));
    const auto bundleBytes = bundle.readAll();
    bundle.close();
    REQUIRE(bundleBytes.startsWith("PK\003\004") &&
            bundleBytes.contains("doc_0.md") &&
            bundleBytes.contains("layout_det_res_0.jpg") &&
            bundleBytes.contains("layout_det_res_1.jpg") &&
            bundleBytes.contains("imgs/img_in_image_box_10_20_50_80.jpg") &&
            bundleBytes.contains("imgs/img_in_image_box_10_20_50_80_1.jpg"));
    // An input failure must preserve the previous complete output byte for
    // byte.
    REQUIRE(!PpocrDocument::exportBundle(exportDocument, temporary.path(),
                                         {"/missing/page.png"}, bundlePath,
                                         &error));
    REQUIRE(bundle.open(QIODevice::ReadOnly) &&
            bundle.readAll() == bundleBytes);
    bundle.close();
    auto invalidImages = exportDocument;
    auto invalidMeta = invalidImages.value("_ppocr_meta").toObject();
    invalidMeta.insert("block_image_paths",
                       QJsonObject{{"page_2:block_4", "../outside.png"}});
    invalidImages.insert("_ppocr_meta", invalidMeta);
    REQUIRE(!PpocrDocument::exportBundle(invalidImages, temporary.path(),
                                         {imagePath}, bundlePath, &error));
    // The shared video writer retains its enclosing directory and NewOnly
    // contract. OCR opts out of that prefix to match upstream's ZIP layout.
    REQUIRE(writeZipDirectory(temporary.path() + "/absent",
                              temporary.path() + "/empty.zip",
                              &error) == false);
    const QString videoDirectory = temporary.path() + "/video_dataset";
    REQUIRE(QDir().mkpath(videoDirectory));
    QFile videoText(videoDirectory + "/label.txt");
    REQUIRE(videoText.open(QIODevice::WriteOnly) &&
            videoText.write("video labels") == 12);
    videoText.close();
    const QString videoArchive = temporary.path() + "/video.zip";
    REQUIRE(writeZipDirectory(videoDirectory, videoArchive, &error));
    QFile videoZip(videoArchive);
    REQUIRE(videoZip.open(QIODevice::ReadOnly) &&
            videoZip.readAll().contains("video_dataset/label.txt"));
    REQUIRE(!writeZipDirectory(videoDirectory, videoArchive, &error));
    if (argc == 2) {
        const QString artifact =
                QFileInfo(QString::fromLocal8Bit(argv[1])).absolutePath() +
                "/document-bundle.zip";
        QFile::remove(artifact);
        REQUIRE(QFile::copy(bundlePath, artifact));
        QFile expected(QFileInfo(artifact).absolutePath() +
                       "/bundle-input.json");
        REQUIRE(expected.open(QIODevice::WriteOnly));
        const auto bytes = QJsonDocument(exportDocument).toJson();
        REQUIRE(expected.write(bytes) == bytes.size());
    }
    PpocrServiceClient client;
    PpocrServiceClient::Settings settings;
    REQUIRE(settings.model == "PaddleOCR-VL-1.6" &&
            settings.timeoutMs == 180000);
    settings.jobsUrl = QUrl(base + "/jobs");
    settings.apiKey = "secret-test";
    settings.timeoutMs = 3000;
    int finished = 0;
    QJsonObject received;
    QString failure;
    QObject::connect(&client, &PpocrServiceClient::finished, &client,
                     [&](const QJsonObject& value, const QString& message) {
                         ++finished;
                         received = value;
                         failure = message;
                     });
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 1; }));
    REQUIRE(!client.busy() && failure.isEmpty() && polls == 1 &&
            requests.size() == 3);
    REQUIRE(received.value("layoutParsingResults").toArray().size() == 2);
    REQUIRE(received.value("_ppocr_meta").toObject().value("api_job_id") ==
            "test-job");
    REQUIRE(requests[0].headers.contains("Authorization: bearer secret-test"));
    REQUIRE(requests[0].headers.contains("multipart/form-data"));
    REQUIRE(requests[0].body.contains("PaddleOCR-VL-1.6") &&
            requests[0].body.contains("147384") &&
            requests[0].body.contains("useLayoutDetection"));
    REQUIRE(!PpocrServiceClient::optionalPayload().contains("fileType"));
    REQUIRE(!requests[2].headers.contains("Authorization"));
    mode = 1;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 2; }) && failure == "wrong key");
    mode = 3;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 3; }) && !failure.isEmpty());
    mode = 4;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 4; }) &&
            failure == "bad document");
    mode = 2;
    client.parseFile(imagePath, settings);
    const int count = requests.size();
    REQUIRE(waitFor([&]() { return requests.size() > count; }));
    client.cancel();
    REQUIRE(finished == 5 && !client.busy() && !failure.isEmpty());
    settings.timeoutMs = 30;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 6; }) &&
            failure.contains("timed out"));
    // A cancelled/timed-out reply cannot complete the next generation.
    mode = 0;
    settings.timeoutMs = 3000;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 7; }) && failure.isEmpty());
    settings.apiKey.clear();
    client.parseFile(imagePath, settings);
    REQUIRE(finished == 8 && !failure.isEmpty() && !client.busy());
    settings.apiKey = "secret-test";
    settings.timeoutMs = 500;
    mode = 5;
    polls = 0;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 9; }) &&
            failure.contains("deadline") && polls == 1);
    mode = 6;
    polls = 0;
    settings.timeoutMs = 3000;
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 10; }) && failure.isEmpty() &&
            polls == 2);
    // User cancellation from a progress callback must also stop scheduled
    // polls.
    mode = 5;
    polls = 0;
    const auto cancelOnProgress =
            QObject::connect(&client, &PpocrServiceClient::progress, &client,
                             [&](const QString& text) {
                                 if (text.contains("queued")) client.cancel();
                             });
    client.parseFile(imagePath, settings);
    REQUIRE(waitFor([&]() { return finished == 11; }) && !client.busy() &&
            polls == 1);
    QObject::disconnect(cancelOnProgress);

    // Remote pipeline discovery/prediction uses the same owner and generations
    // as official asynchronous jobs, but a distinct Token auth contract.
    QTcpServer remoteServer;
    REQUIRE(remoteServer.listen(QHostAddress::LocalHost, 0));
    const QString remoteBase = QStringLiteral("http://127.0.0.1:%1")
                                       .arg(remoteServer.serverPort());
    QVector<Request> remoteRequests;
    QByteArray modelResponse =
            R"({"data":{"z-first":{"display_name":"First OCR","capabilities":{"ppocr_pipeline":true}},"not-ocr":{"capabilities":{"ppocr_pipeline":false}},"empty-map":{"capabilities":{"ppocr_pipeline":{}}},"disabled":{"capabilities":{"ppocr_pipeline":{"enabled":false}}},"a-last":{"display_name":" Last OCR ","capabilities":{"ppocr_pipeline":"TRUE"}},"numeric":{"capabilities":{"ppocr_pipeline":1}}}})";
    auto rawRemoteBlock = block;
    rawRemoteBlock.insert("block_bbox", QJsonArray{10.5, 20.5, 50.5, 80.5});
    rawRemoteBlock.insert("block_label", " Header-Image ");
    rawRemoteBlock.insert("points", QJsonArray{QJsonArray{10.5, 20.5},
                                               QJsonArray{50.5, 80.5}});
    auto rawRemotePage = page;
    auto rawRemotePruned = rawRemotePage.value("prunedResult").toObject();
    rawRemotePruned.insert("parsing_res_list",
                           QJsonArray{rawRemoteBlock, 42, rawRemoteBlock});
    rawRemotePage.insert("prunedResult", rawRemotePruned);
    rawRemotePage.insert("markdown", QJsonObject());
    const QJsonObject remoteData{
            {"layoutParsingResults", QJsonArray{false, rawRemotePage}}};
    QByteArray prediction =
            QJsonDocument(QJsonObject{{"success", true}, {"data", remoteData}})
                    .toJson();
    int remoteMode = 0;
    QObject::connect(
            &remoteServer, &QTcpServer::newConnection, &remoteServer, [&]() {
                auto* socket = remoteServer.nextPendingConnection();
                QObject::connect(socket, &QTcpSocket::disconnected, socket,
                                 &QObject::deleteLater);
                QObject::connect(
                        socket, &QTcpSocket::readyRead, socket, [&, socket]() {
                            const QByteArray bytes =
                                    socket->property("buffer").toByteArray() +
                                    socket->readAll();
                            socket->setProperty("buffer", bytes);
                            const int split = bytes.indexOf("\r\n\r\n");
                            if (split < 0 ||
                                socket->property("answered").toBool())
                                return;
                            int length = 0;
                            for (const auto& line :
                                 bytes.left(split).split('\n'))
                                if (line.toLower().startsWith(
                                            "content-length:"))
                                    length = line.mid(15).trimmed().toInt();
                            if (bytes.size() < split + 4 + length) return;
                            socket->setProperty("answered", true);
                            Request request{bytes.left(split),
                                            bytes.mid(split + 4, length)};
                            remoteRequests.append(request);
                            if (remoteMode == 1) return;
                            const bool models = request.headers.startsWith(
                                    "GET /v1/models ");
                            const bool predict = request.headers.startsWith(
                                    "POST /v1/predict ");
                            const auto body = models    ? modelResponse
                                              : predict ? prediction
                                                        : QByteArray("{}");
                            const QByteArray status = models || predict
                                                              ? "200 OK"
                                                              : "404 Not Found";
                            socket->write(
                                    "HTTP/1.1 " + status +
                                    "\r\nContent-Type: "
                                    "application/json\r\nContent-Length: " +
                                    QByteArray::number(body.size()) +
                                    "\r\nConnection: close\r\n\r\n" + body);
                            socket->disconnectFromHost();
                        });
            });
    PpocrServiceClient::RemoteSettings remote;
    REQUIRE(remote.timeoutMs == 180000 &&
            remote.serverUrl == "http://127.0.0.1:8000");
    remote.serverUrl = remoteBase + "/v1/predict";
    remote.apiKey = "remote-secret";
    remote.timeoutMs = 3000;
    int probes = 0;
    QJsonArray discovered;
    QString selectedModel, discoveredBase, probeError;
    QObject::connect(&client, &PpocrServiceClient::serviceProbed, &client,
                     [&](const QJsonArray& models, const QString& selected,
                         const QString& url, const QString& message) {
                         ++probes;
                         discovered = models;
                         selectedModel = selected;
                         discoveredBase = url;
                         probeError = message;
                     });
    client.probeService(remote, "missing");
    REQUIRE(waitFor([&]() { return probes == 1; }) && !client.busy());
    REQUIRE(probeError.isEmpty() && finished == 11 &&
            remoteRequests.size() == 2);
    REQUIRE(discovered.size() == 3 && selectedModel == "z-first" &&
            discoveredBase == remoteBase);
    REQUIRE(discovered[1].toObject().value("model_id") == "empty-map");
    REQUIRE(discovered[2].toObject().value("display_name") == "Last OCR");
    REQUIRE(remoteRequests[0].headers.startsWith("GET /v1/predict/v1/models "));
    REQUIRE(remoteRequests[1].headers.contains("Token: remote-secret") &&
            !remoteRequests[1].headers.contains("Authorization:"));
    remote.serverUrl = remoteBase + "/v1/models";
    client.probeService(remote, "a-last");
    REQUIRE(waitFor([&]() { return probes == 2; }) &&
            selectedModel == "a-last");
    const QString remoteWork = temporary.path() + "/remote-work";
    REQUIRE(QDir().mkpath(remoteWork + "/files"));
    const QString transparentPath = remoteWork + "/files/transparent.png";
    QImage transparent(100, 120, QImage::Format_ARGB32);
    transparent.fill(Qt::transparent);
    transparent.setPixelColor(20, 30, QColor(255, 0, 0, 128));
    REQUIRE(transparent.save(transparentPath));
    client.parseRemoteFile(transparentPath, remote, "a-last");
    REQUIRE(waitFor([&]() { return finished == 12; }) && failure.isEmpty());
    REQUIRE(probes == 3 && !client.busy());
    const auto remoteDocument = received;
    const auto sent =
            QJsonDocument::fromJson(remoteRequests.last().body).object();
    REQUIRE(sent.value("model") == "a-last" &&
            sent.value("params").toObject() ==
                    (QJsonObject{{"page_no", 1},
                                 {"source_file_type", "image"}}));
    const auto encoded =
            sent.value("image").toString().section(',', 1).toLatin1();
    const auto decoded =
            QImage::fromData(QByteArray::fromBase64(encoded), "PNG");
    REQUIRE(decoded.size() == QSize(100, 120) &&
            decoded.pixelColor(0, 0) == QColor(Qt::white));
    REQUIRE(decoded.pixelColor(20, 30) == QColor(255, 127, 127));
    REQUIRE(remoteDocument.value("_ppocr_meta")
                    .toObject()
                    .value("pipeline_model") == "a-last");
    const auto normalized = remoteDocument.value("layoutParsingResults")
                                    .toArray()
                                    .first()
                                    .toObject();
    const auto normalizedBlocks = normalized.value("prunedResult")
                                          .toObject()
                                          .value("parsing_res_list")
                                          .toArray();
    REQUIRE(normalizedBlocks.size() == 2 &&
            normalized.value("inputImage") == "files/transparent.png");
    REQUIRE(normalizedBlocks[0].toObject().value("block_bbox").toArray() ==
            (QJsonArray{10, 20, 50, 80}));
    REQUIRE(normalizedBlocks[0].toObject().value("block_label") ==
            "header_image");
    REQUIRE(normalizedBlocks[0].toObject().value("block_order") == 1 &&
            normalizedBlocks[1].toObject().value("block_order") == 3);
    REQUIRE(!normalizedBlocks[0].toObject().contains("unrecognized_attribute"));
    const auto cropPaths = remoteDocument.value("_ppocr_meta")
                                   .toObject()
                                   .value("block_image_paths")
                                   .toObject();
    REQUIRE(cropPaths.size() == 2 &&
            cropPaths.contains("page_1:global_block_1") &&
            cropPaths.contains("page_1:global_block_3"));
    for (auto it = cropPaths.begin(); it != cropPaths.end(); ++it)
        REQUIRE(QImage(remoteWork + '/' + it.value().toString()).size() ==
                QSize(40, 60));
    // A discovery-only cancel emits no document completion or empty record.
    remoteMode = 1;
    client.probeService(remote, "a-last");
    const int beforeProbe = remoteRequests.size();
    REQUIRE(waitFor([&]() { return remoteRequests.size() > beforeProbe; }));
    client.cancel();
    REQUIRE(probes == 4 && !probeError.isEmpty() && finished == 12 &&
            !client.busy());
    remote.timeoutMs = 25;
    client.parseRemoteFile(transparentPath, remote, "a-last");
    REQUIRE(waitFor([&]() { return finished == 13; }) &&
            failure.contains("timed out"));
    remote.timeoutMs = 3000;
    remoteMode = 0;
    prediction = R"({"success":false,"error":{"message":"remote failure"}})";
    client.parseRemoteFile(transparentPath, remote, "a-last");
    REQUIRE(waitFor([&]() { return finished == 14; }) &&
            failure == "remote failure");
    prediction = QJsonDocument(QJsonObject{{"data", remoteData}}).toJson();
    // Cached PDF pages are sent in source order and retain PDF/page_no params.
    const QString pdfPath = remoteWork + "/files/document.pdf";
    REQUIRE(QDir().mkpath(remoteWork + "/files/__PDF_document"));
    REQUIRE(transparent.save(remoteWork +
                             "/files/__PDF_document/page_001.png"));
    REQUIRE(transparent.save(remoteWork +
                             "/files/__PDF_document/page_002.png"));
    client.parseRemoteFile(pdfPath, remote, "a-last");
    REQUIRE(waitFor([&]() { return finished == 15; }) && failure.isEmpty());
    REQUIRE(received.value("layoutParsingResults").toArray().size() == 2 &&
            received.value("dataInfo").toObject().value("numPages") == 2);
    const auto pdfSent =
            QJsonDocument::fromJson(remoteRequests.last().body).object();
    REQUIRE(pdfSent.value("params").toObject() ==
            (QJsonObject{{"page_no", 2}, {"source_file_type", "pdf"}}));
    client.parseRemoteFile(remoteWork + "/files/uncached.pdf", remote,
                           "a-last");
    REQUIRE(waitFor([&]() { return finished == 16; }) &&
            failure.contains("PDF page images are unavailable"));
    const auto goodModels = modelResponse;
    const auto goodDiscovered = discovered;
    modelResponse =
            R"({"data":{"ordinary":{"capabilities":{"ppocr_pipeline":false}}}})";
    client.probeService(remote);
    const int previousProbes = probes;
    REQUIRE(waitFor([&]() { return probes > previousProbes; }) &&
            probeError.contains("No PPOCR pipeline model"));
    modelResponse = goodModels;
    if (argc == 2) {
        const QString artifactRoot =
                QFileInfo(QString::fromLocal8Bit(argv[1])).absolutePath();
        REQUIRE(PpocrDocument::save(
                artifactRoot + "/remote-pipeline-contract.json",
                {{"models_response", QString::fromUtf8(modelResponse)},
                 {"models", goodDiscovered},
                 {"raw_result", remoteData},
                 {"document", remoteDocument}},
                &error));
    }

    // Real dialog with an isolated upstream-compatible on-disk workspace.
    const QString work = temporary.path() + "/workspace";
    auto* dialog = new PpocrDialog(nullptr, work);
    REQUIRE(dialog->size() == QSize(1560, 920) &&
            dialog->minimumSize() == QSize(1200, 760));
    bool editedSettings = false;
    QTimer::singleShot(0, dialog, [&]() {
        auto* settingsDialog = dialog->findChild<QDialog*>("PpocrApiSettings");
        if (!settingsDialog) return;
        settingsDialog->findChild<QLineEdit*>("PpocrApiUrl")
                ->setText(base + "/jobs");
        settingsDialog->findChild<QLineEdit*>("PpocrApiKey")
                ->setText("secret-test");
        settingsDialog->findChild<QLineEdit*>("PpocrServerUrl")
                ->setText(remoteBase);
        settingsDialog->findChild<QLineEdit*>("PpocrServerToken")
                ->setText("remote-secret");
        editedSettings = true;
        settingsDialog->findChild<QPushButton*>("PpocrSettingsSave")->click();
    });
    dialog->findChild<QPushButton*>("PpocrSettings")->click();
    REQUIRE(editedSettings);
    // Saving an intentionally empty credential must remove the old secret.
    bool clearedKey = false;
    QTimer::singleShot(0, dialog, [&]() {
        auto* settingsDialog = dialog->findChild<QDialog*>("PpocrApiSettings");
        if (!settingsDialog) return;
        settingsDialog->findChild<QLineEdit*>("PpocrApiKey")->clear();
        clearedKey = true;
        settingsDialog->findChild<QPushButton*>("PpocrSettingsSave")->click();
    });
    dialog->findChild<QPushButton*>("PpocrSettings")->click();
    QFile apiConfig(work + "/xanylabeling_data/paddleocr/api_settings.json");
    REQUIRE(clearedKey && apiConfig.open(QIODevice::ReadOnly) &&
            QJsonDocument::fromJson(apiConfig.readAll())
                            .object()
                            .value("api_key") == "");
    apiConfig.close();
    QTimer::singleShot(0, dialog, [&]() {
        auto* settingsDialog = dialog->findChild<QDialog*>("PpocrApiSettings");
        if (!settingsDialog) return;
        settingsDialog->findChild<QLineEdit*>("PpocrApiKey")
                ->setText("secret-test");
        settingsDialog->findChild<QPushButton*>("PpocrSettingsSave")->click();
    });
    dialog->findChild<QPushButton*>("PpocrSettings")->click();
    dialog->importFiles({imagePath, imagePath});
    auto* records = dialog->findChild<QListWidget*>("PpocrRecords");
    REQUIRE(records && records->count() == 2);
    auto* typeFilter = dialog->findChild<QComboBox*>("PpocrTypeFilter");
    typeFilter->setCurrentIndex(typeFilter->findData("pdf"));
    REQUIRE(records->count() == 0);
    typeFilter->setCurrentIndex(typeFilter->findData("image"));
    REQUIRE(records->count() == 2);
    typeFilter->setCurrentIndex(0);
    auto* sort = dialog->findChild<QComboBox*>("PpocrSort");
    REQUIRE(sort && sort->currentData() == "newest");
    sort->setCurrentIndex(sort->findData("oldest"));
    REQUIRE(records->count() == 2);
    const auto name = records->currentItem()->data(Qt::UserRole).toString();
    REQUIRE(PpocrDocument::save(
            work + "/xanylabeling_data/paddleocr/jsons/" + name + ".json",
            document, &error));
    records->setCurrentRow(-1);
    for (int i = 0; i < records->count(); ++i)
        if (records->item(i)->data(Qt::UserRole).toString() == name)
            records->setCurrentRow(i);
    auto* list = dialog->findChild<QListWidget*>("PpocrBlocks");
    REQUIRE(list && list->count() == 2);
    list->setCurrentRow(1);
    dialog->findChild<QPushButton*>("PpocrCopy")->click();
    REQUIRE(QApplication::clipboard()->text() == QString::fromUtf8("修订内容"));
    dialog->findChild<QPushButton*>("PpocrFavorite")->click();
    QFile state(work + "/xanylabeling_data/paddleocr/ui_state.json");
    REQUIRE(state.open(QIODevice::ReadOnly));
    REQUIRE(QJsonDocument::fromJson(state.readAll())
                    .object()
                    .value("favorites")
                    .toArray()
                    .contains(name));
    // Parse through the dialog's saved URL and credentials. No real service is
    // contacted. Switching the selected record must not redirect the result.
    mode = 0;
    requests.clear();
    dialog->findChild<QPushButton*>("PpocrParse")->click();
    records->setCurrentRow(records->currentRow() == 0 ? 1 : 0);
    const auto otherName =
            records->currentItem()->data(Qt::UserRole).toString();
    REQUIRE(waitFor([&]() {
        QFile result(work + "/xanylabeling_data/paddleocr/jsons/" + name +
                     ".json");
        if (!result.open(QIODevice::ReadOnly)) return false;
        return QJsonDocument::fromJson(result.readAll())
                       .object()
                       .value("_ppocr_meta")
                       .toObject()
                       .value("api_job_id") == "test-job";
    }));
    REQUIRE(requests.size() == 3 &&
            requests[0].headers.startsWith("POST /jobs") &&
            requests[0].headers.contains("Authorization: bearer secret-test"));
    REQUIRE(!QFileInfo::exists(work + "/xanylabeling_data/paddleocr/jsons/" +
                               otherName + ".json"));
    auto* modelCombo = dialog->findChild<QComboBox*>("PpocrModel");
    REQUIRE(modelCombo->currentData() == "__ppocr_api__:PaddleOCR-VL-1.6");
    dialog->findChild<QPushButton*>("PpocrRefreshService")->click();
    REQUIRE(waitFor([&]() { return modelCombo->findData("z-first") >= 0; }));
    REQUIRE(modelCombo->currentData() == "z-first" && modelCombo->count() == 5);
    REQUIRE(!QFileInfo::exists(work +
                               "/xanylabeling_data/paddleocr/jsons/.json"));
    dialog->findChild<QPushButton*>("PpocrParse")->click();
    REQUIRE(waitFor([&]() {
        QFile result(work + "/xanylabeling_data/paddleocr/jsons/" + otherName +
                     ".json");
        return result.open(QIODevice::ReadOnly) &&
               QJsonDocument::fromJson(result.readAll())
                               .object()
                               .value("_ppocr_meta")
                               .toObject()
                               .value("pipeline_model") == "z-first";
    }));
    // Restore the edited document used by the persistence/screenshot checks.
    REQUIRE(PpocrDocument::save(
            work + "/xanylabeling_data/paddleocr/jsons/" + name + ".json",
            document, &error));
    delete dialog;
    auto* reopened = new PpocrDialog(nullptr, work);
    auto* reopenedRecords = reopened->findChild<QListWidget*>("PpocrRecords");
    REQUIRE(reopenedRecords->count() == 2);
    // Upstream initially selects the newest file; select the saved document
    // by identity, independently of filesystem timestamp granularity.
    for (int i = 0; i < reopenedRecords->count(); ++i)
        if (reopenedRecords->item(i)->data(Qt::UserRole).toString() == name)
            reopenedRecords->setCurrentRow(i);
    REQUIRE(reopened->findChild<QListWidget*>("PpocrBlocks")->count() == 2);
    if (argc == 2) {
        reopened->show();
        QCoreApplication::processEvents();
        auto* preview = reopened->findChild<QGraphicsView*>("PpocrPreview");
        REQUIRE(preview && preview->mapFromScene(QRectF(0, 0, 100, 120))
                                           .boundingRect()
                                           .width() > 200);
        REQUIRE(reopened->grab().save(QString::fromLocal8Bit(argv[1])));
    }
    // Real inline rich-text Save/Cancel, strikeout serialization and stale
    // callbacks use the same atomic document boundary as the table editor.
    for (bool accept : {false, true}) {
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        auto* richBlocks = reopened->findChild<QListWidget*>("PpocrBlocks");
        richBlocks->setCurrentRow(0);
        const QString before =
                richBlocks->currentItem()->data(Qt::UserRole + 1).toString();
        reopened->findChild<QPushButton*>("PpocrEdit")->click();
        auto* editor = reopened->findChild<PpocrRichTextEditor*>(
                "PPOCRRichTextBlockEditor");
        REQUIRE(editor && !editor->isWindow() &&
                richBlocks->itemWidget(richBlocks->item(0)) == editor);
        auto* content = editor->findChild<QTextEdit*>("PpocrRichContent");
        REQUIRE(content);
        content->setPlainText("corrected rich text");
        content->selectAll();
        editor->findChild<QToolButton*>("PpocrRichStrike")->click();
        const QString corrected = editor->content();
        REQUIRE(corrected.contains("line-through") &&
                corrected.contains("corrected rich text"));
        editor->findChild<QPushButton*>(accept ? "PpocrRichSave"
                                               : "PpocrRichCancel")
                ->click();
        REQUIRE(richBlocks->currentRow() == 0);
        REQUIRE(richBlocks->currentItem()->data(Qt::UserRole + 1).toString() ==
                (accept ? corrected : before));
    }
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    reopened->findChild<QPushButton*>("PpocrEdit")->click();
    auto* staleRich = reopened->findChild<PpocrRichTextEditor*>(
            "PPOCRRichTextBlockEditor");
    REQUIRE(staleRich);
    const QString richPath =
            work + "/xanylabeling_data/paddleocr/jsons/" + name + ".json";
    QFile richBefore(richPath);
    REQUIRE(richBefore.open(QIODevice::ReadOnly));
    const auto richBytes = richBefore.readAll();
    richBefore.close();
    reopenedRecords->setCurrentRow(-1);
    REQUIRE(QMetaObject::invokeMethod(
            staleRich, "saveRequested", Qt::DirectConnection,
            Q_ARG(QString, QString("stale rich edit"))));
    QFile richAfter(richPath);
    REQUIRE(richAfter.open(QIODevice::ReadOnly) &&
            richAfter.readAll() == richBytes);
    richAfter.close();

    // Stored upstream image-block metadata reaches the actual inline editor.
    auto previewDocument = original;
    previewDocument.insert("layoutParsingResults", QJsonArray{exportPage});
    previewDocument.insert(
            "_ppocr_meta",
            QJsonObject{{"block_image_paths",
                         QJsonObject{{"page_1:block_4",
                                      QString("files/%1").arg(name)}}}});
    REQUIRE(PpocrDocument::save(richPath, previewDocument, &error));
    for (int i = 0; i < reopenedRecords->count(); ++i)
        if (reopenedRecords->item(i)->data(Qt::UserRole).toString() == name)
            reopenedRecords->setCurrentRow(i);
    auto* previewBlocks = reopened->findChild<QListWidget*>("PpocrBlocks");
    REQUIRE(previewBlocks->count() == 2);
    previewBlocks->setCurrentRow(1);  // The duplicate key uses its base path.
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    reopened->findChild<QPushButton*>("PpocrEdit")->click();
    auto* previewEditor = reopened->findChild<PpocrRichTextEditor*>(
            "PPOCRRichTextBlockEditor");
    REQUIRE(previewEditor);
    auto* cropPreview =
            previewEditor->findChild<QLabel*>("PPOCRRichTextBlockEditorImage");
    REQUIRE(cropPreview && !cropPreview->isHidden() &&
            !cropPreview->property("pixmap").value<QPixmap>().isNull());
    if (argc == 2) {
        for (int i = 0; i < 4; ++i) QCoreApplication::processEvents();
        REQUIRE(reopened->grab().save(QString::fromLocal8Bit(argv[1]) +
                                      ".inline-rich-image.png"));
    }
    previewEditor->findChild<QPushButton*>("PpocrRichCancel")->click();
    reopenedRecords->setCurrentRow(-1);

    // Exercise the real document -> inline table editor -> atomic save path.
    // Unknown block attributes and cancellation must survive that round trip.
    auto tableBlock = block;
    tableBlock["block_label"] = "table";
    tableBlock["block_content"] = "<fcel>original<ecel><nl>";
    auto tablePage = page;
    auto pruned = tablePage.value("prunedResult").toObject();
    pruned["parsing_res_list"] = QJsonArray{tableBlock};
    tablePage["prunedResult"] = pruned;
    auto tableDocument = original;
    tableDocument["layoutParsingResults"] = QJsonArray{tablePage};
    const QString tablePath =
            work + "/xanylabeling_data/paddleocr/jsons/" + name + ".json";
    REQUIRE(PpocrDocument::save(tablePath, tableDocument, &error));
    reopenedRecords->setCurrentRow(-1);
    for (int i = 0; i < reopenedRecords->count(); ++i)
        if (reopenedRecords->item(i)->data(Qt::UserRole).toString() == name)
            reopenedRecords->setCurrentRow(i);
    auto* reopenedBlocks = reopened->findChild<QListWidget*>("PpocrBlocks");
    REQUIRE(reopenedBlocks->count() == 1);
    reopenedBlocks->setCurrentRow(0);
    for (bool accept : {false, true}) {
        QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
        reopenedBlocks->setCurrentRow(0);
        reopened->findChild<QPushButton*>("PpocrEdit")->click();
        auto* editor =
                reopened->findChild<PpocrTableEditor*>("PPOCRTableBlockEditor");
        REQUIRE(editor && !editor->isWindow() &&
                reopenedBlocks->itemWidget(reopenedBlocks->item(0)) == editor);
        auto* grid = editor->findChild<QTableWidget*>("PpocrTableGrid");
        REQUIRE(grid && grid->item(0, 0));
        grid->item(0, 0)->setText("corrected table");
        if (argc == 2 && !accept) {
            for (int i = 0; i < 4; ++i) QCoreApplication::processEvents();
            REQUIRE(reopened->grab().save(QString::fromLocal8Bit(argv[1]) +
                                          ".inline-table.png"));
        }
        editor->findChild<QPushButton*>(accept ? "PpocrTableSave"
                                               : "PpocrTableCancel")
                ->click();
        REQUIRE(reopenedBlocks->currentRow() == 0);
        QFile persisted(tablePath);
        REQUIRE(persisted.open(QIODevice::ReadOnly));
        const auto result = PpocrDocument::parse(persisted.readAll(), &error);
        REQUIRE(error.isEmpty() &&
                result.value("future_schema") == "preserved");
        const auto savedBlocks = PpocrDocument::blocks(result);
        REQUIRE(savedBlocks.size() == 1 &&
                savedBlocks[0].text ==
                        (accept ? "<fcel>corrected table<ecel><nl>"
                                : "<fcel>original<ecel><nl>"));
        REQUIRE(result.value("layoutParsingResults")
                        .toArray()[0]
                        .toObject()
                        .value("prunedResult")
                        .toObject()
                        .value("parsing_res_list")
                        .toArray()[0]
                        .toObject()
                        .value("unrecognized_attribute") == 17);
    }
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    reopenedBlocks->setCurrentRow(0);
    reopened->findChild<QPushButton*>("PpocrEdit")->click();
    auto* staleEditor =
            reopened->findChild<PpocrTableEditor*>("PPOCRTableBlockEditor");
    REQUIRE(staleEditor);
    QFile beforeStale(tablePath);
    REQUIRE(beforeStale.open(QIODevice::ReadOnly));
    const auto beforeStaleBytes = beforeStale.readAll();
    beforeStale.close();
    reopenedRecords->setCurrentRow(-1);
    // QListWidget defers deleting item widgets. Even a queued signal from that
    // still-live old editor must not write into the newly selected document.
    REQUIRE(QMetaObject::invokeMethod(
            staleEditor, "saveRequested", Qt::DirectConnection,
            Q_ARG(QString, QString("<fcel>stale<nl>"))));
    QFile afterStale(tablePath);
    REQUIRE(afterStale.open(QIODevice::ReadOnly) &&
            afterStale.readAll() == beforeStaleBytes);
    delete reopened;
    std::puts(
            "PASS: OCR document editing, JSONL, service "
            "success/error/cancel/timeout/restart, workspace persistence");
    return 0;
}
