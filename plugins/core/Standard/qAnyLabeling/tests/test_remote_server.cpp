// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Offscreen test for the remote inference client: an in-process QTcpServer
// mocks the /v1/models, /v1/predict and /v1/video/* endpoints and the client
// must round-trip the upstream protocol (data-URI frame in, shapes out,
// session id in, SSE propagation stream back) without any external service.

#include <QApplication>
#include <QEventLoop>
#include <QHostAddress>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <cstdio>

#include "RemoteServerClient.h"

namespace {

int failures = 0;
bool cleanupSeen = false;
bool predictResetTracker = false;
bool promptCarriedObjId = false;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}

void expectMsg(bool ok, const char* what, const QString& error) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s (%s)\n", what,
                     error.toLocal8Bit().constData());
        ++failures;
    }
}

void writeJson(QTcpSocket* socket, const QByteArray& body) {
    // Keep-alive: QNetworkAccessManager reuses the connection for follow-up
    // requests, so the mock answers repeatedly on one socket.
    socket->write(
            "HTTP/1.1 200 OK\r\nContent-Type: "
            "application/json\r\nContent-Length: " +
            QByteArray::number(body.size()) +
            "\r\nConnection: keep-alive\r\n\r\n" + body);
    socket->flush();
}

// HTTP keep-alive mock: requests accumulate in socket properties and every
// fully-arrived request (header + Content-Length body) is dispatched in
// order; TCP segmentation never duplicates or drops a response.
void handleConnection(QTcpSocket* socket) {
    QByteArray buffer = socket->property("req").toByteArray();
    buffer += socket->readAll();
    socket->setProperty("req", buffer);
    int processed = socket->property("processed").toInt();
    for (;;) {
        const int headEnd = buffer.indexOf("\r\n\r\n", processed);
        if (headEnd < 0) break;
        int contentLength = 0;
        const QList<QByteArray> headers =
                buffer.mid(processed, headEnd - processed).split('\n');
        for (const QByteArray& header : headers) {
            const QByteArray trimmed = header.trimmed();
            if (trimmed.startsWith("Content-Length:"))
                contentLength = trimmed.mid(15).trimmed().toInt();
        }
        const int bodyEnd = headEnd + 4 + contentLength;
        if (buffer.size() < bodyEnd) break;
        const int lineEnd = buffer.indexOf('\r', processed);
        const QByteArray line = buffer.mid(processed, lineEnd - processed);
        const QByteArray request = buffer.mid(processed, bodyEnd - processed);
        processed = bodyEnd;
        if (line.contains("/v1/models")) {
            writeJson(socket,
                      QByteArrayLiteral(
                              "{\"data\":{\"demo-det\":{\"classes\":[\"cat\"]},"
                              "\"demo-seg\":{\"classes\":[]}}}"));
        } else if (line.contains("/v1/predict")) {
            predictResetTracker = request.contains("\"reset_tracker\":true");
            writeJson(socket,
                      QByteArrayLiteral(
                              "{\"data\":{\"shapes\":[{\"label\":\"cat\","
                              "\"shape_type\":\"rectangle\","
                              "\"points\":[[10,20],[110,120]],\"score\":0.9,"
                              "\"group_id\":3,\"difficult\":false}]}}"));
        } else if (line.contains("/v1/video/init")) {
            writeJson(socket,
                      QByteArrayLiteral(
                              "{\"data\":{\"session_id\":\"sess-1\"}}"));
        } else if (line.contains("/v1/video/prompt")) {
            promptCarriedObjId = request.contains("\"obj_id\":10000");
            writeJson(socket,
                      QByteArrayLiteral(
                              "{\"data\":{\"masks\":[{\"label\":\"car\","
                              "\"shape_type\":\"rectangle\","
                              "\"points\":[[5,5],[60,80]],\"score\":0.8}]}}"));
        } else if (line.contains("/v1/video/cleanup/")) {
            cleanupSeen = true;
            writeJson(socket, QByteArrayLiteral("{\"ok\":true}"));
        } else if (line.contains("/v1/video/propagate/stream")) {
            // SSE stream: header without content length, then events, and
            // the connection closes after the stream.
            socket->write(
                    "HTTP/1.1 200 OK\r\nContent-Type: "
                    "text/event-stream\r\nConnection: close\r\n\r\n");
            socket->write(
                    "data: {\"type\":\"started\",\"total_frames\":2,"
                    "\"start_frame_index\":0}\n\n"
                    "data: {\"type\":\"progress\",\"current_frame\":0}\n\n"
                    "data: {\"type\":\"completed\",\"results\":"
                    "{\"0\":{\"masks\":[{\"label\":\"car\","
                    "\"shape_type\":\"rectangle\","
                    "\"points\":[[1,2],[30,40]]}]},"
                    "\"1\":{\"masks\":[]}}}"
                    "\n\n");
            socket->flush();
            socket->disconnectFromHost();
            socket->setProperty("processed", processed);
            return;
        } else {
            writeJson(socket, QByteArrayLiteral("{}"));
        }
    }
    socket->setProperty("processed", processed);
}

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    QTcpServer server;
    if (!server.listen(QHostAddress::LocalHost)) {
        std::fprintf(stderr, "mock server failed to listen\n");
        return 1;
    }
    QObject::connect(&server, &QTcpServer::newConnection, &server, [&]() {
        QTcpSocket* socket = server.nextPendingConnection();
        QObject::connect(socket, &QTcpSocket::readyRead, socket,
                         [socket]() { handleConnection(socket); });
        QObject::connect(socket, &QTcpSocket::disconnected, socket,
                         &QTcpSocket::deleteLater);
    });

    RemoteServerClient client;
    RemoteServerClient::Settings settings;
    settings.serverUrl =
            QStringLiteral("http://127.0.0.1:%1").arg(server.serverPort());
    settings.apiKey = QStringLiteral("test-key");
    settings.timeoutSec = 10;
    client.applySettings(settings);

    // /v1/models round trip.
    QJsonObject modelData;
    QString modelError;
    QEventLoop modelsLoop;
    QObject::connect(&client, &RemoteServerClient::modelsFetched, &modelsLoop,
                     [&](const QJsonObject& data, const QString& error) {
                         modelData = data;
                         modelError = error;
                         modelsLoop.quit();
                     });
    client.fetchModels();
    QTimer::singleShot(5000, &modelsLoop, &QEventLoop::quit);
    modelsLoop.exec();
    expect(modelError.isEmpty(), "model list must succeed");
    expect(modelData.contains(QStringLiteral("demo-det")),
           "model list must carry demo-det");

    // /v1/predict round trip with a real image payload.
    const QImage frame(64, 48, QImage::Format_RGB888);
    QJsonArray shapes;
    QString predictError;
    QEventLoop predictLoop;
    QObject::connect(&client, &RemoteServerClient::predictFinished,
                     &predictLoop,
                     [&](const QJsonArray& result, const QString& error) {
                         shapes = result;
                         predictError = error;
                         predictLoop.quit();
                     });
    client.predict(frame, QString(), QStringLiteral("demo-det"),
                   QVariantMap{{QStringLiteral("conf_threshold"), 0.25}});
    QTimer::singleShot(5000, &predictLoop, &QEventLoop::quit);
    predictLoop.exec();
    expect(predictError.isEmpty(), "prediction must succeed");
    expect(shapes.size() == 1, "prediction must return one shape");
    if (shapes.size() == 1) {
        const QJsonObject shape = shapes.at(0).toObject();
        expect(shape.value(QStringLiteral("label")).toString() ==
                       QStringLiteral("cat"),
               "shape label must round-trip");
        expect(shape.value(QStringLiteral("shape_type")).toString() ==
                       QStringLiteral("rectangle"),
               "shape type must round-trip");
        expect(shape.value(QStringLiteral("points")).toArray().size() == 2,
               "shape points must round-trip");
        expect(shape.value(QStringLiteral("score")).toDouble() > 0.89,
               "shape score must round-trip");
    }

    // /v1/video/init + prompt + SSE propagation + cleanup round trip.
    QString sessionId;
    QString initError;
    QEventLoop initLoop;
    QObject::connect(&client, &RemoteServerClient::videoInitFinished, &initLoop,
                     [&](const QString& id, const QString& error) {
                         sessionId = id;
                         initError = error;
                         initLoop.quit();
                     });
    client.videoInit(QStringList{QStringLiteral("data:image/png;base64,AAAA")},
                     QStringLiteral("demo-video"));
    QTimer::singleShot(5000, &initLoop, &QEventLoop::quit);
    initLoop.exec();
    expectMsg(initError.isEmpty(), "video init must succeed", initError);
    expect(sessionId == QStringLiteral("sess-1"),
           "video init must carry the session id");
    expect(client.hasVideoSession(), "video session must be tracked");

    QJsonArray masks;
    QString promptError;
    QEventLoop promptLoop;
    QObject::connect(&client, &RemoteServerClient::videoPromptFinished,
                     &promptLoop,
                     [&](const QJsonArray& result, const QString& error) {
                         masks = result;
                         promptError = error;
                         promptLoop.quit();
                     });
    client.videoPromptText(
            QStringLiteral("a person."), 0, QStringLiteral("demo-video"),
            QVariantMap{{QStringLiteral("conf_threshold"), 0.25}});
    QTimer::singleShot(5000, &promptLoop, &QEventLoop::quit);
    promptLoop.exec();
    expectMsg(promptError.isEmpty(), "video text prompt must succeed",
              promptError);
    expect(masks.size() == 1, "video prompt must return one mask");

    // Point prompt carries the upstream obj_id sentinel.
    QEventLoop pointLoop;
    QObject::connect(
            &client, &RemoteServerClient::videoPromptFinished, &pointLoop,
            [&](const QJsonArray&, const QString&) { pointLoop.quit(); });
    client.videoPromptPoints(QVariantList{QVariantList{0.5, 0.5}},
                             QVariantList{1}, 1, QStringLiteral("demo-video"),
                             QVariantMap{});
    QTimer::singleShot(5000, &pointLoop, &QEventLoop::quit);
    pointLoop.exec();
    expect(promptCarriedObjId, "point prompt must carry obj_id 10000");

    // SSE propagation: started/progress/completed events, results keyed by
    // frame index.
    QJsonObject propagateResults;
    QString propagateError;
    int propagateTotal = -1;
    QEventLoop propagateLoop;
    QObject::connect(
            &client, &RemoteServerClient::videoPropagateStarted, &propagateLoop,
            [&](int totalFrames, int) { propagateTotal = totalFrames; });
    QObject::connect(&client, &RemoteServerClient::videoPropagateFinished,
                     &propagateLoop,
                     [&](const QJsonObject& results, const QString& error) {
                         propagateResults = results;
                         propagateError = error;
                         propagateLoop.quit();
                     });
    client.videoPropagate(QStringLiteral("demo-video"), 0, -1);
    QTimer::singleShot(5000, &propagateLoop, &QEventLoop::quit);
    propagateLoop.exec();
    expectMsg(propagateError.isEmpty(), "propagation must succeed",
              propagateError);
    expect(propagateTotal == 2, "propagation must report total frames");
    expect(propagateResults.contains(QStringLiteral("0")),
           "propagation results must carry frame 0");
    if (propagateResults.contains(QStringLiteral("0"))) {
        const QJsonArray frameMasks =
                propagateResults.value(QStringLiteral("0"))
                        .toObject()
                        .value(QStringLiteral("masks"))
                        .toArray();
        expect(frameMasks.size() == 1, "frame 0 must carry one mask");
    }

    // One-shot tracker reset rides the next predict params and clears —
    // the second predict must NOT carry the flag again.
    QEventLoop resetLoop;
    QObject::connect(
            &client, &RemoteServerClient::predictFinished, &resetLoop,
            [&](const QJsonArray&, const QString&) { resetLoop.quit(); });
    client.setResetTracker();
    client.predict(frame, QString(), QStringLiteral("demo-det"), QVariantMap{});
    QTimer::singleShot(5000, &resetLoop, &QEventLoop::quit);
    resetLoop.exec();
    expect(predictResetTracker,
           "reset_tracker must ride the next predict params");
    client.predict(frame, QString(), QStringLiteral("demo-det"), QVariantMap{});
    QTimer::singleShot(5000, &resetLoop, &QEventLoop::quit);
    resetLoop.exec();
    expect(!predictResetTracker, "reset_tracker must be one-shot");

    // Cleanup resets the local session and notifies the server.
    QEventLoop cleanupLoop;
    QTimer::singleShot(300, &cleanupLoop, &QEventLoop::quit);
    client.resetVideoSession();
    cleanupLoop.exec();
    expect(!client.hasVideoSession(), "cleanup must clear the local session");
    expect(cleanupSeen, "cleanup must reach the server");

    if (failures != 0) {
        std::fprintf(stderr, "test_qanylabeling_remote: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_qanylabeling_remote: ok\n");
    return 0;
}
