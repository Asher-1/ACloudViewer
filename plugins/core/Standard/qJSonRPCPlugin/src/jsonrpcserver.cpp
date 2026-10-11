// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "jsonrpcserver.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QPointer>
#include <QWebSocketCorsAuthenticator>

namespace {

void sendResponse(QWebSocket *client,
                  const QJsonValue &id,
                  const JsonRPCResult &result) {
    QJsonObject response;
    response["jsonrpc"] = "2.0";
    response["id"] = id;
    if (result.isError) {
        QJsonObject error;
        error["code"] = result.error_code;
        if (!result.error_message.isEmpty()) {
            error["message"] = result.error_message;
        }
        if (!result.error_data.isNull() && result.error_data.isValid()) {
            error["data"] = QJsonValue::fromVariant(result.error_data);
        }
        response["error"] = error;
    } else {
        response["result"] = QJsonValue::fromVariant(result.result);
    }
    client->sendTextMessage(QJsonDocument(response).toJson());
}

}  // namespace

JsonRPCServer::JsonRPCServer(QObject *parent)
    : QObject(parent),
      ws_server(new QWebSocketServer(QStringLiteral("CloudViewer"),
                                     QWebSocketServer::NonSecureMode)) {
    if (ws_server) {
        connect(ws_server, &QWebSocketServer::newConnection, this,
                &JsonRPCServer::onNewConnection);
        connect(ws_server, &QWebSocketServer::closed, this,
                &JsonRPCServer::onClosed);
        connect(ws_server, &QWebSocketServer::originAuthenticationRequired,
                this, &JsonRPCServer::onOriginAuthenticationRequired);
    }
}

JsonRPCServer::~JsonRPCServer() {
    for (QWebSocket *conn : connections) {
        conn->close();
        delete conn;
    }
}

bool JsonRPCServer::listen(quint16 port) {
    qDebug() << "JsonRPCServer::listen";
    if (ws_server == nullptr) {
        return false;
    }
    if (ws_server->isListening()) {
        ws_server->close();
    }
    // Loopback only: the methods open/export arbitrary paths and run child
    // programs (process.run_cli, colmap.run), with no authentication.
    return ws_server->listen(QHostAddress::LocalHost, port);
}

bool JsonRPCServer::isListening() const {
    return ws_server && ws_server->isListening();
}

QString JsonRPCServer::errorString() const {
    return ws_server ? ws_server->errorString() : QString();
}

void JsonRPCServer::close() {
    qDebug() << "JsonRPCServer::close";
    if (ws_server == nullptr) {
        return;
    }
    ws_server->close();

    for (QWebSocket *conn : connections) {
        conn->close();
        conn->deleteLater();
    }
    connections.clear();
}

void JsonRPCServer::onNewConnection() {
    qDebug() << "JsonRPCServer::onNewConnection";
    QWebSocket *pSocket = ws_server->nextPendingConnection();
    if (pSocket == nullptr) {
        return;
    }
    connect(pSocket, &QWebSocket::textMessageReceived, this,
            &JsonRPCServer::processTextMessage);
    connect(pSocket, &QWebSocket::binaryMessageReceived, this,
            &JsonRPCServer::processBinaryMessage);
    connect(pSocket, &QWebSocket::disconnected, this,
            &JsonRPCServer::socketDisconnected);

    connections.append(pSocket);
}

void JsonRPCServer::onClosed() { qDebug() << "JsonRPCServer::onClosed"; }

void JsonRPCServer::processTextMessage(QString message) {
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    if (pClient == nullptr) {
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument doc =
            QJsonDocument::fromJson(message.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        sendResponse(pClient, QJsonValue::Null,
                     JsonRPCResult::error(-32700, "Parse error",
                                          parseError.errorString()));
        return;
    }
    const QJsonObject obj = doc.object();
    const QJsonValue id = obj.value("id");
    const QString method = obj.value("method").toString();
    if (!doc.isObject() || method.isEmpty() ||
        obj.value("jsonrpc").toString().isEmpty()) {
        // Batch arrays are not supported.
        sendResponse(pClient, id.isUndefined() ? QJsonValue::Null : id,
                     JsonRPCResult::error(-32600, "Invalid Request"));
        return;
    }
    const JsonRPCResult result =
            emit execute(method, obj.value("params").toVariant().toMap());
    if (!obj.contains("id")) {
        return;  // notification: no response, even for a deferred result
    }
    if (result.pending) {
        // The client may disconnect (and be deleted) before the child ends.
        QPointer<QWebSocket> client(pClient);
        result.pending->onComplete([client, id](const JsonRPCResult &done) {
            if (client) sendResponse(client, id, done);
        });
        return;
    }
    sendResponse(pClient, id, result);
}

void JsonRPCServer::processBinaryMessage(QByteArray message) {
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    qDebug() << "Binary Message received:" << message;
    if (pClient) {
        pClient->sendBinaryMessage(message);
    }
}

void JsonRPCServer::onOriginAuthenticationRequired(
        QWebSocketCorsAuthenticator *authenticator) {
    // Browsers always send Origin and let any web page open ws:// to
    // loopback; the agent clients (Python websockets, CLI, MCP) send none.
    authenticator->setAllowed(authenticator->origin().isEmpty());
}

void JsonRPCServer::socketDisconnected() {
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    qDebug() << "socketDisconnected:" << pClient;
    if (pClient) {
        connections.removeAll(pClient);
        pClient->deleteLater();
    }
}
