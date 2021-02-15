#include "jsonrpcserver.h"
#include <QJsonDocument>
#include <QJsonObject>

JsonRPCServer::JsonRPCServer(QObject *parent) : QObject(parent),
        ws_server(new QWebSocketServer(QStringLiteral("CloudViewer"), QWebSocketServer::NonSecureMode))
{
    if(ws_server) {
        connect(ws_server, &QWebSocketServer::newConnection, this, &JsonRPCServer::onNewConnection);
        connect(ws_server, &QWebSocketServer::closed, this, &JsonRPCServer::onClosed);
    }
}

JsonRPCServer::~JsonRPCServer()
{
    for(QWebSocket *conn: connections) {
        conn->close();
        delete conn;
    }
}

void JsonRPCServer::listen(unsigned int port)
{
    qDebug() << "JsonRPCServer::listen";
    if(ws_server == nullptr) {
        return;
    }
    if(ws_server->isListening()) {
        ws_server->close();
    }
    ws_server->listen(QHostAddress::Any, port);
}

void JsonRPCServer::close()
{
    qDebug() << "JsonRPCServer::close";
    if(ws_server == nullptr) {
        return;
    }
    ws_server->close();

    for(QWebSocket *conn: connections) {
        conn->close();
        conn->deleteLater();
    }
    connections.clear();
}

void JsonRPCServer::onNewConnection()
{
    qDebug() << "JsonRPCServer::onNewConnection";
    QWebSocket *pSocket = ws_server->nextPendingConnection();
    if(pSocket == nullptr) {
        return;
    }
    connect(pSocket, &QWebSocket::textMessageReceived, this, &JsonRPCServer::processTextMessage);
    connect(pSocket, &QWebSocket::binaryMessageReceived, this, &JsonRPCServer::processBinaryMessage);
    connect(pSocket, &QWebSocket::disconnected, this, &JsonRPCServer::socketDisconnected);

    connections.append(pSocket);
}

void JsonRPCServer::onClosed()
{
    qDebug() << "JsonRPCServer::onClosed";
}

void JsonRPCServer::processTextMessage(QString message)
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    qDebug() << "Message received:" << message;
    if (pClient == nullptr) {
        return;
    }
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
	QJsonObject obj = doc.object();
    auto method = obj.value("method").toString();
    auto params = obj.value("params").toVariant().toMap();

    qDebug() << "method: " << method << ", params: " << params;
    // check invalid JSON RPC
    JsonRPCResult result;
    if(!method.isEmpty() && !obj.value("jsonrpc").toString().isEmpty()) {
        // perform the RPC
        result = emit execute(method, params);
    } else {
        result = JsonRPCResult::error(-32600, "Invalid Request");
    }
    // now build JSON response
    if(!doc.object().contains("id")) {
        // abort here, on notification there is NO RESPONSE
        return;
    }
    QJsonObject response;
    response["jsonrpc"] = "2.0";
    response["id"] = obj.value("id");
    if(result.isError) {
        QJsonObject error;
        error["code"] = result.error_code;
        if(!result.error_message.isEmpty()) {
            error["message"] = result.error_message;
        }
        response["error"] = error;
    } else {
        response["result"] = QJsonValue::fromVariant(result.result);
    }

    pClient->sendTextMessage(QJsonDocument(response).toJson());
}

void JsonRPCServer::processBinaryMessage(QByteArray message)
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    qDebug() << "Binary Message received:" << message;
    if (pClient) {
        pClient->sendBinaryMessage(message);
    }
}

void JsonRPCServer::socketDisconnected()
{
    QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
    qDebug() << "socketDisconnected:" << pClient;
    if (pClient) {
        connections.removeAll(pClient);
        pClient->deleteLater();
    }
}

