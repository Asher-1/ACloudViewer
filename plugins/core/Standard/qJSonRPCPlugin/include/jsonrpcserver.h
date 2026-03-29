// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QJsonDocument>
#include <QJsonObject>
#include <QMap>
#include <QObject>
#include <QWebSocket>
#include <QWebSocketServer>
// STL
#include <functional>

struct JsonRPCResult {
    static JsonRPCResult error(int code,
                               const QString& message,
                               const QVariant& data = {}) {
        JsonRPCResult r;
        r.isError = true;
        r.error_code = code;
        r.error_message = message;
        r.error_data = data;
        return r;
    }

    static JsonRPCResult error(int code,
                               const QString& message,
                               const QJsonObject& data) {
        return error(code, message, QJsonDocument(data).toVariant());
    }

    static JsonRPCResult success(const QVariant& value) {
        JsonRPCResult r;
        r.isError = false;
        r.result = value;
        return r;
    }

    bool isError{true};
    int error_code{-32601};
    QString error_message = "Method not found";
    QVariant error_data;
    QVariant result;
};

class JsonRPCServer : public QObject {
    Q_OBJECT
public:
    explicit JsonRPCServer(QObject* parent = nullptr);
    ~JsonRPCServer();

    void listen(unsigned int port);
    void close();

signals:
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);
private slots:
    void onNewConnection();
    void onClosed();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();

private:
    QWebSocketServer* ws_server{nullptr};
    QList<QWebSocket*> connections;
};
