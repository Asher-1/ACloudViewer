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
#include <memory>

class JsonRPCDeferred;

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

    /// The answer arrives later through `reply` (a child process is
    /// running); the server sends it when the handler completes it.
    static JsonRPCResult deferred(std::shared_ptr<JsonRPCDeferred> reply) {
        JsonRPCResult r;
        r.pending = std::move(reply);
        return r;
    }

    bool isError{true};
    int error_code{-32601};
    QString error_message = "Method not found";
    QVariant error_data;
    QVariant result;
    std::shared_ptr<JsonRPCDeferred> pending;
};

/// One answer completed after its handler returned, on the GUI thread. The
/// consumer attaches onComplete() once execute() has returned; a result
/// completed before that (a child that failed to start) is held until then.
class JsonRPCDeferred {
public:
    void complete(const JsonRPCResult& result) {
        if (m_done) return;
        m_done = true;
        m_result = result;
        if (m_reply) m_reply(m_result);
    }

    void onComplete(std::function<void(const JsonRPCResult&)> reply) {
        m_reply = std::move(reply);
        if (m_done) m_reply(m_result);
    }

private:
    bool m_done{false};
    JsonRPCResult m_result;
    std::function<void(const JsonRPCResult&)> m_reply;
};

class JsonRPCServer : public QObject {
    Q_OBJECT
public:
    explicit JsonRPCServer(QObject* parent = nullptr);
    ~JsonRPCServer();

    bool listen(quint16 port);
    bool isListening() const;
    QString errorString() const;
    void close();

signals:
    JsonRPCResult execute(QString method, QMap<QString, QVariant> params);
private slots:
    void onNewConnection();
    void onClosed();
    void processTextMessage(QString message);
    void processBinaryMessage(QByteArray message);
    void socketDisconnected();
    void onOriginAuthenticationRequired(
            QWebSocketCorsAuthenticator* authenticator);

private:
    QWebSocketServer* ws_server{nullptr};
    QList<QWebSocket*> connections;
};
