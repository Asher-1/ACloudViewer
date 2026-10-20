// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ChatServiceClient.h"

// aicore-boundary: remote-inference-transport
#include <QFile>
#include <QJsonDocument>
#include <QMimeDatabase>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <algorithm>

namespace {
QString serviceError(const QJsonObject& json) {
    const auto error = json.value(QStringLiteral("error"));
    return error.isObject() ? error.toObject()
                                      .value(QStringLiteral("message"))
                                      .toString()
                            : error.toString();
}
}  // namespace

ChatServiceClient::ChatServiceClient(QObject* parent) : QObject(parent) {}
ChatServiceClient::~ChatServiceClient() { cancel(); }

QJsonObject ChatServiceClient::userMessage(const QString& text,
                                           const QString& imagePath,
                                           QString* error) {
    if (error) error->clear();
    QJsonValue content(text);
    if (!imagePath.isEmpty()) {
        QFile image(imagePath);
        if (!image.open(QIODevice::ReadOnly)) {
            if (error) *error = tr("Unable to read image: %1").arg(imagePath);
            return {};
        }
        const QString mime = QMimeDatabase().mimeTypeForFile(imagePath).name();
        if (!mime.startsWith(QStringLiteral("image/"))) {
            if (error) *error = tr("Unsupported image: %1").arg(imagePath);
            return {};
        }
        content = QJsonArray{
                QJsonObject{
                        {"type", "image_url"},
                        {"image_url",
                         QJsonObject{
                                 {"url",
                                  QStringLiteral("data:%1;base64,%2")
                                          .arg(mime,
                                               QString::fromLatin1(
                                                       image.readAll()
                                                               .toBase64()))}}}},
                QJsonObject{{"type", "text"}, {"text", text}}};
    }
    return {{"role", "user"}, {"content", content}};
}

void ChatServiceClient::cancel() {
    if (!m_reply) return;
    auto* reply = m_reply.data();
    m_reply.clear();
    reply->disconnect(this);
    reply->abort();
    reply->deleteLater();
    m_buffer.clear();
}

void ChatServiceClient::fetchModels(const Settings& settings) {
    start(settings, QStringLiteral("models"), nullptr, true, false);
}

void ChatServiceClient::complete(const Settings& settings,
                                 const QJsonArray& messages,
                                 bool stream) {
    if (settings.model.trimmed().isEmpty() || messages.isEmpty()) {
        emit finished({}, tr("Choose a model and enter a message."));
        return;
    }
    QJsonObject body{{"model", settings.model},
                     {"messages", messages},
                     {"temperature", settings.temperature},
                     {"stream", stream}};
    if (settings.maxTokens > 0) body.insert("max_tokens", settings.maxTokens);
    start(settings, QStringLiteral("chat/completions"), &body, false, stream);
}

void ChatServiceClient::start(const Settings& settings,
                              const QString& endpoint,
                              const QJsonObject* body,
                              bool models,
                              bool stream) {
    cancel();
    QUrl url = settings.baseUrl;
    QString path = url.path();
    while (path.endsWith(QLatin1Char('/'))) path.chop(1);
    if (!path.endsWith(QLatin1Char('/') + endpoint))
        path += QLatin1Char('/') + endpoint;
    url.setPath(path);
    if (!url.isValid() || url.host().isEmpty() ||
        (url.scheme() != QStringLiteral("http") &&
         url.scheme() != QStringLiteral("https"))) {
        const auto error = tr("Enter a valid HTTP or HTTPS service address.");
        if (models)
            emit modelsReady({}, error);
        else
            emit finished({}, error);
        return;
    }
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    if (models &&
        settings.baseUrl.toString().contains(QStringLiteral("anthropic"))) {
        request.setRawHeader("x-api-key", settings.apiKey.trimmed().toUtf8());
        request.setRawHeader("anthropic-version", "2023-06-01");
    } else if (!settings.apiKey.isEmpty())
        request.setRawHeader("Authorization",
                             "Bearer " + settings.apiKey.toUtf8());
    auto* reply =
            body ? m_network.post(request, QJsonDocument(*body).toJson(
                                                   QJsonDocument::Compact))
                 : m_network.get(request);
    m_reply = reply;
    m_buffer.clear();
    m_text.clear();
    m_streamError.clear();
    m_streamAnswer = false;
    auto* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, reply, [reply]() {
        reply->setProperty("chatTimeout", true);
        reply->abort();
    });
    timeout->start(std::max(1, settings.timeoutMs));
    if (stream)
        connect(reply, &QIODevice::readyRead, this, [this, reply]() {
            if (m_reply != reply) return;
            m_buffer += reply->readAll();
            consumeStream(false);
        });
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, models, stream]() {
                if (m_reply != reply) return;
                m_buffer += reply->readAll();
                QString error;
                QJsonObject json;
                const bool sse =
                        stream &&
                        (reply->header(QNetworkRequest::ContentTypeHeader)
                                 .toString()
                                 .contains(
                                         QStringLiteral("text/event-stream")) ||
                         m_buffer.startsWith("data:") || !m_text.isEmpty());
                if (sse)
                    consumeStream(true);
                else {
                    QJsonParseError parse;
                    const auto document =
                            QJsonDocument::fromJson(m_buffer, &parse);
                    if (parse.error != QJsonParseError::NoError ||
                        !document.isObject())
                        error = tr("Invalid JSON response from chat service.");
                    else
                        json = document.object();
                }
                const auto detail = serviceError(json);
                if (!detail.isEmpty()) error = detail;
                if (!m_streamError.isEmpty()) error = m_streamError;
                if (reply->property("chatTimeout").toBool())
                    error = tr("Chat request timed out.");
                else if (reply->error() != QNetworkReply::NoError &&
                         detail.isEmpty())
                    error = reply->errorString();
                m_reply.clear();
                reply->deleteLater();
                if (models) {
                    if (error.isEmpty() && !json.value("data").isArray())
                        error = tr("The service returned no model list.");
                    emit modelsReady(json.value("data").toArray(), error);
                } else {
                    if (sse && error.isEmpty() && !m_streamAnswer)
                        error = tr("The service returned no answer.");
                    if (!sse && error.isEmpty()) {
                        const auto choices = json.value("choices").toArray();
                        const auto message = choices.isEmpty()
                                                     ? QJsonObject()
                                                     : choices.first()
                                                               .toObject()
                                                               .value("message")
                                                               .toObject();
                        if (!message.value("content").isString())
                            error = tr("The service returned no answer.");
                        else
                            m_text = message.value("content").toString();
                    }
                    emit finished(m_text, error);
                }
            });
}

void ChatServiceClient::consumeStream(bool final) {
    // Parse complete SSE lines as bytes, then decode JSON: UTF-8 characters
    // and event records may be split across arbitrary network packets.
    while (true) {
        int end = m_buffer.indexOf('\n');
        if (end < 0) {
            if (!final || m_buffer.isEmpty()) break;
            end = m_buffer.size();
        }
        const auto line = m_buffer.left(end).trimmed();
        if (!line.startsWith("data:") && !line.isEmpty() &&
            !line.startsWith(':') && !line.startsWith("event:")) {
            // A non-stream JSON response is retained for the finished path.
            if (!final) break;
        }
        m_buffer.remove(0, end + 1);
        if (!line.startsWith("data:")) continue;
        const auto data = line.mid(5).trimmed();
        if (data == "[DONE]") continue;
        QJsonParseError parse;
        const auto document = QJsonDocument::fromJson(data, &parse);
        if (parse.error != QJsonParseError::NoError || !document.isObject()) {
            m_streamError = tr("Invalid chat stream event.");
            continue;
        }
        const auto json = document.object();
        const auto error = serviceError(json);
        if (!error.isEmpty()) m_streamError = error;
        const auto choices = json.value("choices").toArray();
        if (choices.isEmpty()) continue;  // e.g. final usage-only event.
        m_streamAnswer = true;
        const auto delta = choices.first()
                                   .toObject()
                                   .value("delta")
                                   .toObject()
                                   .value("content")
                                   .toString();
        m_text += delta;
        if (!delta.isEmpty()) emit textReceived(delta);
    }
}
