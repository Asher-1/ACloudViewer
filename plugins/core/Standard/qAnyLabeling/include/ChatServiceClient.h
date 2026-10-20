// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// aicore-boundary: remote-inference-transport
// User-configured chat service transport only; no local model assets/graphs.
#include <QJsonArray>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QObject>
#include <QPointer>
#include <QUrl>

class QNetworkReply;

class ChatServiceClient final : public QObject {
    Q_OBJECT
public:
    struct Settings {
        QUrl baseUrl{QStringLiteral("http://localhost:11434/v1")};
        QString apiKey{QStringLiteral("ollama")};
        QString model;
        double temperature = 1.0;  // Chatbot: source slider 10 / 10.
        int maxTokens = 0;         // Unset, like the source default.
        int timeoutMs = 120000;
    };
    explicit ChatServiceClient(QObject* parent = nullptr);
    ~ChatServiceClient() override;
    void fetchModels(const Settings& settings);
    void complete(const Settings& settings,
                  const QJsonArray& messages,
                  bool stream = true);
    void cancel();
    bool busy() const { return !m_reply.isNull(); }
    static QJsonObject userMessage(const QString& text,
                                   const QString& imagePath,
                                   QString* error = nullptr);

signals:
    void modelsReady(const QJsonArray& models, const QString& error);
    void textReceived(const QString& delta);
    void finished(const QString& text, const QString& error);

private:
    void start(const Settings& settings,
               const QString& endpoint,
               const QJsonObject* body,
               bool models,
               bool stream);
    void consumeStream(bool final);
    QNetworkAccessManager m_network;
    QPointer<QNetworkReply> m_reply;
    QByteArray m_buffer;
    QString m_text;
    QString m_streamError;
    bool m_streamAnswer = false;
};
