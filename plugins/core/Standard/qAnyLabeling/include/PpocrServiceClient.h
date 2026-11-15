// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// aicore-boundary: remote-inference-transport
// Official PaddleOCR asynchronous document service; no local models/graphs.
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QPointer>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <functional>

class QNetworkReply;
class PpocrServiceClient final : public QObject {
    Q_OBJECT
public:
    struct Settings {
        QUrl jobsUrl{QStringLiteral(
                "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")};
        QString apiKey;
        QString model{QStringLiteral("PaddleOCR-VL-1.6")};
        int timeoutMs = 180000;
    };
    struct RemoteSettings {
        QString serverUrl{QStringLiteral("http://127.0.0.1:8000")};
        QString apiKey;
        int timeoutMs = 180000;
    };
    explicit PpocrServiceClient(QObject* parent = nullptr);
    ~PpocrServiceClient() override;
    bool busy() const { return m_busy; }
    void parseFile(const QString& path, const Settings& settings);
    void probeService(const RemoteSettings& settings,
                      const QString& selectedModel = {});
    void parseRemoteFile(const QString& path,
                         const RemoteSettings& settings,
                         const QString& model);
    void cancel();
    static QJsonObject optionalPayload();
signals:
    void progress(const QString& message);
    void finished(const QJsonObject& document, const QString& error);
    void serviceProbed(const QJsonArray& models,
                       const QString& selectedModel,
                       const QString& baseUrl,
                       const QString& error);

private:
    using Response = std::function<void(const QByteArray&)>;
    using Failure = std::function<void(const QString&)>;
    void watch(QNetworkReply* reply,
               Response response,
               int timeoutMs = 0,
               Failure failure = {});
    void poll();
    void download(const QUrl& url);
    void finish(const QJsonObject& document, const QString& error);
    void beginProbe(const RemoteSettings& settings,
                    const QString& selectedModel,
                    const QString& path);
    void probeNext();
    void predictRemote();
    QNetworkAccessManager m_network;
    QPointer<QNetworkReply> m_reply;
    Failure m_failure;
    QTimer m_timeout, m_poll;
    QElapsedTimer m_pollDeadline;
    Settings m_settings;
    RemoteSettings m_remoteSettings;
    QStringList m_probeUrls;
    int m_probeIndex = 0;
    bool m_probing = false;
    QString m_remotePath, m_remoteModel, m_remoteBase, m_probeError;
    QStringList m_remotePages;
    QJsonArray m_remoteResults, m_remoteSizes;
    QJsonObject m_remoteImages;
    QString m_job, m_path;
    bool m_busy = false;
    quint64 m_generation = 0;
};
