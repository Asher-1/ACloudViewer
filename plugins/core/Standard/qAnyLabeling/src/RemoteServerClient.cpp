// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RemoteServerClient.h"

#include <QBuffer>
#include <QEventLoop>
#include <QFile>
#include <QHash>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPointer>
#include <QStringList>
#include <QTimer>
#include <QUrl>

#include "AnnotationLogging.h"

namespace {

constexpr int kMinTimeoutSec = 5;

QString trimServerUrl(QString url) {
    while (url.endsWith(QLatin1Char('/'))) url.chop(1);
    return url;
}

QJsonObject parseJsonObject(const QByteArray& body, QString& error) {
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(body, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        error = QStringLiteral("Invalid JSON from remote server: %1")
                        .arg(parseError.errorString());
        return {};
    }
    if (!document.isObject()) {
        error = QStringLiteral("Remote server returned a non-object body");
        return {};
    }
    return document.object();
}

// Upstream strips trailing periods from text prompts before sending.
QString stripPromptSuffix(QString text) {
    while (text.endsWith(QLatin1Char('.'))) text.chop(1);
    return text;
}

}  // namespace

RemoteServerClient::RemoteServerClient(QObject* parent) : QObject(parent) {}

void RemoteServerClient::applySettings(const Settings& settings) {
    m_settings = settings;
    if (m_settings.timeoutSec < kMinTimeoutSec)
        m_settings.timeoutSec = kMinTimeoutSec;
    m_settings.serverUrl = trimServerUrl(m_settings.serverUrl);
}

RemoteServerClient::Settings RemoteServerClient::settings() const {
    return m_settings;
}

void RemoteServerClient::fetchModels() {
    if (m_settings.serverUrl.isEmpty()) {
        emit modelsFetched({}, QStringLiteral("Remote server URL is empty"));
        return;
    }
    QNetworkRequest request(
            QUrl(m_settings.serverUrl + QStringLiteral("/v1/models")));
    request.setHeader(QNetworkRequest::ContentTypeHeader,
                      QStringLiteral("application/json"));
    if (!m_settings.apiKey.isEmpty())
        request.setRawHeader(QByteArrayLiteral("Token"),
                             m_settings.apiKey.toUtf8());

    // Qt 5.12 has no QNetworkReply::setTransferTimeout; a one-shot timer
    // aborts the reply on the configured budget instead.
    QNetworkReply* reply = m_nam.get(request);
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 1000);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        QString error;
        if (reply->error() != QNetworkReply::NoError)
            error = QStringLiteral("Model list request failed: %1")
                            .arg(reply->errorString());
        QJsonObject body;
        if (error.isEmpty()) body = parseJsonObject(reply->readAll(), error);
        if (error.isEmpty()) {
            const QJsonObject data =
                    body.value(QStringLiteral("data")).toObject();
            if (data.isEmpty()) {
                error = QStringLiteral("Remote server returned no models");
            } else {
                emit modelsFetched(data, QString());
                return;
            }
        }
        qany_log::log(QStringLiteral("[Error] Remote server model list: %1")
                              .arg(error));
        emit modelsFetched({}, error);
    });
}

QString RemoteServerClient::dataUri(const QImage& image,
                                    const QString& imagePath) const {
    QByteArray base64;
    QString mime = QStringLiteral("image/png");
    const QString suffix =
            imagePath.mid(imagePath.lastIndexOf(QLatin1Char('.')) + 1)
                    .toLower();
    const QHash<QString, QString> mimeBySuffix = {
            {QStringLiteral("jpg"), QStringLiteral("image/jpeg")},
            {QStringLiteral("jpeg"), QStringLiteral("image/jpeg")},
            {QStringLiteral("png"), QStringLiteral("image/png")},
            {QStringLiteral("bmp"), QStringLiteral("image/bmp")},
            {QStringLiteral("webp"), QStringLiteral("image/webp")},
    };
    // Upstream parity: the original file bytes travel out when available so
    // the remote service sees exactly what the user annotated.
    QFile file(imagePath);
    if (!imagePath.isEmpty() && suffix.length() <= 5 &&
        file.open(QIODevice::ReadOnly)) {
        base64 = file.readAll().toBase64();
        mime = mimeBySuffix.value(suffix, QStringLiteral("image/jpeg"));
    } else {
        QByteArray png;
        QBuffer buffer(&png);
        buffer.open(QIODevice::WriteOnly);
        image.save(&buffer, "PNG");
        base64 = png.toBase64();
    }
    return QStringLiteral("data:%1;base64,%2")
            .arg(mime, QString::fromLatin1(base64));
}

void RemoteServerClient::predict(const QImage& image,
                                 const QString& imagePath,
                                 const QString& modelId,
                                 const QVariantMap& extraParams) {
    if (m_settings.serverUrl.isEmpty()) {
        emit predictFinished({}, QStringLiteral("Remote server URL is empty"));
        return;
    }
    if (modelId.isEmpty()) {
        emit predictFinished({}, QStringLiteral("No remote model selected"));
        return;
    }
    const QString imageUri = dataUri(image, imagePath);
    if (imageUri.size() <= QStringLiteral("data:;base64,").size()) {
        emit predictFinished({}, QStringLiteral("Failed to encode the image"));
        return;
    }
    QJsonObject params;
    for (auto it = extraParams.constBegin(); it != extraParams.constEnd(); ++it)
        params.insert(it.key(), QJsonValue::fromVariant(it.value()));
    // One-shot upstream set_auto_labeling_reset_tracker parity: the flag
    // rides the next predict() params and clears itself.
    if (m_resetTracker) {
        params.insert(QStringLiteral("reset_tracker"), true);
        m_resetTracker = false;
    }
    QJsonObject payload;
    payload.insert(QStringLiteral("model"), modelId);
    payload.insert(QStringLiteral("image"), imageUri);
    payload.insert(QStringLiteral("params"), params);

    QNetworkRequest request(
            QUrl(m_settings.serverUrl + QStringLiteral("/v1/predict")));
    request.setHeader(QNetworkRequest::ContentTypeHeader,
                      QStringLiteral("application/json"));
    if (!m_settings.apiKey.isEmpty())
        request.setRawHeader(QByteArrayLiteral("Token"),
                             m_settings.apiKey.toUtf8());

    QNetworkReply* reply = m_nam.post(
            request, QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 1000);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        QString error;
        if (reply->error() != QNetworkReply::NoError)
            error = QStringLiteral("Remote prediction failed: %1")
                            .arg(reply->errorString());
        QJsonObject body;
        if (error.isEmpty()) body = parseJsonObject(reply->readAll(), error);
        QJsonArray shapes;
        if (error.isEmpty()) {
            const QJsonObject data =
                    body.value(QStringLiteral("data")).toObject();
            shapes = data.value(QStringLiteral("shapes")).toArray();
            if (shapes.isEmpty() && !data.contains(QStringLiteral("shapes")))
                error = QStringLiteral("Remote server returned no shapes");
        }
        if (!error.isEmpty())
            qany_log::log(
                    QStringLiteral("[Error] Remote prediction: %1").arg(error));
        emit predictFinished(shapes, error);
    });
}

void RemoteServerClient::setResetTracker() { m_resetTracker = true; }

QStringList RemoteServerClient::frameDataUris(
        const QStringList& imagePaths) const {
    QStringList uris;
    uris.reserve(imagePaths.size());
    for (const QString& path : imagePaths) uris.append(dataUri(QImage(), path));
    return uris;
}

bool RemoteServerClient::hasVideoSession() const {
    return !m_videoSessionId.isEmpty();
}

QNetworkRequest RemoteServerClient::jsonRequest(const QString& path) const {
    QNetworkRequest request(QUrl(m_settings.serverUrl + path));
    request.setHeader(QNetworkRequest::ContentTypeHeader,
                      QStringLiteral("application/json"));
    if (!m_settings.apiKey.isEmpty())
        request.setRawHeader(QByteArrayLiteral("Token"),
                             m_settings.apiKey.toUtf8());
    return request;
}

void RemoteServerClient::videoInit(const QStringList& frameDataUris,
                                   const QString& modelId) {
    if (m_settings.serverUrl.isEmpty()) {
        emit videoInitFinished({},
                               QStringLiteral("Remote server URL is empty"));
        return;
    }
    if (modelId.isEmpty() || frameDataUris.isEmpty()) {
        emit videoInitFinished(
                {}, QStringLiteral("No remote model or frames to initialize"));
        return;
    }
    QJsonObject payload;
    payload.insert(QStringLiteral("model"), modelId);
    QJsonArray frames;
    for (const QString& uri : frameDataUris) frames.append(uri);
    payload.insert(QStringLiteral("frames"), frames);
    payload.insert(QStringLiteral("start_frame_index"), 0);

    // Upstream doubles the timeout for the session initialization round
    // (frame upload + server-side warmup is the slowest call).
    QNetworkReply* reply =
            m_nam.post(jsonRequest(QStringLiteral("/v1/video/init")),
                       QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 2 * 1000);
    QObject::connect(
            reply, &QNetworkReply::finished, this, [this, reply, modelId]() {
                reply->deleteLater();
                QString error;
                if (reply->error() != QNetworkReply::NoError)
                    error = QStringLiteral("Video init failed: %1")
                                    .arg(reply->errorString());
                QJsonObject body;
                if (error.isEmpty())
                    body = parseJsonObject(reply->readAll(), error);
                QString sessionId;
                if (error.isEmpty()) {
                    sessionId = body.value(QStringLiteral("data"))
                                        .toObject()
                                        .value(QStringLiteral("session_id"))
                                        .toString();
                    if (sessionId.isEmpty())
                        error = QStringLiteral(
                                "Video init returned no session id");
                }
                if (!error.isEmpty())
                    qany_log::log(
                            QStringLiteral("[Error] Remote video init: %1")
                                    .arg(error));
                if (error.isEmpty()) {
                    m_videoSessionId = sessionId;
                    m_videoModelId = modelId;
                }
                emit videoInitFinished(m_videoSessionId, error);
            });
}

void RemoteServerClient::videoPromptText(const QString& textPrompt,
                                         int frameIndex,
                                         const QString& modelId,
                                         const QVariantMap& params) {
    if (m_videoSessionId.isEmpty()) {
        emit videoPromptFinished(
                {}, QStringLiteral("No video session initialized"));
        return;
    }
    QJsonObject payload;
    payload.insert(QStringLiteral("session_id"), m_videoSessionId);
    payload.insert(QStringLiteral("model"), modelId);
    payload.insert(QStringLiteral("text_prompt"),
                   stripPromptSuffix(textPrompt));
    payload.insert(QStringLiteral("frame_index"), frameIndex);
    QJsonObject paramsJson;
    for (auto it = params.constBegin(); it != params.constEnd(); ++it)
        paramsJson.insert(it.key(), QJsonValue::fromVariant(it.value()));
    payload.insert(QStringLiteral("params"), paramsJson);

    QNetworkReply* reply =
            m_nam.post(jsonRequest(QStringLiteral("/v1/video/prompt")),
                       QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 1000);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        QString error;
        if (reply->error() != QNetworkReply::NoError)
            error = QStringLiteral("Video prompt failed: %1")
                            .arg(reply->errorString());
        const QByteArray raw = reply->readAll();
        QJsonObject body;
        if (error.isEmpty()) body = parseJsonObject(raw, error);
        QJsonArray masks;
        if (error.isEmpty()) {
            masks = body.value(QStringLiteral("data"))
                            .toObject()
                            .value(QStringLiteral("masks"))
                            .toArray();
        }
        if (!error.isEmpty()) {
            qany_log::log(QStringLiteral("[Error] Remote video prompt: %1")
                                  .arg(error));
            resetVideoSession();
        }
        emit videoPromptFinished(masks, error);
    });
}

void RemoteServerClient::videoPromptPoints(const QVariantList& relativePoints,
                                           const QVariantList& pointLabels,
                                           int frameIndex,
                                           const QString& modelId,
                                           const QVariantMap& params) {
    if (m_videoSessionId.isEmpty()) {
        emit videoPromptFinished(
                {}, QStringLiteral("No video session initialized"));
        return;
    }
    if (relativePoints.isEmpty()) {
        emit videoPromptFinished({}, QStringLiteral("No valid point prompts"));
        return;
    }
    QJsonObject payload;
    payload.insert(QStringLiteral("session_id"), m_videoSessionId);
    payload.insert(QStringLiteral("model"), modelId);
    payload.insert(QStringLiteral("frame_index"), frameIndex);
    QJsonArray points;
    for (const QVariant& point : relativePoints)
        points.append(QJsonValue::fromVariant(point));
    payload.insert(QStringLiteral("points"), points);
    QJsonArray labels;
    for (const QVariant& label : pointLabels)
        labels.append(QJsonValue::fromVariant(label));
    payload.insert(QStringLiteral("point_labels"), labels);
    // Upstream pins the point-prompt object id to a fixed sentinel.
    payload.insert(QStringLiteral("obj_id"), 10000);
    QJsonObject paramsJson;
    for (auto it = params.constBegin(); it != params.constEnd(); ++it)
        paramsJson.insert(it.key(), QJsonValue::fromVariant(it.value()));
    payload.insert(QStringLiteral("params"), paramsJson);

    QNetworkReply* reply =
            m_nam.post(jsonRequest(QStringLiteral("/v1/video/prompt")),
                       QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 1000);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        QString error;
        if (reply->error() != QNetworkReply::NoError)
            error = QStringLiteral("Video prompt failed: %1")
                            .arg(reply->errorString());
        QJsonObject body;
        if (error.isEmpty()) body = parseJsonObject(reply->readAll(), error);
        QJsonArray masks;
        if (error.isEmpty()) {
            masks = body.value(QStringLiteral("data"))
                            .toObject()
                            .value(QStringLiteral("masks"))
                            .toArray();
        }
        if (!error.isEmpty()) {
            qany_log::log(QStringLiteral("[Error] Remote video prompt: %1")
                                  .arg(error));
            resetVideoSession();
        }
        emit videoPromptFinished(masks, error);
    });
}

void RemoteServerClient::handleStreamEvent(const QJsonObject& event) {
    const QString type = event.value(QStringLiteral("type")).toString();
    if (type == QLatin1String("started")) {
        emit videoPropagateStarted(
                event.value(QStringLiteral("total_frames")).toInt(),
                event.value(QStringLiteral("start_frame_index")).toInt());
    } else if (type == QLatin1String("progress")) {
        emit videoPropagateProgress(
                event.value(QStringLiteral("current_frame")).toInt());
    } else if (type == QLatin1String("completed")) {
        const QJsonObject results =
                event.value(QStringLiteral("results")).toObject();
        emit videoPropagateFinished(results, QString());
    } else if (type == QLatin1String("error")) {
        const QString message =
                event.value(QStringLiteral("message")).toString();
        qany_log::log(
                QStringLiteral("[Error] Remote propagation: %1").arg(message));
        resetVideoSession();
        emit videoPropagateFinished({}, message);
    }
    // Unknown event types are ignored, matching upstream's switch.
}

void RemoteServerClient::videoPropagate(const QString& modelId,
                                        int startFrame,
                                        int endFrame) {
    if (m_videoSessionId.isEmpty()) {
        emit videoPropagateFinished(
                {}, QStringLiteral("No video session initialized"));
        return;
    }
    QJsonObject payload;
    payload.insert(QStringLiteral("session_id"), m_videoSessionId);
    payload.insert(QStringLiteral("model"), modelId);
    if (startFrame >= 0)
        payload.insert(QStringLiteral("start_frame"), startFrame);
    if (endFrame >= 0)
        // Upstream sends an inclusive end_frame from an exclusive range.
        payload.insert(QStringLiteral("end_frame"), endFrame - 1);

    // SSE stream: no timer timeout (upstream timeout=None); the reply is
    // consumed line by line and drained on finished.
    m_streamBuffer.clear();
    QNetworkReply* reply = m_nam.post(
            jsonRequest(QStringLiteral("/v1/video/propagate/stream")),
            QJsonDocument(payload).toJson(QJsonDocument::Compact));
    QObject::connect(reply, &QNetworkReply::readyRead, this, [this, reply]() {
        m_streamBuffer += reply->readAll();
        int newline = -1;
        while ((newline = m_streamBuffer.indexOf('\n')) >= 0) {
            const QByteArray line = m_streamBuffer.left(newline).trimmed();
            m_streamBuffer.remove(0, newline + 1);
            if (!line.startsWith("data: ")) continue;
            QString error;
            const QJsonObject event = parseJsonObject(line.mid(6), error);
            if (!error.isEmpty()) continue;  // keep-alive/comment
            handleStreamEvent(event);
        }
    });
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            const QString error =
                    QStringLiteral("Propagation stream failed: %1")
                            .arg(reply->errorString());
            qany_log::log(QStringLiteral("[Error] Remote propagation: %1")
                                  .arg(error));
            resetVideoSession();
            emit videoPropagateFinished({}, error);
        }
        // A clean stream already emitted videoPropagateFinished
        // from the completed SSE event; nothing to do here.
    });
}

void RemoteServerClient::resetVideoSession() {
    const QString sessionId = m_videoSessionId;
    const QString modelId = m_videoModelId;
    m_videoSessionId.clear();
    m_videoModelId.clear();
    if (sessionId.isEmpty()) return;
    // Upstream posts the owning model id as a query parameter; failures
    // only log — the local session state is already reset.
    const QString path = QStringLiteral("/v1/video/cleanup/") +
                         QString::fromUtf8(QUrl::toPercentEncoding(sessionId)) +
                         QStringLiteral("?model=") +
                         QString::fromUtf8(QUrl::toPercentEncoding(modelId));
    QNetworkReply* reply = m_nam.post(jsonRequest(path), QByteArray());
    QTimer* timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    QObject::connect(timeout, &QTimer::timeout, reply, &QNetworkReply::abort);
    timeout->start(m_settings.timeoutSec * 1000);
    QObject::connect(reply, &QNetworkReply::finished, this, [reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError)
            qany_log::log(
                    QStringLiteral("[Warning] Failed to cleanup remote video "
                                   "session"));
    });
}
