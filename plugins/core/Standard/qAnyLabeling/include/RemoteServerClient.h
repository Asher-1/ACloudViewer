// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QObject>
#include <QPoint>
#include <QString>
#include <QStringList>
#include <QVariantMap>

class QImage;

/** Client for the upstream remote inference service. The remote service is
 *  the inference backend by explicit user configuration: frames travel out
 *  as base64 data URIs and annotated shapes come back as JSON, so the
 *  plugin implements transport only (no private model cache, scheduler, or
 *  graph). Transport maps onto Qt-native QNetworkAccessManager/QImage/
 *  QJsonDocument — zero third-party dependencies.
 *  Protocol parity: anylabeling/services/auto_labeling/remote_server.py. */
class RemoteServerClient final : public QObject {
    Q_OBJECT
public:
    struct Settings {
        QString serverUrl;
        QString apiKey;
        int timeoutSec = 30;
    };

    explicit RemoteServerClient(QObject* parent = nullptr);

    void applySettings(const Settings& settings);
    Settings settings() const;

    /** GET {server}/v1/models -> modelsFetched(data, error). "data" carries
     *  model_id -> {classes, filter_classes, tasks...} exactly as served. */
    void fetchModels();

    /** POST {server}/v1/predict -> predictFinished(shapes, error). The
     *  image travels as a base64 data URI: the original file bytes when
     *  imagePath exists (lossless, upstream parity), otherwise the QImage
     *  re-encoded as PNG. extraParams carries the upstream params object
     *  (conf_threshold, iou_threshold, text_prompt, marks, current_task...). */
    void predict(const QImage& image,
                 const QString& imagePath,
                 const QString& modelId,
                 const QVariantMap& extraParams);

    /** One-shot tracker reset (upstream set_auto_labeling_reset_tracker):
     *  the next predict() injects params.reset_tracker=true and clears the
     *  flag, exactly like the source client. */
    void setResetTracker();

    /** Encode a frame list for /v1/video/init: the original file bytes as
     *  data URIs (upstream parity with the video-session frame packing). */
    QStringList frameDataUris(const QStringList& imagePaths) const;

    // Video session endpoints (upstream _handle_video_prompt /
    // _handle_video_point_prompt / _handle_video_propagation). A session
    // carries frames to the server once, then text or point prompts land on
    // single frames and propagation streams per-frame masks back over SSE.
    bool hasVideoSession() const;

    /** POST /v1/video/init with the frame data URIs (timeout x2, upstream
     *  parity) -> videoInitFinished(sessionId, error). */
    void videoInit(const QStringList& frameDataUris, const QString& modelId);

    /** POST /v1/video/prompt with a text prompt ->
     *  videoPromptFinished(masks, error). masks are the raw server shapes
     *  (label/shape_type/points/score/group_id). */
    void videoPromptText(const QString& textPrompt,
                         int frameIndex,
                         const QString& modelId,
                         const QVariantMap& params);

    /** POST /v1/video/prompt with point prompts (relative coords plus
     *  0/1 labels, obj_id pinned to 10000 as upstream) ->
     *  videoPromptFinished(masks, error). */
    void videoPromptPoints(const QVariantList& relativePoints,
                           const QVariantList& pointLabels,
                           int frameIndex,
                           const QString& modelId,
                           const QVariantMap& params);

    /** POST /v1/video/propagate/stream and parse the SSE events: started /
     *  progress / completed / error. A negative startFrame omits the field;
     *  endFrame is exclusive as upstream (sent as end_frame - 1).
     *  videoPropagateFinished carries results as {frameIndex: masks}. The
     *  stream itself has no timeout (upstream timeout=None). */
    void videoPropagate(const QString& modelId, int startFrame, int endFrame);

    /** POST /v1/video/cleanup/{sessionId} and clear local session state;
     *  failures still reset the local state (upstream parity). */
    void resetVideoSession();

signals:
    void modelsFetched(const QJsonObject& data, const QString& error);
    void predictFinished(const QJsonArray& shapes, const QString& error);
    void videoInitFinished(const QString& sessionId, const QString& error);
    void videoPromptFinished(const QJsonArray& masks, const QString& error);
    void videoPropagateStarted(int totalFrames, int startFrameIndex);
    void videoPropagateProgress(int currentFrame);
    void videoPropagateFinished(const QJsonObject& results,
                                const QString& error);

private:
    QString dataUri(const QImage& image, const QString& imagePath) const;
    QNetworkRequest jsonRequest(const QString& path) const;
    void handleStreamEvent(const QJsonObject& event);

    QNetworkAccessManager m_nam;
    Settings m_settings;
    QString m_videoSessionId;
    QString m_videoModelId;
    bool m_resetTracker = false;
    QByteArray m_streamBuffer;
};
