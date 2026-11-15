// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PpocrServiceClient.h"

// aicore-boundary: remote-inference-transport
// Document-service requests and result images; local model assets use AICore.

#include <QBuffer>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHttpMultiPart>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPainter>
#include <QRegularExpression>
#include <QSaveFile>
#include <algorithm>
#include <cmath>
#include <limits>

#include "AnnotationOrderedJson.h"
#include "PpocrDocument.h"

namespace {
QJsonObject payload(const QJsonObject& object) {
    for (const auto& key : {"result", "data", "job"})
        if (object.value(key).isObject()) return object.value(key).toObject();
    return object;
}
QString first(const QJsonObject& object,
              std::initializer_list<const char*> keys) {
    for (auto key : keys) {
        const auto value = object.value(key);
        const auto text = value.isString() ? value.toString().trimmed()
                          : value.isDouble()
                                  ? QString::number(value.toDouble(), 'g', 16)
                                  : QString();
        if (!text.isEmpty()) return text;
    }
    return {};
}
QString resultUrl(const QJsonObject& data) {
    QVector<QJsonObject> containers{data};
    for (const auto& key : {"resultUrl", "result_url", "result"}) {
        const auto value = data.value(key);
        if (value.isString() && !value.toString().trimmed().isEmpty())
            return value.toString().trimmed();
        if (value.isObject()) containers.append(value.toObject());
    }
    for (const auto& container : containers) {
        const auto url =
                first(container, {"jsonUrl", "json_url", "url", "downloadUrl"});
        if (!url.isEmpty()) return url;
    }
    return {};
}
QNetworkRequest request(const QUrl& url, const QString& key = {}) {
    QNetworkRequest result(url);
    result.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                        QNetworkRequest::NoLessSafeRedirectPolicy);
    result.setRawHeader("Client-Platform", "x-anylabeling-client");
    if (!key.isEmpty())
        result.setRawHeader("Authorization", "bearer " + key.toUtf8());
    return result;
}
QString apiError(const QJsonObject& object) {
    if (!object.contains("errorCode")) return {};
    bool valid = false;
    const int code = object.value("errorCode").toVariant().toInt(&valid);
    if (valid && code == 0) return {};
    auto message = first(object, {"errorMsg", "message"});
    return message.isEmpty() ? QStringLiteral("PaddleOCR API failed") : message;
}

QNetworkRequest serverRequest(const QString& url, const QString& key) {
    auto result = request(QUrl(url));
    result.setRawHeader("Content-Type", "application/json");
    if (!key.isEmpty()) result.setRawHeader("Token", key.toUtf8());
    return result;
}
bool pipelineCapability(const QJsonObject& model) {
    const auto capability =
            model.value("capabilities").toObject().value("ppocr_pipeline");
    if (capability.isBool()) return capability.toBool();
    if (capability.isObject()) {
        const auto enabled = capability.toObject().value("enabled");
        return !enabled.isBool() || enabled.toBool();
    }
    return capability.isString() &&
           QStringList{"1", "true", "yes", "enabled"}.contains(
                   capability.toString().toCaseFolded());
}
bool truth(const QJsonValue& value) {
    if (value.isUndefined() || value.isNull()) return false;
    if (value.isBool()) return value.toBool();
    if (value.isDouble()) return value.toDouble() != 0;
    if (value.isString()) return !value.toString().isEmpty();
    if (value.isArray()) return !value.toArray().isEmpty();
    return !value.toObject().isEmpty();
}
// Python round uses ties-to-even, regardless of the process FP rounding mode.
bool coordinate(const QJsonValue& value, int* output, bool round = true) {
    bool ok = true;
    double number = value.isString() ? value.toString().toDouble(&ok)
                    : value.isBool() ? double(value.toBool())
                    : value.isDouble()
                            ? value.toDouble()
                            : std::numeric_limits<double>::quiet_NaN();
    if (!ok || !std::isfinite(number)) return false;
    if (round) {
        const double low = std::floor(number), fraction = number - low;
        number = low +
                 (fraction > .5 || (fraction == .5 && std::fmod(low, 2.) != 0));
    } else
        number = std::trunc(number);
    if (number < std::numeric_limits<int>::min() ||
        number > std::numeric_limits<int>::max())
        return false;
    *output = int(number);
    return true;
}
int integer(const QJsonObject& object,
            const char* key,
            int fallback,
            QString* error) {
    const auto value = object.value(key);
    if (!truth(value)) return fallback;
    int result;
    if (coordinate(value, &result, false)) return result;
    *error = QStringLiteral("Invalid integer in PPOCR result: %1")
                     .arg(QString::fromLatin1(key));
    return fallback;
}
QJsonArray bbox(const QJsonValue& value, int width, int height) {
    const auto array = value.toArray();
    if (array.size() < 4) return {};
    int v[4];
    for (int i = 0; i < 4; ++i) {
        if (!coordinate(array[i], &v[i])) return {};
        v[i] = qBound(0, v[i], i % 2 ? height : width);
    }
    if (v[2] <= v[0]) v[2] = qMin(width, v[0] + 1);
    if (v[3] <= v[1]) v[3] = qMin(height, v[1] + 1);
    return {v[0], v[1], v[2], v[3]};
}
QJsonObject normalizedPage(const QJsonObject& data,
                           const QImage& image,
                           const QString& relativePath,
                           const QString& model,
                           QString* error) {
    QJsonObject page;
    if (data.value("layoutParsingResults").isArray()) {
        for (const auto& value : data.value("layoutParsingResults").toArray())
            if (value.isObject()) {
                page = value.toObject();
                break;
            }
    } else if (data.value("prunedResult").isObject())
        page = data;
    if (page.isEmpty()) {
        *error = QStringLiteral(
                "Invalid PPOCR pipeline response: missing page result");
        return {};
    }
    const auto pruned = page.value("prunedResult").toObject();
    const int width = integer(pruned, "width", image.width(), error);
    const int height = integer(pruned, "height", image.height(), error);
    if (width <= 0 || height <= 0) {
        *error = QStringLiteral("Invalid PPOCR page dimensions");
        return {};
    }
    QJsonArray blocks;
    const auto rawBlocks = pruned.value("parsing_res_list").toArray();
    for (int i = 0; i < rawBlocks.size(); ++i) {
        if (!rawBlocks[i].isObject()) continue;
        const auto block = rawBlocks[i].toObject();
        auto rawPoints = block.value("block_polygon_points");
        if (!truth(rawPoints)) rawPoints = block.value("points");
        QJsonArray points;
        for (const auto& value : rawPoints.toArray()) {
            const auto pair = value.toArray();
            if (pair.size() < 2) continue;
            int x, y;
            if (!coordinate(pair[0], &x) || !coordinate(pair[1], &y)) {
                *error = QStringLiteral("Invalid PPOCR polygon coordinate");
                return {};
            }
            points.append(QJsonArray{x, y});
        }
        auto rectangle = bbox(block.value("block_bbox"), width, height);
        if (rectangle.isEmpty() && !points.isEmpty()) {
            int x1 = std::numeric_limits<int>::max(), y1 = x1;
            int x2 = std::numeric_limits<int>::min(), y2 = x2;
            for (const auto& value : points) {
                const auto point = value.toArray();
                x1 = qMin(x1, point[0].toInt());
                y1 = qMin(y1, point[1].toInt());
                x2 = qMax(x2, point[0].toInt());
                y2 = qMax(y2, point[1].toInt());
            }
            rectangle = bbox(QJsonArray{x1, y1, x2, y2}, width, height);
        }
        if (rectangle.isEmpty()) rectangle = {0, 0, width, height};
        const int order = integer(block, "block_order", i + 1, error);
        const int group = integer(block, "group_id", order, error);
        QString label = first(block, {"block_label", "label"}).toCaseFolded();
        label.replace(QRegularExpression(QStringLiteral("[\\s-]+")), "_");
        if (label.isEmpty()) label = QStringLiteral("text");
        auto content = block.value("block_content");
        if (!truth(content)) content = block.value("description");
        blocks.append(QJsonObject{
                {"block_label", label},
                {"block_content", content.toString()},
                {"block_bbox", rectangle},
                {"block_polygon_points", points},
                {"block_id", integer(block, "block_id", order, error)},
                {"block_order", order},
                {"group_id", group},
                {"global_block_id",
                 integer(block, "global_block_id", order, error)},
                {"global_group_id",
                 integer(block, "global_group_id", group, error)}});
    }
    auto settings = pruned.value("model_settings").toObject();
    if (!settings.contains("pipeline_model"))
        settings.insert("pipeline_model", model);
    auto markdown = page.value("markdown").toObject();
    if (!truth(markdown.value("text"))) {
        QStringList paragraphs;
        for (const auto& value : blocks) {
            const auto text =
                    value.toObject().value("block_content").toString();
            if (!text.isEmpty()) paragraphs.append(text);
        }
        markdown.insert("text", paragraphs.join("\n\n"));
    }
    return {{"prunedResult",
             QJsonObject{
                     {"page_count", integer(pruned, "page_count", 1, error)},
                     {"width", width},
                     {"height", height},
                     {"model_settings", settings},
                     {"parsing_res_list", blocks}}},
            {"markdown", markdown},
            {"outputImages", page.value("outputImages").toObject()},
            {"inputImage", relativePath}};
}
}  // namespace

PpocrServiceClient::PpocrServiceClient(QObject* parent) : QObject(parent) {
    m_timeout.setSingleShot(true);
    m_poll.setSingleShot(true);
    // Coarse timers may fire before the job deadline and start an extra poll
    // with only a few milliseconds left, producing a request-timeout race.
    m_timeout.setTimerType(Qt::PreciseTimer);
    m_poll.setTimerType(Qt::PreciseTimer);
    connect(&m_timeout, &QTimer::timeout, this, [this]() {
        if (!m_failure) {
            finish({}, tr("PaddleOCR request timed out"));
            return;
        }
        auto failure = std::move(m_failure);
        if (m_reply) {
            auto* reply = m_reply.data();
            m_reply.clear();
            reply->disconnect(this);
            reply->abort();
            reply->deleteLater();
        }
        failure(tr("PaddleOCR request timed out"));
    });
    connect(&m_poll, &QTimer::timeout, this, &PpocrServiceClient::poll);
}
PpocrServiceClient::~PpocrServiceClient() {
    if (m_reply) {
        m_reply->disconnect(this);
        m_reply->abort();
    }
}
QJsonObject PpocrServiceClient::optionalPayload() {
    return {{"useDocOrientationClassify", false},
            {"useDocUnwarping", false},
            {"useLayoutDetection", true},
            {"useChartRecognition", false},
            {"useSealRecognition", true},
            {"useOcrForImageBlock", false},
            {"mergeTables", true},
            {"relevelTitles", true},
            {"layoutShapeMode", "auto"},
            {"repetitionPenalty", 1},
            {"temperature", 0},
            {"topP", 1},
            {"minPixels", 147384},
            {"maxPixels", 2822400},
            {"layoutNms", true},
            {"restructurePages", true}};
}
void PpocrServiceClient::watch(QNetworkReply* reply,
                               Response response,
                               int timeoutMs,
                               Failure failure) {
    m_reply = reply;
    m_failure = std::move(failure);
    const auto generation = m_generation;
    m_timeout.start(
            std::max(1, timeoutMs > 0 ? timeoutMs : m_settings.timeoutMs));
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, generation, response]() {
                reply->deleteLater();
                if (generation != m_generation || reply != m_reply) return;
                m_reply.clear();
                m_timeout.stop();
                auto failure = std::move(m_failure);
                if (reply->error() != QNetworkReply::NoError) {
                    if (failure)
                        failure(reply->errorString());
                    else
                        finish({}, reply->errorString());
                    return;
                }
                response(reply->readAll());
            });
}
void PpocrServiceClient::parseFile(const QString& path,
                                   const Settings& settings) {
    if (m_busy) return;
    m_busy = true;
    ++m_generation;
    m_probing = false;
    m_remotePath.clear();
    m_settings = settings;
    m_path = path;
    m_job.clear();
    if (settings.apiKey.trimmed().isEmpty()) {
        finish({}, tr("API_KEY is empty."));
        return;
    }
    if (!settings.jobsUrl.isValid() || (settings.jobsUrl.scheme() != "http" &&
                                        settings.jobsUrl.scheme() != "https")) {
        finish({}, tr("Invalid PaddleOCR jobs URL"));
        return;
    }
    auto* multipart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    auto* file = new QFile(path, multipart);
    if (!file->open(QIODevice::ReadOnly)) {
        const auto error = file->errorString();
        delete multipart;
        finish({}, error);
        return;
    }
    auto add = [multipart](const QByteArray& name, const QByteArray& bytes) {
        QHttpPart part;
        part.setHeader(QNetworkRequest::ContentDispositionHeader,
                       QStringLiteral("form-data; name=\"%1\"")
                               .arg(QString::fromLatin1(name)));
        part.setBody(bytes);
        multipart->append(part);
    };
    add("model", settings.model.toUtf8());
    add("optionalPayload",
        QJsonDocument(optionalPayload()).toJson(QJsonDocument::Compact));
    QHttpPart part;
    QString name = QFileInfo(path).fileName();
    name.replace('"', '_');
    name.remove('\r');
    name.remove('\n');
    part.setHeader(QNetworkRequest::ContentDispositionHeader,
                   QStringLiteral("form-data; name=\"file\"; filename=\"%1\"")
                           .arg(name));
    part.setBodyDevice(file);
    multipart->append(part);  // Stream the file, including PDFs.
    auto* reply = m_network.post(
            request(settings.jobsUrl, settings.apiKey.trimmed()), multipart);
    multipart->setParent(reply);
    watch(reply, [this](const QByteArray& bytes) {
        QJsonParseError error;
        const auto json = QJsonDocument::fromJson(bytes, &error);
        const auto object = json.object();
        if (error.error != QJsonParseError::NoError || !json.isObject()) {
            finish({}, tr("Invalid job response"));
            return;
        }
        const auto failure = apiError(object);
        if (!failure.isEmpty()) {
            finish({}, failure);
            return;
        }
        m_job = first(object, {"jobId", "job_id", "id", "taskId", "task_id"});
        if (m_job.isEmpty())
            m_job = first(payload(object),
                          {"jobId", "job_id", "id", "taskId", "task_id"});
        if (m_job.isEmpty()) {
            finish({}, tr("Job response missing job id"));
            return;
        }
        m_pollDeadline.start();
        poll();
    });
    emit progress(tr("Submitting document…"));
}
void PpocrServiceClient::poll() {
    if (!m_busy) return;
    if (m_pollDeadline.elapsed() >= m_settings.timeoutMs) {
        finish({}, tr("PaddleOCR job did not finish before the deadline"));
        return;
    }
    auto url = m_settings.jobsUrl;
    QString path = url.path();
    while (path.endsWith('/')) path.chop(1);
    url.setPath(path + '/' + m_job);
    watch(
            m_network.get(request(url, m_settings.apiKey.trimmed())),
            [this](const QByteArray& bytes) {
                QJsonParseError error;
                const auto json = QJsonDocument::fromJson(bytes, &error);
                const auto object = json.object();
                if (error.error != QJsonParseError::NoError ||
                    !json.isObject()) {
                    finish({}, tr("Invalid polling response"));
                    return;
                }
                const auto failure = apiError(object);
                if (!failure.isEmpty()) {
                    finish({}, failure);
                    return;
                }
                const auto data = payload(object);
                const auto state = first(data, {"state", "status"}).toLower();
                if (QStringList{"failed", "failure", "error", "cancelled",
                                "canceled"}
                            .contains(state)) {
                    auto message =
                            first(data.value("error").toObject(), {"message"});
                    if (message.isEmpty())
                        message = first(
                                data, {"errorMsg", "error_message", "message"});
                    finish({}, message.isEmpty() ? tr("PaddleOCR job failed")
                                                 : message);
                    return;
                }
                const auto link = resultUrl(data);
                if (!link.isEmpty()) {
                    download(m_settings.jobsUrl.resolved(QUrl(link)));
                    return;
                }
                if (QStringList{"done", "completed", "complete", "success",
                                "succeeded"}
                            .contains(state)) {
                    finish({}, tr("Job result missing JSON URL"));
                    return;
                }
                const int interval = std::min(
                        5000, std::max(1000, m_settings.timeoutMs / 60));
                m_poll.start(qMax(
                        1,
                        qMin(interval, m_settings.timeoutMs -
                                               int(m_pollDeadline.elapsed()))));
                emit progress(tr("Parsing document… %1").arg(state));
            },
            qMin(30000, m_settings.timeoutMs - int(m_pollDeadline.elapsed())));
}
void PpocrServiceClient::download(const QUrl& url) {
    if (!url.isValid() || (url.scheme() != "http" && url.scheme() != "https")) {
        finish({}, tr("Invalid result URL"));
        return;
    }
    // Signed result downloads can live on another host. Do not forward the API
    // key.
    watch(m_network.get(request(url)), [this](const QByteArray& bytes) {
        QString error;
        auto document = PpocrDocument::parse(bytes, &error);
        if (!error.isEmpty()) {
            finish({}, error);
            return;
        }
        auto meta = document.value("_ppocr_meta").toObject();
        meta.insert("status", "parsed");
        meta.insert(
                "source_path",
                QStringLiteral("files/%1").arg(QFileInfo(m_path).fileName()));
        meta.insert("pipeline_model", m_settings.model);
        meta.insert("api_mode", "async_jobs");
        meta.insert("api_job_id", m_job);
        meta.insert("updated_at", QDateTime::currentDateTime().toString(
                                          "yyyy-MM-dd HH:mm:ss"));
        meta.insert("error_message", QString());
        if (!meta.contains("edited_blocks"))
            meta.insert("edited_blocks", QJsonArray());
        if (!meta.contains("block_image_paths"))
            meta.insert("block_image_paths", QJsonObject());
        auto info = document.value("dataInfo").toObject();
        const auto pages = document.value("layoutParsingResults").toArray();
        QJsonArray sizes;
        for (const auto& page : pages) {
            const auto dimensions =
                    page.toObject().value("prunedResult").toObject();
            sizes.append(QJsonObject{{"width", dimensions.value("width")},
                                     {"height", dimensions.value("height")}});
        }
        info.insert("numPages", pages.size());
        info.insert("pages", sizes);
        info.insert("type", QFileInfo(m_path).suffix().compare(
                                    "pdf", Qt::CaseInsensitive) == 0
                                    ? "pdf"
                                    : "image");
        document.insert("dataInfo", info);
        document.insert("_ppocr_meta", meta);
        finish(document, {});
    });
    emit progress(tr("Downloading result…"));
}
void PpocrServiceClient::finish(const QJsonObject& document,
                                const QString& error) {
    if (!m_busy) return;
    m_busy = false;
    ++m_generation;
    m_timeout.stop();
    m_poll.stop();
    m_failure = {};
    if (m_reply) {
        auto* reply = m_reply.data();
        m_reply.clear();
        reply->disconnect(this);
        reply->abort();
        reply->deleteLater();
    }
    const bool probeOnly = m_probing && m_remotePath.isEmpty();
    m_probing = false;
    if (probeOnly)
        emit serviceProbed({}, m_remoteModel, m_remoteSettings.serverUrl,
                           error);
    else
        emit finished(document, error);
}

void PpocrServiceClient::probeService(const RemoteSettings& settings,
                                      const QString& selectedModel) {
    beginProbe(settings, selectedModel, {});
}
void PpocrServiceClient::parseRemoteFile(const QString& path,
                                         const RemoteSettings& settings,
                                         const QString& model) {
    beginProbe(settings, model, path);
}
void PpocrServiceClient::beginProbe(const RemoteSettings& settings,
                                    const QString& selectedModel,
                                    const QString& path) {
    if (m_busy) return;
    m_busy = true;
    ++m_generation;
    m_probing = true;
    m_remoteSettings = settings;
    m_remoteSettings.apiKey = settings.apiKey.trimmed();
    m_remotePath = path;
    m_remoteModel = selectedModel;
    m_remoteBase.clear();
    m_probeUrls.clear();
    m_probeIndex = 0;
    m_remotePages.clear();
    m_remoteResults = {};
    m_remoteSizes = {};
    m_remoteImages = {};
    m_probeError = tr("Remote server URL is not configured.");
    auto append = [this](QString url) {
        url = url.trimmed();
        while (url.endsWith('/')) url.chop(1);
        if (!url.isEmpty() && !m_probeUrls.contains(url))
            m_probeUrls.append(url);
    };
    append(settings.serverUrl);
    const QUrl url(settings.serverUrl.trimmed());
    if (url.isValid() && !url.host().isEmpty()) {
        auto candidatePath = url.path();
        while (candidatePath.endsWith('/')) candidatePath.chop(1);
        for (const QString suffix :
             {QStringLiteral("/v1/predict"), QStringLiteral("/predict"),
              QStringLiteral("/v1/models")}) {
            if (!candidatePath.endsWith(suffix)) continue;
            auto candidate = url;
            candidate.setPath(
                    candidatePath.left(candidatePath.size() - suffix.size()));
            candidate.setQuery(QString());
            candidate.setFragment(QString());
            append(candidate.toString());
        }
    }
    probeNext();
}
void PpocrServiceClient::probeNext() {
    if (!m_busy) return;
    if (m_probeIndex >= m_probeUrls.size()) {
        finish({}, m_probeError);
        return;
    }
    QString base = m_probeUrls[m_probeIndex++];
    if (base.endsWith("/v1/models")) base.chop(10);
    const QUrl url(base + "/v1/models");
    if (!url.isValid() || (url.scheme() != "http" && url.scheme() != "https")) {
        m_probeError = tr("Invalid remote server URL");
        probeNext();
        return;
    }
    auto failed = [this](const QString& error) {
        m_probeError = error;
        probeNext();
    };
    watch(
            m_network.get(
                    serverRequest(url.toString(), m_remoteSettings.apiKey)),
            [this, base, failed](const QByteArray& bytes) {
                QJsonParseError parseError;
                const auto json = QJsonDocument::fromJson(bytes, &parseError);
                if (parseError.error != QJsonParseError::NoError ||
                    !json.isObject() ||
                    !json.object().value("data").isObject()) {
                    failed(tr("Invalid /v1/models response data"));
                    return;
                }
                QByteArray rawModels;
                for (const auto& member :
                     AnnotationOrderedJson::members(bytes.trimmed()))
                    if (member.first == "data") rawModels = member.second;
                QJsonArray models;
                const auto data = json.object().value("data").toObject();
                for (const auto& member :
                     AnnotationOrderedJson::members(rawModels)) {
                    const auto model = data.value(member.first).toObject();
                    if (!pipelineCapability(model)) continue;
                    auto name = first(model, {"display_name"});
                    models.append(QJsonObject{
                            {"model_id", member.first},
                            {"display_name",
                             name.isEmpty() ? member.first : name}});
                }
                if (models.isEmpty()) {
                    failed(
                            tr("No PPOCR pipeline model found. Expected "
                               "capabilities.ppocr_pipeline = true."));
                    return;
                }
                bool selected = false;
                for (const auto& value : models)
                    selected |=
                            value.toObject().value("model_id") == m_remoteModel;
                if (!selected)
                    m_remoteModel = models.first()
                                            .toObject()
                                            .value("model_id")
                                            .toString();
                m_remoteBase = base;
                if (m_remotePath.isEmpty()) {
                    m_busy = false;
                    m_probing = false;
                    ++m_generation;
                    emit serviceProbed(models, m_remoteModel, base, {});
                    return;
                }
                const auto generation = m_generation;
                emit serviceProbed(models, m_remoteModel, base, {});
                if (generation != m_generation || !m_busy) return;
                if (QFileInfo(m_remotePath)
                            .suffix()
                            .compare("pdf", Qt::CaseInsensitive) == 0) {
                    // Use the workspace's existing page cache; never send PDF
                    // bytes as PNG.
                    const QDir pages(
                            QFileInfo(m_remotePath).absolutePath() + "/__PDF_" +
                            QFileInfo(m_remotePath).completeBaseName());
                    for (const auto& name : pages.entryList(
                                 {"page_*.png"}, QDir::Files, QDir::Name))
                        m_remotePages.append(pages.absoluteFilePath(name));
                    if (m_remotePages.isEmpty()) {
                        finish({},
                               tr("PDF page images are unavailable. Render the "
                                  "PDF pages before remote inference."));
                        return;
                    }
                } else
                    m_remotePages.append(m_remotePath);
                predictRemote();
            },
            qMin(10000, qMax(1, m_remoteSettings.timeoutMs)), failed);
    emit progress(tr("Checking PPOCR service…"));
}
void PpocrServiceClient::predictRemote() {
    if (!m_busy) return;
    const int pageIndex = m_remoteResults.size();
    if (pageIndex == m_remotePages.size()) {
        const auto name = QFileInfo(m_remotePath).fileName();
        finish({{"layoutParsingResults", m_remoteResults},
                {"preprocessedImages", QJsonArray()},
                {"dataInfo",
                 QJsonObject{{"type", QFileInfo(name).suffix().compare(
                                              "pdf", Qt::CaseInsensitive) == 0
                                              ? "pdf"
                                              : "image"},
                             {"numPages", m_remoteResults.size()},
                             {"pages", m_remoteSizes}}},
                {"_ppocr_meta",
                 QJsonObject{
                         {"status", "parsed"},
                         {"source_path", QStringLiteral("files/%1").arg(name)},
                         {"updated_at", ""},
                         {"error_message", ""},
                         {"edited_blocks", QJsonArray()},
                         {"block_image_paths", m_remoteImages},
                         {"pipeline_model", m_remoteModel}}}},
               {});
        return;
    }
    QImageReader reader(m_remotePages[pageIndex]);
    reader.setAutoTransform(true);
    const auto source = reader.read();
    if (source.isNull()) {
        finish({}, reader.errorString());
        return;
    }
    // One decoded page plus PNG/base64/request buffers at a time; no whole-PDF
    // image list.
    QImage image(source.size(), QImage::Format_RGB32);
    image.fill(Qt::white);
    {
        QPainter painter(&image);
        painter.drawImage(0, 0, source);
    }
    QByteArray png;
    QBuffer buffer(&png);
    buffer.open(QIODevice::WriteOnly);
    if (!image.save(&buffer, "PNG")) {
        finish({}, tr("Could not encode OCR input PNG"));
        return;
    }
    const QJsonObject body{
            {"model", m_remoteModel},
            {"image", QStringLiteral("data:image/png;base64,%1")
                              .arg(QString::fromLatin1(png.toBase64()))},
            {"params",
             QJsonObject{
                     {"page_no", pageIndex + 1},
                     {"source_file_type",
                      QFileInfo(m_remotePath)
                                              .suffix()
                                              .compare("pdf",
                                                       Qt::CaseInsensitive) == 0
                              ? "pdf"
                              : "image"}}}};
    watch(
            m_network.post(serverRequest(m_remoteBase + "/v1/predict",
                                         m_remoteSettings.apiKey),
                           QJsonDocument(body).toJson(QJsonDocument::Compact)),
            [this, image, pageIndex](const QByteArray& bytes) {
                QJsonParseError parseError;
                const auto json = QJsonDocument::fromJson(bytes, &parseError);
                if (parseError.error != QJsonParseError::NoError ||
                    !json.isObject()) {
                    finish({}, tr("Invalid remote inference response"));
                    return;
                }
                const auto object = json.object();
                if (object.contains("success") &&
                    !truth(object.value("success"))) {
                    const auto message = first(object.value("error").toObject(),
                                               {"message"});
                    finish({}, message.isEmpty() ? tr("Remote inference failed")
                                                 : message);
                    return;
                }
                const auto data = object.value("data").toObject();
                const QDir root(QFileInfo(m_remotePath).absolutePath() + "/..");
                QString error;
                auto page = normalizedPage(
                        data, image,
                        root.relativeFilePath(m_remotePages[pageIndex]),
                        m_remoteModel, &error);
                if (!error.isEmpty()) {
                    finish({}, error);
                    return;
                }
                // Build keys through the same duplicate-ID contract used by
                // rendering/editing.
                auto markdown = page.value("markdown").toObject();
                auto images = markdown.value("images").toObject();
                const auto blocks = PpocrDocument::blocks(
                        {{"layoutParsingResults", QJsonArray{page}}});
                auto serverImages = data.value("block_image_paths").toObject();
                const auto metaImages = data.value("_ppocr_meta")
                                                .toObject()
                                                .value("block_image_paths")
                                                .toObject();
                for (auto it = metaImages.begin(); it != metaImages.end(); ++it)
                    serverImages.insert(it.key(), it.value());
                const auto rawBlocks = page.value("prunedResult")
                                               .toObject()
                                               .value("parsing_res_list")
                                               .toArray();
                for (const auto& block : blocks) {
                    if (!QStringList{"image", "header_image", "footer_image"}
                                 .contains(block.label))
                        continue;
                    const QString key = QStringLiteral("page_%1:%2")
                                                .arg(pageIndex + 1)
                                                .arg(block.key.section(':', 1));
                    const auto baseKey = key.section("__dup_", 0, 0);
                    auto asset = serverImages.value(key).toString();
                    if (asset.isEmpty())
                        asset = serverImages.value(baseKey).toString();
                    const auto canonical =
                            QFileInfo(root.filePath(asset)).canonicalFilePath();
                    if (!asset.isEmpty() && QFileInfo(canonical).isFile() &&
                        canonical.startsWith(root.canonicalPath() + '/')) {
                        images.insert(key, root.relativeFilePath(canonical));
                        continue;
                    }
                    const auto rectangle =
                            bbox(rawBlocks[block.index].toObject().value(
                                         "block_bbox"),
                                 image.width(), image.height());
                    if (rectangle.size() != 4) continue;
                    const QRect crop(
                            rectangle[0].toInt(), rectangle[1].toInt(),
                            rectangle[2].toInt() - rectangle[0].toInt(),
                            rectangle[3].toInt() - rectangle[1].toInt());
                    if (crop.isEmpty()) continue;
                    const QString relative =
                            QStringLiteral(
                                    "files/__BLOCK_IMAGES_%1/"
                                    "page_%2_block_%3.png")
                                    .arg(QFileInfo(m_remotePath).fileName())
                                    .arg(pageIndex + 1, 3, 10, QChar('0'))
                                    .arg(block.index + 1, 4, 10, QChar('0'));
                    if (!QDir().mkpath(QFileInfo(root.filePath(relative))
                                               .absolutePath())) {
                        finish({},
                               tr("Could not create block image directory"));
                        return;
                    }
                    QSaveFile file(root.filePath(relative));
                    if (!file.open(QIODevice::WriteOnly) ||
                        !image.copy(crop).save(&file, "PNG") ||
                        !file.commit()) {
                        finish({}, tr("Could not save OCR block image"));
                        return;
                    }
                    images.insert(key, relative);
                }
                markdown.insert("images", images);
                page.insert("markdown", markdown);
                for (auto it = images.begin(); it != images.end(); ++it)
                    m_remoteImages.insert(it.key(), it.value());
                const auto pruned = page.value("prunedResult").toObject();
                m_remoteSizes.append(
                        QJsonObject{{"width", pruned.value("width")},
                                    {"height", pruned.value("height")}});
                m_remoteResults.append(page);
                predictRemote();
            },
            qMax(1, m_remoteSettings.timeoutMs));
    emit progress(tr("Parsing page %1 of %2…")
                          .arg(pageIndex + 1)
                          .arg(m_remotePages.size()));
}

void PpocrServiceClient::cancel() { finish({}, tr("Parsing cancelled")); }
