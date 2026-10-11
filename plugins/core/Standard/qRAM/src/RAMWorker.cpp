// SPDX-License-Identifier: MIT
#include "RAMWorker.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QThread>
#include <memory>

#include "ecvAICoreRuntimeHelpers.h"
#include "ecvAssetIntegrity.h"

bool RAMSettings::sameSession(const RAMSettings& b) const {
    return model == b.model && device == b.device && threads == b.threads &&
           caption == b.caption && maxLength == b.maxLength &&
           beamBatch == b.beamBatch && tagList == b.tagList &&
           thresholds == b.thresholds && embedding == b.embedding &&
           cpuPrecision == b.cpuPrecision;
}
RAMWorker::RAMWorker() : m_cancel(aicore_cancel_token_new()) {}
RAMWorker::~RAMWorker() {
    release();
    aicore_cancel_token_free(m_cancel);
}
void RAMWorker::requestCancel() { aicore_cancel_token_request(m_cancel); }
void RAMWorker::resetCancel() { aicore_cancel_token_reset(m_cancel); }
void RAMWorker::release() {
    aicore_tagging_free(m_context);
    m_context = nullptr;
}
void RAMWorker::process(const RAMSettings& settings,
                        QImage image,
                        const QString& source,
                        quint64 generation) {
    Q_ASSERT(QThread::currentThread() == thread());
    QElapsedTimer elapsed;
    elapsed.start();
    double deviceWaitMs = 0;
    double contextInitMs = 0;
    double imagePrepareMs = 0;
    bool contextReused = false;
    RAMResult out;
    out.generation = generation;
    out.source = source;
    aicore::runtime::CancelScope scope(m_cancel);
    auto device = settings.device.toUtf8();
    const bool locked = aicore_device_task_lock_cancelable(device.constData(),
                                                           m_cancel) == 0;
    deviceWaitMs = elapsed.nsecsElapsed() / 1e6;
    struct Unlock {
        bool locked;
        ~Unlock() {
            if (locked) aicore_device_task_unlock();
        }
    } unlock{locked};
    if (!locked)
        out.error = tr("Cancelled while waiting for the device.");
    else if (aicore_cancel_requested())
        out.error = tr("Cancelled.");
    else {
        if (!m_context || !m_settings.sameSession(settings)) {
            QElapsedTimer contextTimer;
            contextTimer.start();
            release();
            const aicore_tagging_model_entry* entry = nullptr;
            for (int i = 0; i < aicore_tagging_model_count(); ++i) {
                auto* candidate = aicore_tagging_model_at(i);
                if (QFileInfo(settings.model).fileName() ==
                    QString::fromUtf8(candidate->filename))
                    entry = candidate;
            }
            if (entry &&
                !ecvAssetIntegrity::isVerified(
                        settings.model,
                        {QCryptographicHash::Sha256, QByteArray(entry->sha256)},
                        65536, true, ecvAssetIntegrity::OnMiss::DeepVerify,
                        entry->size_bytes)) {
                out.error =
                        tr("Model checksum failed. Download the model again.");
                emit completed(out);
                return;
            }
            auto model = settings.model.toUtf8(),
                 labels = settings.tagList.toUtf8();
            auto thresholds = settings.thresholds.toUtf8(),
                 embedding = settings.embedding.toUtf8();
            auto opts = aicore_tagging_default_options();
            opts.model_path = model.constData();
            opts.device = device.constData();
            opts.threads = settings.threads;
            opts.cpu_precision = settings.cpuPrecision;
            opts.generate_caption = settings.caption;
            opts.max_length = settings.maxLength;
            opts.beam_batch = settings.beamBatch;
            opts.tag_list_path =
                    labels.isEmpty() ? nullptr : labels.constData();
            opts.threshold_path =
                    thresholds.isEmpty() ? nullptr : thresholds.constData();
            opts.label_embedding_path =
                    embedding.isEmpty() ? nullptr : embedding.constData();
            m_context = aicore_tagging_create(&opts);
            m_settings = settings;
            contextInitMs = contextTimer.nsecsElapsed() / 1e6;
        } else {
            contextReused = true;
        }
        if (!aicore_tagging_ready(m_context)) {
            out.error = QString::fromUtf8(aicore_tagging_last_error(m_context));
            release();
        } else {
            QElapsedTimer imageTimer;
            imageTimer.start();
            if (image.isNull())
                image.load(source);  // decode on worker, never on the GUI's
                                     // frame path
            auto view = ecvAICoreRuntime::makeImageView(image);
            if (!view.data && !image.isNull()) {
                image = image.convertToFormat(QImage::Format_RGB888);
                view = ecvAICoreRuntime::makeImageView(image);
            }
            imagePrepareMs = imageTimer.nsecsElapsed() / 1e6;
            auto req = aicore_tagging_default_request();
            req.threshold = settings.threshold;
            req.threshold_epsilon = settings.thresholdEpsilon;
            auto specified = settings.specifiedTags.toUtf8();
            req.specified_tags = specified.constData();
            aicore_tagging_result* raw = nullptr;
            int rc = aicore_tagging_infer_image(m_context, &view, &req, &raw);
            std::unique_ptr<aicore_tagging_result,
                            decltype(&aicore_tagging_result_free)>
                    result(raw, aicore_tagging_result_free);
            if (rc)
                out.error =
                        QString::fromUtf8(aicore_tagging_last_error(m_context));
            else {
                QJsonArray tags, selected;
                for (size_t i = 0; i < raw->tag_count; ++i) {
                    const auto& t = raw->tags[i];
                    tags.append(QJsonObject{{"index", t.index},
                                            {"name", QString::fromUtf8(t.name)},
                                            {"logit", t.logit},
                                            {"probability", t.probability},
                                            {"threshold", t.threshold},
                                            {"selected", bool(t.selected)}});
                    if (t.selected) selected.append(QString::fromUtf8(t.name));
                }
                out.image = image;
                out.device = QString::fromUtf8(
                        aicore_tagging_context_device(m_context));
                out.json = {
                        {"source", source},
                        {"model", QFileInfo(settings.model).fileName()},
                        {"device", out.device},
                        {"cpu_precision",
                         settings.cpuPrecision == AICORE_TAGGING_CPU_AUTO
                                 ? "auto"
                                 : "upstream"},
                        {"caption", QString::fromUtf8(raw->caption)},
                        {"selected_tags", selected},
                        {"tags", tags},
                        {"timings",
                         QJsonObject{
                                 {"valid_fields",
                                  int(raw->timings.valid_fields)},
                                 {"preprocess_ms", raw->timings.preprocess_ms},
                                 {"inference_ms", raw->timings.inference_ms},
                                 {"postprocess_ms",
                                  raw->timings.postprocess_ms},
                                 {"e2e_ms", raw->timings.e2e_ms},
                                 {"device_wait_ms", deviceWaitMs},
                                 {"context_init_ms", contextInitMs},
                                 {"image_prepare_ms", imagePrepareMs},
                                 {"context_reused", contextReused}}}};
            }
        }
    }
    if (out.error.isEmpty()) {
        out.e2eMs = elapsed.nsecsElapsed() / 1e6;
        out.json.insert("processing_ms", out.e2eMs);
        auto timings = out.json.value("timings").toObject();
        timings.insert("worker_wall_ms", out.e2eMs);
        out.json.insert("timings", timings);
    }
    emit completed(out);  // exactly one completion, including invalid
                          // inputs/cancellation
}
