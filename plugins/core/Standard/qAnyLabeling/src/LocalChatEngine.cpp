// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LocalChatEngine.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QImage>
#include <QJsonObject>
#include <QMetaObject>
#include <QObject>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "aicore/llm_capi.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvAssetIntegrity.h"

namespace LocalChatEngine {
namespace {

constexpr qint64 kMinBytes = 1024 * 1024;

void stopEngine();

QString cacheDir() {
    char* dir = aicore_llm_model_cache_dir();
    const QString path = dir ? QString::fromUtf8(dir) : QString();
    aicore_llm_free_buffer(dir);
    return path;
}

bool verified(const QString& path, const aicore_llm_model_entry& entry) {
    return ecvAssetIntegrity::isVerified(
            path, {QCryptographicHash::Sha256, QByteArray(entry.sha256)},
            kMinBytes, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            entry.size_bytes);
}

const aicore_llm_model_entry* chatModel(const QString& id) {
    const QByteArray name = id.toUtf8();
    const auto* entry = aicore_llm_model_by_filename(name.constData());
    return entry && entry->role == AICORE_LLM_ROLE_MODEL ? entry : nullptr;
}

struct Job {
    quint64 id = 0;
    QString model, mmproj;
    QJsonArray messages;
    double temperature = 0;
    int maxTokens = 0;
    QObject* receiver = nullptr;  // guarded by Engine::mutex
    Callbacks callbacks;
    std::atomic<bool> alive{true};  // false once cancelled by the client
};

// OpenAI chat messages -> llm messages. Content is a string or a list of
// {type:text} / {type:image_url, image_url:{url:"data:...;base64,..."}}
// parts; each image becomes the task's media marker at its position.
bool convert(const QJsonArray& input,
             std::vector<QByteArray>* roles,
             std::vector<QByteArray>* texts,
             std::vector<QImage>* images,
             QString* error) {
    const QString marker = QString::fromUtf8(aicore_llm_media_marker());
    for (const auto& value : input) {
        const auto message = value.toObject();
        QString text;
        const auto content = message.value("content");
        if (content.isString()) {
            text = content.toString();
        } else {
            for (const auto& part : content.toArray()) {
                const auto object = part.toObject();
                const auto type = object.value("type").toString();
                if (type == "text") {
                    text += object.value("text").toString();
                } else if (type == "image_url") {
                    const auto url = object.value("image_url")
                                             .toObject()
                                             .value("url")
                                             .toString();
                    const int comma = url.indexOf(',');
                    if (!url.startsWith("data:") || comma < 0 ||
                        !url.left(comma).endsWith(";base64")) {
                        *error = QObject::tr(
                                "The local model accepts embedded images "
                                "only.");
                        return false;
                    }
                    QImage image = QImage::fromData(QByteArray::fromBase64(
                            url.mid(comma + 1).toLatin1()));
                    if (image.isNull()) {
                        *error = QObject::tr("Unable to decode image.");
                        return false;
                    }
                    images->push_back(
                            image.convertToFormat(QImage::Format_RGB888));
                    text += marker;
                }
            }
        }
        roles->push_back(message.value("role").toString().toUtf8());
        texts->push_back(text.toUtf8());
    }
    return true;
}

class Engine {
public:
    ~Engine() {
        // Plugin unload before the application: drop the post routine that
        // would otherwise point into the unloaded library.
        if (QCoreApplication::instance()) qRemovePostRoutine(stopEngine);
        stop();
    }

    // Idempotent. Runs from the QCoreApplication destructor (post routine)
    // so the worker frees its context while the GPU runtimes are alive;
    // static destruction would run after their exit handlers.
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            queue_.clear();
            if (running_) {
                running_->alive = false;
                running_->receiver = nullptr;
            }
            if (ctx_) aicore_llm_cancel(ctx_);
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    quint64 submit(std::shared_ptr<Job> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        job->id = ++next_;
        if (stop_) return job->id;  // application is shutting down
        queue_.push_back(job);
        if (!thread_.joinable()) {
            thread_ = std::thread([this]() { loop(); });
            if (QCoreApplication::instance()) qAddPostRoutine(stopEngine);
        }
        cv_.notify_one();
        return job->id;
    }

    void cancel(quint64 id) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = queue_.begin(); it != queue_.end(); ++it)
            if ((*it)->id == id) {
                (*it)->alive = false;
                queue_.erase(it);
                return;
            }
        if (running_ && running_->id == id) {
            running_->alive = false;
            running_->receiver = nullptr;
            if (ctx_) aicore_llm_cancel(ctx_);
        }
    }

    // Posts fn to the job's receiver unless the client cancelled. The
    // mutex keeps the receiver alive while posting (cancel, called from its
    // destructor, waits); `alive` drops events queued before cancel.
    template <typename Fn>
    void post(const std::shared_ptr<Job>& job, Fn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!job->receiver) return;
        QMetaObject::invokeMethod(
                job->receiver,
                [job, fn]() {
                    if (job->alive) fn();
                },
                Qt::QueuedConnection);
    }

private:
    void loop() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_) break;
                job = queue_.front();
                queue_.pop_front();
                running_ = job;
            }
            QString text, error;
            run(job, &text, &error);
            if (job->callbacks.done)
                post(job, [job, text, error]() {
                    job->callbacks.done(text, error);
                });
            std::lock_guard<std::mutex> lock(mutex_);
            running_.reset();
        }
        setContext(nullptr);  // contexts die on the thread that made them
    }

    void setContext(aicore_llm_ctx* ctx) {
        aicore_llm_ctx* old = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            old = ctx_;
            ctx_ = ctx;
        }
        aicore_llm_free(old);
    }

    using StreamUser = std::pair<Engine*, std::shared_ptr<Job>>;
    static int streamed(const char* chunk, size_t bytes, void* user) {
        auto* self = static_cast<StreamUser*>(user);
        Job* job = self->second.get();
        if (!job->alive) return 1;  // client cancel stops at the next token
        if (job->callbacks.delta) {
            const QString delta = QString::fromUtf8(chunk, int(bytes));
            self->first->post(self->second,
                              [job, delta]() { job->callbacks.delta(delta); });
        }
        return 0;
    }

    void run(const std::shared_ptr<Job>& job, QString* text, QString* error) {
        std::vector<QByteArray> roles, texts;
        std::vector<QImage> images;
        if (!convert(job->messages, &roles, &texts, &images, error)) return;
        // One device queue for model load and generation: a backend is not
        // safe under concurrent compute from two AICore workers.
        auto deviceLock = ecvAICoreRuntime::makeDeviceTaskLock("auto");
        if (!deviceLock.isLocked()) {
            *error = QObject::tr("The inference device is busy.");
            return;
        }
        if (!ctx_ || model_ != job->model || mmproj_ != job->mmproj) {
            setContext(nullptr);
            const QByteArray model = job->model.toUtf8(),
                             mmproj = job->mmproj.toUtf8();
            auto options = aicore_llm_default_options();
            options.model_path = model.constData();
            options.mmproj_path =
                    mmproj.isEmpty() ? nullptr : mmproj.constData();
            options.device = "auto";
            aicore_llm_ctx* ctx = aicore_llm_create(&options);
            if (!aicore_llm_ready(ctx)) {
                *error = QString::fromUtf8(aicore_llm_last_error(ctx));
                aicore_llm_free(ctx);
                return;
            }
            setContext(ctx);
            model_ = job->model;
            mmproj_ = job->mmproj;
        }
        if (!job->alive) return;
        if (!images.empty() && !aicore_llm_supports_vision(ctx_)) {
            *error = QObject::tr("This local model does not accept images.");
            return;
        }
        std::vector<aicore_llm_message> messages;
        for (size_t i = 0; i < roles.size(); ++i)
            messages.push_back({roles[i].constData(), texts[i].constData()});
        std::vector<aicore_image_view> views;
        for (const auto& image : images)
            views.push_back(ecvAICoreRuntime::makeImageView(image));
        StreamUser user{this, job};
        auto request = aicore_llm_default_request();
        request.messages = messages.data();
        request.message_count = messages.size();
        request.images = views.empty() ? nullptr : views.data();
        request.image_count = views.size();
        if (job->maxTokens > 0) request.max_tokens = uint32_t(job->maxTokens);
        request.temperature = float(job->temperature);
        request.stream_cb = streamed;
        request.stream_user_data = &user;
        aicore_llm_result* result = nullptr;
        const int rc = aicore_llm_generate(ctx_, &request, &result);
        if (rc < 0)
            *error = QString::fromUtf8(aicore_llm_last_error(ctx_));
        else if (result)
            *text = QString::fromUtf8(result->text, int(result->text_bytes));
        aicore_llm_result_free(result);
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::shared_ptr<Job> running_;
    std::thread thread_;
    bool stop_ = false;
    quint64 next_ = 0;
    aicore_llm_ctx* ctx_ = nullptr;  // written by the worker under mutex_
    QString model_, mmproj_;         // worker thread only
};

Engine& engine() {
    static Engine instance;
    return instance;
}

void stopEngine() { engine().stop(); }

}  // namespace

bool isLocal(const QUrl& url) {
    return url.scheme() == QStringLiteral("aicore");
}

QJsonArray models() {
    const QDir dir(cacheDir());
    QJsonArray list;
    for (int i = 0; i < aicore_llm_model_count(); ++i) {
        const auto* entry = aicore_llm_model_at(i);
        if (!entry || entry->role != AICORE_LLM_ROLE_MODEL) continue;
        bool downloaded = verified(dir.filePath(entry->filename), *entry);
        if (entry->mmproj_filename) {
            const auto* projector =
                    aicore_llm_model_by_filename(entry->mmproj_filename);
            downloaded =
                    downloaded && projector &&
                    verified(dir.filePath(projector->filename), *projector);
        }
        list.append(
                QJsonObject{{"id", QString::fromUtf8(entry->filename)},
                            {"name", QString::fromUtf8(entry->display_name)},
                            {"vision", entry->mmproj_filename != nullptr},
                            {"downloaded", downloaded},
                            {"owned_by", "aicore"}});
    }
    return list;
}

bool missingAssets(const QString& model,
                   QVector<Asset>* missing,
                   QString* error) {
    missing->clear();
    const auto* entry = chatModel(model);
    if (!entry) {
        *error = QObject::tr("Unknown local model: %1").arg(model);
        return false;
    }
    const QDir dir(cacheDir());
    std::vector<const aicore_llm_model_entry*> files{entry};
    if (entry->mmproj_filename)
        files.push_back(aicore_llm_model_by_filename(entry->mmproj_filename));
    for (const auto* file : files) {
        if (!file) {
            *error = QObject::tr("Projector missing from the catalog.");
            return false;
        }
        const QString path = dir.filePath(file->filename);
        if (!verified(path, *file))
            missing->append({path, QString::fromUtf8(file->download_url),
                             QByteArray(file->sha256), file->size_bytes});
    }
    return true;
}

quint64 submit(const QString& model,
               const QJsonArray& messages,
               double temperature,
               int maxTokens,
               QObject* receiver,
               Callbacks callbacks) {
    auto job = std::make_shared<Job>();
    const auto* entry = chatModel(model);
    const QDir dir(cacheDir());
    job->model = entry ? dir.filePath(entry->filename) : model;
    if (entry && entry->mmproj_filename)
        job->mmproj = dir.filePath(entry->mmproj_filename);
    job->messages = messages;
    job->temperature = temperature;
    job->maxTokens = maxTokens;
    job->receiver = receiver;
    job->callbacks = std::move(callbacks);
    return engine().submit(job);
}

void cancel(quint64 id) { engine().cancel(id); }

}  // namespace LocalChatEngine
