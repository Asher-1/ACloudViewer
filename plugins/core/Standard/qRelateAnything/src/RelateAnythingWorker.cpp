// SPDX-License-Identifier: MIT
#include "RelateAnythingWorker.h"

#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QPainter>
#include <QThread>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ecvAICoreRuntimeHelpers.h"
#include "ecvAssetIntegrity.h"
namespace {
void check(bool condition, const QString& error) {
    if (!condition) throw std::runtime_error(error.toStdString());
}
void verify(const QString& path,
            const ecvAssetIntegrity::Anchor& anchor,
            qint64 size = 0,
            bool gguf = true) {
    check(ecvAssetIntegrity::isVerified(path, anchor, 65536, gguf,
                                        ecvAssetIntegrity::OnMiss::DeepVerify,
                                        size),
          QObject::tr("Asset verification failed: %1").arg(path));
}
}  // namespace
bool RelateAnythingSettings::sameSession(
        const RelateAnythingSettings& s) const {
    return model == s.model && detector == s.detector && device == s.device &&
           threads == s.threads && maxBoxes == s.maxBoxes && topk == s.topk &&
           threshold == s.threshold && strict == s.strict &&
           decompose == s.decompose && predicates == s.predicates &&
           student == s.student && tokenizer == s.tokenizer;
}
RelateAnythingWorker::RelateAnythingWorker()
    : m_cancel(aicore_cancel_token_new()) {}
RelateAnythingWorker::~RelateAnythingWorker() {
    release();
    aicore_cancel_token_free(m_cancel);
}
void RelateAnythingWorker::release() {
    aicore_relation_free(m_context);
    m_context = nullptr;
    aicore_yolo_free(m_detector);
    m_detector = nullptr;
}
void RelateAnythingWorker::requestCancel() {
    aicore_cancel_token_request(m_cancel);
}
void RelateAnythingWorker::resetCancel() {
    aicore_cancel_token_reset(m_cancel);
}
void RelateAnythingWorker::process(const RelateAnythingSettings& settings,
                                   QImage image,
                                   const QString& source,
                                   quint64 generation) {
    Q_ASSERT(QThread::currentThread() == thread());
    RelateAnythingResult out;
    out.source = source;
    out.generation = generation;
    QElapsedTimer wall;
    wall.start();
    aicore::runtime::CancelScope scope(m_cancel);
    const auto device = settings.device.toUtf8();
    const bool locked = aicore_device_task_lock_cancelable(device.constData(),
                                                           m_cancel) == 0;
    const double waitMs = wall.nsecsElapsed() / 1e6;
    struct Unlock {
        bool locked;
        ~Unlock() {
            if (locked) aicore_device_task_unlock();
        }
    } unlock{locked};
    try {
        check(locked && !aicore_cancel_requested(), tr("Cancelled."));
        const bool reused =
                m_context && m_detector && m_settings.sameSession(settings);
        QElapsedTimer timer;
        timer.start();
        if (!reused) {
            release();
            for (int i = 0; i < aicore_relation_model_count(); ++i) {
                const auto* e = aicore_relation_model_at(i);
                for (const auto& path : {settings.model,
                                         settings.predicates.trimmed().isEmpty()
                                                 ? QString()
                                                 : settings.student,
                                         settings.predicates.trimmed().isEmpty()
                                                 ? QString()
                                                 : settings.tokenizer}) {
                    if (!path.isEmpty() &&
                        QFileInfo(path).fileName() ==
                                QString::fromUtf8(e->filename))
                        verify(path,
                               {QCryptographicHash::Sha256,
                                QByteArray(e->sha256)},
                               e->size_bytes,
                               e->role != AICORE_RELATION_MODEL_TOKENIZER);
                }
            }
            verify(settings.detector,
                   {QCryptographicHash::Sha256,
                    ecvAssetIntegrity::PinnedDigest(
                            QFileInfo(settings.detector).fileName())});
            std::unique_ptr<aicore_yolo_options,
                            decltype(&aicore_yolo_options_free)>
                    detectorOpts(aicore_yolo_options_new(),
                                 aicore_yolo_options_free);
            check(bool(detectorOpts),
                  tr("Could not allocate detection options."));
            aicore_yolo_options_set_device(detectorOpts.get(),
                                           device.constData());
            aicore_yolo_options_set_threads(detectorOpts.get(),
                                            settings.threads);
            aicore_yolo_options_set_conf_thres(detectorOpts.get(),
                                               settings.threshold);
            aicore_yolo_options_set_top_k(detectorOpts.get(),
                                          settings.maxBoxes);
            auto detectorPath = settings.detector.toUtf8();
            m_detector = aicore_yolo_load_opts(detectorPath.constData(),
                                               detectorOpts.get());
            check(aicore_yolo_is_ready(m_detector),
                  QString::fromUtf8(aicore_yolo_last_error(m_detector)));
            auto model = settings.model.toUtf8(),
                 student = settings.student.toUtf8(),
                 tokenizer = settings.tokenizer.toUtf8();
            QList<QByteArray> names;
            for (const auto& name :
                 settings.predicates.split('|', Qt::SkipEmptyParts))
                if (!name.trimmed().isEmpty()) names << name.trimmed().toUtf8();
            std::vector<const char*> pointers;
            for (const auto& name : names) pointers.push_back(name.constData());
            auto o = aicore_relation_default_options();
            o.model_path = model.constData();
            o.device = device.constData();
            o.threads = settings.threads;
            o.max_boxes = settings.maxBoxes;
            o.topk = settings.topk;
            o.strict_f32 = settings.strict;
            o.decompose = settings.decompose;
            if (!pointers.empty()) {
                o.text_student_path = student.constData();
                o.tokenizer_path = tokenizer.constData();
                o.predicate_texts = pointers.data();
                o.predicate_text_count = pointers.size();
            }
            m_context = aicore_relation_create(&o);
            check(aicore_relation_ready(m_context),
                  QString::fromUtf8(aicore_relation_last_error(m_context)));
            m_settings = settings;
        }
        const double initMs = reused ? 0 : timer.nsecsElapsed() / 1e6;
        check(!aicore_cancel_requested(), tr("Cancelled."));
        timer.restart();
        if (image.isNull()) image.load(source);
        auto view = ecvAICoreRuntime::makeImageView(image);
        if (!view.data && !image.isNull()) {
            image = image.convertToFormat(QImage::Format_RGB888);
            view = ecvAICoreRuntime::makeImageView(image);
        }
        check(view.data, tr("Cannot decode image: %1").arg(source));
        const double prepareMs = timer.nsecsElapsed() / 1e6;
        timer.restart();
        check(aicore_yolo_detect_image(m_detector, &view) == 0,
              QString::fromUtf8(aicore_yolo_last_error(m_detector)));
        const double detectMs = timer.nsecsElapsed() / 1e6;
        std::vector<float> boxes, scores;
        QStringList labels;
        QJsonArray objects;
        const auto count = std::min<int>(
                aicore_yolo_detection_count(m_detector), settings.maxBoxes);
        for (int i = 0; i < count; ++i) {
            const auto d = aicore_yolo_detection_at(m_detector, i);
            boxes.insert(boxes.end(), {d.x1, d.y1, d.x2, d.y2});
            scores.push_back(d.score);
            const auto label = QString::fromUtf8(
                    aicore_yolo_detection_class_name(m_detector, i));
            labels << label;
            objects.append(
                    QJsonObject{{"index", i},
                                {"name", label},
                                {"score", d.score},
                                {"box", QJsonArray{d.x1, d.y1, d.x2, d.y2}}});
        }
        QJsonArray relations, spatial, semantic;
        aicore_pipeline_timings timings{};
        double forwardMs = 0;
        if (count >= 2) {
            auto request = aicore_relation_default_request();
            request.boxes_xyxy = boxes.data();
            request.object_count = count;
            request.box_scores = scores.data();
            aicore_relation_result* raw = nullptr;
            const int rc = aicore_relation_infer_image(m_context, &view,
                                                       &request, &raw);
            std::unique_ptr<aicore_relation_result,
                            decltype(&aicore_relation_result_free)>
                    result(raw, aicore_relation_result_free);
            check(rc == 0,
                  QString::fromUtf8(aicore_relation_last_error(m_context)));
            auto convert = [&](const aicore_relation_edge* edges, size_t n) {
                QJsonArray array;
                for (size_t i = 0; i < n; ++i) {
                    const auto e = edges[i];
                    check(e.subject < size_t(labels.size()) &&
                                  e.object < size_t(labels.size()) &&
                                  e.predicate < raw->predicate_count,
                          tr("Invalid relation indices."));
                    array.append(QJsonObject{
                            {"subject", int(e.subject)},
                            {"object", int(e.object)},
                            {"predicate_index", int(e.predicate)},
                            {"subject_name",
                             QString(labels[int(e.subject)] + " #" +
                                     QString::number(e.subject))},
                            {"object_name",
                             QString(labels[int(e.object)] + " #" +
                                     QString::number(e.object))},
                            {"predicate",
                             QString::fromUtf8(
                                     raw->predicate_names[e.predicate])},
                            {"score", e.score}});
                }
                return array;
            };
            relations = convert(raw->relations, raw->relation_count);
            spatial = convert(raw->spatial_relations, raw->spatial_count);
            semantic = convert(raw->semantic_relations, raw->semantic_count);
            timings = raw->timings;
            forwardMs = raw->forward_ms;
        }
        check(!aicore_cancel_requested(), tr("Cancelled."));
        out.image = image;
        out.device =
                QString::fromUtf8(aicore_relation_context_device(m_context));
        out.annotated = image.convertToFormat(QImage::Format_RGB32);
        timer.restart();
        QPainter painter(&out.annotated);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(QColor(70, 200, 130), 2));
        for (int i = 0; i < count; ++i) {
            const float* b = boxes.data() + i * 4;
            painter.drawRect(QRectF(QPointF(b[0], b[1]), QPointF(b[2], b[3])));
            painter.drawText(QPointF(b[0], std::max(12.0f, b[1] - 3)),
                             labels[i] + " #" + QString::number(i));
        }
        painter.setPen(QPen(QColor(255, 170, 50), 2));
        for (int i = 0; i < std::min(5, relations.size()); ++i) {
            const auto e = relations[i].toObject();
            const float* s = boxes.data() + e["subject"].toInt() * 4;
            const float* o = boxes.data() + e["object"].toInt() * 4;
            const QPointF a((s[0] + s[2]) * .5, (s[1] + s[3]) * .5),
                    b((o[0] + o[2]) * .5, (o[1] + o[3]) * .5);
            painter.drawLine(a, b);
            const double angle = std::atan2(b.y() - a.y(), b.x() - a.x());
            for (double delta : {-.4, .4})
                painter.drawLine(b, b - QPointF(10 * std::cos(angle + delta),
                                                10 * std::sin(angle + delta)));
            painter.drawText((a + b) * .5, e["predicate"].toString());
        }
        painter.end();
        const double renderMs = timer.nsecsElapsed() / 1e6;
        out.e2eMs = wall.nsecsElapsed() / 1e6;
        out.json = {{"source", source},
                    {"model", QFileInfo(settings.model).fileName()},
                    {"detector", QFileInfo(settings.detector).fileName()},
                    {"device", out.device},
                    {"objects", objects},
                    {"relations", relations},
                    {"spatial", spatial},
                    {"semantic", semantic},
                    {"timings",
                     QJsonObject{{"valid_fields", int(timings.valid_fields)},
                                 {"preprocess_ms", timings.preprocess_ms},
                                 {"relation_e2e_ms", timings.e2e_ms},
                                 {"relation_forward_ms", forwardMs},
                                 {"detection_ms", detectMs},
                                 {"device_wait_ms", waitMs},
                                 {"context_init_ms", initMs},
                                 {"context_reused", reused},
                                 {"image_prepare_ms", prepareMs},
                                 {"visualization_ms", renderMs},
                                 {"worker_wall_ms", out.e2eMs}}}};
    } catch (const std::exception& e) {
        out.error = QString::fromUtf8(e.what());
        if (!aicore_relation_ready(m_context) ||
            !aicore_yolo_is_ready(m_detector))
            release();
    } catch (...) {
        out.error = tr("Unexpected relation processing error.");
        release();
    }
    emit completed(out);
}
