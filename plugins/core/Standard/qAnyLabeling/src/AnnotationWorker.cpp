// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationWorker.h"

#include <QtCompat.h>

#include <QFileInfo>
#include <QJsonDocument>
#include <QSet>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "AnnotationSemantic.h"
#include "AnnotationTagging.h"
#include "MaskPolygon.h"
#include "RelateAnythingDecode.h"
#include "SlicedDetection.h"
#include "aicore/aliked_capi.h"
#include "aicore/attribute_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/image_view.h"
#include "aicore/ocr_capi.h"
#include "aicore/relation_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/tagging_capi.h"
#include "aicore/yolo_capi.h"

namespace {
AnnotationShape rectangle(const aicore_yolo_detection& box,
                          const QString& label) {
    AnnotationShape shape;
    shape.type = QStringLiteral("rectangle");
    shape.label = label;
    shape.points = {QPointF(box.x1, box.y1), QPointF(box.x2, box.y1),
                    QPointF(box.x2, box.y2), QPointF(box.x1, box.y2)};
    shape.extra.insert(QStringLiteral("score"), double(box.score));
    return shape;
}

QString className(const char* name, int classId) {
    return name ? QString::fromUtf8(name)
                : QObject::tr("class %1").arg(classId);
}

QImage instanceMask(const uint8_t* data,
                    int width,
                    int height,
                    size_t stride,
                    const QSize& imageSize) {
    if (!data || width <= 0 || height <= 0 || stride < size_t(width)) return {};
    QImage mask(width, height, QImage::Format_ARGB32_Premultiplied);
    if (mask.isNull()) return {};
    for (int y = 0; y < height; ++y) {
        QRgb* row = reinterpret_cast<QRgb*>(mask.scanLine(y));
        const uint8_t* source = data + size_t(y) * stride;
        for (int x = 0; x < width; ++x)
            row[x] = source[x] ? qRgba(0, 78, 110, 110) : 0;
    }
    return mask.size() == imageSize
                   ? mask
                   : mask.scaled(imageSize, Qt::IgnoreAspectRatio,
                                 Qt::FastTransformation);
}
}  // namespace

void AnnotationWorker::releaseRelateDetector() {
    if (!m_relateDetector) return;
    if (m_relateDetectorRfdetr)
        aicore_rfdetr_free(static_cast<aicore_rfdetr_ctx*>(m_relateDetector));
    else
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_relateDetector));
    m_relateDetector = nullptr;
    m_relateDetectorPath.clear();
    m_relateDetectorDevice.clear();
}

void AnnotationWorker::releaseObjectTagger() {
    aicore_tagging_free(static_cast<aicore_tagging_ctx*>(m_objectTagger));
    m_objectTagger = nullptr;
    m_objectTaggerPath.clear();
    m_objectTaggerDevice.clear();
}

AnnotationWorker::~AnnotationWorker() {
    releaseObjectTagger();
    releaseRelateDetector();
    if (m_landmarkContext)
        aicore_facedetect_free(
                static_cast<aicore_facedetect_ctx*>(m_landmarkContext));
    if (m_kind == ModelKind::Depth)
        aicore_depth_free(static_cast<aicore_depth_ctx*>(m_context));
    else if (m_kind == ModelKind::RFDetr)
        aicore_rfdetr_free(static_cast<aicore_rfdetr_ctx*>(m_context));
    else if (m_kind == ModelKind::SAM3)
        aicore_sam3_free(static_cast<aicore_sam3_ctx*>(m_context));
    else if (m_kind == ModelKind::RMBG)
        aicore_rmbg_free(static_cast<aicore_rmbg_ctx*>(m_context));
    else if (m_kind == ModelKind::GKD)
        aicore_gkd_free(static_cast<aicore_gkd_ctx*>(m_context));
    else if (m_kind == ModelKind::FaceDetect)
        aicore_facedetect_free(static_cast<aicore_facedetect_ctx*>(m_context));
    else if (m_kind == ModelKind::Ocr)
        aicore_ocr_context_destroy(static_cast<aicore_ocr_ctx*>(m_context));
    else if (m_kind == ModelKind::Attribute)
        aicore_attribute_free(static_cast<aicore_attribute_ctx*>(m_context));
    else if (m_kind == ModelKind::DeepLSD)
        aicore_deeplsd_free(static_cast<aicore_deeplsd_ctx*>(m_context));
    else if (m_kind == ModelKind::ALIKED)
        aicore_aliked_free(static_cast<aicore_aliked_ctx*>(m_context));
    else if (m_kind == ModelKind::Ram)
        aicore_tagging_free(static_cast<aicore_tagging_ctx*>(m_context));
    else if (m_kind == ModelKind::RelateAnything)
        aicore_relation_free(static_cast<aicore_relation_ctx*>(m_context));
    else
        aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_context));
}

void AnnotationWorker::infer(const QString& modelPath,
                             const QString& metricModelPath,
                             const QString& device,
                             const QString& task,
                             const QString& imagePath,
                             const QImage& image,
                             quint64 generation,
                             double confidence,
                             quint64 promptRevision,
                             const AnnotationPrompt& prompt) {
    AnnotationInference result;
    if (m_objectTagger &&
        (prompt.objectTaggingModelPath != m_objectTaggerPath ||
         device != m_objectTaggerDevice))
        releaseObjectTagger();
    result.generation = generation;
    result.promptRevision = promptRevision;
    result.imagePath = imagePath;
    result.modelPath = modelPath;
    result.task = task;
    if (task == QStringLiteral("gkd") || task == QStringLiteral("relate") ||
        (task == QStringLiteral("sam3-pvs") && !prompt.rois.isEmpty())) {
        result.sourceRois = prompt.rois;
        result.sourceIndices = prompt.roiSourceIndices;
    }
    const bool rfdetr = task.startsWith(QStringLiteral("rfdetr-"));
    const bool world = task == QStringLiteral("world");
    result.replaceShapes = !prompt.objectTaggingModelPath.isEmpty() &&
                           (world || task == QStringLiteral("detect"));
    const bool yoloe = task == QStringLiteral("yoloe");
    const bool visualPrompt = yoloe && !prompt.visualPrompts.isEmpty();
    QStringList vocabulary;
    if (world || yoloe) {
        for (const QString& name :
             prompt.text.split(QLatin1Char(','), QtCompat::SkipEmptyParts)) {
            const QString trimmed = name.trimmed();
            if (!trimmed.isEmpty()) vocabulary.append(trimmed);
        }
    }
    const ModelKind kind =
            task.startsWith(QStringLiteral("depth"))   ? ModelKind::Depth
            : task.startsWith(QStringLiteral("sam3-")) ? ModelKind::SAM3
            : rfdetr                                   ? ModelKind::RFDetr
            : task == QStringLiteral("rmbg")           ? ModelKind::RMBG
            : task == QStringLiteral("gkd")            ? ModelKind::GKD
            : task.startsWith(QStringLiteral("facedetect"))
                    ? ModelKind::FaceDetect
            : task == QStringLiteral("ocr")     ? ModelKind::Ocr
            : task == QStringLiteral("attribute") ? ModelKind::Attribute
            : task == QStringLiteral("deeplsd") ? ModelKind::DeepLSD
            : task == QStringLiteral("aliked")  ? ModelKind::ALIKED
            : (task == QStringLiteral("ram") ||
               task.startsWith(QStringLiteral("ram-")))
                    ? ModelKind::Ram
            : (task == QStringLiteral("relate") ||
               task.startsWith(QStringLiteral("relate-")))
                    ? ModelKind::RelateAnything
                    : ModelKind::Yolo;
    // The predicate vocabulary is encoded into the relation context at load,
    // so a different list (or text-student) needs a new context.
    const QString relateKey = kind == ModelKind::RelateAnything
                                      ? prompt.relate.predicates.trimmed() +
                                                QLatin1Char('\n') +
                                                prompt.relateStudentPath
                                      : QString();
    const bool modelChanged =
            m_modelPath != modelPath || m_metricModelPath != metricModelPath ||
            m_device != device || m_kind != kind ||
            (kind == ModelKind::Ocr &&
             (m_ocrLayoutModelPath != prompt.ocrLayoutModelPath ||
              m_ocrClassifierModelPath != prompt.ocrClassifierModelPath)) ||
            (kind == ModelKind::RelateAnything && m_vocabText != relateKey);
    const bool vocabularyChanged =
            (world || yoloe) &&
            (m_textModelPath != prompt.textModelPath ||
             m_promptCount != vocabulary.size() || m_vocabText != prompt.text ||
             (yoloe && m_visualPrompts != prompt.visualPrompts));
    if (m_context && (modelChanged || vocabularyChanged)) {
        if (m_landmarkContext) {
            aicore_facedetect_free(
                    static_cast<aicore_facedetect_ctx*>(m_landmarkContext));
            m_landmarkContext = nullptr;
        }
        if (m_kind == ModelKind::Depth)
            aicore_depth_free(static_cast<aicore_depth_ctx*>(m_context));
        else if (m_kind == ModelKind::RFDetr)
            aicore_rfdetr_free(static_cast<aicore_rfdetr_ctx*>(m_context));
        else if (m_kind == ModelKind::SAM3)
            aicore_sam3_free(static_cast<aicore_sam3_ctx*>(m_context));
        else if (m_kind == ModelKind::RMBG)
            aicore_rmbg_free(static_cast<aicore_rmbg_ctx*>(m_context));
        else if (m_kind == ModelKind::GKD)
            aicore_gkd_free(static_cast<aicore_gkd_ctx*>(m_context));
        else if (m_kind == ModelKind::FaceDetect)
            aicore_facedetect_free(
                    static_cast<aicore_facedetect_ctx*>(m_context));
        else if (m_kind == ModelKind::Ocr)
            aicore_ocr_context_destroy(static_cast<aicore_ocr_ctx*>(m_context));
        else if (m_kind == ModelKind::Attribute)
            aicore_attribute_free(static_cast<aicore_attribute_ctx*>(m_context));
        else if (m_kind == ModelKind::DeepLSD)
            aicore_deeplsd_free(static_cast<aicore_deeplsd_ctx*>(m_context));
        else if (m_kind == ModelKind::ALIKED)
            aicore_aliked_free(static_cast<aicore_aliked_ctx*>(m_context));
        else if (m_kind == ModelKind::Ram)
            aicore_tagging_free(static_cast<aicore_tagging_ctx*>(m_context));
        else if (m_kind == ModelKind::RelateAnything)
            aicore_relation_free(static_cast<aicore_relation_ctx*>(m_context));
        else
            aicore_yolo_free(static_cast<aicore_yolo_ctx*>(m_context));
        m_context = nullptr;
        m_metricModelPath.clear();
    }
    if (!m_context) {
        const QByteArray deviceBytes = device.toUtf8();
        const QByteArray pathBytes =
                QFileInfo(modelPath).absoluteFilePath().toUtf8();
        if (kind == ModelKind::Depth) {
            aicore_depth_options* options = aicore_depth_options_new();
            if (!options) {
                result.error = tr("Unable to create depth options");
                emit finished(result);
                return;
            }
            aicore_depth_options_set_device(options, deviceBytes.constData());
            if (metricModelPath.isEmpty()) {
                m_context =
                        aicore_depth_load_opts(pathBytes.constData(), options);
            } else {
                const QByteArray metricBytes =
                        QFileInfo(metricModelPath).absoluteFilePath().toUtf8();
                m_context = aicore_depth_load_nested_opts(
                        pathBytes.constData(), metricBytes.constData(),
                        options);
            }
            aicore_depth_options_free(options);
            if (!m_context) {
                result.error = tr("Unable to load depth model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_metricModelPath = metricModelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::SAM3) {
            aicore_sam3_options* options = aicore_sam3_options_new();
            if (!options) {
                result.error = tr("Unable to create SAM options");
                emit finished(result);
                return;
            }
            aicore_sam3_options_set_device(options, deviceBytes.constData());
            m_context = aicore_sam3_load_opts(pathBytes.constData(), options);
            aicore_sam3_options_free(options);
            if (!m_context) {
                result.error = QString::fromUtf8(aicore_sam3_last_load_error());
                if (result.error.isEmpty())
                    result.error = tr("Unable to load SAM model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (rfdetr) {
            aicore_rfdetr_options* options = aicore_rfdetr_options_new();
            if (!options) {
                result.error = tr("Unable to create RF-DETR options");
                emit finished(result);
                return;
            }
            aicore_rfdetr_options_set_device(options, deviceBytes.constData());
            m_context = aicore_rfdetr_load_opts(pathBytes.constData(), options);
            aicore_rfdetr_options_free(options);
            if (!m_context) {
                result.error =
                        QString::fromUtf8(aicore_rfdetr_last_error(nullptr));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load RF-DETR model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::RMBG) {
            aicore_rmbg_options* options = aicore_rmbg_options_new();
            if (!options) {
                result.error = tr("Unable to create RMBG options");
                emit finished(result);
                return;
            }
            aicore_rmbg_options_set_device(options, deviceBytes.constData());
            m_context = aicore_rmbg_load_opts(pathBytes.constData(), options);
            aicore_rmbg_options_free(options);
            if (!m_context) {
                result.error =
                        QString::fromUtf8(aicore_rmbg_last_error(nullptr));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load RMBG model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::GKD) {
            aicore_gkd_options* options = aicore_gkd_options_new();
            if (!options) {
                result.error = tr("Unable to create GKD options");
                emit finished(result);
                return;
            }
            aicore_gkd_options_set_device(options, deviceBytes.constData());
            m_context = aicore_gkd_load_opts(pathBytes.constData(), options);
            aicore_gkd_options_free(options);
            if (!m_context) {
                result.error =
                        QString::fromUtf8(aicore_gkd_last_error(nullptr));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load GKD model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::FaceDetect) {
            aicore_facedetect_options* options =
                    aicore_facedetect_options_new();
            if (!options) {
                result.error = tr("Unable to create face detector options");
                emit finished(result);
                return;
            }
            aicore_facedetect_options_set_device(options,
                                                 deviceBytes.constData());
            m_context =
                    aicore_facedetect_load_opts(pathBytes.constData(), options);
            if (m_context && task == QStringLiteral("facedetect-dense")) {
                if (metricModelPath.isEmpty()) {
                    result.error =
                            tr("Dense face landmarks require a landmark model");
                } else {
                    const QByteArray landmarkPath = QFileInfo(metricModelPath)
                                                            .absoluteFilePath()
                                                            .toUtf8();
                    m_landmarkContext = aicore_facedetect_load_opts(
                            landmarkPath.constData(), options);
                }
                if (!m_landmarkContext) {
                    if (result.error.isEmpty())
                        result.error = QString::fromUtf8(
                                aicore_facedetect_last_error(nullptr));
                    aicore_facedetect_free(
                            static_cast<aicore_facedetect_ctx*>(m_context));
                    m_context = nullptr;
                }
            }
            aicore_facedetect_options_free(options);
            if (!m_context) {
                if (result.error.isEmpty())
                    result.error = QString::fromUtf8(
                            aicore_facedetect_last_error(nullptr));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load face detector");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_metricModelPath = metricModelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::Attribute) {
            auto options = aicore_attribute_default_options();
            options.model_path = pathBytes.constData();
            options.device = deviceBytes.constData();
            auto* context = aicore_attribute_create(&options);
            if (!aicore_attribute_ready(context)) {
                result.error = QString::fromUtf8(aicore_attribute_last_error(context));
                aicore_attribute_free(context);
                emit finished(result);
                return;
            }
            m_context = context;
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::Ocr) {
            // The picker entry is the rec GGUF; the paired det GGUF arrives
            // through the dependency channel (metricModelPath), exactly like
            // the face landmark model for facedetect-dense.
            if (metricModelPath.isEmpty()) {
                result.error = tr("OCR requires the paired det model");
                emit finished(result);
                return;
            }
            aicore_ocr_options options{};
            options.struct_size = sizeof(options);
            options.abi_version = AICORE_OCR_ABI_VERSION;
            const QByteArray detBytes =
                    QFileInfo(metricModelPath).absoluteFilePath().toUtf8();
            options.det_model_path = detBytes.constData();
            options.rec_model_path = pathBytes.constData();
            options.device = deviceBytes.constData();
            // X-AnyLabeling's PPOCRv4.parse_args, rather than the standalone
            // PaddleOCR 3.x pipeline defaults. All image processing stays in
            // the public OCR runtime.
            options.det_limit_type = 1;
            options.det_max_side = 960;
            options.det_box_thresh = 0.3f;
            options.det_box_score_thresh = 0.6f;
            options.det_unclip_ratio = 1.5f;
            options.rec_max_batch = 6;
            const auto* preset = aicore_ocr_preset_by_rec_filename(
                    QFileInfo(modelPath).fileName().toUtf8().constData());
            if (preset) {
                options.det_max_side = preset->det_max_side;
                options.det_box_thresh = preset->det_thresh;
                options.det_box_score_thresh = preset->det_box_thresh;
                options.det_unclip_ratio = preset->det_unclip_ratio;
                options.det_max_candidates = preset->det_max_candidates;
                options.rec_max_batch = preset->rec_batch;
                options.cls_max_batch = preset->cls_batch;
                options.cls_thresh = preset->cls_thresh;
            }
            const QByteArray clsBytes = prompt.ocrClassifierModelPath.toUtf8();
            options.cls_model_path =
                    clsBytes.isEmpty() ? nullptr : clsBytes.constData();
            const QByteArray layoutBytes = prompt.ocrLayoutModelPath.toUtf8();
            options.layout_model_path =
                    layoutBytes.isEmpty() ? nullptr : layoutBytes.constData();
            m_context = aicore_ocr_context_create(&options);
            if (!m_context) {
                result.error = QString::fromUtf8(aicore_ocr_last_error());
                if (result.error.isEmpty())
                    result.error = tr("Unable to load OCR models");
                emit finished(result);
                return;
            }
            m_ocrLayoutModelPath = prompt.ocrLayoutModelPath;
            m_ocrClassifierModelPath = prompt.ocrClassifierModelPath;
            m_modelPath = modelPath;
            m_metricModelPath = metricModelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::DeepLSD) {
            aicore_deeplsd_options* options = aicore_deeplsd_options_new();
            if (!options) {
                result.error = tr("Unable to create DeepLSD options");
                emit finished(result);
                return;
            }
            aicore_deeplsd_options_set_device(options, deviceBytes.constData());
            auto* context =
                    aicore_deeplsd_load_opts(pathBytes.constData(), options);
            aicore_deeplsd_options_free(options);
            if (!context || !aicore_deeplsd_is_ready(context)) {
                result.error =
                        context ? QString::fromUtf8(
                                          aicore_deeplsd_last_error(context))
                                : tr("Unable to load DeepLSD model");
                if (result.error.isEmpty())
                    result.error = tr("Unable to load DeepLSD model");
                aicore_deeplsd_free(context);
                emit finished(result);
                return;
            }
            m_context = context;
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::ALIKED) {
            aicore_aliked_options* options = aicore_aliked_options_new();
            if (!options) {
                result.error = tr("Unable to create ALIKED options");
                emit finished(result);
                return;
            }
            aicore_aliked_options_set_device(options, deviceBytes.constData());
            aicore_aliked_options_set_max_keypoints(options, 1024);
            aicore_aliked_options_set_resize_long_edge(options, 1024);
            auto* context =
                    aicore_aliked_load_opts(pathBytes.constData(), options);
            aicore_aliked_options_free(options);
            if (!context || !aicore_aliked_is_ready(context)) {
                result.error =
                        context ? QString::fromUtf8(
                                          aicore_aliked_last_error(context))
                                : tr("Unable to load ALIKED model");
                if (result.error.isEmpty())
                    result.error = tr("Unable to load ALIKED model");
                aicore_aliked_free(context);
                emit finished(result);
                return;
            }
            m_context = context;
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::Ram) {
            aicore_tagging_options opts = aicore_tagging_default_options();
            opts.model_path = pathBytes.constData();
            opts.device = deviceBytes.constData();
            opts.threads = 8;
            opts.generate_caption = task.contains(QStringLiteral("caption"));
            opts.cpu_precision = AICORE_TAGGING_CPU_AUTO;

            m_context = aicore_tagging_create(&opts);
            if (!m_context ||
                !aicore_tagging_ready(
                        static_cast<aicore_tagging_ctx*>(m_context))) {
                result.error = QString::fromUtf8(aicore_tagging_last_error(
                        static_cast<aicore_tagging_ctx*>(m_context)));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load RAM model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
        } else if (kind == ModelKind::RelateAnything) {
            aicore_relation_options relOpts = aicore_relation_default_options();
            relOpts.model_path = pathBytes.constData();
            relOpts.device = deviceBytes.constData();
            relOpts.threads = 8;
            relOpts.max_boxes = unsigned(prompt.relate.maxBoxes);
            relOpts.topk = 128;
            relOpts.decompose = 1;
            relOpts.raw_logits = 1;
            // Custom predicates are encoded by the relation model's own text
            // student + CLIP tokenizer, so no retraining is needed.
            const QStringList predicates = prompt.relate.predicateList();
            const QByteArray studentBytes = prompt.relateStudentPath.toUtf8();
            const QByteArray tokenizerBytes =
                    prompt.relateTokenizerPath.toUtf8();
            std::vector<QByteArray> predicateStorage;
            std::vector<const char*> predicatePointers;
            for (const QString& name : predicates)
                predicateStorage.push_back(name.toUtf8());
            for (const QByteArray& name : predicateStorage)
                predicatePointers.push_back(name.constData());
            if (!predicatePointers.empty()) {
                if (prompt.relateStudentPath.isEmpty() ||
                    prompt.relateTokenizerPath.isEmpty()) {
                    result.error =
                            tr("Custom predicates need the relation text "
                               "student and tokenizer; download them first");
                    emit finished(result);
                    return;
                }
                relOpts.text_student_path = studentBytes.constData();
                relOpts.tokenizer_path = tokenizerBytes.constData();
                relOpts.predicate_texts = predicatePointers.data();
                relOpts.predicate_text_count = predicatePointers.size();
            }

            m_context = aicore_relation_create(&relOpts);
            if (!m_context ||
                !aicore_relation_ready(
                        static_cast<aicore_relation_ctx*>(m_context))) {
                result.error = QString::fromUtf8(aicore_relation_last_error(
                        static_cast<aicore_relation_ctx*>(m_context)));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load RelateAnything model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
            m_vocabText = relateKey;
        } else {
            aicore_yolo_options* options = aicore_yolo_options_new();
            if (!options) {
                result.error = tr("Unable to create YOLO options");
                emit finished(result);
                return;
            }
            aicore_yolo_options_set_device(options, deviceBytes.constData());
            if (visualPrompt) {
                if (aicore_yolo_gguf_has_savpe(pathBytes.constData()) == 0) {
                    aicore_yolo_options_free(options);
                    result.error =
                            tr("YOLOE visual prompts require a GGUF with SAVPE "
                               "weights; select a visual-prompt checkpoint");
                    emit finished(result);
                    return;
                }
                std::vector<float> boxes;
                boxes.reserve(size_t(prompt.visualPrompts.size()) * 4);
                for (const QRectF& box : prompt.visualPrompts) {
                    const QRectF normalized = box.normalized();
                    boxes.push_back(float(normalized.left()));
                    boxes.push_back(float(normalized.top()));
                    boxes.push_back(float(normalized.right()));
                    boxes.push_back(float(normalized.bottom()));
                }
                aicore_yolo_options_set_visual_prompts(
                        options, boxes.data(),
                        int32_t(prompt.visualPrompts.size()));
            }
            std::vector<QByteArray> names;
            std::vector<const char*> classPointers;
            names.reserve(size_t(vocabulary.size()));
            classPointers.reserve(size_t(vocabulary.size()));
            for (const QString& name : vocabulary)
                names.push_back(name.toUtf8());
            for (const QByteArray& name : names)
                classPointers.push_back(name.constData());
            if (!visualPrompt && !classPointers.empty()) {
                aicore_yolo_options_set_classes(options, classPointers.data(),
                                                int32_t(classPointers.size()));
                const QByteArray textPath = prompt.textModelPath.toUtf8();
                aicore_yolo_options_set_text_model(options,
                                                   textPath.constData());
            }
            m_context = aicore_yolo_load_opts(pathBytes.constData(), options);
            aicore_yolo_options_free(options);
            if (!m_context) {
                result.error =
                        QString::fromUtf8(aicore_yolo_last_error(nullptr));
                if (result.error.isEmpty())
                    result.error = tr("Unable to load model");
                emit finished(result);
                return;
            }
            m_modelPath = modelPath;
            m_device = device;
            m_kind = kind;
            m_textModelPath = prompt.textModelPath;
            m_vocabText = prompt.text;
            m_promptCount = vocabulary.size();
            m_visualPrompts = prompt.visualPrompts;
        }
    }
    if (kind == ModelKind::SAM3) {
        auto* context = static_cast<aicore_sam3_ctx*>(m_context);
        const bool textPrompt = task == QStringLiteral("sam3-pcs");
        if (textPrompt ? prompt.text.isEmpty()
                       : (prompt.positive.isEmpty() && !prompt.useBox &&
                          prompt.rois.isEmpty())) {
            result.error = tr("Add a positive point or box prompt");
        } else {
            const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
            std::vector<aicore_sam3_point> positive, negative;
            positive.reserve(prompt.positive.size());
            negative.reserve(prompt.negative.size());
            for (const QPointF& point : prompt.positive)
                positive.push_back({float(point.x()), float(point.y())});
            for (const QPointF& point : prompt.negative)
                negative.push_back({float(point.x()), float(point.y())});
            const int promptCount =
                    prompt.rois.isEmpty() ? 1 : prompt.rois.size();
            for (int roi = 0; roi < promptCount; ++roi) {
                aicore_sam3_seg_result* segment = nullptr;
                if (textPrompt) {
                    const QByteArray text = prompt.text.toUtf8();
                    aicore_sam3_pcs_prompt request{};
                    request.text = text.constData();
                    request.score_threshold = float(confidence);
                    segment = aicore_sam3_segment_pcs_rgb(
                            context, &request, rgb.constBits(), rgb.width(),
                            rgb.height(), size_t(rgb.bytesPerLine()));
                } else {
                    aicore_sam3_pvs_prompt request{};
                    request.pos_points =
                            prompt.rois.isEmpty() && !positive.empty()
                                    ? positive.data()
                                    : nullptr;
                    request.n_pos_points =
                            prompt.rois.isEmpty() ? int(positive.size()) : 0;
                    request.neg_points =
                            prompt.rois.isEmpty() && !negative.empty()
                                    ? negative.data()
                                    : nullptr;
                    request.n_neg_points =
                            prompt.rois.isEmpty() ? int(negative.size()) : 0;
                    request.use_box =
                            prompt.rois.isEmpty() ? int(prompt.useBox) : 1;
                    const QRectF box = prompt.rois.isEmpty()
                                               ? prompt.box
                                               : prompt.rois.at(roi);
                    if (request.use_box)
                        request.box = {float(box.left()), float(box.top()),
                                       float(box.right()), float(box.bottom())};
                    request.multimask = 0;
                    segment = aicore_sam3_segment_pvs_rgb(
                            context, &request, rgb.constBits(), rgb.width(),
                            rgb.height(), size_t(rgb.bytesPerLine()));
                }
                if (!segment) {
                    result.error =
                            QString::fromUtf8(aicore_sam3_last_error(context));
                    if (result.error.isEmpty())
                        result.error = tr("SAM3 segmentation failed");
                    result.shapes.clear();
                    break;
                }
                for (int i = 0; i < aicore_sam3_seg_det_count(segment); ++i) {
                    const aicore_sam3_plane_view mask =
                            aicore_sam3_seg_mask_at(segment, i);
                    const QVector<QPointF> polygon = largestMaskPolygon(
                            static_cast<const uint8_t*>(mask.data), mask.width,
                            mask.height, mask.row_stride_bytes);
                    if (polygon.size() < 3) continue;
                    AnnotationShape shape;
                    shape.type = QStringLiteral("polygon");
                    shape.label =
                            textPrompt ? prompt.text
                            : !prompt.rois.isEmpty() &&
                                            roi < prompt.roiLabels.size() &&
                                            !prompt.roiLabels.at(roi).isEmpty()
                                    ? prompt.roiLabels.at(roi)
                            : prompt.label.isEmpty() ? tr("object")
                                                     : prompt.label;
                    shape.points = polygon;
                    shape.mask = instanceMask(
                            static_cast<const uint8_t*>(mask.data), mask.width,
                            mask.height, mask.row_stride_bytes, image.size());
                    shape.extra.insert(
                            QStringLiteral("score"),
                            double(aicore_sam3_seg_det_score_at(segment, i)));
                    if (!prompt.rois.isEmpty())
                        shape.extra.insert(QStringLiteral("source_roi"), roi);
                    result.shapes.append(shape);
                }
                aicore_sam3_seg_result_free(segment);
            }
        }
        const char* resolved = aicore_sam3_context_backend_name(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    const QImage pixels = image.convertToFormat(QImage::Format_RGBA8888);
    const aicore_image_view view{pixels.constBits(), pixels.width(),
                                 pixels.height(), size_t(pixels.bytesPerLine()),
                                 AICORE_IMAGE_RGBA8};
    if (kind == ModelKind::Attribute) {
        auto* context = static_cast<aicore_attribute_ctx*>(m_context);
        aicore_attribute_result* output = nullptr;
        if (aicore_attribute_run(context, &view, &output) != 0 || !output) {
            result.error = QString::fromUtf8(aicore_attribute_last_error(context));
        } else {
            AnnotationShape shape;
            shape.type = QStringLiteral("rectangle");
            shape.label = QString::fromUtf8(output->label);
            shape.points = {{0, 0}, {double(output->width), 0},
                {double(output->width), double(output->height)}, {0, double(output->height)}};
            QJsonObject attributes;
            for (size_t i = 0; i < output->attribute_count; ++i) {
                const auto& attribute = output->attributes[i];
                attributes.insert(QString::fromUtf8(attribute.name), QString::fromUtf8(attribute.value));
            }
            shape.extra.insert(QStringLiteral("attributes"), attributes);
            result.shapes.append(shape);
            result.replaceShapes = true;
            result.summary = tr("%1 attributes: %2").arg(shape.label).arg(output->attribute_count);
        }
        aicore_attribute_result_free(output);
        result.device = QString::fromUtf8(aicore_attribute_context_device(context));
        emit finished(result);
        return;
    }
    if (kind == ModelKind::Depth) {
        auto* context = static_cast<aicore_depth_ctx*>(m_context);
        aicore_depth_dense_result dense{};
        if (aicore_depth_depth_image(context, &view, &dense) != 0 ||
            !dense.depth || dense.width <= 0 || dense.height <= 0) {
            result.error = QString::fromUtf8(aicore_depth_last_error(context));
            if (result.error.isEmpty())
                result.error = tr("Depth estimation failed");
        } else {
            const size_t count = size_t(dense.width) * size_t(dense.height);
            float minimum = std::numeric_limits<float>::infinity();
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(dense.depth[i])) continue;
                minimum = std::min(minimum, dense.depth[i]);
                maximum = std::max(maximum, dense.depth[i]);
            }
            if (!(minimum < maximum)) {
                result.error = tr("Depth output has no usable range");
            } else {
                QImage depth(dense.width, dense.height,
                             QImage::Format_Grayscale8);
                for (int y = 0; y < dense.height; ++y) {
                    uchar* row = depth.scanLine(y);
                    for (int x = 0; x < dense.width; ++x) {
                        const float value =
                                dense.depth[size_t(y) * size_t(dense.width) +
                                            size_t(x)];
                        row[x] = std::isfinite(value)
                                         ? uchar(std::clamp(
                                                   (value - minimum) /
                                                           (maximum - minimum) *
                                                           255.0f,
                                                   0.0f, 255.0f))
                                         : 0;
                    }
                }
                result.depthPreview =
                        depth.scaled(image.size(), Qt::IgnoreAspectRatio,
                                     Qt::SmoothTransformation);
                result.summary = tr("Depth map generated");
            }
        }
        aicore_depth_dense_result_free(&dense);
        const char* resolved = aicore_depth_device_name(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::FaceDetect) {
        auto* context = static_cast<aicore_facedetect_ctx*>(m_context);
        const bool dense = task == QStringLiteral("facedetect-dense");
        const int status =
                dense ? aicore_facedetect_dense_landmarks_image(
                                context,
                                static_cast<aicore_facedetect_ctx*>(
                                        m_landmarkContext),
                                &view, float(confidence))
                      : aicore_facedetect_detect_image(context, &view);
        if (status != 0) {
            result.error =
                    QString::fromUtf8(aicore_facedetect_last_error(context));
        } else {
            const char* names[] = {"left_eye", "right_eye", "nose",
                                   "left_mouth", "right_mouth"};
            const size_t count =
                    dense ? aicore_facedetect_dense_face_count(context)
                          : aicore_facedetect_detection_count(context);
            for (size_t i = 0; i < count; ++i) {
                aicore_facedetect_detection face{};
                if ((dense ? aicore_facedetect_dense_detection_at(context, i,
                                                                  &face)
                           : aicore_facedetect_detection_at(context, i,
                                                            &face)) != 0 ||
                    face.score < confidence)
                    continue;
                AnnotationShape box;
                box.type = QStringLiteral("rectangle");
                box.label = tr("face");
                box.points = {
                        QPointF(face.x1, face.y1), QPointF(face.x2, face.y1),
                        QPointF(face.x2, face.y2), QPointF(face.x1, face.y2)};
                box.extra.insert(QStringLiteral("score"), double(face.score));
                box.extra.insert(QStringLiteral("group_id"), int(i));
                result.shapes.append(box);
                for (int k = 0; k < 5; ++k) {
                    const float x = face.landmarks_xy10[2 * k];
                    const float y = face.landmarks_xy10[2 * k + 1];
                    if (!std::isfinite(x) || !std::isfinite(y)) continue;
                    AnnotationShape point;
                    point.type = QStringLiteral("point");
                    point.label = QString::fromLatin1(names[k]);
                    point.points = {QPointF(x, y)};
                    point.extra.insert(QStringLiteral("group_id"), int(i));
                    result.shapes.append(point);
                }
                if (dense) {
                    for (int threeD = 0; threeD <= 1; ++threeD) {
                        const size_t points =
                                aicore_facedetect_dense_point_count(context, i,
                                                                    threeD);
                        for (size_t j = 0; j < points; ++j) {
                            aicore_facedetect_landmark_point value{};
                            if (aicore_facedetect_dense_point_at(
                                        context, i, threeD, j, &value) != 0 ||
                                !std::isfinite(value.x) ||
                                !std::isfinite(value.y) ||
                                !std::isfinite(value.z))
                                continue;
                            AnnotationShape point;
                            point.type = QStringLiteral("point");
                            point.label =
                                    threeD ? tr("landmark3d_%1").arg(j + 1)
                                           : tr("landmark2d_%1").arg(j + 1);
                            point.points = {QPointF(value.x, value.y)};
                            point.extra.insert(QStringLiteral("group_id"),
                                               int(i));
                            if (threeD)
                                point.extra.insert(QStringLiteral("z"),
                                                   double(value.z));
                            result.shapes.append(point);
                        }
                    }
                }
            }
        }
        result.device = device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::Ocr) {
        auto* context = static_cast<aicore_ocr_ctx*>(m_context);
        // Selected annotations retain their original polygons and metadata;
        // the shared OCR task owns source crop normalization and recognition.
        if (!prompt.rois.isEmpty()) {
            std::vector<std::vector<float>> coordinates(prompt.rois.size());
            std::vector<aicore_ocr_text_region> regions(prompt.rois.size());
            for (int i = 0; i < prompt.rois.size(); ++i) {
                const QRectF& r = prompt.rois.at(i);
                const QVector<QPointF> points = prompt.ocrQuads.value(
                        i, {r.topLeft(), r.topRight(), r.bottomRight(),
                            r.bottomLeft()});
                if (points.size() < 3) {
                    result.error =
                            tr("OCR needs at least three text-region points");
                    emit finished(result);
                    return;
                }
                for (const auto& point : points) {
                    // The source adapter truncates selected coordinates to
                    // integer pixels before passing them to TextSystem.
                    for (const double value : {point.x(), point.y()}) {
                        if (!std::isfinite(value) ||
                            value < std::numeric_limits<int>::min() ||
                            value > std::numeric_limits<int>::max()) {
                            result.error = tr("Invalid OCR region coordinate");
                            emit finished(result);
                            return;
                        }
                        coordinates[i].push_back(float(int(value)));
                    }
                }
                regions[i] = {coordinates[i].data(), int32_t(points.size())};
            }
            aicore_ocr_result* rec = aicore_ocr_recognize_regions(
                    context, &view, regions.data(),
                    static_cast<int32_t>(prompt.rois.size()));
            if (!rec) {
                result.error = QString::fromUtf8(aicore_ocr_last_error());
                if (result.error.isEmpty()) result.error = tr("OCR failed");
            } else {
                const int n = aicore_ocr_box_count(rec);
                for (int i = 0; i < n; ++i) {
                    const aicore_ocr_box* box = aicore_ocr_box_at(rec, i);
                    if (!box || !box->text) continue;
                    AnnotationShape shape;
                    shape.type = QStringLiteral("rectangle");
                    shape.label = QStringLiteral("text");
                    shape.points = {QPointF(box->quad[0][0], box->quad[0][1]),
                                    QPointF(box->quad[1][0], box->quad[1][1]),
                                    QPointF(box->quad[2][0], box->quad[2][1]),
                                    QPointF(box->quad[3][0], box->quad[3][1])};
                    shape.extra.insert(QStringLiteral("description"),
                                       QString::fromUtf8(box->text));
                    shape.extra.insert(QStringLiteral("score"),
                                       double(box->score));
                    shape.extra.insert(QStringLiteral("language"),
                                       QString::fromLatin1(box->language));
                    result.shapes.append(shape);
                    result.sourceIndices.append(
                            prompt.roiSourceIndices.value(i, -1));
                }
                result.summary = tr("%1 text region(s) recognized").arg(n);
                result.sourceRois = prompt.rois;
                aicore_ocr_result_release(rec);
            }
            result.device =
                    QString::fromUtf8(aicore_ocr_context_device(context));
            emit finished(result);
            return;
        }
        aicore_ocr_result* ocr = aicore_ocr_run(context, &view);
        if (!ocr) {
            result.error = QString::fromUtf8(aicore_ocr_last_error());
            if (result.error.isEmpty()) result.error = tr("OCR failed");
        } else {
            // Upstream PP-OCR auto-labeling semantics (X-AnyLabeling
            // ppocr_v4): every region is a "text" shape whose description
            // carries the transcription, the quad becomes a rectangle only
            // when it is axis-aligned (otherwise a quadrilateral), and the
            // rec score gate is the model's drop_score (0.5) - upstream
            // does not wire the generic confidence slider for OCR.
            constexpr float kOcrDropScore = 0.5f;
            auto squareDist = [](const QPoint& a, const QPoint& b) {
                const qint64 dx = qint64(a.x()) - b.x();
                const qint64 dy = qint64(a.y()) - b.y();
                return dx * dx + dy * dy;
            };
            const int count = aicore_ocr_box_count(ocr);
            int textGroup = 0;
            for (int i = 0; i < count; ++i) {
                const aicore_ocr_box* box = aicore_ocr_box_at(ocr, i);
                if (!box || !box->text || box->score < kOcrDropScore) continue;
                const QString text = QString::fromUtf8(box->text);
                // The source casts detection corners to int32 before both
                // shape classification and JSON publication.
                const QPoint corners[4] = {
                        QPoint(int(box->quad[0][0]), int(box->quad[0][1])),
                        QPoint(int(box->quad[1][0]), int(box->quad[1][1])),
                        QPoint(int(box->quad[2][0]), int(box->quad[2][1])),
                        QPoint(int(box->quad[3][0]), int(box->quad[3][1]))};
                AnnotationShape shape;
                // Upstream is_possible_rectangle: the two smallest
                // consecutive edge lengths equal the two largest.
                qint64 edges[4];
                for (int e = 0; e < 4; ++e)
                    edges[e] = squareDist(corners[e], corners[(e + 1) % 4]);
                std::sort(edges, edges + 4);
                if (edges[0] == edges[1] && edges[2] == edges[3]) {
                    shape.type = QStringLiteral("rectangle");
                    // Upstream rebuilds the four corners from the first
                    // and third points so the axis-aligned rectangle
                    // round-trips through the two-point editors.
                    shape.points = {corners[0],
                                    QPointF(corners[2].x(), corners[0].y()),
                                    corners[2],
                                    QPointF(corners[0].x(), corners[2].y())};
                } else {
                    shape.type = QStringLiteral("quadrilateral");
                    shape.points = {corners[0], corners[1], corners[2],
                                    corners[3]};
                }
                shape.label = QStringLiteral("text");
                shape.extra.insert(QStringLiteral("description"), text);
                shape.extra.insert(QStringLiteral("score"), double(box->score));
                shape.extra.insert(QStringLiteral("language"),
                                   QString::fromLatin1(box->language));
                shape.extra.insert(QStringLiteral("group_id"), textGroup++);
                result.shapes.append(shape);
            }
            result.summary = tr("%1 text region(s) recognized").arg(textGroup);
            // Optional layout stage: PP-Structure region classification
            // becomes a rectangle annotation per region labeled with the
            // CDLA class name; the reading order lands in extra.
            const int regionCount = aicore_ocr_region_count(ocr);
            auto* ocrCtx = static_cast<aicore_ocr_ctx*>(m_context);
            for (int r = 0; r < regionCount; ++r) {
                const aicore_ocr_region* region = aicore_ocr_region_at(ocr, r);
                if (!region || region->score <= 0) continue;
                const QPointF tl(region->quad[0][0], region->quad[0][1]);
                const QPointF br(region->quad[2][0], region->quad[2][1]);
                AnnotationShape shape;
                shape.type = QStringLiteral("rectangle");
                shape.points = {tl, QPointF(br.x(), tl.y()), br,
                                QPointF(tl.x(), br.y())};
                // The CDLA class name from the layout GGUF's embedded
                // table, falling back to the numeric id.
                const char* className =
                        aicore_ocr_label_at(ocrCtx, region->class_id);
                shape.label = className ? QString::fromUtf8(className)
                                        : QString::number(region->class_id);
                shape.extra.insert(QStringLiteral("class_id"),
                                   region->class_id);
                shape.extra.insert(QStringLiteral("reading_order"),
                                   region->reading_order);
                shape.extra.insert(QStringLiteral("score"),
                                   double(region->score));
                result.shapes.append(shape);
            }
            if (regionCount > 0) {
                result.summary += tr(" · %1 layout region(s)").arg(regionCount);
            }
            aicore_ocr_result_release(ocr);
        }
        result.device = QString::fromUtf8(aicore_ocr_context_device(context));
        emit finished(result);
        return;
    }
    if (kind == ModelKind::DeepLSD) {
        auto* context = static_cast<aicore_deeplsd_ctx*>(m_context);
        const int maxEdge = 1024;
        const qreal scale = std::min(qreal(maxEdge) / image.width(),
                                     qreal(maxEdge) / image.height());
        const QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
        const QImage processed =
                (scale < 1.0)
                        ? gray.scaled(std::max(1, int(gray.width() * scale)),
                                      std::max(1, int(gray.height() * scale)),
                                      Qt::IgnoreAspectRatio,
                                      Qt::SmoothTransformation)
                        : gray;
        aicore_deeplsd_segment* segments = nullptr;
        int32_t segmentCount = 0;
        float* distance = nullptr;
        float* angle = nullptr;
        int32_t outputWidth = 0;
        int32_t outputHeight = 0;
        const int status = aicore_deeplsd_extract_segments(
                context, processed.constBits(), processed.width(),
                processed.height(), processed.bytesPerLine(), &segments,
                &segmentCount, &distance, &angle, &outputWidth, &outputHeight);
        if (status != 0) {
            result.error =
                    QString::fromUtf8(aicore_deeplsd_last_error(context));
        } else {
            const qreal sx = qreal(image.width()) / processed.width();
            const qreal sy = qreal(image.height()) / processed.height();
            for (int32_t i = 0; i < segmentCount; ++i) {
                const auto& segment = segments[i];
                if (!std::isfinite(segment.x1) || !std::isfinite(segment.y1) ||
                    !std::isfinite(segment.x2) || !std::isfinite(segment.y2) ||
                    !std::isfinite(segment.score) || segment.score < confidence)
                    continue;
                AnnotationShape shape;
                shape.type = QStringLiteral("linestrip");
                shape.label = tr("line");
                shape.points = {QPointF(std::clamp(qreal(segment.x1) * sx, 0.0,
                                                   qreal(image.width() - 1)),
                                        std::clamp(qreal(segment.y1) * sy, 0.0,
                                                   qreal(image.height() - 1))),
                                QPointF(std::clamp(qreal(segment.x2) * sx, 0.0,
                                                   qreal(image.width() - 1)),
                                        std::clamp(qreal(segment.y2) * sy, 0.0,
                                                   qreal(image.height() - 1)))};
                shape.extra.insert(QStringLiteral("score"),
                                   double(segment.score));
                result.shapes.append(shape);
            }
            result.summary = tr("%1 line segments").arg(result.shapes.size());
        }
        std::free(segments);
        std::free(distance);
        std::free(angle);
        if (char* info = aicore_deeplsd_info_json(context)) {
            const QJsonObject object =
                    QJsonDocument::fromJson(QByteArray(info)).object();
            result.device = object.value(QStringLiteral("device")).toString();
            aicore_deeplsd_free_buffer(info);
        }
        if (result.device.isEmpty()) result.device = device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::ALIKED) {
        auto* context = static_cast<aicore_aliked_ctx*>(m_context);
        const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
        const aicore_image_view rgbView{
                rgb.constBits(), rgb.width(), rgb.height(),
                size_t(rgb.bytesPerLine()), AICORE_IMAGE_RGB8};
        aicore_lightglue_features features{};
        if (aicore_aliked_extract_image_view(context, &rgbView, &features) !=
            0) {
            result.error = QString::fromUtf8(aicore_aliked_last_error(context));
        } else {
            for (int32_t i = 0; i < features.n_keypoints; ++i) {
                const auto& keypoint = features.keypoints[i];
                if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y) ||
                    keypoint.x < 0.0f || keypoint.y < 0.0f ||
                    keypoint.x >= image.width() || keypoint.y >= image.height())
                    continue;
                AnnotationShape shape;
                shape.type = QStringLiteral("point");
                shape.label = tr("keypoint");
                shape.points = {QPointF(keypoint.x, keypoint.y)};
                shape.extra.insert(QStringLiteral("feature_index"), int(i));
                shape.extra.insert(QStringLiteral("scale"),
                                   double(keypoint.scale));
                shape.extra.insert(QStringLiteral("orientation"),
                                   double(keypoint.orientation));
                result.shapes.append(shape);
            }
            result.summary = tr("%1 keypoints").arg(result.shapes.size());
        }
        aicore_lightglue_free_features(&features);
        if (char* info = aicore_aliked_info_json(context)) {
            const QJsonObject object =
                    QJsonDocument::fromJson(QByteArray(info)).object();
            result.device = object.value(QStringLiteral("device")).toString();
            aicore_aliked_free_buffer(info);
        }
        if (result.device.isEmpty()) result.device = device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::Ram) {
        auto* context = static_cast<aicore_tagging_ctx*>(m_context);
        const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
        aicore_image_view view{};
        view.data = rgb.constBits();
        view.width = rgb.width();
        view.height = rgb.height();
        view.row_stride_bytes = size_t(rgb.bytesPerLine());
        view.format = AICORE_IMAGE_RGB8;

        // Default request: the model's embedded per-class thresholds, as the
        // upstream RAM adapter does. The confidence slider must not override
        // them with a single global value.
        auto req = aicore_tagging_default_request_v2();
        if (task == QStringLiteral("ram"))
            req.resize_mode = AICORE_TAGGING_RESIZE_OPENCV;

        aicore_tagging_result* raw = nullptr;
        int rc = aicore_tagging_infer_image_v2(context, &view, &req, &raw);

        if (rc != 0 || !raw) {
            result.error =
                    QString::fromUtf8(aicore_tagging_last_error(context));
            if (result.error.isEmpty())
                result.error = tr("RAM inference failed");
        } else {
            const QSet<QString> deleted(prompt.ramDeleteTags.cbegin(),
                                        prompt.ramDeleteTags.cend());
            const QSet<QString> retained(prompt.ramFilterTags.cbegin(),
                                         prompt.ramFilterTags.cend());
            for (size_t i = 0; i < raw->tag_count; ++i) {
                const auto& t = raw->tags[i];
                if (!t.selected || !t.name) continue;
                const QString name = QString::fromUtf8(t.name);
                if (annotationKeepRamTag(name, deleted, retained))
                    result.tags.append(
                            annotationRamTag(name, prompt.ramTagMode));
            }
            result.hasTags = true;
            result.summary = tr("%1 tags").arg(result.tags.size());
            if (task == QStringLiteral("ram-caption")) {
                result.description =
                        QString::fromUtf8(raw->caption ? raw->caption : "");
                result.hasDescription = true;
            }
            if (!result.description.isEmpty()) {
                result.summary += QStringLiteral(": ") + result.description;
            }

            aicore_tagging_result_free(raw);
        }

        const char* resolved = aicore_tagging_context_device(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::RelateAnything) {
        auto* context = static_cast<aicore_relation_ctx*>(m_context);
        const RelateOptions& options = prompt.relate;
        const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
        const aicore_image_view view{rgb.constBits(), rgb.width(), rgb.height(),
                                     size_t(rgb.bytesPerLine()),
                                     AICORE_IMAGE_RGB8};
        // Objects: the caller's selected rectangles when there are at least
        // two (edges then refer to canvas shape indices), otherwise the
        // detector stage of the official demos proposes them and they are
        // returned as ordinary rectangles (edges then refer to positions in
        // result.shapes, which the window offsets onto the canvas).
        std::vector<float> boxes;
        std::vector<float> confidences;
        QStringList labels;
        QVector<int> shapeIndex;
        const bool selected =
                prompt.rois.size() >= 2 &&
                prompt.roiSourceIndices.size() == prompt.rois.size();
        if (selected) {
            for (int i = 0; i < prompt.rois.size(); ++i) {
                const QRectF& roi = prompt.rois[i];
                boxes.insert(boxes.end(),
                             {float(roi.left()), float(roi.top()),
                              float(roi.right()), float(roi.bottom())});
                labels << (i < prompt.roiLabels.size() ? prompt.roiLabels[i]
                                                       : QString());
                shapeIndex << prompt.roiSourceIndices.at(i);
            }
        } else if (!prompt.relateDetectorPath.isEmpty()) {
            const QByteArray detectorBytes =
                    QFileInfo(prompt.relateDetectorPath)
                            .absoluteFilePath()
                            .toUtf8();
            const bool rfdetr = QFileInfo(prompt.relateDetectorPath)
                                        .fileName()
                                        .startsWith(QStringLiteral("rfdetr-"));
            // Confidence, NMS, box cap and the baked vocabulary are load-time
            // options, so they are part of the reuse key.
            const QString detectorKey =
                    prompt.relateDetectorPath + QLatin1Char('\n') + device +
                    QLatin1Char('\n') +
                    QString::number(options.detectConfidence) +
                    QLatin1Char('/') + QString::number(options.detectIou) +
                    QLatin1Char('/') + QString::number(options.maxBoxes) +
                    QLatin1Char('\n') + prompt.relateDetectorTextModel;
            if (!m_relateDetector || m_relateDetectorDevice != detectorKey) {
                releaseRelateDetector();
                const QByteArray deviceBytes = device.toUtf8();
                if (rfdetr) {
                    aicore_rfdetr_options* o = aicore_rfdetr_options_new();
                    aicore_rfdetr_options_set_device(o,
                                                     deviceBytes.constData());
                    m_relateDetector = aicore_rfdetr_load_opts(
                            detectorBytes.constData(), o);
                    aicore_rfdetr_options_free(o);
                } else {
                    aicore_yolo_options* o = aicore_yolo_options_new();
                    aicore_yolo_options_set_device(o, deviceBytes.constData());
                    aicore_yolo_options_set_conf_thres(
                            o, options.detectConfidence);
                    aicore_yolo_options_set_iou_thres(o, options.detectIou);
                    aicore_yolo_options_set_top_k(
                            o, uint32_t(std::max(options.maxBoxes * 4, 100)));
                    QList<QByteArray> classNames;
                    std::vector<const char*> classPointers;
                    QByteArray textModel;
                    if (!prompt.relateDetectorClasses.isEmpty() &&
                        !prompt.relateDetectorTextModel.isEmpty()) {
                        for (const auto& name : prompt.relateDetectorClasses)
                            classNames << name.toUtf8();
                        for (const auto& name : classNames)
                            classPointers.push_back(name.constData());
                        textModel = prompt.relateDetectorTextModel.toUtf8();
                        aicore_yolo_options_set_classes(
                                o, classPointers.data(),
                                int32_t(classPointers.size()));
                        aicore_yolo_options_set_text_model(
                                o, textModel.constData());
                    }
                    m_relateDetector =
                            aicore_yolo_load_opts(detectorBytes.constData(), o);
                    aicore_yolo_options_free(o);
                }
                m_relateDetectorRfdetr = rfdetr;
                m_relateDetectorPath = prompt.relateDetectorPath;
                m_relateDetectorDevice = detectorKey;
            }
            std::vector<RelateDecode::Candidate> found;
            QString detectorError;
            if (!m_relateDetector) {
                detectorError =
                        tr("Unable to load the RelateAnything "
                           "detector");
            } else if (rfdetr) {
                auto* rf = static_cast<aicore_rfdetr_ctx*>(m_relateDetector);
                if (aicore_rfdetr_detect_image(
                            rf, &view, options.detectConfidence,
                            uint32_t(std::max(options.maxBoxes * 4, 100))) !=
                    0) {
                    detectorError =
                            QString::fromUtf8(aicore_rfdetr_last_error(rf));
                } else {
                    for (int i = 0; i < aicore_rfdetr_detection_count(rf);
                         ++i) {
                        aicore_rfdetr_detection d{};
                        if (aicore_rfdetr_detection_at(rf, i, &d) != 0)
                            continue;
                        found.push_back(
                                {{d.x1, d.y1, d.x2, d.y2},
                                 d.score,
                                 d.class_name && d.class_name[0]
                                         ? QString::fromUtf8(d.class_name)
                                         : tr("object"),
                                 i});
                    }
                }
            } else {
                auto* yolo = static_cast<aicore_yolo_ctx*>(m_relateDetector);
                // Segmenters (the YOLOE prompt-free default) and plain
                // detectors use different entry points.
                if (QByteArray(aicore_yolo_context_task(yolo)) == "segment") {
                    std::unique_ptr<aicore_yolo_segment_result,
                                    decltype(&aicore_yolo_seg_result_free)>
                            seg(aicore_yolo_seg_image(yolo, &view),
                                aicore_yolo_seg_result_free);
                    if (!seg) {
                        detectorError =
                                QString::fromUtf8(aicore_yolo_last_error(yolo));
                    } else {
                        for (int i = 0;
                             i < aicore_yolo_seg_det_count(seg.get()); ++i) {
                            const auto d = aicore_yolo_seg_det_at(seg.get(), i);
                            const char* name = aicore_yolo_seg_det_class_name(
                                    seg.get(), i);
                            found.push_back({{d.x1, d.y1, d.x2, d.y2},
                                             d.score,
                                             name && name[0]
                                                     ? QString::fromUtf8(name)
                                                     : tr("object"),
                                             i});
                        }
                    }
                } else if (aicore_yolo_detect_image(yolo, &view) != 0) {
                    detectorError =
                            QString::fromUtf8(aicore_yolo_last_error(yolo));
                } else {
                    for (int i = 0; i < aicore_yolo_detection_count(yolo);
                         ++i) {
                        const auto d = aicore_yolo_detection_at(yolo, i);
                        const char* name =
                                aicore_yolo_detection_class_name(yolo, i);
                        found.push_back({{d.x1, d.y1, d.x2, d.y2},
                                         d.score,
                                         name && name[0]
                                                 ? QString::fromUtf8(name)
                                                 : tr("object"),
                                         i});
                    }
                }
            }
            if (!detectorError.isEmpty()) {
                result.error = detectorError;
            } else {
                found = RelateDecode::keepBest(std::move(found),
                                               options.maxBoxes, image.size(),
                                               options.maxBoxFraction, .7f);
                for (const auto& c : found) {
                    boxes.insert(boxes.end(), c.box, c.box + 4);
                    confidences.push_back(c.score);
                    labels << c.name;
                    AnnotationShape shape;
                    shape.type = QStringLiteral("rectangle");
                    shape.label = c.name;
                    shape.points = {QPointF(c.box[0], c.box[1]),
                                    QPointF(c.box[2], c.box[1]),
                                    QPointF(c.box[2], c.box[3]),
                                    QPointF(c.box[0], c.box[3])};
                    shape.extra.insert(QStringLiteral("score"),
                                       double(c.score));
                    shapeIndex << result.shapes.size();
                    result.shapes.append(shape);
                }
                result.relateAddedObjects = true;
            }
        }
        if (result.error.isEmpty() && boxes.size() < 8) {
            result.shapes.clear();
            result.relateAddedObjects = false;
            result.error =
                    prompt.relateDetectorPath.isEmpty()
                            ? tr("Select at least two visible rectangles "
                                 "to relate")
                            : tr("Fewer than two objects were found; "
                                 "select rectangles or lower the "
                                 "detector confidence");
        }
        if (result.error.isEmpty()) {
            aicore_relation_request request = aicore_relation_default_request();
            request.boxes_xyxy = boxes.data();
            request.object_count = boxes.size() / 4;
            aicore_relation_result* raw = nullptr;
            const int rc =
                    aicore_relation_infer_image(context, &view, &request, &raw);
            if (rc != 0 || !raw) {
                result.shapes.clear();
                result.relateAddedObjects = false;
                result.error =
                        QString::fromUtf8(aicore_relation_last_error(context));
                if (result.error.isEmpty())
                    result.error = rc == 1 ? tr("RelateAnything cancelled")
                                           : tr("RelateAnything failed");
            } else {
                const bool decomposed =
                        raw->spatial_count > 0 || raw->semantic_count > 0;
                float calA = 1.0f, calB = 0.0f;
                if (decomposed &&
                    aicore_relation_calibration(context, &calA, &calB) != 0) {
                    calA = 1.0f;
                    calB = 0.0f;
                }
                RelateDecode::Params params;
                params.relationThreshold = options.relationThreshold;
                params.spatialThreshold = options.spatialThreshold;
                params.maxRelations = options.maxRelations;
                params.maxSpatial = options.maxSpatial;
                const std::vector<RelateDecode::PickedEdge> picked =
                        RelateDecode::pickEdges(*raw, params, decomposed, calA,
                                                calB, confidences);
                const int objects = int(request.object_count);
                for (const RelateDecode::PickedEdge& p : picked) {
                    const auto& e = p.edge;
                    if (int(e.subject) >= objects || int(e.object) >= objects ||
                        e.subject == e.object || !raw->predicate_names ||
                        e.predicate >= raw->predicate_count)
                        continue;
                    QJsonObject triplet;
                    triplet.insert(QStringLiteral("subject_index"),
                                   shapeIndex.at(int(e.subject)));
                    triplet.insert(QStringLiteral("object_index"),
                                   shapeIndex.at(int(e.object)));
                    triplet.insert(QStringLiteral("subject"),
                                   labels.at(int(e.subject)));
                    triplet.insert(QStringLiteral("object"),
                                   labels.at(int(e.object)));
                    triplet.insert(QStringLiteral("predicate"),
                                   QString::fromUtf8(
                                           raw->predicate_names[e.predicate]));
                    triplet.insert(QStringLiteral("score"), double(e.score));
                    triplet.insert(QStringLiteral("type"),
                                   !decomposed ? QStringLiteral("relation")
                                   : p.spatial ? QStringLiteral("spatial")
                                               : QStringLiteral("semantic"));
                    result.relations.append(triplet);
                }
                result.summary = tr("%1 relations among %2 objects")
                                         .arg(result.relations.size())
                                         .arg(objects);
                aicore_relation_result_free(raw);
            }
        }
        const char* resolved = aicore_relation_context_device(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::GKD) {
        auto* context = static_cast<aicore_gkd_ctx*>(m_context);
        const QStringList names =
                prompt.text.split(QLatin1Char(','), QtCompat::SkipEmptyParts);
        std::vector<QByteArray> textStorage;
        textStorage.reserve(size_t(names.size()));
        for (const QString& name : names) {
            const QByteArray value = name.trimmed().toUtf8();
            if (!value.isEmpty()) textStorage.push_back(value);
        }
        if (textStorage.empty() && prompt.visualSupportPoints.isEmpty()) {
            result.error =
                    tr("Enter keypoint names or select visual support points");
        } else if (!textStorage.empty() &&
                   textStorage.size() !=
                           size_t(prompt.visualSupportPoints.size()) &&
                   !prompt.visualSupportPoints.isEmpty()) {
            result.error =
                    tr("GKD text names and visual support points must have the "
                       "same count");
        } else if (!prompt.visualSupportPoints.isEmpty() &&
                   prompt.visualSupportImage.isNull()) {
            result.error = tr("GKD visual support image is unavailable");
        } else {
            std::vector<const char*> texts;
            texts.reserve(textStorage.size());
            for (const QByteArray& value : textStorage)
                texts.push_back(value.constData());
            float box[4]{};
            aicore_gkd_detect_request request{};
            request.struct_size = sizeof(request);
            request.kps_texts = texts.data();
            request.n_kps_texts = int32_t(texts.size());
            const QImage support =
                    prompt.visualSupportImage.isNull()
                            ? QImage()
                            : prompt.visualSupportImage.convertToFormat(
                                      QImage::Format_RGBA8888);
            const aicore_image_view supportView{
                    support.constBits(), support.width(), support.height(),
                    size_t(support.bytesPerLine()), AICORE_IMAGE_RGBA8};
            std::vector<float> supportPoints;
            supportPoints.reserve(size_t(prompt.visualSupportPoints.size()) *
                                  2);
            for (const QPointF& point : prompt.visualSupportPoints) {
                supportPoints.push_back(float(point.x()));
                supportPoints.push_back(float(point.y()));
            }
            if (!supportPoints.empty()) {
                request.support_image = &supportView;
                request.support_kps_xy = supportPoints.data();
                request.n_support_kps =
                        int32_t(prompt.visualSupportPoints.size());
            }
            if (prompt.useBox && prompt.box.isValid()) {
                box[0] = float(prompt.box.left());
                box[1] = float(prompt.box.top());
                box[2] = float(prompt.box.right());
                box[3] = float(prompt.box.bottom());
                request.bbox_xyxy = box;
            }
            std::vector<float> rois;
            rois.reserve(size_t(prompt.rois.size()) * 4);
            for (const QRectF& roi : prompt.rois) {
                const QRectF normalized = roi.normalized();
                rois.push_back(float(normalized.left()));
                rois.push_back(float(normalized.top()));
                rois.push_back(float(normalized.right()));
                rois.push_back(float(normalized.bottom()));
            }
            if (prompt.rois.size() == 1 && !prompt.useBox) {
                std::copy_n(rois.data(), 4, box);
                request.bbox_xyxy = box;
            }
            const bool multiRoi = prompt.rois.size() > 1;
            const int detectStatus =
                    multiRoi
                            ? aicore_gkd_detect_image_multi(
                                      context, &view, &request, rois.data(),
                                      int32_t(prompt.rois.size()))
                            : aicore_gkd_detect_image(context, &view, &request);
            if (detectStatus != 0) {
                result.error =
                        QString::fromUtf8(aicore_gkd_last_error(context));
            } else {
                const int roiCount =
                        multiRoi ? aicore_gkd_result_roi_count(context) : 1;
                for (int roi = 0; roi < roiCount; ++roi) {
                    const int count =
                            multiRoi
                                    ? aicore_gkd_result_keypoint_count_at(
                                              context, roi)
                                    : aicore_gkd_result_keypoint_count(context);
                    const int groupId =
                            roi < prompt.roiGroupIds.size()
                                    ? prompt.roiGroupIds.at(roi)
                                    : (prompt.rois.isEmpty() ? -1 : roi);
                    for (int i = 0; i < count; ++i) {
                        const aicore_gkd_keypoint keypoint =
                                multiRoi ? aicore_gkd_result_keypoint_at_roi(
                                                   context, roi, i)
                                         : aicore_gkd_result_keypoint_at(
                                                   context, i);
                        if (keypoint.score < confidence ||
                            !std::isfinite(keypoint.x) ||
                            !std::isfinite(keypoint.y))
                            continue;
                        AnnotationShape shape;
                        shape.type = QStringLiteral("point");
                        const char* name =
                                multiRoi ? aicore_gkd_result_prompt_at_roi(
                                                   context, roi, i)
                                         : aicore_gkd_result_prompt_at(context,
                                                                       i);
                        shape.label = name ? QString::fromUtf8(name)
                                           : tr("keypoint %1").arg(i + 1);
                        shape.points = {QPointF(keypoint.x, keypoint.y)};
                        shape.extra.insert(QStringLiteral("score"),
                                           double(keypoint.score));
                        if (groupId >= 0)
                            shape.extra.insert(QStringLiteral("group_id"),
                                               groupId);
                        if (roi < prompt.rois.size())
                            shape.extra.insert(QStringLiteral("source_roi"),
                                               roi);
                        result.shapes.append(shape);
                    }
                }
            }
        }
        const char* resolved = aicore_gkd_context_device(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    if (kind == ModelKind::RMBG) {
        auto* context = static_cast<aicore_rmbg_ctx*>(m_context);
        uint8_t* alpha = nullptr;
        int32_t width = 0, height = 0;
        if (aicore_rmbg_alpha_mat_image_view(context, &view, &alpha, &width,
                                             &height) != 0) {
            result.error = QString::fromUtf8(aicore_rmbg_last_error(context));
        } else if (alpha && width > 0 && height > 0) {
            QByteArray binary(width * height, Qt::Uninitialized);
            for (int i = 0; i < binary.size(); ++i)
                binary[i] = alpha[i] >= 128 ? char(255) : char(0);
            const QVector<QPointF> polygon = largestMaskPolygon(
                    reinterpret_cast<const uint8_t*>(binary.constData()), width,
                    height, size_t(width));
            if (polygon.size() >= 3) {
                AnnotationShape shape;
                shape.type = QStringLiteral("polygon");
                shape.label = tr("foreground");
                shape.points = polygon;
                shape.mask = instanceMask(
                        reinterpret_cast<const uint8_t*>(binary.constData()),
                        width, height, size_t(width), pixels.size());
                result.shapes.append(shape);
            }
        }
        aicore_rmbg_free_buffer(alpha);
        result.device = device;
        emit finished(result);
        return;
    }
    if (rfdetr) {
        auto* context = static_cast<aicore_rfdetr_ctx*>(m_context);
        if (aicore_rfdetr_detect_image(context, &view, float(confidence),
                                       300) != 0) {
            result.error = QString::fromUtf8(aicore_rfdetr_last_error(context));
        } else {
            const bool segmentation =
                    aicore_rfdetr_context_has_segmentation(context) != 0;
            const int count = aicore_rfdetr_detection_count(context);
            result.shapes.reserve(count);
            for (int i = 0; i < count; ++i) {
                aicore_rfdetr_detection box{};
                if (aicore_rfdetr_detection_at(context, i, &box) != 0) continue;
                AnnotationShape shape;
                shape.type = QStringLiteral("rectangle");
                shape.label = className(box.class_name, int(box.class_id));
                shape.points = {
                        QPointF(box.x1, box.y1), QPointF(box.x2, box.y1),
                        QPointF(box.x2, box.y2), QPointF(box.x1, box.y2)};
                shape.extra.insert(QStringLiteral("score"), double(box.score));
                if (segmentation) {
                    int32_t width = 0, height = 0;
                    const int length = aicore_rfdetr_detection_mask(
                            context, i, nullptr, 0, &width, &height);
                    if (length > 0 && width > 0 && height > 0) {
                        QByteArray mask(length, Qt::Uninitialized);
                        if (aicore_rfdetr_detection_mask(
                                    context, i,
                                    reinterpret_cast<unsigned char*>(
                                            mask.data()),
                                    length, &width, &height) == length) {
                            QVector<QPointF> polygon = largestMaskPolygon(
                                    reinterpret_cast<const uint8_t*>(
                                            mask.constData()),
                                    width, height, size_t(width));
                            if (polygon.size() >= 3) {
                                for (QPointF& point : polygon) {
                                    point.setX(point.x() * pixels.width() /
                                               width);
                                    point.setY(point.y() * pixels.height() /
                                               height);
                                }
                                shape.type = QStringLiteral("polygon");
                                shape.points = polygon;
                                shape.mask = instanceMask(
                                        reinterpret_cast<const uint8_t*>(
                                                mask.constData()),
                                        width, height, size_t(width),
                                        pixels.size());
                            }
                        }
                    }
                }
                result.shapes.append(shape);
            }
        }
        const char* resolved = aicore_rfdetr_context_device(context);
        result.device = resolved ? QString::fromUtf8(resolved) : device;
        emit finished(result);
        return;
    }
    auto* context = static_cast<aicore_yolo_ctx*>(m_context);
    aicore_yolo_set_detect_thresholds(context, float(confidence),
                                      float(prompt.iouThreshold), 0);
    if (task == QStringLiteral("detect") || world) {
        std::vector<QByteArray> names;
        std::vector<const char*> classPointers;
        if (world) {
            names.reserve(size_t(vocabulary.size()));
            for (const QString& name : vocabulary)
                names.push_back(name.toUtf8());
            for (const QByteArray& name : names)
                classPointers.push_back(name.constData());
        }
        QVector<SlicedDetectionBox> detections;
        const bool sliced =
                task == QStringLiteral("detect") && prompt.slicedDetection;
        const QVector<QRect> slices =
                sliced ? detectionSlices(
                                 pixels.size(),
                                 QSize(prompt.sliceWidth, prompt.sliceHeight),
                                 prompt.sliceOverlap)
                       : QVector<QRect>();
        auto collect = [&](const aicore_image_view& input,
                           const QPoint& offset) -> bool {
            const int status =
                    world ? aicore_yolo_detect_image_with_classes(
                                    context, &input, classPointers.data(),
                                    int32_t(classPointers.size()))
                          : aicore_yolo_detect_image(context, &input);
            if (status != 0) {
                result.error =
                        QString::fromUtf8(aicore_yolo_last_error(context));
                return false;
            }
            const int count = aicore_yolo_detection_count(context);
            for (int i = 0; i < count; ++i) {
                const aicore_yolo_detection box =
                        aicore_yolo_detection_at(context, i);
                SlicedDetectionBox item;
                item.bounds = QRectF(box.x1 + offset.x(), box.y1 + offset.y(),
                                     box.x2 - box.x1, box.y2 - box.y1);
                item.score = box.score;
                item.classId = box.class_id;
                item.label =
                        className(aicore_yolo_detection_class_name(context, i),
                                  box.class_id);
                detections.append(std::move(item));
            }
            return true;
        };
        if (!sliced) {
            collect(view, QPoint());
        } else {
            // SAHI runs each tile and performs a full-image pass whenever the
            // image produced more than one tile, preserving large-object
            // recall.
            for (const QRect& tile : slices) {
                const uchar* base = pixels.constBits() +
                                    tile.y() * pixels.bytesPerLine() +
                                    tile.x() * 4;
                const aicore_image_view tileView{
                        base, tile.width(), tile.height(),
                        size_t(pixels.bytesPerLine()), AICORE_IMAGE_RGBA8};
                if (!collect(tileView, tile.topLeft())) break;
            }
            if (result.error.isEmpty() && slices.size() > 1)
                collect(view, QPoint());
            if (result.error.isEmpty())
                detections = mergeSlicedDetections(detections,
                                                   prompt.sliceMergeThreshold);
        }
        if (result.error.isEmpty()) {
            result.shapes.reserve(detections.size());
            for (const SlicedDetectionBox& item : detections) {
                AnnotationShape shape;
                shape.type = QStringLiteral("rectangle");
                shape.label = item.label;
                shape.points = {
                        item.bounds.topLeft(),
                        QPointF(item.bounds.right(), item.bounds.top()),
                        item.bounds.bottomRight(),
                        QPointF(item.bounds.left(), item.bounds.bottom())};
                shape.extra.insert(QStringLiteral("score"), double(item.score));
                result.shapes.append(std::move(shape));
            }
        }
    } else if (task == QStringLiteral("segment") || yoloe) {
        aicore_yolo_segment_result* segment =
                aicore_yolo_seg_image(context, &view);
        if (!segment)
            result.error = QString::fromUtf8(aicore_yolo_last_error(context));
        else {
            for (int i = 0; i < aicore_yolo_seg_det_count(segment); ++i) {
                const aicore_yolo_plane_view mask =
                        aicore_yolo_seg_mask_at(segment, i);
                const QVector<QPointF> polygon = largestMaskPolygon(
                        static_cast<const uint8_t*>(mask.data), mask.width,
                        mask.height, mask.row_stride_bytes);
                if (polygon.size() < 3) continue;
                const aicore_yolo_detection box =
                        aicore_yolo_seg_det_at(segment, i);
                AnnotationShape shape;
                shape.type = QStringLiteral("polygon");
                shape.label =
                        className(aicore_yolo_seg_det_class_name(segment, i),
                                  box.class_id);
                shape.points = polygon;
                shape.mask = instanceMask(
                        static_cast<const uint8_t*>(mask.data), mask.width,
                        mask.height, mask.row_stride_bytes, pixels.size());
                shape.extra.insert(QStringLiteral("score"), double(box.score));
                result.shapes.append(shape);
            }
            aicore_yolo_seg_result_free(segment);
        }
    } else if (task == QStringLiteral("semantic")) {
        aicore_yolo_semantic_result* semantic =
                aicore_yolo_semantic_image(context, &view);
        if (!semantic) {
            result.error = QString::fromUtf8(aicore_yolo_last_error(context));
        } else {
            const aicore_yolo_plane_view map =
                    aicore_yolo_semantic_class_map(semantic);
            const int classCount = aicore_yolo_semantic_num_classes(semantic);
            if (map.width != pixels.width() || map.height != pixels.height() ||
                classCount <= 0 || classCount > 256) {
                result.error =
                        tr("Semantic map does not match the source image");
            } else {
                QStringList names;
                names.reserve(classCount);
                for (int id = 0; id < classCount; ++id)
                    names.append(className(
                            aicore_yolo_semantic_class_name(semantic, id), id));
                SemanticConversion converted = convertSemanticClassMap(
                        static_cast<const uint8_t*>(map.data), map.width,
                        map.height, map.row_stride_bytes, names);
                result.shapes = std::move(converted.shapes);
                result.error = std::move(converted.error);
            }
            aicore_yolo_semantic_result_free(semantic);
        }
    } else if (task == QStringLiteral("pose")) {
        aicore_yolo_pose_result* pose = aicore_yolo_pose_image(context, &view);
        if (!pose)
            result.error = QString::fromUtf8(aicore_yolo_last_error(context));
        else {
            static const char* kCocoKeypoints[] = {
                    "nose",           "left_eye",    "right_eye",
                    "left_ear",       "right_ear",   "left_shoulder",
                    "right_shoulder", "left_elbow",  "right_elbow",
                    "left_wrist",     "right_wrist", "left_hip",
                    "right_hip",      "left_knee",   "right_knee",
                    "left_ankle",     "right_ankle"};
            for (int i = 0; i < aicore_yolo_pose_det_count(pose); ++i) {
                const aicore_yolo_detection box =
                        aicore_yolo_pose_det_at(pose, i);
                AnnotationShape person = rectangle(
                        box, className(aicore_yolo_pose_det_class_name(pose, i),
                                       box.class_id));
                person.extra.insert(QStringLiteral("group_id"), i);
                result.shapes.append(person);
                for (int k = 0; k < aicore_yolo_pose_kpt_count(pose); ++k) {
                    const aicore_yolo_keypoint keypoint =
                            aicore_yolo_pose_kpt_at(pose, i, k);
                    if (keypoint.visibility < 0.5f ||
                        (keypoint.x == 0 && keypoint.y == 0) ||
                        keypoint.x < box.x1 || keypoint.x > box.x2 ||
                        keypoint.y < box.y1 || keypoint.y > box.y2)
                        continue;
                    AnnotationShape point;
                    point.type = QStringLiteral("point");
                    point.label =
                            k < 17 ? QString::fromLatin1(kCocoKeypoints[k])
                                   : tr("keypoint %1").arg(k);
                    point.points = {QPointF(keypoint.x, keypoint.y)};
                    point.extra.insert(QStringLiteral("score"),
                                       double(keypoint.visibility));
                    point.extra.insert(QStringLiteral("group_id"), i);
                    result.shapes.append(point);
                }
            }
            aicore_yolo_pose_result_free(pose);
        }
    } else if (task == QStringLiteral("obb")) {
        aicore_yolo_obb_result* obb = aicore_yolo_obb_image(context, &view);
        if (!obb)
            result.error = QString::fromUtf8(aicore_yolo_last_error(context));
        else {
            for (int i = 0; i < aicore_yolo_obb_count(obb); ++i) {
                const aicore_yolo_obb_box box = aicore_yolo_obb_at(obb, i);
                const double c = std::cos(box.angle), s = std::sin(box.angle);
                AnnotationShape shape;
                shape.type = QStringLiteral("rotation");
                shape.label = className(aicore_yolo_obb_class_name(obb, i),
                                        box.class_id);
                for (const QPointF& p : {QPointF(-box.w / 2, -box.h / 2),
                                         QPointF(box.w / 2, -box.h / 2),
                                         QPointF(box.w / 2, box.h / 2),
                                         QPointF(-box.w / 2, box.h / 2)})
                    shape.points.append(
                            QPointF(box.cx + c * p.x() - s * p.y(),
                                    box.cy + s * p.x() + c * p.y()));
                shape.extra.insert(QStringLiteral("score"), double(box.score));
                result.shapes.append(shape);
            }
            aicore_yolo_obb_result_free(obb);
        }
    } else if (task == QStringLiteral("classify")) {
        aicore_yolo_classify_result* cls =
                aicore_yolo_classify_image(context, &view);
        if (!cls)
            result.error = QString::fromUtf8(aicore_yolo_last_error(context));
        else {
            int best = -1;
            float bestProbability = -1;
            const int count = aicore_yolo_classify_count(cls);
            for (int i = 0; i < count; ++i) {
                const float probability = aicore_yolo_classify_prob_at(cls, i);
                if (probability > bestProbability) {
                    best = i;
                    bestProbability = probability;
                }
            }
            if (best >= 0) {
                for (int i = 0; i < count; ++i)
                    result.flags.insert(
                            className(aicore_yolo_classify_class_name(cls, i),
                                      i),
                            i == best);
                result.hasFlags = true;
                result.summary =
                        tr("%1 (%2%)")
                                .arg(className(aicore_yolo_classify_class_name(
                                                       cls, best),
                                               best))
                                .arg(double(bestProbability) * 100.0, 0, 'f',
                                     1);
            }
            aicore_yolo_classify_result_free(cls);
        }
    } else {
        result.error = tr("Unsupported model task: %1").arg(task);
    }
    const char* resolved = aicore_yolo_context_device(context);
    result.device = resolved ? QString::fromUtf8(resolved) : device;
    if (result.error.isEmpty() && !prompt.objectTaggingModelPath.isEmpty() &&
        !result.shapes.isEmpty() &&
        (task == QStringLiteral("world") || task == QStringLiteral("detect"))) {
        if (!m_objectTagger) {
            auto options = aicore_tagging_default_options();
            const auto path = prompt.objectTaggingModelPath.toUtf8(),
                       requested = device.toUtf8();
            options.model_path = path.constData();
            options.device = requested.constData();
            options.threads = 8;
            m_objectTagger = aicore_tagging_create(&options);
            m_objectTaggerPath = prompt.objectTaggingModelPath;
            m_objectTaggerDevice = device;
        }
        auto* tagger = static_cast<aicore_tagging_ctx*>(m_objectTagger);
        if (!tagger || !aicore_tagging_ready(tagger)) {
            result.error = QString::fromUtf8(aicore_tagging_last_error(tagger));
            if (result.error.isEmpty())
                result.error = tr("Object tagging model is unavailable");
            releaseObjectTagger();
        } else {
            // O(boxes) calls, one transient crop: upstream truncates endpoints
            // to integers and uses half-open NumPy image[y1:y2, x1:x2].
            const QSet<QString> deleted(prompt.ramDeleteTags.cbegin(),
                                        prompt.ramDeleteTags.cend());
            const QSet<QString> retained(prompt.ramFilterTags.cbegin(),
                                         prompt.ramFilterTags.cend());
            for (auto& shape : result.shapes) {
                const QRectF bounds = QPolygonF(shape.points).boundingRect();
                const int x1 = std::clamp(int(bounds.left()), 0, image.width()),
                          y1 = std::clamp(int(bounds.top()), 0, image.height()),
                          x2 = std::clamp(int(bounds.right()), 0,
                                          image.width()),
                          y2 = std::clamp(int(bounds.bottom()), 0,
                                          image.height());
                if (x2 <= x1 || y2 <= y1) {
                    result.error = tr("Detected object has an empty crop");
                    break;
                }
                const QImage crop =
                        image.copy(x1, y1, x2 - x1, y2 - y1)
                                .convertToFormat(QImage::Format_RGB888);
                const aicore_image_view cropView{
                        crop.constBits(), crop.width(), crop.height(),
                        size_t(crop.bytesPerLine()), AICORE_IMAGE_RGB8};
                auto request = aicore_tagging_default_request_v2();
                request.resize_mode = AICORE_TAGGING_RESIZE_OPENCV;
                aicore_tagging_result* tags = nullptr;
                if (aicore_tagging_infer_image_v2(tagger, &cropView, &request,
                                                  &tags) != 0 ||
                    !tags) {
                    result.error = QString::fromUtf8(
                            aicore_tagging_last_error(tagger));
                    if (result.error.isEmpty())
                        result.error = tr("Object tagging failed");
                    aicore_tagging_result_free(tags);
                    break;
                }
                QStringList names;
                for (size_t i = 0; i < tags->tag_count; ++i) {
                    const auto& tag = tags->tags[i];
                    const QString name =
                            QString::fromUtf8(tag.name ? tag.name : "");
                    if (tag.selected &&
                        annotationKeepRamTag(name, deleted, retained))
                        names.append(annotationRamTag(name, prompt.ramTagMode));
                }
                shape.extra.insert(QStringLiteral("description"),
                                   names.join(QStringLiteral(" | ")));
                shape.points = {QPointF(x1, y1), QPointF(x2, y1),
                                QPointF(x2, y2), QPointF(x1, y2)};
                aicore_tagging_result_free(tags);
            }
        }
        if (!result.error.isEmpty()) result.shapes.clear();
    }
    emit finished(result);
}
