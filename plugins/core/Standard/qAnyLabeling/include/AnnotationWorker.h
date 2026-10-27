// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>

#include "AnnotationCanvas.h"

struct AnnotationInference {
    quint64 generation = 0;
    quint64 promptRevision = 0;
    QString imagePath;
    QString modelPath;
    QString task;
    QVector<AnnotationShape> shapes;
    // Source AutoLabelingResult.replace, including an empty successful result.
    bool replaceShapes = false;
    QVector<QRectF> sourceRois;
    QVector<int> sourceIndices;
    QImage depthPreview;
    QJsonObject flags;
    bool hasFlags = false;
    // Image-level tags (upstream AutoLabelingResult.tags): stored in the
    // top-level sidecar "tags" array, never as object shapes.
    QStringList tags;
    bool hasTags = false;
    QString description;
    bool hasDescription = false;
    // RelateAnything detector stage: true when the objects were found by the
    // worker. They are then ordinary rectangle shapes in result.shapes, and
    // each relation's subject_index / object_index is the position in
    // result.shapes, not a canvas index.
    bool relateAddedObjects = false;
    // RelateAnything triplets (subject_index/object_index are canvas shape
    // indices). Display-only: drawn as a transient overlay and listed in the
    // relation table, never written to the sidecar.
    QJsonArray relations;
    QString summary;
    QString error;
    QString device;
};

Q_DECLARE_METATYPE(AnnotationInference)

class AnnotationWorker final : public QObject {
    Q_OBJECT
public:
    ~AnnotationWorker() override;

public slots:
    void infer(const QString& modelPath,
               const QString& metricModelPath,
               const QString& device,
               const QString& task,
               const QString& imagePath,
               const QImage& image,
               quint64 generation,
               double confidence,
               quint64 promptRevision,
               const AnnotationPrompt& prompt);

signals:
    void finished(const AnnotationInference& result);

private:
    enum class ModelKind {
        Yolo,
        RFDetr,
        SAM3,
        RMBG,
        GKD,
        FaceDetect,
        Ocr,
        Depth,
        DeepLSD,
        ALIKED,
        Ram,            // Recognize Anything Model (image-level tagging)
        RelateAnything,  // Scene relation inference (object + relation edges)
        Attribute
    };
    void* m_context = nullptr;
    void* m_landmarkContext = nullptr;
    QString m_modelPath;
    QString m_metricModelPath;
    QString m_ocrLayoutModelPath;
    QString m_ocrClassifierModelPath;
    QString m_device;
    QString m_textModelPath;
    QString m_vocabText;
    QVector<QRectF> m_visualPrompts;
    int m_promptCount = 0;
    ModelKind m_kind = ModelKind::Yolo;
    // RelateAnything's own detector (automatic object stage). Loaded lazily on
    // this thread and reused until the path or device changes.
    void* m_relateDetector = nullptr;
    bool m_relateDetectorRfdetr = false;
    QString m_relateDetectorPath;
    QString m_relateDetectorDevice;
    void releaseRelateDetector();
    void* m_objectTagger = nullptr;
    QString m_objectTaggerPath;
    QString m_objectTaggerDevice;
    void releaseObjectTagger();
};
