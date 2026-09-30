// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
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
    QVector<QRectF> sourceRois;
    QVector<int> sourceIndices;
    QImage depthPreview;
    QJsonObject flags;
    bool hasFlags = false;
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
        ALIKED
    };
    void* m_context = nullptr;
    void* m_landmarkContext = nullptr;
    QString m_modelPath;
    QString m_metricModelPath;
    QString m_device;
    QString m_textModelPath;
    QString m_vocabText;
    QVector<QRectF> m_visualPrompts;
    int m_promptCount = 0;
    ModelKind m_kind = ModelKind::Yolo;
};
