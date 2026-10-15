// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QHash>
#include <QImage>
#include <QJsonObject>
#include <QObject>
#include <QSize>
#include <QString>
#include <QStringList>
#include <memory>
#include <vector>

#include "aicore/relation_capi.h"
#include "aicore/runtime_capi.h"
#include "RelateAnythingTemporal.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/yolo_capi.h"
struct RelateAnythingSettings {
    QString model, detector, device, student, tokenizer, predicates;
    QString regions, masks, vocabulary, vocabularyNames, predicateTypes;
    // Open-vocabulary detectors (YOLO-World / non-prompt-free YOLOE): the class
    // list and the matching text-encoder GGUF.
    QStringList detectorClasses;
    QString detectorTextModel;
    bool detectorIsRfdetr = false;
    // Defaults follow upstream: detector conf 0.25, NMS IoU 0.5, 32 boxes to
    // the relation head (deploy/demo_webcam.py); relation floors 0.45
    // (interaction) / 0.42 (spatial), at most 8 relations of which 3 spatial
    // (deploy/render_video.py). topk is the model's pair budget (K=128).
    int threads = 8, maxBoxes = 24, topk = 128, maxRelations = 8,
        maxSpatial = 3;
    float threshold = .20f, detIou = .5f, relationThreshold = .45f,
          spatialThreshold = .42f;
    // Drop detections covering more than this fraction of the image (upstream
    // --max_box_frac 0.85): a relation to "the whole picture" is noise.
    float maxBoxFraction = .85f;
    int strict = -1;
    bool decompose = true;
    // Video / camera frames: stabilise with the ported upstream tracker and
    // relation-belief filter (render_video.py). Single images and batches are
    // independent and never use it. Temporal mode always decomposes.
    bool temporal = false;
    // Rank relations by score * conf(subject) * conf(object), the SGDet
    // convention (upstream box_score_weight, default on). Ranking only: the
    // reported score and the floors stay in relation-confidence units.
    bool boxScoreWeight = true;
    RelateTemporal::Params tracking;
    bool sameSession(const RelateAnythingSettings&) const;
};
struct RelateAnythingResult {
    quint64 generation = 0;
    QString source, error, device;
    QImage image, annotated;
    QJsonObject json;
    double e2eMs = 0;
};
Q_DECLARE_METATYPE(RelateAnythingResult)
class RelateAnythingWorker : public QObject {
    Q_OBJECT
public:
    RelateAnythingWorker();
    ~RelateAnythingWorker() override;
    void requestCancel();
    void resetCancel();
    void process(const RelateAnythingSettings&,
                 QImage,
                 const QString&,
                 quint64);
    void release();
    // Forget tracks and relation beliefs (new video, seek, stop, source change).
    void resetTemporal();
signals:
    void completed(const RelateAnythingResult&);

private:
    aicore_relation_ctx* m_context = nullptr;
    aicore_yolo_ctx* m_detector = nullptr;
    aicore_rfdetr_ctx* m_rfdetr = nullptr;
    // Tracker + relation beliefs of the current video (opaque; see .cpp).
    std::shared_ptr<void> m_temporal;
    aicore_cancel_token* m_cancel = nullptr;
    RelateAnythingSettings m_settings;
    std::vector<float> m_regionBoxes, m_regionScores;
    QStringList m_regionLabels;
    QByteArray m_regionMasks;
    QSize m_regionSize;
    QHash<QString, QPair<qint64, qint64>> m_sessionFiles;
};
