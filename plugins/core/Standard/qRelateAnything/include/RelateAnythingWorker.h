// SPDX-License-Identifier: MIT
#pragma once
#include <QImage>
#include <QJsonObject>
#include <QObject>
#include <QString>

#include "aicore/relation_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
struct RelateAnythingSettings {
    QString model, detector, device, student, tokenizer, predicates;
    int threads = 8, maxBoxes = 60, topk = 128;
    float threshold = .25f;
    bool strict = false, decompose = false;
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
signals:
    void completed(const RelateAnythingResult&);

private:
    aicore_relation_ctx* m_context = nullptr;
    aicore_yolo_ctx* m_detector = nullptr;
    aicore_cancel_token* m_cancel = nullptr;
    RelateAnythingSettings m_settings;
};
