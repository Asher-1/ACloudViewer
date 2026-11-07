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
#include <QString>

#include "aicore/runtime_capi.h"
#include "aicore/tagging_capi.h"

struct RAMSettings {
    QString model, device, tagList, thresholds, embedding, specifiedTags;
    int threads = 8, maxLength = 30, beamBatch = 0;
    int cpuPrecision = AICORE_TAGGING_CPU_AUTO;
    bool caption = false;
    float threshold = -1, thresholdEpsilon = 0;
    bool sameSession(const RAMSettings& other) const;
};
struct RAMResult {
    quint64 generation = 0;
    QString source, error, device;
    QImage image;
    QImage annotated;  // image with caption / selected-tag overlay
    QJsonObject json;
    double e2eMs = 0;
};
Q_DECLARE_METATYPE(RAMResult)
/** One QObject on a persistent QThread. All context operations run there. */
class RAMWorker : public QObject {
    Q_OBJECT
public:
    RAMWorker();
    ~RAMWorker() override;
    void requestCancel();  // token operations are thread safe
    void resetCancel();    // called before enqueueing a single job
    void process(const RAMSettings&, QImage, const QString&, quint64);
    void release();
signals:
    void completed(const RAMResult&);

private:
    aicore_tagging_ctx* m_context = nullptr;
    aicore_cancel_token* m_cancel = nullptr;
    RAMSettings m_settings;
};
