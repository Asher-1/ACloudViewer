// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMetaType>
#include <QThread>
#include <QVector>

#include "DA3Dialog.h"

struct aicore_depth_ctx;

struct DA3DepthResult {
    QString sourceName;
    int width = 0;
    int height = 0;
    QVector<float> depth;
    QVector<float> confidence;
    bool hasPose = false;
    float extrinsics[12] = {};
    float intrinsics[9] = {};
};
Q_DECLARE_METATYPE(DA3DepthResult)

struct DA3ReconResult {
    QString sourceName;
    int count = 0;
    QVector<float> positions;
    QVector<float> colors;
    QVector<float> scales;
    QVector<float> opacities;
};
Q_DECLARE_METATYPE(DA3ReconResult)

class DA3Worker : public QThread {
    Q_OBJECT

public:
    explicit DA3Worker(const DA3Dialog::Settings& settings,
                       QObject* parent = nullptr);

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void depthResultReady(const DA3DepthResult& result);
    void reconResultReady(const DA3ReconResult& result);
    void modelInfoReady(const QString& info);
    void taskFinished(bool success);

protected:
    void run() override;

private:
    bool runDepthSingle();
    bool runDepthPose();
    bool runDepthMultiView();
    bool runReconstruct();
    bool runExportGLB();
    bool runExportCOLMAP();
    bool runQuantize();
    bool runModelInfo();

    struct CtxGuard {
        aicore_depth_ctx* ctx = nullptr;
        ~CtxGuard();
        explicit operator bool() const { return ctx != nullptr; }
    };
    CtxGuard loadModel();

    DA3Dialog::Settings m_settings;
};
