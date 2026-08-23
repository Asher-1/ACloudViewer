// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QVector>

struct aicore_trellis_ctx;
struct aicore_trellis_mesh;

/** Result envelope of one TRELLIS.2 image-to-3D generation. Buffers are
 *  copied out of the AICore mesh handle before it is freed. */
struct TrellisRunResult {
    QString sourceImage;
    QString presetName;
    /** Centered unit-cube mesh ([-0.5, 0.5]^3). */
    QVector<float> verts;    // 3 * nVerts
    QVector<float> normals;  // 3 * nVerts
    QVector<int> tris;       // 3 * nTris
    /** Per-vertex PBR (6 * nVerts: base_color rgb, metallic, roughness,
     *  alpha); empty when untextured. */
    QVector<float> pbr;
    bool hasPbr = false;
    /** AI background-removal result (full-resolution RGBA), null when no
     *  RMBG model ran (e.g. solid-color fallback). */
    QImage rmbgImage;
    double totalRuntimeMs = 0.0;
    QString backend;
    QString modelPath;
};

Q_DECLARE_METATYPE(TrellisRunResult)

/** Background TRELLIS.2 generation worker. The pipeline context is created
 *  inside run() on the worker thread and released there too. */
class TrellisWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QStringList modelPaths;  // resolved absolute GGUF paths
        QString inputPath;
        QString presetName;
        int pipelineType = 0;    // aicore_trellis_pipeline_type (auto=0)
        int backgroundMode = 0;  // aicore_trellis_background_mode (auto=0)
        int steps = 0;           // <=0 -> pipeline default (12)
        double guidance = -1.0;  // <0 -> pipeline default (7.5)
        int textureSteps = 0;    // <=0 -> pipeline default (12)
        uint64_t seed = 0;
        int threads = 0;
        QString device = QStringLiteral("auto");
        QString shapeDecPlacement = QStringLiteral("auto");
        bool useRmbg = false;
        QString rmbgModelPath;  // used when useRmbg
        bool textureEnabled = true;
    };

    explicit TrellisWorker(const Settings& settings, QObject* parent = nullptr);
    ~TrellisWorker() override;

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int stage, int step, int total);
    void resultReady(const TrellisRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool resolveRmbgModel();
    void applySettingsToOptions(struct aicore_trellis_options* opts);
#endif

    Settings m_settings;
    struct aicore_trellis_ctx* m_ctx = nullptr;
};
