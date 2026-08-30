// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QElapsedTimer>
#include <QHash>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QVector>

struct aicore_trellis_ctx;
struct aicore_trellis_mesh;

/** One live intermediate-stage preview: a self-describing AICore preview blob
 *  (T2VOX01 voxel set / T2MESH01 mesh keyframe) already rendered to a QImage
 *  on the worker thread. */
struct TrellisStagePreview {
    int stage = -1;  // aicore_trellis_stage
    int step = 0;
    int total = 0;
    QString label;  // human-readable stage + step tag
    QImage image;
};

Q_DECLARE_METATYPE(TrellisStagePreview)

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
    /** UV-atlas textured GLB baked on the worker thread (2048, keep-tiny
     *  filter) right after the mesh lands; empty when the bake failed. The
     *  add-to-DB path imports it for the full PBR material display and the
     *  GLB export reuses the same bytes instead of re-baking on the GUI
     *  thread. */
    QByteArray glb;
    /** Seven-channel decoded dual grid carried through for standalone
     *  re-texturing (aicore_trellis_texture_mesh). Empty on the coarse
     *  pipeline. */
    QVector<float> gridFeats;  // 7 * nvox
    QVector<int> gridCoords;   // 3 * nvox
    int gridRes = 0;
    /** AI background-removal result (full-resolution RGBA), null when no
     *  RMBG model ran (e.g. solid-color fallback). */
    QImage rmbgImage;
    double totalRuntimeMs = 0.0;
    /** Per-stage wall times (key: aicore_trellis_stage, value: ms). */
    QHash<int, double> stageMs;
    QString backend;
    QString modelPath;
    QString quantization;  // "q8" | "f16"
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
        /** Weight-precision chain (only affects which GGUFs the dialog
         *  resolved; kept here for the result metadata). */
        QString quantization = QStringLiteral("q8");
        /** Live per-step previews (T2VOX01 voxel sets + mesh keyframes) via
         *  aicore_trellis_generate_ex. */
        bool livePreview = true;
    };

    explicit TrellisWorker(const Settings& settings, QObject* parent = nullptr);
    ~TrellisWorker() override;

    /** Render one AICore preview blob (T2VOX01 / T2MESH01) to a QImage.
     *  Exposed for the unit tests. */
    static QImage renderPreviewBlob(const char* data, int len, int size = 256);

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int stage, int step, int total);
    void stagePreview(const TrellisStagePreview& preview);
    void resultReady(const TrellisRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool resolveRmbgModel();
    void applySettingsToOptions(struct aicore_trellis_options* opts);
    void emitPreviewBlob(
            int stage, int step, int total, const void* data, int len);
    /** Progress trampoline: forwards to progressUpdate and accumulates the
     *  per-stage wall times (the C callback is capture-less, so the state
     *  lives on the worker). */
    void onProgress(int stage, int step, int total);
    QElapsedTimer m_stageTimer;
    int m_lastStage = -1;
    QHash<int, double> m_stageMs;
#endif

    Settings m_settings;
    struct aicore_trellis_ctx* m_ctx = nullptr;
};
