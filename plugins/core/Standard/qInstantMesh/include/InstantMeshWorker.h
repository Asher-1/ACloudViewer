// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <aicore/pipeline_timing.h>

#include <QByteArray>
#include <QElapsedTimer>
#include <QHash>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QVector>

struct aicore_cancel_token;

/** Result envelope of one InstantMesh image-to-3D generation. Buffers are
 *  copied out of the AICore mesh handle before it is freed. */
struct InstantMeshRunResult {
    QString sourceImage;
    uint64_t generation = 0;
    QString presetName;
    /** Mesh in the FlexiCubes world frame ([-grid_scale/2, grid_scale/2]^3). */
    QVector<float> verts;  // 3 * nVerts
    QVector<int> tris;     // 3 * nTris
    /** Per-vertex RGB in [0,1] (3 * nVerts); always present. */
    QVector<float> vcolors;
    /** xatlas per-corner UVs (3 * nTris * 2) + baked RGBA atlas; empty when
     *  the texture bake was disabled or failed. */
    QVector<float> uvs;
    QVector<uint8_t> texRgba;
    int texW = 0;
    int texH = 0;
    /** Worker-generated GLB export bytes; DB consumes typed mesh/UV/texture
     * directly, without GLB import. Empty on export failure. */
    QByteArray glb;
    /** Final shaded render of the generated mesh (256 px) for the pipeline
     *  strip; rendered on the worker thread. */
    QImage previewImage;
    double totalRuntimeMs = 0.0;
    double loadRuntimeMs = 0.0;
    double exportRuntimeMs = 0.0;
    aicore_pipeline_timings timings{};
    /** Per-stage wall times (key: aicore_instantmesh_stage, value: ms). */
    QHash<int, double> stageMs;
    QString backend;
    QString quantization;  // "f16" | "q8" | "f32"
    bool multiview = false;
    /** Plugin-side destination flag carried through the envelope (the
     *  worker has no DB access). */
    bool addResultToDb = true;
};

Q_DECLARE_METATYPE(InstantMeshRunResult)

/** Background InstantMesh generation worker. The pipeline context is created
 *  inside run() on the worker thread and released there too. */
class InstantMeshWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QStringList modelPaths;  // resolved absolute GGUF paths (paths-field
                                 // order: cond, unet, vae, dino, transformer,
                                 // synthesizer)
        QString inputPath;
        uint64_t generation = 0;
        QString presetName;
        int steps = 0;  // <=0 -> pipeline default (75)
        uint64_t seed = 42;
        double guidance = -1.0;   // <=0 -> pipeline default (4.0)
        int gridRes = 0;          // <=0 -> pipeline default (64)
        double gridScale = -1.0;  // <=0 -> pipeline default (2.1)
        int textureRes = 0;       // <=0 -> pipeline default (2048)
        bool bakeTexture = true;
        int threads = 0;
        QString device = QStringLiteral("auto");
        bool useRmbg = false;
        QString rmbgModelPath;  // used when useRmbg
        QString quantization = QStringLiteral("f16");
        bool addResultToDb = true;
    };

    explicit InstantMeshWorker(const Settings& settings,
                               QObject* parent = nullptr);
    ~InstantMeshWorker() override;
#ifdef AICore_ENABLED
    /** Plugin CPU preview; no model inference or GUI/OpenGL access. */
    static QImage renderResultPreview(const InstantMeshRunResult& result,
                                      int size);
#endif

public slots:
    void cancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int stage, int step, int total);
    void resultReady(const InstantMeshRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool resolveRmbgModel();
    void onProgress(int stage, int step, int total);
    void emitRunSummary(const InstantMeshRunResult& result, double glbBakeMs);
    QElapsedTimer m_stageTimer;
    int m_lastStage = -1;
    QHash<int, double> m_stageMs;
#endif

    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
};
