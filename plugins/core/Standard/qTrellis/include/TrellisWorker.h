// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
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
    /** Raw AICore payload for voxel stages (T2VOX01), retained so the chip
     *  click can open the 3D voxel viewer; empty for mesh stages. */
    QByteArray rawBlob;
};

Q_DECLARE_METATYPE(TrellisStagePreview)

/** What one worker run should do:
 *  Full — inference + GLB bake (the classic one-click flow);
 *  GeometryOnly — inference only, no bake (the result lands untextured and
 *    the GLB can be baked later from the Export page or Generate + GLB);
 *  BakeOnly — no inference: bake the retained result (see bakeInput) into a
 *    GLB and emit bakeGlbReady (the Export page's re-bake path). */
enum class TrellisRunMode { Full, GeometryOnly, BakeOnly };

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
    /** Final shaded render of the generated mesh (256 px, per-vertex PBR
     *  base colours when textured) for the pipeline-step strip; rendered on
     *  the worker thread. */
    QImage previewImage;
    /** Geometry-only render (256 px, amber shading, no PBR) for the Mesh
     *  chip: the textured render belongs to the Texture/GLB chips, and
     *  filling Mesh with the same image read as a broken stage. Rendered
     *  only for textured runs (an untextured previewImage already is the
     *  geometry render). */
    QImage geometryPreview;
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
        /** Full / GeometryOnly / BakeOnly — see TrellisRunMode. */
        TrellisRunMode runMode = TrellisRunMode::Full;
        /** BakeOnly: the retained result to bake (mesh arrays only; the
         *  inference context is NOT needed by the bake). */
        TrellisRunResult bakeInput;
        int bakeTextureSize = 2048;
        int bakeComponentFilter = 0;
    };

    explicit TrellisWorker(const Settings& settings, QObject* parent = nullptr);
    ~TrellisWorker() override;

    /** Render one AICore preview blob (T2VOX01 / T2MESH01) to a QImage.
     *  Exposed for the unit tests. */
    static QImage renderPreviewBlob(const char* data, int len, int size = 256);

    /** Cooperative bake cancel: checked at AICore bake stage boundaries
     *  (takes effect at the next boundary, not mid-library-call). */
    void cancelBake() { m_bakeCancel = 1; }

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int stage, int step, int total);
    /** BakeOnly: fired at AICore bake stage boundaries (BakeStage
     *  checkpoint / done) with the human-readable stage description and the
     *  seconds elapsed since that stage started. The bake has no global step
     *  count, so consumers show a busy bar + stage text instead of a
     *  percentage. */
    void bakeProgress(const QString& stage, double elapsedS);
    void stagePreview(const TrellisStagePreview& preview);
    void resultReady(const TrellisRunResult& result);
    /** BakeOnly: the finished GLB bytes; routed by the plugin to the export
     *  destination. */
    void bakeGlbReady(const QByteArray& glb);
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
    /** Post-run console digest: end-to-end wall time, per-stage wall times
     *  with one-line explanations, and mesh/quality stats. */
    void emitRunSummary(const TrellisRunResult& result, double glbBakeMs);
    static void bakeProgressTrampoline(const char* stage,
                                       double elapsed_s,
                                       void* user);
    void onBakeProgress(const QString& stage, double elapsedS);
    bool runBakeOnly();
    QElapsedTimer m_stageTimer;
    int m_lastStage = -1;
    QHash<int, double> m_stageMs;
    /** Bake cooperative-cancel flag; written from the GUI thread, polled at
     *  AICore bake stage boundaries (relaxed int flag). */
    volatile int m_bakeCancel = 0;
#endif

    Settings m_settings;
    struct aicore_trellis_ctx* m_ctx = nullptr;
};

/** Print-wrap (CGAL Alpha Wrap) job input: a private snapshot of the last
 *  generation result plus the export-page settings. The snapshot decouples
 *  the job from the dialog (a new generation may land while the wrap runs). */
struct TrellisPrintRequest {
    TrellisRunResult source;  // deep copy of the last generation result
    int componentFilter = 0;  // export-page component filter (0/1/2)
    int textureSize = 2048;   // projected-GLB atlas hint (0 = default)
    bool bakeGlb = false;     // projected PBR GLB (textured sources only)
};

/** Print-wrap job output: watertight wrap mesh plus the optional projected
 *  PBR GLB (wrap geometry as target, dense source for texture projection). */
struct TrellisPrintResult {
    QVector<float> verts;    // 3 * nVerts (centered unit cube)
    QVector<float> normals;  // 3 * nVerts
    QVector<float> pbr;      // 6 * nVerts projected per-vertex preview
    QVector<int> tris;       // 3 * nTris
    bool hasPbr = false;
    QByteArray glb;  // projected PBR GLB (empty when skipped/failed)
    double wrapMs = 0.0;
    double bakeMs = 0.0;
    int sourceNVerts = 0;  // input mesh size for the completion log
    int sourceNTris = 0;
};

Q_DECLARE_METATYPE(TrellisPrintResult)

/** Background CGAL Alpha-Wrap print remeshing of the last generation result.
 *  Pure CPU geometry work: no model context and no AICore device lock, so it
 *  may run alongside an active generation (CPU contention only). */
class TrellisPrintWorker : public QThread {
    Q_OBJECT

public:
    explicit TrellisPrintWorker(const TrellisPrintRequest& request,
                                QObject* parent = nullptr);
    ~TrellisPrintWorker() override;

signals:
    void logMessage(const QString& msg);
    void printResultReady(const TrellisPrintResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
    TrellisPrintRequest m_request;
};
