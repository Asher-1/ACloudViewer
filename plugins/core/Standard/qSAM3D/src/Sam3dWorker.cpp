// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Sam3dWorker.h"

#include <QColor>
#include <QCoreApplication>
#include <QDataStream>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QPainter>
#include <QStandardPaths>
#include <algorithm>

#include "Sam3dVertexColors.h"
#include "aicore/image_view.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/sam3d_capi.h"
#include "aicore/trellis_capi.h"
#include "ecvAICoreRuntimeHelpers.h"

namespace {

const char* stageName(int stage) {
    switch (stage) {
        case AICORE_SAM3D_STAGE_LOAD:
            return "load";
        case AICORE_SAM3D_STAGE_POINTMAP:
            return "moge pointmap";
        case AICORE_SAM3D_STAGE_CONDITION:
            return "condition";
        case AICORE_SAM3D_STAGE_SS_FLOW:
            return "ss flow";
        case AICORE_SAM3D_STAGE_SS_DECODE:
            return "ss decode";
        case AICORE_SAM3D_STAGE_SLAT_FLOW:
            return "slat flow";
        case AICORE_SAM3D_STAGE_GS_DECODE:
            return "gaussian decode";
        case AICORE_SAM3D_STAGE_MESH_DECODE:
            return "mesh-decode";
        case AICORE_SAM3D_STAGE_PBR:
            return "pbr";
        default:
            return "stage";
    }
}

// Static trampoline for the C progress callback (no captures allowed).
void sam3dProgressTrampoline(void* user, int stage, int step, int total) {
    auto* worker = static_cast<Sam3dWorker*>(user);
    emit worker->stageChanged(stage, step, total);
    emit worker->progressChanged(
            worker->progressPercentFor(stage, step, total));
    emit worker->logMessage(QStringLiteral("[SAM3D] stage: %1")
                                    .arg(QLatin1String(stageName(stage))));
}

}  // namespace

int Sam3dWorker::progressPercentFor(int stage, int step, int total) const {
    // Per-stage spans of one object's 0..100 range, in pipeline order:
    // load / pointmap / condition / ss flow / ss decode / slat flow /
    // gs decode / mesh decode / bake. The two diffusion flows interpolate
    // inside their span by step/total (stage 8 = the GLB bake, driven by
    // the worker itself).
    static constexpr int kSpan[9] = {8, 22, 5, 25, 3, 22, 5, 3, 7};
    const int idx = stage < 0 ? 0 : (stage > 8 ? 8 : stage);
    int start = 0;
    for (int i = 0; i < idx; ++i) start += kSpan[i];
    int end = start + kSpan[idx];
    if ((stage == AICORE_SAM3D_STAGE_SS_FLOW ||
         stage == AICORE_SAM3D_STAGE_SLAT_FLOW) &&
        total > 0) {
        const int s = step < 0 ? 0 : (step > total ? total : step);
        start += (end - start) * s / total;
    }
    const int base = m_objBase.load(std::memory_order_relaxed);
    const int span = m_objSpan.load(std::memory_order_relaxed);
    return base + span * start / 100;
}

Sam3dWorker::Sam3dWorker(QObject* parent) : QThread(parent) {}

void Sam3dWorker::configure(const Sam3dDialog::Settings& settings) {
    m_settings = settings;
}

Sam3dWorker::~Sam3dWorker() {
    m_cancel = true;
    if (!isRunning()) return;
    wait(15000);
}

void Sam3dWorker::requestCancel() { m_cancel = true; }

bool Sam3dWorker::decodeInputImage() {
    // Normalize once to the non-premultiplied Format_ARGB32: on
    // little-endian systems its memory order is BGRA8, which the image-view
    // contract accepts natively, so the storage maps straight into the view
    // (rows stay QImage-owned; the real bytesPerLine is the stride).
    QImage image(m_settings.imagePath);
    if (image.isNull()) {
        m_error = QStringLiteral("cannot decode image '%1'")
                          .arg(m_settings.imagePath);
        return false;
    }
    m_sourceImage = image.format() == QImage::Format_ARGB32
                            ? image
                            : image.convertToFormat(QImage::Format_ARGB32);
    return true;
}

bool Sam3dWorker::resolveSam3dModels() {
    // The model cache follows the shared AICore data-root layout; plugins
    // never invent their own download state machines.
    const char* cache = aicore_sam3d_model_cache_dir();
    const QDir dir(cache ? QString::fromUtf8(cache) : QString());
    if (!dir.exists()) {
        m_error = QStringLiteral(
                "SAM 3D model cache not found — download the GGUF models "
                "first (see the plugin README).");
        return false;
    }
    static const char* kQuant[] = {"f16", "q8_0", "q4_k"};
    const QString quant = QString::fromLatin1(kQuant[m_settings.dtypeIndex]);
    static const char* kStages[] = {"ss_generator", "ss_decoder",
                                    "slat_generator", "slat_decoder_gs"};
    for (const char* stage : kStages) {
        const QString name = QStringLiteral("%1-%2.gguf").arg(stage, quant);
        if (!dir.exists(name)) {
            m_error = QStringLiteral("missing model '%1' in %2")
                              .arg(name, dir.absolutePath());
            return false;
        }
    }
    if (m_settings.outputTexturedMesh &&
        !dir.exists(QStringLiteral("slat_decoder_mesh-%1.gguf").arg(quant))) {
        m_error = QStringLiteral(
                          "missing model 'slat_decoder_mesh-%1.gguf' in %2")
                          .arg(quant, dir.absolutePath());
        return false;
    }
    if (!dir.exists(QStringLiteral("moge_vitl-f16.gguf"))) {
        m_error = QStringLiteral("missing model 'moge_vitl-f16.gguf' in %1")
                          .arg(dir.absolutePath());
        return false;
    }
    m_modelsDir = dir.absolutePath();
    return true;
}

bool Sam3dWorker::applyRmbgMask() {
    if (!m_settings.useRmbg) return true;
    // RMBG is a shared AICore task: its physical cache belongs to the rmbg
    // task (download via the qRMBG plugin). Sam3d only borrows the model.
    char* rmbg_cache = aicore_rmbg_model_cache_dir();
    const QDir rmbgDir(rmbg_cache ? QString::fromUtf8(rmbg_cache) : QString());
    if (rmbg_cache) aicore_rmbg_free_buffer(rmbg_cache);

    QString modelPath;
    // The dialog's RMBG quantization combobox pins the exact file; no silent
    // fallback to the other quantization (an implicit numerics switch).
    static const char* kRmbg[] = {"rmbg_q8.gguf", "rmbg_f16.gguf"};
    const QString candidate =
            rmbgDir.filePath(QLatin1String(kRmbg[m_settings.rmbgDtypeIndex]));
    if (QFileInfo::exists(candidate)) modelPath = candidate;
    if (modelPath.isEmpty()) {
        emit logMessage(
                QStringLiteral(
                        "[SAM3D] RMBG model not found under %1 — using the "
                        "image as-is. Download a model with the qRMBG plugin "
                        "or provide an image with an alpha/mask channel.")
                        .arg(rmbgDir.absolutePath()));
        return true;
    }
    emit logMessage(QStringLiteral("[SAM3D] Removing background with %1")
                            .arg(QFileInfo(modelPath).fileName()));

    m_rmbgModel = modelPath.toUtf8();
    aicore_rmbg_options* rmbg_opts = aicore_rmbg_options_new();
    aicore_rmbg_options_set_device(rmbg_opts,
                                   m_settings.device.toUtf8().constData());
    aicore_rmbg_options_set_threads(rmbg_opts, 8);
    aicore_rmbg_ctx* rmbg =
            aicore_rmbg_load_opts(m_rmbgModel.constData(), rmbg_opts);
    aicore_rmbg_options_free(rmbg_opts);
    if (rmbg == nullptr) {
        emit logMessage(
                QStringLiteral("[SAM3D] RMBG load failed — using the "
                               "image as-is."));
        return true;
    }

    aicore_image_view view{};
    view.data = const_cast<uint8_t*>(m_sourceImage.constBits());
    view.width = m_sourceImage.width();
    view.height = m_sourceImage.height();
    view.row_stride_bytes = static_cast<size_t>(m_sourceImage.bytesPerLine());
    view.format = AICORE_IMAGE_BGRA8;
    uint8_t* alpha = nullptr;
    int32_t alpha_w = 0;
    int32_t alpha_h = 0;
    const int rc = aicore_rmbg_alpha_mat_image_view(rmbg, &view, &alpha,
                                                    &alpha_w, &alpha_h);
    aicore_rmbg_free(rmbg);
    if (rc != 0 || alpha == nullptr) {
        emit logMessage(
                QStringLiteral("[SAM3D] RMBG matting failed — using the "
                               "image as-is."));
        return true;
    }
    if (alpha_w != m_sourceImage.width() || alpha_h != m_sourceImage.height()) {
        aicore_rmbg_free_buffer(alpha);
        emit logMessage(
                QStringLiteral("[SAM3D] RMBG size mismatch — using the "
                               "image as-is."));
        return true;
    }
    // Official binary-mask semantics: alpha > 0 keeps the pixel. Both
    // Format_ARGB32 and the matting output keep alpha in byte 3, so write
    // the mask straight onto the ARGB32 scanlines.
    for (int y = 0; y < m_sourceImage.height(); ++y) {
        uint8_t* row = m_sourceImage.scanLine(y);
        const uint8_t* maskRow = alpha + static_cast<size_t>(y) * alpha_w;
        for (int x = 0; x < m_sourceImage.width(); ++x) {
            row[static_cast<size_t>(x) * 4 + 3] = maskRow[x] == 0 ? 0 : 255;
        }
    }
    aicore_rmbg_free_buffer(alpha);
    return true;
}

bool Sam3dWorker::runGeneration() {
    static const char* kQuant[] = {"f16", "q8_0", "q4_k"};
    const QByteArray modelsDirUtf8 = m_modelsDir.toUtf8();
    // The PLY file export is opt-in; the default result path is the
    // in-memory artifact sink (no file IO inside the pipeline). The official
    // PBR assembly reads the Gaussian PLY back, so it always needs a PLY
    // path — a TempLocation scratch file stands in when the user did not
    // opt into the file export.
    const QString plyName =
            QStringLiteral("%1_sam3d.ply")
                    .arg(QFileInfo(m_settings.imagePath).completeBaseName());
    QString plyPath = m_settings.exportPly
                              ? QDir(m_settings.outputDir).filePath(plyName)
                              : QString();
    QString plyScratch;
    if (m_settings.pbrBake && plyPath.isEmpty()) {
        plyScratch = QDir(QStandardPaths::writableLocation(
                                  QStandardPaths::TempLocation))
                             .filePath(QStringLiteral("SAM3D_pbr_%1.ply")
                                               .arg(QCoreApplication::
                                                            applicationPid()));
        plyPath = plyScratch;
    }
    const QByteArray plyUtf8 = plyPath.toUtf8();

    aicore_sam3d_options* opts = aicore_sam3d_options_new();
    aicore_sam3d_options_set_models_dir(opts, modelsDirUtf8.constData());
    aicore_sam3d_options_set_dtype(
            opts, m_settings.dtypeIndex == 0   ? AICORE_SAM3D_DTYPE_F16
                  : m_settings.dtypeIndex == 1 ? AICORE_SAM3D_DTYPE_Q8_0
                                               : AICORE_SAM3D_DTYPE_Q4_K);
    aicore_sam3d_options_set_device(opts,
                                    m_settings.device.toUtf8().constData());
    aicore_sam3d_options_set_threads(opts, 8);
    aicore_sam3d_options_set_seed(opts, m_settings.seed);
    aicore_sam3d_options_set_steps(opts, m_settings.steps, m_settings.steps);
    // Bit-for-bit parity with the acceptance probe / upstream reference:
    // the MoGe point-map cache round-trip drifts the diffusion input.
    aicore_sam3d_options_set_disable_moge_cache(opts, 1);
    if (m_settings.pbrBake) {
        aicore_sam3d_options_set_pbr(opts, 1, m_settings.pbrTextureSize,
                                     m_settings.pbrBakeSteps);
    }
    char err[512] = {0};
    aicore_sam3d_ctx* ctx = aicore_sam3d_load_opts(opts, err, sizeof(err));
    aicore_sam3d_options_free(opts);
    if (ctx == nullptr) {
        m_error = QStringLiteral("AICore sam3d load failed: %1")
                          .arg(QString::fromUtf8(err));
        return false;
    }
    emit logMessage(QStringLiteral("[SAM3D] Backend resolved: %1")
                            .arg(QString::fromUtf8(aicore_sam3d_backend(ctx))));

    aicore_image_view view{};
    view.data = const_cast<uint8_t*>(m_sourceImage.constBits());
    view.width = m_sourceImage.width();
    view.height = m_sourceImage.height();
    view.row_stride_bytes = static_cast<size_t>(m_sourceImage.bytesPerLine());
    view.format = AICORE_IMAGE_BGRA8;

    aicore_sam3d_result* sam3d = aicore_sam3d_generate(
            ctx, &view, nullptr,
            plyPath.isEmpty() ? nullptr : plyUtf8.constData(),
            m_settings.outputTexturedMesh ? 1 : 0, sam3dProgressTrampoline,
            this);

    if (sam3d == nullptr) {
        m_error = QStringLiteral("AICore sam3d generation failed: %1")
                          .arg(QString::fromUtf8(
                                  aicore_sam3d_last_error(ctx) != nullptr
                                          ? aicore_sam3d_last_error(ctx)
                                          : "?"));
        aicore_sam3d_free(ctx);
        return false;
    }

    Sam3dRunResult result;
    result.sourceImage = m_settings.imagePath;
    result.backend = QString::fromUtf8(aicore_sam3d_backend(ctx));
    result.dtype = QString::fromLatin1(kQuant[m_settings.dtypeIndex]);
    // The scratch PLY is deleted right below — only a user-requested file
    // export is reported as a result path.
    result.plyPath = m_settings.exportPly ? plyPath : QString();
    fillRunResult(ctx, sam3d, result);
    aicore_sam3d_result_free(sam3d);
    aicore_sam3d_free(ctx);
    if (!plyScratch.isEmpty()) {
        QFile::remove(plyScratch);
    }

    if (m_settings.outputTexturedMesh && !m_settings.pbrBake &&
        result.meshTriangleCount > 0) {
        bakeTexturedGlb(result);
    }
    if (m_settings.pbrBake && result.hasPbr && !result.glb.isEmpty()) {
        writeGlbFile(result);
        emit logMessage(
                QStringLiteral("[SAM3D] Official PBR GLB ready "
                               "(official 2500-step bake)."));
    }

    emit resultReady(result);
    return true;
}

void Sam3dWorker::fillRunResult(aicore_sam3d_ctx* ctx,
                                aicore_sam3d_result* sam3d,
                                Sam3dRunResult& result) {
    result.gaussianCount =
            static_cast<int>(aicore_sam3d_result_gaussian_count(sam3d));
    // Official PBR GLB (PBR-enabled requests): the textured-GLB producer for
    // this run — it rides the same result.glb field the projection bake uses.
    if (aicore_sam3d_result_has_pbr(sam3d)) {
        size_t glb_len = 0;
        const uint8_t* glb = aicore_sam3d_result_pbr_glb(sam3d, &glb_len);
        if (glb && glb_len > 0) {
            result.hasPbr = true;
            result.glb = QByteArray(reinterpret_cast<const char*>(glb),
                                    static_cast<int>(glb_len));
        }
    }
    result.meshVertexCount = aicore_sam3d_result_mesh_vertex_count(sam3d);
    result.meshTriangleCount = aicore_sam3d_result_mesh_triangle_count(sam3d);
    // In-memory splat artifacts (world-domain centers + display RGB).
    if (result.gaussianCount > 0) {
        const float* centers = aicore_sam3d_result_splat_centers(sam3d);
        const float* rgb = aicore_sam3d_result_splat_rgb(sam3d);
        if (centers && rgb) {
            const int count = result.gaussianCount * 3;
            result.splatCenters.resize(count);
            std::copy(centers, centers + count, result.splatCenters.data());
            result.splatRgb.resize(count);
            std::copy(rgb, rgb + count, result.splatRgb.data());
        }
    }
    if (aicore_sam3d_result_has_mesh(sam3d)) {
        const float* verts = aicore_sam3d_result_mesh_vertices(sam3d);
        const uint32_t* tris = aicore_sam3d_result_mesh_triangles(sam3d);
        // QVector's iterator-pair constructor requires Qt >= 5.14; copy
        // through resize() so the plugin also builds against Qt 5.12.
        const int vertCount = result.meshVertexCount * 3;
        result.vertices.resize(vertCount);
        std::copy(verts, verts + vertCount, result.vertices.data());
        const int triCount = result.meshTriangleCount * 3;
        result.triangles.resize(triCount);
        std::copy(tris, tris + triCount, result.triangles.data());
        // Official use_color payload (RGB + learned normal): the texture
        // bake's preferred color source when present.
        if (const float* attrs = aicore_sam3d_result_mesh_vertex_attrs(sam3d)) {
            const int attrCount = result.meshVertexCount * 6;
            result.meshVertexAttrs.resize(attrCount);
            std::copy(attrs, attrs + attrCount, result.meshVertexAttrs.data());
        }
    }
    // Scene-composer interchange attributes + pose receipt (scene mode only).
    if (sceneModeRequested()) {
        if (aicore_sam3d_result_has_pose(sam3d)) {
            const float* pose = aicore_sam3d_result_pose(sam3d);
            result.pose.resize(10);
            std::copy(pose, pose + 10, result.pose.data());
        }
        if (result.gaussianCount > 0) {
            const auto copy_attr = [&](QVector<float>& dst, const float* src,
                                       int floats) {
                if (src) {
                    dst.resize(floats);
                    std::copy(src, src + floats, dst.data());
                }
            };
            copy_attr(result.splatSh0, aicore_sam3d_result_splat_sh0(sam3d),
                      result.gaussianCount * 3);
            copy_attr(result.splatLogScale,
                      aicore_sam3d_result_splat_log_scale(sam3d),
                      result.gaussianCount * 3);
            copy_attr(result.splatOpacityLogit,
                      aicore_sam3d_result_splat_opacity_logit(sam3d),
                      result.gaussianCount);
            copy_attr(result.splatRotPly,
                      aicore_sam3d_result_splat_rot_ply(sam3d),
                      result.gaussianCount * 4);
        }
    }
    aicore_pipeline_timings timings{};
    if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0) {
        result.e2eMs = timings.e2e_ms;
    }
}

// Official make_scene position action on a plain mesh (no rotation/scale
// attributes): p' = R^T @ diag(pose_scale) @ p + t. Matches the composer's
// splat placement so per-object GLBs land exactly on their splats.
static void transformMeshByScenePose(QVector<float>& vertices,
                                     const QVector<float>& pose) {
    if (pose.size() != 10) return;
    const float* q = pose.constData();  // wxyz
    const float norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (!(norm2 > 0.0f)) return;
    const float two_s = 2.0f / norm2;
    // Row-major quaternion_to_matrix (pytorch3d convention).
    const float R[3][3] = {{1.0f - two_s * (q[2] * q[2] + q[3] * q[3]),
                            two_s * (q[1] * q[2] - q[3] * q[0]),
                            two_s * (q[1] * q[3] + q[2] * q[0])},
                           {two_s * (q[1] * q[2] + q[3] * q[0]),
                            1.0f - two_s * (q[1] * q[1] + q[3] * q[3]),
                            two_s * (q[2] * q[3] - q[1] * q[0])},
                           {two_s * (q[1] * q[3] - q[2] * q[0]),
                            two_s * (q[2] * q[3] + q[1] * q[0]),
                            1.0f - two_s * (q[1] * q[1] + q[2] * q[2])}};
    const float* s = q + 7;  // pose scale
    const float* t = q + 4;  // translation
    const int nv = vertices.size() / 3;
    for (int i = 0; i < nv; ++i) {
        float* p = vertices.data() + i * 3;
        const float scaled[3] = {s[0] * p[0], s[1] * p[1], s[2] * p[2]};
        for (int c = 0; c < 3; ++c) {
            p[c] = R[0][c] * scaled[0] + R[1][c] * scaled[1] +
                   R[2][c] * scaled[2] + t[c];
        }
    }
}

bool Sam3dWorker::resolveSceneMasks(QVector<QImage>& masks) {
    m_sceneMaskLabels.clear();
    masks.clear();
    m_segEndPercent = 0;  // mask-directory runs own the whole 0..100 axis

    // Fallback: explicit mask directory (official <n>.png naming; numeric
    // order so 2.png sorts before 10.png).
    if (m_settings.conceptPrompts.isEmpty()) {
        QFileInfoList files =
                QDir(m_settings.masksDir)
                        .entryInfoList(QStringList{QStringLiteral("*.png")},
                                       QDir::Files, QDir::Name);
        std::sort(files.begin(), files.end(),
                  [](const QFileInfo& a, const QFileInfo& b) {
                      bool aOk = false;
                      bool bOk = false;
                      const int aIdx = a.baseName().toInt(&aOk);
                      const int bIdx = b.baseName().toInt(&bOk);
                      if (aOk && bOk) return aIdx < bIdx;
                      return a.baseName() < b.baseName();
                  });
        for (const QFileInfo& info : files) {
            QImage mask(info.absoluteFilePath());
            if (mask.isNull()) {
                emit logMessage(QStringLiteral("[SAM3D] skipping unreadable "
                                               "mask '%1'.")
                                        .arg(info.fileName()));
                continue;
            }
            masks.append(mask);
            m_sceneMaskLabels.append(info.baseName());
        }
        if (masks.isEmpty()) {
            m_error = QStringLiteral("no readable PNG mask in %1")
                              .arg(m_settings.masksDir);
        }
        return !masks.isEmpty();
    }

    // Two-step workflow, step 2: reuse the preview-confirmed masks when
    // nothing that influences the segmentation changed (exact fingerprint
    // match). A hit skips the whole SAM3 chain — the model is not loaded.
    if (loadPreviewMaskCache(masks)) {
        m_segEndPercent = m_settings.previewOnly ? 99 : 10;
        emit progressChanged(m_segEndPercent);
        emit sceneObjectTotalChanged(masks.size());
        return true;
    }

    // Concept-prompt path: the AICore sam3 task's text-prompted
    // multi-instance segmentation produces the masks fully in memory. The
    // published model catalog owns the filename resolution (no plugin model
    // table); the physical cache is shared with the qSAM3 plugin.
    // SAM3 model selection is dialog-driven (AICore sam3 catalog,
    // text-prompt capable family; best default = q8_0).
    const QByteArray sam3ModelUtf8 = m_settings.sam3ModelFile.toUtf8();
    const aicore_sam3_model_entry* entry =
            aicore_sam3_model_by_filename(sam3ModelUtf8.constData());
    char* cache = aicore_sam3_model_cache_dir();
    const QDir cacheDir(cache ? QString::fromUtf8(cache) : QString());
    if (cache) aicore_sam3_free_buffer(cache);
    if (entry == nullptr || cacheDir.path().isEmpty()) {
        m_error = QStringLiteral("sam3 model '%1' is not in the AICore catalog")
                          .arg(m_settings.sam3ModelFile);
        return false;
    }
    const QString modelPath = cacheDir.filePath(QLatin1String(entry->filename));
    if (!QFileInfo::exists(modelPath)) {
        m_error = QStringLiteral(
                          "sam3 model '%1' not found in %2 — download it "
                          "with the qSAM3 plugin first.")
                          .arg(QLatin1String(entry->filename),
                               cacheDir.absolutePath());
        return false;
    }

    // Order-preserving dedup: a repeated prompt would re-run an identical
    // detection pass and yield zero new candidates.
    QStringList prompts;
    for (const QString& promptText : m_settings.conceptPrompts) {
        if (!prompts.contains(promptText, Qt::CaseInsensitive)) {
            prompts << promptText;
        }
    }
    if (prompts.size() < m_settings.conceptPrompts.size()) {
        emit logMessage(
                QStringLiteral("[SAM3D] dropped %1 duplicate prompt(s).")
                        .arg(m_settings.conceptPrompts.size() -
                             prompts.size()));
    }

    QElapsedTimer segTimer;
    segTimer.start();
    aicore_sam3_options* sam3_opts = aicore_sam3_options_new();
    aicore_sam3_options_set_device(sam3_opts,
                                   m_settings.device.toUtf8().constData());
    aicore_sam3_options_set_threads(sam3_opts, 8);
    const QByteArray modelUtf8 = modelPath.toUtf8();
    aicore_sam3_ctx* sam3 =
            aicore_sam3_load_opts(modelUtf8.constData(), sam3_opts);
    aicore_sam3_options_free(sam3_opts);
    if (sam3 == nullptr) {
        m_error = QStringLiteral("AICore sam3 load failed: %1")
                          .arg(QString::fromUtf8(
                                  aicore_sam3_last_load_error() != nullptr
                                          ? aicore_sam3_last_load_error()
                                          : "?"));
        return false;
    }
    const QString backendName =
            QString::fromUtf8(aicore_sam3_context_backend_name(sam3));
    emit logMessage(QStringLiteral("[SAM3D] sam3 backend: %1 (loaded in %2 s) "
                                   "— segmenting %3 prompt set(s)...")
                            .arg(backendName)
                            .arg(segTimer.elapsed() / 1000.0, 0, 'f', 1)
                            .arg(prompts.size()));
    // The segmentation owns [segStart, segEnd] of the global progress axis:
    // the whole task in preview mode, the 1..10% head when Generate follows
    // with per-object reconstruction sliced out of [segEnd, 100].
    const int segStart = 1;
    const int segEnd = m_settings.previewOnly ? 95 : 10;
    m_segEndPercent = segEnd;
    const int segSpan = segEnd - segStart;
    emit progressChanged(segStart);

    // Text-prompted segmentation consumes an RGB24 buffer; convert once.
    const QImage rgb = m_sourceImage.convertToFormat(QImage::Format_RGB888);
    const int width = rgb.width();
    const int height = rgb.height();
    const size_t stride = static_cast<size_t>(rgb.bytesPerLine());

    struct Candidate {
        float score;
        aicore_sam3_box box;
        QString label;        // sanitized prompt provenance
        const uint8_t* mask;  // borrowed from its seg result
        size_t mask_stride;
    };
    std::vector<aicore_sam3_seg_result*> results;  // owns the borrowed masks
    std::vector<Candidate> candidates;
    for (int pass = 0; pass < prompts.size(); ++pass) {
        if (m_cancel.load()) break;
        const QString& promptText = prompts.at(pass);
        QElapsedTimer passTimer;
        passTimer.start();
        aicore_sam3_pcs_prompt prompt{};
        const QByteArray textUtf8 = promptText.toUtf8();
        prompt.text = textUtf8.constData();
        // <= 0 keeps the context defaults (an explicit UI override).
        prompt.score_threshold = m_settings.samScoreThreshold;
        prompt.nms_threshold = m_settings.samNmsThreshold;
        aicore_sam3_seg_result* res = aicore_sam3_segment_pcs_rgb(
                sam3, &prompt, rgb.constBits(), width, height, stride);
        if (res == nullptr) {
            emit logMessage(
                    QStringLiteral("[SAM3D] prompt '%1' failed: %2 — "
                                   "skipped.")
                            .arg(promptText,
                                 QString::fromUtf8(
                                         aicore_sam3_last_error(sam3) != nullptr
                                                 ? aicore_sam3_last_error(sam3)
                                                 : "?")));
            continue;
        }
        results.push_back(res);
        const int count = aicore_sam3_seg_det_count(res);
        QString label = promptText.simplified();
        label.replace(QLatin1Char(' '), QLatin1Char('_'));
        for (int i = 0; i < count; ++i) {
            aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(res, i);
            if (mask.data == nullptr) continue;
            candidates.push_back({aicore_sam3_seg_det_score_at(res, i),
                                  aicore_sam3_seg_det_box_at(res, i), label,
                                  static_cast<const uint8_t*>(mask.data),
                                  mask.row_stride_bytes});
        }
        emit logMessage(QStringLiteral("[SAM3D] prompt '%1': %2 instance(s) "
                                       "(%3 ms).")
                                .arg(promptText)
                                .arg(count)
                                .arg(passTimer.elapsed()));
        // Real per-prompt progress: every finished pass advances the axis
        // visibly instead of jumping 1 → 3.
        emit progressChanged(segStart + segSpan * (pass + 1) / prompts.size());
    }
    aicore_sam3_free(sam3);  // masks are owned by the seg results
    emit logMessage(
            QStringLiteral("[SAM3D] segmentation finished in %1 s (%2 passes "
                           "on %3; the image encoder ran once and was reused "
                           "across all prompts).")
                    .arg(segTimer.elapsed() / 1000.0, 0, 'f', 1)
                    .arg(prompts.size())
                    .arg(backendName));

    // Cross-prompt dedup: greedy, score-descending, reject on IoU over the
    // threshold with an already accepted box (different prompts hit the
    // same object). <= 0 keeps the balanced plugin default (0.5).
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.score > b.score;
              });
    const float maxIou = m_settings.samDedupIouThreshold > 0.0f
                                 ? m_settings.samDedupIouThreshold
                                 : 0.5f;
    // A degenerate sliver would waste a full single-object GPU run.
    static constexpr float kMinAreaFraction = 0.0005f;
    const float minArea = std::max(64.0f, kMinAreaFraction * width * height);
    QHash<QString, int> labelUses;

    // Copy the accepted masks out of the seg results, then release them.
    std::vector<const Candidate*> accepted;
    for (const Candidate& c : candidates) {
        // O(1) box pre-filter: a mask can never exceed its box, so a box
        // below the floor rejects the candidate with zero pixel work.
        const float boxArea = std::max(0.0f, c.box.x1 - c.box.x0) *
                              std::max(0.0f, c.box.y1 - c.box.y0);
        if (boxArea < minArea) {
            emit logMessage(QStringLiteral("[SAM3D] skipping a tiny '%1' box "
                                           "(%2 px).")
                                    .arg(c.label)
                                    .arg(boxArea));
            continue;
        }
        bool overlaps = false;
        for (const Candidate* a : accepted) {
            const float ix0 = std::max(c.box.x0, a->box.x0);
            const float iy0 = std::max(c.box.y0, a->box.y0);
            const float ix1 = std::min(c.box.x1, a->box.x1);
            const float iy1 = std::min(c.box.y1, a->box.y1);
            const float inter =
                    std::max(0.0f, ix1 - ix0) * std::max(0.0f, iy1 - iy0);
            const float areaC = (c.box.x1 - c.box.x0) * (c.box.y1 - c.box.y0);
            const float areaA =
                    (a->box.x1 - a->box.x0) * (a->box.y1 - a->box.y0);
            const float unionArea = areaA + areaC - inter;
            if (unionArea > 0.0f && inter / unionArea > maxIou) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) continue;
        // Tiny-mask check inside the box only: the shape lives within its
        // box, so clamped box-local counting replaces a full-image scan per
        // candidate (~30 MP × ~100 candidates was a measurable stall).
        const int bx0 = std::max(0, std::min(width - 1, int(c.box.x0)));
        const int by0 = std::max(0, std::min(height - 1, int(c.box.y0)));
        const int bx1 = std::max(0, std::min(width, int(c.box.x1) + 1));
        const int by1 = std::max(0, std::min(height, int(c.box.y1) + 1));
        size_t pixels = 0;
        for (int y = by0; y < by1; ++y) {
            const uint8_t* row =
                    c.mask + static_cast<size_t>(y) * c.mask_stride;
            for (int x = bx0; x < bx1; ++x) pixels += row[x] != 0 ? 1 : 0;
        }
        if (static_cast<float>(pixels) < minArea) {
            emit logMessage(QStringLiteral("[SAM3D] skipping a tiny '%1' "
                                           "mask (%2 px).")
                                    .arg(c.label)
                                    .arg(pixels));
            continue;
        }
        QImage mask(width, height, QImage::Format_Grayscale8);
        for (int y = 0; y < height; ++y) {
            std::copy_n(c.mask + static_cast<size_t>(y) * c.mask_stride, width,
                        mask.scanLine(y));
        }
        const int use = labelUses[c.label]++;
        masks.append(mask);
        m_sceneMaskLabels.append(QStringLiteral("%1_%2").arg(c.label).arg(use));
        accepted.push_back(&c);
    }
    for (aicore_sam3_seg_result* res : results)
        aicore_sam3_seg_result_free(res);

    if (masks.isEmpty()) {
        m_error = m_cancel.load()
                          ? QStringLiteral("cancelled by user")
                          : QStringLiteral(
                                    "sam3 detected no object for the given "
                                    "prompts — try other prompts or lower "
                                    "the score threshold.");
        return false;
    }
    emit logMessage(QStringLiteral("[SAM3D] %1 unique object mask(s) after "
                                   "dedup.")
                            .arg(masks.size()));
    emit progressChanged(segEnd);
    emit sceneObjectTotalChanged(masks.size());
    // Persist the confirmed set: the next Generate with the same inputs
    // reloads these masks and never re-runs the SAM3 model.
    savePreviewMaskCache(masks);
    return true;
}

// Preview-confirmed mask cache (two-step workflow, step 1 → step 2): the
// accepted masks of the last segmentation persist as ONE stream file plus a
// fingerprint. Generate reloads them verbatim on an exact match, so the
// confirmed result never re-runs the SAM3 model. RLE (not PNG): a binary
// full-resolution mask collapses to a handful of runs per row, making the
// whole scene a few MB that round-trips in milliseconds — 27 full-resolution
// PNG re-encodes would cost seconds of pure compression.
QString previewMaskCachePath() {
    return QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
            .filePath(QStringLiteral("SAM3D_preview_masks.bin"));
}

// Everything that can change the segmentation outcome. Any field drift
// invalidates the cache and falls through to a fresh segmentation.
QString previewMaskFingerprint(const Sam3dDialog::Settings& s) {
    const QFileInfo img(s.imagePath);
    return QStringLiteral("%1|%2|%3|%4|%5|%6|%7|%8")
            .arg(s.imagePath, QString::number(img.size()),
                 QString::number(img.lastModified().toMSecsSinceEpoch()),
                 s.conceptPrompts.join(QLatin1Char('\x1f')),
                 QString::number(s.samScoreThreshold),
                 QString::number(s.samNmsThreshold),
                 QString::number(s.samDedupIouThreshold), s.sam3ModelFile);
}

// One palette color per accepted mask (coprime-stride HSV ring): the preview
// tells the objects apart the same way the per-object DB entities do.
static QColor sceneMaskColor(int index) {
    return QColor::fromHsv((index * 47 + 11) % 360, 200, 255);
}

// Binary shape of one scene mask as single-channel gray: official RGBA masks
// carry the shape in the ALPHA channel (the RGB bytes may be noise), plain
// masks are already gray.
static QImage maskAsGray(const QImage& mask) {
    if (!mask.hasAlphaChannel()) {
        return mask.convertToFormat(QImage::Format_Grayscale8);
    }
    const QImage argb = mask.convertToFormat(QImage::Format_ARGB32);
    QImage gray(argb.size(), QImage::Format_Grayscale8);
    for (int y = 0; y < argb.height(); ++y) {
        const QRgb* src = reinterpret_cast<const QRgb*>(argb.constScanLine(y));
        uchar* dst = gray.scanLine(y);
        for (int x = 0; x < argb.width(); ++x) dst[x] = qAlpha(src[x]);
    }
    return gray;
}

bool Sam3dWorker::loadPreviewMaskCache(QVector<QImage>& masks) {
    if (m_settings.conceptPrompts.isEmpty()) return false;
    QFile file(previewMaskCachePath());
    if (!file.open(QIODevice::ReadOnly)) return false;
    QDataStream in(&file);
    in.setVersion(qtCompatDataStreamVersion());
    QByteArray magic;
    quint32 version = 0;
    QByteArray fingerprint;
    quint32 count = 0;
    in >> magic >> version >> fingerprint >> count;
    if (in.status() != QDataStream::Ok || magic != QByteArray("SAM3DMASK") ||
        version != 1 ||
        fingerprint != previewMaskFingerprint(m_settings).toUtf8() ||
        count == 0 || count > 4096u) {
        return false;
    }
    QVector<QImage> loaded;
    loaded.reserve(int(count));
    for (quint32 i = 0; i < count && in.status() == QDataStream::Ok; ++i) {
        quint32 w = 0;
        quint32 h = 0;
        in >> w >> h;
        if (in.status() != QDataStream::Ok || w == 0 || h == 0 ||
            w != quint32(m_sourceImage.width()) ||
            h != quint32(m_sourceImage.height())) {
            return false;  // stale cache (source image changed)
        }
        QImage mask(int(w), int(h), QImage::Format_Grayscale8);
        if (mask.isNull() || mask.bytesPerLine() != int(w)) return false;
        const quint64 total = quint64(w) * h;
        quint64 done = 0;
        uchar* base = mask.bits();
        while (done < total && in.status() == QDataStream::Ok) {
            quint32 run = 0;
            quint8 value = 0;
            in >> run >> value;
            if (in.status() != QDataStream::Ok || run == 0 ||
                done + run > total) {
                return false;
            }
            memset(base + done, value, run);
            done += run;
        }
        if (in.status() != QDataStream::Ok) return false;
        loaded.append(mask);
    }
    QStringList labels;
    in >> labels;
    if (in.status() != QDataStream::Ok || labels.size() != loaded.size()) {
        return false;
    }
    masks = loaded;
    m_sceneMaskLabels = labels;
    emit logMessage(
            QStringLiteral("[SAM3D] reusing %1 preview-confirmed mask(s) — "
                           "SAM3 segmentation skipped (fingerprint match).")
                    .arg(loaded.size()));
    return true;
}

void Sam3dWorker::savePreviewMaskCache(const QVector<QImage>& masks) {
    if (m_settings.conceptPrompts.isEmpty() || masks.isEmpty()) return;
    QFile file(previewMaskCachePath());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
    QDataStream out(&file);
    out.setVersion(qtCompatDataStreamVersion());
    out << QByteArray("SAM3DMASK") << quint32(1)
        << previewMaskFingerprint(m_settings).toUtf8() << quint32(masks.size());
    for (const QImage& mask : masks) {
        const QImage gray = maskAsGray(mask);
        out << quint32(gray.width()) << quint32(gray.height());
        for (int y = 0; y < gray.height(); ++y) {
            // Row-wise RLE: a binary mask collapses to a handful of runs
            // per row, so the whole scene persists in milliseconds.
            const uchar* row = gray.constScanLine(y);
            int x = 0;
            while (x < gray.width()) {
                const uchar value = row[x];
                int run = 1;
                while (x + run < gray.width() && row[x + run] == value) {
                    ++run;
                }
                out << quint32(run) << quint8(value != 0 ? 255 : 0);
                x += run;
            }
        }
    }
    out << m_sceneMaskLabels;
    emit logMessage(QStringLiteral("[SAM3D] %1 confirmed mask(s) cached — "
                                   "Generate will reuse them without "
                                   "re-segmenting.")
                            .arg(masks.size()));
}

void Sam3dWorker::emitSegmentationPreview(const QVector<QImage>& masks) {
    if (!m_settings.segPreview || masks.isEmpty() || m_sourceImage.isNull()) {
        return;
    }
    QImage preview = m_sourceImage.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&preview);
    // Constant overlay alpha: strong enough to read the shapes, weak enough
    // to keep the underlying image recognizable.
    constexpr int kOverlayAlpha = 110;
    const int fontSize = std::max(10, preview.width() / 60);
    QFont labelFont = painter.font();
    labelFont.setBold(true);
    labelFont.setPixelSize(fontSize);
    painter.setFont(labelFont);
    int labeled = 0;
    for (int i = 0; i < masks.size(); ++i) {
        const QImage gray = maskAsGray(masks.at(i));
        if (gray.isNull() || gray.width() != preview.width() ||
            gray.height() != preview.height()) {
            continue;
        }
        const QColor color = sceneMaskColor(i);
        const QColor fill(color.red(), color.green(), color.blue(),
                          kOverlayAlpha);
        // Half-transparent fill over the mask pixels. The scan also
        // accumulates the shape bbox for the label chip below. Direct
        // scanline compositing (binary masks, no antialiasing) — O(pixels)
        // per object, no QPainter path building.
        long long count = 0;
        qreal minX = qreal(preview.width()), minY = qreal(preview.height());
        for (int y = 0; y < gray.height(); ++y) {
            const uchar* row = gray.constScanLine(y);
            const uchar* up = y > 0 ? gray.constScanLine(y - 1) : nullptr;
            const uchar* down =
                    y + 1 < gray.height() ? gray.constScanLine(y + 1) : nullptr;
            QRgb* dst = reinterpret_cast<QRgb*>(preview.scanLine(y));
            for (int x = 0; x < gray.width(); ++x) {
                if (row[x] == 0) continue;
                ++count;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                const QRgb src = dst[x];
                dst[x] = qRgba((qRed(src) * (255 - kOverlayAlpha) +
                                fill.red() * kOverlayAlpha) /
                                       255,
                               (qGreen(src) * (255 - kOverlayAlpha) +
                                fill.green() * kOverlayAlpha) /
                                       255,
                               (qBlue(src) * (255 - kOverlayAlpha) +
                                fill.blue() * kOverlayAlpha) /
                                       255,
                               qAlpha(src));
                // Opaque 1px outline in the same palette color: the target's
                // extent stays readable where fills overlap (the label chip
                // carries the same color, so chip ↔ target is unambiguous).
                const bool edge = x == 0 || row[x - 1] == 0 ||
                                  x + 1 >= gray.width() || row[x + 1] == 0 ||
                                  up == nullptr || up[x] == 0 ||
                                  down == nullptr || down[x] == 0;
                if (edge) {
                    dst[x] = qRgba(color.red(), color.green(), color.blue(),
                                   qAlpha(src));
                }
            }
        }
        if (count <= 0) continue;
        const QString& label =
                m_sceneMaskLabels.value(i, QStringLiteral("obj_%1").arg(i));
        // Category label chip: an opaque palette-colored box hugging the
        // shape bbox's top-left corner (detection-visualization convention)
        // with high-contrast text — far more readable than free-floating
        // text over the busy overlay.
        const qreal pad = std::max<qreal>(2, fontSize / 5.0);
        QRectF textRect = painter.boundingRect(QRectF(), 0, label);
        textRect.adjust(-pad, -pad / 2, pad, pad / 2);
        qreal chipX = minX;
        qreal chipY = minY - textRect.height();
        if (chipY < 0) chipY = minY;  // shape touches the top edge
        if (chipX + textRect.width() > preview.width()) {
            chipX = qreal(preview.width()) - textRect.width();
        }
        chipX = std::clamp(chipX, 0.0, qreal(preview.width()));
        chipY = std::clamp(chipY, 0.0,
                           qreal(preview.height()) - textRect.height());
        painter.fillRect(QRectF(QPointF(chipX, chipY), textRect.size()), color);
        // The palette is high-value (V=255); dark text reads on it.
        painter.setPen(Qt::black);
        painter.drawText(QRectF(QPointF(chipX, chipY), textRect.size()),
                         Qt::AlignCenter, label);
        ++labeled;
    }
    painter.end();
    emit segmentationPreviewReady(preview, labeled);
    emit logMessage(QStringLiteral("[SAM3D] segmentation preview: %1 object "
                                   "mask(s) composited over the image.")
                            .arg(labeled));
}

bool Sam3dWorker::runSceneGeneration() {
    // 1. Resolve the per-object masks: non-empty concept prompts drive the
    // AICore sam3 text-prompted segmentation (fully in-memory, end-to-end);
    // otherwise the explicit mask directory is the manual fallback.
    QVector<QImage> masks;
    if (!resolveSceneMasks(masks)) return false;
    const int maskCount = masks.size();
    // Diagnostic preview (opt-out): one image in the DB telling the user
    // exactly which objects the segmentation accepted — objects missing
    // relative to the upstream full-scene reconstruction are almost always
    // a segmentation-recall question (prompt coverage / thresholds), not an
    // inference one.
    emitSegmentationPreview(masks);

    static const char* kQuant[] = {"f16", "q8_0", "q4_k"};
    const QByteArray modelsDirUtf8 = m_modelsDir.toUtf8();
    aicore_sam3d_options* opts = aicore_sam3d_options_new();
    aicore_sam3d_options_set_models_dir(opts, modelsDirUtf8.constData());
    aicore_sam3d_options_set_dtype(
            opts, m_settings.dtypeIndex == 0   ? AICORE_SAM3D_DTYPE_F16
                  : m_settings.dtypeIndex == 1 ? AICORE_SAM3D_DTYPE_Q8_0
                                               : AICORE_SAM3D_DTYPE_Q4_K);
    aicore_sam3d_options_set_device(opts,
                                    m_settings.device.toUtf8().constData());
    aicore_sam3d_options_set_threads(opts, 8);
    aicore_sam3d_options_set_seed(opts, m_settings.seed);
    aicore_sam3d_options_set_steps(opts, m_settings.steps, m_settings.steps);
    // Every mask shares the same RGB source: the MoGe point map is computed
    // once and reused (bit-identical replay of the same forward).
    aicore_sam3d_options_set_disable_moge_cache(opts, 0);
    // The composer inputs (PLY-semantic splat rows + pose receipt).
    aicore_sam3d_options_set_scene_attributes(opts, 1);
    if (m_settings.pbrBake) {
        // Official PBR bake per object; needs a PLY path to read back.
        aicore_sam3d_options_set_pbr(opts, 1, m_settings.pbrTextureSize,
                                     m_settings.pbrBakeSteps);
    }
    char err[512] = {0};
    aicore_sam3d_ctx* ctx = aicore_sam3d_load_opts(opts, err, sizeof(err));
    aicore_sam3d_options_free(opts);
    if (ctx == nullptr) {
        m_error = QStringLiteral("AICore sam3d load failed: %1")
                          .arg(QString::fromUtf8(err));
        return false;
    }
    // One shared TempLocation PLY scratch file serves every per-object run
    // (each run overwrites it; the composer consumes the typed attributes).
    QString plyScratch;
    if (m_settings.pbrBake) {
        plyScratch = QDir(QStandardPaths::writableLocation(
                                  QStandardPaths::TempLocation))
                             .filePath(QStringLiteral("SAM3D_pbr_%1.ply")
                                               .arg(QCoreApplication::
                                                            applicationPid()));
    }
    // TempLocation scratch cleanup on EVERY exit path (success, per-object
    // failure, cancel): the bake reads the Gaussian PLY back through a file
    // path, but the file itself is an implementation detail the user never
    // sees. The single-object path removes it explicitly; the scene path's
    // five returns made a manual call leak-prone, hence the guard.
    struct PlyScratchGuard {
        QString path;
        ~PlyScratchGuard() {
            if (!path.isEmpty()) QFile::remove(path);
        }
    } plyScratchGuard{plyScratch};
    const QString backend = QString::fromUtf8(aicore_sam3d_backend(ctx));
    emit logMessage(QStringLiteral("[SAM3D] Backend resolved: %1 — scene "
                                   "mode with %2 object mask(s).")
                            .arg(backend)
                            .arg(masks.size()));

    aicore_image_view image_view{};
    image_view.data = const_cast<uint8_t*>(m_sourceImage.constBits());
    image_view.width = m_sourceImage.width();
    image_view.height = m_sourceImage.height();
    image_view.row_stride_bytes =
            static_cast<size_t>(m_sourceImage.bytesPerLine());
    image_view.format = AICORE_IMAGE_BGRA8;

    QList<Sam3dRunResult> objects;
    int bakeFailures = 0;
    for (int index = 0; index < maskCount; ++index) {
        if (m_cancel.load()) {
            m_error = QStringLiteral("cancelled by user");
            aicore_sam3d_free(ctx);
            return false;
        }
        const QString label = m_sceneMaskLabels.value(
                index, QStringLiteral("obj_%1").arg(index));
        // Slice the reconstruction share of the global 0..100 progress for
        // this object: [m_segEndPercent, 100] — the segmentation owns the
        // head of the axis (0 when masks came from a directory, no seg).
        const int segEnd = m_segEndPercent;
        const int objBase = segEnd + index * (100 - segEnd) / maskCount;
        const int objEnd = segEnd + (index + 1) * (100 - segEnd) / maskCount;
        m_objBase.store(objBase, std::memory_order_relaxed);
        m_objSpan.store(objEnd - objBase, std::memory_order_relaxed);
        const QImage& maskImage = masks[index];
        if (maskImage.width() != m_sourceImage.width() ||
            maskImage.height() != m_sourceImage.height()) {
            emit logMessage(QStringLiteral("[SAM3D] skipping '%1' — resolution "
                                           "mismatch.")
                                    .arg(label));
            continue;
        }
        // Official scene masks are RGBA with the binary shape in the ALPHA
        // channel (notebook load_mask: mask > 0 on byte 3; RGB may be
        // noise). Keep the alpha byte as the last view channel so the C API
        // mask merge (last decoded channel, value > 0 keeps the pixel)
        // matches the official semantics. Plain gray masks stay GRAY8.
        const bool alphaMask = maskImage.hasAlphaChannel();
        const QImage maskArgb =
                alphaMask ? maskImage.convertToFormat(QImage::Format_ARGB32)
                          : QImage();
        const QImage gray =
                alphaMask ? QImage()
                          : (maskImage.format() == QImage::Format_Grayscale8
                                     ? maskImage
                                     : maskImage.convertToFormat(
                                               QImage::Format_Grayscale8));
        const QImage& maskHost = alphaMask ? maskArgb : gray;
        aicore_image_view mask_view{};
        mask_view.data = const_cast<uint8_t*>(maskHost.constBits());
        mask_view.width = maskHost.width();
        mask_view.height = maskHost.height();
        mask_view.row_stride_bytes =
                static_cast<size_t>(maskHost.bytesPerLine());
        mask_view.format = alphaMask ? AICORE_IMAGE_BGRA8 : AICORE_IMAGE_GRAY8;

        emit logMessage(QStringLiteral("[SAM3D] object %1/%2: '%3'")
                                .arg(index + 1)
                                .arg(maskCount)
                                .arg(label));
        // Per-object runs keep the official single-object numerics; the PLY
        // export stays off (the composer consumes the typed attributes) and
        // the PBR bake reads its scratch PLY back in AICore.
        aicore_sam3d_result* sam3d = aicore_sam3d_generate(
                ctx, &image_view, &mask_view,
                plyScratch.isEmpty() ? nullptr
                                     : plyScratch.toUtf8().constData(),
                m_settings.outputTexturedMesh ? 1 : 0, sam3dProgressTrampoline,
                this);
        if (sam3d == nullptr) {
            m_error = QStringLiteral("object %1 (%2) failed: %3")
                              .arg(index + 1)
                              .arg(label)
                              .arg(QString::fromUtf8(
                                      aicore_sam3d_last_error(ctx) != nullptr
                                              ? aicore_sam3d_last_error(ctx)
                                              : "?"));
            aicore_sam3d_free(ctx);
            return false;
        }
        Sam3dRunResult object;
        object.sourceImage = m_settings.imagePath;
        object.backend = backend;
        object.dtype = QString::fromLatin1(kQuant[m_settings.dtypeIndex]);
        object.objectLabel = label;
        fillRunResult(ctx, sam3d, object);
        aicore_sam3d_result_free(sam3d);

        if (object.gaussianCount <= 0 || object.pose.size() != 10) {
            emit logMessage(
                    QStringLiteral("[SAM3D] object %1 produced no usable "
                                   "artifacts — skipped.")
                            .arg(object.objectLabel));
            continue;
        }
        // Place the object into the scene frame right after the run: the
        // official make_scene position action applies to BOTH the mesh
        // vertices and the splat centers, so the GLB texture query and the
        // per-object DB point cloud share one frame and every entity lands
        // on its splats.
        transformMeshByScenePose(object.vertices, object.pose);
        transformMeshByScenePose(object.splatCenters, object.pose);
        if (m_settings.outputTexturedMesh && !m_settings.pbrBake &&
            object.meshTriangleCount > 0) {
            if (!bakeSceneObjectGlb(object, index)) {
                // The bake failure is already logged; keep the object's
                // point cloud (below) and count the mesh as missing so the
                // final summary reports the degradation explicitly.
                ++bakeFailures;
            }
        } else if (m_settings.pbrBake && object.hasPbr) {
            // Official PBR GLB (already in memory from AICore): opt-in file
            // export through the explicit output directory.
            writeGlbFile(object);
        }
        // Bounded memory: the local display RGB fed the vertex coloring and
        // the per-object DB point cloud takes its colors from the PLY-semantic
        // sh0, so only the RGB table is released here.
        object.splatRgb.clear();
        // Incremental delivery: hand the finished object to the GUI thread
        // right away (queued connection) — the DB tree grows while the
        // remaining objects are still generating.
        emit objectReady(object);
        objects.append(object);
    }
    aicore_sam3d_free(ctx);
    if (objects.isEmpty()) {
        m_error = QStringLiteral("no object produced usable artifacts");
        return false;
    }

    Sam3dSceneResult sceneResult;
    sceneResult.sourceImage = m_settings.imagePath;
    sceneResult.backend = backend;
    sceneResult.dtype = QString::fromLatin1(kQuant[m_settings.dtypeIndex]);
    sceneResult.objects = objects;
    sceneResult.e2eMs = 0.0;
    for (const Sam3dRunResult& object : objects) {
        sceneResult.e2eMs += object.e2eMs;
    }

    if (bakeFailures > 0) {
        emit logMessage(QStringLiteral("[SAM3D] %1 object GLB bake(s) failed "
                                       "— those objects keep their splat "
                                       "point cloud only.")
                                .arg(bakeFailures));
    }
    emit sceneReady(sceneResult);
    return true;
}

bool Sam3dWorker::bakeSceneObjectGlb(Sam3dRunResult& result, int objectIndex) {
    // The vertices AND the splat centers are already in the scene frame
    // (the caller applied the official make_scene position action to both),
    // so the nearest-splat texture query runs in the same frame the GLB is
    // written in — colors land exactly on their surfaces.
    if (!bakeTexturedGlb(result)) return false;
    // Scene GLBs are always file-backed (the dialog requires an output
    // directory in scene mode); release the bytes, the path is the DB
    // import source.
    result.glb.clear();
    Q_UNUSED(objectIndex);
    return true;
}

bool Sam3dWorker::bakeTexturedGlb(Sam3dRunResult& result) {
    const int nv = result.meshVertexCount;
    const int nt = result.meshTriangleCount;
    // Per-vertex PBR (6*nv: base_color rgb, metallic, roughness, alpha) from
    // the nearest gaussian splat's display color. The FlexiCubes surface and
    // the splats live on the same object surface, so the projected atlas
    // carries the splat rendering's look into the GLB texture.
    sam3d_colors::SplatColorGrid grid(result.splatCenters, result.splatRgb);
    // Preferred color source: the official SLatMeshDecoder use_color payload
    // (smooth neural-field vertex colors — the same source the official
    // non-baked GLB path writes as COLOR_0). Falls back to the nearest-splat
    // grid only when the decode ran without color attributes.
    const bool haveAttrs =
            result.meshVertexAttrs.size() == static_cast<qsizetype>(nv) * 6;
    const float* attrs =
            haveAttrs ? result.meshVertexAttrs.constData() : nullptr;
    std::vector<float> pbr(static_cast<size_t>(nv) * 6, 0.0f);
    const float* verts = result.vertices.constData();
    for (int i = 0; i < nv; ++i) {
        float rgb[3];
        if (attrs) {
            for (int c = 0; c < 3; ++c) {
                rgb[c] = std::min(1.0f, std::max(0.0f, attrs[i * 6 + c]));
            }
        } else {
            grid.colorAt(verts[i * 3 + 0], verts[i * 3 + 1], verts[i * 3 + 2],
                         rgb);
        }
        float* dst = pbr.data() + static_cast<size_t>(i) * 6;
        dst[0] = rgb[0];
        dst[1] = rgb[1];
        dst[2] = rgb[2];
        dst[3] = 0.0f;  // metallic
        dst[4] = 1.0f;  // roughness (matte diffuse)
        dst[5] = 1.0f;  // alpha
    }
    // The bake API takes signed triangle indices; FlexiCubes indices fit
    // int32 by construction (mesh_vertex_count < 2^31).
    std::vector<int32_t> tris(result.triangles.cbegin(),
                              result.triangles.cend());

    emit logMessage(
            QStringLiteral("[SAM3D] Baking textured GLB (%1 vertices, %2 "
                           "faces, 2048px UV atlas)...")
                    .arg(nv)
                    .arg(nt));
    emit progressChanged(progressPercentFor(8, 0, 0));

    // Cooperative cancel: the flag is refreshed inside the progress sink,
    // which the bake pipeline fires at the same stage boundaries it polls.
    volatile int cancel_flag = 0;
    struct BakeContext {
        Sam3dWorker* worker;
        volatile int* cancel;
    } bake_ctx{this, &cancel_flag};
    auto bake_progress = [](const char* stage, double elapsed_s, void* user) {
        auto* ctx = static_cast<BakeContext*>(user);
        *ctx->cancel =
                ctx->worker->m_cancel.load(std::memory_order_relaxed) ? 1 : 0;
        ctx->worker->logMessage(QStringLiteral("[SAM3D] bake: %1 (%2 s)")
                                        .arg(QLatin1String(stage))
                                        .arg(elapsed_s, 0, 'f', 1));
    };

    char err[512] = {0};
    int glb_len = 0;
    // The bake pipeline is task-independent (any mesh + per-vertex PBR);
    // the plugin orchestrates it through the public AICore bake API.
    uint8_t* glb = aicore_trellis_bake_glb_ex(
            verts, nv, tris.data(), nt, pbr.data(), 2048, /*RemoveTiny*/ 0,
            bake_progress, &bake_ctx, &cancel_flag, &glb_len, err, sizeof(err));
    if (m_cancel.load()) {
        if (glb) aicore_trellis_free_buffer(glb);
        return false;
    }
    if (glb == nullptr) {
        emit logMessage(
                QStringLiteral("[SAM3D] GLB bake failed: %1 — the vertex-"
                               "color mesh in the DB is still valid.")
                        .arg(QString::fromUtf8(err)));
        return false;
    }
    result.glb = QByteArray(reinterpret_cast<const char*>(glb), glb_len);
    aicore_trellis_free_buffer(glb);
    writeGlbFile(result);
    return true;
}

bool Sam3dWorker::writeGlbFile(Sam3dRunResult& result) {
    // File export is opt-in through the explicit output directory (an empty
    // field never writes anything — see the dialog validation).
    if (result.glb.isEmpty() || m_settings.outputDir.isEmpty()) {
        return false;
    }
    const QString base = QFileInfo(m_settings.imagePath).completeBaseName();
    const QString glbName =
            result.objectLabel.isEmpty()
                    ? QStringLiteral("%1_sam3d.glb").arg(base)
                    : QStringLiteral("%1_%2.glb").arg(base, result.objectLabel);
    const QString glbPath = QDir(m_settings.outputDir).filePath(glbName);
    QFile f(glbPath);
    if (f.open(QIODevice::WriteOnly)) {
        f.write(result.glb);
        f.close();
        result.glbPath = glbPath;
        emit logMessage(
                QStringLiteral("[SAM3D] Textured GLB written to "
                               "%1 (%2 MB).")
                        .arg(glbPath)
                        .arg(result.glb.size() / (1024.0 * 1024.0), 0, 'f', 1));
        return true;
    }
    emit logMessage(
            QStringLiteral("[SAM3D] GLB file write failed: %1").arg(glbPath));
    return false;
}

void Sam3dWorker::run() {
    if (!decodeInputImage()) {
        emit taskFinished(false);
        return;
    }
    // Preview-only mode (the dialog's "Generate mask preview" button):
    // run the segmentation pass and deliver the masks-over-image preview,
    // then stop — no sam3d model checks, no per-object reconstruction. The
    // user iterates on prompts until the preview is satisfactory, and only
    // then presses Generate for the full pipeline.
    if (m_settings.previewOnly) {
        auto previewLock =
                ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
        if (!previewLock.isLocked()) {
            m_error = QStringLiteral(
                    "Failed to acquire the inference device; another AI task "
                    "is running.");
            emit taskFinished(false);
            return;
        }
        QVector<QImage> masks;
        if (!resolveSceneMasks(masks)) {
            emit taskFinished(false);
            return;
        }
        emitSegmentationPreview(masks);
        emit progressChanged(100);
        emit taskFinished(true);
        return;
    }
    if (!resolveSam3dModels()) {
        emit taskFinished(false);
        return;
    }
    auto deviceLock = ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
    if (!deviceLock.isLocked()) {
        m_error = QStringLiteral(
                "Failed to acquire the inference device; another AI task is "
                "running.");
        emit taskFinished(false);
        return;
    }
    if (m_cancel.load()) {
        emit taskFinished(false);
        return;
    }
    // Scene mode: per-object runs (automatic sam3 segmentation from
    // concept prompts, or explicit mask files); RMBG (a single-object
    // foreground extractor) has no role there.
    const bool sceneMode = sceneModeRequested();
    if (!sceneMode && (!applyRmbgMask() || m_cancel.load())) {
        emit taskFinished(false);
        return;
    }
    if (sceneMode ? !runSceneGeneration() : !runGeneration()) {
        emit taskFinished(false);
        return;
    }
    emit taskFinished(true);
}
