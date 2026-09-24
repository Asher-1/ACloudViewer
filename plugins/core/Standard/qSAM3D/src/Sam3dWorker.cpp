// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Sam3dWorker.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <algorithm>

#include "Sam3dVertexColors.h"
#include "aicore/image_view.h"
#include "aicore/rmbg_capi.h"
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
            return "mesh decode";
        default:
            return "stage";
    }
}

// Static trampoline for the C progress callback (no captures allowed).
void sam3dProgressTrampoline(void* user, int stage, int step, int total) {
    auto* worker = static_cast<Sam3dWorker*>(user);
    emit worker->stageChanged(stage, step, total);
    emit worker->logMessage(QStringLiteral("[SAM3D] stage: %1")
                                    .arg(QLatin1String(stageName(stage))));
}

}  // namespace

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
    // in-memory artifact sink (no file IO inside the pipeline).
    const QString plyName =
            QStringLiteral("%1_sam3d.ply")
                    .arg(QFileInfo(m_settings.imagePath).completeBaseName());
    const QString plyPath =
            m_settings.exportPly ? QDir(m_settings.outputDir).filePath(plyName)
                                 : QString();
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
            m_settings.exportPly ? plyUtf8.constData() : nullptr,
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
    result.plyPath = plyPath;
    fillRunResult(ctx, sam3d, result);
    aicore_sam3d_result_free(sam3d);
    aicore_sam3d_free(ctx);

    if (m_settings.outputTexturedMesh && result.meshTriangleCount > 0) {
        bakeTexturedGlb(result);
    }

    emit resultReady(result);
    return true;
}

void Sam3dWorker::fillRunResult(aicore_sam3d_ctx* ctx,
                                aicore_sam3d_result* sam3d,
                                Sam3dRunResult& result) {
    result.gaussianCount =
            static_cast<int>(aicore_sam3d_result_gaussian_count(sam3d));
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
    }
    // Scene-composer interchange attributes + pose receipt (scene mode only).
    if (!m_settings.masksDir.isEmpty()) {
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

bool Sam3dWorker::runSceneGeneration() {
    // 1. Enumerate the per-object masks in numeric order (0.png, 1.png, ...).
    QFileInfoList masks =
            QDir(m_settings.masksDir)
                    .entryInfoList(QStringList{QStringLiteral("*.png")},
                                   QDir::Files, QDir::Name);
    // Numeric order for the official <n>.png mask naming (2.png < 10.png).
    std::sort(masks.begin(), masks.end(),
              [](const QFileInfo& a, const QFileInfo& b) {
                  bool aOk = false;
                  bool bOk = false;
                  const int aIdx = a.baseName().toInt(&aOk);
                  const int bIdx = b.baseName().toInt(&bOk);
                  if (aOk && bOk) return aIdx < bIdx;
                  return a.baseName() < b.baseName();
              });
    if (masks.isEmpty()) {
        m_error = QStringLiteral("the mask directory has no PNG masks");
        return false;
    }

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
    char err[512] = {0};
    aicore_sam3d_ctx* ctx = aicore_sam3d_load_opts(opts, err, sizeof(err));
    aicore_sam3d_options_free(opts);
    if (ctx == nullptr) {
        m_error = QStringLiteral("AICore sam3d load failed: %1")
                          .arg(QString::fromUtf8(err));
        return false;
    }
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
    for (int index = 0; index < masks.size(); ++index) {
        if (m_cancel.load()) {
            m_error = QStringLiteral("cancelled by user");
            aicore_sam3d_free(ctx);
            return false;
        }
        const QFileInfo& maskInfo = masks[index];
        QImage mask(maskInfo.absoluteFilePath());
        if (mask.isNull() || mask.width() != m_sourceImage.width() ||
            mask.height() != m_sourceImage.height()) {
            emit logMessage(QStringLiteral("[SAM3D] skipping '%1' — missing or "
                                           "resolution mismatch.")
                                    .arg(maskInfo.fileName()));
            continue;
        }
        const QImage gray =
                mask.format() == QImage::Format_Grayscale8
                        ? mask
                        : mask.convertToFormat(QImage::Format_Grayscale8);
        aicore_image_view mask_view{};
        mask_view.data = const_cast<uint8_t*>(gray.constBits());
        mask_view.width = gray.width();
        mask_view.height = gray.height();
        mask_view.row_stride_bytes = static_cast<size_t>(gray.bytesPerLine());
        mask_view.format = AICORE_IMAGE_GRAY8;

        emit logMessage(QStringLiteral("[SAM3D] object %1/%2: mask '%3'")
                                .arg(index + 1)
                                .arg(masks.size())
                                .arg(maskInfo.fileName()));
        // Per-object runs keep the official single-object numerics; the PLY
        // export stays off (the composer consumes the typed attributes).
        aicore_sam3d_result* sam3d =
                aicore_sam3d_generate(ctx, &image_view, &mask_view, nullptr,
                                      m_settings.outputTexturedMesh ? 1 : 0,
                                      sam3dProgressTrampoline, this);
        if (sam3d == nullptr) {
            m_error = QStringLiteral("object %1 (%2) failed: %3")
                              .arg(index + 1)
                              .arg(maskInfo.fileName())
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
        object.objectLabel = QStringLiteral("obj_%1").arg(index);
        fillRunResult(ctx, sam3d, object);
        aicore_sam3d_result_free(sam3d);

        if (object.gaussianCount <= 0 || object.pose.size() != 10) {
            emit logMessage(
                    QStringLiteral("[SAM3D] object %1 produced no usable "
                                   "artifacts — skipped.")
                            .arg(object.objectLabel));
            continue;
        }
        if (m_settings.outputTexturedMesh && object.meshTriangleCount > 0) {
            bakeSceneObjectGlb(object, index);
        }
        // Bounded memory: the local display RGB fed the vertex coloring;
        // the composer consumes the world-domain centers and the
        // PLY-semantic attributes, so only the RGB table is released.
        object.splatRgb.clear();
        objects.append(object);
    }
    aicore_sam3d_free(ctx);
    if (objects.isEmpty()) {
        m_error = QStringLiteral("no object produced usable artifacts");
        return false;
    }

    // 2. Compose: activation + make_scene pose per object, concatenation.
    // The input arrays are borrowed from the per-object envelopes, which
    // stay alive in the returned scene result.
    std::vector<aicore_sam3d_scene_object> inputs;
    inputs.reserve(objects.size());
    for (const Sam3dRunResult& object : objects) {
        aicore_sam3d_scene_object input{};
        input.splat_count = object.gaussianCount;
        input.centers = object.splatCenters.constData();
        input.sh0 = object.splatSh0.constData();
        input.opacity_logit = object.splatOpacityLogit.constData();
        input.log_scale = object.splatLogScale.constData();
        input.rot_ply = object.splatRotPly.constData();
        input.pose = object.pose.constData();
        inputs.push_back(input);
    }
    char compose_err[512] = {0};
    aicore_sam3d_scene_result* scene = aicore_sam3d_scene_assemble(
            inputs.data(), static_cast<int>(inputs.size()),
            /*normalize*/ 0, compose_err, sizeof(compose_err));
    if (scene == nullptr) {
        m_error = QStringLiteral("scene assembly failed: %1")
                          .arg(QString::fromUtf8(compose_err));
        return false;
    }

    Sam3dSceneResult sceneResult;
    sceneResult.sourceImage = m_settings.imagePath;
    sceneResult.backend = backend;
    sceneResult.dtype = QString::fromLatin1(kQuant[m_settings.dtypeIndex]);
    sceneResult.objects = objects;
    sceneResult.sceneSplatCount =
            static_cast<int>(aicore_sam3d_scene_result_splat_count(scene));
    const float* positions = aicore_sam3d_scene_result_positions(scene);
    const float* sh0 = aicore_sam3d_scene_result_sh0(scene);
    if (positions && sh0 && sceneResult.sceneSplatCount > 0) {
        const int count = sceneResult.sceneSplatCount * 3;
        sceneResult.sceneCenters.resize(count);
        std::copy(positions, positions + count,
                  sceneResult.sceneCenters.data());
        sceneResult.sceneRgb.resize(count);
        for (int i = 0; i < count; ++i) {
            float rgb = 0.5f + 0.28209479177387814f * sh0[i];
            sceneResult.sceneRgb[i] = rgb < 0.f ? 0.f : (rgb > 1.f ? 1.f : rgb);
        }
    }
    aicore_pipeline_timings timings{};
    if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0) {
        sceneResult.e2eMs = timings.e2e_ms;
    }
    aicore_sam3d_scene_result_free(scene);

    emit sceneReady(sceneResult);
    return true;
}

bool Sam3dWorker::bakeSceneObjectGlb(Sam3dRunResult& result, int objectIndex) {
    // Bake in the scene frame: the composer places the splats with the
    // official make_scene position action, so the GLB vertices get the same
    // rigid transform and every per-object GLB lands on its splats. The
    // vertex colors are sampled from the LOCAL splat frame (both
    // representations coincide before the transform).
    transformMeshByScenePose(result.vertices, result.pose);
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
    std::vector<float> pbr(static_cast<size_t>(nv) * 6, 0.0f);
    const float* verts = result.vertices.constData();
    for (int i = 0; i < nv; ++i) {
        float rgb[3];
        grid.colorAt(verts[i * 3 + 0], verts[i * 3 + 1], verts[i * 3 + 2], rgb);
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

    if (!m_settings.outputDir.isEmpty()) {
        const QString base = QFileInfo(m_settings.imagePath).completeBaseName();
        const QString glbName =
                result.objectLabel.isEmpty()
                        ? QStringLiteral("%1_sam3d.glb").arg(base)
                        : QStringLiteral("%1_%2.glb")
                                  .arg(base, result.objectLabel);
        const QString glbPath = QDir(m_settings.outputDir).filePath(glbName);
        QFile f(glbPath);
        if (f.open(QIODevice::WriteOnly)) {
            f.write(result.glb);
            f.close();
            result.glbPath = glbPath;
            emit logMessage(QStringLiteral("[SAM3D] Textured GLB written to "
                                           "%1 (%2 MB).")
                                    .arg(glbPath)
                                    .arg(result.glb.size() / (1024.0 * 1024.0),
                                         0, 'f', 1));
        } else {
            emit logMessage(QStringLiteral("[SAM3D] GLB file write failed: %1")
                                    .arg(glbPath));
        }
    }
    return true;
}

void Sam3dWorker::run() {
    if (!decodeInputImage() || !resolveSam3dModels()) {
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
    // Scene mode consumes explicit per-object mask files; RMBG (a single-
    // object foreground extractor) has no role there.
    const bool sceneMode = !m_settings.masksDir.isEmpty();
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
