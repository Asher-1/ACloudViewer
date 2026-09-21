// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Sam3dWorker.h"

#include <QDir>
#include <QFileInfo>

#include "aicore/image_view.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3d_capi.h"
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
    if (m_settings.generateMesh &&
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
    for (const QString& name :
         {QStringLiteral("rmbg_q8.gguf"), QStringLiteral("rmbg_f16.gguf")}) {
        const QString candidate = rmbgDir.filePath(name);
        if (QFileInfo::exists(candidate)) {
            modelPath = candidate;
            break;
        }
    }
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
    const QString plyName =
            QStringLiteral("%1_sam3d.ply")
                    .arg(QFileInfo(m_settings.imagePath).completeBaseName());
    const QByteArray plyUtf8 =
            QDir(m_settings.outputDir).filePath(plyName).toUtf8();

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
            ctx, &view, nullptr, plyUtf8.constData(),
            m_settings.generateMesh ? 1 : 0, sam3dProgressTrampoline, this);

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
    result.plyPath = QString::fromUtf8(plyUtf8);
    result.gaussianCount =
            static_cast<int>(aicore_sam3d_result_gaussian_count(sam3d));
    result.meshVertexCount = aicore_sam3d_result_mesh_vertex_count(sam3d);
    result.meshTriangleCount = aicore_sam3d_result_mesh_triangle_count(sam3d);
    if (aicore_sam3d_result_has_mesh(sam3d)) {
        const float* verts = aicore_sam3d_result_mesh_vertices(sam3d);
        const uint32_t* tris = aicore_sam3d_result_mesh_triangles(sam3d);
        result.vertices =
                QVector<float>(verts, verts + result.meshVertexCount * 3);
        result.triangles =
                QVector<uint32_t>(tris, tris + result.meshTriangleCount * 3);
    }
    aicore_pipeline_timings timings{};
    if (aicore_sam3d_last_pipeline_timings(ctx, &timings) == 0) {
        result.e2eMs = timings.e2e_ms;
    }
    aicore_sam3d_result_free(sam3d);
    aicore_sam3d_free(ctx);

    emit resultReady(result);
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
    if (!applyRmbgMask() || m_cancel.load()) {
        emit taskFinished(false);
        return;
    }
    if (!runGeneration()) {
        emit taskFinished(false);
        return;
    }
    emit taskFinished(true);
}
