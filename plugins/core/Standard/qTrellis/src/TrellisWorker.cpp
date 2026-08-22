// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisWorker.h"

#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>

#include <cstring>
#include <vector>

#include "TrellisModelCatalog.h"
#include "aicore/backend_capi.h"
#include "aicore/trellis_capi.h"

namespace {

// Stage names for the progress log (mirror aicore_trellis_stage).
const char* stageName(int stage) {
    switch (stage) {
        case AICORE_TRELLIS_STAGE_PREPROCESS: return "preprocess";
        case AICORE_TRELLIS_STAGE_DINO: return "dino";
        case AICORE_TRELLIS_STAGE_SS_FLOW: return "ss_flow";
        case AICORE_TRELLIS_STAGE_SS_DEC: return "ss_dec";
        case AICORE_TRELLIS_STAGE_SLAT_FLOW: return "slat_flow";
        case AICORE_TRELLIS_STAGE_SHAPE_DEC: return "shape_dec";
        case AICORE_TRELLIS_STAGE_MESH: return "mesh";
        case AICORE_TRELLIS_STAGE_UPSAMPLE: return "upsample";
        case AICORE_TRELLIS_STAGE_SLAT_FLOW_HR: return "slat_flow_hr";
        case AICORE_TRELLIS_STAGE_SHAPE_DEC_HR: return "shape_dec_hr";
        case AICORE_TRELLIS_STAGE_TEXTURE: return "texture";
        default: return "?";
    }
}

}  // namespace

TrellisWorker::TrellisWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {}

TrellisWorker::~TrellisWorker() {
    // Never free m_ctx from this destructor: QThread objects are destroyed on
    // the thread that owns them (usually the GUI thread), while the AICore
    // context belongs to the worker thread. Freeing it here would race with
    // run() and crash. run() always releases m_ctx before it returns; this
    // destructor only joins a still-running thread so no context leaks.
    if (isRunning()) {
        requestInterruption();
        wait(5000);
    }
}

void TrellisWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(QStringLiteral("[TRELLIS] AICore not enabled."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED

void TrellisWorker::applySettingsToOptions(aicore_trellis_options* opts) {
    aicore_trellis_options_set_device(opts, m_settings.device.toUtf8().constData());
    aicore_trellis_options_set_threads(opts, m_settings.threads);
    aicore_trellis_options_set_shape_dec_placement(
            opts, m_settings.shapeDecPlacement.toUtf8().constData());
    if (m_settings.useRmbg && !m_settings.rmbgModelPath.isEmpty()) {
        aicore_trellis_options_set_rmbg_gguf(
                opts, m_settings.rmbgModelPath.toUtf8().constData());
    }
}

bool TrellisWorker::resolveRmbgModel() {
    // The rmbg model comes from the same trellis2-ggml release; reuse the
    // trellis catalog entry (role == "rmbg").
    const QString cacheDir = TrellisHelpers::modelCacheDir();
    const QString path = cacheDir + QLatin1Char('/') + QStringLiteral("rmbg_f16.gguf");
    if (QFile::exists(path)) {
        m_settings.rmbgModelPath = path;
        return true;
    }
    emit logMessage(QStringLiteral(
            "[TRELLIS] RMBG model not found: %1 (download rmbg_f16.gguf "
            "first, e.g. via the qRMBG plugin). Falling back to solid-color "
            "background removal.").arg(path));
    m_settings.useRmbg = false;
    return true;
}

bool TrellisWorker::runInference() {
    if (m_settings.modelPaths.size() < 3) {
        emit logMessage(QStringLiteral("[TRELLIS] Model set incomplete."));
        return false;
    }

    // Load the image bytes up-front so the pipeline can be released right
    // after generation without keeping the file open.
    QFile file(m_settings.inputPath);
    if (!file.open(QIODevice::ReadOnly)) {
        emit logMessage(QStringLiteral("[TRELLIS] Cannot open image: %1")
                                .arg(m_settings.inputPath));
        return false;
    }
    const QByteArray imageBytes = file.readAll();
    file.close();

    // Keep the UTF-8 payloads alive for the whole call: QString::toUtf8()
    // returns a temporary whose .constData() would dangle as soon as the
    // expression ends, and aicore_trellis_load_opts reads these pointers
    // *after* loading the RMBG model (a long-running call).
    //
    // Index contract: m_settings.modelPaths MUST be in aicore_trellis_model_paths
    // field order (dino, ss_flow, ss_dec, slat_flow, slat_hr_flow, shape_dec,
    // shape_enc, tex_dec, tex_flow, tex_flow_hr). Omitted fields stay as empty
    // strings (the C API treats "" like NULL: "omit this model"). The presets
    // in TrellisModelCatalog.cpp guarantee this ordering.
    QByteArray utf8Paths[10];
    auto keepAlive = [&](int i) -> const char* {
        utf8Paths[i] = m_settings.modelPaths.value(i).toUtf8();
        return utf8Paths[i].constData();
    };

    aicore_trellis_model_paths paths{};
    paths.dino_gguf = keepAlive(0);
    paths.ss_flow_gguf = keepAlive(1);
    paths.ss_dec_gguf = keepAlive(2);
    // Optional models: 512 (+3), 1024 (+4), PBR (+5..+8).
    const int n = m_settings.modelPaths.size();
    if (n > 3) paths.slat_flow_gguf = keepAlive(3);
    if (n > 4) paths.slat_hr_flow_gguf = keepAlive(4);
    if (n > 5) paths.shape_dec_gguf = keepAlive(5);
    if (m_settings.textureEnabled) {
        if (n > 6) paths.shape_enc_gguf = keepAlive(6);
        if (n > 7) paths.tex_dec_gguf = keepAlive(7);
        if (n > 8) paths.tex_flow_gguf = keepAlive(8);
        if (n > 9) paths.tex_flow_hr_gguf = keepAlive(9);
    }

    if (m_settings.useRmbg) resolveRmbgModel();

    // Same keep-alive requirement for the RMBG path: the options struct
    // copies the string immediately, but keep it unambiguous anyway.
    QByteArray rmbgUtf8;
    if (m_settings.useRmbg && !m_settings.rmbgModelPath.isEmpty()) {
        rmbgUtf8 = m_settings.rmbgModelPath.toUtf8();
    }

    aicore_trellis_options* opts = aicore_trellis_options_new();
    applySettingsToOptions(opts);
    if (m_settings.useRmbg && !rmbgUtf8.isEmpty()) {
        aicore_trellis_options_set_rmbg_gguf(opts, rmbgUtf8.constData());
    }
    aicore_trellis_ctx* ctx = aicore_trellis_load_opts(&paths, opts);
    aicore_trellis_options_free(opts);
    if (!ctx) {
        emit logMessage(QStringLiteral(
                "[TRELLIS] Model load failed (check that every preset file "
                "exists and is a valid GGUF)."));
        return false;
    }
    m_ctx = ctx;

    aicore_trellis_generate_params params{};
    params.pipeline_type = m_settings.pipelineType;
    params.background_mode = m_settings.backgroundMode;
    params.seed = m_settings.seed;
    params.steps = m_settings.steps;
    params.guidance = static_cast<float>(m_settings.guidance);
    params.texture_steps = m_settings.textureSteps;

    char err[512] = {0};
    QElapsedTimer timer;
    timer.start();
    aicore_trellis_mesh* mesh = aicore_trellis_generate(
            ctx, imageBytes.constData(), imageBytes.size(), &params,
            [](void* user, int stage, int step, int total) {
                auto* self = static_cast<TrellisWorker*>(user);
                self->emit progressUpdate(stage, step, total);
            },
            this, err, sizeof(err));
    const double elapsedMs = static_cast<double>(timer.elapsed());

    if (!mesh) {
        emit logMessage(QStringLiteral("[TRELLIS] Generation failed: %1")
                                .arg(QString::fromUtf8(err)));
        aicore_trellis_free(ctx);
        m_ctx = nullptr;
        return false;
    }

    TrellisRunResult result;
    result.sourceImage = m_settings.inputPath;
    result.presetName = m_settings.presetName;
    result.totalRuntimeMs = elapsedMs;
    result.backend = QString::fromUtf8(aicore_trellis_backend(ctx));

    const int nv = aicore_trellis_mesh_n_verts(mesh);
    const int nt = aicore_trellis_mesh_n_tris(mesh);
    if (nv > 0) {
        const float* v = aicore_trellis_mesh_verts(mesh);
        const float* nn = aicore_trellis_mesh_normals(mesh);
        result.verts.resize(nv * 3);
        result.normals.resize(nv * 3);
        std::memcpy(result.verts.data(), v, sizeof(float) * nv * 3);
        if (nn) {
            std::memcpy(result.normals.data(), nn, sizeof(float) * nv * 3);
        }
    }
    if (nt > 0) {
        const int* t = aicore_trellis_mesh_tris(mesh);
        result.tris.resize(nt * 3);
        std::memcpy(result.tris.data(), t, sizeof(int) * nt * 3);
    }
    if (aicore_trellis_mesh_has_pbr(mesh)) {
        const float* pbr = aicore_trellis_mesh_pbr(mesh);
        result.pbr.resize(nv * 6);
        std::memcpy(result.pbr.data(), pbr, sizeof(float) * nv * 6);
        result.hasPbr = true;
    }
    aicore_trellis_mesh_free(mesh);

    emit logMessage(QStringLiteral(
            "[TRELLIS] Mesh %1 verts / %2 tris generated in %3 ms (backend %4)")
                            .arg(nv)
                            .arg(nt)
                            .arg(elapsedMs, 0, 'f', 0)
                            .arg(result.backend));

    aicore_trellis_free(ctx);
    m_ctx = nullptr;

    emit resultReady(result);
    return true;
}

#endif  // AICore_ENABLED
