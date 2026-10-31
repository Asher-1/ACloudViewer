// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "InstantMeshWorker.h"

#include <QFile>
#include <QFileInfo>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#ifdef AICore_ENABLED
#include <aicore/instantmesh_capi.h>
#include <aicore/rmbg_capi.h>

#include "InstantMeshModelCatalog.h"
#include "ecvAICoreRuntimeHelpers.h"
#include "ecvAssetIntegrity.h"
#endif

namespace {

#ifdef AICore_ENABLED

// ---------------------------------------------------------------------------
// Deterministic view picking + shaded preview render (adapted from the
// qTrellis worker: same conventions, textured by the baked RGBA atlas when
// present and by the per-vertex net_rgb colours otherwise).
// ---------------------------------------------------------------------------
struct ViewBasis {
    float r[3], u[3], d[3];  // screen right / screen up / toward camera
    float px_min = 0, px_max = 0, py_min = 0, py_max = 0;
};

void basisFromDir(const float dir[3], const float up[3], ViewBasis& b) {
    float d[3], r[3], u[3];
    const float dl =
            std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int i = 0; i < 3; ++i) d[i] = dir[i] / dl;
    r[0] = up[1] * d[2] - up[2] * d[1];
    r[1] = up[2] * d[0] - up[0] * d[2];
    r[2] = up[0] * d[1] - up[1] * d[0];
    const float rl =
            std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) + 1e-12f;
    for (int i = 0; i < 3; ++i) r[i] /= rl;
    u[0] = d[1] * r[2] - d[2] * r[1];
    u[1] = d[2] * r[0] - d[0] * r[2];
    u[2] = d[0] * r[1] - d[1] * r[0];
    for (int i = 0; i < 3; ++i) {
        b.d[i] = d[i];
        b.r[i] = r[i];
        b.u[i] = u[i];
    }
}

ViewBasis pickViewBasis(const float* verts, size_t nv) {
    struct Candidate {
        float dir[3];
        float up[3];
    };
    // InstantMesh meshes are Z-up (the zero123plus cameras orbit Z).
    static const Candidate kCandidates[] = {
            {{0.55f, 0.52f, 0.65f}, {0, 0, 1}},  // favours a Z-up mesh
            {{0.55f, 0.65f, 0.52f}, {0, 1, 0}},  // favours a Y-up mesh
            {{0.65f, 0.52f, 0.55f}, {1, 0, 0}},  // favours an X-up mesh
    };
    ViewBasis best{};
    float best_score = -1.0f;
    for (const Candidate& cand : kCandidates) {
        ViewBasis b;
        basisFromDir(cand.dir, cand.up, b);
        float px_min = 1e9f, px_max = -1e9f, py_min = 1e9f, py_max = -1e9f;
        for (size_t i = 0; i < nv; ++i) {
            const float* v = verts + i * 3;
            const float px = v[0] * b.r[0] + v[1] * b.r[1] + v[2] * b.r[2];
            const float py = v[0] * b.u[0] + v[1] * b.u[1] + v[2] * b.u[2];
            px_min = std::min(px_min, px);
            px_max = std::max(px_max, px);
            py_min = std::min(py_min, py);
            py_max = std::max(py_max, py);
        }
        const float ex = px_max - px_min, ey = py_max - py_min;
        const float score = std::min(ex, ey) / (std::max(ex, ey) + 1e-9f);
        if (score > best_score) {
            best = b;
            best.px_min = px_min;
            best.px_max = px_max;
            best.py_min = py_min;
            best.py_max = py_max;
            best_score = score;
        }
        if (score >= 0.15f) break;
    }
    return best;
}

void fitTransform(
        const ViewBasis& b, int size, float& scale, float& ox, float& oy) {
    const float ex = b.px_max - b.px_min, ey = b.py_max - b.py_min;
    scale = (float)(size - 8) / std::max(std::max(ex, ey), 1e-6f);
    ox = 4.0f + ((float)size - 8 - ex * scale) * 0.5f - b.px_min * scale;
    oy = 4.0f + ((float)size - 8 - ey * scale) * 0.5f + b.py_max * scale;
}

void viewProject(const ViewBasis& b, const float* v, float* out) {
    out[0] = v[0] * b.r[0] + v[1] * b.r[1] + v[2] * b.r[2];
    out[1] = v[0] * b.u[0] + v[1] * b.u[1] + v[2] * b.u[2];
    out[2] = v[0] * b.d[0] + v[1] * b.d[1] + v[2] * b.d[2];
}

const char* stageName(int stage) {
    switch (stage) {
        case AICORE_INSTANTMESH_STAGE_PREPROCESS:
            return "preprocess";
        case AICORE_INSTANTMESH_STAGE_RMBG:
            return "rmbg";
        case AICORE_INSTANTMESH_STAGE_MULTIVIEW:
            return "multiview";
        case AICORE_INSTANTMESH_STAGE_VAE_DECODE:
            return "vae_decode";
        case AICORE_INSTANTMESH_STAGE_ENCODE:
            return "encode";
        case AICORE_INSTANTMESH_STAGE_TRIPLANE:
            return "triplane";
        case AICORE_INSTANTMESH_STAGE_SYNTH:
            return "synth";
        case AICORE_INSTANTMESH_STAGE_MESH:
            return "mesh";
        case AICORE_INSTANTMESH_STAGE_TEXTURE:
            return "texture";
        default:
            return "stage";
    }
}

#endif  // AICore_ENABLED

}  // namespace

InstantMeshWorker::InstantMeshWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    qRegisterMetaType<InstantMeshRunResult>("InstantMeshRunResult");
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

InstantMeshWorker::~InstantMeshWorker() {
    cancel();
    // Wait for actual completion; the token must outlive the worker thread and
    // in-flight backend work.
    if (isRunning()) wait();
#ifdef AICore_ENABLED
    aicore_cancel_token_free(m_cancelToken);
#endif
}

void InstantMeshWorker::cancel() {
    requestInterruption();
#ifdef AICore_ENABLED
    aicore_cancel_token_request(m_cancelToken);
#endif
}

void InstantMeshWorker::run() {
#ifdef AICore_ENABLED
    bool ok = false;
    try {
        aicore::runtime::CancelScope cancelScope(m_cancelToken);
        ok = m_cancelToken && !aicore_cancel_requested() && runInference();
    } catch (const std::exception& e) {
        emit logMessage(QStringLiteral("[InstantMesh] Worker exception: %1")
                                .arg(QString::fromUtf8(e.what())));
    } catch (...) {
        emit logMessage(
                QStringLiteral("[InstantMesh] Worker exception: unknown"));
    }
    emit taskFinished(ok);
#else
    emit logMessage(QStringLiteral(
            "[InstantMesh] Built without AICore — inference unavailable."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED

bool InstantMeshWorker::resolveRmbgModel() {
    if (m_settings.rmbgModelPath.isEmpty()) {
        const auto* entry =
                aicore_rmbg_model_at(aicore_rmbg_model_default_index());
        if (entry && entry->filename)
            m_settings.rmbgModelPath = InstantMeshHelpers::rmbgModelCacheDir() +
                                       QLatin1Char('/') +
                                       QString::fromUtf8(entry->filename);
    }
    const QString filename = QFileInfo(m_settings.rmbgModelPath).fileName();
    const QByteArray name = filename.toUtf8();
    const auto* entry = aicore_rmbg_model_by_filename(name.constData());
    const QByteArray digest = ecvAssetIntegrity::PinnedDigest(filename);
    if (!entry || digest.isEmpty() ||
        !ecvAssetIntegrity::isVerified(
                m_settings.rmbgModelPath, {QCryptographicHash::Sha256, digest},
                0, true, ecvAssetIntegrity::OnMiss::DeepVerify)) {
        emit logMessage(tr("[InstantMesh] RMBG model missing or corrupt: %1")
                                .arg(m_settings.rmbgModelPath));
        return false;
    }
    return true;
}

void InstantMeshWorker::onProgress(int stage, int step, int total) {
    if (m_lastStage != stage) {
        if (m_lastStage >= 0) {
            m_stageMs[m_lastStage] +=
                    static_cast<double>(m_stageTimer.elapsed());
        }
        m_lastStage = stage;
        m_stageTimer.restart();
    }
    emit progressUpdate(stage, step, total);
}

// Plugin-local orthographic triangle rasterization; per-corner UVs preserve
// seams and directly sample top-first RGBA. Thumbnails do not replace native
// GUI material validation and call neither OpenGL nor inference backends.
QImage InstantMeshWorker::renderResultPreview(
        const InstantMeshRunResult& result, int size) {
    const auto& verts = result.verts;
    const int nv = verts.size() / 3;
    if (nv <= 0 || verts.size() % 3 || result.tris.isEmpty() ||
        result.tris.size() % 3 || size <= 8 || size > 4096)
        return QImage();
    for (float value : verts)
        if (!std::isfinite(value)) return QImage();
    for (float value : result.vcolors)
        if (!std::isfinite(value)) return QImage();
    for (int index : result.tris)
        if (index < 0 || index >= nv) return QImage();
    const bool textured = !result.texRgba.isEmpty() || !result.uvs.isEmpty();
    if (textured) {
        if (result.texW <= 0 || result.texH <= 0 ||
            uint64_t(result.texW) * result.texH * 4 !=
                    uint64_t(result.texRgba.size()) ||
            uint64_t(result.uvs.size()) != uint64_t(result.tris.size()) * 2)
            return QImage();
        for (float value : result.uvs)
            if (!std::isfinite(value)) return QImage();
    }
    QImage img(size, size, QImage::Format_ARGB32);
    img.fill(Qt::transparent);
    std::vector<float> depth((size_t)size * size, -1e9f), vp((size_t)nv * 3);
    const ViewBasis basis = pickViewBasis(verts.constData(), nv);
    float scale = 1.0f, ox = 0.0f, oy = 0.0f;
    fitTransform(basis, size, scale, ox, oy);
    for (int i = 0; i < nv; ++i)
        viewProject(basis, verts.constData() + i * 3, vp.data() + i * 3);
    for (int t = 0; t < result.tris.size() / 3; ++t) {
        float p[3][3], sx[3], sy[3];
        for (int k = 0; k < 3; ++k) {
            std::memcpy(p[k], vp.data() + result.tris[t * 3 + k] * 3,
                        sizeof(p[k]));
            sx[k] = ox + p[k][0] * scale;
            sy[k] = oy - p[k][1] * scale;
        }
        const float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) -
                           (sy[1] - sy[0]) * (sx[2] - sx[0]);
        if (std::fabs(area) < 1e-9f) continue;
        const float e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1],
                             p[1][2] - p[0][2]};
        const float e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1],
                             p[2][2] - p[0][2]};
        const float normal[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                                 e1[2] * e2[0] - e1[0] * e2[2],
                                 e1[0] * e2[1] - e1[1] * e2[0]};
        const float nl =
                std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                          normal[2] * normal[2]) +
                1e-20f;
        const float shade =
                0.25f + 0.75f * std::fabs((normal[0] * 0.4f + normal[1] * 0.7f +
                                           normal[2] * 0.6f) /
                                          nl);
        const int x0 =
                std::max(0, int(std::floor(std::min({sx[0], sx[1], sx[2]}))));
        const int x1 = std::min(
                size - 1, int(std::ceil(std::max({sx[0], sx[1], sx[2]}))));
        const int y0 =
                std::max(0, int(std::floor(std::min({sy[0], sy[1], sy[2]}))));
        const int y1 = std::min(
                size - 1, int(std::ceil(std::max({sy[0], sy[1], sy[2]}))));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float px = x + 0.5f, py = y + 0.5f;
                const float weights[3] = {((sx[1] - px) * (sy[2] - py) -
                                           (sy[1] - py) * (sx[2] - px)) /
                                                  area,
                                          ((sx[2] - px) * (sy[0] - py) -
                                           (sy[2] - py) * (sx[0] - px)) /
                                                  area,
                                          ((sx[0] - px) * (sy[1] - py) -
                                           (sy[0] - py) * (sx[1] - px)) /
                                                  area};
                if (weights[0] < 0 || weights[1] < 0 || weights[2] < 0)
                    continue;
                const float z = weights[0] * p[0][2] + weights[1] * p[1][2] +
                                weights[2] * p[2][2];
                float& previous = depth[(size_t)y * size + x];
                if (z <= previous) continue;
                float color[3] = {0.710f, 0.651f, 0.884f};
                int alpha = 255;
                if (textured) {
                    float u = 0, v = 0;
                    for (int k = 0; k < 3; ++k) {
                        u += weights[k] * result.uvs[t * 6 + k * 2];
                        v += weights[k] * result.uvs[t * 6 + k * 2 + 1];
                    }
                    const int tx = std::min(
                            result.texW - 1,
                            int(std::clamp(u, 0.0f, 1.0f) * result.texW));
                    const int ty =
                            std::min(result.texH - 1,
                                     int((1.0f - std::clamp(v, 0.0f, 1.0f)) *
                                         result.texH));
                    const uint8_t* texel = result.texRgba.constData() +
                                           ((size_t)ty * result.texW + tx) * 4;
                    for (int c = 0; c < 3; ++c) color[c] = texel[c] / 255.0f;
                    alpha = texel[3];
                } else if (result.vcolors.size() == verts.size()) {
                    for (int c = 0; c < 3; ++c) {
                        color[c] = 0;
                        for (int k = 0; k < 3; ++k)
                            color[c] +=
                                    weights[k] *
                                    result.vcolors[result.tris[t * 3 + k] * 3 +
                                                   c];
                    }
                }
                if (!alpha) continue;
                previous = z;
                img.setPixel(
                        x, y,
                        qRgba(int(std::clamp(color[0] * shade, 0.0f, 1.0f) *
                                  255),
                              int(std::clamp(color[1] * shade, 0.0f, 1.0f) *
                                  255),
                              int(std::clamp(color[2] * shade, 0.0f, 1.0f) *
                                  255),
                              alpha));
            }
    }
    return img;
}

bool InstantMeshWorker::runInference() {
    const QByteArray device = m_settings.device.toUtf8();
    aicore::runtime::DeviceTaskLock taskGuard(device.constData(),
                                              m_cancelToken);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[InstantMesh] Failed to acquire the inference device; "
                   "another task is running."));
        return false;
    }
    if (m_settings.modelPaths.size() < 3) {
        emit logMessage(QStringLiteral(
                "[InstantMesh] Model set incomplete (dino, transformer and "
                "synthesizer are required)."));
        return false;
    }

    QImage image(m_settings.inputPath);
    if (image.isNull()) {
        emit logMessage(tr("[InstantMesh] Cannot decode image: %1")
                                .arg(m_settings.inputPath));
        return false;
    }
    auto imageView = ecvAICoreRuntime::makeImageView(image);
    if (!imageView.data) {
        image = image.convertToFormat(QImage::Format_RGBA8888);
        imageView = ecvAICoreRuntime::makeImageView(image);
    }
    if (!imageView.data) return false;

    // Keep the UTF-8 payloads alive for the whole call.
    // Index contract: modelPaths is in aicore_instantmesh_model_paths field
    // order (cond, unet, vae, dino, transformer, synthesizer); omitted fields
    // stay as empty strings ("omit this model").
    QByteArray utf8Paths[6];
    auto keepAlive = [&](int i) -> const char* {
        utf8Paths[i] = m_settings.modelPaths.value(i).toUtf8();
        return utf8Paths[i].constData();
    };
    aicore_instantmesh_model_paths paths = AICORE_INSTANTMESH_MODEL_PATHS_INIT;
    paths.cond_gguf = keepAlive(0);
    paths.unet_gguf = keepAlive(1);
    paths.vae_gguf = keepAlive(2);
    paths.dino_gguf = keepAlive(3);
    paths.transformer_gguf = keepAlive(4);
    paths.synthesizer_gguf = keepAlive(5);

    if (m_settings.useRmbg && !resolveRmbgModel()) return false;
    QByteArray rmbgUtf8;
    if (m_settings.useRmbg && !m_settings.rmbgModelPath.isEmpty()) {
        rmbgUtf8 = m_settings.rmbgModelPath.toUtf8();
    }

    auto optionsOwner =
            std::unique_ptr<aicore_instantmesh_options,
                            decltype(&aicore_instantmesh_options_free)>(
                    aicore_instantmesh_options_new(),
                    aicore_instantmesh_options_free);
    auto* opts = optionsOwner.get();
    if (!opts) {
        emit logMessage(tr("[InstantMesh] Cannot create inference options."));
        return false;
    }
    if (opts) {
        aicore_instantmesh_options_set_device(
                opts, m_settings.device.toUtf8().constData());
        if (m_settings.threads > 0)
            aicore_instantmesh_options_set_threads(opts, m_settings.threads);
        if (m_settings.steps > 0)
            aicore_instantmesh_options_set_steps(opts, m_settings.steps);
        aicore_instantmesh_options_set_seed(opts, m_settings.seed);
        if (m_settings.guidance > 0)
            aicore_instantmesh_options_set_guidance(
                    opts, static_cast<float>(m_settings.guidance));
        if (m_settings.gridRes > 0)
            aicore_instantmesh_options_set_grid_res(opts, m_settings.gridRes);
        if (m_settings.gridScale > 0)
            aicore_instantmesh_options_set_grid_scale(
                    opts, static_cast<float>(m_settings.gridScale));
        if (m_settings.textureRes > 0)
            aicore_instantmesh_options_set_texture_res(opts,
                                                       m_settings.textureRes);
        aicore_instantmesh_options_set_bake_texture(
                opts, m_settings.bakeTexture ? 1 : 0);
        if (m_settings.useRmbg && !rmbgUtf8.isEmpty()) {
            aicore_instantmesh_options_set_rmbg_gguf(opts,
                                                     rmbgUtf8.constData());
        }
    }
    char err[512] = {};
    QElapsedTimer loadTimer;
    loadTimer.start();
    auto contextOwner = std::unique_ptr<aicore_instantmesh_ctx,
                                        decltype(&aicore_instantmesh_free)>(
            aicore_instantmesh_load_opts_ex(&paths, opts, err, sizeof(err)),
            aicore_instantmesh_free);
    const double loadMs = loadTimer.nsecsElapsed() / 1e6;
    optionsOwner.reset();
    auto* ctx = contextOwner.get();
    if (!ctx) {
        emit logMessage(tr("[InstantMesh] Model load failed: %1")
                                .arg(QString::fromUtf8(err)));
        return false;
    }

    aicore_instantmesh_generate_params params =
            AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
    params.steps = m_settings.steps;
    params.seed = m_settings.seed;
    params.guidance = static_cast<float>(m_settings.guidance);
    params.grid_res = m_settings.gridRes;
    params.grid_scale = static_cast<float>(m_settings.gridScale);
    params.texture_res = m_settings.textureRes;
    params.bake_texture = m_settings.bakeTexture ? 1 : 0;

    m_stageTimer.start();
    m_lastStage = -1;
    m_stageMs.clear();
    aicore_instantmesh_progress_fn progressFn = [](void* user, int stage,
                                                   int step, int total) {
        auto* self = static_cast<InstantMeshWorker*>(user);
        self->onProgress(stage, step, total);
    };
    auto meshOwner = std::unique_ptr<aicore_instantmesh_mesh,
                                     decltype(&aicore_instantmesh_mesh_free)>(
            aicore_instantmesh_generate_image_view(ctx, &imageView, &params,
                                                   progressFn, this, err,
                                                   sizeof(err)),
            aicore_instantmesh_mesh_free);
    auto* mesh = meshOwner.get();
    aicore_pipeline_timings timings{};
    if (mesh && aicore_instantmesh_last_pipeline_timings(ctx, &timings) != 0) {
        emit logMessage(
                tr("[InstantMesh] Common pipeline timings are unavailable."));
        return false;
    }
    const double elapsedMs = timings.e2e_ms;
    if (m_lastStage >= 0) {
        m_stageMs[m_lastStage] += static_cast<double>(m_stageTimer.elapsed());
    }

    if (!mesh) {
        emit logMessage(QStringLiteral("[InstantMesh] Generation failed: %1")
                                .arg(QString::fromUtf8(err)));
        return false;
    }

    InstantMeshRunResult result;
    result.sourceImage = m_settings.inputPath;
    result.generation = m_settings.generation;
    result.presetName = m_settings.presetName;
    result.totalRuntimeMs = elapsedMs;
    result.loadRuntimeMs = loadMs;
    result.timings = timings;
    result.backend = QString::fromUtf8(aicore_instantmesh_backend(ctx));
    result.quantization = m_settings.quantization;
    result.multiview = aicore_instantmesh_has_multiview(ctx);
    result.addResultToDb = m_settings.addResultToDb;
    result.stageMs = m_stageMs;

    const int nv = aicore_instantmesh_mesh_n_verts(mesh);
    const int nt = aicore_instantmesh_mesh_n_tris(mesh);
    if (nv > 0) {
        const float* v = aicore_instantmesh_mesh_verts(mesh);
        result.verts.resize(nv * 3);
        std::memcpy(result.verts.data(), v, sizeof(float) * nv * 3);
        const float* vc = aicore_instantmesh_mesh_vcolors(mesh);
        if (vc) {
            result.vcolors.resize(nv * 3);
            std::memcpy(result.vcolors.data(), vc, sizeof(float) * nv * 3);
        }
    }
    if (nt > 0) {
        const int* t = aicore_instantmesh_mesh_tris(mesh);
        result.tris.resize(nt * 3);
        std::memcpy(result.tris.data(), t, sizeof(int) * nt * 3);
        const float* uv = aicore_instantmesh_mesh_uvs(mesh);
        const uint8_t* tex = aicore_instantmesh_mesh_texture(mesh);
        const int tw = aicore_instantmesh_mesh_texture_w(mesh);
        const int th = aicore_instantmesh_mesh_texture_h(mesh);
        if (uv && tex && tw > 0 && th > 0) {
            result.uvs.resize(nt * 3 * 2);
            std::memcpy(result.uvs.data(), uv, sizeof(float) * nt * 3 * 2);
            result.texRgba.resize((size_t)tw * th * 4);
            std::memcpy(result.texRgba.data(), tex, result.texRgba.size());
            result.texW = tw;
            result.texH = th;
        }
    }
    meshOwner.reset();

    result.previewImage = renderResultPreview(result, 256);

    // Bake the UV-atlas textured GLB on the worker thread (CPU-bound, uses
    // the mesh + atlas already copied out of the AICore handle).
    double glbBakeMs = 0.0;
    if (!result.verts.isEmpty() && !result.tris.isEmpty()) {
        emit logMessage(QStringLiteral(
                "[InstantMesh] Baking textured GLB on the worker thread..."));
        QElapsedTimer glbTimer;
        glbTimer.start();
        char glbErr[512] = {0};
        int glbLen = 0;
        auto glbOwner = std::unique_ptr<
                uint8_t, decltype(&aicore_instantmesh_free_buffer)>(
                aicore_instantmesh_bake_glb(
                        result.verts.constData(), result.verts.size() / 3,
                        result.tris.constData(), result.tris.size() / 3,
                        result.uvs.isEmpty() ? nullptr : result.uvs.constData(),
                        result.texRgba.isEmpty() ? nullptr
                                                 : result.texRgba.constData(),
                        result.texW, result.texH, &glbLen, glbErr,
                        sizeof(glbErr)),
                aicore_instantmesh_free_buffer);
        auto* glb = glbOwner.get();
        if (glb && glbLen > 0) {
            result.glb = QByteArray(reinterpret_cast<const char*>(glb), glbLen);
            emit logMessage(
                    QStringLiteral("[InstantMesh] Textured GLB baked (%1 MB)")
                            .arg(glbLen / (1024.0 * 1024.0), 0, 'f', 1));
        } else {
            emit logMessage(
                    QStringLiteral("[InstantMesh] GLB bake failed (%1); the "
                                   "mesh stays available with vertex "
                                   "colours")
                            .arg(glbErr[0] ? QString::fromUtf8(glbErr)
                                           : QStringLiteral("unknown")));
        }
        glbOwner.reset();
        glbBakeMs = glbTimer.nsecsElapsed() / 1e6;
    }

    emit logMessage(QStringLiteral("[InstantMesh] Mesh %1 verts / %2 tris "
                                   "generated in %3 ms (backend %4, %5 "
                                   "chain, %6)")
                            .arg(nv)
                            .arg(nt)
                            .arg(elapsedMs, 0, 'f', 0)
                            .arg(result.backend)
                            .arg(result.quantization)
                            .arg(result.multiview ? QStringLiteral("6-view "
                                                                   "diffusion")
                                                  : QStringLiteral("replica"
                                                                   "ted "
                                                                   "views")));

    contextOwner.reset();

    if (aicore_cancel_requested()) return false;
    result.exportRuntimeMs = glbBakeMs;
    emitRunSummary(result, glbBakeMs);
    emit resultReady(result);
    return true;
}

void InstantMeshWorker::emitRunSummary(const InstantMeshRunResult& result,
                                       double glbBakeMs) {
    struct StageInfo {
        int stage;
        const char* name;
        const char* hint;
    };
    static const StageInfo kStages[] = {
            {AICORE_INSTANTMESH_STAGE_PREPROCESS, "preprocess",
             "image preprocess"},
            {AICORE_INSTANTMESH_STAGE_RMBG, "rmbg", "AI background removal"},
            {AICORE_INSTANTMESH_STAGE_MULTIVIEW, "multiview",
             "Zero123++ diffusion (per-step)"},
            {AICORE_INSTANTMESH_STAGE_VAE_DECODE, "vae_decode",
             "latent -> 6-view grid"},
            {AICORE_INSTANTMESH_STAGE_ENCODE, "encode",
             "DINO conditioning encode"},
            {AICORE_INSTANTMESH_STAGE_TRIPLANE, "triplane",
             "TriplaneTransformer"},
            {AICORE_INSTANTMESH_STAGE_SYNTH, "synth",
             "OSGDecoder sdf/deform/weights"},
            {AICORE_INSTANTMESH_STAGE_MESH, "mesh",
             "FlexiCubes extraction (CPU)"},
            {AICORE_INSTANTMESH_STAGE_TEXTURE, "texture",
             "UV unwrap + net_rgb bake"},
    };

    auto tag = [](const char* text) {
        return QStringLiteral("[InstantMesh] %1: ")
                .arg(QString::fromLatin1(text).leftJustified(11));
    };

    emit logMessage(QStringLiteral(
            "[InstantMesh] ---- Run summary ------------------------------"
            "---------------"));
    emit logMessage(tag("Preset") + result.presetName +
                    QStringLiteral(" (%1)").arg(result.quantization));
    emit logMessage(tag("Device") + result.backend);
    emit logMessage(tag("Views") +
                    (result.multiview ? QStringLiteral("Zero123++ diffusion")
                                      : QStringLiteral("replicated input")));
    for (const StageInfo& s : kStages) {
        const auto it = result.stageMs.constFind(s.stage);
        if (it != result.stageMs.constEnd() && it.value() > 0.0) {
            emit logMessage(tag(s.name) +
                            QStringLiteral("%1 ms  (%2)")
                                    .arg(it.value(), 0, 'f', 0)
                                    .arg(QString::fromLatin1(s.hint)));
        }
    }
    if (glbBakeMs > 0.0) {
        emit logMessage(tag("glb_bake") + QStringLiteral("%1 ms  (GLB encode)")
                                                  .arg(glbBakeMs, 0, 'f', 0));
    }
    emit logMessage(tag("Total") + QStringLiteral("%1 ms").arg(
                                           result.totalRuntimeMs, 0, 'f', 0));
    emit logMessage(QStringLiteral(
            "[InstantMesh] --------------------------------------------"
            "---------------------"));
}

#endif  // AICore_ENABLED
