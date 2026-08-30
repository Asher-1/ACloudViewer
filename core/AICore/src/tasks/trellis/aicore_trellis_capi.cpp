// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AICore C ABI wrapper around the trellis2 C++ pipeline (see
// aicore/trellis_capi.h). Port of the upstream trellis-ggml trellis2_capi.cpp;
// image decode now uses Qt QImage, RMBG delegates to the in-tree rmbg task,
// and the upstream T2_* / TRELLIS2_* environment variables became explicit
// options / generate parameters (AICore reads no environment variables for
// logic control).

#include <QFileInfo>
#include <QImage>

#include "aicore/backend_capi.h"
#include "aicore/trellis_capi.h"
#include "common/aicore_log.hpp"
#include "common/capi_utils.hpp"
#include "common/data_root_util.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "flexible_dual_grid.h"
#include "marching_cubes.h"
#include "mesh_export.h"
#include "pbr_utils.h"
#include "trellis2.h"
#ifdef TRELLIS2_HAVE_MESH_TO_GRID
#include "mesh_to_dual_grid.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void copy_err(char *err, int err_len, const std::string &msg) {
    if (!err || err_len <= 0) return;
    std::snprintf(err, (size_t)err_len, "%s", msg.c_str());
}

// Rough peak VRAM the shape decode's transient buffers need at each tier
// (measured on the reference image: ~2.4 GB at 512³, ~6.75 GB for the 1024³
// level-3 conv output; rounded up for headroom). Mesh-dependent, so treated as
// a threshold, not an exact reservation — the free-flows fallback then gives a
// several-GB cushion if the estimate is low.
size_t decode_vram_peak(int pipeline_type) {
    const double GB = (double)(1ULL << 30);
    return (size_t)((pipeline_type == AICORE_TRELLIS_PIPE_1024 ? 7.5 : 3.0) *
                    GB);
}

}  // namespace

struct aicore_trellis_ctx {
    trellis2_dino_model *dino = nullptr;
    trellis2_ss_flow_model *flow = nullptr;
    trellis2_ss_dec_model *dec = nullptr;
    trellis2_slat_flow_model *slat = nullptr;      // 512 model (fine path)
    trellis2_slat_flow_model *slat_hr = nullptr;   // 1024 model (cascade)
    trellis2_shape_dec_model *shapedec = nullptr;  // shared by 512 + cascade
    std::string backend;
    // Resolved (VRAM-aware) device used for every model load: e.g. the user
    // asked for "auto"/"cuda" but no GPU can hold this preset, so all models
    // load on "vulkan" or "cpu" instead of aborting on a CUDA OOM.
    std::string device;
    std::string device_family;  // backend family of `device` ("cuda"/...)
    std::string backend_note;   // user-facing downgrade reason (empty = none)
    bool fine = false;          // 512 dual-grid available
    bool cascade = false;       // 1024 cascade available
    bool texture =
            false;  // PBR texturing available (shape_enc + tex_dec + tex_flow)
    bool shapedec_gpu =
            false;  // shape decoder placed on the GPU (VRAM permitting)
    std::string last_error;
#ifdef TRELLIS2_HAVE_RMBG
    trellis2_rmbg_model *rmbg = nullptr;  // optional AI background removal
#endif
    // gguf paths, so the flow DiTs can be freed to make VRAM room for a GPU
    // decode and lazily reloaded on the next generate (see ensure_decode_vram).
    std::string ss_flow_path, slat_path, slat_hr_path;
    // Texture-stage gguf paths. The tex models (~4 GB) are loaded lazily inside
    // the texture stage — after the geometry flow DiTs are freed — and freed
    // again, so they never coexist in VRAM with the geometry flows.
    std::string shapeenc_path, texdec_path, texflow_path, texflow_hr_path;
};

// A generated mesh: verts (3/vertex), normals (3/vertex), tris (3/tri), and
// optional per-vertex PBR (6/vertex: base_color rgb, metallic, roughness,
// alpha).
struct aicore_trellis_mesh {
    std::vector<float> verts;
    std::vector<float> normals;
    std::vector<int> tris;
    std::vector<float> pbr;  // empty when untextured
    int grid_res = 0;
    std::vector<float> grid_feats;     // 7 * nvox dual-grid decode output
    std::vector<int32_t> grid_coords;  // 3 * nvox voxel indices
    // Optional AI background-removal result (RGBA at decoded-input size);
    // empty when no RMBG model ran.
    std::vector<uint8_t> rmbg_rgba;
    int rmbg_w = 0, rmbg_h = 0;
};

namespace {

// Reload any flow DiT freed by a previous GPU decode (see ensure_decode_vram).
// No-op the common case where nothing was freed (pointers still set).
bool reload_flows(aicore_trellis_ctx *p, std::string &e) {
    if (!p->flow && !p->ss_flow_path.empty()) {
        p->flow = trellis2_ss_flow_load(p->ss_flow_path.c_str(), true, &e,
                                        p->device.c_str());
        if (!p->flow) return false;
    }
    if (!p->slat && !p->slat_path.empty()) {
        p->slat = trellis2_slat_flow_load(p->slat_path.c_str(), true, &e,
                                          p->device.c_str());
        if (!p->slat) return false;
    }
    if (!p->slat_hr && !p->slat_hr_path.empty()) {
        p->slat_hr = trellis2_slat_flow_load(p->slat_hr_path.c_str(), true, &e,
                                             p->device.c_str());
        if (!p->slat_hr) return false;
    }
    return true;
}

// Before a GPU shape decode, make room: if the decode's transient buffers would
// not fit in current free VRAM, free the flow DiTs (all finished by decode
// time) to reclaim their ~5-7 GB. reload_flows() brings them back on the next
// generate. No-op for a CPU decoder or when the decode already fits.
void ensure_decode_vram(aicore_trellis_ctx *p, int pipeline_type) {
    if (!p->shapedec_gpu) return;
    if (trellis2_gpu_free_vram(p->device_family.c_str()) <
        decode_vram_peak(pipeline_type)) {
        trellis2_ss_flow_free(p->flow);
        p->flow = nullptr;
        trellis2_slat_flow_free(p->slat);
        p->slat = nullptr;
        trellis2_slat_flow_free(p->slat_hr);
        p->slat_hr = nullptr;
    }
}

// Shared tail: shape_enc -> tex_flow -> tex_dec -> vertex PBR sample.
bool run_texture_stage_core(aicore_trellis_ctx *p,
                            const float *in6,
                            int nvox,
                            const int32_t *coords,
                            const std::vector<float> &mesh_verts,
                            int grid,
                            int pt,
                            const trellis2_dino_cond &cond,
                            uint64_t seed,
                            int texture_steps,
                            aicore_trellis_progress_fn progress,
                            void *user,
                            std::vector<float> &pbr_out,
                            std::string &e) {
    if (nvox <= 0 || !in6 || !coords) {
        e = "invalid shape-encoder input";
        return false;
    }

    if (progress)
        progress(user, AICORE_TRELLIS_STAGE_TEXTURE, 0,
                 texture_steps > 0 ? texture_steps : 12);
    trellis2_shape_enc_model *enc = trellis2_shape_enc_load(
            p->shapeenc_path.c_str(), true, &e, p->device.c_str());
    if (!enc) {
        e = "shape_enc load: " + e;
        return false;
    }
    std::vector<float> shape_slat;
    std::vector<int32_t> lat_coords;
    std::vector<trellis2_subdiv_level> subs;
    bool ok = trellis2_shape_enc_encode(enc, in6, nvox, coords, shape_slat,
                                        lat_coords, subs, nullptr, &e);
    trellis2_shape_enc_free(enc);
    if (!ok) {
        e = "shape encode: " + e;
        return false;
    }

    const int Nl = (int)(lat_coords.size() / 3);
    if (Nl <= 0 || shape_slat.size() != (size_t)Nl * 32) {
        e = "invalid generated shape SLAT";
        return false;
    }
    if (subs.empty()) {
        e = "missing shape decoder subdivision guide";
        return false;
    }

    if (pt == AICORE_TRELLIS_PIPE_1024 && p->texflow_hr_path.empty()) {
        e = "1024 texture flow model is not loaded";
        return false;
    }
    const std::string &fp = pt == AICORE_TRELLIS_PIPE_1024 ? p->texflow_hr_path
                                                           : p->texflow_path;
    trellis2_slat_flow_model *flow =
            trellis2_slat_flow_load(fp.c_str(), true, &e, p->device.c_str());
    if (!flow) {
        e = "tex_flow load: " + e;
        return false;
    }
    std::vector<float> tex_slat((size_t)Nl * 32);
    trellis2_ss_sampler_params tp;
    tp.steps = texture_steps > 0 ? texture_steps : 12;
    tp.guidance_strength = 1.0f;
    tp.guidance_rescale = 0.0f;
    tp.guidance_interval_min = 0.6f;
    tp.guidance_interval_max = 0.9f;
    tp.rescale_t = 3.0f;
    tp.seed = seed ^ 0x7e00ULL;
    tp.verbose = false;
    struct tex_progress_ctx {
        aicore_trellis_progress_fn fn;
        void *user;
    } pc{progress, user};
    if (progress) {
        progress(user, AICORE_TRELLIS_STAGE_TEXTURE, 0, tp.steps);
        tp.progress = [](void *u, int step, int total) {
            auto *c = (tex_progress_ctx *)u;
            c->fn(c->user, AICORE_TRELLIS_STAGE_TEXTURE, step, total);
        };
        tp.progress_user = &pc;
    }
    ok = trellis2_slat_flow_sample_tex(
            flow, Nl, lat_coords.data(), cond.data.data(), (int)cond.tokens(),
            (int)cond.channels(), shape_slat.data(), &tp, nullptr,
            /*denormalize*/ true, tex_slat.data(), &e);
    trellis2_slat_flow_free(flow);
    if (!ok) {
        e = "tex sample: " + e;
        return false;
    }

    trellis2_shape_dec_model *texdec = trellis2_tex_dec_load(
            p->texdec_path.c_str(), true, &e, p->device.c_str());
    if (!texdec) {
        e = "tex_dec load: " + e;
        return false;
    }
    std::vector<float> pbr;
    std::vector<int32_t> pbr_coords;
    ok = trellis2_tex_dec_decode(texdec, tex_slat.data(), Nl, lat_coords.data(),
                                 subs, pbr, pbr_coords, &e);
    trellis2_shape_dec_free(texdec);
    if (!ok) {
        e = "tex decode: " + e;
        return false;
    }

    const int M = (int)(pbr_coords.size() / 3);
    if (M <= 0 || pbr.size() != (size_t)M * 6) {
        e = "texture decoder returned an invalid PBR volume";
        return false;
    }
    const int nv = (int)(mesh_verts.size() / 3);
    std::vector<float> query((size_t)nv * 3), weights((size_t)nv);
    for (int v = 0; v < nv; ++v)
        for (int c = 0; c < 3; ++c)
            query[(size_t)v * 3 + c] =
                    (mesh_verts[(size_t)v * 3 + c] + 0.5f) * grid;
    pbr_out.resize((size_t)nv * 6);
    t2pbr::sample_sparse_trilinear(pbr.data(), M, 6, pbr_coords.data(),
                                   query.data(), nv, pbr_out.data(),
                                   weights.data());
    for (int v = 0; v < nv; ++v) {
        float *d = pbr_out.data() + (size_t)v * 6;
        if (weights[(size_t)v] <= 1e-6f) {
            d[0] = d[1] = d[2] = 0.5f;
            d[3] = 0.0f;
            d[4] = 0.5f;
            d[5] = 1.0f;
        } else {
            for (int c = 0; c < 6; ++c)
                d[c] = std::max(0.0f, std::min(1.0f, d[c]));
        }
    }

    if (t2pbr::is_collapsed_saturated(pbr_out.data(), nv)) {
        pbr_out.clear();
        e = "texture decoder produced a collapsed saturated material";
        return false;
    }
    return true;
}

// PBR-texture stage on the freshly decoded dual grid (integrated / sidecar
// path).
bool run_texture_stage(aicore_trellis_ctx *p,
                       const std::vector<float> &dec_feats,
                       const std::vector<int32_t> &dec_coords,
                       const std::vector<float> &mesh_verts,
                       int grid,
                       int pt,
                       const trellis2_dino_cond &cond,
                       uint64_t seed,
                       int texture_steps,
                       aicore_trellis_progress_fn progress,
                       void *user,
                       std::vector<float> &pbr_out,
                       std::string &e) {
    const int nvox = (int)(dec_coords.size() / 3);
    if (nvox <= 0 || dec_feats.size() != (size_t)nvox * 7) {
        e = "invalid decoded dual grid";
        return false;
    }

    const float mg = 0.5f;
    std::vector<float> in6((size_t)nvox * 6);
    for (int v = 0; v < nvox; ++v) {
        const float *f = dec_feats.data() + (size_t)v * 7;
        for (int c = 0; c < 3; ++c) {
            const float s = 1.0f / (1.0f + std::exp(-f[c]));
            in6[(size_t)v * 6 + c] = (1.0f + 2.0f * mg) * s - mg;
            in6[(size_t)v * 6 + 3 + c] = f[3 + c] > 0.0f ? 1.0f : 0.0f;
        }
    }
    return run_texture_stage_core(p, in6.data(), nvox, dec_coords.data(),
                                  mesh_verts, grid, pt, cond, seed,
                                  texture_steps, progress, user, pbr_out, e);
}

// Arbitrary-mesh path: QEF dual grid -> shape_enc (matches
// dump_texture_reference.py).
bool run_texture_stage_qef(aicore_trellis_ctx *p,
                           const float *verts,
                           int n_verts,
                           const int *tris,
                           int n_tris,
                           int grid,
                           const std::vector<float> &mesh_verts,
                           int pt,
                           const trellis2_dino_cond &cond,
                           uint64_t seed,
                           int texture_steps,
                           aicore_trellis_progress_fn progress,
                           void *user,
                           std::vector<float> &pbr_out,
                           std::vector<int32_t> &qef_coords,
                           std::string &e) {
#ifndef TRELLIS2_HAVE_MESH_TO_GRID
    (void)verts;
    (void)n_verts;
    (void)tris;
    (void)n_tris;
    (void)grid;
    (void)mesh_verts;
    (void)pt;
    (void)cond;
    (void)seed;
    (void)texture_steps;
    (void)progress;
    (void)user;
    (void)pbr_out;
    (void)qef_coords;
    e = "mesh->grid QEF was not built (Eigen missing)";
    return false;
#else
    if (!verts || n_verts <= 0 || !tris || n_tris <= 0 || grid <= 0) {
        e = "invalid mesh for QEF";
        return false;
    }
    AICORE_LOG_INFO("trellis", "QEF mesh->dual grid @ %d^3 (%d tris)", grid,
                    n_tris);
    mtdg::Result qef = mtdg::mesh_to_flexible_dual_grid(
            verts, n_verts, tris, n_tris, grid, 1.f, 0.2f, 1e-2f);
    const int nvox = (int)(qef.coords.size() / 3);
    if (nvox <= 0) {
        e = "QEF returned empty dual grid";
        return false;
    }
    AICORE_LOG_INFO("trellis", "QEF: %d active voxels", nvox);

    std::vector<float> in6((size_t)nvox * 6);
    qef_coords = qef.coords;
    for (int v = 0; v < nvox; ++v) {
        for (int c = 0; c < 3; ++c) {
            in6[(size_t)v * 6 + c] =
                    qef.dual_verts[(size_t)v * 3 + c] * (float)grid -
                    (float)qef.coords[(size_t)v * 3 + c];
            in6[(size_t)v * 6 + 3 + c] =
                    qef.intersected[(size_t)v * 3 + c] ? 1.f : 0.f;
        }
    }
    return run_texture_stage_core(p, in6.data(), nvox, qef.coords.data(),
                                  mesh_verts, grid, pt, cond, seed,
                                  texture_steps, progress, user, pbr_out, e);
#endif
}

// ── live intermediate 3D previews (voxel sets + mesh keyframes) ──────
//
// Port of the upstream trellis2_capi.cpp preview machinery: self-describing
// blobs the host streams to its viewer.
//   "T2VOX01"  magic[8], u32 res, u32 nvox, u16[3*nvox] coords in [0,res)
//   "T2MESH01" magic[8], u32 nv, u32 nt, f32[3nv] verts, f32[3nv] normals,
//              i32[3nt] tris (little-endian)

// Collect the occupied cells (logit > 0) of a dense [res^3] occupancy grid.
void collect_occupied(const float *occ, int res, std::vector<int32_t> &cells) {
    cells.clear();
    for (int x = 0; x < res; ++x)
        for (int y = 0; y < res; ++y)
            for (int z = 0; z < res; ++z) {
                if (occ[((size_t)x * res + y) * res + z] > 0.0f) {
                    cells.push_back(x);
                    cells.push_back(y);
                    cells.push_back(z);
                }
            }
}

// Pack a flat [x,y,z,...] cell list into a T2VOX01 blob and hand it to the
// host callback. Best-effort: never fatal.
void emit_voxels(aicore_trellis_preview_fn fn,
                 void *user,
                 int stage,
                 int step,
                 int total,
                 int res,
                 const std::vector<int32_t> &cells) {
    if (!fn) return;
    const uint32_t nvox = (uint32_t)(cells.size() / 3);
    const uint32_t r = (uint32_t)res;
    std::vector<uint8_t> blob(16 + (size_t)nvox * 3 * 2);
    std::memcpy(blob.data(), "T2VOX01", 8);  // 7 chars + NUL
    std::memcpy(blob.data() + 8, &r, 4);
    std::memcpy(blob.data() + 12, &nvox, 4);
    uint8_t *dst = blob.data() + 16;
    for (uint32_t i = 0; i < nvox * 3; ++i) {
        const uint16_t c = (uint16_t)cells[i];
        std::memcpy(dst, &c, 2);
        dst += 2;
    }
    fn(user, stage, step, total, blob.data(), (int)blob.size());
}

// One shape-flow stage's captured intermediate x_0 latents (denormalized)
// plus the scaffold they sit on, awaiting post-decode replay.
struct kf_capture {
    int stage = 0;    // SLAT_FLOW or SLAT_FLOW_HR
    int res_in = 0;   // scaffold resolution (32 LR / 64 HR)
    int levels = 0;   // upsample levels to the ~128^3 keyframe grid
    int stride = 1;   // capture every `stride` steps (plus the last)
    int channels = 0; // latent channels (32)
    const float *norm_mean = nullptr;
    const float *norm_std = nullptr;
    std::vector<int32_t> coords;              // scaffold coords, copied once
    std::vector<int> steps, totals;           // per capture, for labelling
    std::vector<std::vector<float>> latents;  // denormalized [L*channels] each
};

// Sampler preview trampoline: denormalize + stash the step's x_0 estimate.
void kf_capture_cb(void *u, int step, int total, const float *latent, int n) {
    auto *c = (kf_capture *)u;
    if (c->stride < 1) return;
    if (step != total && (step % c->stride) != 0) return;  // stride + last
    const int C = c->channels;
    std::vector<float> den((size_t)n);
    for (int i = 0; i < n; ++i)
        den[i] = latent[i] * c->norm_std[i % C] + c->norm_mean[i % C];
    c->steps.push_back(step);
    c->totals.push_back(total);
    c->latents.push_back(std::move(den));
}

// Replay captured latents into coarse MC-mesh keyframes and stream each. Runs
// after the final decode (decoder owns VRAM). Best-effort: skips on failure.
void emit_keyframes(trellis2_shape_dec_model *dec,
                    kf_capture &kf,
                    aicore_trellis_preview_fn fn,
                    void *user) {
    if (!fn || kf.latents.empty() || !dec) return;
    const int L = (int)(kf.coords.size() / 3);
    const int tgt = kf.res_in << kf.levels;  // keyframe grid (128^3)
    for (size_t k = 0; k < kf.latents.size(); ++k) {
        std::vector<int32_t> up;
        std::string e;
        if (!trellis2_shape_dec_upsample(dec, kf.latents[k].data(), L,
                                         kf.coords.data(), kf.levels, up, &e))
            continue;
        std::vector<float> field((size_t)tgt * tgt * tgt, 0.0f);
        for (size_t i = 0; i + 2 < up.size(); i += 3) {
            const int x = up[i], y = up[i + 1], z = up[i + 2];
            if (x >= 0 && x < tgt && y >= 0 && y < tgt && z >= 0 && z < tgt)
                field[((size_t)x * tgt + y) * tgt + z] = 1.0f;
        }
        mc::Mesh m = mc::extract(field.data(), tgt, tgt, tgt, 0.5f);
        if (m.verts.empty()) continue;
        const float inv = 1.0f / (float)tgt;
        for (auto &v : m.verts) v = v * inv - 0.5f;  // -> centered unit cube
        const uint32_t nv = (uint32_t)(m.verts.size() / 3);
        const uint32_t nt = (uint32_t)(m.tris.size() / 3);
        std::vector<uint8_t> blob(16 + (size_t)nv * 24 + (size_t)nt * 12);
        std::memcpy(blob.data(), "T2MESH01", 8);
        std::memcpy(blob.data() + 8, &nv, 4);
        std::memcpy(blob.data() + 12, &nt, 4);
        size_t o = 16;
        std::memcpy(blob.data() + o, m.verts.data(), (size_t)nv * 12);
        o += (size_t)nv * 12;
        std::memcpy(blob.data() + o, m.normals.data(), (size_t)nv * 12);
        o += (size_t)nv * 12;
        std::memcpy(blob.data() + o, m.tris.data(), (size_t)nt * 12);
        fn(user, kf.stage, kf.steps[k], kf.totals[k], blob.data(),
           (int)blob.size());
    }
}

// Context for the SS-sampler preview trampoline: decode the handed-out x_0
// estimate into a 64^3 occupancy and stream it as voxels (stride-gated).
struct ss_preview_ctx {
    aicore_trellis_preview_fn fn = nullptr;
    void *user = nullptr;
    trellis2_ss_dec_model *dec = nullptr;
    int res = 0;
    int stride = 1;
    std::vector<float> *occ = nullptr;  // scratch [res^3], caller-owned
};

}  // namespace

extern "C" {

int aicore_trellis_abi_version(void) { return 2; }

// ─────────────────────────────────────────────────────────────────────────
// Options builder
// ─────────────────────────────────────────────────────────────────────────

struct aicore_trellis_options {
    std::string device = "auto";
    int32_t threads = 0;
    std::string rmbg_gguf;
    std::string shape_dec_placement = "auto";
    bool sdpa_exact = false;
    bool sdpa_flash = false;
    bool timing = false;
};

aicore_trellis_options *aicore_trellis_options_new(void) {
    return new aicore_trellis_options();
}

void aicore_trellis_options_free(aicore_trellis_options *opts) { delete opts; }

void aicore_trellis_options_set_device(aicore_trellis_options *opts,
                                       const char *device) {
    if (opts && device) opts->device = device;
}

void aicore_trellis_options_set_threads(aicore_trellis_options *opts,
                                        int n_threads) {
    if (opts) opts->threads = n_threads;
}

void aicore_trellis_options_set_rmbg_gguf(aicore_trellis_options *opts,
                                          const char *rmbg_gguf) {
    if (opts && rmbg_gguf) opts->rmbg_gguf = rmbg_gguf;
}

void aicore_trellis_options_set_shape_dec_placement(
        aicore_trellis_options *opts, const char *placement) {
    if (opts && placement) opts->shape_dec_placement = placement;
}

void aicore_trellis_options_set_sdpa_exact(aicore_trellis_options *opts,
                                           int exact) {
    if (opts) opts->sdpa_exact = exact != 0;
}

void aicore_trellis_options_set_sdpa_flash(aicore_trellis_options *opts,
                                           int flash) {
    if (opts) opts->sdpa_flash = flash != 0;
}

void aicore_trellis_options_set_timing(aicore_trellis_options *opts,
                                       int enabled) {
    if (opts) opts->timing = enabled != 0;
}

// ─────────────────────────────────────────────────────────────────────────
// VRAM-aware device resolution
// ─────────────────────────────────────────────────────────────────────────
// TRELLIS is memory-hungry (the f16 512 fine path wants ~16 GB) and ggml's
// CUDA backend aborts the process when a device allocation fails.  A GPU
// whose free VRAM cannot hold this preset must therefore be skipped *before*
// any weights are allocated: "auto" slides CUDA → Vulkan → CPU instead of
// crashing on the first cuMemCreate.

namespace {

// Weights + graph activations ≈ 2x the weight bytes; plus a fixed base
// margin for the CUDA/Vulkan context, graph buffers and fragmentation.
constexpr double kWeightsVramFactor = 2.0;
constexpr size_t kVramBaseMargin = 1u << 30;  // 1 GiB

// GGUF payloads are stored uncompressed, so the file size ≈ weight bytes.
size_t fileSizeOrZero(const char *path) {
    if (!path || !path[0]) return 0;
    const QFileInfo fi(QString::fromUtf8(path));
    return fi.exists() ? (size_t)fi.size() : 0;
}

// Model weights that coexist on the GPU while the pipeline is loaded.  The
// occupancy decoder always runs on the CPU and the texture models are loaded
// lazily after the flow DiTs are freed, so neither is counted here.
size_t coresident_weights_bytes(const aicore_trellis_model_paths *paths,
                                const aicore_trellis_options *opts) {
    if (!paths) return 0;
    size_t total = 0;
    total += fileSizeOrZero(paths->dino_gguf);
    total += fileSizeOrZero(paths->ss_flow_gguf);
    total += fileSizeOrZero(paths->slat_flow_gguf);
    total += fileSizeOrZero(paths->slat_hr_flow_gguf);
    total += fileSizeOrZero(paths->shape_dec_gguf);
    if (opts && !opts->rmbg_gguf.empty()) {
        total += fileSizeOrZero(opts->rmbg_gguf.c_str());
    }
    return total;
}

// Free VRAM on the want_idx-th GPU device of `family` (matching rules as in
// ggml_common::find_gpu_backend). Returns 0 when no such device exists.
size_t familyFreeVram(const std::string &family, int want_idx) {
    ggml_common::load_backends_once();
    const std::string want_reg = ggml_common::normalize_backend_name(family);
    int gpu_idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char *reg =
                ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (!reg || ggml_common::to_lower(reg) != want_reg) continue;
        if (gpu_idx++ != want_idx) continue;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        return free;
    }
    return 0;
}

}  // namespace

// Resolve the requested device against the actual free VRAM.  Returns the
// device string every model load should use ("cpu" when no GPU can hold the
// pipeline); `note` collects a user-facing explanation when the request was
// downgraded.  `requested` may carry an index ("cuda:1"), which is
// preserved on the winning family.
std::string resolveVramAwareDevice(const std::string &requested,
                                   const aicore_trellis_model_paths *paths,
                                   const aicore_trellis_options *opts,
                                   std::string *note) {
    note->clear();
    std::string fam;
    int want_idx = 0;
    ggml_common::parse_device(requested, fam, want_idx);
    if (fam == "cpu") return "cpu";

    const size_t weights = coresident_weights_bytes(paths, opts);
    const size_t need =
            (size_t)((double)weights * kWeightsVramFactor) + kVramBaseMargin;

    // Candidate order: the explicitly requested family first, then the
    // platform auto order (CUDA → Vulkan on Linux/Windows, Metal on macOS).
    std::vector<std::string> candidates;
    const bool generic = fam.empty() || fam == "auto" || fam == "gpu";
    if (!generic) candidates.push_back(fam);
    for (const char *const *p = ggml_common::auto_backend_ids(); *p; ++p) {
        if (std::find(candidates.begin(), candidates.end(), *p) ==
            candidates.end()) {
            candidates.push_back(*p);
        }
    }

    // Remember the first-choice device's free VRAM so a downgrade to
    // another family (or to CPU) can be explained to the user.  The whole
    // pipeline shares the resolved device — no model ever picks a backend
    // on its own (see the load call sites in aicore_trellis_load_opts).
    const std::string &preferred = candidates.front();
    const size_t preferred_free = familyFreeVram(preferred, want_idx);

    for (const std::string &c : candidates) {
        const size_t free = familyFreeVram(c, want_idx);
        if (free >= need) {
            if (c != preferred) {
                // Downgraded to another GPU family: make it visible instead
                // of silently running somewhere else than requested.
                char buf[320];
                if (preferred_free > 0 && preferred_free < need) {
                    std::snprintf(buf, sizeof(buf),
                                  "'%s' VRAM too small for this preset "
                                  "(~%.1f GiB needed, %.1f GiB free) — "
                                  "using %s instead.",
                                  preferred.c_str(), (double)need / (1u << 30),
                                  (double)preferred_free / (1u << 30),
                                  c.c_str());
                } else {
                    std::snprintf(buf, sizeof(buf),
                                  "device '%s' is unavailable — using %s "
                                  "instead.",
                                  preferred.c_str(), c.c_str());
                }
                *note = buf;
            }
            return want_idx > 0 ? c + ":" + std::to_string(want_idx) : c;
        }
    }

    // No GPU (CUDA/Vulkan, incl. iGPU) can hold the pipeline: fall back to
    // CPU with an explanation.  The user's GGUF selection is never changed
    // automatically — only the device downgrades; fitting a GPU requires a
    // manual switch to q8 models or a coarser preset.
    size_t best = 0;
    for (const std::string &c : candidates) {
        best = std::max(best, familyFreeVram(c, want_idx));
    }
    char buf[360];
    std::snprintf(buf, sizeof(buf),
                  "GPU VRAM too small for this preset (~%.1f GiB needed, "
                  "best available %.1f GiB) — falling back to CPU "
                  "inference (your GGUF selection is unchanged). Switching "
                  "to q8 models or the coarse preset would fit the GPU.",
                  (double)need / (1u << 30), (double)best / (1u << 30));
    *note = buf;
    return "cpu";
}

// ─────────────────────────────────────────────────────────────────────────
// Pipeline load / free / introspection
// ─────────────────────────────────────────────────────────────────────────

aicore_trellis_ctx *aicore_trellis_load_opts(
        const aicore_trellis_model_paths *paths,
        const aicore_trellis_options *opts) {
    if (!paths || !paths->dino_gguf || !paths->dino_gguf[0] ||
        !paths->ss_flow_gguf || !paths->ss_flow_gguf[0] ||
        !paths->ss_dec_gguf || !paths->ss_dec_gguf[0]) {
        return nullptr;
    }
    const std::string requested_device =
            opts ? opts->device : std::string("auto");
    trellis2_set_threads(opts ? opts->threads : 0);
    trellis2_set_sdpa_exact(opts ? opts->sdpa_exact : false);
    trellis2_set_sdpa_flash(opts ? opts->sdpa_flash : false);
    trellis2_set_timing(opts ? opts->timing : false);

    std::string e;
    auto *p = new aicore_trellis_ctx();

    // Decide the compute device before any model load (see
    // resolveVramAwareDevice): a small-VRAM card must never reach a CUDA
    // allocation, which would abort the whole process.  Every model of the
    // pipeline (dino / flows / decoders / RMBG / texture stage) loads with
    // this same resolved device, so a downgrade applies to the whole
    // pipeline, never to individual models.
    p->device = resolveVramAwareDevice(requested_device, paths, opts,
                                       &p->backend_note);
    {
        std::string f;
        int idx = 0;
        ggml_common::parse_device(p->device, f, idx);
        p->device_family = f;
    }
    AICORE_LOG_INFO("[trellis] ",
                    "device request '%s' resolved to '%s' (shared by all "
                    "pipeline models)\n",
                    requested_device.c_str(), p->device.c_str());
    if (!p->backend_note.empty()) {
        AICORE_LOG_WARN("[trellis] ", "[WARN] %s\n", p->backend_note.c_str());
    }

    const char *rmbg_gguf = opts ? opts->rmbg_gguf.c_str() : "";
#ifdef TRELLIS2_HAVE_RMBG
    if (rmbg_gguf && rmbg_gguf[0]) {
        p->rmbg = trellis2_rmbg_load(rmbg_gguf, p->device.c_str(), &e);
        if (!p->rmbg) {
            p->last_error = "rmbg: " + e;
            aicore_trellis_free(p);
            return nullptr;
        }
    }
#endif
    p->dino = trellis2_dino_load(paths->dino_gguf, true, &e, p->device.c_str());
    if (!p->dino) {
        p->last_error = "dino: " + e;
        aicore_trellis_free(p);
        return nullptr;
    }
    // Free VRAM before the flow DiTs are loaded == the VRAM reclaimable by
    // freeing them again at decode time. Drives the shape-decoder placement.
    const size_t free_pre_flows =
            trellis2_gpu_free_vram(p->device_family.c_str());
    p->flow = trellis2_ss_flow_load(paths->ss_flow_gguf, true, &e,
                                    p->device.c_str());
    if (!p->flow) {
        p->last_error = "ss_flow: " + e;
        aicore_trellis_free(p);
        return nullptr;
    }
    // The SS occupancy decoder uses a genuine dense CONV_3D, for which ggml has
    // no CUDA kernel, so it stays on the CPU (it is only ~3 s / 4 % anyway).
    p->dec = trellis2_ss_dec_load(paths->ss_dec_gguf, true, &e, "cpu");
    if (!p->dec) {
        p->last_error = "ss_dec: " + e;
        aicore_trellis_free(p);
        return nullptr;
    }

    auto present = [](const char *s) { return s && s[0]; };

    if (present(paths->slat_flow_gguf) && present(paths->shape_dec_gguf)) {
        p->slat = trellis2_slat_flow_load(paths->slat_flow_gguf, true, &e,
                                          p->device.c_str());
        if (!p->slat) {
            p->last_error = "slat_flow: " + e;
            aicore_trellis_free(p);
            return nullptr;
        }
        const bool will_cascade = present(paths->slat_hr_flow_gguf);

        // Remember the flow-DiT gguf paths so a GPU decode can free them for
        // VRAM and reload them next generate (ensure_decode_vram /
        // reload_flows).
        p->ss_flow_path = paths->ss_flow_gguf;
        p->slat_path = paths->slat_flow_gguf;
        p->slat_hr_path = will_cascade ? paths->slat_hr_flow_gguf : "";

        // Auto-place the shape (FlexiDualGrid VAE) decoder — the biggest
        // fine-path stage (~44 s CPU / 59 %). Its mask-based submanifold conv
        // runs ~20x faster on the GPU (~2 s), but the decode's transient
        // buffers need the flow DiTs' VRAM freed first (done per-request in
        // ensure_decode_vram). "gpu"/"cpu" force the placement; "auto" puts it
        // on the GPU when the card can hold that decode once the flows are
        // freed (upstream's TRELLIS2_SHAPE_DEC_GPU / TRELLIS2_SHAPE_DEC_CPU env
        // vars).
        bool sd_gpu;
        const std::string placement =
                opts ? opts->shape_dec_placement : std::string("auto");
        if (placement == "gpu")
            sd_gpu = free_pre_flows > 0;
        else if (placement == "cpu")
            sd_gpu = false;
        else if (free_pre_flows == 0)
            sd_gpu = false;  // no GPU
        else {
            const size_t margin = (size_t)3 << 29;  // ~1.5 GB (weights + slack)
            sd_gpu = free_pre_flows >=
                     decode_vram_peak(will_cascade ? AICORE_TRELLIS_PIPE_1024
                                                   : AICORE_TRELLIS_PIPE_512) +
                             margin;
        }
        p->shapedec = trellis2_shape_dec_load(
                paths->shape_dec_gguf, true, &e,
                sd_gpu ? p->device_family.c_str() : "cpu");
        if (!p->shapedec &&
            sd_gpu) {  // unexpected GPU load OOM — fall back to CPU
            sd_gpu = false;
            p->shapedec = trellis2_shape_dec_load(paths->shape_dec_gguf, true,
                                                  &e, "cpu");
        }
        if (!p->shapedec) {
            p->last_error = "shape_dec: " + e;
            aicore_trellis_free(p);
            return nullptr;
        }
        p->shapedec_gpu = sd_gpu;
        p->fine = true;

        // The 1024 model is optional; when present the cascade path is enabled
        // and reuses p->shapedec for both the upsample and the 1024^3 decode.
        if (will_cascade) {
            p->slat_hr = trellis2_slat_flow_load(paths->slat_hr_flow_gguf, true,
                                                 &e, p->device.c_str());
            if (!p->slat_hr) {
                p->last_error = "slat_hr_flow: " + e;
                aicore_trellis_free(p);
                return nullptr;
            }
            p->cascade = true;
        }

        // PBR texturing: enabled when the shape encoder, texture decoder, and
        // (at least the 512) texture flow are present. The tex models are
        // loaded lazily per-generate (run_texture_stage), so only their paths
        // are kept.
        if (present(paths->shape_enc_gguf) && present(paths->tex_dec_gguf) &&
            present(paths->tex_flow_gguf)) {
            std::string ve;
            trellis2_shape_enc_model *te =
                    trellis2_shape_enc_load(paths->shape_enc_gguf, false, &ve);
            if (!te) {
                p->last_error = "shape_enc: " + ve;
                aicore_trellis_free(p);
                return nullptr;
            }
            trellis2_shape_enc_free(te);
            trellis2_shape_dec_model *td =
                    trellis2_tex_dec_load(paths->tex_dec_gguf, false, &ve);
            if (!td) {
                p->last_error = "tex_dec: " + ve;
                aicore_trellis_free(p);
                return nullptr;
            }
            if (trellis2_shape_dec_hparams_of(td).out_channels != 6) {
                trellis2_shape_dec_free(td);
                p->last_error = "tex_dec: expected 6 output channels";
                aicore_trellis_free(p);
                return nullptr;
            }
            trellis2_shape_dec_free(td);
            trellis2_slat_flow_model *tf =
                    trellis2_slat_flow_load(paths->tex_flow_gguf, false, &ve);
            if (!tf) {
                p->last_error = "tex_flow: " + ve;
                aicore_trellis_free(p);
                return nullptr;
            }
            if (trellis2_slat_flow_hparams_of(tf).concat_cond_channels != 32) {
                trellis2_slat_flow_free(tf);
                p->last_error =
                        "tex_flow: expected 32 concat-conditioning channels";
                aicore_trellis_free(p);
                return nullptr;
            }
            trellis2_slat_flow_free(tf);
            if (present(paths->tex_flow_hr_gguf)) {
                tf = trellis2_slat_flow_load(paths->tex_flow_hr_gguf, false,
                                             &ve);
                if (!tf) {
                    p->last_error = "tex_flow_hr: " + ve;
                    aicore_trellis_free(p);
                    return nullptr;
                }
                if (trellis2_slat_flow_hparams_of(tf).concat_cond_channels !=
                    32) {
                    trellis2_slat_flow_free(tf);
                    p->last_error =
                            "tex_flow_hr: expected 32 concat-conditioning "
                            "channels";
                    aicore_trellis_free(p);
                    return nullptr;
                }
                trellis2_slat_flow_free(tf);
            }
            p->shapeenc_path = paths->shape_enc_gguf;
            p->texdec_path = paths->tex_dec_gguf;
            p->texflow_path = paths->tex_flow_gguf;
            p->texflow_hr_path = present(paths->tex_flow_hr_gguf)
                                         ? paths->tex_flow_hr_gguf
                                         : "";
            p->texture = true;
        }
    }

    p->backend = trellis2_ss_flow_backend_name(p->flow);
    return p;
}

int aicore_trellis_caps(const aicore_trellis_ctx *p) {
    if (!p) return 0;
    int c = AICORE_TRELLIS_CAP_COARSE;
    if (p->fine) c |= AICORE_TRELLIS_CAP_512;
    if (p->cascade) c |= AICORE_TRELLIS_CAP_1024;
    if (p->texture) c |= AICORE_TRELLIS_CAP_TEXTURE;
    return c;
}

int aicore_trellis_is_ready(const aicore_trellis_ctx *ctx) {
    return ctx && ctx->dino && ctx->flow && ctx->dec ? 1 : 0;
}

const char *aicore_trellis_last_error(const aicore_trellis_ctx *ctx) {
    return ctx ? ctx->last_error.c_str() : "null context";
}

const char *aicore_trellis_backend(const aicore_trellis_ctx *p) {
    return p ? p->backend.c_str() : "none";
}

const char *aicore_trellis_backend_note(const aicore_trellis_ctx *p) {
    return p ? p->backend_note.c_str() : "";
}

void aicore_trellis_free(aicore_trellis_ctx *p) {
    if (!p) return;
    trellis2_dino_free(p->dino);
    trellis2_ss_flow_free(p->flow);
    trellis2_ss_dec_free(p->dec);
    trellis2_slat_flow_free(p->slat);
    trellis2_slat_flow_free(p->slat_hr);
    trellis2_shape_dec_free(p->shapedec);
#ifdef TRELLIS2_HAVE_RMBG
    trellis2_rmbg_free(p->rmbg);
#endif
    delete p;
}

void aicore_trellis_free_buffer(void *p) { std::free(p); }

// ─────────────────────────────────────────────────────────────────────────
// Image decode (Qt QImage) + TRELLIS preprocessing
// ─────────────────────────────────────────────────────────────────────────

// Decode encoded image bytes into tightly packed RGBA. Rejects absurd
// dimensions before decoding (upload DoS / decompression bombs): 16 MPixel is
// far beyond anything useful at a 512^2 / 1024^2 target.
bool decode_image_rgba(const void *image_bytes,
                       int image_len,
                       std::vector<uint8_t> &rgba,
                       int &w,
                       int &h,
                       std::string &e) {
    if (!image_bytes || image_len <= 0) {
        e = "invalid image bytes";
        return false;
    }
    QImage img = QImage::fromData((const uchar *)image_bytes, image_len);
    if (img.isNull()) {
        e = "image decode failed (unsupported format?)";
        return false;
    }
    if (img.width() <= 0 || img.height() <= 0 ||
        (int64_t)img.width() * img.height() > (int64_t)16 * 1024 * 1024) {
        e = "image dimensions out of range";
        return false;
    }
    const QImage conv = img.convertToFormat(QImage::Format_RGBA8888);
    w = conv.width();
    h = conv.height();
    rgba.resize((size_t)w * h * 4);
    std::memcpy(rgba.data(), conv.constBits(), rgba.size());
    return true;
}

// Decode + (solid-background removal) + alpha-bbox crop + resize to out_size.
int preprocess_image_bytes_mode(const void *image_bytes,
                                int image_len,
                                int out_size,
                                unsigned char *out_rgb,
                                int background_mode,
                                char *err,
                                int err_len) {
    if (!image_bytes || image_len <= 0 || out_size <= 0 || !out_rgb) {
        copy_err(err, err_len, "invalid arguments");
        return 1;
    }
    std::vector<uint8_t> rgba;
    int w = 0, h = 0;
    std::string e;
    if (!decode_image_rgba(image_bytes, image_len, rgba, w, h, e)) {
        copy_err(err, err_len, e);
        return 1;
    }
    if (background_mode < AICORE_TRELLIS_BG_AUTO ||
        background_mode > AICORE_TRELLIS_BG_WHITE) {
        copy_err(err, err_len, "invalid background mode");
        return 1;
    }
    trellis2_remove_solid_background_rgba(rgba.data(), w, h, background_mode);
    std::vector<uint8_t> rgb;
    const bool ok =
            trellis2_preprocess_rgba(rgba.data(), w, h, out_size, rgb, &e);
    if (!ok) {
        copy_err(err, err_len, "preprocess failed: " + e);
        return 1;
    }
    std::memcpy(out_rgb, rgb.data(), rgb.size());
    return 0;
}

int aicore_trellis_preprocess_image_bytes(const void *image_bytes,
                                          int image_len,
                                          int out_size,
                                          unsigned char *out_rgb,
                                          int background_mode,
                                          char *err,
                                          int err_len) {
    return preprocess_image_bytes_mode(image_bytes, image_len, out_size,
                                       out_rgb, background_mode, err, err_len);
}

// ─────────────────────────────────────────────────────────────────────────
// Generation
// ─────────────────────────────────────────────────────────────────────────

aicore_trellis_mesh *generate_impl(
        aicore_trellis_ctx *p,
        const void *image_bytes,
        int image_len,
        const aicore_trellis_generate_params *params,
        aicore_trellis_progress_fn progress,
        void *user,
        aicore_trellis_preview_fn preview,
        void *preview_user,
        char *err,
        int err_len) {
    if (!p) {
        copy_err(err, err_len, "null context");
        return nullptr;
    }
    std::string e;

    int pipeline_type =
            params ? params->pipeline_type : AICORE_TRELLIS_PIPE_AUTO;
    int background_mode =
            params ? params->background_mode : AICORE_TRELLIS_BG_AUTO;
    uint64_t seed = params ? params->seed : 0;
    int steps = params ? params->steps : 0;
    float guidance = params ? params->guidance : -1.0f;
    int texture_steps = params ? params->texture_steps : 0;
    int preview_stride = params ? params->preview_stride : 0;
    int keyframes =
            preview ? (params ? params->keyframes : 0) : 0;
    keyframes = keyframes < 0 ? 0 : (keyframes > 8 ? 8 : keyframes);
    if (pipeline_type < AICORE_TRELLIS_PIPE_AUTO ||
        pipeline_type > AICORE_TRELLIS_PIPE_1024) {
        copy_err(err, err_len, "invalid pipeline type");
        return nullptr;
    }
    if (background_mode < AICORE_TRELLIS_BG_AUTO ||
        background_mode > AICORE_TRELLIS_BG_WHITE) {
        copy_err(err, err_len, "invalid background mode");
        return nullptr;
    }

    // Reload any flow DiT a previous GPU decode freed for VRAM (usually a
    // no-op).
    if (!reload_flows(p, e)) {
        copy_err(err, err_len, "reload flow models: " + e);
        return nullptr;
    }

    // Resolve the requested path to what is actually loaded.
    int pt = pipeline_type;
    if (pt == AICORE_TRELLIS_PIPE_AUTO) {
        pt = p->cascade ? AICORE_TRELLIS_PIPE_1024
                        : (p->fine ? AICORE_TRELLIS_PIPE_512
                                   : AICORE_TRELLIS_PIPE_COARSE);
    }
    if (pt == AICORE_TRELLIS_PIPE_1024 && !p->cascade)
        pt = p->fine ? AICORE_TRELLIS_PIPE_512 : AICORE_TRELLIS_PIPE_COARSE;
    if (pt == AICORE_TRELLIS_PIPE_512 && !p->fine)
        pt = AICORE_TRELLIS_PIPE_COARSE;

    const int S = 512;

    if (progress) progress(user, AICORE_TRELLIS_STAGE_PREPROCESS, 0, 0);

    // Single decode pass: image bytes -> RGBA, then optional AI background
    // removal (in-tree RMBG task), then preprocessing. The upstream port
    // round-tripped through PNG between RMBG and preprocess; the RGBA result
    // is fed straight in here (no intermediate encode/decode).
    std::vector<uint8_t> src_rgba;
    int iw = 0, ih = 0;
    if (!decode_image_rgba(image_bytes, image_len, src_rgba, iw, ih, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
#ifdef TRELLIS2_HAVE_RMBG
    // AI background-removal result, carried out on the mesh for callers that
    // want the matted image (e.g. DB-tree ccImage output). Kept only when the
    // AI matting actually ran; the solid-color heuristic does not produce one.
    std::vector<uint8_t> rmbg_rgba;
    if (p->rmbg) {
        uint8_t *rmbg_out = nullptr;
        int rmbg_out_len = 0;
        std::string rmbg_err;
        int rc = trellis2_rmbg_remove_background(p->rmbg, src_rgba.data(), iw,
                                                 ih, &rmbg_out, &rmbg_out_len,
                                                 &rmbg_err);
        if (rc != 0 || !rmbg_out) {
            copy_err(err, err_len, "RMBG: " + rmbg_err);
            return nullptr;
        }
        src_rgba.assign(rmbg_out, rmbg_out + (size_t)rmbg_out_len);
        trellis2_rmbg_free_buffer(rmbg_out);
        rmbg_rgba = src_rgba;
        background_mode = AICORE_TRELLIS_BG_KEEP;
    }
#endif

    std::vector<unsigned char> rgb((size_t)S * S * 3);
    if (!trellis2_preprocess_rgba(src_rgba.data(), iw, ih, S, rgb, &e)) {
        copy_err(err, err_len, "preprocess failed: " + e);
        return nullptr;
    }

    if (progress) progress(user, AICORE_TRELLIS_STAGE_DINO, 0, 0);
    trellis2_dino_cond cond;  // 512-res conditioning (SS + LR flow)
    if (!trellis2_dino_encode_rgb(p->dino, rgb.data(), S, cond, &e)) {
        copy_err(err, err_len, "dino encode: " + e);
        return nullptr;
    }

    const trellis2_ss_flow_hparams &fhp = trellis2_ss_flow_hparams_of(p->flow);
    if (cond.channels() != fhp.cond_channels) {
        copy_err(err, err_len, "cond/flow channel mismatch");
        return nullptr;
    }

    // SS-decoder geometry (also the occupancy scratch reused by the settled
    // decode below).
    const trellis2_ss_dec_hparams &dechp = trellis2_ss_dec_hparams_of(p->dec);
    const int Rout = dechp.res_out();  // 64
    std::vector<float> occ((size_t)dechp.out_channels * Rout * Rout * Rout);

    trellis2_ss_sampler_params sp;
    if (steps > 0) sp.steps = steps;
    if (guidance >= 0) sp.guidance_strength = guidance;
    sp.seed = seed;
    sp.verbose = false;
    struct cb_ctx {
        aicore_trellis_progress_fn fn;
        void *user;
        int stage;
    } cbc{progress, user, AICORE_TRELLIS_STAGE_SS_FLOW};
    if (progress) {
        progress(user, AICORE_TRELLIS_STAGE_SS_FLOW, 0, sp.steps);
        sp.progress = [](void *u, int step, int total) {
            auto *c = (cb_ctx *)u;
            c->fn(c->user, c->stage, step, total);
        };
        sp.progress_user = &cbc;
    }
    // Live per-step previews: decode each step's x_0 estimate into a 64^3
    // occupancy and stream it as voxels (stride-gated; the last step is left
    // to the settled SS_DEC checkpoint below). pctx/occ outlive the sampler.
    ss_preview_ctx pctx;
    if (preview) {
        if (preview_stride < 0) {
            // Stage-checkpoints-only mode: no per-step SS previews.
        } else {
            if (preview_stride == 0)
                preview_stride = std::max(1, sp.steps / 4);
            pctx = ss_preview_ctx{preview, preview_user, p->dec, Rout,
                                  preview_stride, &occ};
            sp.preview = [](void *u, int step, int total,
                            const float *latent, int /*n*/) {
                auto *c = (ss_preview_ctx *)u;
                if (step % c->stride != 0 || step == total) return;
                std::string de;
                if (!trellis2_ss_dec_decode(c->dec, latent, c->occ->data(),
                                            &de))
                    return;
                std::vector<int32_t> cells;
                collect_occupied(c->occ->data(), c->res, cells);
                emit_voxels(c->fn, c->user, AICORE_TRELLIS_STAGE_SS_FLOW,
                            step, total, c->res, cells);
            };
            sp.preview_user = &pctx;
        }
    }

    const int R = fhp.resolution;
    std::vector<float> latent((size_t)fhp.in_channels * R * R * R);
    if (!trellis2_ss_flow_sample(p->flow, cond.data.data(), (int)cond.tokens(),
                                 (int)cond.channels(), &sp, nullptr,
                                 latent.data(), &e)) {
        copy_err(err, err_len, "ss_flow sample: " + e);
        return nullptr;
    }

    if (progress) progress(user, AICORE_TRELLIS_STAGE_SS_DEC, 0, 0);
    if (!trellis2_ss_dec_decode(p->dec, latent.data(), occ.data(), &e)) {
        copy_err(err, err_len, "ss_dec decode: " + e);
        return nullptr;
    }
    // Settled-occupancy checkpoint (the clean sparse structure, 64^3 voxels).
    if (preview) {
        std::vector<int32_t> cells;
        collect_occupied(occ.data(), Rout, cells);
        emit_voxels(preview, preview_user, AICORE_TRELLIS_STAGE_SS_DEC, 0, 0,
                    Rout, cells);
    }

    auto *r = new aicore_trellis_mesh();
#ifdef TRELLIS2_HAVE_RMBG
    if (!rmbg_rgba.empty()) {
        r->rmbg_rgba = std::move(rmbg_rgba);
        r->rmbg_w = iw;
        r->rmbg_h = ih;
    }
#endif

    if (pt == AICORE_TRELLIS_PIPE_COARSE) {
        // ── coarse path: marching cubes on the 64^3 occupancy ────────────────
        if (progress) progress(user, AICORE_TRELLIS_STAGE_MESH, 0, 0);
        mc::Mesh mesh = mc::extract(occ.data(), Rout, Rout, Rout, /*iso*/ 0.0f);
        if (mesh.verts.empty()) {
            copy_err(err, err_len, "empty mesh (no occupied voxels at iso 0)");
            delete r;
            return nullptr;
        }
        const float inv = 1.0f / (float)Rout;
        for (size_t i = 0; i < mesh.verts.size(); ++i)
            mesh.verts[i] = mesh.verts[i] * inv - 0.5f;
        r->verts = std::move(mesh.verts);
        r->normals = std::move(mesh.normals);
        r->tris = std::move(mesh.tris);
        return r;
    }

    // ── fine / cascade: 64^3 occupancy -> 32^3 voxel scaffold ────────────────
    const trellis2_slat_flow_hparams &shp =
            trellis2_slat_flow_hparams_of(p->slat);
    const int ss_res = shp.resolution;  // 32
    const int ratio = Rout / ss_res;    // 2 (max-pool 64 -> 32)
    std::vector<int32_t> coords;
    for (int x = 0; x < ss_res; ++x)
        for (int y = 0; y < ss_res; ++y)
            for (int z = 0; z < ss_res; ++z) {
                bool any = false;
                for (int dx = 0; dx < ratio && !any; ++dx)
                    for (int dy = 0; dy < ratio && !any; ++dy)
                        for (int dz = 0; dz < ratio && !any; ++dz) {
                            const int xi = x * ratio + dx, yi = y * ratio + dy,
                                      zi = z * ratio + dz;
                            const size_t idx =
                                    ((size_t)xi * Rout + yi) * Rout + zi;
                            if (occ[idx] > 0.0f) any = true;
                        }
                if (any) {
                    coords.push_back(x);
                    coords.push_back(y);
                    coords.push_back(z);
                }
            }
    int L = (int)(coords.size() / 3);
    if (L == 0) {
        copy_err(err, err_len, "empty voxel scaffold");
        delete r;
        return nullptr;
    }

    // shape-SLAT sampler params (shared LR + HR)
    auto make_slp = [&](int stage, uint64_t sd) {
        trellis2_ss_sampler_params slp;
        if (steps > 0) slp.steps = steps;
        if (guidance >= 0) slp.guidance_strength = guidance;
        slp.guidance_rescale = 0.5f;
        slp.rescale_t = 3.0f;
        slp.seed = sd;
        slp.verbose = false;
        if (progress) {
            progress(user, stage, 0, slp.steps);
            slp.progress = [](void *u, int step, int total) {
                auto *c = (cb_ctx *)u;
                c->fn(c->user, c->stage, step, total);
            };
            cbc.stage = stage;
            slp.progress_user = &cbc;
        }
        return slp;
    };

    kf_capture kf_lr;  // intermediate shape-flow keyframes (opt-in)
    kf_capture kf_hr;  // HR shape-flow keyframes (1024 model, opt-in)

    // ── LR shape-SLAT flow (512 model, 512-res cond) ─────────────────────────
    std::vector<float> slat((size_t)L * shp.in_channels);
    {
        trellis2_ss_sampler_params slp =
                make_slp(AICORE_TRELLIS_STAGE_SLAT_FLOW, seed ^ 0x51a7ULL);
        if (keyframes > 0) {
            kf_lr.stage = AICORE_TRELLIS_STAGE_SLAT_FLOW;
            kf_lr.res_in = ss_res;
            kf_lr.channels = shp.in_channels;
            kf_lr.norm_mean = shp.norm_mean;
            kf_lr.norm_std = shp.norm_std;
            kf_lr.coords = coords;
            while ((ss_res << (kf_lr.levels + 1)) <= 128) kf_lr.levels++;
            if (kf_lr.levels < 1) kf_lr.levels = 1;
            kf_lr.stride = std::max(1, slp.steps / keyframes);
            slp.preview = kf_capture_cb;
            slp.preview_user = &kf_lr;
        }
        if (!trellis2_slat_flow_sample(p->slat, L, coords.data(),
                                       cond.data.data(), (int)cond.tokens(),
                                       (int)cond.channels(), &slp, nullptr,
                                       /*denormalize*/ true, slat.data(), &e)) {
            copy_err(err, err_len, "slat sample: " + e);
            delete r;
            return nullptr;
        }
    }

    const trellis2_shape_dec_hparams &dhp2 =
            trellis2_shape_dec_hparams_of(p->shapedec);
    std::vector<float> dec_feats;  // 7-ch decoder output to mesh
    std::vector<int32_t> dec_coords;
    int grid = 0;
    trellis2_dino_cond
            cond1024;  // filled by the cascade branch; reused by texturing

    if (pt == AICORE_TRELLIS_PIPE_512) {
        // ── 512 fine: decode the LR slat directly at grid 512 ────────────────
        if (progress) progress(user, AICORE_TRELLIS_STAGE_SHAPE_DEC, 0, 0);
        ensure_decode_vram(
                p, AICORE_TRELLIS_PIPE_512);  // free the flow DiTs if a GPU
                                              // decode needs the room
        if (!trellis2_shape_dec_decode(p->shapedec, slat.data(), L,
                                       coords.data(), dec_feats, dec_coords,
                                       nullptr, &e)) {
            copy_err(err, err_len, "shape decode: " + e);
            delete r;
            return nullptr;
        }
        grid = ss_res * dhp2.upscale();  // 32 * 16 = 512
    } else {
        // ── 1024 cascade: upsample -> quantize -> HR flow -> decode grid 1024
        // ─
        if (progress) progress(user, AICORE_TRELLIS_STAGE_UPSAMPLE, 0, 0);
        std::vector<int32_t> up_coords;  // 512^3 candidate coords
        if (!trellis2_shape_dec_upsample(p->shapedec, slat.data(), L,
                                         coords.data(),
                                         /*upsample_times*/ 4, up_coords, &e)) {
            copy_err(err, err_len, "shape upsample: " + e);
            delete r;
            return nullptr;
        }
        // quantize (c+0.5)/512*64 and dedup into the 64^3 HR scaffold
        const int lr_res = ss_res * dhp2.upscale();  // 512
        const int hr_grid = shp.resolution * 2;      // 64 (HR flow resolution)
        std::unordered_set<uint64_t> seen;
        std::vector<int32_t> hr_coords;
        auto key = [](int32_t a, int32_t b, int32_t c) {
            return ((uint64_t)(uint32_t)a << 40) |
                   ((uint64_t)(uint32_t)b << 20) | (uint64_t)(uint32_t)c;
        };
        for (size_t i = 0; i < up_coords.size(); i += 3) {
            int32_t qx = (int32_t)((up_coords[i] + 0.5f) / lr_res * hr_grid);
            int32_t qy =
                    (int32_t)((up_coords[i + 1] + 0.5f) / lr_res * hr_grid);
            int32_t qz =
                    (int32_t)((up_coords[i + 2] + 0.5f) / lr_res * hr_grid);
            if (seen.insert(key(qx, qy, qz)).second) {
                hr_coords.push_back(qx);
                hr_coords.push_back(qy);
                hr_coords.push_back(qz);
            }
        }
        const int Lhr = (int)(hr_coords.size() / 3);
        if (Lhr == 0) {
            copy_err(err, err_len, "empty HR scaffold");
            delete r;
            return nullptr;
        }

        // Sharper 64^3 HR-scaffold checkpoint (the cascade's refined
        // structure).
        if (preview) {
            emit_voxels(preview, preview_user, AICORE_TRELLIS_STAGE_UPSAMPLE,
                        0, 0, hr_grid, hr_coords);
        }

        // 1024-res conditioning (separate preprocess + encode at 1024)
        std::vector<unsigned char> rgb1024((size_t)1024 * 1024 * 3);
        if (!trellis2_preprocess_rgba(src_rgba.data(), iw, ih, 1024, rgb1024,
                                      &e)) {
            copy_err(err, err_len, "preprocess failed: " + e);
            delete r;
            return nullptr;
        }
        if (!trellis2_dino_encode_rgb(p->dino, rgb1024.data(), 1024, cond1024,
                                      &e)) {
            copy_err(err, err_len, "dino encode 1024: " + e);
            delete r;
            return nullptr;
        }

        // HR shape-SLAT flow (1024 model)
        const trellis2_slat_flow_hparams &shp_hr =
                trellis2_slat_flow_hparams_of(p->slat_hr);
        std::vector<float> hr_slat((size_t)Lhr * shp_hr.in_channels);
        trellis2_ss_sampler_params slp =
                make_slp(AICORE_TRELLIS_STAGE_SLAT_FLOW_HR, seed ^ 0x1024ULL);
        if (keyframes > 0) {
            kf_hr.stage = AICORE_TRELLIS_STAGE_SLAT_FLOW_HR;
            kf_hr.res_in = hr_grid;
            kf_hr.channels = shp_hr.in_channels;
            kf_hr.norm_mean = shp_hr.norm_mean;
            kf_hr.norm_std = shp_hr.norm_std;
            kf_hr.coords = hr_coords;
            while ((hr_grid << (kf_hr.levels + 1)) <= 128) kf_hr.levels++;
            if (kf_hr.levels < 1) kf_hr.levels = 1;
            kf_hr.stride = std::max(1, slp.steps / keyframes);
            slp.preview = kf_capture_cb;
            slp.preview_user = &kf_hr;
        }
        if (!trellis2_slat_flow_sample(
                    p->slat_hr, Lhr, hr_coords.data(), cond1024.data.data(),
                    (int)cond1024.tokens(), (int)cond1024.channels(), &slp,
                    nullptr, /*denormalize*/ true, hr_slat.data(), &e)) {
            copy_err(err, err_len, "HR slat sample: " + e);
            delete r;
            return nullptr;
        }

        if (progress) progress(user, AICORE_TRELLIS_STAGE_SHAPE_DEC_HR, 0, 0);
        ensure_decode_vram(
                p, AICORE_TRELLIS_PIPE_1024);  // free the flow DiTs (all done)
                                               // for the 1024^3 decode
        if (!trellis2_shape_dec_decode(p->shapedec, hr_slat.data(), Lhr,
                                       hr_coords.data(), dec_feats, dec_coords,
                                       nullptr, &e)) {
            copy_err(err, err_len, "HR shape decode: " + e);
            delete r;
            return nullptr;
        }
        grid = hr_grid * dhp2.upscale();  // 64 * 16 = 1024
    }

    // ── mesh extraction (shared) ─────────────────────────────────────────────
    if (progress) progress(user, AICORE_TRELLIS_STAGE_MESH, 0, 0);
    const int nvox = (int)(dec_coords.size() / 3);
    fdg::Mesh mesh =
            fdg::extract(dec_feats.data(), dec_coords.data(), nvox, grid);
    if (mesh.verts.empty()) {
        copy_err(err, err_len, "empty mesh (dual grid found no faces)");
        delete r;
        return nullptr;
    }
    // Clean up the raw dual-grid soup: drop floating specks, fill the closed
    // boundary loops, then close the remaining open cracks with centroid
    // fans. All three only add triangles (the fan adds one synthetic centroid
    // vertex per crack), so the generated material volume can still be sampled
    // at every vertex afterwards.
    fdg::drop_small_components(mesh);
    fdg::fill_holes(mesh);
    fdg::fill_open_chains(mesh);
    r->verts = std::move(mesh.verts);
    r->tris = std::move(mesh.tris);
    r->normals = fdg::vertex_normals(fdg::Mesh{r->verts, r->tris});
    r->grid_res = grid;
    r->grid_feats = dec_feats;
    r->grid_coords = dec_coords;

    // ── shape-flow keyframe replay ──────────────────────────────────
    // Now is the safe window: the flow DiTs are freed (ensure_decode_vram)
    // and the shape decoder owns VRAM, before the texture stage loads its
    // models.
    if (preview && keyframes > 0) {
        emit_keyframes(p->shapedec, kf_lr, preview, preview_user);
        emit_keyframes(p->shapedec, kf_hr, preview, preview_user);  // empty 512
    }

    // ── PBR texture stage (optional) ─────────────────────────────────────────
    if (p->texture && pt != AICORE_TRELLIS_PIPE_COARSE) {
        // Free the (finished) geometry flow DiTs so the ~4 GB of tex models fit
        // in VRAM; reload_flows() restores them on the next generate.
        if (trellis2_gpu_free_vram(p->device_family.c_str()) > 0) {
            if (p->flow) {
                trellis2_ss_flow_free(p->flow);
                p->flow = nullptr;
            }
            if (p->slat) {
                trellis2_slat_flow_free(p->slat);
                p->slat = nullptr;
            }
            if (p->slat_hr) {
                trellis2_slat_flow_free(p->slat_hr);
                p->slat_hr = nullptr;
            }
        }
        const trellis2_dino_cond &texcond =
                (pt == AICORE_TRELLIS_PIPE_1024) ? cond1024 : cond;
        std::vector<float> pbr;
        std::string te;
        if (!run_texture_stage(p, dec_feats, dec_coords, r->verts, grid, pt,
                               texcond, seed ^ 0x7ec0ULL, texture_steps,
                               progress, user, pbr, te)) {
            copy_err(err, err_len, "texture: " + te);
            delete r;
            return nullptr;
        }
        r->pbr = std::move(pbr);
    }
    return r;
}

aicore_trellis_mesh *aicore_trellis_generate(
        aicore_trellis_ctx *p,
        const void *image_bytes,
        int image_len,
        const aicore_trellis_generate_params *params,
        aicore_trellis_progress_fn progress,
        void *progress_user,
        char *err,
        int err_len) {
    return generate_impl(p, image_bytes, image_len, params, progress,
                         progress_user, nullptr, nullptr, err, err_len);
}

aicore_trellis_mesh *aicore_trellis_generate_ex(
        aicore_trellis_ctx *p,
        const void *image_bytes,
        int image_len,
        const aicore_trellis_generate_params *params,
        aicore_trellis_progress_fn progress,
        void *progress_user,
        aicore_trellis_preview_fn preview,
        void *preview_user,
        char *err,
        int err_len) {
    return generate_impl(p, image_bytes, image_len, params, progress,
                         progress_user, preview, preview_user, err, err_len);
}

// ──────────────────────────────────────────────────────────────────────────
// Standalone texturing / export preparation
// ──────────────────────────────────────────────────────────────────────────

aicore_trellis_mesh *aicore_trellis_texture_mesh(
        aicore_trellis_ctx *p,
        const float *verts,
        int n_verts,
        const int *tris,
        int n_tris,
        const float *grid_feats,
        int grid_nvox,
        const int *grid_coords,
        int grid_res,
        int pipeline_type,
        const void *image_bytes,
        int image_len,
        int background_mode,
        uint64_t seed,
        int texture_steps,
        aicore_trellis_progress_fn progress,
        void *user,
        char *err,
        int err_len) {
    if (!p) {
        copy_err(err, err_len, "null context");
        return nullptr;
    }
    if (!p->texture) {
        copy_err(err, err_len, "texture models not loaded");
        return nullptr;
    }
    if (!verts || n_verts <= 0 || !tris || n_tris <= 0) {
        copy_err(err, err_len, "invalid mesh");
        return nullptr;
    }
    const bool use_qef = !grid_feats || grid_nvox <= 0 || !grid_coords;
    if (grid_res <= 0) {
        copy_err(err, err_len, "grid_res required");
        return nullptr;
    }
    if (!image_bytes || image_len <= 0) {
        copy_err(err, err_len, "invalid image");
        return nullptr;
    }
    if (pipeline_type != AICORE_TRELLIS_PIPE_512 &&
        pipeline_type != AICORE_TRELLIS_PIPE_1024) {
        copy_err(err, err_len, "pipeline_type must be 512 or 1024");
        return nullptr;
    }
    if (pipeline_type == AICORE_TRELLIS_PIPE_1024 &&
        p->texflow_hr_path.empty()) {
        copy_err(err, err_len, "1024 texture flow model not loaded");
        return nullptr;
    }

    std::string e;
    const int S = (pipeline_type == AICORE_TRELLIS_PIPE_1024) ? 1024 : 512;
    if (progress) progress(user, AICORE_TRELLIS_STAGE_PREPROCESS, 0, 0);

    // Single decode pass (Qt QImage), then optional in-tree RMBG matting,
    // then preprocessing — mirrors the integrated generate path.
    std::vector<uint8_t> src_rgba;
    int iw = 0, ih = 0;
    if (!decode_image_rgba(image_bytes, image_len, src_rgba, iw, ih, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
#ifdef TRELLIS2_HAVE_RMBG
    if (p->rmbg) {
        uint8_t *rmbg_out = nullptr;
        int rmbg_out_len = 0;
        std::string rmbg_err;
        int rc = trellis2_rmbg_remove_background(p->rmbg, src_rgba.data(), iw,
                                                 ih, &rmbg_out, &rmbg_out_len,
                                                 &rmbg_err);
        if (rc != 0 || !rmbg_out) {
            copy_err(err, err_len, "RMBG: " + rmbg_err);
            return nullptr;
        }
        src_rgba.assign(rmbg_out, rmbg_out + (size_t)rmbg_out_len);
        trellis2_rmbg_free_buffer(rmbg_out);
        background_mode = AICORE_TRELLIS_BG_KEEP;
    }
#endif
    std::vector<unsigned char> rgb((size_t)S * S * 3);
    if (!trellis2_preprocess_rgba(src_rgba.data(), iw, ih, S, rgb, &e)) {
        copy_err(err, err_len, "preprocess failed: " + e);
        return nullptr;
    }

    if (progress) progress(user, AICORE_TRELLIS_STAGE_DINO, 0, 0);
    trellis2_dino_cond cond;
    if (!trellis2_dino_encode_rgb(p->dino, rgb.data(), S, cond, &e)) {
        copy_err(err, err_len, "dino encode: " + e);
        return nullptr;
    }

    auto *r = new aicore_trellis_mesh();
    r->verts.assign(verts, verts + (size_t)n_verts * 3);
    r->tris.assign(tris, tris + (size_t)n_tris * 3);
    r->normals = fdg::vertex_normals(fdg::Mesh{r->verts, r->tris});
    r->grid_res = grid_res;

    std::vector<float> pbr;
    if (use_qef) {
        std::vector<int32_t> qef_coords;
        if (!run_texture_stage_qef(p, verts, n_verts, tris, n_tris, grid_res,
                                   r->verts, pipeline_type, cond, seed,
                                   texture_steps, progress, user, pbr,
                                   qef_coords, e)) {
            copy_err(err, err_len, "texture: " + e);
            delete r;
            return nullptr;
        }
        r->grid_coords = std::move(qef_coords);
    } else {
        r->grid_feats.assign(grid_feats,
                             grid_feats + (size_t)grid_nvox * 7);
        r->grid_coords.assign(grid_coords,
                              grid_coords + (size_t)grid_nvox * 3);
        if (!run_texture_stage(p, r->grid_feats, r->grid_coords, r->verts,
                               grid_res, pipeline_type, cond, seed,
                               texture_steps, progress, user, pbr, e)) {
            copy_err(err, err_len, "texture: " + e);
            delete r;
            return nullptr;
        }
    }
    r->pbr = std::move(pbr);
    return r;
}

aicore_trellis_mesh *aicore_trellis_prepare_mesh(
        const float *verts,
        int n_verts,
        const int *tris,
        int n_tris,
        const float *pbr,
        int component_filter,
        char *err,
        int err_len) {
    if (!verts || !tris || n_verts <= 0 || n_tris <= 0) {
        copy_err(err, err_len, "empty mesh");
        return nullptr;
    }
    if (component_filter < 0 || component_filter > 2) {
        copy_err(err, err_len, "bad component filter");
        return nullptr;
    }
    t2glb::MeshExportOptions opt;
    opt.components = (t2glb::ComponentFilter)component_filter;
    t2glb::PreparedMesh prepared;
    std::string e;
    if (!t2glb::prepare_mesh(verts, n_verts, (const int32_t *)tris, n_tris,
                             pbr, opt, prepared, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
    auto *r = new aicore_trellis_mesh();
    r->verts = std::move(prepared.verts);
    r->normals = std::move(prepared.normals);
    r->tris.assign(prepared.tris.begin(), prepared.tris.end());
    r->pbr = std::move(prepared.pbr);
    return r;
}

int aicore_trellis_print_remesh_available(void) {
    return t2glb::print_remesh_available() ? 1 : 0;
}

aicore_trellis_mesh *aicore_trellis_prepare_print_mesh(
        const float *verts,
        int n_verts,
        const int *tris,
        int n_tris,
        const float *pbr,
        int component_filter,
        float alpha_ratio,
        float offset_ratio,
        char *err,
        int err_len) {
    if (!verts || !tris || n_verts <= 0 || n_tris <= 0) {
        copy_err(err, err_len, "empty mesh");
        return nullptr;
    }
    if (component_filter < 0 || component_filter > 2) {
        copy_err(err, err_len, "bad component filter");
        return nullptr;
    }
    t2glb::MeshExportOptions opt;
    opt.components = (t2glb::ComponentFilter)component_filter;
    t2glb::PreparedMesh prepared;
    std::string e;
    if (!t2glb::prepare_print_mesh(verts, n_verts, (const int32_t *)tris,
                                   n_tris, pbr, opt, alpha_ratio, offset_ratio,
                                   prepared, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
    auto *r = new aicore_trellis_mesh();
    r->verts = std::move(prepared.verts);
    r->normals = std::move(prepared.normals);
    r->tris.assign(prepared.tris.begin(), prepared.tris.end());
    r->pbr = std::move(prepared.pbr);
    return r;
}

uint8_t *aicore_trellis_bake_projected_glb(
        const float *target_verts,
        int target_n_verts,
        const int *target_tris,
        int target_n_tris,
        const float *source_verts,
        int source_n_verts,
        const int *source_tris,
        int source_n_tris,
        const float *source_pbr,
        int texture_size,
        int source_component_filter,
        int *out_len,
        char *err,
        int err_len) {
    if (out_len) *out_len = 0;
    if (!target_verts || !target_tris || target_n_verts <= 0 ||
        target_n_tris <= 0 || !source_verts || !source_tris || !source_pbr ||
        source_n_verts <= 0 || source_n_tris <= 0) {
        copy_err(err, err_len, "empty projected GLB mesh");
        return nullptr;
    }
    if (source_component_filter < 0 || source_component_filter > 2) {
        copy_err(err, err_len, "bad component filter");
        return nullptr;
    }
    t2glb::MeshExportOptions opt;
    if (texture_size > 0) opt.texture_size = texture_size;
    opt.components = (t2glb::ComponentFilter)source_component_filter;
    std::vector<uint8_t> glb;
    std::string e;
    if (!t2glb::mesh_to_projected_glb(
                target_verts, target_n_verts, (const int32_t *)target_tris,
                target_n_tris, source_verts, source_n_verts,
                (const int32_t *)source_tris, source_n_tris, source_pbr, opt,
                glb, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
    uint8_t *buf = (uint8_t *)std::malloc(glb.size());
    if (!buf) {
        copy_err(err, err_len, "out of memory");
        return nullptr;
    }
    std::memcpy(buf, glb.data(), glb.size());
    if (out_len) *out_len = (int)glb.size();
    return buf;
}


// ─────────────────────────────────────────────────────────────────────────
// Mesh accessors (buffers borrowed until aicore_trellis_mesh_free)
// ─────────────────────────────────────────────────────────────────────────

int aicore_trellis_mesh_n_verts(const aicore_trellis_mesh *r) {
    return r ? (int)(r->verts.size() / 3) : 0;
}
int aicore_trellis_mesh_n_tris(const aicore_trellis_mesh *r) {
    return r ? (int)(r->tris.size() / 3) : 0;
}
const float *aicore_trellis_mesh_verts(const aicore_trellis_mesh *r) {
    return r ? r->verts.data() : nullptr;
}
const float *aicore_trellis_mesh_normals(const aicore_trellis_mesh *r) {
    return r ? r->normals.data() : nullptr;
}
const int *aicore_trellis_mesh_tris(const aicore_trellis_mesh *r) {
    return r ? r->tris.data() : nullptr;
}
int aicore_trellis_mesh_has_pbr(const aicore_trellis_mesh *r) {
    return (r && !r->pbr.empty()) ? 1 : 0;
}
const float *aicore_trellis_mesh_pbr(const aicore_trellis_mesh *r) {
    return (r && !r->pbr.empty()) ? r->pbr.data() : nullptr;
}
int aicore_trellis_mesh_grid_res(const aicore_trellis_mesh *r) {
    return r ? r->grid_res : 0;
}
int aicore_trellis_mesh_grid_nvox(const aicore_trellis_mesh *r) {
    return r ? (int)(r->grid_coords.size() / 3) : 0;
}
const float *aicore_trellis_mesh_grid_feats(const aicore_trellis_mesh *r) {
    return (r && !r->grid_feats.empty()) ? r->grid_feats.data() : nullptr;
}
const int *aicore_trellis_mesh_grid_coords(const aicore_trellis_mesh *r) {
    return (r && !r->grid_coords.empty()) ? r->grid_coords.data() : nullptr;
}
int aicore_trellis_mesh_has_rmbg(const aicore_trellis_mesh *r) {
    return (r && !r->rmbg_rgba.empty()) ? 1 : 0;
}
const uint8_t *aicore_trellis_mesh_rmbg_rgba(const aicore_trellis_mesh *r) {
    return (r && !r->rmbg_rgba.empty()) ? r->rmbg_rgba.data() : nullptr;
}
int aicore_trellis_mesh_rmbg_w(const aicore_trellis_mesh *r) {
    return r ? r->rmbg_w : 0;
}
int aicore_trellis_mesh_rmbg_h(const aicore_trellis_mesh *r) {
    return r ? r->rmbg_h : 0;
}
void aicore_trellis_mesh_free(aicore_trellis_mesh *r) { delete r; }

// ─────────────────────────────────────────────────────────────────────────
// GLB export
// ─────────────────────────────────────────────────────────────────────────

uint8_t *aicore_trellis_bake_glb(const float *verts,
                                 int n_verts,
                                 const int *tris,
                                 int n_tris,
                                 const float *pbr,
                                 int texture_size,
                                 int component_filter,
                                 int *out_len,
                                 char *err,
                                 int err_len) {
    if (out_len) *out_len = 0;
    if (!verts || !tris || n_verts <= 0 || n_tris <= 0) {
        copy_err(err, err_len, "empty mesh");
        return nullptr;
    }
    if (component_filter < 0 || component_filter > 2) {
        copy_err(err, err_len, "bad component filter");
        return nullptr;
    }
    t2glb::MeshExportOptions opt;
    if (texture_size > 0) opt.texture_size = texture_size;
    opt.components = (t2glb::ComponentFilter)component_filter;
    std::vector<uint8_t> glb;
    std::string e;
    if (!t2glb::mesh_to_glb(verts, n_verts, tris, n_tris, pbr, opt, glb, e)) {
        copy_err(err, err_len, e);
        return nullptr;
    }
    uint8_t *buf = (uint8_t *)std::malloc(glb.size());
    if (!buf) {
        copy_err(err, err_len, "out of memory");
        return nullptr;
    }
    std::memcpy(buf, glb.data(), glb.size());
    if (out_len) *out_len = (int)glb.size();
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────
// Runtime plumbing
// ─────────────────────────────────────────────────────────────────────────

int aicore_trellis_warmup_backend(const char *device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

void aicore_trellis_shutdown(void) {
    // Reclaims process-wide backend registry entries whose owners are gone
    // (expired leases). Live contexts are never touched; ggml backends stay
    // registered for the process lifetime.
    aicore::runtime::purge_inactive_backend_leases();
}

char *aicore_trellis_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::trellis_model_cache_dir());
}

char *aicore_trellis_info_json(aicore_trellis_ctx *ctx) {
    if (!ctx) return aicore::capi::dup_cstr("{\"error\":\"null context\"}");
    std::string j = "{";
    j += "\"caps\":" + std::to_string(aicore_trellis_caps(ctx));
    j += ",\"backend\":\"" + aicore::capi::json_escape(ctx->backend) + "\"";
    j += ",\"shapedec_gpu\":" +
         std::string(ctx->shapedec_gpu ? "true" : "false");
    j += "}";
    return aicore::capi::dup_cstr(j);
}

}  // extern "C"
