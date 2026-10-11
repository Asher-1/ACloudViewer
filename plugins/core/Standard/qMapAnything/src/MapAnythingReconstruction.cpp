// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MapAnythingReconstruction.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace MapAnythingReconstruction {

namespace {

using MapAnythingGeometry::Sim3;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
// aicore_map_anything_run_images: device cannot hold this pass size.
constexpr int kRunOutOfMemory = -3;
// Correspondence cap for one boundary fit. A Sim3 has 7 DoF; beyond a few
// hundred thousand well-spread pixels the estimate no longer moves, while
// Horn's covariance and the MAD rejection stay O(m) in time and memory.
constexpr size_t kMaxCorrespondences = 262144;

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

std::string ctx_error(const aicore_map_anything_ctx* ctx) {
    const char* e = aicore_map_anything_last_error(ctx);
    return e != nullptr && e[0] != '\0' ? std::string(e)
                                        : std::string("unknown error");
}

struct Rect {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// The view's resized content inside the common padded grid; the white
// filler outside it carries no prediction worth keeping.
Rect valid_rect(const aicore_map_anything_ctx* ctx, int view, int H, int W) {
    int32_t vw = W, vh = H;
    if (aicore_map_anything_result_view_valid_size(ctx, view, &vw, &vh) != 0 ||
        vw <= 0 || vh <= 0 || vw > W || vh > H) {
        vw = W;
        vh = H;
    }
    return Rect{(W - vw) / 2, (H - vh) / 2, vw, vh};
}

bool finite3(const float* p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// Chunk-frame world point of every pixel of one view (H*W*3, NaN = not a
// cloud pixel: outside the valid rect, out of the mapanything mask, on a
// depth edge or non-finite). Source per architecture:
//   dust3r                  -> local points (both views in view1's frame)
//   mapanything / vggt-1b   -> published world points
//   pi3 / pi3x              -> local points lifted by the c2w pose
//   vggt-omega / vggt       -> depth unprojected via FoV K + pose_enc
// O(H*W) (the edge window is a fixed 3x3).
bool view_points(const aicore_map_anything_ctx* ctx,
                 const std::string& arch,
                 int i,
                 int H,
                 int W,
                 float edgeRtol,
                 std::vector<float>& P,
                 std::string* error) {
    const size_t plane = (size_t)H * W;
    const float* depth = aicore_map_anything_result_depth(ctx);
    const float* world = aicore_map_anything_result_points(ctx);
    const float* local = aicore_map_anything_result_local_points(ctx);
    const float* mask = aicore_map_anything_result_mask(ctx);
    const float* pose = aicore_map_anything_result_pose(ctx);
    const int32_t stride = aicore_map_anything_result_pose_stride(ctx);

    enum { kCopy, kLift, kUnproject } mode;
    const float* src = nullptr;
    if (arch == "dust3r" && local != nullptr) {
        mode = kCopy;
        src = local;
    } else if (world != nullptr) {
        mode = kCopy;
        src = world;
    } else if (stride == 16 && local != nullptr && pose != nullptr) {
        mode = kLift;
        src = local;
    } else if (stride == 9 && depth != nullptr && pose != nullptr) {
        mode = kUnproject;
    } else {
        set_error(error, "architecture '" + arch +
                                 "' published no usable geometry output");
        return false;
    }

    P.assign(plane * 3, kNaN);
    std::vector<uint8_t> edge;
    if (edgeRtol > 0.0f && depth != nullptr) {
        edge.assign(plane, 0);
        MapAnythingGeometry::depth_edge_mask(depth + (size_t)i * plane, H, W,
                                             edgeRtol, edge.data());
    }
    float R[9], T[3], K[4];
    if (mode == kUnproject) {
        const float* row = pose + (size_t)i * 9;
        MapAnythingGeometry::pose_enc_to_w2c(row, R, T);
        MapAnythingGeometry::fov_to_intrinsics(row[7], row[8], W, H, K);
    }
    const Rect r = valid_rect(ctx, i, H, W);
    for (int y = r.y0; y < r.y0 + r.h; ++y) {
        for (int x = r.x0; x < r.x0 + r.w; ++x) {
            const size_t idx = (size_t)y * W + x;
            const size_t g = (size_t)i * plane + idx;  // view-major offset
            if (mask != nullptr && !(mask[g] >= 0.5f)) continue;
            if (!edge.empty() && edge[idx]) continue;
            float* out = &P[idx * 3];
            if (mode == kUnproject) {
                MapAnythingGeometry::unproject_depth_pixel(R, T, K, x, y,
                                                           depth[g], out);
            } else if (mode == kLift) {
                MapAnythingGeometry::transform_by_c2w(pose + (size_t)i * 16,
                                                      src + g * 3, out);
            } else {
                out[0] = src[g * 3 + 0];
                out[1] = src[g * 3 + 1];
                out[2] = src[g * 3 + 2];
            }
            if (!finite3(out)) out[0] = out[1] = out[2] = kNaN;
        }
    }
    return true;
}

void apply_to_finite(const Sim3& s, std::vector<float>& P) {
    const size_t n = P.size() / 3;
    for (size_t k = 0; k < n; ++k) {
        float* p = &P[k * 3];
        if (std::isfinite(p[0])) {
            MapAnythingGeometry::apply_sim3(s.R, s.t, s.s, p, 1, p);
        }
    }
}

// Nearest source-pixel color through the view's valid rect, honoring the
// borrowed view's real stride and channel order.
void sample_rgb(
        const aicore_image_view& v, const Rect& r, int x, int y, uint8_t* out) {
    const int su =
            std::min(v.width - 1, (x - r.x0) * v.width / std::max(1, r.w));
    const int sv =
            std::min(v.height - 1, (y - r.y0) * v.height / std::max(1, r.h));
    const uint8_t* row = static_cast<const uint8_t*>(v.data) +
                         (size_t)sv * v.row_stride_bytes;
    switch (v.format) {
        case AICORE_IMAGE_RGB8:
            std::memcpy(out, row + (size_t)su * 3, 3);
            return;
        case AICORE_IMAGE_BGR8: {
            const uint8_t* p = row + (size_t)su * 3;
            out[0] = p[2];
            out[1] = p[1];
            out[2] = p[0];
            return;
        }
        case AICORE_IMAGE_RGBA8:
            std::memcpy(out, row + (size_t)su * 4, 3);
            return;
        case AICORE_IMAGE_BGRA8: {
            const uint8_t* p = row + (size_t)su * 4;
            out[0] = p[2];
            out[1] = p[1];
            out[2] = p[0];
            return;
        }
        case AICORE_IMAGE_GRAY8:
        default:
            out[0] = out[1] = out[2] = row[su];
            return;
    }
}

}  // namespace

int run(aicore_map_anything_ctx* ctx,
        const aicore_image_view* views,
        int n,
        const Options& options,
        Result* out,
        std::string* error,
        const ViewHook& hook,
        const ChunkGate& gate) {
    if (ctx == nullptr || views == nullptr || n < 1 || out == nullptr) {
        set_error(error, "invalid arguments");
        return kError;
    }
    *out = Result();
    int K = std::min(n, std::max(2, options.maxFramesPerPass));
    std::vector<int> starts =
            MapAnythingGeometry::plan_chunk_starts(n, K, options.overlap);
    int nc = (int)starts.size();
    // Smallest pass that still advances the window past the shared frames.
    const int minK = std::min(n, std::max(2, options.overlap + 1));
    out->requestedFramesPerPass = K;
    out->views = n;
    out->c2w.assign((size_t)n * 16, kNaN);
    out->focalPx.assign((size_t)n, kNaN);
    out->cloud.populationOffset.assign((size_t)n + 1, 0);

    std::vector<char> covered((size_t)n, 0);
    // Aligned (global-frame) points of the frames the NEXT chunk shares:
    // at most overlap-ish views * H*W*3 floats alive at once.
    std::vector<std::vector<float>> anchors((size_t)n);
    std::vector<float> P;
    Cloud& cloud = out->cloud;

    for (int ci = 0; ci < nc; ++ci) {
        if (gate && !gate(ci, nc)) {
            set_error(error, "cancelled");
            return kCancelled;
        }
        const int st = starts[(size_t)ci];
        const int count = std::min(K, n - st);
        const int runRc =
                aicore_map_anything_run_images(ctx, views + st, count);
        if (runRc == kRunOutOfMemory && ci == 0 && K > minK) {
            // The device refused this pass size before anything ran and
            // before any hook fired, so re-planning the whole scene with a
            // halved pass is side-effect free. Halving bounds the retries
            // at log2(K) (each re-stages the weights); memory is ~linear in
            // the pass size, so the largest admitted power-of-two fraction
            // is at most 2x below the true fit.
            K = std::max(minK, K / 2);
            starts = MapAnythingGeometry::plan_chunk_starts(n, K,
                                                            options.overlap);
            nc = (int)starts.size();
            out->memoryNotes.push_back(ctx_error(ctx));
            --ci;
            continue;
        }
        if (runRc != 0) {
            set_error(error, "chunk " + std::to_string(ci) + " (frames " +
                                     std::to_string(st) + ".." +
                                     std::to_string(st + count - 1) +
                                     "): " + ctx_error(ctx));
            return kError;
        }
        out->framesPerPass = K;
        const int H = aicore_map_anything_result_height(ctx);
        const int W = aicore_map_anything_result_width(ctx);
        const char* archC = aicore_map_anything_result_architecture(ctx);
        const std::string arch = archC != nullptr ? archC : "";
        if (ci == 0) {
            out->architecture = arch;
            const char* dev = aicore_map_anything_context_device(ctx);
            out->device = dev != nullptr ? dev : "";
            out->height = H;
            out->width = W;
            out->poseStride = aicore_map_anything_result_pose_stride(ctx);
        } else if (H != out->height || W != out->width) {
            // Mixed grids make dense per-pixel anchors meaningless.
            set_error(error, "chunk " + std::to_string(ci) +
                                     " dense grid differs from chunk 0");
            return kError;
        }
        if (H <= 0 || W <= 0) {
            set_error(error, "empty dense output grid");
            return kError;
        }
        Chunk chunk;
        chunk.start = st;
        chunk.count = count;
        aicore_pipeline_timings timings{};
        if (aicore_map_anything_last_pipeline_timings(ctx, &timings) == 0) {
            chunk.e2eMs = timings.e2e_ms;
            out->e2eMs += timings.e2e_ms;
        }

        // ---- boundary fit: raw shared-frame points -> their aligned copy
        if (ci > 0) {
            const int prevEnd = starts[(size_t)ci - 1] +
                                std::min(K, n - starts[(size_t)ci - 1]);
            std::vector<float> src, dst;
            for (int g = st; g < prevEnd; ++g) {
                if (!view_points(ctx, arch, g - st, H, W, options.depthEdgeRtol,
                                 P, error))
                    return kError;
                const std::vector<float>& A = anchors[(size_t)g];
                if (A.size() != P.size()) continue;
                for (size_t k = 0; k < P.size(); k += 3) {
                    if (std::isfinite(P[k]) && std::isfinite(A[k])) {
                        src.insert(src.end(), &P[k], &P[k] + 3);
                        dst.insert(dst.end(), &A[k], &A[k] + 3);
                    }
                }
            }
            for (int g = st; g < prevEnd; ++g) anchors[(size_t)g].clear();
            size_t m = src.size() / 3;
            if (m > kMaxCorrespondences) {
                // Deterministic stride subsample (reproducible fits).
                const size_t step =
                        (m + kMaxCorrespondences - 1) / kMaxCorrespondences;
                size_t w = 0;
                for (size_t k = 0; k < m; k += step, ++w) {
                    std::memmove(&src[w * 3], &src[k * 3], 3 * sizeof(float));
                    std::memmove(&dst[w * 3], &dst[k * 3], 3 * sizeof(float));
                }
                m = w;
                src.resize(m * 3);
                dst.resize(m * 3);
            }
            Sim3& s = chunk.toGlobal;
            if (m < 3 ||
                MapAnythingGeometry::robust_sim3(src.data(), dst.data(), m,
                                                 &s.s, s.R, s.t) != 0) {
                set_error(error, "chunk " + std::to_string(ci) +
                                         ": shared frames give no usable "
                                         "Sim3 anchor (" +
                                         std::to_string(m) + " pixels)");
                return kError;
            }
            std::vector<float> res(m);
            for (size_t k = 0; k < m; ++k) {
                float p[3];
                MapAnythingGeometry::apply_sim3(s.R, s.t, s.s, &src[k * 3], 1,
                                                p);
                res[k] = std::sqrt(
                        (p[0] - dst[k * 3]) * (p[0] - dst[k * 3]) +
                        (p[1] - dst[k * 3 + 1]) * (p[1] - dst[k * 3 + 1]) +
                        (p[2] - dst[k * 3 + 2]) * (p[2] - dst[k * 3 + 2]));
            }
            std::nth_element(res.begin(), res.begin() + m / 2, res.end());
            chunk.correspondences = m;
            chunk.residualMedian = res[m / 2];
        }

        // ---- emit: each global view once, from its first covering chunk
        const float* conf = aicore_map_anything_result_conf(ctx);
        const float* pose = aicore_map_anything_result_pose(ctx);
        const int32_t stride = aicore_map_anything_result_pose_stride(ctx);
        const size_t plane = (size_t)H * W;
        const int nextStart = ci + 1 < nc ? starts[(size_t)ci + 1] : n;
        const ChunkView cv{ctx, ci, nc, st, count};
        for (int i = 0; i < count; ++i) {
            const int g = st + i;
            const bool first = !covered[(size_t)g];
            const bool wantCloud = first && options.buildCloud;
            const bool wantAnchor = ci + 1 < nc && g >= nextStart;
            if (wantCloud || wantAnchor) {
                if (!view_points(ctx, arch, i, H, W, options.depthEdgeRtol, P,
                                 error))
                    return kError;
                apply_to_finite(chunk.toGlobal, P);
            }
            if (wantCloud) {
                const Rect r = valid_rect(ctx, i, H, W);
                const float* vc =
                        conf != nullptr ? conf + (size_t)i * plane : nullptr;
                cloud.populationOffset[(size_t)g] = cloud.confPopulation.size();
                for (int y = r.y0; y < r.y0 + r.h; ++y) {
                    for (int x = r.x0; x < r.x0 + r.w; ++x) {
                        const size_t idx = (size_t)y * W + x;
                        const float c = vc != nullptr ? vc[idx] : 0.0f;
                        if (vc != nullptr && std::isfinite(c))
                            cloud.confPopulation.push_back(c);
                        const float* p = &P[idx * 3];
                        if (!std::isfinite(p[0])) continue;
                        // A NaN confidence cannot be ranked by any rule.
                        if (vc != nullptr && !std::isfinite(c)) continue;
                        uint8_t rgb[3];
                        sample_rgb(views[g], r, x, y, rgb);
                        cloud.xyz.insert(cloud.xyz.end(), p, p + 3);
                        cloud.rgb.insert(cloud.rgb.end(), rgb, rgb + 3);
                        if (vc != nullptr) cloud.conf.push_back(c);
                        cloud.view.push_back(g);
                    }
                }
                cloud.populationOffset[(size_t)g + 1] =
                        cloud.confPopulation.size();
            }
            if (wantAnchor) anchors[(size_t)g] = P;
            if (!first) continue;
            covered[(size_t)g] = 1;
            float* c2w = &out->c2w[(size_t)g * 16];
            if (pose != nullptr && stride == 9) {
                const float* row = pose + (size_t)i * 9;
                MapAnythingGeometry::pose_enc_to_c2w(row, c2w);
                float k[4];
                MapAnythingGeometry::fov_to_intrinsics(row[7], row[8], W, H, k);
                out->focalPx[(size_t)g] = k[0];
            } else if (pose != nullptr && stride == 16) {
                std::memcpy(c2w, pose + (size_t)i * 16, 16 * sizeof(float));
            }
            if (pose != nullptr && (stride == 9 || stride == 16)) {
                MapAnythingGeometry::sim3_apply_c2w(chunk.toGlobal, c2w, c2w);
            }
            if (hook) hook(cv, i, g);
        }

        // Scene-level scalars belong to the global frame = chunk 0's frame.
        if (ci == 0) {
            if (const float* scale = aicore_map_anything_result_scale(ctx)) {
                out->hasMetricScale = true;
                out->metricScale = scale[0];
            }
            int32_t dim = 0;
            const float* emb =
                    aicore_map_anything_result_text_embedding(ctx, &dim);
            if (emb != nullptr && dim > 0) {
                out->textEmbedding.assign(emb, emb + dim);
            }
        }
        out->chunks.push_back(chunk);
    }
    return kOk;
}

FilterReport filter_scene(Cloud& cloud,
                          const std::string& arch,
                          const FilterOptions& o,
                          bool metricFrame) {
    FilterReport rep;
    const size_t n = cloud.size();
    rep.input = n;
    const bool hasConf = !cloud.conf.empty();
    const std::vector<float>& pop = cloud.confPopulation;
    // keep(c, view): the per-architecture official confidence rule.
    std::function<bool(float, int32_t)> keep;
    rep.confFloor = kNaN;
    if (!hasConf) {
        rep.rule = "none";
        keep = [](float, int32_t) { return true; };
    } else if (arch == "mapanything") {
        // MapAnything inference.py: PER VIEW, threshold =
        // torch.quantile(conf[view], p/100) over the whole view, keep
        // conf > threshold (strict).
        rep.rule = "percentile/view";
        const std::vector<size_t>& off = cloud.populationOffset;
        std::vector<float> thr(off.empty() ? 0 : off.size() - 1, kNaN);
        for (size_t g = 0; g < thr.size(); ++g) {
            if (off[g + 1] > off[g]) {
                thr[g] = MapAnythingGeometry::percentile_value(
                        pop.data() + off[g], off[g + 1] - off[g], o.minConf);
            }
        }
        keep = [thr](float c, int32_t v) {
            return (size_t)v < thr.size() && c > thr[(size_t)v];
        };
    } else if (arch == "vggt_omega" || arch == "vggt" || arch == "dust3r") {
        // 1+exp confidences: floor max(2, p) + normalized (c-1)/c gate.
        rep.rule = "floor+normalized";
        const float floor = MapAnythingGeometry::official_conf_floor(
                pop.data(), pop.size(), o.confPercentile);
        rep.confFloor = floor;
        const float minConf = o.minConf;
        keep = [floor, minConf](float c, int32_t) {
            if (!(c >= floor)) return false;
            if (minConf <= 0.0f) return true;
            const float cn = c > 1.0f ? (c - 1.0f) / c : 0.0f;
            return cn >= minConf;
        };
    } else if (arch == "pi3" || arch == "pi3x") {
        // Pi3 publishes raw logits: the official rule is sigmoid(c) > t.
        // A >= 2 floor has no meaning on a logit scale.
        rep.rule = "sigmoid";
        const float minConf = o.minConf;
        keep = [minConf](float c, int32_t) {
            return MapAnythingGeometry::sigmoid(c) > minConf;
        };
    } else {
        rep.ok = false;
        rep.rule = "unknown architecture '" + arch + "'";
        return rep;
    }

    size_t w = 0;
    for (size_t k = 0; k < n; ++k) {
        if (hasConf && !keep(cloud.conf[k], cloud.view[k])) continue;
        if (w != k) {
            std::memcpy(&cloud.xyz[w * 3], &cloud.xyz[k * 3],
                        3 * sizeof(float));
            std::memcpy(&cloud.rgb[w * 3], &cloud.rgb[k * 3], 3);
            if (hasConf) cloud.conf[w] = cloud.conf[k];
            cloud.view[w] = cloud.view[k];
        }
        ++w;
    }
    cloud.xyz.resize(w * 3);
    cloud.rgb.resize(w * 3);
    if (hasConf) cloud.conf.resize(w);
    cloud.view.resize(w);
    rep.kept = w;

    // 0.02 is a metric length: meaningless in a learned, scale-free frame.
    if (metricFrame && o.voxelSize > 0.0f && w > 0) {
        std::vector<float> zeros;
        const float* cf = cloud.conf.data();
        if (!hasConf) {
            zeros.assign(w, 0.0f);
            cf = zeros.data();
        }
        rep.fused = MapAnythingGeometry::conf_priority_voxel_indices(
                cloud.xyz.data(), cf, w, o.voxelSize);
        rep.voxelApplied = true;
    }
    return rep;
}

void transform_result(Result& r, const Sim3& s) {
    MapAnythingGeometry::apply_sim3(s.R, s.t, s.s, r.cloud.xyz.data(),
                                    r.cloud.xyz.size() / 3, r.cloud.xyz.data());
    for (size_t v = 0; v * 16 < r.c2w.size(); ++v) {
        float* c = &r.c2w[v * 16];
        if (std::isfinite(c[0])) MapAnythingGeometry::sim3_apply_c2w(s, c, c);
    }
    for (Chunk& c : r.chunks) {
        c.toGlobal = MapAnythingGeometry::sim3_compose(s, c.toGlobal);
    }
}

bool write_ply(const std::string& path,
               const Cloud& cloud,
               const std::vector<size_t>* indices) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const size_t count = indices != nullptr ? indices->size() : cloud.size();
    const int hn = std::fprintf(f,
                                "ply\n"
                                "format binary_little_endian 1.0\n"
                                "element vertex %zu\n"
                                "property float x\n"
                                "property float y\n"
                                "property float z\n"
                                "property uchar red\n"
                                "property uchar green\n"
                                "property uchar blue\n"
                                "end_header\n",
                                count);
    bool ok = hn > 0;
    for (size_t k = 0; ok && k < count; ++k) {
        const size_t i = indices != nullptr ? (*indices)[k] : k;
        ok = std::fwrite(&cloud.xyz[i * 3], sizeof(float), 3, f) == 3 &&
             std::fwrite(&cloud.rgb[i * 3], 1, 3, f) == 3;
    }
    return std::fclose(f) == 0 && ok;
}

}  // namespace MapAnythingReconstruction
