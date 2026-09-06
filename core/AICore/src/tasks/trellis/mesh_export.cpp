// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mesh_export.h"

#include "print_remesh.h"

#ifdef TRELLIS2_HAVE_CUMESH
#include "cumesh_glue.h"
#endif

#include "meshoptimizer.h"
#include "xatlas.h"

// AICore image encode goes through Qt (QImage built-in PNG codec); the
// upstream stb_image_write dependency is not used in-tree.
#include <QBuffer>
#include <QImage>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <tuple>
#include <unordered_map>

#include "aicore_log.hpp"

namespace t2glb {
namespace {

// Stage logging at DEBUG level (keeps the library quiet by default while
// giving the CLI / debugging a progress trace). Was the T2GLB_VERBOSE
// environment variable upstream: the log level is now owned by the shared
// AICore logging layer (aicore_set_log_level).
double now_s() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
}
#define GLBLOG(...) AICORE_LOG_DEBUG("trellis", "[glb] " __VA_ARGS__)

// Bake stages run seconds-to-minutes and were historically invisible (the
// fine-grained trace above stays at DEBUG), so a stalled bake gave the
// console nothing to attribute the time to (observed: 12.5 s vs 243.9 s for
// same-scale meshes, mesh-shape dependent). BakeStage logs each stage at
// INFO on entry and with its duration on done(); in a log the stage whose
// done-line is missing is the one that stalled.
class BakeStage {
public:
    BakeStage(const char *what, const MeshExportOptions &opt)
        : what_(what), opt_(opt), t0_(now_s()) {
        AICORE_LOG_INFO("trellis", "[glb] bake: %s ...", what_);
    }
    void done() {
        if (done_) return;
        done_ = true;
        const double elapsed = now_s() - t0_;
        AICORE_LOG_INFO("trellis", "[glb] bake: %s done in %.2f s", what_,
                        elapsed);
        if (opt_.progress) opt_.progress(what_, elapsed, opt_.progress_user);
    }
    /** Cooperative-cancel + progress hook for the stage that is about to
     *  run; throws (caught by the C ABI fence) when the cancel flag is set,
     *  so a pathological stage is abandoned at the next boundary. */
    void checkpoint() {
        if (done_) return;
        if (opt_.cancel && *opt_.cancel != 0)
            throw std::runtime_error("bake cancelled");
        // Throttle interim progress to 1 s so long stages do not flood the
        // console (done() carries the final duration).
        const double elapsed = now_s() - t0_;
        if (opt_.progress && elapsed - last_fired_ >= 1.0) {
            last_fired_ = elapsed;
            opt_.progress(what_, elapsed, opt_.progress_user);
        }
    }
    ~BakeStage() { done(); }
    BakeStage(const BakeStage &) = delete;
    BakeStage &operator=(const BakeStage &) = delete;

private:
    const char *what_;
    const MeshExportOptions &opt_;
    double t0_;
    double last_fired_ = 0.0;
    bool done_ = false;
};

// ─────────────────────────────────────────────────────────────────────────
// Copy the original topology while dropping only invalid/degenerate faces and
// unreferenced vertices. Export and showcase deliberately retain full polygon
// density; component cleanup below is the only operation allowed to delete
// valid faces.
// ─────────────────────────────────────────────────────────────────────────
bool copy_mesh(const float *verts,
               int nv,
               const int32_t *tris,
               int nt,
               const float *pbr,
               std::vector<float> &dverts,
               std::vector<int32_t> &dtris,
               std::vector<float> &dpbr,
               std::string &err) {
    std::vector<int> remap((size_t)nv, -1);
    dverts.clear();
    dverts.reserve((size_t)nv * 3);
    dtris.clear();
    dtris.reserve((size_t)nt * 3);
    dpbr.clear();
    if (pbr) dpbr.reserve((size_t)nv * 6);
    for (int t = 0; t < nt; ++t) {
        const int32_t a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
        if (a < 0 || b < 0 || c < 0 || a >= nv || b >= nv || c >= nv) {
            err = "triangle index out of range";
            return false;
        }
        if (a == b || b == c || a == c) continue;
        for (int32_t v : {a, b, c}) {
            if (remap[v] < 0) {
                remap[v] = (int)(dverts.size() / 3);
                dverts.push_back(verts[3 * v + 0]);
                dverts.push_back(verts[3 * v + 1]);
                dverts.push_back(verts[3 * v + 2]);
                if (pbr)
                    dpbr.insert(dpbr.end(), pbr + (size_t)v * 6,
                                pbr + (size_t)v * 6 + 6);
            }
            dtris.push_back(remap[v]);
        }
    }
    return true;
}

// Remove triangles belonging to small connected components (islands). The
// dual-grid surface can contain tiny disconnected bits that each force a
// separate UV chart and may correspond to unwanted background geometry.
// Mirrors the reference's remove_small_connected_components. Keeps components
// with at least `min_frac` of the triangles; recompacts verts/tris in place.
void filter_components(std::vector<float> &v,
                       std::vector<int32_t> &f,
                       std::vector<float> &pbr,
                       float min_frac,
                       ComponentFilter mode) {
    if (mode == ComponentFilter::KeepAll) return;
    const int nv = (int)(v.size() / 3), nt = (int)(f.size() / 3);
    if (nv == 0 || nt == 0) return;
    std::vector<int> parent((size_t)nv), sz((size_t)nv, 1);
    for (int i = 0; i < nv; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto uni = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) std::swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
    };
    for (int t = 0; t < nt; ++t) {
        uni(f[3 * t + 0], f[3 * t + 1]);
        uni(f[3 * t + 1], f[3 * t + 2]);
    }

    std::unordered_map<int, int> tris_in;  // component root -> triangle count
    for (int t = 0; t < nt; ++t) tris_in[find(f[3 * t + 0])]++;
    const int keep_min = std::max(1, (int)(nt * min_frac));
    int largest = -1, largest_n = -1;
    for (const auto &it : tris_in) {
        if (it.second > largest_n) {
            largest = it.first;
            largest_n = it.second;
        }
    }

    std::vector<int> vremap((size_t)nv, -1);
    std::vector<float> nvts;
    nvts.reserve(v.size());
    std::vector<float> npbr;
    if (!pbr.empty()) npbr.reserve(pbr.size());
    std::vector<int32_t> nfs;
    nfs.reserve(f.size());
    for (int t = 0; t < nt; ++t) {
        const int root = find(f[3 * t + 0]);
        if (mode == ComponentFilter::KeepLargest ? root != largest
                                                 : tris_in[root] < keep_min)
            continue;
        for (int k = 0; k < 3; ++k) {
            int vi = f[3 * t + k];
            if (vremap[vi] < 0) {
                vremap[vi] = (int)(nvts.size() / 3);
                nvts.push_back(v[3 * vi + 0]);
                nvts.push_back(v[3 * vi + 1]);
                nvts.push_back(v[3 * vi + 2]);
                if (!pbr.empty())
                    npbr.insert(npbr.end(), pbr.begin() + (size_t)vi * 6,
                                pbr.begin() + (size_t)vi * 6 + 6);
            }
            nfs.push_back(vremap[vi]);
        }
    }
    if (!nfs.empty()) {
        v.swap(nvts);
        f.swap(nfs);
        if (!pbr.empty()) pbr.swap(npbr);
    }
}

// Taubin (λ|μ) smoothing optionally regularises zig-zag sliver triangles so
// per-face normals become coherent. Without this, xatlas may split a chart at
// every normal jump (tens of thousands of one-off charts).
// Taubin's alternating positive/negative passes smooth without the shrinkage of
// plain Laplacian. The texture is unaffected (it samples the dense source).
void taubin_smooth(std::vector<float> &v,
                   const std::vector<int32_t> &f,
                   int iters) {
    const int nv = (int)(v.size() / 3), nt = (int)(f.size() / 3);
    if (nv == 0 || nt == 0) return;
    std::vector<int> deg((size_t)nv, 0);
    // Accumulate neighbour sums per pass; build degree once.
    for (int t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k)
            deg[f[3 * t + k]] += 2;  // each vertex in 2 edges of its triangle
    const float lambda = 0.5f, mu = -0.53f;
    std::vector<float> acc((size_t)nv * 3);
    auto pass = [&](float w) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int t = 0; t < nt; ++t) {
            int a = f[3 * t + 0], b = f[3 * t + 1], c = f[3 * t + 2];
            for (int e = 0; e < 3; ++e) {
                int i = (e == 0   ? a
                         : e == 1 ? b
                                  : c),
                    j = (e == 0   ? b
                         : e == 1 ? c
                                  : a);
                acc[3 * i + 0] += v[3 * j + 0];
                acc[3 * i + 1] += v[3 * j + 1];
                acc[3 * i + 2] += v[3 * j + 2];
                acc[3 * j + 0] += v[3 * i + 0];
                acc[3 * j + 1] += v[3 * i + 1];
                acc[3 * j + 2] += v[3 * i + 2];
            }
        }
        for (int i = 0; i < nv; ++i) {
            if (deg[i] == 0) continue;
            float inv = 1.0f / deg[i];
            for (int k = 0; k < 3; ++k)
                v[3 * i + k] +=
                        w * (acc[3 * i + k] * inv -
                             v[3 * i + k]);  // move toward neighbour centroid
        }
    };
    for (int it = 0; it < iters; ++it) {
        pass(lambda);
        pass(mu);
    }
}

// Area-weighted per-vertex normals over a mesh.
void vertex_normals(const std::vector<float> &v,
                    const std::vector<int32_t> &f,
                    std::vector<float> &n) {
    const size_t nv = v.size() / 3;
    n.assign(nv * 3, 0.0f);
    for (size_t t = 0; t < f.size() / 3; ++t) {
        int a = f[3 * t + 0], b = f[3 * t + 1], c = f[3 * t + 2];
        float e1[3] = {v[3 * b + 0] - v[3 * a + 0], v[3 * b + 1] - v[3 * a + 1],
                       v[3 * b + 2] - v[3 * a + 2]};
        float e2[3] = {v[3 * c + 0] - v[3 * a + 0], v[3 * c + 1] - v[3 * a + 1],
                       v[3 * c + 2] - v[3 * a + 2]};
        float fn[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                       e1[2] * e2[0] - e1[0] * e2[2],
                       e1[0] * e2[1] - e1[1] * e2[0]};
        for (int k : {a, b, c}) {
            n[3 * k + 0] += fn[0];
            n[3 * k + 1] += fn[1];
            n[3 * k + 2] += fn[2];
        }
    }
    for (size_t i = 0; i < nv; ++i) {
        float l = std::sqrt(n[3 * i + 0] * n[3 * i + 0] +
                            n[3 * i + 1] * n[3 * i + 1] +
                            n[3 * i + 2] * n[3 * i + 2]) +
                  1e-20f;
        n[3 * i + 0] /= l;
        n[3 * i + 1] /= l;
        n[3 * i + 2] /= l;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// PNG encode to an in-memory byte vector.
// ─────────────────────────────────────────────────────────────────────────
// PNG encode to an in-memory byte vector via Qt (built-in PNG codec).
bool encode_png(
        int w, int h, int comp, const uint8_t *pix, std::vector<uint8_t> &out) {
    out.clear();
    if (w <= 0 || h <= 0 || comp < 3 || comp > 4 || !pix) return false;
    const QImage::Format fmt =
            comp == 4 ? QImage::Format_RGBA8888 : QImage::Format_RGB888;
    // QImage::Format_RGB888 requires 32-bit row alignment; build a copy
    // with the padded stride when the input row stride is not aligned.
    const int srcStride = w * comp;
    const int dstStride = (w * comp + 3) & ~3;
    QImage img(w, h, fmt);
    if (dstStride == srcStride) {
        std::memcpy(img.bits(), pix, (size_t)srcStride * h);
    } else {
        const uint8_t *src = pix;
        uint8_t *dst = img.bits();
        for (int y = 0; y < h; ++y) {
            std::memcpy(dst, src, (size_t)srcStride);
            src += srcStride;
            dst += dstStride;
        }
    }
    QByteArray bytes;
    QBuffer buf(&bytes);
    if (!buf.open(QIODevice::WriteOnly)) return false;
    if (!img.save(&buf, "PNG")) return false;
    buf.close();
    out.assign(bytes.begin(), bytes.end());
    return !out.empty();
}

// ─────────────────────────────────────────────────────────────────────────
// glTF 2.0 binary (GLB) assembly. One buffer holding, in order: POSITION,
// NORMAL, TEXCOORD_0, indices, baseColor PNG, metallicRoughness PNG.
// ─────────────────────────────────────────────────────────────────────────
void put_u32(std::vector<uint8_t> &b, uint32_t v) {
    b.push_back(v & 0xFF);
    b.push_back((v >> 8) & 0xFF);
    b.push_back((v >> 16) & 0xFF);
    b.push_back((v >> 24) & 0xFF);
}
void pad4(std::vector<uint8_t> &b, uint8_t fill) {
    while (b.size() % 4) b.push_back(fill);
}

void write_glb(const std::vector<float> &pos,
               const std::vector<float> &nrm,
               const std::vector<float> &uv,
               const std::vector<uint32_t> &idx,
               const std::vector<uint8_t> &basecolor_png,
               const std::vector<uint8_t> &metalrough_png,
               bool transparent,
               std::vector<uint8_t> &out) {
    const uint32_t nv = (uint32_t)(pos.size() / 3);
    const uint32_t ni = (uint32_t)idx.size();

    // Assemble the BIN buffer, tracking 4-aligned bufferView offsets.
    std::vector<uint8_t> bin;
    auto view = [&](const void *data, size_t bytes, uint32_t &off,
                    uint32_t &len) {
        pad4(bin, 0);
        off = (uint32_t)bin.size();
        len = (uint32_t)bytes;
        const uint8_t *p = (const uint8_t *)data;
        bin.insert(bin.end(), p, p + bytes);
    };
    uint32_t off_pos, len_pos, off_nrm, len_nrm, off_uv, len_uv, off_idx,
            len_idx;
    uint32_t off_bc, len_bc, off_mr, len_mr;
    view(pos.data(), pos.size() * 4, off_pos, len_pos);
    view(nrm.data(), nrm.size() * 4, off_nrm, len_nrm);
    view(uv.data(), uv.size() * 4, off_uv, len_uv);
    view(idx.data(), idx.size() * 4, off_idx, len_idx);
    view(basecolor_png.data(), basecolor_png.size(), off_bc, len_bc);
    view(metalrough_png.data(), metalrough_png.size(), off_mr, len_mr);
    pad4(bin, 0);

    float pmin[3] = {1e30f, 1e30f, 1e30f}, pmax[3] = {-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k) {
            pmin[k] = std::min(pmin[k], pos[3 * i + k]);
            pmax[k] = std::max(pmax[k], pos[3 * i + k]);
        }

    char buf[2048];
    std::string j =
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis2cpp\"},";
    j += "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],";
    j += "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,"
         "\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}],";
    j += "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{"
         "\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},"
         "\"metallicFactor\":1.0,\"roughnessFactor\":1.0},";
    if (transparent) j += "\"alphaMode\":\"BLEND\",";
    j += "\"doubleSided\":true}],";
    j += "\"textures\":[{\"source\":0,\"sampler\":0},{\"source\":1,\"sampler\":"
         "0}],";
    j += "\"images\":[{\"bufferView\":4,\"mimeType\":\"image/"
         "png\"},{\"bufferView\":5,\"mimeType\":\"image/png\"}],";
    // No mipmaps: opt-in xatlas charts use explicit dilation around their
    // seams.
    j += "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,"
         "\"wrapT\":33071}],";
    snprintf(buf, sizeof buf,
             "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,"
             "\"count\":%u,\"type\":\"VEC3\",\"min\":[%.8g,%.8g,%.8g],\"max\":["
             "%.8g,%.8g,%.8g]},"
             "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,\"type\":"
             "\"VEC3\"},"
             "{\"bufferView\":2,\"componentType\":5126,\"count\":%u,\"type\":"
             "\"VEC2\"},"
             "{\"bufferView\":3,\"componentType\":5125,\"count\":%u,\"type\":"
             "\"SCALAR\"}],",
             nv, pmin[0], pmin[1], pmin[2], pmax[0], pmax[1], pmax[2], nv, nv,
             ni);
    j += buf;
    snprintf(buf, sizeof buf,
             "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%"
             "u,\"target\":34962},"
             "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":"
             "34962},"
             "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":"
             "34962},"
             "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":"
             "34963},"
             "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u},"
             "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],",
             off_pos, len_pos, off_nrm, len_nrm, off_uv, len_uv, off_idx,
             len_idx, off_bc, len_bc, off_mr, len_mr);
    j += buf;
    snprintf(buf, sizeof buf, "\"buffers\":[{\"byteLength\":%u}]}",
             (uint32_t)bin.size());
    j += buf;

    std::vector<uint8_t> json(j.begin(), j.end());
    while (json.size() % 4) json.push_back(0x20);  // pad JSON chunk with spaces

    const uint32_t total =
            12 + 8 + (uint32_t)json.size() + 8 + (uint32_t)bin.size();
    out.clear();
    out.reserve(total);
    put_u32(out, 0x46546C67);
    put_u32(out, 2);
    put_u32(out, total);  // header
    put_u32(out, (uint32_t)json.size());
    put_u32(out, 0x4E4F534A);  // JSON chunk
    out.insert(out.end(), json.begin(), json.end());
    put_u32(out, (uint32_t)bin.size());
    put_u32(out, 0x004E4942);  // BIN chunk
    out.insert(out.end(), bin.begin(), bin.end());
}

// Portable full-density export. glTF natively supports interpolated RGBA
// vertex colours, which are a much better representation for TRELLIS' dense
// per-vertex material field than assigning millions of triangles to a finite
// 2D atlas. Metallic/roughness have no standard per-vertex glTF semantics, so
// their averages drive the standard material while the original quantised
// values are retained in the application attribute _METALLIC_ROUGHNESS.
void write_vertex_glb(const PreparedMesh &mesh, std::vector<uint8_t> &out) {
    const uint32_t nv = (uint32_t)(mesh.verts.size() / 3);
    const uint32_t ni = (uint32_t)mesh.tris.size();
    const bool textured = mesh.pbr.size() == (size_t)nv * 6;
    auto to8 = [](float f) {
        int v = (int)std::lround(f * 255.0f);
        return (uint8_t)std::min(255, std::max(0, v));
    };
    auto clamp01 = [](float f) { return std::min(1.0f, std::max(0.0f, f)); };
    auto srgb_to_linear = [&](float f) {
        f = clamp01(f);
        return f <= 0.04045f ? f / 12.92f
                             : std::pow((f + 0.055f) / 1.055f, 2.4f);
    };
    auto to16 = [](float f) {
        int v = (int)std::lround(f * 65535.0f);
        return (uint16_t)std::min(65535, std::max(0, v));
    };

    std::vector<float> pos((size_t)nv * 3), nrm((size_t)nv * 3);
    // glTF COLOR_0 is linear rather than sRGB. Sixteen-bit UNORM keeps dark
    // generated colours precise after converting from the texture model's
    // sRGB-like output.
    std::vector<uint16_t> color((size_t)nv * 4);
    std::vector<uint8_t> metalrough((size_t)nv * 2);
    std::vector<uint32_t> idx(ni);
    double metal_sum = 0.0, rough_sum = 0.0, weight_sum = 0.0;
    uint32_t translucent = 0;
    for (uint32_t i = 0; i < nv; ++i) {
        // Trellis -> glTF Y-up, preserving handedness.
        pos[3 * i + 0] = mesh.verts[3 * i + 0];
        pos[3 * i + 1] = mesh.verts[3 * i + 2];
        pos[3 * i + 2] = -mesh.verts[3 * i + 1];
        nrm[3 * i + 0] = mesh.normals[3 * i + 0];
        nrm[3 * i + 1] = mesh.normals[3 * i + 2];
        nrm[3 * i + 2] = -mesh.normals[3 * i + 1];
        const float *p = textured ? mesh.pbr.data() + (size_t)i * 6 : nullptr;
        const float r = p ? p[0] : 0.7f, g = p ? p[1] : 0.7f,
                    b = p ? p[2] : 0.7f;
        const float m = p ? clamp01(p[3]) : 0.0f;
        const float ro = p ? clamp01(p[4]) : 0.6f;
        const float a = p ? clamp01(p[5]) : 1.0f;
        color[4 * i + 0] = to16(srgb_to_linear(r));
        color[4 * i + 1] = to16(srgb_to_linear(g));
        color[4 * i + 2] = to16(srgb_to_linear(b));
        color[4 * i + 3] = to16(a);
        metalrough[2 * i + 0] = to8(m);
        metalrough[2 * i + 1] = to8(ro);
        // Near-transparent samples should not skew the visible material's
        // standard fallback factors.
        const double w = std::max(0.001f, a);
        metal_sum += m * w;
        rough_sum += ro * w;
        weight_sum += w;
        if (a < 0.95f) ++translucent;
    }
    for (uint32_t i = 0; i < ni; ++i) idx[i] = (uint32_t)mesh.tris[i];
    const float metallic = weight_sum ? (float)(metal_sum / weight_sum) : 0.0f;
    const float roughness = weight_sum ? (float)(rough_sum / weight_sum) : 0.6f;
    // BLEND disables normal opaque depth writes in many viewers. Do not switch
    // an entire multi-million-face primitive to blending because of a handful
    // of near-one decoder outliers; upstream likewise exports ordinary surfaces
    // as opaque. Keep blending for a material with meaningful transparency.
    const bool transparent =
            translucent > 0 && (uint64_t)translucent * 1000 >= (uint64_t)nv;

    std::vector<uint8_t> bin;
    auto view = [&](const void *data, size_t bytes, uint32_t &off,
                    uint32_t &len) {
        pad4(bin, 0);
        off = (uint32_t)bin.size();
        len = (uint32_t)bytes;
        const uint8_t *p = (const uint8_t *)data;
        bin.insert(bin.end(), p, p + bytes);
    };
    uint32_t off_pos, len_pos, off_nrm, len_nrm, off_col, len_col;
    uint32_t off_mr, len_mr, off_idx, len_idx;
    view(pos.data(), pos.size() * 4, off_pos, len_pos);
    view(nrm.data(), nrm.size() * 4, off_nrm, len_nrm);
    view(color.data(), color.size() * sizeof(uint16_t), off_col, len_col);
    view(metalrough.data(), metalrough.size(), off_mr, len_mr);
    view(idx.data(), idx.size() * 4, off_idx, len_idx);
    pad4(bin, 0);

    float pmin[3] = {1e30f, 1e30f, 1e30f}, pmax[3] = {-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k) {
            pmin[k] = std::min(pmin[k], pos[3 * i + k]);
            pmax[k] = std::max(pmax[k], pos[3 * i + k]);
        }

    char buf[3072];
    std::string j =
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"trellis2cpp\"},";
    j += "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],";
    j += "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,"
         "\"NORMAL\":1,\"COLOR_0\":2,\"_METALLIC_ROUGHNESS\":3},\"indices\":4,"
         "\"material\":0}]}],";
    std::snprintf(
            buf, sizeof buf,
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,"
            "1,1,1],\"metallicFactor\":%.8g,\"roughnessFactor\":%.8g},",
            metallic, roughness);
    j += buf;
    if (transparent) j += "\"alphaMode\":\"BLEND\",";
    j += "\"doubleSided\":true}],";
    std::snprintf(buf, sizeof buf,
                  "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,"
                  "\"count\":%u,\"type\":\"VEC3\",\"min\":[%.8g,%.8g,%.8g],"
                  "\"max\":[%.8g,%.8g,%.8g]},"
                  "{\"bufferView\":1,\"componentType\":5126,\"count\":%u,"
                  "\"type\":\"VEC3\"},"
                  "{\"bufferView\":2,\"componentType\":5123,\"normalized\":"
                  "true,\"count\":%u,\"type\":\"VEC4\"},"
                  "{\"bufferView\":3,\"componentType\":5121,\"normalized\":"
                  "true,\"count\":%u,\"type\":\"VEC2\"},"
                  "{\"bufferView\":4,\"componentType\":5125,\"count\":%u,"
                  "\"type\":\"SCALAR\"}],",
                  nv, pmin[0], pmin[1], pmin[2], pmax[0], pmax[1], pmax[2], nv,
                  nv, nv, ni);
    j += buf;
    std::snprintf(buf, sizeof buf,
                  "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":%u,"
                  "\"byteLength\":%u,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,"
                  "\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,"
                  "\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,"
                  "\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,"
                  "\"target\":34963}],",
                  off_pos, len_pos, off_nrm, len_nrm, off_col, len_col, off_mr,
                  len_mr, off_idx, len_idx);
    j += buf;
    std::snprintf(buf, sizeof buf, "\"buffers\":[{\"byteLength\":%u}]}",
                  (uint32_t)bin.size());
    j += buf;

    std::vector<uint8_t> json(j.begin(), j.end());
    while (json.size() % 4) json.push_back(0x20);
    const uint32_t total =
            12 + 8 + (uint32_t)json.size() + 8 + (uint32_t)bin.size();
    out.clear();
    out.reserve(total);
    put_u32(out, 0x46546C67);
    put_u32(out, 2);
    put_u32(out, total);
    put_u32(out, (uint32_t)json.size());
    put_u32(out, 0x4E4F534A);
    out.insert(out.end(), json.begin(), json.end());
    put_u32(out, (uint32_t)bin.size());
    put_u32(out, 0x004E4942);
    out.insert(out.end(), bin.begin(), bin.end());
}

// Chart-based unwrap via xatlas → per-output-vertex position/normal/uv (texel
// space) + index buffer + atlas dims. Proper editable charts, but only fast on
// clean, manifold meshes. Opt-in with T2GLB_XATLAS.
bool xatlas_unwrap(const std::vector<float> &dverts,
                   const std::vector<float> &dnrm,
                   const std::vector<float> &dpbr,
                   int dnv,
                   const std::vector<int32_t> &dtris,
                   int dnt,
                   const MeshExportOptions &opt,
                   int TS,
                   std::vector<float> &opos,
                   std::vector<float> &onrm,
                   std::vector<float> &ouv,
                   std::vector<float> &opbr,
                   std::vector<uint32_t> &oidx,
                   int &AW,
                   int &AH,
                   std::string &err) {
    xatlas::Atlas *atlas = xatlas::Create();
    xatlas::MeshDecl md{};
    md.vertexCount = (uint32_t)dnv;
    md.vertexPositionData = dverts.data();
    md.vertexPositionStride = sizeof(float) * 3;
    md.indexCount = (uint32_t)(dnt * 3);
    md.indexData = dtris.data();
    md.indexFormat = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, md) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        err = "xatlas AddMesh failed";
        return false;
    }
    GLBLOG("xatlas computing charts ...");
    xatlas::ChartOptions co{};
    co.normalDeviationWeight = 0.5f;
    co.roundnessWeight = 0.0f;
    co.straightnessWeight = 1.0f;
    co.normalSeamWeight = 1.0f;
    co.textureSeamWeight = 0.0f;
    co.maxCost = 8.0f;
    co.maxIterations = 1;
    xatlas::ComputeCharts(atlas, co);

    xatlas::PackOptions po{};
    po.padding = (uint32_t)opt.padding;
    po.bilinear = true;
    po.createImage = false;
    uint32_t res = (uint32_t)TS;
    for (int attempt = 0; attempt < 4; ++attempt) {
        po.resolution = res;
        xatlas::PackCharts(atlas, po);
        if (atlas->atlasCount <= 1) break;
        res = (uint32_t)(res * 1.5f);
    }
    GLBLOG("atlas %ux%u, %u pages, %u charts", atlas->width, atlas->height,
           atlas->atlasCount, atlas->chartCount);
    if (atlas->meshCount != 1 || atlas->atlasCount != 1 || atlas->width == 0 ||
        atlas->height == 0) {
        xatlas::Destroy(atlas);
        err = "xatlas packing failed (charts did not fit one atlas)";
        return false;
    }
    AW = (int)atlas->width;
    AH = (int)atlas->height;
    const xatlas::Mesh &om = atlas->meshes[0];
    const uint32_t onv = om.vertexCount;
    opos.resize(onv * 3);
    onrm.resize(onv * 3);
    ouv.resize(onv * 2);
    if (!dpbr.empty()) opbr.resize(onv * 6);
    for (uint32_t i = 0; i < onv; ++i) {
        uint32_t xr = om.vertexArray[i].xref;
        if (xr >= (uint32_t)dnv) xr = 0;
        opos[3 * i + 0] = dverts[3 * xr + 0];
        opos[3 * i + 1] = dverts[3 * xr + 1];
        opos[3 * i + 2] = dverts[3 * xr + 2];
        onrm[3 * i + 0] = dnrm[3 * xr + 0];
        onrm[3 * i + 1] = dnrm[3 * xr + 1];
        onrm[3 * i + 2] = dnrm[3 * xr + 2];
        ouv[2 * i + 0] = om.vertexArray[i].uv[0];
        ouv[2 * i + 1] = om.vertexArray[i].uv[1];
        if (!dpbr.empty())
            std::memcpy(opbr.data() + (size_t)i * 6,
                        dpbr.data() + (size_t)xr * 6, 6 * sizeof(float));
    }
    oidx.assign(om.indexArray, om.indexArray + om.indexCount);
    xatlas::Destroy(atlas);
    return true;
}

// Chartless UV unwrap: group triangles by dominant normal axis (6 bins),
// project each bin onto its own plane, and pack the bins into one atlas with a
// simple skyline layout. Runs in seconds on sloppy-decimated / non-manifold
// meshes where xatlas charting is pathologically slow; the trade-off is UV
// seams at bin boundaries instead of per-chart seams.
bool simple_unwrap(const std::vector<float> &dverts,
                   const std::vector<float> &dnrm,
                   const std::vector<float> &dpbr,
                   int dnv,
                   const std::vector<int32_t> &dtris,
                   int dnt,
                   int TS,
                   int padding,
                   std::vector<float> &opos,
                   std::vector<float> &onrm,
                   std::vector<float> &ouv,
                   std::vector<float> &opbr,
                   std::vector<uint32_t> &oidx,
                   int &AW,
                   int &AH,
                   std::string &err) {
    if (dnt <= 0) {
        err = "empty mesh";
        return false;
    }

    // Per-triangle dominant axis (0..5 = +x,-x,+y,-y,+z,-z).
    std::vector<uint8_t> tbin((size_t)dnt);
    std::vector<uint64_t>
            vbin;  // (vertex * 6 + bin) -> output vertex, built below
    std::unordered_map<uint64_t, uint32_t> vmap;
    vmap.reserve((size_t)dnv * 2);

    auto norm_axis = [](float nx, float ny, float nz) -> int {
        const float ax = std::fabs(nx), ay = std::fabs(ny), az = std::fabs(nz);
        if (ax >= ay && ax >= az) return nx >= 0 ? 0 : 1;
        if (ay >= az) return ny >= 0 ? 2 : 3;
        return nz >= 0 ? 4 : 5;
    };
    auto key = [](uint32_t v, int b) -> uint64_t {
        return ((uint64_t)v << 3) | (uint32_t)b;
    };

    // Bin-local 2D projection basis: u/v are the two axes orthogonal to the
    // dominant axis, ordered so all six bins keep a consistent handedness.
    static const int BASES[6][2] = {
            {1, 2}, {1, 2}, {0, 2}, {0, 2}, {0, 1}, {0, 1},  // axis, u, v
    };
    std::vector<float> bin_minx(6, FLT_MAX), bin_miny(6, FLT_MAX);
    std::vector<float> bin_maxx(6, -FLT_MAX), bin_maxy(6, -FLT_MAX);
    std::vector<float> bin_area(6, 0.0f);

    for (int t = 0; t < dnt; ++t) {
        const int32_t ia = dtris[3 * t + 0], ib = dtris[3 * t + 1],
                      ic = dtris[3 * t + 2];
        const float *a = &dverts[(size_t)ia * 3];
        const float *b = &dverts[(size_t)ib * 3];
        const float *c = &dverts[(size_t)ic * 3];
        float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz,
              nz = ux * vy - uy * vx;
        const float nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nlen < 1e-12f) continue;
        const int bin = norm_axis(nx, ny, nz);
        tbin[(size_t)t] = (uint8_t)bin;
        const int au = BASES[bin][0], av = BASES[bin][1];
        const float ax2 = a[au], ay2 = a[av];
        const float bx2 = b[au], by2 = b[av];
        const float cx2 = c[au], cy2 = c[av];
        const float mnx = std::min({ax2, bx2, cx2}),
                    mny = std::min({ay2, by2, cy2});
        const float mxx = std::max({ax2, bx2, cx2}),
                    mxy = std::max({ay2, by2, cy2});
        bin_minx[bin] = std::min(bin_minx[bin], mnx);
        bin_miny[bin] = std::min(bin_miny[bin], mny);
        bin_maxx[bin] = std::max(bin_maxx[bin], mxx);
        bin_maxy[bin] = std::max(bin_maxy[bin], mxy);
        // projected area via 2D cross
        const float ar = (bx2 - ax2) * (cy2 - ay2) - (by2 - ay2) * (cx2 - ax2);
        bin_area[bin] += std::fabs(ar) * 0.5f;
    }

    // Scale each bin's UV so its projected area maps to a texel budget
    // proportional to the mesh area, leaving padding for seams.
    float tot_area = 0.0f;
    for (int b = 0; b < 6; ++b) tot_area += bin_area[b];
    if (tot_area <= 0.0f) {
        err = "degenerate geometry";
        return false;
    }
    const float pad = (float)padding;
    const float TSf = (float)TS;
    const float budget = TSf - 4.0f * pad;  // usable texel span per side

    // Pack the six bins into a 2x3 grid: bins sorted by area fill rows left/
    // right, row heights are proportional to the row's total area, and each
    // cell letterboxes its bin preserving aspect ratio. Simpler and more robust
    // than skyline for the typical case of two large planar bins (front/back)
    // plus four slim side bins.
    struct BinOrder {
        int idx;
        float area;
    };
    BinOrder bo[6];
    for (int b = 0; b < 6; ++b) bo[b] = {b, bin_area[b]};
    std::sort(bo, bo + 6, [](const BinOrder &x, const BinOrder &y) {
        return x.area > y.area;
    });
    const float cw = TSf / 2.0f;
    float row_top[3] = {0.0f, 0.0f, 0.0f};
    for (int r = 0; r < 3; ++r)
        row_top[r] = (r == 0 ? 0.0f : row_top[r - 1]) +
                     (bo[2 * r].area + bo[2 * r + 1].area) / tot_area * TSf;
    struct Place {
        float x, y, w, h;
        float sx, sy, ox, oy;
        bool used;
    };
    Place P[6] = {};
    for (int r = 0; r < 3; ++r) {
        const float cy = row_top[r];
        const float chh = (r == 2 ? TSf : row_top[r + 1]) - cy;
        for (int c = 0; c < 2; ++c) {
            const int b = bo[2 * r + c].idx;
            if (bin_area[b] <= 0.0f) continue;
            const float bw = std::max(bin_maxx[b] - bin_minx[b], 1e-6f);
            const float bh = std::max(bin_maxy[b] - bin_miny[b], 1e-6f);
            const float cx = c * cw;
            float s = std::min((cw - 2.0f * pad) / bw, (chh - 2.0f * pad) / bh);
            if (s < 0.0f) s = 0.0f;
            const float w = bw * s, h = bh * s;
            const float ox = cx + pad + (cw - 2.0f * pad - w) * 0.5f;
            const float oy = cy + pad + (chh - 2.0f * pad - h) * 0.5f;
            P[b] = {ox, oy, w, h, s, s, ox, oy, true};
        }
    }

    // Build output vertices: each (input vertex, bin) pair gets one output
    // vertex; bin-projected UV is normalized into the bin's packed rectangle.
    opos.clear();
    onrm.clear();
    ouv.clear();
    opbr.clear();
    oidx.clear();
    opos.reserve((size_t)dnv * 2 * 3);
    oidx.reserve((size_t)dnt * 3);
    for (int t = 0; t < dnt; ++t) {
        const int bin = tbin[(size_t)t];
        if (!P[bin].used) continue;
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = (uint32_t)dtris[3 * t + k];
            const auto it = vmap.find(key(v, bin));
            uint32_t ov;
            if (it != vmap.end()) {
                ov = it->second;
            } else {
                ov = (uint32_t)(opos.size() / 3);
                vmap.emplace(key(v, bin), ov);
                const float *p = &dverts[(size_t)v * 3];
                opos.insert(opos.end(), {p[0], p[1], p[2]});
                if (!dnrm.empty()) {
                    const float *n = &dnrm[(size_t)v * 3];
                    onrm.insert(onrm.end(), {n[0], n[1], n[2]});
                }
                if (!dpbr.empty())
                    opbr.insert(opbr.end(), dpbr.begin() + (size_t)v * 6,
                                dpbr.begin() + (size_t)v * 6 + 6);
                const int au = BASES[bin][0], av = BASES[bin][1];
                const float u = (p[au] - bin_minx[bin]) * P[bin].sx + P[bin].ox;
                const float vv =
                        (p[av] - bin_miny[bin]) * P[bin].sy + P[bin].oy;
                ouv.insert(ouv.end(), {u, vv});
            }
            oidx.push_back(ov);
        }
    }
    AW = TS;
    AH = TS;
    GLBLOG("simple unwrap: %u verts, %u tris, 6 bins packed @%ux%u",
           (unsigned)(opos.size() / 3), (unsigned)(oidx.size() / 3), AW, AH);
    return true;
}

// Shared tail of both chart-clustering unwrap paths (the CPU normal-cone
// clusterer and CuMesh GPU below): per-face chart ids are handed to xatlas
// as hard chart boundaries via faceMaterialData, then xatlas only has to
// parameterize and pack — the expensive autonomous charting stage is skipped,
// which is what makes this path finish in seconds where bare xatlas takes
// minutes.
static bool xatlas_unwrap_with_face_material(
        const std::vector<float> &dverts,
        const std::vector<float> &dnrm,
        const std::vector<float> &dpbr,
        int dnv,
        const std::vector<int32_t> &dtris,
        int nt_actual,
        const std::vector<uint32_t> &face_mat,
        const MeshExportOptions &opt,
        int TS,
        std::vector<float> &opos,
        std::vector<float> &onrm,
        std::vector<float> &ouv,
        std::vector<float> &opbr,
        std::vector<uint32_t> &oidx,
        int &AW,
        int &AH,
        std::string &err) {
    if (nt_actual <= 0 || face_mat.empty()) {
        err = "no charts";
        return false;
    }
    const double t0 = now_s();

    // One xatlas mesh with faceMaterialData = chart ids. xatlas treats each
    // unique material as one chart boundary.
    xatlas::Atlas *atlas = xatlas::Create();
    {
        xatlas::MeshDecl md{};
        md.vertexCount = (uint32_t)dnv;
        md.vertexPositionData = dverts.data();
        md.vertexPositionStride = sizeof(float) * 3;
        md.indexCount = (uint32_t)((size_t)nt_actual * 3);
        md.indexData = dtris.data();
        md.indexFormat = xatlas::IndexFormat::UInt32;
        md.faceMaterialData = face_mat.data();
        md.faceCount = (uint32_t)nt_actual;
        if (xatlas::AddMesh(atlas, md) != xatlas::AddMeshError::Success) {
            xatlas::Destroy(atlas);
            err = "xatlas AddMesh with faceMaterialData failed";
            return false;
        }
    }

    // ComputeCharts with lenient settings — faceMaterialData already defines
    // hard chart boundaries, so maxCost = 100 prevents xatlas from splitting.
    GLBLOG("xatlas per-chart parameterization (%u meshes) ...",
           atlas->meshCount);
    {
        xatlas::ChartOptions co{};
        co.normalDeviationWeight = 1.0f;
        co.roundnessWeight = 0.0f;
        co.straightnessWeight = 0.0f;
        co.normalSeamWeight = 1.0f;
        co.textureSeamWeight = 0.0f;
        co.maxCost = 100.0f;   // very lenient → avoid splits
        co.maxIterations = 0;  // skip re-clustering, use material boundaries
        co.fixWinding = true;
        xatlas::ComputeCharts(atlas, co);
    }

    GLBLOG("xatlas packing %u charts ...", atlas->chartCount);
    {
        xatlas::PackOptions po{};
        po.padding = 0;  // our bake dilate handles chart seams
        po.bilinear = false;
        po.createImage = false;
        uint32_t res = (uint32_t)TS;
        for (int attempt = 0; attempt < 6; ++attempt) {
            po.resolution = res;
            xatlas::PackCharts(atlas, po);
            if (atlas->atlasCount <= 1) break;
            res = (uint32_t)(res * 1.5f);
        }
    }
    GLBLOG("atlas %ux%u, %u pages, %u charts (%.2fs)", atlas->width,
           atlas->height, atlas->atlasCount, atlas->chartCount, now_s() - t0);

    if (atlas->meshCount == 0 || atlas->atlasCount != 1 || atlas->width == 0 ||
        atlas->height == 0) {
        xatlas::Destroy(atlas);
        err = "xatlas packing failed (charts did not fit one atlas)";
        return false;
    }

    AW = (int)atlas->width;
    AH = (int)atlas->height;

    // Read back the single xatlas mesh.
    const xatlas::Mesh &xm = atlas->meshes[0];
    const uint32_t onv = xm.vertexCount;

    opos.resize((size_t)onv * 3);
    onrm.resize((size_t)onv * 3);
    ouv.resize((size_t)onv * 2);
    if (!dpbr.empty()) opbr.resize((size_t)onv * 6);

    for (uint32_t i = 0; i < onv; ++i) {
        const uint32_t orig = xm.vertexArray[i].xref;
        const uint32_t src = (orig < (uint32_t)dnv) ? orig : 0;
        opos[(size_t)3 * i + 0] = dverts[(size_t)3 * src + 0];
        opos[(size_t)3 * i + 1] = dverts[(size_t)3 * src + 1];
        opos[(size_t)3 * i + 2] = dverts[(size_t)3 * src + 2];
        onrm[(size_t)3 * i + 0] = dnrm[(size_t)3 * src + 0];
        onrm[(size_t)3 * i + 1] = dnrm[(size_t)3 * src + 1];
        onrm[(size_t)3 * i + 2] = dnrm[(size_t)3 * src + 2];
        ouv[(size_t)2 * i + 0] = xm.vertexArray[i].uv[0];
        ouv[(size_t)2 * i + 1] = xm.vertexArray[i].uv[1];
        if (!dpbr.empty())
            std::memcpy(&opbr[(size_t)i * 6], &dpbr[(size_t)src * 6],
                        6 * sizeof(float));
    }

    oidx.assign(xm.indexArray, xm.indexArray + xm.indexCount);

    GLBLOG("chart unwrap: %u verts, %u tris, %u charts packed @%dx%d (%.2fs)",
           onv, (unsigned)(oidx.size() / 3),
           (unsigned)(face_mat.empty()
                              ? 0u
                              : 1u + *std::max_element(face_mat.begin(),
                                                       face_mat.end())),
           AW, AH, now_s() - t0);
    xatlas::Destroy(atlas);
    return true;
}

// CuMesh GPU chart clustering + xatlas per-chart parameterization + packing.
// Uses the same pyramid as the Python reference (cumesh/cumesh.py::uv_unwrap):
//
//   CuMesh.compute_charts          — GPU normal-cone clustering
//   xatlas faceMaterialData        — force chart boundaries from CuMesh
//   xatlas::ComputeCharts          — per-chart parameterization
//   xatlas::PackCharts             — optimal packing
//
// The env-var T2GLB_NOCUMESH disables this and falls back to the CPU clusterer.
#ifdef TRELLIS2_HAVE_CUMESH
static bool cumesh_unwrap(const std::vector<float> &dverts,
                          const std::vector<float> &dnrm,
                          const std::vector<float> &dpbr,
                          int dnv,
                          const std::vector<int32_t> &dtris,
                          int dnt,
                          const MeshExportOptions &opt,
                          int TS,
                          std::vector<float> &opos,
                          std::vector<float> &onrm,
                          std::vector<float> &ouv,
                          std::vector<float> &opbr,
                          std::vector<uint32_t> &oidx,
                          int &AW,
                          int &AH,
                          std::string &err) {
    if (dnt <= 0) {
        err = "empty mesh";
        return false;
    }
    const double t0 = now_s();
    const float cone_half_angle = 1.57f;

    int *face_cid = nullptr, *chart_vmap = nullptr;
    int *chart_faces = nullptr, *chart_faces_off = nullptr,
        *chart_vert_off = nullptr;
    int n_charts = 0, nt_out = 0;

    if (!cumesh_compute_charts(
                dverts.data(), dnv, dtris.data(), dnt, cone_half_angle, 100, 3,
                1.0f, -0.1f, 0.0f, &face_cid, &chart_vmap, &chart_faces,
                &chart_faces_off, &chart_vert_off, &n_charts, &nt_out)) {
        err = "CuMesh compute_charts failed on GPU";
        return false;
    }

    // Save CuMesh mapping arrays BEFORE we free them.
    std::vector<int> saved_vmap(chart_vmap, chart_vmap + (size_t)dnv);
    std::vector<int> saved_vert_off(chart_vert_off,
                                    chart_vert_off + (size_t)(n_charts + 1));
    std::vector<int> saved_face_off(chart_faces_off,
                                    chart_faces_off + (size_t)(n_charts + 1));
    std::vector<int> saved_face_cid(face_cid, face_cid + (size_t)nt_out);

    // CuMesh returns chart_faces which separates faces per chart, but the
    // face_cid maps every output face back to its chart.  We need the
    // face_cid in original-triangle order.  CuMesh's remove_degenerate_faces
    // may drop faces, so the nt_out in face_cid may differ from dnt.  Our
    // decimated mesh is already clean, so nt_out == dnt for practical cases.
    // We build face_material from face_cid (the chart assignment) and feed
    // the ORIGINAL mesh indices to xatlas (the decimated geometry is already
    // a clean manifold mesh).  Per-face material ID makes each CuMesh chart
    // a hard xatlas chart boundary.
    const int nt_actual = (int)saved_face_cid.size();

    GLBLOG("CuMesh clustered %d -> %d tris into %d charts (%.2fs)", dnt, nt_out,
           n_charts, now_s() - t0);

    if (n_charts <= 0 || nt_actual <= 0) {
        cumesh_free_buffer(face_cid);
        cumesh_free_buffer(chart_vmap);
        cumesh_free_buffer(chart_faces);
        cumesh_free_buffer(chart_faces_off);
        cumesh_free_buffer(chart_vert_off);
        err = "CuMesh produced zero charts";
        return false;
    }

    // Build face_material from face_cid (the chart assignment) and feed the
    // ORIGINAL mesh indices to xatlas (the decimated geometry is already a
    // clean manifold mesh).  Per-face material ID makes each CuMesh chart a
    // hard xatlas chart boundary.
    std::vector<uint32_t> face_mat((size_t)nt_actual);
    for (int i = 0; i < nt_actual; ++i)
        face_mat[i] = (uint32_t)saved_face_cid[i];

    // CuMesh buffers no longer needed.
    cumesh_free_buffer(face_cid);
    cumesh_free_buffer(chart_vmap);
    cumesh_free_buffer(chart_faces);
    cumesh_free_buffer(chart_faces_off);
    cumesh_free_buffer(chart_vert_off);

    const bool ok = xatlas_unwrap_with_face_material(
            dverts, dnrm, dpbr, dnv, dtris, nt_actual, face_mat, opt, TS, opos,
            onrm, ouv, opbr, oidx, AW, AH, err);
    if (ok)
        GLBLOG("cumesh unwrap: %u verts, %u tris, %d charts packed @%dx%d "
               "(%.2fs)",
               (unsigned)(opos.size() / 3), (unsigned)(oidx.size() / 3),
               n_charts, AW, AH, now_s() - t0);
    return ok;
}

#endif  // TRELLIS2_HAVE_CUMESH

// ─────────────────────────────────────────────────────────────────────────
// CPU normal-cone chart clustering — a dependency-free port of CuMesh's
// compute_charts (third_party/CuMesh/src/atlas.cu), i.e. the exact algorithm
// behind the reference UV path, so the libtorch-free build gets the same
// chart quality as the CuMesh-linked one. Faces start as singleton charts;
// then, for `global_iterations` outer rounds:
//   collapse — merge adjacent chart pairs whose union keeps the merged
//     normal cone under threshold_cone_half_angle_rad, cheapest pair first,
//     an independent set per round (both endpoints must agree, like CuMesh's
//     parallel edge collapse);
//   refine — `refine_iterations` passes that re-hang each face on the
//     neighbouring chart with the best axis-agreement + shared-boundary
//     score;
//   reassign — split charts that refinement left disconnected (union-find).
// Merge cost = merged cone half-angle + area_penalty_weight * area
// (negative weight rewards larger charts) + perimeter_area_ratio_weight *
// perimeter²/area. Defaults mirror the pipeline's CuMesh call:
//   threshold 1.57 rad, refine 100, global 3, smooth 1.0, area -0.1, p/a 0.
// ─────────────────────────────────────────────────────────────────────────
static bool cone_cluster_charts(const std::vector<float> &verts,
                                const std::vector<int32_t> &tris,
                                float threshold_cone_half_angle_rad,
                                int refine_iterations,
                                int global_iterations,
                                float smooth_strength,
                                float area_penalty_weight,
                                float perimeter_area_ratio_weight,
                                std::vector<uint32_t> &face_chart,
                                std::string &err) {
    const int F = (int)(tris.size() / 3);
    const int nv = (int)(verts.size() / 3);
    if (F <= 0 || nv <= 0) {
        err = "empty mesh";
        return false;
    }

    // Per-face normals / areas / perimeters.
    std::vector<float> fn(3 * (size_t)F);
    std::vector<float> fa((size_t)F), fperim((size_t)F);
    for (int f = 0; f < F; ++f) {
        const float *a = &verts[3 * (size_t)tris[3 * f]];
        const float *b = &verts[3 * (size_t)tris[3 * f + 1]];
        const float *c = &verts[3 * (size_t)tris[3 * f + 2]];
        const float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        const float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz,
                    nz = ux * vy - uy * vx;
        const float l = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (l > 1e-20f) {
            fn[3 * f] = nx / l;
            fn[3 * f + 1] = ny / l;
            fn[3 * f + 2] = nz / l;
        } else {
            fn[3 * f] = 0;
            fn[3 * f + 1] = 1;
            fn[3 * f + 2] = 0;
        }
        fa[f] = 0.5f * l;
        const float wx = a[0] - c[0], wy = a[1] - c[1], wz = a[2] - c[2];
        fperim[f] = std::sqrt(ux * ux + uy * uy + uz * uz) +
                    std::sqrt(vx * vx + vy * vy + vz * vz) +
                    std::sqrt(wx * wx + wy * wy + wz * wz);
    }

    // Static edge table: each undirected edge gets an id, the (up to two)
    // incident faces and its length. Manifoldize guarantees ≤2 faces per edge;
    // extra faces on a non-manifold remnant simply share the first slot.
    std::vector<int> eface_a,
            eface_b;  // per edge: first/second incident face (-1 = boundary)
    std::vector<float> elen;  // per edge: length
    std::vector<int> face_edge(3 * (size_t)F);
    {
        std::vector<uint64_t> keys;
        std::vector<int> vals;  // f*3 + corner
        keys.reserve(3 * (size_t)F);
        vals.reserve(3 * (size_t)F);
        for (int f = 0; f < F; ++f)
            for (int k = 0; k < 3; ++k) {
                const int a = tris[3 * f + k], b = tris[3 * f + (k + 1) % 3];
                keys.push_back(
                        a < b ? ((uint64_t)(uint32_t)a << 32) | (uint32_t)b
                              : ((uint64_t)(uint32_t)b << 32) | (uint32_t)a);
                vals.push_back(f * 3 + k);
            }
        std::vector<size_t> order(keys.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t x, size_t y) { return keys[x] < keys[y]; });
        size_t i = 0;
        while (i < order.size()) {
            size_t j = i;
            while (j < order.size() && keys[order[j]] == keys[order[i]]) ++j;
            const int f0 = vals[order[i]] / 3;
            const int f1 = (j - i >= 2) ? vals[order[i + 1]] / 3 : -1;
            const uint32_t va = (uint32_t)(keys[order[i]] >> 32);
            const uint32_t vb = (uint32_t)(keys[order[i]] & 0xffffffffu);
            const float dx = verts[3 * va] - verts[3 * vb],
                        dy = verts[3 * va + 1] - verts[3 * vb + 1],
                        dz = verts[3 * va + 2] - verts[3 * vb + 2];
            const int eid = (int)eface_a.size();
            eface_a.push_back(f0);
            eface_b.push_back(f1);
            elen.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
            for (size_t r = i; r < j; ++r) face_edge[vals[order[r]]] = eid;
            i = j;
        }
    }
    const int M = (int)eface_a.size();
    GLBLOG("cone clustering: F=%d M=%d", F, M);

    // Per-round scratch.
    std::vector<int> cf((size_t)F);  // face → chart
    for (int f = 0; f < F; ++f) cf[f] = f;
    int C = F;
    std::vector<int> ccount, coff, cfaces;  // chart → faces CSR
    std::vector<float> caxis(3 * (size_t)F), chalf((size_t)F);
    std::vector<float> carea((size_t)F), cperim((size_t)F);
    auto compact = [&](int &next) {
        std::vector<int> old2new((size_t)C, -1);
        next = 0;
        for (int f = 0; f < F; ++f) {
            int &m = old2new[cf[f]];
            if (m < 0) m = next++;
            cf[f] = m;
        }
        C = next;
    };
    // Recompute chart cones exactly: axis = normalized sum of face normals
    // (equal weight, like CuMesh's Float3Add), half-angle = max angle from
    // the axis. Also refresh areas.
    auto recompute_cones = [&]() {
        std::fill(carea.begin(), carea.begin() + C, 0.0f);
        std::fill(caxis.begin(), caxis.begin() + 3 * C, 0.0f);
        std::fill(ccount.begin(), ccount.begin() + C, 0);
        for (int f = 0; f < F; ++f) {
            carea[cf[f]] += fa[f];
            caxis[3 * (size_t)cf[f] + 0] += fn[3 * f];
            caxis[3 * (size_t)cf[f] + 1] += fn[3 * f + 1];
            caxis[3 * (size_t)cf[f] + 2] += fn[3 * f + 2];
            ++ccount[cf[f]];
        }
        for (int c = 0; c < C; ++c) {
            float x = caxis[3 * c], y = caxis[3 * c + 1], z = caxis[3 * c + 2];
            const float l = std::sqrt(x * x + y * y + z * z);
            if (l > 1e-20f) {
                x /= l;
                y /= l;
                z /= l;
            }
            caxis[3 * c] = x;
            caxis[3 * c + 1] = y;
            caxis[3 * c + 2] = z;
            chalf[c] = 0.0f;
        }
        for (int f = 0; f < F; ++f) {
            const int c = cf[f];
            const float d = std::min(
                    1.0f,
                    std::max(-1.0f, caxis[3 * c] * fn[3 * f] +
                                            caxis[3 * c + 1] * fn[3 * f + 1] +
                                            caxis[3 * c + 2] * fn[3 * f + 2]));
            const float ang = std::acos(d);
            if (ang > chalf[c]) chalf[c] = ang;
        }
    };
    auto rebuild_csr = [&]() {
        std::fill(ccount.begin(), ccount.begin() + C, 0);
        for (int f = 0; f < F; ++f) ++ccount[cf[f]];
        coff.assign((size_t)C + 1, 0);
        for (int c = 0; c < C; ++c) coff[c + 1] = coff[c] + ccount[c];
        cfaces.resize((size_t)F);
        std::vector<int> cur(coff.begin(), coff.begin() + C);
        for (int f = 0; f < F; ++f) cfaces[cur[cf[f]]++] = f;
    };
    ccount.assign((size_t)F, 0);
    coff.reserve((size_t)F + 1);
    cfaces.reserve((size_t)F);
    caxis.resize(3 * (size_t)F);
    chalf.resize((size_t)F);
    carea.resize((size_t)F);
    cperim.resize((size_t)F);

    // Merge-cone helper: half-angle of the union of two cones (the same
    // approximation compute_chart_adjacency_cost_kernel uses).
    auto merged_half_angle = [&](int c0, int c1) {
        const float ax0 = caxis[3 * c0], ay0 = caxis[3 * c0 + 1],
                    az0 = caxis[3 * c0 + 2];
        const float ax1 = caxis[3 * c1], ay1 = caxis[3 * c1 + 1],
                    az1 = caxis[3 * c1 + 2];
        float d = ax0 * ax1 + ay0 * ay1 + az0 * az1;
        d = std::min(1.0f, std::max(-1.0f, d));
        const float axis_angle = std::acos(d);
        const float lo = std::min(-chalf[c0], axis_angle - chalf[c1]);
        const float hi = std::max(chalf[c0], axis_angle + chalf[c1]);
        return (hi - lo) * 0.5f;
    };

    for (int g = 0; g < global_iterations; ++g) {
        GLBLOG("cone clustering: global round %d (C=%d)", g, C);
        // ── collapse: global cheapest-first chart merges (Garland-style heap).
        // The GPU original expresses the same greedy as rounds of parallel
        // mutual-min collapses, which converge in milliseconds on a device but
        // degenerate to tens of thousands of near-empty rounds on a CPU; a
        // lazy-deletion heap performs the identical cheapest-first merge order
        // in one pass. Cones/areas/perimeters update incrementally with the
        // rotating-axis cone union of CuMesh's collapse_edges_kernel.
        {
            recompute_cones();
            // chart boundary perimeter = sum of incident face perimeters
            std::fill(cperim.begin(), cperim.begin() + C, 0.0f);
            for (int f = 0; f < F; ++f) cperim[cf[f]] += fperim[f];

            // chart-pair adjacency edges (dedupe pairs, sum shared length)
            struct PairEdge {
                int a, b;
                float len;
                bool dead;
            };
            std::vector<PairEdge> pedge;
            std::vector<uint64_t> pkey;
            std::vector<float> plen;
            for (int e = 0; e < M; ++e) {
                const int f1 = eface_b[e];
                if (f1 < 0) continue;
                const int c0 = cf[eface_a[e]], c1 = cf[f1];
                if (c0 == c1) continue;
                pkey.push_back(c0 < c1 ? ((uint64_t)(uint32_t)c0 << 32) |
                                                 (uint32_t)c1
                                       : ((uint64_t)(uint32_t)c1 << 32) |
                                                 (uint32_t)c0);
                plen.push_back(elen[e]);
            }
            {
                std::vector<size_t> ord(pkey.size());
                for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
                std::sort(ord.begin(), ord.end(), [&](size_t x, size_t y) {
                    return pkey[x] < pkey[y];
                });
                pedge.reserve(ord.size());
                for (size_t i = 0; i < ord.size();) {
                    size_t j = i;
                    float s = 0.0f;
                    while (j < ord.size() && pkey[ord[j]] == pkey[ord[i]]) {
                        s += plen[ord[j]];
                        ++j;
                    }
                    const int c0 = (int)(pkey[ord[i]] >> 32);
                    const int c1 = (int)(pkey[ord[i]] & 0xffffffffu);
                    pedge.push_back({c0, c1, s, false});
                    i = j;
                }
            }
            const int E = (int)pedge.size();

            // per-chart incident edges + chart DSU
            std::vector<std::vector<int>> inc((size_t)C);
            for (int e = 0; e < E; ++e) {
                inc[pedge[e].a].push_back(e);
                inc[pedge[e].b].push_back(e);
            }
            std::vector<int> root((size_t)C);
            for (int c = 0; c < C; ++c) root[c] = c;
            std::function<int(int)> cfind = [&](int c) {
                while (root[c] != c) {
                    root[c] = root[root[c]];
                    c = root[c];
                }
                return c;
            };

            // lazy-deletion heap: (cost, edge id, rep pair at push time).
            // A popped entry is valid iff the charts' current representatives
            // still match the recorded pair; otherwise it is recomputed.
            typedef std::tuple<float, int, int, int> HEntry;
            std::priority_queue<HEntry, std::vector<HEntry>,
                                std::greater<HEntry>>
                    heap;
            auto push_edge = [&](int eid) {
                if (pedge[eid].dead) return;
                const int ra = cfind(pedge[eid].a), rb = cfind(pedge[eid].b);
                if (ra == rb) {
                    pedge[eid].dead = true;
                    return;
                }
                const float mha = merged_half_angle(ra, rb);
                const float new_area = carea[ra] + carea[rb];
                float cost = mha + area_penalty_weight * new_area;
                if (perimeter_area_ratio_weight != 0.0f) {
                    const float np =
                            cperim[ra] + cperim[rb] - 2.0f * pedge[eid].len;
                    cost += perimeter_area_ratio_weight * (np * np / new_area);
                }
                heap.emplace(cost, eid, std::min(ra, rb), std::max(ra, rb));
            };
            for (int e = 0; e < E; ++e) push_edge(e);

            while (!heap.empty()) {
                const float cost = std::get<0>(heap.top());
                const int eid = std::get<1>(heap.top());
                const int ra = std::get<2>(heap.top()),
                          rb = std::get<3>(heap.top());
                heap.pop();
                if (pedge[eid].dead) continue;
                const int a = cfind(pedge[eid].a), b = cfind(pedge[eid].b);
                const int lo = std::min(a, b), hi = std::max(a, b);
                if (lo != ra || hi != rb) {
                    push_edge(eid);
                    continue;
                }  // stale
                if (cost > threshold_cone_half_angle_rad)
                    continue;  // heap-ordered: safe to skip
                // rotating-axis cone union (collapse_edges_kernel)
                const float ax0 = caxis[3 * lo], ay0 = caxis[3 * lo + 1],
                            az0 = caxis[3 * lo + 2];
                const float ax1 = caxis[3 * hi], ay1 = caxis[3 * hi + 1],
                            az1 = caxis[3 * hi + 2];
                float d = ax0 * ax1 + ay0 * ay1 + az0 * az1;
                d = std::min(1.0f, std::max(-1.0f, d));
                const float axis_angle = std::acos(d);
                const float clo = std::min(-chalf[lo], axis_angle - chalf[hi]);
                const float chi = std::max(chalf[lo], axis_angle + chalf[hi]);
                chalf[lo] = (chi - clo) * 0.5f;
                if (axis_angle < 1e-3f) {
                    // keep lo axis
                } else {
                    const float naa = (chi + clo) * 0.5f;
                    float px = ax1 - ax0 * d, py = ay1 - ay0 * d,
                          pz = az1 - az0 * d;
                    const float pl = std::sqrt(px * px + py * py + pz * pz);
                    if (pl > 1e-20f) {
                        px /= pl;
                        py /= pl;
                        pz /= pl;
                    }
                    const float ca = std::cos(naa), sa = std::sin(naa);
                    float nx = ax0 * ca + px * sa, ny = ay0 * ca + py * sa,
                          nz = az0 * ca + pz * sa;
                    const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (nl > 1e-20f) {
                        nx /= nl;
                        ny /= nl;
                        nz /= nl;
                    }
                    caxis[3 * lo] = nx;
                    caxis[3 * lo + 1] = ny;
                    caxis[3 * lo + 2] = nz;
                }
                carea[lo] += carea[hi];
                cperim[lo] = cperim[lo] + cperim[hi] - 2.0f * pedge[eid].len;
                root[hi] = lo;
                pedge[eid].dead = true;
                for (int oe : inc[hi]) push_edge(oe);
                inc[lo].insert(inc[lo].end(), inc[hi].begin(), inc[hi].end());
                inc[hi].clear();
            }
            for (int f = 0; f < F; ++f) cf[f] = cfind(cf[f]);
            int next;
            compact(next);
        }

        // ── refine: re-hang faces on the best neighbouring chart ──
        rebuild_csr();
        GLBLOG("cone clustering: refine phase (C=%d)", C);
        for (int it = 0; it < refine_iterations; ++it) {
            recompute_cones();
            std::vector<int> nf((size_t)F);
            for (int f = 0; f < F; ++f) {
                int cand[4];
                float smooth[4];
                int ncand = 0;
                cand[0] = cf[f];
                smooth[0] = 0.0f;
                ncand = 1;
                for (int k = 0; k < 3; ++k) {
                    const int e = face_edge[3 * (size_t)f + k];
                    const int other = eface_a[e] == f ? eface_b[e] : eface_a[e];
                    if (other < 0) continue;
                    const int cc = cf[other];
                    int idx = -1;
                    for (int i = 0; i < ncand; ++i)
                        if (cand[i] == cc) {
                            idx = i;
                            break;
                        }
                    if (idx < 0 && ncand < 4) {
                        idx = ncand++;
                        cand[idx] = cc;
                        smooth[idx] = 0.0f;
                    }
                    if (idx >= 0) smooth[idx] += elen[e];
                }
                int best_c = cf[f];
                float best = -1e9f;
                bool have = false;
                for (int i = 0; i < ncand; ++i) {
                    const float geo = caxis[3 * cand[i]] * fn[3 * f] +
                                      caxis[3 * cand[i] + 1] * fn[3 * f + 1] +
                                      caxis[3 * cand[i] + 2] * fn[3 * f + 2];
                    if (geo <= 0.0f) continue;
                    const float score = geo + smooth_strength * smooth[i];
                    if (!have) {
                        best = score;
                        best_c = cand[i];
                        have = true;
                        continue;
                    }
                    const float diff = score - best;
                    if (diff > 1e-5f) {
                        best = score;
                        best_c = cand[i];
                    } else if (diff >= -1e-5f && cand[i] < best_c) {
                        best = score;
                        best_c = cand[i];
                    }
                }
                nf[f] = best_c;
            }
            bool changed = false;
            for (int f = 0; f < F; ++f)
                if (nf[f] != cf[f]) {
                    changed = true;
                    break;
                }
            cf.swap(nf);
            int next;
            compact(next);
            if (!changed) break;
        }

        // ── reassign: split charts that refinement left disconnected ──
        {
            std::vector<int> parent((size_t)F);
            for (int f = 0; f < F; ++f) parent[f] = f;
            std::function<int(int)> find = [&](int x) {
                while (parent[x] != x) {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                return x;
            };
            for (int e = 0; e < M; ++e) {
                const int f1 = eface_b[e];
                if (f1 < 0) continue;
                const int a = eface_a[e];
                if (cf[a] != cf[f1]) continue;
                int ra = find(a), rb = find(f1);
                if (ra != rb) parent[ra < rb ? rb : ra] = ra < rb ? ra : rb;
            }
            // find() yields face indices ([0,F)), so compact over the face-id
            // domain — the chart-count-sized map in compact() would overflow.
            for (int f = 0; f < F; ++f) cf[f] = find(f);
            {
                std::vector<int> old2new((size_t)F, -1);
                int next = 0;
                for (int f = 0; f < F; ++f) {
                    int &m = old2new[cf[f]];
                    if (m < 0) m = next++;
                    cf[f] = m;
                }
                C = next;
            }
        }
    }

    face_chart.resize((size_t)F);
    for (int f = 0; f < F; ++f) face_chart[f] = (uint32_t)cf[f];
    return true;
}

// CPU chart clustering + the shared xatlas parameterization/packing tail:
// the libtorch-free equivalent of cumesh_unwrap (same algorithm, same
// defaults, no CUDA and no PyTorch).
static bool cone_cluster_unwrap(const std::vector<float> &dverts,
                                const std::vector<float> &dnrm,
                                const std::vector<float> &dpbr,
                                int dnv,
                                const std::vector<int32_t> &dtris,
                                int dnt,
                                const MeshExportOptions &opt,
                                int TS,
                                std::vector<float> &opos,
                                std::vector<float> &onrm,
                                std::vector<float> &ouv,
                                std::vector<float> &opbr,
                                std::vector<uint32_t> &oidx,
                                int &AW,
                                int &AH,
                                std::string &err) {
    if (dnt <= 0) {
        err = "empty mesh";
        return false;
    }
    const double t0 = now_s();
    std::vector<uint32_t> face_mat;
    if (!cone_cluster_charts(dverts, dtris,
                             /*threshold_cone_half_angle_rad=*/1.57f,
                             /*refine_iterations=*/20,
                             /*global_iterations=*/3, /*smooth_strength=*/1.0f,
                             /*area_penalty_weight=*/-0.1f,
                             /*perimeter_area_ratio_weight=*/0.0f, face_mat,
                             err))
        return false;
    const unsigned n_charts =
            face_mat.empty()
                    ? 0u
                    : 1u + *std::max_element(face_mat.begin(), face_mat.end());
    GLBLOG("cone clustering: %d tris into %u charts (%.2fs)", dnt, n_charts,
           now_s() - t0);
    const bool ok = xatlas_unwrap_with_face_material(
            dverts, dnrm, dpbr, dnv, dtris, dnt, face_mat, opt, TS, opos, onrm,
            ouv, opbr, oidx, AW, AH, err);
    if (ok)
        GLBLOG("cone unwrap: %u verts, %u tris, %u charts packed @%dx%d "
               "(%.2fs)",
               (unsigned)(opos.size() / 3), (unsigned)(oidx.size() / 3),
               n_charts, AW, AH, now_s() - t0);
    return ok;
}

// UV unwrap and raster bake.  In the ordinary xatlas mode, PBR is interpolated
// from the atlas mesh's own vertices.  For print wrapping, projection_source is
// the dense pre-wrap mesh: every covered atlas texel is mapped to a 3D point on
// the wrapped target, projected to the closest source triangle, and sampled
// barycentrically.  That mirrors upstream's remesh -> UV raster -> BVH ->
// attribute-field path without requiring CUDA.
bool bake_atlas_locked(const PreparedMesh &mesh,
                       const PreparedMesh *projection_source,
                       const MeshExportOptions &opt,
                       std::vector<uint8_t> &out,
                       std::string &err) {
    const int TS = opt.texture_size;
    const int dnv = (int)(mesh.verts.size() / 3);
    const int dnt = (int)(mesh.tris.size() / 3);
    const bool vertex_pbr = mesh.pbr.size() == (size_t)dnv * 6;
    const bool projected_pbr = projection_source != nullptr;
    if (projected_pbr && projection_source->pbr.size() !=
                                 projection_source->verts.size() / 3 * 6) {
        err = "PBR projection source has no six-channel material";
        return false;
    }

    std::vector<float> opos, onrm, ouv, opbr;
    std::vector<uint32_t> oidx;
    int AW = TS, AH = TS;
    // UV unwrap priority (first success wins; all paths bake identically):
    //   1. T2GLB_XATLAS env — explicit bare-xatlas chart unwrap (debug /
    //      regression; its own clustering is minutes-slow on large meshes)
    //   2. CuMesh GPU chart clustering — only in libtorch-linked builds
    //      (TRELLIS2_CUMESH=ON); T2GLB_NOCUMESH skips it
    //   3. cone_cluster_unwrap — CPU normal-cone clustering (CuMesh's
    //      algorithm, dependency-free) + xatlas parameterization: the
    //      chart-quality default without libtorch, seconds not minutes
    //   4. simple_unwrap — chartless 6-bin projection (last resort)
    // Unwrap backend selection via the explicit UnwrapMode option (was the
    // T2GLB_XATLAS / T2GLB_NOCUMESH environment pair upstream): Auto follows
    // the full quality chain (cumesh GPU when compiled in -> CPU cone
    // clustering -> 6-bin projection); XAtlas skips straight to xatlas
    // charting; Simple skips GPU/xatlas and starts at cone clustering.
    const bool force_xatlas = opt.unwrap == UnwrapMode::XAtlas;
    bool unwrapped = false;
    const char *unwrap_mode = "none";
    BakeStage bs_unwrap("uv unwrap chain", opt);
    if (force_xatlas) {
        unwrapped = xatlas_unwrap(mesh.verts, mesh.normals, mesh.pbr, dnv,
                                  mesh.tris, dnt, opt, TS, opos, onrm, ouv,
                                  opbr, oidx, AW, AH, err);
        if (unwrapped) unwrap_mode = "bare xatlas charting";
        if (!unwrapped)
            GLBLOG("xatlas unwrap failed (%s); falling back", err.c_str());
    }
#ifdef TRELLIS2_HAVE_CUMESH
    // The cumesh_unwrap definition above lives behind this same guard; the
    // call site must be preprocessed out as well (a runtime bool is not
    // enough) or CPU-only builds fail with an undeclared identifier.
    const bool use_cumesh = opt.unwrap == UnwrapMode::Auto;
    if (!unwrapped && use_cumesh) {
        unwrapped = cumesh_unwrap(mesh.verts, mesh.normals, mesh.pbr, dnv,
                                  mesh.tris, dnt, opt, TS, opos, onrm, ouv,
                                  opbr, oidx, AW, AH, err);
        if (unwrapped) unwrap_mode = "cumesh GPU clustering";
        if (!unwrapped)
            GLBLOG("cumesh unwrap failed (%s); falling back to CPU cone "
                   "clustering",
                   err.c_str());
    }
#endif
    if (!unwrapped) {
        unwrapped = cone_cluster_unwrap(mesh.verts, mesh.normals, mesh.pbr, dnv,
                                        mesh.tris, dnt, opt, TS, opos, onrm,
                                        ouv, opbr, oidx, AW, AH, err);
        if (unwrapped)
            unwrap_mode =
                    "cone-cluster charts + xatlas "
                    "parameterize/pack";
        if (!unwrapped)
            GLBLOG("cone clustering unwrap failed (%s); falling back to "
                   "simple_unwrap",
                   err.c_str());
    }
    if (!unwrapped) {
        BakeStage bs_simple("uv unwrap: 6-bin simple projection", opt);
        if (!simple_unwrap(mesh.verts, mesh.normals, mesh.pbr, dnv, mesh.tris,
                           dnt, TS, opt.padding, opos, onrm, ouv, opbr, oidx,
                           AW, AH, err))
            return false;
        unwrap_mode = "simple 6-bin projection";
    }
    AICORE_LOG_INFO("trellis", "[glb] bake: uv unwrap succeeded via %s",
                    unwrap_mode);
    bs_unwrap.done();
    const uint32_t onv = (uint32_t)(opos.size() / 3);
    const uint32_t ntri = (uint32_t)(oidx.size() / 3);

    GLBLOG("rasterizing %u tris%s ...", ntri,
           projected_pbr ? " for source-surface PBR projection" : "");
    BakeStage bs_raster("rasterize triangles into atlas", opt);
    const int NP = AW * AH;
    std::vector<float> bc((size_t)NP * 3, 0.0f), met(NP, 0.0f), rou(NP, 0.0f),
            alp(NP, 0.0f);
    std::vector<uint8_t> mask(NP, 0);
    std::vector<float> query_points;
    std::vector<int32_t> query_pixels;
    if (projected_pbr) {
        query_points.reserve((size_t)NP *
                             2);  // atlases are normally 60-80% occupied
        query_pixels.reserve((size_t)NP * 2 / 3);
    }
    auto set_default = [&](int pix) {
        bc[3 * pix + 0] = bc[3 * pix + 1] = bc[3 * pix + 2] = 0.7f;
        met[pix] = 0.0f;
        rou[pix] = 0.6f;
        alp[pix] = 1.0f;
    };

    for (uint32_t t = 0; t < ntri; ++t) {
        if ((t & 0xFFFF) == 0) bs_raster.checkpoint();  // ~65k-tri cadence
        const uint32_t ia = oidx[3 * t + 0], ib = oidx[3 * t + 1],
                       ic = oidx[3 * t + 2];
        const float ax = ouv[2 * ia + 0], ay = ouv[2 * ia + 1];
        const float bx = ouv[2 * ib + 0], by = ouv[2 * ib + 1];
        const float cx = ouv[2 * ic + 0], cy = ouv[2 * ic + 1];
        const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::fabs(area) < 1e-9f) continue;
        const float inv_area = 1.0f / area;
        const int x0 = std::max(0, (int)std::floor(std::min({ax, bx, cx}) - 1));
        const int x1 =
                std::min(AW - 1, (int)std::ceil(std::max({ax, bx, cx}) + 1));
        const int y0 = std::max(0, (int)std::floor(std::min({ay, by, cy}) - 1));
        const int y1 =
                std::min(AH - 1, (int)std::ceil(std::max({ay, by, cy}) + 1));
        for (int py = y0; py <= y1; ++py) {
            const float sy = py + 0.5f;
            for (int px = x0; px <= x1; ++px) {
                const float sx = px + 0.5f;
                const float w0 =
                        ((bx - sx) * (cy - sy) - (by - sy) * (cx - sx)) *
                        inv_area;
                const float w1 =
                        ((cx - sx) * (ay - sy) - (cy - sy) * (ax - sx)) *
                        inv_area;
                const float w2 = 1.0f - w0 - w1;
                const float e = -0.001f;
                if (w0 < e || w1 < e || w2 < e) continue;
                const int pix = py * AW + px;
                if (projected_pbr) {
                    // Shared triangle edges can cover the same pixel twice.
                    // The positions agree, so retain the first and issue one
                    // expensive closest-surface query per atlas texel.
                    if (mask[pix]) continue;
                    for (int k = 0; k < 3; ++k)
                        query_points.push_back(w0 * opos[3 * ia + k] +
                                               w1 * opos[3 * ib + k] +
                                               w2 * opos[3 * ic + k]);
                    query_pixels.push_back(pix);
                    mask[pix] = 1;
                    continue;
                }
                if (!vertex_pbr) {
                    set_default(pix);
                    mask[pix] = 1;
                    continue;
                }
                const float *a = opbr.data() + (size_t)ia * 6;
                const float *b = opbr.data() + (size_t)ib * 6;
                const float *c = opbr.data() + (size_t)ic * 6;
                float val[6];
                for (int ch = 0; ch < 6; ++ch)
                    val[ch] = w0 * a[ch] + w1 * b[ch] + w2 * c[ch];
                bc[3 * pix + 0] = val[0];
                bc[3 * pix + 1] = val[1];
                bc[3 * pix + 2] = val[2];
                met[pix] = val[3];
                rou[pix] = val[4];
                alp[pix] = val[5];
                mask[pix] = 1;
            }
        }
    }

    if (projected_pbr) {
        GLBLOG("projecting %zu covered texels to %zu source tris ...",
               query_pixels.size(), projection_source->tris.size() / 3);
        BakeStage bs_project("project PBR (closest-surface per covered texel)",
                             opt);
        std::vector<float> query_pbr;
        if (!t2print::project_pbr(
                    projection_source->verts, projection_source->tris,
                    projection_source->pbr, query_points, query_pbr, err))
            return false;
        if (query_pbr.size() != query_pixels.size() * 6) {
            err = "PBR projection returned an invalid sample count";
            return false;
        }
        for (size_t i = 0; i < query_pixels.size(); ++i) {
            const int pix = query_pixels[i];
            const float *val = query_pbr.data() + i * 6;
            bc[3 * pix + 0] = val[0];
            bc[3 * pix + 1] = val[1];
            bc[3 * pix + 2] = val[2];
            met[pix] = val[3];
            rou[pix] = val[4];
            alp[pix] = val[5];
        }
        query_points.clear();
        query_points.shrink_to_fit();
        query_pbr.clear();
        query_pbr.shrink_to_fit();
    }

    // Edge-pad the gutter so bilinear sampling cannot bleed empty texels
    // across chart seams.  RGB/metallic/roughness are averaged from the
    // neighbours, but alpha is carried as the neighbour MAX so the padding
    // stays fully opaque instead of forming a semi-transparent halo (which
    // renders as "holes" in BLEND mode and does not exist in upstream's
    // nvdiffrast bake).
    {
        std::vector<uint8_t> m = mask;
        for (int pass = 0; pass < opt.dilate; ++pass) {
            std::vector<uint8_t> nm = m;
            for (int py = 0; py < AH; ++py)
                for (int px = 0; px < AW; ++px) {
                    const int pix = py * AW + px;
                    if (m[pix]) continue;
                    float acc[5] = {0, 0, 0, 0, 0};
                    float amax = 0.0f;
                    int cnt = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int qx = px + dx, qy = py + dy;
                            if (qx < 0 || qx >= AW || qy < 0 || qy >= AH)
                                continue;
                            const int q = qy * AW + qx;
                            if (!m[q]) continue;
                            acc[0] += bc[3 * q + 0];
                            acc[1] += bc[3 * q + 1];
                            acc[2] += bc[3 * q + 2];
                            acc[3] += met[q];
                            acc[4] += rou[q];
                            ++cnt;
                            if (alp[q] > amax) amax = alp[q];
                        }
                    if (cnt) {
                        bc[3 * pix + 0] = acc[0] / cnt;
                        bc[3 * pix + 1] = acc[1] / cnt;
                        bc[3 * pix + 2] = acc[2] / cnt;
                        met[pix] = acc[3] / cnt;
                        rou[pix] = acc[4] / cnt;
                        alp[pix] = amax;
                        nm[pix] = 1;
                    }
                }
            m.swap(nm);
        }
    }

    // glTF packs metallic in B and roughness in G. Base color is stored as
    // generated (sRGB-like), matching upstream's base_color*255 texture path.
    //
    // Alpha cleanup (T2GLB_ALPHA_CLEAN, default on): the decoder's activation
    // noise scatters a band of near-transparent texels over the surface — a
    // thick 0.03-0.25 band (2-4% of the atlas) plus a shallower 0.25-0.5 tail
    // on GPU lineages (f16 activation rounding, deeper under q8); the exact
    // f32 CPU bake bottoms out at ~0.55. Everything between the upstream floor
    // (0.03) and 0.5 sits on solid surface and is decoder noise, not material:
    // snap it to fully opaque. Snapping to 0 would punch BLEND holes through to
    // the unlit interior, which renders as dark erosion blotches in unsorted
    // viewers (ParaView). Only texels already below the upstream floor (<0.03)
    // keep their near-transparent value.
    const bool alpha_clean = opt.alpha_clean;
    auto to8 = [](float f) {
        int v = (int)std::lround(f * 255.0f);
        return (uint8_t)std::min(255, std::max(0, v));
    };
    std::vector<uint8_t> bc8((size_t)NP * 4), mr8((size_t)NP * 3);
    int covered = 0, translucent = 0;
    for (int i = 0; i < NP; ++i) {
        float a = alp[i];
        if (alpha_clean && a > 0.03f && a < 0.5f) a = 1.0f;
        bc8[4 * i + 0] = to8(bc[3 * i + 0]);
        bc8[4 * i + 1] = to8(bc[3 * i + 1]);
        bc8[4 * i + 2] = to8(bc[3 * i + 2]);
        bc8[4 * i + 3] = to8(a);
        mr8[3 * i + 0] = 0;
        mr8[3 * i + 1] = to8(rou[i]);
        mr8[3 * i + 2] = to8(met[i]);
        if (mask[i]) {
            ++covered;
            // Judge blending on the *encoded* alpha: opaque noise texels need
            // no BLEND ordering, only visible translucency does.
            if (a >= 0.25f && a < 0.95f) ++translucent;
        }
    }
    const bool transparent =
            translucent > 0 && (int64_t)translucent * 1000 >= (int64_t)covered;
    BakeStage bs_encode("dilate + inpaint + PNG encode", opt);
    GLBLOG("inpaint + PNG encode ...");
    std::vector<uint8_t> bc_png, mr_png;
    if (!encode_png(AW, AH, 4, bc8.data(), bc_png) ||
        !encode_png(AW, AH, 3, mr8.data(), mr_png)) {
        err = "PNG encode failed";
        return false;
    }

    std::vector<float> gpos((size_t)onv * 3), gnrm((size_t)onv * 3),
            guv((size_t)onv * 2);
    for (uint32_t i = 0; i < onv; ++i) {
        gpos[3 * i + 0] = opos[3 * i + 0];
        gpos[3 * i + 1] = opos[3 * i + 2];
        gpos[3 * i + 2] = -opos[3 * i + 1];
        gnrm[3 * i + 0] = onrm[3 * i + 0];
        gnrm[3 * i + 1] = onrm[3 * i + 2];
        gnrm[3 * i + 2] = -onrm[3 * i + 1];
        guv[2 * i + 0] = ouv[2 * i + 0] / (float)AW;
        guv[2 * i + 1] = ouv[2 * i + 1] / (float)AH;
    }
    write_glb(gpos, gnrm, guv, oidx, bc_png, mr_png, transparent, out);
    GLBLOG("GLB %zu bytes (%u verts, %u tris; %s PBR atlas)", out.size(), onv,
           ntri, projected_pbr ? "projected" : "vertex");
    return true;
}

bool prepare_mesh_locked(const float *verts,
                         int nv,
                         const int32_t *tris,
                         int nt,
                         const float *pbr,
                         const MeshExportOptions &opt,
                         PreparedMesh &out,
                         std::string &err,
                         bool allow_atlas_smoothing = true) {
    if (!verts || !tris || nv <= 0 || nt <= 0) {
        err = "empty mesh";
        return false;
    }
    out = PreparedMesh{};

    GLBLOG("input %d verts %d tris; preserving original topology", nv, nt);
    if (!copy_mesh(verts, nv, tris, nt, pbr, out.verts, out.tris, out.pbr, err))
        return false;

    filter_components(out.verts, out.tris, out.pbr, 0.0005f, opt.components);
    const int dnv = (int)(out.verts.size() / 3),
              dnt = (int)(out.tris.size() / 3);
    if (dnv < 3 || dnt < 1) {
        err = "component cleanup produced an empty mesh";
        return false;
    }
    GLBLOG("after component filter -> %d verts %d tris", dnv, dnt);

    // xatlas optionally regularises the geometry; do it here so the preview is
    // the geometry that will actually be written to the GLB.
    if (allow_atlas_smoothing && opt.taubin_smooth)
        taubin_smooth(out.verts, out.tris, 3);
    vertex_normals(out.verts, out.tris, out.normals);
    return true;
}

std::mutex g_bake_mu;  // serialize bakes (bounds peak RAM; keeps xatlas tidy)

// Reduce triangle count for UV atlas export. Topology-only decimation keeps the
// full vertex pool so CGAL can project PBR from the dense source onto
// simplified UV geometry (mirrors upstream simplify-then-BVH-reproject).
static float decimation_error(float opt_error) {
    // opt_error defaults to 2e-1 in MeshExportOptions: loose enough for a
    // 3.7M-tri mesh to reach a 281K target with plain (non-sloppy)
    // simplification; tighter limits stall above target and fall into the
    // sloppy path, whose slivers make xatlas charting pathologically slow
    // (150K tris took >10 min in ComputeCharts). Was the
    // T2GLB_DECIMATION_ERROR environment variable upstream.
    return opt_error > 0.0f ? opt_error : 2e-1f;
}
static bool decimate_topology(PreparedMesh &mesh,
                              size_t target_tris,
                              float error_limit,
                              std::string &err) {
    const size_t nt = mesh.tris.size() / 3;
    if (nt <= target_tris || nt == 0) return true;
    const size_t nv = mesh.verts.size() / 3;
    if (nv < 3) {
        err = "empty mesh";
        return false;
    }

    std::vector<unsigned int> idx((size_t)nt * 3);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (unsigned int)mesh.tris[i];

    float result_error = 0.0f;
    std::vector<unsigned int> dst(idx.size());
    size_t n = meshopt_simplify(dst.data(), idx.data(), idx.size(),
                                mesh.verts.data(), nv, sizeof(float) * 3,
                                target_tris * 3, decimation_error(error_limit),
                                0, &result_error);
    if (n == 0) {
        err = "mesh decimation failed";
        return false;
    }
    GLBLOG("decimated %zu -> %zu tris (error %.4g)", nt, n / 3, result_error);
    mesh.tris.resize(n);
    for (size_t i = 0; i < n; ++i) mesh.tris[i] = (int32_t)dst[i];
    return true;
}

// Split non-manifold edges so every undirected edge is shared by at most two
// triangles. Sloppy decimation and flexible-dual-grid extraction both leave
// many edges with 3+ triangles; xatlas charting is pathologically slow on such
// meshes (boundary tracing explodes), while a quick vertex duplication makes
// them ordinary manifold input. Geometry is unchanged; duplicate vertices get
// copies of the original position/normal/PBR.
static void manifoldize(std::vector<float> &verts,
                        std::vector<float> &normals,
                        std::vector<float> &pbr,
                        std::vector<int32_t> &tris) {
    struct Ref {
        uint32_t tri, corner;
    };
    std::unordered_map<uint64_t, std::vector<Ref>> emap;
    const uint32_t ntri = (uint32_t)(tris.size() / 3);
    for (uint32_t t = 0; t < ntri; ++t) {
        for (uint32_t k = 0; k < 3; ++k) {
            uint32_t x = (uint32_t)tris[3 * t + k],
                     y = (uint32_t)tris[3 * t + (k + 1) % 3];
            if (x > y) std::swap(x, y);
            emap[((uint64_t)x << 32) | y].push_back({t, k});
        }
    }
    size_t split = 0;
    for (auto &kv : emap) {
        auto &refs = kv.second;
        if (refs.size() <= 2) continue;
        for (size_t i = 2; i < refs.size(); ++i) {
            const uint32_t t = refs[i].tri, k = refs[i].corner;
            const uint32_t a = (uint32_t)tris[3 * t + k],
                           b = (uint32_t)tris[3 * t + (k + 1) % 3];
            const uint32_t na = (uint32_t)(verts.size() / 3), nb = na + 1;
            verts.insert(verts.end(),
                         {verts[3 * a], verts[3 * a + 1], verts[3 * a + 2],
                          verts[3 * b], verts[3 * b + 1], verts[3 * b + 2]});
            if (!normals.empty())
                normals.insert(normals.end(),
                               {normals[3 * a], normals[3 * a + 1],
                                normals[3 * a + 2], normals[3 * b],
                                normals[3 * b + 1], normals[3 * b + 2]});
            if (!pbr.empty()) {
                pbr.insert(pbr.end(), pbr.begin() + 6 * a,
                           pbr.begin() + 6 * a + 6);
                pbr.insert(pbr.end(), pbr.begin() + 6 * b,
                           pbr.begin() + 6 * b + 6);
            }
            tris[3 * t + k] = (int32_t)na;
            tris[3 * t + (k + 1) % 3] = (int32_t)nb;
            ++split;
        }
    }
    if (split)
        GLBLOG("manifoldized %zu non-manifold edge refs -> %zu verts", split,
               verts.size() / 3);
}

// Sloppy decimation reaches the triangle target where plain simplification
// stalls on non-manifold input (flexible dual-grid output). It emits slivers;
// they are dropped here, and edges shared by 3+ triangles are split so the
// result is ordinary manifold input for charting. Per-vertex PBR attributes
// follow the same vertex remap when present.
static bool decimate_sloppy(PreparedMesh &mesh,
                            size_t target_tris,
                            float error_limit,
                            float sliver_aspect,
                            std::string &err) {
    const size_t nt = mesh.tris.size() / 3;
    if (nt <= target_tris || nt == 0) return true;
    const size_t nv = mesh.verts.size() / 3;
    if (nv < 3) {
        err = "empty mesh";
        return false;
    }
    const bool has_pbr = mesh.pbr.size() == (size_t)nv * 6;

    std::vector<unsigned int> idx((size_t)nt * 3);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (unsigned int)mesh.tris[i];

    float result_error = 0.0f;
    const size_t n = meshopt_simplifySloppy(
            idx.data(), idx.data(), idx.size(), mesh.verts.data(), nv,
            sizeof(float) * 3, target_tris * 3, decimation_error(error_limit),
            &result_error);
    if (n == 0) {
        err = "mesh sloppy decimation failed";
        return false;
    }
    GLBLOG("sloppy decimated %zu -> %zu tris (error %.4g)", nt, n / 3,
           result_error);

    std::vector<unsigned int> remap(nv);
    const size_t new_nv = meshopt_generateVertexRemap(remap.data(), idx.data(),
                                                      n, mesh.verts.data(), nv,
                                                      sizeof(float) * 3);

    std::vector<float> new_verts(new_nv * 3);
    meshopt_remapVertexBuffer(new_verts.data(), mesh.verts.data(), nv,
                              sizeof(float) * 3, remap.data());
    std::vector<float> new_pbr;
    if (has_pbr) {
        new_pbr.resize(new_nv * 6);
        meshopt_remapVertexBuffer(new_pbr.data(), mesh.pbr.data(), nv,
                                  sizeof(float) * 6, remap.data());
    }
    meshopt_remapIndexBuffer(idx.data(), idx.data(), n, remap.data());

    mesh.verts = std::move(new_verts);
    mesh.pbr = std::move(new_pbr);
    mesh.tris.resize(n);
    for (size_t i = 0; i < n; ++i) mesh.tris[i] = (int32_t)idx[i];

    // Sloppy simplification emits near-degenerate slivers; they make charting
    // pathologically slow, so drop them before unwrapping. The filter removes
    // triangles whose squared area is tiny or whose longest edge is huge
    // relative to that area (aspect-ratio slivers). T2GLB_SLIVER_ASPECT tunes
    // the aspect cutoff (default 100): xatlas boundary tracing explodes on
    // long thin triangles, so the chart path can tighten it to 10-20.
    {
        const float aspect_lim = sliver_aspect;
        const float lim2 =
                aspect_lim > 1.0f ? aspect_lim * aspect_lim * 4.0f : 400.0f;
        const size_t ntri = mesh.tris.size() / 3;
        std::vector<int32_t> keep;
        keep.reserve(mesh.tris.size());
        size_t dropped = 0;
        for (size_t i = 0; i < ntri; ++i) {
            const int32_t t0 = mesh.tris[3 * i + 0], t1 = mesh.tris[3 * i + 1],
                          t2 = mesh.tris[3 * i + 2];
            if (t0 == t1 || t1 == t2 || t0 == t2) {
                ++dropped;
                continue;
            }
            const float *a = &mesh.verts[(size_t)t0 * 3];
            const float *b = &mesh.verts[(size_t)t1 * 3];
            const float *c = &mesh.verts[(size_t)t2 * 3];
            const float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
            const float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
            const float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz,
                        nz = ux * vy - uy * vx;
            const float area2 = nx * nx + ny * ny + nz * nz;  // (2*area)^2
            if (area2 < 1e-16f) {
                ++dropped;
                continue;
            }  // fully degenerate
            const float maxe2 = std::max(
                    {ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz,
                     (b[0] - c[0]) * (b[0] - c[0]) +
                             (b[1] - c[1]) * (b[1] - c[1]) +
                             (b[2] - c[2]) * (b[2] - c[2])});
            // aspect = max_edge / min_altitude = maxe2 / (2*sqrt(area2));
            // drop triangles with aspect > limit:  maxe2^2 > (2*limit)^2 *
            // area2
            if (maxe2 * maxe2 > lim2 * area2) {
                ++dropped;
                continue;
            }
            keep.push_back(t0);
            keep.push_back(t1);
            keep.push_back(t2);
        }
        if (dropped) {
            GLBLOG("dropped %zu degenerate/sliver tris (aspect>%g)", dropped,
                   aspect_lim);
            mesh.tris.swap(keep);
        }
    }

    // Sloppy decimation leaves many edges shared by 3+ triangles; xatlas needs
    // a manifold mesh, so split those edges into duplicated vertices.
    manifoldize(mesh.verts, mesh.normals, mesh.pbr, mesh.tris);

    mesh.normals.clear();
    vertex_normals(mesh.verts, mesh.tris, mesh.normals);
    GLBLOG("sloppy decimated %zu/%zu -> %zu/%zu tris (error %.4g)", nv, nt,
           new_nv, n / 3, result_error);

    return true;
}

// Portable fallback when CGAL projection is unavailable: collapse geometry and
// carry per-vertex PBR through meshopt_simplifyWithUpdate.
static bool decimate_with_pbr(PreparedMesh &mesh,
                              size_t target_tris,
                              float error_limit,
                              float sliver_aspect,
                              std::string &err) {
    const size_t nt = mesh.tris.size() / 3;
    if (nt <= target_tris || nt == 0) return true;
    const size_t nv = mesh.verts.size() / 3;
    if (mesh.pbr.size() != nv * 6) {
        err = "decimate_with_pbr requires six-channel vertex PBR";
        return false;
    }

    std::vector<unsigned int> idx((size_t)nt * 3);
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (unsigned int)mesh.tris[i];

    const float attr_w[6] = {1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.25f};
    float result_error = 0.0f;
    size_t n = meshopt_simplifyWithUpdate(
            idx.data(), idx.size(), mesh.verts.data(), nv, sizeof(float) * 3,
            mesh.pbr.data(), sizeof(float) * 6, attr_w, 6, nullptr,
            target_tris * 3, decimation_error(error_limit), 0, &result_error);
    if (n == 0) {
        err = "mesh decimation with PBR failed";
        return false;
    }
    if (n / 3 > target_tris) {
        // Non-manifold edges and complex vertices cap plain simplification far
        // above target; sloppy decimation ignores topology and reaches it.
        return decimate_sloppy(mesh, target_tris, error_limit, sliver_aspect,
                               err);
    }
    GLBLOG("decimated+pbr %zu/%zu -> %zu tris (error %.4g)", nt, nv, n / 3,
           result_error);

    std::vector<unsigned int> remap(nv);
    const size_t new_nv = meshopt_generateVertexRemap(remap.data(), idx.data(),
                                                      n, mesh.verts.data(), nv,
                                                      sizeof(float) * 3);

    std::vector<float> new_verts(new_nv * 3);
    std::vector<float> new_pbr(new_nv * 6);
    meshopt_remapVertexBuffer(new_verts.data(), mesh.verts.data(), nv,
                              sizeof(float) * 3, remap.data());
    meshopt_remapVertexBuffer(new_pbr.data(), mesh.pbr.data(), nv,
                              sizeof(float) * 6, remap.data());
    meshopt_remapIndexBuffer(idx.data(), idx.data(), n, remap.data());

    mesh.verts = std::move(new_verts);
    mesh.pbr = std::move(new_pbr);
    mesh.tris.resize(n);
    for (size_t i = 0; i < n; ++i) mesh.tris[i] = (int32_t)idx[i];
    return true;
}

static bool export_atlas_glb(const PreparedMesh &source,
                             const MeshExportOptions &opt,
                             std::vector<uint8_t> &out,
                             std::string &err) {
    if (source.tris.size() / 3 == 0) {
        err = "empty mesh";
        return false;
    }
    const size_t target_tris = (size_t)std::max(1, opt.decimation_target);
    GLBLOG("atlas export: %zu source tris -> target %zu (T2GLB_DECIMATION)",
           source.tris.size() / 3, target_tris);
    BakeStage bs_dec1("decimate topology pass 1 (meshopt)", opt);

    PreparedMesh target = source;
    if (!decimate_topology(target, target_tris, opt.decimation_error, err))
        return false;

    if (t2print::available() && !source.pbr.empty()) {
        // The UV mesh must reach the target. Flexible dual-grid output is
        // heavily non-manifold, which caps plain simplification far above the
        // target; splitting the non-manifold edges first makes the mesh
        // ordinary input for edge-collapse simplification (mirrors upstream's
        // simplify-then-unwrap on a manifold mesh). Sloppy decimation remains
        // the fallback for meshes that still stall.
        if (target.tris.size() / 3 > target_tris) {
            BakeStage bs_manifold("manifoldize + decimate topology pass 2",
                                  opt);
            manifoldize(target.verts, target.normals, target.pbr, target.tris);
            target.normals.clear();
            vertex_normals(target.verts, target.tris, target.normals);
            if (!decimate_topology(target, target_tris, opt.decimation_error,
                                   err))
                return false;
        }
        if (target.tris.size() / 3 > target_tris) {
            BakeStage bs_sloppy("decimate sloppy fallback (sliver-prone)", opt);
            if (!decimate_sloppy(target, target_tris, opt.decimation_error,
                                 opt.sliver_aspect, err))
                return false;
        }
        GLBLOG("atlas bake: simplified UV mesh + CGAL PBR projection from "
               "source");
        return bake_atlas_locked(target, &source, opt, out, err);
    }

    PreparedMesh baked = source;
    if (baked.tris.size() / 3 > target_tris) {
        if (!decimate_with_pbr(baked, target_tris, opt.decimation_error,
                               opt.sliver_aspect, err))
            return false;
    }
    GLBLOG("atlas bake: simplified mesh + vertex PBR interpolation");
    return bake_atlas_locked(baked, nullptr, opt, out, err);
}

}  // namespace

bool prepare_mesh(const float *verts,
                  int nv,
                  const int32_t *tris,
                  int nt,
                  const float *pbr,
                  const MeshExportOptions &opt,
                  PreparedMesh &out,
                  std::string &err) {
    std::lock_guard<std::mutex> lock(g_bake_mu);
    return prepare_mesh_locked(verts, nv, tris, nt, pbr, opt, out, err);
}

bool print_remesh_available() { return t2print::available(); }

bool prepare_print_mesh(const float *verts,
                        int nv,
                        const int32_t *tris,
                        int nt,
                        const float *pbr,
                        const MeshExportOptions &opt,
                        float alpha_ratio,
                        float offset_ratio,
                        PreparedMesh &out,
                        std::string &err) {
    // Carry the source material through component filtering so it can be
    // sampled onto the wrap for a textured preview (source.pbr stays aligned
    // with source.verts/source.tris; empty when the caller passed no material).
    PreparedMesh source;
    if (!prepare_mesh(verts, nv, tris, nt, pbr, opt, source, err)) return false;

    PreparedMesh wrapped;
    if (!t2print::alpha_wrap(source.verts, source.tris, alpha_ratio,
                             offset_ratio, wrapped.verts, wrapped.normals,
                             wrapped.tris, err))
        return false;
    // Alpha Wrap constructs entirely new offset vertices, so source per-vertex
    // attributes cannot be carried directly. When the source is textured,
    // sample its material at each wrap vertex by closest-surface projection —
    // the same transfer the GLB download bakes per texel, here per vertex for a
    // cheap, approximate preview. Projection is best-effort: on failure the
    // wrap still previews untextured rather than aborting the print preview
    // entirely.
    if (!source.pbr.empty() &&
        source.pbr.size() == source.verts.size() / 3 * 6) {
        std::string perr;
        if (!t2print::project_pbr(source.verts, source.tris, source.pbr,
                                  wrapped.verts, wrapped.pbr, perr)) {
            wrapped.pbr.clear();
        }
    } else {
        wrapped.pbr.clear();
    }
    out = std::move(wrapped);
    return true;
}

bool mesh_to_glb(const float *verts,
                 int nv,
                 const int32_t *tris,
                 int nt,
                 const float *pbr,
                 const MeshExportOptions &opt,
                 std::vector<uint8_t> &out,
                 std::string &err) {
    if (nv <= 0 || nt <= 0) {
        err = "empty mesh";
        return false;
    }
    const int TS = opt.texture_size;
    if (TS < 16 || TS > 8192) {
        err = "bad texture_size";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_bake_mu);
    BakeStage bs_prepare("prepare (component filter + cleanup)", opt);

    // All knobs come from the explicit MeshExportOptions (the T2GLB_*
    // environment overrides upstream have no in-tree equivalent).
    MeshExportOptions bake_opt = opt;
    bake_opt.decimation_target = std::max(1, opt.decimation_target);

    PreparedMesh prepared;
    if (!prepare_mesh_locked(verts, nv, tris, nt, pbr, bake_opt, prepared, err))
        return false;
    const int dnv = (int)(prepared.verts.size() / 3),
              dnt = (int)(prepared.tris.size() / 3);

    // Legacy vertex-colour export (debug / regression). Default is UV atlas.
    if (bake_opt.vertex_pbr) {
        write_vertex_glb(prepared, out);
        GLBLOG("GLB %zu bytes (%d verts, %d tris; vertex PBR)", out.size(), dnv,
               dnt);
        return true;
    }

    // XAtlas mode without decimation: opt-in for pre-simplified meshes.
    if (bake_opt.unwrap == UnwrapMode::XAtlas &&
        bake_opt.decimation_target <= 0) {
        return bake_atlas_locked(prepared, nullptr, bake_opt, out, err);
    }

    if (!export_atlas_glb(prepared, bake_opt, out, err)) return false;
    GLBLOG("GLB %zu bytes (%d source verts, %d source tris; UV atlas PBR)",
           out.size(), dnv, dnt);
    return true;
}

bool mesh_to_projected_glb(const float *target_verts,
                           int target_nv,
                           const int32_t *target_tris,
                           int target_nt,
                           const float *source_verts,
                           int source_nv,
                           const int32_t *source_tris,
                           int source_nt,
                           const float *source_pbr,
                           const MeshExportOptions &opt,
                           std::vector<uint8_t> &out,
                           std::string &err) {
    if (!t2print::available()) {
        err = "PBR projection is unavailable (rebuild with CGAL >= 5.5)";
        return false;
    }
    if (!target_verts || !target_tris || target_nv <= 0 || target_nt <= 0 ||
        !source_verts || !source_tris || !source_pbr || source_nv <= 0 ||
        source_nt <= 0) {
        err = "empty projected GLB mesh";
        return false;
    }
    if (opt.texture_size < 16 || opt.texture_size > 8192) {
        err = "bad texture_size";
        return false;
    }

    std::lock_guard<std::mutex> lock(g_bake_mu);
    PreparedMesh target, source;
    MeshExportOptions target_opt = opt;
    target_opt.components = ComponentFilter::KeepAll;
    // The target is the exact already-previewed Alpha Wrap.  Do not let the
    // legacy T2GLB_XATLAS smoothing switch move it during export.
    if (!prepare_mesh_locked(target_verts, target_nv, target_tris, target_nt,
                             nullptr, target_opt, target, err, false))
        return false;
    if (!prepare_mesh_locked(source_verts, source_nv, source_tris, source_nt,
                             source_pbr, opt, source, err, false))
        return false;
    return bake_atlas_locked(target, &source, opt, out, err);
}

}  // namespace t2glb
