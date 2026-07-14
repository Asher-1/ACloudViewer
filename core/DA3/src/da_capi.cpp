#include "da_capi.h"
#include "engine.hpp"
#include "preprocess.hpp"
#include "image_io.hpp"
#include "glb_export.hpp"
#include "colmap_export.hpp"
#include "reconstruct.hpp"
#include "path_util.hpp"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <utility>
#include <vector>

struct da_ctx { std::unique_ptr<da::Engine> engine; std::string last_error; };

static char* dup_cstr(const std::string& s){
    char* p = (char*)std::malloc(s.size()+1);
    if (p) std::memcpy(p, s.c_str(), s.size()+1);
    return p;
}
// Minimal JSON string escaping for interpolated values (quotes, backslash, controls).
static std::string json_escape(const std::string& s){
    std::string o; o.reserve(s.size()+2);
    for (char ch : s){
        switch (ch){
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)ch < 0x20){ char b[8]; std::snprintf(b,sizeof(b),"\\u%04x",ch); o += b; }
                else o += ch;
        }
    }
    return o;
}
// Best-effort metric detection from the checkpoint name: metric/nested/mono
// variants produce metric-scale depth; relative DualDPT (base/giant/large/small)
// do not. Unknown -> 0.
static bool capi_is_metric(const da::Config& cfg){
    // DA2 metric models carry a positive head_max_depth (20/80); authoritative.
    if (cfg.head_max_depth > 0.f) return true;
    std::string n = cfg.checkpoint_name;
    for (char& ch : n) ch = (char)std::tolower((unsigned char)ch);
    return n.find("metric") != std::string::npos ||
           n.find("nested") != std::string::npos ||
           n.find("mono")   != std::string::npos;
}

// Run the nested metric pipeline (anyview GIANT + metric ViT-L branches ->
// alignment) for a single image. Fills depth + scaled ext/intr + processed dims.
// Returns false with c->last_error set on failure.
static bool capi_run_nested(da_ctx* c, const char* image_path,
                            std::vector<float>& depth,
                            std::array<float,12>& ext, std::array<float,9>& intr,
                            int& H, int& W){
    da::NestedOut out;
    if (!c->engine->depth_metric_path(image_path, out, H, W)){
        c->last_error = "nested: depth_metric failed"; return false; }
    depth = std::move(out.depth);
    ext = out.extrinsics;
    intr = out.intrinsics;
    return true;
}

extern "C" {
int da_capi_abi_version(void){ return 4; }
da_ctx* da_capi_load(const char* path, int n_threads){
    if (!path) return nullptr;
    auto e = da::Engine::load(path, n_threads);
    if (!e) return nullptr;
    auto* c = new da_ctx(); c->engine = std::move(e); return c;
}
da_ctx* da_capi_load_nested(const char* anyview, const char* metric, int n_threads){
    if (!anyview || !metric) return nullptr;
    auto e = da::Engine::load_nested(anyview, metric, n_threads);
    if (!e) return nullptr;
    auto* c = new da_ctx(); c->engine = std::move(e); return c;
}
void da_capi_free(da_ctx* c){ delete c; }
char* da_capi_info_json(da_ctx* c){
    if (!c || !c->engine) return nullptr;
    const auto& cfg = c->engine->config();
    std::string j = "{\"checkpoint\":\"" + json_escape(cfg.checkpoint_name) + "\",\"embed_dim\":" +
        std::to_string(cfg.embed_dim) + ",\"depth\":" + std::to_string(cfg.depth) +
        ",\"num_heads\":" + std::to_string(cfg.num_heads) + "}";
    return dup_cstr(j);
}
void da_capi_free_string(char* s){ std::free(s); }
const char* da_capi_last_error(da_ctx* c){ return c ? c->last_error.c_str() : ""; }
float* da_capi_depth_path(da_ctx* c, const char* image_path, int* out_h, int* out_w){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "depth: bad args"; return nullptr; }
    std::vector<float> depth, conf; int H = 0, W = 0;
    if (c->engine->is_nested()){
        std::array<float,12> ext; std::array<float,9> intr;
        if (!capi_run_nested(c, image_path, depth, ext, intr, H, W)) return nullptr;
    } else if (c->engine->is_da2()){
        if (!c->engine->depth_relative_path(image_path, depth, H, W)){ c->last_error = "depth: da2 failed"; return nullptr; }
    } else if (!c->engine->depth_native(image_path, depth, conf, H, W)){ c->last_error = "depth: failed"; return nullptr; }
    float* p = (float*)std::malloc(depth.size() * sizeof(float));
    if (!p){ c->last_error = "depth: oom"; return nullptr; }
    std::memcpy(p, depth.data(), depth.size() * sizeof(float));
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    return p;
}
void da_capi_free_floats(float* p){ std::free(p); }
int da_capi_pose_path(da_ctx* c, const char* image_path, float out_ext[12], float out_intr[9]){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "pose: bad args"; return -1; }
    if (c->engine->is_da2()){ c->last_error = "pose: da2 model has no camera pose"; return -1; }
    std::vector<float> depth, conf; std::array<float,12> ext; std::array<float,9> intr; int H = 0, W = 0;
    if (c->engine->is_nested()){
        if (!capi_run_nested(c, image_path, depth, ext, intr, H, W)) return -1;
    } else if (!c->engine->depth_pose_native_path(image_path, depth, conf, ext, intr, H, W)){ c->last_error = "pose: failed"; return -1; }
    if (out_ext)  std::memcpy(out_ext,  ext.data(),  12 * sizeof(float));
    if (out_intr) std::memcpy(out_intr, intr.data(),  9 * sizeof(float));
    return 0;
}
float* da_capi_depth_pose_multi(da_ctx* c, const char** image_paths, int n_images,
                                int* out_h, int* out_w, int* out_n,
                                float* out_ext, float* out_intr){
    if (!c || !c->engine || !image_paths || n_images <= 0){ if (c) c->last_error = "depth_multi: bad args"; return nullptr; }
    std::vector<da::Image> imgs(n_images);
    for (int i = 0; i < n_images; ++i){
        if (!image_paths[i] || !da::load_image_rgb(image_paths[i], imgs[i])){
            c->last_error = "depth_multi: load image failed"; return nullptr;
        }
    }
    int H = 0, W = 0;
    const int n = n_images;
    if (c->engine->is_nested()){
        std::vector<da::NestedOut> nested;
        if (!c->engine->depth_metric_multi(imgs, nested, H, W)){
            c->last_error = "depth_multi: nested failed"; return nullptr;
        }
        if ((int)nested.size() != n){ c->last_error = "depth_multi: view count mismatch"; return nullptr; }
        const size_t per_view = (size_t)H * W;
        float* p = (float*)std::malloc((size_t)n * per_view * sizeof(float));
        if (!p){ c->last_error = "depth_multi: oom"; return nullptr; }
        for (int i = 0; i < n; ++i){
            std::memcpy(p + (size_t)i * per_view, nested[i].depth.data(), per_view * sizeof(float));
            if (out_ext)  std::memcpy(out_ext  + (size_t)i * 12, nested[i].extrinsics.data(), 12 * sizeof(float));
            if (out_intr) std::memcpy(out_intr + (size_t)i * 9,  nested[i].intrinsics.data(),  9 * sizeof(float));
        }
        if (out_h) *out_h = H;
        if (out_w) *out_w = W;
        if (out_n) *out_n = n;
        return p;
    }

    std::vector<da::ViewResult> views;
    if (!c->engine->depth_pose_multi(imgs, views, H, W)){ c->last_error = "depth_multi: failed"; return nullptr; }
    const int n_views = (int)views.size();
    const size_t per_pixels = (size_t)H * W;
    float* p = (float*)std::malloc((size_t)n_views * per_pixels * sizeof(float));
    if (!p){ c->last_error = "depth_multi: oom"; return nullptr; }
    for (int i = 0; i < n_views; ++i){
        std::memcpy(p + (size_t)i * per_pixels, views[i].depth.data(), per_pixels * sizeof(float));
        if (out_ext)  std::memcpy(out_ext  + (size_t)i * 12, views[i].ext.data(),  12 * sizeof(float));
        if (out_intr) std::memcpy(out_intr + (size_t)i * 9,  views[i].intr.data(),  9 * sizeof(float));
    }
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    if (out_n) *out_n = n_views;
    return p;
}

// Shared single-image export prep: run native depth+pose, capture processed RGB,
// build N=1 exporter inputs. Returns false (with c->last_error set) on failure.
static bool capi_export_prep(da_ctx* c, const char* image_path,
                             std::vector<float>& depth, std::vector<float>& conf,
                             std::vector<std::array<float,9>>& K,
                             std::vector<std::array<float,16>>& E,
                             std::vector<uint8_t>& rgb_u8,
                             da::Image& img, int& H, int& W){
    if (!da::load_image_rgb(image_path, img)){ c->last_error = "export: load image failed"; return false; }
    std::array<float,12> ext; std::array<float,9> intr;
    if (!c->engine->depth_pose_native(img, depth, conf, ext, intr, H, W)){
        c->last_error = "export: depth+pose failed"; return false; }
    da::Preprocessed pp;
    if (!da::preprocess_real(img, c->engine->config(), pp, &rgb_u8) || pp.H != H || pp.W != W){
        c->last_error = "export: capture processed colors failed"; return false; }
    std::array<float,16> ext4{};
    for (int i = 0; i < 12; ++i) ext4[i] = ext[i];
    ext4[12] = 0.f; ext4[13] = 0.f; ext4[14] = 0.f; ext4[15] = 1.f;
    K = { intr }; E = { ext4 };
    return true;
}
int da_capi_export_glb(da_ctx* c, const char* image_path, const char* out_glb){
    if (!c || !c->engine || !image_path || !out_glb){ if (c) c->last_error = "export_glb: bad args"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    if (!da::write_glb(out_glb, depth, conf, K, E, imgs_u8, H, W, 1, da::GlbOptions{})){
        c->last_error = "export_glb: write failed"; return -1; }
    return 0;
}
int da_capi_export_colmap(da_ctx* c, const char* image_path, const char* out_dir, int binary){
    if (!c || !c->engine || !image_path || !out_dir){ if (c) c->last_error = "export_colmap: bad args"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    std::string path(image_path);
    size_t s = path.find_last_of("/\\");
    std::vector<std::string> names { s == std::string::npos ? path : path.substr(s + 1) };
    std::vector<std::pair<int,int>> orig_wh { { img.w, img.h } };
    if (!da::write_colmap(out_dir, depth, conf, K, E, imgs_u8, names, orig_wh, H, W, 1, binary != 0)){
        c->last_error = "export_colmap: write failed"; return -1; }
    return 0;
}
int da_capi_export_colmap_multi(da_ctx* c, const char** image_paths, int n_images,
                                const char* out_dir, int binary){
    if (!c || !c->engine || !image_paths || n_images <= 0 || !out_dir){
        if (c) c->last_error = "export_colmap_multi: bad args";
        return -1;
    }
    std::vector<da::Image> imgs(static_cast<size_t>(n_images));
    std::vector<std::string> names(static_cast<size_t>(n_images));
    for (int i = 0; i < n_images; ++i){
        if (!image_paths[i] || !da::load_image_rgb(image_paths[i], imgs[static_cast<size_t>(i)])){
            c->last_error = std::string("export_colmap_multi: load image failed: ") +
                            (image_paths[i] ? image_paths[i] : "");
            return -1;
        }
        std::string path(image_paths[i]);
        const size_t s = path.find_last_of("/\\");
        names[static_cast<size_t>(i)] = s == std::string::npos ? path : path.substr(s + 1);
    }
    if (!c->engine->export_colmap_multi(imgs, names, out_dir, binary != 0)){
        if (c->last_error.empty()) c->last_error = "export_colmap_multi: failed";
        return -1;
    }
    return 0;
}
int da_capi_depth_dense(da_ctx* c, const char* image_path, int* out_h, int* out_w,
                        float** out_depth, float** out_conf, float** out_sky,
                        float out_ext[12], float out_intr[9], int* out_is_metric){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "depth_dense: bad args"; return -1; }
    // Default outputs to a clean state.
    if (out_depth) *out_depth = nullptr;
    if (out_conf)  *out_conf  = nullptr;
    if (out_sky)   *out_sky   = nullptr;
    if (out_ext)  std::memset(out_ext,  0, 12 * sizeof(float));
    if (out_intr) std::memset(out_intr, 0,  9 * sizeof(float));
    int H = 0, W = 0;
    // Nested metric model: run both branches + alignment -> metric-scale depth +
    // scaled pose. No conf/sky surface (sky is already folded into depth).
    if (c->engine->is_nested()){
        std::vector<float> ndepth; std::array<float,12> next; std::array<float,9> nintr;
        if (!capi_run_nested(c, image_path, ndepth, next, nintr, H, W)) return -1;
        const size_t hw = (size_t)H * W;
        if (hw == 0 || ndepth.size() != hw){ c->last_error = "depth_dense: nested empty/size mismatch"; return -1; }
        if (out_depth){
            float* dptr = (float*)std::malloc(hw * sizeof(float));
            if (!dptr){ c->last_error = "depth_dense: oom"; return -1; }
            std::memcpy(dptr, ndepth.data(), hw * sizeof(float));
            *out_depth = dptr;
        }
        if (out_ext)  std::memcpy(out_ext,  next.data(),  12 * sizeof(float));
        if (out_intr) std::memcpy(out_intr, nintr.data(),  9 * sizeof(float));
        if (out_h) *out_h = H;
        if (out_w) *out_w = W;
        if (out_is_metric) *out_is_metric = 1;
        return 0;
    }
    // Depth Anything V2: depth only. No conf/sky surface, no camera pose. ext/intr
    // stay zeroed (memset above); metric iff head_max_depth > 0 (sigmoid x max_depth).
    if (c->engine->is_da2()){
        std::vector<float> d2;
        if (!c->engine->depth_relative_path(image_path, d2, H, W)){
            c->last_error = "depth_dense: da2 failed"; return -1; }
        const size_t hw = (size_t)H * W;
        if (hw == 0 || d2.size() != hw){ c->last_error = "depth_dense: da2 empty/size mismatch"; return -1; }
        if (out_depth){
            float* dptr = (float*)std::malloc(hw * sizeof(float));
            if (!dptr){ c->last_error = "depth_dense: oom"; return -1; }
            std::memcpy(dptr, d2.data(), hw * sizeof(float));
            *out_depth = dptr;
        }
        if (out_h) *out_h = H;
        if (out_w) *out_w = W;
        if (out_is_metric) *out_is_metric = (c->engine->config().head_max_depth > 0.f) ? 1 : 0;
        return 0;
    }
    da::Image img;
    if (!da::load_image_rgb(image_path, img)){ c->last_error = "depth_dense: load image failed"; return -1; }
    const bool mono = c->engine->is_mono();
    std::vector<float> depth, second; // second = conf (DualDPT) or sky (mono)
    if (mono){
        if (!c->engine->depth_mono(img, depth, second, H, W)){
            c->last_error = "depth_dense: depth_mono failed"; return -1; }
    } else {
        std::array<float,12> ext; std::array<float,9> intr;
        if (!c->engine->depth_pose_native(img, depth, second, ext, intr, H, W)){
            c->last_error = "depth_dense: depth+pose failed"; return -1; }
        if (out_ext)  std::memcpy(out_ext,  ext.data(),  12 * sizeof(float));
        if (out_intr) std::memcpy(out_intr, intr.data(),  9 * sizeof(float));
    }
    const size_t hw = (size_t)H * W;
    if (hw == 0 || depth.size() != hw){ c->last_error = "depth_dense: empty/size mismatch"; return -1; }
    float* dptr = (float*)std::malloc(hw * sizeof(float));
    float* sptr = (float*)std::calloc(hw, sizeof(float));
    if (!dptr || !sptr){ std::free(dptr); std::free(sptr); c->last_error = "depth_dense: oom"; return -1; }
    std::memcpy(dptr, depth.data(), hw * sizeof(float));
    std::memcpy(sptr, second.data(), std::min(second.size(), hw) * sizeof(float));
    if (out_depth) *out_depth = dptr; else std::free(dptr);
    if (mono){
        if (out_sky)  *out_sky  = sptr; else std::free(sptr);
    } else {
        if (out_conf) *out_conf = sptr; else std::free(sptr);
    }
    if (out_h) *out_h = H;
    if (out_w) *out_w = W;
    if (out_is_metric) *out_is_metric = capi_is_metric(c->engine->config()) ? 1 : 0;
    return 0;
}
int da_capi_points(da_ctx* c, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb){
    if (!c || !c->engine || !image_path){ if (c) c->last_error = "points: bad args"; return -1; }
    if (out_xyz) *out_xyz = nullptr;
    if (out_rgb) *out_rgb = nullptr;
    if (c->engine->is_mono() || c->engine->is_da2()){ c->last_error = "points: this model has no camera pose; use a DualDPT model"; return -1; }
    std::vector<float> depth, conf; std::vector<std::array<float,9>> K; std::vector<std::array<float,16>> E;
    std::vector<uint8_t> rgb_u8; da::Image img; int H=0, W=0;
    if (!capi_export_prep(c, image_path, depth, conf, K, E, rgb_u8, img, H, W)) return -1;
    // back_project expects world-to-camera extrinsics; capi_export_prep yields the
    // same 4x4 ext used by glb/colmap export (mirrors examples/cli cmd_depth_export).
    std::vector<const uint8_t*> imgs_u8 { rgb_u8.data() };
    da::WorldPoints wp = da::back_project(depth, conf, K, E, imgs_u8, H, W, 1, conf_thresh);
    const size_t n = wp.xyz.size() / 3;
    float* xyz = (float*)std::malloc(wp.xyz.size() * sizeof(float));
    unsigned char* rgb = (unsigned char*)std::malloc(wp.rgb.size() * sizeof(unsigned char));
    if ((wp.xyz.size() && !xyz) || (wp.rgb.size() && !rgb)){
        std::free(xyz); std::free(rgb); c->last_error = "points: oom"; return -1; }
    if (wp.xyz.size()) std::memcpy(xyz, wp.xyz.data(), wp.xyz.size() * sizeof(float));
    if (wp.rgb.size()) std::memcpy(rgb, wp.rgb.data(), wp.rgb.size() * sizeof(unsigned char));
    if (out_xyz) *out_xyz = xyz; else std::free(xyz);
    if (out_rgb) *out_rgb = rgb; else std::free(rgb);
    if (out_n) *out_n = (int)n;
    return 0;
}
void da_capi_free_bytes(unsigned char* p){ std::free(p); }
char* da_capi_model_cache_dir(void){ return dup_cstr(da::default_model_cache_dir()); }
}
