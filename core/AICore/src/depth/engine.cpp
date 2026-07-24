// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "engine.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cam_pose.hpp"
#include "colmap_export.hpp"
#include "common.hpp"
#include "compute_mode.hpp"
#include "dino_backbone.hpp"
#include "dpt_head.hpp"
#include "gs_adapter.hpp"
#include "gs_head.hpp"
#include "image_io.hpp"
#include "nested.hpp"
#include "preprocess.hpp"
#include "ray_pose.hpp"
#include "vram_budget.hpp"

namespace aicore {
namespace depth {
std::unique_ptr<Engine> Engine::load(const std::string& path, int n_threads) {
    const char* device = std::getenv("DA_DEVICE");
    return load_device(path, n_threads, device ? device : "auto");
}
std::unique_ptr<Engine> Engine::load_device(const std::string& path,
                                            int n_threads,
                                            const std::string& device) {
    std::unique_ptr<Engine> e(new Engine(device));
    if (e->be_.has_error()) {
        DA_ERR("engine: backend init failed: %s", e->be_.error().c_str());
        return nullptr;
    }
    if (!e->ml_.load(path)) {
        DA_ERR("engine: load failed");
        return nullptr;
    }
    e->be_.set_n_threads(n_threads > 0 ? n_threads : 1);
    if (!e->ml_.offload_weights(e->be_)) {
        DA_ERR("engine: offload failed");
        return nullptr;
    }
    // Route graph builders to GPU-friendly standard ops iff weights are
    // device-resident.
    aicore::depth::set_gpu_mode(e->be_.is_offloading());
    return e;
}
std::unique_ptr<Engine> Engine::load_nested(const std::string& anyview_gguf,
                                            const std::string& metric_gguf,
                                            int n_threads) {
    const char* device = std::getenv("DA_DEVICE");
    return load_nested_device(anyview_gguf, metric_gguf, n_threads,
                              device ? device : "auto");
}
std::unique_ptr<Engine> Engine::load_nested_device(
        const std::string& anyview_gguf,
        const std::string& metric_gguf,
        int n_threads,
        const std::string& device) {
    auto e = load_device(anyview_gguf, n_threads, device);
    if (!e) {
        DA_ERR("engine: anyview load failed");
        return nullptr;
    }
    e->metric_ml_.reset(new ModelLoader());
    if (!e->metric_ml_->load(metric_gguf)) {
        DA_ERR("engine: metric load failed");
        return nullptr;
    }
    // Metric weights stay on host until ensure_metric_gpu() so anyview
    // multiview (the VRAM peak) does not compete with the metric branch on the
    // same GPU.
    return e;
}

bool Engine::ensure_metric_gpu() {
    if (!metric_ml_) {
        return false;
    }
    if (metric_ml_->weights_on_device()) {
        return true;
    }
    if (!metric_ml_->offload_weights(be_)) {
        DA_ERR("ensure_metric_gpu: metric offload failed");
        return false;
    }
    return true;
}

bool Engine::ensure_anyview_gpu() {
    if (!be_.is_offloading()) {
        return true;
    }
    if (ml_.weights_on_device()) {
        return true;
    }
    if (!ml_.offload_weights(be_)) {
        DA_ERR("ensure_anyview_gpu: anyview offload failed");
        return false;
    }
    return true;
}

void Engine::release_gpu_working_memory() {
    be_.release_graph_memory();
    if (be_.is_offloading()) {
        ml_.release_gpu_weights();
        if (metric_ml_) {
            metric_ml_->release_gpu_weights();
        }
    }
}

int Engine::cap_img_resize_target(int requested) const {
    return cap_resize_target_for_vram(requested, metric_ml_ != nullptr,
                                      query_gpu_memory(be_), be_.device_name());
}

namespace {

bool force_joint_multiview() {
    const char* v = std::getenv("DA3_FORCE_JOINT_MV");
    return v && v[0] != '\0' && v[0] != '0' && std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

}  // namespace

void Engine::set_img_resize_target(uint32_t target) {
    ml_.set_img_resize_target(target);
    if (metric_ml_) {
        metric_ml_->set_img_resize_target(target);
    }
}

bool Engine::backbone_features(const std::vector<float>& input_chw,
                               int H,
                               int W,
                               std::vector<std::vector<float>>& feats_out) {
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> cams;
    return bb.forward(input_chw, H, W, feats_out, cams);
}
bool Engine::depth(const std::string& image_path,
                   std::vector<float>& depth_out,
                   std::vector<float>& conf_out,
                   int& H,
                   int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth: load image failed");
        return false;
    }
    return depth_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_image(const Image& img,
                         std::vector<float>& depth_out,
                         std::vector<float>& conf_out,
                         int& H,
                         int& W) {
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) {
        DA_ERR("depth: preprocess failed");
        return false;
    }
    H = p.H;
    W = p.W;
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) {
        DA_ERR("depth: backbone failed");
        return false;
    }
    DptHead head(ml_, be_);
    return head.depth(feats, H, W, depth_out, conf_out);
}
bool Engine::is_mono() const {
    // out2b out-channels == 1 (depth only) AND a parallel sky head present.
    ggml_tensor* out2b = ml_.tensor("head.scratch.out2b.weight");
    ggml_tensor* sky = ml_.tensor("head.scratch.sky_out2b.weight");
    return out2b && sky && (int)out2b->ne[3] == 1;
}
bool Engine::depth_mono(const Image& img,
                        std::vector<float>& depth_out,
                        std::vector<float>& sky_out,
                        int& H,
                        int& W) {
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_mono: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_mono: backbone failed");
        return false;
    }
    DptHead head(ml_, be_);
    return head.depth_sky(feats, H, W, depth_out, sky_out);
}
bool Engine::depth_mono_path(const std::string& image_path,
                             std::vector<float>& depth_out,
                             std::vector<float>& sky_out,
                             int& H,
                             int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_mono: load image failed");
        return false;
    }
    return depth_mono(img, depth_out, sky_out, H, W);
}
bool Engine::depth_relative(const Image& img,
                            std::vector<float>& depth_out,
                            int& H,
                            int& W) {
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_relative: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_relative: backbone failed");
        return false;
    }
    DptHead head(ml_, be_);
    return head.depth_relative(feats, H, W, ml_.config().head_max_depth,
                               depth_out);
}
bool Engine::depth_relative_path(const std::string& image_path,
                                 std::vector<float>& depth_out,
                                 int& H,
                                 int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_relative: load image failed");
        return false;
    }
    return depth_relative(img, depth_out, H, W);
}
bool Engine::depth_native(const std::string& image_path,
                          std::vector<float>& depth_out,
                          std::vector<float>& conf_out,
                          int& H,
                          int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_native: load image failed");
        return false;
    }
    return depth_native_image(img, depth_out, conf_out, H, W);
}
bool Engine::depth_native_image(const Image& img,
                                std::vector<float>& depth_out,
                                std::vector<float>& conf_out,
                                int& H,
                                int& W) {
    // Fused backbone+head graph by default (feats stay device-resident).
    // DA_FUSED=0 forces the original two-graph path; cat_token=false models
    // always use unfused.
    const char* fenv = std::getenv("DA_FUSED");
    const bool fused_off = fenv && std::string(fenv) == "0";
    if (ml_.config().cat_token && !fused_off)
        return depth_native_fused(img, depth_out, conf_out, H, W);
    return depth_native_unfused(img, depth_out, conf_out, H, W);
}
bool Engine::depth_native_unfused(const Image& img,
                                  std::vector<float>& depth_out,
                                  std::vector<float>& conf_out,
                                  int& H,
                                  int& W) {
    const bool prof = std::getenv("DA_PROFILE") != nullptr;
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    Preprocessed p;
    auto t0 = now();
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_native: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    auto t1 = now();
    std::vector<std::vector<float>> feats;
    if (!backbone_features(p.chw, H, W, feats)) {
        DA_ERR("depth_native: backbone failed");
        return false;
    }
    auto t2 = now();
    DptHead head(ml_, be_);
    bool ok = head.depth(feats, H, W, depth_out, conf_out);
    auto t3 = now();
    if (prof)
        DA_DEBUG_LOG(
                "profile: [unfused] preprocess=%.1fms backbone=%.1fms "
                "head=%.1fms",
                ms(t0, t1), ms(t1, t2), ms(t2, t3));
    return ok;
}
bool Engine::depth_native_fused(const Image& img,
                                std::vector<float>& depth_out,
                                std::vector<float>& conf_out,
                                int& H,
                                int& W) {
    const bool prof = std::getenv("DA_PROFILE") != nullptr;
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    Preprocessed p;
    auto t0 = now();
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_native: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    auto t1 = now();
    DinoBackbone bb(ml_, be_);
    DptHead head(ml_, be_);
    // output_dim from out2b out-channels (depth+conf for base/giant -> 2).
    ggml_tensor* out2b_w = ml_.tensor("head.scratch.out2b.weight");
    const int output_dim = out2b_w ? (int)out2b_w->ne[3] : 2;
    GraphInputPool pool;
    std::vector<float> logits;
    // ONE graph: backbone produces the 4 feat tensors in-graph, the head
    // consumes them directly -> feats never leave the device.
    bool ok = be_.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* feat[4] = {nullptr, nullptr, nullptr, nullptr};
                if (!bb.build_feats_graph(ctx, p.chw, H, W, pool, feat))
                    return nullptr;
                return head.build_depth_graph(ctx, feat, H, W, pool, nullptr,
                                              nullptr, nullptr);
            },
            logits);
    auto t2 = now();
    if (!ok) {
        DA_ERR("depth_native_fused: compute failed");
        return false;
    }
    const size_t HW = (size_t)H * W;
    if (logits.size() != (size_t)output_dim * HW) {
        DA_ERR("depth_native_fused: bad logits size");
        return false;
    }
    // Same host post-process as DptHead::run -> bit-identical depth/conf given
    // equal logits.
    depth_out.resize(HW);
    for (size_t i = 0; i < HW; ++i) depth_out[i] = std::exp(logits[i]);
    if (output_dim >= 2) {
        conf_out.resize(HW);
        for (size_t i = 0; i < HW; ++i)
            conf_out[i] = std::exp(logits[HW + i]) + 1.0f;
    } else {
        conf_out.clear();
    }
    if (prof)
        DA_DEBUG_LOG(
                "profile: [fused] preprocess=%.1fms "
                "graph(backbone+head)=%.1fms",
                ms(t0, t1), ms(t1, t2));
    return ok;
}
bool Engine::depth_pose_native(const Image& img,
                               std::vector<float>& depth,
                               std::vector<float>& conf,
                               std::array<float, 12>& ext,
                               std::array<float, 9>& intr,
                               int& H,
                               int& W) {
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_pose_native: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_pose_native: backbone failed");
        return false;
    }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) {
        DA_ERR("depth_pose_native: depth head failed");
        return false;
    }
    if (cam_tokens.size() < 4) {
        DA_ERR("depth_pose_native: missing layer-11 cam token");
        return false;
    }
    CamPose cam(ml_, be_);
    std::array<float, 9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) {
        DA_ERR("depth_pose_native: cam pose failed");
        return false;
    }
    return true;
}
bool Engine::depth_pose_native_path(const std::string& image_path,
                                    std::vector<float>& depth,
                                    std::vector<float>& conf,
                                    std::array<float, 12>& ext,
                                    std::array<float, 9>& intr,
                                    int& H,
                                    int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_pose_native: load image failed");
        return false;
    }
    return depth_pose_native(img, depth, conf, ext, intr, H, W);
}
bool Engine::has_aux() const {
    return ml_.tensor("head.scratch.rn1_aux.out.weight") != nullptr;
}
bool Engine::depth_pose_rays_native(const Image& img,
                                    std::vector<float>& depth,
                                    std::vector<float>& conf,
                                    std::array<float, 12>& ext,
                                    std::array<float, 9>& intr,
                                    int& H,
                                    int& W) {
    if (!has_aux()) {
        DA_WARN("depth_pose_rays: this GGUF has no auxiliary ray head (rebuild "
                "with --with-aux for --ray-pose)");
        return false;
    }
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_pose_rays: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_pose_rays: backbone failed");
        return false;
    }
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) {
        DA_ERR("depth_pose_rays: depth head failed");
        return false;
    }
    // Ray field -> pose (use_ray_pose). The aux head runs on the SAME backbone
    // feats.
    std::vector<float> ray, ray_conf;
    int ray_h = 0, ray_w = 0;
    if (!head.rays(feats, H, W, ray, ray_conf, ray_h, ray_w)) {
        DA_ERR("depth_pose_rays: aux ray head failed");
        return false;
    }
    RayPoseParams pp;
    RayPoseOut o;
    // Production path: no fed indices -> internal SEEDED deterministic RANSAC
    // sampling.
    if (!solve_ray_pose(ray.data(), ray_conf.data(), ray_h, ray_w, H, W,
                        /*rand_idx=*/nullptr, /*sorted_idx=*/nullptr,
                        /*refit_idx=*/nullptr, /*n_refit=*/0, pp, o)) {
        DA_ERR("depth_pose_rays: ray->pose solver failed");
        return false;
    }
    ext = o.extrinsics;
    intr = o.intrinsics;
    return true;
}
bool Engine::depth_pose_rays_native_path(const std::string& image_path,
                                         std::vector<float>& depth,
                                         std::vector<float>& conf,
                                         std::array<float, 12>& ext,
                                         std::array<float, 9>& intr,
                                         int& H,
                                         int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_pose_rays: load image failed");
        return false;
    }
    return depth_pose_rays_native(img, depth, conf, ext, intr, H, W);
}
bool Engine::depth_pose(const Image& img,
                        std::vector<float>& depth,
                        std::vector<float>& conf,
                        std::array<float, 12>& ext,
                        std::array<float, 9>& intr,
                        int& H,
                        int& W) {
    Preprocessed p;
    if (!preprocess(img, ml_.config(), p)) {
        DA_ERR("depth_pose: preprocess failed");
        return false;
    }
    H = p.H;
    W = p.W;
    if (!ensure_anyview_gpu()) {
        return false;
    }
    // Run the backbone ONCE; reuse feats for depth and cam_tokens[3] for pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_pose: backbone failed");
        return false;
    }
    be_.release_graph_memory();
    DptHead head(ml_, be_);
    if (!head.depth(feats, H, W, depth, conf)) {
        DA_ERR("depth_pose: depth head failed");
        return false;
    }
    be_.release_graph_memory();
    if (cam_tokens.size() < 4) {
        DA_ERR("depth_pose: missing layer-11 cam token");
        return false;
    }
    CamPose cam(ml_, be_);
    std::array<float, 9> pe;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) {
        DA_ERR("depth_pose: cam pose failed");
        return false;
    }
    return true;
}
bool Engine::depth_pose_multi_joint(const std::vector<Image>& imgs,
                                    std::vector<ViewResult>& out,
                                    int& H,
                                    int& W) {
    out.clear();
    if (imgs.empty()) {
        DA_ERR("depth_pose_multi: no images");
        return false;
    }
    // Preprocess every image; all must yield identical H,W.
    std::vector<std::vector<float>> views_chw;
    views_chw.reserve(imgs.size());
    H = 0;
    W = 0;
    for (size_t i = 0; i < imgs.size(); ++i) {
        Preprocessed p;
        if (!preprocess_real(imgs[i], ml_.config(), p)) {
            DA_ERR("depth_pose_multi: preprocess_real failed");
            return false;
        }
        if (i == 0) {
            H = p.H;
            W = p.W;
        } else if (p.H != H || p.W != W) {
            DA_ERR("depth_pose_multi: views differ in H,W");
            return false;
        }
        views_chw.push_back(std::move(p.chw));
    }
    const int S = (int)views_chw.size();
    // One backbone pass over all views (cross-view global attention).
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<std::vector<float>>> feats,
            cam_tokens;  // [L=4][S][...]
    if (!bb.forward_mv(views_chw, H, W, feats, cam_tokens)) {
        DA_ERR("depth_pose_multi: backbone failed");
        return false;
    }
    // Drop the largest activation buffer before per-view depth heads.
    be_.release_graph_memory();
    if (feats.size() < 4 || cam_tokens.size() < 4) {
        DA_ERR("depth_pose_multi: missing out layers");
        return false;
    }
    out.resize(S);
    for (int v = 0; v < S; ++v) {
        ViewResult& r = out[v];
        std::vector<std::vector<float>> feats4_v = {feats[0][v], feats[1][v],
                                                    feats[2][v], feats[3][v]};
        DptHead head(ml_, be_);
        if (!head.depth(feats4_v, H, W, r.depth, r.conf)) {
            DA_ERR("depth_pose_multi: depth head failed");
            return false;
        }
        CamPose cam(ml_, be_);
        std::array<float, 9> pe;
        if (!cam.pose(cam_tokens[3][v], H, W, pe, r.ext, r.intr)) {
            DA_ERR("depth_pose_multi: cam pose failed");
            return false;
        }
    }
    return true;
}

bool Engine::depth_pose_multi(const std::vector<Image>& imgs,
                              std::vector<ViewResult>& out,
                              int& H,
                              int& W) {
    if (imgs.size() > 1 && force_joint_multiview()) {
        return depth_pose_multi_joint(imgs, out, H, W);
    }
    out.clear();
    if (imgs.empty()) {
        DA_ERR("depth_pose_multi: no images");
        return false;
    }
    H = 0;
    W = 0;
    out.reserve(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
        ViewResult view;
        int h = 0;
        int w = 0;
        if (!depth_pose(imgs[i], view.depth, view.conf, view.ext, view.intr, h,
                        w)) {
            DA_ERR("depth_pose_multi: sequential depth_pose failed");
            return false;
        }
        if (i == 0) {
            H = h;
            W = w;
        } else if (h != H || w != W) {
            DA_ERR("depth_pose_multi: views differ in H,W");
            return false;
        }
        out.push_back(std::move(view));
        release_gpu_working_memory();
    }
    return true;
}
bool Engine::reconstruct(const Image& img, Gaussians& g, int& H, int& W) {
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("reconstruct: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;
    // One backbone pass: feats[4] feed depth + gs_head; cam_tokens[3] feeds
    // pose.
    DinoBackbone bb(ml_, be_);
    std::vector<std::vector<float>> feats, cam_tokens;
    if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
        DA_ERR("reconstruct: backbone failed");
        return false;
    }
    if (feats.size() < 4 || cam_tokens.size() < 4) {
        DA_ERR("reconstruct: missing out layers");
        return false;
    }
    DptHead head(ml_, be_);
    std::vector<float> depth, conf;
    if (!head.depth(feats, H, W, depth, conf)) {
        DA_ERR("reconstruct: depth head failed");
        return false;
    }
    CamPose cam(ml_, be_);
    std::array<float, 9> pe;
    std::array<float, 12> ext;
    std::array<float, 9> intr;
    if (!cam.pose(cam_tokens[3], H, W, pe, ext, intr)) {
        DA_ERR("reconstruct: cam pose failed");
        return false;
    }
    GsHead gs(ml_, be_);
    std::vector<float> raw_gs, gs_conf;
    if (!gs.raw_gaussians(feats, p.chw, H, W, raw_gs, gs_conf)) {
        DA_ERR("reconstruct: gs_head failed");
        return false;
    }
    GsAdapter ad;
    if (!ad.build(raw_gs, depth, gs_conf, ext, intr, H, W, g)) {
        DA_ERR("reconstruct: gs_adapter failed");
        return false;
    }
    return true;
}
bool Engine::reconstruct_path(const std::string& image_path,
                              Gaussians& g,
                              int& H,
                              int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("reconstruct: load image failed");
        return false;
    }
    return reconstruct(img, g, H, W);
}
bool Engine::depth_pose_path(const std::string& image_path,
                             std::vector<float>& depth,
                             std::vector<float>& conf,
                             std::array<float, 12>& ext,
                             std::array<float, 9>& intr,
                             int& H,
                             int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_pose: load image failed");
        return false;
    }
    return depth_pose(img, depth, conf, ext, intr, H, W);
}
bool Engine::depth_metric(const Image& img, NestedOut& out, int& H, int& W) {
    if (!metric_ml_) {
        DA_ERR("depth_metric: engine not loaded via load_nested");
        return false;
    }
    // Both branches consume the SAME preprocessed input x (da3.py
    // NestedDepthAnything3Net).
    Preprocessed p;
    if (!preprocess_real(img, ml_.config(), p)) {
        DA_ERR("depth_metric: preprocess_real failed");
        return false;
    }
    H = p.H;
    W = p.W;

    if (!ensure_anyview_gpu()) {
        return false;
    }

    // --- anyview (GIANT): backbone once -> depth + conf + cam pose ---
    AnyviewOut any;
    {
        DinoBackbone bb(ml_, be_);
        std::vector<std::vector<float>> feats, cam_tokens;
        if (!bb.forward(p.chw, H, W, feats, cam_tokens)) {
            DA_ERR("depth_metric: anyview backbone failed");
            return false;
        }
        if (cam_tokens.size() < 4) {
            DA_ERR("depth_metric: missing cam token");
            return false;
        }
        be_.release_graph_memory();
        DptHead head(ml_, be_);
        if (!head.depth(feats, H, W, any.depth, any.depth_conf)) {
            DA_ERR("depth_metric: anyview depth head failed");
            return false;
        }
        be_.release_graph_memory();
        CamPose cam(ml_, be_);
        std::array<float, 9> pe;
        if (!cam.pose(cam_tokens[3], H, W, pe, any.extrinsics,
                      any.intrinsics)) {
            DA_ERR("depth_metric: cam pose failed");
            return false;
        }
    }
    be_.release_graph_memory();
    if (be_.is_offloading()) {
        ml_.release_gpu_weights();
    }

    // --- metric (ViT-L + DPT/sky): backbone + depth_sky head ---
    MetricOut metric;
    {
        if (!ensure_metric_gpu()) {
            return false;
        }
        DinoBackbone bb(*metric_ml_, be_);
        std::vector<std::vector<float>> feats_m, cams_m;
        if (!bb.forward(p.chw, H, W, feats_m, cams_m)) {
            DA_ERR("depth_metric: metric backbone failed");
            return false;
        }
        be_.release_graph_memory();
        DptHead head(*metric_ml_, be_);
        if (!head.depth_sky(feats_m, H, W, metric.depth, metric.sky)) {
            DA_ERR("depth_metric: metric depth_sky failed");
            return false;
        }
        be_.release_graph_memory();
    }
    // The metric branch applies its own sky-fill inside da3_metric(x) before
    // alignment.
    NestedAligner::process_mono_sky(metric.depth, metric.sky);

    out = NestedAligner::align(any, metric, H, W);
    release_gpu_working_memory();
    return true;
}
bool Engine::depth_metric_path(const std::string& image_path,
                               NestedOut& out,
                               int& H,
                               int& W) {
    Image img;
    if (!load_image_rgb(image_path, img)) {
        DA_ERR("depth_metric: load image failed");
        return false;
    }
    return depth_metric(img, out, H, W);
}
bool Engine::depth_metric_multi_joint(const std::vector<Image>& imgs,
                                      std::vector<NestedOut>& out,
                                      int& H,
                                      int& W) {
    if (!metric_ml_) {
        DA_ERR("depth_metric_multi: engine not loaded via load_nested");
        return false;
    }
    out.clear();
    if (imgs.empty()) {
        DA_ERR("depth_metric_multi: no images");
        return false;
    }

    std::vector<ViewResult> any_views;
    if (!depth_pose_multi_joint(imgs, any_views, H, W)) {
        return false;
    }

    be_.release_graph_memory();
    if (be_.is_offloading()) {
        ml_.release_gpu_weights();
    }
    if (!ensure_metric_gpu()) {
        return false;
    }

    out.resize(any_views.size());
    for (size_t v = 0; v < any_views.size(); ++v) {
        Preprocessed p;
        if (!preprocess_real(imgs[v], ml_.config(), p)) {
            DA_ERR("depth_metric_multi: preprocess_real failed");
            return false;
        }

        MetricOut metric;
        {
            DinoBackbone bb(*metric_ml_, be_);
            std::vector<std::vector<float>> feats_m, cams_m;
            if (!bb.forward(p.chw, H, W, feats_m, cams_m)) {
                DA_ERR("depth_metric_multi: metric backbone failed");
                return false;
            }
            DptHead head(*metric_ml_, be_);
            if (!head.depth_sky(feats_m, H, W, metric.depth, metric.sky)) {
                DA_ERR("depth_metric_multi: metric depth_sky failed");
                return false;
            }
        }
        NestedAligner::process_mono_sky(metric.depth, metric.sky);

        AnyviewOut any;
        any.depth = any_views[v].depth;
        any.depth_conf = any_views[v].conf;
        any.extrinsics = any_views[v].ext;
        any.intrinsics = any_views[v].intr;

        out[v] = NestedAligner::align(any, metric, H, W);
    }
    release_gpu_working_memory();
    return true;
}

bool Engine::depth_metric_multi(const std::vector<Image>& imgs,
                                std::vector<NestedOut>& out,
                                int& H,
                                int& W) {
    if (!metric_ml_) {
        DA_ERR("depth_metric_multi: engine not loaded via load_nested");
        return false;
    }
    out.clear();
    if (imgs.empty()) {
        DA_ERR("depth_metric_multi: no images");
        return false;
    }
    if (imgs.size() > 1 && force_joint_multiview()) {
        return depth_metric_multi_joint(imgs, out, H, W);
    }

    H = 0;
    W = 0;
    out.reserve(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
        NestedOut view;
        int h = 0;
        int w = 0;
        if (!depth_metric(imgs[i], view, h, w)) {
            DA_ERR("depth_metric_multi: sequential depth_metric failed");
            return false;
        }
        if (i == 0) {
            H = h;
            W = w;
        } else if (h != H || w != W) {
            DA_ERR("depth_metric_multi: views differ in H,W");
            return false;
        }
        out.push_back(std::move(view));
    }
    return true;
}

namespace {

std::array<float, 16> ext12_to_ext16(const std::array<float, 12>& ext12) {
    std::array<float, 16> ext4{};
    for (int i = 0; i < 12; ++i) {
        ext4[i] = ext12[i];
    }
    ext4[12] = 0.f;
    ext4[13] = 0.f;
    ext4[14] = 0.f;
    ext4[15] = 1.f;
    return ext4;
}

}  // namespace

bool Engine::export_colmap_multi(const std::vector<Image>& imgs,
                                 const std::vector<std::string>& image_names,
                                 const std::string& dir,
                                 bool binary) {
    if (imgs.empty() || imgs.size() != image_names.size()) {
        DA_ERR("export_colmap_multi: bad args");
        return false;
    }
    const int N = static_cast<int>(imgs.size());
    int H = 0;
    int W = 0;

    std::vector<float> depth_all;
    std::vector<float> conf_all;
    std::vector<std::array<float, 9>> K(static_cast<size_t>(N));
    std::vector<std::array<float, 16>> E(static_cast<size_t>(N));
    std::vector<std::vector<uint8_t>> rgb_bufs(static_cast<size_t>(N));
    std::vector<const uint8_t*> rgb_ptrs(static_cast<size_t>(N));
    std::vector<std::pair<int, int>> orig_wh(static_cast<size_t>(N));

    if (is_nested()) {
        std::vector<NestedOut> nested;
        if (!depth_metric_multi(imgs, nested, H, W)) {
            return false;
        }
        if (static_cast<int>(nested.size()) != N) {
            DA_ERR("export_colmap_multi: view count mismatch");
            return false;
        }
        const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
        depth_all.resize(static_cast<size_t>(N) * plane);
        conf_all.resize(static_cast<size_t>(N) * plane);
        for (int v = 0; v < N; ++v) {
            std::memcpy(depth_all.data() + static_cast<size_t>(v) * plane,
                        nested[static_cast<size_t>(v)].depth.data(),
                        plane * sizeof(float));
            K[static_cast<size_t>(v)] =
                    nested[static_cast<size_t>(v)].intrinsics;
            E[static_cast<size_t>(v)] =
                    ext12_to_ext16(nested[static_cast<size_t>(v)].extrinsics);
        }
    } else {
        std::vector<ViewResult> views;
        if (!depth_pose_multi(imgs, views, H, W)) {
            return false;
        }
        if (static_cast<int>(views.size()) != N) {
            DA_ERR("export_colmap_multi: view count mismatch");
            return false;
        }
        const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
        depth_all.resize(static_cast<size_t>(N) * plane);
        bool have_conf = false;
        for (const auto& view : views) {
            if (!view.conf.empty()) {
                have_conf = true;
                break;
            }
        }
        if (have_conf) {
            conf_all.resize(static_cast<size_t>(N) * plane);
        }
        for (int v = 0; v < N; ++v) {
            std::memcpy(depth_all.data() + static_cast<size_t>(v) * plane,
                        views[v].depth.data(), plane * sizeof(float));
            if (have_conf && !views[v].conf.empty()) {
                std::memcpy(conf_all.data() + static_cast<size_t>(v) * plane,
                            views[v].conf.data(), plane * sizeof(float));
            }
            K[static_cast<size_t>(v)] = views[v].intr;
            E[static_cast<size_t>(v)] = ext12_to_ext16(views[v].ext);
        }
    }

    for (int v = 0; v < N; ++v) {
        Preprocessed pp;
        if (!preprocess_real(imgs[v], ml_.config(), pp,
                             &rgb_bufs[static_cast<size_t>(v)]) ||
            pp.H != H || pp.W != W) {
            DA_ERR("export_colmap_multi: capture processed RGB failed");
            return false;
        }
        rgb_ptrs[static_cast<size_t>(v)] =
                rgb_bufs[static_cast<size_t>(v)].data();
        orig_wh[static_cast<size_t>(v)] = {imgs[v].w, imgs[v].h};
    }

    if (!write_colmap(dir, depth_all, conf_all, K, E, rgb_ptrs, image_names,
                      orig_wh, H, W, N, binary)) {
        DA_ERR("export_colmap_multi: write_colmap failed");
        return false;
    }
    return true;
}
}  // namespace depth
}  // namespace aicore
