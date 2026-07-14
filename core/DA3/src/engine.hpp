#pragma once
#include "model_loader.hpp"
#include "backend.hpp"
#include "image_io.hpp"
#include "gs_adapter.hpp"
#include "nested.hpp"
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace da {
enum class TaskMode { DEPTH, DEPTH_POSE, MULTIVIEW, RECONSTRUCT, NESTED_METRIC };

// Per-view result of the multi-view pipeline.
struct ViewResult {
    std::vector<float> depth, conf;  // each [H*W] row-major
    std::array<float,12> ext;        // 3x4 row-major
    std::array<float,9>  intr;       // 3x3 row-major
};

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf_path, int n_threads);
    // Nested metric: loads BOTH the anyview (GIANT) GGUF and the metric (ViT-L
    // + DPT/sky) GGUF. depth_metric() then runs both branches + alignment.
    static std::unique_ptr<Engine> load_nested(const std::string& anyview_gguf,
                                               const std::string& metric_gguf, int n_threads);
    const Config& config() const { return ml_.config(); }
    // True iff this engine was created via load_nested() (anyview + metric
    // branches both loaded). depth_metric() is then the valid inference path.
    bool is_nested() const { return metric_ml_ != nullptr; }
    // True iff this is a standalone monocular checkpoint: single-head DPT
    // (output_dim==1) with a parallel sky head. Routes the CLI depth command to
    // depth_mono (depth + sky) instead of the DualDPT depth_native (depth + conf).
    bool is_mono() const;
    // True iff this is a Depth Anything V2 checkpoint (arch=="depthanything2"):
    // plain DINOv2 + single-channel DPT, relative or metric (head.max_depth>0).
    // Routes the CLI depth command to depth_relative (depth only, no pose/conf/sky).
    bool is_da2() const { return config().arch == "depthanything2"; }
    // DA2 relative/metric depth: preprocess_real (lower_bound 518) -> backbone
    // (cat_token=false) -> depth_relative head. depth_out [H*W] row-major at the
    // processed resolution (== net.forward()); CLI handles resize-to-original.
    bool depth_relative(const Image& img, std::vector<float>& depth_out, int& H, int& W);
    bool depth_relative_path(const std::string& image_path, std::vector<float>& depth_out,
                             int& H, int& W);
    // M1: debug entry returning backbone features for out_layers (filled in T16).
    bool backbone_features(const std::vector<float>& input_image, int H, int W,
                           std::vector<std::vector<float>>& feats_out);
    // Full pipeline: image file -> preprocess -> backbone -> DualDPT depth head.
    // depth_out/conf_out are [H*W] row-major; H,W set to the processed dims.
    bool depth(const std::string& image_path, std::vector<float>& depth_out,
               std::vector<float>& conf_out, int& H, int& W);
    bool depth_image(const Image& img, std::vector<float>& depth_out,
                     std::vector<float>& conf_out, int& H, int& W);
    // Native-resolution path: uses the REAL DA3 resize policy (preprocess_real,
    // upper_bound longest->target, round-to-patch, cv2 cubic/area + ImageNet
    // normalize) instead of the legacy floor-to-patch preprocess. This is the
    // production path for arbitrary-resolution real photos. H,W set to the DA3
    // processed dims (long side ~target, both multiples of patch_size).
    bool depth_native(const std::string& image_path, std::vector<float>& depth_out,
                      std::vector<float>& conf_out, int& H, int& W);
    bool depth_native_image(const Image& img, std::vector<float>& depth_out,
                            std::vector<float>& conf_out, int& H, int& W);
    // Standalone monocular path (DA3MONO): preprocess_real -> backbone (cat_token
    // false -> feat = vit.norm(x)) -> DPT depth_sky head. Produces output_dim==1
    // depth = exp(logit) plus the parallel sky map = relu(logit). Mirrors
    // depth_native_image but calls depth_sky instead of depth.
    bool depth_mono(const Image& img, std::vector<float>& depth_out,
                    std::vector<float>& sky_out, int& H, int& W);
    bool depth_mono_path(const std::string& image_path, std::vector<float>& depth_out,
                         std::vector<float>& sky_out, int& H, int& W);
    // Native-resolution depth + pose (one backbone pass).
    bool depth_pose_native(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                           std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    bool depth_pose_native_path(const std::string& image_path, std::vector<float>& depth,
                                std::vector<float>& conf, std::array<float,12>& ext,
                                std::array<float,9>& intr, int& H, int& W);
    // True iff the loaded GGUF carries the DualDPT auxiliary ray head (--with-aux),
    // i.e. the ray->pose (use_ray_pose) path is available.
    bool has_aux() const;
    // Native-resolution depth + RAY-BASED pose (use_ray_pose). One backbone pass ->
    // depth from the main head + extrinsics/intrinsics solved from the aux ray field
    // (bypasses cam_dec). Requires an aux-bearing GGUF; returns false (with a clear
    // log) on a non-aux GGUF. Production RANSAC sampling is SEEDED+deterministic.
    bool depth_pose_rays_native(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                                std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    bool depth_pose_rays_native_path(const std::string& image_path, std::vector<float>& depth,
                                     std::vector<float>& conf, std::array<float,12>& ext,
                                     std::array<float,9>& intr, int& H, int& W);
    // Full pipeline incl pose. ext = 3x4 row-major (12), intr = 3x3 row-major (9).
    bool depth_pose(const Image& img, std::vector<float>& depth, std::vector<float>& conf,
                    std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    bool depth_pose_path(const std::string& image_path, std::vector<float>& depth, std::vector<float>& conf,
                         std::array<float,12>& ext, std::array<float,9>& intr, int& H, int& W);
    // Multi-view: one backbone_mv pass over all images -> per-view depth + pose.
    // All images must preprocess to the same H,W (else returns false).
    bool depth_pose_multi(const std::vector<Image>& imgs, std::vector<ViewResult>& out, int& H, int& W);
    // Full 3D-Gaussian reconstruction (DA3-GIANT only): backbone -> depth + cam
    // pose + GSDPT raw_gs -> GaussianAdapter -> world-space Gaussians (N=H*W).
    bool reconstruct(const Image& img, Gaussians& g, int& H, int& W);
    bool reconstruct_path(const std::string& image_path, Gaussians& g, int& H, int& W);
    // Nested metric depth: anyview GIANT (depth+conf+pose) + metric (depth+sky)
    // branches -> NestedAligner -> final metric-scale depth + scaled extrinsics.
    // Requires the engine to have been created via load_nested(). out.depth is
    // [H*W] row-major.
    bool depth_metric(const Image& img, NestedOut& out, int& H, int& W);
    bool depth_metric_path(const std::string& image_path, NestedOut& out, int& H, int& W);
    // Multi-view nested metric: cross-view anyview backbone -> per-view metric
    // branch + NestedAligner. Requires load_nested(). All images must share H,W.
    bool depth_metric_multi(const std::vector<Image>& imgs, std::vector<NestedOut>& out,
                            int& H, int& W);
    // Multi-view COLMAP sparse export: depth+pose (+ metric align if nested),
    // back-project to points3D, write cameras/images/points3D under `dir`.
    bool export_colmap_multi(const std::vector<Image>& imgs,
                             const std::vector<std::string>& image_names,
                             const std::string& dir, bool binary = true);
private:
    // Fused single-image depth: backbone feats + DPT head built into ONE ggml graph
    // (feats stay device-resident — no GPU->host->GPU round-trip). Parity-exact with
    // the unfused path. cat_token=true only; depth_native_image falls back to unfused
    // otherwise or when DA_FUSED=0. The unfused variant keeps the original two-graph
    // path (used by ctest's separate backbone/head gates, A/B, and as fallback).
    bool depth_native_fused(const Image& img, std::vector<float>& depth_out,
                            std::vector<float>& conf_out, int& H, int& W);
    bool depth_native_unfused(const Image& img, std::vector<float>& depth_out,
                              std::vector<float>& conf_out, int& H, int& W);
    // Backend MUST be declared before ModelLoader: ~ModelLoader frees gpu_buf_
    // allocated on the backend; if ~Backend runs first the heap is corrupted.
    Backend be_;
    ModelLoader ml_;
    std::unique_ptr<Backend> metric_be_;
    std::unique_ptr<ModelLoader> metric_ml_;
};
} // namespace da
