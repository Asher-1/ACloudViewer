// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "base/reconstruction_manager.h"
#include "controllers/da3_depth_controller.h"
#include "retrieval/resources.h"
#include "util/option_manager.h"
#include "util/ply_point_filter.h"
#include "util/threading.h"

namespace colmap {

class AutomaticReconstructionController : public Thread {
public:
    enum class DataType { INDIVIDUAL, VIDEO, INTERNET };
    enum class Quality { LOW, MEDIUM, HIGH, EXTREME };
    enum class Mesher { POISSON, DELAUNAY };

    struct Options {
        // The path to the workspace folder in which all results are stored.
        std::string workspace_path;

        // The path to the image folder which are used as input.
        std::string image_path;

        // The path to the mask folder which are used as input.
        std::string mask_path;

        // The path to the vocabulary tree for feature matching.
        std::string vocab_tree_path = retrieval::kDefaultVocabTreeUri;

        // The type of input data used to choose optimal mapper settings.
        DataType data_type = DataType::INDIVIDUAL;

        // Whether to perform low- or high-quality reconstruction.
        Quality quality = Quality::HIGH;

        // Whether to use shared intrinsics or not.
        bool single_camera = false;

        // Which camera model to use for images.
        std::string camera_model = "SIMPLE_RADIAL";

        // Whether to perform sparse mapping.
        bool sparse = true;

// Whether to perform dense mapping.
#ifdef CUDA_ENABLED
        bool dense = true;
#else
        bool dense = false;
#endif

        // The meshing algorithm to be used.
#ifdef CGAL_ENABLED
        Mesher mesher = Mesher::DELAUNAY;
#else
        Mesher mesher = Mesher::POISSON;
#endif

        // Whether to perform surface meshing (Poisson / Delaunay).
        bool meshing = true;

        // Whether to perform surface texturing.
        bool texturing = true;

        // The number of threads to use in all stages.
        int num_threads = -1;

        // Whether to use the GPU in feature extraction and matching.
#ifdef CUDA_ENABLED
        bool use_gpu = true;
#else
        bool use_gpu = false;
#endif

        // Index of the GPU used for GPU stages. For multi-GPU computation,
        // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
        // By default, all GPUs will be used in all stages.
        std::string gpu_index = "-1";

        // --- DA3 (Depth Anything V3) integration options ---

        // Sparse model generation mode:
        //   COLMAP_NATIVE  - traditional COLMAP SfM pipeline (default)
        //   DA3_DEPTH_POSE - DA3 monocular depth+pose estimation
        SparseModelMode sparse_mode = SparseModelMode::COLMAP_NATIVE;

        // Dense/stereo pipeline mode:
        //   COLMAP_PATCH_MATCH  - traditional COLMAP PatchMatch stereo
        //   (default) DA3_DEPTH_INFERENCE - DA3 ggml-based depth inference
        //   (faster)
        StereoPipelineMode stereo_mode = StereoPipelineMode::COLMAP_PATCH_MATCH;

        // DA3 sparse-step model (when sparse_mode == DA3_DEPTH_POSE)
        DA3ModelType da3_sparse_model_type = DA3ModelType::BASE;
        DA3QuantType da3_sparse_quant_type = DA3QuantType::Q8_0;
        std::string da3_sparse_model_path;
        std::string da3_sparse_metric_model_path;

        // DA3 stereo-step model (when stereo_mode == DA3_DEPTH_INFERENCE;
        // nested only)
        DA3ModelType da3_stereo_model_type = DA3ModelType::NESTED_ANYVIEW;
        DA3QuantType da3_stereo_quant_type = DA3QuantType::Q8_0;
        std::string da3_stereo_model_path;
        std::string da3_stereo_metric_model_path;

        // When true, ignore cached sparse/stereo/fusion outputs and recompute.
        bool da3_force_recompute = false;

        // Hybrid dense: fuse DA3 priors directly (skip PatchMatch geometric).
        // Auto-fallback to geometric refine when fusion point count is low.
        bool da3_skip_geometric_refine = false;

        // Optional voxel + SOR cleanup on fused.ply before Poisson meshing.
        FusedPointFilterOptions fused_point_filter;
    };

    AutomaticReconstructionController(
            const Options& options,
            ReconstructionManager* reconstruction_manager);

    void Stop() override;

protected:
    virtual void RunDenseMapper();

    // Hook methods for derived classes to customize behavior
    virtual void OnFusedPointsGenerated(size_t reconstruction_idx,
                                        const std::vector<PlyPoint>& points) {}
    virtual void OnMeshGenerated(size_t reconstruction_idx,
                                 const std::string& mesh_path) {}
    virtual void OnTexturedMeshGenerated(size_t reconstruction_idx,
                                         const std::string& textured_path) {}

    // Protected members for derived classes
    const Options options_;
    OptionManager option_manager_;
    ReconstructionManager* reconstruction_manager_;
    Thread* active_thread_;

private:
    void Run() override;
    void RunFeatureExtraction();
    void RunFeatureMatching();
    void RunSparseMapper();
    void RunDA3SparseMapper();
    void RunDA3DepthMaps();

    bool use_da3_stereo_maps_ = false;
    bool da3_reuse_sparse_stereo_ = false;
    // Single multiview on undistorted images; workspace sparse synced from
    // dense.
    bool da3_unified_undistorted_ = false;
    // Large sets: COLMAP SfM sparse + DA3 sequential depth (not DA3 joint
    // sparse).
    bool da3_auto_colmap_sparse_ = false;
    // COLMAP SfM + DA3 photometric priors + PatchMatch refine + StereoFusion.
    bool da3_patchmatch_refine_ = false;
    // StereoFusion on DA3 priors only (DA3_SKIP_GEOMETRIC_REFINE=1).
    bool da3_skip_geometric_refine_ = false;
    DA3MultiviewCache da3_multiview_cache_;

    std::unique_ptr<Thread> feature_extractor_;
    std::unique_ptr<Thread> exhaustive_matcher_;
    std::unique_ptr<Thread> sequential_matcher_;
    std::unique_ptr<Thread> vocab_tree_matcher_;
};

}  // namespace colmap
