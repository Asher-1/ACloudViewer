// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "controllers/base_option_manager.h"

namespace colmap {

struct ImageReaderOptions;
struct SiftExtractionOptions;
struct SiftMatchingOptions;
struct ExhaustiveMatchingOptions;
struct SequentialMatchingOptions;
struct VocabTreeMatchingOptions;
struct SpatialMatchingOptions;
struct TransitiveMatchingOptions;
struct ImagePairsMatchingOptions;
struct BundleAdjustmentOptions;
struct IncrementalMapperOptions;
struct GlobalPipelineOptions;
struct GravityRefinerOptions;
struct TexturingOptions;
struct ReconstructionClusteringOptions;
namespace mvs {
struct MeshTextureMappingOptions;
}  // namespace mvs
struct RenderOptions;

namespace mvs {
struct PatchMatchOptions;
struct StereoFusionOptions;
struct PoissonMeshingOptions;
struct DelaunayMeshingOptions;
struct AdvancingFrontMeshingOptions;
struct MeshSimplificationOptions;
struct MeshPostProcessingOptions;
}  // namespace mvs

class OptionManager : public BaseOptionManager {
public:
    OptionManager(bool add_project_options = true);

    // Create "optimal" set of options for different reconstruction scenarios.
    // Note that the existing options are modified, so if your parameters are
    // already low quality, they will be further modified.
    void ModifyForIndividualData();
    void ModifyForVideoData();
    void ModifyForInternetData();

    // Create "optimal" set of options for different quality settings.
    // Note that the existing options are modified, so if your parameters are
    // already low quality, they will be further degraded.
    void ModifyForLowQuality();
    void ModifyForMediumQuality();
    void ModifyForHighQuality();
    void ModifyForExtremeQuality();

    void AddAllOptions() override;
    void PostParse() override;

    // Fork legacy flag spellings (--log_to_stderr/--random_seed); the base
    // group entry points are virtual so these stay authoritative. The
    // database/image groups reuse BaseOptionManager's (identical contracts).
    void AddRandomOptions() override;
    void AddLogOptions() override;
    void AddExtractionOptions();
    void AddMatchingOptions();
    void AddExhaustiveMatchingOptions();
    void AddSequentialMatchingOptions();
    void AddVocabTreeMatchingOptions();
    void AddSpatialMatchingOptions();
    void AddTransitiveMatchingOptions();
    void AddImagePairsMatchingOptions();
    void AddBundleAdjustmentOptions();
    void AddMapperOptions();
    // Upstream parity (dbb41680 controllers/option_manager.h): options for
    // the global SfM pipeline (the global_mapper command).
    void AddGlobalMapperOptions();
    // Upstream parity (dbb41680 controllers/option_manager.h): options for
    // the gravity refinement pass (the rotation_averager command).
    void AddGravityRefinerOptions();
    void AddPatchMatchStereoOptions();
    void AddStereoFusionOptions();
    void AddPoissonMeshingOptions();
    void AddDelaunayMeshingOptions();
    void AddAdvancingFrontMeshingOptions();
    void AddMeshSimplificationOptions();
    void AddMeshPostProcessingOptions();
    void AddTexturingOptions();
    void AddReconstructionClustererOptions();
    void AddMeshTextureMappingOptions();
    void AddRenderOptions();

    void Reset(bool reset_logging = true) override;
    void ResetOptions(const bool reset_paths) override;

    bool Check() override;

    bool Read(const std::filesystem::path& path,
              bool allow_unregistered = true) override;
    void PrintHelp() const override;
    // Fork legacy: the fork sets glog flags directly (Reset/log_to_stderr);
    // the base's log_target_ mapping must not override them after Parse.
    void ApplyLogFlags() override;

    // The project/database/image path shared_ptrs are inherited from
    // BaseOptionManager (allocated in its constructor).

    std::shared_ptr<ImageReaderOptions> image_reader;
    std::shared_ptr<SiftExtractionOptions> sift_extraction;

    std::shared_ptr<SiftMatchingOptions> sift_matching;
    std::shared_ptr<ExhaustiveMatchingOptions> exhaustive_matching;
    std::shared_ptr<SequentialMatchingOptions> sequential_matching;
    std::shared_ptr<VocabTreeMatchingOptions> vocab_tree_matching;
    std::shared_ptr<SpatialMatchingOptions> spatial_matching;
    std::shared_ptr<TransitiveMatchingOptions> transitive_matching;
    std::shared_ptr<ImagePairsMatchingOptions> image_pairs_matching;

    std::shared_ptr<BundleAdjustmentOptions> bundle_adjustment;
    std::shared_ptr<IncrementalMapperOptions> mapper;
    std::shared_ptr<GlobalPipelineOptions> global_mapper;
    // Upstream parity (dbb41680 controllers/option_manager.h): the gravity
    // refiner options consumed by the rotation_averager command.
    std::shared_ptr<GravityRefinerOptions> gravity_refiner;

    std::shared_ptr<mvs::PatchMatchOptions> patch_match_stereo;
    std::shared_ptr<mvs::StereoFusionOptions> stereo_fusion;
    std::shared_ptr<mvs::PoissonMeshingOptions> poisson_meshing;
    std::shared_ptr<mvs::DelaunayMeshingOptions> delaunay_meshing;
    std::shared_ptr<mvs::AdvancingFrontMeshingOptions> advancing_front_meshing;
    std::shared_ptr<mvs::MeshSimplificationOptions> mesh_simplification;
    std::shared_ptr<mvs::MeshPostProcessingOptions> mesh_post_processing;
    std::shared_ptr<TexturingOptions> texturing;
    std::shared_ptr<ReconstructionClusteringOptions> reconstruction_clusterer;
    std::shared_ptr<mvs::MeshTextureMappingOptions> mesh_texture_mapping;

    std::shared_ptr<RenderOptions> render;

private:
    void ResetOptionStructs();

    // The group-registration call sites in option_manager.cc keep the fork
    // spelling; both forward to the base (which also registers the option
    // into the typed registries, upstream semantics).
    template <typename T>
    void AddAndRegisterRequiredOption(const std::string& name,
                                      T* option,
                                      const std::string& help_text = "") {
        BaseOptionManager::AddRequiredOption(name, option, help_text);
    }
    template <typename T>
    void AddAndRegisterDefaultOption(const std::string& name,
                                     T* option,
                                     const std::string& help_text = "") {
        BaseOptionManager::AddDefaultOption(name, option, help_text);
    }
    template <typename T>
    void AddOptionAlias(const std::string& name,
                        T* option,
                        const std::string& help_text = "") {
        BaseOptionManager::AddAliasOption(name, option, help_text);
    }

    bool added_extraction_options_ = false;
    bool added_match_options_ = false;
    bool added_exhaustive_match_options_ = false;
    bool added_sequential_match_options_ = false;
    bool added_vocab_tree_match_options_ = false;
    bool added_spatial_match_options_ = false;
    bool added_transitive_match_options_ = false;
    bool added_image_pairs_match_options_ = false;
    bool added_ba_options_ = false;
    bool added_mapper_options_ = false;
    bool added_global_mapper_options_ = false;
    bool added_gravity_refiner_options_ = false;
    bool added_reconstruction_clusterer_options_ = false;
    bool added_mesh_texture_mapping_options_ = false;
    std::filesystem::path mapper_image_list_path_;
    std::filesystem::path global_mapper_image_list_path_;

    bool added_patch_match_stereo_options_ = false;
    bool added_stereo_fusion_options_ = false;
    bool added_poisson_meshing_options_ = false;
    bool added_delaunay_meshing_options_ = false;
    bool added_advancing_front_meshing_options_ = false;
    bool added_mesh_simplification_options_ = false;
    bool added_mesh_post_processing_options_ = false;
    bool added_texturing_options_ = false;
    bool added_render_options_ = false;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

}  // namespace colmap
