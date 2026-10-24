// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>

#include "controllers/incremental_pipeline.h"
#include "scene/reconstruction_manager.h"
#include "scene/scene_clustering.h"
#include "util/threading.h"

namespace colmap {

struct HierarchicalPipelineOptions {
    // The path to the image folder which are used as input.
    std::filesystem::path image_path;

    // The number of trials to initialize a cluster.
    int init_num_trials = 10;

    // The total number of threads for the entire pipeline, shared across the
    // workers (-1 uses all available cores).
    int num_threads = -1;

    // The number of workers used to reconstruct clusters in parallel.
    int num_workers = -1;

    // Options for clustering the scene graph.
    SceneClustering::Options clustering_options;

    // Options used to reconstruct the individual clusters.
    IncrementalPipelineOptions incremental_options;

    bool Check() const;
};

// Hierarchical mapping first hierarchically partitions the scene into multiple
// overlapping clusters, then reconstructs them separately using incremental
// mapping, and finally merges them all into a globally consistent
// reconstruction. This is especially useful for larger-scale scenes, since
// incremental mapping becomes slow with an increasing number of images.
//
// The fork runs its pipelines as util/threading Thread objects (upstream uses
// BaseController + ControllerThread); the database is addressed by path.
class HierarchicalPipeline : public Thread {
public:
    HierarchicalPipeline(const HierarchicalPipelineOptions& options,
                         const std::filesystem::path& database_path,
                         ReconstructionManager* reconstruction_manager);

private:
    void Run() override;

    const HierarchicalPipelineOptions options_;
    const std::filesystem::path database_path_;
    ReconstructionManager* reconstruction_manager_;
};

}  // namespace colmap
