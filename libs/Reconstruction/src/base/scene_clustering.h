// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <list>
#include <vector>

#include "base/database.h"
#include "util/types.h"

namespace colmap {

// Scene clustering approach using normalized cuts on the scene graph. The scene
// is hierarchically partitioned into overlapping clusters until a maximum
// number of images is in a leaf node.
class SceneClustering {
public:
    struct Options {
        // Flag for hierarchical vs flat clustering
        bool is_hierarchical = true;

        // The branching factor of the hierarchical clustering.
        int branching = 2;

        // The number of overlapping images between child clusters.
        int image_overlap = 50;

        // The max related images matches to look for in a flat cluster
        int num_image_matches = 20;

        // The maximum number of images in a leaf node cluster, otherwise the
        // cluster is further partitioned using the given branching factor. Note
        // that a cluster leaf node will have at most `leaf_max_num_images +
        // overlap` images to satisfy the overlap constraint.
        int leaf_max_num_images = 500;

        bool Check() const;
    };

    struct Cluster {
        std::vector<image_t> image_ids;
        std::vector<Cluster> child_clusters;
    };

    SceneClustering(const Options& options);

    void Partition(const std::vector<std::pair<image_t, image_t>>& image_pairs,
                   const std::vector<int>& num_inliers);

    const Cluster* GetRootCluster() const;
    std::vector<const Cluster*> GetLeafClusters() const;

    static SceneClustering Create(const Options& options,
                                  const Database& database);

private:
    void PartitionHierarchicalCluster(
            const std::vector<std::pair<int, int>>& edges,
            const std::vector<int>& weights,
            Cluster* cluster);

    void PartitionFlatCluster(const std::vector<std::pair<int, int>>& edges,
                              const std::vector<int>& weights);

    const Options options_;
    std::unique_ptr<Cluster> root_cluster_;
};

}  // namespace colmap
