// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "controllers/hierarchical_pipeline.h"

#include "scene/database_cache.h"
#include "scene/scene_clustering.h"
#include "sfm/observation_manager.h"
#include "util/misc.h"

namespace colmap {
namespace {

void MergeClusters(
    const SceneClustering::Cluster& cluster,
    std::unordered_map<const SceneClustering::Cluster*, ReconstructionManager>*
        reconstruction_managers) {
  // Extract all reconstructions from all child clusters.
  std::vector<Reconstruction*> reconstructions;
  for (const auto& child_cluster : cluster.child_clusters) {
    if (!child_cluster.child_clusters.empty()) {
      MergeClusters(child_cluster, reconstruction_managers);
    }

    auto& reconstruction_manager = reconstruction_managers->at(&child_cluster);
    for (size_t i = 0; i < reconstruction_manager.Size(); ++i) {
      reconstructions.push_back(&reconstruction_manager.Get(i));
    }
  }

  // Try to merge all child cluster reconstruction.
  while (reconstructions.size() > 1) {
    bool merge_success = false;
    for (size_t i = 0; i < reconstructions.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        const double kMaxReprojError = 8.0;
        // Upstream parity (a2bd15c2): merge j into i, then filter the merged
        // points by reprojection error.
        if (MergeAndFilterReconstructions(
                kMaxReprojError, *reconstructions[j], *reconstructions[i])) {
          reconstructions.erase(reconstructions.begin() + j);
          merge_success = true;
          break;
        }
      }

      if (merge_success) {
        break;
      }
    }

    if (!merge_success) {
      break;
    }
  }

  // Insert a new reconstruction manager for merged cluster.
  auto& reconstruction_manager = (*reconstruction_managers)[&cluster];
  for (const auto& reconstruction : reconstructions) {
    reconstruction_manager.Add();
    reconstruction_manager.Get(reconstruction_manager.Size() - 1) =
        *reconstruction;
  }

  // Delete all merged child cluster reconstruction managers.
  for (const auto& child_cluster : cluster.child_clusters) {
    reconstruction_managers->erase(&child_cluster);
  }
}

}  // namespace

bool HierarchicalPipelineOptions::Check() const {
  CHECK_OPTION_GT(init_num_trials, -1);
  CHECK_OPTION_GE(num_threads, -1);
  CHECK_OPTION_GE(num_workers, -1);
  CHECK_OPTION(clustering_options.Check());
  CHECK_OPTION(incremental_options.Check());
  return true;
}

HierarchicalPipeline::HierarchicalPipeline(
    const HierarchicalPipelineOptions& options,
    const std::filesystem::path& database_path,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_.Check());
  CHECK_EQ(options_.clustering_options.branching, 2);
}

void HierarchicalPipeline::Run() {
  PrintHeading1("Partitioning the scene");

  //////////////////////////////////////////////////////////////////////////////
  // Cluster scene
  //////////////////////////////////////////////////////////////////////////////

  // Upstream parity (1882ef4b #3926): load the database once into a cache
  // shared by the clustering and every cluster worker.
  std::shared_ptr<const DatabaseCache> database_cache;
  {
    std::cout << "Loading database" << std::endl;
    Timer timer;
    timer.Start();
    DatabaseCache::Options database_cache_options;
    database_cache_options.min_num_matches =
        static_cast<size_t>(options_.incremental_options.min_num_matches);
    database_cache_options.ignore_watermarks =
        options_.incremental_options.ignore_watermarks;
    auto database = Database::Open(database_path_);
    database_cache = DatabaseCache::Create(*database, database_cache_options);
    timer.PrintMinutes();
  }

  std::unordered_map<image_t, std::string> image_id_to_name;
  image_id_to_name.reserve(database_cache->NumImages());
  for (const auto& [image_id, image] : database_cache->Images()) {
    image_id_to_name.emplace(image_id, image.Name());
  }

  SceneClustering scene_clustering =
      SceneClustering::Create(options_.clustering_options, *database_cache);

  auto leaf_clusters = scene_clustering.GetLeafClusters();

  size_t total_num_images = 0;
  for (size_t i = 0; i < leaf_clusters.size(); ++i) {
    total_num_images += leaf_clusters[i]->image_ids.size();
    std::cout << StringPrintf("  Cluster %d with %d images", i + 1,
                              leaf_clusters[i]->image_ids.size())
              << std::endl;
  }

  std::cout << StringPrintf("Clusters have %d images", total_num_images)
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Reconstruct clusters
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Reconstructing clusters");

  // Determine the number of workers and threads per worker. The total thread
  // budget is divided across workers to avoid oversubscription (upstream
  // a2bd15c2 semantics: Mapper.num_threads is overridden per worker).
  if (options_.incremental_options.num_threads > 0) {
    LOG(WARNING)
        << "Mapper.num_threads is ignored in hierarchical mapping. Use "
           "num_threads to control the total thread budget instead.";
  }
  const int num_total_threads = GetEffectiveNumThreads(options_.num_threads);
  const int kDefaultNumWorkers = 8;
  const int num_eff_workers = std::max(
      1, std::min(static_cast<int>(leaf_clusters.size()),
                  std::min(options_.num_workers < 1 ? kDefaultNumWorkers
                                                    : options_.num_workers,
                           num_total_threads)));
  const int num_threads_per_worker =
      std::max(1, num_total_threads / num_eff_workers);

  // Function to reconstruct one cluster using incremental mapping.
  auto ReconstructCluster = [&, this](
                                const SceneClustering::Cluster& cluster,
                                ReconstructionManager* reconstruction_manager) {
    if (cluster.image_ids.empty()) {
      return;
    }

    IncrementalPipelineOptions custom_options = options_.incremental_options;
    custom_options.max_model_overlap = 3;
    custom_options.init_num_trials = options_.init_num_trials;
    custom_options.num_threads = num_threads_per_worker;

    // Create a filtered database cache for this cluster.
    DatabaseCache::Options cluster_cache_options;
    cluster_cache_options.min_num_matches =
        static_cast<size_t>(options_.incremental_options.min_num_matches);
    for (const auto image_id : cluster.image_ids) {
      cluster_cache_options.image_names.insert(image_id_to_name.at(image_id));
    }
    std::shared_ptr<const DatabaseCache> cluster_database_cache =
        DatabaseCache::CreateFromCache(*database_cache, cluster_cache_options);

    IncrementalPipeline mapper(&custom_options, options_.image_path,
                               std::move(cluster_database_cache),
                               reconstruction_manager);
    mapper.Start();
    mapper.Wait();
  };

  // Start reconstructing the bigger clusters first for resource usage.
  std::sort(leaf_clusters.begin(), leaf_clusters.end(),
            [](const SceneClustering::Cluster* cluster1,
               const SceneClustering::Cluster* cluster2) {
              return cluster1->image_ids.size() > cluster2->image_ids.size();
            });

  // Start the reconstruction workers.

  std::unordered_map<const SceneClustering::Cluster*, ReconstructionManager>
      reconstruction_managers;
  reconstruction_managers.reserve(leaf_clusters.size());

  ThreadPool thread_pool(num_eff_workers);
  for (const auto& cluster : leaf_clusters) {
    thread_pool.AddTask(ReconstructCluster, *cluster,
                        &reconstruction_managers[cluster]);
  }
  thread_pool.Wait();

  //////////////////////////////////////////////////////////////////////////////
  // Merge clusters
  //////////////////////////////////////////////////////////////////////////////

  if (leaf_clusters.size() > 1) {
    PrintHeading1("Merging clusters");

    MergeClusters(*scene_clustering.GetRootCluster(), &reconstruction_managers);
  }

  CHECK_EQ(reconstruction_managers.size(), 1);
  *reconstruction_manager_ = std::move(reconstruction_managers.begin()->second);

  // Cluster filtering stores normalized/angular residuals in Point3D::Error.
  // Publish the final model with pixel reprojection errors, matching the
  // reconstruction contract consumed by model analysis and the upstream
  // hierarchical pipeline.
  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    reconstruction_manager_->Get(i).UpdatePoint3DErrors();
  }

  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

}  // namespace colmap
