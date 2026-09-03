// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------

#include "controllers/global_mapper.h"

#include <algorithm>
#include <unordered_map>

#include "base/database.h"

namespace colmap {
namespace {

class DisjointSet {
public:
    explicit DisjointSet(const std::vector<image_t>& ids) {
        for (const image_t id : ids) parent_[id] = id;
    }

    image_t Find(const image_t id) {
        image_t& parent = parent_.at(id);
        if (parent != id) parent = Find(parent);
        return parent;
    }

    void Union(const image_t first, const image_t second) {
        const image_t root_first = Find(first);
        const image_t root_second = Find(second);
        if (root_first != root_second) parent_[root_second] = root_first;
    }

private:
    std::unordered_map<image_t, image_t> parent_;
};

}  // namespace

bool GlobalMapperController::Options::Check() const {
    CHECK_OPTION_GE(min_component_size, 1);
    CHECK_OPTION_GE(num_workers, -1);
    CHECK_OPTION_GT(init_num_trials, 0);
    return true;
}

GlobalMapperController::GlobalMapperController(
    const Options& options, const IncrementalMapperOptions& mapper_options,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      mapper_options_(mapper_options),
      reconstruction_manager_(reconstruction_manager) {
    CHECK(options_.Check());
    CHECK(mapper_options_.Check());
    CHECK(reconstruction_manager_ != nullptr);
}

void GlobalMapperController::Run() {
    Database database(options_.database_path);
    const auto images = database.ReadAllImages();
    std::vector<image_t> image_ids;
    std::unordered_map<image_t, std::string> image_id_to_name;
    image_ids.reserve(images.size());
    for (const auto& image : images) {
        image_ids.push_back(image.ImageId());
        image_id_to_name.emplace(image.ImageId(), image.Name());
    }

    DisjointSet components(image_ids);
    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::vector<int> num_inliers;
    database.ReadTwoViewGeometryNumInliers(&image_pairs, &num_inliers);
    for (const auto& pair : image_pairs) components.Union(pair.first, pair.second);

    std::unordered_map<image_t, std::vector<image_t>> grouped;
    for (const image_t image_id : image_ids) {
        grouped[components.Find(image_id)].push_back(image_id);
    }

    std::vector<std::vector<image_t>> component_ids;
    for (auto& group : grouped) {
        if (group.second.size() >=
            static_cast<size_t>(options_.min_component_size)) {
            component_ids.emplace_back(std::move(group.second));
        }
    }
    std::sort(component_ids.begin(), component_ids.end(),
              [](const auto& first, const auto& second) {
                  return first.size() > second.size();
              });

    std::vector<ReconstructionManager> managers(component_ids.size());
    const int num_threads = GetEffectiveNumThreads(-1);
    const int workers = options_.num_workers < 1
                            ? std::max(1, std::min<int>(num_threads,
                                                        component_ids.size()))
                            : options_.num_workers;
    ThreadPool pool(workers);
    for (size_t component_idx = 0; component_idx < component_ids.size();
         ++component_idx) {
        pool.AddTask([&, component_idx] {
            IncrementalMapperOptions options = mapper_options_;
            options.init_num_trials = options_.init_num_trials;
            options.multiple_models = false;
            options.image_names.clear();
            for (const image_t image_id : component_ids[component_idx]) {
                options.image_names.insert(image_id_to_name.at(image_id));
            }
            IncrementalMapperController mapper(
                &options, options_.image_path, options_.database_path,
                &managers[component_idx]);
            mapper.Start();
            mapper.Wait();
        });
    }
    pool.Wait();

    reconstruction_manager_->Clear();
    for (const auto& manager : managers) {
        for (size_t idx = 0; idx < manager.Size(); ++idx) {
            reconstruction_manager_->Add();
            reconstruction_manager_->Get(reconstruction_manager_->Size() - 1) =
                manager.Get(idx);
        }
    }
}

}  // namespace colmap
