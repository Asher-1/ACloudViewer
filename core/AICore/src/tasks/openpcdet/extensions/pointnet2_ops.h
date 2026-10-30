#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace openpcdet::ext {

// Reusable source-point index for the multiple radius branches in one
// PointNet2 set-abstraction stage. Queries preserve the source-index ordering
// of the reference linear scan, including its first-neighbor padding rule.
class ball_query_index {
public:
    ball_query_index();
    ~ball_query_index();
    ball_query_index(ball_query_index &&) noexcept;
    ball_query_index & operator=(ball_query_index &&) noexcept;
    ball_query_index(const ball_query_index &) = delete;
    ball_query_index & operator=(const ball_query_index &) = delete;

    bool build(const float * xyz, int64_t point_count, float cell_size,
               std::string & error);
    bool query(const float * centers, int64_t center_count, float radius,
               int nsample, std::vector<int32_t> & indices,
               std::string & error) const;

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

bool farthest_point_sample(const float * xyz, int64_t point_count, int sample_count,
                           std::vector<int32_t> & indices, std::string & error);

// Returns [center_count, nsample]. Missing neighborhoods contain -1; short
// neighborhoods repeat their first member, matching PointNet2 grouping use.
bool ball_query(const float * xyz, int64_t point_count,
                const float * centers, int64_t center_count,
                float radius, int nsample,
                std::vector<int32_t> & indices, std::string & error);

// Input features are [point_count, channels], output is
// [group_count, samples_per_group, channels]. Negative indices produce zeros.
bool group_points(const float * features, int64_t point_count, int channels,
                  const int32_t * indices, int64_t group_count, int samples_per_group,
                  std::vector<float> & output, std::string & error);

// For each unknown point, return squared distances and indices of the three
// nearest known points. If fewer than three known points exist, the nearest is
// repeated.
bool three_nearest_neighbors(const float * unknown, int64_t unknown_count,
                             const float * known, int64_t known_count,
                             std::vector<float> & squared_distances,
                             std::vector<int32_t> & indices,
                             std::string & error);

bool three_interpolate(const float * known_features, int64_t known_count, int channels,
                       const int32_t * indices, const float * weights,
                       int64_t output_count, std::vector<float> & output,
                       std::string & error);

// PointNet2 set-abstraction geometry for one scale. `center_indices` indexes
// xyz and is normally produced by FPS. Output layout is
// [center_count, nsample, (use_xyz ? 3 : 0) + channels], with xyz expressed
// relative to each center and input features appended after it.
bool sample_and_group(const float * xyz, int64_t point_count,
                      const float * features, int channels,
                      const int32_t * center_indices, int64_t center_count,
                      float radius, int nsample, bool use_xyz,
                      std::vector<float> & center_xyz,
                      std::vector<int32_t> & group_indices,
                      std::vector<float> & grouped_features,
                      std::string & error,
                      const ball_query_index * query_index = nullptr);

// PointNet2 feature-propagation geometry. Three-neighbor inverse-distance
// interpolation is concatenated with optional skip features. Output is
// [unknown_count, known_channels + skip_channels].
//
// AICore port note: the upstream CUDA geometry extensions (.cu) are dropped;
// this host implementation runs identically on every backend.
bool feature_propagate(const float * unknown_xyz, int64_t unknown_count,
                       const float * known_xyz, int64_t known_count,
                       const float * known_features, int known_channels,
                       const float * skip_features, int skip_channels,
                       std::vector<float> & output, std::string & error);

enum class vector_pool_mode {
    local_interpolation,
    voxel_average,
    voxel_first,
};

struct vector_pool_params {
    int grid_x = 1;
    int grid_y = 1;
    int grid_z = 1;
    float max_neighbor_distance = 0.0f;
    int nsample = -1;
    bool ball_neighbors = false;
    float neighbor_distance_multiplier = 2.0f;
    int reduced_channels = 0;
    vector_pool_mode mode = vector_pool_mode::local_interpolation;
};

// PV-RCNN++ VectorPool geometry before the grouped 1x1 MLP. Input channels
// are reduced with view(N,-1,reduced_channels).sum(1). The output is
// [center_count, grid_count * (reduced_channels + 9)] for interpolation and
// [center_count, grid_count * (reduced_channels + 3)] for voxel pooling.
bool vector_pool(const float * support_xyz, int64_t support_count,
                 const float * support_features, int channels,
                 const float * centers, int64_t center_count,
                 const vector_pool_params & params,
                 std::vector<float> & output, std::string & error);

} // namespace openpcdet::ext
