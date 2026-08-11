// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "util/ply_point_filter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "FLANN/flann.hpp"
#include "util/logging.h"
#include "util/threading.h"

namespace colmap {
namespace {

struct AccumulatedPlyPoint {
    int count = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    std::vector<int> visibility;

    void Add(const PlyPoint& point) {
        x += point.x;
        y += point.y;
        z += point.z;
        if (std::isfinite(point.nx) && std::isfinite(point.ny) &&
            std::isfinite(point.nz)) {
            nx += point.nx;
            ny += point.ny;
            nz += point.nz;
        }
        r += point.r;
        g += point.g;
        b += point.b;
        ++count;
    }

    void MergeVisibility(const std::vector<int>& image_indices) {
        visibility.insert(visibility.end(), image_indices.begin(),
                          image_indices.end());
        std::sort(visibility.begin(), visibility.end());
        visibility.erase(std::unique(visibility.begin(), visibility.end()),
                         visibility.end());
    }

    PlyPoint Average() const {
        PlyPoint out;
        if (count <= 0) {
            return out;
        }
        const double inv = 1.0 / static_cast<double>(count);
        out.x = static_cast<float>(x * inv);
        out.y = static_cast<float>(y * inv);
        out.z = static_cast<float>(z * inv);
        const double nlen =
            std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nlen > 1e-12) {
            out.nx = static_cast<float>(nx / nlen);
            out.ny = static_cast<float>(ny / nlen);
            out.nz = static_cast<float>(nz / nlen);
        }
        out.r = static_cast<uint8_t>(std::lround(std::clamp(r * inv, 0.0, 255.0)));
        out.g = static_cast<uint8_t>(std::lround(std::clamp(g * inv, 0.0, 255.0)));
        out.b = static_cast<uint8_t>(std::lround(std::clamp(b * inv, 0.0, 255.0)));
        return out;
    }
};

struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& key) const {
        const size_t hx = std::hash<int>{}(key.x);
        const size_t hy = std::hash<int>{}(key.y);
        const size_t hz = std::hash<int>{}(key.z);
        return hx ^ (hy << 1) ^ (hz << 2);
    }
};

void ComputeBounds(const std::vector<PlyPoint>& points,
                   double& min_x,
                   double& min_y,
                   double& min_z,
                   double& max_x,
                   double& max_y,
                   double& max_z) {
    min_x = min_y = min_z = std::numeric_limits<double>::infinity();
    max_x = max_y = max_z = -std::numeric_limits<double>::infinity();
    for (const auto& point : points) {
        min_x = std::min(min_x, static_cast<double>(point.x));
        min_y = std::min(min_y, static_cast<double>(point.y));
        min_z = std::min(min_z, static_cast<double>(point.z));
        max_x = std::max(max_x, static_cast<double>(point.x));
        max_y = std::max(max_y, static_cast<double>(point.y));
        max_z = std::max(max_z, static_cast<double>(point.z));
    }
}

bool ShouldVoxelDownsample(const FusedPointFilterOptions& options) {
    return !options.skip_voxel_downsample && options.voxel_size > 0.0;
}

void ComputeStatisticalOutlierAvgDistances(
        const std::vector<PlyPoint>& points,
        int nb_neighbors,
        int num_threads,
        flann::Index<flann::L2<double>>& index,
        const std::vector<double>& dataset,
        std::vector<double>& avg_distances) {
    const int k = nb_neighbors + 1;
    const flann::SearchParams search_params(128);

#ifdef OPENMP_ENABLED
    const int eff_threads = GetEffectiveNumThreads(num_threads);
    omp_set_num_threads(eff_threads);
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < points.size(); ++i) {
#else
    for (size_t i = 0; i < points.size(); ++i) {
#endif
        std::vector<int> indices(k);
        std::vector<double> dists(k);
        flann::Matrix<double> query(const_cast<double*>(&dataset[i * 3]), 1, 3);
        flann::Matrix<int> indices_mat(indices.data(), 1, k);
        flann::Matrix<double> dists_mat(dists.data(), 1, k);
        const int found =
                index.knnSearch(query, indices_mat, dists_mat, k, search_params);
        if (found <= 1) {
            continue;
        }

        double mean = 0.0;
        int used = 0;
        for (int j = 1; j < found; ++j) {
            mean += std::sqrt(dists[static_cast<size_t>(j)]);
            ++used;
        }
        if (used > 0) {
            avg_distances[i] = mean / static_cast<double>(used);
        }
    }
}

std::vector<size_t> KeepIndicesFromAvgDistances(
        const std::vector<double>& avg_distances, double std_ratio) {
    size_t valid_distances = 0;
    double cloud_mean = 0.0;
    for (const double dist : avg_distances) {
        if (dist >= 0.0) {
            cloud_mean += dist;
            ++valid_distances;
        }
    }
    std::vector<size_t> kept_indices;
    kept_indices.reserve(avg_distances.size());
    if (valid_distances == 0) {
        kept_indices.resize(avg_distances.size());
        std::iota(kept_indices.begin(), kept_indices.end(), 0);
        return kept_indices;
    }
    cloud_mean /= static_cast<double>(valid_distances);

    double sq_sum = 0.0;
    for (const double dist : avg_distances) {
        if (dist >= 0.0) {
            const double delta = dist - cloud_mean;
            sq_sum += delta * delta;
        }
    }
    const double std_dev =
            valid_distances > 1
                    ? std::sqrt(sq_sum / static_cast<double>(valid_distances - 1))
                    : 0.0;
    const double distance_threshold = cloud_mean + std_ratio * std_dev;

    for (size_t i = 0; i < avg_distances.size(); ++i) {
        if (avg_distances[i] >= 0.0 &&
            avg_distances[i] < distance_threshold) {
            kept_indices.push_back(i);
        }
    }
    return kept_indices;
}

std::vector<PlyPoint> RunStatisticalOutlierRemoval(
        const std::vector<PlyPoint>& points,
        int nb_neighbors,
        double std_ratio,
        int num_threads) {
    if (points.empty() || nb_neighbors < 1 || std_ratio <= 0.0) {
        return points;
    }
    if (static_cast<int>(points.size()) <= nb_neighbors) {
        return points;
    }

    std::vector<double> dataset(points.size() * 3);
    for (size_t i = 0; i < points.size(); ++i) {
        dataset[i * 3 + 0] = points[i].x;
        dataset[i * 3 + 1] = points[i].y;
        dataset[i * 3 + 2] = points[i].z;
    }

    flann::Matrix<double> flann_data(dataset.data(), points.size(), 3);
    flann::Index<flann::L2<double>> index(
            flann_data, flann::KDTreeIndexParams(4));
    index.buildIndex();

    std::vector<double> avg_distances(points.size(), -1.0);
    ComputeStatisticalOutlierAvgDistances(points, nb_neighbors, num_threads,
                                          index, dataset, avg_distances);
    const auto kept_indices =
            KeepIndicesFromAvgDistances(avg_distances, std_ratio);

    std::vector<PlyPoint> filtered;
    filtered.reserve(kept_indices.size());
    for (const size_t idx : kept_indices) {
        filtered.push_back(points[idx]);
    }
    return filtered;
}

std::vector<size_t> StatisticalOutlierRemovalIndices(
        const std::vector<PlyPoint>& points,
        int nb_neighbors,
        double std_ratio,
        int num_threads) {
    if (points.empty() || nb_neighbors < 1 || std_ratio <= 0.0) {
        std::vector<size_t> kept_indices(points.size());
        std::iota(kept_indices.begin(), kept_indices.end(), 0);
        return kept_indices;
    }
    if (static_cast<int>(points.size()) <= nb_neighbors) {
        std::vector<size_t> kept_indices(points.size());
        std::iota(kept_indices.begin(), kept_indices.end(), 0);
        return kept_indices;
    }

    std::vector<double> dataset(points.size() * 3);
    for (size_t i = 0; i < points.size(); ++i) {
        dataset[i * 3 + 0] = points[i].x;
        dataset[i * 3 + 1] = points[i].y;
        dataset[i * 3 + 2] = points[i].z;
    }

    flann::Matrix<double> flann_data(dataset.data(), points.size(), 3);
    flann::Index<flann::L2<double>> index(
            flann_data, flann::KDTreeIndexParams(4));
    index.buildIndex();

    std::vector<double> avg_distances(points.size(), -1.0);
    ComputeStatisticalOutlierAvgDistances(points, nb_neighbors, num_threads,
                                          index, dataset, avg_distances);
    return KeepIndicesFromAvgDistances(avg_distances, std_ratio);
}

FusedPointCloudWithVisibility VoxelDownsamplePlyPointsWithVisibility(
        const FusedPointCloudWithVisibility& cloud, double voxel_size) {
    FusedPointCloudWithVisibility filtered = cloud;
    if (cloud.points.empty() || voxel_size <= 0.0) {
        return filtered;
    }
    if (cloud.visibility.size() != cloud.points.size()) {
        filtered.visibility.assign(cloud.points.size(), std::vector<int>{});
    }

    double min_x = 0.0;
    double min_y = 0.0;
    double min_z = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double max_z = 0.0;
    ComputeBounds(cloud.points, min_x, min_y, min_z, max_x, max_y, max_z);

    const double half = 0.5 * voxel_size;
    const double origin_x = min_x - half;
    const double origin_y = min_y - half;
    const double origin_z = min_z - half;
    const double max_extent =
            std::max({max_x - min_x, max_y - min_y, max_z - min_z}) + voxel_size;
    if (voxel_size * static_cast<double>(std::numeric_limits<int>::max()) <
        max_extent) {
        LOG(WARNING) << "Fused point voxel size is too small; skipping voxel "
                        "downsample.";
        return filtered;
    }

    std::unordered_map<VoxelKey, AccumulatedPlyPoint, VoxelKeyHash> voxels;
    voxels.reserve(cloud.points.size() / 8 + 1);
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        const auto& point = cloud.points[i];
        const double ref_x =
                (static_cast<double>(point.x) - origin_x) / voxel_size;
        const double ref_y =
                (static_cast<double>(point.y) - origin_y) / voxel_size;
        const double ref_z =
                (static_cast<double>(point.z) - origin_z) / voxel_size;
        VoxelKey key{static_cast<int>(std::floor(ref_x)),
                     static_cast<int>(std::floor(ref_y)),
                     static_cast<int>(std::floor(ref_z))};
        auto& acc = voxels[key];
        acc.Add(point);
        if (i < cloud.visibility.size()) {
            acc.MergeVisibility(cloud.visibility[i]);
        }
    }

    filtered.points.clear();
    filtered.visibility.clear();
    filtered.points.reserve(voxels.size());
    filtered.visibility.reserve(voxels.size());
    for (const auto& [_, acc] : voxels) {
        filtered.points.push_back(acc.Average());
        filtered.visibility.push_back(acc.visibility);
    }
    return filtered;
}

std::vector<PlyPoint> VoxelDownsampleImpl(const std::vector<PlyPoint>& points,
                                          double voxel_size) {
    if (points.empty() || voxel_size <= 0.0) {
        return points;
    }

    double min_x = 0.0;
    double min_y = 0.0;
    double min_z = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
    double max_z = 0.0;
    ComputeBounds(points, min_x, min_y, min_z, max_x, max_y, max_z);

    const double half = 0.5 * voxel_size;
    const double origin_x = min_x - half;
    const double origin_y = min_y - half;
    const double origin_z = min_z - half;
    const double max_extent =
            std::max({max_x - min_x, max_y - min_y, max_z - min_z}) + voxel_size;
    if (voxel_size * static_cast<double>(std::numeric_limits<int>::max()) <
        max_extent) {
        LOG(WARNING) << "Fused point voxel size is too small; skipping voxel "
                        "downsample.";
        return points;
    }

    std::unordered_map<VoxelKey, AccumulatedPlyPoint, VoxelKeyHash> voxels;
    voxels.reserve(points.size() / 8 + 1);
    for (const auto& point : points) {
        const double ref_x = (static_cast<double>(point.x) - origin_x) / voxel_size;
        const double ref_y = (static_cast<double>(point.y) - origin_y) / voxel_size;
        const double ref_z = (static_cast<double>(point.z) - origin_z) / voxel_size;
        VoxelKey key{static_cast<int>(std::floor(ref_x)),
                     static_cast<int>(std::floor(ref_y)),
                     static_cast<int>(std::floor(ref_z))};
        voxels[key].Add(point);
    }

    std::vector<PlyPoint> filtered;
    filtered.reserve(voxels.size());
    for (const auto& [_, acc] : voxels) {
        filtered.push_back(acc.Average());
    }
    return filtered;
}

}  // namespace

std::vector<PlyPoint> VoxelDownsamplePlyPoints(
        const std::vector<PlyPoint>& points, double voxel_size) {
    return VoxelDownsampleImpl(points, voxel_size);
}

std::vector<PlyPoint> StatisticalOutlierRemovalPlyPoints(
        const std::vector<PlyPoint>& points,
        int nb_neighbors,
        double std_ratio) {
    return RunStatisticalOutlierRemoval(points, nb_neighbors, std_ratio, -1);
}

std::vector<PlyPoint> FilterFusedPlyPoints(
        const std::vector<PlyPoint>& points,
        const FusedPointFilterOptions& options) {
    if (!options.enabled || points.empty()) {
        return points;
    }

    std::vector<PlyPoint> filtered = points;
    if (ShouldVoxelDownsample(options)) {
        const size_t before = filtered.size();
        filtered = VoxelDownsamplePlyPoints(filtered, options.voxel_size);
        std::cout << "Fused point filter: voxel downsample (size="
                  << options.voxel_size << "): " << before << " -> "
                  << filtered.size() << std::endl;
    }

    if (options.sor_enabled && !filtered.empty()) {
        const size_t before = filtered.size();
        filtered = RunStatisticalOutlierRemoval(
                filtered, options.sor_nb_neighbors, options.sor_std_ratio,
                options.num_threads);
        std::cout << "Fused point filter: statistical outlier removal (k="
                  << options.sor_nb_neighbors
                  << ", std_ratio=" << options.sor_std_ratio << "): "
                  << before << " -> " << filtered.size() << std::endl;
    }

    return filtered;
}

FusedPointCloudWithVisibility FilterFusedPlyPointsWithVisibility(
        const FusedPointCloudWithVisibility& cloud,
        const FusedPointFilterOptions& options) {
    if (!options.enabled || cloud.points.empty()) {
        return cloud;
    }

    FusedPointCloudWithVisibility filtered = cloud;
    if (filtered.visibility.size() != filtered.points.size()) {
        filtered.visibility.assign(filtered.points.size(), std::vector<int>{});
    }

    if (ShouldVoxelDownsample(options)) {
        const size_t before = filtered.points.size();
        filtered = VoxelDownsamplePlyPointsWithVisibility(filtered,
                                                          options.voxel_size);
        std::cout << "Fused point filter: voxel downsample (size="
                  << options.voxel_size << "): " << before << " -> "
                  << filtered.points.size() << std::endl;
    }

    if (options.sor_enabled && !filtered.points.empty()) {
        const size_t before = filtered.points.size();
        const auto kept_indices = StatisticalOutlierRemovalIndices(
                filtered.points, options.sor_nb_neighbors, options.sor_std_ratio,
                options.num_threads);
        FusedPointCloudWithVisibility sor_filtered;
        sor_filtered.points.reserve(kept_indices.size());
        sor_filtered.visibility.reserve(kept_indices.size());
        for (const size_t idx : kept_indices) {
            sor_filtered.points.push_back(filtered.points[idx]);
            sor_filtered.visibility.push_back(filtered.visibility[idx]);
        }
        filtered = std::move(sor_filtered);
        std::cout << "Fused point filter: statistical outlier removal (k="
                  << options.sor_nb_neighbors
                  << ", std_ratio=" << options.sor_std_ratio << "): "
                  << before << " -> " << filtered.points.size() << std::endl;
    }

    return filtered;
}

}  // namespace colmap
